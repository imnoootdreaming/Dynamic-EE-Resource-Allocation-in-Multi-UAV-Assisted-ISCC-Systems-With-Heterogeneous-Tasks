import collections
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal


def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, continuous_action, discrete_index, reward, next_state, done):
        self.buffer.append((
            np.asarray(state, dtype=np.float32),
            np.asarray(continuous_action, dtype=np.float32),
            int(discrete_index),
            float(reward),
            np.asarray(next_state, dtype=np.float32),
            float(done),
        ))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        states, continuous_actions, discrete_indices, rewards, next_states, dones = zip(*transitions)
        return {
            "states": np.asarray(states, dtype=np.float32),
            "continuous_actions": np.asarray(continuous_actions, dtype=np.float32),
            "discrete_indices": np.asarray(discrete_indices, dtype=np.int64),
            "rewards": np.asarray(rewards, dtype=np.float32),
            "next_states": np.asarray(next_states, dtype=np.float32),
            "dones": np.asarray(dones, dtype=np.float32),
        }

    def size(self):
        return len(self.buffer)


class GaussianHead(nn.Module):
    def __init__(self, hidden_dim, action_dim, log_std_min=-20.0, log_std_max=2.0):
        super(GaussianHead, self).__init__()
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

        orthogonal_init(self.mean_layer, gain=0.01)
        orthogonal_init(self.log_std_layer, gain=0.01)

    def forward(self, features):
        mean = self.mean_layer(features)
        log_std = torch.clamp(self.log_std_layer(features), min=self.log_std_min, max=self.log_std_max)
        return mean, log_std


class HybridActor(nn.Module):
    def __init__(self, state_dim, hidden_dim, continuous_action_splits, discrete_action_dims):
        super(HybridActor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.activate_func = nn.ReLU()

        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)

        self.continuous_action_splits = dict(continuous_action_splits)
        self.continuous_head_names = list(self.continuous_action_splits.keys())
        self.continuous_heads = nn.ModuleDict({
            head_name: GaussianHead(hidden_dim, head_dim)
            for head_name, head_dim in self.continuous_action_splits.items()
        })

        self.discrete_action_dims = [int(dim) for dim in discrete_action_dims]
        self.discrete_heads = nn.ModuleList([
            nn.Linear(hidden_dim, action_dim) for action_dim in self.discrete_action_dims
        ])
        for head in self.discrete_heads:
            orthogonal_init(head, gain=0.01)

    def _forward_features(self, state):
        x = self.activate_func(self.fc1(state))
        x = self.activate_func(self.fc2(x))
        return x

    def _build_joint_distribution(self, probs_list, log_probs_list):
        joint_probs = probs_list[0]
        joint_log_probs = log_probs_list[0]
        batch_size = joint_probs.size(0)

        for probs, log_probs in zip(probs_list[1:], log_probs_list[1:]):
            joint_probs = (joint_probs.unsqueeze(-1) * probs.unsqueeze(1)).reshape(batch_size, -1)
            joint_log_probs = (joint_log_probs.unsqueeze(-1) + log_probs.unsqueeze(1)).reshape(batch_size, -1)

        return joint_probs, joint_log_probs

    def sample(self, state, deterministic=False):
        features = self._forward_features(state)
        batch_size = state.size(0)
        device = state.device

        continuous_actions = []
        continuous_log_prob = torch.zeros((batch_size, 1), dtype=torch.float32, device=device)
        for head_name in self.continuous_head_names:
            mean, log_std = self.continuous_heads[head_name](features)
            std = torch.exp(log_std)
            dist = Normal(mean, std)
            raw_action = mean if deterministic else dist.rsample()
            action = torch.sigmoid(raw_action)
            log_prob = dist.log_prob(raw_action) - torch.log(action * (1.0 - action) + 1e-8)
            continuous_actions.append(action)
            continuous_log_prob = continuous_log_prob + log_prob.sum(dim=-1, keepdim=True)

        if continuous_actions:
            continuous_actions = torch.cat(continuous_actions, dim=-1)
        else:
            continuous_actions = torch.empty((batch_size, 0), dtype=torch.float32, device=device)

        discrete_actions = []
        selected_discrete_log_prob = torch.zeros((batch_size, 1), dtype=torch.float32, device=device)
        probs_list = []
        log_probs_list = []
        discrete_entropy = torch.zeros((batch_size, 1), dtype=torch.float32, device=device)
        for head in self.discrete_heads:
            logits = head(features)
            probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)
            dist = Categorical(probs=probs)
            action = torch.argmax(probs, dim=-1) if deterministic else dist.sample()
            discrete_actions.append(action.unsqueeze(-1))
            selected_discrete_log_prob = selected_discrete_log_prob + log_probs.gather(1, action.unsqueeze(-1))
            probs_list.append(probs)
            log_probs_list.append(log_probs)
            discrete_entropy = discrete_entropy - (probs * log_probs).sum(dim=-1, keepdim=True)

        if discrete_actions:
            discrete_actions = torch.cat(discrete_actions, dim=-1)
            joint_probs, joint_log_probs = self._build_joint_distribution(probs_list, log_probs_list)
        else:
            discrete_actions = torch.empty((batch_size, 0), dtype=torch.long, device=device)
            joint_probs = torch.ones((batch_size, 1), dtype=torch.float32, device=device)
            joint_log_probs = torch.zeros((batch_size, 1), dtype=torch.float32, device=device)

        return {
            "continuous_actions": continuous_actions,
            "continuous_log_prob": continuous_log_prob,
            "discrete_actions": discrete_actions,
            "selected_discrete_log_prob": selected_discrete_log_prob,
            "joint_probs": joint_probs,
            "joint_log_probs": joint_log_probs,
            "discrete_entropy": discrete_entropy,
        }


class QNetwork(nn.Module):
    def __init__(self, state_dim, continuous_action_dim, hidden_dim, joint_action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + continuous_action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, joint_action_dim)
        self.activate_func = nn.ReLU()

        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)
        orthogonal_init(self.fc3, gain=0.01)

    def forward(self, state, continuous_action):
        x = torch.cat([state, continuous_action], dim=-1)
        x = self.activate_func(self.fc1(x))
        x = self.activate_func(self.fc2(x))
        return self.fc3(x)


class HSAC:
    def __init__(self, state_dim, hidden_dim, continuous_action_splits, discrete_action_dims,
                 action_low, action_high, actor_lr, critic_lr, gamma, tau,
                 alpha_continuous, alpha_discrete, device,
                 target_continuous_log_prob=0.3, target_discrete_entropy_ratio=0.48,
                 alpha_lr=None, alpha_min=1e-4, alpha_max=5.0):
        self.state_dim = int(state_dim)
        self.hidden_dim = int(hidden_dim)
        self.continuous_action_splits = dict(continuous_action_splits)
        self.continuous_action_dim = int(sum(self.continuous_action_splits.values()))
        self.discrete_action_dims = np.asarray(discrete_action_dims, dtype=np.int64)
        self.action_low = np.asarray(action_low, dtype=np.float32)
        self.action_high = np.asarray(action_high, dtype=np.float32)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.device = device
        # 20260409 - 更强的 HSAC 修改
        self.target_continuous_log_prob = float(target_continuous_log_prob)
        self.target_discrete_entropy = float(
            target_discrete_entropy_ratio * np.sum(np.log(np.maximum(self.discrete_action_dims, 1)))
        )
        # 20260409 - 更强的 HSAC 修改
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.alpha_lr = float(actor_lr if alpha_lr is None else alpha_lr)

        self.action_radices = np.cumprod(
            np.concatenate(([1], self.discrete_action_dims[::-1]))
        )[:-1][::-1].astype(np.int64)
        self.joint_action_dim = int(np.prod(self.discrete_action_dims))

        self.actor = HybridActor(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            continuous_action_splits=self.continuous_action_splits,
            discrete_action_dims=self.discrete_action_dims,
        ).to(device)

        self.critic1 = QNetwork(state_dim, self.continuous_action_dim, hidden_dim, self.joint_action_dim).to(device)
        self.critic2 = QNetwork(state_dim, self.continuous_action_dim, hidden_dim, self.joint_action_dim).to(device)
        self.critic1_target = QNetwork(state_dim, self.continuous_action_dim, hidden_dim, self.joint_action_dim).to(device)
        self.critic2_target = QNetwork(state_dim, self.continuous_action_dim, hidden_dim, self.joint_action_dim).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, eps=1e-5)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=critic_lr, eps=1e-5)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=critic_lr, eps=1e-5)
        # 20260409 - 更强的 HSAC 修改
        self.log_alpha_continuous = torch.tensor(
            np.log(max(float(alpha_continuous), 1e-8)),
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        self.log_alpha_discrete = torch.tensor(
            np.log(max(float(alpha_discrete), 1e-8)),
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        # 20260409 - 更强的 HSAC 修改
        self.alpha_continuous_optimizer = torch.optim.Adam([self.log_alpha_continuous], lr=self.alpha_lr, eps=1e-5)
        self.alpha_discrete_optimizer = torch.optim.Adam([self.log_alpha_discrete], lr=self.alpha_lr, eps=1e-5)

    def _current_alpha_continuous(self):
        # 20260409 - 更强的 HSAC 修改
        return torch.clamp(self.log_alpha_continuous.exp(), min=self.alpha_min, max=self.alpha_max)

    def _current_alpha_discrete(self):
        # 20260409 - 更强的 HSAC 修改
        return torch.clamp(self.log_alpha_discrete.exp(), min=self.alpha_min, max=self.alpha_max)

    def encode_discrete_action(self, discrete_action):
        discrete_action = np.asarray(discrete_action, dtype=np.int64)
        return int(np.sum(discrete_action * self.action_radices))

    def decode_discrete_index(self, action_index):
        remaining = int(action_index)
        decoded = np.zeros_like(self.discrete_action_dims)
        for i, radix in enumerate(self.action_radices):
            decoded[i] = remaining // radix
            remaining = remaining % radix
        return decoded.astype(np.int64)

    def choose_action(self, state, evaluate=False):
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            sampled = self.actor.sample(state_tensor, deterministic=evaluate)

        discrete_action = sampled["discrete_actions"].squeeze(0).cpu().numpy().astype(np.int64)
        continuous_action = sampled["continuous_actions"].squeeze(0).cpu().numpy().astype(np.float32)
        discrete_index = self.encode_discrete_action(discrete_action)

        return {
            "continuous": continuous_action,
            "discrete": discrete_action,
            "discrete_index": discrete_index,
            "continuous_log_prob": float(sampled["continuous_log_prob"].item()),
            "selected_discrete_log_prob": float(sampled["selected_discrete_log_prob"].item()),
            "discrete_entropy": float(sampled["discrete_entropy"].item()),
        }

    def update(self, transition_dict, writer=None, step=None, agent_name="BS"):
        states = torch.tensor(transition_dict["states"], dtype=torch.float32, device=self.device)
        continuous_actions = torch.tensor(
            transition_dict["continuous_actions"], dtype=torch.float32, device=self.device
        )
        discrete_indices = torch.tensor(
            transition_dict["discrete_indices"], dtype=torch.long, device=self.device
        ).view(-1, 1)
        rewards = torch.tensor(
            transition_dict["rewards"], dtype=torch.float32, device=self.device
        ).view(-1, 1)
        next_states = torch.tensor(
            transition_dict["next_states"], dtype=torch.float32, device=self.device
        )
        dones = torch.tensor(
            transition_dict["dones"], dtype=torch.float32, device=self.device
        ).view(-1, 1)

        with torch.no_grad():
            next_sampled = self.actor.sample(next_states, deterministic=False)
            next_continuous_actions = next_sampled["continuous_actions"]
            next_continuous_log_prob = next_sampled["continuous_log_prob"]
            next_joint_probs = next_sampled["joint_probs"]
            next_joint_log_probs = next_sampled["joint_log_probs"]
            # 20260409 - 更强的 HSAC 修改
            alpha_continuous = self._current_alpha_continuous()
            alpha_discrete = self._current_alpha_discrete()

            next_q1_values = self.critic1_target(next_states, next_continuous_actions)
            next_q2_values = self.critic2_target(next_states, next_continuous_actions)
            next_min_q_values = torch.min(next_q1_values, next_q2_values)
            next_expected_q = (next_joint_probs * next_min_q_values).sum(dim=-1, keepdim=True)
            next_expected_discrete_log_prob = (next_joint_probs * next_joint_log_probs).sum(dim=-1, keepdim=True)

            target_v = (
                next_expected_q
                - alpha_continuous * next_continuous_log_prob
                - alpha_discrete * next_expected_discrete_log_prob
            )
            q_target = rewards + self.gamma * (1.0 - dones) * target_v

        q1_values = self.critic1(states, continuous_actions)
        q2_values = self.critic2(states, continuous_actions)
        q1_pred = q1_values.gather(1, discrete_indices)
        q2_pred = q2_values.gather(1, discrete_indices)
        critic1_loss = F.mse_loss(q1_pred, q_target)
        critic2_loss = F.mse_loss(q2_pred, q_target)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), 1.0)
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), 1.0)
        self.critic2_optimizer.step()

        sampled = self.actor.sample(states, deterministic=False)
        actor_continuous_actions = sampled["continuous_actions"]
        actor_continuous_log_prob = sampled["continuous_log_prob"]
        actor_joint_probs = sampled["joint_probs"]
        actor_joint_log_probs = sampled["joint_log_probs"]
        actor_discrete_entropy = sampled["discrete_entropy"]
        # 20260409 - 更强的 HSAC 修改
        alpha_continuous = self._current_alpha_continuous().detach()
        alpha_discrete = self._current_alpha_discrete().detach()

        actor_q1_values = self.critic1(states, actor_continuous_actions)
        actor_q2_values = self.critic2(states, actor_continuous_actions)
        actor_min_q_values = torch.min(actor_q1_values, actor_q2_values)
        actor_expected_q = (actor_joint_probs * actor_min_q_values).sum(dim=-1, keepdim=True)
        actor_expected_discrete_log_prob = (actor_joint_probs * actor_joint_log_probs).sum(dim=-1, keepdim=True)
        actor_loss = (
            alpha_continuous * actor_continuous_log_prob
            + alpha_discrete * actor_expected_discrete_log_prob
            - actor_expected_q
        ).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        # 20260409 - 更强的 HSAC 修改
        alpha_continuous_loss = (
            self.log_alpha_continuous * (self.target_continuous_log_prob - actor_continuous_log_prob).detach()
        ).mean()
        alpha_discrete_loss = (
            self.log_alpha_discrete * (actor_discrete_entropy - self.target_discrete_entropy).detach()
        ).mean()

        self.alpha_continuous_optimizer.zero_grad()
        alpha_continuous_loss.backward()
        self.alpha_continuous_optimizer.step()
        # 20260409 - 更强的 HSAC 修改
        with torch.no_grad():
            self.log_alpha_continuous.clamp_(np.log(self.alpha_min), np.log(self.alpha_max))

        self.alpha_discrete_optimizer.zero_grad()
        alpha_discrete_loss.backward()
        self.alpha_discrete_optimizer.step()
        # 20260409 - 更强的 HSAC 修改
        with torch.no_grad():
            self.log_alpha_discrete.clamp_(np.log(self.alpha_min), np.log(self.alpha_max))

        self._soft_update(self.critic1, self.critic1_target)
        self._soft_update(self.critic2, self.critic2_target)

        if writer is not None and step is not None:
            writer.add_scalar(f"{agent_name}/Actor_Loss", actor_loss.item(), step)
            writer.add_scalar(f"{agent_name}/Critic1_Loss", critic1_loss.item(), step)
            writer.add_scalar(f"{agent_name}/Critic2_Loss", critic2_loss.item(), step)
            writer.add_scalar(f"{agent_name}/Q_Target", q_target.mean().item(), step)
            writer.add_scalar(f"{agent_name}/Q1_Pred", q1_pred.mean().item(), step)
            writer.add_scalar(f"{agent_name}/Q2_Pred", q2_pred.mean().item(), step)
            writer.add_scalar(
                f"{agent_name}/Continuous_LogProb",
                actor_continuous_log_prob.mean().item(),
                step,
            )
            writer.add_scalar(
                f"{agent_name}/Expected_Discrete_LogProb",
                actor_expected_discrete_log_prob.mean().item(),
                step,
            )
            writer.add_scalar(
                f"{agent_name}/Discrete_Entropy",
                actor_discrete_entropy.mean().item(),
                step,
            )
            # 20260409 - 更强的 HSAC 修改
            writer.add_scalar(f"{agent_name}/Alpha_Continuous", self._current_alpha_continuous().item(), step)
            writer.add_scalar(f"{agent_name}/Alpha_Discrete", self._current_alpha_discrete().item(), step)
            writer.add_scalar(f"{agent_name}/Alpha_Continuous_Loss", alpha_continuous_loss.item(), step)
            writer.add_scalar(f"{agent_name}/Alpha_Discrete_Loss", alpha_discrete_loss.item(), step)
            writer.add_scalar(f"{agent_name}/Target_Continuous_LogProb", self.target_continuous_log_prob, step)
            writer.add_scalar(f"{agent_name}/Target_Discrete_Entropy", self.target_discrete_entropy, step)

    def _soft_update(self, net, target_net):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
