import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from algorithms.training_schedule import ppo_lr_multiplier
from algorithms.SHBPPO.matching import beta_bin_probabilities, expected_duplicate_matching


def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


class BetaHead(nn.Module):
    """单个 Beta 分布头，输出 action_dim 维独立 Beta 分布（所有维度落在 [0, 1]）。"""

    def __init__(self, hidden_dim, action_dim):
        super(BetaHead, self).__init__()
        self.alpha_layer = nn.Linear(hidden_dim, action_dim)
        self.beta_layer = nn.Linear(hidden_dim, action_dim)
        orthogonal_init(self.alpha_layer, gain=0.01)
        orthogonal_init(self.beta_layer, gain=0.01)

    def get_dist(self, features):
        alpha = F.softplus(self.alpha_layer(features)) + 1.0
        beta = F.softplus(self.beta_layer(features)) + 1.0
        return Beta(alpha, beta)


class SingleHeadActor(nn.Module):
    """单头 Beta Actor：只有一个 Beta 分布头，覆盖所有动作维度（连续 + 离散统一编码为 [0,1]）。"""

    def __init__(self, state_dim, hidden_dim, action_dim, uavs_num=0, cus_num=0):
        super(SingleHeadActor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.activate_func = nn.Tanh()
        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)

        self.action_dim = action_dim
        self.uavs_num = int(uavs_num)
        self.cus_num = int(cus_num)
        self.beta_head = BetaHead(hidden_dim, action_dim)

    def _forward_features(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        return s

    def sample(self, s):
        features = self._forward_features(s)
        dist = self.beta_head.get_dist(features)
        action = dist.sample()                       # [0, 1]^action_dim
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        return action, log_prob, entropy

    def evaluate_actions(self, s, actions, *, return_expected_duplicate_matching=False):
        features = self._forward_features(s)
        dist = self.beta_head.get_dist(features)
        log_prob = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        if not return_expected_duplicate_matching:
            return log_prob, entropy
        if not 0 < self.uavs_num <= self.action_dim or self.cus_num <= 0:
            raise ValueError("CU-bin penalty requires valid uavs_num and cus_num")
        probabilities = beta_bin_probabilities(
            dist.concentration1[:, -self.uavs_num:],
            dist.concentration0[:, -self.uavs_num:], self.cus_num)
        return log_prob, entropy, expected_duplicate_matching(probabilities)


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.activate_func = nn.Tanh()
        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)
        orthogonal_init(self.fc3)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        return self.fc3(s)


class SHBPPO:
    """单头 Beta PPO（Single-Head Beta PPO）。

    Actor 只输出一个 Beta 分布向量（维度 =
    连续动作维度 + uavs_num），不再区分连续/离散头。离散动作（每个 UAV 选
    哪个 CU）在 [0,1] 采样空间中作为连续值输出，在交给环境前由外部映射函数
    离散化为 CU 索引。PPO 内部的 log_prob / entropy / ratio 全部统一处理。
    """

    def __init__(self, state_dim, hidden_dim, action_dim, action_low, action_high,
                 actor_lr, critic_lr, lmbda, eps, gamma, epochs, num_episodes,
                 device, entropy_coef=0.01, lr_min_ratio=0.1,
                 uavs_num=0, cus_num=0, expected_duplicate_matching_coef=0.0):
        self.action_dim = action_dim
        self.expected_duplicate_matching_coef = float(expected_duplicate_matching_coef)
        if not np.isfinite(self.expected_duplicate_matching_coef) or self.expected_duplicate_matching_coef < 0:
            raise ValueError("expected_duplicate_matching_coef must be nonnegative and finite")
        if self.expected_duplicate_matching_coef > 0 and not (0 < uavs_num <= action_dim and cus_num > 0):
            raise ValueError("CU-bin penalty requires valid uavs_num and cus_num")
        self.actor = SingleHeadActor(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim, uavs_num=uavs_num, cus_num=cus_num
        ).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, eps=1e-5)

        self.critic = Critic(state_dim, hidden_dim).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr, eps=1e-5)

        self.gamma = gamma
        self.lmbda = lmbda
        self.eps = eps
        self.epochs = epochs
        self.num_episodes = num_episodes
        self.lr_min_ratio = float(lr_min_ratio)
        self.device = device
        self.entropy_coef = entropy_coef
        self.action_low = np.asarray(action_low, dtype=np.float32)
        self.action_high = np.asarray(action_high, dtype=np.float32)

    def choose_action(self, s):
        s = torch.tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, log_prob, _ = self.actor.sample(s)
        raw_action = action.squeeze(0).cpu().numpy().astype(np.float32)
        return raw_action, log_prob.squeeze(0).cpu().numpy()

    def update(self, transition_dict, step=None, writer=None, agent_name="Agent"):
        states = torch.tensor(transition_dict["states"], dtype=torch.float32, device=self.device)
        actions = torch.tensor(
            transition_dict["actions"], dtype=torch.float32, device=self.device
        )
        rewards = torch.tensor(
            transition_dict["rewards"], dtype=torch.float32, device=self.device
        ).view(-1, 1)
        next_states = torch.tensor(
            transition_dict["next_states"], dtype=torch.float32, device=self.device
        )
        old_log_probs = torch.tensor(
            transition_dict["old_log_probs"], dtype=torch.float32, device=self.device
        ).view(-1, 1)
        dones = torch.tensor(
            transition_dict["dones"], dtype=torch.float32, device=self.device
        ).view(-1, 1)
        real_dones = torch.tensor(
            transition_dict["real_dones"], dtype=torch.float32, device=self.device
        ).view(-1, 1)

        adv = []
        gae = 0.0
        with torch.no_grad():
            vs = self.critic(states)
            vs_ = self.critic(next_states)
            td_target = rewards + self.gamma * vs_ * (1 - real_dones)
            td_delta = td_target - vs

            td_delta = td_delta.cpu().numpy()
            dones_np = dones.cpu().numpy()

            for delta, done in zip(reversed(td_delta), reversed(dones_np)):
                gae = delta + self.gamma * self.lmbda * gae * (1.0 - done)
                adv.insert(0, gae)

            adv = torch.tensor(adv, dtype=torch.float32, device=self.device).view(-1, 1)
            v_target = adv + self.critic(states)
            adv = (adv - adv.mean()) / (adv.std() + 1e-5)

        actor_losses = []
        critic_losses = []
        log_prob_means = []
        entropy_means = []

        batch_size = states.size(0)
        mini_batch_size = min(32, batch_size)
        use_duplicate_penalty = self.expected_duplicate_matching_coef > 0

        def _surr(ratio, a):
            return -torch.min(ratio * a,
                              torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * a)

        for _ in range(self.epochs):
            for index in BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, False):
                evaluation = self.actor.evaluate_actions(
                    states[index], actions[index],
                    return_expected_duplicate_matching=use_duplicate_penalty,
                )
                log_probs, entropy = evaluation[:2]
                ratio = torch.exp(log_probs - old_log_probs[index])
                actor_loss = torch.mean(
                    _surr(ratio, adv[index]) - self.entropy_coef * entropy
                )
                if use_duplicate_penalty:
                    actor_loss = actor_loss + self.expected_duplicate_matching_coef * evaluation[2].mean()
                critic_loss = F.mse_loss(self.critic(states[index]), v_target[index].detach())

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1)
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
                self.actor_optimizer.step()
                self.critic_optimizer.step()

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                log_prob_means.append(log_probs.mean().item())
                entropy_means.append(entropy.mean().item())

        if step is not None:
            self.lr_decay(step)

        expected_duplicates = float("nan")
        if use_duplicate_penalty:
            with torch.no_grad():
                expected_duplicates = self.actor.evaluate_actions(
                    states, actions, return_expected_duplicate_matching=True)[2].mean().item()

        if writer is not None and step is not None:
            writer.add_scalar(f"{agent_name}/Actor_Loss", np.mean(actor_losses), step)
            writer.add_scalar(f"{agent_name}/Critic_Loss", np.mean(critic_losses), step)
            writer.add_scalar(f"{agent_name}/LogProb", np.mean(log_prob_means), step)
            writer.add_scalar(f"{agent_name}/Entropy", np.mean(entropy_means), step)
            res = self.actor.evaluate_actions(states, actions)
            kl = (old_log_probs - res[0]).mean().item()
            writer.add_scalar(f"{agent_name}/KL_Divergence", kl, step)
            if use_duplicate_penalty:
                writer.add_scalar(f"{agent_name}/actor_expected_duplicate_matching", expected_duplicates, step)
        return expected_duplicates

    def lr_decay(self, total_steps):
        decay = ppo_lr_multiplier(total_steps, self.num_episodes, self.lr_min_ratio)
        lr_a_now = self.actor_optimizer.defaults["lr"] * decay
        lr_c_now = self.critic_optimizer.defaults["lr"] * decay

        for param_group in self.actor_optimizer.param_groups:
            param_group["lr"] = lr_a_now
        for param_group in self.critic_optimizer.param_groups:
            param_group["lr"] = lr_c_now
