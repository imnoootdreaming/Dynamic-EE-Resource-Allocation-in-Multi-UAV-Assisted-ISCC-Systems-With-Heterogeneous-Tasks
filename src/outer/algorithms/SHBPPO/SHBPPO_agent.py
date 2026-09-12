import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


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

    def __init__(self, state_dim, hidden_dim, action_dim):
        super(SingleHeadActor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.activate_func = nn.Tanh()
        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)

        self.action_dim = action_dim
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

    def evaluate_actions(self, s, actions):
        features = self._forward_features(s)
        dist = self.beta_head.get_dist(features)
        log_prob = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        return log_prob, entropy


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

    与多头 HPPO 的唯一区别：Actor 只输出一个 Beta 分布向量（维度 =
    连续动作维度 + uavs_num），不再区分连续/离散头。离散动作（每个 UAV 选
    哪个 CU）在 [0,1] 采样空间中作为连续值输出，在交给环境前由外部映射函数
    离散化为 CU 索引。PPO 内部的 log_prob / entropy / ratio 全部统一处理。
    """

    def __init__(self, state_dim, hidden_dim, action_dim, action_low, action_high,
                 actor_lr, critic_lr, lmbda, eps, gamma, epochs, num_episodes,
                 device, entropy_coef=0.01):
        self.action_dim = action_dim
        self.actor = SingleHeadActor(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim
        ).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, eps=1e-5)

        self.critic = Critic(state_dim, hidden_dim).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr, eps=1e-5)

        self.gamma = gamma
        self.lmbda = lmbda
        self.eps = eps
        self.epochs = epochs
        self.num_episodes = num_episodes
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

        def _surr(ratio, a):
            return -torch.min(ratio * a,
                              torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * a)

        for _ in range(self.epochs):
            for index in BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, False):
                log_probs, entropy = self.actor.evaluate_actions(
                    states[index], actions[index]
                )
                ratio = torch.exp(log_probs - old_log_probs[index])
                actor_loss = torch.mean(
                    _surr(ratio, adv[index]) - self.entropy_coef * entropy
                )
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

        if writer is not None and step is not None:
            writer.add_scalar(f"{agent_name}/Actor_Loss", np.mean(actor_losses), step)
            writer.add_scalar(f"{agent_name}/Critic_Loss", np.mean(critic_losses), step)
            writer.add_scalar(f"{agent_name}/LogProb", np.mean(log_prob_means), step)
            writer.add_scalar(f"{agent_name}/Entropy", np.mean(entropy_means), step)
            res = self.actor.evaluate_actions(states, actions)
            kl = (old_log_probs - res[0]).mean().item()
            writer.add_scalar(f"{agent_name}/KL_Divergence", kl, step)

    def lr_decay(self, total_steps):
        lr_a_now = self.actor_optimizer.defaults["lr"] * (1 - total_steps / self.num_episodes)
        lr_c_now = self.critic_optimizer.defaults["lr"] * (1 - total_steps / self.num_episodes)

        for param_group in self.actor_optimizer.param_groups:
            param_group["lr"] = lr_a_now
        for param_group in self.critic_optimizer.param_groups:
            param_group["lr"] = lr_c_now
