import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta, Categorical, Normal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


class BetaHead(nn.Module):
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


class GaussianHead(nn.Module):
    # 20260403 - 引入高斯分布对应的修改
    def __init__(self, hidden_dim, action_dim):
        super(GaussianHead, self).__init__()
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)
        orthogonal_init(self.mean_layer, gain=0.01)
        orthogonal_init(self.log_std_layer, gain=0.01)

    def get_dist(self, features):
        mean = self.mean_layer(features)
        log_std = torch.clamp(self.log_std_layer(features), min=-20.0, max=2.0)
        std = torch.exp(log_std)
        return Normal(mean, std)


class MultiHeadActor(nn.Module):
    def __init__(self, state_dim, hidden_dim, continuous_action_splits, discrete_action_dims, continuous_dist_type="beta"):
        super(MultiHeadActor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.activate_func = nn.Tanh()

        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)

        self.continuous_action_splits = continuous_action_splits
        self.continuous_head_names = list(continuous_action_splits.keys())
        # 20260403 - 引入高斯分布对应的修改
        self.continuous_dist_type = str(continuous_dist_type).lower()
        if self.continuous_dist_type == "beta":
            self.continuous_heads = nn.ModuleDict({
                head_name: BetaHead(hidden_dim, head_dim)
                for head_name, head_dim in continuous_action_splits.items()
            })
        elif self.continuous_dist_type == "gaussian":
            self.continuous_heads = nn.ModuleDict({
                head_name: GaussianHead(hidden_dim, head_dim)
                for head_name, head_dim in continuous_action_splits.items()
            })
        else:
            raise ValueError(f"Unsupported continuous_dist_type: {continuous_dist_type}")

        self.discrete_action_dims = [int(dim) for dim in discrete_action_dims]
        # NOTE - Each discrete head corresponds to one UAV choosing one CU index.
        # This keeps the policy factorized instead of modeling the full Cartesian
        # product of all UAV-CU assignments as a single giant categorical action.
        self.discrete_heads = nn.ModuleList([
            nn.Linear(hidden_dim, action_dim) for action_dim in self.discrete_action_dims
        ])
        for head in self.discrete_heads:
            orthogonal_init(head, gain=0.01)

    def _forward_features(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        return s

    def _get_dists(self, s):
        features = self._forward_features(s)
        continuous_dists = {
            head_name: self.continuous_heads[head_name].get_dist(features)
            for head_name in self.continuous_head_names
        }
        discrete_dists = [
            Categorical(logits=head(features))
            for head in self.discrete_heads
        ]
        return continuous_dists, discrete_dists

    # 20260403 - 引入高斯分布对应的修改
    def _sample_continuous_action_and_log_prob(self, dist):
        if self.continuous_dist_type == "beta":
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
            entropy = dist.entropy().sum(dim=-1, keepdim=True)
            return action, log_prob, entropy

        if self.continuous_dist_type == "gaussian":
            raw_action = dist.rsample()
            action = torch.sigmoid(raw_action)
            log_prob = dist.log_prob(raw_action) - torch.log(action * (1.0 - action) + 1e-8)
            log_prob = log_prob.sum(dim=-1, keepdim=True)
            entropy = dist.entropy().sum(dim=-1, keepdim=True)
            return action, log_prob, entropy

        raise ValueError(f"Unsupported continuous_dist_type: {self.continuous_dist_type}")

    # 20260403 - 引入高斯分布对应的修改
    def _evaluate_continuous_log_prob_and_entropy(self, dist, action_slice):
        if self.continuous_dist_type == "beta":
            log_prob = dist.log_prob(action_slice).sum(dim=-1, keepdim=True)
            entropy = dist.entropy().sum(dim=-1, keepdim=True)
            return log_prob, entropy

        if self.continuous_dist_type == "gaussian":
            action_slice = torch.clamp(action_slice, 1e-6, 1.0 - 1e-6)
            raw_action = torch.logit(action_slice, eps=1e-6)
            log_prob = dist.log_prob(raw_action) - torch.log(action_slice * (1.0 - action_slice) + 1e-8)
            log_prob = log_prob.sum(dim=-1, keepdim=True)
            entropy = dist.entropy().sum(dim=-1, keepdim=True)
            return log_prob, entropy

        raise ValueError(f"Unsupported continuous_dist_type: {self.continuous_dist_type}")

    def sample(self, s):
        continuous_dists, discrete_dists = self._get_dists(s)
        batch_size = s.size(0)
        device = s.device

        continuous_actions = []
        total_log_prob = torch.zeros((batch_size, 1), dtype=torch.float32, device=device)
        con_log_prob = torch.zeros((batch_size, 1), dtype=torch.float32, device=device)
        for head_name in self.continuous_head_names:
            dist = continuous_dists[head_name]
            action, action_log_prob, _ = self._sample_continuous_action_and_log_prob(dist)
            continuous_actions.append(action)
            total_log_prob = total_log_prob + action_log_prob
            con_log_prob = con_log_prob + action_log_prob

        if continuous_actions:
            continuous_actions = torch.cat(continuous_actions, dim=-1)
        else:
            continuous_actions = torch.empty((batch_size, 0), dtype=torch.float32, device=device)

        discrete_actions = []
        dis_log_prob = torch.zeros((batch_size, 1), dtype=torch.float32, device=device)
        for dist in discrete_dists:
            action = dist.sample()
            discrete_actions.append(action.unsqueeze(-1))
            total_log_prob = total_log_prob + dist.log_prob(action).unsqueeze(-1)
            dis_log_prob = dis_log_prob + dist.log_prob(action).unsqueeze(-1)
        
        if discrete_actions:
            discrete_actions = torch.cat(discrete_actions, dim=-1)
        else:
            discrete_actions = torch.empty((batch_size, 0), dtype=torch.long, device=device)

        return continuous_actions, discrete_actions, total_log_prob, con_log_prob, dis_log_prob

    def evaluate_actions(self, s, continuous_actions, discrete_actions):
        continuous_dists, discrete_dists = self._get_dists(s)
        batch_size = s.size(0)
        device = s.device

        # NOTE - 统计连续动作和离散动作的 log_prob 和 entropy
        cont_log_prob = torch.zeros((batch_size, 1), device=device)
        disc_log_prob = torch.zeros((batch_size, 1), device=device)
        cont_entropy = torch.zeros((batch_size, 1), device=device)
        disc_entropy = torch.zeros((batch_size, 1), device=device)

        start_idx = 0
        for head_name in self.continuous_head_names:
            head_dim = self.continuous_action_splits[head_name]
            action_slice = continuous_actions[:, start_idx:start_idx + head_dim]
            dist = continuous_dists[head_name]
            head_log_prob, head_entropy = self._evaluate_continuous_log_prob_and_entropy(dist, action_slice)
            cont_log_prob += head_log_prob
            cont_entropy += head_entropy
            start_idx += head_dim

        for head_idx, dist in enumerate(discrete_dists):
            action_slice = discrete_actions[:, head_idx]
            disc_log_prob += dist.log_prob(action_slice).unsqueeze(-1)
            disc_entropy += dist.entropy().unsqueeze(-1)

        # 返回总和以及分量
        return (cont_log_prob + disc_log_prob), (cont_entropy + disc_entropy), \
            cont_log_prob, disc_log_prob, cont_entropy, disc_entropy


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


class HPPO:
    def __init__(self, state_dim, hidden_dim, continuous_action_splits, discrete_action_dims,
                 action_low, action_high, actor_lr, critic_lr, lmbda, eps, gamma, epochs,
                 num_episodes, device, entropy_coef=0.01, discrete_entropy_coef=0.01,
                 continuous_entropy_coef=0.05, continuous_dist_type="beta"):
        self.continuous_action_splits = continuous_action_splits
        self.discrete_action_dims = np.asarray(discrete_action_dims, dtype=np.int64)
        self.actor = MultiHeadActor(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            continuous_action_splits=continuous_action_splits,
            discrete_action_dims=self.discrete_action_dims,
            continuous_dist_type=continuous_dist_type
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
        # NOTE - Actor Loss 中的离散和连续动作的权重
        self.discrete_entropy_coef = discrete_entropy_coef
        self.continuous_entropy_coef = continuous_entropy_coef

    def choose_action(self, s):
        s = torch.tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            continuous_actions, discrete_actions, log_prob, con_log_prob, dis_log_prob = self.actor.sample(s)

        action = {
            "continuous": continuous_actions.squeeze(0).cpu().numpy().astype(np.float32),
            "discrete": discrete_actions.squeeze(0).cpu().numpy().astype(np.int64),
        }
        return action, log_prob.squeeze(0).cpu().numpy(), con_log_prob.squeeze(0).cpu().numpy(), dis_log_prob.squeeze(0).cpu().numpy()

    def update(self, transition_dict, step=None, writer=None, agent_name="Agent"):
        states = torch.tensor(transition_dict["states"], dtype=torch.float32, device=self.device)
        continuous_actions = torch.tensor(
            transition_dict["continuous_actions"], dtype=torch.float32, device=self.device
        )
        discrete_actions = torch.tensor(
            transition_dict["discrete_actions"], dtype=torch.long, device=self.device
        )
        rewards = torch.tensor(
            transition_dict["rewards"], dtype=torch.float32, device=self.device
        ).view(-1, 1)
        next_states = torch.tensor(
            transition_dict["next_states"], dtype=torch.float32, device=self.device
        )
        # NOTE - 连续动作和离散动作各自的 log_probs
        old_cont_lps = torch.tensor(
            transition_dict["old_cont_log_probs"], dtype=torch.float32, device=self.device
        ).view(-1, 1)
        old_disc_lps = torch.tensor(
            transition_dict["old_disc_log_probs"], dtype=torch.float32, device=self.device
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
        # 连续动作和离散动作的 log_prob
        c_lps, d_lps = [], []
        # 连续动作和离散动作的 entropy
        c_ents, d_ents = [], [] 

        batch_size = states.size(0)
        mini_batch_size = min(32, batch_size)
        
        # 辅助函数
        def _surr(ratio, a):
            return -torch.min(ratio * a,
                            torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * a)
    
        for _ in range(self.epochs):
            for index in BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, False):
                log_probs, total_entropy, c_lp, d_lp, c_ent, d_ent = self.actor.evaluate_actions(
                    states[index],
                    continuous_actions[index],
                    discrete_actions[index]
                )
                # 为连续动作和离散动作各自独立 ratio + clip
                cont_ratio = torch.exp(c_lp - old_cont_lps[index])
                disc_ratio = torch.exp(d_lp - old_disc_lps[index])
                actor_loss = torch.mean(
                    _surr(cont_ratio, adv[index])
                    + _surr(disc_ratio, adv[index])
                    - self.continuous_entropy_coef * c_ent
                    - self.discrete_entropy_coef   * d_ent
                )
                # # ratio = torch.exp(log_probs - old_log_probs[index])
                # surr1 = ratio * adv[index]
                # surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * adv[index]
                # actor_loss = torch.mean(
                #     -torch.min(surr1, surr2) - self.discrete_entropy_coef * d_ent - self.continuous_entropy_coef * c_ent)
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
                c_lps.append(c_lp.mean().item())
                d_lps.append(d_lp.mean().item())
                c_ents.append(c_ent.mean().item())
                d_ents.append(d_ent.mean().item())

        if step is not None:
            self.lr_decay(step)

        if writer is not None and step is not None:
            writer.add_scalar(f"{agent_name}/Actor_Loss", np.mean(actor_losses), step)
            writer.add_scalar(f"{agent_name}/Critic_Loss", np.mean(critic_losses), step)
            writer.add_scalar(f"{agent_name}/Continuous_LogProb", np.mean(c_lps), step)
            writer.add_scalar(f"{agent_name}/Discrete_LogProb", np.mean(d_lps), step)
            writer.add_scalar(f"{agent_name}/continuous_entropy", np.mean(c_ents), step)
            writer.add_scalar(f"{agent_name}/discrete_entropy", np.mean(d_ents), step)
            old_lps = old_cont_lps + old_disc_lps          # 仅用于监控，不参与训练
            res = self.actor.evaluate_actions(states, continuous_actions, discrete_actions)
            kl_c_lp, kl_d_lp = res[2], res[3]             # 取分开的 log prob
            kl = (old_lps - (kl_c_lp + kl_d_lp)).mean().item()
            writer.add_scalar(f"{agent_name}/KL_Divergence", kl, step)


    def lr_decay(self, total_steps):
        lr_a_now = self.actor_optimizer.defaults["lr"] * (1 - total_steps / self.num_episodes)
        lr_c_now = self.critic_optimizer.defaults["lr"] * (1 - total_steps / self.num_episodes)

        for param_group in self.actor_optimizer.param_groups:
            param_group["lr"] = lr_a_now
        for param_group in self.critic_optimizer.param_groups:
            param_group["lr"] = lr_c_now