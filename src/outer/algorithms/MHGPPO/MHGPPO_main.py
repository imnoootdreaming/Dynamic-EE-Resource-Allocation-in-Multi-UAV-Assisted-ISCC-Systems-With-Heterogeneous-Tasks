import argparse
from contextlib import contextmanager
import importlib.util
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import sys

# 自动向上定位 outer 根目录（含 algorithms/ 的那一层）
_OUTER_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_OUTER_ROOT, "algorithms")):
    _parent_root = os.path.dirname(_OUTER_ROOT)
    if _parent_root == _OUTER_ROOT:
        break
    _OUTER_ROOT = _parent_root
if _OUTER_ROOT not in sys.path:
    sys.path.insert(0, _OUTER_ROOT)

from algorithms.MHBPPO.MHBPPO_agent import HPPO
from environment.my_env import MyEnv
from utils.normalization import Normalization, RewardScaling


def _load_beta_main_helpers():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    beta_main_path = os.path.join(_OUTER_ROOT, "algorithms", "MHBPPO", "MHBPPO_main.py")
    spec = importlib.util.spec_from_file_location("beta_hppo_main_helpers", beta_main_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_beta_main_helpers = _load_beta_main_helpers()
setSeed = _beta_main_helpers.setSeed
get_tensorboard_writer = _beta_main_helpers.get_tensorboard_writer
get_madrl_args = _beta_main_helpers.get_madrl_args
get_base_args = _beta_main_helpers.get_base_args


# 20260403 - 引入高斯分布对应的修改
@contextmanager
def _ignore_unknown_cli_args():
    original_parse_args = argparse.ArgumentParser.parse_args

    def _parse_args_ignore_unknown(self, args=None, namespace=None):
        parsed_args, _ = self.parse_known_args(args=args, namespace=namespace)
        return parsed_args

    argparse.ArgumentParser.parse_args = _parse_args_ignore_unknown
    try:
        yield
    finally:
        argparse.ArgumentParser.parse_args = original_parse_args


# 20260403 - 引入高斯分布对应的修改
def get_all_args():
    with _ignore_unknown_cli_args():
        base_args = get_base_args()
        madrl_args = get_madrl_args()
    return base_args, madrl_args


if __name__ == "__main__":
    current_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    writer = get_tensorboard_writer(log_dir=f"runs/gaussian-hppo/{current_time_str}_gaussian-hppo_result")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    base_args, madrl_args = get_all_args()  # 20260403 - 引入高斯分布对应的修改
    setSeed(seed=base_args.seed)

    env = MyEnv(base_args=base_args, madrl_args=madrl_args)

    state_dim_bs = env.observation_space["bs"].shape[0]
    bs_continuous_action_space = env.action_space["bs"]["continuous"]
    bs_discrete_action_space = env.action_space["bs"]["discrete"]

    agent_bs = HPPO(
        state_dim=state_dim_bs,
        hidden_dim=madrl_args.hidden_dim,
        continuous_action_splits=env.bs_continuous_action_splits,
        discrete_action_dims=bs_discrete_action_space.nvec,
        action_low=bs_continuous_action_space.low,
        action_high=bs_continuous_action_space.high,
        actor_lr=madrl_args.actor_lr,
        critic_lr=madrl_args.critic_lr,
        lmbda=madrl_args.lmbda,
        eps=madrl_args.eps,
        gamma=madrl_args.gamma,
        epochs=madrl_args.epochs,
        num_episodes=madrl_args.episodes,
        device=device,
        entropy_coef=0.01,
        continuous_dist_type="gaussian",  # 20260403 - 引入高斯分布对应的修改
    )

    running_norm_bs = Normalization(state_dim_bs)
    reward_scaler_bs = RewardScaling(shape=1, gamma=madrl_args.gamma)

    reward_res = []
    all_agents_rewards = []
    max_avg_reward = -np.inf
    best_uav_trajectory = None
    cu_trajectory = None
    target_trajectory = None

    with tqdm(total=int(madrl_args.episodes), desc="Training Progress") as pbar:
        for i_episode in range(int(madrl_args.episodes)):
            episode_rewards_total = []
            episode_reward_bs = 0.0

            transition_dict_bs = {
                "states": [],
                "continuous_actions": [],
                "discrete_actions": [],
                "next_states": [],
                "rewards": [],
                "old_cont_log_probs": [],
                "old_disc_log_probs": [],
                "dones": [],
                "real_dones": [],
            }

            s = env.reset()
            terminal = False
            uav_positions_episode = []
            cu_positions_episode = []
            target_positions_episode = []
            reward_scaler_bs.reset()

            while not terminal:
                uav_positions_episode.append(env.getPosUAV())
                cu_positions_episode.append(env.getPosCU())
                target_positions_episode.append(env.getPosTarget())

                state_bs_norm = running_norm_bs(np.array(s["bs"], dtype=np.float32))
                action_bs, old_log_probs_bs, old_con_log_probs_bs, old_dis_log_probs_bs = agent_bs.choose_action(state_bs_norm)

                next_s, total_reward, r_dict, done = env.step({"bs": action_bs}, i_episode)

                r_bs = float(r_dict["bs"])
                r_bs_norm = float(np.asarray(reward_scaler_bs(r_bs)).item())

                episode_rewards_total.append(float(total_reward))
                episode_reward_bs += r_bs

                next_state_bs_norm = running_norm_bs(np.array(next_s["bs"], dtype=np.float32))
                transition_dict_bs["states"].append(state_bs_norm)
                transition_dict_bs["continuous_actions"].append(action_bs["continuous"])
                transition_dict_bs["discrete_actions"].append(action_bs["discrete"])
                transition_dict_bs["next_states"].append(next_state_bs_norm)
                transition_dict_bs["rewards"].append(r_bs_norm)
                transition_dict_bs["old_cont_log_probs"].append(float(np.asarray(old_con_log_probs_bs).item()))
                transition_dict_bs["old_disc_log_probs"].append(float(np.asarray(old_dis_log_probs_bs).item()))
                transition_dict_bs["dones"].append(bool(done))
                transition_dict_bs["real_dones"].append(bool(done))

                s = next_s
                terminal = done

            if np.mean(episode_rewards_total) > max_avg_reward:
                max_avg_reward = np.mean(episode_rewards_total)
                if len(uav_positions_episode) > 0:
                    best_uav_trajectory = np.array(uav_positions_episode, copy=True)
                if len(cu_positions_episode) > 0:
                    cu_trajectory = np.array(cu_positions_episode, copy=True)
                if len(target_positions_episode) > 0:
                    target_trajectory = np.array(target_positions_episode, copy=True)

            agent_bs.update(transition_dict_bs, i_episode, writer, agent_name="BS")

            avg_total_reward = np.mean(episode_rewards_total)
            avg_bs_reward = episode_reward_bs / len(episode_rewards_total)

            reward_res.append(avg_total_reward)
            all_agents_rewards.append([avg_bs_reward])

            writer.add_scalar("Reward/episode", avg_total_reward, i_episode)

            pbar.set_postfix({"avg_reward": f"{avg_total_reward:.3f}"})
            pbar.update(1)

    writer.close()

    reward_array = np.array(reward_res)
    episodes_list = np.arange(reward_array.shape[0])
    plt.plot(episodes_list, reward_array)
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.title("Single-Agent Gaussian-PPO training performance")  # 20260403 - 引入高斯分布对应的修改
    plt.show()

    filename = f"{current_time_str}_HPPO_training_rewards_seed_{base_args.seed}.csv"
    df = pd.DataFrame({
        "episode": episodes_list,
        "reward": reward_array
    })
    df.to_csv(filename, index=False)
    print(f"HPPO 训练文件已保存至 {filename}")

    all_agents_rewards = np.array(all_agents_rewards)
    df_agents = pd.DataFrame(all_agents_rewards, columns=["BS_reward"])
    df_agents.insert(0, "episode", np.arange(madrl_args.episodes))
    filename_agents = f"{current_time_str}_HPPO_all_agents_rewards_seed_{base_args.seed}.csv"
    df_agents.to_csv(filename_agents, index=False)
    print(f"HPPO BS Agent 训练奖励已保存至 {filename_agents}")

    pthname = f"{current_time_str}_mhgppo_seed_{base_args.seed}"
    os.makedirs(pthname, exist_ok=True)
    actor_path = os.path.join(pthname, f"mhgppo_actor.pth")
    critic_path = os.path.join(pthname, f"mhgppo_critic.pth")
    torch.save(agent_bs.actor.state_dict(), actor_path)
    torch.save(agent_bs.critic.state_dict(), critic_path)
    print(f"模型权重已保存至: mhgppo_actor.pth, mhgppo_critic.pth")