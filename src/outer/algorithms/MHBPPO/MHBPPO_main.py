import os
import random
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
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
from configs.base_params import get_base_args
from configs.madrl_args import get_madrl_args
from environment.my_env import MyEnv
from utils.normalization import Normalization, RewardScaling


def setSeed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_tensorboard_writer(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    return SummaryWriter(log_dir=log_dir)


if __name__ == "__main__":
    current_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    writer = get_tensorboard_writer(log_dir=f"runs/beta-hppo/{current_time_str}_beta-hppo_result")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    base_args = get_base_args()
    madrl_args = get_madrl_args()
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
        entropy_coef=0.01
    )

    running_norm_bs = Normalization(state_dim_bs)
    reward_scaler_bs = RewardScaling(shape=1, gamma=madrl_args.gamma)

    reward_res = []
    completion_rate_res = []
    obj_fun_res = []
    all_agents_rewards = []
    max_avg_reward = -np.inf
    best_uav_trajectory = None
    cu_trajectory = None
    target_trajectory = None

    with tqdm(total=int(madrl_args.episodes), desc="Training Progress") as pbar:
        for i_episode in range(int(madrl_args.episodes)):
            episode_rewards_total = []
            obj_fun_total = []
            success_slots = 0  # 用于统计本 episode 内成功时隙数
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
            state_bs_norm = running_norm_bs(np.array(s["bs"], dtype=np.float32))
            terminal = False
            uav_positions_episode = []
            cu_positions_episode = []
            target_positions_episode = []
            reward_scaler_bs.reset()

            while not terminal:
                uav_positions_episode.append(env.getPosUAV())
                cu_positions_episode.append(env.getPosCU())
                target_positions_episode.append(env.getPosTarget())

                action_bs, old_log_probs_bs, old_con_log_probs_bs, old_dis_log_probs_bs = agent_bs.choose_action(state_bs_norm)

                next_s, total_reward, r_dict, done, obj_fun, success_flag = env.step({"bs": action_bs}, i_episode)
                success_slots += success_flag
                
                r_bs = float(r_dict["bs"])
                r_bs_norm = float(np.asarray(reward_scaler_bs(r_bs)).item())

                episode_rewards_total.append(float(total_reward))
                obj_fun_total.append(float(obj_fun))
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
                # NOTE - 调整了 real_dones，因为这是一个三十个时隙的感知任务
                transition_dict_bs["real_dones"].append(bool(done))
                
                state_bs_norm = next_state_bs_norm
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
            avg_obj_fun = np.mean(obj_fun_total)
            avg_bs_reward = episode_reward_bs / len(episode_rewards_total)
            completion_rate = (success_slots / madrl_args.total_time_slots) * 100.0

            reward_res.append(avg_total_reward)
            obj_fun_res.append(avg_obj_fun)
            all_agents_rewards.append([avg_bs_reward])
            completion_rate_res.append(completion_rate)  # 新增列表存储完成率

            writer.add_scalar("Reward/episode", avg_total_reward, i_episode)
            writer.add_scalar("Obj/episode", avg_obj_fun, i_episode)
            writer.add_scalar("Completion Rate/episode", completion_rate, i_episode)


            pbar.set_postfix({"avg_reward": f"{avg_total_reward:.3f}"})
            pbar.update(1)

    writer.close()

    reward_array = np.array(reward_res)
    obj_fun_array = np.array(obj_fun_res)
    completion_rate_array = np.array(completion_rate_res)
    episodes_list = np.arange(reward_array.shape[0])
    plt.plot(episodes_list, reward_array)
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.title("Single-Agent Beta-PPO training performance")
    plt.show()

    filename = f"{current_time_str}_HPPO_training_rewards_seed_{base_args.seed}.csv"
    df = pd.DataFrame({
        "episode": episodes_list,
        "reward": reward_array,
        "obj" : obj_fun_array,
        "completion_rate" : completion_rate_array
    })
    df.to_csv(filename, index=False)
    print(f"HPPO 训练文件已保存至 {filename}")

    pthname = f"{current_time_str}_mhbppo_seed_{base_args.seed}"
    os.makedirs(pthname, exist_ok=True)
    actor_path = os.path.join(pthname, f"mhbppo_actor.pth")
    critic_path = os.path.join(pthname, f"mhbppo_critic.pth")
    torch.save(agent_bs.actor.state_dict(), actor_path)
    torch.save(agent_bs.critic.state_dict(), critic_path)
    print(f"模型权重已保存至: mhbppo_actor.pth, mhbppo_critic.pth")
