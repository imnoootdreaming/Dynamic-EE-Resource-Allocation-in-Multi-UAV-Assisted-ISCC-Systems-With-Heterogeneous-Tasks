"""
pure-beta-HPPO_main.py
~~~~~~~~~~~~~~~~~~~~~~~
纯端到端 PPO 训练脚本：所有 P0 优化变量由单个多头 HPPO agent 直接输出，
不依赖 CCCP 内层优化。参数体系沿用 beta-HPPO_main.py。

单 agent 架构：
  - MultiHeadActor（Beta 连续分布 + 多离散头）输出全部 P0 变量
  - 11 个连续 Beta 头 → 飞行 / 波束 / 频率 / CU 功率时长
  - I 个离散头 → 每 UAV 选择一个 CU
"""

import os
import random
import sys
import importlib.util
from datetime import datetime
from math import pi, sqrt

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# 自动向上定位 outer 根目录（含 algorithms/ 的那一层）
_current_dir = os.path.dirname(os.path.abspath(__file__))
_OUTER_ROOT = _current_dir
while not os.path.isdir(os.path.join(_OUTER_ROOT, "algorithms")):
    _parent_root = os.path.dirname(_OUTER_ROOT)
    if _parent_root == _OUTER_ROOT:
        break
    _OUTER_ROOT = _parent_root
_parent_dir = _OUTER_ROOT
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# ── 动态加载 algorithms/MHBPPO/MHBPPO_agent 模块（避免 IDE "Unresolved reference"） ──
_hppo_path = os.path.join(_parent_dir, "algorithms", "MHBPPO", "MHBPPO_agent.py")
_spec_hppo = importlib.util.spec_from_file_location("_HPPO_agent", _hppo_path)
_mod_hppo = importlib.util.module_from_spec(_spec_hppo)
_spec_hppo.loader.exec_module(_mod_hppo)
HPPO = _mod_hppo.HPPO

# ── 动态加载 utils.normalization 模块 ──
_norm_path = os.path.join(_parent_dir, "utils", "normalization.py")
_spec_norm = importlib.util.spec_from_file_location("_normalization", _norm_path)
_mod_norm = importlib.util.module_from_spec(_spec_norm)
_spec_norm.loader.exec_module(_mod_norm)
Normalization = _mod_norm.Normalization
RewardScaling = _mod_norm.RewardScaling

# ── 动态加载 environment/pure-my-env 模块 ──
_env_path = os.path.join(_parent_dir, "environment", "pure-my-env.py")
_spec_env = importlib.util.spec_from_file_location("_pure_my_env", _env_path)
_mod_env = importlib.util.module_from_spec(_spec_env)
_spec_env.loader.exec_module(_mod_env)
PureMyEnv = _mod_env.PureMyEnv


# ──────────────────────── 参数解析 ────────────────────────
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


def get_madrl_args():
    """MADRL 超参数（沿用 beta-IPPO_main.py 的默认值）。"""
    madrl_parser = argparse.ArgumentParser(description="MADRL 超参数")
    madrl_parser.add_argument("--actor_lr", type=float, default=3e-4)
    madrl_parser.add_argument("--critic_lr", type=float, default=3e-4)
    madrl_parser.add_argument("--lmbda", type=float, default=0.95)
    madrl_parser.add_argument("--eps", type=float, default=0.2)
    madrl_parser.add_argument("--gamma", type=float, default=0.99)
    madrl_parser.add_argument("--epochs", type=int, default=5)
    madrl_parser.add_argument("--total_time_slots", type=int, default=40)
    madrl_parser.add_argument("--hidden_dim", type=int, default=128)
    madrl_parser.add_argument("--episodes", type=int, default=8000)
    return madrl_parser.parse_args()


def get_base_args():
    """场景参数（沿用 beta-IPPO_main.py）。"""
    base_parser = argparse.ArgumentParser(description="场景的基本参数")

    base_parser.add_argument("--num_cases", type=int, default=30)
    base_parser.add_argument("--seed", type=int, default=42)
    base_parser.add_argument("--targets_num", type=int, default=40)
    base_parser.add_argument("--uavs_num", type=int, default=4)
    base_parser.add_argument("--cus_num", type=int, default=10)
    base_parser.add_argument("--uav_height", type=float, default=170)
    base_parser.add_argument("--radius", type=float, default=600)
    base_parser.add_argument("--center", type=float, default=[0, 0])

    base_parser.add_argument("--ref_path_loss_db", type=float, default=-30)
    base_parser.add_argument("--frac_d_lambda", type=float, default=0.5)
    base_parser.add_argument("--alpha_uav_link", type=float, default=2)
    base_parser.add_argument("--alpha_cu_link", type=float, default=2.5)
    base_parser.add_argument("--rician_factor_db", type=float, default=10)
    base_parser.add_argument("--antenna_nums", type=int, default=6)
    base_parser.add_argument("--radar_rcs", type=float, default=10)
    base_parser.add_argument("--noise_power_density_dbm", type=float, default=-174)
    base_parser.add_argument("--bandwidth", type=float, default=10e6)

    base_parser.add_argument("--uav_c1", type=float, default=0.00614)
    base_parser.add_argument("--uav_c2", type=float, default=15.976)
    base_parser.add_argument("--kappa", type=float, default=1e-28)
    base_parser.add_argument("--bs_max_freq", type=float, default=10e9)
    base_parser.add_argument("--freq_scale", type=float, default=1e9)
    base_parser.add_argument("--z_scale", type=float, default=1e5)
    base_parser.add_argument("--bs_cycles_per_bit", type=float, default=1000)
    base_parser.add_argument("--time_slot_duration", type=float, default=0.6)
    base_parser.add_argument("--uav_sen_duration", type=float, default=0.1)
    base_parser.add_argument("--cu_max_power_dbm", type=float, default=23)
    base_parser.add_argument("--uav_max_power", type=float, default=10)
    base_parser.add_argument("--cu_max_delay", type=float, default=0.6)
    base_parser.add_argument("--uav_max_delay", type=float, default=0.2)
    base_parser.add_argument("--uav_max_speed", type=float, default=40.0)
    base_parser.add_argument("--uav_min_speed", type=float, default=5.0)
    base_parser.add_argument("--uav_safe_distance", type=float, default=5.0)
    base_parser.add_argument("--sen_sinr", type=float, default=20)

    base_parser.add_argument("--omega_weight_1", type=float, default=0.2)
    base_parser.add_argument("--omega_weight_2", type=float, default=0.4)
    base_parser.add_argument("--omega_weight_3", type=float, default=0.4)

    base_parser.add_argument("--radar_duty_ratio", type=float, default=0.01)
    base_parser.add_argument("--var_range_fluctuation", type=float, default=1e-14)
    base_parser.add_argument("--radar_impulse_duration", type=float, default=2e-5)
    base_parser.add_argument("--radar_spectrum_shape", type=float, default=pi / sqrt(3))

    base_parser.add_argument("--markov_velocity", type=float, default=[1, 0, 0])
    base_parser.add_argument("--markov_memory_level", type=float, default=0.4)
    base_parser.add_argument("--markov_asymptotic_mean_of_velocity", type=float, default=[1, 0, 0])
    base_parser.add_argument("--markov_standard_deviation_of_velocity", type=float, default=2)

    base_parser.add_argument("--max_iterations", type=int, default=10)
    base_parser.add_argument("--cccp_threshold", type=float, default=1e-4)
    base_parser.add_argument("--rank1_threshold", type=float, default=1e-4)
    base_parser.add_argument("--penalty_factor", type=float, default=0.1)
    base_parser.add_argument("--zoom_factor", type=float, default=2)
    base_parser.add_argument("--enable_cccp_diagnostics", default="True")
    base_parser.add_argument("--diagnostic_violation_tol", type=float, default=1e-7)
    base_parser.add_argument("--diagnostic_top_k", type=int, default=5)
    base_parser.add_argument("--constraint_include_groups", type=str, default="4.5,4.12,4.23,4.25,4.27,4.28,4.29,4.32,4.39,4.40,4.44,4.45,auxiliary_t,var")
    base_parser.add_argument("--constraint_exclude_groups", type=str, default="")
    base_parser.add_argument("--linearization_psi_floor", type=float, default=1e-10)
    base_parser.add_argument("--enable_first_iter_rank_boost", type=lambda x: str(x).lower() == "true", default=False)
    base_parser.add_argument("--first_iter_rank_boost_eps", type=float, default=0.1)
    base_parser.add_argument("--solver_backend", type=str, default="fusion", choices=["fusion", "cvxpy"])
    base_parser.add_argument("--enable_initial_anchor", type=lambda x: str(x).lower() == "true", default=False,
                             help="是否启用初始化描点/锚点可行性检查，默认为 False")

    return base_parser.parse_args()


# ──────────────────────── 主训练入口 ────────────────────────
if __name__ == "__main__":
    current_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    writer = get_tensorboard_writer(
        log_dir=f"runs/pure-beta-hppo/{current_time_str}_pure_beta_hppo_result"
    )
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    base_args = get_base_args()
    madrl_args = get_madrl_args()
    setSeed(seed=base_args.seed)

    # ── 预生成跨 episode 固定的 SINR 序列（每个 time-slot 一个值）──
    sinr_sequence = np.random.uniform(5.0, 25.0, size=madrl_args.total_time_slots)

    env = PureMyEnv(base_args=base_args, madrl_args=madrl_args)

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
        continuous_dist_type="beta",
    )

    running_norm_bs = Normalization(state_dim_bs)
    reward_scaler_bs = RewardScaling(shape=1, gamma=madrl_args.gamma)

    reward_res = []
    obj_fun_res = []
    completion_rate_res = []
    violation_res = []
    max_avg_reward = -np.inf
    best_uav_trajectory = None
    cu_trajectory = None
    target_trajectory = None

    with tqdm(total=int(madrl_args.episodes), desc="Training Progress") as pbar:
        for i_episode in range(int(madrl_args.episodes)):
            episode_rewards_total = []
            obj_fun_total = []
            success_slots = 0
            violation_slots = 0
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
                base_args.sen_sinr = sinr_sequence[env.t]

                uav_positions_episode.append(env.getPosUAV())
                cu_positions_episode.append(env.getPosCU())
                target_positions_episode.append(env.getPosTarget())

                (
                    action_bs,
                    old_log_probs_bs,
                    old_con_log_probs_bs,
                    old_dis_log_probs_bs,
                ) = agent_bs.choose_action(state_bs_norm)

                next_s, total_reward, r_dict, done, obj_fun, success_flag = env.step(
                    {"bs": action_bs}, i_episode
                )
                success_slots += success_flag
                comps = r_dict.get("components", {})
                has_violation = (
                    comps.get("boundary_penalty", 0) > 0
                    or comps.get("collision_penalty", 0) > 0
                    or comps.get("bs_alloc_spectrum_penalty", 0) > 0
                    or comps.get("tanh_penalty_raw", 0) > 2.0
                )
                violation_slots += int(has_violation)

                r_bs = float(r_dict["bs"])
                r_bs_norm = float(np.asarray(reward_scaler_bs(r_bs)).item())

                episode_rewards_total.append(float(total_reward))
                obj_fun_total.append(float(obj_fun))
                episode_reward_bs += r_bs

                next_state_bs_norm = running_norm_bs(
                    np.array(next_s["bs"], dtype=np.float32)
                )
                transition_dict_bs["states"].append(state_bs_norm)
                transition_dict_bs["continuous_actions"].append(action_bs["continuous"])
                transition_dict_bs["discrete_actions"].append(action_bs["discrete"])
                transition_dict_bs["next_states"].append(next_state_bs_norm)
                transition_dict_bs["rewards"].append(r_bs_norm)
                transition_dict_bs["old_cont_log_probs"].append(
                    float(np.asarray(old_con_log_probs_bs).item())
                )
                transition_dict_bs["old_disc_log_probs"].append(
                    float(np.asarray(old_dis_log_probs_bs).item())
                )
                transition_dict_bs["dones"].append(bool(done))
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
            avg_violations = violation_slots / madrl_args.total_time_slots

            reward_res.append(avg_total_reward)
            obj_fun_res.append(avg_obj_fun)
            completion_rate_res.append(completion_rate)
            violation_res.append(avg_violations)

            writer.add_scalar("Reward/episode", avg_total_reward, i_episode)
            writer.add_scalar("Obj/E_sum", avg_obj_fun, i_episode)
            writer.add_scalar("Completion_Rate/episode", completion_rate, i_episode)
            writer.add_scalar("Violations/avg_per_slot", avg_violations, i_episode)

            pbar.set_postfix(
                {
                    "avg_reward": f"{avg_total_reward:.3f}",
                    "comp_rate": f"{completion_rate:.1f}%",
                    "viol": f"{avg_violations:.1f}",
                }
            )
            pbar.update(1)

    writer.close()

    # ── 保存训练曲线 ──
    reward_array = np.array(reward_res)
    obj_fun_array = np.array(obj_fun_res)
    completion_rate_array = np.array(completion_rate_res)
    violation_array = np.array(violation_res)
    episodes_list = np.arange(reward_array.shape[0])

    plt.plot(episodes_list, reward_array)
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.title("Pure-Beta-HPPO (end-to-end) training performance")
    plt.show()

    # ── 保存 CSV ──
    filename = f"{current_time_str}_pure_beta_hppo_training_seed_{base_args.seed}.csv"
    df = pd.DataFrame(
        {
            "episode": episodes_list,
            "reward": reward_array,
            "E_sum": obj_fun_array,
            "completion_rate": completion_rate_array,
            "avg_violations_per_slot": violation_array,
        }
    )
    df.to_csv(filename, index=False)
    print(f"训练文件已保存至 {filename}")

    # ── 保存模型 ──
    pthname = f"{current_time_str}_pure_beta_hppo_seed_{base_args.seed}"
    os.makedirs(pthname, exist_ok=True)
    actor_path = os.path.join(pthname, "pure_beta_hppo_actor.pth")
    critic_path = os.path.join(pthname, "pure_beta_hppo_critic.pth")
    torch.save(agent_bs.actor.state_dict(), actor_path)
    torch.save(agent_bs.critic.state_dict(), critic_path)
    print(f"模型权重已保存至: {actor_path}, {critic_path}")

    # ── 保存最佳轨迹 ──
    # if best_uav_trajectory is not None:
    #     uav_traj_list = []
    #     for t in range(best_uav_trajectory.shape[0]):
    #         for uav_i in range(base_args.uavs_num):
    #             x, y, z = best_uav_trajectory[t, uav_i]
    #             uav_traj_list.append([t, uav_i, x, y, z])
    #     df_uav = pd.DataFrame(uav_traj_list, columns=["time_slot", "uav_id", "x", "y", "z"])
    #     df_uav.to_csv(f"{current_time_str}_SEED{base_args.seed}_best_uav_traj.csv", index=False)
    #     print(f"最佳 UAV 轨迹已保存")

    # if cu_trajectory is not None:
    #     cu_traj_list = []
    #     for t in range(cu_trajectory.shape[0]):
    #         for cu_i in range(base_args.cus_num):
    #             x, y, z = cu_trajectory[t, cu_i]
    #             cu_traj_list.append([t, cu_i, x, y, z])
    #     df_cu = pd.DataFrame(cu_traj_list, columns=["time_slot", "cu_id", "x", "y", "z"])
    #     df_cu.to_csv(f"{current_time_str}_SEED{base_args.seed}_cu_traj.csv", index=False)

    # if target_trajectory is not None:
    #     target_traj_list = []
    #     for t in range(target_trajectory.shape[0]):
    #         for target_i in range(base_args.targets_num):
    #             x, y, z = target_trajectory[t, target_i]
    #             target_traj_list.append([t, target_i, x, y, z])
    #     df_target = pd.DataFrame(target_traj_list, columns=["time_slot", "target_id", "x", "y", "z"])
    #     df_target.to_csv(f"{current_time_str}_SEED{base_args.seed}_target_traj.csv", index=False)

    print("\nPure-Beta-HPPO 训练完成!")