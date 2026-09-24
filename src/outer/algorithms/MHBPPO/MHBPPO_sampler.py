"""MHBPPO 异步采样进程（生产者）。

IMPALA 式 actor-learner 架构中的采样端：
- 本进程持有 MyEnv、CPU 版 actor 副本、Normalization / RewardScaling、最优轨迹记录，
  持续 rollout 完整 episode 并将结果放入有界 sample_queue（满则阻塞 => 限制样本 staleness）。
- 每个 episode 开始前从 weight_queue 取最新 actor 权重并加载（单 episode 内参数版本一致），
  old_log_probs 随采样时的参数记录，PPO ratio 语义保持正确。
- 本进程不打印任何常规日志（避免与主进程 tqdm 输出交错），异常通过 error sentinel 上报。

与学习进程（MHBPPO_main.py）的数据契约：
- sample_queue 元素（本进程 -> 学习进程）:
    {
        "transition_dict": dict,     # 与原 transition_dict_bs 结构完全一致
        "avg_total_reward": float,
        "avg_obj_fun": float,
        "avg_bs_reward": float,
        "completion_rate": float,
        "is_best": bool,             # 是否刷新历史最优平均奖励
        "weights_version": int,      # 本 episode 采样时使用的参数版本（-1 = 从未收到权重）
        "weights_checksum": float,   # 本进程 actor 加载后的参数指纹（供学习端比对验证）
    }
    异常哨兵: {"error": "<traceback 字符串>"}
- weight_queue 元素（学习进程 -> 本进程, maxsize=1）:
    {"actor_state_dict": OrderedDict}  # 已转 CPU 的 actor state_dict
- stop_event: 学习进程置位后本进程在 episode 边界 / put 等待处感知并退出。

注意：Windows/Linux 均以 spawn 方式启动，本 worker 必须保持为模块顶层函数，
且内部自行构建 env / actor（不跨进程传递 env、tensor、agent 对象）。
"""

import queue as queue_module
import os
import random
import sys
import traceback

import numpy as np
import torch

# 自动向上定位 outer 根目录（含 algorithms/ 的那一层），与 MHBPPO_main.py 保持一致。
# spawn 子进程重新 import 本模块时，该 bootstrap 保证依赖模块可导入。
_OUTER_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_OUTER_ROOT, "algorithms")):
    _parent_root = os.path.dirname(_OUTER_ROOT)
    if _parent_root == _OUTER_ROOT:
        break
    _OUTER_ROOT = _parent_root
if _OUTER_ROOT not in sys.path:
    sys.path.insert(0, _OUTER_ROOT)

from algorithms.MHBPPO.MHBPPO_agent import MultiHeadActor
from environment.my_env import MyEnv
from utils.normalization import Normalization, RewardScaling


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _choose_action(actor, state_norm):
    """CPU 版动作采样，返回与 HPPO.choose_action 等价的结果（标量 log_prob）。"""
    s = torch.tensor(state_norm, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        continuous_actions, discrete_actions, _, con_log_prob, dis_log_prob = actor.sample(s)
    action = {
        "continuous": continuous_actions.squeeze(0).cpu().numpy().astype(np.float32),
        "discrete": discrete_actions.squeeze(0).cpu().numpy().astype(np.int64),
    }
    return action, float(np.asarray(con_log_prob).item()), float(np.asarray(dis_log_prob).item())


def _actor_checksum(actor):
    """本进程 actor 参数的标量指纹（与学习端 actor_checksum 同算法）。

    采样端加载权重后用**自己的 actor** 计算并回传，学习端比对即可证明权重真正生效。
    """
    total = 0.0
    with torch.no_grad():
        for param in actor.parameters():
            total += float(param.detach().abs().sum().cpu().item())
    return round(total, 6)


def _load_latest_weights(weight_queue, actor):
    """清空 weight_queue 并加载最新一份 actor 权重（仅在本函数末尾 load，保证取到的是最新版本）。

    Returns:
        (version, None)：本次加载到的参数版本号；从未收到任何权重时返回 (None, None)。
    """
    latest_state_dict = None
    latest_version = None
    while True:
        try:
            payload = weight_queue.get_nowait()
        except queue_module.Empty:
            break
        if payload is not None:
            latest_state_dict = payload["actor_state_dict"]
            latest_version = payload.get("weights_version")
    if latest_state_dict is not None:
        actor.load_state_dict(latest_state_dict)
    return latest_version


def _put_with_stop(sample_queue, item, stop_event, timeout=1.0):
    """向有界队列 put，等待期间周期性检查 stop_event。返回 False 表示收到停止信号。"""
    while not stop_event.is_set():
        try:
            sample_queue.put(item, timeout=timeout)
            return True
        except queue_module.Full:
            continue
    return False


def _put_error_sentinel(sample_queue, error_msg):
    """尽力而为地上报异常哨兵（即使队列满也只等待有限时间，避免子进程悬挂）。"""
    for _ in range(5):
        try:
            sample_queue.put({"error": error_msg}, timeout=1.0)
            return True
        except queue_module.Full:
            continue
    return False


def sampler_worker(base_args, madrl_args, sample_queue, weight_queue, stop_event, seed_offset=1):
    """采样进程主函数（必须是模块顶层函数，spawn 可 pickle）。

    Args:
        base_args / madrl_args: 主进程已解析的 argparse Namespace（可 pickle）。
        sample_queue: 有界样本队列（采样 -> 学习），maxsize 建议 2~4。
        weight_queue: 权重队列（学习 -> 采样），maxsize=1，学习端 drain-then-put。
        stop_event: 学习进程控制的停止事件。
        seed_offset: 采样进程随机种子偏移，避免与学习进程使用完全相同的 RNG 流。
    """
    try:
        _set_seed(base_args.seed + seed_offset)
        # 采样进程不触碰 CUDA：actor 副本固定在 CPU 上推理（小 MLP，开销可忽略），
        # 也避免与学习进程争用 GPU 以及 spawn 下子进程使用 CUDA 的潜在问题。
        device = torch.device("cpu")

        env = MyEnv(base_args=base_args, madrl_args=madrl_args)

        state_dim_bs = env.observation_space["bs"].shape[0]
        bs_continuous_action_space = env.action_space["bs"]["continuous"]
        bs_discrete_action_space = env.action_space["bs"]["discrete"]

        # 与学习进程 HPPO 内部构造的 actor 结构完全一致（连续分布默认 beta），
        # 初始随机权重由学习进程通过 weight_queue 下发的第一份 state_dict 覆盖。
        actor = MultiHeadActor(
            state_dim=state_dim_bs,
            hidden_dim=madrl_args.hidden_dim,
            continuous_action_splits=env.bs_continuous_action_splits,
            discrete_action_dims=bs_discrete_action_space.nvec,
            continuous_dist_type="beta",
        ).to(device)

        running_norm_bs = Normalization(state_dim_bs)
        reward_scaler_bs = RewardScaling(shape=1, gamma=madrl_args.gamma)

        max_avg_reward = -np.inf
        best_uav_trajectory = None
        best_cu_trajectory = None
        best_target_trajectory = None

        episode_idx = 0
        # 探针：本进程当前使用的参数版本号与指纹（随每个样本回传，供学习端验证权重同步）
        # -1 表示从未收到任何权重（采样端将一直使用随机初始策略）
        current_weights_version = -1
        current_weights_checksum = None

        while not stop_event.is_set():
            # episode 边界：加载学习端最新下发的 actor 权重，本 episode 全程使用该版本
            loaded_version = _load_latest_weights(weight_queue, actor)
            if loaded_version is not None:
                current_weights_version = loaded_version
                # 用**本进程 actor 加载后**的参数计算指纹，证明权重确实落到了网络上
                current_weights_checksum = _actor_checksum(actor)

            episode_rewards_total = []
            obj_fun_total = []
            success_slots = 0
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

                action_bs, old_con_log_probs_bs, old_dis_log_probs_bs = _choose_action(actor, state_bs_norm)

                next_s, total_reward, r_dict, done, obj_fun, success_flag = env.step({"bs": action_bs}, episode_idx)
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
                transition_dict_bs["old_cont_log_probs"].append(old_con_log_probs_bs)
                transition_dict_bs["old_disc_log_probs"].append(old_dis_log_probs_bs)
                transition_dict_bs["dones"].append(bool(done))
                # NOTE - 调整了 real_dones，因为这是一个三十个时隙的感知任务
                transition_dict_bs["real_dones"].append(bool(done))

                state_bs_norm = next_state_bs_norm
                terminal = done

            avg_total_reward = float(np.mean(episode_rewards_total))
            is_best = False
            if avg_total_reward > max_avg_reward:
                max_avg_reward = avg_total_reward
                is_best = True
                if len(uav_positions_episode) > 0:
                    best_uav_trajectory = np.array(uav_positions_episode, copy=True)
                if len(cu_positions_episode) > 0:
                    best_cu_trajectory = np.array(cu_positions_episode, copy=True)
                if len(target_positions_episode) > 0:
                    best_target_trajectory = np.array(target_positions_episode, copy=True)

            sample_result = {
                "transition_dict": transition_dict_bs,
                "avg_total_reward": avg_total_reward,
                "avg_obj_fun": float(np.mean(obj_fun_total)),
                "avg_bs_reward": episode_reward_bs / len(episode_rewards_total),
                "completion_rate": (success_slots / madrl_args.total_time_slots) * 100.0,
                "is_best": is_best,
                # ── 探针字段：本 episode 实际使用的参数版本 / 指纹 ──
                "weights_version": current_weights_version,
                "weights_checksum": current_weights_checksum,
            }

            # 有界队列：队列满时阻塞在此（背压 => staleness 上限 = maxsize 个参数版本）
            if not _put_with_stop(sample_queue, sample_result, stop_event):
                break

            episode_idx += 1

    except Exception:
        # 异常哨兵：主进程收到后终止训练并抛出，避免其在空队列上永久阻塞
        _put_error_sentinel(sample_queue, traceback.format_exc())
