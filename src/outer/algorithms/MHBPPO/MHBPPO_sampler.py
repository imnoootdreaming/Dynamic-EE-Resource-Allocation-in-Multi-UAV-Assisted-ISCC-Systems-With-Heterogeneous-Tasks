"""MHBPPO 完整 episode 采样进程（生产者）。

IMPALA 式 actor-learner 架构中的采样端：
- 本进程持有 MyEnv、CPU 版 actor 副本；状态和奖励归一化统计量由 learner 统一维护，
  持续 rollout 完整 episode 并将结果放入有界 sample_queue（满则阻塞 => 限制样本 staleness）。
- 每个 episode 开始前从 weight_queue 取最新 actor 权重并加载（单 episode 内参数版本一致），
  old_log_probs 随采样时的参数记录，PPO ratio 语义保持正确。
- 新同步模式：所有 worker 使用共同场景 seed/offset，保持外生时序固定；
  动作 RNG 从共同 master seed 的命名域派生，各 worker 独立且不随 worker 数改变。
  每轮只采一次完整 episode，等待 learner 聚合更新后再接收下一轮许可。
- 旧异步接口保留不同 worker 的 scene offset 与逐 episode 随机化配置。
- 本进程不打印任何常规日志（避免与主进程 tqdm 输出交错），异常通过 error sentinel 上报。

与学习进程（MHBPPO_main.py）的数据契约：
- sample_queue 元素（本进程 -> 学习进程）:
    {
        "transition_dict": dict,     # 与原 transition_dict_bs 结构完全一致
        "action_traces": list,       # 各时隙 env.step 实际变换后的动作快照
        "sampler_id": int,           # 0-based worker 编号
        "seed_offset": int,
        "episode_idx": int,          # 本 worker 内 0-based episode 序号
        "avg_total_reward": float,
        "avg_obj_fun": float,
        "avg_bs_reward": float,
        "completion_rate": float,
        "all_slots_success": bool,
        "episode_success_bonus_total": float,
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
import time
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

from algorithms.MHBPPO.policy_diagnostics import (
    single_state_policy_distribution, summarize_policy_distribution,
)
from algorithms.MHBPPO.log_compaction import compact_action_trace, compact_env_trace, compact_episode_payload
from algorithms.MHBPPO.energy_statistics import ENERGY_OUTCOME_KEYS, summarize_energy
from algorithms.MHBPPO.MHBPPO_agent import MultiHeadActor
from algorithms.MHBPPO.policy_initialization import actor_architecture_metadata
from algorithms.MHBPPO.rollout_batch import (
    actor_state_signature, actor_parameter_checksum, derive_action_seed, normalization_signature,
)
from environment.my_env import MyEnv
from algorithms.MHBPPO.unknown_logging import unknown_start_iteration
from p5_failure_capture import capture_unknown_slot


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _choose_action(actor, state_norm, *, return_distribution_stats=False):
    """CPU 版动作采样，返回与 HPPO.choose_action 等价的结果（标量 log_prob）。"""
    s = torch.tensor(state_norm, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        result = actor.sample(s, return_distribution_stats=return_distribution_stats)
        continuous_actions, discrete_actions, _, con_log_prob, dis_log_prob = result[:5]
    action = {
        "continuous": continuous_actions.squeeze(0).cpu().numpy().astype(np.float32),
        "discrete": discrete_actions.squeeze(0).cpu().numpy().astype(np.int64),
    }
    result_base = (action, float(np.asarray(con_log_prob).item()), float(np.asarray(dis_log_prob).item()))
    if return_distribution_stats:
        return (*result_base, single_state_policy_distribution(result[5]))
    return result_base


def _actor_checksum(actor):
    """加载后在本地actor上计算，与learner共用不依赖Torch线程数的归约。"""
    return actor_parameter_checksum(actor)


def summarize_sensing_penalty(action_traces):
    """Reward accounting only; exclude unavailable scores from the Q mean."""
    outcomes = [trace.get("outcome", {}) for trace in action_traces]
    scores = [float(outcome["sensing_penalty_score"]) for outcome in outcomes
              if outcome.get("sensing_penalty_status") == "available"
              and outcome.get("sensing_penalty_score") is not None
              and np.isfinite(outcome["sensing_penalty_score"])]
    unavailable = sum(outcome.get("sensing_penalty_status") == "unavailable"
                      for outcome in outcomes)
    removed = any(outcome.get("sensing_penalty_status") == "removed" for outcome in outcomes)
    base_rewards = [float(outcome.get("base_reward", outcome["total_reward"]))
                    for outcome in outcomes]
    applied = [float(outcome.get("sensing_penalty_applied", 0.0)) for outcome in outcomes]
    return {
        "avg_base_reward": float(np.mean(base_rewards)) if outcomes else float("nan"),
        "avg_sensing_penalty_score": float(np.mean(scores)) if scores else
            (float("nan") if unavailable or removed else 0.0),
        "avg_sensing_penalty_applied": float(np.mean(applied)) if outcomes else 0.0,
        "sensing_penalty_score_valid_count": len(scores),
        "sensing_penalty_unavailable_count": unavailable,
    }


def _numpy_state_dict_to_tensor(state_dict):
    """将跨进程以 numpy 形式传递的 state_dict 还原为张量，供 load_state_dict 使用。"""
    return {k: torch.as_tensor(np.asarray(v)) for k, v in state_dict.items()}


def _load_latest_weights(weight_queue, actor):
    """清空 weight_queue 并加载最新一份 actor 权重（仅在本函数末尾 load，保证取到的是最新版本）。

    Returns:
        (version, mean, std)：本次加载到的参数版本号和 learner 统一维护的状态统计量。
    """
    latest_state_dict = None
    latest_version = None
    latest_mean = None
    latest_std = None
    while True:
        try:
            payload = weight_queue.get_nowait()
        except queue_module.Empty:
            break
        if payload is not None:
            _validate_payload_architecture(payload, actor)
            latest_state_dict = payload["actor_state_dict"]
            latest_version = payload.get("weights_version")
            latest_mean = payload.get("state_norm_mean")
            latest_std = payload.get("state_norm_std")
    if latest_state_dict is not None:
        actor.load_state_dict(_numpy_state_dict_to_tensor(latest_state_dict))
    return latest_version, latest_mean, latest_std


def _validate_payload_architecture(payload, actor):
    """New broadcasts declare the class; legacy payloads retain strict state loading."""
    declared = payload.get("actor_architecture")
    if declared is not None and declared != actor_architecture_metadata(actor):
        raise ValueError("sampling actor architecture differs from learner broadcast")


def _load_weights_until(weight_queue, actor, target_version, stop_event, timeout=1.0,
                        *, strict_version=False):
    """同步模式：阻塞等到 version >= target_version 的权重到达后加载。

    不能用 _load_latest_weights 的非阻塞读法：learner 下发的 actor state_dict 是 MB 级
    对象，由 feeder 线程异步写入管道；采样端拿到许可令牌后立刻 get_nowait() 会读到空
    队列，而 learner 下一次 broadcast 的 drain-then-put 又会把这份权重清掉，采样端将
    永远停留在初始版本（表现为 wv 恒为 0、策略从不更新，症状与「内层恒不可行」一致）。
    """
    latest_state_dict = None
    latest_version = None
    latest_mean = None
    latest_std = None
    while not stop_event.is_set():
        try:
            payload = weight_queue.get(timeout=timeout)
        except queue_module.Empty:
            continue
        if payload is None:
            continue
        _validate_payload_architecture(payload, actor)
        latest_state_dict = payload["actor_state_dict"]
        latest_version = payload.get("weights_version")
        latest_mean = payload.get("state_norm_mean")
        latest_std = payload.get("state_norm_std")
        if strict_version:
            if latest_version is None or target_version is None:
                raise ValueError("synchronous episode requires an explicit policy version")
            if latest_version > target_version:
                raise ValueError(f"future sampling policy version {latest_version}, expected {target_version}")
            if latest_version < target_version:
                latest_state_dict = None
                continue
        # Legacy callers retain the previous >= rule; shared-scene rounds are exact.
        if latest_version is not None and target_version is not None and latest_version >= target_version:
            break
    if strict_version and stop_event.is_set():
        return None, None, None
    if latest_state_dict is not None:
        actor.load_state_dict(_numpy_state_dict_to_tensor(latest_state_dict))
    return latest_version, latest_mean, latest_std


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


def sampler_worker(base_args, madrl_args, sample_queue, weight_queue, stop_event,
                   seed_offset=1, control_queue=None, *, worker_id=None, scene_seed_offset=None,
                   capture_unknown=False):
    # 环境的逐时隙诊断由 learner 侧聚合日志统一记录；采样子进程不向终端刷屏。
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
    """采样进程主函数（必须是模块顶层函数，spawn 可 pickle）。

    Args:
        base_args / madrl_args: 主进程已解析的 argparse Namespace（可 pickle）。
        sample_queue: 有界样本队列（采样 -> 学习），maxsize 建议 2~4。
        weight_queue: 权重队列（学习 -> 采样），maxsize=1，学习端 drain-then-put。
        stop_event: 学习进程控制的停止事件。
        seed_offset: 旧接口的场景/动作种子偏移。
        worker_id / scene_seed_offset: 新同步接口的动作流身份与共同场景偏移。
    """
    try:
        shared_scene_contract = worker_id is not None
        if shared_scene_contract:
            if isinstance(worker_id, bool) or int(worker_id) != worker_id or worker_id < 0:
                raise ValueError("worker_id must be a nonnegative integer")
            worker_id = int(worker_id)
            if control_queue is None:
                raise ValueError("shared-scene sampling requires a synchronous control queue")
            if scene_seed_offset is None:
                scene_seed_offset = int(getattr(madrl_args, "scene_seed_offset", 0))
            if int(getattr(madrl_args, "num_samplers", 1)) > 1 and any(bool(getattr(madrl_args, name, False)) for name in (
                    "randomize_layout_per_episode", "randomize_cu_traj_per_episode", "randomize_nlos_per_episode")):
                raise ValueError("shared-scene sampling requires fixed per-episode external scenes")
            action_seed = derive_action_seed(int(base_args.seed), worker_id)
            effective_scene_offset = int(scene_seed_offset)
        else:
            # Old positional interface preserves its scene and action seed behavior.
            worker_id = int(seed_offset-1)
            action_seed = int(base_args.seed+seed_offset)
            effective_scene_offset = int(seed_offset)
        worker_threads = int(getattr(madrl_args, "worker_num_threads", 0))
        if worker_threads > 0:
            torch.set_num_threads(worker_threads)
        _set_seed(action_seed)
        # 采样进程不触碰 CUDA：actor 副本固定在 CPU 上推理（小 MLP，开销可忽略），
        # 也避免与学习进程争用 GPU 以及 spawn 下子进程使用 CUDA 的潜在问题。
        device = torch.device("cpu")

        env = MyEnv(base_args=base_args, madrl_args=madrl_args, seed_offset=effective_scene_offset)

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
            structured_beam_alpha=0.0,
        ).to(device)
        actor_architecture = actor_architecture_metadata(actor)

        # 状态归一化统计量由 learner 统一维护；sampler 在一个 episode 内固定快照。
        state_norm_mean = np.zeros(state_dim_bs, dtype=np.float64)
        state_norm_std = np.ones(state_dim_bs, dtype=np.float64)

        episode_idx = 0
        # 探针：本进程当前使用的参数版本号与指纹（随每个样本回传，供学习端验证权重同步）
        # -1 表示从未收到任何权重（采样端将一直使用随机初始策略）
        current_weights_version = -1
        current_weights_checksum = None
        last_collection_round = -1
        capture_start = unknown_start_iteration(getattr(madrl_args, "max_updates", 0))

        while not stop_event.is_set():
            # 同步模式下由 learner 发放一次性许可；worker 采完一个 episode
            # 后必须等待 learner 完成 PPO 更新并广播新权重。
            if control_queue is not None:
                target_version = None
                collection_round = episode_idx
                while not stop_event.is_set():
                    try:
                        # 许可令牌携带本 episode 应使用的参数版本号
                        token = control_queue.get(timeout=1.0)
                        if isinstance(token, dict):
                            target_version = token["expected_policy_version"]
                            collection_round = token["collection_round"]
                            token_episode = token["episode_idx"]
                            if collection_round != token_episode:
                                raise ValueError("collection round and episode index must agree")
                            if shared_scene_contract and collection_round != last_collection_round+1:
                                raise ValueError("duplicate, reordered, or skipped synchronous collection round")
                            episode_idx = int(token_episode)
                        else:
                            target_version = token
                        break
                    except queue_module.Empty:
                        continue
                if stop_event.is_set():
                    break
                # episode 边界：阻塞等到该版本权重到手后加载，本 episode 全程使用该版本
                loaded_version, loaded_mean, loaded_std = _load_weights_until(
                    weight_queue, actor, target_version, stop_event,
                    strict_version=shared_scene_contract)
                if stop_event.is_set():
                    break
            else:
                collection_round = episode_idx
                # 异步模式：非阻塞取最新一份（允许落后若干版本，stale 由队列容量限制）
                loaded_version, loaded_mean, loaded_std = _load_latest_weights(weight_queue, actor)
            if loaded_version is not None:
                current_weights_version = loaded_version
                # 用**本进程 actor 加载后**的参数计算指纹，证明权重确实落到了网络上
                current_weights_checksum = _actor_checksum(actor)
                if loaded_mean is not None and loaded_std is not None:
                    state_norm_mean = np.asarray(loaded_mean, dtype=np.float64).copy()
                    state_norm_std = np.maximum(np.asarray(loaded_std, dtype=np.float64), 1e-8)
            if shared_scene_contract and (loaded_mean is None or loaded_std is None):
                raise ValueError("synchronous episode requires a supplied frozen normalization snapshot")
            current_actor_signature = actor_state_signature(actor)
            current_norm_signature = normalization_signature(state_norm_mean, state_norm_std)

            episode_rewards_total = []
            obj_fun_total = []
            success_slots = 0
            episode_reward_bs = 0.0
            episode_success_bonus_total = 0.0
            finite_obj_slots = 0
            no_solution_slots = 0
            collision_slots = 0
            spectrum_penalty_slots = 0
            solver_status_counts = {k: 0 for k in ("optimal", "infeasible", "unknown", "error", "other")}
            action_traces = []
            diagnostic_traces = []
            unknown_capture_payloads = {}
            # Async update assignment is only known when the learner consumes it.
            capture_enabled = (capture_unknown and capture_start is not None and (
                control_queue is None or capture_start <= collection_round + 1 <= madrl_args.max_updates))
            action_min = float("inf")
            action_max = float("-inf")

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
                "raw_states": [],
                "raw_next_states": [],
            }

            collection_started_at = time.perf_counter()
            s = env.reset(episode_idx)
            exogenous_scene_signature = env.get_exogenous_scene_signature()
            # Independent of env_diagnostics; pair this episode's layout with its action traces.
            trajectory_scene = {
                "target_positions": env.init_targets_pos.tolist(),
                "uav_positions": env.init_uavs_pos.tolist(),
                "target_sector": env.target_sector.tolist(),
                "target_layout": env.target_layout_info,
                "center": np.asarray(base_args.center, dtype=float).tolist(),
                "deploy_radius": float(base_args.radius),
                "bs_position": np.asarray(base_args.bs_pos, dtype=float).tolist(),
            }
            diagnostic_scene = env.get_diagnostic_scene()
            raw_state_bs = np.asarray(s["bs"], dtype=np.float32)
            state_bs_norm = (raw_state_bs - state_norm_mean) / (state_norm_std + 1e-8)
            terminal = False

            while not terminal:
                action_bs, old_con_log_probs_bs, old_dis_log_probs_bs, policy_stats = _choose_action(
                    actor, state_bs_norm, return_distribution_stats=True,
                )

                with capture_unknown_slot(capture_enabled) as capture_state:
                    next_s, total_reward, r_dict, done, obj_fun, success_flag = env.step({"bs": action_bs}, episode_idx)
                if capture_state is not None and capture_state["payload"] is not None:
                    unknown_capture_payloads[len(action_traces)] = capture_state["payload"]
                if env.last_action_trace is None:
                    raise RuntimeError("env.step 未产生动作诊断快照")
                action_traces.append({
                    **env.last_action_trace,
                    "policy_distribution": policy_stats,
                    "outcome": {
                        "total_reward": float(total_reward),
                        "obj_fun": float(obj_fun),
                        "success": int(success_flag),
                        "done": bool(done),
                        "solver_status_kind": str(r_dict.get("components", {}).get("solver_status_kind", "other")),
                        **{key: r_dict["components"][key] for key in ENERGY_OUTCOME_KEYS
                           if key in r_dict.get("components", {})},
                        **{key: r_dict["components"][key] for key in (
                            "solver_status", "raw_solver_status", "raw_solver_status_kind",
                            "raw_problem_status", "status_source", "optimizer_response_code",
                            "presolve_contract", "solver_skipped", "pc3p_called", "presolve_seconds",
                            "presolve_certificate",
                            "infeasibility_certificate",
                            "base_reward", "base_no_solution_penalty", "no_solution_penalty",
                            "sensing_penalty_score", "sensing_penalty_applied",
                            "sensing_penalty_per_uav", "sensing_margin",
                            "sensing_penalty_status", "sensing_penalty_reason",
                            "sensing_penalty_contract",
                            "slot_success", "episode_all_success",
                            "slot_reward_before_episode_bonus",
                            "episode_success_bonus", "episode_success_bonus_applied",
                            "episode_success_bonus_contract",
                        ) if key in r_dict.get("components", {})},
                    },
                })
                if env.env_diagnostics:
                    if env.last_env_diagnostic is None:
                        raise RuntimeError("env.step 未产生已启用的环境诊断")
                    diagnostic_traces.append(
                        compact_env_trace(env.last_env_diagnostic)
                        if getattr(madrl_args, "log_detail_level", "compact") == "compact"
                        else env.last_env_diagnostic)
                if getattr(madrl_args, "log_detail_level", "compact") == "compact":
                    action_traces[-1] = compact_action_trace(action_traces[-1], keep_policy=True)
                success_slots += success_flag
                finite_obj_slots += int(np.isfinite(obj_fun))
                no_solution_slots += int(r_dict.get("components", {}).get("no_solution_penalty", 0) > 0)
                collision_slots += int(r_dict.get("components", {}).get("uav_collision_penalty_sum", 0) > 0)
                spectrum_penalty_slots += int(r_dict.get("components", {}).get("bs_alloc_spectrum_penalty", 0) > 0)
                solver_kind = str(r_dict.get("components", {}).get("solver_status_kind", "other"))
                solver_status_counts[solver_kind] = solver_status_counts.get(solver_kind, 0) + 1
                action_min = min(action_min, float(np.min(action_bs["continuous"])))
                action_max = max(action_max, float(np.max(action_bs["continuous"])))

                r_bs = float(r_dict["bs"])
                episode_rewards_total.append(float(total_reward))
                obj_fun_total.append(float(obj_fun))
                episode_reward_bs += r_bs
                episode_success_bonus_total += float(
                    r_dict.get("components", {}).get("episode_success_bonus_applied", 0.0))

                raw_next_state_bs = np.asarray(next_s["bs"], dtype=np.float32)
                next_state_bs_norm = (raw_next_state_bs - state_norm_mean) / (state_norm_std + 1e-8)
                transition_dict_bs["states"].append(state_bs_norm)
                transition_dict_bs["continuous_actions"].append(action_bs["continuous"])
                transition_dict_bs["discrete_actions"].append(action_bs["discrete"])
                transition_dict_bs["next_states"].append(next_state_bs_norm)
                # 发送原始即时奖励；由 learner 使用统一的全局统计量归一化。
                transition_dict_bs["rewards"].append(r_bs)
                transition_dict_bs["old_cont_log_probs"].append(old_con_log_probs_bs)
                transition_dict_bs["old_disc_log_probs"].append(old_dis_log_probs_bs)
                transition_dict_bs["dones"].append(bool(done))
                # NOTE - 调整了 real_dones，因为这是一个三十个时隙的感知任务
                transition_dict_bs["real_dones"].append(bool(done))
                transition_dict_bs["raw_states"].append(raw_state_bs.copy())
                transition_dict_bs["raw_next_states"].append(raw_next_state_bs.copy())

                state_bs_norm = next_state_bs_norm
                raw_state_bs = raw_next_state_bs
                terminal = done

            collection_finished_at = time.perf_counter()
            avg_total_reward = float(np.mean(episode_rewards_total))

            sample_result = {
                "transition_dict": transition_dict_bs,
                "action_traces": action_traces,
                "policy_distribution_summary": summarize_policy_distribution(
                    action_traces, actor.continuous_action_splits,
                ),
                "trajectory_scene": trajectory_scene,
                "diagnostic_traces": diagnostic_traces,
                "unknown_capture_payloads": unknown_capture_payloads,
                "diagnostic_scene": diagnostic_scene,
                "diagnostics_enabled": env.env_diagnostics,
                "sampler_id": worker_id,
                "worker_id": worker_id,
                "seed_offset": effective_scene_offset,
                "scene_seed_offset": effective_scene_offset,
                "action_seed": action_seed,
                "collection_round": int(collection_round),
                "collection_started_at": collection_started_at,
                "collection_finished_at": collection_finished_at,
                "collection_seconds": collection_finished_at-collection_started_at,
                "actor_signature": current_actor_signature,
                "actor_architecture": actor_architecture,
                "norm_signature": current_norm_signature,
                "scene_signature": exogenous_scene_signature["signature"],
                "exogenous_scene_signature": exogenous_scene_signature,
                "continuous_action_splits": dict(actor.continuous_action_splits),
                "episode_idx": int(episode_idx),
                "avg_total_reward": avg_total_reward,
                "avg_obj_fun": float(np.mean(obj_fun_total)),
                "energy_statistics": summarize_energy(action_traces),
                "avg_bs_reward": episode_reward_bs / len(episode_rewards_total),
                "all_slots_success": bool(success_slots == madrl_args.total_time_slots),
                "episode_success_bonus_total": episode_success_bonus_total,
                **summarize_sensing_penalty(action_traces),
                "completion_rate": (success_slots / madrl_args.total_time_slots) * 100.0,
                "finite_obj_rate": (finite_obj_slots / len(episode_rewards_total)) * 100.0,
                "no_solution_rate": (no_solution_slots / len(episode_rewards_total)) * 100.0,
                "collision_rate": (collision_slots / len(episode_rewards_total)) * 100.0,
                "spectrum_penalty_rate": (spectrum_penalty_slots / len(episode_rewards_total)) * 100.0,
                "solver_status_counts": solver_status_counts,
                "action_min": action_min,
                "action_max": action_max,
                "state_norm_count": int(len(episode_rewards_total)),
                "state_norm_mean": state_norm_mean.copy(),
                "state_norm_std": state_norm_std.copy(),
                # ── 探针字段：本 episode 实际使用的参数版本 / 指纹 ──
                "weights_version": current_weights_version,
                "weights_checksum": current_weights_checksum,
                "action_probability_metadata": actor.action_probability_metadata(),
            }

            if getattr(madrl_args, "log_detail_level", "compact") == "compact":
                sample_result = compact_episode_payload(sample_result)
                action_traces = sample_result["action_traces"]
                diagnostic_traces = sample_result["diagnostic_traces"]
            # 有界队列：队列满时阻塞在此（背压 => staleness 上限 = maxsize 个参数版本）
            if not _put_with_stop(sample_queue, sample_result, stop_event):
                break

            last_collection_round = int(collection_round)
            episode_idx += 1

    except Exception:
        # 异常哨兵：主进程收到后终止训练并抛出，避免其在空队列上永久阻塞
        _put_error_sentinel(sample_queue, traceback.format_exc())
    finally:
        # On a learner-requested stop, an unread MB-sized episode must not keep
        # this process alive while Queue's feeder waits for pipe space.
        # An error without a stop still flushes its sentinel normally.
        if stop_event.is_set() and hasattr(sample_queue, "cancel_join_thread"):
            sample_queue.cancel_join_thread()
