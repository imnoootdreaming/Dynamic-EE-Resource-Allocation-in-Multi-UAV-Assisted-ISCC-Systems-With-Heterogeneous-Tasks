"""场景拓扑：节点位置生成、CU 马尔可夫轨迹、UAV-目标分配调度。

从 `environment/my_env.py` 抽离的、与环境状态机无关的拓扑类计算：

- `generate_pos`：随机部署 UAV / CU / 目标初始位置
- `generate_cu_trajectory`：CU 马尔可夫移动轨迹（含速度状态递推）
- `get_num_target_windows` / `validate_target_schedule_requirements`：窗口数与可行性校验
- `assign_targets_for_window` / `generate_uav_target_schedule`：按窗口的最近邻目标分配
- `print_precomputed_target_schedule`：调度结果打印（保留原调试输出）

实现与数值逻辑完全取自原 `MyEnv` 方法，仅去掉 `self` 依赖（所需参数全部显式传入）。
"""

import numpy as np


def get_num_target_windows(total_time_slots, hold_slots):
    return int(np.ceil(total_time_slots / hold_slots))


def validate_target_schedule_requirements(uavs_num, targets_num, total_time_slots, hold_slots):
    required_target_num = get_num_target_windows(total_time_slots, hold_slots) * uavs_num
    if targets_num != required_target_num:
        raise ValueError(
            "The current sensing schedule requires "
            f"targets_num == ceil(total_time_slots / hold_slots) * uavs_num. "
            f"Got targets_num={targets_num}, required={required_target_num}, "
            f"hold_slots={hold_slots}, total_time_slots={total_time_slots}."
        )
    return required_target_num


def assign_targets_for_window(reference_positions, candidate_target_indices, uavs_num, target_positions):
    """逐个为 UAV 选择最近的候选目标（贪心：全局最近优先，选中后同时移除 UAV 与目标）。"""
    pending_uavs = list(range(uavs_num))
    pending_targets = list(candidate_target_indices)
    assigned_targets = np.full(uavs_num, -1, dtype=np.int64)
    assigned_distances = np.zeros(uavs_num, dtype=np.float32)

    while pending_uavs:
        best_distance = None
        best_uav_idx = None
        best_target_idx = None
        best_target_list_idx = None

        for uav_idx in pending_uavs:
            candidate_targets = np.asarray(pending_targets, dtype=np.int64)
            candidate_positions = target_positions[candidate_targets]
            candidate_distances = np.linalg.norm(candidate_positions - reference_positions[uav_idx], axis=1)
            nearest_local_idx = int(np.argmin(candidate_distances))
            nearest_distance = float(candidate_distances[nearest_local_idx])
            nearest_target_idx = int(candidate_targets[nearest_local_idx])

            if best_distance is None or nearest_distance < best_distance:
                best_distance = nearest_distance
                best_uav_idx = uav_idx
                best_target_idx = nearest_target_idx
                best_target_list_idx = nearest_local_idx

        assigned_targets[best_uav_idx] = best_target_idx
        assigned_distances[best_uav_idx] = float(best_distance)
        pending_uavs.remove(best_uav_idx)
        pending_targets.pop(best_target_list_idx)

    return assigned_targets, assigned_distances


def generate_uav_target_schedule(uavs_num, targets_num, init_uavs_pos, init_targets_pos,
                                 total_time_slots, hold_slots, initial_reference_positions=None):
    """预计算每个时隙的 UAV→目标分配调度及其规划转移距离。"""
    schedule = np.zeros((total_time_slots + 1, uavs_num), dtype=np.int64)
    schedule_distances = np.zeros((total_time_slots + 1, uavs_num), dtype=np.float32)

    if total_time_slots <= 0:
        return schedule, schedule_distances

    validate_target_schedule_requirements(uavs_num, targets_num, total_time_slots, hold_slots)
    remaining_targets = list(range(targets_num))
    if initial_reference_positions is None:
        reference_positions = init_uavs_pos.copy()
    else:
        reference_positions = np.asarray(initial_reference_positions, dtype=float).copy()

    for window_idx in range(get_num_target_windows(total_time_slots, hold_slots)):
        assigned_targets, assigned_distances = assign_targets_for_window(
            reference_positions=reference_positions,
            candidate_target_indices=remaining_targets,
            uavs_num=uavs_num,
            target_positions=init_targets_pos,
        )
        slot_start = window_idx * hold_slots
        slot_end = min(slot_start + hold_slots, total_time_slots)
        schedule[slot_start:slot_end] = assigned_targets
        schedule_distances[slot_start:slot_end] = assigned_distances

        assigned_target_set = set(assigned_targets.tolist())
        remaining_targets = [target_idx for target_idx in remaining_targets if target_idx not in assigned_target_set]
        reference_positions = init_targets_pos[assigned_targets].copy()

    schedule[total_time_slots] = schedule[total_time_slots - 1]
    schedule_distances[total_time_slots] = schedule_distances[total_time_slots - 1]
    return schedule, schedule_distances


def print_precomputed_target_schedule(uavs_num, schedule, schedule_distances, hold_slots, total_time_slots):
    print("==================================================")
    print("------------ [Precomputed Target Schedule] ------------")
    for t in range(total_time_slots):
        window_idx = t // hold_slots
        window_start = window_idx * hold_slots + 1
        window_end = min((window_idx + 1) * hold_slots, total_time_slots)
        assignment_parts = []
        for uav_idx in range(uavs_num):
            target_idx = int(schedule[t, uav_idx])
            transition_distance = float(schedule_distances[t, uav_idx])
            assignment_parts.append(
                f"UAV-{uav_idx}->Target-{target_idx} (planned_transition_distance={transition_distance:.4f} m)"
            )
        print(
            f"Time Slot {t + 1:02d}/{total_time_slots} | "
            f"Window {window_idx + 1} ({window_start}-{window_end}) | "
            + " | ".join(assignment_parts)
        )
    print("==================================================")


def generate_pos(uavs_num, cus_num, targets_num, center, radius, uav_height):
    """随机部署 CU / 目标（圆内均匀），并令 UAV 初始位于被分配目标的水平投影处。"""
    r_cu = radius * np.sqrt(np.random.rand(cus_num))
    theta_cu = np.random.rand(cus_num) * 2 * np.pi
    cus_pos = np.zeros((cus_num, 3))
    cus_pos[:, 0] = center[0] + r_cu * np.cos(theta_cu)
    cus_pos[:, 1] = center[1] + r_cu * np.sin(theta_cu)

    r_target = radius * np.sqrt(np.random.rand(targets_num))
    theta_target = np.random.rand(targets_num) * 2 * np.pi
    targets_pos = np.zeros((targets_num, 3))
    targets_pos[:, 0] = center[0] + r_target * np.cos(theta_target)
    targets_pos[:, 1] = center[1] + r_target * np.sin(theta_target)

    initial_target_indices = np.random.choice(
        targets_num,
        size=uavs_num,
        replace=targets_num < uavs_num
    )
    uavs_pos = np.zeros((uavs_num, 3))
    uavs_pos[:, :2] = targets_pos[initial_target_indices, :2]
    uavs_pos[:, 2] = uav_height

    return uavs_pos, cus_pos, targets_pos


def generate_cu_trajectory(init_cus_pos, cus_num, total_time_slots,
                           markov_velocity, markov_memory_level,
                           markov_asymptotic_mean_of_velocity,
                           markov_standard_deviation_of_velocity,
                           time_slot_duration, seed):
    """按一阶马尔可夫速度模型预计算 CU 轨迹（含 t=0 初始位置）。"""
    traj = np.zeros((total_time_slots + 1, cus_num, 3), dtype=float)
    velocities = np.zeros((total_time_slots + 1, cus_num, 2), dtype=float)

    if init_cus_pos.shape[0] > 0:
        traj[0] = init_cus_pos.copy()

    v_init = np.array(markov_velocity)[:2]
    velocities[0] = np.tile(v_init, (cus_num, 1))
    rng = np.random.default_rng(seed)

    for t in range(total_time_slots):
        random_component = rng.normal(size=(cus_num, 2))
        v_bar = np.array(markov_asymptotic_mean_of_velocity)[:2]
        velocities[t + 1] = (
            markov_memory_level * velocities[t]
            + (1 - markov_memory_level) * v_bar
            + np.sqrt(1 - markov_memory_level ** 2)
            * markov_standard_deviation_of_velocity
            * random_component
        )
        traj[t + 1, :, :2] = traj[t, :, :2] + velocities[t] * time_slot_duration
        traj[t + 1, :, 2] = traj[t, :, 2]

    return traj
