"""场景拓扑：扇区分区部署、CU 马尔可夫轨迹、扇区内链式目标调度。

从 `environment/my_env.py` 抽离的、与环境状态机无关的拓扑类计算：

- `sector_of` / `sector_centroid` / `sample_in_sector`：等角扇区划分工具
  （对齐 `src/inner/environment.py` 的分区责任制语义）
- `generate_pos`：扇区分区部署——目标按 UAV 数均分到各等角扇区并在扇区内
  均匀采样；UAV 锚定扇区面积质心后悬停本扇区最近目标正上方；CU 全区域随机。
  接受显式 `rng`（`np.random.Generator`）驱动，解除对旧式全局 `np.random`
  的隐式依赖（保证并行进程 / 逐 episode 场景可复现且互异）。
- `generate_cu_trajectory`：CU 马尔可夫移动轨迹（含速度状态递推）；`seed`
  既可传 int 也可直接传 `np.random.Generator`。
- `get_num_target_windows` / `validate_target_schedule_requirements`：窗口数与可行性校验
- `assign_targets_for_window` / `generate_uav_target_schedule`：按窗口的
  **扇区内链式最近邻**目标分配——每个 UAV 只感知本扇区目标，参考点从 UAV
  初始位置起链式推进到上一窗口已感知目标，感知序列完全落在本扇区内。
- `print_precomputed_target_schedule`：调度结果打印（仅 debug 模式调用）
"""

import warnings

import numpy as np


# ── 扇区划分工具（对齐 src/inner/environment.py 的分区责任制语义） ─────────────


def sector_of(positions, center, uavs_num):
    """把 (M, 3) 位置按 x-y 平面极角映射到 [0, uavs_num) 的扇区编号（等角划分）。

    以 `center` 为圆心、极角等分 `uavs_num` 份；z 分量不参与划分。
    """
    positions = np.asarray(positions, dtype=float)
    dx = positions[:, 0] - float(center[0])
    dy = positions[:, 1] - float(center[1])
    angle = np.mod(np.arctan2(dy, dx), 2.0 * np.pi)
    return np.minimum((angle / (2.0 * np.pi / uavs_num)).astype(int), uavs_num - 1)


def sector_centroid(sector, center, radius, uav_height, uavs_num):
    """第 `sector` 个扇区的面积质心位置 (3,)。

    半径 R、半角 alpha = pi / uavs_num 的圆盘扇区，面积质心距圆心 2R sin(alpha) / (3 alpha)。
    """
    half_width = np.pi / uavs_num
    center_angle = (sector + 0.5) * 2.0 * np.pi / uavs_num
    distance = 2.0 * radius * np.sin(half_width) / (3.0 * half_width)
    return np.array([
        float(center[0]) + distance * np.cos(center_angle),
        float(center[1]) + distance * np.sin(center_angle),
        float(uav_height),
    ])


def sample_in_sector(rng, count, sector, center, radius, uavs_num, height):
    """在第 `sector` 个扇区内均匀采样 `count` 个点，返回 (count, 3)。

    径向 r = R·sqrt(U) 保证面积均匀；角度落在扇区中心角 ± 半角内。
    """
    half_width = np.pi / uavs_num
    center_angle = (sector + 0.5) * 2.0 * np.pi / uavs_num
    radius_samples = radius * np.sqrt(rng.random(count))
    angle = center_angle + rng.uniform(-half_width, half_width, count)
    pos = np.zeros((count, 3))
    pos[:, 0] = float(center[0]) + radius_samples * np.cos(angle)
    pos[:, 1] = float(center[1]) + radius_samples * np.sin(angle)
    pos[:, 2] = height
    return pos


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


def assign_targets_for_window(reference_positions, candidate_target_indices, uavs_num,
                              target_positions, uav_sector=None, target_sector=None):
    """为每个 UAV 在候选目标中选择其**扇区内**最近的一个（分区责任制）。

    - 给定 `uav_sector` / `target_sector` 时：UAV i 只在 `target_sector == uav_sector[i]`
      的剩余候选中挑最近目标（扇区内链式最近邻，感知序列不跨区）。
    - 未给定扇区信息时：退化为全局剩余候选中的最近邻（兼容旧调用）。
    - 扇区候选耗尽时回退到全局剩余候选（极端参数下的保护，正常参数
      K == ceil(T/hold)·I 且 K % I == 0 时不会触发）。
    """
    pending_targets = list(candidate_target_indices)
    assigned_targets = np.full(uavs_num, -1, dtype=np.int64)
    assigned_distances = np.zeros(uavs_num, dtype=np.float32)

    for uav_idx in range(uavs_num):
        candidates = pending_targets
        fell_back = False
        if uav_sector is not None and target_sector is not None:
            candidates = [t for t in pending_targets
                          if int(target_sector[t]) == int(uav_sector[uav_idx])]
            if not candidates:
                candidates = pending_targets
                fell_back = True
        if not candidates:
            raise ValueError(
                f"窗口目标分配失败：UAV-{uav_idx} 无任何剩余候选目标"
                f"（剩余 {len(pending_targets)} 个）。"
            )

        candidate_arr = np.asarray(candidates, dtype=np.int64)
        candidate_positions = target_positions[candidate_arr]
        candidate_distances = np.linalg.norm(
            candidate_positions - reference_positions[uav_idx], axis=1
        )
        nearest_local_idx = int(np.argmin(candidate_distances))
        nearest_target_idx = int(candidate_arr[nearest_local_idx])

        assigned_targets[uav_idx] = nearest_target_idx
        assigned_distances[uav_idx] = float(candidate_distances[nearest_local_idx])
        pending_targets.remove(nearest_target_idx)
        if fell_back:
            warnings.warn(
                f"UAV-{uav_idx} 所在扇区目标已耗尽，回退全局剩余候选（分区参数不闭合，"
                f"建议检查 targets_num / uavs_num / hold_slots 的配比）。"
            )

    return assigned_targets, assigned_distances


def generate_uav_target_schedule(uavs_num, targets_num, init_uavs_pos, init_targets_pos,
                                 total_time_slots, hold_slots,
                                 center=(0.0, 0.0), target_sector=None,
                                 initial_reference_positions=None):
    """预计算每个时隙的 UAV→目标分配调度及其规划转移距离（扇区内链式）。

    :param center: 部署区域中心（x, y），用于 `target_sector` 未给定时的扇区推导。
    :param target_sector: 各目标的扇区编号 (targets_num,)；None 时由
        `sector_of(init_targets_pos, center, uavs_num)` 推导。UAV 扇区由
        `sector_of(init_uavs_pos, ...)` 推导，保证每架 UAV 只感知本扇区目标。
    """
    schedule = np.zeros((total_time_slots + 1, uavs_num), dtype=np.int64)
    schedule_distances = np.zeros((total_time_slots + 1, uavs_num), dtype=np.float32)

    if total_time_slots <= 0:
        return schedule, schedule_distances

    validate_target_schedule_requirements(uavs_num, targets_num, total_time_slots, hold_slots)
    if target_sector is None:
        target_sector = sector_of(init_targets_pos, center, uavs_num)
    else:
        target_sector = np.asarray(target_sector, dtype=int)
    uav_sector = sector_of(init_uavs_pos, center, uavs_num)

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
            uav_sector=uav_sector,
            target_sector=target_sector,
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


def generate_pos(uavs_num, cus_num, targets_num, center, radius, uav_height, rng=None):
    """扇区分区部署（对齐 inner 语义）：目标均分到 I 个等角扇区，UAV 各负责一个扇区。

    - CU：全区域圆盘均匀（r = R·sqrt(U)），z = 0；
    - 目标：`counts = [K // I] * I`（余数补给前几个扇区），在各扇区内均匀采样，z = 0；
    - UAV：锚定各扇区面积质心，随后在本扇区目标中取离质心最近者，
      最终悬停在该目标正上方（水平坐标重合），z = uav_height。

    :param rng: 显式随机源 `np.random.Generator`；None 时回退旧式全局 `np.random`
        （仅为兼容无 rng 的旧调用，并行采样 / 逐 episode 重生成场景必须显式传入）。
    :return: (uavs_pos, cus_pos, targets_pos, target_sector)，
        其中 target_sector 为各目标的扇区编号 (targets_num,)，供调度复用。
    """
    if rng is None:
        rng = np.random

    # ── CU：全区域圆盘均匀 ────────────────────────────────────────────────
    r_cu = radius * np.sqrt(rng.random(cus_num))
    theta_cu = rng.random(cus_num) * 2.0 * np.pi
    cus_pos = np.zeros((cus_num, 3))
    cus_pos[:, 0] = center[0] + r_cu * np.cos(theta_cu)
    cus_pos[:, 1] = center[1] + r_cu * np.sin(theta_cu)

    # ── 目标：均分到各扇区并在扇区内均匀采样 ──────────────────────────────
    counts = [targets_num // uavs_num] * uavs_num
    for i in range(targets_num % uavs_num):
        counts[i] += 1
    target_blocks = [
        sample_in_sector(rng, counts[i], i, center, radius, uavs_num, 0.0)
        for i in range(uavs_num)
    ]
    targets_pos = np.vstack(target_blocks)
    # 各目标的扇区编号：第 i 块连续 count_i 个目标属于扇区 i（拼接顺序即扇区顺序）
    target_sector = np.concatenate([np.full(counts[i], i, dtype=int) for i in range(uavs_num)])

    # ── UAV：锚定扇区质心 → 本扇区最近目标 → 悬停其正上方 ─────────────────
    anchor_pos = np.vstack([
        sector_centroid(i, center, radius, uav_height, uavs_num) for i in range(uavs_num)
    ])
    uavs_pos = np.zeros((uavs_num, 3))
    used_target_indices = set()
    for i in range(uavs_num):
        sector_target_indices = np.flatnonzero(target_sector == i)
        # 扇区内无目标时的保护：退化为全局目标（极端参数，正常 K >= I 不触发）
        if sector_target_indices.size == 0:
            sector_target_indices = np.arange(targets_num)
        candidate_positions = targets_pos[sector_target_indices]
        distances = np.linalg.norm(
            candidate_positions[:, :2] - anchor_pos[i, :2], axis=1
        )
        # 已被其他 UAV 悬停占用的目标跳过，避免同扇区多机重叠
        available = [k for k in range(sector_target_indices.size)
                     if int(sector_target_indices[k]) not in used_target_indices]
        if not available:
            available = list(range(sector_target_indices.size))
        nearest_local = available[int(np.argmin(distances[available]))]
        nearest_global = int(sector_target_indices[nearest_local])
        uavs_pos[i, :2] = targets_pos[nearest_global, :2]
        uavs_pos[i, 2] = uav_height
        used_target_indices.add(nearest_global)

    return uavs_pos, cus_pos, targets_pos, target_sector


def generate_cu_trajectory(init_cus_pos, cus_num, total_time_slots,
                           markov_velocity, markov_memory_level,
                           markov_asymptotic_mean_of_velocity,
                           markov_standard_deviation_of_velocity,
                           time_slot_duration, seed):
    """按一阶马尔可夫速度模型预计算 CU 轨迹（含 t=0 初始位置）。

    :param seed: int 种子或现成的 `np.random.Generator`；传 Generator 时直接复用
        （供逐 episode 场景重生成时注入已派生的独立随机流）。
    """
    traj = np.zeros((total_time_slots + 1, cus_num, 3), dtype=float)
    velocities = np.zeros((total_time_slots + 1, cus_num, 2), dtype=float)

    if init_cus_pos.shape[0] > 0:
        traj[0] = init_cus_pos.copy()

    v_init = np.array(markov_velocity)[:2]
    velocities[0] = np.tile(v_init, (cus_num, 1))
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)

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
