"""动作/索引 → 匹配矩阵、接收波束与信道扁平化等映射工具。

从 `environment/my_env.py` 抽离的、与环境状态机无关的纯函数：

- `build_uavs_cus_matched_matrix`：离散 CU 索引动作 → UAV-CU 匹配矩阵 eta
- `build_uav_targets_matched_matrix`：UAV→目标索引 → UAV-目标匹配矩阵
- `build_unit_norm_rec_beam`：实/虚部方向 → 逐 UAV 单位范数接收波束 g（‖g_i‖²=1）
- `flatten_complex`：复数信道 → 拼接的实/虚部 float32 向量（观测用）

实现与数值逻辑完全取自原 `MyEnv` 方法，仅去掉 `self` 依赖（所需维度参数显式传入）。
"""

import numpy as np


def build_uavs_cus_matched_matrix(discrete_actions, uavs_num, cus_num):
    """将离散 CU 索引动作转换为 UAV-CU 匹配矩阵（每 UAV 行 one-hot）。"""
    discrete_actions = np.clip(np.asarray(discrete_actions, dtype=np.int64), 0, cus_num - 1)
    uavs_cus_matched_matrix = np.zeros((uavs_num, cus_num), dtype=np.float32)
    for uav_idx, cu_idx in enumerate(discrete_actions):
        uavs_cus_matched_matrix[uav_idx, int(cu_idx)] = 1.0
    return uavs_cus_matched_matrix


def build_uav_targets_matched_matrix(target_indices, uavs_num, targets_num):
    """将 UAV→目标索引转换为 UAV-目标匹配矩阵（每 UAV 行 one-hot）。"""
    matched_matrix = np.zeros((uavs_num, targets_num), dtype=np.float32)
    clipped_target_indices = np.clip(
        np.asarray(target_indices, dtype=np.int64),
        0,
        targets_num - 1
    )
    for i, target_idx in enumerate(clipped_target_indices):
        matched_matrix[i, target_idx] = 1.0
    return matched_matrix


def build_unit_norm_rec_beam(dir_real_flat, dir_imag_flat, uavs_num, antenna_nums):
    """由实/虚部方向构造单位范数接收波束 g_i（自动满足 ‖g_i‖²=1）。

    :param dir_real_flat: (I*N,) 接收波束方向实部（raw，将被逐 UAV L2 归一化）
    :param dir_imag_flat: (I*N,) 接收波束方向虚部
    :return: (I, N) complex ndarray，逐 UAV 单位范数
    """
    beams = np.zeros((uavs_num, antenna_nums), dtype=complex)
    for i in range(uavs_num):
        start, end = i * antenna_nums, (i + 1) * antenna_nums
        direction = dir_real_flat[start:end] + 1j * dir_imag_flat[start:end]
        norm = float(np.linalg.norm(direction))
        if norm < 1e-12:  # 退化保护：方向近零时回退基向量 e_1
            direction = np.zeros(antenna_nums, dtype=complex)
            direction[0] = 1.0 + 0j
            norm = 1.0
        beams[i] = direction / norm
    return beams


def flatten_complex(channel):
    """复数信道 → 实部/虚部顺序拼接的 float32 向量。"""
    return np.concatenate([channel.real.flatten(), channel.imag.flatten()]).astype(np.float32)
