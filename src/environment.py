"""系统状态与外层给定量（论文 Section \ref{sec: System Description} / \ref{sec: Problem Formulation}）。

说明：内层问题 P1 需要外层给定 g_i(t)、q_{u_i}(t)、D_{u_i}^{off}(t)、η_{i,j}(t)、p_j(t)。
当前外层（MHBPPO）尚未接入，这些量在此以随机方式给出；后续只需把本文件中的
`build_outer_variables` 换成外层智能体的输出即可，其余模块无需改动。
"""

from dataclasses import dataclass

import numpy as np


def _steering_vector(sin_phi, antenna_nums, d_over_lambda):
    """阵列响应向量 [1, e^{j2π(𝔡/λ)sinφ}, ..., e^{j2π(𝔡/λ)(N-1)sinφ}]。"""
    n_range = np.arange(antenna_nums)
    return np.exp(1j * 2.0 * np.pi * d_over_lambda * sin_phi * n_range)


def _rician_mimo(pos_single, pos_array, alpha, params, rng):
    """单天线节点 <-> N 天线节点之间的 Rician 信道 h ∈ C^{1×N}。"""
    diff = pos_array - pos_single
    dist = np.linalg.norm(diff)
    sin_phi = diff[1] / dist
    los = _steering_vector(sin_phi, params.N, params.d_over_lambda)
    nlos = (rng.standard_normal(params.N) + 1j * rng.standard_normal(params.N)) / np.sqrt(2.0)
    return np.sqrt(params.rho_ref * dist ** (-alpha)) * (
        np.sqrt(params.kappa_rician / (1.0 + params.kappa_rician)) * los
        + np.sqrt(1.0 / (1.0 + params.kappa_rician)) * nlos
    )


def _rician_siso(pos_a, pos_b, alpha, params, rng):
    """单天线节点之间的 Rician 信道 h ∈ C^{1×1}。"""
    dist = np.linalg.norm(pos_b - pos_a)
    nlos = (rng.standard_normal() + 1j * rng.standard_normal()) / np.sqrt(2.0)
    return np.sqrt(params.rho_ref * dist ** (-alpha)) * (
        np.sqrt(params.kappa_rician / (1.0 + params.kappa_rician)) * 1.0
        + np.sqrt(1.0 / (1.0 + params.kappa_rician)) * nlos
    )


def _sector_of(positions, params):
    """把 (M, 3) 位置按 x-y 平面极角映射到 [0, I) 的扇区编号（等角划分）。"""
    dx = positions[:, 0] - params.bs_pos[0]
    dy = positions[:, 1] - params.bs_pos[1]
    angle = np.mod(np.arctan2(dy, dx), 2.0 * np.pi)
    return np.minimum((angle / (2.0 * np.pi / params.I)).astype(int), params.I - 1)


def _sector_centroid(sector, params):
    """第 sector 个扇区的面积质心位置 (3,)。

    半径 R、半角 alpha = pi / I 的圆盘扇区，面积质心距圆心 2R sin(alpha) / (3 alpha)。
    """
    half_width = np.pi / params.I
    center_angle = (sector + 0.5) * 2.0 * np.pi / params.I
    distance = 2.0 * params.deploy_radius * np.sin(half_width) / (3.0 * half_width)
    return np.array([
        params.bs_pos[0] + distance * np.cos(center_angle),
        params.bs_pos[1] + distance * np.sin(center_angle),
        params.uav_height,
    ])


def _sample_in_sector(rng, count, sector, params, height):
    """在第 sector 个扇区内均匀采样 count 个点，返回 (count, 3)。"""
    half_width = np.pi / params.I
    center_angle = (sector + 0.5) * 2.0 * np.pi / params.I
    radius = params.deploy_radius * np.sqrt(rng.random(count))
    angle = center_angle + rng.uniform(-half_width, half_width, count)
    pos = np.zeros((count, 3))
    pos[:, 0] = params.bs_pos[0] + radius * np.cos(angle)
    pos[:, 1] = params.bs_pos[1] + radius * np.sin(angle)
    pos[:, 2] = height
    return pos


def _generate_positions(params, rng):
    """按扇区生成位置：CU 全区域随机；UAV 每扇区质心一个；目标均分到各扇区。"""
    def sample_in_disk(num, height):
        radius = params.deploy_radius * np.sqrt(rng.random(num))
        theta = rng.random(num) * 2.0 * np.pi
        pos = np.zeros((num, 3))
        pos[:, 0] = params.bs_pos[0] + radius * np.cos(theta)
        pos[:, 1] = params.bs_pos[1] + radius * np.sin(theta)
        pos[:, 2] = height
        return pos

    q_cu_pos = sample_in_disk(params.J, 0.0)          # CU：全区域随机（不变）

    counts = [params.K // params.I] * params.I        # 目标均分，余数补给前几个扇区
    for i in range(params.K % params.I):
        counts[i] += 1

    q_uav_pos = np.vstack([_sector_centroid(i, params) for i in range(params.I)])
    q_target_pos = np.vstack([_sample_in_sector(rng, counts[i], i, params, 0.0)
                              for i in range(params.I)])
    return q_uav_pos, q_cu_pos, q_target_pos


def _assign_nearest_targets(q_uav_pos, q_target_pos, params):
    """每架 UAV 只在其所在扇区的目标中挑最近的一个作为指定感知目标。"""
    target_sector = _sector_of(q_target_pos, params)
    uav_sector = _sector_of(q_uav_pos, params)
    assigned = np.zeros(params.I, dtype=int)
    for i in range(params.I):
        candidates = np.flatnonzero(target_sector == uav_sector[i])
        if candidates.size == 0:        # 该扇区无目标时退化为全局最近（极端情形保护）
            candidates = np.arange(q_target_pos.shape[0])
        dist = np.linalg.norm(q_target_pos[candidates] - q_uav_pos[i], axis=1)
        assigned[i] = candidates[np.argmin(dist)]
    return assigned


def build_outer_variables(params, rng):
    """外层（MHBPPO）给定量，当前以随机方式生成。

    :return: g_rec_beam (I, N)、eta_share (I, J)、p_cu_power (J,)、
             D_uav_off (I,)、q_uav_pos_next (I, 3)
    """
    # g_i(t)：接收波束成形向量，满足 ‖g_i(t)‖^2 = 1
    g_rec_beam = (rng.standard_normal((params.I, params.N))
                  + 1j * rng.standard_normal((params.I, params.N))) / np.sqrt(2.0)
    g_rec_beam /= np.linalg.norm(g_rec_beam, axis=1, keepdims=True)

    # η_{i,j}(t)：每个 UAV 独占一个 CU 的频谱（满足 Σ_j η_{i,j} = 1 且 Σ_i η_{i,j} ≤ 1）
    eta_share = np.zeros((params.I, params.J))
    cu_indices = rng.permutation(params.J)[:params.I]
    eta_share[np.arange(params.I), cu_indices] = 1.0

    # p_j(t)：CU 发射功率，取值 (0, P^max_CU]
    p_cu_power = rng.uniform(0.1 * params.P_max_cu, params.P_max_cu, params.J)

    # D_{u_i}^{off}(t)：感知任务卸载时长，需满足 D̄^sen + D_{u_i}^{off} < D^sen_max
    D_uav_off = rng.uniform(0.01, params.D_max_sen - params.D_bar_sen - 0.01, params.I)

    # q_{u_i}(t + 1)：下一时隙位置，速度方向随机、速率在 [V_min, 2·V_min] 内
    # 说明：内层 P5 只求解单个时隙，与 UAV 之后的飞行位置无关（E_{u_i}^{fly} 是常数不参与内层决策），因此这里把速度限制在较小范围（5~10 m/s），
    # 避免飞行能耗在目标函数中占绝对主导、掩盖真正与决策相关的能耗项。
    speed = rng.uniform(params.uav_min_speed, 5.0 * params.uav_min_speed, params.I)
    direction_xy = rng.standard_normal((params.I, 2))
    direction_xy /= np.linalg.norm(direction_xy, axis=1, keepdims=True)
    q_uav_pos_step = np.zeros((params.I, 3))
    q_uav_pos_step[:, :2] = direction_xy * speed[:, None] * params.tau_slot  # z 方向位移为 0（定高飞行）

    return g_rec_beam, eta_share, p_cu_power, D_uav_off, q_uav_pos_step


@dataclass
class Environment:
    """系统状态：位置、信道、感知矩阵，以及由外层给定量派生的常数。"""

    # ── 位置 ──────────────────────────────────────────────────────────────
    q_uav_pos: np.ndarray            # q_{u_i}(t)        (I, 3)
    q_uav_pos_next: np.ndarray       # q_{u_i}(t + 1)    (I, 3)
    q_cu_pos: np.ndarray             # q_{c_j}(t)        (J, 3)
    q_target_pos: np.ndarray         # 所有目标位置       (K, 3)
    q_target_designated: np.ndarray  # UAV 指定感知目标   (I, 3)

    # ── 信道 ──────────────────────────────────────────────────────────────
    h_cu_2_uav: np.ndarray           # h_{c_j,u_i}(t) ∈ C^{1×N}   (I, J, N)
    h_uav_2_bs: np.ndarray           # h_{u_i,BS}(t) ∈ C^{N×1}    (I, N)
    h_cu_2_bs: np.ndarray            # h_{c_j,BS}(t) ∈ C^{1×1}    (J,)
    A_theta: np.ndarray              # A(θ_i(t)) ∈ C^{N×N}        (I, N, N)

    # ── 外层给定量（当前随机生成） ────────────────────────────────────────
    g_rec_beam: np.ndarray           # g_i(t)            (I, N)
    eta_share: np.ndarray            # η_{i,j}(t)        (I, J)
    p_cu_power: np.ndarray           # p_j(t)            (J,)
    D_uav_off: np.ndarray            # D_{u_i}^{off}(t)  (I,)

    # ── 由上述量派生的常数（论文 P1 中的简记） ────────────────────────────
    Gamma_sinr: np.ndarray           # Γ_i(t)            (I,)
    Phi_off_inr: np.ndarray          # Φ_i(t)            (I,)
    Theta_cu_time: np.ndarray        # Θ_j(t)            (J,)
    upsilon_cu_rate: np.ndarray      # υ_j(t)            (J,)
    G_sen_corr: np.ndarray           # G_i(t) = A^H g g^H A      (I, N, N)
    H_uav_bs: np.ndarray             # H_{u_i,BS}(t) = h h^H     (I, N, N)
    E_uav_fly: np.ndarray            # E_{u_i}^{fly}(t)          (I,)
    d_uav_target: np.ndarray         # d_i(t)，UAV 到指定目标距离 (I,)


def build_environment(params, rng, outer_variables=None):
    """生成系统状态与派生常数。

    :param outer_variables: 可选，外层给定量的元组
        (g_rec_beam, eta_share, p_cu_power, D_uav_off, q_uav_pos_step)。
        若为 None，则调用 build_outer_variables(rng) 随机生成（默认行为）；
        若为注入值，则直接使用，便于复现确定场景（如从 CSV 读取的样本）。
    """
    bs_pos = np.asarray(params.bs_pos, dtype=float)
    q_uav_pos, q_cu_pos, q_target_pos = _generate_positions(params, rng)

    # ── 每个 UAV 的指定感知目标 ──────────────────────────────────────────
    # （_assign_nearest_targets 用扇区质心作为归属锚点确定各 UAV 的扇区，再在本区目标中取最近）
    target_idx = _assign_nearest_targets(q_uav_pos, q_target_pos, params)
    q_target_designated = q_target_pos[target_idx]

    # ── 初始时 UAV 悬停在指定感知目标的正上方（水平坐标与目标重合） ──────
    q_uav_pos = q_uav_pos.copy()
    q_uav_pos[:, :2] = q_target_designated[:, :2]

    # ── 感知信道矩阵 A(θ_i(t)) = sqrt(ξ_0 ρ d_i^{-4}) a_r(θ_i) a_t^H(θ_i) ──
    d_uav_target = np.linalg.norm(q_uav_pos - q_target_designated, axis=1)
    diff_uav_target = q_target_designated - q_uav_pos
    sin_theta = diff_uav_target[:, 1] / d_uav_target
    A_theta = np.zeros((params.I, params.N, params.N), dtype=complex)
    for i in range(params.I):
        a_vec = _steering_vector(sin_theta[i], params.N, params.d_over_lambda)
        path_gain = np.sqrt(params.xi_0 * params.rho_ref * d_uav_target[i] ** (-4.0))
        A_theta[i] = path_gain * np.outer(a_vec, a_vec.conj())

    # ── 通信信道（Rician 衰落） ──────────────────────────────────────────
    h_cu_2_uav = np.zeros((params.I, params.J, params.N), dtype=complex)
    for i in range(params.I):
        for j in range(params.J):
            h_cu_2_uav[i, j] = _rician_mimo(q_cu_pos[j], q_uav_pos[i], params.alpha_1, params, rng)

    h_uav_2_bs = np.zeros((params.I, params.N), dtype=complex)
    for i in range(params.I):
        h_uav_2_bs[i] = _rician_mimo(bs_pos, q_uav_pos[i], params.alpha_2, params, rng)

    h_cu_2_bs = np.zeros(params.J, dtype=complex)
    for j in range(params.J):
        h_cu_2_bs[j] = _rician_siso(q_cu_pos[j], bs_pos, params.alpha_3, params, rng)

    # ── 外层给定量（当前随机生成；允许注入确定值以复现场景） ─────────────
    if outer_variables is None:
        g_rec_beam, eta_share, p_cu_power, D_uav_off, q_uav_pos_step = build_outer_variables(params, rng)
    else:
        (g_rec_beam, eta_share, p_cu_power,
         D_uav_off, q_uav_pos_step) = outer_variables
    q_uav_pos_next = q_uav_pos + q_uav_pos_step

    # ── 派生常数 ─────────────────────────────────────────────────────────
    # Γ_i(t) = Σ_j η_{i,j} p_j |g_i^H h_{c_j,u_i}^H|^2 + σ^2
    h_cu_2_uav_sq = np.zeros((params.I, params.J))
    for i in range(params.I):
        for j in range(params.J):
            h_cu_2_uav_sq[i, j] = abs(np.dot(g_rec_beam[i], h_cu_2_uav[i, j])) ** 2
    Gamma_sinr = np.sum(eta_share * p_cu_power[None, :] * h_cu_2_uav_sq, axis=1) + params.sigma_2

    # Φ_i(t) = Σ_j η_{i,j} p_j |h_{c_j,BS}|^2 + σ^2
    h_cu_2_bs_sq = np.abs(h_cu_2_bs) ** 2
    Phi_off_inr = eta_share @ (p_cu_power * h_cu_2_bs_sq) + params.sigma_2

    # Θ_j(t) = Σ_i η_{i,j} D̄^sen + Σ_i η_{i,j} D_{u_i}^{off}
    Theta_cu_time = params.D_bar_sen * np.sum(eta_share, axis=0) + eta_share.T @ D_uav_off

    # υ_j(t) = log_2(1 + p_j |h_{c_j,BS}|^2 / σ^2)
    upsilon_cu_rate = np.log2(1.0 + p_cu_power * h_cu_2_bs_sq / params.sigma_2)

    # G_i(t) = A^H(θ_i) g_i g_i^H A(θ_i)
    G_sen_corr = np.zeros((params.I, params.N, params.N), dtype=complex)
    for i in range(params.I):
        A_g = A_theta[i].conj().T @ g_rec_beam[i]
        G_sen_corr[i] = np.outer(A_g, A_g.conj())

    # H_{u_i,BS}(t) = h_{u_i,BS} h_{u_i,BS}^H
    H_uav_bs = np.zeros((params.I, params.N, params.N), dtype=complex)
    for i in range(params.I):
        H_uav_bs[i] = np.outer(h_uav_2_bs[i], h_uav_2_bs[i].conj())

    # E_{u_i}^{fly}(t) = ϱ_1 ‖q(t+1) - q(t)‖^3 / τ^2 + ϱ_2 τ^2 / ‖q(t+1) - q(t)‖
    displacement = np.linalg.norm(q_uav_pos_next - q_uav_pos, axis=1)
    E_uav_fly = (params.varrho_1 * displacement ** 3 / params.tau_slot ** 2
                 + params.varrho_2 * params.tau_slot ** 2 / displacement)

    return Environment(
        q_uav_pos=q_uav_pos,
        q_uav_pos_next=q_uav_pos_next,
        q_cu_pos=q_cu_pos,
        q_target_pos=q_target_pos,
        q_target_designated=q_target_designated,
        h_cu_2_uav=h_cu_2_uav,
        h_uav_2_bs=h_uav_2_bs,
        h_cu_2_bs=h_cu_2_bs,
        A_theta=A_theta,
        g_rec_beam=g_rec_beam,
        eta_share=eta_share,
        p_cu_power=p_cu_power,
        D_uav_off=D_uav_off,
        Gamma_sinr=Gamma_sinr,
        Phi_off_inr=Phi_off_inr,
        Theta_cu_time=Theta_cu_time,
        upsilon_cu_rate=upsilon_cu_rate,
        G_sen_corr=G_sen_corr,
        H_uav_bs=H_uav_bs,
        E_uav_fly=E_uav_fly,
        d_uav_target=d_uav_target,
    )
