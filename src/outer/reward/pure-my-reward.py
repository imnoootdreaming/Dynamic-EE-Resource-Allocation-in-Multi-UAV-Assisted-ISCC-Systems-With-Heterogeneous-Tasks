"""
pure-my-reward.py
~~~~~~~~~~~~~~~~~
纯端到端 PPO 的 reward 模块：不依赖 CCCP 内层优化，直接根据 P0 原始优化变量
计算加权总能耗 E_sum 并数值检验全部 15 条 P0 约束。

惩罚策略（与 my_reward.py 对齐）：
- UAV 越界 / UAV 碰撞 / BS 频谱过分配：保留与 my_reward.py 完全相同的二进制/连续值惩罚
- no_solution_penalty：仅占位兼容，恒为 0
- 其余 12 条约束：tanh(violation_magnitude / natural_scale) → 按项数归一化到 (0, 1)
"""

import numpy as np


def dbm_2_watt(dbm):
    return 10 ** ((dbm - 30) / 10)


def db_2_watt(db):
    return 10 ** (db / 10)


def _extract_matched_sensing_channel(uavs_targets_matched_matrix, uavs_2_targets_channels):
    """
    根据匹配矩阵提取每个 UAV 对应目标的感知信道矩阵 A_i ∈ C^{N×N}。

    :param uavs_targets_matched_matrix: (I, K) 二元匹配矩阵
    :param uavs_2_targets_channels: (I, K, N, N) 感知信道
    :return: list of A_i (I 个，每个 N×N complex)
    """
    row_indices, col_indices = np.where(uavs_targets_matched_matrix == 1)
    sorted_args = np.argsort(row_indices)
    row_indices = row_indices[sorted_args]
    col_indices = col_indices[sorted_args]
    matched_channels = []
    for i, k in zip(row_indices, col_indices):
        matched_channels.append(uavs_2_targets_channels[i, k, :, :])
    return matched_channels


class PureMyReward:
    """
    纯 P0 reward 计算器。

    职责：
    1. 计算 P0 加权总能耗 E_sum
    2. 数值检验全部 15 条 P0 约束
    3. 合成 reward：
       - 与 my_reward.py 一致的惩罚：越界/碰撞/频谱（二进制/连续值） + no_solution（占位 0）
       - 其余约束：tanh(violation / scale) / num_terms 归一化到 (0, 1)
    """

    def __init__(self, base_args):
        self.base_args = base_args
        self.x_max = float(getattr(base_args, "radius", 600))
        self.y_max = float(getattr(base_args, "radius", 600))
        # ── 预计算雷达统计速率常数 ──
        self.xi1 = (
            base_args.radar_duty_ratio / (2.0 * base_args.radar_impulse_duration)
        )
        self.xi2 = (
            2.0
            * base_args.var_range_fluctuation
            * (base_args.radar_spectrum_shape ** 2)
            * (base_args.bandwidth ** 3)
            * base_args.radar_impulse_duration
        )

    # ------------------------------------------------------------------
    #  reward_compute
    # ------------------------------------------------------------------
    def reward_compute(
        self,
        uavs_2_cus_channels,
        uavs_2_bs_channels,
        cus_2_bs_channels,
        uavs_2_targets_channels,
        uavs_targets_matched_matrix,
        uavs_cus_matched_matrix,
        uavs_pos,
        uavs_pos_cur,
        uavs_off_duration,
        cus_off_power,
        cus_entertaining_task_size,
        uav_sen_beam_vectors,
        uav_off_beam_vectors,
        uav_bs_freqs=None,
        cus_off_durations=None,
        cu_bs_freqs=None,
        feasibility_enforce=False,
    ):
        """
        :param uav_sen_beam_vectors:  list of (N,) complex arrays, 感知波束 w_i
        :param uav_off_beam_vectors:  list of (N,) complex arrays, 卸载波束 b_i
        :param uav_bs_freqs:         (I,) float, BS 分给 UAV 的计算频率 f_{u_i} [Hz]
        :param cus_off_durations:    (J,) float, CU 卸载时长 D^off_{c_j} [s]
        :param cu_bs_freqs:          (J,) float, BS 分给 CU 的计算频率 f_{c_j} [Hz]
        :param feasibility_enforce:  测试层后处理——C7/C9/C10/C11/C12 违反时自动修正
        """
        a = self.base_args
        I = a.uavs_num
        J = a.cus_num
        N = a.antenna_nums
        B_w = a.bandwidth
        sigma2 = dbm_2_watt(a.noise_power_density_dbm) * B_w
        epsilon = db_2_watt(a.sen_sinr)  # 感知 SINR 门限（线性）
        D_sen = a.uav_sen_duration
        D_max_u = a.uav_max_delay
        D_max_c = a.cu_max_delay
        P_max_uav = a.uav_max_power
        P_max_cu = dbm_2_watt(a.cu_max_power_dbm)
        F_max = a.bs_max_freq
        C_bit = a.bs_cycles_per_bit
        V_min_tau = a.uav_min_speed * a.time_slot_duration
        V_max_tau = a.uav_max_speed * a.time_slot_duration
        d_min = a.uav_safe_distance
        omega1 = a.omega_weight_1
        omega2 = a.omega_weight_2
        omega3 = a.omega_weight_3
        kappa = a.kappa

        # ── 转类型 ──
        uavs_pos = np.asarray(uavs_pos, dtype=float)
        uavs_pos_cur = np.asarray(uavs_pos_cur, dtype=float)
        uavs_off_duration = np.asarray(uavs_off_duration, dtype=float)
        cus_off_power = np.asarray(cus_off_power, dtype=float)
        cus_off_durations = np.asarray(cus_off_durations, dtype=float)
        _use_closed_form_f = (uav_bs_freqs is None) or (cu_bs_freqs is None)
        if not _use_closed_form_f:
            uav_bs_freqs = np.asarray(uav_bs_freqs, dtype=float)
            cu_bs_freqs = np.asarray(cu_bs_freqs, dtype=float)
        eta = np.asarray(uavs_cus_matched_matrix, dtype=float)
        cus_entertaining_task_size = np.asarray(cus_entertaining_task_size, dtype=float)

        # ── 预计算常用标量 ──
        h_cj_BS_sq = np.array(
            [abs(cus_2_bs_channels[j, 0]) ** 2 for j in range(J)], dtype=float
        )
        # UAV→BS 信道向量 h_{ui,BS} ∈ C^N，来自 uavs_2_bs_channels[i, 0, :]
        h_ui_BS_vecs = [uavs_2_bs_channels[i, 0, :] for i in range(I)]

        # ── 提取每个 UAV 的感知信道矩阵 A_i ──
        matched_A_list = _extract_matched_sensing_channel(
            uavs_targets_matched_matrix, uavs_2_targets_channels
        )

        # ── 每个 UAV 的干扰+噪声协方差 Delta_i ──
        Delta_list = []
        A_iH_Delta_inv_A_i_list = []  # 用于约束 7 和 R_i_sen 的矩阵
        SINR_i_list = []              # w_i^H A_i^H Delta_i^{-1} A_i w_i
        for i in range(I):
            Delta_i = sigma2 * np.eye(N, dtype=complex)
            for j in range(J):
                if eta[i, j] > 0:
                    h_ij = uavs_2_cus_channels[i, j, :]
                    Delta_i += eta[i, j] * cus_off_power[j] * np.outer(h_ij, np.conj(h_ij))
            Delta_list.append(Delta_i)
            # 约束 7 需要的矩阵：A_i^H Delta_i^{-1} A_i
            A_i = matched_A_list[i]
            Delta_inv_A = np.linalg.solve(Delta_i, A_i)
            AH_Dinv_A = A_i.conj().T @ Delta_inv_A
            AH_Dinv_A = (AH_Dinv_A + AH_Dinv_A.conj().T) / 2.0
            A_iH_Delta_inv_A_i_list.append(AH_Dinv_A)

        # R_i_sen 和 SINR
        R_i_sen_list = []
        for i in range(I):
            w_i = uav_sen_beam_vectors[i]
            SINR_i = np.real(
                np.conj(w_i).T @ A_iH_Delta_inv_A_i_list[i] @ w_i
            )
            SINR_i = np.clip(SINR_i, 0.0, None)  # 保证非负
            SINR_i_list.append(float(SINR_i))
            R_i_sen = self.xi1 * np.log2(1.0 + self.xi2 * SINR_i)
            R_i_sen_list.append(float(R_i_sen))

        # ── R_i_off 卸载速率 ──
        R_i_off_list = []
        for i in range(I):
            interf_plus_noise = sigma2
            for j in range(J):
                interf_plus_noise += eta[i, j] * cus_off_power[j] * h_cj_BS_sq[j]
            b_i = uav_off_beam_vectors[i]
            signal = np.abs(np.conj(h_ui_BS_vecs[i]).T @ b_i) ** 2
            R_i_off = B_w * np.log2(1.0 + signal / max(interf_plus_noise, 1e-30))
            R_i_off_list.append(float(R_i_off))

        # ── 干扰项（用于 CU 卸载速率） ──
        # |h_{u_i,BS}^H * w_i|² — UAV i 感知过程对 BS 的干扰
        interf_sen_list = []
        # |h_{u_i,BS}^H * b_i|² — UAV i 卸载过程对 BS 的干扰
        interf_off_list = []
        for i in range(I):
            interf_sen_list.append(
                float(np.abs(np.conj(h_ui_BS_vecs[i]).T @ uav_sen_beam_vectors[i]) ** 2)
            )
            interf_off_list.append(
                float(np.abs(np.conj(h_ui_BS_vecs[i]).T @ uav_off_beam_vectors[i]) ** 2)
            )

        # ══════════════════════════════════════════════════════════════════
        #  [测试层] feasibility enforcement — 违反约束时自动修正动作
        # ══════════════════════════════════════════════════════════════════
        feasibility_log = {"C7_corrected": [], "C9_corrected": [],
                           "C10_corrected": [], "C11_corrected": []}
        if feasibility_enforce:
            # ---- C7: Sensing-SINR —— scale up w_i to meet ε ----
            for i in range(I):
                if SINR_i_list[i] < epsilon and SINR_i_list[i] > 1e-30:
                    scale = float(np.sqrt(epsilon / SINR_i_list[i]))
                    uav_sen_beam_vectors[i] = uav_sen_beam_vectors[i] * scale
                    feasibility_log["C7_corrected"].append(i)

            # ---- C9: UAV Offloading Success —— scale up b_i ----
            for i in range(I):
                req = D_sen * max(R_i_sen_list[i], 1e-8)
                ach = uavs_off_duration[i] * R_i_off_list[i]
                if ach < req and R_i_off_list[i] > 1e-9 and uavs_off_duration[i] > 1e-12:
                    N0_off = float(sigma2)
                    for j in range(J):
                        N0_off += float(eta[i, j] * cus_off_power[j] * h_cj_BS_sq[j])
                    snr_req = float(2.0 ** (req / (B_w * uavs_off_duration[i])) - 1.0)
                    signal_cur = float(
                        np.abs(np.conj(h_ui_BS_vecs[i]).T @ uav_off_beam_vectors[i]) ** 2
                    )
                    signal_req = max(snr_req * N0_off, signal_cur)
                    if signal_cur > 1e-30:
                        scale = float(np.sqrt(signal_req / signal_cur))
                        uav_off_beam_vectors[i] = uav_off_beam_vectors[i] * scale
                        feasibility_log["C9_corrected"].append(i)

            # ---- C10: Task-Freshness —— clip D_off ----
            for i in range(I):
                rhs = float(np.sum(eta[i, :] * cus_off_durations))
                if float(uavs_off_duration[i]) + D_sen > rhs + 1e-12:
                    uavs_off_duration[i] = max(0.0, rhs - D_sen)
                    feasibility_log["C10_corrected"].append(i)

            # ---- C11: CU Offloading Success —— scale up p_c_j ----
            #      R_j depends on p_j linearly in SNR → p_j' = p_j × (2^(L_j/(B·D))-1)/(2^(data/(B·D))-1)
            for j in range(J):
                # Recompute CU data with current (possibly corrected) values
                matched_uav_idx = None
                for i in range(I):
                    if eta[i, j] > 0.5:
                        matched_uav_idx = i
                        break
                if matched_uav_idx is not None:
                    i = matched_uav_idx
                    R_j1 = float(B_w * np.log2(1.0 + cus_off_power[j] * h_cj_BS_sq[j]
                        / max(float(interf_sen_list[i]) + sigma2, 1e-30)))
                    R_j2 = float(B_w * np.log2(1.0 + cus_off_power[j] * h_cj_BS_sq[j]
                        / max(float(interf_off_list[i]) + sigma2, 1e-30)))
                    R_j3 = float(B_w * np.log2(1.0 + cus_off_power[j] * h_cj_BS_sq[j]
                        / max(sigma2, 1e-30)))
                    total_cu_data = (D_sen * R_j1
                                     + float(uavs_off_duration[i]) * R_j2
                                     + (float(cus_off_durations[j]) - D_sen
                                        - float(uavs_off_duration[i])) * R_j3)
                else:
                    R_j3 = float(B_w * np.log2(1.0 + cus_off_power[j] * h_cj_BS_sq[j]
                                 / max(sigma2, 1e-30)))
                    total_cu_data = float(cus_off_durations[j]) * R_j3

                L_j = float(cus_entertaining_task_size[j])
                if total_cu_data < L_j and h_cj_BS_sq[j] > 1e-30:
                    # Scale CU power proportionally to meet task
                    snr_raw = cus_off_power[j] * h_cj_BS_sq[j] / max(sigma2, 1e-30)
                    if snr_raw > 1e-12:
                        rate_raw_per_hz = np.log2(1.0 + snr_raw)
                        if rate_raw_per_hz > 1e-12:
                            D_c = float(cus_off_durations[j])
                            rate_req_per_hz = max(0.0, 2.0 ** (L_j / (B_w * max(D_c, 1e-12))) - 1.0)
                            p_req = rate_req_per_hz * sigma2 / max(h_cj_BS_sq[j], 1e-30)
                            p_req = min(float(p_req), float(P_max_cu))
                            cus_off_power[j] = max(float(cus_off_power[j]), p_req)
                            feasibility_log["C11_corrected"].append(j)

            # ---- C12: CU-Max-Power —— clip ----
            for j in range(J):
                if float(cus_off_power[j]) > float(P_max_cu):
                    cus_off_power[j] = float(P_max_cu)

            # ═══ Recompute SINR / R_sen / R_off / interference after correction ═══
            if feasibility_log["C7_corrected"] or feasibility_log["C9_corrected"]:
                SINR_i_list.clear()
                R_i_sen_list.clear()
                R_i_off_list.clear()
                interf_sen_list.clear()
                interf_off_list.clear()
                for i in range(I):
                    SINR_i = float(np.real(
                        np.conj(uav_sen_beam_vectors[i]).T @ A_iH_Delta_inv_A_i_list[i] @ uav_sen_beam_vectors[i]
                    ))
                    SINR_i = max(SINR_i, 0.0)
                    SINR_i_list.append(SINR_i)
                    R_i_sen_list.append(float(self.xi1 * np.log2(1.0 + self.xi2 * SINR_i)))

                    n0 = float(sigma2)
                    for j in range(J):
                        n0 += float(eta[i, j] * cus_off_power[j] * h_cj_BS_sq[j])
                    b_i = uav_off_beam_vectors[i]
                    sig = float(np.abs(np.conj(h_ui_BS_vecs[i]).T @ b_i) ** 2)
                    R_i_off_list.append(float(B_w * np.log2(1.0 + sig / max(n0, 1e-30))))
                    interf_sen_list.append(
                        float(np.abs(np.conj(h_ui_BS_vecs[i]).T @ uav_sen_beam_vectors[i]) ** 2))
                    interf_off_list.append(
                        float(np.abs(np.conj(h_ui_BS_vecs[i]).T @ uav_off_beam_vectors[i]) ** 2))

        # ══════════════════════════════════════════════════════════════════
        #  闭式解：C13/C14 取等 → 计算 f_u_i^min, f_c_j^min；C15 合约束缩放
        # ══════════════════════════════════════════════════════════════════
        if _use_closed_form_f:
            # C13 binding → f_u_i^min = C_bit · D_sen · R_i_sen / (D_max^U − D_sen − D_u^off)
            denom_u = np.maximum(D_max_u - D_sen - uavs_off_duration, 1e-12)
            uav_bs_freqs = C_bit * D_sen * np.asarray(R_i_sen_list, dtype=float) / denom_u

            # C14 binding → f_c_j^min = C_bit · L_j / (D_max^C − D_c^off)
            denom_c = np.maximum(D_max_c - cus_off_durations, 1e-12)
            cu_bs_freqs = C_bit * cus_entertaining_task_size / denom_c

            # C15: Σf ≤ F_max — 超出则等比例缩放
            f_total = float(np.sum(uav_bs_freqs) + np.sum(cu_bs_freqs))
            if f_total > F_max:
                scale = F_max / f_total
                uav_bs_freqs = uav_bs_freqs * scale
                cu_bs_freqs = cu_bs_freqs * scale

        # ==================================================================
        #  P0 约束检验
        #
        #  分两类：
        #    第一类（保留原样）：UAV 越界 / UAV 碰撞 / BS 频谱过分配 / no_solution
        #                      → 与 my_reward.py 完全一致的惩罚
        #    第二类（tanh 转换）：其余 12 条约束
        #                      → tanh(violation_magnitude / natural_scale)
        #                      → 按项数归一化到 (0, 1)
        # ==================================================================

        # ══════════════════════════════════════════════════════════════════
        #  第一类：保留 my_reward.py 原样的惩罚
        # ══════════════════════════════════════════════════════════════════

        # --- 约束 2: UAV-Flying-Area（逐 UAV 二进制，与 my_reward.py:27-30 一致）---
        uav_exceed_boundary_penalty = np.zeros(I)
        for i in range(I):
            x, y, _ = uavs_pos_cur[i]
            if x < -self.x_max or x > self.x_max or y < -self.y_max or y > self.y_max:
                uav_exceed_boundary_penalty[i] = 1
        boundary_penalty = float(np.sum(uav_exceed_boundary_penalty))

        # --- 约束 3: UAV-collision（碰撞的两个 UAV 各计 1，与 my_reward.py:32-36 一致）---
        uav_collision_penalty = np.zeros(I)
        for i in range(I):
            for jj in range(i + 1, I):
                dist = float(np.linalg.norm(uavs_pos_cur[i] - uavs_pos_cur[jj]))
                if dist < d_min - 1e-9:
                    uav_collision_penalty[i] = 1
                    uav_collision_penalty[jj] = 1
        collision_penalty = float(np.sum(uav_collision_penalty))

        # --- 约束 5: CU-Less-One-UAV（连续值累加，与 my_reward.py:38 一致）---
        bs_alloc_spectrum_penalty = float(
            np.sum(np.maximum(0, np.sum(eta, axis=0) - 1))
        )

        # --- no_solution_penalty（纯学习版无 CCCP，仅占位兼容，恒为 0）---
        no_solution_penalty = 0.0

        # ══════════════════════════════════════════════════════════════════
        #  第二类：其余 12 条约束 → tanh(violation / scale) 归一化
        #  逐项记录，便于诊断具体哪些约束在违反
        # ══════════════════════════════════════════════════════════════════
        tanh_detail = {}  # key: "C1_U0", "C6_U0", ... → float tanh value

        # ---- C1: UAV-speed（每 UAV，scale = V_max_tau）----
        for i in range(I):
            dist = float(np.linalg.norm(uavs_pos_cur[i] - uavs_pos[i]))
            v = max(0.0, V_min_tau - dist) + max(0.0, dist - V_max_tau)
            tanh_detail[f"C1_UAV_Speed_U{i}"] = np.tanh(v / max(V_max_tau, 1e-8))

        # ---- C4: UAV-one-CU（每 UAV，scale = 1.0，无量纲）----
        for i in range(I):
            v = abs(np.sum(eta[i, :]) - 1.0)
            tanh_detail[f"C4_UAV_OneCU_U{i}"] = np.tanh(v / 1.0)

        # ═══ C6/C8 方向+功率分解保证，C13/C14 闭式解保证，C15 闭式解+缩放保证 ═══

        # ---- C7: Sensing-SINR（每 UAV，scale = epsilon）----
        for i in range(I):
            v = max(0.0, epsilon - SINR_i_list[i])
            tanh_detail[f"C7_UAV_SenSINR_U{i}"] = np.tanh(v / max(epsilon, 1e-8))

        # ---- C9: UAV-task-successful-offloading（每 UAV，scale = D_sen * R_i_sen）----
        for i in range(I):
            req = D_sen * max(R_i_sen_list[i], 1e-8)
            v = max(0.0, req - uavs_off_duration[i] * R_i_off_list[i])
            tanh_detail[f"C9_UAV_OffSuccess_U{i}"] = np.tanh(v / max(req, 1e-8))

        # ---- C10: Task-Fresh（每 UAV，scale = D_max_u）----
        for i in range(I):
            rhs = np.sum(eta[i, :] * cus_off_durations)
            v = max(0.0, D_sen + uavs_off_duration[i] - rhs)
            tanh_detail[f"C10_UAV_Freshness_U{i}"] = np.tanh(v / max(D_max_u, 1e-8))

        # ---- C11: CU-task-successful-offloading（每 CU，scale = L_j）----
        log2_cu_snr_no_interf = np.log2(
            1.0 + cus_off_power * h_cj_BS_sq / max(sigma2, 1e-30)
        )
        R_cu_no_interf = B_w * log2_cu_snr_no_interf

        for j in range(J):
            matched_uav_idx = None
            for i in range(I):
                if eta[i, j] > 0.5:
                    matched_uav_idx = i
                    break

            if matched_uav_idx is not None:
                i = matched_uav_idx
                R_j1 = B_w * np.log2(
                    1.0 + cus_off_power[j] * h_cj_BS_sq[j]
                    / max(interf_sen_list[i] + sigma2, 1e-30)
                )
                R_j2 = B_w * np.log2(
                    1.0 + cus_off_power[j] * h_cj_BS_sq[j]
                    / max(interf_off_list[i] + sigma2, 1e-30)
                )
                R_j3 = R_cu_no_interf[j]
                total_cu_data = (
                    D_sen * R_j1
                    + uavs_off_duration[i] * R_j2
                    + (cus_off_durations[j] - D_sen - uavs_off_duration[i])
                    * R_j3
                )
            else:
                total_cu_data = cus_off_durations[j] * R_cu_no_interf[j]

            L_j = max(cus_entertaining_task_size[j], 1e-8)
            v = max(0.0, cus_entertaining_task_size[j] - total_cu_data)
            tanh_detail[f"C11_CU_OffSuccess_CU{j}"] = np.tanh(v / L_j)

        # ---- C12: CU-Max-Power（每 CU，scale = P_max_cu）----
        for j in range(J):
            v = max(0.0, cus_off_power[j] - P_max_cu)
            tanh_detail[f"C12_CU_MaxPower_CU{j}"] = np.tanh(v / max(P_max_cu, 1e-8))

        # 汇总
        tanh_penalty = sum(tanh_detail.values())
        num_tanh_terms = len(tanh_detail)
        tanh_penalty_normalized = tanh_penalty / max(num_tanh_terms, 1)

        # ══════════════════════════════════════════════════════════════════
        #  有效能耗修正（Effective Energy Correction）
        #
        #  问题：agent 输出的动作决定了 E_sum，但约束是否满足不影响 E_sum。
        #        → 信道变差时 E_sum 可能反而更低（agent 用更少功率）。
        #
        #  修正：对 C7(感知SINR)/C9(卸载成功)，若 agent 选择的功率不足，
        #        反算出满足约束所需的最小功率，用 max(chosen, required)
        #        重新计算 E_sum，使能耗物理正确地反映"完成任务的实际代价"。
        #
        #  注意：修正后的功率仅用于 E_sum，不改变波束向量本身。
        # ══════════════════════════════════════════════════════════════════
        p_sen_chosen = np.array([float(np.real(np.conj(uav_sen_beam_vectors[i]).T @ uav_sen_beam_vectors[i]))
                                 for i in range(I)], dtype=float)
        p_off_chosen = np.array([float(np.real(np.conj(uav_off_beam_vectors[i]).T @ uav_off_beam_vectors[i]))
                                 for i in range(I)], dtype=float)

        # ── C7 修正：感知 SINR 不达标 → 反算最小感知功率 ──
        p_sen_eff = np.clip(p_sen_chosen / P_max_uav, 0.0, 1.0)  # 归一化到 [0,1]
        for i in range(I):
            if p_sen_chosen[i] < 1e-12:
                continue
            # G_dir = d_unit^H M_i d_unit = SINR / ||w||²  (per-watt 方向增益)
            G_dir = SINR_i_list[i] / p_sen_chosen[i]
            if G_dir < 1e-30:
                continue
            p_req = epsilon / (P_max_uav * G_dir)
            p_sen_eff[i] = np.clip(max(p_sen_eff[i], p_req), 0.0, 1.0)

        # ── C9 修正：卸载速率不足 → 反算最小卸载功率 ──
        p_off_eff = np.clip(p_off_chosen / P_max_uav, 0.0, 1.0)
        for i in range(I):
            if p_off_chosen[i] < 1e-12 or uavs_off_duration[i] < 1e-8:
                continue
            # 卸载信道 N0 = 干扰 + 噪声
            N0_off = sigma2
            for j in range(J):
                N0_off += eta[i, j] * cus_off_power[j] * h_cj_BS_sq[j]
            # S_unit = per-watt SNR = |h^H d_unit|² / N0 = signal / (||b||² × N0)
            signal = np.abs(np.conj(h_ui_BS_vecs[i]).T @ uav_off_beam_vectors[i]) ** 2
            S_unit = signal / (p_off_chosen[i] * max(N0_off, 1e-30))
            # 所需速率
            R_req = D_sen * max(R_i_sen_list[i], 1e-8) / uavs_off_duration[i]
            # 所需 SNR → 所需功率
            snr_req = max(0.0, 2.0 ** (R_req / B_w) - 1.0)
            p_req = snr_req / max(S_unit, 1e-30) / P_max_uav
            p_off_eff[i] = np.clip(max(p_off_eff[i], p_req), 0.0, 1.0)

        # ── 修正后的 SINR 和 R_sen（用于 E_cp）──
        SINR_eff_list = []
        R_sen_eff_list = []
        for i in range(I):
            SINR_eff = np.clip(p_sen_eff[i] * P_max_uav * (SINR_i_list[i] / max(p_sen_chosen[i], 1e-12)), 0.0, None)
            SINR_eff_list.append(float(SINR_eff))
            R_sen_eff_list.append(float(self.xi1 * np.log2(1.0 + self.xi2 * SINR_eff)))

        # ==================================================================
        #  P0 加权总能耗 E_sum（使用有效能耗修正）
        # ==================================================================
        # 飞行能耗
        uav_dist_diff = np.linalg.norm(uavs_pos_cur - uavs_pos, axis=1)
        uav_dist_diff = np.maximum(uav_dist_diff, 1e-8)
        fly_energy_per_uav = (
            a.uav_c1 * (uav_dist_diff ** 3) / (a.time_slot_duration ** 2)
            + a.uav_c2 * (a.time_slot_duration ** 2) / uav_dist_diff
        )
        E_fly = omega2 * np.sum(fly_energy_per_uav)

        # 感知能耗：使用 C7 修正功率
        sen_power_eff = p_sen_eff * P_max_uav
        E_sen = omega2 * D_sen * np.sum(sen_power_eff)
        # 记录 agent 原始选择的感知功率（用于诊断）
        sen_power_raw = p_sen_chosen

        # 卸载能耗：使用 C9 修正功率
        off_power_eff = p_off_eff * P_max_uav
        E_off = omega2 * np.sum(uavs_off_duration * off_power_eff)
        off_power_raw = p_off_chosen

        # CU 卸载能耗
        E_cu_off = omega3 * np.sum(cus_off_durations * cus_off_power)

        # BS 计算能耗：使用修正后的 R_sen_eff 和 f_u
        E_cp_uav = omega1 * kappa * np.sum(
            C_bit * D_sen * np.asarray(R_sen_eff_list) * (uav_bs_freqs ** 2)
        )
        # CU 娱乐任务计算能耗：κ * C_bit * L_j * f_c_j^2
        E_cp_cu = omega1 * kappa * np.sum(
            C_bit * cus_entertaining_task_size * (cu_bs_freqs ** 2)
        )
        E_cp = E_cp_uav + E_cp_cu

        E_sum = E_fly + E_sen + E_off + E_cu_off + E_cp

        # ==================================================================
        #  Reward 合成
        # ==================================================================
        total_reward_4_energy = np.exp(-E_sum / 1000.0)

        # 第一类：保留原样惩罚（与 my_reward.py 一致）
        penalty_kept = (
            boundary_penalty
            + collision_penalty
            + bs_alloc_spectrum_penalty
            + no_solution_penalty
        )

        # 软约束超额惩罚：tanh_penalty_normalized 超过阈值时，连续惩罚（提供梯度信号）
        _tanh_threshold = 0.1
        extra_tanh_penalty = 10.0 * max(0.0, tanh_penalty_normalized - _tanh_threshold)

        # 总惩罚 = 保留原样惩罚 + tanh 归一化惩罚 + 超额惩罚
        total_penalty = penalty_kept + tanh_penalty_normalized + extra_tanh_penalty
        total_reward = total_reward_4_energy - total_penalty

        reward = {
            "bs": float(total_reward),
            "components": {
                "total_reward_4_energy": float(total_reward_4_energy),
                # ── 保留原样惩罚（与 my_reward.py 命名一致） ──
                "boundary_penalty": float(boundary_penalty),
                "collision_penalty": float(collision_penalty),
                "bs_alloc_spectrum_penalty": float(bs_alloc_spectrum_penalty),
                "no_solution_penalty": float(no_solution_penalty),
                # ── 闭式解计算频率（用于打印诊断）──
                "uav_bs_freqs": np.asarray(uav_bs_freqs, dtype=float).tolist(),
                "cu_bs_freqs": np.asarray(cu_bs_freqs, dtype=float).tolist(),
                # ── tanh 归一化惩罚 ──
                "tanh_penalty": float(tanh_penalty_normalized),
                "tanh_penalty_raw": float(tanh_penalty),
                "tanh_threshold": float(_tanh_threshold),
                "extra_tanh_penalty": float(extra_tanh_penalty),
                "num_tanh_terms": int(num_tanh_terms),
                # ── 逐项约束违反诊断（tanh 原始值）──
                "tanh_detail": {k: float(v) for k, v in tanh_detail.items()},
                # ── feasibility enforcement 日志 ──
                "feasibility_log": {k: list(v) for k, v in feasibility_log.items()},
                # ── 能耗分解 ──
                "E_sum": float(E_sum),
                "E_fly": float(E_fly),
                "E_sen": float(E_sen),
                "E_off": float(E_off),
                "E_cu_off": float(E_cu_off),
                "E_cp": float(E_cp),
                "avg_uav_speed": float(
                    np.mean(uav_dist_diff / a.time_slot_duration)
                ),
                # ── 有效能耗修正诊断 ──
                "sen_power_raw": sen_power_raw.tolist(),
                "sen_power_eff": sen_power_eff.tolist(),
                "off_power_raw": off_power_raw.tolist(),
                "off_power_eff": off_power_eff.tolist(),
                "SINR_raw": [float(x) for x in SINR_i_list],
                "SINR_eff": [float(x) for x in SINR_eff_list],
            },
        }
        return float(total_reward), reward, float(E_sum)
