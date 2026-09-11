"""P5 的 DPP 参数化：把 CCCP 每轮变化的线性化量做成 cvxpy Parameter，
使 cvxpy 只编译一次、后续求解复用编译缓存（免去每轮重新 canonicalize）。

为什么要拆成实部/虚部两个实 Parameter：
    cvxpy 只要发现任何一个 **复数** Parameter，就会在求解链里强制插入
    ``EvalParams`` 归约，从而关闭 DPP 缓存（见 cvxpy solving_chain.py:241），
    于是每次 solve 都会重跑完整 canonicalize。
    因此这里把所有复数系数矩阵 M 拆成 (M_re, M_im) 两个实 Parameter，
    再用恒等式
        Re Tr(M @ W) = <M_re, Re(W)^T> - <M_im, Im(W)^T>
    在 DPP 允许的「实参数 × 变量仿射表达式」范围内重建同一表达式。
    该改写与原式在数学上完全等价（仅实部/虚部展开）。

参数化的量（全部为每轮 CCCP 线性化点派生）：
    d_prev_z_u, cst_prev                 —— 目标 ⑤（乘积形式感知能耗）的线性化系数
    M_sen / M_off / bias_r1 / *_off      —— 目标 ⑥ 的秩一罚项（系数已折进矩阵）
    M_psi / psi_sen_bias                 —— 约束 c11
    cu_r1 / cu_inv1 / cu_k1 / cu_p1      —— 约束 c12（第 1 个 log 项，已按 Ψ 归一化）
    cu_r2 / cu_inv2 / cu_k2 / cu_p2      —— 约束 c12（第 2 个 log 项，已按 Ψ 归一化）
"""

from dataclasses import dataclass

import numpy as np
import cvxpy as cp

LN2 = np.log(2.0)


def real_trace(A_re, A_im, matrix):
    """Re Tr(A @ W)，其中 A = A_re + j·A_im 由实 Parameter 给出，W 为复 Hermitian 变量。"""
    return cp.sum(cp.multiply(A_re, cp.real(matrix).T)) \
        - cp.sum(cp.multiply(A_im, cp.imag(matrix).T))


@dataclass
class CccpParams:
    """CCCP 线性化量对应的 cvxpy Parameter 集合。"""

    d_prev_z_u: cp.Parameter
    cst_prev: cp.Parameter
    bias_r1: cp.Parameter
    bias_r1_off: cp.Parameter
    M_sen_re: list
    M_sen_im: list
    M_off_re: list
    M_off_im: list
    M_psi_re: list
    M_psi_im: list
    psi_sen_bias: cp.Parameter
    cu_r1: cp.Parameter
    cu_inv1: cp.Parameter
    cu_r2: cp.Parameter
    cu_inv2: cp.Parameter
    cu_k1: cp.Parameter
    cu_k2: cp.Parameter
    cu_p1: cp.Parameter
    cu_p2: cp.Parameter

    @classmethod
    def create(cls, params):
        I, J, N = params.I, params.J, params.N
        return cls(
            d_prev_z_u=cp.Parameter(I),
            cst_prev=cp.Parameter(I),
            bias_r1=cp.Parameter(I),
            bias_r1_off=cp.Parameter(I),
            M_sen_re=[cp.Parameter((N, N)) for _ in range(I)],
            M_sen_im=[cp.Parameter((N, N)) for _ in range(I)],
            M_off_re=[cp.Parameter((N, N)) for _ in range(I)],
            M_off_im=[cp.Parameter((N, N)) for _ in range(I)],
            M_psi_re=[cp.Parameter((N, N)) for _ in range(I)],
            M_psi_im=[cp.Parameter((N, N)) for _ in range(I)],
            psi_sen_bias=cp.Parameter(I),
            cu_r1=cp.Parameter(J),
            cu_inv1=cp.Parameter(J),
            cu_r2=cp.Parameter(J),
            cu_inv2=cp.Parameter(J),
            cu_k1=cp.Parameter(J),
            cu_k2=cp.Parameter(J),
            cu_p1=cp.Parameter(J),
            cu_p2=cp.Parameter(J),
        )


def cu_row_constants(ctx):
    """c12 每行的"迭代无关"常数（用于对约束逐行做无量纲归一化）。

    c12 原式（论文 CCCP:CUs-task-successful-offloading）中每项都带因子
    Σᵢ η_{i,j} D̄^sen·B（或 Σᵢ η_{i,j} D_off·B），而 Ψ_{j,1} / Ψ_{j,2} 在弱接收
    功率下只有 ~1e-10，导致系数 Σᵢ η D̄ B/(ln2·Ψ) 高达 ~1e15、折叠后行系数 ~1e9，
    与量级 ~22 的目标严重失配而使 MOSEK 判定失败。这里把第 j 行整体除以正数
    nrm_j = B·S_tot_j（等价变换），使行系数回到 O(1)~O(10²)。

    返回均为 (J,) 且不随 CCCP 迭代变化：
        S1_j    = Σᵢ η_{i,j} D̄^sen
        S2_j    = Σᵢ η_{i,j} D_{u_i}^off
        S_tot_j = max(S1_j + S2_j, 1.0)
        w1_j    = S1_j / S_tot_j        # log 项权重
        w2_j    = S2_j / S_tot_j        # log 项权重
        dw_j    = υ_j / S_tot_j         # D_cu_off 的系数
        const_j = Θ_j υ_j / S_tot_j + L_j / (B·S_tot_j)
    """
    J = ctx.J
    eta = np.asarray(ctx.eta_share, dtype=float)
    S1 = np.array([ctx.D_bar_sen * float(np.sum(eta[:, j])) for j in range(J)])
    S2 = np.array([float(eta[:, j] @ ctx.D_uav_off) for j in range(J)])
    S_tot = np.maximum(S1 + S2, 1.0)
    w1 = S1 / S_tot
    w2 = S2 / S_tot
    upsilon = np.asarray(ctx.upsilon_cu_rate, dtype=float)
    dw = upsilon / S_tot
    const = ((np.asarray(ctx.Theta_cu_time, dtype=float) * upsilon) / S_tot
             + np.asarray(ctx.L_cu_task, dtype=float) / (ctx.B * S_tot))
    return w1, w2, dw, const, S_tot


def sync(cp_params, ctx):
    """由 ctx 中当前的线性化点（numpy）刷新全部 Parameter 取值。"""
    rho = ctx.rho_penalty
    I, J, N = ctx.I, ctx.J, ctx.N

    z_prev = np.asarray(ctx.z_aux_rate_prev, dtype=float)
    f_prev = np.asarray(ctx.f_uav_freq_prev, dtype=float)
    # 目标 ⑤：d = z⁽ⁿ⁾ - ũ⁽ⁿ⁾，其中 ũ⁽ⁿ⁾ = (f⁽ⁿ⁾/freq_scale)²
    # （在最优处 ũ 恒取下界，故由 f⁽ⁿ⁾ 直接还原即可，无需额外存变量值）
    d_prev = z_prev - (f_prev / ctx.freq_scale) ** 2
    cp_params.d_prev_z_u.value = d_prev
    cp_params.cst_prev.value = d_prev ** 2 / 2.0
    bias_r1 = np.zeros(I)
    bias_r1_off = np.zeros(I)
    psi_bias = np.zeros(I)

    for i in range(I):
        # ── 目标 ⑥：ρ[Tr(W) - ‖W⁽ⁿ⁾‖ - Tr(ννᴴ(W - W⁽ⁿ⁾))] 合并后 ──
        #    系数矩阵 M = ρ(I - ννᴴ)，常数项 bias = ρ(-‖W⁽ⁿ⁾‖ + Tr(ννᴴ W⁽ⁿ⁾))
        m_sen = rho * (np.eye(N) - ctx.nu_max_sen[i])
        m_off = rho * (np.eye(N) - ctx.nu_max_off[i])
        cp_params.M_sen_re[i].value = m_sen.real
        cp_params.M_sen_im[i].value = m_sen.imag
        cp_params.M_off_re[i].value = m_off.real
        cp_params.M_off_im[i].value = m_off.imag
        bias_r1[i] = rho * (-ctx.W_sen_beam_norm_prev[i]
                            + np.real(np.trace(ctx.nu_max_sen[i] @ ctx.W_sen_beam_prev[i])))
        bias_r1_off[i] = rho * (-ctx.B_off_beam_norm_prev[i]
                                + np.real(np.trace(ctx.nu_max_off[i] @ ctx.B_off_beam_prev[i])))

        # ── 约束 c11：系数折进矩阵 M_psi = ξ₁ξ₂/(ln2·Ψ) · G，常数折进 psi_sen_bias ──
        coef = ctx.xi_1 * ctx.xi_2 / (LN2 * ctx.Psi_sen[i])
        m_psi = coef * ctx.G_sen_corr[i]
        cp_params.M_psi_re[i].value = m_psi.real
        cp_params.M_psi_im[i].value = m_psi.imag
        psi_bias[i] = (-ctx.xi_1 * np.log2(ctx.Gamma_sinr[i])
                       + ctx.xi_1 * np.log2(ctx.Psi_sen[i])
                       - coef * np.real(np.trace(ctx.G_sen_corr[i] @ ctx.W_sen_beam_prev[i])))

    cp_params.bias_r1.value = bias_r1
    cp_params.bias_r1_off.value = bias_r1_off
    cp_params.psi_sen_bias.value = psi_bias

    # ── 约束 c12：逐行无量纲归一化（数学恒等，仅改善条件数） ──────────────
    #     行 j 除以 nrm_j = B·S_tot_j，并把 log 自变量写成 (arg_base+P)/Ψ 的形式，
    #     使 log2(Ψ) 与精确 log 项中的大常数精确抵消（见 cu_row_constants 说明）。
    w1, w2, _, _, _ = cu_row_constants(ctx)
    cu_r1 = np.zeros(J)
    cu_r2 = np.zeros(J)
    cu_inv1 = np.zeros(J)
    cu_inv2 = np.zeros(J)
    cu_k1 = np.zeros(J)
    cu_k2 = np.zeros(J)
    cu_p1 = np.zeros(J)
    cu_p2 = np.zeros(J)
    for j in range(J):
        psi1 = ctx.Psi_cu_sen[j]      # Ψ_{j,1} = σ² + Σᵢ η Tr(H W⁽ⁿ⁾)
        psi2 = ctx.Psi_cu_off[j]      # Ψ_{j,2} = σ² + Σᵢ η Tr(H B⁽ⁿ⁾)
        arg_base = ctx.p_cu_power[j] * abs(ctx.h_cu_2_bs[j]) ** 2 + ctx.sigma_2
        inv1 = 1.0 / psi1
        inv2 = 1.0 / psi2
        sw_prev = psi1 - ctx.sigma_2
        sb_prev = psi2 - ctx.sigma_2
        cu_r1[j] = arg_base * inv1    # arg_base / Ψ_{j,1}
        cu_r2[j] = arg_base * inv2    # arg_base / Ψ_{j,2}
        cu_inv1[j] = inv1
        cu_inv2[j] = inv2
        cu_k1[j] = w1[j] / LN2 * inv1
        cu_k2[j] = w2[j] / LN2 * inv2
        cu_p1[j] = cu_k1[j] * sw_prev
        cu_p2[j] = cu_k2[j] * sb_prev

    cp_params.cu_r1.value = cu_r1
    cp_params.cu_r2.value = cu_r2
    cp_params.cu_inv1.value = cu_inv1
    cp_params.cu_inv2.value = cu_inv2
    cp_params.cu_k1.value = cu_k1
    cp_params.cu_k2.value = cu_k2
    cp_params.cu_p1.value = cu_p1
    cp_params.cu_p2.value = cu_p2
