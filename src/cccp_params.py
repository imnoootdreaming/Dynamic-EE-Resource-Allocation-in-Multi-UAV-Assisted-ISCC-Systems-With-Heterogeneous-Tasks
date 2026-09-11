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
    z_prev, f_prev3                      —— 目标 ⑤ 的线性化系数
    obj_bias                             —— 目标中与变量无关的常数（不影响 argmin）
    M_sen / M_off / bias_r1 / *_off      —— 目标 ⑥ 的秩一罚项（系数已折进矩阵）
    M_psi / psi_sen_bias                 —— 约束 c11
    cu_coef_w / cu_coef_b / cu_bias      —— 约束 c12
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

    z_prev: cp.Parameter
    f_prev3: cp.Parameter
    obj_bias: cp.Parameter
    bias_r1: cp.Parameter
    bias_r1_off: cp.Parameter
    M_sen_re: list
    M_sen_im: list
    M_off_re: list
    M_off_im: list
    M_psi_re: list
    M_psi_im: list
    psi_sen_bias: cp.Parameter
    cu_coef_w: cp.Parameter
    cu_coef_b: cp.Parameter
    cu_bias: cp.Parameter

    @classmethod
    def create(cls, params):
        I, J, N = params.I, params.J, params.N
        return cls(
            z_prev=cp.Parameter(I),
            f_prev3=cp.Parameter(I),
            obj_bias=cp.Parameter(),
            bias_r1=cp.Parameter(I),
            bias_r1_off=cp.Parameter(I),
            M_sen_re=[cp.Parameter((N, N)) for _ in range(I)],
            M_sen_im=[cp.Parameter((N, N)) for _ in range(I)],
            M_off_re=[cp.Parameter((N, N)) for _ in range(I)],
            M_off_im=[cp.Parameter((N, N)) for _ in range(I)],
            M_psi_re=[cp.Parameter((N, N)) for _ in range(I)],
            M_psi_im=[cp.Parameter((N, N)) for _ in range(I)],
            psi_sen_bias=cp.Parameter(I),
            cu_coef_w=cp.Parameter(J),
            cu_coef_b=cp.Parameter(J),
            cu_bias=cp.Parameter(J),
        )


def sync(cp_params, ctx):
    """由 ctx 中当前的线性化点（numpy）刷新全部 Parameter 取值。"""
    A = ctx.omega_1 * ctx.kappa_cpu
    Cs = ctx.C_sen
    rho = ctx.rho_penalty
    I, J, N = ctx.I, ctx.J, ctx.N

    z_prev = np.asarray(ctx.z_aux_rate_prev, dtype=float)
    f_prev = np.asarray(ctx.f_uav_freq_prev, dtype=float)
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

    cp_params.z_prev.value = z_prev
    cp_params.f_prev3.value = f_prev ** 3
    cp_params.bias_r1.value = bias_r1
    cp_params.bias_r1_off.value = bias_r1_off
    cp_params.psi_sen_bias.value = psi_bias
    # 目标 ⑤ 线性化中与变量无关的常数部分（A·C^sen[z_prev²/2 + 3/2·f_prev⁴]）
    cp_params.obj_bias.value = float(A * Cs * (np.sum(z_prev ** 2 / 2.0)
                                               + 1.5 * np.sum(f_prev ** 4)))

    cu_coef_w = np.zeros(J)
    cu_coef_b = np.zeros(J)
    cu_bias = np.zeros(J)
    for j in range(J):
        eta_col = ctx.eta_share[:, j]
        sum_d_sen = ctx.D_bar_sen * float(np.sum(eta_col))     # Σᵢ η_{i,j} D̄^sen
        sum_d_off = float(eta_col @ ctx.D_uav_off)             # Σᵢ η_{i,j} D_{u_i}^off

        sw_prev = sum(eta_col[i] * np.real(np.trace(ctx.H_uav_bs[i] @ ctx.W_sen_beam_prev[i]))
                      for i in range(I) if eta_col[i] != 0.0)
        sb_prev = sum(eta_col[i] * np.real(np.trace(ctx.H_uav_bs[i] @ ctx.B_off_beam_prev[i]))
                      for i in range(I) if eta_col[i] != 0.0)

        coef_w = sum_d_sen * ctx.B / (LN2 * ctx.Psi_cu_sen[j]) if sum_d_sen > 0.0 else 0.0
        coef_b = sum_d_off * ctx.B / (LN2 * ctx.Psi_cu_off[j]) if sum_d_off > 0.0 else 0.0
        cu_coef_w[j] = coef_w
        cu_coef_b[j] = coef_b
        # 常数项：L_j + 两个对数项 + (D_cu_off 的常数部分) - 两个线性化常数
        cu_bias[j] = (ctx.L_cu_task[j]
                      + sum_d_sen * ctx.B * np.log2(ctx.Psi_cu_sen[j])
                      + sum_d_off * ctx.B * np.log2(ctx.Psi_cu_off[j])
                      + ctx.Theta_cu_time[j] * ctx.B * ctx.upsilon_cu_rate[j]
                      - coef_w * sw_prev - coef_b * sb_prev)

    cp_params.cu_coef_w.value = cu_coef_w
    cp_params.cu_coef_b.value = cu_coef_b
    cp_params.cu_bias.value = cu_bias
