"""问题 P5 的目标函数（论文 Inner-Level Optimization 中 P5 的 min 部分）。

    min  ω_1 κ Σ_{u_i} C^sen ( z_i + (f_{u_i})² )² / 2
       + ω_1 κ Σ_{c_j} ( C_j L_j )³ / ( D^max_j - D_{c_j}^{off} )²
       + ω_2 Σ_{u_i} ( D̄^sen Tr(W_i) + D_{u_i}^{off} Tr(B_i) + E_{u_i}^{fly} )
       + ω_3 Σ_{c_j} D_{c_j}^{off} p_j
       - ω_1 κ Σ_{u_i} [ C^sen (z_i^{(n)})² / 2 + C^sen (f_{u_i}^{(n)})⁴ / 2
                        + C^sen z_i^{(n)} ( z_i - z_i^{(n)} )
                        + 2 C^sen (f_{u_i}^{(n)})³ ( f_{u_i} - f_{u_i}^{(n)} ) ]
       + ρ Σ_{u_i} [ Tr(W_i) - ‖W_i^{(n)}‖_2
                     - Tr( ν_max(W_i^{(n)}) ν_max^H(W_i^{(n)}) ( W_i - W_i^{(n)} ) )
                     + Tr(B_i) - ‖B_i^{(n)}‖_2
                     - Tr( ν_max(B_i^{(n)}) ν_max^H(B_i^{(n)}) ( B_i - B_i^{(n)} ) ) ]
"""

import numpy as np
import cvxpy as cp

from cccp_params import real_trace

LN2 = np.log(2.0)


def build_objective(ctx):
    """构造并返回 P5 的目标函数表达式（用于 cp.Minimize）。"""
    omega_1, omega_2, omega_3 = ctx.omega_1, ctx.omega_2, ctx.omega_3
    kappa_cpu = ctx.kappa_cpu
    C_sen = ctx.C_sen

    obj = 0.0

    # ── ① BS 感知任务计算能耗：ω_1 κ Σ_{u_i} C^sen ( z_i + (f_{u_i})² )² / 2 ──
    #    由 DCP 辅助变量 τ_i ≥ ( z_i + f_{u_i}² ) / tau_scale（约束 c13）等价写为
    #    ω_1 κ Σ C^sen ( tau_scale · τ_i )² / 2
    for i in range(ctx.I):
        obj = obj + (omega_1 * kappa_cpu * C_sen
                     * cp.square(ctx.tau_zf[i]) * ctx.tau_scale ** 2 / 2.0)

    # ── ② BS 娱乐任务计算能耗：ω_1 κ Σ_{c_j} ( C_j L_j )³ / ( D^max_j - D_{c_j}^{off} )² ──
    for j in range(ctx.J):
        obj = obj + (
            omega_1 * kappa_cpu
            * cp.power(ctx.D_max_cu[j] - ctx.D_cu_off[j], -2)         # 1 / ( D^max_j - D_{c_j}^{off} )²
            * (ctx.C_cu_cycles[j] * ctx.L_cu_task[j]) ** 3            # ( C_j L_j )³
        )

    # ── ③ UAV 感知 / 卸载 / 飞行能耗：ω_2 Σ_{u_i} ( D̄^sen Tr(W_i) + D_{u_i}^{off} Tr(B_i) + E_{u_i}^{fly} ) ──
    for i in range(ctx.I):
        obj = obj + omega_2 * (
            ctx.D_bar_sen * cp.real(cp.trace(ctx.W_sen_beam[i]))      # D̄^sen · Tr(W_i)
            + ctx.D_uav_off[i] * cp.real(cp.trace(ctx.B_off_beam[i]))  # + D_{u_i}^{off} · Tr(B_i)
            + ctx.E_uav_fly[i]                                         # + E_{u_i}^{fly}（常数）
        )

    # ── ④ CU 娱乐任务卸载能耗：ω_3 Σ_{c_j} D_{c_j}^{off} p_j ──
    obj = obj + omega_3 * cp.sum(cp.multiply(ctx.D_cu_off, ctx.p_cu_power))

    # ── ⑤ CCCP 线性化项（对 - C^sen (z_i)² / 2 - C^sen (f_{u_i})⁴ / 2 在 x^(n) 处线性化）──
    #    变量部分：- ω₁κC^sen Σ z_i^{(n)} z_i - 2 ω₁κC^sen Σ (f_{u_i}^{(n)})³ f_{u_i}
    #    常数部分：+ ω₁κC^sen Σ [ (z^{(n)})²/2 + 3/2 (f^{(n)})⁴ ]（不影响 argmin，单独作常数项）
    #    线性化系数由 cccp_params 提供为参数，使该问题只编译一次。
    obj = obj - omega_1 * kappa_cpu * C_sen * cp.sum(
        cp.multiply(ctx.ccp.z_prev, ctx.z_aux_rate))
    obj = obj - 2.0 * omega_1 * kappa_cpu * C_sen * cp.sum(
        cp.multiply(ctx.ccp.f_prev3, ctx.f_uav_freq))
    obj = obj + ctx.ccp.obj_bias

    # ── ⑥ 秩一罚项（对 - ρ ‖W_i‖_2 - ρ ‖B_i‖_2 在 x^(n) 处线性化）──
    #    ρ[Tr(W_i) - ‖W_i^{(n)}‖ - Tr(ν_max ν_max^H (W_i - W_i^{(n)}))] 合并为
    #    Re Tr(M_i W_i) + bias_i，其中 M_i = ρ(I - ν_max ν_max^H)（由 cccp_params 预置）。
    for i in range(ctx.I):
        obj = obj + real_trace(ctx.ccp.M_sen_re[i], ctx.ccp.M_sen_im[i],
                               ctx.W_sen_beam[i]) + ctx.ccp.bias_r1[i]
        obj = obj + real_trace(ctx.ccp.M_off_re[i], ctx.ccp.M_off_im[i],
                               ctx.B_off_beam[i]) + ctx.ccp.bias_r1_off[i]

    return obj
