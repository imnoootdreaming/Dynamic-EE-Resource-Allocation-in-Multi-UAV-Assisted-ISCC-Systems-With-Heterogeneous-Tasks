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

    # ── ① BS 感知任务计算能耗：ω₁κ Σ C^sen z_i f_{u_i}² = g·Σ z_i ũ_i ──
    #    其中 ũ_i = (f_{u_i}/freq_scale)²（由约束 c13 的 SOC 形式 ũ_i ≥ f̃_i² 给出），
    #    g = ω₁κC^sen·freq_scale²。
    #    ★ 关键：先把 f² 归一到与 z 同量级的 ũ（z≈450、ũ≈369）再参与运算。
    #      原写法把 (z + f²) 相加后平方，而 z 与 f² 差 12 个数量级，展开后目标里出现
    #      ~1e4 量级的常数与相消项，MOSEK 的相对容差（1e-8×|obj|）被迫放大到 ~1e-4 J，
    #      导致每轮返回的解在真实目标上可以高于上一轮，迭代不再单调。
    g_sen = omega_1 * kappa_cpu * C_sen * ctx.freq_scale ** 2
    obj = obj + (g_sen / 2.0) * cp.sum(cp.square(ctx.z_aux_rate)
                                       + cp.square(ctx.u_freq_sq))

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

    # ── ⑤ 对乘积 g·z_i·ũ_i 的 CCCP 线性化（AM-GM 拆分后对凹部取切线）──
    #    z ũ = (z²+ũ²)/2 - (z-ũ)²/2 ≤ (z²+ũ²)/2 - d·(z-ũ) + d²/2,  d = z⁽ⁿ⁾ - ũ⁽ⁿ⁾
    #    于是 ① + ⑤ = g·zũ + (g/2)·((z-ũ) - d)² ≥ g·zũ（上界代理），且在线性化点取等号。
    #    所有项都是 O(g·z·ũ) ≈ 1e-6 量级，不再出现大数相消。
    obj = obj - g_sen * cp.sum(
        cp.multiply(ctx.ccp.d_prev_z_u, ctx.z_aux_rate - ctx.u_freq_sq))
    obj = obj + g_sen * cp.sum(ctx.ccp.cst_prev)

    # ── ⑥ 秩一罚项（对 - ρ ‖W_i‖_2 - ρ ‖B_i‖_2 在 x^(n) 处线性化）──
    #    ρ[Tr(W_i) - ‖W_i^{(n)}‖ - Tr(ν_max ν_max^H (W_i - W_i^{(n)}))] 合并为
    #    Re Tr(M_i W_i) + bias_i，其中 M_i = ρ(I - ν_max ν_max^H)（由 cccp_params 预置）。
    for i in range(ctx.I):
        obj = obj + real_trace(ctx.ccp.M_sen_re[i], ctx.ccp.M_sen_im[i],
                               ctx.W_sen_beam[i]) + ctx.ccp.bias_r1[i]
        obj = obj + real_trace(ctx.ccp.M_off_re[i], ctx.ccp.M_off_im[i],
                               ctx.B_off_beam[i]) + ctx.ccp.bias_r1_off[i]

    return obj
