"""约束 \eqref{CCCP:CUs-task-successful-offloading}（约束 \eqref{P2:CUs-task-successful-offloading} 在 x^(n) 处的 CCCP 线性化）：

    L_j(t)
    - Σ_i η_{i,j} D̄^sen B log_2( p_j |h_{c_j,BS}|² + Σ_i η_{i,j} Tr(H_{u_i,BS} W_i) + σ² )
    - Σ_i η_{i,j} D_{u_i}^{off} B log_2( p_j |h_{c_j,BS}|² + Σ_i η_{i,j} Tr(H_{u_i,BS} B_i) + σ² )
    - ( D_{c_j}^{off} - Θ_j ) B υ_j
    + Σ_i η_{i,j} D̄^sen B log_2( Ψ_{j,1}^{(n)} )
    + Σ_i η_{i,j} D_{u_i}^{off} B log_2( Ψ_{j,2}^{(n)} )
    + ( Σ_i η_{i,j} D̄^sen B ) / ( ln2 · Ψ_{j,1}^{(n)} ) · Σ_i η_{i,j} Tr( H_{u_i,BS} ( W_i - W_i^{(n)} ) )
    + ( Σ_i η_{i,j} D_{u_i}^{off} B ) / ( ln2 · Ψ_{j,2}^{(n)} ) · Σ_i η_{i,j} Tr( H_{u_i,BS} ( B_i - B_i^{(n)} ) )
    ≤ 0, ∀c_j ∈ C

其中 Ψ_{j,1}^{(n)}(t) = Σ_i η_{i,j} Tr(H_{u_i,BS} W_i^{(n)}) + σ²，
    Ψ_{j,2}^{(n)}(t) = Σ_i η_{i,j} Tr(H_{u_i,BS} B_i^{(n)}) + σ²。

实现说明：为让 P5 只编译一次（cvxpy DPP 复用），把与 x^(n) 有关的标量系数
cu_coef_w[j]、cu_coef_b[j] 做成参数，并把所有与变量无关的常数（L_j、两个 log_2(Ψ)
常数、-(Θ_j)Bυ_j、两个线性化常数）合并为 cu_bias[j]（见 cccp_params.py）。
"""

import numpy as np
import cvxpy as cp

NAME = "CCCP:CUs-task-successful-offloading"

LN2 = np.log(2.0)


def build(ctx):
    constraints = []
    for j in range(ctx.J):
        eta_col = ctx.eta_share[:, j]
        sum_eta_D_sen = ctx.D_bar_sen * float(np.sum(eta_col))    # Σ_i η_{i,j} D̄^sen
        sum_eta_D_off = float(eta_col @ ctx.D_uav_off)            # Σ_i η_{i,j} D_{u_i}^{off}

        # Σ_i η_{i,j} Tr(H_{u_i,BS} W_i) 与 Σ_i η_{i,j} Tr(H_{u_i,BS} B_i)
        sum_eta_tr_H_W = 0.0
        sum_eta_tr_H_B = 0.0
        for i in range(ctx.I):
            if eta_col[i] == 0.0:
                continue
            sum_eta_tr_H_W = sum_eta_tr_H_W + eta_col[i] * cp.real(
                cp.trace(ctx.H_uav_bs[i] @ ctx.W_sen_beam[i]))
            sum_eta_tr_H_B = sum_eta_tr_H_B + eta_col[i] * cp.real(
                cp.trace(ctx.H_uav_bs[i] @ ctx.B_off_beam[i]))

        # 复噪声基底：p_j |h_{c_j,BS}|² + σ²
        arg_base = ctx.p_cu_power[j] * abs(ctx.h_cu_2_bs[j]) ** 2 + ctx.sigma_2

        # - Σ_i η_{i,j} D̄^sen B log_2( p_j |h_{c_j,BS}|² + Σ_i η_{i,j} Tr(H W_i) + σ² )
        term_1 = (-sum_eta_D_sen * ctx.B * cp.log(arg_base + sum_eta_tr_H_W) / LN2
                  if sum_eta_D_sen > 0.0 else 0.0)
        # - Σ_i η_{i,j} D_{u_i}^{off} B log_2( p_j |h_{c_j,BS}|² + Σ_i η_{i,j} Tr(H B_i) + σ² )
        term_2 = (-sum_eta_D_off * ctx.B * cp.log(arg_base + sum_eta_tr_H_B) / LN2
                  if sum_eta_D_off > 0.0 else 0.0)
        # - ( D_{c_j}^{off}(t) - Θ_j(t) ) B υ_j(t) 的变量部分
        term_3 = -ctx.D_cu_off[j] * ctx.B * ctx.upsilon_cu_rate[j]

        # 线性化修正项（系数为参数）+ 常数项（参数）
        constraints.append(
            term_1 + term_2 + term_3
            + ctx.ccp.cu_coef_w[j] * sum_eta_tr_H_W
            + ctx.ccp.cu_coef_b[j] * sum_eta_tr_H_B
            + ctx.ccp.cu_bias[j]
            <= 0
        )
    return constraints
