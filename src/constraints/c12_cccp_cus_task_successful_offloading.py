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
"""

import numpy as np
import cvxpy as cp

NAME = "CCCP:CUs-task-successful-offloading"

LN2 = np.log(2.0)


def build(ctx):
    constraints = []
    for j in range(ctx.J):
        eta_col = ctx.eta_share[:, j]
        p_j = ctx.p_cu_power[j]
        h_cj_bs_sq = abs(ctx.h_cu_2_bs[j]) ** 2

        sum_eta_D_sen = ctx.D_bar_sen * float(np.sum(eta_col))    # Σ_i η_{i,j} D̄^sen
        sum_eta_D_off = float(eta_col @ ctx.D_uav_off)            # Σ_i η_{i,j} D_{u_i}^{off}

        # Σ_i η_{i,j} Tr(H_{u_i,BS} W_i) 与 Σ_i η_{i,j} Tr(H_{u_i,BS} B_i)
        sum_eta_tr_H_W = 0.0
        sum_eta_tr_H_B = 0.0
        # Σ_i η_{i,j} Tr(H_{u_i,BS} (W_i - W_i^{(n)})) 与 Σ_i η_{i,j} Tr(H_{u_i,BS} (B_i - B_i^{(n)}))
        sum_eta_tr_H_W_diff = 0.0
        sum_eta_tr_H_B_diff = 0.0
        for i in range(ctx.I):
            if eta_col[i] == 0.0:
                continue
            sum_eta_tr_H_W = sum_eta_tr_H_W + eta_col[i] * cp.real(
                cp.trace(ctx.H_uav_bs[i] @ ctx.W_sen_beam[i]))
            sum_eta_tr_H_B = sum_eta_tr_H_B + eta_col[i] * cp.real(
                cp.trace(ctx.H_uav_bs[i] @ ctx.B_off_beam[i]))
            sum_eta_tr_H_W_diff = sum_eta_tr_H_W_diff + eta_col[i] * cp.real(
                cp.trace(ctx.H_uav_bs[i] @ (ctx.W_sen_beam[i] - ctx.W_sen_beam_prev[i])))
            sum_eta_tr_H_B_diff = sum_eta_tr_H_B_diff + eta_col[i] * cp.real(
                cp.trace(ctx.H_uav_bs[i] @ (ctx.B_off_beam[i] - ctx.B_off_beam_prev[i])))

        # - Σ_i η_{i,j} D̄^sen B log_2( p_j |h_{c_j,BS}|² + Σ_i η_{i,j} Tr(H W_i) + σ² )
        term_1 = (-sum_eta_D_sen * ctx.B * cp.log(p_j * h_cj_bs_sq + sum_eta_tr_H_W + ctx.sigma_2) / LN2
                  if sum_eta_D_sen > 0.0 else 0.0)
        # - Σ_i η_{i,j} D_{u_i}^{off} B log_2( p_j |h_{c_j,BS}|² + Σ_i η_{i,j} Tr(H B_i) + σ² )
        term_2 = (-sum_eta_D_off * ctx.B * cp.log(p_j * h_cj_bs_sq + sum_eta_tr_H_B + ctx.sigma_2) / LN2
                  if sum_eta_D_off > 0.0 else 0.0)
        # - ( D_{c_j}^{off}(t) - Θ_j(t) ) B υ_j(t)
        term_3 = -(ctx.D_cu_off[j] - ctx.Theta_cu_time[j]) * ctx.B * ctx.upsilon_cu_rate[j]
        # + Σ_i η_{i,j} D̄^sen B log_2( Ψ_{j,1}^{(n)}(t) )
        term_4 = sum_eta_D_sen * ctx.B * np.log2(ctx.Psi_cu_sen[j])
        # + Σ_i η_{i,j} D_{u_i}^{off} B log_2( Ψ_{j,2}^{(n)}(t) )
        term_5 = sum_eta_D_off * ctx.B * np.log2(ctx.Psi_cu_off[j])
        # + ( Σ_i η_{i,j} D̄^sen B ) / ( ln2 · Ψ_{j,1}^{(n)} ) · Σ_i η_{i,j} Tr( H ( W_i - W_i^{(n)} ) )
        term_6 = (sum_eta_D_sen * ctx.B / (LN2 * ctx.Psi_cu_sen[j]) * sum_eta_tr_H_W_diff
                  if sum_eta_D_sen > 0.0 else 0.0)
        # + ( Σ_i η_{i,j} D_{u_i}^{off} B ) / ( ln2 · Ψ_{j,2}^{(n)} ) · Σ_i η_{i,j} Tr( H ( B_i - B_i^{(n)} ) )
        term_7 = (sum_eta_D_off * ctx.B / (LN2 * ctx.Psi_cu_off[j]) * sum_eta_tr_H_B_diff
                  if sum_eta_D_off > 0.0 else 0.0)

        constraints.append(
            ctx.L_cu_task[j]        # L_j(t)
            + term_1 + term_2 + term_3 + term_4 + term_5 + term_6 + term_7
            <= 0
        )
    return constraints
