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

数值实现（与原式逐项恒等）：
原式每项都带因子 Σᵢ η D̄^sen·B，而 BS 侧接收功率 Ψ_{j,1} 在弱信道下仅 ~1e-10，
使线性化系数 Σᵢ η D̄ B/(ln2·Ψ) 高达 ~1e15、折叠到 H 上仍有 ~1e9，与量级 ~22 的
目标严重失配，直接导致 MOSEK 第 2 轮起报 UNKNOWN/SolverError。这里做两步恒等变换：

1) 逐行除以正数 nrm_j = B·S_tot_j（S_tot_j = max(Σᵢη D̄^sen + Σᵢη D_off, 1)）；
2) 把 log 自变量与线性化项按 Ψ_{j,1}/Ψ_{j,2} 无量纲化，即
       P ↦ P̂ = P/Ψ,   arg_base + P = Ψ·(arg_base/Ψ + P̂)，
   于是原式中的 ±Σᵢη D̄ B·log_2(Ψ) 与精确 log 项中的大常数精确抵消，剩
       - w1·log_2( r1 + P̂ ) + k1·P̂ - p1   (w1 = ΣηD̄/S_tot)
   其中 r1 = arg_base/Ψ_{j,1}、k1 = w1/(ln2·Ψ_{j,1})、p1 = k1·ΣηTr(HW^{(n)})。
   此时行系数回到 O(1)~O(10²)。

与 x^(n) 有关的量（Ψ_{j,1}、Ψ_{j,2} 及其派生量）做成 cvxpy Parameter，使 P5 只编译
一次（DPP 复用）；与迭代无关的常数（w1/w2/dw/const）由 cccp_params.cu_row_constants
直接算出（见 cccp_params.py）。
"""

import numpy as np
import cvxpy as cp

from cccp_params import cu_row_constants

NAME = "CCCP:CUs-task-successful-offloading"

LN2 = np.log(2.0)


def build(ctx):
    w1, w2, dw, const, _ = cu_row_constants(ctx)
    constraints = []
    for j in range(ctx.J):
        eta_col = ctx.eta_share[:, j]

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

        # - ( D_{c_j}^{off}(t) - Θ_j ) B υ_j / nrm_j + L_j/nrm_j
        expr = -ctx.D_cu_off[j] * dw[j] + const[j]

        if w1[j] > 0.0:
            # - w1 log_2( arg_base/Ψ_{j,1} + ΣηTr(HW)/Ψ_{j,1} )
            # + k1 ΣηTr(HW) - p1
            expr = (expr
                    - w1[j] * cp.log(ctx.ccp.cu_r1[j]
                                     + ctx.ccp.cu_inv1[j] * sum_eta_tr_H_W) / LN2
                    + ctx.ccp.cu_k1[j] * sum_eta_tr_H_W
                    - ctx.ccp.cu_p1[j])
        if w2[j] > 0.0:
            expr = (expr
                    - w2[j] * cp.log(ctx.ccp.cu_r2[j]
                                     + ctx.ccp.cu_inv2[j] * sum_eta_tr_H_B) / LN2
                    + ctx.ccp.cu_k2[j] * sum_eta_tr_H_B
                    - ctx.ccp.cu_p2[j])

        constraints.append(expr <= 0)
    return constraints
