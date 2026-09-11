"""约束 \eqref{CCCP:Auxiliary-Variable-1}（约束 \eqref{P2:Auxiliary-Variable-1} 在 x^(n) 处的 CCCP 线性化）：

    - ξ_1 log_2( Γ_i(t) ) - z_i(t) + ξ_1 log_2( Ψ_i^{(n)}(t) )
    + ξ_1 ξ_2 / ( ln2 · Ψ_i^{(n)}(t) ) · Tr( G_i(t) ( W_i(t) - W_i^{(n)}(t) ) ) ≤ 0, ∀u_i ∈ U

其中 Ψ_i^{(n)}(t) = ξ_2 Tr( G_i(t) W_i^{(n)}(t) ) + Γ_i(t)。
"""

import numpy as np
import cvxpy as cp

NAME = "CCCP:Auxiliary-Variable-1"

LN2 = np.log(2.0)


def build(ctx):
    constraints = []
    for i in range(ctx.I):
        constraints.append(
            -ctx.xi_1 * np.log2(ctx.Gamma_sinr[i])                             # - ξ_1 log_2(Γ_i(t))
            - ctx.z_aux_rate[i]                                                # - z_i(t)
            + ctx.xi_1 * np.log2(ctx.Psi_sen[i])                               # + ξ_1 log_2(Ψ_i^{(n)}(t))
            + (ctx.xi_1 * ctx.xi_2 / (LN2 * ctx.Psi_sen[i]))                   # + ξ_1 ξ_2 / (ln2 · Ψ_i^{(n)}(t))
            * cp.real(cp.trace(ctx.G_sen_corr[i]                               #   · Tr(G_i(t)
                               @ (ctx.W_sen_beam[i]                            #       · (W_i(t)
                                  - ctx.W_sen_beam_prev[i])))                  #         - W_i^{(n)}(t)))
            <= 0
        )
    return constraints
