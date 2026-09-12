"""约束 \eqref{P2:Sensing-SINR}：

    ε Γ_i(t) - Tr(G_i(t) W_i(t)) ≤ 0, ∀u_i ∈ U

其中 G_i(t) = A^H(θ_i(t)) g_i(t) g_i^H(t) A(θ_i(t))。
"""

import cvxpy as cp

NAME = "P2:Sensing-SINR"


def build(ctx):
    constraints = []
    for i in range(ctx.I):
        constraints.append(
            ctx.eps_sinr * ctx.Gamma_sinr[i]                            # ε · Γ_i(t)
            - cp.real(cp.trace(ctx.G_sen_corr[i]                         # - Tr(G_i(t)
                               @ ctx.W_sen_beam[i]))                     #      · W_i(t))
            <= 0
        )
    return constraints
