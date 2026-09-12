"""约束 \eqref{P2:UAV-Off-Max-Power}：

    Tr(B_i(t)) - P^max_UAV ≤ 0, ∀u_i ∈ U
"""

import cvxpy as cp

NAME = "P2:UAV-Off-Max-Power"


def build(ctx):
    constraints = []
    for i in range(ctx.I):
        constraints.append(
            cp.real(cp.trace(ctx.B_off_beam[i]))    # Tr(B_i(t))
            - ctx.P_max_uav                          # - P^max_UAV
            <= 0
        )
    return constraints
