"""约束 \eqref{P2:UAVs-task-successful-offloading}：

    z_i(t) - D_{u_i}^{off}(t) B log_2( Tr(H_{u_i,BS}(t) B_i(t)) + Φ_i(t) )
    + D_{u_i}^{off}(t) B log_2( Φ_i(t) ) ≤ 0, ∀u_i ∈ U
"""

import numpy as np
import cvxpy as cp

NAME = "P2:UAVs-task-successful-offloading"

LN2 = np.log(2.0)


def build(ctx):
    constraints = []
    for i in range(ctx.I):
        constraints.append(
            ctx.z_aux_rate[i]                                            # z_i(t)
            - ctx.D_uav_off[i] * ctx.B * cp.log(                          # - D_{u_i}^{off} · B · log(
                cp.real(cp.trace(ctx.H_uav_bs[i] @ ctx.B_off_beam[i]))    #     Tr(H_{u_i,BS} B_i)
                + ctx.Phi_off_inr[i]                                      #     + Φ_i(t)
            ) / LN2                                                      #   ) / ln2
            + ctx.D_uav_off[i] * ctx.B * np.log2(ctx.Phi_off_inr[i])      # + D_{u_i}^{off} · B · log_2(Φ_i(t))
            <= 0
        )
    return constraints
