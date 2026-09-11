"""约束 \eqref{P2:UAVs-task-successful-offloading}：

    z_i(t) - D_{u_i}^{off}(t) B log_2( Tr(H_{u_i,BS}(t) B_i(t)) + Φ_i(t) )
    + D_{u_i}^{off}(t) B log_2( Φ_i(t) ) ≤ 0, ∀u_i ∈ U
"""

import numpy as np
import cvxpy as cp

NAME = "P2:UAVs-task-successful-offloading"

LN2 = np.log(2.0)


def build(ctx):
    # Φ_i ≈ 1e-11 且 D·B ≈ 1e7，原始「log2(Tr+Φ) - log2(Φ)」形式存在两个
    # ~1e7 量级项相消，MOSEK 会返回 status=optimal 却大幅违反该约束的伪解。
    # 利用 log2(a+Φ) - log2(Φ) = log2(1 + a/Φ) 作数学等价改写，消除相消。
    constraints = []
    for i in range(ctx.I):
        off_gain_over_phi = (
            cp.real(cp.trace(ctx.H_uav_bs[i] @ ctx.B_off_beam[i]))
            / ctx.Phi_off_inr[i]
        )
        constraints.append(
            ctx.z_aux_rate[i]
            - ctx.D_uav_off[i] * ctx.B * cp.log(1.0 + off_gain_over_phi) / LN2
            <= 0
        )
    return constraints
