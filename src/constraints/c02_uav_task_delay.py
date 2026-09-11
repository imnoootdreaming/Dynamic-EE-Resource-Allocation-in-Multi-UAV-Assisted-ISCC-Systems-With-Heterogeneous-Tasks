"""约束 \eqref{P1-1:UAV-task-delay}：

    D̄^sen f_{u_i}(t) + D_{u_i}^{off}(t) f_{u_i}(t)
    + C^sen z_i(t) - D^sen_max(t) f_{u_i}(t) ≤ 0, ∀u_i ∈ U
"""

NAME = "P1-1:UAV-task-delay"


def build(ctx):
    constraints = []
    for i in range(ctx.I):
        constraints.append(
            ctx.D_bar_sen * ctx.f_uav_freq[i]              # D̄^sen · f_{u_i}(t)
            + ctx.D_uav_off[i] * ctx.f_uav_freq[i]         # + D_{u_i}^{off}(t) · f_{u_i}(t)
            + ctx.C_sen * ctx.z_aux_rate[i]                # + C^sen · z_i(t)
            - ctx.D_max_sen * ctx.f_uav_freq[i]            # - D^sen_max(t) · f_{u_i}(t)
            <= 0
        )
    return constraints
