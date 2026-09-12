"""约束 \eqref{P1:Task-Fresh}：

    D̄^sen + D_{u_i}^{off}(t) - Σ_{j=1}^{J} η_{i,j}(t) D_{c_j}^{off}(t) ≤ 0, ∀u_i ∈ U
"""

import cvxpy as cp  # noqa: F401（约束表达式由 cvxpy 运算符构造）

NAME = "P1:Task-Fresh"


def build(ctx):
    constraints = []
    for i in range(ctx.I):
        constraints.append(
            ctx.D_bar_sen                       # D̄^sen
            + ctx.D_uav_off[i]                  # + D_{u_i}^{off}(t)
            - ctx.eta_share[i, :]               # - Σ_{j=1}^{J} η_{i,j}(t)
            @ ctx.D_cu_off                      #   · D_{c_j}^{off}(t)
            <= 0
        )
    return constraints
