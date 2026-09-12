"""约束 \eqref{P3:CU-Frequency-Always-Positive}：

    D_{c_j}^{off}(t) - D^max_j(t) < 0, ∀c_j ∈ C

论文为严格不等式，数值实现中以 1e-10 的容差写成 ≤，
以保证 f_{c_j}(t) = C_j(t) L_j(t) / (D^max_j(t) - D_{c_j}^{off}(t)) 恒为正。
"""

NAME = "P3:CU-Frequency-Always-Positive"

STRICT_TOL = 1e-10


def build(ctx):
    constraints = []
    for j in range(ctx.J):
        constraints.append(
            ctx.D_cu_off[j]                 # D_{c_j}^{off}(t)
            - ctx.D_max_cu[j] + STRICT_TOL  # - D^max_j(t)（严格小于，取 1e-10 容差）
            <= 0
        )
    return constraints
