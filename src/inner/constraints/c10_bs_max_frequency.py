"""约束 \eqref{P3:BS-Max-Frequency}：

    Σ_{u_i ∈ U} f_{u_i}(t)
    + Σ_{c_j ∈ C} C_j(t) L_j(t) / (D^max_j(t) - D_{c_j}^{off}(t))
    - F_max ≤ 0

注：本行整体除以 freq_scale（即把频率量的单位由 Hz 换为 MHz）后交给 MOSEK，
仅为改善数值条件（F_max 为 10^10 量级），与原式两端同除以同一常数，数学上完全等价。
"""

import cvxpy as cp

NAME = "P3:BS-Max-Frequency"


def build(ctx):
    sum_freq = cp.sum(ctx.f_uav_freq)                   # Σ_{u_i ∈ U} f_{u_i}(t)
    for j in range(ctx.J):
        sum_freq = sum_freq + (
            ctx.C_cu_cycles[j] * ctx.L_cu_task[j]       # C_j(t) L_j(t)
            * cp.inv_pos(ctx.D_max_cu[j] - ctx.D_cu_off[j])   # / (D^max_j(t) - D_{c_j}^{off}(t))
        )
    return [(sum_freq - ctx.F_max) / ctx.freq_scale <= 0]     # - F_max ≤ 0
