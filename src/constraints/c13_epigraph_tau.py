"""DCP 辅助约束：目标函数中 ( z_i(t) + (f_{u_i}(t))² )² 的上镜图改写。

    z_i(t) + ( f_{u_i}(t) )² ≤ τ_i, ∀u_i ∈ U

τ_i ≥ 0 已由变量定义（nonneg=True）保证，于是
( z_i(t) + (f_{u_i}(t))² )² = (tau_scale · τ_i)²，目标函数中用 τ_i² 代替。

注：τ_i 以 tau_scale = freq_scale² 为度量单位，f_{u_i}(t) 以 freq_scale 为度量单位，
上式等价于两端同除以 tau_scale，仅为改善求解器数值条件（f² 约 10^13 量级）。
"""

import cvxpy as cp

NAME = "DCP-Auxiliary:Epigraph-Tau"


def build(ctx):
    return [
        ctx.z_aux_rate[i] / ctx.tau_scale                        # z_i(t) / tau_scale
        + cp.square(ctx.f_uav_freq[i] / ctx.freq_scale)          # + ( f_{u_i}(t) / freq_scale )²
        - ctx.tau_zf[i]                                          # - τ_i
        <= 0
        for i in range(ctx.I)
    ]
