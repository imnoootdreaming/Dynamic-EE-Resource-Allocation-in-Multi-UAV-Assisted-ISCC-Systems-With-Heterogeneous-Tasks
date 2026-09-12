"""DCP 辅助约束（原上镜图约束 \eqref{DCP-Auxiliary:Epigraph-Tau} 的乘积形式改写）：

    ( f_{u_i}(t) / freq_scale )² ≤ ũ_i(t), ∀u_i ∈ U

原形式为 τ_i ≥ ( z_i + f_{u_i}² ) / tau_scale，配合目标里的 (tau_scale·τ_i)²
一起使用；那种写法会把 z 与 f² 相加后再平方，而 z ≈ 450、f² ≈ 3.6e14（相差 12 个数量级），
平方后 f⁴ 更是把差距放大到 24 个数量级，使目标中出现 ~1e4 量级的大项相消，MOSEK 的相对
容差因此失效、迭代不再单调（详见 objective.py 中 ① / ⑤ 的说明）。

这里改为只对 f 做归一化平方（SOC 形式），把 ũ_i 作为独立的、量级与 z 相当的辅助变量，
感知能耗以乘积 ω₁κC^sen·freq_scale²·z_i·ũ_i 的形式进入目标，从根上消除大数相消。
两者在数学上等价：原式的 τ_scale·τ_i ≥ z_i + f_{u_i}² 与 ũ 的关系是
τ_i = z_i/tau_scale + ũ_i，而目标中的 (z+f²)² 展开后恰好等价于对乘积 z·f² 的 CCCP 上界。
"""

import cvxpy as cp

NAME = "DCP-Auxiliary:Freq-Squared-SOC"


def build(ctx):
    return [
        cp.square(ctx.f_uav_freq[i] / ctx.freq_scale) - ctx.u_freq_sq[i] <= 0
        for i in range(ctx.I)
    ]
