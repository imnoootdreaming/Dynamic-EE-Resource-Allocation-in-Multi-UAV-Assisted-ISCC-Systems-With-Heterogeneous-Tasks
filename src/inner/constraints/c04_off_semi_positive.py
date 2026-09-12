"""约束 \eqref{P2:Off-Semi-Positive}：

    -B_i(t) ⪯ 0, ∀u_i ∈ U
"""

NAME = "P2:Off-Semi-Positive"


def build(ctx):
    return [ctx.B_off_beam[i] >> 0 for i in range(ctx.I)]
