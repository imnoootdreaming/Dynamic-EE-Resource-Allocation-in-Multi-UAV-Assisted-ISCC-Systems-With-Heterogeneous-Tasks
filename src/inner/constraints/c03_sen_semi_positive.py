"""约束 \eqref{P2:Sen-Semi-Positive}：

    -W_i(t) ⪯ 0, ∀u_i ∈ U
"""

NAME = "P2:Sen-Semi-Positive"


def build(ctx):
    return [ctx.W_sen_beam[i] >> 0 for i in range(ctx.I)]
