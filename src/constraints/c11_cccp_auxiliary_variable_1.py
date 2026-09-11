"""约束 \eqref{CCCP:Auxiliary-Variable-1}（约束 \eqref{P2:Auxiliary-Variable-1} 在 x^(n) 处的 CCCP 线性化）：

    - ξ_1 log_2( Γ_i(t) ) - z_i(t) + ξ_1 log_2( Ψ_i^{(n)}(t) )
    + ξ_1 ξ_2 / ( ln2 · Ψ_i^{(n)}(t) ) · Tr( G_i(t) ( W_i(t) - W_i^{(n)}(t) ) ) ≤ 0, ∀u_i ∈ U

其中 Ψ_i^{(n)}(t) = ξ_2 Tr( G_i(t) W_i^{(n)}(t) ) + Γ_i(t)。

实现说明：为让 P5 只编译一次（cvxpy DPP 复用），把与 x^(n) 有关的量全部做成参数：
    M_psi_i = ξ_1 ξ_2 / ( ln2 · Ψ_i^{(n)} ) · G_i(t)        （实/虚部两个实参数）
    psi_sen_bias_i = - ξ_1 log_2(Γ_i) + ξ_1 log_2(Ψ_i^{(n)})
                     - ξ_1 ξ_2 / ( ln2 · Ψ_i^{(n)} ) · Tr( G_i W_i^{(n)} )
于是约束等价写为
    - z_i(t) + Re Tr( M_psi_i W_i(t) ) + psi_sen_bias_i ≤ 0
与原式逐项一致（详见 cccp_params.py）。
"""

from cccp_params import real_trace

NAME = "CCCP:Auxiliary-Variable-1"


def build(ctx):
    return [
        -ctx.z_aux_rate[i]
        + real_trace(ctx.ccp.M_psi_re[i], ctx.ccp.M_psi_im[i], ctx.W_sen_beam[i])
        + ctx.ccp.psi_sen_bias[i]
        <= 0
        for i in range(ctx.I)
    ]
