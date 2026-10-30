from pathlib import Path
s=Path('output/pdf/MVDR_proof_hermitian_P0.tex').read_text(encoding='utf8')
start=s.index(r'\section{MVDR 接收波束的全局最优性证明}')
end=s.index(r'\section*{参考与可核查来源}',start)
part=s[start:end]
def sub(old,new):
 global part
 assert old in part, old[:100]
 part=part.replace(old,new,1)
old=r'''由于 $\sigma^2>0$ 且 $\eta_{i,j}(t)p_j(t)\ge0$，对任意非零向量 $\bm v\in\mathbb C^{N\times1}$，有
\begin{samepage}
\begin{align}
&\bm v^H\!\left(\sum_{j=1}^J\eta_{i,j}(t)p_j(t)
\bm h_{c_j,u_i}^{H}(t)\bm h_{c_j,u_i}(t)+\sigma^2\mathbf I\right)\!\bm v\notag\\
&=\sum_{j=1}^J\eta_{i,j}(t)p_j(t)
\left|\bm h_{c_j,u_i}(t)\bm v\right|^2+\sigma^2\|\bm v\|^2>0.
\end{align}
\end{samepage}
\begin{samepage}
因此，式\,(\ref{detail:eq:denominator}) 括号中的矩阵为 Hermitian 正定矩阵，其正定平方根及逆平方根均存在。为了把分子和这个分母联系起来，在分子的内积中插入该矩阵的平方根及逆平方根，得到'''
new=r'''由于 $\sigma^2>0$ 且 $\eta_{i,j}(t)p_j(t)\ge0$，对任意满足 $\|\bm g_i(t)\|=1$ 的接收波束，有
\begin{samepage}
\begin{align}
&\bm g_i^H(t)\!\left(\sum_{j=1}^J\eta_{i,j}(t)p_j(t)
\bm h_{c_j,u_i}^{H}(t)\bm h_{c_j,u_i}(t)+\sigma^2\mathbf I\right)\!\bm g_i(t)\notag\\
&=\sum_{j=1}^J\eta_{i,j}(t)p_j(t)
\left|\bm h_{c_j,u_i}(t)\bm g_i(t)\right|^2+\sigma^2>0.
\end{align}
\end{samepage}
由于这里的 $\bm g_i(t)$ 是任意单位向量，而任何非零向量都能缩放为单位向量，故式~\eqref{detail:eq:denominator} 括号中的矩阵对所有非零向量的二次型均严格为正。同时，每个 $\bm h_{c_j,u_i}^{H}(t)\bm h_{c_j,u_i}(t)$ 和 $\mathbf I$ 都是 Hermitian 矩阵，且系数为实数，因此该矩阵是 Hermitian 正定矩阵。

为简化文字，记这个矩阵为 $\mathbf Q_i(t)=\sum_{j=1}^{J}\eta_{i,j}(t)p_j(t)\bm h_{c_j,u_i}^{H}(t)\bm h_{c_j,u_i}(t)+\sigma^2\mathbf I$。它的正定平方根 $\mathbf Q_i^{1/2}(t)$ 与逆平方根 $\mathbf Q_i^{-1/2}(t)$ 存在，且 $\mathbf Q_i^{1/2}(t)\mathbf Q_i^{-1/2}(t)=\mathbf I$。

式~\eqref{detail:eq:receive-optimization} 中的分子是 $|\bm g_i^H(t)\bm a_r(\theta_i(t))|^2$，分母是 $\sum_{j=1}^{J}\eta_{i,j}(t)p_j(t)|\bm g_i^H(t)\bm h_{c_j,u_i}^H(t)|^2+\sigma^2$，根据式~\eqref{detail:eq:denominator}，分母也可写为 $\bm g_i^H(t)\mathbf Q_i(t)\bm g_i(t)$。为将分子与该分母联系起来，在分子的内积 $\bm g_i^H(t)\bm a_r(\theta_i(t))$ 中插入 $\mathbf Q_i^{1/2}(t)\mathbf Q_i^{-1/2}(t)=\mathbf I$，得到
\begin{samepage}'''
sub(old,new)
sub(r'这里两个矩阵相乘为单位矩阵。对式\,(\ref{detail:eq:sqrt-insertion}) 右侧的两个向量应用复数 Cauchy--Schwarz 不等式', r'根据 $\mathbf Q_i^{1/2}(t)\mathbf Q_i^{-1/2}(t)=\mathbf I$，对式\,(\ref{detail:eq:sqrt-insertion}) 右侧的两个向量应用复数 Cauchy--Schwarz 不等式')
old=r'''式\,(\ref{detail:eq:cs}) 右侧第一项恰好是式\,(\ref{detail:eq:denominator}) 的分母；将式\,(\ref{detail:eq:cs}) 两侧除以该正数，就得到式\,(\ref{detail:eq:receive-optimization}) 的上界。Cauchy--Schwarz 不等式取等当且仅当式\,(\ref{detail:eq:sqrt-insertion}) 中的两个向量线性相关；将这一条件两侧乘以对应的逆平方根，可得 $\bm g_i(t)$ 与下式同方向，即相差非零复数倍：
\begin{equation}\label{detail:eq:direction}
\left(\sum_{j=1}^J\eta_{i,j}(t)p_j(t)
\bm h_{c_j,u_i}^{H}(t)\bm h_{c_j,u_i}(t)+\sigma^2\mathbf I\right)^{-1}
\bm a_r(\theta_i(t)).
\end{equation}
由此，满足论文单位范数要求的一个解为'''
new=r'''式\,(\ref{detail:eq:cs}) 右侧第一项 $\bm g_i^H(t)\mathbf Q_i(t)\bm g_i(t)$ 是式\,(\ref{detail:eq:receive-optimization}) 的分母；式\,(\ref{detail:eq:denominator}) 则是将该分母写成二次型的等式。将式\,(\ref{detail:eq:cs}) 两侧除以这个正数，可得
\begin{equation}\label{detail:eq:upper-bound}
\frac{|\bm g_i^H(t)\bm a_r(\theta_i(t))|^2}{\bm g_i^H(t)\mathbf Q_i(t)\bm g_i(t)}
\le\bm a_r^H(\theta_i(t))\mathbf Q_i^{-1}(t)\bm a_r(\theta_i(t)).
\end{equation}
这表示右侧是式\,(\ref{detail:eq:receive-optimization}) 目标值的上界；只有当某个可行接收波束能够达到它时，它才是该问题的最大值。

Cauchy--Schwarz 不等式取等当且仅当式\,(\ref{detail:eq:sqrt-insertion}) 中的两个向量线性相关，即存在非零复数 $\alpha_i(t)$，使 $\mathbf Q_i^{1/2}(t)\bm g_i(t)=\alpha_i(t)\mathbf Q_i^{-1/2}(t)\bm a_r(\theta_i(t))$。两侧左乘 $\mathbf Q_i^{-1/2}(t)$，得到
\begin{equation}\label{detail:eq:direction}
\bm g_i(t)=\alpha_i(t)\mathbf Q_i^{-1}(t)\bm a_r(\theta_i(t)).
\end{equation}
因 $\mathbf Q_i(t)$ 正定且 $\bm a_r(\theta_i(t))\ne\bm0$，$\mathbf Q_i^{-1}(t)\bm a_r(\theta_i(t))\ne\bm0$。取 $\alpha_i(t)=1/\|\mathbf Q_i^{-1}(t)\bm a_r(\theta_i(t))\|$，即可使 $\|\bm g_i(t)\|=1$ 且式\,(\ref{detail:eq:upper-bound}) 取等。因此，上界能够达到，就是式\,(\ref{detail:eq:receive-optimization}) 的最大值，其一个单位范数接收波束为'''
sub(old,new)
old=r'''任取 $P0$ 的一个可行解 $\tilde{\bm\zeta}$，保留其中所有时隙的轨迹、频谱匹配、CU 功率、卸载波束、卸载时长及计算频率。对任意 UAV $u_i$ 和时隙 $t$，记干扰加噪声矩阵为
\begin{equation}\label{eq:new-cov}
\mathbf R_i(t)=\sum_{j=1}^{J}\eta_{i,j}(t)p_j(t)\bm h_{c_j,u_i}^{H}(t)\bm h_{c_j,u_i}(t)+\sigma^2\mathbf I.
\end{equation}
由第一步的推导，$\mathbf R_i(t)\succ0$，并且单位范数向量
\begin{equation}\label{eq:new-mvdr}
\bm g_i^{\rm MVDR}(t)=\frac{\mathbf R_i^{-1}(t)\bm a_r(\theta_i(t))}{\|\mathbf R_i^{-1}(t)\bm a_r(\theta_i(t))\|}
\end{equation}
在保持其他变量及感知发射波束不变时，使该 UAV 的感知 SINR 达到最大值。即使原发射目标增益为零，该向量仍为最大 SINR 的一个解。'''
new=r'''任取 $P0$ 的一个可行解 $\tilde{\bm\zeta}$，保留其中所有时隙的轨迹、频谱匹配、CU 功率、卸载波束、卸载时长及计算频率。对任意 UAV $u_i$ 和时隙 $t$，采用第一步的式~\eqref{detail:eq:mvdr} 所得接收波束 $\bm g_i^{\rm MVDR}(t)$。在保持该可行解的其他变量和感知发射波束不变时，该接收波束使感知 SINR 达到最大值。即使原发射目标增益为零，该向量仍为最大 SINR 的一个解。'''
sub(old,new)
Path('tmp/mvdr_clarity/chapter3.tex').write_text(part,encoding='utf8')
print('written',len(part))
