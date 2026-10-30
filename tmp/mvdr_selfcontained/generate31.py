from pathlib import Path

src = Path('output/pdf/MVDR_proof_detailed_P0.tex').read_text(encoding='utf-8')
i = src.index(r'\subsection{MVDR 接收波束与最大感知 SINR 的详细证明}')
j = src.index(r'\subsection{MVDR 接收波束在 P0 中的全局最优性}', i)
s = src[i:j]
s = s.replace(r'\eqref{eq:A}', r'\eqref{detail:eq:channel}')
s = s.replace(r'\ref{eq:A}', r'\ref{detail:eq:channel}')
s = s.replace(r'\eqref{eq:mvdr}', r'\eqref{detail:eq:mvdr}')

start = s.index(r'\noindent\textbf{命题。}')
end = s.index(r'\begin{align}', start)
header = r'''\noindent\textbf{命题。}任取 UAV $u_i$ 和时隙 $t$，固定 UAV 位置、目标、频谱共享关系、CU 发射功率、相关信道及感知发射波束。感知发射波束 $\bm w_i(t)$ 与接收波束 $\bm g_i(t)$ 均属于 $\mathbb C^{N\times1}$，且 $\|\bm g_i(t)\|^2=1$。考虑以下原始感知信道模型和感知 SINR：
\begin{equation}\label{detail:eq:channel}
\mathbf A(\theta_i(t))
=\sqrt{\xi_0\rho d_i^{-4}(t)}\,
\bm a_r(\theta_i(t))\bm a_t^H(\theta_i(t)),
\end{equation}
\begin{equation}\label{detail:eq:original-sinr}
\operatorname{SINR}_i(t)=
\frac{\left|\bm g_i^H(t)\mathbf A(\theta_i(t))\bm w_i(t)\right|^2}
{\displaystyle\sum_{j=1}^J\eta_{i,j}(t)p_j(t)
\left|\bm g_i^H(t)\bm h_{c_j,u_i}^{H}(t)\right|^2+\sigma^2}.
\end{equation}
其中 $\bm a_r$、$\bm a_t$ 是接收和发射导向向量，$\bm h_{c_j,u_i}(t)\in\mathbb C^{1\times N}$ 为 CU 到 UAV 的信道，$\eta_{i,j}(t)$ 和 $p_j(t)$ 分别为频谱共享指标和 CU 发射功率；$d_i(t)>0$ 是 UAV 到目标的距离，$\xi_0\ge0$、$\rho\ge0$ 为信道参数，$\sigma^2$ 是噪声功率。设 $\sigma^2>0$、$\eta_{i,j}(t)p_j(t)\ge0$、$\bm a_r(\theta_i(t))\ne\bm0$。则式~\eqref{detail:eq:mvdr} 所示单位范数接收波束能够达到该时隙最大的感知 SINR。

\medskip
\noindent\textbf{证明。}
任取 UAV $u_i$ 和时隙 $t$。首先将式~\eqref{detail:eq:channel} 代入回波幅值，并利用标量乘法的结合律，逐步得到
'''
s = s[:start] + header + s[end:]
old = r'''&\left|\bm g_i^H(t)\mathbf A(\theta_i(t))\bm w_i(t)\right|^2\notag\\
&=\xi_0\rho d_i^{-4}(t)'''
new = r'''&\left|\bm g_i^H(t)\mathbf A(\theta_i(t))\bm w_i(t)\right|^2\notag\\
&=\left|\sqrt{\xi_0\rho d_i^{-4}(t)}\,
\bm g_i^H(t)\bm a_r(\theta_i(t))\bm a_t^H(\theta_i(t))\bm w_i(t)\right|^2\notag\\
&=\xi_0\rho d_i^{-4}(t)'''
assert old in s
s = s.replace(old, new, 1)
s = s.replace('因此原 SINR 完整地等于',r'将上式代入原始 SINR 定义~\eqref{detail:eq:original-sinr}，得到')
s = s.replace(r'\eqref{eq:mvdr}', r'\eqref{detail:eq:mvdr}')
Path('tmp/mvdr_selfcontained').mkdir(parents=True,exist_ok=True)
Path('tmp/mvdr_selfcontained/section31.tex').write_text(s,encoding='utf-8')
print('written',len(s))
