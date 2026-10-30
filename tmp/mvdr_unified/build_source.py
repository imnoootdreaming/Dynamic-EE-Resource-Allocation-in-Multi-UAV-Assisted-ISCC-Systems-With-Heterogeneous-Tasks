from pathlib import Path

src = Path('output/pdf/MVDR_proof_selfcontained_P0.tex').read_text(encoding='utf-8')
start = src.index(r'\section{MVDR 接收波束的全局最优性证明}')
heading = r'\subsection{MVDR 接收波束与最大感知 SINR 的详细证明}'
at_heading = src.index(heading, start)
theorem = r'''\noindent\textbf{定理。}考虑论文中的系统能耗最小化问题 $P0$。假设 $P0$ 存在全局最优解，且其感知信道为下述式~\eqref{detail:eq:channel} 所给的秩一模型。对所有 UAV 和时隙，设 $\sigma^2>0$、$p_j(t)\ge0$、$\bm a_r(\theta_i(t))\ne\bm0$、$\omega_2\ge0$，感知发射、卸载发射和接收波束均属于 $\mathbb C^{N\times1}$。感知发射波束允许乘以 $[0,1]$ 内的系数，且不存在论文所列以外会阻止该缩放的波束约束。

则 $P0$ 至少存在一个全局最优解，其每个 UAV 在每个时隙的接收波束可以取为
\begin{equation}\label{eq:unified-theorem}
\bm g_i^{\rm MVDR}(t)=
\frac{\left(\displaystyle\sum_{j=1}^J\eta_{i,j}(t)p_j(t)
\bm h_{c_j,u_i}^{H}(t)\bm h_{c_j,u_i}(t)+\sigma^2\mathbf I\right)^{-1}
\bm a_r(\theta_i(t))}
{\left\|\left(\displaystyle\sum_{j=1}^J\eta_{i,j}(t)p_j(t)
\bm h_{c_j,u_i}^{H}(t)\bm h_{c_j,u_i}(t)+\sigma^2\mathbf I\right)^{-1}
\bm a_r(\theta_i(t))\right\|}.
\end{equation}
其中，位置、匹配和 CU 功率取该全局最优方案对应的值，感知发射波束与之联合调整。

\medskip
\noindent\textbf{证明。}分两步证明。首先从感知信道及 SINR 的原始表达式出发，推导式~\eqref{eq:unified-theorem}；然后构造可行解，证明该接收波束可用于 $P0$ 的全局最优方案。

'''
src = src[:at_heading] + theorem + src[at_heading:]
src = src.replace(heading, r'\subsection{第一步：推导 MVDR 接收波束}', 1)
src = src.replace(r'\noindent\textbf{命题。}任取 UAV', r'任取 UAV', 1)
old = r'设 $\sigma^2>0$、$\eta_{i,j}(t)p_j(t)\ge0$、$\bm a_r(\theta_i(t))\ne\bm0$。则式~\eqref{detail:eq:mvdr} 所示单位范数接收波束能够达到该时隙最大的感知 SINR。'
new = r'根据定理的假设，$\sigma^2>0$、$\eta_{i,j}(t)p_j(t)\ge0$ 且 $\bm a_r(\theta_i(t))\ne\bm0$。下面推导最大感知 SINR 对应的接收波束。'
assert old in src
src = src.replace(old, new, 1)
src = src.replace('\\medskip\n\\noindent\\textbf{证明。}\n任取 UAV', '任取 UAV', 1)
src = src.replace(r'此方式不需显式计算矩阵逆。综上，式~\eqref{detail:eq:mvdr} 是该时隙最大感知 SINR 的一个单位范数接收波束。命题得证。\hfill$\square$', r'此方式不需显式计算矩阵逆。综上，式~\eqref{detail:eq:mvdr} 即为定理中的式~\eqref{eq:unified-theorem}，也是该时隙最大感知 SINR 的一个单位范数接收波束。', 1)
old_header = r'\subsection{MVDR 接收波束在 P0 中的全局最优性}'
begin = src.index(old_header, start)
proof = src.index(r'\noindent\textbf{证明。}任取', begin)
src = src[:begin] + r'\subsection{第二步：证明 MVDR 接收波束可用于 P0 的全局最优方案}' + '\n\n' + '任取' + src[proof+len(r'\noindent\textbf{证明。}任取') :]
src = src.replace('由上一命题的推导', '由第一步的推导', 1)
assert '上一命题' not in src[start:src.index(r'\section*{参考与可核查来源}',start)]
Path('output/pdf/MVDR_proof_unified_P0.tex').write_text(src,encoding='utf-8')
print('Unified theorem and two-step proof written')
