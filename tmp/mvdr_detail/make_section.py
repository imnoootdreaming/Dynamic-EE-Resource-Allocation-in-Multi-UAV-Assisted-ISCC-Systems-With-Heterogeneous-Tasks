from pathlib import Path
import re

root = Path('output/pdf/MVDR_proof_complete_P0.tex')
src = root.read_text(encoding='utf-8')
start = src.index(r'\subsection{代入论文原式，不省略物理变量}')
end = src.index(r'\section{MVDR 接收波束的全局最优性证明}', start)
body = src[start:end]
body = re.sub(r'^\\subsection\{[^\n]*\}\s*', '', body, flags=re.M)
labels = re.findall(r'\\label\{([^}]+)\}', body)
for label in sorted(labels, key=len, reverse=True):
    body = body.replace('{'+label+'}', '{detail:'+label+'}')
body = body.replace('Capon [1] 是相关的原始文献；Vorobyov [2, 第 2 节] 讨论最大 SINR、MVDR 和模型误差。',
    '经典无失真 MVDR 的问题定义及闭式解证明见附录~\\ref{app:classic}；相关文献为 Capon [1] 和 Vorobyov [2，第 2.2 节]。')
body = body.replace('由式\\,(\\ref{eq:A}) 直接得到', '任取 UAV $u_i$ 和时隙 $t$。由式\\,(\\ref{eq:A}) 直接得到', 1)
body = body.replace('上面的唯一方向结论针对目标增益非零、目标导向矢量非零的情况。',
    '上面的唯一方向结论针对目标增益非零、目标导向矢量非零的情况。')

head = r'''\subsection{MVDR 接收波束与最大感知 SINR 的详细证明}

\noindent\textbf{命题。}任取 UAV $u_i$ 和时隙 $t$，固定 UAV 位置、目标、频谱共享关系、CU 发射功率、相关信道及感知发射波束。设 $\sigma^2>0$、$\eta_{i,j}(t)p_j(t)\ge0$、$\bm a_r(\theta_i(t))\ne\bm0$，且感知信道满足式~\eqref{eq:A}。则式~\eqref{eq:mvdr} 所示单位范数接收波束能够达到该时隙的最大感知 SINR。

\medskip
\noindent\textbf{证明。}
'''

extra = r'''
由上述上界及取等条件，最大感知 SINR 的值为
\begin{align}\label{detail:eq:maximum-value}
&\max_{\|\bm g_i(t)\|^2=1}\operatorname{SINR}_i(t)\notag\\
&=\xi_0\rho d_i^{-4}(t)
\left|\bm a_t^H(\theta_i(t))\bm w_i(t)\right|^2
\bm a_r^H(\theta_i(t))
\left(\sum_{j=1}^J\eta_{i,j}(t)p_j(t)
\bm h_{c_j,u_i}^{H}(t)\bm h_{c_j,u_i}(t)
+\sigma^2\mathbf I\right)^{-1}\bm a_r(\theta_i(t)).
\end{align}
实际计算时，可先求解线性方程组
\begin{equation}\label{detail:eq:solve}
\left(\sum_{j=1}^J\eta_{i,j}(t)p_j(t)
\bm h_{c_j,u_i}^{H}(t)\bm h_{c_j,u_i}(t)
+\sigma^2\mathbf I\right)\bm x_i(t)=\bm a_r(\theta_i(t)),
\quad
\bm g_i^{\rm MVDR}(t)=\frac{\bm x_i(t)}{\|\bm x_i(t)\|}.
\end{equation}
此方式不需显式计算矩阵逆。综上，式~\eqref{eq:mvdr} 是该时隙最大感知 SINR 的一个单位范数接收波束。命题得证。\hfill$\square$

'''
Path('tmp/mvdr_detail').mkdir(parents=True,exist_ok=True)
Path('tmp/mvdr_detail/section31.tex').write_text(head+body+extra,encoding='utf-8')
print('Wrote 3.1 with',len(labels),'uniquely renamed equations')
