from pathlib import Path
src=Path('output/pdf/MVDR_proof_rank_explained_P0.tex').read_text(encoding='utf-8')
start=src.index(r'\section{MVDR 接收波束的全局最优性证明}')
end=src.index(r'\section*{参考与可核查来源}',start)
part=src[start:end]
def replace(old,new):
    global part
    assert old in part, old
    part=part.replace(old,new,1)
replace('其中，位置、匹配和 CU 功率取该全局最优方案对应的值，感知发射波束与之联合调整。\n','')
replace(r'任取 UAV $u_i$ 和时隙 $t$，固定 UAV 位置、目标、频谱共享关系、CU 发射功率、相关信道及感知发射波束。感知发射波束 $\bm w_i(t)$ 与接收波束 $\bm g_i(t)$ 均属于 $\mathbb C^{N\times1}$，且 $\|\bm g_i(t)\|^2=1$。考虑以下原始感知信道模型和感知 SINR：', r'''任取 UAV $u_i$ 和时隙 $t$，固定 $P0$ 中除 $\bm g_i(t)$ 以外的所有优化变量，即各时隙的
\begin{align*}
&\bm w_{i'}(t'),\ \bm b_{i'}(t'),\ \bm q_{u_{i'}}(t'),\ D_{u_{i'}}^{\rm off}(t'),\ f_{u_{i'}}(t'),\\
&p_{j'}(t'),\ D_{c_{j'}}^{\rm off}(t'),\ f_{c_{j'}}(t'),\ \eta_{i',j'}(t'),\\
&\bm g_{i'}(t')\quad\text{其中 }(i',t')\ne(i,t).
\end{align*}
以上指标 $i',j',t'$ 遍历各 UAV、CU 和时隙。考虑以下原始感知信道模型和感知 SINR：''')
replace(r'根据定理的假设，$\sigma^2>0$、$\eta_{i,j}(t)p_j(t)\ge0$ 且 $\bm a_r(\theta_i(t))\ne\bm0$。','')
replace(r'考虑固定 UAV 位置、频谱共享变量 $\eta_{i,j}(t)$、CU 功率 $p_j(t)$ 和感知发射波束 $\bm w_i(t)$ 的情况。式\,(\ref{detail:eq:factored}) 的分子中，',r'在上述其他优化变量固定的条件下，式\,(\ref{detail:eq:factored}) 的分子中，')
replace(r'若式\,(\ref{detail:eq:tx-factor}) 为零，所有接收波束的感知 SINR 均为零，另见本节的退化情形。',r'若式\,(\ref{detail:eq:tx-factor}) 为零，所有接收波束的感知 SINR 均为零。')
replace(r'其任意整体相位旋转也达到相同 SINR。经典无失真 MVDR 采用 $\bm g_i^H(t)\bm a_r(\theta_i(t))=1$ 的幅度归一化；式\,(\ref{detail:eq:mvdr}) 采用论文要求的单位范数，方向相同，约束不同。','')
replace(r'\textbf{退化情形。}若式\,(\ref{detail:eq:tx-factor}) 为零，包括 $\bm a_t^H(\theta_i(t))\bm w_i(t)=0$ 的情形，则回波分子为零，所有接收方向的 SINR 都为零；在 $\epsilon>0$ 时该方案不可行。上面的唯一方向结论针对目标增益非零、目标导向矢量非零的情况。','')
old=r'''接着构造新的感知发射波束 $\bm w'_i(t)=c_i(t)\tilde{\bm w}_i(t)$，其中
\begin{equation}\label{eq:new-scale}
c_i(t)=\begin{cases}
\sqrt{\tilde s_i(t)/\hat s_i(t)},&\hat s_i(t)>0,\\
0,&\hat s_i(t)=0.
\end{cases}
\end{equation}
由式~\eqref{eq:new-ineq}，$0\le c_i(t)\le1$；当 $\hat s_i(t)=0$ 时，$\tilde s_i(t)=0$。因此新方案的感知 SINR $s'_i(t)$ 与感知发射功率满足'''
new=r'''由式~\eqref{eq:new-ineq}，存在 $c_i(t)\in[0,1]$，使得
\begin{equation}\label{eq:new-scale}
c_i^2(t)\hat s_i(t)=\tilde s_i(t).
\end{equation}
这是因为 $c_i^2(t)\hat s_i(t)$ 在 $c_i(t)\in[0,1]$ 上连续取遍 $[0,\hat s_i(t)]$；若 $\hat s_i(t)=0$，则 $\tilde s_i(t)=0$，可取 $c_i(t)=0$。
构造新的感知发射波束 $\bm w'_i(t)=c_i(t)\tilde{\bm w}_i(t)$。因此新解的感知 SINR $s'_i(t)$ 与感知发射功率满足'''
replace(old,new)
part=part.replace('方案','解')
src=src[:start]+part+src[end:]
Path('output/pdf/MVDR_proof_refined_P0.tex').write_text(src,encoding='utf-8')
print('saved')
