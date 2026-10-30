from pathlib import Path
full = Path('output/pdf/MVDR_proof_hermitian_P0.tex').read_text(encoding='utf-8')
start = full.index(r'\section{MVDR 为什么不依赖尚未求得的发射波束}')
stop = full.index(r'\section{MVDR 接收波束的全局最优性证明}', start)
text = full[start:stop]
def change(old,new):
    global text
    assert old in text, old
    text=text.replace(old,new,1)
change(r'''记式\,(\ref{eq:denominator}) 括号中的矩阵为 $\mathbf Q_i(t)$。式\,(\ref{eq:receive-optimization}) 的分子为 $|\bm g_i^H(t)\bm a_r(\theta_i(t))|^2$，分母为 $\bm g_i^H(t)\mathbf Q_i(t)\bm g_i(t)$。为了联系分子和分母，在分子的内积中插入该矩阵的平方根及逆平方根，得到''', r'''为说明接下来的推导，记式~\eqref{eq:denominator} 括号中的 Hermitian 正定矩阵为 $\mathbf Q_i(t)$，即 $\mathbf Q_i(t)=\sum_{j=1}^{J}\eta_{i,j}(t)p_j(t)\bm h_{c_j,u_i}^{H}(t)\bm h_{c_j,u_i}(t)+\sigma^2\mathbf I$。式~\eqref{eq:receive-optimization} 是要求最大化的分式：其分子是 $|\bm g_i^H(t)\bm a_r(\theta_i(t))|^2$，分母是 $\sum_{j=1}^{J}\eta_{i,j}(t)p_j(t)|\bm g_i^H(t)\bm h_{c_j,u_i}^{H}(t)|^2+\sigma^2$。由式~\eqref{eq:denominator}，这个分母等于 $\bm g_i^H(t)\mathbf Q_i(t)\bm g_i(t)$。

为利用 Cauchy--Schwarz 不等式得到式~\eqref{eq:receive-optimization} 的上界，在分子内积 $\bm g_i^H(t)\bm a_r(\theta_i(t))$ 中插入单位矩阵 $\mathbf Q_i^{1/2}(t)\mathbf Q_i^{-1/2}(t)=\mathbf I$。其中 $\mathbf Q_i^{1/2}(t)$ 为 $\mathbf Q_i(t)$ 的正定平方根，$\mathbf Q_i^{-1/2}(t)$ 为该平方根的逆矩阵。利用 $\mathbf Q_i^{1/2}(t)$ 的 Hermitian 性，可得''')
change(r'''这里两个矩阵相乘为单位矩阵。对式\,(\ref{eq:sqrt-insertion}) 右侧的两个向量应用复数 Cauchy--Schwarz 不等式''', r'''这里相乘得到单位矩阵的是 $\mathbf Q_i^{1/2}(t)$ 与 $\mathbf Q_i^{-1/2}(t)$，即 $\mathbf Q_i^{1/2}(t)\mathbf Q_i^{-1/2}(t)=\mathbf I$。对式\,(\ref{eq:sqrt-insertion}) 右侧的两个向量应用复数 Cauchy--Schwarz 不等式''')
change(r'''式\,(\ref{eq:cs}) 右侧第一项恰好是式\,(\ref{eq:denominator}) 的分母；将式\,(\ref{eq:cs}) 两侧除以该正数，就得到式\,(\ref{eq:receive-optimization}) 的上界。Cauchy--Schwarz 不等式取等当且仅当式\,(\ref{eq:sqrt-insertion}) 中的两个向量线性相关，因此存在非零复数 $\alpha_i(t)$，使接收波束满足''', r'''式~\eqref{eq:cs} 右侧的第一项 $\bm g_i^H(t)\mathbf Q_i(t)\bm g_i(t)$ 等于式~\eqref{eq:receive-optimization} 的分母，等价关系由式~\eqref{eq:denominator} 给出。该分母为正，将式~\eqref{eq:cs} 两侧除以它，得到对任意单位范数接收波束均成立的不等式
\begin{equation}\label{eq:receive-bound}
\frac{|\bm g_i^H(t)\bm a_r(\theta_i(t))|^2}{\bm g_i^H(t)\mathbf Q_i(t)\bm g_i(t)}
\le\bm a_r^H(\theta_i(t))\mathbf Q_i^{-1}(t)\bm a_r(\theta_i(t)).
\end{equation}
右侧是式~\eqref{eq:receive-optimization} 的上界：任何满足约束的 $\bm g_i(t)$ 都不能使该分式超过它。上界本身不一定能达到，还需要证明存在可取等的接收波束。Cauchy--Schwarz 不等式取等当且仅当式~\eqref{eq:sqrt-insertion} 中的两个向量线性相关，即存在非零复数 $\alpha_i(t)$ 使 $\mathbf Q_i^{1/2}(t)\bm g_i(t)=\alpha_i(t)\mathbf Q_i^{-1/2}(t)\bm a_r(\theta_i(t))$。两侧左乘 $\mathbf Q_i^{-1/2}(t)$，得到''')
change(r'''其任意整体相位旋转也达到相同 SINR。''', r'''将式~\eqref{eq:mvdr} 代入式~\eqref{eq:receive-bound} 时等号成立，因此式~\eqref{eq:receive-bound} 的上界可达到，并且就是式~\eqref{eq:receive-optimization} 的最大值。其任意整体相位旋转也达到相同 SINR。''')
Path('tmp/mvdr_clarity/chapter2.tex').write_text(text,encoding='utf-8')
print('chapter 2 ready')
