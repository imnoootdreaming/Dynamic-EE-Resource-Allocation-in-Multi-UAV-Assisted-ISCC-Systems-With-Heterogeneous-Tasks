# 建议修改及验收顺序（仅方案，未修改生产代码）

本建议依据本次5804更新的训练曲线、当前源码和U53既有负结果；新增方法尚未实测收益。各项单独比较，不能同时改动后将收益归因给其中一项。

## 1. 为波束单独控制探索，并补充优化诊断

- `configs/madrl_args.py`：增加可选的beam专属熵系数和退火参数；兼容默认应复现现有连续熵系数0.05。
- `MHBPPO_agent.py`：在当前前向中区分beam 80维与其余22维的entropy，损失写作 `L_surrogate - beta_beam*H_beam - beta_other*H_other - beta_discrete*H_discrete`。原始log-prob仍按活动维求和，old/new概率合同不变；不能把log-prob的sum直接改成mean。若扩展`evaluate_actions`，应保留原默认六元组接口，避免破坏sampler/BC及初始化回归。
- 首轮只比较原0.05与beam逐步降至0.005（也可事前选定0.01），其余连续0.05、离散0.01保留。这是候选实验点，不是最优参数。降低熵允许已学方向集中，但不能凭空建立复相位关系，也不能把raw向量整体缩幅当作物理方向改善。
- 诊断至少记录每头解析 `KL(old_dist || new_dist)`、连续组/离散组ratio分位及clip fraction、PPO与熵梯度范数、critic explained variance。现有signed sampled old/new联合log-prob差不能替代这些指标，也不能因其单次为负就判定训练错误。
- 解析KL应在update开始用同批已归一化states缓存旧Beta参数/Categorical概率并detach，更新后用同一states的分布计算，不额外抽动作、不改变RNG；不能从四位小数的文本POLICY反推分布。末500 rollout的beam熵损失项约+0.01282，nonbeam约+0.07211，因此本次没有证据说beam熵loss数值占优或已经证明熵梯度主导。
- 验收固定seed下原兼容配置的action/log-prob/RNG/梯度一致，再检查实际抽样的方向增益、干扰和后段optimal。仅分布std变小不算成功。

## 2. 若方向仍学不出，加入在线波束辅助训练

U53已有从同一10651来源出发的500/500真实更新：预热分支的初期gain约0.170，末100约0.100，末100 optimal 4.35%，原组5.95%。该历史配置不等于本次N2冷启动，但足以否定“再做一次相同单次预热就会持续加速”的无依据承诺。应测试在线保持机制，而非重做同一实验。

在原状态/80维beam动作/单位范数映射下，可用规范相位的方向教师建立初始方向，再以当前rollout对应的物理量持续辅助学习。候选损失来自本项目感知必要裕度，而非整系统能效最优解：

\[
B_i(g)=\frac{g^H S_i g}{g^H R_i g},\quad
L_{beam}=\frac1I\sum_i\left[1-\frac{B_i(g_{\theta,i})}{B_i(g_i^*)}\right].
\]

其中 `S=A A^H`，按本项目`dot(g,h)`约定 `R=sigma² I + Σ eta*p*h* h^T`；在当前秩1目标模型下，固定几何、匹配和功率时 `g*=normalize(R^-1 a)`。它只最大化这一条件下的感知必要裕度。若精确使用同一S/R的最大值，上式取值[0,1]；不保证完整内层可行或长期能效最优。

实施要求：

1. 原PPO概率/奖励保持，辅助损失权重独立记录并消融；网络实际训练目标已增加，不能声称算法未改变。
2. 当前g头只看s，看不到同次新采样的匹配/功率。最小实现优化对这些其他动作抽样条件的平均物理目标；不能要求同一s精确拟合多个冲突的条件教师。若改成`pi(g|s,eta,p)`，则另行修改采样顺序及条件old/new log-prob合同。
3. 用当前Beta的`rsample()`及Torch同源映射建立计算图，不能把日志中的NumPy Q/M常量直接加到loss期待梯度。不得只归一化接近零的raw均值；实际抽样必须验收。
4. 物理量与移动前信道/目标的现有时序一致；S/R共同缩放保留商，不能用绝对1e-8淹没约1e-14的噪声项。
5. 实虚部教师监督须规范化教师的全局相位；物理商本身相位不变，不必另行约束当前抽样g的全局相位。若更新共享干，应加非beam策略KL保护，或另测已存在的dedicated分支，避免把架构与辅助训练的收益混在一起。PPG原论文仅为辅助阶段+KL的安排提供依据，未验证本项目波束损失。

## 3. 独立实施可证明不可行的求解前筛查，以提高吞吐

`reward/cccp_bridge.py`已在创建ctx后计算感知指标，但随后仍无条件`run_pc3p(ctx)`。`reward/solver_status.py`已有保守的`sensing_infeasibility_certificate(ctx)`，当前只在solver返回unknown后兜底。可将同源证书作为显式可选的求解前筛查：

\[
\operatorname{Tr}(GW)\le\lambda_{max}(G)\operatorname{Tr}(W)
\le\lambda_{max}(G)P_{max}<\epsilon\Gamma
\quad\Rightarrow\quad\text{当前动作不可行}.
\]

只在有限合法输入且超过保守相对容差时跳过该次CCCP；未获证书、边界值或M>=1均继续原求解。状态来源应明确标为`analytic_precheck`，raw solver状态为未执行，不能伪造MOSEK返回。保留原无解奖励、Q、transition、诊断可用性和错误可观测性，不做动作投影或放宽约束。重复固定动作核对两路奖励/状态、必要边界与真实求解调用次数，再比较端到端吞吐。

当前Fusion已有参数化模型缓存，不能把“增加模型缓存”当作尚未实现的方案。也不建议优先放宽求解精度或减少CCCP迭代，因为那会混入内层质量变化。证书可过滤的样本比例不等于墙钟提速比例；实际收益必须由保留日志的失败/通过调用耗时及短基准确认。

## 4. 比较协议与暂不优先的修改

- 主指标：原completion/optimal、slot8–39覆盖、失败Q、实际抽样gain与干扰、原奖励、同墙钟收益。条件于成功样本的能耗必须同时报告成功覆盖，不能直接当作总体能效改善。
- 学习对照至少并列相同新样本量与相同墙钟，多个动作seed用于稳定性判断；若要声称跨场景泛化，还需新外生场景。固定42/0曲线仅说明当前场景内学习。
- 从同一actor/critic/state-normalizer分支时，两臂采用相同optimizer/reward-normalizer/LR/RNG初始化协议。现有checkpoint不是完整resume；不能把fresh optimizer的分支叫精确续训。
- 先做单独beam熵对照，再测试在线辅助；求解前筛查单独验证数值等价/性能。必要时才研究同版本累积多episode增大batch；不能直接增加epochs来代替新样本。
- N2每轮80条仍较小，但历史N4吞吐提高未形成稳定同墙钟总体optimal优势，故不能承诺换N4就解决学习。扩大网络、统一提高学习率、全头同时降熵、直接更换PPO比率都缺少当前诊断支持。
- `episodes=10000`是LR horizon；LR约9000次更新后到下限3e-5（step为更新前计数，不能把9000次完成更新当精确边界），10000不会自动停止，当前`max_updates=0`表示持续运行。

## 原始文献

1. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., Klimov, O. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347，Algorithm 1与式(7)。[PDF](https://arxiv.org/pdf/1707.06347)。支持同旧策略NT样本+多epoch与clip定义，不为本项目连续/离散分组surrogate提供效果保证。
2. Vahapoglu, C., O'Shea, T. J., Liu, W., Roy, T., Ulukus, S. (2026). *Unsupervised Neural Beamforming for Uplink MU-SIMO in 3GPP-Compliant Wireless Channels*. Sensors 26(2), 366. DOI:10.3390/s26020366。[作者PDF](https://user.eng.umd.edu/~ulukus/papers/journal/unsup-neural-bf.pdf)。支持复接收权重的物理目标可微训练；非本项目PPO实验，不保证加速。
3. Cobbe, K. W., Hilton, J., Klimov, O., Schulman, J. (2021). *Phasic Policy Gradient*. ICML, PMLR139:2020–2027。[PDF](https://proceedings.mlr.press/v139/cobbe21a/cobbe21a.pdf)。辅助阶段与策略KL保护；原实验辅助任务为价值拟合，并非接收波束。
