# Fig3：MHGPPO 前期奖励上升较快的证据与解释

分析日期：2026-11-10。按用户补充要求，重点解释前期找到较好解的速度，不把后期离散熵退火作为主要解释。只核验文件与策略前向，不修改训练/求解代码、不重训。

## 结论

1. 当前 CSV 实际表现为 **MHGPPO 较早跨过较低奖励门槛，MHBPPO 更早跨过较高门槛且末段奖励更高**。不能笼统写成 MHGPPO 收敛更快。
2. **前期奖励跃升最直接对应的是匹配冲突更早下降。** 第 41–60 轮 MHGPPO 完成率更高，但候选能耗反而更高，因此该窗口的奖励优势不能解释为连续资源优化更有效。
3. Gaussian 连续分布并不直接改变离散 CU 匹配公式。更深一层的候选机制是连续/离散分支共用特征网络、优势估计与状态分布的耦合，以及不同采样消耗造成的随机路径分化。现有单种子汇总证据尚不能判定各机制的贡献。
4. 当前 MHGPPO 是 **Gaussian 潜变量经过 sigmoid 变换**，不是直接硬裁剪无界 Gaussian。已定位到其概率反算中的额外 clamp 不一致；这使本组对照还包含具体实现差异，不能全部解释成分布家族的固有性质。
5. 已核实真实 IEEE TWC 与 TVT 论文，但它们支持的是特定有界动作任务中 Beta 的优势；没有从这些论文中找到“Gaussian 必然更早找到优解”的结论。

## 1. 数据身份与比较口径

- `src/fig/outer/Fig3/MHBPPO.csv`、`MHGPPO.csv` 各 3000 行，episode 为 0–2999。
- 两者 reward、completion_rate 均与对应 `src/outer/ckpt/<算法>/BEST_CKPT/training_updates.csv` 逐行精确相同；后者更新编号连续 1–3000，每轮 320 个时隙，累计各 960000 个时隙。
- 两份记录的完整外生场景签名相同：`988aeeaa8f2ebda7d84c1aa2b64a0db550ec9b77ab86e8e4201a45c3936fac1e`；seed42、8env、40槽、hidden128、epochs5、minibatch32、actor/critic LR 3e-4、22维连续动作、4个10类离散头相同。
- 已保存超参数的唯一差异为 `discrete_entropy_anneal`。遵从用户要求，本报告不以其解释前期差异；当前退火本来也在 2400 轮以后发生。
- 两套当前 Actor 均有 174804 个参数。Gaussian 不是依靠更少的网络参数获得这组前期优势。
- `Fig3` 绘图脚本使用一个种子，横轴实际为外层更新轮数，一轮含8个episode。跨种子标准差、显著性和泛化尚未由这些文件建立。
- MHBPPO 的 Fig3 `obj` 列会含 inf；能耗比较采用原始更新 CSV 的 canonical energy sum/count，按有限候选数加权，不平均 inf，也不直接比较两种不同聚合方式的 obj 列。

完整输入文件 SHA256、配置对照、归一化/权重哈希与统计结果见 [metrics.json](metrics.json)。

## 2. “更快”的适用范围

以截至当轮的100轮 reward 均值首次达到门槛为描述指标；编号从1开始，相邻窗口重叠，不是预先注册的收敛判据。

| reward 门槛 | MHBPPO | MHGPPO |
|---|---:|---:|
| 0.90 | 274轮，87680时隙 | 161轮，51520时隙 |
| 0.95 | 651轮 | 1029轮 |
| 0.98 | 1558轮 | 3000轮内未达到 |
| 0.99 | 1944轮 | 3000轮内未达到 |

以“此后一直不低于门槛”作事后补充，0.95对应MHBPPO第759轮、MHGPPO第1464轮。Gaussian的优势集中在较低门槛的前期，不是各精度水平都更快。

记录的训练阶段耗时累加约为MHBPPO20.95h、MHGPPO23.08h；末500轮约28.94s/轮与28.88s/轮。两种计时位置略有差别，且不完整覆盖所有聚合/存盘/检查点开销，服务器负载也未受控。这些记录不支持“MHGPPO实际总训练耗时更少”。`solver_seconds`是各worker求解时间的合计，不是并行墙钟时间。

## 3. 前期究竟快在哪里

| 窗口 | 指标 | MHBPPO | MHGPPO |
|---|---|---:|---:|
| 41–60 | 原始reward | 0.396712 | 0.600702 |
| 41–60 | 日志定义的完成率 | 54.8594% | 74.8906% |
| 41–60 | 更新后期望重复匹配惩罚 | 0.501755 | 0.253285 |
| 41–60 | 有限候选加权能耗 J/槽 | 103.0912 | 140.7113 |
| 61–80 | 原始reward | 0.476128 | 0.843191 |
| 61–80 | 日志定义的完成率 | 60.7188% | 93.8594% |
| 61–80 | 更新后期望重复匹配惩罚 | 0.408090 | 0.054289 |
| 61–80 | 有限候选加权能耗 J/槽 | 102.3644 | 100.2748 |
| 81–100 | 日志定义的完成率 | 76.0156% | 97.6719% |
| 81–100 | 更新后期望重复匹配惩罚 | 0.243483 | 0.018372 |

`actor_expected_duplicate_matching` 是更新后策略在该轮状态上的期望重复次数，不是实际冲突概率；完成率对应采样策略。两者不能逐行等同，但变化方向共同支持“匹配更早改善”的判断。

奖励在 `src/outer/reward/my_reward.py:125` 定义为

\[
r=\exp(-E_{\mathrm{weighted}}/1000)-P_{\mathrm{duplicate}}-P_{\mathrm{collision}}-P_{\mathrm{no\ solution}}.
\]

重复匹配惩罚以整数次数进入奖励；从约百J能耗中节省少量能量所带来的奖励提升，通常远小于消除一次重复匹配的1分提升。因此前期主要先学会避免罚分，后期才显著体现低能耗精细优化。不能用 `exp(-平均能耗/1000)` 代替逐时隙指数奖励的均值；上述能耗表用于定位改善方向，不声称是精确reward分解。

**第41–60轮的关键反证：Gaussian候选能耗更差，但reward更高。** 这排除了“该窗口Gaussian更快找到低能耗连续动作”作为主要解释。

### 匹配为什么会更早改善：已知结构与待验证机制

两套 Actor 都是共享两层Tanh特征网络，随后输出连续头和Categorical匹配头。连续/离散各自ratio与clip，但使用同一advantage，并共同更新共享特征。另有同公式、同权重0.3的期望重复匹配辅助项。

设共享参数为ψ，则更新含有

\[
\nabla_\psi L=\nabla_\psi L_{\rm cont}+\nabla_\psi L_{\rm disc}
+0.3\nabla_\psi D-0.05\nabla_\psi H_{\rm cont}-c_d\nabla_\psi H_{\rm disc}.
\]

改变连续分布会改变连续概率梯度、连续熵梯度、执行动作、后续状态及优势估计，进而间接改变离散匹配的学习轨迹。即使离散头公式相同，也不能把它们视为独立于连续策略的两个相同训练问题。

诊断脚本在相同seed42重新初始化两套Actor，确认同一输入上的Categorical概率逐元素精确相同。说明“Gaussian分布直接让CU选择更不重复”不成立；差异在更新和采样轨迹中形成。

此外，Beta与Normal采样消耗随机数的方式不同，相同seed不等于相同的离散探索序列。单种子上较早打破四头对称性可能贡献这组结果，不能据此声称Gaussian具有普遍的前期样本效率优势。

初始化时，Beta约为`Beta(1+log(2),1+log(2))`，Gaussian约为`sigmoid(N(0,1))`。后者初始更集中在中间区域，是可检验的探索差异，但前20–40轮Gaussian候选能耗反而较高，不能直接把初始集中性写成此次加速的已证实原因。`rsample()`也不构成这里的直接优势：rollout在no_grad下，PPO通过log-prob计算策略梯度，没有通过环境reward反向传播。

## 4. sigmoid、硬裁剪与已经定位的数值问题

当前代码在`MHGPPO_agent.py:110`附近执行

\[
z\sim\mathcal N(\mu(s),\sigma(s)^2),\quad a=\operatorname{sigmoid}(z),
\quad u=l+(h-l)a.
\]

这是原有的后处理，不是本轮新增。sigmoid平滑地把实数映射到(0,1)：z=-2、0、2分别映射到约0.119、0.5、0.881；硬裁剪则会把越界值直接压到0或1。严格说，执行动作服从logit-normal分布。环境仍有[0,1]防御性clip，但数学上sigmoid输出已经在范围内。

采样时log-prob包含sigmoid的Jacobian修正。然而更新时又把保存动作clamp到[1e-6,1−1e-6]后再logit反算，导致边界附近恢复出的z不等于原采样z。

使用最终权重和各自配套归一化，对一个重建初始观测抽样4096次，固定权重后立即重算连续log-prob：

| 指标 | MHBPPO | MHGPPO |
|---|---:|---:|
| 连续动作元素落在反算clamp范围外 | 0% | 10.8487% |
| 至少一个元素落在该范围外的动作 | 0% | 92.9688% |
| 参数未变但连续ratio已超出[0.8,1.2] | 0% | 91.6016% |

PPO原始定义要求同一策略的ratio=1，见Schulman等（2017）式(6)前的定义。这项检查证实了当前代码和checkpoint组合的概率计算不一致，但91.6%是**单状态前向诊断**，不是原训练状态分布上的发生率，更不能据此判定它对前100轮学习速度的贡献。

Gaussian连续熵使用的是变换前Normal的熵。真实动作熵则满足

\[
H(a)=H(z)+\mathbb E[\log(a(1-a))].
\]

因此同为0.05的连续熵系数在Beta与当前Gaussian中并不等价。在该初始观测的iter1000/2000/3000前向中，Gaussian全部22维σ均达到exp(2)=7.389的上限。该证据与后期边界饱和、高速长尾及概率反算问题一致。修复应优先统一潜变量old/new概率计算，并明确正则化的是潜变量熵还是动作熵；本轮只定位，不实施修复。

本地重建的初始UAV位置与保存轨迹首点最大差5.68e-14m；UAV/target的浮点字节哈希不同，其余场景分量哈希相同。该前向探针使用当前实现、重建状态和真实checkpoint，不冒充原始rollout的逐状态重放。MHBPPO actor/environment/reward源码与其checkpoint所记哈希相同；当前inner/parameters.py及inner/environment.py已有后续变更，但本探针不调用优化器。

## 5. 末段数据仅用于交代完整现象

| 第2501–3000轮 | MHBPPO | MHGPPO |
|---|---:|---:|
| reward均值 | 0.99308130 | 0.96852139 |
| reward总体标准差 | 0.00210808 | 0.00516357 |
| 完成率 | 99.91625% | 99.83250% |
| 有限候选加权能耗 J/槽 | 6.049986 | 30.925475 |
| 有限候选飞行能耗 J/槽 | 14.999210 | 77.190240 |
| 严格可行样本数 | 0 | 0 |

加权能耗相差约5.11倍，差异几乎全部来自飞行能耗。代码飞行功率为0.00614v³+15.976/v，单项最小点约5.427m/s；高速长尾会因三次项显著增加平均能耗。这是当前候选解与原奖励口径的比较，严格可行性能仍须另行核验。

## 6. 真实文献及其可支持范围

检索范围为Gaussian/Beta、有界连续动作、PPO/actor-critic与IEEE Transactions交叉；优先读取作者原文、期刊PDF、作者机构记录。以下是针对本问题的定向检索，不是系统综述或穷尽性声明。

1. **Wenshuai Liu, Bin Li, Wancheng Xie, Yueyue Dai, Zesong Fei（2023）.** *Energy Efficient Computation Offloading in Aerial Edge Networks With Multi-Agent Cooperation.* **IEEE Transactions on Wireless Communications**, 22(9), 5725–5739. DOI: [10.1109/TWC.2023.3235997](https://doi.org/10.1109/TWC.2023.3235997)。[作者公开稿PDF](https://arxiv.org/pdf/2302.09021)；[作者机构书目记录](https://faculty.nuist.edu.cn/libin/en/lwcg/97520/content/127781.htm)。
   - 最贴近本项目的UAV/MEC、能耗与MAPPO比较。作者稿III节Beta policy段及式(48)讨论有限动作区间与探索；IV-B节、Fig.3–4比较Beta/Gaussian方案，报告Beta方案更高奖励。
   - 其Gaussian边界论证针对越界动作硬裁剪，不能直接套到本项目的sigmoid映射。论文也没有证明本项目所见Gaussian前100轮更快；本文中的attention加速结论不能归给Gaussian。

2. **Iran Mesquita Braga Jr., Roberto Pinto Antonioli, Gábor Fodor, Yuri C. B. Silva, Walter C. Freitas Jr.（2023）.** *Decentralized Joint Pilot and Data Power Control Based on Deep Reinforcement Learning for the Uplink of Cell-Free Systems.* **IEEE Transactions on Vehicular Technology**, 72(1), 957–972. DOI: [10.1109/TVT.2022.3211908](https://doi.org/10.1109/TVT.2022.3211908)。[期刊版PDF](https://www.6g-sky.net/assets/research_papers/Decentralized_Joint_Pilot_and_Data_Power_Control_Based_on_Deep_Reinforcement_Learning_for_the_Uplink_of_Cell_Free_Systems.pdf)。
   - 第963页式(28)之前说明采用Beta处理有界能量分配动作，并援引其参考文献[42]关于Beta学习性能的结果。
   - 是资源分配场景中的先例，不是Gaussian前期更快的独立证据；其相关速度结论主要引用Chou等。

3. **Po-Wei Chou, Daniel Maturana, Sebastian Scherer（2017）.** *Improving Stochastic Policy Gradients in Continuous Control with Deep Reinforcement Learning using the Beta Distribution.* **ICML**, PMLR70, 834–843. [官方记录](https://proceedings.mlr.press/v70/chou17a.html)，[原始PDF](https://proceedings.mlr.press/v70/chou17a/chou17a.pdf)。
   - 上述TVT相关论述的原始来源。比较TRPO/ACER中的Beta与Gaussian，是特定设置的结果；不是IEEE Transactions，也不是Gaussian早期优势的一般理论。

4. **John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov（2017）.** *Proximal Policy Optimization Algorithms.* arXiv:1707.06347，[PDF](https://arxiv.org/pdf/1707.06347)。式(6)附近明确同策略ratio=1，式(7)定义clip目标，式(9)包含熵项。

5. **Tuomas Haarnoja等（2018；公开稿v2为2019）.** *Soft Actor-Critic Algorithms and Applications.* arXiv:1812.05905，[PDF](https://arxiv.org/pdf/1812.05905)。附录C式(25)–(26)给出Gaussian经可逆压缩后的换元密度；文中使用tanh，本项目使用sigmoid，Jacobian相应变化。它支持区分压缩与硬裁剪，不保证本项目任何学习速度顺序。

## 7. 下一步最有辨识力的证据

优先记录前100–200轮：同状态下各匹配头概率、实际重复率、离散/连续KL与clip fraction、共享特征上的各损失梯度，以及同策略重算ratio。先排除概率实现不一致，再比较多个独立训练种子、相同奖励门槛的达到时隙数；连续与离散使用分离随机流有助于减少无关采样差异。若要检验共享网络耦合，可另做共享/分离特征的受控消融，不能仅凭一次seed42的两条曲线确认机制。

复现：在仓库根目录执行 `D:/anaconda3/envs/ML2024_Task1/python.exe output/fig3_policy_analysis_20261110/analyze.py`。脚本验证CSV对齐、样本数、权重/归一化配对与同策略概率一致性，并生成metrics.json；未执行新训练或内层求解。
