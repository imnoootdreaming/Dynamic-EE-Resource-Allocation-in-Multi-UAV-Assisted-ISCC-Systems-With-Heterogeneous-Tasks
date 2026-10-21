# MHBPPO `unknown` 根因与达到 50% 的加速建议

## 结论

1. 截至在线环境日志最后记录的 update 12033（2026-10-21 09:31:18），仍只有 4 个最终状态为 `unknown`，均发生在 update 11638/11683/11698/11807；11807 之后没有新增。
2. 4 个 `unknown` 的直接原因完全一致：MOSEK 返回 `100006`（`trm_stall`），PC3P 在前 3 或 5 次凸化子问题成功后，下一次求解数值停滞。它们不是解析证书已证明的不可行，也没有返回通过原问题验证的解，因此保留为 `unknown` 是正确的。
3. 四条记录的感知必要条件最小裕量分别为 1.78588、4.51943、31.49831、22.44842，均大于 1；当前解析证书只能在裕量小于 `1-1e-6` 时证明不可行，不能由“没有证明不可行”反推出可行。
4. 四次停滞前最后一个已解 PC3P 子问题的秩一 gap 分别为 `3.41e-5`、`1.40e-4`、`1.29e-4`、`3.94e-4`，均超过收敛阈值 `gamma_2=1e-5`。所以程序继续下一次线性化，并在下一次求解停滞；不能直接把上一轮结果记为 `optimal`。
5. 这四个 `unknown` 各自是对应 40 槽 episode 的唯一失败（39 optimal + 1 unknown）。即使事后能把四个全部严格验证为 optimal，最近 1000 个 episode 的全完成率/全 optimal 率最多增加 0.4 个百分点。它是应修的数值鲁棒性问题，但不是训练约 1 万 update 才达到 50% 的主因。
6. 训练窗口首次同时超过 50% 的固定 500-update 区间是 10501–11000：1000 个 episode 中全 completion 51.4%、全 optimal 54.0%。由于 2 env 每 update 产生 80 槽、2 个 episode，这对应约 88 万槽和 2.2 万个 episode；比较训练速度必须同时报告环境槽数和墙钟时间，不能只比较 update 编号。

## `unknown` 的完整因果链

| update/worker/slot | MOSEK code | PC3P 失败轮 / 上一成功轮 | 上一轮 rank gap | 感知最小裕量 | 返回物理解 |
|---|---:|---:|---:|---:|---|
| 11638/1/9 | 100006 | 4 / 3 | 3.4065e-5 | 1.7859 | 否 |
| 11683/0/4 | 100006 | 4 / 3 | 1.3959e-4 | 4.5194 | 否 |
| 11698/0/11 | 100006 | 4 / 3 | 1.2896e-4 | 31.4983 | 否 |
| 11807/1/4 | 100006 | 6 / 5 | 3.9444e-4 | 22.4484 | 否 |

`fusion_pc3p.py` 在每轮 `model.solve()` 后要求 `ProblemStatus.PrimalAndDualFeasible`，否则立即返回；只有通过状态检查后才提取变量、更新 `last_solved_iteration` 和计算 rank gap。`solver_status.py` 对 raw unknown 只使用当前外层动作下的 C03/C05/C06 感知必要条件作单向不可行证书。四条记录的证书均 `certified=false`，协方差、恢复波束和原约束也因没有返回解而未执行验证。

`100006/trm_stall` 精确说明“优化器因进展过慢而终止”；它说明数值算法没有得到可接受的终止状态，不等价于模型不可行。当前数据还不能把深层原因唯一锁定到某一行约束。值得优先验证的原因是 SDP/CCCP 后续线性化后的尺度或病态性：日志中的信道/干扰系数约在 `1e-12–1e-9`，同时模型含功率、频率、时延和罚项；MOSEK 官方也建议对 UNKNOWN/数值不稳定先检查重缩放、近线性相关和病态模型。

## 修复 `unknown` 的顺序

1. **先做可复现诊断，不改标签。** 只在 `100006` 时保存 MOSEK task、solution summary、线性化轮次、系数数量级和 response description；用 MOSEK problem analyzer 检查尺度和近线性相关。
2. **同一线性化点做一次 fresh-model 重试。** 当前 PC3P 在各 CCCP 轮复用 Fusion model 并刷新参数。发生 `trm_stall` 时，用相同系数新建模型重试一次，可以区分模型复用状态与问题本身的数值病态；重试次数必须有上限，避免拖慢训练。
3. **重缩放后离线回放四个案例。** 让主要变量、约束残差和目标项处于可比数量级，再比较默认参数与当前 `pfeas=dfeas=1e-8, relgap=1e-7`。放宽容差只能作为对照组；它不能绕过 `gamma_2=1e-5` 和原约束验证。
4. **允许“上一轮可行解回退”时必须改状态语义。** 保存上一轮变量，重新检查原始约束、恢复波束和 rank 条件。只有全部通过才能作为可用解；未达到 rank/最优标准时应标成 `stalled_feasible/suboptimal`，不能伪装成 `optimal`。当前四条上一轮 rank gap 均超阈值，不能按现有证据直接回退为 optimal。

## 为什么要一万多 update，以及真正有效的加速项

### 墙钟瓶颈

末 500 update 平均每轮 13.926 s，其中采集等待 13.440 s、PPO 0.229 s、日志 0.205 s。采集占 96.5%，说明瓶颈是 80 次环境步内的 PC3P/MOSEK，而不是神经网络优化。

两 env 的当前吞吐约为 `80/13.926=5.74 槽/s`。若改成一 env，要超过这个吞吐，40 槽完整 episode 必须在 6.96 s 内完成；目前一个 worker 完成 40 槽约需 13 s 量级，所以一 env 很可能只让每个编号 update 略短，却让相同样本量所需 update 翻倍。四 env 在 6 个逻辑 CPU 上可能有收益，也可能因四个 MOSEK 进程争用而变慢，必须按槽/s实测。

### 优先级 A：同时提高样本效率和成功率

1. **让 CU matching 条件化于已采样功率，并做一对一约束 mask。** 当前 actor 是 10 维连续功率头加 4 个彼此独立的 10 类 Categorical matching 头；它必须靠负回报学习功率、信道和四个唯一 CU 选择之间的组合约束。最新 40000 槽中 603 个 infeasible 全部存在“在当前采样功率下满足感知必要条件”的一对一候选 matching。这只是必要条件候选，实施前仍需完整求解器反事实验证，但它明确指出 matching 结构是最大的可学习浪费来源。实现时先采样功率，再按每个 UAV-CU 的必要裕量屏蔽边，并在联合分布中保证 CU 唯一；PPO 必须用同一个 masked distribution 计算旧/新 log-prob 和 entropy。
2. **用完整求解通过的动作做 matching/power 预训练，再 PPO 微调。** 现有代码已有 BC 数据和 policy initialization 通道。教师标签应来自完整求解与原约束验证通过的动作，不能把“感知必要条件匹配”当完整 optimal 标签。最省时间的同场景方案是从 iter_12000 的 actor、critic、状态归一化构造兼容 initialization bundle；这属于 warm start，不是从零训练速度证明，且当前初始化逻辑会新建 Adam、reward normalizer 和 RNG 状态。
3. **把“40 槽零失败”写入目标。** 当前 PPO主要优化逐槽累计回报；全成功率取决于失败在 episode 间的分布，末两个 500-update 窗口逐槽 optimal 几乎不变（98.4725%→98.4825%），全 optimal 却从 57.9% 降到 55.3%。建议在状态中加入 `failed_so_far`，并对 episode 零失败加入明确终局奖励，或在第一次失败时施加一次 episode 风险罚项；保留现有逐槽能耗目标，避免第一次失败后后续动作无学习信号。奖励权重需用消融实验选择。

### 优先级 B：减少每槽求解成本

1. **将严格的感知不可行证书移到 MOSEK 前。** 对当前固定外层动作，如果同一个 C03/C05/C06 证书已经严格证明不可行，则跳过 PC3P，并产生与现有后处理完全一致的状态、reward 和诊断。最近窗口仅约 1.5% 槽会因此跳过，近期收益有限；训练早期失败更多，预期更有价值，但必须用历史回放实测，不能先宣称倍数。
2. **保持 `mosek_num_threads=1`，基准 N=1/2/4。** 每个配置从同一固定 checkpoint、相同动作 seed 跑等量槽，报告槽/s、P50/P95 solve time、unknown/infeasible/optimal 和 CPU 利用率。6 逻辑 CPU 上 N=2 是当前合理基线；不要用“秒/update”选配置，因为每 update 的 episode 数不同。
3. **诊断降采样只作为小优化。** 日志项约占 1.5% update 时间；可在根因修完后把候选诊断间隔调大，但关闭诊断不可能解决主要瓶颈。
4. **不要先加 MOSEK 线程或盲目放宽容差。** 小型、频繁 SDP 的线程启动与竞争可能抵消收益；容差变化可能改变可行/optimal 判定。两项都需固定动作离线基准和原约束验证。

## 建议的最小实验矩阵

| 组 | 修改 | 要回答的问题 |
|---|---|---|
| A | 原始 N=2 | 基线槽/s、达到 50% 的槽数/墙钟 |
| B | A + solver 前严格不可行预检 | 只减少墙钟，不应改变动作和标签 |
| C | B + 功率条件化的一对一 matching mask | 是否减少 infeasible、提高样本效率 |
| D | C + 零失败目标项 | 是否让逐槽 98% 转化为更多 40/40 episode |
| E | D + 完整可行教师 BC/warm start | 实际生产训练能否最快达到目标 |

每组至少跨多个训练 seed；每 1000 update 冻结 checkpoint，用独立动作 seed 做固定策略评价。主要终点应为“首次连续三个 500-update 窗口全 completion ≥50%”及其环境槽数和墙钟时间；最终再用未参与选择的至少 1000 个完整 episode 验收。若改变每 update 的 episode 数，禁止只按 update 数宣称加速。

## 证据位置与参考

- 本地冻结报告：`analysis/mhbppo_latest_trend_20261021_0914/report.md`、`episodes.csv`、`frozen/failed_diagnostics.jsonl`。
- 状态解析：`src/outer/reward/solver_status.py:73-89`。
- PC3P 状态返回与收敛判断：`src/inner/fusion_pc3p.py:786-850`。
- 阈值：`src/inner/parameters.py:88-107`。
- MOSEK response code 100006：MOSEK ApS, *Optimizer API response codes*, `rescode.trm_stall`。
- 数值诊断：MOSEK ApS, *MOSEK Modeling Cookbook*, Sec. 7.2–7.4, 2025/2026 online edition。
- PPO：Schulman, Wolski, Dhariwal, Radford, Klimov (2017), “Proximal Policy Optimization Algorithms,” arXiv:1707.06347.
- Action masking：Huang and Ontañón (2022), “A Closer Look at Invalid Action Masking in Policy Gradient Algorithms,” *Proceedings of FLAIRS-35*, Vol. 35, DOI: 10.32473/flairs.v35i.130584.
