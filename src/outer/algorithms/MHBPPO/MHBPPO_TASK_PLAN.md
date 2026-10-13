# MHBPPO 任务计划

## U24：感知短缺惩罚归一化到0–1的方案 — 已完成（范围/单调性与离线定标核验）

- 2026-10-13：用户要求分析新增惩罚是否可归一化到0–1并给方案；沿用U22/U23物理解释与U21冻结日志，比较线性短缺、固定尺度对数截断和有界对数变换的饱和/定标/聚合边界。只给方案，不改生产奖励或训练。
- 验收：推荐q=d/(d+ln F)、d=[-lnM]+，F>1固定，F表示P_req达到Pmax的F倍时q=.5；M0可定义1，达标0，平均聚合仍0–1。F10仅为初始实验参照，不称最优参数。保留原base1时仅新增项归一化，总无解罚1+lambda Q不在0–1；若总项也需0–1，则基罚p0+(1-p0)Q方案须明确原失败尺度改变。
- U21离线评分证据：全四架失败的平均线性不足中位.954，平滑F10中位.623；硬截断F100使41.89%单架失败同为1。范围/单调例值核对通过，仅验证评分分布，不声称已验证加速。公式经独立数学审查。
- 证据：analysis/mhbppo_slow_beam_20261013/normalized_penalty_proposal.md、normalized_penalty_analysis.py、normalized_penalty_summary.json。未修改源码/物理约束/动作/训练。下一步若授权实施，保留base1并只增加有界感知惩罚做单项消融。

## U23：M与最低感知发射功率的原约束推导 — 已完成（原约束与单位功率增益核对）

- 2026-10-13：用户追问为何可直接用M、P_req如何计算；核对c05/c06与诊断代码，从接收信号/干扰门限到最有利发射方向单位功率增益推导P_req，解释必要条件和动作相关反馈，非完整系统功率解。仅解释，不改奖励或训练。
- 验收：v=A^H g、G=vv^H，lambda_max(G)=||A^H g||²（这个恒等式不要求A本身rank1）；信号|v^H w|²<=||v||²||w||²，可在方向w∝v取等号。因此固定g/匹配/CU功率/几何时，单就c05/c06最低感知功率P_req=eps Gamma/||A^H g||²，M=Pmax/P_req=max可达感知SINR/门限。
- 解释边界：M是最有利发射方向下的预算充足程度，不是求解后实际SINR/完成概率；M<1证实当前动作感知不可行，M>=1仅过必要条件。不改变动作/约束，作为分级奖励只是提供可核查的失败程度，不能声称奖励保证加速。无生产修改/训练；下一步无新增实现要求。

## U22：最小分级无解惩罚与通俗物理解释 — 已完成（代数与现有日志定标依据）

- 2026-10-13：用户要求说明能否最简单修改现有惩罚项，以及M和负对数短缺的含义，提供导师可理解的原因。仅分析方案，不修改奖励代码或训练。
- 依赖：U21的失败反馈与感知功率边界证据，重新核对my_reward与required_sensing_power_w；区分感知必要条件与全部内层可行性。
- 验收结论：M=Pmax/P_req，其中P_req=eps Gamma/lambda_max(G)是仅就感知子系统的最低功率；负对数短缺是log(P_req/Pmax)的正部，解释为压缩的功率超限倍数。可仅改现有no_solution_penalty：无解时1+lambda*平均短缺，有解时仍0。
- 最简单替代短缺max(0,1-M)易解释且有界，但在最近窗口全四架失败样本中平均短缺中位0.954、90分位0.993，深失败接近饱和；对数不是必需，适合保留超限倍数差。未选未经验证的最优lambda，不声称确定加速/原最优策略不变。
- 证据：analysis/mhbppo_slow_beam_20261013/simple_penalty_explanation.md及U21 tail_metrics.npz；未修改生产奖励/动作/约束或启动训练。下一步若授权实施，先以一项分级惩罚做同预算消融。

## U21：16-00-10 长训练、同动作空间加速与接收波束奖励文献 — 已完成（长窗口/反事实/原文核验）

- 2026-10-13：用户要求分析2026-10-12_16-00-10运行约8600轮后optimal缓慢上升，判断是否必须匹配波束、能否保留动作空间加速，并查真实接收波束学习与奖励设计论文。此次明确扩展至奖励设计；接受optimal状态，不重新检查解正确性，不改训练或重启进程。
- 依赖：重读实际run metadata与源码，以完整轮次冻结统计；U16短训练结论不能替代本次长训练证据。并行核对原始接收波束文献与同动作空间实现选择。
- 验收：量化趋势/波束增益/感知可行/各头探索，给出证据分级的建议与可核查原论文PDF，区分接收/发射波束、连续向量/码本动作、稠密奖励/收敛加速证据。

### U21 验收证据与下一步

- 冻结8700：前500optimal0.655%，最近500=8.39%，最近20000步1678optimal/18322infeasible，所有非optimal有感知功率必要条件失败；72.44%无最低功率感知可行匹配。源hash全部与run一致，无解正确性复核。
- 增益比均值0.10043→0.10158；后期波束std0.28439/0.28588、浓度2.09/2.058接近Uniform，匹配entropy降至0.475。成功1678步中1677发生slot0–7。同状态固定时隙2000样本原19.35%优于随机15.46%，matched32.75%，maxSINR100%（仅感知条件，不代表P5），确认有少量波束学习；不以gain均值断言完全未学。
- 同动作空间方案：稠密log感知短缺反馈、分头熵退火、波束监督/物理辅助loss、同版本大rollout；当前晚期lr约初始13%。均为待消融建议，不声称已实现加速或最优策略不变。即时状态-动作罚项不是PBRS。
- 原文核验：Kim/Jung/Song2023 IEEE IoTJ连续复数Rx；Zhang/Osman/Alkhateeb2024 TWC量化Rx相位+SINR改善reward；Dao/Miao2026 BeamCraft未同行评议预印本连续Rx+log感知reward；Jia/Zhou2021最终采用解析Rx，避免误引为直接学习成功；另有物理可微接收loss。未找到隔离奖励单项且可保证当前系统加速的证据。
- 证据：analysis/mhbppo_slow_beam_20261013/report.md、metadata/source_check/text_summary/early_summary/tail_summary/tail_slots/counterfactual.json及脚本、literature_rx.md、literature_shaping.md、code_options.md。并行原文与源码核对完成；未改生产代码、奖励参数、日志或启动/重启训练。
- 下一步：若用户授权实施，保留alpha1与动作映射先单独增加感知短缺奖励并消融，再加入波束专用熵控制；以交互步数、多seed、后续目标窗口及原能效验收。当前分析请求已完成。

## U20：波束残差幅度的自适应约束分析 — 已完成（解析关系与可行性边界核对）

- 2026-10-12：用户追问delta幅度如何确定、是否可灵活变化；沿用U16/U19模型，推导正交残差与方向夹角/增益关系，区分状态安全上界与策略选择幅度，说明参考不可行及过度限制的边界。仅分析，不改生产代码。
- 验收结论：正交单位方向u下delta=r u、g=(gref+r u)/sqrt(1+r²)，复向量夹角theta=atan(r)；匹配参考的目标增益比例cos²(theta)。幅度可由策略随状态选择，上限须结合本次matching/power与感知条件，不凭margin直接推定SINR单调。
- 可行性边界：M>=1只验感知/功率子系统；参考不可行时缩小残差无保证，固定小邻域可能排除可行干扰抑制方向；方向扫描的可行幅度集合未必是连续区间，不能无条件用二分。未运行新实验或修改生产逻辑，数值夹角示例仅为解析对应，非已验收超参数。

## U19：匹配接收与最优方向的物理/数学解释 — 已完成（源码与原论文核对）

- 2026-10-12：用户追问接收波束是否必须匹配、为何存在最优方向；核对rank-one A与G/Gamma，解释复数权向量、白噪声匹配、干扰下广义Rayleigh商、相位非唯一与整系统能效边界。只分析，不改训练。
- 验收：用Cauchy–Schwarz推导白噪声g∝s，白化后推导g∝R^-1s；明确最优是固定状态/信号/干扰下的SINR最优，不要求波束几何主瓣唯一或永远指向目标。项目A=beta aa^H使接收信号方向为a，但干扰会改变最优权向量，c11与整系统能效不能忽略。
- 原文核对：Capon (1969), High-Resolution Frequency-Wavenumber Spectrum Analysis, Proceedings of the IEEE 57(8):1408–1418，DOI 10.1109/PROC.1969.7278；原论文扫描PDF已打开核实。无生产修改/训练；本次解释完成，无新增实现待办。

## U18：接收波束先验与预训练方案分析 — 已完成（方案与现有BC源码核对）

- 2026-10-12：用户追问是否应先给波束先验并预训练网络；沿用U16日志与同状态感知反事实证据，核对现有BC与动作条件化。仅分析，不实施或启动训练。
- 验收：区分结构先验、观测先验与监督预训练，指出相位等价、匹配条件化、探索分布及覆盖范围的前提，给出分阶段方案与验收指标。

### U18 验收依据与下一步

- 结论：结构先验优先，预训练可选；仅把参考g放入状态或只拟合均值不能保证实际采样可行。可用normalize(g_ref+有界残差)保留参考方向，在感知可行验证后再联合PPO。
- 当前actor同状态独立产生匹配/功率/波束，波束未条件于本次已采样匹配；干扰相关教师标签必须输入对应matching/power或固定它们。监督须处理全局相位等价、状态覆盖、随机采样方差与normalizer。
- 核对behavior_clone使用所有生效连续头与离散头联合NLL，非波束专用；alpha=0波束头被mask，BC也不会训练这些头。现有generate_bc_dataset的全动作可行演示不等于解析波束监督。
- 文献核验：Ross/Gordon/Bagnell (2011), A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning, AISTATS/PMLR15:627–635，官方PDF https://proceedings.mlr.press/v15/ross11a/ross11a.pdf；仅支撑闭环状态分布偏移讨论。本次无生产修改、无训练。下一步需用户选择是否保持纯学习波束及80维动作约束后再实施。

## U17：核对 checkpoint 周期、退出保存与奖励曲线开销 — 已完成（源码及指定 run 证据核对）

- 2026-10-12：用户要求回答当前保存频率、强制退出保存及奖励曲线绘制频率与时间影响；仅核对源码、指定 run 元数据与现有产物，不改变训练行为。
- 已读取计划；U06/U09 为相关实现背景，本次重新核对实际保存调用与信号处理，不重复既有测试。验收依据为源码行号、run 配置及退出日志，不推测未经测量的耗时比例。
- 验收：main 第858行周期保存、第669–675行注册停止信号、第884–902行 finally 保存与唯一奖励曲线调用；指定14-28-06 run 元数据 checkpoint_interval=1000，main SHA256 与当前源码一致。该run日志15:45:14确认最终iter=480保存，磁盘iter_480含actor/critic/normalizer/CSV/轨迹，根目录含奖励曲线。
- 结论：周期1000轮；可处理的停止信号在轮次边界收尾，硬终止不能保证保存；奖励PNG只在退出且iteration>0时绘制，周期另有轨迹PNG。异常发生于update中途时不保证模型与最后完整iter精确一致。未测耗时，不给耗时比例；未启动/停止训练或修改生产源码。下一步：本次无待办。

## U16：接收波束学习导致低可行率的机制分析 — 已完成（日志与同状态解析反事实验收）

- 2026-10-12：对比 2026-10-12_14-28-06（直接启动）与 2026-10-11_12-28-50（不学习接收波束），分析波束动作维度、感知与通信耦合、可行域及探索；接受 optimal 状态，不分析求解误判或奖励设计，不改生产逻辑或重启训练。
- 依赖：重新核对实际 run metadata、日志统计及当前源码；U04/U13 的历史结果只在配置一致时沿用，U15 固定场景实现需由新 run 元数据确认。
- 验收：冻结日志统计边界，提供可核查机制推导、逐步日志证据、局限与对照实验建议。

### U16 验收证据与下一步

- 两run前480轮各19200步：新alpha1 optimal=126（0.65625%），19074非optimal全部有感知功率必要条件失败；旧alpha0 optimal=5257（27.38021%）。初始场景全部组件hash一致，madrl_args仅alpha不同；存在记录/保存等源码版本差异，未冒称严格单因素训练。
- 新波束感知增益比均值0.10063，中位数0.07378；77.28125%步骤在固定波束/位置下连最低CU功率也没有感知可行匹配。后100轮波束条件std约0.262，尚未收缩。
- 同状态重建19200步（最大相对误差3.10e-7）：仅替换matched g四UAV感知通过率15.5625%，替换最大SINR g为100%；该结果只验c05/c06，不声称整体P5求解或闭环成功率100%。未检查optimal正确性、未分析奖励设计、未调用solver/训练或修改生产逻辑。
- 证据：analysis/mhbppo_beam_learning_20261012/report.md、initial.json、statistics.json、counterfactual.json及重建脚本；原始PPO论文PDF已核对。当前rank-one单目标/unit-norm感知SINR子问题允许g=normalize(R^-1 a)解析最大化；c11数据量与计算耦合仍需保留，不声称完整能效最优或任意P5可行。报告给出推导及适用条件。
- 下一步：若进入实施/实验，优先同状态仅替换g的真实P5对照，或解析参考/低维残差动作；逐头KL与clip统计为尚未量化的学习放大机制。已有U04/U15等验收保持原范围，不重复或重启用户运行。

> 2026-10-11：AGENTS.md 指定的本文件缺失。已检查指定路径、MHBPPO 目录及全项目文件列表，未找到既有任务计划。此文件只登记本次已授权任务，不重建或推定历史任务的完成状态。

## 当前任务

- **U01：诊断 2026-10-11 11:58:53 训练日志的 unknown，并提供消除方案 — 已完成（方案与证据验收；未实施生产修改）。**
- 用户范围：接受求解器输出的 optimal；不复核 optimal 解正确性；本次交付方案，不修改求解逻辑、物理参数、训练日志或重新训练。
- 依赖：三个现有日志、Fusion 状态处理源码、MOSEK 官方文档；历史分析仅在条件和源码有效性核对后作为参考。
- 验收：核对第 21 轮的 11 个 unknown 原始状态与终止码；说明可以确认的原因及证据不足之处；给出有优先级的实现方案与重放验收方法；引用可核查官方资料。

## 执行记录

- 2026-10-11：已读取 AGENTS.md；指定任务计划缺失，因此登记本次任务。训练日志第 21 轮共 40 个样本，optimal=11、infeasible=18、unknown=11。正在核对逐样本诊断和状态映射。

## 下一步

- U02 已实施并验收同源证书；新启动训练使用新分类契约。现有运行进程需要重新启动才能加载修改；本次未停止或重启它。未来若出现无证书 unknown，另行保存实例并诊断数值原因。
- U01 为历史方案验收；U02 为实施验收，证据分别记录。限定样本 unknown 已消除，不据此保证所有未来场景 unknown 为零。

## 验收证据

- 报告：analysis/mhbppo_unknown_20261011/report.md；详细状态、证书与源码一致性：analysis/mhbppo_unknown_20261011/evidence.json。
- 日志在持续追加，冻结前 52 轮 2080 个逐步记录；479 个 unknown 全为 code=100006、第一次 P5 失败，无先前可用解；479/479 对应动作日志；479/479 满足感知解析不可行判据。
- 第 21 轮的 11 个 unknown 证书成立；按建议最终分类为 11 optimal / 29 infeasible / 0 unknown。该统计为规则预测，不是代码实施结果。
- optimal 未作正确性检查；求解器/桥接/约束诊断关键源码与日志元数据 SHA256 一致。学习器、agent、madrl_args 已变化，只采用日志配置和同源求解证据，不作当前学习配置推断。
- 官方文档核实 100006=trm_stall、Unknown 并非不可行；两名只读审查代理独立核对了 C03/C05/C06 的证书推导与范围。
- 已完成用户要求的方案交付准备；仅新增计划与分析材料，未修改生产源码、参数、训练日志或执行重新训练。

## U02：实施 unknown 解析不可行兜底 — 已完成（代码及限定范围验收通过）

- 2026-10-11：用户明确授权实施 U01 方案；只处理 unknown，optimal 直接接受。
- 验收：同源证书、原始状态追溯、关闭诊断仍有效、边界与非法输入保护；第 21 轮及已冻结 479 个 unknown 重放，接入桥接/日志链路验证。
- 旧证据：U01 对原日志的判断仍有效；实现改动后的源码与旧元数据不再同 SHA256，需新增实施证据，不能伪称旧日志为改后训练。

### U02 验收证据与执行记录

- 2026-10-11：新增 solver_status.py，并接入桥接、reward、sampler 和日志元数据；保留 raw status/termination code/证书，optimal 不作正确性复核。
- 报告：analysis/mhbppo_unknown_20261011/implementation/report.md；validation.json、reconstructed_certificates.json、source_manifest.json 同目录。
- 9 项新测试、4 项既有日志契约测试、语法与 diff 格式检查全部通过。
- 479 个原 unknown 实际信道与动作重建，全部获证；第 21 轮 11 个原 unknown 经真实 MOSEK 重跑均由新证书最终归为 infeasible；一例关闭诊断真实调用验证通过。
- 40 步新 sampler/log smoke（无 PPO 更新）：raw=5 optimal/24 infeasible/11 unknown，final=5 optimal/35 infeasible/0 unknown；来源、证书与汇总计数逐步一致。
- 旧日志的源码元数据描述改前实现；新增 source_manifest 描述改后实现。不覆盖原训练日志、物理参数或其它既有验收结果。

## U03：每次训练独立日志目录 — 已完成（目录隔离与日志契约验收通过）

- 2026-10-11：用户要求每次生成日志在 logs/MHBPPO 下建立时间目录，通过目录区分运行，目录内使用统一文件名。
- 采用 YYYY-MM-DD_HH-mm-ss 秒级时间目录，同秒启动通过目录后缀防止覆盖；run_id 使用目录名。
- 验收：路径生成、同秒目录隔离、训练/动作/环境日志共目录、元数据引用和既有日志契约。历史日志留作原始证据，不迁移正在写入的文件。
- U01/U02 的 solver 判据与验收保持有效；日志布局和 action_logging/main 源码哈希将变化，需要新的目录布局证据。

### U03 验收与下一步

- 2026-10-11：实现秒级独立目录、固定文件名、同秒/并发目录冲突防护；run_id 和日志路径元数据同步。
- 验收：setup_logger + ActionLogWriter + EnvDiagnosticLogWriter 实际写入、同秒隔离、4 个并发目录预留、自定义绝对路径与扩展名回退、4 项既有日志契约测试、语法/diff 检查均通过。
- 证据：analysis/mhbppo_log_directories_20261011/report.md、validation.json、smoke/；改前备份 before/。
- 新启动训练使用新布局；现有进程和历史日志保留当前路径。本次没有重启训练或迁移历史数据。

## U04：最新训练有效性与 optimal 95%–100% 稳定性诊断 — 已完成（分析及限定反事实验收）

- 2026-10-12：用户请求分析 12-28-50_2026-10-11_336687 的训练效果、波动来源及 95%–100% 区间优化可行性。本次分析，不改变生产参数或继续训练；沿用 optimal 直接接受口径。
- 冻结训练日志到 iteration 8066（2026-10-12 09:27:40）；JSONL 按此截止点分析，避免把后续追加记录混入。
- 依赖：训练趋势、近阶段逐步 solver/sensing/CU 匹配诊断、运行元数据和超参数、可用 checkpoint、算法源码。U02 unknown 处理与 U03 布局工作不重复。
- 验收：量化训练改善与尾部波动，区分随机采样和策略失效，检查可行性边界，给有依据的目标判断及按优先级排序的方案；原论文/官方资料引用可核查。

### U04 验收证据与下一步

- 2026-10-12：冻结到8066；趋势last1000optimal90.27%，last50091.45%，last10092.475%，last500每轮>=95仅41.6%。训练有效但未达逐轮稳定目标。
- last500的20000步中1709/1710非optimal有感知不可行证书；1694有同功率替代可行感知匹配，1710都有最低功率可行匹配。后20slots占失败77.25%。
- seed20261012随机48输入：40仅改matching和8改matching+功率均实际MOSEK输出optimal；不检查其正确性，非闭环成功率保证。
- 已核验运行metadata/源码hash、TensorBoard（严格截止8066）和iter8000 checkpoint；checkpoint缺normalizer故不作错误归一化的固定policy评估，此限制明确保留。
- 报告与图：analysis/mhbppo_training_20261012/report.md、summary.json、training_analysis.png/svg；tail诊断、TB、evaluation脚本与逐样本结果同目录。
- 生产源码/参数和运行训练未修改。下一步若用户授权实施：完整checkpoint+评估，感知可行匹配/功率概率契约，再后期熵退火及同版本大rollout，逐项消融验收95%目标。
- U02限定479样本验收仍有效；本次发现7941/slot1的无感知证书unknown（P5第9轮stall）按既定规则保留，不强制改状态。

## U05：日志、分迭代 checkpoint 与 UAV 轨迹保存方案 — 已完成（分析与代码方案；未实施生产修改）

- 2026-10-12：用户要求核对当前训练入口的三类保存行为，输出最小化修改代码方案；本次仅分析，未授权实施生产修改或重启训练。
- 执行前核对：U03 已验收日志目录隔离，本次核实当前源码是否仍符合；U04 的归一化 checkpoint 限制仅作为相关背景，不自动扩展本次范围。
- 验收：提供当前源码行号与实际目录证据，明确 reward 索引、异步轨迹来源及 final 保存行为，给出按迭代归档与二维轨迹的最小改动方案。

### U05 验收证据与下一步

- 2026-10-12：核对当前main、sampler、action_logging、my_env、madrl_args与实际目录；U03当前源码仍有效，默认新运行日志已分时间目录，现有六个旧格式日志不迁移。
- 当前ckpt模型平铺，根reward周期覆盖；读取时根CSV8000行、episode0…7999。已有每步UAV位置快照和初始环境绘图，无周期轨迹PNG。
- 方案与证据：analysis/mhbppo_artifact_layout_20261012/proposal.md、source_evidence.json。只需main+sampler：iter_N归档完整CSV/模型/二维轨迹，根CSV每轮追加，独立回传episode场景以兼容诊断关闭，标记真实policy版本。
- 已明确索引、异步rollout/更新后模型区别、final归档及原有异常退出一致性限制；未运行训练/绘图测试，方案尚未实施。核对期间五个生产源码SHA256均未变化。
- 下一步：如进入实施，应用方案并用短周期烟测验收CSV快照、关闭诊断绘图和退出保存；若要求任意中断严格一致性，需同时处理现有提前退出/信号提交边界。U01–U04原证据不因本次只读分析失效。

## U06：实施分迭代 checkpoint、实时 reward CSV 与 UAV 轨迹保存 — 已完成（代码与限定范围验收通过）

- 2026-10-12：用户授权实施U05方案；日志沿用U03已验收实现，不迁移历史产物，不重启现有训练。
- 执行前核对：当前main/sampler仍为U05分析实现；工作区含此前用户/任务改动，先备份当前两文件，仅局部修改。
- 范围：main+sampler，iter_N完整CSV/模型/二维轨迹，根CSV每轮追加，episode场景不依赖环境诊断开关；局部修正同步队列超时导致本轮统计缺失的保存边界，不扩展求解器或完整续训checkpoint。
- 验收：短历史快照冻结、根CSV逐轮追加、周期/非周期退出、实际PNG目视检查、关闭诊断的sampler场景回传、语法及既有日志契约检查；证据保存analysis/mhbppo_artifact_layout_20261012/implementation/。

## U07：核对状态、动作、信道与奖励的时隙语义 — 执行中

- 2026-10-12：用户要求核对当前 MHBPPO_main.py 调用链是否用 t+1 信道处理 t 任务；仅分析源码，不改变训练行为。
- 已读取计划；U06 保存实现不属于本次验收范围，保留其现有状态。依赖当前 main、sampler 与 MyEnv.step 的实际执行顺序。

### U07 验收证据与结论 — 已完成

- main 导入并启动 sampler_worker；sampler 第288–290行依次选动作、调用step，第327–341行同次存入状态/动作/next_state/bs奖励。
- my_env.py 第570–586行先用当前 self.*_channels 与任务尺寸计算奖励；第625–635行之后才推进t、更新位置、刷新信道并构造下一状态。奖励调用同时传入动作前后UAV位置，但信道未提前刷新。
- 结论：经验字段对应 (S_t,A_t,S_{t+1},R_t)，但不是用 t+1 信道处理 t 任务；实际先用 t 信道计算本步奖励，再形成 t+1 状态。源码静态顺序足以验收，未启动训练或修改生产源码。下一步：本次无待办，U06 状态保持原样。

### U06 验收证据与下一步

- 2026-10-12：完成main+sampler改造；iter_N内完整CSV/模型/XY轨迹，根CSV逐轮追加，退出按最终轮次归档，场景回传独立于诊断开关。将正常停止信号改为轮次边界收尾，修复同步许可Full的本轮历史缺失路径；PPO/求解器/物理参数不改。
- 9项保存/轨迹/边界验证、4项既有日志契约、5项既有训练配置回归全部通过；AST与diff格式检查通过，实际PNG已目视验收。新增验证使用真实sampler/MyEnv/绘图、mock reward与learner/worker，不代表完整训练或求解器验收。
- 证据：analysis/mhbppo_artifact_layout_20261012/implementation/report.md、validation.json、validate.py、源码备份与patch、source_manifest.json、uav_trajectory_preview.png。首次测试错误假设默认sector_ring布局必变化，已修正并补验random_sector；保留首次失败记录。
- U05为改前分析，关于“缺少iter目录/周期PNG”的结论已被本次实施更新，旧源码哈希不再描述当前main/sampler；U03日志目录实现不改、回归通过。U01/U02限定证书与U04历史冻结分析仍为原版本证据，不冒充新保存布局训练。
- 未停止/重启现有进程，未迁移旧日志或checkpoint；新启动训练生效。完整续训normalizer等限制仍保留；任意update中途不可恢复异常的精确模型一致性未宣称解决。
- 下一步：用户下次新启动训练时使用新布局；如需完整续训/固定策略评估或迁移历史模型，另行明确范围。本次授权的代码保存改造与验收已完成。

## U08：MDP 奖励结算时序解释 — 执行中

- 2026-10-12：沿用U07源码顺序证据，补核reward中原/新位置用途及MDP原始教材定义；仅分析，不改变训练时序。验收区分奖励接收时刻、奖励物理依据和观测Markov充分性。

### U08 验收证据与结论 — 已完成

- Sutton与Barto（2018）教材第3章定义下一奖励与下一状态联合转移；奖励索引R_t或R_{t+1}是约定，奖励在step返回并不要求使用下一时隙信道。核查教材PDF（Berkeley高校托管版本）https://msdnaa.eecs.berkeley.edu/~cs188/fa19/assets/files/SuttonBartoIPRLBook2ndEd.pdf 。
- 当前reward使用t信道；my_reward.py第25–31行使用移动后位置判碰撞，第37–54行将当前信道与动作前后位置传内层求解，第74–78行使用位移计算飞行能耗。因而本步奖励包含动作后果，并非动作执行前的奖励。
- 当前step顺序符合MDP交互时序，但严格Markov充分性未验收，不将经验格式当作证明。物理合理性取决于通信/感知是在移动前进行还是移动后进行；后者需要对应新位置的信道，独立于奖励编号。仅完成解释，无生产修改。下一步如要求严格审计，需核对时隙阶段定义及观测是否包含影响未来的移动状态、剩余时域等。

## U09：checkpoint 保存对应状态归一化统计 — 已完成（代码与测试验收）

- 2026-10-12：用户明确授权保存状态归一化统计，以支持后续测试；以U06当前iter_N保存边界为基础，不覆盖其轨迹/CSV实现。
- 范围：周期与退出checkpoint增加同版本mean/std/count/m2及epsilon；提供冻结加载/归一化方法；BC初始actor若保存，也同时保存对应统计。保持原actor/critic权重格式，不扩展为完整训练恢复。
- 验收：磁盘往返后与sampler归一化一致，零方差/初始统计/维度检查，周期与最终保存同版本，原权重可直接load_state_dict；旧checkpoint缺统计明确报错，不能用新统计补旧权重。

## U10：移动后信道与奖励物理一致性分析 — 执行中

- 2026-10-12：用户要求仔细分析动作导致的最新信道是否应进入本步收益；沿用U07/U08已核对调用顺序，检查桥接是否重建信道，区分UAV动作与CU运动/随机衰落。仅分析，不修改生产逻辑。

### U10 验收证据与结论 — 已完成

- 当前桥接cccp_bridge.py第231–278行直接传入外层旧信道；inner/environment.py第288–361行只装配这些信道与派生常数，未按新位置重算。U07/U08顺序证据仍有效。
- 结论：用户提出的移动后信道进入本步收益是合理的post-move任务定义，但非MDP唯一要求；当前移动信道收益通过next_state进入未来奖励，飞行能耗/终点碰撞进入当前奖励。若意图先移动后服务，需要采用新几何信道，且显式区分CU移动、NLoS演化及目标窗口切换。
- 建议用H_exec明确执行信道，而非无条件将t+=1与目标切换搬到reward前；冻结本步任务/目标、保留原位置用于飞行能耗、必要时让结构化波束参考执行信道。飞行与服务时间预算须一致；当前inner/environment.py第335–338行按整slot计算飞行能耗，不能不加说明地假设整slot飞行后还有整slot服务。
- 原始文献：Zeng与Zhang，Energy-Efficient UAV Communication With Trajectory Optimization，IEEE TWC，2017；作者稿https://arxiv.org/pdf/1608.01828 第II节式(1)–(3)描述同时刻位置、信道与速率耦合，不能据此声称必须统一用终点信道。
- 本次仅源码分析与文献核对，未改生产源码或重启训练。下一步取决于用户指定的执行时序与时长假设；不承诺仅改变结算信道即可提高完成率。

### U09 验收证据与下一步

- 2026-10-12：U06保存点增加同版本state_normalization.npz，含mean/std/count/m2/epsilon/iteration/dim/actor SHA256；新增冻结加载方法与CHECKPOINT_TESTING.md。周期、退出和BC初始化均覆盖，旧权重格式保留。
- 7项新增单测、2项复用真实入口有限回归通过；iter2/4/5与停止iter3统计和权重匹配。AST/diff检查通过。报告、测试输出、源码hash：analysis/mhbppo_state_normalization_20261012/。
- 保存更新后模型及归一化，与同版broadcast一致；加载失败不使用猜测值。sampler、PPO、物理环境和求解器不变；U06历史checkpoint API因新增必传统计参数不再适用于无统计调用，新真实入口回归证明当前周期/最终CSV、权重及轨迹路径正常。
- U04关于原iter8000缺统计的历史事实不变；新checkpoint解决之后的测试数据缺失，未给旧权重补假统计。现有进程未重启，新启动训练生效。
- 下一步：测试用checkpoint.py加载对应iter_N统计并冻结；完整优化器/reward/RNG续训不属于本次授权范围。

## U11：论文娱乐任务参数与实际观测核对 — 已完成（只读分析与观测实查）

- 2026-10-12：用户提供LaTeX论文，要求核对Outer-Level Optimization / State space中parameters of entertainment task是否缺失，若缺失说明应该添加哪些信息。
- 范围：读取用户论文、observation.py与环境context/内层映射；仅分析，未经要求不改观测或训练。
- 初查observation.py已有task_sizes/task_cycles/task_delays，继续核查实际来源及与求解器一致性；U09 checkpoint和此前训练分析不重复。
- 验收：准确对应论文L_j/C_j/D_j^max，给源码行号、实际观测维度和数据来源；区分字段存在与任务变化/异质性，以及配置不一致风险。

### U11 验收与下一步

- 2026-10-12：论文State space原文1408–1409行三组参数与observation.py task_sizes/task_cycles/task_delays一一对应；实际默认1017维观测末30维含170000 bits/1000 cycles-per-bit/0.6s。
- MyEnv与内层默认参数一致；进程内修改CU任务量后观测与桥接同步，未运行求解器。默认固定任务值符合论文仿真设置；异构/时变扩展需要同源三向量打通观测、reward和内层，并处理数据量硬编码和归一化分布。
- 证据：analysis/mhbppo_task_observation_20261012/report.md、verification.json、source_evidence.json、verify.py。生产源码/参数未修改。
- 下一步：当前无缺失任务字段，不重复添加维度；如用户要求异构/时变任务，再实现三向量来源与时隙更新的一致性。U09 checkpoint功能不改。

## U12：各动作头策略均值/标准差诊断日志 — 已完成（诊断接入与行为不变验收）

- 2026-10-12：用户授权仅增加诊断，重新训练后再决定是否修改连续熵系数；continuous_entropy_coef保持0.05，discrete系数及训练行为不变。
- 范围：采样同一次前向的Beta均值/标准差及浓度，标明未生效接收波束头；离散头用概率/熵表达探索；逐步动作JSONL与按轮文本/TensorBoard汇总，区分条件策略std与实际动作跨状态波动。
- 验收：与分布解析矩一致，不引入RNG抽样或梯度，固定seed动作/logprob不变；真实sampler日志与learner汇总可追溯至实际weights_version，诊断关闭仍可记录。
- U04原训练总熵诊断仍为历史事实，本次仅增加下一次训练的可观测性；不改PPO、熵系数或物理环境。

### U12 验收证据与下一步

- 2026-10-12：采样同一前向的Beta条件mean/std及浓度、离散概率/熵；动作JSONL逐时隙记录，文本[POLICY]与TensorBoard逐分量按实际wv汇总，区分探索std、状态均值变化与采样std。masked头标记不生效。
- 7项新单测、7项既有概率/梯度回归通过；5个40步真实sampler有限集成通过（mock reward/update），200条动作记录与5个policy版本匹配，关闭环境诊断仍记录；AST/diff通过。
- 证据：analysis/mhbppo_policy_distribution_20261012/report.md、unit_tests.json、entry_validation.json、source_manifest.json及smoke/；使用说明POLICY_DIAGNOSTICS.md。
- 未修改entropy系数、优化器/PPO更新、采样分布、环境或求解器；固定seed验证动作/logprob/RNG/梯度相同，actor checkpoint字段不变。U09保存回归通过，历史证据不冒称新诊断日志。
- 首次集成硬编码4生效头但当前默认alpha=1有6头，已改为检查实际配置，未改生产参数；原12-28-50 run alpha=0，重训对比需沿用原配置。
- 下一步：用户自行重新训练后，结合各头条件std、浓度、失败时隙再决定是否降低连续熵系数；本次不启动或重启用户训练。

## U13：直接启动的接收波束与 episode 随机性核对 — 已完成（源码与有限环境重置验收）

- 2026-10-12：用户要求分析直接 python MHBPPO_main.py 的默认接收波束配置，以及不同 episode 是否使用相同随机种子。只读核对配置、环境重置、采样 RNG 与策略更新，不修改训练参数或启动训练。
- 依赖：U12 已确认当前 alpha=1、历史 run alpha=0；重新核实当前实际代码，区分固定场景、episode 随机场景和动作采样随机流。验收用有限环境重置与源码证据。

### U13 验收证据与下一步

- 2026-10-12：当前默认alpha=1，三个场景randomize=True；主seed1208，默认worker种子1209；场景通过spawn_key=(episode_idx,)派生不同流，动作RNG仅worker启动初始化，跨episode继续推进。
- 实测episode0/1的CU布点、轨迹、NLoS不同；sector_ring确定性UAV/目标/调度相同；同episode重复重置重现场景，reset不重置torch RNG；全关开关后场景固定，单开NLoS仍重生成CU布点。
- 证据：analysis/mhbppo_episode_seeds_20261012/report.md、verification.json、verify.py、source_manifest.json；未调用solver、未训练或改生产源码。
- 范围更新：U04历史固定场景与alpha0证据仍只描述旧run；当前随机化默认已变化，不能拿其成功率直接对比新默认run。U12诊断功能验收仍保留原范围；如作熵单因素对照，需要alpha及三个randomize和其余训练配置一致。
- 下一步：用户自行运行的新日志按实际metadata解释；当前没有待实施参数修改。

## U14：默认固定场景的修改方案 — 已完成（方案核对；未实施生产修改）

- 2026-10-12：用户要求将三个随机化开关默认关闭，明确reset及跨episode同一时隙的CU位置/轨迹/NLoS一致性；本次输出方案，不实施生产修改。
- 依赖：核对U13源码哈希与全关开关的已有验收，再检查reset、step取索引及配置覆盖。范围不扩展为动作固定、几何信道固定或跨worker共用场景。

### U14 验收证据与下一步

- 2026-10-12：核对U13所列9个源码SHA256仍一致，沿用全关场景一致性实测；reset固定分支复用数组，step按t索引CU轨迹和NLoS，满足同worker跨episode同一时隙一致性。
- 方案：madrl_args三个默认True改False并更新说明；保留配置/CLI覆盖、现有reset固定分支、动作RNG、alpha与熵。验收覆盖数组逐元素、真实step有限mock、时隙0…40、日志固定scene与递增episode。
- 证据：analysis/mhbppo_fixed_scene_defaults_20261012/proposal.md、source_evidence.json。仅输出方案，生产源码/参数未改，无训练/solver调用。
- 下一步：进入实施时按上述范围变更与验收；不得把当前默认True的U13事实改写为已实施False。固定NLoS不保证不同UAV动作下完整信道一致，跨worker各有固定场景。

## U15：实施默认固定场景 — 已完成（代码与有限环境验收通过）

- 2026-10-12：用户授权实施U14方案。修改三个随机化默认值及说明，保留现有reset固定分支、动作采样、alpha与熵；不重启正在运行的训练。
- 验收：默认配置、CLI/config覆盖、跨episode时隙0…40的CU/NLoS一致性及reset恢复、日志场景/RNG状态；真实env有限step配mock reward，既有训练配置回归。

### U15 验收证据与下一步

- 2026-10-12：三个默认开关False，更新help及reset说明；环境运行AST不改，alpha1.0/熵0.05与0.01保持。配置/CLI可显式开启随机化。
- 真实MyEnv共3个40步episode、120次step（mock reward），逐元素验证时隙0…40的CU及实际信道输入NLoS一致、所有reset恢复起点、固定scene与索引推进、torch RNG不重置；不同UAV动作终点不同。既有5项配置回归、语法与局部diff空白检查通过。
- 证据：analysis/mhbppo_fixed_scene_defaults_20261012/implementation/report.md、validation.json、configuration_tests.json、verify.py、changes.patch及source_manifest.json。未调用solver或启动/重启用户训练。
- 证据有效性更新：U13关于默认三个True的结论只描述改前版本，已由U15更新为False；U14为方案验收，本节为实施验收。此前固定场景历史训练分析及normalizer/日志诊断功能范围保留。
- 下一步：新启动直接python MHBPPO_main.py采用固定场景；旧JSON显式True会覆盖默认，需要自行使用False配置或显式no参数。当前授权修改完成，无待实施项。

