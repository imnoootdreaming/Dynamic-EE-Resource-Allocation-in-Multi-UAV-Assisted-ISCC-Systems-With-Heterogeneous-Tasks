# 本次 N2 的 PPO、探索与源码审计

结论：训练管线和各动作头均存在可核查的更新；没有发现本次默认配置下 GAE 跨环境串接、概率漏算、学习率为零或数值发散的证据。学习明显不均衡：CU 匹配与运动方向已形成状态依赖，接收波束及 off-duration 仍很接近均匀探索。优先做 beam 独立熵消融和优化诊断；在线物理辅助作为后一项独立实验。不能从这些观测直接证明熵是唯一原因，也不能重复推荐 U53 已失败的单次预热组合。

## 证据范围和来源

- 统一冻结 `frozen/training_updates.csv` 完整 update 1–5804，即 464320 transitions；TB 的 `BS/*` step 为更新前 iteration，必须加 1；`PolicyDistribution/*` step 已是完成 update。
- 流式读取本 run TensorBoard 文件，保留 57 个主要 tag，所有 tag 对 5804 次更新均无缺失。TB 按读取开始的字节边界解析完整 TFRecord，再按 update 过滤；未验证 TFRecord CRC，严格验证了记录长度和 protobuf 可解析性。没有读取训练时的网络前向或调用 solver。
- `source_checks.json` 比较 iter1000/2000/3000/4000/5000 的 30 个记录源码 SHA，均与当前字节相同；全部保存 actor/critic 文件哈希相符。它确认当前源码适用于该 run，不等于独立复现训练。
- `hyperparameters.json` 确认 shared encoder、空 policy_initialization/BC、Beta、alpha=1、N2×40、batch32、epochs5、gamma .99/lambda .95、连续/离散熵 .05/.01。未启用历史 dedicated 预热分支。
- 可复算脚本 `audit_ppo.py`；数值详见 `ppo_metrics.json`。本审计只新增分析产物，未改生产代码、启动/停止用户训练。

## 动作头现状

以下为更新 5305–5804 的 rollout 分布指标均值。std 是每状态条件分布 std，再对维度/采样/更新平均；不是确定性动作误差。不同头均值接近 .5 不能单独排除相反维度/状态之间的抵消，故同时报告条件均值的跨状态变化和实际物理证据。

|连续头|维数|u01 mean|条件 std|条件均值跨状态 std|Beta concentration|
|---|---:|---:|---:|---:|---:|
|uav_angles|4|.492239|.236527|.102932|3.828071|
|uav_distances|4|.473551|.275476|.049876|2.274037|
|uav_off_durations|4|.500529|.287351|.005305|2.027767|
|cu_off_powers|10|.451213|.272106|.037865|2.356803|
|receive beam real|40|.498792|.285580|.010978|2.063681|
|receive beam imag|40|.500063|.285920|.010812|2.056134|

均匀 U(0,1) 的 std 为 .288675，Beta(1,1) concentration 为 2。beam 的条件均值变化量约 .011，相比抽样噪声 .286 很小；方向头则约 .103/.237。结合物理分析的实际 gain 才能判断方向增益学习，不能仅凭 std 宣称所有干扰抑制均无学习。

|匹配头|首500 entropy|末500 entropy|末500 pmax|
|---|---:|---:|---:|
|UAV0|1.514319|.282387|.890856|
|UAV1|1.769469|.151725|.944180|
|UAV2|1.709398|.082828|.974819|
|UAV3|1.718073|.448760|.830343|

离散分布已显著集中，不能用“所有动作都没学到”描述本次训练。分布集中不自动等于全局最优匹配，仍应核重复 CU、必要裕度及状态依赖。

## PPO 实现合同及可疑瓶颈

1. `MHBPPO_agent.py:129–166,221–253`：每个连续维 Beta log-prob 和 differential entropy 求和；102 维总连续 log-prob、4 个总离散 log-prob。`467–475` 对连续组/离散组各算一个 ratio 并分别 clip，二者共用同一标量优势。这是本项目现有分组 surrogate；不是 6 个连续 head 各自独立更新，也不是标准联合 ratio。当前证据不足以直接把它换成逐头 ratio 或联合 ratio，且不能把 log-prob 的 sum 改为 mean，这会改变概率合同。
2. `398–445`：GAE 基于 critic TD residual，gamma .99/lambda .95，整批 80 条标准化一次；`rollout_batch.py:134–136,173–190` 强制每个完整 40 槽仅末步 done/real_done 为真，episode-major 拼接。因此 GAE 在每个环境的 40 槽末尾终止，不存在显见跨 worker 泄漏。
3. `462–489`：80 条分成 32/32/16，每 epoch 一次打乱，共 5 epoch，每网络 15 次 Adam；每样本使用 5 遍。全局梯度范数 clip=1。现无 clip fraction/gradient norm/per-head KL，不能证明 80 beam 维导致了 clipping 饱和或共享干梯度冲突；这是需要记录后判断的候选机制。
4. beam 占 80/102=78.43% 维度，但总熵 loss 不能简单按维数推断谁占优。末500 rollout 熵求和：beam −.256446、全部连续 −1.698625；系数 .05 后对应 loss +.012822/+ .084931，离散对应 −.009657。首500 beam 对应 +.149108。当前 beam 已接近最大熵、熵数值项变小；仍不代表其梯度阻力为零，也不能据 loss 值证明其主导梯度。
5. `MHBPPO_main.py:954–972`：奖励由 learner 使用历史统一 mean/std 归一化，然后 update，再更新 reward/state moments。采样和学习使用同一冻结 state-normalizer 快照；不能把这种统计演化直接判作概率错误。奖励规范化会使 critic target 尺度和分布随时间变化，原始 MSE 跨期下降不足以证明 value 已准确。
6. `BS/KL_Divergence` (`MHBPPO_agent.py:509–513`) 是同一采样批上的 signed old−new 联合 log-prob 平均。末500 均值 .029392、5%/95% 分位 −.060339/.125487，165/500 个值为负；负值是有限样本估计允许的现象。它不是精确非负 KL，也没有 per-head 分解，不能排除某个动作头更新过大。
7. critic loss 首500均值 8.349867（最大519.33），末500均值3.079049（最大17.15）；actor loss 末500均值 .012411。现有选取的主要指标均有限，未见持续数值爆炸证据；由于目标规范化演化、minibatch均值权重及缺少解释方差，不能由此认定 critic 充分或不足，亦不应仅据此增大 critic/epochs/LR。

## 优先修改的具体实现

**先补诊断，同时只改 beam 熵作为一个独立消融。** 保持当前动作空间、环境、采样 log-prob、old/new ratio、六元组默认返回不变。可在 actor 增加私有或显式可选的 details API，返回各活动 head entropy；`actor_loss` 将 80 beam 和22 nonbeam 的 entropy 分开系数，默认两者均 .05 精确兼容。实验臂只把 beam 从 .05 预注册退火至 .005（另一研究点可选0，不能事后看峰值切换），其余22维 .05、离散 .01。alpha=0 时仍须排除 masked beam 路径，不能给无效动作添梯度。低熵只能允许已有方向集中，不会凭空学习相位，不能将 std 下降作为验收。

在更新开始对同批冻结 states 缓存 detached old Beta alpha/beta 和 categorical probabilities，更新后对同 states 计算解析 KL；这是确定性前向，不需额外动作抽样，也无需从四位小数日志恢复参数。补连续组/离散组 ratio 分位、clip fraction，各 head KL，PPO/熵梯度范数，value explained variance、return/advantage std。记录 full-batch 指标，避免不等长 minibatch 均值带来的统计权重误读。默认兼容设置应对 action/log-prob/RNG/梯度做回归。

**若方向仍不建立，再单独试在线物理辅助/教师保持。** U53 同源10651的500/500真实更新已显示预热初期 gain≈.170 回到末100≈.100，末100 optimal4.35%低于原组5.95%；这不是本次冷启动的因果实验，但反对重复同一单次预热并承诺持续收益。当前 beam 只看 state，瞬时 teacher 依赖采样匹配/功率；可以对同批实际 eta,p 条件的物理商做期望辅助，不能要求同一 state 精确拟合冲突 teacher。使用 Beta.rsample 与 Torch 同源 2u−1/L2 映射，保持移动前信道/目标时序、复共轭和约1e−14噪声尺度。规范相位只在监督 teacher 中需要；相位不变物理商无需额外相位处理。共享干需监测或约束非beam KL，单独测 existing dedicated 架构，不能混合归因。

这些都是待验证候选，不是本次已实现或已证明的提速。

## LR、停止与保存边界

- `MHBPPO_main.py:652,862–871,1013–1014` 和 `MHBPPO_agent.py:498–528`：episodes10000 是按 update 的 LR horizon，不按 N2 累计 episode10000 自动停止。当前 max_updates0 表示持续运行。
- update5804 实际使用 LR约 .00012594，更新后LR约 .00012591，约初值42%；未到floor。因为调用 step 是更新前 iteration，更新9001后日志首达floor .00003，更新9002首实际用floor。不能把当前慢学习归结为已经到floor。
- `save_artifacts` (`MHBPPO_main.py:274–319`) 保存 actor/critic state_dict 与 exact state norm及版本/哈希，当前最新已落盘 iter5000。norm count800000=400000 transitions×2，含 raw state 与 raw next-state，与合同相符。
- 未保存 Adam moments、reward-normalizer、各进程 RNG、运行队列/时序，现有 policy_initialization 明确 fresh Adam/reward stats 且 is_resume=False。因此 checkpoint 可用于同版规范化评估及受控新分支，不能声称精确断点续训。源码无本次缺失权重证据。

## 对 proposed_changes.md 的限定审阅

beam 独立 entropy、保留六元组默认接口、先诊断再考虑 ratio改造、持续物理辅助分阶段、求解前保守证书均与当前源码相容。前置证书应仅对合法有限输入且 `margin<1−1e−6` 返回有证明的不可行；不满足证书时继续 solver，raw solver status 必须明确未执行。前置证书不是新可行性判据，不应将证书通过当作可行。该优化可能节约 solver 时间，但不得由过滤比例直接声称同等墙钟倍率；当前已存在 Fusion model cache，不能重复建议“增加缓存”。
