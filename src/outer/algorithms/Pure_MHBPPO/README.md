# Pure-MHBPPO：无内层优化求解的直接决策基线

在仓库根目录运行（使用安装 PyTorch、Gymnasium、CVXPY、MOSEK 和 TensorBoard 的现有项目 Python 环境）：

```powershell
python src/outer/algorithms/Pure_MHBPPO/Pure_MHBPPO_main.py --num_samplers 2 --worker_num_threads 1
```

`--max_updates 2` 可限制为两轮短测；`--total_time_slots` 须符合当前目标调度的整除条件。每轮同步收齐所有环境完整 episode 后更新一次，`--max_updates 0` 表示不限制训练轮数。有限训练时学习率衰减尺度自动等于 `--max_updates`；持续训练时使用 `--episodes`，默认 8000。学习率按 MHBPPO 的线性衰减公式下降并保持 `--lr_min_ratio` 下限（默认初始值的 10%）。PPO 的 mini-batch 固定32，连续/离散熵系数0.05/0.01；不支持 MHBPPO 专用初始化、BC、专属编码器及异步训练。

多头 Beta/Categorical 策略输出 8 组连续动作，包括飞行角、飞行距离、UAV 卸载时长、UAV 卸载波束方向的实部和虚部、UAV 卸载功率、CU 发射功率和 CU 卸载时长。每架 UAV 另有一个离散 CU 匹配头。默认 I=4、J=10、N=10 时为 116 维连续动作和 4 个十类离散头。旧 200 维 actor 检查点使用原动作结构。

匹配辅助损失与 MHBPPO 一致：按独立 Categorical 头的概率计算 `E[sum_j max(N_j-1, 0)]`，以 `--expected_duplicate_matching_coef` 加入 Actor loss，保留全部 CU 选择、原始潜变量和原 log-prob/entropy/ratio。Python 参数默认 `0`，后台脚本默认 `0.3`，显式传 `0` 可关闭。CSV/TensorBoard 的 `actor_expected_duplicate_matching` 是更新后策略在本轮状态上的期望；关闭时 CSV 为 NaN。该辅助项是否改善长期完成率需训练消融验证。

奖励预处理与 MHBPPO/SHBPPO/MHGPPO 一致：worker 返回原始 `reward["bs"]`；learner 用更新前的全局历史即时奖励均值和标准差做 `(r-mean)/(std+1e-8)`，PPO 更新成功后加入整轮原始奖励。直接复用 MHBPPO 的 `StateNormStats`，包括初始单位标准差和 `1e-4` 下限。状态归一化沿用现有单环境在线／多环境冻结快照顺序。原始物理 reward 公式及日志/曲线口径保持不变。checkpoint 不保存全局奖励统计、优化器或随机数，仍只用于评估，不能精确续训。

两套多头 PPO 的匹配辅助项与奖励口径回归检查：`python src/outer/algorithms/MHGPPO/test_matching_reward.py -v`。

新运行默认开启 **时间条件解码**（`--pure_time_decoder conditional`）。仍保留全部 116 个连续潜变量，网络输出和 PPO 概率均使用原始 `[0,1]` Beta 样本；只有执行时的 4 个 UAV 时间和 10 个 CU 时间按当前匹配、速率、任务和共享 BS 算力映射。先为未解码任务保留最小必要算力，再逐个确定当前时间的上下界，减少「单个时间都在范围内、组合后却来不及上传或计算」的无效尝试。速率与原物理评价共用一次计算，不新增优化求解器。

重复匹配、零上传速率或当前功率/波束组合无法补全时间时，保留原时间映射及原处罚；不会把它当成可行动作。其余移动、功率、波束范围和原始 reward 公式不变。时间解码本身不改变熵或学习率；是否能在 5000 轮内稳定超过 reward=0.90 仍需完整训练验证。

对照旧映射使用 `--pure_time_decoder legacy`。加载**未记录该选项的旧配置**时默认保持 `legacy`；若沿用旧配置测试新解码，必须显式追加：

```powershell
python src/outer/algorithms/Pure_MHBPPO/Pure_MHBPPO_main.py --config <旧配置路径> --pure_time_decoder conditional
```

启动日志显示 `time_decoder=conditional_v1`，每轮显示 `time_decoded=N/总槽数` 和 `time_fallback=M`；CSV/TensorBoard 保存对应计数。物理 trace 的 `time_decoding` 记录原因、实际区间和算力；`raw_continuous` 保留网络原始样本，`mapped_continuous` 记录执行动作。配置与 checkpoint 元数据保存解码版本，旧 checkpoint 不会默认为新映射。

环境由当前目标信道计算感知发射方向，并根据当前 CU 匹配、CU 功率和 MVDR 接收增益选择感知功率，目标为线性 SINR 门限的 1.01 倍。若所需功率超过上限，则按上限执行并由原物理约束判定是否达标。发射功率计算和 MVDR 接收复用同一次 `numpy.linalg.solve`。物理日志继续记录实际执行的感知波束。

环境复用当前场景、任务、信道和状态构造。当前任务时隙的 `L_j,C_j,D_j` 用于直接计算感知/卸载速率、三阶段 CU 传输、最小计算频率、实际候选能耗和附件 P0 的15类约束。频率总量超限不会被等比例压缩后冒充可行；分母非正的正负载频率不可行。当前实现按已有 `MyEnv` 时序先用当前信道结算、再更新位置和信道。

训练奖励采用 `exp(-candidate_energy/1000) - spectrum_penalty - collision_penalty - candidate_constraint_penalty`，无有限候选能耗时能量项为零；候选约束罚为0或1。重复匹配的 CU 三阶段不存在合法解释，该 CU 不因其无定义的三阶段结果重复增加候选罚，但仍记录原始约束残差并给予频谱罚。`physical_success` 要求动作域和全部15类约束通过，容差与逐项残差保存在物理日志中。这是**附件物理口径的新 reward**，不同于旧 Pure 的虚拟有效功率/tanh 奖励，也不等同于 MHBPPO 的内层求解无解罚。一个候选动作不可行不代表整个优化问题无解。

运行路径不调用 CVXPY、MOSEK Fusion、PC3P 或 CCCP 求解器；但目前复用的 `MyEnv` 顶层仍导入其求解器相关模块，因此**仍要求原项目依赖能够导入**，未承诺在未安装 CVXPY/MOSEK 的环境中启动。`guarded_smoke.py` 可拦截这些求解入口以验收无调用。

产物按同一 run_id 分开保存：`src/outer/ckpt/Pure_MHBPPO/<run_id>/` 含启动时生成的 env.png、scene_signature.json、配置、逐轮 CSV、episode 元数据和 iter_N checkpoint；`src/outer/logs/Pure_MHBPPO/<run_id>/` 含训练日志和逐时隙动作/环境 JSONL；`src/outer/runs/Pure_MHBPPO/<run_id>/` 含 TensorBoard。checkpoint 为更新后的模型，PPO 轨迹为更新前策略；MHSAC 轨迹逐时隙记录策略版本。 Pure 的 physical_traces.jsonl 位于 logs 目录，记录直接候选约束与能耗，始终按完整物理格式保存；不调用内层求解器。

检查点的状态归一化可通过 `algorithms.Pure_MHBPPO.checkpoint.load_normalization(iter_dir, expected_dim)` 校验 actor 哈希并加载；`normalize_state` 负责归一化。验证：

重建评价环境应先用 `checkpoint.load_environment_args(iter_dir)` 恢复配套配置，再构造 `PureMyEnv`，以同时恢复正确的时间映射。不要仅加载 actor 后使用另一种解码模式评价。

```powershell
python -m unittest discover -s src/outer/algorithms/Pure_MHBPPO -p test_pure_contract.py -v
python -m unittest discover -s src/outer/algorithms/Pure_MHBPPO -p test_time_decoder.py -v
python src/outer/algorithms/Pure_MHBPPO/verify_run.py <运行目录>
```

逐步日志的 `--log_detail_level`、`--env_diagnostics`、`--log_rotate_*` 和 `--log_min_free_bytes` 控制记录与保留；低磁盘空间暂停详细日志，训练 CSV 继续记录。`--log_file` 若显式指定自定义路径，则日志目录按其父目录和 run_id 建立。

固定场景同步训练检查每个 worker 的场景签名；跨算法比较：

```powershell
python src/outer/algorithms/compare_baseline_scenes.py <MHBPPO运行目录> <baseline运行目录>
```

签名匹配表示记录覆盖的外生场景与物理常量一致；算法动作、策略轨迹、Pure 奖励及统计显著性仍需单独评估。
