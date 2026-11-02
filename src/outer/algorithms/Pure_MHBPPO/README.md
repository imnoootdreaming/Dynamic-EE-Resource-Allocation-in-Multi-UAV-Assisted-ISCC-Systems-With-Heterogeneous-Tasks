# Pure-MHBPPO：无内层优化求解的直接决策基线

在仓库根目录运行（使用安装 PyTorch、Gymnasium、CVXPY、MOSEK 和 TensorBoard 的现有项目 Python 环境）：

```powershell
python src/outer/algorithms/Pure_MHBPPO/Pure_MHBPPO_main.py --num_samplers 2 --worker_num_threads 1
```

`--max_updates 2` 可限制为两轮短测；`--total_time_slots` 须符合当前目标调度的整除条件。默认 `episodes=8000`，每轮同步收齐所有环境完整 episode 后更新一次，`--max_updates 0` 表示不另设更小轮数上限。旧 PPO 的 mini-batch 固定32，连续/离散熵系数0.05/0.01，学习率按旧版衰减且无下限；不支持 MHBPPO 专用初始化、BC、专属编码器及异步训练。

保留旧版多头 Beta/Categorical 策略和旧 Pure 的 11 组连续动作：飞行角/距离、UAV 卸载时长、感知和卸载波束方向及功率、CU 发射功率和卸载时长；每架 UAV 另有一个离散 CU 头。默认 I=4、J=10、N=10 时为200维连续动作＋4个十类离散头。接收波束为基于当前 CU 匹配、功率和策略感知波束的解析 MVDR；其中 `numpy.linalg.solve` 解线性方程，不是 CVXPY 或内层优化。MHBPPO 的 `structured_beam_alpha` 在此不生效，启动时会提示；不支持接收波束残差、相位残差或混合模式。

环境复用当前场景、任务、信道和状态构造。当前任务时隙的 `L_j,C_j,D_j` 用于直接计算感知/卸载速率、三阶段 CU 传输、最小计算频率、实际候选能耗和附件 P0 的15类约束。频率总量超限不会被等比例压缩后冒充可行；分母非正的正负载频率不可行。当前实现按已有 `MyEnv` 时序先用当前信道结算、再更新位置和信道。

训练奖励采用 `exp(-candidate_energy/1000) - spectrum_penalty - collision_penalty - candidate_constraint_penalty`，无有限候选能耗时能量项为零；候选约束罚为0或1。重复匹配的 CU 三阶段不存在合法解释，该 CU 不因其无定义的三阶段结果重复增加候选罚，但仍记录原始约束残差并给予频谱罚。`physical_success` 要求动作域和全部15类约束通过，容差与逐项残差保存在物理日志中。这是**附件物理口径的新 reward**，不同于旧 Pure 的虚拟有效功率/tanh 奖励，也不等同于 MHBPPO 的内层求解无解罚。一个候选动作不可行不代表整个优化问题无解。

运行路径不调用 CVXPY、MOSEK Fusion、PC3P 或 CCCP 求解器；但目前复用的 `MyEnv` 顶层仍导入其求解器相关模块，因此**仍要求原项目依赖能够导入**，未承诺在未安装 CVXPY/MOSEK 的环境中启动。`guarded_smoke.py` 可拦截这些求解入口以验收无调用。

结果位于 `src/outer/runs/pure-mhbppo/<时间及唯一标识>/`：配置和源码/附件 SHA256、训练 CSV/日志/TensorBoard、episode 版本及场景、`physical_traces.jsonl` 中原始/映射动作、接收波束、物理量和逐项约束；详细日志按现有 `--log_rotate_*` 参数轮转，低于 `--log_min_free_bytes` 时暂停详细日志。`iter_N/` 保存更新后 actor/critic、配对状态归一化、CSV 快照以及更新前 `N-1` 策略的轨迹。多环境首轮使用零均值单位标准差，轮末聚合状态统计；每个 worker 独立维持旧 reward scaler。checkpoint 仅用于评估，不支持精确续训。

检查点的状态归一化可通过 `algorithms.Pure_MHBPPO.checkpoint.load_normalization(iter_dir, expected_dim)` 校验 actor 哈希并加载；`normalize_state` 负责归一化。验证：

```powershell
python -m unittest discover -s src/outer/algorithms/Pure_MHBPPO -p test_pure_contract.py -v
python src/outer/algorithms/Pure_MHBPPO/verify_run.py <运行目录>
```
