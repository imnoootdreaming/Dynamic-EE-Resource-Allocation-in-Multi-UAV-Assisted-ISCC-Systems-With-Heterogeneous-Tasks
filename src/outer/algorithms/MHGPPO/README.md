# MHGPPO：旧版 Gaussian PPO 的当前环境入口

在仓库根目录启动（使用安装 PyTorch、Gymnasium、CVXPY、MOSEK 和 TensorBoard 的 Python）：

```powershell
python src/outer/algorithms/MHGPPO/MHGPPO_main.py --num_samplers 2 --worker_num_threads 1 --mosek_num_threads 1
```

有限短测可增加 `--max_updates 2`；每轮同步收齐 `num_samplers` 个完整 episode 后更新一次。`--total_time_slots` 必须满足当前场景目标调度的整除条件。训练最多执行 `--episodes` 轮（旧版学习率线性衰减且无下限）；`--max_updates 0` 表示不另设更小上限，并非无限训练。同步多环境的场景随机化限制沿用当前参数校验。

旧版 agent 和 `Normalization`/`RewardScaling` 按原代码保留：Gaussian 加 sigmoid、旧 log probability、Normal 熵、连续/离散分别 clip、固定 minibatch 32、熵系数 0.05/0.01 和原学习率衰减。当前 MHBPPO 专属的 batch/熵/学习率下限参数如被改成非默认值会报错，避免误以为它们生效；策略初始化、BC、专属编码器和异步模式不支持。环境的波束映射配置仍由当前环境使用，旧 actor 不屏蔽任何动作概率。

单环境保留旧版逐状态在线更新归一化统计的顺序；多环境每轮共享冻结快照，轮末按 worker 顺序合并原始状态和下一状态，首轮无统计时使用零均值、单位标准差。每个 worker 拥有独立的旧版 reward scaler，episode 开始重置折扣回报，统计跨该 worker 的 episode 保留。多环境训练的统计、batch 和更新轨迹因此不等同于旧版单环境。

产物位于 `src/outer/runs/gaussian-hppo/<时间及唯一标识>/`：配置、每轮 CSV、训练日志、TensorBoard、每 episode 策略/场景身份，以及 `iter_N/` 中更新后的 actor/critic、对应状态归一化、训练曲线 CSV 快照和采样轨迹（NPY、PNG）。轨迹来自更新前的 `policy_version=N-1`，checkpoint 是更新后的 `N`；`completion_rate` 是当前环境返回的成功时隙比例。CSV 中 `finite_obj_mean` 仅平均有限的目标值，同时给出有限样本数；求解状态分布单独记录。

检查点用于评估，不包含优化器、随机数、奖励缩放器或完整训练状态，不支持精确续训。评估时可用 `algorithms.MHGPPO.checkpoint.load_normalization(iter_dir, expected_dim)` 校验 actor 哈希并加载匹配统计，再用 `normalize_state` 处理观测；该文件格式不与 MHBPPO 的同名归一化文件互换。`mhgppo_train.log` 为逐轮摘要；当前实现不生成 MHBPPO 的逐步动作/环境 JSONL 和日志轮转文件。
