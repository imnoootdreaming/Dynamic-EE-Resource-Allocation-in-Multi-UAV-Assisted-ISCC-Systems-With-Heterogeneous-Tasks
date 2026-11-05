# MHGPPO：旧版 Gaussian PPO 的当前环境入口

在仓库根目录启动（使用安装 PyTorch、Gymnasium、CVXPY、MOSEK 和 TensorBoard 的 Python）：

```powershell
python src/outer/algorithms/MHGPPO/MHGPPO_main.py --num_samplers 2 --worker_num_threads 1 --mosek_num_threads 1
```

有限短测可增加 `--max_updates 2`；每轮同步收齐 `num_samplers` 个完整 episode 后更新一次。`--total_time_slots` 必须满足当前场景目标调度的整除条件。`--max_updates 0` 表示不限制训练轮数；正数表示更新次数上限。当 `--max_updates` 为正数时，学习率衰减尺度自动等于该值；持续训练（`--max_updates 0`）时使用 `--episodes`，默认 10000。同步多环境的场景随机化限制沿用当前参数校验。

沿用旧版策略和 `Normalization`/`RewardScaling`：Gaussian 加 sigmoid、旧 log probability、Normal 熵、连续/离散分别 clip、固定 minibatch 32、熵系数 0.05/0.01。学习率按 MHBPPO 的线性衰减公式下降并保持 `--lr_min_ratio` 下限（默认初始值的 10%）；超过衰减尺度后仍可继续训练。非默认的 MHBPPO 专属 batch/熵参数会报错，避免误以为它们生效；策略初始化、BC、专属编码器和异步模式不支持。环境的波束映射配置仍由当前环境使用，旧 actor 不屏蔽任何动作概率。

单环境保留旧版逐状态在线更新归一化统计的顺序；多环境每轮共享冻结快照，轮末按 worker 顺序合并原始状态和下一状态，首轮无统计时使用零均值、单位标准差。每个 worker 拥有独立的旧版 reward scaler，episode 开始重置折扣回报，统计跨该 worker 的 episode 保留。多环境训练的统计、batch 和更新轨迹因此不等同于旧版单环境。

产物按同一 run_id 分开保存：`src/outer/ckpt/MHGPPO/<run_id>/` 含启动时生成的 env.png、scene_signature.json、配置、逐轮 CSV、episode 元数据和 iter_N checkpoint；`src/outer/logs/MHGPPO/<run_id>/` 含训练日志和逐时隙动作/环境 JSONL；`src/outer/runs/MHGPPO/<run_id>/` 含 TensorBoard。checkpoint 为更新后的模型，PPO 轨迹为更新前策略；MHSAC 轨迹逐时隙记录策略版本。

检查点用于评估，不包含优化器、随机数、奖励缩放器或完整训练状态，不支持精确续训。评估时可用 `algorithms.MHGPPO.checkpoint.load_normalization(iter_dir, expected_dim)` 校验 actor 哈希并加载匹配统计，再用 `normalize_state` 处理观测；该文件格式不与 MHBPPO 的同名归一化文件互换。

逐步日志的 `--log_detail_level`、`--env_diagnostics`、`--log_rotate_*` 和 `--log_min_free_bytes` 控制记录与保留；低磁盘空间暂停详细日志，训练 CSV 继续记录。`--log_file` 若显式指定自定义路径，则日志目录按其父目录和 run_id 建立。

固定场景同步训练检查每个 worker 的场景签名；跨算法比较：

```powershell
python src/outer/algorithms/compare_baseline_scenes.py <MHBPPO运行目录> <baseline运行目录>
```

签名匹配表示记录覆盖的外生场景与物理常量一致；算法动作、策略轨迹、Pure 奖励及统计显著性仍需单独评估。
# 接收波束动作

环境解析计算 MVDR 接收波束，MHGPPO 固定使用 MVDR、alpha=0、无残差。策略不输出接收波束幅度和相位，默认连续动作从 102 维减少到 22 维。旧 checkpoint 与新网络维度不兼容，需要重新训练。
