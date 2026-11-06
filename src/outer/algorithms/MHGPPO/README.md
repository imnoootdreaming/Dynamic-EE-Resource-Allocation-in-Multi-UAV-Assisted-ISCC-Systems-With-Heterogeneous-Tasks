# MHGPPO：旧版 Gaussian PPO 的当前环境入口

在仓库根目录启动（使用安装 PyTorch、Gymnasium、CVXPY、MOSEK 和 TensorBoard 的 Python）：

```powershell
python src/outer/algorithms/MHGPPO/MHGPPO_main.py --num_samplers 2 --worker_num_threads 1 --mosek_num_threads 1
```

有限短测可增加 `--max_updates 2`；每轮同步收齐 `num_samplers` 个完整 episode 后更新一次。`--total_time_slots` 必须满足当前场景目标调度的整除条件。`--max_updates 0` 表示不限制训练轮数；正数表示更新次数上限。当 `--max_updates` 为正数时，学习率衰减尺度自动等于该值；持续训练（`--max_updates 0`）时使用 `--episodes`，默认 10000。同步多环境的场景随机化限制沿用当前参数校验。

保留旧版 Gaussian 加 sigmoid、log probability、Normal 熵、连续/离散分别 clip、固定 minibatch 32、熵系数 0.05/0.01。学习率按 MHBPPO 的线性衰减公式下降并保持 `--lr_min_ratio` 下限（默认初始值的 10%）；超过衰减尺度后仍可继续训练。非默认的 MHBPPO 专属 batch/熵参数会报错，避免误以为它们生效；策略初始化、BC、专属编码器和异步模式不支持。

状态归一化：单环境保留旧版逐状态在线更新统计的顺序；多环境每轮共享冻结快照，轮末按 worker 顺序合并原始状态和下一状态，首轮无统计时使用零均值、单位标准差。

奖励预处理与 MHBPPO/SHBPPO 一致：worker 返回原始 `reward["bs"]`，learner 合并整轮样本后使用更新前的全局历史即时奖励统计，计算 `(r-mean)/(std+1e-8)`，PPO 更新成功后才加入本轮原始奖励。直接复用 MHBPPO 的 `StateNormStats`，包括初始单位标准差和 `1e-4` 标准差下限；不再使用各 worker 的折扣回报 scaler。日志和奖励曲线仍记录原始环境 reward。

`--expected_duplicate_matching_coef` 为 Actor loss 增加 `E[sum_j max(N_j-1, 0)]`，按独立 Categorical 头的 CU 概率直接计算，与 MHBPPO 同公式。保留全部 CU 采样选择及原 log-prob/entropy/ratio。Python 参数默认 `0`；后台脚本默认 `0.3`，显式传 `0` 可关闭。CSV 和 TensorBoard 的 `actor_expected_duplicate_matching` 是更新后策略在本轮状态上的期望，关闭时 CSV 为 NaN。该辅助项是否改善长期完成率需训练消融验证。

产物按同一 run_id 分开保存：`src/outer/ckpt/MHGPPO/<run_id>/` 含启动时生成的 env.png、scene_signature.json、配置、逐轮 CSV、episode 元数据和 iter_N checkpoint；`src/outer/logs/MHGPPO/<run_id>/` 含训练日志和逐时隙动作/环境 JSONL；`src/outer/runs/MHGPPO/<run_id>/` 含 TensorBoard。checkpoint 为更新后的模型，PPO 轨迹为更新前策略；MHSAC 轨迹逐时隙记录策略版本。

检查点用于评估，不包含优化器、随机数、全局奖励统计或完整训练状态，不支持精确续训。评估时可用 `algorithms.MHGPPO.checkpoint.load_normalization(iter_dir, expected_dim)` 校验 actor 哈希并加载匹配统计，再用 `normalize_state` 处理观测；该文件格式不与 MHBPPO 的同名归一化文件互换。

两套多头 PPO 的匹配辅助项和奖励口径回归检查（不依赖 `outer_old`）：

```powershell
python src/outer/algorithms/MHGPPO/test_matching_reward.py -v
```

逐步日志的 `--log_detail_level`、`--env_diagnostics`、`--log_rotate_*` 和 `--log_min_free_bytes` 控制记录与保留；低磁盘空间暂停详细日志，训练 CSV 继续记录。`--log_file` 若显式指定自定义路径，则日志目录按其父目录和 run_id 建立。

固定场景同步训练检查每个 worker 的场景签名；跨算法比较：

```powershell
python src/outer/algorithms/compare_baseline_scenes.py <MHBPPO运行目录> <baseline运行目录>
```

签名匹配表示记录覆盖的外生场景与物理常量一致；算法动作、策略轨迹、Pure 奖励及统计显著性仍需单独评估。
# 接收波束动作

环境解析计算 MVDR 接收波束，MHGPPO 固定使用 MVDR、alpha=0、无残差。策略不输出接收波束幅度和相位，默认连续动作从 102 维减少到 22 维。旧 checkpoint 与新网络维度不兼容，需要重新训练。
