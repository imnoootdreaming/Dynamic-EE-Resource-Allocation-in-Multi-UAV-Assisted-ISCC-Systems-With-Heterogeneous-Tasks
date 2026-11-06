# SHBPPO：旧版单头 Beta PPO 的当前环境入口

在仓库根目录启动（使用安装 PyTorch、Gymnasium、CVXPY、MOSEK 和 TensorBoard 的 Python）：

```powershell
python src/outer/algorithms/SHBPPO/SHBPPO_main.py --num_samplers 2 --worker_num_threads 1 --mosek_num_threads 1
```

增加 `--max_updates 2` 可做有限短测。每轮同步收齐 `num_samplers` 个完整 episode，再更新一次；`--total_time_slots` 必须满足当前目标调度的整除条件。`--max_updates 0` 表示不限制训练轮数，正数表示更新次数上限。当 `--max_updates` 为正数时，学习率衰减尺度自动等于该值；持续训练（`--max_updates 0`）时使用 `--episodes`，默认 8000。

策略和动作映射沿用旧版：单个 Beta 向量的前半部分用于当前环境连续动作，后面每 UAV 一维经旧版均匀分桶映射到 CU。PPO 存储分桶前的完整潜动作，使用联合 log probability、entropy 和联合 ratio/clip；不屏蔽当前环境中未生效的维度。固定 minibatch 32、熵系数 0.01，学习率按 MHBPPO 的线性衰减公式下降并保持 `--lr_min_ratio` 下限（默认初始值的 10%）。若修改 MHBPPO 专属 batch/熵参数会报错；策略初始化、BC、专属编码器和异步模式不支持。

状态归一化：单环境保留旧版顺序，reset 后更新一次，随后只在每一步更新下一状态。多环境每轮共享冻结快照，轮末按 worker 顺序合并每个 episode 的初始状态及各下一状态；首轮无统计时用零均值、单位标准差。

奖励预处理与 MHBPPO 一致：worker 只返回原始 `reward["bs"]`；learner 合并整轮样本，用更新前的全局历史即时奖励均值和标准差做 `(r-mean)/(std+1e-8)`，PPO 更新成功后再加入本轮原始奖励。直接复用 MHBPPO 的 `StateNormStats`（包括初始单位标准差、`1e-4` 下限），不再使用各 worker 独立的折扣回报 scaler。日志与奖励曲线继续使用原始环境 reward。状态归一化口径仍与 MHBPPO 不同。

可选的 `--expected_duplicate_matching_coef` 为 Actor loss 加入期望重复 CU 超额次数：先用 Beta CDF 差值计算每个 UAV 的实际 CU 分桶概率，再按独立选择概率计算 `E[sum_j max(N_j-1, 0)]`。CDF 使用可微、双精度连分数计算，保留原有潜动作采样、联合 log probability、entropy 和 ratio/clip。Python 参数默认 `0`；后台脚本与 MHBPPO 一样默认传入 `0.3`，显式传 `0` 可关闭。`training_updates.csv` 和 TensorBoard 的 `actor_expected_duplicate_matching` 记录更新后策略在本轮状态上的期望；关闭时 CSV 记为 NaN。该辅助目标可直接抑制重复匹配倾向，但效果需训练对照验证，不能保证提升完成率。

产物按同一 run_id 分开保存：`src/outer/ckpt/SHBPPO/<run_id>/` 含启动时生成的 env.png、scene_signature.json、配置、逐轮 CSV、episode 元数据和 iter_N checkpoint；`src/outer/logs/SHBPPO/<run_id>/` 含训练日志和逐时隙动作/环境 JSONL；`src/outer/runs/SHBPPO/<run_id>/` 含 TensorBoard。checkpoint 为更新后的模型，PPO 轨迹为更新前策略；MHSAC 轨迹逐时隙记录策略版本。

检查点仅用于评估，不包含优化器、随机数、全局奖励统计或完整训练状态，不能精确续训。可通过 `algorithms.SHBPPO.checkpoint.load_normalization(iter_dir, expected_dim)` 校验 actor 哈希并加载状态归一化，再用 `normalize_state` 处理状态。

新增辅助损失和奖励口径测试（不依赖 `outer_old`）：

```powershell
python src/outer/algorithms/SHBPPO/test_matching_reward.py -v
```

测试旧版一致性：

```powershell
python -m unittest discover -s src/outer/algorithms/SHBPPO -p test_shbppo_contract.py -v
```

检查一次训练保存的结果：

```powershell
python src/outer/algorithms/SHBPPO/verify_run.py <运行目录>
```

逐步日志的 `--log_detail_level`、`--env_diagnostics`、`--log_rotate_*` 和 `--log_min_free_bytes` 控制记录与保留；低磁盘空间暂停详细日志，训练 CSV 继续记录。`--log_file` 若显式指定自定义路径，则日志目录按其父目录和 run_id 建立。

固定场景同步训练检查每个 worker 的场景签名；跨算法比较：

```powershell
python src/outer/algorithms/compare_baseline_scenes.py <MHBPPO运行目录> <baseline运行目录>
```

签名匹配表示记录覆盖的外生场景与物理常量一致；算法动作、策略轨迹、Pure 奖励及统计显著性仍需单独评估。
# 接收波束动作

环境解析计算 MVDR 接收波束，SHBPPO 固定使用 MVDR、alpha=0、无残差。单头策略不输出接收波束幅度和相位，默认连续动作从 102 维减少到 22 维，加上 4 维离散分桶潜动作共 26 维。旧 checkpoint 与新网络维度不兼容，需要重新训练。
