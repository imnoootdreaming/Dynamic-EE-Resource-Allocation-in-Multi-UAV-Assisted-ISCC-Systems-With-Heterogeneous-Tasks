# SHBPPO：旧版单头 Beta PPO 的当前环境入口

在仓库根目录启动（使用安装 PyTorch、Gymnasium、CVXPY、MOSEK 和 TensorBoard 的 Python）：

```powershell
python src/outer/algorithms/SHBPPO/SHBPPO_main.py --num_samplers 2 --worker_num_threads 1 --mosek_num_threads 1
```

增加 `--max_updates 2` 可做有限短测。每轮同步收齐 `num_samplers` 个完整 episode，再更新一次；`--total_time_slots` 必须满足当前目标调度的整除条件。默认 `--episodes 8000` 与旧版一致，且是旧学习率衰减与训练轮数上限；`--max_updates 0` 不另外设小上限。显式 CLI 或配置文件中的 `episodes` 优先。

agent 与旧版文件相同：单个 Beta 向量的前半部分用于当前环境连续动作，后面每 UAV 一维经旧版均匀分桶映射到 CU。PPO 存储分桶前的完整潜动作，使用联合 log probability、entropy 和联合 ratio/clip；不屏蔽当前环境中未生效的维度。固定 minibatch 32、熵系数 0.01，学习率按旧版线性衰减且无下限。若修改 MHBPPO 专属 batch/熵/学习率下限参数会报错；策略初始化、BC、专属编码器和异步模式不支持。

单环境保留旧版归一化顺序：reset 后更新一次，随后只在每一步更新下一状态。多环境每轮共享冻结快照，轮末按 worker 顺序合并每个 episode 的初始状态及各下一状态；首轮无统计时用零均值、单位标准差。每个 worker 的旧版 reward scaler 独立保存运行统计，episode 开始重置折扣回报。多环境统计、batch 及更新轨迹不等同于旧版单环境。

产物位于 `src/outer/runs/shbppo/<时间及唯一标识>/`：配置、每轮 CSV、训练日志、TensorBoard、每 episode 策略/场景身份，以及 `iter_N/` 的更新后 actor/critic、对应状态归一化、训练 CSV 快照和采样轨迹（NPY、PNG）。轨迹来自更新前策略 `N-1`；checkpoint 来自更新后策略 `N`。`completion_rate` 为当前环境成功时隙比例；`finite_obj_mean` 仅平均有限目标值，并附带有限样本数和求解状态分布。

检查点仅用于评估，不包含优化器、随机数、奖励缩放器或完整训练状态，不能精确续训。可通过 `algorithms.SHBPPO.checkpoint.load_normalization(iter_dir, expected_dim)` 校验 actor 哈希并加载归一化，再用 `normalize_state` 处理状态。训练日志仅有逐轮摘要，不生成 MHBPPO 的逐时隙动作/环境 JSONL 和日志轮转文件。

测试旧版一致性：

```powershell
python -m unittest discover -s src/outer/algorithms/SHBPPO -p test_shbppo_contract.py -v
```

检查一次训练保存的结果：

```powershell
python src/outer/algorithms/SHBPPO/verify_run.py <运行目录>
```
