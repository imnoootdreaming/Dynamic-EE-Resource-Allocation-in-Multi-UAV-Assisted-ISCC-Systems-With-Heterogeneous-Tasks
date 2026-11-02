# MHSAC：旧版混合 SAC 的当前环境入口

在仓库根目录启动（使用安装 PyTorch、Gymnasium、CVXPY、MOSEK 和 TensorBoard 的 Python）：

```powershell
python src/outer/algorithms/MHSAC/MHSAC_main.py --num_samplers 2 --worker_num_threads 1 --mosek_num_threads 1
```

旧版 MHSAC agent、Replay Buffer 和归一化代码原样保留：Gaussian＋sigmoid 连续头、Categorical 离散头、离散联合动作精确枚举、双 Q/目标 Q、两种自动温度及软更新。每个时隙所有环境并行执行，learner 按 worker 编号依次把每条 transition 加入共享 Replay Buffer；达到 `max(minimal_size, batch_size)` 后，每条 transition 均执行 `updates_per_step` 次 SAC 更新。单环境按照旧版顺序更新状态统计、奖励缩放、回放和策略。

多环境一个时隙的动作在同一次 learner 采样阶段产生；按 worker 顺序处理其 transition 时，后处理的样本可能落后于已完成的梯度更新，CSV 和 episode 元数据记录每一步的策略版本及滞后。多环境共用一个状态归一化器，按 worker 顺序更新；每个环境独立保留旧版 reward scaler 的统计并在 episode 起始重置折扣回报。各环境使用独立动作随机流；单环境使用旧式 learner 随机流。并行采样顺序与旧版单环境不完全相同。

主要旧版默认参数：`--batch_size 128`、`--buffer_size 100000`、`--updates_per_step 1`、`--tau 0.005`、两个初始温度均为 0.2、`--alpha_lr 0.00003`。默认 `--minimal_size 0` 自动解析为 `max(4096, 200 * total_time_slots)`；40 时隙时预热为 8000 条 transition，在此之前不会更新 SAC 网络。`--episodes` 默认 8000 个同步 episode 轮次；`--max_updates` 是为兼容现有入口而设置的**episode 轮次上限**，不是 SAC 梯度更新次数。`--max_updates 0` 不另设上限。

需要短测梯度链路时可临时减小预热和 batch（属于测试配置）：

```powershell
python src/outer/algorithms/MHSAC/MHSAC_main.py --num_samplers 2 --worker_num_threads 1 --mosek_num_threads 1 --max_updates 2 --total_time_slots 10 --hidden_dim 16 --batch_size 2 --minimal_size 2 --buffer_size 100
```

`--total_time_slots` 必须满足当前目标调度的整除条件。当前 `--no-sync_training`、MHBPPO 的策略初始化、BC 和专属编码器不支持；PPO 的 `epochs`、clip 和熵系数不参与 SAC 更新。`--config` 可读取本入口输出的 `configuration.json`，支持保存的 `hyperparameters`、`sac_hyperparameters`，CLI 参数优先。

启动时会打印精确离散组合数、一个 float32 联合动作 batch 张量的大小，以及装满 replay 时状态和动作数组的估算大小；实际内存还包括 Python 容器、多个中间张量、Q 网络、优化器、环境和求解器。默认 4 UAV、10 CU 时共有 10,000 个联合动作。可用 `python src/outer/algorithms/MHSAC/benchmark_exact.py` 在 CPU 上测量实际默认维度的一次更新；该测量不代表并行完整训练峰值。

产物位于 `src/outer/runs/mhsac/<时间及唯一标识>/`：配置、每轮 CSV、训练日志、TensorBoard、每 episode 逐时隙策略版本/滞后和场景身份。`iter_N/` 保存 actor、双 Q、目标 Q、温度、配对的状态归一化、训练 CSV 快照和轨迹（NPY、PNG）。SAC 轨迹可能在每个时隙使用不同策略版本，不能标成单一策略的轨迹；详见对应 episode 元数据。`completion_rate` 为当前环境成功时隙比例；有限目标值均值、样本数和求解状态分布分别记录。

检查点仅用于评估，不保存优化器、随机数、奖励缩放器或 Replay Buffer，不支持精确续训。可用 `algorithms.MHSAC.checkpoint.load_normalization(iter_dir, expected_dim)` 校验 actor 哈希并加载状态归一化，再用 `normalize_state` 处理状态。当前仅保存逐轮摘要，不生成 MHBPPO 的逐时隙动作/环境诊断 JSONL 和日志轮转文件。

```powershell
python -m unittest discover -s src/outer/algorithms/MHSAC -p test_mhsac_contract.py -v
python src/outer/algorithms/MHSAC/verify_run.py <运行目录>
```
