# MHBPPO 单 worker 同步训练

## 修改结论

现在可以用一个 rollout worker 做严格同步训练。默认配置已改为：

- `num_samplers=1`
- `sample_queue_maxsize=1`
- `sync_training=True`

同步握手保证每轮顺序为：worker 采集一个完整 episode → learner 执行一次 `PPO.update()` → learner 广播 actor 权重和归一化统计量 → worker 才开始下一个 episode。环境类、`reset(episode_idx)`、信道/布局生成逻辑没有修改。

## 固定 seed 的可重复性

worker 仍使用原来的 `seed_offset=worker_id+1`，因此单 worker 使用 `(base_args.seed, 1, episode_idx)` 派生环境场景。相同 seed 从头启动训练时，环境场景序列和采样顺序应保持一致；同步握手也消除了多个 episode 在 learner 更新前积压造成的参数陈旧。

固定 seed 能验证训练流程是否稳定、是否能达到稳定的 reward/完成率平台，但不能单独证明 PPO 在所有随机种子下都收敛。GPU 算子、求解器内部并行和浮点顺序仍可能产生极小差异。建议比较两次运行的 `training_rewards_seed_<seed>.csv` 曲线和最终 checkpoint，而不是要求每个浮点数逐位相等。

## 运行方式

在 `src/outer` 环境下直接运行：

```powershell
python -m algorithms.MHBPPO.MHBPPO_main --seed 1208
```

若需要保留旧的异步 actor–learner 行为，可显式关闭同步，并指定多个 worker：

```powershell
python -m algorithms.MHBPPO.MHBPPO_main --no-sync_training --num_samplers 4 --sample_queue_maxsize 4
```

建议先用较小的 `--checkpoint_interval` 做短跑，确认日志中的 `stale=0`、`ck=ok`，再增加训练轮数。重复相同 seed 时应从新的输出目录启动，避免覆盖上一轮 CSV/checkpoint。

## 后续操作

1. 用同一个 seed 连续运行两次，检查 reward、`completion_rate`、`obj` 是否趋势一致。
2. 确认日志没有 `MISMATCH`、采样进程异常退出或 solver error 激增。
3. 若单 seed 能稳定达到目标，再用多个 seed（例如 1208、1209、1210）评估泛化稳定性。
4. 需要恢复吞吐量时，再使用 `--no-sync_training` 和多个 worker；这时应同时关注 `stale` 和样本队列积压。
