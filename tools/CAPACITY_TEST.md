# Linux 多算法训练容量测试

结论：用每个算法的单任务速度作基线，测实际组合对各算法的减速。
CPU 核数只能给初筛范围；最终容量是满足减速、内存、显存、I/O 和稳定性要求的**具体组合**。
同为 8env 的不同算法不能视为相同负载。

本工具调用当前仓库的真实训练入口，采样、求解、网络更新、诊断和常规日志均保留。
请把 `tools/` 和匹配版本的项目代码一起传到 Linux，在已安装训练依赖和有效 MOSEK 许可证的 Python 环境执行。

## 1. 先看可用资源

在仓库根目录执行；`psutil` 是容量工具额外需要的监控依赖。

```bash
python -m pip install psutil
python tools/linux_capacity_probe.py --inventory-only
lscpu
nproc
free -h
df -h .
nvidia-smi
```

`inventory.json` 保存 CPU 亲和性、宿主机物理核、cgroup v2 的 CPU 配额和内存限制，以及 GPU 概况。
`effective_cpu_upper_bound` 取亲和性逻辑 CPU 数和可见祖先 CPU 配额的最小值。
物理核数显示的是宿主机信息；SMT 的两个逻辑 CPU 不应直接按两个物理核估算计算性能。
若 `cgroup_v1_or_unresolved=true`，须另外确认容器/作业调度器分配，不能把宿主机资源视为自己的额度。
内核文档说明 CPU 和内存限制可沿 cgroup 层级施加：[Tejun Heo，2015 起维护，Control Group v2，Linux Kernel Documentation](https://docs.kernel.org/admin-guide/cgroup-v2.html)。

粗略初筛可按 `8 × 同时任务数 + learner/系统余量` 估算 CPU 需求，但同步训练的采样与更新阶段会交替，
线程、NUMA、内存带宽和求解负载也会改变结果，这不是一个可保证容量的公式。
例如可用 32 个物理核时，先测试 2、3 个任务，再测试 4 个；不要预先承诺 4 个不会减速。

## 2. 第一轮：真实训练短测

先关闭同一资源配额下不相关的计算作业，确保 solo 基线有意义。以下默认选定 GPU 0；CPU 服务器去掉这一行或加 `--cpu`。

```bash
export CUDA_VISIBLE_DEVICES=0

# 只查看计划，不启动训练
python tools/linux_capacity_probe.py \
  --algorithms MHBPPO MHGPPO SHBPPO Pure_MHBPPO MHSAC \
  --settings tools/capacity_settings.example.json \
  --dry-run

# 实际测试：5 个单任务基线，然后测试前 2/3/4/5 个算法的组合
python tools/linux_capacity_probe.py \
  --algorithms MHBPPO MHGPPO SHBPPO Pure_MHBPPO MHSAC \
  --settings tools/capacity_settings.example.json \
  --warmup 5 --measure 20 --timeout 3600
```

默认选中全部五个算法，顺序为 `MHBPPO MHGPPO SHBPPO Pure_MHBPPO MHSAC`。
默认组合是这个列表的前缀，**不会枚举所有组合**。
上面的列表实际测试的是 `MHBPPO+MHGPPO`、再加 SHBPPO、再加 Pure、再加 MHSAC。
如只关心三个指定任务，应直接指定实际组合：

```bash
python tools/linux_capacity_probe.py \
  --algorithms MHBPPO Pure_MHBPPO MHSAC \
  --groups MHBPPO,Pure_MHBPPO MHBPPO,MHSAC MHBPPO,Pure_MHBPPO,MHSAC \
  --settings tools/capacity_settings.example.json \
  --measure 30 --timeout 7200
```

CPU 模式：

```bash
python tools/linux_capacity_probe.py \
  --algorithms MHBPPO MHGPPO Pure_MHBPPO --cpu --measure 30
```

每个任务默认 `--num_samplers 8 --sync_training`，worker、MOSEK、OMP、MKL、OpenBLAS 等线程限制为 1。
这是所测资源配置的一部分；正式训练必须沿用相同限制，才能使用容量结论。
PyTorch 官方将进程各自启动过多 CPU 线程列为 CPU oversubscription 问题：
[PyTorch Contributors，Multiprocessing best practices，官方文档](https://docs.pytorch.org/docs/2.9/notes/multiprocessing.html)。
MOSEK 的 `num_threads=0` 允许优化器自动选择线程数，正值指定线程数：
[MOSEK ApS，Optimizer API for Python: Parameters / num_threads，官方文档](https://docs.mosek.com/latest/pythonapi/parameters.html#iparam.num_threads)。

例子中的 settings 将 PPO 学习率衰减区间设为 3000 轮，与计划的 `max_updates=3000` 正式训练匹配。
若正式训练用 8000 或 10000 轮，修改各算法 `--episodes`，并补齐正式训练的其他参数。
只为测速把 `--max_updates` 设成 20 会同时压缩 PPO 学习率计划、改变策略和求解负载，因此工具使用 `max_updates=0`，由外部驱动结束测量。
本工具默认 MHBPPO 的期望重复匹配系数为 0.3，与当前后台脚本一致，settings 可以覆盖。

任务会从头训练并生成正常的 checkpoint、TensorBoard 和日志。容量测试本身可能需要几十分钟至数小时。
可以在 `tmux` 中运行，避免 SSH 断开；按 Ctrl+C 会请求停止本次工具启动的 learner 并清理其进程组。
若直接强杀容量驱动，操作系统无法保证它执行清理，应优先用 Ctrl+C。

## 3. 测量方法与结果

1. 每个算法单独启动，排除启动和指定预热轮次，记录连续 `--measure` 轮的端到端吞吐。
2. 并发组内各任务独立预热，先完成者暂用 SIGSTOP 暂停自身整个进程组；全部预热后用 SIGCONT 一起释放计时。
3. 对同一个算法，solo 与并发测量使用相同轮次区间；MHSAC 还必须先观察到真实 `sac_updates>0`，再排除至少 5 个学习轮次。
   默认最小 replay 8000 条、8×40 样本/轮时，第 25 轮最后一个样本触发首次更新，
   第 26 轮起进入整轮学习负载；默认计满 5 个含更新的轮次，在第 29 轮后进入测量。
4. 较快任务测完后继续训练，直到组内所有任务测完，保留其竞争负载。
5. 计时采用 CSV 完整行到达的外部单调时钟差，包含测量区间内 checkpoint/日志造成的间隔。
   轮询周期默认 0.25 秒，窗口默认至少 30 秒；太短的窗口会拒绝出具有效比较，须增加 `--measure`。
   暂停与释放存在轮询粒度误差；短窗口或亚秒级轮次尤其应延长测试。
6. 每个 stage 超时默认 3600 秒；内存余量低于 10% 或结果所在文件系统空闲不足 5 GiB 时中止整个扫描。
   监测不是硬隔离，短于采样间隔的内存尖峰仍可能发生。

结果默认放在 `src/outer/logs/capacity/<时间_随机标识>/`：

| 文件 | 用途 |
| --- | --- |
| `inventory.json` | 硬件、CPU 亲和性、可见 cgroup 额度 |
| `probe_config.json` | 测试参数和各算法 settings |
| `comparison.csv` | 每算法 solo/并发吞吐、减速和 p95 计时比较 |
| `summary.json` | 各已测组合是否通过吞吐阈值 |
| `<stage>/resources.jsonl` | CPU、内存、swap、PSI、cgroup 计数、磁盘、任务进程数及线程数 |
| `<stage>/gpu.csv` | 若有 nvidia-smi，记录所有 GPU 的利用率和显存；失败信息也保存在此处 |
| `<stage>/stage_result.json` | 完整或部分结果、异常、退出码和实际训练产物目录 |
| `<stage>/<算法>/console.log` | 启动、进度与异常日志 |
| `<stage>/<算法>/command.json` | 实际命令与资源环境变量 |

训练 checkpoint 和 TensorBoard 继续使用训练入口原有目录，`--output` 仅改变容量报告目录。
工具不修改后台启动脚本的 `.learner.pid` 或 `.latest_log` 文件。
内存记录中的 RSS 总和会重复计算共享页，不能等同于物理内存净占用；同时看可用内存和 cgroup memory.current。

以同一个算法的单任务吞吐 `v_solo` 和并发吞吐 `v_group` 定义：

```text
吞吐保留率 = v_group / v_solo
每步耗时增幅 = v_solo / v_group - 1
```

默认 `--max-slowdown 0.20`：所有算法的每步耗时增幅都不超过 20%，才通过**吞吐**判据。
例如 100 steps/s 降到 80 steps/s，吞吐下降 20%，每步耗时实际增加 25%，不能按 20% 减速通过。
`normalized_throughput_sum` 是各算法吞吐保留率之和，可辅助评估相对各自 solo 的总处理能力，
不能把不同算法的 env steps 当成等价的学习进展。

`csv_iteration_seconds_p95` 是原训练字段的诊断分位数。MHBPPO 使用采样等待+更新+日志三段之和，
其余算法使用 `elapsed_seconds`；这些字段未包括全部轮末开销。它们用于同算法诊断，端到端容量仍按外部吞吐。
20 轮的 p95 波动较大，正式验收宜增加轮数。
`passes_throughput_limit` 只判断吞吐，最终选择还须检查资源余量、异常和长期负载。

## 4. 同时开终端监测瓶颈

Ubuntu/Debian 的 `mpstat/pidstat/iostat` 来自 `sysstat`；若未安装，可用 `sudo apt-get install sysstat`。
在不同终端运行：

```bash
mpstat -P ALL 2
vmstat 2
pidstat -u -r -d -w -p ALL 2
iostat -xz 2
watch -n 2 'cat /proc/pressure/cpu /proc/pressure/memory /proc/pressure/io'
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total --format=csv -l 2
```

| 现象 | 排查方向 |
| --- | --- |
| 吞吐下降、CPU 排队和 CPU PSI 同时上升 | CPU 争用；检查核额度、线程数、亲和性和 NUMA |
| 宿主 CPU 仍空闲，cgroup cpu.stat 的 throttling 计数持续增加 | 作业/容器 CPU 配额限制 |
| 可用内存下降、持续 swap in/out、memory PSI 上升 | 内存压力；降低并发，检查 replay 增长 |
| I/O PSI、读写延迟上升，并与 checkpoint/日志停顿对应 | 存储、网络盘、日志或 checkpoint 竞争 |
| GPU 利用率不高但 CPU 很忙 | 先检查 CPU 采样与内层求解，增加 GPU 未必提速 |
| 显存逼近上限或 CUDA OOM | 减少同 GPU learner 数，或把任务分配至不同 GPU |

PSI 的 `some` 表示至少有任务因资源等待而停顿，`full` 表示所有非空闲任务同时停顿；
系统级 CPU `full` 不作为这里的判据。参见
[Johannes Weiner，PSI — Pressure Stall Information，Linux Kernel Documentation](https://docs.kernel.org/accounting/psi.html)。
命令字段参见 [sysstat 项目，pidstat 官方手册](https://man7.org/linux/man-pages/man1/pidstat.1.html)、
[sysstat 项目，mpstat 官方手册](https://man7.org/linux/man-pages/man1/mpstat.1.html) 和
[NVIDIA，NVIDIA System Management Interface，官方文档](https://docs.nvidia.com/deploy/nvidia-smi/index.html)。
单独的负载均值、CPU/GPU 利用率或上下文切换次数均不能决定容量；要与端到端减速对应。

## 5. 选容量与正式运行

- 先对感兴趣的组合各测一次，找出临界附近组合；对候选组合在相近的空闲条件下复测至少三次，比较波动。
- 初筛可用每步耗时增加 ≤20%；要求更严格时用 `--max-slowdown 0.10`。
  资源验收可先留至少 15%–20% 内存/显存余量，并要求无持续换页、无 OOM 和任务异常。
  这些都是本项目建议的工程阈值，不是行业标准。
- 最终候选组合再运行 30–60 分钟以上，并跨过至少一次正式 checkpoint。默认 1000 轮才保存周期 checkpoint，
  初始短测通常覆盖不到它；若每轮很慢，应按覆盖事件而不是分钟数延长。
- MHSAC 默认 replay 容量 100000；8×40 时第 313 轮才填满，初筛 50 轮左右尚不能代表长期内存峰值。
  用 `--warmup 315 --measure 100 --timeout <足够秒数>` 可作填满 replay 后的复测，但会对所有选中算法预热至少 315 轮，成本较高。
- 内层求解调用次数与策略、成功率和动作有关。起始阶段重复匹配多、直接受罚的槽可能较多，
  不能保证与后期可行策略的求解负载相同；最终容量需要包含后期实际训练监测。
- 工具只支持每组各算法一个实例。多个随机种子副本的容量需另测，不能把一个算法实例的结果直接翻倍外推。
- 多 GPU 可在 settings 为某算法加 `"env": {"CUDA_VISIBLE_DEVICES": "1"}`。solo 与组合必须保持同一映射。
  `--cpu` 优先隐藏所有任务的 CUDA。

确认容量后，正式训练也使用相同线程限制和设备分配，例如分别在终端启动：

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0
bash src/outer/algorithms/MHBPPO/run_background.sh start --max_updates 3000
bash src/outer/algorithms/MHGPPO/run_background.sh start --max_updates 3000
```

当前后台脚本已默认 8env、worker/MOSEK 各 1 线程，外部 export 补上数值库与 learner 的线程预算。
正式参数须与容量测试 settings 对齐。

## 本地验证范围

Windows 本地验证包括语法、命令 dry-run、CSV 增量读取、solo/并发减速公式、MHSAC 学习预热和测量窗口。
Linux 真实进程组暂停/释放/停止、MOSEK 多进程许可证与服务器实际容量仍需在目标服务器运行核验。

```bash
python -m unittest discover -s tools -p test_linux_capacity_probe.py -v
```
