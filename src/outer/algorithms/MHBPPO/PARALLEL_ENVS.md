# MHBPPO 同步并行环境

默认主seed为42、场景seed_offset为0、同步训练、1个env。默认三个randomize_*_per_episode均关闭。

在项目根目录使用现有Python训练环境执行：

```powershell
python src/outer/algorithms/MHBPPO/MHBPPO_main.py --num_samplers 1
python src/outer/algorithms/MHBPPO/MHBPPO_main.py --num_samplers 2
```

两条命令分别启动独立的从头训练；不必同时运行。显式配置同一主seed与场景偏移也可以：

```powershell
python src/outer/algorithms/MHBPPO/MHBPPO_main.py --seed 42 --scene_seed_offset 0 --sync_training --num_samplers 2
```

## 采集与更新含义

- 每个env一个独立进程和独立CCCP/Fusion模型。所有env共享同slot的外生序列：初态、CU轨迹、三类NLoS、目标调度和物理任务参数。每个episode从slot0重放，每个env按本地slot推进；无需逐slot按墙钟锁步。
- worker身份与场景偏移分开。动作抽样使用从主42派生的独立可复现子流，不添加动作噪声，不改变原分布。不同动作会造成不同UAV轨迹、几何信道、状态与奖励。
- 第r次更新收集时，各worker全episode固定使用同一个actor版本r−1及同一状态归一化快照。先检查网络/归一化/场景SHA、完整slot顺序和done，再收齐N个episode。
- 默认T=40，1/2/4env每次更新分别使用40/80/160条新经验。经验按完整episode排列，done隔离GAE，优势全batch归一化一次；不是每收到一个env就更新一次。
- minibatch32、epochs5和原PPO/动作/奖励/新增感知罚不变。默认每个网络每次更新10/15/25次Adam，每个新样本仍使用5遍。

## 产物与checkpoint

- 原CSV四列和文件名保留，每行代表一次网络更新；旧`episode`列继续是从0开始的更新索引。2env第1000次更新已采2000个环境episode，CSV仍1000行。
- `iter_N`仍按更新次数保存：一套共享actor/critic，以及更新N后准备广播的对应状态归一化统计。不是分别保存各env的网络，也不是保存采集本批时的旧actor。
- 新`checkpoint_metadata.json`明确更新后的weights_version=N、采集版本N−1、累计env_steps、env数和本次样本数；归一化文件继续与actor字节SHA配对。
- 原`uav_trajectory.png`是env0；其它env分别保存`uav_trajectory_env_1.png`等。轨迹属于采集策略N−1，标题明确这一点；不能把两条40步轨迹拼成一条80步轨迹。
- 原文本/TensorBoard和奖励CSV统计整批样本，另加每env文本摘要、`training_updates.csv`计数/计时，以及每步JSONL的worker/round/场景/动作流和版本字段。
- 收集未完成时停止，不做部分更新；更新开始后的正常停止信号在完成该更新与归一化后收尾。PPO/归一化途中异常不把部分新网络标为上次完整版本。停止时处理未消费队列的退出等待。

## 耗时与资源

两个独立CCCP进程可以并行。采集耗时接近较慢env的耗时，加上推理、传输和日志；PPO更新使用更多minibatch，耗时也可能增加。不会预先保证2env单轮与1env完全等时。

新增可选资源设置：

```powershell
python src/outer/algorithms/MHBPPO/MHBPPO_main.py --num_samplers 2 --worker_num_threads 1 --mosek_num_threads 1
```

`worker_num_threads`仅限制PyTorch CPU推理线程；`mosek_num_threads`限制每个内层模型的MOSEK线程。默认0保留自动设置。根据真实采样时间选择，不假定自动线程越多越快。启动CPU/模型导入时间与稳态采集时间分开统计。

## 保留的配置语义

默认学习率horizon仍为10000次网络更新，checkpoint仍每1000次更新；没有自动把2env改成5000次或改成固定总环境步数停止。`episodes`仍仅控制LR horizon，训练持续至停止请求。

若以后作相同40万步对照，可明确用N1/N2的LR horizon10000/5000及保存间隔1000/500，但本次没有实施自动停止预算或改变默认时间表。

同步N>1不允许三个逐episode随机化开关开启，启动前报配置错误。N1保留显式随机化选项：该例外不保证跨episode同场景，仍校验当前场景信息、网络/归一化及完整episode。原异步模式保留旧逐episode更新语义，不是此处的同步NT聚合入口。

本次没有实施详细日志裁剪/容量管理、整组checkpoint磁盘事务或优化器/RNG精确续训；`checkpoint_metadata.json`不是任意磁盘写入失败后完整保存的保证。历史日志、模型和正在运行的训练不迁移、不自动重启。
