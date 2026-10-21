# 方案2/3具体修改设计

## 结论

- 方案2在当前配置下应提高效率。`cccp_bridge.solve_inner_energy()` 已在 `run_pc3p()` 前计算感知诊断；修改只是复用现成结果作严格分支，并非再算一次。被证书证明不可行的槽省去整个PC3P/MOSEK调用。
- 若未来同时关闭环境诊断与感知短缺奖励，才需要为预检新增一次极小的NumPy计算。净收益条件为 `C_pre < p_cert * C_solver`。已有整套诊断约3.45 ms/槽（还包含比最小证书更多的检查），内层平均约288 ms/槽；粗略break-even约1.2%。最小证书实际应低于3.45 ms。当前后期证书失败率约1.51%，仍接近或略高于保守break-even；当前配置因检查本来就执行，break-even问题不存在。
- 方案3应由环境维护episode可靠性状态，并把终局零失败奖励真正写入PPO transition reward。推荐加入`failed_so_far`观测，使critic知道终局奖励是否仍可获得。该方案把状态维度1017改为1018，旧checkpoint不能直接strict-load；若必须复用iter_12000，需要显式迁移actor/critic第一层和normalizer。
- 第一轮只实施终局成功bonus，不同时叠加首次失败罚，避免无法判断收益来源。默认系数0保持旧行为；建议主实验系数1.0，消融0/0.5/1/2。

## 方案2：严格不可行预检

### 当前控制流

`cccp_bridge.py:289-320`已经构造`InnerContext`并计算`sensing_diagnostics`。当前命令同时满足：

- `env_diagnostics=true`，所以每槽执行`sensing_diagnostics(..., candidates=...)`；
- `sensing_penalty_coef=1`，即使关闭详细诊断，也会为奖励执行`sensing_diagnostics(..., candidates=False)`。

但当前在`cccp_bridge.py:321-323`仍无条件执行`run_pc3p(ctx)`。因此预检应插在已有诊断之后、solver之前。

### 推荐修改

1. `configs/madrl_args.py`
   - 新增`--analytic_infeasibility_presolve/--no-analytic_infeasibility_presolve`，默认先设`False`完成A/B验收，通过后再考虑改默认。
   - metadata/hyperparameters记录开关和证书contract。
2. `reward/cccp_bridge.py`
   - 将当前动作的基础感知诊断统一计算一次并缓存；候选matching仅按原间隔扩展，不参与solver跳过。
   - 只在`status == checked`且`current_action_infeasible is True`时跳过。
   - 构造与无解分支兼容的空result，明确记录：`solver_status_kind=infeasible`、`status_source=analytic_presolve_c03_c05_c06`、`raw_solver_status=not_run`、`solver_iterations=0`、`solve_seconds=0`、`presolve_skipped=true`及证书裕量。
   - 边界`margin >= 1-1e-6`、非有限输入、诊断异常全部继续调用solver。
3. `reward/my_reward.py`
   - 不改变数学奖励。预检跳过返回`energy_opt=inf`，沿用基础无解罚1和现有感知短缺Q。
   - 增加`presolve_skipped`日志字段；确保precheck开/关对同一严格不可行动作的`base_reward`、Q、actual reward相同。
4. `algorithms/MHBPPO/MHBPPO_sampler.py`、`rollout_batch.py`、`MHBPPO_main.py`、`action_logging.py`
   - 汇总`presolve_skipped_count/rate`、累计节省的solver调用数；保留raw solver未运行的来源，不能伪造MOSEK状态。

核心伪代码：

```python
sensing = cached_or_compute_sensing_diagnostics(..., candidates=False)
certified = (
    presolve_enabled
    and sensing.get("status") == "checked"
    and sensing.get("current_action_infeasible") is True
)
if certified:
    result = empty_infeasible_result(source="analytic_presolve_c03_c05_c06")
    solver_seconds = 0.0
else:
    result = run_pc3p(ctx)
```

不要用以下条件跳过solver：候选rematching不存在、裕量仅略低/等于边界、上一槽失败、网络置信度低。这些都不是当前动作完整问题的严格不可行证书。

### 效率判断

设每槽预检成本`C_pre`、被严格证书拒绝比例`p_cert`、这些槽原solver平均成本`C_solver_fail`：

`每槽净节省 = p_cert * C_solver_fail - C_pre`。

- 当前run：`C_pre`已包含在现有诊断/奖励路径，因此新增成本近似只有一次布尔分支；净节省约为`p_cert*C_solver_fail`，必为正。
- 末期：603/40000=1.5075%严格感知失败，预期墙钟只改善约1–3%，不是数量级加速。
- 初期：前500 update非optimal槽60.58%，但非optimal不全等于可被该证书拒绝；若回放确认多数为相同感知证书，节省会远大于后期。
- 若诊断和感知奖励均关闭：已有整套诊断约0.138 s/40槽=3.45 ms/槽，内层约11.529 s/40槽=288 ms/槽；用偏保守的整套诊断成本估算，break-even约1.2%。最小预检不做候选matching、outer checks和返回解验证，实际应更便宜，但必须基准实测。

### 必须通过的测试

- 证书边界：`M < 1-tol`跳过，`M == 1-tol`及以上不跳过。
- solver mock调用计数：严格失败为0，其余为1。
- 同一冻结输入precheck开/关的resolved status、reward、Q和next state一致；区别只能是raw solver provenance、solver耗时和迭代数。
- 诊断开/关、sensing coef 0/1四种组合都只计算一次基础证书。
- 非有限/异常时fail open，继续solver，不能错误跳过。

## 方案3：40槽零失败目标

### 为什么放在环境层

`MyReward.reward_compute()`只知道当前槽，不知道此前39槽是否失败。episode历史属于`MyEnv`，所以可靠性奖励应在`MyEnv.step()`中、当前槽success计算后加入，而不应塞进单槽物理reward函数。

### 推荐的Markov实现

1. `configs/madrl_args.py`
   - 新增`--episode_success_bonus_coef`，有限且非负，默认0。
   - 建议第一轮主值1.0，消融`0/0.5/1/2`。
2. `environment/my_env.py`
   - `reset()`设置`self.failed_so_far=False`、`self.failure_count=0`。
   - 当前槽按既有三项判据计算`slot_success`后，更新failure状态。
   - 在第40槽且`failure_count==0`时：
     `reliability_bonus=episode_success_bonus_coef`；否则为0。
   - 同时修改`total_reward`、`reward["bs"]`和`reward["components"]`，保证sampler记录与PPO实际输入一致。
   - 字段包括`slot_success`、`failed_so_far_before/after`、`failure_count`、`all_success_terminal`、`episode_success_bonus_applied`。
3. `environment/observation.py`
   - 在`ObsContext`和`BS_OBSERVATION_SEGMENTS`末尾加入`failed_so_far`标量0/1，使状态维度1017→1018。
   - 当前状态表示执行动作前是否已有失败；next state反映当前槽后的新状态。
4. `algorithms/MHBPPO/MHBPPO_sampler.py`
   - 无需事后改transition；直接接收环境返回的已塑形reward。
   - 汇总零失败bonus次数、失败首次出现slot和episode failure count。
5. `MHBPPO_main.py`、`action_logging.py`、checkpoint metadata
   - 保存reward contract、bonus coef和新观测布局；TensorBoard记录`all_success_rate`、bonus均值、首次失败slot。

伪代码：

```python
failed_before = self.failed_so_far
slot_success = existing_success_test(reward_components)
if not slot_success:
    self.failed_so_far = True
    self.failure_count += 1

done = self.t + 1 >= total_time_slots
all_success = done and not self.failed_so_far
bonus = episode_success_bonus_coef if all_success else 0.0
total_reward += bonus
reward["bs"] = total_reward
reward["components"]["episode_success_bonus_applied"] = bonus
```

### checkpoint兼容性

- 最干净的消融是新网络从头训练，因为观测维度改变。
- 若从iter_12000 warm start：actor和critic第一层从`[hidden,1017]`扩成`[hidden,1018]`，复制原1017列，新列置0；normalizer复制原统计，新维度设mean=0、初始scale=1。必须生成新的显式初始化bundle并记录迁移，不可直接strict-load旧checkpoint。
- 若暂时不改观测，可在sampler回合结束后只给最后transition加bonus，保持checkpoint维度；但critic看不到此前是否失败，价值估计方差更大。它适合作为快速原型，不建议作为最终实现。

### bonus与首次失败罚的选择

第一轮只用终局bonus。若同时加入首次失败罚，两个项都在奖励“零失败”，实验无法区分哪一个有效。PPO使用`gamma=0.99`，终局bonus传回第1槽的折扣约`0.99^39=0.676`，信号仍可传播。

若后续确认终局信号方差过大，再单独测试“第一次失败只罚一次”。此时也必须保留`failed_so_far`状态，并避免与终局bonus同时默认开启。

### 必须通过的测试

- coef=0时40步reward、状态和日志除新增零字段外与旧实现一致。
- 40/40仅最后槽加一次bonus；39/40、重复失败均不加。
- `failed_so_far`在首次失败后保持1并在reset清零。
- `total_reward == reward['bs'] == transition reward`，诊断actual_reward同步。
- done/GAE边界不变，bonus只进入对应episode最后transition。
- 新观测布局、normalizer、actor/critic维度、初始化迁移和checkpoint metadata一致。

## 实施顺序

1. 先单独实施方案2并做冻结输入等价性测试及100–200 update测速；它不应改变策略学习数据的reward和next state。
2. 再单独实施方案3，从头或显式迁移初始化运行；不要与方案2同一轮首次启用，否则无法区分样本效率与墙钟收益。
3. 报告环境槽数、solver调用数、槽/s、达到连续三个500-update窗口50%的槽数/墙钟，以及固定checkpoint独立1000回合评价。
