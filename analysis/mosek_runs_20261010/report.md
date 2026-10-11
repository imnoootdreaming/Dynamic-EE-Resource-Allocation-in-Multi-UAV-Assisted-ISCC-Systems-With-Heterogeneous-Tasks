# RelGap 默认值修改与实际入口运行（2026-10-10）

## 结论

默认 mosek_tol_rel_gap=1e-7；PC3P Fusion/备用CVXPY、GC3P松弛与CVXPY恢复统一参数。原始/对偶可行性容差仍为1e-8，其余求解设置未变。基准通过 --mosek-rel-gap 1e-8 覆盖；加速通过 --mosek-rel-gap 1e-7。每次获取缓存模型都会同步容差，避免两种设置共享缓存时残留。

实际执行main和compare入口，均为30样本；compare候选总数50，seed=1208。两个精度各运行一次，不能用单次计时替代S01三轮交错测试，也不能将当前结果直接外推到6轮/2秒。

能耗采用 compute_pure_energy（P4加权能耗，不含秩一罚项）；main的objective_p4仍含罚项，独立保存以区分口径。去除固定飞行能耗后另报告可变能耗偏差。

| 入口/算法 | 基准耗时中位数/s | 加速耗时中位数/s | 配对耗时比 | 最大能耗相对差/% | 最大可变能耗差/% | 外层收敛 基准→加速 | 恢复波束约束通过 基准→加速 |
|---|---:|---:|---:|---:|---:|---|---|
| main_pc3p | 0.3160 | 0.2824 | 0.8900 | 0.000386124 | 0.0490287 | 30→30 | 11→29 |
| compare_pc3p | 0.3184 | 0.2813 | 0.9157 | 0.000386124 | 0.0490287 | 30→30 | 11→29 |
| compare_gc3p | 1.0887 | 1.0449 | 0.9866 | 4.53575e-05 | 0.0045571 | 30→30 | 30→30 |

## 可行性与比较边界

GC3P基准有29/30个回退case，累计通过候选50/1500；加速有29/30个回退case，通过50/1500。回退的有限能量和秩一不能替代原始约束验证。若全部或大量候选失败，此CSV不能独立支持高斯随机化算法的节能优势。

详细CSV记录协方差原约束、恢复波束原约束、严格秩一及完整物理解验收，solver收敛字段保留原语义。既有max_violation是不同物理量的最大值，新max_normalized_violation用于无量纲约束检查。未修改容差来掩盖c08或近零B相对秩一问题。

## 产物与复现

- src/inner下：first_sample_cccp_history.csv、convergence_iterations.csv、convergence_iterations_details.csv、compare_pc3p_gc3p_results.csv、compare_pc3p_gc3p_details.csv。
- baseline/和accelerated/下：各入口CSV与完整stdout/stderr日志。
- mosek_gap_effects.csv：90行逐样本基准/加速比较。
- run_manifest.json：命令、Python、配置与源码/input哈希；before/：修改前源码和原CSV。
- summary.json：统计与输出格式/输入完整性校验。

```powershell
cd src/inner
D:/anaconda3/envs/cvx_new/python.exe main.py
D:/anaconda3/envs/cvx_new/python.exe compare_pc3p_gc3p.py --cases 30 --candidates 50
```

默认命令使用1e-7；加 --mosek-rel-gap 1e-8可复跑基准。原绘图CSV列保持兼容；两个入口明细现在默认写出。
