from pathlib import Path
import json,hashlib
R=Path.cwd();O=R/'analysis/mosek_runs_20261010'
p=R/'src/inner/compare_pc3p_gc3p.py';s=p.read_text(encoding='utf8');s=s.replace('--candidates 50 表示「共 50 个候选可行解」；若要「50 个高斯方向 + 主特征方向」','--candidates 50 表示共 50 个候选方向，不保证全部可行；若要「50 个高斯方向 + 主特征方向」');p.write_text(s,encoding='utf8')
p=R/'src/inner/pc3p.py';s=p.read_text(encoding='utf8');s=s.replace('# 内点法可行性容差：默认 1e-8，放宽到 mosek_tol_feas 可显著减少内点迭代（仅影响收敛判据）','# 可行性容差与最优性间隙分别配置；默认仅放宽 RelGap，不放宽可行性。');p.write_text(s,encoding='utf8')
p=R/'src/inner/main.py';s=p.read_text(encoding='utf8');s=s.replace('# 各 case（全部 30 组）的迭代次数统计；details 文件无需保存，此处直接注释掉','# 各 case 的迭代统计及能耗/精度/约束验收明细');s=s.replace('    :param rows: 每项为 [case_index, sample_id, iterations, converged,\n                        status_kind, elapsed_s]，与 save_iteration_counts 入参一致。','    :param rows: 前六项为 case_index、sample_id、iterations、converged、\n                 status_kind、elapsed_s；其后为精度、目标、纯能耗、秩一和 CHECK_FIELDS。');p.write_text(s,encoding='utf8')
files=['parameters.py','fusion_pc3p.py','pc3p.py','gc3p.py','main.py','compare_pc3p_gc3p.py','solution_checks.py']
for n in files:compile((R/'src/inner'/n).read_text(encoding='utf8'),n,'exec')
m=json.loads((O/'run_manifest.json').read_text());m['final_sha256']={'src/inner/'+n:hashlib.sha256((R/'src/inner'/n).read_bytes()).hexdigest() for n in files};m['post_run_changes']='Documentation comments only in main/compare/pc3p; execution behavior unchanged';m['cache_precision_switch_check']='passed: same cached model 1e-8 -> 1e-7 -> 1e-8; Pfeas/Dfeas remain 1e-8';m['syntax_check']='passed';(O/'run_manifest.json').write_text(json.dumps(m,indent=2),encoding='utf8')
summary=json.loads((O/'summary.json').read_text());g=summary['groups']
plan=R/'src/outer/algorithms/MHBPPO/MHBPPO_TASK_PLAN.md';s=plan.read_text(encoding='utf8');s=s.replace('**用户新增优先任务 S02（2026-10-10，执行中）：','**用户新增优先任务 S02（2026-10-10，已完成）：');s=s.replace('| S02 | 执行中 |','| S02 | 已完成 |')
s+='''

S02 完成记录（2026-10-10）：新增 Parameters.mosek_tol_rel_gap=1e-7；统一 PC3P Fusion/备用CVXPY、GC3P Fusion松弛/CVXPY恢复；缓存每次获取同步容差。同一缓存按1e-8→1e-7→1e-8切换检查通过，Pfeas/DFEAS始终1e-8。main与compare增加--mosek-rel-gap覆盖，启用明细CSV；main新增纯能耗，两个入口新增独立原始约束及恢复波束验收字段，绘图CSV列不变。基准和加速各执行两个真实入口，全部退出码0；30行/ID/精度/有限能耗/发布文件一致性校验通过，输入样本SHA256不变，源码语法通过。原源码和CSV在analysis/mosek_runs_20261010/before，新默认CSV已发布到src/inner。

定量结果：main PC3P配对耗时降低约11.0%，compare中PC3P约8.4%，GC3P约1.3%（本轮单次入口运行，非新的三轮交错统计）。PC3P最大纯能耗相对差3.861e-6，可变能耗相对差0.0491%，均30/30外层收敛；GC3P最大纯能耗相对差4.536e-7，可变能耗差0.00456%，均30/30外层收敛。PC3P恢复波束原始约束11/30→29/30通过，严格完整物理解仍0/30；GC3P两配置严格完整物理解均30/30通过，但29/30 case使用投影回退，累计候选通过50/1500，不能把全部结果当高斯随机化选优成功。

证据：analysis/mosek_runs_20261010/report.md、summary.json、mosek_gap_effects.csv、run_manifest.json、check_cache.py；baseline/accelerated分别保存各入口CSV及完整日志。S01历史证据继续适用于before源码及原配置，新默认结果以上述S02验收为准；T02/T03实现及其历史条件证据未覆盖修改。本次没有重新训练或改变物理参数/诊断验收容差。当前下一步：T04——定位PC3P的c08/恢复波束及近零B相对秩一问题；GC3P候选被拒而回退可行的原因亦需诊断。
'''
plan.write_text(s,encoding='utf8')
print('S02 completed; syntax and CSV checks passed; published files:')
for name in ['first_sample_cccp_history.csv','convergence_iterations.csv','convergence_iterations_details.csv','compare_pc3p_gc3p_results.csv','compare_pc3p_gc3p_details.csv']:print(name,(R/'src/inner'/name).stat().st_size)