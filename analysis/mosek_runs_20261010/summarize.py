import csv,json,hashlib,sys
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[2];O=Path(__file__).resolve().parent
sys.path.insert(0,str(R/'src/inner'))
from parameters import Parameters
from environment import build_environment
from outer_sampler import load_outer_samples

def rows(stage,name):
 with (O/stage/name).open(encoding='utf8',newline='') as f:return list(csv.DictReader(f))
def truth(x):return str(x).lower() in ('1','true')
def main():
 p=Parameters();samples=load_outer_samples(R/'src/inner/feasible_outer_samples.csv',p)
 fly={i+1:float(p.omega_2*np.sum(build_environment(p,np.random.default_rng(p.seed),outer_variables=s).E_uav_fly)) for i,s in enumerate(samples)}
 output=[];groups={};validation={}
 jobs=[('main','pc3p','convergence_iterations_details.csv','case_index','pure_energy','elapsed_s','iterations','converged',''),('compare','pc3p','compare_pc3p_gc3p_details.csv','case','pc3p_energy','pc3p_time','pc3p_iters','pc3p_converged','pc3p_'),('compare','gc3p','compare_pc3p_gc3p_details.csv','case','gc3p_energy','gc3p_time','gc3p_iters','gc3p_converged','gc3p_')]
 for entry,algorithm,name,idfield,energyfield,timefield,iterfield,convfield,prefix in jobs:
  base=rows('baseline',name);acc=rows('accelerated',name)
  assert len(base)==len(acc)==30
  assert [int(x[idfield]) for x in base]==[int(x[idfield]) for x in acc]==list(range(1,31))
  assert all(float(x['mosek_rel_gap'])==1e-8 for x in base)
  assert all(float(x['mosek_rel_gap'])==1e-7 for x in acc)
  rr=[]
  for b,a in zip(base,acc):
   sid=int(a[idfield]);eb=float(b[energyfield]);ea=float(a[energyfield]);tb=float(b[timefield]);ta=float(a[timefield]);assert np.isfinite([eb,ea,tb,ta]).all()
   r=dict(entry=entry,algorithm=algorithm,case_id=sid,baseline_rel_gap=1e-8,accelerated_rel_gap=1e-7,baseline_energy_j=eb,accelerated_energy_j=ea,energy_difference_j=ea-eb,energy_relative_difference=abs(ea-eb)/max(abs(eb),1e-12),variable_energy_relative_difference=abs(ea-eb)/max(abs(eb-fly[sid]),1e-12),baseline_time_s=tb,accelerated_time_s=ta,time_ratio=ta/tb,baseline_iterations=int(b[iterfield]),accelerated_iterations=int(a[iterfield]),baseline_converged=int(truth(b[convfield])),accelerated_converged=int(truth(a[convfield])),baseline_physical_verified=int(truth(b[prefix+'physical_solution_verified'])),accelerated_physical_verified=int(truth(a[prefix+'physical_solution_verified'])),baseline_recovered_passed=int(truth(b[prefix+'recovered_beams_passed'])),accelerated_recovered_passed=int(truth(a[prefix+'recovered_beams_passed'])),baseline_max_normalized_violation=float(b[prefix+'max_normalized_violation']),accelerated_max_normalized_violation=float(a[prefix+'max_normalized_violation']))
   rr.append(r);output.append(r)
  key=entry+'_'+algorithm
  groups[key]=dict(cases=30,baseline_time_median=float(np.median([r['baseline_time_s'] for r in rr])),accelerated_time_median=float(np.median([r['accelerated_time_s'] for r in rr])),median_paired_time_ratio=float(np.median([r['time_ratio'] for r in rr])),baseline_time_total=sum(r['baseline_time_s'] for r in rr),accelerated_time_total=sum(r['accelerated_time_s'] for r in rr),max_energy_relative_difference=max(r['energy_relative_difference'] for r in rr),max_variable_energy_relative_difference=max(r['variable_energy_relative_difference'] for r in rr),max_absolute_energy_difference_j=max(abs(r['energy_difference_j']) for r in rr),baseline_converged=sum(r['baseline_converged'] for r in rr),accelerated_converged=sum(r['accelerated_converged'] for r in rr),baseline_recovered_passed=sum(r['baseline_recovered_passed'] for r in rr),accelerated_recovered_passed=sum(r['accelerated_recovered_passed'] for r in rr),baseline_physical_verified=sum(r['baseline_physical_verified'] for r in rr),accelerated_physical_verified=sum(r['accelerated_physical_verified'] for r in rr),baseline_mean_iterations=float(np.mean([r['baseline_iterations'] for r in rr])),accelerated_mean_iterations=float(np.mean([r['accelerated_iterations'] for r in rr])))
  if algorithm=='gc3p':
   groups[key]['baseline_candidate_counts']=sorted(set(int(x['gc3p_num_candidates']) for x in base));groups[key]['accelerated_candidate_counts']=sorted(set(int(x['gc3p_num_candidates']) for x in acc));groups[key]['baseline_feasible_candidates_total']=sum(int(x['gc3p_feasible_candidates']) for x in base);groups[key]['accelerated_feasible_candidates_total']=sum(int(x['gc3p_feasible_candidates']) for x in acc);groups[key]['baseline_fallback_cases']=sum('fallback' in x['gc3p_recovery_source'] for x in base);groups[key]['accelerated_fallback_cases']=sum('fallback' in x['gc3p_recovery_source'] for x in acc)
 for name,fields in [('compare_pc3p_gc3p_results.csv',['case_id','penalty_based_obj','penalty_based_time','gaussian_based_obj','gaussian_based_time']),('convergence_iterations.csv',['case_id','convergence_iterations'])]:
  for stage in ['baseline','accelerated']:
   rr=rows(stage,name);assert len(rr)==30 and list(rr[0])==fields;validation[stage+'/'+name]='30 rows, schema OK'
 for filename in ['first_sample_cccp_history.csv','convergence_iterations.csv','convergence_iterations_details.csv','compare_pc3p_gc3p_results.csv','compare_pc3p_gc3p_details.csv']:
  assert (R/'src/inner'/filename).read_bytes()==(O/'accelerated'/filename).read_bytes()
  validation['published/'+filename]='matches accelerated output'
 initial=json.loads((O/'before_manifest.json').read_text());assert hashlib.sha256((R/'src/inner/feasible_outer_samples.csv').read_bytes()).hexdigest()==initial['src/inner/feasible_outer_samples.csv']
 validation['input_samples']='SHA256 unchanged'
 with (O/'mosek_gap_effects.csv').open('w',newline='',encoding='utf8') as f:
  w=csv.DictWriter(f,fieldnames=list(output[0]));w.writeheader();w.writerows(output)
 (O/'summary.json').write_text(json.dumps(dict(groups=groups,validation=validation),indent=2),encoding='utf8')
 lines=['# RelGap 默认值修改与实际入口运行（2026-10-10）','','## 结论','','默认 mosek_tol_rel_gap=1e-7；PC3P Fusion/备用CVXPY、GC3P松弛与CVXPY恢复统一参数。原始/对偶可行性容差仍为1e-8，其余求解设置未变。基准通过 --mosek-rel-gap 1e-8 覆盖；加速通过 --mosek-rel-gap 1e-7。每次获取缓存模型都会同步容差，避免两种设置共享缓存时残留。','','实际执行main和compare入口，均为30样本；compare候选总数50，seed=1208。两个精度各运行一次，不能用单次计时替代S01三轮交错测试，也不能将当前结果直接外推到6轮/2秒。','','能耗采用 compute_pure_energy（P4加权能耗，不含秩一罚项）；main的objective_p4仍含罚项，独立保存以区分口径。去除固定飞行能耗后另报告可变能耗偏差。','','| 入口/算法 | 基准耗时中位数/s | 加速耗时中位数/s | 配对耗时比 | 最大能耗相对差/% | 最大可变能耗差/% | 外层收敛 基准→加速 | 恢复波束约束通过 基准→加速 |','|---|---:|---:|---:|---:|---:|---|---|']
 for name,g in groups.items():lines.append(f"| {name} | {g['baseline_time_median']:.4f} | {g['accelerated_time_median']:.4f} | {g['median_paired_time_ratio']:.4f} | {g['max_energy_relative_difference']*100:.6g} | {g['max_variable_energy_relative_difference']*100:.6g} | {g['baseline_converged']}→{g['accelerated_converged']} | {g['baseline_recovered_passed']}→{g['accelerated_recovered_passed']} |")
 gc=groups['compare_gc3p'];lines+=['','## 可行性与比较边界','',f"GC3P基准有{gc['baseline_fallback_cases']}/30个回退case，累计通过候选{gc['baseline_feasible_candidates_total']}/1500；加速有{gc['accelerated_fallback_cases']}/30个回退case，通过{gc['accelerated_feasible_candidates_total']}/1500。回退的有限能量和秩一不能替代原始约束验证。若全部或大量候选失败，此CSV不能独立支持高斯随机化算法的节能优势。",'', '详细CSV记录协方差原约束、恢复波束原约束、严格秩一及完整物理解验收，solver收敛字段保留原语义。既有max_violation是不同物理量的最大值，新max_normalized_violation用于无量纲约束检查。未修改容差来掩盖c08或近零B相对秩一问题。','','## 产物与复现','','- src/inner下：first_sample_cccp_history.csv、convergence_iterations.csv、convergence_iterations_details.csv、compare_pc3p_gc3p_results.csv、compare_pc3p_gc3p_details.csv。','- baseline/和accelerated/下：各入口CSV与完整stdout/stderr日志。','- mosek_gap_effects.csv：90行逐样本基准/加速比较。','- run_manifest.json：命令、Python、配置与源码/input哈希；before/：修改前源码和原CSV。','- summary.json：统计与输出格式/输入完整性校验。','','```powershell','cd src/inner','D:/anaconda3/envs/cvx_new/python.exe main.py','D:/anaconda3/envs/cvx_new/python.exe compare_pc3p_gc3p.py --cases 30 --candidates 50','```','','默认命令使用1e-7；加 --mosek-rel-gap 1e-8可复跑基准。原绘图CSV列保持兼容；两个入口明细现在默认写出。','']
 (O/'report.md').write_text('\n'.join(lines),encoding='utf8');print(json.dumps(groups,indent=2));print('All CSV checks passed')
if __name__=='__main__':main()