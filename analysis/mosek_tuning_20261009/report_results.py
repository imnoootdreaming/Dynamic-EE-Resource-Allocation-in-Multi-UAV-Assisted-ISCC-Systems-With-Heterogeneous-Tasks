import json,hashlib,collections
from pathlib import Path
import numpy as np
import benchmark as b
P=b.OUT

def read(stage):return [json.loads(x) for x in (P/stage/'results.jsonl').read_text().splitlines()]
def main():
 full=read('full');confirm=read('confirm');b.CONFIGS.update(gap_threads_1=dict(intpntCoTolRelGap=1e-7,numThreads=1),gap_threads_2=dict(intpntCoTolRelGap=1e-7,numThreads=2))
 b.summary(confirm,P/'confirm')
 samples=b.load_outer_samples(b.ROOT/'src/inner/feasible_outer_samples.csv',b.Parameters())
 cs={i:b.ctx(s) for i,s in enumerate(samples)}
 fly={i:float(c.omega_2*np.sum(c.E_uav_fly)) for i,c in cs.items()}
 refs={r['sample_id']:r for r in full if r['config']=='baseline'}
 details=[]
 for name in dict.fromkeys(r['config'] for r in full):
  rr=[r for r in full if r['config']==name];energy=[];component={k:[] for k in ['W_sen_beam','B_off_beam','f_uav_freq','z_aux_rate','D_cu_off']};absdiff=[]
  for r in rr:
   sid=r['sample_id'];ref=refs[sid]
   if not r['returned'] or not ref['returned']:continue
   delta=abs(r['objective']-ref['objective']);absdiff.append(delta);energy.append(delta/max(abs(ref['objective']-fly[sid]),1e-12))
   with np.load(P/'full'/f'{name}_s{sid}_r0.npz') as a,np.load(P/'full'/f'baseline_s{sid}_r0.npz') as base:
    for k in component:component[k].append(float(np.linalg.norm(a[k]-base[k])/max(np.linalg.norm(base[k]),1e-12)))
  details.append(dict(config=name,max_variable_energy_relative_difference=max(energy,default=None),max_objective_absolute_difference_j=max(absdiff,default=None),max_component_relative_norm_difference={k:max(v,default=None) for k,v in component.items()},outer_iterations=dict(collections.Counter(r['iterations'] for r in rr)),termination_codes=dict(collections.Counter(str(c) for r in rr for c in r['termination_codes'])),recovered_failed_checks=dict(collections.Counter(k for r in rr for k in r['validation']['recovered_beams']['failed_checks']))))
 (P/'accuracy_details.json').write_text(json.dumps(details,indent=2),encoding='utf8')
 timings=[];rng=np.random.default_rng(1208)
 for name in dict.fromkeys(r['config'] for r in confirm):
  rr=[r for r in confirm if r['config']==name];ratios=[]
  for sid in range(30):
   a=[r['elapsed_s'] for r in rr if r['sample_id']==sid];z=[r['elapsed_s'] for r in confirm if r['config']=='baseline' and r['sample_id']==sid]
   ratios.append(np.median(a)/np.median(z))
  ratios=np.array(ratios);boots=np.median(rng.choice(ratios,size=(5000,30)),axis=1)
  timings.append(dict(config=name,paired_sample_median_ratio=float(np.median(ratios)),bootstrap95_ratio=[float(x) for x in np.percentile(boots,[2.5,97.5])],median_seconds=float(np.median([r['elapsed_s'] for r in rr])),p95_seconds=float(np.percentile([r['elapsed_s'] for r in rr],95))))
 (P/'timing_details.json').write_text(json.dumps(timings,indent=2),encoding='utf8')
 manifest=json.loads((P/'full/manifest.json').read_text());integrity={p:hashlib.sha256((b.ROOT/p).read_bytes()).hexdigest()==h for p,h in manifest['sha256'].items()}
 (P/'source_integrity.json').write_text(json.dumps(integrity,indent=2),encoding='utf8')
 fs=json.loads((P/'full/summary.json').read_text());csum=json.loads((P/'confirm/summary.json').read_text());dd={x['config']:x for x in details};tt={x['config']:x for x in timings}
 lines=['# MOSEK 参数对照测试（2026-10-09）','','## 结论与适用边界','','以现有 30 个 CSV 样本为范围；28 配置全样本扫描（840 次），10 配置交错顺序重复 3 次（900 次），另有 6 样本初筛（168 次）。以下不计首次建模/许可证预热，计入 PC3P 初始化、同步、求解和解提取；独立验证与结果写盘在计时之外。没有修改生产源码、默认参数或输入 CSV。','','目标接近不能代替物理可行。严格验证器的 relative=1e-6、matrix=1e-7、rank_relative=1e-5、rank_absolute=1e-5 保持不变。基准全部返回且外层收敛，但完整验证 0/30：恢复波束 19/30 违反 c08，最大约 272.803 bit；全部未通过严格相对秩一检查，主要涉及近零卸载功率矩阵。不能把 0/30 解读成全场景无可行解，也不因参数目标接近便宣称安全。','','单独放宽 RelGap=1e-7 是精度变化较小的候选；是否有稳定速度收益以交错重复及区间为准。单独放宽 Pfeas 未显示可靠优势且损失原有约束通过样本。单独放宽 Dfeas/MuRed 在本样本集目标与数值路径未改变，未证明有迭代收益。联合 1e-6/1e-5 导致大量外层不收敛，不建议。迭代上限 10/20 无完整返回，30 丢失 2 个样本，50 与基准数值一致但未降低实际内部迭代。时间上限 0.02/0.05/0.1/0.2 s 仅为每次 MOSEK optimize 的限制；返回率依赖当前机器负载，不作为可靠加速方式。','','## 输入与复现','','- Python: '+manifest['python']+'; NumPy '+manifest['numpy']+'; MOSEK '+'.'.join(map(str,manifest['mosek']))+'。','- 固定 seed=1208, I=4, J=10, N=10, SINR=5 dB，PC3P max_iterations=10。','- 主程序同源环境、初始点、Fusion 模型缓存及求解函数；每配置显式复位全部七类参数，避免缓存设置残留。','- 初筛 sid=0,6,7,20,25,27；全样本 sid=0..29。确认阶段按样本随机交错配置，重复3次，order_seed=20261009。','- 配置和源码/输入 SHA256 见 full/manifest.json；结束校验见 source_integrity.json。','- 本次基准外层次数分布 3次23个、4次6个、5次1个，不能直接换算用户所述6次/2秒；既有 convergence_iterations.csv 未被覆盖。','','复现命令：','','```powershell','D:/anaconda3/envs/cvx_new/python.exe analysis/mosek_tuning_20261009/benchmark.py --stage full --samples '+','.join(map(str,range(30))),'D:/anaconda3/envs/cvx_new/python.exe analysis/mosek_tuning_20261009/confirm.py','D:/anaconda3/envs/cvx_new/python.exe analysis/mosek_tuning_20261009/report_results.py','```','','## 全样本精度与外层收敛','','全目标包含不随内层决策变化的加权飞行能耗；同时报告减去该固定项后的可变能耗，避免常数掩盖变化。基准可变能耗范围 0.099992–0.886322 J。','','| 配置 | 返回/30 | 外层收敛/30 | 最大全目标相对差 | 最大可变能耗相对差 | 协方差原约束通过/30 | 恢复波束原约束通过/30 | 内部最大迭代 |','|---|---:|---:|---:|---:|---:|---:|---:|']
 for x in fs:
  d=dd[x['config']];fmt=lambda v:'—' if v is None else f'{v:.6g}'
  lines.append(f"| {x['config']} | {x['returned']} | {x['converged']} | {fmt(x['max_objective_relative_difference'])} | {fmt(d['max_variable_energy_relative_difference'])} | {x['covariance_pass']} | {x['recovered_pass']} | {x['ipm_max']} |")
 lines+=['','## 交错重复计时','','每配置90次。配对比值为：每个样本先取三次耗时中位数，再除以该样本基准中位数，最后对30个样本取中位数；区间以样本为单位 bootstrap 5000次，只反映当前样本/机器条件。比值<1表示更快。分组全扫描的计时受漂移影响，不用于可靠排序。','','| 配置 | 总耗时中位数/s | P95/s | 配对耗时比 | 配对比95%区间 | 外层收敛/90 | 最大目标相对差 |','|---|---:|---:|---:|---|---:|---:|']
 for x in csum:
  t=tt[x['config']];lo,hi=t['bootstrap95_ratio'];lines.append(f"| {x['config']} | {t['median_seconds']:.4f} | {t['p95_seconds']:.4f} | {t['paired_sample_median_ratio']:.4f} | [{lo:.4f}, {hi:.4f}] | {x['converged']} | {x['max_objective_relative_difference']:.6g} |")
 lines+=['','## 解分量与停止机制','','完整解数组保存为各阶段 NPZ；各分量最大相对范数差见 accuracy_details.json。近退化方向（特别是接近零的 B）可有较大的相对差，不能以总能耗差小断言所有资源分配相同。','','基准98次内部求解中6次终止代码100006（当前安装包解释为 rescode.trm_stall），RelGap=1e-7的100次中1次。放宽间隙可能改变停止路径，但这些计数不能单独证明因果。原代码仅检查 problem status，严格约束验证独立执行。','','## 后续建议','','保持生产参数不变，优先推进 T04 的 c08/恢复波束问题。若需要试用加速候选，应接入原始约束验收及基准精度回退，计入回退开销后重新评估；本次未实现或测量回退机制，也未验证新场景泛化。','','官方参考：MOSEK ApS, MOSEK Fusion API for Python, Parameters 与 Conic Optimization（官方文档，非论文）：','- https://docs.mosek.com/latest/pythonfusion/parameters.html','- https://docs.mosek.com/latest/pythonfusion/solving-conic.html','- PDF: https://docs.mosek.com/11.1/pythonfusion.pdf','']
 (P/'report.md').write_text('\n'.join(lines),encoding='utf8')
 print(json.dumps(timings,indent=2));print('integrity',integrity)
if __name__=='__main__':main()