"""Interleaved paired repeats; use benchmark instrumentation and validators."""
import json,random,time
import numpy as np
import benchmark as b

def main():
 d=b.OUT/'confirm';d.mkdir(parents=True,exist_ok=True)
 samples=b.load_outer_samples(b.ROOT/'src/inner/feasible_outer_samples.csv',b.Parameters())
 names=['baseline','intpntCoTolRelGap_1e-07','joint_1e-07','threads_1','threads_2','threads_4','threads_8','iterations_50','gap_threads_1','gap_threads_2']
 b.CONFIGS['gap_threads_1']={'intpntCoTolRelGap':1e-7,'numThreads':1}
 b.CONFIGS['gap_threads_2']={'intpntCoTolRelGap':1e-7,'numThreads':2}
 manifest=json.loads((b.OUT/'full/manifest.json').read_text())
 manifest.update(repeats=3,stage='confirm',interleaved=True,order_seed=20261009,configs={n:dict(b.BASE,**b.CONFIGS[n]) for n in names})
 (d/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
 m=b.prepare(b.ctx(samples[0]),'baseline');b.F.run_pc3p_fusion(b.ctx(samples[0]));rows=[];rng=random.Random(20261009)
 with (d/'results.jsonl').open('w',encoding='utf8') as f:
  for rep in range(3):
   ids=list(range(len(samples)));rng.shuffle(ids)
   for sid in ids:
    order=names.copy();rng.shuffle(order)
    for name in order:
     c=b.ctx(samples[sid]);b.prepare(c,name);b.CALLS.clear();b.SYNCS.clear();t=time.perf_counter();error=None
     try:r=b.F.run_pc3p_fusion(c)
     except Exception as exc:error=repr(exc);r=dict(status='benchmark_exception',converged=False,iterations=0,objective_p4=None,W_sen_beam=None,rank1_gap=None)
     elapsed=time.perf_counter()-t;v=b.validate_original_solution(c.params,c.environment,r);outer=b.outer_action_checks(c.params,c.environment)
     row=dict(config=name,sample_id=sid,repeat=rep,elapsed_s=elapsed,solve_wall_s=sum(x['wall_s'] for x in b.CALLS),optimizer_s=sum(x['optimizerTime'] or 0 for x in b.CALLS),sync_s=sum(b.SYNCS),ipm_iterations=[x['intpntIter'] for x in b.CALLS],termination_codes=[x['optimizeResponse'] for x in b.CALLS],status=r['status'],converged=r['converged'],iterations=r['iterations'],objective=r['objective_p4'],rank1_gap=r['rank1_gap'],returned=r['W_sen_beam'] is not None,verified=v['physical_solution_verified'],outer_passed=outer['passed'],validation=v,outer=outer,error=error,solver_diagnostics=r.get('solver_diagnostics'),obj_history=r.get('obj_history'))
     if row['returned']:np.savez_compressed(d/f'{name}_s{sid}_r{rep}.npz',**{k:r[k].copy() for k in ['W_sen_beam','B_off_beam','w_sen_beam','b_off_beam','f_uav_freq','f_cu_freq','z_aux_rate','D_cu_off']})
     rows.append(row);f.write(json.dumps(row,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n');f.flush()
   b.summary(rows,d);print(f'Completed repetition {rep+1}/3, {len(rows)} runs',flush=True)
 m.dispose();print('DONE '+str(d),flush=True)
if __name__=='__main__':main()