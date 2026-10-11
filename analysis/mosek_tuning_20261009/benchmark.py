import argparse,csv,hashlib,json,os,platform,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src/inner'),str(ROOT/'src/outer/reward')]
import numpy as np
import mosek,mosek.fusion as mf
import fusion_pc3p as F
from parameters import Parameters
from environment import build_environment
from variables import InnerContext,InnerVariables
from outer_sampler import load_outer_samples
from feasibility_diagnostics import validate_original_solution,outer_action_checks
OUT=Path(__file__).resolve().parent
BASE=dict(intpntCoTolPfeas=1e-8,intpntCoTolDfeas=1e-8,intpntCoTolRelGap=1e-8,intpntCoTolMuRed=1e-8,intpntMaxIterations=400,optimizerMaxTime=-1.0,numThreads=0)
CONFIGS={'baseline':{}}
for k in list(BASE)[:4]:
 for t in (1e-7,1e-6,1e-5): CONFIGS[f'{k}_{t:.0e}']={k:t}
for t in (1e-7,1e-6,1e-5): CONFIGS[f'joint_{t:.0e}']={k:t for k in list(BASE)[:4]}
for n in (1,2,4,8): CONFIGS[f'threads_{n}']={'numThreads':n}
for n in (10,20,30,50): CONFIGS[f'iterations_{n}']={'intpntMaxIterations':n}
for t in (.02,.05,.1,.2): CONFIGS[f'time_{t}']={'optimizerMaxTime':t}
solve0=mf.Model.solve
sync0=F.sync_fusion
CALLS=[]
SYNCS=[]
def solve(self,*a,**kw):
 t=time.perf_counter()
 try:return solve0(self,*a,**kw)
 finally:
  r={'wall_s':time.perf_counter()-t}
  for k,fn in [('optimizerTime',self.getSolverDoubleInfo),('intpntIter',self.getSolverIntInfo),('optimizeResponse',self.getSolverIntInfo)]:
   try:r[k]=fn(k)
   except Exception:r[k]=None
  CALLS.append(r)
def sync(*a):
 t=time.perf_counter()
 try:return sync0(*a)
 finally:SYNCS.append(time.perf_counter()-t)
F.sync_fusion=sync
mf.Model.solve=solve
def ctx(s):
 p=Parameters();e=build_environment(p,np.random.default_rng(p.seed),outer_variables=s)
 return InnerContext(p,e,InnerVariables.create(p))
def prepare(c,name):
 m,fp,fv=F.get_fusion_model(c)
 for k,v in dict(BASE,**CONFIGS[name]).items():m.setSolverParam(k,v)
 return m
def summary(rows,d):
 refs={}
 for r in rows:
  if r['config']=='baseline':refs.setdefault((r['sample_id'],r['repeat']),r)
 out=[]
 for name in dict.fromkeys(r['config'] for r in rows):
  rr=[r for r in rows if r['config']==name];diff=[];ratios=[]
  changes=dict(returned_changes=0,convergence_changes=0,verified_losses=0,covariance_pass_losses=0,recovered_pass_losses=0,rank_pass_losses=0)
  for r in rr:
   b=refs.get((r['sample_id'],r['repeat']))
   if b is None:continue
   ratios.append(r['elapsed_s']/b['elapsed_s'])
   if r['returned'] and b['returned']:diff.append(abs(r['objective']-b['objective'])/max(abs(b['objective']),1e-12))
   changes['returned_changes']+=r['returned']!=b['returned']
   changes['convergence_changes']+=r['converged']!=b['converged']
   changes['verified_losses']+=b['verified'] and not r['verified']
   for cat,key in [('covariance','covariance_pass_losses'),('recovered_beams','recovered_pass_losses'),('rank','rank_pass_losses')]:
    changes[key]+=b['validation'][cat]['passed'] and not r['validation'][cat]['passed']
  out.append(dict(config=name,runs=len(rr),settings=CONFIGS[name],elapsed_median=float(np.median([r['elapsed_s'] for r in rr])),elapsed_p95=float(np.percentile([r['elapsed_s'] for r in rr],95)),optimizer_median=float(np.median([r['optimizer_s'] for r in rr])),sync_median=float(np.median([r['sync_s'] for r in rr])),ipm_max=max((n or 0 for r in rr for n in r['ipm_iterations']),default=0),returned=sum(r['returned'] for r in rr),converged=sum(r['converged'] for r in rr),verified=sum(r['verified'] for r in rr),covariance_pass=sum(r['validation']['covariance']['passed'] for r in rr),recovered_pass=sum(r['validation']['recovered_beams']['passed'] for r in rr),rank_pass=sum(r['validation']['rank']['passed'] for r in rr),max_objective_relative_difference=max(diff,default=None),median_paired_time_ratio=float(np.median(ratios)) if ratios else None,**changes))
 (d/'summary.json').write_text(json.dumps(out,indent=2,default=lambda x: x.item() if isinstance(x,np.generic) else str(x)),encoding='utf8')
 with (d/'summary.csv').open('w',newline='',encoding='utf8') as f:
  w=csv.DictWriter(f,fieldnames=out[0].keys());w.writeheader();w.writerows(out)
 return out
def main():
 a=argparse.ArgumentParser();a.add_argument('--stage',default='screen');a.add_argument('--samples',default='0,6,7,20,25,27');a.add_argument('--configs',default=','.join(CONFIGS));a.add_argument('--repeats',type=int,default=1);a=a.parse_args()
 d=OUT/a.stage;d.mkdir(parents=True,exist_ok=True);ids=list(map(int,a.samples.split(',')));names=a.configs.split(',')
 samples=load_outer_samples(ROOT/'src/inner/feasible_outer_samples.csv',Parameters())
 paths=['src/inner/fusion_pc3p.py','src/inner/parameters.py','src/inner/environment.py','src/inner/pc3p.py','src/inner/cccp_params.py','src/inner/variables.py','src/inner/feasible_outer_samples.csv','src/outer/reward/feasibility_diagnostics.py']
 manifest=dict(python=sys.executable,python_version=sys.version,numpy=np.__version__,mosek=mosek.Env.getversion(),platform=platform.platform(),cpu_count=os.cpu_count(),samples=ids,available_samples=len(samples),repeats=a.repeats,configs={n:dict(BASE,**CONFIGS[n]) for n in names},parameters=repr(Parameters()),sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths})
 (d/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
 m=prepare(ctx(samples[ids[0]]),'baseline');F.run_pc3p_fusion(ctx(samples[ids[0]]));rows=[]
 with (d/'results.jsonl').open('w',encoding='utf8') as f:
  for rep in range(a.repeats):
   order=names if rep%2==0 else list(reversed(names))
   for name in order:
    for sid in ids:
     c=ctx(samples[sid]);prepare(c,name);CALLS.clear();SYNCS.clear();t=time.perf_counter();error=None
     try:r=F.run_pc3p_fusion(c)
     except Exception as exc:error=repr(exc);r=dict(status='benchmark_exception',converged=False,iterations=0,objective_p4=None,W_sen_beam=None,rank1_gap=None)
     elapsed=time.perf_counter()-t;v=validate_original_solution(c.params,c.environment,r);outer=outer_action_checks(c.params,c.environment)
     row=dict(config=name,sample_id=sid,repeat=rep,elapsed_s=elapsed,solve_wall_s=sum(x['wall_s'] for x in CALLS),optimizer_s=sum(x['optimizerTime'] or 0 for x in CALLS),sync_s=sum(SYNCS),ipm_iterations=[x['intpntIter'] for x in CALLS],termination_codes=[x['optimizeResponse'] for x in CALLS],status=r['status'],converged=r['converged'],iterations=r['iterations'],objective=r['objective_p4'],rank1_gap=r['rank1_gap'],returned=r['W_sen_beam'] is not None,verified=v['physical_solution_verified'],outer_passed=outer['passed'],validation=v,outer=outer,error=error,solver_diagnostics=r.get('solver_diagnostics'),obj_history=r.get('obj_history'))
     if row['returned']:
      np.savez_compressed(d/f'{name}_s{sid}_r{rep}.npz',**{k:r[k].copy() for k in ['W_sen_beam','B_off_beam','w_sen_beam','b_off_beam','f_uav_freq','f_cu_freq','z_aux_rate','D_cu_off']})
     rows.append(row);f.write(json.dumps(row, default=lambda x: x.item() if isinstance(x, np.generic) else str(x))+'\n');f.flush()
    s=summary(rows,d);entry=next(x for x in s if x['config']==name)
    print(json.dumps(entry,default=lambda x: x.item() if isinstance(x,np.generic) else str(x)),flush=True)
 m.dispose();print('DONE '+str(d),flush=True)
if __name__=='__main__':main()