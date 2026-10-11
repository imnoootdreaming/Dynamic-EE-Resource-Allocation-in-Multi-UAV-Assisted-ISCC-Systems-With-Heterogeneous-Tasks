import os,sys,json,hashlib,shutil,subprocess,time
from pathlib import Path
R=Path(__file__).resolve().parents[2];O=Path(__file__).resolve().parent;INNER=R/'src/inner'
for stage in ['baseline','accelerated']:(O/stage).mkdir(exist_ok=True)
outputs=['first_sample_cccp_history.csv','convergence_iterations.csv','convergence_iterations_details.csv']
manifest=dict(python=sys.executable,config=dict(samples=30,seed=1208,candidates=50,baseline_rel_gap=1e-8,accelerated_rel_gap=1e-7),commands=[],sha256={p.relative_to(R).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in [INNER/n for n in ['parameters.py','fusion_pc3p.py','pc3p.py','gc3p.py','main.py','compare_pc3p_gc3p.py','solution_checks.py','feasible_outer_samples.csv']]})
(O/'run_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
env=os.environ.copy();env['PYTHONIOENCODING']='utf-8';env['PC3P_BACKEND']='fusion'
for stage,tol in [('baseline','1e-8'),('accelerated','1e-7')]:
 jobs=[('main',[sys.executable,'-u','main.py','--mosek-rel-gap',tol]),('compare',[sys.executable,'-u','compare_pc3p_gc3p.py','--mosek-rel-gap',tol,'--cases','30','--candidates','50','--csv','feasible_outer_samples.csv','--output',str(O/stage/'compare_pc3p_gc3p_results.csv'),'--detail-output',str(O/stage/'compare_pc3p_gc3p_details.csv')])]
 for name,cmd in jobs:
  print('START',stage,name,flush=True);start=time.perf_counter()
  with (O/stage/f'{name}.log').open('w',encoding='utf8') as f:
   completed=subprocess.run(cmd,cwd=INNER,env=env,stdout=f,stderr=subprocess.STDOUT)
  record=dict(stage=stage,entry=name,command=cmd,cwd=str(INNER),exit_code=completed.returncode,wall_s=time.perf_counter()-start)
  manifest['commands'].append(record);(O/'run_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
  print('END',json.dumps(record),flush=True)
  if completed.returncode:raise SystemExit('Failed: '+str(O/stage/f'{name}.log'))
  if name=='main':
   for filename in outputs:shutil.copy2(INNER/filename,O/stage/filename)
  elif stage=='accelerated':
   for filename in ['compare_pc3p_gc3p_results.csv','compare_pc3p_gc3p_details.csv']:shutil.copy2(O/stage/filename,INNER/filename)
manifest['input_unchanged']=hashlib.sha256((INNER/'feasible_outer_samples.csv').read_bytes()).hexdigest()==manifest['sha256']['src/inner/feasible_outer_samples.csv']
(O/'run_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8');print('DONE',flush=True)