import sys
from pathlib import Path
sys.path.insert(0,str(Path('src/inner').resolve()))
import numpy as np,mosek
from parameters import Parameters
from environment import build_environment
from variables import InnerContext,InnerVariables
from fusion_pc3p import get_fusion_model,_MODEL_CACHE
for value in [1e-8,1e-7,1e-8]:
 p=Parameters(mosek_tol_rel_gap=value)
 e=build_environment(p,np.random.default_rng(p.seed))
 c=InnerContext(p,e,InnerVariables.create(p))
 m,fp,fv=get_fusion_model(c)
 task=m.getTask()
 actual=task.getdouparam(mosek.dparam.intpnt_co_tol_rel_gap)
 assert actual==value,(actual,value)
 assert task.getdouparam(mosek.dparam.intpnt_co_tol_pfeas)==1e-8
 assert task.getdouparam(mosek.dparam.intpnt_co_tol_dfeas)==1e-8
 print('cache key',(p.I,p.J,p.N),'RelGap',actual,'cache size',len(_MODEL_CACHE))
assert len(_MODEL_CACHE)==1
m.dispose()