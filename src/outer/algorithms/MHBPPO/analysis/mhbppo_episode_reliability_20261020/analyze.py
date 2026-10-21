"""Freeze current log and analyze whole-episode reliability; no training/solver."""
from pathlib import Path
import ast, hashlib, io, json, math, re
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent
OUTER=OUT.parents[3]
RUN=OUTER/'ckpt/MHBPPO/2026-10-19_09-44-15_mhbppo_seed_42'
LOG=OUTER/'logs/MHBPPO/2026-10-19_09-44-15/mhbppo_train.log'
FROZEN=OUT/'frozen';FROZEN.mkdir(exist_ok=True)

def dump(name,value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')

def read_complete(path):
    with path.open('rb') as f:
        f.seek(0,2);bound=f.tell();f.seek(0);raw=f.read(bound)
    return raw[:raw.rfind(b'\n')+1]

def block_difference(a,b,block=25,seed=1020):
    # Update-level means retain the two workers together. Moving-block bootstrap
    # describes uncertainty of nonstationary training windows, not policy test CI.
    rng=np.random.default_rng(seed)
    def draw(x):
        starts=rng.integers(0,len(x),size=(3000,math.ceil(len(x)/block)))
        ids=(starts[:,:,None]+np.arange(block))%len(x)
        return x[ids.reshape(3000,-1)[:,:len(x)]].mean(axis=1)
    diff=draw(b)-draw(a)
    return {'point_pp':float((b.mean()-a.mean())*100),'block_updates':block,
            'bootstrap95_pp':(np.quantile(diff,[.025,.975])*100).tolist(),
            'scope':'Descriptive moving-block bootstrap of training windows, not frozen-policy inference.'}

def main():
    d=pd.read_csv(io.BytesIO(read_complete(RUN/'training_updates.csv')))
    text=read_complete(LOG).decode('utf-8-sig');env=[];times={};pol=[]
    for line in text.splitlines():
        m=re.search(r'\[learner\] iter=(\d+) transitions=',line)
        if m:times[int(m[1])]=line[:19]
        m=re.search(r'\[ENV\] iter=(\d+) worker=(\d+) episode=(\d+) transitions=(\d+) wv=(\d+) reward=([^ ]+) completion=([^%]+)% solver_counts=(\{.*\}) collection=([^s]+)s',line)
        if m:
            counts=ast.literal_eval(m[8]); completion=float(m[7])
            env.append({'update':int(m[1]),'worker':int(m[2]),'episode':int(m[3]),'wv':int(m[5]),
                        'reward':float(m[6]),'completion':completion,'completion_failures':int(round(40*(1-completion/100))),
                        'optimal_failures':40-counts['optimal'],'all_completion':completion==100,
                        'all_optimal':counts['optimal']==40, **counts})
        m=re.search(r'\[POLICY\] iter=(\d+).*?head=(\w+).*?mean=\[([^]]+)\] std=\[([^]]+)\]',line)
        if m:pol.append({'update':int(m[1]),'head':m[2],'mean':float(np.fromstring(m[3],sep=',').mean()),'std':float(np.fromstring(m[4],sep=',').mean())})
    e=pd.DataFrame(env);p=pd.DataFrame(pol)
    complete=e.groupby('update').size();n=min(int(d['update'].max()),int(complete[complete==2].index.max()),max(times))
    d=d[d['update']<=n];e=e[e['update']<=n];p=p[p['update']<=n]
    d.to_csv(FROZEN/'training_updates.csv',index=False);e.to_csv(OUT/'episodes.csv',index=False);p.to_csv(OUT/'policy_moments.csv',index=False)
    (FROZEN/'hyperparameters.json').write_bytes((RUN/'hyperparameters.json').read_bytes())
    kept=[l for l in text.splitlines() if ('iter=' not in l or not re.search(r'\biter=(\d+)',l) or int(re.search(r'\biter=(\d+)',l)[1])<=n)]
    (FROZEN/'training.log').write_text('\n'.join(kept)+'\n',encoding='utf8')
    def window(lo,hi):
        sub=e[e['update'].between(lo,hi)];ds=d[d['update'].between(lo,hi)];ps=p[p['update'].between(lo,hi)]
        return {'range':[lo,hi],'episodes':len(sub),'all_completion_n':int(sub.all_completion.sum()),
                'all_optimal_n':int(sub.all_optimal.sum()),'all_completion_pct':float(sub.all_completion.mean()*100),
                'all_optimal_pct':float(sub.all_optimal.mean()*100),'slot_completion_pct':float(sub.completion.mean()),
                'slot_optimal_pct':float(sub.optimal.mean()/40*100),'mean_completion_failures':float(sub.completion_failures.mean()),
                'mean_optimal_failures':float(sub.optimal_failures.mean()),
                'completion_failure_hist':{str(int(k)):int(v) for k,v in sub.completion_failures.value_counts().sort_index().items()},
                'optimal_failure_hist':{str(int(k)):int(v) for k,v in sub.optimal_failures.value_counts().sort_index().items()},
                'reward':float(ds.reward.mean()),'Q':float(ds.sensing_Q.mean()),
                'policy_std':{str(k):float(v['std'].mean()) for k,v in ps.groupby('head')},
                'per_worker':{str(k):{'all_optimal_pct':float(v.all_optimal.mean()*100),'all_completion_pct':float(v.all_completion.mean()*100)} for k,v in sub.groupby('worker')}}
    result={'freeze':{'last_update':n,'last_update_time':times[n],'read_time':datetime.now(timezone(timedelta(hours=8))).isoformat()},
            'windows':{'last500':window(n-499,n),'previous500':window(n-999,n-500),'last100':window(n-99,n),
                       'U60_window':window(5994,6493)},
            'blocks500':[window(i,min(i+499,n)) for i in range(1,n+1,500)],
            'recent_blocks250':[window(i,i+249) for i in range(n-1499,n+1,250)]}
    ep_by_update=e.groupby('update')[['all_optimal','all_completion']].mean()
    result['recent_difference']={col:{str(b):block_difference(ep_by_update.loc[n-999:n-500,col].to_numpy(),ep_by_update.loc[n-499:n,col].to_numpy(),b) for b in [10,25,50]} for col in ['all_optimal','all_completion']}
    result['target_math']={'independent_per_slot_required':.5**(1/40),'gamma_lambda_39':(.99*.95)**39,
                           'lr_after_update':.0003*max(1-(n-1)/10000,.1),
                           'independent_all_optimal_proxy':(result['windows']['last500']['slot_optimal_pct']/100)**40,
                           'independent_all_completion_proxy':(result['windows']['last500']['slot_completion_pct']/100)**40}
    result['validation']={'continuous_updates':d['update'].tolist()==list(range(1,n+1)),
       'two_workers':e.groupby('update').worker.apply(lambda x:sorted(x)==[0,1]).all().item(),
       'versions':bool((e.wv==e['update']-1).all()),'optimal_matches_csv':bool((e.groupby('update').optimal.sum().to_numpy()==d.optimal_count.to_numpy()).all()),
       'completion_matches_csv':bool(np.allclose(e.groupby('update').completion.mean().to_numpy(),d.completion_rate.to_numpy())),
       'U60_completion_reproduced':result['windows']['U60_window']['all_completion_n']==167}
    dump('results.json',result)
    dump('freeze_manifest.json',{str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in FROZEN.iterdir()})
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
