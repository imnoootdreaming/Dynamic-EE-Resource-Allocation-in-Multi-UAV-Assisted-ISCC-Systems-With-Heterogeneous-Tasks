"""Frozen-log PPO audit and publication-style diagnostic plots; no policy runs."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
from pathlib import Path
import hashlib
import json
import re
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_file_loader import LegacyEventFileLoader
from analyze import OUT, OUTER, FROZEN, write, stats

def audit():
    n=int(pd.read_csv(FROZEN/'training_updates.csv')['update'].max())
    event_path=next((OUTER/'runs/beta-hppo/2026-10-19_09-44-15_beta-hppo_sync_result').glob('events*'))
    tags={}; selected=[]
    for event in LegacyEventFileLoader(str(event_path)).Load():
        for v in event.summary.value:
            if not v.HasField('simple_value'): continue
            # Agent's step is zero-based; learner reward/policy tags are one-based.
            update=int(event.step)+(1 if v.tag.startswith('BS/') else 0)
            if not 1 <= update <= n: continue
            tags.setdefault(v.tag,[]).append((update,float(v.simple_value),float(event.wall_time)))
    result={}
    for tag,values in tags.items():
        a=np.asarray(values)
        finite=np.isfinite(a[:,1])
        result[tag]={'n':len(a),'first_update':int(a[:,0].min()),'last_update':int(a[:,0].max()),
                     'all_finite':bool(finite.all()),'unique_updates':len(set(a[:,0])),
                     'last500':stats(a[a[:,0]>n-500,1]) if finite.all() else None}
        if tag.startswith('BS/'):
            result[tag]['first500']=stats(a[a[:,0]<=500,1]) if finite.all() else None
            selected += [{'tag':tag,'update':int(u),'value':float(v),'wall_time':float(t)} for u,v,t in values]
    pd.DataFrame(selected).to_csv(OUT/'ppo_scalars.csv',index=False)
    write('tensorboard_summary.json',result)
    # Read policy moment summaries from the already frozen text log.
    policy=[]; discrete=[]
    for line in (FROZEN/'mhbppo_train.log').read_text(encoding='utf-8').splitlines():
        m=re.search(r'\[POLICY\] iter=(\d+).*?head=(\w+).*?mean=\[([^]]+)\] std=\[([^]]+)\]',line)
        if m:
            policy.append({'update':int(m[1]),'head':m[2], 'mean':float(np.mean(np.fromstring(m[3],sep=','))), 'std':float(np.mean(np.fromstring(m[4],sep=',')))})
        m=re.search(r'\[POLICY_DISCRETE\] iter=(\d+).*',line)
        if m:
            parts=re.findall(r'entropy=([0-9.]+),pmax=([0-9.]+)',line)
            discrete.append({'update':int(m[1]),'entropy':float(np.mean([float(x[0]) for x in parts])),'pmax':float(np.mean([float(x[1]) for x in parts]))})
    pp=pd.DataFrame(policy);dd=pd.DataFrame(discrete)
    write('policy_summary.json',{'continuous':{name:{str(h):{'mean':float(g['mean'].mean()),'std':float(g['std'].mean())} for h,g in pp[(pp['update']<=500) if name=='first500' else (pp['update']>n-500)].groupby('head')} for name in ['first500','last500']},
                                 'discrete':{name:{k:float(g[k].mean()) for k in ['entropy','pmax']} for name,g in [('first500',dd[dd['update']<=500]),('last500',dd[dd['update']>n-500]) ]}})
    print(json.dumps({'tags':len(result),'BS':{k:v for k,v in result.items() if k.startswith('BS/')},'nonfinite':[k for k,v in result.items() if not v['all_finite']]},ensure_ascii=False))

def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    d=pd.read_csv(OUT/'metrics_with_timestamps.csv')
    p=pd.read_csv(OUT/'physical_steps.csv')
    fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    c=['#0072B2','#D55E00','#009E73','#CC79A7']
    x=d['update']
    ax=axs[0,0]
    ax.plot(x,d.completion_rate.rolling(100).mean(),color=c[0],label='Completion, last 100 updates')
    ax.plot(x,(d.optimal_count*1.25).rolling(100).mean(),color=c[2],label='Solver optimal, last 100 updates')
    ax.set(ylabel='Slots (%)',ylim=(0,101),title='Learning is effective in the fixed training scene')
    ax.legend(fontsize=8,loc='lower right')
    ax=axs[0,1]
    ax.plot(x,d.reward.rolling(100).mean(),color=c[0],label='Total reward')
    ax.set(ylabel='Mean reward',title='Reward improves; sensing shortfall decreases')
    ax2=ax.twinx();ax2.plot(x,d.sensing_Q.rolling(100).mean(),color=c[1],label='Sensing shortfall Q')
    ax2.set_yscale('log');ax2.set_ylabel('Sensing shortfall Q (log)',color=c[1])
    ax.legend(fontsize=8,loc='lower right');ax2.legend(fontsize=8,loc='upper right')
    ax=axs[1,0]
    ax.plot(x,d.collection_wait_seconds.rolling(100).mean(),color=c[0],label='Collection wait')
    ax.plot(x,d.update_seconds.rolling(100).mean(),color=c[1],label='PPO update')
    ax.plot(x,d.logging_seconds.rolling(100).mean(),color=c[2],label='Detailed logging')
    ax.set(ylabel='Seconds per update',title='Sampling dominates elapsed time')
    ax.legend(fontsize=8)
    ax=axs[1,1]
    g=p.groupby('slot')[['success','physical_verified','beam_pass']].mean()*100
    ax.plot(g.index,g.success,color=c[0],marker='.',label='Completion')
    ax.plot(g.index,g.beam_pass,color=c[2],marker='.',label='Recovered-beam constraints pass')
    ax.plot(g.index,g.physical_verified,color=c[1],linestyle=':',label='Strict covariance + rank + beams pass')
    ax.set(xlabel='Slot (zero-based)',ylabel='Slots (%)',ylim=(0,101),title='Last 500 updates: coverage across all 40 slots')
    ax.legend(fontsize=8)
    for a in axs.flat:a.grid(alpha=.2)
    for a in axs[0,:]:a.set_xlabel('Completed PPO updates')
    axs[1,0].set_xlabel('Completed PPO updates')
    fig.savefig(OUT/'training_analysis.png',dpi=180)
    fig.savefig(OUT/'training_analysis.svg')
    plt.close(fig)
    print('plots_saved')

if __name__=='__main__':
    import sys
    plot() if len(sys.argv)>1 and sys.argv[1]=='plot' else audit()
