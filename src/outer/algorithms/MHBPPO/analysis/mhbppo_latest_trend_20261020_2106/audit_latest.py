"""Frozen training health, policy trends and independent arithmetic checks."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
from pathlib import Path
import csv, hashlib, json, re
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_file_loader import LegacyEventFileLoader

OUT=Path(__file__).resolve().parent;OUTER=OUT.parents[3]
RUN=OUTER/'ckpt/MHBPPO/2026-10-19_09-44-15_mhbppo_seed_42'
LOG=OUTER/'logs/MHBPPO/2026-10-19_09-44-15'

def dump(name,data):
    (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')

def stats(a):
    a=np.asarray(a,dtype=float)
    return {'n':len(a),'mean':float(a.mean()),'median':float(np.median(a)),'min':float(a.min()),'max':float(a.max()),'p95':float(np.quantile(a,.95))}

def main():
    r=json.loads((OUT/'results.json').read_text(encoding='utf8'));n=r['freeze']['last_update']
    d=pd.read_csv(OUT/'frozen/training_updates.csv');e=pd.read_csv(OUT/'episodes.csv')
    updates={};discrete=[];env=[];warnings=[]
    for line in (OUT/'frozen/training.log').read_text(encoding='utf8').splitlines():
        m=re.search(r'\[learner\] iter=(\d+) transitions=',line)
        if m:updates[int(m[1])]=line
        m=re.search(r'\[POLICY_DISCRETE\] iter=(\d+)',line)
        if m:
            for head,h,pmax in re.findall(r'(cu_matching_\d+):entropy=([0-9.]+),pmax=([0-9.]+)',line):
                discrete.append({'update':int(m[1]),'head':head,'entropy':float(h),'pmax':float(pmax)})
        m=re.search(r'\[ENV\] iter=(\d+) worker=(\d+).*?collection=([\d.]+)s',line)
        if m:env.append({'update':int(m[1]),'worker':int(m[2]),'collection':float(m[3])})
        if '[WARNING]' in line or '[ERROR]' in line:warnings.append(line)
    pp=pd.DataFrame(discrete);pp.to_csv(OUT/'discrete_policy.csv',index=False)
    ed=pd.DataFrame(env);ed.to_csv(OUT/'worker_collection.csv',index=False)
    d['timestamp']=[pd.Timestamp(updates[u][:19]) for u in d['update']]
    d['cycle_seconds']=d.timestamp.diff().dt.total_seconds()
    d['time_sum']=d.collection_wait_seconds+d.update_seconds+d.logging_seconds
    d['unallocated_seconds']=d.cycle_seconds-d.time_sum
    d.to_csv(OUT/'metrics_with_timestamps.csv',index=False)
    ranges={'U61_last500':(6103,6602),'at8000':(7501,8000),'previous500':(n-999,n-500),'last500':(n-499,n),'last100':(n-99,n)}
    timing={};policy={}
    for name,(a,b) in ranges.items():
        ds=d[d['update'].between(a,b)];ps=pp[pp['update'].between(a,b)]
        timing[name]={k:stats(ds[k]) for k in ['cycle_seconds','collection_wait_seconds','update_seconds','logging_seconds','unallocated_seconds']}
        policy[name]={k:float(ps[k].mean()) for k in ['entropy','pmax']}
        policy[name]['heads']={k:{x:float(v[x].mean()) for x in ['entropy','pmax']} for k,v in ps.groupby('head')}
    # Persist BS values only; all scalar tags are checked for completeness/finiteness.
    tags={};scalars=[]
    event_path=next((OUTER/'runs/beta-hppo/2026-10-19_09-44-15_beta-hppo_sync_result').glob('events*'))
    for event in LegacyEventFileLoader(str(event_path)).Load():
        for v in event.summary.value:
            if not v.HasField('simple_value'):continue
            update=int(event.step)+(1 if v.tag.startswith('BS/') else 0)
            if not 1<=update<=n:continue
            tags.setdefault(v.tag,[]).append((update,float(v.simple_value)))
            if v.tag.startswith('BS/'):scalars.append({'tag':v.tag,'update':update,'value':float(v.simple_value)})
    s=pd.DataFrame(scalars);s.to_csv(OUT/'ppo_scalars.csv',index=False)
    tb={}
    for tag,values in tags.items():
        x=np.asarray(values);finite=np.isfinite(x[:,1]);unique=np.unique(x[:,0])
        item={'n':len(x),'first':int(unique.min()),'last':int(unique.max()),'unique_updates':len(unique),'all_finite':bool(finite.all())}
        if tag.startswith('BS/'):
            item['windows']={name:stats(x[(x[:,0]>=a)&(x[:,0]<=b),1]) for name,(a,b) in ranges.items()} if finite.all() else {}
        tb[tag]=item
    # Current files and run-time metadata both stay bound to the same source/scene.
    with (LOG/'mhbppo_train_env.jsonl').open(encoding='utf8') as f:meta=json.loads(f.readline())
    dump('metadata.json',meta)
    source=[]
    for rel,expected in meta['source_sha256'].items():
        actual=hashlib.sha256((OUTER/rel).read_bytes()).hexdigest()
        source.append({'path':rel,'expected':expected,'actual':actual,'match':actual==expected})
    dump('source_check.json',source)
    prevmeta=json.loads((OUT.parent/'mhbppo_matched_2env_20261020/metadata.json').read_text(encoding='utf8'))
    hyper=json.loads((OUT/'frozen/hyperparameters.json').read_text(encoding='utf8'))
    oldhyper=json.loads((OUT.parent/'mhbppo_episode_reliability_20261020/frozen/hyperparameters.json').read_text(encoding='utf8'))
    checks=dict(r['validation'])
    checks.update({'batch80':bool((d.transitions==80).all()),'saved_versions':bool((d.saved_policy_version==d['update']).all()),
        'all_checksums_ok':all('ck=ok' in updates[u] and 'stale=1 ' in updates[u] for u in range(1,n+1)),
        'source_files_match':all(x['match'] for x in source),'source_matches_U60':meta['source_sha256']==prevmeta['source_sha256'],
        'same_scene':meta['shared_exogenous_scene']==prevmeta['shared_exogenous_scene'],'same_hyperparameters':hyper==oldhyper,
        'BS_all_finite':all(v['all_finite'] for k,v in tb.items() if k.startswith('BS/')),
        'BS_complete':all(v['unique_updates']==n and v['first']==1 and v['last']==n for k,v in tb.items() if k.startswith('BS/'))})
    rewards=pd.read_csv(RUN/'training_rewards_seed_42.csv').iloc[:n]
    checks['reward_second_csv']=bool(np.allclose(d.reward,rewards.reward,atol=1e-13,rtol=0))
    checks['completion_second_csv']=bool(np.allclose(d.completion_rate,rewards.completion_rate,atol=1e-13,rtol=0))
    # Separate csv.DictReader count, independent of pandas group means.
    with (OUT/'episodes.csv').open(newline='',encoding='utf8') as f:rows=list(csv.DictReader(f))
    for key,w in r['windows'].items():
        a,b=w['range'];sel=[x for x in rows if a<=int(x['update'])<=b]
        checks[key+'_independent_counts']=len(sel)==w['episodes'] and sum(int(x['optimal'])==40 for x in sel)==w['all_optimal_n'] and sum(float(x['completion'])==100 for x in sel)==w['all_completion_n']
    result={'freeze':r['freeze'],'timing':timing,'discrete_policy':policy,'tensorboard':tb,'warnings':warnings,
            'latest_solver_counts':{c:int(e.loc[e['update']>n-500,c].sum()) for c in ['optimal','infeasible','unknown','error','other']},
            'checks':checks,'passed':all(checks.values()),'solver_calls':0,'optimizer_steps':0}
    dump('audit.json',result)
    print(json.dumps({'passed':result['passed'],'checks':checks,'timing':timing,'policy':policy,
        'latest_solver_counts':result['latest_solver_counts'],'BS':{k:v for k,v in tb.items() if k.startswith('BS/')},'warnings':warnings},ensure_ascii=False))

if __name__=='__main__':main()
