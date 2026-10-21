"""Reuse accepted U60 step data, append new records and inspect failed-step causes."""
from pathlib import Path
import hashlib,json,os
import numpy as np
import pandas as pd
from analyze import OUT,OUTER,dump

OLD=OUT.parent/'mhbppo_matched_2env_20261020'
LOG=OUTER/'logs/MHBPPO/2026-10-19_09-44-15'

def brief(a):
    a=np.asarray(a,dtype=float)
    return {'n':len(a),'mean':float(a.mean()),'p05':float(np.quantile(a,.05)),
            'median':float(np.median(a)),'p95':float(np.quantile(a,.95)),'min':float(a.min()),'max':float(a.max())}

def matching_exists(m,threshold):
    allow=np.asarray(m)>=threshold;owners={}
    def aug(i,seen):
        for j in np.flatnonzero(allow[i]):
            if int(j) in seen:continue
            seen.add(int(j))
            if int(j) not in owners or aug(owners[int(j)],seen):owners[int(j)]=i;return True
        return False
    return all(aug(i,set()) for i in np.argsort(allow.sum(axis=1)))

def main():
    results=json.loads((OUT/'results.json').read_text(encoding='utf8'))
    n=results['freeze']['last_update'];lo=n-499
    d=pd.read_csv(OLD/'physical_steps.csv',keep_default_na=False)
    old_hi=int(d['update'].max());d=d[d['update'].between(lo,n)].copy()
    new=[];manifest=[]
    for path in sorted(LOG.glob('mhbppo_train_env*.jsonl')):
        with path.open('rb') as f:
            bound=os.fstat(f.fileno()).st_size;selected=0
            while f.tell()<bound:
                offset=f.tell();raw=f.readline()
                if not raw.endswith(b'\n') or f.tell()>bound:break
                # Archived segment ends before old_hi, avoid parsing all old GB.
                if b'"record_type":"env_diagnostic"' not in raw:continue
                import re
                match=re.search(rb'"iteration":(\d+)',raw)
                it=int(match[1]) if match else None
                if it is None or it<=old_hi:continue
                if it>n:break
                v=json.loads(raw);s=v['solver'];sen=v['sensing'];rc=v['reward_components'];val=v['validation']
                new.append({'update':it,'worker':v['worker_id'],'slot':v['time_slot'],'wv':v['weights_version'],
                    'status':s['solver_status_kind'],'success':bool(v['legacy_success']),'reward':v['actual_reward'],
                    'base_reward':v['base_reward'],'Q':rc['sensing_penalty_score'],
                    'solve_seconds':s['solve_seconds'],'diagnostic_seconds':v['diagnostic_seconds'],
                    'cccp_iterations':s['solver_iterations'],'ipm_iterations':s.get('interior_point_iterations'),
                    'sensing_infeasible':sen['current_action_infeasible'],'margin_min':min(sen['margin']),
                    'distance_mean':float(np.mean(sen['target_distances_m'])),'distance_max':max(sen['target_distances_m']),
                    'physical_verified':val['physical_solution_verified'],'cov_pass':val['covariance']['passed'],
                    'beam_pass':val['recovered_beams']['passed'],'outer_pass':v['outer_action']['passed'],
                    'cov_failed':'|'.join(val['covariance']['failed_checks']),'beam_failed':'|'.join(val['recovered_beams']['failed_checks']),
                    'file':path.name,'offset':offset,'raw_sha256':hashlib.sha256(raw).hexdigest()})
                selected+=1
        manifest.append({'file':path.name,'bound':bound,'new_selected':selected})
    d=pd.concat([d,pd.DataFrame(new)],ignore_index=True).sort_values(['update','worker','slot'])
    d['failure']=d.status!='optimal';d.to_csv(OUT/'latest_steps.csv',index=False)
    assert len(d)==40000 and not d.duplicated(['update','worker','slot']).any()
    assert d.groupby(['update','worker']).slot.apply(lambda x:sorted(x)==list(range(40))).all()
    e=pd.read_csv(OUT/'episodes.csv');sel=e[e['update'].between(lo,n)].set_index(['update','worker'])
    agg=d.groupby(['update','worker']).failure.sum()
    assert np.array_equal(agg.to_numpy(),sel.optimal_failures.reindex(agg.index).to_numpy())
    failed=d[d.failure];details=[]
    for name,g in failed.groupby('file'):
        with (LOG/name).open('rb') as f:
            for row in g.sort_values('offset').itertuples():
                f.seek(int(row.offset));raw=f.readline();assert hashlib.sha256(raw).hexdigest()==row.raw_sha256
                v=json.loads(raw);sen=v['sensing'];cand=sen['candidates'];mar=np.asarray(sen['margin'],float)
                details.append({'update':row.update,'worker':row.worker,'slot':row.slot,'status':row.status,
                    'margin_min':float(mar.min()),'failed_uav_count':int((mar<1-1e-6).sum()),
                    'current_power_rematch_possible':cand['current_power_matching'] is not None,
                    'min_power_rematch_possible':cand['min_power_matching'] is not None,
                    'current_power_margin2_matching':matching_exists(cand['current_power_margin'],2),
                    'min_power_margin2_matching':matching_exists(cand['min_power_margin'],2),
                    'failed_uav_ids':'|'.join(map(str,np.flatnonzero(mar<1-1e-6))),
                    'outer_failed':'|'.join(v['outer_action']['failed_checks'])})
    f=pd.DataFrame(details);f.to_csv(OUT/'failed_step_details.csv',index=False)
    one_ids=agg[agg==1].index
    one=failed.set_index(['update','worker']).loc[one_ids].reset_index()
    first=failed.groupby(['update','worker']).slot.min()
    byslot=d.groupby('slot').agg(n=('failure','size'),failures=('failure','sum'),Q=('Q','mean'),margin_median=('margin_min','median'))
    byslot['failure_pct']=byslot.failures/byslot.n*100;byslot.to_csv(OUT/'by_slot.csv')
    # Transition probabilities and run lengths are descriptive, not causal.
    runs=[];pairs=[]
    for _,g in d.groupby(['update','worker']):
        a=g.sort_values('slot').failure.to_numpy();run=0
        for z in a:
            if z:run+=1
            elif run:runs.append(run);run=0
        if run:runs.append(run)
        pairs.extend(zip(a[:-1],a[1:]))
    pairs=np.asarray(pairs)
    success_margin=d.loc[~d.failure,'margin_min']
    r={'range':[lo,n],'records':len(d),'new_records':len(new),'status_counts':d.status.value_counts().to_dict(),
       'failed_sensing_necessary_n':int(failed.sensing_infeasible.sum()),'failed_uavs_hist':{str(k):int(v) for k,v in f.failed_uav_count.value_counts().sort_index().items()},
       'failed_min_margin':brief(failed.margin_min),'successful_min_margin':brief(success_margin),
       'success_margin_below_2_pct':float((success_margin<2).mean()*100),
       'one_failure_episodes':len(one),'one_failure_margin':brief(one.margin_min),
       'failure_slot_groups':{str(a)+'_'+str(b):{'failures':int(failed.slot.between(a,b).sum()),'share_pct':float(failed.slot.between(a,b).mean()*100)} for a,b in [(0,3),(4,7),(8,19),(20,39)]},
       'slot_mod4':{str(k):{'failure_pct':float(g.failure.mean()*100),'failures':int(g.failure.sum()),'n':len(g)} for k,g in d.groupby(d.slot%4)},
       'first_failure_slot':brief(first),'failure_run_length_hist':{str(k):int(v) for k,v in pd.Series(runs).value_counts().sort_index().items()},
       'next_failure_given_failure_pct':float(pairs[pairs[:,0],1].mean()*100),
       'next_failure_given_success_pct':float(pairs[~pairs[:,0],1].mean()*100),
       'candidate_rematching_failure_only':{k:{'n':int(f[k].sum()),'pct':float(f[k].mean()*100)} for k in ['current_power_rematch_possible','min_power_rematch_possible','current_power_margin2_matching','min_power_margin2_matching']},
       'outer_failures_among_nonoptimal':f.outer_failed.value_counts().to_dict(),
       'per_uav_failure_n':{str(i):int(f.failed_uav_ids.str.split('|').apply(lambda a:str(i) in a).sum()) for i in range(4)},
       'checks':{'episode_counts_match_text':True,'no_duplicate_ids':True,'complete_40slots':True,'failed_raw_sha_match':True},
       'source_old_sha':hashlib.sha256((OLD/'physical_steps.csv').read_bytes()).hexdigest(),'input_manifest':manifest,
       'limits':'Rematching is an offline necessary-sensing-condition check at fixed logged positions/receive beams; it neither proves full task feasibility nor a realizable training gain.'}
    dump('failure_summary.json',r);print(json.dumps(r,ensure_ascii=False))

if __name__=='__main__':main()
