"""Read last 500 updates' diagnostic records; no environment/solver calls."""
from pathlib import Path
from collections import Counter
import hashlib, itertools, json, os, re
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent;OUTER=OUT.parents[3]
LOG=OUTER/'logs/MHBPPO/2026-10-19_09-44-15'

def main():
    result=json.loads((OUT/'results.json').read_text(encoding='utf8'));n=result['freeze']['last_update'];lo=n-499
    assignments=np.asarray(list(itertools.permutations(range(10),4)));uavs=np.arange(4)[None,:]
    rows=[];failed=[];sources=Counter();manifest=[];matching_checks=[]
    with (OUT/'frozen/failed_diagnostics.jsonl').open('wb') as snapshot:
        for path in sorted(LOG.glob('mhbppo_train_env*.jsonl')):
            with path.open('rb') as stream:
                bound=os.fstat(stream.fileno()).st_size;selected=0;first=None
                while stream.tell()<bound:
                    offset=stream.tell();raw=stream.readline()
                    if not raw.endswith(b'\n') or stream.tell()>bound:break
                    if b'"record_type":"env_diagnostic"' not in raw:continue
                    match=re.search(rb'"iteration":(\d+)',raw)
                    if not match:continue
                    update=int(match[1]);first=update if first is None else first
                    if update<lo:continue
                    if update>n:break
                    v=json.loads(raw);s=v['solver'];sense=v['sensing'];val=v['validation'];rc=v['reward_components']
                    row={'update':update,'worker':v['worker_id'],'slot':v['time_slot'],'wv':v['weights_version'],
                         'status':s['solver_status_kind'],'success':v['legacy_success'],'reward':v['actual_reward'],
                         'Q':rc['sensing_penalty_score'],'sensing_infeasible':sense['current_action_infeasible'],
                         'margin_min':min(sense['margin']),'distance_mean':float(np.mean(sense['target_distances_m'])),
                         'solve_seconds':s['solve_seconds'],'cccp_iterations':s['solver_iterations'],
                         'beam_pass':val['recovered_beams']['passed'],'outer_pass':v['outer_action']['passed'],
                         'beam_failed':'|'.join(val['recovered_beams']['failed_checks']),
                         'outer_failed':'|'.join(v['outer_action']['failed_checks']),
                         'file':path.name,'offset':offset,'raw_sha256':hashlib.sha256(raw).hexdigest()}
                    rows.append(row);selected+=1
                    if s['solver_status_kind']=='optimal':continue
                    snapshot.write(raw)
                    sources[(s['solver_status_kind'],s['raw_solver_status_kind'],s['status_source'])]+=1
                    cand=sense['candidates'];mar=np.asarray(sense['margin'])
                    f={k:row[k] for k in ['update','worker','slot','status','margin_min','outer_failed']}
                    f['failed_uavs']=int((mar<1-1e-6).sum())
                    for power in ['current_power','min_power']:
                        m=np.asarray(cand[power+'_margin'])
                        for th,label in [(1+1e-6,'matching'),(2,'margin2_matching')]:
                            exists=bool((m[uavs,assignments]>=th).all(axis=1).any())
                            f[power+'_'+label]=exists
                            if label=='matching':matching_checks.append(exists==(cand[power+'_matching'] is not None))
                    failed.append(f)
            manifest.append({'file':path.name,'handle_bound':bound,'first_update':first,'selected':selected})
    d=pd.DataFrame(rows).sort_values(['update','worker','slot']);f=pd.DataFrame(failed)
    d['failure']=d.status!='optimal'
    d.to_csv(OUT/'latest_steps.csv',index=False);f.to_csv(OUT/'failed_step_details.csv',index=False)
    csv=pd.read_csv(OUT/'frozen/training_updates.csv');csv=csv[csv['update'].between(lo,n)].set_index('update')
    ep=pd.read_csv(OUT/'episodes.csv');ep=ep[ep['update'].between(lo,n)].set_index(['update','worker'])
    grouped=d.groupby(['update','worker']);counts=grouped.failure.sum();per_update=d.groupby('update')
    checks={'rows40000':len(d)==40000,'unique':not d.duplicated(['update','worker','slot']).any(),
            'complete40':bool(grouped.slot.apply(lambda x:sorted(x)==list(range(40))).all()),
            'episode_failure_counts':bool(np.array_equal(counts,ep.optimal_failures.reindex(counts.index))),
            'all_completion_count':int(grouped.success.all().sum())==result['windows']['last500']['all_completion_n'],
            'reward_matches':bool(np.allclose(per_update.reward.mean(),csv.reward,atol=1e-12,rtol=0)),
            'Q_matches':bool(np.allclose(per_update.Q.mean(),csv.sensing_Q,atol=1e-12,rtol=0)),
            'completion_matches':bool(np.allclose(per_update.success.mean()*100,csv.completion_rate,atol=1e-12,rtol=0)),
            'versions':bool((d.wv==d['update']-1).all()),'candidate_match_exhaustive':all(matching_checks)}
    assert all(checks.values()),checks
    byslot=d.groupby('slot').agg(failure_rate=('failure','mean'),completion=('success','mean'),mean_Q=('Q','mean'),distance_mean=('distance_mean','mean'))
    byslot.to_csv(OUT/'by_slot.csv')
    summary={'range':[lo,n],'records':len(d),'solver_counts':d.status.value_counts().to_dict(),
        'status_sources':[{'resolved':k[0],'raw':k[1],'source':k[2],'n':v} for k,v in sources.items()],
        'sensing_failure_n':int(d.sensing_infeasible.sum()),'failed_uav_hist':{str(k):int(v) for k,v in f.failed_uavs.value_counts().sort_index().items()},
        'failed_margin_median':float(f.loc[f.status=='infeasible','margin_min'].median()),
        'rematching':{k:{'n':int(f[k].sum()),'pct':float(f[k].mean()*100)} for k in f.columns if k.endswith('matching')},
        'failure_slot_groups':{str(a)+'_'+str(b):{'n':int(d.loc[d.slot.between(a,b),'failure'].sum()),
            'failure_rate_pct':float(d.loc[d.slot.between(a,b),'failure'].mean()*100),
            'share_of_failures_pct':float(d.loc[d.slot.between(a,b),'failure'].sum()/d.failure.sum()*100)} for a,b in [(0,3),(4,7),(8,19),(20,39)]},
        'outer_failure_nonoptimal':f.outer_failed.value_counts().to_dict(),
        'beam_pass_pct':float(d.beam_pass.mean()*100),'beam_and_outer_pass_pct':float((d.beam_pass&d.outer_pass).mean()*100),
        'beam_failure_types':d.beam_failed.value_counts().to_dict(),
        'legacy_success_beam_failed_n':int((d.success&~d.beam_pass).sum()),
        'checks':checks,'matching_checks':len(matching_checks),'manifest':manifest,
        'limits':'Candidate matching only satisfies the sensing necessary condition at a recorded state. No full-task counterfactual solver was run.'}
    (OUT/'step_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':main()
