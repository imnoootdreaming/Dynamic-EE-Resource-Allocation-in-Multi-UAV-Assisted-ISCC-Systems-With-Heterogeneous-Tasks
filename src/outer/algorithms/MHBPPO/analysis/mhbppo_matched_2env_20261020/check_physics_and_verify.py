"""Resolve diagnostic acceptance semantics and independently check frozen statistics."""
from pathlib import Path
import collections
import hashlib
import json
import numpy as np
import pandas as pd
from analyze import OUT, OUTER, LOG, FROZEN, write, stats

def enrich():
    d=pd.read_csv(OUT/'physical_steps.csv')
    rows=[]; examples=[]
    for name,group in d.groupby('file'):
        with (LOG/name).open('rb') as f:
            for row in group.sort_values('offset').itertuples():
                f.seek(int(row.offset)); raw=f.readline()
                assert hashlib.sha256(raw).hexdigest()==row.raw_sha256
                v=json.loads(raw);val=v['validation']; rank=val['rank']
                rc=val['recovered_beams'];cov=val['covariance']
                item={'update':row.update,'worker':row.worker,'slot':row.slot,'success':bool(row.success),
                      'rank_pass':bool(rank['passed']), 'rank_evaluated':bool(rank['evaluated']),
                      'absolute_rank_gap':float(sum(rank.get('absolute_gap_w',[]))),
                      'max_relative_rank_gap':float(max(rank.get('relative_gap',[0]))),
                      'beam_evaluated':bool(rc['evaluated']), 'beam_pass':bool(rc['passed']),
                      'cov_evaluated':bool(cov['evaluated']), 'cov_pass':bool(cov['passed']),
                      'outer_pass':bool(v['outer_action']['passed']), 'solver_converged':bool(v['solver']['solver_converged']),
                      'energy':float(v['legacy_energy']) if np.isfinite(float(v['legacy_energy'])) else None}
                for key,section in [('beam',rc),('cov',cov)]:
                    c=section.get('checks',{}).get('c08_uav_offloading',{})
                    item[key+'_c08_failed']=not c.get('passed',True)
                    item[key+'_c08_relative']=float(c.get('max_violation',0))
                    item[key+'_c08_deficit_bits']=max([0]+list(map(float,c.get('raw_residual',[]))))
                if len(examples)<4 and (item['beam_c08_failed'] or (item['beam_pass'] and not item['rank_pass'])):
                    examples.append({'id':[row.update,row.worker,row.slot], 'rank':rank,
                                     'beam_c08':rc.get('checks',{}).get('c08_uav_offloading'), 'tolerances':val['tolerances']})
                rows.append(item)
    x=pd.DataFrame(rows);x.to_csv(OUT/'physical_acceptance.csv',index=False)
    r={'n':len(x),'rank_evaluated':int(x.rank_evaluated.sum()),'rank_pass':int(x.rank_pass.sum()),
       'both_cov_and_recovered_pass':int((x.cov_pass & x.beam_pass).sum()),
       'beam_c08_failed':int(x.beam_c08_failed.sum()),
       'beam_c08_fail_deficit_bits':stats(x.loc[x.beam_c08_failed,'beam_c08_deficit_bits']),
       'beam_c08_fail_relative':stats(x.loc[x.beam_c08_failed,'beam_c08_relative']),
       'beam_c08_failed_among_legacy_success':int((x.beam_c08_failed & x.success).sum()),
       'recovered_and_outer_pass':int((x.beam_pass & x.outer_pass).sum()),
       'successful_and_recovered_pass':int((x.beam_pass & x.success).sum()),
       'absolute_rank_gap_returned':stats(x.loc[x.rank_evaluated,'absolute_rank_gap']),
       'relative_rank_gap_returned':stats(x.loc[x.rank_evaluated,'max_relative_rank_gap']),
       'converged_returned_rate':float(x.loc[x.beam_evaluated,'solver_converged'].mean()),
       'energy_on_success':stats(x.loc[x.success,'energy']),
       'examples':examples,
       'note':'Composite requires covariance, recovered-beam original constraints, AND all relative/total absolute rank gaps. It is stricter than recovered-beam feasibility alone.'}
    write('physical_acceptance_summary.json',r)
    print(json.dumps({k:v for k,v in r.items() if k!='examples'},ensure_ascii=False))

def verify():
    import csv
    with (FROZEN/'training_updates.csv').open(newline='',encoding='utf-8') as f: rows=list(csv.DictReader(f))
    s=json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
    checks={}
    for name,bounds in [('first500',(1,500)),('previous500',(5494,5993)),('last500',(5994,6493)),('last100',(6394,6493))]:
        part=[r for r in rows if bounds[0]<=int(r['update'])<=bounds[1]]
        for col,key in [('completion_rate','completion'),('reward','reward'),('sensing_Q','Q')]:
            mean=sum(float(r[col]) for r in part)/len(part)
            checks[name+'_'+col]=abs(mean-s[name][key]['mean'])<1e-12
        opt=sum(int(r['optimal_count']) for r in part)/sum(int(r['transitions']) for r in part)*100
        checks[name+'_optimal']=abs(opt-s[name]['optimal_rate'])<1e-12
    p=pd.read_csv(OUT/'physical_steps.csv')
    for col,expected in [('success',37863),('physical_verified',6),('cov_pass',28909),('beam_pass',35128)]:
        checks['physical_'+col]=int(p[col].sum())==expected
    checks['solver_time_within_episode']=bool((pd.read_csv(OUT/'worker_time_decomposition.csv').remaining_seconds>=0).all())
    all_checks=[json.loads((OUT/f).read_text()) for f in ['core_validation.json','physical_validation.json']]
    checks['previous_checks']=all(all(d.values()) for d in all_checks)
    checks['ppo_finite']=all(v['all_finite'] for k,v in json.loads((OUT/'tensorboard_summary.json').read_text()).items() if k.startswith('BS/'))
    # Freeze file signatures and verify the production sources remain at run versions.
    md=json.loads((OUT/'metadata.json').read_text(encoding='utf-8'))
    checks['production_sources_unchanged']=all(hashlib.sha256((OUTER/k).read_bytes()).hexdigest()==v for k,v in md['source_sha256'].items())
    manifest={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.glob('frozen/*')) if p.is_file()}
    write('frozen_manifest.json',manifest)
    write('final_validation.json',{'passed':all(checks.values()),'checks':checks,'solver_calls':0,'policy_evaluations':0,'training_changes':0})
    print(json.dumps({'passed':all(checks.values()),'checks':checks}))

if __name__=='__main__':
    import sys
    verify() if len(sys.argv)>1 and sys.argv[1]=='verify' else enrich()
