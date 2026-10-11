import json,collections,pathlib,numpy as np
p=pathlib.Path('analysis/mosek_tuning_20261009')
rs=[json.loads(x) for x in (p/'full/results.jsonl').read_text().splitlines()]
for name in ['baseline','intpntCoTolRelGap_1e-07','joint_1e-07','joint_1e-06','joint_1e-05','threads_1']:
 rr=[r for r in rs if r['config']==name]
 print(name, 'outer_iterations',dict(collections.Counter(r['iterations'] for r in rr)), 'termination_codes',dict(collections.Counter(c for r in rr for c in r['termination_codes'])))
 print('recovered_failures',dict(collections.Counter(k for r in rr for k in r['validation']['recovered_beams']['failed_checks'])))
 print('c08_max_bits',max(max(r['validation']['recovered_beams']['checks']['c08_uav_offloading']['raw_residual']) for r in rr),'positive_rank_gap_max',max(sum(r['validation']['rank']['absolute_gap_w']) for r in rr))
print('num_samples',len(rs),'sample_energy',[(r['sample_id'],r['objective']) for r in rs[:3]])