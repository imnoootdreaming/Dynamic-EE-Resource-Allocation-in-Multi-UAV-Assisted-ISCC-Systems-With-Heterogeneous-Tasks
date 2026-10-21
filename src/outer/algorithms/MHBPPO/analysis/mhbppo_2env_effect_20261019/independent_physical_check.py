"""Independent arithmetic validation of compact selected records; no solver/model imports."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

p = Path(__file__).resolve().parent
summary = json.loads((p / 'physical_summary.json').read_text(encoding='utf-8'))
z = np.load(p / 'physical_metrics.npz', allow_pickle=False)
r, u = z['rows'], z['uav']
ri = {k: i for i, k in enumerate(summary['schema']['rows'])}
ui = {k: i for i, k in enumerate(summary['schema']['uav'])}
end = 5804
last = r[:, ri['iteration']] >= end - 499
prev = ~last
optimal = r[:, ri['optimal']] > .5
sensing = r[:, ri['sensing_pass']] > .5
slots = r[:, ri['slot']]
check = {'records': len(r), 'last500_records': int(last.sum()),
    'npz_sha256_matches': hashlib.sha256((p/'physical_metrics.npz').read_bytes()).hexdigest() == summary['compact_npz_sha256'],
    'unique_iteration_worker_slot': len(np.unique(r[:, :3], axis=0)) == len(r),
    'all_windows_complete': all(np.count_nonzero(r[:, 0] == k) == 80 for k in range(end-999,end+1)),
    'stored_M_matches_necessary_pass': bool(np.array_equal(np.all(u[:, :, ui['margin']] >= 1-1e-6, axis=1), sensing)),
    'failures_all_violate_necessary_condition': bool(np.all(~sensing[last & ~optimal])),
    'optimal_count_last500': int(optimal[last].sum()),
    'optimal_percent_last500': float(optimal[last].mean()*100),
    'gain_last500': float(u[last, :, ui['gain']].mean()),
    'last32_optimal_percent_previous500': float(optimal[prev & (slots>=8)].mean()*100),
    'last32_optimal_percent_last500': float(optimal[last & (slots>=8)].mean()*100),
    'early8_share_of_optimal_percent': float(np.count_nonzero(last & optimal & (slots<8))/np.count_nonzero(last & optimal)*100),
}
check['slot_groups'] = []
for lo, hi in [(0,3),(4,7),(8,39)]:
    s = last & (slots>=lo) & (slots<=hi)
    check['slot_groups'].append({'slots':[lo,hi], 'samples':int(s.sum()),
        'optimal_percent':float(optimal[s].mean()*100),
        'Q':float(r[s,ri['Q']].mean()), 'distance':float(u[s,:,ui['distance']].mean())})
secs = r[last,ri['solve_seconds']]
check['necessary_failure_solver_seconds_share_percent'] = float(secs[~sensing[last]].sum()/secs.sum()*100)
with (p/'frozen/training_updates.csv').open(encoding='utf-8-sig', newline='') as f:
    c = {int(x['update']):x for x in csv.DictReader(f)}
check['every_selected_update_matches_csv'] = all(
    int(optimal[r[:,0]==k].sum()) == int(c[k]['optimal_count'])
    and abs(float(r[r[:,0]==k,ri['Q']].mean())-float(c[k]['sensing_Q'])) < 1e-12
    and abs(float(r[r[:,0]==k,ri['reward']].mean())-float(c[k]['reward'])) < 1e-12
    for k in range(end-999,end+1))
check['passed'] = all(check[k] for k in ('npz_sha256_matches','unique_iteration_worker_slot',
    'all_windows_complete','stored_M_matches_necessary_pass','failures_all_violate_necessary_condition',
    'every_selected_update_matches_csv')) and check['records']==80000 and check['last500_records']==40000
(p/'root_physical_checks.json').write_text(json.dumps(check,indent=2),encoding='utf-8')
assert check['passed'], check
print(json.dumps(check,indent=2))
