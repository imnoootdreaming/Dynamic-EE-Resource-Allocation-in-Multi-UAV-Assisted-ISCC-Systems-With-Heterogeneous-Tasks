"""Read-only analysis. No training/environment/solver imports or calls."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
from pathlib import Path
from datetime import datetime, timezone, timedelta
import ast
import hashlib
import io
import json
import re
import sys
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[5]
OUTER = ROOT / 'src/outer'
RUN = OUTER / 'ckpt/MHBPPO/2026-10-19_09-44-15_mhbppo_seed_42'
LOG = OUTER / 'logs/MHBPPO/2026-10-19_09-44-15'
FROZEN = OUT / 'frozen'
FROZEN.mkdir(exist_ok=True)

def write(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def bounded_read(path):
    with path.open('rb') as f:
        bound = os.fstat(f.fileno()).st_size
        data = f.read(bound)
    if not data.endswith(b'\n'):
        data = data[:data.rfind(b'\n') + 1]
    return data

def stats(a):
    a = np.asarray(a, dtype=float)
    return {'n': len(a), 'mean': float(a.mean()), 'median': float(np.median(a)),
            'p05': float(np.percentile(a, 5)), 'p95': float(np.percentile(a, 95)),
            'min': float(a.min()), 'max': float(a.max())}

def core():
    raw = bounded_read(RUN / 'training_updates.csv')
    df = pd.read_csv(io.BytesIO(raw))
    reward_raw = bounded_read(RUN / 'training_rewards_seed_42.csv')
    rewards = pd.read_csv(io.BytesIO(reward_raw))
    text_raw = bounded_read(LOG / 'mhbppo_train.log')
    text = text_raw.decode('utf-8-sig')
    lines = text.splitlines()
    update_lines = {}
    envs = []
    for line in lines:
        m = re.search(r'\[learner\] iter=(\d+) transitions=', line)
        if m:
            update_lines[int(m[1])] = line
        m = re.search(r'\[ENV\] iter=(\d+) worker=(\d+) episode=(\d+) transitions=(\d+) wv=(\d+) reward=([^ ]+) completion=([^%]+)% solver_counts=(\{.*\}) collection=([^s]+)s', line)
        if m:
            counts = ast.literal_eval(m[8])
            envs.append({'update': int(m[1]), 'worker': int(m[2]), 'episode': int(m[3]),
                         'transitions': int(m[4]), 'wv': int(m[5]), 'reward': float(m[6]),
                         'completion': float(m[7]), 'collection': float(m[9]), **counts})
    edf = pd.DataFrame(envs)
    complete_envs = edf.groupby('update').size()
    n = min(int(df['update'].max()), len(rewards), max(update_lines), int(complete_envs[complete_envs == 2].index.max()))
    df = df[df['update'] <= n].copy()
    rewards = rewards.iloc[:n].copy()
    edf = edf[edf['update'] <= n].copy()
    df.to_csv(FROZEN / 'training_updates.csv', index=False)
    rewards.to_csv(FROZEN / 'training_rewards_seed_42.csv', index=False)
    edf.to_csv(OUT / 'worker_episodes.csv', index=False)
    kept = []
    for line in lines:
        m = re.search(r'\biter=(\d+)', line)
        if m is None or int(m[1]) <= n:
            kept.append(line)
    (FROZEN / 'mhbppo_train.log').write_text('\n'.join(kept) + '\n', encoding='utf-8')
    (FROZEN / 'hyperparameters.json').write_bytes((RUN / 'hyperparameters.json').read_bytes())
    metadata = json.loads((LOG / 'mhbppo_train_env.jsonl').open(encoding='utf-8').readline())
    write('metadata.json', metadata)
    source_check = []
    for rel, expected in metadata['source_sha256'].items():
        p = OUTER / rel
        actual = hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
        source_check.append({'path': rel, 'expected': expected, 'actual': actual, 'match': actual == expected})
    write('source_check.json', source_check)
    ts = {k: pd.Timestamp(update_lines[k][:19]) for k in update_lines if k <= n}
    df['timestamp'] = [ts[i] for i in df['update']]
    df['cycle_seconds'] = df['timestamp'].diff().dt.total_seconds()
    piv = edf.pivot(index='update', columns='worker', values='collection')
    df['workers_sum'] = piv.sum(axis=1).reindex(df['update']).to_numpy()
    df['workers_mean'] = piv.mean(axis=1).reindex(df['update']).to_numpy()
    df['workers_gap'] = (piv.max(axis=1) - piv.min(axis=1)).reindex(df['update']).to_numpy()
    df['measured_sum'] = df['collection_wait_seconds'] + df['update_seconds'] + df['logging_seconds']
    df['unallocated_seconds'] = df['cycle_seconds'] - df['measured_sum']
    summary = {}
    windows = [('first500', 1, 500), ('501_1000', 501, 1000), ('2001_2500', 2001, 2500),
               ('previous500', n-999, n-500), ('last500', n-499, n), ('last100', n-99, n), ('all', 1, n)]
    for name, lo, hi in windows:
        sub = df[df['update'].between(lo, hi)]
        e = edf[edf['update'].between(lo, hi)]
        # Boundary timestamps include exactly the cycle preceding each included update.
        wall = (ts[hi] - ts[lo-1]).total_seconds() if lo > 1 else (ts[hi] - pd.Timestamp(metadata['started_at']).tz_localize(None)).total_seconds()
        summary[name] = {'range': [lo, hi], 'updates': len(sub), 'steps': int(sub['transitions'].sum()),
                         'wall_seconds': wall, 'steps_per_second': float(sub['transitions'].sum()/wall),
                         'optimal_rate': float(100*sub['optimal_count'].sum()/sub['transitions'].sum()),
                         'completion': stats(sub['completion_rate']), 'reward': stats(sub['reward']),
                         'base_reward': stats(sub['base_reward']), 'Q': stats(sub['sensing_Q']),
                         'solver_counts': {k: int(e[k].sum()) for k in ['optimal','infeasible','unknown','error','other']},
                         'worker': {str(w): {'completion': float(x.completion.mean()), 'collection': stats(x.collection)} for w,x in e.groupby('worker')},
                         'timing': {k: stats(sub[k].dropna()) for k in ['cycle_seconds','collection_wait_seconds','worker_collection_seconds_max','update_seconds','logging_seconds','workers_sum','workers_mean','workers_gap','unallocated_seconds']}}
    for span in [100, 500]:
        roll = df['completion_rate'].rolling(span).mean()
        k = int(roll.idxmax())
        summary['best'+str(span)] = {'end': int(df.loc[k, 'update']), 'completion': float(roll[k])}
    summary['blocks500'] = [{'range': [int(x['update'].min()), int(x['update'].max())],
                             'completion': float(x['completion_rate'].mean()), 'reward': float(x.reward.mean()),
                             'Q': float(x.sensing_Q.mean()), 'wait': float(x.collection_wait_seconds.mean()),
                             'ppo': float(x.update_seconds.mean())} for _,x in df.groupby((df['update']-1)//500)]
    summary['freeze'] = {'time': datetime.now(timezone(timedelta(hours=8))).isoformat(), 'update': n,
                         'last_update_time': str(ts[n]), 'env_steps': int(df.env_steps.iloc[-1]),
                         'episodes': int(df.episodes.iloc[-1]), 'log_bytes_read': len(text_raw),
                         'csv_bytes_read': len(raw), 'sources_match': all(x['match'] for x in source_check)}
    checks = {'update_sequence': df['update'].tolist() == list(range(1,n+1)),
              'env_steps': bool((df.env_steps == df['update']*80).all()),
              'episodes': bool((df.episodes == df['update']*2).all()),
              'all_batches_80': bool((df.transitions == 80).all()),
              'collection_version': bool((df.collection_policy_version == df['update']-1).all()),
              'saved_version': bool((df.saved_policy_version == df['update']).all()),
              'reward_csv': bool(np.allclose(df.reward, rewards.reward, atol=1e-13,rtol=0)),
              'completion_csv': bool(np.allclose(df.completion_rate, rewards.completion_rate,atol=1e-13,rtol=0)),
              'two_workers_each': edf.groupby('update').worker.apply(lambda s: sorted(s)==[0,1]).all().item(),
              'worker_wv': bool((edf.wv == edf['update']-1).all()),
              'worker_slots_40': bool((edf.transitions == 40).all()),
              'worker_optimal_sum': bool((edf.groupby('update').optimal.sum().to_numpy() == df.optimal_count).all()),
              'all_checksums_ok': all('ck=ok' in update_lines[k] and 'stale=1 ' in update_lines[k] for k in range(1,n+1)),
              'sources_match': all(x['match'] for x in source_check)}
    write('core_validation.json', checks)
    write('summary.json', summary)
    df.to_csv(OUT / 'metrics_with_timestamps.csv', index=False)
    print(json.dumps({'freeze': summary['freeze'], 'checks': checks, 'windows': {k:{f:v[f] for f in ['range','optimal_rate','completion','reward','Q','steps_per_second']} for k,v in summary.items() if k in ['first500','previous500','last500','last100']}},ensure_ascii=False))

def physical():
    df = pd.read_csv(FROZEN / 'training_updates.csv')
    n = int(df['update'].max()); lo = n-499
    rows = []; manifests = []
    for p in sorted(LOG.glob('mhbppo_train_env*.jsonl')):
        with p.open('rb') as f:
            bound = os.fstat(f.fileno()).st_size
            first = json.loads(f.readline())
            first_iteration = None; count = 0
            while f.tell() < bound:
                offset = f.tell(); raw = f.readline()
                if not raw.endswith(b'\n') or f.tell() > bound: break
                v = json.loads(raw)
                if v.get('record_type') != 'env_diagnostic': continue
                it = v['iteration']
                if first_iteration is None: first_iteration = it
                if it > n: break
                if it < lo: continue
                s = v['solver']; sense=v['sensing']; val=v['validation']; rc=v['reward_components']
                slots = int(v['time_slot'])
                row={'update': it, 'worker': v['worker_id'], 'slot': slots,
                     'wv': v['weights_version'], 'status': s['solver_status_kind'],
                     'success': bool(v['legacy_success']), 'reward': float(v['actual_reward']),
                     'base_reward': float(v['base_reward']), 'Q': float(rc['sensing_penalty_score']),
                     'solve_seconds': float(s['solve_seconds']), 'diagnostic_seconds': float(v['diagnostic_seconds']),
                     'cccp_iterations': int(s['solver_iterations']), 'ipm_iterations': s.get('interior_point_iterations'),
                     'sensing_infeasible': bool(sense['current_action_infeasible']),
                     'margin_min': min(map(float,sense['margin'])),
                     'distance_mean': float(np.mean(sense['target_distances_m'])),
                     'distance_max': max(map(float,sense['target_distances_m'])),
                     'physical_verified': bool(val['physical_solution_verified']),
                     'cov_pass': bool(val['covariance'].get('passed',False)),
                     'beam_pass': bool(val['recovered_beams'].get('passed',False)),
                     'outer_pass': bool(v['outer_action'].get('passed',False)),
                     'cov_failed': '|'.join(val['covariance'].get('failed_checks',[])),
                     'beam_failed': '|'.join(val['recovered_beams'].get('failed_checks',[])),
                     'file': p.name, 'offset': offset, 'raw_sha256': hashlib.sha256(raw).hexdigest()}
                rows.append(row); count+=1
        manifests.append({'path': str(p), 'input_handle_bound':bound,'first_iteration': first_iteration,'selected':count})
    phy = pd.DataFrame(rows).sort_values(['update','worker','slot'])
    phy.to_csv(OUT/'physical_steps.csv',index=False)
    checks={'rows_40000':len(phy)==40000, 'unique_ids':not phy.duplicated(['update','worker','slot']).any(),
            'full_episodes':phy.groupby(['update','worker']).slot.apply(lambda s: sorted(s)==list(range(40))).all().item(),
            'reward_recomputed':bool(np.allclose(phy.groupby('update').reward.mean(),df[df['update'].between(lo,n)].reward,atol=1e-12,rtol=0)),
            'Q_recomputed':bool(np.allclose(phy.groupby('update').Q.mean(),df[df['update'].between(lo,n)].sensing_Q,atol=1e-12,rtol=0)),
            'completion_recomputed':bool(np.allclose(phy.groupby('update').success.mean()*100,df[df['update'].between(lo,n)].completion_rate,atol=1e-12,rtol=0)),
            'versions':bool((phy.wv==phy['update']-1).all())}
    def group_summary(x):
        return {'n':len(x),'completion':float(100*x.success.mean()),'physical_verified':float(100*x.physical_verified.mean()),
                'cov_pass':float(100*x.cov_pass.mean()),'beam_pass':float(100*x.beam_pass.mean()),
                'Q':float(x.Q.mean()),'distance_mean':float(x.distance_mean.mean()),
                'solve_seconds':stats(x.solve_seconds),'diagnostic_seconds':stats(x.diagnostic_seconds),
                'cccp_iterations':stats(x.cccp_iterations),'sensing_infeasible_rate':float(100*x.sensing_infeasible.mean())}
    result={'range':[lo,n],'all':group_summary(phy),'by_status':{k:group_summary(x) for k,x in phy.groupby('status')},
            'by_slot_group':{name:group_summary(phy[phy.slot.between(a,b)]) for name,a,b in [('0_3',0,3),('4_7',4,7),('8_39',8,39)]},
            'by_slot':{str(k):group_summary(x) for k,x in phy.groupby('slot')},
            'by_worker':{str(k):group_summary(x) for k,x in phy.groupby('worker')},
            'cccp_histogram':{str(k):int(v) for k,v in phy.cccp_iterations.value_counts().sort_index().items()},
            'validation_failure_counts':{'cov':phy.cov_failed.value_counts().to_dict(),'beam':phy.beam_failed.value_counts().to_dict()},
            'successful_but_not_physically_verified':int((phy.success & ~phy.physical_verified).sum()),
            'input_manifests':manifests}
    sums=phy.groupby(['update','worker'])[['solve_seconds','diagnostic_seconds']].sum()
    edf=pd.read_csv(OUT/'worker_episodes.csv').set_index(['update','worker'])
    sums['collection_seconds']=edf.collection.reindex(sums.index)
    sums['remaining_seconds']=sums.collection_seconds-sums.solve_seconds-sums.diagnostic_seconds
    sums.to_csv(OUT/'worker_time_decomposition.csv')
    result['episode_time']={k:stats(sums[k]) for k in sums.columns}
    write('physical_validation.json',checks);write('physical_summary.json',result)
    print(json.dumps({'checks':checks,'range':result['range'],'all':result['all'],'episode_time':result['episode_time']},ensure_ascii=False))

if __name__ == '__main__':
    if len(sys.argv)>1 and sys.argv[1]=='physical': physical()
    else: core()
