"""Independent CSV checks and a static, reproducible training figure; no training imports."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
CSV = ROOT / 'frozen' / 'training_updates.csv'
with CSV.open(encoding='utf-8-sig', newline='') as f:
    rows = list(csv.DictReader(f))
n = len(rows)
updates = np.array([int(r['update']) for r in rows])
transitions = np.array([int(r['transitions']) for r in rows])
optimal = np.array([int(r['optimal_count']) for r in rows])
completion = np.array([float(r['completion_rate']) for r in rows])
reward = np.array([float(r['reward']) for r in rows])
base = np.array([float(r['base_reward']) for r in rows])
q = np.array([float(r['sensing_Q']) for r in rows])
checks = {
    'csv_sha256': hashlib.sha256(CSV.read_bytes()).hexdigest(),
    'updates': n,
    'env_steps': int(transitions.sum()),
    'contiguous_updates': bool(np.array_equal(updates, np.arange(1, n + 1))),
    'exact_80_transitions_per_update': bool(np.all(transitions == 80)),
    'env_step_counters': all(int(r['env_steps']) == 80 * int(r['update']) for r in rows),
    'episode_counters': all(int(r['episodes']) == 2 * int(r['update']) for r in rows),
    'policy_version_chain': all(int(r['collection_policy_version']) == int(r['update']) - 1
                                and int(r['saved_policy_version']) == int(r['update']) for r in rows),
    'reward_decomposition_max_error': float(np.max(np.abs(reward - base + q))),
    'finite_reward_q': bool(np.isfinite(reward).all() and np.isfinite(q).all()),
}
checks['passed'] = all(checks[k] for k in ('contiguous_updates', 'exact_80_transitions_per_update',
    'env_step_counters', 'episode_counters', 'policy_version_chain', 'finite_reward_q')) and checks['reward_decomposition_max_error'] < 1e-12
windows = []
for lo, hi in [(1, 500), (501, 1000), (n-999,n-500), (n-499,n), (n-99,n)]:
    sel = (updates >= lo) & (updates <= hi)
    windows.append({'first': lo, 'last': hi, 'steps': int(transitions[sel].sum()),
        'optimal_count': int(optimal[sel].sum()),
        'optimal_percent': float(optimal[sel].sum()/transitions[sel].sum()*100),
        'completion_percent': float(completion[sel].mean()), 'reward': float(reward[sel].mean()),
        'base_reward': float(base[sel].mean()), 'Q': float(q[sel].mean())})
checks['independent_windows'] = windows
(ROOT / 'root_csv_checks.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
assert checks['passed'], checks

def rolling(values, width=250):
    return np.convolve(values, np.ones(width)/width, mode='valid')

plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(3, 1, figsize=(10, 8.6), sharex=True, constrained_layout=True)
x = updates[249:]
axes[0].plot(x, rolling(optimal / transitions * 100), label='Solver optimal', color='#0072B2')
axes[0].plot(x, rolling(completion), label='Task completion', color='#D55E00', ls='--')
axes[0].set_ylabel('Rate (%)')
axes[0].legend(loc='upper left')
axes[0].set_title(f'Synchronous 2-env training, seed 42 / scene offset 0 | frozen at update {n:,}')
axes[1].plot(x, rolling(reward), label='Total reward', color='#0072B2')
axes[1].plot(x, rolling(base), label='Base reward', color='#009E73')
axes[1].set_ylabel('Mean reward')
axes[1].legend(loc='lower right')
axes[2].plot(x, rolling(q), color='#CC79A7', label='Sensing shortfall Q (lower is better)')
axes[2].set_ylabel('Mean Q')
axes[2].set_xlabel('Completed PPO updates (80 new transitions per update)')
axes[2].legend(loc='upper right')
for ax in axes:
    ax.grid(alpha=.2)
    ax.axvspan(n - 499, n, color='#E69F00', alpha=.10)
fig.suptitle('All curves use 250-update trailing means; shaded region: last 500 updates', fontsize=10)
fig.savefig(ROOT / 'training_progress.png', dpi=170)
fig.savefig(ROOT / 'training_progress.svg')
plt.close(fig)
print(json.dumps({'passed':checks['passed'], 'updates':n, 'env_steps':checks['env_steps'], 'figure':'training_progress.png'}))
