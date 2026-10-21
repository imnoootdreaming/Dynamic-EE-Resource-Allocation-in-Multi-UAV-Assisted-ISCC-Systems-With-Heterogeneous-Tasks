"""Freeze one historical on-policy batch and its pre-update checkpoint. No model/solver."""
import csv
import hashlib
import json
from pathlib import Path
import shutil

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[5]
OUTER = ROOT / 'src/outer'
RUN = OUTER / 'ckpt/MHBPPO/2026-10-18_17-25-12_mhbppo_seed_42'
LOG = OUTER / 'logs/MHBPPO/2026-10-18_17-25-12'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
OUT.mkdir(exist_ok=True, parents=True)
source = OUT / 'source_frozen'
source.mkdir(exist_ok=True)
shutil.copyfile(RUN/'hyperparameters.json', OUT/'hyperparameters.json')
for name in ('mhbppo_actor.pth', 'mhbppo_critic.pth', 'state_normalization.npz', 'checkpoint_metadata.json'):
    target = source / name
    original = RUN / 'iter_5000' / name
    if target.exists():
        assert sha(target) == sha(original), name
    else:
        shutil.copyfile(original, target)

selected = []
metadata = None
path = LOG / 'mhbppo_train_actions.jsonl'
with path.open('rb') as stream:
    size = stream.seek(0, 2)
    stream.seek(0)
    metadata_line = stream.readline()
    metadata = json.loads(metadata_line)
    for line in stream:
        if stream.tell() > size:
            break
        r = json.loads(line)
        if r.get('record_type') != 'action':
            continue
        iteration = int(r['iteration'])
        if iteration > 5001:
            break
        if iteration == 5001:
            selected.append((r, line))
assert len(selected) == 80, len(selected)
selected.sort(key=lambda pair: (int(pair[0]['worker_id']), int(pair[0]['step_idx'])))
assert [(r['worker_id'], r['step_idx']) for r, _ in selected] == [(w,t) for w in (0,1) for t in range(40)]
assert all(r['weights_version'] == 5000 and r['collection_round'] == 5000 for r,_ in selected)
for key in ('actor_signature', 'norm_signature', 'scene_signature'):
    assert len({r[key] for r,_ in selected}) == 1, key
with (OUT / 'actions_update5001.jsonl').open('wb') as f:
    f.write(metadata_line)
    for r, line in selected:
        f.write(line)
with (RUN/'training_updates.csv').open(encoding='utf-8-sig',newline='') as f:
    reader=csv.DictReader(f)
    fields=reader.fieldnames
    rows=[r for r in reader if int(r['update'])<=5001]
assert [int(r['update']) for r in rows] == list(range(1,5002))
with (OUT/'training_updates_through5001.csv').open('w',encoding='utf-8',newline='') as f:
    writer=csv.DictWriter(f,fields);writer.writeheader();writer.writerows(rows)
matches={name:sha(OUTER/name)==expected for name,expected in metadata['source_sha256'].items()}
assert all(matches.values()), matches
protocol={'scope':'Historical initial full-batch gradient at actor/critic v5000 on update5001 rollout; no optimizer steps',
    'records':80,'workers':2,'slots_per_worker':40,'source_weight_version':5000,'source_path':str(RUN/'iter_5000'),
    'log_opened_size':size,'log_path':str(path),'source_hashes':{p.name:sha(p) for p in source.iterdir()},
    'selected_log_sha256':sha(OUT/'actions_update5001.jsonl'),'csv_sha256':sha(OUT/'training_updates_through5001.csv'),
    'production_source_matches':matches,
    'known_missing':['historical reward-normalizer standard deviation/m2','historical minibatch permutations','Adam moments and intermediate parameter iterates'],
    'intended_partial_results':['exact checkpoint/batch entropy gradients after observation reconstruction validation',
        'full-batch initial surrogate-gradient sensitivity to missing reward standard deviation; clearly labelled, never fabricated as actual'],
    'no_training_no_solver':True}
(OUT/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'records':len(selected),'source_files':len(protocol['source_hashes']),
    'source_matches':sum(matches.values()),'last_update':5001}))
