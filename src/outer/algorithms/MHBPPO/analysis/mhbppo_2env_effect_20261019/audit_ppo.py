"""Read-only training/source audit. Never imports training or solver modules."""
import csv, hashlib, json, math, re, struct
from collections import defaultdict
from pathlib import Path
import numpy as np
from tensorboard.compat.proto.event_pb2 import Event

ROOT=Path(__file__).resolve().parents[6]
OUT=Path(__file__).resolve().parent
OUTER=ROOT/'src/outer'
RUN=OUTER/'ckpt/MHBPPO/2026-10-18_17-25-12_mhbppo_seed_42'
LOG=OUTER/'logs/MHBPPO/2026-10-18_17-25-12'
UP=list(csv.DictReader((OUT/'frozen/training_updates.csv').open(encoding='utf-8')))
END=int(UP[-1]['update'])
sha=lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
hyper=json.loads((OUT/'frozen/hyperparameters.json').read_text(encoding='utf-8'))
source={'frozen_end_update':END,'source_scope':'recorded source provenance hashes, current-byte comparisons; no training or solver import',
        'hyperparameters_sha256':sha(OUT/'frozen/hyperparameters.json'),'checkpoint_checks':{},'recorded_metadata':[]}
for p in sorted(RUN.glob('iter_*/checkpoint_metadata.json')):
 m=json.loads(p.read_text(encoding='utf-8'))
 if m['iteration']>END: continue
 checks={name:{'recorded':h,'current':sha(OUTER/name),'match':h==sha(OUTER/name)} for name,h in m['source_sha256'].items()}
 filechecks={n:{'recorded':m[f'{n}_sha256'],'current':sha(p.parent/f'mhbppo_{n}.pth')} for n in ['actor','critic']}
 for z in filechecks.values(): z['match']=z['recorded']==z['current']
 with np.load(p.parent/'state_normalization.npz',allow_pickle=False) as norm:
  norminfo={k:(norm[k].tolist() if norm[k].size<5 else {'shape':list(norm[k].shape)}) for k in norm.files}
 source['checkpoint_checks'][str(m['iteration'])]={'source_hashes':checks,'all_source_match':all(x['match'] for x in checks.values()),'weight_file_hashes':filechecks,'normalization_fields':norminfo,
   'weights_version':m['weights_version'],'collection_policy_version':m['collection_policy_version'],'cumulative_env_steps':m['cumulative_env_steps'],
   'normalization_actor_sha_match':norminfo['actor_sha256']==m['actor_sha256'],
   'normalization_count_expected':m['cumulative_env_steps']*2+1e-8,
   'normalization_count_match':abs(norminfo['count']-(m['cumulative_env_steps']*2+1e-8))<1e-6,
   'normalization_version_match':norminfo['iteration']==norminfo['weights_version']==m['iteration']==m['weights_version']}
for p in sorted(LOG.glob('mhbppo_train_actions*.jsonl')):
 with p.open(encoding='utf-8') as f: m=json.loads(f.readline())
 mchecks={name:{'recorded':h,'current':sha(OUTER/name),'match':h==sha(OUTER/name)} for name,h in m['source_sha256'].items()}
 cfgdiff={k:{'hyperparameters':v,'metadata':m['madrl_args'].get(k)} for k,v in hyper['hyperparameters'].items() if m['madrl_args'].get(k)!=v}
 source['recorded_metadata'].append({'path':str(p.relative_to(ROOT)),'metadata':m,'source_hashes':mchecks,
   'all_source_match':all(x['match'] for x in mchecks.values()),'configuration_differences':cfgdiff})
source['all_saved_source_match']=all(c['all_source_match'] for c in source['checkpoint_checks'].values())
(OUT/'source_checks.json').write_text(json.dumps(source,ensure_ascii=False,indent=2),encoding='utf-8')

series=defaultdict(dict)
tb=next((OUTER/'runs/beta-hppo/2026-10-18_17-25-12_beta-hppo_sync_result').glob('events*'))
tbsize=tb.stat().st_size
def frozen_records(path, limit):
 """Parse TFRecord framing at a fixed file-size boundary; protobuf validates payload.

 CRC fields are skipped, not verified (explicit audit limitation). This avoids
 the fallback tensorboard reader's slow Python CRC over a 300+ MB live file.
 """
 with path.open('rb') as f:
  while f.tell()+12<=limit:
   header=f.read(12)
   n=struct.unpack('<Q',header[:8])[0]
   if n>20_000_000: raise ValueError('implausible event record length')
   if f.tell()+n+4>limit: break
   raw=f.read(n)
   if len(f.read(4))!=4: break
   yield raw
events=0
for raw in frozen_records(tb,tbsize):
 events+=1
 if not any(s in raw for s in (b'BS/',b'_avg',b'cu_matching_',b'Reward/')): continue
 e=Event.FromString(raw)
 for v in e.summary.value:
  t=v.tag
  if not (t.startswith('BS/') or t.startswith('PolicyDistribution/') or t.startswith('Reward/')): continue
  # BS metrics are written with the zero-based pre-increment iteration.
  update=int(e.step)+(1 if t.startswith('BS/') else 0)
  if update<1 or update>END: continue
  if '/dim_' in t: continue
  if v.HasField('simple_value'): value=float(v.simple_value)
  elif v.HasField('tensor') and v.tensor.float_val: value=float(v.tensor.float_val[0])
  else: raise ValueError((t,v))
  series[t][update]=value
windows={'first500':(1,500),'second500':(501,1000),'last1000':(END-999,END),'previous500':(END-999,END-500),'last500':(END-499,END),'last100':(END-99,END)}
def stats(values):
 a=np.asarray(values,dtype=float)
 return {'n':int(a.size),'finite_n':int(np.isfinite(a).sum()),'mean':float(np.mean(a)),'std':float(np.std(a)),
   'min':float(np.min(a)),'median':float(np.median(a)),'q05':float(np.quantile(a,.05)),'q95':float(np.quantile(a,.95)),'max':float(np.max(a)),
   'negative_n':int((a<0).sum())}
summary={'end_update':END,'tb_path':str(tb.relative_to(ROOT)),'tb_bytes_at_read_start':tbsize,'tb_events_read':events,'tfrecord_crc_verified':False,
  'step_semantics':{'BS':'zero-based stored step +1 => completed update','PolicyDistribution':'stored step == completed update; collection policy version update-1'},
  'tag_count':len(series),'windows':{}}
for name,(lo,hi) in windows.items():
 summary['windows'][name]={'updates':[lo,hi],'metrics':{t:stats([x for u,x in d.items() if lo<=u<=hi]) for t,d in series.items() if any(lo<=u<=hi for u in d)}}
summary['coverage']={t:{'n':len(d),'min_update':min(d),'max_update':max(d),'missing_n':END-len(d)} for t,d in series.items()}
summary['all_recorded_metrics_finite']=all(math.isfinite(x) for d in series.values() for x in d.values())
# Empirical entropy terms, using unnormalized dimension sums in actual actor loss.
dims=hyper['effective']['actor_architecture']['continuous_action_splits']
for name,w in summary['windows'].items():
 metrics=w['metrics']
 sum_h=sum(n*metrics[f'PolicyDistribution/{h}/entropy_avg']['mean'] for h,n in dims.items())
 beam_h=sum(dims[h]*metrics[f'PolicyDistribution/{h}/entropy_avg']['mean'] for h in dims if 'beam' in h)
 disc_h=sum(metrics[f'PolicyDistribution/cu_matching_{i}/entropy']['mean'] for i in range(4))
 w['rollout_entropy_scaling']={'continuous_sum':sum_h,'beam_sum':beam_h,'discrete_sum':disc_h,
  'continuous_loss_term':-.05*sum_h,'beam_loss_term':-.05*beam_h,'discrete_loss_term':-.01*disc_h,
  'beam_dimension_fraction':80/102,'uniform_u01_std':1/math.sqrt(12)}
summary['ppo_observability_limit']='No clip fraction, ratio quantile, per-head KL, analytic KL, advantage/return statistics, gradient norm, explained variance. BS/KL_Divergence is a signed sampled old-minus-new joint log likelihood, not exact nonnegative KL.'
summary['learning_rate']={'initial':.0003,'horizon_updates':10000,'floor':.00003,'last_completed_update':END,
 'logged_post_lr_at_last':.0003*max(.1,1-(END-1)/10000),
 'used_lr_at_last':.0003*max(.1,1-(END-2)/10000),
 'post_floor_first_update':9001,'floor_first_used_update':9002,'max_updates':hyper['hyperparameters']['max_updates'],
 'automatic_stop_at_episodes':False}
(OUT/'ppo_metrics.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'source_hashes_match':source['all_saved_source_match'],'checkpoints':list(source['checkpoint_checks']),
  'tb_tags':len(series),'tb_end':END,'tb_coverage_incomplete':{t:c for t,c in summary['coverage'].items() if c['missing_n']},
  'windows':{w:{t:round(v['mean'],8) for t,v in x['metrics'].items() if t.startswith('BS/') or (t.startswith('PolicyDistribution/') and any(k in t for k in ['/std_avg','/mean_avg','/max_probability','/entropy']))} for w,x in summary['windows'].items()}},ensure_ascii=False,indent=2))
