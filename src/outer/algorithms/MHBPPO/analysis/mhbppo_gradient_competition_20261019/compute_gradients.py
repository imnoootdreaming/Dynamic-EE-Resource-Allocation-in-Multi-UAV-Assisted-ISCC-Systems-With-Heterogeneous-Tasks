"""Offline initial-objective gradients with an explicit missing-normalizer sensitivity.

No optimizer, env.step, reward computation, action sampling, or solver calls.
"""
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import contextlib
import io

import numpy as np
import torch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[5]
OUTER = ROOT / 'src/outer'
sys.path.insert(0, str(OUTER))
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from algorithms.MHBPPO.beam_pretraining import build_actor_from_checkpoint
from algorithms.MHBPPO.MHBPPO_agent import Critic
from algorithms.MHBPPO.generate_beam_pretrain_dataset import create_reconstruction_env, reconstruct_raw_state

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(name, data):
    (OUT/name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')

lines=(OUT/'actions_update5001.jsonl').read_text(encoding='utf-8').splitlines()
metadata=json.loads(lines[0]); rows=[json.loads(line) for line in lines[1:]]
assert all(metadata['madrl_args'][key]==value for key,value in {
    'gamma':.99,'lmbda':.95,'continuous_entropy_coef':.05,'discrete_entropy_coef':.01,
    'num_samplers':2,'total_time_slots':40,'structured_beam_alpha':1.,'epochs':5,'batch_size':32}.items())
actor, norm, source=build_actor_from_checkpoint(OUT/'source_frozen')
with torch.random.fork_rng(devices=[]):
    torch.manual_seed(0)
    critic=Critic(1017,128)
critic.load_state_dict(torch.load(OUT/'source_frozen/mhbppo_critic.pth',map_location='cpu',weights_only=True))
critic.eval()
assert all(r['actor_signature']==source['actor_signature'] and r['norm_signature']==source['norm_signature'] for r in rows)
parameters=dict(actor.named_parameters())
before_parameters={name:sha(OUT/'source_frozen'/name) for name in source['source_file_sha256']}
before_state={name:p.detach().clone() for name,p in parameters.items()}

with contextlib.redirect_stdout(io.StringIO()):
    env=create_reconstruction_env(metadata)
raw=[]; raw_next=[]
for r in rows:
    physical=r['environment_action']['physical_action']; t=int(r['step_idx'])
    raw.append(reconstruct_raw_state(env,t,physical['uavs_pos_before']))
    raw_next.append(reconstruct_raw_state(env,t+1,physical['uavs_pos_after']))
raw=np.asarray(raw);raw_next=np.asarray(raw_next)
assert np.array_equal(raw_next[:39],raw[1:40]) and np.array_equal(raw_next[40:79],raw[41:80])
states=torch.tensor(norm.normalize(raw),dtype=torch.float32)
next_states=torch.tensor(norm.normalize(raw_next),dtype=torch.float32)
actions=torch.tensor(np.asarray([r['raw_action']['continuous'] for r in rows]),dtype=torch.float32)
discrete=torch.tensor(np.asarray([r['raw_action']['discrete'] for r in rows]),dtype=torch.long)
reward=np.array([r['raw_bs_reward'] for r in rows],dtype=np.float64)
done=np.array([r['environment_action']['outcome']['done'] for r in rows],dtype=bool)
assert np.flatnonzero(done).tolist()==[39,79]
with torch.no_grad():
    dist, ddist=actor._get_dists(states)
    errors={}
    for head,d in dist.items():
        for field,tensor in [('mean',d.mean),('std',d.stddev),('concentration',d.concentration1+d.concentration0)]:
            expected=np.asarray([r['environment_action']['policy_distribution']['continuous'][head][field] for r in rows])
            errors[head+'/'+field]=float(np.max(np.abs(tensor.numpy()-expected)))
    for i,d in enumerate(ddist):
        expected=np.asarray([r['environment_action']['policy_distribution']['discrete'][f'cu_matching_{i}']['probabilities'] for r in rows])
        errors[f'discrete_{i}/probabilities']=float(np.max(np.abs(d.probs.numpy()-expected)))
    v=critic(states).view(-1).numpy()
    vn=critic(next_states).view(-1).numpy()
assert max(errors.values())<3e-6, errors

with (OUT/'training_updates_through5001.csv').open(encoding='utf-8',newline='') as f:
    csvrows=list(csv.DictReader(f))
historical_means=np.array([float(r['reward']) for r in csvrows[:5000]])
count=400000.+1e-8
mu=float(80*historical_means.sum()/count)
between=float((80*np.square(historical_means-mu).sum()+1e-8*mu*mu)/count)
sigma_lo=max(math.sqrt(max(0.,between))-1e-10,1e-4)
sigma_hi=math.sqrt((1-mu)*(mu+9))+1e-10
assert abs(reward.mean()-float(csvrows[5000]['reward']))<1e-12
gamma=.99; lam=.95
def gae(delta):
    result=np.empty(80,dtype=np.float64);carry=0.
    for i in reversed(range(80)):
        carry=float(delta[i])+gamma*lam*carry*(1-float(done[i]))
        result[i]=carry
    return result
a=gae(reward-mu)
b=gae(gamma*vn.astype(np.float64)*(1-done)-v.astype(np.float64))
ac=a-a.mean();bc=b-b.mean()
va=float(np.dot(ac,ac)/79);vb=float(np.dot(bc,bc)/79);vab=float(np.dot(ac,bc)/79)

# Build each loss from the same graph/current frozen policy. At entry theta=theta_old,
# so both importance ratios are 1 and clipping is locally inactive.
distributions,distributions_discrete=actor._get_dists(states)
lp_c=0.; hb=0.; ho=0.; offset=0
for name,d in distributions.items():
    width=actor.continuous_action_splits[name]
    lp_c=lp_c+d.log_prob(actions[:,offset:offset+width]).sum(-1)
    if name in actor.RECEIVE_BEAM_HEADS:
        hb=hb+d.entropy().sum(-1)
    else:
        ho=ho+d.entropy().sum(-1)
    offset+=width
lp_d=sum(d.log_prob(discrete[:,i]) for i,d in enumerate(distributions_discrete))
hd=sum(d.entropy() for d in distributions_discrete)
names=list(parameters);params=list(parameters.values())
slices={};position=0
for name,p in parameters.items():
    slices[name]=(position,position+p.numel());position+=p.numel()
def flat_gradient(loss):
    gs=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
    return np.concatenate([(torch.zeros_like(p) if g is None else g).detach().numpy().reshape(-1).astype(np.float64)
                           for g,p in zip(gs,params)])
ac_t=torch.tensor(ac,dtype=torch.float32);bc_t=torch.tensor(bc,dtype=torch.float32)
views={'all80':np.ones(80), 'worker0_contribution':np.array([r['worker_id']==0 for r in rows],float),
       'worker1_contribution':np.array([r['worker_id']==1 for r in rows],float),
       'slots0_7_contribution':np.array([r['step_idx']<8 for r in rows],float),
       'slots8_39_contribution':np.array([r['step_idx']>=8 for r in rows],float)}
vectors={}
for view, mask in views.items():
    mask_t=torch.tensor(mask,dtype=torch.float32)
    vectors[view]={'task_a':flat_gradient(-((lp_c+lp_d)*ac_t*mask_t).mean()),
        'task_b':flat_gradient(-((lp_c+lp_d)*bc_t*mask_t).mean()),
        'entropy_beam':flat_gradient(-.05*(hb*mask_t).mean()),
        'entropy_other':flat_gradient(-(.05*ho*mask_t+.01*hd*mask_t).mean())}

group_names={
 'beam_heads':[n for n in names if n.startswith('continuous_heads.uav_rec_beam_dir_')],
 'beam_real':[n for n in names if n.startswith('continuous_heads.uav_rec_beam_dir_real.')],
 'beam_imag':[n for n in names if n.startswith('continuous_heads.uav_rec_beam_dir_imag.')],
 'shared_trunk':[n for n in names if n.startswith(('fc1.','fc2.'))],
 'all_actor':names}
groups={name:np.concatenate([np.arange(*slices[n]) for n in ns]) for name,ns in group_names.items()}
def denominator(sigma):
    q=sigma+1e-8
    return math.sqrt(max(0.,va+2*q*vab+q*q*vb))+q*1e-5
point_cache={}
def point(ga,gb,gh,sigma):
    key=(id(ga),id(gb),id(gh))
    if key not in point_cache:
        point_cache[key]=(ga,gb,gh,float(ga@ga),float(ga@gb),float(gb@gb),float(ga@gh),float(gb@gh),float(gh@gh))
    _,_,_,aa,ab,bb,ah,bh,hh=point_cache[key]
    q=sigma+1e-8;den=denominator(sigma)
    tn=math.sqrt(max(0.,aa+2*q*ab+q*q*bb))/den;hn=math.sqrt(max(0.,hh))
    dot=(ah+q*bh)/den
    return {'reward_std':float(sigma),'task_grad_norm':tn,'beam_entropy_grad_norm':hn,
        'entropy_over_task_norm':hn/tn if tn else None,
        'cosine':dot/(tn*hn) if tn and hn else None,
        'opposing_projection_fraction':-dot/(tn*tn) if tn else None}
def cosine_extrema(ga,gb,gh):
    A=float(ga@ga);B=float(ga@gb);C=float(gb@gb);d=float(ga@gh);e=float(gb@gh)
    candidates=[sigma_lo,sigma_hi]
    if e*B-d*C != 0:
        sigma=(d*B-e*A)/(e*B-d*C)-1e-8
        if sigma_lo<sigma<sigma_hi:candidates.append(sigma)
    values=[point(ga,gb,gh,s) for s in candidates]
    return {'min':min(values,key=lambda p:p['cosine'] if p['cosine'] is not None else math.inf),
            'max':max(values,key=lambda p:p['cosine'] if p['cosine'] is not None else -math.inf),
            'method':'analytic stationary points of linear numerator / norm of affine vector; conditional on reconstructed critic/GAE and reward bounds'}

def norm_ratio_bounds(ga,gb,gh):
    # R = ||gh||*(sqrt(V(q)) + 1e-5*q)/sqrt(T(q)). Bound each
    # positive term over the full interval using all analytic stationary points.
    A=float(ga@ga);B=float(ga@gb);C=float(gb@gb);hn=float(np.linalg.norm(gh))
    qlo=sigma_lo+1e-8;qhi=sigma_hi+1e-8
    coefficients=np.array([vb*B-vab*C, vb*A-va*C, vab*A-va*B])
    coefficients=np.trim_zeros(coefficients,'f')
    candidates=[qlo,qhi]
    if len(coefficients)>1:
        for root in np.roots(coefficients):
            if abs(root.imag)<1e-8 and qlo<float(root.real)<qhi:candidates.append(float(root.real))
    main=[hn*math.sqrt(max(0.,va+2*q*vab+q*q*vb)/(A+2*q*B+q*q*C)) for q in candidates]
    extra_q=[qlo,qhi]
    if B and qlo<-A/B<qhi:extra_q.append(-A/B)
    extra=[hn*1e-5*q/math.sqrt(A+2*q*B+q*q*C) for q in extra_q]
    return {'lower':min(main)+min(extra)-1e-10,'upper':max(main)+max(extra)+1e-10,
        'method':'Stationary roots of V/T and q^2/T, then bound sum; includes advantage normalization epsilon; numerical roundoff margin1e-10.',
        'main_stationary_sigmas':[q-1e-8 for q in candidates]}

results={}
sigma_grid=np.linspace(sigma_lo,sigma_hi,2001)
for view,vec in vectors.items():
    results[view]={}
    for group,index in groups.items():
        ga=vec['task_a'][index];gb=vec['task_b'][index];gh=vec['entropy_beam'][index]
        samples=[point(ga,gb,gh,s) for s in sigma_grid]
        probes=sorted(set([sigma_lo,sigma_hi]+[s for s in (.25,.5,1.,2.) if sigma_lo<s<sigma_hi]))
        results[view][group]={'parameter_count':int(len(index)),
            'beam_entropy_norm':float(np.linalg.norm(gh)),
            'other_entropy_norm':float(np.linalg.norm(vec['entropy_other'][index])),
            'entropy_beam_other_cosine':float(gh@vec['entropy_other'][index]/(np.linalg.norm(gh)*np.linalg.norm(vec['entropy_other'][index]))) if np.linalg.norm(gh)*np.linalg.norm(vec['entropy_other'][index]) else None,
            'cosine_range':cosine_extrema(ga,gb,gh),
            'norm_ratio_analytic_bounds':norm_ratio_bounds(ga,gb,gh),
            'norm_ratio_grid_min':min(samples,key=lambda p:p['entropy_over_task_norm']),
            'norm_ratio_grid_max':max(samples,key=lambda p:p['entropy_over_task_norm']),
            'probes':[point(ga,gb,gh,s) for s in probes]}

# Verify the symbolic sensitivity using the original float32 TD/GAE procedure,
# full-batch autograd and finite perturbations of several missing std values.
direct_checks=[]
for sigma in [sigma_lo,.5,1.,sigma_hi]:
    rn=torch.tensor((reward-mu)/(sigma+1e-8),dtype=torch.float32)
    with torch.no_grad():
        td=rn+gamma*torch.tensor(vn)*(1-torch.tensor(done,dtype=torch.float32))-torch.tensor(v)
        advantage=[];carry=0.
        for delta,end in zip(reversed(td.numpy().reshape(-1,1)),reversed(done)):
            carry=delta+gamma*lam*carry*(1-float(end));advantage.insert(0,carry)
        adv=torch.tensor(np.asarray(advantage),dtype=torch.float32).view(-1)
        adv=(adv-adv.mean())/(adv.std()+1e-5)
    direct=flat_gradient(-((lp_c+lp_d)*adv).mean())
    indirect=(vectors['all80']['task_a']+(sigma+1e-8)*vectors['all80']['task_b'])/denominator(sigma)
    error=float(np.linalg.norm(direct-indirect)/max(np.linalg.norm(direct),1e-30))
    adv_indirect=(ac+(sigma+1e-8)*bc)/denominator(sigma)
    direct_checks.append({'sigma':sigma,'gradient_relative_l2_error':error,
        'advantage_max_abs_error':float(np.max(np.abs(adv.numpy()-adv_indirect)))})
assert max(x['gradient_relative_l2_error'] for x in direct_checks)<3e-5, direct_checks
assert all(torch.equal(p.detach(),before_state[name]) for name,p in parameters.items())
assert all(sha(OUT/'source_frozen'/name)==digest for name,digest in before_parameters.items())
assert all(p.grad is None for p in parameters.values())
np.savez_compressed(OUT/'gradient_vectors.npz',**{f'{view}__{name}':v for view,vec in vectors.items() for name,v in vec.items()})
np.savez_compressed(OUT/'reconstructed_batch.npz',raw_states=raw,raw_next_states=raw_next,
    states=states.numpy(),next_states=next_states.numpy(),continuous_actions=actions.numpy(),discrete_actions=discrete.numpy(),
    raw_reward=reward,done=done,value=v,next_value=vn,gae_a=a,gae_b=b)
result={'scope':'Initial full-batch loss gradients at checkpoint5000 for logged on-policy update5001; no optimizer updates',
 'historical_actual_surrogate_gradient_exactly_identified':False,
 'reason':'Reward-normalizer std/m2 missing. Results retain this as an unknown interval rather than inventing it; historical minibatch trajectory also unavailable.',
 'reconstruction':{'batch_records':80,'actor_norm_identity_match':True,'adjacent_state_exact_match':True,
    'logged_distribution_max_abs_error':max(errors.values()),'errors':errors,'raw_reward_csv_match':True,
    'source_unchanged':True,'model_parameters_unchanged':True,'solver_calls':0,'optimizer_steps':0},
 'reward_normalizer':{'mean_reconstructed_from_400000_rewards_batch_means':mu,
    'unknown_std_lower_bound':sigma_lo,'unknown_std_upper_bound':sigma_hi,
    'bound_assumptions':'Current I4 reward lies in [-9,1] for nonnegative energy; total variance >= variance of equal-sized batch means; population moments plus initialization prior1e-8.',
    'not_an_estimate_of_actual_std':True},
 'entropy_loss_at_checkpoint_batch':{'beam':float((-.05*hb.mean()).detach()),'other':float((-.05*ho.mean()-.01*hd.mean()).detach())},
 'parameter_groups':group_names,'results':results,'direct_affine_checks':direct_checks,
 'limitations':['single update5001, not evidence of repeated conflict at5804 or later',
    'full-batch entry objective, not historical random minibatches,5epochs or Adam preconditioned updates',
    'reward sigma unknown; analytical cos range and grid norm ranges conditional on reconstructed floating-point model and stated reward bounds',
    'gradient norm/angle is parameterization-dependent; does not directly prove a gain or completion improvement from lowering entropy'],
 'artifact_sha256':{name:sha(OUT/name) for name in ('protocol.json','actions_update5001.jsonl','reconstructed_batch.npz','gradient_vectors.npz')}}
save('gradient_results.json',result)
print(json.dumps({'reconstruction_max_error':max(errors.values()),'reward_std_interval':[sigma_lo,sigma_hi],
    'all80':{g:{'entropy_norm':r['beam_entropy_norm'],'cos_min':r['cosine_range']['min']['cosine'],
    'cos_max':r['cosine_range']['max']['cosine'],'ratio_min':r['norm_ratio_grid_min']['entropy_over_task_norm'],
    'ratio_max':r['norm_ratio_grid_max']['entropy_over_task_norm']} for g,r in results['all80'].items()}},indent=2))
