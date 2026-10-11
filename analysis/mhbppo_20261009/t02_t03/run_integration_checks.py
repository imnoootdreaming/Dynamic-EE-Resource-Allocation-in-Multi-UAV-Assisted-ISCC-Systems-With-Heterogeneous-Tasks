"""Bounded real-solver and sampler checks. No PPO update or long training."""
import copy
from datetime import datetime
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[3]
OUTER=ROOT/'src'/'outer'
sys.path.insert(0,str(OUTER))
from configs.base_params import get_base_args
from configs.madrl_args import get_madrl_args
from environment.my_env import MyEnv
from reward.cccp_bridge import build_inner_parameters,_assemble_environment
from reward.feasibility_diagnostics import validate_original_solution
from algorithms.MHBPPO.MHBPPO_sampler import sampler_worker
from algorithms.MHBPPO.MHBPPO_agent import MultiHeadActor
from algorithms.MHBPPO.MHBPPO_main import validate_sample
from algorithms.MHBPPO.action_logging import (
    ActionLogWriter,EnvDiagnosticLogWriter,build_run_log_paths,
    diagnostic_log_path,diagnostic_run_metadata,summarize_diagnostics,
)

HERE=Path(__file__).resolve().parent


def arguments():
    saved=sys.argv
    sys.argv=['check']
    try:args=get_madrl_args()
    finally:sys.argv=saved
    args.structured_beam_alpha=0.0
    args.randomize_layout_per_episode=False
    args.randomize_cu_traj_per_episode=False
    args.randomize_nlos_per_episode=False
    args.env_diagnostics=True
    args.diagnostic_snapshot_limit=1
    args.diagnostic_candidate_interval=3
    return args


def verify_existing_witnesses(base,args):
    env=MyEnv(base,args,seed_offset=1)
    env.reset(0)
    p=build_inner_parameters(base,np.full(base.cus_num,170e3))
    keys={}
    with (ROOT/'analysis/mhbppo_20261009/actions_compact.jsonl').open() as f:
        for line in f:
            r=json.loads(line)
            if r['iter'] in (2001,2002):keys[r['iter'],r['t']]=r
            elif r['iter']>2002:break
    results=[]
    witnesses=json.loads((ROOT/'analysis/mhbppo_20261009/constructive_witnesses.json').read_text())
    for wit in witnesses:
        r=keys[wit['iter'],wit['t']]
        env.t=wit['t'];env.cur_uavs_pos=np.array(r['pos'])
        env.cur_cus_pos=env.precomputed_cus_traj[env.t].copy()
        env._refresh_channels()
        idx=np.array(r['target']);eta=np.zeros((p.I,p.J));eta[np.arange(p.I),wit['cu']]=1
        unpack=lambda a:np.array(a['real'])+1j*np.array(a['imag'])
        e=_assemble_environment(p,q_uav_pos=env.cur_uavs_pos,
            q_uav_pos_next=env.cur_uavs_pos+np.array([r['dist'],np.zeros(p.I),np.zeros(p.I)]).T,
            q_cu_pos=env.cur_cus_pos,q_target_pos=env.init_targets_pos,
            q_target_designated=env.init_targets_pos[idx],
            h_cu_2_uav=env.uavs_2_cus_channels,h_uav_2_bs=env.uavs_2_bs_channels[:,0,:],
            h_cu_2_bs=env.cus_2_bs_channels[:,0],
            A_theta=env.uavs_2_targets_channels[np.arange(p.I),idx],
            g_rec_beam=unpack(wit['g']),eta_share=eta,p_cu_power=np.full(p.J,wit['cu_power']),
            D_uav_off=np.array(wit['uav_off_duration']))
        w,b=unpack(wit['w']),unpack(wit['b'])
        s=dict(W_sen_beam=np.einsum('ij,ik->ijk',w,w.conj()),
               B_off_beam=np.einsum('ij,ik->ijk',b,b.conj()),w_sen_beam=w,b_off_beam=b,
               f_uav_freq=np.array(wit['uav_freq']),f_cu_freq=np.array(wit['cu_freq']),
               z_aux_rate=np.array(wit['z']),D_cu_off=np.array(wit['cu_duration']))
        report=validate_original_solution(p,e,s)
        assert report['physical_solution_verified'], (wit['t'],report)
        results.append(dict(iter=wit['iter'],t=wit['t'],verified=True))
    (HERE/'witness_checks.json').write_text(json.dumps(results,indent=2))
    return len(results)


def compare_diagnostics_toggle(base,args):
    enabled=MyEnv(base,args,seed_offset=1)
    off_args=copy.copy(args);off_args.env_diagnostics=False
    disabled=MyEnv(base,off_args,seed_offset=1)
    enabled.reset(0);disabled.reset(0)
    actions=[]
    for t in range(40):
        continuous=np.full(sum(enabled.bs_continuous_action_splits.values()),0.5,np.float32)
        continuous[12:22]=0.0
        actions.append(dict(continuous=continuous,discrete=np.array([2,5,0,6])))
    left=[];right=[]
    for env,dest in ((enabled,left),(disabled,right)):
        for action in actions:
            next_state,reward,components,done,energy,success=env.step({'bs':action})
            dest.append((next_state['bs'].copy(),reward,done,energy,success))
            if env.env_diagnostics:
                assert env.last_env_diagnostic is not None
                assert env.last_env_diagnostic['solver']['problem_status'] is not None
            else:assert env.last_env_diagnostic is None
    max_reward_diff=0.0;max_energy_diff=0.0
    for a,b in zip(left,right):
        np.testing.assert_array_equal(a[0],b[0])
        assert a[2]==b[2] and a[4]==b[4]
        assert np.isfinite(a[3])==np.isfinite(b[3])
        np.testing.assert_allclose(a[1],b[1],rtol=1e-6,atol=1e-6)
        if np.isfinite(a[3]):
            np.testing.assert_allclose(a[3],b[3],rtol=1e-6,atol=1e-6)
            max_energy_diff=max(max_energy_diff,abs(a[3]-b[3]))
        max_reward_diff=max(max_reward_diff,abs(a[1]-b[1]))
    return dict(slots=40,finite_slots=int(sum(np.isfinite(x[3]) for x in left)),
                max_reward_difference=max_reward_diff,max_energy_difference=max_energy_diff)


def sampler_and_writer(base,args):
    env=MyEnv(base,args,seed_offset=1)
    actor=MultiHeadActor(state_dim=env.observation_space['bs'].shape[0],
        hidden_dim=args.hidden_dim,continuous_action_splits=env.bs_continuous_action_splits,
        discrete_action_dims=env.action_space['bs']['discrete'].nvec,
        continuous_dist_type='beta')
    ctx=mp.get_context('spawn');samples=ctx.Queue(maxsize=1);weights=ctx.Queue(maxsize=1)
    controls=ctx.Queue(maxsize=1);stop=ctx.Event()
    dim=env.observation_space['bs'].shape[0]
    weights.put(dict(actor_state_dict={k:v.detach().cpu().clone() for k,v in actor.state_dict().items()},
                     weights_version=0,state_norm_mean=np.zeros(dim),state_norm_std=np.ones(dim)))
    proc=ctx.Process(target=sampler_worker,args=(base,args,samples,weights,stop,1,controls))
    proc.start();controls.put(0)
    started=time.perf_counter()
    try:
        item=samples.get(timeout=90)
        assert 'error' not in item,item.get('error')
        assert validate_sample(item)==40
    finally:
        stop.set();proc.join(timeout=10)
        if proc.is_alive():proc.terminate();proc.join(timeout=5)
        for q in (samples,weights,controls):q.close()
    assert proc.exitcode==0,proc.exitcode
    metadata=diagnostic_run_metadata(str(OUTER),base,args)
    assert metadata['madrl_args']['structured_beam_alpha']==0.0
    log_path,action_path=build_run_log_paths(str(HERE),'smoke_train.log',datetime.now().astimezone())
    env_path=diagnostic_log_path(action_path)
    run_id=Path(log_path).stem
    metadata.update(run_id=run_id,seed=int(base.seed))
    action_writer=ActionLogWriter(action_path,metadata)
    env_writer=EnvDiagnosticLogWriter(env_path,{**metadata,'action_log_path':action_path})
    try:
        action_writer.write_episode(item,1)
        env_writer.write_episode(item,1)
        write_seconds=env_writer.last_write_seconds;write_bytes=env_writer.last_write_bytes
        # Same frozen scene reused by a second episode, without another solver run.
        item2=copy.copy(item);item2['episode_idx']=1
        action_writer.write_episode(item2,2);env_writer.write_episode(item2,2)
    finally:action_writer.close();env_writer.close()
    records=[json.loads(line) for line in Path(env_path).read_text(encoding='utf8').splitlines()]
    act_records=[json.loads(line) for line in Path(action_path).read_text(encoding='utf8').splitlines()]
    env_records=[x for x in records if x['record_type']=='env_diagnostic']
    act_records=[x for x in act_records if x['record_type']=='action']
    assert len(env_records)==80 and len(act_records)==80
    assert len([x for x in records if x['record_type']=='scene'])==1
    for diagnostic,action in zip(env_records,act_records):
        for k in ('run_id','iteration','episode_idx','sampler_id','step_idx','weights_version'):
            assert diagnostic[k]==action[k]
    assert sum('problem_snapshot' in x for x in env_records[:40])<=args.diagnostic_snapshot_limit
    assert sum(x['sensing']['candidates']['evaluated'] for x in env_records[:40])==14
    summary=summarize_diagnostics(item['diagnostic_traces'])
    summary.update(sample_wall_seconds=time.perf_counter()-started,
                   writer_seconds_per_episode=write_seconds,writer_bytes_per_episode=write_bytes,
                   env_jsonl=env_path,action_jsonl=action_path,scenes_written=1,records_written=80,
                   failure_snapshots_first_episode=sum('problem_snapshot' in x for x in env_records[:40]))
    return summary


if __name__=='__main__':
    base=get_base_args();args=arguments()
    if '--sampler-only' in sys.argv:
        result=json.loads((HERE/'integration_summary.json').read_text())
    else:
        result=dict(python=sys.executable,numpy=np.__version__,witnesses_verified=verify_existing_witnesses(base,args))
        (HERE/'integration_summary.json').write_text(json.dumps(result,indent=2))
        result['diagnostics_toggle']=compare_diagnostics_toggle(base,args)
        (HERE/'integration_summary.json').write_text(json.dumps(result,indent=2))
    result['sampler_log_contract']=sampler_and_writer(base,args)
    (HERE/'integration_summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
