import sys,json,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'/'outer'))
from configs.base_params import get_base_args
from configs.seed_manager import scenario_rng
from environment.topology import generate_pos,generate_cu_trajectory,generate_uav_target_schedule
from environment.channel_models import compute_com_channel_gain,compute_sen_channel_gain
args=get_base_args()
I,J,N=args.uavs_num,args.cus_num,args.antenna_nums
rho=10**(args.ref_path_loss_db/10)
sigma=10**((args.noise_power_density_dbm-30)/10)*args.bandwidth
eps=10**(args.sen_sinr/10)
pmin=0.1*10**((args.cu_max_power_dbm-30)/10)
Pmax=args.uav_max_power
last_ep=None
rows=[]
max_pos_error=max_beam_error=0.0
schedule_errors=0

def matching_exists(allowed):
    order=np.argsort(allowed.sum(axis=1))
    def rec(k,used):
        if k==I:return True
        for j in np.flatnonzero(allowed[order[k]]):
            if not used&(1<<int(j)) and rec(k+1,used|(1<<int(j))):return True
        return False
    return rec(0,0)

with (Path(__file__).parent/'actions_compact.jsonl').open() as f:
    for line in f:
        r=json.loads(line);ep=r['ep'];t=r['t']
        if ep!=last_ep:
            rng=scenario_rng(args.seed,1,0)
            init_uav,init_cu,targets,sectors=generate_pos(I,J,args.targets_num,args.center,args.radius,args.uav_height,rng=rng)
            traj=generate_cu_trajectory(init_cus_pos=init_cu,cus_num=J,total_time_slots=40,markov_velocity=args.markov_velocity,markov_memory_level=args.markov_memory_level,markov_asymptotic_mean_of_velocity=args.markov_asymptotic_mean_of_velocity,markov_standard_deviation_of_velocity=args.markov_standard_deviation_of_velocity,time_slot_duration=args.time_slot_duration,seed=rng)
            nlos={}
            for key,shape in [('uavs_2_cus',(41,I,J,N)),('uavs_2_bs',(41,I,1,N)),('cus_2_bs',(41,J,1))]:
                nlos[key]=(rng.standard_normal(shape)+1j*rng.standard_normal(shape))/np.sqrt(2)
            schedule,_=generate_uav_target_schedule(I,args.targets_num,init_uav,targets,40,4,args.center,sectors)
            max_pos_error=max(max_pos_error,float(np.max(np.abs(init_uav-np.array(r['pos'])))))
            last_ep=ep
        pos=np.array(r['pos']);idx=np.array(r['target']);powers=np.array(r['powers']);g=np.array(r['g']['real'])+1j*np.array(r['g']['imag']);off=np.array(r['off'])
        schedule_errors+=int(not np.array_equal(idx,schedule[t]))
        hc,hb,hcb=compute_com_channel_gain(pos,traj[t],rho,args.frac_d_lambda,args.alpha_uav_link,args.alpha_cu_link,10**(args.rician_factor_db/10),N,{k:v[t] for k,v in nlos.items()})
        d=np.linalg.norm(pos-targets[idx],axis=1)
        diff=pos-targets[idx];ang=np.arctan2(diff[:,1],diff[:,0]);a=np.exp(1j*2*np.pi*args.frac_d_lambda*np.sin(ang)[:,None]*np.arange(N))
        # Eigenvectors have arbitrary phase; compare projection magnitude.
        alignment=np.abs(np.sum(a.conj()*g,axis=1))**2/N
        max_beam_error=max(max_beam_error,float(np.max(np.abs(alignment-1))))
        lam=rho*args.radar_rcs*d**-4*N*np.abs(np.sum(a.conj()*g,axis=1))**2
        projected=np.abs(np.sum(g[:,None,:]*hc,axis=2))**2
        gamma_all=projected*powers[None,:]+sigma
        margins_all=Pmax*lam[:,None]/(eps*gamma_all)
        chosen=margins_all[np.arange(I),r['cu']]
        margins_low=Pmax*lam[:,None]/(eps*(projected*pmin+sigma))
        # Best possible receive beam per candidate CU with minimum legal CU power.
        hnorm=np.sum(np.abs(hc)**2,axis=2)
        overlap=np.abs(np.sum(hc*a[:,None,:],axis=2))**2
        quad=N/sigma-pmin*overlap/(sigma*(sigma+pmin*hnorm))
        margins_free=Pmax*(rho*args.radar_rcs*d**-4*N)[:,None]*quad/eps
        # Necessary CU time/CPU bound, optimistic: no UAV interference.
        rate=args.bandwidth*np.log2(1+powers*np.abs(hcb[:,0])**2/sigma)
        Dmin=args.cus_num*[0.0]
        Dmin=170e3/rate
        for i,j in enumerate(r['cu']):Dmin[j]=max(Dmin[j],args.uav_sen_duration+off[i])
        cpu_bad=bool(np.any(Dmin>=args.cu_max_delay))
        cpu_lb=float('inf') if cpu_bad else float(np.sum(args.bs_cycles_per_bit*170e3/(args.cu_max_delay-Dmin)))
        rows.append(dict(iter=r['iter'],ep=ep,t=t,status=r['status'],success=r['success'],d=d.tolist(),min_margin=float(chosen.min()),c06_bad=bool(np.any(chosen<1-1e-6)),matching_current=matching_exists(margins_all>=1),matching_low=matching_exists(margins_low>=1),matching_free=matching_exists(margins_free>=1),noise_bad=bool(np.any(Pmax*rho*args.radar_rcs*d**-4*N*N/(eps*sigma)<1)),cpu_bad=cpu_bad or cpu_lb>args.bs_max_freq,cpu_lb=cpu_lb))
(Path(__file__).parent/'bounds.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
def summary(xs):
    return dict(n=len(xs),success=sum(x['success'] for x in xs)/len(xs),mean_distance=float(np.mean([x['d'] for x in xs])),median_worst_distance=float(np.median([max(x['d']) for x in xs])),c06_bad=sum(x['c06_bad'] for x in xs)/len(xs),cpu_bad=sum(x['cpu_bad'] for x in xs)/len(xs),noise_bad=sum(x['noise_bad'] for x in xs)/len(xs),matching_current=sum(x['matching_current'] for x in xs)/len(xs),matching_low=sum(x['matching_low'] for x in xs)/len(xs),matching_free=sum(x['matching_free'] for x in xs)/len(xs),by_status={s:dict(n=sum(x['status']==s for x in xs),c06_bad=sum(x['status']==s and x['c06_bad'] for x in xs),cpu_bad=sum(x['status']==s and x['cpu_bad'] for x in xs)) for s in ['optimal','infeasible','unknown']})
late=[x for x in rows if x['iter']>=2001]
result=dict(parameters=args.as_dict(),checks=dict(max_initial_position_error=max_pos_error,max_matched_beam_projection_error=max_beam_error,schedule_errors=schedule_errors),early=summary([x for x in rows if x['iter']<=100]),late=summary(late),windows=[dict(window=w,**summary([x for x in late if x['t']//4==w])) for w in range(10)])
(Path(__file__).parent/'bounds_summary.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='parameters'},indent=2))
