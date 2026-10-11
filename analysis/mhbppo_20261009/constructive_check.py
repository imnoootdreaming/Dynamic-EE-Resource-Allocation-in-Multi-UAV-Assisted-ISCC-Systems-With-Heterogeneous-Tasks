import json,runpy,contextlib
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
HERE=Path(__file__).resolve().parent
with open(HERE/'replay_stdout.txt','w') as fout,contextlib.redirect_stdout(fout):ns=runpy.run_path(str(HERE/'replay_bounds.py'))
globals().update({k:v for k,v in ns.items() if not k.startswith('__')})
xi1=args.uav_sen_duration*args.radar_duty_ratio/(2*args.radar_impulse_duration)
xi2=2*args.var_range_fluctuation*args.radar_spectrum_shape**2*args.bandwidth**3*args.radar_impulse_duration
answers=[];witnesses=[]
with open(HERE/'actions_compact.jsonl') as f:
    for r in map(json.loads,f):
        if r['iter']<2001:continue
        t=r['t'];pos=np.array(r['pos']);idx=np.array(r['target']);off=np.array(r['off']);g=np.array(r['g']['real'])+1j*np.array(r['g']['imag'])
        hc,hb,hcb=compute_com_channel_gain(pos,traj[t],rho,args.frac_d_lambda,args.alpha_uav_link,args.alpha_cu_link,10**(args.rician_factor_db/10),N,{k:v[t] for k,v in nlos.items()})
        h=hb[:,0,:];hnorm=np.sum(np.abs(h)**2,axis=1)
        diff=pos-targets[idx];d=np.linalg.norm(diff,axis=1);ang=np.arctan2(diff[:,1],diff[:,0]);a=np.exp(1j*2*np.pi*args.frac_d_lambda*np.sin(ang)[:,None]*np.arange(N))
        lam=rho*args.radar_rcs*d**-4*N*np.abs(np.sum(a.conj()*g,axis=1))**2
        projected=np.abs(np.sum(g[:,None,:]*hc,axis=2))**2
        margin=Pmax*lam[:,None]/(eps*(projected*pmin+sigma))
        cost=-np.log(np.maximum(margin,1e-300))+1e6*(margin<1+1e-6)
        ii,jj=linear_sum_assignment(cost)
        gamma=projected[np.arange(I),jj]*pmin+sigma
        Psen=eps*gamma/lam*(1+1e-6)
        z=xi1*np.log2(1+xi2*Psen*lam/gamma)*(1+1e-6)
        cu_signal=pmin*np.abs(hcb[:,0])**2
        phi=cu_signal[jj]+sigma
        Poff=phi/hnorm*np.expm1(np.log(2)*z/(off*args.bandwidth))*(1+1e-6)
        trHW=Psen*np.abs(np.sum(h.conj()*a,axis=1))**2/N
        trHB=Poff*hnorm
        rate_direct=args.bandwidth*np.log2(1+cu_signal/sigma)
        D=170e3/rate_direct
        for i,j in enumerate(jj):
            shared_bits=args.bandwidth*(args.uav_sen_duration*np.log2(1+cu_signal[j]/(trHW[i]+sigma))+off[i]*np.log2(1+cu_signal[j]/(trHB[i]+sigma)))
            D[j]=max(args.uav_sen_duration+off[i],args.uav_sen_duration+off[i]+(170e3-shared_bits)/rate_direct[j])
        D+=1e-6
        fu=args.bs_cycles_per_bit*z/(args.uav_max_delay-args.uav_sen_duration-off)*(1+1e-6)
        fcu=args.bs_cycles_per_bit*170e3/(args.cu_max_delay-D)
        cpu=fu.sum()+fcu.sum()
        valid=bool(np.all(Psen<=Pmax) and np.all(Poff<=Pmax) and np.all(D<args.cu_max_delay) and cpu<=args.bs_max_freq)
        answers.append(dict(iter=r['iter'],t=t,original_status=r['status'],original_success=r['success'],valid=valid,max_sensing_power=float(Psen.max()),max_off_power=float(Poff.max()),cpu_ratio=float(cpu/args.bs_max_freq)))
        if valid and not r['success'] and len(witnesses)<40:
            w=np.sqrt(Psen/N)[:,None]*a;b=np.sqrt(Poff/hnorm)[:,None]*h
            witnesses.append(dict(iter=r['iter'],t=t,cu=jj.tolist(),cu_power=pmin,g=r['g'],w={'real':w.real.tolist(),'imag':w.imag.tolist()},b={'real':b.real.tolist(),'imag':b.imag.tolist()},z=z.tolist(),uav_freq=fu.tolist(),cu_freq=fcu.tolist(),cu_duration=D.tolist(),uav_off_duration=off.tolist(),checks=answers[-1]))
summary=dict(n=len(answers),feasible=sum(x['valid'] for x in answers),feasible_rate=sum(x['valid'] for x in answers)/len(answers),repaired_failures=sum(x['valid'] and not x['original_success'] for x in answers),windows=[dict(window=w,n=sum(x['t']//4==w for x in answers),feasible=sum(x['t']//4==w and x['valid'] for x in answers)) for w in range(10)])
(HERE/'constructive_summary.json').write_text(json.dumps(summary,indent=2));(HERE/'constructive_witnesses.json').write_text(json.dumps(witnesses,indent=2));(HERE/'constructive_results.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in answers));print(json.dumps(summary,indent=2))
