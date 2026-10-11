import json,runpy,contextlib
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
with open(HERE/'replay_stdout.txt','w') as fout,contextlib.redirect_stdout(fout):ns=runpy.run_path(str(HERE/'replay_bounds.py'))
globals().update({k:v for k,v in ns.items() if not k.startswith('__')})
from reward.cccp_bridge import build_inner_parameters,_assemble_environment
pars=build_inner_parameters(args,np.full(J,170e3))
with open(HERE/'actions_compact.jsonl') as f:acts={(r['iter'],r['t']):r for r in map(json.loads,f)}
results=[]
for wit in json.loads((HERE/'constructive_witnesses.json').read_text()):
    r=acts[wit['iter'],wit['t']];t=r['t'];pos=np.array(r['pos']);idx=np.array(r['target']);eta=np.zeros((I,J));eta[np.arange(I),wit['cu']]=1
    hc,hb,hcb=compute_com_channel_gain(pos,traj[t],rho,args.frac_d_lambda,args.alpha_uav_link,args.alpha_cu_link,10**(args.rician_factor_db/10),N,{k:v[t] for k,v in nlos.items()})
    A=compute_sen_channel_gain(args.radar_rcs,args.frac_d_lambda,pos,targets,N,rho)[np.arange(I),idx]
    g=np.array(wit['g']['real'])+1j*np.array(wit['g']['imag']);w=np.array(wit['w']['real'])+1j*np.array(wit['w']['imag']);b=np.array(wit['b']['real'])+1j*np.array(wit['b']['imag'])
    off=np.array(wit['uav_off_duration']);D=np.array(wit['cu_duration']);z=np.array(wit['z']);fu=np.array(wit['uav_freq']);fc=np.array(wit['cu_freq'])
    env=_assemble_environment(pars,q_uav_pos=pos,q_uav_pos_next=pos+np.array([r['dist'],np.zeros(I),np.zeros(I)]).T,q_cu_pos=traj[t],q_target_pos=targets,q_target_designated=targets[idx],h_cu_2_uav=hc,h_uav_2_bs=hb[:,0,:],h_cu_2_bs=hcb[:,0],A_theta=A,g_rec_beam=g,eta_share=eta,p_cu_power=np.full(J,wit['cu_power']),D_uav_off=off)
    W=np.einsum('ij,ik->ijk',w,w.conj());B=np.einsum('ij,ik->ijk',b,b.conj())
    s=np.real(np.einsum('ijk,ikj->i',env.G_sen_corr,W));sw=np.real(np.einsum('ijk,ikj->i',env.H_uav_bs,W));sb=np.real(np.einsum('ijk,ikj->i',env.H_uav_bs,B))
    sig=wit['cu_power']*np.abs(hcb[:,0])**2
    cu_bits=pars.B*(pars.D_bar_sen*eta.sum(axis=0)*np.log2(1+sig/(eta.T@sw+pars.sigma_2))+(eta.T@off)*np.log2(1+sig/(eta.T@sb+pars.sigma_2))+(D-env.Theta_cu_time)*env.upsilon_cu_rate)
    checks=dict(sinr=float(np.min(s/(pars.eps_sinr*env.Gamma_sinr))),uav_off=float(np.min(off*pars.B*np.log2(1+sb/env.Phi_off_inr)/z)),aux=float(np.min(z/(pars.xi_1*np.log2(1+pars.xi_2*s/env.Gamma_sinr)))),cu_bits=float(np.min(cu_bits/pars.L_cu_task)),cpu=float((fu.sum()+fc.sum())/pars.F_max),delay=float(np.max(pars.D_bar_sen+off+pars.C_sen*z/fu)),fresh=float(np.min(eta@D-pars.D_bar_sen-off)),power=float(max(np.sum(abs(w)**2,axis=1).max(),np.sum(abs(b)**2,axis=1).max())),cu_deadline=float(D.max()))
    valid=all(checks[k]>=1-1e-8 for k in ['sinr','uav_off','aux','cu_bits']) and checks['cpu']<=1 and checks['delay']<=pars.D_max_sen and checks['fresh']>=0 and checks['power']<=pars.P_max_uav and checks['cu_deadline']<pars.D_max_cu.min() and np.all(fc>=pars.C_cu_cycles*pars.L_cu_task/(pars.D_max_cu-D)*(1-1e-10))
    results.append(dict(iter=wit['iter'],t=t,valid=bool(valid),checks=checks))
(HERE/'witness_verification.json').write_text(json.dumps(dict(n=len(results),passed=sum(x['valid'] for x in results),results=results),indent=2))
