import sys,json,runpy,contextlib
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
HERE=Path(__file__).resolve().parent
with open(HERE/'replay_stdout.txt','w') as fout,contextlib.redirect_stdout(fout):
    ns=runpy.run_path(str(HERE/'replay_bounds.py'))
globals().update({k:v for k,v in ns.items() if not k.startswith('__')})
from reward import cccp_bridge as bridge
original_solver=bridge.run_pc3p
captured={}
def capture(ctx):
    result=original_solver(ctx);captured['ctx']=ctx;captured['result']=result;return result
bridge.run_pc3p=capture
with open(HERE/'actions_compact.jsonl') as f: actions={(r['iter'],r['t']):r for r in map(json.loads,f)}
chosen=[]
for w in range(2,10):
    candidates=[x for x in rows if x['iter']>=2001 and x['t']//4==w and x['t']<39 and not x['success'] and x['matching_low']]
    for k in range(min(4,len(candidates))):chosen.append(candidates[k*len(candidates)//4])
answers=[]
for x in chosen:
    r=actions[x['iter'],x['t']];t=r['t'];pos=np.array(r['pos']);g=np.array(r['g']['real'])+1j*np.array(r['g']['imag']);idx=np.array(r['target']);off=np.array(r['off'])
    hc,hb,hcb=compute_com_channel_gain(pos,traj[t],rho,args.frac_d_lambda,args.alpha_uav_link,args.alpha_cu_link,10**(args.rician_factor_db/10),N,{k:v[t] for k,v in nlos.items()})
    sensing=compute_sen_channel_gain(args.radar_rcs,args.frac_d_lambda,pos,targets,N,rho)
    A=sensing[np.arange(I),idx];ag=np.einsum('ijn,ij->in',A.conj(),g)
    lam=np.sum(np.abs(ag)**2,axis=1)
    proj=np.abs(np.sum(g[:,None,:]*hc,axis=2))**2
    margin=Pmax*lam[:,None]/(eps*(proj*pmin+sigma))
    cost=-np.log(np.maximum(margin,1e-300))+1e6*(margin<1)
    ii,jj=linear_sum_assignment(cost);eta=np.zeros((I,J));eta[ii,jj]=1
    target_eta=np.zeros((I,args.targets_num));target_eta[np.arange(I),idx]=1
    # Keep logged movement distance to preserve flight/collision conditions.
    posnext=np.array(actions[r['iter'],t+1]['pos'])
    result=bridge.solve_inner_energy(args,hc,hb,hcb,sensing,target_eta,eta,pos,posnext,off,np.full(J,pmin),np.full(J,170e3),g)
    ctx=captured['ctx'];sol=captured['result'];W=sol['W_sen_beam'];valid=False;residual={}
    if W is not None:
        B=sol['B_off_beam'];z=sol['z_aux_rate'];D=sol['D_cu_off'];freq=sol['f_uav_freq'];pars=ctx.params;env=ctx.env if hasattr(ctx,'env') else None
        trG=np.real(np.einsum('ijk,ikj->i',ctx.G_sen_corr,W))
        trHW=np.real(np.einsum('ijk,ikj->i',ctx.H_uav_bs,W));trHB=np.real(np.einsum('ijk,ikj->i',ctx.H_uav_bs,B))
        received_w=eta.T@trHW;received_b=eta.T@trHB
        shared_s=pars.D_bar_sen*eta.sum(axis=0);shared_o=eta.T@off
        cu_signal=pmin*np.abs(hcb[:,0])**2
        cu_bits=pars.B*(shared_s*np.log2(1+cu_signal/(received_w+pars.sigma_2))+shared_o*np.log2(1+cu_signal/(received_b+pars.sigma_2))+(D-shared_s-shared_o)*np.log2(1+cu_signal/pars.sigma_2))
        off_cap=off*pars.B*np.log2(1+trHB/ctx.Phi_off_inr)
        sense_rate=pars.xi_1*np.log2(1+pars.xi_2*trG/ctx.Gamma_sinr)
        cpu=float(freq.sum()+np.sum(pars.C_cu_cycles*pars.L_cu_task/(pars.D_max_cu-D)))
        delay=(pars.D_bar_sen+off-pars.D_max_sen)*freq+pars.C_sen*z
        residual=dict(min_sinr_margin=float(np.min(trG/(pars.eps_sinr*ctx.Gamma_sinr))),min_cu_bits_margin=float(np.min(cu_bits/170e3)),min_uav_off_margin=float(np.min(off_cap/np.maximum(z,1e-30))),min_sensing_aux_margin=float(np.min(z/np.maximum(sense_rate,1e-30))),max_delay_residual=float(np.max(delay)),cpu_ratio=cpu/pars.F_max,rank1_gap=float(sol['rank1_gap']),max_power=float(max(np.trace(W,axis1=1,axis2=2).real.max(),np.trace(B,axis1=1,axis2=2).real.max())),min_psd_eig=float(min(np.linalg.eigvalsh(W).min(),np.linalg.eigvalsh(B).min())))
        valid=all(residual[k]>=1-1e-5 for k in ['min_sinr_margin','min_cu_bits_margin','min_uav_off_margin','min_sensing_aux_margin']) and cpu<=pars.F_max*(1+1e-6) and residual['max_delay_residual']<=0.1 and residual['max_power']<=Pmax+1e-5 and residual['min_psd_eig']>=-1e-6 and residual['rank1_gap']<=pars.gamma_2
    answers.append(dict(iter=r['iter'],t=t,original_status=r['status'],new_status=str(sol['status']),new_converged=bool(sol['converged']),new_finite=bool(np.isfinite(result[0])),validated=bool(valid),old_cu=r['cu'],new_cu=jj.tolist(),old_powers=r['powers'],new_power=pmin,off=r['off'],residual=residual))
    (HERE/'counterfactual_results.json').write_text(json.dumps(answers,indent=2))
print(json.dumps(dict(n=len(answers),finite=sum(x['new_finite'] for x in answers),validated=sum(x['validated'] for x in answers),results=answers),indent=2))
