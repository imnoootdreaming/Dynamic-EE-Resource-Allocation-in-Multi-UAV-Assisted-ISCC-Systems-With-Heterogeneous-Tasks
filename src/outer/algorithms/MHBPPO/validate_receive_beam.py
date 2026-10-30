"""Held-out stochastic receiver evaluation, with no environment/CCCP imports.

Only previously reserved test rows are evaluated. Matching and CU power are
fixed to recorded actions; each branch samples its own receive beams.
"""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.distributions import Beta

from algorithms.MHBPPO.beam_pretraining import (
    load_dataset, build_actor_from_checkpoint, beam_dists,
    amplitude_phase_beams, teacher_beta_parameters,
)
from algorithms.MHBPPO.checkpoint import load_state_normalization


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def complex_beams(amplitude, phase_unit, uavs, antennas):
    """Preserve amplitude-all-UAVs then phase-all-UAVs action layout."""
    return amplitude_phase_beams(amplitude, phase_unit, uavs, antennas)


def sensing_metrics(g, target, distance, h_selected, p_selected, params):
    """g/target: [draw,batch,I,N]/[batch,I,N]; h follows stored dot(g,h)."""
    g=np.asarray(g,dtype=np.complex128)
    norms=np.linalg.norm(g,axis=-1,keepdims=True)
    if np.any(norms<1e-12) or not np.all(np.isfinite(norms)):
        raise ValueError("invalid physical receive beam")
    # cccp_bridge re-normalizes the outer float32 beam in complex128 as well.
    g=g/norms
    antennas = g.shape[-1]
    gain = np.abs(np.einsum("bin,dbin->dbi", target.conj(), g))**2
    interference = np.abs(np.einsum("dbin,bin->dbi", g, h_selected))**2 * p_selected[None, :, :]
    gamma = interference + float(params["sigma_2"])
    lam = (float(params["xi_0"])*float(params["rho_ref"])*antennas**2 / distance**4)[None, :, :] * gain
    margin = float(params["P_max_uav"])*lam/(float(params["eps_sinr"])*gamma)
    deficit = np.maximum(0, -np.log(np.maximum(margin, np.finfo(float).tiny)))
    q = deficit/(deficit+np.log(10.0))
    return {"gain": gain, "margin": margin, "gamma": gamma,
            "Q": q.mean(axis=-1), "joint_pass": (margin >= 1).all(axis=-1).astype(float)}


def communication_channels(raw, uavs, cus, antennas):
    """Read the existing named CU-channel segment; no geometry approximation."""
    from environment.observation import BS_OBSERVATION_SEGMENTS
    from types import SimpleNamespace
    shape = SimpleNamespace(uavs_num=uavs, cus_num=cus, antenna_nums=antennas, targets_num=40)
    offset = 0
    for segment in BS_OBSERVATION_SEGMENTS:
        size = int(segment.dim(shape))
        if segment.name == "h_uav_cu":
            flat = np.asarray(raw[:, offset:offset+size], dtype=np.float64)
            half = size//2
            return (flat[:, :half] + 1j*flat[:, half:]).reshape(-1, uavs, cus, antennas)
        offset += size
    raise ValueError("CU channel observation segment not found")


def _summary(state_values, slots):
    gain = state_values["gain"]
    result = {"states": int(len(slots)), "sampled_gain": float(gain.mean()),
              "gain_by_uav": gain.mean(axis=0).tolist(),
              "state_gain_quantiles": np.quantile(gain.mean(axis=1), [.05,.25,.5,.75,.95]).tolist(),
              "Q": float(state_values["Q"].mean()),
              "joint_M_pass_percent": float(100*state_values["joint_pass"].mean()),
              "mean_M_by_uav": state_values["margin"].mean(axis=0).tolist(),
              "mean_gamma_by_uav": state_values["gamma"].mean(axis=0).tolist(),
              "mean_direction_gain": float(state_values["mean_gain"].mean()),
              "mean_coordinate_std": float(state_values["coordinate_std"].mean()),
              "mean_concentration": float(state_values["concentration"].mean()),
              "teacher_kl_per_coordinate": float(state_values["teacher_kl"].mean()),
              "raw_mean_mse": float(state_values["raw_mean_mse"].mean()),
              "gain_MC_standard_error_conditional_states": float(np.sqrt(state_values["draw_mean_gain_variance"].sum())/len(slots)),
              "fallbacks": int(state_values["fallbacks"]),
              "slots": {str(k): int(v) for k,v in sorted(Counter(map(int,slots)).items())}}
    result["slot_groups"] = {}
    for name, mask in (("0_7", slots < 8), ("8_39", slots >= 8)):
        result["slot_groups"][name] = {"states": int(mask.sum()),
            "sampled_gain": float(gain[mask].mean()) if mask.any() else None,
            "Q": float(state_values["Q"][mask].mean()) if mask.any() else None,
            "joint_M_pass_percent": float(100*state_values["joint_pass"][mask].mean()) if mask.any() else None}
    return result


def evaluate_branch(actor, states, teacher, distance, h, powers, params,
                    *, draws=32, seed=420016, batch_size=64, rho=.9, kappa=64, teacher_policy=False):
    uavs, antennas = teacher.shape[1:]
    n = len(states)
    values = {key: np.empty((n,uavs)) for key in ("gain","margin","gamma","mean_gain")}
    values.update({key: np.empty(n) for key in ("Q","joint_pass","coordinate_std","concentration","teacher_kl","raw_mean_mse","draw_mean_gain_variance")})
    values["fallbacks"] = 0
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        torch.manual_seed(seed)
        for start in range(0,n,batch_size):
            stop=min(n,start+batch_size)
            target = teacher[start:stop]
            teacher_params = teacher_beta_parameters(target.real, target.imag, rho=rho, kappa=kappa)
            amplitude_target = torch.tensor(teacher_params["mean"][:, :uavs*antennas], dtype=torch.float32)
            phase_target = torch.tensor(teacher_params["mean"][:, uavs*antennas:], dtype=torch.float32)
            q_amplitude=Beta(kappa*amplitude_target,kappa*(1-amplitude_target))
            q_phase=Beta(kappa*phase_target,kappa*(1-phase_target))
            if teacher_policy:
                amplitude, phase = q_amplitude, q_phase
            else:
                amplitude,phase = beam_dists(actor, torch.tensor(states[start:stop],dtype=torch.float32))
            from torch.distributions import kl_divergence
            values["teacher_kl"][start:stop]=torch.cat((kl_divergence(q_amplitude,amplitude),kl_divergence(q_phase,phase)),dim=1).mean(dim=1).numpy()
            values["raw_mean_mse"][start:stop]=torch.cat(((amplitude.mean-amplitude_target)**2,(phase.mean-phase_target)**2),dim=1).mean(dim=1).numpy()
            g,bad=complex_beams(amplitude.sample((draws,)),phase.sample((draws,)),uavs,antennas)
            mean_g,_=complex_beams(amplitude.mean,phase.mean,uavs,antennas)
            metrics=sensing_metrics(g.numpy().astype(np.complex128),target,distance[start:stop],h[start:stop],powers[start:stop],params)
            for key in ("gain","margin","gamma","Q","joint_pass"):
                values[key][start:stop]=metrics[key].mean(axis=0)
            values["draw_mean_gain_variance"][start:stop]=metrics["gain"].mean(axis=2).var(axis=0,ddof=1)/draws if draws>1 else np.nan
            values["mean_gain"][start:stop]=np.abs(np.einsum("bin,bin->bi",target.conj(),mean_g.numpy()))**2
            values["coordinate_std"][start:stop]=torch.cat((amplitude.stddev,phase.stddev),dim=1).mean(dim=1).numpy()
            values["concentration"][start:stop]=torch.cat((amplitude.concentration1+amplitude.concentration0,phase.concentration1+phase.concentration0),dim=1).mean(dim=1).numpy()
            values["fallbacks"] += int(bad.sum())
    if any(not np.all(np.isfinite(value)) for key,value in values.items() if key != "fallbacks"):
        raise ValueError("nonfinite stochastic physical evaluation")
    return values


def round_bootstrap_difference(before, after, rounds, *, seed=420017, repeats=2000):
    delta=after.mean(axis=1)-before.mean(axis=1)
    unique=np.unique(rounds)
    blocks=np.asarray([delta[rounds==r].mean() for r in unique])
    rng=np.random.default_rng(seed)
    boot=blocks[rng.integers(0,len(blocks),size=(repeats,len(blocks)))].mean(axis=1)
    return {"rounds": unique.tolist(), "paired_round_gain_differences": blocks.tolist(),
            "mean_round_difference": float(blocks.mean()),
            "descriptive_bootstrap_95": np.quantile(boot,[.025,.975]).tolist(),
            "scope": "selected fixed-scene blocks; not independent training-seed evidence"}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",required=True)
    parser.add_argument("--source-checkpoint",required=True)
    parser.add_argument("--pretrained",required=True,help="pretrained actor file or initialization directory")
    parser.add_argument("--output",required=True)
    parser.add_argument("--preregistered",required=True)
    parser.add_argument("--draws",type=int,default=32)
    parser.add_argument("--seed",type=int,default=420016)
    args=parser.parse_args(argv)
    torch.set_num_threads(1)
    started=time.perf_counter()
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    if (out/"validation.json").exists():
        raise FileExistsError(out/"validation.json")
    arrays,manifest=load_dataset(args.dataset)
    actor,norm,sourceinfo=build_actor_from_checkpoint(args.source_checkpoint)
    treated=copy.deepcopy(actor)
    warm_path=Path(args.pretrained)
    if warm_path.is_dir(): warm_path=warm_path/"mhbppo_actor.pth"
    treated.load_state_dict(torch.load(warm_path,map_location="cpu",weights_only=True),strict=True)
    actor.eval();treated.eval()
    warm_norm=load_state_normalization(warm_path.parent/"state_normalization.npz",actor_path=warm_path,expected_iteration=0)
    norm_same=all(np.array_equal(getattr(norm,k),getattr(warm_norm,k)) for k in ("mean","std","m2")) and norm.count==warm_norm.count and norm.epsilon==warm_norm.epsilon
    frozen_diff=[];beam_changed=[]
    for key,value in actor.state_dict().items():
        equal=torch.equal(value,treated.state_dict()[key])
        if any(key.startswith("continuous_heads."+name+".") for name in actor.RECEIVE_BEAM_HEADS):
            if not equal:beam_changed.append(key)
        elif not equal:frozen_diff.append(key)
    mask=np.asarray(arrays["split"])==2
    if not mask.any():raise ValueError("empty reserved test set")
    raw=np.asarray(arrays["raw_states"])[mask]
    states=np.asarray(norm.normalize(raw),dtype=np.float32)
    teacher=np.asarray(arrays["teacher_g_real"])[mask]+1j*np.asarray(arrays["teacher_g_imag"])[mask]
    teacher=teacher.astype(np.complex128)
    distance=np.asarray(arrays["uav_target_distances"])[mask]
    matching=np.asarray(arrays["physical_matching"])[mask].astype(int)
    power=np.asarray(arrays["physical_cu_power"])[mask]
    uavs,antennas=teacher.shape[1:];cus=power.shape[1]
    h_all=communication_channels(raw,uavs,cus,antennas)
    h=h_all[np.arange(len(raw))[:,None],np.arange(uavs)[None,:],matching]
    p=power[np.arange(len(raw))[:,None],matching]
    params=manifest.get("inner_parameter_source",manifest.get("metadata",{}).get("inner_parameter_source"))
    if params is None:raise ValueError("dataset manifest lacks original inner parameters")
    prereg=json.loads(Path(args.preregistered).read_text(encoding="utf-8"))
    if args.draws!=prereg["test"]["draws"] or args.seed!=prereg["test"]["seed"]:
        raise ValueError("test sampling differs from preregistration")
    with torch.no_grad():
        a,_=actor._get_dists(torch.tensor(states[:64]))
        b,_=treated._get_dists(torch.tensor(states[:64]))
        other_distribution_same=all(torch.equal(a[k].concentration1,b[k].concentration1) and torch.equal(a[k].concentration0,b[k].concentration0) for k in a if k not in actor.RECEIVE_BEAM_HEADS)
        _,da=actor._get_dists(torch.tensor(states[:64]));_,db=treated._get_dists(torch.tensor(states[:64]))
        other_distribution_same &= all(torch.equal(x.logits,y.logits) for x,y in zip(da,db))
    slots=np.asarray(arrays["time_slot"])[mask]
    rounds=np.asarray(arrays["collection_round"])[mask]
    branches={};summaries={}
    for name,model,is_teacher in (("original",actor,False),("pretrained",treated,False),("finite_teacher",actor,True)):
        branches[name]=evaluate_branch(model,states,teacher,distance,h,p,params,draws=args.draws,seed=args.seed,
                    rho=prereg["teacher_rho"],kappa=prereg["teacher_kappa"],teacher_policy=is_teacher)
        summaries[name]=_summary(branches[name],slots)
        print(json.dumps({"branch":name,"summary":summaries[name]},ensure_ascii=False),flush=True)
    boot=round_bootstrap_difference(branches["original"]["gain"],branches["pretrained"]["gain"],rounds,
            seed=prereg["test"]["bootstrap_seed"],repeats=prereg["test"]["bootstrap_replicates"])
    original_critic=Path(args.source_checkpoint)/"mhbppo_critic.pth"
    warmed_critic=warm_path.parent/"mhbppo_critic.pth"
    critic_same=original_critic.is_file() and warmed_critic.is_file() and file_sha(original_critic)==file_sha(warmed_critic)
    hard=norm_same and not frozen_diff and other_distribution_same and critic_same and all(s["fallbacks"]==0 for s in summaries.values())
    gate=prereg["direction_learning_gate"]
    direction=hard and summaries["pretrained"]["sampled_gain"]-summaries["original"]["sampled_gain"]>=gate["minimum_absolute_gain_improvement"] and boot["descriptive_bootstrap_95"][0]>gate["round_bootstrap_95_lower_greater_than"]
    physics=summaries["pretrained"]["joint_M_pass_percent"]>summaries["original"]["joint_M_pass_percent"] and summaries["pretrained"]["Q"]<summaries["original"]["Q"]
    result={"schema_version":1,"source_checkpoint":str(Path(args.source_checkpoint).resolve()),
            "sourceinfo":sourceinfo,"dataset":str(Path(args.dataset).resolve()),"dataset_sha256":file_sha(args.dataset),
            "pretrained_actor_sha256":file_sha(warm_path),"preregistered_sha256":file_sha(args.preregistered),
            "test_draws":args.draws,"test_seed":args.seed,"summaries":summaries,"round_bootstrap":boot,
            "normalization_exact":norm_same,"nonbeam_changed_keys":frozen_diff,"beam_changed_keys":beam_changed,
            "nonbeam_distributions_exact":bool(other_distribution_same),"critic_exact":critic_same,
            "artifact_hard_gates_passed":bool(hard),"direction_learning_passed":bool(direction),
            "fixed_action_sensing_improved":bool(physics),"full_PPO_or_optimal_improvement_tested":False,
            "elapsed_seconds":time.perf_counter()-started,
            "notes":["Float32 CU channels reconstructed from the actual observation layout; physical g re-normalized in complex128 as cccp_bridge.","raw_mean_mse measures latent [0,1] Beta means, not physical 2u-1 values.","Historical source encoder/norm may have seen these states; held-out concerns this supervision only.","Bootstrap describes selected fixed-scene rounds, not independent seeds."]}
    (out/"validation.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    payload={"time_slot":slots,"collection_round":rounds}
    for name,value in branches.items():
        for key,data in value.items():payload[name+"_"+key]=data
    np.savez_compressed(out/"test_state_metrics.npz",**payload)
    print(json.dumps({k:result[k] for k in ("artifact_hard_gates_passed","direction_learning_passed","fixed_action_sensing_improved","elapsed_seconds")}),flush=True)
    if not hard:raise SystemExit(2)
    return result


if __name__=="__main__":main()
