"""One-time held-out comparison of a private receive encoder and head-only arm."""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from algorithms.MHBPPO.beam_pretraining import load_dataset,build_actor_from_checkpoint,beam_dists,file_sha256
from algorithms.MHBPPO.beam_branch import branch_dists,load_branch_checkpoint
from algorithms.MHBPPO.checkpoint import load_state_normalization
from algorithms.MHBPPO.validate_receive_beam import (
    evaluate_branch,_summary,communication_channels,round_bootstrap_difference,
)


class BeamEvaluationView(nn.Module):
    """Route the reusable beam-only evaluator through the private encoder.

    This view is not a deployable actor; full policy checks use the real actor.
    """
    def __init__(self,branch):
        super().__init__();self.branch=branch

    @property
    def continuous_heads(self):return self.branch.continuous_heads

    def _forward_features(self,states):return self.branch.beam_features(states)


def exact_norm(first,second):
    return all(np.array_equal(getattr(first,k),getattr(second,k)) for k in ("mean","std","m2")) and first.count==second.count and first.epsilon==second.epsilon


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("dataset","source-checkpoint","experiment","output","preregistered"):parser.add_argument("--"+name,required=True)
    args=parser.parse_args(argv)
    started=time.perf_counter();torch.set_num_threads(1)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if (out/"validation.json").exists():raise FileExistsError(out/"validation.json")
    reg=json.loads(Path(args.preregistered).read_text(encoding="utf-8"))
    arrays,manifest=load_dataset(args.dataset)
    original,norm,sourceinfo=build_actor_from_checkpoint(args.source_checkpoint)
    experiment=Path(args.experiment)
    head=copy.deepcopy(original)
    headpath=experiment/"initial_head_only/mhbppo_actor.pth"
    head.load_state_dict(torch.load(headpath,map_location="cpu",weights_only=True),strict=True)
    headnorm=load_state_normalization(headpath.parent/"state_normalization.npz",expected_iteration=0,actor_path=headpath)
    branch,branchnorm,branchinfo=load_branch_checkpoint(experiment/"initial_branch")
    for model in (original,head,branch):model.eval()
    view=BeamEvaluationView(branch);view.eval()
    normalized_same=exact_norm(norm,headnorm) and exact_norm(norm,branchnorm)
    nonbeam_errors=[]
    for name,tensor in original.state_dict().items():
        if any(name.startswith("continuous_heads."+h+".") for h in original.RECEIVE_BEAM_HEADS):continue
        for arm,model in (("head_only",head),("branch",branch)):
            if name not in model.state_dict() or not torch.equal(tensor,model.state_dict()[name]):nonbeam_errors.append(arm+":"+name)
    mask=arrays["split"]==2
    if not mask.any():raise ValueError("empty reserved test split")
    raw=arrays["raw_states"][mask]
    states=np.asarray(norm.normalize(raw),dtype=np.float32)
    teacher=(arrays["teacher_g_real"]+1j*arrays["teacher_g_imag"])[mask]
    distance=arrays["uav_target_distances"][mask]
    power=arrays["physical_cu_power"][mask];matching=arrays["physical_matching"][mask]
    uavs,antennas=teacher.shape[1:];cus=power.shape[1]
    all_h=communication_channels(raw,uavs,cus,antennas)
    h=all_h[np.arange(len(raw))[:,None],np.arange(uavs)[None,:],matching]
    p=power[np.arange(len(raw))[:,None],matching]
    slots=arrays["time_slot"][mask];rounds=arrays["collection_round"][mask]
    params=manifest["inner_parameter_source"]
    small=torch.tensor(states[:64],dtype=torch.float32)
    with torch.no_grad():
        od,oc=original._get_dists(small)
        nonbeam_distributions=True
        for model in (head,branch):
            md,mc=model._get_dists(small)
            for name in od:
                if name in original.RECEIVE_BEAM_HEADS:continue
                nonbeam_distributions &= torch.equal(od[name].concentration1,md[name].concentration1) and torch.equal(od[name].concentration0,md[name].concentration0)
            nonbeam_distributions &= all(torch.equal(x.logits,y.logits) for x,y in zip(oc,mc))
        direct=branch_dists(branch,small);via_view=beam_dists(view,small);full,_=branch._get_dists(small)
        dispatch_exact=all(torch.equal(x.concentration1,y.concentration1) and torch.equal(x.concentration0,y.concentration0) for x,y in zip(direct,via_view))
        dispatch_exact &= all(torch.equal(direct[i].concentration1,full[name].concentration1) and torch.equal(direct[i].concentration0,full[name].concentration0) for i,name in enumerate(("uav_rec_beam_amplitude","uav_rec_beam_phase")))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(reg["test"]["seed"])
            sampled=branch.sample(small[:8])
            evaluated=branch.evaluate_actions(small[:8],sampled[0],sampled[1])
            sample_evaluate_exact=all(torch.allclose(a,b,atol=4e-5,rtol=2e-5) for a,b in ((sampled[2],evaluated[0]),(sampled[3],evaluated[2]),(sampled[4],evaluated[3])))
    critics_same=all(file_sha256(experiment/name/"mhbppo_critic.pth")==sourceinfo["critic_file_sha256"] for name in ("initial_original","initial_head_only","initial_branch"))
    values={};summaries={}
    for name,model,teacher_policy in (("original",original,False),("head_only",head,False),("branch",view,False),("finite_teacher",original,True)):
        values[name]=evaluate_branch(model,states,teacher,distance,h,p,params,
             draws=reg["test"]["draws"],seed=reg["test"]["seed"],
             rho=reg["teacher_rho"],kappa=reg["teacher_kappa"],teacher_policy=teacher_policy)
        summaries[name]=_summary(values[name],slots)
        print(json.dumps({"arm":name,"summary":summaries[name]}),flush=True)
    comparisons={name:round_bootstrap_difference(values[name]["gain"],values["branch"]["gain"],rounds,
          seed=reg["test"]["bootstrap_seed"],repeats=reg["test"]["bootstrap_replicates"]) for name in ("original","head_only")}
    artifact_hard=normalized_same and not nonbeam_errors and nonbeam_distributions and dispatch_exact and sample_evaluate_exact and critics_same and all(v["fallbacks"]==0 for v in summaries.values())
    gate=reg["direction_learning_gate"]
    direction=(artifact_hard and summaries["branch"]["sampled_gain"]-summaries["original"]["sampled_gain"]>=gate["minimum_absolute_gain_improvement_vs_original"]
       and comparisons["original"]["descriptive_bootstrap_95"][0]>gate["round_bootstrap_95_lower_vs_original_greater_than"]
       and comparisons["head_only"]["descriptive_bootstrap_95"][0]>gate["round_bootstrap_95_lower_vs_head_only_greater_than"])
    physics=summaries["branch"]["Q"]<summaries["original"]["Q"] and summaries["branch"]["joint_M_pass_percent"]>summaries["original"]["joint_M_pass_percent"]
    result={"schema_version":1,"source_checkpoint":args.source_checkpoint,"sourceinfo":sourceinfo,"branchinfo":branchinfo,
      "dataset":args.dataset,"dataset_sha256":file_sha256(args.dataset),"preregistered_sha256":file_sha256(args.preregistered),
      "test_seed":reg["test"]["seed"],"test_draws":reg["test"]["draws"],"summaries":summaries,"paired_round_comparisons":comparisons,
      "normalization_exact":bool(normalized_same),"nonbeam_parameter_errors":nonbeam_errors,"nonbeam_distributions_exact":bool(nonbeam_distributions),
      "branch_evaluation_dispatch_exact":bool(dispatch_exact),"full_policy_sample_evaluate_consistent":bool(sample_evaluate_exact),"critic_exact":bool(critics_same),
      "artifact_hard_gates_passed":bool(artifact_hard),"direction_learning_passed":bool(direction),"fixed_action_sensing_improved":bool(physics),
      "PPO_or_optimal_improvement_tested":False,"elapsed_seconds":time.perf_counter()-started,
      "limitations":["Fresh trajectory records after source checkpoint; fixed exogenous scene and correlated rounds.","Descriptive round bootstrap is not independent-seed superiority.","Only offline matched receiver learning; no original PPO checkpoint compatibility assumed."]}
    (out/"validation.json").write_text(json.dumps(result,indent=2,default=str),encoding="utf-8")
    payload={"time_slot":slots,"collection_round":rounds}
    for name,arm_values in values.items():
        for key,value in arm_values.items():payload[name+"_"+key]=value
    np.savez_compressed(out/"test_state_metrics.npz",**payload)
    print(json.dumps({k:result[k] for k in ("artifact_hard_gates_passed","direction_learning_passed","fixed_action_sensing_improved","elapsed_seconds")}),flush=True)
    if not artifact_hard:raise SystemExit(2)
    return result


if __name__=="__main__":main()
