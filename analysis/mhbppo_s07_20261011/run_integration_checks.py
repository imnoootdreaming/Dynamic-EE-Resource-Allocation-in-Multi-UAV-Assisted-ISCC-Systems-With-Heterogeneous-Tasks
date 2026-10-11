"""Bounded real-solver sampler -> learner checks for S07; never runs long training."""
import copy
from datetime import datetime
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
OUTER = ROOT / "src" / "outer"
sys.path.insert(0, str(OUTER))
from configs.base_params import get_base_args
from configs.madrl_args import get_madrl_args
from environment.my_env import MyEnv
from algorithms.MHBPPO.MHBPPO_agent import HPPO
from algorithms.MHBPPO.MHBPPO_sampler import sampler_worker
from algorithms.MHBPPO.MHBPPO_main import validate_sample
from algorithms.MHBPPO.action_logging import (ActionLogWriter, EnvDiagnosticLogWriter,
    diagnostic_run_metadata, build_run_log_paths, diagnostic_log_path)

HERE = Path(__file__).resolve().parent


class ScalarCapture:
    def __init__(self):
        self.values = {}
    def add_scalar(self, tag, value, step):
        self.values[tag] = float(value)


def run_case(alpha):
    saved = sys.argv
    sys.argv = ["s07-smoke"]
    try:
        args = get_madrl_args()
    finally:
        sys.argv = saved
    args.structured_beam_alpha = alpha
    args.randomize_layout_per_episode = False
    args.randomize_cu_traj_per_episode = False
    args.randomize_nlos_per_episode = False
    args.env_diagnostics = True
    args.diagnostic_snapshot_limit = 0
    base = get_base_args()
    env = MyEnv(base, args, seed_offset=1)
    torch.manual_seed(1208)
    dim = env.observation_space["bs"].shape[0]
    space = env.action_space["bs"]
    learner = HPPO(dim, args.hidden_dim, env.bs_continuous_action_splits,
        space["discrete"].nvec, space["continuous"].low, space["continuous"].high,
        args.actor_lr, args.critic_lr, args.lmbda, args.eps, args.gamma, args.epochs,
        args.episodes, torch.device("cpu"), structured_beam_alpha=alpha)
    metadata = learner.actor.action_probability_metadata()
    ctx = mp.get_context("spawn")
    samples, weights, controls = (ctx.Queue(maxsize=1) for _ in range(3))
    stop = ctx.Event()
    weights.put(dict(actor_state_dict={k:v.detach().cpu().clone() for k,v in learner.actor.state_dict().items()},
                     weights_version=0, state_norm_mean=np.zeros(dim), state_norm_std=np.ones(dim)))
    worker = ctx.Process(target=sampler_worker, args=(base, args, samples, weights, stop, 1, controls))
    started = time.perf_counter()
    worker.start()
    controls.put(0)
    try:
        item = samples.get(timeout=180)
        assert "error" not in item, item.get("error")
        assert validate_sample(item, metadata) == 40
    finally:
        stop.set()
        worker.join(timeout=10)
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=5)
        for queue in (samples, weights, controls):
            queue.close()
    assert worker.exitcode == 0, worker.exitcode
    transition = item["transition_dict"]
    states = torch.as_tensor(np.asarray(transition["states"]), dtype=torch.float32)
    continuous = torch.as_tensor(np.asarray(transition["continuous_actions"]), dtype=torch.float32)
    discrete = torch.as_tensor(np.asarray(transition["discrete_actions"]), dtype=torch.long)
    with torch.no_grad():
        evaluated = learner.actor.evaluate_actions(states, continuous, discrete)
    np.testing.assert_allclose(evaluated[2].numpy().reshape(-1), transition["old_cont_log_probs"], atol=2e-4, rtol=1e-5)
    np.testing.assert_allclose(evaluated[3].numpy().reshape(-1), transition["old_disc_log_probs"], atol=2e-4, rtol=1e-5)
    before = {k:v.detach().clone() for k,v in learner.actor.state_dict().items()}
    capture = ScalarCapture()
    learner.update(transition, step=1, writer=capture, agent_name="BS")
    assert all(torch.isfinite(p).all() for p in learner.actor.parameters())
    beam_keys = [k for k in before if "uav_rec_beam_dir_" in k]
    unchanged = all(torch.equal(before[k], learner.actor.state_dict()[k]) for k in beam_keys)
    if alpha == 0:
        assert unchanged
        assert all(p.grad is None for name,p in learner.actor.named_parameters() if "uav_rec_beam_dir_" in name)
    else:
        assert not unchanged
    run_metadata = diagnostic_run_metadata(str(OUTER), base, args)
    run_metadata.update(metadata)
    run_metadata.update(seed=int(base.seed), continuous_action_splits=env.bs_continuous_action_splits)
    _, action_path = build_run_log_paths(str(HERE), "s07_alpha_"+str(alpha)+".log", datetime.now().astimezone())
    env_path = diagnostic_log_path(action_path)
    aw, ew = ActionLogWriter(action_path, run_metadata), EnvDiagnosticLogWriter(env_path, run_metadata)
    try:
        aw.write_episode(item, iteration=1)
        ew.write_episode(item, iteration=1)
    finally:
        aw.close()
        ew.close()
    actions = [json.loads(line) for line in Path(action_path).read_text(encoding="utf8").splitlines()]
    diagnostics = [json.loads(line) for line in Path(env_path).read_text(encoding="utf8").splitlines()]
    ar = [r for r in actions if r["record_type"] == "action"]
    er = [r for r in diagnostics if r["record_type"] == "env_diagnostic"]
    assert len(ar) == len(er) == 40
    for a,e in zip(ar,er):
        for key in ("iteration", "step_idx", "episode_idx", "weights_version"):
            assert a[key] == e[key]
    statuses = {}
    for r in er:
        kind = r["solver"]["solver_status_kind"]
        statuses[kind] = statuses.get(kind,0)+1
    return dict(alpha=alpha, transitions=40, sampler_exitcode=worker.exitcode,
        probability_contract=metadata, beam_head_unchanged=unchanged,
        finite_update_scalars=all(np.isfinite(v) for v in capture.values.values()),
        scalars=capture.values, status_counts=statuses, layout=env.target_layout_info,
        wall_seconds=time.perf_counter()-started, action_jsonl=action_path, env_jsonl=env_path)


if __name__ == "__main__":
    torch.set_num_threads(1)
    results = []
    for alpha in (0.0, 1.0):
        results.append(run_case(alpha))
        (HERE / "integration_summary.json").write_text(json.dumps(results, indent=2), encoding="utf8")
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)
