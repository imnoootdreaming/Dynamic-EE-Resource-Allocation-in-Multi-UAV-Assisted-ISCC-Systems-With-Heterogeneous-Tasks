"""Legacy multi-head Beta PPO with direct P0 physics and synchronous workers.

The policy and normalization modules derive from outer_old; PPO learning rates
use a floor for unlimited training. One worker
retains the original online normalization order. Multiple workers use a frozen
state-normalization snapshot for each collection round and independent legacy
reward scalers; their statistics and timing therefore differ from one worker.
"""
import csv
import hashlib
import json
import multiprocessing as mp
import os
import queue
import random
import shutil
import sys
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

_OUTER_ROOT = Path(__file__).resolve().parents[2]
if str(_OUTER_ROOT) not in sys.path:
    sys.path.insert(0, str(_OUTER_ROOT))

from algorithms.Pure_MHBPPO.Pure_MHBPPO_agent import HPPO, MultiHeadActor
from algorithms.Pure_MHBPPO.normalization import Normalization, RewardScaling
from algorithms.Pure_MHBPPO.physical import SENSING_SINR_TARGET_RATIO
from algorithms.Pure_MHBPPO.pure_env import PureMyEnv
from algorithms.MHBPPO.action_logging import ActionLogWriter, JsonlRotation
from algorithms.MHBPPO.rollout_batch import derive_action_seed, actor_state_signature
from algorithms.baseline_output import setup_run, check_scene, TraceLogs, atomic_checkpoint, capture, aggregate_metrics, METRIC_COLUMNS, log_scalar_metrics, training_log_stream, log_training_failure
from algorithms.training_schedule import configure_ppo_lr_horizon, training_rounds
from configs.base_params import get_base_args
from configs.madrl_args import get_madrl_args


TRANSITION_KEYS = ("states", "continuous_actions", "discrete_actions", "next_states",
                   "rewards", "old_cont_log_probs", "old_disc_log_probs", "dones", "real_dones")


class PhysicalTraceWriter(ActionLogWriter):
    def write_trace(self, record):
        self._write({"record_type": "physical_trace", **record})
        self._file.flush()


def seed_process(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def snapshot_normalizer(normalizer):
    stats = normalizer.running_ms
    return {"n": stats.n, "mean": np.asarray(stats.mean).copy(),
            "std": np.asarray(stats.std).copy(), "S": np.asarray(stats.S).copy()}


def frozen_normalize(raw, snapshot):
    # Before any observations, the legacy estimator's std is zero.  A unit
    # initial scale avoids dividing unobserved multi-worker states by 1e-8.
    mean = snapshot["mean"]
    std = snapshot["std"] if snapshot["n"] else np.ones_like(mean)
    return (np.asarray(raw, dtype=np.float32) - mean) / (std + 1e-8)


def worker_main(base_args, args, worker_id, control, results):
    """Own the environment and reward scaler for the entire worker lifetime."""
    try:
        if args.worker_num_threads:
            torch.set_num_threads(args.worker_num_threads)
        action_seed = derive_action_seed(base_args.seed, worker_id)
        seed_process(action_seed)
        env = PureMyEnv(base_args=base_args, madrl_args=args, seed_offset=args.scene_seed_offset)
        actor = MultiHeadActor(env.observation_space["bs"].shape[0], args.hidden_dim,
                               env.bs_continuous_action_splits,
                               env.action_space["bs"]["discrete"].nvec,
                               continuous_dist_type="beta")
        normalizer = Normalization(env.observation_space["bs"].shape[0])
        scaler = RewardScaling(shape=1, gamma=args.gamma)
        while True:
            command = control.get()
            if command is None:
                return
            version = command["version"]
            actor.load_state_dict({key: torch.from_numpy(value) for key, value in command["actor"].items()})
            signature = actor_state_signature(actor)
            if signature != command["signature"]:
                raise RuntimeError("worker actor signature does not match dispatched weights")
            snapshot = command["normalizer"]
            state = env.reset(version)
            raw = np.asarray(state["bs"], dtype=np.float32)
            norm = normalizer(raw) if args.num_samplers == 1 else frozen_normalize(raw, snapshot)
            scaler.reset()
            transition = {key: [] for key in TRANSITION_KEYS}
            raw_states, totals, objectives, successes, trajectory, solver_statuses, traces = [raw.copy()], [], [], [], [], [], []
            detailed_traces = []
            done = False
            while not done:
                with torch.no_grad():
                    sample = actor.sample(torch.as_tensor(norm, dtype=torch.float32).unsqueeze(0))
                action = {"continuous": sample[0].squeeze(0).numpy().astype(np.float32),
                          "discrete": sample[1].squeeze(0).numpy().astype(np.int64)}
                trajectory.append(np.asarray(env.getPosUAV(), dtype=float).copy())
                next_state, total, reward, done, objective, success = env.step({"bs": action}, version)
                detailed_traces.append(capture(env, reward, total, objective, success, done, args))
                raw_next = np.asarray(next_state["bs"], dtype=np.float32)
                scaled_reward = float(np.asarray(scaler(float(reward["bs"]))).item())
                norm_next = normalizer(raw_next) if args.num_samplers == 1 else frozen_normalize(raw_next, snapshot)
                transition["states"].append(norm)
                transition["continuous_actions"].append(action["continuous"])
                transition["discrete_actions"].append(action["discrete"])
                transition["next_states"].append(norm_next)
                transition["rewards"].append(scaled_reward)
                transition["old_cont_log_probs"].append(float(sample[3].item()))
                transition["old_disc_log_probs"].append(float(sample[4].item()))
                transition["dones"].append(bool(done))
                transition["real_dones"].append(bool(done))
                raw_states.append(raw_next.copy())
                totals.append(float(total))
                objectives.append(float(objective))
                successes.append(float(success))
                solver_statuses.append(str(reward.get("components", {}).get("solver_status_kind", "unknown")))
                traces.append(env.last_action_trace)
                norm = norm_next
            results.put({"worker_id": worker_id, "version": version, "signature": signature,
                         "transition": transition, "raw_states": raw_states, "traces": traces,
                         "normalizer": snapshot_normalizer(normalizer) if args.num_samplers == 1 else None,
                         "reward": float(np.mean(totals)), "objectives": objectives,
                         "successes": successes, "solver_statuses": solver_statuses, "trajectory": np.asarray(trajectory),
                         "detailed_traces": detailed_traces, "scene_signature": env.get_exogenous_scene_signature(),
                         "action_seed": int(action_seed)})
    except BaseException:
        results.put({"worker_id": worker_id, "error": traceback.format_exc()})


def collect_round(results, processes, version, expected_signature):
    episodes = {}
    while len(episodes) < len(processes):
        try:
            item = results.get(timeout=1.0)
        except queue.Empty:
            failed = [(p.name, p.exitcode) for p in processes if not p.is_alive()]
            if failed:
                raise RuntimeError(f"sampler exited before completing round: {failed}")
            continue
        if "error" in item:
            raise RuntimeError(f"worker {item['worker_id']} failed:\n{item['error']}")
        wid = item["worker_id"]
        if wid in episodes or not 0 <= wid < len(processes):
            raise RuntimeError(f"unexpected or duplicate worker {wid}")
        if item["version"] != version or item["signature"] != expected_signature:
            raise RuntimeError("rollout policy version/signature mismatch")
        transition = item["transition"]
        lengths = {len(transition[key]) for key in TRANSITION_KEYS}
        if len(lengths) != 1 or not lengths or next(iter(lengths)) < 2:
            raise RuntimeError(f"invalid transition lengths: {lengths}")
        if not transition["dones"][-1] or any(transition["dones"][:-1]):
            raise RuntimeError("worker must return one complete episode")
        episodes[wid] = item
    return [episodes[i] for i in range(len(processes))]


def merge_episodes(episodes):
    return {key: [value for episode in episodes for value in episode["transition"][key]]
            for key in TRANSITION_KEYS}


@atomic_checkpoint
def save_checkpoint(output, version, agent, norm, episodes, config):
    path = output / f"iter_{version}"
    path.mkdir(parents=True, exist_ok=True)
    actor_path = path / "pure_mhbppo_actor.pth"
    torch.save(agent.actor.state_dict(), actor_path)
    torch.save(agent.critic.state_dict(), path / "pure_mhbppo_critic.pth")
    actor_hash = hashlib.sha256(actor_path.read_bytes()).hexdigest()
    np.savez(path / "state_normalization.npz", mean=norm["mean"], std=norm["std"],
             S=norm["S"], count=norm["n"], epsilon=1e-8, iteration=version,
             actor_sha256=actor_hash)
    for episode in episodes:
        np.save(path / f"uav_trajectory_worker_{episode['worker_id']}.npy", episode["trajectory"])
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        trajectory = episode["trajectory"]
        for index in range(trajectory.shape[1]):
            ax.plot(trajectory[:, index, 0], trajectory[:, index, 1], label=f"UAV {index}")
        ax.set(xlabel="x (m)", ylabel="y (m)", title=f"Training rollout: policy v{version - 1}")
        ax.legend()
        fig.savefig(path / f"uav_trajectory_worker_{episode['worker_id']}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
    metadata = {**config, "iteration": version, "weights_version": version,
                "trajectory_weights_version": version - 1, "actor_sha256": actor_hash,
                "checkpoint_type": "evaluation_only_no_optimizer_or_rng"}
    (path / "checkpoint_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    shutil.copyfile(output / "training_updates.csv", path / "training_updates.csv")


def run_training(args, *, output_dir=None):
    configure_ppo_lr_horizon(args)
    if not args.sync_training:
        raise ValueError("Pure-MHBPPO currently supports synchronous sampling only")
    if args.bc_dataset:
        raise ValueError("Pure-MHBPPO does not support MHBPPO behavior cloning")
    if (args.batch_size != 32 or args.continuous_entropy_coef != 0.05 or
            args.discrete_entropy_coef != 0.01):
        raise ValueError("legacy Pure-MHBPPO fixes mini-batch=32, entropy coefficients=0.05/0.01, MHBPPO-only mini-batch and entropy overrides are unsupported")
    base_kw = {"seed": int(args.seed), "target_layout": args.target_layout}
    if args.uavs_num is not None:
        base_kw["uavs_num"] = args.uavs_num
    base = get_base_args(**base_kw)
    seed_process(base.seed)
    probe = PureMyEnv(base_args=base, madrl_args=args, seed_offset=args.scene_seed_offset)
    base = probe.base_args
    state_dim = probe.observation_space["bs"].shape[0]
    splits = dict(probe.bs_continuous_action_splits)
    continuous = probe.action_space["bs"]["continuous"]
    discrete = probe.action_space["bs"]["discrete"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = HPPO(state_dim, args.hidden_dim, splits, discrete.nvec, continuous.low, continuous.high,
                 args.actor_lr, args.critic_lr, args.lmbda, args.eps, args.gamma, args.epochs,
                 args.episodes, device, continuous_dist_type="beta", lr_min_ratio=args.lr_min_ratio)
    normalizer = Normalization(state_dim)
    output, log_dir, tb_dir, run_paths, expected_scene = setup_run(_OUTER_ROOT, "Pure_MHBPPO", "Pure_MHBPPO", args, probe, output_dir)
    config = {"algorithm": "Pure-MHBPPO", "hyperparameters": {k: v for k, v in vars(args).items() if k not in ("config", "analytic_receive_beam_only")}, "state_dim": state_dim,
              "backend": "direct_no_optimizer", "receive_beam": "analytic_mvdr",
              "sensing_transmit_beam": "analytic_target_steering",
              "sensing_sinr_target_ratio": SENSING_SINR_TARGET_RATIO,
              "reward_contract": "actual_candidate_energy_and_constraint_penalty_v1",
              "reference_text_sha256": "2d464e85714512582961fcc5d6c711dfe17d433178d24c37a99a2bcb6d9fc1cf",
              "source_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in {
                  "trainer": Path(__file__), "agent": Path(__file__).with_name("Pure_MHBPPO_agent.py"),
                  "physical": Path(__file__).with_name("physical.py"), "pure_env": Path(__file__).with_name("pure_env.py"),
                  "current_env": _OUTER_ROOT / "environment" / "my_env.py"}.items()},
              "initial_exogenous_scene": probe.get_exogenous_scene_signature(),
              "continuous_action_splits": splits, "discrete_action_dims": discrete.nvec.tolist(),
              "action_seeds": [int(derive_action_seed(base.seed, i)) for i in range(args.num_samplers)],
              "scene_seed_offset": args.scene_seed_offset,
              "normalization": "legacy_online" if args.num_samplers == 1 else "frozen_snapshot_per_round",
              "reward_scaling": "legacy_per_worker_persistent_stats_reset_return_each_episode",
              "effective_mini_batch_size": 32, "continuous_entropy_coef": 0.05,
              "discrete_entropy_coef": 0.01, "lr_schedule": "linear_with_floor",
              "checkpoint_note": "evaluation only; no optimizer, RNG, or scaler state"}
    config["run_paths"] = run_paths
    (output / "configuration.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[Pure-MHBPPO] backend=direct_no_optimizer sensing_transmit_beam=analytic_target_steering receive_beam=analytic_mvdr", flush=True)
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=str(tb_dir))
    detailed = TraceLogs(log_dir, Path(run_paths["training_log"]).stem, {"algorithm": config["algorithm"], **run_paths, "log_detail_level": args.log_detail_level, "initial_exogenous_scene": expected_scene}, args)
    trace_writer = PhysicalTraceWriter(str(log_dir / "physical_traces.jsonl"),
        {"algorithm": "Pure-MHBPPO", "log_detail_level": "full", "backend": "direct_no_optimizer"})
    trace_rotation = JsonlRotation(args.log_rotate_iterations, args.log_rotate_max_bytes, args.log_rotate_backup_segments)
    trace_paused = False
    context = mp.get_context("spawn")
    results = context.Queue(maxsize=args.num_samplers)
    controls = [context.Queue(maxsize=1) for _ in range(args.num_samplers)]
    workers = [context.Process(target=worker_main, args=(base, args, i, controls[i], results),
                               name=f"pure-mhbppo-sampler-{i}") for i in range(args.num_samplers)]
    completed, last_episodes, env_steps, checkpoint_ready = 0, None, 0, False
    try:
        for worker in workers:
            worker.start()
        with (output / "training_updates.csv").open("w", newline="", encoding="utf-8") as stream, \
                (output / "episode_metadata.jsonl").open("w", encoding="utf-8") as episode_stream, \
                training_log_stream(run_paths["training_log"]) as log_stream:
            columns = ["iteration", "env_steps", "episodes", "reward", "completion_rate", "finite_obj_mean", "finite_obj_slots", "policy_version", "elapsed_seconds", "solver_status_counts"]
            columns += METRIC_COLUMNS
            csv_writer = csv.DictWriter(stream, fieldnames=columns)
            csv_writer.writeheader()
            for version in training_rounds(args.max_updates):
                started = time.monotonic()
                actor = {key: value.detach().cpu().numpy().copy() for key, value in agent.actor.state_dict().items()}
                signature = actor_state_signature(agent.actor)
                norm = snapshot_normalizer(normalizer)
                command = {"version": version, "actor": actor, "signature": signature, "normalizer": norm}
                for control in controls:
                    control.put(command)
                episodes = collect_round(results, workers, version, signature)
                log_event = detailed.begin_round(version + 1)
                if log_event:
                    log_stream.write(log_event + "\n"); log_stream.flush()
                for ep in episodes:
                    check_scene(expected_scene, ep["scene_signature"], args)
                    for slot, (action_trace, diagnostic_trace) in enumerate(ep["detailed_traces"]):
                        transition = ep["transition"]
                        raw = ({"latent": transition["actions"][slot]} if "actions" in transition else
                               {"continuous": transition["continuous_actions"][slot], "discrete": transition["discrete_actions"][slot]})
                        detailed.record({"iteration": version + 1, "worker_id": ep["worker_id"], "episode": version,
                                         "slot": slot, "policy_version": version, "actor_signature": ep["signature"],
                                         "scene_signature": ep["scene_signature"]["signature"], "raw_action": raw},
                                        action_trace, diagnostic_trace)
                detailed.flush()
                paused = bool(args.log_min_free_bytes and shutil.disk_usage(log_dir).free < args.log_min_free_bytes)
                rotate_index = 0
                if paused and not trace_paused and trace_rotation.active:
                    rotate_index = trace_rotation.force(version + 1)
                elif not paused and trace_rotation.due(version + 1, [trace_writer.size]):
                    rotate_index = trace_rotation.segment
                if rotate_index:
                    trace_writer.rotate(rotate_index)
                    trace_writer.prune(rotate_index + 1 - trace_rotation.keep_segments)
                if paused != trace_paused:
                    print("[Pure-MHBPPO] physical trace logging " + ("paused: low disk space" if paused else "resumed"), flush=True)
                trace_paused = paused
                if not paused:
                    for ep in episodes:
                        for trace in ep["traces"]:
                            trace_writer.write_trace({"iteration": version + 1,
                                "worker_id": ep["worker_id"], "policy_version": version, **trace})
                for ep in episodes:
                    episode_stream.write(json.dumps({"iteration": version + 1, "worker_id": ep["worker_id"],
                        "policy_version": ep["version"], "actor_signature": ep["signature"],
                        "action_seed": ep["action_seed"], "scene": ep["scene_signature"],
                        "steps": len(ep["transition"]["rewards"])}, ensure_ascii=False) + "\n")
                episode_stream.flush()
                transition = merge_episodes(episodes)
                checkpoint_ready = False
                agent.update(transition, version, writer, agent_name="BS")
                if args.num_samplers == 1:
                    state_stats = episodes[0]["normalizer"]
                    stats = normalizer.running_ms
                    stats.n, stats.mean, stats.std, stats.S = (state_stats[key] for key in ("n", "mean", "std", "S"))
                else:
                    for episode in episodes:
                        for raw in episode["raw_states"]:
                            normalizer.running_ms.update(raw)
                completed = version + 1
                last_episodes = episodes
                count = len(transition["rewards"])
                env_steps += count
                reward = sum(ep["reward"] * len(ep["transition"]["rewards"]) for ep in episodes) / count
                completion = 100.0 * sum(sum(ep["successes"]) for ep in episodes) / count
                solver_counts = {}
                for ep in episodes:
                    for status in ep["solver_statuses"]:
                        solver_counts[status] = solver_counts.get(status, 0) + 1
                finite_objectives = [value for ep in episodes for value in ep["objectives"] if np.isfinite(value)]
                row = dict(iteration=completed, env_steps=env_steps, episodes=completed * args.num_samplers,
                           reward=reward, completion_rate=completion,
                           finite_obj_mean=float(np.mean(finite_objectives)) if finite_objectives else "",
                           finite_obj_slots=len(finite_objectives), policy_version=version,
                           elapsed_seconds=time.monotonic() - started,
                           solver_status_counts=json.dumps(solver_counts, sort_keys=True))
                metrics = aggregate_metrics([pair for ep in episodes for pair in ep["detailed_traces"]])
                row.update(metrics)
                log_scalar_metrics(writer, metrics, completed)
                csv_writer.writerow(row)
                stream.flush()
                writer.add_scalar("Reward/iteration", reward, completed)
                writer.add_scalar("Completion Rate/iteration", completion, completed)
                message = f"[Pure-MHBPPO] iter={completed} env_steps={env_steps} reward={reward:.4f} physical_success={completion:.2f}% backend={solver_counts}"
                print(message, flush=True)
                log_stream.write(message + "\n")
                log_stream.flush()
                checkpoint_ready = True
                if args.checkpoint_interval > 0 and completed % args.checkpoint_interval == 0:
                    save_checkpoint(output, completed, agent, snapshot_normalizer(normalizer), episodes, config)
    except BaseException:
        log_training_failure(run_paths["training_log"], traceback.format_exc())
        raise
    finally:
        for control in controls:
            try:
                control.put_nowait(None)
            except queue.Full:
                pass
        for worker in workers:
            if worker.pid is not None:
                worker.join(timeout=5)
                if worker.is_alive():
                    worker.terminate()
                    worker.join(timeout=5)
        if completed and last_episodes and checkpoint_ready:
            save_checkpoint(output, completed, agent, snapshot_normalizer(normalizer), last_episodes, config)
        trace_writer.close()
        detailed.close()
        writer.close()
        for q in [*controls, results]:
            q.cancel_join_thread()
            q.close()
    return output


def get_pure_args(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = get_madrl_args(arguments)
    specified = any(arg == "--episodes" or arg.startswith("--episodes=") for arg in arguments)
    configured = False
    if args.config:
        payload = json.loads(Path(args.config).read_text(encoding="utf-8"))
        configured = "episodes" in payload.get("hyperparameters", payload)
    if not specified and not configured:
        args.episodes = 8000
    return args


if __name__ == "__main__":
    mp.freeze_support()
    run_training(get_pure_args())
