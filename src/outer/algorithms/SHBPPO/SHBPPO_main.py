"""Legacy single-head Beta PPO on the current environment, with synchronous workers.

The policy and normalization modules derive from outer_old; PPO learning rates
use a floor for unlimited training. One worker
retains the original online normalization order. Multiple workers use a frozen
state-normalization snapshot for each collection round. Workers return raw
rewards; the learner standardizes them using MHBPPO's global historical stats.
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

from algorithms.SHBPPO.SHBPPO_agent import SHBPPO, SingleHeadActor
from algorithms.SHBPPO.normalization import Normalization
from algorithms.MHBPPO.MHBPPO_main import StateNormStats
from algorithms.MHBPPO.rollout_batch import derive_action_seed, actor_state_signature
from algorithms.baseline_output import setup_run, check_scene, TraceLogs, atomic_checkpoint, capture, aggregate_metrics, METRIC_COLUMNS, log_scalar_metrics, training_log_stream, log_training_failure
from algorithms.baseline_output import TrainingStop, save_reward_artifacts
from algorithms.training_schedule import configure_ppo_lr_horizon, training_rounds
from configs.base_params import get_base_args
from configs.madrl_args import get_madrl_args
from environment.my_env import MyEnv


TRANSITION_KEYS = ("states", "actions", "next_states", "rewards", "old_log_probs", "dones", "real_dones")


def split_and_discretize(raw_action, continuous_dim, uavs_num, cus_num):
    """The original SHBPPO mapping; the raw vector is retained for PPO."""
    raw_action = np.asarray(raw_action, dtype=np.float32)
    continuous_part = raw_action[:continuous_dim]
    discrete_part = raw_action[continuous_dim:continuous_dim + uavs_num]
    cu_indices = np.minimum((discrete_part * cus_num).astype(np.int64), cus_num - 1)
    return continuous_part.astype(np.float32), cu_indices.astype(np.int64)


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
    """Collect raw rewards; reward normalization belongs to the learner."""
    try:
        TrainingStop.ignore_worker_interrupts()
        if args.worker_num_threads:
            torch.set_num_threads(args.worker_num_threads)
        action_seed = derive_action_seed(base_args.seed, worker_id)
        seed_process(action_seed)
        env = MyEnv(base_args=base_args, madrl_args=args, seed_offset=args.scene_seed_offset)
        continuous_dim = sum(env.bs_continuous_action_splits.values())
        actor = SingleHeadActor(env.observation_space["bs"].shape[0], args.hidden_dim,
                                continuous_dim + base_args.uavs_num)
        normalizer = Normalization(env.observation_space["bs"].shape[0])
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
            transition = {key: [] for key in TRANSITION_KEYS}
            raw_states, totals, objectives, successes, trajectory, solver_statuses = [raw.copy()], [], [], [], [], []
            detailed_traces = []
            done = False
            while not done:
                with torch.no_grad():
                    sample = actor.sample(torch.as_tensor(norm, dtype=torch.float32).unsqueeze(0))
                raw_action = sample[0].squeeze(0).numpy().astype(np.float32)
                continuous, discrete = split_and_discretize(raw_action, continuous_dim,
                                                            base_args.uavs_num, base_args.cus_num)
                action = {"continuous": continuous, "discrete": discrete}
                trajectory.append(np.asarray(env.getPosUAV(), dtype=float).copy())
                next_state, total, reward, done, objective, success = env.step({"bs": action}, version)
                detailed_traces.append(capture(env, reward, total, objective, success, done, args))
                raw_next = np.asarray(next_state["bs"], dtype=np.float32)
                norm_next = normalizer(raw_next) if args.num_samplers == 1 else frozen_normalize(raw_next, snapshot)
                transition["states"].append(norm)
                transition["actions"].append(raw_action)
                transition["next_states"].append(norm_next)
                transition["rewards"].append(float(reward["bs"]))
                transition["old_log_probs"].append(float(sample[1].item()))
                transition["dones"].append(bool(done))
                transition["real_dones"].append(bool(done))
                raw_states.append(raw_next.copy())
                totals.append(float(total))
                objectives.append(float(objective))
                successes.append(float(success))
                solver_statuses.append(str(reward.get("components", {}).get("solver_status_kind", "unknown")))
                norm = norm_next
            results.put({"worker_id": worker_id, "version": version, "signature": signature,
                         "transition": transition, "raw_states": raw_states,
                         "normalizer": snapshot_normalizer(normalizer) if args.num_samplers == 1 else None,
                         "reward": float(np.mean(totals)), "objectives": objectives,
                         "successes": successes, "solver_statuses": solver_statuses, "trajectory": np.asarray(trajectory),
                         "detailed_traces": detailed_traces, "scene_signature": env.get_exogenous_scene_signature(),
                         "action_seed": int(action_seed)})
    except BaseException:
        results.put({"worker_id": worker_id, "error": traceback.format_exc()})


def collect_round(results, processes, version, expected_signature, should_stop=lambda: False):
    episodes = {}
    while len(episodes) < len(processes):
        if should_stop():
            return None
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
    actor_path = path / "shbppo_actor.pth"
    torch.save(agent.actor.state_dict(), actor_path)
    torch.save(agent.critic.state_dict(), path / "shbppo_critic.pth")
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
    save_reward_artifacts(output, config["hyperparameters"]["seed"], path, algorithm=config["algorithm"])


def run_training(args, *, output_dir=None):
    configure_ppo_lr_horizon(args)
    if not args.sync_training:
        raise ValueError("SHBPPO currently supports synchronous sampling only")
    if args.bc_dataset:
        raise ValueError("SHBPPO does not support MHBPPO behavior cloning")
    if (args.batch_size != 32 or args.continuous_entropy_coef != 0.05 or
            args.discrete_entropy_coef != 0.01):
        raise ValueError("legacy SHBPPO fixes mini-batch=32, entropy coefficient=0.01, MHBPPO-only mini-batch and entropy overrides are unsupported")
    base_kw = {"seed": int(args.seed), "target_layout": args.target_layout}
    if args.uavs_num is not None:
        base_kw["uavs_num"] = args.uavs_num
    base = get_base_args(**base_kw)
    seed_process(base.seed)
    probe = MyEnv(base_args=base, madrl_args=args, seed_offset=args.scene_seed_offset)
    base = probe.base_args
    state_dim = probe.observation_space["bs"].shape[0]
    splits = dict(probe.bs_continuous_action_splits)
    continuous = probe.action_space["bs"]["continuous"]
    discrete = probe.action_space["bs"]["discrete"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if len(discrete.nvec) != base.uavs_num or not np.all(discrete.nvec == base.cus_num):
        raise ValueError("legacy SHBPPO requires one CU index with cus_num choices per UAV")
    continuous_dim = sum(splits.values())
    agent = SHBPPO(state_dim, args.hidden_dim, continuous_dim + base.uavs_num,
                   continuous.low, continuous.high, args.actor_lr, args.critic_lr,
                   args.lmbda, args.eps, args.gamma, args.epochs, args.episodes, device,
                   entropy_coef=0.01, lr_min_ratio=args.lr_min_ratio,
                   uavs_num=base.uavs_num, cus_num=base.cus_num,
                   expected_duplicate_matching_coef=args.expected_duplicate_matching_coef)
    normalizer = Normalization(state_dim)
    reward_norm_stats = StateNormStats(1)
    output, log_dir, tb_dir, run_paths, expected_scene = setup_run(_OUTER_ROOT, "SHBPPO", "SHBPPO", args, probe, output_dir)
    config = {"algorithm": "SHBPPO", "receive_beam": "analytic_mvdr_no_policy_actions",
              "hyperparameters": {k: v for k, v in vars(args).items() if k not in ("config", "analytic_receive_beam_only")}, "state_dim": state_dim,
              "initial_exogenous_scene": probe.get_exogenous_scene_signature(),
              "continuous_action_splits": splits, "discrete_action_dims": discrete.nvec.tolist(),
              "continuous_dim": continuous_dim, "total_action_dim": continuous_dim + base.uavs_num,
              "action_encoding": "legacy_single_beta_vector_with_cu_bins",
              "action_seeds": [int(derive_action_seed(base.seed, i)) for i in range(args.num_samplers)],
              "scene_seed_offset": args.scene_seed_offset,
              "normalization": "legacy_online" if args.num_samplers == 1 else "frozen_snapshot_per_round",
              "reward_scaling": "mhbppo_learner_global_history_standardization",
              "duplicate_matching_probability": "beta_cdf_cu_bin_mass",
              "effective_mini_batch_size": 32, "entropy_coef": 0.01, "lr_schedule": "linear_with_floor",
              "checkpoint_note": "evaluation only; no optimizer, RNG, or reward normalization state"}
    config["run_paths"] = run_paths
    (output / "configuration.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=str(tb_dir))
    detailed = TraceLogs(log_dir, Path(run_paths["training_log"]).stem, {"algorithm": config["algorithm"], **run_paths, "log_detail_level": args.log_detail_level, "initial_exogenous_scene": expected_scene}, args)
    context = mp.get_context("spawn")
    results = context.Queue(maxsize=args.num_samplers)
    controls = [context.Queue(maxsize=1) for _ in range(args.num_samplers)]
    workers = [context.Process(target=worker_main, args=(base, args, i, controls[i], results),
                               name=f"shbppo-sampler-{i}") for i in range(args.num_samplers)]
    completed, last_episodes, env_steps, checkpoint_ready = 0, None, 0, False
    stop = TrainingStop()
    try:
        stop.install()
        for worker in workers:
            worker.start()
        with (output / "training_updates.csv").open("w", newline="", encoding="utf-8") as stream, \
                (output / "episode_metadata.jsonl").open("w", encoding="utf-8") as episode_stream, \
                training_log_stream(run_paths["training_log"]) as log_stream:
            columns = ["iteration", "env_steps", "episodes", "reward", "completion_rate", "finite_obj_mean", "finite_obj_slots", "policy_version", "elapsed_seconds", "solver_status_counts", "actor_expected_duplicate_matching"]
            columns += METRIC_COLUMNS
            csv_writer = csv.DictWriter(stream, fieldnames=columns)
            csv_writer.writeheader()
            for version in training_rounds(args.max_updates):
                if stop.requested:
                    break
                started = time.monotonic()
                actor = {key: value.detach().cpu().numpy().copy() for key, value in agent.actor.state_dict().items()}
                signature = actor_state_signature(agent.actor)
                norm = snapshot_normalizer(normalizer)
                command = {"version": version, "actor": actor, "signature": signature, "normalizer": norm}
                for control in controls:
                    control.put(command)
                episodes = collect_round(results, workers, version, signature, should_stop=lambda: stop.requested)
                if episodes is None or stop.requested:
                    break
                log_event = detailed.begin_round(version + 1)
                if log_event:
                    log_stream.write(log_event + "\n"); log_stream.flush()
                for ep in episodes:
                    check_scene(expected_scene, ep["scene_signature"], args)
                    for slot, (action_trace, diagnostic_trace) in enumerate(ep["detailed_traces"]):
                        transition = ep["transition"]
                        raw = ({"latent": transition["actions"][slot], "discrete": split_and_discretize(transition["actions"][slot], continuous_dim, base.uavs_num, base.cus_num)[1]} if "actions" in transition else
                               {"continuous": transition["continuous_actions"][slot], "discrete": transition["discrete_actions"][slot]})
                        detailed.record({"iteration": version + 1, "worker_id": ep["worker_id"], "episode": version,
                                         "slot": slot, "policy_version": version, "actor_signature": ep["signature"],
                                         "scene_signature": ep["scene_signature"]["signature"], "raw_action": raw},
                                        action_trace, diagnostic_trace)
                detailed.flush()
                for ep in episodes:
                    episode_stream.write(json.dumps({"iteration": version + 1, "worker_id": ep["worker_id"],
                        "policy_version": ep["version"], "actor_signature": ep["signature"],
                        "action_seed": ep["action_seed"], "scene": ep["scene_signature"],
                        "steps": len(ep["transition"]["rewards"])}, ensure_ascii=False) + "\n")
                episode_stream.flush()
                transition = merge_episodes(episodes)
                raw_rewards = np.asarray(transition["rewards"], dtype=np.float64).reshape(-1, 1)
                transition["rewards"] = (
                    (raw_rewards - reward_norm_stats.mean) / (reward_norm_stats.std + 1e-8)
                ).reshape(-1).tolist()
                checkpoint_ready = False
                expected_duplicates = agent.update(transition, version, writer, agent_name="BS")
                reward_norm_stats.update(raw_rewards)
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
                           solver_status_counts=json.dumps(solver_counts, sort_keys=True),
                           actor_expected_duplicate_matching=expected_duplicates)
                metrics = aggregate_metrics([pair for ep in episodes for pair in ep["detailed_traces"]])
                row.update(metrics)
                log_scalar_metrics(writer, metrics, completed)
                csv_writer.writerow(row)
                stream.flush()
                writer.add_scalar("Reward/iteration", reward, completed)
                writer.add_scalar("Completion Rate/iteration", completion, completed)
                message = f"[SHBPPO] iter={completed} env_steps={env_steps} reward={reward:.4f} completion={completion:.2f}% solver={solver_counts}"
                print(message, flush=True)
                log_stream.write(message + "\n")
                log_stream.flush()
                checkpoint_ready = True
                if args.checkpoint_interval > 0 and completed % args.checkpoint_interval == 0:
                    save_checkpoint(output, completed, agent, snapshot_normalizer(normalizer), episodes, config)
    except KeyboardInterrupt:
        print("[SHBPPO] interrupted; saving completed training results", flush=True)
    except BaseException:
        log_training_failure(run_paths["training_log"], traceback.format_exc())
        raise
    finally:
        try:
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
            save_reward_artifacts(output, base.seed, algorithm=config["algorithm"])
            if completed and last_episodes and checkpoint_ready:
                save_checkpoint(output, completed, agent, snapshot_normalizer(normalizer), last_episodes, config)
        finally:
            try:
                detailed.close()
                writer.close()
                for q in [*controls, results]:
                    q.cancel_join_thread()
                    q.close()
            finally:
                stop.restore()
    return output


def get_shbppo_args(argv=None):
    """Use the old 8000-step learning-rate horizon unless explicitly configured."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = get_madrl_args(arguments)
    cli_has_episodes = any(arg == "--episodes" or arg.startswith("--episodes=") for arg in arguments)
    config_has_episodes = False
    if args.config:
        payload = json.loads(Path(args.config).read_text(encoding="utf-8"))
        settings = payload.get("hyperparameters", payload)
        config_has_episodes = "episodes" in settings
    if not cli_has_episodes and not config_has_episodes:
        args.episodes = 8000
    return args


if __name__ == "__main__":
    mp.freeze_support()
    run_training(get_shbppo_args())
