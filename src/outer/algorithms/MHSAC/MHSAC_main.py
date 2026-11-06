"""Legacy hybrid SAC with synchronous parallel environment steps.

Only environments run in workers. The learner owns the original actor, twin Q,
replay buffer and running normalization. It processes each transition and its
configured number of SAC updates before processing the next transition.
"""
import argparse
import csv
import hashlib
import json
import multiprocessing as mp
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

from algorithms.MHSAC.MHSAC_agent import HSAC, ReplayBuffer
from algorithms.MHSAC.normalization import Normalization, RewardScaling
from algorithms.MHBPPO.rollout_batch import derive_action_seed, actor_state_signature
from algorithms.baseline_output import setup_run, check_scene, TraceLogs, atomic_checkpoint, capture, aggregate_metrics, METRIC_COLUMNS, log_scalar_metrics, training_log_stream, log_training_failure
from algorithms.baseline_output import TrainingStop, save_reward_artifacts
from algorithms.training_schedule import training_rounds
from configs.base_params import get_base_args
from configs.madrl_args import get_madrl_args
from environment.my_env import MyEnv


def get_all_args(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default="")
    config_arg, _ = config_parser.parse_known_args(arguments)
    saved = {}
    if config_arg.config:
        saved = json.loads(Path(config_arg.config).read_text(encoding="utf-8"))
    sac_parser = argparse.ArgumentParser(add_help=False, description="MHSAC parameters")
    sac_parser.add_argument("--tau", type=float, default=0.005)
    sac_parser.add_argument("--buffer_size", type=int, default=100000)
    sac_parser.add_argument("--minimal_size", type=int, default=0)
    sac_parser.add_argument("--updates_per_step", type=int, default=1)
    sac_parser.add_argument("--alpha_continuous", type=float, default=0.2)
    sac_parser.add_argument("--alpha_discrete", type=float, default=0.2)
    sac_parser.add_argument("--target_continuous_log_prob", type=float, default=0.3)
    sac_parser.add_argument("--target_discrete_entropy_ratio", type=float, default=0.48)
    sac_parser.add_argument("--alpha_lr", type=float, default=3e-5)
    sac_parser.add_argument("--alpha_min", type=float, default=1e-4)
    sac_parser.add_argument("--alpha_max", type=float, default=5.0)
    saved_sac = saved.get("sac_hyperparameters", {})
    if not isinstance(saved_sac, dict):
        raise ValueError("sac_hyperparameters must be a JSON object")
    known_sac = {action.dest for action in sac_parser._actions}
    if set(saved_sac) - known_sac:
        raise ValueError("unknown sac_hyperparameters: " + ", ".join(sorted(set(saved_sac) - known_sac)))
    for action in sac_parser._actions:
        if action.dest in saved_sac:
            value = saved_sac[action.dest]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(action.dest + " must be a JSON number")
            if action.type is int and not isinstance(value, int):
                raise ValueError(action.dest + " must be a JSON integer")
    sac_parser.set_defaults(**saved_sac)
    sac, remaining = sac_parser.parse_known_args(arguments)
    if "--help" in remaining or "-h" in remaining:
        sac_parser.print_help()
    args = get_madrl_args(remaining)
    configured = saved.get("hyperparameters", saved) if isinstance(saved, dict) else {}
    for name, old_default in (("batch_size", 128), ("episodes", 8000)):
        cli_set = any(item == "--" + name or item.startswith("--" + name + "=") for item in arguments)
        if not cli_set and name not in configured:
            setattr(args, name, old_default)
    if sac.minimal_size <= 0:
        sac.minimal_size = max(4096, 200 * args.total_time_slots)
    if sac.buffer_size <= 0 or args.batch_size <= 0 or sac.updates_per_step <= 0:
        raise ValueError("buffer_size, batch_size and updates_per_step must be positive")
    if max(sac.minimal_size, args.batch_size) > sac.buffer_size:
        raise ValueError("replay warm-up threshold must not exceed buffer_size")
    if not (0 < sac.tau <= 1 and 0 < sac.alpha_min <= sac.alpha_max and
            sac.alpha_lr > 0 and sac.alpha_continuous > 0 and sac.alpha_discrete > 0):
        raise ValueError("invalid SAC tau, alpha or alpha bounds")
    if not all(np.isfinite(value) for value in vars(sac).values()):
        raise ValueError("SAC parameters must be finite")
    return args, sac


def seed_process(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_main(base, args, worker_id, control, results):
    """One persistent MyEnv; all actor inference and SAC updates are learner-side."""
    try:
        TrainingStop.ignore_worker_interrupts()
        if args.worker_num_threads:
            torch.set_num_threads(args.worker_num_threads)
        env = MyEnv(base_args=base, madrl_args=args, seed_offset=args.scene_seed_offset)
        while True:
            command = control.get()
            if command is None:
                return
            if command["op"] == "reset":
                state = env.reset(command["episode"])
                results.put({"worker_id": worker_id, "op": "reset", "episode": command["episode"],
                             "raw_state": np.asarray(state["bs"], dtype=np.float32),
                             "scene": env.get_exogenous_scene_signature()})
            elif command["op"] == "step":
                position = np.asarray(env.getPosUAV(), dtype=float).copy()
                state, total, reward, done, obj, success = env.step({"bs": command["action"]}, command["episode"])
                action_trace, diagnostic_trace = capture(env, reward, total, obj, success, done, args)
                results.put({"worker_id": worker_id, "op": "step", "episode": command["episode"],
                             "slot": command["slot"], "raw_next": np.asarray(state["bs"], dtype=np.float32),
                             "reward": float(reward["bs"]), "total": float(total), "done": bool(done),
                             "objective": float(obj), "success": int(success), "position": position,
                             "solver_status": str(reward.get("components", {}).get("solver_status_kind", "unknown")),
                             "action_trace": action_trace, "diagnostic_trace": diagnostic_trace})
            else:
                raise ValueError("unexpected worker command")
    except BaseException:
        results.put({"worker_id": worker_id, "error": traceback.format_exc()})


def collect_batch(results, workers, op, episode, slot=None):
    items = {}
    while len(items) < len(workers):
        try:
            item = results.get(timeout=1)
        except queue.Empty:
            failed = [(p.name, p.exitcode) for p in workers if not p.is_alive()]
            if failed:
                raise RuntimeError(f"sampler exited unexpectedly: {failed}")
            continue
        if "error" in item:
            raise RuntimeError(f"worker {item['worker_id']} failed:\n{item['error']}")
        worker_id = item["worker_id"]
        if (not 0 <= worker_id < len(workers) or worker_id in items or item["op"] != op
                or item["episode"] != episode or (op == "step" and item["slot"] != slot)):
            raise RuntimeError("worker result identity or slot mismatch")
        items[worker_id] = item
    return [items[i] for i in range(len(workers))]


class ActionRandomStreams:
    """Independent multi-worker action RNGs; one worker uses legacy learner RNG."""
    def __init__(self, seed, count, device):
        self.count, self.device = count, device
        self.seeds = [int(derive_action_seed(seed, i)) for i in range(count)] if count > 1 else [int(seed)]
        self.states = []
        if count > 1:
            devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
            for action_seed in self.seeds:
                with torch.random.fork_rng(devices=devices):
                    torch.manual_seed(action_seed)
                    cpu = torch.get_rng_state()
                    gpu = torch.cuda.get_rng_state(device) if devices else None
                    self.states.append((cpu, gpu))

    def choose(self, worker_id, agent, normalized_state):
        if self.count == 1:
            return agent.choose_action(normalized_state)
        devices = [self.device.index if self.device.index is not None else torch.cuda.current_device()] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            cpu, gpu = self.states[worker_id]
            torch.set_rng_state(cpu)
            if devices:
                torch.cuda.set_rng_state(gpu, self.device)
            action = agent.choose_action(normalized_state)
            self.states[worker_id] = (torch.get_rng_state(), torch.cuda.get_rng_state(self.device) if devices else None)
        return action


def learn_one(replay, agent, normalizer, args, sac, transition, writer, env_step):
    """Insert exactly one transition, then run legacy updates_per_step."""
    replay.add(**transition)
    updates = 0
    if replay.size() >= max(sac.minimal_size, args.batch_size):
        for _ in range(sac.updates_per_step):
            batch = replay.sample(args.batch_size)
            batch["states"] = np.asarray(normalizer(batch["states"], update=False), dtype=np.float32)
            batch["next_states"] = np.asarray(normalizer(batch["next_states"], update=False), dtype=np.float32)
            agent.update(batch, writer=writer, step=env_step, agent_name="BS")
            updates += 1
    return updates


@atomic_checkpoint
def save_checkpoint(output, iteration, env_steps, updates, agent, normalizer, episodes, config):
    path = output / f"iter_{iteration}"
    path.mkdir(parents=True, exist_ok=True)
    actor_path = path / "mhsac_actor.pth"
    torch.save(agent.actor.state_dict(), actor_path)
    for name in ("critic1", "critic2", "critic1_target", "critic2_target"):
        torch.save(getattr(agent, name).state_dict(), path / f"mhsac_{name}.pth")
    actor_hash = hashlib.sha256(actor_path.read_bytes()).hexdigest()
    stats = normalizer.running_ms
    np.savez(path / "state_normalization.npz", mean=stats.mean, std=stats.std, S=stats.S,
             count=stats.n, epsilon=1e-8, iteration=iteration, actor_sha256=actor_hash)
    np.savez(path / "temperatures.npz", log_alpha_continuous=agent.log_alpha_continuous.detach().cpu().item(),
             log_alpha_discrete=agent.log_alpha_discrete.detach().cpu().item())
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for index, episode in enumerate(episodes):
        trajectory = np.asarray(episode["trajectory"])
        np.save(path / f"uav_trajectory_worker_{index}.npy", trajectory)
        fig, ax = plt.subplots()
        for uav in range(trajectory.shape[1]):
            ax.plot(trajectory[:, uav, 0], trajectory[:, uav, 1], label=f"UAV {uav}")
        ax.set(xlabel="x (m)", ylabel="y (m)", title="Training rollout: policy may change per slot")
        ax.legend()
        fig.savefig(path / f"uav_trajectory_worker_{index}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
    metadata = {**config, "iteration": iteration, "env_steps": env_steps, "sac_updates": updates,
                "weights_version": updates, "actor_sha256": actor_hash,
                "trajectory_policy": "see per-slot policy versions in episode_metadata.jsonl",
                "checkpoint_type": "evaluation_only_no_optimizer_rng_or_replay"}
    (path / "checkpoint_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copyfile(output / "training_updates.csv", path / "training_updates.csv")
    save_reward_artifacts(output, config["hyperparameters"]["seed"], path, algorithm=config["algorithm"])


def run_training(args, sac, *, output_dir=None):
    if not args.sync_training:
        raise ValueError("MHSAC supports synchronous parallel environment steps only")
    if args.bc_dataset:
        raise ValueError("MHSAC does not support MHBPPO behavior cloning")
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
    joint_dim = int(np.prod(discrete.nvec, dtype=np.int64))
    if joint_dim <= 0:
        raise ValueError("invalid or overflowed joint discrete action dimension")
    print(f"[MHSAC] exact joint actions={joint_dim}; batch={args.batch_size}; "
          f"one float32 joint batch tensor={args.batch_size * joint_dim * 4 / 1024**2:.2f} MiB; "
          f"full replay state/action arrays~{sac.buffer_size * (2 * state_dim + sum(splits.values())) * 4 / 1024**2:.1f} MiB (excluding overhead)", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = HSAC(state_dim, args.hidden_dim, splits, discrete.nvec, continuous.low, continuous.high,
                 args.actor_lr, args.critic_lr, args.gamma, sac.tau, sac.alpha_continuous,
                 sac.alpha_discrete, device, target_continuous_log_prob=sac.target_continuous_log_prob,
                 target_discrete_entropy_ratio=sac.target_discrete_entropy_ratio, alpha_lr=sac.alpha_lr,
                 alpha_min=sac.alpha_min, alpha_max=sac.alpha_max)
    replay = ReplayBuffer(sac.buffer_size)
    normalizer = Normalization(state_dim)
    scalers = [RewardScaling(shape=1, gamma=args.gamma) for _ in range(args.num_samplers)]
    streams = ActionRandomStreams(base.seed, args.num_samplers, device)
    output, log_dir, tb_dir, run_paths, expected_scene = setup_run(_OUTER_ROOT, "MHSAC", "MHSAC", args, probe, output_dir)
    config = {"algorithm": "MHSAC", "receive_beam": "analytic_mvdr_no_policy_actions",
              "hyperparameters": {k: v for k, v in vars(args).items() if k not in ("config", "analytic_receive_beam_only")},
              "sac_hyperparameters": vars(sac), "state_dim": state_dim, "continuous_action_splits": splits,
              "discrete_action_dims": discrete.nvec.tolist(), "joint_action_dim": joint_dim,
              "initial_exogenous_scene": probe.get_exogenous_scene_signature(), "action_seeds": streams.seeds,
              "scene_seed_offset": args.scene_seed_offset, "normalization": "legacy_online_learner_worker_id_order",
              "reward_scaling": "legacy_per_worker_persistent_stats_reset_return_each_episode",
              "sampling": "synchronous_step_barrier_learner_policy_one_update_opportunity_per_transition",
              "checkpoint_note": "evaluation only; no optimizer, RNG, reward scaler or replay"}
    config["run_paths"] = run_paths
    (output / "configuration.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=str(tb_dir))
    detailed = TraceLogs(log_dir, Path(run_paths["training_log"]).stem, {"algorithm": "MHSAC", **run_paths, "log_detail_level": args.log_detail_level, "initial_exogenous_scene": expected_scene}, args)
    context = mp.get_context("spawn")
    results = context.Queue(maxsize=args.num_samplers)
    controls = [context.Queue(maxsize=1) for _ in range(args.num_samplers)]
    workers = [context.Process(target=worker_main, args=(base, args, i, controls[i], results),
                               name=f"mhsac-sampler-{i}") for i in range(args.num_samplers)]
    completed, last_episodes, env_steps, updates, checkpoint_ready = 0, None, 0, 0, False
    stop = TrainingStop()
    try:
        stop.install()
        for worker in workers:
            worker.start()
        with (output / "training_updates.csv").open("w", newline="", encoding="utf-8") as stream, \
                (output / "episode_metadata.jsonl").open("w", encoding="utf-8") as episode_stream, \
                training_log_stream(run_paths["training_log"]) as log_stream:
            columns = ["iteration", "env_steps", "episodes", "sac_updates", "replay_size", "reward",
                       "completion_rate", "finite_obj_mean", "finite_obj_slots", "max_policy_lag_updates",
                       "elapsed_seconds", "solver_status_counts"]
            columns += METRIC_COLUMNS
            csv_writer = csv.DictWriter(stream, fieldnames=columns)
            csv_writer.writeheader()
            for episode_idx in training_rounds(args.max_updates):
                # SAC changes weights and normalizer within a round; finish it before stopping.
                if stop.requested:
                    break
                # Resetting observations also changes the learner normalizer.
                # Do not save a prior-version actor with partially advanced stats
                # if this collection/update round is interrupted.
                checkpoint_ready = False
                started = time.monotonic()
                for q in controls:
                    q.put({"op": "reset", "episode": episode_idx})
                resets = collect_batch(results, workers, "reset", episode_idx)
                for item in resets:
                    check_scene(expected_scene, item["scene"], args)
                log_event = detailed.begin_round(episode_idx + 1)
                if log_event:
                    log_stream.write(log_event + "\n"); log_stream.flush()
                raw_states = [item["raw_state"] for item in resets]
                norm_states = [normalizer(raw) for raw in raw_states]
                for scaler in scalers:
                    scaler.reset()
                episodes = [{"trajectory": [], "rewards": [], "objectives": [], "successes": [],
                             "statuses": [], "policy_versions": [], "policy_signatures": [], "policy_lags": [], "detailed_traces": [],
                             "scene": resets[i]["scene"]} for i in range(args.num_samplers)]
                for slot in range(args.total_time_slots):
                    sampled_version = updates
                    sampled_signature = actor_state_signature(agent.actor)
                    actions = [streams.choose(i, agent, norm_states[i]) for i in range(args.num_samplers)]
                    for i, q in enumerate(controls):
                        q.put({"op": "step", "episode": episode_idx, "slot": slot,
                               "action": {"continuous": actions[i]["continuous"], "discrete": actions[i]["discrete"]}})
                    results_batch = collect_batch(results, workers, "step", episode_idx, slot)
                    checkpoint_ready = False
                    for i, item in enumerate(results_batch):
                        expected_done = slot == args.total_time_slots - 1
                        if item["done"] != expected_done:
                            raise RuntimeError("unexpected episode termination during synchronous step")
                        lag = updates - sampled_version
                        detailed.record({"iteration": episode_idx + 1, "worker_id": i, "episode": episode_idx,
                                         "slot": slot, "policy_version": sampled_version, "actor_signature": sampled_signature,
                                         "policy_lag_updates": lag, "scene_signature": resets[i]["scene"]["signature"],
                                         "raw_action": {"continuous": actions[i]["continuous"], "discrete": actions[i]["discrete"]}},
                                        item["action_trace"], item["diagnostic_trace"])
                        next_raw = item["raw_next"]
                        scaled = float(np.asarray(scalers[i](item["reward"])).item())
                        next_norm = normalizer(next_raw)
                        transition = {"state": raw_states[i], "continuous_action": actions[i]["continuous"],
                                      "discrete_index": actions[i]["discrete_index"], "reward": scaled,
                                      "next_state": next_raw, "done": item["done"]}
                        updates += learn_one(replay, agent, normalizer, args, sac, transition, writer, env_steps)
                        writer.add_scalar("BS/Continuous_LogProb_Action", actions[i]["continuous_log_prob"], env_steps)
                        writer.add_scalar("BS/Selected_Discrete_LogProb_Action", actions[i]["selected_discrete_log_prob"], env_steps)
                        writer.add_scalar("BS/Discrete_Entropy_Action", actions[i]["discrete_entropy"], env_steps)
                        env_steps += 1
                        raw_states[i], norm_states[i] = next_raw, next_norm
                        ep = episodes[i]
                        ep["detailed_traces"].append((item["action_trace"], item["diagnostic_trace"]))
                        ep["trajectory"].append(item["position"])
                        ep["rewards"].append(item["total"])
                        ep["objectives"].append(item["objective"])
                        ep["successes"].append(item["success"])
                        ep["statuses"].append(item["solver_status"])
                        ep["policy_versions"].append(sampled_version)
                        ep["policy_signatures"].append(sampled_signature)
                        ep["policy_lags"].append(lag)
                detailed.flush()
                completed = episode_idx + 1
                last_episodes = episodes
                solver_counts = {}
                for ep in episodes:
                    for status in ep["statuses"]:
                        solver_counts[status] = solver_counts.get(status, 0) + 1
                reward = float(np.mean([v for ep in episodes for v in ep["rewards"]]))
                completion = 100 * float(np.mean([v for ep in episodes for v in ep["successes"]]))
                finite = [v for ep in episodes for v in ep["objectives"] if np.isfinite(v)]
                max_lag = max(v for ep in episodes for v in ep["policy_lags"])
                row = dict(iteration=completed, env_steps=env_steps, episodes=completed * args.num_samplers,
                           sac_updates=updates, replay_size=replay.size(), reward=reward, completion_rate=completion,
                           finite_obj_mean=float(np.mean(finite)) if finite else "", finite_obj_slots=len(finite),
                           max_policy_lag_updates=max_lag, elapsed_seconds=time.monotonic() - started,
                           solver_status_counts=json.dumps(solver_counts, sort_keys=True))
                metrics = aggregate_metrics([pair for ep in episodes for pair in ep["detailed_traces"]])
                row.update(metrics)
                log_scalar_metrics(writer, metrics, completed)
                csv_writer.writerow(row); stream.flush()
                for i, ep in enumerate(episodes):
                    episode_stream.write(json.dumps({"iteration": completed, "worker_id": i,
                        "scene": ep["scene"], "action_seed": streams.seeds[i],
                        "policy_versions": ep["policy_versions"], "policy_signatures": ep["policy_signatures"],
                        "policy_lags": ep["policy_lags"],
                        "steps": len(ep["rewards"])}, ensure_ascii=False) + "\n")
                episode_stream.flush()
                writer.add_scalar("Reward/iteration", reward, completed)
                writer.add_scalar("Completion Rate/iteration", completion, completed)
                message = (f"[MHSAC] iter={completed} env_steps={env_steps} updates={updates} replay={replay.size()} "
                           f"reward={reward:.4f} completion={completion:.2f}% max_lag={max_lag} solver={solver_counts}")
                print(message, flush=True); log_stream.write(message + "\n"); log_stream.flush()
                checkpoint_ready = True
                if args.checkpoint_interval > 0 and completed % args.checkpoint_interval == 0:
                    save_checkpoint(output, completed, env_steps, updates, agent, normalizer, episodes, config)
    except KeyboardInterrupt:
        print("[MHSAC] interrupted; saving completed training results", flush=True)
    except BaseException:
        log_training_failure(run_paths["training_log"], traceback.format_exc())
        raise
    finally:
        try:
            for q in controls:
                try:
                    q.put_nowait(None)
                except queue.Full:
                    pass
            for worker in workers:
                if worker.pid is not None:
                    worker.join(timeout=5)
                    if worker.is_alive():
                        worker.terminate(); worker.join(timeout=5)
            save_reward_artifacts(output, base.seed, algorithm=config["algorithm"])
            if completed and last_episodes and checkpoint_ready:
                save_checkpoint(output, completed, env_steps, updates, agent, normalizer, last_episodes, config)
        finally:
            try:
                detailed.close()
                writer.close()
                for q in [*controls, results]:
                    q.cancel_join_thread(); q.close()
            finally:
                stop.restore()
    return output


if __name__ == "__main__":
    mp.freeze_support()
    run_training(*get_all_args())
