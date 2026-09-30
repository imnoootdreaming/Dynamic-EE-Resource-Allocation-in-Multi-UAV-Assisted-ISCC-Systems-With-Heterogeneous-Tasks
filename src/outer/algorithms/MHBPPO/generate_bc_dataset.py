"""Generate state-action demonstrations for MHBPPO behavior cloning.

This is an offline, expensive utility.  It evaluates a small pool of
candidate outer actions with the existing PC3P bridge and keeps only actions
whose inner solve returns ``optimal``.  It never changes the online trainer.

Example (from ``src/outer``):
    python -m algorithms.MHBPPO.generate_bc_dataset \
        --outer_pool ../inner/feasible_outer_samples.csv \
        --output bc_feasible.npz --episodes 20
"""

import argparse
import copy
import csv
import os
from types import SimpleNamespace

import numpy as np

from algorithms.MHBPPO.behavior_cloning import save_behavior_cloning_dataset
from configs.base_params import get_base_args
from environment.my_env import MyEnv


def _is_optimal(reward_dict):
    return str(reward_dict.get("components", {}).get("solver_status_kind", "")) == "optimal"


def _pool_action_to_actor(env, sample):
    """Convert (g, eta, p, Doff, displacement) to actor normalized actions."""
    g, eta, p_cu, d_off, displacement = sample
    space = env.action_space["bs"]["continuous"]
    low = np.asarray(space.low, dtype=np.float64)
    high = np.asarray(space.high, dtype=np.float64)
    I = env.base_args.uavs_num
    J = env.base_args.cus_num
    N = env.base_args.antenna_nums

    distance = np.linalg.norm(np.asarray(displacement)[:, :2], axis=1)
    angle = np.mod(np.arctan2(displacement[:, 1], displacement[:, 0]), 2.0 * np.pi)
    raw = np.concatenate([
        angle,
        distance,
        np.asarray(d_off, dtype=np.float64).reshape(I),
        np.asarray(p_cu, dtype=np.float64).reshape(J),
        np.asarray(g).real.reshape(I * N),
        np.asarray(g).imag.reshape(I * N),
    ])
    normalized = (raw - low) / np.maximum(high - low, 1e-12)
    normalized = np.clip(normalized, 0.0, 1.0).astype(np.float32)
    discrete = np.argmax(np.asarray(eta), axis=1).astype(np.int64)
    return {"continuous": normalized, "discrete": discrete}


def _make_args(seed, total_time_slots, randomize, structured_beam_alpha=1.0):
    return SimpleNamespace(
        total_time_slots=int(total_time_slots),
        hidden_dim=128,
        randomize_layout_per_episode=bool(randomize),
        randomize_cu_traj_per_episode=bool(randomize),
        randomize_nlos_per_episode=bool(randomize),
        debug_topology=False,
        structured_beam_alpha=float(structured_beam_alpha),
    )


def _parse_seed_offsets(args):
    """Return seed offsets matching the online sampler workers."""
    text = str(getattr(args, "seed_offsets", "") or "").strip()
    if text:
        offsets = [int(item.strip()) for item in text.split(",") if item.strip()]
        if not offsets:
            raise ValueError("--seed_offsets must contain at least one integer")
    else:
        num_workers = int(getattr(args, "num_workers", 1))
        if num_workers <= 0:
            raise ValueError("--num_workers must be positive")
        offsets = list(range(1, num_workers + 1))
    if any(offset < 0 for offset in offsets):
        raise ValueError("seed offsets must be non-negative")
    return offsets


def generate(args):
    base = get_base_args(seed=int(args.seed))
    pool = _load_outer_pool(args.outer_pool, base)
    if not pool:
        raise RuntimeError("外层可行样本池为空")

    madrl = _make_args(
        args.seed,
        args.total_time_slots,
        args.randomize_scenes,
        args.structured_beam_alpha,
    )
    seed_offsets = _parse_seed_offsets(args)
    raw_states = []
    actor_actions = []
    discrete_actions = []
    status_counts = {"optimal": 0, "infeasible": 0, "unknown": 0, "error": 0, "other": 0}

    for seed_offset in seed_offsets:
        # Match one online sampler's deterministic scene stream.
        env = MyEnv(base, madrl, seed_offset=seed_offset)
        for episode_idx in range(int(args.episodes)):
            state = env.reset(episode_idx)
            done = False
            steps_this_episode = 0
            while not done:
                state_raw = np.asarray(state["bs"], dtype=np.float32).copy()
                chosen = None
                chosen_trial = None
                # deepcopy preserves the current dynamic state, so every candidate
                # is evaluated at exactly the same s_t without replaying history.
                for sample in pool:
                    trial_env = copy.deepcopy(env)
                    action = _pool_action_to_actor(trial_env, sample)
                    next_state, _, reward_dict, trial_done, _, _ = trial_env.step(
                        {"bs": action}, episode_idx
                    )
                    kind = str(reward_dict.get("components", {}).get("solver_status_kind", "other"))
                    status_counts[kind] = status_counts.get(kind, 0) + 1
                    if _is_optimal(reward_dict):
                        chosen = action
                        chosen_trial = (trial_env, next_state, trial_done, kind)
                        break
                    # Keep the final candidate as a deterministic fallback. It is
                    # not saved as a positive BC demonstration.
                    chosen = action
                    chosen_trial = (trial_env, next_state, trial_done, kind)

                if chosen_trial is None:
                    raise RuntimeError("candidate pool did not produce any action")
                trial_env, next_state, trial_done, chosen_kind = chosen_trial
                if chosen_kind == "optimal":
                    raw_states.append(state_raw)
                    actor_actions.append(chosen["continuous"])
                    discrete_actions.append(chosen["discrete"])
                env = trial_env
                state = next_state
                done = bool(trial_done)
                steps_this_episode += 1
                max_steps = int(getattr(args, "max_steps", 0) or 0)
                if max_steps > 0 and steps_this_episode >= max_steps:
                    break
    if not raw_states:
        print({
            "output": os.path.abspath(args.output),
            "demonstrations": 0,
            "pool_size": len(pool),
            "seed_offsets": seed_offsets,
            "solver_status_counts": status_counts,
        })
        raise RuntimeError(
            "没有收集到 optimal demonstration；请扩大 outer_pool 或先检查候选池与当前桥接场景是否一致。"
        )
    raw_states = np.asarray(raw_states, dtype=np.float32)
    mean = raw_states.mean(axis=0, dtype=np.float64)
    std = np.maximum(raw_states.std(axis=0, dtype=np.float64), 1e-4)
    states = ((raw_states - mean) / std).astype(np.float32)
    save_behavior_cloning_dataset(
        args.output,
        states,
        np.asarray(actor_actions, dtype=np.float32),
        np.asarray(discrete_actions, dtype=np.int64),
        state_norm_mean=mean,
        state_norm_std=std,
        state_norm_count=len(raw_states),
        raw_states=raw_states,
    )
    print({
        "output": os.path.abspath(args.output),
        "demonstrations": len(raw_states),
        "pool_size": len(pool),
        "seed_offsets": seed_offsets,
        "solver_status_counts": status_counts,
    })


def _load_outer_pool(path, base_args):
    """Read the inner sampler CSV without importing its ``environment`` name.

    The outer process already owns ``sys.modules['environment']``; importing
    ``src/inner/outer_sampler.py`` after that would resolve its flat import to
    the wrong package.  The CSV layout is simple and intentionally duplicated
    here to keep the offline utility isolated from online module state.
    """
    I, J, N = base_args.uavs_num, base_args.cus_num, base_args.antenna_nums
    expected = 1 + 2 * I * N + I * J + J + I + 3 * I
    samples = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        if len(header) != expected:
            raise ValueError(f"外层样本池列数不符: expected={expected}, actual={len(header)}")
        for row in reader:
            if not row:
                continue
            values = [float(x) for x in row[1:]]
            if len(values) != expected - 1:
                raise ValueError("外层样本池数据行列数不符")
            idx = 0
            g_re = np.asarray(values[idx:idx + I * N]).reshape(I, N); idx += I * N
            g_im = np.asarray(values[idx:idx + I * N]).reshape(I, N); idx += I * N
            g = g_re + 1j * g_im
            eta = np.asarray(values[idx:idx + I * J]).reshape(I, J); idx += I * J
            p = np.asarray(values[idx:idx + J]); idx += J
            d_off = np.asarray(values[idx:idx + I]); idx += I
            step = np.asarray(values[idx:idx + 3 * I]).reshape(I, 3)
            samples.append((g, eta, p, d_off, step))
    return samples


def main():
    parser = argparse.ArgumentParser(description="Generate MHBPPO feasible demonstrations")
    parser.add_argument("--outer_pool", required=True, help="inner outer_sampler CSV")
    parser.add_argument("--output", required=True, help="output .npz")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--total_time_slots", type=int, default=40)
    parser.add_argument(
        "--max_steps", type=int, default=0,
        help="per-episode probe limit; 0 means run all steps. Keep total_time_slots legal (normally 40).",
    )
    parser.add_argument("--seed", type=int, default=1208)
    parser.add_argument(
        "--num_workers", type=int, default=1,
        help="generate online-equivalent worker offsets 1..num_workers",
    )
    parser.add_argument(
        "--seed_offsets", default="",
        help="comma-separated explicit offsets; overrides --num_workers (0 is baseline)",
    )
    parser.add_argument("--randomize_scenes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--structured_beam_alpha", type=float, default=1.0)
    generate(parser.parse_args())


if __name__ == "__main__":
    main()
