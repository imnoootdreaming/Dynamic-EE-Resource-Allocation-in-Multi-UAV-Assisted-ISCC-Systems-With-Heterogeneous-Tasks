"""Contracts and accounting for synchronous, complete multi-env rollouts.

This module changes neither the PPO objective nor the physical environment.
Trajectories are kept episode-major, including each terminal boundary.
"""
import hashlib
import json
import math

import numpy as np

from algorithms.MHBPPO.policy_diagnostics import summarize_policy_distribution

SYNC_ROLLOUT_CONTRACT = "shared_scene_complete_episode_batch_v1"
ACTION_STREAM_DOMAIN = 0x4D484250  # Stable named domain: MHBP policy actions.
TRANSITION_FIELDS = (
    "states", "continuous_actions", "discrete_actions", "next_states", "rewards",
    "old_cont_log_probs", "old_disc_log_probs", "dones", "real_dones",
    "raw_states", "raw_next_states",
)


def _nonnegative_int(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def derive_action_seed(master_seed, worker_id):
    """A worker stream independent of the shared scene's seed/episode keys."""
    master_seed = _nonnegative_int(master_seed, "master_seed")
    worker_id = _nonnegative_int(worker_id, "worker_id")
    return int(np.random.SeedSequence(master_seed, spawn_key=(ACTION_STREAM_DOMAIN, worker_id))
               .generate_state(1, dtype=np.uint32)[0])


def actor_state_signature(state_dict_or_actor):
    state = state_dict_or_actor.state_dict() if hasattr(state_dict_or_actor, "state_dict") else state_dict_or_actor
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        array = np.ascontiguousarray(value)
        digest.update(json.dumps([name, str(array.dtype), list(array.shape)], separators=(",", ":")).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def actor_parameter_checksum(actor):
    """Thread-independent diagnostic sum; exact parameter identity uses SHA256.

    Reduce a contiguous CPU float64 copy in NumPy, rather than a float32 Torch
    reduction whose order can differ between learner and sampler thread counts.
    This does not change the actor's tensors, device, RNG or Torch thread setting.
    """
    terms = []
    for param in actor.parameters():
        array = np.ascontiguousarray(param.detach().cpu().numpy(), dtype=np.float64)
        terms.append(float(np.abs(array).sum(dtype=np.float64)))
    return round(math.fsum(terms), 6)


def normalization_signature(mean, std, epsilon=1e-8):
    mean = np.asarray(mean, dtype=np.float64).reshape(-1)
    std = np.maximum(np.asarray(std, dtype=np.float64).reshape(-1), 1e-8)
    if mean.shape != std.shape or not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
        raise ValueError("normalizer mean/std must be finite, aligned vectors")
    digest = hashlib.sha256()
    digest.update(json.dumps([len(mean), float(epsilon)], separators=(",", ":")).encode())
    digest.update(np.ascontiguousarray(mean).tobytes())
    digest.update(np.ascontiguousarray(std).tobytes())
    return digest.hexdigest()


def _scene_signature(value):
    return value.get("signature") if isinstance(value, dict) else value


def validate_sync_episode(item, *, expected_round, expected_worker_ids,
                          expected_policy_version, expected_actor_checksum=None,
                          expected_actor_signature=None, expected_norm_signature,
                          expected_scene_signature, total_time_slots):
    """Reject a wrong version, origin, snapshot, or partial terminal trajectory."""
    if not isinstance(item, dict) or "error" in item:
        raise ValueError("synchronous result must be a complete episode, not an error sentinel")
    worker_id = _nonnegative_int(item.get("worker_id"), "worker_id")
    expected_ids = {_nonnegative_int(value, "expected_worker_id") for value in expected_worker_ids}
    if worker_id not in expected_ids or item.get("sampler_id") != worker_id:
        raise ValueError(f"unexpected worker identity: {worker_id}")
    if item.get("collection_round") != expected_round or item.get("episode_idx") != expected_round:
        raise ValueError("wrong collection round/episode index")
    if item.get("weights_version") != expected_policy_version:
        raise ValueError("wrong sampling policy version")
    if expected_actor_checksum is not None:
        checksum = item.get("weights_checksum")
        if checksum is None or not np.isfinite(checksum) or abs(checksum-expected_actor_checksum) >= 1e-3:
            raise ValueError(
                f"sampling actor checksum mismatch: worker={worker_id}, "
                f"round={expected_round}, version={item.get('weights_version')}, "
                f"expected={expected_actor_checksum!r}, actual={checksum!r}, "
                f"expected_SHA={expected_actor_signature}, actual_SHA={item.get('actor_signature')}"
            )
    if expected_actor_signature is not None and item.get("actor_signature") != expected_actor_signature:
        raise ValueError(
            f"sampling actor SHA mismatch: worker={worker_id}, round={expected_round}, "
            f"version={item.get('weights_version')}, expected={expected_actor_signature}, "
            f"actual={item.get('actor_signature')}"
        )
    if item.get("norm_signature") != expected_norm_signature:
        raise ValueError("sampling normalization signature mismatch")
    if normalization_signature(item.get("state_norm_mean"), item.get("state_norm_std")) != item.get("norm_signature"):
        raise ValueError("reported normalization signature disagrees with supplied snapshot")
    scene_signature = item.get("scene_signature")
    if not isinstance(scene_signature, str) or len(scene_signature) != 64 or any(
            value not in "0123456789abcdef" for value in scene_signature):
        raise ValueError("sampling scene signature must be a nonempty SHA256")
    if not isinstance(item.get("exogenous_scene_signature"), dict):
        raise ValueError("sampling episode must include its exogenous scene signature record")
    if expected_scene_signature is not None and scene_signature != _scene_signature(expected_scene_signature):
        raise ValueError("shared exogenous scene signature mismatch")
    if _scene_signature(item["exogenous_scene_signature"]) != scene_signature:
        raise ValueError("reported exogenous scene signatures disagree")
    count = _nonnegative_int(total_time_slots, "total_time_slots")
    if not count:
        raise ValueError("a complete episode requires positive time slots")
    transition = item.get("transition_dict", {})
    if any(name not in transition or len(transition[name]) != count for name in TRANSITION_FIELDS):
        raise ValueError("incomplete or misaligned transition fields")
    traces = item.get("action_traces", [])
    if len(traces) != count or [trace.get("time_slot") for trace in traces] != list(range(count)):
        raise ValueError("incomplete or reordered local time-slot sequence")
    boundary = [False]*(count-1)+[True]
    if [bool(value) for value in transition["dones"]] != boundary or (
            [bool(value) for value in transition["real_dones"]] != boundary):
        raise ValueError("complete episode requires only its final terminal/done boundary")
    if item.get("diagnostics_enabled") and len(item.get("diagnostic_traces", [])) != count:
        raise ValueError("incomplete enabled diagnostic traces")
    mean = np.asarray(item["state_norm_mean"], dtype=np.float64)
    std = np.maximum(np.asarray(item["state_norm_std"], dtype=np.float64), 1e-8)
    for raw_name, normalized_name in (("raw_states", "states"), ("raw_next_states", "next_states")):
        raw = np.asarray(transition[raw_name], dtype=np.float64)
        normalized = np.asarray(transition[normalized_name], dtype=np.float64)
        if raw.shape != normalized.shape or raw.ndim != 2 or raw.shape[1] != len(mean):
            raise ValueError("raw/normalized observation dimensions disagree")
        if not np.allclose(normalized, (raw-mean)/(std+1e-8), rtol=1e-6, atol=1e-7):
            raise ValueError("stored policy inputs disagree with the frozen normalization snapshot")
    if not np.all(np.isfinite(np.asarray(transition["rewards"], dtype=float))):
        raise ValueError("raw rewards must be finite")
    return count


def _sensing_summary(traces):
    outcomes = [trace.get("outcome", {}) for trace in traces]
    scores = [float(outcome["sensing_penalty_score"]) for outcome in outcomes
              if outcome.get("sensing_penalty_status") == "available"
              and outcome.get("sensing_penalty_score") is not None
              and np.isfinite(outcome["sensing_penalty_score"])]
    unavailable = sum(outcome.get("sensing_penalty_status") == "unavailable" for outcome in outcomes)
    return {
        "avg_base_reward": float(np.mean([outcome.get("base_reward", outcome["total_reward"]) for outcome in outcomes])),
        "avg_sensing_penalty_score": float(np.mean(scores)) if scores else (float("nan") if unavailable else 0.0),
        "avg_sensing_penalty_applied": float(np.mean([outcome.get("sensing_penalty_applied", 0.0) for outcome in outcomes])),
        "sensing_penalty_score_valid_count": len(scores),
        "sensing_penalty_unavailable_count": unavailable,
    }


def assemble_rollout_batch(items, *, continuous_action_splits=None):
    """Sort whole trajectories by worker; merge observations only episode-major."""
    if not items:
        raise ValueError("cannot assemble an empty rollout batch")
    episodes = sorted(items, key=lambda item: item["worker_id"])
    worker_ids = [item["worker_id"] for item in episodes]
    if len(set(worker_ids)) != len(worker_ids):
        raise ValueError("duplicate worker result in one rollout batch")
    first = episodes[0]
    for name in ("collection_round", "weights_version", "weights_checksum", "actor_signature",
                 "norm_signature", "scene_signature", "action_probability_metadata", "diagnostics_enabled"):
        if any(item.get(name) != first.get(name) for item in episodes[1:]):
            raise ValueError(f"cannot mix synchronous episode field {name}")
    count = len(first["transition_dict"]["rewards"])
    for item in episodes:
        validate_sync_episode(item, expected_round=first["collection_round"], expected_worker_ids=worker_ids,
            expected_policy_version=first["weights_version"], expected_actor_checksum=first.get("weights_checksum"),
            expected_actor_signature=first.get("actor_signature"), expected_norm_signature=first["norm_signature"],
            expected_scene_signature=first["scene_signature"], total_time_slots=count)
    transition = {name: [value for item in episodes for value in list(item["transition_dict"][name])]
                  for name in TRANSITION_FIELDS}
    traces = [trace for item in episodes for trace in item["action_traces"]]
    total = len(traces)
    split = continuous_action_splits or first.get("continuous_action_splits")
    if split is None:
        summary = {"available": False, "steps": 0}
    else:
        summary = summarize_policy_distribution(traces, split)
    weighted = lambda key: sum(float(item[key])*len(item["transition_dict"]["rewards"]) for item in episodes)/total
    solver_counts = {}
    for item in episodes:
        for kind, number in item["solver_status_counts"].items():
            solver_counts[kind] = solver_counts.get(kind, 0)+int(number)
    result = dict(first)
    result.update({
        "rollout_batch_contract": SYNC_ROLLOUT_CONTRACT,
        "episodes": episodes, "num_envs": len(episodes), "batch_size": total,
        "transition_dict": transition, "action_traces": traces,
        "diagnostic_traces": [trace for item in episodes for trace in item.get("diagnostic_traces", [])],
        "policy_distribution_summary": summary, "state_norm_count": total,
        "avg_total_reward": weighted("avg_total_reward"),
        "avg_bs_reward": weighted("avg_bs_reward"), "avg_obj_fun": weighted("avg_obj_fun"),
        "solver_status_counts": solver_counts,
        "action_min": min(item["action_min"] for item in episodes),
        "action_max": max(item["action_max"] for item in episodes),
        "collection_seconds": max(float(item.get("collection_seconds", 0.0)) for item in episodes),
        **_sensing_summary(traces),
    })
    for key in ("completion_rate", "finite_obj_rate", "no_solution_rate", "collision_rate", "spectrum_penalty_rate"):
        result[key] = weighted(key)
    return result
