"""Read-only moments of the policy used to sample each transition.

No extra samples or policy forward passes.  std is conditional distribution
spread; sampled_action_std also includes state variation and finite-sample noise.
"""
import numpy as np

POLICY_DISTRIBUTION_CONTRACT = "sampling_policy_conditional_moments_v1"


def capture_policy_distribution(continuous_dists, discrete_dists, active_heads, dist_type):
    """Return batch-first NumPy diagnostics from the already constructed dists."""
    def values(tensor):
        return tensor.detach().cpu().numpy().copy()
    continuous = {}
    for name, dist in continuous_dists.items():
        entry = {"active": name in active_heads, "dim": int(dist.batch_shape[-1]),
                 "space": "unit_interval" if dist_type == "beta" else "pre_sigmoid"}
        if entry["active"]:
            entry.update(mean=values(dist.mean), std=values(dist.stddev),
                         entropy=values(dist.entropy()))
            if dist_type == "beta":
                entry["concentration"] = values(dist.concentration1 + dist.concentration0)
        else:
            entry["reason"] = "excluded_from_policy_probability_and_entropy"
        continuous[name] = entry
    discrete = {f"cu_matching_{i}": {"probabilities": values(dist.probs),
                                     "entropy": values(dist.entropy())}
                for i, dist in enumerate(discrete_dists)}
    return {"contract": POLICY_DISTRIBUTION_CONTRACT, "distribution": dist_type,
            "continuous": continuous, "discrete": discrete}


def single_state_policy_distribution(batch_stats):
    """Remove the single sampler batch axis, producing portable JSON lists."""
    def row(entry):
        output = {}
        for key, value in entry.items():
            if isinstance(value, np.ndarray):
                if value.shape[0] != 1:
                    raise ValueError("sampler distribution diagnostics require one state")
                output[key] = value[0].tolist()
            else:
                output[key] = value
        return output
    return {"contract": batch_stats["contract"], "distribution": batch_stats["distribution"],
            "continuous": {k: row(v) for k, v in batch_stats["continuous"].items()},
            "discrete": {k: row(v) for k, v in batch_stats["discrete"].items()}}


def summarize_policy_distribution(traces, continuous_splits):
    """Aggregate per-coordinate conditional moments over one actual rollout."""
    records = [trace.get("policy_distribution") for trace in traces]
    if not records or all(record is None for record in records):
        return {"available": False, "steps": 0}
    if any(record is None for record in records):
        raise ValueError("partial policy distribution diagnostics in one rollout")
    first = records[0]
    output = {"available": True, "contract": first["contract"],
              "distribution": first["distribution"], "steps": len(records),
              "continuous": {}, "discrete": {}}
    actions = np.asarray([t["clipped_action"]["continuous"] for t in traces], dtype=np.float64)
    start = 0
    for name, size in continuous_splits.items():
        entry = first["continuous"][name]
        sample = actions[:, start:start + size]
        start += size
        if not entry["active"]:
            output["continuous"][name] = dict(entry)
            continue
        means = np.asarray([r["continuous"][name]["mean"] for r in records], dtype=np.float64)
        stds = np.asarray([r["continuous"][name]["std"] for r in records], dtype=np.float64)
        entropies = np.asarray([r["continuous"][name]["entropy"] for r in records], dtype=np.float64)
        summary = {"active": True, "dim": size, "space": entry["space"],
                   "mean": means.mean(axis=0).tolist(),
                   "std": stds.mean(axis=0).tolist(),
                   "state_mean_std": means.std(axis=0).tolist(),
                   "entropy": entropies.mean(axis=0).tolist(),
                   "sampled_action_space": "unit_interval",
                   "sampled_action_mean": sample.mean(axis=0).tolist(),
                   "sampled_action_std": sample.std(axis=0).tolist()}
        if entry["space"] == "unit_interval":
            summary["marginal_std"] = np.sqrt(
                (stds ** 2).mean(axis=0) + means.var(axis=0)).tolist()
        if "concentration" in entry:
            summary["concentration"] = np.asarray(
                [r["continuous"][name]["concentration"] for r in records],
                dtype=np.float64).mean(axis=0).tolist()
        output["continuous"][name] = summary
    for name in first["discrete"]:
        probabilities = np.asarray([r["discrete"][name]["probabilities"] for r in records])
        entropies = np.asarray([r["discrete"][name]["entropy"] for r in records])
        output["discrete"][name] = {
            "probabilities": probabilities.mean(axis=0).tolist(),
            "entropy": float(entropies.mean()),
            "max_probability": float(probabilities.max(axis=-1).mean())}
    return output


def log_policy_distribution(logger, writer, summary, iteration, weights_version):
    """Log rollout-policy moments, never recompute with updated learner weights."""
    if not summary.get("available", False):
        return
    vector = lambda xs: "[" + ",".join(f"{x:.4f}" for x in xs) + "]"
    for name, stats in summary["continuous"].items():
        if not stats["active"]:
            continue
        message = (f"[POLICY] iter={iteration} wv={weights_version} head={name} "
                   f"space={stats['space']} mean={vector(stats['mean'])} "
                   f"std={vector(stats['std'])} state_mean_std={vector(stats['state_mean_std'])} "
                   f"sample_std_u01={vector(stats['sampled_action_std'])}")
        if "concentration" in stats:
            message += f" concentration={vector(stats['concentration'])}"
        logger.info(message)
        prefix = f"PolicyDistribution/{name}"
        for metric in ("mean", "std", "state_mean_std", "sampled_action_std", "concentration"):
            if metric not in stats:
                continue
            writer.add_scalar(f"{prefix}/{metric}_avg", float(np.mean(stats[metric])), iteration)
            for dim, value in enumerate(stats[metric]):
                writer.add_scalar(f"{prefix}/{metric}/dim_{dim}", float(value), iteration)
        writer.add_scalar(f"{prefix}/entropy_avg", float(np.mean(stats["entropy"])), iteration)
    discrete_text = []
    for name, stats in summary["discrete"].items():
        discrete_text.append(f"{name}:entropy={stats['entropy']:.4f},pmax={stats['max_probability']:.4f}")
        writer.add_scalar(f"PolicyDistribution/{name}/entropy", stats["entropy"], iteration)
        writer.add_scalar(f"PolicyDistribution/{name}/max_probability", stats["max_probability"], iteration)
    if discrete_text:
        logger.info(f"[POLICY_DISCRETE] iter={iteration} wv={weights_version} " + " | ".join(discrete_text))
