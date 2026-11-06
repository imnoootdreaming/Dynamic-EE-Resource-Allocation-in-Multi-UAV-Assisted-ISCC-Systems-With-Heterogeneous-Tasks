"""Map unchanged Beta latents to conditional upload times with shared CPU limits.

The decoder is a deterministic part of the environment. PPO continues to use
the original latents and their probabilities, including on fallback branches.
"""
import numpy as np

from algorithms.Pure_MHBPPO.physical import _min_frequency


TIME_DECODER_VERSION = 1
TIME_CONSTRAINTS = ("uav_offloaded_bits", "task_freshness", "cu_offloaded_bits",
                    "uav_delay", "cu_delay", "bs_frequency")


def decode_times(*, params, eta, rates, task_bits, task_cycles, task_deadlines,
                 z_uav, z_cu, lower_uav, upper_uav, lower_cu, upper_cu,
                 legacy_uav, legacy_cu):
    """Decode UAVs then CUs in index order; reserve minimum CPU for future tasks.

    A prefix without a defined, feasible completion retains the legacy times and
    original penalties. It is never relabeled as feasible or silently resampled.
    Returned physical times stay float64, avoiding a float32 round trip at bounds.
    """
    info = {"mode": "conditional", "version": TIME_DECODER_VERSION,
            "applied": False, "order": "uav_index_then_cu_index",
            "uav_seconds": np.asarray(legacy_uav).tolist(),
            "cu_seconds": np.asarray(legacy_cu).tolist(),
            "uav_lower_seconds": np.asarray(lower_uav).tolist(),
            "uav_upper_seconds": np.asarray(upper_uav).tolist(),
            "cu_lower_seconds": np.asarray(lower_cu).tolist(),
            "cu_upper_seconds": np.asarray(upper_cu).tolist()}

    def fallback(reason):
        return {**info, "status": reason}

    eta = np.asarray(eta, dtype=float)
    if (not np.all((eta == 0) | (eta == 1)) or
            not np.all(eta.sum(axis=1) == 1)):
        return fallback("invalid_matching")
    if np.any(eta.sum(axis=0) > 1):
        return fallback("duplicate_matching")
    Q, Ru = (np.asarray(rates[key], dtype=float) for key in ("sensed_bits", "uav_offloading"))
    R1, R2, R3 = (np.asarray(rates[key], dtype=float) for key in ("cu_stage1", "cu_stage2", "cu_stage3"))
    L, C, D = (np.asarray(value, dtype=float) for value in (task_bits, task_cycles, task_deadlines))
    zu, zc = (np.asarray(value, dtype=float) for value in (z_uav, z_cu))
    lu, uu, lc, uc = (np.asarray(value, dtype=float) for value in
                      (lower_uav, upper_uav, lower_cu, upper_cu))
    inputs = (Q, Ru, R1, R2, R3, L, C, D, zu, zc, lu, uu, lc, uc)
    if (not all(np.all(np.isfinite(value)) for value in inputs) or
            np.any(Q < 0) or np.any(L < 0) or np.any(C <= 0) or
            np.any(Ru < 0) or np.any(R1 < 0) or np.any(R2 < 0) or np.any(R3 < 0)):
        return fallback("invalid_rates_or_tasks")
    if np.any((zu < 0) | (zu > 1)) or np.any((zc < 0) | (zc > 1)):
        raise ValueError("time decoder expects original Beta latents in [0, 1]")
    if np.any((Ru == 0) & (Q > 0)) or np.any((R3 == 0) & (L > 0)):
        return fallback("zero_upload_rate")
    # The monotone completion argument requires the current three-stage rate model.
    if np.any(R1 > R3[None, :]) or np.any(R2 > R3[None, :]):
        return fallback("nonmonotone_rates")
    ds, span = float(params.uav_sen_duration), float(params.uav_max_delay - params.uav_sen_duration)
    limit = float(params.bs_max_freq)
    if not np.isfinite(limit) or limit <= 0:
        return fallback("invalid_cpu_limit")
    Wu, Wc = float(params.C_sen) * Q, C * L
    u_min = np.maximum(lu, np.divide(Q, Ru, out=np.zeros_like(Q), where=Ru > 0))
    r3_safe = np.where(R3 > 0, R3, 1.0)
    base = np.divide(L, R3, out=np.zeros_like(L), where=R3 > 0)
    offset = np.sum(eta * ds * (R3[None, :] - R1) / r3_safe, axis=0)
    slope = eta * (R3[None, :] - R2) / r3_safe

    def cu_minimum(u):
        return np.maximum.reduce((lc, eta.T @ (ds + u), base + offset + u @ slope))

    def cpu(u, c):
        if (np.any(u > uu) or np.any(c > uc) or np.any(span - u <= 0) or np.any(D - c <= 0)):
            return float("inf")
        return float(_min_frequency(Wu, span - u).sum() + _min_frequency(Wc, D - c).sum())

    u, c = u_min.copy(), cu_minimum(u_min)
    minimum_cpu = cpu(u, c)
    info["minimum_cpu_hz"] = minimum_cpu if np.isfinite(minimum_cpu) else None
    if minimum_cpu > limit:
        return fallback("no_feasible_time_completion")

    # Retain a feasible lower endpoint throughout the monotone search.
    u_upper = np.empty_like(u)
    for i in range(len(u)):
        left, right = float(u_min[i]), min(float(uu[i]), span)
        for _ in range(40):
            middle = (left + right) / 2
            trial = u.copy()
            trial[i] = middle
            if cpu(trial, cu_minimum(trial)) <= limit:
                left = middle
            else:
                right = middle
        u_upper[i] = left
        u[i] = np.clip(u_min[i] + zu[i] * (left - u_min[i]), u_min[i], left)

    c_lower = cu_minimum(u)
    c, c_upper = c_lower.copy(), np.empty_like(c_lower)
    fu = _min_frequency(Wu, span - u)
    fc = _min_frequency(Wc, D - c)
    for j in range(len(c)):
        available = limit - (fu.sum() + fc.sum() - fc[j])
        if available <= 0:
            return fallback("numerical_cpu_boundary")
        if cpu(u, c) > limit:
            return fallback("numerical_cpu_boundary")
        # The current lower point is already verified feasible. Cancellation in
        # D-W/F can put the analytic endpoint a few ulps below that point.
        high = max(c_lower[j], min(float(uc[j]), float(D[j] - Wc[j] / available)))
        trial = c.copy()
        for _ in range(8):
            trial[j] = high
            if cpu(u, trial) <= limit:
                break
            high = np.nextafter(high, c_lower[j])
        else:
            # Rare ill-conditioned endpoints: keep a feasible lower bracket.
            left, right = float(c_lower[j]), high
            for _ in range(40):
                middle = (left + right) / 2
                trial[j] = middle
                if cpu(u, trial) <= limit:
                    left = middle
                else:
                    right = middle
            high = left
        c_upper[j] = high
        c[j] = np.clip(c_lower[j] + zc[j] * (high - c_lower[j]), c_lower[j], high)
        fc[j] = _min_frequency(Wc[j:j+1], D[j:j+1] - c[j])[0]
    used = cpu(u, c)
    if not np.isfinite(used) or used > limit * (1 + 1e-12):
        return fallback("numerical_cpu_boundary")
    return {**info, "status": "applied", "applied": True,
            "uav_seconds": u.tolist(), "cu_seconds": c.tolist(),
            "uav_lower_seconds": u_min.tolist(), "uav_upper_seconds": u_upper.tolist(),
            "cu_lower_seconds": c_lower.tolist(), "cu_upper_seconds": c_upper.tolist(),
            "used_cpu_hz": used, "cpu_limit_hz": limit}
