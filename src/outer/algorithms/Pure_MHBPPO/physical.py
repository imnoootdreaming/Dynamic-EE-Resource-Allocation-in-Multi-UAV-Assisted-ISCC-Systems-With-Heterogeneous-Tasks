"""Direct P0 physics for Pure-MHBPPO; no optimization solver is invoked."""
import numpy as np


def _check(values, scale, unit, tolerance=1e-6):
    raw = np.asarray(values, dtype=float).reshape(-1)
    scale = np.broadcast_to(np.asarray(scale, dtype=float), np.asarray(values).shape).reshape(-1)
    normalized = np.maximum(raw, 0.0) / np.maximum(np.abs(scale), 1e-30)
    normalized = np.where(np.isfinite(normalized), normalized, np.inf)
    return {"raw_violation": raw.tolist(), "unit": unit,
            "max_normalized_violation": float(np.max(normalized, initial=0.0)),
            "violated_count": int(np.sum(normalized > tolerance)), "tolerance": tolerance,
            "passed": bool(np.all(normalized <= tolerance))}


def _min_frequency(work, remaining):
    """Return +inf for impossible timing; positive epsilon only for zero work."""
    work = np.asarray(work, dtype=float)
    remaining = np.asarray(remaining, dtype=float)
    result = np.full(work.shape, np.inf)
    normal = (work > 0) & (remaining > 0)
    result[normal] = work[normal] / remaining[normal]
    zero_work = (work == 0) & (remaining >= 0)
    result[zero_work] = 1e-12
    return result


def evaluate_physical(*, params, eta, p_cu, duration_uav, duration_cu, sensing_beams,
                      offloading_beams, positions_before, positions_after, h_cu_uav,
                      h_uav_bs, h_cu_bs, sensing_channels, task_bits, task_cycles,
                      task_deadlines, slot_seconds):
    """Evaluate one complete physical candidate and all 15 P0 constraints.

    The CU-UAV channels are row vectors h: interference is |g.T @ h|^2.
    Energies are physical candidate values, never virtual power corrections.
    """
    eta = np.asarray(eta, dtype=float)
    p = np.asarray(p_cu, dtype=float)
    du = np.asarray(duration_uav, dtype=float)
    dc = np.asarray(duration_cu, dtype=float)
    w = np.asarray(sensing_beams, dtype=np.complex128)
    b = np.asarray(offloading_beams, dtype=np.complex128)
    q0, q1 = np.asarray(positions_before, dtype=float), np.asarray(positions_after, dtype=float)
    h = np.asarray(h_cu_uav, dtype=np.complex128)
    hu = np.asarray(h_uav_bs, dtype=np.complex128).reshape(w.shape)
    hc = np.asarray(h_cu_bs, dtype=np.complex128).reshape(-1)
    A = np.asarray(sensing_channels, dtype=np.complex128)
    L = np.asarray(task_bits, dtype=float)
    cycles = np.asarray(task_cycles, dtype=float)
    deadlines = np.asarray(task_deadlines, dtype=float)
    I, N = w.shape
    J = p.size
    if (eta.shape != (I, J) or b.shape != (I, N) or h.shape != (I, J, N)
            or A.shape != (I, N, N) or hu.shape != (I, N) or hc.shape != (J,)
            or du.shape != (I,) or dc.shape != (J,) or L.shape != (J,)
            or cycles.shape != (J,) or deadlines.shape != (J,) or q0.shape != (I, 3) or q1.shape != (I, 3)):
        raise ValueError("physical input shape mismatch")
    all_inputs = (eta, p, du, dc, w, b, q0, q1, h, hu, hc, A, L, cycles, deadlines)
    if not all(np.all(np.isfinite(x)) for x in all_inputs) or not np.isfinite(slot_seconds):
        return {"valid_input": False, "feasible": False, "candidate_energy": None,
                "reward": -1.0, "checks": {"finite_input": {"passed": False}},
                "solver_status_kind": "not_called"}
    if (np.any(p < 0) or params.bandwidth <= 0 or slot_seconds <= 0 or
            np.any(L < 0) or np.any(cycles <= 0) or np.any(deadlines <= 0)):
        return {"valid_input": False, "feasible": False, "candidate_energy": None,
                "reward": -1.0, "checks": {"physical_input_domain": {"passed": False}},
                "solver_status_kind": "not_called"}
    if (params.C_sen <= 0 or params.kappa < 0 or
            min(params.omega_weight_1, params.omega_weight_2, params.omega_weight_3) < 0):
        raise ValueError("minimum-frequency energy requires positive sensing cycles and nonnegative energy weights")
    noise = 10 ** ((float(params.noise_power_density_dbm) - 30) / 10) * params.bandwidth
    if not np.isfinite(noise) or noise <= 0:
        raise ValueError("noise power must be positive and finite")
    threshold = 10 ** (float(params.sen_sinr) / 10)
    g = np.empty((I, N), dtype=np.complex128)
    g_fallback = []
    for i in range(I):
        covariance = noise * np.eye(N, dtype=np.complex128)
        for j in range(J):
            covariance += eta[i, j] * p[j] * np.outer(h[i, j].conj(), h[i, j])
        scale = np.max(np.abs(covariance))
        direction = np.linalg.solve(covariance / scale, A[i] @ w[i])
        norm = np.linalg.norm(direction)
        fallback = not np.isfinite(norm) or norm <= 1e-30
        g[i] = np.eye(1, N, 0, dtype=np.complex128)[0] if fallback else direction / norm
        g_fallback.append(bool(fallback))
    signal = np.array([abs(np.vdot(g[i], A[i] @ w[i])) ** 2 for i in range(I)])
    interference = np.array([[abs(np.dot(g[i], h[i, j])) ** 2 for j in range(J)] for i in range(I)])
    sinr = signal / (noise + np.sum(eta * p[None, :] * interference, axis=1))
    xi2 = 2 * params.var_range_fluctuation * params.radar_spectrum_shape**2 * params.bandwidth**3 * params.radar_impulse_duration
    r_sen = params.radar_duty_ratio / (2 * params.radar_impulse_duration) * np.log2(1 + xi2 * sinr)
    h_c_bs_sq = np.abs(hc) ** 2
    signal_off = np.array([abs(np.vdot(hu[i], b[i])) ** 2 for i in range(I)])
    interference_sen = np.array([abs(np.vdot(hu[i], w[i])) ** 2 for i in range(I)])
    interference_off = signal_off
    r_off = params.bandwidth * np.log2(1 + signal_off / (noise + eta @ (p * h_c_bs_sq)))
    signal_cu = p * h_c_bs_sq
    r3 = params.bandwidth * np.log2(1 + signal_cu / noise)
    r1 = params.bandwidth * np.log2(1 + signal_cu[None, :] / (noise + interference_sen[:, None]))
    r2 = params.bandwidth * np.log2(1 + signal_cu[None, :] / (noise + interference_off[:, None]))
    ds = float(params.uav_sen_duration)
    remaining_stage = dc - eta.T @ (np.full(I, ds) + du)
    cu_bits = np.sum(eta * (ds * r1 + du[:, None] * r2), axis=0) + remaining_stage * r3
    sensed_bits = ds * r_sen
    work_u = float(params.C_sen) * sensed_bits
    work_c = cycles * L
    remaining_u = float(params.uav_max_delay) - ds - du
    remaining_c = deadlines - dc
    freq_u = _min_frequency(work_u, remaining_u)
    freq_c = _min_frequency(work_c, remaining_c)
    delay_u = np.full(I, np.inf)
    delay_c = np.full(J, np.inf)
    valid_u, valid_c = np.isfinite(freq_u) & (freq_u > 0), np.isfinite(freq_c) & (freq_c > 0)
    delay_u[valid_u] = ds + du[valid_u] + work_u[valid_u] / freq_u[valid_u]
    delay_c[valid_c] = dc[valid_c] + work_c[valid_c] / freq_c[valid_c]
    w_power = np.sum(np.abs(w)**2, axis=1)
    b_power = np.sum(np.abs(b)**2, axis=1)
    distance = np.linalg.norm(q1-q0, axis=1)
    pair_distances = np.array([np.linalg.norm(q1[i]-q1[j]) for i in range(I) for j in range(i+1, I)])
    collided = set()
    for i in range(I):
        for j in range(i+1, I):
            if np.linalg.norm(q1[i]-q1[j]) < params.uav_safe_distance:
                collided.update((i, j))
    pmax_cu = 10 ** ((float(params.cu_max_power_dbm)-30)/10)
    row_error = np.abs(np.sum(eta, axis=1)-1)
    col_excess = np.maximum(np.sum(eta, axis=0)-1, 0)
    checks = {
        "uav_one_cu": _check(row_error, 1, "1"),
        "cu_at_most_one_uav": _check(col_excess, 1, "1"),
        "uav_speed": _check(np.maximum(params.uav_min_speed*slot_seconds-distance, distance-params.uav_max_speed*slot_seconds), params.uav_max_speed*slot_seconds, "m"),
        "uav_collision": _check(params.uav_safe_distance-pair_distances, params.uav_safe_distance, "m"),
        "sensing_power": _check(w_power-params.uav_max_power, params.uav_max_power, "W"),
        "receive_norm": _check(np.abs(np.sum(np.abs(g)**2, axis=1)-1), 1, "1"),
        "sensing_sinr": _check(threshold-sinr, threshold, "1"),
        "offloading_power": _check(b_power-params.uav_max_power, params.uav_max_power, "W"),
        "uav_offloaded_bits": _check(sensed_bits-du*r_off, np.maximum(sensed_bits, 1), "bit"),
        "task_freshness": _check(ds+du-eta @ dc, params.uav_max_delay, "s"),
        "cu_offloaded_bits": _check(L-cu_bits, np.maximum(L, 1), "bit"),
        "cu_power": _check(p-pmax_cu, pmax_cu, "W"),
        "uav_delay": _check(delay_u-params.uav_max_delay, params.uav_max_delay, "s"),
        "cu_delay": _check(delay_c-deadlines, deadlines, "s"),
        "bs_frequency": _check(np.array([freq_u.sum()+freq_c.sum()-params.bs_max_freq]), params.bs_max_freq, "Hz"),
    }
    general_domain = (np.all((eta == 0) | (eta == 1)) and np.all(du > 0) and np.all(dc > 0)
                      and np.all(p > 0) and np.all(np.isfinite(freq_u)) and np.all(np.isfinite(freq_c)))
    valid_domain = bool(general_domain and np.all(remaining_stage >= -1e-12))
    finite_energy = bool(np.all(np.isfinite(freq_u)) and np.all(np.isfinite(freq_c)) and np.all(distance > 0))
    if finite_energy:
        flight = float(np.sum(params.uav_c1*distance**3/slot_seconds**2 + params.uav_c2*slot_seconds**2/distance))
        e_bs = float(params.kappa*(np.sum(work_u*freq_u**2)+np.sum(work_c*freq_c**2)))
        e_uav = float(ds*np.sum(w_power)+np.sum(du*b_power)+flight)
        e_cu = float(np.sum(dc*p))
        energy = float(params.omega_weight_1*e_bs+params.omega_weight_2*e_uav+params.omega_weight_3*e_cu)
        if not np.isfinite(energy):
            finite_energy, energy = False, None
    else:
        energy, e_bs, e_uav, e_cu, flight = None, None, None, None, None
    feasible = bool(valid_domain and finite_energy and all(item["passed"] for item in checks.values()))
    # Duplicate sharing already incurs the explicit spectrum penalty. Its CU
    # three-stage schedule is undefined, so do not use its algebraic bits or
    # residual stage duration to add a second candidate penalty for that CU.
    legal_cu = col_excess <= 1e-12
    service_domain = bool(general_domain and np.all(remaining_stage[legal_cu] >= -1e-12))
    legal_cu_bits = _check((L-cu_bits)[legal_cu], np.maximum(L[legal_cu], 1), "bit")
    service_failed = (not service_domain or not finite_energy or not legal_cu_bits["passed"] or
        any(not value["passed"] for key, value in checks.items()
            if key not in ("cu_at_most_one_uav", "uav_collision", "cu_offloaded_bits")))
    spectrum_penalty = float(np.sum(col_excess))
    collision_penalty = float(len(collided))
    reward = (float(np.exp(-energy/1000)) if energy is not None else 0.0) - spectrum_penalty - collision_penalty - float(service_failed)
    return {"valid_input": True, "feasible": feasible, "candidate_energy": energy,
            "reward": float(reward), "solver_status_kind": "not_called", "backend": "direct_no_optimizer",
            "service_candidate_penalty": int(service_failed), "spectrum_penalty": spectrum_penalty,
            "collision_penalty": collision_penalty, "checks": checks,
            "rates": {"sensing": r_sen.tolist(), "uav_offloading": r_off.tolist(), "cu_stage1": r1.tolist(),
                      "cu_stage2": r2.tolist(), "cu_stage3": r3.tolist(), "sensing_sinr": sinr.tolist(),
                      "sensed_bits": sensed_bits.tolist(), "cu_offloaded_bits": cu_bits.tolist(),
                      "cu_stage3_seconds": remaining_stage.tolist()},
            "frequencies": {"uav_hz": freq_u.tolist(), "cu_hz": freq_c.tolist()},
            "energy": {"bs_j": e_bs, "uav_j": e_uav, "cu_j": e_cu, "flight_j": flight,
                       "weighted_total_j": energy},
            "receive_beams": {"real": g.real.tolist(), "imag": g.imag.tolist(), "fallback": g_fallback}}
