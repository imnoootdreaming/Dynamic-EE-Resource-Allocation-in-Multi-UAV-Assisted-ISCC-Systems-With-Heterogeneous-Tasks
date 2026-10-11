"""Read-only feasibility diagnostics; NumPy only, independent of P5 linearization.

Residuals use the convention <= 0. Acceptance uses normalized residuals,
not an absolute watt tolerance. No function edits solver/environment arrays.
"""

import numpy as np

DEFAULT_RELATIVE_TOL = 1e-6
DEFAULT_MATRIX_TOL = 1e-7
DEFAULT_RANK_RELATIVE_TOL = 1e-5
STRICT_DELAY_TOL = 1e-10  # constraints/c09 and fusion_pc3p


def _check(raw, scale, tolerance, unit):
    raw = np.asarray(raw, dtype=float)
    scale = np.maximum(np.asarray(scale, dtype=float), 1e-300)
    with np.errstate(all="ignore"):
        normalized = raw / scale
    finite = bool(np.all(np.isfinite(normalized)))
    return {
        "raw_residual": raw.tolist(), "normalized_residual": normalized.tolist(),
        "scale": np.broadcast_to(scale, raw.shape).tolist(), "unit": unit,
        "tolerance": float(tolerance),
        "passed": finite and bool(np.all(normalized <= tolerance)),
        "max_violation": float(max(0.0, np.max(normalized))) if finite else float("inf"),
    }


def _unavailable(status, detail=None):
    return {"status": status, "evaluated": False, "passed": False,
            "failed_checks": [], "checks": {}, "detail": detail}


def _matching(allowed):
    """Maximum bipartite matching, including Hall conflicts (not row minima)."""
    allowed = np.asarray(allowed, dtype=bool)
    I, J = allowed.shape
    owners = [-1] * J

    def augment(i, visited):
        for j in np.flatnonzero(allowed[i]):
            j = int(j)
            if visited[j]:
                continue
            visited[j] = True
            if owners[j] < 0 or augment(owners[j], visited):
                owners[j] = i
                return True
        return False

    for i in np.argsort(allowed.sum(axis=1)):
        if not augment(int(i), [False] * J):
            return None
    result = [-1] * I
    for j, i in enumerate(owners):
        if i >= 0:
            result[i] = j
    return result


def sensing_diagnostics(params, env, *, cu_min_power=None, candidates=True,
                        relative_tol=DEFAULT_RELATIVE_TOL):
    """Exact c05/c06 necessary condition for fixed geometry and receive beams.

    Candidate matching only certifies sensing feasibility; it does not certify
    CU transmission, computation, or feasibility with other trajectories/beams.
    The CU projection deliberately follows inner/environment.py: dot(g, h).
    """
    I, J = params.I, params.J
    g = np.asarray(env.g_rec_beam, dtype=complex)
    h = np.asarray(env.h_cu_2_uav, dtype=complex)
    eta = np.asarray(env.eta_share, dtype=float)
    powers = np.asarray(env.p_cu_power, dtype=float)
    G = np.asarray(env.G_sen_corr, dtype=complex)
    values = (g, h, eta, powers, G, np.asarray(env.Gamma_sinr))
    if not all(np.all(np.isfinite(x)) for x in values):
        return {"status": "non_finite_input", "current_action_infeasible": None,
                "candidates": {"evaluated": False}}
    if params.sigma_2 <= 0 or params.eps_sinr <= 0 or params.P_max_uav <= 0:
        return {"status": "invalid_parameters", "current_action_infeasible": None,
                "candidates": {"evaluated": False}}
    projected = np.abs(np.sum(g[:, None, :] * h, axis=2)) ** 2
    interference_by_cu = eta * powers[None, :] * projected
    gamma = interference_by_cu.sum(axis=1) + params.sigma_2
    eigenvalues = np.linalg.eigvalsh((G + G.conj().transpose(0, 2, 1)) / 2)
    maximum = np.maximum(eigenvalues[:, -1], 0.0)
    numerator = params.P_max_uav * maximum
    requirement = params.eps_sinr * gamma
    margin = numerator / requirement
    required_power = np.divide(requirement, maximum, out=np.full(I, np.inf),
                               where=maximum > 0)
    failed = np.flatnonzero(margin < 1.0 - relative_tol).tolist()
    result = {
        "status": "checked", "scope": "fixed_position_and_receive_beam",
        "target_distances_m": np.asarray(env.d_uav_target).tolist(),
        "noise_power_w": float(params.sigma_2),
        "interference_by_cu_w": interference_by_cu.tolist(),
        "interference_w": interference_by_cu.sum(axis=1).tolist(),
        "gamma_w": gamma.tolist(), "lambda_max_g": maximum.tolist(),
        "sinr_threshold_linear": float(params.eps_sinr),
        "sinr_threshold_db": float(params.eps_sinr_db),
        "sinr_upper_bound": (numerator / gamma).tolist(),
        "margin": margin.tolist(), "min_margin": float(margin.min()),
        "required_sensing_power_w": required_power.tolist(),
        "current_action_infeasible": bool(failed), "failed_uavs": failed,
        "worst_uav": int(np.argmin(margin)), "relative_tolerance": relative_tol,
        "gamma_assembly_relative_error":
            ((gamma - np.asarray(env.Gamma_sinr)) / gamma).tolist(),
        "candidates": {"evaluated": False},
    }
    if candidates:
        floor = 0.0 if cu_min_power is None else float(cu_min_power)
        if not np.isfinite(floor) or floor < 0 or floor > params.P_max_cu:
            raise ValueError("cu_min_power must be a legal finite power")
        current = numerator[:, None] / (params.eps_sinr *
                                       (projected * powers[None, :] + params.sigma_2))
        lower = numerator[:, None] / (params.eps_sinr *
                                     (projected * floor + params.sigma_2))
        current_match = _matching(current >= 1 + relative_tol)
        lower_match = _matching(lower >= 1 + relative_tol)
        # Certification rejects only margins below 1-tol. Boundary/uncertain
        # edges must remain available when claiming no matching exists.
        possible_match = _matching(lower >= 1 - relative_tol)
        result["candidates"] = {
            "evaluated": True, "scope": "sensing_only_fixed_position_and_receive_beam",
            "current_power_margin": current.tolist(), "min_power_margin": lower.tolist(),
            "cu_min_power_w": floor, "cu_power_floor_source":
                "environment_action_space" if cu_min_power is not None else "relaxed_zero_power",
            "current_power_matching": current_match,
            "min_power_matching": lower_match,
            "no_matching_at_min_power_certified": possible_match is None,
            "complete_task_feasibility": "not_evaluated",
        }
    return result


def outer_action_checks(params, env, *, cu_min_power=None,
                        relative_tol=DEFAULT_RELATIVE_TOL):
    eta = np.asarray(env.eta_share, dtype=float)
    g = np.asarray(env.g_rec_beam, dtype=complex)
    p = np.asarray(env.p_cu_power, dtype=float)
    off = np.asarray(env.D_uav_off, dtype=float)
    before = np.asarray(env.q_uav_pos, dtype=float)
    after = np.asarray(env.q_uav_pos_next, dtype=float)
    if not all(np.all(np.isfinite(x)) for x in (eta, g, p, off, before, after)):
        return _unavailable("non_finite_input")
    checks = {}
    def add(name, raw, scale, unit="1"):
        checks[name] = _check(raw, scale, relative_tol, unit)
    add("matching_binary", np.abs(eta - np.round(eta)), 1)
    add("matching_entry_range", np.maximum(-eta, eta - 1), 1)
    add("matching_row_one", np.abs(eta.sum(axis=1) - 1), 1)
    add("matching_cu_unique", eta.sum(axis=0) - 1, 1)
    add("receive_beam_unit_norm", np.abs(np.sum(np.abs(g) ** 2, axis=1) - 1), 1)
    add("cu_power_upper", p - params.P_max_cu, params.P_max_cu, "W")
    floor = 0.0 if cu_min_power is None else float(cu_min_power)
    add("cu_power_lower", floor - p, params.P_max_cu, "W")
    add("uav_off_duration_nonnegative", -off, params.D_max_sen, "s")
    # A strictly positive remaining UAV compute interval is a domain condition.
    remaining = params.D_max_sen - params.D_bar_sen - off
    add("uav_compute_interval", -remaining, params.D_max_sen, "s")
    checks["uav_compute_interval"]["passed"] &= bool(np.all(remaining > 0))
    displacement = np.linalg.norm(after - before, axis=1)
    add("uav_speed_upper", displacement - params.uav_max_speed * params.tau_slot,
        max(params.uav_max_speed * params.tau_slot, 1e-12), "m")
    add("uav_speed_lower", params.uav_min_speed * params.tau_slot - displacement,
        max(params.uav_max_speed * params.tau_slot, 1e-12), "m")
    add("uav_altitude_constant", np.abs(after[:, 2] - before[:, 2]),
        max(abs(params.uav_height), 1.0), "m")
    pairs = [np.linalg.norm(after[i] - after[j]) for i in range(params.I)
             for j in range(i + 1, params.I)]
    if pairs:
        add("uav_collision", params.uav_safe_distance - np.asarray(pairs),
            max(params.uav_safe_distance, 1e-12), "m")
    failures = [k for k, v in checks.items() if not v["passed"]]
    return {"status": "checked", "evaluated": True, "passed": not failures,
            "failed_checks": failures, "checks": checks,
            "channel_time": "before_move", "collision_time": "after_move"}


def _matrix_validation(params, env, W, B, f, z, D, *, f_cu=None,
                       relative_tol=DEFAULT_RELATIVE_TOL,
                       matrix_tol=DEFAULT_MATRIX_TOL):
    checks = {}
    def add(name, raw, scale, unit="1", tol=None):
        checks[name] = _check(raw, scale, relative_tol if tol is None else tol, unit)
    for label, matrices in (("c03_sensing", W), ("c04_offloading", B)):
        hermitian = (matrices + matrices.conj().transpose(0, 2, 1)) / 2
        norms = np.maximum(np.linalg.norm(matrices, axis=(1, 2)), 1e-12)
        add(label + "_hermitian", np.linalg.norm(matrices - hermitian, axis=(1, 2)),
            norms, "W", matrix_tol)
        add(label + "_psd", -np.linalg.eigvalsh(hermitian)[:, 0], norms, "W", matrix_tol)
    trW = np.trace(W, axis1=1, axis2=2).real
    trB = np.trace(B, axis1=1, axis2=2).real
    signal = np.einsum("ijk,ikj->i", env.G_sen_corr, W).real
    bs_sen = np.einsum("ijk,ikj->i", env.H_uav_bs, W).real
    bs_off = np.einsum("ijk,ikj->i", env.H_uav_bs, B).real
    off = np.asarray(env.D_uav_off, dtype=float)
    eta = np.asarray(env.eta_share, dtype=float)
    dmax = np.asarray(params.D_max_cu, dtype=float)
    remaining = dmax - D
    cu_signal = np.asarray(env.p_cu_power) * np.abs(env.h_cu_2_bs) ** 2
    with np.errstate(all="ignore"):
        off_bits = off * params.B * np.log1p(bs_off / env.Phi_off_inr) / np.log(2)
        sensing_rate = params.xi_1 * np.log1p(params.xi_2 * signal / env.Gamma_sinr) / np.log(2)
        cu_bits = params.B * (
            params.D_bar_sen * eta.sum(axis=0) *
            np.log1p(cu_signal / (eta.T @ bs_sen + params.sigma_2)) / np.log(2)
            + (eta.T @ off) *
            np.log1p(cu_signal / (eta.T @ bs_off + params.sigma_2)) / np.log(2)
            + (D - env.Theta_cu_time) * np.asarray(env.upsilon_cu_rate))
        required_f_cu = params.C_cu_cycles * params.L_cu_task / remaining
    actual_f_cu = required_f_cu if f_cu is None else f_cu
    add("variable_uav_frequency_nonnegative", -f, params.F_max, "Hz")
    add("variable_z_nonnegative", -z, np.maximum(np.abs(z), 1e-12), "bit")
    add("variable_cu_duration_nonnegative", -D, dmax, "s")
    add("c01_task_fresh", params.D_bar_sen + off - eta @ D,
        np.maximum(params.D_bar_sen + off, 1e-12), "s")
    compute_work = params.C_sen * z
    compute_capacity = (params.D_max_sen - params.D_bar_sen - off) * f
    add("c02_uav_task_delay", compute_work - compute_capacity,
        np.maximum(np.maximum(np.abs(compute_work), np.abs(compute_capacity)), 1e-12), "cycles")
    add("c05_sensing_max_power", trW - params.P_max_uav, params.P_max_uav, "W")
    add("c06_sensing_sinr", params.eps_sinr * env.Gamma_sinr - signal,
        params.eps_sinr * env.Gamma_sinr, "W")
    add("c07_offloading_max_power", trB - params.P_max_uav, params.P_max_uav, "W")
    add("c08_uav_offloading", z - off_bits,
        np.maximum(np.maximum(np.abs(z), np.abs(off_bits)), 1e-12), "bit")
    add("c09_cu_delay", D - dmax + STRICT_DELAY_TOL, dmax, "s")
    checks["c09_cu_delay"]["passed"] &= bool(np.all(remaining > 0))
    add("cu_frequency_workload", required_f_cu - actual_f_cu,
        np.maximum(np.abs(required_f_cu), 1e-12), "Hz")
    add("cu_frequency_nonnegative", -actual_f_cu, params.F_max, "Hz")
    add("c10_bs_max_frequency", f.sum() + actual_f_cu.sum() - params.F_max,
        params.F_max, "Hz")
    add("c11_original_sensing_auxiliary", sensing_rate - z,
        np.maximum(np.maximum(np.abs(sensing_rate), np.abs(z)), 1e-12), "bit")
    add("c12_original_cu_offloading", params.L_cu_task - cu_bits,
        np.maximum(params.L_cu_task, 1e-12), "bit")
    failures = [k for k, v in checks.items() if not v["passed"]]
    return {"status": "checked", "evaluated": True, "passed": not failures,
            "failed_checks": failures, "checks": checks,
            "max_normalized_violation": max(x["max_violation"] for x in checks.values()),
            "observables": {"sensing_signal_w": signal.tolist(),
                            "uav_offloading_bits": off_bits.tolist(),
                            "sensing_auxiliary_lower_bound": sensing_rate.tolist(),
                            "cu_offloading_bits": cu_bits.tolist(),
                            "uav_frequency_hz": f.tolist(),
                            "cu_frequency_hz": actual_f_cu.tolist(),
                            "bs_frequency_total_hz": float(f.sum() + actual_f_cu.sum())}}


def validate_original_solution(params, env, solution, *, relative_tol=DEFAULT_RELATIVE_TOL,
                               matrix_tol=DEFAULT_MATRIX_TOL,
                               rank_relative_tol=DEFAULT_RANK_RELATIVE_TOL):
    """Check covariance and recovered beams separately against original c01-c12.

    c13 and CCCP epigraph variables are algorithmic auxiliaries, not original
    physical constraints. Solver 'converged' is recorded separately, never used
    as a substitute for checking the supplied numerical solution.
    """
    unavailable = _unavailable("no_solution")
    result = {"covariance": unavailable, "recovered_beams": dict(unavailable),
              "rank": {"evaluated": False, "passed": False},
              "physical_solution_verified": False,
              "tolerances": {"relative": relative_tol, "matrix": matrix_tol,
                             "rank_relative": rank_relative_tol,
                             "rank_absolute": float(params.gamma_2)}}
    if solution.get("W_sen_beam") is None:
        return result
    shapes = {"W_sen_beam": (params.I, params.N, params.N),
              "B_off_beam": (params.I, params.N, params.N),
              "f_uav_freq": (params.I,), "z_aux_rate": (params.I,),
              "D_cu_off": (params.J,)}
    arrays = {}
    for name, shape in shapes.items():
        value = solution.get(name)
        if value is None or np.asarray(value).shape != shape:
            failure = _unavailable("invalid_shape", name)
            result.update(covariance=failure, recovered_beams=dict(failure))
            return result
        arrays[name] = np.asarray(value)
    if solution.get("f_cu_freq") is not None:
        arrays["f_cu_freq"] = np.asarray(solution["f_cu_freq"])
        if arrays["f_cu_freq"].shape != (params.J,):
            result["covariance"] = _unavailable("invalid_shape", "f_cu_freq")
            return result
    if not all(np.all(np.isfinite(x)) for x in arrays.values()):
        failure = _unavailable("non_finite_solution")
        result.update(covariance=failure, recovered_beams=dict(failure))
        return result
    constants = [env.G_sen_corr, env.H_uav_bs, env.Gamma_sinr, env.Phi_off_inr,
                 env.eta_share, env.Theta_cu_time, env.upsilon_cu_rate,
                 env.p_cu_power, env.h_cu_2_bs]
    if not all(np.all(np.isfinite(x)) for x in constants):
        result["covariance"] = _unavailable("non_finite_input")
        return result
    if np.any(np.asarray(env.Gamma_sinr) <= 0) or np.any(np.asarray(env.Phi_off_inr) <= 0):
        result["covariance"] = _unavailable("invalid_domain", "Gamma/Phi must be positive")
        return result
    W, B = arrays["W_sen_beam"], arrays["B_off_beam"]
    f, z, D = arrays["f_uav_freq"], arrays["z_aux_rate"], arrays["D_cu_off"]
    kwargs = dict(f_cu=arrays.get("f_cu_freq"), relative_tol=relative_tol, matrix_tol=matrix_tol)
    result["covariance"] = _matrix_validation(params, env, W, B, f, z, D, **kwargs)
    rank_gap, rank_relative = [], []
    for matrices in (W, B):
        eig = np.linalg.eigvalsh((matrices + matrices.conj().transpose(0, 2, 1)) / 2)
        # PSD is checked separately; rank measures energy outside the largest eigenvalue.
        gap = np.maximum(eig[:, :-1], 0).sum(axis=1)
        rank_gap.extend(gap.tolist())
        rank_relative.extend((gap / np.maximum(np.maximum(eig[:, -1], 0), 1e-12)).tolist())
    result["rank"] = {"evaluated": True,
        "absolute_gap_w": rank_gap, "relative_gap": rank_relative,
        "passed": bool(sum(rank_gap) <= params.gamma_2 and
                       max(rank_relative, default=0) <= rank_relative_tol)}
    w, b = solution.get("w_sen_beam"), solution.get("b_off_beam")
    if w is None or b is None:
        result["recovered_beams"] = _unavailable("missing_recovered_beams")
    elif np.asarray(w).shape != (params.I, params.N) or np.asarray(b).shape != (params.I, params.N):
        result["recovered_beams"] = _unavailable("invalid_shape", "recovered beams")
    elif not np.all(np.isfinite(w)) or not np.all(np.isfinite(b)):
        result["recovered_beams"] = _unavailable("non_finite_solution", "recovered beams")
    else:
        w, b = np.asarray(w), np.asarray(b)
        recovered_W = np.einsum("ij,ik->ijk", w, w.conj())
        recovered_B = np.einsum("ij,ik->ijk", b, b.conj())
        result["recovered_beams"] = _matrix_validation(
            params, env, recovered_W, recovered_B, f, z, D, **kwargs)
    result["physical_solution_verified"] = bool(
        result["covariance"]["passed"] and result["rank"]["passed"] and
        result["recovered_beams"]["passed"])
    return result
