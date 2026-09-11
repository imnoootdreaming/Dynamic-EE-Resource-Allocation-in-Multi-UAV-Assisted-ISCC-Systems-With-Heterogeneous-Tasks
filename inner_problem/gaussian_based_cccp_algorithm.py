import csv
import importlib
import os
from datetime import datetime

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


def _penalty_module():
    return importlib.import_module("penalty_based_cccp_algorithm")


def build_constraint_map(*args, **kwargs):
    return _penalty_module().build_constraint_map(*args, **kwargs)


def build_static_constraint_data(*args, **kwargs):
    return _penalty_module().build_static_constraint_data(*args, **kwargs)


def compute_largest_eigenvector(*args, **kwargs):
    return _penalty_module().compute_largest_eigenvector(*args, **kwargs)


def compute_obj_fun(*args, **kwargs):
    return _penalty_module().compute_obj_fun(*args, **kwargs)


def compute_pure_energy_value(*args, **kwargs):
    return _penalty_module().compute_pure_energy_value(*args, **kwargs)


def define_constraint(*args, **kwargs):
    return _penalty_module().define_constraint(*args, **kwargs)


def extract_matched_sensing_channel(*args, **kwargs):
    return _penalty_module().extract_matched_sensing_channel(*args, **kwargs)


def initialize_uav_beams(*args, **kwargs):
    return _penalty_module().initialize_uav_beams(*args, **kwargs)


def parse_group_list(*args, **kwargs):
    return _penalty_module().parse_group_list(*args, **kwargs)


def select_constraints(*args, **kwargs):
    return _penalty_module().select_constraints(*args, **kwargs)


def solve_inner_problem_with_fusion(*args, **kwargs):
    return _penalty_module().solve_inner_problem_with_fusion(*args, **kwargs)


def update_constraint_linearization_parameters(*args, **kwargs):
    return _penalty_module().update_constraint_linearization_parameters(*args, **kwargs)


def update_obj5_linearization_parameters(*args, **kwargs):
    return _penalty_module().update_obj5_linearization_parameters(*args, **kwargs)


def _now_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _is_success_status(status):
    status_key = str(status).strip().lower()
    status_key = status_key.split(".")[-1]
    status_key = status_key.replace(" ", "").replace("-", "").replace("_", "")
    return status_key in (
        "optimal",
        "optimalinaccurate",
        "nearoptimal",
        "feasible",
    )


_CVXPY_COO_BACKEND_SUPPORTED = False


def _solve_cvxpy_problem(problem, warm_start=True, verbose=False, allow_fallback_solver=False):
    # The current environment does not support canon_backend="COO" in CVXPY.
    # Use the default canonicalization backend directly. For Gaussian recovery
    # candidates, MOSEK may occasionally fail numerically on poor directions, so
    # we optionally retry with SCS as a robustness fallback.
    solver_attempts = [
        (
            "MOSEK",
            {
                "solver": cp.MOSEK,
                "warm_start": warm_start,
                "verbose": verbose,
            },
        )
    ]
    if allow_fallback_solver:
        solver_attempts.append(
            (
                "SCS",
                {
                    "solver": cp.SCS,
                    "warm_start": False,
                    "verbose": verbose,
                    "eps": 1e-4,
                    "max_iters": 20000,
                },
            )
        )

    last_exc = None
    for solver_name, solver_kwargs in solver_attempts:
        try:
            problem.solve(**solver_kwargs)
            return solver_name
        except cp.error.SolverError as exc:
            last_exc = exc
            continue

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("CVXPY solve failed before any solver attempt completed.")


def _hermitian_psd_projection(matrix):
    matrix = np.asarray(matrix, dtype=complex)
    matrix = (matrix + matrix.conj().T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues = np.clip(np.real(eigenvalues), 0.0, None)
    if np.all(eigenvalues <= 0):
        zero_matrix = np.zeros_like(matrix)
        return zero_matrix, zero_matrix
    sqrt_eigenvalues = np.sqrt(eigenvalues)
    factor = eigenvectors @ np.diag(sqrt_eigenvalues)
    projected = factor @ factor.conj().T
    projected = (projected + projected.conj().T) / 2
    return projected, factor


def _vector_to_unit_trace_rank1(vector, vector_dim):
    vector = np.asarray(vector, dtype=complex).reshape(-1)
    norm_val = np.linalg.norm(vector)
    if norm_val <= 1e-12:
        vector = np.zeros(vector_dim, dtype=complex)
        vector[0] = 1.0
        norm_val = 1.0
    vector = vector / norm_val
    rank1_matrix = np.outer(vector, vector.conj())
    rank1_matrix = (rank1_matrix + rank1_matrix.conj().T) / 2
    return rank1_matrix


def _dominant_rank1_direction(matrix):
    projected, _ = _hermitian_psd_projection(matrix)
    if np.allclose(projected, 0.0):
        return _vector_to_unit_trace_rank1(
            vector=np.zeros(projected.shape[0], dtype=complex),
            vector_dim=projected.shape[0],
        )
    dominant_vector = compute_largest_eigenvector(projected)
    return _vector_to_unit_trace_rank1(
        vector=dominant_vector,
        vector_dim=projected.shape[0],
    )


def _sample_direction_from_factor(factor, fallback_direction, rng):
    vector_dim = fallback_direction.shape[0]
    if factor.size == 0 or np.allclose(factor, 0.0):
        return fallback_direction
    gaussian_vector = (
        rng.standard_normal(vector_dim) + 1j * rng.standard_normal(vector_dim)
    ) / np.sqrt(2.0)
    sample = factor @ gaussian_vector
    if np.linalg.norm(sample) <= 1e-12:
        return fallback_direction
    return _vector_to_unit_trace_rank1(sample, vector_dim)


def _compute_rank1_metrics(sen_beams, off_beams):
    rank1_gap_max = 0.0
    rank1_gap_sum = 0.0
    for sen_beam, off_beam in zip(sen_beams, off_beams):
        sen_beam = np.asarray(sen_beam, dtype=complex)
        off_beam = np.asarray(off_beam, dtype=complex)
        sen_gap = np.real(np.trace(sen_beam)) - np.linalg.norm(sen_beam, 2)
        off_gap = np.real(np.trace(off_beam)) - np.linalg.norm(off_beam, 2)
        sen_gap = max(float(sen_gap), 0.0)
        off_gap = max(float(off_gap), 0.0)
        rank1_gap_sum += sen_gap + off_gap
        rank1_gap_max = max(rank1_gap_max, sen_gap, off_gap)
    return float(rank1_gap_max), float(rank1_gap_sum)


def _dominant_projection_solution(
    args,
    relaxed_solution,
    cus_entertaining_task_size,
    uavs_off_duration,
    cus_off_power,
    uavs_pos_pre,
    uavs_pos_cur,
):
    projected_sen_beams = []
    projected_off_beams = []

    for sen_beam in relaxed_solution["uavs_sen_beams"]:
        direction = _dominant_rank1_direction(sen_beam)
        projected_sen_beams.append(float(np.real(np.trace(sen_beam))) * direction)

    for off_beam in relaxed_solution["uavs_off_beams"]:
        direction = _dominant_rank1_direction(off_beam)
        projected_off_beams.append(float(np.real(np.trace(off_beam))) * direction)

    pure_energy = compute_pure_energy_value(
        args=args,
        cur_uavs_sen_beams=projected_sen_beams,
        cur_uavs_off_beams=projected_off_beams,
        cur_bs_2_uav_freqs_norm=relaxed_solution["bs_2_uav_freqs_norm"],
        cur_auxiliary_variable_z=relaxed_solution["auxiliary_variable_z"],
        cur_cus_off_duration=relaxed_solution["cus_off_duration"],
        cus_entertaining_task_size=cus_entertaining_task_size,
        uavs_off_duration=uavs_off_duration,
        cus_off_power=cus_off_power,
        uavs_pos_pre=uavs_pos_pre,
        uavs_pos_cur=uavs_pos_cur,
    )
    rank1_gap_max, rank1_gap_sum = _compute_rank1_metrics(projected_sen_beams, projected_off_beams)
    return {
        "candidate_name": "dominant-projection-fallback",
        "uavs_sen_beams": projected_sen_beams,
        "uavs_off_beams": projected_off_beams,
        "bs_2_uav_freqs_norm": np.asarray(relaxed_solution["bs_2_uav_freqs_norm"], dtype=float),
        "auxiliary_variable_z": np.asarray(relaxed_solution["auxiliary_variable_z"], dtype=float),
        "cus_off_duration": np.asarray(relaxed_solution["cus_off_duration"], dtype=float),
        "t": np.asarray(relaxed_solution["t"], dtype=float),
        "pure_energy": float(np.real(pure_energy)),
        "rank1_gap_max": rank1_gap_max,
        "rank1_gap_sum": rank1_gap_sum,
    }


def _refresh_linearization_state(state, args, solution):
    for i in range(args.uavs_num):
        state["hat_uav_sen_beams"][i].value = solution["uavs_sen_beams"][i]
        state["hat_uav_off_beams"][i].value = solution["uavs_off_beams"][i]
    state["hat_auxiliary_variable_z"].value = np.asarray(solution["auxiliary_variable_z"], dtype=float)
    state["hat_bs_2_uav_freqs_norm"].value = np.asarray(solution["bs_2_uav_freqs_norm"], dtype=float)

    update_obj5_linearization_parameters(
        args=args,
        hat_auxiliary_variable_z=state["hat_auxiliary_variable_z"],
        hat_bs_2_uav_freqs_norm=state["hat_bs_2_uav_freqs_norm"],
        obj5_coef_z=state["obj5_coef_z"],
        obj5_coef_f=state["obj5_coef_f"],
        obj5_const_terms=state["obj5_const_terms"],
    )
    update_constraint_linearization_parameters(
        args=args,
        hat_uav_sen_beams=state["hat_uav_sen_beams"],
        hat_uav_off_beams=state["hat_uav_off_beams"],
        static_constraint_data=state["static_constraint_data"],
        c44_const_terms=state["c44_const_terms"],
        c44_grad_mats=state["c44_grad_mats"],
        c45_term4_const=state["c45_term4_const"],
        c45_term5_const=state["c45_term5_const"],
        c45_coef_w=state["c45_coef_w"],
        c45_coef_b=state["c45_coef_b"],
        c45_const_w=state["c45_const_w"],
        c45_const_b=state["c45_const_b"],
    )


def _initialize_gaussian_cccp_state(
    args,
    uavs_2_cus_channels,
    uavs_2_bs_channels,
    cus_2_bs_channels,
    uavs_2_targets_channels,
    uavs_targets_matched_matrix,
    uavs_cus_matched_matrix,
    uavs_off_duration,
    cus_off_power,
):
    matched_uav_sensing_channel = extract_matched_sensing_channel(
        uavs_targets_matched_matrix=uavs_targets_matched_matrix,
        uavs_2_targets_channels=uavs_2_targets_channels,
    )

    hat_auxiliary_variable_z = cp.Parameter(
        args.uavs_num,
        nonneg=True,
        value=np.ones(args.uavs_num) * (1e5 / args.z_scale),
    )
    hat_bs_2_uav_freqs_norm = cp.Parameter(
        args.uavs_num,
        nonneg=True,
        value=np.ones(args.uavs_num) * (args.bs_max_freq / args.freq_scale / args.uavs_num / 2),
    )
    hat_uav_sen_beams = [
        cp.Parameter(
            (args.antenna_nums, args.antenna_nums),
            hermitian=True,
            value=np.eye(args.antenna_nums, dtype=complex),
        )
        for _ in range(args.uavs_num)
    ]
    hat_uav_off_beams = [
        cp.Parameter(
            (args.antenna_nums, args.antenna_nums),
            hermitian=True,
            value=np.eye(args.antenna_nums, dtype=complex),
        )
        for _ in range(args.uavs_num)
    ]

    initial_sen_beams, initial_off_beams = initialize_uav_beams(
        args=args,
        matched_uav_sensing_channel=matched_uav_sensing_channel,
        uavs_2_bs_channels=uavs_2_bs_channels,
    )
    for i in range(args.uavs_num):
        hat_uav_sen_beams[i].value = initial_sen_beams[i]
        hat_uav_off_beams[i].value = initial_off_beams[i]

    static_constraint_data = build_static_constraint_data(
        args=args,
        matched_uav_sensing_channel=matched_uav_sensing_channel,
        uavs_2_cus_channels=uavs_2_cus_channels,
        uavs_2_bs_channels=uavs_2_bs_channels,
        cus_2_bs_channels=cus_2_bs_channels,
        uavs_cus_matched_matrix=uavs_cus_matched_matrix,
        uavs_off_duration=uavs_off_duration,
        cus_off_power=cus_off_power,
    )

    obj5_coef_z = cp.Parameter(args.uavs_num, value=np.zeros(args.uavs_num))
    obj5_coef_f = cp.Parameter(args.uavs_num, value=np.zeros(args.uavs_num))
    obj5_const_terms = cp.Parameter(args.uavs_num, value=np.zeros(args.uavs_num))
    c44_const_terms = cp.Parameter(args.uavs_num, value=np.zeros(args.uavs_num))
    c44_grad_mats = [
        cp.Parameter(
            (args.antenna_nums, args.antenna_nums),
            hermitian=True,
            value=np.eye(args.antenna_nums, dtype=complex),
        )
        for _ in range(args.uavs_num)
    ]
    c45_term4_const = cp.Parameter(args.cus_num, value=np.zeros(args.cus_num))
    c45_term5_const = cp.Parameter(args.cus_num, value=np.zeros(args.cus_num))
    c45_coef_w = cp.Parameter(args.cus_num, value=np.zeros(args.cus_num))
    c45_coef_b = cp.Parameter(args.cus_num, value=np.zeros(args.cus_num))
    c45_const_w = cp.Parameter(args.cus_num, value=np.zeros(args.cus_num))
    c45_const_b = cp.Parameter(args.cus_num, value=np.zeros(args.cus_num))

    dummy_rank1_sen_proj_mats = [
        cp.Parameter(
            (args.antenna_nums, args.antenna_nums),
            hermitian=True,
            value=np.eye(args.antenna_nums, dtype=complex),
        )
        for _ in range(args.uavs_num)
    ]
    dummy_rank1_off_proj_mats = [
        cp.Parameter(
            (args.antenna_nums, args.antenna_nums),
            hermitian=True,
            value=np.eye(args.antenna_nums, dtype=complex),
        )
        for _ in range(args.uavs_num)
    ]
    dummy_rank1_sen_scaled_proj_mats = [
        cp.Parameter(
            (args.antenna_nums, args.antenna_nums),
            hermitian=True,
            value=np.eye(args.antenna_nums, dtype=complex),
        )
        for _ in range(args.uavs_num)
    ]
    dummy_rank1_off_scaled_proj_mats = [
        cp.Parameter(
            (args.antenna_nums, args.antenna_nums),
            hermitian=True,
            value=np.eye(args.antenna_nums, dtype=complex),
        )
        for _ in range(args.uavs_num)
    ]
    dummy_rank1_const_terms = cp.Parameter(args.uavs_num, value=np.zeros(args.uavs_num))
    zero_penalty = cp.Parameter(nonneg=True, value=0.0)

    update_obj5_linearization_parameters(
        args=args,
        hat_auxiliary_variable_z=hat_auxiliary_variable_z,
        hat_bs_2_uav_freqs_norm=hat_bs_2_uav_freqs_norm,
        obj5_coef_z=obj5_coef_z,
        obj5_coef_f=obj5_coef_f,
        obj5_const_terms=obj5_const_terms,
    )
    update_constraint_linearization_parameters(
        args=args,
        hat_uav_sen_beams=hat_uav_sen_beams,
        hat_uav_off_beams=hat_uav_off_beams,
        static_constraint_data=static_constraint_data,
        c44_const_terms=c44_const_terms,
        c44_grad_mats=c44_grad_mats,
        c45_term4_const=c45_term4_const,
        c45_term5_const=c45_term5_const,
        c45_coef_w=c45_coef_w,
        c45_coef_b=c45_coef_b,
        c45_const_w=c45_const_w,
        c45_const_b=c45_const_b,
    )

    return {
        "static_constraint_data": static_constraint_data,
        "hat_auxiliary_variable_z": hat_auxiliary_variable_z,
        "hat_bs_2_uav_freqs_norm": hat_bs_2_uav_freqs_norm,
        "hat_uav_sen_beams": hat_uav_sen_beams,
        "hat_uav_off_beams": hat_uav_off_beams,
        "obj5_coef_z": obj5_coef_z,
        "obj5_coef_f": obj5_coef_f,
        "obj5_const_terms": obj5_const_terms,
        "c44_const_terms": c44_const_terms,
        "c44_grad_mats": c44_grad_mats,
        "c45_term4_const": c45_term4_const,
        "c45_term5_const": c45_term5_const,
        "c45_coef_w": c45_coef_w,
        "c45_coef_b": c45_coef_b,
        "c45_const_w": c45_const_w,
        "c45_const_b": c45_const_b,
        "dummy_rank1_sen_proj_mats": dummy_rank1_sen_proj_mats,
        "dummy_rank1_off_proj_mats": dummy_rank1_off_proj_mats,
        "dummy_rank1_sen_scaled_proj_mats": dummy_rank1_sen_scaled_proj_mats,
        "dummy_rank1_off_scaled_proj_mats": dummy_rank1_off_scaled_proj_mats,
        "dummy_rank1_const_terms": dummy_rank1_const_terms,
        "zero_penalty": zero_penalty,
    }


def _build_relaxed_problem_bundle(
    args,
    state,
    cus_entertaining_task_size,
    uavs_off_duration,
    cus_off_power,
    uavs_pos_pre,
    uavs_pos_cur,
):
    var_uavs_sen_beam = [
        cp.Variable((args.antenna_nums, args.antenna_nums), hermitian=True)
        for _ in range(args.uavs_num)
    ]
    var_uavs_off_beam = [
        cp.Variable((args.antenna_nums, args.antenna_nums), hermitian=True)
        for _ in range(args.uavs_num)
    ]
    var_bs_2_uav_freqs_norm = cp.Variable(args.uavs_num, nonneg=True)
    var_auxiliary_variable_z = cp.Variable(args.uavs_num, nonneg=True)
    var_cus_off_duration = cp.Variable(args.cus_num, nonneg=True)
    var_t = cp.Variable(args.uavs_num, nonneg=True)
    first_iter_rank_boost_lb = cp.Parameter(
        nonneg=True,
        value=args.first_iter_rank_boost_eps if args.enable_first_iter_rank_boost else 0.0,
    )

    objective = compute_obj_fun(
        args=args,
        var_uavs_sen_beam=var_uavs_sen_beam,
        var_uavs_off_beam=var_uavs_off_beam,
        var_bs_2_uav_freqs_norm=var_bs_2_uav_freqs_norm,
        var_auxiliary_variable_z=var_auxiliary_variable_z,
        var_cus_off_duration=var_cus_off_duration,
        var_t=var_t,
        hat_auxiliary_variable_z=state["hat_auxiliary_variable_z"],
        hat_bs_2_uav_freqs_norm=state["hat_bs_2_uav_freqs_norm"],
        hat_uav_sen_beams=state["hat_uav_sen_beams"],
        hat_uav_off_beams=state["hat_uav_off_beams"],
        rank1_sen_proj_mats=state["dummy_rank1_sen_proj_mats"],
        rank1_off_proj_mats=state["dummy_rank1_off_proj_mats"],
        rank1_sen_scaled_proj_mats=state["dummy_rank1_sen_scaled_proj_mats"],
        rank1_off_scaled_proj_mats=state["dummy_rank1_off_scaled_proj_mats"],
        rank1_const_terms=state["dummy_rank1_const_terms"],
        obj5_coef_z=state["obj5_coef_z"],
        obj5_coef_f=state["obj5_coef_f"],
        obj5_const_terms=state["obj5_const_terms"],
        cus_entertaining_task_size=cus_entertaining_task_size,
        uavs_off_duration=uavs_off_duration,
        cus_off_power=cus_off_power,
        uavs_pos_pre=uavs_pos_pre,
        uavs_pos_cur=uavs_pos_cur,
        cur_penalty_factor=state["zero_penalty"],
        use_penalty_rank1=False,
    )

    all_constraints = define_constraint(
        args=args,
        var_uavs_sen_beam=var_uavs_sen_beam,
        var_uavs_off_beam=var_uavs_off_beam,
        var_bs_2_uav_freqs_norm=var_bs_2_uav_freqs_norm,
        var_auxiliary_variable_z=var_auxiliary_variable_z,
        var_cus_off_duration=var_cus_off_duration,
        var_t=var_t,
        first_iter_rank_boost_lb=first_iter_rank_boost_lb,
        hat_uav_sen_beams=state["hat_uav_sen_beams"],
        hat_uav_off_beams=state["hat_uav_off_beams"],
        cus_entertaining_task_size=cus_entertaining_task_size,
        uavs_off_duration=uavs_off_duration,
        cus_off_power=cus_off_power,
        static_constraint_data=state["static_constraint_data"],
        c44_const_terms=state["c44_const_terms"],
        c44_grad_mats=state["c44_grad_mats"],
        c45_term4_const=state["c45_term4_const"],
        c45_term5_const=state["c45_term5_const"],
        c45_coef_w=state["c45_coef_w"],
        c45_coef_b=state["c45_coef_b"],
        c45_const_w=state["c45_const_w"],
        c45_const_b=state["c45_const_b"],
    )
    constraint_map = build_constraint_map(args, all_constraints)
    constraints, active_groups = select_constraints(
        constraint_map=constraint_map,
        include_groups=parse_group_list(args.constraint_include_groups),
        exclude_groups=parse_group_list(args.constraint_exclude_groups),
    )
    problem = cp.Problem(cp.Minimize(objective), constraints)
    return {
        "problem": problem,
        "var_uavs_sen_beam": var_uavs_sen_beam,
        "var_uavs_off_beam": var_uavs_off_beam,
        "var_bs_2_uav_freqs_norm": var_bs_2_uav_freqs_norm,
        "var_auxiliary_variable_z": var_auxiliary_variable_z,
        "var_cus_off_duration": var_cus_off_duration,
        "var_t": var_t,
        "first_iter_rank_boost_lb": first_iter_rank_boost_lb,
        "active_groups": active_groups,
    }


def _build_gaussian_recovery_problem_bundle(
    args,
    state,
    cus_entertaining_task_size,
    uavs_off_duration,
    cus_off_power,
    uavs_pos_pre,
    uavs_pos_cur,
):
    sen_direction_params = [
        cp.Parameter(
            (args.antenna_nums, args.antenna_nums),
            hermitian=True,
            value=np.eye(args.antenna_nums, dtype=complex) / args.antenna_nums,
        )
        for _ in range(args.uavs_num)
    ]
    off_direction_params = [
        cp.Parameter(
            (args.antenna_nums, args.antenna_nums),
            hermitian=True,
            value=np.eye(args.antenna_nums, dtype=complex) / args.antenna_nums,
        )
        for _ in range(args.uavs_num)
    ]
    sen_power_vars = cp.Variable(args.uavs_num, nonneg=True)
    off_power_vars = cp.Variable(args.uavs_num, nonneg=True)
    var_bs_2_uav_freqs_norm = cp.Variable(args.uavs_num, nonneg=True)
    var_auxiliary_variable_z = cp.Variable(args.uavs_num, nonneg=True)
    var_cus_off_duration = cp.Variable(args.cus_num, nonneg=True)
    var_t = cp.Variable(args.uavs_num, nonneg=True)

    sen_beam_exprs = [sen_power_vars[i] * sen_direction_params[i] for i in range(args.uavs_num)]
    off_beam_exprs = [off_power_vars[i] * off_direction_params[i] for i in range(args.uavs_num)]
    zero_rank_boost = cp.Parameter(nonneg=True, value=0.0)

    objective = compute_obj_fun(
        args=args,
        var_uavs_sen_beam=sen_beam_exprs,
        var_uavs_off_beam=off_beam_exprs,
        var_bs_2_uav_freqs_norm=var_bs_2_uav_freqs_norm,
        var_auxiliary_variable_z=var_auxiliary_variable_z,
        var_cus_off_duration=var_cus_off_duration,
        var_t=var_t,
        hat_auxiliary_variable_z=state["hat_auxiliary_variable_z"],
        hat_bs_2_uav_freqs_norm=state["hat_bs_2_uav_freqs_norm"],
        hat_uav_sen_beams=state["hat_uav_sen_beams"],
        hat_uav_off_beams=state["hat_uav_off_beams"],
        rank1_sen_proj_mats=state["dummy_rank1_sen_proj_mats"],
        rank1_off_proj_mats=state["dummy_rank1_off_proj_mats"],
        rank1_sen_scaled_proj_mats=state["dummy_rank1_sen_scaled_proj_mats"],
        rank1_off_scaled_proj_mats=state["dummy_rank1_off_scaled_proj_mats"],
        rank1_const_terms=state["dummy_rank1_const_terms"],
        obj5_coef_z=state["obj5_coef_z"],
        obj5_coef_f=state["obj5_coef_f"],
        obj5_const_terms=state["obj5_const_terms"],
        cus_entertaining_task_size=cus_entertaining_task_size,
        uavs_off_duration=uavs_off_duration,
        cus_off_power=cus_off_power,
        uavs_pos_pre=uavs_pos_pre,
        uavs_pos_cur=uavs_pos_cur,
        cur_penalty_factor=state["zero_penalty"],
        use_penalty_rank1=False,
    )

    all_constraints = define_constraint(
        args=args,
        var_uavs_sen_beam=sen_beam_exprs,
        var_uavs_off_beam=off_beam_exprs,
        var_bs_2_uav_freqs_norm=var_bs_2_uav_freqs_norm,
        var_auxiliary_variable_z=var_auxiliary_variable_z,
        var_cus_off_duration=var_cus_off_duration,
        var_t=var_t,
        first_iter_rank_boost_lb=zero_rank_boost,
        hat_uav_sen_beams=state["hat_uav_sen_beams"],
        hat_uav_off_beams=state["hat_uav_off_beams"],
        cus_entertaining_task_size=cus_entertaining_task_size,
        uavs_off_duration=uavs_off_duration,
        cus_off_power=cus_off_power,
        static_constraint_data=state["static_constraint_data"],
        c44_const_terms=state["c44_const_terms"],
        c44_grad_mats=state["c44_grad_mats"],
        c45_term4_const=state["c45_term4_const"],
        c45_term5_const=state["c45_term5_const"],
        c45_coef_w=state["c45_coef_w"],
        c45_coef_b=state["c45_coef_b"],
        c45_const_w=state["c45_const_w"],
        c45_const_b=state["c45_const_b"],
    )
    constraint_map = build_constraint_map(args, all_constraints)
    recovery_exclude_groups = parse_group_list(args.constraint_exclude_groups)
    if recovery_exclude_groups is None:
        recovery_exclude_groups = []
    # In the recovery stage, W_i = p_i V_i and B_i = q_i U_i with p_i,q_i >= 0
    # and rank-one PSD direction matrices V_i,U_i. Hence PSD constraints 4.23
    # and 4.25 are automatically satisfied and can be dropped to avoid
    # unnecessary semidefinite cone canonicalization on affine matrix products.
    recovery_exclude_groups = list(dict.fromkeys(recovery_exclude_groups + ["4.23", "4.25"]))
    constraints, _ = select_constraints(
        constraint_map=constraint_map,
        include_groups=parse_group_list(args.constraint_include_groups),
        exclude_groups=recovery_exclude_groups,
    )
    problem = cp.Problem(cp.Minimize(objective), constraints)
    return {
        "problem": problem,
        "sen_direction_params": sen_direction_params,
        "off_direction_params": off_direction_params,
        "sen_power_vars": sen_power_vars,
        "off_power_vars": off_power_vars,
        "var_bs_2_uav_freqs_norm": var_bs_2_uav_freqs_norm,
        "var_auxiliary_variable_z": var_auxiliary_variable_z,
        "var_cus_off_duration": var_cus_off_duration,
        "var_t": var_t,
        "last_exception": None,
        "last_solve_backend": None,
    }


def _extract_relaxed_solution_from_bundle(relaxed_bundle):
    return {
        "uavs_sen_beams": [np.asarray(var.value, dtype=complex) for var in relaxed_bundle["var_uavs_sen_beam"]],
        "uavs_off_beams": [np.asarray(var.value, dtype=complex) for var in relaxed_bundle["var_uavs_off_beam"]],
        "bs_2_uav_freqs_norm": np.asarray(relaxed_bundle["var_bs_2_uav_freqs_norm"].value, dtype=float),
        "auxiliary_variable_z": np.asarray(relaxed_bundle["var_auxiliary_variable_z"].value, dtype=float),
        "cus_off_duration": np.asarray(relaxed_bundle["var_cus_off_duration"].value, dtype=float),
        "t": np.asarray(relaxed_bundle["var_t"].value, dtype=float),
        "status": str(relaxed_bundle["problem"].status),
        "surrogate_obj_value": float(np.real(relaxed_bundle["problem"].value)),
    }


def _extract_relaxed_solution_from_fusion_result(fusion_result):
    return {
        "uavs_sen_beams": [np.asarray(beam, dtype=complex) for beam in fusion_result["uavs_sen_beams"]],
        "uavs_off_beams": [np.asarray(beam, dtype=complex) for beam in fusion_result["uavs_off_beams"]],
        "bs_2_uav_freqs_norm": np.asarray(fusion_result["bs_2_uav_freqs_norm"], dtype=float),
        "auxiliary_variable_z": np.asarray(fusion_result["auxiliary_variable_z"], dtype=float),
        "cus_off_duration": np.asarray(fusion_result["cus_off_duration"], dtype=float),
        "t": np.asarray(fusion_result["t"], dtype=float),
        "status": str(fusion_result["status"]),
        "surrogate_obj_value": float(np.real(fusion_result["objective_value"])),
    }


def _solve_recovery_candidate(
    args,
    recovery_bundle,
    sen_directions,
    off_directions,
    relaxed_solution,
    cus_entertaining_task_size,
    uavs_off_duration,
    cus_off_power,
    uavs_pos_pre,
    uavs_pos_cur,
):
    for i in range(args.uavs_num):
        recovery_bundle["sen_direction_params"][i].value = sen_directions[i]
        recovery_bundle["off_direction_params"][i].value = off_directions[i]

    recovery_bundle["sen_power_vars"].value = np.asarray(
        [np.real(np.trace(beam)) for beam in relaxed_solution["uavs_sen_beams"]],
        dtype=float,
    )
    recovery_bundle["off_power_vars"].value = np.asarray(
        [np.real(np.trace(beam)) for beam in relaxed_solution["uavs_off_beams"]],
        dtype=float,
    )
    recovery_bundle["var_bs_2_uav_freqs_norm"].value = relaxed_solution["bs_2_uav_freqs_norm"]
    recovery_bundle["var_auxiliary_variable_z"].value = relaxed_solution["auxiliary_variable_z"]
    recovery_bundle["var_cus_off_duration"].value = relaxed_solution["cus_off_duration"]
    recovery_bundle["var_t"].value = relaxed_solution["t"]

    try:
        recovery_bundle["last_exception"] = None
        recovery_bundle["last_solve_backend"] = _solve_cvxpy_problem(
            problem=recovery_bundle["problem"],
            warm_start=False,
            verbose=False,
            allow_fallback_solver=True,
        )
    except Exception as exc:
        recovery_bundle["last_exception"] = f"{type(exc).__name__}: {exc}"
        return None

    if not _is_success_status(recovery_bundle["problem"].status):
        return None

    sen_beams = [
        float(recovery_bundle["sen_power_vars"].value[i]) * sen_directions[i]
        for i in range(args.uavs_num)
    ]
    off_beams = [
        float(recovery_bundle["off_power_vars"].value[i]) * off_directions[i]
        for i in range(args.uavs_num)
    ]
    pure_energy = compute_pure_energy_value(
        args=args,
        cur_uavs_sen_beams=sen_beams,
        cur_uavs_off_beams=off_beams,
        cur_bs_2_uav_freqs_norm=recovery_bundle["var_bs_2_uav_freqs_norm"].value,
        cur_auxiliary_variable_z=recovery_bundle["var_auxiliary_variable_z"].value,
        cur_cus_off_duration=recovery_bundle["var_cus_off_duration"].value,
        cus_entertaining_task_size=cus_entertaining_task_size,
        uavs_off_duration=uavs_off_duration,
        cus_off_power=cus_off_power,
        uavs_pos_pre=uavs_pos_pre,
        uavs_pos_cur=uavs_pos_cur,
    )
    rank1_gap_max, rank1_gap_sum = _compute_rank1_metrics(sen_beams, off_beams)
    return {
        "surrogate_obj_value": float(np.real(recovery_bundle["problem"].value)),
        "pure_energy": float(np.real(pure_energy)),
        "rank1_gap_max": rank1_gap_max,
        "rank1_gap_sum": rank1_gap_sum,
        "uavs_sen_beams": sen_beams,
        "uavs_off_beams": off_beams,
        "bs_2_uav_freqs_norm": np.asarray(recovery_bundle["var_bs_2_uav_freqs_norm"].value, dtype=float),
        "auxiliary_variable_z": np.asarray(recovery_bundle["var_auxiliary_variable_z"].value, dtype=float),
        "cus_off_duration": np.asarray(recovery_bundle["var_cus_off_duration"].value, dtype=float),
        "t": np.asarray(recovery_bundle["var_t"].value, dtype=float),
        "status": str(recovery_bundle["problem"].status),
    }


def gaussian_randomization_rank1_recovery(
    args,
    recovery_bundle,
    relaxed_solution,
    cus_entertaining_task_size,
    uavs_off_duration,
    cus_off_power,
    uavs_pos_pre,
    uavs_pos_cur,
    gaussian_randomization_trials=50,
    rng=None,
):
    if rng is None:
        rng = np.random.default_rng()

    projected_sen_beams = []
    projected_off_beams = []
    sen_factors = []
    off_factors = []
    dominant_sen_directions = []
    dominant_off_directions = []

    for sen_beam in relaxed_solution["uavs_sen_beams"]:
        projected_beam, beam_factor = _hermitian_psd_projection(sen_beam)
        projected_sen_beams.append(projected_beam)
        sen_factors.append(beam_factor)
        dominant_sen_directions.append(_dominant_rank1_direction(projected_beam))

    for off_beam in relaxed_solution["uavs_off_beams"]:
        projected_beam, beam_factor = _hermitian_psd_projection(off_beam)
        projected_off_beams.append(projected_beam)
        off_factors.append(beam_factor)
        dominant_off_directions.append(_dominant_rank1_direction(projected_beam))

    recovery_relaxed_solution = dict(relaxed_solution)
    recovery_relaxed_solution["uavs_sen_beams"] = projected_sen_beams
    recovery_relaxed_solution["uavs_off_beams"] = projected_off_beams

    candidate_specs = [("dominant-eigenvector", dominant_sen_directions, dominant_off_directions)]
    for trial_idx in range(int(max(gaussian_randomization_trials, 0))):
        sen_directions = [
            _sample_direction_from_factor(sen_factors[i], dominant_sen_directions[i], rng)
            for i in range(args.uavs_num)
        ]
        off_directions = [
            _sample_direction_from_factor(off_factors[i], dominant_off_directions[i], rng)
            for i in range(args.uavs_num)
        ]
        candidate_specs.append((f"gaussian-{trial_idx + 1}", sen_directions, off_directions))

    best_candidate = None
    feasible_candidate_count = 0
    status_counter = {}
    for candidate_name, sen_directions, off_directions in candidate_specs:
        candidate = _solve_recovery_candidate(
            args=args,
            recovery_bundle=recovery_bundle,
            sen_directions=sen_directions,
            off_directions=off_directions,
            relaxed_solution=recovery_relaxed_solution,
            cus_entertaining_task_size=cus_entertaining_task_size,
            uavs_off_duration=uavs_off_duration,
            cus_off_power=cus_off_power,
            uavs_pos_pre=uavs_pos_pre,
            uavs_pos_cur=uavs_pos_cur,
        )
        if candidate is None:
            last_exception = recovery_bundle.get("last_exception")
            if last_exception is not None:
                status_key = f"EXC:{last_exception}"
            else:
                status_key = str(recovery_bundle["problem"].status)
            status_counter[status_key] = status_counter.get(status_key, 0) + 1
            continue

        feasible_candidate_count += 1
        candidate["candidate_name"] = candidate_name
        if best_candidate is None:
            best_candidate = candidate
            continue

        pure_energy_gap = candidate["pure_energy"] - best_candidate["pure_energy"]
        if pure_energy_gap < -1e-8:
            best_candidate = candidate
            continue
        if abs(pure_energy_gap) <= 1e-8 and candidate["surrogate_obj_value"] < best_candidate["surrogate_obj_value"]:
            best_candidate = candidate

    if best_candidate is None:
        if status_counter:
            print(f"Gaussian recovery candidate status summary: {status_counter}")
        return None

    best_candidate["feasible_candidate_count"] = feasible_candidate_count
    best_candidate["candidate_count"] = len(candidate_specs)
    return best_candidate


def save_and_plot_gaussian_cccp_history(
    energy_val_list,
    rank1_val_list,
    output_dir=None,
    csv_prefix=None,
    method_tag="gaussian_based_cccp",
    save_plot=True,
    show_plot=False,
):
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)

    if csv_prefix is None:
        csv_prefix = _now_timestamp()

    energy_values = np.atleast_1d(np.asarray(energy_val_list, dtype=float))
    rank1_values = np.atleast_1d(np.asarray(rank1_val_list, dtype=float))

    energy_csv_path = os.path.join(output_dir, f"{csv_prefix}_{method_tag}_energy_val_list.csv")
    rank1_csv_path = os.path.join(output_dir, f"{csv_prefix}_{method_tag}_rank1_val_list.csv")
    figure_path = os.path.join(output_dir, f"{csv_prefix}_{method_tag}_history.png")

    with open(energy_csv_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(energy_values.tolist())
    with open(rank1_csv_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(rank1_values.tolist())

    if save_plot or show_plot:
        plt.rcParams["font.family"] = "Times New Roman"
        fig, ax1 = plt.subplots(figsize=(12, 8))
        ax2 = ax1.twinx()

        energy_color = "#FF6B6B"
        rank1_color = "#4169E1"
        grid_color = "#E0E0E0"

        energy_iters = np.arange(1, energy_values.size + 1)
        rank1_iters = np.arange(1, rank1_values.size + 1)

        line1 = ax1.plot(
            energy_iters,
            energy_values,
            marker="s",
            linestyle="-",
            markersize=8,
            linewidth=2,
            color=energy_color,
            markerfacecolor="white",
            markeredgewidth=1.5,
            label="Energy value",
        )
        line2 = ax2.plot(
            rank1_iters,
            rank1_values,
            marker="o",
            linestyle="--",
            markersize=8,
            linewidth=2,
            color=rank1_color,
            markerfacecolor="white",
            markeredgewidth=1.5,
            label="Rank-1 gap",
        )

        ax1.set_xlabel("The number of iteration", fontsize=24)
        ax1.set_ylabel("The minimum weighted total energy consumption (J)", fontsize=24, color=energy_color)
        ax2.set_ylabel("Rank-1 gap", fontsize=24, color=rank1_color)
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

        ax1.tick_params(axis="x", which="major", labelsize=18)
        ax1.tick_params(axis="y", which="major", labelsize=18, colors=energy_color)
        ax2.tick_params(axis="y", which="major", labelsize=18, colors=rank1_color)

        ax1.spines["left"].set_color(energy_color)
        ax1.spines["left"].set_linewidth(1.6)
        ax2.spines["right"].set_color(rank1_color)
        ax2.spines["right"].set_linewidth(1.6)
        ax2.spines["left"].set_visible(False)
        ax1.spines["top"].set_color(grid_color)
        ax1.spines["bottom"].set_color(grid_color)

        ax1.grid(
            True,
            linestyle=(0, (3, 5)),
            color=grid_color,
            linewidth=1.0,
            alpha=1.0,
            zorder=1,
        )

        lines = line1 + line2
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, fontsize=20)
        plt.tight_layout()

        if save_plot:
            fig.savefig(figure_path, dpi=300, bbox_inches="tight")
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
    else:
        figure_path = None

    return {
        "energy_csv_path": energy_csv_path,
        "rank1_csv_path": rank1_csv_path,
        "figure_path": figure_path,
    }


def gaussian_based_cccp(
    args,
    uavs_2_cus_channels,
    uavs_2_bs_channels,
    cus_2_bs_channels,
    uavs_2_targets_channels,
    uavs_targets_matched_matrix,
    uavs_cus_matched_matrix,
    uavs_pos_pre,
    uavs_pos_cur,
    uavs_off_duration,
    cus_off_power,
    gaussian_randomization_trials=50,
    gaussian_randomization_seed=None,
    cus_entertaining_task_size=None,
    fixed_total_iterations=None,
    disable_early_stop=False,
    export_history=False,
    output_dir=None,
    csv_prefix=None,
    save_plot=True,
    show_plot=False,
):
    """
    CCCP with Gaussian randomization based rank-one recovery.

    Compared with penalty_based_cccp(), this routine removes the rank-one penalty
    term and instead performs:
    1. relaxed CCCP iterations until convergence / termination;
    2. one final Gaussian randomization step on the converged relaxed SDR
       solution to recover a rank-one beamforming candidate.
    """
    if cus_entertaining_task_size is None:
        cus_entertaining_task_size = np.random.uniform(140e3, 200e3, args.cus_num)

    if gaussian_randomization_seed is None and hasattr(args, "seed"):
        gaussian_randomization_seed = int(args.seed)
    rng = np.random.default_rng(gaussian_randomization_seed)
    state = _initialize_gaussian_cccp_state(
        args=args,
        uavs_2_cus_channels=uavs_2_cus_channels,
        uavs_2_bs_channels=uavs_2_bs_channels,
        cus_2_bs_channels=cus_2_bs_channels,
        uavs_2_targets_channels=uavs_2_targets_channels,
        uavs_targets_matched_matrix=uavs_targets_matched_matrix,
        uavs_cus_matched_matrix=uavs_cus_matched_matrix,
        uavs_off_duration=uavs_off_duration,
        cus_off_power=cus_off_power,
    )

    recovery_bundle = _build_gaussian_recovery_problem_bundle(
        args=args,
        state=state,
        cus_entertaining_task_size=cus_entertaining_task_size,
        uavs_off_duration=uavs_off_duration,
        cus_off_power=cus_off_power,
        uavs_pos_pre=uavs_pos_pre,
        uavs_pos_cur=uavs_pos_cur,
    )

    use_fusion = str(args.solver_backend).lower() == "fusion"
    relaxed_bundle = None
    if not use_fusion:
        relaxed_bundle = _build_relaxed_problem_bundle(
            args=args,
            state=state,
            cus_entertaining_task_size=cus_entertaining_task_size,
            uavs_off_duration=uavs_off_duration,
            cus_off_power=cus_off_power,
            uavs_pos_pre=uavs_pos_pre,
            uavs_pos_cur=uavs_pos_cur,
        )
        print("Gaussian-based CCCP uses the CVXPY backend for relaxed subproblems.")
        print(f"Relaxed problem DPP check: {relaxed_bundle['problem'].is_dcp(dpp=True)}")
        active_groups = relaxed_bundle["active_groups"]
    else:
        print("Gaussian-based CCCP uses Fusion for relaxed subproblems and CVXPY for Gaussian recovery.")
        include_groups = parse_group_list(args.constraint_include_groups)
        exclude_groups = parse_group_list(args.constraint_exclude_groups)
        all_groups = ["4.5", "4.12", "4.23", "4.25", "4.27", "4.28", "4.29", "4.32", "4.39", "4.40", "4.44", "4.45", "auxiliary_t", "var"]
        active_groups = all_groups if include_groups is None else [g for g in all_groups if g in include_groups]
        if exclude_groups is not None:
            active_groups = [g for g in active_groups if g not in exclude_groups]

    print("--------------------------------------------------------")
    print("---------------- Gaussian-Based CCCP Start --------------")
    print("--------------------------------------------------------")

    total_iterations = int(args.max_iterations if fixed_total_iterations is None else fixed_total_iterations)
    total_iterations = max(total_iterations, 1)

    iter_count = 0
    obj_fun_opt = float("inf")
    energy_val_list = []
    rank1_val_list = []
    pre_energy_val = None
    final_relaxed_solution = None
    final_relaxed_pure_energy = None
    final_relaxed_rank1_gap_max = None
    final_relaxed_rank1_gap_sum = None

    for iter_idx in range(total_iterations):
        print(f"-------------------- Iteration {iter_idx + 1} --------------------")

        if use_fusion:
            first_iter_rank_boost_lb = cp.Parameter(
                nonneg=True,
                value=args.first_iter_rank_boost_eps if (args.enable_first_iter_rank_boost and iter_idx == 0) else 0.0,
            )
            fusion_result = solve_inner_problem_with_fusion(
                args=args,
                active_groups=active_groups,
                static_constraint_data=state["static_constraint_data"],
                hat_auxiliary_variable_z=state["hat_auxiliary_variable_z"],
                hat_bs_2_uav_freqs_norm=state["hat_bs_2_uav_freqs_norm"],
                hat_uav_sen_beams=state["hat_uav_sen_beams"],
                hat_uav_off_beams=state["hat_uav_off_beams"],
                rank1_sen_scaled_proj_mats=state["dummy_rank1_sen_scaled_proj_mats"],
                rank1_off_scaled_proj_mats=state["dummy_rank1_off_scaled_proj_mats"],
                rank1_const_terms=state["dummy_rank1_const_terms"],
                obj5_coef_z=state["obj5_coef_z"],
                obj5_coef_f=state["obj5_coef_f"],
                obj5_const_terms=state["obj5_const_terms"],
                c44_const_terms=state["c44_const_terms"],
                c44_grad_mats=state["c44_grad_mats"],
                c45_term4_const=state["c45_term4_const"],
                c45_term5_const=state["c45_term5_const"],
                c45_coef_w=state["c45_coef_w"],
                c45_coef_b=state["c45_coef_b"],
                c45_const_w=state["c45_const_w"],
                c45_const_b=state["c45_const_b"],
                cur_penalty_factor=state["zero_penalty"],
                first_iter_rank_boost_lb=first_iter_rank_boost_lb,
                cus_entertaining_task_size=cus_entertaining_task_size,
                uavs_off_duration=uavs_off_duration,
                cus_off_power=cus_off_power,
                uavs_pos_pre=uavs_pos_pre,
                uavs_pos_cur=uavs_pos_cur,
                use_penalty_rank1=False,
            )
            iter_count += 1
            print("compilation_time =", fusion_result["build_time"])
            print("solve_time =", fusion_result["solve_time"])
            print("num_iters =", fusion_result["num_iters"])
            relaxed_solution = _extract_relaxed_solution_from_fusion_result(fusion_result)
        else:
            relaxed_backend_label = _solve_cvxpy_problem(
                problem=relaxed_bundle["problem"],
                warm_start=True,
                verbose=False,
            )
            iter_count += 1
            print(f"cvxpy canon backend = {relaxed_backend_label}")
            print("compilation_time =", relaxed_bundle["problem"].compilation_time)
            print("solve_time =", relaxed_bundle["problem"].solver_stats.solve_time)
            print("num_iters =", relaxed_bundle["problem"].solver_stats.num_iters)
            relaxed_solution = _extract_relaxed_solution_from_bundle(relaxed_bundle)
            if args.enable_first_iter_rank_boost and iter_idx == 0:
                relaxed_bundle["first_iter_rank_boost_lb"].value = 0.0

        if not _is_success_status(relaxed_solution["status"]):
            print(f"Relaxed subproblem status = {relaxed_solution['status']}. Stop Gaussian-based CCCP.")
            break

        relaxed_rank1_gap_max, relaxed_rank1_gap_sum = _compute_rank1_metrics(
            relaxed_solution["uavs_sen_beams"],
            relaxed_solution["uavs_off_beams"],
        )
        relaxed_pure_energy = compute_pure_energy_value(
            args=args,
            cur_uavs_sen_beams=relaxed_solution["uavs_sen_beams"],
            cur_uavs_off_beams=relaxed_solution["uavs_off_beams"],
            cur_bs_2_uav_freqs_norm=relaxed_solution["bs_2_uav_freqs_norm"],
            cur_auxiliary_variable_z=relaxed_solution["auxiliary_variable_z"],
            cur_cus_off_duration=relaxed_solution["cus_off_duration"],
            cus_entertaining_task_size=cus_entertaining_task_size,
            uavs_off_duration=uavs_off_duration,
            cus_off_power=cus_off_power,
            uavs_pos_pre=uavs_pos_pre,
            uavs_pos_cur=uavs_pos_cur,
        )
        print(f"Relaxed surrogate objective = {relaxed_solution['surrogate_obj_value']}")
        print(f"Relaxed pure energy = {relaxed_pure_energy}")
        print(f"Relaxed rank-1 gap sum = {relaxed_rank1_gap_sum}")

        _refresh_linearization_state(
            state=state,
            args=args,
            solution=relaxed_solution,
        )

        cur_pure_energy_val = float(np.real(relaxed_pure_energy))
        cur_rank1_sum = float(relaxed_rank1_gap_sum)
        obj_fun_opt = cur_pure_energy_val
        energy_val_list.append(cur_pure_energy_val)
        rank1_val_list.append(cur_rank1_sum)
        final_relaxed_solution = {
            "uavs_sen_beams": relaxed_solution["uavs_sen_beams"],
            "uavs_off_beams": relaxed_solution["uavs_off_beams"],
            "bs_2_uav_freqs_norm": relaxed_solution["bs_2_uav_freqs_norm"],
            "auxiliary_variable_z": relaxed_solution["auxiliary_variable_z"],
            "cus_off_duration": relaxed_solution["cus_off_duration"],
            "t": relaxed_solution["t"],
        }
        final_relaxed_pure_energy = cur_pure_energy_val
        final_relaxed_rank1_gap_max = float(relaxed_rank1_gap_max)
        final_relaxed_rank1_gap_sum = cur_rank1_sum

        if (
            (not disable_early_stop)
            and pre_energy_val is not None
            and abs(pre_energy_val) > 1e-12
            and abs(cur_pure_energy_val - pre_energy_val) / abs(pre_energy_val) < args.cccp_threshold
        ):
            print(f"Converged: relative energy improvement < {args.cccp_threshold:.3e}")
            break
        pre_energy_val = cur_pure_energy_val

    if final_relaxed_solution is not None:
        print("--------------------------------------------------------")
        print("-------- Final Gaussian Randomization After CCCP --------")
        print("--------------------------------------------------------")

        near_rank1_tol = max(1e-3, 1000.0 * float(args.rank1_threshold))
        _refresh_linearization_state(
            state=state,
            args=args,
            solution=final_relaxed_solution,
        )

        recovered_solution = gaussian_randomization_rank1_recovery(
            args=args,
            recovery_bundle=recovery_bundle,
            relaxed_solution=final_relaxed_solution,
            cus_entertaining_task_size=cus_entertaining_task_size,
            uavs_off_duration=uavs_off_duration,
            cus_off_power=cus_off_power,
            uavs_pos_pre=uavs_pos_pre,
            uavs_pos_cur=uavs_pos_cur,
            gaussian_randomization_trials=gaussian_randomization_trials,
            rng=rng,
        )

        if recovered_solution is None:
            if final_relaxed_rank1_gap_sum <= near_rank1_tol:
                print(
                    "Final Gaussian randomization failed, but the converged relaxed "
                    f"solution is already near rank-one (gap sum = {final_relaxed_rank1_gap_sum:.3e}). "
                    "Use dominant-eigenvector projection as the recovery fallback."
                )
                final_solution = _dominant_projection_solution(
                    args=args,
                    relaxed_solution=final_relaxed_solution,
                    cus_entertaining_task_size=cus_entertaining_task_size,
                    uavs_off_duration=uavs_off_duration,
                    cus_off_power=cus_off_power,
                    uavs_pos_pre=uavs_pos_pre,
                    uavs_pos_cur=uavs_pos_cur,
                )
            else:
                print("Final Gaussian randomization failed to find a feasible rank-one candidate; keep the converged relaxed SDR iterate.")
                final_solution = {
                    "uavs_sen_beams": final_relaxed_solution["uavs_sen_beams"],
                    "uavs_off_beams": final_relaxed_solution["uavs_off_beams"],
                    "bs_2_uav_freqs_norm": final_relaxed_solution["bs_2_uav_freqs_norm"],
                    "auxiliary_variable_z": final_relaxed_solution["auxiliary_variable_z"],
                    "cus_off_duration": final_relaxed_solution["cus_off_duration"],
                    "t": final_relaxed_solution["t"],
                    "pure_energy": float(final_relaxed_pure_energy),
                    "rank1_gap_max": float(final_relaxed_rank1_gap_max),
                    "rank1_gap_sum": float(final_relaxed_rank1_gap_sum),
                }
        else:
            print(
                "Best final randomized candidate = "
                f"{recovered_solution['candidate_name']} "
                f"({recovered_solution['feasible_candidate_count']}/{recovered_solution['candidate_count']} feasible)"
            )
            print(f"Final recovered pure energy = {recovered_solution['pure_energy']}")
            print(f"Final recovered rank-1 gap sum = {recovered_solution['rank1_gap_sum']}")
            final_solution = recovered_solution

        obj_fun_opt = float(final_solution["pure_energy"])
        final_rank1_sum = float(final_solution["rank1_gap_sum"])
        if (
            len(energy_val_list) == 0
            or abs(obj_fun_opt - energy_val_list[-1]) > 1e-12
            or abs(final_rank1_sum - rank1_val_list[-1]) > 1e-12
        ):
            energy_val_list.append(obj_fun_opt)
            rank1_val_list.append(final_rank1_sum)
            print("Append the final Gaussian recovery result as a post-processing point in history.")

    if export_history:
        export_paths = save_and_plot_gaussian_cccp_history(
            energy_val_list=energy_val_list,
            rank1_val_list=rank1_val_list,
            output_dir=output_dir,
            csv_prefix=csv_prefix,
            method_tag="gaussian_based_cccp",
            save_plot=save_plot,
            show_plot=show_plot,
        )
        print(f"Energy history CSV saved to: {export_paths['energy_csv_path']}")
        print(f"Rank-1 history CSV saved to: {export_paths['rank1_csv_path']}")
        if export_paths["figure_path"] is not None:
            print(f"History figure saved to: {export_paths['figure_path']}")

    return obj_fun_opt, iter_count, energy_val_list, rank1_val_list
