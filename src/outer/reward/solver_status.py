"""Resolve only unknown solver outcomes using fixed-action sensing constraints.

The certificate uses the actual P5 G/Gamma, independently of diagnostic logging.
C03/C05/C06 imply Tr(G W) <= lambda_max(G) P_max < eps * Gamma.
A failed certificate is inconclusive; it must never imply feasibility.
"""

import numpy as np


SENSING_CERTIFICATE_TOL = 1e-6


def classify_solver_status(status):
    """Keep the existing Fusion/CVXPY logging categories."""
    if status is None:
        return "other"
    low = str(status).lower()
    if low.startswith("error"):
        return "error"
    if "infeasible" in low or low.startswith("unbounded"):
        return "infeasible"
    if "unknown" in low or "illposed" in low or "ill_posed" in low or "solver_error" in low:
        return "unknown"
    if "optimal" in low or "feasible" in low:
        return "optimal"
    return "other"


def sensing_infeasibility_certificate(ctx, relative_tol=SENSING_CERTIFICATE_TOL):
    """Conservative C03/C05/C06 certificate for this fixed outer action only."""
    unavailable = {"evaluated": False, "certified": False,
                   "scope": "fixed_outer_action_c03_c05_c06"}
    try:
        I, N = int(ctx.I), int(ctx.N)
        G = np.asarray(ctx.G_sen_corr, dtype=complex)
        gamma = np.asarray(ctx.Gamma_sinr, dtype=float)
        power, epsilon = float(ctx.P_max_uav), float(ctx.eps_sinr)
        tol = float(relative_tol)
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        return {**unavailable, "reason": "invalid_input", "detail": type(exc).__name__}
    if I <= 0 or N <= 0 or G.shape != (I, N, N) or gamma.shape != (I,):
        return {**unavailable, "reason": "invalid_shape"}
    if not (np.all(np.isfinite(G)) and np.all(np.isfinite(gamma))
            and np.all(np.isfinite([power, epsilon, tol]))):
        return {**unavailable, "reason": "non_finite_input"}
    if power <= 0 or epsilon <= 0 or np.any(gamma <= 0) or not 0 <= tol < 1:
        return {**unavailable, "reason": "invalid_domain"}
    # ReTr(G W) uses this Hermitian part for every Hermitian W.
    hermitian = 0.5 * G + 0.5 * G.conj().transpose(0, 2, 1)
    try:
        maximum = np.maximum(np.linalg.eigvalsh(hermitian)[:, -1], 0.0)
    except np.linalg.LinAlgError:
        return {**unavailable, "reason": "eigenvalue_failure"}
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        requirement = epsilon * gamma
        upper = power * maximum
        margin = upper / requirement
    # Fusion divides by max(eps*Gamma, 1e-300). Avoid clamped/underflow domains.
    if (np.any(requirement <= 1e-300)
            or not all(np.all(np.isfinite(x)) for x in (requirement, upper, margin))):
        return {**unavailable, "reason": "invalid_numeric_scale"}
    failed = np.flatnonzero(margin < 1.0 - tol).tolist()
    return {"evaluated": True, "certified": bool(failed),
            "scope": "fixed_outer_action_c03_c05_c06",
            "relative_tolerance": tol, "failed_uavs": failed,
            "margin": margin.tolist(), "min_margin": float(margin.min()),
            "lambda_max_g": maximum.tolist(), "gamma_w": gamma.tolist(),
            "required_signal_w": requirement.tolist(), "signal_upper_bound_w": upper.tolist(),
            "max_sensing_power_w": power, "sinr_threshold_linear": epsilon}


def resolve_solver_status(status, ctx, solver_diagnostics=None):
    """Preserve optimal/other outcomes; certify only raw unknown as infeasible."""
    raw_kind = classify_solver_status(status)
    raw = solver_diagnostics or {}
    resolved = {"solver_status": str(status), "solver_status_kind": raw_kind,
                "raw_solver_status": str(status), "raw_solver_status_kind": raw_kind,
                "raw_problem_status": raw.get("problem_status", str(status)),
                "optimizer_response_code": raw.get("optimizer_response_code"),
                "status_source": "solver"}
    if raw_kind == "unknown":
        certificate = sensing_infeasibility_certificate(ctx)
        resolved["infeasibility_certificate"] = certificate
        if certificate["certified"]:
            resolved.update(solver_status="analytic_infeasible:c03_c05_c06",
                            solver_status_kind="infeasible",
                            status_source="analytic_c03_c05_c06")
    return resolved
