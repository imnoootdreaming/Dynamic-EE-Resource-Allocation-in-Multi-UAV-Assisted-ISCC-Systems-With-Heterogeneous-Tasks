"""GC3P 算法：Gaussian-randomization-based CCCP 求解内层问题 P5。

与 PC3P（pc3p.py）相比，两者唯一的差别在于**秩一约束 W_i = w_i w_i^H、B_i = b_i b_i^H
的处理方式**，其余物理模型、CCCP 线性化方式与约束集合完全相同：

    1) 松弛阶段：CCCP 每轮求解的凸子问题 **不含** 秩一罚项（ρ ≡ 0），
       即对 W_i / B_i 只保留半正定松弛；
    2) 恢复阶段：松弛迭代收敛后，对 W_i^⋆、B_i^⋆ 做 Hermitian PSD 分解
       W_i^⋆ = F_i F_i^H，用复高斯向量 z 采样候选方向 v_i ∝ F_i z（归一化后
       V_i = v_i v_i^H，Tr(V_i) = 1）；把「主特征方向 + N_trial 个随机方向」逐个
       固定为波束方向，仅重新优化各波束功率 p_i / q_i 与 f、z、D，
       最后取**真实能量最小**的可行候选作为秩一解。

因此本文件复用 objective.build_objective 与 constraints.collect_constraints：
    - 松弛子问题直接用原 InnerContext（只把 ctx.params.rho_penalty 置 0）；
    - 恢复子问题用 Rank1RecoveryContext 把 W_i / B_i 换成 p_i V_i / q_i U_i。
两个算法因此只在「秩一处理」上不同，性能对比是公平的。

收敛判据与 PC3P 的差异：GC3P 不要求秩一间隙收敛（秩一由恢复阶段保证），
只检查真实能量的相对改善 |E^n - E^{n-1}| ≤ γ_1 |E^{n-1}|。
"""

import numpy as np
import cvxpy as cp

from constraints import collect_constraints
from objective import build_objective
from pc3p import (
    build_initial_point,
    compute_pure_energy,
    compute_rank1_gap,
    compute_rank1_gap_off,
    compute_rank1_gap_sen,
    largest_eigenvector,
    recover_beamforming,
    update_linearization,
)

GC3P_NUM_CANDIDATES = 50           # 候选解总个数（含 1 个主特征方向 → 49 个高斯随机方向）
GC3P_FEAS_TOL = 1e-6               # 候选解的约束违反量上限，超出即判为不可行而丢弃

# 秩一恢复子问题中可安全跳过的约束：W_i = p_i V_i 且 V_i ⪰ 0、p_i ≥ 0 已隐含 PSD
SKIP_RECOVERY_CONSTRAINTS = ("P2:Sen-Semi-Positive", "P2:Off-Semi-Positive")


def solver_mosek_params(ctx):
    """松弛 / 秩一恢复子问题的 MOSEK 求解参数，与 pc3p.run_pc3p 完全一致。

    两者使用同一套内点法可行性容差，保证 PC3P 与 GC3P 的性能差异只来自秩一处理方式，
    而不是求解器设置。
    """
    return {
        "MSK_DPAR_INTPNT_CO_TOL_PFEAS": ctx.mosek_tol_feas,
        "MSK_DPAR_INTPNT_CO_TOL_DFEAS": ctx.mosek_tol_feas,
    }


def worst_constraint_violation(problem):
    """返回 cvxpy 问题全部约束在求解点上的最大违反量（原始量纲，0 表示严格可行）。

    用于识别「status=optimal 但实际不可行」的伪解：高斯恢复子问题病态时，
    求解器可能返回这类解，直接采用会破坏 P1 的 UAV 卸载约束。
    """
    worst = 0.0
    for constraint in problem.constraints:
        violation = constraint.violation()
        if violation is None:
            continue
        value = float(np.max(np.abs(np.asarray(violation, dtype=float))))
        if value > worst:
            worst = value
    return worst


def psd_factorization(matrix):
    """Hermitian 矩阵的 PSD 投影 W⁺ = F Fᴴ，返回 (W⁺, F)。

    负特征值（数值截断误差）截为 0；F = U diag(√λ)，于是 W⁺ = F Fᴴ，
    且对任意 z ~ CN(0, I) 有 v = F z，其协方差 E[v vᴴ] = F Fᴴ = W⁺，
    这正是高斯随机化的采样依据。
    """
    hermitian = (matrix + matrix.conj().T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(hermitian)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    factor = eigenvectors * np.sqrt(eigenvalues)          # 第 k 列整体乘以 √λ_k
    projected = factor @ factor.conj().T
    return (projected + projected.conj().T) / 2.0, factor


def dominant_direction(matrix):
    """主特征方向对应的单位迹秩一矩阵 V = ν_max ν_maxᴴ。"""
    v = largest_eigenvector(matrix)
    return np.outer(v, v.conj())


def sample_gaussian_direction(factor, rng):
    """由 PSD 因子 F 采样复高斯方向，返回单位迹秩一矩阵 V = v vᴴ。

    v ∝ F z，z ~ CN(0, I)。当 F ≈ 0（该波束功率为 0）时退化为 e_1。
    """
    n = factor.shape[0]
    z = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    candidate = factor @ z
    norm = np.linalg.norm(candidate)
    if norm <= 1e-12:
        v = np.zeros(n, dtype=complex)
        v[0] = 1.0
    else:
        v = candidate / norm
    return np.outer(v, v.conj())


class Rank1RecoveryContext:
    """秩一恢复子问题的上下文：W_i = p_i V_i、B_i = q_i U_i（V_i、U_i 固定为秩一方向）。

    除 W_sen_beam / B_off_beam / f_uav_freq / z_aux_rate / D_cu_off / u_freq_sq
    被替换成新的优化变量（各波束功率 p_i / q_i 等）外，其余属性（CCCP 线性化参数 ccp、
    信道、几何、权重、阈值……）全部委托给原 InnerContext，从而直接复用
    objective.build_objective 与 constraints.collect_constraints。
    """

    def __init__(self, ctx, sen_directions, off_directions):
        self._ctx = ctx
        i_count, j_count = ctx.I, ctx.J

        # 波束功率（V_i、U_i 已归一化为 Tr = 1，故 Tr(W_i) = p_i、Tr(B_i) = q_i）
        self.sen_power = cp.Variable(i_count, nonneg=True)
        self.off_power = cp.Variable(i_count, nonneg=True)
        self.f_uav_freq_norm = cp.Variable(i_count, nonneg=True)
        self.z_aux_rate = cp.Variable(i_count, nonneg=True)
        self.D_cu_off = cp.Variable(j_count, nonneg=True)
        self.u_freq_sq = cp.Variable(i_count, nonneg=True)

        self.W_sen_beam = [self.sen_power[i] * sen_directions[i]
                           for i in range(i_count)]
        self.B_off_beam = [self.off_power[i] * off_directions[i]
                           for i in range(i_count)]
        self.f_uav_freq = ctx.freq_scale * self.f_uav_freq_norm

    def __getattr__(self, name):
        # 只在常规属性查找失败时触发：把其余一切委托给原 InnerContext
        return getattr(self._ctx, name)


def solve_rank1_candidate(ctx, sen_directions, off_directions, mosek_params):
    """固定秩一方向后重优化功率，返回可行候选解；不可行/求解失败返回 None。"""
    recovery_ctx = Rank1RecoveryContext(ctx, sen_directions, off_directions)
    problem = cp.Problem(
        cp.Minimize(build_objective(recovery_ctx)),
        collect_constraints(recovery_ctx, skip_names=SKIP_RECOVERY_CONSTRAINTS),
    )
    try:
        problem.solve(solver=cp.MOSEK, mosek_params=mosek_params)
    except Exception:
        # 数值异常（SolverError / Mosek 报错等）统一视为该候选方向不可用
        return None

    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        return None
    if recovery_ctx.sen_power.value is None:
        return None

    # 只接受真正可行的候选：病态病例子问题可能返回「optimal 但违反约束」的伪解
    violation = worst_constraint_violation(problem)
    if violation > GC3P_FEAS_TOL:
        return None

    sen_power = np.maximum(np.asarray(recovery_ctx.sen_power.value,
                                      dtype=float).ravel(), 0.0)
    off_power = np.maximum(np.asarray(recovery_ctx.off_power.value,
                                      dtype=float).ravel(), 0.0)
    f_uav_freq = ctx.freq_scale * np.asarray(recovery_ctx.f_uav_freq_norm.value,
                                             dtype=float).ravel()
    z_aux_rate = np.asarray(recovery_ctx.z_aux_rate.value, dtype=float).ravel()
    D_cu_off = np.asarray(recovery_ctx.D_cu_off.value, dtype=float).ravel()

    W_sen_beam = [sen_power[i] * sen_directions[i] for i in range(ctx.I)]
    B_off_beam = [off_power[i] * off_directions[i] for i in range(ctx.I)]

    return {
        "surrogate": float(problem.value),
        "worst_violation": violation,
        "energy": compute_pure_energy(ctx, W_sen_beam, B_off_beam,
                                      f_uav_freq, z_aux_rate, D_cu_off),
        "W_sen_beam": W_sen_beam,
        "B_off_beam": B_off_beam,
        "sen_power": sen_power,
        "off_power": off_power,
        "f_uav_freq": f_uav_freq,
        "z_aux_rate": z_aux_rate,
        "D_cu_off": D_cu_off,
    }


def gaussian_randomization_recovery(ctx, mosek_params, trials, rng):
    """对松弛解做高斯随机化秩一恢复，返回真实能量最低的可行秩一候选。

    候选集合 = {主特征方向} ∪ {N_trial 个复高斯随机方向}；每个候选都固定波束方向、
    重新优化功率 f / z / D，因此不会因随机方向本身不在可行域而丢掉最优的秩一解。
    """
    sen_factors, off_factors = [], []
    sen_dominant, off_dominant = [], []
    for i in range(ctx.I):
        _, f_sen = psd_factorization(ctx.W_sen_beam_prev[i])
        _, f_off = psd_factorization(ctx.B_off_beam_prev[i])
        sen_factors.append(f_sen)
        off_factors.append(f_off)
        sen_dominant.append(dominant_direction(ctx.W_sen_beam_prev[i]))
        off_dominant.append(dominant_direction(ctx.B_off_beam_prev[i]))

    candidate_specs = [("dominant-eigenvector", sen_dominant, off_dominant)]
    for trial in range(int(max(trials, 0))):
        candidate_specs.append((
            "gaussian-{}".format(trial + 1),
            [sample_gaussian_direction(sen_factors[i], rng) for i in range(ctx.I)],
            [sample_gaussian_direction(off_factors[i], rng) for i in range(ctx.I)],
        ))

    best = None
    feasible_trials = 0
    for source, sen_dirs, off_dirs in candidate_specs:
        solution = solve_rank1_candidate(ctx, sen_dirs, off_dirs, mosek_params)
        if solution is None:
            continue
        feasible_trials += 1
        solution["source"] = source
        # 主序：真实能量；次序：凸子问题代理目标（同能量时取代理更小者）
        if best is None or (solution["energy"], solution["surrogate"]) < (
                best["energy"], best["surrogate"]):
            best = solution

    if best is not None:
        best["feasible_trials"] = feasible_trials
    return best


def fallback_solution(ctx):
    """所有高斯候选均不可行时的兜底解，返回 (W, B, f, z, D, source)。

    直接对松弛解做主特征投影 Tr(W_i)·V_i（仍是秩一解），并沿用松弛解的 f、z、D。
    与秩一恢复一样只改变波束功率标量，故不会破坏 W_i、B_i 的秩一结构。
    """
    W_final = [np.real(np.trace(ctx.W_sen_beam_prev[i]))
               * dominant_direction(ctx.W_sen_beam_prev[i]) for i in range(ctx.I)]
    B_final = [np.real(np.trace(ctx.B_off_beam_prev[i]))
               * dominant_direction(ctx.B_off_beam_prev[i]) for i in range(ctx.I)]
    return (W_final, B_final, ctx.f_uav_freq_prev, ctx.z_aux_rate_prev,
            ctx.D_cu_off_prev, "dominant-projection-fallback")


def run_gc3p(ctx, num_candidates=GC3P_NUM_CANDIDATES, rng=None):
    """执行 GC3P：ρ ≡ 0 的松弛 CCCP 迭代 + 高斯随机化秩一恢复。

    参数
    ----
    num_candidates : int
        **候选解总个数**（含 1 个主特征方向），默认 50；对应高斯随机方向数为
        num_candidates - 1。

    返回 dict 与 pc3p.run_pc3p 结构一致（便于二者直接对比），并额外给出
    relaxed_energy / relaxed_rank1_gap / recovery_source / feasible_candidates。
    """
    gaussian_trials = max(int(num_candidates) - 1, 0)
    if rng is None:
        rng = np.random.default_rng(ctx.params.seed)

    # ρ 置 0：松弛子问题不含秩一罚项（PC3P 的 run_pc3p 会自行复位 ρ，互不影响）
    ctx.params.rho_penalty = 0.0

    build_initial_point(ctx)
    X_prev = compute_pure_energy(ctx, ctx.W_sen_beam_prev, ctx.B_off_beam_prev,
                                 ctx.f_uav_freq_prev, ctx.z_aux_rate_prev,
                                 ctx.D_cu_off_prev)

    result = {
        "status": None,
        "iterations": 0,
        "converged": False,
        "objective_p4": X_prev,
        "rank1_gap": None,
        "W_sen_beam": None,
        "B_off_beam": None,
        "w_sen_beam": None,
        "b_off_beam": None,
        "f_uav_freq": None,
        "f_cu_freq": None,
        "z_aux_rate": None,
        "D_cu_off": None,
        "rho_final": 0.0,
        # 逐轮迭代历史（第 0 项为初始点 x^(0) 处的取值，用于画收敛曲线）
        "obj_history": [X_prev],
        "w_gap_history": [compute_rank1_gap_sen(ctx.W_sen_beam_prev)],
        "b_gap_history": [compute_rank1_gap_off(ctx.B_off_beam_prev)],
        # GC3P 特有：松弛阶段结果 + 高斯恢复信息
        "relaxed_energy": X_prev,
        "relaxed_rank1_gap": compute_rank1_gap(ctx.W_sen_beam_prev,
                                               ctx.B_off_beam_prev),
        "recovery_source": None,
        "num_candidates": int(num_candidates),
        "gaussian_trials": gaussian_trials,
        "feasible_candidates": 0,
    }

    update_linearization(ctx)
    problem = cp.Problem(cp.Minimize(build_objective(ctx)), collect_constraints(ctx))
    mosek_params = solver_mosek_params(ctx)

    for n in range(1, ctx.max_iterations + 1):
        problem.solve(solver=cp.MOSEK, mosek_params=mosek_params)

        result["status"] = problem.status
        result["iterations"] = n
        if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            return result

        W_val = np.array([(W.value + W.value.conj().T) / 2.0 for W in ctx.W_sen_beam])
        B_val = np.array([(B.value + B.value.conj().T) / 2.0 for B in ctx.B_off_beam])
        f_val = np.asarray(ctx.f_uav_freq.value, dtype=float)
        z_val = np.asarray(ctx.z_aux_rate.value, dtype=float)
        D_val = np.asarray(ctx.D_cu_off.value, dtype=float)

        ctx.W_sen_beam_prev[:] = W_val
        ctx.B_off_beam_prev[:] = B_val
        ctx.f_uav_freq_prev[:] = f_val
        ctx.z_aux_rate_prev[:] = z_val
        ctx.D_cu_off_prev[:] = D_val

        X_n = compute_pure_energy(ctx, W_val, B_val, f_val, z_val, D_val)
        gap_n = compute_rank1_gap(W_val, B_val)
        result["objective_p4"] = X_n
        result["rank1_gap"] = gap_n
        result["relaxed_energy"] = X_n
        result["relaxed_rank1_gap"] = gap_n
        result["obj_history"].append(X_n)
        result["w_gap_history"].append(compute_rank1_gap_sen(W_val))
        result["b_gap_history"].append(compute_rank1_gap_off(B_val))

        # GC3P 收敛判据：只看真实能量相对改善，不要求秩一（秩一由恢复阶段保证）
        if abs(X_n - X_prev) <= ctx.gamma_1 * abs(X_prev):
            result["converged"] = True
            break
        X_prev = X_n
        update_linearization(ctx)

    # 让 ccp 参数与最后一个松弛迭代点对齐（恢复子问题复用同一套线性化）
    update_linearization(ctx)

    recovery = gaussian_randomization_recovery(ctx, mosek_params, gaussian_trials, rng)
    if recovery is not None:
        result["recovery_source"] = recovery["source"]
        result["feasible_candidates"] = recovery["feasible_trials"]
        W_final = recovery["W_sen_beam"]
        B_final = recovery["B_off_beam"]
        f_final = recovery["f_uav_freq"]
        z_final = recovery["z_aux_rate"]
        D_final = recovery["D_cu_off"]
    else:
        W_final, B_final, f_final, z_final, D_final, source = fallback_solution(ctx)
        result["recovery_source"] = source

    result["W_sen_beam"] = W_final
    result["B_off_beam"] = B_final
    result["w_sen_beam"], result["b_off_beam"] = recover_beamforming(W_final, B_final)
    result["f_uav_freq"] = f_final
    result["f_cu_freq"] = ctx.C_cu_cycles * ctx.L_cu_task / (ctx.D_max_cu - D_final)
    result["z_aux_rate"] = z_final
    result["D_cu_off"] = D_final
    result["objective_p4"] = compute_pure_energy(ctx, W_final, B_final,
                                                 f_final, z_final, D_final)
    result["rank1_gap"] = compute_rank1_gap(W_final, B_final)
    return result
