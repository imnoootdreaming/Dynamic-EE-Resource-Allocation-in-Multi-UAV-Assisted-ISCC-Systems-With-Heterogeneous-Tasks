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

秩一恢复子问题的求解后端（GC3P 独有部分，不影响与 PC3P 共用的松弛阶段）：
    50 个候选子问题的变量 / 约束 / 目标完全相同，只有秩一方向 V_i / U_i 在变，
    因此把方向做成 cvxpy Parameter、把候选之间不变的 CCCP 线性化量冻结成常数
    （FrozenCcpParams）后，同一个 cp.Problem 只编译一次，后续候选只刷新方向参数
    即可求解：DPP 成立（每个乘积只含一个参数），省掉每候选一次的 canonicalize。
    实测恢复阶段耗时降到约 1/6~1/7，而代理目标、真实能量与可行性判定的差异在
    1e-14（目标）/ 1e-6（解向量，相对量级）以内，最终解与判定结果不变。
    置 "rebuild" 可回退到「每个候选重建问题」的旧实现（仅用于数值对照/回退）。
"""

import numpy as np
import cvxpy as cp
import mosek.fusion as mf

from cccp_params import compute_values
from constraints import collect_constraints
from fusion_pc3p import _hermitian_from, get_fusion_model, sync_fusion
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
    update_linearization_points,
)

GC3P_NUM_CANDIDATES = 50           # 候选解总个数（含 1 个主特征方向 → 49 个高斯随机方向）
GC3P_FEAS_TOL = 1e-6               # 候选解的约束违反量上限，超出即判为不可行而丢弃

# 秩一恢复子问题（GC3P 独有）的求解后端：
#   "cached"  —— 方向做成 Parameter、ccp 冻结为常数，问题只编译一次（默认，快）
#   "rebuild" —— 每个候选都新建并重新编译一次 cvxpy 问题（旧实现，仅用于数值对照）
GC3P_RECOVERY_BACKEND = "cached"

# 松弛阶段（ρ ≡ 0 的 CCCP 子问题）默认后端：与 PC3P 一致改用 MOSEK Fusion。
# 该子问题与 PC3P 的 P5 完全同构，故直接复用 fusion_pc3p 的 Fusion 模型；
# 置 "cvxpy" 可回退到原来的 cvxpy + MOSEK 前端（仅用于数值对照/回退）。
GC3P_RELAXATION_BACKEND = "fusion"

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
        try:
            violation = constraint.violation()
        except ValueError:
            # 求解失败（变量无取值）时 cvxpy 无法评估违反量；跳过即可，
            # 该候选会被调用方的 status 判定拒绝。
            continue
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


class FrozenCcpParams:
    """把 CCCP 线性化量冻结成 numpy 常数，专供秩一恢复子问题使用。

    ccp 量只在 CCCP 迭代之间变化，同一轮松弛解对应的 50 个候选共用同一套线性化值，
    对候选循环而言就是常数。冻结它们同时解决了 DPP 失效问题：
    「方向 Parameter × ccp Parameter × 变量」是参数与参数的乘积，cvxpy 会判定为非 DPP
    （每次 solve 都重新 canonicalize，编译缓存失效）；冻结 ccp 后模型中每个乘积只剩
    一个参数（方向），DPP 成立，同一个 Problem 可以编译一次反复求解。
    属性名与 CccpParams 完全一致，数值逐项相同，故不改变最优解。
    """

    def __init__(self, ctx):
        for name, value in compute_values(ctx).items():
            setattr(self, name, value)


class Rank1RecoveryContext:
    """秩一恢复子问题的上下文：W_i = p_i V_i、B_i = q_i U_i（V_i、U_i 固定为秩一方向）。

    除 W_sen_beam / B_off_beam / f_uav_freq / z_aux_rate / D_cu_off / u_freq_sq
    被替换成新的优化变量（各波束功率 p_i / q_i 等）外，其余属性（CCCP 线性化参数 ccp、
    信道、几何、权重、阈值……）全部委托给原 InnerContext，从而直接复用
    objective.build_objective 与 constraints.collect_constraints。

    frozen_ccp：可选，传入 FrozenCcpParams 时覆盖 ccp 属性（常数版线性化量）。
        秩一方向可为 numpy 常数矩阵（每个候选重建问题），也可为 cp.Parameter
        （编译一次、逐个候选刷方向，见 _recovery_cached）。
    """

    def __init__(self, ctx, sen_directions, off_directions, frozen_ccp=None):
        self._ctx = ctx
        if frozen_ccp is not None:
            # 显式赋值，__getattr__ 不再把 ccp 委托给原 InnerContext
            self.ccp = frozen_ccp
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


def _candidate_from_problem(ctx, recovery_ctx, problem, sen_directions, off_directions):
    """把已求解的恢复子问题整理成候选解 dict；不可行 / 求解失败返回 None。

    从 solve_rank1_candidate 抽出，使「每候选重建」与「编译一次复用」两条路径
    共用完全相同的状态判定、伪解过滤与解提取逻辑（保证两条路径结果一致）。
    """
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


def solve_rank1_candidate(ctx, sen_directions, off_directions, mosek_params):
    """固定秩一方向后重优化功率，返回可行候选解；不可行/求解失败返回 None。

    每个候选都新建 cp.Problem 并重新编译一次（"rebuild" 后端）；
    "cached" 后端请见 _recovery_cached。
    """
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

    return _candidate_from_problem(ctx, recovery_ctx, problem,
                                   sen_directions, off_directions)


def _select_best(candidate_specs, solve_one):
    """按候选顺序求解并挑选最优候选（主序：真实能量；次序：代理目标）。

    solve_one(sen_dirs, off_dirs) 返回候选解 dict 或 None；两条后端共用本函数，
    保证候选选取规则完全一致。
    """
    best = None
    feasible_trials = 0
    for source, sen_dirs, off_dirs in candidate_specs:
        solution = solve_one(sen_dirs, off_dirs)
        if solution is None:
            continue
        feasible_trials += 1
        solution["source"] = source
        if best is None or (solution["energy"], solution["surrogate"]) < (
                best["energy"], best["surrogate"]):
            best = solution

    if best is not None:
        best["feasible_trials"] = feasible_trials
    return best


def _recovery_rebuild(ctx, candidate_specs, mosek_params):
    """旧实现：每个候选都新建并重新编译一次 cvxpy 问题（仅用于数值对照/回退）。"""
    return _select_best(
        candidate_specs,
        lambda sen_dirs, off_dirs: solve_rank1_candidate(ctx, sen_dirs, off_dirs,
                                                         mosek_params))


def _recovery_cached(ctx, candidate_specs, mosek_params):
    """默认实现：方向做成 cp.Parameter、ccp 冻结成常数，问题只编译一次。

    候选之间只有方向参数在变，DPP 成立时 cvxpy 复用同一份编译结果，每次求解只刷新
    参数数值并让 MOSEK 重新优化，省掉每候选一次的 canonicalize 开销（实测 ~92%）。
    若问题不再是 DPP（例如后续新增了参数彼此相乘的约束），自动回退到
    _recovery_rebuild，保证结果与旧实现一致。
    """
    if not candidate_specs:
        return None

    n = ctx.N
    sen_params = [cp.Parameter((n, n), complex=True) for _ in range(ctx.I)]
    off_params = [cp.Parameter((n, n), complex=True) for _ in range(ctx.I)]
    # 先写入第 1 个候选的方向（参数必须全部有取值才能编译）
    for i in range(ctx.I):
        sen_params[i].value = candidate_specs[0][1][i]
        off_params[i].value = candidate_specs[0][2][i]

    recovery_ctx = Rank1RecoveryContext(ctx, sen_params, off_params,
                                        frozen_ccp=FrozenCcpParams(ctx))
    problem = cp.Problem(
        cp.Minimize(build_objective(recovery_ctx)),
        collect_constraints(recovery_ctx, skip_names=SKIP_RECOVERY_CONSTRAINTS),
    )
    if not problem.is_dpp():
        return _recovery_rebuild(ctx, candidate_specs, mosek_params)

    def solve_one(sen_dirs, off_dirs):
        for i in range(ctx.I):
            sen_params[i].value = sen_dirs[i]
            off_params[i].value = off_dirs[i]
        try:
            problem.solve(solver=cp.MOSEK, mosek_params=mosek_params)
        except Exception:
            # 数值异常统一视为该候选方向不可用（与 solve_rank1_candidate 一致）
            return None
        return _candidate_from_problem(ctx, recovery_ctx, problem, sen_dirs, off_dirs)

    return _select_best(candidate_specs, solve_one)


def gaussian_randomization_recovery(ctx, mosek_params, trials, rng,
                                    backend=GC3P_RECOVERY_BACKEND):
    """对松弛解做高斯随机化秩一恢复，返回真实能量最低的可行秩一候选。

    候选集合 = {主特征方向} ∪ {N_trial 个复高斯随机方向}；每个候选都固定波束方向、
    重新优化功率 f / z / D，因此不会因随机方向本身不在可行域而丢掉最优的秩一解。

    backend：候选求解后端（"cached" 默认 / "rebuild" 旧实现）。候选方向的生成与
    RNG 消耗顺序与后端无关，两种后端面对的候选集合完全相同。
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

    if backend == "cached":
        return _recovery_cached(ctx, candidate_specs, mosek_params)
    if backend == "rebuild":
        return _recovery_rebuild(ctx, candidate_specs, mosek_params)
    raise ValueError(
        "未知 recovery backend={!r}（可选 cached / rebuild）".format(backend))


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


def _relaxation_cccp_fusion(ctx, result):
    """松弛阶段 CCCP（ρ ≡ 0）的 MOSEK Fusion 实现。

    该子问题与 PC3P 的 P5 完全同构（唯一差别是秩一罚项系数 ρ 置 0），因此直接复用
    fusion_pc3p 的 Fusion 模型（同一份目标/约束与线性化量，只是求解后端换成 Fusion
    原生锥域）。每轮只刷新 Fusion Parameter 后复用同一模型，避免重建。

    就地更新 ctx 的线性化点与 result 的历史记录；返回 True 表示拿到最优解、可继续
    秩一恢复，False 表示求解失败/非最优（调用方应直接返回）。
    """
    # 按 (I, J, N) 复用进程内单模型（见 fusion_pc3p 模块 docstring）：结构恒定、
    # 每轮只刷新 Parameter，故不重建，避免原生 MOSEK 内存随时隙累积。
    model, fp, fv = get_fusion_model(ctx)
    update_linearization_points(ctx)
    sync_fusion(model, fp, ctx)
    # Fusion 的求解器参数名采用 camelCase，与 cvxpy 侧 mosek_params 的
    # MSK_DPAR_INTPNT_CO_TOL_PFEAS/DFEAS 等价，保证两后端容差一致
    model.setSolverParam("intpntCoTolPfeas", ctx.mosek_tol_feas)
    model.setSolverParam("intpntCoTolDfeas", ctx.mosek_tol_feas)

    x_prev = compute_pure_energy(ctx, ctx.W_sen_beam_prev, ctx.B_off_beam_prev,
                                 ctx.f_uav_freq_prev, ctx.z_aux_rate_prev,
                                 ctx.D_cu_off_prev)

    for n in range(1, ctx.max_iterations + 1):
        try:
            model.solve()
        except Exception as exc:                                  # noqa: BLE001
            result["status"] = "error: %s" % exc
            result["iterations"] = n
            return False

        status = model.getProblemStatus()
        result["status"] = str(status)
        result["iterations"] = n
        if status != mf.ProblemStatus.PrimalAndDualFeasible:
            return False

        W_val = np.array([_hermitian_from(fv.X[i].level(), fv.Y[i].level(), ctx.N)
                          for i in range(ctx.I)])
        B_val = np.array([_hermitian_from(fv.Xb[i].level(), fv.Yb[i].level(), ctx.N)
                          for i in range(ctx.I)])
        f_val = ctx.freq_scale * np.asarray(fv.f_norm.level(), dtype=float).reshape(ctx.I)
        z_val = fp.z_ref_now * np.asarray(fv.zn.level(), dtype=float).reshape(ctx.I)
        D_val = np.asarray(fv.D.level(), dtype=float).reshape(ctx.J)

        ctx.W_sen_beam_prev[:] = W_val
        ctx.B_off_beam_prev[:] = B_val
        ctx.f_uav_freq_prev[:] = f_val
        ctx.z_aux_rate_prev[:] = z_val
        ctx.D_cu_off_prev[:] = D_val

        x_n = compute_pure_energy(ctx, W_val, B_val, f_val, z_val, D_val)
        gap_n = compute_rank1_gap(W_val, B_val)
        result["objective_p4"] = x_n
        result["rank1_gap"] = gap_n
        result["relaxed_energy"] = x_n
        result["relaxed_rank1_gap"] = gap_n
        result["obj_history"].append(x_n)
        result["w_gap_history"].append(compute_rank1_gap_sen(W_val))
        result["b_gap_history"].append(compute_rank1_gap_off(B_val))

        # GC3P 收敛判据：只看真实能量相对改善，不要求秩一（秩一由恢复阶段保证）
        if abs(x_n - x_prev) <= ctx.gamma_1 * abs(x_prev):
            result["converged"] = True
            break
        x_prev = x_n
        update_linearization_points(ctx)
        sync_fusion(model, fp, ctx)

    return True


def _relaxation_cccp_cvxpy(ctx, result):
    """松弛阶段 CCCP（ρ ≡ 0）的 cvxpy + MOSEK 实现（保留作数值对照/回退）。

    逻辑与旧版一致：cp.Problem 只编译一次、每轮复用；返回语义与
    _relaxation_cccp_fusion 相同。
    """
    update_linearization(ctx)
    problem = cp.Problem(cp.Minimize(build_objective(ctx)), collect_constraints(ctx))
    mosek_params = solver_mosek_params(ctx)

    x_prev = compute_pure_energy(ctx, ctx.W_sen_beam_prev, ctx.B_off_beam_prev,
                                 ctx.f_uav_freq_prev, ctx.z_aux_rate_prev,
                                 ctx.D_cu_off_prev)

    for n in range(1, ctx.max_iterations + 1):
        problem.solve(solver=cp.MOSEK, mosek_params=mosek_params)

        result["status"] = problem.status
        result["iterations"] = n
        if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            return False

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

        x_n = compute_pure_energy(ctx, W_val, B_val, f_val, z_val, D_val)
        gap_n = compute_rank1_gap(W_val, B_val)
        result["objective_p4"] = x_n
        result["rank1_gap"] = gap_n
        result["relaxed_energy"] = x_n
        result["relaxed_rank1_gap"] = gap_n
        result["obj_history"].append(x_n)
        result["w_gap_history"].append(compute_rank1_gap_sen(W_val))
        result["b_gap_history"].append(compute_rank1_gap_off(B_val))

        # GC3P 收敛判据：只看真实能量相对改善，不要求秩一（秩一由恢复阶段保证）
        if abs(x_n - x_prev) <= ctx.gamma_1 * abs(x_prev):
            result["converged"] = True
            break
        x_prev = x_n
        update_linearization(ctx)

    return True


def run_gc3p(ctx, num_candidates=GC3P_NUM_CANDIDATES, rng=None,
             relaxation_backend=GC3P_RELAXATION_BACKEND,
             recovery_backend=GC3P_RECOVERY_BACKEND):
    """执行 GC3P：ρ ≡ 0 的松弛 CCCP 迭代 + 高斯随机化秩一恢复。

    参数
    ----
    num_candidates : int
        **候选解总个数**（含 1 个主特征方向），默认 50；对应高斯随机方向数为
        num_candidates - 1。
    relaxation_backend : str
        松弛阶段求解后端："fusion"（默认，MOSEK Fusion 原生锥域）或
        "cvxpy"（cvxpy + MOSEK 前端，仅用于数值对照/回退）。两个后端解同一个
        子问题、共享同一套线性化量与求解容差，结果在数值精度内一致。
        该阶段与 PC3P 完全共用，是两算法的公共部分，不受 GC3P 侧优化影响。
    recovery_backend : str
        秩一恢复后端（GC3P 独有部分）："cached"（默认）把方向做成 cvxpy Parameter、
        把 ccp 线性化量冻结为常数，50 个候选共用一份编译结果；"rebuild" 为旧实现
        （每候选重新建模+编译）。两者候选集合、代理目标、可行性判定与最终解一致
        （实测代理目标相对差 ≤1e-14，解向量相对差 ≤1e-6），仅耗时不同。

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

    if relaxation_backend == "fusion":
        solver_ok = _relaxation_cccp_fusion(ctx, result)
    elif relaxation_backend == "cvxpy":
        solver_ok = _relaxation_cccp_cvxpy(ctx, result)
    else:
        raise ValueError(
            "未知 relaxation_backend={!r}（可选 fusion / cvxpy）".format(relaxation_backend))
    if not solver_ok:
        return result

    # 让 ccp 参数与最后一个松弛迭代点对齐（恢复子问题复用同一套线性化）
    update_linearization(ctx)

    recovery = gaussian_randomization_recovery(ctx, solver_mosek_params(ctx),
                                               gaussian_trials, rng,
                                               backend=recovery_backend)
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
