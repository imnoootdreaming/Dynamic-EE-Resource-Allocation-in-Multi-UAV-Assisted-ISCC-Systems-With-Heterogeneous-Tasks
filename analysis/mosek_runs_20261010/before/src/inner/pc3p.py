"""PC3P 算法（论文 Algorithm \ref{Penalty-Based-CCCP}）：罚函数 + CCCP 迭代求解内层问题 P5。

每轮迭代：
    1) 由上一轮解 x^(n-1) 更新线性化参数 Ψ_i^{(n)}、Ψ_{j,1}^{(n)}、Ψ_{j,2}^{(n)}、ν_max(·)；
    2) 用 cvxpy + MOSEK 求解凸问题 P5，得到 x^⋆；
    3) 令 x^(n) ← x^⋆，按 γ_1、γ_2 判断是否收敛；
    4) 若秩一间隙仍大于 γ_2，则把罚因子放大 rho_penalty_scale 倍（P1），继续迭代。

关于单次求解耗时：P5 的全部线性化量都做成了 cvxpy Parameter（见 cccp_params.py），
因此 problem 只编译一次，后续迭代直接复用编译缓存；MOSEK 内点法的可行性容差由
Parameters.mosek_tol_feas 控制。
"""

import numpy as np
import cvxpy as cp

from cccp_params import sync as sync_ccp_params
from constraints import collect_constraints
from objective import build_objective

LN2 = np.log(2.0)
# 初始 W^(0)、B^(0) 中掺入的满秩（各向同性）项比例，用于保证初始点严格正定。
# 注意：秩一间隙 Tr(·) - ‖·‖₂ 在初值处等于 Σ_{u_i} ε·P_max_uav·(1 - 1/N)，
# 只由「总功率 × 掺入比例」决定、**与波束方向无关**；若两个波束取同一个 ε，
# 感知 / 卸载两条罚项曲线在迭代 0 会完全重合。故这里让两者取不同比例：
# 初始间隙之比 = ε_OFF / ε_SEN = 1.5（当前参数下 0.036 : 0.054）。
RANK1_MIX_RATIO_SEN = 1e-3     # 感知波束 W^(0)：几乎严格秩一（σ 谱几乎只有主特征值）
RANK1_MIX_RATIO_OFF = 1.5e-3   # 卸载波束 B^(0)：满秩分量略多，初始间隙为感知的 1.5 倍


def largest_eigenvector(matrix):
    """返回 Hermitian 矩阵最大特征值对应的特征向量 ν_max(·)。"""
    hermitian = (matrix + matrix.conj().T) / 2.0
    _, eigenvectors = np.linalg.eigh(hermitian)
    return eigenvectors[:, -1]


def spectral_norm(matrix):
    """Hermitian 半正定矩阵的谱范数 ‖·‖_2（等于最大特征值）。"""
    hermitian = (matrix + matrix.conj().T) / 2.0
    return float(np.linalg.eigvalsh(hermitian)[-1])


def build_initial_point(ctx):
    """构造初始可行点 x^(0)。"""
    full_rank = (ctx.P_max_uav / ctx.N) * np.eye(ctx.N, dtype=complex)

    for i in range(ctx.I):
        # W_i^{(0)}：沿 G_i 主特征方向的最大比发射波束
        v_sen = largest_eigenvector(ctx.G_sen_corr[i])
        W_0 = ((1.0 - RANK1_MIX_RATIO_SEN) * ctx.P_max_uav * np.outer(v_sen, v_sen.conj())
               + RANK1_MIX_RATIO_SEN * full_rank)
        ctx.W_sen_beam_prev[i] = (W_0 + W_0.conj().T) / 2.0

        # B_i^{(0)}：指向 BS 的最大比发射波束（掺入比例与感知不同，使初始秩一间隙可区分）
        h_bs = ctx.h_uav_2_bs[i]
        v_off = h_bs.conj() / np.linalg.norm(h_bs)
        B_0 = ((1.0 - RANK1_MIX_RATIO_OFF) * ctx.P_max_uav * np.outer(v_off, v_off.conj())
               + RANK1_MIX_RATIO_OFF * full_rank)
        ctx.B_off_beam_prev[i] = (B_0 + B_0.conj().T) / 2.0

        # z_i^{(0)} = ξ_1 log_2( 1 + ξ_2 Tr(G_i W_i^{(0)}) / Γ_i )
        ctx.z_aux_rate_prev[i] = ctx.xi_1 * np.log2(
            1.0 + ctx.xi_2 * np.real(np.trace(ctx.G_sen_corr[i] @ ctx.W_sen_beam_prev[i]))
            / ctx.Gamma_sinr[i]
        )

        # f_{u_i}^{(0)}：取满足 \eqref{P1-1:UAV-task-delay} 的最小值
        ctx.f_uav_freq_prev[i] = (
            ctx.C_sen * ctx.z_aux_rate_prev[i]
            / (ctx.D_max_sen - ctx.D_bar_sen - ctx.D_uav_off[i])
        )

    for j in range(ctx.J):
        matched = ctx.eta_share[:, j] > 0.0
        if np.any(matched):
            # 满足 \eqref{P1:Task-Fresh}：D̄^sen + D_{u_i}^{off} ≤ D_{c_j}^{off}
            D_0 = float(np.max(ctx.D_bar_sen + ctx.D_uav_off[matched])) + 0.05
        else:
            D_0 = 0.5 * ctx.D_max_cu[j]
        ctx.D_cu_off_prev[j] = min(D_0, ctx.D_max_cu[j] - 1e-3)


def update_linearization_points(ctx):
    """由 x^(n) 更新全部线性化点（纯 numpy，不含任何后端相关的参数写回）。

    cvxpy 后端在 update_linearization 中随后写回 cp.Parameter；
    MOSEK Fusion 后端（fusion_pc3p.py）在写回 Fusion Parameter 前复用本函数，
    从而保证两个后端使用的是完全相同的线性化点（单一数据来源）。
    """
    for i in range(ctx.I):
        # Ψ_i^{(n)}(t) = ξ_2 Tr( G_i(t) W_i^{(n)}(t) ) + Γ_i(t)
        ctx.Psi_sen[i] = (ctx.xi_2
                          * np.real(np.trace(ctx.G_sen_corr[i] @ ctx.W_sen_beam_prev[i]))
                          + ctx.Gamma_sinr[i])

        # ν_max(W_i^{(n)}) ν_max^H(W_i^{(n)})、ν_max(B_i^{(n)}) ν_max^H(B_i^{(n)})
        v_sen = largest_eigenvector(ctx.W_sen_beam_prev[i])
        v_off = largest_eigenvector(ctx.B_off_beam_prev[i])
        ctx.nu_max_sen[i] = np.outer(v_sen, v_sen.conj())
        ctx.nu_max_off[i] = np.outer(v_off, v_off.conj())

        # ‖W_i^{(n)}‖_2、‖B_i^{(n)}‖_2
        ctx.W_sen_beam_norm_prev[i] = spectral_norm(ctx.W_sen_beam_prev[i])
        ctx.B_off_beam_norm_prev[i] = spectral_norm(ctx.B_off_beam_prev[i])

    for j in range(ctx.J):
        # Ψ_{j,1}^{(n)}(t) = Σ_i η_{i,j} Tr( H_{u_i,BS} W_i^{(n)} ) + σ²
        # Ψ_{j,2}^{(n)}(t) = Σ_i η_{i,j} Tr( H_{u_i,BS} B_i^{(n)} ) + σ²
        ctx.Psi_cu_sen[j] = ctx.sigma_2
        ctx.Psi_cu_off[j] = ctx.sigma_2
        for i in range(ctx.I):
            if ctx.eta_share[i, j] == 0.0:
                continue
            ctx.Psi_cu_sen[j] += ctx.eta_share[i, j] * np.real(
                np.trace(ctx.H_uav_bs[i] @ ctx.W_sen_beam_prev[i]))
            ctx.Psi_cu_off[j] += ctx.eta_share[i, j] * np.real(
                np.trace(ctx.H_uav_bs[i] @ ctx.B_off_beam_prev[i]))


def update_linearization(ctx):
    """由 x^(n) 更新线性化点，并写回 cvxpy Parameter（P5 只编译一次、复用缓存）。"""
    update_linearization_points(ctx)
    # 把上述线性化量写入 cvxpy 参数，供已编译的 P5 直接复用（免去每轮重新 canonicalize）
    sync_ccp_params(ctx.ccp, ctx)


def compute_rank1_gap_sen(W_sen_beam):
    """感知波束的秩一间隙：Σ_{u_i} ( Tr(W_i) - ‖W_i‖_2 )。"""
    return sum(np.real(np.trace(W_sen_beam[i])) - spectral_norm(W_sen_beam[i])
               for i in range(len(W_sen_beam)))


def compute_rank1_gap_off(B_off_beam):
    """卸载波束的秩一间隙：Σ_{u_i} ( Tr(B_i) - ‖B_i‖_2 )。"""
    return sum(np.real(np.trace(B_off_beam[i])) - spectral_norm(B_off_beam[i])
               for i in range(len(B_off_beam)))


def compute_rank1_gap(W_sen_beam, B_off_beam):
    """Σ_{u_i} ( Tr(W_i) - ‖W_i‖_2 + Tr(B_i) - ‖B_i‖_2 )。"""
    return compute_rank1_gap_sen(W_sen_beam) + compute_rank1_gap_off(B_off_beam)


def compute_pure_energy(ctx, W_sen_beam, B_off_beam, f_uav_freq, z_aux_rate, D_cu_off):
    """P4 原目标函数中**不含秩一罚项**的能量部分（单位：J）。

    即 ω_1(①+②) + ω_2·③ + ω_3·④，是 PC3P / GC3P 公平比较时使用的真实能量指标：
        - PC3P 的目标里还叠加了 ρ 倍秩一间隙，只有收敛到秩一时两者才一致；
        - GC3P 的松弛子问题 ρ ≡ 0，其迭代判据与秩一候选优选都直接使用本函数。
    """
    # ω_1 κ [ Σ_{u_i} C^sen z_i (f_{u_i})² + Σ_{c_j} ( C_j L_j )³ / ( D^max_j - D_{c_j}^{off} )² ]
    # 其中 ( z_i + f_{u_i}² )² / 2 - z_i² / 2 - (f_{u_i})⁴ / 2 = z_i f_{u_i}²，取乘积形式避免大数相消
    bs_energy = ctx.omega_1 * ctx.kappa_cpu * (
        np.sum(ctx.C_sen * z_aux_rate * f_uav_freq ** 2)
        + np.sum((ctx.C_cu_cycles * ctx.L_cu_task) ** 3 / (ctx.D_max_cu - D_cu_off) ** 2)
    )

    # ω_2 Σ_{u_i} ( D̄^sen Tr(W_i) + D_{u_i}^{off} Tr(B_i) + E_{u_i}^{fly} )
    uav_energy = ctx.omega_2 * sum(
        ctx.D_bar_sen * np.real(np.trace(W_sen_beam[i]))
        + ctx.D_uav_off[i] * np.real(np.trace(B_off_beam[i]))
        + ctx.E_uav_fly[i]
        for i in range(ctx.I)
    )

    # ω_3 Σ_{c_j} D_{c_j}^{off} p_j
    cu_energy = ctx.omega_3 * float(np.sum(D_cu_off * ctx.p_cu_power))

    return float(bs_energy + uav_energy + cu_energy)


def compute_p4_objective(ctx, W_sen_beam, B_off_beam, f_uav_freq, z_aux_rate, D_cu_off):
    """P4 原目标函数（非线性化）在当前解处的取值，用于收敛判定中的 X^(n)。

    等于纯能量 compute_pure_energy 再加上 ρ 倍的秩一间隙（PC3P 的罚项）。
    """
    # ρ Σ_{u_i} ( Tr(W_i) - ‖W_i‖_2 + Tr(B_i) - ‖B_i‖_2 )
    penalty = ctx.rho_penalty * compute_rank1_gap(W_sen_beam, B_off_beam)

    return float(compute_pure_energy(ctx, W_sen_beam, B_off_beam, f_uav_freq,
                                     z_aux_rate, D_cu_off) + penalty)


def recover_beamforming(W_sen_beam, B_off_beam):
    """由 W_i^⋆、B_i^⋆ 经特征值分解恢复 w_i^⋆、b_i^⋆。"""
    w_sen_beam = []
    b_off_beam = []
    for i in range(len(W_sen_beam)):
        v_sen = largest_eigenvector(W_sen_beam[i])
        w_sen_beam.append(np.sqrt(max(spectral_norm(W_sen_beam[i]), 0.0)) * v_sen)
        v_off = largest_eigenvector(B_off_beam[i])
        b_off_beam.append(np.sqrt(max(spectral_norm(B_off_beam[i]), 0.0)) * v_off)
    return w_sen_beam, b_off_beam


def run_pc3p(ctx):
    """执行 PC3P 迭代，返回求解结果与最优解。"""
    # ρ 复位：ctx.params 可能被 main 的循环复用，必须回到初值，否则会跨样本累积
    ctx.params.rho_penalty = ctx.rho_penalty_init
    build_initial_point(ctx)
    X_prev = compute_p4_objective(ctx, ctx.W_sen_beam_prev, ctx.B_off_beam_prev,
                                  ctx.f_uav_freq_prev, ctx.z_aux_rate_prev, ctx.D_cu_off_prev)

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
        "rho_final": ctx.rho_penalty,       # 收敛（或退出）时的罚因子
        # 逐轮迭代历史（第 0 项为第 0 轮：初始点 x^(0) 处的取值，用于画收敛曲线）
        "obj_history": [X_prev],
        "w_gap_history": [compute_rank1_gap_sen(ctx.W_sen_beam_prev)],
        "b_gap_history": [compute_rank1_gap_off(ctx.B_off_beam_prev)],
    }

    # 先按 x^(0) 计算线性化量并写入参数，随后 **只编译一次** P5；
    # 由于全部随迭代变化的量都已做成 cp.Parameter，后续每轮 solve 复用编译缓存。
    update_linearization(ctx)
    problem = cp.Problem(cp.Minimize(build_objective(ctx)), collect_constraints(ctx))

    # 内点法可行性容差：默认 1e-8，放宽到 mosek_tol_feas 可显著减少内点迭代（仅影响收敛判据）
    mosek_params = {
        "MSK_DPAR_INTPNT_CO_TOL_PFEAS": ctx.mosek_tol_feas,
        "MSK_DPAR_INTPNT_CO_TOL_DFEAS": ctx.mosek_tol_feas,
    }

    for n in range(1, ctx.max_iterations + 1):
        problem.solve(solver=cp.MOSEK, mosek_params=mosek_params)

        result["status"] = problem.status
        result["iterations"] = n
        if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            return result

        # x^(n) ← x^⋆
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

        X_n = compute_p4_objective(ctx, W_val, B_val, f_val, z_val, D_val)
        gap_n = compute_rank1_gap(W_val, B_val)
        result["objective_p4"] = X_n
        result["rank1_gap"] = gap_n
        # 记录第 n 轮的原始目标函数值、W 与 B 各自的秩一间隙
        result["obj_history"].append(X_n)
        result["w_gap_history"].append(compute_rank1_gap_sen(W_val))
        result["b_gap_history"].append(compute_rank1_gap_off(B_val))

        # 收敛条件：|X^(n) - X^(n-1)| / |X^(n-1)| ≤ γ_1 且秩一间隙 ≤ γ_2
        if abs(X_n - X_prev) <= ctx.gamma_1 * abs(X_prev) and gap_n <= ctx.gamma_2:
            result["converged"] = True
            X_prev = X_n
            break
        X_prev = X_n

        # P1：秩一间隙未达标则把罚因子放大（ρ ← scale·ρ），提升罚项权重使其在目标中可见
        if gap_n > ctx.gamma_2 and ctx.rho_penalty < ctx.rho_penalty_max:
            new_rho = min(ctx.rho_penalty * ctx.rho_penalty_scale, ctx.rho_penalty_max)
            if new_rho > ctx.rho_penalty:
                ctx.params.rho_penalty = new_rho
                result["rho_final"] = new_rho

        # 用当前解刷新下一轮所需的线性化参数（写入已编译问题绑定的 cp.Parameter）
        update_linearization(ctx)

    w_sen_beam, b_off_beam = recover_beamforming(ctx.W_sen_beam_prev, ctx.B_off_beam_prev)
    # f_{c_j}(t) = C_j(t) L_j(t) / ( D^max_j(t) - D_{c_j}^{off}(t) )
    f_cu_freq = ctx.C_cu_cycles * ctx.L_cu_task / (ctx.D_max_cu - ctx.D_cu_off_prev)

    result["W_sen_beam"] = ctx.W_sen_beam_prev
    result["B_off_beam"] = ctx.B_off_beam_prev
    result["w_sen_beam"] = w_sen_beam
    result["b_off_beam"] = b_off_beam
    result["f_uav_freq"] = ctx.f_uav_freq_prev
    result["f_cu_freq"] = f_cu_freq
    result["z_aux_rate"] = ctx.z_aux_rate_prev
    result["D_cu_off"] = ctx.D_cu_off_prev
    return result
