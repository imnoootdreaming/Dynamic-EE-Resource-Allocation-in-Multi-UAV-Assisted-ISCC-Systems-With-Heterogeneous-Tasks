"""随机内外层变量采样 + 代回**原始问题 P1** 逐约束校验的可行外层变量采样器。

采样方式（与旧版“只随机外层、内层靠 PC3P 求解判定”不同）：
1. 随机生成**外层变量**（g_i、η_{i,j}、p_j、D_{u_i}^{off}、q 位移，由
   `environment.build_outer_variables` 给出）与**内层变量**（P1 的决策变量
   w_i、b_i、f_{u_i}、D_{c_j}^{off}、f_{c_j}，见 `_sample_p1_variables`）；
2. 把内层变量**代回论文原始问题 P1**（eq. 759–784），逐条校验其 9 条约束，
   记录每条约束的违反量；
3. **只保存**使全部 9 条约束都通过的样本所对应的外层变量（便于后续传给 CCCP 求解）；
   若一组都没通过，则不写 CSV。

注意：校验对象是原始问题 P1，而非其等价变换 P5。P5 引入了辅助变量 z_i、上镜图变量 τ_i、
CCCP 线性化（c11/c12）与秩一罚项，这些都属于“为求解而做的变换”，不能用来判定原始问题
的可行性；本模块改为直接按 P1 的非线性原式（含真实 log 项）校验。

判定语义为**充分条件**：随机样本全部约束通过 ⇒ 该外层变量一定可行；反之，即使外层变量
存在可行内层解，随机样本通常也无法命中（可行域极薄），会被保守地判为不可行。

场景几何与信道沿用 `default_rng(params.seed)` 固定生成，保证 CSV 中保存的外层变量可复现；
若需要场景也逐样本随机，把 `build_environment` 的 rng 改为逐样本新建即可。

外层变量元组（与 environment.build_outer_variables 返回值一致）：
    (g_rec_beam (I, N) 复, eta_share (I, J), p_cu_power (J),
     D_uav_off (I), q_uav_pos_step (I, 3))
"""

import csv
from dataclasses import dataclass

import numpy as np

from environment import build_environment, build_outer_variables


VIOLATION_TOL = 1e-6      # 约束违反量容差：<= 该值视为满足

P1_CONSTRAINT_NAMES = (
    "P1:UAV-Sen-Max-Power",
    "P1:Sensing-SINR",
    "P1:UAV-Off-Max-Power",
    "P1:UAVs-task-successful-offloading",
    "P1:CUs-task-successful-offloading",
    "P1:Task-Fresh",
    "P1:UAV-task-delay",
    # "P1:CU-task-delay" 不在此列：f_{c_j} 由闭式解给出（见 _sample_p1_variables），
    # 代回 P1 后该约束恒取等号成立，故不作校验。
    "P1:BS-Max-Frequency",
)


@dataclass
class P1Sample:
    """原始问题 P1 的一组内层变量（论文 eq. 782–783 的 var.）。"""
    w: np.ndarray         # (I, N) 复，UAV 感知波束
    b: np.ndarray         # (I, N) 复，UAV 卸载波束
    f_uav: np.ndarray     # (I,)    BS 分配给感知任务的计算资源 (Hz)
    D_cu_off: np.ndarray  # (J,)    CU 娱乐任务卸载时长 (s)
    f_cu: np.ndarray      # (J,)    BS 分配给 CU 娱乐任务的计算资源 (Hz)


def _principal_direction(matrix):
    """Hermitian 矩阵最大特征值对应的单位特征向量（感知信道 G_i 的主方向）。"""
    hermitian = (matrix + matrix.conj().T) / 2.0
    _, eigenvectors = np.linalg.eigh(hermitian)
    return eigenvectors[:, -1]


def _sample_p1_variables(rng, params, environment):
    """随机生成 P1 的内层变量。

    采样分布：
        w_i       — 方向取感知信道 G_i 的主特征方向（使 |g_i^H A_i w_i|^2 最大），
                    功率 ~ U(0.5 P^max_UAV, P^max_UAV]
        b_i       — 方向为归一化复高斯，功率 ~ U(0, P^max_UAV]
        f_{u_i}   — U(0, F_max / (I + J))
        D_{c_j}^off — 逐元素 U(0, D^max_j)
        f_{c_j}   — **不随机采样**，由闭式解求出（见下）

    f_uav 的采样上界取 F_max/(I+J)，使所有频率之和的均值约为 F_max/2，从而
    P1:BS-Max-Frequency 有被满足的可能（若取 [0, F_max]，该约束会因采样范围而恒被违反，
    无法用于“区分不可行与采样器不合理”）。

    f_{c_j} 用论文 Lemma (CU-Frequency) 的闭式解（eq. \eqref{temp:CU-Frequency}）：
        f_{c_j} = C_j L_j / (D^max_j - D^off_{c_j})
    把它代回 P1 后，约束 P1:CU-task-delay 恒取等号成立（故校验时忽略该约束）。
    """
    def random_beam():
        direction = (rng.standard_normal((params.I, params.N))
                     + 1j * rng.standard_normal((params.I, params.N))) / np.sqrt(2.0)
        direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
        power = rng.uniform(0.0, params.P_max_uav, (params.I, 1))
        return direction * np.sqrt(power)

    sensing_beam = np.zeros((params.I, params.N), dtype=complex)
    for i in range(params.I):
        sensing_beam[i] = (_principal_direction(environment.G_sen_corr[i])
                           * np.sqrt(rng.uniform(0.5 * params.P_max_uav,
                                                 params.P_max_uav)))

    freq_hi = params.F_max / (params.I + params.J)
    D_cu_off = rng.uniform(0.0, params.D_max_cu)
    f_cu = params.C_cu_cycles * params.L_cu_task / (params.D_max_cu - D_cu_off)
    return P1Sample(
        w=sensing_beam,
        b=random_beam(),
        f_uav=rng.uniform(0.0, freq_hi, params.I),
        D_cu_off=D_cu_off,
        f_cu=f_cu,
    )


def _p1_constraint_violations(params, environment, sample):
    """把 P1 内层变量代回原始问题 P1，逐条计算约束违反量。

    对应论文 eq. 766–781 的约束，形式与原式一致（含真实 log 项）；其中
    P1:CU-task-delay 已由 f_{c_j} 的闭式解消除（代回后恒取等号），故不校验：

        (1) ||w_i||^2 - P^max_UAV <= 0
        (2) eps - |g_i^H A_i w_i|^2 / Gamma_i <= 0
        (3) ||b_i||^2 - P^max_UAV <= 0
        (4) xi_1 log2(1 + xi_2 |g_i^H A_i w_i|^2 / Gamma_i)
            - D_{u_i}^off B log2(1 + |h_{u_i,BS}^H b_i|^2 / Phi_i) <= 0
        (5) L_j - sum_i eta_ij D_bar B log2(1 + p_j |h_cj,BS|^2
                        / (sum_i eta_ij |h_{u_i,BS}^H w_i|^2 + sigma^2))
            - sum_i eta_ij D_{u_i}^off B log2(1 + p_j |h_cj,BS|^2
                        / (sum_i eta_ij |h_{u_i,BS}^H b_i|^2 + sigma^2))
            - (D_cj^off - Theta_j) B upsilon_j <= 0
        (6) D_bar + D_{u_i}^off - sum_j eta_ij D_cj^off <= 0
        (7) (D_bar + D_{u_i}^off - D^sen_max) f_{u_i}
            + C^sen xi_1 log2(1 + xi_2 |g_i^H A_i w_i|^2 / Gamma_i) <= 0
        (8) sum_i f_{u_i} + sum_j f_cj - F_max <= 0   （f_cj 取闭式解）

    :return: {约束标签: 违反量}，违反量 = max(0, 该约束所有实例残差的最大值)。
    """
    I, J = params.I, params.J
    sigma_2 = params.sigma_2
    D_sen = params.D_bar_sen

    eta = environment.eta_share                       # (I, J)
    p_cu = environment.p_cu_power                     # (J,)
    D_off_uav = environment.D_uav_off                 # (I,)
    h_bs = environment.h_uav_2_bs                     # (I, N)
    h_cj_bs_sq = np.abs(environment.h_cu_2_bs) ** 2   # (J,)

    # 期望功率项：|g_i^H A_i w_i|^2 = w_i^H G_i w_i，|h_{u_i,BS}^H v_i|^2
    sen_gain = np.array([np.real(np.vdot(sample.w[i], environment.G_sen_corr[i] @ sample.w[i]))
                         for i in range(I)])
    w_bs_gain = np.array([np.abs(np.vdot(h_bs[i], sample.w[i])) ** 2 for i in range(I)])
    b_bs_gain = np.array([np.abs(np.vdot(h_bs[i], sample.b[i])) ** 2 for i in range(I)])

    # 感知速率 ξ_1 log2(1 + ξ_2 sen/Γ) 与卸载速率 B log2(1 + off/Φ)
    sen_rate = params.xi_1 * np.log2(1.0 + params.xi_2 * sen_gain / environment.Gamma_sinr)
    off_rate = params.B * np.log2(1.0 + b_bs_gain / environment.Phi_off_inr)

    # (1) (3) 功率上限
    v_sen_power = np.max(np.sum(np.abs(sample.w) ** 2, axis=1) - params.P_max_uav)
    v_off_power = np.max(np.sum(np.abs(sample.b) ** 2, axis=1) - params.P_max_uav)

    # (2) 感知 SINR 门限
    v_sinr = np.max(params.eps_sinr - sen_gain / environment.Gamma_sinr)

    # (4) 感知任务成功卸载
    v_uav_off = np.max(sen_rate - D_off_uav * off_rate)

    # (5) CU 娱乐任务成功卸载
    share = eta.sum(axis=0)                            # (J,) 共享 CU j 的 UAV 数
    interf_sen = eta.T @ w_bs_gain                     # (J,) Σ_i η_ij |h^H w_i|^2
    interf_off = eta.T @ b_bs_gain                     # (J,) Σ_i η_ij |h^H b_i|^2
    rate1 = params.B * np.log2(1.0 + p_cu * h_cj_bs_sq / (interf_sen + sigma_2))
    rate2 = params.B * np.log2(1.0 + p_cu * h_cj_bs_sq / (interf_off + sigma_2))
    rate3 = params.B * environment.upsilon_cu_rate     # (J,) = B log2(1 + p|h|^2/sigma^2)
    delivered = (share * D_sen * rate1
                 + (eta.T @ D_off_uav) * rate2
                 + (sample.D_cu_off - environment.Theta_cu_time) * rate3)
    v_cu_off = np.max(params.L_cu_task - delivered)

    # (6) 感知任务新鲜度
    v_fresh = np.max(D_sen + D_off_uav - eta @ sample.D_cu_off)

    # (7) 感知任务时延
    v_uav_delay = np.max((D_sen + D_off_uav - params.D_max_sen) * sample.f_uav
                         + params.C_sen * sen_rate)

    # (8) BS 计算能力上限（sample.f_cu 即由闭式解给出的值）
    v_bs_freq = float(np.sum(sample.f_uav) + np.sum(sample.f_cu) - params.F_max)

    raw = {
        "P1:UAV-Sen-Max-Power": v_sen_power,
        "P1:Sensing-SINR": v_sinr,
        "P1:UAV-Off-Max-Power": v_off_power,
        "P1:UAVs-task-successful-offloading": v_uav_off,
        "P1:CUs-task-successful-offloading": v_cu_off,
        "P1:Task-Fresh": v_fresh,
        "P1:UAV-task-delay": v_uav_delay,
        "P1:BS-Max-Frequency": v_bs_freq,
    }
    return {name: max(float(value), 0.0) for name, value in raw.items()}


def _build_report(n_samples, violation_stats, n_feasible):
    """汇总逐约束违反统计。"""
    rows = []
    for name, values in violation_stats.items():
        arr = np.asarray(values, dtype=float)
        rows.append({
            "name": name,
            "violated": int(np.sum(arr > VIOLATION_TOL)) if arr.size else 0,
            "rate": float(np.mean(arr > VIOLATION_TOL)) if arr.size else 0.0,
            "max": float(arr.max()) if arr.size else 0.0,
            "median": float(np.median(arr)) if arr.size else 0.0,
        })
    return {
        "n_samples": n_samples,
        "n_feasible": n_feasible,
        "hit_rate": (n_feasible / n_samples) if n_samples else 0.0,
        "rows": rows,
    }


def _print_report(report):
    print("\n===== 随机变量代回 P1 的可行性采样报告 =====")
    print("样本数：{}，全约束通过：{}，命中率：{:.2%}".format(
        report["n_samples"], report["n_feasible"], report["hit_rate"]))
    print("{:40s} {:>10s} {:>14s} {:>14s}".format(
        "约束", "违反样本", "最大违反量", "中位违反量"))
    for row in report["rows"]:
        print("{:40s} {:>10s} {:>14.3e} {:>14.3e}".format(
            row["name"], "{}/{}".format(row["violated"], report["n_samples"]),
            row["max"], row["median"]))
    worst = max(report["rows"], key=lambda row: row["rate"], default=None)
    if worst is not None and report["n_feasible"] == 0:
        print("提示：违反率最高的约束为 {}（{:.0%}）。".format(worst["name"], worst["rate"]))
    print("=" * 84 + "\n")


def sample_random_feasible(params, n_samples=10000, csv_path="feasible_outer_samples.csv",
                           seed=None, n_feasible=30, verbose=True):
    """随机生成（外层 + P1 内层）变量，代回 P1 逐约束校验，
    并把**通过全部约束**的外层变量写入 CSV（供后续传给 CCCP 求解）。

    停止条件（满足任一即停）：
        - 已收集到 n_feasible 组可行样本（提前停止并保存）；
        - 已抽满 n_samples 组（安全上限）。

    保存语义：只保存全约束通过的样本；若一组都没通过，则不写 CSV。

    :param params: Parameters 实例（决定种子、规模与阈值）。
    :param n_samples: 最多采样组数（安全上限）。
    :param csv_path: 可行外层变量 CSV 的保存路径。
    :param seed: 随机种子；None 时取 params.seed。
    :param n_feasible: 目标可行样本数，达到即提前停止；None 表示不提前停止。
    :param verbose: 是否打印报告。
    :return: (feasible_outer_samples, violation_stats)
        feasible_outer_samples —— 通过全部 P1 约束的外层变量元组列表
        violation_stats        —— {P1 约束标签: 每个样本该约束的最大违反量}
    """
    rng = np.random.default_rng(params.seed if seed is None else seed)
    outer_rng = np.random.default_rng(params.seed)

    violation_stats = {name: [] for name in P1_CONSTRAINT_NAMES}
    feasible_outer_samples = []
    drawn = 0

    for _ in range(n_samples):
        drawn += 1
        outer_variables = build_outer_variables(params, outer_rng)
        environment = build_environment(
            params, np.random.default_rng(params.seed), outer_variables=outer_variables)
        sample = _sample_p1_variables(rng, params, environment)
        violations = _p1_constraint_violations(params, environment, sample)

        all_satisfied = True
        for name, value in violations.items():
            violation_stats[name].append(value)
            if value > VIOLATION_TOL:
                all_satisfied = False

        if all_satisfied:
            feasible_outer_samples.append((
                environment.g_rec_beam,
                environment.eta_share,
                environment.p_cu_power,
                environment.D_uav_off,
                environment.q_uav_pos_next - environment.q_uav_pos,
            ))
            if n_feasible is not None and len(feasible_outer_samples) >= n_feasible:
                break   # 已收集到目标数量的可行样本，停止采样

    report = _build_report(drawn, violation_stats, len(feasible_outer_samples))
    if verbose:
        _print_report(report)
    if feasible_outer_samples:
        save_outer_samples(feasible_outer_samples, csv_path)
        if verbose:
            print("已保存 {} 组可行外层变量到 {}（共抽样 {} 组）".format(
                len(feasible_outer_samples), csv_path, drawn))
    elif verbose:
        print("本次采样没有全约束通过的样本，未写出 CSV。")

    return feasible_outer_samples, violation_stats


def save_outer_samples(samples, csv_path):
    """将样本列表平铺写入 CSV（复数拆实部/虚部，列顺序固定）。"""
    if not samples:
        raise ValueError("样本列表为空，无法写入 CSV。")
    I, N = samples[0][0].shape
    J = samples[0][1].shape[1]

    header = ["sample_id"]
    header += ["g_re_{}_{}".format(i, n) for i in range(I) for n in range(N)]
    header += ["g_im_{}_{}".format(i, n) for i in range(I) for n in range(N)]
    header += ["eta_{}_{}".format(i, j) for i in range(I) for j in range(J)]
    header += ["p_{}".format(j) for j in range(J)]
    header += ["D_{}".format(i) for i in range(I)]
    header += ["step_{}_{}".format(i, k) for i in range(I) for k in range(3)]

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for sid, (g, eta, p, D, step) in enumerate(samples):
            row = [sid]
            row += g.real.flatten().tolist()
            row += g.imag.flatten().tolist()
            row += eta.flatten().tolist()
            row += p.flatten().tolist()
            row += D.flatten().tolist()
            row += step.flatten().tolist()
            writer.writerow(row)


def load_outer_samples(csv_path, params):
    """从 CSV 读回样本列表，依据 params.I/J/N 重塑；列数不符或解析失败抛明确异常。"""
    I, J, N = params.I, params.J, params.N
    expected = 1 + 2 * I * N + I * J + J + I + 3 * I

    samples = []
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        if len(header) != expected:
            raise ValueError("CSV 列数不符：期望 {} 列，实际 {} 列。".format(expected, len(header)))
        for row in reader:
            if not row:
                continue
            vals = [float(x) for x in row[1:]]  # 跳过 sample_id
            if len(vals) != expected - 1:
                raise ValueError("数据行列数不符：期望 {} 个数值，实际 {} 个。".format(
                    expected - 1, len(vals)))
            idx = 0
            g_re = np.array(vals[idx:idx + I * N]).reshape(I, N); idx += I * N
            g_im = np.array(vals[idx:idx + I * N]).reshape(I, N); idx += I * N
            g = g_re + 1j * g_im
            eta = np.array(vals[idx:idx + I * J]).reshape(I, J); idx += I * J
            p = np.array(vals[idx:idx + J]); idx += J
            D = np.array(vals[idx:idx + I]); idx += I
            step = np.array(vals[idx:idx + 3 * I]).reshape(I, 3); idx += 3 * I
            samples.append((g, eta, p, D, step))
    return samples


if __name__ == "__main__":
    from parameters import Parameters

    sample_random_feasible(Parameters(), n_samples=10000, n_feasible=30)
