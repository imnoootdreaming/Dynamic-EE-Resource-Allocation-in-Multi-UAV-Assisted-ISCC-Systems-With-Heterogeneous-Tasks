"""内层问题（P5）求解入口：装配仿真环境 -> PC3P 迭代求解 -> 打印结果。

外层变量（g_i、η_{i,j}、p_j、D_{u_i}^{off}、q 位移）优先从 CSV 读取；CSV 不存在时
先用 outer_sampler.sample_random_feasible 采样（只保留代回 P1 后全约束通过的样本）
并写回 CSV，再逐组代入环境送入 PC3P 求解。

求解后端默认使用 MOSEK Fusion（fusion_pc3p.run_pc3p_fusion），
可用环境变量 PC3P_BACKEND=cvxpy 切换回 cvxpy 后端（pc3p.run_pc3p）。
"""

import csv
import os
import time

import numpy as np

from parameters import Parameters
from environment import build_environment
from variables import InnerVariables, InnerContext
from constraints import constraint_names
from outer_sampler import load_outer_samples, sample_random_feasible


# 默认使用 MOSEK Fusion 后端；如需切回 cvxpy，设置环境变量 PC3P_BACKEND=cvxpy
DEFAULT_BACKEND = "fusion"

OUTER_SAMPLE_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "feasible_outer_samples.csv")
N_OUTER_SAMPLES = 1000000


def select_solver():
    """选择求解后端，返回 (后端名, 求解函数)。

    两个后端签名一致（均为 run(ctx) -> result，返回同结构 dict），可无缝替换：
      - "fusion"：MOSEK Fusion 原生锥域后端（fusion_pc3p.run_pc3p_fusion，默认）
      - "cvxpy" ：cvxpy + MOSEK 前端（pc3p.run_pc3p）
    """
    backend = os.environ.get("PC3P_BACKEND", DEFAULT_BACKEND).strip().lower()
    if backend == "fusion":
        from fusion_pc3p import run_pc3p_fusion
        return "fusion", run_pc3p_fusion
    if backend == "cvxpy":
        from pc3p import run_pc3p
        return "cvxpy", run_pc3p
    raise ValueError("未知后端 PC3P_BACKEND={!r}（可选 fusion / cvxpy）".format(backend))


# 求解状态语义分类。两个后端的状态字符串不同（fusion: ProblemStatus.*；cvxpy: optimal/
# infeasible/solver_error 等），统一归为四类，避免把「数值失败」误报成「无可行解」：
#   optimal    —— 已返回最优/可行解（fusion: PrimalAndDualFeasible；cvxpy: optimal[_inaccurate]）
#   infeasible —— 模型确实不可行（fusion: *Infeasible；cvxpy: infeasible[_inaccurate]/unbounded）
#   unknown    —— 求解器未能判定，属数值失败（fusion: Unknown/IllPosed；cvxpy: solver_error）
#   error      —— 求解器内部异常（fusion 后端在 solve() 抛异常时写成 "error: ..."）
STATUS_LABELS = {
    "optimal": "已返回最优/可行解",
    "infeasible": "模型判定为不可行",
    "unknown": "求解器未能判定（数值失败，并非真的不可行）",
    "error": "求解器内部异常",
    "other": "其他状态",
}


def classify_status(status):
    """把后端状态字符串归为 (类别, 中文说明)，类别 ∈ {optimal, infeasible, unknown, error, other}。"""
    if status is None:
        return "other", STATUS_LABELS["other"]
    low = str(status).lower()
    if low.startswith("error"):
        return "error", STATUS_LABELS["error"]
    # 注意 "Infeasible" 中含有子串 "feasible"，必须先判不可行再判可行
    if "infeasible" in low or low.startswith("unbounded"):
        return "infeasible", STATUS_LABELS["infeasible"]
    if "unknown" in low or "illposed" in low or "ill_posed" in low or "solver_error" in low:
        return "unknown", STATUS_LABELS["unknown"]
    if "optimal" in low or "feasible" in low:
        return "optimal", STATUS_LABELS["optimal"]
    return "other", STATUS_LABELS["other"]


def explain_no_solution(status_kind, status):
    """对「未拿到最优解」的情况给出准确原因：不可行 / 数值失败 / 求解器异常，三者不可混为一谈。"""
    if status_kind == "infeasible":
        return "问题 P5 判定为不可行：该外层样本在内层模型下确实无可行解（{}）。".format(status)
    if status_kind == "unknown":
        return ("求解器未能判定可行 / 不可行（{}），属数值失败而非真的无可行解；"
                "可收紧 mosek_tol_feas 或重试该样本。".format(status))
    if status_kind == "error":
        return "求解器内部异常，未得到可行解：{}".format(status)
    return "求解器未返回最优解，未得到可行解：{}".format(status)


def iterations_of_result(result):
    """取求解结果的迭代次数，统一转成 int（缺失或非数值时返回 0）。"""
    try:
        return int(result.get("iterations"))
    except (TypeError, ValueError):
        return 0


# 保存「迭代次数落在 HISTORY_ITERATION_MIN ~ HISTORY_ITERATION_MAX 区间内、最早出现的那次求解」
# 的逐轮迭代历史，对应其能量项 / 秩一罚项的收敛变化曲线。
# 若没有任何样本落在该区间，则退化为保存迭代次数最多的那一次，并在控制台给出提示。
# 文件名沿用 first_sample_cccp_history.csv（Fig1 绘图脚本按此文件名读取内容，不改名）。
HISTORY_ITERATION_MIN = 5
HISTORY_ITERATION_MAX = 7

HISTORY_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "first_sample_cccp_history.csv")


def save_iteration_history(csv_path, obj_history, w_gap_history, b_gap_history):
    """保存选定样本（迭代次数落在 HISTORY_ITERATION_MIN ~ MAX 区间内、最早出现的那次）的
    逐轮迭代历史。

    每行对应一轮迭代（第 0 行是第 0 轮，即初始点 x^(0)，未做任何 CCCP 更新），
    只记录三项：原始目标函数值、W（感知波束）的秩一间隙（Σ_i[Tr(W_i)-‖W_i‖₂]）、
    B（卸载波束）的秩一间隙（Σ_i[Tr(B_i)-‖B_i‖₂]）。

    关于负值：秩一间隙理论上恒非负（W ⪰ 0 时 Σλ_i - λ_max ≥ 0），但求解器返回的 W / B
    允许带 ~1e-8 量级的负特征值，因此实测值会出现 -1e-8 这类负数。写入时统一按
    max(·, 0) 截断——只是把数值噪声归零，不影响任何优化过程。
    这里刻意**不用 abs()**：负值源于数值误差，取绝对值反而会把它伪造成一个真实的"正向间隙"。
    """
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "objective_value", "w_rank1_gap", "b_rank1_gap"])
        for n, (obj, w_gap, b_gap) in enumerate(zip(obj_history, w_gap_history,
                                                    b_gap_history)):
            writer.writerow([n, repr(float(obj)),
                             repr(max(float(w_gap), 0.0)),
                             repr(max(float(b_gap), 0.0))])


# 每个外层样本（case）的收敛迭代次数统计（读取 feasible_outer_samples.csv 后逐组求解得到），
# 全部 case（30 个）都会写入该文件，绘图横轴为随机 case 序号。
CONVERGENCE_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "convergence_iterations.csv")

# 完整诊断信息（sample_id / converged / status_kind / elapsed_s）。
# 目前无需保存 details 文件，main() 中的写出调用已注释掉，此处仅保留路径与写出函数备用。
CONVERGENCE_DETAIL_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "convergence_iterations_details.csv")


def save_iteration_counts(csv_path, rows):
    """保存每组外层样本（case）收敛所需的迭代次数统计（全部 case 都会写入）。

    输出列与论文绘图脚本所需的格式对齐
    （src/fig/inner/Fig1/plot_energy_rank1_and_convergence_from_csv.py）：
        case_id, convergence_iterations

    :param rows: 每项为 [case_index, sample_id, iterations, converged,
                        status_kind, elapsed_s]
        case_index —— 控制台打印的 1 起始序号，写入为 case_id；
        sample_id  —— 与 feasible_outer_samples.csv 一致的 0 起始编号；
        iterations —— PC3P 实际迭代次数（未收敛时为 max_iterations），
                      写入为 convergence_iterations；
        converged  —— 是否满足收敛判据；
        status_kind—— 求解状态类别（optimal/infeasible/unknown/error/other）；
        elapsed_s  —— 该 case 的求解耗时（s）。
    """
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["case_id", "convergence_iterations"])
        for row in rows:
            writer.writerow([row[0], row[2]])


def save_iteration_counts_detail(csv_path, rows):
    """写出完整逐 case 求解统计（含 sample_id、是否收敛、状态类别、耗时）。

    :param rows: 每项为 [case_index, sample_id, iterations, converged,
                        status_kind, elapsed_s]，与 save_iteration_counts 入参一致。
    """
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["case_index", "sample_id", "iterations", "converged",
                         "status_kind", "elapsed_s"])
        for row in rows:
            writer.writerow(row)


def load_or_generate_outer_samples(params, csv_path=OUTER_SAMPLE_CSV,
                                   n_samples=N_OUTER_SAMPLES):
    """外层变量样本：优先从 CSV 读取；不存在则随机生成 n_samples 组并写回 CSV。"""
    if os.path.exists(csv_path):
        return load_outer_samples(csv_path, params)
    samples, _ = sample_random_feasible(params, n_samples=n_samples, csv_path=csv_path)
    return samples


def main():
    params = Parameters()
    backend_name, solve = select_solver()

    print("求解后端：{}".format(backend_name))
    print("已注册约束：{}".format(", ".join(constraint_names())))

    outer_samples = load_or_generate_outer_samples(params)
    print("外层变量样本数：{}".format(len(outer_samples)))

    status_counts = {kind: 0 for kind in STATUS_LABELS}
    iteration_rows = []          # 每个 case 的迭代次数统计（全部写入 CONVERGENCE_CSV）
    # 写 HISTORY_CSV 用的两份候选：区间内最早的一次 / 迭代次数最多的一次（兜底）
    target_history = None        # 迭代次数 ∈ [HISTORY_ITERATION_MIN, HISTORY_ITERATION_MAX] 的最早样本
    max_iterations = -1
    max_iteration_history = None

    total_start = time.perf_counter()
    for index, outer_variables in enumerate(outer_samples, start=1):
        environment = build_environment(
            params, np.random.default_rng(params.seed), outer_variables=outer_variables)
        variables = InnerVariables.create(params)
        ctx = InnerContext(params, environment, variables)

        start = time.perf_counter()
        result = solve(ctx)
        elapsed = time.perf_counter() - start

        status_kind, status_label = classify_status(result["status"])
        status_counts[status_kind] += 1
        iteration_rows.append([index, index - 1, result["iterations"],
                               result["converged"], status_kind,
                               "{:.6f}".format(elapsed)])

        # 逐轮历史（含第 0 轮）的两份候选记录
        current_iterations = iterations_of_result(result)
        candidate_history = (index, current_iterations,
                             result["obj_history"],
                             result["w_gap_history"],
                             result["b_gap_history"])
        # 目标区间 [5, 7] 内第一次出现的样本（后面的同区间样本不再覆盖）
        if (target_history is None
                and HISTORY_ITERATION_MIN <= current_iterations <= HISTORY_ITERATION_MAX):
            target_history = candidate_history
        # 迭代次数最多的那一次（并列时保留先出现的），用作区间内无样本时的兜底
        if current_iterations > max_iterations:
            max_iterations = current_iterations
            max_iteration_history = candidate_history

        print("\n样本 {}/{}：求解状态：{}（{}），求解耗时：{:.3f} s".format(
            index, len(outer_samples), result["status"], status_label, elapsed))

        if result["W_sen_beam"] is None:
            print(explain_no_solution(status_kind, result["status"]))
            continue

        print("迭代次数：{}，是否收敛：{}，罚因子 ρ：{:.4g}".format(
            result["iterations"], result["converged"], result["rho_final"]))
        print("P4 目标值 E^sum：{:.6f} J".format(result["objective_p4"]))
        print("秩一间隙：{:.3e}".format(result["rank1_gap"]))

        for i in range(params.I):
            print("u_{}: ||w||^2 = {:.4f} W, ||b||^2 = {:.4f} W, "
                  "f_u = {:.4e} Hz, z = {:.4f} bits".format(
                      i,
                      np.real(np.trace(result["W_sen_beam"][i])),
                      np.real(np.trace(result["B_off_beam"][i])),
                      result["f_uav_freq"][i],
                      result["z_aux_rate"][i]))

        for j in range(params.J):
            print("c_{}: D^off = {:.4f} s, f_c = {:.4e} Hz".format(
                j, result["D_cu_off"][j], result["f_cu_freq"][j]))

    print("\n全部 {} 组样本求解总耗时：{:.3f} s".format(
        len(outer_samples), time.perf_counter() - total_start))
    print("状态汇总（共 {} 组）：".format(len(outer_samples)))
    for kind in ("optimal", "infeasible", "unknown", "error", "other"):
        print("  {:<12s} {:>4d} 组    （{}）".format(
            kind, status_counts[kind], STATUS_LABELS[kind]))

    # 各 case（全部 30 组）的迭代次数统计；details 文件无需保存，此处直接注释掉
    save_iteration_counts(CONVERGENCE_CSV, iteration_rows)
    # save_iteration_counts_detail(CONVERGENCE_DETAIL_CSV, iteration_rows)
    print("各 case 的迭代次数统计已保存到：{}".format(CONVERGENCE_CSV))

    # 逐轮迭代历史（能量项与两条秩一罚项的变化曲线）：
    # 优先取迭代次数落在 [5, 7] 区间内最早出现的样本；区间内没有样本时退化为迭代次数最多的那一次
    history_to_save = target_history
    if history_to_save is None:
        history_to_save = max_iteration_history
        if max_iteration_history is not None:
            print("提示：没有任何样本的迭代次数落在 {}-{} 次区间内，"
                  "改用迭代次数最多的那次求解。".format(HISTORY_ITERATION_MIN,
                                                        HISTORY_ITERATION_MAX))
    if history_to_save is not None:
        case_index, iterations, obj_history, w_gap_history, b_gap_history = history_to_save
        save_iteration_history(HISTORY_CSV, obj_history, w_gap_history, b_gap_history)
        print("迭代 {} 次的求解（case_id = {}）的逐轮迭代历史已保存到：{}".format(
            iterations, case_index, HISTORY_CSV))
    # print("各 case 的完整求解统计已保存到：{}".format(CONVERGENCE_DETAIL_CSV))


if __name__ == "__main__":
    main()
