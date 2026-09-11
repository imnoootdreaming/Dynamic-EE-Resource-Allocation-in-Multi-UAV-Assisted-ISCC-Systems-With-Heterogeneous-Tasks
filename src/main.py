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

# 只对「CSV 读入的第一个外层样本」保存逐轮迭代历史
HISTORY_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "first_sample_cccp_history.csv")


def save_iteration_history(csv_path, obj_history, w_gap_history, b_gap_history):
    """保存第一个样本的逐轮迭代历史。

    每行对应一轮迭代（第 0 行是第 0 轮，即初始点 x^(0)，未做任何 CCCP 更新），
    只记录三项：原始目标函数值、W 的秩一间隙（Σ_i[Tr(W_i)-‖W_i‖₂]）、
    B 的秩一间隙（Σ_i[Tr(B_i)-‖B_i‖₂]）。
    """
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "objective_value", "w_rank1_gap", "b_rank1_gap"])
        for n, (obj, w_gap, b_gap) in enumerate(zip(obj_history, w_gap_history,
                                                    b_gap_history)):
            writer.writerow([n, repr(float(obj)), repr(float(w_gap)), repr(float(b_gap))])


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

    total_start = time.perf_counter()
    for index, outer_variables in enumerate(outer_samples, start=1):
        environment = build_environment(
            params, np.random.default_rng(params.seed), outer_variables=outer_variables)
        variables = InnerVariables.create(params)
        ctx = InnerContext(params, environment, variables)

        start = time.perf_counter()
        result = solve(ctx)
        elapsed = time.perf_counter() - start

        print("\n样本 {}/{}：求解状态：{}，求解耗时：{:.3f} s".format(
            index, len(outer_samples), result["status"], elapsed))

        # 只对「CSV 读入的第一个外层样本」保存逐轮迭代历史（含第 0 轮）
        if index == 1:
            save_iteration_history(HISTORY_CSV, result["obj_history"],
                                   result["w_gap_history"], result["b_gap_history"])
            print("逐轮迭代历史已保存到：{}".format(HISTORY_CSV))

        if result["W_sen_beam"] is None:
            print("问题 P5 在无可行解（或求解器未返回最优解）时终止，未得到可行解。")
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


if __name__ == "__main__":
    main()
