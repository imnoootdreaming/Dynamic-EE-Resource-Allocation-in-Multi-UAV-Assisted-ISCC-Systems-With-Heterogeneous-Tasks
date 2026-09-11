"""内层问题（P5）求解入口：装配仿真环境 -> PC3P 迭代求解 -> 打印结果。

外层变量（g_i、η_{i,j}、p_j、D_{u_i}^{off}、q 位移）优先从 CSV 读取；CSV 不存在时
先用 outer_sampler.sample_random_feasible 采样（只保留代回 P1 后全约束通过的样本）
并写回 CSV，再逐组代入环境送入 PC3P 求解。
"""

import os
import time

import numpy as np

from parameters import Parameters
from environment import build_environment
from variables import InnerVariables, InnerContext
from constraints import constraint_names
from pc3p import run_pc3p
from outer_sampler import load_outer_samples, sample_random_feasible


OUTER_SAMPLE_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "feasible_outer_samples.csv")
N_OUTER_SAMPLES = 1000000


def load_or_generate_outer_samples(params, csv_path=OUTER_SAMPLE_CSV,
                                   n_samples=N_OUTER_SAMPLES):
    """外层变量样本：优先从 CSV 读取；不存在则随机生成 n_samples 组并写回 CSV。"""
    if os.path.exists(csv_path):
        return load_outer_samples(csv_path, params)
    samples, _ = sample_random_feasible(params, n_samples=n_samples, csv_path=csv_path)
    return samples


def main():
    params = Parameters()

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
        result = run_pc3p(ctx)
        elapsed = time.perf_counter() - start

        print("\n样本 {}/{}：求解状态：{}，求解耗时：{:.3f} s".format(
            index, len(outer_samples), result["status"], elapsed))
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
