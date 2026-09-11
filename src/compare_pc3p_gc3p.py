"""PC3P 与 GC3P 的对比仿真（对齐 inner_problem/compare_penalty_gaussian_based_algorithm.py）。

在同一环境、同一 MOSEK 求解器下，对每个外层样本分别运行两个算法：

    PC3P（pc3p.run_pc3p）：  秩一罚项 + 自适应 ρ 的 Penalty-CCCP
    GC3P（gc3p.run_gc3p）：  ρ ≡ 0 的 SDR 松弛 CCCP + 高斯随机化秩一恢复

两者物理模型、CCCP 线性化与约束集合完全一致，唯一差别是秩一处理方式，
因此对比结果只反映「秩一处理」带来的性能差异。记录的指标：
最终真实能量（不含罚项，见 pc3p.compute_pure_energy）、耗时、迭代次数、
秩一间隙，以及把解代回原始 P1 后的真实约束违反量（outer_sampler 的校验器）。

GC3P 的候选解个数由 --candidates 控制（默认 50）：
    候选解 = 1 个主特征方向 + (candidates - 1) 个高斯随机方向，
每个候选都固定波束方向后重新优化功率，取真实能量最低的可行候选。
--candidates 50 表示「共 50 个候选可行解」；若要「50 个高斯方向 + 主特征方向」
（共 51 个候选），传 --candidates 51。

用法：
    cd src
    python compare_pc3p_gc3p.py --cases 30 --candidates 50
"""

import argparse
import csv
import time
from types import SimpleNamespace

import numpy as np

from environment import build_environment
from gc3p import GC3P_NUM_CANDIDATES, run_gc3p
from outer_sampler import _p1_constraint_violations, load_outer_samples
from parameters import Parameters
from pc3p import compute_pure_energy, run_pc3p
from variables import InnerContext, InnerVariables

CSV_FIELDS = [
    "case",
    "pc3p_energy", "gc3p_energy", "gc3p_energy_gain_pct",
    "pc3p_time", "gc3p_time", "gc3p_time_speedup",
    "pc3p_iters", "gc3p_iters",
    "pc3p_converged", "gc3p_converged",
    "pc3p_rank1_gap", "gc3p_rank1_gap",
    "gc3p_relaxed_energy", "gc3p_relaxed_rank1_gap",
    "gc3p_recovery_source", "gc3p_num_candidates", "gc3p_feasible_candidates",
    "pc3p_max_violation", "gc3p_max_violation",
    "pc3p_status", "gc3p_status",
]


def _pure_energy(ctx, result):
    """解的真实能量（不含秩一罚项），保证两算法口径一致。"""
    return compute_pure_energy(ctx, result["W_sen_beam"], result["B_off_beam"],
                               result["f_uav_freq"], result["z_aux_rate"],
                               result["D_cu_off"])


def _max_violation(params, environment, result):
    """把最终解代回原始 P1，返回全部约束违反量的最大值（0 表示严格可行）。"""
    sample = SimpleNamespace(
        w=result["w_sen_beam"],
        b=result["b_off_beam"],
        f_uav=result["f_uav_freq"],
        D_cu_off=result["D_cu_off"],
        f_cu=result["f_cu_freq"],
    )
    violations = _p1_constraint_violations(params, environment, sample)
    return float(max(violations.values())) if violations else 0.0


def _run_case(params, environment, case_index, num_candidates, seed):
    """在同一 environment 上依次跑 PC3P、GC3P，返回一行对比指标。"""
    # --- PC3P ---
    pc3p_ctx = InnerContext(params, environment, InnerVariables.create(params))
    start = time.perf_counter()
    pc3p_result = run_pc3p(pc3p_ctx)
    pc3p_time = time.perf_counter() - start

    # --- GC3P ---
    gc3p_ctx = InnerContext(params, environment, InnerVariables.create(params))
    start = time.perf_counter()
    gc3p_result = run_gc3p(gc3p_ctx, num_candidates=num_candidates,
                           rng=np.random.default_rng(seed + case_index))
    gc3p_time = time.perf_counter() - start

    pc3p_ok = pc3p_result["W_sen_beam"] is not None
    gc3p_ok = gc3p_result["W_sen_beam"] is not None

    pc3p_energy = _pure_energy(pc3p_ctx, pc3p_result) if pc3p_ok else float("nan")
    gc3p_energy = _pure_energy(gc3p_ctx, gc3p_result) if gc3p_ok else float("nan")

    # 能量改善率：正数表示 GC3P 更省能量
    if pc3p_ok and gc3p_ok and pc3p_energy > 0:
        gain_pct = (pc3p_energy - gc3p_energy) / pc3p_energy * 100.0
    else:
        gain_pct = float("nan")

    return {
        "case": case_index,
        "pc3p_energy": pc3p_energy,
        "gc3p_energy": gc3p_energy,
        "gc3p_energy_gain_pct": gain_pct,
        "pc3p_time": pc3p_time,
        "gc3p_time": gc3p_time,
        "gc3p_time_speedup": pc3p_time / gc3p_time if gc3p_time > 0 else float("nan"),
        "pc3p_iters": pc3p_result["iterations"],
        "gc3p_iters": gc3p_result["iterations"],
        "pc3p_converged": int(bool(pc3p_result["converged"])),
        "gc3p_converged": int(bool(gc3p_result["converged"])),
        "pc3p_rank1_gap": pc3p_result["rank1_gap"],
        "gc3p_rank1_gap": gc3p_result["rank1_gap"],
        "gc3p_relaxed_energy": gc3p_result["relaxed_energy"],
        "gc3p_relaxed_rank1_gap": gc3p_result["relaxed_rank1_gap"],
        "gc3p_recovery_source": gc3p_result["recovery_source"],
        "gc3p_num_candidates": gc3p_result["num_candidates"],
        "gc3p_feasible_candidates": gc3p_result["feasible_candidates"],
        "pc3p_max_violation": (_max_violation(params, environment, pc3p_result)
                               if pc3p_ok else float("nan")),
        "gc3p_max_violation": (_max_violation(params, environment, gc3p_result)
                               if gc3p_ok else float("nan")),
        "pc3p_status": pc3p_result["status"],
        "gc3p_status": gc3p_result["status"],
    }


def run_comparison(params, samples, num_candidates=GC3P_NUM_CANDIDATES, seed=None):
    if seed is None:
        seed = params.seed
    rows = []
    for case_index, outer_variables in enumerate(samples, start=1):
        # 同一 case 两个算法共用同一环境（几何/信道/权重完全一致）
        environment = build_environment(params, np.random.default_rng(params.seed),
                                        outer_variables=outer_variables)
        row = _run_case(params, environment, case_index, num_candidates, seed)
        rows.append(row)
        print("[case %2d] PC3P E=%.6f (%.2fs, %2d it) | GC3P E=%.6f (%.2fs, %2d it, "
              "gap=%.1e, cand=%d/%d, src=%s)" % (
                  case_index, row["pc3p_energy"], row["pc3p_time"], row["pc3p_iters"],
                  row["gc3p_energy"], row["gc3p_time"], row["gc3p_iters"],
                  row["gc3p_rank1_gap"], row["gc3p_feasible_candidates"],
                  row["gc3p_num_candidates"], row["gc3p_recovery_source"]))
    return rows


def _summary(rows):
    def column(name):
        return np.array([row[name] for row in rows], dtype=float)

    pc3p_energy = column("pc3p_energy")
    gc3p_energy = column("gc3p_energy")
    gain = column("gc3p_energy_gain_pct")
    pc3p_time = column("pc3p_time")
    gc3p_time = column("gc3p_time")

    total_pc3p = np.nansum(pc3p_energy)
    total_gc3p = np.nansum(gc3p_energy)

    return {
        "num_cases": len(rows),
        "pc3p_energy_mean": np.nanmean(pc3p_energy),
        "gc3p_energy_mean": np.nanmean(gc3p_energy),
        "gc3p_energy_gain_pct_mean": np.nanmean(gain),
        "gc3p_total_energy_saving_pct": ((total_pc3p - total_gc3p) / total_pc3p * 100.0
                                         if total_pc3p > 0 else float("nan")),
        "pc3p_time_mean": np.nanmean(pc3p_time),
        "gc3p_time_mean": np.nanmean(gc3p_time),
        "pc3p_time_total": np.nansum(pc3p_time),
        "gc3p_time_total": np.nansum(gc3p_time),
        "pc3p_iters_mean": np.nanmean(column("pc3p_iters")),
        "gc3p_iters_mean": np.nanmean(column("gc3p_iters")),
        "pc3p_rank1_gap_mean": np.nanmean(column("pc3p_rank1_gap")),
        "gc3p_rank1_gap_mean": np.nanmean(column("gc3p_rank1_gap")),
        "gc3p_relaxed_rank1_gap_mean": np.nanmean(column("gc3p_relaxed_rank1_gap")),
        "pc3p_max_violation_mean": np.nanmean(column("pc3p_max_violation")),
        "gc3p_max_violation_mean": np.nanmean(column("gc3p_max_violation")),
        "pc3p_converged": int(np.nansum(column("pc3p_converged"))),
        "gc3p_converged": int(np.nansum(column("gc3p_converged"))),
        "gc3p_num_candidates": int(rows[0]["gc3p_num_candidates"]) if rows else 0,
        "gc3p_feasible_candidates_mean": np.nanmean(column("gc3p_feasible_candidates")),
        "gc3p_feasible_all": int(np.nansum(
            column("gc3p_feasible_candidates") >= np.array(
                [row["gc3p_num_candidates"] for row in rows], dtype=float))),
        "gc3p_random_recovery": sum(1 for row in rows
                                    if str(row["gc3p_recovery_source"]).startswith("gaussian")),
    }


def print_summary(summary):
    print("\n" + "=" * 74)
    print("PC3P vs GC3P 对比汇总（%d 个外层样本）" % summary["num_cases"])
    print("=" * 74)
    print("平均真实能量     : PC3P %.6f J | GC3P %.6f J" % (
        summary["pc3p_energy_mean"], summary["gc3p_energy_mean"]))
    print("平均能量改善     : %+.4f%%   （总能量节省 %+.4f%%）" % (
        summary["gc3p_energy_gain_pct_mean"], summary["gc3p_total_energy_saving_pct"]))
    print("平均耗时         : PC3P %.2f s | GC3P %.2f s" % (
        summary["pc3p_time_mean"], summary["gc3p_time_mean"]))
    print("总耗时           : PC3P %.2f s | GC3P %.2f s" % (
        summary["pc3p_time_total"], summary["gc3p_time_total"]))
    print("平均迭代次数     : PC3P %.2f | GC3P %.2f" % (
        summary["pc3p_iters_mean"], summary["gc3p_iters_mean"]))
    print("收敛样本数       : PC3P %d/%d | GC3P %d/%d" % (
        summary["pc3p_converged"], summary["num_cases"],
        summary["gc3p_converged"], summary["num_cases"]))
    print("平均秩一间隙     : PC3P %.3e | GC3P %.3e (松弛 %.3e)" % (
        summary["pc3p_rank1_gap_mean"], summary["gc3p_rank1_gap_mean"],
        summary["gc3p_relaxed_rank1_gap_mean"]))
    print("平均最大违反量   : PC3P %.3e | GC3P %.3e" % (
        summary["pc3p_max_violation_mean"], summary["gc3p_max_violation_mean"]))
    print("GC3P 候选可行解  : 平均 %.1f / %d 个（%d/%d 个 case 达到全部候选可行）；"
          "由随机方向取胜 %d/%d 个 case" % (
              summary["gc3p_feasible_candidates_mean"], summary["gc3p_num_candidates"],
              summary["gc3p_feasible_all"], summary["num_cases"],
              summary["gc3p_random_recovery"], summary["num_cases"]))
    print("=" * 74)


def save_csv(rows, csv_path):
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print("逐 case 结果已写入:", csv_path)


def parse_args():
    parser = argparse.ArgumentParser(description="PC3P 与 GC3P 对比仿真")
    parser.add_argument("--cases", type=int, default=0,
                        help="使用的外层样本数（0 表示全部，默认全部 30 个）")
    parser.add_argument("--candidates", type=int, default=GC3P_NUM_CANDIDATES,
                        help="GC3P 候选解总个数（含 1 个主特征方向，默认 50）")
    parser.add_argument("--csv", type=str, default="feasible_outer_samples.csv",
                        help="外层样本 CSV 路径")
    parser.add_argument("--output", type=str, default="compare_pc3p_gc3p_results.csv",
                        help="逐 case 结果输出路径")
    parser.add_argument("--seed", type=int, default=None,
                        help="GC3P 高斯采样随机种子（默认取 params.seed）")
    return parser.parse_args()


def main():
    args = parse_args()
    params = Parameters()
    samples = load_outer_samples(args.csv, params)
    if args.cases and args.cases > 0:
        samples = samples[:args.cases]
    if not samples:
        raise SystemExit("未加载到任何外层样本: %s" % args.csv)

    print("开始对比：%d 个 case，GC3P 候选解个数 = %d" % (len(samples), args.candidates))
    rows = run_comparison(params, samples, num_candidates=args.candidates, seed=args.seed)
    save_csv(rows, args.output)
    print_summary(_summary(rows))


if __name__ == "__main__":
    main()
