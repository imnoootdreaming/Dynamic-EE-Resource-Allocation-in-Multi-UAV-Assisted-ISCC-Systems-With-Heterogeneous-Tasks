"""从 compare_pc3p_gc3p.py 输出的 CSV 绘制 PC3P 与 GC3P 的分组柱状图。

对齐 inner_problem/plot_compare_penalty_gaussian_obj_from_csv.py 的配色与风格：
每个随机 case 并排画出两条真实能量柱（不含秩一罚项，口径一致）。

用法：
    cd src/inner
    python plot_compare_pc3p_gc3p.py --csv compare_pc3p_gc3p_results.csv
    python plot_compare_pc3p_gc3p.py --csv _tmp_compare_out.csv --save fig.png
"""

import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Times New Roman"

PC3P_COLOR = "#FF6B6B"
GC3P_COLOR = "#4169E1"
GRID_COLOR = "#E0E0E0"

REQUIRED_COLUMNS = {"case_id", "penalty_based_obj", "gaussian_based_obj"}


def load_compare_data(csv_path):
    """读取对比 CSV，返回 (case_ids, pc3p_energy, gc3p_energy) 三个数组。"""
    case_ids, pc3p_values, gc3p_values = [], [], []
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None or not REQUIRED_COLUMNS.issubset(reader.fieldnames):
            raise ValueError(
                "CSV 需要包含列: case_id, penalty_based_obj, gaussian_based_obj。实际列: {}".format(
                    reader.fieldnames)
            )
        for row in reader:
            case_ids.append(int(row["case_id"]))
            pc3p_values.append(float(row["penalty_based_obj"]))
            gc3p_values.append(float(row["gaussian_based_obj"]))
    return np.array(case_ids), np.array(pc3p_values), np.array(gc3p_values)


def plot(case_ids, pc3p_values, gc3p_values, save_path=None, show=True):
    x = np.arange(case_ids.size)
    bar_width = 0.36

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.bar(
        x - bar_width / 2, pc3p_values,
        width=bar_width, color=PC3P_COLOR, edgecolor="black",
        linewidth=0.6, label="Penalty-based CCCP (PC3P)", zorder=3,
    )
    ax.bar(
        x + bar_width / 2, gc3p_values,
        width=bar_width, color=GC3P_COLOR, edgecolor="black",
        linewidth=0.6, label="Gaussian-randomization-based CCCP (GC3P)", zorder=3,
    )

    ax.set_xlabel("Random cases", fontsize=24)
    ax.set_ylabel("The weighted total energy consumption (J)", fontsize=24)
    ax.set_xticks(x)
    ax.set_xticklabels(case_ids)
    ax.tick_params(axis="x", which="major", labelsize=24)
    ax.tick_params(axis="y", which="major", labelsize=24)
    ax.set_ylim(0, max(pc3p_values.max(), gc3p_values.max()) * 1.22)

    ax.grid(
        True, axis="y", linestyle=(0, (3, 5)), color=GRID_COLOR,
        linewidth=1.0, alpha=1.0, zorder=1,
    )
    ax.legend(fontsize=24)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print("图像已保存:", save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="绘制 PC3P vs GC3P 能量对比柱状图")
    parser.add_argument("--csv", type=str, default="compare_pc3p_gc3p_results.csv",
                        help="对比结果 CSV 路径")
    parser.add_argument("--save", type=str, default=None, help="图像输出路径（默认只显示）")
    parser.add_argument("--no-show", action="store_true", help="不弹出窗口，仅保存")
    return parser.parse_args()


def main():
    args = parse_args()
    csv_path = args.csv
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), csv_path)
    case_ids, pc3p_values, gc3p_values = load_compare_data(csv_path)
    plot(case_ids, pc3p_values, gc3p_values,
         save_path=args.save, show=not args.no_show)


if __name__ == "__main__":
    main()
