import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from matplotlib.ticker import MaxNLocator, MultipleLocator

plt.rcParams["font.family"] = "Times New Roman"

RANK1_OFF_COLOR = "#4169E1"
RANK1_SEN_COLOR = RANK1_OFF_COLOR
OBJECTIVE_COLOR = "#FF6B6B"
GAUSSIAN_COLOR = "#4169E1"
GRID_COLOR = "#E0E0E0"

# 与本脚本同目录（Fig1/）下的数据文件
HISTORY_CSV_NAME = "first_sample_cccp_history.csv"
CONVERGENCE_CSV_NAME = "convergence_iterations.csv"


# ── 数据加载 ──
def load_energy_rank1():
    """读取同目录 first_sample_cccp_history.csv（由 src/inner/main.py 生成）。

    列含义：
        iteration       —— 迭代轮次（第 0 行为初始点 x^(0)）
        objective_value —— 问题 P4 中的能量项
        w_rank1_gap     —— 感知波束成形 W_sen 的秩一罚项
        b_rank1_gap     —— 卸载波束成形 B_off 的秩一罚项
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    history_path = os.path.join(base_dir, HISTORY_CSV_NAME)

    iterations = []
    objective_values = []
    rank1_off_values = []
    rank1_sen_values = []
    with open(history_path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required_columns = {"iteration", "objective_value", "w_rank1_gap", "b_rank1_gap"}
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            raise ValueError(
                "CSV file {} must contain the columns: {}. Found columns: {}".format(
                    history_path, ", ".join(sorted(required_columns)), reader.fieldnames))
        for row in reader:
            iterations.append(int(row["iteration"]))
            objective_values.append(float(row["objective_value"]))
            rank1_sen_values.append(float(row["w_rank1_gap"]))
            rank1_off_values.append(float(row["b_rank1_gap"]))

    iters = np.array(iterations)
    return (iters, np.array(objective_values),
            iters, np.array(rank1_off_values),
            iters, np.array(rank1_sen_values))


def load_case_convergence_csv(csv_path):
    case_ids = []
    convergence_iterations = []
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            case_ids.append(int(row["case_id"]))
            convergence_iterations.append(int(float(row["convergence_iterations"])))
    return np.array(case_ids), np.array(convergence_iterations)


def resolve_convergence_csv(base_dir):
    """定位同目录（Fig1/）下的 convergence_iterations.csv。"""
    csv_path = os.path.join(base_dir, CONVERGENCE_CSV_NAME)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            "未在当前目录找到数据文件：{}".format(csv_path))
    return csv_path


# ── 上子图：energy & rank-one penalty 收敛 ──
def plot_energy_rank1(ax):
    (iters_objective, objective_values,
     iters_rank1_off, rank1_off_values,
     iters_rank1_sen, rank1_sen_values) = load_energy_rank1()

    ax1 = ax
    ax2 = ax1.twinx()

    line1 = ax1.plot(
        iters_objective,
        objective_values,
        marker="s",
        linestyle="-",
        markersize=8,
        label="The energy term in problem P4",
        color=OBJECTIVE_COLOR,
        linewidth=2,
        markerfacecolor="white",
        markeredgewidth=1.5,
    )
    line2 = ax2.plot(
        iters_rank1_off,
        rank1_off_values,
        marker="o",
        linestyle=":",
        markersize=8,
        label="The penalty term for offloading beamforming in problem P4",
        color=RANK1_OFF_COLOR,
        linewidth=2,
        markerfacecolor="white",
        markeredgewidth=1.5,
        clip_on=False,
    )
    line3 = ax2.plot(
        iters_rank1_sen,
        rank1_sen_values,
        marker="^",
        linestyle="--",
        markersize=8,
        label="The penalty term for sensing beamforming in problem P4",
        color=RANK1_SEN_COLOR,
        linewidth=2,
        markerfacecolor="white",
        markeredgewidth=1.5,
        clip_on=False,
    )

    ax1.set_xlabel("The number of iterations", fontsize=20)
    ax1.set_ylabel("The energy term in problem P4", fontsize=20, color=OBJECTIVE_COLOR)
    ax2.set_ylabel("The penalty term in problem P4", fontsize=20, color=RANK1_OFF_COLOR)
    ax1.spines["left"].set_color(OBJECTIVE_COLOR)
    ax2.spines["right"].set_color(RANK1_OFF_COLOR)

    ax1.tick_params(axis="y", which="major", labelsize=20, colors=OBJECTIVE_COLOR)
    ax2.tick_params(axis="y", which="major", labelsize=20, colors=RANK1_OFF_COLOR)
    ax1.spines['left'].set_color(OBJECTIVE_COLOR)
    ax2.spines['right'].set_color(RANK1_OFF_COLOR)
    ax2.spines['left'].set_visible(False)
    ax1.spines['top'].set_color('#E0E0E0')
    ax1.spines['bottom'].set_color('#E0E0E0')
    # y 轴范围按实际数据自适应（不同 λ/样本下能量量级不同，避免曲线被硬编码区间裁掉）
    obj_span = max(float(objective_values.max() - objective_values.min()), 1e-6)
    ax1.set_ylim(objective_values.min() - 0.30 * obj_span,
                 objective_values.max() + 0.15 * obj_span)
    ax1.tick_params(axis="x", which="major", labelsize=20)
    ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax1.grid(
        True,
        linestyle=(0, (3, 5)),
        color=GRID_COLOR,
        linewidth=1.0,
        alpha=1.0,
        zorder=1,
    )

    # 内嵌图
    axins = inset_axes(ax1, width="35%", height="35%", bbox_to_anchor=(0.35, -0.05, 0.6, 0.6), bbox_transform=ax2.transAxes)
    start_idx = 1
    axins.plot(
        iters_objective[start_idx:],
        objective_values[start_idx:],
        marker="s",
        linestyle="-",
        markersize=6,
        color=OBJECTIVE_COLOR,
        linewidth=2,
        markerfacecolor="white",
        markeredgewidth=1.5,
        clip_on=False,
    )
    axins.set_xlim(iters_objective[start_idx], iters_objective[-1])
    axins.set_ylim(objective_values[start_idx:].min() - 0.005, objective_values[start_idx:].max() + 0.005)
    axins.grid(
        True,
        linestyle=(0, (3, 5)),
        color=GRID_COLOR,
        linewidth=0.8,
        alpha=0.8,
        zorder=1,
    )
    # 内嵌图显示 x、y 轴刻度
    axins.xaxis.set_major_locator(MaxNLocator(nbins=4))
    axins.yaxis.set_major_locator(MaxNLocator(nbins=4))
    axins.tick_params(axis="both", which="major", labelsize=12)
    # 仅保留左下角(loc1=3, loc2=4)与右下角(loc1=4, loc2=3)连接虚线
    mark_inset(ax1, axins, loc1=3, loc2=4, fc="none", ec=OBJECTIVE_COLOR, linewidth=1.5, linestyle="--")
    mark_inset(ax1, axins, loc1=4, loc2=3, fc="none", ec=OBJECTIVE_COLOR, linewidth=1.5, linestyle="--")

    # 内嵌图 2：放大 offloading beam（蓝色点线 :）与 sensing beam（蓝色虚线 --）的 rank-1 惩罚项收敛尾部
    # 注意：parent 用 ax2（右轴），保证 mark_inset 连线在右轴坐标下正确指向蓝色曲线区域
    axins2 = inset_axes(ax2, width="35%", height="35%", bbox_to_anchor=(0.01, -0.05, 0.6, 0.6), bbox_transform=ax2.transAxes)
    # offloading beam rank-1 penalty（蓝色点线）
    axins2.plot(
        iters_rank1_off[start_idx:],
        rank1_off_values[start_idx:],
        marker="o",
        linestyle=":",
        markersize=6,
        color=RANK1_OFF_COLOR,
        linewidth=2,
        markerfacecolor="white",
        markeredgewidth=1.5,
        clip_on=False,
    )
    # sensing beam rank-1 penalty（蓝色虚线）
    axins2.plot(
        iters_rank1_sen[start_idx:],
        rank1_sen_values[start_idx:],
        marker="^",
        linestyle="--",
        markersize=6,
        color=RANK1_SEN_COLOR,
        linewidth=2,
        markerfacecolor="white",
        markeredgewidth=1.5,
        clip_on=False,
    )
    axins2.set_xlim(iters_rank1_off[start_idx], iters_rank1_off[-1])
    axins2.set_ylim(
        min(rank1_off_values[start_idx:].min() - 0.00001, rank1_sen_values[start_idx:].min() - 0.00001),
        max(rank1_off_values[start_idx:].max() + 0.000005, rank1_sen_values[start_idx:].max() + 0.000005),
    )
    axins2.grid(
        True,
        linestyle=(0, (3, 5)),
        color=GRID_COLOR,
        linewidth=0.8,
        alpha=0.8,
        zorder=1,
    )
    axins2.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    axins2.yaxis.set_major_locator(MaxNLocator(nbins=3))
    axins2.tick_params(axis="both", which="major", labelsize=12)
    # 接线用蓝色，指向主图中蓝色 rank-1 惩罚项曲线区域；单次 mark_inset 产生两条连线（左下角3 + 右下角4）
    mark_inset(ax2, axins2, loc1=3, loc2=4, fc="none", ec=RANK1_OFF_COLOR, linewidth=1.5, linestyle="--")

    lines = line1 + line2 + line3
    labels = [line.get_label() for line in lines]
    # 图例挂到 figure 上，使其在 figure 层级拥有最高 zorder，不被独立的内嵌图 axes 遮挡
    fig = ax1.figure
    legend = fig.legend(
        lines, labels, fontsize=18, loc="upper right",
        bbox_to_anchor=(1.0, 1.0), bbox_transform=ax1.transAxes,
    )
    legend.set_zorder(100)


# ── 下子图：random case 收敛迭代次数 ──
def plot_case_convergence(ax):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = resolve_convergence_csv(base_dir)
    case_ids, convergence_iterations = load_case_convergence_csv(csv_path)

    ax.plot(
        case_ids,
        convergence_iterations,
        marker="s",
        linestyle="-",
        markersize=8,
        linewidth=2,
        color=GAUSSIAN_COLOR,
        markerfacecolor="white",
        markeredgewidth=1.5,
        clip_on=False,
    )

    ax.set_xlabel("Random cases", fontsize=20)
    ax.set_ylabel("The number of total iterations", fontsize=20)
    ax.tick_params(axis="x", which="major", labelsize=20)
    ax.tick_params(axis="y", which="major", labelsize=20)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlim(1, 30)
    ax.grid(
        True,
        linestyle=(0, (3, 5)),
        color=GRID_COLOR,
        linewidth=1.0,
        alpha=1.0,
        zorder=1,
    )


# ── 主程序：2×1 子图 ──
def main():
    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [1.0, 1.0]}
    )

    plot_energy_rank1(ax_top)
    plot_case_convergence(ax_bottom)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
