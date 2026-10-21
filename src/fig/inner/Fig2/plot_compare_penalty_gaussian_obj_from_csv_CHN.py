import csv
import os

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Rectangle

# 字体与样式设置：英文及数字使用 Times New Roman，中文使用宋体。
plt.rcParams["font.family"] = ["Times New Roman", "SimSun"]
plt.rcParams["font.serif"] = ["Times New Roman", "SimSun"]
plt.rcParams["font.sans-serif"] = ["Times New Roman", "SimSun"]
plt.rcParams["mathtext.fontset"] = "custom"

PENALTY_COLOR = "#FF6B6B"
GAUSSIAN_COLOR = "#4169E1"
GRID_COLOR = "#E0E0E0"


def load_compare_data(csv_path):
    case_ids = []
    penalty_values = []
    gaussian_values = []
    penalty_times = []
    gaussian_times = []

    with open(csv_path, "r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {"case_id", "penalty_based_obj", "gaussian_based_obj", "penalty_based_time", "gaussian_based_time"}
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            raise ValueError(
                f"CSV file must contain the columns: case_id, penalty_based_obj, gaussian_based_obj, penalty_based_time, gaussian_based_time. "
                f"Found columns: {reader.fieldnames}"
            )

        for row in reader:
            case_ids.append(int(row["case_id"]))
            penalty_values.append(float(row["penalty_based_obj"]))
            gaussian_values.append(float(row["gaussian_based_obj"]))
            penalty_times.append(float(row["penalty_based_time"]))
            gaussian_times.append(float(row["gaussian_based_time"]))

    return np.array(case_ids), np.array(penalty_values), np.array(gaussian_values), np.array(penalty_times), np.array(gaussian_times)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(
        base_dir,
        "compare_pc3p_gc3p_results.csv",
    )

    case_ids, penalty_values, gaussian_values, penalty_times, gaussian_times = load_compare_data(csv_path)

    x = np.arange(case_ids.size)
    tick_mask = case_ids % 5 == 0
    bar_width = 0.36
    fig, (ax_time, ax_obj) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [1.0, 1.0]})

    # --- Top subplot: runtime comparison (line plot) ---
    ax_time.plot(
        x,
        penalty_times,
        color=PENALTY_COLOR,
        marker="o",
        linewidth=1.5,
        markersize=6,
        label="PC3P",
        zorder=3,
    )
    ax_time.plot(
        x,
        gaussian_times,
        color=GAUSSIAN_COLOR,
        marker="s",
        linewidth=1.5,
        markersize=6,
        label="GC3P",
        zorder=3,
    )
    ax_time.set_xlabel("随机案例", fontsize=24)
    ax_time.set_ylabel("运行时间 (s)", fontsize=24)
    ax_time.set_xticks(x[tick_mask])
    ax_time.set_xticklabels(case_ids[tick_mask])
    ax_time.tick_params(axis="x", which="major", labelsize=18)
    ax_time.tick_params(axis="y", which="major", labelsize=18)
    ax_time.set_ylim(0, max(max(penalty_times), max(gaussian_times)) * 1.55)
    ax_time.grid(
        True,
        linestyle=(0, (3, 5)),
        color=GRID_COLOR,
        linewidth=1.0,
        alpha=1.0,
        zorder=1,
    )
    ax_time.legend(fontsize=20, loc="upper right")

    # --- Bottom subplot: objective value comparison (bar chart) ---
    ax_obj.bar(
        x - bar_width / 2,
        penalty_values,
        width=bar_width,
        color=PENALTY_COLOR,
        edgecolor="black",
        linewidth=0.6,
        label="PC3P",
        zorder=3,
    )
    ax_obj.bar(
        x + bar_width / 2,
        gaussian_values,
        width=bar_width,
        color=GAUSSIAN_COLOR,
        edgecolor="black",
        linewidth=0.6,
        label="GC3P",
        zorder=3,
    )
    ax_obj.set_xlabel("随机案例", fontsize=24)
    ax_obj.set_ylabel("加权总能耗 (J)", fontsize=24)
    ax_obj.set_xticks(x[tick_mask])
    ax_obj.set_xticklabels(case_ids[tick_mask])
    ax_obj.tick_params(axis="x", which="major", labelsize=18)
    ax_obj.tick_params(axis="y", which="major", labelsize=18)
    ax_obj.set_ylim(0, max(max(penalty_values), max(gaussian_values)) * 1.3)

    ax_obj.grid(
        True,
        linestyle=(0, (3, 5)),
        color=GRID_COLOR,
        linewidth=1.0,
        alpha=1.0,
        zorder=1,
    )
    ax_obj.legend(fontsize=20, loc="upper right")

    # 收紧左右两侧空白：x 轴范围仅比最外侧柱边缘多留少量余量；
    # 同时让上下两个子图共用同一 x 轴范围，保证柱子与折线在横向上对齐。
    side_pad = bar_width / 2 + 0.6
    ax_time.set_xlim(x[0] - side_pad, x[-1] + side_pad)
    ax_obj.set_xlim(x[0] - side_pad, x[-1] + side_pad)

    plt.tight_layout()
    # 按实际文字边界仅上移下子图标签的底部溢出部分，保留 1 像素余量。
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    label = ax_obj.yaxis.label
    overflow = (fig.bbox.y0 + 1) - label.get_window_extent(renderer).y0
    if overflow > 0:
        label.set_y(label.get_position()[1] + overflow / ax_obj.bbox.height)

    plt.show()


if __name__ == "__main__":
    main()
