import csv
import os

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Rectangle

plt.rcParams["font.family"] = "Times New Roman"

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
    ax_time.set_xlabel("Random cases", fontsize=18)
    ax_time.set_ylabel("Running time (s)", fontsize=18)
    ax_time.set_xticks(x)
    ax_time.set_xticklabels(case_ids)
    ax_time.tick_params(axis="x", which="major", labelsize=18)
    ax_time.tick_params(axis="y", which="major", labelsize=18)
    ax_time.set_ylim(-5, 300)
    ax_time.grid(
        True,
        linestyle=(0, (3, 5)),
        color=GRID_COLOR,
        linewidth=1.0,
        alpha=1.0,
        zorder=1,
    )
    ax_time.legend(fontsize=18, loc="upper right")

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
    ax_obj.set_xlabel("Random cases", fontsize=18)
    ax_obj.set_ylabel("The weighted total energy consumption (J)", fontsize=18)
    ax_obj.set_xticks(x)
    ax_obj.set_xticklabels(case_ids)
    ax_obj.tick_params(axis="x", which="major", labelsize=18)
    ax_obj.tick_params(axis="y", which="major", labelsize=18)
    ax_obj.set_ylim(0, max(max(penalty_values), max(gaussian_values)) * 1.5)

    ax_obj.grid(
        True,
        linestyle=(0, (3, 5)),
        color=GRID_COLOR,
        linewidth=1.0,
        alpha=1.0,
        zorder=1,
    )
    ax_obj.legend(fontsize=18, loc="upper right")

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()