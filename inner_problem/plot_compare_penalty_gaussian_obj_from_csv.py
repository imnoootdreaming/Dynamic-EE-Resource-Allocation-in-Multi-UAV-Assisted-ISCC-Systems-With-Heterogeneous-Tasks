import csv
import os

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams["font.family"] = "Times New Roman"

PENALTY_COLOR = "#FF6B6B"
GAUSSIAN_COLOR = "#4169E1"
GRID_COLOR = "#E0E0E0"


def load_compare_data(csv_path):
    case_ids = []
    penalty_values = []
    gaussian_values = []

    with open(csv_path, "r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {"case_id", "penalty_based_obj", "gaussian_based_obj"}
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            raise ValueError(
                f"CSV file must contain the columns: case_id, penalty_based_obj, gaussian_based_obj. "
                f"Found columns: {reader.fieldnames}"
            )

        for row in reader:
            case_ids.append(int(row["case_id"]))
            penalty_values.append(float(row["penalty_based_obj"]))
            gaussian_values.append(float(row["gaussian_based_obj"]))

    return np.array(case_ids), np.array(penalty_values), np.array(gaussian_values)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(
        base_dir,
        "20260323_120237_compare_penalty_gaussian_based_obj_val.csv",
    )

    case_ids, penalty_values, gaussian_values = load_compare_data(csv_path)

    x = np.arange(case_ids.size)
    bar_width = 0.36

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.bar(
        x - bar_width / 2,
        penalty_values,
        width=bar_width,
        color=PENALTY_COLOR,
        edgecolor="black",
        linewidth=0.6,
        label="Penalty-based CCCP",
        zorder=3,
    )
    ax.bar(
        x + bar_width / 2,
        gaussian_values,
        width=bar_width,
        color=GAUSSIAN_COLOR,
        edgecolor="black",
        linewidth=0.6,
        label="Gaussian-randomization-based CCCP",
        zorder=3,
    )

    ax.set_xlabel("Random cases", fontsize=24)
    ax.set_ylabel("The weighted total energy consumption (J)", fontsize=24)
    ax.set_xticks(x)
    ax.set_xticklabels(case_ids)
    ax.tick_params(axis="x", which="major", labelsize=24)
    ax.tick_params(axis="y", which="major", labelsize=24)
    ax.set_ylim(0, max(max(penalty_values), max(gaussian_values)) * 1.22)



    ax.grid(
        True,
        axis="y",
        linestyle=(0, (3, 5)),
        color=GRID_COLOR,
        linewidth=1.0,
        alpha=1.0,
        zorder=1,
    )
    ax.legend(fontsize=24)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
