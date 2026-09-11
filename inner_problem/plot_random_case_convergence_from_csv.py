import csv
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import MultipleLocator  # ← 新增

plt.rcParams["font.family"] = "Times New Roman"

GRID_COLOR = "#E0E0E0"
GAUSSIAN_COLOR = "#4169E1"

def load_case_convergence_csv(csv_path):
    case_ids = []
    convergence_iterations = []

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            case_ids.append(int(row["case_id"]))
            convergence_iterations.append(int(float(row["convergence_iterations"])))

    return case_ids, convergence_iterations


def find_latest_convergence_csv(base_dir):
    candidate_files = [
        file_name
        for file_name in os.listdir(base_dir)
        if file_name.endswith("_random_case_convergence_iterations.csv")
    ]
    if not candidate_files:
        raise FileNotFoundError("No *_random_case_convergence_iterations.csv file was found in the current directory.")
    candidate_files.sort()
    return os.path.join(base_dir, candidate_files[-1])


def plot_case_convergence(csv_path):
    case_ids, convergence_iterations = load_case_convergence_csv(csv_path)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(
        case_ids,
        convergence_iterations,
        marker="s",
        linestyle="-",
        markersize=8,
        linewidth=2,
        color = GAUSSIAN_COLOR,
        markerfacecolor="white",
        markeredgewidth=1.5,
        clip_on=False,  
    )

    ax.set_xlabel("Random cases", fontsize=24)
    ax.set_ylabel("The number of iterations", fontsize=24)
    ax.tick_params(axis="x", which="major", labelsize=18)
    ax.tick_params(axis="y", which="major", labelsize=18)
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

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = find_latest_convergence_csv(base_dir)
    plot_case_convergence(csv_path)
