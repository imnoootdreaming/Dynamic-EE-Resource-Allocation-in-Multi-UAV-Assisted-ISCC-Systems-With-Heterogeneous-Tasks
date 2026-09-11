import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"  # STIX 字体与 Times New Roman 风格一致

GRID_COLOR = "#E0E0E0"


def extract_rho_value(filename):
    match = re.search(r"rho([0-9.]+)\.csv$", filename)
    if match is None:
        raise ValueError(f"Cannot parse rho value from filename: {filename}")
    return float(match.group(1))


def load_energy_series(base_dir):
    energy_files = [
        name for name in os.listdir(base_dir)
        if "compare_energy_val_list_rho" in name and name.endswith(".csv")
    ]
    if len(energy_files) == 0:
        raise FileNotFoundError("No matching energy CSV files were found in the current directory.")

    energy_files = sorted(energy_files, key=extract_rho_value)
    series = []
    for filename in energy_files:
        file_path = os.path.join(base_dir, filename)
        values = np.atleast_1d(np.loadtxt(file_path, delimiter=","))
        rho = extract_rho_value(filename)
        series.append((rho, values))
    return series


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    series = load_energy_series(base_dir)

    fig, ax = plt.subplots(figsize=(10, 8))

    markers = ["s", "o", "^", "D", "v", "P"]
    linestyles = ["-", "--", "-.", ":"]

    for idx, (rho, values) in enumerate(series):
        iterations = np.arange(1, values.size + 1)
        # 判断是否为整数（如 10.0），若是则转为整数显示
        rho_display = int(rho) if rho.is_integer() else rho
        ax.plot(
            iterations,
            values,
            marker=markers[idx % len(markers)],
            linestyle=linestyles[idx % len(linestyles)],
            markersize=8,
            linewidth=2,
            markerfacecolor="white",
            markeredgewidth=1.5,
            label=fr"$\rho = {rho_display}$"
        )

    ax.set_xlabel("The number of iterations", fontsize=24)
    ax.set_ylabel("The weighted total energy consumption (J)", fontsize=24)
    ax.tick_params(axis="both", which="major", labelsize=24)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xticks(range(0, 31, 5))  # 0,5,10,15,20,25,30
    ax.set_xlim(0, 31)  # 设置x轴范围

    ax.grid(
        True,
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
