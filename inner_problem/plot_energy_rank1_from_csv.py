import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"

RANK1_OFF_COLOR = "#4169E1"
RANK1_SEN_COLOR = RANK1_OFF_COLOR
OBJECTIVE_COLOR = "#FF6B6B"
GRID_COLOR = "#E0E0E0"

base_dir = os.path.dirname(os.path.abspath(__file__))
objective_path = os.path.join(base_dir, "20260326_113445_objective_val_list_rho0.1_uav4.csv")
rank1_off_path = os.path.join(base_dir, "20260326_113445_rank1_off_val_list_rho0.1_uav4.csv")
rank1_sen_path = os.path.join(base_dir, "20260326_113445_rank1_sen_val_list_rho0.1_uav4.csv")

objective_values = np.atleast_1d(np.loadtxt(objective_path, delimiter=","))
rank1_off_values = np.atleast_1d(np.loadtxt(rank1_off_path, delimiter=","))
rank1_sen_values = np.atleast_1d(np.loadtxt(rank1_sen_path, delimiter=","))

iters_objective = np.arange(0, objective_values.size)
iters_rank1_off = np.arange(0, rank1_off_values.size)
iters_rank1_sen = np.arange(0, rank1_sen_values.size)

fig, ax1 = plt.subplots(figsize=(10, 8))
ax2 = ax1.twinx()

line1 = ax1.plot(
    iters_objective,
    objective_values,
    marker="s",
    linestyle="-",
    markersize=8,
    label="The value of objective function",
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
    label="The rank-one penalty term for offloading beamforming",
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
    label="The rank-one penalty term for sensing beamforming",
    color=RANK1_SEN_COLOR,
    linewidth=2,
    markerfacecolor="white",
    markeredgewidth=1.5,
    clip_on=False,  
)

ax1.set_xlabel("The number of iterations", fontsize=24)
ax1.set_ylabel("The objective function value of problem P6", fontsize=24, color=OBJECTIVE_COLOR)
ax2.set_ylabel("The value of penalty terms", fontsize=24, color=RANK1_OFF_COLOR)
ax1.spines["left"].set_color(OBJECTIVE_COLOR)
ax2.spines["right"].set_color(RANK1_OFF_COLOR)

ax1.tick_params(axis="y", which="major", labelsize=24, colors=OBJECTIVE_COLOR)
ax2.tick_params(axis="y", which="major", labelsize=24, colors=RANK1_OFF_COLOR)
ax1.spines['left'].set_color(OBJECTIVE_COLOR)
ax2.spines['right'].set_color(RANK1_OFF_COLOR)
ax2.spines['left'].set_visible(False)
ax1.spines['top'].set_color('#E0E0E0')
ax1.spines['bottom'].set_color('#E0E0E0')
ax1.set_ylim(4, 9)
# x轴保持黑色（论文规范）
ax1.tick_params(axis="x", which="major", labelsize=18)
ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
ax1.grid(
    True,
    linestyle=(0, (3, 5)),  # 短虚线
    color=GRID_COLOR,
    linewidth=1.0,
    alpha=1.0,
    zorder=1
)
lines = line1 + line2 + line3
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, fontsize=20, loc="upper right")

plt.tight_layout()
plt.show()
