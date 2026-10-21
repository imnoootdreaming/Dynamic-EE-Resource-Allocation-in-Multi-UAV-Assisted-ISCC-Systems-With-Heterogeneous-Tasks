import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- 字体与样式设置（与 plot_CHN.py 保持一致）---
plt.rcParams["font.family"] = ["Times New Roman", "SimSun"]
plt.rcParams["font.serif"] = ["Times New Roman", "SimSun"]
plt.rcParams["font.sans-serif"] = ["Times New Roman", "SimSun"]
plt.rcParams["mathtext.fontset"] = "custom"

# ===================== 读取数据 =====================
# CSV 结构: 左侧 user num 段 (A-C 列), 空列 (D), 右侧 task_size 段 (E-G 列)
df = pd.read_csv('without_traj_opt.csv')

# 清洗 user num 段 (含制表符/空格)
user_col = 'user num'
df_user = pd.DataFrame({
    'user_num': pd.to_numeric(df[user_col].astype(str).str.replace(r'\s+', '', regex=True), errors='coerce'),
    'opt':     df['opt'],
    'no_opt':  df['no_opt'],
}).dropna().sort_values('user_num').reset_index(drop=True)
df_user['user_num'] = df_user['user_num'].astype(int)

# task_size 段: 列名可能因空列而变成 'task_size' 与 'opt.1'/'no_opt.1', 做兼容处理
task_col = 'task_size'
cols = list(df.columns)
ti = cols.index(task_col)
task_size_raw = df.iloc[:, ti: ti + 3]
task_size_raw.columns = ['task_size', 'opt', 'no_opt']
df_task = task_size_raw.copy()
df_task['task_size'] = df_task['task_size'].astype(float)
df_task = df_task.dropna().sort_values('task_size').reset_index(drop=True)

# ===================== 配色 (沿用 plot_CHN.py 风格) =====================
COLOR_OPT   = '#4169E1'   # Royal Blue  -> 有无人机轨迹优化
COLOR_NOOPT = '#FF6B6B'   # Coral Red   -> 无无人机轨迹优化
GRID_COLOR  = '#E0E0E0'

# ===================== 绘图: 两个子图 =====================
fig, axes = plt.subplots(2, 1, figsize=(10, 8))

# ---------- 上子图: 任务数据量变化 ----------
ax = axes[0]
ax.plot(df_task['task_size'], df_task['opt'],
        marker='o', linestyle='-', markersize=8,
        label="UAV 轨迹优化",
        color=COLOR_OPT, linewidth=2,
        markerfacecolor='white', markeredgewidth=1.5)
ax.plot(df_task['task_size'], df_task['no_opt'],
        marker='s', linestyle='--', markersize=8,
        label="无 UAV 轨迹优化",
        color=COLOR_NOOPT, linewidth=2,
        markerfacecolor='white', markeredgewidth=1.5)
ax.set_xlabel("每个任务的数据量 (knats)", fontsize=24)
ax.set_ylabel("时延 (s)", fontsize=24)
ax.grid(True, linestyle='--', color=GRID_COLOR, alpha=0.8, zorder=1)
ax.tick_params(axis='both', which='major', labelsize=18)
ax.set_xticks(df_task['task_size'].astype(int).tolist())
ax.legend(fontsize=20, loc='upper left')

# ---------- 下子图: 用户数量变化 ----------
ax = axes[1]
ax.plot(df_user['user_num'], df_user['opt'],
        marker='o', linestyle='-', markersize=8,
        label="UAV 轨迹优化",
        color=COLOR_OPT, linewidth=2,
        markerfacecolor='white', markeredgewidth=1.5)
ax.plot(df_user['user_num'], df_user['no_opt'],
        marker='s', linestyle='--', markersize=8,
        label="无 UAV 轨迹优化",
        color=COLOR_NOOPT, linewidth=2,
        markerfacecolor='white', markeredgewidth=1.5)
ax.set_xlabel("用户数量", fontsize=24)
ax.set_ylabel("时延 (s)", fontsize=24)
ax.grid(True, linestyle='--', color=GRID_COLOR, alpha=0.8, zorder=1)
ax.tick_params(axis='both', which='major', labelsize=18)
ax.set_xticks(df_user['user_num'].astype(int).tolist())
ax.legend(fontsize=20, loc='upper left')

plt.tight_layout()
plt.show()
