import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import ConnectionPatch, Rectangle

plt.rcParams['font.family'] = 'Times New Roman'

GRID_COLOR = "#E0E0E0"


def add_inset(ax, algo_names, means, episodes, zoom_left, zoom_right, position,
              colors, zorder_map):
    """在 ax 上添加局部放大图（参考 lr_decay_cmp.py 的内嵌图风格）。

    :param ax:         主坐标轴
    :param algo_names: 需要放大的算法名列表（参与 y 轴范围计算并绘制）
    :param means:      {algo_name: ndarray} 各算法平滑后的均值序列
    :param episodes:   X 轴 episode 数组
    :param zoom_left:  放大区域横坐标左端点（episode 索引）
    :param zoom_right: 放大区域横坐标右端点（episode 索引）
    :param position:   [x, y, w, h] 内嵌子图位置（axes fraction）
    :param colors:     {algo_name: (main_color, shade_color)} 颜色字典
    :param zorder_map: {algo_name: int} 图层顺序字典，与主图一致
    """
    ax_ins = ax.inset_axes(position)

    # 按 zorder 从低到高绘制，保证内嵌图曲线覆盖顺序与主图一致
    for name in sorted(algo_names, key=lambda a: zorder_map.get(a, 1)):
        main_color = colors[name][0]
        zorder = zorder_map.get(name, 1)
        ax_ins.plot(
            episodes,
            means[name],
            color=main_color,
            linewidth=1.5,
            zorder=zorder
        )

    ax_ins.set_xlim(zoom_left, zoom_right)
    # 内嵌图 y 轴范围硬编码限制（与 lr_decay_cmp.py 相同）
    ax_ins.set_ylim(ZOOM_Y_BOTTOM, ZOOM_Y_TOP)
    ax_ins.tick_params(axis='both', which='major', labelsize=14,
                       direction='in', width=0.3)
    ax_ins.grid(True, linestyle=(0, (3, 5)), color=GRID_COLOR,
                linewidth=0.8, alpha=1.0, zorder=1)

    # ── 画矩形框和连接线（点划线，与主图风格一致）──
    # 内嵌子图置于最高层级，框/线位于主图曲线之上、内嵌子图之下
    ax_ins.set_zorder(20)

    rx0, rx1 = zoom_left, zoom_right
    ry0, ry1 = ZOOM_Y_BOTTOM, ZOOM_Y_TOP

    rect = Rectangle((rx0, ry0), rx1 - rx0, ry1 - ry0,
                     fill=False, edgecolor='black', linewidth=1.5,
                     linestyle=(0, (2, 2)), alpha=0.6, transform=ax.transData,
                     zorder=10)
    ax.add_patch(rect)

    corners = [
        (rx0, ry0, 0, 0),
        (rx0, ry1, 0, 1),
        (rx1, ry0, 1, 0),
        (rx1, ry1, 1, 1),
    ]
    for xA, yA, xB, yB in corners:
        con = ConnectionPatch(
            xyA=(xA, yA), xyB=(xB, yB),
            coordsA='data', coordsB='axes fraction',
            axesA=ax, axesB=ax_ins,
            color='black', linewidth=1.5,
            linestyle=(0, (2, 2)), alpha=0.6, zorder=10)
        ax.add_patch(con)

    return ax_ins


# --- 1. 配置参数 ---

# 定义要处理的随机种子列表（每个文件都是seed=42）
SEEDS = [42]

# 定义算法的CSV文件命名模板（使用{}占位符，实际会被seed替换）
ALGORITHM_TEMPLATES = {
    'Pure MHBPPO': 'Pure_MHBPPO.csv',
    'MHGPPO-PC3P': 'MHGPPO.csv',
    'MHSAC-PC3P': 'MHSAC.csv',
    'MHBPPO-PC3P': 'MHBPPO.csv',
    'SHBPPO-PC3P': 'SHBPPO.csv',
}

# 滑动平均窗口大小（1表示不平滑）
SMOOTHING_WINDOW = 1

# 内嵌图放大参数（与 lr_decay_cmp.py 保持一致）
ZOOM_Y_BOTTOM = 0.75
ZOOM_Y_TOP = 1.0
ZOOM_LEFT = 7500
ZOOM_RIGHT = 8025
# 内嵌子图位置（axes fraction）：[x, y, w, h]
ZOOM_POSITION = [0.6, 0.41, 0.30, 0.3]


# --- 2. 数据加载与处理函数 ---
def load_and_process_rewards(file_template, seeds, column_name='reward'):
    """
    加载多个随机种子的奖励数据，并计算均值和标准差。
    """
    all_rewards_data = []

    for seed in seeds:
        try:
            filepath = file_template.format(seed)
            df = pd.read_csv(filepath)
            all_rewards_data.append(df[column_name])
        except FileNotFoundError:
            print(f"警告: 找不到文件 {filepath}，将跳过该种子。")
            continue
        except KeyError:
            print(f"警告: 文件 {filepath} 中找不到列 '{column_name}'，将跳过该种子。")
            continue

    if not all_rewards_data:
        print(f"错误: 无法为模板 '{file_template}' 加载任何数据，请检查文件名和路径。")
        return None, None, None

    # 找到最短的数据长度，对齐所有数据
    min_length = min(len(data) for data in all_rewards_data)
    aligned_data = [data.head(min_length).values for data in all_rewards_data]

    # 计算均值和标准差
    rewards_matrix = np.array(aligned_data)
    mean_rewards = np.mean(rewards_matrix, axis=0)
    std_rewards = np.std(rewards_matrix, axis=0)

    # 平滑处理
    mean_smoothed = pd.Series(mean_rewards).rolling(window=SMOOTHING_WINDOW, min_periods=1).mean()
    std_smoothed = pd.Series(std_rewards).rolling(window=SMOOTHING_WINDOW, min_periods=1).mean()

    episodes = np.arange(min_length)

    return episodes, mean_smoothed, std_smoothed


# --- 3. 算法对比绘图函数 ---
def plot_algorithm_comparison(plot_data):
    """
    在同一张图上绘制不同算法的奖励曲线进行对比，并添加局部放大图。
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # 定义每个算法的颜色方案 (主色, 阴影色)
    # MHBPPO-PC3P: 红色, MHSAC-PC3P: 蓝色, MHGPPO-PC3P: 绿色, Pure MHBPPO: 紫色
    colors = {
        'MHBPPO-PC3P': ('#D62728', '#f4a6a7'),          # 红色
        'MHSAC-PC3P': ('#1F77B4', '#aec7e8'),            # 蓝色
        'MHGPPO-PC3P': ('#2CA02C', '#a1d9a1'),           # 绿色
        'Pure MHBPPO': ('#9467BD', '#c5b0d5'),           # 紫色
        'SHBPPO-PC3P': ('#FF7F0E', '#ffbb86'),           # 橙色
    }

    # 算法显示名称映射（图例中显示更友好的名称）
    label_mapping = {
        'Pure MHBPPO': 'Pure MHBPPO',
        'MHGPPO-PC3P': 'MHGPPO-PC3P',
        'MHBPPO-PC3P': 'MHBPPO-PC3P',
        'MHSAC-PC3P': 'MHSAC-PC3P',
        'SHBPPO-PC3P': 'SHBPPO-PC3P',
    }

    # 存储所有的均值曲线数据，用于局部放大
    means = {}

    # 绘制每个算法的曲线
    for data in plot_data:
        algo_name = data['algorithm']
        episodes = data['episodes']
        mean_rewards = data['mean']
        std_rewards = data['std']

        main_color, shade_color = colors.get(algo_name, ('blue', 'lightblue'))
        linewidth = 1.5
        alpha = 0.14

        # 图层顺序: MHBPPO-PC3P(5) > Pure MHBPPO(4) > MHSAC-PC3P(3) > MHGPPO-PC3P(2) > SHBPPO-PC3P(1)
        zorder_map = {
            'MHBPPO-PC3P': 5,
            'Pure MHBPPO': 4,
            'MHSAC-PC3P': 3,
            'MHGPPO-PC3P': 2,
            'SHBPPO-PC3P': 1,
        }
        zorder = zorder_map.get(algo_name, 2)

        # 绘制标准差阴影（单个种子时阴影宽度为0，可忽略）
        # ax.fill_between(
        #     episodes,
        #     mean_rewards - std_rewards,
        #     mean_rewards + std_rewards,
        #     color=shade_color,
        #     alpha=alpha,
        #     zorder=1
        # )

        # 绘制均值曲线
        ax.plot(
            episodes,
            mean_rewards,
            color=main_color,
            label=label_mapping.get(algo_name, algo_name),
            linewidth=linewidth,
            zorder=zorder
        )

        means[algo_name] = mean_rewards.values

    ax.set_xlabel('Episodes', fontsize=24)
    ax.set_ylabel('Average reward', fontsize=24)
    # 图例顺序（按照希望展示的顺序）
    desired_order = ['MHBPPO-PC3P', 'MHSAC-PC3P', 'MHGPPO-PC3P', 'SHBPPO-PC3P', 'Pure MHBPPO']
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(name) for name in desired_order if name in labels]
    handles = [handles[i] for i in order]
    labels = [labels[i] for i in order]
    ax.legend(handles, labels, fontsize=24, loc='lower right')
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.grid(True, linestyle=(0, (3, 5)), color=GRID_COLOR,
            linewidth=1.0, alpha=1.0, zorder=1)

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dif_alg_cmp.pdf'))
    plt.show()


# --- 4. 主程序 ---

if __name__ == '__main__':
    all_plot_data = []

    # 遍历所有算法，加载并处理数据
    for algo_name, file_template in ALGORITHM_TEMPLATES.items():
        print(f"\n正在处理 {algo_name} 算法的数据...")
        episodes, mean_rewards, std_rewards = load_and_process_rewards(
            file_template, SEEDS, column_name='reward'
        )

        if episodes is not None:
            all_plot_data.append({
                'algorithm': algo_name,
                'episodes': episodes,
                'mean': mean_rewards,
                'std': std_rewards
            })
            print(f"{algo_name}: 成功加载 {len(episodes)} 个episode的数据")
        else:
            print(f"{algo_name}: 数据加载失败")

    # 如果成功加载了数据，则进行绘图
    if all_plot_data:
        print("\n所有算法数据处理完毕，开始绘制对比图...")
        plot_algorithm_comparison(all_plot_data)
    else:
        print("\n未能加载任何有效数据，无法生成图像。请检查您的文件名和路径。")
        print("\n请确保以下文件存在:")
        for algo_name, template in ALGORITHM_TEMPLATES.items():
            for seed in SEEDS:
                print(f"  - {template.format(seed)}")