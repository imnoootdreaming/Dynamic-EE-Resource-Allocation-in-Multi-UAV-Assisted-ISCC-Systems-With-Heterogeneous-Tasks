"""MHBPPO 外层 MADRL 超参数。

从 `algorithms/MHBPPO/MHBPPO_main.py` 中抽离 `get_madrl_args()`，参数名、默认值与
help 文案保持原样（仅搬移，不改逻辑）。
"""

import argparse


def get_madrl_args():
    madrl_parser = argparse.ArgumentParser(description="MADRL 超参数")
    madrl_parser.add_argument("--actor_lr", type=float, default=3e-4, help="Actor 学习率")
    madrl_parser.add_argument("--critic_lr", type=float, default=3e-4, help="Critic 学习率")
    madrl_parser.add_argument("--lmbda", type=float, default=0.95, help="GAE (lambda)")
    madrl_parser.add_argument("--eps", type=float, default=0.2, help="PPO 裁剪因子 (epsilon)")
    madrl_parser.add_argument("--gamma", type=float, default=0.99, help="折扣因子 (gamma)")
    madrl_parser.add_argument("--epochs", type=int, default=5, help="每次更新迭代次数")
    madrl_parser.add_argument("--total_time_slots", type=int, default=40, help="总时隙数量")
    madrl_parser.add_argument("--hidden_dim", type=int, default=128, help="隐藏层维度")
    madrl_parser.add_argument("--episodes", type=int, default=10000, help="lr 衰减总轮次 (horizon)：iteration 在该值内学习率线性衰减至初始值的 10% 下限，之后保持下限不变")
    # NOTE - 异步采样-更新分离架构（IMPALA 式多进程）相关参数
    madrl_parser.add_argument("--sample_queue_maxsize", type=int, default=4, help="样本队列容量（积压上限），多采样进程时建议 >= num_samplers")
    madrl_parser.add_argument("--num_samplers", type=int, default=4, help="并行 rollout 的采样进程数；建议 <= CPU 逻辑核数，过大会使样本 staleness 升高（约等于进程数）")
    madrl_parser.add_argument("--checkpoint_interval", type=int, default=1000, help="每隔多少次 iteration (外层更新) 保存一次完整训练 CSV 与 checkpoint")
    madrl_parser.add_argument("--log_file", type=str, default="logs/mhbppo_train.log", help="实时日志文件路径（相对 src/outer，每次运行覆盖重写）")
    # NOTE - 并行 rollout 状态多样性：逐 episode 场景重生成（种子经 SeedManager 确定性派生，可复现）
    madrl_parser.add_argument("--randomize_layout_per_episode", action=argparse.BooleanOptionalAction, default=True,
                              help="每个 episode 重新生成 UAV/CU/目标初始布点（扇区分区部署）")
    madrl_parser.add_argument("--randomize_cu_traj_per_episode", action=argparse.BooleanOptionalAction, default=True,
                              help="每个 episode 重新生成 CU 马尔可夫轨迹")
    madrl_parser.add_argument("--randomize_nlos_per_episode", action=argparse.BooleanOptionalAction, default=True,
                              help="每个 episode 重新生成 NLoS 信道高斯分量")
    madrl_parser.add_argument("--debug_topology", action="store_true", default=False,
                              help="debug 模式：打印扇区调度表等场景细节（正常训练关闭，避免逐 episode 刷屏）")
    return madrl_parser.parse_args()
