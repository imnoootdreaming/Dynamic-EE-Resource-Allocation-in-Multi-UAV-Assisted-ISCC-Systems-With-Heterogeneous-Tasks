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
    madrl_parser.add_argument("--sample_queue_maxsize", type=int, default=1, help="样本队列容量（同步模式下应为 1）")
    madrl_parser.add_argument("--num_samplers", type=int, default=1, help="rollout 采样进程数；同步模式固定使用 1")
    madrl_parser.add_argument("--sync_training", action=argparse.BooleanOptionalAction, default=True,
                              help="严格同步训练：learner 更新并广播权重后才允许 worker 采集下一个 episode")
    # NOTE - 接收波束结构化：默认 1.0（完全使用 actor 输出的波束）。随机初始策略的波束是
    # 各向同性方向，感知 SINR 约束几乎恒不可行（可行域极薄）；调小该值会把波束向
    # 「指定目标的匹配波束」混合，实测内层 optimal 率由 ~3% 提升到 ~37%。
    # 必须与生成 BC 数据集时使用的 --structured_beam_alpha 保持一致。
    madrl_parser.add_argument("--structured_beam_alpha", type=float, default=1.0,
                              help="接收波束混合系数：0=完全匹配波束，1=完全使用 actor 输出")
    # NOTE - BC 暖机：用 generate_bc_dataset 产出的可行 (s, a) 演示做 actor 监督预训练
    madrl_parser.add_argument("--bc_dataset", type=str, default="",
                              help="BC 暖机数据集 .npz 路径；为空则不预热")
    madrl_parser.add_argument("--bc_epochs", type=int, default=20,
                              help="BC 监督微调轮数（过大易使 Beta 分布坍塌、丧失探索，建议 10~30）")
    madrl_parser.add_argument("--bc_batch_size", type=int, default=128, help="BC mini-batch 大小")
    madrl_parser.add_argument("--bc_lr", type=float, default=1e-4,
                              help="BC 阶段 actor 学习率（应小于 actor_lr）")
    madrl_parser.add_argument("--checkpoint_interval", type=int, default=1000, help="每隔多少次 iteration (外层更新) 保存一次完整训练 CSV 与 checkpoint")
    madrl_parser.add_argument("--log_file", type=str, default="logs/MHBPPO/mhbppo_train.log", help="训练日志基础路径（相对 src/outer；自动添加时分秒、日期前缀，同目录另存 _actions.jsonl 动作日志）")
    madrl_parser.add_argument("--env_diagnostics", action=argparse.BooleanOptionalAction, default=True,
                              help="记录独立 _env.jsonl 可行性与原始约束诊断；不改变奖励或 success")
    madrl_parser.add_argument("--diagnostic_candidate_interval", type=int, default=1,
                              help="每隔多少个时隙检查候选 CU 感知匹配；原动作裕量/返回解每步检查")
    madrl_parser.add_argument("--diagnostic_snapshot_limit", type=int, default=1,
                              help="每个 episode 最多保存多少个返回解未通过验证的完整数值快照；0=不保存")
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
