"""MHBPPO 外层 MADRL 超参数。

统一训练参数入口；支持读取保存的 hyperparameters.json，命令行显式值优先。
默认复用初始化场景；batch size、熵项和学习率下限可配置。
"""

import argparse
import json
import math
from pathlib import Path


def validate_parallel_settings(args):
    """Validate merged rollout settings; absent new fields keep legacy callers usable."""
    nonnegative = ("seed", "scene_seed_offset", "worker_num_threads", "mosek_num_threads")
    defaults = dict(seed=42, scene_seed_offset=0, worker_num_threads=0,
                    mosek_num_threads=0, num_samplers=1)
    for key in nonnegative + ("num_samplers",):
        value = getattr(args, key, defaults[key])
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(key + " must be an integer")
        if value < (1 if key == "num_samplers" else 0):
            raise ValueError(key + (" must be positive" if key == "num_samplers"
                                    else " must be nonnegative"))
    if getattr(args, "seed", 42) > 2**32 - 1:
        raise ValueError("seed must lie in [0, 2**32 - 1] for NumPy's learner RNG")
    if (not getattr(args, "sync_training", True)
            and getattr(args, "seed", 42) + getattr(args, "num_samplers", 1) > 2**32 - 1):
        raise ValueError("asynchronous worker seed + offset must not exceed 2**32 - 1")
    if (getattr(args, "sync_training", True)
            and getattr(args, "num_samplers", 1) > 1):
        randomize = ("randomize_layout_per_episode", "randomize_cu_traj_per_episode",
                     "randomize_nlos_per_episode")
        conflicts = [key for key in randomize if getattr(args, key, False)]
        if conflicts:
            raise ValueError("synchronous multi-env training requires a fixed shared scene; "
                             "disable " + ", ".join(conflicts))


def get_madrl_args(argv=None):
    madrl_parser = argparse.ArgumentParser(description="MADRL 超参数")
    madrl_parser.add_argument("--seed", type=int, default=42,
                              help="MHBPPO 主随机种子；仅覆盖本算法，不改变其他算法默认值")
    madrl_parser.add_argument("--scene_seed_offset", type=int, default=0,
                              help="共同外生场景种子偏移；与 worker 身份及动作随机流独立")
    madrl_parser.add_argument("--worker_num_threads", type=int, default=0,
                              help="每个采样进程的 PyTorch CPU 推理线程数；0 保留原默认设置")
    madrl_parser.add_argument("--mosek_num_threads", type=int, default=0,
                              help="每个采样进程内 MOSEK 优化器线程上限；0 使用 MOSEK 自动设置")
    madrl_parser.add_argument("--actor_lr", type=float, default=3e-4, help="Actor 学习率")
    madrl_parser.add_argument("--critic_lr", type=float, default=3e-4, help="Critic 学习率")
    madrl_parser.add_argument("--lmbda", type=float, default=0.95, help="GAE (lambda)")
    madrl_parser.add_argument("--eps", type=float, default=0.2, help="PPO 裁剪因子 (epsilon)")
    madrl_parser.add_argument("--gamma", type=float, default=0.99, help="折扣因子 (gamma)")
    madrl_parser.add_argument("--epochs", type=int, default=5, help="每次更新迭代次数")
    madrl_parser.add_argument("--total_time_slots", type=int, default=40, help="总时隙数量")
    madrl_parser.add_argument("--hidden_dim", type=int, default=128, help="隐藏层维度")
    madrl_parser.add_argument("--batch_size", type=int, default=32, help="PPO mini-batch大小；实际取不超过本次采样条数的值")
    madrl_parser.add_argument("--continuous_entropy_coef", type=float, default=0.05, help="连续动作熵系数")
    madrl_parser.add_argument("--discrete_entropy_coef", type=float, default=0.01, help="离散动作熵系数")
    madrl_parser.add_argument("--lr_min_ratio", type=float, default=0.1, help="学习率下限相对初始学习率的比例")
    madrl_parser.add_argument("--config", type=str, default="", help="读取保存的hyperparameters.json；命令行显式参数优先")
    madrl_parser.add_argument("--episodes", type=int, default=10000, help="lr 衰减总轮次 (horizon)：iteration 在该值内学习率线性衰减至初始值的 10%% 下限，之后保持下限不变")
    # NOTE - 采样-更新分离；同步模式收齐同版本的 N 个完整 episode 后更新一次。
    madrl_parser.add_argument("--sample_queue_maxsize", type=int, default=1, help="样本队列容量（同步模式下应为 1）")
    madrl_parser.add_argument("--num_samplers", type=int, default=1, help="rollout 采样进程数；同步模式单次聚合 N 个完整 episode")
    madrl_parser.add_argument("--sync_training", action=argparse.BooleanOptionalAction, default=True,
                              help="严格同步训练：全体 worker 固定同一版本采完 episode，learner 聚合更新并广播后才允许下一轮")
    # NOTE - 接收波束结构化：默认 1.0（完全使用 actor 输出的波束）。随机初始策略的波束是
    # 各向同性方向，感知 SINR 约束几乎恒不可行（可行域极薄）；调小该值会把波束向
    # 「指定目标的匹配波束」混合，实测内层 optimal 率由 ~3% 提升到 ~37%。
    # 必须与生成 BC 数据集时使用的 --structured_beam_alpha 保持一致。
    madrl_parser.add_argument("--structured_beam_alpha", type=float, default=1.0,
                              help="接收波束混合系数：0=完全匹配波束，1=完全使用 actor 输出")
    madrl_parser.add_argument("--receive_beam_encoder", choices=("shared", "dedicated"), default="shared",
                              help="shared保留原网络；dedicated使用可训练的接收波束专属编码分支")
    madrl_parser.add_argument("--policy_initialization", type=str, default="",
                              help="可选actor/critic/精确状态归一化初始化包目录；从PPO第0次更新开始，并非续训")
    madrl_parser.add_argument("--max_updates", type=int, default=0,
                              help="完成此数量PPO更新后保存并停止；0保留原来的持续训练")
    madrl_parser.add_argument("--sensing_penalty_coef", type=float, default=1.0,
                              help="无解时新增的归一化感知短缺惩罚权重，范围[0,1]；0关闭")
    madrl_parser.add_argument("--sensing_penalty_reference_factor", type=float, default=10.0,
                              help="感知所需功率达到上限此倍数时，单UAV短缺评分为0.5；须大于1")
    # NOTE - BC 暖机：用 generate_bc_dataset 产出的可行 (s, a) 演示做 actor 监督预训练
    madrl_parser.add_argument("--bc_dataset", type=str, default="",
                              help="BC 暖机数据集 .npz 路径；为空则不预热")
    madrl_parser.add_argument("--bc_epochs", type=int, default=20,
                              help="BC 监督微调轮数（过大易使 Beta 分布坍塌、丧失探索，建议 10~30）")
    madrl_parser.add_argument("--bc_batch_size", type=int, default=128, help="BC mini-batch 大小")
    madrl_parser.add_argument("--bc_lr", type=float, default=1e-4,
                              help="BC 阶段 actor 学习率（应小于 actor_lr）")
    madrl_parser.add_argument("--checkpoint_interval", type=int, default=1000, help="每隔多少次 iteration (外层更新) 保存一次完整训练 CSV 与 checkpoint")
    madrl_parser.add_argument("--log_file", type=str, default="logs/MHBPPO/mhbppo_train.log", help="训练日志基础路径（相对 src/outer；每次在父目录下建立 YYYY-MM-DD_HH-mm-ss 独立目录，文件名固定，另存 _actions.jsonl 和 _env.jsonl）")
    madrl_parser.add_argument("--env_diagnostics", action=argparse.BooleanOptionalAction, default=True,
                              help="记录独立 _env.jsonl 可行性与原始约束诊断；不改变奖励或 success")
    madrl_parser.add_argument("--diagnostic_candidate_interval", type=int, default=1,
                              help="每隔多少个时隙检查候选 CU 感知匹配；原动作裕量/返回解每步检查")
    madrl_parser.add_argument("--diagnostic_snapshot_limit", type=int, default=1,
                              help="每个 episode 最多保存多少个返回解未通过验证的完整数值快照；0=不保存")
    # NOTE - 逐步 JSONL 详细日志按窗口滚动保留，避免长训练无限占用磁盘；两个 writer 同步换段。
    # 最早保留记录与当前 iteration 的差距不超过 (log_rotate_backup_segments + 1) * log_rotate_iterations。
    madrl_parser.add_argument("--log_rotate_iterations", type=int, default=1000,
                              help="每隔多少次 iteration 归档一次 _actions.jsonl/_env.jsonl 并开始新段；0 关闭该触发")
    madrl_parser.add_argument("--log_rotate_max_bytes", type=int, default=2 * 1024 ** 3,
                              help="单个 JSONL 段达到该字节数即归档换新段（0 关闭该触发）；与轮次触发择先")
    madrl_parser.add_argument("--log_rotate_backup_segments", type=int, default=1,
                              help="除当前段外额外保留的历史归档段数；0 表示只保留当前段")
    madrl_parser.add_argument("--log_min_free_bytes", type=int, default=5 * 1024 ** 3,
                              help="输出盘可用空间低于该值时立即归档裁剪一轮并暂停逐步详细日志（根 CSV/TensorBoard 仍记录）；0 关闭")
    # NOTE - 默认固定场景：reset 复用初始布点、CU 轨迹和逐时隙 NLoS 分量。
    # 任一随机化开关开启时重生成整套场景（由 SeedManager 确定性派生，可复现）。
    madrl_parser.add_argument("--randomize_layout_per_episode", action=argparse.BooleanOptionalAction, default=False,
                              help="默认复用 UAV/CU/目标初始布点；开启后逐 episode 重生成整套场景")
    madrl_parser.add_argument("--randomize_cu_traj_per_episode", action=argparse.BooleanOptionalAction, default=False,
                              help="默认复用 CU 马尔可夫轨迹；开启后逐 episode 重生成整套场景")
    madrl_parser.add_argument("--randomize_nlos_per_episode", action=argparse.BooleanOptionalAction, default=False,
                              help="默认复用逐时隙 NLoS 分量；开启后逐 episode 重生成整套场景")
    madrl_parser.add_argument("--debug_topology", action="store_true", default=False,
                              help="debug 模式：打印扇区调度表等场景细节（正常训练关闭，避免逐 episode 刷屏）")
    # Load defaults first; normal argparse parsing then applies explicit CLI overrides.
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--config", default="")
    selected, _ = preliminary.parse_known_args(argv)
    if selected.config:
        try:
            with Path(selected.config).open(encoding="utf-8-sig") as stream:
                payload = json.load(stream)
            settings = payload.get("hyperparameters", payload) if isinstance(payload, dict) else None
            if not isinstance(settings, dict):
                raise ValueError("JSON must be an object or contain a hyperparameters object")
            actions = {action.dest: action for action in madrl_parser._actions
                       if action.dest not in ("help", "config")}
            unknown = set(settings) - set(actions)
            if unknown:
                raise ValueError("Unknown hyperparameters: " + ", ".join(sorted(unknown)))
            defaults = {}
            for key, value in settings.items():
                action = actions[key]
                if isinstance(action, (argparse.BooleanOptionalAction, argparse._StoreTrueAction)):
                    if not isinstance(value, bool):
                        raise ValueError(key + " must be a JSON boolean")
                elif action.type is int:
                    if isinstance(value, bool) or not isinstance(value, int):
                        raise ValueError(key + " must be a JSON integer")
                elif action.type is float:
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                        raise ValueError(key + " must be a finite JSON number")
                    value = float(value)
                elif not isinstance(value, str):
                    raise ValueError(key + " must be a JSON string")
                defaults[key] = value
            madrl_parser.set_defaults(**defaults)
        except (OSError, ValueError, TypeError) as exc:
            madrl_parser.error("Cannot read --config: " + str(exc))
    args = madrl_parser.parse_args(argv)
    try:
        validate_parallel_settings(args)
    except ValueError as exc:
        madrl_parser.error(str(exc))
    for key in ("batch_size", "hidden_dim", "epochs", "episodes"):
        if getattr(args, key) <= 0:
            madrl_parser.error(key + " must be positive")
    for key in ("actor_lr", "critic_lr"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            madrl_parser.error(key + " must be positive and finite")
    for key in ("continuous_entropy_coef", "discrete_entropy_coef"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) < 0:
            madrl_parser.error(key + " must be nonnegative and finite")
    if not math.isfinite(args.lr_min_ratio) or not 0 < args.lr_min_ratio <= 1:
        madrl_parser.error("lr_min_ratio must lie in (0, 1]")
    if args.receive_beam_encoder not in ("shared", "dedicated"):
        madrl_parser.error("receive_beam_encoder must be shared or dedicated")
    if args.max_updates < 0:
        madrl_parser.error("max_updates must be nonnegative")
    if args.policy_initialization and args.bc_dataset:
        madrl_parser.error("policy_initialization and bc_dataset are mutually exclusive")
    if args.policy_initialization and not args.sync_training:
        madrl_parser.error("policy_initialization requires synchronous shared-scene sampling")
    if args.receive_beam_encoder == "dedicated" and args.structured_beam_alpha != 1.0:
        madrl_parser.error("dedicated receive_beam_encoder requires structured_beam_alpha=1")
    if not math.isfinite(args.sensing_penalty_coef) or not 0 <= args.sensing_penalty_coef <= 1:
        madrl_parser.error("sensing_penalty_coef must lie in [0, 1] and be finite")
    if (not math.isfinite(args.sensing_penalty_reference_factor)
            or args.sensing_penalty_reference_factor <= 1):
        madrl_parser.error("sensing_penalty_reference_factor must be greater than 1 and finite")
    return args
