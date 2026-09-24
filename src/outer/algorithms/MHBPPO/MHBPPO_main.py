"""MHBPPO 异步训练入口（学习进程 / 主进程）。

IMPALA 式 actor-learner 架构：
- 采样进程（algorithms/MHBPPO/MHBPPO_sampler.sampler_worker）持续 rollout，
  episode 结果放入有界 sample_queue（满则阻塞 => staleness 上限 = maxsize）。
- 本进程（learner）消费队列执行 PPO 更新：iteration = 外层更新次数，
  每轮更新后向 weight_queue 广播最新 actor 权重（drain-then-put），采样进程在
  episode 边界加载，下一个 episode 即使用最新参数。
- 无固定终止条件：训练持续运行，每 checkpoint_interval 次 iteration 保存一次
  "0~当前全部"的训练 CSV 与 actor/critic checkpoint；Ctrl+C / SIGTERM 优雅退出
  并保存最终结果。

日志约定：采样进程全程静默，所有控制台日志 / tqdm / TensorBoard 均由本进程输出。
"""

import logging
import os
import queue as queue_module
import random
import signal
import sys
import time
from datetime import datetime

import matplotlib
matplotlib.use("Agg")  # 服务器无显示环境兼容，须在 pyplot 导入前设置
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import multiprocessing as mp

# 自动向上定位 outer 根目录（含 algorithms/ 的那一层）
_OUTER_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_OUTER_ROOT, "algorithms")):
    _parent_root = os.path.dirname(_OUTER_ROOT)
    if _parent_root == _OUTER_ROOT:
        break
    _OUTER_ROOT = _parent_root
if _OUTER_ROOT not in sys.path:
    sys.path.insert(0, _OUTER_ROOT)

from algorithms.MHBPPO.MHBPPO_agent import HPPO
from algorithms.MHBPPO.MHBPPO_sampler import sampler_worker
from configs.base_params import get_base_args
from configs.madrl_args import get_madrl_args
from environment.my_env import MyEnv


def setSeed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_tensorboard_writer(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    return SummaryWriter(log_dir=log_dir)


def setup_logger(log_path):
    """实时日志：同时输出到控制台与固定 .log 文件。

    - logging 的 FileHandler 在每条记录 emit 后立即 flush，因此 `tail -f` 可实时看到
      最新日志，无需缓冲到进程结束。
    - tqdm 进度条默认走 stderr，不进入日志文件，避免 `\\r` 刷新刷爆文件。
    """
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("mhbppo")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)
    return logger


def safe_qsize(q):
    """读取队列当前元素个数。不支持 qsize 的平台（如 macOS）降级为 -1，不影响训练。"""
    try:
        return q.qsize()
    except (NotImplementedError, OSError):
        return -1


_TRANSITION_KEYS = (
    "states", "continuous_actions", "discrete_actions", "next_states",
    "rewards", "old_cont_log_probs", "old_disc_log_probs", "dones", "real_dones",
)


def validate_sample(item):
    """校验从队列取出元素的自洽性，防止训练错位。返回该 episode 的 transition 条数。

    错位风险主要来自各字段长度不一致（会导致 PPO 用错位的 old_log_probs / next_states
    参与 ratio 与 GAE 计算），因此在消费前强制检查 9 个字段等长且非空。
    """
    transition_dict = item.get("transition_dict", {})
    missing = [k for k in _TRANSITION_KEYS if k not in transition_dict]
    if missing:
        raise RuntimeError(f"采样数据缺少字段: {missing}")
    lengths = {k: len(transition_dict[k]) for k in _TRANSITION_KEYS}
    if len(set(lengths.values())) != 1:
        raise RuntimeError(f"采样数据错位，各字段长度不一致: {lengths}")
    if lengths["states"] == 0:
        raise RuntimeError("采样到空 episode（0 条 transition）")
    return lengths["states"]


def actor_checksum(actor):
    """actor 参数的确定性标量指纹（|param| 之和）。

    用于探针验证：采样端加载权重后会用**自己的 actor** 计算同一指纹并随样本回传，
    学习端与之比对，即可证明"外层网络确实更新到了内层采样进程"，而不是只送到队列没生效。
    """
    total = 0.0
    with torch.no_grad():
        for param in actor.parameters():
            total += float(param.detach().abs().sum().cpu().item())
    return round(total, 6)


def broadcast_weights(weight_queues, agent, version=-1):
    """drain-then-put：清空每条权重队列后放入最新 actor 权重（已转 CPU）。

    多采样进程架构下必须**每个 sampler 一条独立的 weight_queue**（各自 maxsize=1）：
    若共用一个队列，drain-then-put 会让多个 sampler 争抢同一份权重 —— 先抢到的取走、
    其余 sampler 在自己那轮 drain 里拿到空队列，从而永远停留在旧参数上（饥饿）。
    每条队列独立后，各 sampler 均能在自己的 episode 边界取到最新版本。

    Args:
        weight_queues: list，每个采样进程一条 maxsize=1 的队列。
        version: 本次广播对应的参数版本号（= 已完成的外层更新次数）。

    Returns:
        本次广播权重对应的指纹，供学习端按版本号留存、与采样端回传值比对。
    """
    checksum = actor_checksum(agent.actor)
    payload = {
        "actor_state_dict": {k: v.detach().cpu().clone() for k, v in agent.actor.state_dict().items()},
        "weights_version": int(version),
        "weights_checksum": checksum,
    }
    for weight_queue in weight_queues:
        while True:
            try:
                weight_queue.get_nowait()
            except queue_module.Empty:
                break
        weight_queue.put(payload)
    return checksum


def save_artifacts(output_dir, seed, iteration, reward_res, obj_fun_res, completion_rate_res, agent, final=False):
    """保存训练产物：完整历史 CSV（0~当前）+ actor/critic checkpoint。

    周期保存（final=False）带 iteration 编号，最终保存（final=True）使用 final 后缀。
    """
    os.makedirs(output_dir, exist_ok=True)

    episodes_list = np.arange(len(reward_res))
    df = pd.DataFrame({
        "episode": episodes_list,                      # 被消费的第 n 个 episode（与 iteration 一一对应）
        "reward": np.array(reward_res),
        "obj": np.array(obj_fun_res),
        "completion_rate": np.array(completion_rate_res),
    })
    csv_path = os.path.join(output_dir, f"training_rewards_seed_{seed}.csv")
    df.to_csv(csv_path, index=False)

    suffix = "final" if final else f"iter_{iteration}"
    actor_path = os.path.join(output_dir, f"mhbppo_actor_{suffix}.pth")
    critic_path = os.path.join(output_dir, f"mhbppo_critic_{suffix}.pth")
    torch.save(agent.actor.state_dict(), actor_path)
    torch.save(agent.critic.state_dict(), critic_path)
    return csv_path, actor_path, critic_path


def save_reward_plot(output_dir, reward_res):
    """保存奖励曲线 PNG（服务器无显示环境也可用，替代原 plt.show 阻塞行为）。"""
    try:
        reward_array = np.array(reward_res)
        plt.figure(figsize=(8, 5))
        plt.plot(np.arange(reward_array.shape[0]), reward_array)
        plt.xlabel("Iteration (outer updates)")
        plt.ylabel("Reward")
        plt.title("Single-Agent Beta-PPO training performance (async)")
        plot_path = os.path.join(output_dir, "training_reward_curve.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(plt.gcf())
        return plot_path
    except Exception as exc:  # 绘图失败不应影响训练产物保存
        logging.getLogger("mhbppo").warning(f"[warn] 奖励曲线保存失败: {exc}")
        return None


if __name__ == "__main__":
    current_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    writer = get_tensorboard_writer(log_dir=os.path.join(_OUTER_ROOT, f"runs/beta-hppo/{current_time_str}_beta-hppo_async_result"))
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    base_args = get_base_args()
    madrl_args = get_madrl_args()
    setSeed(seed=base_args.seed)

    # env 仅用于获取观测/动作空间维度（采样在子进程内自建 env）
    env = MyEnv(base_args=base_args, madrl_args=madrl_args)
    state_dim_bs = env.observation_space["bs"].shape[0]
    bs_continuous_action_splits = env.bs_continuous_action_splits
    bs_continuous_action_space = env.action_space["bs"]["continuous"]
    bs_discrete_action_space = env.action_space["bs"]["discrete"]
    del env  # 主进程不需要环境实例，尽早释放

    agent_bs = HPPO(
        state_dim=state_dim_bs,
        hidden_dim=madrl_args.hidden_dim,
        continuous_action_splits=bs_continuous_action_splits,
        discrete_action_dims=bs_discrete_action_space.nvec,
        action_low=bs_continuous_action_space.low,
        action_high=bs_continuous_action_space.high,
        actor_lr=madrl_args.actor_lr,
        critic_lr=madrl_args.critic_lr,
        lmbda=madrl_args.lmbda,
        eps=madrl_args.eps,
        gamma=madrl_args.gamma,
        epochs=madrl_args.epochs,
        num_episodes=madrl_args.episodes,  # lr 衰减 horizon
        device=device,
        entropy_coef=0.01,
    )

    output_dir = os.path.join(_OUTER_ROOT, f"{current_time_str}_mhbppo_seed_{base_args.seed}")
    checkpoint_interval = max(1, int(madrl_args.checkpoint_interval))
    queue_maxsize = max(1, int(madrl_args.sample_queue_maxsize))
    num_samplers = max(1, int(madrl_args.num_samplers))

    # ── 异步架构：队列 + 停止事件 + 采样进程 ──────────────────────────────
    ctx = mp.get_context("spawn")  # Linux 服务器与 Windows 行为一致，且 CUDA 安全
    sample_queue = ctx.Queue(maxsize=queue_maxsize)
    stop_event = ctx.Event()

    # 多采样进程：每个 sampler 一条独立 weight_queue，互不争抢（详见 broadcast_weights 文档）
    weight_queues = [ctx.Queue(maxsize=1) for _ in range(num_samplers)]

    log_path = os.path.join(_OUTER_ROOT, madrl_args.log_file)
    logger = setup_logger(log_path)

    # 关键契约：启动采样进程前先下发初始 actor 权重，
    # 保证各采样端与学习端从同一份初始参数出发（PPO ratio 语义要求）。
    checksum_by_version = {}  # version -> 该版本广播时的 actor 指纹，用于与采样端回传值比对
    checksum_by_version[0] = broadcast_weights(weight_queues, agent_bs, version=0)
    logger.info(f"[learner] 初始 actor 权重已下发 {num_samplers} 条 weight_queue"
                f"（采样端与学习端同源初始化, version=0, ck={checksum_by_version[0]}）")

    sampler_procs = []
    for worker_id in range(num_samplers):
        proc = ctx.Process(
            target=sampler_worker,
            # seed_offset = worker_id + 1：各采样进程使用互不相同的 RNG 流，避免产出同分布副本
            args=(base_args, madrl_args, sample_queue, weight_queues[worker_id], stop_event, worker_id + 1),
            daemon=True,
            name=f"sampler-{worker_id}",
        )
        proc.start()
        sampler_procs.append(proc)

    cpu_count = os.cpu_count() or 1
    if num_samplers > cpu_count:
        logger.warning(f"[learner] num_samplers={num_samplers} 超过 CPU 逻辑核数 {cpu_count}，"
                       f"采样进程将争抢 CPU，建议调小")
    if queue_maxsize < num_samplers:
        logger.warning(f"[learner] sample_queue_maxsize={queue_maxsize} < num_samplers={num_samplers}，"
                       f"采样进程会频繁阻塞在 put 上，建议设 >= num_samplers")

    logger.info(f"[learner] device={device} | num_samplers={num_samplers} | sample_queue_maxsize={queue_maxsize} | "
                f"checkpoint_interval={checkpoint_interval} | lr_decay_horizon={madrl_args.episodes} | cpu={cpu_count}")
    logger.info(f"[learner] 输出目录: {output_dir}")
    logger.info(f"[learner] 实时日志: {log_path}")

    # 信号 -> KeyboardInterrupt：SIGTERM 覆盖 Linux 服务器 kill；SIGBREAK 覆盖 Windows Ctrl+Break
    # （Windows 的 Ctrl+C 本身即触发 KeyboardInterrupt）
    def _sigterm_handler(signum, frame):
        raise KeyboardInterrupt
    for _sig_name in ("SIGTERM", "SIGBREAK"):
        _sig = getattr(signal, _sig_name, None)
        if _sig is None:
            continue
        try:
            signal.signal(_sig, _sigterm_handler)
        except (ValueError, OSError, RuntimeError):
            pass  # 非主线程或不支持的平台

    reward_res = []
    obj_fun_res = []
    completion_rate_res = []

    iteration = 0  # 外层更新次数（每消费一个 episode 并完成一次 update，+1）
    interrupted = False
    dead_reported = set()  # 已告警过的采样进程名，避免每轮空队列轮询重复刷屏
    weights_sync_warned = False  # 权重同步故障只告警一次（该类故障表现为内层恒不可行，必须显眼提示）

    pbar = tqdm(total=None, desc="Training Progress", unit="update", dynamic_ncols=True)
    try:
        while True:
            # 带超时轮询：保证 Ctrl+C / SIGTERM 能及时打断空队列等待
            try:
                item = sample_queue.get(timeout=1.0)
            except queue_module.Empty:
                if not any(proc.is_alive() for proc in sampler_procs):
                    # 全部采样进程已死且未发哨兵：最后再确认一次队列（可能还有在途样本）
                    try:
                        item = sample_queue.get_nowait()
                    except queue_module.Empty:
                        # 未发 error 哨兵即退出 => 多为组信号（Ctrl+Break/SIGKILL）或被系统 kill，
                        # 优雅收尾保存产物；真正的采样端异常会走下方 error 哨兵路径
                        logger.info("[learner] 全部采样进程已退出（未上报错误，可能被信号终止），优雅结束训练。")
                        interrupted = True
                        break
                else:
                    # 部分采样进程提前退出：仅告警一次（其余仍在产出，训练继续但吞吐下降）
                    for proc in sampler_procs:
                        if not proc.is_alive() and proc.name not in dead_reported:
                            dead_reported.add(proc.name)
                            logger.warning(
                                f"[learner] 采样进程 {proc.name} 已提前退出 (exitcode={proc.exitcode})，"
                                f"剩余 {sum(1 for p in sampler_procs if p.is_alive())}/{num_samplers} 个仍在采样，吞吐将下降"
                            )
                    continue

            # 采样进程异常哨兵
            if isinstance(item, dict) and "error" in item:
                raise RuntimeError(f"采样进程异常:\n{item['error']}")

            # ── 消费：一次 update = 一次 iteration ──
            # 消费前校验该 episode 的 9 个字段等长自洽，避免错位数据参与 PPO 更新
            n_transitions = validate_sample(item)

            update_start = time.time()
            agent_bs.update(item["transition_dict"], iteration, writer, agent_name="BS")
            update_cost = time.time() - update_start
            iteration += 1

            # 更新后立即向所有采样进程广播最新 actor 权重，各自下一个 episode 即使用新参数
            checksum_by_version[iteration] = broadcast_weights(weight_queues, agent_bs, version=iteration)
            # 只保留最近若干版本的指纹，避免长时训练下字典无限增长
            if len(checksum_by_version) > 512:
                for stale_version in sorted(checksum_by_version)[:-512]:
                    checksum_by_version.pop(stale_version, None)

            # ── 探针：验证外层网络确实更新到了采样进程 ──
            # wv   = 该 episode 采样时使用的参数版本（-1 表示采样端从未收到权重）
            # stale= 消费时相对该样本所用参数已发生的更新次数（1 为单采样进程的理论最优）
            # ck   = 采样端 actor 指纹 vs 学习端广播指纹是否一致（ok 即证明权重真正生效）
            used_version = item.get("weights_version", -1)
            used_checksum = item.get("weights_checksum")
            expected_checksum = checksum_by_version.get(used_version)
            checksum_ok = (
                expected_checksum is not None and used_checksum is not None
                and abs(expected_checksum - used_checksum) < 1e-3
            )
            stale = iteration - used_version if used_version >= 0 else -1

            if not weights_sync_warned:
                if used_version < 0:
                    weights_sync_warned = True
                    logger.error("[learner] 权重同步故障：采样端从未收到权重（weights_version<0），"
                                 "将一直使用随机初始策略 —— 症状与'内层不可行'完全一致，须先排除此项")
                elif used_version == 0 and iteration > 50:
                    weights_sync_warned = True
                    logger.error("[learner] 权重同步故障：采样端始终停留在初始权重 v0，"
                                 "后续广播未送达 —— 症状与'内层不可行'完全一致，须先排除此项")

            # ── 1. 网络参数更新日志： iteration / transition 数 / lr / 更新耗时 / 该 episode 指标 ──
            lr_actor = agent_bs.actor_optimizer.param_groups[0]["lr"]
            lr_critic = agent_bs.critic_optimizer.param_groups[0]["lr"]
            logger.info(
                f"iter={iteration} | 参数更新 | transitions={n_transitions} | cost={update_cost:.3f}s "
                f"| lr_actor={lr_actor:.3e} lr_critic={lr_critic:.3e} "
                f"| wv={used_version} stale={stale} ck={'ok' if checksum_ok else 'MISMATCH'}"
                f"| reward={item['avg_total_reward']:.3f} obj={item['avg_obj_fun']:.3f} "
                f"completion={item['completion_rate']:.2f}%"
            )
            # ── 2. 队列个数日志：样本队列水位 / 有多少条权重队列积压待取（-1 表示平台不支持 qsize） ──
            pending_weights = sum(1 for q in weight_queues if safe_qsize(q) > 0)
            logger.info(
                f"iter={iteration} | 队列 | sample_queue={safe_qsize(sample_queue)}/{queue_maxsize} "
                f"| weight_queues_pending={pending_weights}/{num_samplers}"
            )

            # 统计与日志（x 轴统一为 iteration）
            reward_res.append(item["avg_total_reward"])
            obj_fun_res.append(item["avg_obj_fun"])
            completion_rate_res.append(item["completion_rate"])

            writer.add_scalar("Reward/iteration", item["avg_total_reward"], iteration)
            writer.add_scalar("Obj/iteration", item["avg_obj_fun"], iteration)
            writer.add_scalar("Completion Rate/iteration", item["completion_rate"], iteration)

            pbar.set_postfix({"iter": iteration, "avg_reward": f"{item['avg_total_reward']:.3f}"})
            pbar.update(1)

            # ── 周期保存：每 checkpoint_interval 次 iteration 保存 0~当前 全部产物 ──
            if iteration % checkpoint_interval == 0:
                csv_path, actor_path, _ = save_artifacts(
                    output_dir, base_args.seed, iteration,
                    reward_res, obj_fun_res, completion_rate_res, agent_bs,
                )
                logger.info(f"iter={iteration} | 已保存 {len(reward_res)} episodes CSV ({csv_path}) + checkpoint ({actor_path})")

    except KeyboardInterrupt:
        interrupted = True
        logger.info("[learner] 收到中断信号，正在保存最终结果并停止采样进程...")
    except Exception as exc:
        # 异常（含采样端 error 哨兵抛出的 RuntimeError）落盘后再抛出，便于事后排查
        logger.exception(f"[learner] 训练异常终止: {exc}")
        raise
    finally:
        # 优雅退出：通知采样进程 -> 等待 -> 兜底 terminate
        stop_event.set()
        for proc in sampler_procs:
            proc.join(timeout=120)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=10)

        if iteration > 0:
            csv_path, actor_path, _ = save_artifacts(
                output_dir, base_args.seed, iteration,
                reward_res, obj_fun_res, completion_rate_res, agent_bs, final=True,
            )
            plot_path = save_reward_plot(output_dir, reward_res)
            logger.info(f"[learner] 最终结果已保存: iter={iteration}, best_avg_reward={max(reward_res):.4f}")
            logger.info(f"[learner] CSV: {csv_path} | checkpoint: {actor_path}" + (f" | 曲线: {plot_path}" if plot_path else ""))
        else:
            logger.info("[learner] 未完成任何 iteration，无产物保存。")

        writer.close()
        pbar.close()
        sample_queue.close()
        for weight_queue in weight_queues:
            weight_queue.close()
        # 仅在"非主动中断"时判定为故障：Ctrl+C / Ctrl+Break 会投递给整个进程组，
        # 采样进程同样收到信号并以 STATUS_CONTROL_C_EXIT (0xC000013A) 退出，属预期而非异常。
        if not interrupted:
            failed = [(p.name, p.exitcode) for p in sampler_procs if p.exitcode not in (0, None)]
            if failed:
                logger.error(f"[learner] 采样进程异常退出: {failed}")
                raise RuntimeError(f"采样进程异常退出: {failed}")
