import argparse
from contextlib import contextmanager
import importlib.util
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import sys

# 自动向上定位 outer 根目录（含 algorithms/ 的那一层）
_OUTER_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_OUTER_ROOT, "algorithms")):
    _parent_root = os.path.dirname(_OUTER_ROOT)
    if _parent_root == _OUTER_ROOT:
        break
    _OUTER_ROOT = _parent_root
if _OUTER_ROOT not in sys.path:
    sys.path.insert(0, _OUTER_ROOT)

from algorithms.MHSAC.MHSAC_agent import HSAC, ReplayBuffer
from environment.my_env import MyEnv
from utils.normalization import Normalization, RewardScaling


def _load_beta_main_helpers():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    beta_main_path = os.path.join(_OUTER_ROOT, "algorithms", "MHBPPO", "MHBPPO_main.py")
    spec = importlib.util.spec_from_file_location("beta_hppo_main_helpers", beta_main_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_beta_main_helpers = _load_beta_main_helpers()
setSeed = _beta_main_helpers.setSeed
get_tensorboard_writer = _beta_main_helpers.get_tensorboard_writer
get_madrl_args = _beta_main_helpers.get_madrl_args
get_base_args = _beta_main_helpers.get_base_args


@contextmanager
def _ignore_unknown_cli_args():
    original_parse_args = argparse.ArgumentParser.parse_args

    def _parse_args_ignore_unknown(self, args=None, namespace=None):
        parsed_args, _ = self.parse_known_args(args=args, namespace=namespace)
        return parsed_args

    argparse.ArgumentParser.parse_args = _parse_args_ignore_unknown
    try:
        yield
    finally:
        argparse.ArgumentParser.parse_args = original_parse_args


def get_hsac_args():
    hsac_parser = argparse.ArgumentParser(description="HSAC unique arguments")
    hsac_parser.add_argument("--tau", type=float, default=5e-3, help="Target network soft update factor")
    hsac_parser.add_argument("--buffer_size", type=int, default=100000, help="Replay buffer capacity")
    # 20260409 - 更强的 HSAC 修改
    hsac_parser.add_argument("--minimal_size", type=int, default=0, help="Minimum samples before updates; <=0 means auto warm-up")
    hsac_parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    hsac_parser.add_argument("--updates_per_step", type=int, default=1, help="Number of updates per environment step")
    hsac_parser.add_argument("--alpha_continuous", type=float, default=0.2, help="Continuous entropy temperature")
    hsac_parser.add_argument("--alpha_discrete", type=float, default=0.2, help="Discrete entropy temperature")
    # 20260409 - 更强的 HSAC 修改
    hsac_parser.add_argument("--target_continuous_log_prob", type=float, default=0.3, help="Target continuous log-probability for automatic temperature tuning")
    hsac_parser.add_argument("--target_discrete_entropy_ratio", type=float, default=0.48, help="Target discrete entropy ratio with respect to the maximum multi-head entropy")
    # 20260409 - 更强的 HSAC 修改
    hsac_parser.add_argument("--alpha_lr", type=float, default=3e-5, help="Learning rate for automatic temperature tuning")
    hsac_parser.add_argument("--alpha_min", type=float, default=1e-4, help="Lower bound of automatic temperature")
    hsac_parser.add_argument("--alpha_max", type=float, default=5.0, help="Upper bound of automatic temperature")
    return hsac_parser.parse_args()


def get_all_args():
    with _ignore_unknown_cli_args():
        base_args = get_base_args()
        madrl_args = get_madrl_args()
        hsac_args = get_hsac_args()

    for key, value in vars(hsac_args).items():
        setattr(madrl_args, key, value)
    # 20260409 - 更强的 HSAC 修改
    if int(madrl_args.minimal_size) <= 0:
        madrl_args.minimal_size = int(max(4096, 200 * madrl_args.total_time_slots))
    return base_args, madrl_args


if __name__ == "__main__":
    current_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    writer = get_tensorboard_writer(log_dir=f"runs/hsac/{current_time_str}_hsac_result")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    base_args, madrl_args = get_all_args()
    setSeed(seed=base_args.seed)

    env = MyEnv(base_args=base_args, madrl_args=madrl_args)

    state_dim_bs = env.observation_space["bs"].shape[0]
    bs_continuous_action_space = env.action_space["bs"]["continuous"]
    bs_discrete_action_space = env.action_space["bs"]["discrete"]

    agent_bs = HSAC(
        state_dim=state_dim_bs,
        hidden_dim=madrl_args.hidden_dim,
        continuous_action_splits=env.bs_continuous_action_splits,
        discrete_action_dims=bs_discrete_action_space.nvec,
        action_low=bs_continuous_action_space.low,
        action_high=bs_continuous_action_space.high,
        actor_lr=madrl_args.actor_lr,
        critic_lr=madrl_args.critic_lr,
        gamma=madrl_args.gamma,
        tau=madrl_args.tau,
        alpha_continuous=madrl_args.alpha_continuous,
        alpha_discrete=madrl_args.alpha_discrete,
        device=device,
        # 20260409 - 更强的 HSAC 修改
        target_continuous_log_prob=madrl_args.target_continuous_log_prob,
        target_discrete_entropy_ratio=madrl_args.target_discrete_entropy_ratio,
        alpha_lr=madrl_args.alpha_lr,
        alpha_min=madrl_args.alpha_min,
        alpha_max=madrl_args.alpha_max,
    )

    replay_buffer = ReplayBuffer(madrl_args.buffer_size)
    running_norm_bs = Normalization(state_dim_bs)
    # 20260409 - 更强的 HSAC 修改
    reward_scaler_bs = RewardScaling(shape=1, gamma=madrl_args.gamma)

    reward_res = []
    all_agents_rewards = []
    max_avg_reward = -np.inf
    best_uav_trajectory = None
    cu_trajectory = None
    target_trajectory = None
    global_step = 0

    with tqdm(total=int(madrl_args.episodes), desc="Training Progress") as pbar:
        for i_episode in range(int(madrl_args.episodes)):
            episode_rewards_total = []
            episode_reward_bs = 0.0

            s = env.reset()
            # 20260409 - 更强的 HSAC 修改
            state_bs_raw = np.array(s["bs"], dtype=np.float32)
            state_bs_norm = running_norm_bs(state_bs_raw)
            terminal = False
            uav_positions_episode = []
            cu_positions_episode = []
            target_positions_episode = []
            reward_scaler_bs.reset()

            while not terminal:
                uav_positions_episode.append(env.getPosUAV())
                cu_positions_episode.append(env.getPosCU())
                target_positions_episode.append(env.getPosTarget())

                action_bs = agent_bs.choose_action(state_bs_norm)

                next_s, total_reward, r_dict, done = env.step(
                    {"bs": {"continuous": action_bs["continuous"], "discrete": action_bs["discrete"]}},
                    i_episode
                )

                r_bs = float(r_dict["bs"])
                # 20260409 - 更强的 HSAC 修改
                r_bs_norm = float(np.asarray(reward_scaler_bs(r_bs)).item())
                next_state_bs_raw = np.array(next_s["bs"], dtype=np.float32)
                next_state_bs_norm = running_norm_bs(next_state_bs_raw)

                replay_buffer.add(
                    # 20260409 - 更强的 HSAC 修改
                    state=state_bs_raw,
                    continuous_action=action_bs["continuous"],
                    discrete_index=action_bs["discrete_index"],
                    reward=r_bs_norm,
                    next_state=next_state_bs_raw,
                    done=bool(done),
                )

                if replay_buffer.size() >= max(madrl_args.minimal_size, madrl_args.batch_size):
                    for _ in range(madrl_args.updates_per_step):
                        transition_dict_bs = replay_buffer.sample(madrl_args.batch_size)
                        # 20260409 - 更强的 HSAC 修改
                        transition_dict_bs["states"] = np.asarray(
                            running_norm_bs(transition_dict_bs["states"], update=False),
                            dtype=np.float32,
                        )
                        transition_dict_bs["next_states"] = np.asarray(
                            running_norm_bs(transition_dict_bs["next_states"], update=False),
                            dtype=np.float32,
                        )
                        agent_bs.update(transition_dict_bs, writer=writer, step=global_step, agent_name="BS")

                episode_rewards_total.append(float(total_reward))
                episode_reward_bs += r_bs

                writer.add_scalar("BS/Continuous_LogProb_Action", action_bs["continuous_log_prob"], global_step)
                writer.add_scalar("BS/Selected_Discrete_LogProb_Action", action_bs["selected_discrete_log_prob"], global_step)
                writer.add_scalar("BS/Discrete_Entropy_Action", action_bs["discrete_entropy"], global_step)

                # 20260409 - 更强的 HSAC 修改
                s = next_s
                state_bs_raw = next_state_bs_raw
                state_bs_norm = next_state_bs_norm
                terminal = done
                global_step += 1

            if np.mean(episode_rewards_total) > max_avg_reward:
                max_avg_reward = np.mean(episode_rewards_total)
                if len(uav_positions_episode) > 0:
                    best_uav_trajectory = np.array(uav_positions_episode, copy=True)
                if len(cu_positions_episode) > 0:
                    cu_trajectory = np.array(cu_positions_episode, copy=True)
                if len(target_positions_episode) > 0:
                    target_trajectory = np.array(target_positions_episode, copy=True)

            avg_total_reward = np.mean(episode_rewards_total)
            avg_bs_reward = episode_reward_bs / max(len(episode_rewards_total), 1)

            reward_res.append(avg_total_reward)
            all_agents_rewards.append([avg_bs_reward])
            writer.add_scalar("Reward/episode", avg_total_reward, i_episode)

            pbar.set_postfix({"avg_reward": f"{avg_total_reward:.3f}"})
            pbar.update(1)

    writer.close()

    reward_array = np.array(reward_res)
    episodes_list = np.arange(reward_array.shape[0])
    plt.plot(episodes_list, reward_array)
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.title("Single-Agent HSAC training performance")
    plt.show()

    filename = f"{current_time_str}_HSAC_training_rewards_seed_{base_args.seed}.csv"
    df = pd.DataFrame({"episode": episodes_list, "reward": reward_array})
    df.to_csv(filename, index=False)
    print(f"HSAC training results saved to {filename}")

    pthname = f"{current_time_str}_hsac_seed_{base_args.seed}"
    os.makedirs(pthname, exist_ok=True)
    actor_path = os.path.join(pthname, "hsac_actor.pth")
    critic1_path = os.path.join(pthname, "hsac_critic1.pth")
    critic2_path = os.path.join(pthname, "hsac_critic2.pth")
    torch.save(agent_bs.actor.state_dict(), actor_path)
    torch.save(agent_bs.critic1.state_dict(), critic1_path)
    torch.save(agent_bs.critic2.state_dict(), critic2_path)
    print(f"模型权重已保存至目录 {pthname}: hsac_actor.pth, hsac_critic1.pth, hsac_critic2.pth")