import os
import importlib.util
import argparse
from contextlib import contextmanager
from datetime import datetime

import numpy as np
import pandas as pd
import torch

import os as _os, sys as _sys

# 自动向上定位 outer 根目录（含 algorithms/ 的那一层）
_OUTER_ROOT = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.isdir(_os.path.join(_OUTER_ROOT, "algorithms")):
    _parent_root = _os.path.dirname(_OUTER_ROOT)
    if _parent_root == _OUTER_ROOT:
        break
    _OUTER_ROOT = _parent_root
if _OUTER_ROOT not in _sys.path:
    _sys.path.insert(0, _OUTER_ROOT)

from algorithms.MHSAC.MHSAC_agent import HSAC
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
    hsac_parser.add_argument("--minimal_size", type=int, default=0, help="Minimum samples before updates")
    hsac_parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    hsac_parser.add_argument("--updates_per_step", type=int, default=1, help="Number of updates per environment step")
    hsac_parser.add_argument("--alpha_continuous", type=float, default=0.2, help="Continuous entropy temperature")
    hsac_parser.add_argument("--alpha_discrete", type=float, default=0.2, help="Discrete entropy temperature")
    hsac_parser.add_argument("--target_continuous_log_prob", type=float, default=0.3, help="Target continuous log-probability")
    hsac_parser.add_argument("--target_discrete_entropy_ratio", type=float, default=0.48, help="Target discrete entropy ratio")
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
    if int(madrl_args.minimal_size) <= 0:
        madrl_args.minimal_size = int(max(4096, 200 * madrl_args.total_time_slots))
    return base_args, madrl_args


if __name__ == "__main__":
    current_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    base_args, madrl_args = get_all_args()
    test_seed = 42

    checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoint", "checkpoint_hsac")
    actor_path = os.path.join(checkpoint_dir, "hsac_actor.pth")
    critic1_path = os.path.join(checkpoint_dir, "hsac_critic1.pth")
    critic2_path = os.path.join(checkpoint_dir, "hsac_critic2.pth")

    task_values = [100e3, 170e3, 240e3, 310e3]
    all_results = []

    for task_size in task_values:
        setSeed(seed=test_seed)
        base_args.seed = test_seed

        env = MyEnv(base_args=base_args, madrl_args=madrl_args)
        env.cus_entertaining_task_size = np.ones(base_args.cus_num) * task_size

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
            target_continuous_log_prob=madrl_args.target_continuous_log_prob,
            target_discrete_entropy_ratio=madrl_args.target_discrete_entropy_ratio,
            alpha_lr=madrl_args.alpha_lr,
            alpha_min=madrl_args.alpha_min,
            alpha_max=madrl_args.alpha_max,
        )

        agent_bs.actor.load_state_dict(torch.load(actor_path, map_location=device))
        agent_bs.critic1.load_state_dict(torch.load(critic1_path, map_location=device))
        agent_bs.critic2.load_state_dict(torch.load(critic2_path, map_location=device))
        agent_bs.actor.eval()
        agent_bs.critic1.eval()
        agent_bs.critic2.eval()

        running_norm_bs = Normalization(state_dim_bs)
        reward_scaler_bs = RewardScaling(shape=1, gamma=madrl_args.gamma)

        # warm-up episode
        for warm_idx in range(1):
            torch.manual_seed(test_seed)
            s_warm = env.reset()
            state_bs_norm_warm = running_norm_bs(np.array(s_warm["bs"], dtype=np.float32))
            terminal_warm = False
            while not terminal_warm:
                action_bs_warm = agent_bs.choose_action(state_bs_norm_warm, evaluate=True)
                next_s_warm, _, _, done_warm, _, _ = env.step(
                    {"bs": {"continuous": action_bs_warm["continuous"], "discrete": action_bs_warm["discrete"]}},
                    i_episode=test_seed
                )
                next_state_bs_norm_warm = running_norm_bs(np.array(next_s_warm["bs"], dtype=np.float32))
                s_warm = next_s_warm
                state_bs_norm_warm = next_state_bs_norm_warm
                terminal_warm = done_warm

        # test episode
        episode_rewards_total = []
        obj_fun_total = []
        uav_sen_power_total = []
        uav_off_power_total = []
        bs_freq_total = []
        cu_off_power_total = []
        bs_cu_freq_total = []
        success_slots = 0

        s = env.reset()
        state_bs_norm = running_norm_bs(np.array(s["bs"], dtype=np.float32), update=False)
        terminal = False
        reward_scaler_bs.reset()

        while not terminal:
            action_bs = agent_bs.choose_action(state_bs_norm, evaluate=True)

            next_s, total_reward, r_dict, done, obj_fun, success_flag = env.step(
                {"bs": {"continuous": action_bs["continuous"], "discrete": action_bs["discrete"]}},
                i_episode=test_seed
            )

            success_slots += success_flag
            episode_rewards_total.append(float(total_reward))
            if np.isfinite(obj_fun):
                obj_fun_total.append(float(obj_fun))
            comps = r_dict.get("components", {})
            if np.isfinite(comps.get("avg_uav_sen_power", float("nan"))):
                uav_sen_power_total.append(comps["avg_uav_sen_power"])
                uav_off_power_total.append(comps["avg_uav_off_power"])
                bs_freq_total.append(comps["avg_bs_freq"])
                cu_off_power_total.append(comps["avg_cu_off_power"])
                bs_cu_freq_total.append(comps["avg_bs_cu_freq"])

            next_state_bs_norm = running_norm_bs(np.array(next_s["bs"], dtype=np.float32), update=False)
            s = next_s
            state_bs_norm = next_state_bs_norm
            terminal = done

        all_results.append({
            "cu_task_size": task_size,
            "run_id": 0,
            "avg_total_reward": float(np.mean(episode_rewards_total)),
            "avg_obj_fun": float(np.mean(obj_fun_total)) if obj_fun_total else float("nan"),
            "completion_rate": (success_slots / madrl_args.total_time_slots) * 100.0,
            "avg_uav_sen_power": float(np.mean(uav_sen_power_total)) if uav_sen_power_total else float("nan"),
            "avg_uav_off_power": float(np.mean(uav_off_power_total)) if uav_off_power_total else float("nan"),
            "avg_bs_freq": float(np.mean(bs_freq_total)) if bs_freq_total else float("nan"),
            "avg_cu_off_power": float(np.mean(cu_off_power_total)) if cu_off_power_total else float("nan"),
            "avg_bs_cu_freq": float(np.mean(bs_cu_freq_total)) if bs_cu_freq_total else float("nan"),
        })

    df = pd.DataFrame(all_results)
    filename = f"{current_time_str}_MHSAC_test_cuTaskNum_results.csv"
    df.to_csv(filename, index=False)
    print(f"Test results saved to {filename}")
    print(df.to_string(index=False))
