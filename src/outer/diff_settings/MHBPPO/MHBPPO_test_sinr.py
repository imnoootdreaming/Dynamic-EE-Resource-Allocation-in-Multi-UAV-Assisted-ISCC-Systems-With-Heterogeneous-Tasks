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

from algorithms.MHBPPO.MHBPPO_agent import HPPO
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


def get_all_args():
    with _ignore_unknown_cli_args():
        base_args = get_base_args()
        madrl_args = get_madrl_args()
    return base_args, madrl_args


if __name__ == "__main__":
    current_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    base_args, madrl_args = get_all_args()
    base_args.uavs_num = 4
    test_seed = 42

    checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoint", "checkpoint_mhbppo")
    actor_path = os.path.join(checkpoint_dir, "mhbppo_actor.pth")
    critic_path = os.path.join(checkpoint_dir, "mhbppo_critic.pth")

    sinr_values = [10, 15, 20, 25]
    all_results = []

    for sinr in sinr_values:
        setSeed(seed=test_seed)
        base_args.seed = test_seed
        base_args.sen_sinr = sinr

        env = MyEnv(base_args=base_args, madrl_args=madrl_args)

        state_dim_bs = env.observation_space["bs"].shape[0]
        bs_continuous_action_space = env.action_space["bs"]["continuous"]
        bs_discrete_action_space = env.action_space["bs"]["discrete"]

        agent_bs = HPPO(
            state_dim=state_dim_bs,
            hidden_dim=madrl_args.hidden_dim,
            continuous_action_splits=env.bs_continuous_action_splits,
            discrete_action_dims=bs_discrete_action_space.nvec,
            action_low=bs_continuous_action_space.low,
            action_high=bs_continuous_action_space.high,
            actor_lr=madrl_args.actor_lr,
            critic_lr=madrl_args.critic_lr,
            lmbda=madrl_args.lmbda,
            eps=madrl_args.eps,
            gamma=madrl_args.gamma,
            epochs=madrl_args.epochs,
            num_episodes=madrl_args.episodes,
            device=device,
            entropy_coef=0.01
        )

        agent_bs.actor.load_state_dict(torch.load(actor_path, map_location=device))
        agent_bs.critic.load_state_dict(torch.load(critic_path, map_location=device))
        agent_bs.actor.eval()
        agent_bs.critic.eval()

        running_norm_bs = Normalization(state_dim_bs)
        reward_scaler_bs = RewardScaling(shape=1, gamma=madrl_args.gamma)

        # warm-up episode
        for warm_idx in range(1):
            torch.manual_seed(test_seed)
            s_warm = env.reset()
            terminal_warm = False
            while not terminal_warm:
                state_bs_norm_warm = running_norm_bs(np.array(s_warm["bs"], dtype=np.float32))
                action_bs_warm, _, _, _ = agent_bs.choose_action(state_bs_norm_warm)
                next_s_warm, _, _, done_warm, _, _ = env.step({"bs": action_bs_warm}, i_episode=test_seed)
                s_warm = next_s_warm
                terminal_warm = done_warm

        # test episode
        episode_rewards_total = []
        obj_fun_total = []
        flight_energy_total = []
        sensing_energy_total = []
        offloading_energy_total = []
        computation_energy_total = []
        success_slots = 0

        s = env.reset()
        state_bs_norm = running_norm_bs(np.array(s["bs"], dtype=np.float32), update=False)
        terminal = False
        reward_scaler_bs.reset()

        while not terminal:
            action_bs, _, _, _ = agent_bs.choose_action(state_bs_norm)

            next_s, total_reward, r_dict, done, obj_fun, success_flag = env.step({"bs": action_bs}, i_episode=test_seed)

            success_slots += success_flag
            episode_rewards_total.append(float(total_reward))
            if np.isfinite(obj_fun):
                obj_fun_total.append(float(obj_fun))
            comps = r_dict.get("components", {})
            if np.isfinite(comps.get("flight_energy", float("nan"))):
                flight_energy_total.append(comps["flight_energy"])
                sensing_energy_total.append(comps["sensing_energy"])
                offloading_energy_total.append(comps["offloading_energy"])
                computation_energy_total.append(comps["computation_energy"])

            next_state_bs_norm = running_norm_bs(np.array(next_s["bs"], dtype=np.float32), update=False)
            state_bs_norm = next_state_bs_norm
            terminal = done

        all_results.append({
            "sen_sinr": sinr,
            "run_id": 0,
            "avg_total_reward": float(np.mean(episode_rewards_total)),
            "avg_obj_fun": float(np.mean(obj_fun_total)) if obj_fun_total else float("nan"),
            "completion_rate": (success_slots / madrl_args.total_time_slots) * 100.0,
            "avg_flight_energy": float(np.mean(flight_energy_total)) if flight_energy_total else float("nan"),
            "avg_sensing_energy": float(np.mean(sensing_energy_total)) if sensing_energy_total else float("nan"),
            "avg_offloading_energy": float(np.mean(offloading_energy_total)) if offloading_energy_total else float("nan"),
            "avg_computation_energy": float(np.mean(computation_energy_total)) if computation_energy_total else float("nan"),
        })

    df = pd.DataFrame(all_results)
    filename = f"{current_time_str}_MHBPPO_test_sinr_results.csv"
    df.to_csv(filename, index=False)
    print(f"Test results saved to {filename}")
    print(df.to_string(index=False))
