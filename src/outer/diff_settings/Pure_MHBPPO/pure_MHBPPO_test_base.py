import os
import importlib.util
import sys
import argparse
from contextlib import contextmanager
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

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
from utils.normalization import Normalization, RewardScaling

# ── 动态加载 PureMyEnv ──
_current_dir = os.path.dirname(os.path.abspath(__file__))
_pure_dir = os.path.join(_OUTER_ROOT, "environment")
if _pure_dir not in sys.path:
    sys.path.insert(0, _pure_dir)

_env_path = os.path.join(_pure_dir, "pure-my-env.py")
_spec_env = importlib.util.spec_from_file_location("_pure_my_env", _env_path)
_mod_env = importlib.util.module_from_spec(_spec_env)
_spec_env.loader.exec_module(_mod_env)
PureMyEnv = _mod_env.PureMyEnv


def _load_pure_beta_main_helpers():
    pure_beta_main_path = os.path.join(_OUTER_ROOT, "algorithms", "Pure_MHBPPO", "P-MHBPPO_main.py")
    spec = importlib.util.spec_from_file_location("pure_beta_hppo_main_helpers", pure_beta_main_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_pure_beta_main_helpers = _load_pure_beta_main_helpers()
setSeed = _pure_beta_main_helpers.setSeed
get_madrl_args = _pure_beta_main_helpers.get_madrl_args
get_base_args = _pure_beta_main_helpers.get_base_args


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

    checkpoint_dir = os.path.join(_current_dir, "checkpoint", "checkpoint_pure_mhbppo_new")
    actor_path = os.path.join(checkpoint_dir, "pure_beta_hppo_actor.pth")
    critic_path = os.path.join(checkpoint_dir, "pure_beta_hppo_critic.pth")

    all_results = []

    for test_seed in tqdm([test_seed], desc="Testing Seeds"):
        setSeed(seed=test_seed)
        base_args.seed = test_seed

        env = PureMyEnv(base_args=base_args, madrl_args=madrl_args)
        env.feasibility_enforce = True  # 测试层后处理：C7/C9/C10/C11/C12 违反时自动修正

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
            entropy_coef=0.01,
            continuous_dist_type="beta",
        )

        # ── 加载 checkpoint，兼容旧维度（705 → 706，多的 1 维补零）──
        _actor_state = torch.load(actor_path, map_location=device)
        _critic_state = torch.load(critic_path, map_location=device)
        for _state, _model in [(_actor_state, agent_bs.actor), (_critic_state, agent_bs.critic)]:
            for _key in list(_state.keys()):
                if _state[_key].shape != _model.state_dict()[_key].shape:
                    _old = _state[_key]
                    _new = torch.zeros(_model.state_dict()[_key].shape, dtype=_old.dtype, device=_old.device)
                    if _old.dim() == 2:
                        _new[:, :_old.shape[1]] = _old
                    elif _old.dim() == 1:
                        _new[:_old.shape[0]] = _old
                    _state[_key] = _new
        agent_bs.actor.load_state_dict(_actor_state)
        agent_bs.critic.load_state_dict(_critic_state)
        agent_bs.actor.eval()
        agent_bs.critic.eval()

        running_norm_bs = Normalization(state_dim_bs)
        reward_scaler_bs = RewardScaling(shape=1, gamma=madrl_args.gamma)

        # warm-up episodes with diverse action paths to initialize Normalization statistics
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

        num_runs = 30
        run_avg_rewards = []
        run_avg_obj_fun = []
        run_avg_bs_rewards = []
        run_completion_rates = []
        all_reward_data = []

        for run_idx in range(num_runs):
            episode_rewards_total = []
            obj_fun_total = []
            success_slots = 0
            episode_reward_bs = 0.0

            s = env.reset()
            state_bs_norm = running_norm_bs(np.array(s["bs"], dtype=np.float32), update=False)
            terminal = False
            slot_idx = 0
            reward_data = []
            reward_scaler_bs.reset()

            while not terminal:
                action_bs, _, _, _ = agent_bs.choose_action(state_bs_norm)

                next_s, total_reward, r_dict, done, obj_fun, success_flag = env.step(
                    {"bs": action_bs}, i_episode=test_seed
                )
                success_slots += success_flag

                r_bs = float(r_dict["bs"])
                r_bs_norm = float(np.asarray(reward_scaler_bs(r_bs)).item())

                episode_rewards_total.append(float(total_reward))
                obj_fun_total.append(float(obj_fun))
                episode_reward_bs += r_bs

                slot_idx += 1
                reward_data.append({
                    "slot": slot_idx,
                    "total_reward": float(total_reward),
                    "obj_fun": float(obj_fun),
                    "bs_reward": float(r_dict["bs"]),
                    "success_flag": int(success_flag),
                })

                next_state_bs_norm = running_norm_bs(np.array(next_s["bs"], dtype=np.float32), update=False)
                state_bs_norm = next_state_bs_norm
                terminal = done

            run_avg_rewards.append(np.mean(episode_rewards_total))
            run_avg_obj_fun.append(np.mean(obj_fun_total))
            run_avg_bs_rewards.append(episode_reward_bs / len(episode_rewards_total))
            run_completion_rates.append((success_slots / madrl_args.total_time_slots) * 100.0)
            all_reward_data.append(reward_data)

        avg_reward_data = []
        for slot in range(len(all_reward_data[0])):
            avg_entry = {}
            for key in ["slot", "total_reward", "obj_fun", "bs_reward", "success_flag"]:
                avg_entry[key] = float(np.mean([all_reward_data[r][slot][key] for r in range(num_runs)]))
            avg_reward_data.append(avg_entry)

        reward_df = pd.DataFrame(avg_reward_data)
        reward_filename = f"{current_time_str}_pure_MHBPPO_seed{test_seed}_rewards.csv"
        reward_df.to_csv(reward_filename, index=False)
        print(f"Rewards saved to {reward_filename}")

        for run_idx in range(num_runs):
            all_results.append({
                "run_id": run_idx,
                "avg_total_reward": run_avg_rewards[run_idx],
                "completion_rate": run_completion_rates[run_idx],
            })

    df = pd.DataFrame(all_results)
    filename = f"{current_time_str}_pure_MHBPPO_test_results.csv"
    df.to_csv(filename, index=False)
    print(f"Test results saved to {filename}")
    print(df.to_string(index=False))
