import os
import importlib.util
import sys
import io
import argparse
from contextlib import contextmanager, redirect_stdout
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

# ── 内层 CCCP：经外层桥接模块接入 ──
# reward/cccp_bridge.py 内部把外层实时信道映射为内层 PC3P 的求解上下文（同一场景），
# 不再直接加载已删除的 inner_problem 旧算法。
_parent_dir = os.path.dirname(_current_dir)
_reward_dir = os.path.join(_OUTER_ROOT, "reward")
if _reward_dir not in _sys.path:
    _sys.path.insert(0, _reward_dir)
from cccp_bridge import solve_inner_energy as penalty_based_cccp


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


def evaluate_via_cccp(base_args, parsed, saved_channels, saved_target_matrix,
                      saved_prev_uavs_pos, cus_entertaining_task_size):
    """
    提取 PureMHBPPO 的飞行+CU 外层决策，丢进 CCCP 优化波束/频率，
    返回 CCCP 的 energy_opt（与 MHBPPO 等算法的对比基准一致）。
    """
    next_uavs_pos = parsed["next_uavs_pos"]
    off_duration_list = [float(x) for x in parsed["uav_off_duration"]]
    cu_off_power_list = [float(x) for x in parsed["cu_off_power"]]
    uavs_cus_matched = parsed["uavs_cus_matched"]

    with redirect_stdout(io.StringIO()):
        cccp_result = penalty_based_cccp(
            args=base_args,
            uavs_2_cus_channels=saved_channels["uavs_2_cus"],
            uavs_2_bs_channels=saved_channels["uavs_2_bs"],
            cus_2_bs_channels=saved_channels["cus_2_bs"],
            uavs_2_targets_channels=saved_channels["uavs_2_targets"],
            uavs_targets_matched_matrix=saved_target_matrix,
            uavs_cus_matched_matrix=uavs_cus_matched,
            uavs_pos_pre=saved_prev_uavs_pos,
            uavs_pos_cur=next_uavs_pos,
            uavs_off_duration=off_duration_list,
            cus_off_power=cu_off_power_list,
            cus_entertaining_task_size=cus_entertaining_task_size,
            return_solution=True,
        )

    energy_opt = cccp_result[0]
    solution = cccp_result[11]
    no_solution = 1 if (energy_opt == float("inf") or solution is None) else 0

    return {
        "energy_opt": float(energy_opt),
        "no_solution": no_solution,
    }


if __name__ == "__main__":
    current_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    base_args, madrl_args = get_all_args()
    base_args.uavs_num = 4
    test_seed = 42

    checkpoint_dir = os.path.join(_current_dir, "checkpoint", "checkpoint_pure_mhbppo_new")
    actor_path = os.path.join(checkpoint_dir, "pure_beta_hppo_actor.pth")
    critic_path = os.path.join(checkpoint_dir, "pure_beta_hppo_critic.pth")

    height_values = [100, 130, 160, 190, 220, 250]
    all_results = []

    for height in height_values:
        setSeed(seed=test_seed)
        base_args.seed = test_seed
        base_args.uav_height = height

        env = PureMyEnv(base_args=base_args, madrl_args=madrl_args)

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
        cccp_energy_total = []
        flight_energy_total = []
        sensing_energy_total = []
        offloading_energy_total = []
        computation_energy_total = []
        avg_uav_speed_total = []
        boundary_penalty_total = []
        collision_penalty_total = []
        spectrum_penalty_total = []
        tanh_penalty_total = []
        success_slots = 0
        cccp_success_slots = 0

        s = env.reset()
        state_bs_norm = running_norm_bs(np.array(s["bs"], dtype=np.float32), update=False)
        terminal = False
        reward_scaler_bs.reset()

        while not terminal:
            action_bs, _, _, _ = agent_bs.choose_action(state_bs_norm)

            # ── 解析动作（获取飞行+CU决策）──
            orig_pure_pos_for_parse = env.cur_uavs_pos.copy()
            parsed = env._parse_actions(action_bs["continuous"], action_bs["discrete"])
            env.cur_uavs_pos = orig_pure_pos_for_parse  # 不改变 env 位置

            # ── Snapshot CCCP 所需的环境状态 ──
            snapshot_channels = {
                "uavs_2_cus": env.uavs_2_cus_channels.copy(),
                "uavs_2_bs": env.uavs_2_bs_channels.copy(),
                "cus_2_bs": env.cus_2_bs_channels.copy(),
                "uavs_2_targets": env.uavs_2_targets_channels.copy(),
            }
            snapshot_target_matrix = env.uavs_targets_matched_matrix.copy()
            snapshot_prev_pos = env.cur_uavs_pos.copy()

            # ── 调用 CCCP 获取对比基准 energy_opt ──
            cccp_eval = evaluate_via_cccp(
                base_args=base_args,
                parsed=parsed,
                saved_channels=snapshot_channels,
                saved_target_matrix=snapshot_target_matrix,
                saved_prev_uavs_pos=snapshot_prev_pos,
                cus_entertaining_task_size=env.cus_entertaining_task_size,
            )
            cccp_nosolution = cccp_eval["no_solution"]

            next_s, total_reward, r_dict, done, obj_fun, success_flag = env.step({"bs": action_bs}, i_episode=test_seed)

            success_slots += success_flag
            cccp_success_slots += int(not cccp_nosolution)
            episode_rewards_total.append(float(total_reward))
            if np.isfinite(obj_fun):
                obj_fun_total.append(float(obj_fun))
            if np.isfinite(cccp_eval["energy_opt"]):
                cccp_energy_total.append(float(cccp_eval["energy_opt"]))
            comps = r_dict.get("components", {})
            if np.isfinite(comps.get("E_fly", float("nan"))):
                flight_energy_total.append(comps["E_fly"] / max(base_args.omega_weight_2, 1e-8))
                sensing_energy_total.append(comps["E_sen"] / max(base_args.omega_weight_2, 1e-8))
                offloading_energy_total.append(comps["E_off"] / max(base_args.omega_weight_2, 1e-8))
                computation_energy_total.append(comps["E_cp"] / max(base_args.omega_weight_1, 1e-8))
                avg_uav_speed_total.append(comps["avg_uav_speed"])
                boundary_penalty_total.append(comps.get("boundary_penalty", 0))
                collision_penalty_total.append(comps.get("collision_penalty", 0))
                spectrum_penalty_total.append(comps.get("bs_alloc_spectrum_penalty", 0))
                tanh_penalty_total.append(comps.get("tanh_penalty", 0))

            next_state_bs_norm = running_norm_bs(np.array(next_s["bs"], dtype=np.float32), update=False)
            state_bs_norm = next_state_bs_norm
            terminal = done

        all_results.append({
            "uav_height": height,
            "run_id": 0,
            "avg_total_reward": float(np.mean(episode_rewards_total)),
            "avg_obj_fun": float(np.mean(obj_fun_total)) if obj_fun_total else float("nan"),
            "avg_cccp_energy": float(np.mean(cccp_energy_total)) if cccp_energy_total else float("nan"),
            "completion_rate": (success_slots / madrl_args.total_time_slots) * 100.0,
            "cccp_success_rate": (cccp_success_slots / madrl_args.total_time_slots) * 100.0,
            "avg_flight_energy": float(np.mean(flight_energy_total)) if flight_energy_total else float("nan"),
            "avg_sensing_energy": float(np.mean(sensing_energy_total)) if sensing_energy_total else float("nan"),
            "avg_offloading_energy": float(np.mean(offloading_energy_total)) if offloading_energy_total else float("nan"),
            "avg_computation_energy": float(np.mean(computation_energy_total)) if computation_energy_total else float("nan"),
            "avg_uav_speed": float(np.mean(avg_uav_speed_total)) if avg_uav_speed_total else float("nan"),
            "avg_boundary_penalty": float(np.mean(boundary_penalty_total)) if boundary_penalty_total else float("nan"),
            "avg_collision_penalty": float(np.mean(collision_penalty_total)) if collision_penalty_total else float("nan"),
            "avg_spectrum_penalty": float(np.mean(spectrum_penalty_total)) if spectrum_penalty_total else float("nan"),
            "avg_tanh_penalty": float(np.mean(tanh_penalty_total)) if tanh_penalty_total else float("nan"),
        })

    df = pd.DataFrame(all_results)
    filename = f"{current_time_str}_pure_MHBPPO_test_uav_height_results.csv"
    df.to_csv(filename, index=False)
    print(f"Test results saved to {filename}")
    print(df.to_string(index=False))
