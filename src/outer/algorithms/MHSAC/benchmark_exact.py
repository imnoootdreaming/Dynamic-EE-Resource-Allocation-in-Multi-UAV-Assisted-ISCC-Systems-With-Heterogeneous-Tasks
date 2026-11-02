"""Measure one CPU SAC update using the actual default environment dimensions."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

OUTER = Path(__file__).resolve().parents[2]
if str(OUTER) not in sys.path:
    sys.path.insert(0, str(OUTER))
from algorithms.MHSAC.MHSAC_agent import HSAC
from algorithms.MHSAC.MHSAC_main import get_all_args
from configs.base_params import get_base_args
from environment.my_env import MyEnv


def benchmark():
    args, sac = get_all_args([])
    env = MyEnv(base_args=get_base_args(seed=args.seed, target_layout=args.target_layout),
                madrl_args=args, seed_offset=args.scene_seed_offset)
    splits = dict(env.bs_continuous_action_splits)
    dims = env.action_space["bs"]["discrete"].nvec
    continuous = env.action_space["bs"]["continuous"]
    state_dim = env.observation_space["bs"].shape[0]
    torch.set_num_threads(1)
    start = time.perf_counter()
    agent = HSAC(state_dim, args.hidden_dim, splits, dims, continuous.low, continuous.high,
                 args.actor_lr, args.critic_lr, args.gamma, sac.tau, sac.alpha_continuous,
                 sac.alpha_discrete, torch.device("cpu"), alpha_lr=sac.alpha_lr)
    init_time = time.perf_counter() - start
    rng = np.random.default_rng(14)
    batch = args.batch_size
    transition = {"states": rng.standard_normal((batch, state_dim)).astype("float32"),
                  "next_states": rng.standard_normal((batch, state_dim)).astype("float32"),
                  "continuous_actions": rng.uniform(0, 1, (batch, sum(splits.values()))).astype("float32"),
                  "discrete_indices": rng.integers(0, agent.joint_action_dim, batch),
                  "rewards": rng.normal(size=batch).astype("float32"),
                  "dones": np.zeros(batch, dtype="float32")}
    start = time.perf_counter()
    agent.update(transition)
    result = {"joint_actions": agent.joint_action_dim, "batch": batch, "hidden_dim": args.hidden_dim,
              "state_dim": state_dim, "continuous_dim": sum(splits.values()), "cpu_threads": 1,
              "init_s": init_time, "single_sac_update_s": time.perf_counter() - start,
              "joint_tensor_mib": batch * agent.joint_action_dim * 4 / 1024**2}
    try:
        import psutil
        memory = psutil.Process().memory_info()
        result["rss_after_update_mib"] = memory.rss / 1024**2
        if hasattr(memory, "peak_wset"):
            result["peak_working_set_mib"] = memory.peak_wset / 1024**2
    except ImportError:
        result["memory"] = "psutil unavailable"
    return result


if __name__ == "__main__":
    print(json.dumps(benchmark()))
