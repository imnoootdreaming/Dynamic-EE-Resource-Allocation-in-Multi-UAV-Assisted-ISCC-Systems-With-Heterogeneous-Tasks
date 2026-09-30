"""Offline feasible-action dataset utilities for MHBPPO warm starts.

The file format is a compressed NumPy archive with at least ``states``,
``continuous_actions`` and ``discrete_actions``.  States must already use the
same normalization statistics as the rollout workers.  Optional
``state_norm_mean`` and ``state_norm_std`` arrays are loaded so the learner can
continue broadcasting the exact statistics used to build the dataset.
"""

import os

import numpy as np


def load_behavior_cloning_dataset(path):
    """Load and validate a BC ``.npz`` dataset without changing its values."""
    if not path:
        return None
    path = os.path.abspath(os.fspath(path))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"BC 数据集不存在: {path}")
    with np.load(path, allow_pickle=False) as data:
        required = ("states", "continuous_actions", "discrete_actions")
        missing = [name for name in required if name not in data]
        if missing:
            raise ValueError(f"BC 数据集缺少字段: {missing}")
        states = np.asarray(data["states"], dtype=np.float32)
        continuous_actions = np.asarray(data["continuous_actions"], dtype=np.float32)
        discrete_actions = np.asarray(data["discrete_actions"], dtype=np.int64)
        state_norm_mean = (
            np.asarray(data["state_norm_mean"], dtype=np.float64)
            if "state_norm_mean" in data else None
        )
        state_norm_std = (
            np.asarray(data["state_norm_std"], dtype=np.float64)
            if "state_norm_std" in data else None
        )
        state_norm_count = int(np.asarray(data["state_norm_count"]).item()) \
            if "state_norm_count" in data else None

    if states.ndim != 2 or continuous_actions.ndim != 2 or discrete_actions.ndim != 2:
        raise ValueError("BC 数据集的 states/actions 必须是二维数组")
    if not (len(states) == len(continuous_actions) == len(discrete_actions)):
        raise ValueError("BC 数据集的 states/actions 样本数不一致")
    if len(states) == 0:
        raise ValueError("BC 数据集为空")
    if not np.all(np.isfinite(states)) or not np.all(np.isfinite(continuous_actions)):
        raise ValueError("BC 数据集包含 NaN/Inf")
    if np.min(continuous_actions) < -1e-5 or np.max(continuous_actions) > 1.0 + 1e-5:
        raise ValueError("BC continuous_actions 必须位于 [0, 1] actor 归一化空间")

    return {
        "path": path,
        "states": states,
        "continuous_actions": np.clip(continuous_actions, 0.0, 1.0),
        "discrete_actions": discrete_actions,
        "state_norm_mean": state_norm_mean,
        "state_norm_std": state_norm_std,
        "state_norm_count": state_norm_count,
    }


def save_behavior_cloning_dataset(path, states, continuous_actions,
                                  discrete_actions, state_norm_mean=None,
                                  state_norm_std=None, **metadata):
    """Save a validated BC dataset as a portable ``.npz`` archive."""
    states = np.asarray(states, dtype=np.float32)
    continuous_actions = np.asarray(continuous_actions, dtype=np.float32)
    discrete_actions = np.asarray(discrete_actions, dtype=np.int64)
    if not (states.ndim == continuous_actions.ndim == discrete_actions.ndim == 2):
        raise ValueError("BC 数据的 states/actions 必须是二维数组")
    if not (len(states) == len(continuous_actions) == len(discrete_actions)):
        raise ValueError("BC 数据的 states/actions 样本数不一致")
    if len(states) == 0:
        raise ValueError("不能保存空 BC 数据集")
    payload = {
        "states": states,
        "continuous_actions": np.clip(continuous_actions, 0.0, 1.0),
        "discrete_actions": discrete_actions,
    }
    if state_norm_mean is not None:
        payload["state_norm_mean"] = np.asarray(state_norm_mean, dtype=np.float64)
    if state_norm_std is not None:
        payload["state_norm_std"] = np.asarray(state_norm_std, dtype=np.float64)
    payload.update(metadata)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, **payload)
