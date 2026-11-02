"""Read SHBPPO evaluation normalization paired with the saved actor."""
import hashlib
from pathlib import Path

import numpy as np


def load_normalization(checkpoint_dir, expected_dim=None):
    directory = Path(checkpoint_dir)
    actor_path = directory / "shbppo_actor.pth"
    with np.load(directory / "state_normalization.npz", allow_pickle=False) as data:
        mean = np.asarray(data["mean"], dtype=np.float64).copy()
        std = np.asarray(data["std"], dtype=np.float64).copy()
        fingerprint = str(data["actor_sha256"])
        count = int(data["count"])
        epsilon = float(data["epsilon"])
        iteration = int(data["iteration"])
    if (mean.ndim != 1 or std.shape != mean.shape or not np.all(np.isfinite(mean))
            or not np.all(np.isfinite(std)) or np.any(std < 0) or count < 1
            or epsilon <= 0 or not np.isfinite(epsilon)):
        raise ValueError("invalid SHBPPO normalization statistics")
    if expected_dim is not None and mean.size != expected_dim:
        raise ValueError("SHBPPO normalization dimension mismatch")
    if hashlib.sha256(actor_path.read_bytes()).hexdigest() != fingerprint:
        raise ValueError("actor does not match its saved normalization")
    return {"mean": mean, "std": std, "epsilon": epsilon, "iteration": iteration}


def normalize_state(raw, stats):
    state = np.asarray(raw, dtype=np.float32)
    if state.shape[-1] != stats["mean"].size:
        raise ValueError("SHBPPO observation dimension mismatch")
    return (state - stats["mean"]) / (stats["std"] + stats["epsilon"])
