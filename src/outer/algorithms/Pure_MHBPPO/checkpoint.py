"""Read Pure-MHBPPO evaluation normalization paired with the saved actor."""
import hashlib
import json
from pathlib import Path

import numpy as np


def load_environment_args(checkpoint_dir):
    """Restore the saved action mapping too; old checkpoints remain legacy."""
    from algorithms.Pure_MHBPPO.Pure_MHBPPO_main import get_pure_args
    from algorithms.Pure_MHBPPO.time_decoder import TIME_DECODER_VERSION

    path = Path(checkpoint_dir) / "checkpoint_metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    args = get_pure_args(["--config", str(path)])
    contract = metadata.get("action_parameterization", {})
    if contract and (contract.get("time_decoder") != args.pure_time_decoder or
                     contract.get("time_decoder_version") != TIME_DECODER_VERSION):
        raise ValueError("unsupported or inconsistent Pure-MHBPPO time decoder contract")
    return args


def load_normalization(checkpoint_dir, expected_dim=None):
    directory = Path(checkpoint_dir)
    actor_path = directory / "pure_mhbppo_actor.pth"
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
        raise ValueError("invalid Pure-MHBPPO normalization statistics")
    if expected_dim is not None and mean.size != expected_dim:
        raise ValueError("Pure-MHBPPO normalization dimension mismatch")
    if hashlib.sha256(actor_path.read_bytes()).hexdigest() != fingerprint:
        raise ValueError("actor does not match its saved normalization")
    return {"mean": mean, "std": std, "epsilon": epsilon, "iteration": iteration}


def normalize_state(raw, stats):
    state = np.asarray(raw, dtype=np.float32)
    if state.shape[-1] != stats["mean"].size:
        raise ValueError("Pure-MHBPPO observation dimension mismatch")
    return (state - stats["mean"]) / (stats["std"] + stats["epsilon"])
