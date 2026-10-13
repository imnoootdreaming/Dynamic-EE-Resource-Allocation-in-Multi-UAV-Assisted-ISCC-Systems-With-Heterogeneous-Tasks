"""State normalization paired with MHBPPO evaluation weights (NumPy-only)."""
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path

import numpy as np


STATE_NORMALIZATION_FILENAME = "state_normalization.npz"
NORMALIZATION_EPSILON = 1e-8


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class SavedStateNormalization:
    mean: np.ndarray
    std: np.ndarray
    count: float
    m2: np.ndarray
    iteration: int
    epsilon: float = NORMALIZATION_EPSILON

    def normalize(self, raw_state):
        """Frozen evaluation transform, exactly as sampler (including raw float32)."""
        raw = np.asarray(raw_state, dtype=np.float32)
        if raw.ndim == 0 or raw.shape[-1] != self.mean.size:
            raise ValueError(f"state dimension mismatch: expected {self.mean.size}, got {raw.shape}")
        return (raw - self.mean) / (self.std + self.epsilon)


def _validate_arrays(mean, std, m2, count, epsilon):
    if mean.ndim != 1 or not mean.size or std.shape != mean.shape or m2.shape != mean.shape:
        raise ValueError("normalization mean/std/m2 must be nonempty vectors of the same shape")
    if not all(np.all(np.isfinite(x)) for x in (mean, std, m2)):
        raise ValueError("normalization statistics must be finite")
    if (not np.isfinite(count) or count < 0 or not np.isfinite(epsilon) or epsilon <= 0
            or np.any(std <= 0) or np.any(m2 < 0)):
        raise ValueError("invalid normalization count/std/m2/epsilon")


def save_state_normalization(destination, stats, iteration, *, actor_path=None):
    """Save effective sampler std plus Welford moments, paired with actor bytes."""
    if stats is None:
        raise ValueError("state_norm_stats is required for a testable checkpoint")
    mean = np.asarray(stats.mean, dtype=np.float64).copy()
    std = np.asarray(stats.std, dtype=np.float64).copy()
    m2 = np.asarray(stats.m2, dtype=np.float64).copy()
    count = float(stats.count)
    _validate_arrays(mean, std, m2, count, NORMALIZATION_EPSILON)
    if int(iteration) != iteration or iteration < 0:
        raise ValueError("checkpoint iteration must be a nonnegative integer")
    payload = dict(format_version=np.int64(1), mean=mean, std=std, m2=m2,
        count=np.float64(count), epsilon=np.float64(NORMALIZATION_EPSILON),
        iteration=np.int64(iteration), weights_version=np.int64(iteration),
        state_dim=np.int64(mean.size),
        actor_filename="" if actor_path is None else Path(actor_path).name,
        actor_sha256="" if actor_path is None else _sha256(actor_path))
    destination = Path(destination)
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        with temporary.open("wb") as output:
            np.savez(output, **payload)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return str(destination)


def load_state_normalization(source, *, expected_dim=None, expected_iteration=None, actor_path=None):
    """Load frozen stats; reject missing legacy stats and mismatched actor files."""
    source = Path(source)
    if not source.is_file():
        raise FileNotFoundError(f"state normalization is missing: {source}; cannot infer it from legacy weights")
    with np.load(source, allow_pickle=False) as data:
        if int(data["format_version"]) != 1:
            raise ValueError("unsupported state normalization format")
        mean = np.array(data["mean"], dtype=np.float64, copy=True)
        std = np.array(data["std"], dtype=np.float64, copy=True)
        m2 = np.array(data["m2"], dtype=np.float64, copy=True)
        count, epsilon = float(data["count"]), float(data["epsilon"])
        iteration = int(data["iteration"])
        filename, fingerprint = str(data["actor_filename"]), str(data["actor_sha256"])
        if int(data["state_dim"]) != mean.size or int(data["weights_version"]) != iteration:
            raise ValueError("inconsistent normalization metadata")
    _validate_arrays(mean, std, m2, count, epsilon)
    if expected_dim is not None and mean.size != expected_dim:
        raise ValueError(f"normalization dimension mismatch: {mean.size} != {expected_dim}")
    if expected_iteration is not None and iteration != expected_iteration:
        raise ValueError(f"normalization iteration mismatch: {iteration} != {expected_iteration}")
    if fingerprint:
        paired_actor = source.parent / filename if actor_path is None else Path(actor_path)
        if _sha256(paired_actor) != fingerprint:
            raise ValueError("actor weights do not match saved state normalization")
    for array in (mean, std, m2):
        array.setflags(write=False)
    return SavedStateNormalization(mean, std, count, m2, iteration, epsilon)
