"""Offline, head-only receive-beam pretraining (no environment/solver calls).

This module is intentionally separate from HPPO.update and behavior_clone.
The caller supplies a paired actor/normalizer and a strictly split dataset.
"""
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.distributions import Beta, kl_divergence

from algorithms.MHBPPO.MHBPPO_agent import MultiHeadActor
from algorithms.MHBPPO.checkpoint import load_state_normalization
from algorithms.MHBPPO.rollout_batch import actor_state_signature, normalization_signature


BEAM_HEAD_NAMES = ("uav_rec_beam_dir_real", "uav_rec_beam_dir_imag")
PRETRAIN_CONTRACT = "canonical_matched_beam_finite_beta_kl_v1"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _integer_vector(values, name, size):
    array = np.asarray(values)
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite vector with {size} entries")
    if np.any(array != np.floor(array)):
        raise ValueError(f"{name} must contain integers")
    return np.asarray(array, dtype=np.int64)


def load_dataset(path):
    """Return (arrays, sidecar_manifest); split -1 is retained but excluded.

    Required names follow the U48 dataset contract. Identifier aliases are
    accepted only when unambiguous; they are returned under canonical names.
    This loader never fits or updates normalization statistics.
    """
    path = Path(path).resolve()
    manifest_path = path.with_suffix(".json")
    if not path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("both dataset NPZ and same-stem JSON are required")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    dataset_sha = file_sha256(path)
    if not isinstance(manifest.get("dataset_sha256"), str) or manifest["dataset_sha256"] != dataset_sha:
        raise ValueError("dataset manifest requires the matching dataset_sha256")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    required = ("raw_states", "teacher_g_real", "teacher_g_imag", "split",
                "uav_target_distances", "physical_cu_power", "physical_matching")
    missing = [name for name in required if name not in arrays]
    if missing:
        raise ValueError(f"dataset is missing {missing}")
    raw = np.asarray(arrays["raw_states"], dtype=np.float32)
    if raw.ndim != 2 or not len(raw) or not np.all(np.isfinite(raw)):
        raise ValueError("raw_states must be a nonempty finite matrix")
    count = len(raw)
    real = np.asarray(arrays["teacher_g_real"], dtype=np.float64)
    imag = np.asarray(arrays["teacher_g_imag"], dtype=np.float64)
    if real.ndim != 3 or real.shape != imag.shape or real.shape[0] != count:
        raise ValueError("teacher real/imag must have aligned [state,UAV,antenna] shapes")
    if not np.all(np.isfinite(real)) or not np.all(np.isfinite(imag)):
        raise ValueError("teacher must be finite")
    uavs, antennas = real.shape[1:]
    if not uavs or not antennas:
        raise ValueError("teacher UAV/antenna dimensions must be positive")
    norms = np.sqrt(np.sum(real * real + imag * imag, axis=-1))
    if not np.allclose(norms, 1.0, rtol=0, atol=2e-6):
        raise ValueError("teacher beams must already have unit norm")
    if np.any(real[:, :, 0] <= 0) or np.max(np.abs(imag[:, :, 0])) > 2e-6:
        raise ValueError("matched teachers must have a positive-real first antenna")
    split = _integer_vector(arrays["split"], "split", count)
    if not set(np.unique(split)).issubset({-1, 0, 1, 2}):
        raise ValueError("split values must be -1/excluded,0/train,1/val,2/test")
    if not all(np.any(split == value) for value in (0, 1, 2)):
        raise ValueError("train, validation and test must all be nonempty")
    aliases = {"episode": ("episode_idx",), "round": ("collection_round",),
               "worker": ("worker_id",), "slot": ("time_slot",)}
    for canonical, alternatives in aliases.items():
        found = [name for name in (canonical,) + alternatives if name in arrays]
        if not found:
            raise ValueError(f"dataset is missing {canonical}")
        vectors = [_integer_vector(arrays[name], name, count) for name in found]
        if any(not np.array_equal(vectors[0], value) for value in vectors[1:]):
            raise ValueError(f"conflicting aliases for {canonical}")
        arrays[canonical] = vectors[0]
    if np.any(arrays["slot"] < 0) or np.any(arrays["slot"] >= 40):
        raise ValueError("U48 data must contain slots 0..39")
    # Every round (including all its workers) belongs to only one active split.
    for round_idx in np.unique(arrays["round"]):
        members = split[(arrays["round"] == round_idx) & (split >= 0)]
        if len(np.unique(members)) > 1:
            raise ValueError("one collection round crosses dataset splits")
    # Strict held-out eligibility: exact float32 observations cannot cross
    # train->validation/test or validation->test. Excluded rows do not count.
    earlier = set()
    for value in (0, 1, 2):
        current = {hashlib.sha256(row.tobytes()).digest() for row in raw[split == value]}
        if earlier.intersection(current):
            raise ValueError("held-out raw states duplicate an earlier split")
        earlier.update(current)
    distances = np.asarray(arrays["uav_target_distances"], dtype=np.float64)
    powers = np.asarray(arrays["physical_cu_power"], dtype=np.float64)
    matching = np.asarray(arrays["physical_matching"])
    if distances.shape != (count, uavs) or np.any(distances <= 0) or not np.all(np.isfinite(distances)):
        raise ValueError("invalid per-UAV target distances")
    if powers.ndim != 2 or powers.shape[0] != count or not np.all(np.isfinite(powers)) or np.any(powers < 0):
        raise ValueError("invalid physical CU powers")
    if matching.shape != (count, uavs) or not np.all(np.isfinite(matching)):
        raise ValueError("invalid physical matching")
    if np.any(matching != np.floor(matching)) or np.any(matching < 0) or np.any(matching >= powers.shape[1]):
        raise ValueError("matching must contain zero-based CU indices")
    arrays.update(raw_states=raw, teacher_g_real=real, teacher_g_imag=imag,
                  split=split, uav_target_distances=distances,
                  physical_cu_power=powers, physical_matching=matching.astype(np.int64))
    manifest = dict(manifest)
    manifest["_dataset_path"] = str(path)
    manifest["_manifest_path"] = str(manifest_path)
    manifest["_dataset_sha256"] = dataset_sha
    manifest["_manifest_sha256"] = file_sha256(manifest_path)
    return arrays, manifest


def _architecture(source, metadata):
    if "architecture" in metadata:
        architecture = dict(metadata["architecture"])
    else:
        hyper_path = source.parent / "hyperparameters.json"
        if not hyper_path.is_file():
            raise FileNotFoundError("source requires architecture metadata or run hyperparameters.json")
        hyper = json.loads(hyper_path.read_text(encoding="utf-8-sig"))
        hp = hyper["hyperparameters"]
        base = hyper["environment"]["base_args"]
        i, j, n = int(base["uavs_num"]), int(base["cus_num"]), int(base["antenna_nums"])
        architecture = {"state_dim": 1017, "hidden_dim": int(hp["hidden_dim"]),
            "continuous_action_splits": {"uav_angles": i, "uav_distances": i,
                "uav_off_durations": i, "cu_off_powers": j,
                "uav_rec_beam_dir_real": i*n, "uav_rec_beam_dir_imag": i*n},
            "discrete_action_dims": [j]*i, "continuous_dist_type": "beta",
            "structured_beam_alpha": float(hp["structured_beam_alpha"])}
    if architecture.get("continuous_dist_type", "beta") != "beta":
        raise ValueError("receive-beam pretraining requires a Beta actor")
    if float(architecture.get("structured_beam_alpha", 1.0)) != 1.0:
        raise ValueError("pretraining validation requires structured_beam_alpha=1")
    return architecture


def build_actor_from_checkpoint(source_checkpoint, *, device="cpu", state_dim=None,
                                uavs_num=None, cus_num=None, antenna_nums=None):
    """Load a complete, paired actor without constructing an env or critic.

    Return (actor, SavedStateNormalization, sourceinfo). Critic bytes are
    checked/preserved for later A/B initialization but never optimized here.
    """
    source = Path(source_checkpoint).resolve()
    meta_path = source / "checkpoint_metadata.json"
    if not meta_path.is_file():
        meta_path = source / "initialization_metadata.json"
    if not meta_path.is_file():
        raise FileNotFoundError("source checkpoint metadata is required")
    metadata = json.loads(meta_path.read_text(encoding="utf-8-sig"))
    architecture = _architecture(source, metadata)
    for name in ("mhbppo_actor.pth", "mhbppo_critic.pth", "state_normalization.npz"):
        if not (source/name).is_file():
            raise FileNotFoundError(source/name)
    dimension = int(architecture["state_dim"])
    splits = architecture["continuous_action_splits"]
    discrete = architecture["discrete_action_dims"]
    inferred_i = len(discrete)
    inferred_j = int(discrete[0])
    inferred_n = int(splits[BEAM_HEAD_NAMES[0]]) // inferred_i
    expected = ((state_dim, dimension, "state"), (uavs_num, inferred_i, "UAV"),
                (cus_num, inferred_j, "CU"), (antenna_nums, inferred_n, "antenna"))
    for requested, actual, label in expected:
        if requested is not None and int(requested) != actual:
            raise ValueError(f"source {label} dimension mismatch")
    iteration = int(metadata.get("iteration", metadata.get("completed_ppo_updates", 0)))
    norm = load_state_normalization(source/"state_normalization.npz", expected_dim=dimension,
        expected_iteration=iteration, actor_path=source/"mhbppo_actor.pth")
    if norm.epsilon != 1e-8:
        raise ValueError("source normalization epsilon differs from sampler")
    effective_std = np.ones_like(norm.mean) if norm.count <= 1e-6 else np.maximum(
        np.sqrt(norm.m2 / max(norm.count, 1e-8)), 1e-4)
    if not np.array_equal(effective_std, norm.std):
        raise ValueError("source std disagrees with exact Welford moments")
    # Model construction is RNG-isolated; weights are fully replaced below.
    with isolated_torch_rng(0, torch.device(device)):
        actor = MultiHeadActor(dimension, int(architecture["hidden_dim"]), splits, discrete,
            continuous_dist_type="beta", structured_beam_alpha=1.0).to(device)
    actor.load_state_dict(torch.load(source/"mhbppo_actor.pth", map_location=device,
                                    weights_only=True), strict=True)
    actor.eval()
    actor_sig = actor_state_signature(actor)
    norm_sig = normalization_signature(norm.mean, norm.std, norm.epsilon)
    if metadata.get("actor_signature") not in (None, actor_sig):
        raise ValueError("source actor SHA disagrees with checkpoint metadata")
    if metadata.get("norm_signature") not in (None, norm_sig):
        raise ValueError("source norm SHA disagrees with checkpoint metadata")
    source_hashes = {p.name: file_sha256(p) for p in source.iterdir() if p.is_file()}
    return actor, norm, {"source_checkpoint": str(source), "source_iteration": iteration,
        "source_metadata": metadata, "architecture": architecture,
        "actor_signature": actor_sig, "norm_signature": norm_sig,
        "normalization_count": norm.count, "source_file_sha256": source_hashes,
        "actor_file_sha256": source_hashes["mhbppo_actor.pth"],
        "critic_file_sha256": source_hashes["mhbppo_critic.pth"]}


@contextmanager
def isolated_torch_rng(seed, device="cpu"):
    device = torch.device(device)
    cuda = [] if device.type != "cuda" else [device.index if device.index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=cuda, enabled=True):
        torch.random.default_generator.manual_seed(int(seed))
        if cuda:
            torch.cuda.default_generators[cuda[0]].manual_seed(int(seed))
        yield


def named_beam_parameters(actor):
    prefixes = tuple(f"continuous_heads.{name}." for name in BEAM_HEAD_NAMES)
    return {name: value for name, value in actor.named_parameters() if name.startswith(prefixes)}


def tensor_hashes(actor, *, beam=False):
    names = set(named_beam_parameters(actor))
    result = {}
    for name, value in actor.state_dict().items():
        if (name in names) == bool(beam):
            array = np.ascontiguousarray(value.detach().cpu().numpy())
            result[name] = hashlib.sha256(array.tobytes()).hexdigest()
    return result


@contextmanager
def head_only_training(actor):
    parameters = dict(actor.named_parameters())
    selected = named_beam_parameters(actor)
    if len(selected) != 8:
        raise ValueError("expected weights/biases of four receive-beam Linear layers")
    flags = {name: value.requires_grad for name, value in parameters.items()}
    gradients = {name: None if value.grad is None else value.grad.detach().clone()
                 for name, value in parameters.items()}
    modes = {module: module.training for module in actor.modules()}
    try:
        actor.eval()
        for name, value in parameters.items():
            value.requires_grad_(name in selected)
            value.grad = None
        yield list(selected.values())
    finally:
        for name, value in parameters.items():
            value.requires_grad_(flags[name])
            value.grad = gradients[name]
        for module, training in modes.items():
            module.training = training


def teacher_beta_parameters(teacher_g_real, teacher_g_imag, *, rho=0.9, kappa=64.0):
    rho, kappa = float(rho), float(kappa)
    if not math.isfinite(rho) or not 0 < rho < 1 or not math.isfinite(kappa) or kappa <= 0:
        raise ValueError("rho must be in (0,1) and kappa finite positive")
    real = np.asarray(teacher_g_real, dtype=np.float64)
    imag = np.asarray(teacher_g_imag, dtype=np.float64)
    if real.ndim != 3 or real.shape != imag.shape:
        raise ValueError("teacher components must have matching [state,UAV,antenna] shapes")
    mean = 0.5*(1.0 + rho*np.concatenate((real.reshape(len(real), -1),
                                        imag.reshape(len(imag), -1)), axis=1))
    alpha, beta = kappa*mean, kappa*(1.0-mean)
    if not np.all(np.isfinite(alpha)) or not np.all(np.isfinite(beta)) or np.any(alpha <= 1) or np.any(beta <= 1):
        raise ValueError("all teacher Beta alpha/beta must exceed one")
    return {"mean": mean, "alpha": alpha, "beta": beta}


def beam_dists(actor, states=None, *, features=None):
    if (states is None) == (features is None):
        raise ValueError("supply exactly one of normalized states or cached features")
    device = next(actor.parameters()).device
    if features is None:
        states = torch.as_tensor(states, dtype=torch.float32, device=device)
        features = actor._forward_features(states)
    else:
        features = torch.as_tensor(features, dtype=torch.float32, device=device)
    return tuple(actor.continuous_heads[name].get_dist(features) for name in BEAM_HEAD_NAMES)


def cache_frozen_features(actor, raw_states, normalizer, indices, *, batch_size=256):
    device = next(actor.parameters()).device
    pieces = []
    with torch.no_grad():
        for start in range(0, len(indices), int(batch_size)):
            selected = np.asarray(indices[start:start+int(batch_size)], dtype=np.int64)
            state = torch.as_tensor(normalizer.normalize(raw_states[selected]),
                                    dtype=torch.float32, device=device)
            pieces.append(actor._forward_features(state).detach().cpu())
    if not pieces:
        raise ValueError("cannot cache an empty split")
    return torch.cat(pieces, dim=0)


def _draw_unit_beams(actor, states=None, *, features=None, draws=8):
    real, imag = beam_dists(actor, states, features=features)
    uavs = len(actor.discrete_action_dims)
    width = actor.continuous_action_splits[BEAM_HEAD_NAMES[0]]
    if width % uavs:
        raise ValueError("beam width is not divisible by UAV count")
    antennas = width // uavs
    r = (2*real.sample((int(draws),))-1).reshape(int(draws), -1, uavs, antennas)
    im = (2*imag.sample((int(draws),))-1).reshape(int(draws), -1, uavs, antennas)
    raw = torch.complex(r, im)
    norm = torch.linalg.vector_norm(raw, dim=-1, keepdim=True)
    fallback = norm < 1e-12
    result = raw / norm.clamp_min(1e-12)
    unit = torch.zeros_like(result)
    unit[..., 0] = 1
    return torch.where(fallback, unit, result), fallback


def sample_beams(actor, states=None, *, features=None, draws=8, seed=None):
    """Return complex Tensor[draw,state,UAV,antenna], sampling beam heads only."""
    if int(draws) <= 0:
        raise ValueError("draws must be positive")
    if seed is None:
        return _draw_unit_beams(actor, states, features=features, draws=draws)[0]
    with isolated_torch_rng(seed, next(actor.parameters()).device):
        return _draw_unit_beams(actor, states, features=features, draws=draws)[0]


def evaluate_target_gain(actor, features, teacher_g, *, draws=8, seed=42, batch_size=256):
    """Evaluate a supplied split only. No test split is accessed implicitly."""
    teacher = np.asarray(teacher_g)
    if len(features) != len(teacher) or not len(teacher) or int(draws) <= 0 or int(batch_size) <= 0:
        raise ValueError("invalid evaluation split or sampling budget")
    device = next(actor.parameters()).device
    modes = {module: module.training for module in actor.modules()}
    gains, fallbacks = [], 0
    started = time.perf_counter()
    try:
        actor.eval()
        with torch.no_grad(), isolated_torch_rng(seed, device):
            for start in range(0, len(features), int(batch_size)):
                target = torch.as_tensor(teacher[start:start+int(batch_size)],
                                         dtype=torch.complex64, device=device)
                beams, small = _draw_unit_beams(actor, features=features[start:start+int(batch_size)], draws=draws)
                numerator = torch.abs(torch.sum(target.conj().unsqueeze(0)*beams, dim=-1)).square()
                denominator = torch.sum(torch.abs(target).square(), dim=-1).unsqueeze(0)
                gain = numerator/denominator.clamp_min(1e-12)
                if not torch.all(torch.isfinite(gain)) or torch.any(gain > 1.0001):
                    raise ValueError("invalid sampled physical gain")
                gains.append(gain.clamp(0, 1).cpu().numpy())
                fallbacks += int(small.sum().item())
    finally:
        for module, training in modes.items():
            module.training = training
    values = np.concatenate(gains, axis=1).astype(np.float64)
    return {"mean_gain": float(values.mean()), "gain_mean": float(values.mean()),
        "gain_std": float(values.std()),
        "per_uav_mean_gain": values.mean(axis=(0, 1)).tolist(),
        "gain_p05": float(np.quantile(values, .05)), "gain_median": float(np.median(values)),
        "gain_p95": float(np.quantile(values, .95)), "states": len(teacher),
        "draws": int(draws), "small_norm_fallback_rate": fallbacks/values.size,
        "seconds": time.perf_counter()-started}


def train_receive_beam(actor, arrays, normalizer, *, rho=0.9, kappa=64.0,
                       lr=1e-3, epochs=100, batch_size=256, seed=42,
                       patience=10, validation_draws=8, min_delta=1e-5,
                       progress_callback=None):
    """Train only on split0, select only on split1, never evaluate split2/-1.

    Returns CPU best/last state_dicts and JSON-compatible metrics. The actor
    itself ends at the selected best state; flags/modes/RNG are restored.
    """
    if not math.isfinite(float(lr)) or float(lr) <= 0:
        raise ValueError("pretrain lr must be positive and finite")
    if any(int(x) <= 0 for x in (epochs, batch_size, patience, validation_draws)):
        raise ValueError("epochs/batch_size/patience/draws must be positive")
    if not math.isfinite(float(min_delta)) or min_delta < 0:
        raise ValueError("min_delta must be finite and nonnegative")
    split = np.asarray(arrays["split"])
    train_indices, val_indices = np.flatnonzero(split == 0), np.flatnonzero(split == 1)
    if not len(train_indices) or not len(val_indices):
        raise ValueError("nonempty train and validation splits are required")
    if arrays["raw_states"].shape[1] != normalizer.mean.size:
        raise ValueError("normalizer state dimension mismatch")
    dimension = int(actor.continuous_action_splits[BEAM_HEAD_NAMES[0]])
    if dimension != int(np.prod(arrays["teacher_g_real"].shape[1:])):
        raise ValueError("teacher/actor beam dimensions differ")
    teacher = teacher_beta_parameters(arrays["teacher_g_real"][train_indices],
        arrays["teacher_g_imag"][train_indices], rho=rho, kappa=kappa)
    teacher_val = (arrays["teacher_g_real"][val_indices] +
                   1j*arrays["teacher_g_imag"][val_indices])
    t_alpha = torch.as_tensor(teacher["alpha"], dtype=torch.float32)
    t_beta = torch.as_tensor(teacher["beta"], dtype=torch.float32)
    original_nonbeam = tensor_hashes(actor)
    original_beam = tensor_hashes(actor, beam=True)
    device = next(actor.parameters()).device
    sequence = np.random.SeedSequence(int(seed), spawn_key=(0x424D5054,))
    shuffle_seq, val_seq = sequence.spawn(2)
    shuffle = np.random.default_rng(shuffle_seq)
    val_seed = int(val_seq.generate_state(1, dtype=np.uint32)[0])
    snapshot = lambda: {name: value.detach().cpu().clone() for name, value in actor.state_dict().items()}
    history = []
    started = time.perf_counter()
    with isolated_torch_rng(seed, device), head_only_training(actor) as parameters:
        features_train = cache_frozen_features(actor, arrays["raw_states"], normalizer,
                                               train_indices, batch_size=batch_size)
        features_val = cache_frozen_features(actor, arrays["raw_states"], normalizer,
                                             val_indices, batch_size=batch_size)
        baseline = evaluate_target_gain(actor, features_val, teacher_val,
            draws=validation_draws, seed=val_seed, batch_size=batch_size)
        best_score, best_epoch, best_state = baseline["mean_gain"], 0, snapshot()
        best_metrics, no_improvement, steps = dict(baseline), 0, 0
        optimizer = torch.optim.Adam(parameters, lr=float(lr), eps=1e-5)
        for epoch in range(1, int(epochs)+1):
            epoch_started = time.perf_counter()
            order = shuffle.permutation(len(train_indices))
            loss_sum, norm_sum, seen, batches = 0., 0., 0, 0
            for start in range(0, len(order), int(batch_size)):
                index = torch.as_tensor(order[start:start+int(batch_size)], dtype=torch.long)
                real, imag = beam_dists(actor, features=features_train[index])
                a, b = t_alpha[index].to(device), t_beta[index].to(device)
                real_kl = kl_divergence(Beta(a[:, :dimension], b[:, :dimension]), real)
                imag_kl = kl_divergence(Beta(a[:, dimension:], b[:, dimension:]), imag)
                loss = torch.cat((real_kl, imag_kl), dim=-1).mean()
                if not torch.isfinite(loss):
                    raise ValueError("non-finite receive-beam KL")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
                optimizer.step()
                size = len(index)
                loss_sum += float(loss.detach().cpu())*size
                norm_sum += float(grad_norm.detach().cpu())
                seen += size
                batches += 1
                steps += 1
            validation = evaluate_target_gain(actor, features_val, teacher_val,
                draws=validation_draws, seed=val_seed, batch_size=batch_size)
            improved = validation["mean_gain"] > best_score + float(min_delta)
            if improved:
                best_score, best_epoch, best_state = validation["mean_gain"], epoch, snapshot()
                best_metrics, no_improvement = dict(validation), 0
            else:
                no_improvement += 1
            row = {"epoch": epoch, "train_kl_per_coordinate": loss_sum/seen,
                "val_mean_gain": validation["mean_gain"], "best_val_mean_gain": best_score,
                "best_epoch": best_epoch, "gradient_norm_mean": norm_sum/batches,
                "optimizer_steps": steps, "seconds": time.perf_counter()-epoch_started,
                "no_improvement_epochs": no_improvement}
            history.append(row)
            if progress_callback is not None:
                progress_callback(dict(row))
            if no_improvement >= int(patience):
                break
        last_state = snapshot()
        actor.load_state_dict(best_state, strict=True)
    if tensor_hashes(actor) != original_nonbeam:
        raise AssertionError("head-only pretraining changed shared/nonbeam parameters")
    return {"contract": PRETRAIN_CONTRACT, "best_state_dict": best_state,
        "last_state_dict": last_state, "history": history, "best_epoch": best_epoch,
        "epochs_completed": len(history), "optimizer_steps": steps,
        "baseline_validation": baseline, "best_validation": best_metrics,
        "validation_improved": best_epoch > 0,
        "beam_parameters_changed": tensor_hashes(actor, beam=True) != original_beam,
        "nonbeam_parameters_unchanged": True, "train_states": len(train_indices),
        "validation_states": len(val_indices), "test_evaluated": False,
        "excluded_rows_trained": False, "validation_seed": val_seed,
        "training_seconds": time.perf_counter()-started, "rho": float(rho),
        "kappa": float(kappa), "lr": float(lr), "batch_size": int(batch_size),
        "max_epochs": int(epochs), "patience": int(patience),
        "validation_draws": int(validation_draws), "nonbeam_tensor_sha256": original_nonbeam}
