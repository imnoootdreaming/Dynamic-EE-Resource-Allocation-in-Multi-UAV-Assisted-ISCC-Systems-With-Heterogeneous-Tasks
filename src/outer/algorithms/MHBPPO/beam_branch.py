"""Independent receive-beam feature branch for a controlled offline study.

The added branch consumes the same normalized full observation as the base
actor. It does not receive a teacher vector, geometry shortcut, or action
from another head. Existing PPO, actor, observation and action code is intact.
"""
from contextlib import contextmanager
import copy
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.distributions import Beta, kl_divergence

from algorithms.MHBPPO.MHBPPO_agent import MultiHeadActor
from algorithms.MHBPPO.beam_pretraining import (
    BEAM_HEAD_NAMES, amplitude_phase_beams, file_sha256, isolated_torch_rng, teacher_beta_parameters,
)
from algorithms.MHBPPO.checkpoint import load_state_normalization
from algorithms.MHBPPO.rollout_batch import actor_state_signature, normalization_signature


BRANCH_ARCHITECTURE_CONTRACT = "dedicated_receive_beam_branch_v1"
BRANCH_PRETRAIN_CONTRACT = "dedicated_receive_beam_finite_beta_kl_v1"


def _trainable_name(name):
    return (name.startswith("beam_fc1.") or name.startswith("beam_fc2.") or
            any(name.startswith(f"continuous_heads.{head}.") for head in BEAM_HEAD_NAMES))


class DedicatedBeamActor(MultiHeadActor):
    """Base actor plus an initially identical private fc1/fc2 beam encoder.

    Only private encoder and existing receive-beam heads are trainable.
    _forward_features intentionally remains the ORIGINAL nonbeam encoder;
    use beam_features/branch_dists for a beam-only validation adapter.
    Inherited sample/evaluate_actions dispatch through this _get_dists.
    """

    def __init__(self, source_actor):
        if not isinstance(source_actor, MultiHeadActor) or isinstance(source_actor, DedicatedBeamActor):
            raise TypeError("source must be an unbranched MultiHeadActor")
        if source_actor.receive_beam_mapping["receive_beam_phase_mode"] != "absolute":
            raise ValueError("dedicated branch does not support phase_residual mapping")
        if source_actor.continuous_dist_type != "beta" or source_actor.structured_beam_alpha_effective != 1.0:
            raise ValueError("branch study requires the original Beta actor with alpha=1")
        device, dtype = source_actor.fc1.weight.device, source_actor.fc1.weight.dtype
        # Orthogonal constructor draws never alter a caller's random stream.
        with isolated_torch_rng(0, device):
            super().__init__(source_actor.fc1.in_features, source_actor.fc1.out_features,
                copy.deepcopy(source_actor.continuous_action_splits),
                list(source_actor.discrete_action_dims), continuous_dist_type="beta",
                structured_beam_alpha=source_actor.structured_beam_alpha_effective)
            self.to(device=device, dtype=dtype)
        self.load_state_dict(source_actor.state_dict(), strict=True)
        self.beam_fc1 = copy.deepcopy(source_actor.fc1)
        self.beam_fc2 = copy.deepcopy(source_actor.fc2)
        self.train(source_actor.training)
        for name, value in self.named_parameters():
            value.requires_grad_(_trainable_name(name))
            value.grad = None

    def beam_features(self, states):
        """Private, trainable features of already-normalized full observations."""
        states = self.activate_func(self.beam_fc1(states))
        return self.activate_func(self.beam_fc2(states))

    def _get_dists(self, states):
        # Retain the base calculation/order for every other action head.
        continuous, discrete = super()._get_dists(states)
        features = self.beam_features(states)
        for name in BEAM_HEAD_NAMES:
            continuous[name] = self.continuous_heads[name].get_dist(features)
        return continuous, discrete

    def architecture_metadata(self):
        return {"actor_class": "DedicatedBeamActor",
            "architecture_contract": BRANCH_ARCHITECTURE_CONTRACT,
            "state_dim": self.fc1.in_features, "hidden_dim": self.fc1.out_features,
            "continuous_action_splits": dict(self.continuous_action_splits),
            "discrete_action_dims": list(self.discrete_action_dims),
            "continuous_dist_type": self.continuous_dist_type,
            "structured_beam_alpha": self.structured_beam_alpha_effective,
            "beam_branch_input": "source_exact_normalized_full_raw_state",
            "beam_branch_activation": "tanh_tanh",
            "base_actor_strict_load_compatible": False}


def build_branch_actor(source_actor):
    """Copy the full source actor, then add independent copied beam layers."""
    actor = DedicatedBeamActor(source_actor)
    if (actor.beam_fc1.weight.data_ptr() == actor.fc1.weight.data_ptr() or
            actor.beam_fc2.weight.data_ptr() == actor.fc2.weight.data_ptr()):
        raise AssertionError("private beam encoder accidentally shares base storage")
    return actor


def branch_dists(actor, states):
    """Return amplitude/phase Beta distributions from normalized state."""
    if not isinstance(actor, DedicatedBeamActor):
        raise TypeError("branch_dists requires DedicatedBeamActor")
    parameter = next(actor.parameters())
    states = torch.as_tensor(states, dtype=parameter.dtype, device=parameter.device)
    features = actor.beam_features(states)
    return tuple(actor.continuous_heads[name].get_dist(features) for name in BEAM_HEAD_NAMES)


def named_branch_parameters(actor):
    if not isinstance(actor, DedicatedBeamActor):
        raise TypeError("branch training requires DedicatedBeamActor")
    result = {name: value for name, value in actor.named_parameters() if _trainable_name(name)}
    if len(result) != 12:
        raise ValueError("expected four private-encoder tensors plus eight beam-head tensors")
    return result


def branch_tensor_hashes(actor, *, trainable=False):
    """Hash state tensors in either the update whitelist or frozen complement."""
    selected = set(named_branch_parameters(actor))
    result = {}
    for name, value in actor.state_dict().items():
        if (name in selected) == bool(trainable):
            array = np.ascontiguousarray(value.detach().cpu().numpy())
            result[name] = hashlib.sha256(array.tobytes()).hexdigest()
    return result


@contextmanager
def branch_training(actor):
    selected = named_branch_parameters(actor)
    parameters = dict(actor.named_parameters())
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


def _unit_beams(amplitude, phase, uavs, antennas, draws):
    return amplitude_phase_beams(amplitude.sample((int(draws),)),
                                 phase.sample((int(draws),)), uavs, antennas)


def evaluate_branch_target_gain(actor, normalized_states, teacher_g, *, draws=8,
                                seed=42, batch_size=256):
    """Validation-only score; caller explicitly supplies the eligible split."""
    teacher = np.asarray(teacher_g)
    if len(normalized_states) != len(teacher) or not len(teacher) or draws <= 0 or batch_size <= 0:
        raise ValueError("invalid evaluation rows/draws/batch size")
    uavs, antennas = teacher.shape[1:]
    if len(actor.discrete_action_dims) != uavs or actor.continuous_action_splits[BEAM_HEAD_NAMES[0]] != uavs*antennas:
        raise ValueError("evaluation teacher/actor shapes disagree")
    modes = {module: module.training for module in actor.modules()}
    device = next(actor.parameters()).device
    gains, fallbacks = [], 0
    started = time.perf_counter()
    try:
        actor.eval()
        with torch.no_grad(), isolated_torch_rng(seed, device):
            for start in range(0, len(teacher), int(batch_size)):
                amplitude, phase = branch_dists(actor, normalized_states[start:start+int(batch_size)])
                beams, bad = _unit_beams(amplitude, phase, uavs, antennas, draws)
                target = torch.as_tensor(teacher[start:start+int(batch_size)], dtype=beams.dtype, device=device)
                numerator = torch.abs(torch.sum(target.conj().unsqueeze(0)*beams, dim=-1)).square()
                denominator = torch.sum(torch.abs(target).square(), dim=-1).unsqueeze(0)
                gain = numerator/denominator.clamp_min(1e-12)
                if not torch.all(torch.isfinite(gain)) or torch.any(gain > 1.0001):
                    raise ValueError("invalid stochastic branch gain")
                gains.append(gain.clamp(0, 1).cpu().numpy())
                fallbacks += int(bad.sum().item())
    finally:
        for module, training in modes.items():
            module.training = training
    values = np.concatenate(gains, axis=1).astype(np.float64)
    return {"mean_gain": float(values.mean()), "gain_mean": float(values.mean()),
        "gain_std": float(values.std()), "per_uav_mean_gain": values.mean(axis=(0, 1)).tolist(),
        "gain_p05": float(np.quantile(values, .05)), "gain_median": float(np.median(values)),
        "gain_p95": float(np.quantile(values, .95)), "states": len(teacher),
        "draws": int(draws), "small_norm_fallback_rate": fallbacks/values.size,
        "seconds": time.perf_counter()-started}


def train_branch(actor, arrays, normalizer, *, rho=.9, kappa=64., lr=1e-3,
                 epochs=100, batch_size=256, seed=42, patience=10,
                 validation_draws=8, min_delta=1e-5, progress_callback=None):
    """Fit private encoder+beam heads; train0/validation1, never test2/-1.

    Only fixed normalized observations are cached. Trainable beam features
    are recomputed every batch and evaluation. The best epoch may be zero.
    """
    if not math.isfinite(float(lr)) or lr <= 0:
        raise ValueError("lr must be finite positive")
    if any(int(x) <= 0 for x in (epochs, batch_size, patience, validation_draws)):
        raise ValueError("epochs/batch_size/patience/draws must be positive")
    if not math.isfinite(float(min_delta)) or min_delta < 0:
        raise ValueError("min_delta must be finite nonnegative")
    selected = named_branch_parameters(actor)
    split = np.asarray(arrays["split"])
    train_indices, val_indices = np.flatnonzero(split == 0), np.flatnonzero(split == 1)
    if not len(train_indices) or not len(val_indices):
        raise ValueError("nonempty train and validation splits required")
    if arrays["raw_states"].shape[1] != actor.fc1.in_features or normalizer.mean.size != actor.fc1.in_features:
        raise ValueError("actor/state/normalizer dimensions differ")
    dimension = actor.continuous_action_splits[BEAM_HEAD_NAMES[0]]
    if dimension != int(np.prod(arrays["teacher_g_real"].shape[1:])):
        raise ValueError("teacher/actor beam dimensions differ")
    teacher = teacher_beta_parameters(arrays["teacher_g_real"][train_indices],
        arrays["teacher_g_imag"][train_indices], rho=rho, kappa=kappa)
    teacher_val = arrays["teacher_g_real"][val_indices] + 1j*arrays["teacher_g_imag"][val_indices]
    # This is the exact source normalization, not a data-set fit or a teacher
    # feature. Neither test nor excluded rows are standardized for training.
    states_train = torch.from_numpy(np.asarray(normalizer.normalize(arrays["raw_states"][train_indices]), dtype=np.float32))
    states_val = torch.from_numpy(np.asarray(normalizer.normalize(arrays["raw_states"][val_indices]), dtype=np.float32))
    t_alpha, t_beta = (torch.as_tensor(teacher[name], dtype=torch.float32) for name in ("alpha", "beta"))
    original_frozen = branch_tensor_hashes(actor)
    original_updated = branch_tensor_hashes(actor, trainable=True)
    device = next(actor.parameters()).device
    # Same shuffle/validation named domain as the U48 head-only control.
    sequence = np.random.SeedSequence(int(seed), spawn_key=(0x424D5054,))
    shuffle_sequence, val_sequence = sequence.spawn(2)
    shuffle = np.random.default_rng(shuffle_sequence)
    val_seed = int(val_sequence.generate_state(1, dtype=np.uint32)[0])
    snapshot = lambda: {name: value.detach().cpu().clone() for name, value in actor.state_dict().items()}
    history = []
    first_batch_gradient_norms = None
    started = time.perf_counter()
    with isolated_torch_rng(seed, device), branch_training(actor) as parameters:
        baseline = evaluate_branch_target_gain(actor, states_val, teacher_val,
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
                amplitude, phase = branch_dists(actor, states_train[index])
                alpha, beta = t_alpha[index].to(device), t_beta[index].to(device)
                amplitude_kl = kl_divergence(Beta(alpha[:, :dimension], beta[:, :dimension]), amplitude)
                phase_kl = kl_divergence(Beta(alpha[:, dimension:], beta[:, dimension:]), phase)
                loss = torch.cat((amplitude_kl, phase_kl), dim=-1).mean()
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite branch KL")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if first_batch_gradient_norms is None:
                    first_batch_gradient_norms = {
                        name: None if value.grad is None else float(
                            torch.linalg.vector_norm(value.grad.detach()).cpu())
                        for name, value in selected.items()}
                grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                optimizer.step()
                size = len(index)
                loss_sum += float(loss.detach().cpu())*size
                norm_sum += float(grad_norm.detach().cpu())
                seen += size
                batches += 1
                steps += 1
            validation = evaluate_branch_target_gain(actor, states_val, teacher_val,
                draws=validation_draws, seed=val_seed, batch_size=batch_size)
            if validation["mean_gain"] > best_score + float(min_delta):
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
    if branch_tensor_hashes(actor) != original_frozen:
        raise AssertionError("branch training changed frozen base/nonbeam tensors")
    changed = branch_tensor_hashes(actor, trainable=True) != original_updated
    def changed_tensors(state):
        return {name: hashlib.sha256(np.ascontiguousarray(state[name].numpy()).tobytes()).hexdigest()
                != original_updated[name] for name in selected}
    return {"contract": BRANCH_PRETRAIN_CONTRACT, "best_state_dict": best_state,
        "last_state_dict": last_state, "history": history, "best_epoch": best_epoch,
        "epochs_completed": len(history), "optimizer_steps": steps,
        "baseline_validation": baseline, "best_validation": best_metrics,
        "validation_improved": best_epoch > 0, "branch_parameters_changed": changed,
        "nonbeam_parameters_unchanged": True, "train_states": len(train_indices),
        "validation_states": len(val_indices), "test_evaluated": False,
        "excluded_rows_trained": False, "validation_seed": val_seed,
        "training_seconds": time.perf_counter()-started, "rho": float(rho),
        "kappa": float(kappa), "lr": float(lr), "batch_size": int(batch_size),
        "max_epochs": int(epochs), "patience": int(patience),
        "validation_draws": int(validation_draws), "frozen_tensor_sha256": original_frozen,
        "trainable_parameter_names": list(selected), "trainable_tensor_count": len(selected),
        "first_batch_gradient_norms_before_clip": first_batch_gradient_norms,
        "best_trainable_tensors_changed": changed_tensors(best_state),
        "last_trainable_tensors_changed": changed_tensors(last_state),
        "input_contract": "source_exact_normalized_full_raw_state"}


def load_branch_checkpoint(path, *, device="cpu"):
    """Load a self-contained NEW architecture bundle; no source model needed.

    The saved normalizer must be actor-bound at initialization version zero.
    Returns (DedicatedBeamActor, SavedStateNormalization, provenance_dict).
    """
    path = Path(path).resolve()
    directory = path if path.is_dir() else path.parent
    actor_path = directory/"mhbppo_actor.pth" if path.is_dir() else path
    metadata_path = directory/"initialization_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    architecture = metadata.get("architecture", {})
    if architecture.get("receive_beam_phase_mode", "absolute") != "absolute":
        raise ValueError("dedicated branch does not support phase_residual mapping")
    if (architecture.get("actor_class") != "DedicatedBeamActor" or
            architecture.get("architecture_contract") != BRANCH_ARCHITECTURE_CONTRACT or
            architecture.get("base_actor_strict_load_compatible") is not False):
        raise ValueError("not a declared dedicated-beam-branch initialization")
    if metadata.get("stage") != "initialization" or metadata.get("iteration") != 0 or metadata.get("weights_version") != 0:
        raise ValueError("branch bundle must declare initialization version zero")
    if architecture.get("beam_branch_input") != "source_exact_normalized_full_raw_state" or architecture.get("beam_branch_activation") != "tanh_tanh":
        raise ValueError("unknown branch input/activation contract")
    with isolated_torch_rng(0, device):
        base = MultiHeadActor(int(architecture["state_dim"]), int(architecture["hidden_dim"]),
            dict(architecture["continuous_action_splits"]), list(architecture["discrete_action_dims"]),
            continuous_dist_type=architecture["continuous_dist_type"],
            structured_beam_alpha=float(architecture["structured_beam_alpha"])).to(device)
        actor = build_branch_actor(base)
    actor.load_state_dict(torch.load(actor_path, map_location=device, weights_only=True), strict=True)
    actor.eval()
    norm = load_state_normalization(directory/"state_normalization.npz",
        expected_dim=actor.fc1.in_features, expected_iteration=0, actor_path=actor_path)
    if norm.epsilon != 1e-8:
        raise ValueError("branch normalization epsilon differs from sampler")
    expected_std = np.ones_like(norm.mean) if norm.count <= 1e-6 else np.maximum(np.sqrt(norm.m2/max(norm.count, 1e-8)), 1e-4)
    if not np.array_equal(expected_std, norm.std):
        raise ValueError("branch saved std disagrees with exact moments")
    actor_signature = actor_state_signature(actor)
    norm_signature = normalization_signature(norm.mean, norm.std, norm.epsilon)
    if metadata.get("actor_signature") != actor_signature or metadata.get("norm_signature") != norm_signature:
        raise ValueError("branch actor/normalizer signature mismatch")
    if metadata.get("actor_sha256") != file_sha256(actor_path):
        raise ValueError("branch actor file SHA mismatch")
    critic_path = directory/"mhbppo_critic.pth"
    if metadata.get("critic_sha256") != file_sha256(critic_path):
        raise ValueError("branch preserved critic SHA mismatch")
    return actor, norm, {"checkpoint": str(directory), "source_checkpoint": metadata.get("source_checkpoint"),
        "source_iteration": metadata.get("source_iteration"), "architecture": architecture,
        "actor_signature": actor_signature, "norm_signature": norm_signature,
        "normalization_count": norm.count, "metadata": metadata,
        "actor_file_sha256": file_sha256(actor_path), "critic_file_sha256": file_sha256(critic_path)}
