"""Explicit weight initialization for a NEW PPO run, never a training resume.

No optimizer/RNG/reward statistics are restored. Default training need not
import or call this module. Offline beam modules/checkpoints remain read-only.
"""
import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from algorithms.MHBPPO.checkpoint import load_state_normalization


INITIALIZATION_CONTRACT = "explicit_policy_weights_fresh_ppo_v1"
SOURCE_BOUND_INITIALIZATION_CONTRACT = "source_checkpoint_bound_beam_initialization_v2"
SHARED_ARCHITECTURE_CONTRACT = "shared_receive_beam_features_v1"
OUTER_ROOT = Path(__file__).resolve().parents[2]
# Main/sampler/config/log plumbing may change for this explicit new feature.
# Input semantics, action mapping, reward and physical computation may not.
PROTECTED_BEHAVIOR_FILES = (
    "algorithms/MHBPPO/MHBPPO_agent.py", "configs/base_params.py", "configs/seed_manager.py",
    "environment/my_env.py", "environment/channel_models.py", "environment/topology.py",
    "environment/observation.py", "environment/mapping.py", "environment/cu_tasks.py", "reward/my_reward.py",
    "reward/cccp_bridge.py", "reward/feasibility_diagnostics.py", "reward/solver_status.py",
    "reward/sensing_penalty.py", "reward/episode_success_bonus.py",
    "algorithms/MHBPPO/checkpoint.py",
    "algorithms/MHBPPO/policy_diagnostics.py", "algorithms/MHBPPO/rollout_batch.py",
    "../inner/parameters.py", "../inner/environment.py", "../inner/target_layout.py",
    "../inner/pc3p.py", "../inner/fusion_pc3p.py", "../inner/variables.py", "../inner/cccp_params.py",
)
ENVIRONMENT_MADRL_FIELDS = (
    "total_time_slots", "structured_beam_alpha", "sensing_penalty_coef",
    "sensing_penalty_reference_factor", "episode_success_bonus",
    "cu_task_profile", "cu_task_l_kbit_min", "cu_task_l_kbit_max",
    "cu_task_c_min", "cu_task_c_max", "cu_task_d_min", "cu_task_d_max",
    "cu_task_slot_duration",
    "randomize_layout_per_episode",
    "randomize_cu_traj_per_episode", "randomize_nlos_per_episode",
)
RESOURCE_FIELDS = frozenset({"mosek_num_threads", "worker_num_threads"})


@dataclass(frozen=True)
class LoadedPolicyInitialization:
    actor: object
    critic_state_dict: dict
    normalizer: object
    metadata: dict
    provenance: dict


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _json_value(value):
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _same(first, second, label):
    if _json_value(first) != _json_value(second):
        raise ValueError(f"policy initialization {label} mismatch")


def _base_architecture(architecture):
    if architecture.get("receive_beam_phase_mode", "absolute") != "absolute":
        raise ValueError("offline initialization/evaluation supports absolute phases only; phase_residual is unsupported")
    required = ("state_dim", "hidden_dim", "continuous_action_splits", "discrete_action_dims",
                "continuous_dist_type", "structured_beam_alpha")
    if any(name not in architecture for name in required):
        raise ValueError("incomplete policy architecture metadata")
    result = {name: copy.deepcopy(architecture[name]) for name in required}
    if result["continuous_dist_type"] != "beta" or float(result["structured_beam_alpha"]) != 1.0:
        raise ValueError("approved beam initializations require Beta and structured_beam_alpha=1")
    if any(int(result[name]) <= 0 or int(result[name]) != result[name] for name in ("state_dim", "hidden_dim")):
        raise ValueError("invalid policy state/hidden dimensions")
    splits = result["continuous_action_splits"]
    if not isinstance(splits, dict) or not splits or any(int(x) != x or x <= 0 for x in splits.values()):
        raise ValueError("invalid continuous action splits")
    discrete = result["discrete_action_dims"]
    if not len(discrete) or any(int(x) != x or x <= 0 for x in discrete):
        raise ValueError("invalid discrete action dimensions")
    result["discrete_action_dims"] = list(discrete)
    return result


def actor_architecture_metadata(actor):
    from algorithms.MHBPPO.beam_branch import DedicatedBeamActor
    if isinstance(actor, DedicatedBeamActor):
        return actor.architecture_metadata()
    from algorithms.MHBPPO.MHBPPO_agent import MultiHeadActor
    if type(actor) is not MultiHeadActor:
        raise TypeError("unknown actor architecture")
    return {"actor_class": "MultiHeadActor", "architecture_contract": SHARED_ARCHITECTURE_CONTRACT,
        "state_dim": actor.fc1.in_features, "hidden_dim": actor.fc1.out_features,
        "continuous_action_splits": dict(actor.continuous_action_splits),
        "discrete_action_dims": list(actor.discrete_action_dims),
        "continuous_dist_type": actor.continuous_dist_type,
        "structured_beam_alpha": actor.structured_beam_alpha_effective,
        "base_actor_strict_load_compatible": True,
        **(actor.receive_beam_mapping if actor.receive_beam_mapping[
            "receive_beam_phase_mode"] == "phase_residual" or actor.receive_beam_mapping[
            "receive_beam_mapping_version"] in ("mvdr_complex_residual_v1", "mvdr_aligned_mix_v1", "matched_unit_norm_v1") else {})}


def build_actor_for_architecture(architecture, *, device="cpu"):
    """RNG-isolated base/branch construction, independent of offline freeze."""
    from algorithms.MHBPPO.MHBPPO_agent import MultiHeadActor
    from algorithms.MHBPPO.beam_branch import BRANCH_ARCHITECTURE_CONTRACT, build_branch_actor
    from algorithms.MHBPPO.beam_pretraining import isolated_torch_rng
    base = _base_architecture(architecture)
    actor_class = architecture.get("actor_class", "MultiHeadActor")
    if actor_class not in ("MultiHeadActor", "DedicatedBeamActor"):
        raise ValueError("unsupported actor class")
    if actor_class == "DedicatedBeamActor" and architecture.get("architecture_contract") != BRANCH_ARCHITECTURE_CONTRACT:
        raise ValueError("unknown dedicated-beam architecture contract")
    if actor_class == "DedicatedBeamActor" and (
            architecture.get("beam_branch_input") != "source_exact_normalized_full_raw_state" or
            architecture.get("beam_branch_activation") != "tanh_tanh" or
            architecture.get("base_actor_strict_load_compatible") is not False):
        raise ValueError("incompatible dedicated-beam input/activation/layout contract")
    if actor_class == "MultiHeadActor" and architecture.get("architecture_contract", SHARED_ARCHITECTURE_CONTRACT) != SHARED_ARCHITECTURE_CONTRACT:
        raise ValueError("unknown shared-beam architecture contract")
    with isolated_torch_rng(0, device):
        actor = MultiHeadActor(**base).to(device)
        if actor_class == "DedicatedBeamActor":
            actor = build_branch_actor(actor)
    for parameter in actor.parameters():
        parameter.requires_grad_(True)
        parameter.grad = None
    return actor


def _effective_std(norm):
    if norm.count <= 1e-6:
        return np.ones_like(norm.mean)
    return np.maximum(np.sqrt(norm.m2/max(norm.count, 1e-8)), 1e-4)


def restore_initial_state_normalization(stats, normalizer):
    """Copy exact moments/count; never infer m2 from floored saved std."""
    if normalizer.epsilon != 1e-8 or not np.array_equal(_effective_std(normalizer), normalizer.std):
        raise ValueError("initialization normalization is not the sampler's exact Welford transform")
    if np.asarray(stats.mean).shape != normalizer.mean.shape:
        raise ValueError("target state normalizer dimension mismatch")
    stats.mean = np.asarray(normalizer.mean, dtype=np.float64).copy()
    stats.m2 = np.asarray(normalizer.m2, dtype=np.float64).copy()
    stats.count = float(normalizer.count)
    if not np.array_equal(stats.std, normalizer.std):
        raise ValueError("restored state normalizer effective std mismatch")


def _validate_environment(manifest, expected):
    if not isinstance(expected, dict):
        raise ValueError("explicit current environment configuration is required")
    for name in ("base_args", "inner_parameter_source"):
        saved, actual = manifest.get(name), expected.get(name)
        if not isinstance(saved, dict) or not isinstance(actual, dict):
            raise ValueError(f"missing {name} configuration snapshot")
        _same({key: value for key, value in saved.items() if key not in RESOURCE_FIELDS},
              {key: value for key, value in actual.items() if key not in RESOURCE_FIELDS}, name)
    saved_madrl, actual_madrl = manifest.get("madrl_args", {}), expected.get("madrl_args", {})
    for name in ENVIRONMENT_MADRL_FIELDS:
        if name not in saved_madrl or name not in actual_madrl:
            raise ValueError(f"missing environment setting {name}")
        _same(saved_madrl[name], actual_madrl[name], name)
        if name.startswith("randomize_") and bool(actual_madrl[name]):
            raise ValueError("fixed-scene beam initialization requires randomize flags false")
    for name in ("continuous_low", "continuous_high"):
        if name not in manifest.get("metadata", {}) or name not in expected:
            raise ValueError(f"missing action mapping {name}")
        _same(manifest["metadata"][name], expected[name], name)


def load_policy_initialization(directory, *, expected_architecture, expected_scene,
                               expected_seed, expected_scene_seed_offset,
                               expected_receive_beam_encoder="shared",
                               expected_environment=None, outer_root=None, device="cpu"):
    """Validate an init package, then return weights/exactnorm BEFORE first v0.

    expected_environment contains base_args, inner_parameter_source, madrl_args
    and continuous_low/high. PPO/resource settings may differ; physical/input/
    reward settings and original implementation fingerprints must agree.
    Legacy format1 remains source3000-only. Explicit format2 binds a positive
    source iteration consistently across checkpoint, data and pretraining.
    """
    import torch
    from algorithms.MHBPPO.MHBPPO_agent import Critic
    from algorithms.MHBPPO.beam_pretraining import isolated_torch_rng
    from algorithms.MHBPPO.rollout_batch import actor_state_signature, normalization_signature
    directory = Path(directory).resolve()
    metadata_path = directory/"initialization_metadata.json"
    metadata = _read_json(metadata_path)
    format_version = metadata.get("format_version")
    if type(format_version) is not int or format_version not in (1, 2):
        raise ValueError("unsupported initialization format version")
    if format_version == 1:
        source_iteration = 3000
    else:
        if metadata.get("initialization_contract") != SOURCE_BOUND_INITIALIZATION_CONTRACT:
            raise ValueError("format2 requires the source-bound initialization contract")
        source_iteration = metadata.get("source_iteration")
        if isinstance(source_iteration, bool) or not isinstance(source_iteration, int) or source_iteration <= 0:
            raise ValueError("format2 source_iteration must be a positive integer")
        pretrain_source_iteration = metadata.get("pretrain_config", {}).get("source_iteration")
        if type(pretrain_source_iteration) is not int or pretrain_source_iteration != source_iteration:
            raise ValueError("pretrain_config source iteration mismatch")
        if any(type(metadata.get(name)) is not int for name in ("iteration", "weights_version", "completed_ppo_updates")):
            raise ValueError("format2 initialization counters must be integer zero")
    if (metadata.get("stage") != "initialization" or
            any(metadata.get(name) != 0 for name in ("iteration", "weights_version", "completed_ppo_updates")) or
            metadata.get("is_resume") is not False):
        raise ValueError("policy package must declare new-run initialization version zero")
    if metadata.get("initialization_mode") not in ("original", "head_only", "branch"):
        raise ValueError("unsupported U49 initialization mode")
    if metadata.get("fresh_optimizer_required") is not True or metadata.get("fresh_reward_normalizer_required") is not True:
        raise ValueError("initialization must explicitly require fresh PPO optimizer/reward statistics")
    if expected_receive_beam_encoder not in ("shared", "dedicated"):
        raise ValueError("receive_beam_encoder must be shared or dedicated")
    architecture = metadata.get("architecture", {})
    actor_class = architecture.get("actor_class")
    wanted_class = "DedicatedBeamActor" if expected_receive_beam_encoder == "dedicated" else "MultiHeadActor"
    if actor_class != wanted_class:
        raise ValueError("initialization actor class conflicts with receive_beam_encoder")
    actual_base, expected_base = _base_architecture(architecture), _base_architecture(expected_architecture)
    for name in actual_base:
        if name == "continuous_action_splits":
            _same(list(actual_base[name].items()), list(expected_base[name].items()), "action split/order")
        else:
            _same(actual_base[name], expected_base[name], name)
    _same(metadata.get("seed"), int(expected_seed), "master seed")
    _same(metadata.get("scene_seed_offset"), int(expected_scene_seed_offset), "scene seed offset")
    scene = metadata.get("shared_exogenous_scene")
    components = scene.get("component_sha256") if isinstance(scene, dict) else None
    expected_components = (expected_scene.get("component_sha256")
                           if isinstance(expected_scene, dict) else None)
    if (not isinstance(components, dict) or not isinstance(expected_components, dict)
            or not expected_components or set(components) != set(expected_components)
            or "cu_velocities" not in expected_components):
        raise ValueError("complete shared scene signature is required")
    _same(scene, expected_scene, "external scene/time series/physical constants")
    if (not isinstance(scene.get("signature"), str) or len(scene["signature"]) != 64 or
            scene["signature"] != hashlib.sha256((scene["scene_id"]+scene["constant_sha256"]).encode()).hexdigest()):
        raise ValueError("inconsistent scene signature")

    source = Path(metadata["source_checkpoint"]).resolve()
    source_hashes = metadata.get("source_file_sha256", {})
    if not source_hashes:
        raise ValueError("missing read-only source checkpoint fingerprints")
    required_source_files = {"mhbppo_actor.pth", "mhbppo_critic.pth", "state_normalization.npz", "checkpoint_metadata.json"}
    if format_version == 2 and (set(source_hashes) != {path.name for path in source.iterdir() if path.is_file()} or
                                not required_source_files.issubset(source_hashes)):
        raise ValueError("format2 source fingerprints must cover every source checkpoint file")
    for name, fingerprint in source_hashes.items():
        if Path(name).name != name or _sha(source/name) != fingerprint:
            raise ValueError(f"source checkpoint file changed: {name}")
    source_metadata = _read_json(source/"checkpoint_metadata.json")
    if metadata.get("source_iteration") != source_iteration or source_metadata.get("iteration") != source_iteration:
        raise ValueError("source checkpoint iteration mismatch; format1 remains source3000-only")
    if format_version == 2 and (type(source_metadata.get("iteration")) is not int or
                                type(source_metadata.get("weights_version")) is not int or
                                source_metadata["weights_version"] != source_iteration or
                                source_metadata.get("checkpoint_policy") != "after_completed_update"):
        raise ValueError("format2 source must be the declared completed-update weights version")
    if format_version == 2:
        _same(source_metadata.get("seed"), int(expected_seed), "source master seed")
        _same(source_metadata.get("scene_seed_offset"), int(expected_scene_seed_offset), "source scene seed offset")
    _same(source_metadata.get("actor_signature"), metadata.get("source_actor_signature"), "source actor signature")
    _same(source_metadata.get("norm_signature"), metadata.get("source_norm_signature"), "source norm signature")
    _same(source_metadata.get("shared_exogenous_scene"), scene, "source scene")
    manifest_path = directory.parent/"dataset_manifest.json"
    manifest = _read_json(manifest_path)
    original_manifest_path = Path(manifest["_manifest_path"])
    if _sha(original_manifest_path) != metadata.get("dataset_manifest_sha256"):
        raise ValueError("original dataset manifest SHA mismatch")
    original_manifest = _read_json(original_manifest_path)
    _same(original_manifest.get("source_checkpoint"), manifest.get("source_checkpoint"), "data source binding")
    _same(original_manifest.get("scene"), scene, "data scene")
    dataset_path = Path(manifest.get("_dataset_path", manifest.get("dataset_path", "")))
    if _sha(dataset_path) != metadata.get("dataset_sha256") or manifest.get("dataset_sha256") != metadata.get("dataset_sha256"):
        raise ValueError("initialization dataset SHA mismatch")
    binding = manifest.get("source_checkpoint", {})
    if Path(binding.get("directory", "")).resolve() != source or binding.get("iteration") != source_iteration:
        raise ValueError("data source checkpoint identity mismatch")
    if format_version == 2 and type(binding.get("iteration")) is not int:
        raise ValueError("format2 data binding iteration must be an integer")
    for name in ("mhbppo_actor.pth", "mhbppo_critic.pth", "state_normalization.npz", "checkpoint_metadata.json"):
        if binding.get("file_sha256", {}).get(name) != source_hashes.get(name):
            raise ValueError(f"data/source SHA binding mismatch: {name}")
    _validate_environment(original_manifest, expected_environment)
    root = OUTER_ROOT if outer_root is None else Path(outer_root).resolve()
    protected = source_metadata.get("source_sha256", {})
    for name in PROTECTED_BEHAVIOR_FILES:
        if protected.get(name) != _sha(root/name):
            raise ValueError(f"input/action/reward/physics source changed: {name}")

    actor_path, critic_path = directory/"mhbppo_actor.pth", directory/"mhbppo_critic.pth"
    if _sha(actor_path) != metadata.get("actor_sha256") or _sha(critic_path) != metadata.get("critic_sha256"):
        raise ValueError("initialization actor/critic file SHA mismatch")
    if metadata["critic_sha256"] != source_hashes["mhbppo_critic.pth"]:
        raise ValueError("initialization critic differs from source critic")
    normalizer = load_state_normalization(directory/"state_normalization.npz",
        expected_dim=actual_base["state_dim"], expected_iteration=0, actor_path=actor_path)
    source_norm = load_state_normalization(source/"state_normalization.npz",
        expected_dim=actual_base["state_dim"], expected_iteration=source_iteration, actor_path=source/"mhbppo_actor.pth")
    if format_version == 2:
        _same(metadata.get("source_norm_signature"), normalization_signature(source_norm.mean, source_norm.std, source_norm.epsilon), "actual source norm signature")
    if (normalizer.epsilon != 1e-8 or not np.array_equal(_effective_std(normalizer), normalizer.std) or
            normalizer.count != source_norm.count or normalizer.epsilon != source_norm.epsilon or
            any(not np.array_equal(getattr(normalizer, name), getattr(source_norm, name)) for name in ("mean", "std", "m2"))):
        raise ValueError("initialization must preserve source exact state normalization")
    _same(metadata.get("normalization_count"), normalizer.count, "normalizer count")
    _same(metadata.get("norm_signature"), normalization_signature(normalizer.mean, normalizer.std, normalizer.epsilon), "normalizer signature")
    actor = build_actor_for_architecture(architecture, device=device)
    actor.load_state_dict(torch.load(actor_path, map_location=device, weights_only=True), strict=True)
    _same(metadata.get("actor_signature"), actor_state_signature(actor), "actor tensor signature")
    if format_version == 2:
        source_actor_state = torch.load(source/"mhbppo_actor.pth", map_location=device, weights_only=True)
        _same(metadata.get("source_actor_signature"), actor_state_signature(source_actor_state), "actual source actor tensor signature")
        if metadata["initialization_mode"] == "original":
            if actor_class != "MultiHeadActor" or _sha(actor_path) != source_hashes["mhbppo_actor.pth"]:
                raise ValueError("format2 original actor must preserve source actor file bytes")
            _same(actor_state_signature(actor), metadata["source_actor_signature"], "original/source actor tensors")
        else:
            if metadata["initialization_mode"] == "branch" and actor_class != "DedicatedBeamActor":
                raise ValueError("format2 branch mode requires the dedicated actor class")
            allowed = ("continuous_heads.uav_rec_beam_amplitude.", "continuous_heads.uav_rec_beam_phase.")
            actual_state = actor.state_dict()
            for name, value in source_actor_state.items():
                if not name.startswith(allowed) and (name not in actual_state or not torch.equal(value, actual_state[name])):
                    raise ValueError(f"format2 pretrained actor changed a source nonbeam tensor: {name}")
    _same(metadata.get("action_probability_contract"), actor.action_probability_metadata(), "action probability contract")
    critic_state = torch.load(critic_path, map_location=device, weights_only=True)
    with isolated_torch_rng(0, device):
        critic_probe = Critic(actual_base["state_dim"], actual_base["hidden_dim"]).to(device)
    critic_probe.load_state_dict(critic_state, strict=True)
    provenance = {"contract": INITIALIZATION_CONTRACT, "directory": str(directory),
        "mode": metadata["initialization_mode"], "receive_beam_encoder": expected_receive_beam_encoder,
        "actor_architecture": actor_architecture_metadata(actor), "is_resume": False,
        "initialization_format_version": format_version,
        "initialization_contract": metadata.get("initialization_contract"),
        "initial_completed_ppo_updates": 0, "initial_weights_version": 0,
        "source_checkpoint": str(source), "source_iteration": source_iteration,
        "actor_signature": metadata["actor_signature"], "norm_signature": metadata["norm_signature"],
        "source_actor_signature": metadata["source_actor_signature"],
        "source_normalization_count": normalizer.count,
        "dataset_sha256": metadata["dataset_sha256"], "dataset_manifest_sha256": metadata["dataset_manifest_sha256"],
        "actor_sha256": _sha(actor_path), "critic_sha256": _sha(critic_path),
        "normalization_sha256": _sha(directory/"state_normalization.npz"),
        "initialization_metadata_sha256": _sha(metadata_path),
        "source_files_verified": dict(source_hashes), "protected_behavior_files_verified": list(PROTECTED_BEHAVIOR_FILES),
        "fresh_optimizer_required": True, "fresh_reward_normalizer_required": True}
    return LoadedPolicyInitialization(actor, critic_state, normalizer, metadata, provenance)


def _all_trainable_fresh_adam(agent):
    import torch
    if not isinstance(agent.actor_optimizer, torch.optim.Adam) or not isinstance(agent.critic_optimizer, torch.optim.Adam):
        raise TypeError("policy initialization expects the existing HPPO Adam optimizers")
    actor_defaults, critic_defaults = copy.deepcopy(agent.actor_optimizer.defaults), copy.deepcopy(agent.critic_optimizer.defaults)
    for model in (agent.actor, agent.critic):
        model.train()
        for parameter in model.parameters():
            parameter.requires_grad_(True)
            parameter.grad = None
    agent.actor_optimizer = torch.optim.Adam(agent.actor.parameters(), **actor_defaults)
    agent.critic_optimizer = torch.optim.Adam(agent.critic.parameters(), **critic_defaults)
    return {"all_actor_parameters_trainable": all(parameter.requires_grad for parameter in agent.actor.parameters()),
        "all_critic_parameters_trainable": all(parameter.requires_grad for parameter in agent.critic.parameters()),
        "actor_trainable_tensors": sum(1 for _ in agent.actor.parameters()),
        "critic_trainable_tensors": sum(1 for _ in agent.critic.parameters()),
        "fresh_actor_optimizer": not bool(agent.actor_optimizer.state),
        "fresh_critic_optimizer": not bool(agent.critic_optimizer.state),
        "actor_lr": agent.actor_optimizer.defaults["lr"], "critic_lr": agent.critic_optimizer.defaults["lr"]}


def apply_policy_initialization(agent, loaded):
    """Install paired weights and reset Adam; no counters/reward stats touched."""
    actual, expected = _base_architecture(actor_architecture_metadata(loaded.actor)), _base_architecture(actor_architecture_metadata(agent.actor))
    for name in actual:
        _same(list(actual[name].items()) if name == "continuous_action_splits" else actual[name],
              list(expected[name].items()) if name == "continuous_action_splits" else expected[name], name)
    agent.actor = loaded.actor.to(agent.device)
    agent.critic.load_state_dict(loaded.critic_state_dict, strict=True)
    optimizer_info = _all_trainable_fresh_adam(agent)
    return {**loaded.provenance, **optimizer_info}


def convert_agent_to_dedicated(agent):
    """Unwarmed SAME-architecture control: copy source layers, then full PPO."""
    from algorithms.MHBPPO.beam_branch import build_branch_actor
    agent.actor = build_branch_actor(agent.actor).to(agent.device)
    return {"contract": INITIALIZATION_CONTRACT, "mode": "dedicated_unwarmed",
        "actor_architecture": actor_architecture_metadata(agent.actor),
        "is_resume": False, **_all_trainable_fresh_adam(agent)}


def load_trained_policy_checkpoint(directory, *, device="cpu"):
    """Read a post-update base/branch snapshot for evaluation, never resume.

    The actor file is bound by the saved normalizer and tensor SHA. No
    optimizer/RNG/reward-normalizer state is inferred or fabricated.
    """
    import torch
    from algorithms.MHBPPO.MHBPPO_agent import Critic
    from algorithms.MHBPPO.beam_pretraining import isolated_torch_rng
    from algorithms.MHBPPO.rollout_batch import actor_state_signature, normalization_signature
    directory = Path(directory).resolve()
    metadata = _read_json(directory/"checkpoint_metadata.json")
    iteration = metadata.get("iteration")
    if (isinstance(iteration, bool) or not isinstance(iteration, int) or iteration <= 0 or
            metadata.get("weights_version") != iteration or
            metadata.get("checkpoint_policy") != "after_completed_update"):
        raise ValueError("expected a positive completed-update checkpoint")
    architecture = metadata.get("actor_architecture", metadata.get("architecture"))
    if not isinstance(architecture, dict):
        raise ValueError("trained checkpoint requires explicit actor architecture")
    actor = build_actor_for_architecture(architecture, device=device)
    actor_path = directory/"mhbppo_actor.pth"
    if metadata.get("actor_sha256") != _sha(actor_path):
        raise ValueError("post-update actor file SHA mismatch")
    actor.load_state_dict(torch.load(actor_path, map_location=device, weights_only=True), strict=True)
    norm = load_state_normalization(directory/"state_normalization.npz",
        expected_dim=actor.fc1.in_features, expected_iteration=iteration, actor_path=actor_path)
    if norm.epsilon != 1e-8 or not np.array_equal(_effective_std(norm), norm.std):
        raise ValueError("trained checkpoint normalization transform mismatch")
    _same(metadata.get("actor_signature"), actor_state_signature(actor), "post-update actor signature")
    _same(metadata.get("norm_signature"), normalization_signature(norm.mean, norm.std, norm.epsilon), "post-update norm signature")
    if "action_probability_contract" in metadata and isinstance(metadata["action_probability_contract"], dict):
        _same(metadata["action_probability_contract"], actor.action_probability_metadata(), "post-update probability contract")
    with isolated_torch_rng(0, device):
        critic_probe = Critic(actor.fc1.in_features, actor.fc1.out_features).to(device)
    critic_path = directory/"mhbppo_critic.pth"
    critic_probe.load_state_dict(torch.load(critic_path, map_location=device, weights_only=True), strict=True)
    if metadata.get("critic_sha256") != _sha(critic_path):
        raise ValueError("post-update critic file SHA mismatch")
    actor.eval()
    return actor, norm, metadata
