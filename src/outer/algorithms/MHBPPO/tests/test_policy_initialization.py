"""Weight initialization/new optimizer/architecture and rejection contracts.

Synthetic small networks only; no environment construction or solver calls.
"""
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

OUTER = Path(__file__).resolve().parents[3]
if str(OUTER) not in sys.path:
    sys.path.insert(0, str(OUTER))

from algorithms.MHBPPO.MHBPPO_agent import MultiHeadActor, Critic, HPPO
from algorithms.MHBPPO.beam_branch import build_branch_actor, DedicatedBeamActor
from algorithms.MHBPPO.beam_pretraining import isolated_torch_rng
from algorithms.MHBPPO.checkpoint import SavedStateNormalization, save_state_normalization
from algorithms.MHBPPO.rollout_batch import actor_state_signature, normalization_signature
from algorithms.MHBPPO.policy_initialization import (
    PROTECTED_BEHAVIOR_FILES, _sha, actor_architecture_metadata,
    SOURCE_BOUND_INITIALIZATION_CONTRACT,
    load_policy_initialization, apply_policy_initialization,
    restore_initial_state_normalization, convert_agent_to_dedicated,
    load_trained_policy_checkpoint,
)


SPLITS = {"uav_angles": 2, "uav_distances": 2, "uav_off_durations": 2,
          "cu_off_powers": 4, "uav_rec_beam_dir_real": 6, "uav_rec_beam_dir_imag": 6}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding="utf-8")


class Stats:
    def __init__(self, size):
        self.mean, self.m2, self.count = np.zeros(size), np.zeros(size), 1e-8

    @property
    def std(self):
        if self.count <= 1e-6:
            return np.ones_like(self.mean)
        return np.maximum(np.sqrt(self.m2/max(self.count, 1e-8)), 1e-4)


class InitializationFixture:
    def __init__(self, root, fingerprints, *, bad_behavior_fingerprint=False,
                 source_iteration=3000, format_version=1, mismatch=None):
        self.root = Path(root)
        self.source, self.experiment = self.root/f"iter_{source_iteration}", self.root/"experiment"
        self.source.mkdir(); self.experiment.mkdir()
        with isolated_torch_rng(42):
            self.actor = MultiHeadActor(7, 16, SPLITS, [4, 4])
            self.critic = Critic(7, 16)
        self.architecture = actor_architecture_metadata(self.actor)
        self.scene = {"scene_id": "a"*64, "constant_sha256": "b"*64,
            "component_sha256": {str(index): str(index)*64 for index in range(8)},
            "master_seed": 42, "scene_seed_offset": 0, "scenario_episode_idx": 0}
        import hashlib
        self.scene["signature"] = hashlib.sha256((self.scene["scene_id"]+self.scene["constant_sha256"]).encode()).hexdigest()
        m2 = np.asarray([0., 1e-12, 20., 30., 40., 50., 60.])
        std = np.maximum(np.sqrt(m2/20.), 1e-4)
        self.norm = SavedStateNormalization(np.arange(7)*.1, std, 20., m2, source_iteration)
        torch.save(self.actor.state_dict(), self.source/"mhbppo_actor.pth")
        torch.save(self.critic.state_dict(), self.source/"mhbppo_critic.pth")
        norm_iteration = source_iteration-1 if mismatch == "norm" else source_iteration
        save_state_normalization(self.source/"state_normalization.npz", self.norm, norm_iteration,
                                 actor_path=self.source/"mhbppo_actor.pth")
        fingerprints = dict(fingerprints)
        if bad_behavior_fingerprint:
            fingerprints["environment/observation.py"] = "0"*64
        source_metadata = {"iteration": source_iteration, "seed": 42, "scene_seed_offset": 0,
            "weights_version": source_iteration-1 if mismatch == "weights_version" else source_iteration,
            "checkpoint_policy": "after_completed_update",
            "actor_signature": actor_state_signature(self.actor),
            "norm_signature": normalization_signature(self.norm.mean, self.norm.std),
            "shared_exogenous_scene": self.scene, "source_sha256": fingerprints}
        write_json(self.source/"checkpoint_metadata.json", source_metadata)
        if format_version == 2:
            for name in ("training_rewards_seed_42.csv", "uav_trajectory.png",
                         "uav_trajectory_env_1.png", "uav_trajectory_env_2.png", "uav_trajectory_env_3.png"):
                (self.source/name).write_bytes(b"synthetic optional checkpoint artifact")
        self.source_hashes = {path.name: _sha(path) for path in self.source.iterdir()}
        self.environment = {"base_args": {"seed": 42, "uavs_num": 2, "radius": 600.},
            "inner_parameter_source": {"I": 2, "J": 4, "N": 3, "mosek_num_threads": 1},
            "madrl_args": {"total_time_slots": 40, "structured_beam_alpha": 1.,
                "sensing_penalty_coef": 1., "sensing_penalty_reference_factor": 10.,
                "randomize_layout_per_episode": False, "randomize_cu_traj_per_episode": False,
                "randomize_nlos_per_episode": False},
            "continuous_low": [0.]*22, "continuous_high": [1.]*22}
        dataset = self.root/"dataset.npz"
        dataset.write_bytes(b"not read as arrays by initialization; SHA binding only")
        manifest = {name: copy.deepcopy(self.environment[name]) for name in
                    ("base_args", "inner_parameter_source", "madrl_args")}
        manifest.update(dataset_path=str(dataset), dataset_sha256=_sha(dataset), scene=self.scene,
            metadata={name: self.environment[name] for name in ("continuous_low", "continuous_high")},
            source_checkpoint={"directory": str(self.source),
                "iteration": source_iteration-1 if mismatch == "data_binding" else source_iteration,
                "file_sha256": self.source_hashes})
        manifest_path = self.root/"dataset.json"
        write_json(manifest_path, manifest)
        copied_manifest = dict(manifest, _manifest_path=str(manifest_path), _dataset_path=str(dataset))
        write_json(self.experiment/"dataset_manifest.json", copied_manifest)
        for name, actor in (("original", self.actor), ("branch", build_branch_actor(self.actor))):
            directory = self.experiment/f"initial_{name}"
            directory.mkdir()
            if format_version == 2 and name == "original":
                shutil.copy2(self.source/"mhbppo_actor.pth", directory/"mhbppo_actor.pth")
            else:
                torch.save(actor.state_dict(), directory/"mhbppo_actor.pth")
            shutil.copy2(self.source/"mhbppo_critic.pth", directory/"mhbppo_critic.pth")
            save_state_normalization(directory/"state_normalization.npz", self.norm, 0,
                                     actor_path=directory/"mhbppo_actor.pth")
            metadata = {"format_version": format_version, "stage": "initialization", "iteration": 0,
                "weights_version": 0, "completed_ppo_updates": 0, "is_resume": False,
                "initialization_mode": name, "architecture": actor_architecture_metadata(actor),
                "source_checkpoint": str(self.source), "source_iteration": source_iteration,
                "source_file_sha256": self.source_hashes,
                "source_actor_signature": source_metadata["actor_signature"],
                "source_norm_signature": source_metadata["norm_signature"],
                "seed": 42, "scene_seed_offset": 0, "shared_exogenous_scene": self.scene,
                "dataset_manifest_sha256": _sha(manifest_path), "dataset_sha256": _sha(dataset),
                "normalization_count": self.norm.count, "norm_signature": source_metadata["norm_signature"],
                "actor_signature": actor_state_signature(actor), "actor_sha256": _sha(directory/"mhbppo_actor.pth"),
                "critic_sha256": _sha(directory/"mhbppo_critic.pth"),
                "action_probability_contract": actor.action_probability_metadata(),
                "fresh_optimizer_required": True, "fresh_reward_normalizer_required": True}
            if format_version == 2:
                metadata.update(initialization_contract=SOURCE_BOUND_INITIALIZATION_CONTRACT,
                    pretrain_config={"source_iteration": source_iteration-1 if mismatch == "pretrain_config" else source_iteration})
            write_json(directory/"initialization_metadata.json", metadata)

    def kwargs(self, encoder):
        return dict(expected_architecture=self.architecture, expected_scene=self.scene,
            expected_seed=42, expected_scene_seed_offset=0, expected_receive_beam_encoder=encoder,
            expected_environment=copy.deepcopy(self.environment), outer_root=OUTER)

    def load(self, arm):
        return load_policy_initialization(self.experiment/f"initial_{arm}",
            **self.kwargs("dedicated" if arm == "branch" else "shared"))


class PolicyInitializationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.fingerprints = {name: _sha(OUTER/name) for name in PROTECTED_BEHAVIOR_FILES}

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.fixture = InitializationFixture(self.temporary.name, self.fingerprints)

    def agent(self):
        with isolated_torch_rng(81):
            return HPPO(7, 16, SPLITS, [4, 4], np.zeros(22), np.ones(22),
                .003, .004, .95, .2, .99, 5, 10000, torch.device("cpu"))

    def source_fixture(self, name, **kwargs):
        directory = self.fixture.root/name
        directory.mkdir()
        return InitializationFixture(directory, self.fingerprints, **kwargs)

    def test_explicit_shared_and_dedicated_load_preserve_rng_and_exact_moments(self):
        before = torch.get_rng_state().clone()
        for arm in ("original", "branch"):
            loaded = self.fixture.load(arm)
            self.assertEqual(isinstance(loaded.actor, DedicatedBeamActor), arm == "branch")
            self.assertEqual(loaded.normalizer.iteration, 0)
            for field in ("mean", "std", "m2"):
                self.assertTrue(np.array_equal(getattr(loaded.normalizer, field), getattr(self.fixture.norm, field)))
            self.assertEqual(loaded.normalizer.count, 20.)
            self.assertFalse(loaded.provenance["is_resume"])
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_apply_unfreezes_every_parameter_and_replaces_optimizers(self):
        agent = self.agent()
        for model, optimizer in ((agent.actor, agent.actor_optimizer), (agent.critic, agent.critic_optimizer)):
            optimizer.zero_grad()
            sum(parameter.square().sum() for parameter in model.parameters()).backward()
            optimizer.step()
            self.assertTrue(optimizer.state)
        loaded = self.fixture.load("branch")
        for parameter in loaded.actor.parameters():
            parameter.requires_grad_(False)
        original_actor_optimizer, original_critic_optimizer = agent.actor_optimizer, agent.critic_optimizer
        sentinel = object(); agent.reward_statistics = sentinel
        info = apply_policy_initialization(agent, loaded)
        self.assertIs(agent.reward_statistics, sentinel)
        self.assertIsNot(agent.actor_optimizer, original_actor_optimizer)
        self.assertIsNot(agent.critic_optimizer, original_critic_optimizer)
        self.assertFalse(agent.actor_optimizer.state); self.assertFalse(agent.critic_optimizer.state)
        self.assertTrue(all(p.requires_grad for p in agent.actor.parameters()))
        self.assertTrue(all(p.requires_grad for p in agent.critic.parameters()))
        actual_ids = {id(p) for group in agent.actor_optimizer.param_groups for p in group["params"]}
        self.assertEqual(actual_ids, {id(p) for p in agent.actor.parameters()})
        self.assertEqual(agent.actor_optimizer.defaults["lr"], .003)
        self.assertEqual(agent.critic_optimizer.defaults["lr"], .004)
        self.assertTrue(info["fresh_actor_optimizer"])
        for name, value in self.fixture.critic.state_dict().items():
            self.assertTrue(torch.equal(value, agent.critic.state_dict()[name]))
        json.dumps(info)

    def test_unwarmed_conversion_keeps_policy_critic_and_rng(self):
        agent = self.agent()
        state = torch.ones(4, 7)
        before = agent.actor._get_dists(state)
        critic = {name: value.clone() for name, value in agent.critic.state_dict().items()}
        rng = torch.get_rng_state().clone()
        info = convert_agent_to_dedicated(agent)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        after = agent.actor._get_dists(state)
        for name in before[0]:
            self.assertTrue(torch.equal(before[0][name].concentration1, after[0][name].concentration1))
            self.assertTrue(torch.equal(before[0][name].concentration0, after[0][name].concentration0))
        for first, second in zip(before[1], after[1]):
            self.assertTrue(torch.equal(first.logits, second.logits))
        self.assertTrue(all(torch.equal(value, agent.critic.state_dict()[name]) for name, value in critic.items()))
        self.assertTrue(info["all_actor_parameters_trainable"])

    def test_restore_uses_original_m2_even_below_std_floor(self):
        stats = Stats(7)
        restore_initial_state_normalization(stats, self.fixture.norm)
        self.assertTrue(np.array_equal(stats.mean, self.fixture.norm.mean))
        self.assertTrue(np.array_equal(stats.m2, self.fixture.norm.m2))
        self.assertTrue(np.array_equal(stats.std, self.fixture.norm.std))
        self.assertNotEqual(stats.m2[0], stats.std[0]**2*stats.count)

    def test_reject_wrong_encoder_dimensions_and_action_order(self):
        directory = self.fixture.experiment/"initial_branch"
        wrong = self.fixture.kwargs("shared")
        with self.assertRaisesRegex(ValueError, "receive_beam_encoder"):
            load_policy_initialization(directory, **wrong)
        wrong = self.fixture.kwargs("dedicated")
        wrong["expected_architecture"] = dict(wrong["expected_architecture"], hidden_dim=32)
        with self.assertRaisesRegex(ValueError, "hidden_dim"):
            load_policy_initialization(directory, **wrong)
        wrong = self.fixture.kwargs("dedicated")
        wrong["expected_architecture"] = dict(wrong["expected_architecture"],
            continuous_action_splits=dict(reversed(list(SPLITS.items()))))
        with self.assertRaisesRegex(ValueError, "split/order"):
            load_policy_initialization(directory, **wrong)

    def test_reject_scene_seed_reward_and_action_mapping_mismatch(self):
        directory = self.fixture.experiment/"initial_original"
        for key, value in (("expected_seed", 43), ("expected_scene_seed_offset", 1)):
            wrong = self.fixture.kwargs("shared"); wrong[key] = value
            with self.assertRaises(ValueError):load_policy_initialization(directory, **wrong)
        for kind in ("scene", "reward", "bounds"):
            wrong = self.fixture.kwargs("shared")
            if kind == "scene":wrong["expected_scene"] = dict(self.fixture.scene, constant_sha256="c"*64)
            if kind == "reward":wrong["expected_environment"]["madrl_args"]["sensing_penalty_coef"] = 2.
            if kind == "bounds":wrong["expected_environment"]["continuous_high"][0] = 2.
            with self.assertRaises(ValueError):load_policy_initialization(directory, **wrong)

    def test_reject_actor_tampering_and_normalizer_hidden_moment_change(self):
        directory = self.fixture.experiment/"initial_branch"
        changed = SavedStateNormalization(self.fixture.norm.mean, self.fixture.norm.std, 20.,
            self.fixture.norm.m2+np.asarray([1e-12, 0, 0, 0, 0, 0, 0]), 0)
        save_state_normalization(directory/"state_normalization.npz", changed, 0, actor_path=directory/"mhbppo_actor.pth")
        with self.assertRaisesRegex(ValueError, "exact state normalization"):
            self.fixture.load("branch")
        actor_path = self.fixture.experiment/"initial_original/mhbppo_actor.pth"
        state = torch.load(actor_path, weights_only=True)
        state["fc1.weight"][0, 0] += 1.
        torch.save(state, actor_path)
        with self.assertRaisesRegex(ValueError, "actor/critic file SHA"):
            self.fixture.load("original")
        self.assertEqual(self.fixture.source_hashes, {path.name: _sha(path) for path in self.fixture.source.iterdir()})

    def test_reject_changed_observation_source_fingerprint(self):
        extra = self.fixture.root/"bad_behavior"; extra.mkdir()
        bad = InitializationFixture(extra, self.fingerprints, bad_behavior_fingerprint=True)
        with self.assertRaisesRegex(ValueError, "observation.py"):
            bad.load("original")

    def test_post_update_generic_loader_preserves_both_architectures(self):
        for arm in ("original", "branch"):
            loaded = self.fixture.load(arm)
            directory = self.fixture.root/f"trained_{arm}"; directory.mkdir()
            with torch.no_grad():next(loaded.actor.parameters()).add_(.01)
            actor_path = directory/"mhbppo_actor.pth"
            torch.save(loaded.actor.state_dict(), actor_path)
            torch.save(loaded.critic_state_dict, directory/"mhbppo_critic.pth")
            save_state_normalization(directory/"state_normalization.npz", loaded.normalizer, 1, actor_path=actor_path)
            metadata = {"iteration": 1, "weights_version": 1, "checkpoint_policy": "after_completed_update",
                "actor_architecture": actor_architecture_metadata(loaded.actor),
                "actor_signature": actor_state_signature(loaded.actor),
                "actor_sha256": _sha(actor_path), "critic_sha256": _sha(directory/"mhbppo_critic.pth"),
                "norm_signature": normalization_signature(loaded.normalizer.mean, loaded.normalizer.std)}
            write_json(directory/"checkpoint_metadata.json", metadata)
            actor, norm, returned = load_trained_policy_checkpoint(directory)
            self.assertEqual(type(actor), type(loaded.actor)); self.assertEqual(norm.iteration, 1)
            self.assertEqual(returned, metadata)
            self.assertTrue(all(torch.equal(value, actor.state_dict()[name]) for name, value in loaded.actor.state_dict().items()))

    def test_format2_actual_source10651_both_classes_and_legacy_stays3000(self):
        fixture = self.source_fixture("v2_good", source_iteration=10651, format_version=2)
        self.assertEqual(len(fixture.source_hashes), 9)
        for arm in ("original", "branch"):
            loaded = fixture.load(arm)
            self.assertEqual(loaded.provenance["source_iteration"], 10651)
            self.assertEqual(loaded.provenance["initialization_format_version"], 2)
            self.assertEqual(loaded.provenance["initialization_contract"], SOURCE_BOUND_INITIALIZATION_CONTRACT)
            self.assertEqual(loaded.normalizer.iteration, 0)
            self.assertEqual(loaded.normalizer.count, fixture.norm.count)
        legacy_wrong = self.source_fixture("v1_wrong_source", source_iteration=10651)
        with self.assertRaisesRegex(ValueError, "source3000-only"):
            legacy_wrong.load("original")
        self.assertEqual(self.fixture.load("original").provenance["source_iteration"], 3000)

    def test_format2_contract_and_positive_integer_source_iteration_required(self):
        fixture = self.source_fixture("v2_header", source_iteration=10651, format_version=2)
        path = fixture.experiment/"initial_original/initialization_metadata.json"
        original = json.loads(path.read_text(encoding="utf-8"))
        for contract in (None, "unknown"):
            invalid = dict(original, initialization_contract=contract)
            write_json(path, invalid)
            with self.assertRaisesRegex(ValueError, "source-bound"):
                fixture.load("original")
        for iteration in (0, -1, True, 10651.0):
            write_json(path, dict(original, source_iteration=iteration))
            with self.assertRaisesRegex(ValueError, "positive integer"):
                fixture.load("original")

    def test_format2_source_version_norm_data_and_pretrain_iteration_all_bound(self):
        for mismatch, error in (("weights_version", "completed-update weights version"),
                                ("norm", "normalization iteration mismatch"),
                                ("data_binding", "data source checkpoint identity"),
                                ("pretrain_config", "pretrain_config source iteration")):
            fixture = self.source_fixture("v2_"+mismatch, source_iteration=10651,
                format_version=2, mismatch=mismatch)
            with self.assertRaisesRegex(ValueError, error):
                fixture.load("original")

    def test_format2_all_source_files_must_be_fingerprinted(self):
        fixture = self.source_fixture("v2_missing_file", source_iteration=10651, format_version=2)
        path = fixture.experiment/"initial_original/initialization_metadata.json"
        metadata = json.loads(path.read_text(encoding="utf-8"))
        metadata["source_file_sha256"].pop("uav_trajectory_env_3.png")
        write_json(path, metadata)
        with self.assertRaisesRegex(ValueError, "every source checkpoint file"):
            fixture.load("original")

    def test_format2_original_rejects_reserialized_actor_even_with_rebound_sha(self):
        fixture = self.source_fixture("v2_bad_original", source_iteration=10651, format_version=2)
        directory = fixture.experiment/"initial_original"
        actor_path = directory/"mhbppo_actor.pth"
        state = torch.load(actor_path, weights_only=True)
        state["fc1.weight"][0, 0] += .5
        torch.save(state, actor_path)
        save_state_normalization(directory/"state_normalization.npz", fixture.norm, 0, actor_path=actor_path)
        path = directory/"initialization_metadata.json"
        metadata = json.loads(path.read_text(encoding="utf-8"))
        metadata.update(actor_sha256=_sha(actor_path), actor_signature=actor_state_signature(state))
        write_json(path, metadata)
        with self.assertRaisesRegex(ValueError, "source actor file bytes"):
            fixture.load("original")

    def test_format2_branch_rejects_changed_source_nonbeam_tensor(self):
        fixture = self.source_fixture("v2_bad_branch", source_iteration=10651, format_version=2)
        directory = fixture.experiment/"initial_branch"
        actor_path = directory/"mhbppo_actor.pth"
        state = torch.load(actor_path, weights_only=True)
        state["fc2.weight"][0, 0] += .5
        torch.save(state, actor_path)
        save_state_normalization(directory/"state_normalization.npz", fixture.norm, 0, actor_path=actor_path)
        path = directory/"initialization_metadata.json"
        metadata = json.loads(path.read_text(encoding="utf-8"))
        metadata.update(actor_sha256=_sha(actor_path), actor_signature=actor_state_signature(state))
        write_json(path, metadata)
        with self.assertRaisesRegex(ValueError, "source nonbeam tensor: fc2.weight"):
            fixture.load("branch")


if __name__ == "__main__":
    unittest.main()
