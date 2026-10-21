"""Identity, frozen-input, and episode-boundary contracts for rollout aggregation."""
import copy
from pathlib import Path
import queue
import sys
import threading
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from algorithms.MHBPPO.MHBPPO_sampler import _load_weights_until
from algorithms.MHBPPO.rollout_batch import (
    actor_state_signature, assemble_rollout_batch, derive_action_seed,
    normalization_signature, validate_sync_episode,
)


def episode(worker, *, available_slots=3, disabled=False):
    count = 3
    mean = np.array([1., 2.])
    std = np.array([2., 3.])
    raw = [np.array([worker*10+t, t+1.], dtype=np.float32) for t in range(count)]
    next_raw = [value+1 for value in raw]
    action_mean = .2 if worker == 0 else .8
    transition = dict(
        states=[(value-mean)/(std+1e-8) for value in raw], raw_states=raw,
        next_states=[(value-mean)/(std+1e-8) for value in next_raw], raw_next_states=next_raw,
        continuous_actions=[[action_mean]]*count, discrete_actions=[[worker]]*count,
        rewards=[float(worker*10+t) for t in range(count)],
        old_cont_log_probs=[-.3]*count, old_disc_log_probs=[-.7]*count,
        dones=[False, False, True], real_dones=[False, False, True])
    traces = []
    for t in range(count):
        status = "disabled" if disabled else ("available" if t < available_slots else "unavailable")
        traces.append(dict(time_slot=t, clipped_action={"continuous": [action_mean]},
            outcome=dict(total_reward=transition["rewards"][t], base_reward=2.,
                         sensing_penalty_score=.2 if worker == 0 else .8,
                         sensing_penalty_status=status, sensing_penalty_applied=.1),
            policy_distribution=dict(contract="sampling_policy_conditional_moments_v1", distribution="beta",
                continuous={"power": dict(active=True, dim=1, space="unit_interval", mean=[action_mean],
                    std=[.1], entropy=[-.3], concentration=[10.])},
                discrete={"cu_matching_0": dict(probabilities=[.25,.75], entropy=.562335)})))
    return dict(worker_id=worker, sampler_id=worker, episode_idx=0, collection_round=0,
        weights_version=0, weights_checksum=2., actor_signature="actor", scene_signature="a"*64,
        exogenous_scene_signature={"signature": "a"*64},
        norm_signature=normalization_signature(mean, std), state_norm_mean=mean, state_norm_std=std,
        action_probability_metadata={"test": True}, continuous_action_splits={"power": 1},
        diagnostics_enabled=False, diagnostic_traces=[], action_traces=traces, transition_dict=transition,
        completion_rate=worker*100., finite_obj_rate=100., no_solution_rate=0., collision_rate=0.,
        spectrum_penalty_rate=0., avg_bs_reward=float(np.mean(transition["rewards"])),
        avg_total_reward=5.+worker, avg_obj_fun=1., solver_status_counts={"optimal": count},
        action_min=action_mean, action_max=action_mean, collection_seconds=2.+worker,
        seed_offset=0, scene_seed_offset=0)


def validate(item):
    return validate_sync_episode(item, expected_round=0, expected_worker_ids=[0,1],
        expected_policy_version=0, expected_actor_checksum=2., expected_actor_signature="actor",
        expected_norm_signature=normalization_signature([1,2], [2,3]),
        expected_scene_signature={"signature": "a"*64}, total_time_slots=3)


class ParallelRolloutTests(unittest.TestCase):
    def test_named_action_streams_are_stable_distinct_and_not_scene_offsets(self):
        values = [derive_action_seed(42, worker) for worker in range(4)]
        self.assertEqual(len(set(values)), 4)
        self.assertEqual(values[0], derive_action_seed(42, 0))
        self.assertNotEqual(values[0], 42)
        for value in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                derive_action_seed(42, value)

    def test_actor_signature_is_order_independent_and_detects_parameter_change(self):
        state = {"a": torch.tensor([1.,2.]), "b": torch.tensor([3.])}
        self.assertEqual(actor_state_signature(state), actor_state_signature(dict(reversed(list(state.items())))))
        changed = copy.deepcopy(state)
        changed["a"][0] += 1e-6
        self.assertNotEqual(actor_state_signature(state), actor_state_signature(changed))

    def test_episode_contract_rejects_wrong_origins_versions_inputs_and_boundaries(self):
        self.assertEqual(validate(episode(0)), 3)
        cases = [("worker_id", 9), ("sampler_id", 1), ("collection_round", 1),
                 ("episode_idx", 1), ("weights_version", 1), ("actor_signature", "wrong"),
                 ("norm_signature", "wrong"), ("scene_signature", "wrong")]
        for key, value in cases:
            with self.subTest(key=key):
                changed = episode(0)
                changed[key] = value
                with self.assertRaises(ValueError):
                    validate(changed)
        for key in ("dones", "real_dones"):
            changed = episode(0)
            changed["transition_dict"][key] = [False]*3
            with self.assertRaises(ValueError):
                validate(changed)
        changed = episode(0)
        changed["transition_dict"]["states"][0][0] += .1
        with self.assertRaisesRegex(ValueError, "policy inputs"):
            validate(changed)
        changed = episode(0)
        changed["action_traces"][1]["time_slot"] = 0
        with self.assertRaisesRegex(ValueError, "time-slot"):
            validate(changed)

    def test_aggregate_is_episode_major_with_all_terminal_and_raw_reward_boundaries(self):
        batch = assemble_rollout_batch([episode(1), episode(0)])
        self.assertEqual([item["worker_id"] for item in batch["episodes"]], [0,1])
        self.assertEqual(batch["transition_dict"]["rewards"], [0.,1.,2.,10.,11.,12.])
        self.assertEqual(batch["transition_dict"]["dones"], [False,False,True]*2)
        self.assertEqual(batch["transition_dict"]["real_dones"], [False,False,True]*2)
        self.assertIsInstance(batch["transition_dict"]["raw_states"], list)
        self.assertEqual(batch["state_norm_count"], 6)
        self.assertEqual(batch["batch_size"], 6)
        self.assertEqual(batch["completion_rate"], 50.)
        self.assertEqual(batch["avg_total_reward"], 5.5)
        self.assertEqual(batch["solver_status_counts"]["optimal"], 6)
        self.assertEqual(batch["collection_seconds"], 3.)

    def test_aggregate_recomputes_cross_env_policy_variation(self):
        batch = assemble_rollout_batch([episode(0), episode(1)])
        summary = batch["policy_distribution_summary"]
        self.assertEqual(summary["steps"], 6)
        self.assertAlmostEqual(summary["continuous"]["power"]["mean"][0], .5)
        self.assertAlmostEqual(summary["continuous"]["power"]["state_mean_std"][0], .3)

    def test_available_score_counts_weight_sensing_mean(self):
        batch = assemble_rollout_batch([episode(0), episode(1, available_slots=1)])
        self.assertAlmostEqual(batch["avg_sensing_penalty_score"], .35)
        self.assertEqual(batch["sensing_penalty_score_valid_count"], 4)
        self.assertEqual(batch["sensing_penalty_unavailable_count"], 2)
        disabled = assemble_rollout_batch([episode(0, disabled=True), episode(1, disabled=True)])
        self.assertEqual(disabled["avg_sensing_penalty_score"], 0.)

    def test_duplicate_or_different_policy_rollouts_cannot_form_a_batch(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            assemble_rollout_batch([episode(0), episode(0)])
        changed = episode(1)
        changed["weights_version"] = 1
        with self.assertRaisesRegex(ValueError, "weights_version"):
            assemble_rollout_batch([episode(0), changed])

    def test_single_randomized_scene_skips_startup_only_and_still_requires_a_scene_record(self):
        kwargs = dict(expected_round=0, expected_worker_ids=[0], expected_policy_version=0,
            expected_norm_signature=normalization_signature([1,2], [2,3]),
            expected_scene_signature=None, total_time_slots=3)
        item = episode(0)
        self.assertEqual(validate_sync_episode(item, **kwargs), 3)
        item.pop("exogenous_scene_signature")
        with self.assertRaisesRegex(ValueError, "scene signature record"):
            validate_sync_episode(item, **kwargs)
        item = episode(0)
        item["scene_signature"] = ""
        with self.assertRaisesRegex(ValueError, "SHA256"):
            validate_sync_episode(item, **kwargs)

    def test_strict_weight_loading_discards_stale_and_rejects_future_version(self):
        actor = torch.nn.Linear(2, 1)
        payload = lambda version, value: dict(weights_version=version, actor_state_dict={
            "weight": torch.full((1,2), value), "bias": torch.full((1,), value)})
        weights = queue.Queue()
        weights.put(payload(0, 0.))
        weights.put(payload(1, 1.))
        version, _, _ = _load_weights_until(weights, actor, 1, threading.Event(), strict_version=True)
        self.assertEqual(version, 1)
        self.assertEqual(float(actor.bias[0]), 1.)
        weights.put(payload(3, 3.))
        with self.assertRaisesRegex(ValueError, "future"):
            _load_weights_until(weights, actor, 2, threading.Event(), strict_version=True)
        self.assertEqual(float(actor.bias[0]), 1.)


if __name__ == "__main__":
    unittest.main()
