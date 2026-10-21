"""Real environmental transitions and sampling, with only the inner reward mocked."""
import copy
from pathlib import Path
import queue
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from algorithms.MHBPPO.MHBPPO_agent import MultiHeadActor
from algorithms.MHBPPO.MHBPPO_sampler import sampler_worker, _actor_checksum
from algorithms.MHBPPO.rollout_batch import (
    actor_state_signature, assemble_rollout_batch, derive_action_seed,
    normalization_signature, validate_sync_episode,
)
from configs.base_params import get_base_args
from environment.my_env import MyEnv
from reward.my_reward import MyReward


def arguments(**overrides):
    values = dict(total_time_slots=40, structured_beam_alpha=1., hidden_dim=16,
        env_diagnostics=False, diagnostic_candidate_interval=1, diagnostic_snapshot_limit=0,
        randomize_layout_per_episode=False, randomize_cu_traj_per_episode=False,
        randomize_nlos_per_episode=False, debug_topology=False, worker_num_threads=1,
        mosek_num_threads=0, scene_seed_offset=0, num_samplers=4)
    values.update(overrides)
    return SimpleNamespace(**values)


def reward(*args, **kwargs):
    return -1.1, {"bs": -1.1, "components": {
        "solver_status_kind": "infeasible", "no_solution_penalty": 1.1,
        "base_reward": -1., "sensing_penalty_score": .1, "sensing_penalty_applied": .1,
        "sensing_penalty_status": "available"}}, float("inf")


def sampled_episodes(worker_id=0, count=2, *, args=None, legacy=False):
    args = args or arguments()
    base = get_base_args(seed=1208 if legacy else 42)
    env = MyEnv(base, args, seed_offset=1 if legacy else 0)
    torch.manual_seed(42)
    actor = MultiHeadActor(env.observation_space["bs"].shape[0], args.hidden_dim,
        env.bs_continuous_action_splits, env.bs_discrete_action_dims, structured_beam_alpha=1.)
    dim = env.observation_space["bs"].shape[0]
    mean, std = np.full(dim, .5), np.full(dim, 2.)
    weights, control = queue.Queue(), queue.Queue()
    for version in range(count):
        weights.put(dict(actor_state_dict=copy.deepcopy(actor.state_dict()), weights_version=version,
            state_norm_mean=mean, state_norm_std=std))
        control.put(version if legacy else dict(collection_round=version,
            expected_policy_version=version, episode_idx=version))
    stop = threading.Event()
    results = []
    cancelled = []
    class Samples:
        def put(self, item, timeout=None):
            results.append(item)
            if "error" in item or len(results) >= count:
                stop.set()
        def cancel_join_thread(self):
            cancelled.append(True)
    original_stdout = sys.stdout
    try:
        with patch.object(MyReward, "reward_compute", reward):
            if legacy:
                sampler_worker(base, args, Samples(), weights, stop, 1, control)
            else:
                sampler_worker(base, args, Samples(), weights, stop, 1, control,
                    worker_id=worker_id, scene_seed_offset=0)
    finally:
        if sys.stdout is not original_stdout:
            sys.stdout.close()
            sys.stdout = original_stdout
    if stop.is_set() and not cancelled:
        raise AssertionError("stopped sampler did not release its sample-queue feeder")
    if any("error" in result for result in results):
        raise AssertionError(results[-1]["error"])
    return results, actor, env.get_exogenous_scene_signature()


class SharedSceneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_all_four_envs_share_external_slot_series_across_episodes_but_uavs_diverge(self):
        envs = [MyEnv(get_base_args(seed=42), arguments(), seed_offset=0) for _ in range(4)]
        signature = envs[0].get_exogenous_scene_signature()
        self.assertIsNone(envs[0].get_diagnostic_scene())
        self.assertEqual(len(signature["component_sha256"]), 8)
        for episode in range(2):
            initial = [env.reset(episode)["bs"] for env in envs]
            for state in initial[1:]:
                np.testing.assert_array_equal(state, initial[0])
            for slot in range(41):
                for env in envs:
                    self.assertEqual(env.t, slot)
                    self.assertEqual(env.get_exogenous_scene_signature(), signature)
                    np.testing.assert_array_equal(env.cur_cus_pos, envs[0].cur_cus_pos)
                    np.testing.assert_array_equal(env._get_current_uav_target_indices(),
                                                  envs[0]._get_current_uav_target_indices())
                    for key in env.precomputed_nlos_components:
                        np.testing.assert_array_equal(env.precomputed_nlos_components[key][slot],
                                                      envs[0].precomputed_nlos_components[key][slot])
                if slot == 40:
                    break
                for worker, env in enumerate(envs):
                    continuous = np.full(sum(env.bs_continuous_action_splits.values()), .6, dtype=np.float32)
                    continuous[:env.base_args.uavs_num] = .1+worker*.2
                    with patch.object(MyReward, "reward_compute", reward):
                        env.step({"bs": {"continuous": continuous,
                                          "discrete": np.zeros(env.base_args.uavs_num, dtype=int)}}, episode)
            for env in envs[1:]:
                self.assertFalse(np.array_equal(env.cur_uavs_pos, envs[0].cur_uavs_pos))
                self.assertFalse(np.array_equal(env.uavs_2_cus_channels, envs[0].uavs_2_cus_channels))

    def test_scene_signature_detects_external_or_task_changes_but_excludes_threads(self):
        baseline = MyEnv(get_base_args(seed=42), arguments(mosek_num_threads=0), seed_offset=0)
        threaded = MyEnv(get_base_args(seed=42), arguments(mosek_num_threads=2), seed_offset=0)
        signature = baseline.get_exogenous_scene_signature()
        self.assertEqual(signature, threaded.get_exogenous_scene_signature())
        threaded.cus_entertaining_task_size[0] += 1.
        self.assertNotEqual(signature["signature"], threaded.get_exogenous_scene_signature()["signature"])
        threaded.cus_entertaining_task_size[0] -= 1.
        threaded.precomputed_nlos_components["cus_2_bs"][0,0,0] += .01
        self.assertNotEqual(signature["signature"], threaded.get_exogenous_scene_signature()["signature"])
        other_seed = MyEnv(get_base_args(seed=43), arguments(), seed_offset=0)
        self.assertNotEqual(signature["signature"], other_seed.get_exogenous_scene_signature()["signature"])

    def test_solver_thread_control_is_on_independent_environment_and_reward_parameter_copy(self):
        base = get_base_args(seed=42)
        first = MyEnv(base, arguments(mosek_num_threads=1))
        second = MyEnv(base, arguments(mosek_num_threads=2))
        self.assertIsNot(first.base_args, base)
        self.assertIsNot(first.base_args._params, base._params)
        self.assertFalse(hasattr(base, "mosek_num_threads"))
        self.assertFalse(hasattr(base._params, "mosek_num_threads"))
        self.assertEqual(first.reward_calculator.base_args.mosek_num_threads, 1)
        self.assertEqual(second.reward_calculator.base_args.mosek_num_threads, 2)

    def test_shared_sampler_40_slots_same_versions_normalization_and_independent_action_streams(self):
        all_results = []
        for worker in range(4):
            results, actor, scene = sampled_episodes(worker, count=2)
            self.assertEqual(len(results), 2)
            for round_idx, item in enumerate(results):
                self.assertEqual(validate_sync_episode(item, expected_round=round_idx,
                    expected_worker_ids=range(4), expected_policy_version=round_idx,
                    expected_actor_checksum=_actor_checksum(actor),
                    expected_actor_signature=actor_state_signature(actor),
                    expected_norm_signature=normalization_signature(item["state_norm_mean"], item["state_norm_std"]),
                    expected_scene_signature=scene, total_time_slots=40), 40)
                self.assertEqual(item["action_seed"], derive_action_seed(42, worker))
                self.assertGreater(item["collection_seconds"], 0.)
                self.assertAlmostEqual(item["collection_seconds"],
                    item["collection_finished_at"]-item["collection_started_at"])
                with torch.no_grad():
                    values = actor.evaluate_actions(torch.tensor(np.asarray(item["transition_dict"]["states"]), dtype=torch.float32),
                        torch.tensor(np.asarray(item["transition_dict"]["continuous_actions"]), dtype=torch.float32),
                        torch.tensor(np.asarray(item["transition_dict"]["discrete_actions"]), dtype=torch.long))
                np.testing.assert_allclose(values[2].numpy().ravel(), item["transition_dict"]["old_cont_log_probs"], atol=1e-4)
                np.testing.assert_allclose(values[3].numpy().ravel(), item["transition_dict"]["old_disc_log_probs"], atol=1e-4)
            self.assertEqual(results[0]["scene_signature"], results[1]["scene_signature"])
            np.testing.assert_array_equal(results[0]["transition_dict"]["raw_states"][0], results[1]["transition_dict"]["raw_states"][0])
            self.assertFalse(np.array_equal(results[0]["transition_dict"]["continuous_actions"][0],
                                            results[1]["transition_dict"]["continuous_actions"][0]))
            all_results.append(results)
        for round_idx in range(2):
            batch = assemble_rollout_batch([results[round_idx] for results in all_results])
            self.assertEqual(batch["batch_size"], 160)
            self.assertEqual([index for index, done in enumerate(batch["transition_dict"]["dones"]) if done], [39,79,119,159])
            self.assertEqual(len({results[round_idx]["scene_signature"] for results in all_results}), 1)
        reference, _, _ = sampled_episodes(0, count=1, args=arguments(num_samplers=1))
        np.testing.assert_array_equal(reference[0]["transition_dict"]["continuous_actions"],
                                      all_results[0][0]["transition_dict"]["continuous_actions"])
        for results in all_results[1:]:
            np.testing.assert_array_equal(results[0]["transition_dict"]["raw_states"][0],
                                          all_results[0][0]["transition_dict"]["raw_states"][0])
            self.assertFalse(np.array_equal(results[0]["transition_dict"]["continuous_actions"],
                                            all_results[0][0]["transition_dict"]["continuous_actions"]))

    def test_explicit_single_env_randomization_compatibility(self):
        args = arguments(num_samplers=1, randomize_nlos_per_episode=True)
        results, actor, initial = sampled_episodes(0, args=args)
        self.assertNotEqual(results[0]["scene_signature"], results[1]["scene_signature"])
        validate_sync_episode(results[1], expected_round=1, expected_worker_ids=[0],
            expected_policy_version=1, expected_actor_signature=actor_state_signature(actor),
            expected_norm_signature=results[1]["norm_signature"], expected_scene_signature=None,
            total_time_slots=40)
        with self.assertRaisesRegex(AssertionError, "fixed per-episode"):
            sampled_episodes(0, args=arguments(num_samplers=2, randomize_nlos_per_episode=True))

    def test_legacy_positional_sampler_retains_seed1209_and_scene_offset1(self):
        results, _, _ = sampled_episodes(0, count=1, legacy=True)
        self.assertEqual(results[0]["action_seed"], 1209)
        self.assertEqual(results[0]["seed_offset"], 1)
        self.assertEqual(results[0]["sampler_id"], 0)
        self.assertEqual(len(results[0]["transition_dict"]["rewards"]), 40)

    def test_error_without_stop_keeps_sentinel_flushing(self):
        stop = threading.Event()
        results, cancelled = [], []
        class Samples:
            def put(self, item, timeout=None):
                results.append(item)
            def cancel_join_thread(self):
                cancelled.append(True)
        original_stdout = sys.stdout
        try:
            sampler_worker(get_base_args(seed=42), arguments(), Samples(), queue.Queue(),
                           stop, worker_id=-1, control_queue=queue.Queue())
        finally:
            if sys.stdout is not original_stdout:
                sys.stdout.close()
                sys.stdout = original_stdout
        self.assertEqual(len(results), 1)
        self.assertIn("worker_id must be", results[0]["error"])
        self.assertFalse(stop.is_set())
        self.assertEqual(cancelled, [])


if __name__ == "__main__":
    unittest.main()
