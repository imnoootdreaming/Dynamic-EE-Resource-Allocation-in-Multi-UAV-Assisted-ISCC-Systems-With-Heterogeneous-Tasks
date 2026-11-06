"""Shared-CPU time decoding, original latent probabilities, and recipe compatibility."""
import copy
import json
import queue
import signal
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

OUTER = Path(__file__).resolve().parents[2]
if str(OUTER) not in sys.path:
    sys.path.insert(0, str(OUTER))
from algorithms.Pure_MHBPPO import Pure_MHBPPO_main as trainer
from algorithms.Pure_MHBPPO.checkpoint import load_environment_args
from algorithms.Pure_MHBPPO.pure_env import PureMyEnv
from algorithms.Pure_MHBPPO.time_decoder import decode_times, TIME_CONSTRAINTS
from algorithms.MHBPPO.rollout_batch import actor_state_signature
from configs.base_params import get_base_args
from configs.madrl_args import get_madrl_args


def timing_case():
    return dict(params=SimpleNamespace(uav_sen_duration=.1, uav_max_delay=.8,
                                      C_sen=1000., bs_max_freq=100000.),
        eta=np.array([[1., 0., 0.], [0., 1., 0.]]),
        rates={"sensed_bits": np.full(2, 10.), "uav_offloading": np.full(2, 100.),
               "cu_stage1": np.full((2, 3), 50.), "cu_stage2": np.full((2, 3), 30.),
               "cu_stage3": np.full(3, 100.)},
        task_bits=np.full(3, 10.), task_cycles=np.full(3, 1000.), task_deadlines=np.ones(3),
        z_uav=np.full(2, .5), z_cu=np.full(3, .5),
        lower_uav=np.full(2, 1e-4), upper_uav=np.full(2, .7),
        lower_cu=np.full(3, 1e-4), upper_cu=np.ones(3),
        legacy_uav=np.full(2, .35), legacy_cu=np.full(3, .5))


class TimeDecoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def assert_valid_times(self, case, result):
        self.assertTrue(result["applied"], result)
        u, c = np.array(result["uav_seconds"]), np.array(result["cu_seconds"])
        p, r = case["params"], case["rates"]
        self.assertTrue(np.all(u * r["uav_offloading"] >= r["sensed_bits"] - 1e-10))
        stage = c - case["eta"].T @ (p.uav_sen_duration + u)
        self.assertTrue(np.all(stage >= -1e-12))
        bits = np.sum(case["eta"] * (p.uav_sen_duration*r["cu_stage1"] + u[:, None]*r["cu_stage2"]), axis=0) + stage*r["cu_stage3"]
        self.assertTrue(np.all(bits >= case["task_bits"] - 1e-10))
        frequency = np.sum(p.C_sen*r["sensed_bits"]/(p.uav_max_delay-p.uav_sen_duration-u))
        frequency += np.sum(case["task_bits"]*case["task_cycles"]/(case["task_deadlines"]-c))
        self.assertLessEqual(frequency, p.bs_max_freq*(1+1e-12))
        for name, z, values in (("uav", case["z_uav"], u), ("cu", case["z_cu"], c)):
            low = np.array(result[name+"_lower_seconds"])
            high = np.array(result[name+"_upper_seconds"])
            self.assertTrue(np.all(high >= low))
            np.testing.assert_allclose(values, low+z*(high-low), atol=1e-15)

    def test_shared_cpu_and_upload_bounds_for_zeros_ones_and_random_latents(self):
        rng = np.random.default_rng(47)
        for index in range(204):
            case = timing_case()
            if index < 4:
                case["z_uav"][:] = index % 2
                case["z_cu"][:] = index // 2
            else:
                case["z_uav"] = rng.beta(.2, .2, 2)
                case["z_cu"] = rng.beta(.2, .2, 3)
            before = case["z_uav"].copy(), case["z_cu"].copy()
            self.assert_valid_times(case, decode_times(**case))
            np.testing.assert_array_equal(case["z_uav"], before[0])
            np.testing.assert_array_equal(case["z_cu"], before[1])

    def test_fallback_preserves_executed_legacy_times_and_reason(self):
        cases = []
        case = timing_case(); case["eta"][1] = [1, 0, 0]; cases.append((case, "duplicate_matching"))
        case = timing_case(); case["params"].bs_max_freq = 1; cases.append((case, "no_feasible_time_completion"))
        case = timing_case(); case["rates"]["cu_stage3"][0] = 0; cases.append((case, "zero_upload_rate"))
        case = timing_case(); case["rates"]["cu_stage1"][0, 0] = 101; cases.append((case, "nonmonotone_rates"))
        case = timing_case(); case["task_deadlines"][:] = .05; cases.append((case, "no_feasible_time_completion"))
        for case, reason in cases:
            with self.subTest(reason=reason):
                result = decode_times(**case)
                self.assertFalse(result["applied"])
                self.assertEqual(result["status"], reason)
                np.testing.assert_array_equal(result["uav_seconds"], case["legacy_uav"])
                np.testing.assert_array_equal(result["cu_seconds"], case["legacy_cu"])

    def test_new_default_saved_recipes_and_pure_only_option(self):
        self.assertEqual(trainer.get_pure_args([]).pure_time_decoder, "conditional")
        self.assertEqual(trainer.get_pure_args(["--pure_time_decoder", "legacy"]).pure_time_decoder, "legacy")
        self.assertFalse(hasattr(get_madrl_args([]), "pure_time_decoder"))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"checkpoint_metadata.json"
            path.write_text(json.dumps({"hyperparameters": {"episodes": 8000}}), encoding="utf-8")
            self.assertEqual(load_environment_args(tmp).pure_time_decoder, "legacy")
            self.assertEqual(trainer.get_pure_args(["--config", str(path), "--pure_time_decoder", "conditional"]).pure_time_decoder, "conditional")
            payload = {"hyperparameters": {"pure_time_decoder": "conditional"},
                       "action_parameterization": {"time_decoder": "conditional", "time_decoder_version": 1}}
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(load_environment_args(tmp).pure_time_decoder, "conditional")
            payload["action_parameterization"]["time_decoder_version"] = 999
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_environment_args(tmp)

    def make_env(self, mode="conditional"):
        args = trainer.get_pure_args(["--total_time_slots", "10", "--hidden_dim", "16",
                                      "--worker_num_threads", "1", "--pure_time_decoder", mode])
        base = get_base_args(seed=args.seed, target_layout=args.target_layout)
        return PureMyEnv(base, args), args, base

    def test_real_physics_extreme_latents_raw_identity_and_single_mvdr_pass(self):
        env, _, _ = self.make_env()
        rng = np.random.default_rng(123)
        applied = 0
        for index in range(104):
            env.reset(0)
            action = {"continuous": np.full(116, .5, dtype=np.float32), "discrete": rng.permutation(10)[:4]}
            action["continuous"][8:12] = index % 2 if index < 4 else rng.beta(.3, .3, 4)
            action["continuous"][-10:] = index // 2 if index < 4 else rng.beta(.3, .3, 10)
            original = action["continuous"].copy()
            with patch("numpy.linalg.solve", wraps=np.linalg.solve) as solve:
                env.step({"bs": action}, 0)
            self.assertEqual(solve.call_count, 4)
            trace = env.last_action_trace
            self.assertTrue(trace["time_decoding"]["applied"], trace["time_decoding"])
            applied += 1
            self.assertTrue(all(trace["physical"]["checks"][key]["passed"] for key in TIME_CONSTRAINTS))
            np.testing.assert_array_equal(action["continuous"], original)
            np.testing.assert_array_equal(trace["raw_continuous"], original)
            mapped, low, high = (np.array(trace[key]) for key in ("mapped_continuous", "continuous_low", "continuous_high"))
            time_indices = np.r_[8:12, 106:116]
            np.testing.assert_allclose(mapped[time_indices], (low+original*(high-low))[time_indices], atol=1e-15)
        self.assertEqual(applied, 104)

    def test_duplicate_and_impossible_cpu_keep_original_physical_penalties(self):
        env, args, base = self.make_env()
        for impossible_cpu in (False, True):
            if impossible_cpu:
                base.bs_max_freq = 1
                env = PureMyEnv(base, args)
            env.reset(0)
            legacy = copy.deepcopy(env); legacy.time_decoder = "legacy"
            action = {"continuous": np.full(116, .5, dtype=np.float32),
                      "discrete": np.arange(4) if impossible_cpu else np.zeros(4, dtype=np.int64)}
            current = env.step({"bs": action}, 0)
            old = legacy.step({"bs": action}, 0)
            self.assertFalse(env.last_action_trace["time_decoding"]["applied"])
            self.assertEqual(current[1:], old[1:])
            self.assertEqual(env.last_action_trace["physical"], legacy.last_action_trace["physical"])

    def test_conditional_mapping_changes_only_times_and_not_next_observation(self):
        env, _, _ = self.make_env(); env.reset(0)
        legacy = copy.deepcopy(env); legacy.time_decoder = "legacy"
        action = {"continuous": np.full(116, .5, dtype=np.float32), "discrete": np.arange(4)}
        current, old = env.step({"bs": action}, 0), legacy.step({"bs": action}, 0)
        np.testing.assert_array_equal(current[0]["bs"], old[0]["bs"])
        indices = np.r_[0:8, 12:106]
        np.testing.assert_array_equal(np.array(env.last_action_trace["mapped_continuous"])[indices],
                                      np.array(legacy.last_action_trace["mapped_continuous"])[indices])

    def test_real_sampler_log_prob_remains_probability_of_original_latent(self):
        env, args, base = self.make_env()
        actor = trainer.MultiHeadActor(env.observation_space["bs"].shape[0], args.hidden_dim,
                                      env.bs_continuous_action_splits, np.full(4, 10))
        control, results = queue.Queue(), queue.Queue()
        control.put({"version": 0, "actor": {k: v.detach().numpy().copy() for k,v in actor.state_dict().items()},
                     "signature": actor_state_signature(actor),
                     "normalizer": {"n": 0, "mean": np.zeros(1151), "std": np.ones(1151)}})
        control.put(None)
        signals = {getattr(signal, name): signal.getsignal(getattr(signal, name))
                   for name in ("SIGINT", "SIGBREAK") if hasattr(signal, name)}
        try:
            trainer.worker_main(base, args, 0, control, results)
        finally:
            for signum, handler in signals.items():
                signal.signal(signum, handler)
        item = results.get_nowait()
        self.assertNotIn("error", item)
        transition = item["transition"]
        self.assertTrue(any(t["time_decoding"]["applied"] for t in item["traces"]))
        for index, trace in enumerate(item["traces"]):
            np.testing.assert_array_equal(transition["continuous_actions"][index], trace["raw_continuous"])
            with torch.no_grad():
                values = actor.evaluate_actions(torch.tensor(transition["states"][index], dtype=torch.float32)[None],
                    torch.tensor(transition["continuous_actions"][index])[None],
                    torch.tensor(transition["discrete_actions"][index])[None])
            self.assertAlmostEqual(values[2].item(), transition["old_cont_log_probs"][index], places=5)
            self.assertAlmostEqual(values[3].item(), transition["old_disc_log_probs"][index], places=5)


if __name__ == "__main__":
    unittest.main()
