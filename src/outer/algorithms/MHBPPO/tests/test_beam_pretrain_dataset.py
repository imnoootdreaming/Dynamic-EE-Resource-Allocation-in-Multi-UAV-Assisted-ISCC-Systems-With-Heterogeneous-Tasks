"""Offline data recovery tests. Real state transitions, stub reward, zero solves."""
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import numpy as np

OUTER_ROOT = Path(__file__).resolve().parents[3]
if str(OUTER_ROOT) not in sys.path:
    sys.path.insert(0, str(OUTER_ROOT))

from algorithms.MHBPPO.generate_beam_pretrain_dataset import (
    DEFAULT_LOG_DIRECTORY, PHYSICAL_SOURCE_FILES, assign_round_splits,
    canonical_matched_teacher, check_physical_diagnostic, create_reconstruction_env,
    file_sha256, iter_log_window, iteration_lower_bound, parse_windows,
    reconstruct_raw_state, reconstructed_physical_metrics, restore_parameters,
    verify_source_fingerprints,
)


def fixture_metadata():
    """Synthetic complete configuration; no dependence on a large training log."""
    from configs.base_params import get_base_args
    from environment.my_env import MyEnv

    base = get_base_args(seed=42)
    madrl = dict(total_time_slots=40, hidden_dim=128, seed=42, scene_seed_offset=0,
        structured_beam_alpha=1.0, randomize_layout_per_episode=False,
        randomize_cu_traj_per_episode=False, randomize_nlos_per_episode=False,
        debug_topology=False, env_diagnostics=False, diagnostic_snapshot_limit=0,
        sensing_penalty_coef=0.0, mosek_num_threads=1)
    from types import SimpleNamespace
    with redirect_stdout(io.StringIO()):
        env = MyEnv(base, SimpleNamespace(**madrl), seed_offset=0)
    return dict(seed=42, scene_seed_offset=0, num_envs=4, base_args=base.as_dict(),
        inner_parameter_source={k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                for k, v in vars(base._params).items()},
        madrl_args=madrl, shared_exogenous_scene=env.get_exogenous_scene_signature(),
        source_sha256={name: file_sha256(OUTER_ROOT/name) for name in PHYSICAL_SOURCE_FILES})


class LogBisectionTests(unittest.TestCase):
    def test_inclusive_windows_and_frozen_boundary_rejection(self):
        self.assertEqual(parse_windows("2001:2025,1001:1025"), [(1001, 1025), (2001, 2025)])
        for text in ("1:3,3:4", "0:1", "1:3618", "3:1", "1", ""):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_windows(text)

    def test_bisection_preserves_first_last_and_all_workers_with_long_uneven_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"data.jsonl"
            offsets = {}
            with path.open("wb") as stream:
                stream.write(b'{"record_type":"metadata"}\n')
                for iteration in range(1, 61):
                    if iteration == 1:
                        stream.write(b'{"record_type":"scene","long":"' + b'x'*17001 + b'"}\n')
                    for worker in range(4):
                        for slot in range(3):
                            offsets.setdefault(iteration, stream.tell())
                            line = json.dumps(dict(record_type="action", iteration=iteration,
                                worker_id=worker, step_idx=slot, long="z"*((iteration*79+slot*137)%6103)))
                            stream.write(line.encode()+b"\n")
                # A partial concurrently appended tail must never become data.
                stream.write(b'{"record_type":"action","iteration":61')
            for target in (1, 2, 17, 30, 59, 60):
                self.assertEqual(iteration_lower_bound(path, target, record_type="action"), offsets[target])
            audit = {}
            records = list(iter_log_window(path, 17, 21, record_type="action", audit=audit))
            self.assertEqual(len(records), 5*4*3)
            self.assertEqual(records[0]["iteration"], 17)
            self.assertEqual(records[-1]["iteration"], 21)
            self.assertEqual(audit["records"], len(records))
            self.assertEqual(audit["start_byte"], offsets[17])
            self.assertEqual(audit["stop_byte"], offsets[22])
            self.assertGreater(audit["probes"], 0)

    def test_window_does_not_read_appended_records_after_snapshot_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"small.jsonl"
            path.write_text(''.join(json.dumps(dict(record_type="action", iteration=i))+"\n"
                                    for i in range(1, 7)), encoding="utf-8")
            frozen_size = path.stat().st_size
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(dict(record_type="action", iteration=7))+"\n")
            records = list(iter_log_window(path, 4, 7, record_type="action", end=frozen_size))
            self.assertEqual([r["iteration"] for r in records], [4, 5, 6])


class SplitTests(unittest.TestCase):
    def test_grouped_splits_exclude_validation_and_test_leakage(self):
        # First obtain the deterministic group assignment, using unique states.
        rounds = np.repeat(np.arange(20), 8)
        raw = np.zeros((len(rounds), 1017), dtype=np.float32)
        raw[:, 0] = np.arange(len(raw))
        _, original, _, _ = assign_round_splits(raw, rounds, seed=42)
        train = np.flatnonzero(original == 0)
        validation = np.flatnonzero(original == 1)
        test = np.flatnonzero(original == 2)
        raw[validation[0]] = raw[train[0]]
        raw[test[0]] = raw[train[1]]
        raw[test[1]] = raw[validation[1]]
        split, expected, hashes, report = assign_round_splits(raw, rounds, seed=42)
        self.assertTrue(np.array_equal(expected, original))
        self.assertEqual(split[validation[0]], -1)
        self.assertEqual(split[test[0]], -1)
        self.assertEqual(split[test[1]], -1)
        self.assertFalse(set(hashes[split == 0]) & set(hashes[split == 1]))
        self.assertFalse(set(hashes[split == 2]) & set(hashes[(original == 0) | (original == 1)]))
        for group in np.unique(rounds):
            self.assertEqual(len(set(original[rounds == group])), 1)
        self.assertEqual(report["excluded_duplicates"]["1"], 1)
        self.assertEqual(report["excluded_duplicates"]["2"], 2)
        self.assertIn("source actor/normalizer may have seen", report["holdout_scope"])


class RealEnvironmentRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metadata = fixture_metadata()

    def make_env(self):
        with redirect_stdout(io.StringIO()):
            return create_reconstruction_env(self.metadata)

    def test_native_fields_restored_and_conflicts_rejected(self):
        metadata = copy.deepcopy(self.metadata)
        metadata["inner_parameter_source"]["alpha_2"] = 2.875
        base, _ = restore_parameters(metadata)
        self.assertEqual(base.alpha_2, 2.875)  # not in BaseArgsAdapter.as_dict
        metadata["base_args"]["frac_d_lambda"] += 0.1
        with self.assertRaisesRegex(ValueError, "conflicting"):
            restore_parameters(metadata)
        missing = copy.deepcopy(self.metadata)
        del missing["inner_parameter_source"]
        with self.assertRaises(ValueError):
            restore_parameters(missing)

    def test_wrong_scene_or_physical_parameter_is_rejected(self):
        bad_scene = copy.deepcopy(self.metadata["shared_exogenous_scene"])
        bad_scene["signature"] = "0"*64
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "scene/config"):
            create_reconstruction_env(self.metadata, expected_scene=bad_scene)
        bad_parameter = copy.deepcopy(self.metadata)
        bad_parameter["inner_parameter_source"]["target_ring_move_ratio"] *= 0.9
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "scene/config"):
            create_reconstruction_env(bad_parameter)
        with self.assertRaisesRegex(ValueError, "seed"):
            restore_parameters(self.metadata, seed=1208)
        with self.assertRaisesRegex(ValueError, "offset"):
            restore_parameters(self.metadata, scene_seed_offset=1)

    def test_physical_source_fingerprints_fail_closed(self):
        checked = verify_source_fingerprints(self.metadata)
        self.assertEqual(set(checked), set(PHYSICAL_SOURCE_FILES))
        bad = copy.deepcopy(self.metadata)
        bad["source_sha256"]["environment/channel_models.py"] = "f"*64
        with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
            verify_source_fingerprints(bad)

    def test_full_40_slots_two_trajectories_equal_original_step_with_stub_reward(self):
        from environment.my_env import MyEnv
        from environment.mapping import build_matched_rec_beam

        observed_endpoints = []
        solve_calls = []
        def forbidden_solver(*args, **kwargs):
            solve_calls.append(1)
            raise AssertionError("test must never reach the real solver")

        # Patching this alias also catches an accidental restoration of MyReward.
        with patch("reward.my_reward.penalty_based_cccp", side_effect=forbidden_solver):
            for trajectory in (731, 947):
                source, rebuilt = self.make_env(), self.make_env()
                calls = []
                def stub_reward(**kwargs):
                    calls.append(1)
                    return 0.0, {"bs": 0.0, "components": {
                        "no_solution_penalty": 0.0, "uav_collision_penalty_sum": 0.0,
                        "bs_alloc_spectrum_penalty": 0.0}}, 0.0
                source.reward_calculator.reward_compute = stub_reward
                rng = np.random.default_rng(trajectory)
                state = source.reset(trajectory)["bs"]
                for slot in range(40):
                    recovered = reconstruct_raw_state(rebuilt, slot, source.cur_uavs_pos)
                    np.testing.assert_array_equal(recovered, state)
                    np.testing.assert_array_equal(rebuilt.cur_cus_pos, source.cur_cus_pos)
                    np.testing.assert_array_equal(rebuilt._get_current_uav_target_indices(),
                                                  source._get_current_uav_target_indices())
                    for name in ("uavs_2_cus_channels", "uavs_2_bs_channels", "cus_2_bs_channels",
                                 "uavs_2_targets_channels"):
                        np.testing.assert_array_equal(getattr(rebuilt, name), getattr(source, name))
                    teacher = canonical_matched_teacher(rebuilt)
                    reference = build_matched_rec_beam(rebuilt.uavs_2_targets_channels,
                                                       rebuilt._get_current_uav_target_indices())
                    np.testing.assert_allclose(abs(np.sum(teacher.conj()*reference, axis=1))**2,
                                               np.ones(4), rtol=1e-12, atol=1e-12)
                    self.assertTrue(np.all(teacher[:, 0].real > 0))
                    np.testing.assert_allclose(teacher[:, 0].imag, 0, atol=1e-15)
                    action = dict(continuous=rng.uniform(.05, .95, 102).astype(np.float32),
                                  discrete=rng.integers(0, 10, 4))
                    with redirect_stdout(io.StringIO()):
                        next_state, _, _, done, _, _ = source.step({"bs": action}, trajectory)
                    next_recovered = reconstruct_raw_state(rebuilt, slot+1, source.cur_uavs_pos)
                    np.testing.assert_array_equal(next_state["bs"], next_recovered)
                    self.assertEqual(bool(done), slot == 39)
                    state = next_state["bs"]
                self.assertEqual(len(calls), 40)
                observed_endpoints.append(source.cur_uavs_pos.copy())
        self.assertFalse(np.array_equal(*observed_endpoints))
        self.assertEqual(solve_calls, [])

    def test_teacher_sign_and_diagnostic_checks_detect_wrong_slot_or_conjugation(self):
        env = self.make_env()
        position = env.init_uavs_pos.copy()
        position[:, 0] += np.array([17, -9, 7, 23])
        position[:, 1] += np.array([31, 15, -19, 11])
        raw = reconstruct_raw_state(env, 13, position)
        from environment.observation import segment_dims
        offset = 0
        for name, length in segment_dims(env.base_args):
            if name == "rel_uav_cur_target":
                polar = raw[offset:offset+length].reshape(4, 3)
                break
            offset += length
        expected = np.exp(-2j*np.pi*env.base_args.frac_d_lambda*
                           polar[:, 1:2]*np.arange(10)[None, :])/np.sqrt(10)
        teacher = canonical_matched_teacher(env)
        np.testing.assert_allclose(teacher, expected, rtol=2e-6, atol=2e-6)
        self.assertLess(float(np.mean(abs(np.sum(teacher.conj()*teacher.conj(), axis=1))**2)), .99)
        eta = np.eye(10)[[0, 2, 4, 6]]
        physical = dict(uavs_rec_beam_vectors=dict(real=teacher.real, imag=teacher.imag),
                        uavs_cus_matched_matrix=eta,
                        uavs_targets_matched_matrix=env.uavs_targets_matched_matrix.copy(),
                        cu_off_powers=np.full(10, .07))
        metrics = reconstructed_physical_metrics(env, physical)
        # The bridge double-normalizes the outer float32-normalized beam. A
        # plausible packing error in its norm must not change inner metrics.
        rounded = copy.deepcopy(physical)
        rounded["uavs_rec_beam_vectors"] = dict(real=teacher.real*(1+2e-7),
                                                imag=teacher.imag*(1+2e-7))
        normalized_metrics = reconstructed_physical_metrics(env, rounded)
        for name in ("gamma_w", "lambda_max_g", "target_distances_m", "margin", "gain"):
            np.testing.assert_allclose(normalized_metrics[name], metrics[name], rtol=1e-12, atol=0)
        diagnostic = dict(target_indices=env._get_current_uav_target_indices().tolist(),
                          sensing={"status": "checked", **{k: metrics[k].tolist() for k in
                              ("gamma_w", "lambda_max_g", "target_distances_m", "margin")}})
        _, errors = check_physical_diagnostic(env, physical, diagnostic)
        self.assertLess(max(errors.values()), 1e-10)
        wrong = copy.deepcopy(diagnostic)
        wrong["sensing"]["gamma_w"][0] *= 1.01
        with self.assertRaisesRegex(ValueError, "reconstruction differs"):
            check_physical_diagnostic(env, physical, wrong)
        reconstruct_raw_state(env, 17, position)
        with self.assertRaisesRegex(ValueError, "schedule mismatch"):
            check_physical_diagnostic(env, physical, diagnostic)

    def test_offline_reward_guard_is_not_a_mock_of_state_recovery(self):
        env = self.make_env()
        with self.assertRaisesRegex(RuntimeError, "never call reward"):
            env.reward_calculator.reward_compute()
        for slot in (-1, 41):
            with self.assertRaises(ValueError):
                reconstruct_raw_state(env, slot, env.init_uavs_pos)


if __name__ == "__main__":
    unittest.main()
