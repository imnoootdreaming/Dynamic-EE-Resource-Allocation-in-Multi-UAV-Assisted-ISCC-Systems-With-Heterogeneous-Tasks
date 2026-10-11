"""Contract tests for deterministic, adaptive target rings (S07)."""
from dataclasses import fields
import importlib.util
import json
from pathlib import Path
import pickle
import sys
import unittest

import numpy as np

INNER = Path(__file__).resolve().parents[4] / "inner"
sys.path.insert(0, str(INNER))
from target_layout import generate_sector_ring_targets
from parameters import Parameters

# Use a unique module name to avoid the outer 'environment' package collision.
_spec = importlib.util.spec_from_file_location("_s07_inner_environment", INNER / "environment.py")
_inner_environment = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _inner_environment
_spec.loader.exec_module(_inner_environment)


class SectorRingLayoutTests(unittest.TestCase):
    def geometry(self, I=4, n=10, L=600.0, **kwargs):
        return generate_sector_ring_targets(I, I * n, (12.0, -8.0, 20.0), L, **kwargs)

    def check_geometry(self, I, n, L):
        targets, sectors, info = self.geometry(I, n, L)
        centers = np.asarray(info["centers"])
        blocks = targets.reshape(I, n, 3)
        offsets = blocks[:, :, :2] - centers[:, None, :2]
        np.testing.assert_allclose(np.linalg.norm(offsets, axis=2), info["ring_radius"], rtol=1e-12)
        np.testing.assert_array_equal(sectors, np.repeat(np.arange(I), n))
        np.testing.assert_array_equal(targets[:, 2], 0.0)
        self.assertLessEqual(info["ring_radius"], info["contain_radius"])
        self.assertLess(np.max(np.linalg.norm(targets[:, :2] - [12.0, -8.0], axis=1)), L)
        if I > 1:
            angles = np.mod(np.arctan2(targets[:, 1] + 8.0, targets[:, 0] - 12.0), 2 * np.pi)
            actual_sectors = np.floor(angles / (2 * np.pi / I)).astype(int)
            np.testing.assert_array_equal(actual_sectors, sectors)
            inward = np.array([12.0, -8.0]) - centers[:, :2]
            first = offsets[:, 0, :]
            np.testing.assert_allclose(first / np.linalg.norm(first, axis=1)[:, None],
                                       inward / np.linalg.norm(inward, axis=1)[:, None], atol=1e-12)
        if n > 1:
            closed_differences = np.roll(blocks, -1, axis=1) - blocks
            distances = np.linalg.norm(closed_differences, axis=2)
            np.testing.assert_allclose(distances, info["adjacent_distance"], rtol=1e-12)
            self.assertLessEqual(info["adjacent_distance"], info["move_budget"] + 1e-12)
        else:
            self.assertIsNone(info["move_radius"])
            self.assertEqual(info["adjacent_distance"], 0.0)
        json.dumps(info, allow_nan=False)
        return info

    def test_geometry_and_budget_across_sizes_and_boundary_cases(self):
        for I, n, L in [(4, 10, 600), (4, 10, 200), (8, 10, 600),
                        (1, 10, 600), (4, 1, 600), (4, 2, 600), (2, 10, 600)]:
            with self.subTest(I=I, n=n, L=L):
                self.check_geometry(I, n, L)

    def test_default_radius_and_adaptation(self):
        standard = self.check_geometry(4, 10, 600)
        small = self.check_geometry(4, 10, 200)
        many = self.check_geometry(8, 10, 600)
        expected = 0.85 * 4 * 0.6 * 22.5 / (2 * np.sin(np.pi / 10))
        self.assertAlmostEqual(standard["ring_radius"], expected)
        self.assertLess(small["ring_radius"], standard["ring_radius"])
        self.assertLess(many["ring_radius"], standard["ring_radius"])
        self.assertAlmostEqual(small["ring_radius"], small["contain_radius"] / 3)

    def test_direction_reverses_visit_order_without_changing_points(self):
        clockwise, _, info = self.geometry(direction="clockwise")
        counterclockwise, _, other_info = self.geometry(direction="counterclockwise")
        cw = clockwise.reshape(4, 10, 3)
        ccw = counterclockwise.reshape(4, 10, 3)
        np.testing.assert_allclose(cw[:, 0], ccw[:, 0])
        np.testing.assert_allclose(cw[:, 1:], ccw[:, :0:-1], atol=1e-12)
        centers = np.asarray(info["centers"])
        first = cw[:, 0, :2] - centers[:, :2]
        second = cw[:, 1, :2] - centers[:, :2]
        cross = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
        self.assertTrue(np.all(cross < 0))
        self.assertEqual(info["ring_radius"], other_info["ring_radius"])

    def test_one_sector_uses_deployment_center_and_fixed_first_angle(self):
        targets, _, info = self.geometry(I=1, n=10)
        np.testing.assert_allclose(info["centers"], [[12, -8, 0]])
        self.assertEqual(info["contain_radius"], 600)
        np.testing.assert_allclose(targets[0], [12 + info["ring_radius"], -8, 0])

    def test_invalid_configuration_is_rejected(self):
        invalid = [{"uavs_num": 0}, {"uavs_num": 2.5}, {"uavs_num": True},
                   {"targets_num": 39}, {"targets_num": 2}, {"deploy_radius": 0},
                   {"deploy_radius": float("nan")}, {"center": [1]},
                   {"hold_slots": 0}, {"time_slot_duration": 0},
                   {"min_speed": -1}, {"max_speed": 0}, {"min_speed": 50},
                   {"space_ratio": 1}, {"space_ratio": 0}, {"move_ratio": 1.1},
                   {"move_ratio": 0}, {"direction": "random"}]
        for changes in invalid:
            with self.subTest(changes=changes):
                args = dict(uavs_num=4, targets_num=40, center=(0, 0), deploy_radius=600)
                args.update(changes)
                with self.assertRaises(ValueError):
                    generate_sector_ring_targets(**args)

    def test_inner_initialization_uses_first_targets_and_preserves_rng_draws(self):
        ring_params = Parameters()
        random_params = Parameters(target_layout="random_sector")
        ring_rng = np.random.default_rng(1208)
        random_rng = np.random.default_rng(1208)
        uavs, cus, targets = _inner_environment._generate_positions(ring_params, ring_rng)
        _, original_cus, original_targets = _inner_environment._generate_positions(random_params, random_rng)
        np.testing.assert_array_equal(cus, original_cus)
        np.testing.assert_array_equal(ring_rng.standard_normal(100), random_rng.standard_normal(100))
        np.testing.assert_allclose(uavs[:, :2], targets[::10, :2])
        np.testing.assert_array_equal(uavs[:, 2], ring_params.uav_height)
        assigned = _inner_environment._assign_nearest_targets(uavs, targets, ring_params)
        np.testing.assert_array_equal(assigned, np.arange(4) * 10)
        self.assertFalse(np.allclose(targets, original_targets))
        other_uavs, _, other_targets = _inner_environment._generate_positions(
            ring_params, np.random.default_rng(555))
        np.testing.assert_array_equal(other_targets, targets)
        np.testing.assert_array_equal(other_uavs, uavs)

    def test_inner_environment_builds_finite_channels(self):
        for I, n, L in [(4, 10, 600), (4, 10, 200), (8, 10, 600), (1, 1, 600), (4, 2, 600)]:
            with self.subTest(I=I, n=n, L=L):
                params = Parameters(I=I, K=I * n, deploy_radius=L)
                env = _inner_environment.build_environment(params, np.random.default_rng(1208))
                np.testing.assert_allclose(env.d_uav_target, params.uav_height)
                np.testing.assert_allclose(env.q_uav_pos[:, :2], env.q_target_designated[:, :2])
                for name in ("h_cu_2_uav", "h_uav_2_bs", "h_cu_2_bs", "A_theta", "Gamma_sinr"):
                    self.assertTrue(np.all(np.isfinite(getattr(env, name))), name)

    def test_window_alias_has_one_serialized_source(self):
        params = Parameters(varkappa=6)
        self.assertEqual(params.target_hold_slots, 6)
        params.target_hold_slots = 2
        self.assertEqual(params.varkappa, 2)
        params.varkappa = 3
        self.assertEqual(params.target_hold_slots, 3)
        field_names = {field.name for field in fields(Parameters)}
        self.assertIn("varkappa", field_names)
        self.assertNotIn("target_hold_slots", field_names)
        self.assertNotIn("target_hold_slots", params.__dict__)
        restored = pickle.loads(pickle.dumps(params))
        self.assertEqual(restored.target_hold_slots, 3)
        with self.assertRaises(ValueError):
            params.target_hold_slots = 0

        outer = Path(__file__).resolve().parents[3]
        sys.path.insert(0, str(outer))
        from configs.base_params import get_base_args
        args = get_base_args(target_hold_slots=2)
        self.assertEqual(args.varkappa, 2)
        self.assertEqual(args.target_hold_slots, 2)
        state = args.__getstate__()
        self.assertEqual(state["params_state"]["varkappa"], 2)
        self.assertNotIn("target_hold_slots", state["params_state"])
        adapter_restored = pickle.loads(pickle.dumps(args))
        self.assertEqual(adapter_restored.varkappa, 2)
        self.assertEqual(adapter_restored.target_hold_slots, 2)
        adapter_restored.varkappa = 5
        self.assertEqual(adapter_restored.target_hold_slots, 5)

    def test_inner_rejects_invalid_ring_and_unsafe_initial_separation(self):
        for params in (Parameters(I=0), Parameters(K=39), Parameters(deploy_radius=0.01)):
            with self.subTest(I=params.I, K=params.K, L=params.deploy_radius):
                with self.assertRaises(ValueError):
                    _inner_environment._generate_positions(params, np.random.default_rng(1208))


if __name__ == "__main__":
    unittest.main()
