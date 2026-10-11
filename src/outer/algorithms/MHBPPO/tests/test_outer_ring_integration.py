"""Integration contracts for adaptive ring layout and the outer environment."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "src" / "outer"))
sys.path.insert(0, str(ROOT))
from configs.base_params import get_base_args
from environment.my_env import MyEnv
from environment.topology import generate_pos, generate_uav_target_schedule, sector_of
from src.inner.environment import _generate_positions


def arguments(total_slots=40):
    return SimpleNamespace(total_time_slots=total_slots, structured_beam_alpha=0.0,
        env_diagnostics=True, diagnostic_candidate_interval=1, diagnostic_snapshot_limit=0,
        randomize_layout_per_episode=False, randomize_cu_traj_per_episode=False,
        randomize_nlos_per_episode=False, debug_topology=False)


class OuterRingIntegrationTests(unittest.TestCase):
    def test_default_scene_has_ordered_targets_and_logged_geometry(self):
        env = MyEnv(get_base_args(), arguments(), seed_offset=1)
        self.assertEqual(env.target_layout, "sector_ring")
        self.assertAlmostEqual(env.target_layout_info["ring_radius"], 74.26776008362017)
        expected = np.arange(10)[:, None] + np.arange(4)[None, :] * 10
        np.testing.assert_array_equal(env.precomputed_uav_target_schedule[:40:4], expected)
        np.testing.assert_array_equal(env.precomputed_uav_target_schedule[-1], expected[-1])
        np.testing.assert_allclose(env.init_uavs_pos[:, :2], env.init_targets_pos[expected[0], :2])
        np.testing.assert_allclose(env.precomputed_uav_target_schedule_distances[4:40:4], 45.9, rtol=1e-6)
        self.assertTrue(np.all(np.isfinite(env.reset(0)["bs"])))
        self.assertEqual(env.get_diagnostic_scene()["target_layout"], env.target_layout_info)

    def test_region_and_uav_count_changes_recompute_radius_and_shape(self):
        for radius, count, expected_r in [(200., 4, 26.65260817079529),
                                          (600., 8, 49.72309714374089),
                                          (200., 8, 16.574365714580292)]:
            with self.subTest(radius=radius, count=count):
                env = MyEnv(get_base_args(deploy_radius=radius, I=count, K=count*10), arguments())
                self.assertAlmostEqual(env.target_layout_info["ring_radius"], expected_r)
                np.testing.assert_array_equal(sector_of(env.init_targets_pos, [0, 0], count), env.target_sector)
                self.assertEqual(env.precomputed_uav_target_schedule.shape, (41, count))
                self.assertEqual(env.bs_continuous_action_splits["uav_rec_beam_dir_real"], count*10)
                self.assertTrue(np.all(np.isfinite(env.reset(0)["bs"])))

    def test_clockwise_and_counterclockwise_reverse_visits_with_same_start(self):
        cw = MyEnv(get_base_args(), arguments())
        ccw = MyEnv(get_base_args(target_ring_direction="counterclockwise"), arguments())
        left = cw.init_targets_pos.reshape(4, 10, 3)
        right = ccw.init_targets_pos.reshape(4, 10, 3)
        np.testing.assert_allclose(left, right[:, [0,9,8,7,6,5,4,3,2,1]], atol=1e-10)
        np.testing.assert_allclose(cw.init_uavs_pos, ccw.init_uavs_pos)

    def test_layout_only_change_preserves_communication_random_stream(self):
        ring = MyEnv(get_base_args(), arguments(), seed_offset=1)
        random = MyEnv(get_base_args(target_layout="random_sector"), arguments(), seed_offset=1)
        np.testing.assert_array_equal(ring.init_cus_pos, random.init_cus_pos)
        np.testing.assert_array_equal(ring.precomputed_cus_traj, random.precomputed_cus_traj)
        for key in ring.precomputed_nlos_components:
            np.testing.assert_array_equal(ring.precomputed_nlos_components[key], random.precomputed_nlos_components[key])
        self.assertFalse(np.array_equal(ring.init_targets_pos, random.init_targets_pos))

    def test_inner_and_outer_targets_and_initial_uavs_agree(self):
        base = get_base_args()
        outer = MyEnv(base, arguments())
        inner_uav, _, inner_targets = _generate_positions(base._params, np.random.default_rng(1208))
        np.testing.assert_allclose(outer.init_targets_pos, inner_targets)
        np.testing.assert_allclose(outer.init_uavs_pos, inner_uav)

    def test_hold_slots_is_configured_and_quantity_contract_is_enforced(self):
        env = MyEnv(get_base_args(target_hold_slots=2, K=80), arguments())
        self.assertEqual(env.target_hold_slots, 2)
        self.assertEqual(env.target_layout_info["targets_per_sector"], 20)
        self.assertEqual(env._get_slots_until_switch()[0], 1)
        with self.assertRaisesRegex(ValueError, "required=80"):
            MyEnv(get_base_args(I=8, K=40), arguments())
        with self.assertRaisesRegex(ValueError, "positive integer"):
            MyEnv(get_base_args(target_hold_slots=0), arguments())

    def test_initial_collision_is_rejected_and_single_uav_works(self):
        with self.assertRaisesRegex(ValueError, "initial UAV spacing"):
            MyEnv(get_base_args(deploy_radius=1.), arguments())
        env = MyEnv(get_base_args(I=1, K=10), arguments())
        self.assertEqual(env.target_layout_info["centroid_distance"], 0.)
        self.assertTrue(np.all(np.isfinite(env.reset(0)["bs"])))

    def test_old_direct_geometry_api_keeps_random_layout_return_shape(self):
        old = generate_pos(4, 10, 40, [0,0], 600., 50., rng=np.random.default_rng(1208))
        explicit = generate_pos(4, 10, 40, [0,0], 600., 50., rng=np.random.default_rng(1208), target_layout="random_sector")
        self.assertEqual(len(old), 4)
        for a, b in zip(old, explicit):
            np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()
