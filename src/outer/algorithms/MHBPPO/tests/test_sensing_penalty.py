"""Numerical and configuration boundaries for the bounded sensing penalty."""

from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


OUTER = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(OUTER))
from reward.sensing_penalty import (
    SENSING_PENALTY_CONTRACT,
    normalized_sensing_shortfall,
    validate_sensing_penalty_settings,
)
from configs.madrl_args import get_madrl_args
from configs.base_params import get_base_args
from reward.my_reward import MyReward


class SensingPenaltyNumericalTests(unittest.TestCase):
    def test_power_excess_examples_and_reference_half_score(self):
        mean_score, scores = normalized_sensing_shortfall([0.5, 0.1, 0.01, 0.001])
        expected = [np.log(2) / np.log(20), 0.5, 2 / 3, 0.75]
        np.testing.assert_allclose(scores, expected, rtol=1e-14)
        self.assertAlmostEqual(mean_score, np.mean(expected))
        for factor in (2.0, 10.0, 100.0):
            self.assertAlmostEqual(normalized_sensing_shortfall([1 / factor], factor)[0], 0.5)

    def test_zero_and_satisfied_sensing_conditions(self):
        mean_score, scores = normalized_sensing_shortfall([0.0, 1.0, 2.0])
        np.testing.assert_array_equal(scores, [1.0, 0.0, 0.0])
        self.assertAlmostEqual(mean_score, 1 / 3)

    def test_score_is_bounded_and_monotone_with_improvement(self):
        margins = np.r_[0.0, np.logspace(-300, 0, 1000), 2.0]
        mean_score, scores = normalized_sensing_shortfall(margins)
        self.assertTrue(np.all((0 <= scores) & (scores <= 1)))
        self.assertTrue(np.all(np.diff(scores) <= 0))
        self.assertTrue(0 <= mean_score <= 1)

    def test_no_epsilon_floor_for_small_positive_margins(self):
        tiniest = np.nextafter(0.0, 1.0)
        with np.errstate(all="raise"):
            _, scores = normalized_sensing_shortfall([tiniest, 1e-300, 1e-100])
        self.assertTrue(np.all(np.isfinite(scores)))
        self.assertTrue(np.all(np.diff(scores) < 0))
        self.assertTrue(np.all(scores < 1))

    def test_reference_factor_close_to_one_and_large_factor(self):
        for factor in (np.nextafter(1.0, 2.0), 1e300):
            with np.errstate(all="raise"):
                mean_score, scores = normalized_sensing_shortfall([0.0, 0.5, 1.0], factor)
            self.assertTrue(np.isfinite(mean_score))
            self.assertTrue(np.all((scores >= 0) & (scores <= 1)))

    def test_mean_has_no_extra_uav_count_scaling(self):
        one = normalized_sensing_shortfall([0.1], expected_uavs=1)[0]
        four = normalized_sensing_shortfall([0.1] * 4, expected_uavs=4)[0]
        self.assertEqual(one, four)

    def test_input_remains_unchanged_and_output_is_independent(self):
        margins = np.array([0.0, 0.1, 1.0, 2.0])
        before = margins.copy()
        margins.setflags(write=False)
        _, scores = normalized_sensing_shortfall(margins)
        scores[:] = 0.0
        np.testing.assert_array_equal(margins, before)

    def test_invalid_margins_are_rejected_instead_of_scored(self):
        for margins in ([], 0.5, [[0.5]], [-0.1], [np.nan], [np.inf],
                        [-np.inf], [1 + 1j], ["invalid"]):
            with self.subTest(margins=margins), self.assertRaises(ValueError):
                normalized_sensing_shortfall(margins)

    def test_expected_uav_count_is_checked(self):
        for count in (0, -1, 1.5, True):
            with self.subTest(count=count), self.assertRaises(ValueError):
                normalized_sensing_shortfall([0.5], expected_uavs=count)
        with self.assertRaises(ValueError):
            normalized_sensing_shortfall([0.5], expected_uavs=4)


class SensingPenaltySettingsTests(unittest.TestCase):
    def test_contract_and_valid_settings_including_disabled_coefficient(self):
        self.assertEqual(SENSING_PENALTY_CONTRACT, "normalized_sensing_shortfall_v1")
        self.assertEqual(validate_sensing_penalty_settings(1, 10), (1.0, 10.0))
        self.assertEqual(validate_sensing_penalty_settings(0, 10), (0.0, 10.0))
        self.assertEqual(validate_sensing_penalty_settings(0.3, 2), (0.3, 2.0))

    def test_invalid_coefficients_are_rejected(self):
        for coefficient in (-0.1, 1.1, np.nan, np.inf, -np.inf, True,
                            np.bool_(False), "invalid", None):
            with self.subTest(coefficient=coefficient), self.assertRaises(ValueError):
                validate_sensing_penalty_settings(coefficient, 10)

    def test_invalid_reference_factors_are_rejected(self):
        for factor in (0, 1, -1, np.nan, np.inf, -np.inf, True,
                       np.bool_(False), "invalid", None):
            with self.subTest(factor=factor), self.assertRaises(ValueError):
                validate_sensing_penalty_settings(1, factor)


class SensingPenaltyConfigurationTests(unittest.TestCase):
    def test_main_configuration_defaults_and_zero_coefficient_override(self):
        args = get_madrl_args([])
        self.assertEqual(args.sensing_penalty_coef, 1.0)
        self.assertEqual(args.sensing_penalty_reference_factor, 10.0)
        self.assertEqual(args.structured_beam_alpha, 1.0)
        self.assertEqual(args.continuous_entropy_coef, 0.05)
        disabled = get_madrl_args(["--sensing_penalty_coef", "0"])
        self.assertEqual(disabled.sensing_penalty_coef, 0.0)

    def test_json_settings_and_explicit_cli_override(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hyperparameters.json"
            path.write_text(json.dumps({"hyperparameters": {
                "sensing_penalty_coef": 0.3,
                "sensing_penalty_reference_factor": 100.0,
            }}), encoding="utf-8")
            args = get_madrl_args(["--config", str(path)])
            self.assertEqual(args.sensing_penalty_coef, 0.3)
            self.assertEqual(args.sensing_penalty_reference_factor, 100.0)
            overridden = get_madrl_args(["--config", str(path),
                                         "--sensing_penalty_coef", "0"])
            self.assertEqual(overridden.sensing_penalty_coef, 0.0)
            self.assertEqual(overridden.sensing_penalty_reference_factor, 100.0)

    def test_invalid_cli_values_are_rejected(self):
        for key, value in (("sensing_penalty_coef", "-0.1"),
                           ("sensing_penalty_coef", "1.1"),
                           ("sensing_penalty_coef", "nan"),
                           ("sensing_penalty_reference_factor", "1"),
                           ("sensing_penalty_reference_factor", "inf")):
            with self.subTest(key=key, value=value), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as exc:
                    get_madrl_args([f"--{key}={value}"])
                self.assertEqual(exc.exception.code, 2)

    def test_invalid_json_values_including_booleans_are_rejected(self):
        for key, value in (("sensing_penalty_coef", True),
                           ("sensing_penalty_coef", 2),
                           ("sensing_penalty_reference_factor", False),
                           ("sensing_penalty_reference_factor", 1),
                           ("sensing_penalty_reference_factor", float("nan"))):
            with self.subTest(key=key, value=value), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "hyperparameters.json"
                path.write_text(json.dumps({key: value}), encoding="utf-8")
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
                    get_madrl_args(["--config", str(path)])
                self.assertEqual(exc.exception.code, 2)


class SensingPenaltyRewardTests(unittest.TestCase):
    def setUp(self):
        self.args = get_base_args(I=2, J=2, K=2, N=2)
        self.inputs = dict(
            uavs_2_cus_channels=np.zeros((2, 2, 2), dtype=complex),
            uavs_2_bs_channels=np.zeros((2, 2, 2), dtype=complex),
            cus_2_bs_channels=np.ones((2, 1), dtype=complex),
            uavs_2_targets_channels=np.zeros((2, 2, 2, 2), dtype=complex),
            uavs_targets_matched_matrix=np.eye(2),
            uavs_cus_matched_matrix=np.eye(2),
            uavs_pos=np.array([[0.0, 0.0, 50.0], [100.0, 0.0, 50.0]]),
            uavs_pos_cur=np.array([[3.0, 0.0, 50.0], [103.0, 0.0, 50.0]]),
            uavs_off_duration=np.array([0.05, 0.05]),
            cus_off_power=np.array([0.1, 0.1]),
            cus_entertaining_task_size=np.array([170000.0, 170000.0]),
            uavs_rec_beam_vectors=np.array([[1.0, 0.0], [1.0, 0.0]], dtype=complex),
        )

    def evaluate(self, energy, metrics, *, coefficient=1.0, diagnostics=False,
                 legacy_constructor=False):
        payload = {"solver_status": "infeasible" if np.isinf(energy) else "optimal",
                   "solver_status_kind": "infeasible" if np.isinf(energy) else "optimal",
                   "auxiliary_variable_z": [0.0, 0.0],
                   "env_diagnostic": {"example": "unchanged"}}
        if metrics is not None:
            payload["sensing_reward_metrics"] = metrics
        solved = (energy, None, None, None, None, None, None,
                  [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [0.1, 0.1], payload)
        if legacy_constructor:
            reward = MyReward(self.args, enable_diagnostics=diagnostics)
        else:
            reward = MyReward(self.args, enable_diagnostics=diagnostics,
                              sensing_penalty_coef=coefficient,
                              sensing_penalty_reference_factor=10.0)
        with patch("reward.my_reward.penalty_based_cccp", return_value=solved) as solver:
            result = reward.reward_compute(**self.inputs)
        return result, solver.call_args.kwargs

    def test_failed_solve_receives_only_the_expected_extra_penalty(self):
        (total, details, energy), calls = self.evaluate(
            float("inf"), {"status": "checked", "margin": [0.1, 0.1]})
        components = details["components"]
        self.assertEqual(total, -1.5)
        self.assertEqual(details["bs"], total)
        self.assertTrue(np.isinf(energy))
        self.assertEqual(components["base_reward"], -1.0)
        self.assertEqual(components["base_no_solution_penalty"], 1.0)
        self.assertAlmostEqual(components["sensing_penalty_score"], 0.5)
        self.assertAlmostEqual(components["sensing_penalty_applied"], 0.5)
        self.assertEqual(components["no_solution_penalty"], 1.5)
        np.testing.assert_allclose(components["sensing_penalty_per_uav"], [0.5, 0.5])
        self.assertEqual(components["solver_status_kind"], "infeasible")
        self.assertTrue(calls["collect_sensing_reward"])

    def test_successful_solve_preserves_original_energy_reward(self):
        (total, details, energy), _ = self.evaluate(
            1000.0, {"status": "checked", "margin": [0.0, 0.0]})
        components = details["components"]
        self.assertEqual(energy, 1000.0)
        self.assertEqual(total, float(np.exp(-1.0)))
        self.assertEqual(components["sensing_penalty_score"], 1.0)
        self.assertEqual(components["sensing_penalty_applied"], 0.0)
        self.assertEqual(components["no_solution_penalty"], 0.0)
        self.assertEqual(components["solver_status_kind"], "optimal")

    def test_other_failure_keeps_basic_no_solution_penalty(self):
        (total, details, _), _ = self.evaluate(
            float("inf"), {"status": "checked", "margin": [1.0, 2.0]})
        self.assertEqual(total, -1.0)
        self.assertEqual(details["components"]["sensing_penalty_score"], 0.0)
        self.assertEqual(details["components"]["no_solution_penalty"], 1.0)

    def test_unavailable_or_invalid_metrics_preserve_original_penalty(self):
        for metrics in (None, [], {"status": "failed"},
                        {"status": "checked", "margin": [0.1]},
                        {"status": "checked", "margin": [np.nan, 0.1]},
                        {"status": "checked", "margin": [-0.1, 0.1]},
                        {"status": "checked", "margin": [[0.1, 0.1]]}):
            with self.subTest(metrics=metrics):
                (total, details, _), _ = self.evaluate(float("inf"), metrics)
                components = details["components"]
                self.assertEqual(total, -1.0)
                self.assertEqual(components["no_solution_penalty"], 1.0)
                self.assertEqual(components["sensing_penalty_applied"], 0.0)
                self.assertEqual(components["sensing_penalty_status"], "unavailable")
                self.assertIsNotNone(components["sensing_penalty_reason"])

    def test_disabled_or_legacy_constructor_preserves_original_solver_call(self):
        for legacy in (False, True):
            for energy in (float("inf"), 1000.0):
                with self.subTest(legacy=legacy, energy=energy):
                    (total, details, _), calls = self.evaluate(
                        energy, {"status": "checked", "margin": [0.0, 0.0]},
                        coefficient=0.0, legacy_constructor=legacy)
                    expected = -1.0 if np.isinf(energy) else float(np.exp(-1.0))
                    self.assertEqual(total, expected)
                    self.assertNotIn("collect_sensing_reward", calls)
                    self.assertEqual(details["components"]["sensing_penalty_status"], "disabled")

    def test_reward_metrics_do_not_depend_on_diagnostic_logging_flag(self):
        rewards = []
        for enabled in (False, True):
            (total, details, _), calls = self.evaluate(
                float("inf"), {"status": "checked", "margin": [0.1, 0.1]},
                diagnostics=enabled)
            rewards.append(total)
            self.assertEqual(calls["collect_diagnostics"], enabled)
            self.assertTrue(calls["collect_sensing_reward"])
            self.assertEqual("diagnostics" in details, enabled)
        self.assertEqual(rewards, [-1.5, -1.5])

    def test_existing_collision_and_matching_penalties_are_unchanged(self):
        self.inputs["uavs_pos_cur"] = np.array([[3.0, 0.0, 50.0], [3.0, 0.0, 50.0]])
        self.inputs["uavs_cus_matched_matrix"] = np.array([[1.0, 0.0], [1.0, 0.0]])
        (total, details, _), _ = self.evaluate(
            float("inf"), {"status": "checked", "margin": [0.1, 0.1]},
            coefficient=0.3)
        components = details["components"]
        self.assertEqual(components["uav_collision_penalty_sum"], 2.0)
        self.assertEqual(components["bs_alloc_spectrum_penalty"], 1.0)
        self.assertEqual(components["base_reward"], -4.0)
        self.assertAlmostEqual(components["sensing_penalty_applied"], 0.15)
        self.assertEqual(total, -4.15)


if __name__ == "__main__":
    unittest.main()
