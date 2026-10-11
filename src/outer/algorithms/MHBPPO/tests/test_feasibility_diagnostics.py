"""Scientific boundary/contract tests for read-only T02/T03 diagnostics."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

OUTER = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(OUTER))
from reward.feasibility_diagnostics import (
    _matching, sensing_diagnostics, outer_action_checks, validate_original_solution,
)
from algorithms.MHBPPO.action_logging import (
    ActionLogWriter, EnvDiagnosticLogWriter, build_run_log_paths, diagnostic_log_path,
    diagnostic_run_metadata, summarize_diagnostics,
)


def fixture():
    p = SimpleNamespace(I=1, J=1, N=2, sigma_2=1.0, eps_sinr=1.0,
        eps_sinr_db=0.0, P_max_uav=10.0, P_max_cu=2.0, gamma_2=1e-5,
        D_bar_sen=0.1, D_max_sen=0.2, B=100.0, xi_1=1.0, xi_2=1.0,
        C_sen=1.0, C_cu_cycles=np.array([1.0]), L_cu_task=np.array([1.0]),
        D_max_cu=np.array([1.0]), F_max=100.0, uav_max_speed=40.0,
        uav_min_speed=5.0, tau_slot=0.6, uav_height=50.0, uav_safe_distance=5.0)
    e = SimpleNamespace(g_rec_beam=np.array([[1.0, 0.0]], dtype=complex),
        h_cu_2_uav=np.zeros((1, 1, 2), complex), eta_share=np.ones((1, 1)),
        p_cu_power=np.ones(1), G_sen_corr=np.array([np.diag([1.0, 0.0])]),
        H_uav_bs=np.array([np.diag([1.0, 0.0])]), Gamma_sinr=np.ones(1),
        Phi_off_inr=np.array([2.0]), Theta_cu_time=np.array([0.15]),
        upsilon_cu_rate=np.array([1.0]), D_uav_off=np.array([0.05]),
        h_cu_2_bs=np.ones(1, complex), d_uav_target=np.array([50.0]),
        q_uav_pos=np.array([[0.0, 0.0, 50.0]]),
        q_uav_pos_next=np.array([[3.0, 0.0, 50.0]]))
    s = dict(W_sen_beam=np.array([np.diag([1.0, 0.0])]),
        B_off_beam=np.array([np.diag([1.0, 0.0])]),
        w_sen_beam=np.array([[1.0, 0.0]], complex),
        b_off_beam=np.array([[1.0, 0.0]], complex),
        f_uav_freq=np.array([21.0]), f_cu_freq=np.array([2.0]),
        z_aux_rate=np.array([1.01]), D_cu_off=np.array([0.5]))
    return p, e, s


class OriginalConstraintTests(unittest.TestCase):
    def test_analytic_feasible_rank_one_solution(self):
        p, e, s = fixture()
        result = validate_original_solution(p, e, s)
        self.assertTrue(result['physical_solution_verified'], result)
        self.assertTrue(outer_action_checks(p, e, cu_min_power=0.2)['passed'])

    def test_tiny_watt_sinr_violation_is_not_hidden_by_absolute_tolerance(self):
        p, e, s = fixture()
        p.sigma_2 = 1e-10
        e.Gamma_sinr[:] = 1e-10
        e.G_sen_corr *= 1e-10
        s['W_sen_beam'] *= 0.5
        result = validate_original_solution(p, e, s)
        check = result['covariance']['checks']['c06_sensing_sinr']
        self.assertAlmostEqual(check['raw_residual'][0], 5e-11)
        self.assertAlmostEqual(check['normalized_residual'][0], 0.5)
        self.assertFalse(check['passed'])

    def test_sinr_boundary_uses_normalized_tolerance(self):
        p, e, s = fixture()
        s['W_sen_beam'] *= 1 - 0.5e-6
        self.assertTrue(validate_original_solution(p, e, s)['covariance']['checks']['c06_sensing_sinr']['passed'])
        s['W_sen_beam'] *= 1 - 2e-6
        self.assertFalse(validate_original_solution(p, e, s)['covariance']['checks']['c06_sensing_sinr']['passed'])

    def test_offloading_shortfall_is_detected(self):
        p, e, s = fixture()
        s['B_off_beam'] *= 1e-8
        result = validate_original_solution(p, e, s)
        self.assertIn('c08_uav_offloading', result['covariance']['failed_checks'])
        self.assertFalse(result['physical_solution_verified'])

    def test_recovered_beam_cannot_inherit_covariance_feasibility(self):
        p, e, s = fixture()
        s['b_off_beam'] = np.array([[0.0, 1.0]], complex)
        result = validate_original_solution(p, e, s)
        self.assertTrue(result['covariance']['passed'])
        self.assertTrue(result['rank']['passed'])
        self.assertIn('c08_uav_offloading', result['recovered_beams']['failed_checks'])
        self.assertFalse(result['physical_solution_verified'])

    def test_indefinite_tiny_offloading_matrix_is_detected_relative_to_matrix(self):
        p, e, s = fixture()
        s['B_off_beam'] = np.array([np.diag([1e-7, -1e-9])])
        result = validate_original_solution(p, e, s)
        self.assertFalse(result['covariance']['checks']['c04_offloading_psd']['passed'])

    def test_high_rank_is_not_accepted_as_physical_solution(self):
        p, e, s = fixture()
        s['B_off_beam'][0, 1, 1] = 1
        result = validate_original_solution(p, e, s)
        self.assertTrue(result['covariance']['passed'])
        self.assertFalse(result['rank']['passed'])
        self.assertFalse(result['physical_solution_verified'])

    def test_deadline_zero_compute_time_is_rejected_even_below_tolerance(self):
        p, e, s = fixture()
        s['D_cu_off'][:] = p.D_max_cu
        result = validate_original_solution(p, e, s)
        self.assertFalse(result['covariance']['checks']['c09_cu_delay']['passed'])

    def test_original_cu_data_and_sensing_auxiliary_are_checked(self):
        p, e, s = fixture()
        p.L_cu_task[:] = 1000
        s['z_aux_rate'][:] = 0.5
        result = validate_original_solution(p, e, s)
        self.assertIn('c12_original_cu_offloading', result['covariance']['failed_checks'])
        self.assertIn('c11_original_sensing_auxiliary', result['covariance']['failed_checks'])

    def test_nonfinite_missing_and_bad_shape_are_not_evaluated_as_pass(self):
        p, e, s = fixture()
        self.assertEqual(validate_original_solution(p, e, {})['covariance']['status'], 'no_solution')
        s['W_sen_beam'][0, 0, 0] = np.nan
        self.assertEqual(validate_original_solution(p, e, s)['covariance']['status'], 'non_finite_solution')
        s['W_sen_beam'] = np.zeros((2, 2))
        self.assertEqual(validate_original_solution(p, e, s)['covariance']['status'], 'invalid_shape')

    def test_validation_is_read_only(self):
        p, e, s = fixture()
        original = copy.deepcopy(s)
        validate_original_solution(p, e, s)
        for k, values in s.items():
            np.testing.assert_array_equal(values, original[k])


class SensingTests(unittest.TestCase):
    def test_matching_checks_column_conflicts(self):
        self.assertIsNone(_matching([[True, False], [True, False]]))
        match = _matching([[True, True], [True, False]])
        self.assertEqual(match, [1, 0])

    def test_projection_matches_actual_complex_dot_convention(self):
        p, e, s = fixture()
        e.g_rec_beam = np.array([[1j, 1]]) / np.sqrt(2)
        e.h_cu_2_uav = np.array([[[1j, 1]]])
        result = sensing_diagnostics(p, e, cu_min_power=0.2)
        self.assertAlmostEqual(result['interference_w'][0], 0.0)
        self.assertEqual(result['candidates']['complete_task_feasibility'], 'not_evaluated')

    def test_zero_sensing_channel_certifies_fixed_beam_impossibility(self):
        p, e, s = fixture()
        e.G_sen_corr[:] = 0
        result = sensing_diagnostics(p, e, cu_min_power=0.2)
        self.assertTrue(result['current_action_infeasible'])
        self.assertTrue(result['candidates']['no_matching_at_min_power_certified'])
        self.assertTrue(np.isinf(result['required_sensing_power_w'][0]))

    def test_candidate_sampling_is_explicit(self):
        p, e, s = fixture()
        result = sensing_diagnostics(p, e, candidates=False)
        self.assertEqual(result['candidates'], {'evaluated': False})
        self.assertIsNotNone(result['min_margin'])


class LogContractTests(unittest.TestCase):
    def episode(self, episode=0):
        p, e, s = fixture()
        diagnostic = dict(scene_id='scene-1', time_slot=0,
            sensing=sensing_diagnostics(p, e), validation=validate_original_solution(p, e, s),
            solver={'solve_seconds': 0.1}, diagnostic_seconds=0.01, legacy_success=1)
        return dict(sampler_id=0, seed_offset=1, episode_idx=episode, weights_version=episode,
            diagnostic_scene={'scene_id': 'scene-1'}, diagnostic_traces=[diagnostic],
            action_traces=[{'time_slot': 0}],
            transition_dict={'continuous_actions': [np.array([0.5])],
                             'discrete_actions': [np.array([0])], 'rewards': [1.0]})

    def test_scene_deduplication_ids_nonfinite_encoding_and_close(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = str(Path(tmp) / 'run_env.jsonl')
            writer = EnvDiagnosticLogWriter(destination, {'run_id': 'run-1'})
            try:
                item = self.episode()
                item['diagnostic_traces'][0]['example_inf'] = float('inf')
                writer.write_episode(item, 1)
                writer.write_episode(self.episode(1), 2)
            finally:
                writer.close()
                writer.close()
            rows = [json.loads(line) for line in Path(destination).read_text(encoding='utf8').splitlines()]
            self.assertEqual([r['record_type'] for r in rows], ['metadata', 'scene', 'env_diagnostic', 'env_diagnostic'])
            self.assertEqual(rows[2]['example_inf'], 'inf')
            self.assertEqual(rows[3]['run_id'], 'run-1')
            self.assertEqual((rows[3]['iteration'], rows[3]['episode_idx'], rows[3]['step_idx']), (2, 1, 0))

    def test_list_of_complex_recovered_beams_is_portable_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = str(Path(tmp) / "complex_env.jsonl")
            writer = EnvDiagnosticLogWriter(destination, {"run_id": "r"})
            try:
                item = self.episode()
                item["diagnostic_traces"][0]["snapshot_example"] = [np.array([1+2j, 3-4j])]
                writer.write_episode(item, 1)
            finally:
                writer.close()
            record = json.loads(Path(destination).read_text(encoding="utf8").splitlines()[-1])
            self.assertEqual(record["snapshot_example"][0][0], {"real": 1.0, "imag": 2.0})

    def test_misaligned_slots_are_rejected_before_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer = EnvDiagnosticLogWriter(str(Path(tmp) / 'bad_env.jsonl'), {'run_id': 'r'})
            try:
                item = self.episode()
                item['diagnostic_traces'][0]['time_slot'] = 1
                with self.assertRaises(RuntimeError):
                    writer.write_episode(item, 1)
            finally:
                writer.close()

    def test_diagnostic_summary_is_separate_from_legacy_success(self):
        item = self.episode()
        item['diagnostic_traces'][0]['legacy_success'] = 0
        summary = summarize_diagnostics(item['diagnostic_traces'])
        self.assertEqual(summary['physical_solution_pass_rate'], 100)
        self.assertEqual(item['diagnostic_traces'][0]['legacy_success'], 0)


if __name__ == '__main__':
    unittest.main()
