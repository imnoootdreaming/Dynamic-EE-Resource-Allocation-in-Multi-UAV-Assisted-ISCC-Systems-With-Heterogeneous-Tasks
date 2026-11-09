"""Independent inner sampling uses analytic beams; online assembly stays explicit."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


INNER = Path(__file__).resolve().parent
sys.path.insert(0, str(INNER))
from environment import _build_mvdr_rec_beam, assemble_environment, build_environment, build_outer_variables
from outer_sampler import load_outer_samples, save_outer_samples
from parameters import Parameters

spec = importlib.util.spec_from_file_location(
    "_mvdr_outer_mapping_reference", INNER.parent / "outer/environment/mapping.py")
outer_mapping = importlib.util.module_from_spec(spec)
spec.loader.exec_module(outer_mapping)


class ReceiveBeamTests(unittest.TestCase):
    def setUp(self):
        self.params = Parameters()
        self.rng = np.random.default_rng(37)

    def outer_beam(self, env):
        return outer_mapping.build_mvdr_rec_beam(
            env.A_theta[:, None], np.zeros(self.params.I, dtype=int),
            env.h_cu_2_uav, env.eta_share, env.p_cu_power, self.params.sigma_2)

    def environment(self, variables=None):
        return build_environment(self.params, np.random.default_rng(self.params.seed), variables)

    def test_complex_channels_match_online_mvdr_exactly(self):
        I, J, N = self.params.I, self.params.J, self.params.N
        a = np.exp(1j * self.rng.uniform(-np.pi, np.pi, (I, N)))
        a[:, 0] = 1
        A = 1e-8 * a[:, :, None] * a[:, None, :].conj()
        h = 1e-5 * (self.rng.normal(size=(I, J, N)) + 1j * self.rng.normal(size=(I, J, N)))
        eta = self.rng.integers(0, 2, size=(I, J)).astype(float)
        power = self.rng.uniform(0, self.params.P_max_cu, J)
        g = _build_mvdr_rec_beam(A, h, eta, power, self.params.sigma_2)
        expected = outer_mapping.build_mvdr_rec_beam(
            A[:, None], np.zeros(I, dtype=int), h, eta, power, self.params.sigma_2)
        np.testing.assert_array_equal(g, expected)
        np.testing.assert_allclose(np.linalg.norm(g, axis=1), 1.0, atol=1e-15)
        for i in range(I):
            R = self.params.sigma_2 * np.eye(N, dtype=complex)
            for j in range(J):
                R += eta[i, j] * power[j] * np.outer(h[i, j].conj(), h[i, j])
            ratio = abs(np.vdot(g[i], a[i])) ** 2 / np.vdot(g[i], R @ g[i]).real
            optimum = np.vdot(a[i], np.linalg.solve(R, a[i])).real
            self.assertAlmostEqual(ratio / optimum, 1.0, places=10)

    def test_zero_interference_and_single_antenna(self):
        A = np.array([[[1, -1j], [1j, 1]]], dtype=complex) * 1e-12
        h = np.array([[[1e-4j, 2e-4]]])
        for eta, power in ((np.zeros((1, 1)), np.ones(1)),
                           (np.ones((1, 1)), np.zeros(1))):
            g = _build_mvdr_rec_beam(A, h, eta, power, self.params.sigma_2)
            np.testing.assert_allclose(g, np.array([[1, 1j]]) / np.sqrt(2))
        g = _build_mvdr_rec_beam(np.ones((1, 1, 1)), h[:, :, :1],
                                np.ones((1, 1)), np.ones(1), self.params.sigma_2)
        np.testing.assert_array_equal(g, np.ones((1, 1), dtype=complex))

    def test_cached_csv_beam_is_ignored_and_file_is_preserved(self):
        path = INNER / "feasible_outer_samples.csv"
        original = path.read_bytes()
        sample = load_outer_samples(str(path), self.params)[0]
        env = self.environment(sample)
        changed = (np.zeros_like(sample[0]),) + sample[1:]
        other = self.environment(changed)
        for name in ("g_rec_beam", "Gamma_sinr", "G_sen_corr"):
            np.testing.assert_array_equal(getattr(env, name), getattr(other, name))
        np.testing.assert_array_equal(env.g_rec_beam, self.outer_beam(env))
        self.assertFalse(np.allclose(env.g_rec_beam, sample[0]))
        self.assertEqual(path.read_bytes(), original)

    def test_new_samples_and_csv_round_trip_use_analytic_beam(self):
        variables = build_outer_variables(self.params, self.rng)
        self.assertIsNone(variables[0])
        env = self.environment(variables)
        np.testing.assert_array_equal(env.g_rec_beam, self.outer_beam(env))
        stored = (env.g_rec_beam, env.eta_share, env.p_cu_power, env.D_uav_off,
                  env.q_uav_pos_next - env.q_uav_pos)
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "samples.csv")
            save_outer_samples([stored], path)
            loaded = load_outer_samples(path, self.params)[0]
            replay = self.environment(loaded)
        for name in ("g_rec_beam", "Gamma_sinr", "G_sen_corr"):
            np.testing.assert_array_equal(getattr(env, name), getattr(replay, name))

    def test_explicit_online_assembly_keeps_supplied_beam(self):
        env = self.environment()
        names = ("q_uav_pos", "q_uav_pos_next", "q_cu_pos", "q_target_pos",
                 "q_target_designated", "h_cu_2_uav", "h_uav_2_bs", "h_cu_2_bs",
                 "A_theta", "g_rec_beam", "eta_share", "p_cu_power", "D_uav_off",
                 "d_uav_target")
        kwargs = {name: getattr(env, name) for name in names}
        explicit = np.zeros_like(env.g_rec_beam)
        explicit[:, 0] = 1j
        kwargs["g_rec_beam"] = explicit
        supplied = assemble_environment(self.params, **kwargs)
        np.testing.assert_array_equal(supplied.g_rec_beam, explicit)
        expected_gamma = np.sum(env.eta_share * env.p_cu_power[None, :]
                                * abs(env.h_cu_2_uav[:, :, 0]) ** 2, axis=1) + self.params.sigma_2
        np.testing.assert_allclose(supplied.Gamma_sinr, expected_gamma, rtol=1e-14, atol=0)

    def test_comparison_passes_same_mvdr_environment_to_both_solvers(self):
        import compare_pc3p_gc3p as comparison

        class BothSolversReached(Exception):
            pass

        sample = load_outer_samples(str(INNER / "feasible_outer_samples.csv"), self.params)[0]
        received = []

        def pc3p_entry(ctx):
            received.append(ctx.environment)
            np.testing.assert_array_equal(ctx.g_rec_beam, self.outer_beam(ctx.environment))
            return {"W_sen_beam": None}

        def gc3p_entry(ctx, **kwargs):
            received.append(ctx.environment)
            np.testing.assert_array_equal(ctx.g_rec_beam, self.outer_beam(ctx.environment))
            raise BothSolversReached

        with patch.object(comparison, "run_pc3p_fusion", side_effect=pc3p_entry), \
                patch.object(comparison, "run_gc3p", side_effect=gc3p_entry):
            with self.assertRaises(BothSolversReached):
                comparison.run_comparison(self.params, [sample], num_candidates=2)
        self.assertEqual(len(received), 2)
        self.assertIs(received[0], received[1])
        self.assertFalse(np.allclose(received[0].g_rec_beam, sample[0]))


if __name__ == "__main__":
    unittest.main()
