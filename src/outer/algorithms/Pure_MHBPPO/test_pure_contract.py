"""Physical P0, no-solver, old-policy and environment regression tests."""
import importlib.util
import queue
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

OUTER = Path(__file__).resolve().parents[2]
if str(OUTER) not in sys.path:
    sys.path.insert(0, str(OUTER))
from algorithms.Pure_MHBPPO.physical import evaluate_physical
from algorithms.Pure_MHBPPO.pure_env import PureMyEnv
from algorithms.Pure_MHBPPO import Pure_MHBPPO_agent as migrated
from algorithms.Pure_MHBPPO import Pure_MHBPPO_main as trainer
from algorithms.MHBPPO.rollout_batch import actor_state_signature
from configs.base_params import get_base_args
from configs.madrl_args import get_madrl_args


def load_old():
    path = OUTER.parent / "outer_old/algorithms/MHBPPO/MHBPPO_agent.py"
    spec = importlib.util.spec_from_file_location("old_pure_hppo", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture():
    p = SimpleNamespace(bandwidth=1e6, noise_power_density_dbm=-174, sen_sinr=-10,
        var_range_fluctuation=1e-20, radar_spectrum_shape=1., radar_impulse_duration=1e-6,
        radar_duty_ratio=.01, uav_sen_duration=.1, uav_max_delay=.8,
        C_sen=1000., uav_min_speed=.5, uav_max_speed=2., uav_safe_distance=.5,
        uav_max_power=10., cu_max_power_dbm=30., bs_max_freq=1e8,
        uav_c1=.006, uav_c2=15., kappa=1e-28,
        omega_weight_1=.2, omega_weight_2=.4, omega_weight_3=.4)
    return dict(params=p, eta=np.array([[1.]]), p_cu=np.array([.1]),
        duration_uav=np.array([.2]), duration_cu=np.array([.5]),
        sensing_beams=np.array([[1.+0j]]), offloading_beams=np.array([[1.+0j]]),
        positions_before=np.array([[0., 0., 100.]]), positions_after=np.array([[1., 0., 100.]]),
        h_cu_uav=np.array([[[0.+0j]]]), h_uav_bs=np.array([[[.001+0j]]]),
        h_cu_bs=np.array([[.001+0j]]), sensing_channels=np.array([[[1e-6+0j]]]),
        task_bits=np.array([100.]), task_cycles=np.array([1000.]),
        task_deadlines=np.array([.9]), slot_seconds=1.)


class Writer:
    def __init__(self): self.values = {}
    def add_scalar(self, name, value, step): self.values[name] = float(value)


class PureContract(unittest.TestCase):
    def test_p0_rates_energy_and_fifteen_constraints(self):
        x = fixture(); p = x["params"]
        result = evaluate_physical(**x)
        self.assertTrue(result["feasible"], result["checks"])
        self.assertEqual(len(result["checks"]), 15)
        noise = 10**((-174-30)/10)*p.bandwidth
        sinr = 1e-12/noise
        self.assertAlmostEqual(result["rates"]["sensing_sinr"][0], sinr)
        xi2 = 2*p.var_range_fluctuation*p.radar_spectrum_shape**2*p.bandwidth**3*p.radar_impulse_duration
        rs = p.radar_duty_ratio/(2*p.radar_impulse_duration)*np.log2(1+xi2*sinr)
        self.assertAlmostEqual(result["rates"]["sensing"][0], rs)
        roff = p.bandwidth*np.log2(1+1e-6/(noise+0.1e-6))
        self.assertAlmostEqual(result["rates"]["uav_offloading"][0], roff)
        r1 = p.bandwidth*np.log2(1+0.1e-6/(noise+1e-6))
        r3 = p.bandwidth*np.log2(1+0.1e-6/noise)
        self.assertAlmostEqual(result["rates"]["cu_offloaded_bits"][0], .1*r1+.2*r1+.2*r3)
        fu = 1000*.1*rs/(.8-.1-.2)
        fc = 1000*100/(.9-.5)
        self.assertAlmostEqual(result["frequencies"]["uav_hz"][0], fu)
        self.assertAlmostEqual(result["frequencies"]["cu_hz"][0], fc)
        e_bs = p.kappa*(1000*.1*rs*fu**2+1000*100*fc**2)
        e_uav = .1+.2+p.uav_c1+p.uav_c2
        expected = .2*e_bs+.4*e_uav+.4*.5*.1
        self.assertAlmostEqual(result["candidate_energy"], expected)
        self.assertEqual(result["solver_status_kind"], "not_called")

    def test_heterogeneous_unmatched_cu_and_complex_interference(self):
        x = fixture(); x.update(eta=np.array([[1., 0.]]), p_cu=np.array([.1, .2]),
            duration_cu=np.array([.5, .4]), h_cu_uav=np.array([[[1e-6+2e-6j], [0j]]]),
            h_cu_bs=np.array([[.001+0j], [.002+0j]]), task_bits=np.array([100., 200.]),
            task_cycles=np.array([1000., 1500.]), task_deadlines=np.array([.9, .8]))
        result = evaluate_physical(**x)
        noise = 10**((-174-30)/10)*x["params"].bandwidth
        expected_r3 = x["params"].bandwidth*np.log2(1+.2*.002**2/noise)
        self.assertAlmostEqual(result["rates"]["cu_offloaded_bits"][1], .4*expected_r3)
        self.assertAlmostEqual(result["frequencies"]["cu_hz"][1], 1500*200/(.8-.4))
        self.assertTrue(result["feasible"], result["checks"])

    def test_rank_one_energy_matches_existing_inner_formula(self):
        import pc3p  # Imported by the existing bridge; no optimization invoked.
        x = fixture(); p = x["params"]
        result = evaluate_physical(**x)
        ctx = SimpleNamespace(C_sen=p.C_sen, C_cu_cycles=x["task_cycles"], L_cu_task=x["task_bits"],
            D_max_cu=x["task_deadlines"], omega_1=p.omega_weight_1, omega_2=p.omega_weight_2,
            omega_3=p.omega_weight_3, kappa_cpu=p.kappa, D_bar_sen=p.uav_sen_duration,
            D_uav_off=x["duration_uav"], E_uav_fly=np.array([p.uav_c1+p.uav_c2]),
            p_cu_power=x["p_cu"], I=1)
        W = np.array([np.outer(x["sensing_beams"][0], x["sensing_beams"][0].conj())])
        B = np.array([np.outer(x["offloading_beams"][0], x["offloading_beams"][0].conj())])
        baseline = pc3p.compute_energy_breakdown(ctx, W, B, np.array(result["frequencies"]["uav_hz"]),
            np.array(result["rates"]["sensed_bits"]), x["duration_cu"])
        self.assertAlmostEqual(result["candidate_energy"], baseline["weighted_total_energy_j"], places=12)

    def test_legacy_pure_reward_on_aligned_feasible_case(self):
        path = OUTER.parent / "outer_old/reward/pure-my-reward.py"
        spec = importlib.util.spec_from_file_location("legacy_pure_reward", path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        x = fixture(); p = x["params"]
        p.uavs_num, p.cus_num, p.antenna_nums = 1, 1, 1
        p.bs_cycles_per_bit, p.time_slot_duration, p.cu_max_delay = 1000., 1., .9
        old_reward = module.PureMyReward(p)
        _, details, old_energy = old_reward.reward_compute(
            uavs_2_cus_channels=x["h_cu_uav"], uavs_2_bs_channels=x["h_uav_bs"],
            cus_2_bs_channels=x["h_cu_bs"], uavs_2_targets_channels=x["sensing_channels"][:, None, :, :],
            uavs_targets_matched_matrix=np.ones((1,1)), uavs_cus_matched_matrix=x["eta"],
            uavs_pos=x["positions_before"], uavs_pos_cur=x["positions_after"],
            uavs_off_duration=x["duration_uav"], cus_off_power=x["p_cu"],
            cus_entertaining_task_size=x["task_bits"], uav_sen_beam_vectors=x["sensing_beams"],
            uav_off_beam_vectors=x["offloading_beams"], cus_off_durations=x["duration_cu"])
        result = evaluate_physical(**x)
        self.assertTrue(result["feasible"])
        self.assertEqual(details["components"]["tanh_penalty_raw"], 0)
        self.assertAlmostEqual(result["candidate_energy"], old_energy, places=10)

    def test_complex_mvdr_and_sensing_sinr_use_correct_conjugates(self):
        x = fixture()
        x.update(sensing_beams=np.array([[1+0.5j, -0.3+0.2j]]),
                 offloading_beams=np.array([[0.2+0.1j, 0.4-0.3j]]),
                 h_cu_uav=np.array([[[1e-6+2e-6j, 3e-6-1e-6j]]]),
                 h_uav_bs=np.array([[[.001+.002j, .002-.001j]]]),
                 sensing_channels=np.array([[[1e-6+0j, 0.5e-6-0.3e-6j],
                                              [0.2e-6+0.1e-6j, 1e-6+0j]]]))
        result = evaluate_physical(**x)
        noise = 10**((-174-30)/10)*x["params"].bandwidth
        h = x["h_cu_uav"][0,0]; w = x["sensing_beams"][0]; A = x["sensing_channels"][0]
        cov = noise*np.eye(2)+.1*np.outer(h.conj(),h)
        direction = np.linalg.solve(cov, A@w)
        g = direction/np.linalg.norm(direction)
        expected = abs(np.vdot(g, A@w))**2 / (noise+.1*abs(np.dot(g,h))**2)
        self.assertAlmostEqual(result["rates"]["sensing_sinr"][0], expected, places=10)

    def test_frequency_infeasibility_and_invalid_boundaries(self):
        x = fixture(); x["params"].bs_max_freq = 1000.
        result = evaluate_physical(**x)
        self.assertFalse(result["feasible"])
        self.assertFalse(result["checks"]["bs_frequency"]["passed"])
        self.assertTrue(result["checks"]["uav_delay"]["passed"])
        self.assertTrue(result["checks"]["cu_delay"]["passed"])
        self.assertIsNotNone(result["candidate_energy"])
        x = fixture(); x["duration_cu"] = np.array([.9])
        result = evaluate_physical(**x)
        self.assertFalse(result["feasible"])
        self.assertFalse(result["checks"]["cu_delay"]["passed"])
        self.assertIsNone(result["candidate_energy"])
        x = fixture(); x["p_cu"][0] = np.nan
        self.assertFalse(evaluate_physical(**x)["valid_input"])

    def test_duplicate_matching_and_zero_beams(self):
        x = fixture(); x.update(eta=np.ones((2, 1)), duration_uav=np.array([.2, .2]),
            sensing_beams=np.array([[0j], [1+0j]]), offloading_beams=np.ones((2, 1), dtype=complex),
            positions_before=np.array([[0, 0, 100], [5, 0, 100.]]),
            positions_after=np.array([[1, 0, 100], [6, 0, 100.]]),
            h_cu_uav=np.zeros((2, 1, 1), complex), h_uav_bs=np.ones((2, 1, 1), complex)*.001,
            sensing_channels=np.ones((2, 1, 1), complex)*1e-6)
        result = evaluate_physical(**x)
        self.assertFalse(result["feasible"])
        self.assertFalse(result["checks"]["cu_at_most_one_uav"]["passed"])
        self.assertFalse(result["checks"]["sensing_sinr"]["passed"])
        self.assertTrue(result["receive_beams"]["fallback"][0])
        self.assertGreater(result["spectrum_penalty"], 0)

    def test_duplicate_schedule_does_not_add_undefined_cu_service_penalty(self):
        x = fixture(); x.update(eta=np.ones((2, 1)), duration_uav=np.array([.2, .2]),
            sensing_beams=np.ones((2, 1), complex), offloading_beams=np.ones((2, 1), complex),
            positions_before=np.array([[0, 0, 100], [5, 0, 100.]]),
            positions_after=np.array([[1, 0, 100], [6, 0, 100.]]),
            h_cu_uav=np.zeros((2, 1, 1), complex), h_uav_bs=np.ones((2, 1, 1), complex)*.001,
            sensing_channels=np.ones((2, 1, 1), complex)*1e-6)
        result = evaluate_physical(**x)
        self.assertFalse(result["feasible"])
        self.assertEqual(result["spectrum_penalty"], 1)
        self.assertLess(result["rates"]["cu_stage3_seconds"][0], 0)
        self.assertEqual(result["service_candidate_penalty"], 0)

    def test_constraint_failure_detection(self):
        cases = []
        x = fixture(); x["eta"][:] = 0; cases.append(("uav_one_cu", x))
        x = fixture(); x["positions_after"][:] = x["positions_before"]; cases.append(("uav_speed", x))
        x = fixture(); x["sensing_beams"][:] = np.sqrt(11); cases.append(("sensing_power", x))
        x = fixture(); x["sensing_beams"][:] = 0; cases.append(("sensing_sinr", x))
        x = fixture(); x["offloading_beams"][:] = np.sqrt(11); cases.append(("offloading_power", x))
        x = fixture(); x["duration_uav"][:] = 1e-12; cases.append(("uav_offloaded_bits", x))
        x = fixture(); x["duration_cu"][:] = .2; cases.append(("task_freshness", x))
        x = fixture(); x["task_bits"][:] = 1e12; cases.append(("cu_offloaded_bits", x))
        x = fixture(); x["p_cu"][:] = 2; cases.append(("cu_power", x))
        x = fixture(); x["duration_uav"][:] = .71; cases.append(("uav_delay", x))
        x = fixture(); x["duration_cu"][:] = .9; cases.append(("cu_delay", x))
        x = fixture(); x["params"].bs_max_freq = 1000; cases.append(("bs_frequency", x))
        x = fixture(); x.update(eta=np.ones((2, 1)), duration_uav=np.array([.2, .2]),
            sensing_beams=np.ones((2, 1), complex), offloading_beams=np.ones((2, 1), complex),
            positions_before=np.array([[0, 0, 100], [0, 0, 100.]]),
            positions_after=np.array([[1, 0, 100], [1, 0, 100.]]),
            h_cu_uav=np.zeros((2, 1, 1), complex), h_uav_bs=np.ones((2, 1, 1), complex)*.001,
            sensing_channels=np.ones((2, 1, 1), complex)*1e-6)
        cases.extend([("cu_at_most_one_uav", x), ("uav_collision", x)])
        for name, candidate in cases:
            with self.subTest(constraint=name):
                result = evaluate_physical(**candidate)
                self.assertFalse(result["checks"][name]["passed"])
                self.assertFalse(result["feasible"])

    def test_legacy_policy_matches_sample_loss_and_update(self):
        old = load_old()
        def create(module):
            return module.HPPO(5, 8, {"w": 2, "b": 3}, [3, 2], np.zeros(5), np.ones(5),
                3e-4, 3e-4, .95, .2, .99, 2, 100, torch.device("cpu"), continuous_dist_type="beta")
        torch.manual_seed(42); previous = create(old)
        torch.manual_seed(42); current = create(migrated)
        rng = np.random.default_rng(9)
        keys = ("states", "continuous_actions", "discrete_actions", "next_states", "rewards",
                "old_cont_log_probs", "old_disc_log_probs", "dones", "real_dones")
        data = {k: [] for k in keys}
        for i in range(8):
            state = rng.normal(size=5).astype(np.float32)
            torch.manual_seed(i); old_a = previous.choose_action(state)
            torch.manual_seed(i); new_a = current.choose_action(state)
            for key in old_a[0]: np.testing.assert_array_equal(old_a[0][key], new_a[0][key])
            for x,y in zip(old_a[1:], new_a[1:]): np.testing.assert_array_equal(x,y)
            values = (state, new_a[0]["continuous"], new_a[0]["discrete"],
                      rng.normal(size=5).astype(np.float32), float(rng.normal()),
                      float(new_a[2].item()), float(new_a[3].item()), i in (3,7), i in (3,7))
            for key, value in zip(keys, values): data[key].append(value)
        left, right = Writer(), Writer()
        torch.manual_seed(88); previous.update(data, 0, left)
        torch.manual_seed(88); current.update(data, 0, right)
        self.assertEqual(left.values, right.values)
        for x,y in zip(previous.actor.parameters(), current.actor.parameters()):
            torch.testing.assert_close(x,y,rtol=0,atol=0)
        for x,y in zip(previous.critic.parameters(), current.critic.parameters()):
            torch.testing.assert_close(x,y,rtol=0,atol=0)

    def test_sampler_preserves_actual_policy_input_and_probabilities(self):
        class FakeEnv:
            def __init__(self, base_args, madrl_args, seed_offset):
                self.observation_space = {"bs": SimpleNamespace(shape=(3,))}
                self.bs_continuous_action_splits = {"x": 2}
                self.action_space = {"bs": {"discrete": SimpleNamespace(nvec=np.array([3]))}}
            def reset(self, version):
                self.t = 0
                return {"bs": np.array([1., 2., 3.], dtype=np.float32)}
            def step(self, action, version):
                self.t += 1
                self.last_action_trace = {"time_slot": self.t-1}
                return ({"bs": np.array([1.+self.t, 2., 3.], dtype=np.float32)},
                        1., {"bs": 1., "components": {"solver_status_kind": "not_called"}},
                        self.t == 2, 1., 1)
            def getPosUAV(self): return np.zeros((1, 3))
            def get_exogenous_scene_signature(self): return {"scene": "fake"}
        base = SimpleNamespace(seed=42)
        args = SimpleNamespace(worker_num_threads=1, scene_seed_offset=0, num_samplers=1,
                               hidden_dim=8, gamma=.99)
        actor = migrated.MultiHeadActor(3, 8, {"x": 2}, [3], continuous_dist_type="beta")
        control, results = queue.Queue(), queue.Queue()
        control.put({"version": 0, "actor": {k:v.detach().numpy().copy() for k,v in actor.state_dict().items()},
                     "signature": actor_state_signature(actor),
                     "normalizer": {"n": 0, "mean": np.zeros(3), "std": np.zeros(3)}})
        control.put(None)
        with patch.object(trainer, "PureMyEnv", FakeEnv):
            trainer.worker_main(base, args, 0, control, results)
        item = results.get_nowait()
        self.assertNotIn("error", item)
        self.assertEqual(item["normalizer"]["n"], 3)
        data = item["transition"]
        self.assertEqual(data["dones"], [False, True])
        for i in range(2):
            with torch.no_grad():
                evaluated = actor.evaluate_actions(torch.tensor(data["states"][i], dtype=torch.float32).unsqueeze(0),
                    torch.tensor(data["continuous_actions"][i]).unsqueeze(0),
                    torch.tensor(data["discrete_actions"][i]).unsqueeze(0))
            self.assertAlmostEqual(evaluated[2].item(), data["old_cont_log_probs"][i], places=5)
            self.assertAlmostEqual(evaluated[3].item(), data["old_disc_log_probs"][i], places=5)

    def test_current_environment_never_calls_old_reward_or_solver(self):
        args = get_madrl_args(["--total_time_slots", "10", "--hidden_dim", "16"])
        env = PureMyEnv(get_base_args(seed=args.seed, target_layout=args.target_layout), args)
        import cvxpy
        import mosek.fusion
        import reward.cccp_bridge as bridge
        with patch.object(env.reward_calculator, "reward_compute", side_effect=AssertionError("old reward called")), \
             patch.object(cvxpy.Problem, "solve", side_effect=AssertionError("CVXPY called")), \
             patch.object(mosek.fusion.Model, "solve", side_effect=AssertionError("Fusion called")), \
             patch.object(bridge, "run_pc3p", side_effect=AssertionError("PC3P called")), \
             patch.object(bridge, "solve_inner_energy", side_effect=AssertionError("inner solver called")):
            state = env.reset(0)
            self.assertEqual(state["bs"].shape, env.observation_space["bs"].shape)
            self.assertEqual(env.action_space["bs"]["continuous"].shape, (200,))
            action = {"continuous": np.full(200, .5, dtype=np.float32), "discrete": np.arange(4)}
            for i in range(10):
                next_state, reward, details, done, energy, success = env.step({"bs": action}, 0)
                self.assertTrue(np.isfinite(reward))
                self.assertEqual(details["components"]["solver_status_kind"], "not_called")
                self.assertEqual(len(details["components"]["constraint_checks"]), 15)
                self.assertEqual(done, int(i == 9))
                self.assertEqual(next_state["bs"].shape, state["bs"].shape)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
