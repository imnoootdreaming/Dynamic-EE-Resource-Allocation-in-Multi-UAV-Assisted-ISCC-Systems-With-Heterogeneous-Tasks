"""PPO probability and optimizer contracts for alpha=0 receive-beam masking."""
import copy
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

OUTER = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(OUTER))
from algorithms.MHBPPO.MHBPPO_agent import HPPO, MultiHeadActor


BEAM_HEADS = {"uav_rec_beam_dir_real", "uav_rec_beam_dir_imag"}


def splits(uavs=4, antennas=10, cus=10):
    return {
        "uav_angles": uavs,
        "uav_distances": uavs,
        "uav_off_durations": uavs,
        "cu_off_powers": cus,
        "uav_rec_beam_dir_real": uavs * antennas,
        "uav_rec_beam_dir_imag": uavs * antennas,
    }


def actor(alpha=1.0, distribution="beta", uavs=4, antennas=10):
    return MultiHeadActor(7, 16, splits(uavs, antennas), [10] * uavs,
                          continuous_dist_type=distribution,
                          structured_beam_alpha=alpha)


def agent(alpha, distribution="beta"):
    action_splits = splits(4, 3)
    dim = sum(action_splits.values())
    return HPPO(7, 16, action_splits, [10] * 4,
                np.zeros(dim), np.ones(dim), 1e-3, 1e-3,
                0.95, 0.2, 0.99, 1, 100, torch.device("cpu"),
                continuous_dist_type=distribution, structured_beam_alpha=alpha)


def alter_beams(continuous_actions, action_splits):
    result = continuous_actions.clone()
    offset = 0
    for name, dim in action_splits.items():
        if name in BEAM_HEADS:
            result[:, offset:offset + dim] = 0.13
        offset += dim
    return result


def beam_parameters(model):
    return {name: value for name, value in model.named_parameters()
            if any("continuous_heads." + head + "." in name for head in BEAM_HEADS)}


class Recorder:
    def __init__(self):
        self.values = {}

    def add_scalar(self, name, value, step):
        self.values[name] = float(value)


class BeamActionMaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(1208)
        self.states = torch.randn(8, 7)

    def test_sample_evaluate_contract_and_dynamic_dimensions(self):
        for distribution in ("beta", "gaussian"):
            for alpha in (0.0, 0.5, 1.0):
                for uavs in (1, 4, 8):
                    with self.subTest(distribution=distribution, alpha=alpha, uavs=uavs):
                        model = actor(alpha, distribution, uavs)
                        with torch.no_grad():
                            continuous, discrete, total_lp, cont_lp, disc_lp = model.sample(self.states)
                            evaluated = model.evaluate_actions(self.states, continuous, discrete)
                        self.assertEqual(tuple(continuous.shape), (8, sum(splits(uavs).values())))
                        self.assertEqual(tuple(discrete.shape), (8, uavs))
                        for sampled, recomputed in ((total_lp, evaluated[0]),
                                                    (cont_lp, evaluated[2]),
                                                    (disc_lp, evaluated[3])):
                            torch.testing.assert_close(sampled, recomputed, rtol=2e-5, atol=3e-5)
                            torch.testing.assert_close(torch.exp(recomputed - sampled),
                                                       torch.ones_like(sampled), rtol=0, atol=4e-5)
                        metadata = model.action_probability_metadata()
                        masked_dim = 2 * uavs * 10 if alpha == 0.0 else 0
                        self.assertEqual(metadata["masked_continuous_action_dim"], masked_dim)
                        self.assertEqual(metadata["active_continuous_action_dim"],
                                         sum(splits(uavs).values()) - masked_dim)
                        self.assertEqual(set(metadata["masked_continuous_heads"]),
                                         BEAM_HEADS if alpha == 0.0 else set())

    def test_masked_beam_perturbations_do_not_change_probability_entropy_or_losses(self):
        for distribution in ("beta", "gaussian"):
            with self.subTest(distribution=distribution):
                model = actor(0.0, distribution)
                with torch.no_grad():
                    continuous, discrete, _, old_cont, old_disc = model.sample(self.states)
                changed = alter_beams(continuous, model.continuous_action_splits)
                first = model.evaluate_actions(self.states, continuous, discrete)
                second = model.evaluate_actions(self.states, changed, discrete)
                for before, after in zip(first, second):
                    torch.testing.assert_close(before, after, rtol=0, atol=0)
                advantage = torch.linspace(-1, 1, 8).unsqueeze(-1)
                def losses(result):
                    cont_ratio = torch.exp(result[2] - old_cont)
                    disc_ratio = torch.exp(result[3] - old_disc)
                    def surrogate(ratio):
                        return -torch.min(ratio * advantage, ratio.clamp(0.8, 1.2) * advantage)
                    ppo = (surrogate(cont_ratio) + surrogate(disc_ratio)
                           - 0.05 * result[4] - 0.01 * result[5]).mean()
                    bc = -(result[2] + result[3]).mean()
                    kl = (old_cont + old_disc - result[0]).mean()
                    return torch.stack((ppo, bc, kl))
                torch.testing.assert_close(losses(first), losses(second), rtol=0, atol=0)
                losses(first)[:2].sum().backward()
                self.assertTrue(all(p.grad is None for p in beam_parameters(model).values()))
                self.assertIsNotNone(model.fc1.weight.grad)

    def test_nonzero_alpha_restores_beam_effect_and_gradient(self):
        for distribution in ("beta", "gaussian"):
            for alpha in (0.5, 1.0):
                with self.subTest(distribution=distribution, alpha=alpha):
                    model = actor(alpha, distribution)
                    continuous = torch.full((8, sum(splits().values())), 0.5)
                    discrete = torch.zeros((8, 4), dtype=torch.long)
                    initial = model.evaluate_actions(self.states, continuous, discrete)
                    changed = model.evaluate_actions(self.states, alter_beams(continuous, splits()), discrete)
                    self.assertGreater(float((initial[2] - changed[2]).abs().max()), 1.0)
                    (-(changed[0] + 0.05 * changed[4]).mean()).backward()
                    params = list(beam_parameters(model).values())
                    self.assertTrue(all(p.grad is not None for p in params))
                    self.assertGreater(sum(float(p.grad.abs().sum()) for p in params), 0.0)

    def test_bc_and_ppo_ignore_resumed_adam_momentum_for_masked_heads(self):
        for distribution in ("beta", "gaussian"):
            with self.subTest(distribution=distribution):
                active = agent(1.0, distribution)
                actions = torch.full((8, sum(splits(4, 3).values())), 0.23)
                discrete = torch.zeros((8, 4), dtype=torch.long)
                active.behavior_clone(self.states, actions, discrete, epochs=1, batch_size=8)
                masked = agent(0.0, distribution)
                masked.actor.load_state_dict(active.actor.state_dict(), strict=True)
                masked.actor_optimizer.load_state_dict(copy.deepcopy(active.actor_optimizer.state_dict()))
                params = beam_parameters(masked.actor)
                weights_before = {name: p.detach().clone() for name, p in params.items()}
                moments_before = {name: copy.deepcopy(masked.actor_optimizer.state[p])
                                  for name, p in params.items()}
                self.assertTrue(any(float(state["exp_avg"].abs().sum()) > 0
                                    for state in moments_before.values()))
                # Cover previously populated gradients as well as optimizer state.
                for p in params.values():
                    p.grad = torch.ones_like(p)
                masked.behavior_clone(self.states, actions, discrete, epochs=1, batch_size=8)
                self._assert_frozen(params, masked.actor_optimizer, weights_before, moments_before)
                with torch.no_grad():
                    continuous, choices, _, cont_lp, disc_lp = masked.actor.sample(self.states)
                transition = {
                    "states": self.states.numpy(),
                    "continuous_actions": continuous.numpy(),
                    "discrete_actions": choices.numpy(),
                    "next_states": (self.states + 0.01).numpy(),
                    "rewards": np.linspace(-1.0, 0.8, 8),
                    "old_cont_log_probs": cont_lp.numpy(),
                    "old_disc_log_probs": disc_lp.numpy(),
                    "dones": [0] * 7 + [1],
                    "real_dones": [0] * 7 + [1],
                }
                for p in params.values():
                    p.grad = torch.ones_like(p)
                recorder = Recorder()
                masked.update(transition, step=1, writer=recorder)
                self._assert_frozen(params, masked.actor_optimizer, weights_before, moments_before)
                self.assertTrue(all(np.isfinite(value) for value in recorder.values.values()))
                self.assertIn("Agent/KL_Divergence", recorder.values)

    def _assert_frozen(self, params, optimizer, weights_before, moments_before):
        for name, p in params.items():
            self.assertIsNone(p.grad, name)
            torch.testing.assert_close(p, weights_before[name], rtol=0, atol=0)
            for key, old_value in moments_before[name].items():
                current = optimizer.state[p][key]
                if torch.is_tensor(old_value):
                    torch.testing.assert_close(current, old_value, rtol=0, atol=0)
                else:
                    self.assertEqual(current, old_value)

    def test_legacy_parameter_schema_strict_load_and_default_alpha_one(self):
        for distribution in ("beta", "gaussian"):
            for uavs in (1, 4, 8):
                with self.subTest(distribution=distribution, uavs=uavs):
                    legacy = MultiHeadActor(7, 16, splits(uavs), [10] * uavs,
                                            continuous_dist_type=distribution)
                    masked = actor(0.0, distribution, uavs)
                    restored = actor(1.0, distribution, uavs)
                    state = legacy.state_dict()
                    self.assertEqual(set(state), set(masked.state_dict()))
                    self.assertFalse(any("mask" in name or "alpha_effective" in name for name in state))
                    masked.load_state_dict(state, strict=True)
                    restored.load_state_dict(masked.state_dict(), strict=True)
                    self.assertEqual(legacy.structured_beam_alpha_effective, 1.0)
                    self.assertEqual(restored.masked_continuous_head_names, ())
                    actions = torch.full((8, sum(splits(uavs).values())), 0.3)
                    choices = torch.zeros((8, uavs), dtype=torch.long)
                    for old, new in zip(legacy.evaluate_actions(self.states, actions, choices),
                                        restored.evaluate_actions(self.states, actions, choices)):
                        torch.testing.assert_close(old, new, rtol=0, atol=0)

    def test_sampler_and_learner_probability_contract(self):
        from algorithms.MHBPPO.MHBPPO_main import validate_sample, _TRANSITION_KEYS
        from algorithms.MHBPPO.MHBPPO_sampler import _choose_action
        learner = agent(0.0)
        sampler = actor(0.0, antennas=3)
        sampler.load_state_dict(learner.actor.state_dict(), strict=True)
        expected = learner.actor.action_probability_metadata()
        item = {
            "transition_dict": {key: [0] for key in _TRANSITION_KEYS},
            "action_traces": [{}],
            "action_probability_metadata": sampler.action_probability_metadata(),
        }
        self.assertEqual(validate_sample(item, expected), 1)
        raw_state = self.states[0].numpy()
        sampled, cont_lp, disc_lp = _choose_action(sampler, raw_state)
        with torch.no_grad():
            evaluated = learner.actor.evaluate_actions(
                self.states[:1], torch.as_tensor(sampled["continuous"]).unsqueeze(0),
                torch.as_tensor(sampled["discrete"]).unsqueeze(0),
            )
        self.assertAlmostEqual(cont_lp, float(evaluated[2]), places=5)
        self.assertAlmostEqual(disc_lp, float(evaluated[3]), places=5)
        legacy = copy.deepcopy(item)
        legacy.pop("action_probability_metadata")
        mismatched = copy.deepcopy(item)
        mismatched["action_probability_metadata"] = actor(1.0, antennas=3).action_probability_metadata()
        for invalid in (legacy, mismatched):
            with self.assertRaisesRegex(RuntimeError, "概率契约"):
                validate_sample(invalid, expected)

    def test_effective_alpha_matches_environment_clipping(self):
        self.assertEqual(actor(-0.3).structured_beam_alpha_effective, 0.0)
        self.assertEqual(actor(1.3).structured_beam_alpha_effective, 1.0)
        self.assertEqual(set(actor(-0.3).masked_continuous_head_names), BEAM_HEADS)
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "finite"):
                actor(value)


if __name__ == "__main__":
    unittest.main(verbosity=2)
