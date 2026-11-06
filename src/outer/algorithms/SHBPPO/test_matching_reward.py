"""Numerical/gradient checks and worker-to-learner reward contract tests."""
import csv
import itertools
import math
import queue
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch
from scipy.special import betainc

OUTER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(OUTER))
from algorithms.SHBPPO import SHBPPO_main as main
from algorithms.SHBPPO.SHBPPO_agent import SHBPPO, SingleHeadActor
from algorithms.SHBPPO.matching import beta_bin_probabilities, expected_duplicate_matching
from algorithms.MHBPPO.MHBPPO_main import StateNormStats


def make_agent(coef=0.0):
    return SHBPPO(3, 8, 6, np.zeros(2), np.ones(2), 3e-4, 3e-4,
                  0.95, 0.2, 0.99, 2, 100, torch.device("cpu"),
                  entropy_coef=0.0, uavs_num=4, cus_num=3,
                  expected_duplicate_matching_coef=coef)


class FakeEnv:
    def __init__(self, base_args, madrl_args, seed_offset):
        self.base_args = base_args
        self.observation_space = {"bs": SimpleNamespace(shape=(3,))}
        self.bs_continuous_action_splits = {"x": 2}
        self.action_space = {"bs": {"continuous": SimpleNamespace(low=np.zeros(2), high=np.ones(2)),
                                    "discrete": SimpleNamespace(nvec=np.full(4, 3))}}

    def reset(self, version):
        self.t = 0
        return {"bs": np.array([1, 2, 3], dtype=np.float32)}

    def step(self, actions, version):
        self.t += 1
        reward = [-2.0, 5.0][self.t - 1]
        return ({"bs": np.array([1+self.t, 2, 3], dtype=np.float32)}, reward,
                {"bs": reward, "components": {}}, self.t == 2, 1.0, 1)

    def getPosUAV(self):
        return np.zeros((4, 3))

    def get_exogenous_scene_signature(self):
        return {"signature": "fake"}


class MatchingTests(unittest.TestCase):
    def test_cu_bin_mass_against_scipy(self):
        rng = np.random.default_rng(101)
        a = np.r_[1+10**rng.uniform(-5, 4, 80), [1., 2., 32., 1e4, 1e4]]
        b = np.r_[1+10**rng.uniform(-5, 4, 80), [1., 3., 32., 1.001, 1e4]]
        for bins in (2, 3, 10):
            actual = beta_bin_probabilities(torch.tensor(a), torch.tensor(b), bins)
            reference = np.diff(betainc(a[:, None], b[:, None], np.arange(bins+1)/bins), axis=-1)
            np.testing.assert_allclose(actual, reference, rtol=1e-7, atol=2e-10)
            np.testing.assert_allclose(actual.sum(-1), 1., atol=2e-15)

    def test_gradients_against_finite_differences(self):
        a = torch.tensor([[1.001, 2., 32., 100.]], dtype=torch.float64, requires_grad=True)
        b = torch.tensor([[100., 3., 32., 1.1]], dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(
            lambda a, b: beta_bin_probabilities(a, b, 10), (a, b), atol=1e-6, rtol=1e-4))
        weights = np.arange(10)**2
        loss = (beta_bin_probabilities(a, b, 10)*torch.tensor(weights)).sum()
        gradients = torch.autograd.grad(loss, (a, b))
        for shape_index, gradient in enumerate(gradients):
            arrays = [a.detach().numpy().copy(), b.detach().numpy().copy()]
            for index in range(4):
                h = 1e-4
                arrays[shape_index][0, index] += h
                plus = np.diff(betainc(arrays[0][..., None], arrays[1][..., None], np.arange(11)/10), axis=-1)
                arrays[shape_index][0, index] -= 2*h
                minus = np.diff(betainc(arrays[0][..., None], arrays[1][..., None], np.arange(11)/10), axis=-1)
                arrays[shape_index][0, index] += h
                reference = ((plus-minus)*weights).sum()/(2*h)
                self.assertAlmostEqual(gradient[0, index].item(), reference, delta=2e-6)

    def test_expected_excess_against_all_81_assignments(self):
        a = torch.tensor([[1.2, 2.3, 4.5, 6.7]], dtype=torch.float64)
        b = torch.tensor([[4.1, 3.2, 2.8, 1.1]], dtype=torch.float64)
        p = beta_bin_probabilities(a, b, 3)
        reference = 0.0
        for assignment in itertools.product(range(3), repeat=4):
            probability = math.prod(p[0, i, j].item() for i, j in enumerate(assignment))
            reference += probability*(4-len(set(assignment)))
        self.assertAlmostEqual(expected_duplicate_matching(p).item(), reference, places=12)
        self.assertEqual(expected_duplicate_matching(p[:, :1]).item(), 0.0)
        self.assertEqual(expected_duplicate_matching(beta_bin_probabilities(a, b, 1)).item(), 3.0)

    def test_actor_probability_unchanged_and_all_cu_shapes_receive_gradient(self):
        torch.manual_seed(13)
        actor = SingleHeadActor(3, 8, 6, uavs_num=4, cus_num=3)
        states = torch.randn(8, 3)
        actions, old_lp, entropy = actor.sample(states)
        basic = actor.evaluate_actions(states, actions)
        extra = actor.evaluate_actions(states, actions, return_expected_duplicate_matching=True)
        torch.testing.assert_close(old_lp, basic[0])
        torch.testing.assert_close(entropy, basic[1])
        for left, right in zip(basic, extra):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        extra[2].mean().backward()
        for layer in (actor.beta_head.alpha_layer, actor.beta_head.beta_layer):
            self.assertTrue(torch.all(layer.bias.grad[-4:].abs() > 1e-8))
            self.assertTrue(torch.all(layer.bias.grad[:2] == 0))

    def test_auxiliary_loss_lowers_duplicates_in_one_ppo_update(self):
        torch.manual_seed(42)
        agent = make_agent(0.3)
        for parameter in agent.critic.parameters():
            parameter.data.zero_()
        states = torch.randn(16, 3).numpy()
        with torch.no_grad():
            actions, log_probs, _ = agent.actor.sample(torch.tensor(states))
            before = agent.actor.evaluate_actions(torch.tensor(states), actions,
                return_expected_duplicate_matching=True)[2].mean().item()
        data = dict(states=states, actions=actions.numpy(), next_states=states, rewards=[0.]*16,
                    old_log_probs=log_probs.numpy(), dones=[True]*16, real_dones=[True]*16)
        after = agent.update(data)
        self.assertLess(after, before)

    def test_zero_coefficient_does_not_evaluate_auxiliary_probabilities(self):
        agent = make_agent()
        states = np.random.default_rng(1).normal(size=(8, 3)).astype(np.float32)
        with torch.no_grad():
            actions, lp, _ = agent.actor.sample(torch.tensor(states))
        data = dict(states=states, next_states=states, actions=actions.numpy(),
                    rewards=list(range(8)), old_log_probs=lp.numpy(), dones=[True]*8, real_dones=[True]*8)
        with patch("algorithms.SHBPPO.SHBPPO_agent.beta_bin_probabilities", side_effect=AssertionError):
            self.assertTrue(math.isnan(agent.update(data)))


class RewardTests(unittest.TestCase):
    def test_workers_return_raw_rewards_for_one_and_multiple_workers(self):
        for workers in (1, 2):
            base = SimpleNamespace(seed=42, uavs_num=4, cus_num=3)
            args = SimpleNamespace(worker_num_threads=1, scene_seed_offset=0, num_samplers=workers, hidden_dim=8)
            actor = SingleHeadActor(3, 8, 6)
            control, results = queue.Queue(), queue.Queue()
            control.put({"version": 0, "actor": {k: v.detach().numpy() for k, v in actor.state_dict().items()},
                         "signature": main.actor_state_signature(actor),
                         "normalizer": {"n": 0, "mean": np.zeros(3), "std": np.ones(3)}})
            control.put(None)
            with patch.object(main, "MyEnv", FakeEnv), patch.object(main, "capture", return_value=({}, {})), \
                    patch.object(main.TrainingStop, "ignore_worker_interrupts"):
                main.worker_main(base, args, 0, control, results)
            episode = results.get_nowait()
            self.assertNotIn("error", episode, episode.get("error"))
            self.assertEqual(episode["transition"]["rewards"], [-2., 5.])
            self.assertEqual(episode["reward"], 1.5)

    def test_learner_uses_previous_global_stats_and_logs_raw_rewards(self):
        self.assertIs(main.StateNormStats, StateNormStats)
        for workers in (1, 2):
            args = main.get_shbppo_args(["--max_updates", "2", "--num_samplers", str(workers)])
            args.hidden_dim = 8
            base = SimpleNamespace(seed=42, uavs_num=4, cus_num=3)
            batches, raw_batches = [], []
            for version in range(2):
                episodes = []
                for worker in range(workers):
                    rewards = [float(2*worker-version), float(2*worker+3+version)]
                    transition = {key: [0, 0] for key in main.TRANSITION_KEYS}
                    transition["rewards"] = rewards
                    episodes.append(dict(worker_id=worker, version=version, signature="fake", action_seed=worker,
                        transition=transition, scene_signature={"signature": "fake"}, detailed_traces=[],
                        normalizer={"n": 3*(version+1), "mean": np.zeros(3), "std": np.ones(3), "S": np.ones(3)},
                        raw_states=[np.ones(3)]*3, reward=np.mean(rewards), successes=[1, 0], objectives=[1., 2.],
                        solver_statuses=["optimal"]*2))
                raw_batches.append(np.array([r for ep in episodes for r in ep["transition"]["rewards"]]).reshape(-1, 1))
                batches.append(episodes)
            observed = []
            def update(agent, transition, *args, **kwargs):
                observed.append(np.array(transition["rewards"]))
                return float("nan")
            with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
                root = Path(temporary)
                paths = {"training_log": str(root/"train.log")}
                context = Mock()
                stack.enter_context(patch.object(main, "get_base_args", return_value=base))
                stack.enter_context(patch.object(main, "MyEnv", FakeEnv))
                stack.enter_context(patch.object(main, "setup_run", return_value=(root, root, root, paths, {})))
                stack.enter_context(patch.object(main.mp, "get_context", return_value=context))
                stack.enter_context(patch.object(main, "collect_round", side_effect=batches))
                stack.enter_context(patch.object(main, "check_scene"))
                traces = stack.enter_context(patch.object(main, "TraceLogs"))
                traces.return_value.begin_round.return_value = None
                stack.enter_context(patch.object(main, "save_checkpoint"))
                stack.enter_context(patch.object(main, "save_reward_artifacts"))
                stack.enter_context(patch.object(main, "aggregate_metrics", return_value={}))
                stack.enter_context(patch("torch.utils.tensorboard.SummaryWriter"))
                stack.enter_context(patch.object(main.SHBPPO, "update", new=update))
                main.run_training(args, output_dir=root)
                with (root/"training_updates.csv").open(encoding="utf-8") as stream:
                    rows = list(csv.DictReader(stream))
                stats = StateNormStats(1)
                for idx, raw in enumerate(raw_batches):
                    np.testing.assert_array_equal(observed[idx], ((raw-stats.mean)/(stats.std+1e-8)).reshape(-1))
                    self.assertEqual(float(rows[idx]["reward"]), float(raw.mean()))
                    stats.update(raw)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
