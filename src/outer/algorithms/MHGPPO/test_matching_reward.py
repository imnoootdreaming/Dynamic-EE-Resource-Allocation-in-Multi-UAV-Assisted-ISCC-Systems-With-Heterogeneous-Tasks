"""Matching-loss and reward-contract checks for Gaussian and Pure Beta PPO."""
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

OUTER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(OUTER))
from algorithms.MHGPPO import MHGPPO_agent as gaussian, MHGPPO_main as gaussian_main
from algorithms.Pure_MHBPPO import Pure_MHBPPO_agent as beta, Pure_MHBPPO_main as beta_main
from algorithms.MHBPPO.MHBPPO_agent import MultiHeadActor as ReferenceActor
from algorithms.MHBPPO.MHBPPO_main import StateNormStats

VARIANTS = ((gaussian, gaussian_main, "gaussian", "MyEnv"),
            (beta, beta_main, "beta", "PureMyEnv"))


def make_agent(module, dist, coef=0.0, dims=(3, 3, 3, 3)):
    return module.HPPO(3, 8, {"x": 2}, dims, np.zeros(2), np.ones(2),
                       3e-4, 3e-4, 0.95, 0.2, 0.99, 2, 100, torch.device("cpu"),
                       continuous_dist_type=dist, continuous_entropy_coef=0.,
                       discrete_entropy_coef=0., expected_duplicate_matching_coef=coef)


def transition_for(agent):
    states = np.random.default_rng(1).normal(size=(16, 3)).astype(np.float32)
    with torch.no_grad():
        cont, disc, _, c_lp, d_lp = agent.actor.sample(torch.tensor(states))
    return dict(states=states, next_states=states, continuous_actions=cont.numpy(),
                discrete_actions=disc.numpy(), rewards=[0.]*16,
                old_cont_log_probs=c_lp.numpy(), old_disc_log_probs=d_lp.numpy(),
                dones=[True]*16, real_dones=[True]*16)


class FakeEnv:
    def __init__(self, base_args, madrl_args, seed_offset):
        self.base_args = base_args
        self.time_decoder = getattr(madrl_args, "pure_time_decoder", "legacy")
        self.observation_space = {"bs": SimpleNamespace(shape=(3,))}
        self.bs_continuous_action_splits = {"x": 2}
        self.action_space = {"bs": {"continuous": SimpleNamespace(low=np.zeros(2), high=np.ones(2)),
                                    "discrete": SimpleNamespace(nvec=np.full(4, 3))}}

    def reset(self, version):
        self.t = 0
        return {"bs": np.array([1, 2, 3], dtype=np.float32)}

    def step(self, actions, version):
        self.t += 1
        self.last_action_trace = {"time_slot": self.t-1}
        reward = [-2., 5.][self.t-1]
        return ({"bs": np.array([1+self.t, 2, 3], dtype=np.float32)}, reward,
                {"bs": reward, "components": {}}, self.t == 2, 1., 1)

    def getPosUAV(self):
        return np.zeros((4, 3))

    def get_exogenous_scene_signature(self):
        return {"signature": "fake"}


class MatchingTests(unittest.TestCase):
    def test_expectation_matches_enumeration_and_mhbppo_without_changing_density(self):
        for module, _, dist, _ in VARIANTS:
            with self.subTest(dist=dist):
                torch.manual_seed(11)
                actor = make_agent(module, dist).actor
                states = torch.randn(5, 3)
                actions = actor.sample(states)
                original = actor.evaluate_actions(states, actions[0], actions[1])
                extended = actor.evaluate_actions(states, actions[0], actions[1],
                    return_expected_duplicate_matching=True)
                for left, right in zip(original, extended):
                    torch.testing.assert_close(left, right, rtol=0, atol=0)
                reference = ReferenceActor(3, 8, {"x": 2}, [3]*4, continuous_dist_type=dist)
                reference.load_state_dict(actor.state_dict())
                expected = reference.evaluate_actions(states, actions[0], actions[1],
                    return_expected_duplicate_matching=True)
                torch.testing.assert_close(extended[6], expected[6], rtol=0, atol=0)
                _, distributions = actor._get_dists(states)
                p = torch.stack([d.probs for d in distributions], dim=1).detach().numpy()
                enumerated = np.zeros(5)
                for assignment in itertools.product(range(3), repeat=4):
                    probability = np.prod([p[:, i, j] for i, j in enumerate(assignment)], axis=0)
                    enumerated += probability*(4-len(set(assignment)))
                np.testing.assert_allclose(extended[6].detach().numpy().reshape(-1), enumerated,
                                           atol=3e-7, rtol=1e-6)

    def test_auxiliary_gradient_reaches_every_discrete_head(self):
        for module, _, dist, _ in VARIANTS:
            actor = make_agent(module, dist).actor
            with torch.no_grad():
                for i, head in enumerate(actor.discrete_heads):
                    head.bias.copy_(torch.tensor([0.6+0.1*i, -0.2*i, -0.3]))
            states = torch.randn(8, 3)
            sampled = actor.sample(states)
            penalty = actor.evaluate_actions(states, sampled[0], sampled[1],
                return_expected_duplicate_matching=True)[6].mean()
            penalty.backward()
            for head in actor.discrete_heads:
                self.assertTrue(torch.isfinite(head.weight.grad).all())
                self.assertGreater(head.weight.grad.abs().sum().item(), 0.)
            self.assertTrue(all(p.grad is None for p in actor.continuous_heads.parameters()))

    def test_auxiliary_loss_reduces_expected_duplicates_in_one_ppo_update(self):
        for module, _, dist, _ in VARIANTS:
            torch.manual_seed(42)
            agent = make_agent(module, dist, 0.3)
            for p in agent.critic.parameters():
                p.data.zero_()
            data = transition_for(agent)
            with torch.no_grad():
                before = agent.actor.evaluate_actions(torch.tensor(data["states"]),
                    torch.tensor(data["continuous_actions"]), torch.tensor(data["discrete_actions"]),
                    return_expected_duplicate_matching=True)[6].mean().item()
            metrics = Mock()
            after = agent.update(data, 0, metrics, "BS")
            self.assertLess(after, before)
            self.assertTrue(any(c.args[0] == "BS/actor_expected_duplicate_matching"
                                for c in metrics.add_scalar.call_args_list))

    def test_zero_coefficient_skips_auxiliary_computation(self):
        for module, _, dist, _ in VARIANTS:
            agent = make_agent(module, dist)
            data = transition_for(agent)
            evaluate = agent.actor.evaluate_actions
            def without_auxiliary(*args, **kwargs):
                self.assertFalse(kwargs.get("return_expected_duplicate_matching", False))
                return evaluate(*args, **kwargs)
            with patch.object(agent.actor, "evaluate_actions", side_effect=without_auxiliary):
                self.assertTrue(math.isnan(agent.update(data)))

    def test_invalid_weights_and_nonshared_cu_sets(self):
        for module, _, dist, _ in VARIANTS:
            for coef in (-1., float("nan"), float("inf")):
                with self.assertRaises(ValueError):
                    make_agent(module, dist, coef)
            with self.assertRaises(ValueError):
                make_agent(module, dist, 0.3, dims=(2, 3))
            make_agent(module, dist, 0., dims=(2, 3))  # Legacy heterogeneous heads remain supported.


class RewardTests(unittest.TestCase):
    def test_single_and_multi_workers_return_raw_rewards(self):
        for module, main, dist, env_name in VARIANTS:
            for workers in (1, 2):
                with self.subTest(dist=dist, workers=workers), ExitStack() as stack:
                    actor = make_agent(module, dist).actor
                    control, results = queue.Queue(), queue.Queue()
                    control.put({"version": 0, "actor": {k: v.detach().numpy() for k, v in actor.state_dict().items()},
                                 "signature": main.actor_state_signature(actor),
                                 "normalizer": {"n": 0, "mean": np.zeros(3), "std": np.ones(3)}})
                    control.put(None)
                    stack.enter_context(patch.object(main, env_name, FakeEnv))
                    stack.enter_context(patch.object(main, "capture", return_value=({}, {})))
                    if main is beta_main:
                        stack.enter_context(patch.object(main.signal, "signal"))
                    else:
                        stack.enter_context(patch.object(main.TrainingStop, "ignore_worker_interrupts"))
                    args = SimpleNamespace(worker_num_threads=1, scene_seed_offset=0, num_samplers=workers, hidden_dim=8)
                    main.worker_main(SimpleNamespace(seed=42), args, 0, control, results)
                    episode = results.get_nowait()
                    self.assertNotIn("error", episode, episode.get("error"))
                    self.assertEqual(episode["transition"]["rewards"], [-2., 5.])
                    self.assertEqual(episode["reward"], 1.5)

    def test_learners_use_previous_global_stats_and_log_raw_rewards(self):
        for _, main, dist, env_name in VARIANTS:
            self.assertIs(main.StateNormStats, StateNormStats)
            for workers in (1, 2):
                with self.subTest(dist=dist, workers=workers), tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
                    args = main.get_madrl_args(["--max_updates", "2", "--num_samplers", str(workers),
                        "--hidden_dim", "8", "--expected_duplicate_matching_coef", "0.3", "--log_min_free_bytes", "0"])
                    args.pure_time_decoder = "conditional"
                    root = Path(temporary)
                    batches, raw_batches, observed = [], [], []
                    for version in range(2):
                        episodes = []
                        for worker in range(workers):
                            rewards = [float(2*worker-version), float(2*worker+3+version)]
                            transition = {key: [0, 0] for key in main.TRANSITION_KEYS}
                            transition["rewards"] = rewards
                            episodes.append(dict(worker_id=worker, version=version, signature="fake", action_seed=worker,
                                transition=transition, scene_signature={"signature": "fake"}, detailed_traces=[], traces=[],
                                normalizer={"n": 3*(version+1), "mean": np.zeros(3), "std": np.ones(3), "S": np.ones(3)},
                                raw_states=[np.ones(3)]*3, raw_pairs=[(np.ones(3), np.ones(3))]*2,
                                reward=np.mean(rewards), successes=[1, 0], objectives=[1., 2.], solver_statuses=["optimal"]*2))
                        batches.append(episodes)
                        raw_batches.append(np.array([r for ep in episodes for r in ep["transition"]["rewards"]]).reshape(-1, 1))
                    def update(agent, transition, *args, **kwargs):
                        observed.append(np.array(transition["rewards"]))
                        self.assertEqual(agent.expected_duplicate_matching_coef, 0.3)
                        return 0.125
                    stack.enter_context(patch.object(main, "get_base_args", return_value=SimpleNamespace(seed=42)))
                    stack.enter_context(patch.object(main, env_name, FakeEnv))
                    stack.enter_context(patch.object(main, "setup_run", return_value=(root, root, root, {"training_log": str(root/"train.log")}, {})))
                    stack.enter_context(patch.object(main.mp, "get_context", return_value=Mock()))
                    stack.enter_context(patch.object(main, "collect_round", side_effect=batches))
                    stack.enter_context(patch.object(main, "check_scene"))
                    trace = stack.enter_context(patch.object(main, "TraceLogs"))
                    trace.return_value.begin_round.return_value = None
                    if main is beta_main:
                        physical = stack.enter_context(patch.object(main, "PhysicalTraceWriter"))
                        physical.return_value.size = 0
                    stack.enter_context(patch.object(main, "save_checkpoint"))
                    stack.enter_context(patch.object(main, "save_reward_artifacts"))
                    stack.enter_context(patch.object(main, "aggregate_metrics", return_value={}))
                    stack.enter_context(patch("torch.utils.tensorboard.SummaryWriter"))
                    stack.enter_context(patch.object(main.HPPO, "update", new=update))
                    main.run_training(args, output_dir=root)
                    with (root/"training_updates.csv").open(encoding="utf-8") as stream:
                        rows = list(csv.DictReader(stream))
                    stats = StateNormStats(1)
                    for i, raw in enumerate(raw_batches):
                        np.testing.assert_array_equal(observed[i], ((raw-stats.mean)/(stats.std+1e-8)).reshape(-1))
                        self.assertEqual(float(rows[i]["reward"]), float(raw.mean()))
                        self.assertEqual(float(rows[i]["actor_expected_duplicate_matching"]), 0.125)
                        stats.update(raw)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
