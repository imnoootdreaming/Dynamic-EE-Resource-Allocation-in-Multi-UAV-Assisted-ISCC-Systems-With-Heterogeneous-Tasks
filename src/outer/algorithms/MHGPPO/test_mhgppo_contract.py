"""Regression checks against the actual outer_old implementation."""
import importlib.util
import queue
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

OUTER = Path(__file__).resolve().parents[2]
if str(OUTER) not in sys.path:
    sys.path.insert(0, str(OUTER))
from algorithms.MHGPPO import MHGPPO_agent as migrated
from algorithms.MHGPPO import normalization as migrated_norm
from algorithms.MHGPPO.MHGPPO_main import merge_episodes, frozen_normalize, collect_round


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


old = load_file("old_mhgppo_agent", OUTER.parent / "outer_old/algorithms/MHBPPO/MHBPPO_agent.py")
old_norm = load_file("old_mhgppo_norm", OUTER.parent / "outer_old/utils/normalization.py")


class Writer:
    def __init__(self):
        self.values = {}

    def add_scalar(self, name, value, step):
        self.values[name] = float(value)


class ContractTest(unittest.TestCase):
    def make_agent(self, module):
        return module.HPPO(5, 8, {"angle": 2, "power": 3}, [3, 2], np.zeros(5),
                           np.ones(5), 0.0003, 0.0003, 0.95, 0.2, 0.99, 2, 100,
                           torch.device("cpu"), continuous_dist_type="gaussian")

    def test_old_actor_and_update(self):
        torch.manual_seed(10)
        legacy = self.make_agent(old)
        torch.manual_seed(10)
        new = self.make_agent(migrated)
        generator = np.random.default_rng(101)
        data = {key: [] for key in ("states", "continuous_actions", "discrete_actions", "next_states",
                "rewards", "old_cont_log_probs", "old_disc_log_probs", "dones", "real_dones")}
        for index in range(8):
            state = generator.normal(size=5).astype(np.float32)
            torch.manual_seed(100 + index)
            expected = legacy.choose_action(state)
            torch.manual_seed(100 + index)
            actual = new.choose_action(state)
            with torch.no_grad():
                evaluated = new.actor.evaluate_actions(torch.tensor(state).unsqueeze(0),
                    torch.tensor(actual[0]["continuous"]).unsqueeze(0),
                    torch.tensor(actual[0]["discrete"]).unsqueeze(0))
            np.testing.assert_allclose(evaluated[2].numpy().item(), actual[2].item(), atol=1e-4)
            np.testing.assert_allclose(evaluated[3].numpy().item(), actual[3].item(), atol=1e-5)
            for a, b in zip(expected[1:], actual[1:]):
                np.testing.assert_array_equal(a, b)
            for key in expected[0]:
                np.testing.assert_array_equal(expected[0][key], actual[0][key])
            data["states"].append(state)
            data["continuous_actions"].append(actual[0]["continuous"])
            data["discrete_actions"].append(actual[0]["discrete"])
            data["next_states"].append(generator.normal(size=5).astype(np.float32))
            data["rewards"].append(float(generator.normal()))
            data["old_cont_log_probs"].append(float(actual[2].item()))
            data["old_disc_log_probs"].append(float(actual[3].item()))
            data["dones"].append(index in (3, 7))
            data["real_dones"].append(index in (3, 7))
        previous, current = Writer(), Writer()
        torch.manual_seed(999)
        legacy.update(data, 0, previous)
        torch.manual_seed(999)
        new.update(data, 0, current)
        self.assertEqual(previous.values, current.values)
        for left, right in zip(legacy.actor.parameters(), new.actor.parameters()):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        for left, right in zip(legacy.critic.parameters(), new.critic.parameters()):
            torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_online_normalization_and_reward_scaler(self):
        left, right = old_norm.Normalization(3), migrated_norm.Normalization(3)
        old_r, new_r = old_norm.RewardScaling(1, 0.99), migrated_norm.RewardScaling(1, 0.99)
        for episode in range(2):
            old_r.reset()
            new_r.reset()
            for i in range(4):
                x = np.array([i, episode, i - episode], dtype=np.float32)
                np.testing.assert_array_equal(left(x), right(x))
                np.testing.assert_array_equal(old_r(float(i + 1)), new_r(float(i + 1)))
        for key in ("mean", "std", "S", "n"):
            np.testing.assert_array_equal(getattr(left.running_ms, key), getattr(right.running_ms, key))

    def test_episode_boundaries_and_initial_snapshot(self):
        episode = {"transition": {key: [False, True] if key in ("dones", "real_dones") else [1, 2]
                   for key in ("states", "continuous_actions", "discrete_actions", "next_states", "rewards",
                               "old_cont_log_probs", "old_disc_log_probs", "dones", "real_dones")}}
        merged = merge_episodes([episode, episode])
        self.assertEqual(merged["dones"], [False, True, False, True])
        value = frozen_normalize(np.array([1.0, 2.0]), {"n": 0, "mean": np.zeros(2), "std": np.zeros(2)})
        np.testing.assert_allclose(value, [1, 2], rtol=1e-7)

    def test_reject_wrong_rollout_policy(self):
        class Alive:
            def is_alive(self):
                return True
        results = queue.Queue()
        results.put({"worker_id": 0, "version": 1, "signature": "older"})
        with self.assertRaisesRegex(RuntimeError, "version/signature"):
            collect_round(results, [Alive()], 2, "expected")


if __name__ == "__main__":
    unittest.main()
