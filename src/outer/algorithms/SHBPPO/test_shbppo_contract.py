"""Compare migration with actual legacy SHBPPO and test the rollout contract."""
import ast
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
from algorithms.SHBPPO import SHBPPO_agent as migrated
from algorithms.SHBPPO import SHBPPO_main as main
from algorithms.SHBPPO import normalization as migrated_norm
from algorithms.MHBPPO.rollout_batch import actor_state_signature


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


OLD = OUTER.parent / "outer_old"
old = load_file("old_shbppo_agent", OLD / "algorithms/SHBPPO/SHBPPO_agent.py")
old_norm = load_file("old_shbppo_normalization", OLD / "utils/normalization.py")
old_tree = ast.parse((OLD / "algorithms/SHBPPO/SHBPPO_main.py").read_text(encoding="utf-8"))
mapping_node = next(node for node in old_tree.body if isinstance(node, ast.FunctionDef) and node.name == "split_and_discretize")
mapping_module = ast.Module(body=[mapping_node], type_ignores=[])
mapping_scope = {"np": np}
exec(compile(mapping_module, "legacy_shbppo_mapping", "exec"), mapping_scope)
old_mapping = mapping_scope["split_and_discretize"]


class Writer:
    def __init__(self):
        self.values = {}

    def add_scalar(self, name, value, step):
        self.values[name] = float(value)


class FakeEnv:
    def __init__(self, base_args, madrl_args, seed_offset):
        type(self).last_instance = self
        self.observation_space = {"bs": SimpleNamespace(shape=(3,))}
        self.bs_continuous_action_splits = {"x": 2}
        self.actions = []

    def reset(self, version):
        self.t = 0
        return {"bs": np.array([1, 2, 3], dtype=np.float32)}

    def step(self, actions, version):
        self.actions.append(actions["bs"])
        self.t += 1
        return ({"bs": np.array([1 + self.t, 2, 3], dtype=np.float32)},
                1.0, {"bs": 1.0, "components": {"solver_status_kind": "optimal"}},
                self.t == 2, 3.0, 1)

    def getPosUAV(self):
        return np.zeros((1, 3))

    def get_exogenous_scene_signature(self):
        return {"scene": "fake"}


class ContractTest(unittest.TestCase):
    def make_agent(self, module):
        return module.SHBPPO(5, 8, 7, np.zeros(5), np.ones(5), 3e-4, 3e-4,
                             0.95, 0.2, 0.99, 2, 100, torch.device("cpu"), entropy_coef=0.01)

    def test_actor_probability_entropy_and_update_match_legacy(self):
        torch.manual_seed(10)
        legacy = self.make_agent(old)
        torch.manual_seed(10)
        new = self.make_agent(migrated)
        rng = np.random.default_rng(101)
        data = {key: [] for key in main.TRANSITION_KEYS}
        for index in range(8):
            state = rng.normal(size=5).astype(np.float32)
            torch.manual_seed(100 + index)
            expected_action, expected_lp = legacy.choose_action(state)
            torch.manual_seed(100 + index)
            action, lp = new.choose_action(state)
            np.testing.assert_array_equal(expected_action, action)
            np.testing.assert_array_equal(expected_lp, lp)
            with torch.no_grad():
                old_eval = legacy.actor.evaluate_actions(torch.tensor(state).unsqueeze(0), torch.tensor(action).unsqueeze(0))
                new_eval = new.actor.evaluate_actions(torch.tensor(state).unsqueeze(0), torch.tensor(action).unsqueeze(0))
            for left, right in zip(old_eval, new_eval):
                torch.testing.assert_close(left, right, rtol=0, atol=0)
            self.assertAlmostEqual(float(lp.item()), float(new_eval[0].item()), places=5)
            for key, value in (("states", state), ("actions", action),
                ("next_states", rng.normal(size=5).astype(np.float32)), ("rewards", float(rng.normal())),
                ("old_log_probs", float(lp.item())), ("dones", index in (3, 7)),
                ("real_dones", index in (3, 7))):
                data[key].append(value)
        old_writer, new_writer = Writer(), Writer()
        torch.manual_seed(999)
        legacy.update(data, 0, old_writer)
        torch.manual_seed(999)
        new.update(data, 0, new_writer)
        self.assertEqual(old_writer.values, new_writer.values)
        for left, right in zip(legacy.actor.parameters(), new.actor.parameters()):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        for left, right in zip(legacy.critic.parameters(), new.critic.parameters()):
            torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_bins_match_actual_old_function(self):
        values = np.array([0, 0.25, 0, 0.099, 0.1, 0.999, 1], dtype=np.float32)
        expected = old_mapping(values, 2, 5, 10)
        actual = main.split_and_discretize(values, 2, 5, 10)
        for left, right in zip(expected, actual):
            np.testing.assert_array_equal(left, right)
        np.testing.assert_array_equal(actual[1], [0, 0, 1, 9, 9])

    def test_legacy_online_normalization_and_scaler(self):
        left, right = old_norm.Normalization(3), migrated_norm.Normalization(3)
        old_r, new_r = old_norm.RewardScaling(1, 0.99), migrated_norm.RewardScaling(1, 0.99)
        for episode in range(2):
            old_r.reset(); new_r.reset()
            for i in range(4):
                state = np.array([i, episode, i - episode], dtype=np.float32)
                np.testing.assert_array_equal(left(state), right(state))
                np.testing.assert_array_equal(old_r(i + 1.), new_r(i + 1.))

    def test_single_worker_keeps_raw_action_and_old_state_update_order(self):
        base = SimpleNamespace(seed=42, uavs_num=1, cus_num=10)
        args = SimpleNamespace(worker_num_threads=1, scene_seed_offset=0, num_samplers=1,
                               hidden_dim=8, gamma=0.99)
        actor = migrated.SingleHeadActor(3, 8, 3)
        control, results = queue.Queue(), queue.Queue()
        control.put({"version": 0, "actor": {k: v.detach().numpy().copy() for k, v in actor.state_dict().items()},
                     "signature": actor_state_signature(actor), "normalizer": {"n": 0, "mean": np.zeros(3), "std": np.zeros(3)}})
        control.put(None)
        with patch.object(main, "MyEnv", FakeEnv):
            main.worker_main(base, args, 0, control, results)
        item = results.get_nowait()
        self.assertNotIn("error", item)
        self.assertEqual(item["normalizer"]["n"], 3)  # reset plus two next states, not 2*T
        for state, action, probability in zip(item["transition"]["states"],
                item["transition"]["actions"], item["transition"]["old_log_probs"]):
            with torch.no_grad():
                lp, _ = actor.evaluate_actions(torch.tensor(state, dtype=torch.float32).unsqueeze(0),
                                                torch.tensor(action).unsqueeze(0))
            self.assertAlmostEqual(lp.item(), probability, places=5)
        for raw_action, env_action in zip(item["transition"]["actions"], FakeEnv.last_instance.actions):
            continuous, discrete = old_mapping(raw_action, 2, 1, 10)
            np.testing.assert_array_equal(env_action["continuous"], continuous)
            np.testing.assert_array_equal(env_action["discrete"], discrete)
        self.assertEqual(item["transition"]["dones"], [False, True])

    def test_multi_worker_uses_frozen_snapshot_and_saves_one_state_per_step(self):
        base = SimpleNamespace(seed=42, uavs_num=1, cus_num=10)
        args = SimpleNamespace(worker_num_threads=1, scene_seed_offset=0, num_samplers=2,
                               hidden_dim=8, gamma=0.99)
        actor = migrated.SingleHeadActor(3, 8, 3)
        control, results = queue.Queue(), queue.Queue()
        snapshot = {"n": 7, "mean": np.array([0.5, 1.0, 1.5]), "std": np.array([1.0, 2.0, 3.0])}
        control.put({"version": 2, "actor": {k: v.detach().numpy().copy() for k, v in actor.state_dict().items()},
                     "signature": actor_state_signature(actor), "normalizer": snapshot})
        control.put(None)
        with patch.object(main, "MyEnv", FakeEnv):
            main.worker_main(base, args, 1, control, results)
        item = results.get_nowait()
        self.assertNotIn("error", item)
        self.assertEqual(item["version"], 2)
        self.assertEqual(len(item["raw_states"]), 3)
        for state, raw in zip(item["transition"]["states"], item["raw_states"][:-1]):
            np.testing.assert_array_equal(state, main.frozen_normalize(raw, snapshot))
        for state, action, old_lp in zip(item["transition"]["states"], item["transition"]["actions"], item["transition"]["old_log_probs"]):
            with torch.no_grad():
                current_lp, _ = actor.evaluate_actions(torch.tensor(state, dtype=torch.float32).unsqueeze(0),
                                                       torch.tensor(action).unsqueeze(0))
            self.assertAlmostEqual(current_lp.item(), old_lp, places=5)

    def test_episode_boundaries_versions_and_defaults(self):
        episode = {"transition": {key: [False, True] if key in ("dones", "real_dones") else [1, 2]
                                  for key in main.TRANSITION_KEYS}}
        self.assertEqual(main.merge_episodes([episode, episode])["dones"], [False, True, False, True])
        class Alive:
            def is_alive(self):
                return True
        results = queue.Queue()
        results.put({"worker_id": 0, "version": 0, "signature": "old"})
        with self.assertRaisesRegex(RuntimeError, "version/signature"):
            main.collect_round(results, [Alive()], 1, "new")
        self.assertEqual(main.get_shbppo_args([]).episodes, 8000)
        self.assertEqual(main.get_shbppo_args(["--episodes", "123"]).episodes, 123)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
