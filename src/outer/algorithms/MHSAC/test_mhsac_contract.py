"""Regression tests against the original MHSAC and the transition scheduler."""
import importlib.util
import random
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

OUTER = Path(__file__).resolve().parents[2]
if str(OUTER) not in sys.path:
    sys.path.insert(0, str(OUTER))
from algorithms.MHSAC import MHSAC_agent as migrated
from algorithms.MHSAC import MHSAC_main as main
from algorithms.MHSAC.normalization import Normalization, RewardScaling


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


old = load_file("old_mhsac_agent", OUTER.parent / "outer_old/algorithms/MHSAC/MHSAC_agent.py")


class Writer:
    def __init__(self):
        self.values = {}

    def add_scalar(self, name, value, step):
        self.values[name] = float(value)


class ContractTest(unittest.TestCase):
    def make_agent(self, module):
        return module.HSAC(5, 8, {"angle": 2, "power": 3}, [3, 2], np.zeros(5), np.ones(5),
                           3e-4, 3e-4, .99, .005, .2, .2, torch.device("cpu"), alpha_lr=3e-5)

    def test_actor_joint_q_temperature_and_update_match_old(self):
        torch.manual_seed(10)
        previous = self.make_agent(old)
        torch.manual_seed(10)
        current = self.make_agent(migrated)
        states = torch.tensor(np.random.default_rng(8).normal(size=(4, 5)), dtype=torch.float32)
        torch.manual_seed(20)
        a = previous.actor.sample(states)
        torch.manual_seed(20)
        b = current.actor.sample(states)
        for key in a:
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
        self.assertEqual(b["joint_probs"].shape, (4, 6))
        torch.testing.assert_close(b["joint_probs"].sum(-1), torch.ones(4))
        for name in ("critic1", "critic2", "critic1_target", "critic2_target"):
            torch.testing.assert_close(getattr(previous, name)(states, a["continuous_actions"]),
                                       getattr(current, name)(states, b["continuous_actions"]), rtol=0, atol=0)
        for idx in range(6):
            decoded = current.decode_discrete_index(idx)
            self.assertEqual(current.encode_discrete_action(decoded), idx)
            np.testing.assert_array_equal(decoded, previous.decode_discrete_index(idx))
        transition = {"states": states.numpy(), "continuous_actions": b["continuous_actions"].detach().numpy(),
                      "discrete_indices": np.array([0, 2, 4, 5]), "rewards": np.array([1., -1., .5, 0.]),
                      "next_states": (states * .5).numpy(), "dones": np.array([0., 0., 0., 1.])}
        writer_old, writer_new = Writer(), Writer()
        torch.manual_seed(99)
        previous.update(transition, writer=writer_old, step=0)
        torch.manual_seed(99)
        current.update(transition, writer=writer_new, step=0)
        self.assertEqual(writer_old.values, writer_new.values)
        for name in ("actor", "critic1", "critic2", "critic1_target", "critic2_target"):
            for left, right in zip(getattr(previous, name).parameters(), getattr(current, name).parameters()):
                torch.testing.assert_close(left, right, rtol=0, atol=0)
        torch.testing.assert_close(previous.log_alpha_continuous, current.log_alpha_continuous, rtol=0, atol=0)
        torch.testing.assert_close(previous.log_alpha_discrete, current.log_alpha_discrete, rtol=0, atol=0)

    def test_replay_and_per_transition_update_count(self):
        args = SimpleNamespace(batch_size=2)
        sac = SimpleNamespace(minimal_size=3, updates_per_step=2)
        replay = migrated.ReplayBuffer(10)
        normalizer = Normalization(2)
        normalizer(np.array([1., 2.]))
        normalizer(np.array([2., 4.]))
        class Agent:
            def __init__(self):
                self.steps = []
            def update(self, batch, writer, step, agent_name):
                self.steps.append(step)
                self.last = batch
        agent = Agent()
        random.seed(1)
        observed = []
        for i in range(5):
            transition = {"state": np.array([i, i]), "continuous_action": [0.5], "discrete_index": 0,
                          "reward": i, "next_state": [i + 1, i + 1], "done": False}
            observed.append(main.learn_one(replay, agent, normalizer, args, sac, transition, None, i))
        self.assertEqual(observed, [0, 0, 2, 2, 2])
        self.assertEqual(agent.steps, [2, 2, 3, 3, 4, 4])
        self.assertEqual(replay.size(), 5)
        np.testing.assert_allclose(agent.last["states"], normalizer(
            np.asarray(normalizer.running_ms.mean + agent.last["states"] * (normalizer.running_ms.std + 1e-8)),
            update=False), rtol=1e-6, atol=1e-6)

    def test_single_environment_transition_sequence_matches_old_loop(self):
        torch.manual_seed(123)
        previous = self.make_agent(old)
        torch.manual_seed(123)
        current = self.make_agent(migrated)
        old_replay, replay = old.ReplayBuffer(20), migrated.ReplayBuffer(20)
        old_n, normalizer = Normalization(5), Normalization(5)
        old_scaler, scaler = RewardScaling(1, .99), RewardScaling(1, .99)
        old_scaler.reset(); scaler.reset()
        raw = np.arange(5, dtype=np.float32)
        old_state, state = old_n(raw), normalizer(raw)
        args = SimpleNamespace(batch_size=2)
        sac = SimpleNamespace(minimal_size=2, updates_per_step=1)
        for step in range(4):
            torch.manual_seed(200 + step)
            action_old = previous.choose_action(old_state)
            torch.manual_seed(200 + step)
            action = current.choose_action(state)
            np.testing.assert_array_equal(action_old["continuous"], action["continuous"])
            np.testing.assert_array_equal(action_old["discrete"], action["discrete"])
            next_raw = raw + np.float32(step + 1)
            old_reward = float(old_scaler(float(step + 1)).item())
            new_reward = float(scaler(float(step + 1)).item())
            old_next, next_state = old_n(next_raw), normalizer(next_raw)
            old_replay.add(raw, action_old["continuous"], action_old["discrete_index"], old_reward, next_raw, step == 3)
            transition = {"state": raw, "continuous_action": action["continuous"],
                          "discrete_index": action["discrete_index"], "reward": new_reward,
                          "next_state": next_raw, "done": step == 3}
            writer_old, writer_new = Writer(), Writer()
            random.seed(300 + step); torch.manual_seed(400 + step)
            if old_replay.size() >= 2:
                batch = old_replay.sample(2)
                batch["states"] = np.asarray(old_n(batch["states"], update=False), dtype=np.float32)
                batch["next_states"] = np.asarray(old_n(batch["next_states"], update=False), dtype=np.float32)
                previous.update(batch, writer=writer_old, step=step)
            random.seed(300 + step); torch.manual_seed(400 + step)
            main.learn_one(replay, current, normalizer, args, sac, transition, writer_new, step)
            self.assertEqual(writer_old.values, writer_new.values)
            for left, right in zip(previous.actor.parameters(), current.actor.parameters()):
                torch.testing.assert_close(left, right, rtol=0, atol=0)
            np.testing.assert_array_equal(old_next, next_state)
            raw, old_state, state = next_raw, old_next, next_state

    def test_parallel_action_streams_are_independent_and_reproducible(self):
        class Agent:
            def choose_action(self, _):
                return torch.rand(4)
        seed_before = torch.get_rng_state().clone()
        stream = main.ActionRandomStreams(42, 2, torch.device("cpu"))
        first = stream.choose(0, Agent(), None)
        second = stream.choose(1, Agent(), None)
        self.assertFalse(torch.equal(first, second))
        self.assertTrue(torch.equal(seed_before, torch.get_rng_state()))
        repeat = main.ActionRandomStreams(42, 2, torch.device("cpu"))
        torch.testing.assert_close(first, repeat.choose(0, Agent(), None), rtol=0, atol=0)
        torch.testing.assert_close(second, repeat.choose(1, Agent(), None), rtol=0, atol=0)

    def test_defaults_and_overrides(self):
        args, sac = main.get_all_args([])
        self.assertEqual((args.batch_size, args.episodes, sac.minimal_size), (128, 8000, 8000))
        args, sac = main.get_all_args(["--batch_size", "2", "--minimal_size", "3", "--buffer_size", "10", "--updates_per_step", "2"])
        self.assertEqual((args.batch_size, sac.minimal_size, sac.updates_per_step), (2, 3, 2))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
