"""Editable JSON recipes must match actual startup/update settings."""
import contextlib
import io
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

OUTER = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(OUTER))
from configs.madrl_args import get_madrl_args
from configs.base_params import get_base_args
from algorithms.MHBPPO.MHBPPO_agent import HPPO
from algorithms.MHBPPO.MHBPPO_main import save_hyperparameters
import algorithms.MHBPPO.MHBPPO_agent as agent_module


def make_agent(args):
    parts = dict(uav_angles=1, uav_distances=1, uav_off_durations=1,
                 cu_off_powers=2, uav_rec_beam_dir_real=2, uav_rec_beam_dir_imag=2)
    dimension = sum(parts.values())
    return HPPO(7, args.hidden_dim, parts, [2], np.zeros(dimension), np.ones(dimension),
                args.actor_lr, args.critic_lr, args.lmbda, args.eps, args.gamma,
                args.epochs, args.episodes, torch.device("cpu"),
                batch_size=args.batch_size, continuous_entropy_coef=args.continuous_entropy_coef,
                discrete_entropy_coef=args.discrete_entropy_coef, lr_min_ratio=args.lr_min_ratio)


class TrainingConfigurationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_defaults_and_json_cli_precedence(self):
        args = get_madrl_args([])
        self.assertEqual(args.batch_size, 32)
        self.assertEqual(args.continuous_entropy_coef, .05)
        self.assertEqual(args.discrete_entropy_coef, .01)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            path.write_text(json.dumps({"hyperparameters": dict(actor_lr=.0001, batch_size=7,
                hidden_dim=16, continuous_entropy_coef=.02, discrete_entropy_coef=.005,
                sync_training=False)}), encoding="utf8")
            parsed = get_madrl_args(["--config", str(path), "--actor_lr", ".0002", "--sync_training"])
            self.assertEqual(parsed.actor_lr, .0002)
            self.assertEqual(parsed.batch_size, 7)
            self.assertEqual(parsed.hidden_dim, 16)
            self.assertEqual(parsed.continuous_entropy_coef, .02)
            self.assertTrue(parsed.sync_training)

    def test_saved_recipe_roundtrips_and_effective_values_are_actual(self):
        args = get_madrl_args(["--hidden_dim", "16", "--batch_size", "64", "--num_samplers", "5"])
        learner = make_agent(args)
        with tempfile.TemporaryDirectory() as directory:
            path = save_hyperparameters(directory, get_base_args(), args, learner)
            data = json.loads(Path(path).read_text(encoding="utf8"))
            self.assertEqual(data["effective"]["num_samplers"], 1)
            self.assertEqual(data["effective"]["batch_size"], 64)
            self.assertEqual(data["effective"]["mini_batch_size_first_rollout"], 40)
            self.assertEqual(data["effective"]["hidden_dim"], learner.actor.fc1.out_features)
            self.assertIsInstance(data["environment"]["inner_parameters"]["L_cu_task"], list)
            reloaded = get_madrl_args(["--config", path])
            self.assertEqual({k:v for k,v in vars(reloaded).items() if k != "config"},
                             {k:v for k,v in vars(args).items() if k != "config"})
            self.assertFalse(Path(path + ".tmp").exists())

    def test_invalid_json_configuration_is_rejected(self):
        cases = [dict(batch_size=0), dict(batch_size=2.5), dict(hidden_dim=-1),
                 dict(continuous_entropy_coef=-.1), dict(sync_training="false"),
                 dict(actor_lr=float("nan")), dict(unknown_parameter=1)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            for values in cases:
                with self.subTest(values=values):
                    path.write_text(json.dumps(values), encoding="utf8")
                    with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                        get_madrl_args(["--config", str(path)])

    def test_configured_batch_size_is_used_by_ppo_update(self):
        args = get_madrl_args(["--hidden_dim", "16", "--batch_size", "7", "--epochs", "1"])
        learner = make_agent(args)
        states = torch.randn(20, 7)
        with torch.no_grad():
            continuous, discrete, _, clp, dlp = learner.actor.sample(states)
        transition = dict(states=states.numpy(), next_states=states.numpy(),
            continuous_actions=continuous.numpy(), discrete_actions=discrete.numpy(),
            rewards=np.linspace(-1, 1, 20), old_cont_log_probs=clp.numpy().reshape(-1),
            old_disc_log_probs=dlp.numpy().reshape(-1), dones=[False]*19+[True],
            real_dones=[False]*19+[True])
        sizes = []
        original = agent_module.BatchSampler
        def capture(sampler, size, drop_last):
            sizes.append(size)
            return original(sampler, size, drop_last)
        with mock.patch.object(agent_module, "BatchSampler", side_effect=capture), \
             mock.patch.object(learner.actor_optimizer, "step", wraps=learner.actor_optimizer.step) as step:
            learner.update(transition)
        self.assertEqual(sizes, [7])
        self.assertEqual(step.call_count, 3)

    def test_real_entry_writes_json_before_starting_workers(self):
        class StartupComplete(Exception):
            pass
        args = get_madrl_args(["--hidden_dim", "16", "--batch_size", "9",
            "--continuous_entropy_coef", ".017", "--discrete_entropy_coef", ".003",
            "--actor_lr", ".0002", "--critic_lr", ".0004", "--num_samplers", "5"])
        main_file = OUTER / "algorithms" / "MHBPPO" / "MHBPPO_main.py"
        original_join = os.path.join
        with tempfile.TemporaryDirectory() as directory:
            def scoped_join(*parts):
                if len(parts) == 4 and parts[1:3] == ("ckpt", "MHBPPO"):
                    return original_join(directory, "run")
                if len(parts) == 2 and str(parts[1]).startswith("runs/beta-hppo/"):
                    return original_join(directory, "tensorboard")
                return original_join(*parts)
            with mock.patch("configs.madrl_args.get_madrl_args", return_value=args), \
                 mock.patch("torch.utils.tensorboard.SummaryWriter"), \
                 mock.patch("os.path.join", side_effect=scoped_join), \
                 mock.patch("multiprocessing.get_context", side_effect=StartupComplete):
                with self.assertRaises(StartupComplete):
                    runpy.run_path(str(main_file), run_name="__main__")
            data = json.loads((Path(directory) / "run" / "hyperparameters.json").read_text(encoding="utf8"))
            for key in ("actor_lr", "critic_lr", "hidden_dim", "batch_size",
                        "continuous_entropy_coef", "discrete_entropy_coef"):
                self.assertEqual(data["effective"][key], getattr(args, key))
            self.assertEqual(data["effective"]["num_samplers"], 1)
            self.assertTrue((Path(directory) / "run" / "env.png").exists())


if __name__ == "__main__":
    unittest.main()
