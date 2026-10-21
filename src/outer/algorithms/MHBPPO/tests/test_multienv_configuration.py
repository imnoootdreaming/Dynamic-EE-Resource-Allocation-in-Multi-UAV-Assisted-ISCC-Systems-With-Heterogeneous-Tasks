"""Merged fixed-scene settings and process-local MOSEK resource settings."""
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import mosek
import numpy as np

OUTER = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(OUTER))
from configs.base_params import get_base_args
from configs.madrl_args import get_madrl_args, validate_parallel_settings
from reward.cccp_bridge import build_inner_parameters, InnerContext, InnerVariables
import fusion_pc3p as fusion


class ParallelConfigurationTests(unittest.TestCase):
    def test_help_exits_successfully_and_renders_literal_percent(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), self.assertRaises(SystemExit) as stopped:
            get_madrl_args(["--help"])
        self.assertEqual(stopped.exception.code, 0)
        self.assertIn("--scene_seed_offset", output.getvalue())
        self.assertIn("--num_samplers", output.getvalue())
        self.assertIn("10%", output.getvalue())
        self.assertNotIn("10%%", output.getvalue())

    def parse_json(self, settings, overrides=()):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({"hyperparameters": settings}), encoding="utf8")
            return get_madrl_args(["--config", str(path), *overrides])

    def assert_rejected(self, argv):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            get_madrl_args(argv)

    def test_old_partial_config_uses_new_defaults_without_changing_ppo(self):
        args = self.parse_json(dict(hidden_dim=16))
        self.assertEqual((args.seed, args.scene_seed_offset, args.num_samplers), (42, 0, 1))
        self.assertEqual((args.worker_num_threads, args.mosek_num_threads), (0, 0))
        self.assertEqual((args.epochs, args.batch_size), (5, 32))
        self.assertEqual((args.structured_beam_alpha, args.sensing_penalty_coef,
                          args.sensing_penalty_reference_factor), (1.0, 1.0, 10.0))
        self.assertEqual(get_base_args().seed, 1208)
        validate_parallel_settings(SimpleNamespace(num_samplers=1))

    def test_json_then_cli_override_precedes_scene_conflict_validation(self):
        args = self.parse_json(dict(seed=123, scene_seed_offset=4, num_samplers=2,
                                   worker_num_threads=2, mosek_num_threads=3,
                                   randomize_nlos_per_episode=True),
                               ["--seed", "42", "--scene_seed_offset", "0",
                                "--mosek_num_threads", "1", "--no-randomize_nlos_per_episode"])
        self.assertEqual((args.seed, args.scene_seed_offset, args.num_samplers), (42, 0, 2))
        self.assertEqual((args.worker_num_threads, args.mosek_num_threads), (2, 1))
        self.assertFalse(args.randomize_nlos_per_episode)

    def test_each_randomize_flag_conflicts_only_with_synchronous_multi_env(self):
        flags = ("randomize_layout_per_episode", "randomize_cu_traj_per_episode",
                 "randomize_nlos_per_episode")
        for flag in flags:
            with self.subTest(flag=flag):
                self.assert_rejected(["--num_samplers", "2", "--" + flag])
                self.assertTrue(getattr(get_madrl_args(["--num_samplers", "1", "--" + flag]), flag))
                self.assertTrue(getattr(get_madrl_args(["--num_samplers", "2",
                    "--no-sync_training", "--" + flag]), flag))
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    self.parse_json({"num_samplers": 2, flag: True})

    def test_integer_ranges_and_json_types_fail_before_process_startup(self):
        cases = (dict(seed=-1), dict(seed=2**32), dict(scene_seed_offset=-1),
                 dict(num_samplers=0), dict(num_samplers=-1), dict(worker_num_threads=-1),
                 dict(mosek_num_threads=-1), dict(num_samplers=True),
                 dict(seed=False), dict(scene_seed_offset=0.5),
                 dict(worker_num_threads="1"), dict(mosek_num_threads=1.0))
        for values in cases:
            with self.subTest(values=values), contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit):
                self.parse_json(values)
        for argv in (["--num_samplers", "0"], ["--mosek_num_threads", "-1"],
                     ["--scene_seed_offset", "-1"], ["--worker_num_threads", "1.5"]):
            with self.subTest(argv=argv):
                self.assert_rejected(argv)
        validate_parallel_settings(SimpleNamespace(seed=0, scene_seed_offset=0,
            num_samplers=4, worker_num_threads=0, mosek_num_threads=0))
        self.assertEqual(get_madrl_args(["--seed", str(2**32-1)]).seed, 2**32-1)
        self.assert_rejected(["--seed", str(2**32-1), "--no-sync_training"])
        self.assert_rejected(["--seed", str(2**32-2), "--num_samplers", "2", "--no-sync_training"])
        self.assertEqual(get_madrl_args(["--seed", str(2**32-2), "--num_samplers", "1",
                                         "--no-sync_training"]).seed, 2**32-2)


class MosekThreadConfigurationTests(unittest.TestCase):
    @staticmethod
    def context(threads=None):
        args = SimpleNamespace(uavs_num=1, cus_num=1, antenna_nums=2)
        if threads is not None:
            args.mosek_num_threads = threads
        params = build_inner_parameters(args)
        return InnerContext(params, None, InnerVariables.create(params))

    def test_bridge_defaults_and_does_not_mutate_or_change_physical_parameters(self):
        base = get_base_args()
        original = copy.deepcopy(base.as_dict())
        default = build_inner_parameters(base)
        configured_base = copy.deepcopy(base)
        configured_base.mosek_num_threads = 2
        limited = build_inner_parameters(configured_base)
        self.assertEqual(default.mosek_num_threads, 0)
        self.assertEqual(limited.mosek_num_threads, 2)
        self.assertEqual(base.as_dict(), original)
        before, after = vars(default).copy(), vars(limited).copy()
        before.pop("mosek_num_threads")
        after.pop("mosek_num_threads")
        self.assertEqual(set(before), set(after))
        for key in before:
            with self.subTest(key=key):
                np.testing.assert_array_equal(before[key], after[key])

    def test_bridge_rejects_invalid_thread_types(self):
        for value in (-1, True, 1.5, "1", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                build_inner_parameters(SimpleNamespace(mosek_num_threads=value))

    def test_real_fusion_new_model_and_cached_model_reset_to_auto(self):
        """Inspect the actual MOSEK Task without solving or invoking the license."""
        ctx = self.context(1)
        model = None
        with mock.patch.dict(fusion._MODEL_CACHE, {}, clear=True):
            try:
                model, fp, fv = fusion.get_fusion_model(ctx)
                self.assertEqual(model.getTask().getintparam(mosek.iparam.num_threads), 1)
                shape = (model.getTask().getnumvar(), model.getTask().getnumcon())
                for threads in (2, 0):
                    ctx.params.mosek_num_threads = threads
                    reused, next_fp, next_fv = fusion.get_fusion_model(ctx)
                    self.assertIs(reused, model)
                    self.assertIs(next_fp, fp)
                    self.assertIs(next_fv, fv)
                    self.assertEqual(reused.getTask().getintparam(mosek.iparam.num_threads), threads)
                    self.assertEqual((reused.getTask().getnumvar(), reused.getTask().getnumcon()), shape)
                    self.assertEqual(reused.getTask().getdouparam(mosek.dparam.intpnt_co_tol_pfeas),
                                     ctx.mosek_tol_feas)
                del ctx.params.mosek_num_threads
                reused, _, _ = fusion.get_fusion_model(ctx)
                self.assertEqual(reused.getTask().getintparam(mosek.iparam.num_threads), 0)
            finally:
                if model is not None:
                    model.dispose()

    def test_non_cached_builder_uses_same_thread_setting(self):
        for value in (0, 1, 2):
            with self.subTest(threads=value):
                model, _, _ = fusion.build_fusion_model(self.context(value))
                try:
                    self.assertEqual(model.getTask().getintparam(mosek.iparam.num_threads), value)
                finally:
                    model.dispose()


if __name__ == "__main__":
    unittest.main()
