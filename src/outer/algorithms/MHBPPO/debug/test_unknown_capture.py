"""Run with the training Python: python src/outer/algorithms/MHBPPO/debug/test_unknown_capture.py."""

import copy
import hashlib
import io
import json
import logging
import multiprocessing as mp
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(ROOT / "src/inner"), str(ROOT / "src/outer")]

from algorithms.MHBPPO.unknown_logging import UnknownLogWriter, unknown_start_iteration
from algorithms.MHBPPO.log_compaction import compact_episode_payload
from p5_failure_capture import capture_unknown_slot


def _spawn_episode(result_queue):
    """Real spawn/actor/environment/solver transport, without a PPO update."""
    sys.path.insert(0, str(ROOT / "src/outer"))
    import torch
    from configs.base_params import get_base_args
    from configs.madrl_args import get_madrl_args
    from environment.my_env import MyEnv
    from algorithms.MHBPPO.MHBPPO_agent import MultiHeadActor
    from algorithms.MHBPPO.MHBPPO_sampler import sampler_worker
    args = get_madrl_args(["--max_updates", "1", "--hidden_dim", "16",
                          "--worker_num_threads", "1", "--mosek_num_threads", "1",
                          "--no-env_diagnostics"])
    base = get_base_args(seed=42, target_layout=args.target_layout)
    env = MyEnv(base_args=base, madrl_args=args, seed_offset=0)
    torch.manual_seed(42)
    actor = MultiHeadActor(
        state_dim=env.observation_space["bs"].shape[0], hidden_dim=16,
        continuous_action_splits=env.bs_continuous_action_splits,
        discrete_action_dims=env.action_space["bs"]["discrete"].nvec,
        continuous_dist_type="beta", structured_beam_alpha=0.0)
    weights = mp.get_context("spawn").Queue()
    weights.put({"actor_state_dict": actor.state_dict(), "version": 0})
    stop = mp.get_context("spawn").Event()

    class OneEpisodeQueue:
        def put(self, item, timeout=None):
            result_queue.put(item)
            stop.set()

    sampler_worker(env.base_args, args, OneEpisodeQueue(), weights, stop,
                   seed_offset=0, capture_unknown=True)


def episode(payload=None, diagnostic=True, resolved="unknown"):
    solver = {"raw_solver_status_kind": "unknown", "solver_status_kind": resolved,
              "optimizer_response_code": 100006}
    return {
        "episode_idx": 2970, "worker_id": 2, "weights_version": 2970,
        "action_traces": [{"time_slot": 3, "physical_action": {}, "outcome": solver}],
        "diagnostic_traces": [{"solver": solver}] if diagnostic else [],
        "unknown_capture_payloads": {0: payload} if payload is not None else {},
    }


def payload():
    return {"files": {"failed_p5.task.gz": b"task", "linearization.npz": b"arrays",
                      "mosek.log": b"complete solver log"},
            "manifest": {"complete": True, "errors": {}, "failed_iteration": 4}}


class WriterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "mhbppo_train_unknown.jsonl"
        self.writer = UnknownLogWriter(str(self.path), 3000, {"run_id": "test"},
                                       logging.getLogger("unknown_capture_test"))
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(self.writer.close)

    def records(self):
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines()]

    def test_window_and_rounding(self):
        self.assertEqual(unknown_start_iteration(3000), 2971)
        self.assertEqual(unknown_start_iteration(3001), 2971)
        self.assertEqual(unknown_start_iteration(100), 100)
        self.assertEqual(unknown_start_iteration(1), 1)
        self.assertIsNone(unknown_start_iteration(0))
        self.writer.write_episode(episode(payload()), 2970)
        self.writer.write_episode(episode(payload()), 3001)
        self.assertFalse(self.path.exists())
        self.assertFalse(Path(self.writer.artifact_directory).exists())
        for iteration in (2971, 3000):
            self.writer.write_episode(episode(payload()), iteration)
        records = self.records()
        self.assertEqual(records[0]["window_updates"], 30)
        self.assertEqual([r["iteration"] for r in records[1:]], [2971, 3000])

    def test_complete_artifacts_no_mutation_and_compaction(self):
        item = episode(payload(), diagnostic=False)
        original = copy.deepcopy(item)
        # Compact logging preserves the separate binary transport, never JSON-encodes it.
        compact = compact_episode_payload(item)
        self.writer.write_episode(compact, 2971)
        self.assertEqual(item, original)
        record = self.records()[1]
        ref = record["unknown_capture"]
        self.assertEqual(ref["status"], "complete")
        manifest_path = self.path.parent / ref["manifest"]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["iteration"], 2971)
        for name, info in manifest["files"].items():
            data = (manifest_path.parent / name).read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), info["sha256"])
            self.assertEqual(len(data), info["bytes"])

    def test_raw_unknown_resolved_infeasible_is_kept(self):
        self.writer.write_episode(episode(payload(), resolved="infeasible"), 2971)
        self.assertEqual(self.records()[1]["solver"]["solver_status_kind"], "infeasible")

    def test_optimal_and_infinite_training_create_nothing(self):
        item = episode(payload())
        item["diagnostic_traces"][0]["solver"]["raw_solver_status_kind"] = "optimal"
        self.writer.write_episode(item, 2971)
        self.writer.max_updates = 0
        self.writer.start_iteration = None
        self.writer.write_episode(episode(payload()), 2971)
        self.assertFalse(self.path.exists())

    def test_existing_jsonl_not_overwritten(self):
        self.path.write_text("existing", encoding="utf-8")
        self.writer.write_episode(episode(payload()), 2971)
        self.assertEqual(self.path.read_text(), "existing")
        self.assertFalse(Path(self.writer.artifact_directory).exists())

    def test_event_cap_prevents_orphan_artifacts(self):
        self.writer.MAX_EVENTS = 1
        self.writer.write_episode(episode(payload()), 2971)
        self.writer.write_episode(episode(payload()), 2972)
        self.assertEqual(len(list(Path(self.writer.artifact_directory).iterdir())), 1)
        self.assertEqual(self.records()[-1]["record_type"], "limit_reached")

    def test_missing_and_failed_capture_are_explicit(self):
        self.writer.write_episode(episode(), 2971)
        self.assertEqual(self.records()[1]["unknown_capture"]["status"], "unavailable")
        with patch("algorithms.MHBPPO.unknown_logging.os.mkdir", side_effect=OSError("disk full")):
            self.writer.write_episode(episode(payload()), 2972)
        self.assertEqual(self.records()[2]["unknown_capture"]["status"], "error")

    def test_oversize_summary_retains_failure_reference(self):
        item = episode(payload())
        item["action_traces"][0]["physical_action"] = {"uav_angles": [0] * 10000}
        self.writer.write_episode(item, 2971)
        record = self.records()[1]
        self.assertEqual(record["details_omitted"], "record_size_limit")
        self.assertEqual(record["unknown_capture"]["status"], "complete")


class RealMosekTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import fusion_pc3p
        from environment import build_environment
        from outer_sampler import load_outer_samples
        from parameters import Parameters
        from variables import InnerContext, InnerVariables
        cls.backend = fusion_pc3p
        cls.build_environment = staticmethod(build_environment)
        cls.Parameters = Parameters
        cls.InnerContext = InnerContext
        cls.InnerVariables = InnerVariables
        cls.samples = load_outer_samples(str(ROOT / "src/inner/feasible_outer_samples.csv"), Parameters())

    def context(self, sample):
        params = self.Parameters()
        params.mosek_num_threads = 1
        params.max_iterations = 2
        env = self.build_environment(params, np.random.default_rng(params.seed), self.samples[sample])
        return self.InnerContext(params, env, self.InnerVariables.create(params))

    def solve(self, sample, enabled):
        ctx = self.context(sample)
        with patch.object(self.backend, "REUSE_MODEL", False), capture_unknown_slot(enabled) as state:
            result = self.backend.run_pc3p_fusion(ctx)
        return ctx, result, state

    def test_normal_numerical_result_is_unchanged(self):
        _, before, _ = self.solve(2, False)
        _, after, state = self.solve(2, True)
        self.assertEqual(before["status"], "ProblemStatus.PrimalAndDualFeasible")
        for key in ("status", "iterations", "last_solved_iteration", "objective_p4", "rank1_gap", "converged"):
            self.assertEqual(before[key], after[key], key)
        for key in ("W_sen_beam", "B_off_beam", "f_uav_freq", "z_aux_rate", "D_cu_off"):
            np.testing.assert_array_equal(before[key], after[key], err_msg=key)
        self.assertIsNone(state["payload"])

    def test_real_unknown_task_snapshot_log_and_reload(self):
        import mosek
        _, before, _ = self.solve(0, False)
        ctx, after, state = self.solve(0, True)
        self.assertIn(after["status"], ("ProblemStatus.Unknown", "ProblemStatus.IllPosed"))
        self.assertEqual(before["status"], after["status"])
        self.assertIsNone(after["W_sen_beam"])
        captured = state["payload"]
        self.assertTrue(captured["manifest"]["complete"], captured["manifest"])
        self.assertIn(b"PC3P iteration 1", captured["files"]["mosek.log"])
        self.assertIn(b"Optimizer", captured["files"]["mosek.log"])
        with np.load(io.BytesIO(captured["files"]["linearization.npz"]), allow_pickle=False) as arrays:
            for key in arrays.files:
                self.assertNotEqual(arrays[key].dtype.kind, "O", key)
            np.testing.assert_array_equal(arrays["variables.W_sen_beam_prev"], ctx.W_sen_beam_prev)
            np.testing.assert_array_equal(arrays["environment.G_sen_corr"], ctx.G_sen_corr)
            self.assertIn("fusion.M_sen_re.0", arrays)
            self.assertIn("fusion.z_ref", arrays)
            self.assertEqual(arrays["fusion.M_sen_re.0"].shape, (ctx.N, ctx.N))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "failed.task.gz"
            path.write_bytes(captured["files"]["failed_p5.task.gz"])
            with mosek.Task() as task:
                task.readdata(str(path))
                self.assertGreater(task.getnumvar(), 0)
                self.assertGreater(task.getnumcon(), 0)
                task.optimize()  # The saved complete P5 can be solved independently of Fusion/ctx.

    def test_capture_detaches_from_reused_model(self):
        with capture_unknown_slot(True) as state:
            self.backend.run_pc3p_fusion(self.context(0))
        self.assertTrue(state["payload"]["manifest"]["complete"])
        # This uses the same cached Model after the previous StringIO was closed.
        result = self.backend.run_pc3p_fusion(self.context(2))
        self.assertEqual(result["status"], "ProblemStatus.PrimalAndDualFeasible")

    def test_task_capture_error_does_not_change_solver_result(self):
        build = self.backend.build_fusion_model

        def broken_capture(ctx):
            model, fp, fv = build(ctx)
            model.writeTaskStream = lambda *args: (_ for _ in ()).throw(OSError("capture failure"))
            return model, fp, fv

        with patch.object(self.backend, "build_fusion_model", side_effect=broken_capture):
            _, result, state = self.solve(0, True)
        self.assertEqual(result["status"], "ProblemStatus.Unknown")
        self.assertIsNone(result["W_sen_beam"])
        self.assertFalse(state["payload"]["manifest"]["complete"])
        self.assertIn("task", state["payload"]["manifest"]["errors"])

    def test_later_unknown_keeps_the_failed_linearization(self):
        backend = self.backend
        build = backend.build_fusion_model

        class UnknownSecondSolve:
            def __init__(self, model):
                self.model, self.calls = model, 0

            def __getattr__(self, name):
                return getattr(self.model, name)

            def solve(self):
                self.calls += 1
                return self.model.solve()

            def getProblemStatus(self):
                return backend.mf.ProblemStatus.Unknown if self.calls == 2 else self.model.getProblemStatus()

        def wrapped(ctx):
            model, fp, fv = build(ctx)
            return UnknownSecondSolve(model), fp, fv

        with patch.object(backend, "build_fusion_model", side_effect=wrapped):
            ctx, result, state = self.solve(2, True)
        self.assertEqual(result["iterations"], 2)
        self.assertEqual(result["last_solved_iteration"], 1)
        self.assertIsNone(result["W_sen_beam"])
        captured = state["payload"]
        self.assertEqual(captured["manifest"]["linearization_iteration"], 1)
        with np.load(io.BytesIO(captured["files"]["linearization.npz"]), allow_pickle=False) as data:
            np.testing.assert_array_equal(data["variables.W_sen_beam_prev"], ctx.W_sen_beam_prev)
        self.assertIn(b"PC3P iteration 2", captured["files"]["mosek.log"])

    def test_spawn_sampler_to_learner_with_env_diagnostics_off(self):
        context = mp.get_context("spawn")
        queue = context.Queue(maxsize=1)
        proc = context.Process(target=_spawn_episode, args=(queue,))
        proc.start()
        try:
            item = queue.get(timeout=50)
            self.assertNotIn("error", item, item.get("error"))
            proc.join(timeout=15)
            self.assertEqual(proc.exitcode, 0)
            self.assertFalse(item["diagnostics_enabled"])
            self.assertEqual(len(item["transition_dict"]["rewards"]), 40)
            captures = item["unknown_capture_payloads"]
            self.assertGreater(len(captures), 0)
            for captured in captures.values():
                self.assertTrue(captured["manifest"]["complete"], captured["manifest"])
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "unknown.jsonl"
                writer = UnknownLogWriter(str(path), 1, {}, logging.getLogger("spawn_capture_test"))
                try:
                    writer.write_episode(item, 1)
                finally:
                    writer.close()
                records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()][1:]
                self.assertGreater(len(records), 0)
                self.assertTrue(all(record["unknown_capture"]["status"] == "complete" for record in records))
        finally:
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5)
            queue.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
