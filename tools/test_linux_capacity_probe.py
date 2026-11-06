"""Regression checks for measurement correctness; no project training is started."""
import csv
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import linux_capacity_probe as probe


def row(iteration, updates=0, **extra):
    return dict(iteration=str(iteration), env_steps=str(iteration * 320),
                sac_updates=str(updates), elapsed_seconds="2", reward="0.5",
                completion_rate="80", **extra)


class ProbeTests(unittest.TestCase):
    def args(self, *extra):
        return probe.parse_args(["--warmup", "2", "--measure", "2", *extra])[0]

    def test_elapsed_wall_time_starts_at_release_and_includes_gaps(self):
        window = probe.Window("Pure_MHBPPO", self.args())
        window.feed(row(1), 1)
        window.feed(row(2), 3)
        self.assertEqual(window.start[:2], (2, 640))
        # A slow peer may take a long time to warm up. This waiting is excluded.
        window.release(100)
        window.feed(row(3), 102)
        window.feed(row(4), 110)
        result = window.result()
        self.assertEqual(result["measured_seconds"], 10)
        self.assertEqual(result["env_steps_per_second"], 64)
        self.assertEqual(result["csv_iteration_seconds_p95"], 2)
        window.feed(row(5), 1000)
        self.assertEqual(window.result(), result)

    def test_sac_window_excludes_prefill_and_first_learning_rounds(self):
        window = probe.Window("MHSAC", self.args("--sac-learning-warmup", "3"))
        for iteration in range(1, 25):
            window.feed(row(iteration), iteration)
        self.assertIsNone(window.start)
        for iteration in range(25, 28):
            window.feed(row(iteration, updates=1 + (iteration - 25) * 320), iteration)
        self.assertEqual(window.start[0], 27)
        self.assertFalse(window.active)
        window.release(100)
        window.feed(row(28, updates=961), 110)
        window.feed(row(29, updates=1281), 120)
        self.assertEqual(window.result()["end_iteration"], 29)

    def test_csv_tail_handles_partial_row_and_quoted_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "updates.csv"
            path.write_text('iteration,env_steps,status\n1,320,"{""ok"": 4}"\n2,', encoding="utf-8")
            follower = probe.CSVFollower()
            self.assertEqual(follower.read(path), [{"iteration": "1", "env_steps": "320", "status": '{"ok": 4}'}])
            self.assertEqual(follower.read(path), [])
            with path.open("a", encoding="utf-8") as stream:
                stream.write('640,ok\n')
            self.assertEqual(follower.read(path)[0]["env_steps"], "640")
            self.assertEqual(follower.read(path), [])

    def test_slowdown_is_time_increase_not_throughput_loss(self):
        args = self.args()
        sample = dict(algorithm="MHBPPO", start_iteration=2, end_iteration=4,
                      env_steps_per_second=100, csv_iteration_seconds_p95=3)
        with tempfile.TemporaryDirectory() as directory:
            result = probe.write_report(Path(directory), {"MHBPPO": sample},
                                        [("two", [dict(sample, env_steps_per_second=80)])], args)
            self.assertAlmostEqual(result[0]["worst_slowdown_fraction"], .25)
            self.assertFalse(result[0]["passes_throughput_limit"])
            with (Path(directory) / "comparison.csv").open() as stream:
                output = next(csv.DictReader(stream))
            self.assertAlmostEqual(float(output["throughput_retention"]), .8)

    def test_invalid_groups_and_resource_override_rejected(self):
        with self.assertRaises(SystemExit), patch("sys.stderr"):
            probe.parse_args(["--algorithms", "MHBPPO", "--groups", "MHBPPO,MHSAC"])
        with self.assertRaises(ValueError):
            probe.build_command("MHBPPO", self.args(), {"MHBPPO": {"args": ["--max_updates=3"]}})

    def test_uses_infinite_run_preserving_configured_lr_horizon(self):
        command = probe.build_command("MHBPPO", self.args(), {"MHBPPO": {"args": ["--episodes", "3000"]}})
        self.assertEqual(command[command.index("--max_updates") + 1], "0")
        self.assertEqual(command[command.index("--episodes") + 1], "3000")

    def test_mhbppo_diagnostic_costs_are_not_used_as_wall_time(self):
        window = probe.Window("MHBPPO", self.args())
        window.feed(row(2), 1)
        window.release(10)
        for iteration, now in ((3, 20), (4, 30)):
            sample = row(iteration)
            sample.pop("elapsed_seconds")
            sample.update(collection_wait_seconds="1", update_seconds="2", logging_seconds="3")
            window.feed(sample, now)
        self.assertEqual(window.result()["csv_iteration_seconds_median"], 6)
        self.assertEqual(window.result()["env_steps_per_second"], 32)

    def test_report_rejects_different_training_phases(self):
        baseline = dict(algorithm="MHBPPO", start_iteration=2, end_iteration=4,
                        env_steps_per_second=100, csv_iteration_seconds_p95=3)
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(RuntimeError):
            probe.write_report(Path(directory), {"MHBPPO": baseline},
                               [("two", [dict(baseline, start_iteration=3)])], self.args())


if __name__ == "__main__":
    unittest.main()
