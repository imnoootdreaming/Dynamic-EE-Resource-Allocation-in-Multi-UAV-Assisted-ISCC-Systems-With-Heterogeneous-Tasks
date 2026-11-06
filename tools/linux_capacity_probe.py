#!/usr/bin/env python3
"""Measure real multi-algorithm training capacity on Linux; see CAPACITY_TEST.md."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import uuid


ALGORITHMS = ("MHBPPO", "MHGPPO", "SHBPPO", "Pure_MHBPPO", "MHSAC")
ROOT = Path(__file__).resolve().parents[1]
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")


def read_text(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return ""


def cgroup_paths():
    """Resolve unified hierarchy, including a cgroup namespace / mount root."""
    membership = next((line[3:] for line in read_text("/proc/self/cgroup").splitlines()
                       if line.startswith("0::")), None)
    if membership is None:
        return []
    for line in read_text("/proc/self/mountinfo").splitlines():
        before, _, after = line.partition(" - ")
        if after.split()[:1] != ["cgroup2"]:
            continue
        fields = before.split()
        mount_root, mount_at = fields[3:5]
        # Escaped whitespace in mountinfo is unusual but valid.
        unescape = lambda s: re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), s)
        mount_root, mount_at = map(unescape, (mount_root, mount_at))
        if ".." in Path(membership).parts:
            continue  # Cannot reliably resolve an out-of-namespace membership.
        if membership == mount_root:
            suffix = ""
        elif mount_root == "/":
            suffix = membership.lstrip("/")
        elif membership.startswith(mount_root.rstrip("/") + "/"):
            suffix = membership[len(mount_root):].lstrip("/")
        else:
            continue
        base, leaf = Path(mount_at), Path(mount_at) / suffix
        if not leaf.is_dir():
            continue
        return [leaf, *[p for p in leaf.parents if p == base or base in p.parents]]
    return []


def cg_limits(paths):
    result = []
    for path in paths:
        row = {"path": str(path)}
        for key in ("cpu.max", "cpu.stat", "memory.max", "memory.current",
                    "memory.events", "memory.swap.current", "cpuset.cpus.effective"):
            row[key] = read_text(path / key)
        result.append(row)
    return result


def inventory(psutil):
    affinity = sorted(os.sched_getaffinity(0))
    paths = cgroup_paths()
    limits = cg_limits(paths)
    quotas = []
    for row in limits:
        fields = row["cpu.max"].split()
        if len(fields) == 2 and fields[0] != "max":
            quotas.append(int(fields[0]) / int(fields[1]))
    return {"platform": sys.platform, "python": sys.executable,
            "host_logical_cpus": os.cpu_count(),
            "host_physical_cores": psutil.cpu_count(logical=False),
            "affinity_cpus": affinity,
            "effective_cpu_upper_bound": min([len(affinity), *quotas]),
            "memory_host": psutil.virtual_memory()._asdict(), "cgroup_v2": limits,
            "cgroup_v1_or_unresolved": not bool(paths),
            "cuda_visible_devices_inherited": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "lscpu": command_output(["lscpu"]),
            "nvidia_smi": command_output(["nvidia-smi"])}


def command_output(command):
    try:
        return subprocess.run(command, capture_output=True, text=True,
                              errors="replace", timeout=5).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return "unavailable"


def pressure():
    return {name: read_text(Path("/proc/pressure") / name)
            for name in ("cpu", "memory", "io")}


def percentile(values, fraction):
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lo, hi = math.floor(index), math.ceil(index)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)


class CSVFollower:
    """Consume only complete CSV records; training's scalar rows contain no LF."""
    def __init__(self):
        self.offset = 0
        self.pending = ""
        self.header = None

    def read(self, path):
        try:
            with Path(path).open(encoding="utf-8", newline="") as stream:
                stream.seek(self.offset)
                self.pending += stream.read()
                self.offset = stream.tell()
        except FileNotFoundError:
            return []
        lines = self.pending.split("\n")
        self.pending = lines.pop()
        rows = []
        for line in lines:
            if not line.strip():
                continue
            fields = next(csv.reader([line]))
            if self.header is None:
                self.header = fields
            elif len(fields) != len(self.header):
                raise RuntimeError(f"Malformed training CSV: {path}")
            else:
                rows.append(dict(zip(self.header, fields)))
        return rows


class Window:
    def __init__(self, algorithm, args):
        self.algorithm, self.args = algorithm, args
        self.learning_rows = 0
        self.start = None
        self.end = None
        self.rows = []
        self.active = False

    def release(self, now):
        self.start = (self.start[0], self.start[1], now)
        self.active = True

    def feed(self, row, now):
        if self.end is not None:
            return
        if self.start is not None and not self.active:
            return
        iteration = int(row.get("iteration", row.get("update", 0)))
        if self.algorithm == "MHSAC" and int(row.get("sac_updates", 0)) > 0:
            self.learning_rows += 1
        ready = iteration >= self.args.warmup
        if self.algorithm == "MHSAC":
            ready = ready and self.learning_rows >= self.args.sac_learning_warmup
        if self.start is None:
            if ready:
                self.start = (iteration, int(row["env_steps"]), now)
            return
        self.rows.append(row)
        if iteration >= self.start[0] + self.args.measure:
            self.end = (iteration, int(row["env_steps"]), now)

    def result(self):
        if self.end is None:
            raise RuntimeError(f"Incomplete measurement: {self.algorithm}")
        seconds = self.end[2] - self.start[2]
        # elapsed_seconds excludes some post-update reporting/checkpoint work.
        # Use observer wall time for throughput; CSV times only for diagnostics.
        costs = [float(r["elapsed_seconds"]) if "elapsed_seconds" in r else
                 sum(float(r[k]) for k in ("collection_wait_seconds", "update_seconds", "logging_seconds"))
                 for r in self.rows]
        return {"algorithm": self.algorithm, "start_iteration": self.start[0],
                "end_iteration": self.end[0], "measured_seconds": seconds,
                "measured_env_steps": self.end[1] - self.start[1],
                "env_steps_per_second": (self.end[1] - self.start[1]) / max(seconds, 1e-9),
                "csv_iteration_seconds_median": percentile(costs, .5),
                "csv_iteration_seconds_p95": percentile(costs, .95),
                "mean_completion_rate": sum(float(r["completion_rate"]) for r in self.rows) / len(self.rows),
                "mean_reward": sum(float(r["reward"]) for r in self.rows) / len(self.rows),
                "last_sac_updates": self.rows[-1].get("sac_updates"),
                "last_replay_size": self.rows[-1].get("replay_size")}


def build_command(name, args, settings):
    extra = settings.get(name, {}).get("args", [])
    if not isinstance(extra, list) or not all(isinstance(v, str) for v in extra):
        raise ValueError(f"{name}.args must be an array of strings")
    # Keep the runner's collection/termination contract; allow normal workload options.
    reserved = {"--num_samplers", "--worker_num_threads", "--mosek_num_threads",
                "--max_updates", "--sync_training", "--no-sync_training", "--log_file"}
    if any(v.split("=", 1)[0] in reserved for v in extra):
        raise ValueError(f"{name}: use probe resource flags instead of reserved training flags")
    defaults = ["--expected_duplicate_matching_coef", "0.3"] if name == "MHBPPO" else []
    return [sys.executable, "-u", str(ROOT / "src/outer/algorithms" / name / f"{name}_main.py"),
            *defaults, *extra, "--num_samplers", str(args.envs), "--sync_training",
            "--worker_num_threads", str(args.threads), "--mosek_num_threads", str(args.threads),
            "--max_updates", "0"]


class Job:
    def __init__(self, name, stage, args, settings):
        self.name, self.window = name, Window(name, args)
        self.directory = stage / name
        self.directory.mkdir()
        self.stdout_path = self.directory / "console.log"
        self.console = self.stdout_path.open("w", encoding="utf-8")
        self.tail = self.stdout_path.open(encoding="utf-8", errors="replace")
        self.output, self.csv = None, CSVFollower()
        self.lines = ""
        self.paused = False
        self.pending_rows = []
        env = os.environ.copy()
        env.update({str(k): str(v) for k, v in settings.get(name, {}).get("env", {}).items()})
        env.update({key: str(args.threads) for key in THREAD_VARS})
        env.update(PYTHONUTF8="1", PYTHONUNBUFFERED="1", MPLBACKEND="Agg",
                   ISCC_BACKGROUND_LOG_FILE=str(self.stdout_path))
        if args.cpu:
            env["CUDA_VISIBLE_DEVICES"] = ""
        self.command = build_command(name, args, settings)
        (self.directory / "command.json").write_text(json.dumps({
            "argv": self.command, "cwd": str(ROOT), "env_overrides": {
                k: env[k] for k in (*THREAD_VARS, "CUDA_VISIBLE_DEVICES", "ISCC_BACKGROUND_LOG_FILE") if k in env}},
            indent=2), encoding="utf-8")
        self.process = subprocess.Popen(self.command, cwd=ROOT, env=env,
                                        stdin=subprocess.DEVNULL, stdout=self.console,
                                        stderr=subprocess.STDOUT, start_new_session=True)

    def poll(self, now):
        if self.paused:
            return
        self.lines += self.tail.read()
        lines = self.lines.split("\n")
        self.lines = lines.pop()
        for line in lines:
            match = re.search(r"checkpoint=(.*?) \| logs=", line)
            if match:
                self.output = Path(match[1])
            match = re.search(r"\[learner\] 输出目录:\s*(.+)", line)
            if match:
                self.output = Path(match[1].strip())
        if self.output is not None:
            self.pending_rows.extend(self.csv.read(self.output / "training_updates.csv"))
            while self.pending_rows:
                row = self.pending_rows.pop(0)
                self.window.feed(row, now)
                if self.window.start is not None and not self.window.active:
                    # Hold the entire owned tree so fast algorithms cannot measure
                    # while a slow peer is still importing or filling its replay.
                    os.killpg(self.process.pid, signal.SIGSTOP)
                    self.paused = True
                    break
        if self.process.poll() is not None:
            raise RuntimeError(f"{self.name} exited before stage cleanup, code={self.process.returncode}; see {self.stdout_path}")


def stop_jobs(jobs, grace):
    """Signal learners first, retain workers for normal cleanup; only kill owned groups."""
    for job in jobs:
        if job.process.poll() is None:
            try:
                os.killpg(job.process.pid, signal.SIGCONT)
            except ProcessLookupError:
                pass
            job.process.send_signal(signal.SIGINT)
    deadline = time.monotonic() + grace
    next_status = time.monotonic() + 30
    while time.monotonic() < deadline and any(j.process.poll() is None for j in jobs):
        if time.monotonic() >= next_status:
            print("  Waiting for learner checkpoint/worker cleanup...", flush=True)
            next_status = time.monotonic() + 30
        time.sleep(.25)
    for job in jobs:
        # The group may retain a worker after a failed learner; always clean it up.
        try:
            os.killpg(job.process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    if jobs:
        time.sleep(1)
    for job in jobs:
        try:
            os.killpg(job.process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        job.process.wait()
        job.console.close()
        job.tail.close()


def resource_sample(psutil, jobs, directory, paths):
    memory = psutil.virtual_memory()
    groups = {j.process.pid for j in jobs}
    rss, threads, count = 0, 0, 0
    for proc in psutil.process_iter(["pid", "memory_info", "num_threads"]):
        try:
            if os.getpgid(proc.pid) in groups:
                count += 1
                rss += proc.info["memory_info"].rss
                threads += proc.info["num_threads"]
        except (ProcessLookupError, PermissionError, psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    limits = cg_limits(paths)
    headrooms = [100 * memory.available / memory.total]
    for row in limits:
        if row["memory.max"].isdigit() and row["memory.current"].isdigit():
            maximum, current = int(row["memory.max"]), int(row["memory.current"])
            if maximum:
                headrooms.append(100 * max(0, maximum - current) / maximum)
    return {"monotonic": time.monotonic(), "cpu_percent_host": psutil.cpu_percent(),
            "loadavg": os.getloadavg(), "memory_available_gib_host": memory.available / 2**30,
            "available_percent_min_host_cgroup": min(headrooms),
            "probe_processes": count, "probe_threads": threads,
            "probe_rss_sum_gib_shared_pages_double_counted": rss / 2**30,
            "swap": psutil.swap_memory()._asdict(), "pressure": pressure(),
            "cgroup_v2": limits, "disk_free_gib": shutil.disk_usage(directory).free / 2**30}


def run_stage(names, directory, args, settings, psutil):
    directory.mkdir()
    print(f"Starting {directory.name}: {', '.join(names)}", flush=True)
    jobs, samples, failure = [], [], None
    gpu_process, gpu_stream = None, None
    started = time.monotonic()
    paths = cgroup_paths()
    try:
        before = resource_sample(psutil, [], directory, paths)
        if before["available_percent_min_host_cgroup"] < args.min_available_percent or before["disk_free_gib"] < args.min_disk_gib:
            raise RuntimeError("Insufficient memory/disk headroom before launch")
        if shutil.which("nvidia-smi"):
            gpu_stream = (directory / "gpu.csv").open("w", encoding="utf-8")
            gpu_process = subprocess.Popen([
                "nvidia-smi", "--query-gpu=timestamp,index,uuid,utilization.gpu,utilization.memory,memory.used,memory.total",
                "--format=csv", "-l", "2"], stdout=gpu_stream, stderr=subprocess.STDOUT)
        for name in names:
            jobs.append(Job(name, directory, args, settings))
        next_sample, next_status = 0, 0
        with (directory / "resources.jsonl").open("w", encoding="utf-8") as stream:
            while True:
                now = time.monotonic()
                for job in jobs:
                    job.poll(now)
                if all(j.paused for j in jobs):
                    released = time.monotonic()
                    for job in jobs:
                        job.window.release(released)
                        os.killpg(job.process.pid, signal.SIGCONT)
                        job.paused = False
                    print("  All learners warmed up; concurrent measurement released", flush=True)
                if now >= next_sample:
                    sample = resource_sample(psutil, jobs, directory, paths)
                    samples.append(sample)
                    stream.write(json.dumps(sample) + "\n"); stream.flush()
                    if sample["available_percent_min_host_cgroup"] < args.min_available_percent:
                        raise RuntimeError("Memory headroom below configured guard; stopping sweep")
                    if sample["disk_free_gib"] < args.min_disk_gib:
                        raise RuntimeError("Disk headroom below configured guard; stopping sweep")
                    next_sample = now + args.sample_interval
                if now >= next_status:
                    status = [f"{j.name}:{'done' if j.window.end else ('measuring' if j.window.start else 'warming up')}"
                              for j in jobs]
                    print(f"  {now - started:.0f}s " + " | ".join(status), flush=True)
                    next_status = now + 30
                # Faster jobs keep training until every peer finishes its own window.
                if all(job.window.end is not None for job in jobs):
                    break
                if now - started > args.timeout:
                    raise RuntimeError(f"Stage timed out after {args.timeout}s")
                time.sleep(args.poll_interval)
        rows = [dict(job.window.result(), output_dir=str(job.output),
                     console_log=str(job.stdout_path)) for job in jobs]
        if any(row["measured_seconds"] < args.min_window_seconds for row in rows):
            raise RuntimeError("Measurement window too short; increase --measure (see stage_result.json)")
    except BaseException as exc:
        failure = str(exc) or type(exc).__name__
        raise
    finally:
        print(f"Stopping owned learners in {directory.name}; saving training artifacts...", flush=True)
        stop_jobs(jobs, args.stop_grace)
        if gpu_process is not None:
            if gpu_process.poll() is None:
                gpu_process.terminate()
                try:
                    gpu_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    gpu_process.kill(); gpu_process.wait()
            gpu_stream.close()
        result = {"algorithms": names, "error": failure,
                  "jobs": [dict(j.window.result(), output_dir=str(j.output))
                           for j in jobs if j.window.end is not None],
                  "exit_codes": {j.name: j.process.returncode for j in jobs},
                  "min_available_percent": min((s["available_percent_min_host_cgroup"] for s in samples), default=None),
                  "peak_rss_sum_gib": max((s["probe_rss_sum_gib_shared_pages_double_counted"] for s in samples), default=None)}
        (directory / "stage_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if any(job.process.returncode != 0 for job in jobs):
        raise RuntimeError("Training did not shut down cleanly; inspect stage_result.json and console logs")
    return rows


def write_report(directory, baselines, groups, args):
    comparisons = []
    for label, rows in groups:
        for row in rows:
            solo = baselines[row["algorithm"]]
            if row["start_iteration"] != solo["start_iteration"] or row["end_iteration"] != solo["end_iteration"]:
                raise RuntimeError("Solo/group iteration windows differ; inspect stage results")
            retention = row["env_steps_per_second"] / solo["env_steps_per_second"]
            slowdown = 1 / retention - 1
            comparisons.append(dict(stage=label, concurrency=len(rows), **row,
                                    solo_env_steps_per_second=solo["env_steps_per_second"],
                                    throughput_retention=retention, slowdown_fraction=slowdown,
                                    p95_cost_ratio=row["csv_iteration_seconds_p95"] / solo["csv_iteration_seconds_p95"],
                                    within_slowdown_limit=slowdown <= args.max_slowdown))
    if comparisons:
        with (directory / "comparison.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(comparisons[0]))
            writer.writeheader(); writer.writerows(comparisons)
    summary = []
    for label, rows in groups:
        matched = [r for r in comparisons if r["stage"] == label]
        summary.append({"stage": label, "algorithms": [r["algorithm"] for r in rows],
                        "normalized_throughput_sum": sum(r["throughput_retention"] for r in matched),
                        "worst_slowdown_fraction": max(r["slowdown_fraction"] for r in matched),
                        "passes_throughput_limit": all(r["within_slowdown_limit"] for r in matched)})
    (directory / "summary.json").write_text(json.dumps({"max_slowdown": args.max_slowdown,
        "tested_groups": summary, "baselines": baselines}, indent=2), encoding="utf-8")
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithms", nargs="+", choices=ALGORITHMS, default=list(ALGORITHMS))
    parser.add_argument("--groups", nargs="+", help="Exact groups, e.g. MHBPPO,MHSAC MHBPPO,MHGPPO,MHSAC; default: prefixes of --algorithms")
    parser.add_argument("--settings", type=Path, help="JSON: algorithm -> {args: [...], env: {...}}")
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5, help="Completed rounds excluded from measurement")
    parser.add_argument("--sac-learning-warmup", type=int, default=5, help="Exclude at least this many MHSAC rounds with actual gradient updates")
    parser.add_argument("--measure", type=int, default=20, help="Measured rounds per algorithm")
    parser.add_argument("--timeout", type=float, default=3600, help="Seconds per stage, including startup/warmup")
    parser.add_argument("--stop-grace", type=float, default=180, help="Seconds allowed for normal checkpoint/worker cleanup")
    parser.add_argument("--poll-interval", type=float, default=.25)
    parser.add_argument("--sample-interval", type=float, default=2)
    parser.add_argument("--min-window-seconds", type=float, default=30)
    parser.add_argument("--max-slowdown", type=float, default=.20, help="Time per step increase allowed, e.g. .20")
    parser.add_argument("--min-available-percent", type=float, default=10)
    parser.add_argument("--min-disk-gib", type=float, default=5)
    parser.add_argument("--cpu", action="store_true", help="Hide CUDA from every learner/worker")
    parser.add_argument("--output", type=Path, help="Fresh result directory; training retains normal ckpt/runs output locations")
    parser.add_argument("--dry-run", action="store_true", help="Print planned commands; launch nothing")
    parser.add_argument("--inventory-only", action="store_true", help="Save hardware/cgroup information without launching training")
    args = parser.parse_args(argv)
    if any(getattr(args, key) <= 0 for key in ("envs", "threads", "warmup", "measure", "sac_learning_warmup", "timeout", "stop_grace", "poll_interval", "sample_interval", "min_window_seconds")):
        parser.error("Counts, intervals and timeout values must be positive")
    if args.max_slowdown < 0 or not 0 <= args.min_available_percent < 100 or args.min_disk_gib < 0:
        parser.error("Invalid slowdown/headroom limits")
    if len(set(args.algorithms)) != len(args.algorithms):
        parser.error("--algorithms must be unique")
    groups = [item.split(",") for item in args.groups] if args.groups else [args.algorithms[:n] for n in range(2, len(args.algorithms) + 1)]
    if any(not group or len(set(group)) != len(group) or set(group) - set(args.algorithms) for group in groups):
        parser.error("Each group must contain distinct names selected in --algorithms")
    return args, groups


def main(argv=None):
    args, groups = parse_args(argv)
    settings = json.loads(args.settings.read_text(encoding="utf-8-sig")) if args.settings else {}
    if not isinstance(settings, dict) or set(settings) - set(ALGORITHMS):
        raise ValueError("settings must map known algorithm names to objects")
    for name in args.algorithms:
        item = settings.get(name, {})
        if not isinstance(item, dict) or set(item) - {"args", "env"} or not isinstance(item.get("env", {}), dict):
            raise ValueError(f"{name}: expected {{args: [...], env: {{...}}}}")
        if "ISCC_BACKGROUND_LOG_FILE" in item.get("env", {}):
            raise ValueError("ISCC_BACKGROUND_LOG_FILE is managed by the probe")
        print(shlex.join(build_command(name, args, settings)))
    if args.dry_run:
        print("Solo baselines:", args.algorithms, "Concurrent groups:", groups)
        return 0
    if not sys.platform.startswith("linux"):
        raise SystemExit("Real capacity tests require Linux. --dry-run works on other platforms.")
    try:
        import psutil
    except ImportError:
        raise SystemExit("Install the monitor dependency in your training environment: python -m pip install psutil")
    destination = (args.output or ROOT / "src/outer/logs/capacity" /
                   (time.strftime("%Y-%m-%d_%H-%M-%S") + "_" + uuid.uuid4().hex[:8])).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "inventory.json").write_text(json.dumps(inventory(psutil), indent=2), encoding="utf-8")
    (destination / "probe_config.json").write_text(json.dumps({
        "args": vars(args), "settings": settings, "groups": groups}, default=str, indent=2), encoding="utf-8")
    print("Results:", destination, flush=True)
    if args.inventory_only:
        print("Inventory saved; no training launched.", flush=True)
        return 0
    baselines, completed_groups = {}, []
    try:
        for name in args.algorithms:
            rows = run_stage([name], destination / f"solo_{name}", args, settings, psutil)
            baselines[name] = rows[0]
            write_report(destination, baselines, completed_groups, args)
        for number, group in enumerate(groups, 1):
            label = f"group_{number}_{len(group)}jobs"
            rows = run_stage(group, destination / label, args, settings, psutil)
            completed_groups.append((label, rows))
            summary = write_report(destination, baselines, completed_groups, args)
            print(json.dumps(summary[-1], ensure_ascii=False), flush=True)
    except (KeyboardInterrupt, RuntimeError) as exc:
        print(f"Probe stopped: {exc}. Partial results: {destination}", file=sys.stderr)
        return 1
    print(f"Complete. Read {destination / 'comparison.csv'} and summary.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
