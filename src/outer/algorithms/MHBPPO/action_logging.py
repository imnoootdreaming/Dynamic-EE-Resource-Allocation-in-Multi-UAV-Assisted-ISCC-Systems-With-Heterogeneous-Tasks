"""MHBPPO run paths and per-transition JSONL action logging."""

import json
import atexit
import hashlib
import importlib.metadata
import platform
import sys
from time import perf_counter
import math
import os
from datetime import datetime
from algorithms.MHBPPO.log_compaction import (
    LOG_COMPACTION_CONTRACT, compact_action_trace, compact_env_trace,
)
from algorithms.MHBPPO.energy_statistics import ENERGY_STATISTICS_CONTRACT, ENERGY_VALUE_KEYS


def build_run_log_paths(outer_root, log_file, started_at):
    """Use the reserved background directory or create a foreground run directory."""
    background_log = os.environ.get("ISCC_BACKGROUND_LOG_FILE")
    if background_log:
        path = os.path.abspath(background_log)
        stem = os.path.splitext(path)[0]
        return path, stem + "_actions.jsonl"
    configured_path = os.path.join(outer_root, log_file)
    directory, filename = os.path.split(configured_path)
    os.makedirs(directory, exist_ok=True)
    timestamp = started_at.strftime("%Y-%m-%d_%H-%M-%S")
    stem, extension = os.path.splitext(filename)
    extension = extension or ".log"
    index = 0
    while True:
        suffix = "" if index == 0 else f"_{index:02d}"
        run_directory = os.path.join(directory, timestamp + suffix)
        try:
            # Atomic reservation also isolates concurrent starts in the same second.
            os.mkdir(run_directory)
        except FileExistsError:
            index += 1
            continue
        return (os.path.join(run_directory, stem + extension),
                os.path.join(run_directory, stem + "_actions.jsonl"))


def segment_path(path, index):
    """Archive name of a finished segment; the configured path always holds the newest one."""
    stem, extension = os.path.splitext(path)
    return f"{stem}_part{index:03d}{extension or '.jsonl'}"


class JsonlRotation:
    """Bound detailed JSONL logs by iteration window and/or segment size.

    Both detailed writers are fed in lockstep per update, so a single shared policy
    rotates them to the same segment index and keeps their records aligned by update.
    Whichever trigger fires first wins. Retained history therefore never spans more
    than (keep_segments + 1) windows, i.e. the oldest retained record lags the current
    update by at most (keep_segments + 1) * iterations updates when that trigger
    is enabled. Size is checked before an update, so max_bytes may be exceeded
    by one update's records. Set both triggers to 0 for unbounded behaviour.
    """

    def __init__(self, iterations=0, max_bytes=0, keep_segments=1):
        self.iterations = max(0, int(iterations))
        self.max_bytes = max(0, int(max_bytes))
        self.keep_segments = max(0, int(keep_segments))
        self.segment = 0
        self._next_iteration = None

    @property
    def active(self):
        return bool(self.iterations or self.max_bytes)

    def due(self, iteration, sizes):
        """True once per break point; latches the next iteration window."""
        if not self.active:
            return False
        if self._next_iteration is None:
            self._next_iteration = iteration + self.iterations
        by_iteration = self.iterations > 0 and iteration >= self._next_iteration
        by_size = self.max_bytes > 0 and any(size >= self.max_bytes for size in sizes)
        if not (by_iteration or by_size):
            return False
        self.segment += 1
        self._next_iteration = iteration + self.iterations
        return True

    def force(self, iteration):
        """Out-of-band rotation used by the disk guard; keeps the same window bookkeeping."""
        if not self.active:
            return 0
        self.segment += 1
        self._next_iteration = iteration + self.iterations
        return self.segment


def _json_value(value):
    """Preserve full arrays; encode non-finite diagnostics as strings, not invalid JSON."""
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, complex):
        return {"real": _json_value(float(value.real)), "imag": _json_value(float(value.imag))}
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def _episode_identity(item):
    """Additional explicit fields; preserve legacy sampler/seed_offset columns."""
    return {
        "worker_id": int(item.get("worker_id", item["sampler_id"])),
        "collection_round": item.get("collection_round"),
        "policy_version": int(item["weights_version"]),
        "scene_seed_offset": item.get("scene_seed_offset", item["seed_offset"]),
        "action_seed": item.get("action_seed"),
        "actor_signature": item.get("actor_signature"),
        "norm_signature": item.get("norm_signature"),
        "scene_signature": item.get("scene_signature"),
    }


class ActionLogWriter:
    """Learner-only writer: one metadata line, then one line per consumed transition."""

    def __init__(self, path, metadata):
        self._path = path
        self._metadata = dict(metadata)
        self.detail_level = metadata.get("log_detail_level", metadata.get("madrl_args", {}).get("log_detail_level", "full"))
        if self.detail_level not in ("compact", "full"):
            raise ValueError("unknown logging detail level")
        self.schema_version = 2 if self.detail_level == "compact" else 1
        if self.detail_level == "compact":
            self._metadata.update(log_detail_level="compact", log_compaction_contract=LOG_COMPACTION_CONTRACT)
        self._file = open(path, "x", encoding="utf-8")
        self.run_id = metadata.get("run_id")
        try:
            self._write({**self._metadata, "record_type": "metadata", "schema_version": self.schema_version})
            self._file.flush()
            atexit.register(self.close)
        except BaseException:
            self._file.close()
            raise

    @property
    def path(self):
        return self._path

    @property
    def size(self):
        """Size of the current segment in bytes; the file is flushed after each episode."""
        return self._file.tell()

    def rotate(self, segment_index):
        """Archive the finished segment and reopen the configured path with fresh metadata.

        Segment boundaries are only taken between episodes, so a 40-slot episode is
        never split across two files.
        """
        self._file.close()
        archived = segment_path(self._path, segment_index)
        os.replace(self._path, archived)
        self._file = open(self._path, "x", encoding="utf-8")
        self._write({**self._metadata, "record_type": "metadata", "schema_version": self.schema_version})
        self._file.flush()
        return archived

    def prune(self, earliest_segment):
        """Delete archived segments older than the window; the current file is never removed."""
        removed = []
        for index in range(1, earliest_segment):
            path = segment_path(self._path, index)
            if os.path.exists(path):
                os.remove(path)
                removed.append(path)
        return removed

    def _write(self, record):
        self._file.write(json.dumps(
            _json_value(record), ensure_ascii=False, allow_nan=False, separators=(",", ":"),
        ) + "\n")

    def write_episode(self, item, iteration):
        transitions = item["transition_dict"]
        traces = item["action_traces"]
        count = len(transitions["continuous_actions"])
        if any(len(values) != count for values in (
                transitions["discrete_actions"], transitions["rewards"], traces)):
            raise RuntimeError("动作日志与 transition 数量不一致")
        logged_at = datetime.now().astimezone().isoformat(timespec="milliseconds")
        if self.detail_level == "full" and item.get("log_detail_level") == "compact":
            raise ValueError("full writer cannot recover a compact worker payload")
        if self.detail_level == "compact":
            self._write({"record_type": "episode_summary", "run_id": self.run_id,
                         "iteration": int(iteration), "sampler_id": int(item["sampler_id"]),
                         "episode_idx": int(item["episode_idx"]), **_episode_identity(item),
                         "steps": count, "policy_distribution_summary": item.get("policy_distribution_summary"),
                         "energy_statistics": item.get("energy_statistics"),
                         "completion_rate": item.get("completion_rate"),
                         "all_slots_success": item.get("all_slots_success"),
                         "collection_seconds": item.get("collection_seconds")})
        for step, trace in enumerate(traces):
            self._write({
                "record_type": "action",
                "run_id": self.run_id,
                "logged_at": logged_at,
                "iteration": int(iteration),  # intended update, 1-based; logged before update
                "sampler_id": int(item["sampler_id"]),
                "seed_offset": int(item["seed_offset"]),
                "episode_idx": int(item["episode_idx"]),
                "step_idx": step,  # 0-based, as are env time_slot and episode_idx
                "weights_version": int(item["weights_version"]),
                **_episode_identity(item),
                **({"raw_action": {
                    "continuous": transitions["continuous_actions"][step],
                    "discrete": transitions["discrete_actions"][step],
                }} if self.detail_level == "full" else {}),
                "environment_action": compact_action_trace(trace) if self.detail_level == "compact" else trace,
                "raw_bs_reward": transitions["rewards"][step],
            })
        self._file.flush()  # one flush per episode, before learner mutates rewards / updates

    def close(self):
        if not self._file.closed:
            self._file.close()


def diagnostic_log_path(action_path):
    suffix = "_actions.jsonl"
    if not action_path.endswith(suffix):
        raise ValueError("expected _actions.jsonl path")
    return action_path[:-len(suffix)] + "_env.jsonl"


def diagnostic_run_metadata(outer_root, base_args, madrl_args):
    """Actual supplied configuration and relevant source fingerprints, no git dependency."""
    versions = {"python": sys.version, "python_executable": sys.executable,
                "platform": platform.platform()}
    for name in ("numpy", "torch", "cvxpy", "Mosek"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    files = (
        "algorithms/MHBPPO/MHBPPO_main.py", "algorithms/MHBPPO/MHBPPO_sampler.py",
        "algorithms/MHBPPO/MHBPPO_agent.py", "algorithms/MHBPPO/action_logging.py",
        "algorithms/MHBPPO/log_compaction.py",
        "algorithms/MHBPPO/unknown_logging.py", "../inner/p5_failure_capture.py",
        "configs/base_params.py", "configs/madrl_args.py", "configs/seed_manager.py",
        "environment/my_env.py", "environment/channel_models.py", "environment/topology.py",
        "environment/observation.py", "environment/mapping.py", "environment/cu_tasks.py",
        "reward/my_reward.py", "reward/cccp_bridge.py", "reward/feasibility_diagnostics.py",
        "reward/solver_status.py", "reward/sensing_penalty.py",
        "reward/episode_success_bonus.py",
        "algorithms/MHBPPO/checkpoint.py",
        "algorithms/MHBPPO/policy_diagnostics.py",
        "algorithms/MHBPPO/rollout_batch.py",
        "algorithms/MHBPPO/energy_statistics.py",
        "algorithms/MHBPPO/policy_initialization.py",
        "algorithms/MHBPPO/beam_branch.py", "algorithms/MHBPPO/beam_pretraining.py",
        "../inner/parameters.py", "../inner/environment.py", "../inner/target_layout.py", "../inner/pc3p.py",
        "../inner/fusion_pc3p.py", "../inner/variables.py", "../inner/cccp_params.py",
    )
    fingerprints = {}
    for name in files:
        with open(os.path.join(outer_root, name), "rb") as source:
            fingerprints[name] = hashlib.sha256(source.read()).hexdigest()
    params = getattr(base_args, "_params", None)
    return {
        "runtime": versions, "source_sha256": fingerprints,
        "solver_status_contract": "strict_sensing_presolve_and_unknown_reclassification_v2",
        "energy_statistics_contract": ENERGY_STATISTICS_CONTRACT,
        "energy_value_keys": ENERGY_VALUE_KEYS,
        "energy_statistics_semantics": {
            "finite_candidate": "finite_returned_objective_independent_of_success",
            "strict_feasible": "finite_candidate_and_physical_solution_verified_and_outer_action_passed",
            "missing_verification": "unknown_not_pass; separately_counted",
            "missing_energy": "excluded_per_field; mean_nan_and_count_zero_if_empty",
            "aggregation": "per_field_sums_and_counts_across_slots_and_workers",
            "legacy_obj": "mean_of_all_returned_objectives_including_failure_inf; not_episode_total",
        },
        "base_args": base_args.as_dict() if hasattr(base_args, "as_dict") else vars(base_args),
        "inner_parameter_source": None if params is None else vars(params).copy(),
        "madrl_args": vars(madrl_args).copy(),
        "diagnostic_semantics": "read_only_original_constraints; diagnostics_do_not_change_reward_or_success",
        "channel_time": "before_move", "collision_time": "after_move",
        "candidate_matching_scope": "sensing_only_fixed_position_and_receive_beam",
    }


class EnvDiagnosticLogWriter(ActionLogWriter):
    """Learner-only JSONL: metadata, deduplicated scenes, aligned per-slot records."""

    def __init__(self, path, metadata):
        self._scenes = set()
        self.last_write_seconds = 0.0
        self.last_write_bytes = 0
        super().__init__(path, metadata)

    def rotate(self, segment_index):
        archived = super().rotate(segment_index)
        # Each archived segment restates the scenes it references, so the set restarts.
        self._scenes.clear()
        return archived

    def write_episode(self, item, iteration):
        traces = item.get("diagnostic_traces", [])
        if self.detail_level == "full" and item.get("log_detail_level") == "compact":
            raise ValueError("full writer cannot recover a compact worker payload")
        count = len(item["transition_dict"]["rewards"])
        if len(traces) != count or len(item["action_traces"]) != count:
            raise RuntimeError("环境诊断与 transition 数量不一致")
        scene = item.get("diagnostic_scene")
        if not scene or any(t.get("scene_id") != scene["scene_id"] for t in traces):
            raise RuntimeError("环境诊断场景指纹错位")
        if any(t.get("time_slot") != a.get("time_slot")
               for t, a in zip(traces, item["action_traces"])):
            raise RuntimeError("环境诊断与动作时隙错位")
        started = perf_counter()
        position = self._file.tell()
        key = (int(item["sampler_id"]), scene["scene_id"])
        if key not in self._scenes:
            self._write({"record_type": "scene", "run_id": self.run_id,
                         "sampler_id": int(item["sampler_id"]),
                         **_episode_identity(item),
                         "first_episode_idx": int(item["episode_idx"]), **scene})
            self._scenes.add(key)
        logged_at = datetime.now().astimezone().isoformat(timespec="milliseconds")
        for step, trace in enumerate(traces):
            if self.detail_level == "compact":
                trace = compact_env_trace(trace)
            self._write({"record_type": "env_diagnostic", "run_id": self.run_id,
                         "logged_at": logged_at, "iteration": int(iteration),
                         "sampler_id": int(item["sampler_id"]),
                         "seed_offset": int(item["seed_offset"]),
                         "episode_idx": int(item["episode_idx"]), "step_idx": step,
                         "weights_version": int(item["weights_version"]),
                         **_episode_identity(item), **trace})
        self._file.flush()
        self.last_write_bytes = self._file.tell() - position
        self.last_write_seconds = perf_counter() - started


def summarize_diagnostics(traces):
    """Diagnostic rates only; never replace the legacy completion/reward."""
    count = len(traces)
    def rate(predicate):
        return 100.0 * sum(bool(predicate(t)) for t in traces) / count if count else float("nan")
    return {
        "count": count,
        "sensing_infeasible_rate": rate(lambda t: t["sensing"].get("current_action_infeasible") is True),
        "covariance_pass_rate": rate(lambda t: t["validation"]["covariance"]["passed"]),
        "recovered_beam_pass_rate": rate(lambda t: t["validation"]["recovered_beams"]["passed"]),
        "physical_solution_pass_rate": rate(lambda t: t["validation"]["physical_solution_verified"]),
        "diagnostic_seconds": sum(float(t.get("diagnostic_seconds", 0)) for t in traces),
        "solver_seconds": sum(float(t["solver"].get("solve_seconds", 0)) for t in traces),
        "solver_skipped_count": sum(t["solver"].get("solver_skipped") is True for t in traces),
        "presolve_seconds": sum(float(t["solver"].get("presolve_seconds", 0)) for t in traces),
    }
