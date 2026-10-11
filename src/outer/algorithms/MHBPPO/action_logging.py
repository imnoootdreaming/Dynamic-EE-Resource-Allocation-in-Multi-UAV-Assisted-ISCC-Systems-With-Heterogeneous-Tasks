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


def build_run_log_paths(outer_root, log_file, started_at):
    """Keep explicit directories; prefix both files with the same run timestamp."""
    configured_path = os.path.join(outer_root, log_file)
    directory, filename = os.path.split(configured_path)
    os.makedirs(directory, exist_ok=True)
    prefix = started_at.strftime("%H-%M-%S_%Y-%m-%d_%f")
    stem, extension = os.path.splitext(filename)
    extension = extension or ".log"
    index = 0
    while True:
        suffix = "" if index == 0 else f"_{index}"
        run_stem = f"{prefix}_{stem}{suffix}"
        training_path = os.path.join(directory, run_stem + extension)
        action_path = os.path.join(directory, run_stem + "_actions.jsonl")
        env_path = diagnostic_log_path(action_path)
        if not any(os.path.exists(p) for p in (training_path, action_path, env_path)):
            return training_path, action_path
        index += 1


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


class ActionLogWriter:
    """Learner-only writer: one metadata line, then one line per consumed transition."""

    def __init__(self, path, metadata):
        self._file = open(path, "x", encoding="utf-8")
        self.run_id = metadata.get("run_id")
        try:
            self._write({"record_type": "metadata", "schema_version": 1, **metadata})
            self._file.flush()
            atexit.register(self.close)
        except BaseException:
            self._file.close()
            raise

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
                "raw_action": {
                    "continuous": transitions["continuous_actions"][step],
                    "discrete": transitions["discrete_actions"][step],
                },
                "environment_action": trace,
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
        "configs/base_params.py", "configs/madrl_args.py", "configs/seed_manager.py",
        "environment/my_env.py", "environment/channel_models.py", "environment/topology.py",
        "environment/observation.py", "environment/mapping.py",
        "reward/my_reward.py", "reward/cccp_bridge.py", "reward/feasibility_diagnostics.py",
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
        "base_args": base_args.as_dict() if hasattr(base_args, "as_dict") else vars(base_args),
        "inner_parameter_source": None if params is None else vars(params).copy(),
        "madrl_args": vars(madrl_args).copy(),
        "diagnostic_semantics": "read_only_original_constraints; legacy reward/success unchanged",
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

    def write_episode(self, item, iteration):
        traces = item.get("diagnostic_traces", [])
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
                         "first_episode_idx": int(item["episode_idx"]), **scene})
            self._scenes.add(key)
        logged_at = datetime.now().astimezone().isoformat(timespec="milliseconds")
        for step, trace in enumerate(traces):
            self._write({"record_type": "env_diagnostic", "run_id": self.run_id,
                         "logged_at": logged_at, "iteration": int(iteration),
                         "sampler_id": int(item["sampler_id"]),
                         "seed_offset": int(item["seed_offset"]),
                         "episode_idx": int(item["episode_idx"]), "step_idx": step,
                         "weights_version": int(item["weights_version"]), **trace})
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
    }
