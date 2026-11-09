"""Last-1% UNKNOWN summaries and complete P5 artifacts, learner-only disk I/O."""

import hashlib
import json
import os
import uuid
from datetime import datetime

from algorithms.MHBPPO.action_logging import _json_value


def unknown_start_iteration(max_updates):
    """One-based inclusive window containing ceil(max_updates / 100) updates."""
    return (99 * int(max_updates)) // 100 + 1 if max_updates > 0 else None


class UnknownLogWriter:
    MAX_BYTES = 1024 * 1024
    MAX_EVENTS = 200
    MAX_RECORD_BYTES = 8192

    def __init__(self, path, max_updates, metadata, logger):
        self.path = path
        self.max_updates = int(max_updates)
        self.start_iteration = unknown_start_iteration(self.max_updates)
        self.artifact_directory = os.path.splitext(os.fspath(path))[0] + "_artifacts"
        self.metadata = metadata
        self.logger = logger
        self.events = 0
        self.bytes_written = 0
        self._file = None
        self._stopped = False

    @staticmethod
    def _encode(record):
        return (json.dumps(_json_value(record), ensure_ascii=False, allow_nan=False,
                           separators=(",", ":")) + "\n").encode("utf-8")

    def _write(self, data):
        self._file.write(data)
        self._file.flush()
        self.bytes_written += len(data)

    def _save_capture(self, payload, record):
        if payload is None:
            return {"status": "unavailable", "reason": "worker_capture_missing"}
        directory_name = (f"iter_{record['iteration']:06d}_worker_{record['worker_id']}_"
                          f"step_{record['step_idx']:03d}_" + uuid.uuid4().hex)
        directory = os.path.join(self.artifact_directory, directory_name)
        relative = os.path.relpath(directory, os.path.dirname(os.path.abspath(self.path))).replace(os.sep, "/")
        manifest = {**payload.get("manifest", {}),
                    **{key: record[key] for key in (
                        "iteration", "worker_id", "episode_idx", "step_idx", "time_slot")},
                    "run_metadata": self.metadata, "files": {}}
        reference = {"status": "partial", "directory": relative,
                     "manifest": relative + "/manifest.json"}
        try:
            os.makedirs(self.artifact_directory, exist_ok=True)
            os.mkdir(directory)
            for name in ("failed_p5.task.gz", "linearization.npz", "mosek.log"):
                if name not in payload.get("files", {}):
                    continue
                data = payload["files"][name]
                with open(os.path.join(directory, name), "xb") as stream:
                    stream.write(data)
                manifest["files"][name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            manifest["complete"] = bool(manifest.get("complete") and len(manifest["files"]) == 3)
            with open(os.path.join(directory, "manifest.json"), "xb") as stream:
                stream.write(self._encode(manifest))
            reference["status"] = "complete" if manifest["complete"] else "partial"
            if not manifest["complete"]:
                reference["errors"] = {str(key): str(value)[:256]
                                       for key, value in list(manifest.get("errors", {}).items())[:4]}
                self.logger.warning("[learner] UNKNOWN 完整采集不完整: %s", directory)
        except Exception as exc:
            reference.update(status="error", error=str(exc)[:512])
            self.logger.warning("[learner] UNKNOWN 附件写入失败: %s", exc)
        return reference

    def write_episode(self, episode, iteration):
        if (self.start_iteration is None or not self.start_iteration <= iteration <= self.max_updates
                or self._stopped):
            return
        diagnostics = episode.get("diagnostic_traces", [])
        for step, action in enumerate(episode["action_traces"]):
            diagnostic = diagnostics[step] if step < len(diagnostics) else {}
            outcome = action.get("outcome", {})
            solver = diagnostic.get("solver", outcome)
            if solver.get("raw_solver_status_kind", solver.get("solver_status_kind")) != "unknown":
                continue
            physical = action.get("physical_action", {})
            matching = physical.get("uavs_cus_matched_matrix", [])
            sensing = diagnostic.get("sensing", {})
            record = {
                "record_type": "unknown", "iteration": int(iteration),
                "worker_id": episode.get("worker_id", episode.get("sampler_id")),
                "episode_idx": episode["episode_idx"], "step_idx": step,
                "time_slot": action.get("time_slot"),
                "policy_version": episode.get("weights_version"),
                "scene_signature": episode.get("scene_signature"),
                "logged_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "environment_diagnostics_available": bool(diagnostic),
                "solver": {key: solver[key] for key in (
                    "solver_status", "solver_status_kind", "raw_solver_status",
                    "raw_solver_status_kind", "problem_status", "primal_solution_status",
                    "dual_solution_status", "optimizer_response_code", "interior_point_iterations",
                    "intpntPrimalObj", "intpntDualObj", "intpntPrimalFeas", "intpntDualFeas",
                    "intpntOptStatus", "unavailable_fields", "solve_exception",
                    "solver_iterations", "failed_iteration", "last_solved_iteration",
                    "returned_solution", "solver_converged", "rank1_gap", "solve_seconds",
                    "presolve_certificate", "infeasibility_certificate",
                ) if key in solver},
                "physical_action": {key: physical[key] for key in (
                    "uav_angles", "uav_distances", "uav_off_durations", "cu_off_powers",
                    "uavs_pos_before",
                ) if key in physical},
                "cu_matching": [max(range(len(row)), key=row.__getitem__) for row in matching],
                "target_indices": diagnostic.get("target_indices"),
                "cu_tasks": action.get("cu_tasks"),
                "sensing": {key: sensing[key] for key in (
                    "margin", "min_margin", "required_sensing_power_w", "gamma_w",
                    "target_distances_m", "current_action_infeasible",
                ) if key in sensing},
                "outer_failed_checks": diagnostic.get("outer_action", {}).get("failed_checks"),
            }
            data = self._encode(record)
            if len(data) > self.MAX_RECORD_BYTES - 2048:
                record = {key: record[key] for key in (
                    "record_type", "iteration", "worker_id", "episode_idx", "step_idx", "time_slot",
                )}
                record.update(optimizer_response_code=solver.get("optimizer_response_code"),
                              raw_solver_status=str(solver.get("raw_solver_status"))[:256],
                              details_omitted="record_size_limit")
                data = self._encode(record)
            try:
                if self._file is None:
                    self._file = open(self.path, "xb")
                    self._write(self._encode({
                        "record_type": "metadata", "schema_version": 2,
                        "start_iteration": self.start_iteration, "max_updates": self.max_updates,
                        "end_iteration": self.max_updates,
                        "window_updates": self.max_updates - self.start_iteration + 1,
                        "max_bytes": self.MAX_BYTES, "max_events": self.MAX_EVENTS,
                        "scope": "summary_with_exact_failed_p5_artifacts", **self.metadata,
                    }))
                # Reserve space for the artifact reference and a possible stop reason.
                if self.events >= self.MAX_EVENTS or self.bytes_written + len(data) + 4096 > self.MAX_BYTES:
                    self._write(self._encode({"record_type": "limit_reached", "saved_events": self.events}))
                    self._stopped = True
                    self.close()
                    self.logger.warning("[learner] UNKNOWN 专项日志达到保存上限，停止追加")
                    return
                payload = episode.get("unknown_capture_payloads", {}).get(step)
                record["unknown_capture"] = self._save_capture(payload, record)
                data = self._encode(record)
                self._write(data)
                self.events += 1
            except OSError as exc:
                self._stopped = True
                self.logger.warning("[learner] UNKNOWN 专项日志写入失败，停止追加: %s", exc)
                self.close()
                return

    def close(self):
        if self._file is not None:
            stream = self._file
            self._file = None
            try:
                stream.close()
            except OSError as exc:
                self.logger.warning("[learner] UNKNOWN 专项日志关闭失败: %s", exc)
