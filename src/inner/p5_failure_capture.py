"""Read-only P5 failure capture; disk persistence belongs to the learner.

The scope is local to one env.step, including when detailed env logging is off.
No solver settings, solution acceptance, linearization points or RNGs are changed.
"""

from contextlib import contextmanager
from contextvars import ContextVar
import io
import math

import numpy as np


_SLOT_CAPTURE = ContextVar("p5_failure_capture", default=None)


@contextmanager
def capture_unknown_slot(enabled):
    state = {"payload": None} if enabled else None
    token = _SLOT_CAPTURE.set(state)
    try:
        yield state
    finally:
        _SLOT_CAPTURE.reset(token)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def _numeric_snapshot(ctx, fp):
    """Keep exact dtypes/complex arrays; never serialize CVXPY symbolic objects."""
    arrays = {}

    def visit(prefix, value, fusion=False):
        if fusion and hasattr(value, "getValue"):
            arrays[prefix] = np.asarray(value.getValue()).reshape(tuple(value.getShape())).copy()
        elif isinstance(value, dict):
            for key, item in value.items():
                visit(prefix + "." + str(key), item, fusion)
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                visit(prefix + "." + str(index), item, fusion)
        elif isinstance(value, (np.ndarray, np.number, int, float, complex, bool)):
            item = np.asarray(value)
            if item.dtype.kind in "biufc":
                arrays[prefix] = item.copy()

    for name in ("params", "environment", "variables"):
        visit(name, vars(getattr(ctx, name)))
    visit("fusion", vars(fp), fusion=True)
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    return stream.getvalue(), sorted(arrays)


class SolverFailureCapture:
    """Buffer the entire slot's solver output, persist only UNKNOWN/IllPosed."""

    def __init__(self, model):
        self.model = model
        self.state = _SLOT_CAPTURE.get()
        self.log = None
        self.attached = False
        self.errors = {}
        if self.state is not None:
            try:
                self.log = io.StringIO()
                model.setLogHandler(self.log)
                self.attached = True
            except Exception as exc:
                self.errors["solver_log"] = str(exc)

    def begin_iteration(self, iteration):
        if self.log is not None:
            self.log.write("\n===== PC3P iteration %d =====\n" % iteration)

    def save_failure(self, ctx, fp, result):
        if self.state is None:
            return
        status = str(result.get("solver_diagnostics", {}).get("problem_status", "")).lower()
        if not any(word in status for word in ("unknown", "illposed", "ill_posed")):
            return
        # All capture failures are diagnostic only; retain the original solver result.
        payload = {"files": {}, "manifest": {
            "schema_version": 1, "scope": "exact_failed_p5_task_and_linearization",
            "failed_iteration": result.get("iterations"),
            "linearization_iteration": result.get("iterations", 0) - 1,
            "last_solved_iteration": result.get("last_solved_iteration"),
            "solver_diagnostics": _json_safe(result.get("solver_diagnostics", {})),
            "objective_history": _json_safe(result.get("obj_history", [])),
            "rank1_gap": _json_safe(result.get("rank1_gap")),
            "errors": dict(self.errors),
        }}
        self.state["payload"] = payload
        manifest = payload["manifest"]
        try:
            manifest["mosek_version"] = str(self.model.getVersion())
        except Exception as exc:
            manifest["errors"]["mosek_version"] = str(exc)
        try:
            stream = io.BytesIO()
            self.model.writeTaskStream("task.gz", stream)
            payload["files"]["failed_p5.task.gz"] = stream.getvalue()
        except Exception as exc:
            manifest["errors"]["task"] = str(exc)
        try:
            data, keys = _numeric_snapshot(ctx, fp)
            payload["files"]["linearization.npz"] = data
            manifest["snapshot_array_keys"] = keys
        except Exception as exc:
            manifest["errors"]["linearization"] = str(exc)
        if self.log is not None:
            payload["files"]["mosek.log"] = self.log.getvalue().encode("utf-8")
        manifest["complete"] = (not manifest["errors"] and len(payload["files"]) == 3)

    def close(self):
        if self.attached:
            try:
                self.model.setLogHandler(None)
            except Exception:
                pass
        if self.log is not None:
            self.log.close()
