"""Read-only analysis of frozen N2 metrics and the matching text-log prefix.

Reproduce: D:/anaconda3/envs/ML2024_Task1/python.exe <this file>
No training modules, torch, environment, or solvers are imported.
"""
from __future__ import annotations

import ast
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]
METRICS = HERE / "frozen/training_updates.csv"
HYPER = HERE / "frozen/hyperparameters.json"
LOG = ROOT / "src/outer/logs/MHBPPO/2026-10-18_17-25-12/mhbppo_train.log"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


with METRICS.open(encoding="utf-8-sig", newline="") as f:
    records = list(csv.DictReader(f))
data = {key: np.array([float(r[key]) for r in records]) for key in records[0]}
END = int(data["update"][-1])
updates = {}
workers = {0: {}, 1: {}}
policy = {}
discrete = {}
alerts = []
events = []
iter_re = re.compile(r"\biter=(\d+)")
pair_re = re.compile(r"\b([A-Za-z_][A-Za-z_0-9]*)=([^\s|]+)")
array_re = re.compile(r"\b([A-Za-z_0-9]+)=\[([^\]]*)\]")
update_line_last = None
read_lines = 0
for line in LOG.open(encoding="utf-8", errors="replace"):
    read_lines += 1
    m = iter_re.search(line)
    iteration = int(m.group(1)) if m else None
    if iteration is not None and iteration > END:
        break
    if any(x in line for x in ("| [WARNING] |", "| [ERROR] |", "| [CRITICAL] |", "Traceback", "MISMATCH")):
        alerts.append({"line": read_lines, "iteration_explicit": iteration,
                       "last_completed_text_update": max(updates) if updates else 0,
                       "text": line.rstrip()})
    if "详细日志滚动" in line or "暂停逐步详细日志" in line or "恢复逐步详细日志" in line or "已保存" in line:
        events.append({"line": read_lines, "last_completed_text_update": max(updates) if updates else 0,
                       "text": line.rstrip()})
    if iteration is None:
        continue
    pairs = dict(pair_re.findall(line))
    if "[UPDATE]" in line and "[learner] iter=" in line:
        d = {"timestamp": line[:19]}
        for k, v in pairs.items():
            try:
                d[k] = float(v.rstrip("%s"))
            except ValueError:
                d[k] = v
        statuses = re.search(r"solver_status=([^ ]+)", line)
        d["solver_status"] = {s: float(rate.rstrip("%")) for s, rate in (p.split(":") for p in statuses.group(1).split(","))}
        updates[iteration] = d
        update_line_last = read_lines
    elif "[ENV]" in line:
        wid = int(pairs["worker"])
        counts = ast.literal_eval(re.search(r"solver_counts=(\{.*?\})", line).group(1))
        workers[wid][iteration] = {
            "reward": float(pairs["reward"]), "completion": float(pairs["completion"].rstrip("%")),
            "collection": float(pairs["collection"].rstrip("s")),
            "episode": int(pairs["episode"]), "wv": int(pairs["wv"]), "transitions": int(pairs["transitions"]),
            "solver_counts": counts,
        }
    elif "[POLICY]" in line:
        head = pairs["head"]
        policy.setdefault(head, {})[iteration] = {
            key: [float(x) for x in values.split(",") if x]
            for key, values in array_re.findall(line)
        }
    elif "[POLICY_DISCRETE]" in line:
        discrete[iteration] = [tuple(float(x) for x in p) for p in re.findall(r"entropy=([\d.]+),pmax=([\d.]+)", line)]

assert np.array_equal(data["update"], np.arange(1, END + 1))
assert set(updates) == set(range(1, END + 1)), (len(updates), END)
assert all(set(v) == set(updates) for v in workers.values())
assert all(np.isfinite(x).all() for x in data.values())
assert np.all(data["transitions"] == 80)
assert np.array_equal(data["env_steps"], data["update"] * 80)
assert np.array_equal(data["episodes"], data["update"] * 2)
assert np.array_equal(data["saved_policy_version"], data["update"])
assert np.array_equal(data["collection_policy_version"], data["update"] - 1)

times = np.array([datetime.fromisoformat(updates[i]["timestamp"]).timestamp() for i in range(1, END + 1)])
first_log_timestamp = datetime.fromisoformat("2026-10-18 17:25:16").timestamp()
start_metadata = datetime.fromisoformat(json.loads(HYPER.read_text(encoding="utf-8"))["run"]["started_at"])
# Log timestamps have one-second precision; wall durations are descriptive.
metadata_seconds = start_metadata.replace(tzinfo=None).timestamp()
snapshot_end = times[-1]


def describe(values):
    a = np.asarray(values, dtype=float)
    return {"mean": float(a.mean()), "median": float(np.median(a)), "p05": float(np.quantile(a, .05)),
            "p95": float(np.quantile(a, .95)), "min": float(a.min()), "max": float(a.max())}


def window(start, end):
    ids = range(start, end + 1)
    s = slice(start - 1, end)
    steps = int(data["transitions"][s].sum())
    n = end - start + 1
    # Completion is a percentage in the CSV, while optimal_count is an integer.
    complete_float = data["completion_rate"][s] * data["transitions"][s] / 100
    complete_int = np.rint(complete_float).astype(int)
    assert np.max(np.abs(complete_float - complete_int)) < 1e-9
    opt = int(data["optimal_count"][s].sum())
    counts = {key: sum(workers[w][i]["solver_counts"].get(key, 0) for w in workers for i in ids)
              for key in ("optimal", "infeasible", "unknown", "error", "other")}
    assert counts["optimal"] == opt and sum(counts.values()) == steps
    wall = times[end - 1] - (times[start - 2] if start > 1 else first_log_timestamp)
    durations = {k: float(data[k][s].sum()) for k in
                 ("collection_wait_seconds", "worker_collection_seconds_max", "update_seconds", "logging_seconds")}
    # collection wait can overlap small amounts of current worker collection,
    # whereas max worker collection is NOT an additive wall-time component.
    additive = sum(durations[k] for k in ("collection_wait_seconds", "update_seconds", "logging_seconds"))
    reward, base, q = [float(data[k][s].mean()) for k in ("reward", "base_reward", "sensing_Q")]
    d = {
        "updates": [start, end], "n_updates": n, "steps": steps, "episodes": n * 2,
        "timestamp_first": updates[start]["timestamp"], "timestamp_last": updates[end]["timestamp"],
        "completion_count": int(complete_int.sum()), "completion_percent": float(complete_int.sum() / steps * 100),
        "optimal_count": opt, "optimal_percent": opt / steps * 100,
        "optimal_minus_completion_count": opt - int(complete_int.sum()),
        "solver_counts": counts, "reward": reward, "base_reward": base, "sensing_Q": q,
        "wall_seconds": wall, "wall_steps_per_second": steps / wall,
        "duration_seconds": durations,
        "duration_percent_of_wall": {k: v / wall * 100 for k, v in durations.items()},
        "sum_measured_additive_seconds": additive,
        "wall_minus_measured_additive_seconds": wall - additive,
        "cost_per_update_seconds": {k: v / n for k, v in durations.items()},
        "seconds_per_update_wall": wall / n,
        "no_optimal_updates": int(np.sum(data["optimal_count"][s] == 0)),
        "update_optimal_count": describe(data["optimal_count"][s]),
        "worker": {},
        "diagnostic_means_text_rounded": {},
        "policy_head_array_means_text_rounded": {},
    }
    for key in ("finite_obj", "no_solution", "collision", "spectrum", "diag_c06_bad", "diag_cov_valid", "diag_beam_valid",
                "sensing_Q_valid", "sensing_unavailable", "lr_actor", "norm_count", "stale", "steps_per_second"):
        vals = [updates[i][key] for i in ids]
        d["diagnostic_means_text_rounded"][key] = describe(vals)
    for w in workers:
        wd = [workers[w][i] for i in ids]
        d["worker"][str(w)] = {
            "reward_mean_text_rounded": float(np.mean([r["reward"] for r in wd])),
            "completion_percent_text_rounded": float(np.mean([r["completion"] for r in wd])),
            "optimal_count": sum(r["solver_counts"]["optimal"] for r in wd),
            "optimal_percent": sum(r["solver_counts"]["optimal"] for r in wd) / (40 * n) * 100,
            "collection_seconds": describe([r["collection"] for r in wd]),
        }
    for head, pd in policy.items():
        use = [pd[i] for i in ids if i in pd]
        d["policy_head_array_means_text_rounded"][head] = {
            k: float(np.mean([np.mean(v[k]) for v in use])) for k in use[0]
        }
    duse = [discrete[i] for i in ids if i in discrete]
    if duse:
        da = np.asarray(duse)
        d["policy_discrete_text_rounded"] = {
            "entropy_mean": float(da[:, :, 0].mean()), "pmax_mean": float(da[:, :, 1].mean()),
            "entropy_per_head": da[:, :, 0].mean(0).tolist(), "pmax_per_head": da[:, :, 1].mean(0).tolist(),
        }
    return d


windows = {"all": window(1, END), "first100": window(1, 100), "first500": window(1, 500),
           "first1000": window(1, 1000), "last100": window(END - 99, END),
           "last500": window(END - 499, END), "previous500": window(END - 999, END - 500),
           "last1000": window(END - 999, END), "previous1000": window(END - 1999, END - 1000)}
for step in (500, 1000):
    for start in range(1, END + 1, step):
        end = min(END, start + step - 1)
        windows[f"block{step}_{start}_{end}"] = window(start, end)

roll500 = np.convolve(data["optimal_count"], np.ones(500) / (500 * 80) * 100, mode="valid")
bestend = int(np.argmax(roll500)) + 500
windows["best500_descriptive_only"] = window(bestend - 499, bestend)
q_roll500 = np.convolve(data["sensing_Q"], np.ones(500) / 500, mode="valid")
reward_roll500 = np.convolve(data["reward"], np.ones(500) / 500, mode="valid")
timing_top = np.argsort(data["collection_wait_seconds"])[-10:][::-1]
out = {
    "schema": 1, "boundary": {"last_update": END, "env_steps": int(data["env_steps"][-1]),
        "episodes": int(data["episodes"][-1]), "first_update_time": updates[1]["timestamp"],
        "last_update_time": updates[END]["timestamp"], "run_metadata_start": start_metadata.isoformat(),
        "metadata_to_last_update_seconds": snapshot_end - metadata_seconds,
        "first_learner_log_to_last_update_seconds": snapshot_end - first_log_timestamp,
        "text_update_line_last": update_line_last, "text_lines_read": read_lines,
        "csv_sha256": sha(METRICS), "hyperparameters_sha256": sha(HYPER)},
    "integrity": {
        "csv_rows": len(records), "update_log_rows": len(updates), "worker_rows": {str(w): len(v) for w, v in workers.items()},
        "all_fields_finite": True, "all_batches_80": True,
        "all_collection_version_is_update_minus_one": True, "all_saved_version_is_update": True,
        "text_checksums_ok": sum(updates[i].get("ck") == "ok" for i in updates),
        "text_wv_correct": sum(updates[i].get("wv") == i - 1 for i in updates),
        "text_stale_one": sum(updates[i].get("stale") == 1 for i in updates),
        "worker_episode_and_wv_correct": sum(v["episode"] == i - 1 and v["wv"] == i - 1 and v["transitions"] == 40
                                                   for w, wd in workers.items() for i, v in wd.items()),
        "reward_minus_base_plus_Q_max_abs": float(np.max(np.abs(data["reward"] - data["base_reward"] + data["sensing_Q"]))),
        "completion_exceeds_optimal_updates": int(np.sum(data["completion_rate"] * .8 > data["optimal_count"] + 1e-9)),
    },
    "windows": windows,
    "rolling500": {"best_optimal_end_update": bestend, "best_optimal_percent": float(roll500.max()),
                   "latest_optimal_percent": float(roll500[-1]), "Q_min": float(q_roll500.min()),
                   "Q_latest": float(q_roll500[-1]), "reward_max": float(reward_roll500.max()),
                   "reward_latest": float(reward_roll500[-1])},
    "unknown_solver_transitions_by_update_worker": [
        {"update": i, "worker": w, "timestamp": updates[i]["timestamp"],
         "unknown_count": d["solver_counts"]["unknown"]}
        for w, wd in workers.items() for i, d in wd.items() if d["solver_counts"]["unknown"]
    ],
    "alerts_with_prefix_boundary_caveat": alerts, "log_events_with_prefix_boundary_caveat": events,
    "slowest_collection_wait_updates": [{"update": int(i + 1), "timestamp": updates[int(i + 1)]["timestamp"],
        "seconds": float(data["collection_wait_seconds"][i]), "optimal_count": int(data["optimal_count"][i])} for i in timing_top],
    "notes": [
        "Frozen CSV is authoritative; live log is streamed only through matching update prefix.",
        "All success rates are transition rates, not probability that an entire 40-slot episode finishes.",
        "Log timestamp resolution is one second; wall-time cost sums omit broadcast, CSV, text/TensorBoard, checkpoint and other overhead.",
        "worker_collection_seconds_max is an overlapping diagnostic, never added to collection wait/PPO/logging.",
        "Text diagnostics/policy arrays are rounded; CSV reward/Q are full precision.",
        "last500 and previous500 are adjacent fixed trailing windows; best500 is descriptive and selected retrospectively.",
        "No frozen-policy evaluation, new-seed validation, or controlled ablation was run. Training improvement does not establish deployment success or isolate causality.",
    ],
}
(HERE / "trends.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")

md = ["# N2 run frozen training trend and throughput analysis", "",
      f"Boundary: update {END}, {int(data['env_steps'][-1])} transitions, {int(data['episodes'][-1])} episodes. "
      f"Last complete UPDATE: {updates[END]['timestamp']}. Fixed seed 42 / scene offset 0 / fixed shared exogenous scene.", "",
      "The training learns in aggregate and recent completion still improves, but absolute success remains low. "
      "Q decreases over the long run and rises slightly between the two latest adjacent 500-update windows. "
      "This is an on-policy training curve conclusion, without a frozen-policy or independent-scene evaluation.", "",
      "| Window | Completion % | Optimal % | Reward | Base reward | Q | Steps/s wall |", "|---|---:|---:|---:|---:|---:|---:|"]
for key in ("first100", "first500", "first1000", "previous1000", "last1000", "previous500", "last500", "last100", "all"):
    w = windows[key]
    md.append(f"| {key} ({w['updates'][0]}–{w['updates'][1]}) | {w['completion_percent']:.5f} | {w['optimal_percent']:.5f} | "
              f"{w['reward']:.6f} | {w['base_reward']:.6f} | {w['sensing_Q']:.6f} | {w['wall_steps_per_second']:.3f} |")
md += ["", "## Non-overlapping 500-update windows", "",
       "| Updates | Completion % | Optimal % | Reward | Base reward | Q | Wall s/update |", "|---|---:|---:|---:|---:|---:|---:|"]
for key, w in windows.items():
    if key.startswith("block500_"):
        md.append(f"| {w['updates'][0]}–{w['updates'][1]} | {w['completion_percent']:.5f} | {w['optimal_percent']:.5f} | "
                  f"{w['reward']:.6f} | {w['base_reward']:.6f} | {w['sensing_Q']:.6f} | {w['seconds_per_update_wall']:.3f} |")
md += ["", "## Timing (additive measurements and overlap)", "",
       "| Window | Wall seconds | Collection wait % | PPO % | Detailed JSONL logging % | Unmeasured % |", "|---|---:|---:|---:|---:|---:|"]
for key in ("all", "first500", "previous500", "last500", "last100"):
    w = windows[key]
    p = w["duration_percent_of_wall"]
    md.append(f"| {key} | {w['wall_seconds']:.0f} | {p['collection_wait_seconds']:.2f} | {p['update_seconds']:.2f} | "
              f"{p['logging_seconds']:.2f} | {w['wall_minus_measured_additive_seconds'] / w['wall_seconds'] * 100:.2f} |")
md += ["", "Worker max collection time overlaps collection wait, so its percentage must never be added. "
       "CSV/text/TensorBoard/checkpoint and broadcast overhead are in the unmeasured residual; detailed JSONL logging_seconds alone does not capture all I/O.", "",
       "## Integrity and anomalies", "",
       f"- {len(updates)} UPDATE / {sum(len(v) for v in workers.values())} ENV rows match the frozen boundary. "
       f"Checksums ok {out['integrity']['text_checksums_ok']}/{END}; policy versions and stale=1 follow the synchronous contract.",
       f"- Solver counts: {windows['all']['solver_counts']}. Optimal minus completion: {windows['all']['optimal_minus_completion_count']} transitions.",
       f"- Reward = base_reward − Q maximum numeric discrepancy: {out['integrity']['reward_minus_base_plus_Q_max_abs']:.3g}.",
       f"- Prefix warning/error/critical or checksum mismatch lines: {len(alerts)}; log events: {len(events)}. See trends.json for exact evidence.", "",
       "## Scope limits", ""]
md.extend(f"- {x}" for x in out["notes"])
(HERE / "trends.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print(json.dumps({"boundary": out["boundary"], "integrity": out["integrity"],
                  "selected": {key: {k: v for k, v in windows[key].items() if k in
                      ("updates", "completion_percent", "optimal_percent", "reward", "base_reward", "sensing_Q", "solver_counts",
                       "wall_seconds", "wall_steps_per_second", "duration_percent_of_wall", "worker")}
                               for key in ("all", "first500", "previous500", "last500", "last100")},
                  "alerts": alerts, "events": events, "rolling500": out["rolling500"]}, indent=2, ensure_ascii=False))
