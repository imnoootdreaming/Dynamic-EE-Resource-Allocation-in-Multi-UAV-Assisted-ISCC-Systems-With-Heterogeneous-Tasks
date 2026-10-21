"""Read frozen-range JSONL by byte bisection; NumPy only, no training/solver imports."""
import collections
import csv
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[5]
LOG = ROOT / "src/outer/logs/MHBPPO/2026-10-18_17-25-12"
PATTERN = re.compile(rb'"iteration":\s*(\d+)')
ROWS = ["iteration", "worker", "slot", "optimal", "Q", "base_reward", "reward", "sensing_pass",
        "collision_penalty", "matching_penalty", "min_margin", "no_min_matching", "current_feasible_matching",
        "min_feasible_matching", "avg_cu_power", "avg_speed", "solve_seconds", "diagnostic_seconds",
        "solver_iterations", "q_error", "reward_error", "infeasible", "unknown", "error", "outer_pass",
        "physical_solution_verified", "success"]
UAV = ["margin", "distance", "gain", "gamma", "q", "interference", "noise_only_margin", "best_current_margin",
       "best_min_power_margin", "selected_min_power_margin"]
ACTION = ["matched_cu", "matched_power", "off_duration", "step_distance", "angle", "target_distance_after",
          "progress_toward_current_target", "beam_std", "beam_signed_mean_abs"]


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def seek_before(stream, iteration, size):
    low, high = 0, size
    while high - low > 65536:
        mid = (low + high) // 2
        stream.seek(mid); stream.readline(); pos = stream.tell(); line = stream.readline()
        match = PATTERN.search(line[:768])
        if match and int(match[1]) < iteration:
            low = pos
        else:
            high = mid
    stream.seek(low)
    return low


def fragments():
    csv_rows = list(csv.DictReader((OUT / "frozen/training_updates.csv").open(encoding="utf-8-sig")))
    frozen = max(int(x["update"]) for x in csv_rows)
    assert frozen == 5804, frozen
    start = frozen - 999
    return csv_rows, start, frozen, [("_part005", start, 5000), ("", 5001, frozen)]


def records(path, lo, hi, record_type, scans):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        size = f.seek(0, 2)
        offset = seek_before(f, lo, size)
        start = time.perf_counter(); n = 0; final_pos = offset
        for line in f:
            if f.tell() > size:
                break
            match = PATTERN.search(line[:768])
            if not match:
                continue
            iteration = int(match[1])
            if iteration < lo:
                continue
            if iteration > hi:
                break
            r = json.loads(line)
            if r.get("record_type") != record_type:
                continue
            digest.update(line); n += 1; final_pos = f.tell()
            yield r
        scans.append({"path": str(path), "lo": lo, "hi": hi, "opened_size_bytes": size,
                      "start_byte": offset, "selected_end_byte": final_pos, "records": n,
                      "selected_lines_sha256": digest.hexdigest(), "seconds": time.perf_counter() - start})


def summarize(rows, uav, action, lo, hi, slots=None):
    mask = (rows[:, 0] >= lo) & (rows[:, 0] <= hi)
    if slots is not None:
        mask &= (rows[:, 2] >= slots[0]) & (rows[:, 2] <= slots[1])
    r, u, a = rows[mask], uav[mask], action[mask]
    optimal = r[:, 3].astype(bool); failed = ~optimal; failed_sense = r[:, 7] == 0
    late = r[:, 2] >= 8
    actual_sense = r[:, 7] > 0
    matching_salvage = failed_sense & (r[:, 12] > 0)
    min_salvage = failed_sense & (r[:, 13] > 0)
    ret = {"lo": lo, "hi": hi, "slots": slots, "records": len(r), "optimal_count": int(optimal.sum()),
        "optimal_percent": float(100 * optimal.mean()), "completion_percent": float(100 * r[:, 26].mean()),
        "Q": float(r[:, 4].mean()), "failed_Q": float(r[failed, 4].mean()), "base_reward": float(r[:, 5].mean()),
        "reward": float(r[:, 6].mean()), "sensing_pass_percent": float(100 * actual_sense.mean()),
        "failed_steps_rejected_by_sensing_percent": float(100 * failed_sense[failed].mean()),
        "sensing_pass_but_not_optimal_count": int((actual_sense & failed).sum()),
        "per_uav_sensing_pass_percent": (100 * (u[:, :, 0] >= 1 - 1e-6).mean(axis=0)).tolist(),
        "gain_mean": float(u[:, :, 2].mean()), "gain_per_uav_mean": u[:, :, 2].mean(axis=0).tolist(),
        "gain_quantiles": np.quantile(u[:, :, 2], [.1, .5, .9]).tolist(), "distance_mean": float(u[:, :, 1].mean()),
        "distance_per_uav_mean": u[:, :, 1].mean(axis=0).tolist(), "gamma_mean": float(u[:, :, 3].mean()),
        "gamma_median": float(np.median(u[:, :, 3])), "min_margin_median": float(np.median(r[:, 10])),
        "interference_noise_ratio_median": float(np.median(u[:, :, 5] / PARAMS["sigma_2"])),
        "no_min_matching_percent": float(100 * r[:, 11].mean()),
        "current_power_matching_exists_percent": float(100 * r[:, 12].mean()),
        "min_power_matching_exists_percent": float(100 * r[:, 13].mean()),
        "sensing_failed_but_current_power_matching_exists_percent_all_steps": float(100 * matching_salvage.mean()),
        "sensing_failed_but_min_power_matching_exists_percent_all_steps": float(100 * min_salvage.mean()),
        "selected_edges_at_min_power_joint_pass_percent": float(100 * (u[:, :, 9] >= 1 - 1e-6).all(axis=1).mean()),
        "noise_only_joint_sensing_pass_percent": float(100 * (u[:, :, 6] >= 1 - 1e-6).all(axis=1).mean()),
        "collision_penalty": float(r[:, 8].mean()), "matching_penalty": float(r[:, 9].mean()),
        "avg_cu_power": float(r[:, 14].mean()), "matched_power": float(a[:, :, 1].mean()),
        "matched_power_per_uav_mean": a[:, :, 1].mean(axis=0).tolist(),
        "avg_speed": float(r[:, 15].mean()), "mean_solver_seconds": float(r[:, 16].mean()),
        "solver_seconds_sensing_failed_sum": float(r[failed_sense, 16].sum()),
        "solver_seconds_sensing_passed_sum": float(r[~failed_sense, 16].sum()),
        "solver_seconds_sensing_failed_mean": float(r[failed_sense, 16].mean()) if failed_sense.any() else None,
        "solver_seconds_sensing_passed_mean": float(r[~failed_sense, 16].mean()) if (~failed_sense).any() else None,
        "solver_seconds_sensing_failed_share": float(r[failed_sense, 16].sum()/r[:, 16].sum()),
        "sensing_failed_count": int(failed_sense.sum()),
        "mean_solver_iterations": float(r[:, 18].mean()), "physical_solution_verified_percent": float(100 * r[:, 25].mean()),
        "outer_pass_percent": float(100 * r[:, 24].mean()), "unknown_count": int(r[:, 22].sum()),
        "error_count": int(r[:, 23].sum()), "beam_std_mean": float(a[:, :, 7].mean()),
        "beam_signed_mean_abs": float(a[:, :, 8].mean()),
        "progress_toward_current_target_m_mean": float(a[:, :, 6].mean()),
        "off_duration_mean": float(a[:, :, 2].mean()),
        "per_uav_CU_counts": [[int((a[:, i, 0] == j).sum()) for j in range(PARAMS["J"])] for i in range(PARAMS["I"])]}
    if slots is None:
        ret["last32_optimal_percent"] = float(100 * r[late, 3].mean())
        ret["first8_share_of_optimal_percent"] = float(100 * r[~late, 3].sum() / max(1, optimal.sum()))
    return ret


def main():
    global PARAMS
    csv_rows, lo, hi, intervals = fragments()
    metadata = json.loads((LOG / "mhbppo_train_env_part005.jsonl").open("rb").readline())
    PARAMS = metadata["inner_parameter_source"]
    save("physical_metadata.json", metadata)
    denominator = PARAMS["xi_0"] * PARAMS["rho_ref"] * PARAMS["N"] ** 2
    count = (hi - lo + 1) * 80
    rows = np.empty((count, len(ROWS))); uav = np.empty((count, PARAMS["I"], len(UAV)))
    action = np.empty((count, PARAMS["I"], len(ACTION)))
    scans = []; count_by_update = collections.Counter(); ids = {}; scenes = {}
    for suffix, start, end in intervals:
        path = LOG / f"mhbppo_train_env{suffix}.jsonl"
        with path.open("rb") as f:
            current_meta = json.loads(f.readline())
            assert current_meta["source_sha256"] == metadata["source_sha256"]
            assert current_meta["shared_exogenous_scene"] == metadata["shared_exogenous_scene"]
            for _ in range(3):
                v = json.loads(f.readline())
                if v.get("record_type") == "scene":
                    scenes[v["scene_id"]] = v
        for r in records(path, start, end, "env_diagnostic", scans):
            iteration, worker, slot = r["iteration"], r["worker_id"], r["step_idx"]
            idx = (iteration - lo) * 80 + worker * 40 + slot
            assert (worker, slot) == divmod(count_by_update[iteration], 40)
            assert r["weights_version"] == r["collection_round"] == iteration - 1
            assert r["scene_signature"] == metadata["shared_exogenous_scene"]["signature"]
            count_by_update[iteration] += 1
            sense, comp, solver = r["sensing"], r["reward_components"], r["solver"]
            assert sense["status"] == "checked" and comp["sensing_penalty_status"] == "available"
            margin = np.asarray(sense["margin"]); distance = np.asarray(sense["target_distances_m"])
            lam = np.asarray(sense["lambda_max_g"]); gamma = np.asarray(sense["gamma_w"])
            gain = lam * distance**4 / denominator
            np.testing.assert_allclose(margin, PARAMS["P_max_uav"] * lam / (PARAMS["eps_sinr"] * gamma), rtol=1e-12)
            deficit = np.maximum(0, -np.log(np.maximum(margin, np.finfo(float).tiny)))
            q = deficit / (deficit + np.log(10)); q[margin == 0] = 1
            score, extra = float(comp["sensing_penalty_score"]), float(comp["sensing_penalty_applied"])
            base, reward, kind = float(r["base_reward"]), float(r["actual_reward"]), solver["solver_status_kind"]
            c = sense["candidates"]
            rows[idx] = [iteration, worker, slot, kind == "optimal", score, base, reward, not sense["current_action_infeasible"],
                comp["uav_collision_penalty_sum"], comp["bs_alloc_spectrum_penalty"], sense["min_margin"],
                c["no_matching_at_min_power_certified"], c["current_power_matching"] is not None,
                c["min_power_matching"] is not None, comp["avg_cu_off_power"], comp["avg_uav_speed"],
                solver["solve_seconds"], r["diagnostic_seconds"], solver["solver_iterations"], score - q.mean(),
                reward - (base - extra), kind == "infeasible", kind == "unknown", kind == "error", r["outer_action"]["passed"],
                r["validation"]["physical_solution_verified"], r["legacy_success"]]
            current = np.asarray(c["current_power_margin"]); minimum = np.asarray(c["min_power_margin"])
            uav[idx] = np.stack([margin, distance, gain, gamma, q, sense["interference_w"],
                PARAMS["P_max_uav"] * lam / (PARAMS["eps_sinr"] * PARAMS["sigma_2"]),
                current.max(axis=1), minimum.max(axis=1), np.zeros(PARAMS["I"])], axis=1)
            ids[idx] = (r["actor_signature"], r["norm_signature"], r["scene_signature"], r["scene_id"], r["target_indices"], minimum)
        print("env", start, end, "done", flush=True)
    assert len(ids) == count and set(count_by_update.values()) == {80}
    action_seen = set()
    for suffix, start, end in intervals:
        for r in records(LOG / f"mhbppo_train_actions{suffix}.jsonl", start, end, "action", scans):
            iteration, worker, slot = r["iteration"], r["worker_id"], r["step_idx"]
            idx = (iteration - lo) * 80 + worker * 40 + slot
            assert idx not in action_seen; action_seen.add(idx)
            sig = ids[idx]
            assert (r["actor_signature"], r["norm_signature"], r["scene_signature"]) == sig[:3]
            a = r["environment_action"]; phys = a["physical_action"]
            eta = np.asarray(phys["uavs_cus_matched_matrix"]); selected = eta.argmax(axis=1)
            powers = np.asarray(phys["cu_off_powers"]); matched_power = powers[selected]
            assert abs(powers.mean() - rows[idx, 14]) < 1e-12
            # All logged rows are exact binary, one CU per UAV. Duplicated CUs are legal to log but fail outer checks.
            assert np.all(eta.sum(axis=1) == 1)
            uav[idx, :, 9] = sig[5][np.arange(PARAMS["I"]), selected]
            pos_before, pos_after = np.asarray(phys["uavs_pos_before"]), np.asarray(phys["uavs_pos_after"])
            targets = np.asarray(scenes[sig[3]]["target_positions"])[sig[4]]
            before = np.linalg.norm(pos_before - targets, axis=1); after = np.linalg.norm(pos_after - targets, axis=1)
            np.testing.assert_allclose(before, uav[idx, :, 1], rtol=1e-12)
            pd = a["policy_distribution"]["continuous"]
            beam_std = (np.asarray(pd["uav_rec_beam_dir_real"]["std"]).reshape(PARAMS["I"], -1).mean(axis=1)
                      + np.asarray(pd["uav_rec_beam_dir_imag"]["std"]).reshape(PARAMS["I"], -1).mean(axis=1)) / 2
            signed_abs = (np.abs(2*np.asarray(pd["uav_rec_beam_dir_real"]["mean"]).reshape(PARAMS["I"], -1)-1).mean(axis=1)
                        + np.abs(2*np.asarray(pd["uav_rec_beam_dir_imag"]["mean"]).reshape(PARAMS["I"], -1)-1).mean(axis=1)) / 2
            action[idx] = np.stack([selected, matched_power, phys["uav_off_durations"], phys["uav_distances"],
                phys["uav_angles"], after, before-after, beam_std, signed_abs], axis=1)
        print("actions", start, end, "done", flush=True)
    assert len(action_seen) == count
    cross = []
    for iteration in range(lo, hi + 1):
        r = rows[(iteration-lo)*80:(iteration-lo+1)*80]
        original = csv_rows[iteration-1]
        assert int(r[:, 3].sum()) == int(original["optimal_count"])
        assert abs(r[:, 4].mean() - float(original["sensing_Q"])) < 1e-12
        assert abs(r[:, 5].mean() - float(original["base_reward"])) < 1e-12
        assert abs(r[:, 6].mean() - float(original["reward"])) < 1e-12
        cross.append(iteration)
    assert np.abs(rows[:, 19:21]).max() < 1e-12
    assert (uav[:, :, 2] >= 0).all() and (uav[:, :, 2] <= 1 + 1e-10).all()
    np.savez_compressed(OUT/"physical_metrics.npz", rows=rows, uav=uav, action=action)
    result = {"schema": {"rows": ROWS, "uav": UAV, "action": ACTION},
        "coverage": {"frozen_last_update": hi, "lo": lo, "hi": hi, "records": count, "scans": scans,
        "complete_80_step_batches": len(cross), "csv_matched": [lo, hi],
        "max_Q_identity_error": float(np.abs(rows[:,19]).max()), "max_reward_identity_error": float(np.abs(rows[:,20]).max())},
        "windows": {"previous500": summarize(rows,uav,action,lo,hi-500), "last500": summarize(rows,uav,action,hi-499,hi),
                    "last100": summarize(rows,uav,action,hi-99,hi)},
        "slot_groups_last500": [summarize(rows,uav,action,hi-499,hi,[s,s+3]) for s in range(0,40,4)],
        "broad_slot_groups_last500": [summarize(rows,uav,action,hi-499,hi,slots) for slots in ([0,3],[4,7],[8,39])],
        "last500_per_worker": {str(w): summarize(rows[rows[:,1]==w],uav[rows[:,1]==w],action[rows[:,1]==w],hi-499,hi) for w in (0,1)},
        "limitations": ["M>=1 and any candidate matching concern sensing necessary conditions only; no complete task feasibility certificate.",
                        "Noise-only margin and minimum-power alternatives are diagnostics at fixed geometry/beam, not executed actions.",
                        "Only updates 4805..5804 physically extracted; no extrapolation of detailed metrics to earlier run."]}
    save("physical_summary.json", result)
    print(json.dumps({k:{x:v[x] for x in ("optimal_percent","Q","gain_mean","sensing_pass_percent","last32_optimal_percent") } for k,v in result["windows"].items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
