"""Ad-hoc verification of MHBPPO JSONL segment rotation (no training, no solver)."""
import json
import os
import sys
import tempfile

OUTER = r"D:\Desktop\Dynamic-EE-Resource-Allocation-in-Multi-UAV-Assisted-ISCC-Systems-With-Heterogeneous-Tasks\src\outer"
sys.path.insert(0, OUTER)

from algorithms.MHBPPO.action_logging import (
    ActionLogWriter, EnvDiagnosticLogWriter, JsonlRotation, segment_path,
)


def make_episode(worker, steps=4):
    return {
        "transition_dict": {
            "continuous_actions": [[0.1, 0.2] for _ in range(steps)],
            "discrete_actions": [[0, 1] for _ in range(steps)],
            "rewards": [0.5] * steps,
        },
        "action_traces": [{"time_slot": t, "physical_action": {"x": t}} for t in range(steps)],
        "diagnostic_traces": [{"time_slot": t, "scene_id": "0"} for t in range(steps)],
        "diagnostic_scene": {"scene_id": "0", "note": "fixed"},
        "sampler_id": worker, "seed_offset": 0, "episode_idx": 0, "weights_version": 0,
        "iteration": 0,
    }


def run_case(name, iterations, max_bytes, keep_segments, updates):
    root = tempfile.mkdtemp(prefix="rot_")
    action_path = os.path.join(root, "mhbppo_train_actions.jsonl")
    env_path = os.path.join(root, "mhbppo_train_env.jsonl")
    meta = {"run_id": name, "schema_version": 1}
    action_writer = ActionLogWriter(action_path, meta)
    env_writer = EnvDiagnosticLogWriter(env_path, dict(meta, action_log_path=action_path))
    writers = [action_writer, env_writer]
    rotation = JsonlRotation(iterations=iterations, max_bytes=max_bytes, keep_segments=keep_segments)
    rotations = []
    for update in range(1, updates + 1):
        rotated = rotation.due(update, [w.size for w in writers])
        if rotated:
            index = rotation.segment
            rotations.append((update, index))
            archived = [w.rotate(index) for w in writers]
            earliest = index + 1 - rotation.keep_segments
            for writer in writers:
                archived.extend(writer.prune(earliest))
        for worker in (0, 1):
            action_writer.write_episode(make_episode(worker), iteration=update)
            env_writer.write_episode(make_episode(worker), iteration=update)
    for writer in writers:
        writer.close()

    parts = sorted(p for p in os.listdir(root) if "_part" in p)
    action_parts = sorted(p for p in parts if p.startswith("mhbppo_train_actions"))
    env_parts = sorted(p for p in parts if p.startswith("mhbppo_train_env"))
    total_bytes = sum(os.path.getsize(os.path.join(root, p))
                      for p in os.listdir(root) if os.path.isfile(os.path.join(root, p)))

    # every file must parse, and every file must start with the metadata line
    for filename in sorted(os.listdir(root)):
        path = os.path.join(root, filename)
        with open(path, encoding="utf-8") as stream:
            lines = stream.read().splitlines()
        records = [json.loads(line) for line in lines]
        assert lines, f"{filename} empty"
        assert records[0]["record_type"] == "metadata", filename
        assert records[0]["run_id"] == name, filename
        action_records = [r for r in records if r.get("record_type") == "action"]
        env_records = [r for r in records if r.get("record_type") == "env_diagnostic"]
        scene_records = [r for r in records if r.get("record_type") == "scene"]
        window = sorted(r["iteration"] for r in action_records + env_records)
        span = f"iters=[{window[0]}..{window[-1]}] n={len(window)}" if window else "iters=[]"
        print(f"  {filename}: records={len(records)} action={len(action_records)} "
              f"env={len(env_records)} scene={len(scene_records)} {span}")

    # oldest retained iteration vs current update
    oldest = None
    for filename in sorted(os.listdir(root)):
        with open(os.path.join(root, filename), encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                if record.get("iteration") is not None:
                    oldest = record["iteration"] if oldest is None else min(oldest, record["iteration"])
    lag = updates - oldest
    print(f"[{name}] updates={updates} rotations={rotations}")
    print(f"[{name}] action parts={action_parts}")
    print(f"[{name}] env parts={env_parts}")
    print(f"[{name}] part count per stream={len(action_parts)} (keep_segments={keep_segments}) "
          f"| oldest iteration={oldest} | lag={lag} | total={total_bytes}B")
    # invariants
    assert len(action_parts) == len(env_parts), "streams must stay aligned"
    assert len(action_parts) <= keep_segments + 1, f"too many parts: {action_parts}"
    if iterations and not max_bytes:
        assert lag <= (keep_segments + 1) * iterations, f"iteration lag too large: {lag}"
    return total_bytes


if __name__ == "__main__":
    print("== case 1: iteration window (2 updates x keep 1) ==")
    run_case("iter-case", iterations=2, max_bytes=0, keep_segments=1, updates=10)
    print("== case 2: user default shape (1000 / keep 1) over 5000 updates ==")
    run_case("default", iterations=1000, max_bytes=0, keep_segments=1, updates=5000)
    print("== case 3: size trigger (max_bytes=4000) ==")
    run_case("bytes", iterations=0, max_bytes=4000, keep_segments=1, updates=40)
    print("== case 4: disabled (previous behaviour) ==")
    run_case("off", iterations=0, max_bytes=0, keep_segments=1, updates=10)
    print("ALL CHECKS PASSED")
