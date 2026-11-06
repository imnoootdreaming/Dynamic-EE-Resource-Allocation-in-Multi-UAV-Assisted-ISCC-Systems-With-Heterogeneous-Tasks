"""Verify rollout identity, physical logs and Pure evaluation checkpoint."""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

OUTER = Path(__file__).resolve().parents[2]
if str(OUTER) not in sys.path:
    sys.path.insert(0, str(OUTER))
from algorithms.Pure_MHBPPO.checkpoint import load_environment_args, load_normalization, normalize_state
from algorithms.Pure_MHBPPO.time_decoder import TIME_CONSTRAINTS
from algorithms.Pure_MHBPPO.Pure_MHBPPO_agent import MultiHeadActor
from algorithms.MHBPPO.rollout_batch import actor_state_signature


def verify(run_dir):
    root = Path(run_dir)
    config = json.loads((root / "configuration.json").read_text(encoding="utf-8"))
    with (root / "training_updates.csv").open(encoding="utf-8", newline="") as stream:
        records = list(csv.DictReader(stream))
    episodes = [json.loads(line) for line in (root / "episode_metadata.jsonl").read_text(encoding="utf-8").splitlines()]
    count = config["hyperparameters"]["num_samplers"]
    assert len(episodes) == len(records) * count
    for idx, record in enumerate(records, start=1):
        assert int(record["iteration"]) == idx and int(record["policy_version"]) == idx-1
        batch = [ep for ep in episodes if ep["iteration"] == idx]
        assert sorted(ep["worker_id"] for ep in batch) == list(range(count))
        assert all(ep["policy_version"] == idx-1 for ep in batch)
        assert len({ep["actor_signature"] for ep in batch}) == 1
        if idx > 1:
            previous = root / f"iter_{idx-1}" / "pure_mhbppo_actor.pth"
            if previous.exists():
                assert actor_state_signature(torch.load(previous, map_location="cpu", weights_only=True)) == batch[0]["actor_signature"]
    final = root / f"iter_{len(records)}"
    environment_args = load_environment_args(final)
    assert environment_args.pure_time_decoder == config["hyperparameters"].get("pure_time_decoder", "legacy")
    stats = load_normalization(final, expected_dim=config["state_dim"])
    assert stats["iteration"] == len(records)
    with np.load(final / "state_normalization.npz", allow_pickle=False) as saved:
        assert int(saved["count"]) == int(records[-1]["env_steps"]) + len(episodes)
    actor = MultiHeadActor(config["state_dim"], config["hyperparameters"]["hidden_dim"],
                           config["continuous_action_splits"], config["discrete_action_dims"],
                           continuous_dist_type="beta")
    actor.load_state_dict(torch.load(final / "pure_mhbppo_actor.pth", map_location="cpu", weights_only=True))
    state = normalize_state(np.zeros(config["state_dim"], dtype=np.float32), stats)
    with torch.no_grad():
        sample = actor.sample(torch.as_tensor(state, dtype=torch.float32).unsqueeze(0))
    assert all(torch.isfinite(part).all() for part in sample)
    assert sample[0].shape[-1] == sum(config["continuous_action_splits"].values())
    for worker in range(count):
        assert np.load(final / f"uav_trajectory_worker_{worker}.npy").shape[0] > 1
        assert (final / f"uav_trajectory_worker_{worker}.png").is_file()
    traces = []
    log_root = Path(config.get("run_paths", {}).get("log_dir", root))
    for path in log_root.glob("physical_traces*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            value = json.loads(line)
            if value.get("record_type") == "physical_trace":
                traces.append(value)
                assert value["physical"]["solver_status_kind"] == "not_called"
                assert len(value["physical"]["checks"]) == 15
                assert len(value["raw_continuous"]) == sample[0].shape[-1]
                timing = value.get("time_decoding")
                if timing is not None:
                    assert timing["mode"] == environment_args.pure_time_decoder
                    if timing["applied"]:
                        assert all(value["physical"]["checks"][key]["passed"] for key in TIME_CONSTRAINTS)
                        assert timing["used_cpu_hz"] <= timing["cpu_limit_hz"] * (1 + 1e-12)
                        mapped = np.asarray(value["mapped_continuous"])
                        latent = np.asarray(value["raw_continuous"])
                        low, high = np.asarray(value["continuous_low"]), np.asarray(value["continuous_high"])
                        # Legacy non-time mapping was float32; decoded times are float64.
                        np.testing.assert_allclose(mapped, low + latent * (high - low), atol=2e-6, rtol=2e-6)
    assert traces, "no retained physical traces (check disk logging guard)"
    return {"rounds": len(records), "workers": count, "env_steps": int(records[-1]["env_steps"]),
            "retained_traces": len(traces), "status": "passed"}


if __name__ == "__main__":
    print(json.dumps(verify(sys.argv[1]), ensure_ascii=False))
