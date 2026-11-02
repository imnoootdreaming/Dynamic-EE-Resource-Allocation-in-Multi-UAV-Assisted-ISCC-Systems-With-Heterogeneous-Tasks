"""Check recorded MHGPPO sample versions and evaluation checkpoints."""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

OUTER = Path(__file__).resolve().parents[2]
if str(OUTER) not in sys.path:
    sys.path.insert(0, str(OUTER))
from algorithms.MHGPPO.checkpoint import load_normalization, normalize_state
from algorithms.MHGPPO.MHGPPO_agent import MultiHeadActor
from algorithms.MHBPPO.rollout_batch import actor_state_signature


def verify(run_dir):
    root = Path(run_dir)
    config = json.loads((root / "configuration.json").read_text(encoding="utf-8"))
    records = list(csv.DictReader((root / "training_updates.csv").open(encoding="utf-8", newline="")))
    episodes = [json.loads(line) for line in (root / "episode_metadata.jsonl").read_text(encoding="utf-8").splitlines()]
    count = config["hyperparameters"]["num_samplers"]
    assert len(episodes) == len(records) * count
    for idx, record in enumerate(records, start=1):
        assert int(record["iteration"]) == idx and int(record["policy_version"]) == idx - 1
        batch = [ep for ep in episodes if ep["iteration"] == idx]
        assert sorted(ep["worker_id"] for ep in batch) == list(range(count))
        assert all(ep["policy_version"] == idx - 1 for ep in batch)
        assert len({ep["actor_signature"] for ep in batch}) == 1
        if idx > 1:
            previous = root / f"iter_{idx - 1}" / "mhgppo_actor.pth"
            if previous.exists():
                state = torch.load(previous, map_location="cpu", weights_only=True)
                assert actor_state_signature(state) == batch[0]["actor_signature"]
    final = root / f"iter_{len(records)}"
    stats = load_normalization(final, expected_dim=config["state_dim"])
    assert stats["iteration"] == len(records)
    actor = MultiHeadActor(config["state_dim"], config["hyperparameters"]["hidden_dim"],
                           config["continuous_action_splits"], config["discrete_action_dims"],
                           continuous_dist_type="gaussian")
    actor.load_state_dict(torch.load(final / "mhgppo_actor.pth", map_location="cpu", weights_only=True))
    state = normalize_state(np.zeros(config["state_dim"], dtype=np.float32), stats)
    with torch.no_grad():
        action = actor.sample(torch.as_tensor(state, dtype=torch.float32).unsqueeze(0))
    assert all(torch.isfinite(part).all() for part in action)
    for worker_id in range(count):
        trajectory = np.load(final / f"uav_trajectory_worker_{worker_id}.npy")
        assert trajectory.shape[0] > 1
        assert (final / f"uav_trajectory_worker_{worker_id}.png").is_file()
    assert (final / "training_updates.csv").is_file()
    return {"rounds": len(records), "workers": count, "env_steps": int(records[-1]["env_steps"]),
            "normalization_iteration": stats["iteration"], "status": "passed"}


if __name__ == "__main__":
    print(json.dumps(verify(sys.argv[1]), ensure_ascii=False))
