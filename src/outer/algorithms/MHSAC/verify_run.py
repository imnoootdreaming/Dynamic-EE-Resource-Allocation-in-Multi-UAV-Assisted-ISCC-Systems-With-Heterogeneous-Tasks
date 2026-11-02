"""Verify MHSAC accounting, step versions and evaluation checkpoints."""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

OUTER = Path(__file__).resolve().parents[2]
if str(OUTER) not in sys.path:
    sys.path.insert(0, str(OUTER))
from algorithms.MHSAC.checkpoint import load_normalization, normalize_state
from algorithms.MHSAC.MHSAC_agent import HybridActor
from algorithms.MHBPPO.rollout_batch import actor_state_signature


def verify(run_dir):
    root = Path(run_dir)
    config = json.loads((root / "configuration.json").read_text(encoding="utf-8"))
    with (root / "training_updates.csv").open(encoding="utf-8", newline="") as stream:
        records = list(csv.DictReader(stream))
    episodes = [json.loads(line) for line in (root / "episode_metadata.jsonl").read_text(encoding="utf-8").splitlines()]
    workers = config["hyperparameters"]["num_samplers"]
    warmup = max(config["sac_hyperparameters"]["minimal_size"], config["hyperparameters"]["batch_size"])
    update_rate = config["sac_hyperparameters"]["updates_per_step"]
    assert len(episodes) == len(records) * workers
    for idx, record in enumerate(records, start=1):
        steps = int(record["env_steps"])
        assert int(record["iteration"]) == idx
        assert int(record["sac_updates"]) == max(0, steps - warmup + 1) * update_rate
        assert int(record["replay_size"]) == min(steps, config["sac_hyperparameters"]["buffer_size"])
        batch = [ep for ep in episodes if ep["iteration"] == idx]
        assert sorted(ep["worker_id"] for ep in batch) == list(range(workers))
        for ep in batch:
            assert ep["steps"] == len(ep["policy_versions"]) == len(ep["policy_lags"]) == len(ep["policy_signatures"])
        for slot in range(batch[0]["steps"]):
            assert len({ep["policy_versions"][slot] for ep in batch}) == 1
            assert len({ep["policy_signatures"][slot] for ep in batch}) == 1
        if idx > 1:
            previous = root / f"iter_{idx - 1}" / "mhsac_actor.pth"
            if previous.exists():
                state = torch.load(previous, map_location="cpu", weights_only=True)
                assert actor_state_signature(state) == batch[0]["policy_signatures"][0]
    final = root / f"iter_{len(records)}"
    stats = load_normalization(final, expected_dim=config["state_dim"])
    assert stats["iteration"] == len(records)
    with np.load(final / "state_normalization.npz", allow_pickle=False) as saved:
        assert int(saved["count"]) == int(records[-1]["env_steps"]) + len(episodes)
    actor = HybridActor(config["state_dim"], config["hyperparameters"]["hidden_dim"],
                        config["continuous_action_splits"], config["discrete_action_dims"])
    actor.load_state_dict(torch.load(final / "mhsac_actor.pth", map_location="cpu", weights_only=True))
    state = normalize_state(np.zeros(config["state_dim"], dtype=np.float32), stats)
    with torch.no_grad():
        action = actor.sample(torch.as_tensor(state, dtype=torch.float32).unsqueeze(0))
    assert action["joint_probs"].shape[-1] == config["joint_action_dim"]
    assert all(torch.isfinite(part).all() for part in action.values())
    for name in ("critic1", "critic2", "critic1_target", "critic2_target"):
        assert (final / f"mhsac_{name}.pth").is_file()
    with np.load(final / "temperatures.npz", allow_pickle=False) as temps:
        assert np.isfinite(temps["log_alpha_continuous"]) and np.isfinite(temps["log_alpha_discrete"])
    for worker_id in range(workers):
        assert np.load(final / f"uav_trajectory_worker_{worker_id}.npy").shape[0] > 1
        assert (final / f"uav_trajectory_worker_{worker_id}.png").is_file()
    return {"rounds": len(records), "workers": workers, "env_steps": int(records[-1]["env_steps"]),
            "sac_updates": int(records[-1]["sac_updates"]), "joint_actions": config["joint_action_dim"],
            "status": "passed"}


if __name__ == "__main__":
    print(json.dumps(verify(sys.argv[1]), ensure_ascii=False))
