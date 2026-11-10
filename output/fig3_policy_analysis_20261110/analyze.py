"""Audit Fig3 CSVs and saved policies without training or calling a solver.

Run from any directory with the project's ML2024_Task1 Python environment.
The policy probe uses one reconstructed initial observation, not a rollout
evaluation or an estimate of the frequency over the original training states.
"""
import csv
import hashlib
import json
import platform
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
OUTER = ROOT / "src/outer"
sys.path.insert(0, str(OUTER))
from algorithms.MHBPPO.MHBPPO_agent import MultiHeadActor as BetaActor
from algorithms.MHGPPO.MHGPPO_agent import MultiHeadActor as GaussianActor
from configs.base_params import get_base_args
from configs.madrl_args import get_madrl_args
from environment.my_env import MyEnv


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def mean(rows, field):
    return statistics.mean(float(row[field]) for row in rows)


def phase_time(row):
    if "elapsed_seconds" in row:
        return float(row["elapsed_seconds"])
    return sum(float(row[key]) for key in
               ("collection_wait_seconds", "update_seconds", "logging_seconds"))


def weighted_energy(rows, key):
    prefix = "energy_finite_candidate_" + key
    count = sum(int(row[prefix + "_count"]) for row in rows)
    return {"mean": sum(float(row[prefix + "_sum"]) for row in rows) / count,
            "count": count}


def summarize(algorithm):
    figure = ROOT / "src/fig/outer/Fig3" / (algorithm + ".csv")
    run = OUTER / "ckpt" / algorithm / "BEST_CKPT"
    source = run / "training_updates.csv"
    rows, updates = read_csv(figure), read_csv(source)
    assert len(rows) == len(updates) == 3000
    assert [int(row["episode"]) for row in rows] == list(range(3000))
    iteration_key = "update" if algorithm == "MHBPPO" else "iteration"
    assert [int(row[iteration_key]) for row in updates] == list(range(1, 3001))
    for row, update in zip(rows, updates):
        assert float(row["reward"]) == float(update["reward"])
        assert float(row["completion_rate"]) == float(update["completion_rate"])
    assert int(updates[-1]["env_steps"]) == 960000
    rewards = [float(row["reward"]) for row in rows]
    rolling = [statistics.mean(rewards[i-99:i+1]) for i in range(99, 3000)]
    thresholds = {}
    for target in (.90, .95, .98, .99):
        first = next((i + 100 for i, value in enumerate(rolling) if value >= target), None)
        last_below = max((i + 100 for i, value in enumerate(rolling) if value < target), default=99)
        sustained = last_below + 1 if last_below < 3000 else None
        thresholds[str(target)] = {"first_100_round_mean": first,
                                   "stays_above_until_end": sustained,
                                   "first_env_steps": first * 320 if first else None}
    windows = []
    for start, end in ((1, 100), (101, 500), (501, 1000), (1001, 1500),
                       (1501, 2000), (2001, 2500), (2501, 3000)):
        window = updates[start-1:end]
        windows.append({"start": start, "end": end,
            "reward_mean": mean(window, "reward"),
            "reward_population_sd": statistics.pstdev(float(row["reward"]) for row in window),
            "completion_percent": mean(window, "completion_rate"),
            "recorded_phase_seconds_mean": statistics.mean(map(phase_time, window)),
            "weighted_energy_j": weighted_energy(window, "weighted_total_energy_j")["mean"],
            "flight_energy_j": weighted_energy(window, "uav_flight_energy_j")["mean"]})
    tail = updates[-500:]
    early_windows = []
    for start, end in ((1, 20), (21, 40), (41, 60), (61, 80), (81, 100), (101, 200)):
        window = updates[start-1:end]
        early_windows.append({"start": start, "end": end,
            "reward_mean": mean(window, "reward"),
            "completion_percent": mean(window, "completion_rate"),
            "updated_actor_expected_duplicate_matching_mean": mean(window, "actor_expected_duplicate_matching"),
            "weighted_energy_j": weighted_energy(window, "weighted_total_energy_j")["mean"]})
    diagnostics = {"strict_feasible_count": sum(int(row["energy_strict_feasible_count"]) for row in tail),
        "failed_slots": round(sum((100-float(row["completion_rate"]))*320/100 for row in tail)),
        "finite_weighted_energy": weighted_energy(tail, "weighted_total_energy_j"),
        "finite_physical_energy": weighted_energy(tail, "physical_total_energy_j"),
        "finite_flight_energy": weighted_energy(tail, "uav_flight_energy_j"),
        "recorded_solver_seconds_sum_per_round_mean": mean(tail, "solver_seconds")}
    if algorithm == "MHBPPO":
        diagnostics.update({"duplicate_slots": round(sum(float(row["spectrum_penalty_rate"])*3.2 for row in tail)),
                            "unknown_slots": sum(int(row["solver_unknown_count"]) for row in tail)})
    else:
        counts = {}
        for row in tail:
            for key, value in json.loads(row["solver_status_counts"]).items():
                counts[key] = counts.get(key, 0) + value
        diagnostics["solver_status_counts"] = counts
    return {"input_hashes": {str(p.relative_to(ROOT)): sha(p) for p in (figure, source)},
            "rows": len(rows), "env_steps": int(updates[-1]["env_steps"]),
            "windows": windows, "early_windows": early_windows,
            "thresholds": thresholds, "tail_diagnostics": diagnostics,
            "recorded_phase_seconds_total": sum(map(phase_time, updates))}


def probe_policies(gconfig):
    args = get_madrl_args([])
    for key, value in gconfig["hyperparameters"].items():
        setattr(args, key, value)
    args.analytic_receive_beam_only = True
    env = MyEnv(get_base_args(seed=42, target_layout=args.target_layout), args, seed_offset=0)
    raw = np.asarray(env.reset(episode_idx=0)["bs"], dtype=np.float32)
    reconstructed_scene = env.get_exogenous_scene_signature()
    original_scene = gconfig["initial_exogenous_scene"]
    trajectory = np.load(OUTER / "ckpt/MHGPPO/BEST_CKPT/iter_3000/uav_trajectory_worker_0.npy")
    probes = {"scope": "one reconstructed initial observation; not a rollout performance estimate",
        "changed_component_hashes": [key for key, value in reconstructed_scene["component_sha256"].items()
                                     if value != original_scene["component_sha256"][key]],
        "initial_uav_max_absolute_difference_from_saved_trajectory_m":
            float(np.max(np.abs(env.init_uavs_pos - trajectory[0]))), "policies": {}}
    # At the same weights/input, the continuous distribution does not directly
    # alter categorical logits or the analytic duplicate-matching penalty.
    fresh = []
    for actor_class, distribution in ((BetaActor, "beta"), (GaussianActor, "gaussian")):
        torch.manual_seed(42)
        actor = actor_class(gconfig["state_dim"], 128, gconfig["continuous_action_splits"],
                            gconfig["discrete_action_dims"], continuous_dist_type=distribution)
        with torch.no_grad():
            _, categorical = actor._get_dists(torch.tensor(raw).unsqueeze(0))
            fresh.append(torch.cat([dist.probs for dist in categorical], dim=1))
    probes["fresh_seed42_categorical_probabilities_exactly_equal"] = bool(torch.equal(*fresh))
    probes["fresh_seed42_categorical_probability_max_difference"] = float((fresh[0]-fresh[1]).abs().max())
    for algorithm, actor_class, distribution in (("MHBPPO", BetaActor, "beta"),
                                                ("MHGPPO", GaussianActor, "gaussian")):
        probes["policies"][algorithm] = {}
        for iteration in (1000, 2000, 3000):
            folder = OUTER / "ckpt" / algorithm / "BEST_CKPT" / f"iter_{iteration}"
            actor_file = folder / (algorithm.lower() + "_actor.pth")
            norm_file = folder / "state_normalization.npz"
            actor = actor_class(gconfig["state_dim"], 128, gconfig["continuous_action_splits"],
                                gconfig["discrete_action_dims"], continuous_dist_type=distribution)
            actor.load_state_dict(torch.load(actor_file, map_location="cpu", weights_only=True))
            actor.eval()
            with np.load(norm_file, allow_pickle=False) as normalizer:
                assert str(normalizer["actor_sha256"].item()) == sha(actor_file)
                state = torch.tensor((raw-normalizer["mean"])/(normalizer["std"]+float(normalizer["epsilon"])),
                                     dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                distributions, _ = actor._get_dists(state)
                distance_dist = distributions["uav_distances"]
                torch.manual_seed(422026+iteration)
                x = distance_dist.sample((50000,))
                x = x if distribution == "beta" else x.sigmoid()
                speeds = 5 + 35*x.numpy()[:, 0, :]
                flight = .65*(.00614*speeds**3+15.976/speeds)
                torch.manual_seed(422026)
                states = state.expand(4096, -1)
                continuous, discrete, _, old_cont_lp, _ = actor.sample(states)
                evaluation = actor.evaluate_actions(states, continuous, discrete)
                difference = (evaluation[2]-old_cont_lp).squeeze()
                ratio = difference.exp()
                boundary = (continuous < 1e-6) | (continuous > 1-1e-6)
                result = {"actor_sha256": sha(actor_file), "normalization_sha256": sha(norm_file),
                    "actor_parameters": sum(p.numel() for p in actor.parameters()),
                    "speed_mean_by_uav": speeds.mean(axis=0).tolist(),
                    "speed_p90_by_uav": np.quantile(speeds, .9, axis=0).tolist(),
                    "flight_energy_mc_mean_j": float(flight.sum(axis=1).mean()),
                    "mc_distance_samples": 50000, "same_policy_samples": 4096,
                    "elements_outside_evaluation_clamp": float(boundary.float().mean()),
                    "actions_with_any_element_outside_evaluation_clamp": float(boundary.any(dim=-1).float().mean()),
                    "same_policy_cont_ratio_outside_0p8_1p2": float(((ratio < .8)|(ratio > 1.2)).float().mean()),
                    "same_policy_cont_logprob_absolute_difference_quantiles":
                        np.quantile(difference.abs().numpy(), [.5, .9, .99, 1]).tolist()}
                if distribution == "gaussian":
                    result["distance_mu"] = distance_dist.mean.flatten().tolist()
                    result["std_by_head"] = {key: value.stddev.flatten().tolist()
                                             for key, value in distributions.items()}
                else:
                    result["distance_alpha"] = distance_dist.concentration1.flatten().tolist()
                    result["distance_beta"] = distance_dist.concentration0.flatten().tolist()
            probes["policies"][algorithm][str(iteration)] = result
    return probes


def main():
    torch.set_num_threads(1)
    bmeta = json.loads((OUTER / "ckpt/MHBPPO/BEST_CKPT/iter_3000/checkpoint_metadata.json").read_text(encoding="utf8"))
    bconfig = json.loads((OUTER / "ckpt/MHBPPO/BEST_CKPT/hyperparameters.json").read_text(encoding="utf8"))
    gconfig = json.loads((OUTER / "ckpt/MHGPPO/BEST_CKPT/configuration.json").read_text(encoding="utf8"))
    assert bmeta["shared_exogenous_scene"] == gconfig["initial_exogenous_scene"]
    data = {"runtime": {"python": sys.version, "torch": torch.__version__,
                        "numpy": np.__version__, "platform": platform.platform()},
        "recorded_scenes_identical": True,
        "hyperparameter_differences": {key: [value, gconfig["hyperparameters"].get(key)]
            for key, value in bconfig["hyperparameters"].items()
            if value != gconfig["hyperparameters"].get(key)},
        "algorithms": {name: summarize(name) for name in ("MHBPPO", "MHGPPO")},
        "source_matches_B_checkpoint": {key: sha(OUTER/key) == value
                                        for key, value in bmeta["source_sha256"].items()},
        "policy_probe": probe_policies(gconfig),
        "flight_power_minimizer_m_per_s": (15.976/(3*.00614))**.25,
        "timing_scope": "CSV measured phases only; timers differ and exclude some aggregation/checkpoints",
        "probe_scope": "current implementation + saved weights/normalization; no solver or training",
        "user_scope": "discrete entropy annealing is not used as the main explanation"}
    relevant_sources = ["algorithms/MHBPPO/MHBPPO_agent.py", "algorithms/MHGPPO/MHGPPO_agent.py",
                        "environment/my_env.py", "reward/my_reward.py"]
    data["current_source_sha256"] = {key: sha(OUTER/key) for key in relevant_sources}
    destination = Path(__file__).with_name("metrics.json")
    destination.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf8")
    print(destination)
    for name, values in data["algorithms"].items():
        print(name, "tail", values["windows"][-1], "thresholds", values["thresholds"])
        print("final ratio probe", data["policy_probe"]["policies"][name]["3000"]["same_policy_cont_ratio_outside_0p8_1p2"])


if __name__ == "__main__":
    main()
