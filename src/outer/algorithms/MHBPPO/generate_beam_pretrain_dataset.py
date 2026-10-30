"""Bounded offline receive-beam data recovery. Never calls reward or a solver.

Run from src/outer with ``python -m algorithms.MHBPPO.generate_beam_pretrain_dataset``.
The source files remain read-only. Imports of the environment are deliberately lazy.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np


OUTER_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = OUTER_ROOT.parents[1]
SOURCE_RUN = "2026-10-15_21-21-28"
DEFAULT_CHECKPOINT = (OUTER_ROOT / "ckpt/MHBPPO" /
                      (SOURCE_RUN + "_mhbppo_seed_42") / "iter_3000")
DEFAULT_LOG_DIRECTORY = OUTER_ROOT / "logs/MHBPPO" / SOURCE_RUN
PHYSICAL_SOURCE_FILES = (
    "configs/base_params.py", "configs/seed_manager.py",
    "environment/my_env.py", "environment/channel_models.py",
    "environment/topology.py", "environment/observation.py", "environment/mapping.py",
    "reward/cccp_bridge.py",
    "../inner/parameters.py", "../inner/target_layout.py", "../inner/environment.py",
)
DATASET_CONTRACT = "matched_receive_beam_offline_v1"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_windows(text, frozen_update=3617):
    windows = []
    for part in str(text).split(","):
        bounds = part.strip().split(":")
        if len(bounds) != 2:
            raise ValueError("windows must be comma-separated inclusive first:last pairs")
        first, last = map(int, bounds)
        if first < 1 or last < first or last > int(frozen_update):
            raise ValueError("window is outside the frozen completed-update prefix")
        windows.append((first, last))
    windows.sort()
    if not windows or any(b[0] <= a[1] for a, b in zip(windows, windows[1:])):
        raise ValueError("windows must be nonempty and disjoint")
    return windows


def read_log_metadata(path):
    with Path(path).open("rb") as stream:
        line = stream.readline()
    metadata = json.loads(line)
    if metadata.get("record_type") != "metadata":
        raise ValueError("log must begin with its metadata record")
    return metadata, len(line), hashlib.sha256(line).hexdigest()


def _next_record(stream, position, end, record_type, statistics):
    """Read the next complete matching line, aligning a random byte position."""
    position = max(0, int(position))
    if position >= end:
        return None
    stream.seek(position)
    if position:
        stream.seek(position - 1)
        previous = stream.read(1)
        stream.seek(position)
        if previous != b"\n":
            skipped = stream.readline(end - position)
            statistics["bytes_read"] += len(skipped)
    while stream.tell() < end:
        start = stream.tell()
        line = stream.readline(end - start)
        statistics["bytes_read"] += len(line)
        if not line.endswith(b"\n"):
            return None  # an incomplete concurrently written tail is never a sample
        record = json.loads(line)
        if record.get("record_type") == record_type:
            return start, stream.tell(), record, line
    return None


def iteration_lower_bound(path, target, *, record_type, start=0, end=None, statistics=None):
    """Monotonic iteration byte bisection; final scan covers a boundary line only.

    Metadata/scene records are skipped. The data stream must be monotonic; selected
    windows independently validate this assumption and all expected identities.
    """
    stats = statistics if statistics is not None else {"bytes_read": 0, "probes": 0}
    stats.setdefault("bytes_read", 0)
    stats.setdefault("probes", 0)
    end = Path(path).stat().st_size if end is None else int(end)
    low, high = int(start), end
    with Path(path).open("rb") as stream:
        while low < high:
            middle = (low + high) // 2
            stats["probes"] += 1
            candidate = _next_record(stream, middle, end, record_type, stats)
            if candidate is None or int(candidate[2]["iteration"]) >= int(target):
                high = middle
            else:
                low = candidate[1]
        candidate = _next_record(stream, low, end, record_type, stats)
        while candidate is not None and int(candidate[2]["iteration"]) < int(target):
            candidate = _next_record(stream, candidate[1], end, record_type, stats)
        return end if candidate is None else candidate[0]


def iter_log_window(path, first, last, *, record_type, end=None, audit=None):
    audit = {} if audit is None else audit
    stats = {"bytes_read": 0, "probes": 0}
    end = Path(path).stat().st_size if end is None else int(end)
    start = iteration_lower_bound(path, first, record_type=record_type, end=end,
                                  statistics=stats)
    stop = iteration_lower_bound(path, last + 1, record_type=record_type, start=start,
                                 end=end, statistics=stats)
    digest, count, previous_iteration = hashlib.sha256(), 0, int(first)
    with Path(path).open("rb") as stream:
        stream.seek(start)
        while stream.tell() < stop:
            offset = stream.tell()
            line = stream.readline(stop - offset)
            stats["bytes_read"] += len(line)
            if not line.endswith(b"\n"):
                raise ValueError("incomplete line inside a frozen selected window")
            digest.update(line)
            record = json.loads(line)
            if record.get("record_type") != record_type:
                continue
            iteration = int(record["iteration"])
            if not first <= iteration <= last or iteration < previous_iteration:
                raise ValueError("selected log window is not monotonic or was bisected incorrectly")
            previous_iteration = iteration
            count += 1
            yield record
    audit.update(path=str(Path(path).resolve()), first_update=first, last_update=last,
                 start_byte=start, stop_byte=stop, selected_bytes=stop-start,
                 selected_sha256=digest.hexdigest(), records=count, **stats)


def verify_source_fingerprints(metadata):
    expected = metadata.get("source_sha256", {})
    checked = {}
    for name in PHYSICAL_SOURCE_FILES:
        if name not in expected:
            raise ValueError("missing physical source fingerprint: " + name)
        actual = file_sha256(OUTER_ROOT / name)
        if actual != expected[name]:
            raise ValueError("physical source fingerprint mismatch: " + name)
        checked[name] = actual
    return checked


def restore_parameters(metadata, *, seed=42, scene_seed_offset=0):
    """Restore inner native fields *and* adapter extras, rejecting silent defaults."""
    from configs.base_params import get_base_args, _LEGACY_TO_INNER

    required = ("base_args", "inner_parameter_source", "madrl_args", "shared_exogenous_scene")
    if any(not isinstance(metadata.get(key), dict) for key in required):
        raise ValueError("metadata is missing full parameters or shared scene")
    native, legacy = metadata["inner_parameter_source"], metadata["base_args"]
    for key in ("I", "J", "K", "N", "seed", "target_layout", "rho_ref", "sigma_2",
                "eps_sinr", "P_max_uav", "d_over_lambda", "xi_0", "bs_pos"):
        if key not in native:
            raise ValueError("missing native physical parameter: " + key)
    if int(seed) != int(native["seed"]) or int(seed) != int(metadata.get("seed", -1)):
        raise ValueError("requested seed differs from source metadata")
    if int(scene_seed_offset) != int(metadata.get("scene_seed_offset", -1)):
        raise ValueError("requested scene offset differs from source metadata")
    base = get_base_args(seed=int(seed))
    for name, value in native.items():
        setattr(base._params, name, copy.deepcopy(value))
    for name, value in legacy.items():
        native_name = _LEGACY_TO_INNER.get(name)
        if native_name in native and not np.array_equal(np.asarray(value), np.asarray(native[native_name])):
            raise ValueError("conflicting native/adapter parameter: " + name)
        setattr(base, name, copy.deepcopy(value))
    madrl = SimpleNamespace(**copy.deepcopy(metadata["madrl_args"]))
    if getattr(madrl, "receive_beam_phase_mode", "absolute") != "absolute":
        raise ValueError("offline beam dataset reconstruction does not support phase_residual mapping")
    if int(getattr(madrl, "total_time_slots", -1)) != 40:
        raise ValueError("this dataset contract requires complete 40-slot episodes")
    for name in ("randomize_layout_per_episode", "randomize_cu_traj_per_episode",
                 "randomize_nlos_per_episode"):
        if not hasattr(madrl, name) or bool(getattr(madrl, name)):
            raise ValueError("fixed-scene dataset requires explicit false " + name)
    if float(getattr(madrl, "structured_beam_alpha", -1)) != 1.0:
        raise ValueError("source must execute the actor's full receive beam")
    if (base.uavs_num, base.cus_num, base.antenna_nums) != (4, 10, 10):
        raise ValueError("this dataset contract requires I=4,J=10,N=10")
    return base, madrl


def create_reconstruction_env(metadata, *, seed=42, scene_seed_offset=0, expected_scene=None):
    from environment.my_env import MyEnv

    base, madrl = restore_parameters(metadata, seed=seed, scene_seed_offset=scene_seed_offset)
    env = MyEnv(base, madrl, seed_offset=int(scene_seed_offset))
    env.reset(0)
    expected = metadata["shared_exogenous_scene"] if expected_scene is None else expected_scene
    actual = env.get_exogenous_scene_signature()
    if actual != expected:
        raise ValueError("regenerated scene/config signature differs from source")
    # A fail-fast guard: future accidental step() cannot launch an inner solve.
    def no_reward(*args, **kwargs):
        raise RuntimeError("offline state reconstruction must never call reward/CCCP")
    env.reward_calculator.reward_compute = no_reward
    return env


def reconstruct_raw_state(env, slot, uavs_pos_before):
    from environment.mapping import build_uav_targets_matched_matrix

    slot = int(slot)
    if not 0 <= slot <= int(env.madrl_args.total_time_slots):
        raise ValueError("invalid reconstruction time slot")
    position = np.asarray(uavs_pos_before, dtype=np.float64)
    if position.shape != (env.base_args.uavs_num, 3) or not np.all(np.isfinite(position)):
        raise ValueError("invalid UAV positions")
    env.t = slot
    env.cur_uavs_pos = position.copy()
    env.cur_cus_pos = env.precomputed_cus_traj[slot].copy()
    env.uavs_targets_matched_matrix = build_uav_targets_matched_matrix(
        env.precomputed_uav_target_schedule[slot], env.base_args.uavs_num,
        env.base_args.targets_num)
    env._refresh_channels()
    raw = np.asarray(env._build_bs_observation(), dtype=np.float32).copy()
    if raw.shape != env.observation_space["bs"].shape or not np.all(np.isfinite(raw)):
        raise ValueError("invalid recovered raw observation")
    return raw


def canonical_matched_teacher(env):
    indices = env._get_current_uav_target_indices()
    delta = env.cur_uavs_pos - env.init_targets_pos[indices]
    angle = np.arctan2(delta[:, 1], delta[:, 0])
    phase = (2j * np.pi * float(env.base_args.frac_d_lambda) *
             np.sin(angle)[:, None] * np.arange(env.base_args.antenna_nums)[None, :])
    return np.exp(phase) / np.sqrt(float(env.base_args.antenna_nums))


def reconstructed_physical_metrics(env, physical_action):
    packed = physical_action["uavs_rec_beam_vectors"]
    g = np.asarray(packed["real"], dtype=np.float64) + 1j*np.asarray(packed["imag"], dtype=np.float64)
    eta = np.asarray(physical_action["uavs_cus_matched_matrix"], dtype=np.float64)
    power = np.asarray(physical_action["cu_off_powers"], dtype=np.float64)
    I, J, N = env.base_args.uavs_num, env.base_args.cus_num, env.base_args.antenna_nums
    if g.shape != (I, N) or eta.shape != (I, J) or power.shape != (J,):
        raise ValueError("physical action shape mismatch")
    if not all(np.all(np.isfinite(x)) for x in (g, eta, power)):
        raise ValueError("nonfinite physical action")
    # The outer mapping normalizes float32 latent coordinates before packing
    # complex128; the bridge then normalizes that logged beam once more in
    # complex128. Reproduce both stages instead of relaxing physical checks.
    if not np.allclose(np.sum(abs(g)**2, axis=1), 1, rtol=0, atol=1e-5):
        raise ValueError("logged receive beam is not unit norm")
    g_norm = np.linalg.norm(g, axis=1, keepdims=True)
    g = g / np.where(g_norm > 0, g_norm, 1.0)
    if not np.all((eta == 0) | (eta == 1)) or not np.all(eta.sum(axis=1) == 1):
        raise ValueError("matching rows must be one-hot")
    if np.any(power < 0) or np.any(power > float(env.base_args.P_max_cu)*(1+1e-7)):
        raise ValueError("illegal logged CU powers")
    h = env.uavs_2_cus_channels
    gamma = np.sum(eta*power[None, :]*abs(np.sum(g[:, None, :]*h, axis=2))**2, axis=1)
    gamma += float(env.base_args.sigma_2)
    indices = env._get_current_uav_target_indices()
    A = env.uavs_2_targets_channels[np.arange(I), indices]
    Ag = np.einsum("inj,in->ij", A.conj(), g)
    maximum = np.sum(abs(Ag)**2, axis=1)
    distance = np.linalg.norm(env.cur_uavs_pos-env.init_targets_pos[indices], axis=1)
    teacher = canonical_matched_teacher(env)
    gain = abs(np.sum(teacher.conj()*g, axis=1))**2
    margin = float(env.base_args.P_max_uav)*maximum/(float(env.base_args.eps_sinr)*gamma)
    return dict(gamma_w=gamma, lambda_max_g=maximum, target_distances_m=distance,
                margin=margin, gain=gain, physical_cu_power=power,
                physical_matching=np.argmax(eta, axis=1), teacher=teacher)


def _relative_error(actual, expected):
    actual, expected = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    if actual.shape != expected.shape or not np.all(np.isfinite(expected)):
        raise ValueError("physical reference shape/nonfinite mismatch")
    return float(np.max(abs(actual-expected)/np.maximum(abs(expected), 1e-300)))


def check_physical_diagnostic(env, physical, diagnostic, *, relative_tolerance=1e-7):
    metrics = reconstructed_physical_metrics(env, physical)
    sensing = diagnostic.get("sensing", {})
    if sensing.get("status") != "checked":
        raise ValueError("source sensing diagnostic was not checked")
    if not np.array_equal(diagnostic.get("target_indices"), env._get_current_uav_target_indices()):
        raise ValueError("source target schedule mismatch")
    if not np.array_equal(physical["uavs_targets_matched_matrix"], env.uavs_targets_matched_matrix):
        raise ValueError("logged target matching matrix mismatch")
    errors = {name: _relative_error(metrics[name], sensing[name]) for name in
              ("gamma_w", "lambda_max_g", "target_distances_m", "margin")}
    reference_gain = np.asarray(sensing["lambda_max_g"])/(float(env.base_args.radar_rcs)*
        float(env.base_args.rho_ref)*env.base_args.antenna_nums**2/
        np.asarray(sensing["target_distances_m"])**4)
    errors["gain"] = _relative_error(metrics["gain"], reference_gain)
    if max(errors.values()) > relative_tolerance:
        raise ValueError("physical reconstruction differs from logged diagnostics: " + str(errors))
    return metrics, errors


def record_key(record):
    return (int(record["iteration"]), int(record["worker_id"]),
            int(record["episode_idx"]), int(record["time_slot"] if "time_slot" in record
                                         else record["step_idx"]))


def assign_round_splits(raw_states, collection_round, *, seed=42):
    raw_states = np.asarray(raw_states, dtype=np.float32)
    rounds = np.asarray(collection_round, dtype=np.int64)
    groups = np.unique(rounds)
    if len(groups) < 5:
        raise ValueError("at least five complete collection rounds are required for 60/20/20 split")
    shuffled = np.random.default_rng(seed).permutation(groups)
    n_train, n_validation = int(len(groups)*0.6), int(len(groups)*0.2)
    group_split = {int(g): (0 if k < n_train else 1 if k < n_train+n_validation else 2)
                   for k, g in enumerate(shuffled)}
    original = np.array([group_split[int(g)] for g in rounds], dtype=np.int8)
    split = original.copy()
    hashes = np.array([hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest()
                       for row in raw_states], dtype="U64")
    seen_train = set(hashes[original == 0])
    seen_validation = set(hashes[original == 1])
    for index, digest in enumerate(hashes):
        if original[index] == 1 and digest in seen_train:
            split[index] = -1
        elif original[index] == 2 and digest in seen_train | seen_validation:
            split[index] = -1
    if any(not np.any(split == k) for k in (0, 1, 2)):
        raise ValueError("duplicate exclusion left an empty split")
    report = dict(seed=int(seed), grouping="complete collection_round, all workers together",
                  fractions=[0.6, 0.2, 0.2],
                  rounds={str(k): sorted(int(g) for g in groups if group_split[int(g)] == k)
                          for k in (0, 1, 2)},
                  original_rows={str(k): int(np.sum(original == k)) for k in (0, 1, 2)},
                  retained_rows={str(k): int(np.sum(split == k)) for k in (0, 1, 2)},
                  excluded_duplicates={str(k): int(np.sum((original == k) & (split == -1)))
                                       for k in (1, 2)},
                  exact_unique_states=int(len(set(hashes))),
                  duplicate_policy="validation excludes train; test excludes train union all validation",
                  holdout_scope="only this beam supervision/model selection; source actor/normalizer may have seen states")
    return split, original, hashes, report


def _validate_action_geometry(env, action, previous_after, expected_slot):
    from environment.mapping import build_unit_norm_rec_beam

    trace, physical = action["environment_action"], action["environment_action"]["physical_action"]
    if int(action["step_idx"]) != expected_slot or int(trace["time_slot"]) != expected_slot:
        raise ValueError("episode slots are incomplete or out of order")
    before = np.asarray(physical["uavs_pos_before"], dtype=np.float64)
    after = np.asarray(physical["uavs_pos_after"], dtype=np.float64)
    reference = env.init_uavs_pos if expected_slot == 0 else previous_after
    if before.shape != (4, 3) or not np.allclose(before, reference, rtol=0, atol=1e-9):
        raise ValueError("logged UAV position chain is broken")
    latent = np.asarray(action["raw_action"]["continuous"], dtype=np.float32)
    if latent.shape != (102,) or not np.all(np.isfinite(latent)):
        raise ValueError("invalid latent action")
    space = env.action_space["bs"]["continuous"]
    mapped = space.low + np.clip(latent, 0, 1)*(space.high-space.low)
    if not np.array_equal(mapped, np.asarray(trace["mapped_continuous"], dtype=np.float32)):
        raise ValueError("logged latent/physical action mapping mismatch")
    angle, distance = mapped[:4], mapped[4:8]
    delta = np.stack([distance*np.cos(angle), distance*np.sin(angle), np.zeros_like(distance)], axis=1)
    if not np.allclose(before+delta, after, rtol=0, atol=1e-9):
        raise ValueError("logged displacement differs from source action mapping")
    if float(trace["structured_beam_alpha"]) != 1.0:
        raise ValueError("mixed/structured source beams are outside this dataset contract")
    beam = build_unit_norm_rec_beam(mapped[22:62], mapped[62:102], 4, 10)
    packed = physical["uavs_rec_beam_vectors"]
    logged_g = np.asarray(packed["real"])+1j*np.asarray(packed["imag"])
    if not np.allclose(beam, logged_g, rtol=1e-10, atol=1e-12):
        raise ValueError("logged receive beam differs from actor mapping")
    if not np.array_equal(mapped[12:22].astype(np.float64), physical["cu_off_powers"]):
        raise ValueError("logged CU power differs from actor mapping")
    return physical, after.copy()


def build_dataset(args):
    from algorithms.MHBPPO.checkpoint import load_state_normalization

    windows = parse_windows(args.windows, args.frozen_update)
    checkpoint = Path(args.source_checkpoint).resolve()
    if checkpoint.is_file():
        checkpoint = checkpoint.parent
    output = Path(args.output).resolve()
    manifest_path = output.with_suffix(".json")
    if output.exists() or manifest_path.exists():
        raise FileExistsError("refusing to overwrite dataset or manifest")
    checkpoint_meta = json.loads((checkpoint/"checkpoint_metadata.json").read_text(encoding="utf-8"))
    if checkpoint_meta.get("iteration") != 3000 or checkpoint_meta.get("seed") != 42:
        raise ValueError("U48 source checkpoint must be seed42 iter3000")
    norm = load_state_normalization(checkpoint/"state_normalization.npz", expected_dim=1017,
        expected_iteration=3000, actor_path=checkpoint/"mhbppo_actor.pth")
    directory = Path(args.log_directory).resolve()
    action_path, diagnostic_path = directory/"mhbppo_train_actions.jsonl", directory/"mhbppo_train_env.jsonl"
    metadata, _, metadata_hash = read_log_metadata(diagnostic_path)
    action_metadata, _, action_metadata_hash = read_log_metadata(action_path)
    if any(m.get("log_compaction_contract") for m in (metadata, action_metadata)):
        raise ValueError("exact offline dataset reconstruction requires --log_detail_level full logs")
    for key in ("base_args", "inner_parameter_source", "madrl_args", "shared_exogenous_scene", "source_sha256"):
        if metadata.get(key) != action_metadata.get(key):
            raise ValueError("action/diagnostic source metadata mismatch: " + key)
    if checkpoint_meta.get("shared_exogenous_scene") != metadata["shared_exogenous_scene"]:
        raise ValueError("checkpoint/log scene mismatch")
    for name in PHYSICAL_SOURCE_FILES:
        if checkpoint_meta.get("source_sha256", {}).get(name) != metadata["source_sha256"].get(name):
            raise ValueError("checkpoint/log physical source mismatch: " + name)
    source_fingerprints = verify_source_fingerprints(metadata)
    env = create_reconstruction_env(metadata, seed=args.seed, scene_seed_offset=args.scene_seed_offset)
    num_envs = int(metadata["num_envs"])
    if num_envs != 4:
        raise ValueError("U48 source must contain four synchronous environments")
    limits = {str(p): p.stat().st_size for p in (action_path, diagnostic_path)}
    fields = {k: [] for k in ("raw_states", "teacher_g_real", "teacher_g_imag",
        "uav_target_distances", "physical_cu_power", "physical_matching", "episode_idx",
        "collection_round", "worker_id", "time_slot", "source_policy_version")}
    maximum_errors, audits, episode_keys = {}, [], set()
    sentinel = object()
    for first, last in windows:
        action_audit, diagnostic_audit = {}, {}
        actions = iter_log_window(action_path, first, last, record_type="action",
                                  end=limits[str(action_path)], audit=action_audit)
        diagnostics = iter_log_window(diagnostic_path, first, last, record_type="env_diagnostic",
                                      end=limits[str(diagnostic_path)], audit=diagnostic_audit)
        previous_group, previous_after, expected_slot, count = None, None, 0, 0
        while True:
            action, diagnostic = next(actions, sentinel), next(diagnostics, sentinel)
            if action is sentinel or diagnostic is sentinel:
                if action is not sentinel or diagnostic is not sentinel:
                    raise ValueError("action/diagnostic window lengths differ")
                break
            key = record_key(action)
            if key != record_key(diagnostic):
                raise ValueError("action/diagnostic identities are misaligned")
            for name in ("actor_signature", "norm_signature", "collection_round", "scene_signature"):
                if action.get(name) is None or action.get(name) != diagnostic.get(name):
                    raise ValueError("action/diagnostic identity signature mismatch: " + name)
            iteration, worker, episode, slot = key
            group = (iteration, worker, episode)
            if group != previous_group:
                if previous_group is not None and expected_slot != 40:
                    raise ValueError("incomplete episode in selected window")
                if group in episode_keys:
                    raise ValueError("duplicate/interleaved episode")
                episode_keys.add(group)
                expected_slot, previous_after, previous_group = 0, None, group
                env.reset(episode)
            if not 0 <= worker < num_envs:
                raise ValueError("worker ID outside source worker count")
            for record in (action, diagnostic):
                if int(record["collection_round"]) != iteration-1 or int(record["weights_version"]) != iteration-1:
                    raise ValueError("source synchronous version/round contract differs")
                if record["scene_signature"] != metadata["shared_exogenous_scene"]["signature"]:
                    raise ValueError("per-record scene signature mismatch")
                if int(record["scene_seed_offset"]) != int(args.scene_seed_offset):
                    raise ValueError("per-record scene offset mismatch")
            physical, previous_after = _validate_action_geometry(env, action, previous_after, expected_slot)
            raw = reconstruct_raw_state(env, slot, physical["uavs_pos_before"])
            metrics, errors = check_physical_diagnostic(env, physical, diagnostic)
            for name, error in errors.items():
                maximum_errors[name] = max(maximum_errors.get(name, 0), error)
            fields["raw_states"].append(raw)
            fields["teacher_g_real"].append(metrics["teacher"].real)
            fields["teacher_g_imag"].append(metrics["teacher"].imag)
            for name in ("uav_target_distances", "physical_cu_power", "physical_matching"):
                fields[name].append(metrics["target_distances_m"] if name == "uav_target_distances" else metrics[name])
            for name, value in (("episode_idx", episode), ("collection_round", iteration-1),
                ("worker_id", worker), ("time_slot", slot), ("source_policy_version", iteration-1)):
                fields[name].append(value)
            expected_slot += 1
            count += 1
        if expected_slot != 40 or count != (last-first+1)*num_envs*40:
            raise ValueError("selected update window lacks complete source episodes")
        expected_groups = {(r, w) for r in range(first, last+1) for w in range(num_envs)}
        actual_groups = {(r, w) for r, w, _ in episode_keys if first <= r <= last}
        if actual_groups != expected_groups:
            raise ValueError("selected source round/worker coverage differs")
        audits.append(dict(actions=action_audit, diagnostics=diagnostic_audit))
    payload = {name: np.asarray(values, dtype=(np.float32 if name == "raw_states" else
                np.float64 if name in ("teacher_g_real", "teacher_g_imag", "uav_target_distances", "physical_cu_power")
                else np.int64)) for name, values in fields.items()}
    split, original, hashes, split_report = assign_round_splits(payload["raw_states"],
        payload["collection_round"], seed=args.seed)
    payload.update(split=split, split_original=original, raw_state_sha256=hashes)
    source_files = {name: file_sha256(checkpoint/name) for name in
                    ("mhbppo_actor.pth", "mhbppo_critic.pth", "state_normalization.npz", "checkpoint_metadata.json")}
    manifest = dict(contract=DATASET_CONTRACT, dataset_path=str(output),
        rows=len(payload["raw_states"]), episodes=len(episode_keys), windows=windows,
        frozen_completed_update=int(args.frozen_update), seed=int(args.seed),
        scene_seed_offset=int(args.scene_seed_offset), solver_calls=0,
        channel_time="before_move", teacher="canonical steering a/sqrt(N), first component positive real",
        teacher_latent_scale="not applied here; trainer owns rho=0.9",
        source_checkpoint=dict(directory=str(checkpoint), iteration=3000, file_sha256=source_files,
            actor_sha256=source_files["mhbppo_actor.pth"], normalization_sha256=source_files["state_normalization.npz"],
            state_dim=int(norm.mean.size), count=float(norm.count)),
        metadata=metadata, base_args=metadata["base_args"], madrl_args=metadata["madrl_args"],
        inner_parameter_source=metadata["inner_parameter_source"], scene=metadata["shared_exogenous_scene"],
        source=dict(log_directory=str(directory), metadata_sha256=metadata_hash,
            action_metadata_sha256=action_metadata_hash, observed_log_sizes=limits, windows=audits),
        physical_check=dict(passed=True, records=len(payload["raw_states"]), relative_tolerance=1e-7,
            maximum_relative_errors=maximum_errors,
            scope="logged original actions only; necessary sensing metrics, no optimal verification"),
        source_fingerprints=source_fingerprints, split=split_report,
        holdout_scope=split_report["holdout_scope"], generator_sha256=file_sha256(__file__))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary, temporary_manifest = output.with_name(output.name+".tmp"), manifest_path.with_name(manifest_path.name+".tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **payload)
        manifest["dataset_sha256"] = file_sha256(temporary)
        temporary_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        os.replace(temporary, output)
        os.replace(temporary_manifest, manifest_path)
    finally:
        for path in (temporary, temporary_manifest):
            if path.exists():
                path.unlink()
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--log_directory", default=str(DEFAULT_LOG_DIRECTORY))
    parser.add_argument("--windows", default="1001:1025,2001:2025")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scene_seed_offset", type=int, default=0)
    parser.add_argument("--frozen_update", type=int, default=3617)
    args = parser.parse_args(argv)
    manifest = build_dataset(args)
    print(json.dumps({key: manifest[key] for key in
                     ("dataset_path", "rows", "episodes", "physical_check", "split")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
