"""Standalone beam-head pretraining; never starts PPO or an inner solver.

The source checkpoint is read-only. Outputs are two initialization bundles
for a subsequent, separately authorized paired PPO experiment, not resumes.
"""
import argparse
import csv
import json
from pathlib import Path
import shutil
import sys
import time
import traceback


OUTER_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = OUTER_ROOT.parents[1]
APPROVED_SOURCE = (OUTER_ROOT / "ckpt" / "MHBPPO" /
    "2026-10-15_21-21-28_mhbppo_seed_42" / "iter_3000")


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", "--output-dir", dest="output", required=True, type=Path)
    parser.add_argument("--source-checkpoint", "--source_checkpoint", "--source",
                        dest="source_checkpoint", required=True, type=Path)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--rho", type=float, default=.9)
    parser.add_argument("--kappa", type=float, default=64.)
    parser.add_argument("--validation-draws", dest="validation_draws", type=int, default=8)
    parser.add_argument("--min-delta", dest="min_delta", type=float, default=1e-5)
    return parser


def _within(path, parent):
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _check_output(args):
    source, output = args.source_checkpoint.resolve(), args.output.resolve()
    if source != APPROVED_SOURCE.resolve():
        raise ValueError(f"U48 source must be the approved iter_3000: {APPROVED_SOURCE}")
    if not source.is_dir():
        raise FileNotFoundError(source)
    if _within(output, REPOSITORY_ROOT / "src") or _within(output, source):
        raise ValueError("output must be outside production src and the read-only source checkpoint")
    if output.exists():
        raise FileExistsError("refusing to overwrite an existing output directory")
    if args.threads <= 0:
        raise ValueError("--threads must be positive; this offline entry is CPU-only")
    if args.seed < 0:
        raise ValueError("--seed must be nonnegative")
    return source, output


def _validate_source_binding(manifest, arrays, source, normalizer, sourceinfo, sha256):
    binding = manifest.get("source_checkpoint")
    if not isinstance(binding, dict):
        raise ValueError("dataset manifest requires structured source_checkpoint binding")
    if Path(binding.get("directory", "")).resolve() != source:
        raise ValueError("dataset was built for a different source checkpoint")
    if int(binding.get("iteration", -1)) != 3000 or sourceinfo["source_iteration"] != 3000:
        raise ValueError("U48 requires source iteration 3000")
    expected_files = binding.get("file_sha256", {})
    required = ("mhbppo_actor.pth", "mhbppo_critic.pth", "state_normalization.npz", "checkpoint_metadata.json")
    for name in required:
        if expected_files.get(name) != sha256(source / name):
            raise ValueError(f"dataset source binding mismatch: {name}")
    if binding.get("actor_sha256") != sourceinfo["actor_file_sha256"]:
        raise ValueError("dataset actor SHA binding mismatch")
    if binding.get("normalization_sha256") != expected_files["state_normalization.npz"]:
        raise ValueError("dataset normalizer SHA binding mismatch")
    if int(binding.get("state_dim", -1)) != normalizer.mean.size or float(binding.get("count", -1)) != normalizer.count:
        raise ValueError("dataset normalizer dimension/count mismatch")
    if manifest.get("dataset_sha256") != manifest["_dataset_sha256"]:
        raise ValueError("dataset SHA differs from its manifest")
    if manifest.get("seed") != 42 or manifest.get("scene_seed_offset") != 0:
        raise ValueError("U48 requires the fixed 42/0 scene dataset")
    metadata = sourceinfo["source_metadata"]
    if metadata.get("seed") != 42 or metadata.get("scene_seed_offset") != 0 or metadata.get("num_envs") != 4:
        raise ValueError("source metadata is not the approved 42/0 four-env run")
    source_scene = metadata.get("shared_exogenous_scene", {}).get("signature")
    if not source_scene or manifest.get("scene", {}).get("signature") != source_scene:
        raise ValueError("dataset and source checkpoint scene signatures disagree")
    if arrays["raw_states"].shape[1] != 1017 or arrays["teacher_g_real"].shape[1:] != (4, 10):
        raise ValueError("U48 source expects state1017/UAV4/antenna10")
    if arrays["physical_cu_power"].shape[1] != 10:
        raise ValueError("U48 source expects ten CUs")
    if not manifest.get("physical_check", {}).get("passed") or manifest.get("solver_calls") != 0:
        raise ValueError("dataset must pass reconstruction checks without solver calls")


def _save_actor(path, state, torch):
    temporary = path.with_name(path.name + ".tmp")
    try:
        torch.save(state, temporary)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_initialization(destination, actor, actor_state, normalizer, source,
                          sourceinfo, config, manifest, training, *, original,
                          torch, save_norm, actor_signature, norm_signature, sha256, write_json):
    destination.mkdir()
    actor_path, critic_path = destination / "mhbppo_actor.pth", destination / "mhbppo_critic.pth"
    if original:
        shutil.copy2(source / "mhbppo_actor.pth", actor_path)
    else:
        _save_actor(actor_path, actor_state, torch)
    shutil.copy2(source / "mhbppo_critic.pth", critic_path)
    save_norm(destination / "state_normalization.npz", normalizer, iteration=0, actor_path=actor_path)
    metadata = {"format_version": 1, "stage": "initialization", "iteration": 0,
        "weights_version": 0, "completed_ppo_updates": 0, "is_resume": False,
        "initialization_mode": "original" if original else "beam_pretrained",
        "source_checkpoint": str(source), "source_iteration": sourceinfo["source_iteration"],
        "source_actor_signature": sourceinfo["actor_signature"],
        "source_norm_signature": sourceinfo["norm_signature"],
        "source_file_sha256": sourceinfo["source_file_sha256"],
        "architecture": sourceinfo["architecture"], "pretrain_config": config,
        "dataset_sha256": manifest["_dataset_sha256"],
        "dataset_manifest_sha256": manifest["_manifest_sha256"],
        "best_epoch": 0 if original else training["best_epoch"],
        "test_used_for_selection": False, "critic_sha256": sha256(critic_path),
        "state_normalization_file": "state_normalization.npz",
        "normalization_count": normalizer.count,
        "norm_signature": norm_signature(normalizer.mean, normalizer.std, normalizer.epsilon),
        "actor_signature": actor_signature(actor),
        "actor_sha256": sha256(actor_path),
        "action_probability_contract": actor.action_probability_metadata(),
        "seed": 42, "scene_seed_offset": 0,
        "shared_exogenous_scene": sourceinfo["source_metadata"]["shared_exogenous_scene"],
        "fresh_optimizer_required": True, "fresh_reward_normalizer_required": True}
    if metadata["critic_sha256"] != sourceinfo["critic_file_sha256"]:
        raise AssertionError("initialization critic bytes changed")
    write_json(destination / "initialization_metadata.json", metadata)
    return {"directory": str(destination), "actor_sha256": metadata["actor_sha256"],
        "critic_sha256": metadata["critic_sha256"],
        "normalization_sha256": sha256(destination / "state_normalization.npz"),
        "actor_signature": metadata["actor_signature"], "norm_signature": metadata["norm_signature"]}


def run(args):
    source, output = _check_output(args)
    # The entry is import-safe: heavyweight libraries are imported only here.
    if str(OUTER_ROOT) not in sys.path:
        sys.path.insert(0, str(OUTER_ROOT))
    import numpy as np
    import torch
    from algorithms.MHBPPO.beam_pretraining import (PRETRAIN_CONTRACT,
        build_actor_from_checkpoint, file_sha256, load_dataset,
        train_receive_beam, write_json)
    from algorithms.MHBPPO.checkpoint import save_state_normalization
    from algorithms.MHBPPO.rollout_batch import actor_state_signature, normalization_signature

    old_threads = torch.get_num_threads()
    output.mkdir(parents=True)
    started = time.perf_counter()
    source_before = {p.name: file_sha256(p) for p in source.iterdir() if p.is_file()}
    config = {"contract": PRETRAIN_CONTRACT, "dataset": str(args.dataset.resolve()),
        "output": str(output), "source_checkpoint": str(source), "device": "cpu",
        "lr": args.lr, "epochs": args.epochs, "batch_size": args.batch_size,
        "threads": args.threads, "seed": args.seed, "patience": args.patience,
        "rho": args.rho, "kappa": args.kappa, "validation_draws": args.validation_draws,
        "min_delta": args.min_delta, "selection": "heldout validation sampled target gain",
        "source_iteration": 3000, "fresh_ppo_updates": 0,
        "test_used_for_selection": False, "production_files_modified": False}
    try:
        torch.set_num_threads(args.threads)
        write_json(output / "pretrain_config.json", config)
        arrays, manifest = load_dataset(args.dataset)
        actor, normalizer, sourceinfo = build_actor_from_checkpoint(source, device="cpu",
            state_dim=arrays["raw_states"].shape[1], uavs_num=4, cus_num=10, antenna_nums=10)
        _validate_source_binding(manifest, arrays, source, normalizer, sourceinfo, file_sha256)
        write_json(output / "dataset_manifest.json", manifest)
        write_json(output / "source_provenance.json", sourceinfo)
        original_state = {name: tensor.detach().cpu().clone() for name, tensor in actor.state_dict().items()}
        with (output / "pretrain_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
            fields = ["epoch", "train_kl_per_coordinate", "val_mean_gain", "best_val_mean_gain",
                      "best_epoch", "gradient_norm_mean", "optimizer_steps", "seconds", "no_improvement_epochs"]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            def progress(row):
                writer.writerow(row)
                stream.flush()
                print(json.dumps(row, ensure_ascii=False, allow_nan=False), flush=True)
            training = train_receive_beam(actor, arrays, normalizer, rho=args.rho, kappa=args.kappa,
                lr=args.lr, epochs=args.epochs, batch_size=args.batch_size, seed=args.seed,
                patience=args.patience, validation_draws=args.validation_draws,
                min_delta=args.min_delta, progress_callback=progress)
        _save_actor(output / "last_actor.pth", training["last_state_dict"], torch)
        _save_actor(output / "best_actor.pth", training["best_state_dict"], torch)
        bundle_args = dict(normalizer=normalizer, source=source, sourceinfo=sourceinfo,
            config=config, manifest=manifest, training=training, torch=torch,
            save_norm=save_state_normalization, actor_signature=actor_state_signature,
            norm_signature=normalization_signature, sha256=file_sha256, write_json=write_json)
        # Metadata SHA is computed from the actor matching the file in each bundle.
        actor.load_state_dict(original_state, strict=True)
        original = _write_initialization(output / "initial_original", actor, original_state,
                                         original=True, **bundle_args)
        actor.load_state_dict(training["best_state_dict"], strict=True)
        warmed = _write_initialization(output / "initial_pretrained", actor, training["best_state_dict"],
                                       original=False, **bundle_args)
        summary = {key: value for key, value in training.items()
                   if key not in ("best_state_dict", "last_state_dict")}
        summary.update(output=str(output), source_checkpoint=str(source), source_iteration=3000,
            source_unchanged=source_before == {p.name: file_sha256(p) for p in source.iterdir() if p.is_file()},
            split_counts={str(key): int(np.count_nonzero(arrays["split"] == key)) for key in (-1, 0, 1, 2)},
            initial_original=original, initial_pretrained=warmed,
            elapsed_seconds=time.perf_counter()-started, completed=True,
            offline_validation_only=True, ppo_started=False, solver_calls=0)
        if not summary["source_unchanged"]:
            raise AssertionError("read-only source checkpoint changed during pretraining")
        write_json(output / "validation.json", {"baseline": training["baseline_validation"],
            "best": training["best_validation"], "best_epoch": training["best_epoch"],
            "selection_split": "validation only", "test_evaluated": False})
        write_json(output / "result.json", summary)
        print(json.dumps({key: summary[key] for key in ("completed", "best_epoch", "epochs_completed",
            "validation_improved", "nonbeam_parameters_unchanged", "source_unchanged", "elapsed_seconds")},
            ensure_ascii=False, allow_nan=False), flush=True)
        return summary
    except BaseException as error:
        write_json(output / "failure.json", {"completed": False, "error": str(error),
            "error_type": type(error).__name__, "traceback": traceback.format_exc(),
            "source_unchanged": source_before == {p.name: file_sha256(p) for p in source.iterdir() if p.is_file()},
            "elapsed_seconds": time.perf_counter()-started, "ppo_started": False, "solver_calls": 0})
        raise
    finally:
        torch.set_num_threads(old_threads)


def main():
    run(make_parser().parse_args())


if __name__ == "__main__":
    main()
