"""Serial, paired offline head-only versus private-beam-encoder experiment.

No existing actor/PPO/environment/reward file is modified. Source is read-only;
test rows are reserved for a separate final validator. This never starts PPO.
"""
import argparse
import copy
import csv
import json
from pathlib import Path
import sys
import time
import traceback


OUTER_ROOT = Path(__file__).resolve().parents[2]


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--source-checkpoint", "--source_checkpoint", dest="source_checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--lr", type=float, default=.001)
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


def initial_distribution_checks(source, head_only, branch, states, torch):
    """Check all action distributions at t0 without consuming RNG/test rows."""
    with torch.no_grad():
        original_continuous, original_discrete = source._get_dists(states)
        for name, actor in (("head_only", head_only), ("branch", branch)):
            continuous, discrete = actor._get_dists(states)
            for key, distribution in original_continuous.items():
                if not (torch.equal(distribution.concentration1, continuous[key].concentration1) and
                        torch.equal(distribution.concentration0, continuous[key].concentration0)):
                    raise AssertionError(f"{name} initial {key} differs from source")
            if any(not torch.equal(first.logits, second.logits) for first, second in zip(original_discrete, discrete)):
                raise AssertionError(f"{name} initial matching distribution differs from source")
    return True


def run(args):
    if str(OUTER_ROOT) not in sys.path:
        sys.path.insert(0, str(OUTER_ROOT))
    # Reuse U48's checked source, data binding, atomic writes and bundle format;
    # do not change the old module or call its CLI/training entry implicitly.
    from algorithms.MHBPPO.pretrain_receive_beam import (
        _check_output, _validate_source_binding, _save_actor, _write_initialization,
    )
    source, output = _check_output(args)
    import numpy as np
    import torch
    from algorithms.MHBPPO.beam_pretraining import (
        build_actor_from_checkpoint, file_sha256, load_dataset,
        train_receive_beam, write_json,
    )
    from algorithms.MHBPPO.beam_branch import (
        BRANCH_PRETRAIN_CONTRACT, build_branch_actor, train_branch,
    )
    from algorithms.MHBPPO.checkpoint import save_state_normalization
    from algorithms.MHBPPO.rollout_batch import actor_state_signature, normalization_signature

    old_threads = torch.get_num_threads()
    output.mkdir(parents=True)
    started = time.perf_counter()
    source_before = {path.name: file_sha256(path) for path in source.iterdir() if path.is_file()}
    config = {"contract": BRANCH_PRETRAIN_CONTRACT, "dataset": str(args.dataset.resolve()),
        "output": str(output), "source_checkpoint": str(source), "device": "cpu",
        "lr": args.lr, "epochs": args.epochs, "batch_size": args.batch_size,
        "threads": args.threads, "seed": args.seed, "patience": args.patience,
        "rho": args.rho, "kappa": args.kappa, "validation_draws": args.validation_draws,
        "min_delta": args.min_delta, "selection": "heldout validation sampled target gain",
        "execution": "serial head_only then branch; each starts from same source",
        "branch_input": "source_exact_normalized_full_raw_state",
        "source_iteration": 3000, "fresh_ppo_updates": 0,
        "test_used_for_selection": False, "existing_production_files_modified": False}
    try:
        torch.set_num_threads(args.threads)
        write_json(output/"pretrain_config.json", config)
        arrays, manifest = load_dataset(args.dataset)
        original, normalizer, sourceinfo = build_actor_from_checkpoint(source, device="cpu",
            state_dim=arrays["raw_states"].shape[1], uavs_num=4, cus_num=10, antenna_nums=10)
        _validate_source_binding(manifest, arrays, source, normalizer, sourceinfo, file_sha256)
        write_json(output/"dataset_manifest.json", manifest)
        write_json(output/"source_provenance.json", sourceinfo)
        source_actor_signature = actor_state_signature(original)
        head_only, branch = copy.deepcopy(original), build_branch_actor(original)
        # t0 is checked on eligible train observations; test remains untouched.
        initial_rows = np.flatnonzero(arrays["split"] == 0)[:64]
        states = torch.as_tensor(normalizer.normalize(arrays["raw_states"][initial_rows]), dtype=torch.float32)
        initial_exact = initial_distribution_checks(original, head_only, branch, states, torch)
        if not (torch.equal(original.fc1.weight, branch.beam_fc1.weight) and
                torch.equal(original.fc1.bias, branch.beam_fc1.bias) and
                torch.equal(original.fc2.weight, branch.beam_fc2.weight) and
                torch.equal(original.fc2.bias, branch.beam_fc2.bias)):
            raise AssertionError("new beam encoder is not an exact source-layer clone")
        fit_args = dict(rho=args.rho, kappa=args.kappa, lr=args.lr, epochs=args.epochs,
            batch_size=args.batch_size, seed=args.seed, patience=args.patience,
            validation_draws=args.validation_draws, min_delta=args.min_delta)
        arms, summaries = {}, {}
        for arm, actor, trainer in (("head_only", head_only, train_receive_beam), ("branch", branch, train_branch)):
            arm_dir = output/arm
            arm_dir.mkdir()
            arm_started = time.perf_counter()
            with (arm_dir/"metrics.csv").open("w", newline="", encoding="utf-8") as stream:
                fields = ["epoch", "train_kl_per_coordinate", "val_mean_gain", "best_val_mean_gain",
                    "best_epoch", "gradient_norm_mean", "optimizer_steps", "seconds", "no_improvement_epochs"]
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                def progress(row):
                    writer.writerow(row)
                    stream.flush()
                    print(json.dumps(dict(arm=arm, **row), ensure_ascii=False, allow_nan=False), flush=True)
                result = trainer(actor, arrays, normalizer, progress_callback=progress, **fit_args)
            result["arm_elapsed_seconds"] = time.perf_counter()-arm_started
            _save_actor(arm_dir/"best_actor.pth", result["best_state_dict"], torch)
            _save_actor(arm_dir/"last_actor.pth", result["last_state_dict"], torch)
            summary = {key: value for key, value in result.items() if key not in ("best_state_dict", "last_state_dict")}
            summary["arm"] = arm
            arms[arm], summaries[arm] = result, summary
            write_json(arm_dir/"result.json", summary)
        if (arms["head_only"]["validation_seed"] != arms["branch"]["validation_seed"] or
                arms["head_only"]["baseline_validation"]["mean_gain"] != arms["branch"]["baseline_validation"]["mean_gain"]):
            raise AssertionError("paired arms did not start at identical sampled validation gain/seed")
        if actor_state_signature(original) != source_actor_signature:
            raise AssertionError("paired training mutated the original actor")

        bundles = {}
        for name, actor, training, is_original in (
            ("original", original, arms["head_only"], True),
            ("head_only", head_only, arms["head_only"], False),
            ("branch", branch, arms["branch"], False)):
            bundle_info = dict(sourceinfo)
            if name == "branch":
                bundle_info["architecture"] = branch.architecture_metadata()
            else:
                bundle_info["architecture"] = dict(sourceinfo["architecture"], actor_class="MultiHeadActor",
                    base_actor_strict_load_compatible=True)
            destination = output/f"initial_{name}"
            state = {key: value.detach().cpu().clone() for key, value in actor.state_dict().items()}
            bundles[name] = _write_initialization(destination, actor, state, normalizer,
                source, bundle_info, config, manifest, training, original=is_original,
                torch=torch, save_norm=save_state_normalization,
                actor_signature=actor_state_signature, norm_signature=normalization_signature,
                sha256=file_sha256, write_json=write_json)
            metadata_path = destination/"initialization_metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata.update(initialization_mode=name, paired_experiment_contract=BRANCH_PRETRAIN_CONTRACT,
                actor_class="DedicatedBeamActor" if name == "branch" else "MultiHeadActor",
                base_actor_strict_load_compatible=name != "branch",
                original_model_source_signature=source_actor_signature)
            write_json(metadata_path, metadata)
        source_after = {path.name: file_sha256(path) for path in source.iterdir() if path.is_file()}
        if source_before != source_after:
            raise AssertionError("read-only source checkpoint files changed")
        summary = {"completed": True, "contract": BRANCH_PRETRAIN_CONTRACT,
            "source_checkpoint": str(source), "source_iteration": 3000,
            "source_unchanged": True, "source_file_sha256": source_before,
            "source_actor_signature": source_actor_signature,
            "dataset_sha256": manifest["_dataset_sha256"],
            "manifest_sha256": manifest["_manifest_sha256"],
            "split_counts": {str(value): int(np.count_nonzero(arrays["split"] == value)) for value in (-1, 0, 1, 2)},
            "initial_distributions_exact": initial_exact,
            "initial_branch_layers_exact_source_copies": True,
            "arms": summaries, "initializations": bundles,
            "elapsed_seconds": time.perf_counter()-started,
            "test_evaluated": False, "ppo_started": False, "solver_calls": 0,
            "branch_requires_new_actor_class": True,
            "base_training_and_U48_code_modified": False}
        write_json(output/"result.json", summary)
        print(json.dumps({"completed": True, "initial_distributions_exact": initial_exact,
            "head_only_best_epoch": arms["head_only"]["best_epoch"],
            "branch_best_epoch": arms["branch"]["best_epoch"],
            "source_unchanged": True, "elapsed_seconds": summary["elapsed_seconds"]},
            ensure_ascii=False, allow_nan=False), flush=True)
        return summary
    except BaseException as error:
        write_json(output/"failure.json", {"completed": False, "error_type": type(error).__name__,
            "error": str(error), "traceback": traceback.format_exc(),
            "source_unchanged": source_before == {path.name: file_sha256(path) for path in source.iterdir() if path.is_file()},
            "elapsed_seconds": time.perf_counter()-started, "test_evaluated": False,
            "ppo_started": False, "solver_calls": 0})
        raise
    finally:
        torch.set_num_threads(old_threads)


def main():
    run(make_parser().parse_args())


if __name__ == "__main__":
    main()
