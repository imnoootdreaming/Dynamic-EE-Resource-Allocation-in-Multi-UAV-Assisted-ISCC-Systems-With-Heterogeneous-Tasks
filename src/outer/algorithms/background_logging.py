"""Prepare background logs and display baseline training progress."""

import argparse
import csv
import json
import os
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from tempfile import mkdtemp


BACKGROUND_LOG_ENV = "ISCC_BACKGROUND_LOG_FILE"
DEFAULT_LOG = "logs/MHBPPO/mhbppo_train.log"


def prepare_background_log(algorithm, outer_root, arguments):
    """Reserve a run directory before Python training begins, including startup errors."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default="")
    parser.add_argument("--log_file", default=None)
    try:
        args, _ = parser.parse_known_args(arguments)
    except SystemExit:
        # The real training parser will report malformed options in the log.
        args = argparse.Namespace(config="", log_file=None)
    configured = DEFAULT_LOG
    if args.config:
        try:
            payload = json.loads(Path(args.config).read_text(encoding="utf-8"))
            settings = payload.get("hyperparameters", payload)
            if isinstance(settings, dict) and isinstance(settings.get("log_file"), str):
                configured = settings["log_file"]
        except (OSError, ValueError, TypeError, AttributeError):
            # The training parser will include the original error in this log.
            pass
    if args.log_file is not None:
        configured = args.log_file
    outer_root = Path(outer_root).resolve()
    if algorithm != "MHBPPO" and Path(configured) == Path(DEFAULT_LOG):
        configured = str(Path("logs") / algorithm / (algorithm.lower() + "_train.log"))
    configured_path = Path(configured)
    if not configured_path.is_absolute():
        configured_path = outer_root / configured_path
    name = configured_path.name or algorithm.lower() + "_train.log"
    if algorithm == "MHBPPO" and not Path(name).suffix:
        name += ".log"
    configured_path.parent.mkdir(parents=True, exist_ok=True)
    prefix = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_")
    directory = Path(mkdtemp(prefix=prefix, dir=str(configured_path.parent)))
    return directory / name


def complete_lines(stream):
    """Read through the last fully written line, retaining any partial line."""
    while True:
        position = stream.tell()
        line = stream.readline()
        if not line.endswith("\n"):
            stream.seek(position)
            return
        yield line


def follow_baseline_progress(algorithm, log_file):
    prefix = f"[{algorithm}] checkpoint="
    with Path(log_file).open(encoding="utf-8", errors="replace") as log:
        checkpoint = None
        while checkpoint is None:
            for line in complete_lines(log):
                if line.startswith(prefix) and " | logs=" in line:
                    checkpoint = Path(line[len(prefix):].split(" | logs=", 1)[0])
                    break
            if checkpoint is None:
                time.sleep(1)

    updates = checkpoint / "training_updates.csv"
    while not updates.is_file():
        time.sleep(1)
    with updates.open(encoding="utf-8", newline="") as stream:
        header = None
        while header is None:
            header = next(complete_lines(stream), None)
            if header is None:
                time.sleep(1)
        columns = next(csv.reader([header]))
        total = 0.0

        def progress_lines():
            nonlocal total
            for row in csv.DictReader(complete_lines(stream), fieldnames=columns):
                elapsed = float(row["elapsed_seconds"])
                total += elapsed
                hours, remainder = divmod(int(total), 3600)
                minutes, seconds = divmod(remainder, 60)
                yield (f"第 {row['iteration']} 轮 | 平均奖励 {float(row['reward']):.4f} | "
                       f"本轮约 {elapsed:.1f} 秒 | 当前总耗时 {hours:02d}:{minutes:02d}:{seconds:02d}")

        for line in deque(progress_lines(), maxlen=5):
            print(line, flush=True)
        while True:
            time.sleep(1)
            for line in progress_lines():
                print(line, flush=True)


if __name__ == "__main__":
    import sys

    if sys.argv[1] == "progress":
        try:
            follow_baseline_progress(sys.argv[2], sys.argv[3])
        except KeyboardInterrupt:
            pass
    else:
        print(prepare_background_log(sys.argv[1], sys.argv[2], sys.argv[3:]))
