#!/usr/bin/env python3
"""Container-side entrypoint for regional ClinVar M0 training."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DEFAULT_DATASET_CHANNEL = Path("/opt/ml/input/data/dataset")
DEFAULT_REFERENCE_CHANNEL = Path("/opt/ml/input/data/reference")
DEFAULT_CHECKPOINT_CHANNEL = Path("/opt/ml/input/data/checkpoint")
DEFAULT_OUTPUT_DIR = Path("/opt/ml/model")


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-file", default="nonbr_only.parquet")
    parser.add_argument("--fasta-file", default="hg38.fa")
    parser.add_argument("--checkpoint-file", default="best_checkpoint.pt")
    # Model selection is parametrized (default = Pedro's v10 baseline) so the same job
    # runs the Beat-v11 port via `--model-family beat-v11 --model-version r1`; without
    # these the _upsert_arg calls below would hard-force v10 (see eval.clinvar.run dispatch).
    parser.add_argument("--model-family", default="lumina")
    parser.add_argument("--model-version", default="beat-v10")
    args, training_args = parser.parse_known_args(argv)
    if training_args and training_args[0] == "--":
        training_args = training_args[1:]
    return args, training_args


def _upsert_arg(args: list[str], flag: str, value: str) -> list[str]:
    updated = list(args)
    if flag in updated:
        idx = updated.index(flag)
        if idx + 1 >= len(updated):
            raise ValueError(f"{flag} requires a value.")
        updated[idx + 1] = value
        return updated
    updated.extend([flag, value])
    return updated


def main(argv: list[str] | None = None) -> int:
    args, training_args = parse_args(argv)
    dataset_path = DEFAULT_DATASET_CHANNEL / args.dataset_file
    fasta_path = DEFAULT_REFERENCE_CHANNEL / args.fasta_file
    checkpoint_path = DEFAULT_CHECKPOINT_CHANNEL / args.checkpoint_file

    for path in [dataset_path, fasta_path, checkpoint_path]:
        if not path.is_file():
            raise FileNotFoundError(f"Required input not found: {path}")

    runtime_args = list(training_args)
    runtime_args = _upsert_arg(runtime_args, "--regime", "A")
    runtime_args = _upsert_arg(runtime_args, "--model-family", args.model_family)
    runtime_args = _upsert_arg(runtime_args, "--model-version", args.model_version)
    runtime_args = _upsert_arg(runtime_args, "--checkpoint-path", str(checkpoint_path))
    runtime_args = _upsert_arg(runtime_args, "--dataset-path", str(dataset_path))
    runtime_args = _upsert_arg(runtime_args, "--fasta-path", str(fasta_path))
    runtime_args = _upsert_arg(runtime_args, "--output-dir", str(DEFAULT_OUTPUT_DIR))
    if "--overwrite" not in runtime_args:
        runtime_args.append("--overwrite")

    command = [sys.executable, "-m", "eval.clinvar.run", *runtime_args]
    print("command=" + " ".join(command), flush=True)
    subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
