#!/usr/bin/env python3
"""Container-side entrypoint for regional ClinVar evaluation."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DEFAULT_DATASET_CHANNEL = Path("/opt/ml/input/data/dataset")
DEFAULT_REFERENCE_CHANNEL = Path("/opt/ml/input/data/reference")
DEFAULT_BASE_CHANNEL = Path("/opt/ml/input/data/base")
DEFAULT_FINETUNED_CHANNEL = Path("/opt/ml/input/data/finetuned")
DEFAULT_OUTPUT_DIR = Path("/opt/ml/model")


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta-file", default="hg38.fa")
    parser.add_argument("--base-checkpoint-file", default="best_checkpoint.pt")
    args, eval_args = parser.parse_known_args(argv)
    if eval_args and eval_args[0] == "--":
        eval_args = eval_args[1:]
    return args, eval_args


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
    args, eval_args = parse_args(argv)
    fasta_path = DEFAULT_REFERENCE_CHANNEL / args.fasta_file
    base_checkpoint = DEFAULT_BASE_CHANNEL / args.base_checkpoint_file
    for path in [fasta_path, base_checkpoint]:
        if not path.is_file():
            raise FileNotFoundError(f"Required input not found: {path}")

    runtime_args = list(eval_args)
    runtime_args = _upsert_arg(runtime_args, "--dataset-dir", str(DEFAULT_DATASET_CHANNEL))
    runtime_args = _upsert_arg(runtime_args, "--fasta", str(fasta_path))
    runtime_args = _upsert_arg(runtime_args, "--base-checkpoint", str(base_checkpoint))
    runtime_args = _upsert_arg(runtime_args, "--finetuned-root", str(DEFAULT_FINETUNED_CHANNEL))
    runtime_args = _upsert_arg(runtime_args, "--output-dir", str(DEFAULT_OUTPUT_DIR))
    if "--overwrite" not in runtime_args:
        runtime_args.append("--overwrite")

    command = [sys.executable, "scripts/evaluate_clinvar_finetuned_model.py", *runtime_args]
    print("command=" + " ".join(command), flush=True)
    subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
