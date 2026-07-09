#!/usr/bin/env python3
"""Container-side SageMaker entrypoint for ABRAOM frequency adapter training."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

DEFAULT_FREQUENCY_CHANNEL = Path("/opt/ml/input/data/frequency")
DEFAULT_REFERENCE_CHANNEL = Path("/opt/ml/input/data/reference")
DEFAULT_CHECKPOINT_CHANNEL = Path("/opt/ml/input/data/checkpoint")
DEFAULT_OUTPUT_DIR = Path("/opt/ml/model")


def _default_path(channel: Path, filename: str) -> str:
    return str(channel / filename)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-parquet",
        default=_default_path(DEFAULT_FREQUENCY_CHANNEL, "abraom_frequency_train.parquet"),
    )
    parser.add_argument(
        "--val-parquet",
        default=_default_path(DEFAULT_FREQUENCY_CHANNEL, "abraom_frequency_val.parquet"),
    )
    parser.add_argument(
        "--test-parquet",
        default=_default_path(DEFAULT_FREQUENCY_CHANNEL, "abraom_frequency_test.parquet"),
    )
    parser.add_argument("--fasta", default=_default_path(DEFAULT_REFERENCE_CHANNEL, "hg38.fa"))
    parser.add_argument("--checkpoint-path", default=_default_path(DEFAULT_CHECKPOINT_CHANNEL, "best_checkpoint.pt"))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--nproc-per-node",
        type=int,
        default=None,
        help="Reserved for future DDP support. Current adapter training runs as a single process.",
    )
    parser.add_argument("training_args", nargs=argparse.REMAINDER)
    return parser.parse_args(argv)


def _strip_separator(args: list[str]) -> list[str]:
    return args[1:] if args and args[0] == "--" else args


def _write_input_manifest(args: argparse.Namespace, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "train_parquet": args.train_parquet,
        "val_parquet": args.val_parquet,
        "test_parquet": args.test_parquet,
        "fasta": args.fasta,
        "checkpoint_path": args.checkpoint_path,
        "sm_num_gpus": os.environ.get("SM_NUM_GPUS"),
        "sm_training_env": os.environ.get("SM_TRAINING_ENV"),
    }
    (output_dir / "sagemaker_input_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def build_command(args: argparse.Namespace) -> list[str]:
    training_args = _strip_separator(list(args.training_args))
    output_dir = Path(args.output_dir)
    _write_input_manifest(args, output_dir)
    return [
        sys.executable,
        "scripts/train_abraom_frequency_adapter.py",
        "--train-parquet",
        args.train_parquet,
        "--val-parquet",
        args.val_parquet,
        "--test-parquet",
        args.test_parquet,
        "--fasta",
        args.fasta,
        "--checkpoint-path",
        args.checkpoint_path,
        "--output-dir",
        str(output_dir),
        "--overwrite",
        *training_args,
    ]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    command = build_command(args)
    print(f"running={' '.join(command)}", flush=True)
    subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
