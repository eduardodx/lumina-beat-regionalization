#!/usr/bin/env python3
"""Container-side entrypoint for regional ClinVar adapter fusion."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from eval.clinvar.fusion_lora import resolve_adapter_checkpoint, resolve_finetuned_model_checkpoint

DEFAULT_DATA_ROOT = Path("/opt/ml/input/data")
DEFAULT_DATASET_CHANNEL = DEFAULT_DATA_ROOT / "dataset"
DEFAULT_REFERENCE_CHANNEL = DEFAULT_DATA_ROOT / "reference"
DEFAULT_CHECKPOINT_CHANNEL = DEFAULT_DATA_ROOT / "checkpoint"
DEFAULT_INIT_MODEL_CHANNEL = DEFAULT_DATA_ROOT / "init_model"
DEFAULT_OUTPUT_DIR = Path("/opt/ml/model")


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-file", default="nonbr_only.parquet")
    parser.add_argument("--fasta-file", default="hg38.fa")
    parser.add_argument("--checkpoint-file", default="best_checkpoint.pt")
    parser.add_argument("--fusion-mode", choices=["static_lora", "dynamic_lora"], default="static_lora")
    parser.add_argument("--fusion-adapter-names", nargs="*", default=["abraom", "gnomad"])
    # Parametrized model selection (default = Pedro's v10 baseline); pass
    # `--model-family beat-v11 --model-version r1` for the Beat-v11 port. Without these the
    # _upsert_arg calls below would hard-force v10.
    parser.add_argument("--model-family", default="lumina")
    parser.add_argument("--model-version", default="beat-v10")
    parser.add_argument(
        "--freeze-backbone-for-fusion",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args, training_args = parser.parse_known_args(argv)
    if training_args and training_args[0] == "--":
        training_args = training_args[1:]
    return args, training_args


def _has_arg(args: list[str], flag: str) -> bool:
    prefix = f"{flag}="
    return any(token == flag or token.startswith(prefix) for token in args)


def _upsert_arg(args: list[str], flag: str, value: str) -> list[str]:
    updated = list(args)
    if flag in updated:
        idx = updated.index(flag)
        if idx + 1 >= len(updated):
            raise ValueError(f"{flag} requires a value.")
        updated[idx + 1] = value
        return updated
    if any(token.startswith(f"{flag}=") for token in updated):
        return updated
    updated.extend([flag, value])
    return updated


def _append_multi_arg(args: list[str], flag: str, values: list[str]) -> list[str]:
    if _has_arg(args, flag):
        return args
    return [*args, flag, *values]


def _adapter_channel(adapter_name: str) -> Path:
    preferred = DEFAULT_DATA_ROOT / f"{adapter_name}_adapter"
    if preferred.exists():
        return preferred
    fallback = DEFAULT_DATA_ROOT / adapter_name
    if fallback.exists():
        return fallback
    return preferred


def main(argv: list[str] | None = None) -> int:
    args, training_args = parse_args(argv)
    dataset_path = DEFAULT_DATASET_CHANNEL / args.dataset_file
    fasta_path = DEFAULT_REFERENCE_CHANNEL / args.fasta_file
    checkpoint_path = DEFAULT_CHECKPOINT_CHANNEL / args.checkpoint_file
    init_checkpoint_path = resolve_finetuned_model_checkpoint(DEFAULT_INIT_MODEL_CHANNEL)
    adapter_paths = [
        resolve_adapter_checkpoint(_adapter_channel(adapter_name))
        for adapter_name in args.fusion_adapter_names
    ]

    for path in [dataset_path, fasta_path, checkpoint_path, init_checkpoint_path, *adapter_paths]:
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
    runtime_args = _upsert_arg(runtime_args, "--fusion-mode", args.fusion_mode)
    runtime_args = _upsert_arg(
        runtime_args,
        "--init-finetuned-checkpoint-path",
        str(init_checkpoint_path),
    )
    runtime_args = _append_multi_arg(runtime_args, "--fusion-adapter-names", list(args.fusion_adapter_names))
    runtime_args = _append_multi_arg(
        runtime_args,
        "--fusion-adapter-paths",
        [str(path) for path in adapter_paths],
    )
    if args.freeze_backbone_for_fusion and not _has_arg(runtime_args, "--freeze-backbone-for-fusion"):
        runtime_args.append("--freeze-backbone-for-fusion")
    if "--overwrite" not in runtime_args:
        runtime_args.append("--overwrite")

    command = [sys.executable, "-m", "eval.clinvar.run", *runtime_args]
    print("command=" + " ".join(command), flush=True)
    subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
