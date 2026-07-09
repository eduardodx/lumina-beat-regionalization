#!/usr/bin/env python3
"""Container-side entrypoint for ABRAOM regionalization evaluation."""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path

from scripts.evaluate_abraom_regionalization import main as evaluate_main

DEFAULT_OUTPUT_DIR = Path("/opt/ml/model")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--candidate-root", required=True)
    parser.add_argument("--candidate-checkpoint-file", default="best_checkpoint.pt")
    parser.add_argument("--pairs-parquet", required=True)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--progress-every", type=int, default=50)
    return parser.parse_args(argv)


def resolve_candidate_checkpoint(candidate_root: Path, checkpoint_file: str) -> Path:
    direct_path = candidate_root / checkpoint_file
    if direct_path.is_file():
        return direct_path

    model_tar = candidate_root / "model.tar.gz"
    if not model_tar.is_file():
        matches = sorted(candidate_root.rglob(checkpoint_file))
        if matches:
            return matches[0]
        raise FileNotFoundError(f"Could not find {checkpoint_file!r} or model.tar.gz under {candidate_root}.")

    extract_dir = Path("/tmp/lumina_candidate_model")
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(model_tar, "r:gz") as archive:
        archive.extractall(extract_dir, filter="data")

    extracted_path = extract_dir / checkpoint_file
    if extracted_path.is_file():
        return extracted_path
    matches = sorted(extract_dir.rglob(checkpoint_file))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not find {checkpoint_file!r} inside {model_tar}.")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    candidate_checkpoint = resolve_candidate_checkpoint(Path(args.candidate_root), args.candidate_checkpoint_file)
    eval_args = [
        "--base-checkpoint",
        args.base_checkpoint,
        "--candidate-checkpoint",
        str(candidate_checkpoint),
        "--pairs-parquet",
        args.pairs_parquet,
        "--output-dir",
        args.output_dir,
        "--split",
        args.split,
        "--batch-size",
        str(args.batch_size),
        "--device",
        args.device,
        "--progress-every",
        str(args.progress_every),
    ]
    if args.limit is not None:
        eval_args.extend(["--limit", str(args.limit)])
    print(f"candidate_checkpoint={candidate_checkpoint}", flush=True)
    return evaluate_main(eval_args)


if __name__ == "__main__":
    raise SystemExit(main())
