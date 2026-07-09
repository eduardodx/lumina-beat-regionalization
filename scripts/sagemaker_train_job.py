#!/usr/bin/env python3
"""Container-side entrypoint for the generic Lumina training launcher."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from src.sagemaker_utils import SM_DATA

DEFAULT_OUTPUT_DIR = Path("/opt/ml/model")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Lumina training inside SageMaker.")
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--abraom-data-root", default=None)
    parser.add_argument("--init-from-checkpoint", default=None)
    parser.add_argument("--nproc-per-node", type=int, default=None)
    return parser.parse_args(argv)


def build_torchrun_command(
    *,
    config_path: str,
    resume_from: str | None,
    abraom_data_root: str | None,
    init_from_checkpoint: str | None,
    nproc_per_node: int | None,
) -> list[str]:
    if resume_from and init_from_checkpoint:
        raise ValueError("--resume-from and --init-from-checkpoint are mutually exclusive.")
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nnodes",
        os.environ.get("SM_HOST_COUNT", "1"),
        "--node-rank",
        os.environ.get("SM_CURRENT_HOST_RANK", "0"),
        "--nproc-per-node",
        str(nproc_per_node) if nproc_per_node is not None else os.environ.get("SM_NUM_GPUS", "1"),
        "--rdzv-backend",
        "c10d",
        "--rdzv-endpoint",
        f"{os.environ.get('SM_MASTER_ADDR', '127.0.0.1')}:{os.environ.get('SM_MASTER_PORT', '29500')}",
        "--rdzv-id",
        os.environ.get("TRAINING_JOB_NAME", "lumina-ssm-rdzv"),
        "-m",
        "src.train",
        "--config",
        config_path,
        "--fasta-path",
        f"{SM_DATA}/hg38/hg38.fa",
        "--phylo100-bw-path",
        f"{SM_DATA}/phylo/hg38.phyloP100way.bw",
        "--phylo470-bw-path",
        f"{SM_DATA}/phylo/hg38.phyloP470way.bw",
        "--gtf-path",
        f"{SM_DATA}/gencode/gencode.v38.annotation.gtf.gz",
        "--output-dir",
        str(DEFAULT_OUTPUT_DIR),
    ]
    if resume_from:
        command.extend(["--resume-from", resume_from])
    if abraom_data_root:
        command.extend(["--abraom-data-root", abraom_data_root])
    if init_from_checkpoint:
        command.extend(["--init-from-checkpoint", init_from_checkpoint])
    return command


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    command = build_torchrun_command(
        config_path=args.config_path,
        resume_from=args.resume_from,
        abraom_data_root=args.abraom_data_root,
        init_from_checkpoint=args.init_from_checkpoint,
        nproc_per_node=args.nproc_per_node,
    )
    print(f"running={' '.join(command)}")
    subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
