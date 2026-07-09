#!/usr/bin/env python3
"""Fast SageMaker input-channel sanity check for Lumina training data."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.sagemaker_utils import SM_DATA

EXPECTED_CHROMOSOMES = [f"chr{i}" for i in range(1, 23)] + ["chrX"]


def read_fasta_keys(fasta_path: Path) -> tuple[list[str], bool]:
    fai_path = Path(str(fasta_path) + ".fai")
    if fai_path.is_file():
        with fai_path.open(encoding="utf-8") as handle:
            return [line.split("\t", 1)[0] for line in handle if line.strip()], True

    names: list[str] = []
    with fasta_path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith(">"):
                names.append(line[1:].strip().split()[0])
    return names, False


def probe_training_data(data_root: str | Path) -> None:
    root = Path(data_root)
    fasta_path = root / "hg38" / "hg38.fa"
    required_paths = [
        fasta_path,
        root / "phylo" / "hg38.phyloP100way.bw",
        root / "phylo" / "hg38.phyloP470way.bw",
        root / "gencode" / "gencode.v38.annotation.gtf.gz",
    ]

    for path in required_paths:
        exists = path.is_file()
        size = path.stat().st_size if exists else 0
        print(f"sagemaker_data_probe path={path} exists={exists} size={size}")
        if not exists:
            raise SystemExit(f"SageMaker data preflight failed: missing {path}")

    names, used_fai = read_fasta_keys(fasta_path)
    missing = [chrom for chrom in EXPECTED_CHROMOSOMES if chrom not in set(names)]
    print(f"sagemaker_data_probe fasta={fasta_path} fai_exists={used_fai} key_count={len(names)}")
    print(f"sagemaker_data_probe fasta_keys_head={','.join(names[:30])}")
    if missing:
        raise SystemExit("SageMaker data preflight failed: hg38.fa missing " + ",".join(missing))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe SageMaker-mounted Lumina training data.")
    parser.add_argument("--data-root", default=SM_DATA)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    probe_training_data(args.data_root)


if __name__ == "__main__":
    main()
