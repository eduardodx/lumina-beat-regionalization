#!/usr/bin/env python3
"""Extract and SAVE the frozen two-tower variant features (site_ref, variant_repr, local_context).

Runs on the SageMaker notebook (needs the r1 checkpoint + hg38 fasta + the slices). ONE backbone
pass, no training -- the same machinery scripts/extract_native_pathogenicity_features.py uses.

WHY. The k-fold "train on Brazilian" screen (scripts/kfold_train_on_brazilian.py) needs the head's
input features for every variant, so it can retrain a head with Brazilian variants IN the training
set instead of held out. Those features -- site_ref, variant_repr = alt-ref, local_context (mean
+-64bp) -- are a DETERMINISTIC function of the FROZEN backbone (adapters.py:_extract_paired_variant_features),
so we extract them once here and all the k-fold head-fitting happens locally on the cached matrix.

This deliberately uses the frozen backbone WITHOUT the M0 LoRA: the screen asks whether adding
Brazilian training examples helps a head over the frozen representation. If it does not even at the
head level, the far more expensive full LoRA SageMaker k-fold is very unlikely to differ; if it
does, that is the signal to escalate to the real thing.

WHAT IT SAVES. Per dataset, an .npz with:
    features       [N, 3*d_full]  float32  (site_ref | variant_repr | local_context)
    original_index [N]            int
    label          [N]            int
    split          [N]            str  (split_within_gene: train/test/holdout)
plus a sidecar {dataset}.two_tower_index.parquet with original_index/label/split for easy joins.

USAGE
-----
    python scripts/extract_two_tower_features.py \
        --slice-dir ~/slices --fasta ~/hg38/hg38.fa \
        --datasets br_only nonbr_only abraom_pathogenic_present abraom_common_benign \
        --output-dir ~/v11eval/two_tower_features
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.clinvar.adapters import build_finetune_adapter  # noqa: E402
from eval.clinvar.dataset import DEFAULT_SPLIT_COLUMN, build_variant_cache  # noqa: E402

DEFAULT_CHECKPOINT = (
    "s3://ai4bio-lumina/releases/lumina-beat-v11v5-r1-202607071631/ckpt/best_checkpoint.pt"
)
DEFAULT_DATASETS = ["br_only", "nonbr_only", "abraom_pathogenic_present", "abraom_common_benign"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--slice-dir", type=Path, required=True)
    p.add_argument("--fasta", type=Path, required=True)
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    p.add_argument("--splits", nargs="*", default=["train", "test", "holdout"],
                   help="which split_within_gene values to keep (br needs all; nonbr needs train)")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, default=None)
    p.add_argument("--context-size", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--split-column", default=DEFAULT_SPLIT_COLUMN)
    p.add_argument("--device", default="cuda")
    return p.parse_args(argv)


@torch.no_grad()
def extract_dataset(adapter, cache: pd.DataFrame, batch_size: int) -> np.ndarray:
    chunks: list[np.ndarray] = []
    n = len(cache)
    for start in range(0, n, batch_size):
        chunk = cache.iloc[start:start + batch_size]
        site_ref, variant_repr, local_context = adapter.extract_variant_features(
            ref_seqs=[str(s) for s in chunk["ref_seq"].tolist()],
            alt_seqs=[str(s) for s in chunk["alt_seq"].tolist()],
            variant_offsets=[int(o) for o in chunk["variant_offset"].tolist()],
            ref_alleles=[str(a) for a in chunk["ref_allele"].tolist()],
            alt_alleles=[str(a) for a in chunk["alt_allele"].tolist()],
        )
        feats = torch.cat([site_ref, variant_repr, local_context], dim=-1)
        chunks.append(feats.float().cpu().numpy())
        print(f"    {min(start + batch_size, n)}/{n}", end="\r", flush=True)
    print()
    return np.concatenate(chunks, axis=0)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir or (args.output_dir / "cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if (torch.cuda.is_available() or args.device == "cpu") else "cpu")

    adapter = build_finetune_adapter("beat-v11", "r1", device, checkpoint_path=str(args.checkpoint))

    wanted_splits = {s.lower() for s in args.splits}
    for dataset in args.datasets:
        dataset_path = args.slice_dir / f"{dataset}.parquet"
        if not dataset_path.is_file():
            print(f"skip_missing_slice={dataset_path}", flush=True)
            continue
        cache_path = build_variant_cache(
            dataset_path=dataset_path,
            fasta_path=args.fasta,
            context_size=args.context_size,
            regime="A",
            cache_dir=cache_dir,
            split_column=args.split_column,
        )
        cache = pd.read_parquet(cache_path)
        keep = cache[args.split_column].astype(str).str.lower().isin(wanted_splits)
        cache = cache.loc[keep].reset_index(drop=True)
        if cache.empty:
            print(f"skip_empty dataset={dataset} (no rows in splits {sorted(wanted_splits)})", flush=True)
            continue

        print(f"extracting dataset={dataset} n={len(cache)}", flush=True)
        features = extract_dataset(adapter, cache, args.batch_size)

        original_index = cache["original_index"].astype(int).to_numpy()
        label = cache["label"].astype(int).to_numpy() if "label" in cache.columns else np.full(len(cache), -1)
        split = cache[args.split_column].astype(str).to_numpy()

        npz_path = args.output_dir / f"{dataset}.two_tower.npz"
        np.savez_compressed(npz_path, features=features.astype(np.float32),
                            original_index=original_index, label=label, split=split)
        pd.DataFrame({"original_index": original_index, "label": label, "split": split}).to_parquet(
            args.output_dir / f"{dataset}.two_tower_index.parquet", index=False
        )
        print(f"  saved {npz_path.name}  features={features.shape}  dim={features.shape[1]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
