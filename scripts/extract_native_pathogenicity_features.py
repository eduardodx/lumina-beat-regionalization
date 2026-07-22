#!/usr/bin/env python3
"""Extract per-variant conservation + ESM-2 missense-severity from the FROZEN Beat-v11 native
heads, for the calibration's molecular guard ("A-guarda" / Lever A).

WHY. On v11 the M5_v3 safety guard lost to M5_v2 because it protects variants by their
``molecular_probability`` -- which the stronger v11 molecular head OVER-assigns to many common
benigns, so the guard "un-discounts" 208 benign false-positives. Conservation (phyloP) and
ESM-2 missense-severity are pathogenicity-specific signals ORTHOGONAL to the ABRAOM allele
frequency: a genuine founder P/LP is conserved / high-severity; an over-scored common benign is
not. Guarding the frequency discount on THESE lets the guard recover P/LP recall WITHOUT the
benign false-positives -> raises br_only on both axes (moves the recall<->br_only frontier up
instead of sliding along it). The Beat-v10 backbone had no such heads, so this is a genuinely
v11-specific lever.

WHAT. These features are BACKBONE-ONLY (independent of the M5-bounded/M0 heads), so this runs
standalone -- it does NOT touch or re-run the M5-bounded eval. It reuses the eval's
``build_variant_cache`` (same windows) and the new
``FineTuneBeatV11Adapter.extract_native_pathogenicity_features``. Writes
``{slice}.{split}.native_features.parquet`` (original_index, label, phylo100, zoo241, phylo470,
missense_severity) to be merged by (slice, split, original_index) into the calibration.

Runs on the SageMaker notebook (needs the r1 checkpoint + hg38 fasta + the slices).

    python scripts/extract_native_pathogenicity_features.py \
        --slice-dir ~/slices --fasta ~/hg38/hg38.fa \
        --datasets br_only br_any regional_benchmark_any abraom_common_benign \
                   abraom_pathogenic_present abraom_pathogenic_common global_nonbr_no_abraom nonbr_only \
        --splits test holdout --output-dir ~/v11eval/native_features
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
DEFAULT_DATASETS = [
    "br_only", "br_any", "regional_benchmark_any", "abraom_common_benign",
    "abraom_pathogenic_present", "abraom_pathogenic_common", "global_nonbr_no_abraom", "nonbr_only",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--slice-dir", type=Path, required=True, help="dir with {dataset}.parquet slices")
    p.add_argument("--fasta", type=Path, required=True, help="hg38 fasta (to rebuild the variant windows)")
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Beat-v11 r1 backbone (s3:// ok)")
    p.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    p.add_argument("--splits", nargs="*", default=["test", "holdout"])
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, default=None, help="variant-cache dir (default: <output>/cache)")
    p.add_argument("--context-size", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--split-column", default=DEFAULT_SPLIT_COLUMN)
    p.add_argument("--device", default="cuda")
    return p.parse_args(argv)


def _slice_for_split(cache: pd.DataFrame, split_column: str, split: str) -> pd.DataFrame:
    if split == "all":
        return cache
    return cache.loc[cache[split_column].astype(str).str.lower() == split.lower()]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir or (args.output_dir / "cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if (torch.cuda.is_available() or args.device == "cpu") else "cpu")

    # Family "beat-v11" routes to FineTuneBeatV11Adapter (the checkpoint carries the config).
    adapter = build_finetune_adapter("beat-v11", "r1", device, checkpoint_path=str(args.checkpoint))

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

        for split in args.splits:
            rows = _slice_for_split(cache, args.split_column, split).reset_index(drop=True)
            n = len(rows)
            if n == 0:
                print(f"skip_empty dataset={dataset} split={split}", flush=True)
                continue
            print(f"extracting dataset={dataset} split={split} n={n}", flush=True)

            collected: dict[str, list[float]] = {}
            for start in range(0, n, args.batch_size):
                chunk = rows.iloc[start:start + args.batch_size]
                feats = adapter.extract_native_pathogenicity_features(
                    ref_seqs=[str(s) for s in chunk["ref_seq"].tolist()],
                    variant_offsets=[int(o) for o in chunk["variant_offset"].tolist()],
                    alt_alleles=[str(a) for a in chunk["alt_allele"].tolist()],
                )
                for key, values in feats.items():
                    collected.setdefault(key, []).extend(values)

            out = pd.DataFrame({
                "dataset": dataset,
                "split": split,
                "original_index": rows["original_index"].astype(int).tolist(),
                "label": rows["label"].astype(int).tolist() if "label" in rows.columns else -1,
                **collected,
            })
            out_path = args.output_dir / f"{dataset}.{split}.native_features.parquet"
            out.to_parquet(out_path, index=False)
            # sanity: conservation should be higher on pathogenic than benign if the head is informative
            if "label" in rows.columns and "phylo100" in out.columns:
                pos = out.loc[out["label"] == 1, "phylo100"].mean()
                neg = out.loc[out["label"] == 0, "phylo100"].mean()
                print(f"  saved {out_path.name}  | phylo100 mean  P/LP={pos:.3f}  B/LB={neg:.3f}", flush=True)
            else:
                print(f"  saved {out_path.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
