#!/usr/bin/env python3
"""Paired bootstrap comparison of frequency-adapter runs (spearman vs af_abraom).

Each training run writes ``{split}_predictions.parquet`` with columns
``variant_id, metric_target (=af_abraom), pred_af_abraom, af_gnomad, ...``. Because
every run is evaluated on the SAME rows (fixed splits), we can compare runs with a
PAIRED bootstrap -- the standard error of the *difference* in spearman is far smaller
than the ~0.014 SE of each spearman alone, which is what actually decides whether a
+0.01 gap (e.g. A_BR vs A_gnomAD at v11) is real.

For each split it reports:
  * point spearman(pred_af_abraom, af_abraom) per run  (sanity-check vs summary.json)
  * for every non-reference run: paired-bootstrap of (ref - run), 95% CI + P(ref>run)

Self-contained (pandas + numpy only; no model / mamba imports), so it runs anywhere
the extracted artifacts live:

    python scripts/compare_freq_adapters.py \
        --run A_BR:~/v11eval/abr \
        --run A_gnomAD:~/v11eval/gnomad \
        --run A_scrambled:~/v11eval/scrambled \
        --ref A_BR
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    # Spearman == Pearson of average ranks (pandas rank handles ties); no scipy needed.
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _find_predictions(run_dir: str, split: str) -> str | None:
    run_dir = os.path.expanduser(run_dir)
    hits = glob.glob(os.path.join(run_dir, "**", f"{split}_predictions.parquet"), recursive=True)
    return hits[0] if hits else None


def _load(run_dir: str, split: str) -> pd.DataFrame | None:
    path = _find_predictions(run_dir, split)
    if path is None:
        return None
    df = pd.read_parquet(path, columns=["variant_id", "metric_target", "pred_af_abraom"])
    return df.rename(columns={"pred_af_abraom": "pred"})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, metavar="LABEL:DIR",
                    help="repeatable; e.g. A_BR:~/v11eval/abr")
    ap.add_argument("--ref", default=None, help="label used as the reference (default: first --run)")
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    runs: dict[str, str] = {}
    for spec in args.run:
        if ":" not in spec:
            raise SystemExit(f"--run must be LABEL:DIR, got {spec!r}")
        label, path = spec.split(":", 1)
        runs[label] = path
    ref = args.ref or next(iter(runs))
    if ref not in runs:
        raise SystemExit(f"--ref {ref!r} not among runs {list(runs)}")
    rng = np.random.default_rng(args.seed)

    for split in args.splits:
        print("=" * 72)
        print(f"SPLIT: {split}")
        frames: dict[str, pd.DataFrame] = {}
        for label, path in runs.items():
            df = _load(path, split)
            if df is None:
                print(f"  {label:12s} -- no {split}_predictions.parquet under {path}")
                continue
            frames[label] = df

        if ref not in frames:
            print(f"  reference {ref!r} missing predictions for {split}; skipping split.")
            continue

        # Inner-join every run on variant_id so the bootstrap is truly paired.
        base = frames[ref][["variant_id", "metric_target", "pred"]].rename(columns={"pred": f"pred_{ref}"})
        for label, df in frames.items():
            if label == ref:
                continue
            base = base.merge(
                df[["variant_id", "pred"]].rename(columns={"pred": f"pred_{label}"}),
                on="variant_id", how="inner",
            )
        n = len(base)
        target = base["metric_target"].to_numpy(dtype=np.float64)
        preds = {label: base[f"pred_{label}"].to_numpy(dtype=np.float64) for label in frames}
        print(f"  paired rows (variant_id inner-join): {n}")

        # Point spearman per run (cross-check against summary.json).
        print("  point spearman(pred, af_abraom):")
        for label in frames:
            print(f"    {label:12s} {_spearman(preds[label], target):.4f}")

        # Paired bootstrap of (ref - other).
        others = [l for l in frames if l != ref]
        if not others:
            continue
        print(f"  paired bootstrap ({ref} - other), B={args.bootstrap}:")
        for label in others:
            diffs = np.empty(args.bootstrap)
            pr, po = preds[ref], preds[label]
            for b in range(args.bootstrap):
                idx = rng.integers(0, n, n)
                t = target[idx]
                diffs[b] = _spearman(pr[idx], t) - _spearman(po[idx], t)
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            p_gt = float((diffs > 0).mean())
            point = _spearman(pr, target) - _spearman(po, target)
            flag = "  <-- CI excludes 0" if (lo > 0 or hi < 0) else ""
            print(f"    {ref} - {label:12s} Δ={point:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
                  f"P({ref}>{label})={p_gt:.3f}{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
