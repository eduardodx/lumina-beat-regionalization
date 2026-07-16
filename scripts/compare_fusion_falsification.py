#!/usr/bin/env python3
"""Paired bootstrap of a classification metric between fusion runs on a regional slice.

Phase-3 falsification: does the REAL ABRAOM fusion beat its SCRAMBLED control on a slice
(e.g. ``br_only`` MCC), or is the gap within noise? Each fusion's regional eval writes
``{slice}.{split}.predictions.parquet`` (columns ``original_index, label, probability``)
plus a ``regional_eval_summary.json`` carrying the model's ``threshold`` (its val-optimal
operating point). Because every model is evaluated on the SAME slice variants, we compare a
pair with a PAIRED bootstrap on ``original_index`` -- the standard error of the *difference*
in the metric is far smaller than each model's own SE, which is what actually decides
whether e.g. a +0.054 br_only MCC gap on n=504 is real.

Metric + threshold rule match ``eval/clinvar/metrics.py`` EXACTLY (``preds = prob >=
threshold``; ``mcc = (tp*tn - fp*fn) / sqrt((tp+fp)(tp+fn)(tn+fp)(tn+fn))``), so each run's
point value reproduces its ``regional_eval_summary.json`` -- a built-in sanity check.

Self-contained (pandas + numpy); runs anywhere the extracted regional-eval artifacts live:

    python scripts/compare_fusion_falsification.py \
        --run M4:~/v11eval/reval_m4b  --run M7s:~/v11eval/reval_m7sb \
        --run M5:~/v11eval/reval_m5b  --run M7d:~/v11eval/reval_m7db \
        --pair M4:M7s --pair M5:M7d \
        --slice br_only --metric mcc --split test
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os

import numpy as np
import pandas as pd


def _find(run_dir: str, pattern: str) -> str:
    run_dir = os.path.expanduser(run_dir)
    hits = glob.glob(os.path.join(run_dir, "**", pattern), recursive=True)
    if not hits:
        raise SystemExit(f"no {pattern} under {run_dir}")
    return hits[0]


def _threshold(run_dir: str) -> float:
    return float(json.load(open(_find(run_dir, "regional_eval_summary.json")))["threshold"])


def _load(run_dir: str, slice_name: str, split: str) -> pd.DataFrame:
    path = _find(run_dir, f"{slice_name}.{split}.predictions.parquet")
    return pd.read_parquet(path, columns=["original_index", "label", "probability"])


def _metric(label: np.ndarray, prob: np.ndarray, threshold: float, name: str) -> float:
    # Mirrors eval/clinvar/metrics.py:classification_metrics exactly.
    label = label.astype(np.int64)
    pred = (prob >= threshold).astype(np.int64)
    tp = int(((label == 1) & (pred == 1)).sum())
    tn = int(((label == 0) & (pred == 0)).sum())
    fp = int(((label == 0) & (pred == 1)).sum())
    fn = int(((label == 1) & (pred == 0)).sum())
    if name == "mcc":
        den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        return (tp * tn - fp * fn) / den if den > 0 else 0.0
    if name == "specificity":
        return tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    if name == "recall":
        return tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    raise SystemExit(f"unknown metric {name!r}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, metavar="LABEL:DIR",
                    help="repeatable; e.g. M4:~/v11eval/reval_m4b")
    ap.add_argument("--pair", action="append", required=True, metavar="REAL:SCRAMBLED",
                    help="repeatable; e.g. M4:M7s")
    ap.add_argument("--slice", default="br_only")
    ap.add_argument("--split", default="test")
    ap.add_argument("--metric", default="mcc", choices=["mcc", "specificity", "recall"])
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    runs = dict(spec.split(":", 1) for spec in args.run)

    print(f"point {args.metric}({args.slice}.{args.split}) per run "
          f"(should match regional_eval_summary.json):")
    data: dict[str, tuple[pd.DataFrame, float]] = {}
    for label, path in runs.items():
        df = _load(path, args.slice, args.split)
        thr = _threshold(path)
        data[label] = (df, thr)
        pt = _metric(df["label"].to_numpy(), df["probability"].to_numpy(), thr, args.metric)
        print(f"  {label:6s} {pt:.4f}   (n={len(df)}, thr={thr:.4f})")

    rng = np.random.default_rng(args.seed)
    print(f"\npaired bootstrap of ({args.metric}: real - scrambled), B={args.bootstrap}:")
    for spec in args.pair:
        real, scr = spec.split(":", 1)
        dfr, tr = data[real]
        dfs, ts = data[scr]
        merged = dfr.rename(columns={"probability": "p_real"}).merge(
            dfs[["original_index", "probability"]].rename(columns={"probability": "p_scr"}),
            on="original_index", how="inner",
        )
        n = len(merged)
        y = merged["label"].to_numpy(np.int64)
        pr = merged["p_real"].to_numpy(np.float64)
        ps = merged["p_scr"].to_numpy(np.float64)
        point = _metric(y, pr, tr, args.metric) - _metric(y, ps, ts, args.metric)
        diffs = np.empty(args.bootstrap)
        for b in range(args.bootstrap):
            idx = rng.integers(0, n, n)
            diffs[b] = (_metric(y[idx], pr[idx], tr, args.metric)
                        - _metric(y[idx], ps[idx], ts, args.metric))
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        p_gt = float((diffs > 0).mean())
        flag = "  <-- CI excludes 0" if (lo > 0 or hi < 0) else ""
        print(f"  {real} - {scr:6s} Δ={point:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
              f"P({real}>{scr})={p_gt:.3f}  (paired n={n}){flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
