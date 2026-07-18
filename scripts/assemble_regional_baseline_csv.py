#!/usr/bin/env python3
"""Assemble the baseline-csv the M5_v2/v3 calibration needs, from regional-eval summaries.

``calibrate_m5_v2_regional_scores.py`` / ``calibrate_m5_v3_safety.py`` require a baseline
CSV with columns ``model,dataset,n,auroc,auprc,mcc,recall,specificity`` (one row per
model x slice). Its HARD CONSTRAINTS only query the ``M0`` row (``br_only`` mcc,
``global_nonbr_no_abraom`` mcc + specificity); the other models are for the comparison
report. This reads each model's extracted ``regional_eval_summary.json`` and emits the rows
for the requested split -- matching the exact header of Pedro's v10 baseline CSVs.

    python scripts/assemble_regional_baseline_csv.py \
        --run M0:~/v11eval/reval_m0b  --run M4:~/v11eval/reval_m4b \
        --run M5:~/v11eval/reval_m5b  --run M7s:~/v11eval/reval_m7sb --run M7d:~/v11eval/reval_m7db \
        --split test  --out ~/v11eval/baseline_v11_test.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os

COLS = ["model", "dataset", "n", "auroc", "auprc", "mcc", "recall", "specificity"]


def _summary(run_dir: str) -> dict:
    hits = glob.glob(
        os.path.join(os.path.expanduser(run_dir), "**", "regional_eval_summary.json"),
        recursive=True,
    )
    if not hits:
        raise SystemExit(f"no regional_eval_summary.json under {run_dir}")
    with open(hits[0]) as fh:
        return json.load(fh)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, metavar="MODEL:DIR",
                    help="repeatable; e.g. M0:~/v11eval/reval_m0b  (M0 is required for the constraints)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows: list[dict] = []
    for spec in args.run:
        model, path = spec.split(":", 1)
        summary = _summary(path)
        for e in summary.get("evaluations", []):
            if e.get("split") != args.split:
                continue
            rows.append({
                "model": model,
                "dataset": e.get("dataset"),
                "n": e.get("n"),
                "auroc": e.get("auroc"),
                "auprc": e.get("auprc"),
                "mcc": e.get("mcc"),
                "recall": e.get("recall"),
                "specificity": e.get("specificity"),
            })

    out = os.path.expanduser(args.out)
    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows ({args.split}) to {out}")

    # Sanity: the exact M0 values the tuning's hard constraints read.
    m0 = {r["dataset"]: r for r in rows if r["model"] == "M0"}
    if not m0:
        print("  WARNING: no M0 rows -- the calibration's hard constraints will be NaN.")
    for ds in ["br_only", "global_nonbr_no_abraom"]:
        r = m0.get(ds)
        if r:
            print(f"  M0 {ds:24s} mcc={r['mcc']}  specificity={r['specificity']}")
        else:
            print(f"  WARNING: M0 row for {ds!r} missing (needed by hard constraints).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
