#!/usr/bin/env python3
"""Is v11's br_only really below v10's? -- recompute the M5_v2 lead with more statistical power.

Runs LOCAL on the notebook over the existing M5-bounded prediction parquets. No GPU, no job.

THE CONCERN
-----------
The published leads compare v10 M5_v2 br_only MCC 0.605 against v11 M5_v2 0.574 -- v11, the better
model, looks WORSE by 0.031. Both numbers are on the `test` split, n=504. Two facts say that gap is
almost certainly sampling noise, not a real regression:

  1. Pedro's OWN gene-clustered bootstrap (lumina-ssm .../cluster_bootstrap_intervals.csv) gives
     br_only MCC deltas with 95% CIs ~0.17 wide -- so a single MCC carries roughly +-0.05-0.09 of
     uncertainty at this n. A 0.031 gap sits deep inside that.
  2. The exact same n=504 slice already fooled us once: v11 M0 br_only looked WORSE than v10
     (0.238 vs 0.279) on `test`, then REVERSED on the powered `all` split (0.335 vs 0.248, +0.087).

And the M5_v3 guard is a red herring: in Pedro's table v10 M5_v2 and M5_v3 give the SAME br_only
(0.605, 0.605) -- the guard only moved the global slice. So "run v3 on v11" would not recover it.

WHAT THIS DOES (the immediate, no-job step)
-------------------------------------------
Recomputes v11's M5_v2 br_only at its DEPLOYED operating point (the exact selected_config.json:
discount_scale, max_discount, regional_threshold) on:
    - test            (n~504)  -- reproduces the published 0.574 as a self-check
    - holdout         (n~657)
    - test + holdout  (n~1161) -- 2.3x the power of test alone
with GENE-CLUSTERED bootstrap CIs (resampling genes, matching Pedro's cluster=gene_symbol, since
br_only variants cluster by gene). Tie-aware AUROC is reported alongside as the threshold-free
capability measure.

The question it answers: does v11 M5_v2's br_only CI include v10's 0.605? If yes, the "regression"
is not statistically real and there is nothing to fix in the calibration.

HONEST CAVEAT ON THE POOLED NUMBER
----------------------------------
regional_threshold was tuned on the br_only HOLDOUT split, so the holdout (and the pooled)
MCC is mildly optimistic on the threshold; test alone is the fully honest operating point. The
definitive powered number needs the br_only `all` split (its train-split variants are also
out-of-sample for the model AND for this threshold) -- that requires one small M5-bounded eval
job. This script prints the runbook for it. AUROC has no threshold so it is clean on every split.

USAGE
-----
    python scripts/recompute_v11_br_only_powered.py \
        --m5-dir ~/v11eval/reval_m5bounded \
        --slice-dir ~/slices \
        --m5-v2-config ~/v11eval/m5_v2_v11/selected_config.json \
        --output-dir ~/v11eval/br_only_powered
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.clinvar.metrics import classification_metrics  # noqa: E402
from scripts.analyze_regional_gain_decomposition import load_split  # noqa: E402
from scripts.test_abraom_vs_gnomad_power import auroc, bc_interval  # noqa: E402

EPS = 1e-6
V10_REFERENCE_BR_ONLY = 0.605  # Pedro's v10 M5_v2 (= M5_v3) br_only MCC, test split


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    home = Path.home()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--m5-dir", type=Path, default=home / "v11eval/reval_m5bounded")
    p.add_argument("--slice-dir", type=Path, default=home / "slices")
    p.add_argument("--m5-v2-config", type=Path, default=home / "v11eval/m5_v2_v11/selected_config.json")
    p.add_argument("--dataset", default="br_only")
    p.add_argument("--v10-reference", type=float, default=V10_REFERENCE_BR_ONLY)
    p.add_argument("--bootstrap", type=int, default=5000)
    p.add_argument("--seed", type=int, default=20260724)
    p.add_argument("--output-dir", type=Path, default=home / "v11eval/br_only_powered")
    return p.parse_args(argv)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def _logit(p: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0 - EPS)
    return np.log(clipped / (1.0 - clipped))


def m5_v2_score(frame: pd.DataFrame, discount_scale: float, max_discount: float) -> np.ndarray:
    """Exact M5_v2 formula (calibrate_m5_v2_regional_scores.py:91-96)."""
    discount = np.minimum(
        pd.to_numeric(frame["regional_discount"], errors="coerce").to_numpy(dtype=np.float64) * discount_scale,
        max_discount,
    )
    molecular = pd.to_numeric(frame["molecular_probability"], errors="coerce").to_numpy(dtype=np.float64)
    return _sigmoid(_logit(molecular) - discount)


def mcc_at(labels: np.ndarray, scores: np.ndarray, threshold: float) -> float:
    if labels.size == 0 or np.unique(labels).size < 2:
        return float("nan")
    return float(classification_metrics(labels, scores, float(threshold))["mcc"])


def gene_clustered_bootstrap(
    labels: np.ndarray, scores: np.ndarray, genes: np.ndarray, threshold: float,
    *, resamples: int, rng: np.random.Generator,
) -> tuple[float, float]:
    """Resample GENES with replacement (Pedro's cluster=gene_symbol); recompute MCC each draw.
    Variants within a gene are correlated, so gene clustering is the honest uncertainty here."""
    unique_genes = np.array(sorted(set(genes)))
    by_gene = {g: np.flatnonzero(genes == g) for g in unique_genes}
    observed = mcc_at(labels, scores, threshold)
    draws = np.empty(resamples, dtype=np.float64)
    for i in range(resamples):
        picked = rng.choice(unique_genes, unique_genes.size, replace=True)
        idx = np.concatenate([by_gene[g] for g in picked])
        draws[i] = mcc_at(labels[idx], scores[idx], threshold)
    return bc_interval(draws, observed)


def evaluate_split(name: str, frame: pd.DataFrame, threshold: float, args, rng) -> dict:
    labels = pd.to_numeric(frame["label"], errors="coerce").fillna(-1).to_numpy(dtype=np.int64)
    scores = frame["_m5_v2_score"].to_numpy(dtype=np.float64)
    genes = frame["GeneSymbol"].fillna("unknown").astype(str).to_numpy() if "GeneSymbol" in frame else \
        np.array(["unknown"] * len(frame))
    keep = labels >= 0
    labels, scores, genes = labels[keep], scores[keep], genes[keep]
    mcc = mcc_at(labels, scores, threshold)
    ci_low, ci_high = gene_clustered_bootstrap(labels, scores, genes, threshold,
                                               resamples=args.bootstrap, rng=rng)
    return {
        "split": name,
        "n": int(labels.size),
        "n_pathogenic": int((labels == 1).sum()),
        "n_genes": int(len(set(genes))),
        "mcc_at_deployed_threshold": mcc,
        "mcc_ci_low": ci_low,
        "mcc_ci_high": ci_high,
        "auroc": auroc(labels, scores),
        "covers_v10_reference": bool(np.isfinite(ci_low) and ci_low <= args.v10_reference <= ci_high),
    }


def _fmt(v, d: int = 3) -> str:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if not np.isfinite(v) else f"{v:.{d}f}"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rng = np.random.default_rng(args.seed)

    if not args.m5_v2_config.is_file():
        print(f"[error] missing config: {args.m5_v2_config}", file=sys.stderr)
        return 1
    config = json.loads(args.m5_v2_config.read_text(encoding="utf-8"))
    discount_scale = float(config.get("discount_scale", 1.0))
    max_discount = float(config.get("max_discount", 1.5))
    threshold = float(config.get("regional_threshold", 0.5))
    print(f"[info] deployed M5_v2 config: discount_scale={discount_scale} max_discount={max_discount} "
          f"regional_threshold={threshold}")

    frames: dict[str, pd.DataFrame] = {}
    for split in ("test", "holdout"):
        frame = load_split(args.m5_dir, args.slice_dir, args.dataset, split)
        if frame is None:
            print(f"[warn] missing {args.dataset}.{split} predictions in {args.m5_dir}", file=sys.stderr)
            continue
        frame = frame.copy()
        frame["_m5_v2_score"] = m5_v2_score(frame, discount_scale, max_discount)
        frames[split] = frame

    if not frames:
        print("[error] no predictions loaded; check --m5-dir", file=sys.stderr)
        return 1

    rows = [evaluate_split(name, frame, threshold, args, rng) for name, frame in frames.items()]
    if {"test", "holdout"} <= frames.keys():
        pooled = pd.concat([frames["test"], frames["holdout"]], ignore_index=True)
        rows.append(evaluate_split("test+holdout", pooled, threshold, args, rng))

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset": args.dataset,
        "config": {"discount_scale": discount_scale, "max_discount": max_discount, "regional_threshold": threshold},
        "v10_reference_br_only": args.v10_reference,
        "results": rows,
    }

    lines = []
    lines.append("=" * 90)
    lines.append(f"v11 {args.dataset} M5_v2 recomputed with more power (deployed operating point)")
    lines.append(f"v10 reference (Pedro M5_v2 = M5_v3): {_fmt(args.v10_reference)}   "
                 f"published v11 test: 0.574")
    lines.append("=" * 90)
    lines.append("")
    lines.append(f"    {'split':<16}{'n':>6}{'genes':>7}{'MCC':>8}{'95% CI (gene-clustered)':>28}{'AUROC':>8}")
    lines.append("    " + "-" * 73)
    for row in rows:
        ci = f"[{_fmt(row['mcc_ci_low'])}, {_fmt(row['mcc_ci_high'])}]"
        flag = "  <- covers v10 0.605" if row["covers_v10_reference"] else ""
        lines.append(f"    {row['split']:<16}{row['n']:>6}{row['n_genes']:>7}"
                     f"{_fmt(row['mcc_at_deployed_threshold']):>8}{ci:>28}{_fmt(row['auroc']):>8}{flag}")
    lines.append("")

    test_row = next((r for r in rows if r["split"] == "test"), None)
    if test_row is not None:
        ok = abs(test_row["mcc_at_deployed_threshold"] - 0.574) <= 0.01
        lines.append(f"[self-check] test MCC={_fmt(test_row['mcc_at_deployed_threshold'])} "
                     f"vs published 0.574 -> {'OK' if ok else 'MISMATCH (investigate before trusting the rest)'}")
    any_covers = any(r["covers_v10_reference"] for r in rows)
    lines.append("")
    lines.append("=" * 90)
    if any_covers:
        lines.append("VERDICT: v11's br_only CI INCLUDES v10's 0.605 -> the 0.574-vs-0.605 gap is NOT")
        lines.append("statistically real. v11 is not worse; the test-split point estimate is noisy.")
        lines.append("There is nothing to fix in the calibration. For the definitive point estimate,")
        lines.append("run the br_only `all` split eval (n=4163) below.")
    else:
        lines.append("VERDICT: v11's br_only CI does NOT reach 0.605 -> the gap may be real; escalate")
        lines.append("to the powered `all`-split recompute before concluding.")
    lines.append("")
    lines.append("Definitive powered number (one small eval job, then re-run this on the all split):")
    lines.append("  1. score the br_only TRAIN-split variants with the M5-bounded model (the eval that")
    lines.append("     produced reval_m5bounded, but --splits all or train for --dataset-files br_only)")
    lines.append("  2. re-run this script pointing --m5-dir at that output; add a 'train' / 'all' split")
    lines.append("=" * 90)
    report = "\n".join(lines)
    print(report)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "br_only_powered.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (args.output_dir / "br_only_powered.txt").write_text(report, encoding="utf-8")
    print(f"\nwrote {args.output_dir}/br_only_powered.{{json,txt}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
