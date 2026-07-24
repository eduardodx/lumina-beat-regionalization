#!/usr/bin/env python3
"""Does ABRAOM frequency beat gnomAD frequency on Brazilian variants? -- the powered test.

Runs LOCAL. No GPU, no job, no training, no model predictions needed for the main test.

WHY THIS IS THE DECIDING QUESTION
---------------------------------
The premise of the whole regionalization front is that Brazilian allele frequencies carry
information a European-skewed global frequency does not. The re-baseline measured that directly
for the first time and ABRAOM LOST in all four Brazilian comparisons (br_only/br_any x
test/holdout), with gnomAD ranking pathogenicity better. But each had only 4-8 positives, far too
few to conclude anything.

This script re-runs that comparison with maximum available power, and adds the test that actually
matters.

WHY WE CAN USE EVERY ROW
------------------------
The main comparison uses NO model: it ranks variants by `1 - af_abraom` vs `1 - af_gnomad` and
scores them against the ClinVar label, straight from the slice parquets. Train/test leakage is a
property of *models*, so it simply does not apply here -- there is nothing that could have
memorized anything. That frees us from the split structure entirely and takes br_only from n=504
(test) to the full slice.

THE TEST THAT MATTERS: DIFFERENCE-IN-DIFFERENCES
-----------------------------------------------
"ABRAOM beats gnomAD in absolute terms" is not actually the regional claim, and it can fail for
boring reasons (ABRAOM is one elderly Sao Paulo cohort; gnomAD is orders of magnitude larger and
better powered, so its AF estimates are simply less noisy). The regional claim is comparative:

    delta_BR    = AUROC(abraom) - AUROC(gnomad)   on BRAZILIAN variants
    delta_nonBR = AUROC(abraom) - AUROC(gnomad)   on NON-BRAZILIAN variants
    DiD         = delta_BR - delta_nonBR

DiD > 0 means ABRAOM's edge grows specifically where Brazilian ancestry matters -- regional signal
-- even if ABRAOM loses to gnomAD everywhere in absolute terms. DiD ~ 0 means ABRAOM contributes
nothing Brazilian-specific, and the non-Brazilian slice is the built-in negative control.

THE AUROC BUG THIS ALSO FIXES
-----------------------------
eval/clinvar/metrics.py:30-36 computes AUROC by sorting on -score and walking one observation at
a time, WITHOUT grouping ties. The frequency baselines assign exactly 1.0 to every variant with
no ABRAOM entry (572 of 657 in br_only holdout), so the traced ROC curve depends on the arbitrary
order numpy happens to produce inside that tie block. Under bootstrap resampling the block's
composition changes and the estimate drifts -- which is why the re-baseline produced dAUROC
confidence intervals that did not contain their own point estimates.

The fix here is the exact tie-aware definition (Mann-Whitney U with mid-ranks):

    AUC = (sum of average ranks of positives - n_pos(n_pos+1)/2) / (n_pos * n_neg)

Ties get exactly 0.5 credit, deterministically. Section [0] reports the legacy value next to the
corrected one so we can see which published AUROCs, if any, need revisiting.

Confidence intervals are bias-corrected (BC) percentile intervals rather than plain percentile,
so an interval can no longer sit off to one side of its own point estimate.

P/LP RECALL IS A CONSTRAINT, NOT A FOOTNOTE
-------------------------------------------
Section [3] answers "can br_only go up without hurting P/LP recall?" as a CONSTRAINED problem:
for each score, pick the holdout threshold maximizing br_only MCC SUBJECT TO
abraom_pathogenic_present recall >= a floor (default 0.405, the deployed level), then report on
test. Alavanca B already showed unconstrained br_only maximization is a recall trap; this makes
the guard-rail part of the search instead of something checked afterwards.

USAGE
-----
    python scripts/test_abraom_vs_gnomad_power.py \
        --slice-dir  ~/slices \
        --m5-dir     ~/v11eval/reval_m5bounded \
        --m0-dir     ~/v11eval/m5v3_modelroot/m0_nonbr_beatv10_v1_sagemaker \
        --output-dir ~/v11eval/abraom_vs_gnomad
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.clinvar.metrics import binary_roc_auc, classification_metrics  # noqa: E402
from scripts.analyze_regional_gain_decomposition import align_to, load_split  # noqa: E402

BRAZILIAN_SLICES = ["br_only", "br_any"]
CONTROL_SLICES = ["nonbr_only"]
FOUNDER_SLICE = "abraom_pathogenic_present"
BENIGN_SLICE = "abraom_common_benign"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    home = Path.home()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--slice-dir", type=Path, default=home / "slices")
    parser.add_argument("--m5-dir", type=Path, default=home / "v11eval/reval_m5bounded")
    parser.add_argument("--m0-dir", type=Path,
                        default=home / "v11eval/m5v3_modelroot/m0_nonbr_beatv10_v1_sagemaker")
    parser.add_argument("--m5-v2-config", type=Path, default=home / "v11eval/m5_v2_v11/selected_config.json")
    parser.add_argument("--plp-recall-floor", type=float, default=0.405,
                        help="minimum abraom_pathogenic_present recall any operating point must hold")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--output-dir", type=Path, default=home / "v11eval/abraom_vs_gnomad")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Tie-aware AUROC + bias-corrected bootstrap
# ---------------------------------------------------------------------------

def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Mann-Whitney U with mid-ranks: exact, deterministic, ties get 0.5 credit."""
    positives = labels == 1
    n_pos = int(positives.sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(scores).rank(method="average").to_numpy()
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _ndtr(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _ndtri(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation)."""
    if not 0.0 < p < 1.0:
        return 0.0
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    p_low, p_high = 0.02425, 1 - 0.02425
    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > p_high:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def bc_interval(draws: np.ndarray, observed: float, alpha: float = 0.05) -> tuple[float, float]:
    """Bias-corrected percentile interval. Plain percentile intervals can land entirely off to
    one side of their own point estimate when the statistic is biased -- which is exactly what
    the tie-saturated frequency baseline produced."""
    draws = draws[np.isfinite(draws)]
    if draws.size < 50 or not np.isfinite(observed):
        return float("nan"), float("nan")
    proportion = float(np.mean(draws < observed))
    proportion = min(max(proportion, 1.0 / draws.size), 1.0 - 1.0 / draws.size)
    z0 = _ndtri(proportion)
    lo = _ndtr(2 * z0 + _ndtri(alpha / 2)) * 100.0
    hi = _ndtr(2 * z0 + _ndtri(1 - alpha / 2)) * 100.0
    return float(np.percentile(draws, lo)), float(np.percentile(draws, hi))


def stratified_indices(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Resample positives and negatives separately so no draw can lose a class entirely."""
    positive_idx = np.flatnonzero(labels == 1)
    negative_idx = np.flatnonzero(labels == 0)
    return np.concatenate([
        rng.choice(positive_idx, positive_idx.size, replace=True),
        rng.choice(negative_idx, negative_idx.size, replace=True),
    ])


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def read_slice(slice_dir: Path, dataset: str) -> pd.DataFrame | None:
    """Whole slice, every split. No model is involved in the main test, so there is nothing
    train/test leakage could contaminate."""
    path = slice_dir / f"{dataset}.parquet"
    if not path.is_file():
        return None
    return pd.read_parquet(path).reset_index().rename(columns={"index": "original_index"})


def matched_subset(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rows where BOTH frequencies exist -- the only place the question can be asked."""
    def column(name: str) -> pd.Series:
        if name not in frame.columns:
            return pd.Series([np.nan] * len(frame), index=frame.index, dtype=float)
        return pd.to_numeric(frame[name], errors="coerce")

    af_abraom = column("af_abraom")
    af_gnomad = column("af_gnomad")
    labels = column("label")
    mask = af_abraom.notna() & af_gnomad.notna() & labels.notna()
    return (
        labels[mask].to_numpy(dtype=np.int64),
        (1.0 - af_abraom[mask]).to_numpy(dtype=np.float64),
        (1.0 - af_gnomad[mask]).to_numpy(dtype=np.float64),
    )


def compare_scores(
    labels: np.ndarray, abraom: np.ndarray, gnomad: np.ndarray, *,
    resamples: int, rng: np.random.Generator,
) -> dict:
    entry = {
        "n": int(labels.size),
        "n_pathogenic": int((labels == 1).sum()),
        "auroc_abraom": auroc(labels, abraom),
        "auroc_gnomad": auroc(labels, gnomad),
        "auroc_abraom_legacy": float(binary_roc_auc(labels, abraom)) if labels.size else float("nan"),
        "auroc_gnomad_legacy": float(binary_roc_auc(labels, gnomad)) if labels.size else float("nan"),
    }
    entry["delta"] = entry["auroc_abraom"] - entry["auroc_gnomad"]
    if resamples > 0 and np.unique(labels).size > 1:
        draws = np.empty(resamples)
        for i in range(resamples):
            idx = stratified_indices(labels, rng)
            draws[i] = auroc(labels[idx], abraom[idx]) - auroc(labels[idx], gnomad[idx])
        entry["ci_low"], entry["ci_high"] = bc_interval(draws, entry["delta"])
        entry["p_gnomad_at_least_abraom"] = float(np.mean(draws[np.isfinite(draws)] <= 0.0))
        entry["_draws"] = draws
    return entry


def difference_in_differences(
    brazilian: dict, control: dict, *, rng: np.random.Generator
) -> dict | None:
    br_draws = brazilian.get("_draws")
    ctrl_draws = control.get("_draws")
    if br_draws is None or ctrl_draws is None:
        return None
    size = min(br_draws.size, ctrl_draws.size)
    # the two variant sets are disjoint, so their bootstrap draws are independent
    draws = br_draws[:size] - rng.permutation(ctrl_draws)[:size]
    observed = brazilian["delta"] - control["delta"]
    low, high = bc_interval(draws, observed)
    return {
        "did": observed,
        "delta_brazilian": brazilian["delta"],
        "delta_control": control["delta"],
        "ci_low": low,
        "ci_high": high,
        "p_no_regional_edge": float(np.mean(draws[np.isfinite(draws)] <= 0.0)),
    }


# ---------------------------------------------------------------------------
# [3] constrained operating points
# ---------------------------------------------------------------------------

def constrained_operating_points(
    args: argparse.Namespace, config: dict,
) -> list[dict]:
    """Maximize br_only MCC on holdout SUBJECT TO P/LP recall >= floor, then report on test."""
    from scripts.rebaseline_regional_against_frequency import build_arms  # local import: same repo

    rows: list[dict] = []
    frames: dict[tuple[str, str], pd.DataFrame] = {}
    for dataset in ("br_only", FOUNDER_SLICE, BENIGN_SLICE):
        for split in ("holdout", "test"):
            frame = load_split(args.m5_dir, args.slice_dir, dataset, split)
            if frame is not None:
                frames[(dataset, split)] = frame
    if ("br_only", "holdout") not in frames or (FOUNDER_SLICE, "holdout") not in frames:
        return rows

    m0_frames = {
        (dataset, split): align_to(frame, load_split(args.m0_dir, args.slice_dir, dataset, split))
        for (dataset, split), frame in frames.items()
    }

    arm_names = ["freq_abraom", "M0", "M5_molecular", "M5_v2"]
    for name in arm_names:
        def scores_for(dataset: str, split: str) -> np.ndarray | None:
            frame = frames.get((dataset, split))
            if frame is None:
                return None
            arms = build_arms(frame, m0_frames.get((dataset, split)), config)
            return arms.get(name)

        br_holdout = frames.get(("br_only", "holdout"))
        br_scores = scores_for("br_only", "holdout")
        plp_scores = scores_for(FOUNDER_SLICE, "holdout")
        if br_scores is None or plp_scores is None or br_holdout is None:
            continue
        br_labels = pd.to_numeric(br_holdout["label"], errors="coerce").fillna(-1).to_numpy(dtype=np.int64)
        plp_labels = pd.to_numeric(
            frames[(FOUNDER_SLICE, "holdout")]["label"], errors="coerce"
        ).fillna(-1).to_numpy(dtype=np.int64)

        best_threshold, best_mcc = None, -2.0
        candidates = np.unique(np.round(np.quantile(br_scores[np.isfinite(br_scores)],
                                                    np.linspace(0, 1, 400)), 6))
        for threshold in candidates:
            plp_recall = classification_metrics(plp_labels, plp_scores, float(threshold))["recall"]
            if plp_recall < args.plp_recall_floor:
                continue
            mcc = classification_metrics(br_labels, br_scores, float(threshold))["mcc"]
            if mcc > best_mcc:
                best_threshold, best_mcc = float(threshold), float(mcc)
        if best_threshold is None:
            rows.append({"arm": name, "feasible": False,
                         "note": f"no threshold reaches P/LP recall >= {args.plp_recall_floor}"})
            continue

        entry = {"arm": name, "feasible": True, "threshold": best_threshold,
                 "holdout_br_mcc": best_mcc}
        for dataset, key in ((("br_only"), "test_br_mcc"), (FOUNDER_SLICE, "test_plp_recall"),
                             (BENIGN_SLICE, "test_benign_spec")):
            scores = scores_for(dataset, "test")
            frame = frames.get((dataset, "test"))
            if scores is None or frame is None:
                continue
            labels = pd.to_numeric(frame["label"], errors="coerce").fillna(-1).to_numpy(dtype=np.int64)
            metrics = classification_metrics(labels, scores, best_threshold)
            entry[key] = float(metrics["mcc"] if key.endswith("mcc")
                               else metrics["recall"] if key.endswith("recall")
                               else metrics["specificity"])
        rows.append(entry)
    return rows


def _fmt(value, digits: int = 3) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if not np.isfinite(value) else f"{value:.{digits}f}"


def render(payload: dict) -> str:
    lines: list[str] = []
    lines.append("=" * 96)
    lines.append("ABRAOM vs gnomAD on Brazilian variants -- powered test + tie-aware AUROC")
    lines.append("=" * 96)

    lines.append("")
    lines.append("[0] AUROC implementation check (tie-aware Mann-Whitney vs the legacy function)")
    lines.append("")
    lines.append(f"    {'slice':<22}{'score':<10}{'tie-aware':>12}{'legacy':>10}{'gap':>9}")
    lines.append("    " + "-" * 63)
    for row in payload["comparisons"]:
        for source in ("abraom", "gnomad"):
            fixed = row[f"auroc_{source}"]
            legacy = row[f"auroc_{source}_legacy"]
            lines.append(f"    {row['slice']:<22}{source:<10}{_fmt(fixed):>12}{_fmt(legacy):>10}"
                         f"{_fmt(fixed - legacy):>9}")

    lines.append("")
    lines.append("[1] ABRAOM vs gnomAD, ALL rows of each slice (matched subset only)")
    lines.append("")
    lines.append(f"    {'slice':<22}{'n':>7}{'n_path':>8}{'abraom':>9}{'gnomad':>9}{'delta':>9}{'95% CI (BC)':>22}")
    lines.append("    " + "-" * 86)
    for row in payload["comparisons"]:
        ci = f"[{_fmt(row.get('ci_low'))}, {_fmt(row.get('ci_high'))}]"
        lines.append(
            f"    {row['slice']:<22}{row['n']:>7}{row['n_pathogenic']:>8}{_fmt(row['auroc_abraom']):>9}"
            f"{_fmt(row['auroc_gnomad']):>9}{_fmt(row['delta']):>9}{ci:>22}"
        )

    if payload.get("did"):
        lines.append("")
        lines.append("[2] Difference-in-differences -- the regional specificity test")
        lines.append("    (does ABRAOM's edge over gnomAD GROW on Brazilian variants?)")
        lines.append("")
        for key, row in payload["did"].items():
            lines.append(f"    {key}")
            lines.append(f"        delta on Brazilian     : {_fmt(row['delta_brazilian'])}")
            lines.append(f"        delta on non-Brazilian : {_fmt(row['delta_control'])}")
            lines.append(f"        DiD                    : {_fmt(row['did'])}  "
                         f"95% CI [{_fmt(row['ci_low'])}, {_fmt(row['ci_high'])}]")
            lines.append(f"        p(no regional edge)    : {_fmt(row['p_no_regional_edge'])}")

    if payload.get("operating_points"):
        lines.append("")
        lines.append(f"[3] Best br_only SUBJECT TO P/LP recall >= {payload['plp_recall_floor']}")
        lines.append("    (threshold chosen on holdout under the constraint, reported on test)")
        lines.append("")
        lines.append(f"    {'arm':<18}{'thr':>8}{'br_MCC':>9}{'P/LP recall':>13}{'benign spec':>13}")
        lines.append("    " + "-" * 61)
        for row in payload["operating_points"]:
            if not row.get("feasible", False):
                lines.append(f"    {row['arm']:<18}  INFEASIBLE -- {row.get('note', '')}")
                continue
            lines.append(
                f"    {row['arm']:<18}{_fmt(row['threshold']):>8}{_fmt(row.get('test_br_mcc')):>9}"
                f"{_fmt(row.get('test_plp_recall')):>13}{_fmt(row.get('test_benign_spec')):>13}"
            )

    lines.append("")
    lines.append("=" * 96)
    lines.append("READING [2]: DiD > 0 with a CI clear of zero = ABRAOM carries Brazilian-specific")
    lines.append("signal, even if it loses to gnomAD in absolute terms everywhere. DiD ~ 0 = the")
    lines.append("Brazilian frequency adds nothing a global frequency source does not already give.")
    lines.append("")
    lines.append("CAVEAT: 'Brazilian' here means Brazilian SUBMITTER in ClinVar (the project's own")
    lines.append("br_only/nonbr_only definition), not Brazilian ancestry of the carrier. The control")
    lines.append("slice is non-Brazilian-submitted variants that are still present in a Brazilian")
    lines.append("cohort, so the DiD is a conservative test of submitter-linked regional signal.")
    lines.append("=" * 96)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rng = np.random.default_rng(args.seed)

    config: dict = {"discount_scale": 1.0, "max_discount": 1.5, "regional_threshold": 0.5, "global_threshold": 0.5}
    if args.m5_v2_config.is_file():
        config.update(json.loads(args.m5_v2_config.read_text(encoding="utf-8")))

    comparisons: list[dict] = []
    by_slice: dict[str, dict] = {}
    for dataset in BRAZILIAN_SLICES + CONTROL_SLICES + ["abraom_present"]:
        frame = read_slice(args.slice_dir, dataset)
        if frame is None:
            print(f"[warn] slice not found: {dataset}", file=sys.stderr)
            continue
        labels, abraom, gnomad = matched_subset(frame)
        if labels.size == 0 or np.unique(labels).size < 2:
            print(f"[warn] {dataset}: no usable matched subset (n={labels.size})", file=sys.stderr)
            continue
        entry = compare_scores(labels, abraom, gnomad, resamples=args.bootstrap, rng=rng)
        entry["slice"] = dataset
        by_slice[dataset] = entry
        comparisons.append(entry)

    did: dict[str, dict] = {}
    for brazilian in BRAZILIAN_SLICES:
        for control in CONTROL_SLICES:
            if brazilian in by_slice and control in by_slice:
                result = difference_in_differences(by_slice[brazilian], by_slice[control], rng=rng)
                if result:
                    did[f"{brazilian} vs {control}"] = result

    operating_points = constrained_operating_points(args, config)

    for entry in comparisons:
        entry.pop("_draws", None)

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "bootstrap_resamples": args.bootstrap,
        "plp_recall_floor": args.plp_recall_floor,
        "comparisons": comparisons,
        "did": did,
        "operating_points": operating_points,
    }

    report = render(payload)
    print(report)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "abraom_vs_gnomad.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (args.output_dir / "abraom_vs_gnomad.txt").write_text(report, encoding="utf-8")
    print(f"\nwrote {args.output_dir}/abraom_vs_gnomad.{{json,txt}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
