#!/usr/bin/env python3
"""Where does the br_only gain (0.238 -> 0.574) actually come from?

Runs LOCAL on existing prediction parquets. No GPU, no job, no training.

THE QUESTION
------------
The regionalization headline is that calibration lifts `br_only` MCC from 0.238 (M0) to 0.574
(M5_v2). That gain is attributed to the ABRAOM frequency discount. But the M5_v2 calibration
changes TWO things at once:

    1. the SCORE      -- subtracts a frequency discount:
                         sigmoid(logit(molecular) - min(discount * scale, max_discount))
    2. the THRESHOLD  -- regional decisions move from ~0.5 to the tuned regional_threshold (0.235)

Those are separable, and they are not equally interesting. A threshold shift is free
recalibration that any model can have; a frequency discount is the actual regionalization claim.

There is a strong reason to suspect the threshold does much of the work: the diagnostic in
scripts/diagnose_molecular_contamination.py showed that in `br_only` (test) only ~71 of 504
variants carry any allele frequency at all. `af_gnomad`/`af_abraom` are populated ONLY for
variants matched into the ABRAOM index (scripts/prepare_regional_clinvar_dataset.py:348-349,
set to NA at :371). For the other ~86% the discount has nothing to fire on -- yet they are all
re-decided at the new, much lower regional threshold.

WHAT THIS SCRIPT MEASURES
-------------------------
[A] Coverage audit -- per slice: how many variants actually carry a frequency, and what the
    learned `regional_discount` looks like for the ones that do not.

[B] Arm comparison under the M5_v2 protocol -- threshold tuned on HOLDOUT, applied to TEST.
    Tuning each arm's threshold on the same split we report would flatter every arm and destroy
    the comparison, so we mirror exactly what the real calibration did.
        M0              : the frequency-blind baseline
        M5_molecular    : bounded head's molecular_probability, NO post-hoc discount
        M5_regional_raw : bounded head's raw output (molecular - learned discount)
        M5_v2           : the lead, reconstructed with the selected config

    M0 @ tuned threshold is the counterfactual that matters most: if a plain M0 with nothing but
    a re-tuned threshold already reaches most of 0.574, then most of the headline is
    recalibration, not regionalization.

[C] Subgroup split -- every arm recomputed separately on the variants WITH a frequency and
    WITHOUT one. If the gain is concentrated in the ~14% with ABRAOM data, the mechanism is
    working as designed but on a small minority; if it is uniform across both subgroups, the
    gain cannot be coming from the frequency discount at all.

This script does not prove or disprove the falsification result (p=0.0196), which tested a
different thing: whether the discount tracks specific variants. It tests how much of the
headline METRIC the discount is responsible for.

USAGE
-----
    python scripts/analyze_regional_gain_decomposition.py \
        --m5-dir     ~/v11eval/reval_m5bounded \
        --m0-dir     ~/v11eval/m5v3_modelroot/m0_nonbr_beatv10_v1_sagemaker \
        --slice-dir  ~/slices \
        --m5-v2-config ~/v11eval/m5_v2_v11/selected_config.json \
        --output-dir ~/v11eval/diag_gain_decomposition
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
from scripts.calibrate_m5_v3_safety import load_metadata  # noqa: E402

EPS = 1e-6

# Fallback if selected_config.json is absent: the v11 M5_v2 config from the handoff.
DEFAULT_DISCOUNT_SCALE = 1.0
DEFAULT_MAX_DISCOUNT = 1.5


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    home = Path.home()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--m5-dir", type=Path, default=home / "v11eval/reval_m5bounded")
    parser.add_argument("--m0-dir", type=Path,
                        default=home / "v11eval/m5v3_modelroot/m0_nonbr_beatv10_v1_sagemaker")
    parser.add_argument("--slice-dir", type=Path, default=home / "slices")
    parser.add_argument("--m5-v2-config", type=Path, default=home / "v11eval/m5_v2_v11/selected_config.json")
    # nonbr_only matters as much as the Brazilian slices here: it is the TRAINING distribution,
    # so its frequency coverage tells us how much of the data the model ever saw an AF for.
    parser.add_argument("--datasets", nargs="*",
                        default=["br_only", "br_any", "abraom_common_benign", "abraom_pathogenic_present",
                                 "nonbr_only", "global_nonbr_no_abraom"])
    parser.add_argument("--tune-split", default="holdout")
    parser.add_argument("--report-split", default="test")
    parser.add_argument("--output-dir", type=Path, default=home / "v11eval/diag_gain_decomposition")
    return parser.parse_args(argv)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def _logit(p: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0 - EPS)
    return np.log(clipped / (1.0 - clipped))


def load_split(model_dir: Path, slice_dir: Path, dataset: str, split: str) -> pd.DataFrame | None:
    path = model_dir / f"{dataset}.{split}.predictions.parquet"
    if not path.is_file():
        return None
    predictions = pd.read_parquet(path).copy()
    metadata = load_metadata(slice_dir, dataset, split)
    return predictions.merge(metadata, on="original_index", how="left", validate="one_to_one").reset_index(drop=True)


def align_to(reference: pd.DataFrame | None, other: pd.DataFrame | None) -> pd.DataFrame | None:
    """Reindex `other` onto `reference`'s variants, same order.

    The subgroup analysis applies a mask derived from the M5 frame to the M0 arm, so the two must
    be row-aligned -- equal length is not enough, the order has to match too. The two evals were
    separate jobs, so this is an assumption worth enforcing rather than trusting.
    """
    if reference is None or other is None:
        return other
    indexed = other.set_index("original_index")
    common = [idx for idx in reference["original_index"].to_numpy() if idx in indexed.index]
    return indexed.loc[common].reset_index()


def regional_score(frame: pd.DataFrame, discount_scale: float, max_discount: float) -> np.ndarray:
    """Exact reproduction of _regional_score in scripts/calibrate_m5_v2_regional_scores.py:91."""
    discount = np.minimum(
        pd.to_numeric(frame["regional_discount"], errors="coerce").to_numpy(dtype=np.float64) * discount_scale,
        max_discount,
    )
    return _sigmoid(_logit(pd.to_numeric(frame["molecular_probability"], errors="coerce").to_numpy()) - discount)


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------

def build_arms(
    m5: pd.DataFrame | None, m0: pd.DataFrame | None, discount_scale: float, max_discount: float
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """name -> (labels, scores). Every arm is on its own frame's variants."""
    arms: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if m0 is not None and "probability" in m0.columns:
        arms["M0"] = (
            pd.to_numeric(m0["label"], errors="coerce").to_numpy(dtype=np.int64),
            pd.to_numeric(m0["probability"], errors="coerce").to_numpy(dtype=np.float64),
        )
    if m5 is not None:
        labels = pd.to_numeric(m5["label"], errors="coerce").to_numpy(dtype=np.int64)
        if "molecular_probability" in m5.columns:
            arms["M5_molecular"] = (
                labels, pd.to_numeric(m5["molecular_probability"], errors="coerce").to_numpy(dtype=np.float64)
            )
            if "regional_discount" in m5.columns:
                arms["M5_v2"] = (labels, regional_score(m5, discount_scale, max_discount))
        if "probability" in m5.columns:
            arms["M5_regional_raw"] = (
                labels, pd.to_numeric(m5["probability"], errors="coerce").to_numpy(dtype=np.float64)
            )
    return arms


def tune_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Pick the MCC-maximizing threshold. Returns (threshold, mcc_at_that_threshold)."""
    finite = np.isfinite(scores)
    labels, scores = labels[finite], scores[finite]
    if labels.size == 0 or np.unique(labels).size < 2:
        return 0.5, float("nan")
    best_threshold, best_mcc = 0.5, -2.0
    for threshold in np.round(np.arange(0.01, 1.00, 0.005), 4):
        mcc = classification_metrics(labels, scores, float(threshold))["mcc"]
        if mcc > best_mcc:
            best_threshold, best_mcc = float(threshold), float(mcc)
    return best_threshold, best_mcc


def score_at(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    finite = np.isfinite(scores)
    labels, scores = labels[finite], scores[finite]
    if labels.size == 0:
        return {"n": 0, "mcc": float("nan"), "recall": float("nan"), "specificity": float("nan")}
    metrics = classification_metrics(labels, scores, float(threshold))
    return {
        "n": int(labels.size),
        "n_positive": int(np.sum(labels == 1)),
        "mcc": float(metrics["mcc"]),
        "recall": float(metrics["recall"]),
        "specificity": float(metrics["specificity"]),
    }


# ---------------------------------------------------------------------------
# [A] coverage
# ---------------------------------------------------------------------------

def coverage_audit(frame: pd.DataFrame, dataset: str, split: str) -> dict:
    n = len(frame)
    present = pd.to_numeric(frame.get("abraom_present", pd.Series([np.nan] * n)), errors="coerce").fillna(0) > 0
    af_abraom = pd.to_numeric(frame.get("af_abraom", pd.Series([np.nan] * n)), errors="coerce")
    af_gnomad = pd.to_numeric(frame.get("af_gnomad", pd.Series([np.nan] * n)), errors="coerce")
    discount = pd.to_numeric(frame.get("regional_discount", pd.Series([np.nan] * n)), errors="coerce")
    entry = {
        "dataset": dataset,
        "split": split,
        "n": n,
        "n_abraom_present": int(present.sum()),
        "pct_abraom_present": float(100.0 * present.sum() / n) if n else float("nan"),
        "n_af_abraom": int(af_abraom.notna().sum()),
        "n_af_gnomad": int(af_gnomad.notna().sum()),
    }
    if discount.notna().any():
        entry["discount_median_with_af"] = float(discount[present].median()) if present.any() else float("nan")
        entry["discount_median_without_af"] = float(discount[~present].median()) if (~present).any() else float("nan")
        entry["discount_max_without_af"] = float(discount[~present].max()) if (~present).any() else float("nan")
    return entry


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt(value, digits: int = 3) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if not np.isfinite(value) else f"{value:.{digits}f}"


def render(payload: dict) -> str:
    lines: list[str] = []
    lines.append("=" * 92)
    lines.append("WHERE DOES THE br_only GAIN COME FROM? -- threshold vs frequency discount")
    lines.append(f"tuned on '{payload['tune_split']}', reported on '{payload['report_split']}'"
                 f"   config: scale={payload['discount_scale']}, max={payload['max_discount']}")
    lines.append("=" * 92)

    lines.append("")
    lines.append("[A] Frequency coverage -- how many variants can the discount even fire on?")
    lines.append("")
    lines.append(f"    {'slice':<28}{'split':<10}{'n':>7}{'abraom%':>10}{'af_abraom':>11}{'af_gnomad':>11}")
    lines.append("    " + "-" * 77)
    for row in payload["coverage"]:
        lines.append(
            f"    {row['dataset']:<28}{row['split']:<10}{row['n']:>7}"
            f"{_fmt(row['pct_abraom_present'], 1):>10}{row['n_af_abraom']:>11}{row['n_af_gnomad']:>11}"
        )
    for row in payload["coverage"]:
        if "discount_max_without_af" in row and np.isfinite(row.get("discount_max_without_af", np.nan)):
            lines.append(
                f"      {row['dataset']}/{row['split']}: learned discount without AF -> "
                f"median {_fmt(row['discount_median_without_af'])}, max {_fmt(row['discount_max_without_af'])}"
                f" | with AF -> median {_fmt(row['discount_median_with_af'])}"
            )

    lines.append("")
    lines.append("[B] Arms under the M5_v2 protocol (threshold tuned on holdout, applied to test)")
    for dataset, arms in payload["arms"].items():
        lines.append("")
        lines.append(f"    --- {dataset} ---")
        lines.append(f"    {'arm':<20}{'thr(tuned)':>12}{'MCC':>9}{'recall':>9}{'spec':>9}{'n':>7}")
        lines.append("    " + "-" * 66)
        for name, row in arms.items():
            lines.append(
                f"    {name:<20}{_fmt(row['threshold']):>12}{_fmt(row['mcc']):>9}"
                f"{_fmt(row['recall']):>9}{_fmt(row['specificity']):>9}{row['n']:>7}"
            )

    if payload.get("subgroups"):
        lines.append("")
        lines.append("[C] Same arms, split by whether the variant HAS a frequency at all")
        for dataset, groups in payload["subgroups"].items():
            lines.append("")
            lines.append(f"    --- {dataset} ---")
            lines.append(f"    {'subgroup':<22}{'arm':<20}{'MCC':>9}{'n':>7}{'n_pos':>8}")
            lines.append("    " + "-" * 66)
            for group_name, arms in groups.items():
                for name, row in arms.items():
                    lines.append(
                        f"    {group_name:<22}{name:<20}{_fmt(row['mcc']):>9}{row['n']:>7}{row.get('n_positive', 0):>8}"
                    )

    attribution = payload.get("attribution", {})
    if attribution:
        lines.append("")
        lines.append("=" * 92)
        lines.append("ATTRIBUTION on br_only (all deltas at honestly-tuned thresholds)")
        for key, value in attribution.items():
            lines.append(f"    {key:<52}{_fmt(value)}")
        lines.append("=" * 92)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    discount_scale, max_discount = DEFAULT_DISCOUNT_SCALE, DEFAULT_MAX_DISCOUNT
    if args.m5_v2_config.is_file():
        config = json.loads(args.m5_v2_config.read_text(encoding="utf-8"))
        discount_scale = float(config.get("discount_scale", discount_scale))
        max_discount = float(config.get("max_discount", max_discount))
        print(f"[info] loaded M5_v2 config from {args.m5_v2_config}: "
              f"scale={discount_scale}, max_discount={max_discount}")
    else:
        print(f"[warn] {args.m5_v2_config} not found -- falling back to scale={discount_scale}, "
              f"max_discount={max_discount}", file=sys.stderr)

    coverage: list[dict] = []
    arms_out: dict[str, dict] = {}
    subgroups_out: dict[str, dict] = {}

    for dataset in args.datasets:
        frames = {}
        for split in (args.tune_split, args.report_split):
            m5 = load_split(args.m5_dir, args.slice_dir, dataset, split)
            m0 = align_to(m5, load_split(args.m0_dir, args.slice_dir, dataset, split))
            frames[split] = (m5, m0)
            if m5 is not None:
                coverage.append(coverage_audit(m5, dataset, split))

        m5_tune, m0_tune = frames[args.tune_split]
        m5_report, m0_report = frames[args.report_split]
        if m5_report is None and m0_report is None:
            continue

        tune_arms = build_arms(m5_tune, m0_tune, discount_scale, max_discount)
        report_arms = build_arms(m5_report, m0_report, discount_scale, max_discount)

        dataset_rows: dict[str, dict] = {}
        thresholds: dict[str, float] = {}
        for name, (labels, scores) in report_arms.items():
            if name in tune_arms:
                threshold, _ = tune_threshold(*tune_arms[name])
            else:
                threshold = 0.5
            thresholds[name] = threshold
            row = score_at(labels, scores, threshold)
            row["threshold"] = threshold
            dataset_rows[name] = row
        arms_out[dataset] = dataset_rows

        # [C] subgroup split, reusing the SAME thresholds (no re-tuning per subgroup)
        if m5_report is not None and "abraom_present" in m5_report.columns:
            present_mask = pd.to_numeric(m5_report["abraom_present"], errors="coerce").fillna(0).to_numpy() > 0
            groups: dict[str, dict] = {}
            for group_name, mask in (("has_frequency", present_mask), ("no_frequency", ~present_mask)):
                group_rows: dict[str, dict] = {}
                for name, (labels, scores) in report_arms.items():
                    if labels.size != mask.size:
                        continue  # M0 frame may not align row-for-row; skip rather than mislead
                    group_rows[name] = score_at(labels[mask], scores[mask], thresholds[name])
                if group_rows:
                    groups[group_name] = group_rows
            if groups:
                subgroups_out[dataset] = groups

    attribution: dict[str, float] = {}
    br = arms_out.get("br_only", {})
    if {"M0", "M5_molecular", "M5_v2"} <= set(br):
        m0_mcc = br["M0"]["mcc"]
        mol_mcc = br["M5_molecular"]["mcc"]
        v2_mcc = br["M5_v2"]["mcc"]
        attribution["M0 @ tuned threshold"] = m0_mcc
        attribution["M5_v2 @ tuned threshold (the lead)"] = v2_mcc
        attribution["total gain M0 -> M5_v2"] = v2_mcc - m0_mcc
        attribution["  of which: bounded model, no post-hoc discount"] = mol_mcc - m0_mcc
        attribution["  of which: the post-hoc frequency discount"] = v2_mcc - mol_mcc
        if np.isfinite(v2_mcc - m0_mcc) and abs(v2_mcc - m0_mcc) > 1e-9:
            attribution["discount share of total gain (%)"] = 100.0 * (v2_mcc - mol_mcc) / (v2_mcc - m0_mcc)

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "tune_split": args.tune_split,
        "report_split": args.report_split,
        "discount_scale": discount_scale,
        "max_discount": max_discount,
        "coverage": coverage,
        "arms": arms_out,
        "subgroups": subgroups_out,
        "attribution": attribution,
    }

    report = render(payload)
    print(report)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "gain_decomposition.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (args.output_dir / "gain_decomposition.txt").write_text(report, encoding="utf-8")
    print(f"\nwrote {args.output_dir}/gain_decomposition.{{json,txt}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
