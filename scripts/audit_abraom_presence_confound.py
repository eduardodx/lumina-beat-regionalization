#!/usr/bin/env python3
"""Is the br_only headline explained by a single bit -- "is this variant in the ABRAOM index"?

Runs LOCAL on existing prediction parquets. No GPU, no job, no training.

WHY THIS EXISTS
---------------
scripts/analyze_regional_gain_decomposition.py showed that in `br_only` (test) all of the
performance lives in the 86% of variants that carry NO allele frequency, while the 14% that do
carry one score a NEGATIVE MCC. It also showed the class composition is inverted between those
groups:

    br_only     present in ABRAOM ->   6 / 71  pathogenic  (8%)
                absent            -> 371 / 433 pathogenic  (86%)
    nonbr_only  present           ->   7 / 558 pathogenic  (1.3%)
                absent            -> 1348 / 1989 pathogenic (68%)

So "present in ABRAOM" is very nearly a benign label, and `abraom_present` plus the three
`*_missing` flags are 4 of the 11 explicit features the bounded head receives. A hand
calculation on the printed counts suggests the trivial rule

    present in ABRAOM => benign ;  absent => pathogenic

reaches br_only MCC ~= 0.62 -- ABOVE the published M5_v2 (0.574). This script checks that
properly, on both splits, and asks whether the model adds anything on top of it.

THE CONFOUND INSIDE THE CONFOUND
--------------------------------
`build_abraom_matches` (scripts/prepare_regional_clinvar_dataset.py:324-327) matches ONLY
`is_snv` rows. Indels and MNVs therefore have `abraom_present = False` by construction, and
indels are strongly enriched for pathogenic (frameshift, etc.). So the presence bit partly
encodes VARIANT TYPE, not population frequency -- and `is_snv` is itself one of the 11 explicit
features.

That makes the fair question: restricted to SNVs only -- the variants that could actually have
matched -- how much does the presence bit still explain? This script reports both.

WHAT IT MEASURES
----------------
[A] Composition: pathogenic rate by (abraom_present x is_snv), per slice and split.
[B] Trivial rules, no model at all:
        rule_presence      : pathogenic <=> NOT abraom_present
        rule_not_snv       : pathogenic <=> NOT is_snv          (how much is just variant type?)
        rule_presence_snvs : rule_presence restricted to SNVs   (the fair version)
[C] The model arms (M0 / M5_molecular / M5_v2), thresholds tuned on holdout, same variants.
[D] Paired bootstrap of MCC(model) - MCC(trivial rule) on identical variants. This is the
    decisive number: if the CI crosses zero, the whole pipeline adds nothing over one bit.
[E] The allele-frequency distribution of matched variants, to see whether the ABRAOM index was
    itself pre-filtered to common variants (which would make presence == common by construction).

USAGE
-----
    python scripts/audit_abraom_presence_confound.py \
        --m5-dir     ~/v11eval/reval_m5bounded \
        --m0-dir     ~/v11eval/m5v3_modelroot/m0_nonbr_beatv10_v1_sagemaker \
        --slice-dir  ~/slices \
        --output-dir ~/v11eval/diag_presence_confound
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
from scripts.analyze_regional_gain_decomposition import (  # noqa: E402
    DEFAULT_DISCOUNT_SCALE,
    DEFAULT_MAX_DISCOUNT,
    align_to,
    build_arms,
    load_split,
    tune_threshold,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    home = Path.home()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--m5-dir", type=Path, default=home / "v11eval/reval_m5bounded")
    parser.add_argument("--m0-dir", type=Path,
                        default=home / "v11eval/m5v3_modelroot/m0_nonbr_beatv10_v1_sagemaker")
    parser.add_argument("--slice-dir", type=Path, default=home / "slices")
    parser.add_argument("--m5-v2-config", type=Path, default=home / "v11eval/m5_v2_v11/selected_config.json")
    parser.add_argument("--datasets", nargs="*", default=["br_only", "br_any", "nonbr_only"])
    parser.add_argument("--splits", nargs="*", default=["test", "holdout"])
    parser.add_argument("--tune-split", default="holdout")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--output-dir", type=Path, default=home / "v11eval/diag_presence_confound")
    return parser.parse_args(argv)


def _bool_col(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns:
        return np.zeros(len(frame), dtype=bool)
    series = frame[column]
    if series.dtype == bool:
        return series.to_numpy()
    return pd.to_numeric(series, errors="coerce").fillna(0).to_numpy() > 0


def _labels(frame: pd.DataFrame) -> np.ndarray:
    return pd.to_numeric(frame["label"], errors="coerce").fillna(-1).to_numpy(dtype=np.int64)


def mcc(labels: np.ndarray, predictions: np.ndarray) -> float:
    """MCC from hard predictions (the trivial rules produce 0/1 directly, not scores)."""
    if labels.size == 0 or np.unique(labels).size < 2:
        return float("nan")
    return float(classification_metrics(labels, predictions.astype(np.float64), 0.5)["mcc"])


# ---------------------------------------------------------------------------
# [A] composition
# ---------------------------------------------------------------------------

def composition(frame: pd.DataFrame, dataset: str, split: str) -> list[dict]:
    labels = _labels(frame)
    present = _bool_col(frame, "abraom_present")
    is_snv = _bool_col(frame, "is_snv")
    rows: list[dict] = []
    for present_value in (True, False):
        for snv_value in (True, False):
            mask = (present == present_value) & (is_snv == snv_value)
            n = int(mask.sum())
            if n == 0:
                continue
            positives = int(np.sum(labels[mask] == 1))
            rows.append({
                "dataset": dataset,
                "split": split,
                "abraom_present": bool(present_value),
                "is_snv": bool(snv_value),
                "n": n,
                "n_pathogenic": positives,
                "pct_pathogenic": round(100.0 * positives / n, 1),
            })
    return rows


# ---------------------------------------------------------------------------
# [B] trivial rules
# ---------------------------------------------------------------------------

def trivial_rules(frame: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """name -> (labels, hard 0/1 predictions), each on its own variant subset."""
    labels = _labels(frame)
    present = _bool_col(frame, "abraom_present")
    is_snv = _bool_col(frame, "is_snv")
    rules: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "rule_presence": (labels, (~present).astype(np.int64)),
        "rule_not_snv": (labels, (~is_snv).astype(np.int64)),
    }
    if is_snv.any():
        rules["rule_presence_snvs"] = (labels[is_snv], (~present[is_snv]).astype(np.int64))
    return rules


# ---------------------------------------------------------------------------
# [D] paired bootstrap: model vs the one-bit rule, identical variants
# ---------------------------------------------------------------------------

def paired_bootstrap(
    labels: np.ndarray, model_scores: np.ndarray, threshold: float,
    rule_predictions: np.ndarray, *, resamples: int, rng: np.random.Generator,
) -> dict:
    model_hard = (model_scores >= threshold).astype(np.int64)
    observed_model = mcc(labels, model_hard)
    observed_rule = mcc(labels, rule_predictions)
    out = {
        "mcc_model": observed_model,
        "mcc_rule": observed_rule,
        "delta": observed_model - observed_rule,
        "ci_low": float("nan"),
        "ci_high": float("nan"),
        "n": int(labels.size),
    }
    if resamples <= 0 or labels.size == 0:
        return out
    draws = np.empty(resamples, dtype=np.float64)
    for i in range(resamples):
        idx = rng.integers(0, labels.size, labels.size)
        draws[i] = mcc(labels[idx], model_hard[idx]) - mcc(labels[idx], rule_predictions[idx])
    draws = draws[np.isfinite(draws)]
    if draws.size:
        out["ci_low"] = float(np.percentile(draws, 2.5))
        out["ci_high"] = float(np.percentile(draws, 97.5))
        out["p_rule_at_least_model"] = float(np.mean(draws <= 0.0))
    return out


# ---------------------------------------------------------------------------
# [E] is the ABRAOM index itself filtered to common variants?
# ---------------------------------------------------------------------------

def af_profile(frame: pd.DataFrame, dataset: str, split: str) -> dict | None:
    af = pd.to_numeric(frame.get("af_abraom", pd.Series(dtype=float)), errors="coerce").dropna()
    if af.empty:
        return None
    return {
        "dataset": dataset,
        "split": split,
        "n_matched": int(af.size),
        "min": float(af.min()),
        "p05": float(af.quantile(0.05)),
        "median": float(af.median()),
        "max": float(af.max()),
        "pct_below_1pct": float(100.0 * np.mean(af < 0.01)),
    }


def _fmt(value, digits: int = 3) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if not np.isfinite(value) else f"{value:.{digits}f}"


def render(payload: dict) -> str:
    lines: list[str] = []
    lines.append("=" * 94)
    lines.append("ABRAOM-PRESENCE CONFOUND AUDIT -- does one bit explain the br_only headline?")
    lines.append("=" * 94)

    lines.append("")
    lines.append("[A] Pathogenic rate by (in ABRAOM x is SNV)")
    lines.append("")
    lines.append(f"    {'slice':<14}{'split':<10}{'inABRAOM':<11}{'isSNV':<8}{'n':>8}{'n_path':>9}{'%path':>8}")
    lines.append("    " + "-" * 66)
    for row in payload["composition"]:
        lines.append(
            f"    {row['dataset']:<14}{row['split']:<10}{str(row['abraom_present']):<11}"
            f"{str(row['is_snv']):<8}{row['n']:>8}{row['n_pathogenic']:>9}{row['pct_pathogenic']:>8}"
        )

    lines.append("")
    lines.append("[B]+[C] Trivial rules vs the models (model thresholds tuned on holdout)")
    for key, rows in payload["scoreboard"].items():
        lines.append("")
        lines.append(f"    --- {key} ---")
        lines.append(f"    {'arm':<24}{'MCC':>9}{'n':>8}")
        lines.append("    " + "-" * 41)
        for name, row in rows.items():
            lines.append(f"    {name:<24}{_fmt(row['mcc']):>9}{row['n']:>8}")

    if payload.get("head_to_head"):
        lines.append("")
        lines.append("[D] Paired bootstrap -- model MINUS the one-bit presence rule, same variants")
        lines.append("")
        lines.append(f"    {'slice/split':<20}{'model':<18}{'MCC mod':>9}{'MCC bit':>9}{'delta':>9}{'95% CI':>22}")
        lines.append("    " + "-" * 87)
        for key, entries in payload["head_to_head"].items():
            for name, row in entries.items():
                ci = f"[{_fmt(row['ci_low'])}, {_fmt(row['ci_high'])}]"
                lines.append(
                    f"    {key:<20}{name:<18}{_fmt(row['mcc_model']):>9}{_fmt(row['mcc_rule']):>9}"
                    f"{_fmt(row['delta']):>9}{ci:>22}"
                )

    if payload.get("af_profile"):
        lines.append("")
        lines.append("[E] AF distribution of ABRAOM-matched variants (is the index pre-filtered to common?)")
        lines.append("")
        lines.append(f"    {'slice':<14}{'split':<10}{'n':>8}{'min':>10}{'p05':>10}{'median':>10}{'%AF<1%':>9}")
        lines.append("    " + "-" * 71)
        for row in payload["af_profile"]:
            lines.append(
                f"    {row['dataset']:<14}{row['split']:<10}{row['n_matched']:>8}{_fmt(row['min'], 5):>10}"
                f"{_fmt(row['p05'], 5):>10}{_fmt(row['median'], 4):>10}{_fmt(row['pct_below_1pct'], 1):>9}"
            )

    lines.append("")
    lines.append("=" * 94)
    lines.append("HOW TO READ [D]: if the CI crosses 0, the full pipeline is statistically")
    lines.append("indistinguishable from the single bit 'is this variant in the ABRAOM index'.")
    lines.append("=" * 94)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rng = np.random.default_rng(args.seed)

    discount_scale, max_discount = DEFAULT_DISCOUNT_SCALE, DEFAULT_MAX_DISCOUNT
    if args.m5_v2_config.is_file():
        config = json.loads(args.m5_v2_config.read_text(encoding="utf-8"))
        discount_scale = float(config.get("discount_scale", discount_scale))
        max_discount = float(config.get("max_discount", max_discount))

    composition_rows: list[dict] = []
    af_rows: list[dict] = []
    scoreboard: dict[str, dict] = {}
    head_to_head: dict[str, dict] = {}

    for dataset in args.datasets:
        # thresholds are always tuned on the tune split, never on the split being reported
        m5_tune = load_split(args.m5_dir, args.slice_dir, dataset, args.tune_split)
        m0_tune = align_to(m5_tune, load_split(args.m0_dir, args.slice_dir, dataset, args.tune_split))
        tune_arms = build_arms(m5_tune, m0_tune, discount_scale, max_discount)
        thresholds = {name: tune_threshold(*payload)[0] for name, payload in tune_arms.items()}

        for split in args.splits:
            m5 = load_split(args.m5_dir, args.slice_dir, dataset, split)
            if m5 is None:
                continue
            m0 = align_to(m5, load_split(args.m0_dir, args.slice_dir, dataset, split))
            composition_rows.extend(composition(m5, dataset, split))
            profile = af_profile(m5, dataset, split)
            if profile:
                af_rows.append(profile)

            key = f"{dataset}/{split}"
            rows: dict[str, dict] = {}

            rules = trivial_rules(m5)
            for name, (labels, predictions) in rules.items():
                rows[name] = {"mcc": mcc(labels, predictions), "n": int(labels.size)}

            report_arms = build_arms(m5, m0, discount_scale, max_discount)
            for name, (labels, scores) in report_arms.items():
                threshold = thresholds.get(name, 0.5)
                finite = np.isfinite(scores)
                rows[name] = {
                    "mcc": mcc(labels[finite], (scores[finite] >= threshold).astype(np.int64)),
                    "n": int(finite.sum()),
                    "threshold": threshold,
                }
            scoreboard[key] = rows

            # [D] paired against the one-bit rule, on identical variants
            rule_labels, rule_predictions = rules["rule_presence"]
            entries: dict[str, dict] = {}
            for name, (labels, scores) in report_arms.items():
                if labels.size != rule_labels.size:
                    continue
                finite = np.isfinite(scores)
                entries[name] = paired_bootstrap(
                    labels[finite], scores[finite], thresholds.get(name, 0.5),
                    rule_predictions[finite], resamples=args.bootstrap, rng=rng,
                )
            if entries:
                head_to_head[key] = entries

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "discount_scale": discount_scale,
        "max_discount": max_discount,
        "bootstrap_resamples": args.bootstrap,
        "composition": composition_rows,
        "scoreboard": scoreboard,
        "head_to_head": head_to_head,
        "af_profile": af_rows,
    }

    report = render(payload)
    print(report)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "presence_confound.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (args.output_dir / "presence_confound.txt").write_text(report, encoding="utf-8")
    print(f"\nwrote {args.output_dir}/presence_confound.{{json,txt}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
