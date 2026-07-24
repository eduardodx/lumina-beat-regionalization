#!/usr/bin/env python3
"""Re-baseline the regionalization against a FREQUENCY-ONLY classifier, not against M0.

Runs LOCAL on existing prediction parquets. No GPU, no job, no training.

WHY
---
Every regionalization claim in this project (v10 and v11 alike) is measured as a gain over M0,
the molecular baseline. That is the wrong reference. The mechanism being claimed IS a frequency
effect, so the honest question is not "do we beat a model that ignores frequency?" but "do we
beat simply using the frequency?".

The audit in scripts/audit_abraom_presence_confound.py showed why this matters: the single bit
"is this variant in the ABRAOM index" scores br_only MCC 0.619 (test) / 0.609 (holdout), ABOVE
the published M5_v2 lead of 0.574. And that bit is really a frequency threshold -- the ABRAOM v2
index has a hard floor at af_abraom ~= 0.00512, so "present" means "AF >= ~0.5% in Brazil".

TWO DESIGN DECISIONS THAT MAKE THIS FAIR
----------------------------------------
1. The baseline is a SCORE, not a hard rule. `rule_presence` has no threshold to tune, while
   every model arm gets its threshold tuned on holdout -- comparing them directly hands the
   models a free parameter the baseline does not have. So the primary baseline here is
   `freq_abraom` = 1 - af_abraom (missing AF -> 0 -> scores as "rare"), thresholded, tuned on
   holdout exactly like the models. It strictly dominates the bit rule (the bit is one of its
   achievable thresholds), so it is the harder, fairer opponent.

2. AUROC is reported alongside MCC. AUROC needs no threshold at all, so it sidesteps the entire
   tuning question. If a model cannot beat frequency-alone on AUROC, no threshold choice will
   save the claim.

WHAT IT REPORTS
---------------
[0] SELF-VALIDATION -- reproduces the PUBLISHED M5_v2 numbers from the raw predictions using the
    deployed selected_config.json. If this does not match the TCC table, nothing below can be
    trusted and the script says so loudly.
[1] Scoreboard: every arm x slice x split, MCC / recall / specificity / AUROC.
[2] Head-to-head: paired bootstrap of (model - frequency baseline) on identical variants, for
    both MCC and AUROC, both splits.
[3] The ABRAOM-vs-gnomAD question, restricted to the only variants where BOTH exist (the matched
    subset). This is the sharpest available test of "is ABRAOM adding anything over a global
    frequency source", and it is small -- reported with its n, not oversold.

USAGE
-----
    python scripts/rebaseline_regional_against_frequency.py \
        --m5-dir     ~/v11eval/reval_m5bounded \
        --m0-dir     ~/v11eval/m5v3_modelroot/m0_nonbr_beatv10_v1_sagemaker \
        --slice-dir  ~/slices \
        --m5-v2-config ~/v11eval/m5_v2_v11/selected_config.json \
        --output-dir ~/v11eval/rebaseline_frequency
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

from eval.clinvar.metrics import binary_roc_auc, classification_metrics  # noqa: E402
from scripts.analyze_regional_gain_decomposition import align_to, load_split  # noqa: E402

EPS = 1e-6

# Slices whose decisions use the regional score+threshold, mirroring
# calibrate_m5_v2_regional_scores.py:136-143.
REGIONAL_DATASETS = {
    "br_only", "br_any", "regional_benchmark_any",
    "abraom_common_benign", "abraom_pathogenic_present", "abraom_pathogenic_common",
}
GLOBAL_DATASETS = {"global_nonbr_no_abraom", "nonbr_only"}

# Published v11 M5_v2 numbers (test split) from TCC_REGIONALIZACAO_V11.md 10.5, used to prove the
# reconstruction is faithful before any new claim is built on it.
PUBLISHED_TEST = {
    ("br_only", "mcc"): 0.574,
    ("abraom_common_benign", "specificity"): 0.951,
    ("abraom_pathogenic_present", "recall"): 0.405,
    ("global_nonbr_no_abraom", "mcc"): 0.626,
}
PUBLISHED_TOLERANCE = 0.02

PRIMARY_BASELINE = "freq_abraom"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    home = Path.home()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--m5-dir", type=Path, default=home / "v11eval/reval_m5bounded")
    parser.add_argument("--m0-dir", type=Path,
                        default=home / "v11eval/m5v3_modelroot/m0_nonbr_beatv10_v1_sagemaker")
    parser.add_argument("--slice-dir", type=Path, default=home / "slices")
    parser.add_argument("--m5-v2-config", type=Path, default=home / "v11eval/m5_v2_v11/selected_config.json")
    parser.add_argument("--datasets", nargs="*",
                        default=["br_only", "br_any", "abraom_common_benign", "abraom_pathogenic_present",
                                 "nonbr_only", "global_nonbr_no_abraom"])
    parser.add_argument("--splits", nargs="*", default=["test", "holdout"])
    parser.add_argument("--tune-split", default="holdout")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--output-dir", type=Path, default=home / "v11eval/rebaseline_frequency")
    return parser.parse_args(argv)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def _logit(p: np.ndarray) -> np.ndarray:
    return np.log(np.clip(p, EPS, 1 - EPS) / (1 - np.clip(p, EPS, 1 - EPS)))


def _num(frame: pd.DataFrame, column: str, fill: float = np.nan) -> np.ndarray:
    if column not in frame.columns:
        return np.full(len(frame), fill, dtype=np.float64)
    return pd.to_numeric(frame[column], errors="coerce").fillna(fill).to_numpy(dtype=np.float64)


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------

def build_arms(m5: pd.DataFrame, m0: pd.DataFrame | None, config: dict) -> dict[str, np.ndarray]:
    """name -> score aligned to m5's rows. Higher score = more pathogenic."""
    arms: dict[str, np.ndarray] = {}

    # --- frequency-only baselines -------------------------------------------------
    # Missing AF fills to 0.0, i.e. "not observed as common in Brazil" -> scores as rare.
    # That is precisely the information content of the presence bit, made continuous.
    arms["freq_abraom"] = 1.0 - _num(m5, "af_abraom", fill=0.0)
    arms["freq_gnomad"] = 1.0 - _num(m5, "af_gnomad", fill=0.0)

    # --- models -------------------------------------------------------------------
    if m0 is not None and "probability" in m0.columns:
        arms["M0"] = _num(m0, "probability")
    if "molecular_probability" in m5.columns:
        molecular = _num(m5, "molecular_probability")
        arms["M5_molecular"] = molecular
        if "regional_discount" in m5.columns:
            discount = np.minimum(
                _num(m5, "regional_discount") * float(config.get("discount_scale", 1.0)),
                float(config.get("max_discount", 1.5)),
            )
            arms["M5_v2"] = _sigmoid(_logit(molecular) - discount)
    return arms


def deployed_threshold(dataset: str, config: dict) -> float:
    """The threshold the published pipeline actually used for this slice."""
    if dataset in GLOBAL_DATASETS:
        return float(config.get("global_threshold", 0.5))
    return float(config.get("regional_threshold", 0.5))


def deployed_score(dataset: str, arms: dict[str, np.ndarray]) -> np.ndarray | None:
    """Global slices are decided on the molecular score, regional ones on the discounted score
    (calibrate_m5_v2_regional_scores.py:136-143)."""
    if dataset in GLOBAL_DATASETS:
        return arms.get("M5_molecular")
    return arms.get("M5_v2")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def evaluate(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    keep = np.isfinite(scores) & (labels >= 0)
    labels, scores = labels[keep], scores[keep]
    out = {"n": int(labels.size), "threshold": float(threshold),
           "mcc": float("nan"), "recall": float("nan"), "specificity": float("nan"), "auroc": float("nan")}
    if labels.size == 0:
        return out
    metrics = classification_metrics(labels, scores, float(threshold))
    out.update(mcc=float(metrics["mcc"]), recall=float(metrics["recall"]),
               specificity=float(metrics["specificity"]))
    if np.unique(labels).size > 1:
        out["auroc"] = float(binary_roc_auc(labels, scores))
    else:
        # Single-class slice: MCC is structurally 0 and AUROC undefined. Say so rather than
        # letting a meaningless 0.000 be read as a result.
        out["mcc"] = float("nan")
    return out


def tune_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    keep = np.isfinite(scores) & (labels >= 0)
    labels, scores = labels[keep], scores[keep]
    if labels.size == 0 or np.unique(labels).size < 2:
        return 0.5
    best_threshold, best_mcc = 0.5, -2.0
    for threshold in np.unique(np.round(np.quantile(scores, np.linspace(0, 1, 400)), 6)):
        mcc = classification_metrics(labels, scores, float(threshold))["mcc"]
        if mcc > best_mcc:
            best_threshold, best_mcc = float(threshold), float(mcc)
    return best_threshold


def paired_delta(
    labels: np.ndarray, model: np.ndarray, model_threshold: float,
    baseline: np.ndarray, baseline_threshold: float, *, resamples: int, rng: np.random.Generator,
) -> dict:
    keep = np.isfinite(model) & np.isfinite(baseline) & (labels >= 0)
    labels, model, baseline = labels[keep], model[keep], baseline[keep]
    multiclass = np.unique(labels).size > 1

    def _mcc(idx: np.ndarray, scores: np.ndarray, threshold: float) -> float:
        if np.unique(labels[idx]).size < 2:
            return float("nan")
        return float(classification_metrics(labels[idx], scores[idx], threshold)["mcc"])

    def _auroc(idx: np.ndarray, scores: np.ndarray) -> float:
        if np.unique(labels[idx]).size < 2:
            return float("nan")
        return float(binary_roc_auc(labels[idx], scores[idx]))

    everything = np.arange(labels.size)
    out = {
        "n": int(labels.size),
        "delta_mcc": _mcc(everything, model, model_threshold) - _mcc(everything, baseline, baseline_threshold),
        "delta_auroc": (_auroc(everything, model) - _auroc(everything, baseline)) if multiclass else float("nan"),
    }
    if resamples <= 0 or labels.size == 0 or not multiclass:
        return out
    mcc_draws = np.empty(resamples)
    auroc_draws = np.empty(resamples)
    for i in range(resamples):
        idx = rng.integers(0, labels.size, labels.size)
        mcc_draws[i] = _mcc(idx, model, model_threshold) - _mcc(idx, baseline, baseline_threshold)
        auroc_draws[i] = _auroc(idx, model) - _auroc(idx, baseline)
    for name, draws in (("mcc", mcc_draws), ("auroc", auroc_draws)):
        finite = draws[np.isfinite(draws)]
        if finite.size:
            out[f"{name}_ci_low"] = float(np.percentile(finite, 2.5))
            out[f"{name}_ci_high"] = float(np.percentile(finite, 97.5))
            out[f"{name}_p_baseline_wins"] = float(np.mean(finite <= 0.0))
    return out


# ---------------------------------------------------------------------------
# [3] ABRAOM vs gnomAD, only where both exist
# ---------------------------------------------------------------------------

def abraom_vs_gnomad(m5: pd.DataFrame, dataset: str, split: str) -> dict | None:
    af_abraom = pd.to_numeric(m5.get("af_abraom", pd.Series(dtype=float)), errors="coerce")
    af_gnomad = pd.to_numeric(m5.get("af_gnomad", pd.Series(dtype=float)), errors="coerce")
    labels = pd.to_numeric(m5["label"], errors="coerce")
    mask = af_abraom.notna() & af_gnomad.notna() & labels.notna()
    if mask.sum() < 20 or labels[mask].nunique() < 2:
        return {"dataset": dataset, "split": split, "n": int(mask.sum()), "note": "too few / single class"}
    y = labels[mask].to_numpy(dtype=np.int64)
    return {
        "dataset": dataset,
        "split": split,
        "n": int(mask.sum()),
        "n_pathogenic": int((y == 1).sum()),
        "auroc_abraom": float(binary_roc_auc(y, (1.0 - af_abraom[mask]).to_numpy(dtype=np.float64))),
        "auroc_gnomad": float(binary_roc_auc(y, (1.0 - af_gnomad[mask]).to_numpy(dtype=np.float64))),
    }


def _fmt(value, digits: int = 3) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if not np.isfinite(value) else f"{value:.{digits}f}"


def render(payload: dict) -> str:
    lines: list[str] = []
    lines.append("=" * 100)
    lines.append("RE-BASELINE: the regionalization measured against FREQUENCY-ONLY, not against M0")
    lines.append("=" * 100)

    lines.append("")
    lines.append("[0] SELF-VALIDATION -- can we reproduce the PUBLISHED M5_v2 numbers from raw predictions?")
    lines.append("")
    for row in payload["validation"]:
        status = "OK " if row["match"] else "MISMATCH"
        lines.append(f"    [{status}] {row['dataset']:<28}{row['metric']:<14}"
                     f"reproduced={_fmt(row['reproduced'])}  published={_fmt(row['published'])}")
    if not all(row["match"] for row in payload["validation"]):
        lines.append("")
        lines.append("    !! The reconstruction does NOT match the published table. Everything below is suspect.")

    lines.append("")
    lines.append("[1] Scoreboard (model thresholds tuned on holdout; AUROC is threshold-free)")
    for key, arms in payload["scoreboard"].items():
        lines.append("")
        lines.append(f"    --- {key} ---")
        lines.append(f"    {'arm':<22}{'MCC':>9}{'AUROC':>9}{'recall':>9}{'spec':>9}{'thr':>9}{'n':>8}")
        lines.append("    " + "-" * 75)
        for name, row in arms.items():
            marker = " *" if name == PRIMARY_BASELINE else "  "
            lines.append(
                f"  {marker}{name:<20}{_fmt(row['mcc']):>9}{_fmt(row['auroc']):>9}{_fmt(row['recall']):>9}"
                f"{_fmt(row['specificity']):>9}{_fmt(row['threshold']):>9}{row['n']:>8}"
            )
    lines.append("")
    lines.append(f"    (* = the frequency-only baseline every model must beat)")

    lines.append("")
    lines.append("[2] Head-to-head: MODEL minus FREQUENCY BASELINE, paired bootstrap, same variants")
    lines.append("")
    lines.append(f"    {'slice/split':<24}{'model':<16}{'dMCC':>8}{'95% CI':>20}{'dAUROC':>9}{'95% CI':>20}")
    lines.append("    " + "-" * 97)
    for key, entries in payload["head_to_head"].items():
        for name, row in entries.items():
            mcc_ci = f"[{_fmt(row.get('mcc_ci_low'))}, {_fmt(row.get('mcc_ci_high'))}]"
            auroc_ci = f"[{_fmt(row.get('auroc_ci_low'))}, {_fmt(row.get('auroc_ci_high'))}]"
            lines.append(
                f"    {key:<24}{name:<16}{_fmt(row['delta_mcc']):>8}{mcc_ci:>20}"
                f"{_fmt(row['delta_auroc']):>9}{auroc_ci:>20}"
            )

    if payload.get("abraom_vs_gnomad"):
        lines.append("")
        lines.append("[3] Does ABRAOM beat gnomAD? -- only on variants where BOTH frequencies exist")
        lines.append("")
        lines.append(f"    {'slice':<22}{'split':<10}{'n':>7}{'n_path':>8}{'AUROC abraom':>14}{'AUROC gnomad':>14}")
        lines.append("    " + "-" * 75)
        for row in payload["abraom_vs_gnomad"]:
            if "note" in row:
                lines.append(f"    {row['dataset']:<22}{row['split']:<10}{row['n']:>7}   ({row['note']})")
                continue
            lines.append(
                f"    {row['dataset']:<22}{row['split']:<10}{row['n']:>7}{row['n_pathogenic']:>8}"
                f"{_fmt(row['auroc_abraom']):>14}{_fmt(row['auroc_gnomad']):>14}"
            )

    lines.append("")
    lines.append("=" * 100)
    lines.append("READING [2]: a CI crossing 0 means that model is NOT distinguishable from using")
    lines.append("frequency alone on that slice. dAUROC is the threshold-free version of the same")
    lines.append("question and is the one to quote when threshold choice is contested.")
    lines.append("=" * 100)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rng = np.random.default_rng(args.seed)

    config: dict = {"discount_scale": 1.0, "max_discount": 1.5, "regional_threshold": 0.5, "global_threshold": 0.5}
    if args.m5_v2_config.is_file():
        config.update(json.loads(args.m5_v2_config.read_text(encoding="utf-8")))
        print(f"[info] deployed config: {config}")
    else:
        print(f"[warn] {args.m5_v2_config} missing; deployed-arm numbers will NOT match the paper",
              file=sys.stderr)

    scoreboard: dict[str, dict] = {}
    head_to_head: dict[str, dict] = {}
    validation: list[dict] = []
    avg: list[dict] = []

    for dataset in args.datasets:
        m5_tune = load_split(args.m5_dir, args.slice_dir, dataset, args.tune_split)
        if m5_tune is None:
            continue
        m0_tune = align_to(m5_tune, load_split(args.m0_dir, args.slice_dir, dataset, args.tune_split))
        tune_arms = build_arms(m5_tune, m0_tune, config)
        tune_labels = pd.to_numeric(m5_tune["label"], errors="coerce").fillna(-1).to_numpy(dtype=np.int64)
        thresholds = {name: tune_threshold(tune_labels, scores) for name, scores in tune_arms.items()}

        for split in args.splits:
            m5 = load_split(args.m5_dir, args.slice_dir, dataset, split)
            if m5 is None:
                continue
            m0 = align_to(m5, load_split(args.m0_dir, args.slice_dir, dataset, split))
            labels = pd.to_numeric(m5["label"], errors="coerce").fillna(-1).to_numpy(dtype=np.int64)
            arms = build_arms(m5, m0, config)
            key = f"{dataset}/{split}"

            rows: dict[str, dict] = {}
            for name, scores in arms.items():
                rows[name] = evaluate(labels, scores, thresholds.get(name, 0.5))

            # the arm that reproduces the published pipeline exactly
            published_scores = deployed_score(dataset, arms)
            if published_scores is not None:
                rows["M5_v2_DEPLOYED"] = evaluate(labels, published_scores, deployed_threshold(dataset, config))
            scoreboard[key] = rows

            if split == "test" and published_scores is not None:
                for (ds, metric), expected in PUBLISHED_TEST.items():
                    if ds != dataset:
                        continue
                    got = rows["M5_v2_DEPLOYED"][metric]
                    validation.append({
                        "dataset": ds, "metric": metric, "reproduced": got, "published": expected,
                        "match": bool(np.isfinite(got) and abs(got - expected) <= PUBLISHED_TOLERANCE),
                    })

            baseline_scores = arms.get(PRIMARY_BASELINE)
            if baseline_scores is not None:
                entries: dict[str, dict] = {}
                for name in ("M0", "M5_molecular", "M5_v2"):
                    if name not in arms:
                        continue
                    entries[name] = paired_delta(
                        labels, arms[name], thresholds.get(name, 0.5),
                        baseline_scores, thresholds.get(PRIMARY_BASELINE, 0.5),
                        resamples=args.bootstrap, rng=rng,
                    )
                if entries:
                    head_to_head[key] = entries

            profile = abraom_vs_gnomad(m5, dataset, split)
            if profile:
                avg.append(profile)

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "deployed_config": config,
        "primary_baseline": PRIMARY_BASELINE,
        "bootstrap_resamples": args.bootstrap,
        "validation": validation,
        "scoreboard": scoreboard,
        "head_to_head": head_to_head,
        "abraom_vs_gnomad": avg,
    }

    report = render(payload)
    print(report)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "rebaseline.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (args.output_dir / "rebaseline.txt").write_text(report, encoding="utf-8")
    print(f"\nwrote {args.output_dir}/rebaseline.{{json,txt}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
