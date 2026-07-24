#!/usr/bin/env python3
"""Diagnose whether M5-bounded's `molecular_probability` is allele-frequency contaminated.

Runs LOCAL on the already-produced prediction parquets. No GPU, no job, no training.

THE HYPOTHESIS (pre-registered 2026-07-24, before looking at any number)
-----------------------------------------------------------------------
`RegimeABoundedRegionalHead` (eval/clinvar/heads.py:303-311) feeds BOTH output heads from a
single shared trunk that already consumed the 11 explicit allele-frequency features:

    hidden            = shared([site_ref, variant_repr, local_context, explicit_repr])
    molecular_logit   = molecular_head(hidden)                      # <- trunk saw AF
    regional_discount = 4.0 * sigmoid(discount_head(hidden))
    regional_logit    = molecular_logit - regional_discount

The `molecular_probability` column that the calibration thresholds its guard on
(eval/clinvar/train.py:195 -> :500) therefore is NOT frequency-blind, and frequency is counted
TWICE: implicitly inside `molecular_logit`, and again as the explicit `regional_discount`
subtracted from it. If so, the documented decomposition ("molecular = pure biology",
TCC_REGIONALIZACAO_V11.md 5.10) does not hold, and the two components are not identifiable.

WHAT THIS WOULD EXPLAIN
-----------------------
The A-guarda refutation rested on the observation that the ABRAOM founder P/LP have LOW
`molecular_probability` (median 0.374; only 6/187 clear the 0.65 guard). Founders are, by
definition, COMMON in ABRAOM. A frequency-contaminated molecular head would push exactly those
variants down -- so the guard would find nobody left to protect. That would make the "6/187"
an artifact of the score, not a property of the variants.

PREDICTIONS IF THE HYPOTHESIS IS TRUE
-------------------------------------
P1. WITHIN a fixed label stratum, `molecular_probability` still tracks allele frequency.
    Stratifying by label is essential and is the whole methodological point: AF genuinely
    predicts the label (common => benign), so an UNSTRATIFIED correlation would come out
    negative even for a perfectly clean molecular head. Only the within-label part is evidence.
P2. The effect is stronger than M0's. M0 is the frequency-blind reference: it was trained with
    `explicit_feature_columns=[]` and never received an AF number as input. M0 plays the role
    A_scrambled played for the adapters -- without a negative control, a raw correlation means
    nothing, because some within-label correlation is expected from biology alone (conserved
    sites are both rarer and more pathogenic).
P3. Among the founder P/LP (all label=1), the more common the variant, the LOWER its
    "molecular" score.

PRE-REGISTERED VERDICT CRITERION (fixed before running)
-------------------------------------------------------
Let d = |rho(molecular, af_gnomad)|_M5 - |rho(probability, af_gnomad)|_M0, computed WITHIN
label strata on the same variants.

    CONFIRMED    if d >= 0.15 on the benign stratum AND rho(molecular, af_abraom) < 0
                 among abraom_pathogenic_present (label=1).
    REFUTED      if d < 0.05 everywhere.
    INCONCLUSIVE otherwise -- report it as such and do NOT act on it.

A refutation here kills the proposed "decouple the molecular head" change before it costs a
single training job. That is the point of running this first.

USAGE (on the SageMaker notebook)
---------------------------------
    python scripts/diagnose_molecular_contamination.py \
        --m5-dir      ~/v11eval/reval_m5bounded \
        --m0-dir      ~/v11eval/m5v3_modelroot/m0_nonbr_beatv10_v1_sagemaker \
        --slice-dir   ~/slices \
        --split       test \
        --output-dir  ~/v11eval/diag_molecular_contamination
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

# Reuse the calibration's own loaders so paths/columns cannot drift apart.
from scripts.calibrate_m5_v3_safety import load_metadata  # noqa: E402

# The guard threshold the M5_v3 safety layer actually used.
MOLECULAR_GUARD_THRESHOLD = 0.65

FOUNDER_SLICE = "abraom_pathogenic_present"
COMMON_BENIGN_SLICE = "abraom_common_benign"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    home = Path.home()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--m5-dir", type=Path, default=home / "v11eval/reval_m5bounded",
                        help="dir with {slice}.{split}.predictions.parquet from the M5-bounded eval")
    parser.add_argument("--m0-dir", type=Path,
                        default=home / "v11eval/m5v3_modelroot/m0_nonbr_beatv10_v1_sagemaker",
                        help="same layout, for M0 (the frequency-blind negative control)")
    parser.add_argument("--slice-dir", type=Path, default=home / "slices")
    parser.add_argument("--split", default="test", choices=["test", "holdout"])
    parser.add_argument("--datasets", nargs="*",
                        default=["br_only", COMMON_BENIGN_SLICE, FOUNDER_SLICE, "global_nonbr_no_abraom"])
    parser.add_argument("--bootstrap", type=int, default=2000, help="resamples for the rho CIs (0 to skip)")
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--output-dir", type=Path, default=home / "v11eval/diag_molecular_contamination")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Rank correlation (dependency-free: rank, then Pearson -- exact Spearman)
# ---------------------------------------------------------------------------

def spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    """Spearman rho over pairwise-complete observations. Returns (rho, n_used)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    n = int(x.size)
    if n < 10:
        return float("nan"), n
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    if rx.std() == 0 or ry.std() == 0:
        return float("nan"), n
    return float(np.corrcoef(rx, ry)[0, 1]), n


def spearman_ci(x: np.ndarray, y: np.ndarray, *, resamples: int, rng: np.random.Generator) -> dict:
    rho, n = spearman(x, y)
    out = {"rho": rho, "n": n, "ci_low": float("nan"), "ci_high": float("nan")}
    if resamples <= 0 or not np.isfinite(rho) or n < 10:
        return out
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    draws = np.empty(resamples, dtype=np.float64)
    for i in range(resamples):
        idx = rng.integers(0, x.size, x.size)
        draws[i], _ = spearman(x[idx], y[idx])
    draws = draws[np.isfinite(draws)]
    if draws.size:
        out["ci_low"] = float(np.percentile(draws, 2.5))
        out["ci_high"] = float(np.percentile(draws, 97.5))
    return out


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_predictions(model_dir: Path, slice_dir: Path, dataset: str, split: str, score_column: str) -> pd.DataFrame:
    """Load {dataset}.{split}.predictions.parquet and attach the slice metadata (AF columns)."""
    path = model_dir / f"{dataset}.{split}.predictions.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"missing predictions: {path}")
    predictions = pd.read_parquet(path).copy()
    if score_column not in predictions.columns:
        raise ValueError(
            f"{path} has no '{score_column}' column (found: {sorted(predictions.columns)}). "
            "For M0 pass the plain 'probability'; only the bounded head emits 'molecular_probability'."
        )
    metadata = load_metadata(slice_dir, dataset, split)
    merged = predictions.merge(metadata, on="original_index", how="left", validate="one_to_one")
    return merged.reset_index(drop=True)


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns:
        return np.full(len(frame), np.nan)
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)


def align_on_variants(m5: pd.DataFrame, m0: pd.DataFrame | None) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Restrict both models to the SAME variants, in the same order.

    Every comparison in this project is paired (see the bootstrap in
    scripts/compare_fusion_falsification.py). Comparing an M5 correlation computed on one set of
    variants against an M0 correlation computed on a slightly different set would reintroduce
    exactly the composition noise the pairing exists to remove -- and the two evals were launched
    as separate jobs, so identical coverage is an assumption, not a fact.
    """
    if m0 is None:
        return m5, None
    common = np.intersect1d(m5["original_index"].to_numpy(), m0["original_index"].to_numpy())
    m5_aligned = m5.loc[m5["original_index"].isin(common)].sort_values("original_index").reset_index(drop=True)
    m0_aligned = m0.loc[m0["original_index"].isin(common)].sort_values("original_index").reset_index(drop=True)
    return m5_aligned, m0_aligned


# ---------------------------------------------------------------------------
# Test 1/2 -- within-label correlation of the "molecular" score with frequency
# ---------------------------------------------------------------------------

def within_label_correlations(
    m5: pd.DataFrame, m0: pd.DataFrame | None, dataset: str, *, resamples: int, rng: np.random.Generator
) -> list[dict]:
    rows: list[dict] = []
    # Paired: M5 and M0 must be scored on identical variants for the excess to mean anything.
    m5, m0 = align_on_variants(m5, m0)
    labels = sorted(pd.to_numeric(m5["label"], errors="coerce").dropna().unique())
    for af_column in ("af_gnomad", "af_abraom"):
        for label in labels:
            m5_stratum = m5.loc[pd.to_numeric(m5["label"], errors="coerce") == label]
            entry = {
                "dataset": dataset,
                "af_column": af_column,
                "label": int(label),
                "label_name": "pathogenic" if int(label) == 1 else "benign",
                "paired_n_variants": int(len(m5)),
            }
            m5_stats = spearman_ci(
                _numeric(m5_stratum, "molecular_probability"), _numeric(m5_stratum, af_column),
                resamples=resamples, rng=rng,
            )
            entry["m5_molecular"] = m5_stats
            if m0 is not None:
                m0_stratum = m0.loc[pd.to_numeric(m0["label"], errors="coerce") == label]
                entry["m0_control"] = spearman_ci(
                    _numeric(m0_stratum, "probability"), _numeric(m0_stratum, af_column),
                    resamples=resamples, rng=rng,
                )
                if np.isfinite(m5_stats["rho"]) and np.isfinite(entry["m0_control"]["rho"]):
                    entry["excess_abs_rho"] = abs(m5_stats["rho"]) - abs(entry["m0_control"]["rho"])
            rows.append(entry)
    return rows


# ---------------------------------------------------------------------------
# Test 3 -- is the decomposition degenerate?
# ---------------------------------------------------------------------------

def decomposition_coupling(m5: pd.DataFrame, dataset: str) -> dict:
    """If `molecular` and `regional_discount` are strongly coupled they are not independent
    components and 'molecular - discount' is not an attribution."""
    rho, n = spearman(_numeric(m5, "molecular_probability"), _numeric(m5, "regional_discount"))
    rho_disc_af, _ = spearman(_numeric(m5, "regional_discount"), _numeric(m5, "af_abraom"))
    return {
        "dataset": dataset,
        "n": n,
        "rho_molecular_vs_discount": rho,
        "rho_discount_vs_af_abraom": rho_disc_af,
    }


# ---------------------------------------------------------------------------
# Test 4 -- the payoff: does contamination break the guard?
# ---------------------------------------------------------------------------

def guard_budget_comparison(
    m5_founders: pd.DataFrame, m5_benigns: pd.DataFrame,
    m0_founders: pd.DataFrame | None, m0_benigns: pd.DataFrame | None,
) -> dict:
    """Compare WHO the guard protects, at a matched budget.

    The v3 guard protects everything scoring >= 0.65 on `molecular_probability`. Pooling the
    founder P/LP (must protect) with the ABRAOM common benigns (must NOT protect), we count how
    many of each land above that bar. Then we give M0 the SAME BUDGET -- its top-K by score,
    K = the number M5 guarded -- and count again.

    Absolute thresholds are not comparable across two differently-calibrated models; a matched
    budget is. If M0 (which never saw a frequency) captures more founders per guarded benign
    than M5's "molecular" does, then the guard's failure is an artifact of the contaminated
    score rather than a property of these variants -- and the A-guarda conclusion needs revisiting.
    """
    result: dict = {}

    # Paired on both slices independently: the budget handed to M0 must be earned on the same
    # variants M5 was scored on, or the "founders recovered" number is partly a coverage artifact.
    m5_founders, m0_founders = align_on_variants(m5_founders, m0_founders)
    m5_benigns, m0_benigns = align_on_variants(m5_benigns, m0_benigns)

    m5_founder_scores = _numeric(m5_founders, "molecular_probability")
    m5_benign_scores = _numeric(m5_benigns, "molecular_probability")
    guarded_founders = int(np.sum(m5_founder_scores >= MOLECULAR_GUARD_THRESHOLD))
    guarded_benigns = int(np.sum(m5_benign_scores >= MOLECULAR_GUARD_THRESHOLD))
    budget = guarded_founders + guarded_benigns

    result["m5_molecular"] = {
        "n_founders": int(np.sum(np.isfinite(m5_founder_scores))),
        "n_common_benigns": int(np.sum(np.isfinite(m5_benign_scores))),
        "median_founder_score": float(np.nanmedian(m5_founder_scores)) if m5_founder_scores.size else float("nan"),
        "median_benign_score": float(np.nanmedian(m5_benign_scores)) if m5_benign_scores.size else float("nan"),
        "guard_threshold": MOLECULAR_GUARD_THRESHOLD,
        "guarded_founders": guarded_founders,
        "guarded_common_benigns": guarded_benigns,
        "benigns_per_founder": (guarded_benigns / guarded_founders) if guarded_founders else float("inf"),
        "guard_budget": budget,
    }

    if m0_founders is None or m0_benigns is None or budget == 0:
        return result

    m0_founder_scores = _numeric(m0_founders, "probability")
    m0_benign_scores = _numeric(m0_benigns, "probability")
    pooled = np.concatenate([m0_founder_scores, m0_benign_scores])
    is_founder = np.concatenate([
        np.ones(m0_founder_scores.size, dtype=bool),
        np.zeros(m0_benign_scores.size, dtype=bool),
    ])
    finite = np.isfinite(pooled)
    pooled, is_founder = pooled[finite], is_founder[finite]
    if pooled.size == 0:
        return result

    k = min(budget, pooled.size)
    top_k = np.argsort(-pooled, kind="stable")[:k]
    m0_guarded_founders = int(np.sum(is_founder[top_k]))
    m0_guarded_benigns = int(k - m0_guarded_founders)

    result["m0_matched_budget"] = {
        "n_founders": int(m0_founder_scores.size),
        "n_common_benigns": int(m0_benign_scores.size),
        "median_founder_score": float(np.nanmedian(m0_founder_scores)) if m0_founder_scores.size else float("nan"),
        "median_benign_score": float(np.nanmedian(m0_benign_scores)) if m0_benign_scores.size else float("nan"),
        "guard_budget": int(k),
        "guarded_founders": m0_guarded_founders,
        "guarded_common_benigns": m0_guarded_benigns,
        "benigns_per_founder": (m0_guarded_benigns / m0_guarded_founders) if m0_guarded_founders else float("inf"),
    }
    result["founders_recovered_by_m0"] = m0_guarded_founders - guarded_founders
    return result


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

def decide_verdict(correlations: list[dict]) -> dict:
    """Apply the criterion fixed in the module docstring. No post-hoc rule changes."""
    benign_excess = [
        row["excess_abs_rho"]
        for row in correlations
        if row["af_column"] == "af_gnomad" and row["label"] == 0 and "excess_abs_rho" in row
    ]
    founder_af_rho = [
        row["m5_molecular"]["rho"]
        for row in correlations
        if row["dataset"] == FOUNDER_SLICE and row["af_column"] == "af_abraom" and row["label"] == 1
    ]
    max_excess = max(benign_excess) if benign_excess else float("nan")
    founder_rho = founder_af_rho[0] if founder_af_rho else float("nan")

    if np.isfinite(max_excess) and max_excess >= 0.15 and np.isfinite(founder_rho) and founder_rho < 0:
        verdict = "CONFIRMED"
        action = ("Decouple the molecular head from the explicit features (molecular_logit from a "
                  "sequence-only trunk; only discount_head sees AF), retrain the M5-bounded fusion, "
                  "re-run the eval and re-calibrate.")
    elif np.isfinite(max_excess) and max_excess < 0.05:
        verdict = "REFUTED"
        action = ("Do NOT rebuild the head. The molecular path is no more frequency-dependent than "
                  "M0. Drop this lever and reconsider curation / the k-fold leave-Brazilian-out "
                  "relaxation instead.")
    else:
        verdict = "INCONCLUSIVE"
        action = ("Do not act on this. Effect sits between the pre-registered bounds; report as "
                  "inconclusive rather than reading it in the direction we hoped for.")

    return {
        "verdict": verdict,
        "max_benign_excess_abs_rho": max_excess,
        "founder_rho_molecular_vs_af_abraom": founder_rho,
        "criterion": "CONFIRMED if excess >= 0.15 and founder rho < 0; REFUTED if excess < 0.05",
        "recommended_action": action,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt(value: float, digits: int = 3) -> str:
    return "n/a" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


def render_report(payload: dict) -> str:
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("DIAGNOSTIC -- is M5-bounded's `molecular_probability` frequency-contaminated?")
    lines.append(f"split={payload['split']}   generated={payload['generated_at']}")
    lines.append("=" * 78)

    lines.append("")
    lines.append("[1] WITHIN-LABEL rank correlation of the molecular score with allele frequency")
    lines.append("    (within-label is the honest test: AF predicts the label, so an unstratified")
    lines.append("     correlation would be negative even for a clean head. M0 = frequency-blind control.)")
    lines.append("")
    header = f"    {'slice':<26}{'AF':<12}{'label':<11}{'M5 rho':>9}{'M0 rho':>9}{'excess':>9}{'n':>8}"
    lines.append(header)
    lines.append("    " + "-" * (len(header) - 4))
    for row in payload["within_label_correlations"]:
        m5_rho = row["m5_molecular"]["rho"]
        m0_rho = row.get("m0_control", {}).get("rho", float("nan"))
        excess = row.get("excess_abs_rho", float("nan"))
        lines.append(
            f"    {row['dataset']:<26}{row['af_column']:<12}{row['label_name']:<11}"
            f"{_fmt(m5_rho):>9}{_fmt(m0_rho):>9}{_fmt(excess):>9}{row['m5_molecular']['n']:>8}"
        )

    lines.append("")
    lines.append("[2] Is the decomposition degenerate? (molecular vs the discount subtracted from it)")
    lines.append("")
    lines.append(f"    {'slice':<26}{'rho(mol,disc)':>15}{'rho(disc,af_abraom)':>22}{'n':>8}")
    lines.append("    " + "-" * 67)
    for row in payload["decomposition_coupling"]:
        lines.append(
            f"    {row['dataset']:<26}{_fmt(row['rho_molecular_vs_discount']):>15}"
            f"{_fmt(row['rho_discount_vs_af_abraom']):>22}{row['n']:>8}"
        )

    guard = payload.get("guard_budget_comparison") or {}
    if guard:
        lines.append("")
        lines.append("[3] WHO does the guard protect, at a matched budget?")
        lines.append("    (founder P/LP = must protect; ABRAOM common benigns = must not)")
        lines.append("")
        m5 = guard.get("m5_molecular", {})
        lines.append(f"    M5 molecular @ {MOLECULAR_GUARD_THRESHOLD}:")
        lines.append(f"        founders guarded      : {m5.get('guarded_founders')} / {m5.get('n_founders')}")
        lines.append(f"        common benigns guarded: {m5.get('guarded_common_benigns')} / {m5.get('n_common_benigns')}")
        lines.append(f"        benigns per founder   : {_fmt(m5.get('benigns_per_founder', float('nan')), 1)}")
        lines.append(f"        median score founder / benign: "
                     f"{_fmt(m5.get('median_founder_score', float('nan')))} / {_fmt(m5.get('median_benign_score', float('nan')))}")
        m0 = guard.get("m0_matched_budget")
        if m0:
            lines.append(f"    M0 (frequency-blind), same budget of {m0.get('guard_budget')} guarded:")
            lines.append(f"        founders guarded      : {m0.get('guarded_founders')} / {m0.get('n_founders')}")
            lines.append(f"        common benigns guarded: {m0.get('guarded_common_benigns')} / {m0.get('n_common_benigns')}")
            lines.append(f"        benigns per founder   : {_fmt(m0.get('benigns_per_founder', float('nan')), 1)}")
            lines.append("")
            recovered = guard.get("founders_recovered_by_m0")
            lines.append(f"    >> founders recovered by using the frequency-blind score: {recovered:+d}")

    verdict = payload["verdict"]
    lines.append("")
    lines.append("=" * 78)
    lines.append(f"VERDICT: {verdict['verdict']}")
    lines.append(f"    criterion   : {verdict['criterion']}")
    lines.append(f"    max excess |rho| (benign stratum, af_gnomad): {_fmt(verdict['max_benign_excess_abs_rho'])}")
    lines.append(f"    founder rho(molecular, af_abraom)           : {_fmt(verdict['founder_rho_molecular_vs_af_abraom'])}")
    lines.append(f"    action      : {verdict['recommended_action']}")
    lines.append("=" * 78)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rng = np.random.default_rng(args.seed)

    m5_frames: dict[str, pd.DataFrame] = {}
    m0_frames: dict[str, pd.DataFrame] = {}
    for dataset in args.datasets:
        m5_frames[dataset] = load_predictions(args.m5_dir, args.slice_dir, dataset, args.split, "molecular_probability")
        try:
            m0_frames[dataset] = load_predictions(args.m0_dir, args.slice_dir, dataset, args.split, "probability")
        except (FileNotFoundError, ValueError) as exc:
            print(f"[warn] no M0 control for '{dataset}': {exc}", file=sys.stderr)

    correlations: list[dict] = []
    coupling: list[dict] = []
    for dataset, m5 in m5_frames.items():
        correlations.extend(
            within_label_correlations(m5, m0_frames.get(dataset), dataset, resamples=args.bootstrap, rng=rng)
        )
        coupling.append(decomposition_coupling(m5, dataset))

    guard: dict = {}
    if FOUNDER_SLICE in m5_frames and COMMON_BENIGN_SLICE in m5_frames:
        guard = guard_budget_comparison(
            m5_frames[FOUNDER_SLICE], m5_frames[COMMON_BENIGN_SLICE],
            m0_frames.get(FOUNDER_SLICE), m0_frames.get(COMMON_BENIGN_SLICE),
        )

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "split": args.split,
        "m5_dir": str(args.m5_dir),
        "m0_dir": str(args.m0_dir),
        "datasets": list(args.datasets),
        "bootstrap_resamples": args.bootstrap,
        "within_label_correlations": correlations,
        "decomposition_coupling": coupling,
        "guard_budget_comparison": guard,
        "verdict": decide_verdict(correlations),
    }

    report = render_report(payload)
    print(report)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"diagnostic.{args.split}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (args.output_dir / f"diagnostic.{args.split}.txt").write_text(report, encoding="utf-8")
    print(f"\nwrote {args.output_dir}/diagnostic.{args.split}.{{json,txt}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
