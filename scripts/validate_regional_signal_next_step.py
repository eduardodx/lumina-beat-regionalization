#!/usr/bin/env python3
"""Post-v3 regional signal validation, critical error audit, and stronger controls."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.calibrate_m5_v3_safety import (  # noqa: E402
    DEFAULT_DATASETS,
    GLOBAL_DATASETS,
    KEY_CONTROL_METRICS,
    SafetyConfig,
    load_m5_split,
    metrics_for_config,
    value,
)

DEFAULT_MASTER = Path("data/datasets/clinvar/regional_abraom/clinvar_regional_abraom_master.parquet")
DEFAULT_RAW_TEST_DIR = Path("artifacts/clinvar_regional_eval/m5_v2_bounded_regional_eval_beatv10_v1_sagemaker/_extracted")
DEFAULT_SLICE_DIR = Path("data/datasets/clinvar/regional_abraom/slices")
DEFAULT_M5_V2_DIR = Path("artifacts/clinvar_regional_eval/m5_v2_calibrated_holdout_tuned")
DEFAULT_M5_V3_DIR = Path("artifacts/clinvar_regional_eval/m5_v3_safety_calibrated")
DEFAULT_OUTPUT_DIR = Path("artifacts/clinvar_regional_eval/regional_signal_validation_next_step")

MASTER_ENRICH_COLUMNS = [
    "variant_key",
    "source_review_status",
    "regional_clinical_significance_values",
    "clinvar_variation_ids",
    "clinvar_variation_id_count",
    "max_review_status_rank_submission",
    "max_review_status_rank_aggregate",
    "regional_submitters",
    "exact_submission_match_rows",
    "exact_variant_match_rows",
    "brazilian_submission_rows",
    "non_brazilian_submission_rows",
]

CRITICAL_COLUMNS = [
    "audit_type",
    "priority_score",
    "priority_tier",
    "recommended_action",
    "failure_category",
    "variant_key",
    "source_variant_id",
    "GeneSymbol",
    "Chromosome",
    "Start",
    "ReferenceAlleleVCF",
    "AlternateAlleleVCF",
    "variant_type",
    "label",
    "prediction_used",
    "score_used",
    "threshold_used",
    "score_margin",
    "molecular_probability",
    "regional_discount",
    "effective_regional_discount",
    "safety_guard_triggered",
    "af_abraom",
    "af_gnomad",
    "af_abraom_bin",
    "specificity",
    "specificity_bin",
    "source_review_status",
    "regional_clinical_significance_values",
    "clinvar_variation_ids",
    "max_review_status_rank_aggregate",
    "has_brazilian_submitter",
    "has_non_brazilian_submitter",
]

STRONG_CONTROL_MODES = [
    ("within_gene_af_bin", ["GeneSymbol", "af_abraom_bin"]),
    ("within_af_bin_variant_type", ["af_abraom_bin", "variant_type"]),
    ("within_specificity_bin_variant_type", ["specificity_bin", "variant_type"]),
    ("within_chromosome_af_bin", ["Chromosome", "af_abraom_bin"]),
    ("within_chromosome_af_bin_variant_type", ["Chromosome", "af_abraom_bin", "variant_type"]),
    ("within_gene_af_bin_variant_type", ["GeneSymbol", "af_abraom_bin", "variant_type"]),
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-test-dir", type=Path, default=DEFAULT_RAW_TEST_DIR)
    parser.add_argument("--slice-dir", type=Path, default=DEFAULT_SLICE_DIR)
    parser.add_argument("--clinvar-master", type=Path, default=DEFAULT_MASTER)
    parser.add_argument("--m5-v2-dir", type=Path, default=DEFAULT_M5_V2_DIR)
    parser.add_argument("--m5-v3-dir", type=Path, default=DEFAULT_M5_V3_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    parser.add_argument("--control-seeds", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260629)
    return parser.parse_args(argv)


def parquet_columns(path: Path) -> list[str]:
    return list(pq.ParquetFile(path).schema_arrow.names)


def load_selected_config(path: Path) -> SafetyConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return SafetyConfig(**payload)


def load_master_enrichment(path: Path) -> pd.DataFrame:
    available = set(parquet_columns(path))
    columns = [column for column in MASTER_ENRICH_COLUMNS if column in available]
    if "variant_key" not in columns:
        raise ValueError(f"{path} is missing variant_key")
    frame = pd.read_parquet(path, columns=columns)
    return frame.drop_duplicates("variant_key").reset_index(drop=True)


def load_predictions_with_enrichment(
    m5_v3_dir: Path,
    m5_v2_dir: Path,
    master_enrichment: pd.DataFrame,
    dataset: str,
) -> pd.DataFrame:
    v3_path = m5_v3_dir / f"{dataset}.test.predictions.parquet"
    v2_path = m5_v2_dir / f"{dataset}.test.predictions.parquet"
    if not v3_path.is_file():
        raise FileNotFoundError(v3_path)
    if not v2_path.is_file():
        raise FileNotFoundError(v2_path)
    v3 = pd.read_parquet(v3_path).copy()
    v2 = pd.read_parquet(v2_path)[["original_index", "score_used", "threshold_used", "prediction_used"]].rename(
        columns={
            "score_used": "m5_v2_score_used",
            "threshold_used": "m5_v2_threshold_used",
            "prediction_used": "m5_v2_prediction_used",
        }
    )
    frame = v3.merge(v2, on="original_index", how="left", validate="one_to_one")
    frame = frame.merge(master_enrichment, on="variant_key", how="left", validate="many_to_one")
    frame["score_margin"] = frame["score_used"].astype(float) - frame["threshold_used"].astype(float)
    frame["GeneSymbol"] = frame["GeneSymbol"].fillna("unknown").astype(str).replace("", "unknown")
    frame["variant_type"] = frame["variant_type"].fillna("unknown").astype(str)
    frame["af_abraom_bin"] = frame["af_abraom_bin"].fillna("unknown").astype(str)
    frame["specificity_bin"] = frame["specificity_bin"].fillna("unknown").astype(str)
    frame["dataset"] = dataset
    return frame


def review_rank(row: pd.Series) -> float:
    for column in ["max_review_status_rank_aggregate", "max_review_status_rank_submission"]:
        value_ = row.get(column)
        if pd.notna(value_):
            return float(value_)
    return 0.0


def boolish(value_: Any) -> bool:
    if pd.isna(value_):
        return False
    if isinstance(value_, bool):
        return value_
    return str(value_).strip().lower() in {"1", "true", "yes", "y"}


def categorize_false_benign(row: pd.Series) -> str:
    molecular = float(row.get("molecular_probability", 0.0) or 0.0)
    score_margin = float(row.get("score_margin", -1.0) or -1.0)
    af_abraom = float(row.get("af_abraom", 0.0) or 0.0)
    effective_discount = float(row.get("effective_regional_discount", row.get("regional_discount", 0.0)) or 0.0)
    if molecular >= 0.65:
        return "guard_should_have_protected_review"
    if score_margin >= -0.05:
        return "threshold_near_miss"
    if molecular < 0.35:
        return "weak_molecular_signal"
    if af_abraom >= 0.01:
        return "common_plp_recessive_or_founder_review"
    if effective_discount >= 0.25 and molecular >= 0.35:
        return "regional_discount_residual_suppression"
    return "unresolved_false_benign"


def categorize_false_pathogenic(row: pd.Series) -> str:
    molecular = float(row.get("molecular_probability", 0.0) or 0.0)
    score_margin = float(row.get("score_margin", 1.0) or 1.0)
    af_abraom = float(row.get("af_abraom", 0.0) or 0.0)
    specificity = float(row.get("specificity", 0.0) or 0.0)
    effective_discount = float(row.get("effective_regional_discount", row.get("regional_discount", 0.0)) or 0.0)
    if molecular >= 0.80:
        return "molecular_score_overdominant"
    if score_margin <= 0.05:
        return "threshold_near_miss"
    if af_abraom >= 0.01 and effective_discount < 0.25:
        return "regional_discount_insufficient_common"
    if specificity >= 0.01:
        return "abraom_specific_common_benign"
    return "unresolved_false_pathogenic"


def priority_false_benign(row: pd.Series) -> float:
    molecular = float(row.get("molecular_probability", 0.0) or 0.0)
    af_abraom = float(row.get("af_abraom", 0.0) or 0.0)
    score_margin = float(row.get("score_margin", -1.0) or -1.0)
    score = 3.0
    score += min(2.0, 2.0 * molecular)
    score += min(1.5, max(0.0, af_abraom) * 20.0)
    score += 1.0 if score_margin >= -0.05 else 0.0
    score += 0.75 if boolish(row.get("has_brazilian_submitter")) else 0.0
    score += min(1.0, review_rank(row) / 3.0)
    return float(score)


def priority_false_pathogenic(row: pd.Series) -> float:
    molecular = float(row.get("molecular_probability", 0.0) or 0.0)
    specificity = float(row.get("specificity", 0.0) or 0.0)
    score_margin = float(row.get("score_margin", 0.0) or 0.0)
    score = 2.0
    score += min(2.5, 2.5 * molecular)
    score += min(1.5, max(0.0, specificity) * 10.0)
    score += 1.0 if score_margin > 0.25 else 0.0
    score += 0.5 if boolish(row.get("has_brazilian_submitter")) else 0.0
    return float(score)


def priority_tier(score: float, audit_type: str) -> str:
    if audit_type == "false_benign_plp":
        if score >= 6.0:
            return "P0_manual_review"
        if score >= 4.5:
            return "P1_manual_review"
        return "P2_model_diagnostic"
    if score >= 5.0:
        return "P1_manual_review"
    if score >= 3.5:
        return "P2_model_diagnostic"
    return "P3_low_priority"


def recommended_action(row: pd.Series) -> str:
    audit_type = row["audit_type"]
    category = row["failure_category"]
    if audit_type == "false_benign_plp":
        if category == "weak_molecular_signal":
            return "curate ClinVar evidence and molecular context; model lacks pathogenic signal"
        if category == "threshold_near_miss":
            return "evaluate threshold/guard relaxation on independent validation only"
        if category == "common_plp_recessive_or_founder_review":
            return "manual review for recessive/founder/plausible high-frequency P/LP explanation"
        if category == "regional_discount_residual_suppression":
            return "test stricter molecular guard or lower residual frequency discount"
        return "manual clinical-evidence review before using as sentinel"
    if category == "molecular_score_overdominant":
        return "inspect molecular false-positive signal and ClinVar benign assertion quality"
    if category == "threshold_near_miss":
        return "candidate for conservative threshold adjustment, not retraining"
    if category == "regional_discount_insufficient_common":
        return "candidate for regional benign rule after label review"
    if category == "abraom_specific_common_benign":
        return "high-value ABRAOM-specific benign review candidate"
    return "manual benign-label and population-frequency review"


def compact_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in CRITICAL_COLUMNS if column in frame.columns]


def build_critical_audit(frames: dict[str, pd.DataFrame], output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    audit_dir = output_dir / "critical_error_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    plp = frames["abraom_pathogenic_present"]
    false_benign = plp.loc[plp["label"].eq(1) & plp["prediction_used"].eq(0)].copy()
    false_benign["audit_type"] = "false_benign_plp"
    false_benign["failure_category"] = false_benign.apply(categorize_false_benign, axis=1)
    false_benign["priority_score"] = false_benign.apply(priority_false_benign, axis=1)

    benign = frames["abraom_common_benign"]
    false_pathogenic = benign.loc[benign["label"].eq(0) & benign["prediction_used"].eq(1)].copy()
    false_pathogenic["audit_type"] = "false_pathogenic_common_benign"
    false_pathogenic["failure_category"] = false_pathogenic.apply(categorize_false_pathogenic, axis=1)
    false_pathogenic["priority_score"] = false_pathogenic.apply(priority_false_pathogenic, axis=1)

    critical = pd.concat([false_benign, false_pathogenic], ignore_index=True)
    critical["priority_tier"] = [
        priority_tier(float(score), str(audit_type))
        for score, audit_type in zip(critical["priority_score"], critical["audit_type"], strict=False)
    ]
    critical["recommended_action"] = critical.apply(recommended_action, axis=1)
    critical = critical.sort_values(["priority_tier", "priority_score"], ascending=[True, False]).reset_index(drop=True)

    false_benign[compact_columns(false_benign)].sort_values("priority_score", ascending=False).to_csv(
        audit_dir / "false_benign_plp_review_queue.csv",
        index=False,
    )
    false_pathogenic[compact_columns(false_pathogenic)].sort_values("priority_score", ascending=False).to_csv(
        audit_dir / "false_pathogenic_common_benign_review_queue.csv",
        index=False,
    )
    critical[compact_columns(critical)].to_csv(audit_dir / "combined_manual_review_queue.csv", index=False)
    critical.to_parquet(audit_dir / "combined_manual_review_queue.parquet", index=False)

    summary = (
        critical.groupby(["audit_type", "failure_category", "priority_tier"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["audit_type", "n"], ascending=[True, False])
    )
    summary.to_csv(output_dir / "critical_error_category_summary.csv", index=False)

    gene_summary = (
        critical.groupby(["audit_type", "GeneSymbol"], dropna=False)
        .agg(
            n=("variant_key", "count"),
            mean_priority=("priority_score", "mean"),
            max_priority=("priority_score", "max"),
            median_af_abraom=("af_abraom", "median"),
            median_molecular_probability=("molecular_probability", "median"),
        )
        .reset_index()
        .sort_values(["audit_type", "n", "max_priority"], ascending=[True, False, False])
    )
    gene_summary.to_csv(output_dir / "critical_error_gene_summary.csv", index=False)
    return critical, summary, gene_summary


def build_derived_review_panel(frames: dict[str, pd.DataFrame], critical: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    plp_all = frames["abraom_pathogenic_present"].copy()
    plp_all["panel_role"] = "plp_abraom_present_test_sensitivity"
    critical_panel = critical.copy()
    critical_panel["panel_role"] = critical_panel["audit_type"].map(
        {
            "false_benign_plp": "critical_plp_false_benign",
            "false_pathogenic_common_benign": "critical_common_benign_false_pathogenic",
        }
    )
    panel = pd.concat([plp_all, critical_panel], ignore_index=True)
    panel["panel_is_clinically_curated"] = False
    panel["panel_usage"] = "review_and_sensitivity_only_do_not_train"
    panel = panel.drop_duplicates(["variant_key", "panel_role"]).reset_index(drop=True)
    columns = [
        "panel_role",
        "panel_is_clinically_curated",
        "panel_usage",
        *compact_columns(panel),
    ]
    columns = list(dict.fromkeys([column for column in columns if column in panel.columns]))
    panel[columns].to_csv(output_dir / "derived_review_sentinel_panel.csv", index=False)
    panel[columns].to_parquet(output_dir / "derived_review_sentinel_panel.parquet", index=False)
    panel_summary = panel.groupby("panel_role", dropna=False).size().reset_index(name="n")
    panel_summary.to_csv(output_dir / "derived_review_sentinel_panel_summary.csv", index=False)
    return panel


def make_group_key(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    parts = []
    for column in columns:
        if column not in frame.columns:
            parts.append(pd.Series(["missing"] * len(frame), index=frame.index))
        else:
            parts.append(frame[column].fillna("missing").astype(str))
    key = parts[0]
    for part in parts[1:]:
        key = key + "||" + part
    return key


def grouped_permutation(values: pd.Series, groups: pd.Series, rng: np.random.Generator) -> tuple[pd.Series, float, int, int]:
    output = values.copy().reset_index(drop=True)
    groups = groups.reset_index(drop=True)
    changed = np.zeros(len(output), dtype=bool)
    informative_groups = 0
    total_groups = int(groups.nunique(dropna=False))
    for _, index in groups.groupby(groups, dropna=False).groups.items():
        idx = np.asarray(index)
        if len(idx) < 2:
            continue
        informative_groups += 1
        current = output.iloc[idx].to_numpy(copy=True)
        permuted = current[rng.permutation(len(current))]
        output.iloc[idx] = permuted
        changed[idx] = permuted != current
    return output, float(changed.mean()), informative_groups, total_groups


def run_stronger_controls(
    raw_test: dict[str, pd.DataFrame],
    config: SafetyConfig,
    *,
    seeds: int,
    seed_base: int,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    real_rows, _ = metrics_for_config(raw_test, config, model_name="M5_v3_safety")
    real = pd.DataFrame(real_rows)
    rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for mode_index, (mode, columns) in enumerate(STRONG_CONTROL_MODES):
        for seed_offset in range(seeds):
            seed = seed_base + 10_000 * mode_index + seed_offset
            rng = np.random.default_rng(seed)
            scrambled: dict[str, pd.DataFrame] = {}
            for dataset, frame in raw_test.items():
                group_key = make_group_key(frame, columns)
                new_values, changed_fraction, informative_groups, total_groups = grouped_permutation(
                    frame["regional_discount"].astype(float),
                    group_key,
                    rng,
                )
                altered = frame.copy()
                altered["regional_discount"] = new_values
                scrambled[dataset] = altered
                diagnostics.append(
                    {
                        "control_mode": mode,
                        "seed": seed,
                        "dataset": dataset,
                        "group_columns": "+".join(columns),
                        "changed_discount_fraction": changed_fraction,
                        "informative_groups": informative_groups,
                        "total_groups": total_groups,
                    }
                )
            control_rows, _ = metrics_for_config(scrambled, config, model_name=f"M5_v3_safety_{mode}_control")
            for row in control_rows:
                rows.append({"control_mode": mode, "seed": seed, **row})
    controls = pd.DataFrame(rows)
    comparison_rows: list[dict[str, Any]] = []
    diagnostics_frame = pd.DataFrame(diagnostics)
    for dataset, metric in KEY_CONTROL_METRICS:
        real_value = value(real, dataset, metric)
        for mode, columns in STRONG_CONTROL_MODES:
            subset = controls.loc[controls["dataset"].eq(dataset) & controls["control_mode"].eq(mode)]
            values = subset[metric].astype(float).to_numpy()
            diag_subset = diagnostics_frame.loc[
                diagnostics_frame["dataset"].eq(dataset) & diagnostics_frame["control_mode"].eq(mode)
            ]
            comparison_rows.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "control_mode": mode,
                    "group_columns": "+".join(columns),
                    "real_value": real_value,
                    "control_mean": float(np.nanmean(values)),
                    "control_p05": float(np.nanpercentile(values, 5)),
                    "control_p50": float(np.nanpercentile(values, 50)),
                    "control_p95": float(np.nanpercentile(values, 95)),
                    "real_minus_control_mean": float(real_value - np.nanmean(values)),
                    "empirical_p_control_ge_real": float((np.sum(values >= real_value) + 1) / (len(values) + 1)),
                    "mean_changed_discount_fraction": float(diag_subset["changed_discount_fraction"].mean()),
                    "mean_informative_groups": float(diag_subset["informative_groups"].mean()),
                    "mean_total_groups": float(diag_subset["total_groups"].mean()),
                    "n_control_runs": int(len(values)),
                    "control_interpretability": "low_permutation"
                    if float(diag_subset["changed_discount_fraction"].mean()) < 0.20
                    else "interpretable",
                }
            )
    controls.to_csv(output_dir / "strong_negative_control_runs.csv", index=False)
    controls.to_parquet(output_dir / "strong_negative_control_runs.parquet", index=False)
    diagnostics_frame.to_csv(output_dir / "strong_negative_control_permutation_diagnostics.csv", index=False)
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(output_dir / "strong_negative_control_comparison.csv", index=False)
    comparison.to_parquet(output_dir / "strong_negative_control_comparison.parquet", index=False)
    return controls, comparison


def fmt(value_: Any, digits: int = 3) -> str:
    try:
        if pd.isna(value_):
            return "NA"
        return f"{float(value_):.{digits}f}"
    except Exception:
        return str(value_)


def summarize_decision(critical: pd.DataFrame, controls: pd.DataFrame) -> str:
    false_benign = int(critical["audit_type"].eq("false_benign_plp").sum())
    high_priority_plp = int(
        (
            critical["audit_type"].eq("false_benign_plp")
            & critical["priority_tier"].isin(["P0_manual_review", "P1_manual_review"])
        ).sum()
    )
    interpretable = controls.loc[controls["control_interpretability"].eq("interpretable")]
    weak_specificity = interpretable.loc[
        interpretable["dataset"].eq("abraom_common_benign")
        & interpretable["metric"].eq("specificity")
        & (interpretable["empirical_p_control_ge_real"] > 0.10)
    ]
    weak_plp = interpretable.loc[
        interpretable["dataset"].eq("abraom_pathogenic_present")
        & interpretable["metric"].eq("recall")
        & (interpretable["empirical_p_control_ge_real"] > 0.10)
    ]
    if false_benign > 0 and (not weak_specificity.empty or not weak_plp.empty):
        return (
            "do_not_train_next: prioritize manual critical-error review and external validation; "
            f"{false_benign} P/LP false benign remain, {high_priority_plp} high-priority."
        )
    if not weak_specificity.empty or not weak_plp.empty:
        return "do_not_claim_regional_specificity: stronger controls still explain key gains."
    return "regional_signal_supported_conditionally: proceed only with external sentinel validation."


def write_report(
    output_dir: Path,
    *,
    config: SafetyConfig,
    critical: pd.DataFrame,
    category_summary: pd.DataFrame,
    gene_summary: pd.DataFrame,
    panel: pd.DataFrame,
    controls: pd.DataFrame,
    decision: str,
) -> None:
    false_benign_n = int(critical["audit_type"].eq("false_benign_plp").sum())
    false_pathogenic_n = int(critical["audit_type"].eq("false_pathogenic_common_benign").sum())
    top_genes = gene_summary.groupby("audit_type").head(8)
    lines = [
        "# Regional Signal Validation Next Step",
        "",
        f"Generated at UTC: `{datetime.now(UTC).isoformat()}`",
        "",
        "## Bottom Line",
        "",
        f"Decision: `{decision}`.",
        "",
        "This package audits the locked `M5_v3_safety` outputs. It is not a new training run and it does not use test errors to select a model.",
        "",
        "## Locked Config",
        "",
        "```json",
        json.dumps(asdict(config), indent=2),
        "```",
        "",
        "## Critical Error Audit",
        "",
        f"- P/LP ABRAOM-present false benign variants: `{false_benign_n}`",
        f"- ABRAOM-common benign false pathogenic variants: `{false_pathogenic_n}`",
        f"- Derived review/sentinel rows: `{len(panel)}`",
        "",
        "| Audit type | Category | Tier | n |",
        "|---|---|---|---:|",
    ]
    for row in category_summary.itertuples(index=False):
        lines.append(f"| `{row.audit_type}` | `{row.failure_category}` | `{row.priority_tier}` | {int(row.n)} |")
    lines.extend(["", "## Top Error Genes", "", "| Audit type | Gene | n | Mean priority | Median AF ABRAOM |", "|---|---|---:|---:|---:|"])
    for row in top_genes.itertuples(index=False):
        lines.append(
            f"| `{row.audit_type}` | `{row.GeneSymbol}` | {int(row.n)} | "
            f"{fmt(row.mean_priority)} | {fmt(row.median_af_abraom, 4)} |"
        )
    lines.extend(
        [
            "",
            "## Strong Negative Controls",
            "",
            "| Dataset | Metric | Control | Real | Control mean | P95 | P(control >= real) | Changed discount | Interpretation |",
            "|---|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in controls.itertuples(index=False):
        lines.append(
            f"| `{row.dataset}` | `{row.metric}` | `{row.control_mode}` | {fmt(row.real_value)} | "
            f"{fmt(row.control_mean)} | {fmt(row.control_p95)} | "
            f"{fmt(row.empirical_p_control_ge_real, 4)} | {fmt(row.mean_changed_discount_fraction)} | "
            f"`{row.control_interpretability}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The remaining P/LP false benign variants are the immediate scientific and safety bottleneck.",
            "- Strong controls that preserve AF-related strata are the key falsification test. If they match or exceed the real run, ABRAOM specificity is not proven.",
            "- The derived panel is a review queue, not a clinically curated truth set.",
            "",
            "## Key Artifacts",
            "",
            "- `critical_error_audit/combined_manual_review_queue.csv`",
            "- `critical_error_audit/false_benign_plp_review_queue.csv`",
            "- `critical_error_audit/false_pathogenic_common_benign_review_queue.csv`",
            "- `critical_error_category_summary.csv`",
            "- `critical_error_gene_summary.csv`",
            "- `derived_review_sentinel_panel.csv`",
            "- `strong_negative_control_comparison.csv`",
            "- `strong_negative_control_permutation_diagnostics.csv`",
        ]
    )
    (output_dir / "REGIONAL_SIGNAL_VALIDATION_NEXT_STEP_REPORT.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = load_selected_config(args.m5_v3_dir / "selected_config.json")
    master_enrichment = load_master_enrichment(args.clinvar_master)
    frames = {
        dataset: load_predictions_with_enrichment(args.m5_v3_dir, args.m5_v2_dir, master_enrichment, dataset)
        for dataset in args.datasets
    }
    critical, category_summary, gene_summary = build_critical_audit(frames, args.output_dir)
    panel = build_derived_review_panel(frames, critical, args.output_dir)
    raw_test = load_m5_split(args.raw_test_dir, args.slice_dir, args.datasets, "test")
    _, control_comparison = run_stronger_controls(
        raw_test,
        config,
        seeds=args.control_seeds,
        seed_base=args.seed,
        output_dir=args.output_dir,
    )
    decision = summarize_decision(critical, control_comparison)
    write_report(
        args.output_dir,
        config=config,
        critical=critical,
        category_summary=category_summary,
        gene_summary=gene_summary,
        panel=panel,
        controls=control_comparison,
        decision=decision,
    )
    summary = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "output_dir": str(args.output_dir),
        "decision": decision,
        "critical_errors": {
            "false_benign_plp": int(critical["audit_type"].eq("false_benign_plp").sum()),
            "false_pathogenic_common_benign": int(critical["audit_type"].eq("false_pathogenic_common_benign").sum()),
        },
        "derived_review_sentinel_rows": int(len(panel)),
        "control_modes": [mode for mode, _ in STRONG_CONTROL_MODES],
    }
    (args.output_dir / "regional_signal_validation_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
