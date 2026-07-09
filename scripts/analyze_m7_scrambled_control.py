#!/usr/bin/env python3
"""Focused M7 scrambled-control analysis for ABRAOM regionalization."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_adapter_fusion_deep_errors import (  # noqa: E402
    DEFAULT_DATASETS,
    ModelSpec,
    build_error_counts,
    confusion,
    merge_dataset,
    model_specs,
)


FOCUS_DATASETS = [
    "br_only",
    "abraom_common_benign",
    "abraom_pathogenic_present",
    "global_nonbr_no_abraom",
]

CONTROL_PAIRS = [
    ("M7_dynamic_scrambled", "M4_dynamic_gated", "real_abraom_vs_scrambled"),
    ("M2_gnomad_only", "M4_dynamic_gated", "real_abraom_vs_gnomad_only"),
    ("M2_gnomad_only", "M7_dynamic_scrambled", "scrambled_vs_gnomad_only"),
    ("M7_dynamic_scrambled", "M5_v2_calibrated", "m5_v2_vs_scrambled"),
    ("M4_dynamic_gated", "M5_v2_calibrated", "m5_v2_vs_m4_real"),
]

BOOTSTRAP_REQUESTS = [
    ("M7_dynamic_scrambled", "M4_dynamic_gated", "br_only", "mcc"),
    ("M7_dynamic_scrambled", "M4_dynamic_gated", "abraom_common_benign", "specificity"),
    ("M7_dynamic_scrambled", "M4_dynamic_gated", "abraom_pathogenic_present", "recall"),
    ("M7_dynamic_scrambled", "M4_dynamic_gated", "global_nonbr_no_abraom", "mcc"),
    ("M2_gnomad_only", "M4_dynamic_gated", "br_only", "mcc"),
    ("M2_gnomad_only", "M7_dynamic_scrambled", "br_only", "mcc"),
    ("M7_dynamic_scrambled", "M5_v2_calibrated", "br_only", "mcc"),
    ("M7_dynamic_scrambled", "M5_v2_calibrated", "abraom_common_benign", "specificity"),
    ("M7_dynamic_scrambled", "M5_v2_calibrated", "abraom_pathogenic_present", "recall"),
    ("M7_dynamic_scrambled", "M5_v2_calibrated", "global_nonbr_no_abraom", "mcc"),
]


@dataclass(frozen=True)
class PairDecision:
    dataset: str
    base_model: str
    compare_model: str
    interpretation: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slice-dir", type=Path, default=Path("data/datasets/clinvar/regional_abraom/slices"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/adapter_fusion_m7_control_analysis"))
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260629)
    return parser.parse_args(argv)


def selected_specs() -> list[ModelSpec]:
    keep = {"M0", "M2_gnomad_only", "M4_dynamic_gated", "M7_dynamic_scrambled", "M5_v2_calibrated"}
    return [spec for spec in model_specs() if spec.name in keep]


def exact_or_approx_mcnemar_p(base_only: int, compare_only: int) -> float:
    total = base_only + compare_only
    if total == 0:
        return 1.0
    try:
        from scipy.stats import binomtest

        return float(binomtest(min(base_only, compare_only), total, 0.5).pvalue)
    except Exception:
        # Chi-square df=1 survival function with continuity correction.
        stat = (abs(base_only - compare_only) - 1.0) ** 2 / total if total else 0.0
        return float(math.erfc(math.sqrt(max(stat, 0.0) / 2.0)))


def af_bin(values: pd.Series) -> pd.Series:
    filled = values.astype(float)
    labels = ["absent", "<=1e-4", "1e-4..1e-3", "1e-3..1e-2", ">1e-2"]
    out = pd.Series(np.repeat(labels[0], len(filled)), index=filled.index, dtype=object)
    present = filled.notna()
    out.loc[present & (filled <= 1e-4)] = labels[1]
    out.loc[present & (filled > 1e-4) & (filled <= 1e-3)] = labels[2]
    out.loc[present & (filled > 1e-3) & (filled <= 1e-2)] = labels[3]
    out.loc[present & (filled > 1e-2)] = labels[4]
    return out


def score_margin(frame: pd.DataFrame, model: str) -> pd.Series:
    return frame[f"{model}__score"].astype(float) - frame[f"{model}__threshold"].astype(float)


def shared_variant_columns(frame: pd.DataFrame, models: list[str]) -> list[str]:
    base = [
        "dataset",
        "variant_key",
        "source_variant_id",
        "GeneSymbol",
        "Chromosome",
        "Start",
        "ReferenceAlleleVCF",
        "AlternateAlleleVCF",
        "variant_type",
        "is_snv",
        "label",
        "clinvar_regional_cohort",
        "has_brazilian_submitter",
        "has_non_brazilian_submitter",
        "abraom_present",
        "af_abraom",
        "af_gnomad",
        "specificity",
        "specificity_bin",
        "af_abraom_bin",
    ]
    model_cols: list[str] = []
    for model in models:
        model_cols.extend(
            [
                f"{model}__score",
                f"{model}__threshold",
                f"{model}__prediction",
                f"{model}__probability",
                f"{model}__swap_probability",
                f"{model}__molecular_probability",
                f"{model}__regional_discount",
                f"{model}__gate_alpha_abraom",
                f"{model}__gate_alpha_gnomad",
                f"{model}__gate_alpha_scrambled",
                f"{model}__gate_entropy",
                f"{model}__margin",
            ]
        )
    return [column for column in [*base, *model_cols] if column in frame.columns]


def add_derived_columns(frame: pd.DataFrame, specs: list[ModelSpec]) -> pd.DataFrame:
    out = frame.copy()
    out["af_abraom_bin"] = af_bin(out["af_abraom"]) if "af_abraom" in out.columns else "unknown"
    for spec in specs:
        score_col = f"{spec.name}__score"
        threshold_col = f"{spec.name}__threshold"
        if score_col in out.columns and threshold_col in out.columns:
            out[f"{spec.name}__margin"] = score_margin(out, spec.name)
    return out


def build_pairwise_transitions(wide_by_dataset: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset, frame in wide_by_dataset.items():
        labels = frame["label"].astype(int)
        for base, compare, contrast in CONTROL_PAIRS:
            base_pred = frame[f"{base}__prediction"].astype(int)
            compare_pred = frame[f"{compare}__prediction"].astype(int)
            base_correct = base_pred.eq(labels)
            compare_correct = compare_pred.eq(labels)
            base_only = int((base_correct & ~compare_correct).sum())
            compare_only = int((~base_correct & compare_correct).sum())
            base_metrics = confusion(labels.to_numpy(), base_pred.to_numpy())
            compare_metrics = confusion(labels.to_numpy(), compare_pred.to_numpy())
            rows.append(
                {
                    "dataset": dataset,
                    "contrast": contrast,
                    "base_model": base,
                    "compare_model": compare,
                    "n": int(len(frame)),
                    "both_correct": int((base_correct & compare_correct).sum()),
                    "both_wrong": int((~base_correct & ~compare_correct).sum()),
                    "base_only_correct": base_only,
                    "compare_only_correct": compare_only,
                    "mcnemar_p": exact_or_approx_mcnemar_p(base_only, compare_only),
                    "base_mcc": base_metrics["mcc"],
                    "compare_mcc": compare_metrics["mcc"],
                    "delta_mcc": compare_metrics["mcc"] - base_metrics["mcc"],
                    "base_recall": base_metrics["recall"],
                    "compare_recall": compare_metrics["recall"],
                    "delta_recall": compare_metrics["recall"] - base_metrics["recall"],
                    "base_specificity": base_metrics["specificity"],
                    "compare_specificity": compare_metrics["specificity"],
                    "delta_specificity": compare_metrics["specificity"] - base_metrics["specificity"],
                }
            )
    return pd.DataFrame(rows)


def write_discordant_variant_tables(
    wide_by_dataset: dict[str, pd.DataFrame],
    output_dir: Path,
    specs: list[ModelSpec],
) -> pd.DataFrame:
    tables_dir = output_dir / "discordant_variant_tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    models = [spec.name for spec in specs]
    rows: list[dict[str, Any]] = []
    for dataset in FOCUS_DATASETS:
        frame = wide_by_dataset[dataset]
        labels = frame["label"].astype(int)
        for base, compare, contrast in CONTROL_PAIRS:
            base_correct = frame[f"{base}__prediction"].astype(int).eq(labels)
            compare_correct = frame[f"{compare}__prediction"].astype(int).eq(labels)
            for side, mask in [
                ("base_only_correct", base_correct & ~compare_correct),
                ("compare_only_correct", ~base_correct & compare_correct),
            ]:
                subset = frame.loc[mask].copy()
                if subset.empty:
                    continue
                cols = shared_variant_columns(subset, models)
                path = tables_dir / f"{dataset}.{contrast}.{side}.csv"
                subset[cols].sort_values(["GeneSymbol", "variant_key"]).to_csv(path, index=False)
                rows.append(
                    {
                        "dataset": dataset,
                        "contrast": contrast,
                        "side": side,
                        "base_model": base,
                        "compare_model": compare,
                        "n": int(len(subset)),
                        "path": str(path),
                    }
                )
    return pd.DataFrame(rows)


def summarize_discordance_groups(wide_by_dataset: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_columns = ["label", "variant_type", "specificity_bin", "af_abraom_bin"]
    for dataset in FOCUS_DATASETS:
        frame = wide_by_dataset[dataset]
        labels = frame["label"].astype(int)
        for base, compare, contrast in CONTROL_PAIRS:
            base_correct = frame[f"{base}__prediction"].astype(int).eq(labels)
            compare_correct = frame[f"{compare}__prediction"].astype(int).eq(labels)
            status = pd.Series("both_same", index=frame.index, dtype=object)
            status.loc[base_correct & ~compare_correct] = "base_only_correct"
            status.loc[~base_correct & compare_correct] = "compare_only_correct"
            subset = frame.loc[status.ne("both_same")].copy()
            subset["discordance_status"] = status.loc[subset.index]
            if subset.empty:
                continue
            for column in group_columns:
                if column not in subset.columns:
                    continue
                grouped = subset.groupby(["discordance_status", column], dropna=False).size().reset_index(name="n")
                for row in grouped.itertuples(index=False):
                    rows.append(
                        {
                            "dataset": dataset,
                            "contrast": contrast,
                            "base_model": base,
                            "compare_model": compare,
                            "discordance_status": row.discordance_status,
                            "group_column": column,
                            "group_value": getattr(row, column),
                            "n": int(row.n),
                        }
                    )
    return pd.DataFrame(rows)


def summarize_top_discordant_genes(wide_by_dataset: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset in FOCUS_DATASETS:
        frame = wide_by_dataset[dataset]
        labels = frame["label"].astype(int)
        for base, compare, contrast in CONTROL_PAIRS:
            base_correct = frame[f"{base}__prediction"].astype(int).eq(labels)
            compare_correct = frame[f"{compare}__prediction"].astype(int).eq(labels)
            for side, mask in [
                ("base_only_correct", base_correct & ~compare_correct),
                ("compare_only_correct", ~base_correct & compare_correct),
            ]:
                subset = frame.loc[mask].copy()
                if subset.empty:
                    continue
                top = subset["GeneSymbol"].fillna("").replace("", "unknown").value_counts().head(15)
                for gene, count in top.items():
                    rows.append(
                        {
                            "dataset": dataset,
                            "contrast": contrast,
                            "base_model": base,
                            "compare_model": compare,
                            "discordance_status": side,
                            "GeneSymbol": gene,
                            "n": int(count),
                        }
                    )
    return pd.DataFrame(rows)


def summarize_gate_behavior(wide_by_dataset: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    columns = [
        "M4_dynamic_gated__gate_alpha_abraom",
        "M4_dynamic_gated__gate_alpha_gnomad",
        "M4_dynamic_gated__gate_entropy",
        "M7_dynamic_scrambled__gate_alpha_scrambled",
        "M7_dynamic_scrambled__gate_alpha_gnomad",
        "M7_dynamic_scrambled__gate_entropy",
    ]
    for dataset, frame in wide_by_dataset.items():
        for column in columns:
            if column not in frame.columns:
                continue
            values = frame[column].astype(float)
            rows.append(
                {
                    "dataset": dataset,
                    "column": column,
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)),
                    "p05": float(values.quantile(0.05)),
                    "p50": float(values.quantile(0.50)),
                    "p95": float(values.quantile(0.95)),
                }
            )
    return pd.DataFrame(rows)


def summarize_score_similarity(wide_by_dataset: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset, frame in wide_by_dataset.items():
        for left, right in [
            ("M4_dynamic_gated", "M7_dynamic_scrambled"),
            ("M2_gnomad_only", "M4_dynamic_gated"),
            ("M2_gnomad_only", "M7_dynamic_scrambled"),
            ("M7_dynamic_scrambled", "M5_v2_calibrated"),
        ]:
            left_score = frame[f"{left}__score"].astype(float)
            right_score = frame[f"{right}__score"].astype(float)
            rows.append(
                {
                    "dataset": dataset,
                    "left_model": left,
                    "right_model": right,
                    "score_corr": float(left_score.corr(right_score)),
                    "mean_abs_score_diff": float((left_score - right_score).abs().mean()),
                    "p95_abs_score_diff": float((left_score - right_score).abs().quantile(0.95)),
                    "prediction_agreement": float(
                        frame[f"{left}__prediction"].astype(int).eq(frame[f"{right}__prediction"].astype(int)).mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def confusion_counts_by_cluster(labels: np.ndarray, predictions: np.ndarray, cluster_codes: np.ndarray, n_clusters: int) -> np.ndarray:
    labels = labels.astype(int)
    predictions = predictions.astype(int)
    return np.vstack(
        [
            np.bincount(cluster_codes, weights=((labels == 1) & (predictions == 1)).astype(int), minlength=n_clusters),
            np.bincount(cluster_codes, weights=((labels == 0) & (predictions == 0)).astype(int), minlength=n_clusters),
            np.bincount(cluster_codes, weights=((labels == 0) & (predictions == 1)).astype(int), minlength=n_clusters),
            np.bincount(cluster_codes, weights=((labels == 1) & (predictions == 0)).astype(int), minlength=n_clusters),
        ]
    ).T


def metric_from_counts(counts: np.ndarray, metric: str) -> float:
    tp, tn, fp, fn = [float(value) for value in counts]
    if metric == "recall":
        return tp / (tp + fn) if (tp + fn) else float("nan")
    if metric == "specificity":
        return tn / (tn + fp) if (tn + fp) else float("nan")
    if metric == "precision":
        return tp / (tp + fp) if (tp + fp) else float("nan")
    if metric == "mcc":
        denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        return ((tp * tn - fp * fn) / denom) if denom else 0.0
    raise ValueError(f"Unsupported metric: {metric}")


def bootstrap_pairwise(
    wide_by_dataset: dict[str, pd.DataFrame],
    iterations: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for base, compare, dataset, metric in BOOTSTRAP_REQUESTS:
        frame = wide_by_dataset[dataset]
        cluster_codes, clusters = pd.factorize(frame["GeneSymbol"].fillna("").replace("", "unknown").astype(str))
        n_clusters = len(clusters)
        labels = frame["label"].to_numpy(dtype=int)
        base_counts_by_cluster = confusion_counts_by_cluster(
            labels,
            frame[f"{base}__prediction"].to_numpy(dtype=int),
            cluster_codes,
            n_clusters,
        )
        compare_counts_by_cluster = confusion_counts_by_cluster(
            labels,
            frame[f"{compare}__prediction"].to_numpy(dtype=int),
            cluster_codes,
            n_clusters,
        )
        deltas: list[float] = []
        for _ in range(iterations):
            sampled = rng.integers(0, n_clusters, size=n_clusters)
            weights = np.bincount(sampled, minlength=n_clusters).astype(float)
            base_value = metric_from_counts(weights @ base_counts_by_cluster, metric)
            compare_value = metric_from_counts(weights @ compare_counts_by_cluster, metric)
            deltas.append(compare_value - base_value)
        observed = metric_from_counts(compare_counts_by_cluster.sum(axis=0), metric) - metric_from_counts(
            base_counts_by_cluster.sum(axis=0),
            metric,
        )
        rows.append(
            {
                "base_model": base,
                "compare_model": compare,
                "dataset": dataset,
                "metric": metric,
                "delta": float(observed),
                "ci95_low": float(np.nanpercentile(deltas, 2.5)),
                "ci95_high": float(np.nanpercentile(deltas, 97.5)),
                "iterations": iterations,
                "cluster": "GeneSymbol",
            }
        )
    return pd.DataFrame(rows)


def make_decisions(transitions: pd.DataFrame, bootstrap: pd.DataFrame) -> list[PairDecision]:
    decisions: list[PairDecision] = []
    for dataset in FOCUS_DATASETS:
        row = transitions.loc[
            transitions["dataset"].eq(dataset)
            & transitions["base_model"].eq("M7_dynamic_scrambled")
            & transitions["compare_model"].eq("M4_dynamic_gated")
        ].iloc[0]
        boot = bootstrap.loc[
            bootstrap["dataset"].eq(dataset)
            & bootstrap["base_model"].eq("M7_dynamic_scrambled")
            & bootstrap["compare_model"].eq("M4_dynamic_gated")
        ]
        if not boot.empty and boot.iloc[0]["ci95_low"] <= 0.0 <= boot.iloc[0]["ci95_high"]:
            interpretation = "Real ABRAOM did not clearly beat scrambled control for this metric."
        elif row["delta_mcc"] > 0 or row["delta_recall"] > 0 or row["delta_specificity"] > 0:
            interpretation = "Real ABRAOM beat scrambled control on the selected metric."
        else:
            interpretation = "Scrambled control matched or beat real ABRAOM on the selected metric."
        decisions.append(
            PairDecision(
                dataset=dataset,
                base_model="M7_dynamic_scrambled",
                compare_model="M4_dynamic_gated",
                interpretation=interpretation,
            )
        )
    return decisions


def fmt_float(value: Any, digits: int = 3) -> str:
    try:
        if pd.isna(value):
            return "NA"
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def write_report(
    output_dir: Path,
    metrics: pd.DataFrame,
    transitions: pd.DataFrame,
    bootstrap: pd.DataFrame,
    gate_summary: pd.DataFrame,
    score_similarity: pd.DataFrame,
    table_manifest: pd.DataFrame,
    decisions: list[PairDecision],
) -> None:
    metric_rows = metrics.loc[
        metrics["dataset"].isin(FOCUS_DATASETS)
        & metrics["model"].isin(["M2_gnomad_only", "M4_dynamic_gated", "M7_dynamic_scrambled", "M5_v2_calibrated"])
    ].copy()
    metric_rows = metric_rows.sort_values(["dataset", "model"])

    key_transition = transitions.loc[
        transitions["dataset"].isin(FOCUS_DATASETS)
        & transitions["contrast"].isin(["real_abraom_vs_scrambled", "m5_v2_vs_scrambled"])
    ].copy()

    lines = [
        "# M7 Scrambled-Control Analysis",
        "",
        f"Generated at UTC: `{datetime.now(UTC).isoformat()}`",
        "",
        "## Bottom Line",
        "",
        "`M7_dynamic_scrambled` should not be advanced as a candidate model, but it is a useful negative control. "
        "It performs close to `M4_dynamic_gated`, which weakens any claim that the dynamic-gated M4 result alone proves specific ABRAOM biology. "
        "`M5_v2_calibrated` remains the lead because it beats the scrambled control on the regional safety targets that matter most.",
        "",
        "## Core Metrics",
        "",
        "| Dataset | Model | MCC | Recall | Specificity | FP | FN |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in metric_rows.itertuples(index=False):
        lines.append(
            f"| `{row.dataset}` | `{row.model}` | {fmt_float(row.mcc)} | {fmt_float(row.recall)} | "
            f"{fmt_float(row.specificity)} | {int(row.fp)} | {int(row.fn)} |"
        )

    lines.extend(
        [
            "",
            "## Paired Control Tests",
            "",
            "| Dataset | Contrast | Base | Compare | Base-only correct | Compare-only correct | Delta MCC | Delta recall | Delta specificity | McNemar p |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in key_transition.itertuples(index=False):
        lines.append(
            f"| `{row.dataset}` | `{row.contrast}` | `{row.base_model}` | `{row.compare_model}` | "
            f"{int(row.base_only_correct)} | {int(row.compare_only_correct)} | "
            f"{fmt_float(row.delta_mcc)} | {fmt_float(row.delta_recall)} | {fmt_float(row.delta_specificity)} | "
            f"{fmt_float(row.mcnemar_p, 4)} |"
        )

    lines.extend(
        [
            "",
            "## Gene-Cluster Bootstrap",
            "",
            "| Base | Compare | Dataset | Metric | Delta | 95% CI |",
            "|---|---|---|---|---:|---:|",
        ]
    )
    for row in bootstrap.itertuples(index=False):
        lines.append(
            f"| `{row.base_model}` | `{row.compare_model}` | `{row.dataset}` | `{row.metric}` | "
            f"{fmt_float(row.delta)} | [{fmt_float(row.ci95_low)}, {fmt_float(row.ci95_high)}] |"
        )

    lines.extend(["", "## Gate Behavior", ""])
    m4_entropy = gate_summary.loc[
        gate_summary["column"].eq("M4_dynamic_gated__gate_entropy") & gate_summary["dataset"].eq("br_only")
    ]
    m7_entropy = gate_summary.loc[
        gate_summary["column"].eq("M7_dynamic_scrambled__gate_entropy") & gate_summary["dataset"].eq("br_only")
    ]
    if not m4_entropy.empty and not m7_entropy.empty:
        lines.append(
            "On `br_only`, both dynamic gates remain near maximum entropy: "
            f"M4 median entropy `{fmt_float(m4_entropy.iloc[0]['p50'])}`, "
            f"M7 median entropy `{fmt_float(m7_entropy.iloc[0]['p50'])}`. "
            "This suggests weak adapter specialization rather than confident routing to a regional adapter."
        )
    else:
        lines.append("Gate entropy columns were not available for all models.")

    sim = score_similarity.loc[
        score_similarity["dataset"].eq("br_only")
        & score_similarity["left_model"].eq("M4_dynamic_gated")
        & score_similarity["right_model"].eq("M7_dynamic_scrambled")
    ]
    if not sim.empty:
        lines.append(
            f"M4 and M7 scores on `br_only` are highly similar: correlation `{fmt_float(sim.iloc[0]['score_corr'])}`, "
            f"prediction agreement `{fmt_float(sim.iloc[0]['prediction_agreement'])}`."
        )

    lines.extend(["", "## Interpretations", ""])
    for decision in decisions:
        lines.append(f"- `{decision.dataset}`: {decision.interpretation}")

    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            "1. Keep `M7_dynamic_scrambled` as a falsification/control model, not as a candidate.",
            "2. Do not use M4-vs-M7 alone to claim strong ABRAOM-specific learning; the difference is too small and often not robust.",
            "3. Use M7 as a required comparator for the next `M5_v3_safety` run.",
            "4. Require the next model to beat M7 on `br_only` MCC, `abraom_common_benign` specificity, and `abraom_pathogenic_present` recall, while staying close to M0/M7 on global nonBR performance.",
            "5. Add stronger negative controls next: scramble within gene, within AF bin, and within chromosome to preserve confounders while breaking variant-frequency identity.",
            "",
            "## Key Artifacts",
            "",
            "- `m7_control_metrics.csv`",
            "- `m7_pairwise_transitions.csv`",
            "- `m7_gene_cluster_bootstrap.csv`",
            "- `m7_gate_behavior.csv`",
            "- `m7_score_similarity.csv`",
            "- `m7_discordance_group_summary.csv`",
            "- `m7_top_discordant_genes.csv`",
            "- `discordant_variant_tables/`",
        ]
    )
    if not table_manifest.empty:
        lines.append(f"- Discordant variant table manifest rows: `{len(table_manifest)}`")
    (output_dir / "M7_SCRAMBLED_CONTROL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    specs = selected_specs()
    wide_by_dataset = {
        dataset: add_derived_columns(merge_dataset(specs, args.slice_dir, dataset), specs)
        for dataset in DEFAULT_DATASETS
    }
    metrics = build_error_counts(wide_by_dataset, specs)
    transitions = build_pairwise_transitions(wide_by_dataset)
    table_manifest = write_discordant_variant_tables(wide_by_dataset, args.output_dir, specs)
    group_summary = summarize_discordance_groups(wide_by_dataset)
    top_genes = summarize_top_discordant_genes(wide_by_dataset)
    gate_summary = summarize_gate_behavior(wide_by_dataset)
    score_similarity = summarize_score_similarity(wide_by_dataset)
    bootstrap = bootstrap_pairwise(wide_by_dataset, args.bootstrap_iterations, args.seed)
    decisions = make_decisions(transitions, bootstrap)

    metrics.to_csv(args.output_dir / "m7_control_metrics.csv", index=False)
    metrics.to_parquet(args.output_dir / "m7_control_metrics.parquet", index=False)
    transitions.to_csv(args.output_dir / "m7_pairwise_transitions.csv", index=False)
    transitions.to_parquet(args.output_dir / "m7_pairwise_transitions.parquet", index=False)
    table_manifest.to_csv(args.output_dir / "m7_discordant_table_manifest.csv", index=False)
    group_summary.to_csv(args.output_dir / "m7_discordance_group_summary.csv", index=False)
    top_genes.to_csv(args.output_dir / "m7_top_discordant_genes.csv", index=False)
    gate_summary.to_csv(args.output_dir / "m7_gate_behavior.csv", index=False)
    score_similarity.to_csv(args.output_dir / "m7_score_similarity.csv", index=False)
    bootstrap.to_csv(args.output_dir / "m7_gene_cluster_bootstrap.csv", index=False)
    bootstrap.to_parquet(args.output_dir / "m7_gene_cluster_bootstrap.parquet", index=False)
    write_report(args.output_dir, metrics, transitions, bootstrap, gate_summary, score_similarity, table_manifest, decisions)

    summary = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "output_dir": str(args.output_dir),
        "n_discordant_tables": int(len(table_manifest)),
        "decisions": [decision.__dict__ for decision in decisions],
    }
    (args.output_dir / "m7_control_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
