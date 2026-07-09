#!/usr/bin/env python3
"""Final analysis for holdout-tuned M5_v2 regional calibration."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.clinvar.metrics import classification_metrics  # noqa: E402

EPS = 1e-6
DEFAULT_DATASETS = [
    "br_only",
    "br_any",
    "regional_benchmark_any",
    "abraom_common_benign",
    "abraom_pathogenic_present",
    "abraom_pathogenic_common",
    "global_nonbr_no_abraom",
    "nonbr_only",
]


@dataclass(frozen=True)
class ModelSource:
    name: str
    root: Path
    kind: str
    threshold: float = 0.5


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--m5-v2-calibrated-dir",
        type=Path,
        default=Path("artifacts/clinvar_regional_eval/m5_v2_calibrated_holdout_tuned"),
    )
    parser.add_argument(
        "--m5-v2-raw-dir",
        type=Path,
        default=Path("artifacts/clinvar_regional_eval/m5_v2_bounded_regional_eval_beatv10_v1_sagemaker/_extracted"),
    )
    parser.add_argument("--model-root", type=Path, default=Path("artifacts/clinvar_regional_eval"))
    parser.add_argument(
        "--calibration-v2-dir",
        type=Path,
        default=Path("artifacts/clinvar_regional_calibration_v2_holdout_tuned"),
    )
    parser.add_argument("--slice-dir", type=Path, default=Path("data/datasets/clinvar/regional_abraom/slices"))
    parser.add_argument(
        "--baseline-csv",
        type=Path,
        default=Path(
            "artifacts/clinvar_regional_comparison/"
            "m0_m4_m5_m6_m5cal_m6cal_m7_m5v2cal_regional_test_summary.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/clinvar_regional_eval/m5_v2_calibrated_holdout_tuned/final_analysis"),
    )
    parser.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260623)
    return parser.parse_args(argv)


def logit(values: pd.Series | np.ndarray) -> np.ndarray:
    probs = np.clip(np.asarray(values, dtype=np.float64), EPS, 1.0 - EPS)
    return np.log(probs / (1.0 - probs))


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -60.0, 60.0)))


def normalized_score(probability: pd.Series | np.ndarray, threshold: float) -> np.ndarray:
    return sigmoid(logit(probability) - logit(np.asarray(threshold, dtype=np.float64)))


def read_threshold(model_dir: Path) -> float:
    summary_path = model_dir / "regional_eval_summary.json"
    if not summary_path.is_file():
        return 0.5
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return float(summary.get("threshold", 0.5))


def load_metadata(slice_dir: Path, dataset: str, split: str = "test") -> pd.DataFrame:
    path = slice_dir / f"{dataset}.parquet"
    frame = pd.read_parquet(path).reset_index().rename(columns={"index": "original_index"})
    split_values = frame["split_within_gene"].astype(str).str.lower()
    frame = frame.loc[split_values == split.lower()].copy()
    columns = [
        "original_index",
        "variant_key",
        "GeneSymbol",
        "Chromosome",
        "Start",
        "ReferenceAlleleVCF",
        "AlternateAlleleVCF",
        "variant_type",
        "is_snv",
        "clinvar_regional_cohort",
        "has_brazilian_submitter",
        "abraom_present",
        "af_abraom",
        "af_gnomad",
        "specificity",
        "specificity_bin",
    ]
    return frame[[column for column in columns if column in frame.columns]]


def load_model_scores(source: ModelSource, dataset: str) -> pd.DataFrame:
    path = source.root / f"{dataset}.test.predictions.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"Missing predictions: {path}")
    frame = pd.read_parquet(path).copy()
    if source.kind == "raw_probability":
        frame["score"] = normalized_score(frame["probability"], source.threshold)
        frame["threshold_for_score"] = 0.5
    elif source.kind == "calibrated_v2":
        frame["score"] = normalized_score(frame["score_used"], frame["threshold_used"])
        frame["threshold_for_score"] = 0.5
    elif source.kind == "regional_score":
        frame["score"] = frame["regional_score"].astype(float)
        frame["threshold_for_score"] = 0.5
    else:
        raise ValueError(f"Unsupported model source kind: {source.kind!r}")
    return frame[["dataset", "split", "original_index", "label", "score"]]


def metric_value(labels: np.ndarray, scores: np.ndarray, metric: str) -> float:
    metrics = classification_metrics(labels.astype(np.int64), scores.astype(np.float64), 0.5)
    return float(metrics[metric])


def paired_delta(frame: pd.DataFrame, metric: str) -> float:
    labels = frame["label"].to_numpy(dtype=np.int64)
    return metric_value(labels, frame["score_compare"].to_numpy(), metric) - metric_value(
        labels,
        frame["score_base"].to_numpy(),
        metric,
    )


def bootstrap_delta(
    frame: pd.DataFrame,
    *,
    metric: str,
    iterations: int,
    seed: int,
) -> tuple[float, float, float]:
    estimate = paired_delta(frame, metric)
    if iterations <= 0:
        return estimate, float("nan"), float("nan")
    clusters = frame["cluster"].fillna("").astype(str)
    clusters = clusters.where(clusters != "", frame["original_index"].astype(str))
    unique_clusters = clusters.unique()
    by_cluster = {cluster: np.flatnonzero(clusters.to_numpy() == cluster) for cluster in unique_clusters}
    rng = np.random.default_rng(seed)
    deltas = np.empty(iterations, dtype=np.float64)
    labels = frame["label"].to_numpy(dtype=np.int64)
    base_scores = frame["score_base"].to_numpy(dtype=np.float64)
    compare_scores = frame["score_compare"].to_numpy(dtype=np.float64)
    for index in range(iterations):
        sampled = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)
        row_indices = np.concatenate([by_cluster[cluster] for cluster in sampled])
        deltas[index] = metric_value(labels[row_indices], compare_scores[row_indices], metric) - metric_value(
            labels[row_indices],
            base_scores[row_indices],
            metric,
        )
    low, high = np.quantile(deltas, [0.025, 0.975])
    return float(estimate), float(low), float(high)


def build_sources(args: argparse.Namespace) -> dict[str, ModelSource]:
    return {
        "M0": ModelSource(
            "M0",
            args.model_root / "m0_nonbr_beatv10_v1_sagemaker",
            "raw_probability",
            read_threshold(args.model_root / "m0_nonbr_beatv10_v1_sagemaker"),
        ),
        "M5_calibrated": ModelSource(
            "M5_calibrated",
            args.calibration_v2_dir / "M5_calibrated",
            "regional_score",
        ),
        "M6_calibrated": ModelSource(
            "M6_calibrated",
            args.calibration_v2_dir / "M6_calibrated",
            "regional_score",
        ),
        "M7_scrambled": ModelSource(
            "M7_scrambled",
            args.calibration_v2_dir / "M7_scrambled",
            "regional_score",
        ),
        "M5_v2": ModelSource(
            "M5_v2",
            args.m5_v2_raw_dir,
            "raw_probability",
            read_threshold(args.m5_v2_raw_dir),
        ),
        "M5_v2_calibrated": ModelSource(
            "M5_v2_calibrated",
            args.m5_v2_calibrated_dir,
            "calibrated_v2",
        ),
    }


def build_bootstrap(args: argparse.Namespace, sources: dict[str, ModelSource], output_dir: Path) -> pd.DataFrame:
    comparisons = [
        ("M0", "M5_v2_calibrated", "br_only", "mcc"),
        ("M0", "M5_v2_calibrated", "abraom_common_benign", "specificity"),
        ("M0", "M5_v2_calibrated", "abraom_pathogenic_present", "recall"),
        ("M0", "M5_v2_calibrated", "global_nonbr_no_abraom", "mcc"),
        ("M0", "M5_v2_calibrated", "global_nonbr_no_abraom", "specificity"),
        ("M5_calibrated", "M5_v2_calibrated", "br_only", "mcc"),
        ("M5_calibrated", "M5_v2_calibrated", "abraom_common_benign", "specificity"),
        ("M5_calibrated", "M5_v2_calibrated", "abraom_pathogenic_present", "recall"),
        ("M7_scrambled", "M5_v2_calibrated", "br_only", "mcc"),
        ("M7_scrambled", "M5_v2_calibrated", "abraom_common_benign", "specificity"),
        ("M7_scrambled", "M5_v2_calibrated", "abraom_pathogenic_present", "recall"),
        ("M5_v2", "M5_v2_calibrated", "abraom_pathogenic_present", "recall"),
    ]
    rows: list[dict[str, Any]] = []
    for index, (base_model, compare_model, dataset, metric) in enumerate(comparisons):
        base = load_model_scores(sources[base_model], dataset)
        compare = load_model_scores(sources[compare_model], dataset)
        metadata = load_metadata(args.slice_dir, dataset)
        frame = base.merge(
            compare[["original_index", "score"]],
            on="original_index",
            suffixes=("_base", "_compare"),
            validate="one_to_one",
        ).merge(
            metadata[["original_index", "GeneSymbol"]],
            on="original_index",
            how="left",
            validate="one_to_one",
        )
        frame["cluster"] = frame["GeneSymbol"].fillna("").astype(str)
        delta, low, high = bootstrap_delta(
            frame,
            metric=metric,
            iterations=args.bootstrap_iterations,
            seed=args.seed + index,
        )
        rows.append(
            {
                "base_model": base_model,
                "compare_model": compare_model,
                "dataset": dataset,
                "metric": metric,
                "delta": delta,
                "ci95_low": low,
                "ci95_high": high,
                "iterations": args.bootstrap_iterations,
                "cluster": "GeneSymbol",
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "cluster_bootstrap_intervals.csv", index=False)
    result.to_parquet(output_dir / "cluster_bootstrap_intervals.parquet", index=False)
    return result


def enrich_predictions(args: argparse.Namespace, dataset: str) -> pd.DataFrame:
    predictions = pd.read_parquet(args.m5_v2_calibrated_dir / f"{dataset}.test.predictions.parquet").copy()
    metadata = load_metadata(args.slice_dir, dataset)
    return predictions.merge(metadata, on="original_index", how="left", validate="one_to_one")


def false_benign_category(row: pd.Series) -> str:
    if float(row["molecular_probability"]) < float(row["threshold_used"]):
        return "weak_molecular_score"
    if float(row["regional_discount"]) >= 0.5:
        return "regional_discount_still_too_strong"
    if float(row["score_used"]) >= float(row["threshold_used"]) * 0.8:
        return "threshold_borderline"
    if float(row.get("af_abraom", 0.0) or 0.0) >= 0.01:
        return "common_plp_founder_or_recessive_review"
    return "needs_manual_review"


def false_pathogenic_category(row: pd.Series) -> str:
    if float(row["molecular_probability"]) >= 0.8:
        return "high_molecular_score"
    if float(row["regional_discount"]) <= 0.1:
        return "insufficient_regional_discount"
    if float(row["score_used"]) < float(row["threshold_used"]) * 1.2:
        return "threshold_borderline"
    return "needs_manual_review"


def build_error_tables(args: argparse.Namespace, output_dir: Path) -> dict[str, Path]:
    error_dir = output_dir / "error_analysis"
    error_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    common_columns = [
        "dataset",
        "split",
        "original_index",
        "variant_key",
        "GeneSymbol",
        "Chromosome",
        "Start",
        "ReferenceAlleleVCF",
        "AlternateAlleleVCF",
        "label",
        "prediction_used",
        "score_used",
        "threshold_used",
        "molecular_probability",
        "regional_score_calibrated",
        "regional_discount",
        "af_abraom",
        "af_gnomad",
        "specificity",
        "specificity_bin",
        "failure_category",
    ]
    benign = enrich_predictions(args, "abraom_common_benign")
    false_pathogenic = benign.loc[benign["prediction_used"] == 1].copy()
    false_pathogenic["failure_category"] = false_pathogenic.apply(false_pathogenic_category, axis=1)
    path = error_dir / "M5_v2_calibrated.false_pathogenic_abraom_common_benign.csv"
    false_pathogenic[[column for column in common_columns if column in false_pathogenic.columns]].to_csv(
        path,
        index=False,
    )
    outputs["false_pathogenic_abraom_common_benign"] = path

    pathogenic = enrich_predictions(args, "abraom_pathogenic_present")
    false_benign = pathogenic.loc[pathogenic["prediction_used"] == 0].copy()
    false_benign["failure_category"] = false_benign.apply(false_benign_category, axis=1)
    path = error_dir / "M5_v2_calibrated.false_benign_abraom_pathogenic_present.csv"
    false_benign[[column for column in common_columns if column in false_benign.columns]].to_csv(path, index=False)
    outputs["false_benign_abraom_pathogenic_present"] = path

    rescued = pathogenic.loc[pathogenic["prediction_used"] == 1].copy()
    raw = pd.read_parquet(args.m5_v2_raw_dir / "abraom_pathogenic_present.test.predictions.parquet")
    raw_threshold = read_threshold(args.m5_v2_raw_dir)
    raw["raw_prediction"] = (raw["probability"] >= raw_threshold).astype(int)
    rescued = rescued.merge(raw[["original_index", "raw_prediction"]], on="original_index", how="left")
    rescued = rescued.loc[rescued["raw_prediction"] == 0].copy()
    path = error_dir / "M5_v2_calibrated.rescued_from_raw_false_benign_abraom_pathogenic_present.csv"
    rescued[[column for column in common_columns if column in rescued.columns]].to_csv(path, index=False)
    outputs["rescued_from_raw_false_benign_abraom_pathogenic_present"] = path
    return outputs


def bin_af(values: pd.Series) -> pd.Series:
    bins = [-np.inf, 0.0, 1e-4, 1e-3, 5e-3, 1e-2, 5e-2, np.inf]
    labels = ["missing_or_zero", "<0.01%", "0.01-0.1%", "0.1-0.5%", "0.5-1%", "1-5%", ">5%"]
    return pd.cut(pd.to_numeric(values, errors="coerce").fillna(0.0), bins=bins, labels=labels)


def diagnostic_group(frame: pd.DataFrame, group_column: str) -> pd.DataFrame:
    rows = []
    for group, part in frame.groupby(group_column, dropna=False, observed=False):
        if part.empty:
            continue
        labels = part["label"].to_numpy(dtype=np.int64)
        scores = normalized_score(part["score_used"], part["threshold_used"])
        metrics = classification_metrics(labels, scores, 0.5)
        rows.append(
            {
                group_column: str(group),
                "n": len(part),
                "n_positive": int((labels == 1).sum()),
                "n_negative": int((labels == 0).sum()),
                "mcc": metrics["mcc"],
                "recall": metrics["recall"],
                "specificity": metrics["specificity"],
                "mean_score": float(part["score_used"].mean()),
                "mean_molecular_probability": float(part["molecular_probability"].mean()),
                "mean_regional_discount": float(part["regional_discount"].mean()),
            }
        )
    return pd.DataFrame(rows)


def build_diagnostics(args: argparse.Namespace, output_dir: Path) -> dict[str, Path]:
    diag_dir = output_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for dataset in args.datasets:
        frame = enrich_predictions(args, dataset)
        frame["af_abraom_bin"] = bin_af(frame.get("af_abraom", pd.Series(index=frame.index, dtype=float)))
        frame["discount_bin"] = pd.cut(
            frame["regional_discount"],
            bins=[-np.inf, 0.0, 0.1, 0.25, 0.5, 1.0, np.inf],
            labels=["0", "(0,0.1]", "(0.1,0.25]", "(0.25,0.5]", "(0.5,1.0]", ">1.0"],
        )
        frames.append(frame)
    all_predictions = pd.concat(frames, ignore_index=True)
    outputs: dict[str, Path] = {}
    for column in ["dataset", "specificity_bin", "af_abraom_bin", "discount_bin"]:
        diagnostic = diagnostic_group(all_predictions, column)
        path = diag_dir / f"m5_v2_calibrated_by_{column}.csv"
        diagnostic.to_csv(path, index=False)
        outputs[column] = path
    errors = all_predictions.loc[all_predictions["prediction_used"] != all_predictions["label"]].copy()
    top_genes = (
        errors.groupby("GeneSymbol", dropna=False)
        .agg(
            errors=("original_index", "count"),
            false_benign=("label", "sum"),
            mean_score=("score_used", "mean"),
            mean_discount=("regional_discount", "mean"),
        )
        .reset_index()
        .sort_values("errors", ascending=False)
        .head(50)
    )
    path = diag_dir / "m5_v2_calibrated_top_error_genes.csv"
    top_genes.to_csv(path, index=False)
    outputs["top_error_genes"] = path
    return outputs


def build_sensitivity_panel(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    panel_dir = output_dir / "sensitivity_panel"
    panel_dir.mkdir(parents=True, exist_ok=True)
    source = args.slice_dir / "abraom_pathogenic_present.parquet"
    panel = pd.read_parquet(source).copy()
    panel.to_parquet(panel_dir / "clinvar_plp_abraom_present.parquet", index=False)
    test_predictions = enrich_predictions(args, "abraom_pathogenic_present")
    by_gene = (
        test_predictions.groupby("GeneSymbol", dropna=False)
        .agg(
            n=("original_index", "count"),
            recall=("prediction_used", "mean"),
            mean_af_abraom=("af_abraom", "mean"),
            mean_score=("score_used", "mean"),
            mean_discount=("regional_discount", "mean"),
        )
        .reset_index()
        .sort_values(["recall", "n"], ascending=[True, False])
    )
    by_gene.to_csv(panel_dir / "clinvar_plp_abraom_present_test_sensitivity_by_gene.csv", index=False)
    manifest = {
        "clinvar_plp_abraom_present_rows": len(panel),
        "test_rows": len(test_predictions),
        "test_recall": float(test_predictions["prediction_used"].mean()),
        "panel_path": str(panel_dir / "clinvar_plp_abraom_present.parquet"),
        "by_gene_path": str(panel_dir / "clinvar_plp_abraom_present_test_sensitivity_by_gene.csv"),
        "known_brazilian_founder_variants_available": False,
        "manual_brazilian_plp_curation_available": False,
    }
    (panel_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def write_report(
    *,
    output_dir: Path,
    summary: pd.DataFrame,
    bootstrap: pd.DataFrame,
    error_outputs: dict[str, Path],
    diagnostic_outputs: dict[str, Path],
    sensitivity_manifest: dict[str, Any],
) -> Path:
    def get(model: str, dataset: str, metric: str) -> float:
        row = summary.loc[(summary["model"] == model) & (summary["dataset"] == dataset)]
        return float(row.iloc[0][metric]) if not row.empty else float("nan")

    lines = [
        "# M5_v2 Calibrated Final Analysis",
        "",
        "## Decision",
        "",
        "`M5_v2_calibrated` remains the lead scientific candidate after holdout-tuned calibration.",
        "",
        "## Key Metrics",
        "",
        "| Metric | M0 | M5_calibrated | M7_scrambled | M5_v2_calibrated |",
        "|---|---:|---:|---:|---:|",
        (
            f"| br_only MCC | {get('M0','br_only','mcc'):.3f} | "
            f"{get('M5_calibrated','br_only','mcc'):.3f} | "
            f"{get('M7_scrambled','br_only','mcc'):.3f} | "
            f"{get('M5_v2_calibrated','br_only','mcc'):.3f} |"
        ),
        (
            f"| ABRAOM-common benign specificity | {get('M0','abraom_common_benign','specificity'):.3f} | "
            f"{get('M5_calibrated','abraom_common_benign','specificity'):.3f} | "
            f"{get('M7_scrambled','abraom_common_benign','specificity'):.3f} | "
            f"{get('M5_v2_calibrated','abraom_common_benign','specificity'):.3f} |"
        ),
        (
            f"| ABRAOM-present P/LP recall | {get('M0','abraom_pathogenic_present','recall'):.3f} | "
            f"{get('M5_calibrated','abraom_pathogenic_present','recall'):.3f} | "
            f"{get('M7_scrambled','abraom_pathogenic_present','recall'):.3f} | "
            f"{get('M5_v2_calibrated','abraom_pathogenic_present','recall'):.3f} |"
        ),
        (
            f"| global nonBR MCC | {get('M0','global_nonbr_no_abraom','mcc'):.3f} | "
            f"{get('M5_calibrated','global_nonbr_no_abraom','mcc'):.3f} | "
            f"{get('M7_scrambled','global_nonbr_no_abraom','mcc'):.3f} | "
            f"{get('M5_v2_calibrated','global_nonbr_no_abraom','mcc'):.3f} |"
        ),
        "",
        "## Bootstrap Deltas",
        "",
        "| Comparison | Dataset | Metric | Delta | 95% CI |",
        "|---|---|---|---:|---:|",
    ]
    for row in bootstrap.itertuples(index=False):
        lines.append(
            f"| {row.compare_model} - {row.base_model} | {row.dataset} | {row.metric} | "
            f"{row.delta:.3f} | [{row.ci95_low:.3f}, {row.ci95_high:.3f}] |"
        )
    lines.extend(
        [
            "",
            "## Error Review",
            "",
            "- False pathogenic ABRAOM-common benign table: "
            f"`{error_outputs['false_pathogenic_abraom_common_benign']}`",
            f"- False benign ABRAOM-present P/LP table: `{error_outputs['false_benign_abraom_pathogenic_present']}`",
            "- P/LP variants rescued from raw M5_v2: "
            f"`{error_outputs['rescued_from_raw_false_benign_abraom_pathogenic_present']}`",
            "",
            "## Diagnostics",
            "",
        ]
    )
    for name, path in diagnostic_outputs.items():
        lines.append(f"- `{name}`: `{path}`")
    lines.extend(
        [
            "",
            "## Sensitivity Panel",
            "",
            f"- ClinVar P/LP ABRAOM-present panel rows: `{sensitivity_manifest['clinvar_plp_abraom_present_rows']}`",
            f"- Test sensitivity rows: `{sensitivity_manifest['test_rows']}`",
            f"- Test recall: `{sensitivity_manifest['test_recall']:.3f}`",
            "- Curated Brazilian founder/manual P/LP panels are still not available locally.",
            "",
            "## Remaining Work",
            "",
            "1. Add a curated Brazilian founder/P/LP panel and rerun this analysis.",
            "2. Review the false-benign table before treating the candidate as stable.",
            "3. Decide operating-point policy explicitly: Brazilian regional triage versus global molecular "
            "sensitivity mode.",
        ]
    )
    report_path = output_dir / "M5_V2_CALIBRATED_FINAL_ANALYSIS.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sources = build_sources(args)
    summary = pd.read_csv(args.baseline_csv)
    bootstrap = build_bootstrap(args, sources, args.output_dir)
    error_outputs = build_error_tables(args, args.output_dir)
    diagnostic_outputs = build_diagnostics(args, args.output_dir)
    sensitivity_manifest = build_sensitivity_panel(args, args.output_dir)
    report_path = write_report(
        output_dir=args.output_dir,
        summary=summary,
        bootstrap=bootstrap,
        error_outputs=error_outputs,
        diagnostic_outputs=diagnostic_outputs,
        sensitivity_manifest=sensitivity_manifest,
    )
    print(f"report={report_path}")
    print(f"bootstrap={args.output_dir / 'cluster_bootstrap_intervals.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
