#!/usr/bin/env python3
"""Deep error analysis for ABRAOM adapter-fusion regionalization."""

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

EPS = 1e-6


@dataclass(frozen=True)
class ModelSpec:
    name: str
    root: Path
    kind: str


DEFAULT_DATASETS = [
    "br_only",
    "br_any",
    "regional_benchmark_any",
    "abraom_common_benign",
    "abraom_pathogenic_present",
    "abraom_pathogenic_common",
    "global_nonbr_no_abraom",
]

FOCUS_DATASETS = [
    "br_only",
    "abraom_common_benign",
    "abraom_pathogenic_present",
    "global_nonbr_no_abraom",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slice-dir", type=Path, default=Path("data/datasets/clinvar/regional_abraom/slices"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/adapter_fusion_error_analysis_deep"))
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260629)
    return parser.parse_args(argv)


def logit(values: pd.Series | np.ndarray | float) -> np.ndarray:
    probs = np.clip(np.asarray(values, dtype=np.float64), EPS, 1.0 - EPS)
    return np.log(probs / (1.0 - probs))


def sigmoid(values: pd.Series | np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(values, dtype=np.float64), -60.0, 60.0)))


def read_threshold(root: Path) -> float:
    summary_path = root / "regional_eval_summary.json"
    if not summary_path.is_file():
        return 0.5
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    return float(payload.get("threshold", 0.5))


def model_specs() -> list[ModelSpec]:
    base = Path("artifacts/clinvar_regional_eval")
    return [
        ModelSpec("M0", base / "m0_nonbr_beatv10_v1_sagemaker", "raw"),
        ModelSpec("M2_gnomad_only", base / "m2_dynamic_gnomadonly_nonbr_beatv10_v1_sagemaker", "raw"),
        ModelSpec("M4_dynamic_gated", base / "m4_dynamic_gated_nonbr_beatv10_v1_sagemaker", "raw"),
        ModelSpec("M5_dynamic_gated", base / "m5_dynamic_gated_bounded_nonbr_beatv10_v1_sagemaker", "raw"),
        ModelSpec("M7_dynamic_scrambled", base / "m7_dynamic_scrambled_nonbr_beatv10_v1_sagemaker", "raw"),
        ModelSpec("M5_v2_calibrated", base / "m5_v2_calibrated_holdout_tuned", "calibrated_v2"),
    ]


def load_metadata(slice_dir: Path, dataset: str) -> pd.DataFrame:
    path = slice_dir / f"{dataset}.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path).reset_index().rename(columns={"index": "original_index"})
    if "split_within_gene" in frame.columns:
        frame = frame.loc[frame["split_within_gene"].astype(str).str.lower().eq("test")].copy()
    columns = [
        "original_index",
        "variant_key",
        "source_variant_id",
        "GeneSymbol",
        "Chromosome",
        "Start",
        "ReferenceAlleleVCF",
        "AlternateAlleleVCF",
        "variant_type",
        "is_snv",
        "clinvar_regional_cohort",
        "has_brazilian_submitter",
        "has_non_brazilian_submitter",
        "abraom_present",
        "af_abraom",
        "af_gnomad",
        "specificity",
        "specificity_bin",
    ]
    return frame[[column for column in columns if column in frame.columns]].copy()


def load_predictions(spec: ModelSpec, dataset: str) -> pd.DataFrame:
    path = spec.root / f"{dataset}.test.predictions.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path).copy()
    if spec.kind == "calibrated_v2":
        frame["score"] = frame["score_used"].astype(float)
        frame["threshold"] = frame["threshold_used"].astype(float)
        frame["prediction"] = frame["prediction_used"].astype(int)
    else:
        threshold = read_threshold(spec.root)
        frame["score"] = frame["probability"].astype(float)
        frame["threshold"] = threshold
        frame["prediction"] = (frame["score"] >= threshold).astype(int)
    keep = [
        "dataset",
        "split",
        "original_index",
        "label",
        "probability",
        "swap_probability",
        "score",
        "threshold",
        "prediction",
        "molecular_probability",
        "regional_discount",
        "regional_score_calibrated",
        "global_score_calibrated",
        "gate_alpha_abraom",
        "gate_alpha_gnomad",
        "gate_alpha_scrambled",
        "gate_entropy",
    ]
    return frame[[column for column in keep if column in frame.columns]].copy()


def merge_dataset(specs: list[ModelSpec], slice_dir: Path, dataset: str) -> pd.DataFrame:
    metadata = load_metadata(slice_dir, dataset)
    wide = metadata.copy()
    label: pd.Series | None = None
    for spec in specs:
        preds = load_predictions(spec, dataset)
        if label is None:
            label = preds.set_index("original_index")["label"]
        rename = {
            column: f"{spec.name}__{column}"
            for column in preds.columns
            if column not in {"dataset", "split", "original_index", "label"}
        }
        model_frame = preds.rename(columns=rename)
        wide = wide.merge(model_frame.drop(columns=["dataset", "split", "label"]), on="original_index", how="inner")
    if label is None:
        raise RuntimeError(f"No predictions loaded for {dataset}")
    wide = wide.merge(label.rename("label").reset_index(), on="original_index", how="left")
    wide["GeneSymbol"] = wide.get("GeneSymbol", pd.Series([""] * len(wide))).fillna("").astype(str)
    wide.loc[wide["GeneSymbol"].eq(""), "GeneSymbol"] = wide["original_index"].astype(str)
    return wide


def confusion(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    labels = labels.astype(int)
    predictions = predictions.astype(int)
    tp = int(((labels == 1) & (predictions == 1)).sum())
    tn = int(((labels == 0) & (predictions == 0)).sum())
    fp = int(((labels == 0) & (predictions == 1)).sum())
    fn = int(((labels == 1) & (predictions == 0)).sum())
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn - fp * fn) / denom) if denom else 0.0
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "recall": recall,
        "specificity": specificity,
        "precision": precision,
        "mcc": mcc,
    }


def metric_from_predictions(labels: np.ndarray, predictions: np.ndarray, metric: str) -> float:
    return float(confusion(labels, predictions)[metric])


def build_error_counts(wide_by_dataset: dict[str, pd.DataFrame], specs: list[ModelSpec]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset, frame in wide_by_dataset.items():
        labels = frame["label"].to_numpy(dtype=int)
        for spec in specs:
            pred_col = f"{spec.name}__prediction"
            if pred_col not in frame.columns:
                continue
            row = {
                "model": spec.name,
                "dataset": dataset,
                "n": int(len(frame)),
                "n_positive": int((labels == 1).sum()),
                "n_negative": int((labels == 0).sum()),
            }
            row.update(confusion(labels, frame[pred_col].to_numpy(dtype=int)))
            rows.append(row)
    return pd.DataFrame(rows)


def _model_score_columns(specs: list[ModelSpec]) -> list[str]:
    cols: list[str] = []
    for spec in specs:
        cols.extend(
            [
                f"{spec.name}__score",
                f"{spec.name}__prediction",
                f"{spec.name}__probability",
                f"{spec.name}__threshold",
                f"{spec.name}__molecular_probability",
                f"{spec.name}__regional_discount",
                f"{spec.name}__gate_alpha_abraom",
                f"{spec.name}__gate_alpha_gnomad",
                f"{spec.name}__gate_alpha_scrambled",
                f"{spec.name}__gate_entropy",
            ]
        )
    return cols


def false_benign_category(row: pd.Series, model: str) -> str:
    discount = float(row.get(f"{model}__regional_discount", 0.0) or 0.0)
    molecular = float(row.get(f"{model}__molecular_probability", row.get(f"{model}__score", 0.0)) or 0.0)
    af_abraom = float(row.get("af_abraom", 0.0) or 0.0)
    score = float(row.get(f"{model}__score", 0.0) or 0.0)
    threshold = float(row.get(f"{model}__threshold", 0.5) or 0.5)
    if discount >= 0.5 and molecular >= 0.35:
        return "regional_af_over_suppression"
    if molecular < 0.35:
        return "weak_molecular_signal"
    if af_abraom >= 0.01:
        return "common_plp_recessive_or_founder_review"
    if score >= threshold - 0.05:
        return "threshold_borderline"
    return "unresolved_false_benign"


def false_pathogenic_category(row: pd.Series, model: str) -> str:
    molecular = float(row.get(f"{model}__molecular_probability", row.get(f"{model}__score", 0.0)) or 0.0)
    score = float(row.get(f"{model}__score", 0.0) or 0.0)
    threshold = float(row.get(f"{model}__threshold", 0.5) or 0.5)
    af_abraom = float(row.get("af_abraom", 0.0) or 0.0)
    specificity = float(row.get("specificity", 0.0) or 0.0)
    if molecular >= 0.8:
        return "molecular_score_overdominant"
    if af_abraom >= 0.01 and specificity <= 0.005:
        return "frequency_discount_insufficient_common"
    if score <= threshold + 0.05:
        return "threshold_borderline"
    return "unresolved_false_pathogenic"


def write_error_tables(
    wide_by_dataset: dict[str, pd.DataFrame],
    specs: list[ModelSpec],
    output_dir: Path,
) -> dict[str, str]:
    error_dir = output_dir / "variant_error_tables"
    error_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    base_columns = [
        "dataset",
        "original_index",
        "variant_key",
        "GeneSymbol",
        "Chromosome",
        "Start",
        "ReferenceAlleleVCF",
        "AlternateAlleleVCF",
        "variant_type",
        "label",
        "af_abraom",
        "af_gnomad",
        "specificity",
        "specificity_bin",
        "failure_category",
    ]
    model_columns = [col for col in _model_score_columns(specs)]
    for spec in specs:
        model = spec.name
        pred_col = f"{model}__prediction"
        if pred_col not in wide_by_dataset["abraom_common_benign"].columns:
            continue
        benign = wide_by_dataset["abraom_common_benign"].copy()
        benign["dataset"] = "abraom_common_benign"
        fps = benign.loc[(benign["label"] == 0) & (benign[pred_col] == 1)].copy()
        fps["failure_category"] = fps.apply(false_pathogenic_category, axis=1, model=model)
        fps = fps.sort_values([f"{model}__score", "af_abraom"], ascending=[False, False])
        fp_path = error_dir / f"{model}.false_pathogenic_abraom_common_benign.csv"
        fps[[col for col in [*base_columns, *model_columns] if col in fps.columns]].to_csv(fp_path, index=False)
        outputs[f"{model}_false_pathogenic"] = str(fp_path)

        pathogenic = wide_by_dataset["abraom_pathogenic_present"].copy()
        pathogenic["dataset"] = "abraom_pathogenic_present"
        fns = pathogenic.loc[(pathogenic["label"] == 1) & (pathogenic[pred_col] == 0)].copy()
        fns["failure_category"] = fns.apply(false_benign_category, axis=1, model=model)
        fns = fns.sort_values([f"{model}__score", "af_abraom"], ascending=[True, False])
        fn_path = error_dir / f"{model}.false_benign_abraom_pathogenic_present.csv"
        fns[[col for col in [*base_columns, *model_columns] if col in fns.columns]].to_csv(fn_path, index=False)
        outputs[f"{model}_false_benign"] = str(fn_path)

    benign = wide_by_dataset["abraom_common_benign"].copy()
    benign["dataset"] = "abraom_common_benign"
    rescued = benign.loc[
        (benign["label"] == 0)
        & (benign["M5_v2_calibrated__prediction"] == 1)
        & (benign["M5_dynamic_gated__prediction"] == 0)
    ].copy()
    rescued["dynamic_minus_v2_score"] = rescued["M5_dynamic_gated__score"] - rescued["M5_v2_calibrated__score"]
    rescued_path = error_dir / "M5_dynamic.rescued_false_pathogenic_vs_M5_v2.csv"
    rescued[[col for col in [*base_columns, "dynamic_minus_v2_score", *model_columns] if col in rescued.columns]].to_csv(
        rescued_path,
        index=False,
    )
    outputs["m5_dynamic_rescued_false_pathogenic_vs_m5_v2"] = str(rescued_path)

    pathogenic = wide_by_dataset["abraom_pathogenic_present"].copy()
    pathogenic["dataset"] = "abraom_pathogenic_present"
    regressions = pathogenic.loc[
        (pathogenic["label"] == 1)
        & (pathogenic["M5_v2_calibrated__prediction"] == 1)
        & (pathogenic["M5_dynamic_gated__prediction"] == 0)
    ].copy()
    regressions["dynamic_minus_v2_score"] = (
        regressions["M5_dynamic_gated__score"] - regressions["M5_v2_calibrated__score"]
    )
    regressions["failure_category"] = regressions.apply(false_benign_category, axis=1, model="M5_dynamic_gated")
    regression_path = error_dir / "M5_dynamic.dangerous_false_benign_regressions_vs_M5_v2.csv"
    regressions[
        [col for col in [*base_columns, "dynamic_minus_v2_score", *model_columns] if col in regressions.columns]
    ].to_csv(regression_path, index=False)
    outputs["m5_dynamic_dangerous_false_benign_regressions_vs_m5_v2"] = str(regression_path)
    return outputs


def build_transition_tables(wide_by_dataset: dict[str, pd.DataFrame], output_dir: Path) -> pd.DataFrame:
    pairs = [
        ("M0", "M5_v2_calibrated"),
        ("M0", "M5_dynamic_gated"),
        ("M5_v2_calibrated", "M5_dynamic_gated"),
        ("M2_gnomad_only", "M4_dynamic_gated"),
        ("M7_dynamic_scrambled", "M4_dynamic_gated"),
    ]
    rows: list[dict[str, Any]] = []
    for dataset, frame in wide_by_dataset.items():
        labels = frame["label"].to_numpy(dtype=int)
        for base, compare in pairs:
            b = frame[f"{base}__prediction"].to_numpy(dtype=int)
            c = frame[f"{compare}__prediction"].to_numpy(dtype=int)
            base_correct = b == labels
            compare_correct = c == labels
            rows.append(
                {
                    "dataset": dataset,
                    "base_model": base,
                    "compare_model": compare,
                    "n": int(len(frame)),
                    "both_correct": int((base_correct & compare_correct).sum()),
                    "both_wrong": int((~base_correct & ~compare_correct).sum()),
                    "base_only_correct": int((base_correct & ~compare_correct).sum()),
                    "compare_only_correct": int((~base_correct & compare_correct).sum()),
                    "base_pred_0_compare_pred_0": int(((b == 0) & (c == 0)).sum()),
                    "base_pred_0_compare_pred_1": int(((b == 0) & (c == 1)).sum()),
                    "base_pred_1_compare_pred_0": int(((b == 1) & (c == 0)).sum()),
                    "base_pred_1_compare_pred_1": int(((b == 1) & (c == 1)).sum()),
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "paired_prediction_transitions.csv", index=False)
    return table


def summarize_errors_by_group(wide_by_dataset: dict[str, pd.DataFrame], specs: list[ModelSpec], output_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    for dataset in ["abraom_common_benign", "abraom_pathogenic_present", "br_only"]:
        frame = wide_by_dataset[dataset].copy()
        for spec in specs:
            pred_col = f"{spec.name}__prediction"
            frame["_error"] = frame[pred_col].astype(int) != frame["label"].astype(int)
            for group_col in ["GeneSymbol", "specificity_bin", "variant_type", "clinvar_regional_cohort"]:
                if group_col not in frame.columns:
                    continue
                grouped = frame.groupby(group_col, dropna=False)
                for value, group in grouped:
                    if len(group) < 2 and group["_error"].sum() == 0:
                        continue
                    rows.append(
                        {
                            "dataset": dataset,
                            "model": spec.name,
                            "group_column": group_col,
                            "group_value": value,
                            "n": int(len(group)),
                            "errors": int(group["_error"].sum()),
                            "error_rate": float(group["_error"].mean()),
                            "mean_af_abraom": float(pd.to_numeric(group.get("af_abraom"), errors="coerce").mean()),
                            "mean_score": float(pd.to_numeric(group.get(f"{spec.name}__score"), errors="coerce").mean()),
                        }
                    )
    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "error_rates_by_group.csv", index=False)
    out.sort_values(["errors", "error_rate"], ascending=[False, False]).head(300).to_csv(
        output_dir / "top_error_groups.csv",
        index=False,
    )


def bootstrap_rows(
    wide_by_dataset: dict[str, pd.DataFrame],
    *,
    iterations: int,
    seed: int,
) -> pd.DataFrame:
    comparisons = [
        ("M0", "M5_v2_calibrated", "br_only", "mcc"),
        ("M0", "M5_dynamic_gated", "br_only", "mcc"),
        ("M5_v2_calibrated", "M5_dynamic_gated", "br_only", "mcc"),
        ("M0", "M5_v2_calibrated", "abraom_common_benign", "specificity"),
        ("M0", "M5_dynamic_gated", "abraom_common_benign", "specificity"),
        ("M5_v2_calibrated", "M5_dynamic_gated", "abraom_common_benign", "specificity"),
        ("M0", "M5_v2_calibrated", "abraom_pathogenic_present", "recall"),
        ("M0", "M5_dynamic_gated", "abraom_pathogenic_present", "recall"),
        ("M5_v2_calibrated", "M5_dynamic_gated", "abraom_pathogenic_present", "recall"),
        ("M0", "M5_dynamic_gated", "global_nonbr_no_abraom", "mcc"),
        ("M0", "M5_v2_calibrated", "global_nonbr_no_abraom", "mcc"),
        ("M7_dynamic_scrambled", "M4_dynamic_gated", "br_only", "mcc"),
    ]
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for base, compare, dataset, metric in comparisons:
        frame = wide_by_dataset[dataset]
        labels = frame["label"].to_numpy(dtype=int)
        base_pred = frame[f"{base}__prediction"].to_numpy(dtype=int)
        compare_pred = frame[f"{compare}__prediction"].to_numpy(dtype=int)
        estimate = metric_from_predictions(labels, compare_pred, metric) - metric_from_predictions(labels, base_pred, metric)
        clusters = frame["GeneSymbol"].fillna("").astype(str)
        clusters = clusters.where(clusters != "", frame["original_index"].astype(str))
        unique = clusters.unique()
        by_cluster = {cluster: np.flatnonzero(clusters.to_numpy() == cluster) for cluster in unique}
        deltas = np.empty(iterations, dtype=float)
        for i in range(iterations):
            sampled = rng.choice(unique, size=len(unique), replace=True)
            idx = np.concatenate([by_cluster[c] for c in sampled])
            deltas[i] = metric_from_predictions(labels[idx], compare_pred[idx], metric) - metric_from_predictions(
                labels[idx],
                base_pred[idx],
                metric,
            )
        low, high = np.quantile(deltas, [0.025, 0.975]) if iterations else (float("nan"), float("nan"))
        rows.append(
            {
                "base_model": base,
                "compare_model": compare,
                "dataset": dataset,
                "metric": metric,
                "delta": float(estimate),
                "ci95_low": float(low),
                "ci95_high": float(high),
                "iterations": iterations,
                "cluster": "GeneSymbol",
            }
        )
    return pd.DataFrame(rows)


def evaluate_prediction_rule(
    frames: dict[str, pd.DataFrame],
    score_by_dataset: dict[str, np.ndarray],
    threshold: float,
) -> dict[str, float]:
    out: dict[str, float] = {"threshold": float(threshold)}
    for dataset in FOCUS_DATASETS:
        frame = frames[dataset]
        labels = frame["label"].to_numpy(dtype=int)
        pred = (score_by_dataset[dataset] >= threshold).astype(int)
        metrics = confusion(labels, pred)
        prefix = dataset
        out[f"{prefix}_mcc"] = metrics["mcc"]
        out[f"{prefix}_recall"] = metrics["recall"]
        out[f"{prefix}_specificity"] = metrics["specificity"]
        out[f"{prefix}_fp"] = metrics["fp"]
        out[f"{prefix}_fn"] = metrics["fn"]
    return out


def build_m5_dynamic_calibration_grid(
    wide_by_dataset: dict[str, pd.DataFrame],
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frames = {dataset: wide_by_dataset[dataset] for dataset in FOCUS_DATASETS}
    rows: list[dict[str, Any]] = []
    for scale in np.round(np.linspace(0.0, 1.25, 26), 3):
        score_by_dataset: dict[str, np.ndarray] = {}
        for dataset, frame in frames.items():
            molecular = frame["M5_dynamic_gated__molecular_probability"].to_numpy(dtype=float)
            discount = frame["M5_dynamic_gated__regional_discount"].to_numpy(dtype=float)
            score_by_dataset[dataset] = sigmoid(logit(molecular) - scale * discount)
        for threshold in np.round(np.linspace(0.20, 0.80, 121), 3):
            row = evaluate_prediction_rule(frames, score_by_dataset, float(threshold))
            row["discount_scale"] = float(scale)
            rows.append(row)
    grid = pd.DataFrame(rows)
    m0_plp_floor = 0.4171779141104294 - 0.05
    constrained = grid.loc[
        (grid["abraom_common_benign_specificity"] >= 0.95)
        & (grid["abraom_pathogenic_present_recall"] >= m0_plp_floor)
        & (grid["global_nonbr_no_abraom_mcc"] >= 0.46)
    ].copy()
    constrained = constrained.sort_values(
        [
            "br_only_mcc",
            "abraom_common_benign_specificity",
            "abraom_pathogenic_present_recall",
            "global_nonbr_no_abraom_mcc",
        ],
        ascending=[False, False, False, False],
    )
    near = grid.copy()
    plp_floor = m0_plp_floor
    near["constraint_gap"] = (
        (0.95 - near["abraom_common_benign_specificity"]).clip(lower=0) * 2
        + (plp_floor - near["abraom_pathogenic_present_recall"]).clip(lower=0) * 2
        + (0.46 - near["global_nonbr_no_abraom_mcc"]).clip(lower=0)
    )
    near = near.sort_values(["constraint_gap", "br_only_mcc"], ascending=[True, False]).head(100)
    grid.to_csv(output_dir / "m5_dynamic_discount_scale_grid.csv", index=False)
    constrained.head(100).to_csv(output_dir / "m5_dynamic_discount_scale_candidates.csv", index=False)
    near.to_csv(output_dir / "m5_dynamic_discount_scale_near_misses.csv", index=False)
    return grid, constrained.head(100), near


def write_failure_category_summary(output_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted((output_dir / "variant_error_tables").glob("*.csv")):
        frame = pd.read_csv(path)
        if "failure_category" not in frame.columns:
            continue
        for category, count in frame["failure_category"].value_counts(dropna=False).items():
            rows.append(
                {
                    "file": path.name,
                    "failure_category": category,
                    "n": int(count),
                    "fraction": float(count / len(frame)) if len(frame) else 0.0,
                }
            )
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "failure_category_summary.csv", index=False)
    return summary


def write_report(
    output_dir: Path,
    error_counts: pd.DataFrame,
    transitions: pd.DataFrame,
    bootstrap: pd.DataFrame,
    candidates: pd.DataFrame,
    near_misses: pd.DataFrame,
    failure_summary: pd.DataFrame,
    error_outputs: dict[str, str],
) -> None:
    def get(model: str, dataset: str, metric: str) -> float:
        row = error_counts.loc[(error_counts["model"] == model) & (error_counts["dataset"] == dataset)]
        if row.empty:
            return float("nan")
        return float(row.iloc[0][metric])

    def count(model: str, dataset: str, field: str) -> int:
        row = error_counts.loc[(error_counts["model"] == model) & (error_counts["dataset"] == dataset)]
        if row.empty:
            return 0
        return int(row.iloc[0][field])

    reg_path = output_dir / "variant_error_tables" / "M5_dynamic.dangerous_false_benign_regressions_vs_M5_v2.csv"
    rescue_path = output_dir / "variant_error_tables" / "M5_dynamic.rescued_false_pathogenic_vs_M5_v2.csv"
    regressions = pd.read_csv(reg_path) if reg_path.exists() else pd.DataFrame()
    rescues = pd.read_csv(rescue_path) if rescue_path.exists() else pd.DataFrame()

    top_regression_genes = (
        regressions.groupby("GeneSymbol").size().sort_values(ascending=False).head(10).to_dict()
        if not regressions.empty
        else {}
    )
    top_rescue_genes = (
        rescues.groupby("GeneSymbol").size().sort_values(ascending=False).head(10).to_dict() if not rescues.empty else {}
    )

    lines = [
        "# Deep Error Analysis: ABRAOM Adapter Fusion",
        "",
        f"Generated at UTC: `{datetime.now(UTC).isoformat()}`",
        "",
        "## Bottom Line",
        "",
        "`M5_dynamic_gated` is not ready as the lead model. It is excellent at suppressing ABRAOM-common benign false positives, but the same regional discount mechanism creates a large false-benign safety failure on ABRAOM-present P/LP variants.",
        "",
        "## Core Counts",
        "",
        "| Model | ABRAOM-common benign FP | ABRAOM-common specificity | ABRAOM P/LP false benign | ABRAOM P/LP recall | br_only MCC | global MCC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in ["M0", "M4_dynamic_gated", "M5_dynamic_gated", "M7_dynamic_scrambled", "M5_v2_calibrated"]:
        lines.append(
            f"| `{model}` | {count(model, 'abraom_common_benign', 'fp')} | "
            f"{get(model, 'abraom_common_benign', 'specificity'):.3f} | "
            f"{count(model, 'abraom_pathogenic_present', 'fn')} | "
            f"{get(model, 'abraom_pathogenic_present', 'recall'):.3f} | "
            f"{get(model, 'br_only', 'mcc'):.3f} | "
            f"{get(model, 'global_nonbr_no_abraom', 'mcc'):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Paired M5 Dynamic vs M5 v2",
            "",
            f"- `M5_dynamic_gated` rescues `{len(rescues)}` ABRAOM-common benign false positives that `M5_v2_calibrated` still calls pathogenic.",
            f"- `M5_dynamic_gated` creates `{len(regressions)}` dangerous false-benign regressions among ABRAOM-present P/LP variants that `M5_v2_calibrated` keeps positive.",
            f"- Top regression genes: `{top_regression_genes}`.",
            f"- Top rescued benign genes: `{top_rescue_genes}`.",
            "",
            "## Statistical Checks",
            "",
            "Cluster bootstrap by `GeneSymbol` is written to `paired_bootstrap_by_gene.csv`. Treat wide CIs in small P/LP slices as uncertainty, not model victory.",
            "",
            "## Post-hoc Dynamic Calibration Probe",
            "",
        ]
    )
    if candidates.empty:
        lines.append(
            "No simple discount-scale/threshold configuration satisfied all hard constraints: ABRAOM-common specificity >= 0.95, ABRAOM P/LP recall near M0, and global MCC floor >= 0.46."
        )
    else:
        best = candidates.iloc[0]
        lines.extend(
            [
                "A simple discount-scale/threshold rule can partially rescue the dynamic model in post-hoc analysis.",
                "",
                "| discount_scale | threshold | br_only MCC | ABRAOM-common specificity | ABRAOM P/LP recall | global MCC |",
                "|---:|---:|---:|---:|---:|---:|",
                f"| {best['discount_scale']:.3f} | {best['threshold']:.3f} | {best['br_only_mcc']:.3f} | {best['abraom_common_benign_specificity']:.3f} | {best['abraom_pathogenic_present_recall']:.3f} | {best['global_nonbr_no_abraom_mcc']:.3f} |",
                "",
                "This is not a final model; it is evidence that the next training/calibration cycle should constrain the regional discount rather than discard dynamic fusion.",
            ]
        )
    if not near_misses.empty:
        near = near_misses.iloc[0]
        lines.extend(
            [
                "",
                "Best near miss when all constraints cannot be met:",
                "",
                "| discount_scale | threshold | constraint_gap | br_only MCC | ABRAOM-common specificity | ABRAOM P/LP recall | global MCC | global specificity |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|",
                f"| {near['discount_scale']:.3f} | {near['threshold']:.3f} | {near['constraint_gap']:.3f} | {near['br_only_mcc']:.3f} | {near['abraom_common_benign_specificity']:.3f} | {near['abraom_pathogenic_present_recall']:.3f} | {near['global_nonbr_no_abraom_mcc']:.3f} | {near['global_nonbr_no_abraom_specificity']:.3f} |",
                "",
                "The near misses recover P/LP recall and keep ABRAOM-common specificity, but global nonBR performance collapses. This means threshold-only rescue is not enough.",
            ]
        )
    if not failure_summary.empty:
        focused = failure_summary.loc[
            failure_summary["file"].isin(
                [
                    "M5_dynamic_gated.false_benign_abraom_pathogenic_present.csv",
                    "M5_dynamic.dangerous_false_benign_regressions_vs_M5_v2.csv",
                    "M5_v2_calibrated.false_benign_abraom_pathogenic_present.csv",
                    "M5_dynamic_gated.false_pathogenic_abraom_common_benign.csv",
                    "M5_v2_calibrated.false_pathogenic_abraom_common_benign.csv",
                ]
            )
        ]
        lines.extend(["", "## Failure Categories", "", "| File | Category | n | fraction |", "|---|---|---:|---:|"])
        for row in focused.itertuples(index=False):
            lines.append(f"| `{row.file}` | `{row.failure_category}` | {row.n} | {row.fraction:.3f} |")
    lines.extend(
        [
            "",
            "## Recommended Action",
            "",
            "1. Keep `M5_v2_calibrated` as the current safest candidate.",
            "2. Use the dangerous regression table as the P/LP sentinel target for the next dynamic calibration.",
            "3. Train or calibrate `M5_dynamic_v2_safety` with an explicit molecular guard and a stricter cap on regional discount.",
            "4. Do not optimize only ABRAOM-common specificity; require P/LP recall and global non-inferiority constraints at selection time.",
            "",
            "## Key Artifacts",
            "",
        ]
    )
    for name, path in sorted(error_outputs.items()):
        lines.append(f"- `{name}`: `{path}`")
    lines.extend(
        [
            "- `error_counts_by_model_dataset.csv`",
            "- `paired_prediction_transitions.csv`",
            "- `paired_bootstrap_by_gene.csv`",
            "- `error_rates_by_group.csv`",
            "- `failure_category_summary.csv`",
            "- `m5_dynamic_discount_scale_grid.csv`",
            "- `m5_dynamic_discount_scale_candidates.csv`",
            "- `m5_dynamic_discount_scale_near_misses.csv`",
        ]
    )
    (output_dir / "DEEP_ERROR_ANALYSIS_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "m5_dynamic_rescued_false_pathogenic_vs_m5_v2": int(len(rescues)),
        "m5_dynamic_dangerous_false_benign_regressions_vs_m5_v2": int(len(regressions)),
        "top_regression_genes": top_regression_genes,
        "top_rescue_genes": top_rescue_genes,
        "best_posthoc_candidate": candidates.head(1).to_dict(orient="records"),
        "best_posthoc_near_miss": near_misses.head(1).to_dict(orient="records"),
        "error_outputs": error_outputs,
    }
    (output_dir / "deep_error_analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    specs = model_specs()
    wide_by_dataset = {dataset: merge_dataset(specs, args.slice_dir, dataset) for dataset in DEFAULT_DATASETS}
    error_counts = build_error_counts(wide_by_dataset, specs)
    error_counts.to_csv(args.output_dir / "error_counts_by_model_dataset.csv", index=False)
    error_counts.to_parquet(args.output_dir / "error_counts_by_model_dataset.parquet", index=False)
    error_outputs = write_error_tables(wide_by_dataset, specs, args.output_dir)
    transitions = build_transition_tables(wide_by_dataset, args.output_dir)
    transitions.to_parquet(args.output_dir / "paired_prediction_transitions.parquet", index=False)
    summarize_errors_by_group(wide_by_dataset, specs, args.output_dir)
    bootstrap = bootstrap_rows(
        wide_by_dataset,
        iterations=args.bootstrap_iterations,
        seed=args.seed,
    )
    bootstrap.to_csv(args.output_dir / "paired_bootstrap_by_gene.csv", index=False)
    bootstrap.to_parquet(args.output_dir / "paired_bootstrap_by_gene.parquet", index=False)
    _, candidates, near_misses = build_m5_dynamic_calibration_grid(wide_by_dataset, args.output_dir)
    failure_summary = write_failure_category_summary(args.output_dir)
    write_report(args.output_dir, error_counts, transitions, bootstrap, candidates, near_misses, failure_summary, error_outputs)
    print(json.dumps({"output_dir": str(args.output_dir), "error_outputs": error_outputs}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
