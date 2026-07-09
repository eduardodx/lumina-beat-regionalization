#!/usr/bin/env python3
"""Holdout-tuned calibration for the bounded M5_v2 regional head."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.clinvar.metrics import binary_auprc, binary_roc_auc, classification_metrics  # noqa: E402

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
REGIONAL_DATASETS = {
    "br_only",
    "br_any",
    "regional_benchmark_any",
    "abraom_common_benign",
    "abraom_pathogenic_present",
    "abraom_pathogenic_common",
}
GLOBAL_DATASETS = {"global_nonbr_no_abraom", "nonbr_only"}


@dataclass(frozen=True)
class RegionalCalibrationConfig:
    discount_scale: float
    max_discount: float
    regional_threshold: float
    global_threshold: float


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout-dir", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--baseline-csv", type=Path, required=True)
    parser.add_argument("--tuning-baseline-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    parser.add_argument("--model-name", default="M5_v2_calibrated")
    return parser.parse_args(argv)


def _logit(values: pd.Series | np.ndarray) -> np.ndarray:
    probs = np.clip(np.asarray(values, dtype=np.float64), EPS, 1.0 - EPS)
    return np.log(probs / (1.0 - probs))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -60.0, 60.0)))


def _read_predictions(root: Path, dataset: str, split: str) -> pd.DataFrame:
    path = root / f"{dataset}.{split}.predictions.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"Missing predictions: {path}")
    predictions = pd.read_parquet(path)
    required = {"dataset", "split", "original_index", "label", "probability", "molecular_probability"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
    if "regional_discount" not in predictions.columns:
        predictions["regional_discount"] = np.maximum(
            0.0,
            _logit(predictions["molecular_probability"]) - _logit(predictions["probability"]),
        )
    return predictions


def _regional_score(predictions: pd.DataFrame, discount_scale: float, max_discount: float) -> np.ndarray:
    discount = np.minimum(
        predictions["regional_discount"].to_numpy(dtype=np.float64) * discount_scale,
        max_discount,
    )
    return _sigmoid(_logit(predictions["molecular_probability"]) - discount)


def _metric_row(
    *,
    model: str,
    dataset: str,
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    labels = labels.astype(np.int64)
    scores = scores.astype(np.float64)
    metrics = classification_metrics(labels, scores, threshold)
    if len(np.unique(labels)) > 1:
        auroc = binary_roc_auc(labels, scores)
        auprc = binary_auprc(labels, scores)
    else:
        auroc = float("nan")
        auprc = binary_auprc(labels, scores)
    return {
        "model": model,
        "dataset": dataset,
        "n": len(labels),
        "auroc": auroc,
        "auprc": auprc,
        "mcc": metrics["mcc"],
        "recall": metrics["recall"],
        "specificity": metrics["specificity"],
        "threshold": threshold,
    }


def _metrics_for_config(
    predictions_by_dataset: dict[str, pd.DataFrame],
    config: RegionalCalibrationConfig,
    *,
    model_name: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset, predictions in predictions_by_dataset.items():
        labels = predictions["label"].to_numpy(dtype=np.int64)
        if dataset in GLOBAL_DATASETS:
            scores = predictions["molecular_probability"].to_numpy(dtype=np.float64)
            threshold = config.global_threshold
        else:
            scores = _regional_score(predictions, config.discount_scale, config.max_discount)
            threshold = config.regional_threshold
        rows.append(
            _metric_row(
                model=model_name,
                dataset=dataset,
                labels=labels,
                scores=scores,
                threshold=threshold,
            )
        )
    return rows


def _value(rows: list[dict[str, Any]], dataset: str, metric: str) -> float:
    for row in rows:
        if row["dataset"] == dataset:
            return float(row[metric])
    return float("nan")


def _baseline_value(baseline: pd.DataFrame, model: str, dataset: str, metric: str) -> float:
    match = baseline.loc[(baseline["model"] == model) & (baseline["dataset"] == dataset)]
    if match.empty:
        return float("nan")
    return float(match.iloc[0][metric])


def _candidate_regional_configs(global_threshold: float) -> list[RegionalCalibrationConfig]:
    configs: list[RegionalCalibrationConfig] = []
    for discount_scale in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]:
        for max_discount in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
            for regional_threshold in np.round(np.linspace(0.20, 0.80, 121), 3):
                configs.append(
                    RegionalCalibrationConfig(
                        discount_scale=float(discount_scale),
                        max_discount=float(max_discount),
                        regional_threshold=float(regional_threshold),
                        global_threshold=float(global_threshold),
                    )
                )
    return configs


def _choose_global_threshold(predictions_by_dataset: dict[str, pd.DataFrame]) -> float:
    best_threshold = 0.5
    best_score = -float("inf")
    for threshold in np.round(np.linspace(0.20, 0.80, 121), 3):
        rows = []
        for dataset in ["global_nonbr_no_abraom", "nonbr_only"]:
            predictions = predictions_by_dataset[dataset]
            rows.append(
                _metric_row(
                    model="global_candidate",
                    dataset=dataset,
                    labels=predictions["label"].to_numpy(dtype=np.int64),
                    scores=predictions["molecular_probability"].to_numpy(dtype=np.float64),
                    threshold=float(threshold),
                )
            )
        score = (
            1.5 * _value(rows, "global_nonbr_no_abraom", "mcc")
            + _value(rows, "global_nonbr_no_abraom", "specificity")
            + 0.5 * _value(rows, "nonbr_only", "mcc")
        )
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def _score_holdout_config(rows: list[dict[str, Any]], baseline: pd.DataFrame) -> float:
    br_mcc = _value(rows, "br_only", "mcc")
    br_spec = _value(rows, "br_only", "specificity")
    benign_spec = _value(rows, "abraom_common_benign", "specificity")
    plp_recall = _value(rows, "abraom_pathogenic_present", "recall")
    plp_common_recall = _value(rows, "abraom_pathogenic_common", "recall")
    global_mcc = _value(rows, "global_nonbr_no_abraom", "mcc")
    global_spec = _value(rows, "global_nonbr_no_abraom", "specificity")

    m0_br_mcc = _baseline_value(baseline, "M0", "br_only", "mcc")
    m0_global_mcc = _baseline_value(baseline, "M0", "global_nonbr_no_abraom", "mcc")
    m0_global_spec = _baseline_value(baseline, "M0", "global_nonbr_no_abraom", "specificity")

    reward = (
        4.0 * plp_recall
        + 0.75 * plp_common_recall
        + 1.25 * br_mcc
        + 0.50 * br_spec
        + 0.75 * benign_spec
        + 0.75 * global_mcc
        + 0.50 * global_spec
    )
    penalty = 0.0
    penalty += 25.0 * max(0.0, 0.95 - benign_spec)
    if not math.isnan(m0_br_mcc):
        penalty += 8.0 * max(0.0, m0_br_mcc - br_mcc)
    if not math.isnan(m0_global_mcc):
        penalty += 6.0 * max(0.0, (m0_global_mcc - 0.03) - global_mcc)
    if not math.isnan(m0_global_spec):
        penalty += 6.0 * max(0.0, (m0_global_spec - 0.03) - global_spec)
    penalty += 4.0 * max(0.0, 0.25 - plp_recall)
    return reward - penalty


def _passes_hard_constraints(rows: list[dict[str, Any]], baseline: pd.DataFrame) -> bool:
    br_mcc = _value(rows, "br_only", "mcc")
    benign_spec = _value(rows, "abraom_common_benign", "specificity")
    global_mcc = _value(rows, "global_nonbr_no_abraom", "mcc")
    global_spec = _value(rows, "global_nonbr_no_abraom", "specificity")
    m0_br_mcc = _baseline_value(baseline, "M0", "br_only", "mcc")
    m0_global_mcc = _baseline_value(baseline, "M0", "global_nonbr_no_abraom", "mcc")
    m0_global_spec = _baseline_value(baseline, "M0", "global_nonbr_no_abraom", "specificity")
    if benign_spec < 0.95:
        return False
    if not math.isnan(m0_br_mcc) and br_mcc < m0_br_mcc:
        return False
    if not math.isnan(m0_global_mcc) and global_mcc < (m0_global_mcc - 0.05):
        return False
    return not (not math.isnan(m0_global_spec) and global_spec < (m0_global_spec - 0.05))


def tune_config(
    predictions_by_dataset: dict[str, pd.DataFrame],
    baseline: pd.DataFrame,
    *,
    model_name: str,
) -> tuple[RegionalCalibrationConfig, pd.DataFrame]:
    global_threshold = _choose_global_threshold(predictions_by_dataset)
    tuning_rows: list[dict[str, Any]] = []
    best_config: RegionalCalibrationConfig | None = None
    best_score = -float("inf")
    best_valid_config: RegionalCalibrationConfig | None = None
    best_valid_score = -float("inf")
    for config in _candidate_regional_configs(global_threshold):
        rows = _metrics_for_config(predictions_by_dataset, config, model_name=model_name)
        score = _score_holdout_config(rows, baseline)
        passes_constraints = _passes_hard_constraints(rows, baseline)
        row = {
            "score": score,
            "passes_hard_constraints": passes_constraints,
            **asdict(config),
            "br_only_mcc": _value(rows, "br_only", "mcc"),
            "br_only_recall": _value(rows, "br_only", "recall"),
            "br_only_specificity": _value(rows, "br_only", "specificity"),
            "abraom_common_benign_specificity": _value(rows, "abraom_common_benign", "specificity"),
            "abraom_pathogenic_present_recall": _value(rows, "abraom_pathogenic_present", "recall"),
            "abraom_pathogenic_common_recall": _value(rows, "abraom_pathogenic_common", "recall"),
            "global_nonbr_no_abraom_mcc": _value(rows, "global_nonbr_no_abraom", "mcc"),
            "global_nonbr_no_abraom_specificity": _value(
                rows, "global_nonbr_no_abraom", "specificity"
            ),
        }
        tuning_rows.append(row)
        if score > best_score:
            best_score = score
            best_config = config
        if passes_constraints and score > best_valid_score:
            best_valid_score = score
            best_valid_config = config
    if best_valid_config is not None:
        return best_valid_config, pd.DataFrame(tuning_rows).sort_values("score", ascending=False)
    if best_config is None:
        raise RuntimeError("No calibration config was evaluated.")
    return best_config, pd.DataFrame(tuning_rows).sort_values("score", ascending=False)


def apply_config(
    predictions_by_dataset: dict[str, pd.DataFrame],
    config: RegionalCalibrationConfig,
    *,
    model_name: str,
    output_dir: Path,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for dataset, predictions in predictions_by_dataset.items():
        labels = predictions["label"].to_numpy(dtype=np.int64)
        regional_score = _regional_score(predictions, config.discount_scale, config.max_discount)
        global_score = predictions["molecular_probability"].to_numpy(dtype=np.float64)
        if dataset in GLOBAL_DATASETS:
            score_used = global_score
            threshold = config.global_threshold
            decision_context = "global_molecular"
        else:
            score_used = regional_score
            threshold = config.regional_threshold
            decision_context = "brazil_regional"
        output = predictions.copy()
        output["regional_score_calibrated"] = regional_score
        output["global_score_calibrated"] = global_score
        output["score_used"] = score_used
        output["threshold_used"] = threshold
        output["decision_context"] = decision_context
        output["prediction_used"] = (score_used >= threshold).astype(int)
        output.to_parquet(output_dir / f"{dataset}.test.predictions.parquet", index=False)
        rows.append(
            _metric_row(
                model=model_name,
                dataset=dataset,
                labels=labels,
                scores=score_used,
                threshold=threshold,
            )
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "m5_v2_calibrated_regional_test_summary.csv", index=False)
    summary.to_parquet(output_dir / "m5_v2_calibrated_regional_test_summary.parquet", index=False)
    return summary


def _load_split(root: Path, datasets: list[str], split: str) -> dict[str, pd.DataFrame]:
    return {dataset: _read_predictions(root, dataset, split) for dataset in datasets}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline = pd.read_csv(args.baseline_csv)
    tuning_baseline = pd.read_csv(args.tuning_baseline_csv) if args.tuning_baseline_csv else baseline

    holdout = _load_split(args.holdout_dir, args.datasets, "holdout")
    test = _load_split(args.test_dir, args.datasets, "test")
    selected_config, tuning = tune_config(holdout, tuning_baseline, model_name=args.model_name)
    tuning.to_csv(args.output_dir / "holdout_tuning_results.csv", index=False)
    tuning.to_parquet(args.output_dir / "holdout_tuning_results.parquet", index=False)
    (args.output_dir / "selected_config.json").write_text(
        json.dumps(asdict(selected_config), indent=2),
        encoding="utf-8",
    )

    test_summary = apply_config(test, selected_config, model_name=args.model_name, output_dir=args.output_dir)
    combined = pd.concat([baseline, test_summary[baseline.columns]], ignore_index=True)
    combined.to_csv(args.output_dir / "m0_m4_m5_m6_m5cal_m6cal_m7_m5v2cal_summary.csv", index=False)
    combined.to_parquet(args.output_dir / "m0_m4_m5_m6_m5cal_m6cal_m7_m5v2cal_summary.parquet", index=False)
    print(f"selected_config={args.output_dir / 'selected_config.json'}")
    print(test_summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
