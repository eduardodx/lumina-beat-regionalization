#!/usr/bin/env python3
"""Post-hoc regional calibration for ClinVar ABRAOM experiments.

The v1 M5/M6 runs showed that explicit ABRAOM frequency evidence improves
regional specificity but can over-suppress ABRAOM-present P/LP variants.  This
script builds a v2 calibration layer over the existing regional predictions:

* molecular_score comes from M0 and is never increased by regional evidence;
* learned regional discount comes from M5/M6 only when those models lower M0;
* explicit ABRAOM frequency gates cap how far that discount may lower M0;
* high ABRAOM-vs-gnomAD specificity reduces the discount as a founder-safety
  heuristic;
* M7_scrambled uses the same learned discounts with frequency metadata
  deterministically scrambled across regional rows.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.clinvar.metrics import (  # noqa: E402
    binary_auprc,
    binary_log_loss,
    binary_roc_auc,
    brier_score,
    classification_metrics,
)

EPS = 1e-6
DEFAULT_MODEL_ROOT = Path("artifacts/clinvar_regional_eval")
DEFAULT_SLICE_DIR = Path("data/datasets/clinvar/regional_abraom/slices")
DEFAULT_COMPARISON_DIR = Path("artifacts/clinvar_regional_comparison")
DEFAULT_OUTPUT_DIR = Path("artifacts/clinvar_regional_calibration_v2_holdout_tuned")
DEFAULT_BASELINE_CSV = DEFAULT_COMPARISON_DIR / "m0_m4_m5_m6_regional_test_summary.csv"
DEFAULT_TUNE_SPLIT = "holdout"
DEFAULT_EVAL_SPLIT = "test"
DEFAULT_DATASETS = [
    "br_only",
    "br_any",
    "regional_benchmark_any",
    "global_nonbr_no_abraom",
    "nonbr_only",
    "abraom_common_benign",
    "abraom_pathogenic_present",
    "abraom_pathogenic_common",
]
MODEL_DIRS = {
    "M0": "m0_nonbr_beatv10_v1_sagemaker",
    "M4": "m4_staticfusion_nonbr_beatv10_v1_sagemaker",
    "M5": "m5_staticfusion_explicitfreq_nonbr_beatv10_v1_sagemaker",
    "M6": "m6_explicitfreq_nonbr_beatv10_v1_sagemaker",
}


@dataclass(frozen=True)
class CalibrationConfig:
    """Parameters controlling the bounded regional frequency discount."""

    max_down_margin: float = 4.0
    af_midpoint: float = 0.05
    af_log10_temperature: float = 0.25
    specificity_protect_threshold: float = 0.05
    specificity_temperature: float = 0.01
    scrambled_seed: int = 1729


@dataclass(frozen=True)
class CalibratedModelSpec:
    """A calibrated model defined by a raw regional prediction source."""

    name: str
    raw_model: str
    frequency_source: str


CALIBRATED_MODELS = [
    CalibratedModelSpec("M5_calibrated", "M5", "real"),
    CalibratedModelSpec("M6_calibrated", "M6", "real"),
    CalibratedModelSpec("M7_scrambled", "M5", "scrambled"),
]

TUNING_DATASETS = [
    "br_only",
    "abraom_common_benign",
    "abraom_pathogenic_present",
    "abraom_pathogenic_common",
    "global_nonbr_no_abraom",
]


def _logit_scalar(probability: float) -> float:
    probability = min(1.0 - EPS, max(EPS, float(probability)))
    return math.log(probability / (1.0 - probability))


def logit_array(probabilities: pd.Series | np.ndarray) -> np.ndarray:
    probs = np.clip(np.asarray(probabilities, dtype=np.float64), EPS, 1.0 - EPS)
    return np.log(probs / (1.0 - probs))


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def frequency_cap(
    af_abraom: pd.Series | np.ndarray,
    specificity: pd.Series | np.ndarray,
    config: CalibrationConfig,
) -> np.ndarray:
    """Return per-row maximum allowed downward margin movement."""

    af = np.nan_to_num(np.asarray(af_abraom, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    spec = np.nan_to_num(np.asarray(specificity, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    log_af = np.log10(np.maximum(af, EPS))
    log_mid = math.log10(max(config.af_midpoint, EPS))
    af_gate = sigmoid((log_af - log_mid) / config.af_log10_temperature)
    specificity_gate = sigmoid(
        (config.specificity_protect_threshold - spec) / config.specificity_temperature
    )
    return config.max_down_margin * af_gate * specificity_gate


def apply_capped_frequency_discount(
    *,
    molecular_probability: pd.Series | np.ndarray,
    raw_regional_probability: pd.Series | np.ndarray,
    molecular_threshold: float,
    raw_regional_threshold: float,
    af_abraom: pd.Series | np.ndarray,
    specificity: pd.Series | np.ndarray,
    config: CalibrationConfig,
) -> pd.DataFrame:
    """Apply the v2 safety rule and return normalized scores and diagnostics."""

    molecular_margin = logit_array(molecular_probability) - _logit_scalar(molecular_threshold)
    raw_regional_margin = logit_array(raw_regional_probability) - _logit_scalar(raw_regional_threshold)
    learned_discount = np.maximum(0.0, molecular_margin - raw_regional_margin)
    cap = frequency_cap(af_abraom, specificity, config)
    capped_discount = np.minimum(learned_discount, cap)
    regional_margin = molecular_margin - capped_discount
    return pd.DataFrame(
        {
            "molecular_score": sigmoid(molecular_margin),
            "raw_regional_score": sigmoid(raw_regional_margin),
            "regional_score": sigmoid(regional_margin),
            "molecular_margin": molecular_margin,
            "raw_regional_margin": raw_regional_margin,
            "regional_margin": regional_margin,
            "learned_frequency_discount": learned_discount,
            "frequency_cap": cap,
            "capped_frequency_discount": capped_discount,
        }
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_thresholds(model_root: Path) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for model, dirname in MODEL_DIRS.items():
        summary_path = model_root / dirname / "regional_eval_summary.json"
        if summary_path.is_file():
            thresholds[model] = float(_read_json(summary_path)["threshold"])
    missing = sorted(set(MODEL_DIRS) - set(thresholds))
    if missing:
        raise FileNotFoundError(f"Missing regional_eval_summary.json for models: {', '.join(missing)}")
    return thresholds


def load_predictions(model_root: Path, model: str, dataset: str, split: str) -> pd.DataFrame:
    path = model_root / MODEL_DIRS[model] / f"{dataset}.{split}.predictions.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"Missing predictions: {path}")
    return pd.read_parquet(path)


def load_slice_metadata(slice_dir: Path, dataset: str, split: str) -> pd.DataFrame:
    path = slice_dir / f"{dataset}.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"Missing regional slice: {path}")
    df = pd.read_parquet(path)
    split_values = df["split_within_gene"].astype(str).str.lower()
    df = df.loc[split_values == split.lower()].copy()
    df = df.reset_index(drop=False).rename(columns={"index": "original_index"})
    keep = [
        "original_index",
        "af_abraom",
        "af_gnomad",
        "specificity",
        "specificity_bin",
        "abraom_present",
        "variant_key",
        "GeneSymbol",
    ]
    available = [column for column in keep if column in df.columns]
    return df[available]


def build_frequency_table(slice_dir: Path, datasets: list[str], split: str) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for dataset in datasets:
        meta = load_slice_metadata(slice_dir, dataset, split).copy()
        meta.insert(0, "dataset", dataset)
        rows.append(meta)
    return pd.concat(rows, ignore_index=True)


def build_scrambled_frequency_table(
    frequency_table: pd.DataFrame,
    *,
    seed: int,
) -> pd.DataFrame:
    """Shuffle frequency triples across all evaluated rows deterministically."""

    scrambled = frequency_table.copy()
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(scrambled))
    for column in ["af_abraom", "af_gnomad", "specificity"]:
        values = scrambled[column].to_numpy(copy=True)
        scrambled[f"scrambled_{column}"] = values[order]
    return scrambled[
        [
            "dataset",
            "original_index",
            "scrambled_af_abraom",
            "scrambled_af_gnomad",
            "scrambled_specificity",
        ]
    ]


def _metric_row(model: str, dataset: str, labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    labels = labels.astype(np.int64)
    scores = scores.astype(np.float64)
    metrics = classification_metrics(labels, scores, 0.5)
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
    }


def calibrate_dataset(
    *,
    spec: CalibratedModelSpec,
    dataset: str,
    split: str,
    model_root: Path,
    slice_dir: Path,
    thresholds: dict[str, float],
    scrambled_frequency: pd.DataFrame,
    config: CalibrationConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    molecular = load_predictions(model_root, "M0", dataset, split)
    raw = load_predictions(model_root, spec.raw_model, dataset, split)
    meta = load_slice_metadata(slice_dir, dataset, split)
    merged = molecular.merge(
        raw,
        on=["dataset", "split", "original_index", "label"],
        suffixes=("_molecular", "_raw"),
        validate="one_to_one",
    ).merge(meta, on="original_index", how="left", validate="one_to_one")
    if spec.frequency_source == "scrambled":
        scrambled = scrambled_frequency.loc[scrambled_frequency["dataset"] == dataset]
        merged = merged.merge(scrambled, on=["dataset", "original_index"], how="left", validate="one_to_one")
        af_abraom = merged["scrambled_af_abraom"]
        af_gnomad = merged["scrambled_af_gnomad"]
        specificity = merged["scrambled_specificity"]
    elif spec.frequency_source == "real":
        af_abraom = merged["af_abraom"]
        af_gnomad = merged["af_gnomad"]
        specificity = merged["specificity"]
    else:
        raise ValueError(f"Unsupported frequency_source: {spec.frequency_source!r}")

    scored = apply_capped_frequency_discount(
        molecular_probability=merged["probability_molecular"],
        raw_regional_probability=merged["probability_raw"],
        molecular_threshold=thresholds["M0"],
        raw_regional_threshold=thresholds[spec.raw_model],
        af_abraom=af_abraom,
        specificity=specificity,
        config=config,
    )
    output = pd.DataFrame(
        {
            "dataset": merged["dataset"],
            "split": merged["split"],
            "original_index": merged["original_index"].astype(int),
            "label": merged["label"].astype(int),
            "molecular_probability": merged["probability_molecular"].astype(float),
            "raw_regional_probability": merged["probability_raw"].astype(float),
            "af_abraom": pd.to_numeric(af_abraom, errors="coerce"),
            "af_gnomad": pd.to_numeric(af_gnomad, errors="coerce"),
            "specificity": pd.to_numeric(specificity, errors="coerce"),
            "specificity_bin": merged.get("specificity_bin", ""),
            "abraom_present": merged.get("abraom_present", False),
            "variant_key": merged.get("variant_key", ""),
            "gene_symbol": merged.get("GeneSymbol", ""),
            "frequency_source": spec.frequency_source,
            **{column: scored[column] for column in scored.columns},
        }
    )
    row = _metric_row(
        spec.name,
        dataset,
        output["label"].to_numpy(dtype=np.int64),
        output["regional_score"].to_numpy(dtype=np.float64),
    )
    return output, row


def _score_raw_predictions(predictions: pd.DataFrame, threshold: float) -> np.ndarray:
    return sigmoid(logit_array(predictions["probability"]) - _logit_scalar(threshold))


def raw_model_metric_row(
    *,
    model: str,
    dataset: str,
    split: str,
    model_root: Path,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    predictions = load_predictions(model_root, model, dataset, split)
    scores = _score_raw_predictions(predictions, thresholds[model])
    return _metric_row(model, dataset, predictions["label"].to_numpy(dtype=np.int64), scores)


def _candidate_configs(seed: int) -> list[CalibrationConfig]:
    configs: list[CalibrationConfig] = []
    for max_down_margin in [1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]:
        for af_midpoint in [0.01, 0.02, 0.05, 0.10]:
            for af_log10_temperature in [0.15, 0.25, 0.35, 0.50]:
                for specificity_protect_threshold in [0.02, 0.05, 0.10]:
                    configs.append(
                        CalibrationConfig(
                            max_down_margin=max_down_margin,
                            af_midpoint=af_midpoint,
                            af_log10_temperature=af_log10_temperature,
                            specificity_protect_threshold=specificity_protect_threshold,
                            specificity_temperature=0.01,
                            scrambled_seed=seed,
                        )
                    )
    return configs


def _summary_value(rows: list[dict[str, Any]], model: str, dataset: str, metric: str) -> float:
    for row in rows:
        if row["model"] == model and row["dataset"] == dataset:
            return float(row[metric])
    return float("nan")


def _tuning_score(
    *,
    rows: list[dict[str, Any]],
    reference_rows: list[dict[str, Any]],
    model: str,
) -> float:
    br_mcc = _summary_value(rows, model, "br_only", "mcc")
    benign_spec = _summary_value(rows, model, "abraom_common_benign", "specificity")
    path_recall = _summary_value(rows, model, "abraom_pathogenic_present", "recall")
    path_common_recall = _summary_value(rows, model, "abraom_pathogenic_common", "recall")
    global_mcc = _summary_value(rows, model, "global_nonbr_no_abraom", "mcc")
    global_spec = _summary_value(rows, model, "global_nonbr_no_abraom", "specificity")

    m0_br_mcc = _summary_value(reference_rows, "M0", "br_only", "mcc")
    m0_global_mcc = _summary_value(reference_rows, "M0", "global_nonbr_no_abraom", "mcc")
    m0_global_spec = _summary_value(reference_rows, "M0", "global_nonbr_no_abraom", "specificity")
    m6_path_recall = _summary_value(reference_rows, "M6", "abraom_pathogenic_present", "recall")

    reward = (
        2.0 * path_recall
        + 0.50 * path_common_recall
        + 1.25 * br_mcc
        + 0.75 * benign_spec
        + 0.25 * global_mcc
        + 0.25 * global_spec
    )
    penalty = 0.0
    penalty += 20.0 * max(0.0, 0.95 - benign_spec)
    penalty += 8.0 * max(0.0, m0_br_mcc - br_mcc)
    penalty += 8.0 * max(0.0, (m0_global_mcc - 0.03) - global_mcc)
    penalty += 8.0 * max(0.0, (m0_global_spec - 0.03) - global_spec)
    penalty += 8.0 * max(0.0, (m6_path_recall + 0.10) - path_recall)
    return reward - penalty


def tune_calibration_configs(
    *,
    model_root: Path,
    slice_dir: Path,
    datasets: list[str],
    tune_split: str,
    thresholds: dict[str, float],
    seed: int,
) -> tuple[dict[str, CalibrationConfig], pd.DataFrame, pd.DataFrame]:
    frequency_table = build_frequency_table(slice_dir, datasets, tune_split)
    scrambled_frequency = build_scrambled_frequency_table(frequency_table, seed=seed)
    reference_rows = [
        raw_model_metric_row(
            model=model,
            dataset=dataset,
            split=tune_split,
            model_root=model_root,
            thresholds=thresholds,
        )
        for model in ["M0", "M5", "M6"]
        for dataset in datasets
    ]

    tuning_rows: list[dict[str, Any]] = []
    selected: dict[str, CalibrationConfig] = {}
    for spec in [CALIBRATED_MODELS[0], CALIBRATED_MODELS[1]]:
        best_score = -float("inf")
        best_config: CalibrationConfig | None = None
        for config in _candidate_configs(seed):
            rows = [
                calibrate_dataset(
                    spec=spec,
                    dataset=dataset,
                    split=tune_split,
                    model_root=model_root,
                    slice_dir=slice_dir,
                    thresholds=thresholds,
                    scrambled_frequency=scrambled_frequency,
                    config=config,
                )[1]
                for dataset in datasets
            ]
            score = _tuning_score(rows=rows, reference_rows=reference_rows, model=spec.name)
            tuning_rows.append(
                {
                    "model": spec.name,
                    "score": score,
                    **asdict(config),
                    "br_only_mcc": _summary_value(rows, spec.name, "br_only", "mcc"),
                    "abraom_common_benign_specificity": _summary_value(
                        rows, spec.name, "abraom_common_benign", "specificity"
                    ),
                    "abraom_pathogenic_present_recall": _summary_value(
                        rows, spec.name, "abraom_pathogenic_present", "recall"
                    ),
                    "abraom_pathogenic_common_recall": _summary_value(
                        rows, spec.name, "abraom_pathogenic_common", "recall"
                    ),
                    "global_nonbr_no_abraom_mcc": _summary_value(
                        rows, spec.name, "global_nonbr_no_abraom", "mcc"
                    ),
                    "global_nonbr_no_abraom_specificity": _summary_value(
                        rows, spec.name, "global_nonbr_no_abraom", "specificity"
                    ),
                }
            )
            if score > best_score:
                best_score = score
                best_config = config
        if best_config is None:
            raise RuntimeError(f"No calibration config was evaluated for {spec.name}.")
        selected[spec.name] = best_config

    selected["M7_scrambled"] = selected["M5_calibrated"]
    return selected, pd.DataFrame(tuning_rows), pd.DataFrame(reference_rows)


def expected_calibration_error(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    bins: int = 10,
) -> tuple[float, float]:
    labels = labels.astype(np.float64)
    scores = np.clip(scores.astype(np.float64), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    mce = 0.0
    for left, right in itertools.pairwise(edges):
        mask = (scores >= left) & (scores <= right) if right == 1.0 else (scores >= left) & (scores < right)
        if not np.any(mask):
            continue
        confidence = float(np.mean(scores[mask]))
        accuracy = float(np.mean(labels[mask]))
        gap = abs(confidence - accuracy)
        ece += float(np.mean(mask)) * gap
        mce = max(mce, gap)
    return float(ece), float(mce)


def calibration_slope_intercept(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    labels = labels.astype(np.float64)
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return float("nan"), float("nan")
    x = logit_array(scores)
    design = np.column_stack([np.ones_like(x), x])
    beta = np.zeros(2, dtype=np.float64)
    ridge = np.diag([1e-6, 1e-6])
    for _ in range(50):
        p = sigmoid(design @ beta)
        weight = np.maximum(p * (1.0 - p), 1e-8)
        grad = design.T @ (labels - p) - ridge @ beta
        hessian = -(design.T * weight) @ design - ridge
        step = np.linalg.solve(hessian, grad)
        beta -= step
        if float(np.max(np.abs(step))) < 1e-8:
            break
    return float(beta[0]), float(beta[1])


def calibration_metric_row(model: str, dataset: str, labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    ece, mce = expected_calibration_error(labels, scores)
    intercept, slope = calibration_slope_intercept(labels, scores)
    return {
        "model": model,
        "dataset": dataset,
        "n": len(labels),
        "brier": brier_score(labels, scores),
        "log_loss": binary_log_loss(labels, scores),
        "ece_10bin": ece,
        "mce_10bin": mce,
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }


def _load_scored_predictions(
    *,
    model: str,
    dataset: str,
    split: str,
    model_root: Path,
    output_dir: Path,
    thresholds: dict[str, float],
) -> pd.DataFrame:
    if model in MODEL_DIRS:
        predictions = load_predictions(model_root, model, dataset, split).copy()
        predictions["score"] = _score_raw_predictions(predictions, thresholds[model])
        return predictions[["dataset", "split", "original_index", "label", "score"]]
    path = output_dir / model / f"{dataset}.{split}.predictions.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"Missing calibrated predictions: {path}")
    predictions = pd.read_parquet(path).copy()
    predictions["score"] = predictions["regional_score"].astype(float)
    return predictions


def build_calibration_metrics(
    *,
    model_root: Path,
    output_dir: Path,
    thresholds: dict[str, float],
    datasets: list[str],
    split: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model in ["M0", "M4", "M5", "M6", "M5_calibrated", "M6_calibrated", "M7_scrambled"]:
        for dataset in datasets:
            predictions = _load_scored_predictions(
                model=model,
                dataset=dataset,
                split=split,
                model_root=model_root,
                output_dir=output_dir,
                thresholds=thresholds,
            )
            rows.append(
                calibration_metric_row(
                    model,
                    dataset,
                    predictions["label"].to_numpy(dtype=np.int64),
                    predictions["score"].to_numpy(dtype=np.float64),
                )
            )
    return pd.DataFrame(rows)


def _metric_value(labels: np.ndarray, scores: np.ndarray, metric: str) -> float:
    metrics = classification_metrics(labels, scores, 0.5)
    return float(metrics[metric])


def _paired_bootstrap_delta(
    *,
    frame: pd.DataFrame,
    metric: str,
    iterations: int,
    seed: int,
) -> tuple[float, float, float]:
    labels = frame["label"].to_numpy(dtype=np.int64)
    base_scores = frame["score_base"].to_numpy(dtype=np.float64)
    compare_scores = frame["score_compare"].to_numpy(dtype=np.float64)
    estimate = _metric_value(labels, compare_scores, metric) - _metric_value(labels, base_scores, metric)
    if iterations <= 0:
        return estimate, float("nan"), float("nan")
    clusters = frame["cluster"].astype(str).to_numpy()
    unique_clusters = np.unique(clusters)
    by_cluster = {cluster: np.flatnonzero(clusters == cluster) for cluster in unique_clusters}
    rng = np.random.default_rng(seed)
    deltas = np.empty(iterations, dtype=np.float64)
    for i in range(iterations):
        sampled = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)
        indices = np.concatenate([by_cluster[cluster] for cluster in sampled])
        deltas[i] = _metric_value(labels[indices], compare_scores[indices], metric) - _metric_value(
            labels[indices],
            base_scores[indices],
            metric,
        )
    low, high = np.quantile(deltas, [0.025, 0.975])
    return float(estimate), float(low), float(high)


def build_bootstrap_intervals(
    *,
    model_root: Path,
    output_dir: Path,
    thresholds: dict[str, float],
    split: str,
    iterations: int,
    seed: int,
) -> pd.DataFrame:
    comparisons = [
        ("M0", "M5_calibrated", "br_only", "mcc"),
        ("M0", "M5_calibrated", "abraom_common_benign", "specificity"),
        ("M0", "M5_calibrated", "abraom_pathogenic_present", "recall"),
        ("M0", "M5_calibrated", "global_nonbr_no_abraom", "mcc"),
        ("M0", "M6_calibrated", "br_only", "mcc"),
        ("M0", "M6_calibrated", "abraom_common_benign", "specificity"),
        ("M0", "M6_calibrated", "abraom_pathogenic_present", "recall"),
        ("M0", "M6_calibrated", "global_nonbr_no_abraom", "mcc"),
        ("M7_scrambled", "M5_calibrated", "br_only", "mcc"),
        ("M7_scrambled", "M5_calibrated", "abraom_common_benign", "specificity"),
    ]
    rows: list[dict[str, Any]] = []
    for base_model, compare_model, dataset, metric in comparisons:
        base = _load_scored_predictions(
            model=base_model,
            dataset=dataset,
            split=split,
            model_root=model_root,
            output_dir=output_dir,
            thresholds=thresholds,
        )
        compare = _load_scored_predictions(
            model=compare_model,
            dataset=dataset,
            split=split,
            model_root=model_root,
            output_dir=output_dir,
            thresholds=thresholds,
        )
        frame = base[["original_index", "label", "score"]].merge(
            compare[["original_index", "score", "gene_symbol"]]
            if "gene_symbol" in compare
            else compare[["original_index", "score"]],
            on="original_index",
            suffixes=("_base", "_compare"),
            validate="one_to_one",
        )
        frame["cluster"] = (
            frame["gene_symbol"].fillna("").astype(str)
            if "gene_symbol" in frame
            else frame["original_index"].astype(str)
        )
        frame.loc[frame["cluster"] == "", "cluster"] = frame.loc[frame["cluster"] == "", "original_index"].astype(str)
        estimate, low, high = _paired_bootstrap_delta(
            frame=frame,
            metric=metric,
            iterations=iterations,
            seed=seed,
        )
        rows.append(
            {
                "base_model": base_model,
                "compare_model": compare_model,
                "dataset": dataset,
                "metric": metric,
                "delta": estimate,
                "ci95_low": low,
                "ci95_high": high,
                "iterations": iterations,
                "cluster": "gene_symbol",
            }
        )
    return pd.DataFrame(rows)


def _false_benign_category(row: pd.Series) -> str:
    if float(row.get("capped_frequency_discount", 0.0)) > 0.5:
        return "regional_af_over_suppression"
    if float(row.get("af_abraom", 0.0) or 0.0) >= 0.01:
        return "common_plp_recessive_or_founder_review"
    if float(row.get("molecular_score", 0.0)) < 0.5:
        return "weak_molecular_signal"
    return "threshold_borderline"


def _false_pathogenic_category(row: pd.Series) -> str:
    if float(row.get("molecular_score", 0.0)) >= 0.8:
        return "molecular_score_overdominant"
    if float(row.get("frequency_cap", 0.0)) < 0.5:
        return "frequency_protected_or_underweighted"
    if float(row.get("regional_score", 0.0)) < 0.6:
        return "threshold_borderline"
    return "frequency_discount_insufficient"


def build_error_analysis(output_dir: Path, split: str) -> dict[str, str]:
    error_dir = output_dir / "error_analysis"
    error_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    columns = [
        "dataset",
        "split",
        "original_index",
        "variant_key",
        "gene_symbol",
        "label",
        "af_abraom",
        "af_gnomad",
        "specificity",
        "specificity_bin",
        "molecular_score",
        "raw_regional_score",
        "regional_score",
        "learned_frequency_discount",
        "frequency_cap",
        "capped_frequency_discount",
        "score_drop",
        "failure_category",
    ]
    for model in ["M5_calibrated", "M6_calibrated", "M7_scrambled"]:
        benign = pd.read_parquet(output_dir / model / f"abraom_common_benign.{split}.predictions.parquet")
        benign = benign.loc[benign["regional_score"] >= 0.5].copy()
        benign["score_drop"] = benign["molecular_score"] - benign["regional_score"]
        benign["failure_category"] = benign.apply(_false_pathogenic_category, axis=1)
        benign_path = error_dir / f"{model}.false_pathogenic_abraom_common_benign.csv"
        benign[[column for column in columns if column in benign.columns]].to_csv(benign_path, index=False)
        outputs[f"{model}_false_pathogenic"] = str(benign_path)

        pathogenic = pd.read_parquet(output_dir / model / f"abraom_pathogenic_present.{split}.predictions.parquet")
        pathogenic = pathogenic.loc[pathogenic["regional_score"] < 0.5].copy()
        pathogenic["score_drop"] = pathogenic["molecular_score"] - pathogenic["regional_score"]
        pathogenic["failure_category"] = pathogenic.apply(_false_benign_category, axis=1)
        pathogenic_path = error_dir / f"{model}.false_benign_abraom_pathogenic_present.csv"
        pathogenic[[column for column in columns if column in pathogenic.columns]].to_csv(pathogenic_path, index=False)
        outputs[f"{model}_false_benign"] = str(pathogenic_path)
    return outputs


def build_sensitivity_panel(slice_dir: Path, output_dir: Path) -> dict[str, Any]:
    panel_dir = output_dir / "sensitivity_panel"
    panel_dir.mkdir(parents=True, exist_ok=True)
    pathogenic_present = pd.read_parquet(slice_dir / "abraom_pathogenic_present.parquet")
    pathogenic_present.to_parquet(panel_dir / "clinvar_plp_abraom_present.parquet", index=False)
    manifest = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "components": [
            {
                "name": "clinvar_plp_abraom_present",
                "path": str(panel_dir / "clinvar_plp_abraom_present.parquet"),
                "rows": len(pathogenic_present),
                "description": "ClinVar pathogenic/likely-pathogenic rows present in ABRAOM.",
                "available": True,
            },
            {
                "name": "known_brazilian_founder_variants",
                "path": None,
                "rows": 0,
                "description": "Not found in local repository artifacts.",
                "available": False,
            },
            {
                "name": "manual_brazilian_plp_curation",
                "path": None,
                "rows": 0,
                "description": "Not found in local repository artifacts.",
                "available": False,
            },
        ],
    }
    (panel_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _criterion_rows(summary: pd.DataFrame) -> list[dict[str, Any]]:
    def value(model: str, dataset: str, metric: str) -> float:
        row = summary.loc[(summary["model"] == model) & (summary["dataset"] == dataset)]
        if row.empty:
            return float("nan")
        return float(row.iloc[0][metric])

    rows: list[dict[str, Any]] = []
    m0_br = value("M0", "br_only", "mcc")
    m0_global_mcc = value("M0", "global_nonbr_no_abraom", "mcc")
    m0_global_spec = value("M0", "global_nonbr_no_abraom", "specificity")
    m0_path_recall = value("M0", "abraom_pathogenic_present", "recall")
    m6_path_recall = value("M6", "abraom_pathogenic_present", "recall")
    for model in ["M5_calibrated", "M6_calibrated", "M7_scrambled"]:
        rows.extend(
            [
                {
                    "model": model,
                    "criterion": "br_only_mcc_above_m0",
                    "value": value(model, "br_only", "mcc"),
                    "reference": m0_br,
                    "passed": bool(value(model, "br_only", "mcc") > m0_br),
                },
                {
                    "model": model,
                    "criterion": "abraom_common_benign_specificity_ge_0_95",
                    "value": value(model, "abraom_common_benign", "specificity"),
                    "reference": 0.95,
                    "passed": bool(value(model, "abraom_common_benign", "specificity") >= 0.95),
                },
                {
                    "model": model,
                    "criterion": "abraom_pathogenic_present_recall_not_collapsed_like_m6",
                    "value": value(model, "abraom_pathogenic_present", "recall"),
                    "reference": m6_path_recall,
                    "m0_reference": m0_path_recall,
                    "passed": bool(value(model, "abraom_pathogenic_present", "recall") > m6_path_recall + 0.10),
                },
                {
                    "model": model,
                    "criterion": "global_nonbr_no_abraom_near_m0",
                    "value": value(model, "global_nonbr_no_abraom", "mcc"),
                    "specificity": value(model, "global_nonbr_no_abraom", "specificity"),
                    "reference": m0_global_mcc,
                    "specificity_reference": m0_global_spec,
                    "passed": bool(
                        value(model, "global_nonbr_no_abraom", "mcc") >= m0_global_mcc - 0.03
                        and value(model, "global_nonbr_no_abraom", "specificity") >= m0_global_spec - 0.03
                    ),
                },
            ]
        )

    m7_br = value("M7_scrambled", "br_only", "mcc")
    m7_benign = value("M7_scrambled", "abraom_common_benign", "specificity")
    real_best_br = max(value("M5_calibrated", "br_only", "mcc"), value("M6_calibrated", "br_only", "mcc"))
    real_best_benign = max(
        value("M5_calibrated", "abraom_common_benign", "specificity"),
        value("M6_calibrated", "abraom_common_benign", "specificity"),
    )
    rows.append(
        {
            "model": "M7_scrambled",
            "criterion": "negative_control_not_equal_or_better_than_real_abraom",
            "value": m7_br,
            "abraom_common_benign_specificity": m7_benign,
            "reference": real_best_br,
            "real_best_abraom_common_benign_specificity": real_best_benign,
            "passed": bool(m7_br < real_best_br or m7_benign < real_best_benign),
        }
    )
    return rows


def _format_metric(value: float) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.3f}"


def write_report(
    *,
    path: Path,
    summary: pd.DataFrame,
    criteria: pd.DataFrame,
    selected_configs: dict[str, CalibrationConfig],
    tune_split: str,
    eval_split: str,
    tuning_top: pd.DataFrame,
    calibration_metrics: pd.DataFrame,
    bootstrap_intervals: pd.DataFrame,
    error_outputs: dict[str, str],
    baseline_manifest: dict[str, Any],
    sensitivity_manifest: dict[str, Any],
) -> None:
    focus = summary.loc[
        summary["dataset"].isin(
            ["br_only", "abraom_common_benign", "abraom_pathogenic_present", "global_nonbr_no_abraom"]
        )
    ].copy()
    lines = [
        "# ClinVar Regional Calibration v2",
        "",
        f"Generated at UTC: `{datetime.now(UTC).isoformat()}`",
        "",
        "## Calibration Rule",
        "",
        "- `molecular_score`: threshold-normalized M0 score.",
        "- `regional_score`: M0 score after a bounded regional discount.",
        "- Regional evidence can only lower M0; it cannot raise pathogenicity above M0.",
        "- The learned M5/M6 discount is capped by ABRAOM AF and reduced for high ABRAOM specificity.",
        "- `M7_scrambled` uses M5 learned discounts with frequency triples scrambled across evaluated rows.",
        f"- Parameters were selected on `{tune_split}` and frozen before `{eval_split}` evaluation.",
        "",
        "## Selected Parameters",
        "",
        "```json",
        json.dumps({model: asdict(config) for model, config in selected_configs.items()}, indent=2, sort_keys=True),
        "```",
        "",
        "## Holdout Tuning Winners",
        "",
        "| Model | Score | Max down margin | AF midpoint | AF temp | Specificity protect | "
        "BR MCC | Benign specificity | P/LP recall |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in tuning_top.itertuples(index=False):
        lines.append(
            f"| {row.model} | {_format_metric(row.score)} | {_format_metric(row.max_down_margin)} | "
            f"{_format_metric(row.af_midpoint)} | {_format_metric(row.af_log10_temperature)} | "
            f"{_format_metric(row.specificity_protect_threshold)} | {_format_metric(row.br_only_mcc)} | "
            f"{_format_metric(row.abraom_common_benign_specificity)} | "
            f"{_format_metric(row.abraom_pathogenic_present_recall)} |"
        )
    lines.extend(
        [
        "",
        "## Focus Metrics",
        "",
        "| Model | Dataset | N | AUROC | AUPRC | MCC | Recall | Specificity |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in focus.itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.dataset} | {int(row.n)} | {_format_metric(row.auroc)} | "
            f"{_format_metric(row.auprc)} | {_format_metric(row.mcc)} | {_format_metric(row.recall)} | "
            f"{_format_metric(row.specificity)} |"
        )
    lines.extend(
        [
            "",
            "## Criteria",
            "",
            "| Model | Criterion | Value | Reference | Passed |",
            "|---|---|---:|---:|---|",
        ]
    )
    for row in criteria.itertuples(index=False):
        lines.append(
            "| {model} | {criterion} | {value} | {reference} | {passed} |".format(
                model=row.model,
                criterion=row.criterion,
                value=_format_metric(row.value),
                reference=_format_metric(row.reference),
                passed="yes" if bool(row.passed) else "no",
            )
        )
    lines.extend(
        [
            "",
            "## Calibration Metrics",
            "",
            "| Model | Dataset | Brier | ECE | MCE | Slope | Intercept |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    calibration_focus = calibration_metrics.loc[
        calibration_metrics["dataset"].isin(["br_only", "global_nonbr_no_abraom", "regional_benchmark_any"])
    ]
    for row in calibration_focus.itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.dataset} | {_format_metric(row.brier)} | {_format_metric(row.ece_10bin)} | "
            f"{_format_metric(row.mce_10bin)} | {_format_metric(row.calibration_slope)} | "
            f"{_format_metric(row.calibration_intercept)} |"
        )
    lines.extend(
        [
            "",
            "## Cluster Bootstrap",
            "",
            "| Comparison | Dataset | Metric | Delta | 95% CI |",
            "|---|---|---|---:|---:|",
        ]
    )
    for row in bootstrap_intervals.itertuples(index=False):
        lines.append(
            f"| {row.compare_model} - {row.base_model} | {row.dataset} | {row.metric} | "
            f"{_format_metric(row.delta)} | [{_format_metric(row.ci95_low)}, {_format_metric(row.ci95_high)}] |"
        )
    lines.extend(["", "## Error Analysis", ""])
    for name, output_path in sorted(error_outputs.items()):
        lines.append(f"- `{name}`: `{output_path}`")
    lines.extend(
        [
            "",
            "## Baseline v1",
            "",
            f"- CSV: `{baseline_manifest['path']}`",
            f"- SHA256: `{baseline_manifest['sha256']}`",
            "- Conclusion frozen: ABRAOM improves regional specificity, but v1 frequency weighting is too strong.",
            "",
            "## Sensitivity Panel",
            "",
        ]
    )
    for component in sensitivity_manifest["components"]:
        status = "available" if component["available"] else "missing"
        lines.append(f"- `{component['name']}`: {status}, rows={component['rows']}")
    lines.extend(
        [
            "",
            "## Release Recommendation",
            "",
            (
                "Research-only. The holdout-tuned calibration passes the current slice criteria, but the curated "
                "founder/manual P/LP sentinel panel is still missing and the constrained rule is post-hoc rather "
                "than a trained dynamic fusion controller."
            ),
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_calibration(
    *,
    model_root: Path,
    slice_dir: Path,
    output_dir: Path,
    baseline_csv: Path,
    datasets: list[str],
    tune_split: str,
    eval_split: str,
    seed: int,
    bootstrap_iterations: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = load_thresholds(model_root)
    tuning_datasets = [dataset for dataset in TUNING_DATASETS if dataset in datasets]
    selected_configs, tuning_results, tuning_reference = tune_calibration_configs(
        model_root=model_root,
        slice_dir=slice_dir,
        datasets=tuning_datasets,
        tune_split=tune_split,
        thresholds=thresholds,
        seed=seed,
    )
    tuning_results_path = output_dir / "holdout_tuning_results.csv"
    tuning_results.to_csv(tuning_results_path, index=False)
    tuning_results.to_parquet(output_dir / "holdout_tuning_results.parquet", index=False)
    tuning_reference.to_csv(output_dir / "holdout_reference_metrics.csv", index=False)
    selected_config_path = output_dir / "selected_calibration_config.json"
    selected_config_path.write_text(
        json.dumps(
            {
                "selected_on_split": tune_split,
                "evaluated_on_split": eval_split,
                "configs": {model: asdict(config) for model, config in selected_configs.items()},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    frequency_table = build_frequency_table(slice_dir, datasets, eval_split)
    scrambled_frequency = build_scrambled_frequency_table(frequency_table, seed=seed)

    calibrated_rows: list[dict[str, Any]] = []
    for spec in CALIBRATED_MODELS:
        model_dir = output_dir / spec.name
        model_dir.mkdir(parents=True, exist_ok=True)
        spec_config = selected_configs[spec.name]
        for dataset in datasets:
            predictions, row = calibrate_dataset(
                spec=spec,
                dataset=dataset,
                split=eval_split,
                model_root=model_root,
                slice_dir=slice_dir,
                thresholds=thresholds,
                scrambled_frequency=scrambled_frequency,
                config=spec_config,
            )
            predictions.to_parquet(model_dir / f"{dataset}.{eval_split}.predictions.parquet", index=False)
            calibrated_rows.append(row)

    baseline = pd.read_csv(baseline_csv)
    baseline = baseline.loc[baseline["dataset"].isin(datasets)].copy()
    calibrated = pd.DataFrame(calibrated_rows)
    summary = pd.concat([baseline, calibrated], ignore_index=True)
    summary_path = output_dir / "m0_m4_m5_m6_m5cal_m6cal_m7_regional_test_summary.csv"
    summary.to_csv(summary_path, index=False)
    summary.to_parquet(output_dir / "m0_m4_m5_m6_m5cal_m6cal_m7_regional_test_summary.parquet", index=False)

    comparison_copy = DEFAULT_COMPARISON_DIR / summary_path.name
    comparison_copy.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(comparison_copy, index=False)

    baseline_manifest = {
        "path": str(baseline_csv),
        "sha256": _sha256(baseline_csv),
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "conclusion": "ABRAOM improves regional specificity, but v1 frequency weighting is too strong.",
    }
    (output_dir / "baseline_v1_manifest.json").write_text(
        json.dumps(baseline_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    sensitivity_manifest = build_sensitivity_panel(slice_dir, output_dir)
    criteria = pd.DataFrame(_criterion_rows(summary))
    criteria.to_csv(output_dir / "success_criteria.csv", index=False)
    criteria.to_parquet(output_dir / "success_criteria.parquet", index=False)
    calibration_metrics = build_calibration_metrics(
        model_root=model_root,
        output_dir=output_dir,
        thresholds=thresholds,
        datasets=datasets,
        split=eval_split,
    )
    calibration_metrics.to_csv(output_dir / "calibration_metrics.csv", index=False)
    calibration_metrics.to_parquet(output_dir / "calibration_metrics.parquet", index=False)
    bootstrap_intervals = build_bootstrap_intervals(
        model_root=model_root,
        output_dir=output_dir,
        thresholds=thresholds,
        split=eval_split,
        iterations=bootstrap_iterations,
        seed=seed,
    )
    bootstrap_intervals.to_csv(output_dir / "cluster_bootstrap_intervals.csv", index=False)
    bootstrap_intervals.to_parquet(output_dir / "cluster_bootstrap_intervals.parquet", index=False)
    error_outputs = build_error_analysis(output_dir, eval_split)

    report_path = output_dir / "REGIONAL_CALIBRATION_V2_SUMMARY.md"
    tuning_top = (
        tuning_results.sort_values(["model", "score"], ascending=[True, False])
        .groupby("model", as_index=False)
        .head(1)
        .sort_values("model")
    )
    write_report(
        path=report_path,
        summary=summary,
        criteria=criteria,
        selected_configs=selected_configs,
        tune_split=tune_split,
        eval_split=eval_split,
        tuning_top=tuning_top,
        calibration_metrics=calibration_metrics,
        bootstrap_intervals=bootstrap_intervals,
        error_outputs=error_outputs,
        baseline_manifest=baseline_manifest,
        sensitivity_manifest=sensitivity_manifest,
    )
    (DEFAULT_COMPARISON_DIR / report_path.name).write_text(report_path.read_text(encoding="utf-8"), encoding="utf-8")

    return {
        "summary_csv": str(summary_path),
        "comparison_csv": str(comparison_copy),
        "criteria_csv": str(output_dir / "success_criteria.csv"),
        "selected_config": str(selected_config_path),
        "tuning_results": str(tuning_results_path),
        "calibration_metrics": str(output_dir / "calibration_metrics.csv"),
        "bootstrap_intervals": str(output_dir / "cluster_bootstrap_intervals.csv"),
        "error_analysis_dir": str(output_dir / "error_analysis"),
        "report": str(report_path),
        "baseline_manifest": str(output_dir / "baseline_v1_manifest.json"),
        "sensitivity_manifest": str(output_dir / "sensitivity_panel" / "manifest.json"),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--slice-dir", type=Path, default=DEFAULT_SLICE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--baseline-csv", type=Path, default=DEFAULT_BASELINE_CSV)
    parser.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    parser.add_argument("--tune-split", default=DEFAULT_TUNE_SPLIT)
    parser.add_argument("--eval-split", default=DEFAULT_EVAL_SPLIT)
    parser.add_argument("--scrambled-seed", type=int, default=CalibrationConfig.scrambled_seed)
    parser.add_argument("--bootstrap-iterations", type=int, default=500)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_calibration(
        model_root=args.model_root,
        slice_dir=args.slice_dir,
        output_dir=args.output_dir,
        baseline_csv=args.baseline_csv,
        datasets=list(args.datasets),
        tune_split=args.tune_split,
        eval_split=args.eval_split,
        seed=args.scrambled_seed,
        bootstrap_iterations=args.bootstrap_iterations,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
