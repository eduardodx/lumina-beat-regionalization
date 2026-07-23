#!/usr/bin/env python3
"""Calibrate and falsify-test an M5_v3 safety layer for ABRAOM regionalization.

This is a post-hoc safety calibration over the already trained M5_v2 regional
head. It deliberately selects all parameters on holdout and only audits the
locked configuration on test.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

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

KEY_CONTROL_METRICS = [
    ("br_only", "mcc"),
    ("abraom_common_benign", "specificity"),
    ("abraom_pathogenic_present", "recall"),
    ("global_nonbr_no_abraom", "mcc"),
]


@dataclass(frozen=True)
class SafetyConfig:
    discount_scale: float
    max_discount: float
    molecular_guard_threshold: float
    guarded_max_discount: float
    guard_score_floor: float
    regional_threshold: float
    global_threshold: float
    # A-guarda (v11): additionally require phyloP100 conservation >= this to guard a variant.
    # 0.0 = off (identical to Pedro's molecular-only guard). ~0.5 separates founder P/LP
    # (phyloP ~1.0-1.4) from the common benigns the v11 molecular head over-scores (phyloP ~0.05).
    conservation_guard_threshold: float = 0.0


@dataclass(frozen=True)
class ConstraintTargets:
    m0_br_mcc: float
    m7_br_mcc: float
    m0_plp_recall: float
    m7_plp_recall: float
    v2_plp_recall: float
    v2_benign_spec: float
    m0_global_mcc: float
    v2_global_mcc: float
    m0_global_spec: float
    v2_br_mcc: float


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--holdout-dir",
        type=Path,
        default=Path("artifacts/clinvar_regional_eval/m5_v2_bounded_regional_holdout_eval_beatv10_v1_sagemaker/_extracted"),
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=Path("artifacts/clinvar_regional_eval/m5_v2_bounded_regional_eval_beatv10_v1_sagemaker/_extracted"),
    )
    parser.add_argument("--slice-dir", type=Path, default=Path("data/datasets/clinvar/regional_abraom/slices"))
    parser.add_argument(
        "--native-dir",
        type=Path,
        default=None,
        help="dir with {slice}.{split}.native_features.parquet (A-guarda: enables the phyloP "
             "conservation gate on the molecular guard). Omit to keep the original v11 molecular-only guard.",
    )
    parser.add_argument("--model-root", type=Path, default=Path("artifacts/clinvar_regional_eval"))
    parser.add_argument(
        "--m5-v2-config",
        type=Path,
        default=Path("artifacts/clinvar_regional_eval/m5_v2_calibrated_holdout_tuned/selected_config.json"),
    )
    parser.add_argument(
        "--baseline-csv",
        type=Path,
        default=Path("artifacts/clinvar_regional_eval/m5_v2_calibrated_holdout_tuned/m0_m4_m5_m6_m5cal_m6cal_m7_m5v2cal_summary.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/clinvar_regional_eval/m5_v3_safety_calibrated"),
    )
    parser.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    parser.add_argument("--control-seeds", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260629)
    parser.add_argument("--model-name", default="M5_v3_safety")
    return parser.parse_args(argv)


def logit(values: pd.Series | np.ndarray | float) -> np.ndarray:
    probs = np.clip(np.asarray(values, dtype=np.float64), EPS, 1.0 - EPS)
    return np.log(probs / (1.0 - probs))


def sigmoid(values: pd.Series | np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(values, dtype=np.float64), -60.0, 60.0)))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_threshold(model_dir: Path) -> float:
    path = model_dir / "regional_eval_summary.json"
    if not path.is_file():
        return 0.5
    return float(read_json(path).get("threshold", 0.5))


def af_bin(values: pd.Series) -> pd.Series:
    numeric = values.astype(float)
    output = pd.Series("absent", index=values.index, dtype=object)
    present = numeric.notna()
    output.loc[present & (numeric <= 1e-4)] = "<=1e-4"
    output.loc[present & (numeric > 1e-4) & (numeric <= 1e-3)] = "1e-4..1e-3"
    output.loc[present & (numeric > 1e-3) & (numeric <= 1e-2)] = "1e-3..1e-2"
    output.loc[present & (numeric > 1e-2)] = ">1e-2"
    return output


def load_metadata(slice_dir: Path, dataset: str, split: str) -> pd.DataFrame:
    path = slice_dir / f"{dataset}.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path).reset_index().rename(columns={"index": "original_index"})
    split_mask = frame["split_within_gene"].astype(str).str.lower().eq(split.lower())
    frame = frame.loc[split_mask].copy()
    frame["GeneSymbol"] = frame["GeneSymbol"].fillna("").astype(str).replace("", "unknown")
    frame["Chromosome"] = frame["Chromosome"].fillna("unknown").astype(str)
    frame["af_abraom_bin"] = af_bin(frame["af_abraom"]) if "af_abraom" in frame.columns else "unknown"
    keep = [
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
        "af_abraom_bin",
        "specificity",
        "specificity_bin",
    ]
    return frame[[column for column in keep if column in frame.columns]].copy()


def read_m5_predictions(
    root: Path, slice_dir: Path, dataset: str, split: str, native_dir: Path | None = None
) -> pd.DataFrame:
    path = root / f"{dataset}.{split}.predictions.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    predictions = pd.read_parquet(path).copy()
    required = {"dataset", "split", "original_index", "label", "probability", "molecular_probability", "regional_discount"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
    metadata = load_metadata(slice_dir, dataset, split)
    merged = predictions.merge(metadata, on="original_index", how="left", validate="one_to_one")
    if native_dir is not None:
        # A-guarda: attach the v11 native conservation/missense-severity for the guard
        # (produced by scripts/extract_native_pathogenicity_features.py).
        native_path = native_dir / f"{dataset}.{split}.native_features.parquet"
        if native_path.is_file():
            native = pd.read_parquet(native_path)
            keep = ["original_index"] + [c for c in ("phylo100", "missense_severity") if c in native.columns]
            merged = merged.merge(native[keep], on="original_index", how="left")
    return merged.reset_index(drop=True)


def read_raw_model_predictions(model_dir: Path, dataset: str, split: str) -> pd.DataFrame:
    path = model_dir / f"{dataset}.{split}.predictions.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path).copy()
    threshold = read_threshold(model_dir)
    frame["score_used"] = frame["probability"].astype(float)
    frame["threshold_used"] = threshold
    frame["prediction_used"] = (frame["score_used"] >= threshold).astype(int)
    return frame[["dataset", "split", "original_index", "label", "score_used", "threshold_used", "prediction_used"]]


def load_m5_split(
    root: Path, slice_dir: Path, datasets: Iterable[str], split: str, native_dir: Path | None = None
) -> dict[str, pd.DataFrame]:
    return {dataset: read_m5_predictions(root, slice_dir, dataset, split, native_dir) for dataset in datasets}


def _guard_mask(predictions: pd.DataFrame, config: SafetyConfig) -> np.ndarray:
    """A-guarda: which variants the frequency discount must NOT erase. Baseline = high molecular
    probability (Pedro's v10 signal). On v11 that alone over-protects the common benigns the
    stronger molecular head over-scores; when ``conservation_guard_threshold > 0`` we ALSO require
    the v11 native phyloP100 conservation >= it -- a founder P/LP is conserved (~1.0-1.4), a common
    benign is not (~0.05), and conservation is orthogonal to the ABRAOM frequency. Falls back to
    molecular-only if phylo100 is absent (so the script still runs without native features)."""
    if config.conservation_guard_threshold > 0.0 and "phylo100" in predictions.columns:
        # Conservation REPLACES the molecular gate (it does NOT and-with it). Measured on v11:
        # the founder P/LP we must protect have LOW molecular probability (median 0.374) -- only
        # 6/187 clear the 0.65 gate -- so and-ing it blocked 97% of them. phyloP is the signal
        # they do carry (though weakly: medians overlap, so only high thresholds discriminate).
        conservation = np.nan_to_num(predictions["phylo100"].to_numpy(dtype=np.float64), nan=-1e9)
        return conservation >= config.conservation_guard_threshold
    molecular = predictions["molecular_probability"].to_numpy(dtype=np.float64)
    return molecular >= config.molecular_guard_threshold


def apply_safety_config(predictions: pd.DataFrame, config: SafetyConfig, dataset: str) -> pd.DataFrame:
    output = predictions.copy()
    molecular = output["molecular_probability"].to_numpy(dtype=np.float64)
    raw_discount = output["regional_discount"].to_numpy(dtype=np.float64) * config.discount_scale
    base_capped_discount = np.minimum(raw_discount, config.max_discount)
    guard_mask = _guard_mask(output, config)
    guarded_discount = np.where(
        guard_mask,
        np.minimum(base_capped_discount, config.guarded_max_discount),
        base_capped_discount,
    )
    regional_score = sigmoid(logit(molecular) - guarded_discount)
    safety_floor = np.where(guard_mask, np.maximum(regional_score, config.guard_score_floor), regional_score)
    global_score = molecular
    if dataset in GLOBAL_DATASETS:
        score_used = global_score
        threshold = config.global_threshold
        decision_context = "global_molecular"
    else:
        score_used = safety_floor
        threshold = config.regional_threshold
        decision_context = "brazil_regional_safety"
    output["effective_regional_discount"] = guarded_discount
    output["safety_guard_triggered"] = guard_mask
    output["regional_score_calibrated"] = regional_score
    output["safety_floor_score"] = safety_floor
    output["global_score_calibrated"] = global_score
    output["score_used"] = score_used
    output["threshold_used"] = threshold
    output["decision_context"] = decision_context
    output["prediction_used"] = (score_used >= threshold).astype(int)
    return output


def scores_for_config(predictions: pd.DataFrame, config: SafetyConfig, dataset: str) -> tuple[np.ndarray, float]:
    molecular = predictions["molecular_probability"].to_numpy(dtype=np.float64)
    raw_discount = predictions["regional_discount"].to_numpy(dtype=np.float64) * config.discount_scale
    base_capped_discount = np.minimum(raw_discount, config.max_discount)
    guard_mask = _guard_mask(predictions, config)
    guarded_discount = np.where(
        guard_mask,
        np.minimum(base_capped_discount, config.guarded_max_discount),
        base_capped_discount,
    )
    regional_score = sigmoid(logit(molecular) - guarded_discount)
    safety_floor = np.where(guard_mask, np.maximum(regional_score, config.guard_score_floor), regional_score)
    if dataset in GLOBAL_DATASETS:
        return molecular, config.global_threshold
    return safety_floor, config.regional_threshold


def metric_row(
    model: str,
    dataset: str,
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    *,
    include_rank_metrics: bool = True,
) -> dict[str, Any]:
    metrics = classification_metrics(labels.astype(np.int64), scores.astype(np.float64), threshold)
    if include_rank_metrics and len(np.unique(labels)) > 1:
        auroc = binary_roc_auc(labels, scores)
        auprc = binary_auprc(labels, scores)
    else:
        auroc = float("nan")
        auprc = float("nan")
    return {
        "model": model,
        "dataset": dataset,
        "n": int(len(labels)),
        "auroc": auroc,
        "auprc": auprc,
        "mcc": metrics["mcc"],
        "recall": metrics["recall"],
        "specificity": metrics["specificity"],
        "threshold": metrics["threshold"],
        "tp": metrics["tp"],
        "tn": metrics["tn"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
    }


def metrics_for_config(
    predictions_by_dataset: dict[str, pd.DataFrame],
    config: SafetyConfig,
    *,
    model_name: str,
) -> tuple[list[dict[str, Any]], dict[str, pd.DataFrame]]:
    rows: list[dict[str, Any]] = []
    outputs: dict[str, pd.DataFrame] = {}
    for dataset, predictions in predictions_by_dataset.items():
        output = apply_safety_config(predictions, config, dataset)
        outputs[dataset] = output
        rows.append(
            metric_row(
                model_name,
                dataset,
                output["label"].to_numpy(dtype=np.int64),
                output["score_used"].to_numpy(dtype=np.float64),
                float(output["threshold_used"].iloc[0]),
            )
        )
    return rows, outputs


def metric_rows_for_config(
    predictions_by_dataset: dict[str, pd.DataFrame],
    config: SafetyConfig,
    *,
    model_name: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset, predictions in predictions_by_dataset.items():
        scores, threshold = scores_for_config(predictions, config, dataset)
        rows.append(
            metric_row(
                model_name,
                dataset,
                predictions["label"].to_numpy(dtype=np.int64),
                scores,
                float(threshold),
                include_rank_metrics=False,
            )
        )
    return rows


def value(rows: list[dict[str, Any]] | pd.DataFrame, dataset: str, metric: str) -> float:
    if isinstance(rows, pd.DataFrame):
        match = rows.loc[rows["dataset"].eq(dataset)]
        return float(match.iloc[0][metric]) if not match.empty else float("nan")
    for row in rows:
        if row["dataset"] == dataset:
            return float(row[metric])
    return float("nan")


def baseline_metrics_for_split(
    model_name: str,
    model_dir: Path,
    datasets: Iterable[str],
    split: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        predictions = read_raw_model_predictions(model_dir, dataset, split)
        rows.append(
            metric_row(
                model_name,
                dataset,
                predictions["label"].to_numpy(dtype=np.int64),
                predictions["score_used"].to_numpy(dtype=np.float64),
                float(predictions["threshold_used"].iloc[0]),
            )
        )
    return pd.DataFrame(rows)


def config_from_v2(path: Path) -> SafetyConfig:
    payload = read_json(path)
    regional_threshold = float(payload["regional_threshold"])
    return SafetyConfig(
        discount_scale=float(payload["discount_scale"]),
        max_discount=float(payload["max_discount"]),
        molecular_guard_threshold=1.01,
        guarded_max_discount=float(payload["max_discount"]),
        guard_score_floor=regional_threshold,
        regional_threshold=regional_threshold,
        global_threshold=float(payload["global_threshold"]),
    )


def choose_global_threshold(predictions_by_dataset: dict[str, pd.DataFrame]) -> float:
    best_threshold = 0.5
    best_score = -float("inf")
    for threshold in np.round(np.linspace(0.20, 0.85, 131), 3):
        rows: list[dict[str, Any]] = []
        for dataset in sorted(GLOBAL_DATASETS):
            predictions = predictions_by_dataset[dataset]
            rows.append(
                metric_row(
                    "global_threshold_candidate",
                    dataset,
                    predictions["label"].to_numpy(dtype=np.int64),
                    predictions["molecular_probability"].to_numpy(dtype=np.float64),
                    float(threshold),
                    include_rank_metrics=False,
                )
            )
        score = (
            1.5 * value(rows, "global_nonbr_no_abraom", "mcc")
            + 0.75 * value(rows, "global_nonbr_no_abraom", "specificity")
            + 0.5 * value(rows, "nonbr_only", "mcc")
        )
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def candidate_configs(global_threshold: float) -> list[SafetyConfig]:
    configs: list[SafetyConfig] = []
    discount_scales = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]
    max_discounts = [0.0, 0.25, 0.5, 0.75, 1.0]
    # Shrunk from 5 to 2 (the A-guarda conservation gate is now the primary founder-vs-benign
    # signal); 1.01 = molecular guard effectively off, so conservation alone can gate.
    guard_thresholds = [0.65, 1.01]
    guarded_caps = [0.0, 0.10, 0.25, 0.50]
    # A-guarda phyloP100 gate (replaces the molecular gate when > 0; 0.0 = old molecular-only v3).
    # Range set from the MEASURED founder-vs-benign separation on holdout: the medians overlap
    # (founders 0.357, benigns -0.183), so only HIGH thresholds discriminate -- enrichment climbs
    # to 5.4x at 3.0, where only 5% of common benigns fire (vs 16% at 0.5). The 0-1.0 window tried
    # first was the wrong range.
    conservation_guards = [0.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    regional_thresholds = np.round(np.linspace(0.25, 0.55, 31), 3)
    for discount_scale in discount_scales:
        for max_discount in max_discounts:
            for guard_threshold in guard_thresholds:
                for guarded_cap in guarded_caps:
                    if guard_threshold > 1.0 and guarded_cap != 0.50:
                        continue
                    for conservation_guard in conservation_guards:
                        for regional_threshold in regional_thresholds:
                            configs.append(
                                SafetyConfig(
                                    discount_scale=float(discount_scale),
                                    max_discount=float(max_discount),
                                    molecular_guard_threshold=float(guard_threshold),
                                    guarded_max_discount=float(min(guarded_cap, max_discount)),
                                    guard_score_floor=float(regional_threshold),
                                    regional_threshold=float(regional_threshold),
                                    global_threshold=float(global_threshold),
                                    conservation_guard_threshold=float(conservation_guard),
                                )
                            )
    return configs


def hard_constraints(
    rows: list[dict[str, Any]],
    *,
    targets: ConstraintTargets,
) -> dict[str, bool]:
    br_mcc = value(rows, "br_only", "mcc")
    benign_spec = value(rows, "abraom_common_benign", "specificity")
    plp_recall = value(rows, "abraom_pathogenic_present", "recall")
    global_mcc = value(rows, "global_nonbr_no_abraom", "mcc")
    global_spec = value(rows, "global_nonbr_no_abraom", "specificity")

    return {
        "br_mcc_ge_m0_and_m7": br_mcc >= max(targets.m0_br_mcc, targets.m7_br_mcc),
        "benign_specificity_ge_0_95": benign_spec >= 0.95,
        "benign_specificity_near_v2": benign_spec >= (targets.v2_benign_spec - 0.005),
        "plp_recall_ge_v2": plp_recall >= targets.v2_plp_recall,
        "plp_recall_ge_m0_minus_0_02": plp_recall >= (targets.m0_plp_recall - 0.02),
        "plp_recall_ge_m7": plp_recall >= targets.m7_plp_recall,
        "global_mcc_ge_m0_minus_0_05": global_mcc >= (targets.m0_global_mcc - 0.05),
        "global_mcc_near_v2": global_mcc >= (targets.v2_global_mcc - 0.02),
        "global_specificity_ge_m0_minus_0_05": global_spec >= (targets.m0_global_spec - 0.05),
    }


def score_config(
    rows: list[dict[str, Any]],
    *,
    constraints: dict[str, bool],
    targets: ConstraintTargets,
) -> float:
    br_mcc = value(rows, "br_only", "mcc")
    br_spec = value(rows, "br_only", "specificity")
    benign_spec = value(rows, "abraom_common_benign", "specificity")
    plp_recall = value(rows, "abraom_pathogenic_present", "recall")
    plp_common_recall = value(rows, "abraom_pathogenic_common", "recall")
    global_mcc = value(rows, "global_nonbr_no_abraom", "mcc")
    global_spec = value(rows, "global_nonbr_no_abraom", "specificity")

    reward = (
        5.0 * plp_recall
        + 1.5 * plp_common_recall
        + 1.25 * br_mcc
        + 0.50 * br_spec
        + 1.50 * benign_spec
        + 0.75 * global_mcc
        + 0.50 * global_spec
    )
    reward += 1.25 * max(0.0, plp_recall - targets.v2_plp_recall)
    reward += 0.50 * max(0.0, br_mcc - targets.v2_br_mcc)
    penalty = 0.0
    penalty += 40.0 * max(0.0, 0.95 - benign_spec)
    penalty += 25.0 * max(0.0, (targets.v2_benign_spec - 0.005) - benign_spec)
    penalty += 30.0 * max(0.0, targets.v2_plp_recall - plp_recall)
    penalty += 12.0 * max(0.0, (targets.v2_global_mcc - 0.02) - global_mcc)
    penalty += 2.0 * sum(1 for passed in constraints.values() if not passed)
    return reward - penalty


def tune_config(
    holdout: dict[str, pd.DataFrame],
    *,
    m0_holdout: pd.DataFrame,
    m7_holdout: pd.DataFrame,
    m5_v2_holdout: pd.DataFrame,
    model_name: str,
) -> tuple[SafetyConfig, pd.DataFrame]:
    global_threshold = choose_global_threshold(holdout)
    targets = ConstraintTargets(
        m0_br_mcc=value(m0_holdout, "br_only", "mcc"),
        m7_br_mcc=value(m7_holdout, "br_only", "mcc"),
        m0_plp_recall=value(m0_holdout, "abraom_pathogenic_present", "recall"),
        m7_plp_recall=value(m7_holdout, "abraom_pathogenic_present", "recall"),
        v2_plp_recall=value(m5_v2_holdout, "abraom_pathogenic_present", "recall"),
        v2_benign_spec=value(m5_v2_holdout, "abraom_common_benign", "specificity"),
        m0_global_mcc=value(m0_holdout, "global_nonbr_no_abraom", "mcc"),
        v2_global_mcc=value(m5_v2_holdout, "global_nonbr_no_abraom", "mcc"),
        m0_global_spec=value(m0_holdout, "global_nonbr_no_abraom", "specificity"),
        v2_br_mcc=value(m5_v2_holdout, "br_only", "mcc"),
    )
    records: list[dict[str, Any]] = []
    best_config: SafetyConfig | None = None
    best_score = -float("inf")
    best_valid_config: SafetyConfig | None = None
    best_valid_score = -float("inf")
    for config in candidate_configs(global_threshold):
        rows = metric_rows_for_config(holdout, config, model_name=model_name)
        constraints = hard_constraints(rows, targets=targets)
        score = score_config(rows, constraints=constraints, targets=targets)
        passes = all(constraints.values())
        record = {
            "score": score,
            "passes_hard_constraints": passes,
            **{f"constraint_{name}": passed for name, passed in constraints.items()},
            **asdict(config),
            "br_only_mcc": value(rows, "br_only", "mcc"),
            "br_only_recall": value(rows, "br_only", "recall"),
            "br_only_specificity": value(rows, "br_only", "specificity"),
            "abraom_common_benign_specificity": value(rows, "abraom_common_benign", "specificity"),
            "abraom_pathogenic_present_recall": value(rows, "abraom_pathogenic_present", "recall"),
            "abraom_pathogenic_common_recall": value(rows, "abraom_pathogenic_common", "recall"),
            "global_nonbr_no_abraom_mcc": value(rows, "global_nonbr_no_abraom", "mcc"),
            "global_nonbr_no_abraom_specificity": value(rows, "global_nonbr_no_abraom", "specificity"),
        }
        records.append(record)
        if score > best_score:
            best_score = score
            best_config = config
        if passes and score > best_valid_score:
            best_valid_score = score
            best_valid_config = config
    if best_valid_config is not None:
        selected = best_valid_config
    elif best_config is not None:
        selected = best_config
    else:
        raise RuntimeError("No M5_v3 safety configs were evaluated.")
    tuning = pd.DataFrame(records).sort_values(
        ["passes_hard_constraints", "score"],
        ascending=[False, False],
    )
    return selected, tuning


def apply_and_write(
    predictions_by_dataset: dict[str, pd.DataFrame],
    config: SafetyConfig,
    *,
    output_dir: Path,
    model_name: str,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rows, outputs = metrics_for_config(predictions_by_dataset, config, model_name=model_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    for dataset, frame in outputs.items():
        frame.to_parquet(output_dir / f"{dataset}.test.predictions.parquet", index=False)
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "m5_v3_safety_regional_test_summary.csv", index=False)
    summary.to_parquet(output_dir / "m5_v3_safety_regional_test_summary.parquet", index=False)
    return summary, outputs


def grouped_permutation(
    values: pd.Series,
    groups: pd.Series,
    *,
    rng: np.random.Generator,
) -> tuple[pd.Series, float]:
    output = values.copy()
    changed = np.zeros(len(values), dtype=bool)
    for _, index in groups.groupby(groups, dropna=False).groups.items():
        idx = np.asarray(index)
        if len(idx) < 2:
            continue
        current = values.iloc[idx].to_numpy(copy=True)
        permuted = current[rng.permutation(len(current))]
        output.iloc[idx] = permuted
        changed[idx] = permuted != current
    return output, float(changed.mean())


def scramble_discounts(
    frame: pd.DataFrame,
    *,
    mode: str,
    seed: int,
) -> tuple[pd.DataFrame, float]:
    rng = np.random.default_rng(seed)
    output = frame.copy()
    values = output["regional_discount"].astype(float)
    if mode == "global":
        permuted = values.to_numpy(copy=True)
        rng.shuffle(permuted)
        output["regional_discount"] = permuted
        changed_fraction = float(np.mean(permuted != values.to_numpy()))
    elif mode == "within_gene":
        output["regional_discount"], changed_fraction = grouped_permutation(values, output["GeneSymbol"], rng=rng)
    elif mode == "within_af_bin":
        output["regional_discount"], changed_fraction = grouped_permutation(values, output["af_abraom_bin"], rng=rng)
    elif mode == "within_chromosome":
        output["regional_discount"], changed_fraction = grouped_permutation(values, output["Chromosome"], rng=rng)
    else:
        raise ValueError(f"Unsupported scramble mode: {mode}")
    return output, changed_fraction


def run_negative_controls(
    test: dict[str, pd.DataFrame],
    config: SafetyConfig,
    *,
    model_name: str,
    seeds: int,
    seed_base: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    control_rows: list[dict[str, Any]] = []
    modes = ["global", "within_gene", "within_af_bin", "within_chromosome"]
    for mode in modes:
        for seed_index in range(seeds):
            seed = seed_base + seed_index + 1000 * modes.index(mode)
            scrambled_by_dataset: dict[str, pd.DataFrame] = {}
            changed_by_dataset: dict[str, float] = {}
            for dataset, frame in test.items():
                scrambled, changed = scramble_discounts(frame, mode=mode, seed=seed)
                scrambled_by_dataset[dataset] = scrambled
                changed_by_dataset[dataset] = changed
            rows, _ = metrics_for_config(scrambled_by_dataset, config, model_name=f"{model_name}_{mode}_control")
            for row in rows:
                control_rows.append(
                    {
                        "control_mode": mode,
                        "seed": seed,
                        "changed_discount_fraction": changed_by_dataset[row["dataset"]],
                        **row,
                    }
                )
    controls = pd.DataFrame(control_rows)
    real_rows, _ = metrics_for_config(test, config, model_name=model_name)
    real = pd.DataFrame(real_rows)
    comparison_rows: list[dict[str, Any]] = []
    for dataset, metric in KEY_CONTROL_METRICS:
        real_value = value(real, dataset, metric)
        for mode in modes:
            subset = controls.loc[controls["dataset"].eq(dataset) & controls["control_mode"].eq(mode)]
            values = subset[metric].astype(float).to_numpy()
            comparison_rows.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "control_mode": mode,
                    "real_value": real_value,
                    "control_mean": float(np.nanmean(values)),
                    "control_p05": float(np.nanpercentile(values, 5)),
                    "control_p50": float(np.nanpercentile(values, 50)),
                    "control_p95": float(np.nanpercentile(values, 95)),
                    "real_minus_control_mean": float(real_value - np.nanmean(values)),
                    "empirical_p_control_ge_real": float((np.sum(values >= real_value) + 1) / (len(values) + 1)),
                    "mean_changed_discount_fraction": float(subset["changed_discount_fraction"].mean()),
                    "n_control_runs": int(len(values)),
                }
            )
    return controls, pd.DataFrame(comparison_rows)


def variant_columns(frame: pd.DataFrame) -> list[str]:
    columns = [
        "variant_key",
        "source_variant_id",
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
        "molecular_probability",
        "regional_discount",
        "effective_regional_discount",
        "safety_guard_triggered",
        "regional_score_calibrated",
        "safety_floor_score",
        "score_used",
        "threshold_used",
        "prediction_used",
    ]
    return [column for column in columns if column in frame.columns]


def build_sentinel_audit(
    test: dict[str, pd.DataFrame],
    m5_v2_outputs: dict[str, pd.DataFrame],
    v3_outputs: dict[str, pd.DataFrame],
    output_dir: Path,
) -> pd.DataFrame:
    audit_dir = output_dir / "sentinel_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    plp = v3_outputs["abraom_pathogenic_present"].copy()
    v2_plp = m5_v2_outputs["abraom_pathogenic_present"][
        ["original_index", "prediction_used", "score_used", "threshold_used"]
    ].rename(
        columns={
            "prediction_used": "m5_v2_prediction_used",
            "score_used": "m5_v2_score_used",
            "threshold_used": "m5_v2_threshold_used",
        }
    )
    plp = plp.merge(v2_plp, on="original_index", how="left", validate="one_to_one")
    dangerous = plp.loc[plp["m5_v2_prediction_used"].eq(1) & plp["prediction_used"].eq(0)].copy()
    rescued_plp = plp.loc[plp["m5_v2_prediction_used"].eq(0) & plp["prediction_used"].eq(1)].copy()
    dangerous.to_csv(audit_dir / "M5_v3_safety.dangerous_false_benign_regressions_vs_M5_v2.csv", index=False)
    rescued_plp.to_csv(audit_dir / "M5_v3_safety.rescued_plp_vs_M5_v2.csv", index=False)
    rows.extend(
        [
            {
                "audit": "plp_dangerous_regressions_vs_m5_v2",
                "dataset": "abraom_pathogenic_present",
                "n": int(len(dangerous)),
                "path": str(audit_dir / "M5_v3_safety.dangerous_false_benign_regressions_vs_M5_v2.csv"),
            },
            {
                "audit": "plp_rescued_vs_m5_v2",
                "dataset": "abraom_pathogenic_present",
                "n": int(len(rescued_plp)),
                "path": str(audit_dir / "M5_v3_safety.rescued_plp_vs_M5_v2.csv"),
            },
        ]
    )

    benign = v3_outputs["abraom_common_benign"].copy()
    v2_benign = m5_v2_outputs["abraom_common_benign"][
        ["original_index", "prediction_used", "score_used", "threshold_used"]
    ].rename(
        columns={
            "prediction_used": "m5_v2_prediction_used",
            "score_used": "m5_v2_score_used",
            "threshold_used": "m5_v2_threshold_used",
        }
    )
    benign = benign.merge(v2_benign, on="original_index", how="left", validate="one_to_one")
    resolved_fp = benign.loc[benign["m5_v2_prediction_used"].eq(1) & benign["prediction_used"].eq(0)].copy()
    new_fp = benign.loc[benign["m5_v2_prediction_used"].eq(0) & benign["prediction_used"].eq(1)].copy()
    resolved_fp.to_csv(audit_dir / "M5_v3_safety.resolved_common_benign_fp_vs_M5_v2.csv", index=False)
    new_fp.to_csv(audit_dir / "M5_v3_safety.new_common_benign_fp_vs_M5_v2.csv", index=False)
    rows.extend(
        [
            {
                "audit": "common_benign_fp_resolved_vs_m5_v2",
                "dataset": "abraom_common_benign",
                "n": int(len(resolved_fp)),
                "path": str(audit_dir / "M5_v3_safety.resolved_common_benign_fp_vs_M5_v2.csv"),
            },
            {
                "audit": "common_benign_new_fp_vs_m5_v2",
                "dataset": "abraom_common_benign",
                "n": int(len(new_fp)),
                "path": str(audit_dir / "M5_v3_safety.new_common_benign_fp_vs_M5_v2.csv"),
            },
        ]
    )

    for name, frame in [
        ("dangerous_false_benign_regressions_vs_M5_v2", dangerous),
        ("rescued_plp_vs_M5_v2", rescued_plp),
        ("resolved_common_benign_fp_vs_M5_v2", resolved_fp),
        ("new_common_benign_fp_vs_M5_v2", new_fp),
    ]:
        if not frame.empty:
            frame[variant_columns(frame)].to_csv(audit_dir / f"M5_v3_safety.{name}.compact.csv", index=False)

    audit = pd.DataFrame(rows)
    audit.to_csv(output_dir / "m5_v3_safety_sentinel_audit.csv", index=False)
    return audit


def combine_baseline_summary(
    baseline_csv: Path,
    v3_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    if not baseline_csv.is_file():
        return
    baseline = pd.read_csv(baseline_csv)
    shared = [column for column in baseline.columns if column in v3_summary.columns]
    combined = pd.concat([baseline, v3_summary[shared]], ignore_index=True)
    combined.to_csv(output_dir / "m0_m4_m5_m6_m5cal_m6cal_m7_m5v2cal_m5v3_summary.csv", index=False)
    combined.to_parquet(output_dir / "m0_m4_m5_m6_m5cal_m6cal_m7_m5v2cal_m5v3_summary.parquet", index=False)


def fmt(value_: Any, digits: int = 3) -> str:
    try:
        if pd.isna(value_):
            return "NA"
        return f"{float(value_):.{digits}f}"
    except Exception:
        return str(value_)


def report_decision(v3_summary: pd.DataFrame, v2_summary: pd.DataFrame, controls: pd.DataFrame, audit: pd.DataFrame) -> str:
    plp_delta = value(v3_summary, "abraom_pathogenic_present", "recall") - value(
        v2_summary,
        "abraom_pathogenic_present",
        "recall",
    )
    benign_delta = value(v3_summary, "abraom_common_benign", "specificity") - value(
        v2_summary,
        "abraom_common_benign",
        "specificity",
    )
    global_delta = value(v3_summary, "global_nonbr_no_abraom", "mcc") - value(
        v2_summary,
        "global_nonbr_no_abraom",
        "mcc",
    )
    dangerous = int(audit.loc[audit["audit"].eq("plp_dangerous_regressions_vs_m5_v2"), "n"].iloc[0])
    regional_control = controls.loc[
        controls["dataset"].eq("br_only")
        & controls["metric"].eq("mcc")
        & controls["control_mode"].isin(["within_gene", "within_af_bin", "within_chromosome"])
    ]
    weak_falsification = bool((regional_control["empirical_p_control_ge_real"] > 0.10).any())
    if dangerous > 0:
        return "do_not_advance: M5_v3 creates dangerous P/LP regressions versus M5_v2."
    if plp_delta >= 0 and benign_delta >= -0.005 and global_delta >= -0.02 and not weak_falsification:
        return "advance_candidate: M5_v3 improves safety without failing falsification controls."
    if plp_delta >= 0 and benign_delta >= -0.005 and global_delta >= -0.02:
        return "conditional_candidate: M5_v3 is safe versus M5_v2 but regional specificity is not fully falsified."
    return "hold_current_lead: M5_v2 remains the safer lead."


def write_report(
    output_dir: Path,
    *,
    selected_config: SafetyConfig,
    holdout_tuning: pd.DataFrame,
    v3_summary: pd.DataFrame,
    v2_summary: pd.DataFrame,
    m0_summary: pd.DataFrame,
    m7_summary: pd.DataFrame,
    controls: pd.DataFrame,
    audit: pd.DataFrame,
    decision: str,
) -> None:
    top_holdout = holdout_tuning.head(5)
    lines = [
        "# M5_v3 Safety Calibration",
        "",
        f"Generated at UTC: `{datetime.now(UTC).isoformat()}`",
        "",
        "## Bottom Line",
        "",
        f"Decision: `{decision}`.",
        "",
        "`M5_v3_safety` is a locked holdout-selected safety calibration over M5_v2 raw outputs. "
        "It adds a molecular guard so strong molecular evidence cannot be fully erased by an ABRAOM frequency discount.",
        "",
        "## Selected Config",
        "",
        "```json",
        json.dumps(asdict(selected_config), indent=2),
        "```",
        "",
        "## Test Metrics",
        "",
        "| Dataset | M0 | M7 | M5_v2 | M5_v3 | Metric |",
        "|---|---:|---:|---:|---:|---|",
    ]
    metric_map = [
        ("br_only", "mcc"),
        ("abraom_common_benign", "specificity"),
        ("abraom_pathogenic_present", "recall"),
        ("global_nonbr_no_abraom", "mcc"),
        ("global_nonbr_no_abraom", "specificity"),
    ]
    for dataset, metric in metric_map:
        lines.append(
            f"| `{dataset}` | {fmt(value(m0_summary, dataset, metric))} | {fmt(value(m7_summary, dataset, metric))} | "
            f"{fmt(value(v2_summary, dataset, metric))} | {fmt(value(v3_summary, dataset, metric))} | `{metric}` |"
        )
    lines.extend(
        [
            "",
            "## Sentinel Audit",
            "",
            "| Audit | Dataset | n |",
            "|---|---|---:|",
        ]
    )
    for row in audit.itertuples(index=False):
        lines.append(f"| `{row.audit}` | `{row.dataset}` | {int(row.n)} |")
    lines.extend(
        [
            "",
            "## Negative Controls",
            "",
            "| Dataset | Metric | Control | Real | Control mean | P95 | Empirical P(control >= real) | Changed discount |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in controls.itertuples(index=False):
        lines.append(
            f"| `{row.dataset}` | `{row.metric}` | `{row.control_mode}` | {fmt(row.real_value)} | "
            f"{fmt(row.control_mean)} | {fmt(row.control_p95)} | {fmt(row.empirical_p_control_ge_real, 4)} | "
            f"{fmt(row.mean_changed_discount_fraction)} |"
        )
    lines.extend(
        [
            "",
            "## Top Holdout Candidates",
            "",
            "| Rank | Score | Pass | discount_scale | max_discount | guard_threshold | guarded_cap | threshold | br MCC | benign spec | P/LP recall | global MCC |",
            "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for rank, row in enumerate(top_holdout.itertuples(index=False), start=1):
        lines.append(
            f"| {rank} | {fmt(row.score)} | `{bool(row.passes_hard_constraints)}` | {fmt(row.discount_scale)} | "
            f"{fmt(row.max_discount)} | {fmt(row.molecular_guard_threshold)} | {fmt(row.guarded_max_discount)} | "
            f"{fmt(row.regional_threshold)} | {fmt(row.br_only_mcc)} | {fmt(row.abraom_common_benign_specificity)} | "
            f"{fmt(row.abraom_pathogenic_present_recall)} | {fmt(row.global_nonbr_no_abraom_mcc)} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- If `empirical_p_control_ge_real` is high, the real ABRAOM discount is not clearly better than that scrambled control.",
            "- The test sentinel audit is not used for selection; it only audits the locked holdout-selected configuration.",
            "- A clinically relevant next step still requires external Brazilian P/LP curation; this artifact is scientific validation, not clinical validation.",
            "",
            "## Key Artifacts",
            "",
            "- `selected_config.json`",
            "- `holdout_tuning_results.csv`",
            "- `m5_v3_safety_regional_test_summary.csv`",
            "- `m5_v3_negative_control_comparison.csv`",
            "- `m5_v3_negative_control_runs.csv`",
            "- `m5_v3_safety_sentinel_audit.csv`",
            "- `sentinel_audit/`",
        ]
    )
    (output_dir / "M5_V3_SAFETY_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    holdout = load_m5_split(args.holdout_dir, args.slice_dir, args.datasets, "holdout", args.native_dir)
    test = load_m5_split(args.test_dir, args.slice_dir, args.datasets, "test", args.native_dir)

    m0_dir = args.model_root / "m0_nonbr_beatv10_v1_sagemaker"
    m7_dir = args.model_root / "m7_dynamic_scrambled_nonbr_beatv10_v1_sagemaker"
    m0_holdout = baseline_metrics_for_split("M0", m0_dir, args.datasets, "holdout")
    m7_holdout = baseline_metrics_for_split("M7_dynamic_scrambled", m7_dir, args.datasets, "holdout")
    m0_test = baseline_metrics_for_split("M0", m0_dir, args.datasets, "test")
    m7_test = baseline_metrics_for_split("M7_dynamic_scrambled", m7_dir, args.datasets, "test")

    v2_config = config_from_v2(args.m5_v2_config)
    v2_holdout_rows, _ = metrics_for_config(holdout, v2_config, model_name="M5_v2_calibrated")
    v2_holdout = pd.DataFrame(v2_holdout_rows)
    v2_test_rows, v2_test_outputs = metrics_for_config(test, v2_config, model_name="M5_v2_calibrated")
    v2_test = pd.DataFrame(v2_test_rows)

    selected_config, tuning = tune_config(
        holdout,
        m0_holdout=m0_holdout,
        m7_holdout=m7_holdout,
        m5_v2_holdout=v2_holdout,
        model_name=args.model_name,
    )
    tuning.to_csv(args.output_dir / "holdout_tuning_results.csv", index=False)
    tuning.to_parquet(args.output_dir / "holdout_tuning_results.parquet", index=False)
    (args.output_dir / "selected_config.json").write_text(
        json.dumps(asdict(selected_config), indent=2),
        encoding="utf-8",
    )

    v3_summary, v3_outputs = apply_and_write(
        test,
        selected_config,
        output_dir=args.output_dir,
        model_name=args.model_name,
    )
    controls, control_comparison = run_negative_controls(
        test,
        selected_config,
        model_name=args.model_name,
        seeds=args.control_seeds,
        seed_base=args.seed,
    )
    controls.to_csv(args.output_dir / "m5_v3_negative_control_runs.csv", index=False)
    controls.to_parquet(args.output_dir / "m5_v3_negative_control_runs.parquet", index=False)
    control_comparison.to_csv(args.output_dir / "m5_v3_negative_control_comparison.csv", index=False)
    control_comparison.to_parquet(args.output_dir / "m5_v3_negative_control_comparison.parquet", index=False)

    audit = build_sentinel_audit(test, v2_test_outputs, v3_outputs, args.output_dir)
    combine_baseline_summary(args.baseline_csv, v3_summary, args.output_dir)
    decision = report_decision(v3_summary, v2_test, control_comparison, audit)

    m0_test.to_csv(args.output_dir / "m0_test_metrics_recomputed.csv", index=False)
    m7_test.to_csv(args.output_dir / "m7_test_metrics_recomputed.csv", index=False)
    v2_test.to_csv(args.output_dir / "m5_v2_recomputed_test_metrics.csv", index=False)
    write_report(
        args.output_dir,
        selected_config=selected_config,
        holdout_tuning=tuning,
        v3_summary=v3_summary,
        v2_summary=v2_test,
        m0_summary=m0_test,
        m7_summary=m7_test,
        controls=control_comparison,
        audit=audit,
        decision=decision,
    )
    summary = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "output_dir": str(args.output_dir),
        "decision": decision,
        "selected_config": asdict(selected_config),
        "test_metrics": {
            f"{dataset}.{metric}": value(v3_summary, dataset, metric)
            for dataset, metric in KEY_CONTROL_METRICS
        },
        "sentinel_audit": audit[["audit", "n"]].to_dict(orient="records"),
    }
    (args.output_dir / "m5_v3_safety_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
