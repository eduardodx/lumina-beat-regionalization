"""Pure NumPy classification metrics for ClinVar evaluation."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0 else 0.0


def _trapezoid_integral(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    deltas = x[1:] - x[:-1]
    heights = (y[1:] + y[:-1]) * 0.5
    return float(np.sum(deltas * heights))


def binary_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    labels = y_true.astype(np.int64)
    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())
    if positives == 0 or negatives == 0:
        return float("nan")

    order = np.argsort(-y_score)
    ranked = labels[order]
    tps = np.cumsum(ranked == 1)
    fps = np.cumsum(ranked == 0)
    tpr = np.concatenate(([0.0], tps / positives, [1.0]))
    fpr = np.concatenate(([0.0], fps / negatives, [1.0]))
    return _trapezoid_integral(tpr, fpr)


def binary_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    labels = y_true.astype(np.int64)
    positives = int((labels == 1).sum())
    if positives == 0:
        return float("nan")

    order = np.argsort(-y_score)
    ranked = labels[order]
    tps = np.cumsum(ranked == 1)
    fps = np.cumsum(ranked == 0)
    precision = tps / np.maximum(tps + fps, 1)
    recall = tps / positives
    precision = np.concatenate(([1.0], precision))
    recall = np.concatenate(([0.0], recall))
    return _trapezoid_integral(precision, recall)


def binary_log_loss(y_true: np.ndarray, y_score: np.ndarray) -> float:
    probs = np.clip(y_score.astype(np.float64), a_min=1e-12, a_max=1.0 - 1e-12)
    labels = y_true.astype(np.float64)
    loss = -(labels * np.log(probs) + (1.0 - labels) * np.log(1.0 - probs))
    return float(np.mean(loss))


def brier_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    labels = y_true.astype(np.float64)
    probs = y_score.astype(np.float64)
    return float(np.mean((probs - labels) ** 2))


def classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Compute binary classification metrics at a given threshold."""
    labels = y_true.astype(np.int64)
    preds = (y_prob >= threshold).astype(np.int64)

    tp = int(((labels == 1) & (preds == 1)).sum())
    tn = int(((labels == 0) & (preds == 0)).sum())
    fp = int(((labels == 0) & (preds == 1)).sum())
    fn = int(((labels == 1) & (preds == 0)).sum())

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    f1 = _safe_div(2.0 * precision * recall, precision + recall) if precision + recall > 0 else 0.0
    accuracy = _safe_div(tp + tn, len(labels))
    balanced_accuracy = float(np.mean([recall, specificity]))

    mcc_num = tp * tn - fp * fn
    mcc_den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = _safe_div(mcc_num, mcc_den) if mcc_den > 0 else 0.0

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "mcc": float(mcc),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def optimize_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric: str = "mcc",
) -> tuple[float, dict[str, Any]]:
    """Find the threshold that maximises a target metric."""
    if metric not in {"mcc", "f1"}:
        raise ValueError(f"Unsupported threshold metric {metric!r}.")

    candidates = np.unique(np.clip(
        np.concatenate([np.asarray([0.0, 1.0], dtype=np.float64), y_score.astype(np.float64)]),
        a_min=0.0, a_max=1.0,
    ))

    best_threshold = float(candidates[0])
    best_metrics = classification_metrics(y_true, y_score, best_threshold)
    best_value = float(best_metrics[metric])

    for threshold in candidates[1:]:
        current = classification_metrics(y_true, y_score, float(threshold))
        current_value = float(current[metric])
        if current_value > best_value + 1e-12:
            best_threshold = float(threshold)
            best_metrics = current
            best_value = current_value
        elif math.isclose(current_value, best_value, rel_tol=0.0, abs_tol=1e-12):
            if abs(float(threshold) - 0.5) < abs(best_threshold - 0.5):
                best_threshold = float(threshold)
                best_metrics = current
                best_value = current_value

    return best_threshold, best_metrics
