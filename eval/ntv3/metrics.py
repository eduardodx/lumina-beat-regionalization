from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.zeros_like(num, dtype=np.float64)
    mask = den > 0
    out[mask] = num[mask] / den[mask]
    return out


@dataclass
class FunctionalTracksMetrics:
    track_ids: list[str]

    def __post_init__(self) -> None:
        num_tracks = len(self.track_ids)
        self.count = 0
        self.sum_x = np.zeros(num_tracks, dtype=np.float64)
        self.sum_y = np.zeros(num_tracks, dtype=np.float64)
        self.sum_x2 = np.zeros(num_tracks, dtype=np.float64)
        self.sum_y2 = np.zeros(num_tracks, dtype=np.float64)
        self.sum_xy = np.zeros(num_tracks, dtype=np.float64)
        self.losses: list[float] = []

    def update(self, predictions: torch.Tensor, targets: torch.Tensor, loss: float) -> None:
        pred_flat = predictions.detach().reshape(-1, len(self.track_ids)).to(torch.float64).cpu().numpy()
        target_flat = targets.detach().reshape(-1, len(self.track_ids)).to(torch.float64).cpu().numpy()
        self.count += pred_flat.shape[0]
        self.sum_x += pred_flat.sum(axis=0)
        self.sum_y += target_flat.sum(axis=0)
        self.sum_x2 += np.square(pred_flat).sum(axis=0)
        self.sum_y2 += np.square(target_flat).sum(axis=0)
        self.sum_xy += (pred_flat * target_flat).sum(axis=0)
        self.losses.append(float(loss))

    def compute(self) -> dict[str, Any]:
        n = float(self.count)
        numerator = n * self.sum_xy - self.sum_x * self.sum_y
        denom_x = n * self.sum_x2 - np.square(self.sum_x)
        denom_y = n * self.sum_y2 - np.square(self.sum_y)
        denominator = np.sqrt(np.maximum(denom_x * denom_y, 0.0))
        correlations = np.where(denominator > 0, numerator / denominator, 0.0)
        metrics = {
            f"{track_id}/pearson": float(correlations[index])
            for index, track_id in enumerate(self.track_ids)
        }
        metrics["mean/pearson"] = float(correlations.mean()) if len(correlations) else 0.0
        metrics["loss"] = float(np.mean(self.losses)) if self.losses else 0.0
        return metrics


@dataclass
class AnnotationMetrics:
    element_names: list[str]

    def __post_init__(self) -> None:
        num_elements = len(self.element_names)
        self.tp = np.zeros(num_elements, dtype=np.float64)
        self.tn = np.zeros(num_elements, dtype=np.float64)
        self.fp = np.zeros(num_elements, dtype=np.float64)
        self.fn = np.zeros(num_elements, dtype=np.float64)
        self.losses: list[float] = []

    def update(self, logits: torch.Tensor, labels: torch.Tensor, loss: float) -> None:
        predictions = logits.detach().argmax(dim=-1).cpu().numpy().reshape(-1, len(self.element_names))
        target = labels.detach().cpu().numpy().reshape(-1, len(self.element_names))
        self.tp += ((predictions == 1) & (target == 1)).sum(axis=0)
        self.tn += ((predictions == 0) & (target == 0)).sum(axis=0)
        self.fp += ((predictions == 1) & (target == 0)).sum(axis=0)
        self.fn += ((predictions == 0) & (target == 1)).sum(axis=0)
        self.losses.append(float(loss))

    def compute(self) -> dict[str, Any]:
        numerator = self.tp * self.tn - self.fp * self.fn
        denominator = np.sqrt((self.tp + self.fp) * (self.tp + self.fn) * (self.tn + self.fp) * (self.tn + self.fn))
        mcc = np.where(denominator > 0, numerator / denominator, 0.0)
        metrics = {
            f"{element_name}/mcc": float(mcc[index])
            for index, element_name in enumerate(self.element_names)
        }
        metrics["mean/mcc"] = float(mcc.mean()) if len(mcc) else 0.0
        metrics["loss"] = float(np.mean(self.losses)) if self.losses else 0.0
        return metrics


def format_seconds_as_duration(seconds: float) -> str:
    total_microseconds = round(max(seconds, 0.0) * 1_000_000)
    total_seconds, micros = divmod(total_microseconds, 1_000_000)
    days, remainder = divmod(total_seconds, 86_400)
    hours, remainder = divmod(remainder, 3_600)
    minutes, secs = divmod(remainder, 60)
    return f"{days} days {hours:02d}:{minutes:02d}:{secs:02d}.{micros:06d}"


def metric_value_for_task(task_type: str, metrics: dict[str, Any]) -> float:
    if task_type == "functional":
        return float(metrics.get("mean/pearson", 0.0))
    if task_type == "annotation":
        return float(metrics.get("mean/mcc", 0.0))
    raise ValueError(f"Unsupported task_type {task_type!r}.")
