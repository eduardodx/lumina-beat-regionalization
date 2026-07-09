from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from src.constants import (
    AA_NON_CDS,
    DNA_VOCAB,
    NUM_REGION_CLASSES,
    NUM_STRUCTURE_CLASSES,
    PAD_ID,
    REGION_CDS,
    REGION_INTERGENIC,
    REGION_INTRON,
    REGION_NONCODING_EXON,
    REGION_UTR,
    SNV_BASES,
    STRUCT_BACKGROUND,
    STRUCT_SPLICE_CORE,
    STRUCT_SPLICE_REGION,
)

METRIC_STAT_KEYS = (
    "mlm_accuracy",
    "aa_accuracy",
    "phylo100_pearson",
    "phylo470_pearson",
    "codon_phylo_pearson",
    "splice_positive_precision",
    "splice_positive_recall",
    "splice_positive_f1",
    "splice_background_precision",
    "splice_background_recall",
    "splice_background_f1",
    "splice_core_precision",
    "splice_core_recall",
    "splice_core_f1",
    "splice_region_precision",
    "splice_region_recall",
    "splice_region_f1",
    "region_intergenic_precision",
    "region_intergenic_recall",
    "region_intergenic_f1",
    "region_intron_precision",
    "region_intron_recall",
    "region_intron_f1",
    "region_noncoding_exon_precision",
    "region_noncoding_exon_recall",
    "region_noncoding_exon_f1",
    "region_utr_precision",
    "region_utr_recall",
    "region_utr_f1",
    "region_cds_precision",
    "region_cds_recall",
    "region_cds_f1",
)


@dataclass
class CorrelationAccumulator:
    count: int = 0
    sum_x: float = 0.0
    sum_y: float = 0.0
    sum_xx: float = 0.0
    sum_yy: float = 0.0
    sum_xy: float = 0.0

    def update(self, pred: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor) -> None:
        x = pred[valid_mask].detach().float().reshape(-1)
        if x.numel() == 0:
            return

        y = target[valid_mask].detach().float().reshape(-1)
        self.count += int(x.numel())
        self.sum_x += float(x.sum().item())
        self.sum_y += float(y.sum().item())
        self.sum_xx += float((x * x).sum().item())
        self.sum_yy += float((y * y).sum().item())
        self.sum_xy += float((x * y).sum().item())

    def merge(self, other: CorrelationAccumulator) -> None:
        self.count += other.count
        self.sum_x += other.sum_x
        self.sum_y += other.sum_y
        self.sum_xx += other.sum_xx
        self.sum_yy += other.sum_yy
        self.sum_xy += other.sum_xy

    def reset(self) -> None:
        self.count = 0
        self.sum_x = 0.0
        self.sum_y = 0.0
        self.sum_xx = 0.0
        self.sum_yy = 0.0
        self.sum_xy = 0.0

    def value(self) -> float:
        if self.count < 2:
            return 0.0

        numerator = self.count * self.sum_xy - self.sum_x * self.sum_y
        denom_x = self.count * self.sum_xx - self.sum_x * self.sum_x
        denom_y = self.count * self.sum_yy - self.sum_y * self.sum_y
        if denom_x <= 0.0 or denom_y <= 0.0:
            return 0.0
        return numerator / math.sqrt(denom_x * denom_y)


def empty_confusion_matrix() -> torch.Tensor:
    return torch.zeros((NUM_STRUCTURE_CLASSES, NUM_STRUCTURE_CLASSES), dtype=torch.int64)


def empty_region_confusion_matrix() -> torch.Tensor:
    return torch.zeros((NUM_REGION_CLASSES, NUM_REGION_CLASSES), dtype=torch.int64)


def _metric_safe_name(name: str) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in name).strip("_")


def _rankdata(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _spearman_rank_correlation(preds: np.ndarray, targets: np.ndarray) -> float:
    if preds.size < 2 or targets.size < 2 or preds.size != targets.size:
        return 0.0
    pred_ranks = _rankdata(preds)
    target_ranks = _rankdata(targets)
    pred_centered = pred_ranks - float(np.mean(pred_ranks))
    target_centered = target_ranks - float(np.mean(target_ranks))
    denom = float(np.linalg.norm(pred_centered) * np.linalg.norm(target_centered))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(pred_centered, target_centered) / denom)


def map_mlm_labels_for_logits(mlm_logits: torch.Tensor, mlm_labels: torch.Tensor) -> torch.Tensor:
    if mlm_logits.shape[-1] != len(SNV_BASES):
        return mlm_labels

    mapped = torch.full_like(mlm_labels, -100)
    for index, base in enumerate(SNV_BASES):
        mapped = torch.where(mlm_labels == DNA_VOCAB[base], torch.full_like(mapped, index), mapped)
    return mapped


def safe_precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision_den = tp + fp
    recall_den = tp + fn
    precision = tp / precision_den if precision_den > 0 else 0.0
    recall = tp / recall_den if recall_den > 0 else 0.0
    if precision + recall == 0.0:
        return precision, recall, 0.0
    return precision, recall, 2.0 * precision * recall / (precision + recall)


def confusion_metrics(confusion: torch.Tensor) -> dict[str, float]:
    summary: dict[str, float] = {}
    for class_name, class_index in (
        ("background", STRUCT_BACKGROUND),
        ("core", STRUCT_SPLICE_CORE),
        ("region", STRUCT_SPLICE_REGION),
    ):
        tp = int(confusion[class_index, class_index].item())
        fp = int(confusion[:, class_index].sum().item()) - tp
        fn = int(confusion[class_index, :].sum().item()) - tp
        precision, recall, f1 = safe_precision_recall_f1(tp, fp, fn)
        summary[f"splice_{class_name}_precision"] = precision
        summary[f"splice_{class_name}_recall"] = recall
        summary[f"splice_{class_name}_f1"] = f1

    positive_tp = int(confusion[1:, 1:].sum().item())
    positive_fp = int(confusion[STRUCT_BACKGROUND, 1:].sum().item())
    positive_fn = int(confusion[1:, STRUCT_BACKGROUND].sum().item())
    precision, recall, f1 = safe_precision_recall_f1(positive_tp, positive_fp, positive_fn)
    summary["splice_positive_precision"] = precision
    summary["splice_positive_recall"] = recall
    summary["splice_positive_f1"] = f1
    return summary


def region_confusion_metrics(confusion: torch.Tensor) -> dict[str, float]:
    summary: dict[str, float] = {}
    for class_name, class_index in (
        ("intergenic", REGION_INTERGENIC),
        ("intron", REGION_INTRON),
        ("noncoding_exon", REGION_NONCODING_EXON),
        ("utr", REGION_UTR),
        ("cds", REGION_CDS),
    ):
        tp = int(confusion[class_index, class_index].item())
        fp = int(confusion[:, class_index].sum().item()) - tp
        fn = int(confusion[class_index, :].sum().item()) - tp
        precision, recall, f1 = safe_precision_recall_f1(tp, fp, fn)
        summary[f"region_{class_name}_precision"] = precision
        summary[f"region_{class_name}_recall"] = recall
        summary[f"region_{class_name}_f1"] = f1
    return summary


@dataclass
class MetricAccumulator:
    mlm_correct: int = 0
    mlm_total: int = 0
    aa_correct: int = 0
    aa_total: int = 0
    phylo100: CorrelationAccumulator = field(default_factory=CorrelationAccumulator)
    phylo470: CorrelationAccumulator = field(default_factory=CorrelationAccumulator)
    codon_phylo: CorrelationAccumulator = field(default_factory=CorrelationAccumulator)
    splice_confusion: torch.Tensor = field(default_factory=empty_confusion_matrix)
    region_confusion: torch.Tensor = field(default_factory=empty_region_confusion_matrix)
    encode_tracks: dict[str, CorrelationAccumulator] = field(default_factory=dict)
    encode_track_values: dict[str, tuple[list[float], list[float]]] = field(default_factory=dict)

    def update_from_batch(self, outputs: dict[str, torch.Tensor], batch: dict[str, Any]) -> None:
        mlm_labels = map_mlm_labels_for_logits(outputs["mlm_logits"], batch["mlm_labels"])
        ignore_index = -100 if outputs["mlm_logits"].shape[-1] == len(SNV_BASES) else PAD_ID
        mlm_mask = mlm_labels != ignore_index
        if torch.any(mlm_mask):
            mlm_pred = outputs["mlm_logits"].argmax(dim=-1)
            self.mlm_correct += int((mlm_pred[mlm_mask] == mlm_labels[mlm_mask]).sum().item())
            self.mlm_total += int(mlm_mask.sum().item())

        aux_valid_mask = batch["aux_valid_mask"]
        if "phylo100_pred" in outputs:
            self.phylo100.update(outputs["phylo100_pred"], batch["phylo100"], aux_valid_mask)
        if "phylo470_pred" in outputs:
            self.phylo470.update(outputs["phylo470_pred"], batch["phylo470"], aux_valid_mask)
        aa_valid_mask = aux_valid_mask & (batch["aa_labels"] != AA_NON_CDS)
        if "aa_logits" in outputs and torch.any(aa_valid_mask):
            aa_preds = outputs["aa_logits"].argmax(dim=-1)
            self.aa_correct += int((aa_preds[aa_valid_mask] == batch["aa_labels"][aa_valid_mask]).sum().item())
            self.aa_total += int(aa_valid_mask.sum().item())
        if "codon_phylo_pred" in outputs:
            self.codon_phylo.update(outputs["codon_phylo_pred"], batch["codon_phylo_target"], aa_valid_mask)

        if "structure_logits" not in outputs:
            return

        labels = batch["structure_labels"][aux_valid_mask].detach().to(dtype=torch.int64, device="cpu")
        if labels.numel() == 0:
            return

        preds = outputs["structure_logits"].argmax(dim=-1)[aux_valid_mask].detach().to(dtype=torch.int64, device="cpu")
        encoded = labels * NUM_STRUCTURE_CLASSES + preds
        confusion = torch.bincount(encoded, minlength=NUM_STRUCTURE_CLASSES * NUM_STRUCTURE_CLASSES)
        self.splice_confusion += confusion.reshape(NUM_STRUCTURE_CLASSES, NUM_STRUCTURE_CLASSES)

        if "region_logits" in outputs:
            region_labels = batch["region_labels"][aux_valid_mask].detach().to(dtype=torch.int64, device="cpu")
            if region_labels.numel() > 0:
                region_preds = outputs["region_logits"].argmax(dim=-1)[aux_valid_mask].detach().to(
                    dtype=torch.int64, device="cpu"
                )
                region_encoded = region_labels * NUM_REGION_CLASSES + region_preds
                region_cm = torch.bincount(region_encoded, minlength=NUM_REGION_CLASSES * NUM_REGION_CLASSES)
                self.region_confusion += region_cm.reshape(NUM_REGION_CLASSES, NUM_REGION_CLASSES)

        if "encode_pred" in outputs and "encode_targets" in batch:
            track_names = batch.get("encode_track_names", [])
            if isinstance(track_names, list):
                for track_index, track_name in enumerate(track_names):
                    valid_mask = torch.isfinite(batch["encode_targets"][..., track_index])
                    if valid_mask.sum() == 0:
                        continue
                    accumulator = self.encode_tracks.setdefault(track_name, CorrelationAccumulator())
                    accumulator.update(
                        outputs["encode_pred"][..., track_index],
                        batch["encode_targets"][..., track_index],
                        valid_mask,
                    )
                    preds_cpu = outputs["encode_pred"][..., track_index][valid_mask].detach().float().cpu().tolist()
                    targets_cpu = batch["encode_targets"][..., track_index][valid_mask].detach().float().cpu().tolist()
                    pred_values, target_values = self.encode_track_values.setdefault(track_name, ([], []))
                    pred_values.extend(preds_cpu)
                    target_values.extend(targets_cpu)

    def merge(self, other: MetricAccumulator) -> None:
        self.mlm_correct += other.mlm_correct
        self.mlm_total += other.mlm_total
        self.aa_correct += other.aa_correct
        self.aa_total += other.aa_total
        self.phylo100.merge(other.phylo100)
        self.phylo470.merge(other.phylo470)
        self.codon_phylo.merge(other.codon_phylo)
        self.splice_confusion += other.splice_confusion
        self.region_confusion += other.region_confusion
        for track_name, accumulator in other.encode_tracks.items():
            self.encode_tracks.setdefault(track_name, CorrelationAccumulator()).merge(accumulator)
        for track_name, values in other.encode_track_values.items():
            pred_values, target_values = self.encode_track_values.setdefault(track_name, ([], []))
            pred_values.extend(values[0])
            target_values.extend(values[1])

    def reset(self) -> None:
        self.mlm_correct = 0
        self.mlm_total = 0
        self.aa_correct = 0
        self.aa_total = 0
        self.phylo100.reset()
        self.phylo470.reset()
        self.codon_phylo.reset()
        self.splice_confusion.zero_()
        self.region_confusion.zero_()
        self.encode_tracks.clear()
        self.encode_track_values.clear()

    def all_reduce(self, device: torch.device) -> None:
        """Aggregate accumulator state across all DDP ranks in-place.

        All fields are summed (not averaged) because they are raw counts and
        running sums.  Derived metrics (accuracy, Pearson r, F1) are computed
        afterwards by ``summary()``, which handles division correctly.
        """
        import torch.distributed as _dist

        if not _dist.is_initialized():
            return

        stats = torch.tensor(
            [
                self.mlm_correct,
                self.mlm_total,
                self.aa_correct,
                self.aa_total,
                self.phylo100.count,
                self.phylo100.sum_x,
                self.phylo100.sum_y,
                self.phylo100.sum_xx,
                self.phylo100.sum_yy,
                self.phylo100.sum_xy,
                self.phylo470.count,
                self.phylo470.sum_x,
                self.phylo470.sum_y,
                self.phylo470.sum_xx,
                self.phylo470.sum_yy,
                self.phylo470.sum_xy,
                self.codon_phylo.count,
                self.codon_phylo.sum_x,
                self.codon_phylo.sum_y,
                self.codon_phylo.sum_xx,
                self.codon_phylo.sum_yy,
                self.codon_phylo.sum_xy,
            ],
            dtype=torch.float64,
            device=device,
        )
        _dist.all_reduce(stats, op=_dist.ReduceOp.SUM)
        s = stats.cpu().tolist()

        self.mlm_correct = int(s[0])
        self.mlm_total = int(s[1])
        self.aa_correct = int(s[2])
        self.aa_total = int(s[3])
        self.phylo100.count = int(s[4])
        self.phylo100.sum_x = s[5]
        self.phylo100.sum_y = s[6]
        self.phylo100.sum_xx = s[7]
        self.phylo100.sum_yy = s[8]
        self.phylo100.sum_xy = s[9]
        self.phylo470.count = int(s[10])
        self.phylo470.sum_x = s[11]
        self.phylo470.sum_y = s[12]
        self.phylo470.sum_xx = s[13]
        self.phylo470.sum_yy = s[14]
        self.phylo470.sum_xy = s[15]
        self.codon_phylo.count = int(s[16])
        self.codon_phylo.sum_x = s[17]
        self.codon_phylo.sum_y = s[18]
        self.codon_phylo.sum_xx = s[19]
        self.codon_phylo.sum_yy = s[20]
        self.codon_phylo.sum_xy = s[21]

        cm = self.splice_confusion.to(dtype=torch.int64, device=device)
        _dist.all_reduce(cm, op=_dist.ReduceOp.SUM)
        self.splice_confusion = cm.cpu()

        rcm = self.region_confusion.to(dtype=torch.int64, device=device)
        _dist.all_reduce(rcm, op=_dist.ReduceOp.SUM)
        self.region_confusion = rcm.cpu()

        encode_payload = {
            track_name: {
                "preds": list(values[0]),
                "targets": list(values[1]),
            }
            for track_name, values in self.encode_track_values.items()
        }
        gathered_payloads: list[dict[str, dict[str, list[float]]] | None] = [
            None for _ in range(_dist.get_world_size())
        ]
        _dist.all_gather_object(gathered_payloads, encode_payload)
        merged_encode_tracks: dict[str, CorrelationAccumulator] = {}
        merged_encode_values: dict[str, tuple[list[float], list[float]]] = {}
        for payload in gathered_payloads:
            if payload is None:
                continue
            for track_name, values in payload.items():
                preds = values.get("preds", [])
                targets = values.get("targets", [])
                pred_store, target_store = merged_encode_values.setdefault(track_name, ([], []))
                pred_store.extend(preds)
                target_store.extend(targets)
        for track_name, values in merged_encode_values.items():
            pred_array = torch.tensor(values[0], dtype=torch.float32)
            target_array = torch.tensor(values[1], dtype=torch.float32)
            valid_mask = torch.ones_like(pred_array, dtype=torch.bool)
            accumulator = CorrelationAccumulator()
            accumulator.update(pred_array, target_array, valid_mask)
            merged_encode_tracks[track_name] = accumulator
        self.encode_tracks = merged_encode_tracks
        self.encode_track_values = merged_encode_values

    def summary(self) -> dict[str, float]:
        summary = {
            "mlm_accuracy": self.mlm_correct / self.mlm_total if self.mlm_total > 0 else 0.0,
            "aa_accuracy": self.aa_correct / self.aa_total if self.aa_total > 0 else 0.0,
            "phylo100_pearson": self.phylo100.value(),
            "phylo470_pearson": self.phylo470.value(),
            "codon_phylo_pearson": self.codon_phylo.value(),
        }
        summary.update(confusion_metrics(self.splice_confusion))
        summary.update(region_confusion_metrics(self.region_confusion))
        for track_name, accumulator in sorted(self.encode_tracks.items()):
            safe_name = _metric_safe_name(track_name)
            summary[f"encode_{safe_name}_pearson"] = accumulator.value()
            pred_values, target_values = self.encode_track_values.get(track_name, ([], []))
            summary[f"encode_{safe_name}_spearman"] = _spearman_rank_correlation(
                np.asarray(pred_values, dtype=np.float32),
                np.asarray(target_values, dtype=np.float32),
            )
        return summary
