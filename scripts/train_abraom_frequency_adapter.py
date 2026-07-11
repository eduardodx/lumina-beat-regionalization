#!/usr/bin/env python3
"""Train/evaluate the ABRAOM regional frequency adapter (A_BR).

The target is allele frequency, not clinical pathogenicity. For each ABRAOM
SNV, the script extracts REF/ALT hg38 windows, runs the Lumina backbone through
the existing fine-tune adapter, and trains only LoRA parameters plus a small
frequency head against a soft ``af_abraom`` target.

By default no ABRAOM-derived quantity such as ``specificity`` is used as an
input feature. An optional gnomAD prior can be enabled explicitly because it is
external to ABRAOM and useful as a calibrated global baseline.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from pyfaidx import Fasta
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.clinvar.adapters import FineTuneAdapter, build_finetune_adapter  # noqa: E402
from eval.clinvar.lora import (  # noqa: E402
    LoRASummary,
    apply_lora,
    count_trainable_parameters,
    enable_layernorm_training,
)
from eval.clinvar.variant_utils import extract_variant_window  # noqa: E402
from src.precision import (  # noqa: E402
    PrecisionMode,
    autocast_context,
    configure_float32_precision,
    precision_metadata,
    resolve_precision_policy,
)

DEFAULT_DATA_DIR = Path("data/datasets/abraom_frequency_adapter")
DEFAULT_FASTA = Path("/home/sagemaker-user/lumina/data/genomes/hg38/raw/hg38.fa")
DEFAULT_BASE_CHECKPOINT = Path(
    "/home/sagemaker-user/lumina-ssm/artifacts/abraom_regional_eval/checkpoints/base/best_checkpoint.pt"
)
DEFAULT_OUTPUT_DIR = Path("outputs/abraom_frequency_adapter")

FREQUENCY_COLUMNS = [
    "variant_id",
    "variant_key",
    "chrom",
    "Start",
    "ref",
    "alt",
    "af_abraom",
    "af_gnomad",
    "logit_af_gnomad",
    "scrambled_af_abraom",
    "delta_logit",
    "scrambled_delta_logit",
    "af_abraom_bin",
    "specificity_bin",
    "gnomad_zero",
    "split",
]

AF_BIN_ORDER = (
    "[0,0.005]",
    "(0.005,0.01]",
    "(0.01,0.05]",
    "(0.05,0.1]",
    "(0.1,0.2]",
    "(0.2,0.5]",
    "(0.5,1]",
)


@dataclass
class TrainConfig:
    train_parquet: str
    val_parquet: str
    test_parquet: str | None
    fasta: str
    output_dir: str
    checkpoint_path: str
    model_family: str = "lumina"
    model_version: str = "beat-v10"
    context_size: int = 4096
    target_column: str = "af_abraom"
    metric_target_column: str = "af_abraom"
    use_gnomad_prior: bool = False
    batch_size: int = 2
    eval_batch_size: int = 4
    num_workers: int = 0
    max_train_rows: int | None = None
    max_val_rows: int | None = None
    max_test_rows: int | None = None
    row_sample_strategy: str = "head"
    max_steps: int = 10_000
    grad_accum_steps: int = 8
    eval_every_steps: int = 500
    save_every_steps: int = 0
    lr_lora: float = 5e-6
    lr_head: float = 5e-4
    weight_decay_lora: float = 0.01
    weight_decay_head: float = 1e-4
    warmup_fraction: float = 0.05
    grad_clip: float = 0.5
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05
    train_layernorm: bool = False
    head_hidden_dim: int = 256
    head_dropout: float = 0.1
    precision: PrecisionMode = "auto"
    allow_tf32: bool = True
    device: str = "cuda"
    seed: int = 42
    n_fraction_max: float = 0.25
    log_every_steps: int = 25
    write_predictions: bool = True
    eval_only: bool = False
    adapter_checkpoint: str | None = None
    overwrite: bool = False
    # bce = [0,1] frequency target (default, e.g. af_abraom). mse/huber = regression for
    # unbounded targets like delta_logit (the ABRAOM-vs-gnomAD residual = direct regional test).
    loss: str = "bce"


@dataclass(frozen=True)
class FrequencyExample:
    variant_id: int
    variant_key: str
    chrom: str
    start_1based: int
    ref: str
    alt: str
    af_abraom: float
    af_gnomad: float
    logit_af_gnomad: float
    scrambled_af_abraom: float
    delta_logit: float
    scrambled_delta_logit: float
    af_abraom_bin: str
    specificity_bin: str
    gnomad_zero: bool
    split: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _read_selected_rows(parquet_file: pq.ParquetFile, columns: list[str], selected: np.ndarray) -> pd.DataFrame:
    if len(selected) == 0:
        return pd.DataFrame(columns=columns)
    selected = np.asarray(np.sort(selected), dtype=np.int64)
    frames = []
    row_offset = 0
    for batch in parquet_file.iter_batches(batch_size=65_536, columns=columns):
        batch_end = row_offset + batch.num_rows
        left = int(np.searchsorted(selected, row_offset, side="left"))
        right = int(np.searchsorted(selected, batch_end, side="left"))
        if right > left:
            local_indices = selected[left:right] - row_offset
            sampled = batch.take(pa.array(local_indices, type=pa.int64()))
            frames.append(pa.Table.from_batches([sampled]).to_pandas())
        row_offset = batch_end
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)


def _balanced_quotas(counts: dict[str, int], limit: int) -> dict[str, int]:
    ordered_bins = [bin_name for bin_name in AF_BIN_ORDER if counts.get(bin_name, 0) > 0]
    ordered_bins.extend(
        sorted(bin_name for bin_name in counts if bin_name not in AF_BIN_ORDER and counts[bin_name] > 0)
    )
    available = {bin_name: int(counts[bin_name]) for bin_name in ordered_bins}
    quotas = {bin_name: 0 for bin_name in ordered_bins}
    remaining = min(int(limit), sum(available.values()))

    while remaining > 0:
        active = [bin_name for bin_name in ordered_bins if available[bin_name] > 0]
        if not active:
            break
        base = max(remaining // len(active), 1)
        extra = remaining % len(active)
        for idx, bin_name in enumerate(active):
            desired = base + (1 if idx < extra else 0)
            take = min(desired, available[bin_name], remaining)
            quotas[bin_name] += take
            available[bin_name] -= take
            remaining -= take
            if remaining == 0:
                break
    return quotas


def _sample_balanced_by_af_bin(path: Path, limit: int, seed: int) -> np.ndarray:
    bin_table = pq.read_table(path, columns=["af_abraom_bin"])
    bin_column = bin_table.column("af_abraom_bin")
    counts: dict[str, int] = {}
    value_counts = pc.value_counts(bin_column).to_pylist()
    for row in value_counts:
        counts[str(row["values"])] = int(row["counts"])

    quotas = _balanced_quotas(counts, limit)
    rng = np.random.default_rng(seed)
    selected_parts: list[np.ndarray] = []
    for bin_name, quota in quotas.items():
        if quota <= 0:
            continue
        bin_indices = pc.indices_nonzero(pc.equal(bin_column, bin_name)).to_numpy(zero_copy_only=False)
        if quota >= len(bin_indices):
            selected_parts.append(np.asarray(bin_indices, dtype=np.int64))
            continue
        chosen = rng.choice(bin_indices, size=quota, replace=False)
        selected_parts.append(np.asarray(chosen, dtype=np.int64))

    if not selected_parts:
        return np.asarray([], dtype=np.int64)
    return np.sort(np.concatenate(selected_parts).astype(np.int64, copy=False))


def _read_parquet_limited(
    path: Path,
    columns: list[str],
    limit: int | None,
    *,
    strategy: str = "head",
    seed: int = 0,
) -> pd.DataFrame:
    available = set(pq.ParquetFile(path).schema_arrow.names)
    missing = sorted(set(columns) - available)
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")

    if limit is None:
        return pd.read_parquet(path, columns=columns)
    if limit <= 0:
        return pd.DataFrame(columns=columns)

    parquet_file = pq.ParquetFile(path)
    if strategy == "random":
        total_rows = parquet_file.metadata.num_rows
        if limit >= total_rows:
            return pd.read_parquet(path, columns=columns)
        rng = np.random.default_rng(seed)
        selected = np.sort(rng.choice(total_rows, size=limit, replace=False))
        return _read_selected_rows(parquet_file, columns, selected)
    if strategy == "balanced-af":
        total_rows = parquet_file.metadata.num_rows
        if limit >= total_rows:
            return pd.read_parquet(path, columns=columns)
        selected = _sample_balanced_by_af_bin(path, limit, seed)
        return _read_selected_rows(parquet_file, columns, selected)
    if strategy != "head":
        raise ValueError(f"Unsupported row sample strategy: {strategy!r}")

    frames: list[pd.DataFrame] = []
    remaining = limit
    for batch in parquet_file.iter_batches(batch_size=min(max(limit, 1), 65_536), columns=columns):
        frame = pa.Table.from_batches([batch]).to_pandas()
        if len(frame) > remaining:
            frame = frame.iloc[:remaining].copy()
        frames.append(frame)
        remaining -= len(frame)
        if remaining <= 0:
            break
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)


def load_frequency_examples(
    path: Path,
    *,
    limit: int | None,
    strategy: str = "head",
    seed: int = 0,
) -> list[FrequencyExample]:
    df = _read_parquet_limited(path, FREQUENCY_COLUMNS, limit, strategy=strategy, seed=seed)
    if df.empty:
        return []
    examples: list[FrequencyExample] = []
    for row in df.itertuples(index=False):
        examples.append(
            FrequencyExample(
                variant_id=int(row.variant_id),
                variant_key=str(row.variant_key),
                chrom=str(row.chrom),
                start_1based=int(row.Start),
                ref=str(row.ref).upper(),
                alt=str(row.alt).upper(),
                af_abraom=float(row.af_abraom),
                af_gnomad=float(row.af_gnomad),
                logit_af_gnomad=float(row.logit_af_gnomad),
                scrambled_af_abraom=float(row.scrambled_af_abraom),
                delta_logit=float(row.delta_logit),
                scrambled_delta_logit=float(row.scrambled_delta_logit),
                af_abraom_bin=str(row.af_abraom_bin),
                specificity_bin=str(row.specificity_bin),
                gnomad_zero=bool(row.gnomad_zero),
                split=str(row.split),
            )
        )
    return examples


class AbraomFrequencyDataset(Dataset):
    def __init__(
        self,
        examples: list[FrequencyExample],
        *,
        fasta_path: Path,
        context_size: int,
        target_column: str,
        metric_target_column: str,
        n_fraction_max: float,
        use_gnomad_prior: bool,
    ) -> None:
        self.examples = examples
        self.fasta_path = fasta_path
        self.context_size = context_size
        self.target_column = target_column
        self.metric_target_column = metric_target_column
        self.n_fraction_max = n_fraction_max
        self.use_gnomad_prior = use_gnomad_prior
        self._fasta: Fasta | None = None

    def __len__(self) -> int:
        return len(self.examples)

    @property
    def fasta(self) -> Fasta:
        if self._fasta is None:
            self._fasta = Fasta(str(self.fasta_path), sequence_always_upper=True)
        return self._fasta

    def _value_for_column(self, example: FrequencyExample, column: str) -> float:
        if column == "af_abraom":
            return example.af_abraom
        if column == "scrambled_af_abraom":
            return example.scrambled_af_abraom
        if column == "af_gnomad":
            return example.af_gnomad
        if column == "delta_logit":
            return example.delta_logit
        if column == "scrambled_delta_logit":
            return example.scrambled_delta_logit
        raise ValueError(f"Unsupported frequency target column: {column!r}")

    def __getitem__(self, index: int) -> dict[str, Any] | None:
        example = self.examples[index]
        chrom = example.chrom.removeprefix("chr")
        window = extract_variant_window(
            self.fasta,
            chrom=chrom,
            pos=example.start_1based,
            ref=example.ref,
            alt=example.alt,
            context_size=self.context_size,
        )
        if window is None or window.n_fraction > self.n_fraction_max:
            return None

        scalar_features: list[float] = []
        if self.use_gnomad_prior:
            scalar_features.extend([example.logit_af_gnomad, float(example.gnomad_zero)])

        return {
            "variant_id": example.variant_id,
            "variant_key": example.variant_key,
            "ref_seq": window.ref_seq,
            "alt_seq": window.alt_seq,
            "variant_offset": int(window.variant_offset),
            "ref_allele": example.ref,
            "alt_allele": example.alt,
            "target": self._value_for_column(example, self.target_column),
            "metric_target": self._value_for_column(example, self.metric_target_column),
            "af_abraom": example.af_abraom,
            "af_gnomad": example.af_gnomad,
            "af_abraom_bin": example.af_abraom_bin,
            "specificity_bin": example.specificity_bin,
            "split": example.split,
            "scalar_features": scalar_features,
            "window_status": window.status,
        }


def frequency_collate_fn(batch: list[dict[str, Any] | None]) -> dict[str, Any] | None:
    rows = [row for row in batch if row is not None]
    if not rows:
        return None
    scalar_features = [row["scalar_features"] for row in rows]
    scalar_tensor = None
    if scalar_features and len(scalar_features[0]) > 0:
        scalar_tensor = torch.tensor(scalar_features, dtype=torch.float32)
    return {
        "variant_ids": [int(row["variant_id"]) for row in rows],
        "variant_keys": [str(row["variant_key"]) for row in rows],
        "ref_seqs": [str(row["ref_seq"]) for row in rows],
        "alt_seqs": [str(row["alt_seq"]) for row in rows],
        "variant_offsets": [int(row["variant_offset"]) for row in rows],
        "ref_alleles": [str(row["ref_allele"]) for row in rows],
        "alt_alleles": [str(row["alt_allele"]) for row in rows],
        "targets": torch.tensor([float(row["target"]) for row in rows], dtype=torch.float32),
        "metric_targets": torch.tensor([float(row["metric_target"]) for row in rows], dtype=torch.float32),
        "af_abraom": np.asarray([float(row["af_abraom"]) for row in rows], dtype=np.float32),
        "af_gnomad": np.asarray([float(row["af_gnomad"]) for row in rows], dtype=np.float32),
        "af_abraom_bins": [str(row["af_abraom_bin"]) for row in rows],
        "specificity_bins": [str(row["specificity_bin"]) for row in rows],
        "splits": [str(row["split"]) for row in rows],
        "scalar_features": scalar_tensor,
        "window_statuses": [str(row["window_status"]) for row in rows],
    }


class AbraomFrequencyModel(nn.Module):
    def __init__(
        self,
        adapter: FineTuneAdapter,
        *,
        scalar_feature_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.adapter = adapter
        self.backbone = adapter.backbone
        self.scalar_feature_dim = scalar_feature_dim
        input_dim = adapter.d_model * 3 + scalar_feature_dim
        self.head = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        ref_seqs: list[str],
        alt_seqs: list[str],
        variant_offsets: list[int],
        ref_alleles: list[str],
        alt_alleles: list[str],
        scalar_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        site_ref, variant_repr, local_context = self.adapter.extract_variant_features(
            ref_seqs,
            alt_seqs,
            variant_offsets,
            ref_alleles,
            alt_alleles,
        )
        pieces = [site_ref, variant_repr, local_context]
        if self.scalar_feature_dim:
            if scalar_features is None:
                raise ValueError("scalar_features are required by this model.")
            pieces.append(scalar_features.to(device=site_ref.device, dtype=site_ref.dtype))
        return self.head(torch.cat(pieces, dim=-1)).squeeze(-1)


def freeze_backbone(backbone: nn.Module) -> None:
    for parameter in backbone.parameters():
        parameter.requires_grad = False


def build_model(config: TrainConfig, device: torch.device, precision_policy: Any) -> tuple[AbraomFrequencyModel, Any]:
    adapter = build_finetune_adapter(
        config.model_family,
        config.model_version,
        device,
        checkpoint_path=config.checkpoint_path,
        precision=precision_policy,
        native_feature_heads=["none"],
    )
    freeze_backbone(adapter.backbone)
    lora_summary: LoRASummary | None = None
    if config.lora_rank > 0:
        lora_summary = apply_lora(
            adapter.backbone,
            rank=config.lora_rank,
            alpha=config.lora_alpha,
            dropout=config.lora_dropout,
        )
    if config.train_layernorm:
        enable_layernorm_training(adapter.backbone)

    scalar_feature_dim = 2 if config.use_gnomad_prior else 0
    model = AbraomFrequencyModel(
        adapter,
        scalar_feature_dim=scalar_feature_dim,
        hidden_dim=config.head_hidden_dim,
        dropout=config.head_dropout,
    ).to(device)
    return model, lora_summary


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def _pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2:
        return None
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x_std == 0.0 or y_std == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2:
        return None
    return _pearson(_rankdata(x), _rankdata(y))


def _nll_from_probs(target: np.ndarray, prob: np.ndarray, eps: float = 1e-7) -> float:
    prob = np.clip(prob.astype(np.float64), eps, 1.0 - eps)
    target = target.astype(np.float64)
    return float(-(target * np.log(prob) + (1.0 - target) * np.log1p(-prob)).mean())


def build_metrics(target: np.ndarray, pred: np.ndarray, *, prefix: str = "") -> dict[str, Any]:
    target = target.astype(np.float64)
    pred = np.clip(pred.astype(np.float64), 0.0, 1.0)
    payload: dict[str, Any] = {
        f"{prefix}n": len(target),
        f"{prefix}nll": _nll_from_probs(target, pred),
        f"{prefix}brier": float(np.mean((pred - target) ** 2)),
        f"{prefix}mae": float(np.mean(np.abs(pred - target))),
        f"{prefix}mean_target": float(np.mean(target)),
        f"{prefix}mean_pred": float(np.mean(pred)),
        f"{prefix}pearson": _pearson(target, pred),
        f"{prefix}spearman": _spearman(target, pred),
    }
    return payload


def calibration_table(
    *,
    target: np.ndarray,
    pred: np.ndarray,
    groups: list[str],
    group_name: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    frame = pd.DataFrame({"target": target, "pred": pred, "group": groups})
    for group, part in frame.groupby("group", sort=True):
        target_arr = part["target"].to_numpy(dtype=np.float64)
        pred_arr = part["pred"].to_numpy(dtype=np.float64)
        rows.append(
            {
                group_name: str(group),
                "n": len(part),
                "mean_target": float(np.mean(target_arr)),
                "mean_pred": float(np.mean(pred_arr)),
                "brier": float(np.mean((pred_arr - target_arr) ** 2)),
                "mae": float(np.mean(np.abs(pred_arr - target_arr))),
            }
        )
    return pd.DataFrame(rows)


def _move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    batch["targets"] = batch["targets"].to(device)
    batch["metric_targets"] = batch["metric_targets"].to(device)
    scalar_features = batch.get("scalar_features")
    if scalar_features is not None:
        batch["scalar_features"] = scalar_features.to(device)
    return batch


@torch.no_grad()
def predict(
    model: AbraomFrequencyModel,
    loader: DataLoader,
    *,
    device: torch.device,
    precision_policy: Any,
) -> pd.DataFrame:
    model.eval()
    rows: list[dict[str, Any]] = []
    for batch in loader:
        if batch is None:
            continue
        batch = _move_batch_to_device(batch, device)
        with autocast_context(precision_policy):
            logits = model(
                batch["ref_seqs"],
                batch["alt_seqs"],
                batch["variant_offsets"],
                batch["ref_alleles"],
                batch["alt_alleles"],
                batch.get("scalar_features"),
            )
        probs = torch.sigmoid(logits.float()).detach().cpu().numpy()
        metric_targets = batch["metric_targets"].detach().cpu().numpy()
        loss_targets = batch["targets"].detach().cpu().numpy()
        for idx, prob in enumerate(probs):
            rows.append(
                {
                    "variant_id": batch["variant_ids"][idx],
                    "variant_key": batch["variant_keys"][idx],
                    "split": batch["splits"][idx],
                    "target": float(loss_targets[idx]),
                    "metric_target": float(metric_targets[idx]),
                    "pred_af_abraom": float(prob),
                    "af_abraom": float(batch["af_abraom"][idx]),
                    "af_gnomad": float(batch["af_gnomad"][idx]),
                    "af_abraom_bin": batch["af_abraom_bins"][idx],
                    "specificity_bin": batch["specificity_bins"][idx],
                    "window_status": batch["window_statuses"][idx],
                }
            )
    return pd.DataFrame(rows)


def evaluate_predictions(predictions: pd.DataFrame) -> dict[str, Any]:
    if predictions.empty:
        raise ValueError("No predictions were produced.")
    target = predictions["metric_target"].to_numpy(dtype=np.float64)
    pred = predictions["pred_af_abraom"].to_numpy(dtype=np.float64)
    gnomad = predictions["af_gnomad"].to_numpy(dtype=np.float64)
    model_metrics = build_metrics(target, pred, prefix="model_")
    gnomad_metrics = build_metrics(target, gnomad, prefix="gnomad_")
    return {
        "overall": {
            **model_metrics,
            **gnomad_metrics,
            "delta_nll_model_minus_gnomad": model_metrics["model_nll"] - gnomad_metrics["gnomad_nll"],
            "delta_brier_model_minus_gnomad": model_metrics["model_brier"] - gnomad_metrics["gnomad_brier"],
        },
        "by_af_bin": calibration_table(
            target=target,
            pred=pred,
            groups=predictions["af_abraom_bin"].astype(str).tolist(),
            group_name="af_abraom_bin",
        ).to_dict(orient="records"),
        "by_specificity_bin": calibration_table(
            target=target,
            pred=pred,
            groups=predictions["specificity_bin"].astype(str).tolist(),
            group_name="specificity_bin",
        ).to_dict(orient="records"),
    }


def _cosine_lr_scale(step: int, *, total_steps: int, warmup_steps: int, min_lr_ratio: float = 0.01) -> float:
    if step < warmup_steps:
        return step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress))


def make_optimizer(model: AbraomFrequencyModel, config: TrainConfig) -> torch.optim.Optimizer:
    lora_params = [p for name, p in model.backbone.named_parameters() if "lora_" in name and p.requires_grad]
    norm_params = [
        p
        for name, p in model.backbone.named_parameters()
        if "lora_" not in name and p.requires_grad
    ]
    head_params = [p for p in model.head.parameters() if p.requires_grad]
    groups: list[dict[str, Any]] = []
    if lora_params:
        groups.append({"params": lora_params, "lr": config.lr_lora, "weight_decay": config.weight_decay_lora})
    if norm_params:
        groups.append({"params": norm_params, "lr": config.lr_lora, "weight_decay": config.weight_decay_lora})
    groups.append({"params": head_params, "lr": config.lr_head, "weight_decay": config.weight_decay_head})
    return torch.optim.AdamW(groups)


def save_adapter_checkpoint(
    path: Path,
    *,
    model: AbraomFrequencyModel,
    config: TrainConfig,
    lora_summary: LoRASummary | None,
    metrics: dict[str, Any],
    step: int,
    precision: dict[str, Any],
) -> None:
    trainable_state = {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    payload = {
        "format": "abraom_frequency_adapter_v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "step": step,
        "config": asdict(config),
        "precision": precision,
        "metrics": metrics,
        "lora": asdict(lora_summary) if lora_summary is not None else None,
        "trainable_state_dict": trainable_state,
    }
    torch.save(payload, path)


def load_adapter_checkpoint(path: Path, model: AbraomFrequencyModel, device: torch.device) -> dict[str, Any]:
    payload = torch.load(path, map_location=device, weights_only=False)
    state = payload.get("trainable_state_dict")
    if not isinstance(state, dict):
        raise RuntimeError(f"{path} does not contain trainable_state_dict.")
    missing, unexpected = model.load_state_dict(state, strict=False)
    unexpected_trainable = [name for name in unexpected if "lora_" in name or name.startswith("head.")]
    if unexpected_trainable:
        raise RuntimeError(f"Unexpected trainable checkpoint keys in {path}: {unexpected_trainable}")
    payload["load_missing_keys"] = list(missing)
    return payload


def build_loader(
    examples: list[FrequencyExample],
    *,
    config: TrainConfig,
    target_column: str,
    shuffle: bool,
    batch_size: int,
) -> DataLoader:
    dataset = AbraomFrequencyDataset(
        examples,
        fasta_path=Path(config.fasta),
        context_size=config.context_size,
        target_column=target_column,
        metric_target_column=config.metric_target_column,
        n_fraction_max=config.n_fraction_max,
        use_gnomad_prior=config.use_gnomad_prior,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        collate_fn=frequency_collate_fn,
        drop_last=False,
    )


def evaluate_split(
    *,
    name: str,
    model: AbraomFrequencyModel,
    examples: list[FrequencyExample],
    config: TrainConfig,
    device: torch.device,
    precision_policy: Any,
    output_dir: Path,
    write_predictions: bool,
) -> dict[str, Any]:
    loader = build_loader(
        examples,
        config=config,
        target_column=config.target_column,
        shuffle=False,
        batch_size=config.eval_batch_size,
    )
    predictions = predict(model, loader, device=device, precision_policy=precision_policy)
    metrics = evaluate_predictions(predictions)
    metrics["split"] = name
    metrics["rows_loaded"] = len(examples)
    metrics["rows_scored"] = len(predictions)
    if write_predictions:
        predictions.to_parquet(output_dir / f"{name}_predictions.parquet", index=False)
    pd.DataFrame(metrics["by_af_bin"]).to_parquet(output_dir / f"{name}_calibration_by_af_bin.parquet", index=False)
    pd.DataFrame(metrics["by_specificity_bin"]).to_parquet(
        output_dir / f"{name}_calibration_by_specificity_bin.parquet",
        index=False,
    )
    return metrics


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# ABRAOM Frequency Adapter Run",
        "",
        f"Generated at UTC: `{summary['generated_at_utc']}`",
        f"Best step: `{summary.get('best_step')}`",
        "",
        "## Final Metrics",
        "",
        "| split | rows_scored | model_nll | gnomad_nll | delta_nll | model_brier | gnomad_brier | spearman |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ["val", "test"]:
        payload = summary.get(f"{split}_metrics")
        if not payload:
            continue
        overall = payload["overall"]
        lines.append(
            (
                "| {split} | {rows_scored} | {model_nll:.6f} | {gnomad_nll:.6f} | "
                "{delta_nll:.6f} | {model_brier:.6f} | {gnomad_brier:.6f} | {spearman} |"
            ).format(
                split=split,
                rows_scored=payload["rows_scored"],
                model_nll=overall["model_nll"],
                gnomad_nll=overall["gnomad_nll"],
                delta_nll=overall["delta_nll_model_minus_gnomad"],
                model_brier=overall["model_brier"],
                gnomad_brier=overall["gnomad_brier"],
                spearman=(
                    f"{overall['model_spearman']:.6f}" if overall["model_spearman"] is not None else "NA"
                ),
            )
        )
    lines.extend(["", "## Outputs", ""])
    for key, value in summary["outputs"].items():
        lines.append(f"- `{key}`: `{value}`")
    path.write_text("\n".join(lines) + "\n")


def run(config: TrainConfig) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not config.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}. Pass --overwrite to replace outputs.")
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(config.seed)
    requested_device = config.device
    if requested_device == "cuda" and not torch.cuda.is_available():
        requested_device = "cpu"
    device = torch.device(requested_device)
    configure_float32_precision(config.allow_tf32)
    precision_policy = resolve_precision_policy(device, config.precision)

    train_examples = [] if config.eval_only else load_frequency_examples(
        Path(config.train_parquet),
        limit=config.max_train_rows,
        strategy=config.row_sample_strategy,
        seed=config.seed,
    )
    val_examples = load_frequency_examples(
        Path(config.val_parquet),
        limit=config.max_val_rows,
        strategy=config.row_sample_strategy,
        seed=config.seed + 1,
    )
    test_examples = (
        load_frequency_examples(
            Path(config.test_parquet),
            limit=config.max_test_rows,
            strategy=config.row_sample_strategy,
            seed=config.seed + 2,
        )
        if config.test_parquet is not None
        else []
    )
    if not config.eval_only and not train_examples:
        raise ValueError("No training examples loaded.")
    if not val_examples:
        raise ValueError("No validation examples loaded.")

    model, lora_summary = build_model(config, device, precision_policy)
    param_counts = count_trainable_parameters(model)
    precision_payload = precision_metadata(precision_policy)

    if config.adapter_checkpoint is not None:
        load_adapter_checkpoint(Path(config.adapter_checkpoint), model, device)

    train_history: list[dict[str, Any]] = []
    best_val_nll = math.inf
    best_step = 0
    best_metrics: dict[str, Any] | None = None

    if not config.eval_only:
        optimizer = make_optimizer(model, config)
        warmup_steps = int(config.max_steps * config.warmup_fraction)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: _cosine_lr_scale(
                step,
                total_steps=config.max_steps,
                warmup_steps=warmup_steps,
            ),
        )
        train_loader = build_loader(
            train_examples,
            config=config,
            target_column=config.target_column,
            shuffle=True,
            batch_size=config.batch_size,
        )
        loader_iter = iter(train_loader)
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        running_items = 0
        start_time = time.time()

        for step in range(1, config.max_steps + 1):
            model.train()
            step_loss_value = 0.0
            step_items = 0
            for _ in range(config.grad_accum_steps):
                try:
                    batch = next(loader_iter)
                except StopIteration:
                    loader_iter = iter(train_loader)
                    batch = next(loader_iter)
                if batch is None:
                    continue
                batch = _move_batch_to_device(batch, device)
                with autocast_context(precision_policy):
                    logits = model(
                        batch["ref_seqs"],
                        batch["alt_seqs"],
                        batch["variant_offsets"],
                        batch["ref_alleles"],
                        batch["alt_alleles"],
                        batch.get("scalar_features"),
                    )
                if config.loss == "bce":
                    loss = F.binary_cross_entropy_with_logits(logits.float(), batch["targets"])
                elif config.loss == "huber":
                    loss = F.smooth_l1_loss(logits.float(), batch["targets"])
                else:  # mse
                    loss = F.mse_loss(logits.float(), batch["targets"])
                (loss / config.grad_accum_steps).backward()
                batch_size = int(batch["targets"].shape[0])
                step_loss_value += float(loss.detach().cpu()) * batch_size
                step_items += batch_size

            if step_items == 0:
                continue
            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    config.grad_clip,
                )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            mean_step_loss = step_loss_value / step_items
            running_loss += step_loss_value
            running_items += step_items

            should_log = config.log_every_steps > 0 and (step == 1 or step % config.log_every_steps == 0)
            should_eval = config.eval_every_steps > 0 and (step == 1 or step % config.eval_every_steps == 0)
            if should_log:
                elapsed = max(time.time() - start_time, 1e-6)
                print(
                    json.dumps(
                        {
                            "event": "train_step",
                            "step": step,
                            "loss": mean_step_loss,
                            "running_loss": running_loss / max(running_items, 1),
                            "items": running_items,
                            "items_per_sec": running_items / elapsed,
                            "lr": [group["lr"] for group in optimizer.param_groups],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

            if should_eval:
                val_metrics = evaluate_split(
                    name="val",
                    model=model,
                    examples=val_examples,
                    config=config,
                    device=device,
                    precision_policy=precision_policy,
                    output_dir=output_dir,
                    write_predictions=False,
                )
                current_nll = float(val_metrics["overall"]["model_nll"])
                train_history.append({"step": step, "train_loss": mean_step_loss, "val_metrics": val_metrics})
                print(
                    json.dumps(
                        {
                            "event": "validation",
                            "step": step,
                            "model_nll": current_nll,
                            "gnomad_nll": val_metrics["overall"]["gnomad_nll"],
                            "model_brier": val_metrics["overall"]["model_brier"],
                            "model_spearman": val_metrics["overall"]["model_spearman"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                if current_nll < best_val_nll:
                    best_val_nll = current_nll
                    best_step = step
                    best_metrics = val_metrics
                    save_adapter_checkpoint(
                        output_dir / "best_adapter.pt",
                        model=model,
                        config=config,
                        lora_summary=lora_summary,
                        metrics=val_metrics,
                        step=step,
                        precision=precision_payload,
                    )

            if config.save_every_steps > 0 and step % config.save_every_steps == 0:
                save_adapter_checkpoint(
                    output_dir / f"adapter_step_{step:06d}.pt",
                    model=model,
                    config=config,
                    lora_summary=lora_summary,
                    metrics=train_history[-1] if train_history else {},
                    step=step,
                    precision=precision_payload,
                )

        save_adapter_checkpoint(
            output_dir / "final_adapter.pt",
            model=model,
            config=config,
            lora_summary=lora_summary,
            metrics=train_history[-1] if train_history else {},
            step=config.max_steps,
            precision=precision_payload,
        )
        best_path = output_dir / "best_adapter.pt"
        if best_path.exists():
            load_adapter_checkpoint(best_path, model, device)

    val_metrics = evaluate_split(
        name="val",
        model=model,
        examples=val_examples,
        config=config,
        device=device,
        precision_policy=precision_policy,
        output_dir=output_dir,
        write_predictions=config.write_predictions,
    )
    test_metrics = None
    if test_examples:
        test_metrics = evaluate_split(
            name="test",
            model=model,
            examples=test_examples,
            config=config,
            device=device,
            precision_policy=precision_policy,
            output_dir=output_dir,
            write_predictions=config.write_predictions,
        )

    summary_path = output_dir / "summary.json"
    report_path = output_dir / "README.md"
    summary: dict[str, Any] = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "config": asdict(config),
        "precision": precision_payload,
        "trainable_parameters": param_counts,
        "lora": asdict(lora_summary) if lora_summary is not None else None,
        "rows_loaded": {
            "train": len(train_examples),
            "val": len(val_examples),
            "test": len(test_examples),
        },
        "best_step": best_step,
        "best_val_metrics": best_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "train_history": train_history,
        "outputs": {
            "output_dir": str(output_dir),
            "best_adapter": str(output_dir / "best_adapter.pt"),
            "final_adapter": str(output_dir / "final_adapter.pt"),
            "summary": str(summary_path),
            "report": str(report_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_report(report_path, summary)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--train-parquet", type=Path)
    parser.add_argument("--val-parquet", type=Path)
    parser.add_argument("--test-parquet", type=Path)
    parser.add_argument("--fasta", type=Path, default=DEFAULT_FASTA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_BASE_CHECKPOINT)
    parser.add_argument("--model-family", default="lumina")
    parser.add_argument("--model-version", default="beat-v10")
    parser.add_argument("--context-size", type=int, default=4096)
    parser.add_argument(
        "--target-column",
        choices=["af_abraom", "scrambled_af_abraom", "af_gnomad", "delta_logit", "scrambled_delta_logit"],
        default="af_abraom",
    )
    parser.add_argument(
        "--metric-target-column",
        choices=["af_abraom", "scrambled_af_abraom", "af_gnomad", "delta_logit", "scrambled_delta_logit"],
        default="af_abraom",
    )
    parser.add_argument("--use-gnomad-prior", action="store_true")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--max-val-rows", type=int)
    parser.add_argument("--max-test-rows", type=int)
    parser.add_argument("--row-sample-strategy", choices=["head", "random", "balanced-af"], default="head")
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--eval-every-steps", type=int, default=500)
    parser.add_argument("--save-every-steps", type=int, default=0)
    parser.add_argument("--lr-lora", type=float, default=5e-6)
    parser.add_argument("--lr-head", type=float, default=5e-4)
    parser.add_argument("--weight-decay-lora", type=float, default=0.01)
    parser.add_argument("--weight-decay-head", type=float, default=1e-4)
    parser.add_argument("--warmup-fraction", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=0.5)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--train-layernorm", action="store_true")
    parser.add_argument("--head-hidden-dim", type=int, default=256)
    parser.add_argument("--head-dropout", type=float, default=0.1)
    parser.add_argument("--precision", choices=["auto", "mxfp8", "bf16", "fp32"], default="auto")
    parser.add_argument(
        "--loss", choices=["bce", "mse", "huber"], default="bce",
        help="bce: [0,1] frequency target (default). mse/huber: regression for unbounded "
             "targets like delta_logit (the ABRAOM-vs-gnomAD residual). Spearman metric is "
             "rank-based so it is unaffected; Brier/NLL are meaningless for non-bce runs.",
    )
    parser.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-fraction-max", type=float, default=0.25)
    parser.add_argument("--log-every-steps", type=int, default=25)
    parser.add_argument("--write-predictions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--adapter-checkpoint", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    train_parquet = args.train_parquet or args.data_dir / "abraom_frequency_train.parquet"
    val_parquet = args.val_parquet or args.data_dir / "abraom_frequency_val.parquet"
    test_parquet = args.test_parquet or args.data_dir / "abraom_frequency_test.parquet"
    return TrainConfig(
        train_parquet=str(train_parquet),
        val_parquet=str(val_parquet),
        test_parquet=str(test_parquet) if test_parquet is not None else None,
        fasta=str(args.fasta),
        output_dir=str(args.output_dir),
        checkpoint_path=str(args.checkpoint_path),
        model_family=args.model_family,
        model_version=args.model_version,
        context_size=args.context_size,
        target_column=args.target_column,
        metric_target_column=args.metric_target_column,
        use_gnomad_prior=args.use_gnomad_prior,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        max_train_rows=args.max_train_rows,
        max_val_rows=args.max_val_rows,
        max_test_rows=args.max_test_rows,
        row_sample_strategy=args.row_sample_strategy,
        max_steps=args.max_steps,
        grad_accum_steps=args.grad_accum_steps,
        eval_every_steps=args.eval_every_steps,
        save_every_steps=args.save_every_steps,
        lr_lora=args.lr_lora,
        lr_head=args.lr_head,
        weight_decay_lora=args.weight_decay_lora,
        weight_decay_head=args.weight_decay_head,
        warmup_fraction=args.warmup_fraction,
        grad_clip=args.grad_clip,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        train_layernorm=args.train_layernorm,
        head_hidden_dim=args.head_hidden_dim,
        head_dropout=args.head_dropout,
        precision=args.precision,
        allow_tf32=args.allow_tf32,
        device=args.device,
        seed=args.seed,
        n_fraction_max=args.n_fraction_max,
        log_every_steps=args.log_every_steps,
        write_predictions=args.write_predictions,
        eval_only=args.eval_only,
        adapter_checkpoint=str(args.adapter_checkpoint) if args.adapter_checkpoint else None,
        overwrite=args.overwrite,
        loss=args.loss,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    summary = run(config_from_args(args))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
