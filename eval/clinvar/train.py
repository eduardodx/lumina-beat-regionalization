"""Training loop for ClinVar fine-tuning with DDP support.

Handles optimizer setup, gradient accumulation, backbone freeze schedule,
cosine LR with warmup, distributed evaluation, and artifact saving.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from eval.clinvar.adapters import build_finetune_adapter
from eval.clinvar.config import FineTuneConfig
from eval.clinvar.dataset import (
    ClinVarFineTuneDataset,
    build_variant_cache,
    clinvar_collate_fn,
    load_variant_cache,
    resolve_variant_cache_path,
    stratified_val_split,
)
from eval.clinvar.diagnostics import cross_gene_diagnostic_report
from eval.clinvar.fusion_lora import (
    AdapterState,
    FusionSummary,
    collect_fusion_gate_diagnostics,
    convert_lora_backbone_to_dynamic_fusion,
    convert_lora_backbone_to_static_fusion,
    freeze_backbone_for_static_fusion,
    load_adapter_state,
    load_matching_finetuned_state,
)
from eval.clinvar.lora import (
    apply_lora,
    count_trainable_parameters,
    enable_layernorm_training,
    freeze_native_feature_heads,
)
from eval.clinvar.losses import build_loss, pairwise_ranking_loss, swap_consistency_loss
from eval.clinvar.metrics import (
    binary_auprc,
    binary_log_loss,
    binary_roc_auc,
    brier_score,
    classification_metrics,
    optimize_threshold,
)
from eval.clinvar.model import EndToEndClinVarModel
from src.precision import (
    PrecisionPolicy,
    autocast_context,
    configure_float32_precision,
    resolve_precision_policy,
)

log = logging.getLogger(__name__)

TRAIN_LOG_EVERY_STEPS = 10


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------


def _is_distributed() -> bool:
    return dist.is_initialized()


def _rank() -> int:
    return dist.get_rank() if _is_distributed() else 0


def _world_size() -> int:
    return dist.get_world_size() if _is_distributed() else 1


def _is_main() -> bool:
    return _rank() == 0


def setup_distributed() -> bool:
    """Initialize DDP if torchrun environment variables are set."""
    rank = os.environ.get("RANK")
    local_rank = os.environ.get("LOCAL_RANK")
    world_size = os.environ.get("WORLD_SIZE")
    if rank is None or local_rank is None or world_size is None:
        return False

    torch.cuda.set_device(int(local_rank))
    dist.init_process_group(backend="nccl")
    log.info("DDP initialized: rank=%s local_rank=%s world_size=%s", rank, local_rank, world_size)
    return True


def cleanup_distributed() -> None:
    if _is_distributed():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------


def _build_cosine_schedule(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
    freeze_backbone_steps: int,
    min_lr_ratio: float = 0.01,
) -> torch.optim.lr_scheduler.LambdaLR:
    def _base(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress))

    def backbone_lambda(step: int) -> float:
        if step < freeze_backbone_steps:
            return 0.0
        return _base(step)

    def task_lambda(step: int) -> float:
        return _base(step)

    # Order matches optimizer param groups built in run_finetune:
    #   [0]=lora_params, [1]=norm_params, [2]=task_params
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=[backbone_lambda, backbone_lambda, task_lambda],
    )


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


@torch.no_grad()
def _predict(
    model: EndToEndClinVarModel,
    dataloader: DataLoader,
    device: torch.device,
    precision: PrecisionPolicy,
) -> dict[str, np.ndarray]:
    """Run inference and collect probabilities, labels, indices."""
    model.eval()
    all_probs: list[Tensor] = []
    all_swap_probs: list[Tensor] = []
    all_molecular_probs: list[Tensor] = []
    all_regional_discounts: list[Tensor] = []
    all_gate_alpha: dict[str, list[Tensor]] = {}
    all_gate_entropy: list[Tensor] = []
    all_labels: list[Tensor] = []
    all_indices: list[Tensor] = []

    for batch in dataloader:
        bio = batch.get("bio_features")
        if bio is not None:
            bio = bio.to(device)
        with autocast_context(precision):
            logits = model(
                batch["ref_seqs"], batch["alt_seqs"], batch["variant_offsets"],
                bio_features=bio,
            )
            components = getattr(model.head, "last_components", {})
            molecular_logit = components.get("molecular_logit") if isinstance(components, dict) else None
            regional_discount = components.get("regional_discount") if isinstance(components, dict) else None
            gate_diagnostics = collect_fusion_gate_diagnostics(model)
            swap_logits = model(
                batch["alt_seqs"],
                batch["ref_seqs"],
                batch["variant_offsets"],
                ref_alleles=batch["alt_alleles"],
                alt_alleles=batch["ref_alleles"],
                bio_features=bio,
            )
        all_probs.append(torch.sigmoid(logits.float()).cpu())
        all_swap_probs.append(torch.sigmoid(swap_logits.float()).cpu())
        if molecular_logit is not None and regional_discount is not None:
            all_molecular_probs.append(torch.sigmoid(molecular_logit.float()).cpu())
            all_regional_discounts.append(regional_discount.float().cpu())
        if gate_diagnostics is not None:
            adapter_names = gate_diagnostics["adapter_names"]
            gate_alpha = gate_diagnostics["gate_alpha"]
            gate_entropy = gate_diagnostics["gate_entropy"]
            if isinstance(adapter_names, tuple) and isinstance(gate_alpha, Tensor) and isinstance(gate_entropy, Tensor):
                for idx, adapter_name in enumerate(adapter_names):
                    all_gate_alpha.setdefault(str(adapter_name), []).append(gate_alpha[:, idx].cpu())
                all_gate_entropy.append(gate_entropy.cpu())
        all_labels.append(batch["labels"])
        all_indices.append(batch["original_indices"])

    if not all_probs:
        return {
            "probs": np.asarray([], dtype=np.float32),
            "swap_probs": np.asarray([], dtype=np.float32),
            "labels": np.asarray([], dtype=np.float32),
            "indices": np.asarray([], dtype=np.int64),
        }

    result = {
        "probs": torch.cat(all_probs).numpy(),
        "swap_probs": torch.cat(all_swap_probs).numpy(),
        "labels": torch.cat(all_labels).numpy(),
        "indices": torch.cat(all_indices).numpy(),
    }
    if all_molecular_probs and all_regional_discounts:
        result["molecular_probs"] = torch.cat(all_molecular_probs).numpy()
        result["regional_discounts"] = torch.cat(all_regional_discounts).numpy()
    for adapter_name, chunks in all_gate_alpha.items():
        if chunks:
            result[f"gate_alpha_{adapter_name}"] = torch.cat(chunks).numpy()
    if all_gate_entropy:
        result["gate_entropy"] = torch.cat(all_gate_entropy).numpy()
    return result


def _gather_predictions(local: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Gather predictions across DDP ranks and deduplicate."""
    if not _is_distributed():
        return local

    world = _world_size()
    device = torch.device(f"cuda:{os.environ.get('LOCAL_RANK', 0)}")

    gathered: dict[str, np.ndarray] = {}
    for key in local:
        local_tensor = torch.from_numpy(local[key]).to(device)
        all_tensors = [torch.zeros_like(local_tensor) for _ in range(world)]
        dist.all_gather(all_tensors, local_tensor)
        gathered[key] = torch.cat(all_tensors).cpu().numpy()

    # Deduplicate by original index
    _, first_occurrence = np.unique(gathered["indices"], return_index=True)
    return {k: v[first_occurrence] for k, v in gathered.items()}


def _swap_diagnostics(preds: dict[str, np.ndarray]) -> dict[str, float]:
    swap_probs = preds.get("swap_probs")
    if swap_probs is None or len(swap_probs) == 0:
        return {
            "swap_mean_abs_prob_sum_minus_one": 0.0,
            "swap_directional_fraction": 0.0,
        }
    probs = preds["probs"]
    return {
        "swap_mean_abs_prob_sum_minus_one": float(np.mean(np.abs(probs + swap_probs - 1.0))),
        "swap_directional_fraction": float(np.mean(probs > swap_probs)),
    }


def _consequence_stratified_metrics(
    variants: list[Any],
    preds: dict[str, np.ndarray],
    threshold: float,
) -> dict[str, dict[str, float]]:
    bucket_by_index = {int(variant.index): getattr(variant, "consequence_bucket", "unknown") for variant in variants}
    buckets = np.asarray([bucket_by_index.get(int(index), "unknown") for index in preds["indices"]])
    result: dict[str, dict[str, float]] = {}
    for bucket in sorted(set(buckets.tolist())):
        mask = buckets == bucket
        if not np.any(mask):
            continue
        metrics = classification_metrics(preds["labels"][mask], preds["probs"][mask], threshold)
        result[str(bucket)] = {
            "count": float(mask.sum()),
            "mcc": float(metrics["mcc"]),
            "f1": float(metrics["f1"]),
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
        }
        labels = preds["labels"][mask]
        if len(np.unique(labels)) > 1:
            result[str(bucket)]["auroc"] = float(binary_roc_auc(labels, preds["probs"][mask]))
    return result


# ---------------------------------------------------------------------------
# Pos weight resolution
# ---------------------------------------------------------------------------


def _resolve_pos_weight(config_value: str | float, labels: np.ndarray) -> float:
    if isinstance(config_value, str) and config_value.lower() == "auto":
        n_pos = int((labels == 1).sum())
        n_neg = int((labels == 0).sum())
        return n_neg / max(n_pos, 1)
    return float(config_value)


def _build_eval_loader(dataset: ClinVarFineTuneDataset, config: FineTuneConfig) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=clinvar_collate_fn,
        num_workers=config.num_workers,
        drop_last=False,
    )


def _load_checkpoint_bundle(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        bundle = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        bundle = torch.load(path, map_location=device)
    if not isinstance(bundle, dict):
        raise TypeError(f"Checkpoint {path} must contain a dict, got {type(bundle).__name__}.")
    return bundle


def _raise_final_eval_error(error: tuple[str, str]) -> None:
    error_type, message = error
    if error_type == "FileNotFoundError":
        raise FileNotFoundError(message)
    if error_type == "ValueError":
        raise ValueError(message)
    raise RuntimeError(f"{error_type}: {message}")


def _load_best_checkpoint(
    raw_model: EndToEndClinVarModel,
    output_dir: Path,
    device: torch.device,
    best_epoch: int,
) -> None:
    """Load the best checkpoint weights into the model. Safe to call on all ranks."""
    best_ckpt = output_dir / "best_model.pt"
    if not best_ckpt.is_file():
        raise FileNotFoundError(f"Best checkpoint not found at {best_ckpt}")

    state = _load_checkpoint_bundle(best_ckpt, device)
    model_state = state.get("model_state_dict")
    if not isinstance(model_state, dict):
        raise ValueError(f"Checkpoint {best_ckpt} is missing `model_state_dict`.")
    raw_model.load_state_dict(model_state)
    log.info("Loaded best model from epoch %d", best_epoch)


def _compute_and_save_final_metrics(
    *,
    val_preds: dict[str, np.ndarray],
    test_preds: dict[str, np.ndarray],
    raw_model: EndToEndClinVarModel,
    config: FineTuneConfig,
    output_dir: Path,
    train_variants: list[Any],
    val_variants: list[Any],
    test_variants: list[Any],
    pos_weight_val: float,
    best_epoch: int,
    wandb_run: Any | None,
    precision: PrecisionPolicy,
) -> dict[str, Any]:
    """Compute final metrics and save artifacts. Rank-0 only."""
    if len(val_preds["labels"]) > 0:
        val_optimal_threshold, val_threshold_metrics = optimize_threshold(
            val_preds["labels"],
            val_preds["probs"],
            "mcc",
        )
        val_auroc = binary_roc_auc(val_preds["labels"], val_preds["probs"])
        val_auprc = binary_auprc(val_preds["labels"], val_preds["probs"])
    else:
        val_optimal_threshold = 0.5
        val_threshold_metrics = classification_metrics(
            np.asarray([], dtype=np.int64),
            np.asarray([], dtype=np.float64),
            0.5,
        )
        val_auroc = float("nan")
        val_auprc = float("nan")

    auroc = binary_roc_auc(test_preds["labels"], test_preds["probs"])
    auprc = binary_auprc(test_preds["labels"], test_preds["probs"])
    logloss = binary_log_loss(test_preds["labels"], test_preds["probs"])
    brier = brier_score(test_preds["labels"], test_preds["probs"])
    test_at_val_threshold = classification_metrics(test_preds["labels"], test_preds["probs"], val_optimal_threshold)
    test_at_default_threshold = classification_metrics(test_preds["labels"], test_preds["probs"], 0.5)
    test_optimal_threshold, test_optimal_metrics = optimize_threshold(test_preds["labels"], test_preds["probs"], "mcc")
    cross_gene_diagnostic = cross_gene_diagnostic_report(
        test_variants,
        test_preds,
        threshold=val_optimal_threshold,
        seed=config.seed,
    )
    swap_diagnostics = _swap_diagnostics(test_preds)
    consequence_metrics = _consequence_stratified_metrics(test_variants, test_preds, val_optimal_threshold)

    results = {
        "metric": f"clinvar_finetune_regime_{config.regime}",
        "value": test_at_val_threshold["mcc"],
        "regime": config.regime,
        "model_family": config.model_family,
        "model_version": config.model_version,
        "context_size": config.context_size,
        "loss_type": config.loss_type,
        "requested_precision": precision.requested,
        "resolved_precision": precision.resolved,
        "allow_tf32": config.allow_tf32,
        "auroc": auroc,
        "auprc": auprc,
        "mcc": test_at_val_threshold["mcc"],
        "f1": test_at_val_threshold["f1"],
        "balanced_accuracy": test_at_val_threshold["balanced_accuracy"],
        "precision": test_at_val_threshold["precision"],
        "recall": test_at_val_threshold["recall"],
        "specificity": test_at_val_threshold["specificity"],
        "brier_score": brier,
        "log_loss": logloss,
        "val_mcc": val_threshold_metrics["mcc"],
        "val_auprc": val_auprc,
        "val_auroc": val_auroc,
        "val_optimal_threshold": val_optimal_threshold,
        "best_epoch": best_epoch,
        "test_at_val_threshold": test_at_val_threshold,
        "test_at_default_threshold": test_at_default_threshold,
        "test_at_test_optimal_threshold_LEAKY": {
            **test_optimal_metrics,
            "threshold": test_optimal_threshold,
        },
        "cross_gene_diagnostic": cross_gene_diagnostic,
        "swap_diagnostics": swap_diagnostics,
        "consequence_stratified_metrics": consequence_metrics,
        "hyperparameters": {
            "lora_rank": config.lora_rank,
            "lora_alpha": config.lora_alpha,
            "lora_dropout": config.lora_dropout,
            "lr_backbone": config.lr_backbone,
            "lr_head": config.lr_head,
            "freeze_backbone_steps": config.freeze_backbone_steps,
            "grad_clip": config.grad_clip,
            "warmup_fraction": config.warmup_fraction,
            "batch_size": config.batch_size,
            "grad_accum_steps": config.grad_accum_steps,
            "focal_gamma": config.focal_gamma if config.loss_type == "focal" else None,
            "pairwise_rank_weight": config.pairwise_rank_weight,
            "pairwise_rank_margin": config.pairwise_rank_margin,
            "swap_consistency_weight": config.swap_consistency_weight,
            "swap_consistency_margin": config.swap_consistency_margin,
            "explicit_feature_columns": list(config.explicit_feature_columns),
            "init_finetuned_checkpoint_path": config.init_finetuned_checkpoint_path,
            "fusion_mode": config.fusion_mode,
            "fusion_adapter_names": list(config.fusion_adapter_names),
            "freeze_backbone_for_fusion": config.freeze_backbone_for_fusion,
            "fusion_gate_hidden_dim": config.fusion_gate_hidden_dim,
            "fusion_gate_dropout": config.fusion_gate_dropout,
        },
        "architecture": {
            "proj_dim": config.proj_dim,
            "hidden_dim": config.hidden_dim,
            "head_dropout": config.head_dropout,
            "d_model": raw_model.adapter.d_model,
            "head_type": config.head_type,
            "explicit_feature_dim": len(config.explicit_feature_columns),
            "fusion_mode": config.fusion_mode,
            "fusion_gate_hidden_dim": config.fusion_gate_hidden_dim,
        },
        "trainable_params": count_trainable_parameters(raw_model),
        "data_summary": {
            "train_size": len(train_variants),
            "val_size": len(val_variants),
            "test_size": len(test_variants),
            "pos_weight": pos_weight_val,
        },
    }

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Saved metrics to %s", metrics_path)

    try:
        import pandas as pd

        pred_df = pd.DataFrame(
            {
                "original_index": test_preds["indices"].astype(int),
                "label": test_preds["labels"].astype(int),
                "probability": test_preds["probs"].astype(float),
                "swap_probability": test_preds.get("swap_probs", np.zeros_like(test_preds["probs"])).astype(float),
            }
        )
        if "molecular_probs" in test_preds:
            pred_df["molecular_probability"] = test_preds["molecular_probs"].astype(float)
        if "regional_discounts" in test_preds:
            pred_df["regional_discount"] = test_preds["regional_discounts"].astype(float)
        for key, value in sorted(test_preds.items()):
            if key.startswith("gate_alpha_"):
                pred_df[key] = value.astype(float)
        if "gate_entropy" in test_preds:
            pred_df["gate_entropy"] = test_preds["gate_entropy"].astype(float)
        pred_path = output_dir / "test_predictions.parquet"
        pred_df.to_parquet(pred_path, index=False)
        log.info("Saved predictions to %s", pred_path)
    except Exception as exc:
        log.warning("Failed to save predictions: %s", exc)

    if wandb_run:
        wandb_run.log(
            {
                "final/auroc": auroc,
                "final/auprc": auprc,
                "final/mcc": test_at_val_threshold["mcc"],
                "final/val_mcc": val_threshold_metrics["mcc"],
            }
        )
        wandb_run.finish()

    return results


def _prepare_fusion(
    model: EndToEndClinVarModel,
    config: FineTuneConfig,
    *,
    device: torch.device,
) -> tuple[dict[str, int] | None, FusionSummary | None]:
    init_summary: dict[str, int] | None = None
    fusion_summary: FusionSummary | None = None
    if config.init_finetuned_checkpoint_path:
        init_summary = load_matching_finetuned_state(
            model,
            Path(config.init_finetuned_checkpoint_path),
            map_location=device,
        )

    if config.fusion_mode in ("static_lora", "dynamic_lora"):
        adapters: list[AdapterState] = []
        for adapter_name, adapter_path in zip(
            config.fusion_adapter_names,
            config.fusion_adapter_paths,
            strict=True,
        ):
            loaded = load_adapter_state(Path(adapter_path), map_location="cpu")
            adapters.append(
                AdapterState(
                    name=adapter_name,
                    state_dict=loaded.state_dict,
                    scaling=loaded.scaling,
                )
            )
        if config.fusion_mode == "static_lora":
            fusion_summary = convert_lora_backbone_to_static_fusion(model.backbone, adapters=adapters)
        else:
            fusion_summary = convert_lora_backbone_to_dynamic_fusion(
                model.backbone,
                adapters=adapters,
                gate_hidden_dim=config.fusion_gate_hidden_dim,
                gate_dropout=config.fusion_gate_dropout,
            )
        if config.freeze_backbone_for_fusion:
            freeze_backbone_for_static_fusion(model)
    return init_summary, fusion_summary


# ---------------------------------------------------------------------------
# Main training entry
# ---------------------------------------------------------------------------


def run_finetune(config: FineTuneConfig) -> dict[str, Any]:
    """Train and evaluate a ClinVar fine-tuning run.

    Returns a metrics dict with the primary metric under "value" (MCC).
    """
    config.validate()
    distributed = setup_distributed()
    device = torch.device(config.device if not distributed else f"cuda:{os.environ.get('LOCAL_RANK', 0)}")
    configure_float32_precision(config.allow_tf32)
    precision_policy = resolve_precision_policy(device, config.precision)
    log.info(
        "ClinVar precision configured: requested=%s resolved=%s allow_tf32=%s",
        precision_policy.requested,
        precision_policy.resolved,
        config.allow_tf32,
    )

    output_dir = Path(config.output_dir)
    if _is_main():
        output_dir.mkdir(parents=True, exist_ok=True)

    # -- Check for cached results --
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists() and not config.overwrite:
        log.info("Results already exist: %s", metrics_path)
        with open(metrics_path) as f:
            return json.load(f)

    # -- Build variant cache --
    dataset_path = Path(config.dataset_path)
    fasta_path = Path(config.fasta_path)
    cache_dir = Path(config.cache_dir) if config.cache_dir else None
    cache_path = resolve_variant_cache_path(
        dataset_path,
        config.context_size,
        config.regime,
        cache_dir,
        explicit_feature_columns=config.explicit_feature_columns,
    )

    if _is_main():
        cache_path = build_variant_cache(
            dataset_path, fasta_path, config.context_size, config.regime,
            phylo100_bw_path=config.phylo100_bw_path,
            phylo470_bw_path=config.phylo470_bw_path,
            gtf_path=config.gtf_path,
            cache_dir=cache_dir,
            explicit_feature_columns=config.explicit_feature_columns,
        )
    if distributed:
        dist.barrier()

    train_variants, test_variants = load_variant_cache(cache_path, config.regime)
    train_variants, val_variants = stratified_val_split(train_variants, config.val_fraction, config.seed)

    train_dataset = ClinVarFineTuneDataset(train_variants)
    val_dataset = ClinVarFineTuneDataset(val_variants)
    test_dataset = ClinVarFineTuneDataset(test_variants)

    # -- Build model --
    adapter = build_finetune_adapter(
        config.model_family, config.model_version, device,
        checkpoint_path=config.checkpoint_path, precision=precision_policy,
        native_feature_heads=config.native_feature_heads,
    )
    model = EndToEndClinVarModel(
        adapter, regime=config.regime,
        proj_dim=config.proj_dim,
        hidden_dim=config.hidden_dim,
        dropout=config.head_dropout,
        head_type=config.head_type,
        explicit_feature_dim=len(config.explicit_feature_columns),
    )
    model.to(device)

    # -- Apply LoRA --
    lora_summary = apply_lora(
        model.backbone, rank=config.lora_rank,
        alpha=config.lora_alpha, dropout=config.lora_dropout,
    )
    enable_layernorm_training(model.backbone)

    native_selection = getattr(adapter, "native_variant_head_selection", None)
    if native_selection:
        freeze_native_feature_heads(model.backbone, native_selection)

    init_summary, fusion_summary = _prepare_fusion(model, config, device=device)
    if init_summary is not None:
        log.info("Initial fine-tuned checkpoint load summary: %s", init_summary)
    if fusion_summary is not None:
        log.info(
            "Fusion summary: mode=%s adapters=%s modules=%d gate_params=%d",
            fusion_summary.mode,
            list(fusion_summary.adapter_names),
            fusion_summary.module_count,
            fusion_summary.trainable_gate_params,
        )

    param_counts = count_trainable_parameters(model)
    log.info("Trainable parameters: %s", param_counts)

    # -- DDP wrapping --
    if distributed:
        # static_graph=True is required because beat-v5/v6 wrap mixers with
        # torch.utils.checkpoint(..., use_reentrant=True); without it the
        # reentrant re-forward during backward double-fires DDP's reducer
        # hooks. Mirrors src/train.py:1897.
        model = DDP(model, device_ids=[device.index], static_graph=True)  # type: ignore[assignment]

    raw_model: EndToEndClinVarModel = model.module if isinstance(model, DDP) else model  # type: ignore[assignment]

    # -- DataLoaders --
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if distributed and len(val_dataset) > 0 else None
    test_sampler = DistributedSampler(test_dataset, shuffle=False) if distributed else None

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size,
        sampler=train_sampler, shuffle=(train_sampler is None),
        collate_fn=clinvar_collate_fn, num_workers=config.num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size,
        sampler=val_sampler, shuffle=False,
        collate_fn=clinvar_collate_fn, num_workers=config.num_workers,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size,
        sampler=test_sampler, shuffle=False,
        collate_fn=clinvar_collate_fn, num_workers=config.num_workers,
        drop_last=False,
    )

    # -- Loss --
    labels_arr = train_dataset.labels
    pos_weight_val = _resolve_pos_weight(config.pos_weight, labels_arr)
    pos_weight_tensor = None if math.isclose(pos_weight_val, 1.0) else torch.tensor([pos_weight_val], device=device)
    criterion = build_loss(config.loss_type, gamma=config.focal_gamma, pos_weight=pos_weight_tensor)

    # -- Optimizer --
    lora_params = [p for n, p in raw_model.backbone.named_parameters() if "lora_" in n and p.requires_grad]
    norm_params = [
        p for n, p in raw_model.backbone.named_parameters()
        if "lora_" not in n and p.requires_grad
    ]
    head_params = list(raw_model.head.parameters())

    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": config.lr_backbone, "weight_decay": config.wd_backbone},
        {"params": norm_params, "lr": config.lr_backbone, "weight_decay": config.wd_backbone},
        {"params": head_params, "lr": config.lr_head, "weight_decay": config.wd_head},
    ])

    if config.freeze_backbone_steps > 0:
        log.info("Backbone LR held at 0 for first %d steps", config.freeze_backbone_steps)

    steps_per_epoch = math.ceil(len(train_loader) / config.grad_accum_steps)
    total_steps = steps_per_epoch * config.max_epochs
    warmup_steps = int(total_steps * config.warmup_fraction)
    scheduler = _build_cosine_schedule(
        optimizer, total_steps, warmup_steps, config.freeze_backbone_steps,
    )

    # -- W&B --
    wandb_run = None
    if config.wandb_enabled and _is_main():
        try:
            import wandb
            wandb_run = wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                name=config.wandb_run_name or f"{config.model_family}_{config.model_version}_regime{config.regime}",
                tags=config.wandb_tags or [],
                config=asdict(config),
            )
        except Exception as exc:
            log.warning("W&B init failed: %s", exc)

    # -- Training loop --
    global_step = 0
    best_mcc = -1.0
    best_epoch = -1

    for epoch in range(config.max_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if val_sampler is not None:
            val_sampler.set_epoch(epoch)

        model.train()
        optimizer.zero_grad()
        epoch_loss = 0.0
        micro_count = 0
        t0 = time.time()

        for batch_idx, batch in enumerate(train_loader):
            labels = batch["labels"].to(device)
            bio = batch.get("bio_features")
            if bio is not None:
                bio = bio.to(device)

            with autocast_context(precision_policy):
                logits = model(
                    batch["ref_seqs"], batch["alt_seqs"], batch["variant_offsets"],
                    bio_features=bio,
                )
                swap_logits = (
                    model(
                        batch["alt_seqs"],
                        batch["ref_seqs"],
                        batch["variant_offsets"],
                        ref_alleles=batch["alt_alleles"],
                        alt_alleles=batch["ref_alleles"],
                        bio_features=bio,
                    )
                    if config.swap_consistency_weight > 0.0
                    else None
                )
            base_loss = criterion(logits.float(), labels)
            rank_loss = (
                pairwise_ranking_loss(logits.float(), labels, margin=config.pairwise_rank_margin)
                if config.pairwise_rank_weight > 0.0
                else (logits * 0.0).sum()
            )
            swap_loss = (
                swap_consistency_loss(
                    logits.float(),
                    swap_logits.float(),
                    labels,
                    margin=config.swap_consistency_margin,
                )
                if swap_logits is not None
                else (logits * 0.0).sum()
            )
            loss = (
                base_loss
                + config.pairwise_rank_weight * rank_loss
                + config.swap_consistency_weight * swap_loss
            ) / config.grad_accum_steps
            loss.backward()
            epoch_loss += loss.item() * config.grad_accum_steps
            micro_count += 1

            if (batch_idx + 1) % config.grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if _is_main() and global_step == config.freeze_backbone_steps and config.freeze_backbone_steps > 0:
                    log.info("Backbone LR released from 0 at step %d", global_step)

                # Logging
                if _is_main() and global_step % TRAIN_LOG_EVERY_STEPS == 0:
                    train_log_payload = {
                        "train/loss": epoch_loss / micro_count,
                        "train/pairwise_rank_loss": float(rank_loss.detach().cpu().item()),
                        "train/lr_backbone": optimizer.param_groups[0]["lr"],
                        "train/lr_head": optimizer.param_groups[2]["lr"],
                        "train/step": global_step,
                    }
                    if config.swap_consistency_weight > 0.0:
                        train_log_payload["train/swap_consistency_loss"] = float(
                            swap_loss.detach().cpu().item()
                        )
                    log.info(
                        "Step %d | loss=%.4f | lr_backbone=%.6g | lr_head=%.6g",
                        global_step,
                        train_log_payload["train/loss"],
                        train_log_payload["train/lr_backbone"],
                        train_log_payload["train/lr_head"],
                    )
                    if wandb_run:
                        wandb_run.log(train_log_payload)

        epoch_time = time.time() - t0
        avg_loss = epoch_loss / max(micro_count, 1)

        # -- Evaluate --
        preds = _predict(raw_model, val_loader, device, precision_policy)
        preds = _gather_predictions(preds)

        if _is_main():
            if len(preds["labels"]) > 0:
                auroc = binary_roc_auc(preds["labels"], preds["probs"])
                auprc = binary_auprc(preds["labels"], preds["probs"])
                _, threshold_metrics = optimize_threshold(preds["labels"], preds["probs"], "mcc")
                mcc = threshold_metrics["mcc"]
            else:
                auroc = float("nan")
                auprc = float("nan")
                threshold_metrics = classification_metrics(
                    np.asarray([], dtype=np.int64),
                    np.asarray([], dtype=np.float64),
                    0.5,
                )
                mcc = -1.0

            log.info(
                "Epoch %d/%d | loss=%.4f | val_AUROC=%.4f | val_AUPRC=%.4f | val_MCC=%.4f | %.1fs",
                epoch + 1, config.max_epochs, avg_loss, auroc, auprc, mcc, epoch_time,
            )

            if wandb_run:
                wandb_run.log({
                    "val/auroc": auroc, "val/auprc": auprc, "val/mcc": mcc,
                    "val/f1": threshold_metrics["f1"],
                    "train/epoch_loss": avg_loss, "epoch": epoch + 1,
                })

            if mcc > best_mcc:
                best_mcc = mcc
                best_epoch = epoch + 1
                _save_checkpoint(raw_model, config, lora_summary, output_dir / "best_model.pt", precision_policy)

    # -- Final evaluation (distributed prediction, rank-0 metrics) --
    if distributed:
        best_epoch_tensor = torch.tensor([best_epoch], dtype=torch.long, device=device)
        dist.broadcast(best_epoch_tensor, src=0)
        best_epoch = int(best_epoch_tensor.item())
        dist.barrier()

    # All ranks: reload best checkpoint and run distributed prediction
    _load_best_checkpoint(raw_model, output_dir, device, best_epoch)
    val_preds = _predict(raw_model, val_loader, device, precision_policy)
    val_preds = _gather_predictions(val_preds)
    test_preds = _predict(raw_model, test_loader, device, precision_policy)
    test_preds = _gather_predictions(test_preds)

    # Rank 0 only: compute metrics and save artifacts
    results: dict[str, Any] = {}
    final_error: tuple[str, str] | None = None
    if _is_main():
        try:
            results = _compute_and_save_final_metrics(
                val_preds=val_preds,
                test_preds=test_preds,
                raw_model=raw_model,
                config=config,
                output_dir=output_dir,
                train_variants=train_variants,
                val_variants=val_variants,
                test_variants=test_variants,
                pos_weight_val=pos_weight_val,
                best_epoch=best_epoch,
                wandb_run=wandb_run,
                precision=precision_policy,
            )
        except Exception as exc:
            final_error = (type(exc).__name__, str(exc))

    if distributed:
        error_payload: list[tuple[str, str] | None] = [final_error]
        dist.broadcast_object_list(error_payload, src=0)
        final_error = error_payload[0]
        dist.barrier()

    if final_error is not None:
        cleanup_distributed()
        _raise_final_eval_error(final_error)

    cleanup_distributed()
    return results


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------


def _save_checkpoint(
    model: EndToEndClinVarModel,
    config: FineTuneConfig,
    lora_summary: Any,
    path: Path,
    precision: PrecisionPolicy,
) -> None:
    """Save a reproducible model bundle."""
    trainable_state = {
        name: param.data for name, param in model.named_parameters() if param.requires_grad
    }
    bundle = {
        "format": "clinvar_finetune_v2",
        "regime": config.regime,
        "model_family": config.model_family,
        "model_version": config.model_version,
        "config": asdict(config),
        "lora": {
            "rank": lora_summary.rank,
            "alpha": lora_summary.alpha,
            "dropout": lora_summary.dropout,
            "module_names": list(lora_summary.module_names),
        },
        "fusion": {
            "mode": config.fusion_mode,
            "adapter_names": list(config.fusion_adapter_names),
            "freeze_backbone_for_fusion": config.freeze_backbone_for_fusion,
            "init_finetuned_checkpoint_path": config.init_finetuned_checkpoint_path,
            "gate_hidden_dim": config.fusion_gate_hidden_dim,
            "gate_dropout": config.fusion_gate_dropout,
        },
        "precision": {
            "requested": precision.requested,
            "resolved": precision.resolved,
            "allow_tf32": config.allow_tf32,
        },
        "model_state_dict": model.state_dict(),
        "trainable_state_dict": trainable_state,
    }
    torch.save(bundle, path)
    log.info("Saved checkpoint to %s", path)
