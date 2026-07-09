#!/usr/bin/env python3
"""Evaluate a saved ClinVar fine-tuned model on regional slices without retraining."""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.clinvar.adapters import build_finetune_adapter  # noqa: E402
from eval.clinvar.config import FineTuneConfig  # noqa: E402
from eval.clinvar.dataset import (  # noqa: E402
    CachedVariant,
    ClinVarFineTuneDataset,
    build_variant_cache,
    clinvar_collate_fn,
    infer_consequence_bucket,
)
from eval.clinvar.fusion_lora import (  # noqa: E402
    convert_lora_backbone_to_dynamic_fusion_from_checkpoint_state,
    convert_lora_backbone_to_static_fusion_from_checkpoint_state,
)
from eval.clinvar.lora import (  # noqa: E402
    apply_lora,
    count_trainable_parameters,
    enable_layernorm_training,
    freeze_native_feature_heads,
)
from eval.clinvar.metrics import (  # noqa: E402
    binary_auprc,
    binary_log_loss,
    binary_roc_auc,
    brier_score,
    classification_metrics,
    optimize_threshold,
)
from eval.clinvar.model import EndToEndClinVarModel  # noqa: E402
from eval.clinvar.train import _consequence_stratified_metrics, _predict, _swap_diagnostics  # noqa: E402
from src.precision import configure_float32_precision, resolve_precision_policy  # noqa: E402

DEFAULT_DATASET_FILES = [
    "br_only.parquet",
    "br_any.parquet",
    "mixed_br_nonbr.parquet",
    "regional_benchmark_any.parquet",
    "abraom_present.parquet",
    "abraom_high_specificity.parquet",
    "abraom_common.parquet",
    "abraom_common_benign.parquet",
    "abraom_pathogenic_present.parquet",
    "abraom_pathogenic_common.parquet",
    "global_nonbr_no_abraom.parquet",
    "nonbr_only.parquet",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finetuned-checkpoint", type=Path, default=None)
    parser.add_argument("--finetuned-root", type=Path, default=None)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--dataset-files", nargs="*", default=DEFAULT_DATASET_FILES)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--split-column", default="split_within_gene")
    parser.add_argument("--splits", nargs="*", default=["test", "holdout", "all"])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("auto", "bf16", "fp32"), default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _load_torch_bundle(path: Path, map_location: torch.device | str = "cpu") -> dict[str, Any]:
    try:
        bundle = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        bundle = torch.load(path, map_location=map_location)
    if not isinstance(bundle, dict):
        raise TypeError(f"Expected checkpoint dict at {path}, got {type(bundle).__name__}.")
    return bundle


def _extract_model_tar(root: Path) -> Path:
    model_tar = root / "model.tar.gz"
    if not model_tar.is_file():
        matches = sorted(root.rglob("model.tar.gz"))
        if not matches:
            raise FileNotFoundError(f"Could not find model.tar.gz under {root}")
        model_tar = matches[0]
    extract_dir = root / "_extracted_model"
    marker = extract_dir / ".complete"
    if not marker.exists():
        extract_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(model_tar, "r:gz") as tar:
            tar.extractall(extract_dir)
        marker.write_text("ok\n", encoding="utf-8")
    return extract_dir


def resolve_finetuned_checkpoint(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.finetuned_checkpoint is not None:
        checkpoint = args.finetuned_checkpoint
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Fine-tuned checkpoint not found: {checkpoint}")
        return checkpoint, checkpoint.parent
    if args.finetuned_root is None:
        raise ValueError("Pass either --finetuned-checkpoint or --finetuned-root.")
    root = args.finetuned_root
    direct = root / "best_model.pt"
    if direct.is_file():
        return direct, root
    extracted = _extract_model_tar(root)
    direct = extracted / "best_model.pt"
    if direct.is_file():
        return direct, extracted
    matches = sorted(extracted.rglob("best_model.pt"))
    if not matches:
        raise FileNotFoundError(f"Could not find best_model.pt under {root}")
    return matches[0], matches[0].parent


def _load_training_metrics(root: Path) -> dict[str, Any]:
    for candidate in [root / "metrics.json", *sorted(root.rglob("metrics.json"))]:
        if candidate.is_file():
            return json.loads(candidate.read_text(encoding="utf-8"))
    return {}


def _build_config(
    checkpoint_bundle: dict[str, Any],
    *,
    base_checkpoint: Path,
    fasta: Path,
    output_dir: Path,
    batch_size: int | None,
    num_workers: int,
    device: str,
    precision: str | None,
) -> FineTuneConfig:
    raw_config = checkpoint_bundle.get("config")
    if not isinstance(raw_config, dict):
        raise ValueError("Fine-tuned checkpoint is missing config dict.")
    allowed = {field.name for field in fields(FineTuneConfig)}
    config_kwargs = {key: value for key, value in raw_config.items() if key in allowed}
    config = FineTuneConfig(**config_kwargs)
    config.checkpoint_path = str(base_checkpoint)
    config.fasta_path = str(fasta)
    config.output_dir = str(output_dir)
    config.batch_size = int(batch_size if batch_size is not None else config.batch_size)
    config.num_workers = int(num_workers)
    config.device = device
    if precision is not None:
        config.precision = precision
    return config


def load_model(
    *,
    checkpoint_path: Path,
    base_checkpoint: Path,
    fasta: Path,
    output_dir: Path,
    batch_size: int | None,
    num_workers: int,
    device: torch.device,
    precision_override: str | None,
) -> tuple[EndToEndClinVarModel, FineTuneConfig, Any, dict[str, Any]]:
    bundle = _load_torch_bundle(checkpoint_path, map_location="cpu")
    config = _build_config(
        bundle,
        base_checkpoint=base_checkpoint,
        fasta=fasta,
        output_dir=output_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        device=str(device),
        precision=precision_override,
    )
    config.validate()
    configure_float32_precision(config.allow_tf32)
    precision_policy = resolve_precision_policy(device, config.precision)

    adapter = build_finetune_adapter(
        config.model_family,
        config.model_version,
        device,
        checkpoint_path=str(base_checkpoint),
        precision=precision_policy,
        native_feature_heads=config.native_feature_heads,
    )
    model = EndToEndClinVarModel(
        adapter,
        regime=config.regime,
        proj_dim=config.proj_dim,
        hidden_dim=config.hidden_dim,
        dropout=config.head_dropout,
        head_type=config.head_type,
        explicit_feature_dim=len(config.explicit_feature_columns),
    )
    model.to(device)
    apply_lora(
        model.backbone,
        rank=config.lora_rank,
        alpha=config.lora_alpha,
        dropout=config.lora_dropout,
    )
    enable_layernorm_training(model.backbone)
    native_selection = getattr(adapter, "native_variant_head_selection", None)
    if native_selection:
        freeze_native_feature_heads(model.backbone, native_selection)

    state = bundle.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError(f"Fine-tuned checkpoint {checkpoint_path} is missing model_state_dict.")
    if config.fusion_mode == "static_lora":
        convert_lora_backbone_to_static_fusion_from_checkpoint_state(
            model.backbone,
            model_state_dict=state,
            adapter_names=config.fusion_adapter_names,
        )
    elif config.fusion_mode == "dynamic_lora":
        convert_lora_backbone_to_dynamic_fusion_from_checkpoint_state(
            model.backbone,
            model_state_dict=state,
            adapter_names=config.fusion_adapter_names,
        )
    model.load_state_dict(state)
    model.eval()
    return model, config, precision_policy, bundle


def _row_str(row: pd.Series, key: str, default: str = "") -> str:
    if key in row and pd.notna(row[key]):
        return str(row[key])
    return default


def _variant_from_cache_row(row: pd.Series, split_column: str) -> CachedVariant:
    _ = split_column
    ref_seq = str(row["ref_seq"])
    alt_seq = str(row["alt_seq"])
    offset = int(row["variant_offset"])
    ref_allele = _row_str(row, "ref_allele", ref_seq[offset])
    alt_allele = _row_str(row, "alt_allele", alt_seq[offset])
    consequence = _row_str(row, "consequence_bucket", infer_consequence_bucket(ref_allele, alt_allele))
    bio_features: list[float] | None = None
    if "bio_features" in row and pd.notna(row["bio_features"]):
        raw = row["bio_features"]
        if isinstance(raw, str):
            decoded = json.loads(raw)
            bio_features = [float(value) for value in decoded]
        elif isinstance(raw, list):
            bio_features = [float(value) for value in raw]

    return CachedVariant(
        ref_seq=ref_seq,
        alt_seq=alt_seq,
        variant_offset=offset,
        label=int(row["label"]),
        index=int(row["original_index"]),
        ref_allele=ref_allele,
        alt_allele=alt_allele,
        bio_features=bio_features,
        consequence_bucket=consequence,
        gene_symbol=_row_str(row, "gene_symbol", ""),
    )


def load_variants_for_split(cache_path: Path, split_column: str, split_name: str) -> list[CachedVariant]:
    df = pd.read_parquet(cache_path)
    if split_name != "all":
        split_values = df[split_column].astype(str).str.lower()
        df = df.loc[split_values == split_name.lower()]
    return [_variant_from_cache_row(row, split_column) for _, row in df.iterrows()]


def _safe_metric_block(labels: np.ndarray, probs: np.ndarray, threshold: float) -> dict[str, Any]:
    labels = labels.astype(np.int64)
    probs = probs.astype(np.float64)
    result: dict[str, Any] = {
        "n": len(labels),
        "n_positive": int((labels == 1).sum()),
        "n_negative": int((labels == 0).sum()),
        "prevalence": float(np.mean(labels == 1)) if len(labels) else float("nan"),
        "mean_probability": float(np.mean(probs)) if len(probs) else float("nan"),
        "std_probability": float(np.std(probs)) if len(probs) else float("nan"),
        "mean_probability_positive": float(np.mean(probs[labels == 1])) if np.any(labels == 1) else float("nan"),
        "mean_probability_negative": float(np.mean(probs[labels == 0])) if np.any(labels == 0) else float("nan"),
    }
    if len(labels) == 0:
        result["at_training_threshold"] = classification_metrics(labels, probs, threshold)
        return result

    result["brier_score"] = brier_score(labels, probs)
    result["log_loss"] = binary_log_loss(labels, probs)
    result["at_training_threshold"] = classification_metrics(labels, probs, threshold)
    result["at_default_threshold"] = classification_metrics(labels, probs, 0.5)
    if len(np.unique(labels)) > 1:
        result["auroc"] = binary_roc_auc(labels, probs)
        result["auprc"] = binary_auprc(labels, probs)
        optimal_threshold, optimal_metrics = optimize_threshold(labels, probs, "mcc")
        result["at_slice_optimal_threshold_LEAKY"] = {
            **optimal_metrics,
            "threshold": optimal_threshold,
        }
    else:
        result["auroc"] = float("nan")
        result["auprc"] = binary_auprc(labels, probs)
        result["at_slice_optimal_threshold_LEAKY"] = None
    return result


def evaluate_dataset_split(
    *,
    model: EndToEndClinVarModel,
    variants: list[CachedVariant],
    config: FineTuneConfig,
    device: torch.device,
    precision_policy: Any,
    threshold: float,
    dataset_name: str,
    split_name: str,
    output_dir: Path,
) -> dict[str, Any]:
    dataset = ClinVarFineTuneDataset(variants)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=clinvar_collate_fn,
        num_workers=config.num_workers,
        drop_last=False,
    )
    preds = _predict(model, loader, device, precision_policy)
    metrics = _safe_metric_block(preds["labels"], preds["probs"], threshold)
    metrics.update(
        {
            "dataset": dataset_name,
            "split": split_name,
            "threshold_source": "training_val_optimal_threshold",
            "training_threshold": float(threshold),
            "swap_diagnostics": _swap_diagnostics(preds),
            "consequence_stratified_metrics": _consequence_stratified_metrics(
                variants,
                preds,
                threshold,
            ),
        }
    )

    pred_df = pd.DataFrame(
        {
            "dataset": dataset_name,
            "split": split_name,
            "original_index": preds["indices"].astype(int),
            "label": preds["labels"].astype(int),
            "probability": preds["probs"].astype(float),
            "swap_probability": preds["swap_probs"].astype(float),
        }
    )
    if "molecular_probs" in preds:
        pred_df["molecular_probability"] = preds["molecular_probs"].astype(float)
    if "regional_discounts" in preds:
        pred_df["regional_discount"] = preds["regional_discounts"].astype(float)
    for key, value in sorted(preds.items()):
        if key.startswith("gate_alpha_"):
            pred_df[key] = value.astype(float)
    if "gate_entropy" in preds:
        pred_df["gate_entropy"] = preds["gate_entropy"].astype(float)
    pred_path = output_dir / f"{dataset_name}.{split_name}.predictions.parquet"
    pred_df.to_parquet(pred_path, index=False)
    metrics["predictions_path"] = str(pred_path)
    return metrics


def _dataset_name(path: Path) -> str:
    return path.name.removesuffix(".parquet")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir or (args.output_dir / "cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path, checkpoint_root = resolve_finetuned_checkpoint(args)
    training_metrics = _load_training_metrics(checkpoint_root)
    threshold = (
        float(args.threshold)
        if args.threshold is not None
        else float(training_metrics.get("val_optimal_threshold", 0.5))
    )

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model, config, precision_policy, checkpoint_bundle = load_model(
        checkpoint_path=checkpoint_path,
        base_checkpoint=args.base_checkpoint,
        fasta=args.fasta,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        precision_override=args.precision,
    )

    run_summary: dict[str, Any] = {
        "finetuned_checkpoint": str(checkpoint_path),
        "base_checkpoint": str(args.base_checkpoint),
        "model_family": checkpoint_bundle.get("model_family"),
        "model_version": checkpoint_bundle.get("model_version"),
        "regime": checkpoint_bundle.get("regime"),
        "context_size": config.context_size,
        "batch_size": config.batch_size,
        "device": str(device),
        "precision": {
            "requested": precision_policy.requested,
            "resolved": precision_policy.resolved,
            "allow_tf32": config.allow_tf32,
        },
        "threshold": threshold,
        "trainable_params": count_trainable_parameters(model),
        "evaluations": [],
    }

    all_metrics: list[dict[str, Any]] = []
    for dataset_file in args.dataset_files:
        dataset_path = args.dataset_dir / dataset_file
        if not dataset_path.is_file():
            print(f"skip_missing_dataset={dataset_path}", flush=True)
            continue
        dataset_name = _dataset_name(dataset_path)
        print(f"building_cache dataset={dataset_name}", flush=True)
        cache_path = build_variant_cache(
            dataset_path=dataset_path,
            fasta_path=args.fasta,
            context_size=config.context_size,
            regime=config.regime,
            cache_dir=cache_dir,
            split_column=args.split_column,
            explicit_feature_columns=config.explicit_feature_columns,
        )
        for split_name in args.splits:
            variants = load_variants_for_split(cache_path, args.split_column, split_name)
            print(f"evaluating dataset={dataset_name} split={split_name} n={len(variants)}", flush=True)
            metrics = evaluate_dataset_split(
                model=model,
                variants=variants,
                config=config,
                device=device,
                precision_policy=precision_policy,
                threshold=threshold,
                dataset_name=dataset_name,
                split_name=split_name,
                output_dir=args.output_dir,
            )
            all_metrics.append(metrics)
            run_summary["evaluations"].append(
                {
                    "dataset": dataset_name,
                    "split": split_name,
                    "n": metrics["n"],
                    "n_positive": metrics["n_positive"],
                    "n_negative": metrics["n_negative"],
                    "auroc": metrics.get("auroc"),
                    "auprc": metrics.get("auprc"),
                    "mcc": metrics.get("at_training_threshold", {}).get("mcc"),
                    "f1": metrics.get("at_training_threshold", {}).get("f1"),
                    "recall": metrics.get("at_training_threshold", {}).get("recall"),
                    "specificity": metrics.get("at_training_threshold", {}).get("specificity"),
                }
            )

    metrics_path = args.output_dir / "regional_eval_metrics.json"
    summary_path = args.output_dir / "regional_eval_summary.json"
    table_path = args.output_dir / "regional_eval_summary.parquet"
    metrics_path.write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    pd.DataFrame(run_summary["evaluations"]).to_parquet(table_path, index=False)
    print(f"saved_metrics={metrics_path}", flush=True)
    print(f"saved_summary={summary_path}", flush=True)
    print(f"saved_table={table_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
