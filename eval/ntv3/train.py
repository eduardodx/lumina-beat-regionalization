from __future__ import annotations

import contextlib
import csv
import itertools
import json
import logging
import math
import os
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler

from eval.ntv3.adapters import build_ntv3_adapter
from eval.ntv3.config import NTv3BenchmarkConfig
from eval.ntv3.dataset import (
    AnnotationElementInfo,
    FunctionalTrackInfo,
    GenomeAnnotationDataset,
    GenomeBigWigDataset,
    SpeciesAssets,
    create_functional_targets_scaler,
    load_species_assets,
    prepare_fasta_index,
)
from eval.ntv3.diagnostics import (
    DEBUG_DATASET_ITEMS_ENV,
    DEBUG_EARLY_STEPS_ENV,
    DEBUG_OUTPUT_DIR_ENV,
    DEBUG_RANK_ENV,
    PhaseRecorder,
    StepEventLogger,
    env_int,
    resolve_eval_debug_batch_window,
)
from eval.ntv3.heads import (
    AnnotationModel,
    FunctionalTracksModel,
    functional_required_backbone_module_names,
    initialize_functional_head_output_bias,
)
from eval.ntv3.losses import focal_loss, poisson_multinomial_loss
from eval.ntv3.metrics import (
    AnnotationMetrics,
    FunctionalTracksMetrics,
    format_seconds_as_duration,
    metric_value_for_task,
)
from src.constants import DNA_VOCAB, MASK_ID, PAD_ID, UNK_ID
from src.precision import autocast_context, configure_float32_precision, resolve_precision_policy

log = logging.getLogger(__name__)
CHECKPOINT_DIGITS = 6
TRAIN_HISTORY_FIELDNAMES = [
    "step",
    "tokens_seen",
    "elapsed_seconds",
    "loss",
    "learning_rate",
    "head_learning_rate",
    "decoder_learning_rate",
    "backbone_learning_rate",
]
VAL_HISTORY_FIELDNAMES = ["step", "tokens_seen", "elapsed_seconds", "loss", "metric"]
DATASET_SCORES_FIELDNAMES = [
    "Metric",
    "run_id",
    "model_name",
    "best_step",
    "best_step_time",
    "training_tokens",
    "running_time",
    "species",
    "datasets",
    "assay_type",
    "track_name_clean",
]
LARGE_DATASET_SHUFFLE_THRESHOLD = 1_000_000
SAMPLER_RANDOM_CHUNK_SIZE = 32


@dataclass(frozen=True)
class DistributedContext:
    distributed: bool = False
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


@dataclass(frozen=True)
class RuntimeBatchSchedule:
    per_rank_batch_size: int
    grad_accum_steps: int
    effective_global_batch_size: int


class ExponentialMovingAverage:
    def __init__(self, model: torch.nn.Module, *, decay: float) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must be in (0, 1).")
        self.decay = float(decay)
        self.shadow: dict[str, torch.Tensor] = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
            if torch.is_floating_point(value)
        }

    def update(self, model: torch.nn.Module) -> None:
        with torch.no_grad():
            state = model.state_dict()
            for key, value in state.items():
                if not torch.is_floating_point(value):
                    continue
                detached = value.detach()
                if key not in self.shadow:
                    self.shadow[key] = detached.clone()
                    continue
                self.shadow[key].mul_(self.decay).add_(detached, alpha=1.0 - self.decay)

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.shadow = {key: value.detach().clone() for key, value in state_dict.items()}

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {key: value.detach().clone() for key, value in self.shadow.items()}

    def model_state_dict(self, model: torch.nn.Module) -> dict[str, torch.Tensor]:
        state = model.state_dict()
        merged: dict[str, torch.Tensor] = {}
        for key, value in state.items():
            source = self.shadow.get(key, value)
            merged[key] = source.detach().clone().to(device=value.device, dtype=value.dtype)
        return merged

    @contextlib.contextmanager
    def apply_to(self, model: torch.nn.Module) -> Any:
        state = model.state_dict()
        backup: dict[str, torch.Tensor] = {}
        try:
            with torch.no_grad():
                for key, shadow_value in self.shadow.items():
                    if key not in state:
                        continue
                    backup[key] = state[key].detach().clone()
                    state[key].copy_(shadow_value.to(device=state[key].device, dtype=state[key].dtype))
            yield
        finally:
            with torch.no_grad():
                restored_state = model.state_dict()
                for key, backup_value in backup.items():
                    restored_state[key].copy_(
                        backup_value.to(device=restored_state[key].device, dtype=restored_state[key].dtype)
                    )


class _ReplacementShuffleSampler(Sampler[int]):
    """Memory-safe sampler for huge NTv3 datasets.

    PyTorch's default RandomSampler/DistributedSampler builds a full
    `randperm(...).tolist()` when shuffling without replacement. For NTv3
    functional training, that can mean tens of millions of indices before the
    first batch is yielded, which is enough to kill the process without a
    Python traceback. This sampler keeps the same epoch length but generates
    indices lazily in small `torch.randint` chunks.
    """

    def __init__(
        self,
        data_source: Any,
        *,
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 0,
    ) -> None:
        self.data_source = data_source
        self.dataset_size = len(data_source)
        if self.dataset_size <= 0:
            raise ValueError("Replacement shuffle sampler requires a non-empty dataset.")
        self.num_replicas = max(1, int(num_replicas))
        self.rank = int(rank)
        if self.rank < 0 or self.rank >= self.num_replicas:
            raise ValueError(f"Invalid rank {self.rank} for num_replicas={self.num_replicas}.")
        self.seed = int(seed)
        self.epoch = 0
        self.num_samples = math.ceil(self.dataset_size / self.num_replicas)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Any:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch * self.num_replicas + self.rank)
        remaining = self.num_samples
        while remaining > 0:
            chunk_size = min(SAMPLER_RANDOM_CHUNK_SIZE, remaining)
            yield from torch.randint(
                high=self.dataset_size,
                size=(chunk_size,),
                dtype=torch.int64,
                generator=generator,
            ).tolist()
            remaining -= chunk_size


def _distributed_timeout() -> timedelta:
    return timedelta(seconds=int(os.environ.get("LUMINA_DDP_TIMEOUT_SECONDS", "7200")))


def _setup_distributed() -> DistributedContext:
    rank_env = os.environ.get("RANK")
    local_rank_env = os.environ.get("LOCAL_RANK")
    world_size_env = os.environ.get("WORLD_SIZE")
    if rank_env is None or local_rank_env is None or world_size_env is None:
        return DistributedContext()

    rank = int(rank_env)
    local_rank = int(local_rank_env)
    world_size = int(world_size_env)
    if world_size <= 1:
        return DistributedContext()

    if not torch.cuda.is_available():
        raise RuntimeError("NTv3 distributed benchmark requires CUDA.")

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        device_id=torch.device("cuda", local_rank),
        timeout=_distributed_timeout(),
    )
    return DistributedContext(distributed=True, rank=rank, local_rank=local_rank, world_size=world_size)


def _cleanup_distributed(ctx: DistributedContext) -> None:
    if ctx.distributed and dist.is_initialized():
        dist.destroy_process_group()


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def _distributed_sum_scalar(value: int | float, *, device: torch.device, dtype: torch.dtype) -> int | float:
    if not dist.is_initialized():
        return value
    tensor = torch.tensor(value, device=device, dtype=dtype)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    if dtype.is_floating_point:
        return float(tensor.item())
    return int(tensor.item())


def _resolve_runtime_batch_schedule(
    config: NTv3BenchmarkConfig,
    ctx: DistributedContext,
) -> RuntimeBatchSchedule:
    effective_global_batch_size = config.batch_size * config.grad_accum_steps
    max_per_rank_batch_size = config.batch_size
    if config.max_runtime_batch_size_per_rank is not None:
        max_per_rank_batch_size = min(config.batch_size, config.max_runtime_batch_size_per_rank)
    if not ctx.distributed:
        return RuntimeBatchSchedule(
            per_rank_batch_size=max_per_rank_batch_size,
            grad_accum_steps=config.grad_accum_steps,
            effective_global_batch_size=max_per_rank_batch_size * config.grad_accum_steps,
        )

    for per_rank_batch_size in range(max_per_rank_batch_size, 0, -1):
        denominator = per_rank_batch_size * ctx.world_size
        if effective_global_batch_size % denominator != 0:
            continue
        grad_accum_steps = effective_global_batch_size // denominator
        if grad_accum_steps < 1:
            continue
        return RuntimeBatchSchedule(
            per_rank_batch_size=per_rank_batch_size,
            grad_accum_steps=grad_accum_steps,
            effective_global_batch_size=effective_global_batch_size,
        )

    raise ValueError(
        "Cannot preserve the configured effective global batch size under DDP. "
        f"batch_size={config.batch_size}, grad_accum_steps={config.grad_accum_steps}, "
        f"max_runtime_batch_size_per_rank={config.max_runtime_batch_size_per_rank}, world_size={ctx.world_size}."
    )


def _maybe_no_sync(
    model: torch.nn.Module,
    *,
    should_sync: bool,
    ctx: DistributedContext,
    allow_no_sync: bool = True,
) -> Any:
    if allow_no_sync and not should_sync and ctx.distributed and isinstance(model, DDP):
        return model.no_sync()
    return contextlib.nullcontext()


def _set_model_train_mode(model: torch.nn.Module, *, freeze_backbone: bool) -> None:
    base_model = _unwrap_model(model)
    model.train()
    backbone = getattr(base_model, "backbone", None)
    if freeze_backbone and isinstance(backbone, torch.nn.Module):
        backbone.eval()
    frozen_backbone_head_names = getattr(base_model, "_frozen_backbone_head_names", ())
    if isinstance(backbone, torch.nn.Module):
        for head_name in frozen_backbone_head_names:
            head = getattr(backbone, str(head_name), None)
            if isinstance(head, torch.nn.Module):
                head.eval()


def _uses_custom_optimizer_config(config: NTv3BenchmarkConfig) -> bool:
    return (
        config.head_learning_rate is not None
        or config.backbone_learning_rate is not None
        or config.decoder_learning_rate is not None
        or config.backbone_layerwise_lr_decay is not None
        or config.head_only_warmup_steps > 0
        or config.no_weight_decay_norm_bias
    )


def _parameter_role(name: str) -> str:
    normalized = name.removeprefix("module.")
    if normalized.startswith("head."):
        return "head"
    if normalized.startswith("backbone.decoder.") or normalized.startswith("backbone.final_norm."):
        return "decoder"
    if normalized.startswith("backbone."):
        return "backbone"
    return "head"


def _parameter_weight_decay(config: NTv3BenchmarkConfig, name: str, parameter: torch.nn.Parameter) -> float:
    if not config.no_weight_decay_norm_bias:
        return config.weight_decay
    normalized = name.lower()
    if parameter.ndim < 2 or normalized.endswith(".bias") or "norm" in normalized:
        return 0.0
    return config.weight_decay


def _role_learning_rate(config: NTv3BenchmarkConfig, role: str) -> float:
    if role == "head":
        return float(config.head_learning_rate or config.learning_rate)
    if role == "decoder":
        return float(config.decoder_learning_rate or config.backbone_learning_rate or config.learning_rate)
    if role == "backbone":
        return float(config.backbone_learning_rate or config.learning_rate)
    return float(config.learning_rate)


def _max_backbone_block_index(model: torch.nn.Module) -> int | None:
    indices = []
    for name, _parameter in model.named_parameters():
        normalized = name.removeprefix("module.")
        match = re.match(r"backbone\.blocks\.(\d+)\.", normalized)
        if match is not None:
            indices.append(int(match.group(1)))
    return max(indices) if indices else None


def _parameter_backbone_lr_scale(
    name: str,
    *,
    config: NTv3BenchmarkConfig,
    max_block_index: int | None,
) -> tuple[float, int | None]:
    decay = config.backbone_layerwise_lr_decay
    if decay is None or max_block_index is None:
        return 1.0, None
    normalized = name.removeprefix("module.")
    block_match = re.match(r"backbone\.blocks\.(\d+)\.", normalized)
    if block_match is not None:
        layer_index = int(block_match.group(1))
        return float(decay) ** max(0, max_block_index - layer_index), layer_index
    if normalized.startswith("backbone.token_emb."):
        return float(decay) ** (max_block_index + 1), -1
    return 1.0, None


def _parameter_learning_rate(
    config: NTv3BenchmarkConfig,
    *,
    name: str,
    role: str,
    max_block_index: int | None,
) -> tuple[float, float, int | None]:
    base_lr = _role_learning_rate(config, role)
    if role != "backbone":
        return base_lr, 1.0, None
    scale, layer_index = _parameter_backbone_lr_scale(name, config=config, max_block_index=max_block_index)
    return base_lr * scale, scale, layer_index


def _build_optimizer(model: torch.nn.Module, *, config: NTv3BenchmarkConfig) -> AdamW:
    if not _uses_custom_optimizer_config(config):
        trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        return AdamW(trainable_parameters, lr=config.learning_rate, weight_decay=config.weight_decay)

    grouped: dict[tuple[str, float, float, int | None], dict[str, Any]] = {}
    max_block_index = _max_backbone_block_index(model)
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        role = _parameter_role(name)
        weight_decay = _parameter_weight_decay(config, name, parameter)
        learning_rate, lr_scale, layer_index = _parameter_learning_rate(
            config,
            name=name,
            role=role,
            max_block_index=max_block_index,
        )
        key = (role, weight_decay, learning_rate, layer_index)
        if key not in grouped:
            grouped[key] = {
                "params": [],
                "lr": learning_rate,
                "weight_decay": weight_decay,
                "ntv3_role": role,
                "ntv3_lr_scale": lr_scale,
                "ntv3_layer_index": layer_index,
            }
        grouped[key]["params"].append(parameter)

    role_order = {"head": 0, "decoder": 1, "backbone": 2}
    param_groups = sorted(
        grouped.values(),
        key=lambda group: (
            role_order.get(str(group["ntv3_role"]), 99),
            -float(group["lr"]),
            float(group["weight_decay"]) > 0.0,
        ),
    )
    return AdamW(param_groups)


def _learning_rate_summary(optimizer: torch.optim.Optimizer) -> dict[str, float | None]:
    role_lrs: dict[str, float] = {}
    all_lrs: list[float] = []
    for group in optimizer.param_groups:
        lr = float(group["lr"])
        all_lrs.append(lr)
        role = group.get("ntv3_role")
        if isinstance(role, str):
            role_lrs[role] = max(role_lrs.get(role, 0.0), lr)
    primary = role_lrs.get("head", max(all_lrs) if all_lrs else 0.0)
    return {
        "primary": primary,
        "head": role_lrs.get("head"),
        "decoder": role_lrs.get("decoder"),
        "backbone": role_lrs.get("backbone"),
    }


def _suppress_head_only_warmup_grads(
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    config: NTv3BenchmarkConfig,
) -> int:
    if config.head_only_warmup_steps <= 0 or step > config.head_only_warmup_steps:
        return 0
    suppressed = 0
    for group in optimizer.param_groups:
        if group.get("ntv3_role") not in {"backbone", "decoder"}:
            continue
        for parameter in group["params"]:
            if parameter.grad is not None:
                parameter.grad = None
                suppressed += 1
    return suppressed


def _freeze_pretraining_heads(backbone: torch.nn.Module, head_names: set[str]) -> list[str]:
    frozen_names: list[str] = []
    for module_name in sorted(head_names):
        module = getattr(backbone, module_name, None)
        if not isinstance(module, torch.nn.Module):
            continue
        for parameter in module.parameters():
            parameter.requires_grad_(False)
        module.eval()
        frozen_names.append(module_name)
    return frozen_names


def _freeze_unused_pretraining_heads(
    backbone: torch.nn.Module,
    *,
    preserve_names: set[str] | None = None,
) -> list[str]:
    # NTv3 fine-tuning uses backbone.encode() plus a fresh benchmark head. The auxiliary
    # pretraining heads are absent from the NTv3 loss, so we delete them outright instead
    # of just freezing — under DDP, frozen-but-present params trigger reduction errors
    # (and `find_unused_parameters=True` adds collective overhead that hangs on multi-rank
    # NVLink hosts).
    preserve_names = preserve_names or set()
    removed_names: list[str] = []
    for module_name in (
        "mlm_head",
        "phylo100_head",
        "phylo470_head",
        "phylo100_subst_head",
        "zoo241_subst_head",
        "structure_head",
        "region_head",
        "splice_class_head",
        "splice_distance_head",
        "gnomad_af_head",
        "gnomad_observed_head",
        "counterfactual_snv_head",
        "edit_localize_head",
        "regulatory_head",
        "sequence_embedding_head",
        "aa_head",
        "codon_phylo_head",
        "mutation_effect_head",
        "codon_head",
        "encode_head",
        "global_proj",
    ):
        if module_name in preserve_names:
            continue
        if hasattr(backbone, module_name):
            delattr(backbone, module_name)
            removed_names.append(module_name)
    return removed_names


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_rows(path: Path, rows: list[dict[str, Any]], *, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _checkpoint_dir(output_dir: Path) -> Path:
    return output_dir / "checkpoints"


def _checkpoint_path(output_dir: Path, step: int) -> Path:
    return _checkpoint_dir(output_dir) / f"step_{step:0{CHECKPOINT_DIGITS}d}.pt"


def _checkpoint_step(path: Path) -> int:
    try:
        return int(path.stem.split("_", maxsplit=1)[1])
    except (IndexError, ValueError):
        return -1


def _cleanup_old_checkpoints(output_dir: Path, *, keep: int) -> None:
    checkpoint_dir = _checkpoint_dir(output_dir)
    if not checkpoint_dir.exists():
        return

    checkpoints = sorted(checkpoint_dir.glob("step_*.pt"), key=_checkpoint_step)
    if len(checkpoints) <= keep:
        return

    for path in checkpoints[:-keep]:
        path.unlink(missing_ok=True)


def _write_history_artifacts(
    output_dir: Path,
    *,
    train_history_rows: list[dict[str, Any]],
    val_history_rows: list[dict[str, Any]],
) -> None:
    _write_rows(output_dir / "metrics_train.csv", train_history_rows, fieldnames=TRAIN_HISTORY_FIELDNAMES)
    _write_rows(output_dir / "metrics_val.csv", val_history_rows, fieldnames=VAL_HISTORY_FIELDNAMES)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def _capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _rng_state_tensor(value: torch.Tensor) -> torch.Tensor:
    # Checkpoint resume may load CPU RNG tensors onto CUDA through map_location.
    # PyTorch RNG restore APIs require ByteTensor states on CPU.
    return value.detach().to(device="cpu", dtype=torch.uint8)


def _restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return

    torch_state = state.get("torch")
    if isinstance(torch_state, torch.Tensor):
        torch.set_rng_state(_rng_state_tensor(torch_state))

    numpy_state = state.get("numpy")
    if isinstance(numpy_state, tuple):
        np.random.set_state(numpy_state)

    cuda_state = state.get("cuda")
    if cuda_state is not None and torch.cuda.is_available():
        cuda_states = [
            _rng_state_tensor(value)
            for value in cuda_state
            if isinstance(value, torch.Tensor)
        ]
        if cuda_states:
            torch.cuda.set_rng_state_all(cuda_states)


def _init_wandb_run(config: NTv3BenchmarkConfig, output_dir: Path) -> Any | None:
    status_path = output_dir / "wandb_status.json"
    if not config.wandb_enabled:
        _write_json(status_path, {"enabled": False, "status": "disabled"})
        return None

    try:
        import wandb
    except ImportError as exc:
        log.warning(
            "wandb logging is enabled for NTv3, but the package is not installed. Continuing without wandb: %s",
            exc,
        )
        _write_json(
            status_path,
            {
                "enabled": True,
                "status": "degraded",
                "reason": "wandb package is not installed",
            },
        )
        return None
    wandb = cast(Any, wandb)

    try:
        run = wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=config.wandb_run_name,
            tags=config.wandb_tags or [],
            config=asdict(config),
            dir=str(output_dir),
        )
    except Exception as exc:
        log.warning("wandb initialization failed; continuing without wandb: %s", exc)
        _write_json(
            status_path,
            {
                "enabled": True,
                "status": "degraded",
                "reason": f"wandb initialization failed: {exc}",
            },
        )
        return None

    if run is not None:
        run.define_metric("global_step")
        run.define_metric("*", step_metric="global_step")
        _write_json(
            status_path,
            {
                "enabled": True,
                "status": "active",
                "project": config.wandb_project,
                "entity": config.wandb_entity,
                "run_name": run.name,
            },
        )
    return run


def _wandb_numeric_payload(metrics: dict[str, Any], *, prefix: str) -> dict[str, float]:
    payload: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
            payload[f"{prefix}/{key}"] = float(value)
    return payload


def _wandb_log(wandb_run: Any | None, *, step: int, payload: dict[str, Any]) -> None:
    if wandb_run is None:
        return
    serializable: dict[str, bool | float] = {"global_step": float(step)}
    for key, value in payload.items():
        if isinstance(value, (bool, int, float, np.integer, np.floating)):
            serializable[key] = float(value) if not isinstance(value, bool) else value
    try:
        wandb_run.log(serializable, step=int(step))
    except Exception as exc:
        log.warning("wandb logging failed at step %d; continuing: %s", step, exc)


def _build_base_lr_lambda(*, config: NTv3BenchmarkConfig) -> Any:
    total_steps = config.num_steps_training
    warmup_steps = config.resolved_num_steps_warmup()

    if config.scheduler_name == "modified_square_decay":
        optimizer_lr = float(config.learning_rate)
        initial_lr = float(config.resolved_initial_learning_rate())
        final_lr_multiplier = float(config.final_learning_rate_multiplier)
        if warmup_steps <= 0:
            alpha_polynomial_decay = 0.0
        else:
            numerator = np.log(1.0 / final_lr_multiplier)
            denominator = np.log(float(total_steps) / float(warmup_steps))
            alpha_polynomial_decay = float(numerator / denominator) if denominator > 0 else 0.0

        def lr_lambda(step: int) -> float:
            current_step = max(int(step), 0)
            if optimizer_lr == 0.0:
                return 0.0
            if warmup_steps > 0 and current_step < warmup_steps:
                start_multiplier = initial_lr / optimizer_lr
                progress = float(current_step) / float(warmup_steps)
                return start_multiplier + (1.0 - start_multiplier) * progress
            if warmup_steps <= 0:
                return 1.0
            denominator = float(current_step + 1)
            decay_multiplier = (float(warmup_steps) / denominator) ** alpha_polynomial_decay
            return min(decay_multiplier, 1.0)

        return lr_lambda

    def lr_lambda(step: int) -> float:
        if total_steps <= 0:
            return 1.0
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress_steps = max(total_steps - warmup_steps, 1)
        progress = min(max((step - warmup_steps) / progress_steps, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return lr_lambda


def _build_scheduler(optimizer: torch.optim.Optimizer, *, config: NTv3BenchmarkConfig) -> LambdaLR:
    base_lr_lambda = _build_base_lr_lambda(config=config)

    if not any("ntv3_role" in group for group in optimizer.param_groups):
        return LambdaLR(optimizer, lr_lambda=base_lr_lambda)

    def make_group_lambda(role: str) -> Any:
        def lr_lambda(step: int) -> float:
            if role in {"backbone", "decoder"} and int(step) < config.head_only_warmup_steps:
                return 0.0
            return float(base_lr_lambda(step))

        return lr_lambda

    group_lambdas = [
        make_group_lambda(str(group.get("ntv3_role", "head")))
        for group in optimizer.param_groups
    ]
    return LambdaLR(optimizer, lr_lambda=group_lambdas)



def _reverse_complement_input_ids(input_ids: torch.Tensor) -> torch.Tensor:
    max_token_id = max(int(input_ids.max().detach().item()), MASK_ID, UNK_ID)
    lookup = torch.arange(max_token_id + 1, device=input_ids.device, dtype=torch.long)
    lookup[PAD_ID] = PAD_ID
    lookup[DNA_VOCAB["A"]] = DNA_VOCAB["T"]
    lookup[DNA_VOCAB["C"]] = DNA_VOCAB["G"]
    lookup[DNA_VOCAB["G"]] = DNA_VOCAB["C"]
    lookup[DNA_VOCAB["T"]] = DNA_VOCAB["A"]
    lookup[DNA_VOCAB["N"]] = DNA_VOCAB["N"]
    lookup[MASK_ID] = MASK_ID
    lookup[UNK_ID] = UNK_ID
    rc = lookup[input_ids.to(dtype=torch.long)]
    return torch.flip(rc.to(dtype=input_ids.dtype), dims=(1,))


def _strand_partner_candidates(name: str) -> list[str]:
    pairs = (
        ("_plus", "_minus"),
        (".plus", ".minus"),
        ("-plus", "-minus"),
        ("_pos", "_neg"),
        (".pos", ".neg"),
        ("-pos", "-neg"),
        ("_p", "_m"),
        (".p", ".m"),
        ("-p", "-m"),
        ("_forward", "_reverse"),
        (".forward", ".reverse"),
        ("-forward", "-reverse"),
    )
    lower_name = name.lower()
    candidates: list[str] = []
    for left, right in pairs:
        if lower_name.endswith(left):
            candidates.append(name[: -len(left)] + right)
        if lower_name.endswith(right):
            candidates.append(name[: -len(right)] + left)
    return candidates


def _has_strand_marker(name: str) -> bool:
    lower_name = name.lower()
    markers = (
        "plus",
        "minus",
        "forward",
        "reverse",
        "+",
        "_p",
        "_m",
        ".p",
        ".m",
        "-p",
        "-m",
    )
    return any(marker in lower_name for marker in markers)


def _functional_reverse_complement_track_mapping(
    track_infos: list[FunctionalTrackInfo],
) -> tuple[list[int], list[int], int]:
    id_to_index = {track.dataset_id.lower(): index for index, track in enumerate(track_infos)}
    display_to_index = {track.track_name_clean.lower(): index for index, track in enumerate(track_infos)}
    mapping_indices: list[int] = []
    active_indices: list[int] = []
    paired_count = 0
    for index, track in enumerate(track_infos):
        partner_index: int | None = None
        for candidate in _strand_partner_candidates(track.dataset_id):
            partner_index = id_to_index.get(candidate.lower())
            if partner_index is not None:
                break
        if partner_index is None:
            for candidate in _strand_partner_candidates(track.track_name_clean):
                partner_index = display_to_index.get(candidate.lower())
                if partner_index is not None:
                    break
        if partner_index is not None:
            mapping_indices.append(partner_index)
            active_indices.append(index)
            paired_count += int(partner_index != index)
            continue

        mapping_indices.append(index)
        if not (_has_strand_marker(track.dataset_id) or _has_strand_marker(track.track_name_clean)):
            active_indices.append(index)
    return mapping_indices, active_indices, paired_count


def _functional_rc_consistency_loss(
    predictions: torch.Tensor,
    rc_predictions: torch.Tensor,
    *,
    track_indices: torch.Tensor,
    active_track_indices: torch.Tensor,
) -> torch.Tensor:
    if active_track_indices.numel() == 0:
        return predictions.new_zeros(())
    aligned_rc = torch.flip(rc_predictions, dims=(1,)).index_select(-1, track_indices)
    predictions = predictions.index_select(-1, active_track_indices)
    aligned_rc = aligned_rc.index_select(-1, active_track_indices)
    return F.mse_loss(torch.log1p(predictions), torch.log1p(aligned_rc))


def _move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=(device.type == "cuda")) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _build_data_loader(
    dataset: Any,
    *,
    batch_size: int,
    shuffle: bool,
    sampler: Any = None,
    num_workers: int,
    prefetch_factor: int,
    device: torch.device,
) -> DataLoader:
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle if sampler is None else False,
        "sampler": sampler,
        "num_workers": num_workers,
        "drop_last": False,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": _worker_init_fn,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = prefetch_factor
        if device.type == "cuda" and sys.platform.startswith("linux"):
            kwargs["multiprocessing_context"] = "spawn"
    return DataLoader(dataset, **kwargs)


def _build_train_sampler(
    dataset: Any,
    *,
    ctx: DistributedContext,
    seed: int,
) -> tuple[Any | None, bool]:
    dataset_size = len(dataset)
    use_replacement_sampler = dataset_size >= LARGE_DATASET_SHUFFLE_THRESHOLD

    if ctx.distributed:
        if use_replacement_sampler:
            return (
                _ReplacementShuffleSampler(
                    dataset,
                    num_replicas=ctx.world_size,
                    rank=ctx.rank,
                    seed=seed,
                ),
                False,
            )
        return (
            DistributedSampler(
                dataset,
                num_replicas=ctx.world_size,
                rank=ctx.rank,
                shuffle=True,
                seed=seed,
                drop_last=False,
            ),
            False,
        )

    if use_replacement_sampler:
        return _ReplacementShuffleSampler(dataset, seed=seed), False
    return None, True


def _worker_init_fn(worker_id: int) -> None:
    del worker_id
    info = torch.utils.data.get_worker_info()
    if info is not None:
        random.seed(info.seed % (2**32))


def _save_model_bundle(
    *,
    model: torch.nn.Module,
    config: NTv3BenchmarkConfig,
    precision: str,
    path: Path,
    step: int,
    tokens_seen: int,
    metric_value: float,
    elapsed_seconds: float,
    model_state_dict: dict[str, torch.Tensor] | None = None,
    optimizer_state_dict: dict[str, Any] | None = None,
    scheduler_state_dict: dict[str, Any] | None = None,
    ema_state_dict: dict[str, torch.Tensor] | None = None,
    training_state: dict[str, Any] | None = None,
    rng_state: dict[str, Any] | None = None,
) -> None:
    bundle = {
        "format": "ntv3_benchmark_v2",
        "config": asdict(config),
        "precision": precision,
        "step": step,
        "tokens_seen": tokens_seen,
        "metric_value": metric_value,
        "elapsed_seconds": elapsed_seconds,
        "model_state_dict": model_state_dict if model_state_dict is not None else model.state_dict(),
    }
    if optimizer_state_dict is not None:
        bundle["optimizer_state_dict"] = optimizer_state_dict
    if scheduler_state_dict is not None:
        bundle["scheduler_state_dict"] = scheduler_state_dict
    if ema_state_dict is not None:
        bundle["ema_state_dict"] = ema_state_dict
    if training_state is not None:
        bundle["training_state"] = training_state
    if rng_state is not None:
        bundle["rng_state"] = rng_state
    torch.save(bundle, path)


def _load_model_bundle(path: Path, *, map_location: torch.device) -> dict[str, Any]:
    try:
        bundle = torch.load(str(path), map_location=map_location, weights_only=False)
    except TypeError:
        bundle = torch.load(str(path), map_location=map_location)
    if not isinstance(bundle, dict):
        raise TypeError(f"Model bundle {path} must contain a dict, got {type(bundle).__name__}.")
    return bundle


def _resolve_resume_checkpoint_path(
    config: NTv3BenchmarkConfig,
    *,
    output_dir: Path,
    map_location: torch.device,
) -> Path | None:
    explicit_path = config.resolved_resume_checkpoint_path()
    if explicit_path is not None:
        _load_model_bundle(explicit_path, map_location=map_location)
        return explicit_path

    if not config.auto_resume:
        return None

    checkpoint_dir = _checkpoint_dir(output_dir)
    candidates = sorted(checkpoint_dir.glob("step_*.pt"), key=_checkpoint_step, reverse=True)
    for candidate in candidates:
        try:
            _load_model_bundle(candidate, map_location=map_location)
        except Exception as exc:
            log.warning("Skipping unreadable NTv3 checkpoint %s: %s", candidate, exc)
            continue
        return candidate
    return None


def _evaluate_functional(
    model: FunctionalTracksModel,
    loader: DataLoader,
    *,
    device: torch.device,
    precision_policy: Any,
    track_infos: list[FunctionalTrackInfo],
) -> dict[str, Any]:
    metrics = FunctionalTracksMetrics(track_ids=[track.dataset_id for track in track_infos])
    split_name = str(getattr(loader.dataset, "split", "eval"))
    debug_batches, start_batch, max_batches = resolve_eval_debug_batch_window(
        default_debug_batches=max(0, env_int(DEBUG_EARLY_STEPS_ENV))
    )
    try:
        total_batches = len(loader)
    except TypeError:
        total_batches = None
    stop_batch = start_batch + max_batches if max_batches > 0 else None
    capped_loader = itertools.islice(loader, start_batch, stop_batch)
    model.eval()
    with torch.no_grad():
        batch_count = 0
        for batch_index, batch in enumerate(capped_loader, start=start_batch):
            trace_ordinal = batch_index - start_batch
            batch_count = trace_ordinal + 1
            if trace_ordinal < debug_batches:
                log.info("NTv3 eval trace: split=%s batch_index=%d phase=batch_loaded", split_name, batch_index)
            batch = _move_batch_to_device(batch, device)
            if trace_ordinal < debug_batches:
                log.info("NTv3 eval trace: split=%s batch_index=%d phase=batch_on_device", split_name, batch_index)
            with autocast_context(precision_policy):
                if trace_ordinal < debug_batches:
                    log.info(
                        "NTv3 eval trace: split=%s batch_index=%d phase=forward_started",
                        split_name,
                        batch_index,
                    )
                predictions = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            if trace_ordinal < debug_batches:
                log.info(
                    "NTv3 eval trace: split=%s batch_index=%d phase=forward_completed",
                    split_name,
                    batch_index,
                )
            loss = poisson_multinomial_loss(predictions, batch["targets"])
            if trace_ordinal < debug_batches:
                log.info(
                    "NTv3 eval trace: split=%s batch_index=%d phase=loss_completed loss=%f",
                    split_name,
                    batch_index,
                    float(loss.item()),
                )
            metrics.update(predictions, batch["targets"], float(loss.item()))
    if debug_batches > 0 or start_batch > 0 or max_batches > 0:
        batch_cap_reached = max_batches > 0 and total_batches is not None and total_batches > max_batches
        log.info(
            "NTv3 eval trace: split=%s phase=completed start_batch=%d "
            "batches=%d total_batches=%s max_batches=%s batch_cap_reached=%s",
            split_name,
            start_batch,
            batch_count,
            total_batches if total_batches is not None else "unknown",
            max_batches if max_batches > 0 else "all",
            batch_cap_reached,
        )
    return metrics.compute()


def _evaluate_annotation(
    model: AnnotationModel,
    loader: DataLoader,
    *,
    device: torch.device,
    precision_policy: Any,
    element_infos: list[AnnotationElementInfo],
) -> dict[str, Any]:
    metrics = AnnotationMetrics(element_names=[element.dataset_id for element in element_infos])
    split_name = str(getattr(loader.dataset, "split", "eval"))
    debug_batches, start_batch, max_batches = resolve_eval_debug_batch_window(
        default_debug_batches=max(0, env_int(DEBUG_EARLY_STEPS_ENV))
    )
    try:
        total_batches = len(loader)
    except TypeError:
        total_batches = None
    stop_batch = start_batch + max_batches if max_batches > 0 else None
    capped_loader = itertools.islice(loader, start_batch, stop_batch)
    model.eval()
    with torch.no_grad():
        batch_count = 0
        for batch_index, batch in enumerate(capped_loader, start=start_batch):
            trace_ordinal = batch_index - start_batch
            batch_count = trace_ordinal + 1
            if trace_ordinal < debug_batches:
                log.info("NTv3 eval trace: split=%s batch_index=%d phase=batch_loaded", split_name, batch_index)
            batch = _move_batch_to_device(batch, device)
            if trace_ordinal < debug_batches:
                log.info("NTv3 eval trace: split=%s batch_index=%d phase=batch_on_device", split_name, batch_index)
            with autocast_context(precision_policy):
                if trace_ordinal < debug_batches:
                    log.info(
                        "NTv3 eval trace: split=%s batch_index=%d phase=forward_started",
                        split_name,
                        batch_index,
                    )
                logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            if trace_ordinal < debug_batches:
                log.info(
                    "NTv3 eval trace: split=%s batch_index=%d phase=forward_completed",
                    split_name,
                    batch_index,
                )
            loss = focal_loss(logits, batch["targets"])
            if trace_ordinal < debug_batches:
                log.info(
                    "NTv3 eval trace: split=%s batch_index=%d phase=loss_completed loss=%f",
                    split_name,
                    batch_index,
                    float(loss.item()),
                )
            metrics.update(logits, batch["targets"], float(loss.item()))
    if debug_batches > 0 or start_batch > 0 or max_batches > 0:
        batch_cap_reached = max_batches > 0 and total_batches is not None and total_batches > max_batches
        log.info(
            "NTv3 eval trace: split=%s phase=completed start_batch=%d "
            "batches=%d total_batches=%s max_batches=%s batch_cap_reached=%s",
            split_name,
            start_batch,
            batch_count,
            total_batches if total_batches is not None else "unknown",
            max_batches if max_batches > 0 else "all",
            batch_cap_reached,
        )
    return metrics.compute()


def _functional_dataset_rows(
    *,
    track_infos: list[FunctionalTrackInfo],
    test_metrics: dict[str, Any],
    config: NTv3BenchmarkConfig,
    best_step: int,
    best_tokens_seen: int,
    best_elapsed_seconds: float,
    total_tokens_seen: int,
    total_elapsed_seconds: float,
) -> list[dict[str, Any]]:
    run_id = config.run_id or Path(config.output_dir).name
    rows: list[dict[str, Any]] = []
    for track in track_infos:
        rows.append(
            {
                "Metric": float(test_metrics[f"{track.dataset_id}/pearson"]),
                "run_id": run_id,
                "model_name": config.model_name_for_leaderboard,
                "best_step": float(best_step),
                "best_step_time": format_seconds_as_duration(best_elapsed_seconds),
                "training_tokens": float(total_tokens_seen),
                "running_time": float(total_elapsed_seconds),
                "species": config.species_name,
                "datasets": track.dataset_id,
                "assay_type": track.assay_type,
                "track_name_clean": track.track_name_clean,
            }
        )
    return rows


def _annotation_dataset_rows(
    *,
    element_infos: list[AnnotationElementInfo],
    test_metrics: dict[str, Any],
    config: NTv3BenchmarkConfig,
    best_step: int,
    best_tokens_seen: int,
    best_elapsed_seconds: float,
    total_tokens_seen: int,
    total_elapsed_seconds: float,
) -> list[dict[str, Any]]:
    run_id = config.run_id or Path(config.output_dir).name
    rows: list[dict[str, Any]] = []
    for element in element_infos:
        rows.append(
            {
                "Metric": float(test_metrics[f"{element.dataset_id}/mcc"]),
                "run_id": run_id,
                "model_name": config.model_name_for_leaderboard,
                "best_step": float(best_step),
                "best_step_time": format_seconds_as_duration(best_elapsed_seconds),
                "training_tokens": float(total_tokens_seen),
                "running_time": float(total_elapsed_seconds),
                "species": config.species_name,
                "datasets": element.dataset_id,
                "assay_type": "Annotation",
                "track_name_clean": "",
            }
        )
    return rows


def _build_datasets(config: NTv3BenchmarkConfig, assets: SpeciesAssets) -> tuple[Any, Any, Any]:
    if config.task_type == "functional":
        if not assets.functional_tracks:
            raise ValueError(f"No functional tracks found for species {config.species_name!r}.")
        track_infos = list(assets.functional_tracks)
        transform_fn = create_functional_targets_scaler(track_infos)
        train_dataset = GenomeBigWigDataset(
            fasta_path=assets.fasta_path,
            split_regions=assets.split_regions,
            track_infos=track_infos,
            split="train",
            sequence_length=config.sequence_length,
            transform_fn=transform_fn,
            overlap=config.train_overlap,
            keep_target_center_fraction=config.keep_target_center_fraction,
            seed=config.seed,
        )
        val_dataset = GenomeBigWigDataset(
            fasta_path=assets.fasta_path,
            split_regions=assets.split_regions,
            track_infos=track_infos,
            split="val",
            sequence_length=config.sequence_length,
            transform_fn=transform_fn,
            keep_target_center_fraction=config.keep_target_center_fraction,
            limit_num_samples=config.num_validation_samples,
            seed=config.seed,
        )
        test_dataset = GenomeBigWigDataset(
            fasta_path=assets.fasta_path,
            split_regions=assets.split_regions,
            track_infos=track_infos,
            split="test",
            sequence_length=config.sequence_length,
            transform_fn=transform_fn,
            keep_target_center_fraction=config.keep_target_center_fraction,
            limit_num_samples=config.num_test_samples,
            seed=config.seed,
        )
        return train_dataset, val_dataset, test_dataset

    if not assets.annotation_elements:
        raise ValueError(f"No genome annotation elements found for species {config.species_name!r}.")
    element_infos = list(assets.annotation_elements)
    train_dataset = GenomeAnnotationDataset(
        fasta_path=assets.fasta_path,
        split_regions=assets.split_regions,
        element_infos=element_infos,
        split="train",
        sequence_length=config.sequence_length,
        overlap=config.train_overlap,
        keep_target_center_fraction=config.keep_target_center_fraction,
        seed=config.seed,
    )
    val_dataset = GenomeAnnotationDataset(
        fasta_path=assets.fasta_path,
        split_regions=assets.split_regions,
        element_infos=element_infos,
        split="val",
        sequence_length=config.sequence_length,
        keep_target_center_fraction=config.keep_target_center_fraction,
        limit_num_samples=config.num_validation_samples,
        seed=config.seed,
    )
    test_dataset = GenomeAnnotationDataset(
        fasta_path=assets.fasta_path,
        split_regions=assets.split_regions,
        element_infos=element_infos,
        split="test",
        sequence_length=config.sequence_length,
        keep_target_center_fraction=config.keep_target_center_fraction,
        limit_num_samples=config.num_test_samples,
        seed=config.seed,
    )
    return train_dataset, val_dataset, test_dataset


def run_ntv3_benchmark(config: NTv3BenchmarkConfig) -> dict[str, Any]:
    config.validate()
    output_dir = config.resolved_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "metrics_test.json"
    dataset_scores_path = output_dir / "dataset_scores.csv"
    if summary_path.exists() and dataset_scores_path.exists() and not config.overwrite:
        return json.loads(summary_path.read_text(encoding="utf-8"))

    ctx = _setup_distributed()
    device = torch.device("cuda", ctx.local_rank) if ctx.distributed else torch.device(config.device)
    phase_recorder = PhaseRecorder(
        output_dir=output_dir,
        rank=ctx.rank,
        world_size=ctx.world_size,
        metadata={
            "species": config.species_name,
            "task_type": config.task_type,
            "model_family": config.model_family,
            "model_version": config.model_version,
            "feature_source": config.feature_source,
            "functional_head_type": config.functional_head_type,
            "functional_head_hidden_dim": config.functional_head_hidden_dim,
            "functional_head_dropout": config.functional_head_dropout,
            "functional_head_kernel_size": config.functional_head_kernel_size,
            "functional_head_aux_features": config.functional_head_aux_features,
            "functional_head_aux_projection_dim": config.functional_head_aux_projection_dim,
            "functional_rc_consistency_weight": config.functional_rc_consistency_weight,
            "num_workers": config.num_workers,
            "prefetch_factor": config.prefetch_factor,
            "batch_size": config.batch_size,
            "grad_accum_steps": config.grad_accum_steps,
            "head_only_warmup_steps": config.head_only_warmup_steps,
        },
    )
    phase_recorder.mark(
        "distributed_initialized",
        status="in_progress",
        details={
            "distributed": ctx.distributed,
            "rank": ctx.rank,
            "local_rank": ctx.local_rank,
            "world_size": ctx.world_size,
            "device": str(device),
        },
        primary=ctx.is_main,
    )
    configure_float32_precision(config.allow_tf32)
    precision_policy = resolve_precision_policy(device, config.precision)
    runtime_batch_schedule = _resolve_runtime_batch_schedule(config, ctx)
    ddp_static_graph = bool(ctx.distributed)
    # static_graph is required for Lumina's reentrant/checkpointed backbone, but
    # PyTorch 2.8 DDP can assert internally if no_sync is nested with it.
    ddp_no_sync_enabled = bool(ctx.distributed and not ddp_static_graph)
    resume_checkpoint_path = _resolve_resume_checkpoint_path(config, output_dir=output_dir, map_location=device)
    resumed_from_checkpoint = str(resume_checkpoint_path) if resume_checkpoint_path is not None else None
    phase_recorder.update_metadata(
        requested_precision=precision_policy.requested,
        resolved_precision=precision_policy.resolved,
        runtime_batch_size_per_rank=runtime_batch_schedule.per_rank_batch_size,
        runtime_grad_accum_steps=runtime_batch_schedule.grad_accum_steps,
        effective_global_batch_size=runtime_batch_schedule.effective_global_batch_size,
        ddp_static_graph=ddp_static_graph,
        ddp_no_sync_enabled=ddp_no_sync_enabled,
        resume_from_checkpoint=resumed_from_checkpoint,
    )
    phase_recorder.mark(
        "runtime_resolved",
        details={
            "runtime_batch_size_per_rank": runtime_batch_schedule.per_rank_batch_size,
            "runtime_grad_accum_steps": runtime_batch_schedule.grad_accum_steps,
            "effective_global_batch_size": runtime_batch_schedule.effective_global_batch_size,
            "ddp_static_graph": ddp_static_graph,
            "ddp_no_sync_enabled": ddp_no_sync_enabled,
            "resume_from_checkpoint": resumed_from_checkpoint,
        },
        primary=ctx.is_main,
    )

    if ctx.is_main:
        _checkpoint_dir(output_dir).mkdir(parents=True, exist_ok=True)
        run_config_payload = asdict(config)
        run_config_payload["resolved_resume_from_checkpoint"] = resumed_from_checkpoint
        _write_json(output_dir / "run_config.json", run_config_payload)
    if ctx.distributed:
        dist.barrier()
    phase_recorder.mark("run_config_written", primary=ctx.is_main)

    _seed_everything(config.seed)
    wandb_run = _init_wandb_run(config, output_dir) if ctx.is_main else None

    try:
        assets = load_species_assets(config.dataset_root, config.species_name)
        prepare_fasta_index(assets.fasta_path)
        phase_recorder.mark(
            "assets_loaded",
            details={
                "fasta_path": str(assets.fasta_path),
                "num_functional_tracks": len(assets.functional_tracks),
                "num_annotation_elements": len(assets.annotation_elements),
            },
            primary=ctx.is_main,
        )

        functional_rc_track_indices: torch.Tensor | None = None
        functional_rc_active_track_indices: torch.Tensor | None = None
        if config.task_type == "functional" and config.functional_rc_consistency_weight > 0.0:
            rc_mapping_indices, rc_active_indices, rc_paired_count = _functional_reverse_complement_track_mapping(
                list(assets.functional_tracks)
            )
            functional_rc_track_indices = torch.tensor(rc_mapping_indices, device=device, dtype=torch.long)
            functional_rc_active_track_indices = torch.tensor(rc_active_indices, device=device, dtype=torch.long)
            if ctx.is_main:
                log.info(
                    "NTv3 functional RC consistency enabled: weight=%f active_tracks=%d paired_tracks=%d",
                    config.functional_rc_consistency_weight,
                    len(rc_active_indices),
                    rc_paired_count,
                )
            phase_recorder.mark(
                "functional_rc_consistency_ready",
                details={
                    "weight": config.functional_rc_consistency_weight,
                    "active_tracks": len(rc_active_indices),
                    "paired_tracks": rc_paired_count,
                },
                primary=ctx.is_main,
            )

        train_dataset, val_dataset, test_dataset = _build_datasets(config, assets)
        phase_recorder.mark(
            "datasets_built",
            details={
                "train_samples": len(train_dataset),
                "val_samples": len(val_dataset),
                "test_samples": len(test_dataset),
            },
            primary=ctx.is_main,
        )
        train_sampler, train_shuffle = _build_train_sampler(
            train_dataset,
            ctx=ctx,
            seed=config.seed,
        )
        if isinstance(train_sampler, _ReplacementShuffleSampler) and ctx.is_main:
            log.info(
                "Using memory-safe replacement sampler for NTv3 train dataset: "
                "dataset_size=%d world_size=%d",
                len(train_dataset),
                ctx.world_size,
            )
        train_loader = _build_data_loader(
            train_dataset,
            batch_size=runtime_batch_schedule.per_rank_batch_size,
            shuffle=train_shuffle,
            sampler=train_sampler,
            num_workers=config.num_workers,
            prefetch_factor=config.prefetch_factor,
            device=device,
        )
        val_loader = _build_data_loader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            prefetch_factor=config.prefetch_factor,
            device=device,
        )
        test_loader = _build_data_loader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            prefetch_factor=config.prefetch_factor,
            device=device,
        )
        phase_recorder.mark(
            "dataloaders_built",
            details={
                "train_batches_per_epoch": len(train_loader),
                "val_batches": len(val_loader),
                "test_batches": len(test_loader),
                "num_workers": config.num_workers,
                "prefetch_factor": config.prefetch_factor,
            },
            primary=ctx.is_main,
        )

        adapter = build_ntv3_adapter(
            model_family=config.model_family,
            model_version=config.model_version,
            checkpoint_path=str(config.resolved_checkpoint_path()),
            device=device,
            precision=precision_policy,
        )
        phase_recorder.mark(
            "adapter_loaded",
            details={
                "backbone_type": type(adapter.backbone).__name__,
                "d_model": adapter.d_model,
                "decoder_dim": getattr(adapter, "decoder_dim", adapter.d_model),
            },
            primary=ctx.is_main,
        )

        feature_dim = adapter.d_model
        if config.feature_source == "decoder":
            feature_dim = int(getattr(adapter, "decoder_dim", adapter.d_model))
        if config.task_type == "functional":
            model = FunctionalTracksModel(
                adapter.backbone,
                feature_dim,
                num_tracks=len(assets.functional_tracks),
                keep_target_center_fraction=config.keep_target_center_fraction,
                feature_source=config.feature_source,
                head_type=config.functional_head_type,
                head_hidden_dim=config.functional_head_hidden_dim,
                head_dropout=config.functional_head_dropout,
                head_kernel_size=config.functional_head_kernel_size,
                head_aux_features=config.functional_head_aux_features,
                head_aux_projection_dim=config.functional_head_aux_projection_dim,
            )
        else:
            model = AnnotationModel(
                adapter.backbone,
                feature_dim,
                num_elements=len(assets.annotation_elements),
                keep_target_center_fraction=config.keep_target_center_fraction,
                feature_source=config.feature_source,
            )
        if config.task_type == "functional" and config.functional_head_output_bias_init == "scaled-track-mean":
            target_means = torch.ones(len(assets.functional_tracks), dtype=torch.float32)
            initialize_functional_head_output_bias(cast(FunctionalTracksModel, model).head, target_means)
            if ctx.is_main:
                log.info(
                    "Initialized functional head output bias to transformed scaled-track mean baseline: tracks=%d",
                    len(assets.functional_tracks),
                )
        model.to(device)
        frozen_pretraining_head_names: set[str] = set()
        preserved_backbone_module_names: set[str] = set()
        if config.task_type == "functional":
            frozen_pretraining_head_names = functional_required_backbone_module_names(
                config.functional_head_type,
                config.functional_head_aux_features,
            )
            preserved_backbone_module_names.update(frozen_pretraining_head_names)
            if bool(getattr(cast(Any, model).head, "uses_global_context", False)):
                preserved_backbone_module_names.add("global_proj")

        frozen_preserved_head_names: list[str] = []
        if frozen_pretraining_head_names and hasattr(model, "backbone"):
            frozen_preserved_head_names = _freeze_pretraining_heads(model.backbone, frozen_pretraining_head_names)
            cast(Any, model)._frozen_backbone_head_names = tuple(frozen_preserved_head_names)
            if ctx.is_main:
                log.info(
                    "Preserved frozen Lumina pretraining heads for NTv3 auxiliary readout: %s",
                    ", ".join(frozen_preserved_head_names) if frozen_preserved_head_names else "(none found)",
                )

        if config.freeze_backbone and hasattr(model, "backbone"):
            for parameter in model.backbone.parameters():
                parameter.requires_grad_(False)
            model.backbone.eval()
        elif hasattr(model, "backbone"):
            frozen_head_names = _freeze_unused_pretraining_heads(
                model.backbone,
                preserve_names=preserved_backbone_module_names,
            )
            if frozen_head_names and ctx.is_main:
                log.info(
                    "Removed unused Lumina pretraining heads for NTv3 fine-tuning: %s",
                    ", ".join(frozen_head_names),
                )

        if ctx.distributed:
            model = DDP(
                model,
                device_ids=[ctx.local_rank],
                find_unused_parameters=False,
                static_graph=ddp_static_graph,
            )
        eval_model = _unwrap_model(model)
        trainable_parameter_count = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )
        phase_recorder.mark(
            "model_ready",
            details={
                "trainable_parameter_count": trainable_parameter_count,
                "freeze_backbone": config.freeze_backbone,
                "feature_source": config.feature_source,
                "functional_head_type": config.functional_head_type if config.task_type == "functional" else None,
                "functional_head_aux_features": (
                    config.functional_head_aux_features if config.task_type == "functional" else None
                ),
                "functional_head_aux_projection_dim": (
                    config.functional_head_aux_projection_dim if config.task_type == "functional" else None
                ),
                "functional_rc_consistency_weight": (
                    config.functional_rc_consistency_weight if config.task_type == "functional" else None
                ),
                "frozen_preserved_head_names": frozen_preserved_head_names,
                "feature_dim": feature_dim,
                "ddp_static_graph": ddp_static_graph,
                "ddp_no_sync_enabled": ddp_no_sync_enabled,
            },
            primary=ctx.is_main,
        )

        if config.task_type == "functional":
            functional_eval_model = cast(FunctionalTracksModel, eval_model)
            eval_fn = lambda loader: _evaluate_functional(  # noqa: E731
                functional_eval_model,
                loader,
                device=device,
                precision_policy=precision_policy,
                track_infos=list(assets.functional_tracks),
            )
        else:
            annotation_eval_model = cast(AnnotationModel, eval_model)
            eval_fn = lambda loader: _evaluate_annotation(  # noqa: E731
                annotation_eval_model,
                loader,
                device=device,
                precision_policy=precision_policy,
                element_infos=list(assets.annotation_elements),
            )

        trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not trainable_parameters:
            raise RuntimeError("NTv3 benchmark requires at least one trainable parameter.")

        if ctx.is_main:
            log.info(
                "NTv3 runtime schedule: world_size=%d per_rank_batch_size=%d "
                "grad_accum_steps=%d effective_global_batch=%d ddp_static_graph=%s ddp_no_sync_enabled=%s",
                ctx.world_size,
                runtime_batch_schedule.per_rank_batch_size,
                runtime_batch_schedule.grad_accum_steps,
                runtime_batch_schedule.effective_global_batch_size,
                ddp_static_graph,
                ddp_no_sync_enabled,
            )

        optimizer = _build_optimizer(model, config=config)
        scheduler = _build_scheduler(optimizer, config=config)

        train_history_rows: list[dict[str, Any]] = (
            _read_rows(output_dir / "metrics_train.csv") if ctx.is_main and resume_checkpoint_path is not None else []
        )
        val_history_rows: list[dict[str, Any]] = (
            _read_rows(output_dir / "metrics_val.csv") if ctx.is_main and resume_checkpoint_path is not None else []
        )
        best_metric_value = float("-inf")
        best_step = 0
        best_tokens_seen = 0
        best_elapsed_seconds = 0.0
        tokens_seen = 0
        best_model_path = output_dir / "best_model.pt"
        start_time = time.perf_counter()
        train_epoch = 0
        batches_consumed_in_epoch = 0
        start_step = 1
        resume_ema_state_dict: dict[str, torch.Tensor] | None = None

        if resume_checkpoint_path is not None:
            resume_bundle = _load_model_bundle(resume_checkpoint_path, map_location=device)
            eval_model.load_state_dict(resume_bundle["model_state_dict"])

            optimizer_state_dict = resume_bundle.get("optimizer_state_dict")
            if isinstance(optimizer_state_dict, dict):
                optimizer.load_state_dict(optimizer_state_dict)

            scheduler_state_dict = resume_bundle.get("scheduler_state_dict")
            if isinstance(scheduler_state_dict, dict):
                scheduler.load_state_dict(scheduler_state_dict)

            raw_ema_state_dict = resume_bundle.get("ema_state_dict")
            if isinstance(raw_ema_state_dict, dict):
                resume_ema_state_dict = cast(dict[str, torch.Tensor], raw_ema_state_dict)

            training_state = resume_bundle.get("training_state")
            if not isinstance(training_state, dict):
                training_state = {}

            best_metric_value = float(training_state.get("best_metric_value", best_metric_value))
            best_step = int(training_state.get("best_step", best_step))
            best_tokens_seen = int(training_state.get("best_tokens_seen", best_tokens_seen))
            best_elapsed_seconds = float(training_state.get("best_elapsed_seconds", best_elapsed_seconds))
            train_epoch = int(training_state.get("train_epoch", train_epoch))
            batches_consumed_in_epoch = int(training_state.get("batches_consumed_in_epoch", batches_consumed_in_epoch))

            tokens_seen = int(resume_bundle.get("tokens_seen", tokens_seen))
            resumed_elapsed_seconds = float(resume_bundle.get("elapsed_seconds", 0.0))
            start_time = time.perf_counter() - resumed_elapsed_seconds
            start_step = max(1, int(resume_bundle.get("step", 0)) + 1)
            _restore_rng_state(resume_bundle.get("rng_state") if isinstance(resume_bundle, dict) else None)

            if ctx.is_main:
                log.info(
                    "Resuming NTv3 benchmark from %s at step=%d epoch=%d consumed_batches=%d",
                    resume_checkpoint_path,
                    start_step,
                    train_epoch,
                    batches_consumed_in_epoch,
                )
            phase_recorder.mark(
                "resume_checkpoint_loaded",
                details={
                    "path": str(resume_checkpoint_path),
                    "start_step": start_step,
                    "train_epoch": train_epoch,
                    "batches_consumed_in_epoch": batches_consumed_in_epoch,
                },
                primary=ctx.is_main,
            )
        else:
            phase_recorder.mark("resume_checkpoint_checked", details={"path": None}, primary=ctx.is_main)

        ema: ExponentialMovingAverage | None = None
        if config.ema_decay is not None:
            ema = ExponentialMovingAverage(eval_model, decay=config.ema_decay)
            if resume_ema_state_dict is not None:
                ema.load_state_dict(resume_ema_state_dict)
            if ctx.is_main:
                log.info("NTv3 EMA enabled: decay=%f resumed=%s", config.ema_decay, resume_ema_state_dict is not None)

        train_batches_per_epoch = len(train_loader)
        if train_batches_per_epoch <= 0:
            raise RuntimeError("NTv3 benchmark produced an empty training DataLoader.")
        if batches_consumed_in_epoch >= train_batches_per_epoch:
            batches_consumed_in_epoch %= train_batches_per_epoch

        if train_sampler is not None:
            train_sampler.set_epoch(train_epoch)
        train_iter = iter(train_loader)
        for _ in range(batches_consumed_in_epoch):
            next(train_iter)
        phase_recorder.mark(
            "train_iterator_ready",
            details={
                "train_epoch": train_epoch,
                "batches_consumed_in_epoch": batches_consumed_in_epoch,
                "train_batches_per_epoch": train_batches_per_epoch,
            },
            primary=ctx.is_main,
        )

        if ctx.is_main:
            _write_history_artifacts(
                output_dir,
                train_history_rows=train_history_rows,
                val_history_rows=val_history_rows,
            )
        _set_model_train_mode(model, freeze_backbone=config.freeze_backbone)
        first_phase_markers: set[str] = set()
        early_steps_window = env_int(DEBUG_EARLY_STEPS_ENV)
        os.environ[DEBUG_RANK_ENV] = str(ctx.rank)
        if env_int(DEBUG_DATASET_ITEMS_ENV) > 0:
            os.environ[DEBUG_OUTPUT_DIR_ENV] = str(output_dir)
        step_event_logger = StepEventLogger(
            output_dir=output_dir,
            rank=ctx.rank,
            max_steps=early_steps_window,
            task_type=config.task_type,
            resolved_precision=precision_policy.resolved,
        )
        if step_event_logger.enabled and ctx.is_main:
            log.info(
                "NTv3 early-step diagnostics enabled: window=%d output_path=%s",
                step_event_logger.max_steps,
                step_event_logger.path,
            )

        def record_step_phase(
            phase: str,
            *,
            step: int,
            micro_step: int,
            details: dict[str, Any] | None = None,
        ) -> None:
            step_event_logger.record(
                step=step,
                micro_step=micro_step,
                phase=phase,
                extra=details,
            )
            if step_event_logger.is_active_for_step(step) and ctx.is_main:
                detail_suffix = f" details={details}" if details else ""
                log.info(
                    "NTv3 step trace: step=%d micro_step=%d phase=%s%s",
                    step,
                    micro_step,
                    phase,
                    detail_suffix,
                )
            manifest_phase = f"first_{phase}"
            if manifest_phase in first_phase_markers:
                return
            first_phase_markers.add(manifest_phase)
            manifest_details: dict[str, Any] = {"step": step, "micro_step": micro_step}
            if details:
                manifest_details.update(details)
            phase_recorder.mark(manifest_phase, details=manifest_details, primary=ctx.is_main)
            if ctx.is_main:
                detail_suffix = f" details={manifest_details}" if manifest_details else ""
                log.info("NTv3 phase reached: %s%s", manifest_phase, detail_suffix)

        for step in range(start_step, config.num_steps_training + 1):
            optimizer.zero_grad(set_to_none=True)
            step_loss_total = 0.0
            step_tokens = 0
            for micro_step in range(runtime_batch_schedule.grad_accum_steps):
                record_step_phase("batch_requested", step=step, micro_step=micro_step)
                batch = next(train_iter)
                record_step_phase("batch_loaded", step=step, micro_step=micro_step)
                batches_consumed_in_epoch += 1
                reached_epoch_boundary = batches_consumed_in_epoch >= train_batches_per_epoch
                step_tokens += int(batch["attention_mask"].sum().item())
                batch = _move_batch_to_device(batch, device)
                record_step_phase(
                    "batch_on_device",
                    step=step,
                    micro_step=micro_step,
                    details={"device": str(device)},
                )
                should_sync = micro_step == runtime_batch_schedule.grad_accum_steps - 1
                with _maybe_no_sync(
                    model,
                    should_sync=should_sync,
                    ctx=ctx,
                    allow_no_sync=ddp_no_sync_enabled,
                ):
                    with autocast_context(precision_policy):
                        rc_consistency_loss: torch.Tensor | None = None
                        if config.task_type == "functional":
                            predictions = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                            record_step_phase(
                                "forward_completed",
                                step=step,
                                micro_step=micro_step,
                                details={"task_type": config.task_type},
                            )
                            loss = poisson_multinomial_loss(predictions, batch["targets"])
                            if (
                                config.functional_rc_consistency_weight > 0.0
                                and functional_rc_track_indices is not None
                                and functional_rc_active_track_indices is not None
                            ):
                                rc_input_ids = _reverse_complement_input_ids(batch["input_ids"])
                                rc_attention_mask = torch.flip(batch["attention_mask"], dims=(1,))
                                with torch.no_grad():
                                    rc_predictions = model(input_ids=rc_input_ids, attention_mask=rc_attention_mask)
                                rc_consistency_loss = _functional_rc_consistency_loss(
                                    predictions,
                                    rc_predictions,
                                    track_indices=functional_rc_track_indices,
                                    active_track_indices=functional_rc_active_track_indices,
                                )
                                loss = loss + config.functional_rc_consistency_weight * rc_consistency_loss
                            else:
                                rc_consistency_loss = None
                        else:
                            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                            record_step_phase(
                                "forward_completed",
                                step=step,
                                micro_step=micro_step,
                                details={"task_type": config.task_type},
                            )
                            loss = focal_loss(logits, batch["targets"])
                        loss_details: dict[str, Any] = {"loss": float(loss.item())}
                        if config.task_type == "functional" and rc_consistency_loss is not None:
                            loss_details["rc_consistency_loss"] = float(rc_consistency_loss.item())
                        record_step_phase(
                            "loss_computed",
                            step=step,
                            micro_step=micro_step,
                            details=loss_details,
                        )
                    (loss / runtime_batch_schedule.grad_accum_steps).backward()
                record_step_phase("backward_completed", step=step, micro_step=micro_step)
                step_loss_total += float(loss.item())
                if reached_epoch_boundary:
                    train_epoch += 1
                    batches_consumed_in_epoch = 0
                    if train_sampler is not None:
                        train_sampler.set_epoch(train_epoch)
                    train_iter = iter(train_loader)

            _suppress_head_only_warmup_grads(optimizer, step=step, config=config)
            torch.nn.utils.clip_grad_norm_(trainable_parameters, config.grad_clip)
            optimizer.step()
            if ema is not None:
                ema.update(eval_model)
            scheduler.step()
            record_step_phase("optimizer_step_completed", step=step, micro_step=0)
            if ctx.distributed:
                step_loss_total = float(_distributed_sum_scalar(step_loss_total, device=device, dtype=torch.float64))
                step_loss_total /= float(ctx.world_size)
                step_tokens = int(_distributed_sum_scalar(step_tokens, device=device, dtype=torch.int64))
            tokens_seen += step_tokens
            elapsed_seconds = time.perf_counter() - start_time
            avg_train_loss = step_loss_total / runtime_batch_schedule.grad_accum_steps

            if ctx.is_main and (step % config.log_every_n_steps == 0 or step == config.num_steps_training):
                lr_summary = _learning_rate_summary(optimizer)
                train_row = {
                    "step": step,
                    "tokens_seen": tokens_seen,
                    "elapsed_seconds": round(elapsed_seconds, 6),
                    "loss": round(avg_train_loss, 8),
                    "learning_rate": lr_summary["primary"],
                    "head_learning_rate": lr_summary["head"],
                    "decoder_learning_rate": lr_summary["decoder"],
                    "backbone_learning_rate": lr_summary["backbone"],
                }
                train_history_rows.append(train_row)
                _wandb_log(
                    wandb_run,
                    step=step,
                    payload={
                        "train/loss": avg_train_loss,
                        "train/learning_rate": lr_summary["primary"],
                        "train/head_learning_rate": lr_summary["head"],
                        "train/decoder_learning_rate": lr_summary["decoder"],
                        "train/backbone_learning_rate": lr_summary["backbone"],
                        "train/tokens_seen": tokens_seen,
                        "train/elapsed_seconds": elapsed_seconds,
                    },
                )
                _write_history_artifacts(
                    output_dir,
                    train_history_rows=train_history_rows,
                    val_history_rows=val_history_rows,
                )

            should_validate = step % config.validate_every_n_steps == 0 or step == config.num_steps_training
            should_save_checkpoint = config.save_every_n_steps > 0 and step % config.save_every_n_steps == 0
            if should_validate or should_save_checkpoint:
                if ctx.distributed:
                    dist.barrier()
                if ctx.is_main:
                    metric_value_for_checkpoint = best_metric_value
                    if should_validate:
                        eval_debug_batches, eval_start_batch, eval_max_batches = resolve_eval_debug_batch_window(
                            default_debug_batches=max(0, early_steps_window)
                        )
                        log.info(
                            "NTv3 validation starting: step=%d val_batches=%d "
                            "debug_batches=%d start_batch=%d max_batches=%s",
                            step,
                            len(val_loader),
                            eval_debug_batches,
                            eval_start_batch,
                            eval_max_batches if eval_max_batches > 0 else "all",
                        )
                        phase_recorder.mark(
                            "validation_started",
                            details={
                                "step": step,
                                "debug_batches": eval_debug_batches,
                                "start_batch": eval_start_batch,
                                "max_batches": eval_max_batches if eval_max_batches > 0 else None,
                            },
                            primary=True,
                        )
                        eval_context = ema.apply_to(eval_model) if ema is not None else contextlib.nullcontext()
                        with eval_context:
                            val_metrics = eval_fn(val_loader)
                        metric_value = metric_value_for_task(config.task_type, val_metrics)
                        log.info(
                            "NTv3 validation completed: step=%d metric=%f loss=%f",
                            step,
                            float(metric_value),
                            float(val_metrics["loss"]),
                        )
                        phase_recorder.mark(
                            "validation_completed",
                            details={
                                "step": step,
                                "metric": float(metric_value),
                                "loss": float(val_metrics["loss"]),
                            },
                            primary=True,
                        )
                        metric_value_for_checkpoint = metric_value
                        val_history_rows.append(
                            {
                                "step": step,
                                "tokens_seen": tokens_seen,
                                "elapsed_seconds": round(elapsed_seconds, 6),
                                "loss": round(float(val_metrics["loss"]), 8),
                                "metric": round(metric_value, 8),
                            }
                        )
                        if metric_value > best_metric_value:
                            best_metric_value = metric_value
                            best_step = step
                            best_tokens_seen = tokens_seen
                            best_elapsed_seconds = elapsed_seconds
                            log.info("NTv3 best-model save starting: step=%d path=%s", step, best_model_path)
                            _save_model_bundle(
                                model=eval_model,
                                config=config,
                                precision=precision_policy.resolved,
                                path=best_model_path,
                                step=step,
                                tokens_seen=tokens_seen,
                                metric_value=metric_value,
                                elapsed_seconds=elapsed_seconds,
                                model_state_dict=ema.model_state_dict(eval_model) if ema is not None else None,
                            )
                            log.info("NTv3 best-model save completed: step=%d path=%s", step, best_model_path)
                        _wandb_log(
                            wandb_run,
                            step=step,
                            payload={
                                "val/metric": metric_value,
                                "val/loss": float(val_metrics["loss"]),
                                "val/tokens_seen": tokens_seen,
                                "val/elapsed_seconds": elapsed_seconds,
                                "val/best_metric": best_metric_value,
                                "val/best_step": best_step,
                                **_wandb_numeric_payload(val_metrics, prefix="val"),
                            },
                        )
                        metric_value_for_checkpoint = best_metric_value

                    if should_save_checkpoint:
                        periodic_checkpoint_path = _checkpoint_path(output_dir, step)
                        log.info(
                            "NTv3 periodic checkpoint save starting: step=%d path=%s",
                            step,
                            periodic_checkpoint_path,
                        )
                        _save_model_bundle(
                            model=eval_model,
                            config=config,
                            precision=precision_policy.resolved,
                            path=periodic_checkpoint_path,
                            step=step,
                            tokens_seen=tokens_seen,
                            metric_value=metric_value_for_checkpoint,
                            elapsed_seconds=elapsed_seconds,
                            optimizer_state_dict=optimizer.state_dict(),
                            scheduler_state_dict=scheduler.state_dict(),
                            ema_state_dict=ema.state_dict() if ema is not None else None,
                            training_state={
                                "best_metric_value": best_metric_value,
                                "best_step": best_step,
                                "best_tokens_seen": best_tokens_seen,
                                "best_elapsed_seconds": best_elapsed_seconds,
                                "train_epoch": train_epoch,
                                "batches_consumed_in_epoch": batches_consumed_in_epoch,
                            },
                            rng_state=_capture_rng_state(),
                        )
                        _cleanup_old_checkpoints(output_dir, keep=config.max_checkpoints_to_keep)
                        log.info(
                            "NTv3 periodic checkpoint save completed: step=%d path=%s",
                            step,
                            periodic_checkpoint_path,
                        )

                    _write_history_artifacts(
                        output_dir,
                        train_history_rows=train_history_rows,
                        val_history_rows=val_history_rows,
                    )
                if ctx.distributed:
                    dist.barrier()
                _set_model_train_mode(model, freeze_backbone=config.freeze_backbone)

        if ctx.distributed:
            dist.barrier()

        if ctx.is_main:
            if not best_model_path.is_file():
                raise FileNotFoundError(f"Best model checkpoint was not written to {best_model_path}.")

            bundle = _load_model_bundle(best_model_path, map_location=device)
            eval_model.load_state_dict(bundle["model_state_dict"])
            best_step = int(bundle.get("step", best_step))
            best_tokens_seen = int(bundle.get("tokens_seen", best_tokens_seen))
            best_elapsed_seconds = float(bundle.get("elapsed_seconds", best_elapsed_seconds))

            total_elapsed_seconds = time.perf_counter() - start_time
            phase_recorder.mark("test_started", details={"step": best_step}, primary=True)
            test_metrics = eval_fn(test_loader)

            if config.task_type == "functional":
                dataset_rows = _functional_dataset_rows(
                    track_infos=list(assets.functional_tracks),
                    test_metrics=test_metrics,
                    config=config,
                    best_step=best_step,
                    best_tokens_seen=best_tokens_seen,
                    best_elapsed_seconds=best_elapsed_seconds,
                    total_tokens_seen=tokens_seen,
                    total_elapsed_seconds=total_elapsed_seconds,
                )
            else:
                dataset_rows = _annotation_dataset_rows(
                    element_infos=list(assets.annotation_elements),
                    test_metrics=test_metrics,
                    config=config,
                    best_step=best_step,
                    best_tokens_seen=best_tokens_seen,
                    best_elapsed_seconds=best_elapsed_seconds,
                    total_tokens_seen=tokens_seen,
                    total_elapsed_seconds=total_elapsed_seconds,
                )

            _write_history_artifacts(
                output_dir,
                train_history_rows=train_history_rows,
                val_history_rows=val_history_rows,
            )
            _write_rows(
                dataset_scores_path,
                dataset_rows,
                fieldnames=DATASET_SCORES_FIELDNAMES,
            )

            summary = {
                "species": config.species_name,
                "task_type": config.task_type,
                "model_family": config.model_family,
                "model_version": config.model_version,
                "requested_precision": precision_policy.requested,
                "resolved_precision": precision_policy.resolved,
                "world_size": ctx.world_size,
                "runtime_batch_size_per_rank": runtime_batch_schedule.per_rank_batch_size,
                "runtime_grad_accum_steps": runtime_batch_schedule.grad_accum_steps,
                "effective_global_batch_size": runtime_batch_schedule.effective_global_batch_size,
                "resume_from_checkpoint": resumed_from_checkpoint,
                "resumed_from_step": max(0, start_step - 1),
                "best_step": best_step,
                "best_tokens_seen": best_tokens_seen,
                "best_step_time": format_seconds_as_duration(best_elapsed_seconds),
                "training_tokens": tokens_seen,
                "running_time": total_elapsed_seconds,
                "test_metrics": test_metrics,
                "dataset_scores_path": str(dataset_scores_path),
            }
            _write_json(summary_path, summary)
            phase_recorder.mark(
                "summary_written",
                details={
                    "best_step": best_step,
                    "training_tokens": tokens_seen,
                    "dataset_scores_path": str(dataset_scores_path),
                },
                primary=True,
            )
            _wandb_log(
                wandb_run,
                step=config.num_steps_training,
                payload={
                    "final/best_step": best_step,
                    "final/best_tokens_seen": best_tokens_seen,
                    "final/training_tokens": tokens_seen,
                    "final/running_time": total_elapsed_seconds,
                    **_wandb_numeric_payload(test_metrics, prefix="test"),
                },
            )
            if wandb_run is not None:
                try:
                    wandb_run.summary["world_size"] = ctx.world_size
                    wandb_run.summary["runtime_batch_size_per_rank"] = runtime_batch_schedule.per_rank_batch_size
                    wandb_run.summary["runtime_grad_accum_steps"] = runtime_batch_schedule.grad_accum_steps
                    wandb_run.summary["effective_global_batch_size"] = (
                        runtime_batch_schedule.effective_global_batch_size
                    )
                    wandb_run.summary["resume_from_checkpoint"] = resumed_from_checkpoint
                    wandb_run.summary["resumed_from_step"] = max(0, start_step - 1)
                    wandb_run.summary["best_step"] = best_step
                    wandb_run.summary["best_tokens_seen"] = best_tokens_seen
                    wandb_run.summary["training_tokens"] = tokens_seen
                    wandb_run.summary["running_time"] = total_elapsed_seconds
                    wandb_run.summary["dataset_scores_path"] = str(dataset_scores_path)
                except Exception as exc:
                    log.warning("wandb summary update failed; continuing: %s", exc)

        if ctx.distributed:
            dist.barrier()
        phase_recorder.mark("benchmark_completed", details={"summary_path": str(summary_path)}, primary=ctx.is_main)
        if summary_path.is_file():
            return json.loads(summary_path.read_text(encoding="utf-8"))
        return {
            "species": config.species_name,
            "task_type": config.task_type,
            "world_size": ctx.world_size,
        }
    except Exception as exc:
        phase_recorder.mark(
            "benchmark_failed",
            status="failed",
            details={
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
            primary=ctx.is_main,
        )
        raise
    finally:
        if wandb_run is not None:
            try:
                wandb_run.finish()
            except Exception as exc:
                log.warning("wandb shutdown failed; continuing: %s", exc)
        _cleanup_distributed(ctx)
