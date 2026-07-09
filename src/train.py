from __future__ import annotations

import argparse
import contextlib
import gc
import json
import math
import os
import random
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Literal, cast

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.constants import (
    DEFAULT_CHROMOSOMES,
    DEFAULT_VAL_CHROMOSOMES,
)
from src.dataset import AbraomDatasetArm, AbraomSequenceDataset, HG38SplicePhyloDataset, MultiTaskCollator
from src.defaults import (
    DEFAULT_FASTA_PATH,
    DEFAULT_GTF_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PHYLO100_BW_PATH,
    DEFAULT_PHYLO470_BW_PATH,
)
from src.encode_tracks import EncodeTrackSpec
from src.hard_position_tracker import RegionLossEMA
from src.metrics import MetricAccumulator
from src.model_utils import count_parameters
from src.models import (
    DEFAULT_MODEL_KEY,
    build_registered_model,
    get_model_spec,
    registered_model_keys,
    resolve_model_config_dict,
)
from src.models.beat_shared import get_cuda_device_capability, normalize_mamba3_runtime_config
from src.objectives import (
    GradNormBalancer,
    compute_multitask_loss,
    compute_uncertainty_weighted_loss,
    rc_embedding_loss,
)
from src.precision import (
    PrecisionMode,
    PrecisionPolicy,
    apply_fp8_linear_replacements,
    autocast_context,
    configure_float32_precision,
    precision_log_string,
    precision_metadata,
    resolve_precision_policy,
)
from src.repo_paths import resolve_repo_relative_path

CHECKPOINT_DIGITS = 8
LR_SCHEDULER_CHOICES = ("cosine", "constant", "cosine_rewarm")
CHROMOSOME_SPLIT_MODE_CHOICES = ("full_hg38", "heldout")
LOSS_BALANCING_CHOICES = ("uncertainty", "gradnorm", "fixed")
COUNTERFACTUAL_WEIGHTING_CHOICES = ("fixed", "uncertainty")
AUXILIARY_SCHEDULE_CHOICES = ("all", "alternate_allele_counterfactual")
DATASET_MODE_CHOICES = ("hg38_splice_phylo", "abraom_sequences")
LRSchedulerName = Literal["cosine", "constant", "cosine_rewarm"]
LossBalancingMode = Literal["uncertainty", "gradnorm", "fixed"]
ChromosomeSplitMode = Literal["full_hg38", "heldout"]
CounterfactualWeightingMode = Literal["fixed", "uncertainty"]
AuxiliaryScheduleMode = Literal["all", "alternate_allele_counterfactual"]
DatasetMode = Literal["hg38_splice_phylo", "abraom_sequences"]
LEGACY_MODEL_CONFIG_FIELDS = frozenset({"d_model", "n_layers", "d_state", "d_conv", "expand", "dropout"})
LEGACY_TOP_LEVEL_CONFIG_FIELDS = frozenset({"preset"}) | LEGACY_MODEL_CONFIG_FIELDS

LOSS_STAT_KEYS = (
    "loss",
    "loss_mlm",
    "loss_phylo100",
    "loss_phylo470",
    "loss_structure",
    "loss_region",
    "loss_aa",
    "loss_codon_phylo",
    "loss_mutation_effect",
    "loss_mutation_effect_rank",
    "loss_counterfactual",
    "loss_counterfactual_local",
    "loss_counterfactual_far",
    "loss_allele",
    "loss_allele_effect",
    "loss_allele_severity",
    "loss_allele_rank",
    "loss_allele_swap",
    "loss_allele_far",
    "loss_codon",
    "loss_encode",
    "loss_conservation_bin",
    "loss_splice_distance",
    "loss_codon_pos",
    "loss_exon_phase",
    "loss_counterfactual_snv",
    "loss_counterfactual_snv_severity",
    "loss_rc",
)
COUNT_STAT_KEYS = (
    "mlm_supervised_tokens",
    "aux_valid_tokens",
    "aa_cds_tokens",
    "counterfactual_snv_valid",
    "allele_valid_pairs",
    "splice_bg_tokens",
    "splice_core_tokens",
    "splice_region_tokens",
    "region_intergenic_tokens",
    "region_intron_tokens",
    "region_noncoding_exon_tokens",
    "region_utr_tokens",
    "region_cds_tokens",
)
WEIGHT_STAT_KEYS = (
    "splice_weight_bg",
    "splice_weight_core",
    "splice_weight_region",
    "region_weight_intergenic",
    "region_weight_intron",
    "region_weight_noncoding_exon",
    "region_weight_utr",
    "region_weight_cds",
)
MLM_REGION_LOSS_STAT_KEYS = (
    "mlm_region_loss_intergenic",
    "mlm_region_loss_intron",
    "mlm_region_loss_noncoding_exon",
    "mlm_region_loss_utr",
    "mlm_region_loss_cds",
)
UNCERTAINTY_STAT_KEYS = ("sigma_mlm", "sigma_phylo100", "sigma_phylo470", "sigma_structure")
REGION_UNCERTAINTY_STAT_KEYS = ("sigma_region",)
AA_UNCERTAINTY_STAT_KEYS = ("sigma_aa", "sigma_codon_phylo", "sigma_codon")
MUTATION_EFFECT_UNCERTAINTY_STAT_KEYS = ("sigma_mutation_effect", "sigma_counterfactual", "sigma_encode")
ALLELE_UNCERTAINTY_STAT_KEYS = ("sigma_allele",)
COUNTERFACTUAL_STAT_KEYS = (
    "cf_active_fraction",
    "cf_local_cosine",
    "cf_local_margin_loss",
    "cf_far_distance",
    "counterfactual_effective_weight",
    "allele_effect_accuracy",
    "allele_severity_mae",
)
AUXILIARY_COMPUTE_STAT_KEYS = (
    "allele_active_rows",
    "allele_scored_alts",
    "allele_score_seq_len",
    "allele_rank_step",
    "allele_compute_active",
    "counterfactual_compute_active",
)
COMPOSITION_STAT_KEYS = (
    "n_fraction",
    "splice_positive_fraction",
    "splice_core_fraction",
    "exon_fraction",
    "cds_fraction",
    "utr_fraction",
    "intron_fraction",
    "n_filter_fallback_fraction",
    "mask_density_intergenic",
    "mask_density_intron",
    "mask_density_noncoding_exon",
    "mask_density_utr",
    "mask_density_cds",
)
STEP_STAT_KEYS = (
    LOSS_STAT_KEYS
    + COUNT_STAT_KEYS
    + WEIGHT_STAT_KEYS
    + MLM_REGION_LOSS_STAT_KEYS
    + UNCERTAINTY_STAT_KEYS
    + REGION_UNCERTAINTY_STAT_KEYS
    + AA_UNCERTAINTY_STAT_KEYS
    + MUTATION_EFFECT_UNCERTAINTY_STAT_KEYS
    + ALLELE_UNCERTAINTY_STAT_KEYS
    + COUNTERFACTUAL_STAT_KEYS
    + AUXILIARY_COMPUTE_STAT_KEYS
    + COMPOSITION_STAT_KEYS
)


# ---------------------------------------------------------------------------
# Distributed training context
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DistributedContext:
    distributed: bool = False
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def setup_distributed() -> DistributedContext:
    """Initialize DDP if torchrun environment variables are present."""
    rank_env = os.environ.get("RANK")
    local_rank_env = os.environ.get("LOCAL_RANK")
    world_size_env = os.environ.get("WORLD_SIZE")

    if rank_env is None or local_rank_env is None or world_size_env is None:
        return DistributedContext()

    rank = int(rank_env)
    local_rank = int(local_rank_env)
    world_size = int(world_size_env)

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", device_id=torch.device("cuda", local_rank))

    return DistributedContext(distributed=True, rank=rank, local_rank=local_rank, world_size=world_size)


def cleanup_distributed(ctx: DistributedContext) -> None:
    if ctx.distributed:
        dist.destroy_process_group()


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying model, stripping DDP wrapper if present."""
    return model.module if isinstance(model, DDP) else model


def ddp_kwargs_for_train_config(cfg: TrainConfig) -> dict[str, Any]:
    """Resolve DDP reducer settings for the configured training graph."""
    uses_reentrant_checkpointing = bool(cfg.model_config.get("activation_checkpointing", False)) and bool(
        cfg.model_config.get("checkpoint_use_reentrant", True)
    )
    if cfg.model == "beat-v10" and not uses_reentrant_checkpointing:
        return {"find_unused_parameters": True}
    if cfg.model == "beat-v9" and not uses_reentrant_checkpointing:
        return {"find_unused_parameters": True}
    if _uses_alternating_auxiliary_schedule(cfg) and not uses_reentrant_checkpointing:
        return {"find_unused_parameters": True}
    return {"static_graph": True}


def _allreduce_log_sigma_grads(log_sigmas: dict[str, torch.Tensor], world_size: int) -> None:
    """Average log_sigma gradients across DDP ranks."""
    for param in log_sigmas.values():
        if param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad.div_(world_size)


def _allreduce_scalar_stats(
    scalar_stats: dict[str, float],
    keys: tuple[str, ...],
    device: torch.device,
    world_size: int,
) -> dict[str, float]:
    """Average selected scalar stats across DDP ranks in-place."""
    if not keys:
        return scalar_stats
    vals = torch.tensor([scalar_stats[k] for k in keys], dtype=torch.float64, device=device)
    dist.all_reduce(vals, op=dist.ReduceOp.SUM)
    vals /= world_size
    result = dict(scalar_stats)
    for i, k in enumerate(keys):
        result[k] = float(vals[i])
    return result


def _allreduce_sum_stats(
    scalar_stats: dict[str, float],
    keys: tuple[str, ...],
    device: torch.device,
) -> dict[str, float]:
    """Sum selected scalar stats across DDP ranks in-place."""
    if not keys:
        return scalar_stats
    vals = torch.tensor([scalar_stats[k] for k in keys], dtype=torch.float64, device=device)
    dist.all_reduce(vals, op=dist.ReduceOp.SUM)
    result = dict(scalar_stats)
    for i, k in enumerate(keys):
        result[k] = float(vals[i])
    return result


@dataclass
class TrainConfig:
    fasta_path: str = DEFAULT_FASTA_PATH
    phylo100_bw_path: str = DEFAULT_PHYLO100_BW_PATH
    phylo470_bw_path: str = DEFAULT_PHYLO470_BW_PATH
    gtf_path: str = DEFAULT_GTF_PATH
    dataset_mode: DatasetMode = "hg38_splice_phylo"
    abraom_data_root: str | None = None
    abraom_dataset_arm: AbraomDatasetArm = "abraom_weighted"
    abraom_train_split: str = "train"
    abraom_val_split: str = "val"
    abraom_max_shards_per_split: int | None = None
    abraom_shuffle_shards: bool = True
    chromosomes: list[str] | None = None
    train_chromosomes: list[str] | None = None
    val_chromosomes: list[str] | None = None
    chromosome_split_mode: ChromosomeSplitMode = "full_hg38"

    model: str = DEFAULT_MODEL_KEY
    model_config: dict[str, Any] = field(default_factory=dict)
    seq_len: int = 4096
    length_warmup_enabled: bool = False
    length_warmup_initial_seq_len: int = 16384
    length_warmup_transition_fraction: float = 0.8
    length_warmup_stage1_warmup_fraction: float = 0.05
    length_warmup_stage1_end_lr_scale: float = 0.1
    length_warmup_stage2_warmup_fraction: float = 0.05
    length_warmup_stage2_peak_lr_scale: float = 0.2
    length_warmup_final_lr_scale: float = 0.01
    num_workers: int = 0
    prefetch_factor: int = 4

    mask_prob: float = 0.15
    mean_span_len: int = 8
    conservation_mix: float = 0.0
    cds_enrichment_fraction: float = 0.0
    splice_window_oversample_fraction: float = 0.0
    hard_position_mix: float = 0.0
    hard_position_ema_decay: float = 0.99
    hard_position_warmup_steps: int = 500
    counterfactual_fraction: float = 0.0
    counterfactual_local_radius: int = 8
    counterfactual_far_radius: int = 64
    counterfactual_local_similarity_target: float = 0.8
    counterfactual_weighting: CounterfactualWeightingMode = "uncertainty"
    allele_fraction: float = 0.0
    allele_score_window: int = 4096
    allele_max_rows_per_batch: int = 2
    allele_rank_every_n_steps: int = 4
    auxiliary_schedule: AuxiliaryScheduleMode = "all"
    clinvar_blocklist_bed_path: str | None = "data/clinvar/blocklist_v1.bed"
    encode_track_specs: list[dict[str, Any]] = field(default_factory=list)
    curriculum: dict[str, Any] = field(default_factory=dict)

    batch_size: int = 1
    max_steps: int = 1_000
    lr: float = 3e-4
    lr_scheduler: LRSchedulerName = "cosine"
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_steps: int = 100
    log_every: int = 10
    eval_every: int = 100
    save_every: int = 100
    max_checkpoints_to_keep: int = 3

    grad_accum_steps: int = 1

    w_mlm: float = 1.0
    w_phylo100: float = 0.25
    w_phylo470: float = 0.25
    w_structure: float = 0.25
    w_rc: float = 0.0
    w_region: float = 0.0
    w_aa: float = 0.0
    w_codon_phylo: float = 0.0
    w_mutation_effect: float = 0.15
    w_counterfactual: float = 0.0
    w_allele: float = 0.0
    w_codon: float = 0.0
    w_encode: float = 0.0
    w_conservation_bin: float = 0.0
    w_splice_distance: float = 0.0
    w_codon_pos: float = 0.0
    w_exon_phase: float = 0.0
    w_counterfactual_snv: float = 0.0
    w_counterfactual_severity: float = 0.0
    lambda_mutation_effect_rank: float = 0.5
    lambda_allele_rank: float = 0.5
    lambda_allele_severity: float = 1.0
    lambda_allele_swap: float = 0.25
    lambda_allele_far: float = 0.1
    splice_class_weight_cap: float = 8.0
    region_class_weight_cap: float = 4.0
    aa_class_weight_cap: float = 4.0
    rc_loss_type: str = "cosine"
    use_uncertainty_weighting: bool = False
    loss_balancing: LossBalancingMode = "uncertainty"
    aux_loss_warmup_steps: int = 0
    uncertainty_init_log_sigma: dict[str, float] | None = None

    phylo_weighted_mlm: bool = False
    phylo_mlm_boost: float = 2.0

    seed: int = 42
    output_dir: str = DEFAULT_OUTPUT_DIR
    resume_from: str | None = None
    init_from_checkpoint: str | None = None
    precision: PrecisionMode = "auto"
    allow_tf32: bool = True
    memory_preflight_enabled: bool = False

    wandb_enabled: bool = False
    wandb_project: str = "lumina"
    wandb_entity: str | None = None
    wandb_run_name: str | None = None
    wandb_tags: list[str] | None = None
    param_budget: dict[str, Any] = field(default_factory=dict)


TRAIN_CONFIG_FIELD_NAMES = frozenset(field.name for field in fields(TrainConfig))
CSV_CONFIG_FIELDS = frozenset({"chromosomes", "train_chromosomes", "val_chromosomes", "wandb_tags"})
IGNORED_LEGACY_CONFIG_FIELDS = frozenset({"samples_per_epoch"})


@dataclass
class TrainStepResult:
    stats: dict[str, float]
    metrics: MetricAccumulator
    grad_norm: float


@dataclass
class BestCheckpointState:
    step: int = 0
    loss: float = float("inf")
    path: str | None = None

    def should_update(self, scalar_stats: dict[str, float]) -> bool:
        return float(scalar_stats["loss"]) < self.loss - 1e-12

    def update(self, step: int, scalar_stats: dict[str, float], path: Path) -> None:
        self.step = step
        self.loss = float(scalar_stats["loss"])
        self.path = str(path)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "loss": self.loss,
            "path": self.path,
        }

    @classmethod
    def from_metadata(cls, metadata: Any) -> BestCheckpointState:
        if not isinstance(metadata, dict):
            return cls()
        return cls(
            step=int(metadata.get("step", 0)),
            loss=float(metadata.get("loss", float("inf"))),
            path=cast(str | None, metadata.get("path")),
        )


@dataclass(frozen=True)
class LengthWarmupState:
    enabled: bool
    phase: int
    seq_len: int
    phase_step: int
    phase_steps: int


@dataclass(frozen=True)
class TrainingComponents:
    optimizer: torch.optim.Optimizer
    effective_balancing: LossBalancingMode
    balancing_keys: tuple[str, ...]
    log_sigmas: dict[str, torch.Tensor] | None = None
    gradnorm_balancer: GradNormBalancer | None = None


@dataclass(frozen=True)
class SamplingScheduleState:
    curriculum_phase: str
    conservation_mix: float
    cds_enrichment_fraction: float
    splice_window_oversample_fraction: float
    counterfactual_fraction: float
    allele_fraction: float
    hard_position_mix: float


@dataclass(frozen=True)
class MemoryPreflightPhaseSpec:
    phase: int
    seq_len: int
    phase_start_step: int
    auxiliary_path: str = "standard"


@dataclass(frozen=True)
class AuxiliaryComputeState:
    allele: bool
    counterfactual: bool
    allele_rank_step: bool


@dataclass(frozen=True)
class PreparedAlleleScoringBatch:
    loss_batch: dict[str, Any]
    scorer_kwargs: dict[str, torch.Tensor]
    active_rows: int
    scored_alts: int
    score_seq_len: int


class MemoryPreflightError(RuntimeError):
    def __init__(self, message: str, report: dict[str, Any]) -> None:
        super().__init__(message)
        self.report = report


class SmoothedValue:
    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float) -> None:
        self.total += float(value)
        self.count += 1

    @property
    def avg(self) -> float:
        return self.total / max(1, self.count)

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0


def get_device(ctx: DistributedContext | None = None) -> torch.device:
    if ctx is not None and ctx.distributed:
        return torch.device("cuda", ctx.local_rank)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None

    items = [part.strip() for part in value.split(",") if part.strip()]
    return items or None


def parse_csv_or_list(value: Any, field_name: str) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return parse_csv(value)
    if isinstance(value, list):
        items: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise TypeError(f"Field {field_name!r} must contain only strings, got {type(item).__name__}.")
            stripped = item.strip()
            if stripped:
                items.append(stripped)
        return items or None
    raise TypeError(f"Field {field_name!r} must be a comma-separated string or a list of strings.")


def normalize_model_config_mapping(value: Any, *, source: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"{source} must be a mapping, got {type(value).__name__}.")

    normalized: dict[str, Any] = {}
    for raw_key, nested_value in value.items():
        if not isinstance(raw_key, str):
            raise TypeError(f"All keys in {source} must be strings, got {type(raw_key).__name__}.")
        key = raw_key.replace("-", "_")
        if key in normalized:
            raise ValueError(f"Duplicate config key {key!r} detected in {source}.")
        normalized[key] = nested_value
    return normalized


def legacy_top_level_key_error(source: str, legacy_keys: list[str]) -> ValueError:
    joined = ", ".join(legacy_keys)
    return ValueError(
        f"Legacy config keys in {source}: {joined}. "
        "The preset interface has been removed; use `model: bimamba` and put architecture overrides under "
        "the nested `model_config:` mapping."
    )


def canonicalize_train_config_keys(raw_config: dict[str, Any], source: str) -> dict[str, Any]:
    canonical: dict[str, Any] = {}
    for raw_key, value in raw_config.items():
        if not isinstance(raw_key, str):
            raise TypeError(f"All keys in {source} must be strings, got {type(raw_key).__name__}.")
        key = raw_key.replace("-", "_")
        if key in IGNORED_LEGACY_CONFIG_FIELDS:
            continue
        if key in canonical:
            raise ValueError(f"Duplicate config key {key!r} detected in {source}.")
        canonical[key] = value
    legacy_keys = sorted(set(canonical) & LEGACY_TOP_LEVEL_CONFIG_FIELDS)
    if legacy_keys:
        raise legacy_top_level_key_error(source, legacy_keys)
    unknown_keys = sorted(set(canonical) - TRAIN_CONFIG_FIELD_NAMES)
    if unknown_keys:
        raise ValueError(f"Unknown config keys in {source}: {', '.join(unknown_keys)}")
    return canonical


def normalize_train_config_overrides(raw_config: dict[str, Any], source: str) -> dict[str, Any]:
    canonical = canonicalize_train_config_keys(raw_config, source)
    normalized: dict[str, Any] = {}
    for key, value in canonical.items():
        if key in CSV_CONFIG_FIELDS:
            normalized[key] = parse_csv_or_list(value, key)
        elif key == "model_config":
            normalized[key] = normalize_model_config_mapping(value, source=f"field 'model_config' in {source}")
        else:
            normalized[key] = value
    return normalized


_INHERIT_KEY = "_inherit"
_MAX_INHERIT_DEPTH = 8


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge *override* on top of *base*, recursing into nested dicts (e.g. ``model_config``)."""
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml_file(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required to load --config files. Install the project dependencies first."
        ) from exc

    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must contain a top-level mapping.")
    return dict(data)


def _resolve_inherit_chain(path: Path, *, _depth: int = 0) -> dict[str, Any]:
    """Load a YAML config and recursively resolve its ``_inherit`` chain."""
    if _depth > _MAX_INHERIT_DEPTH:
        raise ValueError(
            f"Config inheritance depth exceeded {_MAX_INHERIT_DEPTH} "
            f"(possible cycle). Last file: {path}"
        )
    data = _load_yaml_file(path)
    inherit_rel = data.pop(_INHERIT_KEY, None)
    if inherit_rel is None:
        return data
    if not isinstance(inherit_rel, str):
        raise TypeError(f"{_INHERIT_KEY} in {path} must be a string path, got {type(inherit_rel).__name__}.")
    parent_path = (path.parent / inherit_rel).resolve()
    if not parent_path.is_file():
        raise FileNotFoundError(f"Inherited config not found: {parent_path} (referenced by {path})")
    parent_data = _resolve_inherit_chain(parent_path, _depth=_depth + 1)
    return _deep_merge(parent_data, data)


def load_yaml_train_config(path: Path) -> dict[str, Any]:
    resolved_path = resolve_repo_relative_path(path)
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Config file not found: {path} (resolved to {resolved_path})")
    resolved = _resolve_inherit_chain(resolved_path.resolve())
    return normalize_train_config_overrides(resolved, source=f"config file {resolved_path}")


def parse_chromosome_csv(value: str | None) -> list[str] | None:
    return parse_csv(value)


def dedupe_chromosomes(chromosomes: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for chrom in chromosomes:
        if chrom not in seen:
            seen.add(chrom)
            deduped.append(chrom)
    return deduped


def validate_train_config(cfg: TrainConfig) -> None:
    resolved_model_config = resolve_model_config_dict(cfg.model, cfg.model_config, source="TrainConfig.model_config")
    encode_track_specs = resolve_encode_track_specs(cfg)
    num_encode_tracks = len(encode_track_specs)
    model_num_encode_tracks = int(resolved_model_config.get("num_encode_tracks", 0))
    if num_encode_tracks == 0 and cfg.w_encode != 0.0:
        raise ValueError("w_encode requires a non-empty encode_track_specs list.")
    if num_encode_tracks > 0 and "num_encode_tracks" not in resolved_model_config:
        raise ValueError("encode_track_specs require a model family that exposes model_config.num_encode_tracks.")
    if model_num_encode_tracks != num_encode_tracks:
        raise ValueError(
            "model_config.num_encode_tracks must match len(encode_track_specs), "
            f"got num_encode_tracks={model_num_encode_tracks} and {num_encode_tracks} track specs."
        )
    if cfg.lr_scheduler not in LR_SCHEDULER_CHOICES:
        choices = ", ".join(LR_SCHEDULER_CHOICES)
        raise ValueError(f"Unsupported lr_scheduler {cfg.lr_scheduler!r}. Expected one of: {choices}.")
    if cfg.chromosome_split_mode not in CHROMOSOME_SPLIT_MODE_CHOICES:
        choices = ", ".join(CHROMOSOME_SPLIT_MODE_CHOICES)
        raise ValueError(
            f"Unsupported chromosome_split_mode {cfg.chromosome_split_mode!r}. Expected one of: {choices}."
        )
    if cfg.counterfactual_weighting not in COUNTERFACTUAL_WEIGHTING_CHOICES:
        choices = ", ".join(COUNTERFACTUAL_WEIGHTING_CHOICES)
        raise ValueError(
            f"Unsupported counterfactual_weighting {cfg.counterfactual_weighting!r}. Expected one of: {choices}."
        )
    if cfg.auxiliary_schedule not in AUXILIARY_SCHEDULE_CHOICES:
        choices = ", ".join(AUXILIARY_SCHEDULE_CHOICES)
        raise ValueError(f"Unsupported auxiliary_schedule {cfg.auxiliary_schedule!r}. Expected one of: {choices}.")
    if cfg.dataset_mode not in DATASET_MODE_CHOICES:
        choices = ", ".join(DATASET_MODE_CHOICES)
        raise ValueError(f"Unsupported dataset_mode {cfg.dataset_mode!r}. Expected one of: {choices}.")
    if cfg.resume_from is not None and cfg.init_from_checkpoint is not None:
        raise ValueError("resume_from and init_from_checkpoint are mutually exclusive.")
    if cfg.dataset_mode == "abraom_sequences":
        if not cfg.abraom_data_root:
            raise ValueError("dataset_mode='abraom_sequences' requires abraom_data_root.")
        if cfg.length_warmup_enabled:
            raise ValueError("ABRAOM generated sequences have fixed length; disable length_warmup_enabled.")
        if cfg.abraom_max_shards_per_split is not None and cfg.abraom_max_shards_per_split <= 0:
            raise ValueError("abraom_max_shards_per_split must be positive when provided.")
    if cfg.lr_scheduler == "cosine" and cfg.min_lr > cfg.lr:
        raise ValueError(
            f"Cosine lr_scheduler requires min_lr <= lr, but got min_lr={cfg.min_lr} and lr={cfg.lr}."
        )
    if cfg.lr_scheduler == "cosine_rewarm" and not cfg.length_warmup_enabled:
        raise ValueError("cosine_rewarm lr_scheduler requires length_warmup_enabled=True.")
    for field_name, value in (
        ("conservation_mix", cfg.conservation_mix),
        ("cds_enrichment_fraction", cfg.cds_enrichment_fraction),
        ("splice_window_oversample_fraction", cfg.splice_window_oversample_fraction),
        ("hard_position_mix", cfg.hard_position_mix),
        ("counterfactual_fraction", cfg.counterfactual_fraction),
        ("allele_fraction", cfg.allele_fraction),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{field_name} must be in [0, 1], got {value}.")
    if cfg.counterfactual_local_radius < 0 or cfg.counterfactual_far_radius < 0:
        raise ValueError("counterfactual radii must be non-negative.")
    for field_name, value in (
        ("allele_score_window", cfg.allele_score_window),
        ("allele_max_rows_per_batch", cfg.allele_max_rows_per_batch),
        ("allele_rank_every_n_steps", cfg.allele_rank_every_n_steps),
    ):
        if value <= 0:
            raise ValueError(f"{field_name} must be positive, got {value}.")
    for field_name, value in (
        ("splice_class_weight_cap", cfg.splice_class_weight_cap),
        ("region_class_weight_cap", cfg.region_class_weight_cap),
        ("aa_class_weight_cap", cfg.aa_class_weight_cap),
    ):
        if value <= 0.0:
            raise ValueError(f"{field_name} must be positive, got {value}.")
    if not -1.0 <= cfg.counterfactual_local_similarity_target <= 1.0:
        raise ValueError(
            "counterfactual_local_similarity_target must be in [-1, 1], "
            f"got {cfg.counterfactual_local_similarity_target}."
        )

    if cfg.length_warmup_enabled:
        if cfg.length_warmup_initial_seq_len <= 0:
            raise ValueError("length_warmup_initial_seq_len must be positive when length warmup is enabled.")
        if cfg.length_warmup_initial_seq_len >= cfg.seq_len:
            raise ValueError(
                "length_warmup_initial_seq_len must be smaller than seq_len when length warmup is enabled."
            )
        if not 0.0 < cfg.length_warmup_transition_fraction < 1.0:
            raise ValueError("length_warmup_transition_fraction must be between 0 and 1.")
        for field_name, value in (
            ("length_warmup_stage1_warmup_fraction", cfg.length_warmup_stage1_warmup_fraction),
            ("length_warmup_stage2_warmup_fraction", cfg.length_warmup_stage2_warmup_fraction),
        ):
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{field_name} must be in [0, 1).")
        for field_name, value in (
            ("length_warmup_stage1_end_lr_scale", cfg.length_warmup_stage1_end_lr_scale),
            ("length_warmup_stage2_peak_lr_scale", cfg.length_warmup_stage2_peak_lr_scale),
            ("length_warmup_final_lr_scale", cfg.length_warmup_final_lr_scale),
        ):
            if value < 0.0:
                raise ValueError(f"{field_name} must be non-negative.")
        if cfg.length_warmup_stage1_end_lr_scale > cfg.length_warmup_stage2_peak_lr_scale:
            raise ValueError(
                "length_warmup_stage1_end_lr_scale must be <= length_warmup_stage2_peak_lr_scale."
            )
        if cfg.length_warmup_final_lr_scale > cfg.length_warmup_stage2_peak_lr_scale:
            raise ValueError("length_warmup_final_lr_scale must be <= length_warmup_stage2_peak_lr_scale.")
        stage1_steps, stage2_steps = resolve_length_warmup_stage_steps(cfg)
        if stage1_steps <= 0 or stage2_steps <= 0:
            raise ValueError("length warmup requires at least one step in each phase.")


def resolve_train_config(cfg: TrainConfig) -> TrainConfig:
    validate_train_config(cfg)
    spec = get_model_spec(cfg.model)
    resolved_model_config = resolve_model_config_dict(spec.key, cfg.model_config, source="TrainConfig.model_config")
    return replace(cfg, model=spec.key, model_config=resolved_model_config)


def normalize_runtime_train_config(
    cfg: TrainConfig,
    *,
    device: torch.device,
    precision: PrecisionPolicy,
) -> tuple[TrainConfig, list[str]]:
    resolved = resolve_model_config_dict(cfg.model, cfg.model_config, source="TrainConfig.model_config")
    normalized_model_config, notes = normalize_mamba3_runtime_config(
        cfg.model,
        resolved,
        uses_bf16_compute=precision.uses_bf16_compute,
        cuda_device_capability=get_cuda_device_capability(device),
    )
    if normalized_model_config == cfg.model_config:
        return cfg, notes
    return replace(cfg, model_config=normalized_model_config), notes


def resolve_chromosome_splits(cfg: TrainConfig) -> tuple[list[str], list[str]]:
    if cfg.chromosome_split_mode == "heldout":
        val_chromosomes = dedupe_chromosomes(cfg.val_chromosomes or list(DEFAULT_VAL_CHROMOSOMES))
        train_source = cfg.train_chromosomes or cfg.chromosomes

        if train_source is None:
            val_set = set(val_chromosomes)
            train_chromosomes = [chrom for chrom in DEFAULT_CHROMOSOMES if chrom not in val_set]
        else:
            train_chromosomes = dedupe_chromosomes(list(train_source))

        if not train_chromosomes:
            raise ValueError("Resolved training chromosome split is empty.")
        if not val_chromosomes:
            raise ValueError("Resolved validation chromosome split is empty.")

        overlap = sorted(set(train_chromosomes) & set(val_chromosomes))
        if overlap:
            raise ValueError(f"Train/validation chromosome split overlaps: {overlap}")
        return train_chromosomes, val_chromosomes

    explicit_train = cfg.train_chromosomes
    explicit_val = cfg.val_chromosomes
    if (explicit_train is None) != (explicit_val is None):
        raise ValueError(
            "full_hg38 chromosome_split_mode requires both train_chromosomes and val_chromosomes when overriding "
            "either one."
        )

    if explicit_train is not None and explicit_val is not None:
        train_chromosomes = dedupe_chromosomes(list(explicit_train))
        val_chromosomes = dedupe_chromosomes(list(explicit_val))
    else:
        shared_source = cfg.chromosomes or list(DEFAULT_CHROMOSOMES)
        shared = dedupe_chromosomes(list(shared_source))
        train_chromosomes = list(shared)
        val_chromosomes = list(shared)

    if not train_chromosomes:
        raise ValueError("Resolved training chromosome split is empty.")
    if not val_chromosomes:
        raise ValueError("Resolved validation chromosome split is empty.")
    return train_chromosomes, val_chromosomes


def seed_everything(seed: int, rank: int = 0) -> None:
    effective_seed = seed + rank
    random.seed(effective_seed)
    torch.manual_seed(effective_seed)
    torch.cuda.manual_seed_all(effective_seed)


def aux_loss_warmup_factor(step: int, warmup_steps: int) -> float:
    """Linear ramp from 0 to 1 over *warmup_steps*. Returns 1.0 when disabled (warmup_steps <= 0)."""
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, step / warmup_steps)


def resolve_length_warmup_stage_steps(cfg: TrainConfig) -> tuple[int, int]:
    if not cfg.length_warmup_enabled:
        return cfg.max_steps, 0
    stage1_steps = int(cfg.max_steps * cfg.length_warmup_transition_fraction)
    stage2_steps = cfg.max_steps - stage1_steps
    return stage1_steps, stage2_steps


def resolve_length_warmup_state(cfg: TrainConfig, step: int) -> LengthWarmupState:
    clamped_step = min(max(step, 0), cfg.max_steps)
    if not cfg.length_warmup_enabled:
        return LengthWarmupState(
            enabled=False,
            phase=0,
            seq_len=cfg.seq_len,
            phase_step=clamped_step,
            phase_steps=cfg.max_steps,
        )

    stage1_steps, stage2_steps = resolve_length_warmup_stage_steps(cfg)
    if clamped_step <= stage1_steps:
        return LengthWarmupState(
            enabled=True,
            phase=1,
            seq_len=cfg.length_warmup_initial_seq_len,
            phase_step=clamped_step,
            phase_steps=stage1_steps,
        )
    return LengthWarmupState(
        enabled=True,
        phase=2,
        seq_len=cfg.seq_len,
        phase_step=clamped_step - stage1_steps,
        phase_steps=stage2_steps,
    )


def _uses_alternating_auxiliary_schedule(cfg: TrainConfig) -> bool:
    return (
        cfg.auxiliary_schedule == "alternate_allele_counterfactual"
        and cfg.w_allele != 0.0
        and cfg.w_counterfactual != 0.0
    )


def _step_for_auxiliary_path(step: int, auxiliary_path: str) -> int:
    if auxiliary_path == "allele":
        return step if step % 2 == 1 else step + 1
    if auxiliary_path == "counterfactual":
        return step if step % 2 == 0 else step + 1
    return step


def _first_auxiliary_active_step(cfg: TrainConfig, phase_start_step: int) -> int:
    curriculum = cfg.curriculum if isinstance(cfg.curriculum, dict) else {}
    if not bool(curriculum.get("enabled", False)):
        return phase_start_step
    warmup_fraction = float(curriculum.get("counterfactual_warmup_fraction", 0.05))
    warmup_steps = max(0, int(cfg.max_steps * warmup_fraction))
    return max(phase_start_step, warmup_steps + 1)


def _expand_memory_preflight_auxiliary_paths(
    cfg: TrainConfig,
    specs: list[MemoryPreflightPhaseSpec],
) -> list[MemoryPreflightPhaseSpec]:
    if not _uses_alternating_auxiliary_schedule(cfg):
        return specs
    expanded: list[MemoryPreflightPhaseSpec] = []
    for spec in specs:
        base_step = _first_auxiliary_active_step(cfg, spec.phase_start_step)
        for auxiliary_path in ("allele", "counterfactual"):
            expanded.append(
                MemoryPreflightPhaseSpec(
                    phase=spec.phase,
                    seq_len=spec.seq_len,
                    phase_start_step=_step_for_auxiliary_path(base_step, auxiliary_path),
                    auxiliary_path=auxiliary_path,
                )
            )
    return expanded


def resolve_memory_preflight_phases(cfg: TrainConfig) -> list[MemoryPreflightPhaseSpec]:
    if not cfg.length_warmup_enabled:
        return _expand_memory_preflight_auxiliary_paths(
            cfg,
            [MemoryPreflightPhaseSpec(phase=0, seq_len=cfg.seq_len, phase_start_step=1)],
        )

    stage1_steps, _stage2_steps = resolve_length_warmup_stage_steps(cfg)
    return _expand_memory_preflight_auxiliary_paths(
        cfg,
        [
            MemoryPreflightPhaseSpec(phase=1, seq_len=cfg.length_warmup_initial_seq_len, phase_start_step=1),
            MemoryPreflightPhaseSpec(phase=2, seq_len=cfg.seq_len, phase_start_step=stage1_steps + 1),
        ],
    )


def resolve_phase_warmup_steps(phase_steps: int, warmup_fraction: float) -> int:
    if phase_steps <= 1 or warmup_fraction <= 0.0:
        return 0
    return min(phase_steps - 1, max(1, int(phase_steps * warmup_fraction)))


def estimate_tokens_seen(cfg: TrainConfig, completed_steps: int, *, world_size: int = 1) -> int:
    clamped_steps = min(max(completed_steps, 0), cfg.max_steps)
    phase2_tokens_per_step = cfg.batch_size * cfg.grad_accum_steps * world_size * cfg.seq_len
    if not cfg.length_warmup_enabled:
        return clamped_steps * phase2_tokens_per_step

    stage1_steps, _stage2_steps = resolve_length_warmup_stage_steps(cfg)
    stage1_completed = min(clamped_steps, stage1_steps)
    stage2_completed = max(0, clamped_steps - stage1_steps)
    phase1_tokens_per_step = cfg.batch_size * cfg.grad_accum_steps * world_size * cfg.length_warmup_initial_seq_len
    return stage1_completed * phase1_tokens_per_step + stage2_completed * phase2_tokens_per_step


def build_memory_preflight_report(
    *,
    enabled: bool,
    status: str,
    device: torch.device,
    phases: list[dict[str, Any]] | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "enabled": enabled,
        "status": status,
        "device": device.type,
        "phases": list(phases or []),
    }
    if reason is not None:
        report["reason"] = reason
    return report


def _bytes_to_gib(num_bytes: int | None) -> float | None:
    if num_bytes is None:
        return None
    return num_bytes / float(1024**3)


def _format_gib(num_bytes: int | None) -> str:
    gib = _bytes_to_gib(num_bytes)
    if gib is None:
        return "n/a"
    return f"{gib:.2f}GiB"


def _is_cuda_oom_error(exc: Exception) -> bool:
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    if not isinstance(exc, RuntimeError):
        return False
    message = str(exc).lower()
    return "out of memory" in message and ("cuda" in message or "cudnn" in message or "cublas" in message)


def _is_checkpoint_runtime_incompatible_error(exc: Exception) -> bool:
    if not isinstance(exc, RuntimeError):
        return False
    message = str(exc).lower()
    return "torch.utils.checkpoint" in message and "already unpacked once" in message


def _clear_cuda_memory(device: torch.device) -> None:
    if device.type != "cuda":
        return
    with contextlib.suppress(Exception):
        torch.cuda.synchronize(device)
    with contextlib.suppress(Exception):
        torch.cuda.empty_cache()


def _gather_memory_preflight_phase_results(
    local_result: dict[str, Any],
    ctx: DistributedContext,
) -> list[dict[str, Any]]:
    if not ctx.distributed:
        return [local_result]

    gathered: list[dict[str, Any] | None] = [None for _ in range(ctx.world_size)]
    dist.all_gather_object(gathered, local_result)
    return [cast(dict[str, Any], result) for result in gathered]


def _aggregate_memory_preflight_phase_results(
    spec: MemoryPreflightPhaseSpec,
    phase_results: list[dict[str, Any]],
) -> dict[str, Any]:
    passed_results = [result for result in phase_results if result.get("status") == "passed"]
    aggregate: dict[str, Any] = {
        "phase": spec.phase,
        "seq_len": spec.seq_len,
        "phase_start_step": spec.phase_start_step,
        "auxiliary_path": spec.auxiliary_path,
        "status": "passed",
        "max_memory_allocated_bytes": None,
        "max_memory_reserved_bytes": None,
    }
    if passed_results:
        allocated_values = [int(result["max_memory_allocated_bytes"]) for result in passed_results]
        reserved_values = [int(result["max_memory_reserved_bytes"]) for result in passed_results]
        aggregate["max_memory_allocated_bytes"] = max(allocated_values)
        aggregate["max_memory_reserved_bytes"] = max(reserved_values)
        aggregate["resolved_precision"] = passed_results[0].get("resolved_precision")

    failed_result = next((result for result in phase_results if result.get("status") != "passed"), None)
    if failed_result is not None:
        aggregate["status"] = "failed"
        aggregate["failed_rank"] = int(failed_result["rank"])
        aggregate["resolved_precision"] = failed_result.get("resolved_precision")
        aggregate["error_type"] = failed_result.get("error_type")
        aggregate["error"] = failed_result.get("error")
    if len(phase_results) > 1:
        aggregate["rank_results"] = phase_results
    return aggregate


def get_lr(step: int, cfg: TrainConfig) -> float:
    if cfg.lr_scheduler != "cosine_rewarm" and step < cfg.warmup_steps:
        return cfg.lr * step / max(1, cfg.warmup_steps)

    if cfg.lr_scheduler == "constant":
        return cfg.lr
    if cfg.lr_scheduler == "cosine":
        progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return cfg.min_lr + (cfg.lr - cfg.min_lr) * cosine
    if cfg.lr_scheduler == "cosine_rewarm":
        state = resolve_length_warmup_state(cfg, step)
        if state.phase == 1:
            warmup_steps = resolve_phase_warmup_steps(state.phase_steps, cfg.length_warmup_stage1_warmup_fraction)
            if state.phase_step <= warmup_steps:
                return cfg.lr * state.phase_step / max(1, warmup_steps)
            progress = (state.phase_step - warmup_steps) / max(1, state.phase_steps - warmup_steps)
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            end_lr = cfg.lr * cfg.length_warmup_stage1_end_lr_scale
            return end_lr + (cfg.lr - end_lr) * cosine

        warmup_steps = resolve_phase_warmup_steps(state.phase_steps, cfg.length_warmup_stage2_warmup_fraction)
        start_lr = cfg.lr * cfg.length_warmup_stage1_end_lr_scale
        peak_lr = cfg.lr * cfg.length_warmup_stage2_peak_lr_scale
        final_lr = cfg.lr * cfg.length_warmup_final_lr_scale
        if state.phase_step <= warmup_steps:
            progress = state.phase_step / max(1, warmup_steps)
            return start_lr + (peak_lr - start_lr) * progress
        progress = (state.phase_step - warmup_steps) / max(1, state.phase_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return final_lr + (peak_lr - final_lr) * cosine
    choices = ", ".join(LR_SCHEDULER_CHOICES)
    raise ValueError(f"Unsupported lr_scheduler {cfg.lr_scheduler!r}. Expected one of: {choices}.")


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        if group.get("name") == "uncertainty":
            continue
        group["lr"] = lr


def resolve_effective_loss_balancing(cfg: TrainConfig) -> LossBalancingMode:
    # ``use_uncertainty_weighting`` is the legacy flag; ``loss_balancing`` is the
    # authoritative field. Keep the current semantics unchanged.
    effective_balancing: LossBalancingMode = cfg.loss_balancing
    if cfg.use_uncertainty_weighting and cfg.loss_balancing == "uncertainty":
        effective_balancing = "uncertainty"
    return effective_balancing


def resolve_optional_uncertainty_task_keys(cfg: TrainConfig) -> tuple[str, ...]:
    task_keys: list[str] = ["mlm", "phylo100", "phylo470", "structure"]
    optional_pairs = (
        ("region", cfg.w_region),
        ("aa", cfg.w_aa),
        ("codon_phylo", cfg.w_codon_phylo),
        ("mutation_effect", cfg.w_mutation_effect),
        ("allele", cfg.w_allele),
        ("codon", cfg.w_codon),
        ("encode", cfg.w_encode),
    )
    for key, weight in optional_pairs:
        if weight != 0.0:
            task_keys.append(key)
    if cfg.w_counterfactual != 0.0 and cfg.counterfactual_weighting == "uncertainty":
        task_keys.append("counterfactual")
    return tuple(task_keys)


def resolve_encode_track_specs(cfg: TrainConfig) -> list[EncodeTrackSpec]:
    specs: list[EncodeTrackSpec] = []
    for raw_spec in cfg.encode_track_specs:
        if not isinstance(raw_spec, dict):
            raise TypeError(
                f"Each entry in encode_track_specs must be a mapping, got {type(raw_spec).__name__}."
            )
        specs.append(
            EncodeTrackSpec(
                name=str(raw_spec["name"]),
                bw_path=resolve_repo_relative_path(Path(str(raw_spec["bw_path"]))),
                transform=cast(Any, raw_spec.get("transform", "asinh")),
                normalize=cast(Any, raw_spec.get("normalize", "per_chromosome_zscore")),
            )
        )
    return specs


def resolve_sampling_schedule_state(cfg: TrainConfig, step: int) -> SamplingScheduleState:
    conservation_mix = cfg.conservation_mix
    cds_enrichment_fraction = cfg.cds_enrichment_fraction
    splice_window_oversample_fraction = cfg.splice_window_oversample_fraction
    counterfactual_fraction = cfg.counterfactual_fraction
    allele_fraction = cfg.allele_fraction
    hard_position_mix = cfg.hard_position_mix if step > cfg.hard_position_warmup_steps else 0.0
    curriculum_phase = "genome_wide"

    curriculum = cfg.curriculum if isinstance(cfg.curriculum, dict) else {}
    if bool(curriculum.get("enabled", False)):
        exon_fraction = float(curriculum.get("exon_centric_fraction", 0.25))
        exon_steps = max(1, int(cfg.max_steps * exon_fraction))
        if step <= exon_steps:
            curriculum_phase = "exon_centric"
            conservation_mix = 0.75
            cds_enrichment_fraction = 0.6
            splice_window_oversample_fraction = 0.25

        counterfactual_warmup_fraction = float(curriculum.get("counterfactual_warmup_fraction", 0.05))
        counterfactual_warmup_steps = max(0, int(cfg.max_steps * counterfactual_warmup_fraction))
        if step <= counterfactual_warmup_steps:
            counterfactual_fraction = 0.0
            allele_fraction = 0.0

    return SamplingScheduleState(
        curriculum_phase=curriculum_phase,
        conservation_mix=conservation_mix,
        cds_enrichment_fraction=cds_enrichment_fraction,
        splice_window_oversample_fraction=splice_window_oversample_fraction,
        counterfactual_fraction=counterfactual_fraction,
        allele_fraction=allele_fraction,
        hard_position_mix=hard_position_mix,
    )


def build_training_components(
    cfg: TrainConfig,
    model: torch.nn.Module,
    device: torch.device,
) -> TrainingComponents:
    effective_balancing = resolve_effective_loss_balancing(cfg)
    task_keys = resolve_optional_uncertainty_task_keys(cfg)

    if effective_balancing == "uncertainty":
        init_log_sigmas = cfg.uncertainty_init_log_sigma or {}
        log_sigmas = cast(
            dict[str, torch.Tensor],
            {
                key: torch.nn.Parameter(
                    torch.tensor([float(init_log_sigmas.get(key, 0.0))], device=device, dtype=torch.float32)
                )
                for key in task_keys
            },
        )
        optimizer = AdamW(
            [
                {"params": list(model.parameters())},
                {"params": list(log_sigmas.values()), "weight_decay": 0.0, "name": "uncertainty"},
            ],
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=(cfg.beta1, cfg.beta2),
        )
        return TrainingComponents(
            optimizer=optimizer,
            effective_balancing=effective_balancing,
            balancing_keys=task_keys,
            log_sigmas=log_sigmas,
        )

    if effective_balancing == "gradnorm":
        optimizer = AdamW(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=(cfg.beta1, cfg.beta2),
        )
        return TrainingComponents(
            optimizer=optimizer,
            effective_balancing=effective_balancing,
            balancing_keys=task_keys,
            gradnorm_balancer=GradNormBalancer(task_keys),
        )

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(cfg.beta1, cfg.beta2),
    )
    return TrainingComponents(
        optimizer=optimizer,
        effective_balancing=effective_balancing,
        balancing_keys=(),
    )


def _worker_init_fn(worker_id: int) -> None:
    """Seed Python stdlib RNG per DataLoader worker from PyTorch's worker seed."""
    info = torch.utils.data.get_worker_info()
    if info is not None:
        random.seed(info.seed % (2**32))


def build_dataloader(
    cfg: TrainConfig,
    device: torch.device,
    chromosomes: list[str],
    *,
    dataset_split: str = "train",
    seq_len: int | None = None,
    sampling_state: SamplingScheduleState | None = None,
    region_loss_ema: RegionLossEMA | None = None,
) -> DataLoader:
    effective_seq_len = cfg.seq_len if seq_len is None else seq_len
    effective_sampling_state = sampling_state or resolve_sampling_schedule_state(cfg, step=1)
    encode_track_specs = resolve_encode_track_specs(cfg)
    if cfg.dataset_mode == "abraom_sequences":
        if encode_track_specs:
            raise ValueError("ABRAOM sequence dataset mode does not currently materialize ENCODE targets.")
        dataset = AbraomSequenceDataset(
            data_root=cast(str, cfg.abraom_data_root),
            dataset_arm=cfg.abraom_dataset_arm,
            split=dataset_split,
            fasta_path=cfg.fasta_path,
            phylo100_bw_path=cfg.phylo100_bw_path,
            phylo470_bw_path=cfg.phylo470_bw_path,
            gtf_path=cfg.gtf_path,
            seq_len=effective_seq_len,
            chromosomes=chromosomes,
            core_radius=2,
            region_radius=10,
            max_shards=cfg.abraom_max_shards_per_split,
            seed=cfg.seed + (0 if dataset_split == cfg.abraom_train_split else 10_000),
            shuffle_shards=cfg.abraom_shuffle_shards,
        )
    else:
        dataset = HG38SplicePhyloDataset(
            fasta_path=cfg.fasta_path,
            phylo100_bw_path=cfg.phylo100_bw_path,
            phylo470_bw_path=cfg.phylo470_bw_path,
            gtf_path=cfg.gtf_path,
            seq_len=effective_seq_len,
            chromosomes=chromosomes,
            core_radius=2,
            region_radius=10,
            cds_enrichment_fraction=effective_sampling_state.cds_enrichment_fraction,
            splice_window_oversample_fraction=effective_sampling_state.splice_window_oversample_fraction,
            encode_track_specs=encode_track_specs,
            clinvar_blocklist_bed_path=cfg.clinvar_blocklist_bed_path,
            curriculum_phase=cast(Any, effective_sampling_state.curriculum_phase),
        )

    collator = MultiTaskCollator(
        mask_prob=cfg.mask_prob,
        mean_span_len=cfg.mean_span_len,
        conservation_mix=effective_sampling_state.conservation_mix,
        hard_position_mix=effective_sampling_state.hard_position_mix,
        counterfactual_fraction=effective_sampling_state.counterfactual_fraction,
        allele_fraction=effective_sampling_state.allele_fraction,
        region_loss_ema=region_loss_ema.values if region_loss_ema is not None else None,
        encode_track_names=[spec.name for spec in encode_track_specs],
    )
    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": cfg.batch_size,
        "shuffle": False,
        "num_workers": cfg.num_workers,
        "pin_memory": (device.type == "cuda"),
        "collate_fn": collator,
        "persistent_workers": (cfg.num_workers > 0),
        "worker_init_fn": _worker_init_fn,
    }
    if cfg.num_workers > 0:
        loader_kwargs["prefetch_factor"] = cfg.prefetch_factor
        if device.type == "cuda" and sys.platform.startswith("linux"):
            # Avoid CUDA/NCCL poison-fork hangs when workers spin up after distributed init.
            loader_kwargs["multiprocessing_context"] = "spawn"
    return DataLoader(**loader_kwargs)


def build_dataloaders(
    cfg: TrainConfig,
    device: torch.device,
    *,
    train_chromosomes: list[str] | None = None,
    val_chromosomes: list[str] | None = None,
    seq_len: int | None = None,
    sampling_state: SamplingScheduleState | None = None,
    region_loss_ema: RegionLossEMA | None = None,
) -> tuple[DataLoader, DataLoader, list[str], list[str]]:
    resolved_train_chromosomes = train_chromosomes
    resolved_val_chromosomes = val_chromosomes
    if resolved_train_chromosomes is None or resolved_val_chromosomes is None:
        resolved_train_chromosomes, resolved_val_chromosomes = resolve_chromosome_splits(cfg)
    train_loader = build_dataloader(
        cfg,
        device,
        resolved_train_chromosomes,
        dataset_split=cfg.abraom_train_split,
        seq_len=seq_len,
        sampling_state=sampling_state,
        region_loss_ema=region_loss_ema,
    )
    val_loader = build_dataloader(
        cfg,
        device,
        resolved_val_chromosomes,
        dataset_split=cfg.abraom_val_split,
        seq_len=seq_len,
        sampling_state=sampling_state,
        region_loss_ema=region_loss_ema,
    )
    return train_loader, val_loader, resolved_train_chromosomes, resolved_val_chromosomes


def shutdown_dataloader(loader: DataLoader | None) -> None:
    if loader is None:
        return
    iterator = getattr(loader, "_iterator", None)
    shutdown_workers = getattr(iterator, "_shutdown_workers", None)
    if callable(shutdown_workers):
        with contextlib.suppress(Exception):
            shutdown_workers()
        with contextlib.suppress(Exception):
            loader._iterator = None  # type: ignore[attr-defined]


def build_model(cfg: TrainConfig, device: torch.device) -> torch.nn.Module:
    resolved_cfg = resolve_train_config(cfg)
    return build_registered_model(resolved_cfg.model, resolved_cfg.model_config).to(device)


def _should_convert_lumina_linear_for_fp8(name: str, _module: torch.nn.Module) -> bool:
    parts = name.split(".")
    if "token_emb" in parts:
        return False
    if name == "mlm_head" or parts[-1] == "mlm_head":
        return False
    return not any(part in {"fwd_mixer", "bwd_mixer", "mixer"} for part in parts)


def prepare_model_for_precision(model: torch.nn.Module, precision: PrecisionPolicy) -> PrecisionPolicy:
    return apply_fp8_linear_replacements(model, precision, _should_convert_lumina_linear_for_fp8)


def _run_memory_preflight_phase(
    cfg: TrainConfig,
    *,
    spec: MemoryPreflightPhaseSpec,
    device: torch.device,
    precision: PrecisionPolicy,
    ctx: DistributedContext,
    train_chromosomes: list[str],
) -> dict[str, Any]:
    loader: DataLoader | None = None
    model: torch.nn.Module | None = None
    optimizer: torch.optim.Optimizer | None = None
    batches: list[dict[str, Any]] = []
    phase_precision = precision

    try:
        seed_everything(cfg.seed, rank=ctx.rank)
        _clear_cuda_memory(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        loader = build_dataloader(
            cfg,
            device,
            train_chromosomes,
            seq_len=spec.seq_len,
            sampling_state=resolve_sampling_schedule_state(cfg, spec.phase_start_step),
        )
        model = build_model(cfg, device)
        phase_precision = prepare_model_for_precision(model, precision)
        components = build_training_components(cfg, model, device)
        optimizer = components.optimizer

        set_optimizer_lr(optimizer, get_lr(spec.phase_start_step, cfg))

        iterator = iter(loader)
        for _ in range(cfg.grad_accum_steps):
            batches.append(next(iterator))

        step_result = train_step(
            model,
            batches,
            optimizer,
            device,
            cfg,
            phase_precision,
            components.log_sigmas,
            components.gradnorm_balancer,
            step=spec.phase_start_step,
        )
        if components.gradnorm_balancer is not None:
            components.gradnorm_balancer.update(step_result.stats)

        allocated = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        reserved = int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
        return {
            "phase": spec.phase,
            "seq_len": spec.seq_len,
            "phase_start_step": spec.phase_start_step,
            "auxiliary_path": spec.auxiliary_path,
            "status": "passed",
            "rank": ctx.rank,
            "resolved_precision": phase_precision.resolved,
            "max_memory_allocated_bytes": allocated,
            "max_memory_reserved_bytes": reserved,
        }
    except Exception as exc:
        return {
            "phase": spec.phase,
            "seq_len": spec.seq_len,
            "phase_start_step": spec.phase_start_step,
            "auxiliary_path": spec.auxiliary_path,
            "status": "failed",
            "rank": ctx.rank,
            "resolved_precision": phase_precision.resolved,
            "error_type": (
                "oom"
                if _is_cuda_oom_error(exc)
                else "checkpoint_runtime_incompatible"
                if _is_checkpoint_runtime_incompatible_error(exc)
                else "error"
            ),
            "error": str(exc),
            "max_memory_allocated_bytes": None,
            "max_memory_reserved_bytes": None,
        }
    finally:
        batches.clear()
        shutdown_dataloader(loader)
        del optimizer, model, loader, batches
        gc.collect()
        _clear_cuda_memory(device)


def _print_memory_preflight_phase(phase_result: dict[str, Any]) -> None:
    phase = int(phase_result["phase"])
    seq_len = int(phase_result["seq_len"])
    auxiliary_path = str(phase_result.get("auxiliary_path", "standard"))
    status = str(phase_result["status"])
    if status == "passed":
        print(
            "memory_preflight "
            f"phase={phase} seq_len={seq_len} status=pass "
            f"auxiliary_path={auxiliary_path} "
            f"peak_allocated={_format_gib(cast(int | None, phase_result.get('max_memory_allocated_bytes')))} "
            f"peak_reserved={_format_gib(cast(int | None, phase_result.get('max_memory_reserved_bytes')))}"
        )
        return

    print(
        "memory_preflight "
        f"phase={phase} seq_len={seq_len} status=fail "
        f"auxiliary_path={auxiliary_path} "
        f"failed_rank={phase_result.get('failed_rank', phase_result.get('rank'))} "
        f"error_type={phase_result.get('error_type', 'error')} "
        f"error={phase_result.get('error')}"
    )


def _raise_memory_preflight_failure(
    cfg: TrainConfig,
    *,
    phase_result: dict[str, Any],
    report: dict[str, Any],
) -> None:
    phase = int(phase_result["phase"])
    seq_len = int(phase_result["seq_len"])
    auxiliary_path = str(phase_result.get("auxiliary_path", "standard"))
    rank = phase_result.get("failed_rank", phase_result.get("rank"))
    resolved_precision = phase_result.get("resolved_precision")
    error_type = phase_result.get("error_type", "error")
    error = phase_result.get("error")
    if error_type == "oom":
        raise MemoryPreflightError(
            "memory preflight failed: "
            f"phase={phase} seq_len={seq_len} batch_size={cfg.batch_size} "
            f"auxiliary_path={auxiliary_path} "
            f"grad_accum_steps={cfg.grad_accum_steps} resolved_precision={resolved_precision} "
            f"rank={rank} error={error}",
            report,
        )
    raise MemoryPreflightError(
        "memory preflight hit a non-OOM error: "
        f"phase={phase} seq_len={seq_len} batch_size={cfg.batch_size} "
        f"auxiliary_path={auxiliary_path} "
        f"grad_accum_steps={cfg.grad_accum_steps} resolved_precision={resolved_precision} "
        f"rank={rank} error={error}",
        report,
    )


def run_memory_preflight(
    cfg: TrainConfig,
    *,
    device: torch.device,
    precision: PrecisionPolicy,
    ctx: DistributedContext,
    train_chromosomes: list[str],
) -> dict[str, Any]:
    if device.type != "cuda":
        reason = f"non_cuda_device:{device.type}"
        if ctx.is_main:
            print(f"memory_preflight=skipped reason={reason}")
        return build_memory_preflight_report(enabled=True, status="skipped", device=device, reason=reason)

    phase_reports: list[dict[str, Any]] = []
    for spec in resolve_memory_preflight_phases(cfg):
        local_result = _run_memory_preflight_phase(
            cfg,
            spec=spec,
            device=device,
            precision=precision,
            ctx=ctx,
            train_chromosomes=train_chromosomes,
        )
        gathered_results = _gather_memory_preflight_phase_results(local_result, ctx)
        phase_result = _aggregate_memory_preflight_phase_results(spec, gathered_results)
        phase_reports.append(phase_result)

        if phase_result["status"] != "passed":
            if ctx.is_main:
                _print_memory_preflight_phase(phase_result)
            report = build_memory_preflight_report(
                enabled=True,
                status="failed",
                device=device,
                phases=phase_reports,
            )
            _raise_memory_preflight_failure(cfg, phase_result=phase_result, report=report)
        if ctx.is_main:
            _print_memory_preflight_phase(phase_result)

    return build_memory_preflight_report(enabled=True, status="passed", device=device, phases=phase_reports)


def resolved_config_dict(
    cfg: TrainConfig,
    *,
    precision: PrecisionPolicy,
    parameter_count: int,
) -> dict[str, Any]:
    resolved_cfg = resolve_train_config(cfg)
    data = asdict(resolved_cfg)
    train_chromosomes, val_chromosomes = resolve_chromosome_splits(cfg)
    data["chromosome_split_mode"] = cfg.chromosome_split_mode
    data["chromosomes"] = list(train_chromosomes)
    data["train_chromosomes"] = list(train_chromosomes)
    data["val_chromosomes"] = list(val_chromosomes)
    data["trainable_params"] = parameter_count
    data.update(precision_metadata(precision))
    data["best_metric_name"] = "loss"
    return data


def ensure_output_layout(output_dir: Path) -> None:
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def write_config_json(output_dir: Path, config: dict[str, Any]) -> None:
    write_json(output_dir / "train_config.json", config)


def history_path(output_dir: Path) -> Path:
    return output_dir / "metrics_history.jsonl"


def run_summary_path(output_dir: Path) -> Path:
    return output_dir / "run_summary.json"


def best_checkpoint_path(output_dir: Path) -> Path:
    return output_dir / "best_checkpoint.pt"


def reset_run_artifacts(output_dir: Path) -> None:
    history_path(output_dir).write_text("")


def append_history_row(output_dir: Path, row: dict[str, Any]) -> None:
    with history_path(output_dir).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def checkpoint_path(output_dir: Path, step: int) -> Path:
    return output_dir / "checkpoints" / f"step_{step:0{CHECKPOINT_DIGITS}d}.pt"


def final_checkpoint_path(output_dir: Path) -> Path:
    return output_dir / "final_checkpoint.pt"


def checkpoint_step(path: Path) -> int:
    try:
        return int(path.stem.split("_", maxsplit=1)[1])
    except (IndexError, ValueError):
        return -1


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    step: int,
    config: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model": _unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "step": step,
        "config": config,
        "metadata": metadata or {},
    }
    torch.save(checkpoint, path)


def torch_load_checkpoint(path: str, map_location: str | torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> tuple[int, dict[str, Any]]:
    checkpoint = torch_load_checkpoint(path, map_location=map_location)
    _unwrap_model(model).load_state_dict(checkpoint["model"])
    optimizer_state = checkpoint.get("optimizer")
    if optimizer is not None and optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    metadata = checkpoint.get("metadata")
    return int(checkpoint.get("step", 0)), metadata if isinstance(metadata, dict) else {}


def init_model_weights_from_checkpoint(
    path: str,
    model: torch.nn.Module,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint = torch_load_checkpoint(path, map_location=map_location)
    state_dict = checkpoint.get("model")
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"Checkpoint {path!r} is missing a usable 'model' state_dict.")
    _unwrap_model(model).load_state_dict(state_dict)
    metadata = checkpoint.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def cleanup_old_checkpoints(output_dir: Path, keep: int = 3) -> None:
    checkpoint_dir = output_dir / "checkpoints"
    if not checkpoint_dir.exists():
        return

    checkpoints = sorted(checkpoint_dir.glob("step_*.pt"), key=checkpoint_step)
    if len(checkpoints) <= keep:
        return

    for path in checkpoints[:-keep]:
        path.unlink(missing_ok=True)


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    def move_value(value: Any) -> Any:
        if torch.is_tensor(value):
            return value.to(device, non_blocking=(device.type == "cuda"))
        if isinstance(value, dict):
            return {nested_key: move_value(nested_value) for nested_key, nested_value in value.items()}
        return value

    return {
        key: move_value(value)
        for key, value in batch.items()
    }


def batch_composition_stats(batch: dict[str, Any]) -> dict[str, float]:
    return {key: float(batch[key]) for key in COMPOSITION_STAT_KEYS}


def batch_token_count(batch: dict[str, Any]) -> int:
    attention_mask = batch.get("attention_mask")
    if torch.is_tensor(attention_mask):
        return int(attention_mask.sum().item())
    input_ids = batch.get("input_ids")
    if torch.is_tensor(input_ids):
        return int(input_ids.numel())
    raise KeyError("Batch is missing both attention_mask and input_ids for token counting.")


def resolve_auxiliary_compute_state(
    cfg: TrainConfig,
    step: int,
    *,
    grad_enabled: bool,
    has_allele: bool,
    has_counterfactual: bool,
) -> AuxiliaryComputeState:
    step_index = max(int(step), 1)
    alternating = _uses_alternating_auxiliary_schedule(cfg) and grad_enabled and has_allele and has_counterfactual
    if alternating:
        allele_active = step_index % 2 == 1
        counterfactual_active = not allele_active
        allele_step_index = (step_index + 1) // 2
    else:
        allele_active = has_allele
        counterfactual_active = has_counterfactual
        allele_step_index = step_index
    allele_rank_step = allele_active and allele_step_index % cfg.allele_rank_every_n_steps == 0
    return AuxiliaryComputeState(
        allele=allele_active,
        counterfactual=counterfactual_active,
        allele_rank_step=allele_rank_step,
    )


def _selected_allele_rows(batch: dict[str, Any], max_rows: int) -> torch.Tensor:
    valid_mask = batch["allele_valid_mask"].to(dtype=torch.bool)
    active_rows = torch.nonzero(valid_mask.any(dim=1), as_tuple=False).flatten()
    if active_rows.numel() == 0:
        return active_rows

    severity = torch.where(
        valid_mask,
        batch["allele_severity_targets"].float().clamp_min(0.0),
        torch.zeros_like(batch["allele_severity_targets"], dtype=torch.float32),
    )
    row_scores = severity.max(dim=1).values
    order = torch.argsort(row_scores[active_rows], descending=True)
    return active_rows[order[:max_rows]]


def _sample_one_valid_alt_per_row(
    valid_mask: torch.Tensor,
    severity_targets: torch.Tensor,
) -> torch.Tensor:
    slots: list[int] = []
    for row_index in range(valid_mask.shape[0]):
        valid = valid_mask[row_index]
        weights = torch.where(
            valid,
            severity_targets[row_index].float().clamp_min(0.0) + 0.05,
            torch.zeros_like(severity_targets[row_index], dtype=torch.float32),
        )
        slots.append(int(torch.multinomial(weights / weights.sum().clamp_min(1e-8), num_samples=1).item()))
    return torch.tensor(slots, dtype=torch.long, device=valid_mask.device)


def _crop_allele_scoring_sequences(
    *,
    ref_input_ids: torch.Tensor,
    alt_input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    allele_position: torch.Tensor,
    window: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    seq_len = ref_input_ids.shape[1]
    score_seq_len = min(seq_len, int(window))
    if score_seq_len == seq_len:
        return ref_input_ids, alt_input_ids, allele_position, attention_mask

    starts = (allele_position - score_seq_len // 2).clamp(0, seq_len - score_seq_len)
    offsets = torch.arange(score_seq_len, device=ref_input_ids.device).unsqueeze(0)
    gather_idx = starts.unsqueeze(1) + offsets
    ref_crop = ref_input_ids.gather(dim=1, index=gather_idx)
    attention_crop = attention_mask.gather(dim=1, index=gather_idx)
    alt_gather_idx = gather_idx.unsqueeze(1).expand(-1, alt_input_ids.shape[1], -1)
    alt_crop = alt_input_ids.gather(dim=2, index=alt_gather_idx)
    return ref_crop, alt_crop, allele_position - starts, attention_crop


def prepare_allele_scoring_batch(
    batch: dict[str, Any],
    cfg: TrainConfig,
    *,
    rank_step: bool,
) -> PreparedAlleleScoringBatch | None:
    row_indices = _selected_allele_rows(batch, cfg.allele_max_rows_per_batch)
    if row_indices.numel() == 0:
        return None

    ref_input_ids = batch["allele_ref_input_ids"].index_select(0, row_indices)
    allele_position = batch["allele_position"].index_select(0, row_indices)
    allele_alt_ids = batch["allele_alt_ids"].index_select(0, row_indices)
    allele_effect_labels = batch["allele_effect_labels"].index_select(0, row_indices)
    allele_severity_targets = batch["allele_severity_targets"].index_select(0, row_indices)
    allele_valid_mask = batch["allele_valid_mask"].index_select(0, row_indices)
    allele_locus_ids = batch["allele_locus_ids"].index_select(0, row_indices)
    alt_input_ids = batch["allele_alt_input_ids"].index_select(0, row_indices)

    if not rank_step:
        alt_slots = _sample_one_valid_alt_per_row(allele_valid_mask, allele_severity_targets)
        row_range = torch.arange(row_indices.numel(), device=row_indices.device)
        alt_input_ids = alt_input_ids[row_range, alt_slots].unsqueeze(1)
        allele_alt_ids = allele_alt_ids.gather(dim=1, index=alt_slots.unsqueeze(1))
        allele_effect_labels = allele_effect_labels.gather(dim=1, index=alt_slots.unsqueeze(1))
        allele_severity_targets = allele_severity_targets.gather(dim=1, index=alt_slots.unsqueeze(1))
        allele_valid_mask = allele_valid_mask.gather(dim=1, index=alt_slots.unsqueeze(1))
        allele_locus_ids = allele_locus_ids.gather(dim=1, index=alt_slots.unsqueeze(1))

    attention_mask = (ref_input_ids != 0).long()
    ref_input_ids, alt_input_ids, allele_position, attention_mask = _crop_allele_scoring_sequences(
        ref_input_ids=ref_input_ids,
        alt_input_ids=alt_input_ids,
        attention_mask=attention_mask,
        allele_position=allele_position,
        window=cfg.allele_score_window,
    )
    alt_attention_mask = attention_mask.unsqueeze(1).expand_as(alt_input_ids)

    loss_batch = dict(batch)
    loss_batch.update(
        {
            "allele_ref_input_ids": ref_input_ids,
            "allele_alt_input_ids": alt_input_ids,
            "allele_position": allele_position,
            "allele_alt_ids": allele_alt_ids,
            "allele_effect_labels": allele_effect_labels,
            "allele_severity_targets": allele_severity_targets,
            "allele_valid_mask": allele_valid_mask,
            "allele_locus_ids": allele_locus_ids,
        }
    )
    return PreparedAlleleScoringBatch(
        loss_batch=loss_batch,
        scorer_kwargs={
            "ref_input_ids": ref_input_ids,
            "alt_input_ids": alt_input_ids,
            "allele_position": allele_position,
            "allele_alt_ids": allele_alt_ids,
            "attention_mask": attention_mask,
            "alt_attention_mask": alt_attention_mask,
        },
        active_rows=int(row_indices.numel()),
        scored_alts=int(row_indices.numel() * alt_input_ids.shape[1]),
        score_seq_len=int(ref_input_ids.shape[1]),
    )


def format_chromosome_frequency(chrom_counts: Counter[str], max_items: int = 6) -> str:
    total = sum(chrom_counts.values())
    if total == 0:
        return "none"

    ranked = sorted(chrom_counts.items(), key=lambda item: (-item[1], item[0]))
    head = ranked[:max_items]
    parts = [f"{chrom}:{count / total:.1%}" for chrom, count in head]
    if len(ranked) > max_items:
        remainder = total - sum(count for _chrom, count in head)
        parts.append(f"other:{remainder / total:.1%}")
    return ",".join(parts)


def _execute_model_forward_passes(
    model: torch.nn.Module,
    batch: dict[str, Any],
    *,
    want_rc: bool,
    counterfactual_active: bool,
    allele_forward_kwargs: dict[str, torch.Tensor],
    model_forward_kwargs: dict[str, Any] | None = None,
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, torch.Tensor] | None,
    dict[str, torch.Tensor] | None,
]:
    """Run auxiliary and main forward passes in DDP-safe order.

    With ``find_unused_parameters=True``, DDP calls
    ``reducer.prepare_for_backward(_find_tensors(output))`` after every
    ``model(...)`` invocation and the LAST call wins. The auxiliary forwards
    return only encoder-derived tensors (``sequence_embedding``,
    ``hidden_states``) that do not trace back through the token heads, so
    running them last would mark every token head as unused. DDP's synthetic
    zero-grad sweep would then collide with the real autograd hook fired by
    the loss path, producing ``mark_variable_ready called twice`` errors on
    parameters such as ``region_head.3.weight``.

    The main forward (with ``return_token_heads=True`` by default) MUST be
    invoked last so DDP's used-parameter trace covers the full token-head
    graph. Order: counterfactual → reverse-complement → main.
    """
    alt_outputs: dict[str, torch.Tensor] | None = None
    if counterfactual_active:
        alt_outputs = model(
            input_ids=batch["alt_input_ids"],
            attention_mask=batch["alt_attention_mask"],
            return_token_heads=False,
            return_sequence_embedding=False,
            return_hidden=True,
        )

    rc_outputs: dict[str, torch.Tensor] | None = None
    if want_rc:
        rc_outputs = model(
            input_ids=batch["rc_input_ids"],
            attention_mask=batch["rc_attention_mask"],
            return_token_heads=False,
            return_sequence_embedding=True,
        )

    model_forward_kwargs = model_forward_kwargs or {}
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        return_sequence_embedding=want_rc,
        **model_forward_kwargs,
        **allele_forward_kwargs,
    )
    return outputs, rc_outputs, alt_outputs


def run_model_step(
    model: torch.nn.Module,
    batch: dict[str, Any],
    device: torch.device,
    cfg: TrainConfig,
    precision: PrecisionPolicy,
    log_sigmas: dict[str, torch.Tensor] | None = None,
    gradnorm_balancer: GradNormBalancer | None = None,
    step: int = 0,
) -> tuple[torch.Tensor, dict[str, float], MetricAccumulator]:
    batch = move_batch_to_device(batch, device)
    batch_for_loss = batch
    aux_scale = aux_loss_warmup_factor(step, cfg.aux_loss_warmup_steps)
    with autocast_context(precision):
        want_rc = cfg.w_rc != 0.0
        has_counterfactual = cfg.w_counterfactual != 0.0 and "alt_input_ids" in batch
        has_allele = cfg.w_allele != 0.0 and "allele_alt_input_ids" in batch
        aux_state = resolve_auxiliary_compute_state(
            cfg,
            step,
            grad_enabled=torch.is_grad_enabled(),
            has_allele=has_allele,
            has_counterfactual=has_counterfactual,
        )
        fixed_counterfactual_weight = (
            cfg.w_counterfactual
            if aux_state.counterfactual and cfg.counterfactual_weighting == "fixed"
            else 0.0
        )
        allele_stats = {
            "allele_active_rows": 0.0,
            "allele_scored_alts": 0.0,
            "allele_score_seq_len": 0.0,
            "allele_rank_step": 1.0 if aux_state.allele_rank_step else 0.0,
            "allele_compute_active": 0.0,
            "counterfactual_compute_active": 0.0,
        }
        prepared_allele: PreparedAlleleScoringBatch | None = None
        score_alleles = None
        if aux_state.allele:
            scorer_model = _unwrap_model(model)
            score_alleles = getattr(scorer_model, "score_alleles_from_ids", None)
            if callable(score_alleles):
                prepared_allele = prepare_allele_scoring_batch(
                    batch,
                    cfg,
                    rank_step=aux_state.allele_rank_step,
                )
                if prepared_allele is not None:
                    batch_for_loss = prepared_allele.loss_batch
                    allele_stats.update(
                        {
                            "allele_active_rows": float(prepared_allele.active_rows),
                            "allele_scored_alts": float(prepared_allele.scored_alts),
                            "allele_score_seq_len": float(prepared_allele.score_seq_len),
                            "allele_compute_active": 1.0,
                        }
                    )

        allele_forward_kwargs: dict[str, torch.Tensor] = {}
        if cfg.model == "beat-v8" and prepared_allele is not None:
            allele_forward_kwargs = {
                "allele_ref_input_ids": prepared_allele.scorer_kwargs["ref_input_ids"],
                "allele_alt_input_ids": prepared_allele.scorer_kwargs["alt_input_ids"],
                "allele_position": prepared_allele.scorer_kwargs["allele_position"],
                "allele_alt_ids": prepared_allele.scorer_kwargs["allele_alt_ids"],
                "allele_attention_mask": prepared_allele.scorer_kwargs["attention_mask"],
                "allele_alt_attention_mask": prepared_allele.scorer_kwargs["alt_attention_mask"],
            }
        model_forward_kwargs: dict[str, Any] = {}
        if cfg.model == "beat-v9":
            model_forward_kwargs["return_counterfactual_repr"] = False

        # See _execute_model_forward_passes for the DDP correctness invariant
        # (main forward must be the LAST model call). RC loss is a global
        # consistency term added with a fixed weight outside the uncertainty
        # / gradnorm system.
        outputs, rc_outputs, alt_outputs = _execute_model_forward_passes(
            model,
            batch,
            want_rc=want_rc,
            counterfactual_active=aux_state.counterfactual,
            allele_forward_kwargs=allele_forward_kwargs,
            model_forward_kwargs=model_forward_kwargs,
        )
        if aux_state.counterfactual:
            allele_stats["counterfactual_compute_active"] = 1.0
        if prepared_allele is not None and "allele_effect_logits" not in outputs and callable(score_alleles):
            allele_outputs = cast(
                dict[str, torch.Tensor],
                score_alleles(**prepared_allele.scorer_kwargs),
            )
            outputs.update(allele_outputs)

        if log_sigmas is not None:
            loss, stats = compute_uncertainty_weighted_loss(
                outputs=outputs,
                batch=batch_for_loss,
                log_sigmas=log_sigmas,
                alt_outputs=alt_outputs,
                phylo_weighted_mlm=cfg.phylo_weighted_mlm,
                phylo_mlm_boost=cfg.phylo_mlm_boost,
                aux_scale=aux_scale,
                lambda_mutation_effect_rank=cfg.lambda_mutation_effect_rank,
                counterfactual_radius=cfg.counterfactual_local_radius,
                counterfactual_far_radius=cfg.counterfactual_far_radius,
                counterfactual_local_similarity_target=cfg.counterfactual_local_similarity_target,
                fixed_counterfactual_weight=fixed_counterfactual_weight,
                lambda_allele_rank=cfg.lambda_allele_rank,
                lambda_allele_severity=cfg.lambda_allele_severity,
                lambda_allele_swap=cfg.lambda_allele_swap,
                lambda_allele_far=cfg.lambda_allele_far,
                splice_class_weight_cap=cfg.splice_class_weight_cap,
                region_class_weight_cap=cfg.region_class_weight_cap,
                aa_class_weight_cap=cfg.aa_class_weight_cap,
            )
            # Add RC loss outside the uncertainty system (fixed weight).
            if want_rc and rc_outputs is not None:
                rc_loss = rc_embedding_loss(
                    outputs["sequence_embedding"],
                    rc_outputs["sequence_embedding"],
                    loss_type=cfg.rc_loss_type,
                )
                loss = loss + cfg.w_rc * aux_scale * rc_loss
                stats["loss"] = loss.detach()
                stats["loss_rc"] = rc_loss.detach()
        else:
            # Use GradNorm-adjusted weights when the balancer is active.
            if gradnorm_balancer is not None:
                gn_weights = gradnorm_balancer.get_weights()
                loss, stats = compute_multitask_loss(
                    outputs=outputs,
                    batch=batch_for_loss,
                    rc_outputs=rc_outputs,
                    w_mlm=gn_weights.get("mlm", cfg.w_mlm),
                    w_phylo100=gn_weights.get("phylo100", cfg.w_phylo100) * aux_scale,
                    w_phylo470=gn_weights.get("phylo470", cfg.w_phylo470) * aux_scale,
                    w_structure=gn_weights.get("structure", cfg.w_structure) * aux_scale,
                    w_rc=cfg.w_rc * aux_scale,
                    w_region=gn_weights.get("region", cfg.w_region) * aux_scale,
                    w_aa=gn_weights.get("aa", cfg.w_aa) * aux_scale,
                    w_codon_phylo=gn_weights.get("codon_phylo", cfg.w_codon_phylo) * aux_scale,
                    w_mutation_effect=gn_weights.get("mutation_effect", cfg.w_mutation_effect) * aux_scale,
                    w_counterfactual=(
                        gn_weights.get("counterfactual", cfg.w_counterfactual) * aux_scale
                        if aux_state.counterfactual
                        else 0.0
                    ),
                    w_allele=gn_weights.get("allele", cfg.w_allele) * aux_scale if aux_state.allele else 0.0,
                    w_codon=gn_weights.get("codon", cfg.w_codon) * aux_scale,
                    w_encode=gn_weights.get("encode", cfg.w_encode) * aux_scale,
                    w_conservation_bin=cfg.w_conservation_bin * aux_scale,
                    w_splice_distance=cfg.w_splice_distance * aux_scale,
                    w_codon_pos=cfg.w_codon_pos * aux_scale,
                    w_exon_phase=cfg.w_exon_phase * aux_scale,
                    w_counterfactual_snv=cfg.w_counterfactual_snv * aux_scale,
                    w_counterfactual_severity=cfg.w_counterfactual_severity * aux_scale,
                    rc_loss_type=cfg.rc_loss_type,
                    phylo_weighted_mlm=cfg.phylo_weighted_mlm,
                    phylo_mlm_boost=cfg.phylo_mlm_boost,
                    alt_outputs=alt_outputs,
                    lambda_mutation_effect_rank=cfg.lambda_mutation_effect_rank,
                    counterfactual_radius=cfg.counterfactual_local_radius,
                    counterfactual_far_radius=cfg.counterfactual_far_radius,
                    counterfactual_local_similarity_target=cfg.counterfactual_local_similarity_target,
                    lambda_allele_rank=cfg.lambda_allele_rank,
                    lambda_allele_severity=cfg.lambda_allele_severity,
                    lambda_allele_swap=cfg.lambda_allele_swap,
                    lambda_allele_far=cfg.lambda_allele_far,
                    splice_class_weight_cap=cfg.splice_class_weight_cap,
                    region_class_weight_cap=cfg.region_class_weight_cap,
                    aa_class_weight_cap=cfg.aa_class_weight_cap,
                )
            else:
                loss, stats = compute_multitask_loss(
                    outputs=outputs,
                    batch=batch_for_loss,
                    rc_outputs=rc_outputs,
                    w_mlm=cfg.w_mlm,
                    w_phylo100=cfg.w_phylo100 * aux_scale,
                    w_phylo470=cfg.w_phylo470 * aux_scale,
                    w_structure=cfg.w_structure * aux_scale,
                    w_rc=cfg.w_rc * aux_scale,
                    w_region=cfg.w_region * aux_scale,
                    w_aa=cfg.w_aa * aux_scale,
                    w_codon_phylo=cfg.w_codon_phylo * aux_scale,
                    w_mutation_effect=cfg.w_mutation_effect * aux_scale,
                    w_counterfactual=cfg.w_counterfactual * aux_scale if aux_state.counterfactual else 0.0,
                    w_allele=cfg.w_allele * aux_scale if aux_state.allele else 0.0,
                    w_codon=cfg.w_codon * aux_scale,
                    w_encode=cfg.w_encode * aux_scale,
                    w_conservation_bin=cfg.w_conservation_bin * aux_scale,
                    w_splice_distance=cfg.w_splice_distance * aux_scale,
                    w_codon_pos=cfg.w_codon_pos * aux_scale,
                    w_exon_phase=cfg.w_exon_phase * aux_scale,
                    w_counterfactual_snv=cfg.w_counterfactual_snv * aux_scale,
                    w_counterfactual_severity=cfg.w_counterfactual_severity * aux_scale,
                    rc_loss_type=cfg.rc_loss_type,
                    phylo_weighted_mlm=cfg.phylo_weighted_mlm,
                    phylo_mlm_boost=cfg.phylo_mlm_boost,
                    alt_outputs=alt_outputs,
                    lambda_mutation_effect_rank=cfg.lambda_mutation_effect_rank,
                    counterfactual_radius=cfg.counterfactual_local_radius,
                    counterfactual_far_radius=cfg.counterfactual_far_radius,
                    counterfactual_local_similarity_target=cfg.counterfactual_local_similarity_target,
                    lambda_allele_rank=cfg.lambda_allele_rank,
                    lambda_allele_severity=cfg.lambda_allele_severity,
                    lambda_allele_swap=cfg.lambda_allele_swap,
                    lambda_allele_far=cfg.lambda_allele_far,
                    splice_class_weight_cap=cfg.splice_class_weight_cap,
                    region_class_weight_cap=cfg.region_class_weight_cap,
                    aa_class_weight_cap=cfg.aa_class_weight_cap,
                )

    for key, value in allele_stats.items():
        stats[key] = outputs["mlm_logits"].new_tensor(value)

    metric_update = MetricAccumulator()
    metric_update.update_from_batch(outputs, batch_for_loss)

    scalar_stats = {key: float(value) for key, value in stats.items()}
    # Ensure uncertainty and region stat keys are always present for consistent logging.
    for key in (
        UNCERTAINTY_STAT_KEYS
        + REGION_UNCERTAINTY_STAT_KEYS
        + AA_UNCERTAINTY_STAT_KEYS
        + MUTATION_EFFECT_UNCERTAINTY_STAT_KEYS
        + ALLELE_UNCERTAINTY_STAT_KEYS
        + COUNTERFACTUAL_STAT_KEYS
        + AUXILIARY_COMPUTE_STAT_KEYS
    ):
        scalar_stats.setdefault(key, 0.0)
    scalar_stats.update(batch_composition_stats(batch))
    if "cf_active_fraction" in batch:
        scalar_stats["cf_active_fraction"] = float(batch["cf_active_fraction"])
    return loss, scalar_stats, metric_update


def _no_sync_context(model: torch.nn.Module, ctx: DistributedContext | None) -> Any:
    """Return a DDP no_sync context manager, or a no-op if not distributed."""
    if ctx is not None and ctx.distributed and isinstance(model, DDP):
        return model.no_sync()
    return contextlib.nullcontext()


def train_step(
    model: torch.nn.Module,
    batches: list[dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg: TrainConfig,
    precision: PrecisionPolicy,
    log_sigmas: dict[str, torch.Tensor] | None = None,
    gradnorm_balancer: GradNormBalancer | None = None,
    ctx: DistributedContext | None = None,
    step: int = 0,
) -> TrainStepResult:
    model.train()
    optimizer.zero_grad(set_to_none=True)

    accum_steps = len(batches)
    all_stats: list[dict[str, float]] = []
    merged_metrics = MetricAccumulator()

    for i, batch in enumerate(batches):
        is_last = i == accum_steps - 1
        sync_ctx = contextlib.nullcontext() if is_last else _no_sync_context(model, ctx)
        with sync_ctx:
            loss, stats, metric_update = run_model_step(
                model, batch, device, cfg, precision, log_sigmas, gradnorm_balancer, step=step,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss detected: {float(loss)}")
            (loss / accum_steps).backward()
        all_stats.append(stats)
        merged_metrics.merge(metric_update)

    if ctx is not None and ctx.distributed and log_sigmas is not None:
        _allreduce_log_sigma_grads(log_sigmas, ctx.world_size)

    all_params: list[torch.Tensor] = list(model.parameters())
    if log_sigmas is not None:
        all_params.extend(log_sigmas.values())
    grad_norm = torch.nn.utils.clip_grad_norm_(all_params, cfg.grad_clip)
    optimizer.step()

    avg_stats = {key: sum(s[key] for s in all_stats) / accum_steps for key in all_stats[0]}
    return TrainStepResult(stats=avg_stats, metrics=merged_metrics, grad_norm=float(grad_norm))


@torch.no_grad()
def eval_steps(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: TrainConfig,
    precision: PrecisionPolicy,
    num_steps: int = 20,
    ctx: DistributedContext | None = None,
) -> tuple[dict[str, float], dict[str, float], Counter[str]]:
    model.eval()

    scalar_running: dict[str, SmoothedValue] = {}
    metric_running = MetricAccumulator()
    chrom_counts: Counter[str] = Counter()

    iterator = iter(loader)
    for _ in range(num_steps):
        batch = next(iterator)

        chrom_counts.update(cast(list[str], batch["chroms"]))
        _loss, scalar_stats, metric_update = run_model_step(model, batch, device, cfg, precision)
        for key, value in scalar_stats.items():
            scalar_running.setdefault(key, SmoothedValue()).update(value)
        metric_running.merge(metric_update)

    scalar_summary = {key: value.avg for key, value in scalar_running.items()}

    if ctx is not None and ctx.distributed:
        # Average loss/weight/composition stats across ranks
        avg_keys = tuple(key for key in scalar_summary if key not in COUNT_STAT_KEYS)
        scalar_summary = _allreduce_scalar_stats(scalar_summary, avg_keys, device, ctx.world_size)
        # Sum count stats (these are totals, not averages)
        sum_keys = tuple(key for key in COUNT_STAT_KEYS if key in scalar_summary)
        scalar_summary = _allreduce_sum_stats(scalar_summary, sum_keys, device)
        # Aggregate biological metrics across ranks before computing derived values
        metric_running.all_reduce(device)

    return scalar_summary, metric_running.summary(), chrom_counts


def format_metric_summary(metric_stats: dict[str, float]) -> str:
    parts: list[str] = []
    if "mlm_accuracy" in metric_stats:
        parts.append(f"mlm_acc={metric_stats['mlm_accuracy']:.3f}")
    if "aa_accuracy" in metric_stats:
        parts.append(f"aa_acc={metric_stats['aa_accuracy']:.3f}")
    if "mutation_effect_accuracy" in metric_stats:
        parts.append(f"mut_eff_acc={metric_stats['mutation_effect_accuracy']:.3f}")
    if "mutation_effect_macro_f1" in metric_stats:
        parts.append(f"mut_eff_macro_f1={metric_stats['mutation_effect_macro_f1']:.3f}")
    if "phylo100_pearson" in metric_stats:
        parts.append(f"p100_r={metric_stats['phylo100_pearson']:.3f}")
    if "phylo470_pearson" in metric_stats:
        parts.append(f"p470_r={metric_stats['phylo470_pearson']:.3f}")
    if "codon_phylo_pearson" in metric_stats:
        parts.append(f"codon_p100_r={metric_stats['codon_phylo_pearson']:.3f}")
    if all(
        key in metric_stats
        for key in ("splice_positive_precision", "splice_positive_recall", "splice_positive_f1")
    ):
        parts.append(
            "splice_pos(P/R/F1)="
            f"{metric_stats['splice_positive_precision']:.3f}/"
            f"{metric_stats['splice_positive_recall']:.3f}/"
            f"{metric_stats['splice_positive_f1']:.3f}"
        )
    if "splice_core_f1" in metric_stats:
        parts.append(f"splice_core_f1={metric_stats['splice_core_f1']:.3f}")
    if "splice_region_f1" in metric_stats:
        parts.append(f"splice_region_f1={metric_stats['splice_region_f1']:.3f}")
    if "region_cds_f1" in metric_stats:
        parts.append(
            f"region_cds_f1={metric_stats['region_cds_f1']:.3f}"
            f" region_utr_f1={metric_stats['region_utr_f1']:.3f}"
            f" region_intron_f1={metric_stats['region_intron_f1']:.3f}"
        )
    return " ".join(parts) if parts else "no_metrics"


def build_history_row(
    *,
    phase: str,
    step: int,
    lr: float,
    elapsed_minutes: float,
    seq_len: int,
    length_phase: int,
    scalar_stats: dict[str, float],
    metric_stats: dict[str, float],
    grad_norm: float | None = None,
    step_time: float | None = None,
    tokens_per_sec: float | None = None,
    tokens_seen: int | None = None,
    chrom_freq: str | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "phase": phase,
        "step": step,
        "lr": lr,
        "elapsed_minutes": elapsed_minutes,
        "seq_len": seq_len,
        "length_phase": length_phase,
    }
    if grad_norm is not None:
        row["grad_norm"] = grad_norm
    if step_time is not None:
        row["step_time"] = step_time
    if tokens_per_sec is not None:
        row["tokens_per_sec"] = tokens_per_sec
    if tokens_seen is not None:
        row["tokens_seen"] = tokens_seen
    if chrom_freq is not None:
        row["chrom_freq"] = chrom_freq
    row.update(scalar_stats)
    row.update(metric_stats)
    return row


def init_wandb_run(cfg: TrainConfig, config: dict[str, Any], output_dir: Path) -> Any | None:
    if not cfg.wandb_enabled:
        return None

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "wandb logging is enabled, but the `wandb` package is not installed. Run `uv sync --extra tracking`."
        ) from exc

    from datetime import datetime

    run_name = cfg.wandb_run_name
    if run_name is not None:
        run_name = f"{run_name}-{datetime.now().strftime('%Y%m%d%H%M')}"

    try:
        return wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=run_name,
            tags=cfg.wandb_tags,
            config=config,
            dir=str(output_dir),
        )
    except Exception as exc:
        raise RuntimeError(f"wandb initialization failed: {exc}") from exc


def log_wandb_row(wandb_run: Any | None, row: dict[str, Any], *, commit: bool = True) -> None:
    if wandb_run is None:
        return

    phase = str(row["phase"])
    payload: dict[str, bool | int | float] = {"global_step": int(row["step"])}
    for key, value in row.items():
        if key in {"phase", "step", "chrom_freq"}:
            continue
        if isinstance(value, (bool, int, float)):
            payload[f"{phase}/{key}"] = value

    wandb_run.log(payload, step=int(row["step"]), commit=commit)


def checkpoint_metadata(
    best_state: BestCheckpointState,
    wandb_run: Any | None,
    latest_validation: dict[str, Any] | None,
    tokens_seen: int,
    precision: PrecisionPolicy,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "best_checkpoint": best_state.to_dict(),
        "tokens_seen": tokens_seen,
        "precision": precision_metadata(precision),
    }
    if latest_validation is not None:
        metadata["latest_validation"] = latest_validation
    if wandb_run is not None:
        metadata["wandb_run_id"] = getattr(wandb_run, "id", None)
        metadata["wandb_run_name"] = getattr(wandb_run, "name", None)
    return metadata


def build_run_summary(
    *,
    cfg: TrainConfig,
    device: torch.device,
    precision: PrecisionPolicy,
    parameter_count: int | None,
    train_chromosomes: list[str],
    val_chromosomes: list[str],
    best_state: BestCheckpointState,
    last_step: int,
    latest_validation: dict[str, Any] | None,
    tokens_seen: int,
    wandb_run: Any | None,
    status: str,
    memory_preflight: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary = {
        "status": status,
        "model": cfg.model,
        "model_config": cfg.model_config,
        "chromosome_split_mode": cfg.chromosome_split_mode,
        "seed": cfg.seed,
        "device": device.type,
        "trainable_params": parameter_count,
        "train_chromosomes": train_chromosomes,
        "val_chromosomes": val_chromosomes,
        "last_step": last_step,
        "tokens_seen": tokens_seen,
        "best_checkpoint": best_state.to_dict(),
        "latest_validation": latest_validation,
        "memory_preflight": memory_preflight
        or build_memory_preflight_report(enabled=False, status="disabled", device=device),
    }
    summary.update(precision_metadata(precision))
    if wandb_run is not None:
        summary["wandb_run_id"] = getattr(wandb_run, "id", None)
        summary["wandb_run_name"] = getattr(wandb_run, "name", None)
        summary["wandb_project"] = cfg.wandb_project
    return summary


def train(cfg: TrainConfig) -> None:
    ctx = setup_distributed()
    train_loader: DataLoader | None = None
    val_loader: DataLoader | None = None
    wandb_run: Any | None = None

    try:
        cfg = resolve_train_config(cfg)
        seed_everything(cfg.seed, rank=ctx.rank)

        output_dir = Path(cfg.output_dir)
        if ctx.is_main:
            ensure_output_layout(output_dir)
            if cfg.resume_from is None:
                reset_run_artifacts(output_dir)
        if ctx.distributed:
            dist.barrier()

        device = get_device(ctx)
        configure_float32_precision(cfg.allow_tf32)
        precision = resolve_precision_policy(device, cfg.precision)
        cfg, runtime_model_notes = normalize_runtime_train_config(cfg, device=device, precision=precision)
        stage1_steps, stage2_steps = resolve_length_warmup_stage_steps(cfg)
        train_chromosomes, val_chromosomes = resolve_chromosome_splits(cfg)
        memory_preflight_report = build_memory_preflight_report(enabled=False, status="disabled", device=device)

        if cfg.memory_preflight_enabled:
            try:
                memory_preflight_report = run_memory_preflight(
                    cfg,
                    device=device,
                    precision=precision,
                    ctx=ctx,
                    train_chromosomes=train_chromosomes,
                )
            except MemoryPreflightError as exc:
                memory_preflight_report = exc.report
                if ctx.is_main:
                    write_json(
                        run_summary_path(output_dir),
                        build_run_summary(
                            cfg=cfg,
                            device=device,
                            precision=precision,
                            parameter_count=None,
                            train_chromosomes=train_chromosomes,
                            val_chromosomes=val_chromosomes,
                            best_state=BestCheckpointState(),
                            last_step=0,
                            latest_validation=None,
                            tokens_seen=0,
                            wandb_run=None,
                            status="failed_preflight",
                            memory_preflight=memory_preflight_report,
                        ),
                    )
                raise
            seed_everything(cfg.seed, rank=ctx.rank)

        model = build_model(cfg, device)
        if cfg.init_from_checkpoint is not None:
            _metadata = init_model_weights_from_checkpoint(
                cfg.init_from_checkpoint,
                model=model,
                map_location=device,
            )
            if ctx.is_main:
                checkpoint_step = _metadata.get("step")
                checkpoint_detail = f" checkpoint_step={checkpoint_step}" if checkpoint_step is not None else ""
                print(f"initialized_from_checkpoint={cfg.init_from_checkpoint}{checkpoint_detail}")
        precision = prepare_model_for_precision(model, precision)

        if ctx.distributed:
            # static_graph=True is required for reentrant activation
            # checkpointing. V8's alternating allele/counterfactual path uses
            # non-reentrant checkpointing plus find_unused_parameters=True as
            # defense-in-depth: the per-batch presence of allele/CF rows is
            # data-dependent, so the "used parameter" set legitimately varies
            # across iterations. The main forward is ordered last in
            # run_model_step so DDP's prepare_for_backward sees the full
            # token-head graph regardless of which auxiliaries fired.
            model = DDP(model, device_ids=[ctx.local_rank], **ddp_kwargs_for_train_config(cfg))

        parameter_count = count_parameters(_unwrap_model(model))
        config = resolved_config_dict(cfg, precision=precision, parameter_count=parameter_count)
        if ctx.is_main:
            write_config_json(output_dir, config)

        training_components = build_training_components(cfg, model, device)
        optimizer = training_components.optimizer
        log_sigmas = training_components.log_sigmas
        gradnorm_balancer = training_components.gradnorm_balancer
        effective_balancing = training_components.effective_balancing
        region_loss_ema = RegionLossEMA(decay=cfg.hard_position_ema_decay)

        if ctx.is_main:
            if effective_balancing == "uncertainty":
                print(f"loss_balancing=uncertainty tasks={','.join(training_components.balancing_keys)}")
            elif effective_balancing == "gradnorm":
                print(f"loss_balancing=gradnorm tasks={','.join(training_components.balancing_keys)}")
            else:
                print("loss_balancing=fixed")

        best_state = BestCheckpointState()
        latest_validation: dict[str, Any] | None = None
        start_step = 0
        tokens_seen = 0
        if cfg.resume_from is not None:
            start_step, metadata = load_checkpoint(
                cfg.resume_from,
                model=model,
                optimizer=optimizer,
                map_location=device,
            )
            best_state = BestCheckpointState.from_metadata(metadata.get("best_checkpoint"))
            latest_validation = cast(dict[str, Any] | None, metadata.get("latest_validation"))
            tokens_seen = int(
                metadata.get("tokens_seen", estimate_tokens_seen(cfg, start_step, world_size=ctx.world_size))
            )
            if ctx.is_main:
                print(
                    f"resumed_from={cfg.resume_from} start_step={start_step} "
                    f"best_loss={best_state.loss:.4f}"
                )

        next_step = min(cfg.max_steps, start_step + 1)
        current_length_state = resolve_length_warmup_state(cfg, next_step)
        current_sampling_state = resolve_sampling_schedule_state(cfg, next_step)
        train_loader, val_loader, _resolved_train, _resolved_val = build_dataloaders(
            cfg,
            device,
            train_chromosomes=train_chromosomes,
            val_chromosomes=val_chromosomes,
            seq_len=current_length_state.seq_len,
            sampling_state=current_sampling_state,
            region_loss_ema=region_loss_ema,
        )

        running_scalars = {"grad_norm": SmoothedValue(), "step_time": SmoothedValue()}
        running_metrics = MetricAccumulator()
        train_chrom_counts: Counter[str] = Counter()
        running_token_count = 0
        running_step_time_total = 0.0

        train_iterator = iter(train_loader)
        train_start = time.time()
        last_completed_step = start_step
        logged_first_train_fetch = False
        logged_first_train_step = False
        logged_first_eval_fetch = False
        wandb_run = init_wandb_run(cfg, config, output_dir) if ctx.is_main else None
        if ctx.is_main:
            print(
                f"device={device.type} "
                f"model={cfg.model} "
                f"world_size={ctx.world_size} "
                f"effective_batch_size={cfg.batch_size * ctx.world_size} "
                f"lr_scheduler={cfg.lr_scheduler} "
                f"{precision_log_string(precision, allow_tf32=cfg.allow_tf32)}"
            )
            print(f"chromosome_split_mode={cfg.chromosome_split_mode}")
            print(f"train_chromosomes={','.join(train_chromosomes)}")
            print(f"val_chromosomes={','.join(val_chromosomes)}")
            print(f"trainable_params={parameter_count:,}")
            for note in runtime_model_notes:
                print(f"runtime_model_adjustment={note}")
            if cfg.length_warmup_enabled:
                print(
                    f"length_warmup=enabled stage1_steps={stage1_steps} stage2_steps={stage2_steps} "
                    f"current_phase={current_length_state.phase} current_seq_len={current_length_state.seq_len}"
                )
            if bool(cfg.curriculum.get("enabled", False)):
                print(
                    "curriculum="
                    f"{current_sampling_state.curriculum_phase} "
                    f"counterfactual_fraction={current_sampling_state.counterfactual_fraction:.3f} "
                    f"hard_position_mix={current_sampling_state.hard_position_mix:.3f}"
                )
            write_json(
                run_summary_path(output_dir),
                build_run_summary(
                    cfg=cfg,
                    device=device,
                    precision=precision,
                    parameter_count=parameter_count,
                    train_chromosomes=train_chromosomes,
                    val_chromosomes=val_chromosomes,
                    best_state=best_state,
                    last_step=start_step,
                    latest_validation=latest_validation,
                    tokens_seen=tokens_seen,
                    wandb_run=wandb_run,
                    status="running",
                    memory_preflight=memory_preflight_report,
                ),
            )

        for step in range(start_step + 1, cfg.max_steps + 1):
            length_state = resolve_length_warmup_state(cfg, step)
            sampling_state = resolve_sampling_schedule_state(cfg, step)
            if length_state.seq_len != current_length_state.seq_len or sampling_state != current_sampling_state:
                shutdown_dataloader(train_loader)
                shutdown_dataloader(val_loader)
                train_loader, val_loader, _resolved_train, _resolved_val = build_dataloaders(
                    cfg,
                    device,
                    train_chromosomes=train_chromosomes,
                    val_chromosomes=val_chromosomes,
                    seq_len=length_state.seq_len,
                    sampling_state=sampling_state,
                    region_loss_ema=region_loss_ema,
                )
                train_iterator = iter(train_loader)
                if ctx.is_main:
                    if length_state.seq_len != current_length_state.seq_len:
                        print(
                            f"length_warmup_phase_switch step={step} "
                            f"phase={length_state.phase} seq_len={length_state.seq_len}"
                        )
                    if sampling_state != current_sampling_state:
                        print(
                            f"sampling_schedule_switch step={step} "
                            f"curriculum_phase={sampling_state.curriculum_phase} "
                            f"cf_fraction={sampling_state.counterfactual_fraction:.3f} "
                            f"hard_position_mix={sampling_state.hard_position_mix:.3f} "
                            f"conservation_mix={sampling_state.conservation_mix:.3f} "
                            f"cds_enrichment_fraction={sampling_state.cds_enrichment_fraction:.3f} "
                            f"splice_window_fraction={sampling_state.splice_window_oversample_fraction:.3f}"
                        )
            current_length_state = length_state
            current_sampling_state = sampling_state
            lr = get_lr(step, cfg)
            set_optimizer_lr(optimizer, lr)

            micro_batches: list[dict[str, Any]] = []
            step_token_count = 0
            for _ in range(cfg.grad_accum_steps):
                if ctx.is_main and not logged_first_train_fetch:
                    print("[startup] rank0 fetching first train batch")
                    logged_first_train_fetch = True
                batch = next(train_iterator)
                batch_tokens = batch_token_count(batch) * ctx.world_size
                tokens_seen += batch_tokens
                step_token_count += batch_tokens
                train_chrom_counts.update(cast(list[str], batch["chroms"]))
                micro_batches.append(batch)

            t0 = time.time()
            if ctx.is_main and not logged_first_train_step:
                print("[startup] rank0 entering first train_step")
                logged_first_train_step = True
            step_result = train_step(
                model,
                micro_batches,
                optimizer,
                device,
                cfg,
                precision,
                log_sigmas,
                gradnorm_balancer,
                ctx,
                step=step,
            )
            if gradnorm_balancer is not None:
                gradnorm_balancer.update(step_result.stats)
            if cfg.hard_position_mix > 0.0 and step >= cfg.hard_position_warmup_steps:
                region_loss_ema.update(
                    {
                        0: step_result.stats.get("mlm_region_loss_intergenic", 0.0),
                        1: step_result.stats.get("mlm_region_loss_intron", 0.0),
                        2: step_result.stats.get("mlm_region_loss_noncoding_exon", 0.0),
                        3: step_result.stats.get("mlm_region_loss_utr", 0.0),
                        4: step_result.stats.get("mlm_region_loss_cds", 0.0),
                    }
                )
            step_time = time.time() - t0
            last_completed_step = step
            running_token_count += step_token_count
            running_step_time_total += step_time

            for key, value in step_result.stats.items():
                running_scalars.setdefault(key, SmoothedValue()).update(value)
            running_scalars["grad_norm"].update(step_result.grad_norm)
            running_scalars["step_time"].update(step_time)
            running_metrics.merge(step_result.metrics)

            if (step % cfg.log_every == 0 or step == 1) and ctx.is_main:
                scalar_summary = {key: value.avg for key, value in running_scalars.items()}
                metric_summary = running_metrics.summary()
                tokens_per_sec = running_token_count / max(1e-8, running_step_time_total)
                elapsed_minutes = (time.time() - train_start) / 60.0
                chrom_freq = format_chromosome_frequency(train_chrom_counts)

                loss_line = (
                    f"[step {step:>7d}/{cfg.max_steps}] "
                    f"phase={current_length_state.phase} "
                    f"seq_len={current_length_state.seq_len} "
                    f"lr={lr:.6e} "
                    f"loss={scalar_summary['loss']:.4f} "
                    f"mlm={scalar_summary['loss_mlm']:.4f} "
                    f"p100={scalar_summary['loss_phylo100']:.4f} "
                    f"p470={scalar_summary['loss_phylo470']:.4f} "
                    f"splice={scalar_summary['loss_structure']:.4f} "
                    f"region={scalar_summary['loss_region']:.4f} "
                    f"mut={scalar_summary['loss_mutation_effect']:.4f} "
                    f"rc={scalar_summary['loss_rc']:.4f} "
                    f"gnorm={scalar_summary['grad_norm']:.4f} "
                    f"step_time={scalar_summary['step_time']:.3f}s "
                    f"tok/s={tokens_per_sec:,.0f} "
                    f"elapsed={elapsed_minutes:.1f}m"
                )
                if effective_balancing == "uncertainty":
                    loss_line += (
                        "\n" + " " * 12
                        + f"sig(mlm)={scalar_summary['sigma_mlm']:.4f} "
                        + f"sig(p100)={scalar_summary['sigma_phylo100']:.4f} "
                        + f"sig(p470)={scalar_summary['sigma_phylo470']:.4f} "
                        + f"sig(splice)={scalar_summary['sigma_structure']:.4f} "
                        + f"sig(mut)={scalar_summary['sigma_mutation_effect']:.4f}"
                    )
                    if cfg.w_counterfactual != 0.0:
                        loss_line += f" sig(cf)={scalar_summary['sigma_counterfactual']:.4f}"
                    if cfg.w_allele != 0.0:
                        loss_line += f" sig(allele)={scalar_summary['sigma_allele']:.4f}"
                    if cfg.w_codon != 0.0:
                        loss_line += f" sig(codon)={scalar_summary['sigma_codon']:.4f}"
                    if cfg.w_encode != 0.0:
                        loss_line += f" sig(encode)={scalar_summary['sigma_encode']:.4f}"
                print(loss_line)
                if (
                    cfg.w_counterfactual != 0.0
                    or cfg.w_codon != 0.0
                    or cfg.w_encode != 0.0
                    or cfg.w_mutation_effect != 0.0
                    or cfg.w_allele != 0.0
                ):
                    print(
                        " " * 12
                        + f"mut_rank={scalar_summary['loss_mutation_effect_rank']:.4f} "
                        + f"allele={scalar_summary['loss_allele']:.4f} "
                        + f"allele_rank={scalar_summary['loss_allele_rank']:.4f} "
                        + f"allele_acc={scalar_summary['allele_effect_accuracy']:.3f} "
                        + f"allele_mae={scalar_summary['allele_severity_mae']:.3f} "
                        + f"cf={scalar_summary['loss_counterfactual']:.4f} "
                        + f"cf_local={scalar_summary['loss_counterfactual_local']:.4f} "
                        + f"cf_far={scalar_summary['loss_counterfactual_far']:.4f} "
                        + f"cf_active={scalar_summary['cf_active_fraction']:.3f} "
                        + f"cf_w={scalar_summary['counterfactual_effective_weight']:.4f} "
                        + f"codon={scalar_summary['loss_codon']:.4f} "
                        + f"encode={scalar_summary['loss_encode']:.4f}"
                    )
                print(
                    " " * 12
                    + f"mlm_tok={scalar_summary['mlm_supervised_tokens']:.1f} "
                    + f"aux_tok={scalar_summary['aux_valid_tokens']:.1f} "
                    + "splice_tok(bg/core/region)="
                    + f"{scalar_summary['splice_bg_tokens']:.1f}/"
                    + f"{scalar_summary['splice_core_tokens']:.1f}/"
                    + f"{scalar_summary['splice_region_tokens']:.1f} "
                    + "splice_w(bg/core/region)="
                    + f"{scalar_summary['splice_weight_bg']:.2f}/"
                    + f"{scalar_summary['splice_weight_core']:.2f}/"
                    + f"{scalar_summary['splice_weight_region']:.2f}"
                )
                if cfg.w_region != 0.0:
                    print(
                        " " * 12
                        + "region_tok(ig/intron/ncexon/utr/cds)="
                        + f"{scalar_summary['region_intergenic_tokens']:.1f}/"
                        + f"{scalar_summary['region_intron_tokens']:.1f}/"
                        + f"{scalar_summary['region_noncoding_exon_tokens']:.1f}/"
                        + f"{scalar_summary['region_utr_tokens']:.1f}/"
                        + f"{scalar_summary['region_cds_tokens']:.1f}"
                    )
                print(
                    " " * 12
                    + f"n_frac={scalar_summary['n_fraction']:.3f} "
                    + f"splice_pos={scalar_summary['splice_positive_fraction']:.3f} "
                    + f"splice_core={scalar_summary['splice_core_fraction']:.3f} "
                    + f"exon_frac={scalar_summary['exon_fraction']:.3f} "
                    + f"cds_frac={scalar_summary['cds_fraction']:.3f} "
                    + f"utr_frac={scalar_summary['utr_fraction']:.3f} "
                    + f"intron_frac={scalar_summary['intron_fraction']:.3f} "
                    + f"n_fallback={scalar_summary['n_filter_fallback_fraction']:.3f} "
                    + f"chrom_freq={chrom_freq}"
                )
                print(" " * 12 + format_metric_summary(metric_summary))

                train_row = build_history_row(
                    phase="train",
                    step=step,
                    lr=lr,
                    elapsed_minutes=elapsed_minutes,
                    seq_len=current_length_state.seq_len,
                    length_phase=current_length_state.phase,
                    scalar_stats={
                        key: value for key, value in scalar_summary.items() if key not in {"grad_norm", "step_time"}
                    },
                    metric_stats=metric_summary,
                    grad_norm=scalar_summary["grad_norm"],
                    step_time=scalar_summary["step_time"],
                    tokens_per_sec=tokens_per_sec,
                    tokens_seen=tokens_seen,
                    chrom_freq=chrom_freq,
                )
                append_history_row(output_dir, train_row)
                train_step_has_eval = cfg.eval_every > 0 and step % cfg.eval_every == 0
                log_wandb_row(wandb_run, train_row, commit=not train_step_has_eval)

            if step % cfg.log_every == 0 or step == 1:
                for value in running_scalars.values():
                    value.reset()
                running_metrics.reset()
                train_chrom_counts = Counter()
                running_token_count = 0
                running_step_time_total = 0.0

            if cfg.eval_every > 0 and step % cfg.eval_every == 0:
                if ctx.is_main and not logged_first_eval_fetch:
                    print("[startup] rank0 fetching first eval batch")
                    logged_first_eval_fetch = True
                val_scalar_stats, val_metric_stats, val_chrom_counts = eval_steps(
                    model,
                    val_loader,
                    device,
                    cfg,
                    precision,
                    num_steps=20,
                    ctx=ctx,
                )

                if ctx.is_main:
                    elapsed_minutes = (time.time() - train_start) / 60.0
                    chrom_freq = format_chromosome_frequency(val_chrom_counts)

                    print(
                        f"[eval step {step}] "
                        f"phase={current_length_state.phase} "
                        f"seq_len={current_length_state.seq_len} "
                        f"loss={val_scalar_stats['loss']:.4f} "
                        f"mlm={val_scalar_stats['loss_mlm']:.4f} "
                        f"p100={val_scalar_stats['loss_phylo100']:.4f} "
                        f"p470={val_scalar_stats['loss_phylo470']:.4f} "
                        f"splice={val_scalar_stats['loss_structure']:.4f} "
                        f"region={val_scalar_stats['loss_region']:.4f} "
                        f"mut={val_scalar_stats['loss_mutation_effect']:.4f} "
                        f"allele={val_scalar_stats['loss_allele']:.4f} "
                        f"rc={val_scalar_stats['loss_rc']:.4f}"
                    )
                    print(
                        " " * 12
                        + f"mlm_tok={val_scalar_stats['mlm_supervised_tokens']:.1f} "
                        + f"aux_tok={val_scalar_stats['aux_valid_tokens']:.1f} "
                        + "splice_tok(bg/core/region)="
                        + f"{val_scalar_stats['splice_bg_tokens']:.1f}/"
                        + f"{val_scalar_stats['splice_core_tokens']:.1f}/"
                        + f"{val_scalar_stats['splice_region_tokens']:.1f} "
                        + "splice_w(bg/core/region)="
                        + f"{val_scalar_stats['splice_weight_bg']:.2f}/"
                        + f"{val_scalar_stats['splice_weight_core']:.2f}/"
                        + f"{val_scalar_stats['splice_weight_region']:.2f}"
                    )
                    print(
                        " " * 12
                        + f"n_frac={val_scalar_stats['n_fraction']:.3f} "
                        + f"splice_pos={val_scalar_stats['splice_positive_fraction']:.3f} "
                        + f"splice_core={val_scalar_stats['splice_core_fraction']:.3f} "
                        + f"exon_frac={val_scalar_stats['exon_fraction']:.3f} "
                        + f"cds_frac={val_scalar_stats['cds_fraction']:.3f} "
                        + f"utr_frac={val_scalar_stats['utr_fraction']:.3f} "
                        + f"intron_frac={val_scalar_stats['intron_fraction']:.3f} "
                        + f"n_fallback={val_scalar_stats['n_filter_fallback_fraction']:.3f} "
                        + f"chrom_freq={chrom_freq}"
                    )
                    print(" " * 12 + format_metric_summary(val_metric_stats))

                    val_row = build_history_row(
                        phase="val",
                        step=step,
                        lr=lr,
                        elapsed_minutes=elapsed_minutes,
                        seq_len=current_length_state.seq_len,
                        length_phase=current_length_state.phase,
                        scalar_stats=val_scalar_stats,
                        metric_stats=val_metric_stats,
                        tokens_seen=tokens_seen,
                        chrom_freq=chrom_freq,
                    )
                    latest_validation = val_row
                    append_history_row(output_dir, val_row)
                    log_wandb_row(wandb_run, val_row, commit=True)

                    if best_state.should_update(val_scalar_stats):
                        best_path = best_checkpoint_path(output_dir)
                        best_state.update(step, val_scalar_stats, best_path)
                        save_checkpoint(
                            best_path,
                            model,
                            optimizer,
                            step,
                            config,
                            metadata=checkpoint_metadata(
                                best_state,
                                wandb_run,
                                latest_validation,
                                tokens_seen,
                                precision,
                            ),
                        )
                        print(f"saved_best_checkpoint={best_path} loss={best_state.loss:.4f}")

                    write_json(
                        run_summary_path(output_dir),
                        build_run_summary(
                            cfg=cfg,
                            device=device,
                            precision=precision,
                            parameter_count=parameter_count,
                            train_chromosomes=train_chromosomes,
                            val_chromosomes=val_chromosomes,
                            best_state=best_state,
                            last_step=step,
                            latest_validation=latest_validation,
                            tokens_seen=tokens_seen,
                            wandb_run=wandb_run,
                            status="running",
                            memory_preflight=memory_preflight_report,
                        ),
                    )

            if cfg.save_every > 0 and step % cfg.save_every == 0 and ctx.is_main:
                path = checkpoint_path(output_dir, step)
                save_checkpoint(
                    path,
                    model,
                    optimizer,
                    step,
                    config,
                    metadata=checkpoint_metadata(best_state, wandb_run, latest_validation, tokens_seen, precision),
                )
                cleanup_old_checkpoints(output_dir, cfg.max_checkpoints_to_keep)
                print(f"saved_checkpoint={path}")

        if ctx.is_main:
            final_path = final_checkpoint_path(output_dir)
            save_checkpoint(
                final_path,
                model,
                optimizer,
                last_completed_step,
                config,
                metadata=checkpoint_metadata(best_state, wandb_run, latest_validation, tokens_seen, precision),
            )
            print(f"saved_final_checkpoint={final_path}")
            write_json(
                run_summary_path(output_dir),
                build_run_summary(
                    cfg=cfg,
                    device=device,
                    precision=precision,
                    parameter_count=parameter_count,
                    train_chromosomes=train_chromosomes,
                    val_chromosomes=val_chromosomes,
                    best_state=best_state,
                    last_step=last_completed_step,
                    latest_validation=latest_validation,
                    tokens_seen=tokens_seen,
                    wandb_run=wandb_run,
                    status="completed",
                    memory_preflight=memory_preflight_report,
                ),
            )
    finally:
        shutdown_dataloader(train_loader)
        shutdown_dataloader(val_loader)
        if wandb_run is not None:
            wandb_run.finish()
        cleanup_distributed(ctx)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the reproducible Lumina baseline trainer.")
    parser.add_argument("--config")

    def add_train_arg(*name_or_flags: str, **kwargs: Any) -> None:
        parser.add_argument(*name_or_flags, default=argparse.SUPPRESS, **kwargs)

    add_train_arg("--fasta-path")
    add_train_arg("--phylo100-bw-path")
    add_train_arg("--phylo470-bw-path")
    add_train_arg("--gtf-path")
    add_train_arg("--dataset-mode", choices=DATASET_MODE_CHOICES)
    add_train_arg("--abraom-data-root")
    add_train_arg("--abraom-dataset-arm", choices=("wild_only", "abraom_uniform", "abraom_weighted"))
    add_train_arg("--abraom-train-split")
    add_train_arg("--abraom-val-split")
    add_train_arg("--abraom-max-shards-per-split", type=int)
    add_train_arg("--abraom-shuffle-shards", action=argparse.BooleanOptionalAction)
    add_train_arg("--chromosomes")
    add_train_arg("--train-chromosomes")
    add_train_arg("--val-chromosomes")
    add_train_arg("--chromosome-split-mode", choices=CHROMOSOME_SPLIT_MODE_CHOICES)

    add_train_arg("--model", choices=registered_model_keys())
    add_train_arg("--seq-len", type=int)
    add_train_arg("--length-warmup-enabled", action=argparse.BooleanOptionalAction)
    add_train_arg("--length-warmup-initial-seq-len", type=int)
    add_train_arg("--length-warmup-transition-fraction", type=float)
    add_train_arg("--length-warmup-stage1-warmup-fraction", type=float)
    add_train_arg("--length-warmup-stage1-end-lr-scale", type=float)
    add_train_arg("--length-warmup-stage2-warmup-fraction", type=float)
    add_train_arg("--length-warmup-stage2-peak-lr-scale", type=float)
    add_train_arg("--length-warmup-final-lr-scale", type=float)
    add_train_arg("--batch-size", type=int)
    add_train_arg("--num-workers", type=int)
    add_train_arg("--prefetch-factor", type=int)
    add_train_arg("--mask-prob", type=float)
    add_train_arg("--mean-span-len", type=int)
    add_train_arg("--conservation-mix", type=float)
    add_train_arg("--cds-enrichment-fraction", type=float)
    add_train_arg("--splice-window-oversample-fraction", type=float)
    add_train_arg("--hard-position-mix", type=float)
    add_train_arg("--hard-position-ema-decay", type=float)
    add_train_arg("--hard-position-warmup-steps", type=int)
    add_train_arg("--counterfactual-fraction", type=float)
    add_train_arg("--counterfactual-local-radius", type=int)
    add_train_arg("--counterfactual-far-radius", type=int)
    add_train_arg("--counterfactual-local-similarity-target", type=float)
    add_train_arg("--counterfactual-weighting", choices=COUNTERFACTUAL_WEIGHTING_CHOICES)
    add_train_arg("--allele-fraction", type=float)
    add_train_arg("--allele-score-window", type=int)
    add_train_arg("--allele-max-rows-per-batch", type=int)
    add_train_arg("--allele-rank-every-n-steps", type=int)
    add_train_arg("--auxiliary-schedule", choices=AUXILIARY_SCHEDULE_CHOICES)
    add_train_arg("--clinvar-blocklist-bed-path")

    add_train_arg("--grad-accum-steps", type=int)
    add_train_arg("--max-steps", type=int)
    add_train_arg("--lr", type=float)
    add_train_arg("--lr-scheduler", choices=LR_SCHEDULER_CHOICES)
    add_train_arg("--min-lr", type=float)
    add_train_arg("--weight-decay", type=float)
    add_train_arg("--beta1", type=float)
    add_train_arg("--beta2", type=float)
    add_train_arg("--grad-clip", type=float)
    add_train_arg("--warmup-steps", type=int)
    add_train_arg("--log-every", type=int)
    add_train_arg("--eval-every", type=int)
    add_train_arg("--save-every", type=int)
    add_train_arg("--max-checkpoints-to-keep", type=int)

    add_train_arg("--w-mlm", type=float)
    add_train_arg("--w-phylo100", type=float)
    add_train_arg("--w-phylo470", type=float)
    add_train_arg("--w-structure", type=float)
    add_train_arg("--w-rc", type=float)
    add_train_arg("--w-region", type=float)
    add_train_arg("--w-aa", type=float)
    add_train_arg("--w-codon-phylo", type=float)
    add_train_arg("--w-mutation-effect", type=float)
    add_train_arg("--w-counterfactual", type=float)
    add_train_arg("--w-allele", type=float)
    add_train_arg("--w-codon", type=float)
    add_train_arg("--w-encode", type=float)
    add_train_arg("--w-conservation-bin", type=float)
    add_train_arg("--w-splice-distance", type=float)
    add_train_arg("--w-codon-pos", type=float)
    add_train_arg("--w-exon-phase", type=float)
    add_train_arg("--w-counterfactual-snv", type=float)
    add_train_arg("--w-counterfactual-severity", type=float)
    add_train_arg("--lambda-mutation-effect-rank", type=float)
    add_train_arg("--lambda-allele-rank", type=float)
    add_train_arg("--lambda-allele-severity", type=float)
    add_train_arg("--lambda-allele-swap", type=float)
    add_train_arg("--lambda-allele-far", type=float)
    add_train_arg("--splice-class-weight-cap", type=float)
    add_train_arg("--region-class-weight-cap", type=float)
    add_train_arg("--aa-class-weight-cap", type=float)
    add_train_arg("--rc-loss-type", choices=("cosine", "mse"))
    add_train_arg("--use-uncertainty-weighting", action=argparse.BooleanOptionalAction)
    add_train_arg("--loss-balancing", choices=LOSS_BALANCING_CHOICES)
    add_train_arg("--aux-loss-warmup-steps", type=int)

    add_train_arg("--seed", type=int)
    add_train_arg("--output-dir")
    add_train_arg("--resume-from")
    add_train_arg("--init-from-checkpoint")
    add_train_arg("--precision", choices=("auto", "mxfp8", "bf16", "fp32"))
    add_train_arg("--allow-tf32", action=argparse.BooleanOptionalAction)
    add_train_arg("--memory-preflight-enabled", action=argparse.BooleanOptionalAction)

    add_train_arg("--wandb-enabled", action=argparse.BooleanOptionalAction)
    add_train_arg("--wandb-project")
    add_train_arg("--wandb-entity")
    add_train_arg("--wandb-run-name")
    add_train_arg("--wandb-tags")
    return parser


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    raw_args = dict(vars(args))
    config_path = raw_args.pop("config", None)

    config_data = asdict(TrainConfig())
    if config_path is not None:
        config_data.update(load_yaml_train_config(Path(config_path).expanduser()))

    config_data.update(normalize_train_config_overrides(raw_args, source="command line"))
    return resolve_train_config(TrainConfig(**config_data))


def main() -> None:
    args = build_arg_parser().parse_args()
    train(config_from_args(args))


if __name__ == "__main__":
    main()
