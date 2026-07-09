from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from src.models import get_model_spec
from src.precision import PrecisionMode
from src.repo_paths import REPO_ROOT

TaskType = Literal["functional", "annotation"]
SchedulerName = Literal["cosine", "modified_square_decay"]
FeatureSource = Literal["hidden", "decoder"]
FunctionalHeadType = Literal[
    "linear",
    "mlp",
    "local-conv",
    "gated-hybrid",
    "multi-scale-dilated",
    "global-context",
    "context-pyramid",
    "v10-representation-pyramid",
    "v10-bio-aux-pyramid",
    "v10-assay-gated-bio-pyramid",
    "v10-bioprogram-stack",
    "v10-profile-count-bioaux",
    "v10-profile-count-bioaux-rc-gated-residual",
    "v10-assay-rescue-hybrid",
    "v10-biocov-residual",
]
FunctionalHeadAuxFeatures = Literal["none", "phylo", "structure", "phylo-structure"]
FunctionalHeadOutputBiasInit = Literal["none", "scaled-track-mean"]

DEFAULT_MODEL_VERSION = "beat-v7"
DEFAULT_MODEL_NAME_FOR_LEADERBOARD = "Lumina beat-v7"
DEFAULT_CHECKPOINT_S3_PREFIX = (
    "s3://ai4bio-lumina-experiments-v2/lumina-ssm/experiments/beat-v7-12m-15ep-32k/"
    "lumina-ssm-beat-v7-12m-15ep-32k-20260423192051/output/model.tar.gz"
)
DEFAULT_DATASET_REPO_ID = "InstaDeepAI/NTv3_benchmark_dataset"
DEFAULT_LOCAL_CHECKPOINT_DIR = (
    REPO_ROOT / "data" / "checkpoints" / "ntv3" / "lumina-ssm-beat-v7-12m-15ep-32k-20260423192051"
)
DEFAULT_DATASET_ROOT = REPO_ROOT / "data" / "datasets" / "ntv3"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "ntv3"
OFFICIAL_HUMAN_FUNCTIONAL_PRESET = "official-human-functional"


def official_human_functional_overrides() -> dict[str, object]:
    return {
        "preset_name": OFFICIAL_HUMAN_FUNCTIONAL_PRESET,
        "sequence_length": 32_768,
        "keep_target_center_fraction": 0.375,
        "train_overlap": 0.999,
        "batch_size": 4,
        "grad_accum_steps": 8,
        "num_steps_training": 19_932,
        "validate_every_n_steps": 500,
        "num_validation_samples": 1_000,
        "initial_learning_rate": 1e-5,
        "learning_rate": 5e-5,
        "num_steps_warmup": 598,
        "scheduler_name": "modified_square_decay",
        "final_learning_rate_multiplier": 0.5,
        "weight_decay": 0.01,
        "log_every_n_steps": 50,
        "save_every_n_steps": 4_000,
        "max_checkpoints_to_keep": 3,
        "freeze_backbone": False,
        "seed": 0,
        "precision": "fp32",
        "num_workers": 16,
    }


@dataclass
class NTv3BenchmarkConfig:
    model_family: str = "lumina"
    model_version: str = DEFAULT_MODEL_VERSION
    checkpoint_dir: str = str(DEFAULT_LOCAL_CHECKPOINT_DIR)
    checkpoint_path: str | None = None
    checkpoint_s3_prefix: str = DEFAULT_CHECKPOINT_S3_PREFIX
    resume_from_checkpoint: str | None = None
    auto_resume: bool = False

    dataset_repo_id: str = DEFAULT_DATASET_REPO_ID
    dataset_root: str = str(DEFAULT_DATASET_ROOT)
    species_name: str = "human"
    task_type: TaskType = "functional"

    sequence_length: int = 32_768
    keep_target_center_fraction: float = 0.375
    train_overlap: float = 0.999

    batch_size: int = 2
    grad_accum_steps: int = 4
    max_runtime_batch_size_per_rank: int | None = None
    num_steps_training: int = 2_000
    validate_every_n_steps: int = 200
    num_validation_samples: int = 1_000
    num_test_samples: int | None = None
    initial_learning_rate: float | None = None
    learning_rate: float = 1e-4
    num_steps_warmup: int | None = None
    scheduler_name: SchedulerName = "cosine"
    final_learning_rate_multiplier: float = 0.5
    weight_decay: float = 0.01
    head_learning_rate: float | None = None
    backbone_learning_rate: float | None = None
    decoder_learning_rate: float | None = None
    backbone_layerwise_lr_decay: float | None = None
    head_only_warmup_steps: int = 0
    no_weight_decay_norm_bias: bool = False
    ema_decay: float | None = None
    warmup_fraction: float = 0.03
    grad_clip: float = 1.0
    log_every_n_steps: int = 50
    save_every_n_steps: int = 4_000
    max_checkpoints_to_keep: int = 3
    freeze_backbone: bool = True
    feature_source: FeatureSource = "hidden"
    functional_head_type: FunctionalHeadType = "linear"
    functional_head_hidden_dim: int | None = None
    functional_head_dropout: float = 0.05
    functional_head_kernel_size: int = 15
    functional_head_aux_features: FunctionalHeadAuxFeatures = "none"
    functional_head_aux_projection_dim: int = 16
    functional_head_output_bias_init: FunctionalHeadOutputBiasInit = "none"
    functional_rc_consistency_weight: float = 0.0

    seed: int = 42
    device: str = "cuda"
    precision: PrecisionMode = "auto"
    allow_tf32: bool = True
    num_workers: int = 6
    prefetch_factor: int = 4

    output_dir: str = str(DEFAULT_OUTPUT_ROOT / "human" / "functional")
    overwrite: bool = False

    wandb_enabled: bool = False
    wandb_project: str = "lumina-ntv3"
    wandb_entity: str | None = None
    wandb_run_name: str | None = None
    wandb_tags: list[str] = field(default_factory=list)

    preset_name: str | None = None
    model_name_for_leaderboard: str = DEFAULT_MODEL_NAME_FOR_LEADERBOARD
    run_id: str | None = None

    def resolved_checkpoint_path(self) -> Path:
        if self.checkpoint_path:
            return Path(self.checkpoint_path).expanduser().resolve()
        return (Path(self.checkpoint_dir).expanduser().resolve() / "best_checkpoint.pt").resolve()

    def resolved_resume_checkpoint_path(self) -> Path | None:
        if not self.resume_from_checkpoint:
            return None
        return Path(self.resume_from_checkpoint).expanduser().resolve()

    def resolved_output_dir(self) -> Path:
        return Path(self.output_dir).expanduser().resolve()

    def resolved_dataset_root(self) -> Path:
        return Path(self.dataset_root).expanduser().resolve()

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    def resolved_initial_learning_rate(self) -> float:
        return float(self.learning_rate if self.initial_learning_rate is None else self.initial_learning_rate)

    def resolved_num_steps_warmup(self) -> int:
        if self.num_steps_warmup is not None:
            return int(self.num_steps_warmup)
        return max(0, int(self.num_steps_training * self.warmup_fraction))

    def validate(self, *, require_paths: bool = True) -> None:
        if self.model_family != "lumina":
            raise ValueError(
                "NTv3 benchmark currently supports only "
                f"model_family='lumina', got {self.model_family!r}."
            )
        try:
            get_model_spec(self.model_version)
        except ValueError as exc:
            raise ValueError(
                "NTv3 benchmark requires a registered Lumina model_version, "
                f"got {self.model_version!r}."
            ) from exc
        if self.task_type not in {"functional", "annotation"}:
            raise ValueError(f"task_type must be 'functional' or 'annotation', got {self.task_type!r}.")
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive.")
        if not 0.0 < self.keep_target_center_fraction <= 1.0:
            raise ValueError("keep_target_center_fraction must be in (0, 1].")
        if not 0.0 <= self.train_overlap < 1.0:
            raise ValueError("train_overlap must be in [0, 1).")
        if self.batch_size <= 0 or self.grad_accum_steps <= 0:
            raise ValueError("batch_size and grad_accum_steps must both be positive.")
        if self.max_runtime_batch_size_per_rank is not None and self.max_runtime_batch_size_per_rank <= 0:
            raise ValueError("max_runtime_batch_size_per_rank must be positive when provided.")
        if self.num_steps_training <= 0:
            raise ValueError("num_steps_training must be positive.")
        if self.validate_every_n_steps <= 0 or self.log_every_n_steps <= 0:
            raise ValueError("validate_every_n_steps and log_every_n_steps must be positive.")
        if self.num_validation_samples <= 0:
            raise ValueError("num_validation_samples must be positive.")
        if self.num_test_samples is not None and self.num_test_samples <= 0:
            raise ValueError("num_test_samples must be positive when provided.")
        if self.initial_learning_rate is not None and self.initial_learning_rate <= 0.0:
            raise ValueError("initial_learning_rate must be positive when provided.")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("learning_rate must be positive and weight_decay must be non-negative.")
        for field_name, value in (
            ("head_learning_rate", self.head_learning_rate),
            ("backbone_learning_rate", self.backbone_learning_rate),
            ("decoder_learning_rate", self.decoder_learning_rate),
            ("backbone_layerwise_lr_decay", self.backbone_layerwise_lr_decay),
            ("ema_decay", self.ema_decay),
        ):
            if value is not None and value <= 0.0:
                raise ValueError(f"{field_name} must be positive when provided.")
        if self.backbone_layerwise_lr_decay is not None and self.backbone_layerwise_lr_decay > 1.0:
            raise ValueError("backbone_layerwise_lr_decay must be in (0, 1] when provided.")
        if self.ema_decay is not None and not 0.0 < self.ema_decay < 1.0:
            raise ValueError("ema_decay must be in (0, 1) when provided.")
        if self.head_only_warmup_steps < 0:
            raise ValueError("head_only_warmup_steps must be non-negative.")
        if self.scheduler_name not in {"cosine", "modified_square_decay"}:
            raise ValueError(
                f"scheduler_name must be 'cosine' or 'modified_square_decay', got {self.scheduler_name!r}."
            )
        if self.resolved_num_steps_warmup() < 0:
            raise ValueError("num_steps_warmup must be non-negative.")
        if not 0.0 < self.final_learning_rate_multiplier <= 1.0:
            raise ValueError("final_learning_rate_multiplier must be in (0, 1].")
        if not 0.0 <= self.warmup_fraction < 1.0:
            raise ValueError("warmup_fraction must be in [0, 1).")
        if self.grad_clip <= 0.0:
            raise ValueError("grad_clip must be positive.")
        if self.save_every_n_steps < 0:
            raise ValueError("save_every_n_steps must be non-negative.")
        if self.max_checkpoints_to_keep <= 0:
            raise ValueError("max_checkpoints_to_keep must be positive.")
        if self.feature_source not in {"hidden", "decoder"}:
            raise ValueError(f"feature_source must be 'hidden' or 'decoder', got {self.feature_source!r}.")
        if self.functional_head_type not in {
            "linear",
            "mlp",
            "local-conv",
            "gated-hybrid",
            "multi-scale-dilated",
            "global-context",
            "context-pyramid",
            "v10-representation-pyramid",
            "v10-bio-aux-pyramid",
            "v10-assay-gated-bio-pyramid",
            "v10-bioprogram-stack",
            "v10-profile-count-bioaux",
            "v10-profile-count-bioaux-rc-gated-residual",
            "v10-assay-rescue-hybrid",
            "v10-biocov-residual",
        }:
            raise ValueError(
                "functional_head_type must be 'linear', 'mlp', 'local-conv', 'gated-hybrid', "
                "'multi-scale-dilated', 'global-context', 'context-pyramid', "
                "'v10-representation-pyramid', 'v10-bio-aux-pyramid', "
                "'v10-assay-gated-bio-pyramid', 'v10-bioprogram-stack', "
                "'v10-profile-count-bioaux', 'v10-profile-count-bioaux-rc-gated-residual', "
                "'v10-assay-rescue-hybrid', or 'v10-biocov-residual', "
                f"got {self.functional_head_type!r}."
            )
        if self.functional_head_hidden_dim is not None and self.functional_head_hidden_dim <= 0:
            raise ValueError("functional_head_hidden_dim must be positive when provided.")
        if not 0.0 <= self.functional_head_dropout < 1.0:
            raise ValueError("functional_head_dropout must be in [0, 1).")
        if self.functional_head_kernel_size <= 0 or self.functional_head_kernel_size % 2 == 0:
            raise ValueError("functional_head_kernel_size must be a positive odd integer.")
        if self.functional_head_aux_features not in {"none", "phylo", "structure", "phylo-structure"}:
            raise ValueError(
                "functional_head_aux_features must be 'none', 'phylo', 'structure', "
                f"or 'phylo-structure', got {self.functional_head_aux_features!r}."
            )
        if self.functional_head_aux_features != "none" and self.feature_source != "hidden":
            raise ValueError("functional_head_aux_features requires feature_source='hidden'.")
        if self.functional_head_aux_features != "none" and self.functional_head_type not in {
            "mlp",
            "gated-hybrid",
            "multi-scale-dilated",
            "global-context",
            "context-pyramid",
        }:
            raise ValueError(
                "functional_head_aux_features currently requires functional_head_type='mlp', "
                "'gated-hybrid', 'multi-scale-dilated', 'global-context', or 'context-pyramid'."
            )
        if self.functional_head_aux_projection_dim <= 0:
            raise ValueError("functional_head_aux_projection_dim must be positive.")
        if self.functional_head_output_bias_init not in {"none", "scaled-track-mean"}:
            raise ValueError(
                "functional_head_output_bias_init must be 'none' or 'scaled-track-mean', "
                f"got {self.functional_head_output_bias_init!r}."
            )
        if self.functional_rc_consistency_weight < 0.0:
            raise ValueError("functional_rc_consistency_weight must be non-negative.")
        if self.functional_rc_consistency_weight > 0.0 and self.task_type != "functional":
            raise ValueError("functional_rc_consistency_weight is only valid for task_type='functional'.")
        if self.precision not in {"auto", "bf16", "fp32"}:
            raise ValueError(f"precision must be 'auto', 'bf16', or 'fp32', got {self.precision!r}.")
        if self.prefetch_factor <= 0:
            raise ValueError("prefetch_factor must be positive.")
        if self.preset_name == OFFICIAL_HUMAN_FUNCTIONAL_PRESET:
            if self.species_name != "human" or self.task_type != "functional":
                raise ValueError(
                    "official-human-functional preset requires species_name='human' and task_type='functional'."
                )
            if self.freeze_backbone:
                raise ValueError("official-human-functional preset requires full fine-tuning (--train-backbone).")
            if self.feature_source != "hidden":
                raise ValueError("official-human-functional preset requires feature_source='hidden'.")
            if self.functional_head_type != "linear":
                raise ValueError("official-human-functional preset requires functional_head_type='linear'.")
            if self.functional_head_aux_features != "none":
                raise ValueError("official-human-functional preset requires functional_head_aux_features='none'.")
            if any(
                value is not None
                for value in (self.head_learning_rate, self.backbone_learning_rate, self.decoder_learning_rate)
            ):
                raise ValueError("official-human-functional preset does not allow custom optimizer group LRs.")
            if self.head_only_warmup_steps != 0 or self.no_weight_decay_norm_bias:
                raise ValueError("official-human-functional preset does not allow coupled optimizer warmup settings.")
        if require_paths:
            checkpoint_path = self.resolved_checkpoint_path()
            if not checkpoint_path.is_file():
                raise FileNotFoundError(f"Lumina checkpoint not found: {checkpoint_path}")
            resume_checkpoint = self.resolved_resume_checkpoint_path()
            if resume_checkpoint is not None and not resume_checkpoint.is_file():
                raise FileNotFoundError(f"NTv3 resume checkpoint not found: {resume_checkpoint}")
            dataset_root = self.resolved_dataset_root()
            if not dataset_root.is_dir():
                raise FileNotFoundError(f"NTv3 dataset root not found: {dataset_root}")
