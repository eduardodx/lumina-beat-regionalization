"""Unified configuration for ClinVar fine-tuning (Regimes A and B)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from src.precision import PrecisionMode


@dataclass
class FineTuneConfig:
    """Configuration for ClinVar pathogenicity fine-tuning.

    Regime A: Embedding-only head (representation quality comparison).
    Regime B: Embeddings + biological features (practical utility comparison).
    """

    # -- Model --
    model_family: str = "lumina"
    model_version: str = "beat-v2"
    checkpoint_path: str | None = None

    # -- Regime --
    regime: str = "A"

    # -- Data --
    dataset_path: str = "data/datasets/clinvar/processed/clinvar_dataset.parquet"
    fasta_path: str = "data/hg38/hg38.fa"
    phylo100_bw_path: str | None = None
    phylo470_bw_path: str | None = None
    gtf_path: str | None = None
    context_size: int = 4096
    cache_dir: str | None = None

    # -- Head architecture (shared across regimes) --
    proj_dim: int = 256
    hidden_dim: int = 128
    head_dropout: float = 0.2

    # -- Regime A backbone-native features (Lumina beat-v6/v8) --
    # Selection of pretrained per-position heads whose outputs at the variant
    # position are concatenated into the Regime A variant_repr.  Unsupported
    # backbones silently ignore this field.  Pass ["none"] to force the
    # legacy two-pass alt-ref subtraction path.
    native_feature_heads: list[str] = field(
        default_factory=lambda: ["mutation_effect", "aa", "codon_phylo", "phylo470"]
    )
    head_type: Literal[
        "regime_a",
        "regime_a_v7",
        "regime_a_v8",
        "allele_scorer",
        "regime_a_plus_features",
        "regime_a_bounded_regional",
    ] = "regime_a"
    explicit_feature_columns: list[str] = field(default_factory=list)

    # -- Regional adapter fusion --
    init_finetuned_checkpoint_path: str | None = None
    fusion_mode: Literal["none", "static_lora", "dynamic_lora"] = "none"
    fusion_adapter_paths: list[str] = field(default_factory=list)
    fusion_adapter_names: list[str] = field(default_factory=list)
    freeze_backbone_for_fusion: bool = False
    fusion_gate_hidden_dim: int = 64
    fusion_gate_dropout: float = 0.0

    # -- LoRA (identical across model families for fair comparison) --
    lora_rank: int = 4
    lora_alpha: float = 8.0
    lora_dropout: float = 0.1

    # -- Optimizer --
    lr_backbone: float = 5e-6
    lr_head: float = 5e-4
    wd_backbone: float = 0.01
    wd_head: float = 1e-4

    # -- Training --
    batch_size: int = 4
    grad_accum_steps: int = 16
    max_epochs: int = 5
    warmup_fraction: float = 0.10
    grad_clip: float = 0.5
    freeze_backbone_steps: int = 100
    val_fraction: float = 0.1
    seed: int = 42

    # -- Loss --
    loss_type: str = "focal"
    focal_gamma: float = 2.0
    pos_weight: str | float = 1.0
    pairwise_rank_weight: float = 0.0
    pairwise_rank_margin: float = 0.5
    swap_consistency_weight: float = 0.0
    swap_consistency_margin: float = 0.5

    # -- System --
    device: str = "cuda"
    precision: PrecisionMode = "auto"
    allow_tf32: bool = True
    num_workers: int = 0

    # -- Output --
    output_dir: str = "outputs/clinvar_finetune"
    overwrite: bool = False

    # -- Experiment tracking --
    wandb_enabled: bool = False
    wandb_project: str = "lumina-clinvar"
    wandb_entity: str | None = None
    wandb_run_name: str | None = None
    wandb_tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.model_family == "lumina" and self.model_version == "beat-v8" and self.regime == "A":
            default_native = ["mutation_effect", "aa", "codon_phylo", "phylo470"]
            if self.head_type == "regime_a":
                self.head_type = "regime_a_v8"
            if self.native_feature_heads == default_native:
                self.native_feature_heads = [
                    "allele_repr",
                    "allele_effect_logits",
                    "allele_severity_score",
                    "allele_swap_severity_score",
                    "allele_far_distance",
                    "allele_site_delta",
                    "allele_local_delta",
                    "mlm_logit_ratio",
                    "is_snv",
                ]
            if self.pairwise_rank_weight == 0.0:
                self.pairwise_rank_weight = 0.2
            if self.swap_consistency_weight == 0.0:
                self.swap_consistency_weight = 0.1

    def validate(self) -> None:
        """Raise on invalid configuration."""
        if self.regime not in ("A", "B"):
            raise ValueError(f"regime must be 'A' or 'B', got {self.regime!r}")
        if self.regime == "B" and self.phylo100_bw_path is None:
            raise ValueError("Regime B requires --phylo100-bw-path for biological features")
        if self.regime == "B" and self.gtf_path is None:
            raise ValueError("Regime B requires --gtf-path for CDS/codon features")
        if self.loss_type not in ("bce", "focal"):
            raise ValueError(f"loss_type must be 'bce' or 'focal', got {self.loss_type!r}")
        if self.precision not in ("auto", "mxfp8", "bf16", "fp32"):
            raise ValueError(f"precision must be 'auto', 'mxfp8', 'bf16', or 'fp32', got {self.precision!r}")
        if self.head_type not in (
            "regime_a",
            "regime_a_v7",
            "regime_a_v8",
            "allele_scorer",
            "regime_a_plus_features",
            "regime_a_bounded_regional",
        ):
            raise ValueError(
                "head_type must be 'regime_a', 'regime_a_v7', 'regime_a_v8', "
                "'allele_scorer', 'regime_a_plus_features', or "
                f"'regime_a_bounded_regional', got {self.head_type!r}"
            )
        if (
            self.head_type in ("regime_a_plus_features", "regime_a_bounded_regional")
            and not self.explicit_feature_columns
        ):
            raise ValueError(f"head_type={self.head_type!r} requires explicit_feature_columns.")
        if self.explicit_feature_columns and self.regime != "A":
            raise ValueError("explicit_feature_columns are currently supported only for Regime A.")
        if self.fusion_mode not in ("none", "static_lora", "dynamic_lora"):
            raise ValueError(
                "fusion_mode must be 'none', 'static_lora', or "
                f"'dynamic_lora', got {self.fusion_mode!r}."
            )
        if self.fusion_mode == "none" and (self.fusion_adapter_paths or self.fusion_adapter_names):
            raise ValueError("fusion adapters require fusion_mode='static_lora' or 'dynamic_lora'.")
        if self.fusion_mode in ("static_lora", "dynamic_lora"):
            if not self.init_finetuned_checkpoint_path:
                raise ValueError(f"fusion_mode={self.fusion_mode!r} requires init_finetuned_checkpoint_path.")
            if not self.fusion_adapter_paths:
                raise ValueError(f"fusion_mode={self.fusion_mode!r} requires fusion_adapter_paths.")
            if len(self.fusion_adapter_paths) != len(self.fusion_adapter_names):
                raise ValueError(
                    "fusion_adapter_paths and fusion_adapter_names must have identical lengths "
                    f"({len(self.fusion_adapter_paths)} != {len(self.fusion_adapter_names)})."
                )
            if len(set(self.fusion_adapter_names)) != len(self.fusion_adapter_names):
                raise ValueError("fusion_adapter_names must not contain duplicates.")
        if self.fusion_gate_hidden_dim <= 0:
            raise ValueError(f"fusion_gate_hidden_dim must be positive, got {self.fusion_gate_hidden_dim}.")
        if not 0.0 <= self.fusion_gate_dropout < 1.0:
            raise ValueError(f"fusion_gate_dropout must be in [0, 1), got {self.fusion_gate_dropout}.")
        forbidden_features = {
            "has_brazilian_submitter",
            "has_non_brazilian_submitter",
            "brazilian_submission_rows",
            "non_brazilian_submission_rows",
            "regional_submission_rows",
            "clinvar_regional_cohort",
        }
        leaked = sorted(forbidden_features.intersection(self.explicit_feature_columns))
        if leaked:
            raise ValueError(
                "explicit_feature_columns cannot include ClinVar submitter/provenance fields: "
                + ", ".join(leaked)
            )
        if self.pairwise_rank_weight < 0.0:
            raise ValueError(f"pairwise_rank_weight must be non-negative, got {self.pairwise_rank_weight}.")
        if self.pairwise_rank_margin <= 0.0:
            raise ValueError(f"pairwise_rank_margin must be positive, got {self.pairwise_rank_margin}.")
        if self.swap_consistency_weight < 0.0:
            raise ValueError(
                f"swap_consistency_weight must be non-negative, got {self.swap_consistency_weight}."
            )
        if self.swap_consistency_margin <= 0.0:
            raise ValueError(
                f"swap_consistency_margin must be positive, got {self.swap_consistency_margin}."
            )
        if not 0.0 <= self.val_fraction < 1.0:
            raise ValueError(f"val_fraction must be in [0, 1), got {self.val_fraction}.")
        self._validate_native_feature_heads()

    def _validate_native_feature_heads(self) -> None:
        allowed = {
            "mutation_effect",
            "aa",
            "codon_phylo",
            "phylo470",
            "mlm_logit_ratio",
            "is_snv",
            "allele_repr",
            "allele_effect_logits",
            "allele_severity_score",
            "allele_swap_severity_score",
            "allele_far_distance",
            "allele_site_delta",
            "allele_local_delta",
            "none",
        }
        if not self.native_feature_heads:
            raise ValueError("native_feature_heads must contain at least one entry")
        unknown = [h for h in self.native_feature_heads if h not in allowed]
        if unknown:
            raise ValueError(
                f"native_feature_heads contains unknown entries {unknown!r}; "
                f"allowed values are {sorted(allowed)}"
            )
        if "none" in self.native_feature_heads and len(self.native_feature_heads) > 1:
            raise ValueError("native_feature_heads='none' may only appear on its own")
        if len(set(self.native_feature_heads)) != len(self.native_feature_heads):
            raise ValueError("native_feature_heads must not contain duplicates")
