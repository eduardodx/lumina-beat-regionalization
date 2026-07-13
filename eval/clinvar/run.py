"""CLI entry point for ClinVar fine-tuning.

Supports all four model families under a single interface with identical
hyperparameters for fair comparison.

Examples
--------
# Regime A -- Lumina (representation quality)
# Defaults: --precision bf16 and --allow-tf32 on supported CUDA devices.
uv run python -m eval.clinvar.run --regime A \\
    --model-family lumina --model-version beat-v2 \\
    --checkpoint-path outputs/beat_v2/best_checkpoint.pt \\
    --fasta-path data/hg38/hg38.fa \\
    --output-dir outputs/clinvar/lumina_beat_v2_A

# Regime B -- Lumina (practical utility)
uv run python -m eval.clinvar.run --regime B \\
    --model-family lumina --model-version beat-v2 \\
    --checkpoint-path outputs/beat_v2/best_checkpoint.pt \\
    --fasta-path data/hg38/hg38.fa \\
    --phylo100-bw-path data/phylo/hg38.phyloP100way.bw \\
    --phylo470-bw-path data/phylo/hg38.phyloP470way.bw \\
    --gtf-path data/gencode/gencode.v38.annotation.gtf.gz \\
    --output-dir outputs/clinvar/lumina_beat_v2_B

# Regime A -- NTv3 8M
python -m eval.clinvar.run --regime A \\
    --model-family ntv3 --model-version 8M_pre \\
    --fasta-path data/hg38/hg38.fa \\
    --output-dir outputs/clinvar/ntv3_8m_A

# Regime A -- Caduceus
python -m eval.clinvar.run --regime A \\
    --model-family caduceus --model-version caduceus-ph \\
    --fasta-path data/hg38/hg38.fa \\
    --output-dir outputs/clinvar/caduceus_ph_A

# Regime A -- DNABERT-2
python -m eval.clinvar.run --regime A \\
    --model-family dnabert2 --model-version 117M \\
    --fasta-path data/hg38/hg38.fa \\
    --output-dir outputs/clinvar/dnabert2_117m_A

# Multi-GPU (DDP auto-detected from torchrun)
torchrun --nproc_per_node 8 -m eval.clinvar.run \\
    --regime A --model-family lumina --model-version beat-v2 \\
    --checkpoint-path outputs/beat_v2/best_checkpoint.pt \\
    --fasta-path data/hg38/hg38.fa \\
    --output-dir outputs/clinvar/lumina_beat_v2_A
"""

from __future__ import annotations

import argparse
import logging

from eval.clinvar.adapters import normalize_finetune_model_family
from eval.clinvar.config import FineTuneConfig
from eval.clinvar.train import run_finetune

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> FineTuneConfig:
    p = argparse.ArgumentParser(
        description="ClinVar fine-tuning with two evaluation regimes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Regime
    p.add_argument("--regime", choices=["A", "B"], default="A",
                    help="A = embeddings only, B = embeddings + bio features")

    # Model
    p.add_argument(
        "--model-family",
        required=True,
        # beat-v11 aliases mirror normalize_finetune_model_family (adapters.py); argparse's
        # choices is a separate gate that must also admit them or dispatch is never reached.
        choices=[
            "lumina", "ntv3", "caduceus", "dnabert2", "dnabert-2",
            "beat-v11", "beat_v11", "beat-v11-bioprime", "beat_v11_bioprime", "beatv11",
        ],
    )
    p.add_argument("--model-version", required=True)
    p.add_argument("--checkpoint-path", default=None)

    # Data
    p.add_argument("--dataset-path", default=FineTuneConfig.dataset_path)
    p.add_argument("--fasta-path", default=FineTuneConfig.fasta_path)
    p.add_argument("--phylo100-bw-path", default=None)
    p.add_argument("--phylo470-bw-path", default=None)
    p.add_argument("--gtf-path", default=None)
    p.add_argument("--context-size", type=int, default=4096)
    p.add_argument("--cache-dir", default=None)

    # Architecture (shared across regimes for fairness)
    p.add_argument("--proj-dim", type=int, default=256)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--head-dropout", type=float, default=0.2)
    p.add_argument(
        "--head-type",
        choices=[
            "regime_a",
            "regime_a_v7",
            "regime_a_v8",
            "allele_scorer",
            "regime_a_plus_features",
            "regime_a_bounded_regional",
        ],
        default=None,
    )

    # Regime A backbone-native features (Lumina beat-v6/v8; silently ignored elsewhere)
    p.add_argument(
        "--native-feature-heads",
        nargs="*",
        default=None,
        choices=[
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
        ],
        help=(
            "Pretrained heads whose variant-position outputs form Regime A's "
            "variant_repr on supported Lumina backbones. Default keeps all four "
            "beat-v6 heads (A2); pass only "
            "'mutation_effect' for A1; pass 'none' to force the legacy two-pass path. "
            "Optional extras: 'mlm_logit_ratio' adds the log-odds alt/ref at the site, "
            "'is_snv' adds a binary flag to disambiguate zeroed SNV-only slots."
        ),
    )
    p.add_argument(
        "--explicit-feature-columns",
        nargs="*",
        default=[],
        help=(
            "Scalar dataset columns or engineered feature names to concatenate "
            "with Regime A embeddings. Used by M6 via --head-type "
            "regime_a_plus_features or M5_v2 via --head-type "
            "regime_a_bounded_regional. Supported engineered names include "
            "log10_af_abraom, log10_af_gnomad, af_delta, af_abs_delta, "
            "af_ratio_log10, af_abraom_missing, af_gnomad_missing, and "
            "specificity_missing."
        ),
    )
    p.add_argument(
        "--init-finetuned-checkpoint-path",
        default=None,
        help="Prior ClinVar checkpoint used to initialize A_path before adapter fusion.",
    )
    p.add_argument(
        "--fusion-mode",
        choices=["none", "static_lora", "dynamic_lora"],
        default="none",
        help=(
            "Regional adapter fusion mode. static_lora learns layer-wise static gates; "
            "dynamic_lora learns per-example hidden-state gates over frozen adapters."
        ),
    )
    p.add_argument(
        "--fusion-adapter-paths",
        nargs="*",
        default=[],
        help="Population adapter checkpoint paths used by LoRA fusion.",
    )
    p.add_argument(
        "--fusion-adapter-names",
        nargs="*",
        default=[],
        help="Names matching --fusion-adapter-paths, e.g. abraom gnomad.",
    )
    p.add_argument(
        "--freeze-backbone-for-fusion",
        action="store_true",
        help="Train only fusion gate parameters plus the ClinVar head.",
    )
    p.add_argument("--fusion-gate-hidden-dim", type=int, default=FineTuneConfig.fusion_gate_hidden_dim)
    p.add_argument("--fusion-gate-dropout", type=float, default=FineTuneConfig.fusion_gate_dropout)

    # LoRA (identical across model families)
    p.add_argument("--lora-rank", type=int, default=4)
    p.add_argument("--lora-alpha", type=float, default=8.0)
    p.add_argument("--lora-dropout", type=float, default=0.1)

    # Optimizer
    p.add_argument("--lr-backbone", type=float, default=5e-6)
    p.add_argument("--lr-head", type=float, default=5e-4)
    p.add_argument("--wd-backbone", type=float, default=0.01)
    p.add_argument("--wd-head", type=float, default=1e-4)

    # Training
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum-steps", type=int, default=16)
    p.add_argument("--max-epochs", type=int, default=5)
    p.add_argument("--warmup-fraction", type=float, default=0.10)
    p.add_argument("--grad-clip", type=float, default=0.5)
    p.add_argument("--freeze-backbone-steps", type=int, default=100)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)

    # Loss
    p.add_argument("--loss-type", choices=["bce", "focal"], default="focal")
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--pos-weight", default="1.0")
    p.add_argument("--pairwise-rank-weight", type=float, default=None)
    p.add_argument("--pairwise-rank-margin", type=float, default=FineTuneConfig.pairwise_rank_margin)
    p.add_argument("--swap-consistency-weight", type=float, default=None)
    p.add_argument("--swap-consistency-margin", type=float, default=FineTuneConfig.swap_consistency_margin)

    # System
    p.add_argument("--device", default="cuda")
    p.add_argument("--precision", choices=("auto", "bf16", "fp32"), default=FineTuneConfig.precision)
    p.add_argument("--allow-tf32", action="store_true", default=True)
    p.add_argument("--no-tf32", dest="allow_tf32", action="store_false")
    p.add_argument("--num-workers", type=int, default=0)

    # Output
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")

    # W&B
    p.add_argument("--wandb-enabled", action="store_true")
    p.add_argument("--wandb-project", default="lumina-clinvar")
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--wandb-tags", nargs="*", default=[])

    args = p.parse_args(argv)

    # Parse pos_weight
    pos_weight: str | float
    try:
        pos_weight = float(args.pos_weight)
    except ValueError:
        pos_weight = args.pos_weight

    model_family = normalize_finetune_model_family(args.model_family)
    head_type = args.head_type
    is_lumina_beat_v8_regime_a = model_family == "lumina" and args.model_version == "beat-v8" and args.regime == "A"
    if head_type is None:
        if is_lumina_beat_v8_regime_a:
            head_type = "regime_a_v8"
        elif model_family == "lumina" and args.model_version == "beat-v7":
            head_type = "regime_a_v7"
        else:
            head_type = "regime_a"

    native_feature_heads_kwargs: dict[str, list[str]] = {}
    if args.native_feature_heads is not None:
        native_feature_heads_kwargs["native_feature_heads"] = list(args.native_feature_heads)
    elif is_lumina_beat_v8_regime_a:
        native_feature_heads_kwargs["native_feature_heads"] = [
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

    config = FineTuneConfig(
        regime=args.regime,
        model_family=model_family,
        model_version=args.model_version,
        checkpoint_path=args.checkpoint_path,
        dataset_path=args.dataset_path,
        fasta_path=args.fasta_path,
        phylo100_bw_path=args.phylo100_bw_path,
        phylo470_bw_path=args.phylo470_bw_path,
        gtf_path=args.gtf_path,
        context_size=args.context_size,
        cache_dir=args.cache_dir,
        proj_dim=args.proj_dim,
        hidden_dim=args.hidden_dim,
        head_dropout=args.head_dropout,
        head_type=head_type,
        explicit_feature_columns=list(args.explicit_feature_columns or []),
        init_finetuned_checkpoint_path=args.init_finetuned_checkpoint_path,
        fusion_mode=args.fusion_mode,
        fusion_adapter_paths=list(args.fusion_adapter_paths or []),
        fusion_adapter_names=list(args.fusion_adapter_names or []),
        freeze_backbone_for_fusion=bool(args.freeze_backbone_for_fusion),
        fusion_gate_hidden_dim=args.fusion_gate_hidden_dim,
        fusion_gate_dropout=args.fusion_gate_dropout,
        **native_feature_heads_kwargs,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lr_backbone=args.lr_backbone,
        lr_head=args.lr_head,
        wd_backbone=args.wd_backbone,
        wd_head=args.wd_head,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        max_epochs=args.max_epochs,
        warmup_fraction=args.warmup_fraction,
        grad_clip=args.grad_clip,
        freeze_backbone_steps=args.freeze_backbone_steps,
        val_fraction=args.val_fraction,
        seed=args.seed,
        loss_type=args.loss_type,
        focal_gamma=args.focal_gamma,
        pos_weight=pos_weight,
        pairwise_rank_weight=(
            0.2
            if args.pairwise_rank_weight is None
            and is_lumina_beat_v8_regime_a
            else (
                FineTuneConfig.pairwise_rank_weight
                if args.pairwise_rank_weight is None
                else args.pairwise_rank_weight
            )
        ),
        pairwise_rank_margin=args.pairwise_rank_margin,
        swap_consistency_weight=(
            0.1
            if args.swap_consistency_weight is None
            and is_lumina_beat_v8_regime_a
            else (
                FineTuneConfig.swap_consistency_weight
                if args.swap_consistency_weight is None
                else args.swap_consistency_weight
            )
        ),
        swap_consistency_margin=args.swap_consistency_margin,
        device=args.device,
        precision=args.precision,
        allow_tf32=args.allow_tf32,
        num_workers=args.num_workers,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        wandb_enabled=args.wandb_enabled,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        wandb_tags=args.wandb_tags or [],
    )
    if args.pairwise_rank_weight is not None:
        config.pairwise_rank_weight = args.pairwise_rank_weight
    if args.swap_consistency_weight is not None:
        config.swap_consistency_weight = args.swap_consistency_weight
    return config


def main(argv: list[str] | None = None) -> None:
    config = parse_args(argv)
    log.info(
        "Starting ClinVar fine-tuning: regime=%s model=%s/%s",
        config.regime, config.model_family, config.model_version,
    )
    results = run_finetune(config)
    if results:
        log.info(
            "Regime %s | %s/%s | MCC=%.4f | AUROC=%.4f | AUPRC=%.4f",
            results.get("regime", "?"),
            results.get("model_family", "?"),
            results.get("model_version", "?"),
            results.get("mcc", 0),
            results.get("auroc", 0),
            results.get("auprc", 0),
        )


if __name__ == "__main__":
    main()
