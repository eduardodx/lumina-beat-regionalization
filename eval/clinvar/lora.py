"""Lightweight LoRA for ClinVar fine-tuning.

Applied uniformly across all backbone linear layers (except output heads)
with identical rank across model families to ensure a fair comparison.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)

_EXCLUDE_PATTERNS = frozenset({
    "lm_head", "cls", "classifier", "mlm_head",
    "phylo100_head", "phylo470_head", "structure_head",
    "region_head", "global_proj", "aa_head", "codon_phylo_head",
    "phylo100_subst_head", "zoo241_subst_head", "splice_class_head", "splice_distance_head",
    "gnomad_af_head", "gnomad_observed_head", "counterfactual_snv_head", "edit_localize_head",
    "regulatory_head", "sequence_embedding_head",
    # --- Beat-v11 BioPrime (renamed heads vs v10 + new MoE). Without these the substring
    # match misses them and apply_lora wraps them by mistake. The v11 heads already covered
    # above via substring: mlm_head, region_head, splice_class_head, splice_distance_head,
    # counterfactual_snv_head, regulatory_head. Newly needed:
    "population_af_head", "population_observed_head",     # v11 renamed the gnomAD prior heads
    "conservation_scalar_head", "conservation_bin_head",  # v11 renamed the phylo/conservation heads
    "missense_severity_head",                             # new in v11 (ESM-2 severity)
    "hic_left", "hic_right", "cell_film",                 # Hi-C / cell-FiLM readout (if built)
    # MoE frozen by default = v10-parity LoRA surface (v10 had no MoE): excluding the whole
    # MoE keeps the trainable surface == Mamba in/out_proj + attention, like Pedro's v10 run.
    # ABLATION "LoRA the experts": drop ".experts." from this set (keep ".router").
    ".router", ".experts.",
})


class LoRALinear(nn.Module):
    """Linear layer with frozen base weights and trainable low-rank delta."""

    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.base = base
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

        in_features = base.in_features
        out_features = base.out_features
        self.scaling = alpha / rank
        param_kwargs = {
            "device": base.weight.device,
            "dtype": base.weight.dtype,
        }

        self.lora_a = nn.Parameter(torch.empty(rank, in_features, **param_kwargs))
        self.lora_b = nn.Parameter(torch.zeros(out_features, rank, **param_kwargs))
        nn.init.kaiming_uniform_(self.lora_a)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    @property
    def weight(self) -> torch.Tensor:
        """Expose the wrapped weight for modules that inspect Linear metadata."""
        return self.base.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.base.bias

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = F.linear(self.dropout(x), self.lora_a) @ self.lora_b.T
        return base_out + lora_out * self.scaling


@dataclass(frozen=True)
class LoRASummary:
    """Summary of LoRA application for reproducibility logging."""

    rank: int
    alpha: float
    dropout: float
    module_names: tuple[str, ...]
    total_params: int

    @property
    def module_count(self) -> int:
        return len(self.module_names)


def _should_exclude(name: str) -> bool:
    return any(pattern in name for pattern in _EXCLUDE_PATTERNS)


def apply_lora(
    backbone: nn.Module,
    rank: int = 4,
    alpha: float = 8.0,
    dropout: float = 0.1,
) -> LoRASummary:
    """Replace eligible nn.Linear layers with LoRA-wrapped versions.

    All model families receive the same rank, alpha, and dropout for fair
    comparison.  Only output/prediction heads are excluded.
    """
    targets: list[str] = []
    for name, module in backbone.named_modules():
        if isinstance(module, nn.Linear) and not _should_exclude(name):
            targets.append(name)

    for name in targets:
        parts = name.split(".")
        parent = backbone
        for part in parts[:-1]:
            parent = getattr(parent, part)
        original = getattr(parent, parts[-1])
        wrapped = LoRALinear(original, rank=rank, alpha=alpha, dropout=dropout)
        setattr(parent, parts[-1], wrapped)

    total = sum(
        p.numel() for n, p in backbone.named_parameters()
        if "lora_" in n and p.requires_grad
    )
    log.info("LoRA applied to %d modules (%d trainable params, rank=%d)", len(targets), total, rank)
    return LoRASummary(
        rank=rank, alpha=alpha, dropout=dropout,
        module_names=tuple(targets), total_params=total,
    )


def enable_layernorm_training(backbone: nn.Module) -> list[str]:
    """Unfreeze all LayerNorm parameters in the backbone."""
    unfrozen: list[str] = []
    for name, module in backbone.named_modules():
        if isinstance(module, nn.LayerNorm):
            for pname, param in module.named_parameters():
                param.requires_grad = True
                unfrozen.append(f"{name}.{pname}")
    log.info("Unfrozen %d LayerNorm parameters for lora_plus_norm", len(unfrozen))
    return unfrozen


_NATIVE_HEAD_PARAM_PREFIXES: dict[str, str] = {
    "mutation_effect": "mutation_effect_head.",
    "aa": "aa_head.",
    "codon_phylo": "codon_phylo_head.",
    "phylo470": "phylo470_head.",
    "mlm_logit_ratio": "mlm_head.",
    "allele_repr": "allele_scorer_head.",
    "allele_effect_logits": "allele_scorer_head.",
    "allele_severity_score": "allele_scorer_head.",
    "allele_swap_severity_score": "allele_scorer_head.",
    "allele_far_distance": "allele_scorer_head.",
    "allele_site_delta": "allele_scorer_head.",
    "allele_local_delta": "allele_scorer_head.",
}


def _pin_eval_mode(module: nn.Module) -> None:
    """Put ``module`` in eval mode and override ``.train()`` to keep it there.

    Dropout/LayerNorm inside pretrained heads used as frozen feature extractors
    must not react to the outer ``model.train()`` propagation; otherwise their
    outputs become stochastic on every forward.
    """
    module.eval()

    def _pinned_train(mode: bool = True) -> nn.Module:
        _ = mode
        return module

    module.train = _pinned_train  # type: ignore[method-assign]


def freeze_native_feature_heads(
    backbone: nn.Module,
    selection: tuple[str, ...] | list[str],
) -> list[str]:
    """Freeze pretrained heads used as feature extractors on the native Regime A path.

    Necessary because these heads are excluded from LoRA wrapping but
    otherwise remain trainable; without this call they leak into the
    backbone optimizer group.  Also pins each resolved head in eval mode
    so dropout stays silent during ``model.train()``.  Synthetic heads
    (e.g. ``is_snv``) have no backbone attribute and are skipped.
    """
    from eval.clinvar.adapters import NATIVE_HEAD_ATTRIBUTES

    prefixes = tuple(
        _NATIVE_HEAD_PARAM_PREFIXES[h] for h in selection if h in _NATIVE_HEAD_PARAM_PREFIXES
    )
    frozen: list[str] = []
    for name, param in backbone.named_parameters():
        if name.startswith(prefixes):
            param.requires_grad = False
            frozen.append(name)

    locked: list[str] = []
    for head_name in selection:
        attr = NATIVE_HEAD_ATTRIBUTES.get(head_name)
        if attr is None:
            continue
        module = getattr(backbone, attr, None)
        if isinstance(module, nn.Module):
            _pin_eval_mode(module)
            locked.append(attr)

    log.info(
        "Froze %d native-head parameters and pinned %d modules in eval mode "
        "(selection=%s, locked=%s) for Regime A feature extraction",
        len(frozen), len(locked), list(selection), locked,
    )
    return frozen


def count_trainable_parameters(model: nn.Module) -> dict[str, int]:
    """Report trainable parameter counts by category."""
    lora_params = 0
    fusion_params = 0
    norm_params = 0
    head_params = 0
    other_params = 0

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        n = param.numel()
        if "lora_" in name:
            lora_params += n
        elif name.endswith("adapter_logits") or ".gate." in name:
            fusion_params += n
        elif "head." in name:
            head_params += n
        elif any(ln in name for ln in ("layer_norm", "layernorm", "norm.")):
            norm_params += n
        else:
            other_params += n

    return {
        "lora": lora_params,
        "fusion": fusion_params,
        "layernorm": norm_params,
        "head": head_params,
        "other": other_params,
        "total": lora_params + fusion_params + norm_params + head_params + other_params,
    }
