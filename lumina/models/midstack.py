from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from lumina.constants import (
    NUM_COUNTERFACTUAL_EFFECT_CLASSES,
    NUM_REGION_CLASSES,
    VOCAB_SIZE,
)
from lumina.models.backbone import (
    BidiMambaMidBlock,
    SparseGlobalAttention,
)
from lumina.models.local_attn import LocalWindowAttention


@dataclass
class LuminaBackboneConfig:
    vocab_size: int = VOCAB_SIZE
    l_max: int = 32768

    d_model: int = 256
    d_pure: int = 64
    d_full: int = 320
    # d_state == Mamba-3 headdim_qk; must be a MIMO-supported value (128) — see _base.yaml.
    d_state: int = 128
    d_embed: int = 256

    expand: int = 2
    headdim: int = 64
    ngroups: int = 1
    # B1 (multi-timescale Mamba): 1.0 => every d_state channel rotates (dephases at long range); 0.5 =>
    # half the channels are zero-angle pure-real integrators (stable long-range/DC carriers). Asserted
    # {0.5,1.0} in the kernel. 0.5 changes in_proj out-dim (num_rope_angles halves) => from-scratch only.
    rope_fraction: float = 1.0
    chunk_size: int = 64
    is_mimo: bool = False
    mimo_rank: int = 4
    is_outproj_norm: bool = True
    dropout: float = 0.05
    activation_checkpointing: bool = True
    checkpoint_use_reentrant: bool = False

    # --- Phase-1 arch levers (Idea B2 + Idea F2/F3); all default to the pre-Phase-1 behavior ------------
    # B2 * dt-bias banding: re-init the first `dt_band_heads` heads of each direction's Mamba3 toward a
    # long-memory timescale. In Mamba's selectivity convention a SMALL dt persists state / ignores input
    # (long memory); large dt resets to the current input (short memory). 0 => off (untouched log-uniform
    # init). A prior/nudge only — dt_bias is learnable + the decay is data-dependent — so pair with Idea A.
    dt_band_heads: int = 0
    dt_band_dt: float = 0.002  # target discrete step for the band (small = long memory); softplus^-1 => dt_bias.
    # F2 * RMSNorm (no mean-subtraction => RC-safe on mid-block hidden states) in place of the per-direction
    # LayerNorm pair. Default False => LayerNorm (byte-identical to pre-Phase-1). NB: distinct from the
    # REJECTED pooled-embedding LayerNorm (that broke rc_cos); this norms per-token mid-stack states.
    mid_block_rmsnorm_enabled: bool = False
    # F3 * torch.compile the mid-block forward at BLOCK granularity (Inductor fuses the norm/flip/add chain —
    # the launch-bound win; norm/elementwise is 44.7% of step CPU). CUDA-only + gated; graph-breaks cleanly
    # around the Mamba custom kernel. Compiled per block, not per step, to respect DDP static_graph-off +
    # find_unused_parameters=True + the multi-pass forward. Default False.
    compile_mid_block_enabled: bool = False

    position_encoding: str = "sinusoidal"
    position_embedding: str | None = None
    use_rope_in_attention: bool = True

    conv_stem_kernels: list[int] = field(default_factory=lambda: [3, 7, 15])
    purity_branch: bool = True
    downsample_factor: int = 4
    downsample_factor_default: int = 4
    downsample_factor_fallback: int = 8
    n_downsample_stages: int | None = None

    n_mid_bidi_mamba: int = 12
    n_mid_blocks: int | None = None
    n_local_attention: int = 4
    n_sparse_global_attention: int = 2
    local_attention_window_mid_tokens: int = 256
    local_attention_window: int | None = None
    local_attention_heads: int = 4
    # Shakeout S3 (gnorm hypothesis H3): L2-normalize q,k (learnable per-head temperature) before the
    # local-attention dot product to bound attention-logit growth. Default False => bit-identical.
    local_attention_qk_norm: bool = False
    sparse_global_stride: int = 16
    sparse_global_heads: int = 4
    # Tier 2A: 'strided' (z[::stride], current) or 'mean_pool' (pool each stride-block → every
    # position contributes to a global key). A trained operator; default preserves current behavior.
    sparse_global_anchor_mode: str = "strided"
    # Tier 2B: learned global register/memory tokens prepended to the mid-stack (0 = off, identical).
    n_register_tokens: int = 0

    use_variant_token_residual: bool = True
    variant_residual_gamma: float = 0.5

    num_region_classes: int = NUM_REGION_CLASSES
    num_splice_classes: int = 5
    num_counterfactual_effect_classes: int = NUM_COUNTERFACTUAL_EFFECT_CLASSES
    num_regulatory_tracks: int = 20
    num_conservation_targets: int = 3  # phyloP100 (idx0) + Zoonomia-241 (idx1) + phyloP470 (idx2)
    heads: dict[str, bool] = field(default_factory=dict)

    def resolved_downsample_stages(self) -> int:
        if self.n_downsample_stages is not None:
            return int(self.n_downsample_stages)
        factor = int(self.downsample_factor or self.downsample_factor_default)
        if factor <= 0 or factor & (factor - 1):
            raise ValueError(f"Lumina downsample_factor must be a positive power of 2, got {factor}.")
        return int(math.log2(factor))

    def resolved_position_encoding(self) -> str:
        return self.position_embedding or self.position_encoding

    def resolved_local_attention_window(self) -> int:
        return int(self.local_attention_window or self.local_attention_window_mid_tokens)

    def resolved_mid_mamba_blocks(self) -> int:
        return int(self.n_mid_blocks or self.n_mid_bidi_mamba)


def _evenly_spaced_points(total: int, count: int) -> set[int]:
    if total <= 0 or count <= 0:
        return set()
    points: set[int] = set()
    for index in range(1, count + 1):
        points.add(max(1, min(total, round(index * total / (count + 1)))))
    return points


class LuminaMidStack(nn.Module):
    def __init__(self, cfg: LuminaBackboneConfig) -> None:
        super().__init__()
        # Tier 2B: learned global register/memory tokens prepended to the mid-stack as a shared
        # scratchpad that every position can read/write via attention. n_register_tokens=0 → None →
        # the forward is bit-identical to the pre-2B model.
        self.n_register_tokens = int(cfg.n_register_tokens)
        self.register_tokens: nn.Parameter | None = None
        if self.n_register_tokens > 0:
            self.register_tokens = nn.Parameter(torch.randn(self.n_register_tokens, cfg.d_model) * 0.02)
        self.layers = nn.ModuleList()
        self.layer_kinds: list[str] = []
        local_points = _evenly_spaced_points(cfg.resolved_mid_mamba_blocks(), cfg.n_local_attention)
        sparse_points = _evenly_spaced_points(cfg.resolved_mid_mamba_blocks(), cfg.n_sparse_global_attention)
        for block_index in range(1, cfg.resolved_mid_mamba_blocks() + 1):
            self.layers.append(BidiMambaMidBlock(cast(Any, cfg)))
            self.layer_kinds.append("mamba")
            if block_index in local_points:
                self.layers.append(
                    nn.ModuleDict(
                        {
                            "norm": nn.LayerNorm(cfg.d_model),
                            "attn": LocalWindowAttention(
                                d_model=cfg.d_model,
                                n_heads=cfg.local_attention_heads,
                                window=cfg.resolved_local_attention_window(),
                                dropout=cfg.dropout,
                                qk_norm=cfg.local_attention_qk_norm,
                            ),
                        }
                    )
                )
                self.layer_kinds.append("local")
            if block_index in sparse_points:
                self.layers.append(
                    SparseGlobalAttention(
                        d_model=cfg.d_model,
                        n_heads=cfg.sparse_global_heads,
                        stride=cfg.sparse_global_stride,
                        dropout=cfg.dropout,
                        anchor_mode=cfg.sparse_global_anchor_mode,
                    )
                )
                self.layer_kinds.append("sparse")

    def forward(self, x: torch.Tensor, edit_mid_mask: torch.Tensor | None = None) -> torch.Tensor:
        n_reg = self.n_register_tokens
        if self.register_tokens is not None:
            # Prepend registers and shift the edit-anchor mask (registers are never edit anchors).
            # Cast to x's dtype first: under bf16 autocast x is bf16, but the fp32 register parameter
            # would promote the cat result to fp32 and break dtype-matching downstream (sparse-attn
            # edit-anchor index_put). The cast stays autograd-connected to the fp32 master param.
            reg = self.register_tokens.to(dtype=x.dtype).unsqueeze(0).expand(x.shape[0], -1, -1)
            x = torch.cat([reg, x], dim=1)
            if edit_mid_mask is not None:
                edit_mid_mask = F.pad(edit_mid_mask, (n_reg, 0), value=False)
        for kind, layer in zip(self.layer_kinds, self.layers, strict=True):
            if kind == "mamba":
                x = layer(x)
            elif kind == "local":
                layer_dict = cast(nn.ModuleDict, layer)
                norm = cast(nn.LayerNorm, layer_dict["norm"])
                attn = cast(LocalWindowAttention, layer_dict["attn"])
                x = x + attn(norm(x), None)
            elif kind == "sparse":
                x = layer(x, edit_mid_mask=edit_mid_mask)
        if self.register_tokens is not None:
            x = x[:, n_reg:]  # strip registers → mid_hidden_state keeps its original length
        return x
