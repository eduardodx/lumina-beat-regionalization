"""Beat-v11 BioPrime — standalone model.

Self-contained port of the Lumina ``beat_v11_bioprime`` (BioPrime) architecture for
inference and review:

    multi-kernel conv stem + purity branch -> 4x conv-downsample hourglass
    -> bidirectional Mamba mid-stack (interleaved local-window + sparse-global attention,
       optional learned register tokens)
    -> gated conv-upsample with full-resolution skips
    -> optional final trunk RMSNorm on the concatenated (trunk + purity) hidden state

plus the per-position BioPrime head set (MLM, conservation scalar + bins + variant-delta,
splice class/distance, region, counterfactual ref/alt, ESM-2 missense-severity, weak
population prior [AF + observed]) and the multi-scale build-out heads (ENCODE 50-track
regulatory head + within-window Hi-C contact band with cell-type FiLM conditioning).

This module has NO imports from the training repository. The backbone, MoE sublayer,
mid-stack, and model class bodies below are reproduced verbatim from the training sources
(``src/models/backbone.py``, ``src/models/moe.py``, ``src/models/midstack.py``,
``src/models/bioprime/model.py``); only the imports are rewired to this package, and the
training-only loss stack that the model's ``__init__`` touched is replaced by the minimal
``loss_ema`` extract (which reproduces the checkpoint's ``loss_ema.*`` buffers).

See ``TECHNICAL.md`` for the architecture and output contract.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from .constants import (
    NUM_COUNTERFACTUAL_EFFECT_CLASSES,
    NUM_REGION_CLASSES,
    PAD_ID,
    SNV_BASES,
    VOCAB_SIZE,
)
from .local_attn import LocalWindowAttention
from .loss_ema import BIOPRIME_LOSS_NAMES, LossEMANormalizer

try:
    _mamba_ssm = importlib.import_module("mamba_ssm")
except ImportError:
    _Mamba3 = None
else:
    _Mamba3 = getattr(_mamba_ssm, "Mamba3", None)


def require_mamba3() -> type:
    if _Mamba3 is None:
        raise ImportError(
            "Mamba3 is not available in the installed mamba-ssm package. "
            "Install the latest mamba-ssm from source: "
            "MAMBA_FORCE_BUILD=TRUE pip install git+https://github.com/state-spaces/mamba.git --no-build-isolation"
        )
    return _Mamba3


def resolve_chunk_size(cfg: Any) -> int:
    """Mamba3 MIMO with bf16 needs chunk_size = 64 / mimo_rank."""
    if cfg.is_mimo and cfg.chunk_size == 64:
        return max(1, 64 // cfg.mimo_rank)
    return cfg.chunk_size


# --- backbone (src/models/backbone.py) ---
# ====================================================================================================
class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, l_max: int, d_model: int) -> None:
        super().__init__()
        position = torch.arange(l_max, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(l_max, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("weight", pe, persistent=False)

    def forward(self, seq_len: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        weight = cast(torch.Tensor, self.weight)
        return weight[:seq_len].to(device=device, dtype=dtype)


class MultiKernelStem(nn.Module):
    def __init__(self, d_model: int, d_pure: int, kernels: Sequence[int]) -> None:
        super().__init__()
        if not kernels:
            raise ValueError("Beat-v10 stem requires at least one convolution kernel.")
        self.branches = nn.ModuleList(
            [
                nn.Conv1d(
                    d_model,
                    d_model,
                    kernel_size=int(kernel),
                    padding=int(kernel) // 2,
                    groups=d_model,
                )
                for kernel in kernels
            ]
        )
        self.pointwise = nn.Conv1d(d_model * len(kernels), d_model, kernel_size=1)
        self.norm = nn.LayerNorm(d_model)
        self.purity = nn.Linear(d_model, d_pure)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = x.shape[1]
        x_t = x.transpose(1, 2)
        branch_outputs: list[torch.Tensor] = []
        for branch in self.branches:
            y = branch(x_t)
            if y.shape[-1] > seq_len:
                y = y[..., :seq_len]
            elif y.shape[-1] < seq_len:
                y = F.pad(y, (0, seq_len - y.shape[-1]))
            branch_outputs.append(y)
        mixed = self.pointwise(torch.cat(branch_outputs, dim=1)).transpose(1, 2)
        h_stem = self.norm(F.gelu(mixed))
        h_pure = self.purity(x)
        return h_stem, h_pure


class DownStage(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.pre_norm = nn.LayerNorm(d_model)
        self.down = nn.Conv1d(d_model, d_model, kernel_size=4, stride=2, padding=1)
        self.post_norm = nn.LayerNorm(d_model)
        self.refine = nn.Conv1d(d_model, d_model, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pre_norm(x).transpose(1, 2)
        y = F.gelu(self.down(y)).transpose(1, 2)
        y = self.post_norm(y).transpose(1, 2)
        return self.refine(y).transpose(1, 2)


class UpStage(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose1d(d_model, d_model, kernel_size=4, stride=2, padding=1)
        self.norm = nn.LayerNorm(d_model)
        self.gate = nn.Linear(d_model, d_model)
        self.refine = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)

    @staticmethod
    def _match_length(x: torch.Tensor, seq_len: int) -> torch.Tensor:
        if x.shape[1] > seq_len:
            return x[:, :seq_len]
        if x.shape[1] < seq_len:
            return F.pad(x, (0, 0, 0, seq_len - x.shape[1]))
        return x

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        y = self.up(x.transpose(1, 2)).transpose(1, 2)
        y = self._match_length(y, skip.shape[1])
        y = F.gelu(self.norm(y))
        y = y + torch.sigmoid(self.gate(skip)) * skip
        return self.refine(y.transpose(1, 2)).transpose(1, 2)


class BidiMambaMidBlock(nn.Module):
    def __init__(self, cfg: Any) -> None:
        super().__init__()
        Mamba3 = require_mamba3()
        chunk_size = resolve_chunk_size(cfg)
        kwargs = dict(
            d_model=cfg.d_model,
            d_state=cfg.d_state,
            expand=cfg.expand,
            headdim=cfg.headdim,
            ngroups=cfg.ngroups,
            rope_fraction=cfg.rope_fraction,
            chunk_size=chunk_size,
            is_mimo=cfg.is_mimo,
            mimo_rank=cfg.mimo_rank,
            is_outproj_norm=cfg.is_outproj_norm,
        )
        # Spec §4.4 calls for a single post-fusion LayerNorm on raw x. We instead pre-norm
        # each direction: Mamba3 with is_outproj_norm=True already normalizes branch outputs,
        # so a post-fusion norm would be duplicative, and pre-norm-on-input is the bf16-stable
        # convention at depth 8. F2: RMSNorm (no mean-subtraction) when enabled — RC-safe here
        # (per-token mid-stack states) and one reduction cheaper than LayerNorm.
        norm_cls = nn.RMSNorm if bool(getattr(cfg, "mid_block_rmsnorm_enabled", False)) else nn.LayerNorm
        self.norm_fwd = norm_cls(cfg.d_model)
        self.norm_bwd = norm_cls(cfg.d_model)
        self.fwd = Mamba3(**kwargs)
        self.bwd = Mamba3(**kwargs)
        # B2: bias a band of heads toward long memory (small dt). Post-construction param surgery — no
        # kernel touch; a no-op on backends without dt_bias (e.g. the FakeMamba CPU test double).
        dt_band_heads = int(getattr(cfg, "dt_band_heads", 0) or 0)
        if dt_band_heads > 0:
            self._apply_dt_band(dt_band_heads, float(getattr(cfg, "dt_band_dt", 0.002)))
        self.activation_checkpointing = cfg.activation_checkpointing
        self.checkpoint_use_reentrant = cfg.checkpoint_use_reentrant
        # F3: optional block-level torch.compile (CUDA-only). Fuses the norm/flip/add chain; the Mamba
        # custom kernel graph-breaks cleanly (fullgraph=False default). Per-block (not per-step) so it
        # respects DDP static_graph-off + find_unused_parameters=True + the multi-pass forward.
        self._mid_compiled: Any = None
        if bool(getattr(cfg, "compile_mid_block_enabled", False)) and torch.cuda.is_available():
            self._mid_compiled = torch.compile(self._mid_forward)

    def _apply_dt_band(self, band: int, target_dt: float) -> None:
        """Re-init the first ``band`` heads' dt_bias (each direction) toward ``target_dt`` (long memory).
        Mamba parameterizes dt = softplus(dt_bias), so dt_bias = target_dt + log(-expm1(-target_dt)) — the
        module's own init transform. No-op if the backend exposes no dt_bias (the FakeMamba test double)."""
        target_dt = max(1e-4, float(target_dt))
        dt_bias_value = target_dt + math.log(-math.expm1(-target_dt))
        for module in (self.fwd, self.bwd):
            dt_bias = getattr(module, "dt_bias", None)
            if dt_bias is None:
                continue
            with torch.no_grad():
                dt_bias[: min(band, dt_bias.shape[0])] = dt_bias_value

    def _maybe_checkpoint(self, module: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if (
            not self.activation_checkpointing
            or not self.training
            or not torch.is_grad_enabled()
            or not x.requires_grad
        ):
            return module(x)
        return cast(torch.Tensor, activation_checkpoint(module, x, use_reentrant=self.checkpoint_use_reentrant))

    def _mid_forward(self, x: torch.Tensor) -> torch.Tensor:
        fwd = self._maybe_checkpoint(self.fwd, self.norm_fwd(x))
        bwd = torch.flip(self._maybe_checkpoint(self.bwd, self.norm_bwd(torch.flip(x, dims=[1]))), dims=[1])
        return x + fwd + bwd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._mid_compiled is not None:
            return cast(torch.Tensor, self._mid_compiled(x))
        return self._mid_forward(x)


class SparseGlobalAttention(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, stride: int, dropout: float, anchor_mode: str = "strided"
    ) -> None:
        super().__init__()
        self.stride = max(1, int(stride))
        if anchor_mode not in ("strided", "mean_pool"):
            raise ValueError(f"sparse_global_anchor_mode must be 'strided' or 'mean_pool', got {anchor_mode!r}.")
        self.anchor_mode = anchor_mode
        self.norm = nn.LayerNorm(d_model)
        self.strided_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.anchor_query_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.anchor_key_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def _global_keys(self, z: torch.Tensor) -> torch.Tensor:
        """Global key/value anchors. Tier 2A: 'mean_pool' pools each stride-sized block into one
        anchor so EVERY position contributes to a global key (the strided path uses only z[::stride],
        leaving 15/16 mid-tokens off the key side — the long-range broadcast bottleneck). Same anchor
        count → same attention compute."""
        if self.anchor_mode != "mean_pool" or self.stride <= 1:
            return z[:, :: self.stride]
        b, length, d = z.shape
        pad = (-length) % self.stride
        if pad:
            z = F.pad(z, (0, 0, 0, pad))  # pad the length axis up to a multiple of stride
        n_blocks = z.shape[1] // self.stride
        return z.view(b, n_blocks, self.stride, d).mean(dim=2)

    def forward(self, x: torch.Tensor, edit_mid_mask: torch.Tensor | None = None) -> torch.Tensor:
        z = self.norm(x)
        keys = self._global_keys(z)
        attended, _ = self.strided_attn(z, keys, keys, need_weights=False)
        out = x + self.dropout(attended)

        if edit_mid_mask is None or not torch.any(edit_mid_mask):
            # No edit anchors in this batch, so the anchor attentions below are
            # skipped. Tie their parameters into the autograd graph with a
            # zero-magnitude touch so DDP static_graph sees a constant set of
            # trainable parameters across iterations and ranks. A data-dependent
            # parameter set (anchor params used only on variant batches) desyncs
            # the reducer buckets across ranks and hangs the second all-reduce.
            # This leaves `out` numerically unchanged and consumes no RNG/dropout.
            anchor_touch = out.new_zeros(())
            for param in self.anchor_query_attn.parameters():
                anchor_touch = anchor_touch + param.sum()
            for param in self.anchor_key_attn.parameters():
                anchor_touch = anchor_touch + param.sum()
            return out + 0.0 * anchor_touch

        anchor_updates = torch.zeros_like(out)
        for batch_index in range(z.shape[0]):
            anchor_idx = torch.nonzero(edit_mid_mask[batch_index], as_tuple=False).flatten()
            if anchor_idx.numel() == 0:
                continue
            anchors = z[batch_index : batch_index + 1, anchor_idx]
            full = z[batch_index : batch_index + 1]
            anchor_out, _ = self.anchor_query_attn(anchors, full, full, need_weights=False)
            anchor_updates[batch_index, anchor_idx] = anchor_out.squeeze(0)
            broadcast, _ = self.anchor_key_attn(full, anchors, anchors, need_weights=False)
            anchor_updates[batch_index : batch_index + 1] += broadcast
        return out + self.dropout(anchor_updates)


# --- MoE sublayer (src/models/moe.py) ---
# ====================================================================================================
class SwiGLUExpert(nn.Module):
    """A single SwiGLU FFN expert: ``w_down(silu(w_gate(x)) * w_up(x))``."""

    def __init__(self, d_model: int, d_hidden: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_hidden, bias=False)
        self.w_up = nn.Linear(d_model, d_hidden, bias=False)
        self.w_down = nn.Linear(d_hidden, d_model, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(self.dropout(F.silu(self.w_gate(x)) * self.w_up(x)))


def resolve_moe_hidden(cfg: Any) -> int:
    hidden = int(getattr(cfg, "moe_expert_hidden", 0) or 0)
    return hidden if hidden > 0 else 2 * int(cfg.d_model)


class MoEMLPBlock(nn.Module):
    """Top-k SwiGLU MoE residual sublayer with an auxiliary load-balance loss."""

    def __init__(self, cfg: Any) -> None:
        super().__init__()
        d = int(cfg.d_model)
        self.num_experts = int(getattr(cfg, "moe_num_experts", 8))
        self.top_k = min(int(getattr(cfg, "moe_top_k", 2)), self.num_experts)
        hidden = resolve_moe_hidden(cfg)
        self.d_hidden = hidden
        norm_cls = nn.RMSNorm if str(getattr(cfg, "moe_norm", "rmsnorm")).lower() == "rmsnorm" else nn.LayerNorm
        self.norm = norm_cls(d)
        self.router = nn.Linear(d, self.num_experts, bias=False)
        dropout = float(getattr(cfg, "dropout", 0.0))
        self.experts = nn.ModuleList([SwiGLUExpert(d, hidden, dropout) for _ in range(self.num_experts)])
        self.router_jitter = float(getattr(cfg, "moe_router_jitter", 0.0))
        # Router z-loss (ST-MoE): penalize large router logits (logsumexp²) to keep routing well-conditioned
        # and resist collapse; folded into the aux term via the weight ratio so the loss layer's single
        # moe_aux_weight recovers ``aux_weight·balance + z_weight·z``. 0 => balance-only (plan default alpha).
        self._aux_weight = float(getattr(cfg, "moe_aux_weight", 0.01))
        self._z_weight = float(getattr(cfg, "moe_z_weight", 0.0))
        # Recomputed each forward (not buffers): read by the mid-stack.
        self.last_aux_loss: torch.Tensor | None = None
        self.last_expert_fraction: torch.Tensor | None = None  # [E], fp32, sums to 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.norm(x)
        b, length, d = h.shape
        flat = h.reshape(-1, d)  # [N, d]
        n_tokens = flat.shape[0]

        router_in = flat
        if self.training and self.router_jitter > 0.0:
            router_in = flat * (1.0 + self.router_jitter * torch.empty_like(flat).uniform_(-1.0, 1.0))
        # fp32 router logits/softmax for stability under bf16 autocast.
        router_logits = self.router(router_in).float()  # [N, E]
        probs = torch.softmax(router_logits, dim=-1)
        top_val, top_idx = torch.topk(probs, self.top_k, dim=-1)  # [N, k]
        top_val = top_val / top_val.sum(dim=-1, keepdim=True).clamp_min(1e-9)  # renorm over selected
        gate = top_val.to(flat.dtype)

        out = torch.zeros_like(flat)
        for e in range(self.num_experts):
            sel = top_idx == e  # [N, k]
            if not torch.any(sel):
                continue
            tok, slot = torch.nonzero(sel, as_tuple=True)  # [M]
            weight = gate[tok, slot].unsqueeze(-1)  # [M, 1]
            out.index_add_(0, tok, self.experts[e](flat[tok]) * weight)
        out = out.reshape(b, length, d)

        # --- auxiliary load-balance loss (gradient through P only) ---
        with torch.no_grad():
            assign = torch.zeros(self.num_experts, device=flat.device, dtype=torch.float32)
            assign.index_add_(0, top_idx.reshape(-1), torch.ones(top_idx.numel(), device=flat.device))
            f = assign / max(1, n_tokens * self.top_k)  # [E], sums to 1 when balanced -> 1/E each
        prob_mass = probs.mean(dim=0)  # [E], differentiable
        aux = self.num_experts * torch.sum(f * prob_mass)  # ≈1 balanced
        if self._z_weight > 0.0 and self._aux_weight > 0.0:
            z_loss = (torch.logsumexp(router_logits, dim=-1) ** 2).mean()
            aux = aux + (self._z_weight / self._aux_weight) * z_loss
        self.last_aux_loss = aux
        self.last_expert_fraction = f
        return residual + out


def expert_fraction_stats(fractions: list[torch.Tensor], num_experts: int) -> dict[str, float]:
    """Aggregate per-layer expert usage into scalar telemetry (collapse watch).

    Returns the worst-case single-expert token share (``moe_expert_frac_max``, 1/E when
    perfectly balanced) and the mean normalized routing entropy (``moe_expert_entropy``,
    1.0 = uniform, →0 = collapse) across MoE layers.
    """
    if not fractions:
        return {}
    frac_max = 0.0
    entropy_sum = 0.0
    log_e = math.log(max(2, num_experts))
    for frac in fractions:
        frac = frac.detach()
        frac_max = max(frac_max, float(frac.max()))
        p = frac.clamp_min(1e-9)
        entropy_sum += float(-(p * p.log()).sum() / log_e)
    return {
        "moe_expert_frac_max": frac_max,
        "moe_expert_entropy": entropy_sum / len(fractions),
    }


# --- mid-stack (src/models/midstack.py) ---
# ====================================================================================================
@dataclass
class BeatV11Config:
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
    # B2 · dt-bias banding: re-init the first `dt_band_heads` heads of each direction's Mamba3 toward a
    # long-memory timescale. In Mamba's selectivity convention a SMALL dt persists state / ignores input
    # (long memory); large dt resets to the current input (short memory). 0 => off (untouched log-uniform
    # init). A prior/nudge only — dt_bias is learnable + the decay is data-dependent — so pair with Idea A.
    dt_band_heads: int = 0
    dt_band_dt: float = 0.002  # target discrete step for the band (small = long memory); softplus^-1 => dt_bias.
    # F2 · RMSNorm (no mean-subtraction => RC-safe on mid-block hidden states) in place of the per-direction
    # LayerNorm pair. Default False => LayerNorm (byte-identical to pre-Phase-1). NB: distinct from the
    # REJECTED pooled-embedding LayerNorm (that broke rc_cos); this norms per-token mid-stack states.
    mid_block_rmsnorm_enabled: bool = False
    # F3 · torch.compile the mid-block forward at BLOCK granularity (Inductor fuses the norm/flip/add chain —
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

    # --- MoE: SwiGLU Mixture-of-Experts sublayers interleaved between mid-stack Mamba blocks.
    # Decouples per-token capacity (top_k of num_experts fire) from the always-active Mamba trunk.
    # Default off => byte-identical to the dense stack (no "moe" layers inserted).
    moe_enabled: bool = False
    moe_num_experts: int = 8
    moe_top_k: int = 2
    moe_expert_hidden: int = 0  # 0 => 2*d_model
    moe_placement: str = "even"  # insert a MoE sublayer after every even-indexed Mamba block
    moe_aux_weight: float = 0.01  # alpha for the load-balance aux loss (applied in compute_bioprime_loss)
    moe_z_weight: float = 0.0  # ST-MoE router z-loss (logsumexp² penalty); anti-collapse / logit conditioning
    moe_router_jitter: float = 0.0  # optional train-time input jitter for router exploration
    moe_norm: str = "rmsnorm"  # rmsnorm|layernorm for the MoE pre-norm

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
            raise ValueError(f"Beat-v11 downsample_factor must be a positive power of 2, got {factor}.")
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


def _moe_points(total: int, enabled: bool, placement: str) -> set[int]:
    """Block indices (1-based) after which a MoE sublayer is inserted. 'even' => every
    even-indexed block; 'all' => every block; otherwise off."""
    if not enabled or total <= 0:
        return set()
    if placement == "all":
        return set(range(1, total + 1))
    # default 'even'
    return {i for i in range(1, total + 1) if i % 2 == 0}


class BeatV11MidStack(nn.Module):
    def __init__(self, cfg: BeatV11Config) -> None:
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
        # MoE: a SwiGLU MoE sublayer after every even-indexed Mamba block (default off).
        self.moe_enabled = bool(getattr(cfg, "moe_enabled", False))
        self.moe_num_experts = int(getattr(cfg, "moe_num_experts", 8))
        moe_points = _moe_points(
            cfg.resolved_mid_mamba_blocks(), self.moe_enabled, str(getattr(cfg, "moe_placement", "even"))
        )
        self.last_moe_aux: torch.Tensor | None = None
        self.last_moe_stats: dict[str, float] = {}

        for block_index in range(1, cfg.resolved_mid_mamba_blocks() + 1):
            self.layers.append(BidiMambaMidBlock(cast(Any, cfg)))
            self.layer_kinds.append("mamba")
            if block_index in moe_points:
                self.layers.append(MoEMLPBlock(cast(Any, cfg)))
                self.layer_kinds.append("moe")
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
        moe_aux: torch.Tensor | None = None
        moe_fracs: list[torch.Tensor] = []
        for kind, layer in zip(self.layer_kinds, self.layers, strict=True):
            if kind == "mamba":
                x = layer(x)
            elif kind == "moe":
                x = layer(x)
                aux = cast(MoEMLPBlock, layer).last_aux_loss
                if aux is not None:
                    moe_aux = aux if moe_aux is None else moe_aux + aux
                frac = cast(MoEMLPBlock, layer).last_expert_fraction
                if frac is not None:
                    moe_fracs.append(frac)
            elif kind == "local":
                layer_dict = cast(nn.ModuleDict, layer)
                norm = cast(nn.LayerNorm, layer_dict["norm"])
                attn = cast(LocalWindowAttention, layer_dict["attn"])
                x = x + attn(norm(x), None)
            elif kind == "sparse":
                x = layer(x, edit_mid_mask=edit_mid_mask)
        # Surface MoE aux + utilization for the loss/telemetry (None when no MoE layers present).
        self.last_moe_aux = moe_aux
        self.last_moe_stats = expert_fraction_stats(moe_fracs, self.moe_num_experts)
        if self.register_tokens is not None:
            x = x[:, n_reg:]  # strip registers → mid_hidden_state keeps its original length
        return x


# --- model (src/models/bioprime/model.py) ---
# ====================================================================================================
@dataclass
class BioPrimeConfig(BeatV11Config):
    """Config for the BioPrime model.

    Subclasses ``BeatV11Config`` so every existing Beat-v11 ``model_config`` key
    validates (the registry rejects unknown keys), and adds the BioPrime-only fields.
    """

    num_conservation_bins: int = 16

    # Shakeout: use constant main-phase loss weights (no warmup/ramp/polish) so the EMA-normalized
    # total loss + grad-norm stay flat over a short run and every objective is active from step 0.
    constant_loss_weights_enabled: bool = False

    # Build-out §4a-ii — Hi-C within-window contact head (Akita/Orca-style). Reads the mid (regional)
    # representation, pools it to 2 kb bins, and predicts the near-diagonal log-O/E contact band via a
    # bilinear (left·right) interaction → direct supervision pressure on the long-range pathway.
    # hic_head_enabled=False → head not constructed (backbone bit-identical); gated additionally by w_hic.
    hic_head_enabled: bool = False
    hic_resolution_bp: int = 2000        # contact-bin size (matches the data/derived/hic_2kb cache)
    hic_max_offset_bins: int = 16        # predict offsets 0..15 (within a 32 kb / 16-bin window)
    hic_proj_dim: int = 64               # rank of the bilinear contact interaction

    # Backlog item 3 — ENCODE regulatory head bottleneck. The default expressive head
    # (LN→Linear(d_model,d_model)→GELU→Drop→Linear(d_model,tracks)) is expressive enough to absorb the
    # ENCODE gradient in the head itself (frozen probe flat at ~0.49); a low-rank bottleneck forces the
    # signal into the backbone. regulatory_head_rank: >0 → LN→Linear(d_model,r)→GELU→Linear(r,tracks);
    # 0 → legacy expressive head (default, bit-identical); <0 → pure linear probe LN→Linear(d_model,tracks).
    regulatory_head_rank: int = 0

    # T2-3 — ESM-2 missense-severity regression head. Per-position, 4 scalars (one per SNV_BASES alt)
    # → [B, L, 4], supervised by the data/derived/esm2_missense cache via a masked Huber loss. Built
    # only when True so the baseline backbone is bit-identical when off; gated additionally by the
    # w_missense_severity loss/data weight (set both together — validate_train_config enforces it).
    missense_severity_head_enabled: bool = False

    # T1-2a — conservation variant-delta. When True the forward also exposes conservation_delta_pred =
    # cons(mut) - cons(ref) at the synthetic-edit positions (the conservation head is linear, so this is
    # F.linear(edit_delta, conservation_head.weight) — NO new parameters, head reused). Supervised by the
    # conservation_delta loss term (magnitude tracks reference phyloP). Off ⇒ no extra output, bit-identical
    # backbone + head_outputs view; gated additionally by w_conservation_delta (validate enforces pairing).
    conservation_delta_head_enabled: bool = False

    # T4-8b — per-loss gating (vs the EMA-floor's up-to-4x amplification of satisfied losses). Threaded
    # into LossEMANormalizer below. Default False ⇒ EMA normalization is byte-identical to the prior runs.
    per_loss_gating_enabled: bool = False

    # Item 5 — cell-type conditioning. Rides the Hi-C contact path (reads the mid representation pooled to
    # contact bins), so it requires hic_head_enabled. Default OFF ⇒ bit-identical to the pre-item-5 model.
    #
    # cell_conditioning_enabled applies a FiLM to the pooled, post-LayerNorm contact bins, conditioned on the
    # per-window sampled cell. (F2): warm-started with a tiny nonzero init (see _cell_film below) so the
    # cell embedding gets gradient from step 0 instead of the old zero-init cold-start. The FiLM modules are
    # built only when the Hi-C head exists, so they stay inert on an unconditioned config.
    cell_conditioning_enabled: bool = False
    cell_cond_dim: int = 64
    n_cell_types: int = 4                # GM12878, K562, HepG2, IMR90 (vocab order = data.base.CELL_TYPES)

    # Step-1000 grad-explosion fix — Final RMSNorm on the trunk output H = cat(h_up, h_pure) (d_full).
    # The full-res token heads read `hidden` RAW (no per-head input norm), so an un-normalized H let the
    # readout/trunk gradients blow up (a data-order fluctuation → compounding escalation grad-clip could
    # only cap in magnitude, not direction). Normalizing H pins per-token ‖H‖ (RMS≈1) → conditioned
    # gradients. Default False ⇒ backbone bit-identical; True builds one nn.RMSNorm(d_full).
    trunk_final_norm_enabled: bool = False


class DNAFoundationBioPrime(nn.Module):
    def __init__(self, cfg: BioPrimeConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.pad_token_id = PAD_ID
        self.n_downsample_stages = cfg.resolved_downsample_stages()
        self.downsample_factor = 2**self.n_downsample_stages
        self.full_hidden_dim = cfg.d_model + cfg.d_pure
        if cfg.d_full != self.full_hidden_dim:
            raise ValueError(
                f"BioPrime d_full must equal d_model + d_pure; got d_full={cfg.d_full} "
                f"and d_model + d_pure={self.full_hidden_dim}."
            )
        if cfg.resolved_position_encoding() != "sinusoidal":
            raise ValueError("BioPrime supports sinusoidal position_encoding only.")

        # --- backbone (identical to Beat-v11) ---
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=PAD_ID)
        self.pos_emb = SinusoidalPositionEmbedding(cfg.l_max, cfg.d_model)
        self.stem = MultiKernelStem(cfg.d_model, cfg.d_pure, cfg.conv_stem_kernels)
        self.down_stages = nn.ModuleList([DownStage(cfg.d_model) for _ in range(self.n_downsample_stages)])
        self.mid_stack = BeatV11MidStack(cfg)
        self.up_stages = nn.ModuleList([UpStage(cfg.d_model) for _ in range(self.n_downsample_stages)])
        self.dropout = nn.Dropout(cfg.dropout)

        d_full = self.full_hidden_dim
        # Step-1000 grad-explosion fix: optional Final RMSNorm on the trunk output H = cat(h_up, h_pure).
        # The full-res token heads read `hidden` RAW, so an un-normed H let readout/trunk gradients blow up
        # (grad-clip bounds magnitude, not direction). Normalizing H pins ‖H‖ (per-token RMS≈1) so those
        # gradients stay conditioned. Default off ⇒ backbone bit-identical; the mid-reading heads
        # (regulatory/Hi-C) already self-normalize their input, so only `hidden` needs this.
        self.trunk_final_norm: nn.Module | None = nn.RMSNorm(d_full) if cfg.trunk_final_norm_enabled else None
        # --- BioPrime heads only ---
        self.mlm_head = nn.Linear(d_full, len(SNV_BASES))
        self.conservation_scalar_head = nn.Linear(d_full, cfg.num_conservation_targets)
        self.conservation_bin_head = nn.Linear(d_full, cfg.num_conservation_bins)
        self.splice_class_head = nn.Sequential(
            nn.Linear(d_full, 64), nn.GELU(), nn.Linear(64, cfg.num_splice_classes)
        )
        self.splice_distance_head = nn.Sequential(nn.Linear(d_full, 64), nn.GELU(), nn.Linear(64, 1))
        self.region_head = nn.Linear(d_full, cfg.num_region_classes)
        # Counterfactual ref/alt head: per-position SNV consequence over precomputed labels.
        self.counterfactual_snv_head = nn.Linear(d_full, len(SNV_BASES) * cfg.num_counterfactual_effect_classes)
        # T2-3 ESM-2 missense-severity regression head: per-position, one scalar per SNV_BASES alt → [B,L,4].
        # Built only when enabled (baseline backbone bit-identical when off); runs every forward when built
        # so its params stay in the DDP static graph (the loss emits a graph-connected zero on empty batches).
        # 2-layer MLP (hidden 64) instead of a bare Linear — the DNA→protein-severity map the ESM-2
        # distillation target teaches needs capacity a linear readout lacks. Mirrors the splice_class_head idiom.
        self.missense_severity_head: nn.Module | None = (
            nn.Sequential(nn.Linear(d_full, 64), nn.GELU(), nn.Linear(64, len(SNV_BASES)))
            if cfg.missense_severity_head_enabled
            else None
        )
        # Weak population-prior head (the single consolidated gnomAD head): observed + log-AF.
        self.population_af_head = nn.Linear(d_full, len(SNV_BASES))
        self.population_observed_head = nn.Linear(d_full, len(SNV_BASES))
        # --- Build-out §4b: ENCODE regulatory head (regional scale) ---
        # Reads the mid representation (d_model channels, 4 bp/token); pooled x2 in forward to the 8 bp
        # ENCODE grid, then predicts the 50 tracks. Constructed only when tracks are requested so the
        # baseline backbone is unchanged when off (num_regulatory_tracks=0).
        if cfg.num_regulatory_tracks > 0:
            r = int(cfg.regulatory_head_rank)
            if r > 0:
                # item 3: low-rank bottleneck — forces the ENCODE signal into the backbone.
                self.regulatory_head: nn.Module | None = nn.Sequential(
                    nn.LayerNorm(cfg.d_model),
                    nn.Linear(cfg.d_model, r),
                    nn.GELU(),
                    nn.Linear(r, cfg.num_regulatory_tracks),
                )
            elif r < 0:
                # item 3: pure linear probe (strongest bottleneck — no head nonlinearity at all).
                self.regulatory_head = nn.Sequential(
                    nn.LayerNorm(cfg.d_model),
                    nn.Linear(cfg.d_model, cfg.num_regulatory_tracks),
                )
            else:
                # legacy expressive head (default; bit-identical to pre-item-3 runs).
                self.regulatory_head = nn.Sequential(
                    nn.LayerNorm(cfg.d_model),
                    nn.Linear(cfg.d_model, cfg.d_model),
                    nn.GELU(),
                    nn.Dropout(cfg.dropout),
                    nn.Linear(cfg.d_model, cfg.num_regulatory_tracks),
                )
        else:
            self.regulatory_head = None

        # --- Build-out §4a-ii: Hi-C contact head (long-range scale) ---
        # Pools the mid representation to hic_resolution_bp bins and predicts the near-diagonal band
        # via a bilinear interaction band[d, k] = scale * (left[d] · right[d+k]) + offset_bias[k].
        if cfg.hic_head_enabled:
            self.hic_norm: nn.Module | None = nn.LayerNorm(cfg.d_model)
            self.hic_left: nn.Module | None = nn.Linear(cfg.d_model, cfg.hic_proj_dim)
            self.hic_right: nn.Module | None = nn.Linear(cfg.d_model, cfg.hic_proj_dim)
            self.hic_offset_bias: nn.Parameter | None = nn.Parameter(torch.zeros(cfg.hic_max_offset_bins))
            self._hic_scale = float(cfg.hic_proj_dim) ** -0.5
        else:
            self.hic_norm = None
            self.hic_left = None
            self.hic_right = None
            self.hic_offset_bias = None
            self._hic_scale = 1.0

        # --- Item 5: cell-type conditioning (FiLM) on the Hi-C contact bins ---
        # Both ride the Hi-C contact path. The FiLM is ZERO-INITIALIZED so gamma=beta=0 at step 0 ⇒
        # FiLM(x) = x·(1+gamma)+beta = x (identity): the Hi-C band output is bit-identical to the no-conditioning
        # model at init, and only diverges as the cell embedding learns. It conditions the POOLED,
        # post-LayerNorm bins (a LayerNorm AFTER the FiLM would cancel gamma/beta — see _contact_bins). Built only
        # when a consumer head exists so an unconditioned config is unchanged.
        self._cell_conditioning = bool(cfg.cell_conditioning_enabled) and cfg.hic_head_enabled
        if self._cell_conditioning:
            self.cell_embedding: nn.Module | None = nn.Embedding(int(cfg.n_cell_types), int(cfg.cell_cond_dim))
            _cell_film = nn.Linear(int(cfg.cell_cond_dim), 2 * cfg.d_model)
            # (F2 "Fix E"): warm-start the FiLM near-identity with a tiny nonzero weight init instead of the
            # old zero-init cold-start (gamma=beta=0 gave the cell embedding no gradient until cell_film moved
            # off zero). The small std keeps the Hi-C band ~identical at init while letting the cell signal flow
            # from step 0; bias stays 0 so gamma≈beta≈0 ⇒ bins·(1+gamma)+beta ≈ bins.
            nn.init.normal_(_cell_film.weight, std=1e-3)
            nn.init.zeros_(_cell_film.bias)
            self.cell_film: nn.Module | None = _cell_film
        else:
            self.cell_embedding = None
            self.cell_film = None

        # EMA per-loss normalizer (plan §10). Lives in the model so its buffers are
        # checkpointed/restored automatically and survive spot resume. No parameters,
        # so it is inert for DDP gradient sync. T4-8b per-loss gating is threaded from the config
        # (default False ⇒ normalization byte-identical to the prior runs).
        self.loss_ema = LossEMANormalizer(
            BIOPRIME_LOSS_NAMES, per_loss_gating_enabled=bool(cfg.per_loss_gating_enabled)
        )

    def _position_embeddings(self, seq_len: int, x: torch.Tensor) -> torch.Tensor:
        if seq_len > self.cfg.l_max:
            raise ValueError(f"BioPrime seq_len={seq_len} exceeds l_max={self.cfg.l_max}.")
        return self.pos_emb(seq_len, device=x.device, dtype=x.dtype).unsqueeze(0)

    def _downsample_edit_mask(self, edit_mask: torch.Tensor, mid_len: int) -> torch.Tensor:
        import torch.nn.functional as F

        pooled = edit_mask.to(dtype=torch.float32).unsqueeze(1)
        for _ in range(self.n_downsample_stages):
            pooled = F.max_pool1d(pooled, kernel_size=2, stride=2, ceil_mode=True)
        if pooled.shape[-1] > mid_len:
            pooled = pooled[..., :mid_len]
        elif pooled.shape[-1] < mid_len:
            pooled = F.pad(pooled, (0, mid_len - pooled.shape[-1]))
        return pooled.squeeze(1).to(dtype=torch.bool)

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        variant_edit_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        import torch.nn.functional as F  # noqa: F401  (parity with v11; F used via UpStage)

        _batch_size, seq_len = input_ids.shape
        x = self.token_emb(input_ids)
        x = x + self._position_embeddings(seq_len, x)
        x = self.dropout(x)

        h_stem, h_pure = self.stem(x)
        skips: list[torch.Tensor] = [h_stem]
        h = h_stem
        for stage in self.down_stages:
            h = stage(h)
            skips.append(h)
        h_mid = h
        edit_mid_mask = (
            None if variant_edit_mask is None else self._downsample_edit_mask(variant_edit_mask, h_mid.shape[1])
        )
        h_mid = self.mid_stack(h_mid, edit_mid_mask=edit_mid_mask)

        h_up = h_mid
        for stage, skip in zip(self.up_stages, reversed(skips[:-1]), strict=True):
            h_up = stage(h_up, skip)
        if h_up.shape[1] != seq_len:
            h_up = UpStage._match_length(h_up, seq_len)

        # Always compute the variant-residual boost when a mask is present (zero-gated by torch.where: on
        # non-variant batches the mask is all-False => h_up returned unchanged, and no gradient flows through
        # `boost`). Dropping the `torch.any(...)` short-circuit keeps the autograd graph IDENTICAL across
        # iterations, so DDP static_graph=True is valid (the mut pass + edit-loc + conservation-delta heads
        # are already run every step). In training step.py always passes the (all-zero-on-non-variant) mask;
        # eval may pass None, which correctly skips this (no DDP there).
        if self.cfg.use_variant_token_residual and variant_edit_mask is not None:
            boost = self.cfg.variant_residual_gamma * h_stem
            h_up = torch.where(variant_edit_mask.unsqueeze(-1).to(dtype=torch.bool), h_up + boost, h_up)

        hidden = torch.cat([h_up, h_pure], dim=-1)
        # Step-1000 grad-explosion fix: bound ‖H‖ so the raw-reading full-res token heads get a normalized
        # readout (no-op when trunk_final_norm_enabled=False). `hidden` is a fresh tensor — normalizing it
        # here does not touch h_mid, the up-stack, or the (already self-normalizing) mid heads.
        if self.trunk_final_norm is not None:
            hidden = self.trunk_final_norm(hidden)
        encoded = {"last_hidden_state": hidden, "mid_hidden_state": h_mid}
        # MoE: surface the load-balance aux loss (+ utilization telemetry) produced by this pass.
        moe_aux = getattr(self.mid_stack, "last_moe_aux", None)
        if moe_aux is not None:
            encoded["moe_aux_loss"] = moe_aux
        encoded["_moe_stats"] = getattr(self.mid_stack, "last_moe_stats", {})  # type: ignore[assignment]
        return encoded

    def _token_head_outputs(self, hidden: torch.Tensor, delta_hidden: torch.Tensor | None) -> dict[str, torch.Tensor]:
        outputs: dict[str, torch.Tensor] = {}
        outputs["mlm_logits"] = self.mlm_head(hidden)
        conservation = self.conservation_scalar_head(hidden)
        outputs["conservation_scalar_pred"] = conservation
        if conservation.shape[-1] > 0:
            outputs["phylo100_pred"] = conservation[..., 0]
        if conservation.shape[-1] > 1:
            outputs["zoo241_pred"] = conservation[..., 1]  # Zoonomia-241 (real, once zoo241_bw_path is fixed)
        if conservation.shape[-1] > 2:
            outputs["phylo470_pred"] = conservation[..., 2]
        outputs["conservation_bin_logits"] = self.conservation_bin_head(hidden)
        outputs["splice_class_logits"] = self.splice_class_head(hidden)
        outputs["splice_distance_pred"] = self.splice_distance_head(hidden).squeeze(-1)
        outputs["region_logits"] = self.region_head(hidden)
        cf_logits = self.counterfactual_snv_head(hidden)
        outputs["counterfactual_effect_logits"] = cf_logits.view(
            *cf_logits.shape[:2], len(SNV_BASES), self.cfg.num_counterfactual_effect_classes
        )
        if self.missense_severity_head is not None:
            outputs["missense_severity_pred"] = self.missense_severity_head(hidden)  # [B, L, 4]
        outputs["gnomad_af_pred"] = self.population_af_head(hidden)
        outputs["gnomad_observed_logits"] = self.population_observed_head(hidden)
        # Synthetic-edit delta (mutant-minus-reference hidden; zero on non-variant batches). Kept because the
        # conservation variant-delta below reuses it; computed every forward so the autograd graph is identical
        # across iterations (DDP static_graph safety). The edit-localization readout that used to consume it was
        # removed (dead objective, offline AUROC below chance).
        edit_delta = delta_hidden if delta_hidden is not None else torch.zeros_like(hidden)
        # T1-2a: predicted Δconservation under the variant. The conservation head is linear, so
        # cons(ref+delta) - cons(ref) = W·delta (bias cancels) — reuse its weight, no new params. Computed
        # every forward when enabled (edit_delta is zeros on non-variant batches) ⇒ DDP-static-graph-safe.
        if self.cfg.conservation_delta_head_enabled:
            outputs["conservation_delta_pred"] = F.linear(
                edit_delta, self.conservation_scalar_head.weight
            )  # [B, L, num_conservation_targets]
        return outputs

    def _contact_bins(
        self, h_mid: torch.Tensor, cell_id: torch.Tensor | None, window_start: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Pool the mid representation to hic_resolution_bp contact bins and (item 5) FiLM-condition them
        on the cell type. Returns [B, n_bins, C]. Read by the Hi-C band (item 5 supervision).

        (F2a): the bins are ALIGNED TO THE GLOBAL 2 kb grid the target uses. The data reads contacts on
        the global grid (target bin d == global bin start//2000 + d), while the old model pooled 512-token
        (2048 bp) bins from window-bp-0 — a per-window RANDOM phase offset (start % 2000, avg half a bin) plus
        a 48 bp/bin scale drift that the model could not learn a correction for, forcing a blurred average and
        capping Pearson for every cell. Given the per-sample ``window_start`` we assign each mid token to its
        global 2 kb bin (res//downsample = 500 tokens/bin) and scatter-mean, so model bin d == target bin d.
        ``window_start=None`` (eval callers that don't thread it) falls back to the legacy window-relative
        reshape-mean. The pool is done in fp32 (summing ~500 bf16 values loses precision).

        FiLM is applied AFTER the pool (F2: warm-started near-identity — cell_film weight std 1e-3, bias 0,
        so gamma≈beta≈0 and the band is ~unchanged at init while the cell embedding gets gradient from step 0).
        """
        assert self.hic_norm is not None
        res = int(self.cfg.hic_resolution_bp)
        seq_len_bp = h_mid.shape[1] * self.downsample_factor
        n_bins = max(1, seq_len_bp // res)
        mid = self.hic_norm(h_mid)                                    # [B, mid_len, C]
        b, mid_len, c = mid.shape
        if window_start is None:
            # Legacy / eval: window-relative reshape-mean from token 0 (res//downsample tokens per bin).
            k = max(1, mid_len // n_bins)
            bins = mid[:, : n_bins * k].reshape(b, n_bins, k, c).mean(dim=2)      # [B, n_bins, C]
        else:
            # F2a: token i (genomic window_start + i*downsample bp) → global bin floor((start%res + i*ds)/res)
            # relative to the window's first global bin; scatter-mean tokens into their bins. DDP-static-graph
            # safe (fixed op/shapes; only the index VALUES vary) and scatter_add's backward is a gather (no
            # adaptive-pool shared-memory launch-bound issue), so it is safe at 32k.
            ds = self.downsample_factor
            off = (window_start.to(mid.device).long() % res).view(b, 1)               # [B,1] bp into first bin
            tok_bp = torch.arange(mid_len, device=mid.device).view(1, mid_len) * ds   # [1, mid_len]
            bin_idx = ((off + tok_bp) // res).clamp_(0, n_bins - 1)                   # [B, mid_len] global bin/token
            idx_c = bin_idx.unsqueeze(-1).expand(b, mid_len, c)
            ones = torch.ones(b, mid_len, 1, device=mid.device, dtype=torch.float32)
            sums = torch.zeros(b, n_bins, c, device=mid.device, dtype=torch.float32).scatter_add_(1, idx_c, mid.float())
            cnts = torch.zeros(b, n_bins, 1, device=mid.device, dtype=torch.float32).scatter_add_(
                1, bin_idx.unsqueeze(-1), ones
            )
            bins = (sums / cnts.clamp_min(1.0)).to(mid.dtype)                         # [B, n_bins, C]
        if self._cell_conditioning and self.cell_film is not None and self.cell_embedding is not None:
            if cell_id is None:
                cell_id = bins.new_zeros(bins.shape[0], dtype=torch.long)  # default cell 0 (GM12878)
            cond = self.cell_film(self.cell_embedding(cell_id.to(torch.long)))  # [B, 2C]
            gamma, beta = cond.chunk(2, dim=-1)                       # each [B, C]
            bins = bins * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)  # warm-started near-identity (F2)
        return bins

    def _hic_band(
        self, h_mid: torch.Tensor, cell_id: torch.Tensor | None = None, window_start: torch.Tensor | None = None
    ) -> torch.Tensor:
        """§4a-ii: predict the within-window near-diagonal contact band via a bilinear interaction
        band[d, k] = scale·(left[d]·right[d+k]) + offset_bias[k] over the (cell-conditioned) contact bins.
        n_bins is derived from the window length so the head is sequence-length agnostic."""
        assert self.hic_left is not None and self.hic_right is not None
        assert self.hic_offset_bias is not None
        n_off = int(self.cfg.hic_max_offset_bins)
        bins = self._contact_bins(h_mid, cell_id, window_start)      # [B, n_bins, C] (F2a: global-2kb-aligned)
        n_bins = bins.shape[1]
        left = self.hic_left(bins)                                   # [B, n_bins, r]
        right = self.hic_right(bins)                                 # [B, n_bins, r]
        contact = torch.einsum("bnr,bmr->bnm", left, right) * self._hic_scale  # [B, n_bins, n_bins]
        d_idx = torch.arange(n_bins, device=contact.device).view(n_bins, 1).expand(n_bins, n_off)
        k_idx = torch.arange(n_off, device=contact.device).view(1, n_off).expand(n_bins, n_off)
        e_idx = (d_idx + k_idx).clamp(max=n_bins - 1)                 # within-window; out-of-band masked in loss
        band = contact[:, d_idx, e_idx]                              # [B, n_bins, n_off]
        return band + self.hic_offset_bias.view(1, 1, n_off)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        return_token_heads: bool = True,
        return_hidden: bool = True,
        variant_edit_mask: torch.Tensor | None = None,
        delta_hidden: torch.Tensor | None = None,
        edit_delta_from_hidden_states: torch.Tensor | None = None,
        cell_id: torch.Tensor | None = None,
        hic_window_start: torch.Tensor | None = None,  # F2a: per-sample window start (bp) for global-2kb bin align
    ) -> dict[str, Any]:
        if attention_mask is None:
            attention_mask = input_ids.ne(PAD_ID)
        encoded = self.encode(input_ids, attention_mask=attention_mask, variant_edit_mask=variant_edit_mask)
        hidden = encoded["last_hidden_state"]
        h_mid = encoded["mid_hidden_state"]

        if delta_hidden is not None and edit_delta_from_hidden_states is not None:
            raise ValueError("Pass either delta_hidden or edit_delta_from_hidden_states, not both.")
        if edit_delta_from_hidden_states is not None:
            if edit_delta_from_hidden_states.shape != hidden.shape:
                raise ValueError(
                    "edit_delta_from_hidden_states must match hidden shape, got "
                    f"{tuple(edit_delta_from_hidden_states.shape)} and {tuple(hidden.shape)}."
                )
            delta_hidden = edit_delta_from_hidden_states - hidden

        outputs: dict[str, Any] = {
            "last_hidden_state": hidden,
            "mid_hidden_state": h_mid,
            "head_outputs": {},
        }
        # MoE: pass the aux load-balance loss + utilization telemetry to the loss layer.
        if "moe_aux_loss" in encoded:
            outputs["moe_aux_loss"] = encoded["moe_aux_loss"]
        if encoded.get("_moe_stats"):
            outputs["moe_stats"] = encoded["_moe_stats"]
        if return_hidden:
            outputs["hidden_states"] = hidden
            outputs["mid_hidden_states"] = h_mid
        else:
            outputs["mid_hidden_states"] = h_mid

        if return_token_heads:
            head_outputs = self._token_head_outputs(hidden, delta_hidden)
            outputs.update(head_outputs)

        # Build-out upper-scale heads read the mid (regional) representation. Gated on return_token_heads
        # so they run on the main + rc passes (consistent across ranks) and are skipped on the mut pass.
        if return_token_heads and self.regulatory_head is not None:
            # §4b: mid is 4 bp/token; ENCODE targets are 8 bp → pool x2 to align before the per-track head.
            mid_t = h_mid.transpose(1, 2)
            pooled_mid = F.avg_pool1d(mid_t, kernel_size=2, stride=2).transpose(1, 2)
            outputs["regulatory_pred"] = self.regulatory_head(pooled_mid)
        if return_token_heads and self.hic_left is not None:
            # §4a-ii: [B, n_bins, n_off] log-O/E contact band
            outputs["hic_band_pred"] = self._hic_band(h_mid, cell_id, hic_window_start)

        # Public plan-named head_outputs view.
        if return_token_heads:
            outputs["head_outputs"] = {
                "mlm": outputs.get("mlm_logits"),
                "conservation": {
                    "scalar": outputs.get("conservation_scalar_pred"),
                    "bins": outputs.get("conservation_bin_logits"),
                },
                "splice": {
                    "class": outputs.get("splice_class_logits"),
                    "distance": outputs.get("splice_distance_pred"),
                },
                "region": outputs.get("region_logits"),
                "counterfactual_variant": outputs.get("counterfactual_effect_logits"),
                "population_prior": {
                    "af": outputs.get("gnomad_af_pred"),
                    "observed": outputs.get("gnomad_observed_logits"),
                },
            }
            # T2-3: only exposed when the head is built, so the off-path head_outputs view is unchanged.
            if "missense_severity_pred" in outputs:
                outputs["head_outputs"]["missense_severity"] = outputs["missense_severity_pred"]
            # T1-2a: only exposed when enabled, so the off-path head_outputs view is unchanged.
            if "conservation_delta_pred" in outputs:
                outputs["head_outputs"]["conservation_delta"] = outputs["conservation_delta_pred"]
        return outputs


def build_bioprime_model(cfg: BioPrimeConfig) -> DNAFoundationBioPrime:
    return DNAFoundationBioPrime(cfg)


# --- standalone aliases (v10-style naming) ---
DNAFoundationBeatV11 = DNAFoundationBioPrime
build_beat_v11_model = build_bioprime_model
