from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from lumina.models.mamba_runtime import require_mamba3, resolve_chunk_size


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
            raise ValueError("Lumina stem requires at least one convolution kernel.")
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
