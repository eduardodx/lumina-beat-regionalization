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
    NUM_V10_SPLICE_CLASSES,
    PAD_ID,
    SNV_BASES,
    VOCAB_SIZE,
)
from .local_attn import LocalWindowAttention

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
            "Install a mamba-ssm build that exposes `mamba_ssm.Mamba3`. "
            "On Linux/CUDA, this usually means a recent state-spaces/mamba build."
        )
    return _Mamba3


def resolve_chunk_size(cfg: Any) -> int:
    """Mamba3 MIMO with bf16 needs chunk_size = 64 / mimo_rank."""

    if cfg.is_mimo and cfg.chunk_size == 64:
        return max(1, 64 // cfg.mimo_rank)
    return cfg.chunk_size


@dataclass
class BeatV10Config:
    # Spec section 3 lists V=6 (A, C, G, T, N, [MASK]); we use VOCAB_SIZE=8 because the pipeline-wide
    # tokenizer reserves PAD_ID=0 and UNK_ID=7. The extra two embedding rows are functionally
    # unused; keeping V=8 avoids a separate vocab map for v10.
    vocab_size: int = VOCAB_SIZE
    d_model: int = 256
    d_state: int = 128
    d_pure: int = 32
    expand: int = 2
    headdim: int = 64
    ngroups: int = 1
    rope_fraction: float = 1.0
    chunk_size: int = 64
    is_mimo: bool = False
    mimo_rank: int = 4
    is_outproj_norm: bool = True
    dropout: float = 0.05
    activation_checkpointing: bool = True
    checkpoint_use_reentrant: bool = False

    l_max: int = 32768
    position_embedding: str = "sinusoidal"
    conv_stem_kernels: list[int] = field(default_factory=lambda: [3, 7, 15])
    n_downsample_stages: int = 3
    n_mid_blocks: int = 8
    local_attention_window: int = 256
    local_attention_heads: int = 4
    sparse_global_stride: int = 16
    sparse_global_heads: int = 4
    variant_residual_gamma: float = 0.5

    num_region_classes: int = NUM_REGION_CLASSES
    num_splice_classes: int = NUM_V10_SPLICE_CLASSES
    num_counterfactual_effect_classes: int = NUM_COUNTERFACTUAL_EFFECT_CLASSES
    num_regulatory_tracks: int = 50
    use_sequence_embedding: bool = True
    sequence_embedding_dim: int = 256
    heads: dict[str, bool] = field(default_factory=dict)


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


class LearnablePositionEmbedding(nn.Module):
    def __init__(self, l_max: int, d_model: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(l_max, d_model))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, seq_len: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return self.weight[:seq_len].to(device=device, dtype=dtype)


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
    def __init__(self, cfg: BeatV10Config) -> None:
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
        # Spec section 4.4 calls for a single post-fusion LayerNorm on raw x. We instead pre-norm
        # each direction: Mamba3 with is_outproj_norm=True already normalizes branch outputs,
        # so a post-fusion norm would be duplicative, and pre-norm-on-input is the bf16-stable
        # convention at depth 8.
        self.norm_fwd = nn.LayerNorm(cfg.d_model)
        self.norm_bwd = nn.LayerNorm(cfg.d_model)
        self.fwd = Mamba3(**kwargs)
        self.bwd = Mamba3(**kwargs)
        self.activation_checkpointing = cfg.activation_checkpointing
        self.checkpoint_use_reentrant = cfg.checkpoint_use_reentrant

    def _maybe_checkpoint(self, module: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if (
            not self.activation_checkpointing
            or not self.training
            or not torch.is_grad_enabled()
            or not x.requires_grad
        ):
            return module(x)
        return cast(torch.Tensor, activation_checkpoint(module, x, use_reentrant=self.checkpoint_use_reentrant))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fwd = self._maybe_checkpoint(self.fwd, self.norm_fwd(x))
        bwd = torch.flip(self._maybe_checkpoint(self.bwd, self.norm_bwd(torch.flip(x, dims=[1]))), dims=[1])
        return x + fwd + bwd


class SparseGlobalAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, stride: int, dropout: float) -> None:
        super().__init__()
        self.stride = max(1, int(stride))
        self.norm = nn.LayerNorm(d_model)
        self.strided_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.anchor_query_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.anchor_key_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edit_mid_mask: torch.Tensor | None = None) -> torch.Tensor:
        z = self.norm(x)
        keys = z[:, :: self.stride]
        attended, _ = self.strided_attn(z, keys, keys, need_weights=False)
        out = x + self.dropout(attended)

        if edit_mid_mask is None or not torch.any(edit_mid_mask):
            return out

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


class MidStack(nn.Module):
    def __init__(self, cfg: BeatV10Config) -> None:
        super().__init__()
        # Spec section 4.4 canonical schedule: 8 BidiMamba + 3 LocalAttention + 2 SparseGlobalAttention = 13 layers.
        # Any other n_mid_blocks count silently breaks the interleaving (Mamba slots would be dropped),
        # so we hard-fail rather than emit a degraded architecture.
        if int(cfg.n_mid_blocks) != 8:
            raise ValueError(
                f"Beat-v10 mid-stack schedule is canonical at n_mid_blocks=8; got n_mid_blocks={cfg.n_mid_blocks}."
            )
        self.layers = nn.ModuleList()
        self.layer_kinds: list[str] = []
        schedule = (
            "mamba",
            "mamba",
            "mamba",
            "local",
            "mamba",
            "sparse",
            "mamba",
            "local",
            "mamba",
            "sparse",
            "mamba",
            "local",
            "mamba",
        )
        for kind in schedule:
            if kind == "mamba":
                self.layers.append(BidiMambaMidBlock(cfg))
                self.layer_kinds.append(kind)
            elif kind == "local":
                self.layers.append(
                    nn.ModuleDict(
                        {
                            "norm": nn.LayerNorm(cfg.d_model),
                            "attn": LocalWindowAttention(
                                d_model=cfg.d_model,
                                n_heads=cfg.local_attention_heads,
                                window=cfg.local_attention_window,
                                dropout=cfg.dropout,
                            ),
                        }
                    )
                )
                self.layer_kinds.append(kind)
            elif kind == "sparse":
                self.layers.append(
                    SparseGlobalAttention(
                        d_model=cfg.d_model,
                        n_heads=cfg.sparse_global_heads,
                        stride=cfg.sparse_global_stride,
                        dropout=cfg.dropout,
                    )
                )
                self.layer_kinds.append(kind)

    def forward(self, x: torch.Tensor, edit_mid_mask: torch.Tensor | None = None) -> torch.Tensor:
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
        return x


class SequenceEmbeddingHead(nn.Module):
    def __init__(self, d_in: int, d_out: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_in, d_out)

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        if attention_mask is None:
            pooled = hidden.mean(dim=1)
        else:
            mask = attention_mask.to(dtype=hidden.dtype).unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.proj(pooled)


class RegulatoryHead(nn.Module):
    def __init__(self, d_model: int, n_tracks: int) -> None:
        super().__init__()
        # Spec section 5.2.1: trunk = Linear(256 -> 256) -> GELU -> Linear(256 -> 128);
        # per-track Linear(128 -> 1).
        # A single Linear(128, n_tracks) is functionally equivalent to n_tracks per-track linears.
        self.trunk = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 128),
        )
        self.out = nn.Linear(128, n_tracks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.trunk(x))


class DNAFoundationBeatV10(nn.Module):
    def __init__(self, cfg: BeatV10Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.pad_token_id = PAD_ID
        self.downsample_factor = 2 ** int(cfg.n_downsample_stages)
        self.full_hidden_dim = cfg.d_model + cfg.d_pure

        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=PAD_ID)
        self.pos_emb: nn.Module
        if cfg.position_embedding == "sinusoidal":
            self.pos_emb = SinusoidalPositionEmbedding(cfg.l_max, cfg.d_model)
        elif cfg.position_embedding == "learned":
            self.pos_emb = LearnablePositionEmbedding(cfg.l_max, cfg.d_model)
        else:
            raise ValueError("Beat-v10 position_embedding must be 'sinusoidal' or 'learned'.")

        self.stem = MultiKernelStem(cfg.d_model, cfg.d_pure, cfg.conv_stem_kernels)
        self.down_stages = nn.ModuleList([DownStage(cfg.d_model) for _ in range(cfg.n_downsample_stages)])
        self.mid_stack = MidStack(cfg)
        self.up_stages = nn.ModuleList([UpStage(cfg.d_model) for _ in range(cfg.n_downsample_stages)])
        # Spec section 4.7 emits the concatenated [h_up, h_pure] without an output LayerNorm; downstream
        # heads handle their own normalization as needed.
        self.dropout = nn.Dropout(cfg.dropout)

        d_full = self.full_hidden_dim
        self.mlm_head = nn.Linear(d_full, len(SNV_BASES))
        self.phylo100_subst_head = nn.Linear(d_full, len(SNV_BASES))
        self.zoo241_subst_head = nn.Linear(d_full, len(SNV_BASES))
        self.splice_class_head = nn.Sequential(nn.Linear(d_full, 64), nn.GELU(), nn.Linear(64, cfg.num_splice_classes))
        self.splice_distance_head = nn.Sequential(nn.Linear(d_full, 64), nn.GELU(), nn.Linear(64, 1))
        self.region_head = nn.Linear(d_full, cfg.num_region_classes)
        self.gnomad_af_head = nn.Linear(d_full, len(SNV_BASES))
        self.gnomad_observed_head = nn.Linear(d_full, len(SNV_BASES))
        self.counterfactual_snv_head = nn.Linear(
            d_full,
            len(SNV_BASES) * cfg.num_counterfactual_effect_classes,
        )
        self.edit_localize_head = nn.Linear(d_full, 1)
        self.regulatory_head: RegulatoryHead | None = None
        if cfg.num_regulatory_tracks > 0:
            self.regulatory_head = RegulatoryHead(cfg.d_model, cfg.num_regulatory_tracks)
        self.sequence_embedding_head: SequenceEmbeddingHead | None = None
        if cfg.use_sequence_embedding:
            self.sequence_embedding_head = SequenceEmbeddingHead(d_full, cfg.sequence_embedding_dim)

    def _head_enabled(self, name: str) -> bool:
        return bool(self.cfg.heads.get(name, True))

    def _position_embeddings(self, seq_len: int, x: torch.Tensor) -> torch.Tensor:
        if seq_len > self.cfg.l_max:
            raise ValueError(f"Beat-v10 seq_len={seq_len} exceeds l_max={self.cfg.l_max}.")
        return self.pos_emb(seq_len, device=x.device, dtype=x.dtype).unsqueeze(0)

    def _downsample_edit_mask(self, edit_mask: torch.Tensor, mid_len: int) -> torch.Tensor:
        pooled = edit_mask.to(dtype=torch.float32).unsqueeze(1)
        for _ in range(self.cfg.n_downsample_stages):
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
        skip_stack = skips[:-1]
        for stage, skip in zip(self.up_stages, reversed(skip_stack), strict=True):
            h_up = stage(h_up, skip)
        if h_up.shape[1] != seq_len:
            h_up = UpStage._match_length(h_up, seq_len)

        if variant_edit_mask is not None and torch.any(variant_edit_mask):
            boost = self.cfg.variant_residual_gamma * h_stem
            h_up = torch.where(variant_edit_mask.unsqueeze(-1).to(dtype=torch.bool), h_up + boost, h_up)

        hidden = torch.cat([h_up, h_pure], dim=-1)
        return {"hidden_states": hidden, "mid_hidden_states": h_mid}

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        return_token_heads: bool = True,
        return_sequence_embedding: bool = False,
        return_hidden: bool = True,
        variant_edit_mask: torch.Tensor | None = None,
        delta_hidden: torch.Tensor | None = None,
        edit_delta_from_hidden_states: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        if attention_mask is None:
            attention_mask = input_ids.ne(PAD_ID)
        encoded = self.encode(input_ids, attention_mask=attention_mask, variant_edit_mask=variant_edit_mask)
        hidden = encoded["hidden_states"]
        h_mid = encoded["mid_hidden_states"]

        outputs: dict[str, Any] = {}
        if return_hidden:
            outputs.update(encoded)
        else:
            outputs["mid_hidden_states"] = h_mid

        if return_token_heads:
            if self._head_enabled("mlm"):
                outputs["mlm_logits"] = self.mlm_head(hidden)
            if self._head_enabled("phylo100"):
                outputs["phylo100_subst_logits"] = self.phylo100_subst_head(hidden)
            if self._head_enabled("zoo241"):
                outputs["zoo241_subst_logits"] = self.zoo241_subst_head(hidden)
            if self._head_enabled("splice"):
                outputs["splice_class_logits"] = self.splice_class_head(hidden)
                outputs["splice_distance_pred"] = self.splice_distance_head(hidden).squeeze(-1)
            if self._head_enabled("region"):
                outputs["region_logits"] = self.region_head(hidden)
            if self._head_enabled("gnomad"):
                outputs["gnomad_af_pred"] = self.gnomad_af_head(hidden)
                outputs["gnomad_observed_logits"] = self.gnomad_observed_head(hidden)
            if self._head_enabled("counterfactual_snv"):
                cf_logits = self.counterfactual_snv_head(hidden)
                outputs["counterfactual_effect_logits"] = cf_logits.view(
                    *cf_logits.shape[:2],
                    len(SNV_BASES),
                    self.cfg.num_counterfactual_effect_classes,
                )
            if self.regulatory_head is not None and self._head_enabled("regulatory"):
                outputs["regulatory_pred"] = self.regulatory_head(h_mid)

        if delta_hidden is not None and edit_delta_from_hidden_states is not None:
            raise ValueError("Pass either delta_hidden or edit_delta_from_hidden_states, not both.")
        if edit_delta_from_hidden_states is not None:
            if edit_delta_from_hidden_states.shape != hidden.shape:
                raise ValueError(
                    "edit_delta_from_hidden_states must match hidden shape, got "
                    f"{tuple(edit_delta_from_hidden_states.shape)} and {tuple(hidden.shape)}."
                )
            delta_hidden = edit_delta_from_hidden_states - hidden
        if delta_hidden is not None:
            outputs["edit_logits"] = self.edit_localize_head(delta_hidden).squeeze(-1)

        if return_sequence_embedding and self.sequence_embedding_head is not None:
            outputs["sequence_embedding"] = self.sequence_embedding_head(hidden, attention_mask=attention_mask)

        return outputs


def build_beat_v10_model(cfg: BeatV10Config) -> DNAFoundationBeatV10:
    return DNAFoundationBeatV10(cfg)
