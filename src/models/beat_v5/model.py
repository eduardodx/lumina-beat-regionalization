from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from src.constants import (
    NUM_AA_CLASSES,
    NUM_MUTATION_EFFECT_CLASSES,
    NUM_STRUCTURE_CLASSES,
    PAD_ID,
    SNV_BASES,
    VOCAB_SIZE,
)
from src.models.beat_shared import require_mamba3, resolve_chunk_size


def _flip_sequence(x: torch.Tensor) -> torch.Tensor:
    return torch.flip(x, dims=[1])


def _apply_depthwise_conv(conv: nn.Conv1d, x: torch.Tensor) -> torch.Tensor:
    return conv(x.transpose(1, 2)).transpose(1, 2)


class BeatV5Block(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        expand: int = 2,
        headdim: int = 64,
        ngroups: int = 1,
        rope_fraction: float = 1.0,
        chunk_size: int = 16,
        is_mimo: bool = True,
        mimo_rank: int = 4,
        is_outproj_norm: bool = True,
        dropout: float = 0.01,
        local_kernel_size: int = 7,
        activation_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self._activation_checkpointing = activation_checkpointing
        self.norm = nn.LayerNorm(d_model)

        Mamba3 = require_mamba3()
        self.mixer = Mamba3(
            d_model=d_model,
            d_state=d_state,
            expand=expand,
            headdim=headdim,
            ngroups=ngroups,
            rope_fraction=rope_fraction,
            chunk_size=chunk_size,
            is_mimo=is_mimo,
            mimo_rank=mimo_rank,
            is_outproj_norm=is_outproj_norm,
        )

        self.local_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=local_kernel_size,
            padding=local_kernel_size // 2,
            groups=d_model,
            bias=False,
        )
        self.fwd_proj = nn.Linear(d_model, d_model, bias=False)
        self.rev_proj = nn.Linear(d_model, d_model, bias=False)
        self.local_proj = nn.Linear(d_model, d_model, bias=False)
        self.gate_proj = nn.Linear(3 * d_model, 3 * d_model, bias=True)
        self.dropout = nn.Dropout(dropout)

    def _maybe_checkpoint(
        self,
        fn: Callable[[torch.Tensor], torch.Tensor],
        x: torch.Tensor,
    ) -> torch.Tensor:
        if (
            not self._activation_checkpointing
            or not self.training
            or not torch.is_grad_enabled()
            or not x.requires_grad
        ):
            return fn(x)
        return cast(torch.Tensor, activation_checkpoint(fn, x, use_reentrant=True))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h_norm = self.norm(h)
        h_fwd = self._maybe_checkpoint(self.mixer, h_norm)
        h_rev = _flip_sequence(self._maybe_checkpoint(self.mixer, _flip_sequence(h_norm)))

        local_fwd = _apply_depthwise_conv(self.local_conv, h_norm)
        local_rev = _flip_sequence(_apply_depthwise_conv(self.local_conv, _flip_sequence(h_norm)))
        h_local = 0.5 * (local_fwd + local_rev)

        h_cat = torch.cat([h_fwd, h_rev, h_local], dim=-1)
        gate_logits = self.gate_proj(h_cat).reshape(h.shape[0], h.shape[1], 3, h.shape[2])
        gate = torch.softmax(gate_logits, dim=2)

        fused = (
            gate[:, :, 0] * self.fwd_proj(h_fwd)
            + gate[:, :, 1] * self.rev_proj(h_rev)
            + gate[:, :, 2] * self.local_proj(h_local)
        )
        return h + self.dropout(fused)


class BeatV5Decoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        decoder_dim: int,
        kernel_small: int,
        kernel_large: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.small_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_small,
            padding=kernel_small // 2,
            groups=d_model,
            bias=False,
        )
        self.large_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_large,
            padding=kernel_large // 2,
            groups=d_model,
            bias=False,
        )
        self.out_proj = nn.Linear(d_model, decoder_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        h_norm = self.norm(hidden_states)

        small_fwd = _apply_depthwise_conv(self.small_conv, h_norm)
        small_rev = _flip_sequence(_apply_depthwise_conv(self.small_conv, _flip_sequence(h_norm)))
        large_fwd = _apply_depthwise_conv(self.large_conv, h_norm)
        large_rev = _flip_sequence(_apply_depthwise_conv(self.large_conv, _flip_sequence(h_norm)))

        fused = 0.5 * (small_fwd + small_rev) + 0.5 * (large_fwd + large_rev)
        return self.dropout(self.act(self.out_proj(fused)))


class AlleleConditionedMutationHead(nn.Module):
    def __init__(self, decoder_dim: int, dropout: float) -> None:
        super().__init__()
        self.alt_embeddings = nn.Embedding(len(SNV_BASES), decoder_dim)
        self.state_proj = nn.Linear(decoder_dim, decoder_dim)
        self.combine_proj = nn.Linear(decoder_dim, decoder_dim)
        self.out_proj = nn.Linear(decoder_dim, NUM_MUTATION_EFFECT_CLASSES)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, decoder_states: torch.Tensor) -> torch.Tensor:
        _batch_size, _seq_len, hidden = decoder_states.shape
        alt = self.alt_embeddings.weight.view(1, 1, len(SNV_BASES), hidden)
        state = self.state_proj(decoder_states).unsqueeze(2)
        combined = self.act(state + alt)
        combined = self.dropout(self.act(self.combine_proj(combined)))
        return self.out_proj(combined)


@dataclass
class BeatV5Config:
    d_model: int = 384
    n_layers: int = 8
    d_state: int = 64
    expand: int = 2
    headdim: int = 64
    ngroups: int = 1
    rope_fraction: float = 1.0
    chunk_size: int = 16
    is_mimo: bool = False
    mimo_rank: int = 4
    is_outproj_norm: bool = True
    dropout: float = 0.01
    num_region_classes: int = 0
    num_aa_classes: int = NUM_AA_CLASSES
    local_kernel_size: int = 7
    decoder_dim: int = 192
    decoder_kernel_small: int = 3
    decoder_kernel_large: int = 9
    activation_checkpointing: bool = True


class DNAFoundationBeatV5(nn.Module):
    def __init__(self, cfg: BeatV5Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.pad_token_id = PAD_ID

        chunk_size = resolve_chunk_size(cfg)
        self.token_emb = nn.Embedding(
            num_embeddings=VOCAB_SIZE,
            embedding_dim=cfg.d_model,
            padding_idx=PAD_ID,
        )
        self.blocks = nn.ModuleList(
            [
                BeatV5Block(
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
                    dropout=cfg.dropout,
                    local_kernel_size=cfg.local_kernel_size,
                    activation_checkpointing=cfg.activation_checkpointing,
                )
                for _ in range(cfg.n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

        self.decoder = BeatV5Decoder(
            d_model=cfg.d_model,
            decoder_dim=cfg.decoder_dim,
            kernel_small=cfg.decoder_kernel_small,
            kernel_large=cfg.decoder_kernel_large,
            dropout=cfg.dropout,
        )

        self.mlm_head = nn.Linear(cfg.d_model, VOCAB_SIZE, bias=False)
        self.mlm_head.weight = self.token_emb.weight

        self.phylo100_head = nn.Linear(cfg.decoder_dim, 1)
        self.phylo470_head = nn.Linear(cfg.decoder_dim, 1)
        self.structure_head = nn.Linear(cfg.decoder_dim, NUM_STRUCTURE_CLASSES)
        self.region_head: nn.Module | None = None
        if cfg.num_region_classes > 0:
            self.region_head = nn.Linear(cfg.decoder_dim, cfg.num_region_classes)
        self.aa_head = nn.Linear(cfg.decoder_dim, cfg.num_aa_classes)
        self.codon_phylo_head = nn.Linear(cfg.decoder_dim, 1)
        self.mutation_effect_head = AlleleConditionedMutationHead(cfg.decoder_dim, cfg.dropout)

    def _reverse_complement_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        complement = input_ids.new_tensor([0, 4, 3, 2, 1, 5, 6, 7])
        return _flip_sequence(complement[input_ids])

    def _embed_bidirectional(self, input_ids: torch.Tensor) -> torch.Tensor:
        rc_ids = self._reverse_complement_ids(input_ids)
        token_states = self.token_emb(input_ids)
        rc_states = _flip_sequence(self.token_emb(rc_ids))
        return 0.5 * (token_states + rc_states)

    def _should_checkpoint(self, *inputs: torch.Tensor) -> bool:
        if not self.cfg.activation_checkpointing:
            return False
        if not self.training or not torch.is_grad_enabled():
            return False
        return any(t.requires_grad for t in inputs)

    def _checkpoint_if_enabled(
        self,
        fn: Callable[[torch.Tensor], torch.Tensor],
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if not self._should_checkpoint(hidden_states):
            return fn(hidden_states)
        return cast(torch.Tensor, activation_checkpoint(fn, hidden_states, use_reentrant=True))

    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden_states = self._embed_bidirectional(input_ids)
        for block in self.blocks:
            hidden_states = block(hidden_states)
        hidden_states = self.final_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

    def extract_sequence_features(self, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden_states = self.encode(input_ids)
        decoder_states = self._checkpoint_if_enabled(self.decoder, hidden_states)
        return {
            "hidden_states": hidden_states,
            "decoder_states": decoder_states,
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        return_token_heads: bool = True,
        return_sequence_embedding: bool = False,
    ) -> dict[str, torch.Tensor]:
        _ = (attention_mask, return_sequence_embedding)
        features = self.extract_sequence_features(input_ids)
        hidden_states = features["hidden_states"]
        decoder_states = features["decoder_states"]

        outputs: dict[str, torch.Tensor] = {
            "hidden_states": hidden_states,
            "decoder_states": decoder_states,
        }
        if return_token_heads:
            outputs["mlm_logits"] = self.mlm_head(hidden_states)
            outputs["phylo100_pred"] = self.phylo100_head(decoder_states).squeeze(-1)
            outputs["phylo470_pred"] = self.phylo470_head(decoder_states).squeeze(-1)
            outputs["structure_logits"] = self.structure_head(decoder_states)
            outputs["aa_logits"] = self.aa_head(decoder_states)
            outputs["codon_phylo_pred"] = self.codon_phylo_head(decoder_states).squeeze(-1)
            outputs["mutation_effect_logits"] = self.mutation_effect_head(decoder_states)
            if self.region_head is not None:
                outputs["region_logits"] = self.region_head(decoder_states)
        return outputs


def build_beat_v5_model(cfg: BeatV5Config) -> DNAFoundationBeatV5:
    return DNAFoundationBeatV5(cfg)
