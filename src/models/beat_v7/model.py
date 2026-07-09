from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
import torch.nn as nn

from src.constants import (
    COMPLEMENT_TABLE,
    NUM_AA_CLASSES,
    NUM_MUTATION_EFFECT_CLASSES,
    NUM_STRUCTURE_CLASSES,
    PAD_ID,
    SNV_BASES,
    VOCAB_SIZE,
)
from src.models.beat_shared import BeatBlock, make_mlp_token_head, resolve_chunk_size
from src.models.beat_v7.local_attn import LocalWindowAttention


@dataclass
class BeatV7Config:
    d_model: int = 256
    n_layers: int = 8
    d_state: int = 128
    expand: int = 2
    headdim: int = 64
    ngroups: int = 1
    rope_fraction: float = 1.0
    chunk_size: int = 64
    is_mimo: bool = False
    mimo_rank: int = 4
    is_outproj_norm: bool = True
    dropout: float = 0.05
    num_region_classes: int = 0
    use_gated_fusion: bool = True
    num_aa_classes: int = NUM_AA_CLASSES
    activation_checkpointing: bool = True
    attention_every_n_blocks: int = 3
    attention_window: int = 256
    attention_n_heads: int = 4
    num_encode_tracks: int = 0


class BeatV7Block(BeatBlock):
    pass


class DNAFoundationBeatV7(nn.Module):
    def __init__(self, cfg: BeatV7Config) -> None:
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
                BeatV7Block(
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
                    use_gated_fusion=cfg.use_gated_fusion,
                    activation_checkpointing=cfg.activation_checkpointing,
                )
                for _ in range(cfg.n_layers)
            ]
        )

        num_attn_layers = 0
        if cfg.attention_every_n_blocks > 0:
            num_attn_layers = max(0, (cfg.n_layers - 1) // cfg.attention_every_n_blocks)

        self.attn_norms = nn.ModuleList([nn.LayerNorm(cfg.d_model) for _ in range(num_attn_layers)])
        self.attn_layers = nn.ModuleList(
            [
                LocalWindowAttention(
                    d_model=cfg.d_model,
                    n_heads=cfg.attention_n_heads,
                    window=cfg.attention_window,
                    dropout=cfg.dropout,
                )
                for _ in range(num_attn_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

        self.mlm_head = nn.Linear(cfg.d_model, VOCAB_SIZE, bias=False)
        self.mlm_head.weight = self.token_emb.weight

        self.phylo100_head = make_mlp_token_head(cfg.d_model, 1, cfg.dropout)
        self.phylo470_head = make_mlp_token_head(cfg.d_model, 1, cfg.dropout)
        self.structure_head = make_mlp_token_head(cfg.d_model, NUM_STRUCTURE_CLASSES, cfg.dropout)
        self.region_head: nn.Module | None = None
        if cfg.num_region_classes > 0:
            self.region_head = make_mlp_token_head(cfg.d_model, cfg.num_region_classes, cfg.dropout)

        self.aa_head = make_mlp_token_head(cfg.d_model, cfg.num_aa_classes, cfg.dropout)
        self.codon_phylo_head = make_mlp_token_head(cfg.d_model, 1, cfg.dropout)
        self.mutation_effect_head = make_mlp_token_head(
            cfg.d_model,
            len(SNV_BASES) * NUM_MUTATION_EFFECT_CLASSES,
            cfg.dropout,
        )
        self.codon_head = nn.Linear(cfg.d_model, 64)
        self.encode_head: nn.Module | None = None
        if cfg.num_encode_tracks > 0:
            self.encode_head = nn.Sequential(
                nn.Linear(cfg.d_model, cfg.d_model),
                nn.GELU(),
                nn.Linear(cfg.d_model, cfg.num_encode_tracks),
            )

        self.global_proj = nn.Linear(cfg.d_model, cfg.d_model)

        self.register_buffer(
            "_complement_table",
            torch.tensor(COMPLEMENT_TABLE, dtype=torch.long),
            persistent=False,
        )

    def _reverse_complement_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        complement_table = cast(torch.Tensor, self._complement_table)
        return torch.flip(complement_table[input_ids], dims=[-1])

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        rc_ids = self._reverse_complement_ids(input_ids)
        hidden = self.token_emb(input_ids)
        hidden_rc = self.token_emb(rc_ids)

        for index, block in enumerate(self.blocks):
            hidden = block(hidden, hidden_rc)
            if (
                self.cfg.attention_every_n_blocks > 0
                and (index + 1) % self.cfg.attention_every_n_blocks == 0
                and index < len(self.blocks) - 1
            ):
                attn_index = index // self.cfg.attention_every_n_blocks
                hidden = hidden + self.attn_layers[attn_index](
                    self.attn_norms[attn_index](hidden),
                    attention_mask,
                )

        hidden = self.final_norm(hidden)
        hidden = self.dropout(hidden)
        return hidden

    def pooled_embedding(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if attention_mask is None:
            pooled = hidden_states.mean(dim=1)
        else:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

        return self.global_proj(pooled)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        return_token_heads: bool = True,
        return_sequence_embedding: bool = False,
        return_hidden: bool = True,
    ) -> dict[str, torch.Tensor]:
        hidden = self.encode(input_ids, attention_mask)

        outputs: dict[str, torch.Tensor] = {}
        if return_hidden:
            outputs["hidden_states"] = hidden
        if return_token_heads:
            outputs["mlm_logits"] = self.mlm_head(hidden)
            outputs["phylo100_pred"] = self.phylo100_head(hidden).squeeze(-1)
            outputs["phylo470_pred"] = self.phylo470_head(hidden).squeeze(-1)
            outputs["structure_logits"] = self.structure_head(hidden)
            outputs["aa_logits"] = self.aa_head(hidden)
            outputs["codon_phylo_pred"] = self.codon_phylo_head(hidden).squeeze(-1)
            outputs["mutation_effect_logits"] = self.mutation_effect_head(hidden).reshape(
                hidden.shape[0],
                hidden.shape[1],
                len(SNV_BASES),
                NUM_MUTATION_EFFECT_CLASSES,
            )
            outputs["codon_logits"] = self.codon_head(hidden)
            if self.region_head is not None:
                outputs["region_logits"] = self.region_head(hidden)
            if self.encode_head is not None:
                outputs["encode_pred"] = self.encode_head(hidden)
        if return_sequence_embedding:
            outputs["sequence_embedding"] = self.pooled_embedding(hidden, attention_mask)
        return outputs


def build_beat_v7_model(cfg: BeatV7Config) -> DNAFoundationBeatV7:
    return DNAFoundationBeatV7(cfg)
