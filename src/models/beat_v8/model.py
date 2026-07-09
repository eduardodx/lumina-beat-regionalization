from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.constants import (
    COMPLEMENT_TABLE,
    NUM_AA_CLASSES,
    NUM_ALLELE_EFFECT_CLASSES,
    NUM_MUTATION_EFFECT_CLASSES,
    NUM_STRUCTURE_CLASSES,
    PAD_ID,
    SNV_BASES,
    VOCAB_SIZE,
)
from src.models.beat_shared import BeatBlock, make_mlp_token_head, resolve_chunk_size
from src.models.beat_v7.local_attn import LocalWindowAttention


@dataclass
class BeatV8Config:
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
    activation_checkpointing: bool = False
    checkpoint_use_reentrant: bool = False
    attention_every_n_blocks: int = 3
    attention_window: int = 256
    attention_n_heads: int = 4
    num_encode_tracks: int = 0
    num_allele_effect_classes: int = NUM_ALLELE_EFFECT_CLASSES
    allele_context_radius: int = 64


class AlleleScorerHead(nn.Module):
    """Directional ref/alt allele scorer trained during leak-free pretraining."""

    def __init__(
        self,
        d_model: int,
        num_effect_classes: int = NUM_ALLELE_EFFECT_CLASSES,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        in_dim = 6 * d_model
        self.repr = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.effect = nn.Linear(d_model, num_effect_classes)
        self.severity = nn.Linear(d_model, 1)

    def forward(
        self,
        site_ref: torch.Tensor,
        site_alt: torch.Tensor,
        local_ref: torch.Tensor,
        local_alt: torch.Tensor,
        ref_base_emb: torch.Tensor,
        alt_base_emb: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        features = torch.cat(
            [
                site_ref,
                site_alt,
                site_alt - site_ref,
                local_alt - local_ref,
                alt_base_emb - ref_base_emb,
                site_ref * site_alt,
            ],
            dim=-1,
        )
        allele_repr = self.repr(features)
        return {
            "allele_repr": allele_repr,
            "allele_effect_logits": self.effect(allele_repr),
            "allele_severity_score": self.severity(allele_repr).squeeze(-1),
        }


class BeatV8Block(BeatBlock):
    pass


class DNAFoundationBeatV8(nn.Module):
    def __init__(self, cfg: BeatV8Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.pad_token_id = PAD_ID

        chunk_size = resolve_chunk_size(cfg)
        self.token_emb = nn.Embedding(VOCAB_SIZE, cfg.d_model, padding_idx=PAD_ID)
        self.blocks = nn.ModuleList(
            [
                BeatV8Block(
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
                    checkpoint_use_reentrant=cfg.checkpoint_use_reentrant,
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
        self.allele_scorer_head = AlleleScorerHead(
            cfg.d_model,
            num_effect_classes=cfg.num_allele_effect_classes,
            dropout=cfg.dropout,
        )
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
        return self.dropout(hidden)

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

    def _pool_local(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor | None,
        positions: torch.Tensor,
        radius: int,
    ) -> torch.Tensor:
        seq_len = hidden.shape[1]
        token_positions = torch.arange(seq_len, device=hidden.device).unsqueeze(0)
        local = (token_positions - positions.unsqueeze(1)).abs() <= radius
        if attention_mask is not None:
            local = local & attention_mask.to(dtype=torch.bool)
        weights = local.unsqueeze(-1).to(dtype=hidden.dtype)
        return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def score_alleles_from_ids(
        self,
        ref_input_ids: torch.Tensor,
        alt_input_ids: torch.Tensor,
        allele_position: torch.Tensor,
        allele_alt_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        alt_attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if alt_input_ids.ndim != 3:
            raise ValueError(f"alt_input_ids must be [B, A, L], got {tuple(alt_input_ids.shape)}.")
        batch_size, num_alts, seq_len = alt_input_ids.shape
        flat_alt_ids = alt_input_ids.reshape(batch_size * num_alts, seq_len)

        ref_hidden = self.encode(ref_input_ids, attention_mask)
        flat_alt_mask = None
        if alt_attention_mask is not None:
            flat_alt_mask = alt_attention_mask.reshape(batch_size * num_alts, seq_len)
        alt_hidden = self.encode(flat_alt_ids, flat_alt_mask).reshape(
            batch_size,
            num_alts,
            seq_len,
            self.cfg.d_model,
        )

        device = ref_hidden.device
        batch_idx = torch.arange(batch_size, device=device)
        allele_position = allele_position.to(device=device, dtype=torch.long).clamp(0, seq_len - 1)
        alt_batch_idx = batch_idx.unsqueeze(1).expand(batch_size, num_alts)
        alt_slot_idx = torch.arange(num_alts, device=device).unsqueeze(0).expand(batch_size, num_alts)
        pos_expanded = allele_position.unsqueeze(1).expand(batch_size, num_alts)

        site_ref = ref_hidden[batch_idx, allele_position]
        site_ref_expanded = site_ref.unsqueeze(1).expand(batch_size, num_alts, self.cfg.d_model)
        site_alt = alt_hidden[alt_batch_idx, alt_slot_idx, pos_expanded]
        local_ref = self._pool_local(ref_hidden, attention_mask, allele_position, self.cfg.allele_context_radius)
        flat_positions = allele_position.unsqueeze(1).expand(batch_size, num_alts).reshape(-1)
        local_alt = self._pool_local(
            alt_hidden.reshape(batch_size * num_alts, seq_len, self.cfg.d_model),
            flat_alt_mask,
            flat_positions,
            self.cfg.allele_context_radius,
        ).reshape(batch_size, num_alts, self.cfg.d_model)

        ref_base_ids = ref_input_ids[batch_idx, allele_position].unsqueeze(1).expand_as(allele_alt_ids)
        ref_base_emb = self.token_emb(ref_base_ids.to(device=device))
        alt_base_emb = self.token_emb(allele_alt_ids.to(device=device, dtype=torch.long))

        scored = self.allele_scorer_head(
            site_ref_expanded,
            site_alt,
            local_ref.unsqueeze(1).expand_as(site_alt),
            local_alt,
            ref_base_emb,
            alt_base_emb,
        )
        swapped = self.allele_scorer_head(
            site_alt,
            site_ref_expanded,
            local_alt,
            local_ref.unsqueeze(1).expand_as(site_alt),
            alt_base_emb,
            ref_base_emb,
        )

        positions = torch.arange(seq_len, device=device).view(1, 1, seq_len)
        far_mask = (positions - allele_position.view(batch_size, 1, 1)).abs() > self.cfg.allele_context_radius
        if attention_mask is not None:
            far_mask = far_mask & attention_mask.to(dtype=torch.bool).unsqueeze(1)
        cosine = F.cosine_similarity(ref_hidden.unsqueeze(1), alt_hidden, dim=-1)
        far_weights = far_mask.to(dtype=cosine.dtype)
        far_distance = ((1.0 - cosine) * far_weights).sum(dim=-1) / far_weights.sum(dim=-1).clamp_min(1.0)

        scored["allele_swap_severity_score"] = swapped["allele_severity_score"]
        scored["allele_far_distance"] = far_distance
        scored["allele_site_ref"] = site_ref
        scored["allele_site_alt"] = site_alt
        scored["allele_local_context"] = local_ref
        scored["allele_local_alt"] = local_alt
        return scored

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        return_token_heads: bool = True,
        return_sequence_embedding: bool = False,
        return_hidden: bool = True,
        allele_ref_input_ids: torch.Tensor | None = None,
        allele_alt_input_ids: torch.Tensor | None = None,
        allele_position: torch.Tensor | None = None,
        allele_alt_ids: torch.Tensor | None = None,
        allele_attention_mask: torch.Tensor | None = None,
        allele_alt_attention_mask: torch.Tensor | None = None,
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
        if (
            allele_ref_input_ids is not None
            and allele_alt_input_ids is not None
            and allele_position is not None
            and allele_alt_ids is not None
        ):
            outputs.update(
                self.score_alleles_from_ids(
                    ref_input_ids=allele_ref_input_ids,
                    alt_input_ids=allele_alt_input_ids,
                    allele_position=allele_position,
                    allele_alt_ids=allele_alt_ids,
                    attention_mask=allele_attention_mask,
                    alt_attention_mask=allele_alt_attention_mask,
                )
            )
        return outputs


def build_beat_v8_model(cfg: BeatV8Config) -> DNAFoundationBeatV8:
    return DNAFoundationBeatV8(cfg)
