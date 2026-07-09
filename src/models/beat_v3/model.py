from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
import torch.nn as nn

from src.constants import COMPLEMENT_TABLE, NUM_AA_CLASSES, NUM_STRUCTURE_CLASSES, PAD_ID, VOCAB_SIZE
from src.models.beat_shared import BeatBlock, make_linear_token_head, resolve_chunk_size


@dataclass
class BeatV3Config:
    d_model: int = 384
    n_layers: int = 12
    d_state: int = 64
    expand: int = 2
    headdim: int = 64
    ngroups: int = 1
    rope_fraction: float = 1.0
    chunk_size: int = 16
    is_mimo: bool = True
    mimo_rank: int = 4
    is_outproj_norm: bool = True
    dropout: float = 0.1
    num_region_classes: int = 0
    use_gated_fusion: bool = True
    num_aa_classes: int = NUM_AA_CLASSES


class BeatV3Block(BeatBlock):
    pass


class DNAFoundationBeatV3(nn.Module):
    """Beat-v3: scaled backbone with linear-probe auxiliary heads."""

    def __init__(self, cfg: BeatV3Config) -> None:
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
                BeatV3Block(
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
                )
                for _ in range(cfg.n_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

        self.mlm_head = nn.Linear(cfg.d_model, VOCAB_SIZE, bias=False)
        self.mlm_head.weight = self.token_emb.weight

        self.phylo100_head = make_linear_token_head(cfg.d_model, 1)
        self.phylo470_head = make_linear_token_head(cfg.d_model, 1)
        self.structure_head = make_linear_token_head(cfg.d_model, NUM_STRUCTURE_CLASSES)

        self.region_head: nn.Module | None = None
        if cfg.num_region_classes > 0:
            self.region_head = make_linear_token_head(cfg.d_model, cfg.num_region_classes)

        self.aa_head = make_linear_token_head(cfg.d_model, cfg.num_aa_classes)
        self.codon_phylo_head = make_linear_token_head(cfg.d_model, 1)

        self.register_buffer(
            "_complement_table",
            torch.tensor(COMPLEMENT_TABLE, dtype=torch.long),
            persistent=False,
        )

    def _reverse_complement_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        complement_table = cast(torch.Tensor, self._complement_table)
        return torch.flip(complement_table[input_ids], dims=[-1])

    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        rc_ids = self._reverse_complement_ids(input_ids)
        x = self.token_emb(input_ids)
        x_rc = self.token_emb(rc_ids)

        for block in self.blocks:
            x = block(x, x_rc)

        x = self.final_norm(x)
        x = self.dropout(x)
        return x

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        return_token_heads: bool = True,
        return_sequence_embedding: bool = False,
    ) -> dict[str, torch.Tensor]:
        hidden = self.encode(input_ids)

        outputs: dict[str, torch.Tensor] = {"hidden_states": hidden}
        if return_token_heads:
            outputs["mlm_logits"] = self.mlm_head(hidden)
            outputs["phylo100_pred"] = self.phylo100_head(hidden).squeeze(-1)
            outputs["phylo470_pred"] = self.phylo470_head(hidden).squeeze(-1)
            outputs["structure_logits"] = self.structure_head(hidden)
            outputs["aa_logits"] = self.aa_head(hidden)
            outputs["codon_phylo_pred"] = self.codon_phylo_head(hidden).squeeze(-1)
            if self.region_head is not None:
                outputs["region_logits"] = self.region_head(hidden)
        return outputs


def build_beat_v3_model(cfg: BeatV3Config) -> DNAFoundationBeatV3:
    return DNAFoundationBeatV3(cfg)
