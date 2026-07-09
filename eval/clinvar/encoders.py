"""Task-specific encoders for ClinVar fine-tuning."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClinVarVariantEncoder(nn.Module):
    """Allele-conditioned variant encoder for ClinVar one-pass feature extraction."""

    def __init__(self, d_model: int, decoder_dim: int, pad_token_id: int) -> None:
        super().__init__()
        self.pad_token_id = pad_token_id
        self.decoder_context_proj = nn.Linear(decoder_dim, d_model)
        self.allele_length_embeddings = nn.Embedding(8, d_model)
        self.variant_condition_proj = nn.Linear(4 * d_model, d_model)
        self.variant_gate_proj = nn.Linear(4 * d_model, d_model)
        self.variant_out_proj = nn.Linear(d_model, d_model)
        self.variant_act = nn.GELU()

    def _pool_alleles(self, allele_ids: torch.Tensor, token_embedding_weight: torch.Tensor) -> torch.Tensor:
        mask = allele_ids != self.pad_token_id
        embedded = F.embedding(
            allele_ids,
            token_embedding_weight,
            padding_idx=self.pad_token_id,
        )
        masked = embedded * mask.unsqueeze(-1).to(dtype=embedded.dtype)
        lengths = mask.sum(dim=1).clamp_min(1)
        pooled = masked.sum(dim=1) / lengths.unsqueeze(-1)
        length_buckets = lengths.clamp(max=7)
        return pooled + self.allele_length_embeddings(length_buckets)

    def forward(
        self,
        *,
        site_ref: torch.Tensor,
        pooled_decoder_states: torch.Tensor,
        ref_allele_ids: torch.Tensor,
        alt_allele_ids: torch.Tensor,
        token_embedding_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        local_context = self.decoder_context_proj(pooled_decoder_states)
        ref_repr = self._pool_alleles(ref_allele_ids, token_embedding_weight)
        alt_repr = self._pool_alleles(alt_allele_ids, token_embedding_weight)
        condition = torch.cat([site_ref, local_context, ref_repr, alt_repr], dim=-1)
        gate = torch.sigmoid(self.variant_gate_proj(condition))
        conditioned = self.variant_act(self.variant_condition_proj(condition))
        variant_repr = self.variant_out_proj(gate * conditioned + (1.0 - gate) * site_ref)
        return variant_repr, local_context
