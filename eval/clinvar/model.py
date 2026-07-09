"""End-to-end ClinVar pathogenicity model: backbone + classification head.

Orchestrates: tokenize -> backbone -> feature extraction -> head -> logits.
Supports both Regime A (embeddings only) and Regime B (embeddings + bio features).
"""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import Tensor, nn

from eval.clinvar.adapters import TokenizedBatch, compute_native_feature_dim
from eval.clinvar.encoders import ClinVarVariantEncoder
from eval.clinvar.heads import build_head

LOCAL_POOL_RADIUS = 64  # +-64bp window for local context


def _nuc_offset_to_token_index(
    adapter: Any,
    batch: TokenizedBatch,
    batch_index: int,
    nuc_offset: int,
) -> int:
    """Convert a nucleotide offset to the corresponding token index."""
    start, end = adapter.nuc_window_to_token_bounds(batch, batch_index, nuc_offset, radius_bp=1)
    token_idx = (start + end) // 2
    seq_len = int(batch["attention_mask"][batch_index].shape[0])
    return min(max(token_idx, 0), seq_len - 1)


def _masked_mean_pool_with_bounds(
    hidden_states: Tensor,
    attention_mask: Tensor,
    starts: Tensor,
    ends: Tensor,
) -> Tensor:
    """Mean-pool over valid positions within per-sample bounds [B, D]."""
    positions = torch.arange(hidden_states.shape[1], device=hidden_states.device).unsqueeze(0)
    region_mask = (positions >= starts.unsqueeze(1)) & (positions < ends.unsqueeze(1))
    valid_mask = (attention_mask > 0) & region_mask
    mask = valid_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
    return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


def _fixed_window(hidden_states: Tensor, centers: Tensor, radius: int) -> Tensor:
    offsets = torch.arange(-radius, radius + 1, device=hidden_states.device)
    indices = centers.unsqueeze(1) + offsets.unsqueeze(0)
    valid_mask = (indices >= 0) & (indices < hidden_states.shape[1])
    clamped_indices = indices.clamp(0, hidden_states.shape[1] - 1)
    gathered = hidden_states.gather(
        1,
        clamped_indices.unsqueeze(-1).expand(-1, -1, hidden_states.shape[-1]),
    )
    return gathered * valid_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)


def _infer_variant_alleles(
    ref_seqs: list[str],
    alt_seqs: list[str],
    variant_offsets: list[int],
) -> tuple[list[str], list[str]]:
    ref_alleles = [ref_seq[offset] for ref_seq, offset in zip(ref_seqs, variant_offsets, strict=True)]
    alt_alleles = [alt_seq[offset] for alt_seq, offset in zip(alt_seqs, variant_offsets, strict=True)]
    return ref_alleles, alt_alleles


class EndToEndClinVarModel(nn.Module):
    """Backbone + classification head for ClinVar pathogenicity.

    The adapter is NOT a submodule; the backbone is registered directly
    so ``model.parameters()`` captures it for LoRA and optimiser setup.
    """

    def __init__(
        self,
        adapter: Any,
        regime: str = "A",
        proj_dim: int = 256,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        head_type: str = "regime_a",
        explicit_feature_dim: int = 0,
    ) -> None:
        super().__init__()
        self._adapter = adapter
        self._regime = regime

        self.backbone: nn.Module = adapter.backbone
        self.variant_encoder: nn.Module = self._build_variant_encoder(adapter)
        native_selection = getattr(adapter, "native_variant_head_selection", None)
        variant_feature_dim: int | None = (
            compute_native_feature_dim(native_selection, adapter.d_model) if native_selection else None
        )
        self.head = build_head(
            regime=regime,
            d_model=adapter.d_model,
            proj_dim=proj_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            variant_feature_dim=variant_feature_dim,
            explicit_feature_dim=explicit_feature_dim,
            head_type=head_type,
        )

    @property
    def adapter(self) -> Any:
        return self._adapter

    @property
    def regime(self) -> str:
        return self._regime

    def _build_variant_encoder(self, adapter: Any) -> nn.Module:
        cfg = getattr(adapter.backbone, "cfg", None)
        decoder_dim = getattr(cfg, "decoder_dim", None)
        pad_token_id = getattr(adapter.backbone, "pad_token_id", None)
        if isinstance(decoder_dim, int) and isinstance(pad_token_id, int):
            return ClinVarVariantEncoder(
                d_model=adapter.d_model,
                decoder_dim=decoder_dim,
                pad_token_id=pad_token_id,
            )
        return nn.Identity()

    def _extract_features(
        self,
        ref_seqs: list[str],
        alt_seqs: list[str],
        variant_offsets: list[int],
        ref_alleles: list[str] | None = None,
        alt_alleles: list[str] | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute (site_ref, variant_repr, local_context) from backbone."""
        adapter = self._adapter
        if ref_alleles is None or alt_alleles is None:
            ref_alleles, alt_alleles = _infer_variant_alleles(ref_seqs, alt_seqs, variant_offsets)

        native_selection = getattr(adapter, "native_variant_head_selection", None)
        if native_selection and hasattr(adapter, "forward_native_variant_features"):
            return cast(Any, adapter).forward_native_variant_features(
                ref_seqs, alt_seqs, variant_offsets, ref_alleles, alt_alleles,
            )

        if (
            isinstance(self.variant_encoder, ClinVarVariantEncoder)
            and hasattr(adapter, "forward_sequence_features")
            and hasattr(adapter, "encode_alleles")
        ):
            ref_batch = adapter.tokenize(ref_seqs)
            features = cast(Any, adapter).forward_sequence_features(ref_batch)
            hidden_states = cast(Tensor, features["hidden_states"])
            decoder_states = cast(Tensor, features["decoder_states"])

            batch_size = hidden_states.shape[0]
            device = hidden_states.device
            batch_idx = torch.arange(batch_size, device=device)
            ref_token_indices = torch.tensor(
                [_nuc_offset_to_token_index(adapter, ref_batch, i, off) for i, off in enumerate(variant_offsets)],
                dtype=torch.long,
                device=device,
            )
            site_ref = hidden_states[batch_idx, ref_token_indices]

            local_bounds = [
                adapter.nuc_window_to_token_bounds(ref_batch, i, off, LOCAL_POOL_RADIUS)
                for i, off in enumerate(variant_offsets)
            ]
            local_starts = torch.tensor([s for s, _ in local_bounds], dtype=torch.long, device=device)
            local_ends = torch.tensor([e for _, e in local_bounds], dtype=torch.long, device=device)
            pooled_decoder = _masked_mean_pool_with_bounds(
                decoder_states,
                ref_batch["attention_mask"],
                local_starts,
                local_ends,
            )

            ref_allele_ids = cast(Any, adapter).encode_alleles(ref_alleles)
            alt_allele_ids = cast(Any, adapter).encode_alleles(alt_alleles)
            token_emb = getattr(self.backbone, "token_emb", None)
            if not isinstance(token_emb, nn.Embedding):
                raise TypeError("Single-pass variant encoding requires backbone.token_emb to be an nn.Embedding.")
            variant_repr, local_context = self.variant_encoder(
                site_ref=site_ref,
                pooled_decoder_states=pooled_decoder,
                ref_allele_ids=ref_allele_ids,
                alt_allele_ids=alt_allele_ids,
                token_embedding_weight=token_emb.weight,
            )
            return site_ref, cast(Tensor, variant_repr), cast(Tensor, local_context)

        if hasattr(adapter, "extract_variant_features"):
            return cast(
                Any,
                adapter,
            ).extract_variant_features(
                ref_seqs,
                alt_seqs,
                variant_offsets,
                ref_alleles,
                alt_alleles,
            )

        ref_batch = adapter.tokenize(ref_seqs)
        alt_batch = adapter.tokenize(alt_seqs)

        ref_hidden = adapter.forward_hidden_states(ref_batch)   # [B, L, D]
        alt_hidden = adapter.forward_hidden_states(alt_batch)   # [B, L, D]

        batch_size = ref_hidden.shape[0]
        device = ref_hidden.device
        batch_idx = torch.arange(batch_size, device=device)

        # Variant token indices
        ref_token_indices = torch.tensor(
            [_nuc_offset_to_token_index(adapter, ref_batch, i, off) for i, off in enumerate(variant_offsets)],
            dtype=torch.long, device=device,
        )
        alt_token_indices = torch.tensor(
            [_nuc_offset_to_token_index(adapter, alt_batch, i, off) for i, off in enumerate(variant_offsets)],
            dtype=torch.long, device=device,
        )

        site_ref = ref_hidden[batch_idx, ref_token_indices]       # [B, D]
        site_alt = alt_hidden[batch_idx, alt_token_indices]       # [B, D]
        variant_repr = site_alt - site_ref                       # [B, D]

        # Local context: mean-pooled reference in +-64bp window
        # Convert nucleotide offsets to token bounds (critical for subword models like DNABERT-2)
        local_bounds = [
            adapter.nuc_window_to_token_bounds(ref_batch, i, off, LOCAL_POOL_RADIUS)
            for i, off in enumerate(variant_offsets)
        ]
        local_starts = torch.tensor(
            [s for s, _ in local_bounds], dtype=torch.long, device=device,
        )
        local_ends = torch.tensor(
            [e for _, e in local_bounds], dtype=torch.long, device=device,
        )
        local_context = _masked_mean_pool_with_bounds(
            ref_hidden, ref_batch["attention_mask"], local_starts, local_ends,
        )  # [B, D]

        return site_ref, variant_repr, local_context

    def forward(
        self,
        ref_seqs: list[str],
        alt_seqs: list[str],
        variant_offsets: list[int],
        ref_alleles: list[str] | None = None,
        alt_alleles: list[str] | None = None,
        bio_features: Tensor | None = None,
    ) -> Tensor:
        """Full forward pass returning logits [B]."""
        if getattr(self.head, "requires_window_embeddings", False):
            adapter = self._adapter
            if ref_alleles is None or alt_alleles is None:
                ref_alleles, alt_alleles = _infer_variant_alleles(ref_seqs, alt_seqs, variant_offsets)

            ref_batch = adapter.tokenize(ref_seqs)
            alt_batch = adapter.tokenize(alt_seqs)
            ref_hidden = adapter.forward_hidden_states(ref_batch)
            alt_hidden = adapter.forward_hidden_states(alt_batch)

            batch_size = ref_hidden.shape[0]
            device = ref_hidden.device
            ref_token_indices = torch.tensor(
                [_nuc_offset_to_token_index(adapter, ref_batch, i, off) for i, off in enumerate(variant_offsets)],
                dtype=torch.long,
                device=device,
            )
            alt_token_indices = torch.tensor(
                [_nuc_offset_to_token_index(adapter, alt_batch, i, off) for i, off in enumerate(variant_offsets)],
                dtype=torch.long,
                device=device,
            )
            context_radius = int(getattr(self.head, "context_radius", LOCAL_POOL_RADIUS))
            ref_windows = _fixed_window(ref_hidden, ref_token_indices, context_radius)
            alt_windows = _fixed_window(alt_hidden, alt_token_indices, context_radius)

            native_selection = getattr(adapter, "native_variant_head_selection", None)
            if native_selection and hasattr(adapter, "forward_native_variant_features"):
                _site_ref, variant_features, _local_context = cast(Any, adapter).forward_native_variant_features(
                    ref_seqs,
                    alt_seqs,
                    variant_offsets,
                    ref_alleles,
                    alt_alleles,
                )
            else:
                batch_idx = torch.arange(batch_size, device=device)
                site_ref = ref_hidden[batch_idx, ref_token_indices]
                site_alt = alt_hidden[batch_idx, alt_token_indices]
                variant_features = site_alt - site_ref

            return cast(Any, self.head)(ref_windows, alt_windows, variant_features)

        site_ref, variant_repr, local_context = self._extract_features(
            ref_seqs, alt_seqs, variant_offsets,
            ref_alleles=ref_alleles,
            alt_alleles=alt_alleles,
        )
        return cast(Any, self.head)(
            site_ref, variant_repr, local_context, bio_features=bio_features,
        )
