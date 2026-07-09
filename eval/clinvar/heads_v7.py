from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class RegimeAHeadV7(nn.Module):
    """Cross-attention regime-A head over local hidden-state windows."""

    head_type = "regime_a_v7"
    requires_window_embeddings = True

    def __init__(
        self,
        d_model: int,
        proj_dim: int,
        hidden_dim: int,
        head_dropout: float,
        variant_feature_dim: int | None = None,
        n_heads: int = 4,
        context_radius: int = 64,
    ) -> None:
        super().__init__()
        variant_dim = variant_feature_dim if variant_feature_dim is not None else d_model
        self.context_radius = context_radius
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=head_dropout,
            batch_first=True,
        )
        self.context_projection = nn.Linear(d_model, proj_dim)
        self.variant_projection = nn.Linear(variant_dim, proj_dim)
        self.classifier = nn.Sequential(
            nn.Linear(4 * proj_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _cross_attend(self, query_window: Tensor, context_window: Tensor) -> Tensor:
        center_index = query_window.shape[1] // 2
        query = query_window[:, center_index : center_index + 1, :]
        attended, _weights = self.cross_attn(query=query, key=context_window, value=context_window, need_weights=False)
        return attended.squeeze(1)

    def forward(
        self,
        ref_emb_window: Tensor,
        alt_emb_window: Tensor,
        variant_features: Tensor,
    ) -> Tensor:
        q_ref = self._cross_attend(ref_emb_window, ref_emb_window)
        q_alt = self._cross_attend(alt_emb_window, alt_emb_window)
        delta = q_alt - q_ref

        features = torch.cat(
            [
                self.context_projection(q_ref),
                self.context_projection(q_alt),
                self.context_projection(delta),
                self.variant_projection(variant_features),
            ],
            dim=-1,
        )
        return self.classifier(features).squeeze(-1)
