"""Classification heads for ClinVar fine-tuning.

Two heads, one per evaluation regime:

Regime A (RegimeAHead):
    Embedding-only.  Projects each model's d_model to a common proj_dim,
    then classifies from [site_ref, variant_repr, local_context].
    Tests backbone representation quality in a model-fair way.

Regime B (RegimeBHead):
    Two-stream: embeddings + explicit biological features.
    Same embedding stream as Regime A, plus a biological feature encoder
    for PhyloP, codon/AA annotations, BLOSUM62, etc.
    Tests practical clinical utility.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from eval.clinvar.bio_features import BIO_FEATURE_DIM
from eval.clinvar.heads_v7 import RegimeAHeadV7

REGIME_A_HEAD = "regime_a"
REGIME_B_HEAD = "regime_b"
REGIME_A_PLUS_FEATURES_HEAD = "regime_a_plus_features"
REGIME_A_BOUNDED_REGIONAL_HEAD = "regime_a_bounded_regional"
REGIME_A_HEAD_V7 = "regime_a_v7"
REGIME_A_HEAD_V8 = "regime_a_v8"
ALLELE_SCORER_HEAD = "allele_scorer"


class RegimeAHead(nn.Module):
    """Embedding-only head for fair backbone comparison (Regime A).

    Embedding features (site_ref, local_context) project through
    ``emb_projection``; ``variant_repr`` projects through a separate
    ``variant_projection`` so its input width can differ from ``d_model``
    when the backbone supplies a native variant feature vector (e.g.
    beat-v6 pretrained-head outputs).

    Input features (3 * proj_dim):
        site_ref        reference embedding at the variant position
        variant_repr    backbone-native variant representation
        local_context   mean-pooled reference in a +-64bp window
    """

    head_type = REGIME_A_HEAD

    def __init__(
        self,
        d_model: int,
        proj_dim: int = 256,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        variant_feature_dim: int | None = None,
    ) -> None:
        super().__init__()
        variant_dim = variant_feature_dim if variant_feature_dim is not None else d_model
        self.emb_projection = nn.Linear(d_model, proj_dim)
        self.variant_projection = nn.Linear(variant_dim, proj_dim)
        self.classifier = nn.Sequential(
            nn.Linear(3 * proj_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    @property
    def projection(self) -> nn.Linear:
        """Compatibility alias for precision-path tests and legacy callers."""
        return self.emb_projection

    def forward(
        self,
        site_ref: Tensor,
        variant_repr: Tensor,
        local_context: Tensor,
        bio_features: Tensor | None = None,
    ) -> Tensor:
        site_ref = self.emb_projection(site_ref)
        local_context = self.emb_projection(local_context)
        variant_repr = self.variant_projection(variant_repr)
        features = torch.cat([site_ref, variant_repr, local_context], dim=-1)
        return self.classifier(features).squeeze(-1)


class RegimeBHead(nn.Module):
    """Two-stream head with embeddings + biological features (Regime B).

    Stream 1: Same embedding features as Regime A, encoded to hidden_dim.
    Stream 2: Explicit biological features (60-dim), encoded to 64-dim.
    Fusion: concatenation -> final classifier.

    The biological features inject coding-region knowledge that compact
    backbones may not have learned during pretraining (codon semantics,
    conservation scores, AA change type, BLOSUM62).
    """

    head_type = REGIME_B_HEAD

    def __init__(
        self,
        d_model: int,
        proj_dim: int = 256,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        variant_feature_dim: int | None = None,
    ) -> None:
        super().__init__()
        variant_dim = variant_feature_dim if variant_feature_dim is not None else d_model
        self.emb_projection = nn.Linear(d_model, proj_dim)
        self.variant_projection = nn.Linear(variant_dim, proj_dim)
        self.emb_encoder = nn.Sequential(
            nn.Linear(3 * proj_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.bio_encoder = nn.Sequential(
            nn.Linear(BIO_FEATURE_DIM, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + 64, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    @property
    def projection(self) -> nn.Linear:
        """Compatibility alias for precision-path utilities."""
        return self.emb_projection

    def forward(
        self,
        site_ref: Tensor,
        variant_repr: Tensor,
        local_context: Tensor,
        bio_features: Tensor | None = None,
    ) -> Tensor:
        if bio_features is None:
            raise ValueError("RegimeBHead requires bio_features tensor")
        site_ref = self.emb_projection(site_ref)
        local_context = self.emb_projection(local_context)
        variant_repr = self.variant_projection(variant_repr)
        emb_features = torch.cat([site_ref, variant_repr, local_context], dim=-1)
        emb_repr = self.emb_encoder(emb_features)
        bio_repr = self.bio_encoder(bio_features)
        combined = torch.cat([emb_repr, bio_repr], dim=-1)
        return self.classifier(combined).squeeze(-1)


class RegimeAPlusFeaturesHead(nn.Module):
    """Regime A embeddings plus explicit scalar features.

    This is the M6 head from the regionalization blueprint: it tests whether
    explicit ABRAOM/gnomAD frequency evidence alone explains regional gains,
    without adding a population adapter.
    """

    head_type = REGIME_A_PLUS_FEATURES_HEAD

    def __init__(
        self,
        d_model: int,
        proj_dim: int = 256,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        variant_feature_dim: int | None = None,
        explicit_feature_dim: int = 0,
    ) -> None:
        super().__init__()
        if explicit_feature_dim <= 0:
            raise ValueError("RegimeAPlusFeaturesHead requires explicit_feature_dim > 0.")
        variant_dim = variant_feature_dim if variant_feature_dim is not None else d_model
        self.explicit_feature_dim = int(explicit_feature_dim)
        self.emb_projection = nn.Linear(d_model, proj_dim)
        self.variant_projection = nn.Linear(variant_dim, proj_dim)
        self.explicit_encoder = nn.Sequential(
            nn.Linear(self.explicit_feature_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(3 * proj_dim + 64, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    @property
    def projection(self) -> nn.Linear:
        """Compatibility alias for precision-path utilities."""
        return self.emb_projection

    def forward(
        self,
        site_ref: Tensor,
        variant_repr: Tensor,
        local_context: Tensor,
        bio_features: Tensor | None = None,
    ) -> Tensor:
        if bio_features is None:
            raise ValueError("RegimeAPlusFeaturesHead requires explicit bio_features tensor")
        if bio_features.shape[-1] != self.explicit_feature_dim:
            raise ValueError(
                "RegimeAPlusFeaturesHead expected "
                f"{self.explicit_feature_dim} explicit features, got {bio_features.shape[-1]}."
            )
        site_ref = self.emb_projection(site_ref)
        local_context = self.emb_projection(local_context)
        variant_repr = self.variant_projection(variant_repr)
        explicit_repr = self.explicit_encoder(bio_features.to(dtype=site_ref.dtype))
        features = torch.cat([site_ref, variant_repr, local_context, explicit_repr], dim=-1)
        return self.classifier(features).squeeze(-1)


class RegimeABoundedRegionalHead(nn.Module):
    """Regime A head with molecular and bounded regional logits.

    The existing training loop consumes the returned regional logit. The head
    also keeps detached molecular/regional components from the most recent
    forward pass, so evaluation code can expose the evidence decomposition.
    """

    head_type = REGIME_A_BOUNDED_REGIONAL_HEAD

    def __init__(
        self,
        d_model: int,
        proj_dim: int = 256,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        variant_feature_dim: int | None = None,
        explicit_feature_dim: int = 0,
        max_regional_discount: float = 4.0,
    ) -> None:
        super().__init__()
        if explicit_feature_dim <= 0:
            raise ValueError("RegimeABoundedRegionalHead requires explicit_feature_dim > 0.")
        variant_dim = variant_feature_dim if variant_feature_dim is not None else d_model
        self.explicit_feature_dim = int(explicit_feature_dim)
        self.max_regional_discount = float(max_regional_discount)
        self.emb_projection = nn.Linear(d_model, proj_dim)
        self.variant_projection = nn.Linear(variant_dim, proj_dim)
        self.explicit_encoder = nn.Sequential(
            nn.Linear(self.explicit_feature_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.shared = nn.Sequential(
            nn.Linear(3 * proj_dim + 64, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.molecular_head = nn.Linear(hidden_dim, 1)
        self.discount_head = nn.Linear(hidden_dim, 1)
        self.last_components: dict[str, Tensor] = {}

    @property
    def projection(self) -> nn.Linear:
        """Compatibility alias for precision-path utilities."""
        return self.emb_projection

    def forward(
        self,
        site_ref: Tensor,
        variant_repr: Tensor,
        local_context: Tensor,
        bio_features: Tensor | None = None,
    ) -> Tensor:
        if bio_features is None:
            raise ValueError("RegimeABoundedRegionalHead requires explicit bio_features tensor")
        if bio_features.shape[-1] != self.explicit_feature_dim:
            raise ValueError(
                "RegimeABoundedRegionalHead expected "
                f"{self.explicit_feature_dim} explicit features, got {bio_features.shape[-1]}."
            )
        site_ref = self.emb_projection(site_ref)
        local_context = self.emb_projection(local_context)
        variant_repr = self.variant_projection(variant_repr)
        explicit_repr = self.explicit_encoder(bio_features.to(dtype=site_ref.dtype))
        features = torch.cat([site_ref, variant_repr, local_context, explicit_repr], dim=-1)
        hidden = self.shared(features)
        molecular_logit = self.molecular_head(hidden).squeeze(-1)
        regional_discount = self.max_regional_discount * torch.sigmoid(self.discount_head(hidden).squeeze(-1))
        regional_logit = molecular_logit - regional_discount
        self.last_components = {
            "molecular_logit": molecular_logit.detach(),
            "regional_logit": regional_logit.detach(),
            "regional_discount": regional_discount.detach(),
        }
        return regional_logit


class RegimeAHeadV8(nn.Module):
    """Beat-v8 Regime A head over reference, local, and allele-native features."""

    head_type = REGIME_A_HEAD_V8

    def __init__(
        self,
        d_model: int,
        proj_dim: int = 256,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        variant_feature_dim: int | None = None,
    ) -> None:
        super().__init__()
        variant_dim = variant_feature_dim if variant_feature_dim is not None else d_model
        self.site_projection = nn.Linear(d_model, proj_dim)
        self.local_projection = nn.Linear(d_model, proj_dim)
        self.variant_projection = nn.Linear(variant_dim, proj_dim)
        self.classifier = nn.Sequential(
            nn.Linear(3 * proj_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    @property
    def projection(self) -> nn.Linear:
        """Compatibility alias for precision-path tests and legacy callers."""
        return self.site_projection

    def forward(
        self,
        site_ref: Tensor,
        variant_repr: Tensor,
        local_context: Tensor,
        bio_features: Tensor | None = None,
    ) -> Tensor:
        _ = bio_features
        site_ref = self.site_projection(site_ref)
        local_context = self.local_projection(local_context)
        variant_repr = self.variant_projection(variant_repr)
        features = torch.cat([site_ref, variant_repr, local_context], dim=-1)
        return self.classifier(features).squeeze(-1)


class AlleleScorerClinVarHead(RegimeAHeadV8):
    """Backward-compatible alias for the Beat-v8 Regime A head."""

    head_type = ALLELE_SCORER_HEAD


def build_head(
    regime: str,
    d_model: int,
    proj_dim: int = 256,
    hidden_dim: int = 128,
    dropout: float = 0.2,
    variant_feature_dim: int | None = None,
    explicit_feature_dim: int = 0,
    head_type: str = REGIME_A_HEAD,
) -> nn.Module:
    """Construct the head for the requested regime."""
    if regime == "A":
        if head_type == REGIME_A_HEAD_V8:
            return RegimeAHeadV8(
                d_model=d_model,
                proj_dim=proj_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
                variant_feature_dim=variant_feature_dim,
            )
        if head_type == ALLELE_SCORER_HEAD:
            return AlleleScorerClinVarHead(
                d_model=d_model,
                proj_dim=proj_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
                variant_feature_dim=variant_feature_dim,
            )
        if head_type == REGIME_A_HEAD_V7:
            return RegimeAHeadV7(
                d_model=d_model,
                proj_dim=proj_dim,
                hidden_dim=hidden_dim,
                head_dropout=dropout,
                variant_feature_dim=variant_feature_dim,
            )
        if head_type == REGIME_A_PLUS_FEATURES_HEAD:
            return RegimeAPlusFeaturesHead(
                d_model=d_model,
                proj_dim=proj_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
                variant_feature_dim=variant_feature_dim,
                explicit_feature_dim=explicit_feature_dim,
            )
        if head_type == REGIME_A_BOUNDED_REGIONAL_HEAD:
            return RegimeABoundedRegionalHead(
                d_model=d_model,
                proj_dim=proj_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
                variant_feature_dim=variant_feature_dim,
                explicit_feature_dim=explicit_feature_dim,
            )
        return RegimeAHead(
            d_model=d_model, proj_dim=proj_dim, hidden_dim=hidden_dim, dropout=dropout,
            variant_feature_dim=variant_feature_dim,
        )
    if regime == "B":
        return RegimeBHead(
            d_model=d_model, proj_dim=proj_dim, hidden_dim=hidden_dim, dropout=dropout,
            variant_feature_dim=variant_feature_dim,
        )
    raise ValueError(f"Unknown regime {regime!r}. Expected 'A' or 'B'.")
