from __future__ import annotations

import pytest
import torch

from eval.clinvar.bio_features import BIO_FEATURE_DIM
from eval.clinvar.heads import (
    ALLELE_SCORER_HEAD,
    REGIME_A_BOUNDED_REGIONAL_HEAD,
    REGIME_A_HEAD,
    REGIME_A_HEAD_V8,
    REGIME_A_PLUS_FEATURES_HEAD,
    REGIME_B_HEAD,
    AlleleScorerClinVarHead,
    RegimeABoundedRegionalHead,
    RegimeAHead,
    RegimeAHeadV8,
    RegimeAPlusFeaturesHead,
    RegimeBHead,
    build_head,
)
from eval.clinvar.model import EndToEndClinVarModel


class _TinyAdapter:
    def __init__(self) -> None:
        self._backbone = torch.nn.Identity()
        self._d_model = 4

    @property
    def d_model(self) -> int:
        return self._d_model

    @property
    def backbone(self) -> torch.nn.Module:
        return self._backbone

    def tokenize(self, sequences: list[str]) -> dict[str, torch.Tensor]:
        _ = sequences
        return {
            "input_ids": torch.zeros((2, 4), dtype=torch.long),
            "attention_mask": torch.ones((2, 4), dtype=torch.long),
        }

    def forward_hidden_states(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        _ = batch
        return torch.ones((2, 4, self._d_model), dtype=torch.float32)

    def nuc_window_to_token_bounds(
        self,
        batch: dict[str, torch.Tensor],
        batch_index: int,
        center_nuc: int,
        radius_bp: int,
    ) -> tuple[int, int]:
        _ = (batch, batch_index, center_nuc, radius_bp)
        return (0, 2)


def test_build_head_returns_regime_specific_types() -> None:
    regime_a = build_head("A", d_model=8)
    regime_b = build_head("B", d_model=8)

    assert isinstance(regime_a, RegimeAHead)
    assert isinstance(regime_b, RegimeBHead)
    assert regime_a.head_type == REGIME_A_HEAD
    assert regime_b.head_type == REGIME_B_HEAD


def test_regime_a_head_output_shape() -> None:
    head = RegimeAHead(d_model=8, proj_dim=4, hidden_dim=6, dropout=0.0)
    logits = head(
        torch.randn(3, 8),
        torch.randn(3, 8),
        torch.randn(3, 8),
    )

    assert logits.shape == (3,)
    assert torch.isfinite(logits).all()


def test_build_head_returns_allele_scorer_head() -> None:
    head = build_head(
        "A",
        d_model=8,
        proj_dim=4,
        hidden_dim=6,
        dropout=0.0,
        variant_feature_dim=13,
        head_type=ALLELE_SCORER_HEAD,
    )

    assert isinstance(head, AlleleScorerClinVarHead)
    assert isinstance(head, RegimeAHeadV8)
    assert head.head_type == ALLELE_SCORER_HEAD
    logits = head(torch.randn(2, 8), torch.randn(2, 13), torch.randn(2, 8))
    assert logits.shape == (2,)


def test_build_head_returns_regime_a_v8_head() -> None:
    head = build_head(
        "A",
        d_model=8,
        proj_dim=4,
        hidden_dim=6,
        dropout=0.0,
        variant_feature_dim=13,
        head_type=REGIME_A_HEAD_V8,
    )

    assert isinstance(head, RegimeAHeadV8)
    assert head.head_type == REGIME_A_HEAD_V8
    assert head.site_projection.in_features == 8
    assert head.local_projection.in_features == 8
    assert head.variant_projection.in_features == 13
    logits = head(torch.randn(2, 8), torch.randn(2, 13), torch.randn(2, 8))
    assert logits.shape == (2,)


def test_regime_b_head_requires_bio_features() -> None:
    head = RegimeBHead(d_model=8, proj_dim=4, hidden_dim=6, dropout=0.0)

    with pytest.raises(ValueError, match="requires bio_features"):
        head(
            torch.randn(2, 8),
            torch.randn(2, 8),
            torch.randn(2, 8),
        )


def test_regime_b_head_output_shape() -> None:
    head = RegimeBHead(d_model=8, proj_dim=4, hidden_dim=6, dropout=0.0)
    logits = head(
        torch.randn(2, 8),
        torch.randn(2, 8),
        torch.randn(2, 8),
        bio_features=torch.zeros((2, BIO_FEATURE_DIM), dtype=torch.float32),
    )

    assert logits.shape == (2,)
    assert torch.isfinite(logits).all()


def test_regime_a_plus_features_head_requires_features() -> None:
    head = RegimeAPlusFeaturesHead(
        d_model=8,
        proj_dim=4,
        hidden_dim=6,
        dropout=0.0,
        explicit_feature_dim=3,
    )

    with pytest.raises(ValueError, match="requires explicit bio_features"):
        head(torch.randn(2, 8), torch.randn(2, 8), torch.randn(2, 8))


def test_regime_a_plus_features_head_output_shape() -> None:
    head = build_head(
        "A",
        d_model=8,
        proj_dim=4,
        hidden_dim=6,
        dropout=0.0,
        head_type=REGIME_A_PLUS_FEATURES_HEAD,
        explicit_feature_dim=3,
    )

    assert isinstance(head, RegimeAPlusFeaturesHead)
    assert head.head_type == REGIME_A_PLUS_FEATURES_HEAD
    logits = head(
        torch.randn(2, 8),
        torch.randn(2, 8),
        torch.randn(2, 8),
        bio_features=torch.zeros((2, 3), dtype=torch.float32),
    )
    assert logits.shape == (2,)
    assert torch.isfinite(logits).all()


def test_regime_a_bounded_regional_head_outputs_components_and_only_discounts() -> None:
    head = build_head(
        "A",
        d_model=8,
        proj_dim=4,
        hidden_dim=6,
        dropout=0.0,
        head_type=REGIME_A_BOUNDED_REGIONAL_HEAD,
        explicit_feature_dim=3,
    )

    assert isinstance(head, RegimeABoundedRegionalHead)
    logits = head(
        torch.randn(2, 8),
        torch.randn(2, 8),
        torch.randn(2, 8),
        bio_features=torch.zeros((2, 3), dtype=torch.float32),
    )

    assert logits.shape == (2,)
    assert torch.isfinite(logits).all()
    assert set(head.last_components) == {"molecular_logit", "regional_logit", "regional_discount"}
    assert torch.all(head.last_components["regional_logit"] <= head.last_components["molecular_logit"])
    assert torch.all(head.last_components["regional_discount"] >= 0)


def test_end_to_end_clinvar_model_runs_with_tiny_adapter_regime_b() -> None:
    model = EndToEndClinVarModel(_TinyAdapter(), regime="B", proj_dim=4, hidden_dim=6, dropout=0.0)

    logits = model(
        ["AAAA", "CCCC"],
        ["CAAA", "TCCC"],
        [1, 1],
        bio_features=torch.zeros((2, BIO_FEATURE_DIM), dtype=torch.float32),
    )

    assert logits.shape == (2,)
    assert torch.isfinite(logits).all()


def test_end_to_end_clinvar_model_runs_with_explicit_features() -> None:
    model = EndToEndClinVarModel(
        _TinyAdapter(),
        regime="A",
        proj_dim=4,
        hidden_dim=6,
        dropout=0.0,
        head_type=REGIME_A_PLUS_FEATURES_HEAD,
        explicit_feature_dim=3,
    )

    logits = model(
        ["AAAA", "CCCC"],
        ["CAAA", "TCCC"],
        [1, 1],
        bio_features=torch.zeros((2, 3), dtype=torch.float32),
    )

    assert logits.shape == (2,)
    assert torch.isfinite(logits).all()


def test_regime_a_head_accepts_variant_feature_dim() -> None:
    head = RegimeAHead(d_model=8, variant_feature_dim=27, proj_dim=4, hidden_dim=6, dropout=0.0)

    assert head.emb_projection.in_features == 8
    assert head.emb_projection.out_features == 4
    assert head.variant_projection.in_features == 27
    assert head.variant_projection.out_features == 4

    logits = head(
        torch.randn(3, 8),
        torch.randn(3, 27),
        torch.randn(3, 8),
    )

    assert logits.shape == (3,)
    assert torch.isfinite(logits).all()


def test_regime_a_head_default_variant_feature_dim_is_d_model() -> None:
    head = RegimeAHead(d_model=8, proj_dim=4, hidden_dim=6, dropout=0.0)

    assert head.variant_projection.in_features == 8
    assert head.variant_projection.out_features == 4


def test_regime_b_head_accepts_variant_feature_dim() -> None:
    head = RegimeBHead(d_model=8, variant_feature_dim=27, proj_dim=4, hidden_dim=6, dropout=0.0)

    assert head.variant_projection.in_features == 27
    logits = head(
        torch.randn(2, 8),
        torch.randn(2, 27),
        torch.randn(2, 8),
        bio_features=torch.zeros((2, BIO_FEATURE_DIM), dtype=torch.float32),
    )

    assert logits.shape == (2,)
