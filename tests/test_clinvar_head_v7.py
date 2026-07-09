from __future__ import annotations

import torch
import torch.nn as nn

from eval.clinvar.heads import REGIME_A_HEAD_V7, build_head
from eval.clinvar.heads_v7 import RegimeAHeadV7
from eval.clinvar.model import EndToEndClinVarModel


class _WindowAdapter:
    def __init__(self, device: torch.device) -> None:
        self._device = device
        self._backbone = nn.Identity()
        self._vocab = {"A": 0, "C": 1, "G": 2, "T": 3}
        self._d_model = 8
        self.forward_calls = 0

    @property
    def d_model(self) -> int:
        return self._d_model

    @property
    def backbone(self) -> nn.Module:
        return self._backbone

    def tokenize(self, sequences: list[str]) -> dict[str, torch.Tensor]:
        input_ids = torch.tensor(
            [[self._vocab[base] for base in sequence] for sequence in sequences],
            dtype=torch.long,
            device=self._device,
        )
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }

    def forward_hidden_states(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        self.forward_calls += 1
        base = torch.nn.functional.one_hot(batch["input_ids"], num_classes=4).float()
        return torch.cat([base, base], dim=-1)

    def nuc_window_to_token_bounds(
        self,
        batch: dict[str, torch.Tensor],
        batch_index: int,
        center_nuc: int,
        radius_bp: int,
    ) -> tuple[int, int]:
        seq_len = int(batch["attention_mask"][batch_index].shape[0])
        start = max(0, center_nuc - radius_bp)
        end = min(seq_len, center_nuc + radius_bp + 1)
        return start, max(start + 1, end)


def test_build_head_returns_regime_a_v7_when_requested() -> None:
    head = build_head("A", d_model=8, proj_dim=4, hidden_dim=6, dropout=0.0, head_type=REGIME_A_HEAD_V7)

    assert isinstance(head, RegimeAHeadV7)
    assert head.head_type == REGIME_A_HEAD_V7
    assert head.requires_window_embeddings is True


def test_regime_a_head_v7_output_shape_and_backward() -> None:
    head = RegimeAHeadV7(
        d_model=8,
        proj_dim=4,
        hidden_dim=6,
        head_dropout=0.0,
        variant_feature_dim=5,
        n_heads=4,
        context_radius=2,
    )
    ref_window = torch.randn(3, 5, 8, requires_grad=True)
    alt_window = torch.randn(3, 5, 8, requires_grad=True)
    variant_features = torch.randn(3, 5, requires_grad=True)

    logits = head(ref_window, alt_window, variant_features)
    loss = logits.square().mean()
    loss.backward()

    assert logits.shape == (3,)
    assert torch.isfinite(logits).all()
    assert ref_window.grad is not None
    assert alt_window.grad is not None
    assert variant_features.grad is not None


def test_end_to_end_clinvar_model_runs_with_regime_a_v7_head() -> None:
    adapter = _WindowAdapter(torch.device("cpu"))
    model = EndToEndClinVarModel(
        adapter,
        regime="A",
        proj_dim=4,
        hidden_dim=6,
        dropout=0.0,
        head_type=REGIME_A_HEAD_V7,
    )

    logits = model(
        ["AAAAA", "CCCCC"],
        ["AATAA", "CGCCC"],
        [2, 1],
        ref_alleles=["A", "C"],
        alt_alleles=["T", "G"],
    )
    loss = logits.sum()
    loss.backward()

    assert logits.shape == (2,)
    assert torch.isfinite(logits).all()
    assert adapter.forward_calls == 2
