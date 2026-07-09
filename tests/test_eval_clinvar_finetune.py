from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from eval.clinvar.adapters import (
    FineTuneLuminaAdapter,
    _huggingface_hub_kwargs,
    _normalize_lumina_model_config_for_precision,
    _resolve_caduceus_hidden_dim,
    compute_native_feature_dim,
)
from eval.clinvar.config import FineTuneConfig
from eval.clinvar.dataset import CachedVariant
from eval.clinvar.encoders import ClinVarVariantEncoder
from eval.clinvar.heads import RegimeAHead
from eval.clinvar.lora import (
    LoRALinear,
    apply_lora,
    enable_layernorm_training,
    freeze_native_feature_heads,
)
from eval.clinvar.model import EndToEndClinVarModel
from eval.clinvar.run import parse_args
from eval.clinvar.train import (
    _build_cosine_schedule,
    _collect_task_parameter_groups,
    _save_checkpoint,
    run_finetune,
)
from src.constants import (
    DNA_VOCAB,
    NUM_AA_CLASSES,
    NUM_ALLELE_EFFECT_CLASSES,
    NUM_MUTATION_EFFECT_CLASSES,
    SNV_BASES,
    VOCAB_SIZE,
)
from src.precision import PrecisionPolicy


def test_lora_linear_zero_init_preserves_base_forward() -> None:
    base = nn.Linear(4, 3, bias=True)
    layer = LoRALinear(base, rank=2, alpha=8.0, dropout=0.0)
    x = torch.randn(5, 4)

    assert torch.allclose(layer(x), base(x))
    assert layer.base.weight.requires_grad is False
    assert layer.lora_a.requires_grad is True
    assert layer.lora_b.requires_grad is True


def test_lora_linear_tracks_base_device_and_dtype() -> None:
    base = nn.Linear(4, 3, bias=False).to(dtype=torch.float64)
    layer = LoRALinear(base, rank=2, alpha=8.0, dropout=0.0)

    assert layer.lora_a.device == base.weight.device
    assert layer.lora_b.device == base.weight.device
    assert layer.lora_a.dtype == base.weight.dtype
    assert layer.lora_b.dtype == base.weight.dtype


def test_lora_linear_exposes_linear_metadata() -> None:
    base = nn.Linear(4, 3, bias=True)
    layer = LoRALinear(base, rank=2, alpha=8.0, dropout=0.0)

    assert layer.weight is base.weight
    assert layer.bias is base.bias
    assert layer.in_features == base.in_features
    assert layer.out_features == base.out_features


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_lora_linear_tracks_cuda_device() -> None:
    base = nn.Linear(4, 3, bias=False, device="cuda")
    layer = LoRALinear(base, rank=2, alpha=8.0, dropout=0.0)

    assert layer.lora_a.device == base.weight.device
    assert layer.lora_b.device == base.weight.device


class _LuminaLikeBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        block = nn.Module()
        block.norm = nn.LayerNorm(4)
        block.fuse = nn.Linear(4, 4)
        self.blocks = nn.ModuleList([block])
        self.final_norm = nn.LayerNorm(4)
        self.decoder = nn.Module()
        self.decoder.out_proj = nn.Linear(4, 4)
        self.mlm_head = nn.Linear(4, 4)
        self.mutation_effect_head = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 3))
        self.global_proj = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))


class _NTv3LikeBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        layer = nn.Module()
        layer.q_proj = nn.Linear(4, 4)
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList([layer])
        self.lm_head = nn.Linear(4, 4)


class _NTv3StyleProjection(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mha_output = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.mha_output.weight.dtype)
        return self.mha_output(x)


class _CaduceusLikeBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        block = nn.Module()
        block.out_proj = nn.Linear(4, 4)
        self.backbone = nn.Module()
        self.backbone.blocks = nn.ModuleList([block])
        self.classifier = nn.Linear(4, 4)


class _DNABERT2LikeBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        layer = nn.Module()
        layer.dense = nn.Linear(4, 4)
        self.bert = nn.ModuleDict(
            {
                "encoder": nn.ModuleDict(
                    {
                        "layer": nn.ModuleList([layer]),
                    }
                ),
            }
        )
        self.cls = nn.Sequential(nn.Linear(4, 4))


@pytest.mark.parametrize(
    ("backbone_cls", "target_name", "excluded_name"),
    [
        (_LuminaLikeBackbone, "blocks.0.fuse", "mlm_head"),
        (_NTv3LikeBackbone, "encoder.layers.0.q_proj", "lm_head"),
        (_CaduceusLikeBackbone, "backbone.blocks.0.out_proj", "classifier"),
        (_DNABERT2LikeBackbone, "bert.encoder.layer.0.dense", "cls.0"),
    ],
)
def test_apply_lora_targets_family_like_backbones(
    backbone_cls: type[nn.Module],
    target_name: str,
    excluded_name: str,
) -> None:
    backbone = backbone_cls()
    for param in backbone.parameters():
        param.requires_grad = False

    summary = apply_lora(backbone, rank=2, alpha=4.0, dropout=0.0)
    modules = dict(backbone.named_modules())
    trainable_names = {name for name, param in backbone.named_parameters() if param.requires_grad}

    assert summary.module_count >= 1
    assert isinstance(modules[target_name], LoRALinear)
    assert isinstance(modules[excluded_name], nn.Linear)
    assert trainable_names
    assert all("lora_" in name for name in trainable_names)


def test_apply_lora_preserves_ntv3_style_weight_inspection() -> None:
    module = _NTv3StyleProjection()
    x = torch.randn(2, 4)
    expected = module(x)

    apply_lora(module, rank=2, alpha=4.0, dropout=0.0)
    actual = module(x)

    assert isinstance(module.mha_output, LoRALinear)
    assert torch.allclose(actual, expected)


def test_enable_layernorm_training_unfreezes_only_layernorms_and_lora() -> None:
    backbone = _LuminaLikeBackbone()
    for param in backbone.parameters():
        param.requires_grad = False

    apply_lora(backbone, rank=2, alpha=4.0, dropout=0.0)
    norm_names = enable_layernorm_training(backbone)
    trainable_names = {name for name, param in backbone.named_parameters() if param.requires_grad}

    assert norm_names
    assert any(name in norm_names for name in trainable_names)
    assert any("lora_" in name for name in trainable_names)
    assert all("lora_" in name or name in norm_names for name in trainable_names)


def test_apply_lora_keeps_beat_v5_decoder_trainable_but_excludes_mutation_effect_head() -> None:
    backbone = _LuminaLikeBackbone()
    for param in backbone.parameters():
        param.requires_grad = False

    summary = apply_lora(backbone, rank=2, alpha=4.0, dropout=0.0)
    modules = dict(backbone.named_modules())

    assert "decoder.out_proj" in summary.module_names
    assert "variant_out_proj" not in summary.module_names
    assert "mutation_effect_head.0" not in summary.module_names
    assert "mutation_effect_head.1" not in summary.module_names
    assert isinstance(modules["decoder.out_proj"], LoRALinear)
    assert isinstance(modules["mutation_effect_head.0"], nn.Linear)


class _ToyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pre_norm = nn.LayerNorm(4)
        self.proj = nn.Linear(4, 4, bias=False)
        self.post_norm = nn.LayerNorm(4)
        self.mlm_head = nn.Linear(4, 4, bias=False)

    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        one_hot = F.one_hot(input_ids, num_classes=4).float()
        hidden = self.pre_norm(one_hot)
        hidden = self.proj(hidden)
        return self.post_norm(hidden)


class _ToyAdapter:
    def __init__(self, device: torch.device) -> None:
        self._device = device
        self._backbone = _ToyBackbone().to(device)
        self._vocab = {"A": 0, "C": 1, "G": 2, "T": 3}
        self._d_model = 4
        self.forward_calls = 0

    @property
    def d_model(self) -> int:
        return self._d_model

    @property
    def backbone(self) -> nn.Module:
        return self._backbone

    def tokenize(self, sequences: list[str]) -> dict[str, torch.Tensor]:
        input_ids = torch.tensor(
            [[self._vocab[base] for base in seq] for seq in sequences],
            dtype=torch.long,
            device=self._device,
        )
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }

    def forward_hidden_states(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        self.forward_calls += 1
        return self._backbone.encode(batch["input_ids"])

    def nuc_window_to_token_bounds(
        self,
        batch: dict[str, torch.Tensor],
        batch_index: int,
        center_nuc: int,
        radius_bp: int,
    ) -> tuple[int, int]:
        seq_len = int(batch["attention_mask"][batch_index].shape[0])
        start = max(0, center_nuc - radius_bp)
        end = min(seq_len, center_nuc + radius_bp)
        return start, max(start + 1, end)


class _SinglePassFeatureAdapter:
    def __init__(self) -> None:
        self._device = torch.device("cpu")
        self._backbone = _ToyBeatV5Backbone()
        self._vocab = {"A": 1, "C": 2, "G": 3, "T": 4}
        self._d_model = 4

    @property
    def d_model(self) -> int:
        return self._d_model

    @property
    def backbone(self) -> nn.Module:
        return self._backbone

    def tokenize(self, sequences: list[str]) -> dict[str, torch.Tensor]:
        max_len = max(len(sequence) for sequence in sequences)
        input_ids = []
        attention_mask = []
        for sequence in sequences:
            ids = [self._vocab[base] for base in sequence]
            pad_len = max_len - len(ids)
            input_ids.append(ids + [0] * pad_len)
            attention_mask.append([1] * len(ids) + [0] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long, device=self._device),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long, device=self._device),
        }

    def forward_hidden_states(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        _ = batch
        raise AssertionError("single-pass adapter should not use paired hidden-state extraction")

    def nuc_window_to_token_bounds(
        self,
        batch: dict[str, torch.Tensor],
        batch_index: int,
        center_nuc: int,
        radius_bp: int,
    ) -> tuple[int, int]:
        _ = (batch, batch_index, center_nuc, radius_bp)
        return (0, 2)

    def forward_sequence_features(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self._backbone.extract_sequence_features(batch["input_ids"])

    def encode_alleles(self, alleles: list[str]) -> torch.Tensor:
        return self.tokenize(alleles)["input_ids"]


class _Bfloat16FeatureAdapter:
    def __init__(self) -> None:
        self._backbone = nn.Identity()
        self._d_model = 4

    @property
    def d_model(self) -> int:
        return self._d_model

    @property
    def backbone(self) -> nn.Module:
        return self._backbone

    def tokenize(self, sequences: list[str]) -> dict[str, torch.Tensor]:
        _ = sequences
        return {
            "input_ids": torch.zeros((2, 4), dtype=torch.long),
            "attention_mask": torch.ones((2, 4), dtype=torch.long),
        }

    def forward_hidden_states(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        _ = batch
        return torch.arange(32, dtype=torch.bfloat16).reshape(2, 4, self._d_model)

    def nuc_window_to_token_bounds(
        self,
        batch: dict[str, torch.Tensor],
        batch_index: int,
        center_nuc: int,
        radius_bp: int,
    ) -> tuple[int, int]:
        _ = (batch, batch_index, center_nuc, radius_bp)
        return (0, 2)


class _ToyBeatV5Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cfg = type("Cfg", (), {"d_model": 4, "decoder_dim": 2})()
        self.pad_token_id = 0
        self.token_emb = nn.Embedding(8, 4, padding_idx=0)
        self.mutation_effect_head = nn.Sequential(nn.Linear(2, 4), nn.Linear(4, 3))
        self.extract_calls = 0

    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        one_hot = F.one_hot(input_ids.clamp(max=3), num_classes=4).float()
        return one_hot

    def extract_sequence_features(self, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        self.extract_calls += 1
        hidden_states = self.encode(input_ids)
        decoder_states = hidden_states[..., :2]
        return {
            "hidden_states": hidden_states,
            "decoder_states": decoder_states,
        }


class _CountingHead(nn.Module):
    """Linear module that tracks how many times it was called."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.call_count = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.call_count += 1
        return self.linear(x)


class _ToyBeatV6Backbone(nn.Module):
    """Minimal stand-in for beat-v6 that exposes the four pretrained heads.

    Matches the interface the native Regime A path depends on: ``cfg.d_model``,
    no ``decoder_dim``, ``pad_token_id``, ``encode(input_ids)``, and
    ``mutation_effect_head``/``aa_head``/``codon_phylo_head``/``phylo470_head``
    modules emitting the per-position widths the real checkpoints produce.
    """

    def __init__(self, d_model: int = 4, vocab_size: int = VOCAB_SIZE) -> None:
        super().__init__()
        self.cfg = type("Cfg", (), {"d_model": d_model})()
        self.pad_token_id = 0
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.encode_calls = 0
        self.mutation_effect_head = _CountingHead(
            d_model, len(SNV_BASES) * NUM_MUTATION_EFFECT_CLASSES,
        )
        self.aa_head = _CountingHead(d_model, NUM_AA_CLASSES)
        self.codon_phylo_head = _CountingHead(d_model, 1)
        self.phylo470_head = _CountingHead(d_model, 1)
        self.mlm_head = _CountingHead(d_model, vocab_size)

    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        self.encode_calls += 1
        return self.token_emb(input_ids)


class _ToyAlleleScorerHead(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class _ToyBeatV8Backbone(_ToyBeatV6Backbone):
    def __init__(self, d_model: int = 4, vocab_size: int = VOCAB_SIZE) -> None:
        super().__init__(d_model=d_model, vocab_size=vocab_size)
        self.cfg = type("Cfg", (), {"d_model": d_model, "allele_context_radius": 1})()
        self.allele_scorer_head = _ToyAlleleScorerHead(d_model)
        self.allele_score_calls = 0

    def score_alleles_from_ids(
        self,
        ref_input_ids: torch.Tensor,
        alt_input_ids: torch.Tensor,
        allele_position: torch.Tensor,
        allele_alt_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        alt_attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        _ = (attention_mask, alt_attention_mask)
        self.allele_score_calls += 1
        hidden = self.encode(ref_input_ids)
        batch_size, num_alts, seq_len = alt_input_ids.shape
        alt_hidden = self.encode(alt_input_ids.reshape(batch_size * num_alts, seq_len)).reshape(
            batch_size,
            num_alts,
            seq_len,
            hidden.shape[-1],
        )
        batch_idx = torch.arange(hidden.shape[0])
        alt_idx = torch.arange(num_alts).unsqueeze(0).expand(batch_size, num_alts)
        pos = allele_position.unsqueeze(1).expand(batch_size, num_alts)
        site = hidden[batch_idx, allele_position]
        site_alt = alt_hidden[batch_idx.unsqueeze(1), alt_idx, pos]
        local_ref = hidden.mean(dim=1)
        local_alt = alt_hidden.mean(dim=2)
        allele_repr = self.allele_scorer_head(site).unsqueeze(1)
        logits = torch.nn.functional.one_hot(
            allele_alt_ids.squeeze(1).clamp(min=0, max=NUM_ALLELE_EFFECT_CLASSES - 1),
            num_classes=NUM_ALLELE_EFFECT_CLASSES,
        ).float().unsqueeze(1)
        severity = allele_alt_ids.float().clamp(max=4).unsqueeze(-1) / 4.0
        return {
            "allele_repr": allele_repr,
            "allele_effect_logits": logits,
            "allele_severity_score": severity.squeeze(-1),
            "allele_swap_severity_score": -severity.squeeze(-1),
            "allele_far_distance": (site_alt - site.unsqueeze(1)).abs().mean(dim=-1),
            "allele_site_ref": site,
            "allele_site_alt": site_alt,
            "allele_local_context": local_ref,
            "allele_local_alt": local_alt,
        }


def _make_beat_v6_adapter(
    monkeypatch: pytest.MonkeyPatch,
    native_feature_heads: list[str] | None,
) -> tuple[FineTuneLuminaAdapter, _ToyBeatV6Backbone]:
    toy_backbone = _ToyBeatV6Backbone()
    monkeypatch.setattr(
        "src.models.registry.build_registered_model", lambda *args, **kwargs: toy_backbone,
    )
    adapter = FineTuneLuminaAdapter(
        "beat-v6",
        torch.device("cpu"),
        native_feature_heads=native_feature_heads,
    )
    return adapter, toy_backbone


def _make_beat_v8_adapter(
    monkeypatch: pytest.MonkeyPatch,
    native_feature_heads: list[str] | None,
) -> tuple[FineTuneLuminaAdapter, _ToyBeatV8Backbone]:
    toy_backbone = _ToyBeatV8Backbone()
    monkeypatch.setattr(
        "src.models.registry.build_registered_model", lambda *args, **kwargs: toy_backbone,
    )
    adapter = FineTuneLuminaAdapter(
        "beat-v8",
        torch.device("cpu"),
        native_feature_heads=native_feature_heads,
    )
    return adapter, toy_backbone


def test_finetune_lumina_adapter_exposes_native_variant_head_selection_for_beat_v6(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, _ = _make_beat_v6_adapter(
        monkeypatch,
        ["mutation_effect", "aa", "codon_phylo", "phylo470"],
    )

    assert adapter.native_variant_head_selection == (
        "mutation_effect", "aa", "codon_phylo", "phylo470",
    )
    assert adapter.native_variant_feature_dim == 3 + NUM_AA_CLASSES + 1 + 1


def test_finetune_lumina_adapter_no_native_selection_for_beat_v4(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    toy_backbone = _ToyBeatV6Backbone()
    monkeypatch.setattr(
        "src.models.registry.build_registered_model", lambda *args, **kwargs: toy_backbone,
    )
    adapter = FineTuneLuminaAdapter(
        "beat-v4",
        torch.device("cpu"),
        native_feature_heads=["mutation_effect", "aa", "codon_phylo", "phylo470"],
    )

    assert adapter.native_variant_head_selection is None
    assert adapter.native_variant_feature_dim is None


def test_end_to_end_beat_v6_single_pass_native_path(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, backbone = _make_beat_v6_adapter(
        monkeypatch,
        ["mutation_effect", "aa", "codon_phylo", "phylo470"],
    )
    model = EndToEndClinVarModel(adapter, regime="A", proj_dim=4, hidden_dim=8, dropout=0.0)

    variant_dim = compute_native_feature_dim(adapter.native_variant_head_selection or ())
    assert variant_dim == 27
    head = cast("RegimeAHead", model.head)
    assert head.variant_projection.in_features == variant_dim

    logits = model(
        ["AAAA", "CCCC"],
        ["CAAA", "TCCC"],
        [1, 1],
        ref_alleles=["A", "C"],
        alt_alleles=["C", "T"],
    )

    assert logits.shape == (2,)
    assert torch.isfinite(logits).all()
    assert backbone.encode_calls == 1
    assert backbone.mutation_effect_head.call_count == 1
    assert backbone.aa_head.call_count == 1
    assert backbone.codon_phylo_head.call_count == 1
    assert backbone.phylo470_head.call_count == 1


def test_beat_v8_defaults_to_allele_scorer_head_and_features() -> None:
    config = FineTuneConfig(model_family="lumina", model_version="beat-v8")

    assert config.head_type == "regime_a_v8"
    assert config.native_feature_heads == [
        "allele_repr",
        "allele_effect_logits",
        "allele_severity_score",
        "allele_swap_severity_score",
        "allele_far_distance",
        "allele_site_delta",
        "allele_local_delta",
        "mlm_logit_ratio",
        "is_snv",
    ]
    assert config.pairwise_rank_weight == pytest.approx(0.2)
    assert config.swap_consistency_weight == pytest.approx(0.1)


def test_end_to_end_beat_v8_uses_pretrained_allele_features(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, backbone = _make_beat_v8_adapter(
        monkeypatch,
        ["allele_repr", "allele_effect_logits", "allele_severity_score"],
    )
    model = EndToEndClinVarModel(
        adapter,
        regime="A",
        proj_dim=4,
        hidden_dim=8,
        dropout=0.0,
        head_type="allele_scorer",
    )

    variant_dim = compute_native_feature_dim(adapter.native_variant_head_selection or (), adapter.d_model)
    assert variant_dim == adapter.d_model + NUM_ALLELE_EFFECT_CLASSES + 1
    head = cast("RegimeAHead", model.head)
    assert head.variant_projection.in_features == variant_dim

    logits = model(
        ["AAAA", "CCCC"],
        ["AGAA", "CTCC"],
        [1, 1],
        ref_alleles=["A", "C"],
        alt_alleles=["G", "T"],
    )

    assert logits.shape == (2,)
    assert torch.isfinite(logits).all()
    assert backbone.allele_score_calls == 1


def test_beat_v8_native_features_mask_snv_only_outputs_and_keep_delta_for_multi_base_variants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection = [
        "allele_repr",
        "allele_effect_logits",
        "allele_severity_score",
        "allele_swap_severity_score",
        "allele_far_distance",
        "allele_site_delta",
        "allele_local_delta",
        "mlm_logit_ratio",
        "is_snv",
    ]
    adapter, _ = _make_beat_v8_adapter(monkeypatch, selection)

    assert adapter.native_variant_feature_dim == 3 * adapter.d_model + NUM_ALLELE_EFFECT_CLASSES + 5

    _site, variant_native, _local = adapter.forward_native_variant_features(
        ref_seqs=["AAAA", "AAAA"],
        alt_seqs=["AGAA", "AGTA"],
        variant_offsets=[1, 1],
        ref_alleles=["A", "AC"],
        alt_alleles=["G", "GT"],
    )

    offset = 0
    allele_repr = variant_native[:, offset : offset + adapter.d_model]
    offset += adapter.d_model
    effect_logits = variant_native[:, offset : offset + NUM_ALLELE_EFFECT_CLASSES]
    offset += NUM_ALLELE_EFFECT_CLASSES
    severity = variant_native[:, offset : offset + 1]
    offset += 1
    swap_severity = variant_native[:, offset : offset + 1]
    offset += 1
    far_distance = variant_native[:, offset : offset + 1]
    offset += 1
    site_delta = variant_native[:, offset : offset + adapter.d_model]
    offset += adapter.d_model
    local_delta = variant_native[:, offset : offset + adapter.d_model]
    offset += adapter.d_model
    mlm_ratio = variant_native[:, offset : offset + 1]
    offset += 1
    is_snv = variant_native[:, offset : offset + 1]

    assert allele_repr[0].abs().sum().item() > 0.0
    assert torch.allclose(allele_repr[1], torch.zeros_like(allele_repr[1]))
    assert torch.allclose(effect_logits[1], torch.zeros_like(effect_logits[1]))
    assert severity[1].item() == 0.0
    assert swap_severity[1].item() == 0.0
    assert far_distance[1].abs().item() > 0.0
    assert site_delta[1].abs().sum().item() > 0.0
    assert local_delta[1].abs().sum().item() > 0.0
    assert mlm_ratio[1].item() == 0.0
    assert is_snv.tolist() == [[1.0], [0.0]]


def test_beat_v6_native_path_zeros_mutation_effect_for_multi_base_variants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, _ = _make_beat_v6_adapter(
        monkeypatch,
        ["mutation_effect", "aa", "codon_phylo", "phylo470"],
    )

    _site_ref, variant_native, _local = adapter.forward_native_variant_features(
        ref_seqs=["AAAA", "AAAA"],
        alt_seqs=["AGAA", "AGTAA"],
        variant_offsets=[1, 1],
        ref_alleles=["A", "AC"],
        alt_alleles=["G", "GT"],
    )

    mutation_slot = variant_native[:, :NUM_MUTATION_EFFECT_CLASSES]
    aa_slot = variant_native[
        :, NUM_MUTATION_EFFECT_CLASSES:NUM_MUTATION_EFFECT_CLASSES + NUM_AA_CLASSES
    ]
    conservation_slot = variant_native[:, NUM_MUTATION_EFFECT_CLASSES + NUM_AA_CLASSES :]

    assert mutation_slot[0].abs().sum().item() > 0
    assert torch.allclose(mutation_slot[1], torch.zeros_like(mutation_slot[1]))
    assert aa_slot[0].abs().sum().item() > 0
    assert aa_slot[1].abs().sum().item() > 0
    assert conservation_slot[0].abs().sum().item() > 0
    assert conservation_slot[1].abs().sum().item() > 0


def test_beat_v6_native_path_a1_uses_only_mutation_effect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, backbone = _make_beat_v6_adapter(monkeypatch, ["mutation_effect"])
    model = EndToEndClinVarModel(adapter, regime="A", proj_dim=4, hidden_dim=8, dropout=0.0)

    assert adapter.native_variant_head_selection == ("mutation_effect",)
    head = cast("RegimeAHead", model.head)
    assert head.variant_projection.in_features == NUM_MUTATION_EFFECT_CLASSES

    logits = model(
        ["AAAA", "CCCC"],
        ["CAAA", "TCCC"],
        [1, 1],
        ref_alleles=["A", "C"],
        alt_alleles=["C", "T"],
    )

    assert logits.shape == (2,)
    assert backbone.mutation_effect_head.call_count == 1
    assert backbone.aa_head.call_count == 0
    assert backbone.codon_phylo_head.call_count == 0
    assert backbone.phylo470_head.call_count == 0


def test_beat_v6_falls_back_to_two_pass_when_native_feature_heads_is_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, backbone = _make_beat_v6_adapter(monkeypatch, ["none"])

    assert adapter.native_variant_head_selection is None
    model = EndToEndClinVarModel(adapter, regime="A", proj_dim=4, hidden_dim=8, dropout=0.0)

    model(
        ["AAAA", "CCCC"],
        ["CAAA", "TCCC"],
        [1, 1],
        ref_alleles=["A", "C"],
        alt_alleles=["C", "T"],
    )

    assert backbone.encode_calls == 2


def test_freeze_native_feature_heads_disables_gradients_for_beat_v6(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, backbone = _make_beat_v6_adapter(
        monkeypatch,
        ["mutation_effect", "aa", "codon_phylo", "phylo470"],
    )
    selection = ("mutation_effect", "aa", "codon_phylo", "phylo470")

    frozen = freeze_native_feature_heads(backbone, selection)

    assert frozen
    for attr in ("mutation_effect_head", "aa_head", "codon_phylo_head", "phylo470_head"):
        for param in getattr(backbone, attr).parameters():
            assert param.requires_grad is False


def test_clinvar_cli_accepts_native_feature_heads_flag() -> None:
    config = parse_args(
        [
            "--model-family",
            "lumina",
            "--model-version",
            "beat-v6",
            "--output-dir",
            "outputs/clinvar/v6",
            "--native-feature-heads",
            "mutation_effect",
        ]
    )

    assert config.native_feature_heads == ["mutation_effect"]


def test_clinvar_cli_default_native_feature_heads_is_a2() -> None:
    config = parse_args(
        [
            "--model-family",
            "lumina",
            "--model-version",
            "beat-v6",
            "--output-dir",
            "outputs/clinvar/v6-default",
        ]
    )

    assert config.native_feature_heads == [
        "mutation_effect", "aa", "codon_phylo", "phylo470",
    ]


def test_clinvar_cli_defaults_to_v7_head_for_beat_v7() -> None:
    config = parse_args(
        [
            "--model-family",
            "lumina",
            "--model-version",
            "beat-v7",
            "--output-dir",
            "outputs/clinvar/v7-default-head",
        ]
    )

    assert config.head_type == "regime_a_v7"


def test_clinvar_cli_defaults_to_regime_a_v8_for_beat_v8() -> None:
    config = parse_args(
        [
            "--model-family",
            "lumina",
            "--model-version",
            "beat-v8",
            "--output-dir",
            "outputs/clinvar/v8-default-head",
        ]
    )

    assert config.head_type == "regime_a_v8"
    assert config.native_feature_heads == [
        "allele_repr",
        "allele_effect_logits",
        "allele_severity_score",
        "allele_swap_severity_score",
        "allele_far_distance",
        "allele_site_delta",
        "allele_local_delta",
        "mlm_logit_ratio",
        "is_snv",
    ]
    assert config.pairwise_rank_weight == pytest.approx(0.2)
    assert config.swap_consistency_weight == pytest.approx(0.1)


def test_clinvar_cli_allows_disabling_beat_v8_swap_and_pairwise_defaults() -> None:
    config = parse_args(
        [
            "--model-family",
            "lumina",
            "--model-version",
            "beat-v8",
            "--output-dir",
            "outputs/clinvar/v8-disable-losses",
            "--pairwise-rank-weight",
            "0.0",
            "--swap-consistency-weight",
            "0.0",
        ]
    )

    assert config.pairwise_rank_weight == 0.0
    assert config.swap_consistency_weight == 0.0


def test_freeze_native_feature_heads_pins_eval_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    _, backbone = _make_beat_v6_adapter(
        monkeypatch,
        ["mutation_effect", "aa", "codon_phylo", "phylo470"],
    )
    selection = ("mutation_effect", "aa", "codon_phylo", "phylo470")

    freeze_native_feature_heads(backbone, selection)

    # Simulate the outer training loop putting the full backbone back in train mode.
    backbone.train()

    for attr in ("mutation_effect_head", "aa_head", "codon_phylo_head", "phylo470_head"):
        assert getattr(backbone, attr).training is False, f"{attr} leaked back to train mode"


def test_end_to_end_beat_v6_native_path_supports_mlm_logit_ratio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, backbone = _make_beat_v6_adapter(
        monkeypatch, ["mutation_effect", "mlm_logit_ratio"],
    )
    model = EndToEndClinVarModel(adapter, regime="A", proj_dim=4, hidden_dim=8, dropout=0.0)

    variant_dim = compute_native_feature_dim(adapter.native_variant_head_selection or ())
    assert variant_dim == NUM_MUTATION_EFFECT_CLASSES + 1
    head = cast("RegimeAHead", model.head)
    assert head.variant_projection.in_features == variant_dim

    logits = model(
        ["AAAA", "CCCC"],
        ["CAAA", "TCCC"],
        [1, 1],
        ref_alleles=["A", "C"],
        alt_alleles=["C", "T"],
    )

    assert logits.shape == (2,)
    assert torch.isfinite(logits).all()
    assert backbone.encode_calls == 1
    assert backbone.mlm_head.call_count == 1


def test_mlm_logit_ratio_equals_alt_minus_ref_for_snv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, backbone = _make_beat_v6_adapter(monkeypatch, ["mlm_logit_ratio"])

    site_hidden, variant_native, _local = adapter.forward_native_variant_features(
        ref_seqs=["AAAA", "CCCC"],
        alt_seqs=["AGAA", "CTCC"],
        variant_offsets=[1, 1],
        ref_alleles=["A", "C"],
        alt_alleles=["G", "T"],
    )

    # Replay the expected logit diff through the underlying linear so the call
    # counter stays at 1 and we compare against the exact weights used above.
    expected_logits = backbone.mlm_head.linear(site_hidden)
    ref_ids = torch.tensor([DNA_VOCAB["A"], DNA_VOCAB["C"]], dtype=torch.long)
    alt_ids = torch.tensor([DNA_VOCAB["G"], DNA_VOCAB["T"]], dtype=torch.long)
    batch_idx = torch.arange(2)
    expected = (
        expected_logits[batch_idx, alt_ids] - expected_logits[batch_idx, ref_ids]
    ).unsqueeze(-1)

    assert variant_native.shape == (2, 1)
    assert torch.allclose(variant_native, expected, atol=1e-6)


def test_mlm_logit_ratio_zeroed_for_indel(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, _ = _make_beat_v6_adapter(monkeypatch, ["mlm_logit_ratio"])

    _site, variant_native, _local = adapter.forward_native_variant_features(
        ref_seqs=["AAAA", "AAAA"],
        alt_seqs=["AGAA", "AGTAA"],
        variant_offsets=[1, 1],
        ref_alleles=["A", "AC"],
        alt_alleles=["G", "GT"],
    )

    assert variant_native.shape == (2, 1)
    assert abs(float(variant_native[0].item())) > 0.0
    assert variant_native[1].item() == 0.0


def test_is_snv_feature_values(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, _ = _make_beat_v6_adapter(monkeypatch, ["is_snv"])

    _site, variant_native, _local = adapter.forward_native_variant_features(
        ref_seqs=["AAAA", "AAAA"],
        alt_seqs=["AGAA", "AGTAA"],
        variant_offsets=[1, 1],
        ref_alleles=["A", "AC"],
        alt_alleles=["G", "GT"],
    )

    assert variant_native.shape == (2, 1)
    assert variant_native[0].item() == 1.0
    assert variant_native[1].item() == 0.0


def test_is_snv_synthetic_head_requires_no_backbone_attribute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _BackboneWithoutMlm(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.cfg = type("Cfg", (), {"d_model": 4})()
            self.pad_token_id = 0
            self.token_emb = nn.Embedding(VOCAB_SIZE, 4, padding_idx=0)
            self.encode_calls = 0

        def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
            self.encode_calls += 1
            return self.token_emb(input_ids)

    toy_backbone = _BackboneWithoutMlm()
    monkeypatch.setattr(
        "src.models.registry.build_registered_model", lambda *args, **kwargs: toy_backbone,
    )
    adapter = FineTuneLuminaAdapter(
        "beat-v6",
        torch.device("cpu"),
        native_feature_heads=["is_snv"],
    )

    assert adapter.native_variant_head_selection == ("is_snv",)
    assert adapter.native_variant_feature_dim == 1

    _site, variant_native, _local = adapter.forward_native_variant_features(
        ref_seqs=["AAAA", "AAAA"],
        alt_seqs=["AGAA", "AGTAA"],
        variant_offsets=[1, 1],
        ref_alleles=["A", "AC"],
        alt_alleles=["G", "GT"],
    )

    assert variant_native.shape == (2, 1)
    assert variant_native[0].item() == 1.0
    assert variant_native[1].item() == 0.0


def test_clinvar_cli_accepts_mlm_logit_ratio_and_is_snv() -> None:
    config = parse_args(
        [
            "--model-family",
            "lumina",
            "--model-version",
            "beat-v6",
            "--output-dir",
            "outputs/clinvar/v6-mlm-snv",
            "--native-feature-heads",
            "mutation_effect",
            "mlm_logit_ratio",
            "is_snv",
        ]
    )

    assert config.native_feature_heads == ["mutation_effect", "mlm_logit_ratio", "is_snv"]


def _make_variants() -> tuple[list[CachedVariant], list[CachedVariant]]:
    train_variants = [
        CachedVariant(
            ref_seq="AAAA",
            alt_seq="AAAA",
            variant_offset=1,
            label=0,
            index=0,
            ref_allele="A",
            alt_allele="A",
        ),
        CachedVariant(
            ref_seq="AAAA",
            alt_seq="CAAA",
            variant_offset=1,
            label=1,
            index=1,
            ref_allele="A",
            alt_allele="C",
        ),
        CachedVariant(
            ref_seq="GGGG",
            alt_seq="GGGG",
            variant_offset=1,
            label=0,
            index=2,
            ref_allele="G",
            alt_allele="G",
        ),
        CachedVariant(
            ref_seq="GGGG",
            alt_seq="TGGG",
            variant_offset=1,
            label=1,
            index=3,
            ref_allele="G",
            alt_allele="T",
        ),
    ]
    test_variants = [
        CachedVariant(
            ref_seq="CCCC",
            alt_seq="CCCC",
            variant_offset=1,
            label=0,
            index=4,
            ref_allele="C",
            alt_allele="C",
        ),
        CachedVariant(
            ref_seq="CCCC",
            alt_seq="TCCC",
            variant_offset=1,
            label=1,
            index=5,
            ref_allele="C",
            alt_allele="T",
        ),
        CachedVariant(
            ref_seq="TTTT",
            alt_seq="TTTT",
            variant_offset=1,
            label=0,
            index=6,
            ref_allele="T",
            alt_allele="T",
        ),
        CachedVariant(
            ref_seq="TTTT",
            alt_seq="ATTT",
            variant_offset=1,
            label=1,
            index=7,
            ref_allele="T",
            alt_allele="A",
        ),
    ]
    return train_variants, test_variants


def _patch_toy_finetune(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, PrecisionPolicy | None]:
    recorded: dict[str, PrecisionPolicy | None] = {"precision": None}
    monkeypatch.setattr("eval.clinvar.train.build_variant_cache", lambda *args, **kwargs: tmp_path / "cache.parquet")
    monkeypatch.setattr("eval.clinvar.train.load_variant_cache", lambda *args, **kwargs: _make_variants())

    def _build_adapter(
        model_family: str,
        model_version: str,
        device: torch.device,
        checkpoint_path: str | None = None,
        precision: PrecisionPolicy | None = None,
        native_feature_heads: list[str] | None = None,
    ) -> _ToyAdapter:
        _ = (model_family, model_version, checkpoint_path, native_feature_heads)
        recorded["precision"] = precision
        return _ToyAdapter(device)

    monkeypatch.setattr("eval.clinvar.train.build_finetune_adapter", _build_adapter)
    return recorded


def _make_config(output_dir: Path, *, max_epochs: int = 1) -> FineTuneConfig:
    return FineTuneConfig(
        model_family="lumina",
        model_version="toy",
        dataset_path=str(output_dir / "unused_dataset.parquet"),
        fasta_path=str(output_dir / "unused.fa"),
        context_size=4,
        batch_size=2,
        grad_accum_steps=1,
        max_epochs=max_epochs,
        lr_backbone=1e-2,
        lr_head=1e-2,
        wd_backbone=0.0,
        wd_head=0.0,
        lora_rank=2,
        lora_alpha=4.0,
        lora_dropout=0.0,
        freeze_backbone_steps=0,
        loss_type="bce",
        device="cpu",
        precision="auto",
        output_dir=str(output_dir),
        overwrite=True,
    )


def test_end_to_end_clinvar_model_runs_with_toy_adapter_regime_a() -> None:
    adapter = _ToyAdapter(torch.device("cpu"))
    model = EndToEndClinVarModel(adapter, regime="A", proj_dim=4, hidden_dim=8, dropout=0.0)

    logits = model(["AAAA", "CCCC"], ["CAAA", "TCCC"], [1, 1])

    assert logits.shape == (2,)
    assert torch.isfinite(logits).all()


def test_end_to_end_clinvar_model_uses_single_pass_variant_encoder_interface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _SinglePassFeatureAdapter()
    model = EndToEndClinVarModel(adapter, regime="A", proj_dim=4, hidden_dim=8, dropout=0.0)
    assert isinstance(model.variant_encoder, ClinVarVariantEncoder)

    variant_encoder_calls = 0
    original_forward = model.variant_encoder.forward

    def _counted_forward(*args: Any, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        nonlocal variant_encoder_calls
        variant_encoder_calls += 1
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(model.variant_encoder, "forward", _counted_forward)

    logits = model(["AAAA", "CCCC"], ["CAAA", "TCCC"], [1, 1], ref_alleles=["A", "C"], alt_alleles=["C", "T"])

    assert adapter.backbone.extract_calls == 1
    assert variant_encoder_calls == 1
    assert logits.shape == (2,)
    assert torch.isfinite(logits).all()


def test_clinvar_cli_defaults_to_auto_precision() -> None:
    config = parse_args(
        [
            "--model-family",
            "lumina",
            "--model-version",
            "beat-v2",
            "--output-dir",
            "outputs/clinvar/test",
        ]
    )

    assert config.precision == "auto"
    assert config.allow_tf32 is True
    assert config.max_epochs == 5
    assert config.freeze_backbone_steps == 100


def test_clinvar_cli_accepts_dnabert_alias_and_normalizes_family() -> None:
    config = parse_args(
        [
            "--model-family",
            "dnabert-2",
            "--model-version",
            "117M",
            "--output-dir",
            "outputs/clinvar/dnabert2-test",
        ]
    )

    assert config.model_family == "dnabert2"


def test_huggingface_hub_kwargs_use_cache_dir_and_offline_mode() -> None:
    kwargs = _huggingface_hub_kwargs(
        env={
            "HF_HUB_CACHE": "/tmp/hf-cache",
            "HF_HUB_OFFLINE": "1",
        }
    )

    assert kwargs["cache_dir"] == "/tmp/hf-cache"
    assert kwargs["local_files_only"] is True


def test_huggingface_hub_kwargs_omit_local_only_when_offline_is_disabled() -> None:
    kwargs = _huggingface_hub_kwargs(
        env={
            "HF_HUB_CACHE": "/tmp/hf-cache",
            "TRANSFORMERS_OFFLINE": "0",
        }
    )

    assert kwargs["cache_dir"] == "/tmp/hf-cache"
    assert "local_files_only" not in kwargs


def test_extract_features_preserves_hidden_state_dtype() -> None:
    model = EndToEndClinVarModel(_Bfloat16FeatureAdapter(), regime="A", proj_dim=4, hidden_dim=8, dropout=0.0)

    site_ref, variant_repr, local_context = model._extract_features(["AAAA", "CCCC"], ["CAAA", "TCCC"], [1, 1])

    assert site_ref.dtype == torch.bfloat16
    assert variant_repr.dtype == torch.bfloat16
    assert local_context.dtype == torch.bfloat16


def test_end_to_end_clinvar_model_falls_back_to_two_pass_hidden_state_extraction_for_legacy_adapters() -> None:
    adapter = _ToyAdapter(torch.device("cpu"))
    model = EndToEndClinVarModel(adapter, regime="A", proj_dim=4, hidden_dim=8, dropout=0.0)

    model._extract_features(["AAAA", "CCCC"], ["CAAA", "TCCC"], [1, 1])

    assert adapter.forward_calls == 2


def test_finetune_lumina_adapter_uses_single_pass_variant_features_for_beat_v5(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    toy_backbone = _ToyBeatV5Backbone()
    monkeypatch.setattr("src.models.registry.build_registered_model", lambda *args, **kwargs: toy_backbone)

    adapter = FineTuneLuminaAdapter("beat-v5", torch.device("cpu"))
    model = EndToEndClinVarModel(adapter, regime="A", proj_dim=4, hidden_dim=8, dropout=0.0)
    assert isinstance(model.variant_encoder, ClinVarVariantEncoder)

    variant_encoder_calls = 0
    original_forward = model.variant_encoder.forward

    def _counted_forward(*args: Any, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        nonlocal variant_encoder_calls
        variant_encoder_calls += 1
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(model.variant_encoder, "forward", _counted_forward)

    logits = model(
        ["AAAA", "CCCC"],
        ["CAAA", "TCCC"],
        [1, 1],
        ref_alleles=["A", "C"],
        alt_alleles=["C", "T"],
    )

    assert toy_backbone.extract_calls == 1
    assert variant_encoder_calls == 1
    assert logits.shape == (2,)


def test_collect_task_parameter_groups_keeps_variant_encoder_outside_backbone() -> None:
    model = EndToEndClinVarModel(_SinglePassFeatureAdapter(), regime="A", proj_dim=4, hidden_dim=8, dropout=0.0)
    for param in model.backbone.parameters():
        param.requires_grad = False
    apply_lora(model.backbone, rank=2, alpha=4.0, dropout=0.0)
    enable_layernorm_training(model.backbone)

    lora_params, norm_params, task_params = _collect_task_parameter_groups(model)
    task_param_ids = {id(param) for param in task_params}
    variant_encoder_param_ids = {id(param) for param in model.variant_encoder.parameters()}
    head_param_ids = {id(param) for param in model.head.parameters()}
    backbone_param_ids = {id(param) for param in model.backbone.parameters()}

    assert variant_encoder_param_ids
    assert variant_encoder_param_ids <= task_param_ids
    assert head_param_ids <= task_param_ids
    assert not task_param_ids & {id(param) for param in lora_params}
    assert not task_param_ids & {id(param) for param in norm_params}
    assert not variant_encoder_param_ids & backbone_param_ids


def _build_three_group_optimizer(
    lr_backbone: float = 5e-6,
    lr_head: float = 5e-4,
) -> torch.optim.Optimizer:
    return torch.optim.AdamW([
        {"params": [nn.Parameter(torch.zeros(1))], "lr": lr_backbone},
        {"params": [nn.Parameter(torch.zeros(1))], "lr": lr_backbone},
        {"params": [nn.Parameter(torch.zeros(1))], "lr": lr_head},
    ])


def test_build_cosine_schedule_holds_backbone_groups_at_zero_during_freeze_window() -> None:
    optimizer = _build_three_group_optimizer()
    scheduler = _build_cosine_schedule(
        optimizer, total_steps=200, warmup_steps=20, freeze_backbone_steps=50,
    )

    for _ in range(5):
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == 0.0
    assert optimizer.param_groups[1]["lr"] == 0.0
    assert optimizer.param_groups[2]["lr"] > 0.0

    for _ in range(50):
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] > 0.0
    assert optimizer.param_groups[1]["lr"] > 0.0


def test_build_cosine_schedule_no_freeze_gives_nonzero_backbone_lr_immediately() -> None:
    optimizer = _build_three_group_optimizer()
    scheduler = _build_cosine_schedule(
        optimizer, total_steps=200, warmup_steps=20, freeze_backbone_steps=0,
    )

    scheduler.step()
    assert optimizer.param_groups[0]["lr"] > 0.0
    assert optimizer.param_groups[1]["lr"] > 0.0
    assert optimizer.param_groups[2]["lr"] > 0.0


def test_clinvar_checkpoint_roundtrip_preserves_variant_encoder_state(tmp_path: Path) -> None:
    model = EndToEndClinVarModel(_SinglePassFeatureAdapter(), regime="A", proj_dim=4, hidden_dim=8, dropout=0.0)
    assert isinstance(model.variant_encoder, ClinVarVariantEncoder)

    lora_summary = apply_lora(model.backbone, rank=2, alpha=4.0, dropout=0.0)
    enable_layernorm_training(model.backbone)
    with torch.no_grad():
        model.variant_encoder.variant_out_proj.weight.fill_(0.25)

    config = _make_config(tmp_path)
    precision = PrecisionPolicy(requested="fp32", resolved="fp32", use_autocast=False)
    checkpoint_path = tmp_path / "best_model.pt"
    _save_checkpoint(model, config, lora_summary, checkpoint_path, precision)

    reloaded = EndToEndClinVarModel(_SinglePassFeatureAdapter(), regime="A", proj_dim=4, hidden_dim=8, dropout=0.0)
    apply_lora(reloaded.backbone, rank=2, alpha=4.0, dropout=0.0)
    enable_layernorm_training(reloaded.backbone)
    bundle = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    reloaded.load_state_dict(bundle["model_state_dict"])

    assert isinstance(reloaded.variant_encoder, ClinVarVariantEncoder)
    assert torch.allclose(
        reloaded.variant_encoder.variant_out_proj.weight,
        model.variant_encoder.variant_out_proj.weight,
    )


def test_normalize_lumina_model_config_clamps_mimo_chunk_size_for_bf16() -> None:
    precision = PrecisionPolicy(
        requested="bf16",
        resolved="bf16",
        use_autocast=True,
        autocast_device_type="cuda",
        autocast_dtype=torch.bfloat16,
    )

    unsafe = _normalize_lumina_model_config_for_precision(
        "beat-v2",
        {"is_mimo": True, "mimo_rank": 4, "chunk_size": 64},
        precision,
    )
    missing_chunk = _normalize_lumina_model_config_for_precision(
        "beat-v2",
        {"is_mimo": True, "mimo_rank": 8},
        precision,
    )

    assert unsafe["chunk_size"] == 16
    assert missing_chunk["chunk_size"] == 8


def test_normalize_lumina_model_config_disables_mimo_on_pre_hopper_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    precision = PrecisionPolicy(
        requested="bf16",
        resolved="bf16",
        use_autocast=True,
        autocast_device_type="cuda",
        autocast_dtype=torch.bfloat16,
    )
    monkeypatch.setattr(
        "eval.clinvar.adapters.get_cuda_device_capability",
        lambda _device: (8, 0),
    )

    resolved = _normalize_lumina_model_config_for_precision(
        "beat-v5",
        {"is_mimo": True, "mimo_rank": 4, "chunk_size": 16},
        precision,
        device=torch.device("cuda", 0),
    )

    assert resolved["is_mimo"] is False
    assert resolved["chunk_size"] == 64


def test_normalize_lumina_model_config_clamps_mimo_chunk_size_for_mxfp8_bf16_compute() -> None:
    precision = PrecisionPolicy(
        requested="auto",
        resolved="mxfp8",
        use_autocast=True,
        autocast_device_type="cuda",
        autocast_dtype=torch.bfloat16,
        fp8_enabled=True,
        fp8_backend="transformer_engine",
        fp8_module_count=4,
    )

    resolved = _normalize_lumina_model_config_for_precision(
        "beat-v2",
        {"is_mimo": True, "mimo_rank": 4, "chunk_size": 64},
        precision,
    )

    assert resolved["chunk_size"] == 16


def test_resolve_caduceus_hidden_dim_preserves_non_rcps_width() -> None:
    config = type("Config", (), {"d_model": 256, "rcps": False})()

    assert _resolve_caduceus_hidden_dim(config) == 256


def test_resolve_caduceus_hidden_dim_doubles_rcps_width() -> None:
    config = type("Config", (), {"d_model": 256, "rcps": True})()

    assert _resolve_caduceus_hidden_dim(config) == 512


def test_run_finetune_single_process_writes_current_artifacts_and_resolves_precision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded = _patch_toy_finetune(monkeypatch, tmp_path)
    torch.manual_seed(0)

    output_dir = tmp_path / "smoke"
    results = run_finetune(_make_config(output_dir))

    assert recorded["precision"] is not None
    assert recorded["precision"].requested == "auto"
    assert recorded["precision"].resolved == "fp32"
    assert (output_dir / "metrics.json").is_file()
    assert (output_dir / "test_predictions.parquet").is_file()
    assert (output_dir / "best_model.pt").is_file()
    saved = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert saved["model_family"] == "lumina"
    assert saved["model_version"] == "toy"
    assert saved["requested_precision"] == "auto"
    assert saved["resolved_precision"] == "fp32"
    assert saved["fp8_enabled"] is False
    assert saved["fp8_module_count"] == 0
    assert saved["fallback_reason"] == "non_cuda_device:cpu"
    assert saved["allow_tf32"] is True
    assert saved["head_type"] == "regime_a"
    assert saved["value"] == saved["test_at_val_threshold"]["mcc"]
    assert saved["mcc"] == saved["test_at_val_threshold"]["mcc"]
    assert saved["data_summary"]["train_size"] == 2
    assert saved["data_summary"]["val_size"] == 2
    assert saved["data_summary"]["test_size"] == 4
    assert "val_optimal_threshold" in saved
    assert "test_at_default_threshold" in saved
    assert "test_at_test_optimal_threshold_LEAKY" in saved
    assert "cross_gene_diagnostic" in saved
    assert results["mcc"] == saved["mcc"]
    assert results["value"] == saved["value"]


def test_run_finetune_logs_train_steps_to_console_on_wandb_cadence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class _FakeWandbRun:
        def __init__(self) -> None:
            self.logged: list[dict[str, float | int]] = []

        def log(self, payload: dict[str, float | int]) -> None:
            self.logged.append(payload)

        def finish(self) -> None:
            return None

    class _FakeWandbModule(ModuleType):
        run: _FakeWandbRun

        def __init__(self) -> None:
            super().__init__("wandb")
            self.run = _FakeWandbRun()

        def init(self, **kwargs: object) -> _FakeWandbRun:
            _ = kwargs
            return self.run

    _patch_toy_finetune(monkeypatch, tmp_path)
    fake_wandb = _FakeWandbModule()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    torch.manual_seed(0)

    config = _make_config(tmp_path / "console", max_epochs=5)
    config.wandb_enabled = True

    caplog.set_level("INFO", logger="eval.clinvar.train")
    run_finetune(config)

    assert any("Epoch 5/5 | loss=" in record.message for record in caplog.records)
    assert not any(entry.get("train/step") == 10 for entry in fake_wandb.run.logged)
    assert any(entry.get("epoch") == 5 for entry in fake_wandb.run.logged)
    assert any("val/mcc" in entry for entry in fake_wandb.run.logged)


def test_run_finetune_computes_and_logs_swap_consistency_only_when_enabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeWandbRun:
        def __init__(self) -> None:
            self.logged: list[dict[str, float | int]] = []

        def log(self, payload: dict[str, float | int]) -> None:
            self.logged.append(payload)

        def finish(self) -> None:
            return None

    class _FakeWandbModule(ModuleType):
        run: _FakeWandbRun

        def __init__(self) -> None:
            super().__init__("wandb")
            self.run = _FakeWandbRun()

        def init(self, **kwargs: object) -> _FakeWandbRun:
            _ = kwargs
            return self.run

    calls: list[float] = []

    def _fake_swap_loss(
        logits: torch.Tensor,
        swap_logits: torch.Tensor,
        labels: torch.Tensor,
        margin: float = 0.5,
    ) -> torch.Tensor:
        _ = (swap_logits, labels)
        calls.append(margin)
        return (logits * 0.0).sum() + logits.new_tensor(0.123)

    _patch_toy_finetune(monkeypatch, tmp_path)
    fake_wandb = _FakeWandbModule()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    monkeypatch.setattr("eval.clinvar.train.TRAIN_LOG_EVERY_STEPS", 1)
    monkeypatch.setattr("eval.clinvar.train.swap_consistency_loss", _fake_swap_loss)

    enabled = _make_config(tmp_path / "swap-enabled")
    enabled.wandb_enabled = True
    enabled.swap_consistency_weight = 0.1
    enabled.swap_consistency_margin = 0.7
    run_finetune(enabled)

    assert len(calls) == 1
    assert calls[0] == pytest.approx(0.7)
    assert any("train/swap_consistency_loss" in entry for entry in fake_wandb.run.logged)

    calls.clear()
    fake_wandb.run.logged.clear()
    disabled = _make_config(tmp_path / "swap-disabled")
    disabled.wandb_enabled = True
    disabled.swap_consistency_weight = 0.0
    run_finetune(disabled)

    assert calls == []
    assert not any("train/swap_consistency_loss" in entry for entry in fake_wandb.run.logged)


def test_run_finetune_loads_best_checkpoint_for_final_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_toy_finetune(monkeypatch, tmp_path)
    load_calls: list[int] = []
    original_load_state_dict = EndToEndClinVarModel.load_state_dict

    def _wrapped_load_state_dict(
        self: EndToEndClinVarModel,
        state_dict: Mapping[str, object],
        strict: bool = True,
        assign: bool = False,
    ) -> object:
        load_calls.append(1)
        return original_load_state_dict(self, state_dict, strict=strict, assign=assign)

    monkeypatch.setattr(EndToEndClinVarModel, "load_state_dict", _wrapped_load_state_dict)
    torch.manual_seed(0)

    output_dir = tmp_path / "reload"
    run_finetune(_make_config(output_dir, max_epochs=2))

    assert load_calls == [1]
    assert (output_dir / "best_model.pt").is_file()


def test_run_finetune_raises_when_best_checkpoint_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_toy_finetune(monkeypatch, tmp_path)
    monkeypatch.setattr("eval.clinvar.train._save_checkpoint", lambda *args, **kwargs: None)
    torch.manual_seed(0)

    with pytest.raises(FileNotFoundError, match="Best checkpoint not found"):
        run_finetune(_make_config(tmp_path / "missing"))
