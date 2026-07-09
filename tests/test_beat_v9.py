from __future__ import annotations

from typing import Any, cast

import pytest
import torch
import torch.nn as nn

from src.constants import (
    CODON_IGNORE_INDEX,
    NUM_COUNTERFACTUAL_EFFECT_CLASSES,
    NUM_REGION_CLASSES,
    NUM_STRUCTURE_CLASSES,
    PAD_ID,
)
from src.models import build_registered_model, resolve_model_config_dict
from src.models.beat_v9.heads import build_counterfactual_snv_mask
from src.objectives import compute_multitask_loss
from src.train import TrainConfig, build_arg_parser, config_from_args, ddp_kwargs_for_train_config


class _FakeMamba3(nn.Module):
    def __init__(self, d_model: int, **_kwargs: Any) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def _small_v9_overrides() -> dict[str, Any]:
    return {
        "d_model": 32,
        "depth": 2,
        "d_state": 16,
        "headdim": 16,
        "dropout": 0.0,
        "use_local_attention": False,
        "conv_stem_kernels": [3, 6],
        "hidden_tap_layers": [1, 2],
        "counterfactual_repr_dim": 16,
        "multiscale_context_radii": [1, 2],
        "sequence_embedding_dim": 32,
    }


def _assert_no_bio_feature_config(config: dict[str, Any]) -> None:
    assert "use_bio_feature_inputs" not in config
    assert all(not key.startswith("bio_feature") for key in config)


def test_registry_resolves_default_beat_v9_model_config() -> None:
    resolved = resolve_model_config_dict("beat-v9", {})

    assert resolved["d_model"] == 256
    assert resolved["depth"] == 8
    _assert_no_bio_feature_config(resolved)
    assert resolved["use_counterfactual_snv_head"] is True
    assert resolved["num_counterfactual_effect_classes"] == NUM_COUNTERFACTUAL_EFFECT_CLASSES
    assert resolved["num_region_classes"] == NUM_REGION_CLASSES


def test_registry_rejects_beat_v9_bio_feature_input_config() -> None:
    with pytest.raises(ValueError, match="use_bio_feature_inputs"):
        resolve_model_config_dict("beat-v9", {"use_bio_feature_inputs": True})


def test_beat_v9_forward_rejects_bio_features_kwarg(monkeypatch: Any) -> None:
    import src.models.beat_shared as beat_shared

    monkeypatch.setattr(beat_shared, "_Mamba3", _FakeMamba3)
    model = cast(Any, build_registered_model("beat-v9", _small_v9_overrides()))
    batch_size, seq_len = 2, 16
    input_ids = torch.randint(1, 5, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

    with pytest.raises(TypeError, match="bio_features"):
        model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            bio_features={},
            return_token_heads=False,
        )


def test_beat_v9_forward_shapes(monkeypatch: Any) -> None:
    import src.models.beat_shared as beat_shared

    monkeypatch.setattr(beat_shared, "_Mamba3", _FakeMamba3)
    model = cast(Any, build_registered_model("beat-v9", _small_v9_overrides()))
    batch_size, seq_len = 2, 16
    input_ids = torch.randint(1, 5, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_token_heads=True,
        return_sequence_embedding=True,
        return_hidden=True,
        return_hidden_taps=True,
        return_counterfactual=True,
    )

    assert outputs["hidden_states"].shape == (batch_size, seq_len, model.cfg.d_model)
    assert outputs["hidden_taps"][1].shape == (batch_size, seq_len, model.cfg.d_model)
    assert outputs["mlm_logits"].shape == (batch_size, seq_len, 8)
    assert outputs["phylo100_pred"].shape == (batch_size, seq_len)
    assert outputs["region_logits"].shape == (batch_size, seq_len, NUM_REGION_CLASSES)
    assert outputs["structure_logits"].shape == (batch_size, seq_len, NUM_STRUCTURE_CLASSES)
    assert outputs["counterfactual_effect_logits"].shape == (
        batch_size,
        seq_len,
        4,
        NUM_COUNTERFACTUAL_EFFECT_CLASSES,
    )
    assert outputs["counterfactual_severity"].shape == (batch_size, seq_len, 4)
    assert outputs["counterfactual_snv_repr"].shape == (batch_size, seq_len, 4, 16)
    assert outputs["sequence_embedding"].shape == (batch_size, model.cfg.sequence_embedding_dim)


def test_beat_v9_loss_is_finite_and_backward(monkeypatch: Any) -> None:
    import src.models.beat_shared as beat_shared

    monkeypatch.setattr(beat_shared, "_Mamba3", _FakeMamba3)
    model = cast(Any, build_registered_model("beat-v9", _small_v9_overrides()))
    batch_size, seq_len = 2, 12
    input_ids = torch.randint(1, 5, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_hidden=True,
    )

    mlm_labels = torch.full_like(input_ids, PAD_ID)
    mlm_labels[:, 0] = input_ids[:, 0]
    cf_mask = build_counterfactual_snv_mask(input_ids)
    batch = {
        "mlm_labels": mlm_labels,
        "aux_valid_mask": attention_mask,
        "phylo100": torch.randn(batch_size, seq_len),
        "phylo470": torch.randn(batch_size, seq_len),
        "structure_labels": torch.randint(0, NUM_STRUCTURE_CLASSES, (batch_size, seq_len)),
        "region_labels": torch.randint(0, NUM_REGION_CLASSES, (batch_size, seq_len)),
        "aa_labels": torch.zeros(batch_size, seq_len, dtype=torch.long),
        "codon_phylo_target": torch.randn(batch_size, seq_len),
        "codon_labels": torch.full((batch_size, seq_len), CODON_IGNORE_INDEX, dtype=torch.long),
        "conservation_bin_labels": torch.randint(0, 7, (batch_size, seq_len)),
        "donor_distance_labels": torch.randint(0, 7, (batch_size, seq_len)),
        "acceptor_distance_labels": torch.randint(0, 7, (batch_size, seq_len)),
        "codon_pos_labels": torch.randint(0, 5, (batch_size, seq_len)),
        "exon_phase_labels": torch.randint(0, 5, (batch_size, seq_len)),
        "counterfactual_effect_labels": torch.randint(
            0,
            NUM_COUNTERFACTUAL_EFFECT_CLASSES,
            (batch_size, seq_len, 4),
        ),
        "counterfactual_severity_targets": torch.rand(batch_size, seq_len, 4),
        "counterfactual_valid_mask": cf_mask,
    }

    loss, stats = compute_multitask_loss(
        outputs,
        batch,
        w_mlm=1.0,
        w_phylo100=0.25,
        w_phylo470=0.25,
        w_structure=0.25,
        w_region=0.25,
        w_codon=0.1,
        w_conservation_bin=0.1,
        w_splice_distance=0.1,
        w_codon_pos=0.1,
        w_exon_phase=0.1,
        w_counterfactual_snv=1.0,
        w_counterfactual_severity=0.25,
    )

    assert torch.isfinite(loss)
    assert torch.isfinite(stats["loss_counterfactual_snv"])
    loss.backward()
    assert any(param.grad is not None for param in model.parameters() if param.requires_grad)


def test_counterfactual_masks_ref_equal_alt() -> None:
    input_ids = torch.tensor([[1, 2, 3, 4]])
    mask = build_counterfactual_snv_mask(input_ids)

    assert not bool(mask[0, 0, 0])
    assert not bool(mask[0, 1, 1])
    assert not bool(mask[0, 2, 2])
    assert not bool(mask[0, 3, 3])
    assert bool(mask[0, 0, 1])
    assert bool(mask[0, 3, 0])


def test_repo_beat_v9_smoke_config_parses() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(["--config", "configs/beat_v9/smoke.yaml"])
    cfg = config_from_args(args)

    assert cfg.model == "beat-v9"
    assert cfg.loss_balancing == "fixed"
    assert cfg.w_counterfactual_snv == 1.0
    assert cfg.w_counterfactual_severity == 0.25
    assert cfg.w_allele == 0.0
    _assert_no_bio_feature_config(cfg.model_config)


def test_repo_beat_v9_long_context_config_parses() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(["--config", "configs/beat_v9/12m_30ep_32k.yaml"])
    cfg = config_from_args(args)

    assert cfg.model == "beat-v9"
    assert cfg.seq_len == 32768
    assert cfg.batch_size == 4
    assert cfg.grad_accum_steps == 4
    assert cfg.max_steps == 35766
    assert cfg.lr_scheduler == "cosine_rewarm"
    assert cfg.length_warmup_enabled is True
    assert cfg.length_warmup_initial_seq_len == 16384
    assert cfg.w_counterfactual_snv == 1.0
    assert cfg.w_counterfactual_severity == 0.25
    assert cfg.w_counterfactual == 0.0
    assert cfg.w_allele == 0.0
    assert cfg.output_dir == "outputs/lumina_beat_v9_30ep_16kto32k"
    _assert_no_bio_feature_config(cfg.model_config)


def test_beat_v9_uses_dynamic_ddp_kwargs_without_reentrant_checkpointing() -> None:
    cfg = TrainConfig(
        model="beat-v9",
        model_config=resolve_model_config_dict("beat-v9", {}),
    )

    assert ddp_kwargs_for_train_config(cfg) == {"find_unused_parameters": True}


def test_beat_v9_reentrant_checkpointing_keeps_static_ddp_kwargs() -> None:
    cfg = TrainConfig(
        model="beat-v9",
        model_config=resolve_model_config_dict(
            "beat-v9",
            {"activation_checkpointing": True, "checkpoint_use_reentrant": True},
        ),
    )

    assert ddp_kwargs_for_train_config(cfg) == {"static_graph": True}
