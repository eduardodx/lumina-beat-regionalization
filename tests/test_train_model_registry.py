from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
import torch
import torch.nn as nn

from src.constants import NUM_ALLELE_EFFECT_CLASSES, NUM_MUTATION_EFFECT_CLASSES, SNV_BASES
from src.model_utils import count_parameters
from src.models import build_registered_model, get_model_spec, resolve_model_config_dict
from src.models.beat_v7.local_attn import _attn_mask_has_padding
from src.objectives import GradNormBalancer
from src.precision import PrecisionPolicy
from src.train import (
    TrainConfig,
    aux_loss_warmup_factor,
    build_arg_parser,
    config_from_args,
    ddp_kwargs_for_train_config,
    normalize_runtime_train_config,
    resolve_optional_uncertainty_task_keys,
)

EXPECTED_BIMAMBA_DEFAULT_PARAM_COUNT = 8_293_765


def parse_config(path: str) -> TrainConfig:
    parser = build_arg_parser()
    args = parser.parse_args(["--config", path])
    return config_from_args(args)


def test_config_loads_nested_model_config_and_merges_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "nested_model.yaml"
    config_path.write_text(
        "\n".join(
            [
                "model: bimamba",
                "model_config:",
                "  d_model: 192",
                "  n_layers: 6",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    cfg = parse_config(str(config_path))

    assert cfg.model == "bimamba"
    assert cfg.model_config["d_model"] == 192
    assert cfg.model_config["n_layers"] == 6
    assert cfg.model_config["d_state"] == 64
    assert cfg.model_config["dropout"] == 0.1


def test_config_resolves_default_bimamba_model_config(tmp_path: Path) -> None:
    config_path = tmp_path / "default_model.yaml"
    config_path.write_text("model: bimamba\n", encoding="utf-8")

    cfg = parse_config(str(config_path))

    assert cfg.model == "bimamba"
    assert cfg.model_config == resolve_model_config_dict("bimamba", {})


def test_config_rejects_legacy_preset_key(tmp_path: Path) -> None:
    config_path = tmp_path / "legacy_preset.yaml"
    config_path.write_text("preset: lumina-8m\n", encoding="utf-8")

    with pytest.raises(ValueError, match="preset"):
        parse_config(str(config_path))


def test_config_rejects_legacy_top_level_architecture_keys(tmp_path: Path) -> None:
    config_path = tmp_path / "legacy_arch.yaml"
    config_path.write_text("model: bimamba\nd_model: 128\n", encoding="utf-8")

    with pytest.raises(ValueError, match="model_config"):
        parse_config(str(config_path))


def test_registry_lookup_returns_bimamba_spec() -> None:
    spec = get_model_spec("bimamba")

    assert spec.key == "bimamba"
    assert spec.config_type.__name__ == "BiMambaConfig"


def test_registry_resolves_default_plus_override_config() -> None:
    resolved = resolve_model_config_dict("bimamba", {"d_model": 128, "n_layers": 4})

    assert resolved["d_model"] == 128
    assert resolved["n_layers"] == 4
    assert resolved["d_state"] == 64
    assert resolved["expand"] == 2


def test_registry_builds_default_bimamba_with_current_parameter_count() -> None:
    model = cast(Any, build_registered_model("bimamba"))

    assert count_parameters(model) == EXPECTED_BIMAMBA_DEFAULT_PARAM_COUNT
    assert model.cfg.d_model == 256


def test_registry_resolves_default_beat_v4_model_config() -> None:
    resolved = resolve_model_config_dict("beat-v4", {})

    assert resolved["d_model"] == 256
    assert resolved["n_layers"] == 8
    assert resolved["num_region_classes"] == 5


def test_registry_resolves_default_beat_v5_model_config() -> None:
    resolved = resolve_model_config_dict("beat-v5", {})

    assert resolved["d_model"] == 384
    assert resolved["n_layers"] == 8
    assert resolved["decoder_dim"] == 192
    assert resolved["activation_checkpointing"] is True
    assert resolved["is_mimo"] is False
    assert resolved["num_region_classes"] == 5


def test_registry_accepts_beat_v5_activation_checkpointing_override() -> None:
    resolved = resolve_model_config_dict("beat-v5", {"activation_checkpointing": False})

    assert resolved["activation_checkpointing"] is False


def test_repo_beat_v5_base_config_parses() -> None:
    cfg = parse_config("configs/beat_v5/_base.yaml")

    assert cfg.model == "beat-v5"
    assert cfg.model_config["d_model"] == 384
    assert cfg.model_config["decoder_dim"] == 192
    assert cfg.model_config["activation_checkpointing"] is True
    assert cfg.model_config["is_mimo"] is False
    assert cfg.w_mutation_effect == 0.15


def test_repo_beat_v5_long_context_config_parses() -> None:
    cfg = parse_config("configs/beat_v5/384w_8l_15ep_32k.yaml")

    assert cfg.model == "beat-v5"
    assert cfg.seq_len == 32768
    assert cfg.grad_accum_steps == 4
    assert cfg.lr_scheduler == "constant"


def test_runtime_train_config_disables_mimo_on_pre_hopper_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    precision = PrecisionPolicy(
        requested="bf16",
        resolved="bf16",
        use_autocast=True,
        autocast_device_type="cuda",
        autocast_dtype=torch.bfloat16,
    )
    monkeypatch.setattr("src.train.get_cuda_device_capability", lambda _device: (8, 0))

    cfg, notes = normalize_runtime_train_config(
        TrainConfig(
            model="beat-v5",
            model_config={"is_mimo": True, "mimo_rank": 4, "chunk_size": 16},
        ),
        device=torch.device("cuda", 0),
        precision=precision,
    )

    assert cfg.model_config["is_mimo"] is False
    assert cfg.model_config["chunk_size"] == 64
    assert any("disabled_mimo_pre_hopper_cuda" in note for note in notes)


def test_runtime_train_config_disables_non_reentrant_mamba3_checkpointing_on_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    precision = PrecisionPolicy(
        requested="bf16",
        resolved="bf16",
        use_autocast=True,
        autocast_device_type="cuda",
        autocast_dtype=torch.bfloat16,
    )
    monkeypatch.setattr("src.train.get_cuda_device_capability", lambda _device: (9, 0))

    cfg, notes = normalize_runtime_train_config(
        TrainConfig(
            model="beat-v8",
            model_config={
                "activation_checkpointing": True,
                "checkpoint_use_reentrant": False,
            },
        ),
        device=torch.device("cuda", 0),
        precision=precision,
    )

    assert cfg.model_config["activation_checkpointing"] is False
    assert any("disabled_non_reentrant_activation_checkpointing_for_mamba3_cuda" in note for note in notes)


class _FakeMamba3(nn.Module):
    def __init__(self, d_model: int, **_kwargs: Any) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class _FakeCudaTensor:
    is_cuda = True

    def transpose(self, *_args: Any) -> _FakeCudaTensor:
        return self

    def contiguous(self) -> _FakeCudaTensor:
        return self


def test_registry_builds_beat_v4_and_exposes_mutation_effect_logits(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.models.beat_shared as beat_shared

    monkeypatch.setattr(beat_shared, "_Mamba3", _FakeMamba3)

    model = cast(Any, build_registered_model("beat-v4"))
    outputs = model(torch.ones((2, 8), dtype=torch.long))

    assert outputs["mutation_effect_logits"].shape == (2, 8, len(SNV_BASES), NUM_MUTATION_EFFECT_CLASSES)


def test_registry_builds_beat_v5_and_exposes_decoder_states(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.models.beat_shared as beat_shared

    monkeypatch.setattr(beat_shared, "_Mamba3", _FakeMamba3)

    model = cast(Any, build_registered_model("beat-v5"))
    outputs = model(torch.ones((2, 8), dtype=torch.long))

    assert outputs["decoder_states"].shape == (2, 8, model.cfg.decoder_dim)
    assert outputs["mutation_effect_logits"].shape == (2, 8, len(SNV_BASES), NUM_MUTATION_EFFECT_CLASSES)


def test_beat_v5_base_model_excludes_clinvar_variant_conditioner_parameters(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.models.beat_shared as beat_shared

    monkeypatch.setattr(beat_shared, "_Mamba3", _FakeMamba3)

    model = cast(Any, build_registered_model("beat-v5", {"dropout": 0.0}))
    parameter_names = tuple(name for name, _param in model.named_parameters())

    assert parameter_names
    assert not any(
        name.startswith(
            (
                "decoder_context_proj.",
                "allele_length_embeddings.",
                "variant_condition_proj.",
                "variant_gate_proj.",
                "variant_out_proj.",
            )
        )
        for name in parameter_names
    )


def test_beat_v5_hidden_and_decoder_states_are_rc_consistent(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.models.beat_shared as beat_shared

    monkeypatch.setattr(beat_shared, "_Mamba3", _FakeMamba3)

    model = cast(
        Any,
        build_registered_model(
            "beat-v5",
            {
                "dropout": 0.0,
                "num_region_classes": 0,
            },
        ),
    )
    model.eval()

    input_ids = torch.tensor([[1, 2, 3, 4, 5, 1, 4, 2]], dtype=torch.long)
    rc_ids = model._reverse_complement_ids(input_ids)

    with torch.no_grad():
        features = model.extract_sequence_features(input_ids)
        rc_features = model.extract_sequence_features(rc_ids)

    assert torch.allclose(features["hidden_states"], torch.flip(rc_features["hidden_states"], dims=[1]), atol=1e-5)
    assert torch.allclose(features["decoder_states"], torch.flip(rc_features["decoder_states"], dims=[1]), atol=1e-5)


def test_beat_v5_uses_activation_checkpointing_in_training(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.models.beat_shared as beat_shared
    import src.models.beat_v5.model as beat_v5_model

    monkeypatch.setattr(beat_shared, "_Mamba3", _FakeMamba3)

    checkpoint_calls: list[dict[str, Any]] = []

    def fake_checkpoint(
        fn: Any,
        hidden_states: torch.Tensor,
        *,
        use_reentrant: bool,
    ) -> torch.Tensor:
        checkpoint_calls.append(
            {
                "fn_type": type(getattr(fn, "__self__", fn)).__name__,
                "shape": tuple(hidden_states.shape),
                "use_reentrant": use_reentrant,
            }
        )
        return fn(hidden_states)

    monkeypatch.setattr(beat_v5_model, "activation_checkpoint", fake_checkpoint)

    model = cast(Any, build_registered_model("beat-v5", {"dropout": 0.0, "num_region_classes": 0}))
    model.train()

    outputs = model(torch.ones((2, 8), dtype=torch.long))

    assert outputs["decoder_states"].shape == (2, 8, model.cfg.decoder_dim)
    # Per-mixer checkpointing: 2 mixer calls per block + 1 decoder call.
    assert len(checkpoint_calls) == 2 * model.cfg.n_layers + 1
    assert {call["fn_type"] for call in checkpoint_calls} == {"_FakeMamba3", "BeatV5Decoder"}
    assert all(call["use_reentrant"] is True for call in checkpoint_calls)


def test_beat_v5_skips_activation_checkpointing_in_eval(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.models.beat_shared as beat_shared
    import src.models.beat_v5.model as beat_v5_model

    monkeypatch.setattr(beat_shared, "_Mamba3", _FakeMamba3)

    checkpoint_calls = 0

    def fake_checkpoint(fn: Any, hidden_states: torch.Tensor, *, use_reentrant: bool) -> torch.Tensor:
        nonlocal checkpoint_calls
        _ = (fn, hidden_states, use_reentrant)
        checkpoint_calls += 1
        return hidden_states

    monkeypatch.setattr(beat_v5_model, "activation_checkpoint", fake_checkpoint)

    model = cast(Any, build_registered_model("beat-v5", {"dropout": 0.0, "num_region_classes": 0}))
    model.eval()

    with torch.no_grad():
        outputs = model(torch.ones((2, 8), dtype=torch.long))

    assert outputs["decoder_states"].shape == (2, 8, model.cfg.decoder_dim)
    assert checkpoint_calls == 0


def test_beat_v5_skips_activation_checkpointing_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.models.beat_shared as beat_shared
    import src.models.beat_v5.model as beat_v5_model

    monkeypatch.setattr(beat_shared, "_Mamba3", _FakeMamba3)

    checkpoint_calls = 0

    def fake_checkpoint(fn: Any, hidden_states: torch.Tensor, *, use_reentrant: bool) -> torch.Tensor:
        nonlocal checkpoint_calls
        _ = (fn, hidden_states, use_reentrant)
        checkpoint_calls += 1
        return hidden_states

    monkeypatch.setattr(beat_v5_model, "activation_checkpoint", fake_checkpoint)

    model = cast(
        Any,
        build_registered_model(
            "beat-v5",
            {
                "dropout": 0.0,
                "num_region_classes": 0,
                "activation_checkpointing": False,
            },
        ),
    )
    model.train()
    _ = model(torch.ones((2, 8), dtype=torch.long))

    assert checkpoint_calls == 0


def test_beat_v5_checkpointed_training_backward_propagates_gradients(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.models.beat_shared as beat_shared

    monkeypatch.setattr(beat_shared, "_Mamba3", _FakeMamba3)

    model = cast(
        Any,
        build_registered_model(
            "beat-v5",
            {
                "dropout": 0.0,
                "num_region_classes": 0,
                "activation_checkpointing": True,
            },
        ),
    )
    model.train()

    outputs = model(torch.ones((2, 8), dtype=torch.long))
    loss = outputs["decoder_states"].sum() + outputs["mutation_effect_logits"].sum()
    loss.backward()

    assert model.token_emb.weight.grad is not None
    assert model.decoder.out_proj.weight.grad is not None
    assert model.blocks[0].fwd_proj.weight.grad is not None


# -- beat-v6 tests -----------------------------------------------------------


def test_registry_resolves_default_beat_v6_model_config() -> None:
    resolved = resolve_model_config_dict("beat-v6", {})

    assert resolved["d_model"] == 256
    assert resolved["n_layers"] == 10
    assert resolved["d_state"] == 64
    assert resolved["dropout"] == 0.05
    assert resolved["activation_checkpointing"] is True
    assert resolved["num_region_classes"] == 5
    assert resolved["use_gated_fusion"] is True
    assert resolved["is_mimo"] is True


def test_registry_accepts_beat_v6_activation_checkpointing_override() -> None:
    resolved = resolve_model_config_dict("beat-v6", {"activation_checkpointing": False})

    assert resolved["activation_checkpointing"] is False


def test_registry_builds_beat_v6_default(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.models.beat_shared as beat_shared

    monkeypatch.setattr(beat_shared, "_Mamba3", _FakeMamba3)

    model = cast(Any, build_registered_model("beat-v6"))
    outputs = model(torch.ones((2, 8), dtype=torch.long))

    assert "hidden_states" in outputs
    assert "mlm_logits" in outputs
    assert "phylo100_pred" in outputs
    assert "phylo470_pred" in outputs
    assert "structure_logits" in outputs
    assert "aa_logits" in outputs
    assert "codon_phylo_pred" in outputs
    assert "mutation_effect_logits" in outputs


def test_beat_v6_mutation_effect_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.models.beat_shared as beat_shared

    monkeypatch.setattr(beat_shared, "_Mamba3", _FakeMamba3)

    model = cast(Any, build_registered_model("beat-v6"))
    outputs = model(torch.ones((2, 8), dtype=torch.long))

    assert outputs["mutation_effect_logits"].shape == (2, 8, len(SNV_BASES), NUM_MUTATION_EFFECT_CLASSES)


def test_beat_v6_exposes_sequence_embedding(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.models.beat_shared as beat_shared

    monkeypatch.setattr(beat_shared, "_Mamba3", _FakeMamba3)

    model = cast(Any, build_registered_model("beat-v6"))
    outputs = model(torch.ones((2, 8), dtype=torch.long), return_sequence_embedding=True)

    assert "sequence_embedding" in outputs
    assert outputs["sequence_embedding"].shape == (2, model.cfg.d_model)


def test_beat_v6_has_global_proj(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.models.beat_shared as beat_shared

    monkeypatch.setattr(beat_shared, "_Mamba3", _FakeMamba3)

    model = cast(Any, build_registered_model("beat-v6"))

    assert hasattr(model, "global_proj")


def test_repo_beat_v6_base_config_parses() -> None:
    cfg = parse_config("configs/beat_v6/_base.yaml")

    assert cfg.model == "beat-v6"
    assert cfg.model_config["d_model"] == 256
    assert cfg.model_config["n_layers"] == 10
    assert cfg.model_config["dropout"] == 0.05
    assert cfg.model_config["activation_checkpointing"] is True
    assert cfg.loss_balancing == "gradnorm"
    assert cfg.w_rc == 0.05
    assert cfg.w_mutation_effect == 0.15
    assert cfg.phylo_weighted_mlm is True
    assert cfg.cds_enrichment_fraction == 0.3
    assert cfg.aux_loss_warmup_steps == 500
    assert cfg.memory_preflight_enabled is True


def test_repo_beat_v6_long_context_config_parses() -> None:
    cfg = parse_config("configs/beat_v6/12m_15ep_32k.yaml")

    assert cfg.model == "beat-v6"
    assert cfg.seq_len == 32768
    assert cfg.batch_size == 16
    assert cfg.grad_accum_steps == 1
    assert cfg.max_steps == 17883
    assert cfg.lr_scheduler == "cosine_rewarm"
    assert cfg.aux_loss_warmup_steps == 1000
    assert cfg.length_warmup_enabled is True
    assert cfg.length_warmup_initial_seq_len == 16384
    assert cfg.length_warmup_transition_fraction == pytest.approx(0.8)
    assert cfg.length_warmup_stage1_warmup_fraction == pytest.approx(0.05)
    assert cfg.length_warmup_stage1_end_lr_scale == pytest.approx(0.1)
    assert cfg.length_warmup_stage2_warmup_fraction == pytest.approx(0.05)
    assert cfg.length_warmup_stage2_peak_lr_scale == pytest.approx(0.2)
    assert cfg.length_warmup_final_lr_scale == pytest.approx(0.01)
    assert cfg.memory_preflight_enabled is True


# -- beat-v7 tests -----------------------------------------------------------


def test_registry_resolves_default_beat_v7_model_config() -> None:
    resolved = resolve_model_config_dict("beat-v7", {})

    assert resolved["d_model"] == 256
    assert resolved["n_layers"] == 8
    assert resolved["d_state"] == 128
    assert resolved["attention_every_n_blocks"] == 3
    assert resolved["attention_window"] == 256
    assert resolved["attention_n_heads"] == 4
    assert resolved["activation_checkpointing"] is True
    assert resolved["is_mimo"] is False
    assert resolved["num_region_classes"] == 5


def test_registry_accepts_beat_v7_attention_override() -> None:
    resolved = resolve_model_config_dict(
        "beat-v7",
        {
            "attention_every_n_blocks": 2,
            "attention_window": 128,
            "attention_n_heads": 8,
        },
    )

    assert resolved["attention_every_n_blocks"] == 2
    assert resolved["attention_window"] == 128
    assert resolved["attention_n_heads"] == 8


def test_registry_builds_beat_v7_default(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.models.beat_shared as beat_shared

    monkeypatch.setattr(beat_shared, "_Mamba3", _FakeMamba3)

    model = cast(Any, build_registered_model("beat-v7"))
    outputs = model(torch.ones((2, 8), dtype=torch.long), attention_mask=torch.ones((2, 8), dtype=torch.long))

    assert "hidden_states" in outputs
    assert "mlm_logits" in outputs
    assert "phylo100_pred" in outputs
    assert "phylo470_pred" in outputs
    assert "structure_logits" in outputs
    assert "aa_logits" in outputs
    assert "codon_phylo_pred" in outputs
    assert "mutation_effect_logits" in outputs
    assert "codon_logits" in outputs
    assert outputs["codon_logits"].shape == (2, 8, 64)


def test_beat_v7_exposes_sequence_embedding(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.models.beat_shared as beat_shared

    monkeypatch.setattr(beat_shared, "_Mamba3", _FakeMamba3)

    model = cast(Any, build_registered_model("beat-v7"))
    outputs = model(
        torch.ones((2, 8), dtype=torch.long),
        attention_mask=torch.ones((2, 8), dtype=torch.long),
        return_sequence_embedding=True,
    )

    assert "sequence_embedding" in outputs
    assert outputs["sequence_embedding"].shape == (2, model.cfg.d_model)


def test_beat_v7_has_local_attention_layers(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.models.beat_shared as beat_shared

    monkeypatch.setattr(beat_shared, "_Mamba3", _FakeMamba3)

    model = cast(Any, build_registered_model("beat-v7"))

    assert len(model.attn_layers) == 2
    assert len(model.attn_norms) == 2


def test_repo_beat_v7_base_config_parses() -> None:
    cfg = parse_config("configs/beat_v7/_base.yaml")

    assert cfg.model == "beat-v7"
    assert cfg.model_config["d_model"] == 256
    assert cfg.model_config["n_layers"] == 8
    assert cfg.model_config["d_state"] == 128
    assert cfg.model_config["attention_every_n_blocks"] == 3
    assert cfg.model_config["attention_window"] == 256
    assert cfg.model_config["attention_n_heads"] == 4
    assert cfg.loss_balancing == "uncertainty"
    assert cfg.counterfactual_weighting == "fixed"
    assert cfg.counterfactual_local_similarity_target == pytest.approx(0.8)
    assert cfg.aux_loss_warmup_steps == 1000
    assert cfg.w_encode == 0.0
    assert cfg.model_config["num_encode_tracks"] == 0
    assert cfg.param_budget["backbone_max_params"] == 30000000
    assert "counterfactual" not in resolve_optional_uncertainty_task_keys(cfg)


def test_repo_beat_v7_long_context_config_parses() -> None:
    cfg = parse_config("configs/beat_v7/12m_15ep_32k.yaml")

    assert cfg.model == "beat-v7"
    assert cfg.seq_len == 32768
    assert cfg.batch_size == 16
    assert cfg.max_steps == 17883
    assert cfg.lr_scheduler == "cosine_rewarm"
    assert cfg.length_warmup_enabled is True
    assert cfg.length_warmup_initial_seq_len == 16384


def test_registry_resolves_default_beat_v8_model_config() -> None:
    resolved = resolve_model_config_dict("beat-v8", {})

    assert resolved["d_model"] == 256
    assert resolved["n_layers"] == 8
    assert resolved["d_state"] == 128
    assert resolved["attention_every_n_blocks"] == 3
    assert resolved["attention_window"] == 256
    assert resolved["num_allele_effect_classes"] == NUM_ALLELE_EFFECT_CLASSES
    assert resolved["allele_context_radius"] == 64
    assert resolved["activation_checkpointing"] is False
    assert resolved["checkpoint_use_reentrant"] is False


def test_beat_v8_scores_same_locus_alleles(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.models.beat_shared as beat_shared

    monkeypatch.setattr(beat_shared, "_Mamba3", _FakeMamba3)

    model = cast(Any, build_registered_model("beat-v8", {"dropout": 0.0}))
    ref = torch.tensor([[1, 2, 3, 4, 1, 2]], dtype=torch.long)
    alt = ref.unsqueeze(1).repeat(1, 3, 1)
    alt[0, 0, 2] = 1
    alt[0, 1, 2] = 2
    alt[0, 2, 2] = 4
    scored = model.score_alleles_from_ids(
        ref,
        alt,
        torch.tensor([2], dtype=torch.long),
        torch.tensor([[1, 2, 4]], dtype=torch.long),
        attention_mask=torch.ones_like(ref),
        alt_attention_mask=torch.ones_like(alt),
    )

    assert scored["allele_repr"].shape == (1, 3, model.cfg.d_model)
    assert scored["allele_effect_logits"].shape == (1, 3, NUM_ALLELE_EFFECT_CLASSES)
    assert scored["allele_severity_score"].shape == (1, 3)
    assert scored["allele_swap_severity_score"].shape == (1, 3)


def test_beat_v8_can_use_non_reentrant_activation_checkpointing_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.models.beat_shared as beat_shared

    monkeypatch.setattr(beat_shared, "_Mamba3", _FakeMamba3)

    checkpoint_calls: list[dict[str, Any]] = []

    def fake_checkpoint(fn: Any, *inputs: torch.Tensor, use_reentrant: bool) -> torch.Tensor:
        checkpoint_calls.append(
            {
                "fn_type": type(getattr(fn, "__self__", fn)).__name__,
                "num_inputs": len(inputs),
                "use_reentrant": use_reentrant,
            }
        )
        return fn(*inputs)

    monkeypatch.setattr(beat_shared, "activation_checkpoint", fake_checkpoint)

    model = cast(
        Any,
        build_registered_model(
            "beat-v8",
            {
                "dropout": 0.0,
                "activation_checkpointing": True,
                "checkpoint_use_reentrant": False,
            },
        ),
    )
    model.train()
    outputs = model(torch.ones((2, 8), dtype=torch.long))

    assert outputs["hidden_states"].shape == (2, 8, model.cfg.d_model)
    assert len(checkpoint_calls) == 2 * model.cfg.n_layers
    assert all(call["use_reentrant"] is False for call in checkpoint_calls)


def test_repo_beat_v8_long_context_config_parses() -> None:
    cfg = parse_config("configs/beat_v8/12m_30ep_32k.yaml")

    assert cfg.model == "beat-v8"
    assert cfg.model_config["activation_checkpointing"] is False
    assert cfg.model_config["checkpoint_use_reentrant"] is False
    assert cfg.seq_len == 32768
    assert cfg.batch_size == 4
    assert cfg.grad_accum_steps == 4
    assert cfg.max_steps == 35766
    assert cfg.w_allele == pytest.approx(0.35)
    assert cfg.w_counterfactual == pytest.approx(0.10)
    assert cfg.w_mutation_effect == pytest.approx(0.05)
    assert cfg.allele_score_window == 4096
    assert cfg.allele_max_rows_per_batch == 2
    assert cfg.allele_rank_every_n_steps == 4
    assert cfg.auxiliary_schedule == "alternate_allele_counterfactual"
    assert cfg.lr_scheduler == "cosine_rewarm"
    assert cfg.length_warmup_enabled is True
    assert "allele" in resolve_optional_uncertainty_task_keys(cfg)


def test_v8_alternating_auxiliary_uses_dynamic_ddp_kwargs() -> None:
    cfg = TrainConfig(
        model="beat-v8",
        model_config=resolve_model_config_dict("beat-v8", {}),
        auxiliary_schedule="alternate_allele_counterfactual",
        w_allele=0.35,
        w_counterfactual=0.10,
    )

    assert ddp_kwargs_for_train_config(cfg) == {"find_unused_parameters": True}


def test_reentrant_checkpointing_keeps_static_ddp_kwargs() -> None:
    cfg = TrainConfig(
        model="beat-v8",
        model_config=resolve_model_config_dict(
            "beat-v8",
            {"activation_checkpointing": True, "checkpoint_use_reentrant": True},
        ),
        auxiliary_schedule="alternate_allele_counterfactual",
        w_allele=0.35,
        w_counterfactual=0.10,
    )

    assert ddp_kwargs_for_train_config(cfg) == {"static_graph": True}


@pytest.mark.parametrize("want_rc", [False, True])
@pytest.mark.parametrize("counterfactual_active", [False, True])
@pytest.mark.parametrize("has_allele_kwargs", [False, True])
def test_execute_model_forward_passes_runs_main_forward_last(
    want_rc: bool,
    counterfactual_active: bool,
    has_allele_kwargs: bool,
) -> None:
    """DDP correctness: main forward (return_token_heads=True) must be the last call.

    With find_unused_parameters=True, DDP's last prepare_for_backward determines
    which parameters are flagged "used"; running an auxiliary forward last would
    mark every token head (e.g., region_head) as unused and trigger
    "marked ready twice" errors when autograd hooks fire on those heads.
    """
    from src.train import _execute_model_forward_passes

    calls: list[dict[str, Any]] = []

    class _RecordingModel:
        def __call__(self, **kwargs: Any) -> dict[str, torch.Tensor]:
            calls.append(kwargs)
            return {"hidden_states": torch.zeros(1)}

    batch: dict[str, Any] = {
        "input_ids": torch.zeros(1, dtype=torch.long),
        "attention_mask": torch.zeros(1),
        "rc_input_ids": torch.zeros(1, dtype=torch.long),
        "rc_attention_mask": torch.zeros(1),
        "alt_input_ids": torch.zeros(1, dtype=torch.long),
        "alt_attention_mask": torch.zeros(1),
    }
    allele_forward_kwargs: dict[str, torch.Tensor] = (
        {"allele_ref_input_ids": torch.zeros(1, dtype=torch.long)} if has_allele_kwargs else {}
    )

    _execute_model_forward_passes(
        cast(Any, _RecordingModel()),
        batch,
        want_rc=want_rc,
        counterfactual_active=counterfactual_active,
        allele_forward_kwargs=allele_forward_kwargs,
    )

    expected_call_count = 1 + int(want_rc) + int(counterfactual_active)
    assert len(calls) == expected_call_count
    # The last call is the main forward — return_token_heads is left at its
    # default (True) so DDP's used-parameter trace covers all token heads.
    assert calls[-1].get("return_token_heads", True) is True
    # All non-final calls are auxiliaries with token heads disabled.
    for call in calls[:-1]:
        assert call["return_token_heads"] is False
    if has_allele_kwargs:
        assert "allele_ref_input_ids" in calls[-1]


def test_config_rejects_encode_weight_without_track_specs(tmp_path: Path) -> None:
    config_path = tmp_path / "beat_v7_encode_weight_only.yaml"
    config_path.write_text(
        "\n".join(
            [
                "model: beat-v7",
                "w_encode: 0.5",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="w_encode requires a non-empty encode_track_specs list"):
        parse_config(str(config_path))


def test_config_rejects_encode_track_count_mismatch(tmp_path: Path) -> None:
    config_path = tmp_path / "beat_v7_encode_mismatch.yaml"
    config_path.write_text(
        "\n".join(
            [
                "model: beat-v7",
                "w_encode: 0.5",
                "model_config:",
                "  num_encode_tracks: 2",
                "encode_track_specs:",
                "  - name: dnase",
                "    bw_path: data/encode/dnase/example.bw",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"model_config\.num_encode_tracks must match len\(encode_track_specs\)"):
        parse_config(str(config_path))


def test_beat_v7_flash_attention_mask_helper_treats_all_ones_as_unpadded() -> None:
    assert _attn_mask_has_padding(None) is False
    assert _attn_mask_has_padding(torch.ones((2, 8), dtype=torch.long)) is False
    assert _attn_mask_has_padding(torch.tensor([[1, 1, 0, 0]], dtype=torch.long)) is True


def test_beat_v7_flash_attention_falls_back_when_optional_kernel_contract_is_unexpected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.models.beat_v7 import local_attn as local_attn_module

    module = local_attn_module.LocalWindowAttention(d_model=16, n_heads=4, window=32, dropout=0.0)
    module._flash_attn_supports_window = True
    monkeypatch.setattr(local_attn_module, "_flash_attn_func", lambda *_args, **_kwargs: ("unexpected",))

    result = module._flash_forward(
        cast(Any, _FakeCudaTensor()),
        cast(Any, _FakeCudaTensor()),
        cast(Any, _FakeCudaTensor()),
        None,
    )

    assert result is None
    assert not module._flash_attn_supports_window


def test_beat_v6_uses_activation_checkpointing_in_training(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.models.beat_shared as beat_shared

    monkeypatch.setattr(beat_shared, "_Mamba3", _FakeMamba3)

    checkpoint_calls: list[dict[str, Any]] = []

    def fake_checkpoint(fn: Any, *inputs: torch.Tensor, use_reentrant: bool) -> torch.Tensor:
        checkpoint_calls.append(
            {
                "fn_type": type(getattr(fn, "__self__", fn)).__name__,
                "num_inputs": len(inputs),
                "use_reentrant": use_reentrant,
            }
        )
        return fn(*inputs)

    monkeypatch.setattr(beat_shared, "activation_checkpoint", fake_checkpoint)

    model = cast(Any, build_registered_model("beat-v6"))
    model.train()

    outputs = model(torch.ones((2, 8), dtype=torch.long))

    assert outputs["hidden_states"].shape == (2, 8, model.cfg.d_model)
    # Per-mixer checkpointing: fwd_mixer + bwd_mixer per block.
    assert len(checkpoint_calls) == 2 * model.cfg.n_layers
    assert {call["fn_type"] for call in checkpoint_calls} == {"_FakeMamba3"}
    assert all(call["num_inputs"] == 1 for call in checkpoint_calls)
    assert all(call["use_reentrant"] is True for call in checkpoint_calls)


def test_beat_v6_skips_activation_checkpointing_in_eval(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.models.beat_shared as beat_shared

    monkeypatch.setattr(beat_shared, "_Mamba3", _FakeMamba3)

    checkpoint_calls = 0

    def fake_checkpoint(fn: Any, *inputs: torch.Tensor, use_reentrant: bool) -> torch.Tensor:
        nonlocal checkpoint_calls
        _ = (fn, inputs, use_reentrant)
        checkpoint_calls += 1
        return inputs[0]

    monkeypatch.setattr(beat_shared, "activation_checkpoint", fake_checkpoint)

    model = cast(Any, build_registered_model("beat-v6"))
    model.eval()

    with torch.no_grad():
        outputs = model(torch.ones((2, 8), dtype=torch.long))

    assert outputs["hidden_states"].shape == (2, 8, model.cfg.d_model)
    assert checkpoint_calls == 0


def test_beat_v6_skips_activation_checkpointing_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.models.beat_shared as beat_shared

    monkeypatch.setattr(beat_shared, "_Mamba3", _FakeMamba3)

    checkpoint_calls = 0

    def fake_checkpoint(fn: Any, *inputs: torch.Tensor, use_reentrant: bool) -> torch.Tensor:
        nonlocal checkpoint_calls
        _ = (fn, inputs, use_reentrant)
        checkpoint_calls += 1
        return inputs[0]

    monkeypatch.setattr(beat_shared, "activation_checkpoint", fake_checkpoint)

    model = cast(Any, build_registered_model("beat-v6", {"activation_checkpointing": False}))
    model.train()
    _ = model(torch.ones((2, 8), dtype=torch.long))

    assert checkpoint_calls == 0


def test_beat_v6_checkpointed_training_backward_propagates_gradients(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.models.beat_shared as beat_shared

    monkeypatch.setattr(beat_shared, "_Mamba3", _FakeMamba3)

    model = cast(Any, build_registered_model("beat-v6", {"activation_checkpointing": True}))
    model.train()

    outputs = model(torch.ones((2, 8), dtype=torch.long), return_sequence_embedding=True)
    loss = (
        outputs["hidden_states"].sum()
        + outputs["mutation_effect_logits"].sum()
        + outputs["sequence_embedding"].sum()
    )
    loss.backward()

    assert model.token_emb.weight.grad is not None
    assert model.global_proj.weight.grad is not None
    assert model.blocks[0].fwd_proj.weight.grad is not None


# -- aux_loss_warmup_factor tests --------------------------------------------


def test_aux_loss_warmup_factor_disabled() -> None:
    assert aux_loss_warmup_factor(0, 0) == 1.0
    assert aux_loss_warmup_factor(100, 0) == 1.0
    assert aux_loss_warmup_factor(50, -1) == 1.0


def test_aux_loss_warmup_factor_ramp() -> None:
    assert aux_loss_warmup_factor(0, 100) == 0.0
    assert aux_loss_warmup_factor(50, 100) == pytest.approx(0.5)
    assert aux_loss_warmup_factor(100, 100) == pytest.approx(1.0)
    assert aux_loss_warmup_factor(200, 100) == 1.0


# -- GradNormBalancer tests --------------------------------------------------


def test_gradnorm_balancer_initial_uniform_weights() -> None:
    balancer = GradNormBalancer(["mlm", "phylo100", "structure"])
    weights = balancer.get_weights()

    assert weights == {"mlm": 1.0, "phylo100": 1.0, "structure": 1.0}


def test_gradnorm_balancer_updates_weights() -> None:
    balancer = GradNormBalancer(["mlm", "phylo100"], alpha=1.5, ema_decay=0.0)

    # First update initializes.
    balancer.update({"loss_mlm": 1.0, "loss_phylo100": 1.0})
    assert balancer.get_weights()["mlm"] == pytest.approx(1.0)
    assert balancer.get_weights()["phylo100"] == pytest.approx(1.0)

    # Second update: mlm dropped to 0.5 (converging fast), phylo100 stayed at 1.0.
    # With ema_decay=0.0, EMA = current value exactly.
    balancer.update({"loss_mlm": 0.5, "loss_phylo100": 1.0})
    weights = balancer.get_weights()
    # phylo100 is lagging (inv_rate = 1.0^1.5 = 1.0), mlm converging (inv_rate = 0.5^1.5 ≈ 0.354)
    # phylo100 should get higher weight.
    assert weights["phylo100"] > weights["mlm"]
