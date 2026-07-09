from __future__ import annotations

from contextlib import nullcontext
from typing import Any, cast

import pytest
import torch
import torch.nn as nn

from eval.clinvar.lora import LoRALinear, apply_lora
from eval.clinvar.model import EndToEndClinVarModel
from eval.clinvar.train import prepare_clinvar_model_for_precision
from src.precision import (
    MXFP8Linear,
    PrecisionPolicy,
    TransformerEngineRuntime,
    resolve_precision_policy,
)
from src.train import prepare_model_for_precision


class _FakeTELinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, **kwargs: Any) -> None:
        super().__init__(
            in_features,
            out_features,
            bias=bias,
            device=kwargs.get("device"),
            dtype=kwargs.get("params_dtype"),
        )


def _fake_fp8_autocast(**_kwargs: Any) -> Any:
    return nullcontext()


def _fake_te_runtime() -> TransformerEngineRuntime:
    return TransformerEngineRuntime(
        backend="transformer_engine",
        linear_cls=_FakeTELinear,
        fp8_autocast=_fake_fp8_autocast,
        recipe=object(),
    )


class _ToyMixer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 4)


class _ToyLuminaBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fwd_mixer = _ToyMixer()
        self.fuse = nn.Linear(8, 4, bias=False)


class _ToyLuminaModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(8, 4)
        self.blocks = nn.ModuleList([_ToyLuminaBlock()])
        self.decoder = nn.Module()
        self.decoder.out_proj = nn.Linear(4, 4)
        self.variant_out_proj = nn.Linear(4, 4)
        self.mutation_effect_head = nn.Sequential(
            nn.Linear(4, 4),
            nn.GELU(),
            nn.Linear(4, 3),
        )
        self.structure_head = nn.Sequential(
            nn.Linear(4, 4),
            nn.GELU(),
            nn.Linear(4, 3),
        )
        self.mlm_head = nn.Linear(4, 8, bias=False)
        self.mlm_head.weight = self.token_emb.weight


class _ToyClinVarAdapter:
    def __init__(self) -> None:
        self._backbone = _ToyLuminaModel()

    @property
    def d_model(self) -> int:
        return 4

    @property
    def backbone(self) -> nn.Module:
        return self._backbone


def test_resolve_precision_policy_auto_cpu_falls_back_to_fp32() -> None:
    policy = resolve_precision_policy(torch.device("cpu"), "auto")

    assert policy.requested == "auto"
    assert policy.resolved == "fp32"
    assert policy.fp8_enabled is False
    assert policy.fallback_reason == "non_cuda_device:cpu"


def test_resolve_precision_policy_auto_cuda_without_fp8_runtime_falls_back_to_bf16(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.precision as precision_mod

    monkeypatch.setattr(precision_mod.torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(precision_mod.torch.cuda, "get_device_capability", lambda _index: (10, 0))
    monkeypatch.setattr(precision_mod, "_load_transformer_engine_runtime", lambda: None)

    policy = resolve_precision_policy(torch.device("cuda", 0), "auto")

    assert policy.resolved == "bf16"
    assert policy.uses_bf16_compute is True
    assert policy.fp8_enabled is False
    assert policy.fallback_reason == "transformer_engine_mxfp8_unavailable"


def test_resolve_precision_policy_auto_blackwell_with_transformer_engine_enables_mxfp8(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.precision as precision_mod

    monkeypatch.setattr(precision_mod.torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(precision_mod.torch.cuda, "get_device_capability", lambda _index: (10, 0))
    monkeypatch.setattr(precision_mod, "_load_transformer_engine_runtime", _fake_te_runtime)

    policy = resolve_precision_policy(torch.device("cuda", 0), "auto")

    assert policy.resolved == "mxfp8"
    assert policy.fp8_enabled is True
    assert policy.fp8_backend == "transformer_engine"
    assert policy.uses_bf16_compute is True


def test_resolve_precision_policy_explicit_mxfp8_errors_when_unsupported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.precision as precision_mod

    monkeypatch.setattr(precision_mod.torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(precision_mod.torch.cuda, "get_device_capability", lambda _index: (10, 0))
    monkeypatch.setattr(precision_mod, "_load_transformer_engine_runtime", lambda: None)

    with pytest.raises(RuntimeError, match="mxfp8 precision was requested"):
        resolve_precision_policy(torch.device("cuda", 0), "mxfp8")


def test_prepare_model_for_precision_converts_supported_linears_and_skips_mlm_head() -> None:
    model = _ToyLuminaModel()
    policy = PrecisionPolicy(
        requested="auto",
        resolved="mxfp8",
        use_autocast=True,
        autocast_device_type="cuda",
        autocast_dtype=torch.bfloat16,
        fp8_enabled=True,
        fp8_backend="transformer_engine",
        fp8_runtime=_fake_te_runtime(),
    )

    updated = prepare_model_for_precision(model, policy)
    block = cast(_ToyLuminaBlock, model.blocks[0])

    assert updated.fp8_module_count == 7
    assert isinstance(block.fuse, MXFP8Linear)
    assert isinstance(model.decoder.out_proj, MXFP8Linear)
    assert isinstance(model.variant_out_proj, MXFP8Linear)
    assert isinstance(model.mutation_effect_head[0], MXFP8Linear)
    assert isinstance(model.mutation_effect_head[2], MXFP8Linear)
    assert isinstance(model.structure_head[0], MXFP8Linear)
    assert isinstance(model.structure_head[2], MXFP8Linear)
    assert not isinstance(block.fwd_mixer.proj, MXFP8Linear)
    assert not isinstance(model.mlm_head, MXFP8Linear)
    assert model.mlm_head.weight is model.token_emb.weight
    assert sorted(block.fuse.state_dict().keys()) == ["weight"]
    assert block.fuse(torch.randn(2, 8)).shape == (2, 4)


def test_prepare_clinvar_model_for_precision_keeps_converted_backbone_linear_lora_eligible() -> None:
    model = EndToEndClinVarModel(_ToyClinVarAdapter(), regime="A", proj_dim=4, hidden_dim=8, dropout=0.0)
    policy = PrecisionPolicy(
        requested="auto",
        resolved="mxfp8",
        use_autocast=True,
        autocast_device_type="cuda",
        autocast_dtype=torch.bfloat16,
        fp8_enabled=True,
        fp8_backend="transformer_engine",
        fp8_runtime=_fake_te_runtime(),
    )

    updated = prepare_clinvar_model_for_precision(model, policy, model_family="lumina")
    summary = apply_lora(model.backbone, rank=2, alpha=4.0, dropout=0.0)
    backbone = cast(_ToyLuminaModel, model.backbone)
    block = cast(_ToyLuminaBlock, backbone.blocks[0])

    assert updated.fp8_module_count >= 8
    assert isinstance(model.head.projection, MXFP8Linear)
    assert "blocks.0.fuse" in summary.module_names
    assert "variant_out_proj" in summary.module_names
    assert "mutation_effect_head.0" not in summary.module_names
    assert isinstance(block.fuse, LoRALinear)
    assert isinstance(block.fuse.base, MXFP8Linear)
