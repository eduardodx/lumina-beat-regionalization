from __future__ import annotations

import torch

from eval.clinvar.fusion_lora import (
    AdapterState,
    DynamicFusionLoRALinear,
    StaticFusionLoRALinear,
    collect_fusion_gate_diagnostics,
    convert_lora_backbone_to_dynamic_fusion,
    convert_lora_backbone_to_dynamic_fusion_from_checkpoint_state,
    convert_lora_backbone_to_static_fusion,
    convert_lora_backbone_to_static_fusion_from_checkpoint_state,
    freeze_backbone_for_static_fusion,
)
from eval.clinvar.lora import apply_lora


class _TinyBackbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(4, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class _TinyModel(torch.nn.Module):
    def __init__(self, backbone: torch.nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = torch.nn.Linear(3, 1)


def _adapter_state(value: float = 0.1) -> dict[str, torch.Tensor]:
    return {
        "backbone.proj.lora_a": torch.full((2, 4), value),
        "backbone.proj.lora_b": torch.full((3, 2), value),
    }


def test_static_fusion_replaces_lora_and_trains_only_gate_when_frozen() -> None:
    backbone = _TinyBackbone()
    apply_lora(backbone, rank=2, alpha=4.0, dropout=0.0)

    summary = convert_lora_backbone_to_static_fusion(
        backbone,
        adapters=[
            AdapterState(name="abraom", state_dict=_adapter_state(0.1), scaling=2.0),
            AdapterState(name="gnomad", state_dict=_adapter_state(0.2), scaling=2.0),
        ],
    )

    assert summary.module_names == ("proj",)
    assert isinstance(backbone.proj, StaticFusionLoRALinear)
    assert backbone.proj.adapter_logits.requires_grad
    assert not backbone.proj.path_lora_a.requires_grad
    assert not backbone.proj.population_lora_a[0].requires_grad
    assert backbone(torch.ones((2, 4))).shape == (2, 3)

    model = _TinyModel(backbone)
    freeze_backbone_for_static_fusion(model)
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}

    assert "backbone.proj.adapter_logits" in trainable
    assert "head.weight" in trainable
    assert "backbone.proj.path_lora_a" not in trainable


def test_static_fusion_topology_can_be_rebuilt_from_checkpoint_state() -> None:
    source = _TinyBackbone()
    apply_lora(source, rank=2, alpha=4.0, dropout=0.0)
    convert_lora_backbone_to_static_fusion(
        source,
        adapters=[
            AdapterState(name="abraom", state_dict=_adapter_state(0.1), scaling=2.0),
            AdapterState(name="gnomad", state_dict=_adapter_state(0.2), scaling=3.0),
        ],
    )
    state = source.state_dict()

    target = _TinyBackbone()
    apply_lora(target, rank=2, alpha=4.0, dropout=0.0)
    summary = convert_lora_backbone_to_static_fusion_from_checkpoint_state(
        target,
        model_state_dict=state,
        adapter_names=["abraom", "gnomad"],
    )
    target.load_state_dict(state)

    assert summary.adapter_names == ("abraom", "gnomad")
    assert isinstance(target.proj, StaticFusionLoRALinear)
    assert torch.equal(target.proj.population_scalings, torch.tensor([2.0, 3.0]))


def test_dynamic_fusion_replaces_lora_and_exports_gate_diagnostics() -> None:
    backbone = _TinyBackbone()
    apply_lora(backbone, rank=2, alpha=4.0, dropout=0.0)

    summary = convert_lora_backbone_to_dynamic_fusion(
        backbone,
        adapters=[
            AdapterState(name="abraom", state_dict=_adapter_state(0.1), scaling=2.0),
            AdapterState(name="gnomad", state_dict=_adapter_state(0.2), scaling=2.0),
        ],
        gate_hidden_dim=5,
    )

    assert summary.mode == "dynamic_lora"
    assert summary.module_names == ("proj",)
    assert isinstance(backbone.proj, DynamicFusionLoRALinear)
    assert backbone.proj.gate[0].weight.requires_grad
    assert not backbone.proj.path_lora_a.requires_grad
    assert not backbone.proj.population_lora_a[0].requires_grad

    output = backbone(torch.ones((2, 7, 4)))
    assert output.shape == (2, 7, 3)
    diagnostics = collect_fusion_gate_diagnostics(backbone)
    assert diagnostics is not None
    assert diagnostics["adapter_names"] == ("abraom", "gnomad")
    alpha = diagnostics["gate_alpha"]
    entropy = diagnostics["gate_entropy"]
    assert isinstance(alpha, torch.Tensor)
    assert isinstance(entropy, torch.Tensor)
    assert alpha.shape == (2, 2)
    assert entropy.shape == (2,)
    assert torch.allclose(alpha.sum(dim=-1), torch.ones(2), atol=1e-6)

    model = _TinyModel(backbone)
    freeze_backbone_for_static_fusion(model)
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}

    assert "backbone.proj.gate.0.weight" in trainable
    assert "head.weight" in trainable
    assert "backbone.proj.path_lora_a" not in trainable


def test_dynamic_fusion_topology_can_be_rebuilt_from_checkpoint_state() -> None:
    source = _TinyBackbone()
    apply_lora(source, rank=2, alpha=4.0, dropout=0.0)
    convert_lora_backbone_to_dynamic_fusion(
        source,
        adapters=[
            AdapterState(name="abraom", state_dict=_adapter_state(0.1), scaling=2.0),
            AdapterState(name="gnomad", state_dict=_adapter_state(0.2), scaling=3.0),
        ],
        gate_hidden_dim=6,
    )
    state = source.state_dict()

    target = _TinyBackbone()
    apply_lora(target, rank=2, alpha=4.0, dropout=0.0)
    summary = convert_lora_backbone_to_dynamic_fusion_from_checkpoint_state(
        target,
        model_state_dict=state,
        adapter_names=["abraom", "gnomad"],
    )
    target.load_state_dict(state)

    assert summary.adapter_names == ("abraom", "gnomad")
    assert isinstance(target.proj, DynamicFusionLoRALinear)
    assert target.proj.gate[0].weight.shape[0] == 6
    assert torch.equal(target.proj.population_scalings, torch.tensor([2.0, 3.0]))
