"""LoRA adapter fusion for regional ClinVar experiments."""

from __future__ import annotations

import logging
import re
import tarfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from eval.clinvar.lora import LoRALinear

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class AdapterState:
    """Frozen population adapter state used by static fusion."""

    name: str
    state_dict: Mapping[str, torch.Tensor]
    scaling: float


@dataclass(frozen=True)
class FusionSummary:
    """Summary of static fusion conversion for reproducibility logging."""

    mode: str
    adapter_names: tuple[str, ...]
    module_names: tuple[str, ...]
    trainable_gate_params: int

    @property
    def module_count(self) -> int:
        return len(self.module_names)


def _sanitize_adapter_name(name: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9_]+", "_", name).strip("_")
    if not sanitized:
        raise ValueError(f"Invalid empty adapter name after sanitization: {name!r}")
    return sanitized


class StaticFusionLoRALinear(nn.Module):
    """Frozen path LoRA plus trainable static softmax over population LoRAs."""

    def __init__(
        self,
        source: LoRALinear,
        *,
        adapter_names: Sequence[str],
        population_lora_a: Sequence[torch.Tensor],
        population_lora_b: Sequence[torch.Tensor],
        population_scalings: Sequence[float],
    ) -> None:
        super().__init__()
        if not adapter_names:
            raise ValueError("StaticFusionLoRALinear requires at least one population adapter.")
        if not (
            len(adapter_names)
            == len(population_lora_a)
            == len(population_lora_b)
            == len(population_scalings)
        ):
            raise ValueError("Population adapter names, tensors, and scalings must have identical lengths.")

        self.base = source.base
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

        self.path_scaling = float(source.scaling)
        self.path_lora_a = nn.Parameter(source.lora_a.detach().clone(), requires_grad=False)
        self.path_lora_b = nn.Parameter(source.lora_b.detach().clone(), requires_grad=False)

        ref_device = source.lora_a.device
        ref_dtype = source.lora_a.dtype
        self.adapter_names = tuple(_sanitize_adapter_name(name) for name in adapter_names)
        self.population_lora_a = nn.ParameterList(
            [
                nn.Parameter(tensor.detach().to(device=ref_device, dtype=ref_dtype).clone(), requires_grad=False)
                for tensor in population_lora_a
            ]
        )
        self.population_lora_b = nn.ParameterList(
            [
                nn.Parameter(tensor.detach().to(device=ref_device, dtype=ref_dtype).clone(), requires_grad=False)
                for tensor in population_lora_b
            ]
        )
        self.register_buffer(
            "population_scalings",
            torch.tensor([float(value) for value in population_scalings], device=ref_device, dtype=torch.float32),
            persistent=True,
        )
        self.adapter_logits = nn.Parameter(torch.zeros(len(self.adapter_names), device=ref_device, dtype=ref_dtype))

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.base.bias

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        path_out = F.linear(x, self.path_lora_a) @ self.path_lora_b.T
        fused = base_out + path_out * self.path_scaling

        weights = torch.softmax(self.adapter_logits.float(), dim=0).to(dtype=x.dtype)
        for idx, (lora_a, lora_b) in enumerate(zip(self.population_lora_a, self.population_lora_b, strict=True)):
            pop_out = F.linear(x, lora_a) @ lora_b.T
            scaling = self.population_scalings[idx].to(device=x.device, dtype=x.dtype)
            fused = fused + weights[idx] * pop_out * scaling
        return fused


class DynamicFusionLoRALinear(nn.Module):
    """Frozen path LoRA plus per-example gate over frozen population LoRAs.

    The gate is conditioned only on hidden states flowing through the layer.
    It deliberately does not consume ClinVar provenance fields, keeping
    submitter metadata out of inference-time decisions.
    """

    def __init__(
        self,
        source: LoRALinear,
        *,
        adapter_names: Sequence[str],
        population_lora_a: Sequence[torch.Tensor],
        population_lora_b: Sequence[torch.Tensor],
        population_scalings: Sequence[float],
        gate_hidden_dim: int = 64,
        gate_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if gate_hidden_dim <= 0:
            raise ValueError(f"gate_hidden_dim must be positive, got {gate_hidden_dim}.")
        if not adapter_names:
            raise ValueError("DynamicFusionLoRALinear requires at least one population adapter.")
        if not (
            len(adapter_names)
            == len(population_lora_a)
            == len(population_lora_b)
            == len(population_scalings)
        ):
            raise ValueError("Population adapter names, tensors, and scalings must have identical lengths.")

        self.base = source.base
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

        self.path_scaling = float(source.scaling)
        self.path_lora_a = nn.Parameter(source.lora_a.detach().clone(), requires_grad=False)
        self.path_lora_b = nn.Parameter(source.lora_b.detach().clone(), requires_grad=False)

        ref_device = source.lora_a.device
        ref_dtype = source.lora_a.dtype
        self.adapter_names = tuple(_sanitize_adapter_name(name) for name in adapter_names)
        self.population_lora_a = nn.ParameterList(
            [
                nn.Parameter(tensor.detach().to(device=ref_device, dtype=ref_dtype).clone(), requires_grad=False)
                for tensor in population_lora_a
            ]
        )
        self.population_lora_b = nn.ParameterList(
            [
                nn.Parameter(tensor.detach().to(device=ref_device, dtype=ref_dtype).clone(), requires_grad=False)
                for tensor in population_lora_b
            ]
        )
        self.register_buffer(
            "population_scalings",
            torch.tensor([float(value) for value in population_scalings], device=ref_device, dtype=torch.float32),
            persistent=True,
        )
        self.gate = nn.Sequential(
            nn.Linear(source.in_features, gate_hidden_dim),
            nn.GELU(),
            nn.Dropout(float(gate_dropout)),
            nn.Linear(gate_hidden_dim, len(self.adapter_names)),
        ).to(device=ref_device, dtype=ref_dtype)
        self.last_gate_weights: torch.Tensor | None = None
        self.last_gate_entropy: torch.Tensor | None = None

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.base.bias

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def _pooled_gate_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 2:
            raise ValueError(f"DynamicFusionLoRALinear expects batched input, got shape {tuple(x.shape)}.")
        if x.ndim == 2:
            return x
        return x.reshape(x.shape[0], -1, x.shape[-1]).mean(dim=1)

    def _broadcast_weight(self, weights: torch.Tensor, adapter_index: int, target: torch.Tensor) -> torch.Tensor:
        shape = [weights.shape[0], *([1] * (target.ndim - 1))]
        return weights[:, adapter_index].reshape(shape).to(dtype=target.dtype, device=target.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        path_out = F.linear(x, self.path_lora_a) @ self.path_lora_b.T
        fused = base_out + path_out * self.path_scaling

        gate_logits = self.gate(self._pooled_gate_input(x))
        weights = torch.softmax(gate_logits.float(), dim=-1).to(dtype=x.dtype)
        entropy = -(weights.float() * weights.float().clamp_min(1e-8).log()).sum(dim=-1)
        self.last_gate_weights = weights.detach()
        self.last_gate_entropy = entropy.detach()

        for idx, (lora_a, lora_b) in enumerate(zip(self.population_lora_a, self.population_lora_b, strict=True)):
            pop_out = F.linear(x, lora_a) @ lora_b.T
            scaling = self.population_scalings[idx].to(device=x.device, dtype=x.dtype)
            fused = fused + self._broadcast_weight(weights, idx, pop_out) * pop_out * scaling
        return fused


def _resolve_child(root: nn.Module, name: str) -> tuple[nn.Module, str]:
    parts = name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def _adapter_lora_tensors(
    adapter: AdapterState,
    module_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    prefix = f"backbone.{module_name}"
    key_a = f"{prefix}.lora_a"
    key_b = f"{prefix}.lora_b"
    if key_a not in adapter.state_dict or key_b not in adapter.state_dict:
        raise KeyError(f"Adapter {adapter.name!r} is missing LoRA tensors for module {module_name!r}.")
    return adapter.state_dict[key_a], adapter.state_dict[key_b]


def convert_lora_backbone_to_static_fusion(
    backbone: nn.Module,
    *,
    adapters: Sequence[AdapterState],
) -> FusionSummary:
    """Replace every ``LoRALinear`` in ``backbone`` with static fusion wrappers."""

    if not adapters:
        raise ValueError("At least one AdapterState is required for static fusion.")
    module_names = [name for name, module in backbone.named_modules() if isinstance(module, LoRALinear)]
    if not module_names:
        raise ValueError("No LoRALinear modules found; call apply_lora before static fusion conversion.")

    adapter_names = tuple(adapter.name for adapter in adapters)
    for module_name in module_names:
        parent, attr = _resolve_child(backbone, module_name)
        source = getattr(parent, attr)
        if not isinstance(source, LoRALinear):
            raise TypeError(f"Expected LoRALinear at {module_name!r}, got {type(source).__name__}.")
        population_a: list[torch.Tensor] = []
        population_b: list[torch.Tensor] = []
        scalings: list[float] = []
        for adapter in adapters:
            lora_a, lora_b = _adapter_lora_tensors(adapter, module_name)
            if tuple(lora_a.shape[1:]) != tuple(source.lora_a.shape[1:]):
                raise ValueError(
                    f"Adapter {adapter.name!r} lora_a shape {tuple(lora_a.shape)} is incompatible "
                    f"with module {module_name!r} shape {tuple(source.lora_a.shape)}."
                )
            if int(lora_b.shape[0]) != int(source.lora_b.shape[0]):
                raise ValueError(
                    f"Adapter {adapter.name!r} lora_b shape {tuple(lora_b.shape)} is incompatible "
                    f"with module {module_name!r} shape {tuple(source.lora_b.shape)}."
                )
            population_a.append(lora_a)
            population_b.append(lora_b)
            scalings.append(adapter.scaling)
        setattr(
            parent,
            attr,
            StaticFusionLoRALinear(
                source,
                adapter_names=adapter_names,
                population_lora_a=population_a,
                population_lora_b=population_b,
                population_scalings=scalings,
            ),
        )

    gate_params = sum(
        parameter.numel()
        for name, parameter in backbone.named_parameters()
        if name.endswith("adapter_logits") and parameter.requires_grad
    )
    log.info(
        "Converted %d LoRA modules to static fusion with adapters=%s (%d trainable gate params).",
        len(module_names),
        list(adapter_names),
        gate_params,
    )
    return FusionSummary(
        mode="static_lora",
        adapter_names=adapter_names,
        module_names=tuple(module_names),
        trainable_gate_params=gate_params,
    )


def convert_lora_backbone_to_dynamic_fusion(
    backbone: nn.Module,
    *,
    adapters: Sequence[AdapterState],
    gate_hidden_dim: int = 64,
    gate_dropout: float = 0.0,
) -> FusionSummary:
    """Replace every ``LoRALinear`` in ``backbone`` with dynamic fusion wrappers."""

    if not adapters:
        raise ValueError("At least one AdapterState is required for dynamic fusion.")
    module_names = [name for name, module in backbone.named_modules() if isinstance(module, LoRALinear)]
    if not module_names:
        raise ValueError("No LoRALinear modules found; call apply_lora before dynamic fusion conversion.")

    adapter_names = tuple(adapter.name for adapter in adapters)
    for module_name in module_names:
        parent, attr = _resolve_child(backbone, module_name)
        source = getattr(parent, attr)
        if not isinstance(source, LoRALinear):
            raise TypeError(f"Expected LoRALinear at {module_name!r}, got {type(source).__name__}.")
        population_a: list[torch.Tensor] = []
        population_b: list[torch.Tensor] = []
        scalings: list[float] = []
        for adapter in adapters:
            lora_a, lora_b = _adapter_lora_tensors(adapter, module_name)
            if tuple(lora_a.shape[1:]) != tuple(source.lora_a.shape[1:]):
                raise ValueError(
                    f"Adapter {adapter.name!r} lora_a shape {tuple(lora_a.shape)} is incompatible "
                    f"with module {module_name!r} shape {tuple(source.lora_a.shape)}."
                )
            if int(lora_b.shape[0]) != int(source.lora_b.shape[0]):
                raise ValueError(
                    f"Adapter {adapter.name!r} lora_b shape {tuple(lora_b.shape)} is incompatible "
                    f"with module {module_name!r} shape {tuple(source.lora_b.shape)}."
                )
            population_a.append(lora_a)
            population_b.append(lora_b)
            scalings.append(adapter.scaling)
        setattr(
            parent,
            attr,
            DynamicFusionLoRALinear(
                source,
                adapter_names=adapter_names,
                population_lora_a=population_a,
                population_lora_b=population_b,
                population_scalings=scalings,
                gate_hidden_dim=gate_hidden_dim,
                gate_dropout=gate_dropout,
            ),
        )

    gate_params = sum(
        parameter.numel()
        for name, parameter in backbone.named_parameters()
        if ".gate." in name and parameter.requires_grad
    )
    log.info(
        "Converted %d LoRA modules to dynamic fusion with adapters=%s (%d trainable gate params).",
        len(module_names),
        list(adapter_names),
        gate_params,
    )
    return FusionSummary(
        mode="dynamic_lora",
        adapter_names=adapter_names,
        module_names=tuple(module_names),
        trainable_gate_params=gate_params,
    )


def convert_lora_backbone_to_static_fusion_from_checkpoint_state(
    backbone: nn.Module,
    *,
    model_state_dict: Mapping[str, torch.Tensor],
    adapter_names: Sequence[str],
) -> FusionSummary:
    """Rebuild static-fusion topology before loading a saved fusion checkpoint."""

    adapters: list[AdapterState] = []
    sanitized_names = tuple(_sanitize_adapter_name(name) for name in adapter_names)
    module_names = [name for name, module in backbone.named_modules() if isinstance(module, LoRALinear)]
    if not module_names:
        raise ValueError("No LoRALinear modules found; call apply_lora before fusion reconstruction.")
    first_module = module_names[0]
    for idx, name in enumerate(sanitized_names):
        state: dict[str, torch.Tensor] = {}
        for module_name in module_names:
            state_prefix = _resolve_state_prefix(model_state_dict, module_name)
            key_a = f"{state_prefix}.population_lora_a.{idx}"
            key_b = f"{state_prefix}.population_lora_b.{idx}"
            if key_a not in model_state_dict or key_b not in model_state_dict:
                raise KeyError(f"Fusion checkpoint is missing population tensors for adapter {name!r}.")
            state[f"backbone.{module_name}.lora_a"] = model_state_dict[key_a]
            state[f"backbone.{module_name}.lora_b"] = model_state_dict[key_b]
        scaling_key = f"{_resolve_state_prefix(model_state_dict, first_module)}.population_scalings"
        scaling_tensor = model_state_dict.get(scaling_key)
        scaling = float(scaling_tensor[idx].item()) if isinstance(scaling_tensor, torch.Tensor) else 1.0
        adapters.append(AdapterState(name=name, state_dict=state, scaling=scaling))
    return convert_lora_backbone_to_static_fusion(backbone, adapters=adapters)


def convert_lora_backbone_to_dynamic_fusion_from_checkpoint_state(
    backbone: nn.Module,
    *,
    model_state_dict: Mapping[str, torch.Tensor],
    adapter_names: Sequence[str],
) -> FusionSummary:
    """Rebuild dynamic-fusion topology before loading a saved fusion checkpoint."""

    adapters: list[AdapterState] = []
    sanitized_names = tuple(_sanitize_adapter_name(name) for name in adapter_names)
    module_names = [name for name, module in backbone.named_modules() if isinstance(module, LoRALinear)]
    if not module_names:
        raise ValueError("No LoRALinear modules found; call apply_lora before fusion reconstruction.")
    first_module = module_names[0]
    first_prefix = _resolve_state_prefix(model_state_dict, first_module)
    gate_key = f"{first_prefix}.gate.0.weight"
    gate_hidden_dim = int(model_state_dict[gate_key].shape[0]) if gate_key in model_state_dict else 64

    for idx, name in enumerate(sanitized_names):
        state: dict[str, torch.Tensor] = {}
        for module_name in module_names:
            state_prefix = _resolve_state_prefix(model_state_dict, module_name)
            key_a = f"{state_prefix}.population_lora_a.{idx}"
            key_b = f"{state_prefix}.population_lora_b.{idx}"
            if key_a not in model_state_dict or key_b not in model_state_dict:
                raise KeyError(f"Fusion checkpoint is missing population tensors for adapter {name!r}.")
            state[f"backbone.{module_name}.lora_a"] = model_state_dict[key_a]
            state[f"backbone.{module_name}.lora_b"] = model_state_dict[key_b]
        scaling_key = f"{first_prefix}.population_scalings"
        scaling_tensor = model_state_dict.get(scaling_key)
        scaling = float(scaling_tensor[idx].item()) if isinstance(scaling_tensor, torch.Tensor) else 1.0
        adapters.append(AdapterState(name=name, state_dict=state, scaling=scaling))
    return convert_lora_backbone_to_dynamic_fusion(
        backbone,
        adapters=adapters,
        gate_hidden_dim=gate_hidden_dim,
    )


def _resolve_state_prefix(model_state_dict: Mapping[str, torch.Tensor], module_name: str) -> str:
    for prefix in (f"backbone.{module_name}", module_name):
        if f"{prefix}.population_lora_a.0" in model_state_dict:
            return prefix
    return f"backbone.{module_name}"


def _is_fusion_gate_parameter(name: str) -> bool:
    return name.endswith("adapter_logits") or ".gate." in name


def freeze_backbone_for_static_fusion(model: nn.Module) -> None:
    """Freeze all backbone weights except fusion gate parameters."""

    backbone = model.backbone
    for _name, parameter in backbone.named_parameters():
        parameter.requires_grad = False
    for name, parameter in backbone.named_parameters():
        if _is_fusion_gate_parameter(name):
            parameter.requires_grad = True
    head = getattr(model, "head", None)
    if isinstance(head, nn.Module):
        for parameter in head.parameters():
            parameter.requires_grad = True


def collect_fusion_gate_diagnostics(model: nn.Module) -> dict[str, torch.Tensor | tuple[str, ...]] | None:
    """Return batch-level gate summaries from the latest dynamic-fusion forward pass."""

    gate_weights: list[torch.Tensor] = []
    gate_entropies: list[torch.Tensor] = []
    adapter_names: tuple[str, ...] | None = None
    for module in model.modules():
        if not isinstance(module, DynamicFusionLoRALinear):
            continue
        if module.last_gate_weights is None or module.last_gate_entropy is None:
            continue
        if adapter_names is None:
            adapter_names = module.adapter_names
        elif adapter_names != module.adapter_names:
            raise ValueError("Dynamic fusion modules expose inconsistent adapter names.")
        gate_weights.append(module.last_gate_weights.float())
        gate_entropies.append(module.last_gate_entropy.float())
    if not gate_weights or adapter_names is None:
        return None
    return {
        "adapter_names": adapter_names,
        "gate_alpha": torch.stack(gate_weights, dim=0).mean(dim=0),
        "gate_entropy": torch.stack(gate_entropies, dim=0).mean(dim=0),
    }


def load_adapter_state(path: Path, *, map_location: torch.device | str = "cpu") -> AdapterState:
    """Load a frequency adapter checkpoint from ``best_adapter.pt`` or ``model.tar.gz``."""

    checkpoint_path = resolve_adapter_checkpoint(path)
    payload = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Adapter checkpoint {checkpoint_path} must contain a dict.")
    state = payload.get("trainable_state_dict")
    if not isinstance(state, dict):
        raise ValueError(f"Adapter checkpoint {checkpoint_path} is missing trainable_state_dict.")
    lora = payload.get("lora")
    rank = None
    alpha = None
    if isinstance(lora, dict):
        rank = lora.get("rank")
        alpha = lora.get("alpha")
    first_lora_a = next((tensor for key, tensor in state.items() if key.endswith(".lora_a")), None)
    inferred_rank = int(first_lora_a.shape[0]) if isinstance(first_lora_a, torch.Tensor) else 1
    scaling = float(alpha) / float(rank or inferred_rank)
    return AdapterState(name=checkpoint_path.parent.name, state_dict=state, scaling=scaling)


def resolve_adapter_checkpoint(path: Path) -> Path:
    """Resolve ``best_adapter.pt`` from a file, directory, or SageMaker ``model.tar.gz`` root."""

    path = path.expanduser()
    if path.is_file():
        if path.name == "model.tar.gz":
            return _extract_and_find(path.parent, archive=path, candidates=("best_adapter.pt",))
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Adapter path does not exist: {path}")
    direct = path / "best_adapter.pt"
    if direct.is_file():
        return direct
    model_tar = path / "model.tar.gz"
    if model_tar.is_file():
        return _extract_and_find(path, archive=model_tar, candidates=("best_adapter.pt",))
    matches = sorted(path.rglob("best_adapter.pt"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not resolve best_adapter.pt under {path}")


def resolve_finetuned_model_checkpoint(path: Path) -> Path:
    """Resolve ``best_model.pt`` from a file, directory, or SageMaker ``model.tar.gz`` root."""

    path = path.expanduser()
    if path.is_file():
        if path.name == "model.tar.gz":
            return _extract_and_find(path.parent, archive=path, candidates=("best_model.pt",))
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Fine-tuned model path does not exist: {path}")
    direct = path / "best_model.pt"
    if direct.is_file():
        return direct
    model_tar = path / "model.tar.gz"
    if model_tar.is_file():
        return _extract_and_find(path, archive=model_tar, candidates=("best_model.pt",))
    matches = sorted(path.rglob("best_model.pt"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not resolve best_model.pt under {path}")


def _extract_and_find(root: Path, *, archive: Path, candidates: Sequence[str]) -> Path:
    extract_dir = root / "_fusion_extracted"
    marker = extract_dir / ".complete"
    if not marker.exists():
        extract_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(extract_dir)
        marker.write_text("ok\n", encoding="utf-8")
    for candidate in candidates:
        direct = extract_dir / candidate
        if direct.is_file():
            return direct
        matches = sorted(extract_dir.rglob(candidate))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Could not find any of {list(candidates)} inside {archive}.")


def load_matching_finetuned_state(
    model: nn.Module,
    path: Path,
    *,
    map_location: torch.device | str = "cpu",
) -> dict[str, int]:
    """Load matching tensors from a prior ClinVar checkpoint, skipping incompatible heads."""

    checkpoint_path = resolve_finetuned_model_checkpoint(path)
    payload = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Fine-tuned checkpoint {checkpoint_path} must contain a dict.")
    state = payload.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError(f"Fine-tuned checkpoint {checkpoint_path} is missing model_state_dict.")

    current = model.state_dict()
    compatible = {
        key: tensor
        for key, tensor in state.items()
        if key in current and tuple(current[key].shape) == tuple(tensor.shape)
    }
    incompatible = len(state) - len(compatible)
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    log.info(
        "Loaded %d compatible tensors from %s; skipped=%d missing_after_partial=%d unexpected=%d.",
        len(compatible),
        checkpoint_path,
        incompatible,
        len(missing),
        len(unexpected),
    )
    return {
        "loaded_tensors": len(compatible),
        "skipped_tensors": incompatible,
        "missing_after_partial": len(missing),
        "unexpected_after_partial": len(unexpected),
    }
