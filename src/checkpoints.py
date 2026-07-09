from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from src.precision import PrecisionPolicy

_OPTIONAL_STATE_DICT_PREFIXES = ("rtd_head.",)


def torch_load_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    try:
        checkpoint = torch.load(str(path), map_location=map_location, weights_only=False)
    except TypeError:
        checkpoint = torch.load(str(path), map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint {path!r} must contain a dict, got {type(checkpoint).__name__}.")
    return checkpoint


def checkpoint_config(checkpoint: Mapping[str, Any], *, path: str | Path | None = None) -> dict[str, Any]:
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        target = f"Checkpoint {path!r}" if path is not None else "Checkpoint"
        raise RuntimeError(f"{target} is missing a usable 'config' dictionary.")
    return dict(config)


def checkpoint_model_key(config: Mapping[str, Any], *, default_model_key: str = "bimamba") -> str:
    raw_model_key = config.get("model")
    if isinstance(raw_model_key, str) and raw_model_key.strip():
        return _normalize_model_key(raw_model_key)
    return _normalize_model_key(default_model_key)


def checkpoint_model_overrides(config: Mapping[str, Any]) -> dict[str, Any]:
    raw_model_config = config.get("model_config")
    if isinstance(raw_model_config, dict):
        return dict(raw_model_config)

    overrides: dict[str, Any] = {}
    for key in (
        "d_model",
        "n_layers",
        "d_state",
        "d_conv",
        "expand",
        "dropout",
        "headdim",
        "ngroups",
        "rope_fraction",
        "chunk_size",
        "is_mimo",
        "mimo_rank",
        "is_outproj_norm",
        "num_region_classes",
        "num_aa_classes",
        "use_gated_fusion",
    ):
        value = config.get(key)
        if value is not None:
            overrides[key] = value
    return overrides


def resolve_lumina_checkpoint_spec(
    requested_model_key: str,
    checkpoint_path: str | Path,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    checkpoint = torch_load_checkpoint(checkpoint_path)
    config = checkpoint_config(checkpoint, path=checkpoint_path)

    normalized_ckpt = checkpoint_model_key(config, default_model_key=requested_model_key)
    normalized_requested = _normalize_model_key(requested_model_key)
    if normalized_ckpt != normalized_requested:
        raise ValueError(
            f"Checkpoint model mismatch: requested {normalized_requested!r}, "
            f"but checkpoint specifies {normalized_ckpt!r}."
        )
    return normalized_ckpt, checkpoint_model_overrides(config), checkpoint


def normalize_lumina_model_config_for_precision(
    model_key: str,
    model_config: Mapping[str, Any] | None,
    precision: PrecisionPolicy | None,
    *,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    from src.models import resolve_model_config_dict

    resolved = resolve_model_config_dict(
        model_key,
        dict(model_config) if model_config is not None else None,
        source="checkpoint['config']['model_config']",
    )

    if not bool(resolved.get("is_mimo", False)):
        return resolved

    mimo_rank = max(1, int(resolved.get("mimo_rank", 1)))
    current_chunk_size = max(1, int(resolved.get("chunk_size", 64)))
    safe_chunk_size = current_chunk_size
    if precision is not None and precision.resolved == "bf16":
        safe_chunk_size = min(safe_chunk_size, max(1, 64 // mimo_rank))

    runtime_chunk_cap = _runtime_chunk_size_cap(device)
    if runtime_chunk_cap is not None:
        safe_chunk_size = min(safe_chunk_size, runtime_chunk_cap)

    resolved["chunk_size"] = safe_chunk_size
    return resolved


def load_lumina_backbone_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    requested_model_key: str = "bimamba",
    precision: PrecisionPolicy | None = None,
    device: torch.device | str | None = None,
) -> tuple[nn.Module, dict[str, Any], dict[str, Any]]:
    from src.models import build_registered_model
    from src.tilelang_compat import ensure_tilelang_triton_cuda_include

    model_key, model_config, checkpoint = resolve_lumina_checkpoint_spec(requested_model_key, checkpoint_path)
    config = checkpoint_config(checkpoint, path=checkpoint_path)
    resolved_model_config = normalize_lumina_model_config_for_precision(
        model_key,
        model_config,
        precision,
        device=device,
    )
    if bool(resolved_model_config.get("is_mimo", False)):
        ensure_tilelang_triton_cuda_include()
    model = build_registered_model(model_key, resolved_model_config)
    state_dict = checkpoint.get("model")
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"Checkpoint {checkpoint_path!r} is missing a usable 'model' state_dict.")

    incompatible = model.load_state_dict(state_dict, strict=False)
    missing_keys = [key for key in incompatible.missing_keys if not key.startswith(_OPTIONAL_STATE_DICT_PREFIXES)]
    unexpected_keys = [key for key in incompatible.unexpected_keys if not key.startswith(_OPTIONAL_STATE_DICT_PREFIXES)]
    if missing_keys or unexpected_keys:
        problems: list[str] = []
        if missing_keys:
            problems.append(f"missing keys: {', '.join(missing_keys)}")
        if unexpected_keys:
            problems.append(f"unexpected keys: {', '.join(unexpected_keys)}")
        raise RuntimeError(
            f"Checkpoint {checkpoint_path!r} is incompatible with the loaded model ({'; '.join(problems)})."
        )

    if device is not None:
        model.to(device)
    return model, resolved_model_config, config


def _normalize_model_key(model_key: str) -> str:
    if not isinstance(model_key, str):
        raise TypeError(f"Model key must be a string, got {type(model_key).__name__}.")
    normalized = model_key.strip().lower()
    if not normalized:
        raise ValueError("Model key must not be empty.")
    return normalized


def _runtime_chunk_size_cap(device: torch.device | str | None) -> int | None:
    override = os.environ.get("LUMINA_MAMBA_MAX_CHUNK_SIZE")
    if override:
        return max(1, int(override))

    if not torch.cuda.is_available():
        return None

    resolved_device = torch.device(device) if device is not None else torch.device("cuda", torch.cuda.current_device())
    if resolved_device.type != "cuda":
        return None

    device_index = resolved_device.index
    if device_index is None:
        device_index = torch.cuda.current_device()

    capability = torch.cuda.get_device_capability(device_index)
    if capability == (8, 9):
        return 8
    if capability >= (10, 0):
        return 8
    return None
