from __future__ import annotations

import importlib
from typing import Any

import torch


def _load_mamba3() -> type | None:
    for package_name in ("mamba_ssm", "mamba3"):
        try:
            module = importlib.import_module(package_name)
        except ImportError:
            continue
        implementation = getattr(module, "Mamba3", None)
        if implementation is not None:
            return implementation
    return None


_Mamba3 = _load_mamba3()

PRE_HOPPER_MAMBA3_MIMO_MIN_COMPUTE_CAPABILITY = (9, 0)
MAMBA3_MODEL_KEYS = frozenset({"lumina"})


def require_mamba3() -> type:
    global _Mamba3
    if _Mamba3 is None:
        _Mamba3 = _load_mamba3()
        if _Mamba3 is None:
            raise ImportError(
                "Mamba3 is unavailable. Install mamba-ssm for the production CUDA kernels or the "
                "pure-PyTorch mamba3 package for CPU/MPS development."
            )
    return _Mamba3


def resolve_chunk_size(cfg: Any) -> int:
    """Mamba3 MIMO with bf16 needs chunk_size = 64 / mimo_rank."""
    if cfg.is_mimo and cfg.chunk_size == 64:
        return max(1, 64 // cfg.mimo_rank)
    return cfg.chunk_size


def get_cuda_device_capability(device: torch.device) -> tuple[int, int] | None:
    if device.type != "cuda":
        return None
    device_index = int(device.index) if device.index is not None else int(torch.cuda.current_device())
    major, minor = torch.cuda.get_device_capability(device_index)
    return int(major), int(minor)


def normalize_mamba3_runtime_config(
    model_key: str,
    resolved: dict[str, Any],
    *,
    uses_bf16_compute: bool,
    cuda_device_capability: tuple[int, int] | None,
) -> tuple[dict[str, Any], list[str]]:
    normalized = dict(resolved)
    notes: list[str] = []

    if model_key not in MAMBA3_MODEL_KEYS:
        return normalized, notes

    if (
        cuda_device_capability is not None
        and bool(normalized.get("activation_checkpointing", False))
        and not bool(normalized.get("checkpoint_use_reentrant", True))
    ):
        normalized["activation_checkpointing"] = False
        notes.append("disabled_non_reentrant_activation_checkpointing_for_mamba3_cuda")

    if not bool(normalized.get("is_mimo", False)):
        return normalized, notes

    if cuda_device_capability is not None and cuda_device_capability < PRE_HOPPER_MAMBA3_MIMO_MIN_COMPUTE_CAPABILITY:
        normalized["is_mimo"] = False
        normalized["chunk_size"] = max(64, int(normalized.get("chunk_size", 64)))
        major, minor = cuda_device_capability
        notes.append(
            f"disabled_mimo_pre_hopper_cuda(compute_capability={major}.{minor}, chunk_size={normalized['chunk_size']})"
        )
        return normalized, notes

    if not uses_bf16_compute:
        return normalized, notes

    mimo_rank = max(1, int(normalized.get("mimo_rank", 1)))
    current_chunk_size = max(1, int(normalized.get("chunk_size", 64)))
    safe_chunk_size = max(1, 64 // mimo_rank)
    effective_chunk_size = min(current_chunk_size, safe_chunk_size)
    if effective_chunk_size != current_chunk_size:
        normalized["chunk_size"] = effective_chunk_size
        notes.append(
            "clamped_mimo_chunk_size_for_bf16"
            f"(from={current_chunk_size}, to={effective_chunk_size}, mimo_rank={mimo_rank})"
        )

    return normalized, notes
