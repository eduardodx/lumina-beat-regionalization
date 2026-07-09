from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any, cast

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as activation_checkpoint

try:
    _mamba_ssm = importlib.import_module("mamba_ssm")
except ImportError:
    _Mamba3 = None
else:
    _Mamba3 = getattr(_mamba_ssm, "Mamba3", None)

PRE_HOPPER_MAMBA3_MIMO_MIN_COMPUTE_CAPABILITY = (9, 0)
MAMBA3_MODEL_KEYS = frozenset(
    {
        "beat-v1",
        "beat-v2",
        "beat-v3",
        "beat-v4",
        "beat-v5",
        "beat-v6",
        "beat-v7",
        "bimamba3",
        "bimamba3-rc",
    }
)


def require_mamba3() -> type:
    if _Mamba3 is None:
        raise ImportError(
            "Mamba3 is not available in the installed mamba-ssm package. "
            "Install the latest mamba-ssm from source: "
            "MAMBA_FORCE_BUILD=TRUE pip install git+https://github.com/state-spaces/mamba.git --no-build-isolation"
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

    if (
        cuda_device_capability is not None
        and cuda_device_capability < PRE_HOPPER_MAMBA3_MIMO_MIN_COMPUTE_CAPABILITY
    ):
        normalized["is_mimo"] = False
        normalized["chunk_size"] = max(64, int(normalized.get("chunk_size", 64)))
        major, minor = cuda_device_capability
        notes.append(
            "disabled_mimo_pre_hopper_cuda"
            f"(compute_capability={major}.{minor}, chunk_size={normalized['chunk_size']})"
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


def make_mlp_token_head(d_model: int, out_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(d_model, d_model),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(d_model, out_dim),
    )


def make_linear_token_head(d_model: int, out_dim: int) -> nn.Linear:
    return nn.Linear(d_model, out_dim)


class BeatBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        expand: int = 2,
        headdim: int = 64,
        ngroups: int = 1,
        rope_fraction: float = 1.0,
        chunk_size: int = 16,
        is_mimo: bool = True,
        mimo_rank: int = 4,
        is_outproj_norm: bool = True,
        dropout: float = 0.1,
        use_gated_fusion: bool = True,
        activation_checkpointing: bool = False,
        checkpoint_use_reentrant: bool = True,
    ) -> None:
        super().__init__()
        self.use_gated_fusion = use_gated_fusion
        self._activation_checkpointing = activation_checkpointing
        self._checkpoint_use_reentrant = checkpoint_use_reentrant
        self.norm_fwd = nn.LayerNorm(d_model)
        self.norm_rc = nn.LayerNorm(d_model)

        Mamba3 = require_mamba3()

        mamba3_kwargs = dict(
            d_model=d_model,
            d_state=d_state,
            expand=expand,
            headdim=headdim,
            ngroups=ngroups,
            rope_fraction=rope_fraction,
            chunk_size=chunk_size,
            is_mimo=is_mimo,
            mimo_rank=mimo_rank,
            is_outproj_norm=is_outproj_norm,
        )
        self.fwd_mixer = Mamba3(**mamba3_kwargs)
        self.bwd_mixer = Mamba3(**mamba3_kwargs)

        if use_gated_fusion:
            self.gate_proj = nn.Linear(2 * d_model, d_model, bias=True)
            self.fwd_proj = nn.Linear(d_model, d_model, bias=False)
            self.bwd_proj = nn.Linear(d_model, d_model, bias=False)
        else:
            self.fuse = nn.Linear(2 * d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def _maybe_checkpoint(
        self,
        fn: Callable[[torch.Tensor], torch.Tensor],
        x: torch.Tensor,
    ) -> torch.Tensor:
        if (
            not self._activation_checkpointing
            or not self.training
            or not torch.is_grad_enabled()
            or not x.requires_grad
        ):
            return fn(x)
        return cast(torch.Tensor, activation_checkpoint(fn, x, use_reentrant=self._checkpoint_use_reentrant))

    def forward(self, x: torch.Tensor, x_rc: torch.Tensor) -> torch.Tensor:
        h_fwd = self._maybe_checkpoint(self.fwd_mixer, self.norm_fwd(x))
        h_bwd = torch.flip(self._maybe_checkpoint(self.bwd_mixer, self.norm_rc(x_rc)), dims=[1])

        if self.use_gated_fusion:
            h_cat = torch.cat([h_fwd, h_bwd], dim=-1)
            gate = torch.sigmoid(self.gate_proj(h_cat))
            h_out = gate * self.fwd_proj(h_fwd) + (1.0 - gate) * self.bwd_proj(h_bwd)
        else:
            h_out = self.fuse(torch.cat([h_fwd, h_bwd], dim=-1))

        return x + self.dropout(h_out)
