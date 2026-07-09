from __future__ import annotations

import importlib
import inspect
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

_flash_attn_func: Any | None = None
try:
    _flash_attn_module = importlib.import_module("flash_attn")
except ImportError:
    _flash_attn_module = None
else:
    _flash_attn_func = getattr(_flash_attn_module, "flash_attn_func", None)


def _masked_fill_value(dtype: torch.dtype) -> float:
    return torch.finfo(dtype if dtype.is_floating_point else torch.float32).min


def _attn_mask_has_padding(attn_mask: torch.Tensor | None) -> bool:
    if attn_mask is None:
        return False
    if attn_mask.dtype == torch.bool:
        return not bool(torch.all(attn_mask).item())
    return not bool(torch.all(attn_mask > 0).item())


class LocalWindowAttention(nn.Module):
    """Masked local self-attention with a fixed symmetric token window."""

    def __init__(self, d_model: int, n_heads: int, window: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}.")
        if window <= 0:
            raise ValueError(f"window must be positive, got {window}.")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window = int(window)
        self.radius = max(0, self.window // 2)
        self.dropout = float(dropout)
        self.query_chunk_size = max(32, min(self.window, 256))

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.out_dropout = nn.Dropout(dropout)

        self._flash_attn_supports_window = False
        if _flash_attn_func is not None:
            with torch.no_grad():
                try:
                    signature = inspect.signature(_flash_attn_func)
                except (TypeError, ValueError):
                    signature = None
            self._flash_attn_supports_window = signature is not None and "window_size" in signature.parameters

    def _reshape_qkv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).reshape(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        return q, k, v

    def _flash_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if (
            _flash_attn_func is None
            or not self._flash_attn_supports_window
            or not q.is_cuda
            or _attn_mask_has_padding(attn_mask)
        ):
            return None

        q_flash = q.transpose(1, 2).contiguous()
        k_flash = k.transpose(1, 2).contiguous()
        v_flash = v.transpose(1, 2).contiguous()
        try:
            output = _flash_attn_func(
                q_flash,
                k_flash,
                v_flash,
                dropout_p=self.dropout if self.training else 0.0,
                causal=False,
                window_size=(self.radius, self.radius),
            )
            if not torch.is_tensor(output) or output.ndim != 4:
                self._flash_attn_supports_window = False
                return None
            return output.transpose(1, 2).contiguous()
        except Exception:
            self._flash_attn_supports_window = False
            return None

    def _chunked_sdpa(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size, _n_heads, seq_len, _head_dim = q.shape
        device = q.device
        dtype = q.dtype
        chunk_outputs: list[torch.Tensor] = []

        key_padding_mask = attn_mask.to(dtype=torch.bool) if attn_mask is not None else None
        query_padding_mask = key_padding_mask
        position_ids = torch.arange(seq_len, device=device)
        neg_inf = _masked_fill_value(dtype)

        for q_start in range(0, seq_len, self.query_chunk_size):
            q_end = min(seq_len, q_start + self.query_chunk_size)
            k_start = max(0, q_start - self.radius)
            k_end = min(seq_len, q_end + self.radius)

            q_chunk = q[:, :, q_start:q_end, :]
            k_chunk = k[:, :, k_start:k_end, :]
            v_chunk = v[:, :, k_start:k_end, :]

            q_positions = position_ids[q_start:q_end]
            k_positions = position_ids[k_start:k_end]
            local_mask = (q_positions[:, None] - k_positions[None, :]).abs() <= self.radius

            attn_bias = torch.zeros(
                (batch_size, 1, q_end - q_start, k_end - k_start),
                dtype=dtype,
                device=device,
            )
            attn_bias.masked_fill_(~local_mask.unsqueeze(0).unsqueeze(0), neg_inf)
            if key_padding_mask is not None:
                valid_keys = key_padding_mask[:, None, None, k_start:k_end]
                attn_bias.masked_fill_(~valid_keys, neg_inf)

            chunk = F.scaled_dot_product_attention(
                q_chunk,
                k_chunk,
                v_chunk,
                attn_mask=attn_bias,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
            )
            if query_padding_mask is not None:
                valid_queries = query_padding_mask[:, None, q_start:q_end, None].to(dtype=chunk.dtype)
                chunk = chunk * valid_queries
            chunk_outputs.append(chunk)

        return torch.cat(chunk_outputs, dim=2)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None) -> torch.Tensor:
        q, k, v = self._reshape_qkv(x)

        attn_output = self._flash_forward(q, k, v, attn_mask)
        if attn_output is None:
            attn_output = self._chunked_sdpa(q, k, v, attn_mask)

        merged = attn_output.transpose(1, 2).reshape(x.shape[0], x.shape[1], self.d_model)
        projected = self.out_proj(merged)
        projected = self.out_dropout(projected)
        if attn_mask is not None:
            projected = projected * attn_mask.unsqueeze(-1).to(dtype=projected.dtype)
        return projected
