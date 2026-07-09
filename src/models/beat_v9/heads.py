from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.constants import DNA_VOCAB, SNV_BASES, VOCAB_SIZE

ACGT_TOKEN_IDS = (DNA_VOCAB["A"], DNA_VOCAB["C"], DNA_VOCAB["G"], DNA_VOCAB["T"])


def token_id_to_acgt_index(input_ids: torch.Tensor) -> torch.Tensor:
    """Map token ids A/C/G/T to 0/1/2/3 and all other ids to -100."""

    indices = torch.full_like(input_ids, -100)
    for acgt_index, token_id in enumerate(ACGT_TOKEN_IDS):
        indices = torch.where(input_ids == token_id, torch.full_like(indices, acgt_index), indices)
    return indices


def build_counterfactual_snv_mask(
    input_ids: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    ref_index = token_id_to_acgt_index(input_ids)
    alt_index = torch.arange(len(SNV_BASES), device=input_ids.device).view(*([1] * input_ids.ndim), len(SNV_BASES))
    mask = (ref_index.unsqueeze(-1) >= 0) & (alt_index != ref_index.unsqueeze(-1))
    if valid_mask is not None:
        mask = mask & valid_mask.to(dtype=torch.bool).unsqueeze(-1)
    return mask


class ClassificationHead(nn.Module):
    def __init__(self, d_model: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RegressionHead(nn.Module):
    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.net = ClassificationHead(d_model, 1, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class ConvStemBranch(nn.Module):
    def __init__(self, d_model: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.depthwise = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            groups=d_model,
        )
        self.pointwise = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_t: torch.Tensor) -> torch.Tensor:
        seq_len = x_t.shape[-1]
        y = self.depthwise(x_t)
        if y.shape[-1] > seq_len:
            y = y[..., :seq_len]
        elif y.shape[-1] < seq_len:
            y = F.pad(y, (0, seq_len - y.shape[-1]))
        y = F.gelu(y)
        return self.dropout(self.pointwise(y))


class MultiKernelConvStem(nn.Module):
    def __init__(self, d_model: int, kernels: Sequence[int] = (3, 6, 11, 21), dropout: float = 0.05) -> None:
        super().__init__()
        if not kernels:
            raise ValueError("MultiKernelConvStem requires at least one kernel.")
        self.branches = nn.ModuleList([ConvStemBranch(d_model, int(kernel), dropout) for kernel in kernels])
        branch_dim = d_model * len(kernels)
        self.gate = nn.Sequential(
            nn.Linear(branch_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )
        self.out = nn.Sequential(
            nn.LayerNorm(branch_dim),
            nn.Linear(branch_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_t = x.transpose(1, 2)
        ys = [branch(x_t).transpose(1, 2) for branch in self.branches]
        y = torch.cat(ys, dim=-1)
        y_proj = self.out(y)
        gate = self.gate(y)
        return x + gate * y_proj


class HiddenTapFusion(nn.Module):
    def __init__(self, d_model: int, num_taps: int, dropout: float = 0.05) -> None:
        super().__init__()
        if num_taps <= 0:
            raise ValueError("HiddenTapFusion requires at least one tap.")
        self.weights = nn.Parameter(torch.zeros(num_taps))
        self.proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

    def forward(self, taps: Sequence[torch.Tensor]) -> torch.Tensor:
        if not taps:
            raise ValueError("HiddenTapFusion received no hidden taps.")
        if len(taps) > self.weights.numel():
            raise ValueError(f"Expected at most {self.weights.numel()} taps, got {len(taps)}.")
        weights = torch.softmax(self.weights[: len(taps)], dim=0)
        fused = torch.zeros_like(taps[0])
        for weight, hidden in zip(weights, taps, strict=False):
            fused = fused + weight * hidden
        return self.proj(fused)


class MultiScaleContext(nn.Module):
    def __init__(self, d_model: int, radii: Sequence[int] = (3, 16, 64, 256, 1024), dropout: float = 0.05) -> None:
        super().__init__()
        self.radii = tuple(int(radius) for radius in radii)
        if any(radius < 0 for radius in self.radii):
            raise ValueError(f"Context radii must be non-negative, got {self.radii}.")
        in_dim = d_model * (len(self.radii) + 1)
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

    @staticmethod
    def masked_avg_pool_1d(x: torch.Tensor, attention_mask: torch.Tensor | None, radius: int) -> torch.Tensor:
        if radius == 0:
            return x
        kernel_size = 2 * int(radius) + 1
        x_t = x.transpose(1, 2)
        if attention_mask is None:
            return F.avg_pool1d(x_t, kernel_size=kernel_size, stride=1, padding=radius).transpose(1, 2)

        mask = attention_mask.to(device=x.device, dtype=x.dtype).unsqueeze(1)
        numerator = F.avg_pool1d(x_t * mask, kernel_size=kernel_size, stride=1, padding=radius) * kernel_size
        denominator = F.avg_pool1d(mask, kernel_size=kernel_size, stride=1, padding=radius) * kernel_size
        pooled = numerator / denominator.clamp_min(1.0)
        return pooled.transpose(1, 2)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        pools = [x]
        for radius in self.radii:
            pools.append(self.masked_avg_pool_1d(x, attention_mask, radius))
        return x + self.proj(torch.cat(pools, dim=-1))


class CounterfactualSNVHead(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_repr: int = 128,
        num_effect_classes: int = 12,
        num_alt_bases: int = len(SNV_BASES),
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if num_alt_bases != len(SNV_BASES):
            raise ValueError(f"BEAT-v9 expects exactly {len(SNV_BASES)} SNV alt bases, got {num_alt_bases}.")
        self.alt_base_emb = nn.Embedding(num_alt_bases, d_model)
        self.ref_base_proj = nn.Linear(d_model, d_model)
        self.trunk = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_repr),
            nn.GELU(),
            nn.LayerNorm(d_repr),
        )
        self.effect = nn.Linear(d_repr, num_effect_classes)
        self.severity = nn.Linear(d_repr, 1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        *,
        return_repr: bool = True,
    ) -> dict[str, torch.Tensor]:
        batch_size, seq_len, d_model = hidden_states.shape
        alt_ids = torch.arange(len(SNV_BASES), device=hidden_states.device)
        alt_emb = self.alt_base_emb(alt_ids).view(1, 1, len(SNV_BASES), d_model)
        alt_emb = alt_emb.expand(batch_size, seq_len, len(SNV_BASES), d_model)

        ref_index = token_id_to_acgt_index(input_ids)
        safe_ref_index = ref_index.clamp_min(0)
        ref_emb = self.alt_base_emb(safe_ref_index)
        ref_emb = torch.where((ref_index >= 0).unsqueeze(-1), ref_emb, torch.zeros_like(ref_emb))
        ref_emb = self.ref_base_proj(ref_emb).unsqueeze(2).expand_as(alt_emb)

        h = hidden_states.unsqueeze(2).expand_as(alt_emb)
        delta_base = alt_emb - ref_emb
        z = torch.cat([h, alt_emb, delta_base], dim=-1)
        repr_ = self.trunk(z)

        outputs = {
            "counterfactual_effect_logits": self.effect(repr_),
            "counterfactual_severity": torch.sigmoid(self.severity(repr_).squeeze(-1)),
        }
        if return_repr:
            outputs["counterfactual_snv_repr"] = repr_
        return outputs


class SequenceEmbeddingHead(nn.Module):
    def __init__(self, d_model: int, sequence_embedding_dim: int | None = None) -> None:
        super().__init__()
        out_dim = d_model if sequence_embedding_dim is None else int(sequence_embedding_dim)
        self.proj = nn.Linear(d_model, out_dim)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        if attention_mask is None:
            pooled = hidden_states.mean(dim=1)
        else:
            mask = attention_mask.to(dtype=hidden_states.dtype).unsqueeze(-1)
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.proj(pooled)


class MLMHead(nn.Module):
    def __init__(self, token_emb: nn.Embedding, d_model: int, vocab_size: int = VOCAB_SIZE) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size, bias=False)
        if token_emb.num_embeddings == vocab_size and token_emb.embedding_dim == d_model:
            self.proj.weight = token_emb.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)
