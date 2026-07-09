"""Loss functions for ClinVar fine-tuning."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FocalLoss(nn.Module):
    """Binary focal loss with optional positive class weighting.

    Down-weights easy examples so training concentrates on hard,
    ambiguous variants where the backbone signal is weakest.
    """

    def __init__(self, gamma: float = 2.0, pos_weight: Tensor | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        pos_weight = self.pos_weight if isinstance(self.pos_weight, Tensor) else None
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none", pos_weight=pos_weight,
        )
        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1 - targets) * (1 - probs)
        focal_weight = (1 - p_t) ** self.gamma
        return (focal_weight * bce).mean()


def build_loss(
    loss_type: str,
    gamma: float = 2.0,
    pos_weight: Tensor | None = None,
) -> nn.Module:
    """Construct the requested loss function."""
    if loss_type == "focal":
        return FocalLoss(gamma=gamma, pos_weight=pos_weight)
    if loss_type == "bce":
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    raise ValueError(f"Unknown loss type: {loss_type!r}")


def pairwise_ranking_loss(
    logits: Tensor,
    targets: Tensor,
    margin: float = 0.5,
    max_pairs: int = 2048,
) -> Tensor:
    positives = torch.nonzero(targets > 0.5, as_tuple=False).flatten()
    negatives = torch.nonzero(targets <= 0.5, as_tuple=False).flatten()
    zero = (logits * 0.0).sum()
    if positives.numel() == 0 or negatives.numel() == 0:
        return zero

    pair_count = min(int(positives.numel() * negatives.numel()), max_pairs)
    pos_indices = positives[torch.randint(0, positives.numel(), (pair_count,), device=logits.device)]
    neg_indices = negatives[torch.randint(0, negatives.numel(), (pair_count,), device=logits.device)]
    return F.relu(float(margin) - (logits[pos_indices].float() - logits[neg_indices].float())).mean()


def swap_consistency_loss(
    logits: Tensor,
    swap_logits: Tensor,
    targets: Tensor,
    margin: float = 0.5,
) -> Tensor:
    """Directional ref/alt swap margin for allele-sensitive Regime A heads."""
    direction = targets.float().mul(2.0).sub(1.0)
    return F.relu(float(margin) - direction * (logits.float() - swap_logits.float())).mean()
