from __future__ import annotations

import torch
import torch.nn.functional as F


def poisson_loss(y_true: torch.Tensor, y_pred: torch.Tensor, *, epsilon: float = 1e-7) -> torch.Tensor:
    return y_pred - y_true * torch.log(y_pred + epsilon)


def safe_for_grad_log_torch(x: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.where(x > 0.0, x, torch.ones_like(x)))


def poisson_multinomial_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    shape_loss_coefficient: float = 5.0,
    epsilon: float = 1e-7,
) -> torch.Tensor:
    batch_size, seq_length, num_tracks = logits.shape
    sum_pred = logits.sum(dim=1)
    sum_true = targets.sum(dim=1)
    scale_loss = poisson_loss(sum_true, sum_pred, epsilon=epsilon) / (seq_length + epsilon)
    scale_loss = scale_loss.mean()

    predicted_counts = logits + epsilon
    targets_with_epsilon = targets + epsilon
    denom = predicted_counts.sum(dim=1, keepdim=True) + epsilon
    p_pred = predicted_counts / denom
    shape_loss = -(targets_with_epsilon * safe_for_grad_log_torch(p_pred))
    shape_loss = shape_loss.sum() / (batch_size * seq_length * num_tracks + epsilon)
    return shape_loss + scale_loss / shape_loss_coefficient


def focal_loss(logits: torch.Tensor, targets: torch.Tensor, *, gamma: float = 2.0) -> torch.Tensor:
    ce_loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    )
    pt = torch.exp(-ce_loss)
    return ((1.0 - pt) ** gamma * ce_loss).mean()
