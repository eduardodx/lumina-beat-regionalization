"""Minimal EMA loss-normalizer extracted from the BioPrime training stack.

The full training losses are intentionally NOT shipped in this inference package.
``DNAFoundationBioPrime`` constructs
``self.loss_ema = LossEMANormalizer(BIOPRIME_LOSS_NAMES, per_loss_gating_enabled=...)``
in its ``__init__``, which registers one ``scale_<name>``, one ``init_<name>`` (bool), and one
``init_scale_<name>`` buffer per loss name. Those buffers are part of every trained checkpoint's
state dict, so reproducing them here verbatim (same names, same order) is what lets a checkpoint
load with ``strict=True``.

``update_and_normalize`` is a training-only method and is never called at inference; it is kept
verbatim for fidelity. This module imports only ``torch`` — none of the training loss machinery.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Canonical BioPrime loss names (order matters only for buffer registration / checkpoint compatibility).
# NOTE: appended names get a fresh EMA buffer; existing checkpoints simply initialize the new
# buffer on first use (LossEMANormalizer registers one buffer per name from this tuple).
BIOPRIME_LOSS_NAMES: tuple[str, ...] = (
    "mlm",
    "conservation",
    "splice",
    "region",
    "counterfactual_variant",
    "population_prior",
    "population_rank",
    "regulatory",
    "hic",
    "missense_severity",
    "conservation_delta",
)

# Default minimum valid-count per loss before its EMA scale updates / its loss counts.
_DEFAULT_MIN_VALID: dict[str, int] = {
    "mlm": 256,
    "conservation": 256,
    "splice": 256,
    "region": 256,
    "counterfactual_variant": 256,
    "population_prior": 256,
    "population_rank": 256,
    "regulatory": 256,
    "hic": 1,
    "missense_severity": 256,
    # Variant-delta is supervised only at the (sparse) synthetic-edit positions, so a batch carries
    # few valid cells; a low floor lets its EMA scale track even on lightly-augmented batches (like rc).
    "conservation_delta": 1,
}


class LossEMANormalizer(nn.Module):
    """Per-loss EMA scale (plan §10): normalized_i = raw_i / stopgrad(ema_i + eps).

    nn.Module with one float buffer + one bool 'initialized' buffer per loss so the
    state is checkpointed/restored automatically. No parameters (inert for DDP).
    """

    def __init__(
        self,
        names: tuple[str, ...] = BIOPRIME_LOSS_NAMES,
        *,
        decay: float = 0.99,
        eps: float = 1e-8,
        min_scale: float = 0.25,
        per_loss_gating_enabled: bool = False,
    ) -> None:
        super().__init__()
        self.names = tuple(names)
        self.decay = float(decay)
        self.eps = float(eps)
        # Floor the per-loss scale used for normalization. Without it, a loss that legitimately
        # collapses toward 0 drives its EMA scale toward 0, and dividing the raw loss by ~0 amplifies
        # that loss's gradient unboundedly (observed: scale 2e-4 → ~4000x amplification → exploding
        # grad-norm → grad-clip discards the real objectives' gradients → no learning). Flooring caps
        # amplification at 1/min_scale and lets a satisfied loss fade to a small contribution instead
        # of being re-inflated to ~1.
        self.min_scale = float(min_scale)
        # Per-loss gating. The min_scale floor still AMPLIFIES any loss whose natural scale is below
        # 1.0: normalized = raw/scale, and a satisfied loss whose scale falls to the 0.25 floor is
        # up-weighted up to ~1/(min_scale)·(init_scale) vs its own faded raw gradient (~4x for an
        # init-scale~1 loss). With gating ON the denominator is additionally floored at each loss's
        # INITIAL scale, so the normalization gain can never exceed its step-0 value: a loss that drops
        # below its initial magnitude FADES proportionally (denom = init_scale ⇒ normalized = raw/init →0)
        # instead of being re-amplified, while a loss that grows is still attenuated (denom = scale).
        # Default OFF ⇒ the denominator branch is byte-identical to the pre-gating behavior. Matters more
        # with many objectives ⇒ more satisfied losses the floor would otherwise mis-amplify.
        self.per_loss_gating_enabled = bool(per_loss_gating_enabled)
        for name in self.names:
            self.register_buffer(f"scale_{name}", torch.ones(()))
            self.register_buffer(f"init_{name}", torch.zeros((), dtype=torch.bool))
            # First-observed scale, captured at init alongside scale_{name}; only READ when gating is on,
            # so adding this buffer is inert for every existing run (and tolerated on load: strict=False).
            self.register_buffer(f"init_scale_{name}", torch.ones(()))

    @torch.no_grad()
    def _update_scale(self, name: str, raw: torch.Tensor) -> None:
        scale = getattr(self, f"scale_{name}")
        init = getattr(self, f"init_{name}")
        init_scale = getattr(self, f"init_scale_{name}")
        value = raw.detach().to(scale.dtype).to(scale.device).clamp_min(0.0)
        first = value.clamp_min(self.eps)
        ema = scale * self.decay + (1.0 - self.decay) * value
        # Branchless equivalent of `if initialized: EMA-update else: capture first-observed scale`.
        # torch.where selects on the pre-update `init` flag, avoiding a per-task host sync (bool(init.item()))
        # on EVERY step -- pure overhead in the launch-bound step. Numerically identical to the branched form.
        scale.copy_(torch.where(init, ema, first))
        init_scale.copy_(torch.where(init, init_scale, first))  # set to `first` only on the init transition
        init.fill_(True)

    def update_and_normalize(
        self,
        raw_losses: dict[str, torch.Tensor],
        valid_counts: dict[str, float],
        *,
        min_valid: dict[str, int] | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
        min_valid = min_valid or _DEFAULT_MIN_VALID
        normalized: dict[str, torch.Tensor] = {}
        scales: dict[str, float] = {}
        scale_tensors: dict[str, torch.Tensor] = {}  # collected for a single batched host sync
        for name, raw in raw_losses.items():
            if name in self.names:
                if float(valid_counts.get(name, 0.0)) >= float(min_valid.get(name, 1)):
                    self._update_scale(name, raw)
                scale = getattr(self, f"scale_{name}")
                denom = scale.detach().to(raw.device).clamp_min(self.min_scale)
                if self.per_loss_gating_enabled:
                    # Floor the denominator at the loss's initial scale so the normalization gain never
                    # exceeds its step-0 value ⇒ a satisfied (shrinking) loss fades instead of amplifying.
                    init_scale = getattr(self, f"init_scale_{name}").detach().to(raw.device)
                    denom = torch.maximum(denom, init_scale)
                denom = denom + self.eps
                normalized[name] = raw / denom
                scale_tensors[name] = scale.detach().reshape(()).float()
            else:
                normalized[name] = raw
                scales[name] = 1.0
        # One D2H copy for all EMA scales instead of a float(scale.item()) per task (the ema_scale_* telemetry).
        if scale_tensors:
            keys = list(scale_tensors)
            for key, val in zip(keys, torch.stack([scale_tensors[k] for k in keys]).tolist(), strict=True):
                scales[key] = val
        return normalized, scales
