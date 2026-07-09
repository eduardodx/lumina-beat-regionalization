from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass, field

import torch

from src.constants import NUM_REGION_CLASSES


@dataclass
class RegionLossEMA:
    """EMA of MLM loss keyed by region class."""

    decay: float = 0.99
    _values: torch.Tensor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._values = torch.ones(NUM_REGION_CLASSES, dtype=torch.float32)
        with suppress(RuntimeError):
            self._values.share_memory_()

    def update(self, region_losses: dict[int, float]) -> None:
        for region_index in range(NUM_REGION_CLASSES):
            current = float(self._values[region_index].item())
            observed = float(region_losses.get(region_index, current))
            updated = self.decay * current + (1.0 - self.decay) * observed
            self._values[region_index] = updated

    def snapshot(self) -> torch.Tensor:
        return self._values.clone()

    @property
    def values(self) -> torch.Tensor:
        return self._values

    def load_snapshot(self, snapshot: torch.Tensor) -> None:
        if snapshot.shape != self._values.shape:
            raise ValueError(
                f"snapshot must have shape {tuple(self._values.shape)}, got {tuple(snapshot.shape)}."
            )
        self._values.copy_(snapshot.to(dtype=self._values.dtype))
