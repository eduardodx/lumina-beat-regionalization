from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class _IntervalIndex:
    starts: np.ndarray
    ends: np.ndarray

    def overlapping_bounds(self, window_start: int, window_end: int) -> tuple[int, int]:
        left = int(np.searchsorted(self.ends, window_start, side="right"))
        right = int(np.searchsorted(self.starts, window_end, side="left"))
        return left, right


@dataclass(frozen=True)
class ClinVarBlocklist:
    intervals_by_chrom: dict[str, _IntervalIndex]

    @classmethod
    def empty(cls) -> ClinVarBlocklist:
        return cls(intervals_by_chrom={})

    @classmethod
    def from_bed(cls, bed_path: str | Path | None) -> ClinVarBlocklist:
        if bed_path is None:
            return cls.empty()

        path = Path(bed_path)
        if not path.is_file():
            return cls.empty()

        starts_by_chrom: dict[str, list[int]] = {}
        ends_by_chrom: dict[str, list[int]] = {}
        with path.open(encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) < 3:
                    continue
                chrom = parts[0]
                try:
                    start = int(parts[1])
                    end = int(parts[2])
                except ValueError:
                    continue
                if end <= start:
                    continue
                starts_by_chrom.setdefault(chrom, []).append(start)
                ends_by_chrom.setdefault(chrom, []).append(end)

        intervals_by_chrom: dict[str, _IntervalIndex] = {}
        for chrom, starts in starts_by_chrom.items():
            ends = ends_by_chrom[chrom]
            ordering = np.argsort(np.asarray(starts, dtype=np.int64), kind="stable")
            ordered_starts = np.asarray(starts, dtype=np.int64)[ordering]
            ordered_ends = np.asarray(ends, dtype=np.int64)[ordering]
            intervals_by_chrom[chrom] = _IntervalIndex(starts=ordered_starts, ends=ordered_ends)
        return cls(intervals_by_chrom=intervals_by_chrom)

    def overlaps(self, chrom: str, start: int, end: int) -> bool:
        interval_index = self.intervals_by_chrom.get(chrom)
        if interval_index is None:
            return False
        left, right = interval_index.overlapping_bounds(start, end)
        return right > left
