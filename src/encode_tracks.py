from __future__ import annotations

import atexit
from collections.abc import Iterator
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Literal

import numpy as np
import pyBigWig

TrackTransform = Literal["log1p", "asinh"]
TrackNormalize = Literal["per_chromosome_zscore", "global_zscore"]

_HANDLE_CACHE: dict[str, pyBigWig.pyBigWig] = {}
_STATS_CACHE: dict[tuple[str, str], tuple[float, float]] = {}
_CACHE_LOCK = Lock()
_STATS_CHUNK_SIZE = 1_000_000


@dataclass(frozen=True)
class EncodeTrackSpec:
    name: str
    bw_path: Path
    transform: TrackTransform = "asinh"
    normalize: TrackNormalize = "per_chromosome_zscore"


def _apply_transform(values: np.ndarray, transform: TrackTransform) -> np.ndarray:
    if transform == "log1p":
        return np.log1p(np.maximum(values, 0.0)).astype(np.float32)
    if transform == "asinh":
        return np.arcsinh(values).astype(np.float32)
    raise ValueError(f"Unsupported transform {transform!r}.")


def _cache_key(spec: EncodeTrackSpec) -> str:
    return str(spec.bw_path.expanduser().resolve())


def _stats_key(spec: EncodeTrackSpec, chrom: str | None) -> tuple[str, str]:
    scope = chrom if chrom is not None else "__global__"
    return (_cache_key(spec), scope)


def _open_track_handle(spec: EncodeTrackSpec) -> pyBigWig.pyBigWig:
    path_key = _cache_key(spec)
    with _CACHE_LOCK:
        handle = _HANDLE_CACHE.get(path_key)
        if handle is None:
            handle = pyBigWig.open(path_key)
            _HANDLE_CACHE[path_key] = handle
        return handle


def _close_all_track_handles() -> None:
    with _CACHE_LOCK:
        handles = list(_HANDLE_CACHE.values())
        _HANDLE_CACHE.clear()
        _STATS_CACHE.clear()
    for handle in handles:
        with suppress(Exception):
            handle.close()


atexit.register(_close_all_track_handles)


def reset_encode_track_caches() -> None:
    """Test/helper hook to clear cached BigWig handles and normalization stats."""
    _close_all_track_handles()


def _iter_chrom_chunks(handle: pyBigWig.pyBigWig, chrom: str) -> Iterator[tuple[int, int]]:
    chrom_length = handle.chroms(chrom)
    if chrom_length is None:
        return
    for start in range(0, chrom_length, _STATS_CHUNK_SIZE):
        yield start, min(start + _STATS_CHUNK_SIZE, chrom_length)


def _transformed_moments_for_chrom(
    handle: pyBigWig.pyBigWig,
    spec: EncodeTrackSpec,
    chrom: str,
) -> tuple[float, float, int]:
    total = 0.0
    total_sq = 0.0
    count = 0

    for start, end in _iter_chrom_chunks(handle, chrom):
        values = handle.values(chrom, start, end, numpy=True)
        if values is None:
            continue
        array = np.asarray(values, dtype=np.float32)
        finite_mask = np.isfinite(array)
        if not finite_mask.any():
            continue
        transformed = _apply_transform(array[finite_mask], spec.transform).astype(np.float64)
        total += float(transformed.sum(dtype=np.float64))
        total_sq += float(np.square(transformed, dtype=np.float64).sum(dtype=np.float64))
        count += int(transformed.size)

    if count == 0:
        return 0.0, 1.0, 0
    mean = total / count
    variance = max(total_sq / count - mean * mean, 1e-12)
    return float(mean), float(np.sqrt(variance)), count


def _resolve_stats(handle: pyBigWig.pyBigWig, spec: EncodeTrackSpec, chrom: str) -> tuple[float, float]:
    stats_scope = chrom if spec.normalize == "per_chromosome_zscore" else None
    cache_key = _stats_key(spec, stats_scope)
    with _CACHE_LOCK:
        cached = _STATS_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if stats_scope is not None:
        mean, std, _count = _transformed_moments_for_chrom(handle, spec, stats_scope)
        stats = (mean, std)
    else:
        chroms = sorted(handle.chroms().keys())
        if not chroms:
            stats = (0.0, 1.0)
        else:
            total_sum = 0.0
            total_sq_sum = 0.0
            total_count = 0
            for chrom_name in chroms:
                mean, std, count = _transformed_moments_for_chrom(handle, spec, chrom_name)
                if count == 0:
                    continue
                total_sum += mean * count
                total_sq_sum += (std * std + mean * mean) * count
                total_count += count
            if total_count == 0:
                stats = (0.0, 1.0)
            else:
                global_mean = total_sum / total_count
                global_variance = max(total_sq_sum / total_count - global_mean * global_mean, 1e-12)
                stats = (float(global_mean), float(np.sqrt(global_variance)))

    with _CACHE_LOCK:
        _STATS_CACHE[cache_key] = stats
    return stats


def load_track_values(spec: EncodeTrackSpec, chrom: str, start: int, end: int) -> np.ndarray:
    handle = _open_track_handle(spec)
    values = handle.values(chrom, start, end, numpy=True)
    if values is None:
        return np.full(end - start, np.nan, dtype=np.float32)

    array = np.asarray(values, dtype=np.float32)
    missing = ~np.isfinite(array)
    normalized = np.full(array.shape, np.nan, dtype=np.float32)
    if missing.all():
        return normalized

    transformed = _apply_transform(array[~missing], spec.transform)
    mean, std = _resolve_stats(handle, spec, chrom)
    normalized[~missing] = (transformed - mean) / max(std, 1e-6)
    return normalized.astype(np.float32)
