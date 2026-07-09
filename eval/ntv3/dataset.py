from __future__ import annotations

import bisect
import csv
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyBigWig
import torch
from pyfaidx import Fasta
from torch.utils.data import Dataset

from eval.ntv3.diagnostics import DatasetItemLogger
from eval.ntv3.heads import crop_center
from src.constants import DNA_VOCAB, PAD_ID, UNK_ID

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SplitRegion:
    chrom: str
    start: int
    end: int
    split: str


@dataclass(frozen=True)
class FunctionalTrackInfo:
    dataset_id: str
    path: Path
    assay_type: str
    track_name_clean: str
    mean: float


@dataclass(frozen=True)
class AnnotationElementInfo:
    dataset_id: str
    path: Path


@dataclass(frozen=True)
class SpeciesAssets:
    species_name: str
    fasta_path: Path
    splits_path: Path
    split_regions: tuple[SplitRegion, ...]
    functional_tracks: tuple[FunctionalTrackInfo, ...]
    annotation_elements: tuple[AnnotationElementInfo, ...]


@dataclass(frozen=True)
class FunctionalTargetsScaler:
    track_means: torch.Tensor

    def __call__(self, targets: torch.Tensor) -> torch.Tensor:
        means = self.track_means.to(device=targets.device, dtype=targets.dtype)
        scaled = targets / means
        return torch.where(scaled > 10.0, 2.0 * torch.sqrt(scaled * 10.0) - 10.0, scaled)


_fasta_cache: dict[tuple[int, str], Fasta] = {}
_bigwig_cache: dict[tuple[int, str], pyBigWig.pyBigWig] = {}
_bed_cache: dict[tuple[int, str], BedIntervalIndex] = {}


def discover_species(dataset_root: str | Path) -> list[str]:
    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        return []
    species_names = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / "genome.fasta").is_file() and (child / "splits.bed").is_file():
            species_names.append(child.name)
    return species_names


def prepare_fasta_index(fasta_path: str | Path) -> Path:
    path = Path(fasta_path).expanduser().resolve()
    Fasta(str(path), as_raw=True, sequence_always_upper=True)
    return path.with_suffix(path.suffix + ".fai") if path.suffix else Path(f"{path}.fai")


def _chrom_aliases(chrom: str) -> tuple[str, ...]:
    aliases = [chrom]
    if chrom.startswith("chr"):
        aliases.append(chrom[3:])
    else:
        aliases.append(f"chr{chrom}")
    return tuple(dict.fromkeys(alias for alias in aliases if alias))


def _resolve_chrom_name(chrom: str, available_names: set[str]) -> str:
    for alias in _chrom_aliases(chrom):
        if alias in available_names:
            return alias
    return chrom


def _normalize_metadata_value(value: str | None) -> str:
    if value is None:
        return ""
    normalized = value.strip().lower()
    for suffix in (".bigwig", ".bw", ".bed"):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
    return normalized


def _metadata_first(row: dict[str, str], *keys: str) -> str | None:
    for key in keys:
        value = row.get(key)
        if value is not None and value.strip():
            return value.strip()
    return None


def _parse_optional_float(raw_value: str | None, *, default: float) -> float:
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except ValueError:
        return default


def _infer_assay_type(*parts: str) -> str:
    joined = " ".join(part.lower() for part in parts if part)
    if "atac" in joined:
        return "ATAC-seq"
    if "eclip" in joined:
        return "eCLIP"
    if "pro-cap" in joined or "procap" in joined:
        return "PRO-cap"
    if "ribo" in joined:
        return "ribo-seq"
    if "rna" in joined:
        return "RNA-seq"
    if "chip" in joined or "h3" in joined or "histone" in joined:
        return "Histone ChIP-seq"
    return "Other"


def _load_metadata_rows(metadata_path: Path) -> list[dict[str, str]]:
    if not metadata_path.is_file():
        return []
    with metadata_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return [dict(row) for row in reader]


def _load_split_regions(splits_path: Path) -> tuple[SplitRegion, ...]:
    rows: list[SplitRegion] = []
    with splits_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if len(row) < 4:
                continue
            rows.append(
                SplitRegion(
                    chrom=row[0],
                    start=int(float(row[1])),
                    end=int(float(row[2])),
                    split=row[3],
                )
            )
    return tuple(rows)


def _species_metadata_lookup(rows: list[dict[str, str]], species_name: str) -> dict[str, dict[str, str]]:
    filtered = rows
    species_keys = ("species_common_name", "species", "species_name", "common_name")
    for key in species_keys:
        if any(key in row for row in rows):
            filtered = [row for row in rows if _normalize_metadata_value(row.get(key)) == species_name.lower()]
            break

    lookup: dict[str, dict[str, str]] = {}
    for row in filtered:
        for key in ("file_id", "datasets", "dataset", "track_id", "accession", "id", "filename"):
            candidate = _normalize_metadata_value(row.get(key))
            if candidate and candidate not in lookup:
                lookup[candidate] = row
    return lookup


def load_species_assets(dataset_root: str | Path, species_name: str) -> SpeciesAssets:
    root = Path(dataset_root).expanduser().resolve()
    species_root = root / species_name
    fasta_path = species_root / "genome.fasta"
    splits_path = species_root / "splits.bed"
    if not fasta_path.is_file():
        raise FileNotFoundError(f"Missing NTv3 FASTA for species {species_name!r}: {fasta_path}")
    if not splits_path.is_file():
        raise FileNotFoundError(f"Missing NTv3 splits for species {species_name!r}: {splits_path}")

    metadata_lookup = _species_metadata_lookup(_load_metadata_rows(root / "benchmark_metadata.tsv"), species_name)
    functional_tracks: list[FunctionalTrackInfo] = []
    functional_dir = species_root / "functional_tracks"
    for bigwig_path in sorted(functional_dir.glob("*.bigwig")):
        dataset_id = bigwig_path.stem
        row = metadata_lookup.get(dataset_id.lower(), {})
        display_name = _metadata_first(
            row,
            "track_name_clean",
            "track_name",
            "display_name",
            "name",
            "description",
        ) or dataset_id
        assay_type = _metadata_first(row, "assay_type", "assay", "track_type") or _infer_assay_type(
            display_name,
            dataset_id,
        )
        mean = _parse_optional_float(_metadata_first(row, "mean", "track_mean"), default=1.0)
        functional_tracks.append(
            FunctionalTrackInfo(
                dataset_id=dataset_id,
                path=bigwig_path.resolve(),
                assay_type=assay_type,
                track_name_clean=display_name,
                mean=max(mean, 1e-6),
            )
        )

    annotation_elements = tuple(
        AnnotationElementInfo(dataset_id=bed_path.stem, path=bed_path.resolve())
        for bed_path in sorted((species_root / "genome_annotation").glob("*.bed"))
    )

    return SpeciesAssets(
        species_name=species_name,
        fasta_path=fasta_path.resolve(),
        splits_path=splits_path.resolve(),
        split_regions=_load_split_regions(splits_path),
        functional_tracks=tuple(functional_tracks),
        annotation_elements=annotation_elements,
    )


def create_functional_targets_scaler(track_infos: list[FunctionalTrackInfo]) -> Callable[[torch.Tensor], torch.Tensor]:
    return FunctionalTargetsScaler(
        track_means=torch.tensor([track.mean for track in track_infos], dtype=torch.float32)
    )


def _encode_dna_sequence(sequence: str, *, sequence_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = torch.full((sequence_length,), PAD_ID, dtype=torch.long)
    attention_mask = torch.zeros((sequence_length,), dtype=torch.long)
    encoded = [DNA_VOCAB.get(base.upper(), UNK_ID) for base in sequence[:sequence_length]]
    if encoded:
        input_ids[: len(encoded)] = torch.tensor(encoded, dtype=torch.long)
        attention_mask[: len(encoded)] = 1
    return input_ids, attention_mask


def _sample_regions_for_total_length(
    regions: list[tuple[str, int, int]],
    total_length_needed: int,
    *,
    seed: int = 0,
) -> list[tuple[str, int, int]]:
    sampled_regions: list[tuple[str, int, int]] = []
    rng = np.random.RandomState(seed)
    accumulated_length = 0
    for chrom, start, end in regions:
        region_length = end - start
        remaining = total_length_needed - accumulated_length
        if remaining <= 0:
            break
        if region_length >= remaining:
            max_start = region_length - remaining
            window_start_offset = int(rng.randint(0, max_start + 1)) if max_start > 0 else 0
            sampled_regions.append((chrom, start + window_start_offset, start + window_start_offset + remaining))
            accumulated_length += remaining
            break
        sampled_regions.append((chrom, start, end))
        accumulated_length += region_length
    return sampled_regions


def _get_fasta_handle(fasta_path: str | Path) -> Fasta:
    process_id = os.getpid()
    resolved = str(Path(fasta_path).expanduser().resolve())
    key = (process_id, resolved)
    if key not in _fasta_cache:
        _fasta_cache[key] = Fasta(resolved, as_raw=True, sequence_always_upper=True)
    return _fasta_cache[key]


def _get_bigwig_handle(bigwig_path: str | Path) -> pyBigWig.pyBigWig:
    process_id = os.getpid()
    resolved = str(Path(bigwig_path).expanduser().resolve())
    key = (process_id, resolved)
    if key not in _bigwig_cache:
        _bigwig_cache[key] = pyBigWig.open(resolved)
    return _bigwig_cache[key]


class BedIntervalIndex:
    def __init__(self, intervals_by_chrom: dict[str, list[tuple[int, int]]]) -> None:
        self.intervals_by_chrom = intervals_by_chrom
        self._start_by_chrom = {
            chrom: [start for start, _ in intervals]
            for chrom, intervals in intervals_by_chrom.items()
        }

    @classmethod
    def from_path(cls, bed_path: str | Path) -> BedIntervalIndex:
        intervals_by_chrom: dict[str, list[tuple[int, int]]] = {}
        with Path(bed_path).open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle, delimiter="\t")
            for row in reader:
                if len(row) < 3:
                    continue
                chrom = row[0]
                start = int(row[1])
                end = int(row[2])
                intervals_by_chrom.setdefault(chrom, []).append((start, end))
        for _chrom, intervals in intervals_by_chrom.items():
            intervals.sort(key=lambda item: (item[0], item[1]))
        return cls(intervals_by_chrom)

    def labels(self, chrom: str, start: int, end: int) -> np.ndarray:
        labels = np.zeros(end - start, dtype=np.int64)
        chrom = _resolve_chrom_name(chrom, set(self.intervals_by_chrom))
        intervals = self.intervals_by_chrom.get(chrom, [])
        if not intervals:
            return labels
        starts = self._start_by_chrom[chrom]
        idx = max(0, bisect.bisect_left(starts, start) - 1)
        for interval_start, interval_end in intervals[idx:]:
            if interval_start >= end:
                break
            if interval_end <= start:
                continue
            local_start = max(interval_start, start) - start
            local_end = min(interval_end, end) - start
            labels[local_start:local_end] = 1
        return labels


def _get_bed_handle(bed_path: str | Path) -> BedIntervalIndex:
    process_id = os.getpid()
    resolved = str(Path(bed_path).expanduser().resolve())
    key = (process_id, resolved)
    if key not in _bed_cache:
        _bed_cache[key] = BedIntervalIndex.from_path(resolved)
    return _bed_cache[key]


class _WindowedGenomeDataset(Dataset):
    def __init__(
        self,
        *,
        fasta_path: Path,
        split_regions: tuple[SplitRegion, ...],
        split: str,
        sequence_length: int,
        overlap: float = 0.0,
        keep_target_center_fraction: float = 1.0,
        limit_num_samples: int | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.fasta_path = fasta_path
        self.split = split
        self.sequence_length = sequence_length
        self.keep_target_center_fraction = keep_target_center_fraction
        self.stride = max(1, int((1.0 - overlap) * sequence_length))

        region_list = [
            (row.chrom, row.start, row.end)
            for row in split_regions
            if row.split == split and (row.end - row.start) >= sequence_length
        ]
        if limit_num_samples is not None:
            region_list = _sample_regions_for_total_length(
                region_list,
                total_length_needed=limit_num_samples * sequence_length,
                seed=seed,
            )
        self.region_info, self._cumulative_starts, self.num_samples = self._process_regions(region_list)
        if self.num_samples <= 0:
            raise ValueError(f"No valid regions available for split {split!r}.")

    def _process_regions(
        self,
        regions: list[tuple[str, int, int]],
    ) -> tuple[list[dict[str, int | str]], list[int], int]:
        region_info: list[dict[str, int | str]] = []
        cumulative_starts: list[int] = []
        sample_count = 0
        for chrom, start, end in regions:
            num_samples = 1 + max(0, (end - start - self.sequence_length) // self.stride)
            if num_samples <= 0:
                continue
            cumulative_starts.append(sample_count)
            region_info.append(
                {
                    "chr_name": chrom,
                    "region_start_offset": start,
                    "num_samples": num_samples,
                }
            )
            sample_count += num_samples
        return region_info, cumulative_starts, sample_count

    def __len__(self) -> int:
        return self.num_samples

    def _window_for_index(self, idx: int) -> tuple[str, int, int]:
        chromosome_idx = bisect.bisect_right(self._cumulative_starts, idx) - 1
        info = self.region_info[chromosome_idx]
        chrom = str(info["chr_name"])
        region_start = int(info["region_start_offset"])
        index_within_region = idx - self._cumulative_starts[chromosome_idx]
        start = region_start + index_within_region * self.stride
        end = start + self.sequence_length
        return chrom, start, end

    def _build_token_tensors(self, chrom: str, start: int, end: int) -> tuple[torch.Tensor, torch.Tensor]:
        fasta = _get_fasta_handle(self.fasta_path)
        chrom = _resolve_chrom_name(chrom, set(fasta.keys()))
        sequence = fasta[chrom][start:end]
        return _encode_dna_sequence(sequence, sequence_length=self.sequence_length)


class GenomeBigWigDataset(_WindowedGenomeDataset):
    def __init__(
        self,
        *,
        fasta_path: Path,
        split_regions: tuple[SplitRegion, ...],
        track_infos: list[FunctionalTrackInfo],
        split: str,
        sequence_length: int,
        transform_fn: Callable[[torch.Tensor], torch.Tensor],
        overlap: float = 0.0,
        keep_target_center_fraction: float = 1.0,
        limit_num_samples: int | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__(
            fasta_path=fasta_path,
            split_regions=split_regions,
            split=split,
            sequence_length=sequence_length,
            overlap=overlap,
            keep_target_center_fraction=keep_target_center_fraction,
            limit_num_samples=limit_num_samples,
            seed=seed,
        )
        self.track_infos = track_infos
        self.transform_fn = transform_fn

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        chrom, start, end = self._window_for_index(idx)
        item_logger = DatasetItemLogger.for_current_process()
        dataset_kind = f"functional:{self.split}"
        item_ordinal = item_logger.preview_ordinal(dataset_kind=dataset_kind) if item_logger is not None else None
        if item_ordinal is not None:
            log.info(
                "NTv3 dataset item requested: kind=%s ordinal=%d idx=%d chrom=%s start=%d end=%d",
                dataset_kind,
                item_ordinal,
                idx,
                chrom,
                start,
                end,
            )
        input_ids, attention_mask = self._build_token_tensors(chrom, start, end)
        targets = np.array(
            [
                _get_bigwig_handle(track.path).values(
                    _resolve_chrom_name(chrom, set(_get_bigwig_handle(track.path).chroms().keys())),
                    start,
                    end,
                    numpy=True,
                )
                for track in self.track_infos
            ],
            dtype=np.float32,
        ).T
        target_tensor = torch.from_numpy(np.nan_to_num(targets, nan=0.0))
        if self.keep_target_center_fraction < 1.0:
            target_tensor = crop_center(target_tensor, self.keep_target_center_fraction)
        target_tensor = self.transform_fn(target_tensor)
        if item_logger is not None:
            event = item_logger.record(
                dataset_kind=dataset_kind,
                idx=idx,
                chrom=chrom,
                start=start,
                end=end,
                input_shape=tuple(input_ids.shape),
                target_shape=tuple(target_tensor.shape),
                extra={"num_tracks": len(self.track_infos)},
            )
            if event is not None:
                log.info(
                    "NTv3 dataset item loaded: kind=%s ordinal=%d idx=%d chrom=%s start=%d end=%d "
                    "input_shape=%s target_shape=%s num_tracks=%d",
                    dataset_kind,
                    event["ordinal"],
                    idx,
                    chrom,
                    start,
                    end,
                    tuple(input_ids.shape),
                    tuple(target_tensor.shape),
                    len(self.track_infos),
                )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "targets": target_tensor,
        }


class GenomeAnnotationDataset(_WindowedGenomeDataset):
    def __init__(
        self,
        *,
        fasta_path: Path,
        split_regions: tuple[SplitRegion, ...],
        element_infos: list[AnnotationElementInfo],
        split: str,
        sequence_length: int,
        overlap: float = 0.0,
        keep_target_center_fraction: float = 1.0,
        limit_num_samples: int | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__(
            fasta_path=fasta_path,
            split_regions=split_regions,
            split=split,
            sequence_length=sequence_length,
            overlap=overlap,
            keep_target_center_fraction=keep_target_center_fraction,
            limit_num_samples=limit_num_samples,
            seed=seed,
        )
        self.element_infos = element_infos

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        chrom, start, end = self._window_for_index(idx)
        item_logger = DatasetItemLogger.for_current_process()
        dataset_kind = f"annotation:{self.split}"
        item_ordinal = item_logger.preview_ordinal(dataset_kind=dataset_kind) if item_logger is not None else None
        if item_ordinal is not None:
            log.info(
                "NTv3 dataset item requested: kind=%s ordinal=%d idx=%d chrom=%s start=%d end=%d",
                dataset_kind,
                item_ordinal,
                idx,
                chrom,
                start,
                end,
            )
        input_ids, attention_mask = self._build_token_tensors(chrom, start, end)
        labels = np.stack(
            [_get_bed_handle(element.path).labels(chrom, start, end) for element in self.element_infos],
            axis=-1,
        )
        label_tensor = torch.from_numpy(labels.astype(np.int64))
        if self.keep_target_center_fraction < 1.0:
            label_tensor = crop_center(label_tensor, self.keep_target_center_fraction)
        if item_logger is not None:
            event = item_logger.record(
                dataset_kind=dataset_kind,
                idx=idx,
                chrom=chrom,
                start=start,
                end=end,
                input_shape=tuple(input_ids.shape),
                target_shape=tuple(label_tensor.shape),
                extra={"num_elements": len(self.element_infos)},
            )
            if event is not None:
                log.info(
                    "NTv3 dataset item loaded: kind=%s ordinal=%d idx=%d chrom=%s start=%d end=%d "
                    "input_shape=%s target_shape=%s num_elements=%d",
                    dataset_kind,
                    event["ordinal"],
                    idx,
                    chrom,
                    start,
                    end,
                    tuple(input_ids.shape),
                    tuple(label_tensor.shape),
                    len(self.element_infos),
                )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "targets": label_tensor,
        }
