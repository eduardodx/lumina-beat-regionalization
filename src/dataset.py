from __future__ import annotations

import gzip
import json
import os
import random
from collections.abc import Iterator, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal, TextIO, TypedDict, cast

import numpy as np
import pyBigWig
import torch
import torch.distributed as dist
from pyfaidx import Fasta
from torch.utils.data import IterableDataset

from src.clinvar_blocklist import ClinVarBlocklist
from src.constants import (
    AA_NON_CDS,
    ALLELE_EFFECT_IGNORE_INDEX,
    CDS_PHASE_NONE,
    CODON_IGNORE_INDEX,
    CODON_TO_AA,
    COUNTERFACTUAL_EFFECT_IGNORE_INDEX,
    DNA_VOCAB,
    MASK_ID,
    MUTATION_EFFECT_IGNORE_INDEX,
    MUTATION_EFFECT_MISSENSE,
    MUTATION_EFFECT_STOP,
    PAD_ID,
    REGION_CDS,
    REGION_INTERGENIC,
    REGION_INTRON,
    REGION_NONCODING_EXON,
    REGION_UTR,
    SNV_ALT_TO_INDEX,
    SNV_BASES,
    STRUCT_BACKGROUND,
    STRUCT_SPLICE_CORE,
    STRUCT_SPLICE_REGION,
    UNK_ID,
    VOCAB_SIZE,
)
from src.encode_tracks import EncodeTrackSpec
from src.mutation_effects import extract_codon

PLUS_CDS_PHASE_PATTERN = (0, 1, 2)
MINUS_CDS_PHASE_PATTERN = (2, 1, 0)
CODON_BASES = SNV_BASES
SNV_TOKEN_ID_TO_BASE = {DNA_VOCAB[base]: base for base in SNV_BASES}
CODON_TO_INDEX = {
    f"{base0}{base1}{base2}": index
    for index, (base0, base1, base2) in enumerate(
        (a, b, c) for a in CODON_BASES for b in CODON_BASES for c in CODON_BASES
    )
}
PATHOGENIC_MUTATION_EFFECT_CLASSES = frozenset({MUTATION_EFFECT_MISSENSE, MUTATION_EFFECT_STOP})
CurriculumPhase = Literal["exon_centric", "genome_wide"]
AbraomDatasetArm = Literal["wild_only", "abraom_uniform", "abraom_weighted"]


@dataclass(frozen=True)
class AbraomShard:
    shard_id: int
    fasta_path: Path
    meta_path: Path
    start_id: int | None = None
    end_id: int | None = None


@dataclass(frozen=True)
class IntervalIndex:
    starts: np.ndarray
    ends: np.ndarray

    @classmethod
    def empty(cls) -> IntervalIndex:
        empty = np.empty(0, dtype=np.int64)
        return cls(starts=empty, ends=empty)

    def overlapping_bounds(self, window_start: int, window_end: int) -> tuple[int, int]:
        left = int(np.searchsorted(self.ends, window_start, side="right"))
        right = int(np.searchsorted(self.starts, window_end, side="left"))
        return left, right


@dataclass(frozen=True)
class CdsDetailedInterval:
    start: int
    end: int
    strand: str
    frame: int


@dataclass(frozen=True)
class CdsDetailedLookup:
    intervals: tuple[CdsDetailedInterval, ...]
    priorities: np.ndarray
    starts: np.ndarray
    max_ends: np.ndarray

    @classmethod
    def empty(cls) -> CdsDetailedLookup:
        empty_int = np.empty(0, dtype=np.int64)
        empty_priority = np.empty(0, dtype=np.int8)
        return cls(intervals=(), priorities=empty_priority, starts=empty_int, max_ends=empty_int)

    @classmethod
    def from_records(cls, records: list[tuple[int, CdsDetailedInterval]]) -> CdsDetailedLookup:
        if not records:
            return cls.empty()

        ordered = sorted(
            records,
            key=lambda item: (item[1].start, item[0], item[1].end, item[1].strand, item[1].frame),
        )
        intervals = tuple(interval for _priority, interval in ordered)
        priorities = np.fromiter((priority for priority, _interval in ordered), dtype=np.int8, count=len(ordered))
        starts = np.fromiter((interval.start for interval in intervals), dtype=np.int64, count=len(intervals))
        ends = np.fromiter((interval.end for interval in intervals), dtype=np.int64, count=len(intervals))
        max_ends = np.maximum.accumulate(ends)
        return cls(intervals=intervals, priorities=priorities, starts=starts, max_ends=max_ends)

    def overlapping_indices(self, window_start: int, window_end: int) -> range:
        if not self.intervals:
            return range(0)

        right = int(np.searchsorted(self.starts, window_end, side="left"))
        if right <= 0:
            return range(0)
        left = int(np.searchsorted(self.max_ends[:right], window_start, side="right"))
        return range(left, right)


@dataclass(frozen=True)
class ChromosomeAnnotationIndex:
    core: IntervalIndex
    region: IntervalIndex
    exon: IntervalIndex
    cds: IntervalIndex = field(default_factory=IntervalIndex.empty)
    utr: IntervalIndex = field(default_factory=IntervalIndex.empty)
    transcript: IntervalIndex = field(default_factory=IntervalIndex.empty)
    cds_detailed: list[CdsDetailedInterval] = field(default_factory=list)
    cds_detailed_lookup: CdsDetailedLookup = field(default_factory=CdsDetailedLookup.empty)


GenomeAnnotationIndex = dict[str, ChromosomeAnnotationIndex]


class DatasetSample(TypedDict):
    input_ids: torch.Tensor
    phylo100: torch.Tensor
    phylo470: torch.Tensor
    structure_labels: torch.Tensor
    region_labels: torch.Tensor
    aa_labels: torch.Tensor
    cds_phase: torch.Tensor
    codon_phylo_target: torch.Tensor
    chrom: str
    start: int
    end: int
    n_fraction: float
    splice_positive_fraction: float
    splice_core_fraction: float
    exon_fraction: float
    cds_fraction: float
    utr_fraction: float
    intron_fraction: float
    n_filter_fallback_used: bool


class MultiTaskBatch(TypedDict):
    input_ids: torch.Tensor
    alt_input_ids: torch.Tensor
    attention_mask: torch.Tensor
    alt_attention_mask: torch.Tensor
    mlm_labels: torch.Tensor
    mask_positions: torch.Tensor
    aux_valid_mask: torch.Tensor
    phylo100: torch.Tensor
    phylo470: torch.Tensor
    structure_labels: torch.Tensor
    region_labels: torch.Tensor
    aa_labels: torch.Tensor
    cds_phase: torch.Tensor
    codon_phylo_target: torch.Tensor
    rc_input_ids: torch.Tensor
    rc_attention_mask: torch.Tensor
    encode_track_names: list[str]
    chroms: list[str]
    n_fraction: float
    splice_positive_fraction: float
    splice_core_fraction: float
    exon_fraction: float
    cds_fraction: float
    utr_fraction: float
    intron_fraction: float
    n_filter_fallback_fraction: float
    mask_density_intergenic: float
    mask_density_intron: float
    mask_density_noncoding_exon: float
    mask_density_utr: float
    mask_density_cds: float


def encode_dna(seq: str) -> list[int]:
    return [DNA_VOCAB.get(base.upper(), UNK_ID) for base in seq]


def reverse_complement_ids(input_ids: torch.Tensor) -> torch.Tensor:
    """
    input_ids: [B, L] ou [L]
    Complementa A<->T, C<->G, mantém PAD/MASK/UNK/N.
    """
    comp = torch.full((VOCAB_SIZE,), fill_value=UNK_ID, dtype=input_ids.dtype, device=input_ids.device)
    comp[PAD_ID] = PAD_ID
    comp[MASK_ID] = MASK_ID
    comp[UNK_ID] = UNK_ID
    comp[DNA_VOCAB["A"]] = DNA_VOCAB["T"]
    comp[DNA_VOCAB["T"]] = DNA_VOCAB["A"]
    comp[DNA_VOCAB["C"]] = DNA_VOCAB["G"]
    comp[DNA_VOCAB["G"]] = DNA_VOCAB["C"]
    comp[DNA_VOCAB["N"]] = DNA_VOCAB["N"]
    rc = comp[input_ids]
    return torch.flip(rc, dims=[-1])


def reverse_complement_dna(seq: str) -> str:
    complement = str.maketrans({"A": "T", "C": "G", "G": "C", "T": "A", "N": "N"})
    return seq.upper().translate(complement)[::-1]


GtfAttributeValue = str | list[str]


def parse_gtf_attributes(attr_str: str) -> dict[str, GtfAttributeValue]:
    attrs: dict[str, GtfAttributeValue] = {}
    parts = [part.strip() for part in attr_str.strip().split(";") if part.strip()]
    for part in parts:
        if " " not in part:
            continue
        key, value = part.split(" ", 1)
        parsed_value = value.strip().strip('"')
        existing = attrs.get(key)
        if existing is None:
            attrs[key] = parsed_value
        elif isinstance(existing, list):
            existing.append(parsed_value)
        else:
            attrs[key] = [existing, parsed_value]
    return attrs


def get_gtf_attribute_values(attrs: dict[str, GtfAttributeValue], key: str) -> tuple[str, ...]:
    value = attrs.get(key)
    if value is None:
        return ()
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def get_gtf_attribute_first(attrs: dict[str, GtfAttributeValue], key: str) -> str | None:
    values = get_gtf_attribute_values(attrs, key)
    if not values:
        return None
    return values[0]


def resolve_cds_priority(attrs: dict[str, GtfAttributeValue]) -> int | None:
    tags = set(get_gtf_attribute_values(attrs, "tag"))
    canonical_values = {value.lower() for value in get_gtf_attribute_values(attrs, "Ensembl_canonical")}
    if "Ensembl_canonical" in tags or "1" in canonical_values or "true" in canonical_values:
        return 0
    if "basic" in tags:
        return 1
    return None


def parse_cds_frame(frame_value: str, *, chrom: str, transcript_id: str) -> int:
    if frame_value == ".":
        raise ValueError(f"CDS feature on {chrom} transcript {transcript_id} is missing a frame value.")
    frame = int(frame_value)
    if frame not in (0, 1, 2):
        raise ValueError(f"CDS frame must be 0, 1, or 2 on {chrom} transcript {transcript_id}, got {frame_value!r}.")
    return frame


def open_text_maybe_gzip(path: str) -> TextIO:
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path)


def close_maybe(handle: Any) -> None:
    close = getattr(handle, "close", None)
    if callable(close):
        with suppress(Exception):
            close()


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []

    sorted_intervals = sorted(intervals, key=lambda interval: (interval[0], interval[1]))
    merged = [sorted_intervals[0]]
    for start, end in sorted_intervals[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def build_interval_index(intervals: list[tuple[int, int]]) -> IntervalIndex:
    merged = merge_intervals(intervals)
    if not merged:
        return IntervalIndex.empty()

    starts = np.fromiter((start for start, _ in merged), dtype=np.int64, count=len(merged))
    ends = np.fromiter((end for _, end in merged), dtype=np.int64, count=len(merged))
    return IntervalIndex(starts=starts, ends=ends)


def validate_coordinate_integrity(
    fasta: Fasta,
    phylo100_bw: pyBigWig.pyBigWig,
    phylo470_bw: pyBigWig.pyBigWig,
    chromosomes: list[str],
) -> dict[str, int]:
    fasta_keys = set(cast(list[str], list(fasta.keys())))
    phylo100_chroms = phylo100_bw.chroms()
    phylo470_chroms = phylo470_bw.chroms()

    errors: list[str] = []
    chrom_sizes: dict[str, int] = {}

    for chrom in chromosomes:
        missing_sources: list[str] = []
        if chrom not in fasta_keys:
            missing_sources.append("FASTA")
        if chrom not in phylo100_chroms:
            missing_sources.append("PhyloP100")
        if chrom not in phylo470_chroms:
            missing_sources.append("PhyloP470")
        if missing_sources:
            errors.append(f"{chrom}: missing in {'/'.join(missing_sources)}")
            continue

        fasta_len = len(fasta[chrom])
        phylo100_len = int(phylo100_chroms[chrom])
        phylo470_len = int(phylo470_chroms[chrom])
        if fasta_len != phylo100_len or fasta_len != phylo470_len:
            errors.append(
                f"{chrom}: FASTA={fasta_len}, PhyloP100={phylo100_len}, PhyloP470={phylo470_len}"
            )
            continue

        chrom_sizes[chrom] = fasta_len

    if errors:
        joined = "\n  - ".join(errors)
        raise ValueError(f"Coordinate integrity validation failed:\n  - {joined}")

    return chrom_sizes


def build_annotation_index_from_gtf(
    gtf_path: str,
    chrom_sizes: dict[str, int],
    transcript_types_keep: set[str] | None = None,
    core_radius: int = 2,
    region_radius: int = 10,
) -> GenomeAnnotationIndex:
    """
    Constrói índices fundidos de éxon e splice a partir de éxons por transcrito.
    Usa convenção BED-like: 0-based half-open [start, end).
    """
    if transcript_types_keep is None:
        transcript_types_keep = {
            "protein_coding",
            "lncRNA",
            "processed_transcript",
            "retained_intron",
            "nonsense_mediated_decay",
        }

    chromosomes = list(chrom_sizes)
    parsed_features = {"exon", "CDS", "UTR", "transcript"}
    exon_by_transcript: dict[tuple[str, str, str], list[tuple[int, int]]] = {}
    exon_intervals_by_chrom: dict[str, list[tuple[int, int]]] = {chrom: [] for chrom in chromosomes}
    core_intervals_by_chrom: dict[str, list[tuple[int, int]]] = {chrom: [] for chrom in chromosomes}
    region_intervals_by_chrom: dict[str, list[tuple[int, int]]] = {chrom: [] for chrom in chromosomes}
    cds_intervals_by_chrom: dict[str, list[tuple[int, int]]] = {chrom: [] for chrom in chromosomes}
    cds_detailed_records_by_chrom: dict[str, list[tuple[int, CdsDetailedInterval]]] = {
        chrom: [] for chrom in chromosomes
    }
    utr_intervals_by_chrom: dict[str, list[tuple[int, int]]] = {chrom: [] for chrom in chromosomes}
    transcript_intervals_by_chrom: dict[str, list[tuple[int, int]]] = {chrom: [] for chrom in chromosomes}

    with open_text_maybe_gzip(gtf_path) as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9:
                continue

            chrom, _source, feature, start, end, _score, strand, frame_value, attrs = fields
            if chrom not in chrom_sizes:
                continue
            if feature not in parsed_features:
                continue

            attr = parse_gtf_attributes(attrs)
            transcript_id = get_gtf_attribute_first(attr, "transcript_id")
            transcript_type = get_gtf_attribute_first(attr, "transcript_type") or get_gtf_attribute_first(
                attr, "transcript_biotype"
            )

            if transcript_id is None:
                continue
            if transcript_type is not None and transcript_type not in transcript_types_keep:
                continue

            start_1 = int(start)
            end_1 = int(end)
            chrom_size = chrom_sizes[chrom]
            if start_1 < 1 or end_1 < start_1 or end_1 > chrom_size:
                raise ValueError(
                    f"GTF coordinate integrity failed for {chrom}: start={start_1} end={end_1} size={chrom_size}"
                )

            bed_start = start_1 - 1

            if feature == "exon":
                exon_by_transcript.setdefault((chrom, strand, transcript_id), []).append((start_1, end_1))
                exon_intervals_by_chrom[chrom].append((bed_start, end_1))
            elif feature == "CDS":
                cds_priority = resolve_cds_priority(attr)
                if cds_priority is None:
                    continue
                cds_frame = parse_cds_frame(frame_value, chrom=chrom, transcript_id=transcript_id)
                cds_intervals_by_chrom[chrom].append((bed_start, end_1))
                cds_detailed_records_by_chrom[chrom].append(
                    (
                        cds_priority,
                        CdsDetailedInterval(start=bed_start, end=end_1, strand=strand, frame=cds_frame),
                    )
                )
            elif feature == "UTR":
                utr_intervals_by_chrom[chrom].append((bed_start, end_1))
            elif feature == "transcript":
                transcript_intervals_by_chrom[chrom].append((bed_start, end_1))

    for (chrom, _strand, _transcript_id), exons in exon_by_transcript.items():
        if len(exons) < 2:
            continue

        sorted_exons = sorted(exons, key=lambda interval: interval[0])
        for i in range(len(sorted_exons) - 1):
            left_exon = sorted_exons[i]
            right_exon = sorted_exons[i + 1]

            donor_1 = left_exon[1]
            acceptor_1 = right_exon[0]

            for pos_1 in (donor_1, acceptor_1):
                core_start_1 = max(1, pos_1 - core_radius)
                core_end_1 = pos_1 + core_radius
                region_start_1 = max(1, pos_1 - region_radius)
                region_end_1 = pos_1 + region_radius

                core_intervals_by_chrom[chrom].append((core_start_1 - 1, core_end_1))
                region_intervals_by_chrom[chrom].append((region_start_1 - 1, region_end_1))

    annotation_index: GenomeAnnotationIndex = {}
    for chrom in chromosomes:
        cds_detailed_lookup = CdsDetailedLookup.from_records(cds_detailed_records_by_chrom[chrom])
        annotation_index[chrom] = ChromosomeAnnotationIndex(
            core=build_interval_index(core_intervals_by_chrom[chrom]),
            region=build_interval_index(region_intervals_by_chrom[chrom]),
            exon=build_interval_index(exon_intervals_by_chrom[chrom]),
            cds=build_interval_index(cds_intervals_by_chrom[chrom]),
            utr=build_interval_index(utr_intervals_by_chrom[chrom]),
            transcript=build_interval_index(transcript_intervals_by_chrom[chrom]),
            cds_detailed=list(cds_detailed_lookup.intervals),
            cds_detailed_lookup=cds_detailed_lookup,
        )

    return annotation_index


def fill_dense_labels(
    labels: np.ndarray,
    window_start: int,
    window_end: int,
    interval_index: IntervalIndex,
    label_value: int,
) -> None:
    left, right = interval_index.overlapping_bounds(window_start, window_end)
    for start, end in zip(interval_index.starts[left:right], interval_index.ends[left:right], strict=False):
        overlap_start = max(window_start, int(start))
        overlap_end = min(window_end, int(end))
        if overlap_start < overlap_end:
            labels[overlap_start - window_start : overlap_end - window_start] = label_value


def intervals_to_dense_labels(
    chrom: str,
    window_start: int,
    window_end: int,
    annotation_index: GenomeAnnotationIndex,
) -> np.ndarray:
    """
    Gera labels por posição para uma janela [window_start, window_end)
    com prioridade core > region > background.
    """
    window_len = window_end - window_start
    labels = np.zeros(window_len, dtype=np.int64)

    annotations = annotation_index.get(chrom)
    if annotations is None:
        return labels

    fill_dense_labels(labels, window_start, window_end, annotations.region, STRUCT_SPLICE_REGION)
    fill_dense_labels(labels, window_start, window_end, annotations.core, STRUCT_SPLICE_CORE)
    return labels


def intervals_to_dense_region_labels(
    chrom: str,
    window_start: int,
    window_end: int,
    annotation_index: GenomeAnnotationIndex,
) -> np.ndarray:
    """Per-position region type labels for [window_start, window_end).

    Priority (each overrides previous): transcript→INTRON, exon→NONCODING_EXON,
    UTR→UTR, CDS→CDS.  Default (unfilled) = REGION_INTERGENIC.
    """
    window_len = window_end - window_start
    labels = np.zeros(window_len, dtype=np.int64)

    annotations = annotation_index.get(chrom)
    if annotations is None:
        return labels

    fill_dense_labels(labels, window_start, window_end, annotations.transcript, REGION_INTRON)
    fill_dense_labels(labels, window_start, window_end, annotations.exon, REGION_NONCODING_EXON)
    fill_dense_labels(labels, window_start, window_end, annotations.utr, REGION_UTR)
    fill_dense_labels(labels, window_start, window_end, annotations.cds, REGION_CDS)
    return labels


def resolve_cds_lookup(
    cds_detailed: Sequence[CdsDetailedInterval],
    cds_lookup: CdsDetailedLookup | None = None,
) -> CdsDetailedLookup:
    if cds_lookup is not None:
        return cds_lookup
    return CdsDetailedLookup.from_records([(1, interval) for interval in cds_detailed])


def codon_window_from_phase_values(phase_values: Sequence[int], position: int) -> tuple[int, int] | None:
    phase = int(phase_values[position])
    if phase < 0:
        return None

    plus_start = position - phase
    if plus_start >= 0 and plus_start + 3 <= len(phase_values):
        plus_pattern = tuple(int(phase_values[plus_start + offset]) for offset in range(3))
        if plus_pattern == PLUS_CDS_PHASE_PATTERN:
            return plus_start, plus_start + 3

    minus_start = position + phase - 2
    if minus_start >= 0 and minus_start + 3 <= len(phase_values):
        minus_pattern = tuple(int(phase_values[minus_start + offset]) for offset in range(3))
        if minus_pattern == MINUS_CDS_PHASE_PATTERN:
            return minus_start, minus_start + 3

    return None


def assign_cds_window(
    window_start: int,
    window_end: int,
    cds_detailed: Sequence[CdsDetailedInterval],
    cds_lookup: CdsDetailedLookup | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    window_len = window_end - window_start
    phase = np.full(window_len, CDS_PHASE_NONE, dtype=np.int64)
    interval_indices = np.full(window_len, -1, dtype=np.int32)
    best_priority = np.full(window_len, 3, dtype=np.int8)

    lookup = resolve_cds_lookup(cds_detailed, cds_lookup)
    for interval_index in lookup.overlapping_indices(window_start, window_end):
        interval = lookup.intervals[interval_index]
        overlap_start = max(window_start, interval.start)
        overlap_end = min(window_end, interval.end)
        if overlap_start >= overlap_end:
            continue

        local_start = overlap_start - window_start
        local_end = overlap_end - window_start
        genomic_positions = np.arange(overlap_start, overlap_end, dtype=np.int64)
        if interval.strand == "+":
            local_phase = (genomic_positions - interval.start + interval.frame) % 3
        else:
            local_phase = (interval.end - 1 - genomic_positions + interval.frame) % 3

        region_best_priority = best_priority[local_start:local_end]
        update_mask = lookup.priorities[interval_index] < region_best_priority
        if not np.any(update_mask):
            continue

        region_best_priority[update_mask] = lookup.priorities[interval_index]
        phase_slice = phase[local_start:local_end]
        phase_slice[update_mask] = local_phase[update_mask]
        interval_slice = interval_indices[local_start:local_end]
        interval_slice[update_mask] = interval_index

    return phase, interval_indices


def build_cds_phase(
    window_start: int,
    window_end: int,
    cds_detailed: Sequence[CdsDetailedInterval],
    *,
    cds_lookup: CdsDetailedLookup | None = None,
) -> np.ndarray:
    phase, _interval_indices = assign_cds_window(window_start, window_end, cds_detailed, cds_lookup)
    return phase


def build_aa_labels(
    window_start: int,
    window_end: int,
    cds_detailed: Sequence[CdsDetailedInterval],
    seq: str,
    *,
    cds_lookup: CdsDetailedLookup | None = None,
    cds_phase: np.ndarray | None = None,
    interval_indices: np.ndarray | None = None,
) -> np.ndarray:
    window_len = window_end - window_start
    aa_labels = np.full(window_len, AA_NON_CDS, dtype=np.int64)
    lookup = resolve_cds_lookup(cds_detailed, cds_lookup)

    if cds_phase is None or interval_indices is None:
        cds_phase, interval_indices = assign_cds_window(window_start, window_end, cds_detailed, lookup)

    for local_position, interval_index in enumerate(interval_indices):
        if interval_index < 0:
            continue

        interval = lookup.intervals[int(interval_index)]
        phase = int(cds_phase[local_position])
        genomic_position = window_start + local_position

        if interval.strand == "+":
            codon_start = genomic_position - phase
            codon_end = codon_start + 3
            if codon_start < interval.start or codon_end > interval.end:
                continue
            if codon_start < window_start or codon_end > window_end:
                continue
            codon_seq = seq[codon_start - window_start : codon_end - window_start].upper()
        else:
            codon_start = genomic_position + phase - 2
            codon_end = codon_start + 3
            if codon_start < interval.start or codon_end > interval.end:
                continue
            if codon_start < window_start or codon_end > window_end:
                continue
            genomic_codon = seq[codon_start - window_start : codon_end - window_start].upper()
            codon_seq = reverse_complement_dna(genomic_codon)

        if len(codon_seq) != 3 or "N" in codon_seq:
            continue

        aa_labels[local_position] = CODON_TO_AA.get(codon_seq, AA_NON_CDS)

    return aa_labels


def build_codon_phylo_target(
    phylo100_np: np.ndarray,
    cds_phase_np: np.ndarray,
) -> np.ndarray:
    codon_phylo_target = np.zeros_like(phylo100_np, dtype=np.float32)
    phase_values = cds_phase_np.tolist()

    for position, phase in enumerate(phase_values):
        if phase < 0:
            continue
        codon_window = codon_window_from_phase_values(phase_values, position)
        if codon_window is None:
            continue
        codon_start, codon_end = codon_window
        codon_phylo_target[position] = float(np.mean(phylo100_np[codon_start:codon_end]))

    return codon_phylo_target


def build_codon_labels(
    window_start: int,
    window_end: int,
    cds_detailed: Sequence[CdsDetailedInterval],
    seq: str,
    *,
    cds_lookup: CdsDetailedLookup | None = None,
    cds_phase: np.ndarray | None = None,
    interval_indices: np.ndarray | None = None,
) -> np.ndarray:
    window_len = window_end - window_start
    codon_labels = np.full(window_len, CODON_IGNORE_INDEX, dtype=np.int64)
    lookup = resolve_cds_lookup(cds_detailed, cds_lookup)

    if cds_phase is None or interval_indices is None:
        cds_phase, interval_indices = assign_cds_window(window_start, window_end, cds_detailed, lookup)

    for local_position, interval_index in enumerate(interval_indices):
        if interval_index < 0 or int(cds_phase[local_position]) != 0:
            continue

        interval = lookup.intervals[int(interval_index)]
        codon_seq = extract_codon(seq, local_position, 0, interval.strand)
        if codon_seq is None:
            continue
        codon_index = CODON_TO_INDEX.get(codon_seq)
        if codon_index is None:
            continue
        codon_labels[local_position] = codon_index

    return codon_labels


def compute_mask_density_per_region(
    mask_positions: torch.Tensor,
    region_labels: torch.Tensor,
) -> dict[str, float]:
    densities: dict[str, float] = {}
    for region_name, region_index in (
        ("intergenic", REGION_INTERGENIC),
        ("intron", REGION_INTRON),
        ("noncoding_exon", REGION_NONCODING_EXON),
        ("utr", REGION_UTR),
        ("cds", REGION_CDS),
    ):
        region_mask = region_labels == region_index
        denom = float(region_mask.sum().item())
        if denom <= 0.0:
            densities[f"mask_density_{region_name}"] = 0.0
        else:
            densities[f"mask_density_{region_name}"] = float(
                (mask_positions & region_mask).sum().item() / denom
            )
    return densities


def build_conservation_bin_labels(phylo: torch.Tensor) -> torch.Tensor:
    thresholds = torch.tensor([-0.5, -0.05, 0.05, 0.25, 0.5], dtype=phylo.dtype, device=phylo.device)
    finite = torch.isfinite(phylo)
    labels = torch.zeros_like(phylo, dtype=torch.long)
    labels[finite] = torch.bucketize(phylo[finite], thresholds) + 1
    return labels


def build_splice_distance_bin_labels(structure_labels: torch.Tensor, region_labels: torch.Tensor) -> torch.Tensor:
    labels = torch.zeros_like(structure_labels, dtype=torch.long)
    labels = torch.where(structure_labels == STRUCT_SPLICE_CORE, torch.ones_like(labels), labels)
    labels = torch.where(
        structure_labels == STRUCT_SPLICE_REGION,
        torch.full_like(labels, 3),
        labels,
    )
    labels = torch.where(
        (labels == 0) & (region_labels == REGION_INTRON),
        torch.full_like(labels, 6),
        labels,
    )
    return labels


def build_codon_pos_labels(cds_phase: torch.Tensor) -> torch.Tensor:
    labels = torch.zeros_like(cds_phase, dtype=torch.long)
    cds_mask = cds_phase >= 0
    labels[cds_mask] = cds_phase[cds_mask].to(dtype=torch.long) + 1
    return labels


def _sample_multinomial_index(weights: torch.Tensor) -> int | None:
    positive = torch.clamp(weights, min=0.0)
    total = float(positive.sum().item())
    if total <= 0.0:
        return None
    return int(torch.multinomial(positive / total, num_samples=1).item())


def _dna_base_mask(input_ids: torch.Tensor) -> torch.Tensor:
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for token_id in (DNA_VOCAB["A"], DNA_VOCAB["C"], DNA_VOCAB["G"], DNA_VOCAB["T"]):
        mask |= input_ids == token_id
    return mask


def _choose_counterfactual_edit(
    *,
    reference_ids: torch.Tensor,
    phylo100: torch.Tensor,
    structure_labels: torch.Tensor,
    mutation_effect_labels: torch.Tensor,
    valid_edit_mask: torch.Tensor,
) -> tuple[int, int] | None:
    valid_base_mask = _dna_base_mask(reference_ids) & valid_edit_mask

    splice_positions = torch.nonzero(
        (structure_labels == STRUCT_SPLICE_CORE) & valid_base_mask,
        as_tuple=False,
    ).flatten()
    if len(splice_positions) > 0 and random.random() < 0.2:
        edit_position = int(splice_positions[random.randint(0, len(splice_positions) - 1)].item())
        ref_id = int(reference_ids[edit_position].item())
        replacement_by_ref = {
            DNA_VOCAB["G"]: DNA_VOCAB["A"],
            DNA_VOCAB["T"]: DNA_VOCAB["C"],
            DNA_VOCAB["A"]: DNA_VOCAB["C"],
            DNA_VOCAB["C"]: DNA_VOCAB["A"],
        }
        alt_id = replacement_by_ref.get(ref_id, DNA_VOCAB["A"])
        if alt_id == ref_id:
            alt_id = DNA_VOCAB["T"]
        return edit_position, alt_id

    alt_token_ids = torch.tensor(
        [DNA_VOCAB[base] for base in SNV_BASES],
        dtype=reference_ids.dtype,
        device=reference_ids.device,
    )
    valid_alt_mask = alt_token_ids.unsqueeze(0) != reference_ids.unsqueeze(1)
    valid_mutation_mask = (
        (mutation_effect_labels != MUTATION_EFFECT_IGNORE_INDEX)
        & valid_base_mask.unsqueeze(1)
        & valid_alt_mask
    )
    synonymous_candidates = torch.nonzero(valid_mutation_mask & (mutation_effect_labels == 0), as_tuple=False)
    pathogenic_candidates = torch.nonzero(
        valid_mutation_mask
        & torch.isin(
            mutation_effect_labels,
            torch.tensor(
                list(PATHOGENIC_MUTATION_EFFECT_CLASSES),
                dtype=mutation_effect_labels.dtype,
                device=mutation_effect_labels.device,
            ),
        ),
        as_tuple=False,
    )
    if (len(synonymous_candidates) > 0 or len(pathogenic_candidates) > 0) and random.random() < 0.5:
        candidate_pool = (
            synonymous_candidates
            if len(synonymous_candidates) > 0 and random.random() < 0.5
            else pathogenic_candidates
        )
        if len(candidate_pool) == 0:
            candidate_pool = synonymous_candidates if len(synonymous_candidates) > 0 else pathogenic_candidates
        if len(candidate_pool) > 0:
            pick = candidate_pool[random.randint(0, len(candidate_pool) - 1)]
            edit_position = int(pick[0].item())
            alt_index = int(pick[1].item())
            return edit_position, DNA_VOCAB[SNV_BASES[alt_index]]

    sampling_weights = torch.clamp(phylo100.detach().float(), min=0.0)
    sampling_weights = torch.where(valid_base_mask, sampling_weights + 1e-3, torch.zeros_like(sampling_weights))
    sampled_index = _sample_multinomial_index(sampling_weights)
    if sampled_index is None:
        valid_positions = torch.nonzero(valid_base_mask, as_tuple=False).flatten()
        if len(valid_positions) == 0:
            return None
        sampled_index = int(valid_positions[random.randint(0, len(valid_positions) - 1)].item())
    ref_id = int(reference_ids[sampled_index].item())
    alt_choices = [DNA_VOCAB[base] for base in SNV_BASES if DNA_VOCAB[base] != ref_id]
    alt_id = alt_choices[random.randint(0, len(alt_choices) - 1)]
    return sampled_index, alt_id


def interval_overlap_fraction(
    window_start: int,
    window_end: int,
    interval_index: IntervalIndex,
) -> float:
    window_len = max(1, window_end - window_start)
    covered = 0
    left, right = interval_index.overlapping_bounds(window_start, window_end)
    for start, end in zip(interval_index.starts[left:right], interval_index.ends[left:right], strict=False):
        overlap_start = max(window_start, int(start))
        overlap_end = min(window_end, int(end))
        if overlap_start < overlap_end:
            covered += overlap_end - overlap_start
    return covered / window_len


class HG38SplicePhyloDataset(IterableDataset[DatasetSample]):
    """
    Retorna janelas aleatórias de hg38 com:
      - input_ids
      - phylo100
      - phylo470
      - structure_labels (splice)
      - metadados de composição da janela
    """

    def __init__(
        self,
        fasta_path: str,
        phylo100_bw_path: str,
        phylo470_bw_path: str,
        gtf_path: str,
        seq_len: int = 4096,
        chromosomes: list[str] | None = None,
        core_radius: int = 2,
        region_radius: int = 10,
        transcript_types_keep: set[str] | None = None,
        normalize_phylo: bool = True,
        phylo_clip_value: float = 10.0,
        max_n_fraction: float = 0.25,
        max_sample_attempts: int = 20,
        cds_enrichment_fraction: float = 0.0,
        splice_window_oversample_fraction: float = 0.0,
        encode_track_specs: Sequence[EncodeTrackSpec] | None = None,
        clinvar_blocklist_bed_path: str | None = None,
        curriculum_phase: CurriculumPhase = "genome_wide",
    ) -> None:
        self.fasta_path = fasta_path
        self.phylo100_bw_path = phylo100_bw_path
        self.phylo470_bw_path = phylo470_bw_path
        self.fasta: Fasta | None = None
        self.phylo100_bw: pyBigWig.pyBigWig | None = None
        self.phylo470_bw: pyBigWig.pyBigWig | None = None
        self.encode_track_specs = list(encode_track_specs or [])
        self.curriculum_phase: CurriculumPhase = curriculum_phase
        self.clinvar_blocklist = ClinVarBlocklist.from_bed(clinvar_blocklist_bed_path)

        initial_fasta = Fasta(fasta_path, as_raw=True, sequence_always_upper=True)
        initial_phylo100_bw = pyBigWig.open(phylo100_bw_path)
        initial_phylo470_bw = pyBigWig.open(phylo470_bw_path)
        try:
            self.seq_len = seq_len
            self.normalize_phylo = normalize_phylo
            self.phylo_clip_value = phylo_clip_value
            self.max_n_fraction = max_n_fraction
            self.max_sample_attempts = max(1, max_sample_attempts)
            self.cds_enrichment_fraction = max(0.0, min(1.0, cds_enrichment_fraction))
            self.splice_window_oversample_fraction = max(0.0, min(1.0, splice_window_oversample_fraction))

            if chromosomes is None:
                available_chromosomes = cast(list[str], list(initial_fasta.keys()))
                resolved_chromosomes = [chrom for chrom in available_chromosomes if chrom.startswith("chr")]
            else:
                resolved_chromosomes = list(chromosomes)
            self.chromosomes = resolved_chromosomes

            self.chrom_sizes = validate_coordinate_integrity(
                initial_fasta,
                initial_phylo100_bw,
                initial_phylo470_bw,
                self.chromosomes,
            )
        finally:
            close_maybe(initial_fasta)
            close_maybe(initial_phylo100_bw)
            close_maybe(initial_phylo470_bw)

        self.annotation_index = build_annotation_index_from_gtf(
            gtf_path=gtf_path,
            chrom_sizes=self.chrom_sizes,
            transcript_types_keep=transcript_types_keep,
            core_radius=core_radius,
            region_radius=region_radius,
        )

        chrom_lengths = np.array([self.chrom_sizes[chrom] for chrom in self.chromosomes], dtype=np.float64)
        if chrom_lengths.sum() <= 0:
            raise ValueError("Chromosome-length-weighted sampling requires positive chromosome sizes.")
        self.chrom_sampling_cdf = np.cumsum(chrom_lengths / chrom_lengths.sum())

    def set_curriculum_phase(self, phase: CurriculumPhase) -> None:
        self.curriculum_phase = phase

    def _ensure_open_handles(self) -> tuple[Fasta, pyBigWig.pyBigWig, pyBigWig.pyBigWig]:
        if self.fasta is None:
            self.fasta = Fasta(self.fasta_path, as_raw=True, sequence_always_upper=True)
        if self.phylo100_bw is None:
            self.phylo100_bw = pyBigWig.open(self.phylo100_bw_path)
        if self.phylo470_bw is None:
            self.phylo470_bw = pyBigWig.open(self.phylo470_bw_path)
        return self.fasta, self.phylo100_bw, self.phylo470_bw

    def close(self) -> None:
        close_maybe(self.fasta)
        close_maybe(self.phylo100_bw)
        close_maybe(self.phylo470_bw)
        self.fasta = None
        self.phylo100_bw = None
        self.phylo470_bw = None

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["fasta"] = None
        state["phylo100_bw"] = None
        state["phylo470_bw"] = None
        return state

    def __del__(self) -> None:
        self.close()

    def _window_is_blocklisted(self, chrom: str, start: int, end: int) -> bool:
        return self.clinvar_blocklist.overlaps(chrom, start, end)

    def _sample_chromosome(self) -> str:
        draw = random.random()
        index = int(np.searchsorted(self.chrom_sampling_cdf, draw, side="right"))
        index = min(index, len(self.chromosomes) - 1)
        return self.chromosomes[index]

    def _sample_interval_window(
        self,
        chrom: str,
        interval_index: IntervalIndex,
        *,
        flank_radius_bp: int,
    ) -> tuple[str, int, int, str, float, bool] | None:
        fasta, _phylo100_bw, _phylo470_bw = self._ensure_open_handles()

        if len(interval_index.starts) == 0:
            return None

        idx = random.randint(0, len(interval_index.starts) - 1)
        center = random.randint(
            max(0, int(interval_index.starts[idx]) - flank_radius_bp),
            min(self.chrom_sizes[chrom] - 1, int(interval_index.ends[idx]) + flank_radius_bp),
        )
        half = self.seq_len // 2
        start = max(0, min(center - half, self.chrom_sizes[chrom] - self.seq_len))
        end = start + self.seq_len
        if self._window_is_blocklisted(chrom, start, end):
            return None

        seq = cast(str, fasta[chrom][start:end])
        n_fraction = seq.count("N") / max(1, len(seq))
        if n_fraction > self.max_n_fraction:
            return None
        return chrom, start, end, seq, n_fraction, False

    def _sample_cds_window(self) -> tuple[str, int, int, str, float, bool] | None:
        """Sample a window centred on a random CDS interval. Returns None if no CDS found."""
        fasta, _phylo100_bw, _phylo470_bw = self._ensure_open_handles()

        chrom = self._sample_chromosome()
        cds_index = self.annotation_index[chrom].cds
        if len(cds_index.starts) == 0:
            return None

        idx = random.randint(0, len(cds_index.starts) - 1)
        cds_start = int(cds_index.starts[idx])
        cds_end = int(cds_index.ends[idx])
        cds_mid = (cds_start + cds_end) // 2

        chrom_len = self.chrom_sizes[chrom]
        half = self.seq_len // 2
        start = max(0, min(cds_mid - half, chrom_len - self.seq_len))
        end = start + self.seq_len
        if self._window_is_blocklisted(chrom, start, end):
            return None

        seq = cast(str, fasta[chrom][start:end])
        n_fraction = seq.count("N") / max(1, len(seq))
        if n_fraction > self.max_n_fraction:
            return None

        return chrom, start, end, seq, n_fraction, False

    def _sample_splice_window(self) -> tuple[str, int, int, str, float, bool] | None:
        chrom = self._sample_chromosome()
        annotations = self.annotation_index[chrom]
        return self._sample_interval_window(chrom, annotations.core, flank_radius_bp=2_000)

    def _sample_exon_centric_window(self) -> tuple[str, int, int, str, float, bool] | None:
        chrom = self._sample_chromosome()
        annotations = self.annotation_index[chrom]
        return self._sample_interval_window(chrom, annotations.exon, flank_radius_bp=10_000)

    def _sample_window(self) -> tuple[str, int, int, str, float, bool]:
        if self.curriculum_phase == "exon_centric":
            result = self._sample_exon_centric_window()
            if result is not None:
                return result

        if self.splice_window_oversample_fraction > 0.0 and random.random() < self.splice_window_oversample_fraction:
            result = self._sample_splice_window()
            if result is not None:
                return result

        if self.cds_enrichment_fraction > 0.0 and random.random() < self.cds_enrichment_fraction:
            result = self._sample_cds_window()
            if result is not None:
                return result

        fasta, _phylo100_bw, _phylo470_bw = self._ensure_open_handles()
        best_candidate: tuple[str, int, int, str, float] | None = None
        best_n_fraction = float("inf")

        for _attempt in range(self.max_sample_attempts * 4):
            chrom = self._sample_chromosome()
            chrom_len = self.chrom_sizes[chrom]
            start = 0 if chrom_len <= self.seq_len else random.randint(0, chrom_len - self.seq_len)
            end = start + self.seq_len
            if self._window_is_blocklisted(chrom, start, end):
                continue
            seq = cast(str, fasta[chrom][start:end])
            n_fraction = seq.count("N") / max(1, len(seq))

            if n_fraction < best_n_fraction:
                best_candidate = (chrom, start, end, seq, n_fraction)
                best_n_fraction = n_fraction

            if n_fraction <= self.max_n_fraction:
                return chrom, start, end, seq, n_fraction, False

        if best_candidate is None:
            raise RuntimeError("Failed to sample a non-blocklisted hg38 window within the attempt budget.")
        chrom, start, end, seq, n_fraction = best_candidate
        return chrom, start, end, seq, n_fraction, True

    def _read_bw(self, bw: Any, chrom: str, start: int, end: int) -> np.ndarray:
        vals = bw.values(chrom, start, end, numpy=True)
        vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        if self.normalize_phylo:
            vals = np.clip(vals, -self.phylo_clip_value, self.phylo_clip_value)
            vals = vals / self.phylo_clip_value

        return vals

    def sample(self) -> DatasetSample:
        _fasta, phylo100_bw, phylo470_bw = self._ensure_open_handles()
        chrom, start, end, seq, n_fraction, n_filter_fallback_used = self._sample_window()
        annotations = self.annotation_index[chrom]

        input_ids = torch.tensor(encode_dna(seq), dtype=torch.long)
        phylo100_np = self._read_bw(phylo100_bw, chrom, start, end)
        phylo470_np = self._read_bw(phylo470_bw, chrom, start, end)
        phylo100 = torch.tensor(phylo100_np, dtype=torch.float32)
        phylo470 = torch.tensor(phylo470_np, dtype=torch.float32)
        structure_labels_np = intervals_to_dense_labels(chrom, start, end, self.annotation_index)
        structure_labels = torch.tensor(structure_labels_np, dtype=torch.long)
        region_labels_np = intervals_to_dense_region_labels(chrom, start, end, self.annotation_index)
        region_labels = torch.tensor(region_labels_np, dtype=torch.long)
        cds_phase_np, cds_interval_indices = assign_cds_window(
            start,
            end,
            annotations.cds_detailed,
            annotations.cds_detailed_lookup,
        )
        aa_labels_np = build_aa_labels(
            start,
            end,
            annotations.cds_detailed,
            seq,
            cds_lookup=annotations.cds_detailed_lookup,
            cds_phase=cds_phase_np,
            interval_indices=cds_interval_indices,
        )
        codon_phylo_target_np = build_codon_phylo_target(phylo100_np, cds_phase_np)
        aa_labels = torch.tensor(aa_labels_np, dtype=torch.long)
        cds_phase = torch.tensor(cds_phase_np, dtype=torch.long)
        codon_phylo_target = torch.tensor(codon_phylo_target_np, dtype=torch.float32)

        splice_positive_fraction = float(np.mean(structure_labels_np != STRUCT_BACKGROUND))
        splice_core_fraction = float(np.mean(structure_labels_np == STRUCT_SPLICE_CORE))
        exon_fraction = interval_overlap_fraction(start, end, annotations.exon)
        cds_fraction = float(np.mean(region_labels_np == REGION_CDS))
        utr_fraction = float(np.mean(region_labels_np == REGION_UTR))
        intron_fraction = float(np.mean(region_labels_np == REGION_INTRON))

        return {
            "input_ids": input_ids,
            "phylo100": phylo100,
            "phylo470": phylo470,
            "structure_labels": structure_labels,
            "region_labels": region_labels,
            "aa_labels": aa_labels,
            "cds_phase": cds_phase,
            "codon_phylo_target": codon_phylo_target,
            "chrom": chrom,
            "start": start,
            "end": end,
            "n_fraction": n_fraction,
            "splice_positive_fraction": splice_positive_fraction,
            "splice_core_fraction": splice_core_fraction,
            "exon_fraction": exon_fraction,
            "cds_fraction": cds_fraction,
            "utr_fraction": utr_fraction,
            "intron_fraction": intron_fraction,
            "n_filter_fallback_used": n_filter_fallback_used,
        }

    def __iter__(self) -> Iterator[DatasetSample]:
        while True:
            yield self.sample()


def _iter_fasta_records(path: Path) -> Iterator[tuple[str, str]]:
    header: str | None = None
    chunks: list[str] = []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks).upper()
                header = line[1:]
                chunks = []
            else:
                chunks.append(line.strip())
    if header is not None:
        yield header, "".join(chunks).upper()


def _read_parquet_columns(path: Path, columns: list[str]) -> dict[str, list[Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to read ABRAOM metadata parquet files.") from exc

    table = pq.read_table(path, columns=columns)
    return table.to_pydict()


def _load_abraom_shards(split_dir: Path, *, max_shards: int | None = None) -> list[AbraomShard]:
    fasta_paths = sorted(split_dir.glob("seqs_*.fa"))
    if not fasta_paths:
        fasta_paths = sorted(split_dir.glob("seqs_*.fa.gz"))
    if max_shards is not None:
        fasta_paths = fasta_paths[: max(0, max_shards)]
    if not fasta_paths:
        raise FileNotFoundError(f"No ABRAOM FASTA shards found in {split_dir}")

    manifest_by_shard: dict[int, dict[str, Any]] = {}
    manifest_path = split_dir / "manifest.json"
    if manifest_path.is_file():
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        for shard in data.get("shards", []):
            if isinstance(shard, dict) and "shard_id" in shard:
                manifest_by_shard[int(shard["shard_id"])] = shard

    shards: list[AbraomShard] = []
    for fasta_path in fasta_paths:
        stem = fasta_path.name
        shard_id = int(stem.split("_", 1)[1].split(".", 1)[0])
        meta_path = fasta_path.with_suffix(".meta.parquet")
        if not meta_path.is_file():
            raise FileNotFoundError(f"Missing ABRAOM metadata parquet for {fasta_path}: {meta_path}")
        manifest_entry = manifest_by_shard.get(shard_id, {})
        start_id = manifest_entry.get("start_id")
        end_id = manifest_entry.get("end_id")
        shards.append(
            AbraomShard(
                shard_id=shard_id,
                fasta_path=fasta_path,
                meta_path=meta_path,
                start_id=int(start_id) if start_id is not None else None,
                end_id=int(end_id) if end_id is not None else None,
            )
        )
    return shards


def _resolve_abraom_split_dir(root: Path, dataset_arm: str, split: str) -> Path:
    candidates = (
        root / "datasets" / dataset_arm / split,
        root / dataset_arm / split,
        root / split,
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    joined = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not resolve ABRAOM split directory. Tried: {joined}")


def _distributed_worker_partition() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))

    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
        worker_id = 0
        num_workers = 1
    else:
        worker_id = int(worker_info.id)
        num_workers = int(worker_info.num_workers)

    global_worker_id = rank * num_workers + worker_id
    total_workers = max(1, world_size * num_workers)
    return global_worker_id, total_workers


class AbraomSequenceDataset(IterableDataset[DatasetSample]):
    """Stream generated ABRAOM FASTA shards and attach hg38 coordinate labels."""

    _META_COLUMNS: ClassVar[list[str]] = [
        "sequence_id",
        "chrom",
        "start",
        "end",
        "mode",
        "n_variants_in_window",
        "n_substitutions",
    ]

    def __init__(
        self,
        *,
        data_root: str,
        dataset_arm: AbraomDatasetArm,
        split: str,
        fasta_path: str,
        phylo100_bw_path: str,
        phylo470_bw_path: str,
        gtf_path: str,
        seq_len: int = 4096,
        chromosomes: list[str] | None = None,
        core_radius: int = 2,
        region_radius: int = 10,
        transcript_types_keep: set[str] | None = None,
        normalize_phylo: bool = True,
        phylo_clip_value: float = 10.0,
        max_shards: int | None = None,
        seed: int = 42,
        shuffle_shards: bool = True,
    ) -> None:
        self.data_root = Path(data_root).expanduser()
        self.dataset_arm = dataset_arm
        self.split = split
        self.seq_len = seq_len
        self.fasta_path = fasta_path
        self.phylo100_bw_path = phylo100_bw_path
        self.phylo470_bw_path = phylo470_bw_path
        self.normalize_phylo = normalize_phylo
        self.phylo_clip_value = phylo_clip_value
        self.seed = int(seed)
        self.shuffle_shards = shuffle_shards
        self.phylo100_bw: pyBigWig.pyBigWig | None = None
        self.phylo470_bw: pyBigWig.pyBigWig | None = None

        split_dir = _resolve_abraom_split_dir(self.data_root, dataset_arm, split)
        self.shards = _load_abraom_shards(split_dir, max_shards=max_shards)

        initial_fasta = Fasta(fasta_path, as_raw=True, sequence_always_upper=True)
        initial_phylo100_bw = pyBigWig.open(phylo100_bw_path)
        initial_phylo470_bw = pyBigWig.open(phylo470_bw_path)
        try:
            if chromosomes is None:
                shard_chromosomes = self._read_manifest_chromosomes()
                resolved_chromosomes = shard_chromosomes or [
                    chrom for chrom in cast(list[str], list(initial_fasta.keys())) if chrom.startswith("chr")
                ]
            else:
                resolved_chromosomes = list(chromosomes)
            self.chromosomes = resolved_chromosomes
            self.chromosome_filter = set(resolved_chromosomes)
            self.chrom_sizes = validate_coordinate_integrity(
                initial_fasta,
                initial_phylo100_bw,
                initial_phylo470_bw,
                self.chromosomes,
            )
        finally:
            close_maybe(initial_fasta)
            close_maybe(initial_phylo100_bw)
            close_maybe(initial_phylo470_bw)

        self.annotation_index = build_annotation_index_from_gtf(
            gtf_path=gtf_path,
            chrom_sizes=self.chrom_sizes,
            transcript_types_keep=transcript_types_keep,
            core_radius=core_radius,
            region_radius=region_radius,
        )

    def _read_manifest_chromosomes(self) -> list[str] | None:
        release_manifest = self.data_root / "release_manifest.json"
        if not release_manifest.is_file():
            return None
        with suppress(Exception):
            data = json.loads(release_manifest.read_text(encoding="utf-8"))
            index = data.get("index")
            if isinstance(index, dict):
                chromosomes = index.get("chromosomes")
                if isinstance(chromosomes, list) and all(isinstance(chrom, str) for chrom in chromosomes):
                    return list(chromosomes)
        return None

    def _ensure_open_handles(self) -> tuple[pyBigWig.pyBigWig, pyBigWig.pyBigWig]:
        if self.phylo100_bw is None:
            self.phylo100_bw = pyBigWig.open(self.phylo100_bw_path)
        if self.phylo470_bw is None:
            self.phylo470_bw = pyBigWig.open(self.phylo470_bw_path)
        return self.phylo100_bw, self.phylo470_bw

    def close(self) -> None:
        close_maybe(self.phylo100_bw)
        close_maybe(self.phylo470_bw)
        self.phylo100_bw = None
        self.phylo470_bw = None

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["phylo100_bw"] = None
        state["phylo470_bw"] = None
        return state

    def __del__(self) -> None:
        self.close()

    def _read_bw(self, bw: Any, chrom: str, start: int, end: int) -> np.ndarray:
        vals = bw.values(chrom, start, end, numpy=True)
        vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        if self.normalize_phylo:
            vals = np.clip(vals, -self.phylo_clip_value, self.phylo_clip_value)
            vals = vals / self.phylo_clip_value
        return vals

    def _sample_from_record(self, meta: dict[str, Any], seq: str) -> DatasetSample:
        chrom = str(meta["chrom"])
        start = int(meta["start"])
        end = int(meta["end"])
        if end - start != self.seq_len:
            raise ValueError(
                f"ABRAOM record length mismatch for {chrom}:{start}-{end}: "
                f"metadata length={end - start}, configured seq_len={self.seq_len}"
            )
        if len(seq) != self.seq_len:
            raise ValueError(
                f"ABRAOM FASTA record length mismatch for {chrom}:{start}-{end}: "
                f"sequence length={len(seq)}, configured seq_len={self.seq_len}"
            )

        phylo100_bw, phylo470_bw = self._ensure_open_handles()
        annotations = self.annotation_index[chrom]
        n_fraction = seq.count("N") / max(1, len(seq))

        input_ids = torch.tensor(encode_dna(seq), dtype=torch.long)
        phylo100_np = self._read_bw(phylo100_bw, chrom, start, end)
        phylo470_np = self._read_bw(phylo470_bw, chrom, start, end)
        phylo100 = torch.tensor(phylo100_np, dtype=torch.float32)
        phylo470 = torch.tensor(phylo470_np, dtype=torch.float32)
        structure_labels_np = intervals_to_dense_labels(chrom, start, end, self.annotation_index)
        structure_labels = torch.tensor(structure_labels_np, dtype=torch.long)
        region_labels_np = intervals_to_dense_region_labels(chrom, start, end, self.annotation_index)
        region_labels = torch.tensor(region_labels_np, dtype=torch.long)
        cds_phase_np, cds_interval_indices = assign_cds_window(
            start,
            end,
            annotations.cds_detailed,
            annotations.cds_detailed_lookup,
        )
        aa_labels_np = build_aa_labels(
            start,
            end,
            annotations.cds_detailed,
            seq,
            cds_lookup=annotations.cds_detailed_lookup,
            cds_phase=cds_phase_np,
            interval_indices=cds_interval_indices,
        )
        codon_phylo_target_np = build_codon_phylo_target(phylo100_np, cds_phase_np)

        splice_positive_fraction = float(np.mean(structure_labels_np != STRUCT_BACKGROUND))
        splice_core_fraction = float(np.mean(structure_labels_np == STRUCT_SPLICE_CORE))
        exon_fraction = interval_overlap_fraction(start, end, annotations.exon)
        cds_fraction = float(np.mean(region_labels_np == REGION_CDS))
        utr_fraction = float(np.mean(region_labels_np == REGION_UTR))
        intron_fraction = float(np.mean(region_labels_np == REGION_INTRON))

        return {
            "input_ids": input_ids,
            "phylo100": phylo100,
            "phylo470": phylo470,
            "structure_labels": structure_labels,
            "region_labels": region_labels,
            "aa_labels": torch.tensor(aa_labels_np, dtype=torch.long),
            "cds_phase": torch.tensor(cds_phase_np, dtype=torch.long),
            "codon_phylo_target": torch.tensor(codon_phylo_target_np, dtype=torch.float32),
            "chrom": chrom,
            "start": start,
            "end": end,
            "n_fraction": n_fraction,
            "splice_positive_fraction": splice_positive_fraction,
            "splice_core_fraction": splice_core_fraction,
            "exon_fraction": exon_fraction,
            "cds_fraction": cds_fraction,
            "utr_fraction": utr_fraction,
            "intron_fraction": intron_fraction,
            "n_filter_fallback_used": False,
        }

    def _iter_shard_records(self, shard: AbraomShard) -> Iterator[tuple[int, dict[str, Any], str]]:
        meta_columns = _read_parquet_columns(shard.meta_path, self._META_COLUMNS)
        fasta_iter = _iter_fasta_records(shard.fasta_path)
        n_rows = len(meta_columns["chrom"])
        local_index = -1
        for local_index, (_header, seq) in enumerate(fasta_iter):
            if local_index >= n_rows:
                raise ValueError(f"FASTA shard {shard.fasta_path} has more records than {shard.meta_path}.")
            meta = {column: meta_columns[column][local_index] for column in self._META_COLUMNS}
            fallback_sequence_id = (shard.start_id if shard.start_id is not None else 0) + local_index
            sequence_id = int(meta.get("sequence_id", fallback_sequence_id))
            yield sequence_id, meta, seq
        if local_index + 1 != n_rows:
            raise ValueError(f"FASTA shard {shard.fasta_path} has fewer records than {shard.meta_path}.")

    def __iter__(self) -> Iterator[DatasetSample]:
        global_worker_id, total_workers = _distributed_worker_partition()
        num_shards = len(self.shards)
        use_shard_partition = num_shards >= total_workers
        epoch = 0
        while True:
            if use_shard_partition:
                shard_indices = [idx for idx in range(num_shards) if idx % total_workers == global_worker_id]
                worker_slot = 0
                workers_for_shard = 1
            else:
                shard_slot = global_worker_id % num_shards
                shard_indices = [shard_slot]
                worker_slot = global_worker_id // num_shards
                workers_for_shard = (total_workers + num_shards - 1 - shard_slot) // num_shards
            if self.shuffle_shards:
                rng = random.Random(self.seed + epoch)
                rng.shuffle(shard_indices)
            for shard_index in shard_indices:
                shard = self.shards[shard_index]
                for local_record_index, (_sequence_id, meta, seq) in enumerate(self._iter_shard_records(shard)):
                    if not use_shard_partition and local_record_index % workers_for_shard != worker_slot:
                        continue
                    if str(meta["chrom"]) not in self.chromosome_filter:
                        continue
                    yield self._sample_from_record(meta, seq)
            epoch += 1


def _build_conservation_sampling_weights(
    seq_len: int,
    unmaskable: torch.Tensor | None,
    position_weights: torch.Tensor,
    conservation_mix: float,
    region_labels: torch.Tensor | None = None,
    region_loss_ema: torch.Tensor | None = None,
    hard_position_mix: float = 0.0,
) -> torch.Tensor:
    clamped = torch.clamp(position_weights, min=0.0)
    uniform = torch.ones(seq_len, dtype=torch.float32)
    hard_position = torch.zeros(seq_len, dtype=torch.float32)
    if unmaskable is not None:
        clamped[unmaskable] = 0.0
        uniform[unmaskable] = 0.0
    if region_labels is not None and region_loss_ema is not None and hard_position_mix > 0.0:
        hard_position = region_loss_ema[region_labels.to(dtype=torch.long)].float()
        if unmaskable is not None:
            hard_position[unmaskable] = 0.0
    cons_sum = clamped.sum()
    unif_sum = uniform.sum()
    hard_sum = hard_position.sum()

    cons_norm = clamped / cons_sum if cons_sum > 0 else torch.zeros_like(clamped)
    unif_norm = uniform / (unif_sum + 1e-8) if unif_sum > 0 else torch.zeros_like(uniform)
    hard_norm = hard_position / hard_sum if hard_sum > 0 else torch.zeros_like(hard_position)

    uniform_mix = max(0.0, 1.0 - conservation_mix - hard_position_mix)
    total_mix = uniform_mix + conservation_mix + hard_position_mix
    if total_mix <= 0.0:
        return unif_norm

    blended = (
        uniform_mix * unif_norm
        + conservation_mix * cons_norm
        + hard_position_mix * hard_norm
    ) / total_mix
    blended_sum = blended.sum()
    if blended_sum <= 0:
        return unif_norm
    return blended / blended_sum


def sample_spans(
    seq_len: int,
    mask_prob: float = 0.15,
    mean_span_len: int = 8,
    unmaskable: torch.Tensor | None = None,
    position_weights: torch.Tensor | None = None,
    conservation_mix: float = 0.0,
    region_labels: torch.Tensor | None = None,
    region_loss_ema: torch.Tensor | None = None,
    hard_position_mix: float = 0.0,
) -> torch.Tensor:
    mask = torch.zeros(seq_len, dtype=torch.bool)
    num_to_mask = max(1, int(seq_len * mask_prob))

    if unmaskable is not None:
        maskable_count = int((~unmaskable).sum().item())
        num_to_mask = min(num_to_mask, maskable_count)

    use_weighted = position_weights is not None and (conservation_mix > 0.0 or hard_position_mix > 0.0)
    sampling_weights: torch.Tensor | None = None
    if use_weighted:
        assert position_weights is not None
        sampling_weights = _build_conservation_sampling_weights(
            seq_len,
            unmaskable,
            position_weights,
            conservation_mix,
            region_labels=region_labels,
            region_loss_ema=region_loss_ema,
            hard_position_mix=hard_position_mix,
        )

    masked = 0
    while masked < num_to_mask:
        if sampling_weights is not None:
            start = int(torch.multinomial(sampling_weights, num_samples=1).item())
        else:
            start = random.randint(0, seq_len - 1)
        span_len = max(1, int(random.expovariate(1.0 / mean_span_len)))
        end = min(seq_len, start + span_len)

        for i in range(start, end):
            if masked >= num_to_mask:
                break
            if unmaskable is not None and unmaskable[i]:
                continue
            if not mask[i]:
                mask[i] = True
                masked += 1

    return mask


def apply_span_masking(
    input_ids: torch.Tensor,
    mask_prob: float = 0.15,
    mean_span_len: int = 8,
    mask_token_id: int = MASK_ID,
    pad_token_id: int = PAD_ID,
    phylo_weights: torch.Tensor | None = None,
    conservation_mix: float = 0.0,
    cds_phase: torch.Tensor | None = None,
    region_labels: torch.Tensor | None = None,
    region_loss_ema: torch.Tensor | None = None,
    hard_position_mix: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    input_ids: [B, L]
    phylo_weights: [B, L] optional per-position conservation weights
    returns:
      corrupted_input_ids: [B, L]
      mlm_labels: [B, L] com PAD fora da máscara
      mask_positions: [B, L] bool
    """
    batch_size, seq_len = input_ids.shape
    corrupted = input_ids.clone()
    mlm_labels = torch.full_like(input_ids, fill_value=pad_token_id)
    mask_positions = torch.zeros_like(input_ids, dtype=torch.bool)

    for batch_index in range(batch_size):
        unmaskable = (input_ids[batch_index] == DNA_VOCAB["N"]) | (input_ids[batch_index] == pad_token_id)
        per_sample_weights = phylo_weights[batch_index] if phylo_weights is not None else None
        span_mask = sample_spans(
            seq_len,
            mask_prob=mask_prob,
            mean_span_len=mean_span_len,
            unmaskable=unmaskable,
            position_weights=per_sample_weights,
            conservation_mix=conservation_mix,
            region_labels=region_labels[batch_index] if region_labels is not None else None,
            region_loss_ema=region_loss_ema,
            hard_position_mix=hard_position_mix,
        )
        if cds_phase is not None:
            phase_values = [int(value) for value in cds_phase[batch_index].tolist()]
            initial_masked_positions = span_mask.tolist()
            masked_cds_positions = [
                position
                for position, is_masked in enumerate(initial_masked_positions)
                if is_masked and phase_values[position] >= 0
            ]
            for position in masked_cds_positions:
                codon_window = codon_window_from_phase_values(phase_values, position)
                if codon_window is None:
                    continue
                codon_start, codon_end = codon_window
                span_mask[codon_start:codon_end] = True
            span_mask[unmaskable] = False
        mask_positions[batch_index] = span_mask
        mlm_labels[batch_index, span_mask] = input_ids[batch_index, span_mask]
        corrupted[batch_index, span_mask] = mask_token_id

    return corrupted, mlm_labels, mask_positions


def _build_counterfactual_batch(
    *,
    reference_input_ids: torch.Tensor,
    corrupted_input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    mask_positions: torch.Tensor,
    phylo100: torch.Tensor,
    structure_labels: torch.Tensor,
    mutation_effect_labels: torch.Tensor,
    counterfactual_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    alt_input_ids = corrupted_input_ids.clone()
    alt_attention_mask = attention_mask.clone()
    edit_positions = torch.zeros(corrupted_input_ids.shape[0], dtype=torch.long)
    cf_active = torch.zeros(corrupted_input_ids.shape[0], dtype=torch.bool)

    for batch_index in range(corrupted_input_ids.shape[0]):
        if random.random() >= counterfactual_fraction:
            continue
        valid_edit_mask = attention_mask[batch_index].to(dtype=torch.bool) & ~mask_positions[batch_index]
        edit = _choose_counterfactual_edit(
            reference_ids=reference_input_ids[batch_index],
            phylo100=phylo100[batch_index],
            structure_labels=structure_labels[batch_index],
            mutation_effect_labels=mutation_effect_labels[batch_index],
            valid_edit_mask=valid_edit_mask,
        )
        if edit is None:
            continue
        edit_position, alt_token_id = edit
        alt_input_ids[batch_index, edit_position] = alt_token_id
        edit_positions[batch_index] = edit_position
        cf_active[batch_index] = True

    return alt_input_ids, alt_attention_mask, edit_positions, cf_active


def _choose_allele_position(
    *,
    reference_ids: torch.Tensor,
    allele_effect_labels: torch.Tensor,
    allele_severity_targets: torch.Tensor,
    valid_mask: torch.Tensor,
) -> int | None:
    valid_base_mask = _dna_base_mask(reference_ids) & valid_mask
    valid_alleles = (allele_effect_labels != ALLELE_EFFECT_IGNORE_INDEX) & valid_base_mask.unsqueeze(1)
    if not torch.any(valid_alleles):
        return None

    per_position_severity = torch.where(
        valid_alleles,
        allele_severity_targets.float().clamp_min(0.0),
        torch.zeros_like(allele_severity_targets, dtype=torch.float32),
    ).max(dim=1).values
    informative_positions = torch.nonzero(per_position_severity > 0.0, as_tuple=False).flatten()
    if len(informative_positions) > 0 and random.random() < 0.85:
        return int(informative_positions[random.randint(0, len(informative_positions) - 1)].item())

    valid_positions = torch.nonzero(valid_base_mask, as_tuple=False).flatten()
    if len(valid_positions) == 0:
        return None
    weights = per_position_severity[valid_positions] + 0.05
    sampled = _sample_multinomial_index(weights)
    if sampled is None:
        return int(valid_positions[random.randint(0, len(valid_positions) - 1)].item())
    return int(valid_positions[sampled].item())


def _build_allele_batch(
    *,
    reference_input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    mask_positions: torch.Tensor,
    allele_effect_labels: torch.Tensor,
    allele_severity_targets: torch.Tensor,
    allele_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = reference_input_ids.shape[0]
    max_alts = len(SNV_BASES) - 1
    allele_alt_input_ids = reference_input_ids.unsqueeze(1).repeat(1, max_alts, 1)
    allele_positions = torch.zeros(batch_size, dtype=torch.long)
    allele_alt_ids = torch.zeros(batch_size, max_alts, dtype=torch.long)
    selected_effect_labels = torch.full(
        (batch_size, max_alts),
        ALLELE_EFFECT_IGNORE_INDEX,
        dtype=torch.long,
    )
    selected_severity_targets = torch.zeros(batch_size, max_alts, dtype=torch.float32)
    allele_valid_mask = torch.zeros(batch_size, max_alts, dtype=torch.bool)

    for batch_index in range(batch_size):
        if random.random() >= allele_fraction:
            continue
        valid_position_mask = attention_mask[batch_index].to(dtype=torch.bool) & ~mask_positions[batch_index]
        position = _choose_allele_position(
            reference_ids=reference_input_ids[batch_index],
            allele_effect_labels=allele_effect_labels[batch_index],
            allele_severity_targets=allele_severity_targets[batch_index],
            valid_mask=valid_position_mask,
        )
        if position is None:
            continue

        ref_id = int(reference_input_ids[batch_index, position].item())
        alt_token_ids = [DNA_VOCAB[base] for base in SNV_BASES if DNA_VOCAB[base] != ref_id]
        if len(alt_token_ids) != max_alts:
            continue

        allele_positions[batch_index] = position
        for alt_slot, alt_id in enumerate(alt_token_ids):
            alt_base = SNV_TOKEN_ID_TO_BASE[alt_id]
            alt_index = SNV_ALT_TO_INDEX[alt_base]
            allele_alt_ids[batch_index, alt_slot] = alt_id
            allele_alt_input_ids[batch_index, alt_slot, position] = alt_id
            selected_effect_labels[batch_index, alt_slot] = allele_effect_labels[
                batch_index,
                position,
                alt_index,
            ]
            selected_severity_targets[batch_index, alt_slot] = allele_severity_targets[
                batch_index,
                position,
                alt_index,
            ]
            allele_valid_mask[batch_index, alt_slot] = (
                selected_effect_labels[batch_index, alt_slot] != ALLELE_EFFECT_IGNORE_INDEX
            )

    return (
        allele_alt_input_ids,
        allele_positions,
        allele_alt_ids,
        selected_effect_labels,
        selected_severity_targets,
        allele_valid_mask,
    )


@dataclass
class MultiTaskCollator:
    mask_prob: float = 0.15
    mean_span_len: int = 8
    conservation_mix: float = 0.0
    hard_position_mix: float = 0.0
    counterfactual_fraction: float = 0.0
    allele_fraction: float = 0.0
    region_loss_ema: torch.Tensor | None = None
    encode_track_names: list[str] = field(default_factory=list)

    def __call__(self, batch: list[DatasetSample]) -> MultiTaskBatch:
        input_ids = torch.stack([sample["input_ids"] for sample in batch], dim=0)
        phylo100 = torch.stack([sample["phylo100"] for sample in batch], dim=0)
        phylo470 = torch.stack([sample["phylo470"] for sample in batch], dim=0)
        structure_labels = torch.stack([sample["structure_labels"] for sample in batch], dim=0)
        region_labels = torch.stack([sample["region_labels"] for sample in batch], dim=0)
        aa_labels = torch.stack([sample["aa_labels"] for sample in batch], dim=0)
        cds_phase = torch.stack([sample["cds_phase"] for sample in batch], dim=0)
        codon_phylo_target = torch.stack([sample["codon_phylo_target"] for sample in batch], dim=0)
        effect_shape = (input_ids.shape[1], len(SNV_BASES))
        codon_labels = torch.stack(
            [
                sample.get(
                    "codon_labels",
                    torch.full((input_ids.shape[1],), CODON_IGNORE_INDEX, dtype=torch.long),
                )
                for sample in batch
            ],
            dim=0,
        )
        mutation_effect_labels = torch.stack(
            [
                sample.get(
                    "mutation_effect_labels",
                    torch.full(effect_shape, MUTATION_EFFECT_IGNORE_INDEX, dtype=torch.long),
                )
                for sample in batch
            ],
            dim=0,
        )
        allele_effect_labels = torch.stack(
            [
                sample.get(
                    "allele_effect_labels",
                    torch.full(effect_shape, ALLELE_EFFECT_IGNORE_INDEX, dtype=torch.long),
                )
                for sample in batch
            ],
            dim=0,
        )
        allele_severity_targets = torch.stack(
            [sample.get("allele_severity_targets", torch.zeros(effect_shape, dtype=torch.float32)) for sample in batch],
            dim=0,
        )
        counterfactual_effect_labels = torch.stack(
            [
                sample.get(
                    "counterfactual_effect_labels",
                    torch.full(effect_shape, COUNTERFACTUAL_EFFECT_IGNORE_INDEX, dtype=torch.long),
                )
                for sample in batch
            ],
            dim=0,
        )
        counterfactual_severity_targets = torch.stack(
            [
                sample.get(
                    "counterfactual_severity_targets",
                    torch.zeros(effect_shape, dtype=torch.float32),
                )
                for sample in batch
            ],
            dim=0,
        )
        encode_targets = torch.stack(
            [
                sample.get("encode_targets", torch.zeros((input_ids.shape[1], 0), dtype=torch.float32))
                for sample in batch
            ],
            dim=0,
        )
        aux_valid_mask = (input_ids != PAD_ID) & (input_ids != DNA_VOCAB["N"])

        cds_without_phase = (region_labels == REGION_CDS) & (cds_phase < 0)
        if torch.any(cds_without_phase):
            raise AssertionError("All CDS positions must carry a valid cds_phase in beat-v7 collation.")

        phylo_weights = phylo100 if self.conservation_mix > 0.0 else None
        corrupted_input_ids, mlm_labels, mask_positions = apply_span_masking(
            input_ids=input_ids,
            mask_prob=self.mask_prob,
            mean_span_len=self.mean_span_len,
            mask_token_id=MASK_ID,
            pad_token_id=PAD_ID,
            phylo_weights=phylo_weights,
            conservation_mix=self.conservation_mix,
            cds_phase=cds_phase,
            region_labels=region_labels,
            region_loss_ema=self.region_loss_ema,
            hard_position_mix=self.hard_position_mix,
        )

        attention_mask = (corrupted_input_ids != PAD_ID).long()
        alt_input_ids, alt_attention_mask, edit_position, cf_active = _build_counterfactual_batch(
            reference_input_ids=input_ids,
            corrupted_input_ids=corrupted_input_ids,
            attention_mask=attention_mask,
            mask_positions=mask_positions,
            phylo100=phylo100,
            structure_labels=structure_labels,
            mutation_effect_labels=mutation_effect_labels,
            counterfactual_fraction=self.counterfactual_fraction,
        )
        rc_input_ids = reverse_complement_ids(corrupted_input_ids)
        rc_attention_mask = (rc_input_ids != PAD_ID).long()
        (
            allele_alt_input_ids,
            allele_position,
            allele_alt_ids,
            selected_allele_effect_labels,
            selected_allele_severity_targets,
            allele_valid_mask,
        ) = _build_allele_batch(
            reference_input_ids=input_ids,
            attention_mask=attention_mask,
            mask_positions=mask_positions,
            allele_effect_labels=allele_effect_labels,
            allele_severity_targets=allele_severity_targets,
            allele_fraction=self.allele_fraction,
        )
        allele_locus_ids = (
            torch.arange(input_ids.shape[0], dtype=torch.long)
            .unsqueeze(1)
            .repeat(1, allele_alt_ids.shape[1])
        )
        mask_density_stats = compute_mask_density_per_region(mask_positions, region_labels)
        counterfactual_valid_mask = (
            (counterfactual_effect_labels != COUNTERFACTUAL_EFFECT_IGNORE_INDEX)
            & aux_valid_mask.unsqueeze(-1)
            & ~mask_positions.unsqueeze(-1)
        )
        conservation_bin_labels = build_conservation_bin_labels(phylo100)
        splice_distance_labels = build_splice_distance_bin_labels(structure_labels, region_labels)
        codon_pos_labels = build_codon_pos_labels(cds_phase)
        return {
            "input_ids": corrupted_input_ids,
            "alt_input_ids": alt_input_ids,
            "attention_mask": attention_mask,
            "alt_attention_mask": alt_attention_mask,
            "mlm_labels": mlm_labels,
            "mask_positions": mask_positions,
            "aux_valid_mask": aux_valid_mask,
            "phylo100": phylo100,
            "phylo470": phylo470,
            "structure_labels": structure_labels,
            "region_labels": region_labels,
            "aa_labels": aa_labels,
            "cds_phase": cds_phase,
            "codon_phylo_target": codon_phylo_target,
            "codon_labels": codon_labels,
            "mutation_effect_labels": mutation_effect_labels,
            "allele_ref_input_ids": input_ids,
            "allele_alt_input_ids": allele_alt_input_ids,
            "allele_position": allele_position,
            "allele_alt_ids": allele_alt_ids,
            "allele_effect_labels": selected_allele_effect_labels,
            "allele_severity_targets": selected_allele_severity_targets,
            "allele_valid_mask": allele_valid_mask,
            "allele_locus_ids": allele_locus_ids,
            "counterfactual_effect_labels": counterfactual_effect_labels,
            "counterfactual_severity_targets": counterfactual_severity_targets,
            "counterfactual_valid_mask": counterfactual_valid_mask,
            "conservation_bin_labels": conservation_bin_labels,
            "donor_distance_labels": splice_distance_labels,
            "acceptor_distance_labels": splice_distance_labels,
            "codon_pos_labels": codon_pos_labels,
            "exon_phase_labels": codon_pos_labels,
            "encode_targets": encode_targets,
            "edit_position": edit_position,
            "cf_active": cf_active,
            "cf_active_fraction": float(cf_active.float().mean().item()),
            "rc_input_ids": rc_input_ids,
            "rc_attention_mask": rc_attention_mask,
            "encode_track_names": list(self.encode_track_names),
            "chroms": [sample["chrom"] for sample in batch],
            "n_fraction": float(np.mean([sample["n_fraction"] for sample in batch])),
            "splice_positive_fraction": float(np.mean([sample["splice_positive_fraction"] for sample in batch])),
            "splice_core_fraction": float(np.mean([sample["splice_core_fraction"] for sample in batch])),
            "exon_fraction": float(np.mean([sample["exon_fraction"] for sample in batch])),
            "cds_fraction": float(np.mean([sample["cds_fraction"] for sample in batch])),
            "utr_fraction": float(np.mean([sample["utr_fraction"] for sample in batch])),
            "intron_fraction": float(np.mean([sample["intron_fraction"] for sample in batch])),
            "n_filter_fallback_fraction": float(
                np.mean([float(sample["n_filter_fallback_used"]) for sample in batch])
            ),
            "mask_density_intergenic": mask_density_stats["mask_density_intergenic"],
            "mask_density_intron": mask_density_stats["mask_density_intron"],
            "mask_density_noncoding_exon": mask_density_stats["mask_density_noncoding_exon"],
            "mask_density_utr": mask_density_stats["mask_density_utr"],
            "mask_density_cds": mask_density_stats["mask_density_cds"],
        }
