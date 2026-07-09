"""Data pipeline for ClinVar fine-tuning.

Pre-extracts VariantWindows and biological features from reference data,
caches them to disk, then serves tokenization-ready data via a PyTorch
Dataset during training.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pyfaidx import Fasta
from torch.utils.data import Dataset

from eval.clinvar.bio_features import BioFeatureExtractor, zero_bio_features
from eval.clinvar.variant_utils import extract_variant_window

log = logging.getLogger(__name__)

DEFAULT_DATASET = "data/datasets/clinvar/processed/clinvar_dataset.parquet"
DEFAULT_LABEL_COLUMN = "label"
DEFAULT_SPLIT_COLUMN = "split_within_gene"
TRAIN_SPLIT_VALUE = "train"
TEST_SPLIT_VALUE = "test"
EXPLICIT_FEATURE_EPS = 1e-8


# ---------------------------------------------------------------------------
# Cached variant
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CachedVariant:
    """Pre-extracted variant window ready for tokenization."""

    ref_seq: str
    alt_seq: str
    variant_offset: int
    label: int
    index: int
    ref_allele: str = ""
    alt_allele: str = ""
    bio_features: list[float] | None = None
    consequence_bucket: str = "unknown"
    gene_symbol: str = ""


def infer_consequence_bucket(ref_allele: str, alt_allele: str) -> str:
    if len(ref_allele) == 1 and len(alt_allele) == 1:
        return "snv"
    if len(ref_allele) == len(alt_allele):
        return "mnv"
    return "indel"


def _get_optional_str(row: pd.Series, candidates: list[str], default: str = "") -> str:
    for candidate in candidates:
        if candidate in row and pd.notna(row[candidate]):
            return str(row[candidate]).strip()
    return default


def _get_optional_float(row: pd.Series, key: str, default: float = 0.0) -> float:
    if key not in row or pd.isna(row[key]):
        return default
    value = row[key]
    if isinstance(value, bool | np.bool_):
        return float(value)
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(parsed) or math.isinf(parsed):
        return default
    return parsed


def _is_missing(row: pd.Series, key: str) -> bool:
    return key not in row or pd.isna(row[key])


def _log10_af(value: float) -> float:
    return float(math.log10(max(value, EXPLICIT_FEATURE_EPS)))


def _explicit_feature_value(row: pd.Series, feature_name: str) -> float:
    af_abraom = _get_optional_float(row, "af_abraom", 0.0)
    af_gnomad = _get_optional_float(row, "af_gnomad", 0.0)
    specificity = _get_optional_float(row, "specificity", 0.0)
    if feature_name == "af_abraom":
        return af_abraom
    if feature_name == "af_gnomad":
        return af_gnomad
    if feature_name == "specificity":
        return specificity
    if feature_name == "abraom_present":
        return _get_optional_float(row, "abraom_present", 0.0)
    if feature_name == "is_snv":
        return _get_optional_float(row, "is_snv", 0.0)
    if feature_name == "af_abraom_missing":
        return float(_is_missing(row, "af_abraom"))
    if feature_name == "af_gnomad_missing":
        return float(_is_missing(row, "af_gnomad"))
    if feature_name == "specificity_missing":
        return float(_is_missing(row, "specificity"))
    if feature_name == "log10_af_abraom":
        return _log10_af(af_abraom)
    if feature_name == "log10_af_gnomad":
        return _log10_af(af_gnomad)
    if feature_name == "af_delta":
        return af_abraom - af_gnomad
    if feature_name == "af_abs_delta":
        return abs(af_abraom - af_gnomad)
    if feature_name == "af_ratio_log10":
        ratio = math.log10((af_abraom + EXPLICIT_FEATURE_EPS) / (af_gnomad + EXPLICIT_FEATURE_EPS))
        return float(min(8.0, max(-8.0, ratio)))
    return _get_optional_float(row, feature_name, 0.0)


def _explicit_feature_vector(row: pd.Series, feature_columns: list[str]) -> list[float]:
    return [float(_explicit_feature_value(row, feature_name)) for feature_name in feature_columns]


# ---------------------------------------------------------------------------
# Cache building
# ---------------------------------------------------------------------------


def _cache_key(
    dataset_path: Path,
    context_size: int,
    regime: str,
    explicit_feature_columns: list[str] | None = None,
) -> str:
    feature_key = ",".join(explicit_feature_columns or [])
    h = hashlib.sha256(
        f"{dataset_path.resolve()}:{context_size}:{regime}:{feature_key}".encode()
    ).hexdigest()[:12]
    return f"variant_windows_ctx{context_size}_{regime}_{h}"


def resolve_variant_cache_path(
    dataset_path: Path,
    context_size: int,
    regime: str,
    cache_dir: Path | None = None,
    explicit_feature_columns: list[str] | None = None,
) -> Path:
    if cache_dir is None:
        cache_dir = dataset_path.parent
    return cache_dir / f"{_cache_key(dataset_path, context_size, regime, explicit_feature_columns)}.parquet"


def build_variant_cache(
    dataset_path: Path,
    fasta_path: Path,
    context_size: int,
    regime: str,
    *,
    phylo100_bw_path: str | Path | None = None,
    phylo470_bw_path: str | Path | None = None,
    gtf_path: str | Path | None = None,
    cache_dir: Path | None = None,
    label_column: str = DEFAULT_LABEL_COLUMN,
    split_column: str = DEFAULT_SPLIT_COLUMN,
    explicit_feature_columns: list[str] | None = None,
) -> Path:
    """Extract all VariantWindows (and bio features for regime B) and cache."""
    if cache_dir is None:
        cache_dir = dataset_path.parent
    cache_dir.mkdir(parents=True, exist_ok=True)
    explicit_feature_columns = list(explicit_feature_columns or [])
    if regime == "B" and explicit_feature_columns:
        raise ValueError("explicit_feature_columns are currently supported only for Regime A caches.")
    cache_path = resolve_variant_cache_path(dataset_path, context_size, regime, cache_dir, explicit_feature_columns)

    if cache_path.exists():
        log.info("Cache exists: %s", cache_path)
        return cache_path

    log.info("Building variant cache: %s", cache_path)
    df = pd.read_parquet(dataset_path)
    log.info("Loaded %d variants from %s", len(df), dataset_path)

    fasta = Fasta(str(fasta_path), as_raw=True, sequence_always_upper=True)

    bio_extractor: BioFeatureExtractor | None = None
    if regime == "B":
        bio_extractor = BioFeatureExtractor(
            phylo100_bw_path=phylo100_bw_path,
            phylo470_bw_path=phylo470_bw_path,
            gtf_path=gtf_path,
        )

    records: list[dict] = []
    skipped = 0

    for idx, row in df.iterrows():
        chrom = str(row["Chromosome"])
        pos = int(row["Start"])
        ref = str(row["ReferenceAlleleVCF"])
        alt = str(row["AlternateAlleleVCF"])

        window = extract_variant_window(
            fasta=fasta, chrom=chrom, pos=pos, ref=ref, alt=alt,
            context_size=context_size,
        )
        if window is None or window.status == "ref_mismatch":
            skipped += 1
            continue

        label_raw = row[label_column]
        label = int(label_raw) if label_raw in (0, 1) else int(float(label_raw))

        record: dict = {
            "ref_seq": window.ref_seq,
            "alt_seq": window.alt_seq,
            "ref_allele": ref,
            "alt_allele": alt,
            "variant_offset": window.variant_offset,
            "label": label,
            "original_index": int(idx),  # type: ignore[arg-type]
            split_column: str(row[split_column]).strip().lower(),
            "consequence_bucket": _get_optional_str(
                row,
                ["consequence_bucket", "ConsequenceBucket", "Consequence", "consequence", "most_severe_consequence"],
                default=infer_consequence_bucket(ref, alt),
            ),
            "gene_symbol": _get_optional_str(
                row,
                ["GeneSymbol", "gene_symbol", "Gene", "gene", "SYMBOL"],
            ),
        }

        if bio_extractor is not None:
            genomic_pos = window.window_start + window.variant_offset
            bio = bio_extractor.extract(
                ref_seq=window.ref_seq,
                variant_offset=window.variant_offset,
                ref_allele=ref,
                alt_allele=alt,
                chrom=window.chrom,
                genomic_pos=genomic_pos,
            )
            record["bio_features"] = json.dumps(bio)
        elif explicit_feature_columns:
            record["bio_features"] = json.dumps(_explicit_feature_vector(row, explicit_feature_columns))
            record["bio_feature_names"] = json.dumps(explicit_feature_columns)

        records.append(record)

        if len(records) % 10000 == 0:
            log.info("Extracted %d windows (skipped %d)", len(records), skipped)

    if bio_extractor is not None:
        bio_extractor.close()

    log.info("Cache complete: %d windows, %d skipped -> %s", len(records), skipped, cache_path)
    cache_df = pd.DataFrame(records)
    cache_df.to_parquet(cache_path, index=False)
    return cache_path


def load_variant_cache(
    cache_path: Path,
    regime: str,
    *,
    split_column: str = DEFAULT_SPLIT_COLUMN,
) -> tuple[list[CachedVariant], list[CachedVariant]]:
    """Load cached variants and split into train/test."""
    df = pd.read_parquet(cache_path)
    log.info("Loaded %d cached variants from %s", len(df), cache_path)

    has_bio = "bio_features" in df.columns
    train_variants: list[CachedVariant] = []
    test_variants: list[CachedVariant] = []

    for _, row in df.iterrows():
        bio: list[float] | None = None
        if has_bio:
            raw = row["bio_features"]
            if isinstance(raw, str):
                bio = json.loads(raw)
            elif isinstance(raw, list):
                bio = [float(v) for v in raw]
        elif regime == "B":
            bio = zero_bio_features()

        variant = CachedVariant(
            ref_seq=str(row["ref_seq"]),
            alt_seq=str(row["alt_seq"]),
            variant_offset=int(row["variant_offset"]),
            label=int(row["label"]),
            index=int(row["original_index"]),
            ref_allele=_get_optional_str(row, ["ref_allele"], default=str(row["ref_seq"])[int(row["variant_offset"])]),
            alt_allele=_get_optional_str(row, ["alt_allele"], default=str(row["alt_seq"])[int(row["variant_offset"])]),
            bio_features=bio,
            consequence_bucket=str(row["consequence_bucket"]) if "consequence_bucket" in row else "unknown",
            gene_symbol=str(row["gene_symbol"]) if "gene_symbol" in row else "",
        )
        split_val = str(row[split_column]).strip().lower()
        if split_val == TRAIN_SPLIT_VALUE:
            train_variants.append(variant)
        elif split_val == TEST_SPLIT_VALUE:
            test_variants.append(variant)

    log.info("Split: %d train, %d test", len(train_variants), len(test_variants))
    return train_variants, test_variants


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------


class ClinVarFineTuneDataset(Dataset):
    """PyTorch Dataset yielding variant data for fine-tuning."""

    def __init__(self, variants: list[CachedVariant]) -> None:
        self._variants = variants

    def __len__(self) -> int:
        return len(self._variants)

    def __getitem__(self, index: int) -> dict:
        v = self._variants[index]
        result: dict = {
            "ref_seq": v.ref_seq,
            "alt_seq": v.alt_seq,
            "variant_offset": v.variant_offset,
            "label": v.label,
            "dataset_index": index,
            "original_index": v.index,
            "ref_allele": v.ref_allele,
            "alt_allele": v.alt_allele,
        }
        if v.bio_features is not None:
            result["bio_features"] = v.bio_features
        return result

    @property
    def labels(self) -> np.ndarray:
        return np.array([v.label for v in self._variants], dtype=np.int64)

    @property
    def original_indices(self) -> np.ndarray:
        return np.array([v.index for v in self._variants], dtype=np.int64)

    def subset(self, indices: np.ndarray) -> ClinVarFineTuneDataset:
        return ClinVarFineTuneDataset([self._variants[i] for i in indices])


def stratified_val_split(
    train_variants: list[CachedVariant],
    val_fraction: float,
    seed: int,
) -> tuple[list[CachedVariant], list[CachedVariant]]:
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}.")
    if val_fraction == 0.0 or len(train_variants) < 2:
        return list(train_variants), []

    bucket_to_indices: dict[tuple[int, str], list[int]] = {}
    for index, variant in enumerate(train_variants):
        key = (int(variant.label), variant.consequence_bucket or "unknown")
        bucket_to_indices.setdefault(key, []).append(index)

    rng = np.random.default_rng(seed)
    val_indices: set[int] = set()
    for indices in bucket_to_indices.values():
        if len(indices) <= 1:
            continue
        shuffled = list(indices)
        rng.shuffle(shuffled)
        bucket_val_count = min(len(indices) - 1, max(1, round(len(indices) * val_fraction)))
        val_indices.update(shuffled[:bucket_val_count])

    if not val_indices:
        return list(train_variants), []

    train_split = [variant for idx, variant in enumerate(train_variants) if idx not in val_indices]
    val_split = [variant for idx, variant in enumerate(train_variants) if idx in val_indices]
    return train_split, val_split


def clinvar_collate_fn(batch: list[dict]) -> dict:
    """Collate variant dicts into batched format for the model."""
    def allele_or_infer(item: dict, allele_key: str, seq_key: str) -> str:
        allele = str(item.get(allele_key, "")).strip()
        if allele:
            return allele
        return str(item[seq_key])[int(item["variant_offset"])]

    result: dict = {
        "ref_seqs": [item["ref_seq"] for item in batch],
        "alt_seqs": [item["alt_seq"] for item in batch],
        "variant_offsets": [item["variant_offset"] for item in batch],
        "ref_alleles": [allele_or_infer(item, "ref_allele", "ref_seq") for item in batch],
        "alt_alleles": [allele_or_infer(item, "alt_allele", "alt_seq") for item in batch],
        "labels": torch.tensor([item["label"] for item in batch], dtype=torch.float32),
        "dataset_indices": torch.tensor([item["dataset_index"] for item in batch], dtype=torch.long),
        "original_indices": torch.tensor([item["original_index"] for item in batch], dtype=torch.long),
    }
    if "bio_features" in batch[0]:
        result["bio_features"] = torch.tensor(
            [item["bio_features"] for item in batch], dtype=torch.float32,
        )
    return result
