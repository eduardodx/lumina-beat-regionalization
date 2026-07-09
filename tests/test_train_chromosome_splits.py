from __future__ import annotations

import pytest

from src.constants import DEFAULT_CHROMOSOMES, DEFAULT_VAL_CHROMOSOMES
from src.train import TrainConfig, resolve_chromosome_splits


def test_resolve_chromosome_splits_full_hg38_defaults_to_full_train_and_val() -> None:
    cfg = TrainConfig(chromosome_split_mode="full_hg38")

    train_chromosomes, val_chromosomes = resolve_chromosome_splits(cfg)

    assert train_chromosomes == DEFAULT_CHROMOSOMES
    assert val_chromosomes == DEFAULT_CHROMOSOMES


def test_resolve_chromosome_splits_heldout_matches_legacy_behavior() -> None:
    cfg = TrainConfig(chromosome_split_mode="heldout")

    train_chromosomes, val_chromosomes = resolve_chromosome_splits(cfg)

    assert val_chromosomes == DEFAULT_VAL_CHROMOSOMES
    assert train_chromosomes == [chrom for chrom in DEFAULT_CHROMOSOMES if chrom not in set(DEFAULT_VAL_CHROMOSOMES)]


def test_resolve_chromosome_splits_full_hg38_rejects_one_sided_override() -> None:
    cfg = TrainConfig(
        chromosome_split_mode="full_hg38",
        train_chromosomes=["chr1", "chr2"],
    )

    with pytest.raises(ValueError, match="requires both train_chromosomes and val_chromosomes"):
        resolve_chromosome_splits(cfg)
