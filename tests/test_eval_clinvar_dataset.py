from __future__ import annotations

from pathlib import Path

import pandas as pd

from eval.clinvar.dataset import clinvar_collate_fn, load_variant_cache


def test_load_variant_cache_round_trips_ref_and_alt_alleles(tmp_path: Path) -> None:
    cache_path = tmp_path / "variant_cache.parquet"
    pd.DataFrame(
        [
            {
                "ref_seq": "AAAA",
                "alt_seq": "CAAA",
                "ref_allele": "A",
                "alt_allele": "C",
                "variant_offset": 1,
                "label": 1,
                "original_index": 10,
                "split_within_gene": "train",
            },
            {
                "ref_seq": "CCCC",
                "alt_seq": "TCCC",
                "ref_allele": "C",
                "alt_allele": "T",
                "variant_offset": 1,
                "label": 0,
                "original_index": 11,
                "split_within_gene": "test",
            },
        ]
    ).to_parquet(cache_path, index=False)

    train_variants, test_variants = load_variant_cache(cache_path, regime="A")

    assert train_variants[0].ref_allele == "A"
    assert train_variants[0].alt_allele == "C"
    assert test_variants[0].ref_allele == "C"
    assert test_variants[0].alt_allele == "T"

    batch = clinvar_collate_fn(
        [
            {
                "ref_seq": train_variants[0].ref_seq,
                "alt_seq": train_variants[0].alt_seq,
                "ref_allele": train_variants[0].ref_allele,
                "alt_allele": train_variants[0].alt_allele,
                "variant_offset": train_variants[0].variant_offset,
                "label": train_variants[0].label,
                "dataset_index": 0,
                "original_index": train_variants[0].index,
            }
        ]
    )

    assert batch["ref_alleles"] == ["A"]
    assert batch["alt_alleles"] == ["C"]


def test_load_variant_cache_round_trips_explicit_features(tmp_path: Path) -> None:
    cache_path = tmp_path / "variant_cache.parquet"
    pd.DataFrame(
        [
            {
                "ref_seq": "AAAA",
                "alt_seq": "CAAA",
                "ref_allele": "A",
                "alt_allele": "C",
                "variant_offset": 1,
                "label": 1,
                "original_index": 10,
                "split_within_gene": "train",
                "bio_features": "[0.1, -2.0, 1.0]",
            }
        ]
    ).to_parquet(cache_path, index=False)

    train_variants, _test_variants = load_variant_cache(cache_path, regime="A")
    batch = clinvar_collate_fn(
        [
            {
                "ref_seq": train_variants[0].ref_seq,
                "alt_seq": train_variants[0].alt_seq,
                "ref_allele": train_variants[0].ref_allele,
                "alt_allele": train_variants[0].alt_allele,
                "variant_offset": train_variants[0].variant_offset,
                "label": train_variants[0].label,
                "dataset_index": 0,
                "original_index": train_variants[0].index,
                "bio_features": train_variants[0].bio_features,
            }
        ]
    )

    assert train_variants[0].bio_features == [0.1, -2.0, 1.0]
    assert batch["bio_features"].shape == (1, 3)
