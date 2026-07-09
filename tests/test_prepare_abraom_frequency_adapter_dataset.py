from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.prepare_abraom_frequency_adapter_dataset import prepare_abraom_frequency_adapter_dataset


def test_prepare_abraom_frequency_adapter_dataset_assigns_splits_and_targets(tmp_path: Path) -> None:
    abraom_path = tmp_path / "abraom.parquet"
    pd.DataFrame(
        [
            {
                "variant_id": 1,
                "chrom": "chr1",
                "pos": 20,
                "ref": "A",
                "alt": "G",
                "af_abraom": 0.02,
                "af_gnomad": 0.01,
                "specificity": 0.01,
            },
            {
                "variant_id": 2,
                "chrom": "chr1",
                "pos": 125,
                "ref": "C",
                "alt": "T",
                "af_abraom": 0.20,
                "af_gnomad": 0.05,
                "specificity": 0.15,
            },
            {
                "variant_id": 3,
                "chrom": "chr1",
                "pos": 225,
                "ref": "G",
                "alt": "A",
                "af_abraom": 0.50,
                "af_gnomad": 0.00,
                "specificity": 0.50,
            },
            {
                "variant_id": 4,
                "chrom": "chr1",
                "pos": 5,
                "ref": "T",
                "alt": "C",
                "af_abraom": 0.03,
                "af_gnomad": 0.02,
                "specificity": 0.01,
            },
        ]
    ).to_parquet(abraom_path, index=False)

    manifest_path = tmp_path / "split_manifest.parquet"
    pd.DataFrame(
        [
            {
                "block_id": "chr1:0-100",
                "chrom": "chr1",
                "block_index": 0,
                "block_start": 0,
                "block_end": 100,
                "split": "train",
                "usable_start": 10,
                "usable_end": 100,
                "usable": True,
            },
            {
                "block_id": "chr1:100-200",
                "chrom": "chr1",
                "block_index": 1,
                "block_start": 100,
                "block_end": 200,
                "split": "val",
                "usable_start": 100,
                "usable_end": 200,
                "usable": True,
            },
            {
                "block_id": "chr1:200-300",
                "chrom": "chr1",
                "block_index": 2,
                "block_start": 200,
                "block_end": 300,
                "split": "test",
                "usable_start": 200,
                "usable_end": 300,
                "usable": True,
            },
        ]
    ).to_parquet(manifest_path, index=False)

    output_dir = tmp_path / "out"
    summary = prepare_abraom_frequency_adapter_dataset(
        abraom_index=abraom_path,
        split_manifest=manifest_path,
        output_dir=output_dir,
        seq_len=16,
        seed=123,
        overwrite=True,
    )

    train = pd.read_parquet(output_dir / "abraom_frequency_train.parquet")
    val = pd.read_parquet(output_dir / "abraom_frequency_val.parquet")
    test = pd.read_parquet(output_dir / "abraom_frequency_test.parquet")

    assert summary["input_rows"] == 4
    assert summary["written_rows"] == 3
    assert summary["dropped_unusable"] == 1
    assert train["variant_id"].tolist() == [1]
    assert val["variant_id"].tolist() == [2]
    assert test["variant_id"].tolist() == [3]

    row = train.iloc[0]
    assert row["variant_key"] == "chr1:21:A:G"
    assert row["Start"] == 21
    assert row["recommended_window_start"] == 12
    assert row["recommended_window_end"] == 28
    assert row["variant_context_offset"] == 8
    assert np.isclose(row["delta_af"], 0.01)
    assert np.isfinite(row["logit_af_abraom"])
    assert row["sampling_weight_raw"] >= 0.0001
