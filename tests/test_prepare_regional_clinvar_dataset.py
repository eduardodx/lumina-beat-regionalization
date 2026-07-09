from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.prepare_regional_clinvar_dataset import prepare_regional_clinvar_dataset


def _write_legacy_split(path: Path, rows: list[dict]) -> None:
    columns = [
        "variant_id",
        "chrom",
        "pos",
        "ref",
        "alt",
        "label",
        "gene",
        "review_status",
        "variant_type",
        "sequence_ref",
        "sequence_alt",
    ]
    pd.DataFrame(rows, columns=columns).to_parquet(path, index=False)


def test_prepare_regional_clinvar_dataset_joins_regional_and_abraom(tmp_path: Path) -> None:
    clinvar_dir = tmp_path / "legacy_clinvar"
    clinvar_dir.mkdir()
    base_rows = [
        {
            "variant_id": "1:100:A:G",
            "chrom": "1",
            "pos": 100,
            "ref": "A",
            "alt": "G",
            "label": 1,
            "gene": "GENE1",
            "review_status": "criteria_provided",
            "variant_type": "SNV",
            "sequence_ref": "A" * 4096,
            "sequence_alt": "A" * 2048 + "G" + "A" * 2047,
        },
        {
            "variant_id": "2:200:C:T",
            "chrom": "2",
            "pos": 200,
            "ref": "C",
            "alt": "T",
            "label": 0,
            "gene": "GENE2",
            "review_status": "criteria_provided",
            "variant_type": "SNV",
            "sequence_ref": "C" * 4096,
            "sequence_alt": "C" * 2048 + "T" + "C" * 2047,
        },
    ]
    _write_legacy_split(clinvar_dir / "train.parquet", [base_rows[0]])
    _write_legacy_split(clinvar_dir / "test.parquet", [base_rows[1]])
    _write_legacy_split(clinvar_dir / "holdout.parquet", [])

    regional_path = tmp_path / "regional.parquet"
    pd.DataFrame(
        [
            {
                "VariationID": "VCV1",
                "cohort": "brazilian",
                "ClinicalSignificance": "Pathogenic",
                "Submitter": "Mendelics",
                "ReviewStatus": "criteria provided, single submitter",
                "Chromosome": "1",
                "Start": "100",
                "ReferenceAlleleVCF": "A",
                "AlternateAlleleVCF": "G",
                "SubmittedGeneSymbol": "GENE1",
                "submission_match_status": "exact",
                "variant_match_status": "exact",
                "review_status_rank_submission": 1,
                "review_status_rank_aggregate": 1,
            },
            {
                "VariationID": "VCV1",
                "cohort": "non_brazilian",
                "ClinicalSignificance": "Pathogenic",
                "Submitter": "Global Lab",
                "ReviewStatus": "criteria provided, single submitter",
                "Chromosome": "1",
                "Start": "100",
                "ReferenceAlleleVCF": "A",
                "AlternateAlleleVCF": "G",
                "SubmittedGeneSymbol": "GENE1",
                "submission_match_status": "exact",
                "variant_match_status": "exact",
                "review_status_rank_submission": 1,
                "review_status_rank_aggregate": 1,
            },
        ]
    ).to_parquet(regional_path, index=False)

    abraom_path = tmp_path / "abraom.parquet"
    pd.DataFrame(
        [
            {
                "variant_id": 7,
                "chrom": "chr1",
                "pos": 99,
                "ref": "A",
                "alt": "G",
                "af_abraom": 0.25,
                "af_gnomad": 0.10,
                "specificity": 0.15,
            }
        ]
    ).to_parquet(abraom_path, index=False)

    output_dir = tmp_path / "out"
    summary = prepare_regional_clinvar_dataset(
        lumina_clinvar_dir=clinvar_dir,
        regional_clinvar_path=regional_path,
        abraom_index=abraom_path,
        output_dir=output_dir,
        overwrite=True,
    )

    master = pd.read_parquet(output_dir / "clinvar_regional_abraom_master.parquet")
    train_row = master.loc[master["source_variant_id"] == "1:100:A:G"].iloc[0]
    test_row = master.loc[master["source_variant_id"] == "2:200:C:T"].iloc[0]

    assert train_row["clinvar_regional_cohort"] == "mixed"
    assert bool(train_row["has_brazilian_submitter"])
    assert bool(train_row["has_non_brazilian_submitter"])
    assert bool(train_row["abraom_present"])
    assert train_row["abraom_variant_id"] == 7
    assert train_row["af_abraom"] == 0.25
    assert test_row["clinvar_regional_cohort"] == "unknown"
    assert not bool(test_row["abraom_present"])

    train_test = pd.read_parquet(output_dir / "clinvar_regional_abraom_train_test.parquet")
    assert set(train_test["split_within_gene"]) == {"train", "test"}
    assert summary["rows"] == 2
    assert summary["abraom"]["matched_rows"] == 1
    assert summary["regional_cohort_counts"]["mixed"] == 1
