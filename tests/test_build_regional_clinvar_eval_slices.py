from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.build_regional_clinvar_eval_slices import build_regional_clinvar_eval_slices


def test_build_regional_clinvar_eval_slices_counts_expected_rows(tmp_path: Path) -> None:
    input_path = tmp_path / "clinvar_regional_abraom_master.parquet"
    pd.DataFrame(
        [
            {
                "Chromosome": "1",
                "Start": 100,
                "ReferenceAlleleVCF": "A",
                "AlternateAlleleVCF": "G",
                "label": 1,
                "split_within_gene": "test",
                "source_variant_id": "v1",
                "GeneSymbol": "GENE1",
                "variant_type": "SNV",
                "is_snv": True,
                "variant_key": "1:100:A:G",
                "clinvar_regional_cohort": "brazilian",
                "has_brazilian_submitter": True,
                "has_non_brazilian_submitter": False,
                "brazilian_submission_rows": 1,
                "non_brazilian_submission_rows": 0,
                "regional_submission_rows": 1,
                "matched_regional_clinvar_benchmark": True,
                "abraom_present": True,
                "abraom_variant_id": 11,
                "af_abraom": 0.02,
                "af_gnomad": 0.01,
                "specificity": 0.01,
                "specificity_bin": "(0.005,0.01]",
            },
            {
                "Chromosome": "2",
                "Start": 200,
                "ReferenceAlleleVCF": "C",
                "AlternateAlleleVCF": "T",
                "label": 0,
                "split_within_gene": "test",
                "source_variant_id": "v2",
                "GeneSymbol": "GENE2",
                "variant_type": "SNV",
                "is_snv": True,
                "variant_key": "2:200:C:T",
                "clinvar_regional_cohort": "non_brazilian",
                "has_brazilian_submitter": False,
                "has_non_brazilian_submitter": True,
                "brazilian_submission_rows": 0,
                "non_brazilian_submission_rows": 2,
                "regional_submission_rows": 2,
                "matched_regional_clinvar_benchmark": True,
                "abraom_present": False,
                "abraom_variant_id": None,
                "af_abraom": None,
                "af_gnomad": None,
                "specificity": None,
                "specificity_bin": "absent",
            },
            {
                "Chromosome": "3",
                "Start": 300,
                "ReferenceAlleleVCF": "G",
                "AlternateAlleleVCF": "A",
                "label": 0,
                "split_within_gene": "holdout",
                "source_variant_id": "v3",
                "GeneSymbol": "GENE3",
                "variant_type": "SNV",
                "is_snv": True,
                "variant_key": "3:300:G:A",
                "clinvar_regional_cohort": "mixed",
                "has_brazilian_submitter": True,
                "has_non_brazilian_submitter": True,
                "brazilian_submission_rows": 1,
                "non_brazilian_submission_rows": 1,
                "regional_submission_rows": 2,
                "matched_regional_clinvar_benchmark": True,
                "abraom_present": True,
                "abraom_variant_id": 12,
                "af_abraom": 0.20,
                "af_gnomad": 0.02,
                "specificity": 0.18,
                "specificity_bin": "(0.1,0.5]",
            },
        ]
    ).to_parquet(input_path, index=False)

    output_dir = tmp_path / "slices"
    summary = build_regional_clinvar_eval_slices(
        input_path=input_path,
        output_dir=output_dir,
        high_specificity_threshold=0.05,
        common_af_threshold=0.01,
        overwrite=True,
    )

    manifest = pd.read_parquet(output_dir / "slice_manifest.parquet")
    counts = dict(zip(manifest["slice"], manifest["rows"], strict=True))
    assert counts["br_only"] == 1
    assert counts["nonbr_only"] == 1
    assert counts["mixed_br_nonbr"] == 1
    assert counts["br_any"] == 2
    assert counts["abraom_present"] == 2
    assert counts["abraom_high_specificity"] == 1
    assert counts["abraom_common_benign"] == 1
    assert counts["abraom_pathogenic_present"] == 1
    assert summary["source_rows"] == 3
