from __future__ import annotations

import pandas as pd

from scripts.build_public_abraom_validation_package import (
    classify_curated_evidence,
    classify_public_evidence,
    clinvar_search_url,
    clinvar_variation_url,
    first_variation_id,
    load_review_queue,
)


def test_first_variation_id_normalizes_numeric_strings() -> None:
    assert first_variation_id("130055.0") == "130055"
    assert first_variation_id("130055;999") == "130055"
    assert first_variation_id("") == ""


def test_classify_public_evidence_supports_plp_label() -> None:
    row = pd.Series(
        {
            "clinvar_variation_ids": "13310",
            "regional_clinical_significance_values": "Pathogenic",
            "label": 1,
        }
    )

    assert classify_public_evidence(row) == (
        "local_public_supports_label",
        "supports_plp_sentinel",
        "moderate",
    )


def test_classify_public_evidence_marks_public_label_conflict() -> None:
    row = pd.Series(
        {
            "clinvar_variation_ids": "550",
            "regional_clinical_significance_values": "Benign",
            "label": 1,
        }
    )

    assert classify_public_evidence(row) == (
        "public_label_conflict",
        "public_benign_conflicts_with_plp_label",
        "high",
    )


def test_classify_public_evidence_marks_missing_lookup() -> None:
    row = pd.Series(
        {
            "clinvar_variation_ids": "",
            "regional_clinical_significance_values": "",
            "label": 0,
        }
    )

    assert classify_public_evidence(row) == (
        "needs_public_lookup",
        "no_local_variation_id_or_significance",
        "none",
    )


def test_public_urls_are_generated_from_variant_fields() -> None:
    assert clinvar_variation_url("130055.0").endswith("/130055/")
    row = pd.Series(
        {
            "variant_key": "1:10:A:G",
            "GeneSymbol": "GENE1",
            "regional_clinical_significance_values": "Pathogenic",
        }
    )
    assert "GENE1" in clinvar_search_url(row)


def test_manual_public_curation_with_source_can_support_label() -> None:
    row = pd.Series(
        {
            "clinvar_variation_ids": "",
            "regional_clinical_significance_values": "",
            "manual_curation_status": "curated",
            "manual_public_classification": "Pathogenic",
            "manual_public_source_url_or_pmid": "PMID:123",
            "manual_evidence_note": "Reviewed against public evidence.",
            "label": 1,
        }
    )

    assert classify_curated_evidence(row) == (
        "manual_public_supports_label",
        "manual_supports_plp_sentinel",
        "manual",
    )


def test_manual_public_curation_requires_public_source() -> None:
    row = pd.Series(
        {
            "clinvar_variation_ids": "",
            "regional_clinical_significance_values": "",
            "manual_curation_status": "curated",
            "manual_public_classification": "Pathogenic",
            "manual_public_source_url_or_pmid": "",
            "manual_evidence_note": "Reviewed without a source.",
            "label": 1,
        }
    )

    assert classify_curated_evidence(row) == (
        "manual_review_incomplete",
        "manual_missing_public_source",
        "low",
    )


def test_load_review_queue_reads_tsv_and_preserves_manual_fields(tmp_path) -> None:
    path = tmp_path / "queue.tsv"
    pd.DataFrame(
        [
            {
                "priority_tier": "P1_manual_review",
                "priority_score": 1.0,
                "clinvar_variation_ids": "",
                "regional_clinical_significance_values": "",
                "variant_key": "1:10:A:G",
                "GeneSymbol": "GENE1",
                "audit_type": "false_benign_plp",
                "manual_curation_status": "curated",
                "manual_public_classification": "Pathogenic",
                "manual_public_source_url_or_pmid": "PMID:123",
                "manual_evidence_note": "Public source supports P/LP.",
                "label": 1,
            }
        ]
    ).to_csv(path, sep="\t", index=False)

    queue = load_review_queue(path, ["P1_manual_review"])

    assert queue.loc[0, "manual_public_source_url_or_pmid"] == "PMID:123"
    assert queue.loc[0, "public_review_status"] == "manual_public_supports_label"
    assert queue.loc[0, "public_evidence_decision"] == "manual_supports_plp_sentinel"
