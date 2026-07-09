from __future__ import annotations

import numpy as np

from src.constants import (
    ALLELE_EFFECT_CONSERVED_NONCODING,
    ALLELE_EFFECT_MISSENSE_NONCONSERVATIVE,
    ALLELE_EFFECT_SPLICE_CORE,
    ALLELE_EFFECT_STOP_GAINED,
    ALLELE_EFFECT_SYNONYMOUS,
    MUTATION_EFFECT_IGNORE_INDEX,
    MUTATION_EFFECT_MISSENSE,
    MUTATION_EFFECT_STOP,
    MUTATION_EFFECT_SYNONYMOUS,
    SNV_ALT_TO_INDEX,
)
from src.dataset import CdsDetailedInterval, build_allele_effect_targets, build_mutation_effect_labels


def test_build_mutation_effect_labels_plus_strand_synonymous() -> None:
    labels = build_mutation_effect_labels(
        100,
        103,
        [CdsDetailedInterval(start=100, end=103, strand="+", frame=0)],
        "GCT",
        cds_phase=np.array([0, 1, 2], dtype=np.int64),
        interval_indices=np.array([0, 0, 0], dtype=np.int32),
    )

    assert labels[2, SNV_ALT_TO_INDEX["C"]] == MUTATION_EFFECT_SYNONYMOUS


def test_build_mutation_effect_labels_plus_strand_missense() -> None:
    labels = build_mutation_effect_labels(
        100,
        103,
        [CdsDetailedInterval(start=100, end=103, strand="+", frame=0)],
        "GCT",
        cds_phase=np.array([0, 1, 2], dtype=np.int64),
        interval_indices=np.array([0, 0, 0], dtype=np.int32),
    )

    assert labels[1, SNV_ALT_TO_INDEX["A"]] == MUTATION_EFFECT_MISSENSE


def test_build_mutation_effect_labels_minus_strand_stop() -> None:
    labels = build_mutation_effect_labels(
        100,
        103,
        [CdsDetailedInterval(start=100, end=103, strand="-", frame=0)],
        "CCA",
        cds_phase=np.array([2, 1, 0], dtype=np.int64),
        interval_indices=np.array([0, 0, 0], dtype=np.int32),
    )

    assert labels[0, SNV_ALT_TO_INDEX["T"]] == MUTATION_EFFECT_STOP


def test_build_mutation_effect_labels_ignore_non_cds_ref_slot_and_unresolved_codon() -> None:
    non_cds_labels = build_mutation_effect_labels(100, 103, [], "GCT")
    assert np.all(non_cds_labels == MUTATION_EFFECT_IGNORE_INDEX)

    plus_labels = build_mutation_effect_labels(
        100,
        103,
        [CdsDetailedInterval(start=100, end=103, strand="+", frame=0)],
        "GCT",
        cds_phase=np.array([0, 1, 2], dtype=np.int64),
        interval_indices=np.array([0, 0, 0], dtype=np.int32),
    )
    assert plus_labels[2, SNV_ALT_TO_INDEX["T"]] == MUTATION_EFFECT_IGNORE_INDEX

    unresolved_labels = build_mutation_effect_labels(
        100,
        103,
        [CdsDetailedInterval(start=100, end=103, strand="+", frame=0)],
        "GCN",
        cds_phase=np.array([0, 1, 2], dtype=np.int64),
        interval_indices=np.array([0, 0, 0], dtype=np.int32),
    )
    assert np.all(unresolved_labels == MUTATION_EFFECT_IGNORE_INDEX)


def test_build_allele_effect_targets_cds_synonymous_missense_and_stop() -> None:
    labels, severity = build_allele_effect_targets(
        100,
        103,
        [CdsDetailedInterval(start=100, end=103, strand="+", frame=0)],
        "TGG",
        structure_labels=np.zeros(3, dtype=np.int64),
        region_labels=np.full(3, 4, dtype=np.int64),
        phylo100=np.zeros(3, dtype=np.float32),
        cds_phase=np.array([0, 1, 2], dtype=np.int64),
        interval_indices=np.array([0, 0, 0], dtype=np.int32),
    )

    assert labels[2, SNV_ALT_TO_INDEX["A"]] == ALLELE_EFFECT_STOP_GAINED
    assert severity[2, SNV_ALT_TO_INDEX["A"]] == 1.0

    labels2, severity2 = build_allele_effect_targets(
        100,
        103,
        [CdsDetailedInterval(start=100, end=103, strand="+", frame=0)],
        "GCT",
        structure_labels=np.zeros(3, dtype=np.int64),
        region_labels=np.full(3, 4, dtype=np.int64),
        phylo100=np.zeros(3, dtype=np.float32),
        cds_phase=np.array([0, 1, 2], dtype=np.int64),
        interval_indices=np.array([0, 0, 0], dtype=np.int32),
    )
    assert labels2[2, SNV_ALT_TO_INDEX["C"]] == ALLELE_EFFECT_SYNONYMOUS
    assert labels2[1, SNV_ALT_TO_INDEX["A"]] == ALLELE_EFFECT_MISSENSE_NONCONSERVATIVE
    assert severity2[1, SNV_ALT_TO_INDEX["A"]] > 0.0


def test_build_allele_effect_targets_splice_and_conserved_noncoding() -> None:
    splice_labels, splice_severity = build_allele_effect_targets(
        100,
        103,
        [],
        "ACG",
        structure_labels=np.array([0, 1, 0], dtype=np.int64),
        region_labels=np.zeros(3, dtype=np.int64),
        phylo100=np.zeros(3, dtype=np.float32),
    )
    assert splice_labels[1, SNV_ALT_TO_INDEX["A"]] == ALLELE_EFFECT_SPLICE_CORE
    assert splice_severity[1, SNV_ALT_TO_INDEX["A"]] >= 0.75

    conserved_labels, conserved_severity = build_allele_effect_targets(
        100,
        103,
        [],
        "ACG",
        structure_labels=np.zeros(3, dtype=np.int64),
        region_labels=np.zeros(3, dtype=np.int64),
        phylo100=np.array([0.0, 0.8, 0.0], dtype=np.float32),
    )
    assert conserved_labels[1, SNV_ALT_TO_INDEX["A"]] == ALLELE_EFFECT_CONSERVED_NONCODING
    assert conserved_severity[1, SNV_ALT_TO_INDEX["A"]] > 0.25
