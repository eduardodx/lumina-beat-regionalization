from __future__ import annotations

import numpy as np

from eval.clinvar.dataset import CachedVariant
from eval.clinvar.diagnostics import cross_gene_diagnostic_report


def _variant(index: int, gene_symbol: str, label: int) -> CachedVariant:
    return CachedVariant(
        ref_seq="AAAA",
        alt_seq="CAAA" if label == 1 else "AAAA",
        variant_offset=1,
        label=label,
        index=index,
        ref_allele="A",
        alt_allele="C" if label == 1 else "A",
        gene_symbol=gene_symbol,
    )


def test_cross_gene_diagnostic_is_seeded_and_not_alphabetical_tail() -> None:
    genes = [f"GENE_{idx:02d}" for idx in range(10)]
    variants: list[CachedVariant] = []
    probs: list[float] = []
    labels: list[int] = []
    for gene_index, gene in enumerate(genes):
        negative = _variant(gene_index * 2, gene, 0)
        positive = _variant(gene_index * 2 + 1, gene, 1)
        variants.extend([negative, positive])
        if gene_index < 5:
            probs.extend([0.05, 0.95])
        else:
            probs.extend([0.95, 0.05])
        labels.extend([0, 1])

    preds = {
        "labels": np.asarray(labels, dtype=np.int64),
        "probs": np.asarray(probs, dtype=np.float32),
    }

    seed_zero = cross_gene_diagnostic_report(variants, preds, threshold=0.5, heldout_gene_fraction=0.2, seed=0)
    seed_zero_repeat = cross_gene_diagnostic_report(
        variants,
        preds,
        threshold=0.5,
        heldout_gene_fraction=0.2,
        seed=0,
    )
    seed_one = cross_gene_diagnostic_report(variants, preds, threshold=0.5, heldout_gene_fraction=0.2, seed=1)

    assert seed_zero == seed_zero_repeat
    assert seed_zero["num_genes"] == 2
    assert seed_zero != seed_one
