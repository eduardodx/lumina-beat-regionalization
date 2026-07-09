from __future__ import annotations

from typing import Any

import numpy as np

from eval.clinvar.dataset import CachedVariant
from eval.clinvar.metrics import binary_auprc, binary_roc_auc, classification_metrics


def cross_gene_diagnostic_report(
    test_variants: list[CachedVariant],
    test_preds: dict[str, np.ndarray],
    *,
    threshold: float,
    heldout_gene_fraction: float = 0.2,
    seed: int = 42,
) -> dict[str, Any]:
    genes = [variant.gene_symbol for variant in test_variants]
    unique_genes = sorted({gene for gene in genes if gene})
    if not unique_genes:
        return {
            "auroc": float("nan"),
            "auprc": float("nan"),
            "mcc_at_val_threshold": float("nan"),
            "num_variants": 0,
            "num_genes": 0,
        }

    num_heldout_genes = max(1, round(len(unique_genes) * heldout_gene_fraction))
    heldout_genes = set(
        np.random.default_rng(seed).choice(unique_genes, size=num_heldout_genes, replace=False).tolist()
    )
    mask = np.asarray(
        [variant.gene_symbol in heldout_genes for variant in test_variants],
        dtype=bool,
    )
    if mask.sum() == 0:
        return {
            "auroc": float("nan"),
            "auprc": float("nan"),
            "mcc_at_val_threshold": float("nan"),
            "num_variants": 0,
            "num_genes": len(heldout_genes),
        }

    labels = test_preds["labels"][mask]
    probs = test_preds["probs"][mask]
    threshold_metrics = classification_metrics(labels, probs, threshold)
    return {
        "auroc": binary_roc_auc(labels, probs),
        "auprc": binary_auprc(labels, probs),
        "mcc_at_val_threshold": float(threshold_metrics["mcc"]),
        "num_variants": int(mask.sum()),
        "num_genes": len(heldout_genes),
    }
