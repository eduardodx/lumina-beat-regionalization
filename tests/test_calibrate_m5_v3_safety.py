from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.calibrate_m5_v3_safety import SafetyConfig, apply_safety_config, scramble_discounts


def _base_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset": ["slice", "slice"],
            "split": ["test", "test"],
            "original_index": [1, 2],
            "label": [1, 0],
            "probability": [0.1, 0.1],
            "swap_probability": [0.1, 0.1],
            "molecular_probability": [0.9, 0.4],
            "regional_discount": [4.0, 4.0],
            "GeneSymbol": ["GENE1", "GENE2"],
            "Chromosome": ["1", "2"],
            "af_abraom_bin": [">1e-2", ">1e-2"],
        }
    )


def test_m5_v3_safety_guard_prevents_strong_molecular_suppression() -> None:
    config = SafetyConfig(
        discount_scale=1.0,
        max_discount=4.0,
        molecular_guard_threshold=0.8,
        guarded_max_discount=0.0,
        guard_score_floor=0.5,
        regional_threshold=0.5,
        global_threshold=0.5,
    )

    scored = apply_safety_config(_base_predictions(), config, "br_only")

    assert bool(scored.loc[0, "safety_guard_triggered"])
    assert float(scored.loc[0, "effective_regional_discount"]) == 0.0
    assert int(scored.loc[0, "prediction_used"]) == 1


def test_m5_v3_safety_guard_does_not_promote_weak_molecular_signal() -> None:
    config = SafetyConfig(
        discount_scale=1.0,
        max_discount=4.0,
        molecular_guard_threshold=0.8,
        guarded_max_discount=0.0,
        guard_score_floor=0.5,
        regional_threshold=0.5,
        global_threshold=0.5,
    )

    scored = apply_safety_config(_base_predictions(), config, "br_only")

    assert not bool(scored.loc[1, "safety_guard_triggered"])
    assert int(scored.loc[1, "prediction_used"]) == 0


def test_stratified_scramble_is_deterministic_and_preserves_group_multisets() -> None:
    frame = pd.DataFrame(
        {
            "regional_discount": np.arange(8, dtype=float),
            "GeneSymbol": ["A"] * 4 + ["B"] * 4,
            "Chromosome": ["1"] * 8,
            "af_abraom_bin": ["low"] * 4 + ["high"] * 4,
        }
    )

    first, first_changed = scramble_discounts(frame, mode="within_gene", seed=123)
    second, second_changed = scramble_discounts(frame, mode="within_gene", seed=123)

    pd.testing.assert_series_equal(first["regional_discount"], second["regional_discount"])
    assert first_changed == second_changed
    for gene in ["A", "B"]:
        original_values = sorted(frame.loc[frame["GeneSymbol"].eq(gene), "regional_discount"].tolist())
        scrambled_values = sorted(first.loc[first["GeneSymbol"].eq(gene), "regional_discount"].tolist())
        assert scrambled_values == original_values
