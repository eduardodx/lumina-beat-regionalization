from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.calibrate_clinvar_regional_scores import (
    CalibrationConfig,
    apply_capped_frequency_discount,
    build_scrambled_frequency_table,
)


def test_capped_frequency_discount_limits_downward_movement() -> None:
    config = CalibrationConfig(
        max_down_margin=1.0,
        af_midpoint=0.01,
        af_log10_temperature=0.1,
        specificity_protect_threshold=1.0,
    )

    scored = apply_capped_frequency_discount(
        molecular_probability=np.array([0.9]),
        raw_regional_probability=np.array([0.01]),
        molecular_threshold=0.5,
        raw_regional_threshold=0.5,
        af_abraom=np.array([0.5]),
        specificity=np.array([0.0]),
        config=config,
    )

    discount = float(scored.loc[0, "capped_frequency_discount"])
    assert discount <= config.max_down_margin
    assert float(scored.loc[0, "regional_margin"]) >= float(scored.loc[0, "molecular_margin"]) - 1.000001


def test_capped_frequency_discount_never_boosts_molecular_score() -> None:
    config = CalibrationConfig(max_down_margin=4.0)

    scored = apply_capped_frequency_discount(
        molecular_probability=np.array([0.4]),
        raw_regional_probability=np.array([0.9]),
        molecular_threshold=0.5,
        raw_regional_threshold=0.5,
        af_abraom=np.array([0.5]),
        specificity=np.array([0.0]),
        config=config,
    )

    assert float(scored.loc[0, "capped_frequency_discount"]) == 0.0
    assert float(scored.loc[0, "regional_score"]) == float(scored.loc[0, "molecular_score"])


def test_build_scrambled_frequency_table_is_deterministic() -> None:
    frequency_table = pd.DataFrame(
        {
            "dataset": ["slice"] * 8,
            "original_index": list(range(8)),
            "af_abraom": np.linspace(0.01, 0.08, 8),
            "af_gnomad": np.linspace(0.02, 0.09, 8),
            "specificity": np.linspace(0.001, 0.008, 8),
        }
    )

    first = build_scrambled_frequency_table(frequency_table, seed=123)
    second = build_scrambled_frequency_table(frequency_table, seed=123)

    pd.testing.assert_frame_equal(first, second)
    assert not np.array_equal(first["scrambled_af_abraom"].to_numpy(), frequency_table["af_abraom"].to_numpy())
