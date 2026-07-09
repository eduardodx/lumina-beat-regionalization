from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.validate_regional_signal_next_step import (
    categorize_false_benign,
    categorize_false_pathogenic,
    grouped_permutation,
    make_group_key,
)


def test_false_benign_threshold_near_miss_category() -> None:
    row = pd.Series(
        {
            "molecular_probability": 0.45,
            "score_margin": -0.01,
            "af_abraom": 0.001,
            "effective_regional_discount": 0.2,
        }
    )

    assert categorize_false_benign(row) == "threshold_near_miss"


def test_false_pathogenic_molecular_overdominant_category() -> None:
    row = pd.Series(
        {
            "molecular_probability": 0.9,
            "score_margin": 0.4,
            "af_abraom": 0.02,
            "specificity": 0.01,
            "effective_regional_discount": 0.5,
        }
    )

    assert categorize_false_pathogenic(row) == "molecular_score_overdominant"


def test_make_group_key_composes_missing_safe_values() -> None:
    frame = pd.DataFrame({"gene": ["A", "B"], "bin": ["low", None]})

    key = make_group_key(frame, ["gene", "bin", "missing_column"])

    assert key.tolist() == ["A||low||missing", "B||missing||missing"]


def test_grouped_permutation_preserves_group_values() -> None:
    values = pd.Series(np.arange(6, dtype=float))
    groups = pd.Series(["A", "A", "A", "B", "B", "B"])

    permuted, changed_fraction, informative_groups, total_groups = grouped_permutation(
        values,
        groups,
        np.random.default_rng(123),
    )

    assert changed_fraction > 0
    assert informative_groups == 2
    assert total_groups == 2
    for group in ["A", "B"]:
        assert sorted(permuted.loc[groups.eq(group)].tolist()) == sorted(values.loc[groups.eq(group)].tolist())
