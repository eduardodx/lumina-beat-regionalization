from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from eval.clinvar.adapters import _extract_lumina_hidden, _resolve_lumina_hidden_dim
from scripts.train_abraom_frequency_adapter import (
    AbraomFrequencyDataset,
    FrequencyExample,
    build_metrics,
    frequency_collate_fn,
    load_frequency_examples,
)


def _write_tiny_fasta(path: Path) -> None:
    sequence = "A" * 100 + "C" + "G" * 100
    path.write_text(">chr1\n" + sequence + "\n")


def test_load_frequency_examples_respects_limit(tmp_path: Path) -> None:
    path = tmp_path / "freq.parquet"
    rows = []
    for idx in range(3):
        rows.append(
            {
                "variant_id": idx,
                "variant_key": f"chr1:{101 + idx}:C:T",
                "chrom": "chr1",
                "Start": 101 + idx,
                "ref": "C",
                "alt": "T",
                "af_abraom": 0.1,
                "af_gnomad": 0.05,
                "logit_af_gnomad": -2.9,
                "scrambled_af_abraom": 0.2,
                "af_abraom_bin": "(0.05,0.1]",
                "specificity_bin": "(0.01,0.05]",
                "gnomad_zero": False,
                "split": "train",
            }
        )
    pd.DataFrame(rows).to_parquet(path, index=False)

    examples = load_frequency_examples(path, limit=2)
    assert len(examples) == 2
    assert examples[0].variant_key == "chr1:101:C:T"


def test_load_frequency_examples_random_sample_is_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "freq.parquet"
    rows = []
    for idx in range(20):
        rows.append(
            {
                "variant_id": idx,
                "variant_key": f"chr1:{101 + idx}:C:T",
                "chrom": "chr1",
                "Start": 101 + idx,
                "ref": "C",
                "alt": "T",
                "af_abraom": 0.1,
                "af_gnomad": 0.05,
                "logit_af_gnomad": -2.9,
                "scrambled_af_abraom": 0.2,
                "af_abraom_bin": "(0.05,0.1]",
                "specificity_bin": "(0.01,0.05]",
                "gnomad_zero": False,
                "split": "train",
            }
        )
    pd.DataFrame(rows).to_parquet(path, index=False)

    first = load_frequency_examples(path, limit=5, strategy="random", seed=123)
    second = load_frequency_examples(path, limit=5, strategy="random", seed=123)
    head = load_frequency_examples(path, limit=5)

    assert [row.variant_id for row in first] == [row.variant_id for row in second]
    assert [row.variant_id for row in first] != [row.variant_id for row in head]


def test_load_frequency_examples_balanced_af_sample_uses_available_bins(tmp_path: Path) -> None:
    path = tmp_path / "freq.parquet"
    rows = []
    bins = ["[0,0.005]"] * 2 + ["(0.005,0.01]"] * 10 + ["(0.5,1]"] * 10
    for idx, af_bin in enumerate(bins):
        rows.append(
            {
                "variant_id": idx,
                "variant_key": f"chr1:{101 + idx}:C:T",
                "chrom": "chr1",
                "Start": 101 + idx,
                "ref": "C",
                "alt": "T",
                "af_abraom": 0.1,
                "af_gnomad": 0.05,
                "logit_af_gnomad": -2.9,
                "scrambled_af_abraom": 0.2,
                "af_abraom_bin": af_bin,
                "specificity_bin": "(0.01,0.05]",
                "gnomad_zero": False,
                "split": "train",
            }
        )
    pd.DataFrame(rows).to_parquet(path, index=False)

    examples = load_frequency_examples(path, limit=9, strategy="balanced-af", seed=123)
    counts: dict[str, int] = {}
    for example in examples:
        counts[example.af_abraom_bin] = counts.get(example.af_abraom_bin, 0) + 1

    assert len(examples) == 9
    assert counts == {"[0,0.005]": 2, "(0.005,0.01]": 4, "(0.5,1]": 3}


def test_frequency_dataset_extracts_ref_alt_window_and_collates(tmp_path: Path) -> None:
    fasta = tmp_path / "tiny.fa"
    _write_tiny_fasta(fasta)
    example = FrequencyExample(
        variant_id=1,
        variant_key="chr1:101:C:T",
        chrom="chr1",
        start_1based=101,
        ref="C",
        alt="T",
        af_abraom=0.2,
        af_gnomad=0.1,
        logit_af_gnomad=-2.1972246,
        scrambled_af_abraom=0.3,
        af_abraom_bin="(0.1,0.2]",
        specificity_bin="(0.05,0.1]",
        gnomad_zero=False,
        split="train",
    )
    dataset = AbraomFrequencyDataset(
        [example],
        fasta_path=fasta,
        context_size=33,
        target_column="af_abraom",
        metric_target_column="af_abraom",
        n_fraction_max=0.25,
        use_gnomad_prior=True,
    )

    item = dataset[0]
    assert item is not None
    assert item["ref_seq"][item["variant_offset"]] == "C"
    assert item["alt_seq"][item["variant_offset"]] == "T"
    assert item["target"] == 0.2
    assert item["scalar_features"] == [-2.1972246, 0.0]

    batch = frequency_collate_fn([item])
    assert batch is not None
    assert batch["targets"].shape == (1,)
    assert batch["scalar_features"].shape == (1, 2)


def test_build_metrics_reports_probabilistic_scores() -> None:
    target = np.asarray([0.1, 0.8, 0.4], dtype=np.float64)
    pred = np.asarray([0.2, 0.7, 0.5], dtype=np.float64)
    metrics = build_metrics(target, pred, prefix="model_")

    assert metrics["model_n"] == 3
    assert metrics["model_nll"] > 0
    assert np.isclose(metrics["model_brier"], np.mean((pred - target) ** 2))
    assert metrics["model_spearman"] is not None


def test_extract_lumina_hidden_accepts_beat_v10_dict_output() -> None:
    hidden = torch.randn(2, 4, 3)
    encoded = {"hidden_states": hidden, "mid_hidden_states": torch.randn(2, 2, 3)}

    assert _extract_lumina_hidden(encoded) is hidden
    assert _extract_lumina_hidden(hidden) is hidden
    with pytest.raises(RuntimeError, match="Lumina encode"):
        _extract_lumina_hidden({"mid_hidden_states": hidden})


def test_resolve_lumina_hidden_dim_prefers_full_hidden_dim() -> None:
    model = SimpleNamespace(cfg=SimpleNamespace(d_model=256), full_hidden_dim=288)
    legacy_model = SimpleNamespace(cfg=SimpleNamespace(d_model=256))

    assert _resolve_lumina_hidden_dim(model) == 288
    assert _resolve_lumina_hidden_dim(legacy_model) == 256
