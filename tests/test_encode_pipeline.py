from __future__ import annotations

from pathlib import Path

import pyBigWig
import pytest
import torch

from src.constants import (
    AA_NON_CDS,
    CDS_PHASE_NONE,
    CODON_IGNORE_INDEX,
    DNA_VOCAB,
    MUTATION_EFFECT_IGNORE_INDEX,
)
from src.dataset import DatasetSample, MultiTaskCollator
from src.encode_tracks import EncodeTrackSpec, load_track_values, reset_encode_track_caches
from src.models.beat_shared import require_mamba3
from src.models.beat_v7.model import BeatV7Config, DNAFoundationBeatV7
from src.objectives import valid_smooth_l1_multitrack


def _write_toy_bigwig(path: Path) -> None:
    with pyBigWig.open(str(path), "w") as handle:
        handle.addHeader([("chr1", 8)])
        handle.addEntries(
            ["chr1", "chr1"],
            [0, 4],
            ends=[2, 6],
            values=[1.0, 3.0],
        )


def _write_dense_bigwig(path: Path) -> None:
    with pyBigWig.open(str(path), "w") as handle:
        handle.addHeader([("chr1", 4)])
        handle.addEntries(
            ["chr1"] * 4,
            [0, 1, 2, 3],
            ends=[1, 2, 3, 4],
            values=[0.0, 1.0, 2.0, 3.0],
        )


def _toy_sample(seq_len: int, num_tracks: int) -> DatasetSample:
    input_ids = torch.tensor([DNA_VOCAB["A"], DNA_VOCAB["C"], DNA_VOCAB["G"], DNA_VOCAB["T"]] * (seq_len // 4))
    return {
        "input_ids": input_ids,
        "phylo100": torch.zeros(seq_len, dtype=torch.float32),
        "phylo470": torch.zeros(seq_len, dtype=torch.float32),
        "structure_labels": torch.zeros(seq_len, dtype=torch.long),
        "region_labels": torch.zeros(seq_len, dtype=torch.long),
        "aa_labels": torch.full((seq_len,), AA_NON_CDS, dtype=torch.long),
        "cds_phase": torch.full((seq_len,), CDS_PHASE_NONE, dtype=torch.long),
        "codon_phylo_target": torch.zeros(seq_len, dtype=torch.float32),
        "codon_labels": torch.full((seq_len,), CODON_IGNORE_INDEX, dtype=torch.long),
        "mutation_effect_labels": torch.full(
            (seq_len, 4),
            MUTATION_EFFECT_IGNORE_INDEX,
            dtype=torch.long,
        ),
        "encode_targets": torch.zeros(seq_len, num_tracks, dtype=torch.float32),
        "chrom": "chr1",
        "start": 0,
        "end": seq_len,
        "n_fraction": 0.0,
        "splice_positive_fraction": 0.0,
        "splice_core_fraction": 0.0,
        "exon_fraction": 0.0,
        "cds_fraction": 0.0,
        "utr_fraction": 0.0,
        "intron_fraction": 1.0,
        "n_filter_fallback_used": False,
    }


def test_load_track_values_preserves_missing_positions_as_nan(tmp_path: Path) -> None:
    bw_path = tmp_path / "toy.bw"
    _write_toy_bigwig(bw_path)
    reset_encode_track_caches()

    spec = EncodeTrackSpec(name="toy_track", bw_path=bw_path)
    values = load_track_values(spec, "chr1", 0, 8)

    assert values.shape == (8,)
    assert torch.isnan(torch.tensor(values[2:4])).all()
    assert torch.isnan(torch.tensor(values[6:8])).all()


def test_load_track_values_normalizes_on_transformed_scale(tmp_path: Path) -> None:
    bw_path = tmp_path / "dense.bw"
    _write_dense_bigwig(bw_path)
    reset_encode_track_caches()

    spec = EncodeTrackSpec(name="dense_track", bw_path=bw_path, transform="asinh")
    values = load_track_values(spec, "chr1", 0, 4)

    raw = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float32)
    transformed = torch.asinh(raw)
    expected = (transformed - transformed.mean()) / transformed.std(unbiased=False)

    assert torch.allclose(torch.tensor(values), expected, atol=1e-5)


def test_encode_targets_collate_to_batch_shape() -> None:
    collator = MultiTaskCollator(
        mask_prob=0.2,
        mean_span_len=2,
        encode_track_names=["dnase", "h3k27ac"],
    )
    batch = collator([_toy_sample(8, 2), _toy_sample(8, 2)])

    assert batch["encode_targets"].shape == (2, 8, 2)
    assert batch["encode_track_names"] == ["dnase", "h3k27ac"]
    assert "bio_features" not in batch
    assert batch["allele_locus_ids"].is_contiguous()
    assert batch["allele_locus_ids"].stride()[-1] > 0


def test_valid_smooth_l1_multitrack_returns_finite_loss_with_nan_masking() -> None:
    predictions = torch.tensor(
        [[[0.1, 0.3], [0.2, 0.4]]],
        dtype=torch.float32,
    )
    targets = torch.tensor(
        [[[0.0, float("nan")], [0.5, 0.1]]],
        dtype=torch.float32,
    )

    loss, per_track = valid_smooth_l1_multitrack(predictions, targets)

    assert torch.isfinite(loss)
    assert per_track.shape == (2,)
    assert torch.isfinite(per_track).all()


def test_beat_v7_encode_head_forward_shape_when_mamba3_available() -> None:
    try:
        require_mamba3()
    except ImportError:
        pytest.skip("Mamba3 is not available in the local macOS package")

    cfg = BeatV7Config(
        d_model=64,
        n_layers=1,
        d_state=64,
        num_region_classes=5,
        num_encode_tracks=2,
        activation_checkpointing=False,
        attention_every_n_blocks=1,
        attention_window=32,
        attention_n_heads=4,
    )
    model = DNAFoundationBeatV7(cfg)
    input_ids = torch.randint(0, len(DNA_VOCAB), (2, 16), dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    assert "encode_pred" in outputs
    assert outputs["encode_pred"].shape == (2, 16, 2)
