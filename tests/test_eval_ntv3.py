from __future__ import annotations

import contextlib
import csv
import json
import pickle
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pyBigWig
import pytest
import torch
import torch.nn as nn

from eval.ntv3.config import DEFAULT_MODEL_VERSION, NTv3BenchmarkConfig
from eval.ntv3.dataset import (
    GenomeAnnotationDataset,
    GenomeBigWigDataset,
    create_functional_targets_scaler,
    load_species_assets,
    prepare_fasta_index,
)
from eval.ntv3.heads import (
    FunctionalTracksContextPyramidHead,
    FunctionalTracksGatedHybridHead,
    FunctionalTracksLocalConvHead,
    FunctionalTracksMlpHead,
    FunctionalTracksModel,
    crop_center,
)
from eval.ntv3.run import _base_config_from_args, build_parser
from eval.ntv3.train import (
    LARGE_DATASET_SHUFFLE_THRESHOLD,
    DistributedContext,
    _build_optimizer,
    _build_scheduler,
    _build_train_sampler,
    _distributed_timeout,
    _freeze_pretraining_heads,
    _freeze_unused_pretraining_heads,
    _load_model_bundle,
    _maybe_no_sync,
    _ReplacementShuffleSampler,
    _resolve_resume_checkpoint_path,
    _resolve_runtime_batch_schedule,
    _setup_distributed,
    _suppress_head_only_warmup_grads,
    run_ntv3_benchmark,
)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_bigwig(path: Path, chrom: str, sequence_length: int, scale: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pyBigWig.open(str(path), "w") as handle:
        handle.addHeader([(chrom, sequence_length)])
        starts = list(range(sequence_length))
        ends = [start + 1 for start in starts]
        values = [scale * float(start + 1) for start in starts]
        handle.addEntries([chrom] * sequence_length, starts, ends=ends, values=values)


def _build_dataset_root(tmp_path: Path) -> Path:
    dataset_root = tmp_path / "data" / "datasets" / "ntv3"
    species_root = dataset_root / "human"
    _write_text(species_root / "genome.fasta", ">chr1\n" + "ACGT" * 16 + "\n")
    _write_text(
        species_root / "splits.bed",
        "chr1\t0\t32\ttrain\nchr1\t16\t48\tval\nchr1\t32\t64\ttest\n",
    )
    _write_text(
        dataset_root / "benchmark_metadata.tsv",
        "\n".join(
            [
                "species_common_name\tfile_id\tmean\tassay_type\ttrack_name_clean",
                "human\tTRACK_A\t2.0\tRNA-seq\tRNA track A",
                "human\tTRACK_B\t4.0\tATAC-seq\tATAC track B",
            ]
        )
        + "\n",
    )
    _write_bigwig(species_root / "functional_tracks" / "TRACK_A.bigwig", "chr1", 64, 1.0)
    _write_bigwig(species_root / "functional_tracks" / "TRACK_B.bigwig", "chr1", 64, 2.0)
    _write_text(species_root / "genome_annotation" / "exon.bed", "chr1\t4\t12\nchr1\t36\t44\n")
    _write_text(species_root / "genome_annotation" / "intron.bed", "chr1\t12\t20\nchr1\t44\t52\n")
    return dataset_root


class _FakeBackbone(nn.Module):
    def __init__(self, d_model: int = 8) -> None:
        super().__init__()
        self.embedding = nn.Embedding(8, d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_token_heads: bool = False,
    ) -> dict[str, torch.Tensor]:
        del attention_mask, return_token_heads
        hidden = self.proj(self.embedding(input_ids))
        return {"hidden_states": hidden}


class _FakeBackboneWithDecoder(_FakeBackbone):
    def __init__(self, d_model: int = 8, decoder_dim: int = 4) -> None:
        super().__init__(d_model=d_model)
        self.decoder = nn.Linear(d_model, decoder_dim)
        self.final_norm = nn.LayerNorm(d_model)

    def extract_sequence_features(self, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.final_norm(self.proj(self.embedding(input_ids)))
        return {
            "hidden_states": hidden,
            "decoder_states": self.decoder(hidden),
        }


class _FakeBackboneWithAuxHeads(_FakeBackbone):
    def __init__(self, d_model: int = 8) -> None:
        super().__init__(d_model=d_model)
        self.phylo100_head = nn.Linear(d_model, 1)
        self.phylo470_head = nn.Linear(d_model, 1)
        self.structure_head = nn.Linear(d_model, 3)
        self.codon_phylo_head = nn.Linear(d_model, 1)
        self.codon_head = nn.Linear(d_model, 64)
        self.encode_head = nn.Linear(d_model, 2)
        self.global_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_token_heads: bool = False,
    ) -> dict[str, torch.Tensor]:
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_token_heads=return_token_heads,
        )
        if not return_token_heads:
            return outputs
        hidden = outputs["hidden_states"]
        outputs["phylo100_pred"] = self.phylo100_head(hidden).squeeze(-1)
        outputs["phylo470_pred"] = self.phylo470_head(hidden).squeeze(-1)
        outputs["structure_logits"] = self.structure_head(hidden)
        return outputs


class _SizedDataset:
    def __init__(self, size: int) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size


def test_ntv3_cli_stage_dataset_defaults() -> None:
    parser = build_parser()
    args = parser.parse_args(["stage-dataset"])
    assert args.dataset_repo_id == "InstaDeepAI/NTv3_benchmark_dataset"
    assert args.species is None
    assert args.skip_functional is False
    assert args.skip_annotation is False


def test_ntv3_cli_evaluate_species_accepts_wandb_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "evaluate-species",
            "--species",
            "human",
            "--task-type",
            "functional",
            "--wandb-enabled",
            "--wandb-project",
            "lumina-ntv3",
            "--wandb-entity",
            "ai4bio-lumina",
            "--wandb-tags",
            "ntv3",
            "human",
        ]
    )
    assert args.wandb_enabled is True
    assert args.model_version == DEFAULT_MODEL_VERSION
    assert args.precision == "auto"
    assert args.wandb_project == "lumina-ntv3"
    assert args.wandb_entity == "ai4bio-lumina"
    assert args.wandb_tags == ["ntv3", "human"]


def test_ntv3_official_human_functional_preset_overrides_runtime_config(tmp_path: Path) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "evaluate-species",
            "--species",
            "human",
            "--task-type",
            "functional",
            "--official-human-functional",
            "--checkpoint-dir",
            str(tmp_path / "checkpoints"),
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--output-root",
            str(tmp_path / "outputs"),
        ]
    )
    config = _base_config_from_args(
        args,
        species_name="human",
        task_type="functional",
        output_dir=tmp_path / "outputs" / "human" / "functional",
    )

    assert config.preset_name == "official-human-functional"
    assert config.freeze_backbone is False
    assert config.batch_size == 4
    assert config.grad_accum_steps == 8
    assert config.num_steps_training == 19_932
    assert config.validate_every_n_steps == 500
    assert config.initial_learning_rate == 1e-5
    assert config.learning_rate == 5e-5
    assert config.num_steps_warmup == 598
    assert config.scheduler_name == "modified_square_decay"
    assert config.save_every_n_steps == 4_000
    assert config.max_checkpoints_to_keep == 3
    assert config.seed == 0
    assert config.precision == "fp32"
    assert config.num_workers == 16


def test_ntv3_cli_supports_custom_model_version_and_prefetch_factor(tmp_path: Path) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "evaluate-species",
            "--species",
            "human",
            "--task-type",
            "functional",
            "--model-version",
            "beat-v5",
            "--prefetch-factor",
            "8",
            "--checkpoint-dir",
            str(tmp_path / "checkpoints"),
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--output-root",
            str(tmp_path / "outputs"),
        ]
    )

    config = _base_config_from_args(
        args,
        species_name="human",
        task_type="functional",
        output_dir=tmp_path / "outputs" / "human" / "functional",
    )

    assert config.model_version == "beat-v5"
    assert config.prefetch_factor == 8


def test_ntv3_cli_maps_coupled_training_options(tmp_path: Path) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "evaluate-species",
            "--species",
            "human",
            "--task-type",
            "functional",
            "--feature-source",
            "decoder",
            "--head-learning-rate",
            "1e-4",
            "--backbone-learning-rate",
            "5e-6",
            "--decoder-learning-rate",
            "1e-5",
            "--head-only-warmup-steps",
            "1000",
            "--no-weight-decay-norm-bias",
            "--checkpoint-dir",
            str(tmp_path / "checkpoints"),
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--output-root",
            str(tmp_path / "outputs"),
        ]
    )

    config = _base_config_from_args(
        args,
        species_name="human",
        task_type="functional",
        output_dir=tmp_path / "outputs" / "human" / "functional",
    )

    assert config.feature_source == "decoder"
    assert config.head_learning_rate == 1e-4
    assert config.backbone_learning_rate == 5e-6
    assert config.decoder_learning_rate == 1e-5
    assert config.head_only_warmup_steps == 1000
    assert config.no_weight_decay_norm_bias is True


def test_ntv3_cli_maps_functional_head_options(tmp_path: Path) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "evaluate-species",
            "--species",
            "human",
            "--task-type",
            "functional",
            "--functional-head-type",
            "local-conv",
            "--functional-head-hidden-dim",
            "64",
            "--functional-head-dropout",
            "0.1",
            "--functional-head-kernel-size",
            "7",
            "--checkpoint-dir",
            str(tmp_path / "checkpoints"),
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--output-root",
            str(tmp_path / "outputs"),
        ]
    )

    config = _base_config_from_args(
        args,
        species_name="human",
        task_type="functional",
        output_dir=tmp_path / "outputs" / "human" / "functional",
    )

    assert config.functional_head_type == "local-conv"
    assert config.functional_head_hidden_dim == 64
    assert config.functional_head_dropout == pytest.approx(0.1)
    assert config.functional_head_kernel_size == 7


def test_ntv3_cli_maps_functional_aux_readout_options(tmp_path: Path) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "evaluate-species",
            "--species",
            "human",
            "--task-type",
            "functional",
            "--functional-head-type",
            "mlp",
            "--functional-head-aux-features",
            "phylo-structure",
            "--functional-head-aux-projection-dim",
            "12",
            "--checkpoint-dir",
            str(tmp_path / "checkpoints"),
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--output-root",
            str(tmp_path / "outputs"),
        ]
    )

    config = _base_config_from_args(
        args,
        species_name="human",
        task_type="functional",
        output_dir=tmp_path / "outputs" / "human" / "functional",
    )

    assert config.functional_head_type == "mlp"
    assert config.functional_head_aux_features == "phylo-structure"
    assert config.functional_head_aux_projection_dim == 12


def test_ntv3_cli_maps_gated_hybrid_head_options(tmp_path: Path) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "evaluate-species",
            "--species",
            "human",
            "--task-type",
            "functional",
            "--functional-head-type",
            "gated-hybrid",
            "--functional-head-aux-features",
            "phylo-structure",
            "--functional-head-kernel-size",
            "15",
            "--checkpoint-dir",
            str(tmp_path / "checkpoints"),
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--output-root",
            str(tmp_path / "outputs"),
        ]
    )

    config = _base_config_from_args(
        args,
        species_name="human",
        task_type="functional",
        output_dir=tmp_path / "outputs" / "human" / "functional",
    )

    assert config.functional_head_type == "gated-hybrid"
    assert config.functional_head_aux_features == "phylo-structure"
    assert config.functional_head_kernel_size == 15


def test_ntv3_official_preset_rejects_coupled_optimizer_options() -> None:
    config = NTv3BenchmarkConfig(
        preset_name="official-human-functional",
        species_name="human",
        task_type="functional",
        freeze_backbone=False,
        feature_source="decoder",
    )

    with pytest.raises(ValueError, match="feature_source='hidden'"):
        config.validate(require_paths=False)


def test_ntv3_official_preset_rejects_non_linear_functional_head() -> None:
    config = NTv3BenchmarkConfig(
        preset_name="official-human-functional",
        species_name="human",
        task_type="functional",
        freeze_backbone=False,
        functional_head_type="mlp",
    )

    with pytest.raises(ValueError, match="functional_head_type='linear'"):
        config.validate(require_paths=False)


def test_ntv3_config_rejects_even_functional_head_kernel() -> None:
    config = NTv3BenchmarkConfig(functional_head_type="local-conv", functional_head_kernel_size=8)

    with pytest.raises(ValueError, match="positive odd"):
        config.validate(require_paths=False)


def test_ntv3_config_rejects_aux_readout_without_hidden_mlp() -> None:
    decoder_config = NTv3BenchmarkConfig(
        feature_source="decoder",
        functional_head_type="mlp",
        functional_head_aux_features="phylo",
    )
    with pytest.raises(ValueError, match="feature_source='hidden'"):
        decoder_config.validate(require_paths=False)

    conv_config = NTv3BenchmarkConfig(
        functional_head_type="local-conv",
        functional_head_aux_features="structure",
    )
    with pytest.raises(ValueError, match="functional_head_type='mlp'"):
        conv_config.validate(require_paths=False)


def test_ntv3_config_allows_aux_readout_with_gated_hybrid() -> None:
    config = NTv3BenchmarkConfig(
        functional_head_type="gated-hybrid",
        functional_head_aux_features="phylo-structure",
    )

    config.validate(require_paths=False)


def test_ntv3_config_allows_aux_readout_with_context_pyramid() -> None:
    config = NTv3BenchmarkConfig(
        functional_head_type="context-pyramid",
        functional_head_aux_features="phylo-structure",
    )

    config.validate(require_paths=False)


def test_ntv3_cli_maps_periodic_checkpoint_options_into_runtime_config(tmp_path: Path) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "evaluate-species",
            "--species",
            "human",
            "--task-type",
            "functional",
            "--checkpoint-dir",
            str(tmp_path / "checkpoints"),
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--output-root",
            str(tmp_path / "outputs"),
            "--save-every-n-steps",
            "123",
            "--max-checkpoints-to-keep",
            "5",
        ]
    )

    config = _base_config_from_args(
        args,
        species_name="human",
        task_type="functional",
        output_dir=tmp_path / "outputs" / "human" / "functional",
    )

    assert config.save_every_n_steps == 123
    assert config.max_checkpoints_to_keep == 5


def test_ntv3_cli_maps_resume_options_into_runtime_config(tmp_path: Path) -> None:
    parser = build_parser()
    resume_path = tmp_path / "outputs" / "human" / "functional" / "checkpoints" / "step_000100.pt"
    args = parser.parse_args(
        [
            "evaluate-species",
            "--species",
            "human",
            "--task-type",
            "functional",
            "--checkpoint-dir",
            str(tmp_path / "checkpoints"),
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--output-root",
            str(tmp_path / "outputs"),
            "--resume-from-checkpoint",
            str(resume_path),
            "--auto-resume",
        ]
    )

    config = _base_config_from_args(
        args,
        species_name="human",
        task_type="functional",
        output_dir=tmp_path / "outputs" / "human" / "functional",
    )

    assert config.resume_from_checkpoint == str(resume_path.resolve())
    assert config.auto_resume is True


def test_load_species_assets_and_functional_dataset(tmp_path: Path) -> None:
    dataset_root = _build_dataset_root(tmp_path)
    assets = load_species_assets(dataset_root, "human")

    assert [track.dataset_id for track in assets.functional_tracks] == ["TRACK_A", "TRACK_B"]
    assert assets.functional_tracks[0].track_name_clean == "RNA track A"
    assert assets.functional_tracks[1].assay_type == "ATAC-seq"
    assert [element.dataset_id for element in assets.annotation_elements] == ["exon", "intron"]

    prepare_fasta_index(assets.fasta_path)
    dataset = GenomeBigWigDataset(
        fasta_path=assets.fasta_path,
        split_regions=assets.split_regions,
        track_infos=list(assets.functional_tracks),
        split="val",
        sequence_length=16,
        transform_fn=create_functional_targets_scaler(list(assets.functional_tracks)),
        keep_target_center_fraction=0.5,
        limit_num_samples=1,
    )
    sample = dataset[0]
    assert tuple(sample["input_ids"].shape) == (16,)
    assert tuple(sample["targets"].shape) == (8, 2)
    assert torch.all(sample["targets"] >= 0.0)


def test_functional_targets_scaler_is_picklable_for_spawn_workers(tmp_path: Path) -> None:
    dataset_root = _build_dataset_root(tmp_path)
    assets = load_species_assets(dataset_root, "human")

    scaler = create_functional_targets_scaler(list(assets.functional_tracks))
    restored = pickle.loads(pickle.dumps(scaler))
    sample_targets = torch.tensor([[2.0, 8.0], [22.0, 88.0]], dtype=torch.float32)

    assert torch.allclose(restored(sample_targets), scaler(sample_targets))


def test_functional_dataset_is_picklable_with_spawn_safe_transform(tmp_path: Path) -> None:
    dataset_root = _build_dataset_root(tmp_path)
    assets = load_species_assets(dataset_root, "human")
    prepare_fasta_index(assets.fasta_path)
    dataset = GenomeBigWigDataset(
        fasta_path=assets.fasta_path,
        split_regions=assets.split_regions,
        track_infos=list(assets.functional_tracks),
        split="train",
        sequence_length=16,
        transform_fn=create_functional_targets_scaler(list(assets.functional_tracks)),
        limit_num_samples=1,
    )

    restored = pickle.loads(pickle.dumps(dataset))
    sample = restored[0]

    assert tuple(sample["targets"].shape) == (16, 2)


def test_annotation_dataset_emits_binary_labels(tmp_path: Path) -> None:
    dataset_root = _build_dataset_root(tmp_path)
    assets = load_species_assets(dataset_root, "human")
    prepare_fasta_index(assets.fasta_path)
    dataset = GenomeAnnotationDataset(
        fasta_path=assets.fasta_path,
        split_regions=assets.split_regions,
        element_infos=list(assets.annotation_elements),
        split="test",
        sequence_length=16,
        keep_target_center_fraction=0.5,
        limit_num_samples=1,
    )
    sample = dataset[0]
    assert tuple(sample["targets"].shape) == (8, 2)
    assert set(sample["targets"].unique().tolist()).issubset({0, 1})


def test_crop_center_supports_annotation_logits_shape() -> None:
    logits = torch.zeros((1, 16, 4, 2), dtype=torch.float32)
    cropped = crop_center(logits, 0.5, sequence_axis=-3)
    assert tuple(cropped.shape) == (1, 8, 4, 2)


def test_run_ntv3_benchmark_writes_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset_root = _build_dataset_root(tmp_path)
    checkpoint_dir = tmp_path / "checkpoints" / "ntv3" / "beat-v2"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "best_checkpoint.pt").write_bytes(b"placeholder")

    fake_adapter = SimpleNamespace(backbone=_FakeBackbone(), d_model=8)
    monkeypatch.setattr("eval.ntv3.train.build_ntv3_adapter", lambda **_: fake_adapter)

    output_dir = tmp_path / "outputs" / "ntv3" / "human" / "functional"
    config = NTv3BenchmarkConfig(
        model_version="beat-v5",
        checkpoint_dir=str(checkpoint_dir),
        dataset_root=str(dataset_root),
        species_name="human",
        task_type="functional",
        sequence_length=16,
        keep_target_center_fraction=0.5,
        train_overlap=0.0,
        batch_size=1,
        grad_accum_steps=1,
        num_steps_training=2,
        validate_every_n_steps=1,
        num_validation_samples=1,
        num_test_samples=1,
        learning_rate=1e-3,
        num_workers=0,
        prefetch_factor=4,
        device="cpu",
        precision="fp32",
        save_every_n_steps=1,
        max_checkpoints_to_keep=2,
        output_dir=str(output_dir),
        overwrite=True,
    )

    summary = run_ntv3_benchmark(config)
    assert summary["species"] == "human"
    assert summary["task_type"] == "functional"
    assert Path(summary["dataset_scores_path"]).is_file()
    assert (output_dir / "best_model.pt").is_file()
    assert (output_dir / "metrics_train.csv").is_file()
    assert (output_dir / "metrics_val.csv").is_file()
    assert (output_dir / "run_config.json").is_file()
    assert (output_dir / "checkpoints" / "step_000001.pt").is_file()
    assert (output_dir / "checkpoints" / "step_000002.pt").is_file()

    dataset_rows = list(csv.DictReader((output_dir / "dataset_scores.csv").open("r", encoding="utf-8")))
    assert len(dataset_rows) == 2
    assert {row["datasets"] for row in dataset_rows} == {"TRACK_A", "TRACK_B"}
    assert {row["best_step"] for row in dataset_rows}.issubset({"1.0", "2.0"})
    assert {row["training_tokens"] for row in dataset_rows} == {"32.0"}

    saved_summary = json.loads((output_dir / "metrics_test.json").read_text(encoding="utf-8"))
    assert saved_summary["resolved_precision"] == "fp32"


def test_ntv3_auto_resume_restores_training_from_latest_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root = _build_dataset_root(tmp_path)
    checkpoint_dir = tmp_path / "checkpoints" / "ntv3" / "beat-v2"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "best_checkpoint.pt").write_bytes(b"placeholder")

    fake_adapter = SimpleNamespace(backbone=_FakeBackbone(), d_model=8)
    monkeypatch.setattr("eval.ntv3.train.build_ntv3_adapter", lambda **_: fake_adapter)

    output_dir = tmp_path / "outputs" / "ntv3" / "human" / "functional"
    base_config = NTv3BenchmarkConfig(
        model_version="beat-v5",
        checkpoint_dir=str(checkpoint_dir),
        dataset_root=str(dataset_root),
        species_name="human",
        task_type="functional",
        sequence_length=16,
        keep_target_center_fraction=0.5,
        train_overlap=0.0,
        batch_size=1,
        grad_accum_steps=1,
        validate_every_n_steps=1,
        num_validation_samples=1,
        num_test_samples=1,
        learning_rate=1e-3,
        num_workers=0,
        prefetch_factor=4,
        device="cpu",
        precision="fp32",
        log_every_n_steps=1,
        save_every_n_steps=1,
        max_checkpoints_to_keep=2,
        output_dir=str(output_dir),
        overwrite=True,
    )

    first_config = replace(base_config, num_steps_training=1)
    first_summary = run_ntv3_benchmark(first_config)
    assert first_summary["resumed_from_step"] == 0

    checkpoint_bundle = _load_model_bundle(
        output_dir / "checkpoints" / "step_000001.pt",
        map_location=torch.device("cpu"),
    )
    assert checkpoint_bundle["optimizer_state_dict"]
    assert checkpoint_bundle["scheduler_state_dict"]
    assert checkpoint_bundle["training_state"]["best_step"] == 1
    assert checkpoint_bundle["rng_state"]["torch"] is not None

    resumed_config = replace(base_config, num_steps_training=2, auto_resume=True)
    resumed_summary = run_ntv3_benchmark(resumed_config)

    assert resumed_summary["resume_from_checkpoint"] == str((output_dir / "checkpoints" / "step_000001.pt").resolve())
    assert resumed_summary["resumed_from_step"] == 1
    assert resumed_summary["best_step"] in {1, 2}

    train_rows = list(csv.DictReader((output_dir / "metrics_train.csv").open("r", encoding="utf-8")))
    assert [int(row["step"]) for row in train_rows] == [1, 2]


def test_ntv3_auto_resume_skips_corrupt_newest_checkpoint(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs" / "ntv3" / "human" / "functional"
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    valid_checkpoint = checkpoint_dir / "step_000001.pt"
    torch.save({"format": "ntv3_benchmark_v2", "step": 1, "model_state_dict": {}}, valid_checkpoint)
    (checkpoint_dir / "step_000002.pt").write_bytes(b"corrupt")

    config = NTv3BenchmarkConfig(output_dir=str(output_dir), auto_resume=True)

    resolved = _resolve_resume_checkpoint_path(config, output_dir=output_dir, map_location=torch.device("cpu"))

    assert resolved == valid_checkpoint


def test_ntv3_functional_model_can_use_decoder_feature_source() -> None:
    backbone = _FakeBackboneWithDecoder(d_model=8, decoder_dim=4)
    model = FunctionalTracksModel(
        backbone,
        embed_dim=4,
        num_tracks=2,
        keep_target_center_fraction=0.5,
        feature_source="decoder",
    )

    predictions = model(input_ids=torch.tensor([[1, 2, 3, 4, 1, 2, 3, 4]]))

    assert predictions.shape == (1, 4, 2)


def test_ntv3_functional_model_can_use_mlp_head() -> None:
    model = FunctionalTracksModel(
        _FakeBackbone(d_model=8),
        embed_dim=8,
        num_tracks=2,
        keep_target_center_fraction=0.5,
        head_type="mlp",
        head_hidden_dim=16,
        head_dropout=0.0,
    )

    predictions = model(input_ids=torch.tensor([[1, 2, 3, 4, 1, 2, 3, 4]]))

    assert isinstance(model.head, FunctionalTracksMlpHead)
    assert predictions.shape == (1, 4, 2)


def test_ntv3_functional_mlp_head_can_use_frozen_aux_readout_features() -> None:
    model = FunctionalTracksModel(
        _FakeBackboneWithAuxHeads(d_model=8),
        embed_dim=8,
        num_tracks=2,
        keep_target_center_fraction=0.5,
        head_type="mlp",
        head_hidden_dim=16,
        head_dropout=0.0,
        head_aux_features="phylo-structure",
        head_aux_projection_dim=4,
    )

    predictions = model(input_ids=torch.tensor([[1, 2, 3, 4, 1, 2, 3, 4]]))

    assert isinstance(model.head, FunctionalTracksMlpHead)
    assert model.head.aux_projector.output_dim == 8
    assert predictions.shape == (1, 4, 2)


def test_ntv3_functional_model_can_use_local_conv_head() -> None:
    model = FunctionalTracksModel(
        _FakeBackbone(d_model=8),
        embed_dim=8,
        num_tracks=2,
        keep_target_center_fraction=0.5,
        head_type="local-conv",
        head_hidden_dim=16,
        head_dropout=0.0,
        head_kernel_size=7,
    )

    predictions = model(input_ids=torch.tensor([[1, 2, 3, 4, 1, 2, 3, 4]]))

    assert isinstance(model.head, FunctionalTracksLocalConvHead)
    assert predictions.shape == (1, 4, 2)


def test_ntv3_functional_model_can_use_gated_hybrid_aux_head() -> None:
    model = FunctionalTracksModel(
        _FakeBackboneWithAuxHeads(d_model=8),
        embed_dim=8,
        num_tracks=2,
        keep_target_center_fraction=0.5,
        head_type="gated-hybrid",
        head_hidden_dim=16,
        head_dropout=0.0,
        head_kernel_size=7,
        head_aux_features="phylo-structure",
        head_aux_projection_dim=4,
    )

    predictions = model(input_ids=torch.tensor([[1, 2, 3, 4, 1, 2, 3, 4]]))

    assert isinstance(model.head, FunctionalTracksGatedHybridHead)
    assert model.head.aux_projector.output_dim == 8
    assert torch.all(torch.sigmoid(model.head.track_gate_logits) < 0.05)
    assert predictions.shape == (1, 4, 2)


def test_ntv3_functional_model_can_use_context_pyramid_aux_head() -> None:
    model = FunctionalTracksModel(
        _FakeBackboneWithAuxHeads(d_model=8),
        embed_dim=8,
        num_tracks=2,
        keep_target_center_fraction=0.5,
        head_type="context-pyramid",
        head_hidden_dim=4,
        head_dropout=0.0,
        head_kernel_size=7,
        head_aux_features="phylo-structure",
        head_aux_projection_dim=4,
    )

    predictions = model(input_ids=torch.tensor([[1, 2, 3, 4, 1, 2, 3, 4]]))

    assert isinstance(model.head, FunctionalTracksContextPyramidHead)
    assert model.head.aux_projector.output_dim == 8
    assert predictions.shape == (1, 4, 2)


def test_ntv3_coupled_optimizer_uses_group_lrs_and_head_only_warmup() -> None:
    model = FunctionalTracksModel(
        _FakeBackboneWithDecoder(d_model=8, decoder_dim=4),
        embed_dim=4,
        num_tracks=2,
        keep_target_center_fraction=0.5,
        feature_source="decoder",
    )
    config = NTv3BenchmarkConfig(
        num_steps_training=10,
        num_steps_warmup=0,
        scheduler_name="modified_square_decay",
        head_learning_rate=1e-4,
        backbone_learning_rate=5e-6,
        decoder_learning_rate=1e-5,
        head_only_warmup_steps=2,
        no_weight_decay_norm_bias=True,
    )

    optimizer = _build_optimizer(model, config=config)
    scheduler = _build_scheduler(optimizer, config=config)

    grouped_lrs = {group["ntv3_role"]: group["lr"] for group in optimizer.param_groups}
    assert grouped_lrs["head"] == pytest.approx(1e-4)
    assert grouped_lrs["decoder"] == pytest.approx(0.0)
    assert grouped_lrs["backbone"] == pytest.approx(0.0)
    assert any(group["weight_decay"] == 0.0 for group in optimizer.param_groups)

    loss = model(input_ids=torch.tensor([[1, 2, 3, 4, 1, 2, 3, 4]])).sum()
    loss.backward()
    suppressed = _suppress_head_only_warmup_grads(optimizer, step=1, config=config)
    assert suppressed > 0
    assert any(parameter.grad is not None for parameter in model.head.parameters())
    assert all(parameter.grad is None for parameter in model.backbone.parameters())

    optimizer.step()
    scheduler.step()
    grouped_lrs = {group["ntv3_role"]: group["lr"] for group in optimizer.param_groups}
    assert grouped_lrs["decoder"] == pytest.approx(0.0)
    assert grouped_lrs["backbone"] == pytest.approx(0.0)

    optimizer.step()
    scheduler.step()
    grouped_lrs = {group["ntv3_role"]: group["lr"] for group in optimizer.param_groups}
    assert grouped_lrs["decoder"] == pytest.approx(1e-5)
    assert grouped_lrs["backbone"] == pytest.approx(5e-6)


def test_runtime_batch_schedule_preserves_effective_global_batch_for_ddp() -> None:
    config = NTv3BenchmarkConfig(batch_size=4, grad_accum_steps=8)

    schedule = _resolve_runtime_batch_schedule(
        config,
        DistributedContext(distributed=True, rank=0, local_rank=0, world_size=8),
    )

    assert schedule.per_rank_batch_size == 4
    assert schedule.grad_accum_steps == 1
    assert schedule.effective_global_batch_size == 32


def test_runtime_batch_schedule_caps_per_rank_batch_for_hardware_and_preserves_global_batch() -> None:
    config = NTv3BenchmarkConfig(
        batch_size=4,
        grad_accum_steps=8,
        max_runtime_batch_size_per_rank=2,
    )

    schedule = _resolve_runtime_batch_schedule(
        config,
        DistributedContext(distributed=True, rank=0, local_rank=0, world_size=8),
    )

    assert schedule.per_rank_batch_size == 2
    assert schedule.grad_accum_steps == 2
    assert schedule.effective_global_batch_size == 32


def test_maybe_no_sync_can_be_disabled_for_static_graph_ddp(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeDDP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.no_sync_calls = 0

        def no_sync(self) -> contextlib.AbstractContextManager[None]:
            self.no_sync_calls += 1
            return contextlib.nullcontext()

    monkeypatch.setattr("eval.ntv3.train.DDP", FakeDDP)
    model = FakeDDP()
    ctx = DistributedContext(distributed=True, rank=0, local_rank=0, world_size=2)

    with _maybe_no_sync(model, should_sync=False, ctx=ctx, allow_no_sync=False):
        pass
    assert model.no_sync_calls == 0

    with _maybe_no_sync(model, should_sync=False, ctx=ctx, allow_no_sync=True):
        pass
    assert model.no_sync_calls == 1


def test_build_train_sampler_uses_memory_safe_sampler_for_huge_single_rank_dataset() -> None:
    dataset = _SizedDataset(LARGE_DATASET_SHUFFLE_THRESHOLD)

    sampler, shuffle = _build_train_sampler(
        dataset,
        ctx=DistributedContext(distributed=False),
        seed=42,
    )

    assert shuffle is False
    assert isinstance(sampler, _ReplacementShuffleSampler)
    indices = []
    iterator = iter(sampler)
    for _ in range(5):
        indices.append(next(iterator))
    assert all(0 <= index < len(dataset) for index in indices)


def test_build_train_sampler_uses_memory_safe_sampler_for_huge_distributed_dataset() -> None:
    dataset = _SizedDataset(LARGE_DATASET_SHUFFLE_THRESHOLD)

    sampler, shuffle = _build_train_sampler(
        dataset,
        ctx=DistributedContext(distributed=True, rank=1, local_rank=1, world_size=8),
        seed=42,
    )

    assert shuffle is False
    assert isinstance(sampler, _ReplacementShuffleSampler)
    assert len(sampler) == (len(dataset) + 7) // 8


def test_build_train_sampler_keeps_default_shuffle_for_small_single_rank_dataset() -> None:
    dataset = _SizedDataset(128)

    sampler, shuffle = _build_train_sampler(
        dataset,
        ctx=DistributedContext(distributed=False),
        seed=42,
    )

    assert sampler is None
    assert shuffle is True


def test_distributed_timeout_defaults_to_two_hours(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LUMINA_DDP_TIMEOUT_SECONDS", raising=False)
    assert _distributed_timeout().total_seconds() == 7200

    monkeypatch.setenv("LUMINA_DDP_TIMEOUT_SECONDS", "1800")
    assert _distributed_timeout().total_seconds() == 1800


def test_setup_distributed_skips_nccl_when_world_size_is_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")

    init_called = False
    set_device_called = False

    def _unexpected_init_process_group(*args: object, **kwargs: object) -> None:
        nonlocal init_called
        init_called = True

    def _unexpected_set_device(*args: object, **kwargs: object) -> None:
        nonlocal set_device_called
        set_device_called = True

    monkeypatch.setattr("eval.ntv3.train.dist.init_process_group", _unexpected_init_process_group)
    monkeypatch.setattr("eval.ntv3.train.torch.cuda.set_device", _unexpected_set_device)
    monkeypatch.setattr("eval.ntv3.train.torch.cuda.is_available", lambda: False)

    assert _setup_distributed() == DistributedContext()
    assert init_called is False
    assert set_device_called is False


def test_freeze_unused_pretraining_heads_removes_auxiliary_backbone_heads() -> None:
    backbone = _FakeBackboneWithAuxHeads()

    removed_names = _freeze_unused_pretraining_heads(backbone)

    assert removed_names == [
        "phylo100_head",
        "phylo470_head",
        "structure_head",
        "codon_phylo_head",
        "codon_head",
        "encode_head",
        "global_proj",
    ]
    assert not hasattr(backbone, "phylo100_head")
    assert not hasattr(backbone, "phylo470_head")
    assert not hasattr(backbone, "structure_head")
    assert not hasattr(backbone, "codon_phylo_head")
    assert not hasattr(backbone, "codon_head")
    assert not hasattr(backbone, "encode_head")
    assert not hasattr(backbone, "global_proj")
    assert all(parameter.requires_grad for parameter in backbone.embedding.parameters())
    assert all(parameter.requires_grad for parameter in backbone.proj.parameters())


def test_freeze_unused_pretraining_heads_preserves_auxiliary_readout_heads() -> None:
    backbone = _FakeBackboneWithAuxHeads()
    preserved = {"phylo100_head", "phylo470_head", "structure_head"}

    frozen_names = _freeze_pretraining_heads(backbone, preserved)
    removed_names = _freeze_unused_pretraining_heads(backbone, preserve_names=preserved)

    assert frozen_names == ["phylo100_head", "phylo470_head", "structure_head"]
    assert removed_names == ["codon_phylo_head", "codon_head", "encode_head", "global_proj"]
    assert hasattr(backbone, "phylo100_head")
    assert hasattr(backbone, "phylo470_head")
    assert hasattr(backbone, "structure_head")
    assert not hasattr(backbone, "codon_phylo_head")
    assert not hasattr(backbone, "codon_head")
    assert not hasattr(backbone, "encode_head")
    assert not hasattr(backbone, "global_proj")
    assert all(not parameter.requires_grad for parameter in backbone.phylo100_head.parameters())
    assert all(not parameter.requires_grad for parameter in backbone.phylo470_head.parameters())
    assert all(not parameter.requires_grad for parameter in backbone.structure_head.parameters())
    assert backbone.phylo100_head.training is False
    assert backbone.phylo470_head.training is False
    assert backbone.structure_head.training is False
