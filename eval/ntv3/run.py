from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download
from huggingface_hub.errors import GatedRepoError

from eval.ntv3.config import (
    DEFAULT_CHECKPOINT_S3_PREFIX,
    DEFAULT_DATASET_REPO_ID,
    DEFAULT_DATASET_ROOT,
    DEFAULT_LOCAL_CHECKPOINT_DIR,
    DEFAULT_MODEL_NAME_FOR_LEADERBOARD,
    DEFAULT_MODEL_VERSION,
    DEFAULT_OUTPUT_ROOT,
    OFFICIAL_HUMAN_FUNCTIONAL_PRESET,
    NTv3BenchmarkConfig,
    official_human_functional_overrides,
)
from eval.ntv3.dataset import discover_species, load_species_assets
from eval.ntv3.train import run_ntv3_benchmark
from src.sagemaker_utils import load_dotenv_if_available

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def _huggingface_hub_kwargs() -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    cache_dir = os.environ.get("HF_HUB_CACHE")
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if token:
        kwargs["token"] = token
    if os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in {"1", "true", "yes", "on"}:
        kwargs["local_files_only"] = True
    return kwargs


def stage_checkpoint(*, checkpoint_s3_prefix: str, checkpoint_dir: Path) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if checkpoint_s3_prefix.rstrip("/").endswith(".tar.gz"):
        log.info("Downloading SageMaker model artifact: %s -> %s", checkpoint_s3_prefix, checkpoint_dir)
        with tempfile.TemporaryDirectory() as tmp_dir:
            archive_path = Path(tmp_dir) / "model.tar.gz"
            subprocess.run(["aws", "s3", "cp", checkpoint_s3_prefix, str(archive_path)], check=True)
            with tarfile.open(archive_path, "r:gz") as archive:
                wanted = {"best_checkpoint.pt", "final_checkpoint.pt", "train_config.json", "run_summary.json"}
                for member in archive.getmembers():
                    member_name = Path(member.name).name
                    if member_name not in wanted or not member.isfile():
                        continue
                    extracted = archive.extractfile(member)
                    if extracted is None:
                        continue
                    target_path = checkpoint_dir / member_name
                    target_path.write_bytes(extracted.read())
    else:
        command = ["aws", "s3", "sync", checkpoint_s3_prefix, str(checkpoint_dir)]
        log.info("Syncing checkpoint bundle: %s -> %s", checkpoint_s3_prefix, checkpoint_dir)
        subprocess.run(command, check=True)
    checkpoint_path = checkpoint_dir / "best_checkpoint.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"best_checkpoint.pt not found after sync: {checkpoint_path}")
    return checkpoint_path


def stage_dataset(
    *,
    repo_id: str,
    dataset_root: Path,
    species_names: list[str] | None = None,
    include_functional: bool = True,
    include_annotation: bool = True,
) -> Path:
    dataset_root.mkdir(parents=True, exist_ok=True)
    allow_patterns = ["benchmark_metadata.tsv"]
    selected_species = species_names or ["*"]
    for species_name in selected_species:
        allow_patterns.extend(
            [
                f"{species_name}/genome.fasta",
                f"{species_name}/splits.bed",
            ]
        )
        if include_functional:
            allow_patterns.append(f"{species_name}/functional_tracks/*.bigwig")
        if include_annotation:
            allow_patterns.append(f"{species_name}/genome_annotation/*.bed")
    log.info("Downloading NTv3 dataset slice into %s", dataset_root)
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(dataset_root),
            allow_patterns=allow_patterns,
            **_huggingface_hub_kwargs(),
        )
    except GatedRepoError as exc:
        raise RuntimeError(
            "Access to the gated NTv3 dataset is not configured in this environment. "
            "Set `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN` for an account with approved access "
            f"to {repo_id!r}, then re-run `stage-dataset`."
        ) from exc
    return dataset_root


def _base_config_from_args(
    args: argparse.Namespace,
    *,
    species_name: str,
    task_type: str,
    output_dir: Path,
) -> NTv3BenchmarkConfig:
    default_model_name = args.model_name or (
        DEFAULT_MODEL_NAME_FOR_LEADERBOARD
        if args.model_version == DEFAULT_MODEL_VERSION
        else f"Lumina {args.model_version}"
    )
    wandb_run_name = args.wandb_run_name
    if args.wandb_enabled and not wandb_run_name:
        run_name_prefix = args.run_id or f"{default_model_name.replace(' ', '_').lower()}"
        wandb_run_name = f"{run_name_prefix}-{species_name}-{task_type}"
    config_kwargs: dict[str, Any] = {
        "model_family": "lumina",
        "model_version": args.model_version,
        "checkpoint_dir": str(Path(args.checkpoint_dir).expanduser().resolve()),
        "checkpoint_s3_prefix": args.checkpoint_s3_prefix,
        "resume_from_checkpoint": (
            str(Path(args.resume_from_checkpoint).expanduser().resolve()) if args.resume_from_checkpoint else None
        ),
        "auto_resume": args.auto_resume,
        "dataset_repo_id": args.dataset_repo_id,
        "dataset_root": str(Path(args.dataset_root).expanduser().resolve()),
        "species_name": species_name,
        "task_type": task_type,
        "sequence_length": args.sequence_length,
        "keep_target_center_fraction": args.keep_target_center_fraction,
        "train_overlap": args.train_overlap,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "max_runtime_batch_size_per_rank": args.max_runtime_batch_size_per_rank,
        "num_steps_training": args.num_steps_training,
        "validate_every_n_steps": args.validate_every_n_steps,
        "num_validation_samples": args.num_validation_samples,
        "num_test_samples": args.num_test_samples,
        "initial_learning_rate": args.initial_learning_rate,
        "learning_rate": args.learning_rate,
        "num_steps_warmup": args.num_steps_warmup,
        "scheduler_name": args.scheduler_name,
        "final_learning_rate_multiplier": args.final_learning_rate_multiplier,
        "weight_decay": args.weight_decay,
        "head_learning_rate": args.head_learning_rate,
        "backbone_learning_rate": args.backbone_learning_rate,
        "decoder_learning_rate": args.decoder_learning_rate,
        "backbone_layerwise_lr_decay": args.backbone_layerwise_lr_decay,
        "head_only_warmup_steps": args.head_only_warmup_steps,
        "no_weight_decay_norm_bias": args.no_weight_decay_norm_bias,
        "ema_decay": args.ema_decay,
        "warmup_fraction": args.warmup_fraction,
        "grad_clip": args.grad_clip,
        "log_every_n_steps": args.log_every_n_steps,
        "save_every_n_steps": args.save_every_n_steps,
        "max_checkpoints_to_keep": args.max_checkpoints_to_keep,
        "freeze_backbone": args.freeze_backbone,
        "feature_source": args.feature_source,
        "functional_head_type": args.functional_head_type,
        "functional_head_hidden_dim": args.functional_head_hidden_dim,
        "functional_head_dropout": args.functional_head_dropout,
        "functional_head_kernel_size": args.functional_head_kernel_size,
        "functional_head_aux_features": args.functional_head_aux_features,
        "functional_head_aux_projection_dim": args.functional_head_aux_projection_dim,
        "functional_head_output_bias_init": args.functional_head_output_bias_init,
        "functional_rc_consistency_weight": args.functional_rc_consistency_weight,
        "seed": args.seed,
        "device": args.device,
        "precision": args.precision,
        "allow_tf32": args.allow_tf32,
        "num_workers": args.num_workers,
        "prefetch_factor": args.prefetch_factor,
        "output_dir": str(output_dir),
        "overwrite": args.overwrite,
        "wandb_enabled": args.wandb_enabled,
        "wandb_project": args.wandb_project,
        "wandb_entity": args.wandb_entity,
        "wandb_run_name": wandb_run_name,
        "wandb_tags": args.wandb_tags or [],
        "model_name_for_leaderboard": default_model_name,
        "run_id": args.run_id,
    }
    if args.official_human_functional:
        if species_name != "human" or task_type != "functional":
            raise ValueError(
                f"{OFFICIAL_HUMAN_FUNCTIONAL_PRESET!r} can only be used with --species human --task-type functional."
            )
        config_kwargs.update(official_human_functional_overrides())
    return NTv3BenchmarkConfig(**config_kwargs)


def _collect_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_aggregate_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "Metric",
        "run_id",
        "model_name",
        "best_step",
        "best_step_time",
        "training_tokens",
        "running_time",
        "species",
        "datasets",
        "assay_type",
        "track_name_clean",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _available_task_types(dataset_root: Path, species_name: str) -> list[str]:
    assets = load_species_assets(dataset_root, species_name)
    task_types: list[str] = []
    if assets.functional_tracks:
        task_types.append("functional")
    if assets.annotation_elements:
        task_types.append("annotation")
    return task_types


def _run_single_species(args: argparse.Namespace, *, species_name: str, task_type: str) -> dict[str, Any]:
    output_dir = Path(args.output_root).expanduser().resolve() / species_name / task_type
    config = _base_config_from_args(args, species_name=species_name, task_type=task_type, output_dir=output_dir)
    log.info("Running NTv3 benchmark: species=%s task=%s", species_name, task_type)
    return run_ntv3_benchmark(config)


def _add_shared_eval_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-version", default=DEFAULT_MODEL_VERSION)
    parser.add_argument("--checkpoint-dir", default=str(DEFAULT_LOCAL_CHECKPOINT_DIR))
    parser.add_argument("--checkpoint-s3-prefix", default=DEFAULT_CHECKPOINT_S3_PREFIX)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--auto-resume", action="store_true")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--dataset-repo-id", default=DEFAULT_DATASET_REPO_ID)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--sequence-length", type=int, default=32_768)
    parser.add_argument("--keep-target-center-fraction", type=float, default=0.375)
    parser.add_argument("--train-overlap", type=float, default=0.999)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--max-runtime-batch-size-per-rank", type=int, default=None)
    parser.add_argument("--num-steps-training", type=int, default=2_000)
    parser.add_argument("--validate-every-n-steps", type=int, default=200)
    parser.add_argument("--num-validation-samples", type=int, default=1_000)
    parser.add_argument("--num-test-samples", type=int, default=None)
    parser.add_argument("--initial-learning-rate", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-steps-warmup", type=int, default=None)
    parser.add_argument("--scheduler-name", choices=("cosine", "modified_square_decay"), default="cosine")
    parser.add_argument("--final-learning-rate-multiplier", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--head-learning-rate", type=float, default=None)
    parser.add_argument("--backbone-learning-rate", type=float, default=None)
    parser.add_argument("--decoder-learning-rate", type=float, default=None)
    parser.add_argument("--backbone-layerwise-lr-decay", type=float, default=None)
    parser.add_argument("--head-only-warmup-steps", type=int, default=0)
    parser.add_argument("--no-weight-decay-norm-bias", action="store_true")
    parser.add_argument("--ema-decay", type=float, default=None)
    parser.add_argument("--warmup-fraction", type=float, default=0.03)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every-n-steps", type=int, default=50)
    parser.add_argument("--save-every-n-steps", type=int, default=4_000)
    parser.add_argument("--max-checkpoints-to-keep", type=int, default=3)
    parser.add_argument("--freeze-backbone", action="store_true", default=True)
    parser.add_argument("--train-backbone", dest="freeze_backbone", action="store_false")
    parser.add_argument("--feature-source", choices=("hidden", "decoder"), default="hidden")
    parser.add_argument(
        "--functional-head-type",
        choices=(
            "linear",
            "mlp",
            "local-conv",
            "gated-hybrid",
            "multi-scale-dilated",
            "global-context",
            "context-pyramid",
            "v10-representation-pyramid",
            "v10-bio-aux-pyramid",
            "v10-assay-gated-bio-pyramid",
            "v10-bioprogram-stack",
            "v10-profile-count-bioaux",
            "v10-profile-count-bioaux-rc-gated-residual",
            "v10-assay-rescue-hybrid",
            "v10-biocov-residual",
        ),
        default="linear",
    )
    parser.add_argument("--functional-head-hidden-dim", type=int, default=None)
    parser.add_argument("--functional-head-dropout", type=float, default=0.05)
    parser.add_argument("--functional-head-kernel-size", type=int, default=15)
    parser.add_argument(
        "--functional-head-aux-features",
        choices=("none", "phylo", "structure", "phylo-structure"),
        default="none",
    )
    parser.add_argument("--functional-head-aux-projection-dim", type=int, default=16)
    parser.add_argument(
        "--functional-head-output-bias-init",
        choices=("none", "scaled-track-mean"),
        default="none",
    )
    parser.add_argument("--functional-rc-consistency-weight", type=float, default=0.0)
    parser.add_argument(
        "--official-human-functional",
        action="store_true",
        help=(
            "Enforce the official-like human functional NTv3 reproduction preset: "
            "full fine-tuning, step budget, scheduler, and validation cadence from the public notebook."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("auto", "bf16", "fp32"), default="auto")
    parser.add_argument("--allow-tf32", action="store_true", default=True)
    parser.add_argument("--no-tf32", dest="allow_tf32", action="store_false")
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--wandb-enabled", action="store_true")
    parser.add_argument("--wandb-project", default="lumina-ntv3")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-tags", nargs="*", default=[])
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--run-id", default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a Lumina checkpoint on the NTv3 benchmark.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    stage_checkpoint_parser = subparsers.add_parser(
        "stage-checkpoint",
        help="Sync the Lumina checkpoint bundle locally.",
    )
    stage_checkpoint_parser.add_argument("--checkpoint-s3-prefix", default=DEFAULT_CHECKPOINT_S3_PREFIX)
    stage_checkpoint_parser.add_argument("--checkpoint-dir", default=str(DEFAULT_LOCAL_CHECKPOINT_DIR))

    stage_dataset_parser = subparsers.add_parser("stage-dataset", help="Download the NTv3 benchmark dataset locally.")
    stage_dataset_parser.add_argument("--dataset-repo-id", default=DEFAULT_DATASET_REPO_ID)
    stage_dataset_parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    stage_dataset_parser.add_argument("--species", nargs="*", default=None)
    stage_dataset_parser.add_argument("--skip-functional", action="store_true")
    stage_dataset_parser.add_argument("--skip-annotation", action="store_true")

    for command in ("fit-species", "evaluate-species"):
        species_parser = subparsers.add_parser(command, help="Train and evaluate a single species/task pair.")
        species_parser.add_argument("--species", required=True)
        species_parser.add_argument("--task-type", choices=("functional", "annotation"), required=True)
        _add_shared_eval_args(species_parser)

    evaluate_all_parser = subparsers.add_parser(
        "evaluate-all",
        help="Run the benchmark for all available species/tasks.",
    )
    evaluate_all_parser.add_argument("--species", nargs="*", default=None)
    evaluate_all_parser.add_argument("--task-types", nargs="*", choices=("functional", "annotation"), default=None)
    _add_shared_eval_args(evaluate_all_parser)

    return parser


def main(argv: list[str] | None = None) -> None:
    load_dotenv_if_available()
    args = build_parser().parse_args(argv)

    if args.command == "stage-checkpoint":
        checkpoint_path = stage_checkpoint(
            checkpoint_s3_prefix=args.checkpoint_s3_prefix,
            checkpoint_dir=Path(args.checkpoint_dir).expanduser().resolve(),
        )
        print(json.dumps({"checkpoint_path": str(checkpoint_path)}, indent=2))
        return

    if args.command == "stage-dataset":
        try:
            dataset_root = stage_dataset(
                repo_id=args.dataset_repo_id,
                dataset_root=Path(args.dataset_root).expanduser().resolve(),
                species_names=args.species,
                include_functional=not args.skip_functional,
                include_annotation=not args.skip_annotation,
            )
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
        print(json.dumps({"dataset_root": str(dataset_root)}, indent=2))
        return

    if args.command in {"fit-species", "evaluate-species"}:
        summary = _run_single_species(args, species_name=args.species, task_type=args.task_type)
        if int(os.environ.get("RANK", "0")) == 0:
            print(json.dumps(summary, indent=2, sort_keys=True))
        return

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    species_names = args.species or discover_species(dataset_root)
    if not species_names:
        raise ValueError(f"No species discovered under dataset root {dataset_root}")

    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for species_name in species_names:
        available_tasks = _available_task_types(dataset_root, species_name)
        requested_tasks = args.task_types or available_tasks
        for task_type in requested_tasks:
            if task_type not in available_tasks:
                log.info("Skipping unavailable task: species=%s task=%s", species_name, task_type)
                continue
            summary = _run_single_species(args, species_name=species_name, task_type=task_type)
            summaries.append(summary)
            all_rows.extend(_collect_rows(Path(summary["dataset_scores_path"])))

    aggregate_path = _write_aggregate_csv(
        Path(args.output_root).expanduser().resolve() / "ntv3_benchmark_results.csv",
        all_rows,
    )
    print(json.dumps({"aggregate_csv": str(aggregate_path), "runs": summaries}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
