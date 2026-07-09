#!/usr/bin/env python3
"""SageMaker submitter for NTv3 benchmark fine-tuning/evaluation."""

from __future__ import annotations

import argparse
import os
import re
import secrets
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.ntv3.config import DEFAULT_CHECKPOINT_S3_PREFIX  # noqa: E402
from src.sagemaker_utils import (  # noqa: E402
    DEFAULT_IMAGE_URI,
    DEFAULT_SAGEMAKER_ENTRY_SCRIPT,
    SM_DATA,
    build_sagemaker_entry_environment,
    cli_arg_value,
    has_cli_flag,
    load_dotenv_if_available,
    package_source,
    parse_s3_uri,
    resolve_packaged_repo_relative_path,
    split_cli_args,
)

JOB_ENTRYPOINT = Path("scripts/ntv3_benchmark_job.py")
SAGEMAKER_ENTRYPOINT = Path(DEFAULT_SAGEMAKER_ENTRY_SCRIPT)
DEFAULT_HUGGINGFACE_HOME = "/tmp/huggingface"
DEFAULT_HUGGINGFACE_HUB_CACHE = f"{DEFAULT_HUGGINGFACE_HOME}/hub"
DEFAULT_HUGGINGFACE_MODULES_CACHE = f"{DEFAULT_HUGGINGFACE_HOME}/modules"
DEFAULT_CHECKPOINT_LOCAL_ROOT = "/opt/ml/checkpoints"
DEFAULT_RESULTS_ROOT = f"{DEFAULT_CHECKPOINT_LOCAL_ROOT}/ntv3-results"
DEFAULT_TILELANG_CACHE_DIR = f"{DEFAULT_CHECKPOINT_LOCAL_ROOT}/tilelang-cache"
DEFAULT_SAGEMAKER_CHECKPOINT_SYNC_ROOT = "/sagemaker-ntv3-checkpoints"
DEFAULT_TILELANG_TMP_DIR = f"{DEFAULT_TILELANG_CACHE_DIR}/tmp"
DEFAULT_TILELANG_EXECUTION_BACKEND = "tvm_ffi"
H200_INSTANCE_TYPE = "ml.p5en.48xlarge"
H200_RUNTIME_BATCH_CAP = 2
SPOT_SAVE_EVERY_N_STEPS = 1000
SAGEMAKER_TRAINING_JOB_NAME_LIMIT = 63
SAGEMAKER_TRAINING_JOB_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9](-*[a-zA-Z0-9]){0,62}$")
SAGEMAKER_TRAINING_JOB_TIMESTAMP_LENGTH = 14
SAGEMAKER_TRAINING_JOB_RESERVED_SUFFIX_LENGTH = 1 + SAGEMAKER_TRAINING_JOB_TIMESTAMP_LENGTH
NTV3_JOB_NAME_PREFIX = "lumina-ssm-ntv3"
NTV3_JOB_NAME_SUFFIX_LENGTH = 6


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    launcher_args, benchmark_args = split_cli_args(argv or sys.argv[1:])
    parser = argparse.ArgumentParser(
        description="Launch a SageMaker job for NTv3 benchmark fine-tuning.",
    )
    parser.add_argument("--experiment", required=True)
    parser.add_argument(
        "--job-name-prefix",
        default=NTV3_JOB_NAME_PREFIX,
        help=(
            "Prefix for SageMaker training job names. "
            f"Default: {NTV3_JOB_NAME_PREFIX}."
        ),
    )
    parser.add_argument("--bucket", required=True)
    parser.add_argument(
        "--role",
        default=os.environ.get("SAGEMAKER_ROLE", ""),
        help="SageMaker execution role ARN (default: SAGEMAKER_ROLE).",
    )
    parser.add_argument(
        "--instance-type",
        default="ml.p6-b200.48xlarge",
        help="SageMaker instance type (default: ml.p6-b200.48xlarge).",
    )
    parser.add_argument("--instance-count", type=int, default=1)
    parser.add_argument(
        "--nproc-per-node",
        type=int,
        default=None,
        help="Override torchrun --nproc-per-node inside the container (default: use SM_NUM_GPUS).",
    )
    parser.add_argument("--spot", action="store_true")
    parser.add_argument("--max-run-hours", type=int, default=72)
    parser.add_argument("--max-wait-hours", type=int, default=None)
    parser.add_argument(
        "--training-image-uri",
        default=os.environ.get("SAGEMAKER_TRAINING_IMAGE", ""),
    )
    parser.add_argument(
        "--checkpoint-s3-prefix",
        default=DEFAULT_CHECKPOINT_S3_PREFIX,
        help="S3 URI for a Lumina checkpoint bundle or SageMaker model artifact to stage in the container.",
    )
    parser.add_argument(
        "--dataset-repo-id",
        default="InstaDeepAI/NTv3_benchmark_dataset",
        help="Hugging Face dataset repo id for NTv3 benchmark data.",
    )
    parser.add_argument(
        "--wandb-key",
        default=os.environ.get("WANDB_API_KEY", ""),
        help="Weights & Biases API key (default: WANDB_API_KEY).",
    )
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN", "") or os.environ.get("HUGGINGFACE_HUB_TOKEN", ""),
        help="Hugging Face token for staging the gated NTv3 dataset.",
    )
    parser.add_argument("--detach", action="store_true")
    parser.add_argument(
        "--no-auto-resume",
        action="store_true",
        help="Do not inject --auto-resume into the NTv3 runtime command.",
    )
    parser.add_argument("--region", default=os.environ.get("AWS_DEFAULT_REGION", "us-east-2"))
    return parser.parse_args(launcher_args), benchmark_args


def _sanitize_training_job_name_component(value: str) -> str:
    sanitized = re.sub(r"[^a-z0-9-]+", "-", value.lower())
    sanitized = re.sub(r"-{2,}", "-", sanitized).strip("-")
    return sanitized or "job"


def _generate_job_name_suffix() -> str:
    return secrets.token_hex((NTV3_JOB_NAME_SUFFIX_LENGTH + 1) // 2)[:NTV3_JOB_NAME_SUFFIX_LENGTH]


def _checkpoint_source_dir_name(checkpoint_s3_prefix: str) -> str:
    source_path = Path(parse_s3_uri(checkpoint_s3_prefix)[1].rstrip("/"))
    source_name = source_path.name
    if source_name.endswith(".tar.gz"):
        if source_name == "model.tar.gz" and source_path.parent.name == "output":
            return source_path.parent.parent.name
        return source_name[: -len(".tar.gz")]
    return source_name


def build_ntv3_base_job_name(
    *,
    experiment: str,
    species: str,
    task_type: str,
    random_suffix: str | None = None,
    job_name_prefix: str = NTV3_JOB_NAME_PREFIX,
) -> str:
    suffix = _sanitize_training_job_name_component(random_suffix or _generate_job_name_suffix())
    prefix_slug = _sanitize_training_job_name_component(job_name_prefix)
    experiment_slug = _sanitize_training_job_name_component(experiment)
    species_slug = _sanitize_training_job_name_component(species)
    task_slug = _sanitize_training_job_name_component(task_type)
    max_base_length = SAGEMAKER_TRAINING_JOB_NAME_LIMIT - SAGEMAKER_TRAINING_JOB_RESERVED_SUFFIX_LENGTH
    min_experiment_length = 1
    fixed_length = len(prefix_slug) + len(species_slug) + len(task_slug) + len(suffix) + 4
    if fixed_length + min_experiment_length > max_base_length:
        max_prefix_length = (
            max_base_length
            - min_experiment_length
            - len(species_slug)
            - len(task_slug)
            - len(suffix)
            - 4
        )
        prefix_slug = prefix_slug[: max(1, max_prefix_length)].rstrip("-") or "job"
    max_experiment_length = (
        max_base_length
        - len(prefix_slug)
        - len(species_slug)
        - len(task_slug)
        - len(suffix)
        - 4
    )
    experiment_slug = experiment_slug[: max(1, max_experiment_length)].rstrip("-") or "job"
    job_name = "-".join([prefix_slug, experiment_slug, species_slug, task_slug, suffix])
    if SAGEMAKER_TRAINING_JOB_NAME_PATTERN.fullmatch(job_name) is None:
        raise ValueError(f"Invalid SageMaker training job name: {job_name!r}")
    return job_name


def _upsert_cli_arg(argv: list[str], flag: str, value: str) -> list[str]:
    updated = list(argv)
    prefix = f"{flag}="
    for index, token in enumerate(updated):
        if token == flag:
            next_index = index + 1
            if next_index < len(updated) and not updated[next_index].startswith("--"):
                updated[next_index] = value
            else:
                updated.insert(next_index, value)
            return updated
        if token.startswith(prefix):
            updated[index] = f"{flag}={value}"
            return updated
    updated.extend([flag, value])
    return updated


def _ensure_cli_flag(argv: list[str], flag: str) -> list[str]:
    if has_cli_flag(argv, flag):
        return argv
    return [*argv, flag]


def _spot_safe_save_every_n_steps(benchmark_args: list[str]) -> list[str]:
    current_value = cli_arg_value(benchmark_args, "--save-every-n-steps")
    if current_value is None:
        return _upsert_cli_arg(benchmark_args, "--save-every-n-steps", str(SPOT_SAVE_EVERY_N_STEPS))

    if int(current_value) > SPOT_SAVE_EVERY_N_STEPS:
        return _upsert_cli_arg(benchmark_args, "--save-every-n-steps", str(SPOT_SAVE_EVERY_N_STEPS))
    return benchmark_args


def _hardware_safe_runtime_batch_cap(benchmark_args: list[str], *, instance_type: str) -> list[str]:
    if instance_type != H200_INSTANCE_TYPE:
        return benchmark_args
    if cli_arg_value(benchmark_args, "--max-runtime-batch-size-per-rank") is not None:
        return benchmark_args
    return _upsert_cli_arg(
        benchmark_args,
        "--max-runtime-batch-size-per-rank",
        str(H200_RUNTIME_BATCH_CAP),
    )


def inject_launcher_benchmark_args(
    benchmark_args: list[str],
    *,
    experiment: str,
    checkpoint_dir: str,
    instance_type: str,
    managed_spot: bool,
    wandb_enabled: bool,
    default_wandb_run_name: str,
    auto_resume: bool = True,
) -> list[str]:
    runtime_args = list(benchmark_args)
    if cli_arg_value(runtime_args, "--species") is None:
        runtime_args.extend(["--species", "human"])
    if cli_arg_value(runtime_args, "--task-type") is None:
        runtime_args.extend(["--task-type", "functional"])
    runtime_args = _ensure_cli_flag(runtime_args, "--train-backbone")
    runtime_args = _ensure_cli_flag(runtime_args, "--overwrite")
    if auto_resume:
        runtime_args = _ensure_cli_flag(runtime_args, "--auto-resume")
    runtime_args = _upsert_cli_arg(runtime_args, "--checkpoint-dir", checkpoint_dir)
    runtime_args = _upsert_cli_arg(runtime_args, "--dataset-root", f"{SM_DATA}/datasets/ntv3")
    runtime_args = _upsert_cli_arg(runtime_args, "--output-root", f"{DEFAULT_RESULTS_ROOT}/{experiment}")
    runtime_args = _upsert_cli_arg(runtime_args, "--run-id", experiment)
    runtime_args = _hardware_safe_runtime_batch_cap(runtime_args, instance_type=instance_type)
    if managed_spot:
        runtime_args = _spot_safe_save_every_n_steps(runtime_args)

    if wandb_enabled:
        runtime_args = _ensure_cli_flag(runtime_args, "--wandb-enabled")
        runtime_args = _upsert_cli_arg(
            runtime_args,
            "--wandb-project",
            cli_arg_value(runtime_args, "--wandb-project") or "lumina-ntv3",
        )
        runtime_args = _upsert_cli_arg(
            runtime_args,
            "--wandb-entity",
            cli_arg_value(runtime_args, "--wandb-entity") or "ai4bio-lumina",
        )
        runtime_args = _upsert_cli_arg(
            runtime_args,
            "--wandb-run-name",
            cli_arg_value(runtime_args, "--wandb-run-name") or default_wandb_run_name,
        )
    return runtime_args


def build_container_environment(
    *,
    experiment: str,
    benchmark_args: list[str],
    wandb_key: str,
    hf_token: str,
) -> dict[str, str]:
    if not hf_token:
        raise ValueError(
            "NTv3 benchmark SageMaker jobs require HF_TOKEN / HUGGINGFACE_HUB_TOKEN to stage the gated dataset."
        )
    environment = {
        "FI_EFA_FORK_SAFE": "1",
        "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "UV_NO_SOURCES_PACKAGE": "mamba-ssm",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HOME": DEFAULT_HUGGINGFACE_HOME,
        "HF_HUB_CACHE": DEFAULT_HUGGINGFACE_HUB_CACHE,
        "HF_MODULES_CACHE": DEFAULT_HUGGINGFACE_MODULES_CACHE,
        "TILELANG_CACHE_DIR": DEFAULT_TILELANG_CACHE_DIR,
        "TILELANG_TMP_DIR": DEFAULT_TILELANG_TMP_DIR,
        "TILELANG_EXECUTION_BACKEND": DEFAULT_TILELANG_EXECUTION_BACKEND,
    }
    environment["HF_TOKEN"] = hf_token
    environment["HUGGINGFACE_HUB_TOKEN"] = hf_token
    if os.environ.get("LUMINA_CUDA_DEBUG_SYNC", "").strip().lower() in {"1", "true", "yes", "on"}:
        environment["CUDA_LAUNCH_BLOCKING"] = "1"
        environment["TORCH_SHOW_CPP_STACKTRACES"] = "1"
        environment["PYTHONFAULTHANDLER"] = "1"

    for diagnostic_env in (
        "LUMINA_NTV3_DEBUG_EARLY_STEPS",
        "LUMINA_NTV3_DEBUG_DATASET_ITEMS",
        "LUMINA_NTV3_DEBUG_EVAL_BATCHES",
        "LUMINA_NTV3_DEBUG_EVAL_START_BATCH",
        "LUMINA_NTV3_DEBUG_EVAL_MAX_BATCHES",
    ):
        raw = os.environ.get(diagnostic_env, "").strip()
        if raw:
            environment[diagnostic_env] = raw

    if has_cli_flag(benchmark_args, "--wandb-enabled"):
        if not wandb_key:
            raise ValueError(
                "W&B logging was requested for NTv3 benchmark, "
                "but WANDB_API_KEY / --wandb-key was not set."
            )
        environment["WANDB_API_KEY"] = wandb_key
        environment["WANDB_PROJECT"] = cli_arg_value(benchmark_args, "--wandb-project") or "lumina-ntv3"
        environment["WANDB_ENTITY"] = cli_arg_value(benchmark_args, "--wandb-entity") or "ai4bio-lumina"
        environment["WANDB_RUN_GROUP"] = f"sagemaker-{experiment}"
    return environment


def build_job_args(
    *,
    experiment: str,
    results_s3_prefix: str,
    checkpoint_s3_prefix: str | None,
    dataset_repo_id: str,
    nproc_per_node: int | None,
    benchmark_args: list[str],
    no_auto_resume: bool = False,
) -> list[str]:
    job_args = [
        "--experiment",
        experiment,
        "--results-s3-prefix",
        results_s3_prefix,
        "--dataset-repo-id",
        dataset_repo_id,
    ]
    if checkpoint_s3_prefix is not None:
        job_args.extend(["--checkpoint-s3-prefix", checkpoint_s3_prefix])
    if no_auto_resume:
        job_args.append("--no-auto-resume")
    if nproc_per_node is not None:
        job_args.extend(["--nproc-per-node", str(nproc_per_node)])
    if benchmark_args:
        job_args.extend(["--", *benchmark_args])
    return job_args


def main(argv: list[str] | None = None) -> int:
    load_dotenv_if_available()
    args, benchmark_args = parse_args(argv)

    if not args.role:
        print("Error: SageMaker execution role is required.")
        print("  Set SAGEMAKER_ROLE in .env or pass --role.")
        return 1

    if not SAGEMAKER_ENTRYPOINT.is_file():
        raise FileNotFoundError(f"SageMaker entrypoint not found at {SAGEMAKER_ENTRYPOINT}")
    if not JOB_ENTRYPOINT.is_file():
        raise FileNotFoundError(f"Container job entrypoint not found at {JOB_ENTRYPOINT}")

    species_name = cli_arg_value(benchmark_args, "--species") or "human"
    task_type = cli_arg_value(benchmark_args, "--task-type") or "functional"
    name_suffix = _generate_job_name_suffix()
    base_job_name = build_ntv3_base_job_name(
        experiment=args.experiment,
        species=species_name,
        task_type=task_type,
        random_suffix=name_suffix,
        job_name_prefix=args.job_name_prefix,
    )
    default_wandb_run_name = f"{base_job_name}-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"
    checkpoint_dir = str(Path(SM_DATA) / "checkpoints" / _checkpoint_source_dir_name(args.checkpoint_s3_prefix))

    benchmark_args = inject_launcher_benchmark_args(
        benchmark_args,
        experiment=args.experiment,
        checkpoint_dir=checkpoint_dir,
        instance_type=args.instance_type,
        managed_spot=args.spot,
        wandb_enabled=bool(args.wandb_key),
        default_wandb_run_name=default_wandb_run_name,
        auto_resume=not args.no_auto_resume,
    )

    try:
        import boto3
        from sagemaker.core.helper.session_helper import Session
        from sagemaker.train import ModelTrainer
        from sagemaker.train.configs import (
            CheckpointConfig,
            Compute,
            OutputDataConfig,
            SourceCode,
            StoppingCondition,
        )
    except ImportError as exc:
        raise RuntimeError(
            "SageMaker launcher dependencies are missing. Install them with `uv sync --extra sagemaker`."
        ) from exc

    s3_prefix = f"s3://{args.bucket}/lumina-ssm"
    s3_results = f"{s3_prefix}/eval/ntv3-benchmark/{args.experiment}/"
    s3_artifacts = f"{s3_prefix}/sagemaker-artifacts/ntv3-benchmark/{args.experiment}/"
    s3_checkpoint = f"{s3_prefix}/checkpoints/ntv3-benchmark/{args.experiment}/"

    image_uri = args.training_image_uri.strip() or DEFAULT_IMAGE_URI
    max_wait_hours = args.max_wait_hours
    if args.spot and max_wait_hours is None:
        max_wait_hours = args.max_run_hours * 2

    job_args = build_job_args(
        experiment=args.experiment,
        results_s3_prefix=s3_results,
        checkpoint_s3_prefix=args.checkpoint_s3_prefix,
        dataset_repo_id=args.dataset_repo_id,
        nproc_per_node=args.nproc_per_node,
        benchmark_args=benchmark_args,
        no_auto_resume=args.no_auto_resume,
    )
    environment = build_container_environment(
        experiment=args.experiment,
        benchmark_args=benchmark_args,
        wandb_key=args.wandb_key,
        hf_token=args.hf_token,
    )
    print(f"Creating SageMaker NTv3 benchmark job for experiment: {args.experiment}")
    print(f"  Instance: {args.instance_type} x {args.instance_count}")
    print(f"  Image: {image_uri}")
    print(f"  Results: {s3_results}")
    print(f"  SageMaker artifacts: {s3_artifacts}")
    print(f"  Checkpoints: {s3_checkpoint}")
    print(f"  Checkpoint sync source: {args.checkpoint_s3_prefix}")
    print(f"  Dataset repo: {args.dataset_repo_id}")
    print(f"  Species/task: {species_name}/{task_type}")

    boto_session = boto3.Session(region_name=args.region)
    sagemaker_session = Session(boto_session=boto_session)

    stopping_kwargs: dict[str, Any] = {
        "max_runtime_in_seconds": args.max_run_hours * 3600,
    }
    if args.spot and max_wait_hours is not None:
        stopping_kwargs["max_wait_time_in_seconds"] = max_wait_hours * 3600

    compute_kwargs: dict[str, Any] = {
        "instance_type": args.instance_type,
        "instance_count": args.instance_count,
        "volume_size_in_gb": 500,
    }
    if args.spot:
        compute_kwargs["enable_managed_spot_training"] = True

    source_dir = package_source()
    try:
        environment = build_sagemaker_entry_environment(
            environment,
            setup_extras="eval,tracking",
            job_script=JOB_ENTRYPOINT,
            job_args=job_args,
            source_dir=source_dir,
        )
        trainer = cast(Any, ModelTrainer)(
            training_image=image_uri,
            role=args.role,
            sagemaker_session=sagemaker_session,
            source_code=SourceCode(
                source_dir=source_dir,
                entry_script=resolve_packaged_repo_relative_path(SAGEMAKER_ENTRYPOINT),
            ),
            compute=Compute(**compute_kwargs),
            output_data_config=OutputDataConfig(s3_output_path=s3_artifacts),
            checkpoint_config=CheckpointConfig(
                s3_uri=s3_checkpoint,
                local_path=DEFAULT_SAGEMAKER_CHECKPOINT_SYNC_ROOT,
            ),
            stopping_condition=StoppingCondition(**stopping_kwargs),
            environment=environment,
            base_job_name=base_job_name,
        )

        print("\nStarting training job...")
        print("=" * 60)
        training_job = cast(Any, trainer).train(
            input_data_config=[],
            wait=not args.detach,
            logs=not args.detach,
        )

        job_name = training_job.training_job_name if hasattr(training_job, "training_job_name") else base_job_name
        if args.detach:
            print("=" * 60)
            print("Job submitted. You can safely close this terminal.")
            print(f"  Job name: {job_name}")
            print(
                f"  Monitor: https://console.aws.amazon.com/sagemaker/home"
                f"?region={args.region}#/training-jobs/{job_name}"
            )
            print(f"  Results: {s3_results}")
        else:
            print("=" * 60)
            print(f"NTv3 benchmark job complete. Final results uploaded to: {s3_results}")
    finally:
        shutil.rmtree(source_dir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
