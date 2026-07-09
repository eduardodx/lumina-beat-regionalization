#!/usr/bin/env python3
"""SageMaker submitter for ClinVar end-to-end fine-tuning."""

from __future__ import annotations

import argparse
import os
import re
import secrets
import shutil
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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

JOB_ENTRYPOINT = Path("scripts/clinvar_finetune_job.py")
SAGEMAKER_ENTRYPOINT = Path(DEFAULT_SAGEMAKER_ENTRY_SCRIPT)
SAGEMAKER_TRAINING_JOB_NAME_LIMIT = 63
SAGEMAKER_TRAINING_JOB_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9](-*[a-zA-Z0-9]){0,62}$")
SAGEMAKER_TRAINING_JOB_TIMESTAMP_LENGTH = 14
SAGEMAKER_TRAINING_JOB_RESERVED_SUFFIX_LENGTH = 1 + SAGEMAKER_TRAINING_JOB_TIMESTAMP_LENGTH
CLINVAR_JOB_NAME_PREFIX = "lumina-ssm-clinvar"
CLINVAR_JOB_NAME_SUFFIX_LENGTH = 6
CLINVAR_JOB_TIMESTAMP_FORMAT = "%Y%m%d-%H%M%S"
DEFAULT_HUGGINGFACE_HOME = "/tmp/huggingface"
DEFAULT_HUGGINGFACE_HUB_CACHE = f"{DEFAULT_HUGGINGFACE_HOME}/hub"
DEFAULT_HUGGINGFACE_MODULES_CACHE = f"{DEFAULT_HUGGINGFACE_HOME}/modules"


@dataclass(frozen=True)
class FineTuneDispatchSpec:
    regime: str
    model_family: str
    model_version: str
    checkpoint_dir: str | None
    checkpoint_path: str | None
    checkpoint_s3_prefix: str | None = None


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    launcher_args, finetune_args = split_cli_args(argv or [])
    parser = argparse.ArgumentParser(
        description="Launch a SageMaker job for ClinVar end-to-end fine-tuning.",
    )
    parser.add_argument("--experiment", required=True, help="Experiment name used for the SageMaker job and S3 paths.")
    parser.add_argument("--bucket", required=True, help="S3 bucket for SageMaker input and output paths.")
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
    parser.add_argument(
        "--instance-count",
        type=int,
        default=1,
        help="Number of SageMaker instances (default: 1).",
    )
    parser.add_argument("--spot", action="store_true", help="Use SageMaker managed spot capacity.")
    parser.add_argument(
        "--max-run-hours",
        type=int,
        default=72,
        help="Maximum training job runtime in hours (default: 72).",
    )
    parser.add_argument(
        "--max-wait-hours",
        type=int,
        default=None,
        help="Maximum wait time for spot capacity in hours (default: 2x max-run-hours).",
    )
    parser.add_argument(
        "--training-image-uri",
        default=os.environ.get("SAGEMAKER_TRAINING_IMAGE", ""),
        help="Optional custom training image override.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help=(
            "Lumina checkpoint bundle directory, either container-local under `/opt/ml/` "
            "or an S3 prefix under `s3://.../data/checkpoints/...`. "
            "The launcher resolves `<checkpoint-dir>/best_checkpoint.pt` for the ClinVar runner."
        ),
    )
    parser.add_argument(
        "--data-s3-prefix",
        default="",
        help="Optional S3 prefix for container-side data sync (default: s3://<bucket>/lumina-ssm/data/).",
    )
    parser.add_argument(
        "--huggingface-cache-s3-prefix",
        default="",
        help="Optional S3 prefix for container-side Hugging Face hub cache sync (default: s3://<bucket>/huggingface/hub/).",
    )
    parser.add_argument(
        "--wandb-key",
        default=os.environ.get("WANDB_API_KEY", ""),
        help="Weights & Biases API key (default: WANDB_API_KEY).",
    )
    parser.add_argument("--detach", action="store_true", help="Submit job and return without waiting.")
    parser.add_argument(
        "--region",
        default=os.environ.get("AWS_DEFAULT_REGION", "us-east-2"),
        help="AWS region (default: AWS_DEFAULT_REGION or us-east-2).",
    )
    return parser.parse_args(launcher_args), finetune_args


def parse_cli(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    return parse_args(argv if argv is not None else sys.argv[1:])


def _sanitize_training_job_name_component(value: str) -> str:
    sanitized = re.sub(r"[^a-z0-9-]+", "-", value.lower())
    sanitized = re.sub(r"-{2,}", "-", sanitized).strip("-")
    return sanitized or "job"


def _normalize_regime_name(value: str | None) -> str:
    if value is None:
        return "A"
    normalized = value.strip().upper()
    return normalized or "A"


def _format_regime_slug(regime: str) -> str:
    return f"regime-{_sanitize_training_job_name_component(regime)}"


def _format_regime_job_component(regime: str) -> str:
    return f"reg-{_sanitize_training_job_name_component(regime)}"


def _generate_job_name_suffix() -> str:
    return secrets.token_hex((CLINVAR_JOB_NAME_SUFFIX_LENGTH + 1) // 2)[:CLINVAR_JOB_NAME_SUFFIX_LENGTH]


def _validate_sagemaker_training_job_name(name: str) -> str:
    if len(name) > SAGEMAKER_TRAINING_JOB_NAME_LIMIT:
        raise ValueError(
            f"Training job name {name!r} exceeds {SAGEMAKER_TRAINING_JOB_NAME_LIMIT} characters."
        )
    if SAGEMAKER_TRAINING_JOB_NAME_PATTERN.fullmatch(name) is None:
        raise ValueError(
            "Training job name must match "
            f"{SAGEMAKER_TRAINING_JOB_NAME_PATTERN.pattern!r}, got {name!r}."
        )
    return name


def _build_clinvar_model_job_slug(
    *,
    experiment: str | None,
    model_family: str | None,
    model_version: str | None,
) -> str:
    family_slug = _sanitize_training_job_name_component(model_family or "")
    version_slug = _sanitize_training_job_name_component(model_version or "")
    experiment_slug = _sanitize_training_job_name_component(experiment or "")

    if family_slug and version_slug:
        if version_slug == family_slug or version_slug.startswith(f"{family_slug}-"):
            return version_slug
        return f"{family_slug}-{version_slug}"
    if version_slug:
        return version_slug
    if family_slug:
        return family_slug
    return experiment_slug


def build_clinvar_base_job_name(
    *,
    experiment: str | None = None,
    model_family: str | None = None,
    model_version: str | None = None,
    regime: str | None = None,
    random_suffix: str | None = None,
) -> str:
    job_slug = _build_clinvar_model_job_slug(
        experiment=experiment,
        model_family=model_family,
        model_version=model_version,
    )
    regime_component = _format_regime_job_component(_normalize_regime_name(regime))
    suffix = _sanitize_training_job_name_component(random_suffix or _generate_job_name_suffix())
    max_base_length = SAGEMAKER_TRAINING_JOB_NAME_LIMIT - SAGEMAKER_TRAINING_JOB_RESERVED_SUFFIX_LENGTH
    max_experiment_length = (
        max_base_length
        - len(CLINVAR_JOB_NAME_PREFIX)
        - len(regime_component)
        - len(suffix)
        - 3
    )
    truncated_slug = job_slug[: max(1, max_experiment_length)].rstrip("-") or "job"
    return _validate_sagemaker_training_job_name(
        "-".join([CLINVAR_JOB_NAME_PREFIX, truncated_slug, regime_component, suffix])
    )


def build_clinvar_wandb_run_name(
    *,
    experiment: str | None = None,
    model_family: str | None = None,
    model_version: str | None = None,
    regime: str | None = None,
    now: datetime | None = None,
    random_suffix: str | None = None,
) -> str:
    timestamp = (now or datetime.now(UTC)).strftime(CLINVAR_JOB_TIMESTAMP_FORMAT)
    job_slug = _build_clinvar_model_job_slug(
        experiment=experiment,
        model_family=model_family,
        model_version=model_version,
    )
    regime_slug = _format_regime_slug(_normalize_regime_name(regime))
    suffix = _sanitize_training_job_name_component(random_suffix or _generate_job_name_suffix())
    return "-".join([CLINVAR_JOB_NAME_PREFIX, job_slug, regime_slug, timestamp, suffix])


def resolve_checkpoint_path(checkpoint_dir: str) -> str:
    checkpoint_root = Path(checkpoint_dir)
    return str(checkpoint_root / "best_checkpoint.pt")


def _resolve_remote_checkpoint_dir(checkpoint_dir: str) -> str:
    _bucket, key = parse_s3_uri(checkpoint_dir)
    checkpoint_name = Path(key.rstrip("/")).name
    if not checkpoint_name:
        raise ValueError(f"Remote checkpoint dir must end with a directory name, got {checkpoint_dir!r}.")
    return str(Path(SM_DATA) / "checkpoints" / checkpoint_name)


def resolve_huggingface_cache_s3_prefix(*, bucket: str, override: str | None) -> str:
    return (override or "").strip() or f"s3://{bucket}/huggingface/hub/"


def validate_finetune_args(
    finetune_args: list[str],
    *,
    checkpoint_dir: str | None,
) -> FineTuneDispatchSpec:
    regime = _normalize_regime_name(cli_arg_value(finetune_args, "--regime"))
    model_family = cli_arg_value(finetune_args, "--model-family")
    if not model_family:
        raise ValueError("ClinVar SageMaker dispatch requires `--model-family` after `--`.")

    model_version = cli_arg_value(finetune_args, "--model-version")
    if not model_version:
        raise ValueError("ClinVar SageMaker dispatch requires `--model-version` after `--`.")

    checkpoint_path = cli_arg_value(finetune_args, "--checkpoint-path")
    resolved_checkpoint_dir = checkpoint_dir
    checkpoint_s3_prefix: str | None = None
    if model_family.lower() == "lumina":
        if checkpoint_path and checkpoint_dir:
            raise ValueError(
                "Pass either the launcher-level `--checkpoint-dir` or the forwarded `--checkpoint-path`, not both."
            )
        if checkpoint_dir:
            if checkpoint_dir.startswith("s3://"):
                checkpoint_s3_prefix = checkpoint_dir.rstrip("/") + "/"
                resolved_checkpoint_dir = _resolve_remote_checkpoint_dir(checkpoint_dir)
                checkpoint_path = resolve_checkpoint_path(resolved_checkpoint_dir)
            else:
                checkpoint_path = resolve_checkpoint_path(checkpoint_dir)
        elif checkpoint_path:
            resolved_checkpoint_dir = str(Path(checkpoint_path).parent)
        if not checkpoint_path:
            raise ValueError(
                "Lumina ClinVar SageMaker jobs require `--checkpoint-dir`. "
                "Pass a container-local `/opt/ml/...` directory or an S3 prefix."
            )
        if checkpoint_s3_prefix is None and not checkpoint_path.startswith("/opt/ml/"):
            raise ValueError(
                "Lumina ClinVar SageMaker jobs require the checkpoint directory to resolve to a container-local "
                "path under `/opt/ml/` unless `--checkpoint-dir` is an S3 prefix."
            )

    return FineTuneDispatchSpec(
        regime=regime,
        model_family=model_family,
        model_version=model_version,
        checkpoint_dir=resolved_checkpoint_dir,
        checkpoint_path=checkpoint_path,
        checkpoint_s3_prefix=checkpoint_s3_prefix,
    )


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


def _ensure_wandb_run_name_has_regime(name: str, regime: str) -> str:
    regime_slug = _format_regime_slug(regime)
    if regime_slug in _sanitize_training_job_name_component(name):
        return name
    return f"{name}-{regime_slug}"


def inject_launcher_finetune_args(
    finetune_args: list[str],
    *,
    dispatch_spec: FineTuneDispatchSpec,
    default_wandb_run_name: str | None = None,
) -> list[str]:
    runtime_args = list(finetune_args)
    if dispatch_spec.checkpoint_path and cli_arg_value(runtime_args, "--checkpoint-path") is None:
        runtime_args.extend(["--checkpoint-path", dispatch_spec.checkpoint_path])
    if _forwarded_wandb_enabled(runtime_args):
        existing_run_name = cli_arg_value(runtime_args, "--wandb-run-name")
        resolved_run_name = (
            _ensure_wandb_run_name_has_regime(existing_run_name, dispatch_spec.regime)
            if existing_run_name is not None
            else default_wandb_run_name
        )
        if resolved_run_name is not None:
            runtime_args = _upsert_cli_arg(runtime_args, "--wandb-run-name", resolved_run_name)
    return runtime_args


def _forwarded_wandb_enabled(finetune_args: list[str]) -> bool:
    if "--no-wandb-enabled" in finetune_args:
        return False
    return has_cli_flag(finetune_args, "--wandb-enabled")


def build_container_environment(*, experiment: str, finetune_args: list[str], wandb_key: str) -> dict[str, str]:
    environment = {
        "FI_EFA_FORK_SAFE": "1",
        "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HOME": DEFAULT_HUGGINGFACE_HOME,
        "HF_HUB_CACHE": DEFAULT_HUGGINGFACE_HUB_CACHE,
        "HF_MODULES_CACHE": DEFAULT_HUGGINGFACE_MODULES_CACHE,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    if not _forwarded_wandb_enabled(finetune_args):
        return environment

    if not wandb_key:
        raise ValueError("W&B logging is enabled for ClinVar fine-tuning, but WANDB_API_KEY / --wandb-key was not set.")

    environment["WANDB_API_KEY"] = wandb_key
    environment["WANDB_PROJECT"] = cli_arg_value(finetune_args, "--wandb-project") or "lumina-ssm"
    environment["WANDB_ENTITY"] = cli_arg_value(finetune_args, "--wandb-entity") or "ai4bio-lumina"
    regime_slug = _format_regime_slug(_normalize_regime_name(cli_arg_value(finetune_args, "--regime")))
    environment["WANDB_RUN_GROUP"] = (
        f"sagemaker-{experiment}-{regime_slug}"
    )
    return environment


def build_job_args(
    *,
    experiment: str,
    results_s3_prefix: str,
    data_s3_prefix: str,
    huggingface_cache_s3_prefix: str,
    checkpoint_s3_prefix: str | None,
    finetune_args: list[str],
) -> list[str]:
    job_args = [
        "--experiment",
        experiment,
        "--results-s3-prefix",
        results_s3_prefix,
        "--data-s3-prefix",
        data_s3_prefix,
        "--huggingface-cache-s3-prefix",
        huggingface_cache_s3_prefix,
    ]
    if checkpoint_s3_prefix is not None:
        job_args.extend(["--checkpoint-s3-prefix", checkpoint_s3_prefix])
    if finetune_args:
        job_args.extend(["--", *finetune_args])
    return job_args


def main(argv: list[str] | None = None) -> int:
    load_dotenv_if_available()
    args, finetune_args = parse_cli(argv)

    if not args.role:
        print("Error: SageMaker execution role is required.")
        print("  Set SAGEMAKER_ROLE in .env or pass --role.")
        return 1

    dispatch_spec = validate_finetune_args(
        finetune_args,
        checkpoint_dir=args.checkpoint_dir,
    )
    launch_now = datetime.now(UTC)
    name_suffix = _generate_job_name_suffix()
    base_job_name = build_clinvar_base_job_name(
        experiment=args.experiment,
        model_family=dispatch_spec.model_family,
        model_version=dispatch_spec.model_version,
        regime=dispatch_spec.regime,
        random_suffix=name_suffix,
    )
    default_wandb_run_name = build_clinvar_wandb_run_name(
        experiment=args.experiment,
        model_family=dispatch_spec.model_family,
        model_version=dispatch_spec.model_version,
        regime=dispatch_spec.regime,
        now=launch_now,
        random_suffix=name_suffix,
    )
    finetune_args = inject_launcher_finetune_args(
        finetune_args,
        dispatch_spec=dispatch_spec,
        default_wandb_run_name=default_wandb_run_name,
    )
    if not SAGEMAKER_ENTRYPOINT.is_file():
        raise FileNotFoundError(f"SageMaker entrypoint not found at {SAGEMAKER_ENTRYPOINT}")
    if not JOB_ENTRYPOINT.is_file():
        raise FileNotFoundError(f"Container entrypoint not found at {JOB_ENTRYPOINT}")

    try:
        import boto3
        from sagemaker.core.helper.session_helper import Session
        from sagemaker.train import ModelTrainer
        from sagemaker.train.configs import Compute, InputData, OutputDataConfig, SourceCode, StoppingCondition
    except ImportError as exc:
        raise RuntimeError(
            "SageMaker launcher dependencies are missing. Install them with `uv sync --extra sagemaker`."
        ) from exc

    s3_prefix = f"s3://{args.bucket}/lumina-ssm"
    s3_data = f"{s3_prefix}/data/"
    s3_results = f"{s3_prefix}/eval/clinvar/finetune/{args.experiment}/"
    s3_artifacts = f"{s3_prefix}/sagemaker-artifacts/clinvar-finetune/{args.experiment}/"
    data_s3_prefix = args.data_s3_prefix.strip() or s3_data
    huggingface_cache_s3_prefix = resolve_huggingface_cache_s3_prefix(
        bucket=args.bucket,
        override=args.huggingface_cache_s3_prefix,
    )

    image_uri = args.training_image_uri.strip() or DEFAULT_IMAGE_URI
    max_wait_hours = args.max_wait_hours
    if args.spot and max_wait_hours is None:
        max_wait_hours = args.max_run_hours * 2

    job_args = build_job_args(
        experiment=args.experiment,
        results_s3_prefix=s3_results,
        data_s3_prefix=data_s3_prefix,
        huggingface_cache_s3_prefix=huggingface_cache_s3_prefix,
        checkpoint_s3_prefix=dispatch_spec.checkpoint_s3_prefix,
        finetune_args=finetune_args,
    )
    environment = build_container_environment(
        experiment=args.experiment,
        finetune_args=finetune_args,
        wandb_key=args.wandb_key,
    )
    print(f"Creating SageMaker ClinVar fine-tuning job for experiment: {args.experiment}")
    print(f"  Instance: {args.instance_type} x {args.instance_count}")
    print(f"  Image: {image_uri}")
    print(f"  Data: {s3_data}")
    print(f"  Data sync source: {data_s3_prefix}")
    print(f"  Hugging Face cache sync source: {huggingface_cache_s3_prefix}")
    print(f"  Results: {s3_results}")
    print(f"  SageMaker artifacts: {s3_artifacts}")
    print(f"  Container entrypoint: {SAGEMAKER_ENTRYPOINT} -> {JOB_ENTRYPOINT}")
    if dispatch_spec.checkpoint_dir is not None:
        print(f"  Checkpoint dir: {dispatch_spec.checkpoint_dir}")
    if dispatch_spec.checkpoint_s3_prefix is not None:
        print(f"  Checkpoint sync source: {dispatch_spec.checkpoint_s3_prefix}")
    if args.spot:
        print(f"  Spot: ENABLED (max_run={args.max_run_hours}h, max_wait={max_wait_hours}h)")

    boto_session = boto3.Session(region_name=args.region)
    session_cls = cast(Any, Session)
    sagemaker_session = session_cls(boto_session=boto_session)

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
            setup_extras="eval,sagemaker,tracking",
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
            stopping_condition=StoppingCondition(**stopping_kwargs),
            environment=environment,
            base_job_name=base_job_name,
        )

        train_data = InputData(channel_name="training", data_source=s3_data)
        print("\nStarting training job...")
        print("=" * 60)
        training_job = cast(Any, trainer).train(
            input_data_config=[train_data],
            wait=not args.detach,
            logs=not args.detach,
        )

        job_name = (
            training_job.training_job_name
            if hasattr(training_job, "training_job_name")
            else base_job_name
        )

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
            print(f"ClinVar fine-tuning complete. Final results uploaded to: {s3_results}")
    finally:
        shutil.rmtree(source_dir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
