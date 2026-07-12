#!/usr/bin/env python3
"""Submit regional ClinVar M0 training to SageMaker."""

from __future__ import annotations

import argparse
import os
import re
import secrets
import shutil
import sys
from pathlib import Path
from typing import Any, cast

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.sagemaker_utils import (  # noqa: E402
    DEFAULT_IMAGE_URI,
    DEFAULT_SAGEMAKER_ENTRY_SCRIPT,
    build_sagemaker_entry_environment,
    load_dotenv_if_available,
    package_source,
    resolve_packaged_repo_relative_path,
    split_cli_args,
)

JOB_ENTRYPOINT = Path("scripts/clinvar_m0_job.py")
SAGEMAKER_ENTRYPOINT = Path(DEFAULT_SAGEMAKER_ENTRY_SCRIPT)
DEFAULT_BUCKET = "ai4bio-lumina-experiments-v2"
DEFAULT_DATASET_S3_PREFIX = (
    f"s3://{DEFAULT_BUCKET}/lumina-ssm/data/datasets/clinvar/regional_abraom/slices/"
)
DEFAULT_REFERENCE_S3_PREFIX = f"s3://{DEFAULT_BUCKET}/lumina-ssm/data/hg38/"
DEFAULT_CHECKPOINT_S3_PREFIX = "s3://ai4bio-lumina/releases/lumina-beat-v10-20260527182934/"
JOB_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9](-*[a-zA-Z0-9]){0,62}$")
JOB_BASE_NAME_LIMIT = 42


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    launcher_args, training_args = split_cli_args(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument(
        "--role",
        default=os.environ.get(
            "SAGEMAKER_ROLE",
            "arn:aws:iam::085188779747:role/service-role/AmazonSageMaker-ExecutionRole-20260115T171165",
        ),
    )
    parser.add_argument("--instance-type", default="ml.g5.2xlarge")
    parser.add_argument("--instance-count", type=int, default=1)
    parser.add_argument("--volume-size-gb", type=int, default=400)
    parser.add_argument("--spot", action="store_true")
    parser.add_argument("--max-run-hours", type=int, default=12)
    parser.add_argument("--max-wait-hours", type=int)
    parser.add_argument("--region", default=os.environ.get("AWS_DEFAULT_REGION", "us-east-2"))
    parser.add_argument("--training-image-uri", default=os.environ.get("SAGEMAKER_TRAINING_IMAGE", ""))
    parser.add_argument("--dataset-s3-prefix", default=DEFAULT_DATASET_S3_PREFIX)
    parser.add_argument("--reference-s3-prefix", default=DEFAULT_REFERENCE_S3_PREFIX)
    parser.add_argument("--checkpoint-s3-prefix", default=DEFAULT_CHECKPOINT_S3_PREFIX)
    parser.add_argument("--dataset-file", default="nonbr_only.parquet")
    # Default = Pedro's v10 baseline; pass `--model-family beat-v11 --model-version r1` for the port.
    # These go to the job BEFORE the `--` so clinvar_m0_job.py's parser consumes them (a passthrough
    # after `--` would be clobbered by the job's own _upsert_arg).
    parser.add_argument("--model-family", default="lumina")
    parser.add_argument("--model-version", default="beat-v10")
    parser.add_argument("--detach", action="store_true")
    return parser.parse_args(launcher_args), training_args


def _sanitize(value: str) -> str:
    sanitized = re.sub(r"[^a-z0-9-]+", "-", value.lower())
    sanitized = re.sub(r"-{2,}", "-", sanitized).strip("-")
    return sanitized or "job"


def build_job_name(experiment: str) -> str:
    suffix = secrets.token_hex(3)
    experiment_name = _sanitize(experiment)
    for redundant_prefix in ("clinvar-m0-", "m0-"):
        if experiment_name.startswith(redundant_prefix):
            experiment_name = experiment_name[len(redundant_prefix) :]
            break
    fixed = "clinvar-m0"
    max_experiment_len = JOB_BASE_NAME_LIMIT - len(fixed) - len(suffix) - 2
    experiment_name = experiment_name[:max_experiment_len].rstrip("-") or "job"
    name = f"{fixed}-{experiment_name}-{suffix}"
    if JOB_NAME_PATTERN.fullmatch(name) is None:
        raise ValueError(f"Invalid SageMaker job name: {name!r}")
    return name


def build_environment() -> dict[str, str]:
    return {
        "FI_EFA_FORK_SAFE": "1",
        "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HOME": "/tmp/huggingface",
        "HF_HUB_CACHE": "/tmp/huggingface/hub",
        "HF_MODULES_CACHE": "/tmp/huggingface/modules",
        "INSTALL_TILELANG": "0",
    }


def build_job_args(
    *, dataset_file: str, model_family: str, model_version: str, training_args: list[str]
) -> list[str]:
    job_args = [
        "--dataset-file", dataset_file,
        "--model-family", model_family,
        "--model-version", model_version,
    ]
    if training_args:
        job_args.extend(["--", *training_args])
    return job_args


def main(argv: list[str] | None = None) -> int:
    load_dotenv_if_available()
    args, training_args = parse_args(argv)
    if not args.role:
        raise ValueError("SageMaker execution role is required.")

    try:
        import boto3
        from sagemaker.core.helper.session_helper import Session
        from sagemaker.train import ModelTrainer
        from sagemaker.train.configs import Compute, InputData, OutputDataConfig, SourceCode, StoppingCondition
    except ImportError as exc:
        raise RuntimeError("Install SageMaker launcher dependencies with `uv sync --extra sagemaker`.") from exc

    image_uri = args.training_image_uri.strip() or DEFAULT_IMAGE_URI
    experiment = _sanitize(args.experiment)
    base_job_name = build_job_name(experiment)
    s3_output = f"s3://{args.bucket}/lumina-ssm/clinvar-m0/{experiment}/sagemaker-artifacts/"
    max_wait_hours = args.max_wait_hours if args.max_wait_hours is not None else args.max_run_hours * 2

    print(f"Creating SageMaker ClinVar M0 job: {args.experiment}")
    print(f"  Job base name: {base_job_name}")
    print(f"  Instance: {args.instance_type} x {args.instance_count}")
    print(f"  Dataset channel: {args.dataset_s3_prefix}")
    print(f"  Dataset file: {args.dataset_file}")
    print(f"  Reference channel: {args.reference_s3_prefix}")
    print(f"  Checkpoint channel: {args.checkpoint_s3_prefix}")
    print(f"  Output artifacts: {s3_output}")
    print(f"  Training args: {' '.join(training_args) if training_args else '(defaults)'}")

    source_dir = package_source(include_uncommitted=True)
    try:
        environment = build_sagemaker_entry_environment(
            build_environment(),
            setup_extras="eval,sagemaker,tracking",
            job_script=JOB_ENTRYPOINT,
            job_args=build_job_args(
                dataset_file=args.dataset_file,
                model_family=args.model_family,
                model_version=args.model_version,
                training_args=training_args,
            ),
            source_dir=source_dir,
        )
        stopping_kwargs: dict[str, Any] = {"max_runtime_in_seconds": args.max_run_hours * 3600}
        if args.spot:
            stopping_kwargs["max_wait_time_in_seconds"] = max_wait_hours * 3600

        compute_kwargs: dict[str, Any] = {
            "instance_type": args.instance_type,
            "instance_count": args.instance_count,
            "volume_size_in_gb": args.volume_size_gb,
        }
        if args.spot:
            compute_kwargs["enable_managed_spot_training"] = True

        boto_session = boto3.Session(region_name=args.region)
        sagemaker_session = cast(Any, Session)(boto_session=boto_session)
        trainer = cast(Any, ModelTrainer)(
            training_image=image_uri,
            role=args.role,
            sagemaker_session=sagemaker_session,
            source_code=SourceCode(
                source_dir=source_dir,
                entry_script=resolve_packaged_repo_relative_path(SAGEMAKER_ENTRYPOINT),
            ),
            compute=Compute(**compute_kwargs),
            output_data_config=OutputDataConfig(s3_output_path=s3_output),
            stopping_condition=StoppingCondition(**stopping_kwargs),
            environment=environment,
            base_job_name=base_job_name,
        )
        training_job = trainer.train(
            input_data_config=[
                InputData(channel_name="dataset", data_source=args.dataset_s3_prefix),
                InputData(channel_name="reference", data_source=args.reference_s3_prefix),
                InputData(channel_name="checkpoint", data_source=args.checkpoint_s3_prefix),
            ],
            wait=not args.detach,
            logs=not args.detach,
        )
        job_name = training_job.training_job_name if hasattr(training_job, "training_job_name") else base_job_name
        print(f"job_name={job_name}")
        if args.detach:
            print(
                "monitor="
                f"https://console.aws.amazon.com/sagemaker/home?region={args.region}#/training-jobs/{job_name}"
            )
        return 0
    finally:
        shutil.rmtree(source_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
