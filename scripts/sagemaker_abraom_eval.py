#!/usr/bin/env python3
"""Launch SageMaker ABRAOM regionalization evaluation."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from datetime import datetime
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
)

JOB_ENTRYPOINT = Path("scripts/sagemaker_abraom_eval_job.py")
SAGEMAKER_ENTRYPOINT = Path(DEFAULT_SAGEMAKER_ENTRY_SCRIPT)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--role", default=os.environ.get("SAGEMAKER_ROLE", ""))
    parser.add_argument("--region", default=os.environ.get("AWS_DEFAULT_REGION", "us-east-2"))
    parser.add_argument("--instance-type", default="ml.p5.48xlarge")
    parser.add_argument("--instance-count", type=int, default=1)
    parser.add_argument("--spot", action="store_true")
    parser.add_argument("--max-run-hours", type=int, default=4)
    parser.add_argument("--max-wait-hours", type=int, default=None)
    parser.add_argument("--training-image-uri", default=os.environ.get("SAGEMAKER_TRAINING_IMAGE", ""))
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--allow-uncommitted-source", action="store_true")
    parser.add_argument("--abraom-data-s3", required=True)
    parser.add_argument("--base-checkpoint-s3", required=True)
    parser.add_argument("--candidate-model-s3", required=True)
    parser.add_argument("--candidate-checkpoint-file", default="best_checkpoint.pt")
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--progress-every", type=int, default=50)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    load_dotenv_if_available()
    args = parse_args(argv)
    if not args.role:
        raise RuntimeError("SageMaker execution role is required. Set SAGEMAKER_ROLE or pass --role.")

    try:
        import boto3
        from sagemaker.core.helper.session_helper import Session
        from sagemaker.core.shapes.shapes import Channel, DataSource, S3DataSource
        from sagemaker.train import ModelTrainer
        from sagemaker.train.configs import Compute, OutputDataConfig, SourceCode, StoppingCondition
    except ImportError as exc:
        raise RuntimeError("Install SageMaker launcher dependencies with `uv sync --extra sagemaker`.") from exc

    max_wait_hours = args.max_wait_hours
    if args.spot and max_wait_hours is None:
        max_wait_hours = args.max_run_hours * 2

    s3_prefix = f"s3://{args.bucket}/lumina-ssm"
    s3_output = f"{s3_prefix}/evaluations/{args.experiment}/"
    image_uri = args.training_image_uri.strip() or DEFAULT_IMAGE_URI

    job_args = [
        "--base-checkpoint",
        "/opt/ml/input/data/base/best_checkpoint.pt",
        "--candidate-root",
        "/opt/ml/input/data/candidate",
        "--candidate-checkpoint-file",
        args.candidate_checkpoint_file,
        "--pairs-parquet",
        "/opt/ml/input/data/abraom/eval_ref_alt_pairs/eval_ref_alt_pairs.parquet",
        "--output-dir",
        "/opt/ml/model",
        "--split",
        args.split,
        "--batch-size",
        str(args.batch_size),
        "--device",
        "cuda",
        "--progress-every",
        str(args.progress_every),
    ]
    if args.limit is not None:
        job_args.extend(["--limit", str(args.limit)])

    environment = {
        "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "INSTALL_FLASH_ATTN": "0",
        "INSTALL_TILELANG": "0",
    }

    print(f"Creating SageMaker ABRAOM evaluation job: {args.experiment}")
    print(f"  Instance: {args.instance_type} x {args.instance_count}")
    print(f"  Image: {image_uri}")
    print(f"  ABRAOM data: {args.abraom_data_s3}")
    print(f"  Base checkpoint: {args.base_checkpoint_s3}")
    print(f"  Candidate model: {args.candidate_model_s3}")
    print(f"  Candidate checkpoint file: {args.candidate_checkpoint_file}")
    print(f"  Split: {args.split}")
    print(f"  Limit: {args.limit if args.limit is not None else 'all'}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Output: {s3_output}")
    if args.spot:
        print(f"  Spot: ENABLED (max_run={args.max_run_hours}h, max_wait={max_wait_hours}h)")

    boto_session = boto3.Session(region_name=args.region)
    sagemaker_session = cast(Any, Session)(boto_session=boto_session)
    source_dir = package_source(include_uncommitted=args.allow_uncommitted_source)

    try:
        environment = build_sagemaker_entry_environment(
            environment,
            setup_extras="tracking",
            job_script=JOB_ENTRYPOINT,
            job_args=job_args,
            source_dir=source_dir,
        )
        compute_kwargs: dict[str, Any] = {
            "instance_type": args.instance_type,
            "instance_count": args.instance_count,
            "volume_size_in_gb": 500,
        }
        if args.spot:
            compute_kwargs["enable_managed_spot_training"] = True

        stopping_kwargs: dict[str, Any] = {"max_runtime_in_seconds": args.max_run_hours * 3600}
        if args.spot and max_wait_hours is not None:
            stopping_kwargs["max_wait_time_in_seconds"] = max_wait_hours * 3600

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
            base_job_name=f"lumina-ssm-{args.experiment}",
            training_input_mode="FastFile",
        )

        def s3_channel(name: str, uri: str) -> Any:
            return Channel(
                channel_name=name,
                data_source=DataSource(
                    s3_data_source=S3DataSource(
                        s3_data_type="S3Prefix",
                        s3_uri=uri,
                        s3_data_distribution_type="FullyReplicated",
                    )
                ),
                input_mode="FastFile",
            )

        training_job = cast(Any, trainer).train(
            input_data_config=[
                s3_channel("abraom", args.abraom_data_s3),
                s3_channel("base", args.base_checkpoint_s3),
                s3_channel("candidate", args.candidate_model_s3),
            ],
            wait=not args.detach,
            logs=not args.detach,
        )
        job_name = (
            training_job.training_job_name
            if hasattr(training_job, "training_job_name")
            else f"lumina-ssm-{args.experiment}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )
        if args.detach:
            print("Job submitted.")
            print(f"  Job name: {job_name}")
            print(f"  Output: {s3_output}")
        else:
            print(f"Evaluation complete. Artifacts at: {s3_output}")
    finally:
        shutil.rmtree(source_dir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
