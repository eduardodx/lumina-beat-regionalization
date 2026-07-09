#!/usr/bin/env python3
"""SageMaker Training Job Launcher for Lumina SSM (SageMaker SDK V3).

Launches a training job on AWS SageMaker using a pre-built ECR image with
PyTorch + CUDA. The mamba-ssm package is built from source inside the
container via scripts/setup-gpu.sh.

Prerequisites:
    1. AWS CLI configured with credentials (``aws configure``)
    2. Training data uploaded to S3::

           aws s3 sync data/ s3://ai4bio-lumina-experiments-v2/lumina-ssm/data/

    3. SageMaker execution role with appropriate permissions
       (set ``SAGEMAKER_ROLE`` in .env or pass ``--role``)

Local dependencies (not needed in the training container)::

    pip install boto3 sagemaker python-dotenv

Usage::

    python scripts/sagemaker_train.py \\
        --experiment beat-v1-test \\
        --config configs/beat_v1/8m_15ep_32k_b200.yaml \\
        --bucket ai4bio-lumina-experiments-v2

    # Spot instance with detach
    python scripts/sagemaker_train.py \\
        --experiment beat-v1-test \\
        --config configs/beat_v1/8m_15ep_32k_b200.yaml \\
        --bucket ai4bio-lumina-experiments-v2 \\
        --spot --detach
"""

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
    normalize_packaged_repo_path,
    package_source,
    resolve_packaged_repo_relative_path,
)
from src.train import load_yaml_train_config  # noqa: E402

JOB_ENTRYPOINT = Path("scripts/sagemaker_train_job.py")
SAGEMAKER_ENTRYPOINT = Path(DEFAULT_SAGEMAKER_ENTRY_SCRIPT)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch SageMaker training job for Lumina SSM (SDK V3)",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="Experiment name used for S3 output and checkpoint paths.",
    )
    parser.add_argument(
        "--bucket",
        type=str,
        required=True,
        help="S3 bucket for data and outputs.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Local YAML config path (e.g. configs/beat_v1/8m_15ep_32k_b200.yaml).",
    )
    parser.add_argument(
        "--role",
        type=str,
        default=os.environ.get("SAGEMAKER_ROLE", ""),
        help="SageMaker execution role ARN (default: from SAGEMAKER_ROLE env var).",
    )
    parser.add_argument(
        "--instance-type",
        type=str,
        default="ml.p6-b200.48xlarge",
        help="SageMaker instance type (default: ml.p6-b200.48xlarge — 8x B200 GPUs).",
    )
    parser.add_argument(
        "--instance-count",
        type=int,
        default=1,
        help="Number of instances (default: 1).",
    )
    parser.add_argument(
        "--spot",
        action="store_true",
        help="Use managed spot instances (up to 90%% cost savings, may be interrupted).",
    )
    parser.add_argument(
        "--max-run-hours",
        type=int,
        default=72,
        help="Maximum training time in hours (default: 72).",
    )
    parser.add_argument(
        "--max-wait-hours",
        type=int,
        default=None,
        help="Maximum wait time for spot capacity in hours (default: 2x max-run-hours).",
    )
    parser.add_argument(
        "--wandb-key",
        type=str,
        default=os.environ.get("WANDB_API_KEY", ""),
        help="Weights & Biases API key (default: from WANDB_API_KEY env var).",
    )
    parser.add_argument(
        "--training-image-uri",
        type=str,
        default=os.environ.get("SAGEMAKER_TRAINING_IMAGE", ""),
        help="Custom ECR image URI override.",
    )
    parser.add_argument(
        "--detach",
        action="store_true",
        help="Submit job and exit immediately (don't wait for completion).",
    )
    parser.add_argument(
        "--region",
        type=str,
        default=os.environ.get("AWS_DEFAULT_REGION", "us-east-2"),
        help="AWS region (default: from AWS_DEFAULT_REGION or us-east-2).",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Checkpoint path inside the container to resume from.",
    )
    parser.add_argument(
        "--init-checkpoint-s3",
        type=str,
        default=None,
        help="S3 prefix mounted as the checkpoint channel for fine-tuning initialization.",
    )
    parser.add_argument(
        "--init-checkpoint-file",
        type=str,
        default="best_checkpoint.pt",
        help="Checkpoint filename inside --init-checkpoint-s3 (default: best_checkpoint.pt).",
    )
    parser.add_argument(
        "--abraom-data-s3",
        type=str,
        default=None,
        help="S3 prefix for gen-abraom-seqs v2, mounted as the abraom channel.",
    )
    parser.add_argument(
        "--allow-uncommitted-source",
        action="store_true",
        help="Package the filtered working tree instead of git archive HEAD.",
    )
    parser.add_argument(
        "--training-input-mode",
        type=str,
        choices=("auto", "File", "FastFile", "Pipe"),
        default="auto",
        help="SageMaker input mode. auto uses FastFile for ABRAOM jobs and File otherwise.",
    )
    parser.add_argument(
        "--nproc-per-node",
        type=int,
        default=None,
        help="Override torchrun processes per node inside the training container.",
    )
    parser.add_argument(
        "--cuda-launch-blocking",
        action="store_true",
        help="Set CUDA_LAUNCH_BLOCKING=1 in the training container for CUDA traceback diagnostics.",
    )
    return parser.parse_args(argv)


def build_job_args(
    *,
    config_path: str,
    resume_from: str | None,
    abraom_data_root: str | None,
    init_from_checkpoint: str | None,
    nproc_per_node: int | None,
) -> list[str]:
    job_args = [
        "--config-path",
        config_path,
    ]
    if resume_from:
        job_args.extend(["--resume-from", resume_from])
    if abraom_data_root:
        job_args.extend(["--abraom-data-root", abraom_data_root])
    if init_from_checkpoint:
        job_args.extend(["--init-from-checkpoint", init_from_checkpoint])
    if nproc_per_node is not None:
        job_args.extend(["--nproc-per-node", str(nproc_per_node)])
    return job_args


def should_install_flash_attn_for_config(config_path: Path) -> bool:
    config = load_yaml_train_config(config_path)
    return config.get("model") == "beat-v7"


def should_install_tilelang_for_config(config_path: Path) -> bool:
    config = load_yaml_train_config(config_path)
    model_config = config.get("model_config", {})
    if not isinstance(model_config, dict):
        return False
    return bool(model_config.get("is_mimo", False))


def resolve_training_input_mode(mode: str, *, uses_abraom_data: bool) -> str:
    if mode != "auto":
        return mode
    return "FastFile" if uses_abraom_data else "File"


def main(argv: list[str] | None = None) -> int:
    load_dotenv_if_available()
    args = parse_args(argv)

    if not args.role:
        print("Error: SageMaker execution role is required.")
        print("  Set SAGEMAKER_ROLE in .env or pass --role.")
        return 1

    if not SAGEMAKER_ENTRYPOINT.is_file():
        raise FileNotFoundError(f"SageMaker entrypoint not found at {SAGEMAKER_ENTRYPOINT}")
    if not JOB_ENTRYPOINT.is_file():
        raise FileNotFoundError(f"Container job entrypoint not found at {JOB_ENTRYPOINT}")

    if args.resume_from and args.init_checkpoint_s3:
        raise ValueError("--resume-from and --init-checkpoint-s3 are mutually exclusive.")
    local_config_path, packaged_config_path = normalize_packaged_repo_path(
        args.config,
        allow_uncommitted=args.allow_uncommitted_source,
    )
    install_flash_attn = should_install_flash_attn_for_config(local_config_path)
    install_tilelang = should_install_tilelang_for_config(local_config_path)
    training_input_mode = resolve_training_input_mode(
        args.training_input_mode,
        uses_abraom_data=bool(args.abraom_data_s3),
    )

    try:
        import boto3
        from sagemaker.core.helper.session_helper import Session
        from sagemaker.core.shapes.shapes import Channel, DataSource, S3DataSource
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

    # S3 paths.
    s3_prefix = f"s3://{args.bucket}/lumina-ssm"
    s3_data = f"{s3_prefix}/data/"
    s3_output = f"{s3_prefix}/experiments/{args.experiment}/"
    s3_checkpoint = f"{s3_prefix}/checkpoints/{args.experiment}/"

    # Resolve image URI.
    image_uri = args.training_image_uri.strip() or DEFAULT_IMAGE_URI

    # Spot wait time.
    max_wait_hours = args.max_wait_hours
    if args.spot and max_wait_hours is None:
        max_wait_hours = args.max_run_hours * 2

    job_args = build_job_args(
        config_path=packaged_config_path,
        resume_from=args.resume_from,
        abraom_data_root="/opt/ml/input/data/abraom" if args.abraom_data_s3 else None,
        init_from_checkpoint=(
            f"/opt/ml/input/data/checkpoint/{args.init_checkpoint_file}" if args.init_checkpoint_s3 else None
        ),
        nproc_per_node=args.nproc_per_node,
    )

    # ------------------------------------------------------------------ #
    # Environment variables for the training container.
    # ------------------------------------------------------------------ #
    environment = {
        "WANDB_PROJECT": "lumina-ssm",
        "WANDB_ENTITY": "ai4bio-lumina",
        "WANDB_RUN_GROUP": f"sagemaker-{args.experiment}",
        "FI_EFA_FORK_SAFE": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "INSTALL_FLASH_ATTN": "1" if install_flash_attn else "0",
        "INSTALL_TILELANG": "1" if install_tilelang else "0",
    }
    if args.cuda_launch_blocking:
        environment["CUDA_LAUNCH_BLOCKING"] = "1"
    if args.wandb_key:
        environment["WANDB_API_KEY"] = args.wandb_key
    # ------------------------------------------------------------------ #
    # Print summary.
    # ------------------------------------------------------------------ #
    print(f"Creating SageMaker training job for experiment: {args.experiment}")
    print(f"  Instance: {args.instance_type} x {args.instance_count}")
    print(f"  Image: {image_uri}")
    print(f"  Config: {packaged_config_path} (local: {local_config_path})")
    print(f"  Container entrypoint: {SAGEMAKER_ENTRYPOINT} -> {JOB_ENTRYPOINT}")
    print(f"  Install TileLang: {'yes' if install_tilelang else 'no'}")
    print(f"  Install flash-attn: {'yes' if install_flash_attn else 'no'}")
    print(f"  Training input mode: {training_input_mode}")
    if args.nproc_per_node is not None:
        print(f"  torchrun nproc-per-node: {args.nproc_per_node}")
    if args.cuda_launch_blocking:
        print("  CUDA launch blocking: yes")
    print(f"  Data: {s3_data}")
    if args.abraom_data_s3:
        print(f"  ABRAOM data: {args.abraom_data_s3}")
    if args.init_checkpoint_s3:
        print(f"  Init checkpoint: {args.init_checkpoint_s3}/{args.init_checkpoint_file}")
    print(f"  Output: {s3_output}")
    if args.spot:
        print(f"  Spot: ENABLED (max_run={args.max_run_hours}h, max_wait={max_wait_hours}h)")
        print(f"  Checkpoints: {s3_checkpoint}")
    if not args.wandb_key:
        print("  Warning: WANDB_API_KEY not set — W&B logging will be disabled")

    # ------------------------------------------------------------------ #
    # Create SageMaker session and trainer.
    # ------------------------------------------------------------------ #
    boto_session = boto3.Session(region_name=args.region)
    session_cls = cast(Any, Session)
    sagemaker_session = session_cls(boto_session=boto_session)

    stopping_kwargs: dict = {
        "max_runtime_in_seconds": args.max_run_hours * 3600,
    }
    if args.spot and max_wait_hours is not None:
        stopping_kwargs["max_wait_time_in_seconds"] = max_wait_hours * 3600

    compute_kwargs: dict = {
        "instance_type": args.instance_type,
        "instance_count": args.instance_count,
        "volume_size_in_gb": 500,
    }
    if args.spot:
        compute_kwargs["enable_managed_spot_training"] = True

    # Package source (git-tracked files only, excludes data/).
    source_dir = package_source(include_uncommitted=args.allow_uncommitted_source)

    try:
        environment = build_sagemaker_entry_environment(
            environment,
            setup_extras="tracking",
            job_script=JOB_ENTRYPOINT,
            job_args=job_args,
            source_dir=source_dir,
        )
        trainer_kwargs: dict = {
            "training_image": image_uri,
            "role": args.role,
            "sagemaker_session": sagemaker_session,
            "source_code": SourceCode(
                source_dir=source_dir,
                entry_script=resolve_packaged_repo_relative_path(SAGEMAKER_ENTRYPOINT),
            ),
            "compute": Compute(**compute_kwargs),
            "output_data_config": OutputDataConfig(s3_output_path=s3_output),
            "stopping_condition": StoppingCondition(**stopping_kwargs),
            "environment": environment,
            "base_job_name": f"lumina-ssm-{args.experiment}",
            "training_input_mode": training_input_mode,
        }
        if args.spot:
            trainer_kwargs["checkpoint_config"] = CheckpointConfig(
                s3_uri=s3_checkpoint,
                local_path="/opt/ml/checkpoints",
            )

        trainer = cast(Any, ModelTrainer)(**trainer_kwargs)

        def s3_input_channel(channel_name: str, s3_uri: str) -> Any:
            return Channel(
                channel_name=channel_name,
                data_source=DataSource(
                    s3_data_source=S3DataSource(
                        s3_data_type="S3Prefix",
                        s3_uri=s3_uri,
                        s3_data_distribution_type="FullyReplicated",
                    )
                ),
                input_mode=training_input_mode,
            )

        input_data_config = [s3_input_channel("training", s3_data)]
        if args.abraom_data_s3:
            input_data_config.append(s3_input_channel("abraom", args.abraom_data_s3))
        if args.init_checkpoint_s3:
            input_data_config.append(s3_input_channel("checkpoint", args.init_checkpoint_s3))

        # -------------------------------------------------------------- #
        # Launch.
        # -------------------------------------------------------------- #
        print("\nStarting training job...")
        print("=" * 60)

        training_job = cast(Any, trainer).train(
            input_data_config=input_data_config,
            wait=not args.detach,
            logs=not args.detach,
        )

        job_name = (
            training_job.training_job_name
            if hasattr(training_job, "training_job_name")
            else f"lumina-ssm-{args.experiment}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )

        if args.detach:
            print("=" * 60)
            print("Job submitted. You can safely close this terminal.")
            print(f"  Job name: {job_name}")
            print(
                f"  Monitor: https://console.aws.amazon.com/sagemaker/home"
                f"?region={args.region}#/training-jobs/{job_name}"
            )
            print(f"  Output: {s3_output}")
        else:
            print("=" * 60)
            print(f"Training complete. Model artifacts at: {s3_output}")

    finally:
        shutil.rmtree(source_dir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
