#!/usr/bin/env python3
"""Container-side entrypoint for NTv3 benchmark evaluation on SageMaker."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eval.ntv3.config import DEFAULT_MODEL_VERSION
from eval.ntv3.diagnostics import (
    PhaseRecorder,
    collect_dataset_item_tails,
    collect_phase_manifests,
    collect_step_phase_tails,
    latest_phase_manifest,
)
from eval.ntv3.run import _base_config_from_args, _collect_rows, _write_aggregate_csv, build_parser, stage_dataset
from src.sagemaker_utils import SM_DATA, cli_arg_value, parse_s3_uri, split_cli_args

DEFAULT_DATASET_ROOT = Path(SM_DATA) / "datasets" / "ntv3"
DEFAULT_CHECKPOINTS_ROOT = Path(SM_DATA) / "checkpoints"
DEFAULT_RESULTS_ROOT = Path("/opt/ml/checkpoints") / "ntv3-results"
DEFAULT_MODEL_OUTPUT_ROOT = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
DEFAULT_SAGEMAKER_CHECKPOINT_SYNC_ROOT = Path(os.environ.get("SM_CHECKPOINT_DIR", "/sagemaker-ntv3-checkpoints"))
DEFAULT_TILELANG_CACHE_ROOT = Path("/opt/ml/checkpoints") / "tilelang-cache"
DEFAULT_TILELANG_TMP_ROOT = DEFAULT_TILELANG_CACHE_ROOT / "tmp"
DEFAULT_TILELANG_EXECUTION_BACKEND = "tvm_ffi"
DEFAULT_FAILURE_OUTPUT_PATH = Path(os.environ.get("SM_OUTPUT_FAILURE", "/opt/ml/output/failure"))
EXPECTED_TASK_RESULT_FILES = (
    "metrics_train.csv",
    "metrics_val.csv",
    "metrics_test.json",
    "dataset_scores.csv",
    "best_model.pt",
    "run_config.json",
)


@dataclass(frozen=True)
class UploadTarget:
    local_path: Path
    s3_uri: str


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    launcher_args, benchmark_args = split_cli_args(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description="Run the NTv3 benchmark inside SageMaker and upload the final artifacts to S3.",
    )
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--results-s3-prefix", required=True)
    parser.add_argument("--checkpoint-s3-prefix", default=None)
    parser.add_argument("--dataset-repo-id", default="InstaDeepAI/NTv3_benchmark_dataset")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--checkpoints-root", default=str(DEFAULT_CHECKPOINTS_ROOT))
    parser.add_argument("--nproc-per-node", type=int, default=None)
    parser.add_argument(
        "--no-auto-resume",
        action="store_true",
        help="Do not inject --auto-resume into the NTv3 runtime command.",
    )
    return parser.parse_args(launcher_args), benchmark_args


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
    if flag in argv:
        return argv
    return [*argv, flag]


def configure_tilelang_runtime_env(
    *,
    env: dict[str, str] | None = None,
    cache_root: Path = DEFAULT_TILELANG_CACHE_ROOT,
    execution_backend: str = DEFAULT_TILELANG_EXECUTION_BACKEND,
) -> dict[str, str]:
    target_env = os.environ if env is None else env
    target_env.setdefault("TILELANG_CACHE_DIR", str(cache_root))
    target_env.setdefault("TILELANG_TMP_DIR", str(cache_root / "tmp"))
    target_env.setdefault("TILELANG_EXECUTION_BACKEND", execution_backend)

    Path(target_env["TILELANG_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(target_env["TILELANG_TMP_DIR"]).mkdir(parents=True, exist_ok=True)
    return {
        "TILELANG_CACHE_DIR": target_env["TILELANG_CACHE_DIR"],
        "TILELANG_TMP_DIR": target_env["TILELANG_TMP_DIR"],
        "TILELANG_EXECUTION_BACKEND": target_env["TILELANG_EXECUTION_BACKEND"],
    }


def count_cache_artifacts(cache_dir: Path) -> int:
    if not cache_dir.exists():
        return 0
    return sum(1 for path in cache_dir.rglob("*") if path.is_file())


def _default_local_checkpoint_dir(*, checkpoints_root: Path, checkpoint_s3_prefix: str) -> Path:
    checkpoint_path = Path(parse_s3_uri(checkpoint_s3_prefix)[1].rstrip("/"))
    checkpoint_name = checkpoint_path.name
    if checkpoint_name.endswith(".tar.gz"):
        if checkpoint_name == "model.tar.gz" and checkpoint_path.parent.name == "output":
            checkpoint_name = checkpoint_path.parent.parent.name
        else:
            checkpoint_name = checkpoint_name[: -len(".tar.gz")]
    if not checkpoint_name:
        raise ValueError(f"Could not infer checkpoint directory name from {checkpoint_s3_prefix!r}.")
    return checkpoints_root / checkpoint_name


def _sync_s3_prefix_to_local(*, s3_uri: str, local_root: Path) -> None:
    import boto3

    client = boto3.client("s3")
    bucket, key_prefix = parse_s3_uri(s3_uri)
    if key_prefix.endswith(".tar.gz"):
        local_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp_dir:
            archive_path = Path(tmp_dir) / "model.tar.gz"
            client.download_file(bucket, key_prefix, str(archive_path))
            with tarfile.open(archive_path, "r:gz") as archive:
                wanted = {"best_checkpoint.pt", "final_checkpoint.pt", "train_config.json", "run_summary.json"}
                for member in archive.getmembers():
                    member_name = Path(member.name).name
                    if member_name not in wanted or not member.isfile():
                        continue
                    extracted = archive.extractfile(member)
                    if extracted is None:
                        continue
                    (local_root / member_name).write_bytes(extracted.read())
        if not (local_root / "best_checkpoint.pt").is_file():
            raise FileNotFoundError(f"best_checkpoint.pt not found after extracting {s3_uri}")
        return

    normalized_prefix = key_prefix.rstrip("/") + "/"
    paginator = client.get_paginator("list_objects_v2")
    found = False
    for page in paginator.paginate(Bucket=bucket, Prefix=normalized_prefix):
        for entry in page.get("Contents", []):
            key = entry["Key"]
            if key.endswith("/"):
                continue
            found = True
            relative = key[len(normalized_prefix) :].lstrip("/")
            local_path = local_root / relative
            local_path.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(bucket, key, str(local_path))
    if not found:
        raise FileNotFoundError(f"No objects found under {s3_uri}")


def build_runtime_eval_args(
    *,
    experiment: str,
    benchmark_args: list[str],
    dataset_root: Path,
    checkpoint_dir: Path | None,
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
    runtime_args = _upsert_cli_arg(runtime_args, "--dataset-root", str(dataset_root))
    runtime_args = _upsert_cli_arg(runtime_args, "--output-root", str(DEFAULT_RESULTS_ROOT / experiment))
    runtime_args = _upsert_cli_arg(runtime_args, "--run-id", experiment)
    if checkpoint_dir is not None:
        runtime_args = _upsert_cli_arg(runtime_args, "--checkpoint-dir", str(checkpoint_dir))
    return runtime_args


def resolve_nproc_per_node(*, nproc_per_node: int | None = None) -> int:
    resolved = int(os.environ.get("SM_NUM_GPUS", "1")) if nproc_per_node is None else int(nproc_per_node)
    if resolved < 1:
        raise ValueError(f"nproc_per_node must be >= 1, got {resolved}.")
    return resolved


def resolve_world_size(*, nproc_per_node: int | None = None) -> int:
    host_count = max(1, int(os.environ.get("SM_HOST_COUNT", "1")))
    return host_count * resolve_nproc_per_node(nproc_per_node=nproc_per_node)


def build_command(*, runtime_args: list[str], nproc_per_node: int | None = None) -> list[str]:
    return build_torchrun_command(runtime_args=runtime_args, nproc_per_node=nproc_per_node)


def build_torchrun_command(*, runtime_args: list[str], nproc_per_node: int | None = None) -> list[str]:
    host_count = int(os.environ.get("SM_HOST_COUNT", "1"))
    resolved_nproc_per_node = str(resolve_nproc_per_node(nproc_per_node=nproc_per_node))
    base_command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
    ]
    if host_count == 1:
        return [
            *base_command,
            "--standalone",
            "--nproc-per-node",
            resolved_nproc_per_node,
            "-m",
            "eval.ntv3.run",
            "evaluate-species",
            *runtime_args,
        ]

    master_addr = os.environ.get("SM_MASTER_ADDR", "127.0.0.1")
    master_port = os.environ.get("SM_MASTER_PORT", "29500")
    return [
        *base_command,
        "--nnodes",
        str(host_count),
        "--node-rank",
        os.environ.get("SM_CURRENT_HOST_RANK", "0"),
        "--nproc-per-node",
        resolved_nproc_per_node,
        "--rdzv-backend",
        "c10d",
        "--rdzv-endpoint",
        f"{master_addr}:{master_port}",
        "--rdzv-id",
        os.environ.get("TRAINING_JOB_NAME", "lumina-ntv3-rdzv"),
        "-m",
        "eval.ntv3.run",
        "evaluate-species",
        *runtime_args,
    ]


def resolve_runtime_benchmark_config(runtime_args: list[str]) -> tuple[Any, Path]:
    parser = build_parser()
    args = parser.parse_args(["evaluate-species", *runtime_args])
    species_name = cli_arg_value(runtime_args, "--species") or "human"
    task_type = cli_arg_value(runtime_args, "--task-type") or "functional"
    output_root = Path(cli_arg_value(runtime_args, "--output-root") or DEFAULT_RESULTS_ROOT).expanduser().resolve()
    output_dir = output_root / species_name / task_type
    config = _base_config_from_args(args, species_name=species_name, task_type=task_type, output_dir=output_dir)
    return config, output_dir


def _distribution_version(distribution_name: str) -> str | None:
    try:
        return importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _distribution_commit_id(distribution_name: str) -> str | None:
    try:
        distribution = importlib.metadata.distribution(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return None

    direct_url_text = distribution.read_text("direct_url.json")
    if not direct_url_text:
        return None

    try:
        direct_url = json.loads(direct_url_text)
    except json.JSONDecodeError:
        return None
    vcs_info = direct_url.get("vcs_info")
    if not isinstance(vcs_info, dict):
        return None
    commit_id = vcs_info.get("commit_id")
    return str(commit_id) if commit_id else None


def collect_runtime_versions() -> dict[str, Any]:
    import torch

    return {
        "python": sys.version,
        "torch": {
            "version": torch.__version__,
            "cuda": getattr(torch.version, "cuda", None),
            "path": str(Path(torch.__file__).resolve()),
        },
        "mamba_ssm": {
            "version": _distribution_version("mamba-ssm"),
            "commit_id": _distribution_commit_id("mamba-ssm"),
        },
        "tilelang": {
            "version": _distribution_version("tilelang"),
        },
        "causal_conv1d": {
            "version": _distribution_version("causal-conv1d"),
        },
    }


def run_tilelang_preflight(*, runtime_args: list[str], nproc_per_node: int | None = None) -> dict[str, Any]:
    import torch

    from eval.ntv3.adapters import build_ntv3_adapter
    from eval.ntv3.train import DistributedContext, _resolve_resume_checkpoint_path, _resolve_runtime_batch_schedule
    from src.precision import autocast_context, configure_float32_precision, resolve_precision_policy
    from src.tilelang_compat import patch_tilelang_nested_loop_checker_bug

    if not torch.cuda.is_available():
        raise RuntimeError("NTv3 TileLang preflight requires CUDA.")

    # Defensive repeat — idempotent.  The first call happens in main() before
    # heavy imports; re-running here guarantees the patch is in place even if
    # something imports tilelang earlier than expected.  If it returns False,
    # log loudly so CloudWatch surfaces the silent failure instead of crashing
    # later in the TVM FFI.
    if not patch_tilelang_nested_loop_checker_bug():
        print(
            "WARNING: patch_tilelang_nested_loop_checker_bug returned False at preflight; "
            "TileLang semantic checkers may still crash."
        )

    config, output_dir = resolve_runtime_benchmark_config(runtime_args)
    output_dir.mkdir(parents=True, exist_ok=True)
    config.validate()

    tilelang_env = configure_tilelang_runtime_env()
    cache_dir = Path(tilelang_env["TILELANG_CACHE_DIR"]).expanduser().resolve()
    cache_artifacts_before = count_cache_artifacts(cache_dir)

    device = torch.device("cuda", 0)
    configure_float32_precision(config.allow_tf32)
    precision_policy = resolve_precision_policy(device, config.precision)
    preflight_world_size = resolve_world_size(nproc_per_node=nproc_per_node)
    resolved_nproc_per_node = resolve_nproc_per_node(nproc_per_node=nproc_per_node)
    runtime_batch_schedule = _resolve_runtime_batch_schedule(
        config,
        DistributedContext(distributed=preflight_world_size > 1, world_size=preflight_world_size),
    )
    resume_checkpoint_path = _resolve_resume_checkpoint_path(config, output_dir=output_dir, map_location=device)

    torch.manual_seed(config.seed)
    adapter = build_ntv3_adapter(
        model_family=config.model_family,
        model_version=config.model_version,
        checkpoint_path=str(config.resolved_checkpoint_path()),
        device=device,
        precision=precision_policy,
    )
    backbone = adapter.backbone
    is_mimo = bool(adapter.model_config.get("is_mimo", adapter.checkpoint_config.get("is_mimo", False)))
    backbone.train()
    backbone.zero_grad(set_to_none=True)

    input_ids = torch.randint(
        low=1,
        high=5,
        size=(runtime_batch_schedule.per_rank_batch_size, config.sequence_length),
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)

    start_time = time.perf_counter()
    outputs: Any | None = None
    hidden_states: torch.Tensor
    with autocast_context(precision_policy):
        if getattr(config, "feature_source", "hidden") == "decoder":
            extract_sequence_features = getattr(backbone, "extract_sequence_features", None)
            if not callable(extract_sequence_features):
                raise RuntimeError("NTv3 decoder preflight requires backbone.extract_sequence_features().")
            features = extract_sequence_features(input_ids)
            candidate = features.get("decoder_states") if isinstance(features, dict) else None
            if not isinstance(candidate, torch.Tensor):
                raise RuntimeError("NTv3 decoder preflight must return decoder states under 'decoder_states'.")
            hidden_states = candidate
        else:
            encode = getattr(backbone, "encode", None)
            if callable(encode):
                encoded = encode(input_ids)
                if isinstance(encoded, torch.Tensor):
                    hidden_states = encoded
                elif isinstance(encoded, dict) and isinstance(encoded.get("hidden_states"), torch.Tensor):
                    hidden_states = encoded["hidden_states"]
                else:
                    raise RuntimeError(
                        "Lumina backbone preflight encode() must return a tensor or a dict with hidden_states."
                    )
            else:
                outputs = backbone(input_ids=input_ids, attention_mask=attention_mask, return_token_heads=False)
                candidate = outputs.get("hidden_states") if isinstance(outputs, dict) else None
                if not isinstance(candidate, torch.Tensor):
                    raise RuntimeError("Lumina backbone preflight must return hidden states under 'hidden_states'.")
                hidden_states = candidate
        loss = hidden_states.float().square().mean()
    loss.backward()
    torch.cuda.synchronize(device)
    elapsed_seconds = time.perf_counter() - start_time

    cache_artifacts_after = count_cache_artifacts(cache_dir)
    if is_mimo and cache_artifacts_after <= 0:
        raise RuntimeError(
            "TileLang preflight completed without any cache artifacts. "
            f"cache_dir={cache_dir}"
        )

    manifest = {
        "status": "ok",
        "species": config.species_name,
        "task_type": config.task_type,
        "preflight_mode": f"backbone_{config.feature_source}_state_loss",
        "requested_precision": precision_policy.requested,
        "resolved_precision": precision_policy.resolved,
        "sequence_length": config.sequence_length,
        "runtime_batch_size_per_rank": runtime_batch_schedule.per_rank_batch_size,
        "runtime_grad_accum_steps": runtime_batch_schedule.grad_accum_steps,
        "effective_global_batch_size": runtime_batch_schedule.effective_global_batch_size,
        "nproc_per_node": resolved_nproc_per_node,
        "preflight_host_count": max(1, int(os.environ.get("SM_HOST_COUNT", "1"))),
        "preflight_world_size": preflight_world_size,
        "elapsed_seconds": elapsed_seconds,
        "reason": "non_mimo_backbone" if not is_mimo else None,
        "resume_from_checkpoint": str(resume_checkpoint_path) if resume_checkpoint_path is not None else None,
        "tilelang": {
            "execution_backend": tilelang_env["TILELANG_EXECUTION_BACKEND"],
            "cache_dir": str(cache_dir),
            "tmp_dir": tilelang_env["TILELANG_TMP_DIR"],
            "cache_artifacts_before": cache_artifacts_before,
            "cache_artifacts_after": cache_artifacts_after,
            "new_cache_artifacts": max(0, cache_artifacts_after - cache_artifacts_before),
        },
        "model_config": dict(adapter.model_config),
        "checkpoint_config": {
            "seq_len": adapter.checkpoint_config.get("seq_len"),
            "model": adapter.checkpoint_config.get("model"),
        },
        "runtime_versions": collect_runtime_versions(),
    }
    _write_json(output_dir / "preflight_manifest.json", manifest)

    del loss
    del hidden_states
    if outputs is not None:
        del outputs
    del attention_mask
    del input_ids
    del backbone
    del adapter
    torch.cuda.empty_cache()
    return manifest


def join_s3_uri(prefix: str, filename: str) -> str:
    return f"{prefix.rstrip('/')}/{filename}"


def build_upload_targets(
    *,
    output_root: Path,
    species_name: str,
    task_type: str,
    results_s3_prefix: str,
) -> list[UploadTarget]:
    output_dir = output_root / species_name / task_type
    targets: list[UploadTarget] = []
    for filename in EXPECTED_TASK_RESULT_FILES:
        local_path = output_dir / filename
        if not local_path.is_file():
            raise FileNotFoundError(f"Required NTv3 artifact not found at {local_path}")
        targets.append(
            UploadTarget(
                local_path=local_path,
                s3_uri=join_s3_uri(results_s3_prefix, f"{species_name}/{task_type}/{filename}"),
            )
        )

    aggregate_path = output_root / "ntv3_benchmark_results.csv"
    if not aggregate_path.is_file():
        dataset_scores_path = output_dir / "dataset_scores.csv"
        _write_aggregate_csv(aggregate_path, _collect_rows(dataset_scores_path))
    targets.append(
        UploadTarget(
            local_path=aggregate_path,
            s3_uri=join_s3_uri(results_s3_prefix, aggregate_path.name),
        )
    )
    return targets


def materialize_sagemaker_model_output(
    *,
    output_root: Path,
    species_name: str,
    task_type: str,
    model_output_root: Path = DEFAULT_MODEL_OUTPUT_ROOT,
) -> list[Path]:
    source_dir = output_root / species_name / task_type
    model_output_root.mkdir(parents=True, exist_ok=True)

    copied_paths: list[Path] = []
    for filename in EXPECTED_TASK_RESULT_FILES:
        source_path = source_dir / filename
        if not source_path.is_file():
            raise FileNotFoundError(f"Required NTv3 artifact not found at {source_path}")
        destination_path = model_output_root / filename
        shutil.copy2(source_path, destination_path)
        copied_paths.append(destination_path)

    aggregate_path = output_root / "ntv3_benchmark_results.csv"
    if aggregate_path.is_file():
        destination_path = model_output_root / aggregate_path.name
        shutil.copy2(aggregate_path, destination_path)
        copied_paths.append(destination_path)
    return copied_paths


def prepare_sagemaker_checkpoint_resume_dir(
    *,
    output_root: Path,
    experiment: str,
    species_name: str,
    task_type: str,
    checkpoint_sync_root: Path = DEFAULT_SAGEMAKER_CHECKPOINT_SYNC_ROOT,
) -> Path:
    """Expose only training checkpoints through SageMaker managed checkpoint sync."""

    checkpoint_target = checkpoint_sync_root / "ntv3-results" / experiment / species_name / task_type / "checkpoints"
    checkpoint_target.mkdir(parents=True, exist_ok=True)

    output_checkpoint_dir = output_root / species_name / task_type / "checkpoints"
    output_checkpoint_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_checkpoint_dir.is_symlink():
        return checkpoint_target
    if output_checkpoint_dir.exists():
        if any(output_checkpoint_dir.iterdir()):
            return output_checkpoint_dir
        output_checkpoint_dir.rmdir()
    output_checkpoint_dir.symlink_to(checkpoint_target, target_is_directory=True)
    return checkpoint_target


def upload_targets_to_s3(
    targets: list[UploadTarget],
    *,
    s3_client: Any | None = None,
    max_attempts: int = 5,
    initial_backoff_seconds: float = 2.0,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> None:
    client = s3_client
    if client is None:
        import boto3

        client = boto3.client("s3")
    for target in targets:
        bucket, key = parse_s3_uri(target.s3_uri)
        for attempt in range(1, max_attempts + 1):
            try:
                client.upload_file(str(target.local_path), bucket, key)
                break
            except Exception:
                if attempt >= max_attempts:
                    raise
                backoff_seconds = initial_backoff_seconds * (2 ** (attempt - 1))
                print(
                    "upload_retry="
                    f"{target.local_path} attempt={attempt} sleep_seconds={backoff_seconds:.1f}"
                )
                sleep_fn(backoff_seconds)
        print(f"uploaded={target.local_path} s3_uri={target.s3_uri}")


def runtime_metadata(
    *,
    benchmark_args: list[str] | None,
    runtime_args: list[str] | None,
    checkpoint_s3_prefix: str | None,
    nproc_per_node: int | None,
    output_root: Path,
) -> dict[str, Any]:
    args = runtime_args or benchmark_args or []
    return {
        "species": cli_arg_value(args, "--species") or "human",
        "task_type": cli_arg_value(args, "--task-type") or "functional",
        "model_version": cli_arg_value(args, "--model-version") or DEFAULT_MODEL_VERSION,
        "num_workers": cli_arg_value(args, "--num-workers"),
        "checkpoint_dir": cli_arg_value(args, "--checkpoint-dir"),
        "checkpoint_source": checkpoint_s3_prefix or cli_arg_value(args, "--checkpoint-dir"),
        "nproc_per_node": resolve_nproc_per_node(nproc_per_node=nproc_per_node),
        "host_count": max(1, int(os.environ.get("SM_HOST_COUNT", "1"))),
        "configured_world_size": resolve_world_size(nproc_per_node=nproc_per_node),
        "output_root": str(output_root),
    }


def _collect_tails_from_tree(
    root: Path,
    *,
    collector: Callable[[Path], dict[str, list[dict[str, Any]]]],
    max_depth: int = 4,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    tails: dict[str, dict[str, list[dict[str, Any]]]] = {}
    if not root.exists():
        return tails
    candidate_dirs: list[Path] = [root]
    for path in root.rglob("*"):
        if not path.is_dir():
            continue
        try:
            depth = len(path.relative_to(root).parts)
        except ValueError:
            continue
        if depth > max_depth:
            continue
        candidate_dirs.append(path)
    for directory in candidate_dirs:
        events = collector(directory)
        if not events:
            continue
        try:
            key = str(directory.relative_to(root)) or "."
        except ValueError:
            key = str(directory)
        tails[key] = events
    return tails


def _latest_jsonl_event(
    tails_by_dir: dict[str, dict[str, list[dict[str, Any]]]],
) -> dict[str, Any] | None:
    latest: dict[str, Any] | None = None
    latest_ts: str = ""
    for _directory, files in tails_by_dir.items():
        for _filename, events in files.items():
            for event in events:
                ts = str(event.get("timestamp", ""))
                if ts >= latest_ts:
                    latest_ts = ts
                    latest = event
    return latest


def write_failure_report(
    *,
    experiment: str,
    exc: BaseException,
    benchmark_args: list[str] | None = None,
    runtime_args: list[str] | None = None,
    output_root: Path | None = None,
    checkpoint_s3_prefix: str | None = None,
    nproc_per_node: int | None = None,
) -> None:
    summary = f"{type(exc).__name__}: {exc}"
    failure_output_path = DEFAULT_FAILURE_OUTPUT_PATH
    resolved_output_root = output_root or (DEFAULT_RESULTS_ROOT / experiment)
    traceback_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    phase_manifests = collect_phase_manifests(resolved_output_root)
    latest_manifest = phase_manifests.get("phase_manifest.json") or latest_phase_manifest(resolved_output_root) or {}
    step_phase_tails = _collect_tails_from_tree(resolved_output_root, collector=collect_step_phase_tails)
    dataset_item_tails = _collect_tails_from_tree(resolved_output_root, collector=collect_dataset_item_tails)
    last_step_event = _latest_jsonl_event(step_phase_tails)
    last_dataset_event = _latest_jsonl_event(dataset_item_tails)
    payload = {
        "experiment": experiment,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "benchmark_args": benchmark_args or [],
        "runtime_args": runtime_args or [],
        "runtime_metadata": runtime_metadata(
            benchmark_args=benchmark_args,
            runtime_args=runtime_args,
            checkpoint_s3_prefix=checkpoint_s3_prefix,
            nproc_per_node=nproc_per_node,
            output_root=resolved_output_root,
        ),
        "last_phase": latest_manifest.get("last_phase"),
        "phase_status": latest_manifest.get("status"),
        "rank": latest_manifest.get("rank"),
        "world_size": latest_manifest.get("world_size"),
        "phase_manifests": phase_manifests,
        "step_phase_tails": step_phase_tails,
        "dataset_item_tails": dataset_item_tails,
        "last_step_phase_event": last_step_event,
        "last_dataset_item_event": last_dataset_event,
        "output_root": str(resolved_output_root),
        "traceback": traceback_text,
    }
    if isinstance(exc, subprocess.CalledProcessError):
        payload["returncode"] = exc.returncode
        payload["cmd"] = exc.cmd
    try:
        report_path = _write_json(resolved_output_root / "failure_report.json", payload)
    except Exception:
        report_path = None

    try:
        failure_output_path.parent.mkdir(parents=True, exist_ok=True)
        message = summary
        if report_path is not None:
            message += f"\nDetailed report: {report_path}"
        failure_output_path.write_text(message + "\n", encoding="utf-8")
    except Exception:
        pass


def install_signal_handlers(
    *,
    experiment: str,
    benchmark_args: list[str],
    state: dict[str, Any],
) -> None:
    def _handle_signal(signum: int, _frame: Any) -> None:
        signal_name = signal.Signals(signum).name
        phase_recorder = state.get("phase_recorder")
        if isinstance(phase_recorder, PhaseRecorder):
            phase_recorder.mark(
                "signal_received",
                status="failed",
                details={"signal": signal_name},
                primary=True,
            )
        exc = RuntimeError(f"Received signal {signal_name}")
        write_failure_report(
            experiment=experiment,
            exc=exc,
            benchmark_args=benchmark_args,
            runtime_args=state.get("runtime_args"),
            output_root=state.get("output_root"),
            checkpoint_s3_prefix=state.get("checkpoint_s3_prefix"),
            nproc_per_node=state.get("nproc_per_node"),
        )
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)


def main(argv: list[str] | None = None) -> int:
    args, benchmark_args = parse_args(argv)
    runtime_args: list[str] | None = None
    output_root = DEFAULT_RESULTS_ROOT / args.experiment
    phase_recorder = PhaseRecorder(
        output_dir=output_root,
        metadata={
            "experiment": args.experiment,
            "dataset_repo_id": args.dataset_repo_id,
        },
    )
    signal_state: dict[str, Any] = {
        "phase_recorder": phase_recorder,
        "runtime_args": runtime_args,
        "output_root": output_root,
        "checkpoint_s3_prefix": args.checkpoint_s3_prefix,
        "nproc_per_node": args.nproc_per_node,
    }
    install_signal_handlers(
        experiment=args.experiment,
        benchmark_args=benchmark_args,
        state=signal_state,
    )
    try:
        configure_tilelang_runtime_env()
        phase_recorder.mark("tilelang_env_configured", status="in_progress", primary=True)

        from src.tilelang_compat import patch_tilelang_nested_loop_checker_bug
        if not patch_tilelang_nested_loop_checker_bug():
            print(
                "WARNING: patch_tilelang_nested_loop_checker_bug returned False at main(); "
                "TileLang semantic-checker crash may surface during kernel compilation."
            )
        phase_recorder.mark("tilelang_patch_ready", primary=True)

        dataset_root = Path(args.dataset_root).expanduser().resolve()
        checkpoints_root = Path(args.checkpoints_root).expanduser().resolve()
        checkpoint_dir: Path | None = None

        if args.checkpoint_s3_prefix:
            phase_recorder.mark(
                "checkpoint_staging_started",
                status="in_progress",
                details={"checkpoint_source": args.checkpoint_s3_prefix},
                primary=True,
            )
            checkpoint_dir = _default_local_checkpoint_dir(
                checkpoints_root=checkpoints_root,
                checkpoint_s3_prefix=args.checkpoint_s3_prefix,
            )
            _sync_s3_prefix_to_local(s3_uri=args.checkpoint_s3_prefix, local_root=checkpoint_dir)
            phase_recorder.mark(
                "checkpoint_staging_completed",
                details={"checkpoint_dir": str(checkpoint_dir)},
                primary=True,
            )

        species_name = cli_arg_value(benchmark_args, "--species") or "human"
        task_type = cli_arg_value(benchmark_args, "--task-type") or "functional"
        phase_recorder.mark(
            "dataset_staging_started",
            status="in_progress",
            details={"species": species_name, "task_type": task_type},
            primary=True,
        )
        stage_dataset(
            repo_id=args.dataset_repo_id,
            dataset_root=dataset_root,
            species_names=[species_name],
            include_functional=task_type == "functional",
            include_annotation=task_type == "annotation",
        )
        phase_recorder.mark(
            "dataset_staging_completed",
            details={"dataset_root": str(dataset_root)},
            primary=True,
        )

        runtime_args = build_runtime_eval_args(
            experiment=args.experiment,
            benchmark_args=benchmark_args,
            dataset_root=dataset_root,
            checkpoint_dir=checkpoint_dir,
            auto_resume=not args.no_auto_resume,
        )
        signal_state["runtime_args"] = runtime_args
        species_name = cli_arg_value(runtime_args, "--species") or species_name
        task_type = cli_arg_value(runtime_args, "--task-type") or task_type
        output_root = Path(cli_arg_value(runtime_args, "--output-root") or (DEFAULT_RESULTS_ROOT / args.experiment))
        output_root.mkdir(parents=True, exist_ok=True)
        if output_root != phase_recorder.output_dir:
            phase_recorder = PhaseRecorder(output_dir=output_root, metadata=phase_recorder.metadata)
            signal_state["phase_recorder"] = phase_recorder
        signal_state["output_root"] = output_root
        metadata = runtime_metadata(
            benchmark_args=benchmark_args,
            runtime_args=runtime_args,
            checkpoint_s3_prefix=args.checkpoint_s3_prefix,
            nproc_per_node=args.nproc_per_node,
            output_root=output_root,
        )
        phase_recorder.update_metadata(**metadata)
        phase_recorder.mark("runtime_args_resolved", details=metadata, primary=True)
        checkpoint_resume_dir = prepare_sagemaker_checkpoint_resume_dir(
            output_root=output_root,
            experiment=args.experiment,
            species_name=species_name,
            task_type=task_type,
        )
        phase_recorder.mark(
            "checkpoint_resume_dir_ready",
            details={"path": str(checkpoint_resume_dir)},
            primary=True,
        )

        phase_recorder.mark("preflight_started", status="in_progress", primary=True)
        preflight_manifest = run_tilelang_preflight(runtime_args=runtime_args, nproc_per_node=args.nproc_per_node)
        phase_recorder.mark(
            "preflight_completed",
            details={
                "status": preflight_manifest["status"],
                "reason": preflight_manifest.get("reason"),
                "preflight_world_size": preflight_manifest.get("preflight_world_size"),
            },
            primary=True,
        )
        if preflight_manifest["status"] == "ok":
            print(
                "tilelang_preflight="
                f"ok cache_dir={preflight_manifest['tilelang']['cache_dir']} "
                f"artifacts={preflight_manifest['tilelang']['cache_artifacts_after']}"
            )
        else:
            print(
                "tilelang_preflight="
                f"{preflight_manifest['status']} reason={preflight_manifest.get('reason', 'unknown')}"
            )

        command = build_command(runtime_args=runtime_args, nproc_per_node=args.nproc_per_node)
        phase_recorder.mark(
            "torchrun_started",
            status="in_progress",
            details={
                "nproc_per_node": resolve_nproc_per_node(nproc_per_node=args.nproc_per_node),
                "configured_world_size": resolve_world_size(nproc_per_node=args.nproc_per_node),
            },
            primary=True,
        )
        print(f"running={shlex.join(command)}")
        subprocess.run(command, check=True)
        phase_recorder.mark("torchrun_completed", primary=True)

        phase_recorder.mark("artifact_upload_started", status="in_progress", primary=True)
        upload_targets = build_upload_targets(
            output_root=output_root,
            species_name=species_name,
            task_type=task_type,
            results_s3_prefix=args.results_s3_prefix,
        )
        phase_recorder.mark("model_output_materialization_started", status="in_progress", primary=True)
        copied_model_paths = materialize_sagemaker_model_output(
            output_root=output_root,
            species_name=species_name,
            task_type=task_type,
        )
        phase_recorder.mark(
            "model_output_materialization_completed",
            details={"num_model_files": len(copied_model_paths)},
            primary=True,
        )
        upload_targets_to_s3(upload_targets)
        phase_recorder.mark(
            "artifact_upload_completed",
            details={"num_upload_targets": len(upload_targets)},
            primary=True,
        )
        phase_recorder.mark("job_completed", primary=True)
        return 0
    except Exception as exc:
        phase_recorder.mark(
            "job_failed",
            status="failed",
            details={"error_type": type(exc).__name__, "error_message": str(exc)},
            primary=True,
        )
        write_failure_report(
            experiment=args.experiment,
            exc=exc,
            benchmark_args=benchmark_args,
            runtime_args=runtime_args,
            output_root=output_root,
            checkpoint_s3_prefix=args.checkpoint_s3_prefix,
            nproc_per_node=args.nproc_per_node,
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
