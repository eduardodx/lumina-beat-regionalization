#!/usr/bin/env python3
"""Container-side entrypoint for ClinVar SageMaker fine-tuning."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.sagemaker_utils import SM_DATA, cli_arg_value, parse_s3_uri, split_cli_args

DEFAULT_DATASET_PATH = f"{SM_DATA}/datasets/clinvar/processed/clinvar_dataset.parquet"
DEFAULT_FASTA_PATH = f"{SM_DATA}/hg38/hg38.fa"
DEFAULT_PHYLO100_BW_PATH = f"{SM_DATA}/phylo/hg38.phyloP100way.bw"
DEFAULT_PHYLO470_BW_PATH = f"{SM_DATA}/phylo/hg38.phyloP470way.bw"
DEFAULT_GTF_PATH = f"{SM_DATA}/gencode/gencode.v38.annotation.gtf.gz"
DEFAULT_RESULTS_ROOT = Path("/tmp/clinvar-results")
DEFAULT_CACHE_ROOT = Path("/tmp/clinvar-cache")
DEFAULT_HUGGINGFACE_HOME = Path("/tmp/huggingface")
DEFAULT_HUGGINGFACE_HUB_CACHE = DEFAULT_HUGGINGFACE_HOME / "hub"
DEFAULT_HUGGINGFACE_MODULES_CACHE = DEFAULT_HUGGINGFACE_HOME / "modules"
REQUIRED_DATA_PREFIXES = (
    "datasets/clinvar/",
    "hg38/",
    "gencode/",
    "phylo/",
)
EXPECTED_RESULT_FILES = (
    "metrics.json",
    "test_predictions.parquet",
    "best_model.pt",
)


@dataclass(frozen=True)
class UploadTarget:
    local_path: Path
    s3_uri: str


@dataclass(frozen=True)
class HuggingFaceCachePaths:
    home: Path
    hub: Path
    modules: Path


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    launcher_args, finetune_args = split_cli_args(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description="Run ClinVar fine-tuning inside SageMaker and upload final artifacts to S3.",
    )
    parser.add_argument("--experiment", required=True, help="Experiment identifier used for local scratch paths.")
    parser.add_argument(
        "--results-s3-prefix",
        required=True,
        help="S3 prefix where the final metrics, test parquet, and model bundle should be uploaded.",
    )
    parser.add_argument(
        "--data-s3-prefix",
        required=True,
        help="S3 prefix used to self-stage ClinVar data assets inside the container.",
    )
    parser.add_argument(
        "--checkpoint-s3-prefix",
        default=None,
        help="Optional S3 prefix for self-staging a Lumina checkpoint bundle before fine-tuning.",
    )
    parser.add_argument(
        "--huggingface-cache-s3-prefix",
        default=None,
        help="Optional S3 prefix for self-staging the Hugging Face hub cache before fine-tuning.",
    )
    return parser.parse_args(launcher_args), finetune_args


def resolve_huggingface_cache_paths(*, env: Mapping[str, str] | None = None) -> HuggingFaceCachePaths:
    cache_env = os.environ if env is None else env
    home = Path(cache_env.get("HF_HOME", str(DEFAULT_HUGGINGFACE_HOME)))
    default_hub = DEFAULT_HUGGINGFACE_HUB_CACHE if home == DEFAULT_HUGGINGFACE_HOME else home / "hub"
    hub = Path(cache_env.get("HF_HUB_CACHE", str(default_hub)))
    modules = Path(
        cache_env.get(
            "HF_MODULES_CACHE",
            str(DEFAULT_HUGGINGFACE_MODULES_CACHE if home == DEFAULT_HUGGINGFACE_HOME else home / "modules"),
        )
    )
    return HuggingFaceCachePaths(home=home, hub=hub, modules=modules)


def bootstrap_huggingface_cache_dirs(
    *,
    env: MutableMapping[str, str] | None = None,
) -> HuggingFaceCachePaths:
    cache_env = os.environ if env is None else env
    cache_paths = resolve_huggingface_cache_paths(env=cache_env)
    cache_env.setdefault("HF_HOME", str(cache_paths.home))
    cache_env.setdefault("HF_HUB_CACHE", str(cache_paths.hub))
    cache_env.setdefault("HF_MODULES_CACHE", str(cache_paths.modules))
    for path in (cache_paths.home, cache_paths.hub, cache_paths.modules):
        path.mkdir(parents=True, exist_ok=True)
    print(
        "huggingface_cache_paths="
        f"home={cache_paths.home} hub={cache_paths.hub} modules={cache_paths.modules}"
    )
    return cache_paths


def sync_s3_prefix_to_local(
    *,
    s3_client: Any,
    bucket: str,
    key_prefix: str,
    local_root: Path,
) -> dict[str, int | str]:
    normalized_prefix = key_prefix.rstrip("/") + "/"
    paginator = s3_client.get_paginator("list_objects_v2")
    downloaded = 0
    skipped = 0
    found = False

    for page in paginator.paginate(Bucket=bucket, Prefix=normalized_prefix):
        for entry in page.get("Contents", []):
            key = entry["Key"]
            size = int(entry.get("Size", 0))
            if key.endswith("/"):
                continue
            found = True
            relative = key[len(normalized_prefix) :].lstrip("/")
            local_path = local_root / relative
            if local_path.is_file() and local_path.stat().st_size == size:
                skipped += 1
                continue
            local_path.parent.mkdir(parents=True, exist_ok=True)
            s3_client.download_file(bucket, key, str(local_path))
            downloaded += 1

    if not found:
        raise FileNotFoundError(f"No objects found under s3://{bucket}/{normalized_prefix}")

    print(
        "synced_prefix="
        f"s3://{bucket}/{normalized_prefix} local_root={local_root} downloaded={downloaded} skipped={skipped}"
    )
    return {
        "bucket": bucket,
        "key_prefix": normalized_prefix,
        "downloaded": downloaded,
        "skipped": skipped,
    }


def sync_required_data_from_s3(
    *,
    data_s3_prefix: str,
    local_root: Path = Path(SM_DATA),
    s3_client: Any | None = None,
) -> list[dict[str, int | str]]:
    if s3_client is None:
        import boto3

        client = boto3.client("s3")
    else:
        client = s3_client
    bucket, key_prefix = parse_s3_uri(data_s3_prefix)
    normalized_base_prefix = key_prefix.rstrip("/")
    results: list[dict[str, int | str]] = []
    for relative_prefix in REQUIRED_DATA_PREFIXES:
        full_prefix = f"{normalized_base_prefix}/{relative_prefix}"
        results.append(
            sync_s3_prefix_to_local(
                s3_client=client,
                bucket=bucket,
                key_prefix=full_prefix,
                local_root=local_root / relative_prefix,
            )
        )
    return results


def sync_huggingface_cache_from_s3(
    *,
    huggingface_cache_s3_prefix: str,
    cache_paths: HuggingFaceCachePaths,
    s3_client: Any,
) -> dict[str, int | str]:
    bucket, key_prefix = parse_s3_uri(huggingface_cache_s3_prefix)
    return sync_s3_prefix_to_local(
        s3_client=s3_client,
        bucket=bucket,
        key_prefix=key_prefix,
        local_root=cache_paths.hub,
    )


def build_runtime_finetune_args(*, experiment: str, finetune_args: list[str]) -> list[str]:
    runtime_args = list(finetune_args)
    if cli_arg_value(runtime_args, "--dataset-path") is None:
        runtime_args.extend(["--dataset-path", DEFAULT_DATASET_PATH])
    if cli_arg_value(runtime_args, "--fasta-path") is None:
        runtime_args.extend(["--fasta-path", DEFAULT_FASTA_PATH])
    regime = (cli_arg_value(runtime_args, "--regime") or "A").strip().upper()
    if regime == "B":
        # Regime B requires the biological-reference assets already staged into SM_DATA.
        if cli_arg_value(runtime_args, "--phylo100-bw-path") is None:
            runtime_args.extend(["--phylo100-bw-path", DEFAULT_PHYLO100_BW_PATH])
        if cli_arg_value(runtime_args, "--phylo470-bw-path") is None:
            runtime_args.extend(["--phylo470-bw-path", DEFAULT_PHYLO470_BW_PATH])
        if cli_arg_value(runtime_args, "--gtf-path") is None:
            runtime_args.extend(["--gtf-path", DEFAULT_GTF_PATH])
    runtime_args.extend(["--output-dir", str(DEFAULT_RESULTS_ROOT / experiment)])
    runtime_args.extend(["--cache-dir", str(DEFAULT_CACHE_ROOT / experiment)])
    return runtime_args


def build_torchrun_command(*, finetune_args: list[str]) -> list[str]:
    master_addr = os.environ.get("SM_MASTER_ADDR", "127.0.0.1")
    master_port = os.environ.get("SM_MASTER_PORT", "29500")
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nnodes",
        os.environ.get("SM_HOST_COUNT", "1"),
        "--node-rank",
        os.environ.get("SM_CURRENT_HOST_RANK", "0"),
        "--nproc-per-node",
        os.environ.get("SM_NUM_GPUS", "1"),
        "--rdzv-backend",
        "c10d",
        "--rdzv-endpoint",
        f"{master_addr}:{master_port}",
        "--rdzv-id",
        os.environ.get("TRAINING_JOB_NAME", "lumina-clinvar-rdzv"),
        "-m",
        "eval.clinvar.run",
        *finetune_args,
    ]


def resolve_huggingface_repo(*, model_family: str, model_version: str) -> str | None:
    from eval.clinvar.adapters import (
        CADUCEUS_REPOS,
        DNABERT2_REPOS,
        NTV3_REPOS,
        normalize_finetune_model_family,
    )

    family = normalize_finetune_model_family(model_family)
    if family == "lumina":
        return None

    repo_maps = {
        "caduceus": CADUCEUS_REPOS,
        "dnabert2": DNABERT2_REPOS,
        "ntv3": NTV3_REPOS,
    }
    repo_map = repo_maps.get(family)
    if repo_map is None:
        return None
    repo = repo_map.get(model_version)
    if repo is None:
        raise ValueError(
            f"Unknown Hugging Face-backed {model_family!r} version {model_version!r}. "
            f"Available: {sorted(repo_map)}"
        )
    return repo


def _validate_sharded_weight_index(snapshot_path: Path, index_filename: str) -> bool:
    index_path = snapshot_path / index_filename
    if not index_path.is_file():
        return False
    with index_path.open("r", encoding="utf-8") as handle:
        index_data = json.load(handle)
    weight_map = index_data.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise RuntimeError(f"Sharded weight index {index_path} is missing a non-empty weight_map.")
    missing_shards = sorted(
        {
            str(snapshot_path / shard_name)
            for shard_name in weight_map.values()
            if not (snapshot_path / shard_name).is_file()
        }
    )
    if missing_shards:
        raise RuntimeError(
            f"Sharded weight index {index_path} references missing files: {', '.join(missing_shards)}"
        )
    return True


def _validate_huggingface_snapshot_contents(*, model_family: str, repo: str, snapshot_path: Path) -> None:
    from eval.clinvar.adapters import normalize_finetune_model_family

    config_path = snapshot_path / "config.json"
    if not config_path.is_file():
        raise RuntimeError(f"Required config.json for {repo} was not found under {snapshot_path}.")

    family = normalize_finetune_model_family(model_family)
    if family == "caduceus":
        weights_path = snapshot_path / "model.safetensors"
        if not weights_path.is_file():
            raise RuntimeError(f"Required Caduceus weights were not found at {weights_path}.")
        return

    if family == "dnabert2":
        if (snapshot_path / "model.safetensors").is_file() or (snapshot_path / "pytorch_model.bin").is_file():
            return
        raise RuntimeError(
            f"DNABERT-2 weights for {repo} are missing from {snapshot_path}; "
            "expected model.safetensors or pytorch_model.bin."
        )

    if family == "ntv3":
        if (snapshot_path / "model.safetensors").is_file() or (snapshot_path / "pytorch_model.bin").is_file():
            return
        if _validate_sharded_weight_index(snapshot_path, "model.safetensors.index.json"):
            return
        if _validate_sharded_weight_index(snapshot_path, "pytorch_model.bin.index.json"):
            return
        raise RuntimeError(
            "NTv3 weights are missing from "
            f"{snapshot_path}; expected model.safetensors, pytorch_model.bin, or a valid sharded index."
        )


def _default_tokenizer_loader(repo: str, cache_dir: str) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        repo,
        cache_dir=cache_dir,
        local_files_only=True,
        trust_remote_code=True,
    )


def _default_config_loader(repo: str, cache_dir: str) -> Any:
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(
        repo,
        cache_dir=cache_dir,
        local_files_only=True,
        trust_remote_code=True,
    )


def validate_huggingface_backbone_cache(
    *,
    finetune_args: list[str],
    cache_paths: HuggingFaceCachePaths,
    huggingface_cache_s3_prefix: str | None = None,
    snapshot_download_fn: Callable[..., str] | None = None,
    tokenizer_loader: Callable[[str, str], Any] | None = None,
    config_loader: Callable[[str, str], Any] | None = None,
) -> str | None:
    model_family = cli_arg_value(finetune_args, "--model-family")
    model_version = cli_arg_value(finetune_args, "--model-version")
    if not model_family or not model_version:
        raise ValueError("ClinVar fine-tune args must include --model-family and --model-version.")

    repo = resolve_huggingface_repo(model_family=model_family, model_version=model_version)
    if repo is None:
        print(f"skipping_huggingface_cache_validation model_family={model_family}")
        return None

    if snapshot_download_fn is None:
        from huggingface_hub import snapshot_download

        snapshot_download_fn = snapshot_download
    if tokenizer_loader is None:
        tokenizer_loader = _default_tokenizer_loader
    if config_loader is None:
        config_loader = _default_config_loader

    try:
        snapshot_path = Path(
            snapshot_download_fn(
                repo_id=repo,
                cache_dir=str(cache_paths.hub),
                local_files_only=True,
            )
        )
    except Exception as exc:
        source_hint = (
            f" after syncing {huggingface_cache_s3_prefix}"
            if huggingface_cache_s3_prefix is not None
            else ""
        )
        raise RuntimeError(
            f"Hugging Face repo {repo} is not available in the local cache at {cache_paths.hub}{source_hint}."
        ) from exc

    try:
        tokenizer_loader(repo, str(cache_paths.hub))
        config_loader(repo, str(cache_paths.hub))
        _validate_huggingface_snapshot_contents(
            model_family=model_family,
            repo=repo,
            snapshot_path=snapshot_path,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Hugging Face repo {repo} could not be loaded fully offline from {cache_paths.hub}: {exc}"
        ) from exc

    print(f"validated_huggingface_repo={repo} snapshot={snapshot_path}")
    return repo


def join_s3_uri(prefix: str, filename: str) -> str:
    return f"{prefix.rstrip('/')}/{filename}"


def is_primary_host() -> bool:
    return os.environ.get("SM_CURRENT_HOST_RANK", "0") == "0"


def build_upload_targets(
    *,
    output_dir: Path,
    results_s3_prefix: str,
) -> list[UploadTarget]:
    targets: list[UploadTarget] = []
    for filename in EXPECTED_RESULT_FILES:
        local_path = output_dir / filename
        if not local_path.is_file():
            raise FileNotFoundError(f"Required ClinVar artifact not found at {local_path}")
        targets.append(
            UploadTarget(
                local_path=local_path,
                s3_uri=join_s3_uri(results_s3_prefix, filename),
            )
        )
    return targets


def upload_targets_to_s3(targets: list[UploadTarget]) -> None:
    import boto3

    client = boto3.client("s3")
    for target in targets:
        bucket, key = parse_s3_uri(target.s3_uri)
        client.upload_file(str(target.local_path), bucket, key)
        print(f"uploaded={target.local_path} s3_uri={target.s3_uri}")


def main(argv: list[str] | None = None) -> int:
    import boto3

    args, finetune_args = parse_args(argv)
    s3_client = boto3.client("s3")
    cache_paths = bootstrap_huggingface_cache_dirs()
    sync_required_data_from_s3(data_s3_prefix=args.data_s3_prefix, s3_client=s3_client)
    if args.huggingface_cache_s3_prefix is not None:
        sync_huggingface_cache_from_s3(
            huggingface_cache_s3_prefix=args.huggingface_cache_s3_prefix,
            cache_paths=cache_paths,
            s3_client=s3_client,
        )
    checkpoint_path = cli_arg_value(finetune_args, "--checkpoint-path")
    if args.checkpoint_s3_prefix is not None:
        if checkpoint_path is None:
            raise ValueError(
                "--checkpoint-s3-prefix requires the forwarded fine-tune args "
                "to include --checkpoint-path."
            )
        checkpoint_bucket, checkpoint_key_prefix = parse_s3_uri(args.checkpoint_s3_prefix)
        sync_s3_prefix_to_local(
            s3_client=s3_client,
            bucket=checkpoint_bucket,
            key_prefix=checkpoint_key_prefix,
            local_root=Path(checkpoint_path).parent,
        )
    validate_huggingface_backbone_cache(
        finetune_args=finetune_args,
        cache_paths=cache_paths,
        huggingface_cache_s3_prefix=args.huggingface_cache_s3_prefix,
    )
    runtime_args = build_runtime_finetune_args(
        experiment=args.experiment,
        finetune_args=finetune_args,
    )

    output_dir = Path(cli_arg_value(runtime_args, "--output-dir") or DEFAULT_RESULTS_ROOT / args.experiment)
    cache_dir = Path(cli_arg_value(runtime_args, "--cache-dir") or DEFAULT_CACHE_ROOT / args.experiment)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    command = build_torchrun_command(finetune_args=runtime_args)
    print(f"running={shlex.join(command)}")
    subprocess.run(command, check=True)

    if not is_primary_host():
        print(f"skipping_uploads=non_primary_host host_rank={os.environ.get('SM_CURRENT_HOST_RANK', '0')}")
        return 0

    upload_targets = build_upload_targets(
        output_dir=output_dir,
        results_s3_prefix=args.results_s3_prefix,
    )
    upload_targets_to_s3(upload_targets)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
