from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import torch

from .model import BioPrimeConfig, DNAFoundationBioPrime

# BioPrimeConfig subclasses BeatV11Config, so fields() returns the full (base + BioPrime) field set,
# i.e. every key present in a trained ``model_config``.
_CONFIG_FIELDS = {field.name for field in fields(BioPrimeConfig)}
_STATE_DICT_KEYS = ("model", "model_state_dict", "state_dict")
_STATE_DICT_PREFIXES = ("module.", "_orig_mod.")
_MODEL_NAMES = {"beat_v11_bioprime", "beat-v11-bioprime", "beat_v11", "beat-v11"}


def is_s3_uri(path: str | os.PathLike[str]) -> bool:
    return os.fspath(path).startswith("s3://")


def parse_s3_uri(uri: str | os.PathLike[str]) -> tuple[str, str]:
    parsed = urlparse(os.fspath(uri))
    if parsed.scheme != "s3":
        raise ValueError(f"Expected an s3:// URI, got {uri!s}.")
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    if not bucket or not key:
        raise ValueError(f"S3 checkpoint URI must include a bucket and key, got {uri!s}.")
    return bucket, key


def download_s3_checkpoint(
    s3_uri: str | os.PathLike[str],
    local_path: str | Path,
    *,
    s3_client: Any | None = None,
) -> Path:
    """Download an S3 checkpoint URI to a local file and return the local path."""

    bucket, key = parse_s3_uri(s3_uri)
    destination = Path(local_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    client = s3_client
    if client is None:
        try:
            import boto3
        except ImportError as exc:
            raise ImportError(
                "Loading checkpoints from s3:// requires boto3. "
                "Install it with `uv sync --extra s3` from the release directory "
                "or `uv pip install boto3` in your active environment."
            ) from exc
        client = boto3.client("s3")

    client.download_file(bucket, key, str(destination))
    return destination


def default_config_dict() -> dict[str, Any]:
    return asdict(BioPrimeConfig())


def normalize_config_dict(
    overrides: Mapping[str, Any] | None = None,
    *,
    source: str = "model_config",
    ignore_unknown: bool = False,
) -> dict[str, Any]:
    data = default_config_dict()
    if overrides is None:
        return data

    canonical: dict[str, Any] = {}
    for raw_key, value in overrides.items():
        if not isinstance(raw_key, str):
            raise TypeError(f"All keys in {source} must be strings, got {type(raw_key).__name__}.")
        key = raw_key.replace("-", "_")
        if key in canonical:
            raise ValueError(f"Duplicate key {key!r} in {source}.")
        canonical[key] = value

    unknown = sorted(set(canonical) - _CONFIG_FIELDS)
    if unknown and not ignore_unknown:
        raise ValueError(f"Unknown Beat-v11 config keys in {source}: {', '.join(unknown)}")
    for key in unknown:
        canonical.pop(key)

    data.update(canonical)
    return data


def config_from_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    config_overrides: Mapping[str, Any] | None = None,
) -> BioPrimeConfig:
    raw_config = checkpoint.get("config")
    train_config = raw_config if isinstance(raw_config, Mapping) else {}

    model_name = train_config.get("model")
    if isinstance(model_name, str) and model_name.strip().lower() not in _MODEL_NAMES:
        raise ValueError(
            f"Checkpoint declares model={model_name!r}, expected one of "
            f"{sorted(_MODEL_NAMES)} (the beat_v11_bioprime family)."
        )

    raw_model_config = train_config.get("model_config")
    if not isinstance(raw_model_config, Mapping):
        raw_model_config = checkpoint.get("model_config")
    if not isinstance(raw_model_config, Mapping):
        raw_model_config = {}

    # EMA checkpoints (final_ema.pt) can carry an empty config; fall back to the packaged defaults.
    data = normalize_config_dict(raw_model_config, source="checkpoint model_config")
    if config_overrides is not None:
        data = normalize_config_dict({**data, **dict(config_overrides)}, source="config_overrides")
    return BioPrimeConfig(**data)


def _torch_load_local_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint {path!s} must contain a dict, got {type(checkpoint).__name__}.")
    return checkpoint


def torch_load_checkpoint(
    path: str | Path,
    map_location: str | torch.device = "cpu",
    *,
    s3_client: Any | None = None,
) -> dict[str, Any]:
    """Load a local or S3 checkpoint dictionary with `torch.load`.

    For `s3://bucket/key` paths, the object is downloaded to a temporary file
    first so PyTorch can load it through its normal checkpoint reader.
    """

    if is_s3_uri(path):
        parsed = urlparse(os.fspath(path))
        filename = Path(parsed.path).name or "checkpoint.pt"
        with tempfile.TemporaryDirectory(prefix="beat-v11-checkpoint-") as tmp_dir:
            local_path = Path(tmp_dir) / filename
            download_s3_checkpoint(path, local_path, s3_client=s3_client)
            return _torch_load_local_checkpoint(local_path, map_location=map_location)

    return _torch_load_local_checkpoint(path, map_location=map_location)


def _looks_like_state_dict(value: Mapping[str, Any]) -> bool:
    return bool(value) and all(isinstance(key, str) and torch.is_tensor(tensor) for key, tensor in value.items())


def extract_model_state_dict(checkpoint: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    for key in _STATE_DICT_KEYS:
        value = checkpoint.get(key)
        if isinstance(value, Mapping) and _looks_like_state_dict(value):
            return {name: tensor for name, tensor in value.items()}
    if _looks_like_state_dict(checkpoint):
        return {name: tensor for name, tensor in checkpoint.items()}
    choices = ", ".join(_STATE_DICT_KEYS)
    raise ValueError(f"Could not find a model state dict. Expected one of: {choices}, or a raw state dict.")


def strip_state_dict_prefixes(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    stripped: dict[str, torch.Tensor] = {}
    for name, tensor in state_dict.items():
        clean_name = name
        changed = True
        while changed:
            changed = False
            for prefix in _STATE_DICT_PREFIXES:
                if clean_name.startswith(prefix):
                    clean_name = clean_name[len(prefix) :]
                    changed = True
        stripped[clean_name] = tensor
    return stripped


def build_model(config_overrides: Mapping[str, Any] | None = None) -> DNAFoundationBioPrime:
    cfg = BioPrimeConfig(**normalize_config_dict(config_overrides, source="config_overrides"))
    return DNAFoundationBioPrime(cfg)


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
    strict: bool = True,
    config_overrides: Mapping[str, Any] | None = None,
    s3_client: Any | None = None,
) -> DNAFoundationBioPrime:
    """Load a Beat-v11 BioPrime model from a native Lumina training checkpoint.

    Native ``final_checkpoint.pt`` / ``best_checkpoint.pt`` files (state dict under ``model``,
    architecture under ``config["model_config"]``) load with ``strict=True``. The EMA export
    ``final_ema.pt`` omits the optimizer / ``loss_ema`` buffers and may carry an empty config;
    pass ``strict=False`` for it (the packaged defaults supply the architecture).
    """
    checkpoint = torch_load_checkpoint(checkpoint_path, map_location="cpu", s3_client=s3_client)
    cfg = config_from_checkpoint(checkpoint, config_overrides=config_overrides)
    model = DNAFoundationBioPrime(cfg)
    model.load_state_dict(strip_state_dict_prefixes(extract_model_state_dict(checkpoint)), strict=strict)
    if device is not None:
        model = model.to(device)
    if dtype is not None:
        model = model.to(dtype=dtype)
    model.eval()
    return model
