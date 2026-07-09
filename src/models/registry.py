from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, fields, is_dataclass
from typing import Any

import torch.nn as nn

from src.constants import NUM_REGION_CLASSES
from src.models.beat_v2 import BeatV2Config, build_beat_v2_model
from src.models.beat_v3 import BeatV3Config, build_beat_v3_model
from src.models.beat_v4 import BeatV4Config, build_beat_v4_model
from src.models.beat_v5 import BeatV5Config, build_beat_v5_model
from src.models.beat_v6 import BeatV6Config, build_beat_v6_model
from src.models.beat_v7 import BeatV7Config, build_beat_v7_model
from src.models.beat_v10 import BeatV10Config, build_beat_v10_model
from src.models.bimamba import BiMambaConfig, build_bimamba_model
from src.models.bimamba3 import BiMamba3Config, build_bimamba3_model
from src.models.bimamba3_rc import BiMamba3RCConfig, build_bimamba3_rc_model

DEFAULT_MODEL_KEY = "bimamba"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    config_type: type[Any]
    default_config: Any
    build_fn: Callable[[Any], nn.Module]
    normalize_config: Callable[[Mapping[str, Any] | None, str], Any]

    def resolve_config(self, overrides: Mapping[str, Any] | None = None, *, source: str = "model_config") -> Any:
        return self.normalize_config(overrides, source)


def _clone_dataclass_config(config: Any) -> Any:
    config_type = type(config)
    return config_type(**asdict(config))


def _normalize_dataclass_config(
    *,
    model_key: str,
    default_config: Any,
    overrides: Mapping[str, Any] | None,
    source: str,
) -> Any:
    if not is_dataclass(default_config):
        raise TypeError(f"Default config for model {model_key!r} must be a dataclass instance.")
    if overrides is None:
        return _clone_dataclass_config(default_config)
    if not isinstance(overrides, Mapping):
        raise TypeError(f"{source} for model {model_key!r} must be a mapping, got {type(overrides).__name__}.")

    valid_fields = {field.name for field in fields(default_config)}
    canonical: dict[str, Any] = {}
    for raw_key, value in overrides.items():
        if not isinstance(raw_key, str):
            raise TypeError(f"All keys in {source} for model {model_key!r} must be strings.")
        key = raw_key.replace("-", "_")
        if key in canonical:
            raise ValueError(f"Duplicate key {key!r} in {source} for model {model_key!r}.")
        canonical[key] = value

    unknown_keys = sorted(set(canonical) - valid_fields)
    if unknown_keys:
        raise ValueError(
            f"Unknown {model_key!r} model_config keys in {source}: {', '.join(unknown_keys)}"
        )

    data = asdict(default_config)
    data.update(canonical)
    return type(default_config)(**data)


def _make_model_spec(
    *,
    key: str,
    config: Any,
    build_fn: Callable[[Any], nn.Module],
) -> ModelSpec:
    return ModelSpec(
        key=key,
        config_type=type(config),
        default_config=config,
        build_fn=build_fn,
        normalize_config=lambda overrides, source: _normalize_dataclass_config(
            model_key=key,
            default_config=config,
            overrides=overrides,
            source=source,
        ),
    )


REGISTERED_MODELS: dict[str, ModelSpec] = {
    DEFAULT_MODEL_KEY: _make_model_spec(
        key=DEFAULT_MODEL_KEY,
        config=BiMambaConfig(),
        build_fn=build_bimamba_model,
    ),
    "bimamba3": _make_model_spec(
        key="bimamba3",
        config=BiMamba3Config(),
        build_fn=build_bimamba3_model,
    ),
    "bimamba3-rc": _make_model_spec(
        key="bimamba3-rc",
        config=BiMamba3RCConfig(),
        build_fn=build_bimamba3_rc_model,
    ),
    "beat-v1": _make_model_spec(
        key="beat-v1",
        config=BiMamba3RCConfig(num_region_classes=NUM_REGION_CLASSES),
        build_fn=build_bimamba3_rc_model,
    ),
    "beat-v2": _make_model_spec(
        key="beat-v2",
        config=BeatV2Config(num_region_classes=NUM_REGION_CLASSES),
        build_fn=build_beat_v2_model,
    ),
    "beat-v3": _make_model_spec(
        key="beat-v3",
        config=BeatV3Config(num_region_classes=NUM_REGION_CLASSES),
        build_fn=build_beat_v3_model,
    ),
    "beat-v4": _make_model_spec(
        key="beat-v4",
        config=BeatV4Config(num_region_classes=NUM_REGION_CLASSES),
        build_fn=build_beat_v4_model,
    ),
    "beat-v5": _make_model_spec(
        key="beat-v5",
        config=BeatV5Config(num_region_classes=NUM_REGION_CLASSES),
        build_fn=build_beat_v5_model,
    ),
    "beat-v6": _make_model_spec(
        key="beat-v6",
        config=BeatV6Config(num_region_classes=NUM_REGION_CLASSES),
        build_fn=build_beat_v6_model,
    ),
    "beat-v7": _make_model_spec(
        key="beat-v7",
        config=BeatV7Config(num_region_classes=NUM_REGION_CLASSES),
        build_fn=build_beat_v7_model,
    ),
    "beat-v10": _make_model_spec(
        key="beat-v10",
        config=BeatV10Config(num_region_classes=NUM_REGION_CLASSES),
        build_fn=build_beat_v10_model,
    ),
}


def registered_model_keys() -> tuple[str, ...]:
    return tuple(sorted(REGISTERED_MODELS))


def normalize_model_key(model_key: str) -> str:
    if not isinstance(model_key, str):
        raise TypeError(f"Model key must be a string, got {type(model_key).__name__}.")
    normalized = model_key.strip().lower()
    if not normalized:
        raise ValueError("Model key must not be empty.")
    return normalized


def get_model_spec(model_key: str) -> ModelSpec:
    normalized_key = normalize_model_key(model_key)
    try:
        return REGISTERED_MODELS[normalized_key]
    except KeyError as exc:
        choices = ", ".join(registered_model_keys())
        raise ValueError(f"Unknown model {model_key!r}. Expected one of: {choices}.") from exc


def resolve_model_config(
    model_key: str,
    overrides: Mapping[str, Any] | None = None,
    *,
    source: str = "model_config",
) -> Any:
    return get_model_spec(model_key).resolve_config(overrides, source=source)


def resolve_model_config_dict(
    model_key: str,
    overrides: Mapping[str, Any] | None = None,
    *,
    source: str = "model_config",
) -> dict[str, Any]:
    return asdict(resolve_model_config(model_key, overrides, source=source))


def build_registered_model(
    model_key: str,
    overrides: Mapping[str, Any] | None = None,
    *,
    config: Any | None = None,
) -> nn.Module:
    spec = get_model_spec(model_key)
    resolved_config = config if config is not None else spec.resolve_config(overrides)
    return spec.build_fn(resolved_config)
