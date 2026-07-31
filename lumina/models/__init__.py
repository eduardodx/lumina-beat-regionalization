"""Lumina model package (inference subset).

Only the pieces needed to build the architecture and run a forward pass are exported here;
the training-only loss/EMA/scoring machinery from ``lumina.models.losses`` is not shipped
(a trimmed ``losses.py`` reproduces ``LossEMANormalizer`` solely for ``strict=True`` loads).
"""

from __future__ import annotations

from lumina.models.model import (
    LuminaConfig,
    LuminaModel,
    build_lumina_model,
)
from lumina.models.registry import (
    DEFAULT_MODEL_KEY,
    REGISTERED_MODELS,
    ModelSpec,
    build_registered_model,
    get_model_spec,
    normalize_model_key,
    registered_model_keys,
    resolve_model_config,
    resolve_model_config_dict,
)

__all__ = [
    "DEFAULT_MODEL_KEY",
    "REGISTERED_MODELS",
    "LuminaConfig",
    "LuminaModel",
    "ModelSpec",
    "build_lumina_model",
    "build_registered_model",
    "get_model_spec",
    "normalize_model_key",
    "registered_model_keys",
    "resolve_model_config",
    "resolve_model_config_dict",
]
