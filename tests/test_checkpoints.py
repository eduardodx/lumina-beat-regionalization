from __future__ import annotations

import torch

from src.checkpoints import _runtime_chunk_size_cap, normalize_lumina_model_config_for_precision
from src.models import get_model_spec
from src.precision import PrecisionPolicy


def test_runtime_chunk_size_cap_honors_env_override(monkeypatch) -> None:
    monkeypatch.setenv("LUMINA_MAMBA_MAX_CHUNK_SIZE", "12")

    assert _runtime_chunk_size_cap("cuda") == 12


def test_normalize_lumina_model_config_caps_blackwell_fp32_chunk_size(monkeypatch) -> None:
    monkeypatch.delenv("LUMINA_MAMBA_MAX_CHUNK_SIZE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _index: (10, 0))

    precision = PrecisionPolicy(requested="fp32", resolved="fp32", use_autocast=False)
    resolved = normalize_lumina_model_config_for_precision(
        "beat-v2",
        {"is_mimo": True, "mimo_rank": 4, "chunk_size": 16},
        precision,
        device="cuda",
    )

    assert resolved["chunk_size"] == 8


def test_model_registry_exposes_beat_v5() -> None:
    assert get_model_spec("beat-v5").key == "beat-v5"
