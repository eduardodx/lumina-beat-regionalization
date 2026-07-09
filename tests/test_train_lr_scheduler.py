from __future__ import annotations

from pathlib import Path

import pytest

from src.train import (
    TrainConfig,
    build_arg_parser,
    config_from_args,
    estimate_tokens_seen,
    get_lr,
    resolve_length_warmup_state,
    validate_train_config,
)


def parse_config(path: str) -> TrainConfig:
    parser = build_arg_parser()
    args = parser.parse_args(["--config", path])
    return config_from_args(args)


def test_constant_scheduler_warms_up_then_stays_flat() -> None:
    cfg = TrainConfig(lr=3e-4, min_lr=1e-5, lr_scheduler="constant", warmup_steps=100, max_steps=1_000)

    assert get_lr(1, cfg) == pytest.approx(3e-6)
    assert get_lr(99, cfg) == pytest.approx(2.97e-4)
    assert get_lr(100, cfg) == pytest.approx(cfg.lr)
    assert get_lr(101, cfg) == pytest.approx(cfg.lr)
    assert get_lr(cfg.max_steps, cfg) == pytest.approx(cfg.lr)


def test_constant_scheduler_ignores_min_lr() -> None:
    cfg = TrainConfig(lr=3e-4, min_lr=3e-5, lr_scheduler="constant", warmup_steps=10, max_steps=100)

    assert get_lr(10, cfg) == pytest.approx(cfg.lr)
    assert get_lr(50, cfg) == pytest.approx(cfg.lr)
    assert get_lr(100, cfg) == pytest.approx(cfg.lr)


def test_cosine_scheduler_matches_warmup_and_decay_shape() -> None:
    cfg = TrainConfig(lr=3e-4, min_lr=3e-5, lr_scheduler="cosine", warmup_steps=100, max_steps=1_000)

    assert get_lr(1, cfg) == pytest.approx(3e-6)
    assert get_lr(100, cfg) == pytest.approx(cfg.lr)
    assert get_lr(550, cfg) == pytest.approx(1.65e-4)
    assert get_lr(1_000, cfg) == pytest.approx(cfg.min_lr)


def test_cosine_rewarm_scheduler_matches_two_phase_shape() -> None:
    cfg = TrainConfig(
        lr=3e-4,
        lr_scheduler="cosine_rewarm",
        max_steps=1_000,
        seq_len=32_768,
        length_warmup_enabled=True,
        length_warmup_initial_seq_len=16_384,
        length_warmup_transition_fraction=0.8,
        length_warmup_stage1_warmup_fraction=0.05,
        length_warmup_stage1_end_lr_scale=0.1,
        length_warmup_stage2_warmup_fraction=0.05,
        length_warmup_stage2_peak_lr_scale=0.2,
        length_warmup_final_lr_scale=0.01,
    )

    assert get_lr(1, cfg) == pytest.approx(7.5e-6)
    assert get_lr(40, cfg) == pytest.approx(cfg.lr)
    assert get_lr(800, cfg) == pytest.approx(cfg.lr * 0.1)
    assert get_lr(810, cfg) == pytest.approx(cfg.lr * 0.2)
    assert get_lr(1_000, cfg) == pytest.approx(cfg.lr * 0.01)


def test_config_defaults_lr_scheduler_to_cosine(tmp_path: Path) -> None:
    config_path = tmp_path / "baseline.yaml"
    config_path.write_text("lr: 0.0003\nmin_lr: 0.00003\n", encoding="utf-8")

    cfg = parse_config(str(config_path))

    assert cfg.lr_scheduler == "cosine"


def test_config_respects_explicit_constant_scheduler(tmp_path: Path) -> None:
    config_path = tmp_path / "constant.yaml"
    config_path.write_text("lr_scheduler: constant\nmin_lr: 0.00003\n", encoding="utf-8")

    cfg = parse_config(str(config_path))

    assert cfg.lr_scheduler == "constant"


def test_validate_train_config_rejects_invalid_cosine_range() -> None:
    cfg = TrainConfig(lr=1e-4, min_lr=2e-4, lr_scheduler="cosine")

    with pytest.raises(ValueError, match="min_lr <= lr"):
        validate_train_config(cfg)


def test_validate_train_config_rejects_cosine_rewarm_without_length_warmup() -> None:
    cfg = TrainConfig(lr_scheduler="cosine_rewarm")

    with pytest.raises(ValueError, match="length_warmup_enabled=True"):
        validate_train_config(cfg)


def test_validate_train_config_rejects_invalid_length_warmup_transition_fraction() -> None:
    cfg = TrainConfig(
        seq_len=2_048,
        length_warmup_enabled=True,
        length_warmup_initial_seq_len=1_024,
        length_warmup_transition_fraction=1.0,
    )

    with pytest.raises(ValueError, match="transition_fraction"):
        validate_train_config(cfg)


def test_validate_train_config_rejects_invalid_length_warmup_lr_scales() -> None:
    cfg = TrainConfig(
        seq_len=2_048,
        length_warmup_enabled=True,
        length_warmup_initial_seq_len=1_024,
        length_warmup_stage1_end_lr_scale=0.3,
        length_warmup_stage2_peak_lr_scale=0.2,
    )

    with pytest.raises(ValueError, match="stage1_end_lr_scale"):
        validate_train_config(cfg)


def test_resume_after_stage1_boundary_selects_final_seq_len() -> None:
    cfg = TrainConfig(
        max_steps=1_000,
        seq_len=32_768,
        length_warmup_enabled=True,
        length_warmup_initial_seq_len=16_384,
        length_warmup_transition_fraction=0.8,
    )

    resume_state = resolve_length_warmup_state(cfg, int(cfg.max_steps * cfg.length_warmup_transition_fraction) + 1)

    assert resume_state.phase == 2
    assert resume_state.seq_len == 32_768


def test_estimate_tokens_seen_uses_mixed_length_schedule() -> None:
    cfg = TrainConfig(
        max_steps=1_000,
        batch_size=2,
        grad_accum_steps=4,
        seq_len=32_768,
        length_warmup_enabled=True,
        length_warmup_initial_seq_len=16_384,
        length_warmup_transition_fraction=0.8,
    )

    expected = (800 * 2 * 4 * 2 * 16_384) + (100 * 2 * 4 * 2 * 32_768)

    assert estimate_tokens_seen(cfg, 900, world_size=2) == expected


def test_validate_train_config_rejects_unsupported_scheduler() -> None:
    cfg = TrainConfig(lr_scheduler="triangle")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Unsupported lr_scheduler"):
        validate_train_config(cfg)
