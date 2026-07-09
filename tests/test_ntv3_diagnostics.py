from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from eval.ntv3.diagnostics import (
    DEBUG_DATASET_ITEMS_ENV,
    DEBUG_EVAL_BATCHES_ENV,
    DEBUG_EVAL_MAX_BATCHES_ENV,
    DEBUG_EVAL_START_BATCH_ENV,
    DEBUG_OUTPUT_DIR_ENV,
    DEBUG_RANK_ENV,
    DatasetItemLogger,
    PhaseRecorder,
    StepEventLogger,
    collect_dataset_item_tails,
    collect_phase_manifests,
    collect_step_phase_tails,
    latest_phase_manifest,
    resolve_eval_debug_batch_window,
)


def test_phase_recorder_writes_primary_and_rank_manifest(tmp_path: Path) -> None:
    recorder = PhaseRecorder(
        output_dir=tmp_path,
        rank=0,
        world_size=2,
        metadata={"experiment": "ntv3-demo"},
    )

    recorder.mark("datasets_built", primary=True)

    primary_payload = json.loads((tmp_path / "phase_manifest.json").read_text(encoding="utf-8"))
    rank_payload = json.loads((tmp_path / "phase_manifest.rank0.json").read_text(encoding="utf-8"))

    assert primary_payload["last_phase"] == "datasets_built"
    assert rank_payload["last_phase"] == "datasets_built"
    assert primary_payload["metadata"]["experiment"] == "ntv3-demo"
    assert rank_payload["rank"] == 0


def test_phase_recorder_preserves_existing_history(tmp_path: Path) -> None:
    first = PhaseRecorder(output_dir=tmp_path, rank=0, world_size=1)
    first.mark("checkpoint_staging_started", primary=True)

    second = PhaseRecorder(output_dir=tmp_path, rank=0, world_size=1)
    second.mark("checkpoint_staging_completed", primary=True)

    payload = json.loads((tmp_path / "phase_manifest.json").read_text(encoding="utf-8"))
    history = payload["history"]

    assert [event["phase"] for event in history] == [
        "checkpoint_staging_started",
        "checkpoint_staging_completed",
    ]


def test_collect_phase_manifests_prefers_primary_manifest(tmp_path: Path) -> None:
    primary = PhaseRecorder(output_dir=tmp_path, rank=0, world_size=2)
    primary.mark("first_forward", primary=True)
    rank_one = PhaseRecorder(output_dir=tmp_path, rank=1, world_size=2)
    rank_one.mark("first_batch_loaded", primary=False)

    manifests = collect_phase_manifests(tmp_path)
    latest = latest_phase_manifest(tmp_path)

    assert sorted(manifests) == ["phase_manifest.json", "phase_manifest.rank0.json", "phase_manifest.rank1.json"]
    assert latest is not None
    assert latest["last_phase"] == "first_forward"


def test_step_event_logger_writes_only_within_window(tmp_path: Path) -> None:
    logger = StepEventLogger(
        output_dir=tmp_path,
        rank=0,
        max_steps=2,
        task_type="functional",
        resolved_precision="bf16",
    )

    logger.record(step=1, micro_step=0, phase="batch_requested")
    logger.record(step=2, micro_step=0, phase="forward_completed", extra={"loss": 0.42})
    logger.record(step=3, micro_step=0, phase="should_not_appear")

    lines = (tmp_path / "step_phase_events.rank0.jsonl").read_text(encoding="utf-8").splitlines()
    events = [json.loads(line) for line in lines if line.strip()]

    assert [event["phase"] for event in events] == ["batch_requested", "forward_completed"]
    assert events[0]["step"] == 1
    assert events[1]["details"] == {"loss": 0.42}
    assert events[1]["task_type"] == "functional"
    assert events[1]["resolved_precision"] == "bf16"


def test_step_event_logger_is_noop_when_window_zero(tmp_path: Path) -> None:
    logger = StepEventLogger(output_dir=tmp_path, rank=None, max_steps=0)
    logger.record(step=1, micro_step=0, phase="batch_requested")

    assert not (tmp_path / "step_phase_events.jsonl").exists()
    assert logger.enabled is False


def test_collect_step_phase_tails_returns_last_n(tmp_path: Path) -> None:
    logger = StepEventLogger(output_dir=tmp_path, rank=0, max_steps=100)
    for step in range(1, 6):
        logger.record(step=step, micro_step=0, phase="batch_requested")

    tails = collect_step_phase_tails(tmp_path, limit_per_rank=2)

    assert list(tails) == ["step_phase_events.rank0.jsonl"]
    assert [event["step"] for event in tails["step_phase_events.rank0.jsonl"]] == [4, 5]


def test_resolve_eval_debug_batch_window_uses_eval_override_and_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(DEBUG_EVAL_BATCHES_ENV, raising=False)
    monkeypatch.delenv(DEBUG_EVAL_MAX_BATCHES_ENV, raising=False)
    monkeypatch.delenv(DEBUG_EVAL_START_BATCH_ENV, raising=False)

    assert resolve_eval_debug_batch_window(default_debug_batches=3) == (3, 0, 0)

    monkeypatch.setenv(DEBUG_EVAL_BATCHES_ENV, "20")
    monkeypatch.setenv(DEBUG_EVAL_START_BATCH_ENV, "12")
    monkeypatch.setenv(DEBUG_EVAL_MAX_BATCHES_ENV, "8")

    assert resolve_eval_debug_batch_window(default_debug_batches=3) == (8, 12, 8)


def test_dataset_item_logger_is_noop_without_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv(DEBUG_DATASET_ITEMS_ENV, raising=False)
    monkeypatch.delenv(DEBUG_OUTPUT_DIR_ENV, raising=False)
    DatasetItemLogger._instances.clear()

    assert DatasetItemLogger.for_current_process() is None


def test_dataset_item_logger_caps_at_max_items(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(DEBUG_DATASET_ITEMS_ENV, "2")
    monkeypatch.setenv(DEBUG_OUTPUT_DIR_ENV, str(tmp_path))
    monkeypatch.setenv(DEBUG_RANK_ENV, "0")
    DatasetItemLogger._instances.clear()

    logger = DatasetItemLogger.for_current_process()
    assert logger is not None
    assert logger.preview_ordinal(dataset_kind="functional:train") == 0

    for idx in range(2):
        event = logger.record(
            dataset_kind="functional:train",
            idx=idx,
            chrom="chr1",
            start=idx * 1024,
            end=idx * 1024 + 1024,
            input_shape=(1, 1024),
            target_shape=(1024, 3),
        )
        if idx == 0:
            assert event is not None
            assert event["ordinal"] == 0
            assert logger.preview_ordinal(dataset_kind="functional:train") == 1

    annotation_event = logger.record(
        dataset_kind="functional:val",
        idx=99,
        chrom="chr2",
        start=0,
        end=1024,
        input_shape=(1, 1024),
        target_shape=(1024, 3),
    )
    assert annotation_event is not None
    assert annotation_event["ordinal"] == 0

    tails = collect_dataset_item_tails(tmp_path, limit_per_writer=10)
    events = next(iter(tails.values()))
    assert [event["ordinal"] for event in events] == [0, 1, 0]
    assert events[0]["chrom"] == "chr1"
    assert events[0]["rank"] == 0
    assert events[0]["pid"] == os.getpid()
    assert logger.preview_ordinal(dataset_kind="functional:train") is None
    assert logger.preview_ordinal(dataset_kind="functional:val") == 1
    DatasetItemLogger._instances.clear()
