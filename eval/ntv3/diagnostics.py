from __future__ import annotations

import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

PHASE_MANIFEST_BASENAME = "phase_manifest.json"
PHASE_MANIFEST_GLOB = "phase_manifest*.json"
STEP_PHASE_EVENTS_BASENAME = "step_phase_events.jsonl"
STEP_PHASE_EVENTS_GLOB = "step_phase_events*.jsonl"
DATASET_ITEMS_GLOB = "dataset_items*.jsonl"

DEBUG_EARLY_STEPS_ENV = "LUMINA_NTV3_DEBUG_EARLY_STEPS"
DEBUG_DATASET_ITEMS_ENV = "LUMINA_NTV3_DEBUG_DATASET_ITEMS"
DEBUG_EVAL_BATCHES_ENV = "LUMINA_NTV3_DEBUG_EVAL_BATCHES"
DEBUG_EVAL_START_BATCH_ENV = "LUMINA_NTV3_DEBUG_EVAL_START_BATCH"
DEBUG_EVAL_MAX_BATCHES_ENV = "LUMINA_NTV3_DEBUG_EVAL_MAX_BATCHES"
DEBUG_OUTPUT_DIR_ENV = "LUMINA_NTV3_DEBUG_OUTPUT_DIR"
DEBUG_RANK_ENV = "LUMINA_NTV3_DEBUG_RANK"


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def env_int(name: str, *, default: int = 0) -> int:
    raw = os.environ.get(name, "")
    if raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def resolve_eval_debug_batch_window(*, default_debug_batches: int = 0) -> tuple[int, int, int]:
    debug_batches = max(0, env_int(DEBUG_EVAL_BATCHES_ENV, default=default_debug_batches))
    start_batch = max(0, env_int(DEBUG_EVAL_START_BATCH_ENV))
    max_batches = max(0, env_int(DEBUG_EVAL_MAX_BATCHES_ENV))
    if max_batches > 0:
        debug_batches = min(debug_batches, max_batches)
    return debug_batches, start_batch, max_batches


def phase_manifest_path(output_dir: Path, *, rank: int | None = None) -> Path:
    if rank is None:
        return output_dir / PHASE_MANIFEST_BASENAME
    return output_dir / f"phase_manifest.rank{rank}.json"


def step_phase_events_path(output_dir: Path, *, rank: int | None = None) -> Path:
    if rank is None:
        return output_dir / STEP_PHASE_EVENTS_BASENAME
    return output_dir / f"step_phase_events.rank{rank}.jsonl"


def dataset_items_path(
    output_dir: Path,
    *,
    rank: int | None = None,
    worker_id: int | None = None,
    pid: int | None = None,
) -> Path:
    parts = ["dataset_items"]
    if rank is not None:
        parts.append(f"rank{rank}")
    if worker_id is not None:
        parts.append(f"worker{worker_id}")
    if pid is not None:
        parts.append(f"pid{pid}")
    return output_dir / f"{'.'.join(parts)}.jsonl"


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _append_jsonl(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")


def _read_jsonl_tail(path: Path, *, limit: int = 20) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    events: list[dict[str, Any]] = []
    for line in lines[-limit:]:
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            events.append(payload)
    return events


def _cuda_memory_stats() -> dict[str, float]:
    try:
        import torch
    except Exception:
        return {}
    try:
        if not torch.cuda.is_available():
            return {}
        device = torch.cuda.current_device()
        return {
            "cuda_mem_allocated_mb": float(torch.cuda.memory_allocated(device)) / (1024 ** 2),
            "cuda_mem_reserved_mb": float(torch.cuda.memory_reserved(device)) / (1024 ** 2),
            "cuda_mem_max_allocated_mb": float(torch.cuda.max_memory_allocated(device)) / (1024 ** 2),
        }
    except Exception:
        return {}


def collect_step_phase_tails(output_dir: Path, *, limit_per_rank: int = 20) -> dict[str, list[dict[str, Any]]]:
    tails: dict[str, list[dict[str, Any]]] = {}
    if not output_dir.exists():
        return tails
    for path in sorted(output_dir.glob(STEP_PHASE_EVENTS_GLOB)):
        events = _read_jsonl_tail(path, limit=limit_per_rank)
        if events:
            tails[path.name] = events
    return tails


def collect_dataset_item_tails(output_dir: Path, *, limit_per_writer: int = 20) -> dict[str, list[dict[str, Any]]]:
    tails: dict[str, list[dict[str, Any]]] = {}
    if not output_dir.exists():
        return tails
    for path in sorted(output_dir.glob(DATASET_ITEMS_GLOB)):
        events = _read_jsonl_tail(path, limit=limit_per_writer)
        if events:
            tails[path.name] = events
    return tails


def collect_phase_manifests(output_dir: Path) -> dict[str, dict[str, Any]]:
    manifests: dict[str, dict[str, Any]] = {}
    if not output_dir.exists():
        return manifests
    for path in sorted(output_dir.glob(PHASE_MANIFEST_GLOB)):
        payload = _read_json(path)
        if payload:
            manifests[path.name] = payload
    return manifests


def latest_phase_manifest(output_dir: Path) -> dict[str, Any] | None:
    manifests = collect_phase_manifests(output_dir)
    if not manifests:
        return None
    if PHASE_MANIFEST_BASENAME in manifests:
        return manifests[PHASE_MANIFEST_BASENAME]
    latest_name = max(
        manifests,
        key=lambda name: str(manifests[name].get("updated_at", "")),
    )
    return manifests[latest_name]


class PhaseRecorder:
    def __init__(
        self,
        *,
        output_dir: Path,
        rank: int | None = None,
        world_size: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.rank = rank
        self.world_size = world_size
        self.metadata: dict[str, Any] = dict(metadata or {})
        self._primary_path = phase_manifest_path(output_dir)
        self._rank_path = phase_manifest_path(output_dir, rank=rank)
        self._history: list[dict[str, Any]] = []
        self._load_existing_history()

    def _load_existing_history(self) -> None:
        sources: list[Path] = []
        if self.rank in (None, 0):
            sources.append(self._primary_path)
        if self._rank_path not in sources:
            sources.append(self._rank_path)

        for path in sources:
            payload = _read_json(path)
            history = payload.get("history")
            if isinstance(history, list):
                self._history = [event for event in history if isinstance(event, dict)]
                metadata = payload.get("metadata")
                if isinstance(metadata, dict):
                    self.metadata.update(metadata)
                return

    def update_metadata(self, **metadata: Any) -> None:
        self.metadata.update(metadata)

    def mark(
        self,
        phase: str,
        *,
        status: str = "ok",
        details: dict[str, Any] | None = None,
        primary: bool = True,
    ) -> dict[str, Any]:
        event: dict[str, Any] = {
            "phase": phase,
            "status": status,
            "timestamp": utc_now_iso(),
        }
        if details:
            event["details"] = details
        self._history.append(event)

        payload: dict[str, Any] = {
            "last_phase": phase,
            "status": status,
            "updated_at": event["timestamp"],
            "rank": self.rank,
            "world_size": self.world_size,
            "history": list(self._history),
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)

        _write_json(self._rank_path, payload)
        if primary and self.rank in (None, 0):
            _write_json(self._primary_path, payload)
        return payload


class StepEventLogger:
    """Append-only JSONL logger for per-(step, micro_step, phase) telemetry.

    The logger is active only for steps 1..max_steps. Outside that window
    ``record`` is a no-op, so call sites can wire the logger unconditionally.
    """

    def __init__(
        self,
        *,
        output_dir: Path,
        rank: int | None = None,
        max_steps: int = 0,
        task_type: str | None = None,
        resolved_precision: str | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.rank = rank
        self.max_steps = max(0, int(max_steps))
        self.task_type = task_type
        self.resolved_precision = resolved_precision
        self._path = step_phase_events_path(output_dir, rank=rank)
        self._start_time = time.perf_counter()

    @property
    def path(self) -> Path:
        return self._path

    @property
    def enabled(self) -> bool:
        return self.max_steps > 0

    def is_active_for_step(self, step: int) -> bool:
        return self.enabled and step <= self.max_steps

    def record(
        self,
        *,
        step: int,
        micro_step: int,
        phase: str,
        status: str = "ok",
        extra: dict[str, Any] | None = None,
    ) -> None:
        if not self.is_active_for_step(step):
            return
        event: dict[str, Any] = {
            "timestamp": utc_now_iso(),
            "wall_seconds": round(time.perf_counter() - self._start_time, 6),
            "step": step,
            "micro_step": micro_step,
            "phase": phase,
            "status": status,
            "rank": self.rank,
            "pid": os.getpid(),
            "task_type": self.task_type,
            "resolved_precision": self.resolved_precision,
        }
        event.update(_cuda_memory_stats())
        if extra:
            event["details"] = extra
        _append_jsonl(self._path, event)


class DatasetItemLogger:
    """Per-process JSONL logger for the first N dataset fetches.

    Counter state is per-process and is reset automatically when workers
    fork/spawn because each worker imports this module fresh. Output paths
    are rank+worker+pid-aware so concurrent writers never collide.
    """

    _instances: ClassVar[dict[str, DatasetItemLogger]] = {}

    def __init__(
        self,
        *,
        output_dir: Path,
        max_items: int = 0,
    ) -> None:
        self.output_dir = output_dir
        self.max_items = max(0, int(max_items))
        self._counts: dict[tuple[int | None, int | None, str], int] = {}
        self._start_time = time.perf_counter()

    @classmethod
    def for_current_process(cls) -> DatasetItemLogger | None:
        max_items = env_int(DEBUG_DATASET_ITEMS_ENV)
        output_dir_raw = os.environ.get(DEBUG_OUTPUT_DIR_ENV, "").strip()
        if max_items <= 0 or not output_dir_raw:
            return None
        output_dir = Path(output_dir_raw)
        key = f"{output_dir}|{max_items}"
        existing = cls._instances.get(key)
        if existing is not None:
            return existing
        instance = cls(output_dir=output_dir, max_items=max_items)
        cls._instances[key] = instance
        return instance

    @property
    def enabled(self) -> bool:
        return self.max_items > 0

    def _current_rank_worker(self) -> tuple[int | None, int | None]:
        rank = env_int(DEBUG_RANK_ENV, default=-1)
        rank_value: int | None = rank if rank >= 0 else None
        worker_id: int | None = None
        try:
            import torch

            info = torch.utils.data.get_worker_info()
            if info is not None:
                worker_id = int(info.id)
        except Exception:
            worker_id = None
        return rank_value, worker_id

    def preview_ordinal(self, *, dataset_kind: str) -> int | None:
        if not self.enabled:
            return None
        rank_value, worker_id = self._current_rank_worker()
        key = (rank_value, worker_id, dataset_kind)
        current_count = self._counts.get(key, 0)
        if current_count >= self.max_items:
            return None
        return current_count

    def record(
        self,
        *,
        dataset_kind: str,
        idx: int,
        chrom: str,
        start: int,
        end: int,
        input_shape: tuple[int, ...] | None = None,
        target_shape: tuple[int, ...] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        rank_value, worker_id = self._current_rank_worker()

        key = (rank_value, worker_id, dataset_kind)
        current_count = self._counts.get(key, 0)
        if current_count >= self.max_items:
            return None
        self._counts[key] = current_count + 1

        pid = os.getpid()
        path = dataset_items_path(self.output_dir, rank=rank_value, worker_id=worker_id, pid=pid)
        event: dict[str, Any] = {
            "timestamp": utc_now_iso(),
            "wall_seconds": round(time.perf_counter() - self._start_time, 6),
            "ordinal": current_count,
            "pid": pid,
            "rank": rank_value,
            "worker_id": worker_id,
            "dataset_kind": dataset_kind,
            "idx": idx,
            "chrom": chrom,
            "start": start,
            "end": end,
            "input_shape": list(input_shape) if input_shape is not None else None,
            "target_shape": list(target_shape) if target_shape is not None else None,
        }
        if extra:
            event["details"] = extra
        _append_jsonl(path, event)
        return event
