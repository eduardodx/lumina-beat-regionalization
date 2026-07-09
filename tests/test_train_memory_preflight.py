from __future__ import annotations

import pytest
import torch

from src.metrics import MetricAccumulator
from src.precision import PrecisionPolicy
from src.train import (
    DistributedContext,
    MemoryPreflightError,
    TrainConfig,
    TrainingComponents,
    TrainStepResult,
    build_arg_parser,
    config_from_args,
    resolve_memory_preflight_phases,
    run_memory_preflight,
)


def _test_precision() -> PrecisionPolicy:
    return PrecisionPolicy(
        requested="auto",
        resolved="bf16",
        use_autocast=False,
    )


def _fake_preflight_batches(cfg: TrainConfig, seq_len: int | None) -> list[dict[str, int]]:
    assert seq_len is not None
    return [{"seq_len": int(seq_len)} for _ in range(cfg.grad_accum_steps)]


def test_resolve_memory_preflight_phases_without_length_warmup() -> None:
    cfg = TrainConfig(max_steps=100, seq_len=4096)

    phases = resolve_memory_preflight_phases(cfg)

    assert [(phase.phase, phase.seq_len, phase.phase_start_step) for phase in phases] == [(0, 4096, 1)]


def test_resolve_memory_preflight_phases_with_length_warmup() -> None:
    cfg = TrainConfig(
        max_steps=1_000,
        seq_len=32_768,
        length_warmup_enabled=True,
        length_warmup_initial_seq_len=16_384,
        length_warmup_transition_fraction=0.8,
    )

    phases = resolve_memory_preflight_phases(cfg)

    assert [(phase.phase, phase.seq_len, phase.phase_start_step) for phase in phases] == [
        (1, 16_384, 1),
        (2, 32_768, 801),
    ]


def test_resolve_memory_preflight_phases_expands_alternating_auxiliary_paths() -> None:
    cfg = TrainConfig(
        max_steps=1_000,
        seq_len=32_768,
        length_warmup_enabled=True,
        length_warmup_initial_seq_len=16_384,
        length_warmup_transition_fraction=0.8,
        w_allele=0.35,
        w_counterfactual=0.10,
        auxiliary_schedule="alternate_allele_counterfactual",
    )

    phases = resolve_memory_preflight_phases(cfg)

    assert [(phase.phase, phase.seq_len, phase.phase_start_step, phase.auxiliary_path) for phase in phases] == [
        (1, 16_384, 1, "allele"),
        (1, 16_384, 2, "counterfactual"),
        (2, 32_768, 801, "allele"),
        (2, 32_768, 802, "counterfactual"),
    ]


def test_run_memory_preflight_skips_non_cuda_device(capsys: pytest.CaptureFixture[str]) -> None:
    report = run_memory_preflight(
        TrainConfig(memory_preflight_enabled=True),
        device=torch.device("cpu"),
        precision=_test_precision(),
        ctx=DistributedContext(),
        train_chromosomes=["chr1"],
    )

    captured = capsys.readouterr()

    assert "memory_preflight=skipped reason=non_cuda_device:cpu" in captured.out
    assert report == {
        "enabled": True,
        "status": "skipped",
        "device": "cpu",
        "phases": [],
        "reason": "non_cuda_device:cpu",
    }


def test_run_memory_preflight_probes_both_warmup_phases(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = TrainConfig(
        batch_size=4,
        grad_accum_steps=2,
        max_steps=1_000,
        seq_len=32_768,
        length_warmup_enabled=True,
        length_warmup_initial_seq_len=16_384,
        length_warmup_transition_fraction=0.8,
        memory_preflight_enabled=True,
    )
    recorded_seeds: list[tuple[int, int]] = []
    recorded_seq_lens: list[int] = []
    recorded_steps: list[int] = []
    allocated_values = iter([3 * 1024**3, 5 * 1024**3])
    reserved_values = iter([4 * 1024**3, 6 * 1024**3])

    monkeypatch.setattr("src.train.seed_everything", lambda seed, rank=0: recorded_seeds.append((seed, rank)))
    monkeypatch.setattr(
        "src.train.build_dataloader",
        lambda cfg, _device, _chromosomes, seq_len=None, sampling_state=None: _fake_preflight_batches(cfg, seq_len),
    )
    monkeypatch.setattr("src.train.build_model", lambda cfg, device: torch.nn.Linear(1, 1))
    monkeypatch.setattr("src.train.prepare_model_for_precision", lambda model, precision: precision)
    monkeypatch.setattr(
        "src.train.build_training_components",
        lambda cfg, model, device: TrainingComponents(
            optimizer=torch.optim.SGD(model.parameters(), lr=cfg.lr),
            effective_balancing="fixed",
            balancing_keys=(),
        ),
    )

    def _fake_train_step(
        model: torch.nn.Module,
        batches: list[dict[str, int]],
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        cfg: TrainConfig,
        precision: PrecisionPolicy,
        log_sigmas: dict[str, torch.Tensor] | None = None,
        gradnorm_balancer: object | None = None,
        ctx: DistributedContext | None = None,
        step: int = 0,
    ) -> TrainStepResult:
        del model, optimizer, device, cfg, precision, log_sigmas, gradnorm_balancer, ctx
        recorded_seq_lens.append(batches[0]["seq_len"])
        recorded_steps.append(step)
        return TrainStepResult(stats={"loss": 1.0}, metrics=MetricAccumulator(), grad_norm=0.0)

    monkeypatch.setattr("src.train.train_step", _fake_train_step)
    monkeypatch.setattr("src.train.torch.cuda.synchronize", lambda device=None: None)
    monkeypatch.setattr("src.train.torch.cuda.empty_cache", lambda: None)
    monkeypatch.setattr("src.train.torch.cuda.reset_peak_memory_stats", lambda device: None)
    monkeypatch.setattr("src.train.torch.cuda.max_memory_allocated", lambda device: next(allocated_values))
    monkeypatch.setattr("src.train.torch.cuda.max_memory_reserved", lambda device: next(reserved_values))

    report = run_memory_preflight(
        cfg,
        device=torch.device("cuda", 0),
        precision=_test_precision(),
        ctx=DistributedContext(),
        train_chromosomes=["chr1"],
    )

    assert recorded_seeds == [(cfg.seed, 0), (cfg.seed, 0)]
    assert recorded_seq_lens == [16_384, 32_768]
    assert recorded_steps == [1, 801]
    assert report["status"] == "passed"
    assert report["device"] == "cuda"
    assert len(report["phases"]) == 2
    assert report["phases"][0]["phase"] == 1
    assert report["phases"][0]["max_memory_allocated_bytes"] == 3 * 1024**3
    assert report["phases"][1]["phase"] == 2
    assert report["phases"][1]["max_memory_reserved_bytes"] == 6 * 1024**3


def test_run_memory_preflight_probes_alternating_auxiliary_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = TrainConfig(
        batch_size=4,
        grad_accum_steps=2,
        max_steps=1_000,
        seq_len=32_768,
        length_warmup_enabled=True,
        length_warmup_initial_seq_len=16_384,
        length_warmup_transition_fraction=0.8,
        memory_preflight_enabled=True,
        w_allele=0.35,
        w_counterfactual=0.10,
        auxiliary_schedule="alternate_allele_counterfactual",
    )
    recorded_seq_lens: list[int] = []
    recorded_steps: list[int] = []
    allocated_values = iter([3 * 1024**3, 4 * 1024**3, 5 * 1024**3, 6 * 1024**3])
    reserved_values = iter([4 * 1024**3, 5 * 1024**3, 6 * 1024**3, 7 * 1024**3])

    monkeypatch.setattr("src.train.seed_everything", lambda seed, rank=0: None)
    monkeypatch.setattr(
        "src.train.build_dataloader",
        lambda cfg, _device, _chromosomes, seq_len=None, sampling_state=None: _fake_preflight_batches(cfg, seq_len),
    )
    monkeypatch.setattr("src.train.build_model", lambda cfg, device: torch.nn.Linear(1, 1))
    monkeypatch.setattr("src.train.prepare_model_for_precision", lambda model, precision: precision)
    monkeypatch.setattr(
        "src.train.build_training_components",
        lambda cfg, model, device: TrainingComponents(
            optimizer=torch.optim.SGD(model.parameters(), lr=cfg.lr),
            effective_balancing="fixed",
            balancing_keys=(),
        ),
    )

    def _fake_train_step(
        model: torch.nn.Module,
        batches: list[dict[str, int]],
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        cfg: TrainConfig,
        precision: PrecisionPolicy,
        log_sigmas: dict[str, torch.Tensor] | None = None,
        gradnorm_balancer: object | None = None,
        ctx: DistributedContext | None = None,
        step: int = 0,
    ) -> TrainStepResult:
        del model, optimizer, device, cfg, precision, log_sigmas, gradnorm_balancer, ctx
        recorded_seq_lens.append(batches[0]["seq_len"])
        recorded_steps.append(step)
        return TrainStepResult(stats={"loss": 1.0}, metrics=MetricAccumulator(), grad_norm=0.0)

    monkeypatch.setattr("src.train.train_step", _fake_train_step)
    monkeypatch.setattr("src.train.torch.cuda.synchronize", lambda device=None: None)
    monkeypatch.setattr("src.train.torch.cuda.empty_cache", lambda: None)
    monkeypatch.setattr("src.train.torch.cuda.reset_peak_memory_stats", lambda device: None)
    monkeypatch.setattr("src.train.torch.cuda.max_memory_allocated", lambda device: next(allocated_values))
    monkeypatch.setattr("src.train.torch.cuda.max_memory_reserved", lambda device: next(reserved_values))

    report = run_memory_preflight(
        cfg,
        device=torch.device("cuda", 0),
        precision=_test_precision(),
        ctx=DistributedContext(),
        train_chromosomes=["chr1"],
    )

    assert recorded_seq_lens == [16_384, 16_384, 32_768, 32_768]
    assert recorded_steps == [1, 2, 801, 802]
    assert [phase["auxiliary_path"] for phase in report["phases"]] == [
        "allele",
        "counterfactual",
        "allele",
        "counterfactual",
    ]
    assert report["status"] == "passed"


def test_run_memory_preflight_raises_descriptive_phase2_oom(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = TrainConfig(
        batch_size=8,
        grad_accum_steps=2,
        max_steps=1_000,
        seq_len=32_768,
        length_warmup_enabled=True,
        length_warmup_initial_seq_len=16_384,
        length_warmup_transition_fraction=0.8,
        memory_preflight_enabled=True,
    )
    call_count = 0

    monkeypatch.setattr("src.train.seed_everything", lambda seed, rank=0: None)
    monkeypatch.setattr(
        "src.train.build_dataloader",
        lambda cfg, _device, _chromosomes, seq_len=None, sampling_state=None: _fake_preflight_batches(cfg, seq_len),
    )
    monkeypatch.setattr("src.train.build_model", lambda cfg, device: torch.nn.Linear(1, 1))
    monkeypatch.setattr("src.train.prepare_model_for_precision", lambda model, precision: precision)
    monkeypatch.setattr(
        "src.train.build_training_components",
        lambda cfg, model, device: TrainingComponents(
            optimizer=torch.optim.SGD(model.parameters(), lr=cfg.lr),
            effective_balancing="fixed",
            balancing_keys=(),
        ),
    )

    def _fake_train_step(
        model: torch.nn.Module,
        batches: list[dict[str, int]],
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        cfg: TrainConfig,
        precision: PrecisionPolicy,
        log_sigmas: dict[str, torch.Tensor] | None = None,
        gradnorm_balancer: object | None = None,
        ctx: DistributedContext | None = None,
        step: int = 0,
    ) -> TrainStepResult:
        nonlocal call_count
        del model, batches, optimizer, device, cfg, precision, log_sigmas, gradnorm_balancer, ctx, step
        call_count += 1
        if call_count == 2:
            raise torch.OutOfMemoryError("CUDA out of memory while probing phase 2")
        return TrainStepResult(stats={"loss": 1.0}, metrics=MetricAccumulator(), grad_norm=0.0)

    monkeypatch.setattr("src.train.train_step", _fake_train_step)
    monkeypatch.setattr("src.train.torch.cuda.synchronize", lambda device=None: None)
    monkeypatch.setattr("src.train.torch.cuda.empty_cache", lambda: None)
    monkeypatch.setattr("src.train.torch.cuda.reset_peak_memory_stats", lambda device: None)
    monkeypatch.setattr("src.train.torch.cuda.max_memory_allocated", lambda device: 2 * 1024**3)
    monkeypatch.setattr("src.train.torch.cuda.max_memory_reserved", lambda device: 3 * 1024**3)

    with pytest.raises(
        MemoryPreflightError,
        match=(
            r"phase=2 seq_len=32768 batch_size=8 auxiliary_path=standard grad_accum_steps=2 "
            r"resolved_precision=bf16 rank=0 error=CUDA out of memory while probing phase 2"
        ),
    ) as exc_info:
        run_memory_preflight(
            cfg,
            device=torch.device("cuda", 0),
            precision=_test_precision(),
            ctx=DistributedContext(),
            train_chromosomes=["chr1"],
        )

    report = exc_info.value.report
    assert report["status"] == "failed"
    assert [phase["status"] for phase in report["phases"]] == ["passed", "failed"]


def test_run_memory_preflight_raises_known_checkpoint_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = TrainConfig(
        batch_size=8,
        grad_accum_steps=2,
        max_steps=1_000,
        seq_len=32_768,
        length_warmup_enabled=True,
        length_warmup_initial_seq_len=16_384,
        length_warmup_transition_fraction=0.8,
        memory_preflight_enabled=True,
    )

    monkeypatch.setattr("src.train.seed_everything", lambda seed, rank=0: None)
    monkeypatch.setattr(
        "src.train.build_dataloader",
        lambda cfg, _device, _chromosomes, seq_len=None, sampling_state=None: _fake_preflight_batches(cfg, seq_len),
    )
    monkeypatch.setattr("src.train.build_model", lambda cfg, device: torch.nn.Linear(1, 1))
    monkeypatch.setattr("src.train.prepare_model_for_precision", lambda model, precision: precision)
    monkeypatch.setattr(
        "src.train.build_training_components",
        lambda cfg, model, device: TrainingComponents(
            optimizer=torch.optim.SGD(model.parameters(), lr=cfg.lr),
            effective_balancing="fixed",
            balancing_keys=(),
        ),
    )

    def _fake_train_step(
        model: torch.nn.Module,
        batches: list[dict[str, int]],
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        cfg: TrainConfig,
        precision: PrecisionPolicy,
        log_sigmas: dict[str, torch.Tensor] | None = None,
        gradnorm_balancer: object | None = None,
        ctx: DistributedContext | None = None,
        step: int = 0,
    ) -> TrainStepResult:
        del model, batches, optimizer, device, cfg, precision, log_sigmas, gradnorm_balancer, ctx, step
        raise RuntimeError(
            "torch.utils.checkpoint: Unpack is being triggered for a tensor that was already unpacked once."
        )

    monkeypatch.setattr("src.train.train_step", _fake_train_step)
    monkeypatch.setattr("src.train.torch.cuda.synchronize", lambda device=None: None)
    monkeypatch.setattr("src.train.torch.cuda.empty_cache", lambda: None)
    monkeypatch.setattr("src.train.torch.cuda.reset_peak_memory_stats", lambda device: None)
    monkeypatch.setattr("src.train.torch.cuda.max_memory_allocated", lambda device: 0)
    monkeypatch.setattr("src.train.torch.cuda.max_memory_reserved", lambda device: 0)

    with pytest.raises(MemoryPreflightError, match="already unpacked once") as exc_info:
        run_memory_preflight(
            cfg,
            device=torch.device("cuda", 0),
            precision=_test_precision(),
            ctx=DistributedContext(),
            train_chromosomes=["chr1"],
        )

    report = exc_info.value.report
    assert report["status"] == "failed"
    assert report["phases"][0]["error_type"] == "checkpoint_runtime_incompatible"


def test_memory_preflight_cli_flag_can_disable_inherited_beat_v6_default() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(["--config", "configs/beat_v6/_base.yaml", "--no-memory-preflight-enabled"])

    cfg = config_from_args(args)

    assert cfg.memory_preflight_enabled is False
