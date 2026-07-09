from __future__ import annotations

import json
import os
import re
import runpy
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from re import Pattern
from typing import Any, cast

import pytest

from eval.ntv3.diagnostics import DatasetItemLogger, StepEventLogger
from src.repo_paths import REPO_ROOT
from src.sagemaker_utils import (
    LUMINA_JOB_ARGS_JSON_ENV,
    LUMINA_JOB_SCRIPT_ENV,
    LUMINA_JOB_SPEC_PATH_ENV,
    LUMINA_SETUP_EXTRAS_ENV,
    build_sagemaker_entry_environment,
    normalize_packaged_repo_path,
)
from src.train import load_yaml_train_config


def load_script(script_name: str) -> dict[str, object]:
    return runpy.run_path(str(REPO_ROOT / "scripts" / script_name))


def test_load_yaml_train_config_resolves_repo_relative_path_outside_repo_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)

    config = load_yaml_train_config(Path("configs/beat_v1/8m_15ep_32k_b200.yaml"))

    assert config["seq_len"] == 32768
    assert config["model"] == "beat-v1"


def test_normalize_packaged_repo_path_rejects_file_missing_from_git_head(tmp_path: Path) -> None:
    repo_temp = REPO_ROOT / ".pytest_cache" / f"{tmp_path.name}_dispatch_config.yaml"
    repo_temp.parent.mkdir(parents=True, exist_ok=True)
    repo_temp.write_text("model: bimamba\n", encoding="utf-8")

    try:
        with pytest.raises(FileNotFoundError, match="git archive HEAD"):
            normalize_packaged_repo_path(repo_temp)
    finally:
        repo_temp.unlink(missing_ok=True)


def test_build_sagemaker_entry_environment_writes_job_spec_file(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    environment = build_sagemaker_entry_environment(
        {"WANDB_PROJECT": "lumina"},
        setup_extras="eval,sagemaker,tracking",
        job_script="scripts/clinvar_finetune_job.py",
        job_args=["--experiment", "demo", "--", "--flag", "value with spaces"],
        source_dir=source_dir,
    )

    assert environment["WANDB_PROJECT"] == "lumina"
    assert LUMINA_SETUP_EXTRAS_ENV not in environment
    assert LUMINA_JOB_SCRIPT_ENV not in environment
    assert LUMINA_JOB_ARGS_JSON_ENV not in environment
    assert environment[LUMINA_JOB_SPEC_PATH_ENV] == ".lumina/sagemaker_job_spec.json"

    spec = json.loads((source_dir / environment[LUMINA_JOB_SPEC_PATH_ENV]).read_text(encoding="utf-8"))
    assert spec == {
        "setup_extras": "eval,sagemaker,tracking",
        "job_script": "scripts/clinvar_finetune_job.py",
        "job_args": ["--experiment", "demo", "--", "--flag", "value with spaces"],
    }


def test_clinvar_launcher_builds_job_args_for_finetune_dispatch() -> None:
    launcher = load_script("sagemaker_clinvar_finetune.py")
    parse_args = launcher["parse_args"]
    resolve_huggingface_cache_s3_prefix = launcher["resolve_huggingface_cache_s3_prefix"]
    validate_finetune_args = launcher["validate_finetune_args"]
    inject_launcher_finetune_args = launcher["inject_launcher_finetune_args"]
    build_job_args = launcher["build_job_args"]

    args, finetune_args = parse_args(  # type: ignore[operator]
        [
            "--experiment",
            "clinvar-demo",
            "--bucket",
            "demo-bucket",
            "--checkpoint-dir",
            "/opt/ml/input/data/training/checkpoints/beat-v1",
            "--",
            "--model-family",
            "lumina",
            "--model-version",
            "beat-v1",
            "--context-size",
            "4096",
        ]
    )

    spec = validate_finetune_args(  # type: ignore[operator]
        finetune_args,
        checkpoint_dir=args.checkpoint_dir,
    )
    runtime_args = inject_launcher_finetune_args(  # type: ignore[operator]
        finetune_args,
        dispatch_spec=spec,
    )
    huggingface_cache_s3_prefix = resolve_huggingface_cache_s3_prefix(  # type: ignore[operator]
        bucket=args.bucket,
        override=args.huggingface_cache_s3_prefix,
    )
    job_args = build_job_args(  # type: ignore[operator]
        experiment=args.experiment,
        results_s3_prefix="s3://demo-bucket/lumina-ssm/eval/clinvar/finetune/clinvar-demo/",
        data_s3_prefix="s3://demo-bucket/lumina-ssm/data/",
        huggingface_cache_s3_prefix=huggingface_cache_s3_prefix,
        checkpoint_s3_prefix=None,
        finetune_args=runtime_args,
    )

    assert spec.model_family == "lumina"
    assert spec.model_version == "beat-v1"
    assert spec.regime == "A"
    assert spec.checkpoint_dir == "/opt/ml/input/data/training/checkpoints/beat-v1"
    assert spec.checkpoint_path == "/opt/ml/input/data/training/checkpoints/beat-v1/best_checkpoint.pt"
    assert job_args[:8] == [
        "--experiment",
        "clinvar-demo",
        "--results-s3-prefix",
        "s3://demo-bucket/lumina-ssm/eval/clinvar/finetune/clinvar-demo/",
        "--data-s3-prefix",
        "s3://demo-bucket/lumina-ssm/data/",
        "--huggingface-cache-s3-prefix",
        "s3://demo-bucket/huggingface/hub/",
    ]
    assert "--" in job_args
    assert job_args[job_args.index("--") + 1 :] == runtime_args
    assert "--model-family" in job_args
    assert job_args[job_args.index("--checkpoint-path") + 1] == spec.checkpoint_path


def test_clinvar_launcher_accepts_s3_checkpoint_dir_and_stages_to_sm_data() -> None:
    launcher = load_script("sagemaker_clinvar_finetune.py")
    resolve_huggingface_cache_s3_prefix = launcher["resolve_huggingface_cache_s3_prefix"]
    validate_finetune_args = launcher["validate_finetune_args"]
    inject_launcher_finetune_args = launcher["inject_launcher_finetune_args"]
    build_job_args = launcher["build_job_args"]

    spec = validate_finetune_args(  # type: ignore[operator]
        [
            "--model-family",
            "lumina",
            "--model-version",
            "beat-v1",
        ],
        checkpoint_dir=(
            "s3://ai4bio-lumina-experiments-v2/"
            "lumina-ssm/data/checkpoints/lumina_8m_bimamba3_rc_5ep_consmasking"
        ),
    )
    runtime_args = inject_launcher_finetune_args(  # type: ignore[operator]
        ["--model-family", "lumina", "--model-version", "beat-v1"],
        dispatch_spec=spec,
    )
    huggingface_cache_s3_prefix = resolve_huggingface_cache_s3_prefix(  # type: ignore[operator]
        bucket="demo-bucket",
        override=None,
    )
    job_args = build_job_args(  # type: ignore[operator]
        experiment="clinvar-demo",
        results_s3_prefix="s3://demo-bucket/lumina-ssm/eval/clinvar/finetune/clinvar-demo/",
        data_s3_prefix="s3://demo-bucket/lumina-ssm/data/",
        huggingface_cache_s3_prefix=huggingface_cache_s3_prefix,
        checkpoint_s3_prefix=spec.checkpoint_s3_prefix,
        finetune_args=runtime_args,
    )

    assert spec.checkpoint_dir == "/opt/ml/input/data/training/checkpoints/lumina_8m_bimamba3_rc_5ep_consmasking"
    assert spec.checkpoint_path == (
        "/opt/ml/input/data/training/checkpoints/lumina_8m_bimamba3_rc_5ep_consmasking/best_checkpoint.pt"
    )
    assert spec.checkpoint_s3_prefix == (
        "s3://ai4bio-lumina-experiments-v2/"
        "lumina-ssm/data/checkpoints/lumina_8m_bimamba3_rc_5ep_consmasking/"
    )
    assert "--checkpoint-s3-prefix" in job_args
    assert job_args[job_args.index("--checkpoint-s3-prefix") + 1] == spec.checkpoint_s3_prefix
    assert job_args[job_args.index("--") + 1 :] == runtime_args


def test_clinvar_launcher_defaults_huggingface_cache_prefix_from_bucket() -> None:
    launcher = load_script("sagemaker_clinvar_finetune.py")
    parse_args = launcher["parse_args"]
    resolve_huggingface_cache_s3_prefix = launcher["resolve_huggingface_cache_s3_prefix"]

    args, _finetune_args = parse_args(  # type: ignore[operator]
        [
            "--experiment",
            "clinvar-demo",
            "--bucket",
            "demo-bucket",
            "--",
            "--model-family",
            "ntv3",
            "--model-version",
            "8M_pre",
        ]
    )

    resolved = resolve_huggingface_cache_s3_prefix(  # type: ignore[operator]
        bucket=args.bucket,
        override=args.huggingface_cache_s3_prefix,
    )

    assert resolved == "s3://demo-bucket/huggingface/hub/"


def test_ntv3_launcher_builds_job_args_for_official_human_functional() -> None:
    launcher = load_script("sagemaker_ntv3_benchmark.py")
    parse_args = launcher["parse_args"]
    inject_launcher_benchmark_args = launcher["inject_launcher_benchmark_args"]
    build_job_args = launcher["build_job_args"]

    args, benchmark_args = parse_args(  # type: ignore[operator]
        [
            "--experiment",
            "ntv3-human-functional",
            "--bucket",
            "demo-bucket",
            "--",
            "--official-human-functional",
        ]
    )
    checkpoint_dir = "/opt/ml/input/data/training/checkpoints/lumina-ssm-beat-v2-9m-15ep-32k-b200-20260401155503"
    runtime_args = inject_launcher_benchmark_args(  # type: ignore[operator]
        benchmark_args,
        experiment=args.experiment,
        checkpoint_dir=checkpoint_dir,
        instance_type="ml.p6-b200.48xlarge",
        managed_spot=False,
        wandb_enabled=True,
        default_wandb_run_name="ntv3-human-functional-run",
    )
    job_args = build_job_args(  # type: ignore[operator]
        experiment=args.experiment,
        results_s3_prefix="s3://demo-bucket/lumina-ssm/eval/ntv3-benchmark/ntv3-human-functional/",
        checkpoint_s3_prefix=args.checkpoint_s3_prefix,
        dataset_repo_id=args.dataset_repo_id,
        nproc_per_node=args.nproc_per_node,
        benchmark_args=runtime_args,
    )

    assert job_args[:6] == [
        "--experiment",
        "ntv3-human-functional",
        "--results-s3-prefix",
        "s3://demo-bucket/lumina-ssm/eval/ntv3-benchmark/ntv3-human-functional/",
        "--dataset-repo-id",
        "InstaDeepAI/NTv3_benchmark_dataset",
    ]
    assert "--checkpoint-s3-prefix" in job_args
    assert job_args[job_args.index("--checkpoint-s3-prefix") + 1] == args.checkpoint_s3_prefix
    assert "--" in job_args
    forwarded_args = job_args[job_args.index("--") + 1 :]
    assert forwarded_args == runtime_args
    assert forwarded_args[forwarded_args.index("--species") + 1] == "human"
    assert forwarded_args[forwarded_args.index("--task-type") + 1] == "functional"
    assert "--official-human-functional" in forwarded_args
    assert "--train-backbone" in forwarded_args
    assert forwarded_args[forwarded_args.index("--checkpoint-dir") + 1] == checkpoint_dir
    assert forwarded_args[forwarded_args.index("--dataset-root") + 1] == "/opt/ml/input/data/training/datasets/ntv3"
    assert forwarded_args[forwarded_args.index("--output-root") + 1] == (
        "/opt/ml/checkpoints/ntv3-results/ntv3-human-functional"
    )
    assert "--auto-resume" in forwarded_args
    assert "--wandb-enabled" in forwarded_args
    assert forwarded_args[forwarded_args.index("--wandb-run-name") + 1] == "ntv3-human-functional-run"


def test_ntv3_launcher_spot_enforces_checkpoint_frequency_and_auto_resume() -> None:
    launcher = load_script("sagemaker_ntv3_benchmark.py")
    inject_launcher_benchmark_args = launcher["inject_launcher_benchmark_args"]

    runtime_args = inject_launcher_benchmark_args(  # type: ignore[operator]
        ["--save-every-n-steps", "4000"],
        experiment="ntv3-human-functional",
        checkpoint_dir="/opt/ml/input/data/training/checkpoints/beat-v2",
        instance_type="ml.p6-b200.48xlarge",
        managed_spot=True,
        wandb_enabled=False,
        default_wandb_run_name="ignored",
    )

    assert "--auto-resume" in runtime_args
    assert runtime_args[runtime_args.index("--save-every-n-steps") + 1] == "1000"


def test_ntv3_launcher_caps_runtime_batch_size_for_h200() -> None:
    launcher = load_script("sagemaker_ntv3_benchmark.py")
    inject_launcher_benchmark_args = launcher["inject_launcher_benchmark_args"]

    runtime_args = inject_launcher_benchmark_args(  # type: ignore[operator]
        [],
        experiment="ntv3-human-functional",
        checkpoint_dir="/opt/ml/input/data/training/checkpoints/beat-v2",
        instance_type="ml.p5en.48xlarge",
        managed_spot=True,
        wandb_enabled=False,
        default_wandb_run_name="ignored",
    )

    assert runtime_args[runtime_args.index("--max-runtime-batch-size-per-rank") + 1] == "2"


def test_ntv3_launcher_defaults_to_stable_runtime_without_official_preset() -> None:
    launcher = load_script("sagemaker_ntv3_benchmark.py")
    inject_launcher_benchmark_args = launcher["inject_launcher_benchmark_args"]

    runtime_args = inject_launcher_benchmark_args(  # type: ignore[operator]
        [],
        experiment="ntv3-human-functional",
        checkpoint_dir="/opt/ml/input/data/training/checkpoints/beat-v5",
        instance_type="ml.p5en.48xlarge",
        managed_spot=False,
        wandb_enabled=False,
        default_wandb_run_name="ignored",
    )

    assert "--official-human-functional" not in runtime_args
    assert "--train-backbone" in runtime_args
    assert "--auto-resume" in runtime_args


def test_ntv3_launcher_forwards_explicit_nproc_per_node() -> None:
    launcher = load_script("sagemaker_ntv3_benchmark.py")
    parse_args = launcher["parse_args"]
    build_job_args = launcher["build_job_args"]

    args, benchmark_args = parse_args(  # type: ignore[operator]
        [
            "--experiment",
            "ntv3-human-functional",
            "--bucket",
            "demo-bucket",
            "--nproc-per-node",
            "1",
        ]
    )
    job_args = build_job_args(  # type: ignore[operator]
        experiment=args.experiment,
        results_s3_prefix="s3://demo-bucket/lumina-ssm/eval/ntv3-benchmark/ntv3-human-functional/",
        checkpoint_s3_prefix=args.checkpoint_s3_prefix,
        dataset_repo_id=args.dataset_repo_id,
        nproc_per_node=args.nproc_per_node,
        benchmark_args=benchmark_args,
    )

    assert job_args[job_args.index("--nproc-per-node") + 1] == "1"


def test_ntv3_launcher_builds_valid_job_name() -> None:
    launcher = load_script("sagemaker_ntv3_benchmark.py")
    build_ntv3_base_job_name = launcher["build_ntv3_base_job_name"]
    pattern = cast(Pattern[str], launcher["SAGEMAKER_TRAINING_JOB_NAME_PATTERN"])

    job_name = build_ntv3_base_job_name(  # type: ignore[operator]
        experiment="ntv3-human-functional-lumina-beat-v2-official",
        species="human",
        task_type="functional",
        random_suffix="a1b2c3",
    )

    final_job_name = f"{job_name}-20260330123456"

    assert len(job_name) <= 48
    assert len(final_job_name) <= 63
    assert job_name.startswith("lumina-ssm-ntv3-")
    assert job_name.endswith("-a1b2c3")
    assert pattern.fullmatch(job_name)
    assert pattern.fullmatch(final_job_name)


def test_ntv3_launcher_builds_valid_job_name_for_val_probe_experiment() -> None:
    launcher = load_script("sagemaker_ntv3_benchmark.py")
    build_ntv3_base_job_name = launcher["build_ntv3_base_job_name"]
    pattern = cast(Pattern[str], launcher["SAGEMAKER_TRAINING_JOB_NAME_PATTERN"])

    job_name = build_ntv3_base_job_name(  # type: ignore[operator]
        experiment="ntv3-beat-v5-val-probe-a-seed42",
        species="human",
        task_type="functional",
        random_suffix="b5a216",
    )

    final_job_name = f"{job_name}-20260420120215"

    assert len(job_name) <= 48
    assert len(final_job_name) <= 63
    assert not job_name.endswith("-")
    assert job_name.endswith("-b5a216")
    assert pattern.fullmatch(job_name)
    assert pattern.fullmatch(final_job_name)


def test_ntv3_launcher_can_preserve_beat_v5_name_with_short_prefix() -> None:
    launcher = load_script("sagemaker_ntv3_benchmark.py")
    build_ntv3_base_job_name = launcher["build_ntv3_base_job_name"]
    pattern = cast(Pattern[str], launcher["SAGEMAKER_TRAINING_JOB_NAME_PATTERN"])

    job_name = build_ntv3_base_job_name(  # type: ignore[operator]
        experiment="beat-v5-dec-coupled",
        species="human",
        task_type="functional",
        random_suffix="c0ffee",
        job_name_prefix="ntv3",
    )

    final_job_name = f"{job_name}-20260512230000"

    assert job_name == "ntv3-beat-v5-dec-coupled-human-functional-c0ffee"
    assert len(job_name) <= 48
    assert len(final_job_name) <= 63
    assert pattern.fullmatch(job_name)
    assert pattern.fullmatch(final_job_name)


def test_ntv3_launcher_isolates_sagemaker_checkpoint_sync_root() -> None:
    launcher = load_script("sagemaker_ntv3_benchmark.py")

    checkpoint_root = cast(str, launcher["DEFAULT_CHECKPOINT_LOCAL_ROOT"])
    results_root = cast(str, launcher["DEFAULT_RESULTS_ROOT"])
    tilelang_cache_dir = cast(str, launcher["DEFAULT_TILELANG_CACHE_DIR"])
    sync_root = cast(str, launcher["DEFAULT_SAGEMAKER_CHECKPOINT_SYNC_ROOT"])

    assert sync_root.startswith("/")
    assert not sync_root.startswith("/opt/ml/")
    assert sync_root != checkpoint_root
    assert sync_root != results_root
    assert sync_root != tilelang_cache_dir


def test_train_launcher_builds_job_args() -> None:
    launcher = load_script("sagemaker_train.py")
    build_job_args = launcher["build_job_args"]

    job_args = build_job_args(  # type: ignore[operator]
        config_path="configs/beat_v1/8m_15ep_32k_b200.yaml",
        resume_from="/opt/ml/checkpoints/step_100.pt",
    )

    assert job_args == [
        "--config-path",
        "configs/beat_v1/8m_15ep_32k_b200.yaml",
        "--resume-from",
        "/opt/ml/checkpoints/step_100.pt",
    ]


def test_sagemaker_entry_runs_setup_then_job(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    entry = load_script("sagemaker_entry.py")
    main = cast(Any, entry["main"])

    source_root = tmp_path / "repo"
    (source_root / "scripts").mkdir(parents=True)
    (source_root / ".lumina").mkdir(parents=True)
    (source_root / ".venv" / "bin").mkdir(parents=True)
    (source_root / "scripts" / "setup-gpu.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    (source_root / "scripts" / "job.py").write_text("print('job')\n", encoding="utf-8")
    (source_root / ".venv" / "bin" / "python").write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    (source_root / ".lumina" / "sagemaker_job_spec.json").write_text(
        json.dumps(
            {
                "setup_extras": "tracking",
                "job_script": "scripts/job.py",
                "job_args": ["--flag", "value with spaces", "--", "--sentinel"],
            }
        ),
        encoding="utf-8",
    )

    main.__globals__["_source_root"] = lambda: source_root
    monkeypatch.setenv(LUMINA_JOB_SPEC_PATH_ENV, ".lumina/sagemaker_job_spec.json")
    monkeypatch.delenv(LUMINA_SETUP_EXTRAS_ENV, raising=False)
    monkeypatch.delenv(LUMINA_JOB_SCRIPT_ENV, raising=False)
    monkeypatch.delenv(LUMINA_JOB_ARGS_JSON_ENV, raising=False)

    calls: list[dict[str, Any]] = []

    def fake_run(command: list[str], *, cwd: str, env: dict[str, str], check: bool) -> None:
        calls.append(
            {
                "command": command,
                "cwd": cwd,
                "env": env,
                "check": check,
            }
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    original_cwd = Path.cwd()
    try:
        assert main([]) == 0
    finally:
        os.chdir(original_cwd)

    assert [call["command"] for call in calls] == [
        ["bash", "scripts/setup-gpu.sh", "tracking"],
        [
            str(source_root / ".venv" / "bin" / "python"),
            str(source_root / "scripts" / "job.py"),
            "--flag",
            "value with spaces",
            "--",
            "--sentinel",
        ],
    ]
    assert all(call["cwd"] == str(source_root) for call in calls)
    assert all(call["check"] is True for call in calls)
    assert calls[0]["env"]["PATH"].startswith("/usr/local/cuda/bin:")
    assert calls[0]["env"]["PYTHONPATH"].split(":")[0] == str(source_root)


def test_sagemaker_entry_requires_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    entry = load_script("sagemaker_entry.py")
    main = cast(Any, entry["main"])

    monkeypatch.delenv(LUMINA_JOB_SPEC_PATH_ENV, raising=False)
    monkeypatch.delenv(LUMINA_SETUP_EXTRAS_ENV, raising=False)
    monkeypatch.delenv(LUMINA_JOB_SCRIPT_ENV, raising=False)
    monkeypatch.delenv(LUMINA_JOB_ARGS_JSON_ENV, raising=False)

    with pytest.raises(RuntimeError, match=LUMINA_SETUP_EXTRAS_ENV):
        main([])


def test_train_container_runner_builds_torchrun_command(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = load_script("sagemaker_train_job.py")
    build_torchrun_command = runner["build_torchrun_command"]

    monkeypatch.setenv("SM_HOST_COUNT", "2")
    monkeypatch.setenv("SM_CURRENT_HOST_RANK", "1")
    monkeypatch.setenv("SM_NUM_GPUS", "8")
    monkeypatch.setenv("SM_MASTER_ADDR", "10.0.0.2")
    monkeypatch.setenv("SM_MASTER_PORT", "23457")
    monkeypatch.setenv("TRAINING_JOB_NAME", "lumina-train-demo")

    command = build_torchrun_command(  # type: ignore[operator]
        config_path="configs/beat_v1/8m_15ep_32k_b200.yaml",
        resume_from="/opt/ml/checkpoints/step_100.pt",
    )

    assert command[:16] == [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nnodes",
        "2",
        "--node-rank",
        "1",
        "--nproc-per-node",
        "8",
        "--rdzv-backend",
        "c10d",
        "--rdzv-endpoint",
        "10.0.0.2:23457",
        "--rdzv-id",
        "lumina-train-demo",
        "-m",
    ]
    assert command[16:26] == [
        "src.train",
        "--config",
        "configs/beat_v1/8m_15ep_32k_b200.yaml",
        "--fasta-path",
        "/opt/ml/input/data/training/hg38/hg38.fa",
        "--phylo100-bw-path",
        "/opt/ml/input/data/training/phylo/hg38.phyloP100way.bw",
        "--phylo470-bw-path",
        "/opt/ml/input/data/training/phylo/hg38.phyloP470way.bw",
        "--gtf-path",
    ]
    assert "/opt/ml/input/data/training/gencode/gencode.v38.annotation.gtf.gz" in command
    assert "--output-dir" in command
    assert command[command.index("--resume-from") + 1] == "/opt/ml/checkpoints/step_100.pt"


def test_clinvar_dispatch_wrapper_forwards_head_and_wandb_args(
    tmp_path: Path,
) -> None:
    capture_path = tmp_path / "dispatch_args.json"
    mock_python = tmp_path / "mock_python.py"
    mock_python.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import json",
                "import os",
                "import sys",
                "from pathlib import Path",
                "Path(os.environ['CAPTURE_PATH']).write_text(json.dumps(sys.argv[1:]), encoding='utf-8')",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    mock_python.chmod(0o755)

    env = {
        **os.environ,
        "PYTHON_BIN": str(mock_python),
        "CAPTURE_PATH": str(capture_path),
        "CHECKPOINT_DIR": "/opt/ml/input/data/training/checkpoints/beat-v1",
        "HEAD_HIDDEN_DIM": "256",
        "HEAD_DROPOUT": "0.1",
        "LORA_R": "12",
        "DATA_S3_PREFIX": "s3://demo-bucket/lumina-ssm/data/",
        "WANDB_ENABLED": "true",
        "WANDB_PROJECT": "lumina-ssm",
        "WANDB_ENTITY": "ai4bio-lumina",
        "WANDB_RUN_NAME": "clinvar-sweep1",
        "WANDB_TAGS": "clinvar,lumina",
    }

    subprocess.run(
        ["bash", str(REPO_ROOT / "scripts" / "dispatch_clinvar_finetune_b200.sh")],
        cwd=REPO_ROOT,
        env=env,
        check=True,
    )

    argv = json.loads(capture_path.read_text(encoding="utf-8"))
    assert "--data-s3-prefix" in argv
    assert argv[argv.index("--data-s3-prefix") + 1] == "s3://demo-bucket/lumina-ssm/data/"
    assert argv[argv.index("--hidden-dim") + 1] == "256"
    assert argv[argv.index("--head-dropout") + 1] == "0.1"
    assert argv[argv.index("--lora-rank") + 1] == "12"
    assert argv[argv.index("--precision") + 1] == "bf16"
    assert "--allow-tf32" in argv
    assert "--wandb-enabled" in argv
    assert argv[argv.index("--wandb-project") + 1] == "lumina-ssm"
    assert argv[argv.index("--wandb-entity") + 1] == "ai4bio-lumina"
    assert argv[argv.index("--wandb-run-name") + 1] == "clinvar-sweep1"
    assert argv[argv.index("--wandb-tags") + 1] == "clinvar,lumina"


def test_clinvar_launcher_builds_unique_base_job_name_within_limit() -> None:
    launcher = load_script("sagemaker_clinvar_finetune.py")
    build_clinvar_base_job_name = launcher["build_clinvar_base_job_name"]
    pattern = launcher["SAGEMAKER_TRAINING_JOB_NAME_PATTERN"]

    job_name = build_clinvar_base_job_name(  # type: ignore[operator]
        experiment="clinvar-finetune-lumina-beat-v1-b200",
        model_family="lumina",
        model_version="beat-v1",
        regime="A",
        random_suffix="a1b2c3",
    )

    assert len(job_name) <= 48
    assert job_name.startswith("lumina-ssm-clinvar-")
    assert "-reg-a-" in job_name
    assert job_name.endswith("-a1b2c3")
    assert "clinvar-finetune" not in job_name
    assert isinstance(pattern, re.Pattern)
    final_job_name = f"{job_name}-20260330123456"
    assert len(final_job_name) <= 63
    assert pattern.fullmatch(final_job_name) is not None


def test_clinvar_launcher_deduplicates_family_prefix_from_model_version() -> None:
    launcher = load_script("sagemaker_clinvar_finetune.py")
    build_clinvar_base_job_name = launcher["build_clinvar_base_job_name"]

    job_name = build_clinvar_base_job_name(  # type: ignore[operator]
        experiment="clinvar-caduceus-ps",
        model_family="caduceus",
        model_version="caduceus-ps",
        regime="B",
        random_suffix="d4e5f6",
    )

    assert "caduceus-caduceus-ps" not in job_name
    assert "lumina-ssm-clinvar-caduceus-ps-reg-b-" in job_name
    assert job_name.endswith("-d4e5f6")


def test_clinvar_launcher_injects_regime_specific_default_wandb_run_name() -> None:
    launcher = load_script("sagemaker_clinvar_finetune.py")
    inject_launcher_finetune_args = launcher["inject_launcher_finetune_args"]
    FineTuneDispatchSpec = cast(Any, launcher["FineTuneDispatchSpec"])

    runtime_args = inject_launcher_finetune_args(  # type: ignore[operator]
        ["--wandb-enabled", "--model-family", "lumina", "--model-version", "beat-v1"],
        dispatch_spec=FineTuneDispatchSpec(
            regime="B",
            model_family="lumina",
            model_version="beat-v1",
            checkpoint_dir="/opt/ml/input/data/training/checkpoints/beat-v1",
            checkpoint_path="/opt/ml/input/data/training/checkpoints/beat-v1/best_checkpoint.pt",
        ),
        default_wandb_run_name="lumina-ssm-clinvar-lumina-beat-v1-regime-b-20260330-123456-a1b2c3",
    )

    assert "--wandb-run-name" in runtime_args
    assert runtime_args[runtime_args.index("--wandb-run-name") + 1] == (
        "lumina-ssm-clinvar-lumina-beat-v1-regime-b-20260330-123456-a1b2c3"
    )


def test_clinvar_launcher_appends_regime_to_explicit_wandb_run_name() -> None:
    launcher = load_script("sagemaker_clinvar_finetune.py")
    inject_launcher_finetune_args = launcher["inject_launcher_finetune_args"]
    FineTuneDispatchSpec = cast(Any, launcher["FineTuneDispatchSpec"])

    runtime_args = inject_launcher_finetune_args(  # type: ignore[operator]
        [
            "--wandb-enabled",
            "--wandb-run-name",
            "clinvar-sweep1",
            "--model-family",
            "lumina",
            "--model-version",
            "beat-v1",
        ],
        dispatch_spec=FineTuneDispatchSpec(
            regime="A",
            model_family="lumina",
            model_version="beat-v1",
            checkpoint_dir="/opt/ml/input/data/training/checkpoints/beat-v1",
            checkpoint_path="/opt/ml/input/data/training/checkpoints/beat-v1/best_checkpoint.pt",
        ),
    )

    assert runtime_args[runtime_args.index("--wandb-run-name") + 1] == "clinvar-sweep1-regime-a"


def test_clinvar_launcher_rejects_non_container_lumina_checkpoint_dir() -> None:
    launcher = load_script("sagemaker_clinvar_finetune.py")
    validate_finetune_args = launcher["validate_finetune_args"]

    with pytest.raises(ValueError, match="container-local"):
        validate_finetune_args(  # type: ignore[operator]
            [
                "--model-family",
                "lumina",
                "--model-version",
                "beat-v1",
            ],
            checkpoint_dir="outputs/lumina_8m_beat_v1",
        )


def test_clinvar_launcher_rejects_both_checkpoint_dir_and_checkpoint_path() -> None:
    launcher = load_script("sagemaker_clinvar_finetune.py")
    validate_finetune_args = launcher["validate_finetune_args"]

    with pytest.raises(ValueError, match="either the launcher-level"):
        validate_finetune_args(  # type: ignore[operator]
            [
                "--model-family",
                "lumina",
                "--model-version",
                "beat-v1",
                "--checkpoint-path",
                "/opt/ml/input/data/training/checkpoints/beat-v1/best_checkpoint.pt",
            ],
            checkpoint_dir="/opt/ml/input/data/training/checkpoints/beat-v1",
        )


def test_clinvar_upload_plan_selects_current_result_artifacts(tmp_path: Path) -> None:
    runner = load_script("clinvar_finetune_job.py")
    build_upload_targets = runner["build_upload_targets"]

    output_dir = tmp_path / "clinvar-demo"
    output_dir.mkdir()
    metrics_path = output_dir / "metrics.json"
    test_predictions = output_dir / "test_predictions.parquet"
    best_model = output_dir / "best_model.pt"
    test_predictions.write_bytes(b"test")
    best_model.write_bytes(b"model")
    metrics_path.write_text(json.dumps({"mcc": 0.5}), encoding="utf-8")

    targets = build_upload_targets(  # type: ignore[operator]
        output_dir=output_dir,
        results_s3_prefix="s3://demo-bucket/lumina-ssm/eval/clinvar/finetune/clinvar-demo/",
    )

    assert [target.local_path.name for target in targets] == [
        "metrics.json",
        "test_predictions.parquet",
        "best_model.pt",
    ]
    assert [target.s3_uri for target in targets] == [
        "s3://demo-bucket/lumina-ssm/eval/clinvar/finetune/clinvar-demo/metrics.json",
        "s3://demo-bucket/lumina-ssm/eval/clinvar/finetune/clinvar-demo/test_predictions.parquet",
        "s3://demo-bucket/lumina-ssm/eval/clinvar/finetune/clinvar-demo/best_model.pt",
    ]


def test_clinvar_container_runner_builds_torchrun_command(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = load_script("clinvar_finetune_job.py")
    build_torchrun_command = runner["build_torchrun_command"]

    monkeypatch.setenv("SM_HOST_COUNT", "2")
    monkeypatch.setenv("SM_CURRENT_HOST_RANK", "1")
    monkeypatch.setenv("SM_NUM_GPUS", "8")
    monkeypatch.setenv("SM_MASTER_ADDR", "10.0.0.1")
    monkeypatch.setenv("SM_MASTER_PORT", "23456")
    monkeypatch.setenv("TRAINING_JOB_NAME", "clinvar-demo")

    command = build_torchrun_command(  # type: ignore[operator]
        finetune_args=["--model-family", "lumina", "--model-version", "beat-v1"],
    )

    assert command[:14] == [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nnodes",
        "2",
        "--node-rank",
        "1",
        "--nproc-per-node",
        "8",
        "--rdzv-backend",
        "c10d",
        "--rdzv-endpoint",
        "10.0.0.1:23456",
        "--rdzv-id",
    ]
    assert command[14:] == [
        "clinvar-demo",
        "-m",
        "eval.clinvar.run",
        "--model-family",
        "lumina",
        "--model-version",
        "beat-v1",
    ]


def test_ntv3_container_runner_uses_standalone_torchrun_for_single_node(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = load_script("ntv3_benchmark_job.py")
    build_torchrun_command = runner["build_torchrun_command"]

    monkeypatch.setenv("SM_HOST_COUNT", "1")
    monkeypatch.setenv("SM_CURRENT_HOST_RANK", "0")
    monkeypatch.setenv("SM_NUM_GPUS", "8")
    monkeypatch.setenv("SM_MASTER_ADDR", "algo-1")
    monkeypatch.setenv("SM_MASTER_PORT", "7777")
    monkeypatch.setenv("TRAINING_JOB_NAME", "ntv3-demo")

    command = build_torchrun_command(  # type: ignore[operator]
        runtime_args=["--species", "human", "--task-type", "functional"],
    )

    assert command[:8] == [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node",
        "8",
        "-m",
        "eval.ntv3.run",
    ]
    assert "--rdzv-endpoint" not in command
    assert "algo-1:7777" not in command
    assert command[8:] == [
        "evaluate-species",
        "--species",
        "human",
        "--task-type",
        "functional",
    ]


def test_ntv3_container_runner_builds_torchrun_command(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = load_script("ntv3_benchmark_job.py")
    build_torchrun_command = runner["build_torchrun_command"]

    monkeypatch.setenv("SM_HOST_COUNT", "2")
    monkeypatch.setenv("SM_CURRENT_HOST_RANK", "1")
    monkeypatch.setenv("SM_NUM_GPUS", "8")
    monkeypatch.setenv("SM_MASTER_ADDR", "10.0.0.1")
    monkeypatch.setenv("SM_MASTER_PORT", "23456")
    monkeypatch.setenv("TRAINING_JOB_NAME", "ntv3-demo")

    command = build_torchrun_command(  # type: ignore[operator]
        runtime_args=["--species", "human", "--task-type", "functional"],
    )

    assert command[:14] == [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nnodes",
        "2",
        "--node-rank",
        "1",
        "--nproc-per-node",
        "8",
        "--rdzv-backend",
        "c10d",
        "--rdzv-endpoint",
        "10.0.0.1:23456",
        "--rdzv-id",
    ]
    assert command[14:] == [
        "ntv3-demo",
        "-m",
        "eval.ntv3.run",
        "evaluate-species",
        "--species",
        "human",
        "--task-type",
        "functional",
    ]


def test_ntv3_container_runner_honors_explicit_nproc_per_node(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = load_script("ntv3_benchmark_job.py")
    build_torchrun_command = runner["build_torchrun_command"]

    monkeypatch.setenv("SM_HOST_COUNT", "2")
    monkeypatch.setenv("SM_CURRENT_HOST_RANK", "1")
    monkeypatch.setenv("SM_NUM_GPUS", "8")
    monkeypatch.setenv("SM_MASTER_ADDR", "10.0.0.1")
    monkeypatch.setenv("SM_MASTER_PORT", "23456")
    monkeypatch.setenv("TRAINING_JOB_NAME", "ntv3-demo")

    command = build_torchrun_command(  # type: ignore[operator]
        runtime_args=["--species", "human", "--task-type", "functional"],
        nproc_per_node=1,
    )

    assert command[command.index("--nproc-per-node") + 1] == "1"


def test_ntv3_container_runner_resolves_world_size_from_nproc_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = load_script("ntv3_benchmark_job.py")
    resolve_world_size = runner["resolve_world_size"]

    monkeypatch.setenv("SM_HOST_COUNT", "2")
    monkeypatch.setenv("SM_NUM_GPUS", "8")

    assert resolve_world_size(nproc_per_node=1) == 2  # type: ignore[operator]


def test_ntv3_container_runtime_args_default_to_checkpoint_backed_output_root() -> None:
    runner = load_script("ntv3_benchmark_job.py")
    build_runtime_eval_args = runner["build_runtime_eval_args"]

    runtime_args = build_runtime_eval_args(  # type: ignore[operator]
        experiment="ntv3-demo",
        benchmark_args=["--species", "human", "--task-type", "functional"],
        dataset_root=Path("/opt/ml/input/data/training/datasets/ntv3"),
        checkpoint_dir=Path("/opt/ml/input/data/training/checkpoints/beat-v2"),
    )

    assert runtime_args[runtime_args.index("--output-root") + 1] == "/opt/ml/checkpoints/ntv3-results/ntv3-demo"
    assert runtime_args[runtime_args.index("--run-id") + 1] == "ntv3-demo"
    assert "--auto-resume" in runtime_args
    assert "--official-human-functional" not in runtime_args


def test_ntv3_model_output_materialization_copies_expected_artifacts(tmp_path: Path) -> None:
    runner = load_script("ntv3_benchmark_job.py")
    materialize_sagemaker_model_output = runner["materialize_sagemaker_model_output"]
    expected_filenames = runner["EXPECTED_TASK_RESULT_FILES"]

    output_root = tmp_path / "ntv3-results"
    task_output = output_root / "human" / "functional"
    task_output.mkdir(parents=True)
    for filename in expected_filenames:  # type: ignore[operator]
        (task_output / filename).write_text(filename, encoding="utf-8")
    aggregate_path = output_root / "ntv3_benchmark_results.csv"
    aggregate_path.write_text("Metric,run_id\n", encoding="utf-8")
    model_output_root = tmp_path / "model"

    copied_paths = materialize_sagemaker_model_output(  # type: ignore[operator]
        output_root=output_root,
        species_name="human",
        task_type="functional",
        model_output_root=model_output_root,
    )

    copied_names = [path.name for path in copied_paths]
    assert copied_names == [*expected_filenames, "ntv3_benchmark_results.csv"]  # type: ignore[list-item]
    assert (model_output_root / "best_model.pt").read_text(encoding="utf-8") == "best_model.pt"


def test_ntv3_checkpoint_resume_dir_uses_isolated_sagemaker_sync_root(tmp_path: Path) -> None:
    runner = load_script("ntv3_benchmark_job.py")
    prepare_sagemaker_checkpoint_resume_dir = runner["prepare_sagemaker_checkpoint_resume_dir"]

    output_root = tmp_path / "ntv3-results"
    sync_root = tmp_path / "sagemaker-checkpoints"

    target = prepare_sagemaker_checkpoint_resume_dir(  # type: ignore[operator]
        output_root=output_root,
        experiment="ntv3-demo",
        species_name="human",
        task_type="functional",
        checkpoint_sync_root=sync_root,
    )

    link_path = output_root / "human" / "functional" / "checkpoints"
    assert target == sync_root / "ntv3-results" / "ntv3-demo" / "human" / "functional" / "checkpoints"
    assert link_path.is_symlink()
    assert link_path.resolve() == target.resolve()

    checkpoint = link_path / "step_000001.pt"
    checkpoint.write_text("checkpoint", encoding="utf-8")
    assert (target / "step_000001.pt").read_text(encoding="utf-8") == "checkpoint"


def test_ntv3_failure_report_includes_phase_and_runtime_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = load_script("ntv3_benchmark_job.py")
    PhaseRecorder = runner["PhaseRecorder"]
    write_failure_report = runner["write_failure_report"]

    monkeypatch.setenv("SM_HOST_COUNT", "2")
    output_root = tmp_path / "ntv3-results"
    recorder = PhaseRecorder(output_dir=output_root, metadata={"experiment": "ntv3-demo"})  # type: ignore[operator]
    recorder.mark("first_backward", primary=True)  # type: ignore[operator]

    write_failure_report(  # type: ignore[operator]
        experiment="ntv3-demo",
        exc=RuntimeError("boom"),
        benchmark_args=["--species", "human", "--task-type", "functional"],
        runtime_args=[
            "--species",
            "human",
            "--task-type",
            "functional",
            "--model-version",
            "beat-v5",
            "--num-workers",
            "6",
            "--checkpoint-dir",
            "/opt/ml/input/data/training/checkpoints/beat-v5",
        ],
        output_root=output_root,
        checkpoint_s3_prefix="s3://demo-bucket/checkpoints/beat-v5/model.tar.gz",
        nproc_per_node=1,
    )

    payload = json.loads((output_root / "failure_report.json").read_text(encoding="utf-8"))
    assert payload["last_phase"] == "first_backward"
    assert payload["phase_manifests"]["phase_manifest.json"]["last_phase"] == "first_backward"
    assert payload["runtime_metadata"]["nproc_per_node"] == 1
    assert payload["runtime_metadata"]["configured_world_size"] == 2
    assert payload["runtime_metadata"]["model_version"] == "beat-v5"
    assert payload["runtime_metadata"]["checkpoint_source"] == "s3://demo-bucket/checkpoints/beat-v5/model.tar.gz"


def test_ntv3_failure_report_includes_step_and_dataset_event_tails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = load_script("ntv3_benchmark_job.py")
    write_failure_report = runner["write_failure_report"]

    output_root = tmp_path / "ntv3-results"
    task_output = output_root / "human" / "functional"
    step_logger = StepEventLogger(
        output_dir=task_output,
        rank=0,
        max_steps=3,
        task_type="functional",
        resolved_precision="bf16",
    )
    step_logger.record(step=1, micro_step=0, phase="batch_requested")
    step_logger.record(step=1, micro_step=0, phase="forward_completed", extra={"loss": 0.42})

    monkeypatch.setenv("LUMINA_NTV3_DEBUG_RANK", "0")
    item_logger = DatasetItemLogger(output_dir=task_output, max_items=2)
    item_logger.record(
        dataset_kind="functional",
        idx=0,
        chrom="chr1",
        start=0,
        end=1024,
        input_shape=(1024,),
        target_shape=(1024, 3),
    )

    write_failure_report(  # type: ignore[operator]
        experiment="ntv3-demo",
        exc=RuntimeError("boom"),
        output_root=output_root,
    )

    payload = json.loads((output_root / "failure_report.json").read_text(encoding="utf-8"))
    step_events = payload["step_phase_tails"]["human/functional"]["step_phase_events.rank0.jsonl"]
    assert step_events[-1]["phase"] == "forward_completed"
    assert payload["dataset_item_tails"]["human/functional"]
    assert payload["last_step_phase_event"]["details"] == {"loss": 0.42}
    assert payload["last_dataset_item_event"]["chrom"] == "chr1"


def test_ntv3_container_configures_tilelang_cache_dirs(tmp_path: Path) -> None:
    runner = load_script("ntv3_benchmark_job.py")
    configure_tilelang_runtime_env = runner["configure_tilelang_runtime_env"]

    env: dict[str, str] = {}
    configured = configure_tilelang_runtime_env(  # type: ignore[operator]
        env=env,
        cache_root=tmp_path / "tilelang-cache",
        execution_backend="tvm_ffi",
    )

    assert configured["TILELANG_CACHE_DIR"] == str(tmp_path / "tilelang-cache")
    assert configured["TILELANG_TMP_DIR"] == str(tmp_path / "tilelang-cache" / "tmp")
    assert configured["TILELANG_EXECUTION_BACKEND"] == "tvm_ffi"
    assert Path(configured["TILELANG_CACHE_DIR"]).is_dir()
    assert Path(configured["TILELANG_TMP_DIR"]).is_dir()


def test_ntv3_launcher_builds_tilelang_runtime_environment() -> None:
    launcher = load_script("sagemaker_ntv3_benchmark.py")
    build_container_environment = launcher["build_container_environment"]

    environment = build_container_environment(  # type: ignore[operator]
        experiment="ntv3-demo",
        benchmark_args=["--species", "human", "--task-type", "functional"],
        wandb_key="",
        hf_token="hf-secret",
    )

    assert environment["HF_TOKEN"] == "hf-secret"
    assert environment["TILELANG_CACHE_DIR"] == "/opt/ml/checkpoints/tilelang-cache"
    assert environment["TILELANG_TMP_DIR"] == "/opt/ml/checkpoints/tilelang-cache/tmp"
    assert environment["TILELANG_EXECUTION_BACKEND"] == "tvm_ffi"


def test_ntv3_launcher_propagates_debug_cuda_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    launcher = load_script("sagemaker_ntv3_benchmark.py")
    build_container_environment = launcher["build_container_environment"]

    monkeypatch.setenv("LUMINA_CUDA_DEBUG_SYNC", "1")
    environment = build_container_environment(  # type: ignore[operator]
        experiment="ntv3-demo",
        benchmark_args=["--species", "human", "--task-type", "functional"],
        wandb_key="",
        hf_token="hf-secret",
    )

    assert environment["CUDA_LAUNCH_BLOCKING"] == "1"
    assert environment["TORCH_SHOW_CPP_STACKTRACES"] == "1"
    assert environment["PYTHONFAULTHANDLER"] == "1"


def test_ntv3_launcher_propagates_debug_diagnostics_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    launcher = load_script("sagemaker_ntv3_benchmark.py")
    build_container_environment = launcher["build_container_environment"]

    monkeypatch.setenv("LUMINA_NTV3_DEBUG_EARLY_STEPS", "3")
    monkeypatch.setenv("LUMINA_NTV3_DEBUG_DATASET_ITEMS", "5")
    monkeypatch.setenv("LUMINA_NTV3_DEBUG_EVAL_BATCHES", "13")
    monkeypatch.setenv("LUMINA_NTV3_DEBUG_EVAL_START_BATCH", "12")
    monkeypatch.setenv("LUMINA_NTV3_DEBUG_EVAL_MAX_BATCHES", "21")
    environment = build_container_environment(  # type: ignore[operator]
        experiment="ntv3-demo",
        benchmark_args=["--species", "human", "--task-type", "functional"],
        wandb_key="",
        hf_token="hf-secret",
    )

    assert environment["LUMINA_NTV3_DEBUG_EARLY_STEPS"] == "3"
    assert environment["LUMINA_NTV3_DEBUG_DATASET_ITEMS"] == "5"
    assert environment["LUMINA_NTV3_DEBUG_EVAL_BATCHES"] == "13"
    assert environment["LUMINA_NTV3_DEBUG_EVAL_START_BATCH"] == "12"
    assert environment["LUMINA_NTV3_DEBUG_EVAL_MAX_BATCHES"] == "21"


class _FlakyUploadClient:
    def __init__(self, *, failures_before_success: int) -> None:
        self.failures_before_success = failures_before_success
        self.calls = 0

    def upload_file(self, filename: str, bucket: str, key: str) -> None:
        del filename, bucket, key
        self.calls += 1
        if self.calls <= self.failures_before_success:
            raise RuntimeError("temporary s3 failure")


def test_ntv3_upload_targets_retry_transient_failures(tmp_path: Path) -> None:
    runner = load_script("ntv3_benchmark_job.py")
    UploadTarget = runner["UploadTarget"]
    upload_targets_to_s3 = runner["upload_targets_to_s3"]

    local_artifact = tmp_path / "metrics_test.json"
    local_artifact.write_text("{}", encoding="utf-8")
    target = UploadTarget(local_path=local_artifact, s3_uri="s3://demo-bucket/results/metrics_test.json")  # type: ignore[operator]
    client = _FlakyUploadClient(failures_before_success=2)
    sleep_calls: list[float] = []

    upload_targets_to_s3(  # type: ignore[operator]
        [target],
        s3_client=client,
        max_attempts=4,
        initial_backoff_seconds=0.1,
        sleep_fn=sleep_calls.append,
    )

    assert client.calls == 3
    assert sleep_calls == [0.1, 0.2]


def test_clinvar_launcher_builds_wandb_environment() -> None:
    launcher = load_script("sagemaker_clinvar_finetune.py")
    build_container_environment = launcher["build_container_environment"]

    environment = build_container_environment(  # type: ignore[operator]
        experiment="clinvar-demo",
        finetune_args=[
            "--regime",
            "B",
            "--wandb-enabled",
            "--wandb-project",
            "lumina-ssm",
            "--wandb-entity",
            "ai4bio-lumina",
        ],
        wandb_key="secret-key",
    )

    assert environment["WANDB_API_KEY"] == "secret-key"
    assert environment["WANDB_PROJECT"] == "lumina-ssm"
    assert environment["WANDB_ENTITY"] == "ai4bio-lumina"
    assert environment["WANDB_RUN_GROUP"] == "sagemaker-clinvar-demo-regime-b"
    assert environment["HF_HOME"] == "/tmp/huggingface"
    assert environment["HF_HUB_CACHE"] == "/tmp/huggingface/hub"
    assert environment["HF_MODULES_CACHE"] == "/tmp/huggingface/modules"
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["TRANSFORMERS_OFFLINE"] == "1"


def test_clinvar_launcher_builds_timestamped_wandb_run_name() -> None:
    launcher = load_script("sagemaker_clinvar_finetune.py")
    build_clinvar_wandb_run_name = launcher["build_clinvar_wandb_run_name"]

    run_name = build_clinvar_wandb_run_name(  # type: ignore[operator]
        experiment="clinvar-demo",
        model_family="caduceus",
        model_version="caduceus-ps",
        regime="A",
        now=datetime(2026, 3, 30, 12, 34, 56, tzinfo=UTC),
        random_suffix="8e3aad",
    )

    assert run_name == "lumina-ssm-clinvar-caduceus-ps-regime-a-20260330-123456-8e3aad"


def test_clinvar_container_runtime_args_fill_default_dataset_and_output_paths() -> None:
    runner = load_script("clinvar_finetune_job.py")
    build_runtime_finetune_args = runner["build_runtime_finetune_args"]

    runtime_args = build_runtime_finetune_args(  # type: ignore[operator]
        experiment="clinvar-demo",
        finetune_args=["--model-family", "lumina", "--model-version", "beat-v1"],
    )

    assert "--dataset-path" in runtime_args
    assert runtime_args[runtime_args.index("--dataset-path") + 1] == (
        "/opt/ml/input/data/training/datasets/clinvar/processed/clinvar_dataset.parquet"
    )
    assert "--fasta-path" in runtime_args
    assert runtime_args[runtime_args.index("--fasta-path") + 1] == "/opt/ml/input/data/training/hg38/hg38.fa"
    assert "--phylo100-bw-path" not in runtime_args
    assert "--phylo470-bw-path" not in runtime_args
    assert "--gtf-path" not in runtime_args
    assert runtime_args[-4:] == [
        "--output-dir",
        "/tmp/clinvar-results/clinvar-demo",
        "--cache-dir",
        "/tmp/clinvar-cache/clinvar-demo",
    ]


def test_clinvar_container_runtime_args_fill_regime_b_bio_feature_paths() -> None:
    runner = load_script("clinvar_finetune_job.py")
    build_runtime_finetune_args = runner["build_runtime_finetune_args"]

    runtime_args = build_runtime_finetune_args(  # type: ignore[operator]
        experiment="clinvar-demo",
        finetune_args=["--regime", "B", "--model-family", "lumina", "--model-version", "beat-v1"],
    )

    assert runtime_args[runtime_args.index("--phylo100-bw-path") + 1] == (
        "/opt/ml/input/data/training/phylo/hg38.phyloP100way.bw"
    )
    assert runtime_args[runtime_args.index("--phylo470-bw-path") + 1] == (
        "/opt/ml/input/data/training/phylo/hg38.phyloP470way.bw"
    )
    assert runtime_args[runtime_args.index("--gtf-path") + 1] == (
        "/opt/ml/input/data/training/gencode/gencode.v38.annotation.gtf.gz"
    )


def test_clinvar_container_runtime_args_preserve_regime_b_explicit_bio_feature_overrides() -> None:
    runner = load_script("clinvar_finetune_job.py")
    build_runtime_finetune_args = runner["build_runtime_finetune_args"]

    runtime_args = build_runtime_finetune_args(  # type: ignore[operator]
        experiment="clinvar-demo",
        finetune_args=[
            "--regime",
            "B",
            "--model-family",
            "lumina",
            "--model-version",
            "beat-v1",
            "--phylo100-bw-path",
            "/custom/phylo100.bw",
            "--gtf-path",
            "/custom/genes.gtf.gz",
        ],
    )

    assert runtime_args[runtime_args.index("--phylo100-bw-path") + 1] == "/custom/phylo100.bw"
    assert runtime_args[runtime_args.index("--gtf-path") + 1] == "/custom/genes.gtf.gz"
    assert runtime_args[runtime_args.index("--phylo470-bw-path") + 1] == (
        "/opt/ml/input/data/training/phylo/hg38.phyloP470way.bw"
    )


def test_clinvar_container_runner_detects_primary_host(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = load_script("clinvar_finetune_job.py")
    is_primary_host = runner["is_primary_host"]

    monkeypatch.setenv("SM_CURRENT_HOST_RANK", "0")
    assert is_primary_host() is True  # type: ignore[operator]

    monkeypatch.setenv("SM_CURRENT_HOST_RANK", "1")
    assert is_primary_host() is False  # type: ignore[operator]


class _FakePaginator:
    def __init__(self, objects: dict[str, bytes]) -> None:
        self._objects = objects

    def paginate(self, *, Bucket: str, Prefix: str) -> list[dict[str, object]]:
        contents = [
            {"Key": key, "Size": len(data)}
            for key, data in self._objects.items()
            if key.startswith(Prefix)
        ]
        return [{"Contents": contents}]


class _FakeS3Client:
    def __init__(self, objects: dict[str, bytes]) -> None:
        self._objects = objects
        self.downloaded: list[tuple[str, str]] = []

    def get_paginator(self, name: str) -> _FakePaginator:
        assert name == "list_objects_v2"
        return _FakePaginator(self._objects)

    def download_file(self, bucket: str, key: str, filename: str) -> None:
        self.downloaded.append((bucket, key))
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self._objects[key])


def test_clinvar_job_self_sync_downloads_required_prefixes_and_skips_same_size_files(tmp_path: Path) -> None:
    runner = load_script("clinvar_finetune_job.py")
    sync_required_data_from_s3 = runner["sync_required_data_from_s3"]

    objects = {
        "lumina-ssm/data/datasets/clinvar/processed/clinvar_dataset.parquet": b"clinvar",
        "lumina-ssm/data/hg38/hg38.fa": b">chr1\nACGT\n",
        "lumina-ssm/data/gencode/gencode.v38.annotation.gtf.gz": b"gtf",
        "lumina-ssm/data/phylo/hg38.phyloP100way.bw": b"phylo100",
        "lumina-ssm/data/phylo/hg38.phyloP470way.bw": b"phylo470",
        "lumina-ssm/data/other/ignored.txt": b"ignore-me",
    }
    client = _FakeS3Client(objects)
    existing_hg38 = tmp_path / "hg38" / "hg38.fa"
    existing_hg38.parent.mkdir(parents=True, exist_ok=True)
    existing_hg38.write_bytes(objects["lumina-ssm/data/hg38/hg38.fa"])

    results = sync_required_data_from_s3(  # type: ignore[operator]
        data_s3_prefix="s3://demo-bucket/lumina-ssm/data/",
        local_root=tmp_path,
        s3_client=client,
    )

    assert len(results) == 4
    assert (tmp_path / "datasets" / "clinvar" / "processed" / "clinvar_dataset.parquet").exists()
    assert (tmp_path / "hg38" / "hg38.fa").exists()
    assert (tmp_path / "gencode" / "gencode.v38.annotation.gtf.gz").exists()
    assert (tmp_path / "phylo" / "hg38.phyloP100way.bw").exists()
    assert (tmp_path / "phylo" / "hg38.phyloP470way.bw").exists()
    assert not (tmp_path / "other" / "ignored.txt").exists()
    assert ("demo-bucket", "lumina-ssm/data/hg38/hg38.fa") not in client.downloaded


def test_clinvar_job_self_sync_downloads_remote_checkpoint_bundle(tmp_path: Path) -> None:
    runner = load_script("clinvar_finetune_job.py")
    sync_s3_prefix_to_local = runner["sync_s3_prefix_to_local"]

    objects = {
        "lumina-ssm/data/checkpoints/lumina_8m_bimamba3_rc_5ep_consmasking/best_checkpoint.pt": b"checkpoint",
        "lumina-ssm/data/checkpoints/lumina_8m_bimamba3_rc_5ep_consmasking/config.json": b"{}",
    }
    client = _FakeS3Client(objects)
    local_root = tmp_path / "checkpoints" / "lumina_8m_bimamba3_rc_5ep_consmasking"

    result = sync_s3_prefix_to_local(  # type: ignore[operator]
        s3_client=client,
        bucket="demo-bucket",
        key_prefix="lumina-ssm/data/checkpoints/lumina_8m_bimamba3_rc_5ep_consmasking",
        local_root=local_root,
    )

    assert result["downloaded"] == 2
    assert (local_root / "best_checkpoint.pt").exists()
    assert (local_root / "config.json").exists()


def test_clinvar_job_self_sync_downloads_huggingface_cache_into_env_hub_dir(tmp_path: Path) -> None:
    runner = load_script("clinvar_finetune_job.py")
    bootstrap_huggingface_cache_dirs = runner["bootstrap_huggingface_cache_dirs"]
    sync_huggingface_cache_from_s3 = runner["sync_huggingface_cache_from_s3"]

    env = {
        "HF_HOME": str(tmp_path / "hf"),
        "HF_HUB_CACHE": str(tmp_path / "hf" / "hub"),
        "HF_MODULES_CACHE": str(tmp_path / "hf" / "modules"),
    }
    cache_paths = bootstrap_huggingface_cache_dirs(env=env)  # type: ignore[operator]

    objects = {
        "huggingface/hub/models--InstaDeepAI--NTv3_8M_pre/refs/main": b"123abc",
        "huggingface/hub/models--InstaDeepAI--NTv3_8M_pre/blobs/deadbeef": b"weights",
    }
    client = _FakeS3Client(objects)

    result = sync_huggingface_cache_from_s3(  # type: ignore[operator]
        huggingface_cache_s3_prefix="s3://demo-bucket/huggingface/hub/",
        cache_paths=cache_paths,
        s3_client=client,
    )

    assert result["downloaded"] == 2
    assert Path(env["HF_HUB_CACHE"]).is_dir()
    assert (Path(env["HF_HUB_CACHE"]) / "models--InstaDeepAI--NTv3_8M_pre" / "refs" / "main").exists()
    assert (Path(env["HF_HUB_CACHE"]) / "models--InstaDeepAI--NTv3_8M_pre" / "blobs" / "deadbeef").exists()


def test_clinvar_job_skips_huggingface_validation_for_lumina(tmp_path: Path) -> None:
    runner = load_script("clinvar_finetune_job.py")
    HuggingFaceCachePaths = runner["HuggingFaceCachePaths"]
    validate_huggingface_backbone_cache = runner["validate_huggingface_backbone_cache"]

    cache_paths = HuggingFaceCachePaths(  # type: ignore[operator]
        home=tmp_path / "hf",
        hub=tmp_path / "hf" / "hub",
        modules=tmp_path / "hf" / "modules",
    )

    validated_repo = validate_huggingface_backbone_cache(  # type: ignore[operator]
        finetune_args=["--model-family", "lumina", "--model-version", "beat-v1"],
        cache_paths=cache_paths,
        snapshot_download_fn=lambda **_kwargs: (_ for _ in ()).throw(AssertionError("should not be called")),
        tokenizer_loader=lambda *_args: (_ for _ in ()).throw(AssertionError("should not be called")),
        config_loader=lambda *_args: (_ for _ in ()).throw(AssertionError("should not be called")),
    )

    assert validated_repo is None


def test_clinvar_job_resolves_dnabert2_repo_for_both_family_spellings() -> None:
    runner = load_script("clinvar_finetune_job.py")
    resolve_huggingface_repo = runner["resolve_huggingface_repo"]

    canonical_repo = resolve_huggingface_repo(  # type: ignore[operator]
        model_family="dnabert2",
        model_version="117M",
    )
    alias_repo = resolve_huggingface_repo(  # type: ignore[operator]
        model_family="dnabert-2",
        model_version="117M",
    )

    assert canonical_repo == "zhihan1996/DNABERT-2-117M"
    assert alias_repo == canonical_repo


def test_clinvar_job_fails_fast_when_huggingface_cache_missing(tmp_path: Path) -> None:
    runner = load_script("clinvar_finetune_job.py")
    HuggingFaceCachePaths = runner["HuggingFaceCachePaths"]
    validate_huggingface_backbone_cache = runner["validate_huggingface_backbone_cache"]

    cache_paths = HuggingFaceCachePaths(  # type: ignore[operator]
        home=tmp_path / "hf",
        hub=tmp_path / "hf" / "hub",
        modules=tmp_path / "hf" / "modules",
    )

    with pytest.raises(RuntimeError, match="InstaDeepAI/NTv3_8M_pre"):
        validate_huggingface_backbone_cache(  # type: ignore[operator]
            finetune_args=["--model-family", "ntv3", "--model-version", "8M_pre"],
            cache_paths=cache_paths,
            huggingface_cache_s3_prefix="s3://demo-bucket/huggingface/hub/",
            snapshot_download_fn=lambda **_kwargs: (_ for _ in ()).throw(FileNotFoundError("missing cache")),
        )


def test_clinvar_job_fails_fast_when_dnabert2_cache_missing(tmp_path: Path) -> None:
    runner = load_script("clinvar_finetune_job.py")
    HuggingFaceCachePaths = runner["HuggingFaceCachePaths"]
    validate_huggingface_backbone_cache = runner["validate_huggingface_backbone_cache"]

    cache_paths = HuggingFaceCachePaths(  # type: ignore[operator]
        home=tmp_path / "hf",
        hub=tmp_path / "hf" / "hub",
        modules=tmp_path / "hf" / "modules",
    )

    with pytest.raises(RuntimeError, match="zhihan1996/DNABERT-2-117M"):
        validate_huggingface_backbone_cache(  # type: ignore[operator]
            finetune_args=["--model-family", "dnabert2", "--model-version", "117M"],
            cache_paths=cache_paths,
            huggingface_cache_s3_prefix="s3://demo-bucket/huggingface/hub/",
            snapshot_download_fn=lambda **_kwargs: (_ for _ in ()).throw(FileNotFoundError("missing cache")),
        )


def test_clinvar_dispatch_wrapper_forwards_explicit_head_type(tmp_path: Path) -> None:
    mock_python = tmp_path / "mock_python.py"
    captured_argv = tmp_path / "captured_argv.json"
    mock_python.write_text(
        (
            "#!/usr/bin/env python3\n"
            "import json, os, sys\n"
            "from pathlib import Path\n"
            "Path(os.environ['CAPTURED_ARGV']).write_text(json.dumps(sys.argv[1:]), encoding='utf-8')\n"
            "raise SystemExit(0)\n"
        ),
        encoding="utf-8",
    )
    mock_python.chmod(0o755)

    env = {
        **os.environ,
        "PYTHON_BIN": str(mock_python),
        "CAPTURED_ARGV": str(captured_argv),
        "CHECKPOINT_DIR": "/opt/ml/input/data/training/checkpoints/beat-v1",
        "HEAD_TYPE": "regime_a_v7",
    }

    result = subprocess.run(
        ["bash", str(REPO_ROOT / "scripts" / "dispatch_clinvar_finetune_b200.sh")],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    argv = json.loads(captured_argv.read_text(encoding="utf-8"))
    assert "--head-type" in argv
    head_type_index = argv.index("--head-type")
    assert argv[head_type_index + 1] == "regime_a_v7"


def test_clinvar_dispatch_wrapper_defaults_v7_head_type_for_lumina_beat_v7(tmp_path: Path) -> None:
    mock_python = tmp_path / "mock_python.py"
    captured_argv = tmp_path / "captured_argv.json"
    mock_python.write_text(
        (
            "#!/usr/bin/env python3\n"
            "import json, os, sys\n"
            "from pathlib import Path\n"
            "Path(os.environ['CAPTURED_ARGV']).write_text(json.dumps(sys.argv[1:]), encoding='utf-8')\n"
            "raise SystemExit(0)\n"
        ),
        encoding="utf-8",
    )
    mock_python.chmod(0o755)

    env = {
        **os.environ,
        "PYTHON_BIN": str(mock_python),
        "CAPTURED_ARGV": str(captured_argv),
        "MODEL_FAMILY": "lumina",
        "MODEL_VERSION": "beat-v7",
        "CHECKPOINT_DIR": "/opt/ml/input/data/training/checkpoints/beat-v7",
    }

    result = subprocess.run(
        ["bash", str(REPO_ROOT / "scripts" / "dispatch_clinvar_finetune_b200.sh")],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    argv = json.loads(captured_argv.read_text(encoding="utf-8"))
    assert "--head-type" in argv
    head_type_index = argv.index("--head-type")
    assert argv[head_type_index + 1] == "regime_a_v7"


def test_clinvar_dispatch_wrapper_defaults_v8_head_type_for_lumina_beat_v8(tmp_path: Path) -> None:
    mock_python = tmp_path / "mock_python.py"
    captured_argv = tmp_path / "captured_argv.json"
    mock_python.write_text(
        (
            "#!/usr/bin/env python3\n"
            "import json, os, sys\n"
            "from pathlib import Path\n"
            "Path(os.environ['CAPTURED_ARGV']).write_text(json.dumps(sys.argv[1:]), encoding='utf-8')\n"
            "raise SystemExit(0)\n"
        ),
        encoding="utf-8",
    )
    mock_python.chmod(0o755)

    env = {
        **os.environ,
        "PYTHON_BIN": str(mock_python),
        "CAPTURED_ARGV": str(captured_argv),
        "MODEL_FAMILY": "lumina",
        "MODEL_VERSION": "beat-v8",
        "CHECKPOINT_DIR": "/opt/ml/input/data/training/checkpoints/beat-v8",
    }

    result = subprocess.run(
        ["bash", str(REPO_ROOT / "scripts" / "dispatch_clinvar_finetune_b200.sh")],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    argv = json.loads(captured_argv.read_text(encoding="utf-8"))
    assert "--head-type" in argv
    head_type_index = argv.index("--head-type")
    assert argv[head_type_index + 1] == "regime_a_v8"
    assert "--pairwise-rank-margin" in argv
    assert "--swap-consistency-margin" in argv
