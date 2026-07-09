#!/usr/bin/env python3
"""Common SageMaker entry script for Lumina jobs."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

LUMINA_JOB_SPEC_PATH_ENV = "LUMINA_JOB_SPEC_PATH"
LUMINA_SETUP_EXTRAS_ENV = "LUMINA_SETUP_EXTRAS"
LUMINA_JOB_SCRIPT_ENV = "LUMINA_JOB_SCRIPT"
LUMINA_JOB_ARGS_JSON_ENV = "LUMINA_JOB_ARGS_JSON"


def _source_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Required environment variable {name} is missing or empty.")
    return value


def _parse_job_args(raw_value: str) -> list[str]:
    try:
        decoded = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{LUMINA_JOB_ARGS_JSON_ENV} must contain valid JSON.") from exc
    if not isinstance(decoded, list) or any(not isinstance(item, str) for item in decoded):
        raise RuntimeError(f"{LUMINA_JOB_ARGS_JSON_ENV} must decode to a JSON list of strings.")
    return list(decoded)


def _load_job_spec_from_file(source_root: Path, spec_path_value: str) -> tuple[str, str, list[str]]:
    spec_path = (source_root / spec_path_value).resolve()
    if not spec_path.is_file():
        raise FileNotFoundError(f"Configured SageMaker job spec not found: {spec_path}")

    try:
        decoded = json.loads(spec_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{LUMINA_JOB_SPEC_PATH_ENV} must point to valid JSON.") from exc

    if not isinstance(decoded, dict):
        raise RuntimeError(f"{LUMINA_JOB_SPEC_PATH_ENV} must decode to a JSON object.")

    setup_extras = decoded.get("setup_extras")
    job_script = decoded.get("job_script")
    job_args = decoded.get("job_args")
    if not isinstance(setup_extras, str) or not setup_extras.strip():
        raise RuntimeError("Job spec is missing a valid `setup_extras` string.")
    if not isinstance(job_script, str) or not job_script.strip():
        raise RuntimeError("Job spec is missing a valid `job_script` string.")
    if not isinstance(job_args, list) or any(not isinstance(item, str) for item in job_args):
        raise RuntimeError("Job spec is missing a valid `job_args` list of strings.")
    return setup_extras, job_script, list(job_args)


def _load_job_configuration(source_root: Path) -> tuple[str, str, list[str]]:
    spec_path_value = os.environ.get(LUMINA_JOB_SPEC_PATH_ENV, "").strip()
    if spec_path_value:
        return _load_job_spec_from_file(source_root, spec_path_value)

    setup_extras = _require_env(LUMINA_SETUP_EXTRAS_ENV)
    job_script = _require_env(LUMINA_JOB_SCRIPT_ENV)
    job_args = _parse_job_args(_require_env(LUMINA_JOB_ARGS_JSON_ENV))
    return setup_extras, job_script, job_args


def _runtime_environment(source_root: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["PATH"] = f"/usr/local/cuda/bin:{env.get('PATH', '')}"
    existing_pythonpath = env.get("PYTHONPATH", "").strip()
    env["PYTHONPATH"] = f"{source_root}:{existing_pythonpath}" if existing_pythonpath else str(source_root)
    return env


def _run_command(command: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    print(f"running={shlex.join(command)}")
    subprocess.run(command, cwd=str(cwd), env=env, check=True)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args:
        raise RuntimeError(f"{Path(__file__).name} does not accept CLI arguments: {args!r}")

    source_root = _source_root()
    os.chdir(source_root)

    setup_extras, job_script, job_args = _load_job_configuration(source_root)

    job_script_path = (source_root / job_script).resolve()
    if not job_script_path.is_file():
        raise FileNotFoundError(f"Configured SageMaker job script not found: {job_script_path}")

    env = _runtime_environment(source_root)
    _run_command(["bash", "scripts/setup-gpu.sh", setup_extras], cwd=source_root, env=env)

    venv_python = source_root / ".venv" / "bin" / "python"
    if not venv_python.is_file():
        raise FileNotFoundError(f"Virtualenv python was not created by setup-gpu.sh: {venv_python}")

    _run_command([str(venv_python), str(job_script_path), *job_args], cwd=source_root, env=env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
