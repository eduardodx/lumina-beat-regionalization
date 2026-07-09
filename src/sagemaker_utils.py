from __future__ import annotations

import fnmatch
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path

from src.repo_paths import REPO_ROOT, require_repo_relative_path, resolve_repo_relative_path

DEFAULT_IMAGE_URI = (
    "763104351884.dkr.ecr.us-east-2.amazonaws.com/"
    "pytorch-training:2.8.0-gpu-py312-cu129-ubuntu22.04-sagemaker"
)
SM_DATA = "/opt/ml/input/data/training"
DEFAULT_SAGEMAKER_ENTRY_SCRIPT = "scripts/sagemaker_entry.py"
DEFAULT_SAGEMAKER_JOB_SPEC_PATH = ".lumina/sagemaker_job_spec.json"
LUMINA_JOB_SPEC_PATH_ENV = "LUMINA_JOB_SPEC_PATH"
LUMINA_SETUP_EXTRAS_ENV = "LUMINA_SETUP_EXTRAS"
LUMINA_JOB_SCRIPT_ENV = "LUMINA_JOB_SCRIPT"
LUMINA_JOB_ARGS_JSON_ENV = "LUMINA_JOB_ARGS_JSON"


def load_dotenv_if_available() -> None:
    env_path = REPO_ROOT / ".env"
    try:
        from dotenv import load_dotenv
    except ImportError:
        if not env_path.is_file():
            return
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            key, separator, value = line.partition("=")
            if not separator:
                continue
            key = key.strip()
            value = value.strip()
            if not key or key in os.environ:
                continue
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            os.environ[key] = value
        return
    load_dotenv(dotenv_path=env_path, override=False)


def split_cli_args(argv: Sequence[str]) -> tuple[list[str], list[str]]:
    argv_list = list(argv)
    if "--" not in argv_list:
        return argv_list, []
    split_index = argv_list.index("--")
    return argv_list[:split_index], argv_list[split_index + 1 :]


def cli_arg_value(argv: Sequence[str], flag: str) -> str | None:
    prefix = f"{flag}="
    for index, token in enumerate(argv):
        if token == flag:
            next_index = index + 1
            if next_index >= len(argv):
                return None
            next_token = argv[next_index]
            if next_token.startswith("--"):
                return None
            return next_token
        if token.startswith(prefix):
            return token[len(prefix) :]
    return None


def has_cli_flag(argv: Sequence[str], flag: str) -> bool:
    prefix = f"{flag}="
    return any(token == flag or token.startswith(prefix) for token in argv)


def resolve_packaged_repo_file(path: str | Path, *, repo_root: Path = REPO_ROOT) -> tuple[Path, Path]:
    resolved, relative = require_repo_relative_path(path, repo_root=repo_root)
    if not resolved.is_file():
        original = Path(path).expanduser()
        fallback = resolve_repo_relative_path(path, repo_root=repo_root)
        raise FileNotFoundError(
            f"File not found: {original} (resolved to {fallback})"
        )
    return resolved, relative


def resolve_packaged_repo_relative_path(path: str | Path, *, repo_root: Path = REPO_ROOT) -> str:
    _resolved, relative = resolve_packaged_repo_file(path, repo_root=repo_root)
    return relative.as_posix()


def ensure_file_in_git_archive(path: str | Path, *, repo_root: Path = REPO_ROOT) -> Path:
    relative = Path(path)
    result = subprocess.run(
        ["git", "cat-file", "-e", f"HEAD:{relative.as_posix()}"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise FileNotFoundError(
            f"{relative.as_posix()} is not present in git HEAD. "
            "SageMaker source packaging uses `git archive HEAD`, so commit this file before dispatching."
        )
    return relative


def normalize_packaged_repo_path(
    path: str | Path,
    *,
    repo_root: Path = REPO_ROOT,
    allow_uncommitted: bool = False,
) -> tuple[Path, str]:
    resolved, relative = resolve_packaged_repo_file(path, repo_root=repo_root)
    if not allow_uncommitted:
        ensure_file_in_git_archive(relative, repo_root=repo_root)
    return resolved, relative.as_posix()


def package_source(*, repo_root: Path = REPO_ROOT, include_uncommitted: bool = False) -> str:
    """Create a clean source directory for SageMaker."""
    tmpdir = tempfile.mkdtemp(prefix="lumina-ssm-source-")
    archive_path = Path(tmpdir) / "source.tar"
    print(f"  Packaging source from {repo_root} to {tmpdir} ...")
    if include_uncommitted:
        print("  Including filtered working-tree files because include_uncommitted=True.")
        _copy_source_tree_fallback(repo_root=repo_root, destination=Path(tmpdir))
        return tmpdir
    try:
        with archive_path.open("wb") as handle:
            subprocess.run(
                ["git", "archive", "--format=tar", "HEAD"],
                cwd=repo_root,
                stdout=handle,
                check=True,
            )
        subprocess.run(["tar", "xf", "source.tar"], cwd=tmpdir, check=True)
        archive_path.unlink()
    except subprocess.CalledProcessError:
        print(
            "  Warning: `git archive HEAD` failed in this checkout; "
            "falling back to a filtered local source copy."
        )
        archive_path.unlink(missing_ok=True)
        _copy_source_tree_fallback(repo_root=repo_root, destination=Path(tmpdir))
    return tmpdir


def build_sagemaker_entry_environment(
    base_environment: dict[str, str] | None,
    *,
    setup_extras: str,
    job_script: str | Path,
    job_args: Sequence[str],
    source_dir: str | Path | None = None,
    spec_relative_path: str = DEFAULT_SAGEMAKER_JOB_SPEC_PATH,
    repo_root: Path = REPO_ROOT,
) -> dict[str, str]:
    if not setup_extras.strip():
        raise ValueError("setup_extras must not be empty.")

    relative_job_script = resolve_packaged_repo_relative_path(job_script, repo_root=repo_root)
    environment = dict(base_environment or {})
    if source_dir is not None:
        spec_path = Path(source_dir) / spec_relative_path
        spec_path.parent.mkdir(parents=True, exist_ok=True)
        spec_path.write_text(
            json.dumps(
                {
                    "setup_extras": setup_extras,
                    "job_script": relative_job_script,
                    "job_args": [str(arg) for arg in job_args],
                }
            ),
            encoding="utf-8",
        )
        environment[LUMINA_JOB_SPEC_PATH_ENV] = spec_relative_path
        environment.pop(LUMINA_SETUP_EXTRAS_ENV, None)
        environment.pop(LUMINA_JOB_SCRIPT_ENV, None)
        environment.pop(LUMINA_JOB_ARGS_JSON_ENV, None)
        return environment

    serialized_args = json.dumps([str(arg) for arg in job_args])
    environment[LUMINA_SETUP_EXTRAS_ENV] = setup_extras
    environment[LUMINA_JOB_SCRIPT_ENV] = relative_job_script
    environment[LUMINA_JOB_ARGS_JSON_ENV] = serialized_args
    return environment


_PACKAGE_SOURCE_IGNORES = {
    ".git",
    ".env",
    ".venv",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "__pycache__",
    ".ipynb_checkpoints",
    "outputs",
    "data",
    "wandb",
}
_PACKAGE_SOURCE_GLOB_IGNORES = ("*.pyc", "*.pyo", "*.pt", "*.bin", "*.safetensors", "*.parquet")


def _should_ignore_source_name(name: str) -> bool:
    if name in _PACKAGE_SOURCE_IGNORES:
        return True
    return any(fnmatch.fnmatch(name, pattern) for pattern in _PACKAGE_SOURCE_GLOB_IGNORES)


def _copy_source_tree_fallback(*, repo_root: Path, destination: Path) -> None:
    for entry in repo_root.iterdir():
        if _should_ignore_source_name(entry.name):
            continue
        target = destination / entry.name
        if entry.is_dir():
            shutil.copytree(
                entry,
                target,
                ignore=shutil.ignore_patterns(*_PACKAGE_SOURCE_IGNORES, *_PACKAGE_SOURCE_GLOB_IGNORES),
            )
        else:
            shutil.copy2(entry, target)


def parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"Expected an s3:// URI, got {uri!r}.")
    remainder = uri[len("s3://") :]
    bucket, separator, key = remainder.partition("/")
    if not bucket or not separator:
        raise ValueError(f"S3 URI must include a bucket and key prefix, got {uri!r}.")
    return bucket, key.rstrip("/")
