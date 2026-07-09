from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve_repo_relative_path(path: str | Path, *, repo_root: Path = REPO_ROOT) -> Path:
    """Resolve a path from the current working directory, then fall back to the repo root."""
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()

    cwd_candidate = (Path.cwd() / candidate).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    repo_candidate = (repo_root / candidate).resolve()
    if repo_candidate.exists():
        return repo_candidate

    return cwd_candidate


def require_repo_relative_path(path: str | Path, *, repo_root: Path = REPO_ROOT) -> tuple[Path, Path]:
    """Resolve a file path and return both its absolute and repo-relative forms."""
    resolved = resolve_repo_relative_path(path, repo_root=repo_root)
    try:
        relative = resolved.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError(
            f"Path must live inside the repository root {repo_root}, but resolved to {resolved}."
        ) from exc
    return resolved, relative
