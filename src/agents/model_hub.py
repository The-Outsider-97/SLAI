"""Resolve SLAI's approved model artifacts from Hugging Face or local storage.

The returned paths are ordinary files/directories consumed by the existing
agent-specific schema, checksum, and inference validation.
"""
from __future__ import annotations

import os

from pathlib import Path

from logs.logger import PrettyPrinter, get_logger # pyright: ignore[reportMissingImports]

LOGGER = get_logger("SLAI Model Hub")
printer = PrettyPrinter()


DEFAULT_REPOSITORY = "The-Outsider/SLAI_models"
_VALID_SOURCES = frozenset({"local", "prefer_hub", "hub"})


class ModelHubError(RuntimeError):
    """A required hosted model could not be resolved."""


def _settings() -> tuple[str, str, str, bool]:
    source = os.getenv("SLAI_MODEL_SOURCE", "prefer_hub").strip().lower()
    if source not in _VALID_SOURCES:
        raise ValueError("SLAI_MODEL_SOURCE must be local, prefer_hub, or hub")
    repository = os.getenv("SLAI_MODEL_REPOSITORY", DEFAULT_REPOSITORY).strip()
    revision = os.getenv("SLAI_MODEL_REVISION", "main").strip()
    if not repository or not revision:
        raise ValueError("SLAI_MODEL_REPOSITORY and SLAI_MODEL_REVISION cannot be empty")
    offline = os.getenv("SLAI_MODEL_OFFLINE", "false").strip().lower() in {
        "1", "true", "yes", "on",
    }
    return source, repository, revision, offline


def _relative_artifact(value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("Model Hub artifact must be a nonempty POSIX relative path")
    pieces = value.split("/")
    if any(part in {"", ".", ".."} for part in pieces) or ":" in pieces[0]:
        raise ValueError("Invalid Model Hub artifact path")
    return value


def resolve_model_file(local_path: str | Path, hub_filename: str) -> Path:
    """Prefer a cached/downloaded trusted-repository file; retain local fallback.

    Use SLAI_MODEL_SOURCE=hub to make download/cache failures explicit, or
    SLAI_MODEL_SOURCE=local to prevent remote access entirely.
    """
    local = Path(local_path).expanduser()
    remote_name = _relative_artifact(hub_filename)
    source, repository, revision, offline = _settings()
    if source == "local":
        return local
    try:
        from huggingface_hub import hf_hub_download # type: ignore

        downloaded = Path(
            hf_hub_download(
                repo_id=repository,
                filename=remote_name,
                revision=revision,
                repo_type="model",
                local_files_only=offline,
            )
        )
        if not downloaded.is_file():
            raise FileNotFoundError(downloaded)
        return downloaded
    except Exception as exc:
        if source == "hub":
            raise ModelHubError(f"Unable to resolve {remote_name} from {repository}@{revision}") from exc
        LOGGER.warning("Hub model unavailable (%s); checking local artifact %s", type(exc).__name__, local)
        return local


def resolve_model_bundle(local_directory: str | Path, hub_directory: str, filenames: tuple[str, ...]) -> Path:
    """Resolve a coherent Hub snapshot for manifest-based model bundles."""
    local = Path(local_directory).expanduser()
    folder = _relative_artifact(hub_directory)
    if not filenames:
        raise ValueError("Model bundle requires at least one file")
    paths = tuple(_relative_artifact(name) for name in filenames)
    if any("/" in path for path in paths):
        raise ValueError("Bundle filenames must be direct children")
    source, repository, revision, offline = _settings()
    if source == "local":
        return local
    try:
        from huggingface_hub import snapshot_download # type: ignore

        snapshot = Path(
            snapshot_download(
                repo_id=repository,
                revision=revision,
                repo_type="model",
                allow_patterns=[f"{folder}/{name}" for name in paths],
                local_files_only=offline,
            )
        )
        directory = snapshot / folder
        if not all((directory / name).is_file() for name in paths):
            raise FileNotFoundError(f"Incomplete Hub model bundle: {folder}")
        return directory
    except Exception as exc:
        if source == "hub":
            raise ModelHubError(f"Unable to resolve {folder} from {repository}@{revision}") from exc
        LOGGER.warning("Hub model bundle unavailable (%s); checking %s", type(exc).__name__, local)
        return local
