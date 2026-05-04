"""
Utility layer for production checkpoint management.

This module intentionally owns the reusable pieces of checkpoint handling:
validation, constants, error classes, manifest serialization, atomic writes,
file hashing, tokenizer persistence, RNG capture/restore, NPZ tensor loading,
and archive helpers.

The manager module should stay focused on orchestration. Any helper or
cross-cutting checkpoint behavior belongs here.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import random
import re
import shutil
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Literal, Mapping, MutableMapping, Optional, Sequence

import numpy as np
import torch

from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("SLAI Checkpoint Utils")
printer = PrettyPrinter()

CheckpointFormat = Literal["torch", "npz"]

MANIFEST_NAME = "manifest.json"
TORCH_CHECKPOINT_NAME = "checkpoint.pt"
LEGACY_TORCH_CHECKPOINT_NAME = "model_weights.pt"
NPZ_WEIGHTS_NAME = "model_weights.npz"
TOKENIZER_VOCAB_NAME = "tokenizer_vocab.json"
TOKENIZER_STATE_NAME = "tokenizer_state.pt"
TOKENIZER_DIR_NAME = "tokenizer"
METADATA_NAME = "metadata.json"
SCHEMA_VERSION = 2

_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------
class CheckpointError(RuntimeError):
    """
    Base exception for all checkpoint operations.

    All checkpoint-specific exceptions inherit from this class, allowing callers
    to catch any checkpoint error without handling the concrete subclass.
    """

    def __init__(self, message: str, *, path: Optional[Path] = None, version: Optional[str] = None) -> None:
        super().__init__(message)
        self.path = path          # Checkpoint directory or file involved, if known
        self.version = version    # Checkpoint version tag, if applicable

    def __str__(self) -> str:
        base = super().__str__()
        parts = [base]
        if self.version:
            parts.append(f"version={self.version!r}")
        if self.path:
            parts.append(f"path={self.path!s}")
        return " (" + ", ".join(parts) + ")" if parts[1:] else base


class CheckpointSaveError(CheckpointError):
    """Raised when persisting a checkpoint fails due to I/O, serialization, or staging issues."""

    def __init__(self, message: str, *,
        path: Optional[Path] = None,
        version: Optional[str] = None,
        stage: Optional[str] = None,   # e.g., "staging", "commit", "archive"
    ) -> None:
        super().__init__(message, path=path, version=version)
        self.stage = stage


class CheckpointLoadError(CheckpointError):
    """Raised when loading a checkpoint fails – missing files, corruption, or incompatible format."""

    def __init__(self, message: str, *,
        path: Optional[Path] = None,
        version: Optional[str] = None,
        format: Optional[str] = None,   # Expected format that caused the failure
    ) -> None:
        super().__init__(message, path=path, version=version)
        self.format = format


class CheckpointIntegrityError(CheckpointError):
    """
    Raised when a checkpoint fails integrity verification.

    This includes missing files, size mismatches, or SHA‑256 hash mismatches.
    More specific subclasses reflect the exact nature of the failure.
    """

    def __init__(self, message: str, *, path: Optional[Path] = None, version: Optional[str] = None) -> None:
        super().__init__(message, path=path, version=version)


class CheckpointHashMismatchError(CheckpointIntegrityError):
    """Raised when a file's computed SHA‑256 does not match the manifest value."""

    def __init__( self, message: str, *,
        path: Optional[Path] = None,
        version: Optional[str] = None,
        relative_path: Optional[str] = None,
        expected_hash: Optional[str] = None,
        actual_hash: Optional[str] = None,
    ) -> None:
        super().__init__(message, path=path, version=version)
        self.relative_path = relative_path
        self.expected_hash = expected_hash
        self.actual_hash = actual_hash


class CheckpointSizeMismatchError(CheckpointIntegrityError):
    """Raised when a file's size does not match the manifest record."""

    def __init__(
        self,
        message: str,
        *,
        path: Optional[Path] = None,
        version: Optional[str] = None,
        relative_path: Optional[str] = None,
        expected_size: Optional[int] = None,
        actual_size: Optional[int] = None,
    ) -> None:
        super().__init__(message, path=path, version=version)
        self.relative_path = relative_path
        self.expected_size = expected_size
        self.actual_size = actual_size


class CheckpointManifestError(CheckpointIntegrityError):
    """Raised when the manifest file itself is missing, malformed, or cannot be parsed."""

    def __init__(self, message: str, *, path: Optional[Path] = None, version: Optional[str] = None) -> None:
        super().__init__(message, path=path, version=version)


class CheckpointVersionError(CheckpointError):
    """
    Raised for version‑related problems.

    This includes unsafe version strings (path traversal), missing version
    resolution when "latest" cannot be resolved, or an invalid version format.
    """

    def __init__(self, message: str, *, version: Optional[str] = None, reason: Optional[str] = None
    ) -> None:
        super().__init__(message, version=version)
        self.reason = reason   # e.g., "path traversal", "invalid characters", "not found"


class CheckpointNotFoundError(CheckpointVersionError):
    """Raised when a specific checkpoint version does not exist (including after "latest" resolution)."""

    def __init__(self, message: str, *, version: Optional[str] = None, path: Optional[Path] = None) -> None:
        super().__init__(message, version=version)
        self.path = path


class CheckpointTokenizerError(CheckpointError):
    """
    Raised when tokenizer persistence fails.

    This can occur when the tokenizer does not implement any supported
    serialization method, or when the saved tokenizer state cannot be restored.
    """

    def __init__(self, message: str, *, tokenizer_type: Optional[str] = None,
        expected_methods: Optional[list[str]] = None, path: Optional[Path] = None ) -> None:
        super().__init__(message, path=path)
        self.tokenizer_type = tokenizer_type
        self.expected_methods = expected_methods or []


class CheckpointIncompatibleError(CheckpointLoadError):
    """
    Raised when a checkpoint's state dict is incompatible with the target model.

    This replaces generic load errors with a detailed breakdown of missing,
    unexpected, or shape‑mismatched keys.
    """

    def __init__(self, message: str, *, path: Optional[Path] = None, version: Optional[str] = None,
                 missing_keys: Optional[list[str]] = None, unexpected_keys: Optional[list[str]] = None,
                 mismatched_keys: Optional[list[dict[str, Any]]] = None) -> None:
        super().__init__(message, path=path, version=version)
        self.missing_keys = missing_keys or []
        self.unexpected_keys = unexpected_keys or []
        self.mismatched_keys = mismatched_keys or []

    def __str__(self) -> str:
        base = super().__str__()
        details = []
        if self.missing_keys:
            details.append(f"missing_keys={self.missing_keys}")
        if self.unexpected_keys:
            details.append(f"unexpected_keys={self.unexpected_keys}")
        if self.mismatched_keys:
            details.append(f"mismatched_keys={self.mismatched_keys}")
        return f"{base} ({', '.join(details)})" if details else base


class CheckpointArchiveError(CheckpointError):
    """Raised when creating, reading, or extracting a checkpoint archive fails."""

    def __init__(self, message: str, *, path: Optional[Path] = None,
                 version: Optional[str] = None, archive_format: str = "tar.gz") -> None:
        super().__init__(message, path=path, version=version)
        self.archive_format = archive_format


class CheckpointRetentionError(CheckpointError):
    """Raised when retention policy enforcement fails (e.g., deletion of a protected checkpoint)."""

    def __init__(
        self, message: str, *, version: Optional[str] = None, keep_limit: Optional[int] = None
    ) -> None:
        super().__init__(message, version=version)
        self.keep_limit = keep_limit

# ---------------------------------------------------------------------------
# Manifest model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CheckpointFileInfo:
    """Integrity metadata for a single file tracked by a checkpoint manifest."""

    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class CheckpointRecord:
    """Serializable checkpoint manifest used for discovery and verification."""

    version: str
    format: CheckpointFormat
    created_at: str
    path: str
    files: Dict[str, CheckpointFileInfo]
    epoch: Optional[int] = None
    step: Optional[int] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    framework: Dict[str, str] = field(default_factory=dict)
    tokenizer_kind: Optional[str] = None
    schema_version: int = SCHEMA_VERSION

    def to_json_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["files"] = {
            name: asdict(info) if isinstance(info, CheckpointFileInfo) else info
            for name, info in self.files.items()
        }
        return data

    @classmethod
    def from_json_dict(cls, data: Mapping[str, Any]) -> "CheckpointRecord":
        files = {
            name: CheckpointFileInfo(
                size_bytes=int(info["size_bytes"]),
                sha256=str(info["sha256"]),
            )
            for name, info in dict(data.get("files", {})).items()
        }
        return cls(
            version=str(data["version"]),
            format=normalize_format(data["format"]),
            created_at=str(data.get("created_at") or utc_now_iso()),
            path=str(data.get("path", "")),
            files=files,
            epoch=data.get("epoch"),
            step=data.get("step"),
            metrics=dict(data.get("metrics") or {}),
            metadata=dict(data.get("metadata") or {}),
            framework=dict(data.get("framework") or {}),
            tokenizer_kind=data.get("tokenizer_kind"),
            schema_version=int(data.get("schema_version", 1)),
        )


# ---------------------------------------------------------------------------
# Validation and path helpers
# ---------------------------------------------------------------------------
def normalize_format(format_value: Any) -> CheckpointFormat:
    """Normalize accepted checkpoint format aliases into canonical names."""
    value = str(format_value).lower().strip()
    if value in {"torch", "pt", "pth", "pytorch"}:
        return "torch"
    if value in {"npz", "numpy"}:
        return "npz"
    raise ValueError("Unsupported checkpoint format. Use 'torch' or 'npz'.")


def utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat()


def make_version(version: Optional[str]) -> str:
    """Create or validate a checkpoint version tag."""
    if version is None:
        return f"v_{_dt.datetime.now(tz=_dt.timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
    return sanitize_version(version)


def sanitize_version(version: str) -> str:
    """Validate a version string and reject path traversal/separator usage."""
    if not isinstance(version, str) or not version:
        raise CheckpointVersionError("Checkpoint version must be a non-empty string.")
    if version == "latest":
        raise CheckpointVersionError("'latest' must be resolved by CheckpointManager before path access.")
    if not _VERSION_RE.match(version) or ".." in version or "/" in version or "\\" in version:
        raise CheckpointVersionError(
            "Unsafe checkpoint version. Use only letters, numbers, '.', '_', and '-' without path separators."
        )
    return version


def ensure_directory(path: str | os.PathLike[str] | Path) -> Path:
    """Create and return an absolute checkpoint base directory."""
    resolved = Path(path).expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    logger.debug("Ensured checkpoint directory exists: %s", resolved)
    return resolved


def resolve_checkpoint_path(base_dir: Path, version: str) -> Path:
    """Resolve a safe checkpoint directory path under ``base_dir``."""
    safe_version = sanitize_version(version)
    resolved_base = Path(base_dir).expanduser().resolve()
    path = (resolved_base / safe_version).resolve()
    if resolved_base not in path.parents and path != resolved_base:
        raise CheckpointVersionError(f"Resolved checkpoint path escapes base_dir: {path}")
    return path


def first_existing(*paths: Path) -> Optional[Path]:
    """Return the first path that exists, or None."""
    for path in paths:
        if path.exists():
            return path
    return None


def safe_rmtree(path: Path) -> None:
    """Remove a directory tree when present."""
    if path.exists():
        shutil.rmtree(path)


# ---------------------------------------------------------------------------
# Atomic I/O and hashing
# ---------------------------------------------------------------------------
def fsync_directory(path: Path) -> None:
    """Best-effort directory fsync for durable atomic replacements."""
    if os.name == "nt":
        return
    try:
        fd = os.open(str(path), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        return


def json_default(value: Any) -> Any:
    """JSON serializer for common checkpoint metadata values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return str(value)


def atomic_text_write(text: str, path: Path) -> None:
    """Atomically write text to a file with file fsync and best-effort dir fsync."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
        fsync_directory(path.parent)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def atomic_json_dump(data: Mapping[str, Any], path: Path) -> None:
    """Atomically write a JSON document."""
    encoded = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True, default=json_default)
    atomic_text_write(encoded + "\n", path)


def atomic_torch_save(obj: Any, path: Path) -> None:
    """Atomically write a torch-serialized object."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)
        fsync_directory(path.parent)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Compute a SHA-256 digest for a file."""
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_checkpoint_files(checkpoint_dir: Path, *, exclude: Iterable[str] = ()) -> Dict[str, CheckpointFileInfo]:
    """Hash all files under a checkpoint directory except excluded relative paths."""
    excluded = set(exclude)
    files: Dict[str, CheckpointFileInfo] = {}
    for file_path in sorted(path for path in checkpoint_dir.rglob("*") if path.is_file()):
        relative = file_path.relative_to(checkpoint_dir).as_posix()
        if relative in excluded:
            continue
        stat = file_path.stat()
        files[relative] = CheckpointFileInfo(size_bytes=stat.st_size, sha256=sha256_file(file_path))
    return files


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------
def write_manifest(
    checkpoint_dir: Path,
    *,
    version: str,
    checkpoint_format: CheckpointFormat,
    checkpoint_path: Path,
    epoch: Optional[int],
    step: Optional[int],
    metadata: Mapping[str, Any],
    metrics: Mapping[str, Any],
    tokenizer_kind: Optional[str],
    exclude: Iterable[str] = (MANIFEST_NAME,),
) -> CheckpointRecord:
    """Build, write, and pre-verify a manifest for a staged checkpoint."""
    files = hash_checkpoint_files(checkpoint_dir, exclude=exclude)
    record = CheckpointRecord(
        version=sanitize_version(version),
        format=checkpoint_format,
        created_at=utc_now_iso(),
        path=str(checkpoint_path),
        files=files,
        epoch=epoch,
        step=step,
        metadata=dict(metadata or {}),
        metrics=dict(metrics or {}),
        tokenizer_kind=tokenizer_kind,
        framework={
            "python": platform.python_version(),
            "torch": getattr(torch, "__version__", "unknown"),
            "numpy": getattr(np, "__version__", "unknown"),
            "platform": platform.platform(),
        },
    )
    atomic_json_dump(record.to_json_dict(), checkpoint_dir / MANIFEST_NAME)
    verify_files_against_record(checkpoint_dir, record)
    logger.debug("Wrote checkpoint manifest for version '%s'", version)
    return record


def read_manifest(checkpoint_dir: Path, *, missing_ok: bool = False) -> Optional[CheckpointRecord]:
    """Read a manifest from a checkpoint directory."""
    manifest_path = checkpoint_dir / MANIFEST_NAME
    if not manifest_path.exists():
        if missing_ok:
            return None
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as fh:
        return CheckpointRecord.from_json_dict(json.load(fh))


def verify_files_against_record(checkpoint_dir: Path, record: CheckpointRecord) -> bool:
    """Verify manifest-tracked files for existence, size, and SHA-256 hash."""
    errors: list[str] = []
    for relative_name, expected in record.files.items():
        file_path = checkpoint_dir / relative_name
        if not file_path.exists():
            errors.append(f"missing file: {relative_name}")
            continue
        actual_size = file_path.stat().st_size
        actual_hash = sha256_file(file_path)
        if actual_size != expected.size_bytes:
            errors.append(f"size mismatch for {relative_name}: {actual_size} != {expected.size_bytes}")
        if actual_hash != expected.sha256:
            errors.append(f"sha256 mismatch for {relative_name}: {actual_hash} != {expected.sha256}")

    if errors:
        raise CheckpointIntegrityError("Checkpoint integrity verification failed: " + "; ".join(errors))
    return True


# ---------------------------------------------------------------------------
# Staging, archives, and legacy support
# ---------------------------------------------------------------------------
def prepare_staging_dir(base_dir: Path, version: str, *, allow_overwrite: bool) -> tuple[Path, Path]:
    """Create a temporary staging directory for a future checkpoint commit."""
    final_dir = resolve_checkpoint_path(base_dir, version)
    if final_dir.exists() and not allow_overwrite:
        raise FileExistsError(f"Checkpoint '{version}' already exists at {final_dir}")
    staging_dir = (base_dir / f".tmp-{version}-{uuid.uuid4().hex}").resolve()
    staging_dir.mkdir(parents=True, exist_ok=False)
    logger.debug("Prepared checkpoint staging dir: %s", staging_dir)
    return final_dir, staging_dir


def commit_staging_dir(staging_dir: Path, final_dir: Path, *, allow_overwrite: bool) -> None:
    """Atomically move a staged checkpoint into its final location."""
    backup_dir: Optional[Path] = None
    base_dir = final_dir.parent
    try:
        if final_dir.exists():
            if not allow_overwrite:
                raise FileExistsError(f"Checkpoint already exists: {final_dir}")
            backup_dir = base_dir / f".replace-{final_dir.name}-{uuid.uuid4().hex}"
            final_dir.rename(backup_dir)
        staging_dir.rename(final_dir)
        fsync_directory(base_dir)
        if backup_dir is not None:
            shutil.rmtree(backup_dir, ignore_errors=True)
    except Exception:
        if backup_dir is not None and backup_dir.exists() and not final_dir.exists():
            backup_dir.rename(final_dir)
        raise


def archive_checkpoint_dir(base_dir: Path, version: str, *, overwrite: bool = True) -> Path:
    """Create a tar.gz archive for a checkpoint directory and write its hash file."""
    safe_version = sanitize_version(version)
    checkpoint_dir = resolve_checkpoint_path(base_dir, safe_version)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint '{safe_version}' does not exist: {checkpoint_dir}")

    archive_base = base_dir / safe_version
    archive_path = base_dir / f"{safe_version}.tar.gz"
    if archive_path.exists() and not overwrite:
        raise FileExistsError(f"Archive already exists: {archive_path}")
    if archive_path.exists():
        archive_path.unlink()

    created_path = Path(
        shutil.make_archive(
            base_name=str(archive_base),
            format="gztar",
            root_dir=str(checkpoint_dir),
        )
    )
    atomic_text_write(sha256_file(created_path), base_dir / f"{safe_version}.tar.gz.sha256")
    logger.info("Archived checkpoint '%s' to %s", safe_version, created_path)
    return created_path


def looks_like_legacy_checkpoint(path: Path) -> bool:
    """Return True for old checkpoint layouts without a manifest."""
    return path.is_dir() and ((path / LEGACY_TORCH_CHECKPOINT_NAME).exists() or (path / NPZ_WEIGHTS_NAME).exists())


def build_legacy_record(path: Path) -> CheckpointRecord:
    """Create an in-memory manifest for a legacy checkpoint directory."""
    if not looks_like_legacy_checkpoint(path):
        raise CheckpointLoadError(f"Directory does not look like a legacy checkpoint: {path}")
    fmt: CheckpointFormat = "torch" if (path / LEGACY_TORCH_CHECKPOINT_NAME).exists() else "npz"
    created_at = _dt.datetime.fromtimestamp(path.stat().st_mtime, tz=_dt.timezone.utc).isoformat()
    return CheckpointRecord(
        version=path.name,
        format=fmt,
        created_at=created_at,
        path=str(path),
        files=hash_checkpoint_files(path, exclude={MANIFEST_NAME}),
        metadata={"legacy_layout": True},
        schema_version=1,
    )


# ---------------------------------------------------------------------------
# Torch, NumPy, RNG, and tokenizer helpers
# ---------------------------------------------------------------------------
def recursive_to_cpu(value: Any) -> Any:
    """Recursively detach tensors and move them to CPU for serialization."""
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, Mapping):
        return {key: recursive_to_cpu(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(recursive_to_cpu(item) for item in value)
    if isinstance(value, list):
        return [recursive_to_cpu(item) for item in value]
    return value


def model_state_to_numpy(model: torch.nn.Module) -> Dict[str, np.ndarray]:
    """Convert a model state_dict into CPU NumPy arrays for NPZ persistence."""
    if not hasattr(model, "state_dict"):
        raise TypeError("model must provide state_dict()")
    arrays: Dict[str, np.ndarray] = {}
    for name, tensor in model.state_dict().items():
        if torch.is_tensor(tensor):
            arrays[name] = tensor.detach().cpu().numpy()
    return arrays


def load_npz_into_model(
    model: torch.nn.Module,
    weight_file: Path,
    *,
    strict: bool = True,
) -> Dict[str, Any]:
    """Load NPZ arrays into a model with compatibility reporting."""
    if not weight_file.exists():
        raise FileNotFoundError(f"NPZ weights file not found: {weight_file}")

    state = model.state_dict()
    loaded: set[str] = set()
    unexpected: list[str] = []
    mismatched: list[Dict[str, Any]] = []

    with np.load(weight_file, allow_pickle=False) as weights:
        for name in weights.files:
            if name not in state:
                unexpected.append(name)
                continue
            source = torch.as_tensor(weights[name], dtype=state[name].dtype, device=state[name].device)
            if tuple(source.shape) != tuple(state[name].shape):
                mismatched.append(
                    {
                        "name": name,
                        "checkpoint_shape": tuple(source.shape),
                        "model_shape": tuple(state[name].shape),
                    }
                )
                continue
            state[name] = source
            loaded.add(name)

    missing = [name for name in state.keys() if name not in loaded]
    if strict and (missing or unexpected or mismatched):
        raise CheckpointLoadError(
            "NPZ checkpoint is incompatible with the model. "
            f"missing={missing}, unexpected={unexpected}, mismatched={mismatched}"
        )

    incompatible = model.load_state_dict(state, strict=False)
    return {
        "missing_keys": missing or list(getattr(incompatible, "missing_keys", [])),
        "unexpected_keys": unexpected or list(getattr(incompatible, "unexpected_keys", [])),
        "mismatched_keys": mismatched,
    }


def capture_rng_state() -> Dict[str, Any]:
    """Capture Python, NumPy, CPU torch, and CUDA RNG state when available."""
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_all"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore RNG state captured by ``capture_rng_state``."""
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch_cpu" in state:
        torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda_all" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda_all"])


def torch_load(path: Path, *, map_location: Any = "cpu") -> Any:
    """Load a trusted torch checkpoint with broad PyTorch-version compatibility."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def save_tokenizer(tokenizer: Any, checkpoint_dir: Path) -> Optional[str]:
    """Persist supported tokenizer styles and return the persistence kind."""
    if tokenizer is None:
        return None

    if hasattr(tokenizer, "save_pretrained"):
        tokenizer_dir = checkpoint_dir / TOKENIZER_DIR_NAME
        tokenizer_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(str(tokenizer_dir))
        logger.debug("Saved tokenizer through save_pretrained() at %s", tokenizer_dir)
        return "save_pretrained"

    if hasattr(tokenizer, "word_to_id"):
        vocab = getattr(tokenizer, "word_to_id")
        if not isinstance(vocab, Mapping):
            raise CheckpointTokenizerError("tokenizer.word_to_id must be a mapping")
        atomic_json_dump(dict(vocab), checkpoint_dir / TOKENIZER_VOCAB_NAME)
        logger.debug("Saved tokenizer word_to_id vocabulary")
        return "word_to_id"

    if hasattr(tokenizer, "state_dict"):
        atomic_torch_save(tokenizer.state_dict(), checkpoint_dir / TOKENIZER_STATE_NAME)
        logger.debug("Saved tokenizer state_dict()")
        return "state_dict"

    raise CheckpointTokenizerError(
        "Unsupported tokenizer. Provide an object with save_pretrained(), word_to_id, or state_dict()."
    )


def load_tokenizer(tokenizer: Any, checkpoint_dir: Path) -> Any:
    """Load tokenizer state into the provided tokenizer object when possible."""
    if tokenizer is None:
        return None

    tokenizer_dir = checkpoint_dir / TOKENIZER_DIR_NAME
    vocab_path = checkpoint_dir / TOKENIZER_VOCAB_NAME
    state_path = checkpoint_dir / TOKENIZER_STATE_NAME

    if tokenizer_dir.exists() and hasattr(tokenizer, "from_pretrained"):
        loaded = tokenizer.from_pretrained(str(tokenizer_dir))
        logger.debug("Loaded tokenizer through from_pretrained() from %s", tokenizer_dir)
        return loaded

    if vocab_path.exists():
        with vocab_path.open("r", encoding="utf-8") as fh:
            word_to_id = json.load(fh)
        tokenizer.word_to_id = word_to_id
        tokenizer.id_to_word = {int(v): k for k, v in word_to_id.items()}
        tokenizer.vocab_size = len(word_to_id)
        logger.debug("Loaded tokenizer vocabulary from %s", vocab_path)
        return tokenizer

    if state_path.exists() and hasattr(tokenizer, "load_state_dict"):
        tokenizer.load_state_dict(torch_load(state_path, map_location="cpu"))
        logger.debug("Loaded tokenizer state_dict() from %s", state_path)
        return tokenizer

    logger.warning("No tokenizer payload found in checkpoint directory %s", checkpoint_dir)
    return tokenizer


def format_bytes(size_bytes: int) -> str:
    """Human-readable file size formatting for diagnostics."""
    value = float(size_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{size_bytes} B"


def checkpoint_summary(record: CheckpointRecord) -> Dict[str, Any]:
    """Return a concise serializable summary for logs or tests."""
    total_size = sum(info.size_bytes for info in record.files.values())
    return {
        "version": record.version,
        "format": record.format,
        "created_at": record.created_at,
        "epoch": record.epoch,
        "step": record.step,
        "file_count": len(record.files),
        "total_size": format_bytes(total_size),
        "tokenizer_kind": record.tokenizer_kind,
    }


__all__ = [
    "CheckpointError",
    "CheckpointFileInfo",
    "CheckpointFormat",
    "CheckpointIntegrityError",
    "CheckpointLoadError",
    "CheckpointRecord",
    "CheckpointSaveError",
    "CheckpointTokenizerError",
    "CheckpointVersionError",
    "LEGACY_TORCH_CHECKPOINT_NAME",
    "MANIFEST_NAME",
    "METADATA_NAME",
    "NPZ_WEIGHTS_NAME",
    "SCHEMA_VERSION",
    "TOKENIZER_DIR_NAME",
    "TOKENIZER_STATE_NAME",
    "TOKENIZER_VOCAB_NAME",
    "TORCH_CHECKPOINT_NAME",
    "archive_checkpoint_dir",
    "atomic_json_dump",
    "atomic_text_write",
    "atomic_torch_save",
    "build_legacy_record",
    "capture_rng_state",
    "checkpoint_summary",
    "commit_staging_dir",
    "ensure_directory",
    "first_existing",
    "format_bytes",
    "fsync_directory",
    "hash_checkpoint_files",
    "json_default",
    "load_npz_into_model",
    "load_tokenizer",
    "logger",
    "looks_like_legacy_checkpoint",
    "make_version",
    "model_state_to_numpy",
    "normalize_format",
    "prepare_staging_dir",
    "printer",
    "read_manifest",
    "recursive_to_cpu",
    "resolve_checkpoint_path",
    "restore_rng_state",
    "safe_rmtree",
    "sanitize_version",
    "save_tokenizer",
    "sha256_file",
    "torch_load",
    "utc_now_iso",
    "verify_files_against_record",
    "write_manifest",
]
