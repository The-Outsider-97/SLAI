"""Integrity-aware file transfer service built on ``src.functions.storage``.

``storage.py`` already owns storage backends and their upload/download API.
This module intentionally does not reimplement those backends. Instead it
adds transfer concerns that storage does not currently own:

* maximum-size enforcement before upload
* suffix/content-type policy validation
* SHA-256 integrity calculation and optional verification
* atomic download-to-path writes
* optional confinement of downloaded files to a configured root
* structured transfer receipts

Large uploads are spooled to memory and then disk before the storage backend
is called. This allows integrity/size validation to finish before a remote or
local object is committed.
"""

from __future__ import annotations

import hashlib
import io
import mimetypes
import os
import tempfile

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Dict, FrozenSet, Mapping, Optional, Union

from .storage import Storage
from .utils.functions_error import *
from logs.logger import get_logger # pyright: ignore[reportMissingImports]

logger = get_logger("File Transfer")

_DEFAULT_MAX_BYTES = 100 * 1024 * 1024
_DEFAULT_CHUNK_SIZE = 1024 * 1024
_DEFAULT_SPOOL_MAX_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True)
class TransferPolicy:
    """Validation and resource limits for file transfer operations."""

    max_bytes: int = _DEFAULT_MAX_BYTES
    chunk_size: int = _DEFAULT_CHUNK_SIZE
    spool_max_bytes: int = _DEFAULT_SPOOL_MAX_BYTES
    allowed_suffixes: FrozenSet[str] = field(default_factory=frozenset)
    allowed_content_types: FrozenSet[str] = field(default_factory=frozenset)
    allow_overwrite: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.max_bytes, bool) or int(self.max_bytes) <= 0:
            raise TransferValidationError("max_bytes must be a positive integer")
        if isinstance(self.chunk_size, bool) or int(self.chunk_size) <= 0:
            raise TransferValidationError("chunk_size must be a positive integer")
        if isinstance(self.spool_max_bytes, bool) or int(self.spool_max_bytes) <= 0:
            raise TransferValidationError(
                "spool_max_bytes must be a positive integer"
            )
        if self.chunk_size > self.max_bytes:
            raise TransferValidationError(
                "chunk_size must not exceed max_bytes"
            )

        suffixes = frozenset(_normalize_suffix(v) for v in self.allowed_suffixes)
        content_types = frozenset(
            _normalize_content_type(v) for v in self.allowed_content_types
        )

        object.__setattr__(self, "max_bytes", int(self.max_bytes))
        object.__setattr__(self, "chunk_size", int(self.chunk_size))
        object.__setattr__(self, "spool_max_bytes", int(self.spool_max_bytes))
        object.__setattr__(self, "allowed_suffixes", suffixes)
        object.__setattr__(self, "allowed_content_types", content_types)


@dataclass(frozen=True)
class TransferReceipt:
    """Result metadata for a completed upload/download."""

    direction: str
    size_bytes: int
    sha256: str
    filename: str
    object_key: Optional[str] = None
    local_path: Optional[str] = None
    content_type: Optional[str] = None
    completed_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "direction": self.direction,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "filename": self.filename,
            "object_key": self.object_key,
            "local_path": self.local_path,
            "content_type": self.content_type,
            "completed_at": self.completed_at.isoformat(),
        }


class FileTransferService:
    """Higher-level transfer operations backed by an existing ``Storage``."""

    def __init__(
        self,
        storage: Storage,
        *,
        policy: Optional[TransferPolicy] = None,
        download_root: Optional[Union[str, Path]] = None,
    ) -> None:
        if not isinstance(storage, Storage):
            raise TypeError("storage must be a Storage instance")

        self.storage = storage
        self.policy = policy or TransferPolicy()
        self.download_root = (
            Path(download_root).expanduser().resolve()
            if download_root is not None
            else None
        )

        if self.download_root is not None:
            self.download_root.mkdir(parents=True, exist_ok=True)

    def upload_path(
        self,
        source: Union[str, Path],
        *,
        filename: Optional[str] = None,
        subpath: str = "",
        content_type: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        expected_sha256: Optional[str] = None,
    ) -> TransferReceipt:
        """Validate and upload a local file."""

        source_path = Path(source).expanduser()
        if not source_path.exists():
            raise TransferValidationError(f"source file does not exist: {source_path}")
        if not source_path.is_file():
            raise TransferValidationError(f"source is not a regular file: {source_path}")

        effective_name = filename or source_path.name
        with source_path.open("rb") as handle:
            return self.upload_stream(
                handle,
                filename=effective_name,
                subpath=subpath,
                content_type=content_type,
                metadata=metadata,
                expected_sha256=expected_sha256,
            )

    def upload_bytes(
        self,
        data: bytes,
        *,
        filename: str,
        subpath: str = "",
        content_type: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        expected_sha256: Optional[str] = None,
    ) -> TransferReceipt:
        """Validate and upload an in-memory byte payload."""

        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError("data must be bytes-like")
        raw = bytes(data)
        if len(raw) > self.policy.max_bytes:
            raise TransferSizeError(len(raw), self.policy.max_bytes)

        return self.upload_stream(
            io.BytesIO(raw),
            filename=filename,
            subpath=subpath,
            content_type=content_type,
            metadata=metadata,
            expected_sha256=expected_sha256,
        )

    def upload_stream(
        self,
        stream: BinaryIO,
        *,
        filename: str,
        subpath: str = "",
        content_type: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        expected_sha256: Optional[str] = None,
    ) -> TransferReceipt:
        """Upload a binary stream after bounded spooling and integrity checks."""

        if stream is None or not callable(getattr(stream, "read", None)):
            raise TypeError("stream must provide a callable read() method")

        filename = _validate_filename(filename)
        normalized_type = self._validate_file_policy(filename, content_type)
        expected_digest = _normalize_sha256(expected_sha256)

        digest = hashlib.sha256()
        size = 0

        with tempfile.SpooledTemporaryFile(
            max_size=self.policy.spool_max_bytes,
            mode="w+b",
        ) as spool:
            while True:
                chunk = stream.read(self.policy.chunk_size)
                if chunk in (b"", None):
                    break
                if not isinstance(chunk, (bytes, bytearray, memoryview)):
                    raise TransferValidationError(
                        "binary stream returned non-bytes content"
                    )

                chunk = bytes(chunk)
                size += len(chunk)
                if size > self.policy.max_bytes:
                    raise TransferSizeError(size, self.policy.max_bytes)

                digest.update(chunk)
                spool.write(chunk)

            actual_digest = digest.hexdigest()
            if expected_digest and actual_digest != expected_digest:
                raise TransferIntegrityError(expected_digest, actual_digest)

            normalized_metadata = _normalize_metadata(metadata)
            normalized_metadata.update(
                {
                    "sha256": actual_digest,
                    "size_bytes": str(size),
                }
            )
            if normalized_type:
                normalized_metadata["content_type"] = normalized_type

            spool.seek(0)
            try:
                object_key = self.storage.upload(
                    spool, # type: ignore
                    filename=filename,
                    subpath=subpath,
                    metadata=normalized_metadata,
                )
            except StorageError as exc:
                raise FileTransferError(
                    f"Storage upload failed for '{filename}': {exc}"
                ) from exc

        receipt = TransferReceipt(
            direction="upload",
            size_bytes=size,
            sha256=actual_digest,
            filename=filename,
            object_key=object_key,
            content_type=normalized_type,
        )
        logger.info(
            "Uploaded object key=%s size_bytes=%s sha256=%s",
            object_key,
            size,
            actual_digest,
        )
        return receipt

    def download_bytes(
        self,
        object_key: str,
        *,
        expected_sha256: Optional[str] = None,
    ) -> tuple[bytes, TransferReceipt]:
        """Download an object and verify its size/integrity.

        Note: the current ``Storage.download`` contract returns complete bytes,
        so the backend has already materialized the object before this layer can
        enforce ``max_bytes``. A future streaming method on ``StorageBackend``
        would permit pre-allocation enforcement for very large downloads.
        """

        key = _require_text(object_key, "object_key")
        expected_digest = _normalize_sha256(expected_sha256)

        try:
            data = self.storage.download(key)
        except StorageError as exc:
            raise FileTransferError(
                f"Storage download failed for '{key}': {exc}"
            ) from exc

        if not isinstance(data, bytes):
            raise FileTransferError("storage backend returned non-bytes content")

        size = len(data)
        if size > self.policy.max_bytes:
            raise TransferSizeError(size, self.policy.max_bytes)

        actual_digest = hashlib.sha256(data).hexdigest()
        if expected_digest and actual_digest != expected_digest:
            raise TransferIntegrityError(expected_digest, actual_digest)

        filename = Path(key.replace("\\", "/")).name or "download.bin"
        receipt = TransferReceipt(
            direction="download",
            size_bytes=size,
            sha256=actual_digest,
            filename=filename,
            object_key=key,
        )
        return data, receipt

    def download_to_path(
        self,
        object_key: str,
        destination: Union[str, Path],
        *,
        expected_sha256: Optional[str] = None,
        overwrite: Optional[bool] = None,
    ) -> TransferReceipt:
        """Download to a local path using an atomic replace."""

        data, receipt = self.download_bytes(
            object_key,
            expected_sha256=expected_sha256,
        )
        destination_path = self._resolve_destination(destination)
        allow_overwrite = (
            self.policy.allow_overwrite if overwrite is None else bool(overwrite)
        )

        if destination_path.exists() and not allow_overwrite:
            raise TransferDestinationError(
                f"destination already exists: {destination_path}"
            )

        destination_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path: Optional[Path] = None

        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=str(destination_path.parent),
                prefix=f".{destination_path.name}.",
                suffix=".part",
                delete=False,
            ) as handle:
                temp_path = Path(handle.name)
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())

            if destination_path.exists() and not allow_overwrite:
                raise TransferDestinationError(
                    f"destination already exists: {destination_path}"
                )

            os.replace(temp_path, destination_path)
        except PermissionError as exc:
            raise TransferDestinationError(
                f"permission denied writing {destination_path}"
            ) from exc
        except OSError as exc:
            raise TransferDestinationError(
                f"failed writing {destination_path}: {exc}"
            ) from exc
        finally:
            if temp_path is not None and temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    logger.warning(
                        "Failed to remove temporary transfer file %s",
                        temp_path,
                    )

        completed = TransferReceipt(
            direction=receipt.direction,
            size_bytes=receipt.size_bytes,
            sha256=receipt.sha256,
            filename=destination_path.name,
            object_key=receipt.object_key,
            local_path=str(destination_path),
            content_type=receipt.content_type,
        )
        logger.info(
            "Downloaded object key=%s destination=%s size_bytes=%s",
            object_key,
            destination_path,
            receipt.size_bytes,
        )
        return completed

    def _validate_file_policy(
        self,
        filename: str,
        content_type: Optional[str],
    ) -> Optional[str]:
        suffix = Path(filename).suffix.lower()
        if self.policy.allowed_suffixes and suffix not in self.policy.allowed_suffixes:
            raise TransferValidationError(
                f"file suffix '{suffix or '<none>'}' is not allowed"
            )

        normalized_type = (
            _normalize_content_type(content_type)
            if content_type is not None
            else None
        )
        if normalized_type is None:
            guessed, _ = mimetypes.guess_type(filename)
            normalized_type = guessed.lower() if guessed else None

        if (
            self.policy.allowed_content_types
            and normalized_type not in self.policy.allowed_content_types
        ):
            raise TransferValidationError(
                f"content type '{normalized_type or '<unknown>'}' is not allowed"
            )

        return normalized_type

    def _resolve_destination(self, destination: Union[str, Path]) -> Path:
        raw = Path(destination).expanduser()

        if self.download_root is None:
            return raw.resolve()

        candidate = (
            raw.resolve()
            if raw.is_absolute()
            else (self.download_root / raw).resolve()
        )
        try:
            candidate.relative_to(self.download_root)
        except ValueError as exc:
            raise TransferDestinationError(
                f"destination escapes download root: {destination}"
            ) from exc
        return candidate


def _normalize_suffix(value: str) -> str:
    normalized = _require_text(value, "allowed suffix").lower()
    return normalized if normalized.startswith(".") else f".{normalized}"


def _normalize_content_type(value: str) -> str:
    normalized = _require_text(value, "content type").lower()
    if "/" not in normalized:
        raise TransferValidationError(
            f"invalid content type '{normalized}'"
        )
    return normalized


def _normalize_sha256(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized = _require_text(value, "expected_sha256").lower()
    if len(normalized) != 64 or any(c not in "0123456789abcdef" for c in normalized):
        raise TransferValidationError(
            "expected_sha256 must be a 64-character hexadecimal digest"
        )
    return normalized


def _validate_filename(filename: str) -> str:
    normalized = _require_text(filename, "filename")
    candidate = Path(normalized).name
    if candidate in {"", ".", ".."}:
        raise TransferValidationError("filename is invalid")
    return candidate


def _require_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise TransferValidationError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise TransferValidationError(f"{field_name} must not be empty")
    return normalized


def _normalize_metadata(
    metadata: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    if metadata is None:
        return {}
    if not isinstance(metadata, Mapping):
        raise TransferValidationError("metadata must be a mapping")
    return {str(key): value for key, value in metadata.items()}


__all__ = [
    "FileTransferError",
    "FileTransferService",
    "TransferDestinationError",
    "TransferIntegrityError",
    "TransferPolicy",
    "TransferReceipt",
    "TransferSizeError",
    "TransferValidationError",
]


# -------------------------------------------------------------------------
# Temporary self-test block
# -------------------------------------------------------------------------
if __name__ == "__main__":
    from pathlib import Path, PurePosixPath
    import shutil

    from .storage import LocalStorage

    print("\n=== Running FileTransfer self-test ===\n")

    root = Path("data/_file_transfer_self_test").resolve()
    if root.exists():
        shutil.rmtree(root)

    backend = LocalStorage(base_path=str(root / "storage"))
    storage = Storage(backend=backend, generate_unique_filename=False)
    service = FileTransferService(
        storage,
        policy=TransferPolicy(
            max_bytes=1024 * 1024,
            allowed_suffixes=frozenset({".txt", ".json"}),
        ),
        download_root=root / "downloads",
    )

    payload = b"SLAI file-transfer integrity test"
    expected = hashlib.sha256(payload).hexdigest()

    uploaded = service.upload_bytes(
        payload,
        filename="sample.txt",
        subpath="tests",
        expected_sha256=expected,
    )
    assert uploaded.object_key == "tests/sample.txt"
    assert uploaded.sha256 == expected

    downloaded = service.download_to_path(
        uploaded.object_key,
        "sample.txt",
        expected_sha256=expected,
    )
    assert Path(downloaded.local_path).read_bytes() == payload # type: ignore

    try:
        service.upload_bytes(
            b"x",
            filename="blocked.exe",
        )
    except TransferValidationError:
        pass
    else:
        raise AssertionError("Disallowed suffix should fail")

    shutil.rmtree(root)
    print("✔ FileTransfer self-test passed")
