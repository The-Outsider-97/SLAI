"""Pluggable file storage abstraction with local filesystem and S3 backends.

Provides a consistent interface for uploading, downloading, deleting, and
resolving URLs for stored objects.
"""

from __future__ import annotations

import io
import os
import shutil
import tempfile
import uuid

from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Dict, Optional
from urllib.parse import quote

from .utils.config_loader import get_config_section
from .utils.functions_error import (StorageBackendError, StorageDeleteError, StorageDownloadError,
                                    StorageError, StorageNotFoundError, StoragePermissionError, StorageUploadError)
from .utils.storage_backend import StorageBackend
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Storage")
printer = PrettyPrinter

_DEFAULT_LOCAL_BASE_PATH = "data/storage"
_DEFAULT_PRESIGNED_URL_TTL_SECONDS = 3600


def _require_non_empty_string(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    return normalized


def _normalize_optional_string(value: Optional[str], field_name: str) -> Optional[str]:
    if value is None:
        return None
    return _require_non_empty_string(value, field_name)


def _normalize_metadata(metadata: Optional[Dict[str, Any]]) -> Dict[str, str]:
    if metadata is None:
        return {}
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be a dictionary or None")
    return {str(key): str(value) for key, value in metadata.items()}


def _ensure_binary_fileobj(file_obj: BinaryIO) -> BinaryIO:
    if file_obj is None:
        raise TypeError("file_obj must not be None")
    read = getattr(file_obj, "read", None)
    if not callable(read):
        raise TypeError("file_obj must provide a callable read() method")
    return file_obj


def _normalize_storage_key(path: str, *, field_name: str = "path") -> str:
    normalized = _require_non_empty_string(path, field_name).replace("\\", "/")
    pure_path = PurePosixPath(normalized)

    if pure_path.is_absolute():
        raise StoragePermissionError(normalized, operation="access")

    invalid_parts = {"", ".", ".."}
    parts = list(pure_path.parts)
    if not parts or any(part in invalid_parts for part in parts):
        raise StoragePermissionError(normalized, operation="access")

    return "/".join(parts)


def _join_storage_key(*parts: str) -> str:
    normalized_parts = [
        _normalize_storage_key(part, field_name="path segment")
        for part in parts
        if part and str(part).strip()
    ]
    if not normalized_parts:
        raise ValueError("at least one path segment is required")
    return "/".join(normalized_parts)


def _sanitize_filename(filename: str) -> str:
    normalized = _require_non_empty_string(filename, "filename")
    candidate = Path(normalized).name.strip()
    if candidate in {"", ".", ".."}:
        raise ValueError("filename must not be empty")
    return candidate


class LocalStorage(StorageBackend):
    """Store files on the local filesystem."""

    def __init__(self, base_path: str, base_url: Optional[str] = None):
        normalized_base_path = _require_non_empty_string(base_path, "base_path")
        self.base_path = Path(normalized_base_path).expanduser().resolve()
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.base_url = base_url.rstrip("/") if base_url else None

    def _resolve_path(self, path: str) -> Path:
        key = _normalize_storage_key(path)
        full_path = (self.base_path / key).resolve()
        try:
            full_path.relative_to(self.base_path)
        except ValueError as exc:
            raise StoragePermissionError(key, operation="access") from exc
        return full_path

    def _local_key(self, path: str) -> str:
        return _normalize_storage_key(path)

    def upload(
        self,
        file_obj: BinaryIO,
        path: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        file_obj = _ensure_binary_fileobj(file_obj)
        key = self._local_key(path)
        full_path = self._resolve_path(key)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        _normalize_metadata(metadata)

        temp_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=str(full_path.parent),
                prefix=f".{full_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as tmp_handle:
                temp_path = Path(tmp_handle.name)
                shutil.copyfileobj(file_obj, tmp_handle)
                tmp_handle.flush()
                os.fsync(tmp_handle.fileno())

            os.replace(temp_path, full_path)
            logger.info(f"Uploaded local object '{key}'")
            return key
        except PermissionError as exc:
            raise StoragePermissionError(key, operation="upload") from exc
        except OSError as exc:
            raise StorageUploadError(key, str(exc)) from exc
        finally:
            if temp_path is not None and temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    logger.warning(f"Failed to clean temporary upload file '{temp_path}'")

    def download(self, path: str) -> bytes:
        key = self._local_key(path)
        full_path = self._resolve_path(key)
        try:
            with full_path.open("rb") as handle:
                return handle.read()
        except FileNotFoundError as exc:
            raise StorageNotFoundError(key) from exc
        except PermissionError as exc:
            raise StoragePermissionError(key, operation="download") from exc
        except IsADirectoryError as exc:
            raise StorageDownloadError(key, "path refers to a directory, not a file") from exc
        except OSError as exc:
            raise StorageDownloadError(key, str(exc)) from exc

    def delete(self, path: str) -> None:
        key = self._local_key(path)
        full_path = self._resolve_path(key)
        try:
            full_path.unlink()
            self._prune_empty_parents(full_path.parent)
            logger.info(f"Deleted local object '{key}'")
        except FileNotFoundError as exc:
            raise StorageNotFoundError(key) from exc
        except IsADirectoryError as exc:
            raise StorageDeleteError(key, "refusing to delete directories via file delete API") from exc
        except PermissionError as exc:
            raise StoragePermissionError(key, operation="delete") from exc
        except OSError as exc:
            raise StorageDeleteError(key, str(exc)) from exc

    def get_url(self, path: str) -> str:
        key = self._local_key(path)
        if self.base_url:
            quoted_key = quote(key, safe="/")
            return f"{self.base_url}/{quoted_key}"
        return self._resolve_path(key).as_uri()

    def _prune_empty_parents(self, start_dir: Path) -> None:
        current = start_dir
        while current != self.base_path:
            try:
                current.rmdir()
            except OSError:
                break
            current = current.parent


class S3Storage(StorageBackend):
    """Amazon S3-compatible storage backend."""

    def __init__(
        self,
        bucket: str,
        region: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        public_url_prefix: Optional[str] = None,
        presigned_url_ttl_seconds: int = _DEFAULT_PRESIGNED_URL_TTL_SECONDS,
    ):
        self.bucket = _require_non_empty_string(bucket, "bucket")
        self.public_url_prefix = public_url_prefix.rstrip("/") if public_url_prefix else None
        self.presigned_url_ttl_seconds = max(1, int(presigned_url_ttl_seconds))

        try:
            import boto3
            from botocore.exceptions import BotoCoreError, ClientError
        except ImportError as exc:
            raise StorageBackendError("s3", "boto3 is required for S3 storage") from exc

        self._boto_errors = (BotoCoreError, ClientError)
        self.client = boto3.client(
            "s3",
            region_name=region,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            endpoint_url=endpoint_url,
        )

    def _s3_key(self, path: str) -> str:
        return _normalize_storage_key(path)

    def upload(
        self,
        file_obj: BinaryIO,
        path: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        file_obj = _ensure_binary_fileobj(file_obj)
        key = self._s3_key(path)
        extra_args: Dict[str, Any] = {}
        normalized_metadata = _normalize_metadata(metadata)
        if normalized_metadata:
            extra_args["Metadata"] = normalized_metadata

        try:
            if extra_args:
                self.client.upload_fileobj(file_obj, self.bucket, key, ExtraArgs=extra_args)
            else:
                self.client.upload_fileobj(file_obj, self.bucket, key)
            logger.info(f"Uploaded s3 object 's3://{self.bucket}/{key}'")
            return key
        except self._boto_errors as exc:
            raise StorageUploadError(key, str(exc)) from exc
        except Exception as exc:
            raise StorageBackendError("s3", f"unexpected upload failure for '{key}': {exc}") from exc

    def download(self, path: str) -> bytes:
        key = self._s3_key(path)
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=key)
            body = response["Body"]
            return body.read()
        except self._boto_errors as exc:
            if self._is_missing_key_error(exc):
                raise StorageNotFoundError(key) from exc
            raise StorageDownloadError(key, str(exc)) from exc
        except Exception as exc:
            raise StorageBackendError("s3", f"unexpected download failure for '{key}': {exc}") from exc

    def delete(self, path: str) -> None:
        key = self._s3_key(path)
        try:
            self.client.delete_object(Bucket=self.bucket, Key=key)
            logger.info(f"Deleted s3 object 's3://{self.bucket}/{key}'")
        except self._boto_errors as exc:
            raise StorageDeleteError(key, str(exc)) from exc
        except Exception as exc:
            raise StorageBackendError("s3", f"unexpected delete failure for '{key}': {exc}") from exc

    def get_url(self, path: str) -> str:
        key = self._s3_key(path)
        if self.public_url_prefix:
            quoted_key = quote(key, safe="/")
            return f"{self.public_url_prefix}/{quoted_key}"

        try:
            return self.client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket, "Key": key},
                ExpiresIn=self.presigned_url_ttl_seconds,
            )
        except self._boto_errors as exc:
            raise StorageBackendError("s3", f"cannot generate URL for '{key}': {exc}") from exc
        except Exception as exc:
            raise StorageBackendError(
                "s3",
                f"unexpected URL generation failure for '{key}': {exc}",
            ) from exc

    @staticmethod
    def _is_missing_key_error(exc: BaseException) -> bool:
        response = getattr(exc, "response", None)
        if not isinstance(response, dict):
            return False
        error = response.get("Error") or {}
        code = str(error.get("Code", ""))
        return code in {"404", "NoSuchKey", "NotFound"}


__all__ = [
    "LocalStorage",
    "S3Storage",
    "Storage",
]


class Storage:
    """Facade that provides a backend-neutral storage interface."""

    def __init__(self, backend: StorageBackend, generate_unique_filename: bool = True):
        if not isinstance(backend, StorageBackend):
            raise TypeError("backend must implement StorageBackend")
        self.backend = backend
        self.generate_unique_filename = bool(generate_unique_filename)

    def _unique_filename(self, original_name: str) -> str:
        """Generate a unique filename while preserving the suffix."""
        sanitized_name = _sanitize_filename(original_name)
        suffix = Path(sanitized_name).suffix
        return f"{uuid.uuid4().hex}{suffix}"

    def _build_object_key(self, filename: str, subpath: str = "") -> str:
        final_name = self._unique_filename(filename) if self.generate_unique_filename else _sanitize_filename(filename)
        if subpath and str(subpath).strip():
            return _join_storage_key(subpath, final_name)
        return _normalize_storage_key(final_name, field_name="filename")

    def upload(
        self,
        file_obj: BinaryIO,
        filename: str,
        subpath: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Upload a file and return the backend-neutral object key."""
        object_key = self._build_object_key(filename=filename, subpath=subpath)
        return self.backend.upload(file_obj, object_key, metadata)

    def download(self, path: str) -> bytes:
        object_key = _normalize_storage_key(path)
        return self.backend.download(object_key)

    def delete(self, path: str) -> None:
        object_key = _normalize_storage_key(path)
        self.backend.delete(object_key)

    def get_url(self, path: str) -> str:
        object_key = _normalize_storage_key(path)
        return self.backend.get_url(object_key)

    @classmethod
    def from_config(cls) -> "Storage":
        """Create a storage facade from the 'storage' configuration section."""
        config = get_config_section("storage") or {}
        if not isinstance(config, dict):
            raise StorageError("storage configuration must be a mapping")

        backend_type = str(config.get("backend", "local")).strip().lower() or "local"
        generate_unique = bool(config.get("generate_unique_filename", True))

        if backend_type == "local":
            backend = LocalStorage(
                base_path=str(config.get("base_path", _DEFAULT_LOCAL_BASE_PATH)),
                base_url=config.get("base_url"),
            )
        elif backend_type == "s3":
            bucket = config.get("bucket")
            if not bucket:
                raise StorageError("storage.s3 configuration requires 'bucket'")
            backend = S3Storage(
                bucket=str(bucket),
                region=_normalize_optional_string(config.get("region"), "region"),
                access_key=_normalize_optional_string(config.get("access_key"), "access_key"),
                secret_key=_normalize_optional_string(config.get("secret_key"), "secret_key"),
                endpoint_url=_normalize_optional_string(config.get("endpoint_url"), "endpoint_url"),
                public_url_prefix=_normalize_optional_string(
                    config.get("public_url_prefix"),
                    "public_url_prefix",
                ),
                presigned_url_ttl_seconds=int(
                    config.get(
                        "presigned_url_ttl_seconds",
                        _DEFAULT_PRESIGNED_URL_TTL_SECONDS,
                    )
                ),
            )
        else:
            raise StorageError(f"Unsupported storage backend: {backend_type}")

        return cls(backend=backend, generate_unique_filename=generate_unique)

if __name__ == "__main__":
    print("\n=== Running Storage ===\n")
    printer.status("TEST", "Starting Storage tests", "info")
    backend = LocalStorage(base_path="data/storage", base_url=None)

    storage = Storage(
        backend=backend,
        generate_unique_filename=True
    )
    print(storage)
    print("\n=== Successfully ran the Storage ===\n")