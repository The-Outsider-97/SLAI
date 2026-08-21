"""Structured exception hierarchy for SLAI checkpointing.

The exception types in this module carry machine-readable context without
performing logging or recovery themselves.  Callers can therefore report an
error, decide whether to retry, and distinguish failures that occurred before
or after a checkpoint became committed.

This module deliberately depends only on the Python standard library.  It is a
foundation module and must never import the checkpoint manager, storage,
manifest, policy, observability, or codec implementations.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Optional


class CheckpointOperation(str, Enum):
    """High-level operation being attempted when an error occurred."""

    SAVE = "save"
    LOAD = "load"
    VERIFY = "verify"
    LIST = "list"
    DELETE = "delete"
    ARCHIVE = "archive"
    RESTORE = "restore"
    RETAIN = "retain"
    LOCK = "lock"
    MIGRATE = "migrate"


class CheckpointStage(str, Enum):
    """Fine-grained stage within a checkpoint operation."""

    VALIDATION = "validation"
    DISCOVERY = "discovery"
    LOCKING = "locking"
    STAGING = "staging"
    SERIALIZATION = "serialization"
    HASHING = "hashing"
    MANIFEST = "manifest"
    COMPATIBILITY = "compatibility"
    INTEGRITY = "integrity"
    COMMIT = "commit"
    DESERIALIZATION = "deserialization"
    CLEANUP = "cleanup"
    ARCHIVAL = "archival"
    RESTORATION = "restoration"


def _enum_value(value: str | Enum | None) -> str | None:
    if isinstance(value, Enum):
        return str(value.value)
    return str(value) if value is not None else None


@dataclass(frozen=True, slots=True)
class CheckpointErrorContext:
    """Immutable, serializable context attached to a checkpoint exception."""

    operation: str | None = None
    stage: str | None = None
    path: Path | None = None
    version: str | None = None
    checkpoint_id: str | None = None
    component: str | None = None
    retryable: bool = False
    committed: bool | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "operation", _enum_value(self.operation))
        object.__setattr__(self, "stage", _enum_value(self.stage))
        if self.path is not None:
            object.__setattr__(self, "path", Path(self.path))
        object.__setattr__(self, "details", MappingProxyType(dict(self.details)))

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible diagnostic context."""

        result: dict[str, Any] = {
            "retryable": self.retryable,
            "committed": self.committed,
        }
        for name in (
            "operation",
            "stage",
            "version",
            "checkpoint_id",
            "component",
        ):
            value = getattr(self, name)
            if value is not None:
                result[name] = value
        if self.path is not None:
            result["path"] = str(self.path)
        if self.details:
            result["details"] = dict(self.details)
        return result


class CheckpointError(RuntimeError):
    """Base class for all expected checkpoint-domain failures."""

    code = "checkpoint_error"
    default_retryable = False

    def __init__(self, message: str, *,
        context: CheckpointErrorContext | None = None,
        operation: str | CheckpointOperation | None = None,
        stage: str | CheckpointStage | None = None,
        path: str | Path | None = None,
        version: str | None = None,
        checkpoint_id: str | None = None,
        component: str | None = None,
        retryable: bool | None = None,
        committed: bool | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(message, str) or not message.strip():
            raise ValueError("checkpoint error message must be a non-empty string")

        supplied = context or CheckpointErrorContext()
        merged_details = dict(supplied.details)
        if details:
            merged_details.update(details)

        self.message = message.strip()
        self.context = CheckpointErrorContext(
            operation=_enum_value(operation) or supplied.operation,
            stage=_enum_value(stage) or supplied.stage,
            path=Path(path) if path is not None else supplied.path,
            version=version if version is not None else supplied.version,
            checkpoint_id=(
                checkpoint_id
                if checkpoint_id is not None
                else supplied.checkpoint_id
            ),
            component=component if component is not None else supplied.component,
            retryable=(
                self.default_retryable
                if retryable is None and context is None
                else supplied.retryable if retryable is None else retryable
            ),
            committed=committed if committed is not None else supplied.committed,
            details=merged_details,
        )
        super().__init__(self.message)

    @property
    def operation(self) -> str | None:
        return self.context.operation

    @property
    def stage(self) -> str | None:
        return self.context.stage

    @property
    def path(self) -> Path | None:
        return self.context.path

    @property
    def version(self) -> str | None:
        return self.context.version

    @property
    def checkpoint_id(self) -> str | None:
        return self.context.checkpoint_id

    @property
    def component(self) -> str | None:
        return self.context.component

    @property
    def retryable(self) -> bool:
        return self.context.retryable

    @property
    def committed(self) -> bool | None:
        return self.context.committed

    @property
    def details(self) -> Mapping[str, Any]:
        return self.context.details

    def with_context(self, **changes: Any) -> "CheckpointError":
        """Return a same-class copy with additional context.

        This is intended for boundary layers that can add an operation, path,
        or version without discarding the original structured details.
        """

        context_fields = set(CheckpointErrorContext.__dataclass_fields__)
        unknown = set(changes) - context_fields
        if unknown:
            raise TypeError(f"unknown checkpoint error context fields: {sorted(unknown)}")
        clone = self.__class__.__new__(self.__class__)
        RuntimeError.__init__(clone, self.message)
        clone.message = self.message
        clone.context = replace(self.context, **changes)
        return clone

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "code": self.code,
            "message": self.message,
            **self.context.to_dict(),
        }

    def __str__(self) -> str:
        fields: list[str] = []
        for name in ("operation", "stage", "version", "checkpoint_id", "component"):
            value = getattr(self.context, name)
            if value is not None:
                fields.append(f"{name}={value!r}")
        if self.path is not None:
            fields.append(f"path={str(self.path)!r}")
        if self.committed is not None:
            fields.append(f"committed={self.committed}")
        return f"{self.message} [{', '.join(fields)}]" if fields else self.message


class CheckpointConfigurationError(CheckpointError):
    code = "checkpoint_configuration_error"


class CheckpointValidationError(CheckpointError):
    code = "checkpoint_validation_error"


class CheckpointVersionError(CheckpointValidationError):
    code = "checkpoint_version_error"

    def __init__(self, message: str, *, reason: str | None = None,
        details: Mapping[str, Any] | None = None, **context: Any) -> None:
        merged = dict(details or {})
        if reason is not None:
            merged["reason"] = reason
        super().__init__(message, details=merged, **context)

    @property
    def reason(self) -> Any:
        return self.details.get("reason")


class CheckpointNotFoundError(CheckpointVersionError):
    code = "checkpoint_not_found"


class CheckpointConflictError(CheckpointError):
    code = "checkpoint_conflict"


class CheckpointSaveError(CheckpointError):
    code = "checkpoint_save_error"


class CheckpointCommitError(CheckpointSaveError):
    code = "checkpoint_commit_error"


class CheckpointLoadError(CheckpointError):
    code = "checkpoint_load_error"

    def __init__(self, message: str, *,
        format: str | None = None,
        details: Mapping[str, Any] | None = None,
        **context: Any) -> None:
        merged = dict(details or {})
        if format is not None:
            merged["format"] = format
        super().__init__(message, details=merged, **context)

    @property
    def format(self) -> Any:
        return self.details.get("format")


class CheckpointObservabilityError(CheckpointError):
    code = "checkpoint_observability_error"


class CheckpointStorageError(CheckpointError):
    code = "checkpoint_storage_error"


class CheckpointPathError(CheckpointStorageError):
    code = "checkpoint_path_error"


class CheckpointLockError(CheckpointStorageError):
    code = "checkpoint_lock_error"


class CheckpointLockTimeoutError(CheckpointLockError):
    code = "checkpoint_lock_timeout"
    default_retryable = True


class CheckpointManifestError(CheckpointValidationError):
    code = "checkpoint_manifest_error"


class CheckpointManifestTooLargeError(CheckpointManifestError):
    code = "checkpoint_manifest_too_large"


class CheckpointManifestVersionError(CheckpointManifestError):
    code = "checkpoint_manifest_version_error"


class CheckpointIntegrityError(CheckpointError):
    code = "checkpoint_integrity_error"


class CheckpointMissingArtifactError(CheckpointIntegrityError):
    code = "checkpoint_missing_artifact"

    def __init__(self, message: str, *,
        relative_path: str | None = None,
        details: Mapping[str, Any] | None = None,
        **context: Any) -> None:
        merged = dict(details or {})
        if relative_path is not None:
            merged["relative_path"] = relative_path
        super().__init__(message, details=merged, **context)

    @property
    def relative_path(self) -> Any:
        return self.details.get("relative_path")


class CheckpointUnexpectedArtifactError(CheckpointIntegrityError):
    code = "checkpoint_unexpected_artifact"

    def __init__(self, message: str, *,
        relative_path: str | None = None,
        details: Mapping[str, Any] | None = None,
        **context: Any) -> None:
        merged = dict(details or {})
        if relative_path is not None:
            merged["relative_path"] = relative_path
        super().__init__(message, details=merged, **context)


class CheckpointHashMismatchError(CheckpointIntegrityError):
    code = "checkpoint_hash_mismatch"

    def __init__(self, message: str,*,
        relative_path: str | None = None,
        expected_hash: str | None = None,
        actual_hash: str | None = None,
        details: Mapping[str, Any] | None = None,
        **context: Any) -> None:
        merged = dict(details or {})
        for name, value in (
            ("relative_path", relative_path),
            ("expected_hash", expected_hash),
            ("actual_hash", actual_hash),
        ):
            if value is not None:
                merged[name] = value
        super().__init__(message, details=merged, **context)

    @property
    def relative_path(self) -> Any:
        return self.details.get("relative_path")

    @property
    def expected_hash(self) -> Any:
        return self.details.get("expected_hash")

    @property
    def actual_hash(self) -> Any:
        return self.details.get("actual_hash")


class CheckpointSizeMismatchError(CheckpointIntegrityError):
    code = "checkpoint_size_mismatch"

    def __init__(self, message: str, *,
        relative_path: str | None = None,
        expected_size: int | None = None,
        actual_size: int | None = None,
        details: Mapping[str, Any] | None = None,
        **context: Any,
    ) -> None:
        merged = dict(details or {})
        for name, value in (
            ("relative_path", relative_path),
            ("expected_size", expected_size),
            ("actual_size", actual_size),
        ):
            if value is not None:
                merged[name] = value
        super().__init__(message, details=merged, **context)

    @property
    def relative_path(self) -> Any:
        return self.details.get("relative_path")

    @property
    def expected_size(self) -> Any:
        return self.details.get("expected_size")

    @property
    def actual_size(self) -> Any:
        return self.details.get("actual_size")


class CheckpointIncompatibleError(CheckpointLoadError):
    code = "checkpoint_incompatible"

    def __init__(self, message: str, *,
        missing_keys: Optional[list[str]] = None,
        unexpected_keys: Optional[list[str]] = None,
        mismatched_keys: Optional[list[Mapping[str, Any]]] = None,
        details: Mapping[str, Any] | None = None, **context: Any) -> None:
        merged = dict(details or {})
        merged.update(
            {
                "missing_keys": list(missing_keys or ()),
                "unexpected_keys": list(unexpected_keys or ()),
                "mismatched_keys": [dict(item) for item in mismatched_keys or ()],
            }
        )
        super().__init__(message, details=merged, **context)

    @property
    def missing_keys(self) -> list[str]:
        return list(self.details.get("missing_keys", ()))

    @property
    def unexpected_keys(self) -> list[str]:
        return list(self.details.get("unexpected_keys", ()))

    @property
    def mismatched_keys(self) -> list[Mapping[str, Any]]:
        return list(self.details.get("mismatched_keys", ()))


class CheckpointCodecError(CheckpointError):
    code = "checkpoint_codec_error"


class CheckpointCodecNotFoundError(CheckpointCodecError):
    code = "checkpoint_codec_not_found"


class CheckpointTokenizerError(CheckpointCodecError):
    """Backward-compatible tokenizer-specific codec failure."""

    code = "checkpoint_tokenizer_error"

    def __init__(self, message: str, *,
        tokenizer_type: str | None = None,
        expected_methods: Optional[list[str]] = None,
        details: Mapping[str, Any] | None = None,
        **context: Any) -> None:
        merged = dict(details or {})
        if tokenizer_type is not None:
            merged["tokenizer_type"] = tokenizer_type
        if expected_methods:
            merged["expected_methods"] = list(expected_methods)
        super().__init__(message, component="tokenizer", details=merged, **context)

    @property
    def tokenizer_type(self) -> Any:
        return self.details.get("tokenizer_type")

    @property
    def expected_methods(self) -> list[str]:
        return list(self.details.get("expected_methods", ()))


class CheckpointArchiveError(CheckpointStorageError):
    code = "checkpoint_archive_error"

    def __init__(self, message: str, *,
        archive_format: str = "tar.gz",
        details: Mapping[str, Any] | None = None,
        **context: Any) -> None:
        merged = dict(details or {})
        merged.setdefault("archive_format", archive_format)
        super().__init__(message, details=merged, **context)

    @property
    def archive_format(self) -> str:
        return str(self.details.get("archive_format", "tar.gz"))


class CheckpointRetentionError(CheckpointError):
    code = "checkpoint_retention_error"

    def __init__(self, message: str, *,
        keep_limit: int | None = None,
        details: Mapping[str, Any] | None = None,
        **context: Any) -> None:
        merged = dict(details or {})
        if keep_limit is not None:
            merged["keep_limit"] = keep_limit
        super().__init__(message, details=merged, **context)

    @property
    def keep_limit(self) -> Any:
        return self.details.get("keep_limit")


__all__ = [
    "CheckpointArchiveError",
    "CheckpointCodecError",
    "CheckpointCodecNotFoundError",
    "CheckpointCommitError",
    "CheckpointConfigurationError",
    "CheckpointConflictError",
    "CheckpointError",
    "CheckpointErrorContext",
    "CheckpointHashMismatchError",
    "CheckpointIncompatibleError",
    "CheckpointIntegrityError",
    "CheckpointLoadError",
    "CheckpointLockError",
    "CheckpointLockTimeoutError",
    "CheckpointManifestError",
    "CheckpointManifestTooLargeError",
    "CheckpointManifestVersionError",
    "CheckpointMissingArtifactError",
    "CheckpointNotFoundError",
    "CheckpointObservabilityError",
    "CheckpointOperation",
    "CheckpointPathError",
    "CheckpointRetentionError",
    "CheckpointSaveError",
    "CheckpointSizeMismatchError",
    "CheckpointStage",
    "CheckpointStorageError",
    "CheckpointTokenizerError",
    "CheckpointUnexpectedArtifactError",
    "CheckpointValidationError",
    "CheckpointVersionError",
]