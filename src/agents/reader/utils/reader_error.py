from __future__ import annotations

"""Production-grade, reader-domain error model.

This module intentionally owns only error taxonomy, typed exceptions, safe
serialization, and exception normalization for the Reader subsystem. Generic
path, hashing, formatting, retry, or validation helpers belong in
``reader_helpers.py`` or other shared utility modules.

Design goals
------------
- Keep every public class already used by the Reader subsystem backward compatible.
- Provide stable error codes for logs, telemetry, checkpoints, and API payloads.
- Preserve useful debugging context without leaking sensitive values by default.
- Keep errors reader-specific: parse, recovery, conversion, merge, persistence,
  cache/checkpoint, reader task validation, and reader runtime failures.
- Avoid importing subsystem helper modules to prevent coupling and circular imports.
"""
import re
import traceback

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional


from .config_loader import get_config_section, load_reader_config
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Reader Error")
printer = PrettyPrinter()


class ReaderErrorType(Enum):
    """Reader-domain error taxonomy.

    Existing enum values intentionally retain their historical display strings so
    serialized payloads and string comparisons remain compatible with earlier
    Reader subsystem versions.
    """

    INVALID_TASK = "Invalid task"
    INVALID_INSTRUCTION = "Invalid instruction"
    FILE_NOT_FOUND = "File not found"
    UNSUPPORTED_FORMAT = "Unsupported format"
    PARSE_FAILURE = "Parse failure"
    CONVERSION_FAILURE = "Conversion failure"
    RECOVERY_FAILURE = "Recovery failure"
    MERGE_FAILURE = "Merge failure"
    PERSISTENCE_FAILURE = "Persistence failure"

    # Production-level Reader-specific additions.
    CONFIGURATION_FAILURE = "Configuration failure"
    INVALID_FILE_PATH = "Invalid file path"
    FILE_NOT_REGULAR = "File is not a regular file"
    FILE_ACCESS_DENIED = "File access denied"
    FILE_TOO_LARGE = "File too large"
    DECODE_FAILURE = "Decode failure"
    EMPTY_CONTENT = "Empty content"
    CORRUPT_CONTENT = "Corrupt content"
    MALFORMED_DOCUMENT = "Malformed reader document"
    OUTPUT_FAILURE = "Output failure"
    CHECKPOINT_FAILURE = "Checkpoint failure"
    CACHE_FAILURE = "Cache failure"
    BATCH_FAILURE = "Batch failure"
    TIMEOUT = "Reader timeout"
    CANCELLED = "Reader task cancelled"
    UNEXPECTED_FAILURE = "Unexpected reader failure"


class ReaderErrorStage(Enum):
    """Execution stage where a Reader error occurred."""

    VALIDATION = "validation"
    PLANNING = "planning"
    CONFIG = "config"
    PARSE = "parse"
    DECODE = "decode"
    RECOVERY = "recovery"
    CONVERSION = "conversion"
    MERGE = "merge"
    PERSISTENCE = "persistence"
    CHECKPOINT = "checkpoint"
    CACHE = "cache"
    MEMORY = "memory"
    BATCH = "batch"
    RUNTIME = "runtime"


class ReaderErrorSeverity(Enum):
    """Operational severity for Reader errors."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass(frozen=True)
class ReaderErrorPolicy:
    """Runtime handling hints attached to each Reader exception.

    ``retryable`` means the same operation may succeed later without changing the
    task payload. ``recoverable`` means the Reader pipeline may continue with a
    fallback, partial result, or user correction.
    """

    retryable: bool = False
    recoverable: bool = False
    user_correctable: bool = True

    def to_dict(self) -> Dict[str, bool]:
        return {
            "retryable": self.retryable,
            "recoverable": self.recoverable,
            "user_correctable": self.user_correctable,
        }


class ReaderError(Exception):
    """Base exception for all Reader subsystem failures.

    The constructor remains compatible with the previous implementation:

    ``ReaderError(error_type, message, context=None, cause=None)``

    New optional metadata is keyword-only, so existing subclasses and call-sites
    keep working while production integrations can consume stable ``code``,
    ``stage``, ``severity``, and policy fields.
    """

    _SENSITIVE_KEY_PATTERN = re.compile(
        r"(password|passwd|pwd|secret|token|api[_-]?key|credential|authorization|auth|private[_-]?key|session|cookie)",
        re.IGNORECASE,
    )
    _MAX_CONTEXT_STRING_LENGTH = 600
    _MAX_CONTEXT_DEPTH = 4
    _REDACTED = "[REDACTED]"
    _TRUNCATED = "...[TRUNCATED]"

    def __init__(
        self,
        error_type: ReaderErrorType,
        message: str,
        context: Optional[Mapping[str, Any]] = None,
        cause: Optional[BaseException] = None,
        *,
        code: Optional[str] = None,
        stage: ReaderErrorStage | str | None = None,
        severity: ReaderErrorSeverity | str = ReaderErrorSeverity.ERROR,
        policy: Optional[ReaderErrorPolicy] = None,
        public_message: Optional[str] = None,
        resolution: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.error_type = self._coerce_error_type(error_type)
        self.context = dict(context or {})
        self.cause = cause
        self.code = code or self._default_code(self.error_type)
        self.stage = self._coerce_stage(stage)
        self.severity = self._coerce_severity(severity)
        self.policy = policy or ReaderErrorPolicy()
        self.public_message = public_message or message
        self.resolution = resolution
        self.created_at = datetime.now(timezone.utc)

    @staticmethod
    def _coerce_error_type(error_type: ReaderErrorType | str) -> ReaderErrorType:
        if isinstance(error_type, ReaderErrorType):
            return error_type
        for candidate in ReaderErrorType:
            if error_type in {candidate.name, candidate.value}:
                return candidate
        raise ValueError(f"Unknown ReaderErrorType: {error_type!r}")

    @staticmethod
    def _coerce_stage(stage: ReaderErrorStage | str | None) -> Optional[ReaderErrorStage]:
        if stage is None:
            return None
        if isinstance(stage, ReaderErrorStage):
            return stage
        for candidate in ReaderErrorStage:
            if stage in {candidate.name, candidate.value}:
                return candidate
        return ReaderErrorStage.RUNTIME

    @staticmethod
    def _coerce_severity(severity: ReaderErrorSeverity | str) -> ReaderErrorSeverity:
        if isinstance(severity, ReaderErrorSeverity):
            return severity
        for candidate in ReaderErrorSeverity:
            if severity in {candidate.name, candidate.value}:
                return candidate
        return ReaderErrorSeverity.ERROR

    @staticmethod
    def _default_code(error_type: ReaderErrorType) -> str:
        return f"READER_{error_type.name}"

    @classmethod
    def _sanitize_context(cls, value: Any, *, key: str = "", depth: int = 0) -> Any:
        """Return a JSON-safe, redacted representation for error payloads.

        This method is intentionally private and narrowly scoped to secure error
        serialization. It is not a general-purpose helper and should not be used
        outside this module.
        """

        if key and cls._SENSITIVE_KEY_PATTERN.search(str(key)):
            return cls._REDACTED

        if depth >= cls._MAX_CONTEXT_DEPTH:
            return cls._TRUNCATED

        if value is None or isinstance(value, (bool, int, float)):
            return value

        if isinstance(value, str):
            if len(value) > cls._MAX_CONTEXT_STRING_LENGTH:
                return value[: cls._MAX_CONTEXT_STRING_LENGTH] + cls._TRUNCATED
            return value

        if isinstance(value, Enum):
            return value.value

        if isinstance(value, Mapping):
            sanitized: Dict[str, Any] = {}
            for raw_key, raw_value in value.items():
                safe_key = str(raw_key)
                sanitized[safe_key] = cls._sanitize_context(raw_value, key=safe_key, depth=depth + 1)
            return sanitized

        if isinstance(value, (list, tuple, set, frozenset)):
            return [cls._sanitize_context(item, depth=depth + 1) for item in value]

        # Avoid exposing repr() of arbitrary objects that may contain secrets or
        # large document fragments. Type name is enough for safe telemetry.
        return f"<{type(value).__name__}>"

    @classmethod
    def _cause_payload(cls, cause: Optional[BaseException], *, include_debug: bool) -> Optional[Dict[str, Any] | str]:
        if cause is None:
            return None
        if not include_debug:
            # Backward-compatible safe default: previous implementation exposed
            # the cause type only.
            return type(cause).__name__
        return {
            "type": type(cause).__name__,
            "message": str(cause),
        }

    def with_context(self, **context: Any) -> "ReaderError":
        """Mutate this error with additional contextual fields and return self."""

        self.context.update(context)
        return self

    def to_dict(
        self,
        *,
        include_debug: bool = False,
        include_traceback: bool = False,
        sanitize: bool = True,
    ) -> Dict[str, Any]:
        """Serialize the error for API responses, checkpoints, or logs.

        The default payload is safe for user-facing responses. Debug fields are
        opt-in so raw exception messages and stack traces do not leak by default.
        """

        context = self._sanitize_context(self.context) if sanitize else dict(self.context)
        payload: Dict[str, Any] = {
            "error_type": self.error_type.value,
            "code": self.code,
            "message": str(self),
            "public_message": self.public_message,
            "stage": self.stage.value if self.stage else None,
            "severity": self.severity.value,
            "context": context,
            "cause": self._cause_payload(self.cause, include_debug=include_debug),
            "policy": self.policy.to_dict(),
            "resolution": self.resolution,
            "created_at": self.created_at.isoformat(),
        }

        if include_debug:
            payload["debug"] = {
                "exception_class": type(self).__name__,
                "cause_chain": self.cause_chain(),
            }

        if include_traceback and self.cause is not None:
            payload["traceback"] = "".join(
                traceback.format_exception(type(self.cause), self.cause, self.cause.__traceback__)
            )

        return payload

    def to_public_dict(self) -> Dict[str, Any]:
        """Serialize a minimal user-facing payload."""

        return {
            "error_type": self.error_type.value,
            "code": self.code,
            "message": self.public_message,
            "stage": self.stage.value if self.stage else None,
            "resolution": self.resolution,
            "policy": self.policy.to_dict(),
            "context": self._sanitize_context(self.context),
        }

    def to_log_dict(self, *, include_traceback: bool = False) -> Dict[str, Any]:
        """Serialize a richer payload for internal logs/telemetry."""

        return self.to_dict(include_debug=True, include_traceback=include_traceback, sanitize=True)

    def cause_chain(self) -> list[Dict[str, str]]:
        chain: list[Dict[str, str]] = []
        current: Optional[BaseException] = self.cause
        seen: set[int] = set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            chain.append({"type": type(current).__name__, "message": str(current)})
            current = current.__cause__ or current.__context__
        return chain

    @classmethod
    def from_exception(
        cls,
        exc: BaseException,
        *,
        message: str = "Reader task execution failed",
        error_type: ReaderErrorType = ReaderErrorType.UNEXPECTED_FAILURE,
        context: Optional[Mapping[str, Any]] = None,
        stage: ReaderErrorStage | str | None = ReaderErrorStage.RUNTIME,
        severity: ReaderErrorSeverity | str = ReaderErrorSeverity.ERROR,
        retryable: bool = False,
        recoverable: bool = False,
        resolution: Optional[str] = None,
    ) -> "ReaderError":
        if isinstance(exc, ReaderError):
            if context:
                exc.context.update(context)
            return exc
        return cls(
            error_type=error_type,
            message=message,
            context=context,
            cause=exc,
            stage=stage,
            severity=severity,
            policy=ReaderErrorPolicy(retryable=retryable, recoverable=recoverable),
            resolution=resolution,
        )


class ReaderValidationError(ReaderError):
    def __init__(
        self,
        message: str,
        context: Optional[Mapping[str, Any]] = None,
        *,
        error_type: ReaderErrorType = ReaderErrorType.INVALID_TASK,
        resolution: str = "Validate the Reader task payload before submitting it.",
    ) -> None:
        super().__init__(
            error_type,
            message,
            context=context,
            stage=ReaderErrorStage.VALIDATION,
            severity=ReaderErrorSeverity.WARNING,
            policy=ReaderErrorPolicy(retryable=False, recoverable=True, user_correctable=True),
            resolution=resolution,
        )


class ReaderInstructionError(ReaderValidationError):
    def __init__(self, instruction: Any, message: str = "Reader instruction is invalid") -> None:
        super().__init__(
            message,
            {"instruction": instruction},
            error_type=ReaderErrorType.INVALID_INSTRUCTION,
            resolution="Provide a non-empty Reader instruction such as 'convert to txt' or 'merge to md'.",
        )


class InvalidFilePathError(ReaderValidationError):
    def __init__(self, file_path: Any, message: str = "Reader file path is invalid") -> None:
        super().__init__(
            message,
            {"file_path": file_path},
            error_type=ReaderErrorType.INVALID_FILE_PATH,
            resolution="Provide a valid non-empty file path string.",
        )


class FileMissingError(ReaderError):
    def __init__(self, file_path: str) -> None:
        super().__init__(
            ReaderErrorType.FILE_NOT_FOUND,
            f"File does not exist: {file_path}",
            {"file_path": file_path},
            stage=ReaderErrorStage.PARSE,
            severity=ReaderErrorSeverity.WARNING,
            policy=ReaderErrorPolicy(retryable=False, recoverable=True, user_correctable=True),
            resolution="Confirm that the file path exists and that the Reader process can access it.",
        )


class FileNotRegularError(ReaderError):
    def __init__(self, file_path: str) -> None:
        super().__init__(
            ReaderErrorType.FILE_NOT_REGULAR,
            f"Input path is not a regular file: {file_path}",
            {"file_path": file_path},
            stage=ReaderErrorStage.PARSE,
            severity=ReaderErrorSeverity.WARNING,
            policy=ReaderErrorPolicy(retryable=False, recoverable=True, user_correctable=True),
            resolution="Use a regular file path instead of a directory, socket, device, or special path.",
        )


class FileAccessDeniedError(ReaderError):
    def __init__(self, file_path: str, operation: str = "read", cause: Optional[BaseException] = None) -> None:
        super().__init__(
            ReaderErrorType.FILE_ACCESS_DENIED,
            f"Reader does not have permission to {operation} file: {file_path}",
            {"file_path": file_path, "operation": operation},
            cause=cause,
            stage=ReaderErrorStage.PARSE,
            severity=ReaderErrorSeverity.ERROR,
            policy=ReaderErrorPolicy(retryable=True, recoverable=True, user_correctable=True),
            resolution="Check file permissions, locks, and whether another process is holding the file.",
        )


class FileTooLargeError(ReaderError):
    def __init__(self, file_path: str, size_bytes: int, max_size_bytes: int) -> None:
        super().__init__(
            ReaderErrorType.FILE_TOO_LARGE,
            f"File exceeds max allowed size of {max_size_bytes} bytes: {file_path}",
            {"file_path": file_path, "size_bytes": size_bytes, "max_size_bytes": max_size_bytes},
            stage=ReaderErrorStage.PARSE,
            severity=ReaderErrorSeverity.WARNING,
            policy=ReaderErrorPolicy(retryable=False, recoverable=True, user_correctable=True),
            resolution="Split the file, reduce its size, or increase the Reader max_file_size_bytes setting deliberately.",
        )


class UnsupportedFormatError(ReaderError):
    def __init__(self, file_path: str, extension: str, allowed_extensions: Iterable[str]) -> None:
        allowed = sorted({str(item).lower() for item in allowed_extensions})
        normalized_extension = str(extension or "").lower()
        super().__init__(
            ReaderErrorType.UNSUPPORTED_FORMAT,
            f"Unsupported input format '{normalized_extension}' for file: {file_path}",
            {
                "file_path": file_path,
                "extension": normalized_extension,
                "allowed_extensions": allowed,
            },
            stage=ReaderErrorStage.PARSE,
            severity=ReaderErrorSeverity.WARNING,
            policy=ReaderErrorPolicy(retryable=False, recoverable=True, user_correctable=True),
            resolution="Use a supported Reader input extension or add a parser for this file type.",
        )


class ParseFailureError(ReaderError):
    def __init__(
        self,
        file_path: str,
        message: str,
        cause: Optional[BaseException] = None,
        *,
        context: Optional[Mapping[str, Any]] = None,
    ) -> None:
        merged_context: Dict[str, Any] = {"file_path": file_path}
        if context:
            merged_context.update(context)
        super().__init__(
            ReaderErrorType.PARSE_FAILURE,
            message,
            merged_context,
            cause=cause,
            stage=ReaderErrorStage.PARSE,
            severity=ReaderErrorSeverity.ERROR,
            policy=ReaderErrorPolicy(retryable=isinstance(cause, OSError), recoverable=True, user_correctable=True),
            resolution="Inspect the source file, parser support, and file permissions.",
        )


class DecodeFailureError(ParseFailureError):
    def __init__(
        self,
        file_path: str,
        encodings_attempted: Optional[Iterable[str]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        super().__init__(
            file_path,
            f"Failed to decode file content: {file_path}",
            cause=cause,
            context={"encodings_attempted": list(encodings_attempted or [])},
        )
        self.error_type = ReaderErrorType.DECODE_FAILURE
        self.code = self._default_code(self.error_type)
        self.stage = ReaderErrorStage.DECODE
        self.resolution = "Add a supported encoding candidate or use a file with a readable text encoding."


class EmptyContentError(ParseFailureError):
    def __init__(self, file_path: str) -> None:
        super().__init__(
            file_path,
            f"Parsed document is empty or whitespace-only: {file_path}",
            context={"empty_content": True},
        )
        self.error_type = ReaderErrorType.EMPTY_CONTENT
        self.code = self._default_code(self.error_type)
        self.severity = ReaderErrorSeverity.WARNING
        self.resolution = "Confirm the document contains readable content before conversion or merge."


class CorruptContentError(ReaderError):
    def __init__(
        self,
        source: str,
        message: str = "Reader detected corrupted or unrecoverable content",
        *,
        confidence: Optional[float] = None,
        corruption_ratio: Optional[float] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        context: Dict[str, Any] = {"source": source}
        if confidence is not None:
            context["confidence"] = confidence
        if corruption_ratio is not None:
            context["corruption_ratio"] = corruption_ratio
        super().__init__(
            ReaderErrorType.CORRUPT_CONTENT,
            message,
            context,
            cause=cause,
            stage=ReaderErrorStage.RECOVERY,
            severity=ReaderErrorSeverity.ERROR,
            policy=ReaderErrorPolicy(retryable=False, recoverable=True, user_correctable=True),
            resolution="Run Reader recovery only on detectable corruption and provide a cleaner source if confidence is too low.",
        )


class MalformedReaderDocumentError(ReaderError):
    def __init__(self, message: str, context: Optional[Mapping[str, Any]] = None) -> None:
        super().__init__(
            ReaderErrorType.MALFORMED_DOCUMENT,
            message,
            context=context,
            stage=ReaderErrorStage.VALIDATION,
            severity=ReaderErrorSeverity.ERROR,
            policy=ReaderErrorPolicy(retryable=False, recoverable=True, user_correctable=False),
            resolution="Ensure parsed documents contain required Reader fields such as source, content, extension, and metadata.",
        )


class ConversionFailureError(ReaderError):
    def __init__(
        self,
        source: str,
        target_format: str,
        message: str,
        cause: Optional[BaseException] = None,
        *,
        context: Optional[Mapping[str, Any]] = None,
    ) -> None:
        merged_context: Dict[str, Any] = {"source": source, "target_format": str(target_format).lower().lstrip(".")}
        if context:
            merged_context.update(context)
        super().__init__(
            ReaderErrorType.CONVERSION_FAILURE,
            message,
            merged_context,
            cause=cause,
            stage=ReaderErrorStage.CONVERSION,
            severity=ReaderErrorSeverity.ERROR,
            policy=ReaderErrorPolicy(retryable=isinstance(cause, OSError), recoverable=True, user_correctable=True),
            resolution="Check the requested output format, output directory, disk permissions, and available converter backend.",
        )


class OutputPathError(ConversionFailureError):
    def __init__(self, output_path: str, message: str, cause: Optional[BaseException] = None) -> None:
        super().__init__(
            source="unknown",
            target_format="unknown",
            message=message,
            cause=cause,
            context={"output_path": output_path},
        )
        self.error_type = ReaderErrorType.OUTPUT_FAILURE
        self.code = self._default_code(self.error_type)
        self.resolution = "Use a writable output directory and avoid unsafe or reserved output paths."


class MergeFailureError(ReaderError):
    def __init__(
        self,
        message: str,
        context: Optional[Mapping[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        super().__init__(
            ReaderErrorType.MERGE_FAILURE,
            message,
            context=context,
            cause=cause,
            stage=ReaderErrorStage.MERGE,
            severity=ReaderErrorSeverity.ERROR,
            policy=ReaderErrorPolicy(retryable=isinstance(cause, OSError), recoverable=True, user_correctable=True),
            resolution="Verify that all parsed documents are valid and the merge output path is writable.",
        )


class RecoveryFailureError(ReaderError):
    def __init__(self, source: str, message: str, cause: Optional[BaseException] = None) -> None:
        super().__init__(
            ReaderErrorType.RECOVERY_FAILURE,
            message,
            {"source": source},
            cause=cause,
            stage=ReaderErrorStage.RECOVERY,
            severity=ReaderErrorSeverity.ERROR,
            policy=ReaderErrorPolicy(retryable=False, recoverable=True, user_correctable=True),
            resolution="Use conservative recovery, inspect corruption metrics, and provide a cleaner source if recovery confidence is low.",
        )


class PersistenceError(ReaderError):
    def __init__(
        self,
        message: str,
        context: Optional[Mapping[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        super().__init__(
            ReaderErrorType.PERSISTENCE_FAILURE,
            message,
            context=context,
            cause=cause,
            stage=ReaderErrorStage.PERSISTENCE,
            severity=ReaderErrorSeverity.ERROR,
            policy=ReaderErrorPolicy(retryable=isinstance(cause, OSError), recoverable=True, user_correctable=True),
            resolution="Check checkpoint/cache paths, disk availability, permissions, and JSON-serializable payloads.",
        )


class CheckpointPersistenceError(PersistenceError):
    def __init__(self, message: str, context: Optional[Mapping[str, Any]] = None, cause: Optional[BaseException] = None) -> None:
        super().__init__(message, context=context, cause=cause)
        self.error_type = ReaderErrorType.CHECKPOINT_FAILURE
        self.code = self._default_code(self.error_type)
        self.stage = ReaderErrorStage.CHECKPOINT
        self.resolution = "Check the Reader checkpoint directory, disk permissions, and checkpoint payload serializability."


class CachePersistenceError(PersistenceError):
    def __init__(self, message: str, context: Optional[Mapping[str, Any]] = None, cause: Optional[BaseException] = None) -> None:
        super().__init__(message, context=context, cause=cause)
        self.error_type = ReaderErrorType.CACHE_FAILURE
        self.code = self._default_code(self.error_type)
        self.stage = ReaderErrorStage.CACHE
        self.resolution = "Check the Reader cache directory, cache TTL handling, and cache payload serializability."


class ReaderConfigurationError(ReaderError):
    def __init__(self, message: str, context: Optional[Mapping[str, Any]] = None, cause: Optional[BaseException] = None) -> None:
        super().__init__(
            ReaderErrorType.CONFIGURATION_FAILURE,
            message,
            context=context,
            cause=cause,
            stage=ReaderErrorStage.CONFIG,
            severity=ReaderErrorSeverity.CRITICAL,
            policy=ReaderErrorPolicy(retryable=False, recoverable=False, user_correctable=True),
            resolution="Validate reader_config.yaml and the reader_agent section in agents_config.yaml.",
        )


class ReaderBatchError(ReaderError):
    def __init__(
        self,
        message: str,
        context: Optional[Mapping[str, Any]] = None,
        cause: Optional[BaseException] = None,
        *,
        failed_count: Optional[int] = None,
        total_count: Optional[int] = None,
    ) -> None:
        merged_context: Dict[str, Any] = dict(context or {})
        if failed_count is not None:
            merged_context["failed_count"] = failed_count
        if total_count is not None:
            merged_context["total_count"] = total_count
        super().__init__(
            ReaderErrorType.BATCH_FAILURE,
            message,
            context=merged_context,
            cause=cause,
            stage=ReaderErrorStage.BATCH,
            severity=ReaderErrorSeverity.ERROR,
            policy=ReaderErrorPolicy(retryable=False, recoverable=True, user_correctable=True),
            resolution="Inspect per-file Reader errors and retry only failed inputs after correction.",
        )


class ReaderTimeoutError(ReaderError):
    def __init__(self, operation: str, timeout_seconds: float, context: Optional[Mapping[str, Any]] = None) -> None:
        merged_context: Dict[str, Any] = {"operation": operation, "timeout_seconds": timeout_seconds}
        if context:
            merged_context.update(context)
        super().__init__(
            ReaderErrorType.TIMEOUT,
            f"Reader operation timed out: {operation}",
            context=merged_context,
            stage=ReaderErrorStage.RUNTIME,
            severity=ReaderErrorSeverity.ERROR,
            policy=ReaderErrorPolicy(retryable=True, recoverable=True, user_correctable=False),
            resolution="Retry with a smaller batch, lower concurrency, or a larger timeout budget.",
        )


class ReaderCancelledError(ReaderError):
    def __init__(self, operation: str = "reader_task", context: Optional[Mapping[str, Any]] = None) -> None:
        merged_context: Dict[str, Any] = {"operation": operation}
        if context:
            merged_context.update(context)
        super().__init__(
            ReaderErrorType.CANCELLED,
            f"Reader operation was cancelled: {operation}",
            context=merged_context,
            stage=ReaderErrorStage.RUNTIME,
            severity=ReaderErrorSeverity.INFO,
            policy=ReaderErrorPolicy(retryable=True, recoverable=True, user_correctable=False),
            resolution="Retry the operation if cancellation was not intentional.",
        )


class ReaderTaskExecutionError(ReaderError):
    def __init__(self, message: str = "Reader task execution failed", context: Optional[Mapping[str, Any]] = None, cause: Optional[BaseException] = None) -> None:
        super().__init__(
            ReaderErrorType.UNEXPECTED_FAILURE,
            message,
            context=context,
            cause=cause,
            stage=ReaderErrorStage.RUNTIME,
            severity=ReaderErrorSeverity.ERROR,
            policy=ReaderErrorPolicy(retryable=False, recoverable=False, user_correctable=False),
            resolution="Inspect the internal Reader error payload and root cause chain.",
        )


def normalize_reader_error(
    exc: BaseException,
    *,
    message: str = "Reader task execution failed",
    context: Optional[Mapping[str, Any]] = None,
    stage: ReaderErrorStage | str | None = ReaderErrorStage.RUNTIME,
) -> ReaderError:
    """Normalize arbitrary exceptions into a ReaderError.

    This is error-handling glue, not a general helper. It lets ReaderAgent and
    subsystem modules keep external exception handling consistent without
    importing helper utilities into the error layer.
    """

    if isinstance(exc, ReaderError):
        if context:
            exc.context.update(context)
        return exc
    normalized_stage = None
    if stage:
        coerced = ReaderError._coerce_stage(stage)
        if coerced is not None:
            normalized_stage = coerced.value
    return ReaderTaskExecutionError(message=message, context=context, cause=exc).with_context(
        normalized_stage=normalized_stage
    )


def reader_error_payload(
    exc: BaseException,
    *,
    include_debug: bool = False,
    include_traceback: bool = False,
) -> Dict[str, Any]:
    """Return a serialized Reader error payload for any exception."""

    error = normalize_reader_error(exc)
    return error.to_dict(include_debug=include_debug, include_traceback=include_traceback)


if __name__ == "__main__":
    print("\n=== Running Reader Error Smoke Test ===\n")

    missing = FileMissingError("missing.pdf")
    assert missing.error_type is ReaderErrorType.FILE_NOT_FOUND
    assert missing.to_dict()["cause"] is None
    assert missing.to_public_dict()["code"] == "READER_FILE_NOT_FOUND"

    unsupported = UnsupportedFormatError("image.png", ".png", {".txt", ".md"})
    assert unsupported.context["allowed_extensions"] == [".md", ".txt"]

    secret_error = ReaderTaskExecutionError(
        context={"api_token": "super-secret", "nested": {"password": "hidden", "safe": "visible"}},
        cause=RuntimeError("backend exploded"),
    )
    safe_payload = secret_error.to_dict()
    assert safe_payload["context"]["api_token"] == "[REDACTED]"
    assert safe_payload["context"]["nested"]["password"] == "[REDACTED]"
    assert safe_payload["cause"] == "RuntimeError"

    debug_payload = secret_error.to_log_dict()
    assert debug_payload["debug"]["exception_class"] == "ReaderTaskExecutionError"
    assert debug_payload["cause"]["type"] == "RuntimeError"

    normalized = normalize_reader_error(ValueError("bad payload"), context={"operation": "parse"})
    assert isinstance(normalized, ReaderTaskExecutionError)
    assert normalized.context["operation"] == "parse"

    print("\n=== Reader Error Smoke Test Passed ===\n")
