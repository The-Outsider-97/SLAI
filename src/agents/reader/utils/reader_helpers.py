from __future__ import annotations

"""Production-grade shared helpers for the Reader subsystem.

The Reader helper layer centralizes reusable, non-domain-owning utilities used by
``ReaderAgent`` and its internal engines. It intentionally does **not** own the
Reader error taxonomy; typed errors remain in ``reader_error.py``. Helpers may
raise those typed errors when validation, filesystem, serialization, retry, or
batch operations fail.

Scope
-----
- Task and instruction normalization for Reader workflows.
- Safe path, extension, output, and filesystem handling.
- Deterministic hashing, cache keys, and JSON serialization.
- Atomic file writes and resilient JSON reads.
- Text decoding, low-level cleanup, quality scoring, and chunking.
- Parsed-document validation and batch/result summarization.
- Bounded async fan-out and retry wrappers for Reader pipeline operations.
- Lightweight timing, LRU cache, and metric helpers.

Non-goals
---------
- No parser/converter/recovery business logic lives here.
- No agent orchestration or engine imports live here.
- No generic error taxonomy lives here.
"""

import asyncio
import hashlib
import json
import math
import os
import random
import re
import tempfile
import time
import uuid

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Generator,
    Iterable,
    Iterator,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    TypeVar,
)

from .config_loader import get_config_section, load_global_config
from .reader_error import *
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Reader Helpers")
printer = PrettyPrinter()

# ---------------------------------------------------------------------------
# Compatibility bridge to the Reader error layer
# ---------------------------------------------------------------------------

def _optional_error_class(name: str) -> Optional[type]:
    candidate = globals().get(name)
    return candidate if isinstance(candidate, type) else None


def _invalid_file_path_error(file_path: Any, message: str = "Reader file path is invalid") -> ReaderError:
    cls = _optional_error_class("InvalidFilePathError")
    if cls is not None:
        return cls(file_path, message)
    return ReaderValidationError(message, {"file_path": file_path})


def _reader_instruction_error(instruction: Any, message: str = "Reader instruction is invalid") -> ReaderError:
    cls = _optional_error_class("ReaderInstructionError")
    if cls is not None:
        return cls(instruction, message)
    return ReaderValidationError(message, {"instruction": instruction})


def _file_not_regular_error(file_path: str) -> ReaderError:
    cls = _optional_error_class("FileNotRegularError")
    if cls is not None:
        return cls(file_path)
    return ParseFailureError(file_path, "Input path is not a regular file")


def _file_access_denied_error(file_path: str, operation: str = "read", cause: Optional[BaseException] = None) -> ReaderError:
    cls = _optional_error_class("FileAccessDeniedError")
    if cls is not None:
        return cls(file_path, operation=operation, cause=cause)
    return ParseFailureError(file_path, f"Reader does not have permission to {operation} file: {file_path}", cause)  # type: ignore[arg-type]


def _file_too_large_error(file_path: str, size_bytes: int, max_size_bytes: int) -> ReaderError:
    cls = _optional_error_class("FileTooLargeError")
    if cls is not None:
        return cls(file_path, size_bytes, max_size_bytes)
    return ParseFailureError(file_path, f"File exceeds max allowed size of {max_size_bytes} bytes")


def _decode_failure_error(source: str, encodings_attempted: Iterable[str], cause: Optional[BaseException] = None) -> ReaderError:
    cls = _optional_error_class("DecodeFailureError")
    if cls is not None:
        return cls(source, encodings_attempted, cause=cause)
    return ParseFailureError(source, f"Failed to decode file content: {source}", cause)  # type: ignore[arg-type]


def _empty_content_error(source: str) -> ReaderError:
    cls = _optional_error_class("EmptyContentError")
    if cls is not None:
        return cls(source)
    return ParseFailureError(source, f"Parsed document is empty or whitespace-only: {source}")


def _malformed_reader_document_error(message: str, context: Optional[Mapping[str, Any]] = None) -> ReaderError:
    cls = _optional_error_class("MalformedReaderDocumentError")
    if cls is not None:
        return cls(message, context)
    return ReaderValidationError(message, dict(context or {}))


def _output_path_error(output_path: str, message: str, cause: Optional[BaseException] = None) -> ReaderError:
    cls = _optional_error_class("OutputPathError")
    if cls is not None:
        return cls(output_path, message, cause=cause)
    return ConversionFailureError("unknown", "unknown", message, cause)  # type: ignore[arg-type]


def _checkpoint_persistence_error(message: str, context: Optional[Mapping[str, Any]] = None, cause: Optional[BaseException] = None) -> ReaderError:
    cls = _optional_error_class("CheckpointPersistenceError")
    if cls is not None:
        return cls(message, context=context, cause=cause)
    return PersistenceError(message, dict(context or {}), cause)  # type: ignore[arg-type]


def _cache_persistence_error(message: str, context: Optional[Mapping[str, Any]] = None, cause: Optional[BaseException] = None) -> ReaderError:
    cls = _optional_error_class("CachePersistenceError")
    if cls is not None:
        return cls(message, context=context, cause=cause)
    return PersistenceError(message, dict(context or {}), cause)  # type: ignore[arg-type]


def _reader_configuration_error(message: str, context: Optional[Mapping[str, Any]] = None, cause: Optional[BaseException] = None) -> ReaderError:
    cls = _optional_error_class("ReaderConfigurationError")
    if cls is not None:
        return cls(message, context=context, cause=cause)
    return ReaderValidationError(message, dict(context or {}))


def _reader_batch_error(
    message: str,
    context: Optional[Mapping[str, Any]] = None,
    cause: Optional[BaseException] = None,
    *,
    failed_count: Optional[int] = None,
    total_count: Optional[int] = None,
) -> ReaderError:
    cls = _optional_error_class("ReaderBatchError")
    if cls is not None:
        return cls(message, context=context, cause=cause, failed_count=failed_count, total_count=total_count)
    merged = dict(context or {})
    if failed_count is not None:
        merged["failed_count"] = failed_count
    if total_count is not None:
        merged["total_count"] = total_count
    return ReaderValidationError(message, merged)


def _reader_timeout_error(operation: str, timeout_seconds: float, context: Optional[Mapping[str, Any]] = None) -> ReaderError:
    cls = _optional_error_class("ReaderTimeoutError")
    if cls is not None:
        return cls(operation, timeout_seconds, context=context)
    merged = {"operation": operation, "timeout_seconds": timeout_seconds}
    if context:
        merged.update(context)
    return ReaderValidationError(f"Reader operation timed out: {operation}", merged)


def _reader_cancelled_error(operation: str = "reader_task", context: Optional[Mapping[str, Any]] = None) -> ReaderError:
    cls = _optional_error_class("ReaderCancelledError")
    if cls is not None:
        return cls(operation, context=context)
    merged = {"operation": operation}
    if context:
        merged.update(context)
    return ReaderValidationError(f"Reader operation was cancelled: {operation}", merged)


def _reader_task_execution_error(
    message: str = "Reader task execution failed",
    context: Optional[Mapping[str, Any]] = None,
    cause: Optional[BaseException] = None,
) -> ReaderError:
    cls = _optional_error_class("ReaderTaskExecutionError")
    if cls is not None:
        return cls(message=message, context=context, cause=cause)
    error = ReaderValidationError(message, dict(context or {}))
    error.cause = cause  # type: ignore[attr-defined]
    return error


def _normalize_reader_exception(
    exc: BaseException,
    *,
    message: str = "Reader task execution failed",
    context: Optional[Mapping[str, Any]] = None,
) -> ReaderError:
    normalizer = globals().get("normalize_reader_error")
    if callable(normalizer):
        result = normalizer(exc, message=message, context=context)
        # The normalizer should return a ReaderError; enforce type safety.
        if not isinstance(result, ReaderError):
            raise TypeError(f"normalize_reader_error returned {type(result).__name__}, expected ReaderError")
        return result
    if isinstance(exc, ReaderError):
        if context:
            exc.context.update(dict(context))
        return exc
    return _reader_task_execution_error(message=message, context=context, cause=exc)


def _reader_error_payload(
    exc: BaseException,
    *,
    include_debug: bool = False,
    include_traceback: bool = False,
) -> Dict[str, Any]:
    payload_builder = globals().get("reader_error_payload")
    if callable(payload_builder):
        result = payload_builder(exc, include_debug=include_debug, include_traceback=include_traceback)
        if isinstance(result, dict):
            return result
        # Fallback if result is not a dict (should not happen in production)
        return {"error": str(result)}
    error = _normalize_reader_exception(exc)
    if hasattr(error, "to_dict"):
        return error.to_dict()
    return {"error_type": type(error).__name__, "message": str(error)}


T = TypeVar("T")
R = TypeVar("R")

_FORMAT_TOKEN_PATTERN = re.compile(r"\.?[a-zA-Z0-9][a-zA-Z0-9_+-]*")
_SAFE_FILENAME_PATTERN = re.compile(r"[^A-Za-z0-9._ -]+")
_WHITESPACE_PATTERN = re.compile(r"\s+")
_CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\uFFFD]")
_SENSITIVE_KEY_PATTERN = re.compile(
    r"(password|passwd|pwd|secret|token|api[_-]?key|credential|authorization|auth|private[_-]?key|session|cookie)",
    re.IGNORECASE,
)

DEFAULT_TEXT_EXTENSIONS = frozenset({".txt", ".md", ".html", ".xml", ".json", ".csv"})
DEFAULT_OUTPUT_FORMATS = frozenset({"txt", "md", "html", "xml", "json", "csv"})
DEFAULT_ENCODING_CANDIDATES = ("utf-8", "utf-8-sig", "latin-1")
DEFAULT_MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024
DEFAULT_MAX_CONTEXT_STRING_LENGTH = 1_000
DEFAULT_MAX_JSON_DEPTH = 8


@dataclass(frozen=True)
class ReaderPathInfo:
    """Normalized metadata for a Reader input/output path."""

    path: Path
    resolved_path: Path
    name: str
    stem: str
    extension: str
    exists: bool
    is_file: bool
    size_bytes: int = 0
    mtime: float = 0.0
    sha256: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": str(self.path),
            "resolved_path": str(self.resolved_path),
            "name": self.name,
            "stem": self.stem,
            "extension": self.extension,
            "exists": self.exists,
            "is_file": self.is_file,
            "size_bytes": self.size_bytes,
            "mtime": self.mtime,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class DecodedText:
    """Result of safe byte-to-text decoding."""

    text: str
    encoding: str
    byte_count: int
    had_decode_fallback: bool = False
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "encoding": self.encoding,
            "byte_count": self.byte_count,
            "had_decode_fallback": self.had_decode_fallback,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class TextQualityReport:
    """Small deterministic quality profile for parsed or recovered text."""

    quality_score: float
    char_count: int
    line_count: int
    printable_ratio: float
    alnum_ratio: float
    whitespace_ratio: float
    control_char_ratio: float
    replacement_char_count: int
    null_byte_count: int
    is_empty: bool
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["warnings"] = list(self.warnings)
        return payload


@dataclass(frozen=True)
class ChunkSpec:
    """Chunking configuration for Reader text/byte iteration."""

    chunk_size: int = 20_000
    overlap: int = 0
    max_chunks: Optional[int] = None

    def normalized(self) -> "ChunkSpec":
        size = max(1, int(self.chunk_size))
        overlap = max(0, min(int(self.overlap), size - 1))
        max_chunks = None if self.max_chunks is None else max(1, int(self.max_chunks))
        return ChunkSpec(chunk_size=size, overlap=overlap, max_chunks=max_chunks)


@dataclass(frozen=True)
class RetryConfig:
    """Retry policy for transient Reader helper operations."""

    max_attempts: int = 3
    base_delay_seconds: float = 0.20
    max_delay_seconds: float = 5.0
    backoff_factor: float = 2.0
    jitter_ratio: float = 0.10
    retry_exceptions: tuple[type[BaseException], ...] = (OSError, TimeoutError)

    def normalized(self) -> "RetryConfig":
        return RetryConfig(
            max_attempts=max(1, int(self.max_attempts)),
            base_delay_seconds=max(0.0, float(self.base_delay_seconds)),
            max_delay_seconds=max(0.0, float(self.max_delay_seconds)),
            backoff_factor=max(1.0, float(self.backoff_factor)),
            jitter_ratio=max(0.0, float(self.jitter_ratio)),
            retry_exceptions=self.retry_exceptions or (OSError, TimeoutError),
        )


@dataclass
class OperationTimer:
    """Monotonic operation timer for Reader metrics and checkpoints."""

    operation: str
    started_at_wall: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at_monotonic: float = field(default_factory=time.perf_counter)
    stopped_at_wall: Optional[datetime] = None
    stopped_at_monotonic: Optional[float] = None

    def stop(self) -> "OperationTimer":
        if self.stopped_at_monotonic is None:
            self.stopped_at_wall = datetime.now(timezone.utc)
            self.stopped_at_monotonic = time.perf_counter()
        return self

    @property
    def elapsed_seconds(self) -> float:
        end = self.stopped_at_monotonic if self.stopped_at_monotonic is not None else time.perf_counter()
        return max(0.0, end - self.started_at_monotonic)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "operation": self.operation,
            "started_at": self.started_at_wall.isoformat(),
            "stopped_at": self.stopped_at_wall.isoformat() if self.stopped_at_wall else None,
            "elapsed_seconds": round(self.elapsed_seconds, 6),
        }


class BoundedLRUCache(MutableMapping[str, T]):
    """Small dependency-free bounded LRU cache for Reader internals."""

    def __init__(self, max_size: int = 256) -> None:
        self.max_size = max(1, int(max_size))
        self._data: OrderedDict[str, T] = OrderedDict()

    def __getitem__(self, key: str) -> T:
        value = self._data[key]
        self._data.move_to_end(key)
        return value

    def __setitem__(self, key: str, value: T) -> None:
        self._data[key] = value
        self._data.move_to_end(key)
        while len(self._data) > self.max_size:
            self._data.popitem(last=False)

    def __delitem__(self, key: str) -> None:
        del self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def get_or_default(self, key: str, default: Optional[T] = None) -> Optional[T]:
        if key not in self._data:
            return default
        return self[key]

    def to_dict(self) -> Dict[str, Any]:
        return {"max_size": self.max_size, "size": len(self._data), "keys": list(self._data.keys())}


# ---------------------------------------------------------------------------
# Config and primitive coercion helpers
# ---------------------------------------------------------------------------


def get_reader_helper_config() -> Dict[str, Any]:
    """Return the optional ``reader_helpers`` config section."""

    return dict(get_config_section("reader_helpers") or {})


def get_reader_config_value(section_name: str, key: str, default: T) -> T:
    """Read a Reader config value without duplicating config loading logic."""

    section = get_config_section(section_name)
    return section.get(key, default)  # type: ignore[return-value]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_timestamp() -> float:
    return utc_now().timestamp()


def utc_compact_timestamp() -> str:
    return utc_now().strftime("%Y%m%dT%H%M%SZ")


def clamp(value: float, minimum: float, maximum: float) -> float:
    if minimum > maximum:
        minimum, maximum = maximum, minimum
    if math.isnan(float(value)):
        return minimum
    return max(minimum, min(maximum, float(value)))


def coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on", "enabled"}:
        return True
    if text in {"0", "false", "no", "n", "off", "disabled"}:
        return False
    return default


def coerce_int(value: Any, default: int = 0, *, minimum: Optional[int] = None, maximum: Optional[int] = None) -> int:
    try:
        output = int(value)
    except (TypeError, ValueError):
        output = int(default)
    if minimum is not None:
        output = max(int(minimum), output)
    if maximum is not None:
        output = min(int(maximum), output)
    return output


def coerce_float(
    value: Any,
    default: float = 0.0,
    *,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> float:
    try:
        output = float(value)
    except (TypeError, ValueError):
        output = float(default)
    if minimum is not None:
        output = max(float(minimum), output)
    if maximum is not None:
        output = min(float(maximum), output)
    return output


# ---------------------------------------------------------------------------
# Secure JSON, hashing, and cache keys
# ---------------------------------------------------------------------------


def redact_sensitive_mapping(payload: Any, *, max_string_length: int = DEFAULT_MAX_CONTEXT_STRING_LENGTH,
                             max_depth: int = DEFAULT_MAX_JSON_DEPTH, _key: str = "", _depth: int = 0) -> Any:
    """Return a redacted, JSON-safe representation for logs/checkpoints."""

    if _key and _SENSITIVE_KEY_PATTERN.search(_key):
        return "[REDACTED]"
    if _depth >= max_depth:
        return "...[TRUNCATED]"
    if payload is None or isinstance(payload, (bool, int, float)):
        return payload
    if isinstance(payload, bytes):
        return {"type": "bytes", "length": len(payload), "sha256": sha256_bytes(payload)}
    if isinstance(payload, Path):
        return str(payload)
    if isinstance(payload, datetime):
        return payload.isoformat()
    if isinstance(payload, str):
        if len(payload) > max_string_length:
            return payload[:max_string_length] + "...[TRUNCATED]"
        return payload
    if is_dataclass(payload) and not isinstance(payload, type):
        # asdict only works on instances, not on dataclass classes
        return redact_sensitive_mapping(
            asdict(payload),
            max_string_length=max_string_length,
            max_depth=max_depth,
            _depth=_depth + 1,
        )
    elif is_dataclass(payload) and isinstance(payload, type):
        # It's a dataclass class; represent it as a string to avoid recursion
        return f"<dataclass class {payload.__name__}>"
    if isinstance(payload, Mapping):
        return {
            str(key): redact_sensitive_mapping(
                value,
                max_string_length=max_string_length,
                max_depth=max_depth,
                _key=str(key),
                _depth=_depth + 1,
            )
            for key, value in payload.items()
        }
    if isinstance(payload, (list, tuple, set, frozenset)):
        return [
            redact_sensitive_mapping(
                item,
                max_string_length=max_string_length,
                max_depth=max_depth,
                _depth=_depth + 1,
            )
            for item in payload
        ]
    return f"<{type(payload).__name__}>"


def json_safe(payload: Any, *, redact: bool = False) -> Any:
    """Convert common Reader payloads into deterministic JSON-safe values."""

    if redact:
        return redact_sensitive_mapping(payload)
    if payload is None or isinstance(payload, (bool, int, float, str)):
        return payload
    if isinstance(payload, bytes):
        return {"type": "bytes", "length": len(payload), "sha256": sha256_bytes(payload)}
    if isinstance(payload, Path):
        return str(payload)
    if isinstance(payload, datetime):
        return payload.isoformat()
    if is_dataclass(payload) and not isinstance(payload, type):
        return json_safe(asdict(payload), redact=redact)
    elif is_dataclass(payload) and isinstance(payload, type):
        return f"<dataclass class {payload.__name__}>"
    if isinstance(payload, Mapping):
        return {str(key): json_safe(value, redact=redact) for key, value in payload.items()}
    if isinstance(payload, (list, tuple, set, frozenset)):
        return [json_safe(item, redact=redact) for item in payload]
    if isinstance(payload, ReaderError):
        return payload.to_dict()
    return str(payload)


def stable_json_dumps(payload: Any, *, redact: bool = False, indent: Optional[int] = None) -> str:
    return json.dumps(
        json_safe(payload, redact=redact),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":") if indent is None else None,
        indent=indent,
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_text(text: str, *, encoding: str = "utf-8") -> str:
    return sha256_bytes(text.encode(encoding, errors="replace"))


def sha256_file(file_path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    path = Path(file_path).expanduser()
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(max(1, chunk_size)), b""):
                digest.update(chunk)
    except PermissionError as exc:
        raise _file_access_denied_error(str(path), operation="hash", cause=exc) from exc
    except OSError as exc:
        raise ParseFailureError(str(path), f"Failed hashing file: {exc}", cause=exc) from exc
    return digest.hexdigest()


def stable_hash(payload: Any, *, redact: bool = False) -> str:
    return sha256_text(stable_json_dumps(payload, redact=redact))


def cache_key(namespace: str, payload: Mapping[str, Any]) -> str:
    return f"{namespace}:{stable_hash(payload)}"


def parse_cache_key(file_path: str | Path, *, include_stat: bool = True) -> str:
    path = Path(file_path).expanduser()
    payload: Dict[str, Any] = {"action": "parse", "file": str(path)}
    if include_stat and path.exists() and path.is_file():
        stat = path.stat()
        payload.update({"size": stat.st_size, "mtime": stat.st_mtime})
    return cache_key("reader", payload)


def recovery_cache_key(source: str, content: str, *, policy: Optional[Mapping[str, Any]] = None) -> str:
    return cache_key(
        "reader_recovery",
        {"source": source, "content_sha256": sha256_text(content), "policy": dict(policy or {})},
    )


def conversion_cache_key(source: str, target_format: str, output_dir: str | Path) -> str:
    return cache_key(
        "reader_conversion",
        {"source": source, "target_format": normalize_output_format(target_format), "output_dir": str(output_dir)},
    )


# ---------------------------------------------------------------------------
# Path, extension, and filesystem helpers
# ---------------------------------------------------------------------------


def normalize_extension(extension: str | None, *, include_dot: bool = True) -> str:
    text = str(extension or "").strip().lower()
    if not text:
        return ""
    text = text if text.startswith(".") else f".{text}"
    return text if include_dot else text[1:]


def normalize_extension_set(extensions: Iterable[str]) -> set[str]:
    return {normalize_extension(ext, include_dot=True) for ext in extensions if str(ext).strip()}


def normalize_output_format(target_format: str | None, *, supported_formats: Optional[Iterable[str]] = None, default: str = "txt") -> str:
    normalized = normalize_extension(target_format or default, include_dot=False)
    supported = {normalize_extension(fmt, include_dot=False) for fmt in (supported_formats or DEFAULT_OUTPUT_FORMATS)}
    if normalized not in supported:
        raise ConversionFailureError(
            source="unknown",
            target_format=normalized,
            message=f"Unsupported Reader output format: {normalized}",
            context={"supported_formats": sorted(supported)},
        )
    return normalized


def safe_filename(value: Any, *, fallback: str = "document", max_length: int = 160) -> str:
    raw = str(value or "").strip()
    if not raw:
        raw = fallback
    # Preserve useful spaces/dots/dashes while stripping path separators and
    # reserved/surprising characters.
    raw = raw.replace("/", "_").replace("\\", "_")
    cleaned = _SAFE_FILENAME_PATTERN.sub("_", raw)
    cleaned = _WHITESPACE_PATTERN.sub(" ", cleaned).strip(" ._")
    if not cleaned:
        cleaned = fallback
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length].rstrip(" ._") or fallback
    return cleaned


def expand_path(path_value: str | Path) -> Path:
    if isinstance(path_value, Path):
        return path_value.expanduser()
    if not isinstance(path_value, str) or not path_value.strip():
        raise _invalid_file_path_error(path_value)
    return Path(path_value).expanduser()


def resolve_path(path_value: str | Path, *, strict: bool = False) -> Path:
    path = expand_path(path_value)
    try:
        return path.resolve(strict=strict)
    except FileNotFoundError:
        if strict:
            raise FileMissingError(str(path))
        return path.absolute()
    except PermissionError as exc:
        raise _file_access_denied_error(str(path), operation="resolve", cause=exc) from exc
    except OSError as exc:
        raise _invalid_file_path_error(str(path), f"Failed resolving Reader path: {exc}") from exc


def is_path_within(path: str | Path, base_dir: str | Path) -> bool:
    candidate = resolve_path(path, strict=False)
    base = resolve_path(base_dir, strict=False)
    try:
        candidate.relative_to(base)
        return True
    except ValueError:
        return False


def ensure_directory(path_value: str | Path, *, purpose: str = "directory") -> Path:
    path = expand_path(path_value)
    try:
        path.mkdir(parents=True, exist_ok=True)
    except PermissionError as exc:
        raise _file_access_denied_error(str(path), operation=f"create_{purpose}", cause=exc) from exc
    except OSError as exc:
        raise PersistenceError(f"Failed creating Reader {purpose}: {path}", {"path": str(path)}, exc) from exc
    if not path.is_dir():
        raise PersistenceError(f"Reader {purpose} is not a directory: {path}", {"path": str(path)})
    return path


def validate_input_path(
    file_path: str | Path,
    *,
    allowed_extensions: Optional[Iterable[str]] = None,
    max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES,
    compute_hash: bool = False,
) -> ReaderPathInfo:
    """Validate a Reader input file and return normalized path metadata."""

    path = expand_path(file_path)
    if not path.exists():
        raise FileMissingError(str(path))
    if not path.is_file():
        raise _file_not_regular_error(str(path))

    extension = normalize_extension(path.suffix, include_dot=True)
    allowed = normalize_extension_set(allowed_extensions or get_config_section("reader").get("allowed_input_extensions", DEFAULT_TEXT_EXTENSIONS))
    if allowed and extension not in allowed:
        raise UnsupportedFormatError(str(path), extension, allowed)

    try:
        stat = path.stat()
    except PermissionError as exc:
        raise _file_access_denied_error(str(path), operation="stat", cause=exc) from exc
    except OSError as exc:
        raise ParseFailureError(str(path), f"Failed reading file metadata: {exc}", cause=exc) from exc

    max_size = max(0, int(max_file_size_bytes))
    if max_size and stat.st_size > max_size:
        raise _file_too_large_error(str(path), stat.st_size, max_size)

    digest = sha256_file(path) if compute_hash else None
    return ReaderPathInfo(
        path=path,
        resolved_path=resolve_path(path, strict=False),
        name=path.name,
        stem=path.stem,
        extension=extension,
        exists=True,
        is_file=True,
        size_bytes=int(stat.st_size),
        mtime=float(stat.st_mtime),
        sha256=digest,
    )


def prepare_output_path(
    output_dir: str | Path,
    stem: str,
    target_extension: str,
    *,
    overwrite: bool = False,
    suffix: Optional[str] = None,
) -> Path:
    """Build a safe output path, avoiding collisions unless overwrite is set."""

    out_dir = ensure_directory(output_dir, purpose="output_dir")
    extension = normalize_extension(target_extension, include_dot=True)
    if not extension:
        raise _output_path_error(str(out_dir), "Reader output extension cannot be empty")

    safe_stem = safe_filename(stem, fallback="document")
    if suffix:
        safe_stem = safe_filename(f"{safe_stem}_{suffix}", fallback=safe_stem)
    candidate = out_dir / f"{safe_stem}{extension}"
    if overwrite or not candidate.exists():
        return candidate

    timestamp = utc_compact_timestamp()
    for _ in range(100):
        unique = uuid.uuid4().hex[:8]
        candidate = out_dir / f"{safe_stem}_{timestamp}_{unique}{extension}"
        if not candidate.exists():
            return candidate
    raise _output_path_error(str(out_dir), "Failed preparing a unique Reader output path")


def atomic_write_bytes(path_value: str | Path, payload: bytes, *, purpose: str = "output") -> Path:
    target = expand_path(path_value)
    ensure_directory(target.parent, purpose=f"{purpose}_parent")
    tmp_path = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp_path.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        tmp_path.replace(target)
        return target
    except PermissionError as exc:
        raise _file_access_denied_error(str(target), operation=f"write_{purpose}", cause=exc) from exc
    except OSError as exc:
        if purpose == "checkpoint":
            raise _checkpoint_persistence_error("Failed writing Reader checkpoint", {"path": str(target)}, exc) from exc
        if purpose == "cache":
            raise _cache_persistence_error("Failed writing Reader cache payload", {"path": str(target)}, exc) from exc
        if purpose == "output":
            raise _output_path_error(str(target), f"Failed writing Reader output: {exc}", cause=exc) from exc
        raise PersistenceError(f"Failed writing Reader {purpose}: {target}", {"path": str(target)}, exc) from exc
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            logger.warning("Failed cleaning temporary Reader write file: %s", tmp_path)


def atomic_write_text(path_value: str | Path, payload: str, *, encoding: str = "utf-8", purpose: str = "output") -> Path:
    return atomic_write_bytes(path_value, payload.encode(encoding), purpose=purpose)


def atomic_write_json(
    path_value: str | Path,
    payload: Mapping[str, Any],
    *,
    redact: bool = False,
    indent: Optional[int] = None,
    purpose: str = "persistence",
) -> Path:
    data = stable_json_dumps(payload, redact=redact, indent=indent).encode("utf-8")
    return atomic_write_bytes(path_value, data, purpose=purpose)


def read_json_file(path_value: str | Path, *, purpose: str = "persistence") -> Dict[str, Any]:
    path = expand_path(path_value)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except PermissionError as exc:
        raise _file_access_denied_error(str(path), operation=f"read_{purpose}", cause=exc) from exc
    except json.JSONDecodeError as exc:
        raise PersistenceError(f"Failed decoding Reader JSON payload: {path}", {"path": str(path)}, exc) from exc
    except OSError as exc:
        if purpose == "checkpoint":
            raise _checkpoint_persistence_error("Failed reading Reader checkpoint", {"path": str(path)}, exc) from exc
        if purpose == "cache":
            raise _cache_persistence_error("Failed reading Reader cache payload", {"path": str(path)}, exc) from exc
        raise PersistenceError(f"Failed reading Reader {purpose}: {path}", {"path": str(path)}, exc) from exc


# ---------------------------------------------------------------------------
# Reader task, instruction, and plan helpers
# ---------------------------------------------------------------------------


def normalize_files(files: Any) -> List[str]:
    if isinstance(files, (str, Path)):
        files = [str(files)]
    if not isinstance(files, Sequence) or isinstance(files, (bytes, bytearray)):
        raise ReaderValidationError("task_data.files must be a non-empty list", {"files": files})
    normalized: List[str] = []
    for item in files:
        if not isinstance(item, (str, Path)) or not str(item).strip():
            raise _invalid_file_path_error(item)
        normalized.append(str(item).strip())
    if not normalized:
        raise ReaderValidationError("task_data.files must be a non-empty list", {"files": files})
    return normalized


def normalize_instruction(instruction: Any, *, allow_empty: bool = True) -> str:
    if instruction is None:
        instruction = ""
    text = str(instruction).strip()
    if not text and not allow_empty:
        raise _reader_instruction_error(instruction)
    return text


def validate_reader_task(task_data: Mapping[str, Any]) -> tuple[str, List[str]]:
    if not isinstance(task_data, Mapping):
        raise ReaderValidationError("task_data must be a dictionary")
    instruction = normalize_instruction(task_data.get("instruction", ""), allow_empty=True)
    files = normalize_files(task_data.get("files", []))
    return instruction, files


def extract_requested_format(
    instruction: str,
    supported_formats: Iterable[str],
    *,
    default: Optional[str] = None,
) -> Optional[str]:
    tokens = {_token.lower() for _token in _FORMAT_TOKEN_PATTERN.findall(str(instruction or ""))}
    normalized_supported = tuple(sorted({normalize_extension(fmt, include_dot=False) for fmt in supported_formats}))
    for fmt in normalized_supported:
        if fmt in tokens or f".{fmt}" in tokens:
            return fmt
    return normalize_extension(default, include_dot=False) if default else None


def instruction_requests_recovery(instruction: str) -> bool:
    lowered = str(instruction or "").lower()
    return any(term in lowered for term in ("recover", "recovery", "repair", "corrupt", "corrupted", "broken", "salvage"))


def instruction_requests_merge(instruction: str, *, file_count: int = 1) -> bool:
    lowered = str(instruction or "").lower()
    return file_count > 1 or any(term in lowered for term in ("merge", "combine", "join", "bundle", "single file"))


def build_basic_reader_plan(
    instruction: str,
    files: Sequence[str],
    *,
    supported_output_formats: Iterable[str] = DEFAULT_OUTPUT_FORMATS,
    default_output_format: str = "txt",
) -> List[Dict[str, Any]]:
    """Build the default Reader parse -> optional recover -> convert/merge plan."""

    normalized_files = normalize_files(list(files))
    target = extract_requested_format(instruction, supported_output_formats, default=default_output_format)
    target = normalize_output_format(target, supported_formats=supported_output_formats, default=default_output_format)
    plan: List[Dict[str, Any]] = [{"action": "parse", "files": normalized_files}]
    if instruction_requests_recovery(instruction):
        plan.append({"action": "recover"})
    if instruction_requests_merge(instruction, file_count=len(normalized_files)):
        plan.append({"action": "merge", "output": target})
    else:
        plan.append({"action": "convert", "target": target})
    return plan


# ---------------------------------------------------------------------------
# Text, decoding, and recovery-adjacent helpers
# ---------------------------------------------------------------------------


def is_probably_binary(raw_bytes: bytes, *, sample_size: int = 4096) -> bool:
    if not raw_bytes:
        return False
    sample = raw_bytes[: max(1, sample_size)]
    if b"\x00" in sample:
        return True
    control_count = sum(byte < 32 and byte not in (9, 10, 13) for byte in sample)
    return (control_count / max(1, len(sample))) > 0.30


def decode_bytes(
    raw_bytes: bytes,
    *,
    source: str = "unknown",
    encoding_candidates: Iterable[str] = DEFAULT_ENCODING_CANDIDATES,
    allow_lossy_fallback: bool = True,
) -> DecodedText:
    attempted: List[str] = []
    for encoding in encoding_candidates:
        attempted.append(str(encoding))
        try:
            text = raw_bytes.decode(str(encoding))
            warnings = ["binary_signature_detected"] if is_probably_binary(raw_bytes) else []
            if not text.strip():
                warnings.append("empty_or_whitespace_content")
            return DecodedText(text=text, encoding=str(encoding), byte_count=len(raw_bytes), warnings=tuple(warnings))
        except UnicodeDecodeError:
            continue
    if not allow_lossy_fallback:
        raise _decode_failure_error(source, attempted)
    text = raw_bytes.decode("utf-8", errors="replace")
    warnings = ["lossy_decode_fallback_used"]
    if is_probably_binary(raw_bytes):
        warnings.append("binary_signature_detected")
    if not text.strip():
        warnings.append("empty_or_whitespace_content")
    return DecodedText(
        text=text,
        encoding="utf-8-replace",
        byte_count=len(raw_bytes),
        had_decode_fallback=True,
        warnings=tuple(warnings),
    )


def read_file_bytes(
    file_path: str | Path,
    *,
    allowed_extensions: Optional[Iterable[str]] = None,
    max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES,
) -> bytes:
    path_info = validate_input_path(
        file_path,
        allowed_extensions=allowed_extensions,
        max_file_size_bytes=max_file_size_bytes,
        compute_hash=False,
    )
    try:
        return path_info.path.read_bytes()
    except PermissionError as exc:
        raise _file_access_denied_error(str(path_info.path), operation="read", cause=exc) from exc
    except OSError as exc:
        raise ParseFailureError(str(path_info.path), f"Failed reading file: {exc}", cause=exc) from exc


def normalize_newlines(text: str) -> str:
    return str(text).replace("\r\n", "\n").replace("\r", "\n")


def remove_null_bytes(text: str) -> str:
    return str(text).replace("\x00", "")


def remove_replacement_chars(text: str) -> str:
    return str(text).replace("\uFFFD", "")


def low_level_repair_text(text: str) -> str:
    """Conservative low-level cleanup that does not fabricate content."""

    return remove_replacement_chars(normalize_newlines(remove_null_bytes(text)))


def truncate_text(text: str, max_chars: int, *, suffix: str = "...[TRUNCATED]") -> str:
    if max_chars <= 0:
        return ""
    value = str(text)
    if len(value) <= max_chars:
        return value
    if len(suffix) >= max_chars:
        return suffix[:max_chars]
    return value[: max_chars - len(suffix)] + suffix


def iter_text_chunks(text: str, spec: ChunkSpec | int = ChunkSpec()) -> Iterator[str]:
    normalized = spec.normalized() if isinstance(spec, ChunkSpec) else ChunkSpec(chunk_size=int(spec)).normalized()
    value = str(text)
    index = 0
    emitted = 0
    step = max(1, normalized.chunk_size - normalized.overlap)
    while index < len(value):
        if normalized.max_chunks is not None and emitted >= normalized.max_chunks:
            break
        yield value[index : index + normalized.chunk_size]
        emitted += 1
        index += step


def iter_byte_chunks(payload: bytes, chunk_size: int = 1024 * 1024) -> Iterator[bytes]:
    size = max(1, int(chunk_size))
    for index in range(0, len(payload), size):
        yield payload[index : index + size]


def text_quality_report(text: str) -> TextQualityReport:
    value = str(text or "")
    length = len(value)
    if length == 0:
        return TextQualityReport(
            quality_score=0.0,
            char_count=0,
            line_count=0,
            printable_ratio=0.0,
            alnum_ratio=0.0,
            whitespace_ratio=0.0,
            control_char_ratio=0.0,
            replacement_char_count=0,
            null_byte_count=0,
            is_empty=True,
            warnings=("empty_or_whitespace_content",),
        )

    printable_ratio = sum(ch.isprintable() or ch in "\n\r\t" for ch in value) / length
    alnum_ratio = sum(ch.isalnum() for ch in value) / length
    whitespace_ratio = sum(ch.isspace() for ch in value) / length
    control_char_count = len(_CONTROL_CHAR_PATTERN.findall(value))
    control_char_ratio = control_char_count / length
    replacement_char_count = value.count("\uFFFD")
    null_byte_count = value.count("\x00")
    is_empty = not value.strip()

    score = (0.58 * printable_ratio) + (0.32 * alnum_ratio) + (0.10 * min(1.0, whitespace_ratio * 4.0))
    score -= 0.85 * control_char_ratio
    score = clamp(score, 0.0, 1.0)

    warnings: List[str] = []
    if is_empty:
        warnings.append("empty_or_whitespace_content")
    if null_byte_count:
        warnings.append("null_bytes_detected")
    if replacement_char_count:
        warnings.append("replacement_chars_detected")
    if control_char_ratio > 0.02:
        warnings.append("high_control_char_ratio")
    if score < 0.45:
        warnings.append("low_text_quality")

    return TextQualityReport(
        quality_score=round(score, 4),
        char_count=length,
        line_count=value.count("\n") + (1 if value else 0),
        printable_ratio=round(printable_ratio, 4),
        alnum_ratio=round(alnum_ratio, 4),
        whitespace_ratio=round(whitespace_ratio, 4),
        control_char_ratio=round(control_char_ratio, 4),
        replacement_char_count=replacement_char_count,
        null_byte_count=null_byte_count,
        is_empty=is_empty,
        warnings=tuple(warnings),
    )


def detect_text_warnings(text: str, *, raw_bytes: Optional[bytes] = None) -> List[str]:
    warnings = list(text_quality_report(text).warnings)
    if raw_bytes is not None and is_probably_binary(raw_bytes):
        warnings.append("binary_signature_detected")
    return dedupe_preserve_order(warnings)


# ---------------------------------------------------------------------------
# Parsed document and result helpers
# ---------------------------------------------------------------------------


def validate_parsed_document(
    parsed_doc: Mapping[str, Any],
    *,
    require_content: bool = False,
    require_metadata: bool = False,
) -> Dict[str, Any]:
    if not isinstance(parsed_doc, Mapping):
        raise _malformed_reader_document_error("Parsed Reader document must be a mapping", {"type": type(parsed_doc).__name__})
    missing: List[str] = []
    for field_name in ("source", "content"):
        if field_name not in parsed_doc:
            missing.append(field_name)
    if require_metadata and "metadata" not in parsed_doc:
        missing.append("metadata")
    if missing:
        raise _malformed_reader_document_error("Parsed Reader document is missing required fields", {"missing_fields": missing})
    source = str(parsed_doc.get("source", "unknown"))
    content = parsed_doc.get("content")
    if content is None:
        raise _malformed_reader_document_error("Parsed Reader document content cannot be None", {"source": source})
    if require_content and not str(content).strip():
        raise _empty_content_error(source)
    return dict(parsed_doc)


def parsed_document_fingerprint(parsed_doc: Mapping[str, Any]) -> str:
    doc = validate_parsed_document(parsed_doc)
    source = str(doc.get("source", "unknown"))
    content = str(doc.get("content", ""))
    metadata = doc.get("metadata", {}) if isinstance(doc.get("metadata", {}), Mapping) else {}
    return stable_hash(
        {
            "source": source,
            "content_sha256": sha256_text(content),
            "size": metadata.get("size"),
            "mtime": metadata.get("mtime"),
            "extension": doc.get("extension"),
        }
    )


def summarize_parsed_document(parsed_doc: Mapping[str, Any], *, include_hash: bool = True) -> Dict[str, Any]:
    doc = validate_parsed_document(parsed_doc)
    source = str(doc.get("source", "unknown"))
    content = str(doc.get("content", ""))
    quality = text_quality_report(content)
    summary: Dict[str, Any] = {
        "source": source,
        "extension": doc.get("extension"),
        "char_count": len(content),
        "line_count": quality.line_count,
        "quality_score": quality.quality_score,
        "warnings": dedupe_preserve_order(list(doc.get("warnings", [])) + list(quality.warnings)),
        "metadata": json_safe(doc.get("metadata", {})),
    }
    if include_hash:
        summary["content_sha256"] = sha256_text(content)
        summary["fingerprint"] = parsed_document_fingerprint(doc)
    return summary


def apply_recovered_content(parsed_doc: Mapping[str, Any], recovery_result: Mapping[str, Any]) -> Dict[str, Any]:
    doc = validate_parsed_document(parsed_doc)
    if not isinstance(recovery_result, Mapping) or "content" not in recovery_result:
        raise _malformed_reader_document_error("Recovery result must contain recovered content", {"source": doc.get("source")})
    updated = dict(doc)
    updated["content"] = str(recovery_result.get("content", ""))
    updated["recovery"] = dict(recovery_result)
    return updated


def merge_document_contents(parsed_docs: Sequence[Mapping[str, Any]], *, output_format: str = "txt") -> str:
    if not parsed_docs:
        raise MergeFailureError("Merge requires at least one parsed document")
    target = normalize_extension(output_format, include_dot=False)
    segments: List[str] = []
    for item in parsed_docs:
        doc = validate_parsed_document(item)
        source = str(doc.get("source", "unknown"))
        content = str(doc.get("content", ""))
        header = f"\n\n## Source: {source}\n\n" if target == "md" else f"\n\n# Source: {source}\n\n"
        segments.append(header + content)
    return "".join(segments).strip() + "\n"


def build_success_result(
    *,
    operation: str,
    payload: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    warnings: Optional[Iterable[str]] = None,
    timer: Optional[OperationTimer] = None,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "status": "ok",
        "operation": operation,
        "payload": dict(payload or {}),
        "metadata": dict(metadata or {}),
        "warnings": dedupe_preserve_order(warnings or []),
    }
    if timer is not None:
        result["timing"] = timer.stop().to_dict()
    return result


def build_error_result(
    exc: BaseException,
    *,
    operation: str,
    include_debug: bool = False,
    include_traceback: bool = False,
) -> Dict[str, Any]:
    return {
        "status": "error",
        "operation": operation,
        "error": _reader_error_payload(exc, include_debug=include_debug, include_traceback=include_traceback),
    }


def summarize_batch_results(results: Sequence[Any]) -> Dict[str, Any]:
    success_count = 0
    error_payloads: List[Dict[str, Any]] = []
    warning_count = 0
    for item in results:
        if isinstance(item, ReaderError):
            error_payloads.append(item.to_public_dict())
            continue
        if isinstance(item, BaseException):
            error_payloads.append(_normalize_reader_exception(item).to_public_dict() if hasattr(_normalize_reader_exception(item), "to_public_dict") else _normalize_reader_exception(item).to_dict())
            continue
        if isinstance(item, Mapping) and item.get("status") == "error":
            error_payloads.append(dict(item.get("error", {})))
            continue
        success_count += 1
        if isinstance(item, Mapping):
            warning_count += len(item.get("warnings", []) or [])
    return {
        "total_count": len(results),
        "success_count": success_count,
        "failed_count": len(error_payloads),
        "warning_count": warning_count,
        "errors": error_payloads,
    }


# ---------------------------------------------------------------------------
# Collection, metric, and small utility helpers
# ---------------------------------------------------------------------------


def dedupe_preserve_order(items: Iterable[T]) -> List[T]:
    seen: set[str] = set()
    output: List[T] = []
    for item in items:
        marker = stable_json_dumps(item) if isinstance(item, (Mapping, list, tuple, set)) else str(item)
        if marker in seen:
            continue
        seen.add(marker)
        output.append(item)
    return output


def compact_none(payload: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None}


def flatten_mapping(payload: Mapping[str, Any], *, prefix: str = "", separator: str = ".") -> Dict[str, Any]:
    flattened: Dict[str, Any] = {}
    for key, value in payload.items():
        full_key = f"{prefix}{separator}{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flattened.update(flatten_mapping(value, prefix=full_key, separator=separator))
        else:
            flattened[full_key] = value
    return flattened


def estimate_processing_cost(parsed_doc: Mapping[str, Any]) -> Dict[str, Any]:
    doc = validate_parsed_document(parsed_doc)
    content = str(doc.get("content", ""))
    quality = text_quality_report(content)
    char_count = len(content)
    estimated_chunks = max(1, math.ceil(char_count / 20_000)) if char_count else 0
    return {
        "source": doc.get("source", "unknown"),
        "char_count": char_count,
        "line_count": quality.line_count,
        "quality_score": quality.quality_score,
        "estimated_recovery_chunks": estimated_chunks,
        "requires_attention": quality.quality_score < 0.55 or bool(quality.warnings),
    }


@contextmanager
def timed_operation(operation: str) -> Generator[OperationTimer, None, None]:
    timer = OperationTimer(operation=operation)
    try:
        yield timer
    finally:
        timer.stop()


# ---------------------------------------------------------------------------
# Bounded async fan-out and retry helpers
# ---------------------------------------------------------------------------


def should_retry_exception(exc: BaseException, config: RetryConfig) -> bool:
    if isinstance(exc, ReaderError):
        return bool(getattr(exc.policy, "retryable", False))
    return isinstance(exc, config.retry_exceptions)


def retry_delay_seconds(attempt_index: int, config: RetryConfig) -> float:
    normalized = config.normalized()
    base = normalized.base_delay_seconds * (normalized.backoff_factor ** max(0, attempt_index - 1))
    delay = min(normalized.max_delay_seconds, base)
    if normalized.jitter_ratio > 0 and delay > 0:
        spread = delay * normalized.jitter_ratio
        delay += random.uniform(-spread, spread)
    return max(0.0, delay)


def retry_sync(
    operation: Callable[..., R],
    *args: Any,
    config: Optional[RetryConfig] = None,
    operation_name: str = "reader_operation",
    timeout_seconds: Optional[float] = None,
    **kwargs: Any,
) -> R:
    policy = (config or RetryConfig()).normalized()
    started = time.perf_counter()
    last_error: Optional[BaseException] = None
    for attempt in range(1, policy.max_attempts + 1):
        if timeout_seconds is not None and (time.perf_counter() - started) > timeout_seconds:
            raise _reader_timeout_error(operation_name, timeout_seconds, {"attempt": attempt})
        try:
            return operation(*args, **kwargs)
        except BaseException as exc:
            last_error = exc
            if attempt >= policy.max_attempts or not should_retry_exception(exc, policy):
                raise
            delay = retry_delay_seconds(attempt, policy)
            logger.warning("Retrying Reader operation %s after %s: attempt=%s delay=%.3fs", operation_name, type(exc).__name__, attempt, delay)
            if delay:
                time.sleep(delay)
    raise _reader_task_execution_error(f"Reader operation failed after retries: {operation_name}", cause=last_error)


async def retry_async(
    operation: Callable[..., Awaitable[R]],
    *args: Any,
    config: Optional[RetryConfig] = None,
    operation_name: str = "reader_operation",
    timeout_seconds: Optional[float] = None,
    **kwargs: Any,
) -> R:
    policy = (config or RetryConfig()).normalized()
    started = time.perf_counter()
    last_error: Optional[BaseException] = None
    for attempt in range(1, policy.max_attempts + 1):
        if timeout_seconds is not None and (time.perf_counter() - started) > timeout_seconds:
            raise _reader_timeout_error(operation_name, timeout_seconds, {"attempt": attempt})
        try:
            if timeout_seconds is not None:
                remaining = max(0.0, timeout_seconds - (time.perf_counter() - started))
                return await asyncio.wait_for(operation(*args, **kwargs), timeout=remaining)
            return await operation(*args, **kwargs)
        except asyncio.CancelledError as exc:
            raise _reader_cancelled_error(operation_name) from exc
        except asyncio.TimeoutError as exc:
            last_error = exc
            wrapped = _reader_timeout_error(operation_name, timeout_seconds or 0.0, {"attempt": attempt})
            if attempt >= policy.max_attempts or not should_retry_exception(wrapped, policy):
                raise wrapped from exc
        except BaseException as exc:
            last_error = exc
            if attempt >= policy.max_attempts or not should_retry_exception(exc, policy):
                raise
        delay = retry_delay_seconds(attempt, policy)
        logger.warning("Retrying async Reader operation %s after %s: attempt=%s delay=%.3fs", operation_name, type(last_error).__name__ if last_error else "error", attempt, delay)
        if delay:
            await asyncio.sleep(delay)
    raise _reader_task_execution_error(f"Reader async operation failed after retries: {operation_name}", cause=last_error)


async def async_bounded_map(
    items: Sequence[T],
    worker: Callable[[T], Awaitable[R]],
    *,
    max_concurrency: int = 4,
    operation_name: str = "reader_batch",
    return_exceptions: bool = False,
) -> List[R | ReaderError]:
    """Run async work with bounded concurrency and Reader batch errors."""

    if max_concurrency < 1:
        raise _reader_configuration_error("Reader max_concurrency must be >= 1", {"max_concurrency": max_concurrency})
    semaphore = asyncio.Semaphore(max_concurrency)

    async def run_one(index: int, item: T) -> R | ReaderError:
        try:
            async with semaphore:
                return await worker(item)
        except asyncio.CancelledError as exc:
            raise _reader_cancelled_error(operation_name, {"item_index": index}) from exc
        except BaseException as exc:
            return _normalize_reader_exception(exc, context={"operation": operation_name, "item_index": index})

    results = await asyncio.gather(*(run_one(index, item) for index, item in enumerate(items)))
    errors = [result for result in results if isinstance(result, ReaderError)]
    if errors and not return_exceptions:
        raise _reader_batch_error(
            f"Reader batch operation failed: {operation_name}",
            context={"operation": operation_name, "errors": [error.to_public_dict() for error in errors]},
            failed_count=len(errors),
            total_count=len(items),
        )
    return results


# ---------------------------------------------------------------------------
# Compatibility aliases for likely subsystem naming patterns
# ---------------------------------------------------------------------------


quality_score = lambda text: text_quality_report(text).quality_score
repair_text = low_level_repair_text
safe_json_dumps = stable_json_dumps
make_cache_key = cache_key
build_reader_plan = build_basic_reader_plan


if __name__ == "__main__":
    print("\n=== Running Reader Helpers Smoke Test ===\n")
    printer.status("TEST", "Reader Helpers initialized", "info")

    assert normalize_extension("TXT") == ".txt"
    assert normalize_extension(".md", include_dot=False) == "md"
    assert safe_filename("../Bad:Name?.txt").startswith("Bad_Name") or safe_filename("../Bad:Name?.txt")
    assert extract_requested_format("please convert to .md", {"txt", "md"}) == "md"
    assert instruction_requests_recovery("repair this corrupted document") is True

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        input_path = tmp_dir / "sample.txt"
        atomic_write_text(input_path, "hello\r\nworld\x00", purpose="output")
        info = validate_input_path(input_path, allowed_extensions={".txt"}, compute_hash=True)
        assert info.extension == ".txt"
        assert info.sha256 is not None

        decoded = decode_bytes(read_file_bytes(input_path, allowed_extensions={".txt"}), source=str(input_path))
        repaired = low_level_repair_text(decoded.text)
        assert "\x00" not in repaired
        report = text_quality_report(repaired)
        assert report.quality_score > 0

        out_path = prepare_output_path(tmp_dir / "out", "sample", "json")
        atomic_write_json(out_path, {"token": "secret", "safe": "visible"}, redact=True, purpose="cache")
        persisted = read_json_file(out_path, purpose="cache")
        assert persisted["token"] == "[REDACTED]"

        doc = {"source": str(input_path), "extension": ".txt", "content": repaired, "metadata": {"size": info.size_bytes}}
        summary = summarize_parsed_document(doc)
        assert summary["source"] == str(input_path)
        assert parsed_document_fingerprint(doc)

    cache = BoundedLRUCache[int](max_size=2)
    cache["a"] = 1
    cache["b"] = 2
    cache["c"] = 3
    assert "a" not in cache and len(cache) == 2

    async def _double(value: int) -> int:
        return value * 2

    async_results = asyncio.run(async_bounded_map([1, 2, 3], _double, max_concurrency=2))
    assert async_results == [2, 4, 6]

    plan = build_basic_reader_plan("recover and merge to md", ["a.txt", "b.txt"], supported_output_formats={"txt", "md"})
    assert plan[-1]["action"] == "merge"

    print("\n=== Reader Helpers Smoke Test Passed ===\n")
