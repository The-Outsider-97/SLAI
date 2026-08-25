"""
Centralised, production-grade helper utilities for the data pipeline.

Covers five responsibility domains, each in its own clearly-delineated
section:

    1. Path & File I/O      – safe resolution, format detection, atomic writes
    2. Sanitization         – HTML escaping, string cleaning, dict / DataFrame
    3. Type Coercion        – safe casts with full error context
    4. Payload Inspection   – null auditing, modality stats, alignment checks
    5. Retry & Resilience   – exponential-backoff decorator, timed execution

Design principles
-----------------
* Every public function raises a typed ``DataError`` subclass — never a bare
  ``ValueError`` or ``RuntimeError`` — so callers can pattern-match reliably.
* All functions are pure or explicitly side-effect-free where possible;
  stateful helpers (e.g. retry) are implemented as decorators or context
  managers to keep call sites clean.
* No module-level I/O or heavy imports at import time (``pandas`` /
  ``numpy`` are guarded inside functions that actually need them).
* Thread-safe by construction — no mutable module-level state.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import html
import json
import math
import os
import re
import tempfile
import time

from contextlib import contextmanager
from pathlib import Path
from typing import (Any, Callable, Dict, Generator, Iterable, List, Type,
                    Mapping, Optional, Sequence, Tuple, TypeVar, Union)

from .data_error import *
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Data Helpers")
printer = PrettyPrinter()

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: File extensions the pipeline knows how to handle.
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {"json", "yaml", "yml", "csv", "parquet", "pickle", "pkl", "npy", "npz"}
)

#: Magic-byte signatures used for content-based format detection.
#: Format: ``{extension: (offset, expected_bytes)}``.
_MAGIC_BYTES: Dict[str, Tuple[int, bytes]] = {
    "parquet": (0, b"PAR1"),
    "pickle":  (0, b"\x80\x04"),   # protocol 4 — most common in production
    "pkl":     (0, b"\x80\x04"),
    "npy":     (0, b"\x93NUMPY"),
    "npz":     (0, b"PK\x03\x04"), # NPZ is a ZIP archive
}

#: Characters that are unconditionally stripped from string fields.
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

#: Matches a valid, absolute-looking path traversal attempt.
_PATH_TRAVERSAL_RE = re.compile(r"(?:^|[\\/])\.\.(?:[\\/]|$)")


# ===========================================================================
# Section 1 — Path & File I/O
# ===========================================================================
def resolve_path(
    raw: Union[str, Path],
    *,
    must_exist: bool = False,
    allow_symlinks: bool = False,
) -> Path:
    """Return a fully resolved, canonical ``Path`` for *raw*.

    Parameters
    ----------
    raw:
        A string or ``Path`` that may be relative or contain ``..`` segments.
    must_exist:
        When ``True``, raise ``DataSourceError`` if the resolved path does
        not exist on disk.
    allow_symlinks:
        When ``False`` (default), raise ``DataSourceError`` if the resolved
        path is a symbolic link — consistent with the security policy in
        ``SafeDataLoader._validate_file_path``.

    Returns
    -------
    Path
        Fully resolved, absolute ``Path``.

    Raises
    ------
    DataConfigError
        If *raw* is empty or contains a path-traversal sequence before
        resolution.
    DataSourceError
        If *must_exist* is ``True`` and the path does not exist, or if
        the path is a symlink and *allow_symlinks* is ``False``.
    """
    raw_str = str(raw).strip()
    if not raw_str:
        raise DataConfigError(
            "Path argument is empty",
            context={"raw": repr(raw)},
        )
    if _PATH_TRAVERSAL_RE.search(raw_str):
        raise DataConfigError(
            "Path contains a traversal sequence ('..')",
            context={"raw": raw_str},
        )

    resolved = Path(raw_str).resolve()

    if resolved.is_symlink() and not allow_symlinks:
        raise DataSourceError(
            "Symbolic links are not permitted",
            context={"path": str(resolved)},
        )
    if must_exist and not resolved.exists():
        raise DataSourceError(
            "Path does not exist",
            context={"path": str(resolved)},
        )

    return resolved


def assert_file_readable(path: Union[str, Path]) -> Path:
    """Resolve *path* and assert it is a readable, non-symlink regular file.

    Returns the resolved ``Path`` for convenience so callers can chain:
    ``f = open(assert_file_readable(p))``.

    Raises
    ------
    DataSourceError
        On any access problem (missing, directory, symlink, unreadable).
    """
    resolved = resolve_path(path, must_exist=True)
    if resolved.is_dir():
        raise DataSourceError(
            "Path is a directory, not a file",
            context={"path": str(resolved)},
        )
    if not os.access(resolved, os.R_OK):
        raise DataSourceError(
            "File exists but is not readable — check permissions",
            context={"path": str(resolved)},
        )
    return resolved


def detect_format(path: Union[str, Path]) -> str:
    """Return the canonical lowercase format token for *path*.

    Detection strategy (in priority order):

    1. **Magic-byte check** — read the first 8 bytes and compare against
       known signatures.  This catches misnamed files.
    2. **Extension fallback** — use the suffix if no magic bytes matched.

    Parameters
    ----------
    path:
        Path to an existing file.  The file must be readable.

    Returns
    -------
    str
        One of the tokens in ``SUPPORTED_EXTENSIONS``
        (e.g. ``"parquet"``, ``"json"``).

    Raises
    ------
    DataSourceError
        If the file is unreadable or the format cannot be determined.
    DataValidationError
        If the format is not in ``SUPPORTED_EXTENSIONS``.
    """
    resolved = assert_file_readable(path)
    ext = resolved.suffix.lstrip(".").lower()
    # Normalise .yml → yaml
    if ext == "yml":
        ext = "yaml"

    try:
        with open(resolved, "rb") as fh:
            header = fh.read(8)
    except OSError as exc:
        raise DataSourceError(
            "Could not read file header for format detection",
            context={"path": str(resolved)},
            cause=exc,
        ) from exc

    # Magic-byte check (highest priority)
    for fmt, (offset, magic) in _MAGIC_BYTES.items():
        if header[offset: offset + len(magic)] == magic:
            return fmt

    # Extension fallback
    if ext not in SUPPORTED_EXTENSIONS:
        raise DataValidationError(
            f"Unsupported file format: '.{ext}'",
            context={
                "path": str(resolved),
                "extension": ext,
                "supported": sorted(SUPPORTED_EXTENSIONS),
            },
        )
    return ext


def atomic_write_json(data: Any, dest: Union[str, Path], *,
                      indent: int = 2, encoding: str = "utf-8") -> Path:
    """Serialise *data* to JSON and write it atomically to *dest*.

    Uses a sibling temp-file + ``os.replace`` so that concurrent readers of
    *dest* never see a partially written file.

    Parameters
    ----------
    data:
        Any JSON-serialisable object.
    dest:
        Target file path.  Parent directory must already exist.
    indent:
        JSON indentation level.
    encoding:
        File encoding (default UTF-8).

    Returns
    -------
    Path
        The fully resolved *dest* path.

    Raises
    ------
    DataConfigError
        If the parent directory does not exist or is not writable.
    DataTransformError
        If *data* is not JSON-serialisable.
    """
    dest_path = Path(dest).resolve()
    parent = dest_path.parent

    if not parent.exists():
        raise DataConfigError(
            "Destination directory does not exist",
            context={"directory": str(parent)},
        )
    if not os.access(parent, os.W_OK):
        raise DataConfigError(
            "Destination directory is not writable",
            context={"directory": str(parent)},
        )

    try:
        payload = json.dumps(data, indent=indent, default=str, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise DataTransformError(
            "Data is not JSON-serialisable",
            context={"type": type(data).__name__},
            cause=exc,
        ) from exc

    # Write to a sibling temp-file then atomically replace
    fd, tmp_path = tempfile.mkstemp(dir=parent, prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding=encoding) as fh:
            fh.write(payload)
        os.replace(tmp_path, dest_path)
    except OSError as exc:
        # Clean up orphan temp file on failure
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise DataTransformError(
            "Atomic write failed",
            context={"dest": str(dest_path), "tmp": tmp_path},
            cause=exc,
        ) from exc

    logger.debug({"event": "atomic_write_json", "dest": str(dest_path)})
    return dest_path


def compute_file_hash(
    path: Union[str, Path],
    algorithm: str = "sha256",
    *,
    chunk_size: int = 1 << 20,  # 1 MiB
) -> str:
    """Return the hex digest of *path* using *algorithm*.

    Streams the file in *chunk_size* byte chunks so large files never fully
    reside in memory.

    Raises
    ------
    DataConfigError
        If *algorithm* is not supported by :mod:`hashlib`.
    DataSourceError
        If the file cannot be read.
    """
    try:
        hash_fn = hashlib.new(algorithm)
    except ValueError as exc:
        raise DataConfigError(
            "Unsupported hash algorithm",
            context={"algorithm": algorithm},
            cause=exc,
        ) from exc

    resolved = assert_file_readable(path)
    try:
        with open(resolved, "rb") as fh:
            while chunk := fh.read(chunk_size):
                hash_fn.update(chunk)
    except OSError as exc:
        raise DataSourceError(
            "File read error during hashing",
            context={"path": str(resolved), "algorithm": algorithm},
            cause=exc,
        ) from exc

    return hash_fn.hexdigest()


# ===========================================================================
# Section 2 — Sanitization
# ===========================================================================

def sanitize_string(
    value: str,
    *,
    escape_html: bool = True,
    strip_control_chars: bool = True,
    max_length: Optional[int] = None,
    field_name: str = "<unknown>",
) -> str:
    """Return a sanitised copy of *value*.

    Operations applied in order:

    1. Strip leading / trailing whitespace.
    2. Remove ASCII control characters (``\\x00``–``\\x1f`` excl. tab/LF/CR).
    3. HTML-escape ``< > & ' "`` to prevent stored-XSS if values are ever
       rendered in a web context.
    4. Truncate to *max_length* if specified.

    Parameters
    ----------
    value:
        The raw string to sanitise.
    escape_html:
        When ``True`` (default), HTML-escape the value.
    strip_control_chars:
        When ``True`` (default), remove non-printable control characters.
    max_length:
        If provided and positive, truncate the result to this length.
    field_name:
        Used in error context if truncation or type issues arise.

    Returns
    -------
    str
        Sanitised string.

    Raises
    ------
    DataValidationError
        If *value* is not a ``str``.
    """
    if not isinstance(value, str):
        raise DataValidationError(
            f"sanitize_string expected str for field '{field_name}', "
            f"got {type(value).__name__}",
            context={"field": field_name, "type": type(value).__name__},
        )

    result = value.strip()

    if strip_control_chars:
        result = _CONTROL_CHAR_RE.sub("", result)

    if escape_html:
        result = html.escape(result, quote=True)

    if max_length is not None and max_length > 0 and len(result) > max_length:
        logger.debug({
            "event": "string_truncated",
            "field": field_name,
            "original_length": len(result),
            "max_length": max_length,
        })
        result = result[:max_length]

    return result


def sanitize_dict(
    data: Dict[str, Any],
    *,
    escape_html: bool = True,
    strip_control_chars: bool = True,
    max_string_length: Optional[int] = None,
    _depth: int = 0,
    max_depth: int = 32,
) -> Dict[str, Any]:
    """Recursively sanitise all string values in *data*.

    Non-string values are passed through unchanged.  Lists are iterated
    element-by-element; nested dicts recurse up to *max_depth* levels.

    Parameters
    ----------
    data:
        The dictionary to sanitise.
    escape_html:
        Forwarded to :func:`sanitize_string`.
    strip_control_chars:
        Forwarded to :func:`sanitize_string`.
    max_string_length:
        Forwarded to :func:`sanitize_string` as *max_length*.
    max_depth:
        Maximum recursion depth before raising ``DataValidationError``.

    Returns
    -------
    dict
        New dictionary with the same structure as *data* but with all string
        leaves sanitised.

    Raises
    ------
    DataValidationError
        If *data* is not a ``dict`` or *max_depth* is exceeded.
    """
    if not isinstance(data, dict):
        raise DataValidationError(
            "sanitize_dict expected a dict",
            context={"type": type(data).__name__},
        )
    if _depth > max_depth:
        raise DataValidationError(
            f"sanitize_dict exceeded max recursion depth ({max_depth})",
            context={"max_depth": max_depth},
        )

    sanitized: Dict[str, Any] = {}
    for key, value in data.items():
        sanitized[key] = _sanitize_value(
            value,
            escape_html=escape_html,
            strip_control_chars=strip_control_chars,
            max_string_length=max_string_length,
            depth=_depth,
            max_depth=max_depth,
            field_name=str(key),
        )
    return sanitized


def _sanitize_value(
    value: Any,
    *,
    escape_html: bool,
    strip_control_chars: bool,
    max_string_length: Optional[int],
    depth: int,
    max_depth: int,
    field_name: str,
) -> Any:
    """Dispatch helper used by :func:`sanitize_dict`."""
    if isinstance(value, str):
        return sanitize_string(
            value,
            escape_html=escape_html,
            strip_control_chars=strip_control_chars,
            max_length=max_string_length,
            field_name=field_name,
        )
    if isinstance(value, dict):
        return sanitize_dict(
            value,
            escape_html=escape_html,
            strip_control_chars=strip_control_chars,
            max_string_length=max_string_length,
            _depth=depth + 1,
            max_depth=max_depth,
        )
    if isinstance(value, list):
        return [
            _sanitize_value(
                item,
                escape_html=escape_html,
                strip_control_chars=strip_control_chars,
                max_string_length=max_string_length,
                depth=depth + 1,
                max_depth=max_depth,
                field_name=f"{field_name}[]",
            )
            for item in value
        ]
    return value


def sanitize_dataframe(
    df: Any,  # pd.DataFrame — typed as Any to avoid hard import at module level
    *,
    escape_html: bool = True,
    strip_control_chars: bool = True,
    max_string_length: Optional[int] = None,
    lowercase_columns: bool = True,
) -> Any:
    """Sanitise all object-dtype columns in a ``pandas.DataFrame``.

    Parameters
    ----------
    df:
        The ``DataFrame`` to sanitise.
    escape_html:
        HTML-escape string cells.
    strip_control_chars:
        Remove control characters from string cells.
    max_string_length:
        Truncate string cells exceeding this length.
    lowercase_columns:
        Normalise column names to lowercase (consistent with
        ``SafeDataLoader._validate_df_schema``).

    Returns
    -------
    pd.DataFrame
        A copy of *df* with sanitised columns.

    Raises
    ------
    DataValidationError
        If *df* is not a ``pandas.DataFrame``.
    """
    try:
        import pandas as pd  # type: ignore # local import — do not pollute module namespace
    except ImportError as exc:
        raise DataConfigError(
            "pandas is required for sanitize_dataframe",
            context={},
            cause=exc,
        ) from exc

    if not isinstance(df, pd.DataFrame):
        raise DataValidationError(
            "sanitize_dataframe expected a pandas DataFrame",
            context={"type": type(df).__name__},
        )

    result = df.copy()

    if lowercase_columns:
        result.columns = result.columns.str.lower()  # type: ignore[assignment]

    str_cols = result.select_dtypes(include=["object"]).columns
    for col in str_cols:
        result[col] = result[col].map(
            lambda x: sanitize_string(
                x,
                escape_html=escape_html,
                strip_control_chars=strip_control_chars,
                max_length=max_string_length,
                field_name=col,
            )
            if isinstance(x, str)
            else x
        )

    return result


# ===========================================================================
# Section 3 — Type Coercion
# ===========================================================================

def safe_cast(
    value: Any,
    target_type: Type[T],
    *,
    field_name: str = "<unknown>",
    modality: str = "<unknown>",
    row_idx: Optional[int] = None,
    nullable: bool = False,
) -> Optional[T]:
    """Cast *value* to *target_type*, raising a typed error on failure.

    Parameters
    ----------
    value:
        The raw value to cast.
    target_type:
        The Python type to cast to.
    field_name:
        Used in error messages for traceability.
    modality:
        The data modality the field belongs to.
    row_idx:
        Row index, if available, for precise error context.
    nullable:
        When ``True``, ``None`` values are returned as-is.

    Returns
    -------
    T | None
        The cast value, or ``None`` if *nullable* and *value* is ``None``.

    Raises
    ------
    DataValidationError
        If the cast fails or if *value* is ``None`` but *nullable* is
        ``False``.
    """
    if value is None:
        if nullable:
            return None
        raise DataValidationError(
            f"Field '{field_name}' is None but not nullable",
            context={
                "field": field_name,
                "modality": modality,
                "row_idx": row_idx,
            },
        )

    if isinstance(value, target_type):
        return value  # type: ignore[return-value]

    try:
        return target_type(value)  # type: ignore[call-arg]
    except (TypeError, ValueError) as exc:
        raise DataValidationError(
            f"Cannot cast field '{field_name}' to {target_type.__name__}: "
            f"got {type(value).__name__} = {value!r}",
            context={
                "field": field_name,
                "modality": modality,
                "row_idx": row_idx,
                "target_type": target_type.__name__,
                "actual_type": type(value).__name__,
                "value_repr": repr(value),
            },
            cause=exc,
        ) from exc


def safe_int(
    value: Any,
    *,
    field_name: str = "<unknown>",
    modality: str = "<unknown>",
    row_idx: Optional[int] = None,
    nullable: bool = False,
    min_value: Optional[int] = None,
    max_value: Optional[int] = None,
) -> Optional[int]:
    """Cast *value* to ``int`` with optional range validation.

    Floats that are not whole numbers (e.g. ``3.7``) are rejected — silent
    truncation would mask data quality issues.

    Raises
    ------
    DataValidationError
        On type mismatch, non-integral float, or out-of-range value.
    """
    result: Optional[int] = safe_cast(
        value,
        int,
        field_name=field_name,
        modality=modality,
        row_idx=row_idx,
        nullable=nullable,
    )
    if result is None:
        return None

    # Guard against silent float truncation: 3.7 → 3
    if isinstance(value, float) and not value.is_integer():
        raise DataValidationError(
            f"Field '{field_name}' has a non-integer float value {value!r}",
            context={
                "field": field_name,
                "modality": modality,
                "row_idx": row_idx,
                "value": value,
            },
        )

    if min_value is not None and result < min_value:
        raise DataValidationError(
            f"Field '{field_name}' value {result} is below minimum {min_value}",
            context={
                "field": field_name,
                "modality": modality,
                "row_idx": row_idx,
                "value": result,
                "min_value": min_value,
            },
        )
    if max_value is not None and result > max_value:
        raise DataValidationError(
            f"Field '{field_name}' value {result} exceeds maximum {max_value}",
            context={
                "field": field_name,
                "modality": modality,
                "row_idx": row_idx,
                "value": result,
                "max_value": max_value,
            },
        )

    return result


def safe_float(
    value: Any,
    *,
    field_name: str = "<unknown>",
    modality: str = "<unknown>",
    row_idx: Optional[int] = None,
    nullable: bool = False,
    allow_nan: bool = False,
    allow_inf: bool = False,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
) -> Optional[float]:
    """Cast *value* to ``float`` with NaN/Inf and range validation.

    Parameters
    ----------
    allow_nan:
        When ``False`` (default), reject ``float('nan')`` values — NaN in a
        feature tensor is typically a transform bug, not valid data.
    allow_inf:
        When ``False`` (default), reject ``float('inf')`` / ``float('-inf')``.

    Raises
    ------
    DataValidationError
        On type mismatch, NaN/Inf when disallowed, or out-of-range value.
    """
    result: Optional[float] = safe_cast(
        value,
        float,
        field_name=field_name,
        modality=modality,
        row_idx=row_idx,
        nullable=nullable,
    )
    if result is None:
        return None

    if not allow_nan and math.isnan(result):
        raise DataValidationError(
            f"Field '{field_name}' contains NaN",
            context={"field": field_name, "modality": modality, "row_idx": row_idx},
        )
    if not allow_inf and math.isinf(result):
        raise DataValidationError(
            f"Field '{field_name}' contains Inf",
            context={"field": field_name, "modality": modality, "row_idx": row_idx},
        )
    if min_value is not None and result < min_value:
        raise DataValidationError(
            f"Field '{field_name}' value {result} is below minimum {min_value}",
            context={
                "field": field_name,
                "modality": modality,
                "row_idx": row_idx,
                "value": result,
                "min_value": min_value,
            },
        )
    if max_value is not None and result > max_value:
        raise DataValidationError(
            f"Field '{field_name}' value {result} exceeds maximum {max_value}",
            context={
                "field": field_name,
                "modality": modality,
                "row_idx": row_idx,
                "value": result,
                "max_value": max_value,
            },
        )

    return result


def safe_bool(
    value: Any,
    *,
    field_name: str = "<unknown>",
    modality: str = "<unknown>",
    row_idx: Optional[int] = None,
    nullable: bool = False,
) -> Optional[bool]:
    """Cast *value* to ``bool`` with explicit string-token handling.

    Accepted truthy strings (case-insensitive): ``"true"``, ``"1"``, ``"yes"``.
    Accepted falsy strings: ``"false"``, ``"0"``, ``"no"``.

    Integers ``0`` and ``1`` are accepted; any other integer is rejected to
    prevent silent misinterpretation of flag fields.

    Raises
    ------
    DataValidationError
        On unrecognised value.
    """
    if value is None:
        if nullable:
            return None
        raise DataValidationError(
            f"Field '{field_name}' is None but not nullable",
            context={"field": field_name, "modality": modality, "row_idx": row_idx},
        )

    if isinstance(value, bool):
        return value

    if isinstance(value, int):
        if value in (0, 1):
            return bool(value)
        raise DataValidationError(
            f"Field '{field_name}' integer {value!r} is not a valid bool (expected 0 or 1)",
            context={"field": field_name, "modality": modality, "row_idx": row_idx, "value": value},
        )

    if isinstance(value, str):
        norm = value.strip().lower()
        if norm in ("true", "1", "yes"):
            return True
        if norm in ("false", "0", "no"):
            return False
        raise DataValidationError(
            f"Field '{field_name}' string {value!r} cannot be interpreted as bool",
            context={"field": field_name, "modality": modality, "row_idx": row_idx, "value": value},
        )

    raise DataValidationError(
        f"Field '{field_name}' type {type(value).__name__!r} cannot be cast to bool",
        context={"field": field_name, "modality": modality, "row_idx": row_idx, "type": type(value).__name__},
    )


# ===========================================================================
# Section 4 — Payload Inspection
# ===========================================================================

def audit_nulls(payload: Mapping[str, Sequence[Mapping[str, Any]]]) -> Dict[str, Any]:
    """Compute per-modality and aggregate null statistics for *payload*.

    A null is defined as any value for which ``value is None``.

    Returns
    -------
    dict
        Structure::

            {
                "total_records": int,
                "total_cells": int,
                "total_nulls": int,
                "null_ratio": float,           # total_nulls / total_cells
                "per_modality": {
                    "<modality>": {
                        "record_count": int,
                        "null_count": int,
                        "null_ratio": float,
                        "null_by_field": {"<field>": int, ...}
                    },
                    ...
                },
            }

    Raises
    ------
    DataIngestionContractError
        If *payload* is not a non-empty mapping.
    """
    if not isinstance(payload, Mapping) or not payload:
        raise DataIngestionContractError(
            "audit_nulls requires a non-empty payload mapping",
            context={"type": type(payload).__name__},
        )

    total_records = 0
    total_cells = 0
    total_nulls = 0
    per_modality: Dict[str, Any] = {}

    for modality, rows in payload.items():
        null_by_field: Dict[str, int] = {}
        modality_nulls = 0

        for row in rows:
            for field, val in row.items():
                total_cells += 1
                if val is None:
                    modality_nulls += 1
                    null_by_field[field] = null_by_field.get(field, 0) + 1

        record_count = len(rows)
        total_records += record_count
        total_nulls += modality_nulls
        modality_cells = sum(len(r) for r in rows)

        per_modality[modality] = {
            "record_count": record_count,
            "null_count": modality_nulls,
            "null_ratio": round(modality_nulls / max(modality_cells, 1), 8),
            "null_by_field": null_by_field,
        }

    return {
        "total_records": total_records,
        "total_cells": total_cells,
        "total_nulls": total_nulls,
        "null_ratio": round(total_nulls / max(total_cells, 1), 8),
        "per_modality": per_modality,
    }


def compute_modality_stats(payload: Mapping[str, Sequence[Mapping[str, Any]]]) -> Dict[str, Any]:
    """Return descriptive statistics for each modality in *payload*.

    For each modality, computes:

    * ``record_count`` — number of rows.
    * ``field_names`` — sorted list of unique field names across all rows.
    * ``field_count_range`` — ``(min, max)`` number of fields per row
      (useful for detecting ragged records).
    * ``unique_field_counts`` — frequency of each distinct per-row field count.

    Returns
    -------
    dict
        Keyed by modality name.

    Raises
    ------
    DataIngestionContractError
        If *payload* is not a non-empty mapping.
    """
    if not isinstance(payload, Mapping) or not payload:
        raise DataIngestionContractError(
            "compute_modality_stats requires a non-empty payload mapping",
            context={"type": type(payload).__name__},
        )

    stats: Dict[str, Any] = {}
    for modality, rows in payload.items():
        all_fields: set[str] = set()
        per_row_field_counts: List[int] = []

        for row in rows:
            all_fields.update(row.keys())
            per_row_field_counts.append(len(row))

        count_freq: Dict[int, int] = {}
        for c in per_row_field_counts:
            count_freq[c] = count_freq.get(c, 0) + 1

        stats[modality] = {
            "record_count": len(rows),
            "field_names": sorted(all_fields),
            "field_count_range": (
                min(per_row_field_counts, default=0),
                max(per_row_field_counts, default=0),
            ),
            "unique_field_counts": count_freq,
            "is_ragged": len(count_freq) > 1,
        }

    return stats


def assert_modalities_aligned(
    payload: Mapping[str, Sequence[Any]],
    *,
    expected_modalities: Optional[Sequence[str]] = None,
) -> Dict[str, int]:
    """Assert that all modalities in *payload* have the same record count.

    Parameters
    ----------
    payload:
        Mapping of modality name → sequence of records.
    expected_modalities:
        If provided, also assert that exactly these modalities are present
        (no more, no fewer).

    Returns
    -------
    dict
        ``{modality: record_count}`` — useful for logging after the assertion.

    Raises
    ------
    DataIngestionContractError
        On empty payload, missing/unexpected modalities, or misaligned lengths.
    """
    if not payload:
        raise DataIngestionContractError(
            "Payload is empty — cannot assert alignment",
            context={},
        )

    if expected_modalities is not None:
        expected_set = set(expected_modalities)
        actual_set = set(payload.keys())
        missing = expected_set - actual_set
        unexpected = actual_set - expected_set
        if missing or unexpected:
            raise DataIngestionContractError(
                "Payload modality mismatch",
                context={
                    "expected": sorted(expected_set),
                    "actual": sorted(actual_set),
                    "missing": sorted(missing),
                    "unexpected": sorted(unexpected),
                },
            )

    lengths = {m: len(rows) for m, rows in payload.items()}
    if len(set(lengths.values())) > 1:
        raise DataIngestionContractError(
            f"Modality alignment failed — unequal record counts: {lengths}",
            context={"lengths": lengths},
        )

    return lengths


def chunk_sequence(sequence: Sequence[T], chunk_size: int, *,
                   drop_remainder: bool = False) -> Generator[Sequence[T], None, None]:
    """Yield successive *chunk_size* slices of *sequence*.

    Parameters
    ----------
    sequence:
        Any indexable sequence.
    chunk_size:
        Number of items per chunk.  Must be ≥ 1.
    drop_remainder:
        If ``True``, discard the final chunk when ``len(sequence)`` is not
        divisible by *chunk_size*.

    Yields
    ------
    Sequence[T]
        Slices of the original sequence.

    Raises
    ------
    DataIngestionContractError
        If *chunk_size* < 1.
    """
    if chunk_size < 1:
        raise DataIngestionContractError(
            f"chunk_size must be >= 1, got {chunk_size}",
            context={"chunk_size": chunk_size},
        )

    total = len(sequence)
    for start in range(0, total, chunk_size):
        end = start + chunk_size
        if drop_remainder and end > total:
            return
        yield sequence[start:end]


def flatten_records(
    records: Iterable[Mapping[str, Any]],
    *,
    separator: str = ".",
    max_depth: int = 8,
) -> List[Dict[str, Any]]:
    """Flatten nested dicts within *records* into dot-separated key paths.

    Useful before passing deeply nested JSON records into a DataFrame.

    Parameters
    ----------
    records:
        Iterable of potentially nested record dicts.
    separator:
        Delimiter for joined key paths (default ``"."``).
    max_depth:
        Maximum nesting depth to traverse.

    Returns
    -------
    list[dict]
        Flat records.

    Raises
    ------
    DataTransformError
        If *max_depth* is exceeded for any record.
    """

    def _flatten(obj: Any, prefix: str, depth: int) -> Dict[str, Any]:
        if depth > max_depth:
            raise DataTransformError(
                f"flatten_records exceeded max_depth={max_depth}",
                context={"prefix": prefix, "max_depth": max_depth},
            )
        out: Dict[str, Any] = {}
        if isinstance(obj, dict):
            for k, v in obj.items():
                new_key = f"{prefix}{separator}{k}" if prefix else k
                out.update(_flatten(v, new_key, depth + 1))
        else:
            out[prefix] = obj
        return out

    return [_flatten(rec, "", 0) for rec in records]


# ===========================================================================
# Section 5 — Retry & Resilience
# ===========================================================================

def with_retry(
    max_attempts: int = 3,
    *,
    base_delay: float = 0.5,
    max_delay: float = 30.0,
    backoff_factor: float = 2.0,
    jitter: float = 0.1,
    retryable_exceptions: Tuple[Type[BaseException], ...] = (DataSourceError,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator: retry the wrapped function with exponential back-off.

    Back-off schedule::

        delay_n = min(base_delay × backoff_factor^(n-1), max_delay)
                  + uniform(-jitter × delay_n, +jitter × delay_n)

    Parameters
    ----------
    max_attempts:
        Total number of attempts (including the first).  Must be ≥ 1.
    base_delay:
        Seconds to wait before the second attempt.
    max_delay:
        Upper cap for the inter-attempt delay.
    backoff_factor:
        Multiplicative growth factor per attempt.
    jitter:
        Fractional random jitter added to the computed delay to avoid
        thundering-herd effects (0 = no jitter, 1 = ±100 % of delay).
    retryable_exceptions:
        Only exceptions of these types (or subclasses) trigger a retry.
        Any other exception propagates immediately.

    Returns
    -------
    Callable
        The decorated function.

    Raises
    ------
    DataConfigError
        If *max_attempts* < 1 or *base_delay* ≤ 0.

    Example
    -------
    ::

        @with_retry(max_attempts=5, retryable_exceptions=(DataSourceError,))
        def fetch_shard(uri: str) -> bytes:
            ...
    """
    if max_attempts < 1:
        raise DataConfigError(
            "with_retry: max_attempts must be >= 1",
            context={"max_attempts": max_attempts},
        )
    if base_delay <= 0:
        raise DataConfigError(
            "with_retry: base_delay must be > 0",
            context={"base_delay": base_delay},
        )

    import random  # local import — only needed when decorator is applied

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exc: Optional[BaseException] = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except retryable_exceptions as exc:
                    last_exc = exc
                    if attempt == max_attempts:
                        break
                    raw_delay = min(
                        base_delay * (backoff_factor ** (attempt - 1)),
                        max_delay,
                    )
                    jitter_delta = raw_delay * jitter * (2 * random.random() - 1)
                    sleep_for = max(0.0, raw_delay + jitter_delta)
                    logger.warning({
                        "event": "retry",
                        "function": fn.__qualname__,
                        "attempt": attempt,
                        "max_attempts": max_attempts,
                        "sleep_seconds": round(sleep_for, 3),
                        "error": str(exc),
                    })
                    time.sleep(sleep_for)

            # Re-raise the last exception, preserving its original traceback
            raise last_exc  # type: ignore[misc]

        return wrapper

    return decorator


@contextmanager
def timed(label: str, *,
    warn_threshold_seconds: Optional[float] = None) -> Generator[None, None, None]:
    """Context manager that logs the wall-clock duration of the enclosed block.

    Parameters
    ----------
    label:
        Human-readable label included in the log line.
    warn_threshold_seconds:
        If the block takes longer than this, emit a WARNING instead of DEBUG.

    Example
    -------
    ::

        with timed("parquet_read", warn_threshold_seconds=5.0):
            df = pd.read_parquet(path)
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        payload = {
            "event": "timed_block",
            "label": label,
            "elapsed_seconds": round(elapsed, 6),
        }
        if warn_threshold_seconds is not None and elapsed > warn_threshold_seconds:
            payload["warn_threshold_seconds"] = warn_threshold_seconds
            logger.warning(payload)
        else:
            logger.debug(payload)
