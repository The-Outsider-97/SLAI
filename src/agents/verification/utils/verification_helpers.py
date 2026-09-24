"""Level-0 helper utilities for the SLAI Verification subsystem.

The helpers are intentionally dependency-light and verification-specific.  They
may be imported by every Verification layer, but they never import ``formal``,
``model``, ``solving`` or level-2 subsystem modules.

Generic validation and deterministic fingerprinting are reused from SLAI's
existing Base/tuning utilities rather than reimplemented here.
"""

from __future__ import annotations

import math

from collections.abc import Callable, Iterable, Mapping
from typing import Hashable, Protocol, TypeVar, runtime_checkable

from ...base.utils.base_errors import (
    ensure_callable as base_ensure_callable,
    ensure_mapping as base_ensure_mapping,
    ensure_non_empty_string as base_ensure_non_empty_string,
)
from src.tuning.utils.tuning_helpers import stable_fingerprint # pyright: ignore[reportMissingImports]

from .verification_errors import MalformedSpecificationError, VerificationError


T = TypeVar("T")
THashable = TypeVar("THashable", bound=Hashable)
MetadataValue = str | int | float | bool | None


@runtime_checkable
class SupportsVerificationDict(Protocol):
    """Protocol for formal artifacts exposing deterministic dictionary data."""

    def to_dict(self) -> Mapping[str, object]:
        ...


def require_non_empty_text(
    value: object,
    field: str,
    *,
    error_cls: type[VerificationError] = MalformedSpecificationError,
) -> str:
    """Return a stripped non-empty string using SLAI Base validation semantics."""

    return base_ensure_non_empty_string(
        value,
        field,
        error_cls=error_cls,
        strip=True,
    )


def require_callable(
    value: object,
    field: str,
    *,
    error_cls: type[VerificationError] = MalformedSpecificationError,
) -> Callable[..., object]:
    """Require a callable without duplicating SLAI Base validation logic."""

    return base_ensure_callable(value, field, error_cls=error_cls)


def require_positive_int(
    value: object,
    field: str,
    *,
    error_cls: type[VerificationError] = MalformedSpecificationError,
) -> int:
    """Require a positive integer, explicitly rejecting ``bool``."""

    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise error_cls(
            f"{field} must be a positive integer",
            context={"field": field, "value": value},
        )
    return value


def require_non_negative_int(
    value: object,
    field: str,
    *,
    error_cls: type[VerificationError] = MalformedSpecificationError,
) -> int:
    """Require a non-negative integer, explicitly rejecting ``bool``."""

    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise error_cls(
            f"{field} must be a non-negative integer",
            context={"field": field, "value": value},
        )
    return value


def require_positive_float(
    value: object,
    field: str,
    *,
    error_cls: type[VerificationError] = MalformedSpecificationError,
) -> float:
    """Require a finite positive numeric value, explicitly rejecting ``bool``."""

    if isinstance(value, bool):
        raise error_cls(
            f"{field} must be a finite positive number",
            context={"field": field, "value": value},
        )
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as exc:
        raise error_cls(
            f"{field} must be a finite positive number",
            context={"field": field, "value": value},
            cause=exc,
        ) from exc
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise error_cls(
            f"{field} must be a finite positive number",
            context={"field": field, "value": value},
        )
    return parsed


def ensure_hashable(
    value: T,
    field: str,
    *,
    error_cls: type[VerificationError] = MalformedSpecificationError,
) -> T:
    """Require a hashable formal state/key and return the original value."""

    try:
        hash(value)
    except (TypeError, ValueError) as exc:
        raise error_cls(
            f"{field} must be hashable",
            context={"field": field, "type": type(value).__name__},
            cause=exc,
        ) from exc
    return value


def normalize_unique_hashables(
    values: Iterable[THashable],
    field: str,
    *,
    require_non_empty: bool = False,
    error_cls: type[VerificationError] = MalformedSpecificationError,
) -> tuple[THashable, ...]:
    """Materialize an ordered hashable sequence and reject duplicate members."""

    try:
        materialized = tuple(values)
    except TypeError as exc:
        raise error_cls(
            f"{field} must be iterable",
            context={"field": field},
            cause=exc,
        ) from exc

    if require_non_empty and not materialized:
        raise error_cls(f"{field} must not be empty", context={"field": field})

    seen: set[THashable] = set()
    for index, value in enumerate(materialized):
        ensure_hashable(value, f"{field}[{index}]", error_cls=error_cls)
        if value in seen:
            raise error_cls(
                f"{field} must contain unique values",
                context={"field": field, "duplicate_index": index, "value": safe_repr(value)},
            )
        seen.add(value)
    return materialized


def normalize_string_values(
    values: Iterable[object] | str | None,
    field: str,
    *,
    sort: bool = True,
    unique: bool = True,
    error_cls: type[VerificationError] = MalformedSpecificationError,
) -> tuple[str, ...]:
    """Normalize non-empty strings with optional deterministic de-duplication."""

    if values is None:
        return ()
    iterable: Iterable[object] = (values,) if isinstance(values, str) else values
    normalized: list[str] = []
    seen: set[str] = set()
    for index, raw in enumerate(iterable):
        text = require_non_empty_text(raw, f"{field}[{index}]", error_cls=error_cls)
        if unique and text in seen:
            continue
        seen.add(text)
        normalized.append(text)
    if sort:
        normalized.sort()
    return tuple(normalized)


def normalize_tags(tags: Iterable[str] | str | None) -> tuple[str, ...]:
    """Return unique, sorted, non-empty tags with deterministic ordering."""

    return normalize_string_values(tags, "verification tags", sort=True, unique=True)


def normalize_metadata(
    metadata: Mapping[object, object] | None,
    *,
    field: str = "metadata",
    error_cls: type[VerificationError] = MalformedSpecificationError,
) -> dict[str, MetadataValue]:
    """Normalize bounded scalar metadata used by formal structural artifacts.

    Metadata is descriptive only; arbitrary executable or nested objects are not
    accepted in formal transition artifacts.
    """

    if metadata is None:
        return {}
    mapping = base_ensure_mapping(metadata, field, error_cls=error_cls)
    normalized: dict[str, MetadataValue] = {}
    for raw_key, value in mapping.items():
        key = require_non_empty_text(raw_key, f"{field} key", error_cls=error_cls)
        if key in normalized:
            raise error_cls(
                f"{field} keys must be unique after normalization",
                context={"field": field, "key": key},
            )
        if value is not None and not isinstance(value, (str, int, float, bool)):
            raise error_cls(
                f"{field} values must be scalar (str/int/float/bool/null)",
                context={"field": field, "key": key, "type": type(value).__name__},
            )
        if isinstance(value, float) and not math.isfinite(value):
            raise error_cls(
                f"{field} floating-point values must be finite",
                context={"field": field, "key": key, "value": value},
            )
        normalized[key] = value
    return normalized


def safe_repr(value: object, *, max_length: int = 240) -> str:
    """Return a bounded diagnostic repr that does not mask verification errors."""

    max_length = require_positive_int(max_length, "max_length")
    try:
        rendered = repr(value)
    except Exception:
        rendered = f"<{type(value).__name__}:unrepresentable>"
    if len(rendered) <= max_length:
        return rendered
    if max_length <= 3:
        return rendered[:max_length]
    return f"{rendered[: max_length - 3]}..."


def artifact_fingerprint(payload: Mapping[str, object]) -> str:
    """Fingerprint deterministic verification metadata.

    Canonical JSON normalization and hashing are delegated to SLAI's existing
    tuning helper. Verification only defines when a fingerprint is semantically
    useful.
    """

    if not isinstance(payload, Mapping):
        raise MalformedSpecificationError(
            "artifact_fingerprint requires a mapping payload"
        )
    return stable_fingerprint(dict(payload))


def result_fingerprint(result: SupportsVerificationDict) -> str:
    """Fingerprint the formal evidence exposed by ``result.to_dict()``."""

    if not isinstance(result, SupportsVerificationDict):
        raise MalformedSpecificationError(
            "result_fingerprint requires an object implementing to_dict()"
        )
    payload = result.to_dict()
    if not isinstance(payload, Mapping):
        raise MalformedSpecificationError(
            "verification result to_dict() must return a mapping"
        )
    return artifact_fingerprint(payload)


__all__ = [
    "MetadataValue",
    "SupportsVerificationDict",
    "artifact_fingerprint",
    "ensure_hashable",
    "normalize_metadata",
    "normalize_string_values",
    "normalize_tags",
    "normalize_unique_hashables",
    "require_callable",
    "require_non_empty_text",
    "require_non_negative_int",
    "require_positive_float",
    "require_positive_int",
    "result_fingerprint",
    "safe_repr",
]
