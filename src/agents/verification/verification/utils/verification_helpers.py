"""Level-0 helper utilities for the SLAI Verification subsystem.

This module is intentionally dependency-light. It may be imported by any
Verification layer, but it must not import ``formal``, ``model``, ``solving``
or level-2 subsystem modules. Generic serialization/fingerprinting mechanics
are reused from ``src.tuning.utils.tuning_helpers`` rather than reimplemented.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Protocol, runtime_checkable

from src.tuning.utils.tuning_helpers import stable_fingerprint
from .verification_errors import MalformedSpecificationError


@runtime_checkable
class SupportsVerificationDict(Protocol):
    """Protocol for formal artifacts that expose deterministic dictionary data."""

    def to_dict(self) -> Mapping[str, object]:
        ...


def normalize_tags(tags: Iterable[str] | str | None) -> tuple[str, ...]:
    """Return unique, sorted, non-empty tags with deterministic ordering."""

    if tags is None:
        return ()
    values = (tags,) if isinstance(tags, str) else tags
    normalized: set[str] = set()
    for raw in values:
        value = str(raw).strip()
        if not value:
            raise MalformedSpecificationError(
                "verification tags must be non-empty strings"
            )
        normalized.add(value)
    return tuple(sorted(normalized))


def artifact_fingerprint(payload: Mapping[str, object]) -> str:
    """Fingerprint deterministic verification metadata.

    Hashing and canonical JSON normalization are delegated to SLAI's existing
    tuning helpers. The Verification subsystem only defines when a fingerprint
    is semantically useful.
    """

    if not isinstance(payload, Mapping):
        raise MalformedSpecificationError(
            "artifact_fingerprint requires a mapping payload"
        )
    return stable_fingerprint(dict(payload))


def result_fingerprint(result: SupportsVerificationDict) -> str:
    """Fingerprint the formal evidence exposed by ``result.to_dict()``.

    The helper intentionally depends on a structural protocol instead of the
    level-0 VerificationResult class itself. This keeps helpers reusable and
    prevents import cycles inside the foundational layer.
    """

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
    "SupportsVerificationDict",
    "artifact_fingerprint",
    "normalize_tags",
    "result_fingerprint",
]
