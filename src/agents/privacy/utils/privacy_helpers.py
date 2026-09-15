"""Shared, production-grade helpers for SLAI's Privacy subsystem.

This module is the reusable utility boundary for ``src.agents.privacy`` and
``src.agents.privacy_agent``.  It centralizes deterministic, cross-cutting
mechanics that were otherwise easy to duplicate across privacy components:

* identifiers and timestamps;
* scalar, identity, mapping, sequence, and timestamp normalization;
* deterministic fingerprints and pseudonymous tokens;
* privacy-safe masking and non-disclosing previews;
* nested field-path matching and bounded payload traversal;
* bounded history utilities;
* privacy-decision aggregation;
* context sanitization and exception normalization; and
* defensive shared-memory publication.

Ownership boundaries
-------------------
The helper layer intentionally does *not* own privacy policy or state.  PII/PHI
classification belongs to ``data_id.py``; minimization policy belongs to
``data_minimization.py``; consent and purpose limitation belong to
``data_consent.py``; retention/deletion policy belongs to ``data_retention.py``;
state belongs to ``privacy_memory.py``; and audit evidence/workflows belong to
``privacy_auditability.py``.

Circular-import safety
----------------------
``privacy_error.py`` remains the owner of the privacy error taxonomy,
``PrivacyDecision``, context sanitization, and exception normalization.  This
module imports those symbols *lazily inside the few functions that require
them*.  Consequently ``privacy_error.py`` may remain independent of this helper
module without an import-time cycle.

Security note
-------------
A deterministic digest is an integrity/fingerprinting primitive, not
anonymization.  For stable pseudonymization of personal data, use
``pseudonymous_token`` with a secret key (HMAC-SHA-256) rather than an unkeyed
fingerprint.
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import math
import re
import time
import uuid

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from ...base.utils.base_helpers import *


MODULE_VERSION = "2.3.0"
HELPER_SCHEMA_VERSION = "privacy_helpers.v1"

DEFAULT_FINGERPRINT_LENGTH = 16
DEFAULT_IDENTIFIER_HEX_LENGTH = 16
DEFAULT_MAX_COLLECTION_ITEMS = 500
DEFAULT_MAX_PAYLOAD_DEPTH = 12
DEFAULT_MAX_MAPPING_FIELDS = 500
DEFAULT_PREVIEW_FINGERPRINT_LENGTH = 16

_DECISION_PRECEDENCE: Dict[str, int] = {
    "allow": 0,
    "modify": 10,
    "escalate": 20,
    "block": 30,
}

_FIELD_NORMALIZATION_RE = re.compile(r"[-\s]+")
_PREFIX_SANITIZATION_RE = re.compile(r"[^A-Za-z0-9_-]+")
_PATH_INDEX_RE = re.compile(r"\[[^\]]*\]")

OverflowPolicy = Literal["raise", "truncate"]


@dataclass(frozen=True, slots=True)
class PayloadLeaf:
    """A single scalar leaf discovered during bounded payload traversal.

    The value is retained for in-process classification/minimization only.  A
    caller must not log or serialize ``value`` without the privacy controls
    appropriate to that data.
    """

    path: str
    value: Any
    depth: int
    parent_type: str


# ---------------------------------------------------------------------------
# Time and identifiers
# ---------------------------------------------------------------------------
def epoch_seconds() -> float:
    """Return the current wall-clock Unix timestamp in seconds."""

    return time.time()


def utc_iso(value: Optional[Union[datetime, float, int]] = None) -> str:
    """Return an aware UTC ISO-8601 timestamp using the ``Z`` suffix.

    ``value`` may be an aware/naive ``datetime`` or Unix timestamp.  Naive
    datetimes are treated as UTC rather than local time so behavior is stable
    across deployment environments.
    """

    if value is None:
        dt = utc_now()
    elif isinstance(value, datetime):
        dt = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
        dt = dt.astimezone(timezone.utc)
    else:
        numeric = require_finite_float(value, "timestamp", minimum=0.0)
        dt = datetime.fromtimestamp(numeric, tz=timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")


def new_privacy_ref(
    prefix: Any = "privacy",
    *,
    hex_length: int = DEFAULT_IDENTIFIER_HEX_LENGTH,
) -> str:
    """Create a compact collision-resistant privacy reference.

    The default 16 hexadecimal UUID characters preserve the identifier shape
    already used by the Privacy Agent, PrivacyMemory, and PrivacyAuditability.
    ``hex_length`` is restricted to the 8..32 range to prevent accidentally
    weak or nonsensically large identifiers.
    """

    length = require_integer(hex_length, "hex_length", minimum=8, maximum=32)
    raw_prefix = str(prefix or "privacy").strip()
    normalized_prefix = _PREFIX_SANITIZATION_RE.sub("-", raw_prefix).strip("-_")
    if not normalized_prefix:
        normalized_prefix = "privacy"
    return f"{normalized_prefix}-{uuid.uuid4().hex[:length]}"


# ---------------------------------------------------------------------------
# Validation and normalization
# ---------------------------------------------------------------------------
def nonempty_or(value: Any, fallback: Any) -> str:
    """Return a stripped non-empty string or a required fallback string."""

    normalized = str(value or "").strip()
    if normalized:
        return normalized
    fallback_value = str(fallback or "").strip()
    if not fallback_value:
        raise ValueError("fallback must resolve to a non-empty string")
    return fallback_value


def normalize_identity(value: Any, field_name: str = "value") -> str:
    """Normalize and require a non-empty identity/reference string."""

    normalized = str(value).strip() if value is not None else ""
    if not normalized:
        raise ValueError(f"'{field_name}' must be a non-empty string")
    return normalized


def normalize_optional_identity(value: Any) -> Optional[str]:
    """Return a stripped optional identity, converting empty values to ``None``."""

    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def resolve_policy_version(value: Any, default: Any = "v1") -> str:
    """Resolve an optional policy version to a required non-empty value."""

    return nonempty_or(value, default)


def normalize_field_name(value: Any) -> str:
    """Normalize a field/path token for deterministic policy matching."""

    text = normalize_text(value, lowercase=True, unicode_form="NFKC")
    return _FIELD_NORMALIZATION_RE.sub("_", text).strip("_")


def require_integer(
    value: Any,
    field_name: str,
    *,
    minimum: Optional[int] = None,
    maximum: Optional[int] = None,
) -> int:
    """Require an integer without silently clamping invalid configuration."""

    if isinstance(value, bool):
        raise ValueError(f"'{field_name}' must be an integer, not a boolean")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"'{field_name}' must be an integer") from exc

    # Reject lossy coercion such as 3.5 -> 3.
    if isinstance(value, float) and (not math.isfinite(value) or not value.is_integer()):
        raise ValueError(f"'{field_name}' must be an integer")
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or not re.fullmatch(r"[+-]?\d+", stripped):
            raise ValueError(f"'{field_name}' must be an integer")

    if minimum is not None and result < minimum:
        raise ValueError(f"'{field_name}' must be >= {minimum}, received {result!r}")
    if maximum is not None and result > maximum:
        raise ValueError(f"'{field_name}' must be <= {maximum}, received {result!r}")
    return result


def require_finite_float(
    value: Any,
    field_name: str,
    *,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> float:
    """Require a finite numeric value with optional inclusive bounds."""

    if isinstance(value, bool):
        raise ValueError(f"'{field_name}' must be numeric, not a boolean")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"'{field_name}' must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"'{field_name}' must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"'{field_name}' must be >= {minimum}, received {result!r}")
    if maximum is not None and result > maximum:
        raise ValueError(f"'{field_name}' must be <= {maximum}, received {result!r}")
    return result


def require_probability(value: Any, field_name: str = "probability") -> float:
    """Require a normalized probability-style scalar in the inclusive [0, 1] range."""

    return require_finite_float(value, field_name, minimum=0.0, maximum=1.0)


def normalize_optional_timestamp(value: Any, field_name: str = "timestamp") -> Optional[float]:
    """Normalize an optional Unix timestamp and reject negative/non-finite values."""

    if value is None:
        return None
    return require_finite_float(value, field_name, minimum=0.0)


def normalize_string_sequence(
    values: Optional[Sequence[Any]],
    *,
    field_name: str = "values",
    max_items: Optional[int] = None,
    overflow: OverflowPolicy = "raise",
    casefold: bool = False,
) -> List[str]:
    """Normalize, de-duplicate, and bound a string sequence while preserving order.

    ``overflow='raise'`` is the correct default for privacy enforcement because
    silent truncation can discard an authorization or restriction.  Callers
    that deliberately use best-effort display/metadata lists may select
    ``'truncate'`` explicitly.
    """

    if overflow not in {"raise", "truncate"}:
        raise ValueError("overflow must be either 'raise' or 'truncate'")
    if values is None:
        return []
    if isinstance(values, (str, bytes, bytearray)):
        raw_values: Sequence[Any] = [values]
    elif isinstance(values, Sequence):
        raw_values = values
    else:
        raise TypeError(f"'{field_name}' must be a sequence or None")

    limit = None
    if max_items is not None:
        limit = require_integer(max_items, "max_items", minimum=0)

    normalized: List[str] = []
    seen: set[str] = set()
    for raw in raw_values:
        item = str(raw).strip()
        if not item:
            continue
        output = item.casefold() if casefold else item
        dedupe_key = output
        if dedupe_key in seen:
            continue
        if limit is not None and len(normalized) >= limit:
            if overflow == "raise":
                raise ValueError(f"'{field_name}' exceeds the configured limit of {limit}")
            break
        normalized.append(output)
        seen.add(dedupe_key)
    return normalized


def normalize_mapping(
    value: Optional[Mapping[Any, Any]],
    *,
    field_name: str = "mapping",
    max_fields: Optional[int] = None,
    deep_copy: bool = True,
) -> Dict[str, Any]:
    """Return a defensively copied string-key mapping with an optional size bound."""

    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"'{field_name}' must be a mapping or None")
    if max_fields is not None:
        limit = require_integer(max_fields, "max_fields", minimum=0)
        if len(value) > limit:
            raise ValueError(f"'{field_name}' exceeds the configured limit of {limit} fields")
    converted: Dict[str, Any] = {}
    for key, item in value.items():
        normalized_key = str(key)
        if normalized_key in converted:
            raise ValueError(
                f"'{field_name}' contains keys that collide after string normalization: "
                f"{normalized_key!r}"
            )
        converted[normalized_key] = item
    return copy.deepcopy(converted) if deep_copy else converted


def safe_mapping(value: Optional[Mapping[Any, Any]]) -> Dict[str, Any]:
    """Compatibility-oriented mapping helper for the top-level Privacy Agent."""

    return normalize_mapping(value, field_name="mapping", deep_copy=False)


# ---------------------------------------------------------------------------
# Copying, bounded history, and context
# ---------------------------------------------------------------------------
def deep_copy(value: Any) -> Any:
    """Return a defensive deep copy.

    A copy failure is intentionally propagated.  Privacy state should not fall
    back to sharing a mutable reference silently.
    """

    return copy.deepcopy(value)


def trim_history(history: Sequence[Any], limit: int, *, copy_items: bool = False) -> List[Any]:
    """Return the newest ``limit`` entries without mutating the input sequence."""

    normalized_limit = require_integer(limit, "limit", minimum=1)
    result = list(history[-normalized_limit:])
    return copy.deepcopy(result) if copy_items else result


def sanitize_privacy_mapping(
    payload: Optional[Mapping[Any, Any]],
    *,
    enabled: bool = True,
    max_fields: Optional[int] = None,
) -> Dict[str, Any]:
    """Defensively copy and optionally sanitize free-form privacy context.

    ``privacy_error.sanitize_privacy_context`` is imported lazily so this module
    can never create an import-time cycle with the error taxonomy.
    """

    normalized = normalize_mapping(
        payload,
        field_name="privacy_context",
        max_fields=max_fields,
        deep_copy=True,
    )
    if not enabled or not normalized:
        return normalized

    from .privacy_error import sanitize_privacy_context

    sanitized = sanitize_privacy_context(normalized)
    if not isinstance(sanitized, Mapping):
        raise TypeError("sanitize_privacy_context must return a mapping")
    return {str(key): copy.deepcopy(item) for key, item in sanitized.items()}


def merge_privacy_context(
    *mappings: Optional[Mapping[Any, Any]],
    sanitize: bool = True,
    max_fields: Optional[int] = None,
) -> Dict[str, Any]:
    """Merge context mappings left-to-right and optionally sanitize the result."""

    merged: Dict[str, Any] = {}
    for mapping in mappings:
        if mapping is None:
            continue
        if not isinstance(mapping, Mapping):
            raise TypeError("all privacy context values must be mappings or None")
        for key, value in mapping.items():
            merged[str(key)] = copy.deepcopy(value)

    if max_fields is not None:
        limit = require_integer(max_fields, "max_fields", minimum=0)
        if len(merged) > limit:
            raise ValueError(f"merged privacy context exceeds the configured limit of {limit} fields")

    return sanitize_privacy_mapping(merged, enabled=sanitize)


def normalize_privacy_error(
    exc: Exception,
    *,
    stage: str,
    context: Optional[Mapping[Any, Any]] = None,
) -> Exception:
    """Normalize a generic exception into the canonical privacy error model.

    Existing ``PrivacyError`` instances pass through unchanged so their type,
    decision, retryability, incident ID, and remediation metadata are preserved.
    """

    if not isinstance(exc, Exception):
        raise TypeError("exc must be an Exception instance")
    normalized_stage = normalize_identity(stage, "stage")

    from .privacy_error import PrivacyError, normalize_privacy_exception

    if isinstance(exc, PrivacyError):
        return exc
    return normalize_privacy_exception(
        exc,
        stage=normalized_stage,
        context=sanitize_privacy_mapping(context),
    )


# ---------------------------------------------------------------------------
# Deterministic fingerprints, tokens, and previews
# ---------------------------------------------------------------------------
def stable_privacy_fingerprint(
    value: Any,
    *,
    length: int = DEFAULT_FINGERPRINT_LENGTH,
    algorithm: str = "sha256",
) -> str:
    """Return a deterministic integrity fingerprint for arbitrary structured data.

    This function intentionally delegates canonical serialization/hashing to the
    Base helper layer rather than maintaining another implementation in Privacy.
    It must not be treated as anonymization of low-entropy personal data.
    """

    normalized_length = require_integer(length, "length", minimum=1, maximum=128)
    return stable_fingerprint(value, algorithm=algorithm, length=normalized_length)


def _canonical_json_bytes(value: Any) -> bytes:
    safe = to_json_safe(value)
    text = json.dumps(
        safe,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return text.encode("utf-8")


def pseudonymous_token(
    value: Any,
    *,
    key: Union[str, bytes, bytearray],
    prefix: str = "tok",
    digest_length: int = DEFAULT_FINGERPRINT_LENGTH,
) -> str:
    """Return a deterministic HMAC-SHA-256 pseudonym for ``value``.

    The HMAC key is deliberately mandatory.  A keyed pseudonym resists trivial
    dictionary reversal significantly better than a plain hash, but it remains
    personal data whenever re-identification is possible through possession of
    the key or auxiliary information.
    """

    if isinstance(key, str):
        key_bytes = key.encode("utf-8")
    elif isinstance(key, (bytes, bytearray)):
        key_bytes = bytes(key)
    else:
        raise TypeError("key must be str, bytes, or bytearray")
    if not key_bytes:
        raise ValueError("key must not be empty")

    length = require_integer(digest_length, "digest_length", minimum=8, maximum=64)
    normalized_prefix = normalize_field_name(prefix) or "tok"
    digest = hmac.new(key_bytes, _canonical_json_bytes(value), hashlib.sha256).hexdigest()
    return f"{normalized_prefix}_{digest[:length]}"


def stable_token(
    value: Any,
    *,
    prefix: str = "tok",
    key: Optional[Union[str, bytes, bytearray]] = None,
    length: int = DEFAULT_FINGERPRINT_LENGTH,
) -> str:
    """Return a stable token, using keyed HMAC when ``key`` is supplied.

    The unkeyed branch exists for backward compatibility with the Privacy
    subsystem's current deterministic token shape.  New pseudonymization use
    cases should pass a secret key or call ``pseudonymous_token`` explicitly.
    """

    normalized_prefix = normalize_field_name(prefix) or "tok"
    if key is not None:
        return pseudonymous_token(
            value,
            key=key,
            prefix=normalized_prefix,
            digest_length=length,
        )
    return f"{normalized_prefix}_{stable_privacy_fingerprint(value, length=length)}"


def privacy_safe_preview(
    value: Any,
    *,
    fingerprint_length: int = DEFAULT_PREVIEW_FINGERPRINT_LENGTH,
) -> str:
    """Return non-disclosing diagnostic metadata instead of raw payload text."""

    safe = to_json_safe(value)
    try:
        serialized = json.dumps(
            safe,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        size = len(serialized.encode("utf-8"))
    except (TypeError, ValueError):
        size = -1
    fingerprint = stable_privacy_fingerprint(value, length=fingerprint_length)
    return f"<{type(value).__name__} sha256:{fingerprint} bytes:{size}>"


# ---------------------------------------------------------------------------
# Privacy-safe masking
# ---------------------------------------------------------------------------
def partial_mask(
    value: Any,
    *,
    visible_prefix: int = 2,
    visible_suffix: int = 2,
    mask_char: str = "*",
) -> str:
    """Mask the middle of a scalar string while preserving configured edges."""

    text = str(value)
    prefix_len = require_integer(visible_prefix, "visible_prefix", minimum=0)
    suffix_len = require_integer(visible_suffix, "visible_suffix", minimum=0)
    if len(mask_char) != 1:
        raise ValueError("mask_char must contain exactly one character")
    if not text:
        return text
    if len(text) <= prefix_len + suffix_len:
        return mask_char * len(text)
    prefix = text[:prefix_len] if prefix_len else ""
    suffix = text[-suffix_len:] if suffix_len else ""
    hidden_count = max(1, len(text) - len(prefix) - len(suffix))
    return f"{prefix}{mask_char * hidden_count}{suffix}"


def mask_email(
    value: Any,
    *,
    visible_prefix: int = 1,
    visible_suffix: int = 0,
    preserve_domain: bool = True,
    mask_char: str = "*",
) -> str:
    """Mask an email-like string without attempting email-address validation."""

    text = str(value).strip()
    if "@" not in text:
        return partial_mask(
            text,
            visible_prefix=visible_prefix,
            visible_suffix=visible_suffix,
            mask_char=mask_char,
        )
    local, domain = text.rsplit("@", 1)
    masked_local = partial_mask(
        local,
        visible_prefix=visible_prefix,
        visible_suffix=visible_suffix,
        mask_char=mask_char,
    )
    if preserve_domain:
        return f"{masked_local}@{domain}"
    return partial_mask(text, visible_prefix=visible_prefix, visible_suffix=0, mask_char=mask_char)


def mask_last4(value: Any, *, mask_char: str = "*") -> str:
    """Mask all but the last four characters; short values are fully masked."""

    text = str(value)
    if len(mask_char) != 1:
        raise ValueError("mask_char must contain exactly one character")
    if len(text) <= 4:
        return mask_char * len(text)
    return f"{mask_char * (len(text) - 4)}{text[-4:]}"


# ---------------------------------------------------------------------------
# Nested field-path helpers
# ---------------------------------------------------------------------------
def strip_path_indices(path: Any) -> str:
    """Remove bracketed sequence indices from a field path."""

    return _PATH_INDEX_RE.sub("", str(path))


def path_variants(path: Any) -> Tuple[str, str, str]:
    """Return ``(normalized, indexless, leaf)`` path forms for policy matching."""

    normalized = normalize_field_name(path)
    indexless = normalize_field_name(strip_path_indices(path))
    leaf = indexless.rsplit(".", 1)[-1] if indexless else ""
    return normalized, indexless, leaf


def pattern_matches_path(pattern: Any, path: Any) -> bool:
    """Match a Privacy field pattern against a nested payload path.

    Supported forms mirror the existing minimization semantics:
    ``*``/``**`` match any path, ``parent.*`` matches the parent and descendants,
    a bare leaf such as ``email`` matches that leaf at any depth, and a dotted
    pattern matches itself and descendants.
    """

    normalized_pattern = normalize_field_name(pattern)
    normalized_path, indexless, leaf = path_variants(path)
    if not normalized_pattern:
        return False
    if normalized_pattern in {"*", "**"}:
        return True
    if normalized_pattern.endswith(".*"):
        prefix = normalized_pattern[:-2]
        return indexless == prefix or indexless.startswith(prefix + ".")
    if normalized_pattern in {normalized_path, indexless}:
        return True
    if "." not in normalized_pattern and normalized_pattern == leaf:
        return True
    return bool(indexless and indexless.startswith(normalized_pattern + "."))


def path_allowed(path: Any, patterns: Optional[Sequence[Any]]) -> bool:
    """Return whether ``path`` is allowed by a pattern list.

    An empty pattern list means unconstrained/allowed, preserving the current
    minimization behavior.
    """

    if not patterns:
        return True
    return any(pattern_matches_path(pattern, path) for pattern in patterns)


def path_has_descendant(path: Any, patterns: Optional[Sequence[Any]]) -> bool:
    """Return whether at least one allowed pattern may exist below ``path``."""

    if not patterns:
        return True
    _, indexless, _ = path_variants(path)
    prefix = f"{indexless}." if indexless else ""

    for pattern in patterns:
        normalized_pattern = normalize_field_name(pattern)
        if normalized_pattern in {"*", "**"}:
            return True
        if normalized_pattern.endswith(".*"):
            normalized_pattern = normalized_pattern[:-2]
        if normalized_pattern == indexless:
            return True
        if prefix and normalized_pattern.startswith(prefix):
            return True
        if (
            "." not in normalized_pattern
            and indexless
            and normalized_pattern == indexless.rsplit(".", 1)[-1]
        ):
            return True
    return False


def join_payload_path(parent: str, child: Any, *, is_index: bool = False) -> str:
    """Join a mapping key or sequence index to a nested payload path."""

    if is_index:
        return f"{parent}[{child}]" if parent else f"[{child}]"
    child_text = str(child)
    return f"{parent}.{child_text}" if parent else child_text


def iter_payload_leaves(
    payload: Any,
    *,
    root_path: str = "",
    max_depth: int = DEFAULT_MAX_PAYLOAD_DEPTH,
    max_items_per_container: int = DEFAULT_MAX_COLLECTION_ITEMS,
    overflow: OverflowPolicy = "raise",
) -> Iterator[PayloadLeaf]:
    """Yield scalar leaves from nested mappings/sequences with explicit bounds.

    Bounds are privacy controls, not only performance controls: unbounded nested
    input can exhaust resources, while silent truncation can hide sensitive
    fields from classifiers.  Consequently the default overflow behavior is to
    raise.  A caller may opt into truncation only when incomplete traversal is
    explicitly acceptable.
    """

    if overflow not in {"raise", "truncate"}:
        raise ValueError("overflow must be either 'raise' or 'truncate'")
    depth_limit = require_integer(max_depth, "max_depth", minimum=0)
    item_limit = require_integer(
        max_items_per_container,
        "max_items_per_container",
        minimum=1,
    )

    def _walk(value: Any, path: str, depth: int, parent_type: str) -> Iterator[PayloadLeaf]:
        if depth > depth_limit:
            if overflow == "raise":
                raise ValueError(
                    f"payload traversal exceeded max_depth={depth_limit} at path {path!r}"
                )
            return

        if isinstance(value, Mapping):
            items = sorted(value.items(), key=lambda item: str(item[0]))
            if len(items) > item_limit:
                if overflow == "raise":
                    raise ValueError(
                        f"mapping at path {path!r} exceeds max_items_per_container={item_limit}"
                    )
                items = items[:item_limit]
            for key, child in items:
                child_path = join_payload_path(path, key)
                yield from _walk(child, child_path, depth + 1, "mapping")
            return

        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            total = len(value)
            if total > item_limit and overflow == "raise":
                raise ValueError(
                    f"sequence at path {path!r} exceeds max_items_per_container={item_limit}"
                )
            count = min(total, item_limit)
            for index in range(count):
                child_path = join_payload_path(path, index, is_index=True)
                yield from _walk(value[index], child_path, depth + 1, "sequence")
            return

        yield PayloadLeaf(path=path, value=value, depth=depth, parent_type=parent_type)

    yield from _walk(payload, root_path, 0, "root")


# ---------------------------------------------------------------------------
# Privacy decision helpers
# ---------------------------------------------------------------------------
def decision_value(value: Any, *, default: Optional[str] = None) -> str:
    """Normalize a privacy decision to its canonical lowercase string value."""

    from .privacy_error import PrivacyDecision

    if isinstance(value, PrivacyDecision):
        return value.value
    candidate = str(getattr(value, "value", value) or "").strip().lower()
    if candidate in _DECISION_PRECEDENCE:
        return candidate
    if default is not None:
        normalized_default = str(default).strip().lower()
        if normalized_default in _DECISION_PRECEDENCE:
            return normalized_default
    raise ValueError(
        f"Unsupported privacy decision {value!r}; expected one of {sorted(_DECISION_PRECEDENCE)}"
    )


def combine_privacy_decisions(*values: Any, default: Any = "allow") -> str:
    """Aggregate stage decisions using conservative, deterministic precedence.

    Precedence is ``block > escalate > modify > allow``.  A definitive block
    therefore cannot be accidentally weakened by a later successful stage.
    """

    if not values:
        return decision_value(default)
    normalized = [decision_value(value) for value in values if value is not None]
    if not normalized:
        return decision_value(default)
    return max(normalized, key=lambda item: _DECISION_PRECEDENCE[item])


def decision_requires_intervention(value: Any) -> bool:
    """Return whether a decision prevents transparent pass-through processing."""

    return decision_value(value) in {"modify", "escalate", "block"}


def decision_is_terminal(value: Any) -> bool:
    """Return whether a decision should stop ordinary downstream processing."""

    return decision_value(value) in {"escalate", "block"}


# ---------------------------------------------------------------------------
# Shared-memory publication helpers
# ---------------------------------------------------------------------------
def publish_shared_value(
    shared_memory: Any,
    key: str,
    value: Any,
    *,
    ttl_seconds: Optional[Any] = None,
    enabled: bool = True,
    best_effort: bool = False,
) -> bool:
    """Publish a defensive copy through a ``SharedMemory.set``-style interface.

    ``best_effort=False`` is intentional: privacy publication failures should be
    visible unless the caller has explicitly selected non-critical telemetry
    behavior.
    """

    if not enabled or shared_memory is None:
        return False
    normalized_key = normalize_identity(key, "key")
    setter = getattr(shared_memory, "set", None)
    if not callable(setter):
        if best_effort:
            return False
        raise TypeError("shared_memory must expose a callable 'set' method")

    ttl: Optional[int] = None
    if ttl_seconds is not None:
        ttl = require_integer(ttl_seconds, "ttl_seconds", minimum=0)

    payload = copy.deepcopy(value)
    try:
        if ttl is None:
            setter(normalized_key, payload)
        else:
            setter(normalized_key, payload, ttl=ttl)
        return True
    except Exception:
        if best_effort:
            return False
        raise


def publish_shared_event(
    shared_memory: Any,
    channel: str,
    payload: Any,
    *,
    enabled: bool = True,
    best_effort: bool = False,
) -> bool:
    """Publish a defensive copy through a ``SharedMemory.publish`` interface."""

    if not enabled or shared_memory is None:
        return False
    normalized_channel = normalize_identity(channel, "channel")
    publisher = getattr(shared_memory, "publish", None)
    if not callable(publisher):
        if best_effort:
            return False
        raise TypeError("shared_memory must expose a callable 'publish' method")

    try:
        publisher(normalized_channel, copy.deepcopy(payload))
        return True
    except Exception:
        if best_effort:
            return False
        raise


__all__ = [
    # Module metadata/constants
    "MODULE_VERSION",
    "HELPER_SCHEMA_VERSION",
    "DEFAULT_FINGERPRINT_LENGTH",
    "DEFAULT_IDENTIFIER_HEX_LENGTH",
    "DEFAULT_MAX_COLLECTION_ITEMS",
    "DEFAULT_MAX_PAYLOAD_DEPTH",
    "DEFAULT_MAX_MAPPING_FIELDS",
    "DEFAULT_PREVIEW_FINGERPRINT_LENGTH",
    # Foundational helpers re-exported from Base
    "Stopwatch",
    "coerce_bool",
    "coerce_float",
    "coerce_int",
    "monotonic_seconds",
    "utc_now",
    # Data model
    "PayloadLeaf",
    # Time/IDs
    "epoch_seconds",
    "utc_iso",
    "new_privacy_ref",
    # Validation/normalization
    "nonempty_or",
    "normalize_identity",
    "normalize_optional_identity",
    "resolve_policy_version",
    "normalize_field_name",
    "require_integer",
    "require_finite_float",
    "require_probability",
    "normalize_optional_timestamp",
    "normalize_string_sequence",
    "normalize_mapping",
    "safe_mapping",
    # Copy/history/context/errors
    "deep_copy",
    "trim_history",
    "sanitize_privacy_mapping",
    "merge_privacy_context",
    "normalize_privacy_error",
    # Fingerprints/tokens/previews
    "stable_privacy_fingerprint",
    "pseudonymous_token",
    "stable_token",
    "privacy_safe_preview",
    # Masking
    "partial_mask",
    "mask_email",
    "mask_last4",
    # Paths/traversal
    "strip_path_indices",
    "path_variants",
    "pattern_matches_path",
    "path_allowed",
    "path_has_descendant",
    "join_payload_path",
    "iter_payload_leaves",
    # Decisions
    "decision_value",
    "combine_privacy_decisions",
    "decision_requires_intervention",
    "decision_is_terminal",
    # Shared memory
    "publish_shared_value",
    "publish_shared_event",
]
