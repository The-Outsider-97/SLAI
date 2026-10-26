"""Strict, deterministic and bounded JSON primitives.

All helpers reject booleans where a number is expected, reject ``NaN`` and
``Infinity``, enforce byte-count and structure budgets **before** allocating
large intermediate objects, and never rely on Python truthiness of
containers (an empty dict is not a Boolean).

Determinism
-----------
:func:`canonical_json` emits UTF-8 JSON with sorted object keys and no
whitespace. ``int`` and ``float`` values are preserved as distinct JSON
types (``1`` serialises as ``1``, ``1.0`` as ``1.0``); callers needing a
single numeric form should normalise before calling.
"""

import hashlib
import json
import math
from .errors import ValidationError, ResourceLimitError

MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_STRING = 1_000_000
MAX_NODES = 300_000
MAX_DEPTH = 64


# ---------------------------------------------------------------------------
# Scalar primitives
# ---------------------------------------------------------------------------

def finite(value, label: str = "number", *, limit: float = 1e12) -> float:
    """Return ``value`` as a finite float within ``±limit``.

    Booleans are rejected even though ``isinstance(True, int)`` is true.
    Integers larger than ``1e18`` short-circuit via ``abs(value) > limit``
    so :func:`math.isfinite` never overflows on huge Python ints.
    """
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or abs(value) > limit
        or not math.isfinite(value)
    ):
        raise ValidationError(
            label + " must be finite and within bounds",
            context={"label": label, "limit": limit},
        )
    return float(value)


def bounded_int(value, label: str = "integer", *, minimum=None, maximum=None) -> int:
    """Validate a strict (non-boolean) integer within inclusive bounds."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(label + " must be an integer")
    if minimum is not None and value < minimum:
        raise ValidationError(label + " must be at least " + str(minimum),
                              context={"label": label, "minimum": minimum})
    if maximum is not None and value > maximum:
        raise ValidationError(label + " must be at most " + str(maximum),
                              context={"label": label, "maximum": maximum})
    return value


def require_bool(value, label: str = "boolean") -> bool:
    if not isinstance(value, bool):
        raise ValidationError(label + " must be a boolean")
    return value


def require_str(
    value,
    label: str = "string",
    *,
    min_length: int = 0,
    max_length: int = MAX_STRING,
) -> str:
    if not isinstance(value, str):
        raise ValidationError(label + " must be a string")
    if len(value) < min_length:
        raise ValidationError(label + " is too short",
                              context={"length": len(value), "min_length": min_length})
    if len(value) > max_length:
        raise ResourceLimitError(label + " is too long",
                                 context={"length": len(value), "max_length": max_length})
    return value


def require_dict(value, label: str = "object") -> dict:
    if not isinstance(value, dict):
        raise ValidationError(label + " must be an object")
    return value


def require_list(
    value,
    label: str = "array",
    *,
    min_items: int = 0,
    max_items: int | None = None,
) -> list | tuple:
    if not isinstance(value, (list, tuple)):
        raise ValidationError(label + " must be an array")
    if len(value) < min_items:
        raise ValidationError(label + " has too few items",
                              context={"length": len(value), "min_items": min_items})
    if max_items is not None and len(value) > max_items:
        raise ResourceLimitError(label + " has too many items",
                                 context={"length": len(value), "max_items": max_items})
    return value


def vector(value, dimension: int, label: str = "vector") -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != dimension:
        raise ValidationError(
            label + " must have " + str(dimension) + " components",
            context={"expected": dimension},
        )
    return tuple(finite(v, label) for v in value)


# ---------------------------------------------------------------------------
# Structural validation and serialization
# ---------------------------------------------------------------------------

def json_safe(value, *, max_depth: int = MAX_DEPTH, max_nodes: int = MAX_NODES):
    """Return a detached copy of ``value`` restricted to plain JSON types.

    Budgets are consumed for each scalar, container and **object key**;
    a single deep or wide object therefore cannot exhaust the budget on
    keys alone. Tuples are accepted and become lists. ``bytes`` and any
    other non-JSON type are rejected.
    """
    nodes_left = max_nodes

    def walk(item, depth):
        nonlocal nodes_left
        nodes_left -= 1
        if nodes_left < 0:
            raise ResourceLimitError(
                "JSON structural budget exceeded",
                context={"max_nodes": max_nodes},
            )
        if depth > max_depth:
            raise ResourceLimitError(
                "JSON nesting budget exceeded",
                context={"max_depth": max_depth, "depth": depth},
            )
        if item is None or isinstance(item, bool):
            return item
        if isinstance(item, str):
            if len(item) > MAX_STRING:
                raise ResourceLimitError(
                    "string too long",
                    context={"length": len(item), "max": MAX_STRING},
                )
            return item
        if isinstance(item, (int, float)):
            finite(item, "JSON numeric value", limit=1e18)
            return item
        if isinstance(item, (list, tuple)):
            return [walk(x, depth + 1) for x in item]
        if isinstance(item, dict):
            # Keys count toward the node budget and are validated once.
            nodes_left -= len(item)
            if nodes_left < 0:
                raise ResourceLimitError(
                    "JSON object exceeds structural budget",
                    context={"keys": len(item)},
                )
            for k in item:
                if not isinstance(k, str) or len(k) > MAX_STRING:
                    raise ValidationError(
                        "JSON objects require bounded string keys",
                        context={"key_type": type(k).__name__},
                    )
            return {k: walk(v, depth + 1) for k, v in item.items()}
        raise ValidationError(
            "unsupported JSON value: " + type(item).__name__,
            context={"type": type(item).__name__},
        )

    return walk(value, 0)


def canonical_json(value) -> bytes:
    """Serialise ``value`` to deterministic UTF-8 JSON bytes.

    Uses sorted keys, no whitespace and ``allow_nan=False``. Raises
    :class:`ResourceLimitError` if the encoded result exceeds
    :data:`MAX_JSON_BYTES`.
    """
    safe = json_safe(value)
    raw = json.dumps(
        safe,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if len(raw) > MAX_JSON_BYTES:
        raise ResourceLimitError(
            "JSON member too large",
            context={"bytes": len(raw), "limit": MAX_JSON_BYTES},
        )
    return raw


def parse_json(raw: bytes):
    """Parse UTF-8 JSON bytes under all structural budgets.

    Rejects duplicate keys and non-standard numeric constants. A UTF-8 BOM
    is rejected (as required by RFC 8259) since ``json.loads`` does.
    """
    if not isinstance(raw, (bytes, bytearray)):
        raise TypeError("parse_json requires bytes")
    if len(raw) > MAX_JSON_BYTES:
        raise ResourceLimitError(
            "JSON member too large",
            context={"bytes": len(raw), "limit": MAX_JSON_BYTES},
        )

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValidationError(
                    "duplicate JSON key: " + key,
                    context={"key": key},
                )
            result[key] = value
        return result

    def invalid_number(token):
        raise ValidationError(
            "invalid JSON numeric token: " + token,
            context={"token": token},
        )

    try:
        text = bytes(raw).decode("utf-8") if isinstance(raw, bytearray) else raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValidationError("malformed UTF-8 JSON") from exc

    try:
        value = json.loads(text, object_pairs_hook=pairs, parse_constant=invalid_number)
    except RecursionError as exc:
        raise ResourceLimitError("JSON nesting exceeds parser budget") from exc
    except json.JSONDecodeError as exc:
        # ValidationError from object_pairs_hook / parse_constant is not a
        # JSONDecodeError subclass, so it propagates cleanly through here.
        raise ValidationError(
            f"malformed JSON at line {exc.lineno} column {exc.colno}: {exc.msg}",
            context={"line": exc.lineno, "column": exc.colno},
        ) from exc
    return json_safe(value)


def digest(data) -> str:
    """Return the lowercase hex SHA-256 of ``data`` (bytes-like)."""
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("digest requires bytes-like input")
    return hashlib.sha256(data).hexdigest()


__all__ = [
    "MAX_JSON_BYTES",
    "MAX_STRING",
    "MAX_NODES",
    "MAX_DEPTH",
    "finite",
    "bounded_int",
    "require_bool",
    "require_str",
    "require_dict",
    "require_list",
    "vector",
    "json_safe",
    "canonical_json",
    "parse_json",
    "digest",
]
