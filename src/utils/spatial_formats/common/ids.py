"""Creation and validation of immutable spatial identifiers.

Identifiers are canonical lowercase hyphenated UUID strings, ≤ 64 bytes.
The nil UUID (``0000…``) is syntactically valid; applications may reject
it at a higher layer where a sentinel would be unsafe.
"""

import os
import time
import uuid as _uuid

from uuid import UUID, uuid4
from .errors import ValidationError, ResourceLimitError

NIL_ID = "00000000-0000-0000-0000-000000000000"
_MAX_ID_STRING = 64


def _uuid7() -> str:
    """Best-effort UUIDv7; monotonicity is not guaranteed across processes."""
    timestamp_ms = int(time.time() * 1000) & 0xFFFF_FFFF_FFFF
    rand_a = int.from_bytes(os.urandom(2), "big") & 0x0FFF
    rand_b = int.from_bytes(os.urandom(8), "big") & 0x3FFF_FFFF_FFFF_FFFF
    value = (timestamp_ms << 80) | (0x7 << 76) | (rand_a << 64) | (0x2 << 62) | rand_b
    return str(UUID(int=value))


def new_id(*, version: int = 4) -> str:
    """Generate a canonical UUID string.

    ``version=4`` (default) is random; ``version=7`` is time-ordered when
    the running Python exposes :func:`uuid.uuid7`, otherwise a compatible
    local implementation is used.
    """
    if version == 4:
        return str(uuid4())
    if version == 7:
        factory = getattr(_uuid, "uuid7", None)
        return str(factory()) if factory else _uuid7()
    raise ValueError(f"unsupported UUID version: {version}")


def is_valid_id(value) -> bool:
    """Return ``True`` when :func:`require_id` would accept ``value``."""
    try:
        require_id(value)
    except ValidationError:
        return False
    return True


def require_id(value: str) -> str:
    if not isinstance(value, str) or len(value) > _MAX_ID_STRING:
        raise ValidationError(
            "object ID must be a UUID string",
            context={"length": len(value) if isinstance(value, str) else None},
        )
    try:
        parsed = UUID(value)
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValidationError("invalid UUID", context={"value": value}) from exc
    if str(parsed) != value.lower():
        raise ValidationError(
            "UUID must use canonical hyphenated notation",
            context={"value": value},
        )
    return str(parsed)


def require_id_or_none(value):
    """Validate ``value`` as a UUID or ``None``; used for optional parents."""
    if value is None:
        return None
    return require_id(value)


def require_id_list(
    values,
    *,
    max_items: int = 100_000,
    allow_empty: bool = True,
) -> list[str]:
    """Validate a bounded array of UUID strings, returning a detached list."""
    if not isinstance(values, (list, tuple)):
        raise ValidationError("ID list must be an array")
    if not values and not allow_empty:
        raise ValidationError("ID list must not be empty")
    if len(values) > max_items:
        raise ResourceLimitError(
            "ID list exceeds budget",
            context={"length": len(values), "limit": max_items},
        )
    return [require_id(v) for v in values]


def coerce_id(value) -> str:
    """Accept ``UUID`` objects or strings and return the canonical string."""
    if isinstance(value, UUID):
        return str(value)
    return require_id(value)


def same_id(left, right) -> bool:
    """Case-insensitive UUID comparison without re-parsing into objects."""
    try:
        return require_id(left) == require_id(right)
    except ValidationError:
        return False


def assert_unique(values) -> None:
    """Raise :class:`ValidationError` if ``values`` contains a duplicate ID.

    Non-iterable input is rejected with the same error class so callers do
    not need a separate ``TypeError`` branch.
    """
    try:
        items = list(values)
    except TypeError as exc:
        raise ValidationError("values must be iterable") from exc
    seen: set[str] = set()
    for raw in items:
        ident = require_id(raw)
        if ident in seen:
            raise ValidationError(
                "duplicate object identifier",
                context={"id": ident},
            )
        seen.add(ident)


class DeterministicGenerator:
    """Sequential UUID factory for reproducible fixtures and tests.

    **Never use in production.** IDs are predictable and monotonic. The
    default ``seed=0`` yields the nil UUID for the first call; pass
    ``seed=1`` when the nil UUID is reserved.
    """

    __slots__ = ("_next",)

    def __init__(self, seed: int = 0):
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        self._next = seed

    def __call__(self) -> str:
        value = self._next
        self._next += 1
        return str(UUID(int=value))


__all__ = [
    "NIL_ID",
    "new_id",
    "is_valid_id",
    "require_id",
    "require_id_or_none",
    "require_id_list",
    "coerce_id",
    "same_id",
    "assert_unique",
    "DeterministicGenerator",
]
