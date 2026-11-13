"""Input validation shared by every SLAI application.

Every function raises :class:`~applications.tools.errors.BadRequest` with a message that names the field, so the client can show it
next to the input. Nothing here coerces silently: a wrong type is an error, not a guess.

Timestamps are exchanged as ISO-8601 with an explicit UTC offset and stored in UTC (``instant``). All stored instants have the
same shape (``YYYY-MM-DDTHH:MM:SS+00:00``), so comparing them as strings orders them in time.
"""
from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import Any, Iterable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .errors import BadRequest

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def utc_now() -> str:
    """The current time as a stored instant."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_id() -> str:
    """An opaque identifier: 32 lowercase hexadecimal characters. Used for records *and* for file names."""
    return uuid.uuid4().hex


_ID = re.compile(r"^[0-9a-f]{32}$")


def record_id(value: Any, label: str = "id") -> str:
    """A well-formed :func:`new_id` value. Anything else is rejected before it can reach a path or a query."""
    if not isinstance(value, str) or not _ID.match(value):
        raise BadRequest(f"{label} is not a valid identifier")
    return value


def text(value: Any, label: str, limit: int, *, empty: bool = False, multiline: bool = False) -> str:
    """A string of at most ``limit`` characters. Titles are stripped; ``empty=True`` keeps the text as given and allows ``""``."""
    if not isinstance(value, str):
        raise BadRequest(f"{label} must be text")
    if len(value) > limit:
        raise BadRequest(f"{label} must be at most {limit} characters")
    if _CONTROL.search(value) or (not multiline and ("\n" in value or "\r" in value)):
        raise BadRequest(f"{label} contains characters that are not allowed")
    if not empty and not value.strip():
        raise BadRequest(f"{label} must not be empty")
    return value if empty else value.strip()


def choice(value: Any, choices: Iterable[Any], label: str) -> Any:
    options = tuple(choices)
    # ``True == 1`` in Python: a boolean must never satisfy a numeric choice list.
    wrong_kind = isinstance(value, bool) and not any(isinstance(o, bool) for o in options)
    if wrong_kind or value not in options:
        raise BadRequest(f"Invalid {label}; expected {', '.join(str(c) for c in options)}")
    return value


def boolean(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise BadRequest(f"{label} must be true or false")
    return value


def integer(value: Any, label: str, low: int, high: int, *, nullable: bool = False) -> int | None:
    if value is None and nullable:
        return None
    if type(value) is not int or not low <= value <= high:
        raise BadRequest(f"{label} must be a whole number from {low} to {high}")
    return value


def number(value: Any, label: str, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != value or not low <= value <= high:
        raise BadRequest(f"{label} must be a number from {low} to {high}")
    return float(value)


def instant(value: Any, label: str, *, nullable: bool = False) -> str | None:
    """Parse an ISO-8601 timestamp that carries a UTC offset and return it in UTC (see the module note)."""
    if value is None and nullable:
        return None
    if not isinstance(value, str) or len(value) > 64:
        raise BadRequest(f"{label} requires an ISO-8601 timestamp with a UTC offset")
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise BadRequest(f"Invalid {label} timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise BadRequest(f"{label} must include a UTC offset")
    try:
        return parsed.astimezone(timezone.utc).isoformat(timespec="seconds")
    except (OverflowError, ValueError) as exc:
        raise BadRequest(f"{label} is out of range") from exc


def parse_instant(stored: str) -> datetime:
    """A stored instant as an aware ``datetime``."""
    return datetime.fromisoformat(stored)


def zone(value: Any, label: str = "time_zone") -> str:
    """An IANA time-zone name. Windows needs the ``tzdata`` package for these."""
    name = text(value, label, 100)
    if name.startswith("/") or ".." in name:
        raise BadRequest(f"{label} is not a known IANA time zone, for example Europe/Amsterdam.")
    try:
        ZoneInfo(name)
    except (ValueError, ZoneInfoNotFoundError, OSError) as exc:
        raise BadRequest(f"{label} is not a known IANA time zone, for example Europe/Amsterdam.") from exc
    return name


def like_pattern(query: str) -> str:
    """Escape ``%``, ``_`` and the escape character itself for a SQL ``LIKE ... ESCAPE '\\'`` match."""
    return "%" + query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
