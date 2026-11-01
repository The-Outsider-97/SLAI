
import hashlib
import json

from typing import Any, Mapping, Sequence
from datetime import datetime, timezone
from datetime import datetime

from .functions_error import *
from logs.logger import PrettyPrinter, get_logger # pyright: ignore[reportMissingImports]

logger = get_logger("Functions Helpers")
printer = PrettyPrinter()


def _utc_now() -> datetime:
    """Return the current timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


def _isoformat_utc(value: datetime) -> str:
    """Serialize a timezone-aware datetime in canonical UTC form."""
    if (not isinstance(value, datetime) or value.tzinfo is None
            or value.utcoffset() is None):
        raise WebhookConfigurationError(
            "created_at must be a timezone-aware datetime")

    return (value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"))


def _derived_idempotency_key(base: str, suffix: str) -> str:
    candidate = f"{base}:{suffix}"
    if len(candidate) <= 200:
        return candidate
    digest = hashlib.sha256(candidate.encode("utf-8")).hexdigest()
    return f"derived:{digest}"


def _fingerprint(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _quote_identifier(identifier: str) -> str:
    return f'"{identifier}"'


def _query_fingerprint(sql: str, params: Sequence[Any]) -> str:
    # Values are included in the local fingerprint but neither SQL nor values
    # are logged. This supports correlation without leaking query content.
    payload = json.dumps(
        {
            "sql": sql,
            "params": [repr(value) for value in params],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


__all__ = [
    "_utc_now",
    "_isoformat_utc",
    "_derived_idempotency_key",
    "_fingerprint",
    "_quote_identifier",
    "_query_fingerprint",
]