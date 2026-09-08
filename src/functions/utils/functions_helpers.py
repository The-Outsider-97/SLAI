from datetime import datetime
from typing import Any, Dict, Optional
from datetime import datetime, timezone


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


__all__ = [
    "_utc_now",
    "_isoformat_utc",
]