"""Shared utility layer for SLAI desktop apps."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable


def fmt_pct(value: float) -> str:
    """Format 0..1 float as percentage string."""
    return f"{value * 100:.0f}%"


def utc_timestamp_label(ts: datetime | None = None) -> str:
    """Return a display-ready UTC timestamp label."""
    current = ts or datetime.now(timezone.utc)
    return current.strftime("%Y-%m-%d %H:%M UTC")


def weighted_average(values: Iterable[float], weights: Iterable[float]) -> float:
    """Compute weighted average safely."""
    vals = list(values)
    wts = list(weights)
    denom = sum(wts)
    if not vals or not wts or denom == 0:
        return 0.0
    return sum(v * w for v, w in zip(vals, wts)) / denom