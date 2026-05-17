from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


class LogDomain(str, Enum):
    RUNTIME = "runtime"
    AUDIT = "audit"
    TRAINING = "training"


BASE_LOG_DIR = Path("logs")
DOMAIN_DIRS = {
    LogDomain.RUNTIME: BASE_LOG_DIR / "runtime",
    LogDomain.AUDIT: BASE_LOG_DIR / "audit",
    LogDomain.TRAINING: BASE_LOG_DIR / "training",
}


@dataclass(frozen=True)
class StandardLogEvent:
    timestamp: str
    agent: str
    trace_id: str
    event: str
    severity: str
    payload: dict[str, Any]


def ensure_log_directories() -> None:
    for path in DOMAIN_DIRS.values():
        path.mkdir(parents=True, exist_ok=True)


def build_event(*, agent: str, event: str, severity: str, payload: Mapping[str, Any] | None = None, trace_id: str | None = None) -> StandardLogEvent:
    return StandardLogEvent(
        timestamp=datetime.now(timezone.utc).isoformat(),
        agent=agent,
        trace_id=trace_id or "unknown",
        event=event,
        severity=severity,
        payload=dict(payload or {}),
    )


def default_log_path(domain: LogDomain, filename: str) -> Path:
    ensure_log_directories()
    return DOMAIN_DIRS[domain] / filename
