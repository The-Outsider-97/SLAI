from __future__ import annotations

import json
import logging

from dataclasses import asdict
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


class StandardJSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        event = build_event(
            agent=record.name,
            trace_id=getattr(record, "trace_id", None),
            event=str(getattr(record, "event", record.msg)),
            severity=record.levelname.lower(),
            payload={
                "message": record.getMessage(),
                "module": record.module,
                "component": getattr(record, "component", None),
                "metadata": getattr(record, "metadata", {}),
            },
        )
        return json.dumps(asdict(event), ensure_ascii=False, default=str)


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
    # ensure_log_directories()
    return DOMAIN_DIRS[domain] / filename
