from __future__ import annotations

import json
import logging
import os
import re
import time

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


DEFAULT_PII_PATTERNS = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "phone": re.compile(r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
}


@dataclass
class LogGovernancePolicy:
    """Governance settings for log security and lifecycle management."""

    rotation_bytes: int = 10 * 1024 * 1024
    rotation_backups: int = 10
    retention_days: int = 30
    file_permissions: int = 0o640
    directory_permissions: int = 0o750
    pii_replacement: str = "[REDACTED]"
    pii_patterns: Dict[str, re.Pattern[str]] = field(default_factory=lambda: DEFAULT_PII_PATTERNS.copy())


@dataclass
class MetricsAlertThresholds:
    """Service-level observability thresholds."""

    min_health_score: float = 0.95
    max_p95_latency_ms: float = 1000.0
    max_error_rate: float = 0.02


@dataclass
class MetricsSnapshot:
    window_requests: int
    window_errors: int
    error_rate: float
    avg_latency_ms: float
    p95_latency_ms: float
    health_score: float
    timestamp: str


class PIIRedactingFormatter(logging.Formatter):
    def __init__(self, policy: LogGovernancePolicy):
        super().__init__()
        self.policy = policy

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": getattr(record, "event", record.msg),
            "message": record.getMessage(),
            "service": getattr(record, "service", "slai"),
            "environment": getattr(record, "environment", "unknown"),
            "trace_id": getattr(record, "trace_id", None),
            "span_id": getattr(record, "span_id", None),
            "component": getattr(record, "component", None),
            "metadata": getattr(record, "metadata", {}),
        }
        redacted = self._redact(payload)
        return json.dumps(redacted, ensure_ascii=False)

    def _redact(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {k: self._redact(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._redact(v) for v in value]
        if isinstance(value, str):
            redacted = value
            for pattern in self.policy.pii_patterns.values():
                redacted = pattern.sub(self.policy.pii_replacement, redacted)
            return redacted
        return value


class StructuredLogger:
    """JSON logger with standardized schema and governance controls."""

    def __init__(
        self,
        name: str,
        log_path: str | Path = "logs/safe_ai.log",
        level: int = logging.INFO,
        policy: Optional[LogGovernancePolicy] = None,
    ) -> None:
        self.policy = policy or LogGovernancePolicy()
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(self.log_path.parent, self.policy.directory_permissions)

        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.propagate = False

        if not self.logger.handlers:
            handler = RotatingFileHandler(
                self.log_path,
                maxBytes=self.policy.rotation_bytes,
                backupCount=self.policy.rotation_backups,
                encoding="utf-8",
            )
            handler.setFormatter(PIIRedactingFormatter(self.policy))
            self.logger.addHandler(handler)

        if self.log_path.exists():
            os.chmod(self.log_path, self.policy.file_permissions)

    def log(
        self,
        level: int,
        event: str,
        message: str,
        *,
        service: str = "slai",
        environment: str = "prod",
        component: Optional[str] = None,
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.logger.log(
            level,
            message,
            extra={
                "event": event,
                "service": service,
                "environment": environment,
                "component": component,
                "trace_id": trace_id,
                "span_id": span_id,
                "metadata": metadata or {},
            },
        )

    def enforce_retention(self, now: Optional[datetime] = None) -> int:
        now = now or datetime.now(timezone.utc)
        deleted = 0
        cutoff = now - timedelta(days=self.policy.retention_days)
        for candidate in self.log_path.parent.glob(f"{self.log_path.name}*"):
            mtime = datetime.fromtimestamp(candidate.stat().st_mtime, tz=timezone.utc)
            if mtime < cutoff:
                candidate.unlink(missing_ok=True)
                deleted += 1
        return deleted


class ServiceMetrics:
    """Tracks service-level health, latency, and error-rate with alerting."""

    def __init__(
        self,
        thresholds: Optional[MetricsAlertThresholds] = None,
        window_size: int = 500,
    ) -> None:
        self.thresholds = thresholds or MetricsAlertThresholds()
        self.window_size = window_size
        self._latencies: list[float] = []
        self._errors: list[int] = []
        self._requests: int = 0

    def record_request(self, latency_ms: float, *, is_error: bool = False) -> None:
        self._requests += 1
        self._latencies.append(float(latency_ms))
        self._errors.append(1 if is_error else 0)
        if len(self._latencies) > self.window_size:
            self._latencies.pop(0)
            self._errors.pop(0)

    def snapshot(self) -> MetricsSnapshot:
        if not self._latencies:
            return MetricsSnapshot(0, 0, 0.0, 0.0, 0.0, 1.0, datetime.now(timezone.utc).isoformat())

        latencies = sorted(self._latencies)
        error_count = sum(self._errors)
        request_count = len(self._latencies)
        p95_idx = max(0, min(len(latencies) - 1, int(0.95 * len(latencies)) - 1))
        p95 = latencies[p95_idx]
        error_rate = error_count / request_count
        avg_latency = sum(latencies) / request_count

        latency_penalty = min(1.0, p95 / max(self.thresholds.max_p95_latency_ms, 1.0))
        error_penalty = min(1.0, error_rate / max(self.thresholds.max_error_rate, 1e-8))
        health_score = max(0.0, 1.0 - 0.5 * latency_penalty - 0.5 * error_penalty)

        return MetricsSnapshot(
            window_requests=request_count,
            window_errors=error_count,
            error_rate=round(error_rate, 6),
            avg_latency_ms=round(avg_latency, 3),
            p95_latency_ms=round(p95, 3),
            health_score=round(health_score, 6),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def evaluate_alerts(self) -> list[str]:
        snap = self.snapshot()
        alerts: list[str] = []
        if snap.health_score < self.thresholds.min_health_score:
            alerts.append(
                f"health_score_breach:{snap.health_score:.3f}<{self.thresholds.min_health_score:.3f}"
            )
        if snap.p95_latency_ms > self.thresholds.max_p95_latency_ms:
            alerts.append(
                f"latency_breach:{snap.p95_latency_ms:.1f}>{self.thresholds.max_p95_latency_ms:.1f}"
            )
        if snap.error_rate > self.thresholds.max_error_rate:
            alerts.append(f"error_rate_breach:{snap.error_rate:.4f}>{self.thresholds.max_error_rate:.4f}")
        return alerts

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self.snapshot())
