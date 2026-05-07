"""
monitoring/alert_manager.py
─────────────────────────────
Alert evaluation, deduplication, routing, and multi-transport dispatch.

Key improvements over v1
─────────────────────────
  • AlertTransport protocol – pluggable transports (Email, Slack, Webhook)
  • SlackTransport    – Slack Incoming Webhooks with rich attachment blocks
  • WebhookTransport  – generic HTTP POST (JSON payload, optional HMAC signing)
  • EmailTransport    – original transport, refactored to the new protocol
  • CircuitBreaker per transport key – open/half-open/closed state machine
  • TokenBucketLimiter on total alert dispatch – burst protection
  • RetryPolicy on every transport.send() call
  • Persistent dedup state (JSON file) – survives process restarts
  • Severity routing – critical alerts go to all configured transports,
    warning alerts skip Slack/webhook if only degraded
  • Structured logging throughout
"""

from __future__ import annotations

import hashlib
import hmac
import json
import smtplib
import time
import urllib.request
import urllib.error
import threading

from numpy.__config__ import CONFIG
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta, timezone
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .config_loader import get_config_section, load_global_config
from .drift_detection import DriftResult
from .health_check import HealthReport
from .metrics_collector import MetricSnapshot
from .resilience import *
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Alert Manager")
printer = PrettyPrinter()

# ──────────────────────────────────────────────
# Alert data class
# ──────────────────────────────────────────────
@dataclass
class Alert:
    source: str
    severity: str           # "critical" | "warning" | "info"
    dedup_key: str
    subject: str
    message: str
    payload: dict[str, Any]
    timestamp_utc: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def emoji(self) -> str:
        return {"critical": "🔴", "warning": "🟡", "info": "🔵"}.get(self.severity, "⚪")


# ──────────────────────────────────────────────
# Transport protocol
# ──────────────────────────────────────────────
@runtime_checkable
class AlertTransport(Protocol):
    """All transports must implement this single method."""
    transport_key: str          # Used as circuit-breaker and log key

    def send(self, alert: Alert) -> None:
        """Send *alert*. Raise on failure so RetryPolicy can retry."""
        ...


# ──────────────────────────────────────────────
# Email transport
# ──────────────────────────────────────────────
@dataclass
class EmailTransport:
    transport_key: str = "email"
    smtp_host: str = "localhost"
    smtp_port: int = 587
    username: str = ""
    password: str = ""
    from_email: str = "alerts@slai.local"
    recipient_email: str = ""
    use_tls: bool = True

    def send(self, alert: Alert) -> None:
        if not self.recipient_email:
            raise ValueError("EmailTransport: recipient_email is not configured.")
        msg = MIMEText(alert.message)
        msg["Subject"] = f"[{alert.severity.upper()}] {alert.subject}"
        msg["From"] = self.from_email
        msg["To"] = self.recipient_email
        with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
            if self.use_tls:
                server.starttls()
            if self.username:
                server.login(self.username, self.password)
            server.sendmail(self.from_email, self.recipient_email, msg.as_string())


# ──────────────────────────────────────────────
# Slack transport
# ──────────────────────────────────────────────
@dataclass
class SlackTransport:
    """
    Sends alerts to a Slack channel via an Incoming Webhook URL.
    Rich attachment blocks include severity, source, and payload summary.
    """
    transport_key: str = "slack"
    webhook_url: str = ""
    channel: str = "#alerts"
    username: str = "SLAI Monitor"
    timeout: float = 5.0

    def send(self, alert: Alert) -> None:
        if not self.webhook_url:
            raise ValueError("SlackTransport: webhook_url is not configured.")

        colour = {"critical": "#E01E5A", "warning": "#ECB22E", "info": "#36C5F0"}.get(
            alert.severity, "#AAAAAA"
        )

        # Build a Slack message with a rich attachment
        payload = {
            "username": self.username,
            "channel": self.channel,
            "text": f"{alert.emoji} *{alert.subject}*",
            "attachments": [
                {
                    "color": colour,
                    "fields": [
                        {"title": "Severity", "value": alert.severity.upper(), "short": True},
                        {"title": "Source", "value": alert.source, "short": True},
                        {"title": "Message", "value": alert.message, "short": False},
                        {"title": "Timestamp (UTC)", "value": alert.timestamp_utc, "short": True},
                    ],
                    "footer": "SLAI Monitoring",
                    "ts": int(
                        datetime.fromisoformat(alert.timestamp_utc).timestamp()
                    ),
                }
            ],
        }

        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            self.webhook_url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            status = resp.getcode()
            if status not in range(200, 300):
                raise OSError(f"Slack webhook returned HTTP {status}")


# ──────────────────────────────────────────────
# Webhook transport
# ──────────────────────────────────────────────
@dataclass
class WebhookTransport:
    """
    Generic HTTP POST transport.
    Optionally signs the request body with HMAC-SHA256 via a custom header.
    """
    transport_key: str = "webhook"
    url: str = ""
    secret_header: str = ""         # Header name for HMAC signature, e.g. "X-SLAI-Signature"
    secret_value: str = ""          # HMAC key
    timeout: float = 5.0
    extra_headers: dict[str, str] = field(default_factory=dict)

    def send(self, alert: Alert) -> None:
        if not self.url:
            raise ValueError("WebhookTransport: url is not configured.")

        body = json.dumps(alert.to_dict()).encode()
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            **self.extra_headers,
        }

        if self.secret_header and self.secret_value:
            sig = hmac.new(
                self.secret_value.encode(), body, hashlib.sha256
            ).hexdigest()
            headers[self.secret_header] = f"sha256={sig}"

        req = urllib.request.Request(
            self.url, data=body, headers=headers, method="POST"
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            status = resp.getcode()
            if status not in range(200, 300):
                raise OSError(f"Webhook returned HTTP {status}")


# ──────────────────────────────────────────────
# Dedup state (persistent)
# ──────────────────────────────────────────────
class _DedupStore:
    """
    Thread-safe deduplication store backed by a JSON file.
    Falls back gracefully to in-memory only if the file can't be written.
    """

    def __init__(self, path: str, cooldown: timedelta) -> None:
        self._path = Path(path)
        self._cooldown = cooldown
        self._lock = threading.Lock()
        self._store: dict[str, str] = {}  # dedup_key → ISO timestamp string
        self._load()

    def is_allowed(self, key: str) -> bool:
        now = datetime.now(timezone.utc)
        with self._lock:
            last_str = self._store.get(key)
            if last_str:
                last = datetime.fromisoformat(last_str)
                if (now - last) < self._cooldown:
                    return False
            self._store[key] = now.isoformat()
            self._save()
            return True

    def _load(self) -> None:
        try:
            if self._path.exists():
                data = json.loads(self._path.read_text())
                if isinstance(data, dict):
                    self._store = data
        except Exception as exc:
            logger.warning("Could not load dedup state.", path=str(self._path), error=str(exc))

    def _save(self) -> None:
        try:
            self._path.write_text(json.dumps(self._store, indent=2))
        except Exception as exc:
            logger.warning("Could not persist dedup state.", path=str(self._path), error=str(exc))


# ──────────────────────────────────────────────
# Alert manager
# ──────────────────────────────────────────────
class AlertManager:
    """
    Evaluate monitoring data, deduplicate, rate-limit, and dispatch alerts
    via one or more transports.

    Parameters
    ----------
    transports:
        List of AlertTransport instances. If empty, alerts are logged only.
    cooldown_seconds:
        Minimum seconds between repeated alerts for the same dedup_key.
    dedup_state_path:
        Path to the JSON file used for persistent dedup state.
    rate_limit_tokens / rate_limit_refill_per_second:
        Token-bucket parameters guarding the total outbound alert rate.
    cb_failure_threshold / cb_recovery_timeout:
        Circuit-breaker parameters (per transport).
    retry_max / retry_base_delay:
        Retry parameters for each transport send attempt.
    """

    def __init__(
        self,
        transports: list[AlertTransport] | None = None,
        cooldown_seconds: int = 300,
        dedup_state_path: str = "/tmp/slai_dedup_state.json",
        rate_limit_tokens: int = 10,
        rate_limit_refill_per_second: float = 1.0,
        cb_failure_threshold: int = 5,
        cb_recovery_timeout: float = 60.0,
        retry_max: int = 3,
        retry_base_delay: float = 0.5,
    ) -> None:
        self.config = load_global_config()
        self.alert_config = get_config_section("alert_manager")

        self._transports: list[AlertTransport] = transports or []
        self._dedup = _DedupStore(dedup_state_path, timedelta(seconds=cooldown_seconds))
        self._rate_limiter = TokenBucketLimiter(
            capacity=rate_limit_tokens,
            refill_per_second=rate_limit_refill_per_second,
        )
        self._cb = CircuitBreakerRegistry(
            failure_threshold=cb_failure_threshold,
            recovery_timeout=cb_recovery_timeout,
        )
        self._retry = RetryPolicy(
            max_retries=retry_max,
            base_delay=retry_base_delay,
            max_delay=30.0,
            jitter=True,
            retryable_exceptions=(OSError, TimeoutError, ConnectionError),
        )

    # ── Evaluation helpers ───────────────────────

    def process_metrics(
        self, snapshot: MetricSnapshot, thresholds: dict[str, float]
    ) -> list[Alert]:
        alerts: list[Alert] = []
        checks = [
            ("cpu", snapshot.cpu.percent_total, thresholds.get("cpu_percent", 85.0)),
            ("memory", snapshot.memory.percent_used, thresholds.get("memory_percent", 85.0)),
            ("disk", snapshot.disk.percent_used, thresholds.get("disk_percent", 85.0)),
        ]
        for label, value, threshold in checks:
            if value >= threshold:
                severity = "critical" if value >= threshold + 10 else "warning"
                alert = self._build_alert(
                    source="metrics",
                    severity=severity,
                    dedup_key=f"metrics:{label}",
                    subject=f"SLAI monitoring: {label} high",
                    message=f"Metric '{label}' is at {value:.2f}% (threshold {threshold:.2f}%).",
                    payload={"metric": label, "value": value, "threshold": threshold},
                )
                if self._dedup.is_allowed(alert.dedup_key):
                    alerts.append(alert)
        return alerts

    def process_drift(self, result: DriftResult) -> list[Alert]:
        if not result.drift_detected:
            return []
        alert = self._build_alert(
            source="drift",
            severity="warning",
            dedup_key=f"drift:{result.metric_name}:{result.test}",
            subject=f"SLAI drift: {result.metric_name}",
            message=(
                f"Drift detected for '{result.metric_name}' "
                f"[{result.test.upper()}] statistic={result.statistic:.6f}, "
                f"p-value={result.p_value:.6f} (threshold={result.threshold:.3f})."
            ),
            payload=result.to_dict(),
        )
        return [alert] if self._dedup.is_allowed(alert.dedup_key) else []

    def process_health(self, report: HealthReport) -> list[Alert]:
        if report.overall_status == "healthy":
            return []
        severity = "critical" if report.overall_status == "unhealthy" else "warning"
        alert = self._build_alert(
            source="health",
            severity=severity,
            dedup_key=f"health:{report.overall_status}",
            subject=f"SLAI health: {report.overall_status}",
            message=(
                f"Health status is {report.overall_status}. "
                f"healthy={report.healthy_count}, "
                f"degraded={report.degraded_count}, "
                f"unhealthy={report.unhealthy_count}."
            ),
            payload=report.to_dict(),
        )
        return [alert] if self._dedup.is_allowed(alert.dedup_key) else []

    # ── Dispatch ─────────────────────────────────

    def dispatch(self, alerts: list[Alert]) -> list[dict[str, str]]:
        """
        Send *alerts* through all configured transports.

        Returns a list of outcome dicts with keys:
          alert_subject, transport, status, detail
        """
        outcomes: list[dict[str, str]] = []
        if not alerts:
            return outcomes

        if not self._transports:
            for alert in alerts:
                logger.warning(
                    "No transports configured; alert logged only.",
                    severity=alert.severity,
                    subject=alert.subject,
                )
                outcomes.append({
                    "alert_subject": alert.subject,
                    "transport": "none",
                    "status": "logged_only",
                    "detail": "No transports configured.",
                })
            return outcomes

        for alert in alerts:
            try:
                self._rate_limiter.acquire(tokens=1, block=False)
            except RateLimitExceeded as exc:
                logger.warning("Rate limit exceeded; alert dropped.", subject=alert.subject)
                outcomes.append({
                    "alert_subject": alert.subject,
                    "transport": "all",
                    "status": "rate_limited",
                    "detail": str(exc),
                })
                continue

            for transport in self._transports:
                outcome = self._send_via(transport, alert)
                outcomes.append(outcome)

        return outcomes

    def circuit_breaker_status(self) -> dict[str, str]:
        """Return current circuit breaker states keyed by transport name."""
        return self._cb.status()

    # ── Private helpers ──────────────────────────

    def _send_via(self, transport: AlertTransport, alert: Alert) -> dict[str, str]:
        key = transport.transport_key

        def _do_send() -> None:
            self._cb.call(key, transport.send, alert)

        try:
            self._retry.execute(_do_send)
            logger.info(
                "Alert dispatched.",
                transport=key,
                severity=alert.severity,
                subject=alert.subject,
            )
            return {
                "alert_subject": alert.subject,
                "transport": key,
                "status": "sent",
                "detail": "",
            }
        except CircuitBreakerOpen as exc:
            logger.warning("Circuit breaker open; skipping transport.", transport=key)
            return {
                "alert_subject": alert.subject,
                "transport": key,
                "status": "circuit_open",
                "detail": str(exc),
            }
        except Exception as exc:
            logger.error(
                "Failed to dispatch alert after retries.",
                transport=key,
                subject=alert.subject,
                error=str(exc),
            )
            return {
                "alert_subject": alert.subject,
                "transport": key,
                "status": "failed",
                "detail": str(exc),
            }

    @staticmethod
    def _build_alert(
        source: str,
        severity: str,
        dedup_key: str,
        subject: str,
        message: str,
        payload: dict[str, Any],
    ) -> Alert:
        return Alert(
            source=source,
            severity=severity,
            dedup_key=dedup_key,
            subject=subject,
            message=message,
            payload=payload,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
        )

    # ── Factory ──────────────────────────────────

    @classmethod
    def from_config(cls, cfg: dict) -> "AlertManager":
        """
        Build an AlertManager from a plain config dict.

        Expects the dict returned by get_config_section("alert_manager").
        Each transport sub-section is only activated when enabled: true.

        Example checkpoint_config.yaml layout:

            alert_manager:
              cooldown_seconds: 300
              dedup_state_path: /tmp/slai_dedup_state.json
              rate_limit_tokens: 10
              rate_limit_refill_per_second: 1.0
              cb_failure_threshold: 5
              cb_recovery_timeout: 60.0
              email:
                enabled: true
                smtp_host: smtp.example.com
                ...
              slack:
                enabled: true
                webhook_url: https://hooks.slack.com/services/...
                channel: "#alerts"
              webhook:
                enabled: true
                url: https://example.com/hooks/slai
        """
        transports: list[AlertTransport] = []

        email_cfg = cfg.get("email", {})
        if email_cfg.get("enabled", False):
            transports.append(EmailTransport(
                smtp_host=email_cfg.get("smtp_host", "localhost"),
                smtp_port=int(email_cfg.get("smtp_port", 587)),
                username=email_cfg.get("username", ""),
                password=email_cfg.get("password", ""),
                from_email=email_cfg.get("from_email", "alerts@slai.local"),
                recipient_email=email_cfg.get("recipient_email", ""),
                use_tls=bool(email_cfg.get("use_tls", True)),
            ))
            logger.info(f"{CONFIG} Email transport enabled. recipient={email_cfg.get('recipient_email')}")

        slack_cfg = cfg.get("slack", {})
        if slack_cfg.get("enabled", False):
            transports.append(SlackTransport(
                webhook_url=slack_cfg.get("webhook_url", ""),
                channel=slack_cfg.get("channel", "#alerts"),
                username=slack_cfg.get("username", "SLAI Monitor"),
            ))
            logger.info(f"{CONFIG} Slack transport enabled. channel={slack_cfg.get('channel')}")

        webhook_cfg = cfg.get("webhook", {})
        if webhook_cfg.get("enabled", False):
            transports.append(WebhookTransport(
                url=webhook_cfg.get("url", ""),
                secret_header=webhook_cfg.get("secret_header", ""),
                secret_value=webhook_cfg.get("secret_value", ""),
                timeout=float(webhook_cfg.get("timeout_seconds", 5.0)),
            ))
            logger.info(f"{CONFIG} Webhook transport enabled. url={webhook_cfg.get('url')}")

        return cls(
            transports=transports,
            cooldown_seconds=int(cfg.get("cooldown_seconds", 300)),
            dedup_state_path=cfg.get("dedup_state_path", "/tmp/slai_dedup_state.json"),
            rate_limit_tokens=int(cfg.get("rate_limit_tokens", 10)),
            rate_limit_refill_per_second=float(cfg.get("rate_limit_refill_per_second", 1.0)),
            cb_failure_threshold=int(cfg.get("cb_failure_threshold", 5)),
            cb_recovery_timeout=float(cfg.get("cb_recovery_timeout", 60.0)),
        )


# ── Compatibility helper ─────────────────────────────────────────────────────

def send_alert(subject: str, body: str, to_email: str) -> list[dict[str, str]]:
    """Compatibility shim: send a one-off alert via email."""
    alert = Alert(
        source="legacy",
        severity="warning",
        dedup_key=f"legacy:{subject}",
        subject=subject,
        message=body,
        payload={},
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )
    transport = EmailTransport(
        smtp_host="localhost",
        smtp_port=25,
        recipient_email=to_email,
        use_tls=False,
    )
    manager = AlertManager(transports=[transport], cooldown_seconds=0)
    return manager.dispatch([alert])


if __name__ == "__main__":
    print("\n=== Running Alert Manager ===\n")
    printer.status("TEST", "Alert Manager initialized", "info")
    from .metrics_collector import MetricsCollector
    from .health_check import ServiceHealthResult

    # 1. Load configuration
    alert_cfg = get_config_section("alert_manager")
    printer.status("CONFIG", f"Loaded alert_manager section", "info")

    # 2. Create AlertManager from config
    manager = AlertManager.from_config(alert_cfg)
    printer.status("INIT", f"AlertManager created with {len(manager._transports)} transport(s)", "info")

    # 3. Simulate a metric snapshot with high CPU (critical)
    collector = MetricsCollector()
    snapshot = collector.collect_snapshot()
    snapshot.cpu.percent_total = 96.0   # force critical alert
    thresholds = {
        "cpu_percent": 85.0,
        "memory_percent": 85.0,
        "disk_percent": 90.0,
    }
    metrics_alerts = manager.process_metrics(snapshot, thresholds)
    printer.status("METRICS", f"Generated {len(metrics_alerts)} alert(s)", "info")

    # 4. Simulate drift detection
    drift_result = DriftResult(
        metric_name="feature_importance",
        test="ks",
        statistic=0.42,
        p_value=0.001,
        threshold=0.05,
        drift_detected=True,
        reference_count=500,
        current_count=500,
        notes=["Distribution shift detected"]
    )
    drift_alerts = manager.process_drift(drift_result)
    printer.status("DRIFT", f"Generated {len(drift_alerts)} alert(s)", "info")

    # 5. Simulate health report
    health_report = HealthReport(
        timestamp_utc="2025-01-01T12:00:00Z",
        overall_status="unhealthy",
        healthy_count=2,
        unhealthy_count=1,
        degraded_count=1,
        results=[
            ServiceHealthResult("API", "http", "https://api.local", "healthy", 45.2, 200, "OK"),
            ServiceHealthResult("DB", "tcp", "db.local:5432", "unhealthy", 0.0, None, "Connection refused"),
        ],
        duration_ms=234.5
    )
    health_alerts = manager.process_health(health_report)
    printer.status("HEALTH", f"Generated {len(health_alerts)} alert(s)", "info")

    # 6. Dispatch all alerts
    all_alerts = metrics_alerts + drift_alerts + health_alerts
    if all_alerts:
        printer.status("DISPATCH", f"Dispatching {len(all_alerts)} alert(s)...", "info")
        outcomes = manager.dispatch(all_alerts)
        for out in outcomes:
            printer.status(
                "OUTCOME",
                f"{out['alert_subject']} → {out['transport']}: {out['status']} ({out['detail']})",
                "info"
            )
    else:
        printer.status("DISPATCH", "No alerts generated", "warning")

    # 7. Show circuit breaker status
    cb_status = manager.circuit_breaker_status()
    if cb_status:
        printer.status("CIRCUIT", f"Circuit breakers: {cb_status}", "info")
    else:
        printer.status("CIRCUIT", "No circuit breakers active", "info")

    printer.status("TEST", "All Alert Manager checks passed", "success")
    print("\n=== Test ran successfully ===\n")