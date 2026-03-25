from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional

import smtplib

from monitoring.drift_detection import DriftResult
from monitoring.health_check import HealthReport
from monitoring.metrics_collector import MetricSnapshot


@dataclass
class Alert:
    source: str
    severity: str
    dedup_key: str
    subject: str
    message: str
    payload: Dict[str, Any]
    timestamp_utc: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EmailAlertTransport:
    """Sends alerts via email; configuration injected externally."""

    def __init__(
        self,
        smtp_host: str,
        smtp_port: int,
        username: str,
        password: str,
        from_email: str,
        use_tls: bool = True,
    ) -> None:
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_email = from_email
        self.use_tls = use_tls

    def send(self, alert: Alert, recipient_email: str) -> None:
        msg = MIMEText(alert.message)
        msg["Subject"] = alert.subject
        msg["From"] = self.from_email
        msg["To"] = recipient_email

        with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
            if self.use_tls:
                server.starttls()
            if self.username:
                server.login(self.username, self.password)
            server.sendmail(self.from_email, recipient_email, msg.as_string())


class AlertManager:
    """Evaluate monitoring data, deduplicate, and emit alerts."""

    def __init__(
        self,
        cooldown_seconds: int = 300,
        transport: Optional[EmailAlertTransport] = None
    ) -> None:
        self.cooldown = timedelta(seconds=cooldown_seconds)
        self.transport = transport
        self._recent_alerts: Dict[str, datetime] = {}

    def process_metrics(self, snapshot: MetricSnapshot, thresholds: Dict[str, float]) -> List[Alert]:
        alerts = []
        checks = [
            ("cpu", snapshot.cpu.percent_total, thresholds.get("cpu_percent", 90.0)),
            ("memory", snapshot.memory.percent_used, thresholds.get("memory_percent", 90.0)),
            ("disk", snapshot.disk.percent_used, thresholds.get("disk_percent", 90.0)),
        ]
        for label, value, threshold in checks:
            if value >= threshold:
                alert = self._build_alert(
                    source="metrics",
                    severity="warning",
                    dedup_key=f"metrics:{label}",
                    subject=f"SLAI monitoring alert: {label} high",
                    message=f"Metric '{label}' reached {value:.2f}% (threshold {threshold:.2f}%).",
                    payload={"metric": label, "value": value, "threshold": threshold},
                )
                if self._is_allowed(alert.dedup_key):
                    alerts.append(alert)
        return alerts

    def process_drift(self, result: DriftResult) -> List[Alert]:
        if not result.drift_detected:
            return []
        alert = self._build_alert(
            source="drift",
            severity="warning",
            dedup_key=f"drift:{result.metric_name}",
            subject=f"SLAI drift alert: {result.metric_name}",
            message=(
                f"Drift detected for '{result.metric_name}' with KS statistic {result.statistic:.6f} "
                f"and p-value {result.p_value:.6f} (threshold {result.threshold:.3f})."
            ),
            payload=result.to_dict(),
        )
        return [alert] if self._is_allowed(alert.dedup_key) else []

    def process_health(self, report: HealthReport) -> List[Alert]:
        if report.overall_status == "healthy":
            return []
        severity = "critical" if report.overall_status == "unhealthy" else "warning"
        alert = self._build_alert(
            source="health",
            severity=severity,
            dedup_key=f"health:{report.overall_status}",
            subject=f"SLAI health alert: {report.overall_status}",
            message=(
                f"Health status is {report.overall_status}. healthy={report.healthy_count}, "
                f"degraded={report.degraded_count}, unhealthy={report.unhealthy_count}."
            ),
            payload=report.to_dict(),
        )
        return [alert] if self._is_allowed(alert.dedup_key) else []

    def dispatch_email(self, alerts: List[Alert], recipient_email: str) -> List[str]:
        """Send alerts via email if transport is configured."""
        outcomes = []
        if not self.transport:
            for alert in alerts:
                outcomes.append(f"Transport not configured. Skipped: {alert.subject}")
            return outcomes

        for alert in alerts:
            try:
                self.transport.send(alert, recipient_email)
                outcomes.append(f"Sent: {alert.subject}")
            except Exception as exc:
                outcomes.append(f"Failed to send '{alert.subject}': {exc}")
        return outcomes

    def _build_alert(
        self,
        source: str,
        severity: str,
        dedup_key: str,
        subject: str,
        message: str,
        payload: Dict[str, Any],
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

    def _is_allowed(self, dedup_key: str) -> bool:
        now = datetime.now(timezone.utc)
        last = self._recent_alerts.get(dedup_key)
        if last and (now - last) < self.cooldown:
            return False
        self._recent_alerts[dedup_key] = now
        return True


def send_alert(subject, body, to_email):
    """Compatibility helper: sends a single alert via email."""
    alert = Alert(
        source="legacy",
        severity="warning",
        dedup_key=f"legacy:{subject}",
        subject=subject,
        message=body,
        payload={},
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )
    # No cooldown, no transport – just a direct email attempt
    transport = EmailAlertTransport(
        smtp_host="localhost",    # Should be configured externally
        smtp_port=25,
        username="",
        password="",
        from_email="alerts@slai.local",
        use_tls=False,
    )
    manager = AlertManager(transport=transport)
    return manager.dispatch_email([alert], to_email)


if __name__ == "__main__":
    print("Alert manager ready. Configure EmailAlertTransport to dispatch emails.")
