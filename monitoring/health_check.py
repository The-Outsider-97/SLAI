from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import socket
import urllib.request


@dataclass
class ServiceHealthResult:
    name: str
    check_type: str      # "tcp" or "http"
    target: str
    status: str          # "healthy", "degraded", "unhealthy"
    latency_ms: float
    status_code: Optional[int]
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HealthReport:
    timestamp_utc: str
    overall_status: str   # "healthy", "degraded", "unhealthy"
    healthy_count: int
    unhealthy_count: int
    degraded_count: int
    results: List[ServiceHealthResult]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp_utc": self.timestamp_utc,
            "overall_status": self.overall_status,
            "healthy_count": self.healthy_count,
            "unhealthy_count": self.unhealthy_count,
            "degraded_count": self.degraded_count,
            "results": [r.to_dict() for r in self.results],
        }


class HealthChecker:
    """Run TCP/HTTP health checks and aggregate results."""

    def run_checks(self, services: List[Dict[str, Any]]) -> HealthReport:
        results: List[ServiceHealthResult] = []
        for svc in services:
            check_type = svc.get("type", "tcp").lower().strip()
            if check_type == "http":
                results.append(self._check_http(svc))
            else:
                results.append(self._check_tcp(svc))

        healthy = sum(1 for r in results if r.status == "healthy")
        degraded = sum(1 for r in results if r.status == "degraded")
        unhealthy = sum(1 for r in results if r.status == "unhealthy")

        overall = "healthy"
        if unhealthy > 0:
            overall = "unhealthy"
        elif degraded > 0:
            overall = "degraded"

        return HealthReport(
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            overall_status=overall,
            healthy_count=healthy,
            unhealthy_count=unhealthy,
            degraded_count=degraded,
            results=results,
        )

    def format_report(self, report: HealthReport) -> str:
        lines = [
            f"Overall: {report.overall_status.upper()} | healthy={report.healthy_count}, degraded={report.degraded_count}, unhealthy={report.unhealthy_count}"
        ]
        for r in report.results:
            lines.append(
                f"- {r.name} [{r.check_type}] {r.status.upper()} ({r.latency_ms:.2f} ms): {r.message}"
            )
        return "\n".join(lines)

    def _check_tcp(self, svc: Dict[str, Any]) -> ServiceHealthResult:
        name = svc.get("name", "Unnamed TCP Service")
        host = svc.get("host", "127.0.0.1")
        port = int(svc.get("port", 80))
        timeout = float(svc.get("timeout", 1.0))

        start = datetime.now(timezone.utc)
        try:
            with socket.create_connection((host, port), timeout=timeout):
                status = "healthy"
                msg = "TCP connection successful."
        except TimeoutError:
            status = "unhealthy"
            msg = "Connection timed out."
        except OSError as e:
            status = "unhealthy"
            msg = f"Connection failed: {e}"
        latency_ms = (datetime.now(timezone.utc) - start).total_seconds() * 1000.0

        return ServiceHealthResult(
            name=name,
            check_type="tcp",
            target=f"{host}:{port}",
            status=status,
            latency_ms=latency_ms,
            status_code=None,
            message=msg,
        )

    def _check_http(self, svc: Dict[str, Any]) -> ServiceHealthResult:
        name = svc.get("name", "Unnamed HTTP Service")
        url = svc.get("url", "http://127.0.0.1")
        timeout = float(svc.get("timeout", 1.5))

        start = datetime.now(timezone.utc)
        status_code = None
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                status_code = resp.getcode()
                if 200 <= status_code < 300:
                    status = "healthy"
                    msg = "HTTP endpoint returned success."
                elif 300 <= status_code < 500:
                    status = "degraded"
                    msg = f"HTTP endpoint returned status {status_code}."
                else:
                    status = "unhealthy"
                    msg = f"HTTP endpoint returned status {status_code}."
        except Exception as e:
            status = "unhealthy"
            msg = f"HTTP request failed: {e}"

        latency_ms = (datetime.now(timezone.utc) - start).total_seconds() * 1000.0
        return ServiceHealthResult(
            name=name,
            check_type="http",
            target=url,
            status=status,
            latency_ms=latency_ms,
            status_code=status_code,
            message=msg,
        )


def service_health_check(host="localhost", port=8000):
    """Compatibility helper: returns True if a single TCP service is healthy."""
    checker = HealthChecker()
    report = checker.run_checks([
        {"name": "legacy", "type": "tcp", "host": host, "port": port}
    ])
    return report.overall_status == "healthy"


if __name__ == "__main__":
    checker = HealthChecker()
    report = checker.run_checks([
        {"name": "Local TCP", "type": "tcp", "host": "127.0.0.1", "port": 8000, "timeout": 0.5},
    ])
    print(checker.format_report(report))
