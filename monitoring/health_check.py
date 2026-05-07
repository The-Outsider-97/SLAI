"""
monitoring/health_check.py
───────────────────────────
TCP and HTTP health checks with production-grade resilience.

Key improvements over v1
─────────────────────────
  • Parallel execution via ThreadPoolExecutor (no more sequential timeouts stacking)
  • RetryPolicy applied per check (exponential back-off + jitter)
  • Configurable expected HTTP status codes per service
  • TLS certificate expiry check (days remaining, configurable warn threshold)
  • Degraded status for latency violations (configurable latency_warn_ms)
  • Structured logging throughout
  • format_report() and to_dict() extended with per-service latency
"""

from __future__ import annotations

import socket
import ssl
import time
import urllib.request
import urllib.error

from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone, timedelta
from typing import Any

from .config_loader import get_config_section, load_global_config
from .resilience import RetryPolicy
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("SLAI Health Check")
printer = PrettyPrinter()

# ──────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────
@dataclass
class ServiceHealthResult:
    name: str
    check_type: str                 # "tcp" | "http" | "tls"
    target: str
    status: str                     # "healthy" | "degraded" | "unhealthy"
    latency_ms: float
    status_code: int | None
    message: str
    tls_days_remaining: int | None = None
    attempt_count: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HealthReport:
    timestamp_utc: str
    overall_status: str             # "healthy" | "degraded" | "unhealthy"
    healthy_count: int
    unhealthy_count: int
    degraded_count: int
    results: list[ServiceHealthResult]
    duration_ms: float = 0.0        # Wall time for the full check run

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp_utc": self.timestamp_utc,
            "overall_status": self.overall_status,
            "healthy_count": self.healthy_count,
            "unhealthy_count": self.unhealthy_count,
            "degraded_count": self.degraded_count,
            "duration_ms": round(self.duration_ms, 2),
            "results": [r.to_dict() for r in self.results],
        }


# ──────────────────────────────────────────────
# Health checker
# ──────────────────────────────────────────────
# Default retry policy for network probes
_DEFAULT_RETRY = RetryPolicy(
    max_retries=2,
    base_delay=0.2,
    max_delay=5.0,
    backoff_factor=2.0,
    jitter=True,
    retryable_exceptions=(OSError, TimeoutError, ConnectionError),
)


class HealthChecker:
    """
    Run TCP / HTTP / TLS health checks in parallel, with retries.

    Parameters
    ----------
    max_workers:
        Thread pool size; defaults to min(32, len(services) + 4).
    retry_policy:
        Override the default RetryPolicy for all checks.
    default_timeout:
        Fall-back timeout (seconds) if a service config omits 'timeout'.
    """

    def __init__(
        self,
        max_workers: int = 8,
        retry_policy: RetryPolicy | None = None,
        default_timeout: float = 5.0,
    ) -> None:
        self.config = load_global_config()
        self.check_config = get_config_section("health_check")

        self.max_workers = max_workers
        self.retry_policy = retry_policy or _DEFAULT_RETRY
        self.default_timeout = default_timeout

    # ── Public API ───────────────────────────────

    def run_checks(self, services: list[dict[str, Any]]) -> HealthReport:
        """
        Execute all service checks in parallel and aggregate results.

        Service config keys
        -------------------
        Required: name, type ("tcp" | "http" | "tls")
        TCP:      host, port, timeout (opt)
        HTTP:     url, timeout (opt), expected_status (opt, list[int])
                  latency_warn_ms (opt)
        TLS:      host, port (default 443), timeout (opt)
                  tls_warn_days (opt, default 30)
        """
        if not services:
            return HealthReport(
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                overall_status="healthy",
                healthy_count=0,
                unhealthy_count=0,
                degraded_count=0,
                results=[],
            )

        wall_start = datetime.now(timezone.utc)
        workers = min(self.max_workers, len(services) + 4)
        results: list[ServiceHealthResult] = [None] * len(services)  # type: ignore[list-item]

        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="slai-health") as pool:
            future_to_idx: dict[Future[ServiceHealthResult], int] = {}
            for idx, svc in enumerate(services):
                fut = pool.submit(self._dispatch, svc)
                future_to_idx[fut] = idx

            for fut in as_completed(future_to_idx):
                idx = future_to_idx[fut]
                try:
                    results[idx] = fut.result()
                except Exception as exc:
                    svc = services[idx]
                    logger.error(
                        "Health check crashed unexpectedly.",
                        service=svc.get("name", "?"),
                        error=str(exc),
                    )
                    results[idx] = ServiceHealthResult(
                        name=svc.get("name", "Unknown"),
                        check_type=svc.get("type", "?"),
                        target="?",
                        status="unhealthy",
                        latency_ms=0.0,
                        status_code=None,
                        message=f"Internal error: {exc}",
                    )

        wall_ms = (datetime.now(timezone.utc) - wall_start).total_seconds() * 1000.0

        healthy = sum(1 for r in results if r.status == "healthy")
        degraded = sum(1 for r in results if r.status == "degraded")
        unhealthy = sum(1 for r in results if r.status == "unhealthy")

        if unhealthy > 0:
            overall = "unhealthy"
        elif degraded > 0:
            overall = "degraded"
        else:
            overall = "healthy"

        report = HealthReport(
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            overall_status=overall,
            healthy_count=healthy,
            unhealthy_count=unhealthy,
            degraded_count=degraded,
            results=results,
            duration_ms=wall_ms,
        )
        logger.info(
            f"Health check run complete. overall={overall}, healthy={healthy}, "
            f"degraded={degraded}, unhealthy={unhealthy}, duration_ms={round(wall_ms, 1)}"
        )
        return report

    def format_report(self, report: HealthReport) -> str:
        lines = [
            f"Overall: {report.overall_status.upper()} | "
            f"healthy={report.healthy_count}, degraded={report.degraded_count}, "
            f"unhealthy={report.unhealthy_count} | "
            f"wall={report.duration_ms:.1f}ms"
        ]
        for r in report.results:
            tls_info = (
                f" | TLS expires in {r.tls_days_remaining}d"
                if r.tls_days_remaining is not None
                else ""
            )
            attempts = f" | attempts={r.attempt_count}" if r.attempt_count > 1 else ""
            lines.append(
                f"  {r.status.upper():10s} {r.name} [{r.check_type}] "
                f"{r.latency_ms:.1f}ms{tls_info}{attempts}: {r.message}"
            )
        return "\n".join(lines)

    # ── Dispatcher ───────────────────────────────

    def _dispatch(self, svc: dict[str, Any]) -> ServiceHealthResult:
        check_type = svc.get("type", "tcp").lower().strip()
        if check_type == "http":
            return self._with_retry(self._check_http, svc)
        elif check_type == "tls":
            return self._with_retry(self._check_tls, svc)
        else:
            return self._with_retry(self._check_tcp, svc)

    def _with_retry(self, fn: Any, svc: dict[str, Any]) -> ServiceHealthResult:
        attempt = 0
        last_exc: Exception | None = None

        for attempt in range(1, self.retry_policy.max_retries + 2):
            try:
                result = fn(svc)
                if result.status == "unhealthy":
                    raise OSError(result.message)
                result.attempt_count = attempt
                return result
            except Exception as exc:
                last_exc = exc
                if attempt <= self.retry_policy.max_retries:
                    delay = self.retry_policy._sleep_duration(attempt - 1)
                    time.sleep(delay)    

        # All attempts exhausted – perform one final non‑retried check to get a result
        fallback = fn(svc)
        fallback.attempt_count = attempt
        return fallback

    # ── TCP check ────────────────────────────────

    def _check_tcp(self, svc: dict[str, Any]) -> ServiceHealthResult:
        name = svc.get("name", "Unnamed TCP")
        host = svc.get("host", "127.0.0.1")
        port = int(svc.get("port", 80))
        timeout = float(svc.get("timeout", self.default_timeout))
        latency_warn_ms = float(svc.get("latency_warn_ms", float("inf")))

        t0 = datetime.now(timezone.utc)
        try:
            with socket.create_connection((host, port), timeout=timeout):
                msg = "TCP connection successful."
                status = "healthy"
        except TimeoutError:
            status, msg = "unhealthy", "Connection timed out."
        except OSError as exc:
            status, msg = "unhealthy", f"Connection failed: {exc}"

        latency_ms = (datetime.now(timezone.utc) - t0).total_seconds() * 1000.0
        if status == "healthy" and latency_ms > latency_warn_ms:
            status = "degraded"
            msg = f"High latency ({latency_ms:.1f}ms > warn threshold {latency_warn_ms:.0f}ms)."

        return ServiceHealthResult(
            name=name, check_type="tcp",
            target=f"{host}:{port}", status=status,
            latency_ms=latency_ms, status_code=None, message=msg,
        )

    # ── HTTP check ───────────────────────────────

    def _check_http(self, svc: dict[str, Any]) -> ServiceHealthResult:
    
        name = svc.get("name", "Unnamed HTTP")
        url = svc.get("url", "http://127.0.0.1")
        timeout = float(svc.get("timeout", self.default_timeout))
        expected = svc.get("expected_status", list(range(200, 300)))
        latency_warn_ms = float(svc.get("latency_warn_ms", float("inf")))
    
        t0 = datetime.now(timezone.utc)
        status_code: int | None = None
        status = "unhealthy"
        msg = ""
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                status_code = resp.getcode()
                # status/msg determined below
        except urllib.error.HTTPError as exc:
            status_code = exc.code
            status = "unhealthy"
            msg = f"HTTP error: {exc.code} {exc.reason}"
        except Exception as exc:
            status = "unhealthy"
            msg = f"HTTP request failed: {exc}"
    
        # Determine status and message based on status_code if available
        if status_code is not None:
            if status_code in expected:
                status, msg = "healthy", f"HTTP {status_code} OK."
            elif 300 <= status_code < 500:
                status = "degraded"
                msg = f"HTTP endpoint returned {status_code}."
            else:
                status = "unhealthy"
                msg = f"HTTP endpoint returned {status_code}."
        else:
            # status and msg already set from exception, ensure msg not empty
            if not msg:
                msg = "No HTTP status code received."
    
        latency_ms = (datetime.now(timezone.utc) - t0).total_seconds() * 1000.0
        if status == "healthy" and latency_ms > latency_warn_ms:
            status = "degraded"
            msg = f"High latency ({latency_ms:.1f}ms > warn threshold {latency_warn_ms:.0f}ms)."
    
        return ServiceHealthResult(
            name=name, check_type="http",
            target=url, status=status,
            latency_ms=latency_ms, status_code=status_code, message=msg,
        )

    # ── TLS cert expiry check ────────────────────
    def _check_tls(self, svc: dict[str, Any]) -> ServiceHealthResult:
        name = svc.get("name", "Unnamed TLS")
        host = svc.get("host", "127.0.0.1")
        port = int(svc.get("port", 443))
        timeout = float(svc.get("timeout", self.default_timeout))
        warn_days = int(svc.get("tls_warn_days", 30))
    
        t0 = datetime.now(timezone.utc)
        days_remaining: int | None = None
        status = "unhealthy"
        msg = ""
    
        try:
            ctx = ssl.create_default_context()
            with socket.create_connection((host, port), timeout=timeout) as sock:
                with ctx.wrap_socket(sock, server_hostname=host) as ssock:
                    cert = ssock.getpeercert()
                    if cert is None:
                        status = "unhealthy"
                        msg = "No TLS certificate provided by peer."
                        days_remaining = None
                    else:
                        not_after_raw = cert.get("notAfter")
                        if not_after_raw is None:
                            status = "unhealthy"
                            msg = "Certificate missing 'notAfter' field."
                            days_remaining = None
                        else:
                            # Extract string from possible tuple format
                            if isinstance(not_after_raw, tuple):
                                # Find the tuple containing "notAfter"
                                not_after_str = None
                                for item in not_after_raw:
                                    if isinstance(item, tuple) and len(item) >= 2 and item[0] == "notAfter":
                                        not_after_str = item[1]
                                        break
                                if not_after_str is None:
                                    status = "unhealthy"
                                    msg = "Certificate 'notAfter' field has unsupported tuple format."
                                    days_remaining = None
                            else:
                                not_after_str = not_after_raw
    
                            if days_remaining is None and not_after_str is not None:
                                try:
                                    not_after = datetime.strptime(not_after_str, "%b %d %H:%M:%S %Y %Z").replace(tzinfo=timezone.utc) # type: ignore
                                    days_remaining = (not_after - datetime.now(timezone.utc)).days
                                    if days_remaining <= 0:
                                        status = "unhealthy"
                                        msg = f"TLS certificate EXPIRED {abs(days_remaining)} day(s) ago."
                                    elif days_remaining <= warn_days:
                                        status = "degraded"
                                        msg = f"TLS certificate expires in {days_remaining} day(s)."
                                    else:
                                        status = "healthy"
                                        msg = f"TLS certificate valid for {days_remaining} day(s)."
                                except ValueError as ve:
                                    status = "unhealthy"
                                    msg = f"Failed to parse certificate expiry: {ve}"
                                    days_remaining = None
        except ssl.SSLCertVerificationError as exc:
            status, msg = "unhealthy", f"TLS verification failed: {exc}"
        except TimeoutError:
            status, msg = "unhealthy", "TLS connection timed out."
        except OSError as exc:
            status, msg = "unhealthy", f"TLS connection failed: {exc}"
        except Exception as exc:
            status, msg = "unhealthy", f"TLS check error: {exc}"
    
        latency_ms = (datetime.now(timezone.utc) - t0).total_seconds() * 1000.0
        return ServiceHealthResult(
            name=name, check_type="tls",
            target=f"{host}:{port}", status=status,
            latency_ms=latency_ms, status_code=None, message=msg,
            tls_days_remaining=days_remaining,
        )

# ── Compatibility helper ─────────────────────────────────────────────────────

def service_health_check(host: str = "localhost", port: int = 8000) -> bool:
    """Compatibility shim: returns True if a single TCP service is healthy."""
    checker = HealthChecker()
    report = checker.run_checks([
        {"name": "legacy", "type": "tcp", "host": host, "port": port}
    ])
    return report.overall_status == "healthy"


if __name__ == "__main__":
    checker = HealthChecker(max_workers=4)
    report = checker.run_checks([
        {"name": "Local TCP", "type": "tcp", "host": "127.0.0.1", "port": 8000, "timeout": 0.5},
        {"name": "Google HTTPS", "type": "http", "url": "https://www.google.com", "timeout": 3.0,
         "latency_warn_ms": 800},
        {"name": "Google TLS", "type": "tls", "host": "www.google.com", "port": 443,
         "tls_warn_days": 30},
    ])
    print(checker.format_report(report))