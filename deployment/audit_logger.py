import json
import os
import socket
import psutil # type: ignore

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any
from tenacity import retry, stop_after_attempt, wait_fixed # pyright: ignore[reportMissingImports]

from ..logs.logger import get_logger
from ..logs.standards import LogDomain, build_event, default_log_path

MAX_LOG_SIZE = 200 * 1024 * 1024  # 200MB
SENSITIVE_KEYS = {"password", "token", "secret"}
LOG_FILE = default_log_path(LogDomain.AUDIT, "deployment_audit.jsonl")
VALID_EVENT_TYPES = {"deploy", "rollback", "config_change", "version_bump", "log_access"}

logger = get_logger("Audit Logger")


def _redact_sensitive(data: Dict[str, Any]) -> Dict[str, Any]:
    return {k: "**REDACTED**" if k in SENSITIVE_KEYS else v for k, v in data.items()}


@retry(stop=stop_after_attempt(3), wait=wait_fixed(0.5))
def log_event(
    event_type: str,
    user: str,
    environment: str,
    branch: Optional[str] = None,
    version: Optional[str] = None,
    success: bool = True,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    if event_type not in VALID_EVENT_TYPES:
        raise ValueError(f"Invalid event type: {event_type}. Valid types: {VALID_EVENT_TYPES}")

    if LOG_FILE.exists() and LOG_FILE.stat().st_size > MAX_LOG_SIZE:
        rotate_logs()

    redacted_details = _redact_sensitive(details or {})
    event = build_event(
        agent="deployment",
        event=event_type,
        severity="INFO" if success else "ERROR",
        payload={
            "user": user,
            "environment": environment,
            "branch": branch,
            "version": version,
            "success": success,
            "details": redacted_details,
            "system_metrics": {
                "cpu_percent": psutil.cpu_percent(),
                "ram_used": psutil.virtual_memory().used,
                "disk_free": psutil.disk_usage('/').free,
            },
            "hostname": socket.gethostname(),
            "captured_at": datetime.now(timezone.utc).isoformat(),
        },
    )

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event.__dict__, ensure_ascii=False) + "\n")

    logger.info(json.dumps(event.__dict__))


def rotate_logs() -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    archived_log = LOG_FILE.with_name(f"deployment_audit_{timestamp}.jsonl")
    os.rename(LOG_FILE, archived_log)


def read_logs(as_dicts: bool = True, limit: int = 100):
    log_event(
        event_type="log_access",
        user=os.getenv("USER", "unknown"),
        environment="audit",
        details={"action": "read_logs", "limit": limit},
    )
    if not LOG_FILE.exists():
        return []

    lines = LOG_FILE.read_text(encoding="utf-8").splitlines()[-limit:]
    return [json.loads(line) for line in lines] if as_dicts else lines
