import logging
import psutil
import json
import os
from datetime import datetime
from typing import Optional, Dict
from tenacity import retry, stop_after_attempt, wait_fixed

MAX_LOG_SIZE = 200 * 1024 * 1024  # 200MB
SENSITIVE_KEYS = {
    "password", 
    "token", 
    "secret"
}
LOG_DIR = "deployment/logs"
LOG_FILE = os.path.join(LOG_DIR, "deployment_audit.jsonl")
os.makedirs(LOG_DIR, exist_ok=True)
VALID_EVENT_TYPES = {
    "deploy", 
    "rollback", 
    "config_change", 
    "version_bump"
}

# Setup logger
logger = logging.getLogger("SLAIDeploymentLogger")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.FileHandler(LOG_FILE)
    formatter = logging.Formatter('%(message)s')  # JSON lines
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def _redact_sensitive(data: dict) -> dict:
    """Scrub sensitive values from details"""
    return {k: "**REDACTED**" if k in SENSITIVE_KEYS else v 
            for k, v in data.items()}

@retry(stop=stop_after_attempt(3), wait=wait_fixed(0.5))
def log_event(
    event_type: str,
    user: str,
    environment: str,
    branch: Optional[str] = None,
    version: Optional[str] = None,
    success: bool = True,
    details: Optional[Dict] = None
):
    if os.path.exists(LOG_FILE):
        if os.path.getsize(LOG_FILE) > MAX_LOG_SIZE:
            rotate_logs()    
    if event_type not in VALID_EVENT_TYPES:
        raise ValueError(f"Invalid event type: {event_type}. Valid types: {VALID_EVENT_TYPES}")

    try:
        system_metrics = {
            "cpu_percent": psutil.cpu_percent(),
            "ram_used": psutil.virtual_memory().used,
            "disk_free": psutil.disk_usage('/').free
        }
        """
        Logs a structured JSON event to the deployment log.
        """
        event = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "event_type": event_type,
            "user": user,
            "environment": environment,
            "branch": branch,
            "version": version,
            "success": success,
            "details": details or {},
            "system_metrics": system_metrics,
            "hostname": os.uname().nodename,
            "ip_address": socket.gethostbyname(socket.gethostname())
        }
        logger.info(json.dumps(event))
        print(f"[{event_type.upper()}] env={environment} | user={user} | branch={branch} | success={success}")

    except Exception as e:
        print(f"Critical logging failure: {str(e)}")
        raise

def rotate_logs():
    """Rotate logs when they reach max size"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archived_log = os.path.join(LOG_DIR, f"deployment_audit_{timestamp}.jsonl")
    os.rename(LOG_FILE, archived_log)

def read_logs(as_dicts=True, limit=100):
    log_event(
        event_type="log_access",
        user=os.getenv("USER", "unknown"),
        environment="audit",
        details={"action": "read_logs", "limit": limit}
    )
    """
    Reads the latest deployment log entries.

    Parameters:
        as_dicts: Return parsed JSON dicts or raw lines.
        limit: Number of most recent entries to return.

    Returns:
        List of deployment events.
    """
    if not os.path.exists(LOG_FILE):
        return []

    with open(LOG_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()[-limit:]

    return [json.loads(line) for line in lines] if as_dicts else lines

details = _redact_sensitive(details or {})
