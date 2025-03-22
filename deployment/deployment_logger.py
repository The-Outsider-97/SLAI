import logging
import json
import os
from datetime import datetime
from typing import Optional, Dict

# Configure basic logger
logger = logging.getLogger("DeploymentAuditLogger")
logger.setLevel(logging.DEBUG)

# File log handler
log_file = "deployment/deployment_audit.log"
os.makedirs(os.path.dirname(log_file), exist_ok=True)

file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)


def log_event(event_type: str,
              user: str,
              branch: str,
              version: Optional[str] = None,
              success: bool = True,
              details: Optional[Dict] = None):
    """
    Log a structured event to deployment audit log.

    Parameters:
    - event_type (str): Type of event ("deploy", "rollback", "merge", etc.)
    - user (str): User or process initiating the action.
    - branch (str): Git branch involved.
    - version (str): Optional git tag/version.
    - success (bool): Whether the action was successful.
    - details (dict): Optional additional metadata.
    """

    log_entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "event_type": event_type,
        "user": user,
        "branch": branch,
        "version": version,
        "success": success,
        "details": details or {}
    }

    logger.info(json.dumps(log_entry, indent=None))
    print(f"Audit Log: {event_type.upper()} | User: {user} | Branch: {branch} | Version: {version} | Success: {success}")
