import json
import os
from datetime import datetime

history_file = "deployment/deployment_history.json"

def add_history_entry(event: dict):
    """Append an event to the JSON history file."""
    if not os.path.exists(history_file):
        with open(history_file, 'w') as f:
            json.dump([], f)

    with open(history_file, 'r+') as f:
        data = json.load(f)
        data.append(event)
        f.seek(0)
        json.dump(data, f, indent=2)
        f.truncate()

def get_history():
    """Fetch full deployment history."""
    if not os.path.exists(history_file):
        return []

    with open(history_file, 'r') as f:
        return json.load(f)

def log_to_history(event_type: str, user: str, branch: str, version=None, success=True, details=None):
    """Convenience wrapper for logging to history."""
    event = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "event_type": event_type,
        "user": user,
        "branch": branch,
        "version": version,
        "success": success,
        "details": details or {}
    }
    add_history_entry(event)
