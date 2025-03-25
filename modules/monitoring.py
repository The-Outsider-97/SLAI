import logging
import time
import sys
import os
import torch
from datetime import datetime

logger = logging.getLogger('SafeAI.Monitoring')
logger.setLevel(logging.INFO)

class Monitoring:
    def __init__(self, shared_memory=None, alert_config=None, output_path="logs/metrics_log.jsonl"):
        """
        Initialize Monitoring system.

        shared_memory: for cross-module access
        alert_config: dict of threshold triggers (e.g., {"accuracy": 0.8})
        output_path: where to log metrics over time
        """
        self.shared_memory = shared_memory
        self.alert_config = alert_config or {}
        self.output_path = output_path

        self.history = []
        with open(self.output_path, "a") as f:
            f.write("")  # touch file

    def record(self, module_name, metrics: dict):
        """
        Logs a new metric entry.
        """
        timestamp = datetime.utcnow().isoformat()
        entry = {
            "timestamp": timestamp,
            "module": module_name,
            "metrics": metrics
        }

        self.history.append(entry)
        self._write_to_file(entry)

        if self.shared_memory:
            self.shared_memory.set(f"monitoring_{module_name}", entry)

        self._check_alerts(module_name, metrics)

    def _write_to_file(self, entry):
        try:
            with open(self.output_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error(f"Failed to write monitoring log: {e}")

    def _check_alerts(self, module, metrics):
        alerts_triggered = []

        for key, threshold in self.alert_config.items():
            if key in metrics:
                value = metrics[key]
                if value < threshold:
                    alert_msg = f"[ALERT] {module}: {key}={value:.4f} fell below threshold {threshold:.4f}"
                    logger.warning(alert_msg)
                    alerts_triggered.append(alert_msg)

        if self.shared_memory and alerts_triggered:
            self.shared_memory.set(f"alerts_{module}", alerts_triggered)

    def get_latest(self, module_name):
        for entry in reversed(self.history):
            if entry["module"] == module_name:
                return entry
        return None

    def print_summary(self):
        print("\n=== Monitoring Summary ===")
        for entry in self.history[-5:]:  # Last 5 entries
            ts = entry["timestamp"]
            mod = entry["module"]
            print(f"[{ts}] {mod} → {entry['metrics']}")
