import hashlib
import json
import random
import time
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

from src.agents.collaborative.utils.config_loader import get_config_section, load_global_config
from logs.logger import PrettyPrinter, get_logger

logger = get_logger("Collaborative Error")
printer = PrettyPrinter


class CollaborationErrorType(Enum):
    OVERLOAD = "Collaboration System Overload"
    NO_CAPABLE_AGENT = "No Capable Agent Found"
    ROUTING_FAILURE = "Task Routing Failure"
    DELEGATION_FAILURE = "Task Delegation Failure"
    REGISTRATION_FAILURE = "Agent Registration Failure"
    SHARED_MEMORY_FAILURE = "Shared Memory Access Failure"


class CollaborationError(Exception):
    def __init__(
        self,
        error_type: CollaborationErrorType,
        message: str,
        severity: str = "medium",
        context: Optional[Dict[str, Any]] = None,
        collaborative_agent_state: Optional[Dict[str, Any]] = None,
        remediation_guidance: Optional[str] = None,
    ):
        super().__init__(message)
        self.context = context or {}
        self.config = load_global_config()
        self.error_config = get_config_section("collaboration_error") or {}

        self.forensic_hash_algorithm = self.error_config.get("forensic_hash_algorithm", "sha256")
        self.error_id_hash_algorithm = self.error_config.get("error_id_hash_algorithm", "sha256")
        self.forensic_hash_salt = self.error_config.get("forensic_hash_salt", "collaboration")
        self.error_id_length = int(self.error_config.get("error_id_length", 12))
        self.report_format = self.error_config.get("report_format", "markdown")

        self.error_type = error_type
        self.severity = severity
        self.collaborative_agent_state = collaborative_agent_state or {}
        self.remediation_guidance = remediation_guidance
        self.timestamp = time.time()
        self.error_id = self._generate_error_id()
        self.forensic_hash = self._generate_forensic_hash()

    def _generate_error_id(self) -> str:
        algorithm = self.error_id_hash_algorithm
        length = self.error_id_length
        hash_func = getattr(hashlib, algorithm, hashlib.sha256)
        raw = f"{self.timestamp}{random.getrandbits(64)}".encode()
        return hash_func(raw).hexdigest()[:length]

    def _generate_forensic_hash(self) -> str:
        algorithm = self.forensic_hash_algorithm
        salt = self.forensic_hash_salt
        hash_func = getattr(hashlib, algorithm, hashlib.sha256)

        context_str = json.dumps(self.context, sort_keys=True, default=str)
        state_str = json.dumps(self.collaborative_agent_state, sort_keys=True, default=str)
        data = (
            f"{salt}{self.timestamp}{self.error_id}{str(self)}"
            f"{self.error_type.value}{self.severity}{context_str}{state_str}"
        ).encode()
        return hash_func(data).hexdigest()

    def to_audit_format(self) -> Dict[str, Any]:
        return {
            "error_id": self.error_id,
            "timestamp": self.timestamp,
            "error_type": self.error_type.value,
            "severity": self.severity,
            "message": str(self),
            "forensic_hash": self.forensic_hash,
            "context": self.context,
            "collaborative_agent_state_snapshot": self.collaborative_agent_state,
            "remediation_guidance": self.remediation_guidance,
        }

    def generate_report(self) -> str:
        if self.report_format == "json":
            return json.dumps(self.to_audit_format(), indent=2)

        audit_data = self.to_audit_format()
        dt = datetime.fromtimestamp(audit_data["timestamp"])
        formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")

        report = [
            "# Collaboration Incident Report",
            f"**Generated**: {formatted_time}",
            f"**Error ID**: `{audit_data['error_id']}`",
            f"**Error Type**: {audit_data['error_type']}",
            f"**Severity**: {audit_data['severity'].upper()}",
            "---",
            f"**Message**: {audit_data['message']}",
        ]

        if self.config.get("include_forensic_hash", True):
            report.append(f"**Forensic Hash**: `{audit_data['forensic_hash']}`")

        if self.config.get("include_context", True):
            context_str = json.dumps(audit_data["context"], indent=2)
            report.append("## Context Details")
            report.append(f"```json\n{context_str}\n```")

        if self.config.get("include_collaborative_agent_state", True):
            state_str = json.dumps(audit_data["collaborative_agent_state_snapshot"], indent=2)
            report.append("## Collaborative Agent State")
            report.append(f"```json\n{state_str}\n```")

        if self.config.get("include_remediation_guidance", True):
            report.append("## Remediation Guidance")
            report.append(audit_data["remediation_guidance"] or "No specific guidance provided.")

        return "\n".join(report)

    def __str__(self) -> str:
        return f"[{self.error_type.name} - {self.severity.upper()}] {super().__str__()}"


class OverloadError(CollaborationError):
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(
            CollaborationErrorType.OVERLOAD,
            message,
            severity="high",
            context=context,
            remediation_guidance="Scale out workers, reduce queue depth, or throttle incoming tasks.",
        )


class NoCapableAgentError(CollaborationError):
    def __init__(self, task_type: str, required_capabilities: Optional[list] = None):
        super().__init__(
            CollaborationErrorType.NO_CAPABLE_AGENT,
            f"No capable agent found for task type '{task_type}'.",
            severity="medium",
            context={"task_type": task_type, "required_capabilities": required_capabilities or []},
            remediation_guidance="Register a compatible agent or relax task capability requirements.",
        )


class RoutingFailureError(CollaborationError):
    def __init__(self, task_type: str, reason: str):
        super().__init__(
            CollaborationErrorType.ROUTING_FAILURE,
            f"Routing failed for task type '{task_type}': {reason}",
            severity="high",
            context={"task_type": task_type, "reason": reason},
            remediation_guidance="Review routing policy, fallback order, and agent runtime health.",
        )


if __name__ == "__main__":
    print("\n=== Running Collaboration Error Tests ===\n")

    overload = OverloadError(
        "System load exceeded (120/100)",
        context={"current_load": 120, "max_load": 100},
    )
    assert overload.error_type == CollaborationErrorType.OVERLOAD
    assert overload.error_id
    assert overload.forensic_hash
    assert "System load exceeded" in str(overload)

    routing = RoutingFailureError("translate", "all workers failed")
    routing_report = routing.generate_report()
    assert "Routing failed" in routing_report
    assert "Collaboration Incident Report" in routing_report

    no_agent = NoCapableAgentError("summarize", ["nlp", "summarization"])
    audit = no_agent.to_audit_format()
    assert audit["error_type"] == CollaborationErrorType.NO_CAPABLE_AGENT.value
    assert audit["context"]["required_capabilities"] == ["nlp", "summarization"]

    print("All collaboration_error.py tests passed.")
