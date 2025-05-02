
import hashlib
import time

from enum import Enum
from typing import Optional, Dict, Any

class SecurityErrorType(Enum):
    CONTENT_VIOLATION = "Content Policy Violation"
    ACCESS_VIOLATION = "Unauthorized Access Attempt"
    DATA_VIOLATION = "Sensitive Data Exposure"
    EXECUTION_VIOLATION = "Unsafe Execution Attempt"
    SYSTEM_VIOLATION = "System Integrity Violation"

class SecurityError(Exception):
    """Base security exception with threat intelligence integration"""
    
    def __init__(self, 
                 error_type: SecurityErrorType,
                 message: str,
                 severity: str = "high",
                 context: Optional[Dict[str, Any]] = None,
                 safety_agent_state: Optional[Dict] = None):
        super().__init__(message)
        self.error_type = error_type
        self.severity = severity
        self.context = context or {}
        self.safety_agent_state = safety_agent_state or {}
        self.timestamp = time.time()
        
        # Generate forensic hash
        self.forensic_hash = hashlib.sha256(
            f"{self.timestamp}{message}{error_type.value}".encode()
        ).hexdigest()

    def to_audit_format(self):
        return {
            "timestamp": self.timestamp,
            "type": self.error_type.value,
            "severity": self.severity,
            "message": str(self),
            "forensic_hash": self.forensic_hash,
            "context": self.context,
            "safety_state": self.safety_agent_state
        }

# Concrete error classes
class PrivacyViolationError(SecurityError):
    def __init__(self, pattern: str, content: str):
        super().__init__(
            SecurityErrorType.DATA_VIOLATION,
            f"PII detection triggered by pattern: {pattern}",
            severity="critical",
            context={
                "detected_pattern": pattern,
                "content_sample": content[:100]
            }
        )

class ToxicContentError(SecurityError):
    def __init__(self, pattern: str, content: str):
        super().__init__(
            SecurityErrorType.CONTENT_VIOLATION,
            f"Toxic content detected: {pattern}",
            context={
                "toxic_pattern": pattern,
                "content_sample": content[:100]
            }
        )

class UnauthorizedAccessError(SecurityError):
    def __init__(self, resource: str, policy: str):
        super().__init__(
            SecurityErrorType.ACCESS_VIOLATION,
            f"Unauthorized access to {resource} violates {policy}",
            context={
                "resource": resource,
                "violated_policy": policy
            }
        )

class UnsafeExecutionError(SecurityError):
    def __init__(self, operation: str, risk_score: float):
        super().__init__(
            SecurityErrorType.EXECUTION_VIOLATION,
            f"Blocked unsafe {operation} (risk score: {risk_score})",
            severity="critical" if risk_score > 0.7 else "high",
            context={
                "operation": operation,
                "risk_score": risk_score
            }
        )

class SystemIntegrityError(SecurityError):
    def __init__(self, operation: str, affected_path: str):
        super().__init__(
            SecurityErrorType.SYSTEM_VIOLATION,
            f"System integrity violation during {operation}",
            severity="critical",
            context={
                "operation": operation,
                "affected_path": affected_path
            }
        )
