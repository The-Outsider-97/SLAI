from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict

class FailureSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class HandlerError(Exception):
    message: str
    error_type: str = "HandlerError"
    severity: FailureSeverity = FailureSeverity.MEDIUM
    retryable: bool = False
    context: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.message

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_type": self.error_type,
            "message": self.message,
            "severity": self.severity.value,
            "retryable": self.retryable,
            "context": self.context,
        }
