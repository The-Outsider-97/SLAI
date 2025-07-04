
import os
import json
import time
import hashlib

from enum import Enum
from typing import Dict, Any, Optional, List

class EvaluationErrorType(Enum):
    """Error types specific to evaluation processes"""
    METRIC_CALCULATION = "Metric Calculation Failure"
    REPORT_GENERATION = "Report Generation Failure"
    CONFIG_LOAD = "Configuration Loading Error"
    VALIDATION_FAILURE = "Validation Rule Violation"
    DATA_INTEGRITY = "Evaluation Data Integrity Error"
    MEMORY_ACCESS = "Evaluation Memory Access Failure"
    VISUALIZATION = "Result Visualization Error"
    THRESHOLD_VIOLATION = "Quality Threshold Violation"
    COMPARISON_FAILURE = "Comparative Analysis Failure"
    TEMPLATE_ERROR = "Report Template Processing Error"

class EvaluationError(Exception):
    """Base exception for evaluation agent errors with forensic capabilities"""
    def __init__(
        self,
        error_type: EvaluationErrorType,
        message: str,
        severity: str = "medium",
        context: Optional[Dict[str, Any]] = None,
        agent_state: Optional[Dict] = None,
        remediation: Optional[str] = None
    ):
        super().__init__(message)
        self.error_type = error_type
        self.severity = severity
        self.context = context or {}
        self.agent_state = agent_state or {}
        self.remediation = remediation
        self.timestamp = time.time()
        self.error_id = self._generate_error_id()
        self.forensic_hash = self._generate_forensic_hash()

    def _generate_error_id(self) -> str:
        """Generate unique error ID using context and timestamp"""
        unique_str = f"{self.timestamp}{self.error_type.value}{json.dumps(self.context)}"
        return hashlib.sha256(unique_str.encode()).hexdigest()[:12]

    def _generate_forensic_hash(self) -> str:
        """Create verifiable hash of error context"""
        data = {
            "timestamp": self.timestamp,
            "error_id": self.error_id,
            "context": self.context,
            "agent_state": self.agent_state
        }
        return hashlib.sha3_256(json.dumps(data).encode()).hexdigest()

    def to_audit_dict(self) -> Dict[str, Any]:
        """Structured representation for logging and auditing"""
        return {
            "error_id": self.error_id,
            "type": self.error_type.value,
            "severity": self.severity,
            "message": str(self),
            "timestamp": self.timestamp,
            "forensic_hash": self.forensic_hash,
            "context": self.context,
            "agent_state_snapshot": self.agent_state,
            "remediation": self.remediation
        }

# Specific Error Classes
class MetricCalculationError(EvaluationError):
    """Failure in metric calculation process"""
    def __init__(self, metric_name: str, inputs: Any, reason: str):
        super().__init__(
            EvaluationErrorType.METRIC_CALCULATION,
            f"Metric calculation failed for '{metric_name}': {reason}",
            severity="high",
            context={
                "metric": metric_name,
                "input_data": str(inputs)[:200] + "..." if len(str(inputs)) > 200 else str(inputs),
                "failure_reason": reason
            },
            remediation="Verify input data format and calculation parameters"
        )

class ReportGenerationError(EvaluationError):
    """Failure in report generation process"""
    def __init__(self, report_type: str, template: str, error_details: str):
        super().__init__(
            EvaluationErrorType.REPORT_GENERATION,
            f"Report generation failed for {report_type} using template {template}",
            severity="medium",
            context={
                "report_type": report_type,
                "template": template,
                "error": error_details
            },
            remediation="Check template syntax and data availability"
        )

class ConfigLoadError(EvaluationError):
    """Failure in configuration loading"""
    def __init__(self, config_path: str, section: str, error_details: str):
        super().__init__(
            EvaluationErrorType.CONFIG_LOAD,
            f"Config loading failed for {section} in {config_path}",
            severity="critical",
            context={
                "config_path": config_path,
                "section": section,
                "error": error_details
            },
            remediation="Validate configuration file structure and permissions"
        )

class ValidationFailureError(EvaluationError):
    """Violation of evaluation validation rules"""
    def __init__(self, rule_name: str, data: Any, expected: Any):
        super().__init__(
            EvaluationErrorType.VALIDATION_FAILURE,
            f"Validation rule '{rule_name}' violated",
            severity="high",
            context={
                "rule": rule_name,
                "actual_value": str(data),
                "expected_value": str(expected)
            },
            remediation="Review data sources and preprocessing steps"
        )

class ThresholdViolationError(EvaluationError):
    """Violation of quality thresholds"""
    def __init__(self, metric: str, value: float, threshold: float):
        super().__init__(
            EvaluationErrorType.THRESHOLD_VIOLATION,
            f"Quality threshold violation: {metric} ({value:.2f} < {threshold:.2f})",
            severity="high",
            context={
                "metric": metric,
                "actual_value": value,
                "threshold": threshold
            },
            remediation="Investigate performance degradation and optimization opportunities"
        )

class MemoryAccessError(EvaluationError):
    """Failure in evaluation memory access"""
    def __init__(self, operation: str, key: str, error_details: str):
        super().__init__(
            EvaluationErrorType.MEMORY_ACCESS,
            f"Memory {operation} failed for key '{key}'",
            severity="medium",
            context={
                "operation": operation,
                "key": key,
                "error": error_details
            },
            remediation="Check memory storage connection and serialization formats"
        )

class VisualizationError(EvaluationError):
    """Failure in results visualization"""
    def __init__(self, chart_type: str, data: Any, error_details: str):
        super().__init__(
            EvaluationErrorType.VISUALIZATION,
            f"Visualization failed for {chart_type} chart",
            severity="low",
            context={
                "chart_type": chart_type,
                "data_sample": str(data)[:200] + "..." if len(str(data)) > 200 else str(data),
                "error": error_details
            },
            remediation="Verify data dimensions and visualization library dependencies"
        )

class ComparisonError(EvaluationError):
    """Failure in comparative analysis"""
    def __init__(self, baseline: str, current: str, error_details: str):
        super().__init__(
            EvaluationErrorType.COMPARISON_FAILURE,
            f"Comparison failed between {baseline} and {current}",
            severity="medium",
            context={
                "baseline": baseline,
                "current": current,
                "error": error_details
            },
            remediation="Check baseline data availability and comparison algorithm"
        )

class TemplateError(EvaluationError):
    """Failure in report template processing"""
    def __init__(self, template_path: str, error_details: str):
        super().__init__(
            EvaluationErrorType.TEMPLATE_ERROR,
            f"Template processing failed for {template_path}",
            severity="high",
            context={
                "template_path": template_path,
                "error": error_details
            },
            remediation="Validate template syntax and data placeholders"
        )

class OperationalError(EvaluationError):
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(
            error_type=EvaluationErrorType.MEMORY_ACCESS,
            message=message,
            severity="critical",
            context=context,
            remediation="Inspect system configuration and runtime dependencies"
        )

class CertificationError(EvaluationError):
    """Custom exception for certification failures"""
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(
            error_type=EvaluationErrorType.VALIDATION_FAILURE,
            message=message,
            severity="high",
            context=context or {},
            remediation="Review architectural safety guarantees and compliance metrics"
        )