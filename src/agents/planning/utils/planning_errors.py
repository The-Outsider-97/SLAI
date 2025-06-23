
import os
import json
import time

from asyncio import Task
from typing import Any, Dict, List, Union

# Custom Exceptions
class AdjustmentError(Exception):
    """Exception for invalid plan adjustments."""
    def __init__(self, message: str, adjustment: Dict, conflict_details: Dict = None):
        super().__init__(message)
        self.adjustment = adjustment
        self.conflict_details = conflict_details or {}
        self.timestamp = time.time()

class ReplanningError(Exception):
    """Exception for failures in recovery planning processes."""
    def __init__(self, message: str, failed_task: Task, candidates: List = None):
        super().__init__(message)
        self.failed_task = failed_task
        self.candidates = candidates or []
        self.metadata = {
            "error_type": "replanning_failure",
            "timestamp": time.time()
        }

class TemporalViolation(Exception):
    """Base class for temporal constraint violations"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message)
        self.violation_details = kwargs

class AcademicPlanningError(Exception):
    """
    Custom exception for type violations and planning semantics.
    Loads additional structured error documentation from a JSON file.
    """
    _error_metadata_path = os.path.join(os.path.dirname(__file__),
                                        'templates/academic_planning_error.json')
    _metadata = None

    @classmethod
    def get_metadata(cls):
        """Lazily load and return the JSON metadata for this error class."""
        if cls._metadata is None:
            try:
                with open(cls._error_metadata_path, 'r', encoding='utf-8') as f:
                    cls._metadata = json.load(f)
            except Exception as e:
                cls._metadata = {"error": f"Failed to load metadata: {e}"}
        return cls._metadata

    def __init__(self, message=None):
        super().__init__(message or "An academic planning error occurred.")

class ResourceViolation(Exception):
    """Exception for unmet resource constraints"""
    def __init__(self, message: str, resource_type: str, requested: Any, available: Any):
        super().__init__(message)
        self.resource_type = resource_type
        self.requested = requested
        self.available = available
        self.metadata = self._generate_metadata()

    def _generate_metadata(self) -> Dict:
        return {
            "violation_type": "resource_constraint",
            "timestamp": time.time(),
            "resolution_strategies": [
                "resource_scaling",
                "task_repurposing",
                "priority_reallocation"
            ]
        }

class SafetyMarginError(ResourceViolation):
    """Exception for safety margin violations"""
    def __init__(self, message: str, resource_type: str, buffer_amount: float):
        super().__init__(message, required={}, available={})
        self.resource_type = resource_type
        self.buffer_amount = buffer_amount
        self.metadata['violation_type'] = 'safety_margin'
