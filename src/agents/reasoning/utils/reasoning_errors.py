"""Reasoning subsystem exception hierarchy.

This module provides a comprehensive and centralized error taxonomy for all
reasoning layers, including:
- `reasoning_agent.py` orchestration lifecycle
- symbolic inference (`rule_engine.py`, `types/`)
- probabilistic and hybrid inference (`probabilistic_models.py`,
  `hybrid_probabilistic_models.py`, `utils/model_compute.py`,
  `utils/adaptive_circuit.py`, `utils/pgmpy_wrapper.py`)
- validation and consistency (`validation.py`)
- persistence and memory (`reasoning_memory.py`)
- configuration and resource loading (`utils/config_loader.py`, templates/networks)

Design constraints:
- Contains only error classes and metadata helpers.
- Must not import helper functions from other reasoning utility modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Reasoning Error")
printer = PrettyPrinter()


@dataclass
class ReasoningError(Exception):
    """Base class for all reasoning failures.

    Attributes:
        message: Human-readable failure explanation.
        code: Stable machine-readable identifier.
        context: Structured metadata to aid debugging/telemetry.
        cause: Optional underlying exception for error chaining.
        recoverable: Indicates if caller may continue safely.
    """
    message: str
    code: str = "reasoning_error"
    context: Dict[str, Any] = field(default_factory=dict)
    cause: Optional[BaseException] = None
    recoverable: bool = False

    def __post_init__(self) -> None:
        Exception.__init__(self, self.message)
        # Automatically capture caller info if needed
        if self.cause and not self.__cause__:
            self.__cause__ = self.cause

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"

    def to_payload(self) -> Dict[str, Any]:
        payload = {
            "error_type": self.__class__.__name__,
            "code": self.code,
            "message": self.message,
            "recoverable": self.recoverable,
            "context": self.context.copy(),
        }
        if self.cause:
            payload["cause"] = f"{type(self.cause).__name__}: {self.cause}"
        return payload

    def with_context(self, **kwargs: Any) -> "ReasoningError":
        """Return a new error with merged context (fluent API)."""
        self.context.update(kwargs)
        return self

    @classmethod
    def chain(cls, message: str, cause: BaseException, **context: Any) -> "ReasoningError":
        """Create an error that chains another exception."""
        return cls(message=message, cause=cause, context=context)

    # Optional: add rich traceback formatting
    def format_trace(self) -> str:
        import traceback
        lines = [f"{self.__class__.__name__}: {self.message}"]
        if self.context:
            lines.append("Context:")
            for k, v in self.context.items():
                lines.append(f"  {k}: {v}")
        if self.cause:
            lines.append("Caused by:")
            lines.extend(traceback.format_exception_only(type(self.cause), self.cause))
        return "\n".join(lines)

# ---------------------------------------------------------------------------
# Input and schema validation errors
# ---------------------------------------------------------------------------
@dataclass
class ReasoningValidationError(ReasoningError):
    """Raised when reasoning inputs or schema-level constraints are invalid."""

    code: str = "reasoning_validation_error"
    recoverable: bool = True


@dataclass
class FactNormalizationError(ReasoningValidationError):
    """Raised when a fact cannot be transformed into canonical (s, p, o) form."""

    code: str = "fact_normalization_error"


@dataclass
class RuleDefinitionError(ReasoningValidationError):
    """Raised when rule callable, signature, weight, or metadata is invalid."""

    code: str = "rule_definition_error"


@dataclass
class ReasoningTypeError(ReasoningValidationError):
    """Raised when reasoning strategy/type creation fails or is unsupported."""

    code: str = "reasoning_type_error"


@dataclass
class ConfidenceBoundsError(ReasoningValidationError):
    """Raised when confidence values violate expected numeric interval constraints."""

    code: str = "confidence_bounds_error"


# ---------------------------------------------------------------------------
# Resource/configuration/loading errors
# ---------------------------------------------------------------------------
@dataclass
class ReasoningConfigurationError(ReasoningError):
    """Raised for invalid or missing reasoning configuration state."""

    code: str = "reasoning_configuration_error"


@dataclass
class ConfigLoadError(ReasoningConfigurationError):
    """Raised when config files fail to load, parse, or type-check."""

    code: str = "config_load_error"


@dataclass
class ResourceLoadError(ReasoningError):
    """Raised when templates, networks, lexicons, or rule assets cannot load."""

    code: str = "resource_load_error"


@dataclass
class DependencyUnavailableError(ReasoningError):
    """Raised when optional runtime dependencies are unavailable."""

    code: str = "dependency_unavailable_error"
    recoverable: bool = True


# ---------------------------------------------------------------------------
# Knowledge base / memory / state errors
# ---------------------------------------------------------------------------
@dataclass
class KnowledgeBaseError(ReasoningError):
    """Base class for knowledge base mutation or consistency failures."""

    code: str = "knowledge_base_error"


@dataclass
class ContradictionError(KnowledgeBaseError):
    """Raised when a fact violates contradiction constraints in knowledge state."""

    code: str = "contradiction_error"
    recoverable: bool = True


@dataclass
class RedundancyError(KnowledgeBaseError):
    """Raised when asserted facts are redundant under configured policy."""

    code: str = "redundancy_error"
    recoverable: bool = True


@dataclass
class KnowledgePersistenceError(KnowledgeBaseError):
    """Raised when knowledge/memory state cannot be persisted or restored."""

    code: str = "knowledge_persistence_error"


@dataclass
class MemoryOperationError(ReasoningError):
    """Raised when reasoning memory append/query/retrieval operations fail."""

    code: str = "memory_operation_error"


# ---------------------------------------------------------------------------
# Inference execution and runtime errors
# ---------------------------------------------------------------------------
@dataclass
class InferenceExecutionError(ReasoningError):
    """Raised when forward/backward/hybrid inference execution fails."""

    code: str = "inference_execution_error"


@dataclass
class RuleExecutionError(InferenceExecutionError):
    """Raised when an individual symbolic rule fails during evaluation."""

    code: str = "rule_execution_error"
    recoverable: bool = True


@dataclass
class CircularReasoningError(InferenceExecutionError):
    """Raised when circular dependency depth exceeds tolerated bounds."""

    code: str = "circular_reasoning_error"


@dataclass
class ReasoningTimeoutError(InferenceExecutionError):
    """Raised when bounded reasoning procedure exceeds runtime limits."""

    code: str = "reasoning_timeout_error"


@dataclass
class ConvergenceError(InferenceExecutionError):
    """Raised when iterative inference does not converge under configured policy."""

    code: str = "convergence_error"


# ---------------------------------------------------------------------------
# Probabilistic / hybrid model errors
# ---------------------------------------------------------------------------
@dataclass
class ProbabilisticModelError(ReasoningError):
    """Base class for probabilistic model construction and inference failures."""

    code: str = "probabilistic_model_error"


@dataclass
class ModelInitializationError(ProbabilisticModelError):
    """Raised when probabilistic model components fail initialization."""

    code: str = "model_initialization_error"


@dataclass
class ModelInferenceError(ProbabilisticModelError):
    """Raised when probabilistic query, MAP, marginal, or hybrid inference fails."""

    code: str = "model_inference_error"


@dataclass
class CircuitConstraintError(ProbabilisticModelError):
    """Raised when probabilistic circuit structural/semantic constraints fail."""
    code: str = "circuit_constraint_error"

    def describe_overlap(self) -> Optional[str]:
        """If the context contains 'violations', return a human-readable description."""
        violations = self.context.get("violations")
        if violations:
            return "; ".join(violations)
        return None


@dataclass
class TrainingError(ProbabilisticModelError):
    """Raised when model revision/training/update operations fail."""

    code: str = "training_error"


# ---------------------------------------------------------------------------
# Validation and external interaction errors
# ---------------------------------------------------------------------------
@dataclass
class ValidationEngineError(ReasoningError):
    """Raised when validation stages fail to execute or aggregate."""

    code: str = "validation_engine_error"


@dataclass
class ExternalServiceError(ReasoningError):
    """Raised when external validators/services used by reasoning fail."""

    code: str = "external_service_error"
    recoverable: bool = True


@dataclass
class AgentLifecycleError(ReasoningError):
    """Raised when reasoning agent lifecycle transitions fail."""

    code: str = "agent_lifecycle_error"