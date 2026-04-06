"""Production-ready error taxonomy and handling utilities for an autonomous financial agent.

This module is designed for agents that orchestrate:
- market and alternative data ingestion
- feature engineering and trend monitoring
- sentiment / NLP pipelines
- adaptive learning and model lifecycle operations
- portfolio optimization and risk controls
- order routing and trade execution

Key goals:
- Standardize failures across heterogeneous subsystems.
- Preserve rich operational context without leaking sensitive internals.
- Distinguish retryable vs. terminal failures.
- Make failures machine-actionable for orchestrators, circuit breakers, logging,
  observability, alerting, and post-trade auditing.
- Remain framework-agnostic and lightweight.
"""

from __future__ import annotations

import functools
import logging
import socket
import traceback
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, Iterable, Iterator, Mapping, MutableMapping, Optional, Tuple, Type, Union

try:
    import requests
    from requests import exceptions as requests_exceptions
except Exception:  # pragma: no cover - optional dependency
    requests = None
    requests_exceptions = None

logger = logging.getLogger(__name__)

JSONDict = Dict[str, Any]
ContextInput = Optional[Union["ErrorContext", Mapping[str, Any]]]


class ErrorSeverity(str, Enum):
    """Operational severity used for alerting and routing."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ErrorCategory(str, Enum):
    """Top-level category for system-wide classification."""

    CONFIGURATION = "configuration"
    VALIDATION = "validation"
    DATA = "data"
    CONNECTIVITY = "connectivity"
    PROVIDER = "provider"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    RATE_LIMIT = "rate_limit"
    PARSING = "parsing"
    SENTIMENT = "sentiment"
    FEATURE_ENGINEERING = "feature_engineering"
    MODEL = "model"
    ADAPTIVE_LEARNING = "adaptive_learning"
    TREND_MONITORING = "trend_monitoring"
    PORTFOLIO = "portfolio"
    RISK = "risk"
    EXECUTION = "execution"
    COMPLIANCE = "compliance"
    PERSISTENCE = "persistence"
    INFRASTRUCTURE = "infrastructure"
    INTERNAL = "internal"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


class RecoveryAction(str, Enum):
    """Recommended next action for the orchestrator."""

    RETRY = "retry"
    BACKOFF_AND_RETRY = "backoff_and_retry"
    REFRESH_STATE = "refresh_state"
    FALLBACK_PROVIDER = "fallback_provider"
    SKIP_SYMBOL = "skip_symbol"
    SKIP_BATCH = "skip_batch"
    PAUSE_STRATEGY = "pause_strategy"
    HALT_TRADING = "halt_trading"
    REQUIRE_HUMAN_REVIEW = "require_human_review"
    CORRECT_CONFIGURATION = "correct_configuration"
    REAUTHENTICATE = "reauthenticate"
    REDUCE_LOAD = "reduce_load"
    DROP_TO_SAFE_MODE = "drop_to_safe_mode"
    NO_RETRY = "no_retry"


@dataclass(slots=True)
class ErrorContext:
    """Structured execution context attached to failures.

    Fields are intentionally broad so the same object can be reused across
    ingestion, research, training, monitoring, and execution pathways.
    """

    agent_id: Optional[str] = None
    workflow_id: Optional[str] = None
    request_id: Optional[str] = None
    correlation_id: Optional[str] = None
    component: Optional[str] = None
    operation: Optional[str] = None
    phase: Optional[str] = None
    symbol: Optional[str] = None
    symbols: Optional[Tuple[str, ...]] = None
    venue: Optional[str] = None
    provider: Optional[str] = None
    endpoint: Optional[str] = None
    strategy_id: Optional[str] = None
    portfolio_id: Optional[str] = None
    order_id: Optional[str] = None
    account_id: Optional[str] = None
    model_name: Optional[str] = None
    dataset_name: Optional[str] = None
    feature_set: Optional[str] = None
    timeframe: Optional[str] = None
    batch_id: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 0
    environment: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ErrorContext":
        """Build an ErrorContext from arbitrary mappings.

        Unknown keys are placed into metadata instead of being dropped.
        """
        valid_fields = cls.__dataclass_fields__.keys()
        init_payload: Dict[str, Any] = {}
        metadata: Dict[str, Any] = dict(data.get("metadata", {}) or {})

        for key, value in data.items():
            if key in valid_fields:
                init_payload[key] = value
            elif key != "metadata":
                metadata[key] = value

        if "symbols" in init_payload and init_payload["symbols"] is not None:
            init_payload["symbols"] = tuple(init_payload["symbols"])
        init_payload["metadata"] = metadata
        return cls(**init_payload)

    def merge(self, other: ContextInput) -> "ErrorContext":
        """Merge another context into this one, preferring non-null values from other."""
        if other is None:
            return self
        other_ctx = ensure_error_context(other)
        base = asdict(self)
        incoming = asdict(other_ctx)
        merged: Dict[str, Any] = {}
        for key, value in base.items():
            if key == "metadata":
                merged[key] = {**(value or {}), **(incoming.get(key) or {})}
            else:
                merged[key] = incoming.get(key) if incoming.get(key) is not None else value
        return ErrorContext.from_mapping(merged)

    def to_dict(self) -> JSONDict:
        payload = asdict(self)
        if self.symbols is not None:
            payload["symbols"] = list(self.symbols)
        return payload


@dataclass(slots=True)
class ErrorReport:
    """Serializable report shape for logs, APIs, alerts, and audit trails."""

    timestamp: str
    error_name: str
    error_code: str
    message: str
    category: str
    severity: str
    retryable: bool
    safe_to_continue: bool
    recovery_action: str
    transient: bool
    tags: Tuple[str, ...]
    context: JSONDict
    details: JSONDict
    cause_name: Optional[str] = None
    cause_message: Optional[str] = None
    stacktrace: Optional[str] = None

    def to_dict(self) -> JSONDict:
        return {
            "timestamp": self.timestamp,
            "error_name": self.error_name,
            "error_code": self.error_code,
            "message": self.message,
            "category": self.category,
            "severity": self.severity,
            "retryable": self.retryable,
            "safe_to_continue": self.safe_to_continue,
            "recovery_action": self.recovery_action,
            "transient": self.transient,
            "tags": list(self.tags),
            "context": self.context,
            "details": self.details,
            "cause_name": self.cause_name,
            "cause_message": self.cause_message,
            "stacktrace": self.stacktrace,
        }


class FinancialAgentError(Exception):
    """Base exception for the financial agent.

    All domain-specific exceptions should inherit from this class so the agent's
    orchestrator can handle failures consistently.
    """

    default_code = "FIN-0000"
    default_category = ErrorCategory.INTERNAL
    default_severity = ErrorSeverity.ERROR
    default_retryable = False
    default_safe_to_continue = False
    default_transient = False
    default_recovery_action = RecoveryAction.NO_RETRY

    def __init__(
        self,
        message: str,
        *,
        code: Optional[str] = None,
        category: Optional[ErrorCategory] = None,
        severity: Optional[ErrorSeverity] = None,
        retryable: Optional[bool] = None,
        safe_to_continue: Optional[bool] = None,
        transient: Optional[bool] = None,
        recovery_action: Optional[RecoveryAction] = None,
        context: ContextInput = None,
        details: Optional[Mapping[str, Any]] = None,
        cause: Optional[BaseException] = None,
        tags: Optional[Iterable[str]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code or self.default_code
        self.category = category or self.default_category
        self.severity = severity or self.default_severity
        self.retryable = self.default_retryable if retryable is None else retryable
        self.safe_to_continue = (
            self.default_safe_to_continue if safe_to_continue is None else safe_to_continue
        )
        self.transient = self.default_transient if transient is None else transient
        self.recovery_action = recovery_action or self.default_recovery_action
        self.context = ensure_error_context(context)
        self.details: Dict[str, Any] = dict(details or {})
        self.cause = cause
        self.tags = tuple(dict.fromkeys(tags or ()))

    def with_context(self, extra: ContextInput) -> "FinancialAgentError":
        self.context = self.context.merge(extra)
        return self

    def add_details(self, **details: Any) -> "FinancialAgentError":
        self.details.update(details)
        return self

    def to_report(self, *, include_traceback: bool = False) -> ErrorReport:
        cause_name = type(self.cause).__name__ if self.cause else None
        cause_message = str(self.cause) if self.cause else None
        stacktrace = None
        if include_traceback and self.__traceback__ is not None:
            stacktrace = "".join(traceback.format_exception(type(self), self, self.__traceback__))

        return ErrorReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            error_name=type(self).__name__,
            error_code=self.code,
            message=self.message,
            category=self.category.value,
            severity=self.severity.value,
            retryable=self.retryable,
            safe_to_continue=self.safe_to_continue,
            recovery_action=self.recovery_action.value,
            transient=self.transient,
            tags=self.tags,
            context=self.context.to_dict(),
            details=dict(self.details),
            cause_name=cause_name,
            cause_message=cause_message,
            stacktrace=stacktrace,
        )

    def to_dict(self, *, include_traceback: bool = False) -> JSONDict:
        return self.to_report(include_traceback=include_traceback).to_dict()

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"


# ---------------------------------------------------------------------------
# Configuration and validation
# ---------------------------------------------------------------------------


class ConfigurationError(FinancialAgentError):
    default_code = "FIN-1000"
    default_category = ErrorCategory.CONFIGURATION
    default_severity = ErrorSeverity.CRITICAL
    default_safe_to_continue = False
    default_recovery_action = RecoveryAction.CORRECT_CONFIGURATION


class MissingConfigurationError(ConfigurationError):
    default_code = "FIN-1001"


class InvalidConfigurationError(ConfigurationError):
    default_code = "FIN-1002"


class ValidationError(FinancialAgentError):
    default_code = "FIN-1100"
    default_category = ErrorCategory.VALIDATION
    default_severity = ErrorSeverity.ERROR
    default_safe_to_continue = True
    default_recovery_action = RecoveryAction.SKIP_BATCH


class InvalidSymbolError(ValidationError):
    default_code = "FIN-1101"
    default_recovery_action = RecoveryAction.SKIP_SYMBOL


class InvalidOrderRequestError(ValidationError):
    default_code = "FIN-1102"
    default_safe_to_continue = False
    default_recovery_action = RecoveryAction.REQUIRE_HUMAN_REVIEW


class UnsupportedAssetClassError(ValidationError):
    default_code = "FIN-1103"


# ---------------------------------------------------------------------------
# Data ingestion and provider failures
# ---------------------------------------------------------------------------


class DataError(FinancialAgentError):
    default_code = "FIN-2000"
    default_category = ErrorCategory.DATA
    default_severity = ErrorSeverity.ERROR
    default_safe_to_continue = True
    default_recovery_action = RecoveryAction.SKIP_SYMBOL


class DataUnavailableError(DataError):
    default_code = "FIN-2001"
    default_retryable = True
    default_transient = True
    default_recovery_action = RecoveryAction.FALLBACK_PROVIDER


class DataStalenessError(DataError):
    default_code = "FIN-2002"
    default_severity = ErrorSeverity.CRITICAL
    default_safe_to_continue = False
    default_recovery_action = RecoveryAction.DROP_TO_SAFE_MODE


class MarketDataGapError(DataError):
    default_code = "FIN-2003"
    default_retryable = True
    default_transient = True
    default_recovery_action = RecoveryAction.REFRESH_STATE


class PartialDataError(DataError):
    default_code = "FIN-2004"
    default_safe_to_continue = True
    default_recovery_action = RecoveryAction.SKIP_BATCH


class ProviderError(FinancialAgentError):
    default_code = "FIN-2100"
    default_category = ErrorCategory.PROVIDER
    default_severity = ErrorSeverity.ERROR
    default_retryable = True
    default_transient = True
    default_safe_to_continue = True
    default_recovery_action = RecoveryAction.FALLBACK_PROVIDER


class ProviderTimeoutError(ProviderError):
    default_code = "FIN-2101"
    default_category = ErrorCategory.TIMEOUT
    default_recovery_action = RecoveryAction.BACKOFF_AND_RETRY


class ProviderRateLimitError(ProviderError):
    default_code = "FIN-2102"
    default_category = ErrorCategory.RATE_LIMIT
    default_recovery_action = RecoveryAction.REDUCE_LOAD


class ProviderAuthenticationError(ProviderError):
    default_code = "FIN-2103"
    default_category = ErrorCategory.AUTHENTICATION
    default_retryable = False
    default_transient = False
    default_safe_to_continue = False
    default_recovery_action = RecoveryAction.REAUTHENTICATE


class ProviderAuthorizationError(ProviderError):
    default_code = "FIN-2104"
    default_category = ErrorCategory.AUTHORIZATION
    default_retryable = False
    default_transient = False
    default_safe_to_continue = False
    default_recovery_action = RecoveryAction.REQUIRE_HUMAN_REVIEW


class ProviderParsingError(ProviderError):
    default_code = "FIN-2105"
    default_category = ErrorCategory.PARSING
    default_retryable = False
    default_transient = False
    default_recovery_action = RecoveryAction.FALLBACK_PROVIDER


class ConnectivityError(FinancialAgentError):
    default_code = "FIN-2200"
    default_category = ErrorCategory.CONNECTIVITY
    default_severity = ErrorSeverity.ERROR
    default_retryable = True
    default_transient = True
    default_safe_to_continue = True
    default_recovery_action = RecoveryAction.BACKOFF_AND_RETRY


class NetworkTimeoutError(ConnectivityError):
    default_code = "FIN-2201"
    default_category = ErrorCategory.TIMEOUT


class DNSResolutionError(ConnectivityError):
    default_code = "FIN-2202"


class ServiceUnavailableError(ConnectivityError):
    default_code = "FIN-2203"
    default_severity = ErrorSeverity.WARNING
    default_recovery_action = RecoveryAction.FALLBACK_PROVIDER


# ---------------------------------------------------------------------------
# Research / sentiment / features / learning
# ---------------------------------------------------------------------------


class SentimentError(FinancialAgentError):
    default_code = "FIN-3000"
    default_category = ErrorCategory.SENTIMENT
    default_severity = ErrorSeverity.ERROR
    default_safe_to_continue = True
    default_recovery_action = RecoveryAction.SKIP_BATCH


class SentimentSourceError(SentimentError):
    default_code = "FIN-3001"
    default_retryable = True
    default_transient = True
    default_recovery_action = RecoveryAction.FALLBACK_PROVIDER


class SentimentModelError(SentimentError):
    default_code = "FIN-3002"
    default_category = ErrorCategory.MODEL
    default_safe_to_continue = False
    default_recovery_action = RecoveryAction.DROP_TO_SAFE_MODE


class FeatureEngineeringError(FinancialAgentError):
    default_code = "FIN-3100"
    default_category = ErrorCategory.FEATURE_ENGINEERING
    default_severity = ErrorSeverity.ERROR
    default_safe_to_continue = True
    default_recovery_action = RecoveryAction.SKIP_BATCH


class TrendMonitoringError(FinancialAgentError):
    default_code = "FIN-3200"
    default_category = ErrorCategory.TREND_MONITORING
    default_severity = ErrorSeverity.ERROR
    default_safe_to_continue = True
    default_recovery_action = RecoveryAction.REFRESH_STATE


class RegimeShiftDetectionError(TrendMonitoringError):
    default_code = "FIN-3201"
    default_safe_to_continue = False
    default_recovery_action = RecoveryAction.DROP_TO_SAFE_MODE


class ModelError(FinancialAgentError):
    default_code = "FIN-3300"
    default_category = ErrorCategory.MODEL
    default_severity = ErrorSeverity.CRITICAL
    default_safe_to_continue = False
    default_recovery_action = RecoveryAction.DROP_TO_SAFE_MODE


class ModelInferenceError(ModelError):
    default_code = "FIN-3301"
    default_severity = ErrorSeverity.ERROR
    default_recovery_action = RecoveryAction.SKIP_BATCH


class ModelTrainingError(ModelError):
    default_code = "FIN-3302"
    default_recovery_action = RecoveryAction.PAUSE_STRATEGY


class ModelDriftError(ModelError):
    default_code = "FIN-3303"
    default_severity = ErrorSeverity.CRITICAL
    default_recovery_action = RecoveryAction.PAUSE_STRATEGY


class AdaptiveLearningError(FinancialAgentError):
    default_code = "FIN-3400"
    default_category = ErrorCategory.ADAPTIVE_LEARNING
    default_severity = ErrorSeverity.CRITICAL
    default_safe_to_continue = False
    default_recovery_action = RecoveryAction.PAUSE_STRATEGY


class FeedbackLoopInstabilityError(AdaptiveLearningError):
    default_code = "FIN-3401"
    default_recovery_action = RecoveryAction.HALT_TRADING


# ---------------------------------------------------------------------------
# Portfolio / risk / execution / compliance
# ---------------------------------------------------------------------------


class PortfolioError(FinancialAgentError):
    default_code = "FIN-4000"
    default_category = ErrorCategory.PORTFOLIO
    default_severity = ErrorSeverity.ERROR
    default_safe_to_continue = False
    default_recovery_action = RecoveryAction.REQUIRE_HUMAN_REVIEW


class AllocationConstraintError(PortfolioError):
    default_code = "FIN-4001"


class PortfolioStateError(PortfolioError):
    default_code = "FIN-4002"
    default_recovery_action = RecoveryAction.REFRESH_STATE


class RiskError(FinancialAgentError):
    default_code = "FIN-4100"
    default_category = ErrorCategory.RISK
    default_severity = ErrorSeverity.CRITICAL
    default_safe_to_continue = False
    default_recovery_action = RecoveryAction.PAUSE_STRATEGY


class RiskLimitBreachError(RiskError):
    default_code = "FIN-4101"
    default_recovery_action = RecoveryAction.HALT_TRADING


class DrawdownLimitExceededError(RiskError):
    default_code = "FIN-4102"
    default_recovery_action = RecoveryAction.HALT_TRADING


class ExposureLimitExceededError(RiskError):
    default_code = "FIN-4103"
    default_recovery_action = RecoveryAction.HALT_TRADING


class LiquidityRiskError(RiskError):
    default_code = "FIN-4104"
    default_recovery_action = RecoveryAction.PAUSE_STRATEGY


class ComplianceError(FinancialAgentError):
    default_code = "FIN-4200"
    default_category = ErrorCategory.COMPLIANCE
    default_severity = ErrorSeverity.CRITICAL
    default_safe_to_continue = False
    default_recovery_action = RecoveryAction.REQUIRE_HUMAN_REVIEW


class RestrictedAssetError(ComplianceError):
    default_code = "FIN-4201"


class PreTradeCheckError(ComplianceError):
    default_code = "FIN-4202"
    default_recovery_action = RecoveryAction.HALT_TRADING


class ExecutionError(FinancialAgentError):
    default_code = "FIN-4300"
    default_category = ErrorCategory.EXECUTION
    default_severity = ErrorSeverity.CRITICAL
    default_safe_to_continue = False
    default_recovery_action = RecoveryAction.PAUSE_STRATEGY


class OrderRejectedError(ExecutionError):
    default_code = "FIN-4301"
    default_severity = ErrorSeverity.ERROR
    default_recovery_action = RecoveryAction.REQUIRE_HUMAN_REVIEW


class OrderNotAcknowledgedError(ExecutionError):
    default_code = "FIN-4302"
    default_retryable = True
    default_transient = True
    default_recovery_action = RecoveryAction.REFRESH_STATE


class OrderRoutingError(ExecutionError):
    default_code = "FIN-4303"
    default_retryable = True
    default_transient = True
    default_recovery_action = RecoveryAction.BACKOFF_AND_RETRY


class SlippageExceededError(ExecutionError):
    default_code = "FIN-4304"
    default_recovery_action = RecoveryAction.PAUSE_STRATEGY


class PositionReconciliationError(ExecutionError):
    default_code = "FIN-4305"
    default_recovery_action = RecoveryAction.HALT_TRADING


# ---------------------------------------------------------------------------
# Storage / infra / generic failures
# ---------------------------------------------------------------------------


class PersistenceError(FinancialAgentError):
    default_code = "FIN-5000"
    default_category = ErrorCategory.PERSISTENCE
    default_severity = ErrorSeverity.ERROR
    default_retryable = True
    default_transient = True
    default_safe_to_continue = True
    default_recovery_action = RecoveryAction.BACKOFF_AND_RETRY


class StateStoreUnavailableError(PersistenceError):
    default_code = "FIN-5001"


class CheckpointWriteError(PersistenceError):
    default_code = "FIN-5002"
    default_safe_to_continue = False
    default_recovery_action = RecoveryAction.DROP_TO_SAFE_MODE


class InfrastructureError(FinancialAgentError):
    default_code = "FIN-5100"
    default_category = ErrorCategory.INFRASTRUCTURE
    default_severity = ErrorSeverity.CRITICAL
    default_retryable = True
    default_transient = True
    default_safe_to_continue = False
    default_recovery_action = RecoveryAction.DROP_TO_SAFE_MODE


class DependencyUnavailableError(InfrastructureError):
    default_code = "FIN-5101"


class ResourceExhaustionError(InfrastructureError):
    default_code = "FIN-5102"
    default_recovery_action = RecoveryAction.REDUCE_LOAD


class InternalAgentError(FinancialAgentError):
    default_code = "FIN-9000"
    default_category = ErrorCategory.INTERNAL
    default_severity = ErrorSeverity.CRITICAL
    default_safe_to_continue = False
    default_recovery_action = RecoveryAction.REQUIRE_HUMAN_REVIEW


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


HTTP_STATUS_CLASS_MAP: Dict[int, Type[FinancialAgentError]] = {
    400: ValidationError,
    401: ProviderAuthenticationError,
    403: ProviderAuthorizationError,
    404: DataUnavailableError,
    408: ProviderTimeoutError,
    409: ProviderError,
    422: ValidationError,
    429: ProviderRateLimitError,
    500: ServiceUnavailableError,
    502: ServiceUnavailableError,
    503: ServiceUnavailableError,
    504: ProviderTimeoutError,
}


def ensure_error_context(context: ContextInput) -> ErrorContext:
    if context is None:
        return ErrorContext()
    if isinstance(context, ErrorContext):
        return context
    return ErrorContext.from_mapping(context)


def _extract_status_code(exc: BaseException) -> Optional[int]:
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    return None


def classify_external_exception(
    exc: BaseException,
    *,
    context: ContextInput = None,
    message: Optional[str] = None,
    details: Optional[Mapping[str, Any]] = None,
) -> FinancialAgentError:
    """Normalize common third-party exceptions into the agent taxonomy."""
    details_dict = dict(details or {})
    details_dict.setdefault("external_exception_type", type(exc).__name__)
    ctx = ensure_error_context(context)

    if isinstance(exc, FinancialAgentError):
        return exc.with_context(ctx).add_details(**details_dict)

    if requests_exceptions is not None:
        if isinstance(exc, requests_exceptions.Timeout):
            return ProviderTimeoutError(
                message or "Provider request timed out.",
                context=ctx,
                details=details_dict,
                cause=exc,
            )
        if isinstance(exc, requests_exceptions.HTTPError):
            status_code = _extract_status_code(exc)
            details_dict.setdefault("status_code", status_code)
            exc_cls = HTTP_STATUS_CLASS_MAP.get(status_code or -1, ProviderError)
            return exc_cls(
                message or f"Provider returned HTTP status {status_code}.",
                context=ctx,
                details=details_dict,
                cause=exc,
            )
        if isinstance(exc, requests_exceptions.ConnectionError):
            return ConnectivityError(
                message or "Network connection to external service failed.",
                context=ctx,
                details=details_dict,
                cause=exc,
            )
        if isinstance(exc, requests_exceptions.RequestException):
            return ProviderError(
                message or "External provider request failed.",
                context=ctx,
                details=details_dict,
                cause=exc,
            )

    if isinstance(exc, TimeoutError):
        return NetworkTimeoutError(
            message or "Operation timed out.",
            context=ctx,
            details=details_dict,
            cause=exc,
        )

    if isinstance(exc, socket.gaierror):
        return DNSResolutionError(
            message or "DNS resolution failed for external service.",
            context=ctx,
            details=details_dict,
            cause=exc,
        )

    if isinstance(exc, (KeyError, ValueError, TypeError)):
        return ValidationError(
            message or "Validation or schema processing failed.",
            context=ctx,
            details=details_dict,
            cause=exc,
        )

    return InternalAgentError(
        message or "Unhandled internal agent exception.",
        context=ctx,
        details=details_dict,
        cause=exc,
    )


def is_retryable_error(exc: BaseException) -> bool:
    if isinstance(exc, FinancialAgentError):
        return exc.retryable
    normalized = classify_external_exception(exc)
    return normalized.retryable


def should_halt_trading(exc: BaseException) -> bool:
    normalized = exc if isinstance(exc, FinancialAgentError) else classify_external_exception(exc)
    return normalized.recovery_action in {
        RecoveryAction.HALT_TRADING,
        RecoveryAction.REQUIRE_HUMAN_REVIEW,
    }


def log_error(
    exc: BaseException,
    *,
    logger_: Optional[logging.Logger] = None,
    include_traceback: bool = True,
    level: Optional[int] = None,
) -> JSONDict:
    """Log a normalized exception and return its serialized report."""
    logger_ = logger_ or logger
    normalized = exc if isinstance(exc, FinancialAgentError) else classify_external_exception(exc)
    report = normalized.to_dict(include_traceback=include_traceback)

    if level is None:
        level = {
            ErrorSeverity.INFO: logging.INFO,
            ErrorSeverity.WARNING: logging.WARNING,
            ErrorSeverity.ERROR: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL,
        }[normalized.severity]

    logger_.log(level, "%s", report)
    return report


def raise_if(condition: bool, exc_factory: Callable[[], BaseException]) -> None:
    """Raise lazily-created exceptions for readable guard clauses."""
    if condition:
        raise exc_factory()


def enrich_and_raise(
    exc: BaseException,
    *,
    context: ContextInput = None,
    details: Optional[Mapping[str, Any]] = None,
    message: Optional[str] = None,
) -> None:
    """Normalize, enrich, and raise an exception in one call."""
    normalized = classify_external_exception(exc, context=context, details=details, message=message)
    raise normalized from exc


def error_boundary(
    *,
    component: Optional[str] = None,
    operation: Optional[str] = None,
    context: ContextInput = None,
    normalize: bool = True,
    reraise: bool = True,
    logger_: Optional[logging.Logger] = None,
    include_traceback: bool = True,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator for standardizing error handling around agent operations.

    Example:
        @error_boundary(component="execution", operation="submit_order")
        def submit_order(...):
            ...
    """

    boundary_context = ensure_error_context(context).merge(
        {"component": component, "operation": operation}
    )

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except BaseException as exc:
                handled = (
                    classify_external_exception(exc, context=boundary_context)
                    if normalize
                    else exc
                )
                log_error(
                    handled,
                    logger_=logger_,
                    include_traceback=include_traceback,
                )
                if reraise:
                    raise handled from exc
                return None

        return wrapper

    return decorator


@contextmanager
def capture_errors(
    *,
    component: Optional[str] = None,
    operation: Optional[str] = None,
    context: ContextInput = None,
    normalize: bool = True,
    logger_: Optional[logging.Logger] = None,
    include_traceback: bool = True,
) -> Iterator[MutableMapping[str, Any]]:
    """Context manager version of error_boundary.

    Yields a mutable state dict that callers can inspect after the block.
    If an exception occurs, the dict contains:
        state["ok"] = False
        state["error"] = serialized report
    """
    state: MutableMapping[str, Any] = {"ok": True, "error": None}
    merged_context = ensure_error_context(context).merge(
        {"component": component, "operation": operation}
    )
    try:
        yield state
    except BaseException as exc:
        handled = classify_external_exception(exc, context=merged_context) if normalize else exc
        report = log_error(handled, logger_=logger_, include_traceback=include_traceback)
        state["ok"] = False
        state["error"] = report
        raise handled from exc


def build_failure_response(
    exc: BaseException,
    *,
    include_traceback: bool = False,
    safe_message: Optional[str] = None,
) -> JSONDict:
    """Build a sanitized failure payload for APIs, agents, or message buses."""
    normalized = exc if isinstance(exc, FinancialAgentError) else classify_external_exception(exc)
    payload = normalized.to_dict(include_traceback=include_traceback)
    if safe_message is not None:
        payload["message"] = safe_message
    return payload


# ---------------------------------------------------------------------------
# Domain-specific guard helpers
# ---------------------------------------------------------------------------


def assert_market_data_fresh(
    *,
    age_seconds: float,
    max_age_seconds: float,
    context: ContextInput = None,
) -> None:
    """Raise when market data exceeds freshness tolerance."""
    if age_seconds > max_age_seconds:
        raise DataStalenessError(
            f"Market data is stale: age={age_seconds:.3f}s exceeds limit={max_age_seconds:.3f}s.",
            context=context,
            details={"age_seconds": age_seconds, "max_age_seconds": max_age_seconds},
            tags=("market_data", "freshness"),
        )


def assert_drawdown_within_limit(
    *,
    current_drawdown: float,
    max_drawdown: float,
    context: ContextInput = None,
) -> None:
    """Raise if realized or projected drawdown breaches policy."""
    if current_drawdown > max_drawdown:
        raise DrawdownLimitExceededError(
            (
                f"Drawdown limit exceeded: current_drawdown={current_drawdown:.6f} "
                f"> max_drawdown={max_drawdown:.6f}."
            ),
            context=context,
            details={
                "current_drawdown": current_drawdown,
                "max_drawdown": max_drawdown,
            },
            tags=("risk", "drawdown"),
        )


def assert_exposure_within_limit(
    *,
    exposure: float,
    limit: float,
    context: ContextInput = None,
    net_or_gross: str = "gross",
) -> None:
    """Raise when portfolio exposure exceeds configured policy."""
    if exposure > limit:
        raise ExposureLimitExceededError(
            f"{net_or_gross.capitalize()} exposure limit exceeded: exposure={exposure:.6f}, limit={limit:.6f}.",
            context=context,
            details={"exposure": exposure, "limit": limit, "type": net_or_gross},
            tags=("risk", "exposure", net_or_gross),
        )


def assert_order_preconditions(
    *,
    quantity: float,
    price: Optional[float] = None,
    symbol: Optional[str] = None,
    context: ContextInput = None,
) -> None:
    """Validate common order request preconditions before submission."""
    if not symbol:
        raise InvalidOrderRequestError(
            "Order request missing symbol.",
            context=context,
            details={"quantity": quantity, "price": price},
            tags=("execution", "validation"),
        )
    if quantity <= 0:
        raise InvalidOrderRequestError(
            "Order quantity must be positive.",
            context=context,
            details={"symbol": symbol, "quantity": quantity, "price": price},
            tags=("execution", "validation"),
        )
    if price is not None and price <= 0:
        raise InvalidOrderRequestError(
            "Order price must be positive when provided.",
            context=context,
            details={"symbol": symbol, "quantity": quantity, "price": price},
            tags=("execution", "validation"),
        )


# ---------------------------------------------------------------------------
# Example policy function used by orchestrators
# ---------------------------------------------------------------------------


def derive_orchestrator_decision(exc: BaseException) -> JSONDict:
    """Translate an exception into an orchestration decision.

    This is intentionally simple but useful as a starting point for real policy
    engines, schedulers, and workflow coordinators.
    """
    normalized = exc if isinstance(exc, FinancialAgentError) else classify_external_exception(exc)
    return {
        "retry": normalized.retryable,
        "safe_to_continue": normalized.safe_to_continue,
        "halt_trading": should_halt_trading(normalized),
        "recovery_action": normalized.recovery_action.value,
        "severity": normalized.severity.value,
        "category": normalized.category.value,
        "error_code": normalized.code,
    }


__all__ = [
    "AdaptiveLearningError",
    "AllocationConstraintError",
    "assert_drawdown_within_limit",
    "assert_exposure_within_limit",
    "assert_market_data_fresh",
    "assert_order_preconditions",
    "build_failure_response",
    "capture_errors",
    "CheckpointWriteError",
    "classify_external_exception",
    "ComplianceError",
    "ConfigurationError",
    "ConnectivityError",
    "DataError",
    "DataStalenessError",
    "DataUnavailableError",
    "DependencyUnavailableError",
    "derive_orchestrator_decision",
    "DNSResolutionError",
    "DrawdownLimitExceededError",
    "enrich_and_raise",
    "ErrorCategory",
    "ErrorContext",
    "error_boundary",
    "ErrorReport",
    "ErrorSeverity",
    "ExecutionError",
    "ExposureLimitExceededError",
    "FeatureEngineeringError",
    "FeedbackLoopInstabilityError",
    "FinancialAgentError",
    "InfrastructureError",
    "InternalAgentError",
    "InvalidConfigurationError",
    "InvalidOrderRequestError",
    "InvalidSymbolError",
    "is_retryable_error",
    "LiquidityRiskError",
    "log_error",
    "MarketDataGapError",
    "MissingConfigurationError",
    "ModelDriftError",
    "ModelError",
    "ModelInferenceError",
    "ModelTrainingError",
    "NetworkTimeoutError",
    "OrderNotAcknowledgedError",
    "OrderRejectedError",
    "OrderRoutingError",
    "PartialDataError",
    "PersistenceError",
    "PortfolioError",
    "PortfolioStateError",
    "PreTradeCheckError",
    "ProviderAuthenticationError",
    "ProviderAuthorizationError",
    "ProviderError",
    "ProviderParsingError",
    "ProviderRateLimitError",
    "ProviderTimeoutError",
    "raise_if",
    "RecoveryAction",
    "RegimeShiftDetectionError",
    "ResourceExhaustionError",
    "RestrictedAssetError",
    "RiskError",
    "RiskLimitBreachError",
    "SentimentError",
    "SentimentModelError",
    "SentimentSourceError",
    "ServiceUnavailableError",
    "should_halt_trading",
    "SlippageExceededError",
    "StateStoreUnavailableError",
    "TrendMonitoringError",
    "UnsupportedAssetClassError",
    "ValidationError",
]
