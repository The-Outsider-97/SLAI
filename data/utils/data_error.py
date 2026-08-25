"""
Centralised exception hierarchy for the data pipeline.
 
Design goals
------------
* Every error carries a machine-readable ``DataErrorCode``, a UTC timestamp,
  and a free-form ``context`` dict so that callers can log or serialise errors
  without string-parsing the message.
* ``DataError.to_dict()`` produces a fully self-contained snapshot that can be
  written to a structured log, a dead-letter queue, or an alerting system.
* Severity levels let operators route errors to different sinks (DEBUG noise vs.
  CRITICAL pages) without inspecting the exception type.
* All subclasses are keyword-only for ``context`` to prevent accidental
  positional mis-ordering at call sites.
* ``SecurityError`` is kept as a *sibling* of ``DataError`` (not a subclass)
  because security violations may need to be handled by a completely separate
  policy layer that must not accidentally swallow data errors.
"""
 
from __future__ import annotations
 
import traceback
import uuid
 
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional
 
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]
 
logger = get_logger("Data Error")
printer = PrettyPrinter()
 
 
# ---------------------------------------------------------------------------
# Severity
# ---------------------------------------------------------------------------
class ErrorSeverity(str, Enum):
    """Operational severity attached to every ``DataError``.
 
    Consumers (loggers, alerting hooks) use this to decide routing without
    inspecting the concrete exception type.
 
    DEBUG   – informational; never pages on-call.
    WARNING – degraded behaviour; pipeline continues; worth monitoring.
    ERROR   – hard failure in one component; pipeline may retry.
    CRITICAL – unrecoverable state; pipeline must halt; page on-call immediately.
    """
 
    DEBUG    = "DEBUG"
    WARNING  = "WARNING"
    ERROR    = "ERROR"
    CRITICAL = "CRITICAL"

class DataErrorCode(str, Enum):
    """Stable, machine-readable identifiers for every error category.
 
    Using ``str`` as a mixin means the value is directly JSON-serialisable
    without calling ``.value``.
    """
 
    # Infrastructure / configuration
    CONFIG_ERROR             = "DATA_CONFIG_ERROR"
    SCHEMA_ERROR             = "DATA_SCHEMA_ERROR"
 
    # Ingestion pipeline
    SOURCE_ERROR             = "DATA_SOURCE_ERROR"
    INGESTION_CONTRACT_ERROR = "DATA_INGESTION_CONTRACT_ERROR"
    TRANSFORM_ERROR          = "DATA_TRANSFORM_ERROR"
 
    # Quality & validation
    VALIDATION_ERROR         = "DATA_VALIDATION_ERROR"
    QUALITY_GATE_ERROR       = "DATA_QUALITY_GATE_ERROR"
 
    # Versioning & lineage
    VERSIONING_ERROR         = "DATA_VERSIONING_ERROR"
 
    # Security
    SECURITY_ERROR           = "DATA_SECURITY_ERROR"
 
 
# ---------------------------------------------------------------------------
# Base exception
# ---------------------------------------------------------------------------
class DataError(Exception):
    """Base exception for all data-pipeline failures.
 
    Attributes
    ----------
    code : DataErrorCode
        Machine-readable error category.
    severity : ErrorSeverity
        Operational severity used for routing to logs / alerts.
    context : dict
        Arbitrary structured payload; **never** put secrets here.
    error_id : str
        UUID4 that uniquely identifies this exception instance.
        Useful for correlating log lines with alerting tickets.
    timestamp : str
        UTC ISO-8601 instant at which the exception was constructed.
    """
 
    #: Subclasses override this to set a different default severity.
    _default_severity: ErrorSeverity = ErrorSeverity.ERROR
 
    def __init__(self, message: str, code: DataErrorCode = DataErrorCode.VALIDATION_ERROR, *,
                 context: Optional[Dict[str, Any]] = None, severity: Optional[ErrorSeverity] = None,
                 cause: Optional[BaseException] = None) -> None:
        super().__init__(message)
        self.code      = code
        self.context   = context or {}
        self.severity  = severity if severity is not None else self._default_severity
        self.error_id  = str(uuid.uuid4())
        self.timestamp = datetime.now(timezone.utc).isoformat()
 
        # Explicit chaining preserves the original traceback when re-raising.
        if cause is not None:
            self.__cause__ = cause
 
        self._log_on_construction()
 
    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _log_on_construction(self) -> None:
        """Emit a structured log line at construction time.
 
        Logging here means every raised DataError is captured even when the
        caller swallows the exception or re-raises a different one.
        """
        log_payload = {
            "error_id":  self.error_id,
            "code":      self.code,
            "severity":  self.severity,
            "message":   str(self),
            "context":   self.context,
            "timestamp": self.timestamp,
        }
 
        if self.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_payload)
        elif self.severity == ErrorSeverity.ERROR:
            logger.error(log_payload)
        elif self.severity == ErrorSeverity.WARNING:
            logger.warning(log_payload)
        else:
            logger.debug(log_payload)
 
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Return a fully self-contained, JSON-serialisable snapshot.
 
        The snapshot is suitable for:
        * Structured log sinks (e.g. Cloud Logging, Datadog).
        * Dead-letter queue messages.
        * API error responses (strip ``traceback`` before sending externally).
        """
        base: Dict[str, Any] = {
            "error_id":  self.error_id,
            "code":      self.code,            # already a str via str-Enum mixin
            "severity":  self.severity,
            "message":   str(self),
            "context":   self.context,
            "timestamp": self.timestamp,
        }
        if self.__cause__ is not None:
            base["cause"] = {
                "type":      type(self.__cause__).__name__,
                "message":   str(self.__cause__),
                "traceback": traceback.format_exception(
                    type(self.__cause__),
                    self.__cause__,
                    self.__cause__.__traceback__,
                ),
            }
        return base
 
    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"code={self.code!r}, "
            f"severity={self.severity!r}, "
            f"error_id={self.error_id!r}, "
            f"message={str(self)!r})"
        )
 
 
# ---------------------------------------------------------------------------
# Infrastructure / configuration errors
# ---------------------------------------------------------------------------
class DataConfigError(DataError):
    """Raised when the YAML config is missing, malformed, or contains invalid
    values that prevent the pipeline from starting.
 
    Typical triggers
    ----------------
    * Config file not found on disk.
    * Required section or key absent (e.g. ``versioning.registry_path``).
    * Unsupported hash algorithm specified.
    * Top-level YAML is not a mapping.
 
    Severity: CRITICAL — a bad config prevents the entire pipeline from running.
 
    Example
    -------
    ::
 
        raise DataConfigError(
            "Missing versioning.registry_path in config",
            context={"config_path": str(config_path)},
        )
    """
 
    _default_severity = ErrorSeverity.CRITICAL
 
    def __init__(self, message: str, *,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        super().__init__(
            message,
            code=DataErrorCode.CONFIG_ERROR,
            context=context,
            cause=cause,
        )
 
 
class DataSchemaError(DataError):
    """Raised when a ``DatasetSchema`` or ``DatasetField`` definition is itself
    invalid — distinct from a record failing schema validation at runtime.
 
    Typical triggers
    ----------------
    * Duplicate field names within a schema.
    * Schema registered for an already-known modality with a conflicting
      definition.
    * ``expected_type`` is not a valid Python type.
 
    Severity: CRITICAL — a bad schema definition breaks an entire modality.
 
    Example
    -------
    ::
 
        raise DataSchemaError(
            "Duplicate field name 'image_tokens' in schema 'vision_v2'",
            context={"schema": "vision_v2", "field": "image_tokens"},
        )
    """
 
    _default_severity = ErrorSeverity.CRITICAL
 
    def __init__(
        self,
        message: str,
        *,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        super().__init__(
            message,
            code=DataErrorCode.SCHEMA_ERROR,
            context=context,
            cause=cause,
        )
 
 
# ---------------------------------------------------------------------------
# Ingestion pipeline errors
# ---------------------------------------------------------------------------
 
class DataSourceError(DataError):
    """Raised when raw data cannot be fetched from an upstream source.
 
    Typical triggers
    ----------------
    * Network timeout reaching a remote dataset URI.
    * Object-store bucket / key not found.
    * Unsupported file format or corrupt archive.
    * Authentication / authorisation failure at the source.
 
    Severity: ERROR — the source may be transient; callers should decide
    whether to retry.
 
    Example
    -------
    ::
 
        raise DataSourceError(
            "S3 key not found",
            context={"uri": "s3://bucket/path/to/shard.tar", "http_status": 404},
            cause=original_exc,
        )
    """
 
    _default_severity = ErrorSeverity.ERROR
 
    def __init__(
        self,
        message: str,
        *,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        super().__init__(
            message,
            code=DataErrorCode.SOURCE_ERROR,
            context=context,
            cause=cause,
        )
 
 
class DataIngestionContractError(DataError):
    """Raised when a structural contract on the ingested payload is violated
    *before* per-field validation runs.
 
    Typical triggers
    ----------------
    * Vision / text / audio arrays are not equal in length when alignment is
      enforced.
    * Empty payload when ``allow_empty_payload`` is ``False``.
    * ``batch_size`` is below the configured minimum.
 
    Severity: ERROR — the payload shape is wrong; the batch is rejected.
 
    Example
    -------
    ::
 
        raise DataIngestionContractError(
            "Modality alignment failed; expected equal sizes but got "
            "{'vision': 128, 'text': 127, 'audio': 128}",
            context={"lengths": {"vision": 128, "text": 127, "audio": 128}},
        )
    """
 
    _default_severity = ErrorSeverity.ERROR
 
    def __init__(self, message: str, *, context: Optional[Dict[str, Any]] = None,
                 cause: Optional[BaseException] = None) -> None:
        super().__init__(
            message,
            code=DataErrorCode.INGESTION_CONTRACT_ERROR,
            context=context,
            cause=cause,
        )
 
 
class DataTransformError(DataError):
    """Raised when a data transformation step fails after ingestion.
 
    Typical triggers
    ----------------
    * Tokenisation failure (e.g. text too long for the tokeniser).
    * Image decode / resize error.
    * Feature extraction producing NaN / Inf values.
    * Normalization producing out-of-range outputs.
 
    Severity: ERROR — a single transform failure rejects the affected shard.
 
    Example
    -------
    ::
 
        raise DataTransformError(
            "Audio resampling produced NaN output",
            context={
                "modality": "audio",
                "row_idx": 7,
                "source_sr": 48000,
                "target_sr": 16000,
            },
            cause=original_exc,
        )
    """
 
    _default_severity = ErrorSeverity.ERROR
 
    def __init__(self, message: str, *, context: Optional[Dict[str, Any]] = None,
                 cause: Optional[BaseException] = None) -> None:
        super().__init__(
            message,
            code=DataErrorCode.TRANSFORM_ERROR,
            context=context,
            cause=cause,
        )
 
 
# ---------------------------------------------------------------------------
# Quality & validation errors
# ---------------------------------------------------------------------------
 
class DataValidationError(DataError):
    """Raised when an individual record fails schema validation.
 
    Typical triggers
    ----------------
    * Required field absent from a row.
    * Field value is ``None`` but the field is not nullable.
    * Field value has the wrong Python type.
    * Sequence field exceeds ``max_items``.
    * Unknown modality when ``fail_on_unknown_modality`` is ``True``.
 
    Severity: ERROR — the offending record(s) are rejected.
 
    Example
    -------
    ::
 
        raise DataValidationError(
            "vision[14] field 'image_tokens' expected list, got str",
            context={
                "modality":      "vision",
                "row_idx":       14,
                "field":         "image_tokens",
                "expected_type": "list",
                "actual_type":   "str",
            },
        )
    """
 
    _default_severity = ErrorSeverity.ERROR
 
    def __init__(self, message: str, *, context: Optional[Dict[str, Any]] = None,
                 cause: Optional[BaseException] = None) -> None:
        super().__init__(
            message,
            code=DataErrorCode.VALIDATION_ERROR,
            context=context,
            cause=cause,
        )
 
 
class DataQualityGateError(DataError):
    """Raised when a payload passes schema validation but fails an aggregate
    quality threshold.
 
    Typical triggers
    ----------------
    * Null ratio across the batch exceeds ``quality_gate.max_null_ratio``.
    * Future gates: duplicate-ID ratio, out-of-distribution feature score, etc.
 
    Severity: ERROR — the entire payload is quarantined pending review.
 
    Example
    -------
    ::
 
        raise DataQualityGateError(
            "Null ratio 0.032500 exceeds threshold 0.000000",
            context={
                "null_ratio":     0.0325,
                "max_null_ratio": 0.0,
                "null_counts":    {"vision": 12, "text": 3, "audio": 0},
            },
        )
    """
 
    _default_severity = ErrorSeverity.ERROR
 
    def __init__(self, message: str, *, cause: Optional[BaseException] = None,
                 context: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(
            message,
            code=DataErrorCode.QUALITY_GATE_ERROR,
            context=context,
            cause=cause,
        )
 
 
# ---------------------------------------------------------------------------
# Versioning & lineage errors
# ---------------------------------------------------------------------------
 
class DataVersioningError(DataError):
    """Raised when the dataset registry detects an integrity or lineage problem.
 
    Typical triggers
    ----------------
    * Duplicate (dataset_name, dataset_version, payload_hash) triple — the
      same payload was registered twice.
    * Registry file is corrupt or cannot be written.
    * Hash algorithm mismatch between registry entries and current config.
 
    Severity: CRITICAL — a versioning collision could silently corrupt the
    lineage graph; the pipeline must halt.
 
    Example
    -------
    ::
 
        raise DataVersioningError(
            "Duplicate dataset version and payload hash detected",
            context={
                "dataset_name":    "multimodal_v3",
                "dataset_version": "2025-07-01",
                "payload_hash":    "a3f9...",
            },
        )
    """
 
    _default_severity = ErrorSeverity.CRITICAL
 
    def __init__(self, message: str, *, context: Optional[Dict[str, Any]] = None,
                 cause: Optional[BaseException] = None) -> None:
        super().__init__(
            message,
            code=DataErrorCode.VERSIONING_ERROR,
            context=context,
            cause=cause,
        )
 
 
# ---------------------------------------------------------------------------
# Security error  (sibling, NOT a DataError subclass)
# ---------------------------------------------------------------------------
 
class SecurityError(Exception):
    """Raised when a security policy is violated in the data pipeline.
 
    Kept as a *sibling* of ``DataError`` — not a subclass — so that a broad
    ``except DataError`` can never silently swallow a security violation.
    Security handlers must be explicit.
 
    Attributes
    ----------
    error_id  : str   – UUID4 for correlation with audit logs.
    timestamp : str   – UTC ISO-8601 instant.
    context   : dict  – Structured payload; **never** put secrets here.
 
    Typical triggers
    ----------------
    * Payload hash does not match the expected value stored in the registry
      (tamper detection).
    * Access to a restricted data source without the required credential scope.
    * Attempt to overwrite an immutable lineage record.
 
    Example
    -------
    ::
 
        raise SecurityError(
            "Payload hash mismatch — possible data tampering",
            context={
                "dataset_name":    "multimodal_v3",
                "expected_hash":   "a3f9...",
                "actual_hash":     "b7c2...",
            },
        )
    """
 
    def __init__(self, message: str, *, context: Optional[Dict[str, Any]] = None,
                 cause: Optional[BaseException] = None) -> None:
        super().__init__(message)
        self.context   = context or {}
        self.error_id  = str(uuid.uuid4())
        self.timestamp = datetime.now(timezone.utc).isoformat()
 
        if cause is not None:
            self.__cause__ = cause
 
        # Security violations always log at CRITICAL.
        logger.critical({
            "error_id":  self.error_id,
            "code":      DataErrorCode.SECURITY_ERROR,
            "severity":  ErrorSeverity.CRITICAL,
            "message":   message,
            "context":   self.context,
            "timestamp": self.timestamp,
        })
 
    def to_dict(self) -> Dict[str, Any]:
        base: Dict[str, Any] = {
            "error_id":  self.error_id,
            "code":      DataErrorCode.SECURITY_ERROR,
            "severity":  ErrorSeverity.CRITICAL,
            "message":   str(self),
            "context":   self.context,
            "timestamp": self.timestamp,
        }
        if self.__cause__ is not None:
            base["cause"] = {
                "type":    type(self.__cause__).__name__,
                "message": str(self.__cause__),
            }
        return base
 
    def __repr__(self) -> str:
        return (
            f"SecurityError("
            f"error_id={self.error_id!r}, "
            f"message={str(self)!r})"
        )
 