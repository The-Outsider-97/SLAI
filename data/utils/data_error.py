from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional


class DataErrorCode(Enum):
    CONFIG_ERROR = "DATA_CONFIG_ERROR"
    VALIDATION_ERROR = "DATA_VALIDATION_ERROR"
    INGESTION_CONTRACT_ERROR = "DATA_INGESTION_CONTRACT_ERROR"
    QUALITY_GATE_ERROR = "DATA_QUALITY_GATE_ERROR"
    VERSIONING_ERROR = "DATA_VERSIONING_ERROR"


class DataError(Exception):
    """Base exception for data module failures."""

    def __init__(
        self,
        message: str,
        code: DataErrorCode = DataErrorCode.VALIDATION_ERROR,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.context = context or {}
        self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code.value,
            "message": str(self),
            "context": self.context,
            "timestamp": self.timestamp,
        }


class DataConfigError(DataError):
    def __init__(self, message: str, *, context: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, code=DataErrorCode.CONFIG_ERROR, context=context)


class DataValidationError(DataError):
    def __init__(self, message: str, *, context: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, code=DataErrorCode.VALIDATION_ERROR, context=context)


class DataIngestionContractError(DataError):
    def __init__(self, message: str, *, context: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, code=DataErrorCode.INGESTION_CONTRACT_ERROR, context=context)


class DataQualityGateError(DataError):
    def __init__(self, message: str, *, context: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, code=DataErrorCode.QUALITY_GATE_ERROR, context=context)


class DataVersioningError(DataError):
    def __init__(self, message: str, *, context: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, code=DataErrorCode.VERSIONING_ERROR, context=context)
