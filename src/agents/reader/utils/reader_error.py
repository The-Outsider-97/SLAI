from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional


class ReaderErrorType(Enum):
    """Reader domain error taxonomy.

    Values preserve legacy labels for backward compatibility with historical
    string comparisons and serialized error payloads.
    """

    INVALID_TASK = "Invalid task"
    INVALID_INSTRUCTION = "Invalid instruction"
    FILE_NOT_FOUND = "File not found"
    UNSUPPORTED_FORMAT = "Unsupported format"
    PARSE_FAILURE = "Parse failure"
    CONVERSION_FAILURE = "Conversion failure"
    RECOVERY_FAILURE = "Recovery failure"
    MERGE_FAILURE = "Merge failure"
    PERSISTENCE_FAILURE = "Persistence failure"


class ReaderError(Exception):
    def __init__(
        self,
        error_type: ReaderErrorType,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.error_type = error_type
        self.context = context or {}
        self.cause = cause

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_type": self.error_type.value,
            "message": str(self),
            "context": self.context,
            "cause": type(self.cause).__name__ if self.cause else None,
        }


class ReaderValidationError(ReaderError):
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(ReaderErrorType.INVALID_TASK, message, context=context)


class FileMissingError(ReaderError):
    def __init__(self, file_path: str):
        super().__init__(
            ReaderErrorType.FILE_NOT_FOUND,
            f"File does not exist: {file_path}",
            {"file_path": file_path},
        )


class UnsupportedFormatError(ReaderError):
    def __init__(self, file_path: str, extension: str, allowed_extensions: set[str]):
        super().__init__(
            ReaderErrorType.UNSUPPORTED_FORMAT,
            f"Unsupported input format '{extension}' for file: {file_path}",
            {
                "file_path": file_path,
                "extension": extension,
                "allowed_extensions": sorted(allowed_extensions),
            },
        )


class ParseFailureError(ReaderError):
    def __init__(self, file_path: str, message: str, cause: Optional[Exception] = None):
        super().__init__(
            ReaderErrorType.PARSE_FAILURE,
            message,
            {"file_path": file_path},
            cause=cause,
        )


class ConversionFailureError(ReaderError):
    def __init__(self, source: str, target_format: str, message: str, cause: Optional[Exception] = None):
        super().__init__(
            ReaderErrorType.CONVERSION_FAILURE,
            message,
            {"source": source, "target_format": target_format},
            cause=cause,
        )


class MergeFailureError(ReaderError):
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None, cause: Optional[Exception] = None):
        super().__init__(ReaderErrorType.MERGE_FAILURE, message, context=context, cause=cause)


class RecoveryFailureError(ReaderError):
    def __init__(self, source: str, message: str, cause: Optional[Exception] = None):
        super().__init__(ReaderErrorType.RECOVERY_FAILURE, message, {"source": source}, cause=cause)


class PersistenceError(ReaderError):
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None, cause: Optional[Exception] = None):
        super().__init__(ReaderErrorType.PERSISTENCE_FAILURE, message, context=context, cause=cause)


__all__ = [
    "ReaderErrorType",
    "ReaderError",
    "ReaderValidationError",
    "FileMissingError",
    "UnsupportedFormatError",
    "ParseFailureError",
    "ConversionFailureError",
    "MergeFailureError",
    "RecoveryFailureError",
    "PersistenceError",
]
