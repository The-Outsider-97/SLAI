from enum import Enum
from typing import Dict, Any, Optional


class ReaderErrorType(Enum):
    INVALID_INSTRUCTION = "Invalid instruction"
    FILE_NOT_FOUND = "File not found"
    PARSE_FAILURE = "Parse failure"
    CONVERSION_FAILURE = "Conversion failure"
    RECOVERY_FAILURE = "Recovery failure"
    MERGE_FAILURE = "Merge failure"


class ReaderError(Exception):
    def __init__(
        self,
        error_type: ReaderErrorType,
        message: str,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.error_type = error_type
        self.context = context or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_type": self.error_type.value,
            "message": str(self),
            "context": self.context,
        }


class FileMissingError(ReaderError):
    def __init__(self, file_path: str):
        super().__init__(
            ReaderErrorType.FILE_NOT_FOUND,
            f"File does not exist: {file_path}",
            {"file_path": file_path},
        )
