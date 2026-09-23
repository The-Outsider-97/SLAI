from __future__ import annotations

__version__ = "2.3.0"

import traceback

from typing import Any, Dict, Optional, Type, TypeVar

from ...base.utils.base_errors import BaseError, BaseErrorType


TVerificationError = TypeVar("TVerificationError", bound="VerificationError")


class VerificationError(BaseError):
    """
    Root of the Reasoning domain error hierarchy.

    Keeps Reasoning-specific stable codes while participating in the
    BaseAgent severity/category/retry/cause/context contract.
    """

    error_type = BaseErrorType.RUNTIME
    default_code = "verification_error"
    default_severity = "medium"
    default_retryable = False
    default_category = "verification"

    def __init__(self, message: str,
            *,
            code: Optional[str] = None,
            context: Optional[Dict[str, Any]] = None) -> None:
        pass


__all__ = []