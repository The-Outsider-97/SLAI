from __future__ import annotations

__version__ = "2.3.0"

from typing import Any, Mapping

from src.agents.base.utils.base_errors import BaseError, BaseErrorType # pyright: ignore[reportMissingImports]


class LLMProviderError(BaseError):
    error_type = BaseErrorType.RUNTIME
    default_code = ""
    default_severity = "medium"
    default_retryable = False
    default_category = "training"

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        context: Mapping[str, Any] | None = None,
        cause: BaseException | None = None,
        retryable: bool | None = None,
    ) -> None:
        kwargs: dict[str, Any] = {
            "code": code or self.default_code,
            "context": dict(context or {}),
            "cause": cause,
        }
        if retryable is not None:
            kwargs["retryable"] = retryable
        super().__init__(message, **kwargs)


class LLMProviderUnavailable(LLMProviderError):
    pass


class LLMProviderRateLimited(LLMProviderError):
    pass


class LLMProviderConfigurationError(LLMProviderError):
    """Disable unchanged credentials/model/endpoint until configuration changes."""


class LLMProviderQuotaError(LLMProviderError):
    pass


class LLMProviderRequestError(LLMProviderError):
    """Do not repeat an invalid or policy-rejected request across providers."""


class LLMProviderResponseError(LLMProviderError):
    pass


__all__ = [
    "LLMProviderError",
    "LLMProviderUnavailable",
    "LLMProviderRateLimited",
    "LLMProviderConfigurationError",
    "LLMProviderQuotaError",
    "LLMProviderRequestError",
    "LLMProviderResponseError",
]