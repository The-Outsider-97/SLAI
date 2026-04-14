from datetime import datetime
from typing import Any, Dict, Optional

# ============================================================================
# Authentication exceptions (production-ready)
# ============================================================================
class AuthError(Exception):
    """Base exception for all authentication and authorisation errors."""

    def __init__(
        self,
        message: str = "Authentication error",
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        username: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.username = username

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API error responses."""
        result: Dict[str, Any] = {"error": self.message, "type": self.__class__.__name__}
        if self.error_code:
            result["code"] = self.error_code
        if self.username:
            result["username"] = self.username
        if self.details:
            result["details"] = self.details
        return result


class UserAlreadyExistsError(AuthError):
    """Raised when trying to register a username that already exists."""

    def __init__(
        self,
        username: str,
        message: Optional[str] = None,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.username = username
        msg = message or f"User '{username}' already exists"
        super().__init__(msg, error_code=error_code, details=details, username=username)


class InvalidCredentialsError(AuthError):
    """Raised when username/password combination is incorrect or user missing."""

    def __init__(
        self,
        username: Optional[str] = None,
        message: Optional[str] = None,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.username = username
        msg = message or "Invalid username or password"
        super().__init__(msg, error_code=error_code, details=details, username=username)


class AccountLockedError(AuthError):
    """Raised when an account is temporarily locked due to too many failed attempts."""

    def __init__(
        self,
        username: str,
        retry_after_seconds: Optional[float] = None,
        lockout_until: Optional[datetime] = None,
        message: Optional[str] = None,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.username = username
        self.retry_after_seconds = retry_after_seconds
        self.lockout_until = lockout_until

        if not message:
            if retry_after_seconds is not None:
                msg = f"Account '{username}' locked. Retry after {retry_after_seconds:.1f} seconds"
            elif lockout_until:
                msg = f"Account '{username}' locked until {lockout_until.isoformat()}"
            else:
                msg = f"Account '{username}' is temporarily locked"
        else:
            msg = message

        # Add lock info to details
        extra_details = {"retry_after_seconds": retry_after_seconds, "lockout_until": lockout_until}
        if details:
            extra_details.update(details)
        super().__init__(msg, error_code=error_code, details=extra_details, username=username)


class InvalidTokenError(AuthError):
    """Raised when a token is malformed, expired, revoked, or otherwise invalid."""

    def __init__(
        self,
        reason: str = "Token invalid",
        token_hint: Optional[str] = None,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.reason = reason
        self.token_hint = token_hint
        msg = reason
        if token_hint:
            msg = f"{reason} (token: {token_hint}...)"
        extra_details = {"reason": reason, "token_hint": token_hint}
        if details:
            extra_details.update(details)
        super().__init__(msg, error_code=error_code, details=extra_details)


# ============================================================================
# Base exception for shared memory / security helpers
# ============================================================================
class FunctionsMemoryError(Exception):
    """Base exception for shared memory/security helpers."""
    pass


class CredentialPolicyError(FunctionsMemoryError):
    """Raised when credential policy configuration is invalid or validation fails."""
    def __init__(self, reason: str, violations: list[str] | None = None):
        self.reason = reason
        self.violations = violations or []

        message = reason
        if self.violations:
            message = f"{reason}: " + "; ".join(self.violations)

        super().__init__(message)


class PasswordHashingError(FunctionsMemoryError):
    """Raised when password hashing parameters or derivation operations fail."""
    def __init__(self, operation: str, reason: str):
        self.operation = operation
        self.reason = reason
        super().__init__(f"Password hashing failed during {operation}: {reason}")


class CacheConfigurationError(FunctionsMemoryError):
    """Raised when TTL cache configuration is invalid."""
    def __init__(self, parameter: str, value: object, reason: str):
        self.parameter = parameter
        self.value = value
        self.reason = reason
        super().__init__(
            f"Invalid cache configuration for '{parameter}' with value {value!r}: {reason}"
        )


class StoreError(FunctionsMemoryError):
    """Base exception for portable store operations."""
    pass


class StoreLockError(StoreError):
    """Raised when file locking fails or times out."""
    def __init__(self, path: str, reason: str, timeout: float | None = None):
        self.path = path
        self.reason = reason
        self.timeout = timeout

        message = f"Failed to acquire store lock for {path}: {reason}"
        if timeout is not None:
            message += f" (timeout={timeout}s)"

        super().__init__(message)


class StoreSerializationError(StoreError):
    """Raised when a payload cannot be serialized to JSON."""
    def __init__(self, path: str, reason: str):
        self.path = path
        self.reason = reason
        super().__init__(f"Failed to serialize payload for {path}: {reason}")


class StoreLoadError(StoreError):
    """Raised when loading or decoding persisted data fails."""
    def __init__(self, path: str, reason: str):
        self.path = path
        self.reason = reason
        super().__init__(f"Failed to load persisted data from {path}: {reason}")


class StoreSaveError(StoreError):
    """Raised when saving persisted data fails."""
    def __init__(self, path: str, reason: str):
        self.path = path
        self.reason = reason
        super().__init__(f"Failed to save persisted data to {path}: {reason}")

# ============================================================================
# Base exception for rate limiting
# ============================================================================
class RateLimitError(Exception):
    """Base exception for rate limiting errors."""
    pass


class RateLimitExceeded(RateLimitError):
    """Raised when a rate limit is exceeded."""
    def __init__(self, key: str, retry_after: float = 0):
        self.key = key
        self.retry_after = retry_after
        msg = f"Rate limit exceeded for key: {key}"
        if retry_after:
            msg += f" (retry after {retry_after:.1f} seconds)"
        super().__init__(msg)


class RateLimitConfigurationError(RateLimitError):
    """Raised when rate limiter configuration is invalid."""
    pass


# ============================================================================
# Base exception for email services
# ============================================================================
class EmailError(Exception):
    """Base exception for email sending failures."""
    pass


class EmailSendError(EmailError):
    """Raised when sending fails (network, server, etc.)."""
    def __init__(self, recipient: str, reason: str):
        self.recipient = recipient
        self.reason = reason
        super().__init__(f"Failed to send email to {recipient}: {reason}")


class EmailTemplateError(EmailError):
    """Raised when template rendering fails."""
    def __init__(self, template_name: str, reason: str):
        self.template_name = template_name
        self.reason = reason
        super().__init__(f"Template error for {template_name}: {reason}")


class EmailAuthError(EmailError):
    """Raised when authentication with the SMTP server fails."""
    pass


class EmailConfigurationError(EmailError):
    """Raised when email service configuration is invalid."""
    pass


# ============================================================================
# Base exception for storage operations
# ============================================================================
class StorageError(Exception):
    """Base exception for storage operations."""
    pass


class StorageNotFoundError(StorageError):
    """Raised when a requested file or path does not exist."""
    def __init__(self, path: str):
        self.path = path
        super().__init__(f"Path not found: {path}")


class StoragePermissionError(StorageError):
    """Raised when a storage operation fails due to permissions."""
    def __init__(self, path: str, operation: str = "access"):
        self.path = path
        self.operation = operation
        super().__init__(f"Permission denied for {operation} on {path}")


class StorageUploadError(StorageError):
    """Raised when file upload fails."""
    def __init__(self, path: str, reason: str):
        self.path = path
        self.reason = reason
        super().__init__(f"Upload failed for {path}: {reason}")


class StorageDownloadError(StorageError):
    """Raised when file download fails."""
    def __init__(self, path: str, reason: str):
        self.path = path
        self.reason = reason
        super().__init__(f"Download failed for {path}: {reason}")


class StorageDeleteError(StorageError):
    """Raised when file deletion fails."""
    def __init__(self, path: str, reason: str):
        self.path = path
        self.reason = reason
        super().__init__(f"Delete failed for {path}: {reason}")


class StorageQuotaError(StorageError):
    """Raised when storage quota is exceeded."""
    pass


class StorageBackendError(StorageError):
    """Raised when a backend‑specific error occurs."""
    def __init__(self, backend: str, reason: str):
        self.backend = backend
        self.reason = reason
        super().__init__(f"{backend} backend error: {reason}")


# ============================================================================
# Base exception for transport operations
# ============================================================================
class TransportError(Exception):
    """Base exception for all transport layer communications."""

    def __init__(
        self,
        message: str = "Transport error",
        adapter_name: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.adapter_name = adapter_name
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"error": self.message, "type": self.__class__.__name__}
        if self.adapter_name:
            result["adapter"] = self.adapter_name
        if self.details:
            result["details"] = self.details
        return result


class TransportChannelError(TransportError):
    """Raised when an adapter/channel cannot connect or becomes unavailable."""

    def __init__(
        self,
        adapter_name: str,
        reason: str = "Channel unavailable",
        channel_state: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.reason = reason
        self.channel_state = channel_state
        msg = f"Channel error on '{adapter_name}': {reason}"
        if channel_state:
            msg += f" (state={channel_state})"
        extra_details = {"reason": reason, "state": channel_state}
        if details:
            extra_details.update(details)
        super().__init__(msg, adapter_name=adapter_name, details=extra_details)


class TransportRetryExhausted(TransportError):
    """Raised when all retry attempts for a transmission have failed."""

    def __init__(
        self,
        adapter_name: str,
        attempts: int,
        last_error: Optional[str] = None,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.attempts = attempts
        self.last_error = last_error
        if not message:
            msg = f"Adapter '{adapter_name}' exhausted {attempts} retries"
            if last_error:
                msg += f": {last_error}"
        else:
            msg = message
        extra_details = {"attempts": attempts, "last_error": last_error}
        if details:
            extra_details.update(details)
        super().__init__(msg, adapter_name=adapter_name, details=extra_details)


# ============================================================================
# Search engine exceptions
# ============================================================================
class SearchError(Exception):
    """Base exception for search engine errors."""
    pass


class IndexBuildError(SearchError):
    """Raised when building an index fails."""
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(f"Index build failed: {reason}")


class IndexLoadError(SearchError):
    """Raised when loading an index from disk fails."""
    def __init__(self, path: str, reason: str):
        self.path = path
        self.reason = reason
        super().__init__(f"Failed to load index from {path}: {reason}")


class IndexSaveError(SearchError):
    """Raised when saving an index to disk fails."""
    def __init__(self, path: str, reason: str):
        self.path = path
        self.reason = reason
        super().__init__(f"Failed to save index to {path}: {reason}")


class DocumentNotFoundError(SearchError):
    """Raised when a document ID does not exist."""
    def __init__(self, doc_id: int):
        self.doc_id = doc_id
        super().__init__(f"Document with id {doc_id} not found")


class InvalidAnalyzerError(SearchError):
    """Raised when an analyzer does not implement the required protocol."""
    pass


class InconsistentFieldsError(SearchError):
    """Raised when adding a document with a different field set."""
    def __init__(self, expected: list, got: list):
        self.expected = expected
        self.got = got
        super().__init__(f"Field mismatch: expected {expected}, got {got}")
