class AuthError(Exception):
    """Base auth exception."""
    pass

class UserAlreadyExistsError(AuthError):
    pass

class InvalidCredentialsError(AuthError):
    pass

class AccountLockedError(AuthError):
    pass

class InvalidTokenError(AuthError):
    pass

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
