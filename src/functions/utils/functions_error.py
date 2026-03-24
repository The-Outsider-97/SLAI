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