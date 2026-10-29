"""Animation-domain errors; no import-time fallback."""
from src.agents.base.utils.base_errors import BaseError, BaseErrorType

class AnimationError(BaseError):
    error_type = BaseErrorType.RUNTIME
    default_code = "ANI-1000"
    def __init__(self, message, **kwargs):
        super().__init__(message, code=self.default_code, **kwargs)

class AnimationRevisionConflictError(AnimationError):
    default_code = "ANI-1200"

class AnimationUnsupportedOperationError(AnimationError):
    default_code = "ANI-1400"
