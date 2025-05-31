
class NaNException(Exception):
    """Raised when a NaN is encountered during learning"""
    def __init__(self, message="NaN value detected in training"):
        super().__init__(message)

class GradientExplosionError(Exception):
    """Raised when gradient norms exceed a safety threshold"""
    def __init__(self, norm, threshold=1e3):
        super().__init__(f"Gradient explosion detected: norm={norm:.2f}, threshold={threshold}")

class InvalidActionError(Exception):
    """Raised when an action fails safety validation or is undefined"""
    def __init__(self, action=None):
        message = f"Invalid or unsafe action: {action}" if action else "Invalid or unsafe action encountered"
        super().__init__(message)

class InvalidConfigError(Exception):
    """Raised when agent configuration validation fails"""
    def __init__(self, message="Invalid agent configuration"):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return f'InvalidConfigError: {self.message}'
