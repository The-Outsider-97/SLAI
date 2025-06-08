
import re
import html
from typing import Union

from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Input Sanitizer")
printer = PrettyPrinter

class InputSanitizer:
    """Utility class for input sanitization and validation."""

    _paranoid_mode = False

    @classmethod
    def enable_paranoid_mode(cls):
        """Activate strict sanitization policies for critical operations."""
        cls._paranoid_mode = True
        logger.warning("InputSanitizer is now running in paranoid mode.")

    @classmethod
    def sanitize_text(cls, text: str) -> str:
        """Basic text sanitization."""
        if not isinstance(text, str):
            return text

        # Escape HTML and remove potentially harmful characters
        text = html.escape(text)
        text = re.sub(r'[^\w\s\-\.,!?]', '', text)
        if cls._paranoid_mode:
            text = re.sub(r'\s+', ' ', text).strip()

        return text

    @classmethod
    def sanitize_dict(cls, data: dict) -> dict:
        """Sanitize all string values in a dictionary."""
        return {
            k: cls.sanitize_text(v) if isinstance(v, str) else v
            for k, v in data.items()
        }

    @classmethod
    def validate_input(cls, value: Union[str, int, float], allow_null: bool = False) -> bool:
        """Basic validation for input values."""
        if value is None:
            return allow_null
        if isinstance(value, str):
            return bool(re.match(r'^[\w\s\-.,!?]*$', value))
        return True
