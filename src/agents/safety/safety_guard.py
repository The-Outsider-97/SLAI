import re
import logging
from typing import List, Tuple


class SafetyGuard:
    """Multi-layered content safety system for redaction, filtering, and escalation"""
    def __init__(self, redact_patterns: List[Tuple[str, str]] = None, toxicity_patterns: List[str] = None):
        self.logger = logging.getLogger("SafetyGuard")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            self.logger.addHandler(handler)

        self.redact_patterns = redact_patterns or [
            (r'\b\d{3}-\d{2}-\d{4}\b', '[REDACTED_SSN]'),
            (r'(?i)\b(credit card|password|email|address)\b', '[REDACTED_PII]')
        ]

        self.toxicity_patterns = toxicity_patterns or [
            r'\b(kill|harm|attack|suicide|murder)\b',
            r'(racial|ethnic|religious)\s+slur',
            r'(hate speech|genocide|terrorist|bomb)'
        ]

    def sanitize(self, text: str) -> str:
        """Apply redaction and toxicity filtering"""
        original = text
        # Redaction layer
        for pattern, replacement in self.redact_patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # Toxicity check
        for pattern in self.toxicity_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                self.logger.warning(f"Blocked toxic content: {pattern} matched in input")
                return "[SAFETY_BLOCK] Content violates safety policy"

        if original != text:
            self.logger.info("Text was sanitized.")
        return text

    def is_safe(self, text: str) -> bool:
        """Check if text is free from toxic or sensitive content"""
        for pattern in self.redact_patterns:
            if re.search(pattern[0], text, re.IGNORECASE):
                return False
        for pattern in self.toxicity_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return False
        return True

    def get_triggered_patterns(self, text: str) -> List[str]:
        """Return a list of patterns that triggered redaction or blocking"""
        triggers = []
        for pattern, _ in self.redact_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                triggers.append(f"Redact: {pattern}")
        for pattern in self.toxicity_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                triggers.append(f"Toxicity: {pattern}")
        return triggers
