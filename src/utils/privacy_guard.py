import re
import random
import numpy as np

class PrivacyGuard:
    """Privacy filter and differential privacy noise engine"""

    @staticmethod
    def scrub(text: str) -> str:
        """Remove or mask personally identifiable information (PII)"""
        patterns = {
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b': '[EMAIL]',
            r'\b\d{3}[-.\s]??\d{2}[-.\s]??\d{4}\b': '[SSN]',
            r'\b\d{10}\b': '[PHONE]',
            r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b': '[IP]',
        }
        for pattern, repl in patterns.items():
            text = re.sub(pattern, repl, text)
        return text

    @staticmethod
    def apply_differential_privacy(value: float, epsilon: float = 1.0) -> float:
        """Applies Laplace noise to protect individual contribution"""
        scale = 1.0 / epsilon
        noise = np.random.laplace(0, scale)
        return max(0.0, value + noise)
