import re
import logging
import numpy as np
from typing import List, Tuple, Optional, Dict

from src.agents.safety.security_error import ToxicContentError, PrivacyViolationError

class SafetyGuard:
    """Unified privacy and safety framework with layered protection"""
    
    def __init__(
        self,
        redact_patterns: Optional[List[Tuple[str, str]]] = None,
        toxicity_patterns: Optional[List[str]] = None,
        privacy_params: Optional[Dict] = None
    ):
        """
        Initialize safety system with configurable protections
        
        Args:
            redact_patterns: List of (regex pattern, replacement) tuples
            toxicity_patterns: List of regex patterns for harmful content
            privacy_params: Differential privacy configuration {
                'epsilon': 1.0,
                'sensitivity': 1.0,
                'mechanism': 'laplace'
            }
        """
        self.logger = logging.getLogger("SafetyGuard")
        self._configure_logging()

        # Initialize protection patterns
        self.redact_patterns = redact_patterns or self._default_redaction_patterns()
        self.toxicity_patterns = toxicity_patterns or self._default_toxicity_patterns()
        
        # Configure differential privacy
        self.privacy_params = privacy_params or {'epsilon': 1.0, 'sensitivity': 1.0}
        self._validate_privacy_params()

    def _configure_logging(self):
        """Ensure proper logging setup"""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s"))
            self.logger.addHandler(handler)

    def _default_redaction_patterns(self):
        """Combined PII patterns from both original classes"""
        return [
            (r'\b\d{3}-\d{2}-\d{4}\b', '[REDACTED_SSN]'),
            (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b', '[REDACTED_EMAIL]'),
            (r'\b\d{10}\b', '[REDACTED_PHONE]'),
            (r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '[REDACTED_IP]'),
            (r'(?i)\b(credit card|password|address)\b', '[REDACTED_PII]'),
            (r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b', '[REDACTED_CC]')
        ]

    def _default_toxicity_patterns(self):
        """Enhanced toxicity detection patterns"""
        return [
            r'\b(kill|harm|attack|suicide|murder)\b',
            r'(racial|ethnic|religious)\s+slur',
            r'(hate speech|genocide|terrorist|bomb)',
            r'\b(abuse|harass|threaten|stalk)\b',
            r'(self-harm|eating disorder|suicidal ideation)'
        ]

    def _validate_privacy_params(self):
        """Ensure valid differential privacy configuration"""
        if self.privacy_params['epsilon'] <= 0:
            raise ValueError("Epsilon must be positive for differential privacy")
        if self.privacy_params.get('mechanism', 'laplace') not in ['laplace', 'gaussian']:
            raise ValueError("Invalid privacy mechanism")

    def sanitize(self, text: str, depth: str = 'full') -> str:
        """
        Multi-stage content sanitization pipeline
        
        Args:
            text: Input content to sanitize
            depth: Protection level ('minimal', 'balanced', 'full')
        
        Returns:
            Sanitized text with PII removed and safety checks applied
        """
        original = text
        protection_stack = []

        # Stage 1: Basic Redaction
        text = self._apply_redaction(text, protection_stack)
        
        # Stage 2: Depth-based protection
        if depth == 'full':
            text = self._apply_context_aware_protection(text, protection_stack)
        
        # Stage 3: Toxicity screening
        toxicity_risk = self._detect_toxicity(text)
        if toxicity_risk['risk_level'] != 'clean':
            self._handle_toxicity_event(toxicity_risk)
            return "[SAFETY_BLOCK] Content violates safety policy"

        # Post-sanitization validation
        if original != text:
            self.logger.info(f"Sanitized content. Changes: {protection_stack}")
            self._post_sanitization_checks(text)

        return text

    def _apply_redaction(self, text: str, stack: list) -> str:
        """Core pattern-based redaction engine"""
        for pattern, replacement in self.redact_patterns:
            modified = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
            if modified != text:
                stack.append(f"Redacted: {pattern}")
                text = modified
        return text

    def _detect_toxicity(self, text: str) -> dict:
        """Advanced toxicity analysis with risk scoring"""
        risk_assessment = {
            'risk_level': 'clean',
            'triggers': [],
            'confidence': 0.0
        }

        for pattern in self.toxicity_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                risk_assessment['triggers'].append(pattern)
                risk_assessment['confidence'] += 0.25

        if risk_assessment['confidence'] > 0:
            risk_assessment['risk_level'] = 'high' if risk_assessment['confidence'] >= 0.75 else 'medium'
        
        return risk_assessment

    def apply_differential_privacy(self, value: float, dataset_size: int = 1) -> float:
        """
        Privacy-preserving noise injection
        
        Args:
            value: Original value to protect
            dataset_size: Number of contributors for sensitivity scaling
            
        Returns:
            Differentially private value
        """
        sensitivity = self.privacy_params['sensitivity'] / dataset_size
        epsilon = self.privacy_params['epsilon']
        
        if self.privacy_params.get('mechanism', 'laplace') == 'laplace':
            noise = np.random.laplace(0, sensitivity/epsilon)
        else:
            noise = np.random.normal(0, sensitivity * np.sqrt(2*np.log(1.25/0.01))/epsilon)
            
        return max(0.0, value + noise)

    def _handle_toxicity_event(self, risk_assessment: dict):
        """Centralized toxicity response handling"""
        self.logger.warning(
            f"Blocked toxic content - Risk: {risk_assessment['risk_level']} "
            f"Triggers: {risk_assessment['triggers']}"
        )
        raise ToxicContentError(
            patterns=risk_assessment['triggers'],
            risk_level=risk_assessment['risk_level']
        )

    def _post_sanitization_checks(self, text: str):
        """Final safety validation after sanitization"""
        for pattern, _ in self.redact_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                raise PrivacyViolationError(
                    pattern=pattern,
                    content=text
                )

    def get_protection_report(self, text: str) -> dict:
        """Comprehensive safety analysis report"""
        return {
            'pii_detected': self._detect_pii(text),
            'toxicity_risk': self._detect_toxicity(text),
            'privacy_config': self.privacy_params
        }

    def _detect_pii(self, text: str) -> list:
        """Identify present PII types in content"""
        detected = []
        for pattern, _ in self.redact_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                detected.append(pattern)
        return detected

    def is_compliant(self, text: str, standard: str = 'gdpr') -> bool:
        """Check compliance with data protection standards"""
        pii_detected = self._detect_pii(text)
        if not pii_detected:
            return True
            
        if standard == 'gdpr':
            return len(pii_detected) == 0
        elif standard == 'hipaa':
            return not any('SSN' in p or 'PHONE' in p for p in pii_detected)
        
        return False

# Enhanced error classes (extend if needed)
class EnhancedToxicContentError(ToxicContentError):
    """Adds risk level context to toxicity errors"""
    def __init__(self, message, patterns, risk_level):
        super().__init__(message, patterns)
        self.risk_level = risk_level

class EnhancedPrivacyViolationError(PrivacyViolationError):
    """Adds compliance context to privacy errors"""
    def __init__(self, message, pattern, content, standard):
        super().__init__(message, pattern, content)
        self.compliance_standard = standard
