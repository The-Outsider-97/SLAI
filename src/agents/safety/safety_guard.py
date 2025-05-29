import re
import yaml, json
import numpy as np

from typing import List, Tuple, Optional, Dict

from src.agents.safety.utils.security_error import(
    ToxicContentError, PrivacyViolationError,
    PiiLeakageError, MisinformationError,
    PromptInjectionError
)
from src.agents.safety.secure_memory import SecureMemory
from logs.logger import get_logger

logger = get_logger("Safety Guard")

CONFIG_PATH = "src/agents/safety/configs/secure_config.yaml"

def load_config(config_path=CONFIG_PATH):
    with open(config_path, "r", encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

class SafetyGuard:
    """Unified privacy and safety framework with layered protection"""
    
    def __init__(
        self, config,
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
        config = load_config() or {}
        self.config = config.get('safety_guard', {})
        self.logger = logger
        memory = SecureMemory(config)
        self.memory = memory
        self.privacy_params = privacy_params or {'epsilon': 1.0, 'sensitivity': 1.0}
        
        # Load patterns from JSON files
        self.redact_patterns = self._load_pii_patterns()
        self.toxicity_patterns = self._load_toxicity_patterns()
        self.authority_phrases = self._load_authority_phrases()
        self.manipulation_patterns = self._load_manipulation_patterns()
        self.group_targeting_patterns = self._load_group_targeting_patterns()
        self.boundary_phrases = self._load_boundary_phrases()
        
        self._validate_privacy_params()

    def _load_pii_patterns(self) -> List[Tuple[str, str]]:
        """Load PII patterns from JSON file"""
        path = self.config.get('pii_patterns_path', '')
        if not path:
            self.logger.warning("PII patterns path not configured, using defaults")
            return self._default_redaction_patterns()
            
        try:
            with open(path, 'r', encoding='utf-8') as f:
                patterns = json.load(f)
            return [(item['pattern'], item['replacement']) for item in patterns]
        except Exception as e:
            self.logger.error(f"Failed to load PII patterns: {str(e)}")
            return self._default_redaction_patterns()

    def _load_authority_phrases(self) -> List[str]:
        """Load authority escalation phrases from JSON file"""
        path = self.config.get('authority_phrases_path', '')
        default_patterns = [ # Default patterns if file loading fails or path not specified
            r"speak to your supervisor",
            r"let me talk to a human",
            r"real person"
        ]
        if not path:
            self.logger.warning("Authority phrases path not configured, using defaults")
            return default_patterns
            
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load authority phrases: {str(e)}")
            return default_patterns

    def _load_manipulation_patterns(self) -> List[str]:
        """Load conversational manipulation patterns from JSON file"""
        path = self.config.get('manipulation_patterns_path', '')
        default_patterns = [ # Default patterns if file loading fails or path not specified
            r"ignore (?:the|your) previous",
            r"forget what I said",
            r"start over",
            r"as a friend"
        ]
        if not path:
            self.logger.warning("Manipulation patterns path not configured, using defaults")
            return default_patterns
            
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load manipulation patterns: {str(e)}")
            return default_patterns

    def _load_group_targeting_patterns(self) -> List[str]:
        """Load group targeting patterns from JSON file"""
        path = self.config.get('group_targeting_patterns_path', '')
        default_patterns = [ # Default patterns if file loading fails or path not specified
            r"your people",
            r"you \w+ are all", # \w+ is broader than just a letter group
            r"your community"
        ]
        if not path:
            self.logger.warning("Group targeting patterns path not configured, using defaults")
            return default_patterns
            
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load group targeting patterns: {str(e)}")
            return default_patterns

    def _load_toxicity_patterns(self) -> List[str]:
        """Load toxicity patterns from JSON file"""
        path = self.config.get('toxicity_patterns_path', '')
        if not path:
            self.logger.warning("Toxicity patterns path not configured, using defaults")
            return self._default_toxicity_patterns()
            
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load toxicity patterns: {str(e)}")
            return self._default_toxicity_patterns()

    def _load_boundary_phrases(self) -> List[str]:
        """Load boundary testing phrases from JSON file"""
        path = self.config.get('boundary_phrases_path', '')
        default_patterns = [
            r"what if I said",
            r"just hypothetically",
            r"suppose I wanted"
        ]
        if not path:
            self.logger.warning("Boundary phrases path not configured, using defaults")
            return default_patterns
            
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load boundary phrases: {str(e)}")
            return default_patterns

    def _default_redaction_patterns(self):
        """Fallback PII patterns"""
        return [
            (r'\b\d{3}-\d{2}-\d{4}\b', '[REDACTED_SSN]'),
            (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b', '[REDACTED_EMAIL]'),
            (r'\b\d{10}\b', '[REDACTED_PHONE]'),
            (r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '[REDACTED_IP]'),
            (r'(?i)\b(credit card|password|address)\b', '[REDACTED_PII]'),
            (r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b', '[REDACTED_CC]')
        ]

    def _default_toxicity_patterns(self):
        """Fallback toxicity patterns"""
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
        if not isinstance(text, str):
            raise ValueError("sanitize() expects a valid non-null string input")
        original = text
        protection_stack = []

        # Stage 1: Basic Redaction
        text = self._apply_redaction(text, protection_stack)

        self.detect_prompt_injection(text)
        self.detect_misinformation(text)
    
        # Stage 2: Depth-based protection
        if depth == 'full':
            text = self._apply_context_aware_protection(text, protection_stack)
        
        # Stage 3: Toxicity screening
        toxicity_risk = self._detect_toxicity(text)
        if toxicity_risk['risk_level'] != 'clean':
            self._handle_toxicity_event(toxicity_risk, text)
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

    def _handle_toxicity_event(self, risk_assessment: dict, text: str):
        self.logger.warning(
            f"Blocked toxic content - Risk: {risk_assessment['risk_level']} "
            f"Triggers: {risk_assessment['triggers']}"
        )
        raise ToxicContentError(
            pattern=", ".join(risk_assessment['triggers']),
            content=text,
            classification_details={
                'risk_level': risk_assessment['risk_level'],
                'confidence': risk_assessment['confidence']
            }  # Add metadata
        )

    def _post_sanitization_checks(self, text: str):
        for pattern, _ in self.redact_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                raise PiiLeakageError(
                    data_description="Detected PII",
                    leakage_source="Content sanitization",
                    suspected_impact="Privacy violation"
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

    def detect_prompt_injection(self, prompt: str) -> bool:
        """Detect prompt injection patterns"""
        injection_patterns = self.config.get('injection_patterns', [r'ignore_previous_instructions'])
        for pattern in injection_patterns:
            if re.search(pattern, prompt, re.IGNORECASE):
                raise PromptInjectionError(
                    detected_pattern=pattern,
                    original_prompt=prompt
                )
        return False
    
    def detect_misinformation(self, content: str) -> bool:
        """Check for factual inaccuracies"""
        # Placeholder: Integrate with fact-checking API
        if "misinformation_trigger_phrase" in content:  # Example logic
            raise MisinformationError(
                content=content,
                identified_falsehood="Sample false claim",
                confidence_of_falsehood=0.95
            )
        return False
    
    def _apply_context_aware_protection(self, text: str, stack: list) -> str:
        """
        Enhanced protection using dialogue context for sophisticated safety measures.
        Applies:
        - Context-aware PII detection
        - Conversational pattern analysis
        - Context-sensitive toxicity screening
        - Temporal risk assessment
        """
        from src.agents.language.dialogue_context import DialogueContext
        
        # Load language config from YAML file
        LANGUAGE_CONFIG_PATH = "src/agents/language/configs/language_config.yaml"
        try:
            with open(LANGUAGE_CONFIG_PATH, 'r', encoding='utf-8') as f:
                language_config = yaml.safe_load(f)
            dialogue_context_settings = language_config.get("dialogue_context_settings", {})
        except Exception as e:
            self.logger.error(f"Failed to load language config: {str(e)}")
            dialogue_context_settings = {}
        
        # Use actual dialogue context object
        if not hasattr(self, "dialogue_context"):
            self.dialogue_context = DialogueContext(config={"dialogue_context_settings": dialogue_context_settings})
        
        dialogue_context = self.dialogue_context
        dialogue_context.add_message("user", text)
    
        context_text = dialogue_context.get_context_for_prompt(
            include_summary=True, include_history=True, history_messages_window=5
        )

        # 1. Context-aware PII detection
        text = self._context_aware_redaction(text, context_text, stack)

        # 2. Conversational pattern analysis
        risk_score = self._analyze_conversational_patterns(text, context_text)
        if risk_score > 0.7:
            stack.append("High-risk conversational pattern detected")
            return "[SAFETY_BLOCK] Suspicious conversational pattern"

        # 3. Context-sensitive toxicity screening
        toxicity_risk = self._context_sensitive_toxicity(text, context_text)
        if toxicity_risk['risk_level'] == 'high':
            stack.append(f"Context-sensitive toxicity: {toxicity_risk['triggers']}")
            self._handle_toxicity_event(toxicity_risk, text)
            return "[SAFETY_BLOCK] Context-sensitive safety violation"

        # 4. Temporal risk assessment
        if self._is_risk_escalation(text, context_text):
            stack.append("Risk escalation detected")
            return "[SAFETY_BLOCK] Risk escalation in conversation"

        return text

    def _context_aware_redaction(self, text: str, context: str, stack: list) -> str:
        """Enhanced redaction based on conversation context"""
        # Apply standard redaction first
        text = self._apply_redaction(text, stack)
        
        # Additional context-sensitive redaction
        sensitive_contexts = self.config.get('sensitive_contexts', {})
        
        for topic, patterns in sensitive_contexts.items():
            if re.search(topic, context, re.IGNORECASE):
                for pattern, replacement in patterns:
                    modified = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
                    if modified != text:
                        stack.append(f"Context-redacted: {pattern} ({topic} context)")
                        text = modified
        return text

    def _analyze_conversational_patterns(self, text: str, context: str) -> float:
        """Detect suspicious conversational patterns"""
        risk_score = 0.0
        
        # 1. Check for repeated sensitive topic requests
        if re.search(r"(again|repeat|re-send|re-?enter)", text, re.IGNORECASE):
            sensitive_topics = ['password', 'ssn', 'credit card', 'security code']
            if any(topic in context for topic in sensitive_topics):
                risk_score += 0.4

        # 2. Check for authority escalation
        if any(re.search(phrase, text, re.IGNORECASE) for phrase in self.authority_phrases):
            # Higher risk if preceded by sensitive requests
            if re.search(r"(account|balance|password|ssn)", context, re.IGNORECASE):
                risk_score += 0.6

        # 3. Check for conversational manipulation
        if any(re.search(pattern, text, re.IGNORECASE) for pattern in self.manipulation_patterns):
            risk_score += 0.3
            
        return min(1.0, risk_score)

    def _context_sensitive_toxicity(self, text: str, context: str) -> dict:
        """Toxicity analysis considering conversation context"""
        base_risk = self._detect_toxicity(text)
        
        # Increase risk if toxicity continues a harmful context
        if base_risk['risk_level'] != 'clean':
            # Check if previous context already contained toxicity
            context_risk = self._detect_toxicity(context)
            if context_risk['risk_level'] != 'clean':
                base_risk['confidence'] = min(1.0, base_risk['confidence'] + 0.3)
                base_risk['risk_level'] = 'high' if base_risk['confidence'] >= 0.6 else 'medium'
        
        # Increase risk for targeted harassment
        if self._is_targeted_harassment(text, context):
            base_risk['confidence'] = 1.0
            base_risk['risk_level'] = 'high'
            base_risk['triggers'].append('targeted_harassment')
            
        return base_risk

    def _is_targeted_harassment(self, text: str, context: str) -> bool:
        """Detect targeted harassment patterns"""
        if any(re.search(pattern, text, re.IGNORECASE) for pattern in self.group_targeting_patterns):
            identity_markers = self.config.get('identity_markers_for_harassment', 
                                               ['gender', 'race', 'religion', 'orientation', 
                                                'nationality', 'ethnicity', 'disability', 
                                                'political affiliation', 'social class']) # Example expanded list

            if any(marker.lower() in context.lower() for marker in identity_markers):
                self.logger.debug(f"Potential targeted harassment: Matched pattern in text and identity marker in context.")
                return True
            
            # A simpler check: if a targeting pattern is found, and the context is generally about social issues or identity
            if re.search(r"\b(social|identity|group|community|rights|discrimination|equality|diversity|inclusion)\b", context, re.IGNORECASE):
                 self.logger.debug(f"Potential targeted harassment: Matched pattern in text and general identity discussion in context.")
                 return True
                
        return False

    def _is_risk_escalation(self, text: str, context: str) -> bool:
        """Detect escalation in risk level compared to conversation history"""
        current_risk = self._detect_toxicity(text)['confidence']
        
        # Analyze context for previous risk level
        context_risk = self._detect_toxicity(context)['confidence']
        
        # Significant escalation (e.g., from 0.2 to 0.8)
        if current_risk > context_risk + 0.5:
            return True
            
        # Check for repeated boundary testing
        if any(re.search(phrase, text, re.IGNORECASE) for phrase in self.boundary_phrases):
            # Verify if previous context contained similar probing
            if re.search(r"(hypothetical|suppose|what if)", context, re.IGNORECASE):
                return True
                
        return False

    def is_minimal_viable(self) -> bool:
        """
        Checks if SafetyGuard has the essential components required to enforce basic safety constraints.
        Used by fallback systems to ensure degraded but functional safety.
    
        Minimal viability requires:
        - Basic PII redaction patterns
        - At least one toxicity pattern
        - Valid differential privacy configuration
        """
        # 1. Check basic PII patterns
        has_pii_protection = bool(self.redact_patterns)
    
        # 2. Check basic toxicity screening
        has_toxicity_screening = bool(self.toxicity_patterns)
    
        # 3. Validate differential privacy configuration
        try:
            self._validate_privacy_params()
            dp_valid = True
        except ValueError:
            dp_valid = False
    
        return has_pii_protection and has_toxicity_screening and dp_valid

if __name__ == "__main__":
    print("\n=== Running Security Reward Model ===\n")
    config = load_config()
    privacy_params = {}

    guard = SafetyGuard(config=config, privacy_params=privacy_params)

    logger.info(f"{guard}")
    print(guard._validate_privacy_params())
    print(f"\n* * * * * Phase 2 * * * * *\n")
    text="The user is called Joe Doe email is john.doe@example.com and SSN is 123-45-6789 and phone number is +31 04384712."
    depth = 'full'

    pipeline = guard.sanitize(text=text, depth=depth)

    logger.info(f"{pipeline}")
    print("\n=== Successfully Ran Security Reward Model ===\n")
