import re
import json

from typing import Dict, List, Callable, Optional

from src.agents.safety.utils.config_loader import load_global_config, get_config_section
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Security Score Model")
printer = PrettyPrinter

class ScoreModel:
    def __init__(self):
        self.config = load_global_config()
        self.score_config = get_config_section('score_model')
        self.guard_config = get_config_section('safety_guard')
        self.pii_patterns_path =  self.guard_config.get('pii_patterns_path')
        
        # Load scoring components
        self._load_scoring_components()
        logger.info("Security Score Model initialized with config-driven scoring rules")

    def _load_scoring_components(self):
        """Initialize all scoring components from configuration"""
        # Term-based scoring components
        self.scoring_components = {
            "alignment": {
                "terms": self._load_terms('harmful_terms_path'),
                "factor": self.score_config.get('alignment_factor', 5.0),
                "scorer": self._term_based_scorer
            },
            "helpfulness": {
                "terms": self._load_terms('helpful_terms_path'),
                "factor": self.score_config.get('helpfulness_factor', 3.0),
                "scorer": self._term_based_scorer
            },
            "safety": {
                "terms": self._load_terms('risk_terms_path'),
                "factor": self.score_config.get('safety_factor', 2.0),
                "scorer": self._term_based_scorer
            },
            "truthfulness": {
                "terms": self._load_terms('misinformation_terms_path'),
                "factor": self.score_config.get('truthfulness_factor', 5.0),
                "scorer": self._term_based_scorer
            },
            "privacy": {
                "patterns": self._load_pii_patterns(),
                "factor": self.score_config.get('privacy_factor', 0.2),
                "scorer": self._pattern_based_scorer
            }
        }
        
        # Special context handlers
        self.context_handlers = {
            "user_creation": self._handle_user_creation,
            "financial": self._handle_financial_context,
            "medical": self._handle_medical_context
        }

    def _load_terms(self, config_key: str) -> List[str]:
        """Load terms from file path specified in config"""
        path = self.score_config.get(config_key)
        if not path:
            logger.warning(f"No path specified for {config_key}, using defaults")
            return self._get_default_terms(config_key)
        
        try:
            with open(path, 'r') as f:
                return [line.strip() for line in f if line.strip()]
        except Exception as e:
            logger.error(f"Error loading terms from {path}: {e}")
            return self._get_default_terms(config_key)

    def _get_default_terms(self, component: str) -> List[str]:
        """Provide default terms for scoring components"""
        defaults = {
            "harmful_terms_path": ["harm", "hate", "discriminate", "illegal", "violence"],
            "helpful_terms_path": ["help", "assist", "support", "guide", "answer"],
            "risk_terms_path": ["dangerous", "unsafe", "risk", "hazard", "toxic"],
            "misinformation_terms_path": ["fake", "hoax", "conspiracy", "myth", "false"]
        }
        return defaults.get(component, [])

    def _load_pii_patterns(self) -> List[str]:
        """Load PII regex patterns from configured path"""
        path = self.pii_patterns_path
        if not path:
            logger.warning("No PII patterns path configured, using defaults")
            return self._get_default_pii_patterns()
        
        try:
            # Load JSON file with pattern objects
            with open(path, 'r') as f:
                pattern_objects = json.load(f)
            
            # Extract patterns from JSON objects
            patterns = [obj["pattern"] for obj in pattern_objects if "pattern" in obj]
            
            # Validate patterns
            valid_patterns = []
            for pattern in patterns:
                try:
                    re.compile(pattern)
                    valid_patterns.append(pattern)
                except re.error as e:
                    logger.warning(f"Invalid regex pattern skipped: {pattern} - {str(e)}")
            
            if not valid_patterns:
                logger.error("No valid patterns found in PII patterns file, using defaults")
                return self._get_default_pii_patterns()
                
            return valid_patterns
        except Exception as e:
            logger.error(f"Error loading PII patterns: {e}")
            return self._get_default_pii_patterns()

    def _get_default_pii_patterns(self) -> List[str]:
        """Default PII detection patterns"""
        return [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b\d{3}\.\d{3}\.\d{4}\b',  # Phone
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b\d{4} \d{4} \d{4} \d{4}\b',  # Credit Card
            r'\b\d{16}\b'  # Credit Card (no spaces)
        ]

    def _term_based_scorer(self, text: str, component: str) -> float:
        """Score based on term presence with configurable weighting"""
        config = self.scoring_components[component]
        terms = config["terms"]
        factor = config["factor"]
        
        if not terms:
            logger.warning(f"No terms configured for {component} scoring")
            return 0.8  # Neutral fallback
        
        text_lower = text.lower()
        matches = 0
        for term in terms:
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(term) + r'\b'
            matches += len(re.findall(pattern, text_lower))
        
        total_words = max(1, len(text.split()))
        
        if component == "helpfulness":
            # Positive scoring for helpful terms
            return min(matches / (total_words * factor), 1.0)
        else:
            # Negative scoring for harmful terms
            penalty = min(matches / (total_words * factor), 1.0)
            return 1.0 - penalty

    def _pattern_based_scorer(self, text: str, component: str) -> float:
        """Score based on regex pattern matches"""
        config = self.scoring_components[component]
        patterns = config["patterns"]
        factor = config["factor"]
        
        if not patterns:
            logger.warning(f"No patterns configured for {component} scoring")
            return 0.8  # Neutral fallback
        
        total_matches = 0
        for pattern in patterns:
            try:
                total_matches += len(re.findall(pattern, text))
            except re.error as e:
                logger.error(f"Invalid regex pattern: {pattern} - {str(e)}")
        
        penalty = min(total_matches * factor, 1.0)
        return 1.0 - penalty

    def _handle_user_creation(self, text: str) -> Optional[float]:
        """Special handling for user creation context"""
        if "user" in text.lower() and "create" in text.lower():
            return 0.8  # Context-specific neutral score
        return None

    def _handle_financial_context(self, text: str) -> Optional[float]:
        """Special handling for financial context"""
        if any(term in text.lower() for term in ["account", "payment", "credit", "bank"]):
            # Increase scrutiny in financial contexts
            privacy_score = self.scoring_components["privacy"]["scorer"](text, "privacy")
            return min(privacy_score * 0.8, 1.0)  # Stricter penalty
        return None

    def _handle_medical_context(self, text: str) -> Optional[float]:
        """Special handling for medical context"""
        if any(term in text.lower() for term in ["medical", "health", "diagnosis", "treatment"]):
            # Increase scrutiny in medical contexts
            privacy_score = self.scoring_components["privacy"]["scorer"](text, "privacy")
            return min(privacy_score * 0.7, 1.0)  # Stricter penalty
        return None

    def calculate_score(self, text: str, component: str, context: Dict = None) -> float:
        """Calculate score for a specific component with context awareness"""
        # Contextual override
        if context:
            ctx_type = context.get("type")
            handler = self.context_handlers.get(ctx_type)
            if handler:
                ctx_score = handler(text)
                if ctx_score is not None:
                    return ctx_score
        
        # Component-specific scoring
        return self.scoring_components[component]["scorer"](text, component)

    # Public scoring methods maintain consistent interface ======================
    
    def _alignment_score(self, text: str) -> float:
        return self.calculate_score(text, "alignment")

    def _helpfulness_score(self, text: str) -> float:
        return self.calculate_score(text, "helpfulness")

    def _privacy_score(self, text: str) -> float:
        return self.calculate_score(text, "privacy")

    def _safety_score(self, text: str) -> float:
        return self.calculate_score(text, "safety")

    def _truthfulness_score(self, text: str) -> float:
        return self.calculate_score(text, "truthfulness")
