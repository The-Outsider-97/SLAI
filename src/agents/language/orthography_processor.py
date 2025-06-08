"""
Orthography Processor with:
- Context-aware spell checking
- Advanced normalization with locale-specific rules
- Phonetic and keyboard-aware corrections
- Compound word handling
- Contraction expansion
- Confidence-based auto-correction
"""

import os
import re
import yaml

from fuzzywuzzy import fuzz
from typing import Optional, Dict, List, Optional, Tuple

from src.agents.language.utils.config_loader import load_global_config, get_config_section
from src.agents.language.utils.spell_checker import SpellChecker
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Orthography Processor")
printer = PrettyPrinter

class OrthographyProcessor:
    def __init__(self):
        self.config = load_global_config()
        self.op_config = get_config_section('orthography_processor')
        self.enable_auto_correction = self.op_config.get('enable_auto_correction')
        self.normalization_map_path = self.op_config.get('normalization_map_path')
        self.log_errors = self.op_config.get('log_errors')
        self.auto_correction_confidence = self.op_config.get('auto_correction_confidence')
        self.allowed_locales = self.op_config.get('allowed_locales')
        self.enable_contraction_expansion = self.op_config.get('enable_contraction_expansion')
        self.enable_compound_handling = self.op_config.get('enable_compound_handling')
        self.max_context_window = self.op_config.get('max_context_window')

        self.spellchecker = SpellChecker()
        self.locale = self._validate_locale()
        self.normalization_map = self._load_normalization_map()
        self.contraction_map = self._load_contraction_map()
        self.locale_specific_rules = self._load_locale_rules()
        
        logger.info("Enhanced Orthography Processor initialized...")
        printer.status("INIT", "Orthography Processor ready", "success")

    def _validate_locale(self) -> str:
        system_locale = "en-US"  # Replace with actual locale detection
        allowed = self.config.get("allowed_locales", [])
        return system_locale if system_locale in allowed else "en-US"

    def _load_normalization_map(self) -> Dict[str, str]:
        """Load normalization rules from YAML file"""
        if not self.normalization_map_path or not os.path.exists(self.normalization_map_path):
            logger.warning("Normalization map path not configured or file missing")
            return {}

        try:
            with open(self.normalization_map_path, 'r') as f:
                data = yaml.safe_load(f)
                return data.get('normalization_rules', {})
        except Exception as e:
            logger.error(f"Error loading normalization map: {str(e)}")
            return {}

    def _load_contraction_map(self) -> Dict[str, str]:
        """Load contraction expansion rules"""
        return {
            "aren't": "are not", "can't": "cannot", "couldn't": "could not",
            "didn't": "did not", "doesn't": "does not", "don't": "do not",
            "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
            "he'd": "he would", "he'll": "he will", "he's": "he is",
            "I'd": "I would", "I'll": "I will", "I'm": "I am", "I've": "I have",
            "isn't": "is not", "it's": "it is", "let's": "let us",
            "mightn't": "might not", "mustn't": "must not", "shan't": "shall not",
            "she'd": "she would", "she'll": "she will", "she's": "she is",
            "shouldn't": "should not", "that's": "that is", "there's": "there is",
            "they'd": "they would", "they'll": "they will", "they're": "they are",
            "they've": "they have", "wasn't": "was not", "we'd": "we would",
            "we'll": "we will", "we're": "we are", "we've": "we have",
            "weren't": "were not", "what'll": "what will", "what're": "what are",
            "what's": "what is", "what've": "what have", "where's": "where is",
            "who'd": "who would", "who'll": "who will", "who're": "who are",
            "who's": "who is", "who've": "who have", "won't": "will not",
            "wouldn't": "would not", "you'd": "you would", "you'll": "you will",
            "you're": "you are", "you've": "you have"
        }

    def _load_locale_rules(self) -> Dict[str, Dict[str, str]]:
        """Load locale-specific transformation rules"""
        return {
            "en-GB": {
                "ize": "ise",
                "yze": "yse",
                "or": "our",
                "ll": "l",
                "re": "er",
                "er": "re",
                "se": "ce"
            },
            "en-US": {
                "our": "or",
                "ise": "ize",
                "yse": "yze",
                "re": "er"
            }
        }

    def _log_issue(self, original: str, corrected: str) -> None:
        if self.log_errors:
            logger.warning(f"Orthographic issue: {original} -> {corrected}")

    def _adjust_case(self, normalized: str, original: str) -> str:
        """Preserve original word casing in normalization"""
        if original.istitle():
            return normalized.title()
        elif original.isupper():
            return normalized.upper()
        return normalized

    def normalize(self, word: str) -> str:
        """Apply normalization rules with case preservation"""
        normalized = word
        lower_word = word.lower()
        
        # Apply direct normalization if exists
        if lower_word in self.normalization_map:
            normalized = self.normalization_map[lower_word]
        
        # Apply locale-specific suffix rules
        if self.locale in self.locale_specific_rules:
            for pattern, replacement in self.locale_specific_rules[self.locale].items():
                if word.endswith(pattern):
                    normalized = normalized[:-len(pattern)] + replacement
        
        return self._adjust_case(normalized, word)

    def expand_contractions(self, word: str) -> str:
        """Expand common English contractions"""
        return self.contraction_map.get(word.lower(), word)

    def _contextual_correction(self, word: str, context: List[str]) -> str:
        """Use surrounding words for better correction accuracy"""
        # Prioritize words that fit contextually
        suggestions = self.spellchecker.suggest_with_scores(word, 10)
        
        if not suggestions:
            return word
        
        # Score suggestions based on contextual fit
        contextual_scores = []
        for candidate, score in suggestions:
            context_sim = 0
            for ctx_word in context:
                ctx_sim = fuzz.ratio(candidate.lower(), ctx_word.lower()) / 100
                context_sim += ctx_sim
            contextual_scores.append((candidate, score * 0.7 + context_sim * 0.3))
        
        # Return best contextual match
        contextual_scores.sort(key=lambda x: x[1], reverse=True)
        return contextual_scores[0][0]

    def correct(self, word: str, context: Optional[List[str]] = None) -> str:
        """Comprehensive correction pipeline with context awareness"""
        # Expand contractions if enabled
        if self.enable_contraction_expansion:
            expanded = self.expand_contractions(word)
            if expanded != word:
                return expanded

        # Handle compound words (e.g., "keyboard" -> "key board")
        if self.enable_compound_handling and len(word) > 8:
            for i in range(3, len(word)-3):
                part1, part2 = word[:i], word[i:]
                if (self.spellchecker.is_correct(part1) and 
                    self.spellchecker.is_correct(part2)):
                    return f"{part1} {part2}"

        # Normalize case for processing
        processed_word = word
        if not self.spellchecker.checker_config.get("case_sensitive", False):
            processed_word = word.lower()

        # Check spelling
        if not self.spellchecker.is_correct(processed_word):
            # Use context if available
            if context:
                corrected = self._contextual_correction(processed_word, context)
            else:
                suggestions = self.spellchecker.suggest(processed_word, 1)
                corrected = suggestions[0] if suggestions else processed_word

            # Confidence check
            confidence = fuzz.ratio(processed_word, corrected) / 100
            if confidence >= self.auto_correction_confidence:
                self._log_issue(word, corrected)
                processed_word = corrected

        # Apply normalization rules
        normalized = self.normalize(processed_word)
        
        # Final locale adjustment
        if self.locale == "en-GB":
            normalized = normalized.replace("z", "s").replace("or", "our")
        
        return normalized

    def batch_process(self, text: str) -> str:
        """Process text with contextual awareness using a sliding window"""
        words = text.split()
        corrected = []
        
        for i, word in enumerate(words):
            # Create context window (previous and next words)
            start = max(0, i - self.max_context_window)
            end = min(len(words), i + self.max_context_window + 1)
            context = [w for j, w in enumerate(words[start:end]) if j != i]
            
            corrected.append(self.correct(word, context))
        
        return " ".join(corrected)

    def _confidence_check(self, original: str, corrected: str) -> bool:
        """Enhanced confidence check considering phonetic similarity"""
        min_confidence = self.auto_correction_confidence
        edit_confidence = fuzz.ratio(original.lower(), corrected.lower()) / 100
        
        # Consider phonetic similarity
        phonetic_sim = self.spellchecker._phonetic_similarity(original, corrected)
        combined_confidence = (edit_confidence * 0.7) + (phonetic_sim * 0.3)
        
        return combined_confidence >= min_confidence

if __name__ == "__main__":
    print("\n=== Running Enhanced Orthography Processor ===\n")
    printer.status("TEST", "Starting comprehensive tests", "info")

    processor = OrthographyProcessor()

    # Test cases
    tests = [
        ("colour", "color"),
        ("flavour", "flavor"),
        ("centre", "center"),
        ("realise", "realize"),
        ("cancelled", "canceled"),
        ("definately", "definitely"),
        ("recieve", "receive"),
        ("seperate", "separate"),
        ("adress", "address"),
        ("goverment", "government"),
        ("I'm", "I am"),
        ("don't", "do not"),
        ("can't", "cannot"),
        ("keyboard", "key board"),
        ("weather", "whether", ["forecast", "rain"]),
        ("their", "there", ["house", "location"])
    ]

    for test in tests:
        original = test[0]
        expected = test[1]
        context = test[2] if len(test) > 2 else None
        result = processor.correct(original, context)
        
        status = "PASS" if result == expected else "FAIL"
        printer.pretty(
            status, 
            f"{original} -> {result} (Expected: {expected})", 
            "success" if status == "PASS" else "error"
        )

    # Full sentence test
    text = "I cant beleive their having a programme in the theatre tommorow"
    expected = "I cannot believe they are having a program in the theater tomorrow"
    result = processor.batch_process(text)
    
    printer.pretty(
        "FULL TEXT", 
        f"Original: {text}\nCorrected: {result}\nExpected: {expected}", 
        "info"
    )
    
    print("\n=== Successfully Ran Enhanced Orthography Processor ===\n")
