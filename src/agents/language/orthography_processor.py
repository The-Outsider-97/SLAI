"""
Detect and correct misspelled words
Normalize orthographic inconsistencies
Help downstream modules avoid errors due to typos.
"""

import yaml

from fuzzywuzzy import fuzz
from typing import Optional, Dict

from src.agents.language.utils.spell_checker import SpellChecker
from logs.logger import get_logger

logger = get_logger("Orthography Processor ")

CONFIG_PATH = "src/agents/language/configs/language_config.yaml"

def load_config(config_path=CONFIG_PATH):
    with open(config_path, "r", encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def get_merged_config(user_config=None):
    base_config = load_config()
    if user_config:
        base_config.update(user_config)
    return base_config

class OrthographyProcessor:
    def __init__(self, config):
        self.config = config.get("orthography_processor", {})
        self.spellchecker = SpellChecker(config.get("spell_checker", {}))
        self.normalization_map = self._load_normalization_map()
        self.locale = self._validate_locale()
        logger.info("Orthography Processor initialized...")

    def _load_normalization_map(self) -> Dict[str, str]:
        """Load spelling standardization rules"""
        path = self.config.get("normalization_map_path")
        if not path:
            logger.error("Normalization map path not specified in configuration.")
            return {}
        try:
            with open(path, 'r', encoding='utf-8') as f:
                normalization_map = yaml.safe_load(f)
            return normalization_map or {}
        except Exception as e:
            logger.error(f"Failed to load normalization map from {path}: {e}")
            return {}


    def _validate_locale(self) -> str:
        """Ensure working with supported locale"""
        system_locale = "en-US"  # Replace with actual locale detection
        allowed = self.config.get("allowed_locales", [])
        return system_locale if system_locale in allowed else "en-US"

    def _log_issue(self, original: str, corrected: str) -> None:
        if self.config.get("log_errors"):
            logger.warning(f"Orthographic issue: {original} -> {corrected}")

    def normalize(self, word: str) -> str:
        """Standardize spelling variants"""
        return self.normalization_map.get(word.lower(), word)

    def correct(self, word: str) -> Optional[str]:
        """Full correction pipeline"""
        # Step 1: Normalize case
        processed_word = word.lower() if not self.spellchecker.config.get("case_sensitive") else word
        
        original_word = word  # Preserve the original input for confidence checking
    
        # Step 2: Check spelling
        if not self.spellchecker.is_correct(processed_word):
            suggestions = self.spellchecker.suggest(processed_word)
            if suggestions:
                best_guess = suggestions[0]
                if processed_word.lower() != best_guess.lower():
                    if self._confidence_check(original_word, best_guess):
                        self._log_issue(original_word, best_guess)
                        processed_word = best_guess
    
        # Step 3: Apply normalization rules
        normalized = self.normalize(processed_word)
        
        # Step 4: Locale-specific adjustments
        if self.locale == "en-GB":
            normalized = normalized.replace("z", "s").replace("or", "our")
    
        return normalized

    def _confidence_check(self, original_word: str, corrected_word: str) -> bool:
        """Verify correction confidence between original and corrected words."""
        min_confidence = self.config.get("auto_correction_confidence", 0.7)
        confidence = fuzz.ratio(original_word.lower(), corrected_word.lower()) / 100.0
        return confidence >= min_confidence

    def batch_process(self, text: str) -> str:
        """Process full text with context awareness"""
        return " ".join([self.correct(word) for word in text.split()])

if __name__ == "__main__":
    print("\n=== Running Orthography Processor ===\n")
    config = get_merged_config()

    processor = OrthographyProcessor(config)

    print(processor.correct("colour"))
    text = "The organisation cancelled the programme due to unfavourable conditions"
    print(processor.batch_process(text))

    test_words = ["colour", "favour", "realise", "cancelled"]
    for word in test_words:
        print(f"{word} -> {processor.correct(word)}")

    print(processor)
    print("\n=== Successfully Ran Orthography Processor ===\n")

