
import os
import re
import yaml

from fuzzywuzzy import fuzz  # For Levenshtein distance
from metaphone import doublemetaphone  # For phonetic matching
from spellchecker import SpellChecker
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Set

from src.agents.language.utils.config_loader import load_global_config, get_config_section
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Spell Checker")
printer = PrettyPrinter

class SpellChecker:
    def __init__(self):
        self.config = load_global_config()
        self.checker_config = get_config_section('spell_checker')
        self.wordlist = self.checker_config.get('wordlist_path')
        self.typo_patterns = self.checker_config.get('typo_patterns_path')
        self.words = self.wordlist.words if self.wordlist and hasattr(self.wordlist, 'words') else []
    
        self.phonetic_map = self._build_phonetic_map() if self.checker_config.get('enable_phonetic') else None

        printer.status("INIT", "Spell Checker succesfully initialized", "info")

    def _build_phonetic_map(self):
        printer.status("INIT", "Phonetic map succesfully initialized", "info")

        phonetic_map = {}
        for word in self.words:
            key = self._get_phonetic_key(word)
            phonetic_map.setdefault(key, []).append(word)
        return phonetic_map

    def _get_phonetic_key(self, word):
        printer.status("INIT", "Phonetic key succesfully initialized", "info")
        if self.config.get("phonetic_algorithm") == "soundex":
            return self._soundex(word)
        return doublemetaphone(word)[0]
    
    def suggest_with_scores(self, word: str, max_suggestions: int = 5) -> List[Tuple[str, float]]:
        printer.status("INIT", "score suggestions succesfully initialized", "info")

        word_lc = word.lower()
        candidates = list(self.words)
    
        # Filter by edit distance rough cutoff using fuzzy
        candidates = sorted(
            candidates,
            key=lambda w: self._calculate_confidence(word_lc, w),
            reverse=True
        )
    
        return candidates[:max_suggestions]

    def _calculate_confidence(self, original: str, candidate: str) -> float:
        printer.status("INIT", "Confidence calculation succesfully initialized", "info")

        edit_dist = self._weighted_edit_distance(original, candidate)
        phonetic_sim = self._phonetic_similarity(original, candidate)
        freq_score = self.wordlist.word_probability(candidate) if self.wordlist else 0.0
        keyboard_sim = self._keyboard_similarity(original, candidate)
    
        edit_score = 1 / (1 + edit_dist)
        phonetic_score = phonetic_sim / 4
        keyboard_score = 1 / (1 + keyboard_sim)
    
        return min(max(
            edit_score * 0.3 +
            phonetic_score * 0.3 +
            keyboard_score * 0.2 +
            freq_score * 0.2, 0.0), 1.0)

    def _weighted_edit_distance(self, original: str, candidate: str) -> float:
        """Calculate weighted edit distance between two strings.
        
        Uses dynamic programming with configurable costs for substitutions
        (based on keyboard layout) and insertions/deletions.
        """
        printer.status("INIT", "Weighted edit calculation succesfully initialized", "info")

        # Get config parameters with defaults
        insertion_cost = self.checker_config.get("insertion_cost", 1.0)
        deletion_cost = self.checker_config.get("deletion_cost", 1.0)
        default_sub_cost = self.checker_config.get("default_substitution_cost", 2.0)
        case_sensitive = self.checker_config.get("case_sensitive", False)
    
        m, n = len(original), len(candidate)
        dp = [[0.0] * (n + 1) for _ in range(m + 1)]
    
        # Initialize base cases
        for i in range(1, m + 1):
            dp[i][0] = dp[i-1][0] + deletion_cost
        for j in range(1, n + 1):
            dp[0][j] = dp[0][j-1] + insertion_cost
    
        # Populate DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                char_orig = original[i-1] if case_sensitive else original[i-1].lower()
                char_cand = candidate[j-1] if case_sensitive else candidate[j-1].lower()
    
                if char_orig == char_cand:
                    sub_cost = 0.0
                else:
                    # Get substitution cost from keyboard layout
                    sub_cost = self._get_keyboard_sub_cost(char_orig, char_cand, default_sub_cost)
    
                dp[i][j] = min(
                    dp[i-1][j] + deletion_cost,    # Deletion
                    dp[i][j-1] + insertion_cost,    # Insertion
                    dp[i-1][j-1] + sub_cost         # Substitution
                )
    
        return dp[m][n]

    def _get_keyboard_sub_cost(self, c1: str, c2: str, default: float) -> float:
        """Access keyboard layout through injected Wordlist instance"""
        printer.status("INIT", "Keyboard access succesfully initialized", "info")

        if self.wordlist and hasattr(self.wordlist, "keyboard_layout"):
            cost = self.wordlist.keyboard_layout.get(c1, {}).get(c2, None)
            if cost is None:
                cost = self.wordlist.keyboard_layout.get(c2, {}).get(c1, default)
            return cost
        return default

    def _phonetic_similarity(self, w1: str, w2: str) -> float:
        printer.status("INIT", "Phonetic similarities succesfully initialized", "info")

        codes1 = {self._get_phonetic_key(w1)}
        codes2 = {self._get_phonetic_key(w2)}
        return len(codes1 & codes2) / max(len(codes1 | codes2), 1)
    
    def _keyboard_similarity(self, w1: str, w2: str) -> float:
        printer.status("INIT", "Keyboard similarities succesfully initialized", "info")

        if not self.wordlist or not hasattr(self.wordlist, "keyboard_layout"):
            return 2.0  # fallback cost
    
        layout = self.wordlist.keyboard_layout
        total = 0
        min_len = min(len(w1), len(w2))
        for c1, c2 in zip(w1[:min_len], w2[:min_len]):
            total += layout.get(c1, {}).get(c2, 2.0)
        return total / min_len if min_len > 0 else 0
    
    def generate_typo_variants(self, word: str) -> Set[str]:
        printer.status("INIT", "Typo handler succesfully initialized", "info")
        variants = set()

        def safe_add(w):  # Avoid adding the original word
            if w != word:
                variants.add(w)

        # Load patterns from file or use built-in
        patterns = self._load_typo_patterns()

        for pat, repl in patterns:
            try:
                safe_add(re.sub(pat, repl, word))
            except re.error:
                continue
    
        # Transpose adjacent characters
        for i in range(len(word) - 1):
            swapped = list(word)
            swapped[i], swapped[i + 1] = swapped[i + 1], swapped[i]
            safe_add(''.join(swapped))
    
        # Remove each character once
        for i in range(len(word)):
            safe_add(word[:i] + word[i + 1:])
    
        # Duplicate each character once
        for i in range(len(word)):
            safe_add(word[:i + 1] + word[i] + word[i + 1:])
    
        # Replace each character with adjacent keyboard key (if keyboard layout provided)
        if self.wordlist and hasattr(self.wordlist, "keyboard_layout"):
            layout = self.wordlist.keyboard_layout
            for i, char in enumerate(word):
                if char in layout:
                    for nearby_char in layout[char]:
                        safe_add(word[:i] + nearby_char + word[i + 1:])
    
        # Insert common letters at every position
        for i in range(len(word) + 1):
            for c in "aeiou":
                safe_add(word[:i] + c + word[i:])
    
        return variants
    
    def _load_typo_patterns(self) -> List[Tuple[str, str]]:
        """Load patterns from YAML file or return built-in patterns"""
        builtin_patterns = [
            (r'ie$', 'ei'), (r'ei$', 'ie'),  # receive, their
            (r'ou$', 'uo'), (r'uo$', 'ou'),  # guard vs gurad
            (r'([aeiou])\1', r'\1'),  # repeated vowels: aa -> a
            (r'([a-z])\1+', r'\1'),   # happpy -> happy
            (r'^[aeiou]', ''),         # about -> bout (edge case)
            (r'(.)\1(.)\2', r'\1\2'), # committee -> commite
            (r'ph', 'f'), (r'f(?!e)', 'ph'),  # phone <-> fone
            (r'(\w)(ie|ei)(\w)', r'\1\3\2'),  # transposes ei/ie in middle
            (r'c([ei])', r's\1'),    # receive -> seive (then validated)
            (r'(\w)or$', r'\1er'), (r'(\w)er$', r'\1or'),  # color <-> colour
            (r're$', 'er'), (r'er$', 're'),  # centre <-> center
            (r'able$', 'ible'), (r'ible$', 'able'),  # convertible
            (r'ance$', 'ence'), (r'ence$', 'ance'),  # persistence
            (r'ary$', 'ery'), (r'ery$', 'ary'),      # stationary vs stationery
            (r'gh', ''),          # though -> thou (but validate via wordlist)
            (r'([td])h$', r'\1'), # with -> wit (edge cases)
            (r'll$', 'l'), (r'l$', 'll'),  # full vs ful
            (r'([aeiou])([^aeiou])$', r'\1\2e'),  # lov -> love
            (r'([^e])e$', r'\1'),  # have -> hav (validate)
            (r'([^c])ie$', r'\1y'),  # tidy -> tidie (reverse)
            (r'([cs])h$', r'\1'), # tech -> tec
            (r'([aeiou])r([e$])', r'\1er'),  # centre -> center
        ]

        if not self.typo_patterns or not os.path.exists(self.typo_patterns):
            return builtin_patterns

        try:
            with open(self.typo_patterns, 'r') as f:
                pattern_list = yaml.safe_load(f)
                return [(p, r) for p, r in pattern_list]
        except Exception as e:
            logger.error(f"Error loading typo patterns: {e}")
            return builtin_patterns

    def _soundex(self, word):
        """Soundex phonetic algorithm implementation."""
        printer.status("INIT", "Soundex succesfully initialized", "info")

        if not word:
            return "0000"
        
        word = word.upper()
        first_char = word[0]
        soundex_code = [first_char]
        
        # Soundex mapping for consonants
        soundex_mapping = {
            'B': '1', 'F': '1', 'P': '1', 'V': '1',
            'C': '2', 'G': '2', 'J': '2', 'K': '2',
            'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
            'D': '3', 'T': '3', 'L': '4', 'M': '5',
            'N': '5', 'R': '6'
        }
        
        previous_code = None
        for char in word[1:]:
            # Skip vowels and H/W/Y
            if char in {'A', 'E', 'I', 'O', 'U', 'Y', 'H', 'W'}:
                continue
            
            code = soundex_mapping.get(char, '')
            if not code:
                continue
            
            # Skip consecutive duplicate codes
            if code != previous_code:
                soundex_code.append(code)
                previous_code = code
        
        # Trim or pad to 4 characters
        if len(soundex_code) == 1:
            soundex_code += ['0', '0', '0']
        else:
            soundex_code_str = ''.join(soundex_code)
            soundex_code = list(soundex_code_str[0] + ''.join(soundex_code_str[1:4]).ljust(4, '0'))
        
        return ''.join(soundex_code)[:4]

    def suggest(self, word, max_suggestions=None):
        printer.status("INIT", "Suggestion succesfully initialized", "info")

        candidates = []
        if not self.checker_config.get("case_sensitive"):
            word = word.lower()

        if "edit_distance" in self.checker_config.get("suggestion_strategies", []):
            candidates.extend(sorted(
                self.words, 
                key=lambda w: fuzz.ratio(w, word), 
                reverse=True
            )[:self.checker_config.get('max_suggestions')])

        if "phonetic" in self.checker_config.get("suggestion_strategies", []) and self.phonetic_map:
            phonetic_key = self._get_phonetic_key(word)
            candidates.extend(self.phonetic_map.get(phonetic_key, []))

        # Deduplicate and return top suggestions
        seen = set()
        final = []
        for candidate in candidates:
            if candidate not in seen:
                final.append(candidate)
                seen.add(candidate)
        max_sug = max_suggestions or self.checker_config.get("max_suggestions", 5)
        return final[:max_sug]

    def is_correct(self, word):
        return word in self.words if self.checker_config.get("case_sensitive") else word.lower() in self.words
    

if __name__ == "__main__":
    print("\n=== Running Spelling Checker ===\n")
    printer.status("Init", "Governor initialized", "success")

    spelling = SpellChecker()

    suggestions = spelling.suggest("Unfortonally") 
    print(spelling._soundex("Killer"))
    print(spelling._soundex("kill"))
    print(spelling._soundex("Killing"))
    print(spelling)
    print(f"Suggestions: {suggestions}")
    print("\n* * * * * Phase 2 * * * * *")

    word= "queens"
    word2= "trainer"
    words= "trainer, queens, boy, iron"
    max_suggestions=5
    original="queen"
    candidate="queens"
    badword= "queeen"

    score = spelling.suggest_with_scores(word=word, max_suggestions=max_suggestions)
    confidence=spelling._calculate_confidence(original=original, candidate=candidate)
    typo = spelling.generate_typo_variants(word=badword)
    soundex = spelling._soundex(word=words)

    printer.pretty("SCORE", score, "success")
    printer.pretty("CALC", confidence, "success")
    printer.pretty("TYPO", typo, "success")
    printer.pretty("SOUND", soundex, "success")
    print("\n=== Successfully Ran Spelling Checker ===\n")
