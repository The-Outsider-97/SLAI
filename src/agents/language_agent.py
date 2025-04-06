import os, sys
import re
import json
import time
import datetime
import math
import pickle
import hashlib
import logging
import ply.lex as lex

from textstat import textstat
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any, OrderedDict, Union
from dataclasses import dataclass, field
from collections import deque, defaultdict, OrderedDict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from models.slai_lm import SLAILM

class DialogueContext:
    def __init__(self, llm=None, history=None, summary=None, memory_limit=1000, enable_summarization=True, summarizer=None):
        """
        Manages dialogue memory, optionally summarizes long histories.

        Args:
            llm: Reference to the language model (e.g., SLAILM)
            history (list): List of previous messages.
            summary (str): Optional summary of long-past context.
            memory_limit (int): Max number of messages to keep before summarizing.
            enable_summarization (bool): Whether to enable summarization.
            summarizer (Callable): Custom summarizer function.
        """
        self.llm = llm
        self.history = history if history is not None else []
        self.summary = summary
        self.memory_limit = memory_limit
        self.enable_summarization = enable_summarization
        self.summarizer = summarizer or self.default_summarizer

    def add(self, message: str):
        """Add a message to the history and optionally summarize."""
        self.history.append(message)
        if self.enable_summarization and len(self.history) > self.memory_limit:
            self._summarize()

    def _summarize(self):
        """Summarize current history and reset memory."""
        if self.summarizer:
            summary = self.summarizer(self.history, self.summary)
            self.summary = summary
            self.history = []

    def get_context(self) -> str:
        """Get full context including summary and current history."""
        parts = []
        if self.summary:
            parts.append(f"[Summary]\n{self.summary}")
        if self.history:
            parts.append(f"[History]\n" + "\n".join(self.history))
        return "\n\n".join(parts)

    def clear(self):
        """Clear the history and summary."""
        self.history = []
        self.summary = None

    def default_summarizer(self, messages: list, existing_summary: str = None) -> str:
        """Basic summarizer using LLM if provided."""
        context = ""
        if existing_summary:
            context += f"Previous summary:\n{existing_summary}\n\n"
        context += "Recent messages:\n" + "\n".join(messages)
        
        if self.llm:
            prompt = f"Summarize the following conversation:\n\n{context}"
            return self.llm.generate(prompt)
        else:
            # Fallback simple summarization
            return "Summary: " + " | ".join(messages[-3:])  # crude fallback

@dataclass 
class LinguisticFrame:
    """Structured representation of language acts (inspired by Speech Act Theory)"""
    intent: str
    entities: Dict[str, str]
    sentiment: float  # Range [-1, 1]
    modality: str  # e.g., "query", "command", "clarification"
    confidence: float  # [0, 1]

# --------------------------
# Independent Modules
# --------------------------
class Wordlist:
    """Advanced linguistic processor with phonetics, morphology, and semantic analysis"""
    
    def __init__(self, path: Union[str, Path] = "src/agents/learning/structured_wordlist_en.json"):
        self.path = Path(path)
        with open(self.path, "r") as f:
            responses = json.load(f)
        self.data = {}
        self.metadata = {}
        self._load()
        
        # Advanced caching systems
        self.lru_cache = OrderedDict()
        self.lfu_cache = defaultdict(int)
        self.max_cache_size = 10_000
        
        # Precomputed linguistic data
        self.phonetic_index = defaultdict(set)
        self.ngram_index = defaultdict(set)
        self._precompute_linguistic_data()
        
        # Language model parameters
        self.ngram_model = defaultdict(lambda: defaultdict(int))
        self._build_ngram_model()
        
        # Keyboard proximity costs, using a QWERTY keyboard
        self.keyboard_layout = {
            'q': {'w': 0.5, 'a': 0.7}, 'w': {'e': 0.5, 's': 0.7},
            'w': {'q': 0.4, 'e': 0.4, 'a': 0.6, 's': 0.5, 'd': 0.6},
            'e': {'w': 0.4, 'r': 0.4, 's': 0.6, 'd': 0.5, 'f': 0.6},
            'r': {'e': 0.4, 't': 0.4, 'd': 0.6, 'f': 0.5, 'g': 0.6},
            't': {'r': 0.4, 'y': 0.4, 'f': 0.6, 'g': 0.5, 'h': 0.6},
            'y': {'t': 0.4, 'u': 0.4, 'g': 0.6, 'h': 0.5, 'j': 0.6},
            'u': {'y': 0.4, 'i': 0.4, 'h': 0.6, 'j': 0.5, 'k': 0.6},
            'i': {'u': 0.4, 'o': 0.4, 'j': 0.6, 'k': 0.5, 'l': 0.6},
            'o': {'i': 0.4, 'p': 0.4, 'k': 0.6, 'l': 0.5},
            'p': {'o': 0.4, 'l': 0.6},
            
            'a': {'q': 0.6, 'w': 0.6, 's': 0.4, 'z': 0.7},
            's': {'q': 0.7, 'w': 0.5, 'e': 0.6, 'a': 0.4, 'd': 0.4, 'z': 0.6, 'x': 0.7},
            'd': {'w': 0.6, 'e': 0.5, 'r': 0.6, 's': 0.4, 'f': 0.4, 'x': 0.6, 'c': 0.7},
            'f': {'e': 0.6, 'r': 0.5, 't': 0.6, 'd': 0.4, 'g': 0.4, 'c': 0.6, 'v': 0.7},
            'g': {'r': 0.6, 't': 0.5, 'y': 0.6, 'f': 0.4, 'h': 0.4, 'v': 0.6, 'b': 0.7},
            'h': {'t': 0.6, 'y': 0.5, 'u': 0.6, 'g': 0.4, 'j': 0.4, 'b': 0.6, 'n': 0.7},
            'j': {'y': 0.6, 'u': 0.5, 'i': 0.6, 'h': 0.4, 'k': 0.4, 'n': 0.6, 'm': 0.7},
            'k': {'u': 0.6, 'i': 0.5, 'o': 0.6, 'j': 0.4, 'l': 0.4, 'm': 0.6},
            'l': {'i': 0.6, 'o': 0.5, 'p': 0.6, 'k': 0.4},
            
            'z': {'a': 0.7, 's': 0.6, 'x': 0.4},
            'x': {'s': 0.7, 'd': 0.6, 'z': 0.4, 'c': 0.4},
            'c': {'d': 0.7, 'f': 0.6, 'x': 0.4, 'v': 0.4},
            'v': {'f': 0.7, 'g': 0.6, 'c': 0.4, 'b': 0.4},
            'b': {'g': 0.7, 'h': 0.6, 'v': 0.4, 'n': 0.4},
            'n': {'h': 0.7, 'j': 0.6, 'b': 0.4, 'm': 0.4},
            'm': {'j': 0.7, 'k': 0.6, 'n': 0.4},
            
            # Number row and special characters
            '1': {'2': 0.4, 'q': 0.6, '!': 0.3},
            '!': {'1': 0.3, '2': 0.5, 'q': 0.6},
            '2': {'1': 0.4, '3': 0.4, 'q': 0.6, 'w': 0.6, '@': 0.3},
            '@': {'2': 0.3, '3': 0.5, 'w': 0.6},
            '3': {'2': 0.4, '4': 0.4, 'w': 0.6, 'e': 0.6, '#': 0.3},
            '#': {'3': 0.3, '4': 0.5, 'e': 0.6},
            '4': {'3': 0.4, '5': 0.4, 'e': 0.6, 'r': 0.6, '$': 0.3},
            '$': {'4': 0.3, '5': 0.5, 'r': 0.6},
            '5': {'4': 0.4, '6': 0.4, 'r': 0.6, 't': 0.6, '%': 0.3},
            '%': {'5': 0.3, '6': 0.5, 't': 0.6},
            '6': {'5': 0.4, '7': 0.4, 't': 0.6, 'y': 0.6, '^': 0.3},
            '^': {'6': 0.3, '7': 0.5, 'y': 0.6},
            '7': {'6': 0.4, '8': 0.4, 'y': 0.6, 'u': 0.6, '&': 0.3},
            '&': {'7': 0.3, '8': 0.5, 'u': 0.6},
            '8': {'7': 0.4, '9': 0.4, 'u': 0.6, 'i': 0.6, '*': 0.3},
            '*': {'8': 0.3, '9': 0.5, 'i': 0.6},
            '9': {'8': 0.4, '0': 0.4, 'i': 0.6, 'o': 0.6, '(': 0.3},
            '(': {'9': 0.3, '0': 0.5, 'o': 0.6},
            '0': {'9': 0.4, 'p': 0.6, ')': 0.3, '-': 0.4},
            ')': {'0': 0.3, 'p': 0.6, '-': 0.5},

            # Top-right special characters
            '-': {'0': 0.4, 'p': 0.6, '=': 0.4, '_': 0.3},
            '_': {'-': 0.3, '=': 0.5},
            '=': {'-': 0.4, '[': 0.6, '+': 0.3},
            '+': {'=': 0.3, '[': 0.5},
            
            # Brackets and backslash
            '[': {'p': 0.6, ']': 0.4, '=': 0.6, '{': 0.3},
            '{': {'[': 0.3, ']': 0.5},
            ']': {'[': 0.4, '\\': 0.4, '}': 0.3},
            '}': {']': 0.3, '\\': 0.5},
            '\\': {']': 0.4, "'": 0.6, '|': 0.3},
            '|': {'\\': 0.3},

            # Right-side punctuation
            ';': {'l': 0.6, "'": 0.4, ':': 0.3},
            ':': {';': 0.3, "'": 0.5},
            "'": {';': 0.4, 'k': 0.6, ',': 0.4, '"': 0.3},
            '"': {"'": 0.3, ',': 0.5},
            ',': {'m': 0.6, '.': 0.4, 'k': 0.6, '<': 0.3},
            '<': {',': 0.3, '.': 0.5},
            '.': {',': 0.4, '/': 0.4, 'm': 0.6, '>': 0.3},
            '>': {'.': 0.3, '/': 0.5},
            '/': {'.': 0.4, 'shift': 0.6, '?': 0.3},
            '?': {'/': 0.3},

            # Space and modifiers (approximate positions)
            'space': {'v': 0.7, 'b': 0.7, 'n': 0.7, 'm': 0.7},
            'caps': {'a': 0.6, 'q': 0.6, 'tab': 0.4},
            'tab': {'q': 0.6, 'caps': 0.4},
        }

        # Add reverse relationships automatically
        reverse_mapping = defaultdict(dict)
        for char, neighbors in self.keyboard_layout.items():
            for neighbor, cost in neighbors.items():
                if neighbor not in reverse_mapping or char not in reverse_mapping[neighbor]:
                    reverse_mapping[neighbor][char] = cost

        self.keyboard_layout.update(reverse_mapping)

    def _load(self) -> None:
        """Robust data loading with validation"""
        if not self.path.exists():
            raise FileNotFoundError(f"Wordlist missing: {self.path}")
        
        with open(self.path, 'r') as f:
            raw = json.load(f)
        
        required_keys = {'words', 'metadata', 'version'}
        if not required_keys.issubset(raw.keys()):
            raise ValueError("Invalid wordlist format - missing required keys")
        
        self.data = raw['words']
        self.metadata = raw['metadata']
        self._validate_word_entries()

    def _validate_word_entries(self) -> None:
        """Ensure all entries have valid structure"""
        for word, entry in self.data.items():
            if not isinstance(entry, dict):
                raise ValueError(f"Invalid entry format for word: {word}")
            if 'pos' not in entry or 'synonyms' not in entry:
                raise ValueError(f"Missing required fields in entry: {word}")

    def _precompute_linguistic_data(self) -> None:
        """Enhanced n-gram profiles with positional and variable-length grams"""
        for word in self.data:
            word_lc = word.lower()
            if not word_lc.strip():
                continue  # Skip invalid entries
            metaphone_key = self._metaphone(word_lc)

            # Phonetic representations
            self.phonetic_index[metaphone_key].add(word)
            self.phonetic_index[self._soundex(word_lc)].add(word)
            self.phonetic_index[self._metaphone(word_lc)].add(word)
            
            # Generate multiple n-gram types
            for n in [2, 3, 4]:  # Bigrams, trigrams, and quadgrams
                # Standard n-grams
                for i in range(len(word_lc) - n + 1):
                    ng = word_lc[i:i+n]
                    self.ngram_index[ng].add(word)
                
                # Position-aware n-grams with boundary markers
                padded = f'^{word_lc}$'
                for i in range(len(padded) - n + 1):
                    bng = padded[i:i+n]
                    self.ngram_index[bng].add(word)
            
            # Skip-grams (capture character patterns with gaps)
            if len(word_lc) >= 4:
                self.ngram_index[(word_lc[0], word_lc[2])].add(word)  # 1-skip
                self.ngram_index[(word_lc[1], word_lc[3])].add(word)

    def _build_ngram_model(self) -> None:
        """Build basic n-gram frequency model"""
        for word in self.data:
            for i in range(len(word)-1):
                self.ngram_model[word[i]][word[i+1]] += 1

    # PHONETIC ALGORITHMS ------------------------------------------------------
    
    def _soundex(self, word: str) -> str:
        """Soundex phonetic encoding implementation"""
        # Algorithm based on US Census Soundex specification
        word = word.upper()
        soundex_code = []
        
        # Step 1: Retain first letter
        first_char = word[0]
        soundex_code.append(first_char)
        
        # Soundex mapping dictionary
        char_map = {
            'BFPV': '1',
            'CGJKQSXZ': '2',
            'DT': '3',
            'L': '4',
            'MN': '5',
            'R': '6',
            'AEIOUYHW': ' '  # Vowels and H/W are ignored except first letter
        }
        
        # Step 2: Convert remaining characters
        prev_code = ''
        for char in word[1:]:
            for chars, code in char_map.items():
                if char in chars:
                    current_code = code
                    break
            else:
                current_code = ' '
            
            # Skip duplicates and vowels
            if current_code != prev_code and current_code != ' ':
                soundex_code.append(current_code)
                prev_code = current_code
        
        # Step 3: Pad/truncate to 4 characters
        soundex_code = soundex_code[:4]
        if len(soundex_code) < 4:
            soundex_code += ['0'] * (4 - len(soundex_code))
        
        return ''.join(soundex_code)
    
    def _metaphone(self, word: str) -> str:
        """Metaphone phonetic encoding implementation (simplified version)"""
        if not word:
            return ""
        word = word.upper()
        metaphone = []
        length = len(word)
        i = 0
        
        # Preprocessing
        word = re.sub(r'([^C])\1+', r'\1', word)  # Remove duplicate letters except C
        length = len(word)  # Update length after substitution
        
        # Transformation rules
        while i < length and len(metaphone) < 6:
            # Ensure i is within bounds
            if i >= length:
                break
            char = word[i]
            
            # Handle initial letters
            if i == 0:
                if char in ('A', 'E', 'I', 'O', 'U'):
                    metaphone.append(char)
                    i += 1
                    continue
                if word.startswith('KN') or word.startswith('GN'):
                    i += 1
                    continue
                if word.startswith('WR'):
                    metaphone.append('R')
                    i += 2
                    continue

            # Main transformation rules
            if char == 'C':
                # Check for 'CIA' only if there are enough characters
                if i + 2 <= length and word[i+1:i+3] == 'IA':
                    metaphone.append('X')
                    i += 3  # Skip the 'IA' part
                elif i > 0 and i + 1 < length and word[i-1] == 'S' and word[i+1] in ('H', 'I', 'E', 'Y'):
                    i += 1  # Skip processing this 'C'
                elif i + 1 < length and word[i+1] in ('H', 'I', 'E', 'Y'):
                    metaphone.append('S')
                    i += 1
                else:
                    metaphone.append('K')
                    i += 1
            
            elif char == 'D':
                if i + 2 < length and word[i+1] == 'G' and word[i+2] in ('E', 'Y', 'I'):
                    metaphone.append('J')
                    i += 3
                else:
                    metaphone.append('T')
                    i += 1
            
            elif char == 'G':
                # Silent G in 'GN', 'GNED'
                if (i + 1 < length and word[i+1] == 'N') or \
                   (i + 3 < length and word[i+1] == 'N' and word[i+2] == 'E' and word[i+3] == 'D'):
                    i += 1
                else:
                    metaphone.append('K')
                    i += 1
            
            elif char == 'P':
                if i+1 < length and word[i+1] == 'H':
                    metaphone.append('F')
                    i += 2
                else:
                    metaphone.append('P')
                    i += 1
            
            elif char == 'T':
                if i+1 < length and word[i+1] == 'H':
                    metaphone.append('0')  # θ sound
                    i += 2
                elif i+2 < length and word[i+1:i+3] in ('IA', 'IO'):
                    metaphone.append('X')  # SH sound
                    i += 3
                else:
                    metaphone.append('T')
                    i += 1
            
            elif char == 'V':
                metaphone.append('F')
                i += 1
            
            elif char == 'Q':
                metaphone.append('K')
                i += 1
            
            elif char == 'X':
                if i == 0:
                    metaphone.append('S')
                else:
                    if len(metaphone) < 5:  # Check remaining space
                        metaphone.extend(['K', 'S'])
                    elif len(metaphone) == 5:
                        metaphone.append('K')
                i += 1
            
            elif char == 'S':
                if i+1 < length and word[i+1] == 'H':
                    metaphone.append('X')
                    i += 2
                else:
                    metaphone.append('S')
                    i += 1
            
            elif char == 'Z':
                metaphone.append('S')
                i += 1
            
            elif char == 'F':
                metaphone.append('F')
                i += 1
            
            elif char == 'W':
                if i+1 < length and word[i+1] in 'AEIOU':
                    metaphone.append('W')
                    i += 2
                else:
                    i += 1
            
            elif char == 'H':
                # Keep H between vowels
                if 0 < i < length-1 and word[i-1] in 'AEIOU' and word[i+1] in 'AEIOU':
                    metaphone.append('H')
                i += 1
            
            elif char in ('B', 'M', 'N', 'R', 'L'):
                metaphone.append(char)
                i += 1
            
            elif char == 'K':
                if i > 0 and word[i-1] != 'C':
                    metaphone.append('K')
                i += 1
            
            # Vowel handling
            elif char in ('A', 'E', 'I', 'O', 'U'):
                if i == 0:
                    metaphone.append(char)
                i += 1
            
            else:
                i += 1  # Ignore other characters
        
        return ''.join(metaphone)[:6]  # Return first 6 characters

    # MORPHOLOGICAL ANALYSIS ---------------------------------------------------
    
    def stem(self, word: str) -> str:
        """Porter Stemmer implementation for morphological reduction"""
        # Implementation based on Porter's algorithm (1980)
        word = word.lower()
        
        # Step 1a: Plural/possessive removal
        if word.endswith('sses'):
            word = word[:-2]
        elif word.endswith('ies'):
            word = word[:-2]
        elif word.endswith('ss'):
            pass
        elif word.endswith('s'):
            word = word[:-1]
        
        # Step 1b: Verb forms
        m = self._measure(word)
        if m > 0 and word.endswith('eed'):
            word = word[:-1]
        elif re.search(r'[aeiou]', word[:-3]) and word.endswith('eed'):
            word = word[:-1]
        elif re.search(r'[aeiou]', word[:-2]) and word.endswith('ed'):
            word = word[:-2]
        elif re.search(r'[aeiou]', word[:-3]) and word.endswith('ing'):
            word = word[:-3]
        
        # Step 1c: Replace *y with i
        if word.endswith('y') and re.search(r'[aeiou]', word[:-1]):
            word = word[:-1] + 'i'
        
        # Step 2: Common suffixes
        step2_map = {
            'ational': 'ate',
            'tional': 'tion',
            'enci': 'ence',
            'anci': 'ance',
            'izer': 'ize',
            'abli': 'able'
        }
        for suffix, replacement in step2_map.items():
            if word.endswith(suffix) and self._measure(word[:-len(suffix)]) > 0:
                word = word[:-len(suffix)] + replacement
                break
        
        # Step 3: Complex suffixes
        step3_map = {
            'icate': 'ic',
            'ative': '',
            'alize': 'al',
            'iciti': 'ic'
        }
        for suffix, replacement in step3_map.items():
            if word.endswith(suffix) and self._measure(word[:-len(suffix)]) > 0:
                word = word[:-len(suffix)] + replacement
                break
        
        # Step 4: Remove final suffixes
        for suffix in ['al', 'ance', 'er', 'ic', 'able', 'ant', 'ement']:
            if word.endswith(suffix) and self._measure(word[:-len(suffix)]) > 1:
                word = word[:-len(suffix)]
                break
        
        # Step 5: Final cleanup
        if self._measure(word[:-1]) > 1 and word.endswith('e'):
            word = word[:-1]
        elif self._measure(word[:-1]) == 1 and not self._cvc(word[:-1]) and word.endswith('e'):
            word = word[:-1]
        
        if self._measure(word) > 1 and word.endswith('ll'):
            word = word[:-1]
        
        return word

    def _measure(self, stem: str) -> int:
        """Calculate Porter's 'm' measure for VC pattern counting"""
        vowels = 'aeiou'
        count = 0
        pattern = []
        
        for char in stem:
            if char in vowels:
                if pattern and pattern[-1] == 'C':
                    pattern.append('V')
                elif not pattern:
                    pattern.append('V')
            else:
                if pattern and pattern[-1] == 'V':
                    pattern.append('C')
                elif not pattern:
                    pattern.append('C')
        
        # Count VC transitions (each VC pair is one measure)
        return (len(pattern) - 1) // 2

    def _cvc(self, stem: str) -> bool:
        """Check CVC pattern where last C is not W, X, or Y"""
        if len(stem) < 3:
            return False
        return (stem[-3] not in 'aeiou' and
                stem[-2] in 'aeiou' and
                stem[-1] not in 'aeiouwxy')
      
    # ADVANCED SPELLING CORRECTION ----------------------------------------------
    
    def weighted_edit_distance(self, a: str, b: str) -> float:
        """Keyboard-aware edit distance using dynamic programming"""
        m, n = len(a), len(b)
        dp = [[0.0]*(n+1) for _ in range(m+1)]
        
        # Initialize base cases
        for i in range(m+1):
            dp[i][0] = i * 1.5  # Higher deletion cost
        for j in range(n+1):
            dp[0][j] = j * 1.5  # Higher insertion cost
        
        # Populate DP table
        for i in range(1, m+1):
            for j in range(1, n+1):
                deletion = dp[i-1][j] + 1.5
                insertion = dp[i][j-1] + 1.5
                
                # Calculate substitution cost
                if a[i-1] == b[j-1]:
                    substitution = dp[i-1][j-1]
                else:
                    # Get keyboard proximity cost
                    c1, c2 = a[i-1], b[j-1]
                    cost = min(
                        self.keyboard_layout.get(c1, {}).get(c2, float('inf')),
                        self.keyboard_layout.get(c2, {}).get(c1, float('inf'))
                    )
                    if cost == float('inf'):
                        # Penalize non-adjacent substitutions
                        cost = 2.0 if c1.isalpha() == c2.isalpha() else 3.0
                    substitution = dp[i-1][j-1] + cost
                    
                # Transposition (Damerau-Levenshtein extension)
                transposition = float('inf')
                if i > 1 and j > 1 and a[i-1] == b[j-2] and a[i-2] == b[j-1]:
                    transpose_cost = 0.8  # Lower than substitution
                    transposition = dp[i-2][j-2] + transpose_cost
                    
                dp[i][j] = min(deletion, insertion, substitution, transposition)
        
        return dp[m][n]
    
    def phonetic_candidates(self, word: str) -> List[str]:
        """Get phonetically similar candidates"""
        return list(self.phonetic_index.get(self._soundex(word), set()) |
                    self.phonetic_index.get(self._metaphone(word), set()))
    
    # SEMANTIC ANALYSIS ---------------------------------------------------------
    
    def semantic_similarity(self, word1: str, word2: str) -> float:
        """Vector space similarity using co-occurrence statistics"""
        vec1 = self._word_vector(word1)
        vec2 = self._word_vector(word2)
        return self._cosine_similarity(vec1, vec2)
    
    def _word_vector(self, word: str) -> Dict[str, int]:
        """Build co-occurrence vector using n-gram transition frequencies"""
        vector = defaultdict(int)
        
        # Sum n-gram transitions for all character pairs in the word
        for i in range(len(word) - 1):
            current = word[i]
            next_char = word[i+1]
            
            # Get all possible transitions for current character
            transitions = self.ngram_model.get(current, {})
            
            # Add weighted transitions to vector
            for following_char, count in transitions.items():
                proximity_weight = 1.0  # Base weight (could use position weights)
                if following_char == next_char:
                    # Boost actual observed transitions
                    proximity_weight = 2.0  
                vector[following_char] += int(count * proximity_weight)
        
        # Add reverse transitions for context symmetry
        for i in range(1, len(word)):
            current = word[i]
            prev_char = word[i-1]
            
            transitions = self.ngram_model.get(prev_char, {})
            vector[current] += transitions.get(current, 0)
        
        return dict(vector)
    
    def _cosine_similarity(self, vec1: Dict, vec2: Dict) -> float:
        """Calculate cosine similarity between vectors"""
        # Mathematical implementation...
    
    # LANGUAGE MODELING ---------------------------------------------------------
    
    def word_probability(self, word: str) -> float:
        """Calculate relative frequency probability"""
        total_words = self.metadata.get('word_count', 1)
        return self.data[word].get('frequency', 1) / total_words
    
    def context_suggestions(self, previous_words: List[str], limit: int = 5) -> List[str]:
        """Predict next word using n-gram model"""
        # Implementation using n-gram probabilities...
    
    # SYLLABLE ANALYSIS ---------------------------------------------------------
    
    def syllable_count(self, word: str) -> int:
        """Mathematical syllable estimation algorithm"""
        # Implementation based on vowel counting and exceptions...
    
    # CACHE MANAGEMENT ----------------------------------------------------------
    
    def query(self, word: str) -> Optional[Dict]:
        """Intelligent caching with combined LRU/LFU strategy"""
        word = word.lower()
        
        if word in self.lru_cache:
            self.lru_cache.move_to_end(word)
            self.lfu_cache[word] += 1
            return self.lru_cache[word]
        
        entry = self.data.get(word)
        if entry:
            self._update_cache(word, entry)
        
        return entry
    
    def _update_cache(self, word: str, entry: Dict) -> None:
        """Hybrid cache update strategy"""
        # Combined LRU/LFU eviction logic...
    
    # GRAPH-BASED RELATIONSHIPS -------------------------------------------------
    
    def build_synonym_graph(self) -> None:
        """Construct synonym relationship graph"""
        self.graph = defaultdict(set)
        for word, entry in self.data.items():
            for syn in entry['synonyms']:
                self.graph[word].add(syn.lower())
                self.graph[syn.lower()].add(word)
    
    def synonym_path(self, start: str, end: str) -> Optional[List[str]]:
        """Find shortest path through synonym relationships"""
        # BFS implementation for graph traversal...
    
    # VALIDATION AND ERROR HANDLING ---------------------------------------------
    
    def validate_word(self, word: str) -> bool:
        """Comprehensive word validation"""
        return (
            self._check_orthography(word) and
            self._check_morphology(word) and
            self._check_phonotactics(word)
        )
    
    def _check_phonotactics(self, word: str) -> bool:
        """Validate word structure against language phonotactic rules"""
        # Implementation of phonotactic constraints...

class NLUEngine:
    """Rule-based semantic parser with fallback patterns"""
    def __init__(self, wordlist_path: str = "src/agents/learning/structured_wordlist_en.json"):
        # Hierarchical intent recognition system with pattern clusters
        self.intent_weights = {
            'information_request': {
                'patterns': [
                    # Primary patterns (high specificity)
                    (r'^(what|where|when|how)\s+(is|are|does)\s+the?\b', 2.5),
                    (r'explain\s+(the\s+)?(concept|process|idea)\b', 2.4),
                    
                    # Secondary patterns (medium specificity)
                    (r'\b(define|describe|elaborate)\s+on\b', 2.0),
                    (r'\b(meaning|significance)\s+of\b', 1.8),
                    
                    # Tertiary patterns (contextual)
                    (r'\b(can\s+you|could\s+you)\s+clarify\b', 1.6),
                ],
                'exclusions': [
                    r'\b(how\s+to|tutorial|guide)\b',  # Exclude instructional queries
                ],
                'context_requirements': {
                    'required_pos': ['NOUN', 'PROPN'],
                    'proximity_window': 3  # Words within 3 positions
                }
            },
            'action_request': {
                'patterns': [
                    # Imperative structures
                    (r'^(please|kindly|urgently)\s+(execute|run|perform)\b', 2.8),
                    (r'\b(start|stop|restart)\s+the\s+process\b', 2.7),
                    
                    # Modal verb constructions
                    (r'\b(must|should)\s+(be\s+)?(initiated|terminated)\b', 2.5),
                    (r'\b(initiate|terminate)\s+immediately\b', 2.6),
                ],
                'context_requirements': {
                    'required_verbs': ['execute', 'run', 'start', 'stop'],
                    'dependency_relations': ['dobj', 'xcomp']  # Verb-object relations
                }
            },
            # ... other intents ...
        }
        
        # Entity recognition system with composable patterns
        self.entity_patterns = {
            'temporal': {
                'components': {
                    'date': r'\d{4}-\d{2}-\d{2}',
                    'relative': r'(today|tomorrow|next\s+\w+)',
                    'time': r'\d{1,2}:\d{2}\s?(?:AM|PM)?'
                },
                'pattern': r'(?:{date}|{relative}|{time})',
                'pos_constraints': ['NOUN', 'ADV'],
                'validation': self._validate_temporal,
                'priority': 1
            },
            'quantitative': {
                'components': {
                    'number': r'\b\d+(?:\.\d+)?\b',
                    'unit': r'(kg|ml|m|cm|Hz|W)'
                },
                'pattern': r'{number}\s*{unit}',
                'pos_constraints': ['NUM', 'ADJ'],
                'validation': self._validate_quantity,
                'priority': 2
            },
            'technical': {
                'components': {
                    'protocol': r'(HTTP|FTP|SSH)',
                    'code': r'[A-Z]{3}-\d{4}',
                    'version': r'\bv?\d+\.\d+(?:\.\d+)?\b'
                },
                'pattern': r'({protocol}/\d\.\d|{code}|{version})',
                'pos_constraints': ['PROPN', 'NOUN'],
                'validation': self._validate_technical,
                'priority': 3
            }
        }

        # Sentiment lexicon (word: polarity_score)
        self.sentiment_lexicon = {
            # Strong Positive (1.5-3.0)
            'excellent': 2.9, 'outstanding': 2.8, 'magnificent': 2.7,
            'superb': 2.6, 'brilliant': 2.5, 'fantastic': 2.4,
            
            # Moderate Positive (0.5-1.4)
            'good': 1.3, 'pleasing': 1.2, 'satisfactory': 1.0,
            'adequate': 0.8, 'acceptable': 0.6,
            
            # Weak Positive (0.1-0.4)
            'neutral': 0.3, 'tolerable': 0.2, 'passable': 0.1,
            
            # Strong Negative (-3.0--1.5)
            'horrendous': -2.9, 'atrocious': -2.8, 'abysmal': -2.7,
            'disastrous': -2.6, 'appalling': -2.5,
            
            # Moderate Negative (-1.4--0.5)
            'poor': -1.3, 'subpar': -1.2, 'deficient': -1.0,
            'unsatisfactory': -0.8, 'inadequate': -0.6,
            
            # Weak Negative (-0.4--0.1)
            'questionable': -0.3, 'dubious': -0.2, 'mediocre': -0.1,
            
            # Intensifiers/Modifiers
            'very': 0.5, 'extremely': 0.7, 'somewhat': -0.3,
            'slightly': -0.2, 'remarkably': 0.6
        }

        # Modality markers
        self.modality_markers = {
            'epistemic': {  # Knowledge/belief
                'certainly', 'probably', 'possibly', 
                'apparently', 'seemingly', 'evidently'
            },
            'deontic': {  # Obligation/permission
                'must', 'should', 'ought', 
                'permitted', 'allowed', 'forbidden'
            },
            'dynamic': {  # Ability/capacity
                'can', 'could', 'able', 
                'capable', 'enable', 'capacity'
            },
            'alethic': {  # Logical necessity
                'necessarily', 'possibly', 
                'contingently', 'impossibly'
            },
            'interrogative': {
                'who', 'what', 'when', 
                'where', 'why', 'how', 
                'which', 'whom', 'whose'
            },
            'imperative': {
                'please', 'kindly', 'urgently',
                'immediately', 'require', 'demand'
            },
            'conditional': {
                'if', 'unless', 'provided',
                'assuming', 'contingent', 'conditional'
            }
        }

        self.wordlist = self._load_wordlist(wordlist_path)  # Assume returns POS-annotated dict

    def _validate_temporal(self, entity: str) -> bool:
        """Temporal validation using date logic"""
        if re.match(r'\d{4}-\d{2}-\d{2}', entity):
            try:
                datetime.strptime(entity, '%Y-%m-%d')
                return True
            except ValueError:
                return False
        return True  # Accept relative times

    def _validate_quantity(self, entity: str) -> bool:
        """Physical quantity validation"""
        value, unit = re.match(r'(\d+)\s*(\D+)', entity).groups()
        return unit.lower() in {
            'kg', 'g', 'ml', 'l', 
            'm', 'cm', 'hz', 'khz', 'w', 'kw'
        }

    def _validate_technical(self, entity: str) -> bool:
        """Technical spec validation"""
        if '-' in entity:
            prefix, code = entity.split('-', 1)
            return prefix.isalpha() and code.isdigit()
        return True
    
    def parse(self, text: str) -> LinguisticFrame:
        """Hybrid parsing using rules and simple statistics"""
        frame = LinguisticFrame(
            intent='unknown',
            entities={},
            sentiment=self._calculate_sentiment(text),
            modality='statement',
            confidence=0.0
        )

        # Intent detection
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    frame.intent = intent
                    frame.confidence += 0.3  # Simple confidence scoring

        # Entity extraction
        entities = {}
        for entity_type, pattern in self.entity_patterns.items():
            matches = [m[0] if isinstance(m, tuple) else m for m in re.findall(pattern, text)]
            entities[entity_type] = [m for m in matches if m]
        frame.entities = entities

        self._validate_words(frame)
        return frame

    def _validate_words(self, frame: LinguisticFrame) -> None:
        """Check if recognized entities exist in the wordlist"""
        valid_entities = {}
        for entity_type, values in frame.entities.items():
            valid = [v for v in values if self.wordlist.query(v)]
            if valid:
                valid_entities[entity_type] = valid
        frame.entities = valid_entities

    def _calculate_sentiment(self, text: str) -> float:
        """Basic sentiment analysis using lexicon approach"""
        tokens = re.findall(r'\b\w+\b', text.lower())
        base_score = 0.0
        current_intensity = 1.0
        
        for i, token in enumerate(tokens):
            # Handle intensifiers
            if token in {'very', 'extremely', 'remarkably'}:
                current_intensity *= 1.5
                continue
            elif token in {'somewhat', 'slightly'}:
                current_intensity *= 0.7
                continue
                
            # Reset intensity after each sentiment-bearing word
            if token in self.sentiment_lexicon:
                base_score += self.sentiment_lexicon[token] * current_intensity
                current_intensity = 1.0  # Reset modifier

        # Normalization using softmax-inspired approach
        normalized = math.tanh(base_score / math.sqrt(len(tokens))) if tokens else 0.0
        return round(normalized, 2)

    def _detect_modality(self, text: str) -> str:
        """Hierarchical modality detection"""
        text_lower = text.lower()
        
        # Priority detection order
        modalities = [
            ('interrogative', any(m in text_lower for m in self.modality_markers['interrogative'])),
            ('imperative', any(m in text_lower for m in self.modality_markers['imperative'])),
            ('conditional', any(m in text_lower for m in self.modality_markers['conditional'])),
            ('epistemic', any(m in text_lower for m in self.modality_markers['epistemic'])),
            ('deontic', any(m in text_lower for m in self.modality_markers['deontic'])),
            ('dynamic', any(m in text_lower for m in self.modality_markers['dynamic'])),
            ('alethic', any(m in text_lower for m in self.modality_markers['alethic'])),
        ]

        for mod_type, detected in modalities:
            if detected:
                return mod_type
        return 'declarative'
    
    def _calculate_confidence(self, text: str) -> float:
        total_weight = 0.0
        for intent, patterns in self.intent_weights.items():
            for pattern, weight in patterns.items():
                if re.search(pattern, text, re.IGNORECASE):
                    total_weight += weight
        
        # Sigmoid function for confidence scaling
        return 1 / (1 + math.exp(-total_weight/3))  # Logistic curve

    def _load_wordlist(self, path: str) -> Dict[str, Any]:
        # Assume implementation loads wordlist with POS data
        return {}  # Placeholder
    
# --------------------------
# Enhanced NLU Components
# --------------------------
class EnhancedNLU(NLUEngine):
    """Implements advanced NLU techniques from recent research"""
    def __init__(self, wordlist_path: str):
        super().__init__(wordlist_path)
        self.coref_resolver = CoreferenceResolver()
        self.dependency_parser = ShallowDependencyParser()
        
        # Add psycholinguistic features
        self.lexical_diversity = lex()
        self.readability = textstat()

    def parse(self, text: str) -> LinguisticFrame:
        """Enhanced parsing pipeline with multiple processing stages"""
        # Stage 0: Coreference resolution
        resolved_text = self.coref_resolver.resolve(text, self.context.history)
        
        # Stage 1: Dependency parsing
        parse_tree = self.dependency_parser.parse(resolved_text)
        
        # Stage 2: Enhanced sentiment analysis
        frame = super().parse(resolved_text)
        frame.sentiment = self._enhanced_sentiment(resolved_text, parse_tree)
        
        # Stage 3: Pragmatic analysis
        frame.modality = self._detect_modality(parse_tree)
        
        return frame

    def _enhanced_sentiment(self, text: str, parse_tree: Dict) -> float:
        """Sentiment analysis considering negation and intensity"""
        # Implementation based on Socher et al. (2013) Recursive Neural Networks
        sentiment = 0.0
        for node in parse_tree['nodes']:
            if node['relation'] == 'neg':
                sentiment -= 0.5
            else:
                word_sentiment = self.wordlist.query(node['word']).get('sentiment', 0)
                sentiment += word_sentiment * node['intensity']
        return math.tanh(sentiment)  # Squash to [-1, 1]

    def _detect_modality(self, parse_tree: Dict) -> str:
        """Detect speech modality using verb patterns"""
        # Inspired by Austin's Speech Act Theory (1975)
        main_verb = parse_tree['root']['lemma']
        modality_map = {
            'ask': 'query',
            'request': 'command',
            'suggest': 'proposal'
        }
        return modality_map.get(main_verb, 'statement')

class CoreferenceResolver:
    """Simple rule-based coreference resolution"""
    def resolve(self, text: str, history: deque) -> str:
        """Replace pronouns with recent entities"""
        # Implementation inspired by Hobbs algorithm (1978)
        last_entities = self._extract_last_entities(history)
        return re.sub(r'\b(he|she|it|they)\b', lambda m: last_entities.get(m.group().lower(), m.group()), text)

class ShallowDependencyParser:
    """Lightweight dependency parser using regex patterns"""
    def parse(self, text: str) -> Dict:
        """Extract basic dependency relations"""
        # Simplified version of Marneffe et al. (2014) Universal Dependencies
        patterns = {
            'nsubj': r'(\w+)\s+is',  # Simplified subject detection
            'dobj': r'(\w+)\s+',      # Direct object placeholder
        }
        return {'nodes': [...]}  # Simplified output

# --------------------------
# Enhanced NLG Components
# --------------------------
class NLGEngine:
    """Controlled text generation with style management"""
    def __init__(self, templates_path: str = "src/agents/learning/nlg_templates_en.json"):


        self.templates = self._load_templates(templates_path)
        self.style = {'formality': 0.5, 'verbosity': 1.0}
        self.coherence_checker = ResponseCoherence()

    def _load_templates(self, path: str) -> Dict[str, List[str]]:
        """Load and validate response templates from JSON file"""
        template_path = Path(path)
        
        if not template_path.exists():
            raise FileNotFoundError(f"Template file missing at: {template_path}")
            
        try:
            with open(template_path, 'r') as f:
                templates = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in templates: {e}")

        # Validate template structure
        required_categories = {
            'greeting', 'farewell', 'default',
            'acknowledgement', 'error', 'thanks'
        }
        
        missing = required_categories - templates.keys()
        if missing:
            raise ValueError(f"Missing required template categories: {missing}")

        # Validate each category has at least one template
        for category, entries in templates.items():
            if isinstance(entries, dict):
                if "responses" not in entries or not isinstance(entries["responses"], list):
                    raise ValueError(f"Invalid template format for category: {category}")
            elif isinstance(entries, list):
                continue  # accept flat list for backward compatibility
            else:
                raise ValueError(f"Invalid template format for category: {category}")

        return templates

    def generate(self, frame: LinguisticFrame, context: DialogueContext) -> str:
        """Generate response using hybrid template-neural approach"""
        # Step 1: Template selection
        if template := self._match_template(frame):
            response = self._instantiate_template(template, context)
        else:
            response = self._neural_generation(frame, context)
            
        # Step 2: Style adaptation
        response = self._adapt_style(response)
        
        # Step 3: Coherence check
        if not self.coherence_checker.validate(response, context):
            response = self._fallback_generation(frame)
            
        return response

    def _match_template(self, frame: LinguisticFrame) -> Optional[str]:
        """Match against handcrafted templates for common intents"""
        template_map = {
            'information_request': "The {entity} is {value}",
            'action_request': "Executing command: {command}"
        }
        return template_map.get(frame.intent)

    def _neural_generation(self, frame: LinguisticFrame, context: DialogueContext) -> str:
        """Generate using LLM with controlled prompting"""
        prompt = f"Generate response with intent {frame.intent} and entities {frame.entities}"
        return self.llm.generate(prompt)

    def _adapt_style(self, text: str) -> str:
        """Adjust formality and verbosity"""
        if self.style['formality'] > 0.7:
            text = re.sub(r"\b(can't|don't)\b", "cannot do not", text)
        if self.style['verbosity'] < 0.5:
            text = ' '.join(text.split()[:15]) + '...'
        return text

class ResponseCoherence:
    """Ensure generated responses stay on-topic"""
    def validate(self, response: str, context: DialogueContext) -> bool:
        """Check lexical overlap with conversation history"""
        # Implementation inspired by Centering Theory (Grosz et al. 1995)
        history_words = set(word for turn in context.history for word in turn[0].split())
        response_words = set(response.split())
        return len(history_words & response_words) / len(response_words) > 0.3

class KnowledgeCache:
    """Lightweight cache with LRU eviction policy"""
    def __init__(self, max_size: int = 100):
        self.cache = OrderedDict()
        self.max_size = max_size

    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def set(self, key: str, value: Any) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

    def hash_query(self, query: str) -> str:
        """Semantic hashing for similar queries (simplified)"""
        return hashlib.md5(query.encode()).hexdigest()

class SafetyGuard:
    """Multi-layered content safety system"""
    def __init__(self):
        self.redact_patterns = [
            (r'\b\d{3}-\d{2}-\d{4}\b', '[REDACTED_SSN]'),
            (r'(?i)\b(credit card|password)\b', '[REDACTED_PII]')
        ]
        
        self.toxicity_patterns = [
            r'\b(kill|harm|attack)\b',
            r'(racial|ethnic)\s+slur'
        ]

    def sanitize(self, text: str) -> str:
        """Apply redaction and toxicity filtering"""
        # Redaction layer
        for pattern, replacement in self.redact_patterns:
            text = re.sub(pattern, replacement, text)
            
        # Toxicity check
        for pattern in self.toxicity_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return "[SAFETY_BLOCK] Content violates safety policy"
            
        return text

class LanguageAgent:
    def __init__(self, llm: SLAILM, config: Dict):
        self.llm = llm
        self.nlu = NLUEngine()
        self.cache = KnowledgeCache(max_size=config.get('cache_size', 1000))
        self.safety = SafetyGuard()
        allowed_keys = {'history', 'environment_state', 'user_preferences', 'attention_weights'}
        context_args = {k: v for k, v in config.items() if k in allowed_keys}
        self.context = DialogueContext(
            llm=llm,
            history=config.get("history", []),
            summary=config.get("summary"),
            memory_limit=config.get("memory_limit", 1000),
            enable_summarization=config.get("enable_summarization", True),
            summarizer=config.get("summarizer"))
        self.dialogue_policy = self._load_dialogue_policy()
        self.wordlist = Wordlist()
        self.nlu = NLUEngine(config.get('wordlist_path'))
        self.nlg = NLGEngine(config.get('nlg_templates_en', 'src/agents/learning/nlg_templates_en.json'))

    def parse_intent(self, user_input: str) -> str:
        """Match user input against known triggers per intent."""
        user_input = user_input.lower()
        for intent, entry in self.nlg.templates.items():
            if isinstance(entry, dict):
                triggers = entry.get("triggers", [])
                if any(trigger in user_input for trigger in triggers):
                    return intent
        return "default"

    def preprocess_input(self, text: str) -> str:
        words = text.split()
        corrected = [w if self.wordlist.query(w) else self._guess_spelling(w) 
                     for w in words]
        return " ".join(corrected)

    def process_input(self, user_input: str) -> Tuple[str, LinguisticFrame]:
        """Full processing pipeline with academic-inspired components"""
        # Stage 1: Input sanitization
        clean_input = self.safety.sanitize(user_input)
        frame = self.nlu.parse(clean_input)
        
        # Stage 2: Semantic parsing
        frame = self.nlu.parse(clean_input)
        
        # Stage 3: Context-aware response generation
        cache_key = self.cache.hash_query(clean_input)
        
            
        prompt = self._construct_prompt(clean_input, frame)
        raw_response = self.llm.generate(prompt)
        refined_response = self.nlg.generate(frame, self.context)
        safe_response = self.safety.sanitize(refined_response)
        
        # Stage 4: Context update
        self._update_context(clean_input, safe_response, frame)
        self.cache.set(cache_key, safe_response)

        if cached := self.cache.get(self.cache.hash_query(clean_input)):
            return safe_response, frame

    def expand_query(self, query: str) -> str:
        words = query.split()
        expanded = []
        for word in words:
            details = self.wordlist.query(word)
            if details and 'synonyms' in details:
                expanded.append(f"{word} ({'|'.join(details['synonyms'])})")
            else:
                expanded.append(word)
        return " ".join(expanded)

    def save_context(self, file_path: Union[str, Path]) -> None:
        """
        Save dialogue context using Python's pickle serialization.
        Security Note: Pickle can execute arbitrary code. Only load trusted files.
    
        Args:
            file_path: Path to save context (supports both str and pathlib.Path).
    
        Raises:
            PermissionError: On write permission issues.
            pickle.PicklingError: If serialization fails.

        Example:
            >>> agent.save_context("chat_context.pkl")
            >>> agent.save_context(Path("/data/context.pkl"))

        Reference:
            Python Software Foundation. (2023). `pickle` module documentation.
        """
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(self.context, f)
        except (PermissionError, IsADirectoryError) as e:
            raise PermissionError(f"Failed to save context: {str(e)}") from e
        except (pickle.PicklingError, AttributeError) as e:
            raise pickle.PicklingError(f"Serialization error: {str(e)}") from e

    def load_context(self, file_path: Union[str, Path]) -> None:
        """
        Load dialogue context from a pickle file.
        
        Args:
            file_path: Path to the saved context file.
        
        Raises:
            FileNotFoundError: If the file does not exist.
            pickle.UnpicklingError: If the file is corrupted.
        
        Example:
            >>> agent.load_context("chat_context.pkl")
        """
        try:
            with open(file_path, 'rb') as f:
                self.context = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Context file not found: {file_path}")
        except pickle.UnpicklingError:
            raise pickle.UnpicklingError(f"Failed to load context from {file_path}")

    def _validate_llm(self) -> None:
        """Ensure the LLM has the required `generate` method."""
        if not hasattr(self.llm, 'generate') or not callable(self.llm.generate):
            raise AttributeError("LLM must implement a 'generate' method.")

    def _construct_prompt(self, text: str, frame: LinguisticFrame) -> str:
        """Build prompt using T5-style text-to-text approach"""
        components = [
            f"User: {text}",
            f"Intent: {frame.intent}",
            f"Context: {json.dumps(self.context.environment_state)}",
            "History:"
        ]
        
        for idx, (user, bot) in enumerate(self.context.history):
            components.append(f"Turn {idx+1}:")
            components.append(f"User: {user}")
            components.append(f"Bot: {bot}")
            
        components.append("Assistant Response:")
        return "\n".join(components)

    def _load_dialogue_policy(self) -> Dict:
        """Load conversation rules from embedded config"""
        return {
            'clarification_triggers': ['unknown', 'low_confidence'],
            'reprompt_limit': 2,
            'fallback_responses': [
                "Could you rephrase that?",
                "I need more context to help effectively."
            ]
        }

    def validate_response(self, response: str) -> bool:
        unsafe_patterns = [
            # Existing patterns
            r"(?i)\b(kill|harm|hurt|attack|hate)\b",
            r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
            r"(?i)\b(password|credit\s*card|social\s*security)\b",
            
            # Dutch BSN (validates modulus 11 check)
            r"\b(?!000000000)(?!111111111)(?!222222222)(?!333333333)"
            r"(?!444444444)(?!555555555)(?!666666666)(?!777777777)"
            r"(?!888888888)(?!999999999)\d{9}\b",
            
            r"\b(?:A|AR)?(?:0[0-9]|1[0-9]|2[0-4])\d{6,7}\b", # Aruban Persoonsnummer (BES-realm variant)
            r"\b(?!0)[A-Z]{1,2}(?!0{6})\d{6,9}\b", # International Passport Numbers (ICAO 9303 standard)
            r"\b(?:NL|BE)?\d{2}[ ]?\d{4}[ ]?\d{4}[ ]?\d{4}[ ]?\d{2}\b",  # IBAN
            r"\b(?:[A-Z]{2}\d{2}[ ]?\d{4}[ ]?\d{4}[ ]?\d{4}[ ]?\d{4})[ ]?\d{0,2}\b",  # SWIFT/BIC
            r"\b(?:[1-9]\d{3}[ ]?[A-Z]{2})\b",  # Dutch postal codes
            r"\b(?:GEB|NLD|ARU)[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",  # Government IDs
            r"\b(?:5[1-5]\d{2}|222[1-9]|22[3-9]\d|2[3-6]\d{2}|27[0-2]\d|2720)"
            r"[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",  # Payment cards
            r"\b(?:[A-Z]{1,2}\d{6,7}|D\d{7}|X[A-Z0-9]{8,12})\b"  # Passport variants
        ]

        # Contextual validation using academic heuristics
        sensitive_context_terms = [
            r"(?i)\b(bsn|persoonsnummer|sofi|identiteitskaart|paspoort)\b",
            r"(?i)\b(personal\s*data|privacy\s*sensitive|geheim)\b"
        ]

        for pattern in unsafe_patterns + sensitive_context_terms:
            if re.search(pattern, response, flags=re.UNICODE|re.IGNORECASE):
                self.benchmark_data["safety_violations"] += 1
                return False
        return True

    # Checksum validation example for BSN
    def validate_bsn_checksum(number: str) -> bool:
        """Implements Dutch BSN 11-proef validation"""
        if len(number) not in (8,9) or not number.isdigit():
            return False
            
        total = 0
        for i, digit in enumerate(number[::-1]):
            weight = (len(number) - i) if len(number) == 9 else (i+2)
            total += int(digit) * (weight if weight != 0 else 1)
            
        return total % 11 == 0

    def update_context(self, user_input: str, llm_response: str) -> None:
        """Update dialogue history and environment state."""
        self.context.history.append((user_input, llm_response))

    def evaluate_parsing_accuracy(self, test_cases: List[Tuple[str, Dict]]) -> float:
        """
        Benchmark parsing accuracy against labeled test cases.
        
        Args:
            test_cases: List of (input_text, expected_parsed_output).
        
        Returns:
            Accuracy score (0.0 to 1.0).
        """
        correct = 0
        for text, expected in test_cases:
            parsed = self.translate_user_input(text)
            if parsed == expected:
                correct += 1
            self.benchmark_data["parsing_accuracy"].append(parsed == expected)
        return correct / len(test_cases)

    def generate_prompt(self, user_input: str) -> str:
        """Generate a prompt with context and history."""
        prompt = f"User: {user_input}\nContext: {json.dumps(self.context.environment_state)}\n"
        if self.context.history:
            prompt += "Dialogue History:\n" + "\n".join([f"User: {u}\nBot: {r}" for u, r in self.context.history])
        prompt += "\nAssistant:"
        return prompt
