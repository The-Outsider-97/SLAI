"""
Core Function:
Performs Natural Language Understanding — extracting structured meaning from raw user input.

Responsibilities:
- Tokenize input (using your BPE tokenizer).
- Identify intents (what the user wants).
- Extract entities (important values like names, dates, etc.).
- Leverage the structured wordlist and embeddings for semantic similarity, synonyms, etc.

Why it matters:
This is the brain of the language agent — it translates text into actionable representations that drive the agent’s behavior.
"""
import datetime
import math
import re
import torch
import yaml, json
import textstat
import ply.lex as lex
import multiprocessing as mp

from pathlib import Path
from functools import partial
from dataclasses import dataclass, field, asdict
from collections import deque, defaultdict
from typing import Dict, Tuple, Optional, List, Any, Union, OrderedDict, Set, Iterable

from src.agents.language.utils.config_loader import load_global_config, get_config_section
from src.agents.language.utils.linguistic_frame import LinguisticFrame, SpeechActType
from src.agents.language.utils.language_tokenizer import LanguageTokenizer
from src.agents.language.utils.language_transformer import LanguageTransformer
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("NLU Engine")
printer = PrettyPrinter

# Forward declaration for Wordlist to satisfy type hints before actual import
class Wordlist:
    """Advanced linguistic processor with phonetics, morphology, and semantic analysis"""
    
    def __init__(self, n: int = 3):
        self.config = load_global_config()
        self.main_wordlist_path = self.config.get('main_wordlist_path')
        self.wordlist_path = self.config.get('wordlist_path')
        self.modality_markers_path = self.config.get('modality_markers_path')

        self.nlu_config = get_config_section('nlu')
        self.sentiment_lexicon_path = self.nlu_config.get('sentiment_lexicon_path')
        self.custom_intent_patterns_path = self.nlu_config.get('custom_intent_patterns_path')
        self.custom_entity_patterns_path = self.nlu_config.get('custom_entity_patterns_path')
        self.morphology_rules_path = self.nlu_config.get('morphology_rules_path')
        self.glove_synonym_threshold = self.nlu_config.get('glove_synonym_threshold')
        self.glove_top_synonyms = self.nlu_config.get('glove_top_synonyms')
        self.glove_path = self.nlu_config.get('glove_path')

        self.cache_config = get_config_section('language_cache')
        self.max_cache_size = self.cache_config.get('max_size')

        self.glove_vectors = self._load_glove_vectors()

        self.tokenizer = LanguageTokenizer()
        self.transformer = LanguageTransformer()

        self.path = Path(self.wordlist_path)
        self.path = Path(self.main_wordlist_path)
        self.n = n
        self.segmented_ngram_models = {}
        self.data = {}
        self.metadata = {}
        try:
            self._load()
        except (FileNotFoundError, ValueError, Exception) as e:
            logger.error(f"Failed to load wordlist data from {self.path}: {e}. Initializing with empty data.")
            self.data = {}
            self.metadata = {}
        
        # Advanced caching systems
        self.lru_cache = OrderedDict()
        self.lfu_cache = defaultdict(int)
        
        # Precomputed linguistic data
        self.phonetic_index = defaultdict(set)
        self.ngram_index = defaultdict(set)
        if self.data:
            self._precompute_linguistic_data()
        else:
            logger.warning("Wordlist data empty, skipping precomputation.")

        self.ngram_model = defaultdict(lambda: defaultdict(int)) # Language model parameters

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

        for char, neighbors in reverse_mapping.items():
             self.keyboard_layout[char] = {**self.keyboard_layout.get(char, {}), **neighbors}

    def add_word(self, word: str, metadata: dict = None):
        word_lower = word.lower()
        if word_lower not in self.data:
            self.data[word_lower] = metadata or {}
            # Update indices
            self._update_linguistic_data(word_lower)

    def _update_linguistic_data(self, text: str):
        """
        Updates the internal vocabulary and tag patterns based on new input text.
        This helps the engine adapt over time with new domain-specific language.
    
        Args:
            text (str): The raw input sentence to be used for updating linguistic data.
        """
        if not text:
            logger.warning("No text provided for linguistic data update.")
            return
    
        # Tokenize and tag the input
        tokens = self.tokenizer.tokenize(text)
        tagged_tokens = self.tagger.tag(tokens)
    
        # Update word frequency in the wordlist
        for token, tag in tagged_tokens:
            if self.wordlist is not None:
                if hasattr(self.wordlist, "update_term"):
                    self.wordlist.update_term(token)
                elif isinstance(self.wordlist, dict):
                    self.wordlist.setdefault("terms", []).append(token)
    
        # Optionally, update any learned patterns or tag associations
        for _, tag in tagged_tokens:
            self.tag_frequencies[tag] += 1
    
        logger.info(f"Linguistic data updated with {len(tagged_tokens)} tokens from new input.")

    def _load_glove_vectors(self) -> Dict[str, List[float]]:
        """Load pre-trained GloVe vectors from JSON file"""
        printer.status("INIT", "GloVe initialized", "info")

        if not self.glove_path or not Path(self.glove_path).exists():
            logger.warning(f"GloVe vectors file not found: {self.glove_path}")
            return {}
        
        try:
            with open(self.glove_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load GloVe vectors: {e}")
            return {}

    def _load(self) -> None:
        """Robust data loading with validation"""
        printer.status("INIT", "Loader initialized", "info")

        if not self.path.exists():
            raise FileNotFoundError(f"Wordlist missing: {self.path}")

        with open(self.path, 'r', encoding='utf-8') as f:
            raw = json.load(f)

        required_keys = {'words', 'metadata'} # Simplified required keys based on usage
        if not all(key in raw for key in required_keys):
             # Allow just a list of words as a fallback simple wordlist
             if isinstance(raw, list) and all(isinstance(item, str) for item in raw):
                 logger.warning(f"Wordlist {self.path} is a simple list. Using basic lookup only.")
                 self.data = {word.lower(): {} for word in raw} # Convert list to dict structure
                 self.metadata = {"version": "N/A", "language": "en (assumed)"}
                 return # Loading successful, but basic structure
             else:
                 raise ValueError("Invalid wordlist format - missing required keys 'words' or 'metadata', and not a simple list.")

        if not isinstance(raw['words'], dict):
             raise ValueError("Invalid wordlist format - 'words' key must contain a dictionary.")

        self.data = raw['words']
        self.metadata = raw.get('metadata', {}) # Metadata is optional

    def _validate_word_entries(self) -> None:
        """Ensure all entries have valid structure"""
        printer.status("INIT", "Word validator initialized", "info")

        for word, entry in self.data.items():
            if not isinstance(entry, dict):
                logger.warning(f"Invalid entry format for word: {word}. Should be dict, got {type(entry)}. Skipping.")
                continue
            if 'pos' not in entry or 'synonyms' not in entry:
                raise ValueError(f"Missing required fields in entry: {word}")
            if 'synonyms' not in entry:
                logger.debug(f"Missing 'synonyms' in entry: {word}")

    def _precompute_linguistic_data(self) -> None:
        """Precompute phonetic and n-gram indices from loaded data."""
        printer.status("INIT", "Data precomputation initialized", "info")

        # Ensure necessary attributes/methods exist (e.g., _metaphone, _soundex)
        if not hasattr(self, '_metaphone') or not hasattr(self, '_soundex'):
             logger.warning("Phonetic methods not available. Skipping phonetic indexing.")
             phonetic_methods_available = False
        else:
             phonetic_methods_available = True

        for word in self.data:
            word_lc = word.lower()
            if not word_lc.strip():
                continue

            # Phonetic representations (only if phonetic methods are available)
            if phonetic_methods_available:
                 try:
                    metaphone_key = self._metaphone(word_lc)
                    soundex_key = self._soundex(word_lc)
                    if metaphone_key: self.phonetic_index[metaphone_key].add(word)
                    if soundex_key: self.phonetic_index[soundex_key].add(word)
                 except Exception as e:
                    logger.warning(f"Error computing phonetic key for '{word}': {e}")

            # Generate n-gram indices
            for n in range(1, self.n + 1): # Unigrams, bigrams, trigrams up to self.n
                # Standard n-grams
                for i in range(len(word_lc) - n + 1):
                    ng = word_lc[i:i+n]
                    self.ngram_index[ng].add(word)

                # Position-aware n-grams with boundary markers (example for n=2,3)
                if n >= 2:
                    padded = f'^{word_lc}$'
                    for i in range(len(padded) - n + 1):
                         bng = padded[i:i+n]
                         self.ngram_index[bng].add(word)

    def _process_segment(self, label: str, word_iterable: Iterable[str]) -> Dict[int, defaultdict[Tuple[str, ...], int]]:
        """
        Processes a single segment to count n-grams.
        """
        printer.status("INIT", "Process segmentation initialized", "info")

        ngram_counts = [defaultdict(int) for _ in range(self.n)]
        for word in word_iterable:
            tokens = word.split()
            for ngram_len in range(1, self.n + 1):
                for i in range(len(tokens) - ngram_len + 1):
                    ngram = tuple(tokens[i : i + ngram_len])
                    ngram_counts[ngram_len - 1][ngram] += 1
        return {label: ngram_counts}

    def _witten_bell_smoothing(self, order_counts: defaultdict[Tuple[str, ...], int],
                                lower_order_counts: defaultdict[Tuple[str, ...], int] = None) -> defaultdict[Tuple[str, ...], float]:
        """
        Applies Witten-Bell smoothing to n-gram counts.
        """
        printer.status("INIT", "Witten Bell initialized", "info")

        smoothed_probs = defaultdict(float)
        total_count = sum(order_counts.values())
        unique_continuations = len(set(ngram[:-1] for ngram in order_counts))

        for ngram, count in order_counts.items():
            context = ngram[:-1]
            if total_count > 0:
                smoothed_probs[ngram] = (count / (total_count + unique_continuations))
            else:
                smoothed_probs[ngram] = 1.0 / (unique_continuations + 1) # Handle cases with no counts

        # Incorporate lower-order probabilities
        if lower_order_counts is not None:
            total_lower_count = sum(lower_order_counts.values())
            unique_lower_continuations = len(set(ngram[:-1] for ngram in lower_order_counts)) if lower_order_counts else 0
            unseen_prob_mass = unique_continuations / (total_count + unique_continuations) if total_count > 0 else 1.0

            for ngram, prob in smoothed_probs.items():
                lower_order_token = ngram[1:]
                lower_order_prob = 0.0
                if lower_order_counts and lower_order_token in lower_order_counts:
                    lower_order_prob = lower_order_counts.get(lower_order_token, 0) / (total_lower_count + unique_lower_continuations) if total_lower_count > 0 else 1.0 / (unique_lower_continuations + 1)

                smoothed_probs[ngram] = prob + unseen_prob_mass * lower_order_prob

        return smoothed_probs

    def build_segmented_ngram_models_streaming(self, segments: Dict[str, Iterable[str]], n: int = 3, num_processes: int = None) -> None:
        """
        Build separate n-gram models for different wordlist segments using streaming
        and parallel processing with Witten-Bell smoothing.
        `segments` is a dict of label -> iterable of words.
        """
        printer.status("INIT", "ngram initialized", "info")

        self.n = n
        self.segmented_ngram_models = {}
        if num_processes is None:
            num_processes = mp.cpu_count()

        pool = mp.Pool(processes=num_processes)
        partial_process_segment = partial(self._process_segment, n=self.n)
        results = pool.starmap(partial_process_segment, segments.items())
        pool.close()
        pool.join()

        # Merge counts from parallel processing
        raw_ngram_counts = {}
        for res in results:
            raw_ngram_counts.update(res)

        # Apply Witten-Bell smoothing
        self.segmented_ngram_models = {}
        for label, counts_list in raw_ngram_counts.items():
            smoothed_models = []
            for i in range(self.n):
                lower_order_counts = counts_list[i - 1] if i > 0 else None
                smoothed_models.append(self._witten_bell_smoothing(counts_list[i], lower_order_counts))
            self.segmented_ngram_models[label] = smoothed_models

    # PHONETIC ALGORITHMS ------------------------------------------------------
    
    def _soundex(self, word: str) -> str:
        """Soundex phonetic encoding implementation"""
        if not word:
            return ""
    
        # Step 0: Convert to uppercase and remove non-alphabetic characters
        word = re.sub(r'[^A-Za-z]', '', word).upper()
        
        # Step 1: Retain first letter
        first_char = word[0]
        soundex_code = [first_char]
        
        # Soundex mapping dictionary
        char_map = {
            'BFPV': '1',
            'CGJKQSXZ': '2',
            'DT': '3',
            'L': '4',
            'MN': '5',
            'R': '6'
        }
        
        # Step 2: Convert remaining characters
        prev_code = ''
        for char in word[1:]:
            # Handle 'H' and 'W' separators
            if char in 'HW':
                continue
                
            matched = False
            for chars, code in char_map.items():
                if char in chars:
                    current_code = code
                    matched = True
                    break
            
            # Handle vowels (including Y after first character)
            if not matched:
                if char in 'AEIOUY':
                    current_code = ''
                else:
                    current_code = ''
                    continue
            
            # Skip duplicates
            if current_code != prev_code:
                soundex_code.append(current_code)
                prev_code = current_code
            
            # Stop when we have 3 digits
            if len(soundex_code) == 4:
                break
        
        # Step 3: Pad/truncate to 4 characters
        if len(soundex_code) < 4:
            soundex_code.extend(['0'] * (4 - len(soundex_code)))
        
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
    def phonetic_candidates(self, word: str) -> List[str]:
        """Get phonetically similar candidates using precomputed index."""
        if not self.phonetic_index:
             logger.warning("Phonetic index not built. Cannot find phonetic candidates.")
             return []
        if not hasattr(self, '_metaphone') or not hasattr(self, '_soundex'):
             logger.warning("Phonetic methods not available. Cannot find phonetic candidates.")
             return []

        word_lower = word.lower()
        candidates = set()
        try:
            meta_key = self._metaphone(word_lower)
            soundex_key = self._soundex(word_lower)
            candidates.update(self.phonetic_index.get(meta_key, set()))
            candidates.update(self.phonetic_index.get(soundex_key, set()))
        except Exception as e:
             logger.warning(f"Error generating phonetic keys for '{word}': {e}")

        return list(candidates)
    
    # SEMANTIC ANALYSIS ---------------------------------------------------------

    def semantic_similarity(self, word1: str, word2: str) -> float:
        """Calculate enhanced semantic similarity using GloVe and Transformer embeddings."""
        # GloVe-based similarity
        glove_sim = 0.0
        if self.glove_vectors:
            vec1 = self.glove_vectors.get(word1.lower())
            vec2 = self.glove_vectors.get(word2.lower())
            if vec1 and vec2:
                glove_sim = self._cosine_similarity(vec1, vec2)
        
        # Transformer-based similarity
        transformer_sim = 0.0
        try:
            emb1 = self._get_transformer_embedding(word1)
            emb2 = self._get_transformer_embedding(word2)
            if emb1 and emb2:
                transformer_sim = self._cosine_similarity(emb1, emb2)
        except Exception as e:
            logger.warning(f"Transformer similarity failed: {e}")
            if glove_sim == 0.0:
                return 0.0
        
        # Combine scores
        combined_sim = 0.7 * transformer_sim + 0.3 * glove_sim
        return max(-1.0, min(1.0, combined_sim))
    
    def _get_transformer_embedding(self, word: str) -> Optional[List[float]]:
        """Get contextual embedding for a word using Transformer"""
        try:
            # Use class tokenizer instead of undefined variable
            tokens = self.tokenizer.tokenize(word)
            input_ids = torch.tensor(
                [self.tokenizer.token_to_id(token) for token in tokens]
            ).unsqueeze(0)
            
            # Use transformer's encoder directly
            with torch.no_grad():
                encoder_output = self.transformer.encode(input_ids)
                return torch.mean(encoder_output, dim=1).squeeze().tolist()
        except Exception as e:
            logger.error(f"Transformer embedding failed: {e}")
            return None

    def _word_vector(self, word: str) -> List[float]:
        """Get pre-trained GloVe vector for a word"""
        if not self.glove_vectors: return None
        return self.glove_vectors.get(word.lower(), None)

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Optimized cosine similarity for numerical lists."""
        dot_product = sum(v1*v2 for v1, v2 in zip(vec1, vec2))
        norm_a = math.sqrt(sum(v**2 for v in vec1))
        norm_b = math.sqrt(sum(v**2 for v in vec2))

        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0 # Avoid division by zero

        return dot_product / (norm_a * norm_b)

    # LANGUAGE MODELING ---------------------------------------------------------
    
    def word_probability(self, word: str, context: List[str] = None) -> float:
        """Calculate relative frequency probability with backoff smoothing"""
        if not context or len(context) < self.ngram_order-1:
            # Fallback to unigram probability
            return self.ngram_counts[0].get((word,), 1) / sum(self.ngram_counts[0].values())
        
        # Try highest order n-gram first
        for order in range(min(self.ngram_order, len(context)+1), 0, -1):
            ngram = tuple(context[-(order-1):] + [word])
            count = self.ngram_counts[order-1].get(ngram, 0)
            
            if count > 0:
                denominator = sum(
                    self.ngram_counts[order-1][tuple(context[-(order-1):] + [w])] 
                    for w in self.vocabulary
                )
                return count / denominator
                
        # Absolute backoff to unigram
        return self.ngram_counts[0].get((word,), 1) / sum(self.ngram_counts[0].values())

    def context_suggestions(self, previous_words: List[str], limit: int = 5) -> List[Tuple[str, float]]:
        """Predict next words using n-gram probabilities with backoff"""
        suggestions = defaultdict(float)
        max_order = min(self.ngram_order, len(previous_words)+1)
        
        # Calculate interpolated probability
        for order in range(max_order, 0, -1):
            context = previous_words[-(order-1):] if order > 1 else []
            lambda_weight = 0.4 ** (max_order - order)  # Weighting scheme
            
            # Get possible next words
            possible_ngrams = [k for k in self.ngram_counts[order-1] 
                             if k[:-1] == tuple(context)]
            
            total = sum(self.ngram_counts[order-1][gram] for gram in possible_ngrams)
            
            for gram in possible_ngrams:
                word = gram[-1]
                prob = self.ngram_counts[order-1][gram] / total
                suggestions[word] += lambda_weight * prob
                
        # Normalize and sort
        total_prob = sum(suggestions.values())
        normalized = [(w, p/total_prob) for w, p in suggestions.items()]
        
        return sorted(normalized, key=lambda x: -x[1])[:limit]

    # Helper methods
    @property
    def vocabulary(self):
        """Returns the list of words in the wordlist data."""
        return list(self.data.keys())
    
    @property
    def words(self):
        """Alias for vocabulary."""
        return list(self.data.keys())

    def query(self, word: str) -> Optional[Dict]:
        """Case-insensitive lookup with caching"""
        if not isinstance(word, str) or not word.strip():
             return None

        word_lower = word.strip().lower()

        # Check cache first
        if word_lower in self.lru_cache:
            self._update_lfu(word_lower)
            self.lru_cache.move_to_end(word_lower)
            return self.lru_cache[word_lower]

        # Check main data
        entry = self.data.get(word_lower)
        if entry:
            self._update_cache(word_lower, entry) # Add/Update cache
            return entry

        # Check stemmed version (optional, based on need and Stemmer reliability)
        stemmed = self.stem(word) # Requires a stemmer implementation
        if stemmed and stemmed != word_lower:
            entry = self.data.get(stemmed)
            if entry:
                # Cache the stemmed version lookup result against the original word
                self._update_cache(word_lower, entry)
                return entry

        return None  
  
    def _update_cache(self, word: str, entry: Dict) -> None:
        """Hybrid cache update strategy"""
        self.lru_cache[word] = entry
        self._update_lfu(word) # Increment LFU count
        self.lru_cache.move_to_end(word) # Move to end for LRU

        if len(self.lru_cache) > self.max_cache_size:
            self._evict_cache_item()

    def _update_lfu(self, word: str) -> None:
        """Increment LFU count for a word."""
        self.lfu_cache[word] += 1

    def _evict_cache_item(self) -> None:
        """Eviction strategy: remove the least frequently used, breaking ties with LRU."""
        if not self.lfu_cache: return # Nothing to evict

        # 1. Find the minimum frequency
        min_freq = float('inf')
        try:
            min_freq = min(self.lfu_cache.values())
        except ValueError: # Handle case where values() is empty (shouldn't happen if lfu_cache is not empty)
             return

        # 2. Find all items with minimum frequency
        min_freq_candidates = [k for k, v in self.lfu_cache.items() if v == min_freq]

        # 3. Of the minimum frequency candidates, find the least recently used (first in OrderedDict)
        for key in self.lru_cache:
            if key in min_freq_candidates:
                logger.debug(f"Evicting '{key}' from cache (LFU/LRU).")
                del self.lru_cache[key]
                del self.lfu_cache[key]
                return # Evicted one item

    def _add_synonym_edge(self, word: str, synonym: str) -> None:
        """Add bidirectional synonym edge if the synonym exists in the wordlist."""
        synonym_lower = synonym.lower()
        if synonym_lower in self.data and synonym_lower != word.lower():
            self.graph[word.lower()].add(synonym_lower)
            self.graph[synonym_lower].add(word.lower())

    def _find_glove_synonyms(self, target_word: str, top_n: int, threshold: float) -> List[str]:
        """Find top-N words with GloVe similarity above threshold."""
        target_vec = self._word_vector(target_word)
        if not target_vec or all(v == 0 for v in target_vec):
            return []
    
        similarities = []
        for candidate in self.data:
            if candidate == target_word:
                continue
            candidate_vec = self._word_vector(candidate)
            if not candidate_vec:
                continue
            sim = self._cosine_similarity(target_vec, candidate_vec)
            if sim >= threshold:
                similarities.append((candidate, sim))

        similarities.sort(key=lambda x: -x[1])
        return [word for word, _ in similarities[:top_n]]

    def validate_word(self, word: str) -> bool:
        """Comprehensive word validation"""
        return (
            self._check_orthography(word) and
            self._check_morphology(word) and
            self._check_phonotactics(word)
        )

    def correct_typo(self, word: str) -> Tuple[str, float]:
        """Main interface for typo correction"""
        if word in self.data:
            return (word, 1.0)  # Already correct

        suggestions = self.spell_checker.suggest(word)
        return (suggestions[0], 1.0) if suggestions else (word, 0.0)

    def _check_orthography(self, word: str) -> bool:
        """Advanced orthographic validation using multiple strategies"""
        # Direct lexicon lookup
        if word in self.data:
            return True
        
        # Phonetic fallback check
        phonetic_matches = self.phonetic_candidates(word)
        if any(match in self.data for match in phonetic_matches):
            return True
        
        # Edit distance to known words
        candidates = self.phonetic_candidates(word)
        if candidates:
            closest = min(candidates, key=lambda x: self.weighted_edit_distance(word, x))
            if self.weighted_edit_distance(word, closest) < 2.0:
                return True  # Valid candidate found
            
            # Proceed to typo correction if no close candidate
            correction, confidence = self.correct_typo(word)
            return confidence > 0.7

        return self._validate_grapheme_phoneme(word) # Fallback to grapheme-phoneme alignment

    def _check_morphology(self, word: str) -> bool:
        """Morphological validation using language-specific rules"""
        morphology_path = self.config.get("nlu", {}).get("morphology_rules_path")
        if not morphology_path or not Path(morphology_path).exists():
            logger.warning("Morphology rules path missing or file not found.")
            return False
    
        with open(morphology_path, "r", encoding="utf-8") as f:
            all_rules = json.load(f)
    
        lang = self.metadata.get('language', 'en')
        rules = all_rules.get(lang, {})
        
        # Character level validation
        if not re.fullmatch(rules.get('valid_chars', r'^\p{L}+$'), word, re.IGNORECASE):
            return False

        if not self._validate_affixes(word, rules['allowed_affixes']): # Affix validation
            return False

        if '-' in word: # Compound word validation
            return self._validate_compounds(word, rules['compound_patterns'])

        if self.syllable_count(word) > rules['max_syllables']:  # Syllable constraints
            return False

        # Affix validation
        affix_rules = rules.get('allowed_affixes', {})
        for affix_type, prefixes in affix_rules.items():
            if affix_type == 'pre':
                if any(word.startswith(p) for p in prefixes) and not self._validate_prefix(word):
                    return False
            elif affix_type == 'suf':
                if any(word.endswith(s) for s in prefixes) and not self._validate_suffix(word):
                    return False

        if '-' in word: # Compound word validation
            if not any(re.fullmatch(p, word) for p in rules.get('compound_patterns', [])):
                return False
            return all(self.validate_word(part) for part in word.split('-'))
        
        # Syllable constraints
        if 'max_syllables' in rules:
            return self.syllable_count(word) <= rules['max_syllables']
        
        return True

    def _validate_affixes(self, word: str, affix_rules: dict) -> bool:
        """Generic affix validation for all languages"""
        for affix_type, affixes in affix_rules.items():
            if affix_type == 'pre':
                if any(word.startswith(p) for p in affixes):
                    stem = next(word[len(p):] for p in affixes if word.startswith(p))
                    if not self._validate_stem(stem):
                        return False
            elif affix_type == 'suf':
                if any(word.endswith(s) for s in affixes):
                    stem = next(word[:-len(s)] for s in affixes if word.endswith(s))
                    if not self._validate_stem(stem):
                        return False
        return True


    def _validate_stem(self, stem: str) -> bool:
        """Validate word stems after affix removal"""
        if not stem:
            return False
        return stem in self.data or self.stem(stem) in self.data

    def _validate_prefix(self, word: str) -> bool:
        """Prefix-stripping validation"""
        stem = self.stem(word)
        return stem != word and stem in self.data

    def _validate_suffix(self, word: str) -> bool:
        """Suffix-stripping validation"""
        base = word
        while base[-1] in {'s', 'd', 'g'} and len(base) > 2:
            base = base[:-1]
            if base in self.data:
                return True
        return False

    def _validate_compounds(self, word: str, patterns: list) -> bool:
        """Validate compound words against language patterns"""
        if not any(re.fullmatch(p, word) for p in patterns):
            return False
        
        # Check each component word
        parts = re.split(r'[-]', word)
        return all(self.validate_word(part) for part in parts)

    def _validate_grapheme_phoneme(self, word: str) -> bool:
        """Phonetic plausibility check"""
        phonetic = self._metaphone(word)
        return any(
            self._metaphone(known) == phonetic 
            for known in self.data.keys()
        )

    def _check_phonotactics(self, word: str) -> bool:
        """Validate word structure against language-specific phonotactic rules"""
        lang = self.metadata.get('language', 'en').lower()
        word_lower = word.lower()
    
        # Universal check: Minimum word structure
        if len(word_lower) < 1:
            return False
    
        # Language-specific rule sets
        if lang == 'en':
            return self._validate_english_phonotactics(word_lower)
        # Add other languages as needed
        return True  # Default pass for unsupported languages
    
    def _validate_english_phonotactics(self, word: str) -> bool:
        """English phonotactic constraints based on Cruttenden (2014)"""
        # 1. Mandatory vowel presence
        if not re.search(r'[aeiouy]', word):
            return False
    
        # 2. Invalid initial clusters (Blevins, 1995)
        invalid_onsets = {
            'tk', 'pf', 'gb', 'sr', 'dlr', 'lv', 'km',
            'bd', 'gt', 'pn', 'ts', 'dz', 'fp', 'vr'
        }
        if any(word.startswith(onset) for onset in invalid_onsets):
            return False
    
        # 3. Invalid final clusters (Harris, 1994)
        invalid_codas = {
            'mt', 'pn', 'dl', 'bm', 'lr', 'nm', 'tn',
            'aa', 'ii', 'uu', 'vv', 'jk', 'qq', 'xz'
        }
        if any(word.endswith(coda) for coda in invalid_codas):
            return False
    
        # 4. Maximal onset principle (Kahn, 1976)
        if re.search(r'[bcdfghjklmnpqrstvwxz]{4}', word):
            return False  # No quad-consonant sequences
    
        # 5. Vowel sequence constraints (Roach, 2000)
        if re.search(r'[aeiouy]{3}', word):
            return False
    
        # 6. Valid syllable structure (C)(C)(C)V(C)(C)(C)(C)
        syllables = self._split_syllables(word)
        for syl in syllables:
            if not re.fullmatch(r'^([bcdfghjklmnpqrstvwxz]{0,3}[aeiouy]+[bcdfghjklmnpqrstvwxz]{0,4})$', syl):
                return False
    
        return True

    def _split_syllables(self, word: str) -> list:
        """Basic syllabification using sonority hierarchy (Selkirk, 1984)"""
        vowels = 'aeiouy'
        syllables = []
        current = ''
        
        for i, char in enumerate(word):
            current += char
            if char in vowels:
                # Look ahead to determine syllable boundary
                if i < len(word)-1 and word[i+1] in vowels:
                    syllables.append(current)
                    current = ''
                elif i < len(word)-2 and word[i+1] not in vowels and word[i+2] in vowels:
                    syllables.append(current)
                    current = ''
        
        if current:
            syllables.append(current)
        return syllables

    def __contains__(self, word: str) -> bool:
        """Case-insensitive check against wordlist data (no cache side effects)."""
        return word.lower() in self.data  # Directly check the source data


class NLUEngine:
    """Rule-based semantic parser with fallback patterns"""
    def __init__(self, wordlist_instance: Wordlist):
        self.config = load_global_config()
        self.main_wordlist_path = self.config.get('main_wordlist_path')
        self.modality_markers_path = self.config.get('modality_markers_path')

        self.nlu_config = get_config_section('nlu')
        self.sentiment_lexicon_path = self.nlu_config.get('sentiment_lexicon_path')
        self.custom_intent_patterns_path = self.nlu_config.get('custom_intent_patterns_path')
        self.custom_entity_patterns_path = self.nlu_config.get('custom_entity_patterns_path')
        self.morphology_rules_path = self.nlu_config.get('morphology_rules_path')
        self.glove_path = self.nlu_config.get('glove_path')

        self.wordlist = wordlist_instance
        self.coherence_checker = None

        # Load intent patterns from JSON file
        with open(self.custom_intent_patterns_path, 'r') as f:
            self.intent_patterns = json.load(f)

        # Load entity patterns from JSON file
        with open(self.custom_entity_patterns_path, 'r') as f:
            self.entity_patterns = json.load(f)

        logger.info("NLU Engine initialized...")

    def _load_intent_patterns(self, path: str) -> Dict[str, Any]:
        """Load intent patterns from JSON file with better error handling"""
        if not path:
            logger.warning("Intent patterns path not provided in config")
            return {}
        
        try:
            file_path = Path(path)
            if not file_path.exists():
                logger.error(f"Intent patterns file not found at: {file_path}")
                return {}
                
            with open(file_path, 'r', encoding='utf-8') as f:
                patterns = json.load(f)
                
            # Convert old format to new format
            processed_patterns = {}
            for intent, pattern_data in patterns.items():
                if isinstance(pattern_data, dict) and "patterns" in pattern_data:
                    processed_patterns[intent] = pattern_data["patterns"]
                elif isinstance(pattern_data, list):
                    processed_patterns[intent] = pattern_data
                else:
                    logger.warning(f"Unexpected pattern format for intent '{intent}'")
                    
            logger.info(f"Loaded {len(processed_patterns)} intents from patterns file")
            return processed_patterns
            
        except Exception as e:
            logger.error(f"Error loading intent patterns: {e}")
            return {}

    def get_intents(self):
        return self.custom_intent_patterns_path

    def get_entities(self):
        return self.custom_entity_patterns_path

    def get_modalities(self):
        return self.modality_markers_path

    def get_lexicons(self):
        return self.sentiment_lexicon_path

    def get_morphologies(self):
        return self.morphology_rules_path
    
    def _match_intent_by_pattern(self, text: str) -> List[Tuple[str, str, float]]:
        """Match text against intent patterns using word boundaries"""
        text_clean = re.sub(r'[^\w\s]', '', text).lower().strip()
        intents = []
        
        for intent_name, patterns in self.intent_patterns.items():
            for pattern in patterns:
                # Clean and escape pattern
                pattern_clean = re.sub(r'[^\w\s]', '', pattern).lower().strip()
                
                # Skip empty patterns
                if not pattern_clean:
                    continue
                    
                # Create regex pattern that matches the entire phrase
                pattern_re = r'\b' + re.escape(pattern_clean) + r'\b'
                
                if re.search(pattern_re, text_clean):
                    # Calculate match score based on length ratio
                    score = len(pattern_clean) / max(1, len(text_clean))
                    intents.append((intent_name, pattern, score))
                    logger.debug(f"Pattern match: '{pattern}' -> '{intent_name}' (score: {score:.2f})")
        
        # Sort by match score descending
        return sorted(intents, key=lambda x: x[2], reverse=True)

    def _validate_temporal(self, entity: str) -> bool:
        """Temporal validation using date logic"""
        if isinstance(entity, set):
            entity = next(iter(entity), "")
        if not isinstance(entity, str):
            return False
    
        if re.match(r'\d{4}-\d{2}-\d{2}', entity):
            try:
                datetime.datetime.strptime(entity, '%Y-%m-%d')
                return True
            except ValueError:
                return False
        return True  # Accept relative times

    def _validate_quantity(self, entity: Any) -> bool:
        """Quantity validation using unit and value"""
        if isinstance(entity, set):
            entity = next(iter(entity), "")
        if not isinstance(entity, str):
            return False
    
        match = re.match(r'(\d+)\s*(\D+)', entity)
        return bool(match)

    def _validate_technical(self, entity: str) -> bool:
        """Technical spec validation"""
        if '-' in entity:
            parts = entity.split('-', 1)
            if len(parts) == 2:
                prefix, code = parts
                return prefix.isalpha() and code.isdigit()
        return True # Default to true if not matching specific invalid patterns

    def _validate_duration(self, entity: str) -> bool:
        """Duration validation using time expressions"""
        if isinstance(entity, set):
            entity = next(iter(entity), "")
        if not isinstance(entity, str):
            return False
        return bool(re.match(r"\d+\s*(second|minute|hour|day|week|month|year|decade)s?", entity, re.IGNORECASE))

    def _validate_term(self, entity: Any) -> bool:
        """Term validation for alphanumeric labels"""
        if isinstance(entity, set):
            entity = next(iter(entity), "")
        if not isinstance(entity, str):
            return False
        return bool(re.match(r"[a-zA-Z0-9\s]+", entity))

    def _validate_boolean(self, entity: Any) -> bool:
        """Boolean validation"""
        if isinstance(entity, set):
            entity = next(iter(entity), "")
        if not isinstance(entity, str):
            return False
    
        return entity.lower() in ['true', 'false', 'yes', 'no']

    def _normalize_entity(self, entity: Any) -> str:
        """Ensure entity is a string for validation purposes"""
        if isinstance(entity, set):
            entity = next(iter(entity), "")
        if not isinstance(entity, str):
            entity = str(entity)
        return entity

    def parse(self, text: str) -> LinguisticFrame:
        """Hybrid parsing using rules and simple statistics"""
        printer.status("INIT", "Parse initialized", "info")
    
        # Initialize frame with defaults
        frame = LinguisticFrame(
            intent='unknown',
            entities={},
            sentiment=0.0,
            modality='declarative',
            confidence=0.0,
            act_type=SpeechActType.ASSERTIVE
        )
    
        detected_intent = 'unknown'
        max_confidence_for_intent = 0.0  # Initialize properly as float

        # Match against intent patterns
        matched_intents = self._match_intent_by_pattern(text)
        if matched_intents:
            intent, pattern, score = matched_intents[0]
            logger.debug(f"Top intent match: '{intent}' with pattern '{pattern}' (score: {score:.2f})")
    
        # Intent detection
        for intent, patterns in self.intent_patterns.items():
            # Handle both list and dict formats in JSON
            actual_patterns = patterns
            if isinstance(patterns, dict) and 'patterns' in patterns:
                actual_patterns = patterns['patterns']
            
            current_intent_confidence = 0.0
            for pattern in actual_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    current_intent_confidence += 0.1
            
            # FIX: Ensure we're comparing floats
            if current_intent_confidence > max_confidence_for_intent:
                max_confidence_for_intent = current_intent_confidence
                detected_intent = intent
        
        frame.intent = detected_intent
        frame.confidence = min(1.0, max_confidence_for_intent)

        # Entity extraction
        entities = {}
        for entity_type, meta in self.entity_patterns.items():
            pattern_str = meta.get('pattern', "")
            # components = meta.get('components', {})
            formatted_pattern = pattern_str 

            matches = [m[0] if isinstance(m, tuple) and m else m for m in re.findall(formatted_pattern, text, re.IGNORECASE)]
            
            valid_matches = []
            if 'validation' in meta:
                validation_func_name = f"_validate_{meta['validation']}"
                validation_func = getattr(self, validation_func_name, None)
                if validation_func:
                    valid_matches = [m for m in matches if m and validation_func(m)]
                else:
                    logger.warning(f"Validation function {validation_func_name} not found for entity {entity_type}")
                    valid_matches = [m for m in matches if m]
            else:
                valid_matches = [m for m in matches if m] # Keep all non-empty matches

            if valid_matches:
                entities[entity_type] = valid_matches
        
        frame.entities = entities

        self._validate_entity_words(frame) # Validate extracted entity values against wordlist

        # Sentiment and Modality
        frame.sentiment = self._calculate_sentiment(text)
        frame.modality = self._detect_modality(text)
        
        # Recalculate confidence based on more factors
        frame.confidence = self._calculate_overall_confidence(text, frame)

        return frame # Return as LinguisticFrame object, not dict

    def _validate_entity_words(self, frame: LinguisticFrame) -> None:
        """Check if recognized entity values (words) exist in the wordlist."""
        printer.status("INIT", "Entity word validation initialized", "info")

        if not self.wordlist:
            logger.error("Wordlist not initialized for entity word validation.")
            return
    
        valid_entities = {}
        for entity_type, values in frame.entities.items():
            validated = []
            for v in values:
                # Convert to string and call query correctly with single argument
                word_str = str(v)
                if self.wordlist.query(word_str):  # CORRECTED: single argument
                    validated.append(v)
            if validated:
                valid_entities[entity_type] = validated
        frame.entities = valid_entities

    def _tokenize(self, text:str) -> List[str]:
        # Simple tokenizer, can be replaced with a more advanced one (e.g., BPE from agent)
        text = text.lower()
        text = re.sub(r"([.,!?;:])", r" \1 ", text) # Add space around punctuation
        tokens = text.split()
        return tokens

    def _calculate_sentiment(self, text: str) -> float:
        """Enhanced sentiment analysis using lexicon and syntactic patterns"""
        printer.status("INIT", "Sentiment Calculations initialized", "info")

        if not self.sentiment_lexicon_path:
            logger.warning("Sentiment lexicon not loaded. Returning neutral sentiment.")
            return 0.0
    
        valence_dict_data = {} # Use a different name to avoid confusion if self.sentiment_lexicon_path was intended for something else
        try:
            with open(self.sentiment_lexicon_path, 'r', encoding='utf-8') as f:
                valence_dict_data = json.load(f) # Load data into this new variable
        except FileNotFoundError:
            logger.error(f"Sentiment lexicon file not found: {self.sentiment_lexicon_path}")
            return 0.0
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from sentiment lexicon: {self.sentiment_lexicon_path}")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to load sentiment lexicon: {e}")
            return 0.0
            
        tokens = self._tokenize(text)
        # REMOVED: valence_dict = self.sentiment_lexicon_path # This was the error
        
        # Use the loaded dictionary: valence_dict_data
        intensifiers = valence_dict_data.get("intensifiers", {})
        negators = valence_dict_data.get("negators", [])
        positive_words = valence_dict_data.get("positive", {})
        negative_words = valence_dict_data.get("negative", {})
    
        sentiment_score = 0.0
        weight_sum = 0.0 
        negate_next = False
        current_intensity = 1.0
    
        for word in tokens:
            w_lower = word.lower()
            if w_lower in negators:
                negate_next = not negate_next 
                continue
            
            if w_lower in intensifiers:
                current_intensity *= intensifiers[w_lower]
                continue
    
            score = positive_words.get(w_lower, 0.0) + negative_words.get(w_lower, 0.0)
    
            if score != 0:
                if negate_next:
                    score *= -1
                    negate_next = False  
                
                effective_score = score * current_intensity
                sentiment_score += effective_score
                weight_sum += abs(effective_score) 
                current_intensity = 1.0  
        
        if weight_sum > 0:
            normalized_score = sentiment_score / weight_sum
            return max(-1.0, min(1.0, normalized_score))
        return 0.0
    
    def _calculate_confidence(self, matches: List[Tuple[str, str, float]]) -> float:
        """Calculate intent confidence based on match quality"""
        if not matches:
            return 0.0
        
        # Get the best match score
        best_score = matches[0][2]
        
        # Normalize score to 0.0-1.0 range
        normalized_score = min(1.0, best_score * 1.5)  # Boost longer matches
        
        # Apply minimum confidence for any match
        return max(0.5, normalized_score)  # Minimum 50% confidence for any match

    def _detect_modality(self, text: str) -> str:
        """Hierarchical modality detection"""
        printer.status("INIT", "Modalities detector initialized", "info")

        text_lower = text.lower()
    
        modality_markers_data = {}
        if self.modality_markers_path:
            try:
                with open(self.modality_markers_path, 'r', encoding='utf-8') as f:
                    modality_markers_data = json.load(f)
            except FileNotFoundError:
                logger.error(f"Modality markers file not found: {self.modality_markers_path}")
            except json.JSONDecodeError:
                logger.error(f"Error decoding JSON from modality markers file: {self.modality_markers_path}")
            except Exception as e:
                logger.error(f"Failed to load modality markers: {e}")
        else:
            logger.warning("Modality markers path not configured.")
    
        # Priority detection order based on typical marker strength
        modality_rules = [
            ('interrogative', modality_markers_data.get('interrogative', []) + ['?']),
            ('imperative', modality_markers_data.get('imperative', [])),
            ('conditional', modality_markers_data.get('conditional', [])),
            ('epistemic', modality_markers_data.get('epistemic', [])),
            ('deontic', modality_markers_data.get('deontic', [])),
            ('dynamic', modality_markers_data.get('dynamic', [])),
            ('alethic', modality_markers_data.get('alethic', [])),
        ]
        
        # Check for question mark first for interrogative
        if text_lower.strip().endswith('?'):
            return 'interrogative'
    
        for mod_type, markers in modality_rules:
            # Ensure marker is not None or empty before using in re.escape
            if any(re.search(r'\b' + re.escape(marker) + r'\b', text_lower) for marker in markers if marker):
                return mod_type
        
        tokens = self._tokenize(text_lower)
        if tokens:
            if tokens[0] in ["tell", "give", "show", "create", "delete", "update"] and len(tokens) > 1: # Imperative verbs
                # Ensure q_word is not None or empty
                if not any(q_word in tokens for q_word in modality_markers_data.get('interrogative', []) if q_word):
                    return 'imperative'
    
        return 'declarative' # Default

    def _calculate_overall_confidence(self, text: str, frame: LinguisticFrame) -> float:
        """Calculate overall confidence based on multiple factors."""
        printer.status("INIT", "Confidence calculations initialized", "info")

        # Base confidence from intent detection
        confidence_score = frame.confidence # This was max_confidence_for_intent

        # Factor in entity presence and validation (simple heuristic)
        if frame.entities:
            confidence_score += 0.1 * len(frame.entities)
        
        # Factor in lexical coverage (words in text that are in wordlist)
        # This requires the main Wordlist object
        if hasattr(self.wordlist, 'query'):
            tokens = self._tokenize(text)
            if tokens:
                known_tokens = sum(1 for token in tokens if self.wordlist.query(token))  # Remove extra argument
                coverage = known_tokens / len(tokens)
                confidence_score += 0.2 * coverage
        
        return min(1.0, max(0.0, confidence_score))
    

@dataclass
class DependencyRelation:
    head: str
    relation: str
    dependent: str

class EnhancedNLU(NLUEngine):
    """Implements advanced NLU techniques"""
    def __init__(self, wordlist_instance: Wordlist):
        super().__init__(wordlist_instance)
        self.config = load_global_config()
        self.enlu_config = get_config_section('nlu')

        # Psycholinguistic features
        # self.lexical_diversity_tool = lex.lex() #might need setup e.g. lex(my_lex) from src.agents.language.my_lex import my_lex
        self.readability_tool = textstat
        logger.info("EnhancedNLU initialized.")

        # Store context history if needed by coreference resolver
        self.context_history: Optional[deque] = None

    def set_context_history(self, history: deque):
        self.context_history = history

    def analyze_text_fully(self, text: str) -> Dict[str, Any]:
        """
        Run full NLU pipeline: coref resolution + dependency parsing + base parsing.
        """
        # Stage 0: Coreference resolution (if context history is available)
        resolved_text = self.coref_resolver.resolve(text, history=self.context_history)
        
        # Stage 1: Basic NLU parsing (intent, entities, sentiment, modality)
        # Use super().parse() to get the LinguisticFrame from the base NLUEngine
        linguistic_frame_obj = super().parse(resolved_text)

        # Stage 2: Dependency parsing on resolved text
        parse_tree_dict = self.dependency_parser.parse(resolved_text) # This needs to return a dict like {'root': ..., 'nodes': ...}
                                                                    # or List[DependencyRelation] based on its actual implementation.
                                                                    # For now, assuming it returns a dict as expected by downstream methods.

        # Stage 3: Enhanced sentiment and modality using parse tree
        linguistic_frame_obj.sentiment = self._enhanced_sentiment(resolved_text, parse_tree_dict)
        linguistic_frame_obj.modality = self._detect_modality_from_tree(parse_tree_dict) # Renamed from _detect_modality

        # Convert LinguisticFrame to dict for the final output structure of analyze_text_fully
        analysis_result = asdict(linguistic_frame_obj)
        analysis_result["resolved_text"] = resolved_text
        analysis_result["dependencies"] = parse_tree_dict # Or however dependencies are represented

        # Add psycholinguistic features
        try:
            # analysis_result["lexical_diversity"] = self.lexical_diversity_tool.ttr(resolved_text) # Example
            analysis_result["flesch_reading_ease"] = self.readability_tool.flesch_reading_ease(resolved_text)
        except Exception as e:
            logger.warning(f"Could not compute some psycholinguistic features: {e}")
            # analysis_result["lexical_diversity"] = None
            analysis_result["flesch_reading_ease"] = None
            
        return analysis_result

    def _enhanced_sentiment(self, text: str, parse_tree: Dict) -> float:
        """Sentiment analysis considering negation and intensity from parse tree"""
        # Implementation based on Socher et al. (2013) Recursive Neural Networks (Conceptual)
        # This requires parse_tree to have specific structure, e.g., nodes with 'relation', 'word', 'intensity'
        sentiment = 0.0
        if not parse_tree or 'nodes' not in parse_tree or not isinstance(parse_tree['nodes'], list):
            logger.warning("Parse tree is not in expected format for enhanced sentiment. Falling back to text-based.")
            return super()._calculate_sentiment(text) # Fallback to base NLU sentiment

        for node in parse_tree['nodes']:
            if not isinstance(node, dict): continue # Skip if node is not a dict

            word = node.get('word')
            relation = node.get('relation')
            intensity = node.get('intensity', 1.0) # Default intensity

            if relation == 'neg': # Assuming 'neg' relation for negation
                sentiment -= 0.5 * intensity # Example impact of negation
            elif word and hasattr(self.wordlist, 'query'):
                word_data = self.wordlist.query(word)
                # Assuming wordlist query returns dict with 'sentiment' score if available
                word_sentiment = word_data.get('sentiment', 0) if word_data else 0 
                sentiment += word_sentiment * intensity
        
        return math.tanh(sentiment)  # Squash to [-1, 1]

    def _detect_modality_from_tree(self, parse_tree: Dict) -> str:
        """Detect speech modality using verb patterns from parse tree"""
        # Inspired by Austin's Speech Act Theory (1975)
        if not parse_tree or 'root' not in parse_tree or not isinstance(parse_tree['root'], dict):
            logger.warning("Parse tree is not in expected format for modality detection.")
            return "statement" # Fallback

        main_verb_lemma = parse_tree['root'].get('lemma')
        
        # Example modality map based on main verb lemma
        modality_map = {
            'ask': 'query', 'enquire': 'query', 'question': 'query',
            'request': 'command', 'order': 'command', 'tell': 'command', # 'tell' can be ambiguous
            'suggest': 'proposal', 'propose': 'proposal',
            'can': 'epistemic', 'may': 'epistemic', 'might': 'epistemic', # Modals
            'should': 'deontic', 'must': 'deontic',
        }
        
        if main_verb_lemma and main_verb_lemma in modality_map:
            return modality_map[main_verb_lemma]
            
        # Further checks can be added here, e.g., for question words, imperative mood markers in the tree
        
        return 'statement' # Default
    

if __name__ == "__main__":
    print("\n=== Running Natural Language Understanding Engine (NLU Engine) ===\n")
    printer.status("Init", "NLU Engine initialized", "success")

    n=3

    list = Wordlist(n=n)
    engine = NLUEngine(wordlist_instance=Wordlist)
    engine2 = EnhancedNLU(wordlist_instance=Wordlist)
    print(f"Suggestions: {list}")
    print(f"Suggestions: {engine}")
    print(f"Suggestions: {engine2}")

    print("\n* * * * * Phase 2 * * * * *\n")
    label= ("speaker", "segment1")
    word ="speaker"
    order_counts = defaultdict(int, {
        ('the', 'cat'): 4,
        ('cat', 'sat'): 2
    })
    lower_order_counts = defaultdict(int, {
        ('the',): 6,
        ('cat',): 3
    })

    segment=list._process_segment(label=label, word_iterable=word)
    smoothing=list._witten_bell_smoothing(order_counts=order_counts, lower_order_counts=lower_order_counts)
    
    printer.pretty("GloVe", list._load_glove_vectors(), "success")
    printer.pretty("segment", segment, "success")
    printer.pretty("Smoothing", smoothing, "success")

    print("\n* * * * * Phase 3 Patterns * * * * *\n")
    printer.status("entity", engine.get_entities(), "success")
    printer.status("intent", engine.get_intents(), "success")
    printer.status("lexicon", engine.get_lexicons(), "success")
    printer.status("modality", engine.get_modalities(), "success")
    printer.status("morphology", engine.get_morphologies(), "success")

    print("\n* * * * * Phase 4 * * * * *\n")
    entity={"2024-11-10"}

    printer.pretty("temporal", engine._validate_temporal(entity=entity), "success")
    printer.pretty("quantity", engine._validate_quantity(entity=entity), "success")
    printer.pretty("technical", engine._validate_technical(entity=entity), "success")
    printer.pretty("duration", engine._validate_duration(entity=entity), "success")
    printer.pretty("term", engine._validate_term(entity=entity), "success")
    printer.pretty("boolean", engine._validate_boolean(entity=entity), "success")
    printer.pretty("norm", engine._normalize_entity(entity=entity), "success")

    print("\n* * * * * Phase 5 * * * * *\n")
    text="There aren't any resources where we are going, so get packing friend."

    printer.pretty("validate", list._validate_word_entries(), "success")
    printer.pretty("precompute", list._precompute_linguistic_data(), "success")


    print("\n=== Successfully Ran NLU Engine ===\n")
