
import unicodedata
import regex as re
import torch
import json

from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any

from src.agents.language.utils.config_loader import load_global_config, get_config_section
from src.agents.base.utils.base_tokenizer import BaseTokenizer
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Language Tokenizer")
printer = PrettyPrinter

class LanguageTokenizer(BaseTokenizer):
    """
    Byte Pair Encoding (BPE) Tokenizer that inherits from BaseTokenizer.
    It loads pre-trained BPE merges and vocabulary.
    """

    def __init__(self):
        super().__init__()
        self.config = load_global_config()
        self.bpe_model_file = self.config.get('bpe_model_path')
        self.bpe_vocab_file = self.config.get('bpe_vocab_path')

        self.lt_config = get_config_section('language_tokenizer')
        self.end_of_word_suffix = self.lt_config.get('end_of_word_suffix')
        self.normalization_rules = self.lt_config.get('normalization_rules', {
            "lowercase": True,
            "form": "NFKC"
        })
        
        self.merges: Dict[Tuple[str, str], str] = {} 
        self.ordered_merges: List[Tuple[str, str]] = []
        
        self.pre_tokenize_pattern = re.compile(
            rf"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{{L}}+| ?\p{{N}}+| ?[^\s\p{{L}}\p{{N}}]+|\s+(?!\S)|\s+""")

        # Load pre-trained BPE if files exist
        if self.bpe_model_file and Path(self.bpe_model_file).exists():
            self._load_bpe_model()
        else:
            logger.warning(f"BPE model file not found: {self.bpe_model_file}")
            
        if self.bpe_vocab_file and Path(self.bpe_vocab_file).exists():
            self._load_bpe_vocab()
        else:
            logger.warning(f"BPE vocab file not found: {self.bpe_vocab_file}")

        # Set trained status if both loaded successfully
        self.is_trained = bool(self.merges and self.vocab)
        if self.is_trained:
            printer.status("INIT", "Loaded pre-trained BPE model and vocab", "success")
        else:
            # Fallback to special tokens only
            self.vocab = {}
            self.inverse_vocab = {}
            for i, token in enumerate(self.special_tokens):
                self.vocab[token] = i
                self.inverse_vocab[i] = token
            printer.status("INIT", "Language Tokenizer initialized without BPE", "warning")
            
        printer.status("INIT", "Language Tokenizer (BPE) successfully initialized.", "success")

    def _load_bpe_model(self):
        """Load BPE merge operations and special tokens from JSON file"""
        try:
            with open(self.bpe_model_file, 'r', encoding='utf-8') as f:
                bpe_data = json.load(f)
            
            # Load merges
            self.ordered_merges = [tuple(pair) for pair in bpe_data.get('merges', [])]
            self.merges = {tuple(pair): ''.join(pair) for pair in bpe_data.get('merges', [])}
            
            # Update special tokens from BPE model
            bpe_specials = bpe_data.get('special_tokens', [])
            if bpe_specials:
                self.special_tokens = bpe_specials
                logger.info(f"Updated special tokens from BPE model: {bpe_specials}")
            
            # Update normalization rules if provided
            if 'normalization' in bpe_data:
                self.normalization_rules.update(bpe_data['normalization'])
                
        except Exception as e:
            logger.error(f"Error loading BPE model: {str(e)}")

    def _load_bpe_vocab(self):
        """Load vocabulary from JSON file"""
        try:
            with open(self.bpe_vocab_file, 'r', encoding='utf-8') as f:
                self.vocab = json.load(f)
            
            # Create inverse mapping
            self.inverse_vocab = {idx: token for token, idx in self.vocab.items()}
            
            # Ensure special tokens exist in vocab
            for token in self.special_tokens:
                if token not in self.vocab:
                    logger.warning(f"Special token '{token}' missing in BPE vocab")
                    
        except Exception as e:
            logger.error(f"Error loading BPE vocab: {str(e)}")

    def _normalize_text(self, text: str) -> str:
        """Applies configured normalization rules to the text."""
        return normalize_text_util(
            text,
            lowercase=self.normalization_rules.get("lowercase", True),
            normalization_form=self.normalization_rules.get("form")
        )

    def _pre_tokenize_words(self, normalized_text: str) -> List[str]:
        """Splits normalized text into word-like units for BPE using the regex pattern."""
        # Remove leading/trailing whitespace that might interfere with regex
        processed_text = normalized_text.strip()
        if not processed_text:
            return []
        # Find all matches; filter out pure whitespace matches if any slip through
        tokens = self.pre_tokenize_pattern.findall(processed_text)
        return [token.strip() for token in tokens if token and not token.isspace()]


    def train(self, corpus: List[str], min_freq: int = 2, **kwargs) -> None:
        """Trains the BPE tokenizer on a given corpus."""
        if not isinstance(corpus, list) or not all(isinstance(text, str) for text in corpus):
            raise ValueError("Corpus must be a list of strings.")
        
        force_retrain = kwargs.get('force_retrain', False)
        if self.is_trained and not force_retrain:
            logger.info("Tokenizer is already trained or has loaded merges. Skipping training. Use force_retrain=True to override.")
            return

        printer.status("TRAIN", "Starting BPE training...", "info")

        # 1. Initialize vocab with current special tokens (already done in __init__, but good to be explicit)
        self.vocab = {}
        self.inverse_vocab = {}
        current_id = 0
        for token in self.special_tokens:
            self.vocab[token] = current_id
            self.inverse_vocab[current_id] = token
            current_id += 1
        
        # 2. Normalization and Pre-tokenization to get word counts
        word_counts = Counter()
        for text in corpus:
            normalized_text = self._normalize_text(text)
            # Pre-tokenize for word counts for BPE (simple whitespace split for this stage)
            for word in normalized_text.split(): 
                if word: # Ensure non-empty words
                    word_counts[word] +=1
        
        self.token_counts = word_counts # Store original word counts

        # 3. Prepare words for BPE (split into chars + end_of_word_suffix) and build initial char vocab
        alphabet = set()
        bpe_corpus_word_freqs = defaultdict(int)

        for word, freq in word_counts.items():
            word_chars = list(word) + [self.end_of_word_suffix]
            bpe_corpus_word_freqs[tuple(word_chars)] += freq
            for char_token in word_chars:
                alphabet.add(char_token)
        
        for char_token in sorted(list(alphabet)):
            if char_token not in self.vocab: # Add if not already a special token
                self.vocab[char_token] = current_id
                self.inverse_vocab[current_id] = char_token
                current_id += 1
        
        # 4. BPE Merging Loop
        self.merges = {}
        self.ordered_merges = []
        # vocab_size is from BaseTokenizer's config (e.g., 30000 or 5000)
        num_merges_to_perform = self.vocab_size - len(self.vocab) 
        printer.status("TRAIN", f"Initial vocab size (specials + chars): {len(self.vocab)}. Target BPE merges: {num_merges_to_perform}", "info")

        if num_merges_to_perform <= 0:
            printer.warning(f"Target vocab size ({self.vocab_size}) is too small for new merges given current vocab. Training might not add merges.")
        
        for i in range(num_merges_to_perform):
            pair_stats = get_pair_stats(bpe_corpus_word_freqs)
            if not pair_stats:
                printer.status("TRAIN", "No more pairs to merge.", "warning")
                break
            
            best_pair = max(pair_stats, key=pair_stats.get)
            
            if pair_stats[best_pair] < min_freq:
                printer.status("TRAIN", f"Best pair frequency ({pair_stats[best_pair]}) < min_freq ({min_freq}). Stopping merge.", "info")
                break

            new_token = "".join(best_pair)
            self.ordered_merges.append(best_pair)
            self.merges[best_pair] = new_token

            bpe_corpus_word_freqs = merge_pair(bpe_corpus_word_freqs, best_pair, new_token)
            
            if new_token not in self.vocab:
                self.vocab[new_token] = current_id
                self.inverse_vocab[current_id] = new_token
                current_id += 1
            
            if (i + 1) % 100 == 0 or i == num_merges_to_perform -1 :
                printer.status("TRAIN", f"Merge {i+1}/{num_merges_to_perform}: Merged {best_pair} -> {new_token} (freq: {pair_stats[best_pair]})", "progress")
        
        self.is_trained = True
        printer.status("TRAIN", f"BPE training complete. Final vocab size: {len(self.vocab)}. Total merges learned: {len(self.ordered_merges)}", "success")

    def _apply_bpe_to_word_chars(self, word_chars: List[str]) -> List[str]:
        """Applies learned BPE merges to a single pre-tokenized word (list of chars)."""
        if not self.merges: # No merges to apply if not trained or loaded
            return word_chars

        tokens = list(word_chars)
        while True:
            best_pair_to_merge_in_word = None
            # Find the highest priority (earliest learned) merge applicable
            # This requires iterating through ordered_merges and checking applicability.
            # A common performance optimization is to precompute merge ranks.
            
            # Simpler greedy approach: find any applicable merge.
            # To better respect merge order from training, one would iterate self.ordered_merges.
            # For now, let's use a greedy approach on current pairs.
            
            # Find the best merge *currently possible* in the `tokens` list
            # This means finding the pair in `tokens` that corresponds to the *earliest* merge in `self.ordered_merges`
            
            current_best_merge_rank = float('inf')
            pair_to_apply_this_iteration = None

            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i+1])
                if pair in self.merges:
                    try:
                        # This assumes self.ordered_merges contains tuples of (tok1, tok2)
                        rank = self.ordered_merges.index(pair)
                        if rank < current_best_merge_rank:
                            current_best_merge_rank = rank
                            pair_to_apply_this_iteration = pair
                    except ValueError:
                        # This pair was in self.merges but not self.ordered_merges (should not happen with current logic)
                        continue 
            
            if pair_to_apply_this_iteration is None:
                break # No known merges can be applied to the current sequence of tokens

            # Perform the highest-priority merge found
            merged_token = self.merges[pair_to_apply_this_iteration]
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == pair_to_apply_this_iteration:
                    new_tokens.append(merged_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
            
        return tokens

    def tokenize(self, text: str, **kwargs) -> List[str]:
        """Converts text to a list of BPE subword tokens."""
        if not self.is_trained and not self.merges: # Not trained and no merges loaded
            logger.warning("Tokenizer has no BPE merges. Falling back to character tokenization after normalization.")
            normalized_text = self._normalize_text(text)
            return list(normalized_text) # Simple character split as a last resort

        normalized_text = self._normalize_text(text)
        pre_tokenized_words = self._pre_tokenize_words(normalized_text)
        
        final_bpe_tokens = []
        for word_segment in pre_tokenized_words:
            if not word_segment: continue

            # Add end_of_word_suffix for BPE processing, unless it's purely punctuation
            # or if the segment itself is already a special token.
            if word_segment in self.special_tokens:
                final_bpe_tokens.append(word_segment)
                continue

            # A simple check if it's mostly non-alphanumeric (likely punctuation)
            if not re.search(r'\p{L}|\p{N}', word_segment): # No letters or numbers
                # If it's a known token (e.g. punctuation already in vocab), use it
                if word_segment in self.vocab:
                    final_bpe_tokens.append(word_segment)
                else: # Otherwise, split punctuation into characters if not in vocab
                    for char_punct in list(word_segment):
                        final_bpe_tokens.append(self.id_to_token(self.token_to_id(char_punct)))
                continue

            # For actual words, process with BPE
            word_chars = list(word_segment) + [self.end_of_word_suffix]
            
            bpe_subwords = self._apply_bpe_to_word_chars(word_chars)
            
            # Validate and handle any resulting subwords not in vocab (should be rare if trained well)
            for subword in bpe_subwords:
                if subword in self.vocab:
                    final_bpe_tokens.append(subword)
                else:
                    # Fallback for subwords not in vocab after BPE (e.g. rare char sequences from unknown words)
                    # This often means splitting the unknown subword into known characters or UNK
                    logger.debug(f"Subword '{subword}' not in vocab after BPE. Splitting into chars or using UNK.")
                    decomposed_subword = []
                    temp_chars = list(subword)
                    while temp_chars:
                        # Try to find longest known prefix
                        found_known_prefix = False
                        for k in range(len(temp_chars), 0, -1):
                            prefix_candidate = "".join(temp_chars[:k])
                            if prefix_candidate in self.vocab:
                                decomposed_subword.append(prefix_candidate)
                                temp_chars = temp_chars[k:]
                                found_known_prefix = True
                                break
                        if not found_known_prefix: # Single char not in vocab (very rare)
                            decomposed_subword.append(self.unk_token)
                            logger.warning(f"Character '{temp_chars[0]}' from unknown subword '{subword}' also not in vocab.")
                            temp_chars = temp_chars[1:]
                    final_bpe_tokens.extend(decomposed_subword)

        return final_bpe_tokens

    def detokenize(self, tokens: List[str]) -> str:
        """Converts a list of BPE tokens back to a string, handling end-of-word suffix."""
        if not tokens:
            return ""
        
        # Join tokens first
        reconstructed_text = "".join(tokens)
        
        # Replace the end_of_word_suffix. Usually, it signifies a space should follow,
        # unless it's the very end of the string or followed by punctuation.
        reconstructed_text = reconstructed_text.replace(self.end_of_word_suffix, " ")
        
        # Clean up potential double spaces and leading/trailing spaces
        # The clean_text method in BaseTokenizer will handle further refinements.
        text = re.sub(r"\s+", " ", reconstructed_text).strip()
        return text

    def save(self, directory: Union[str, Path], name: str = "language_tokenizer") -> List[str]:
        """Saves BPE tokenizer configuration, vocabulary, and merges."""
        # First, get the dictionary from the parent's save logic (which saves vocab etc.)
        # Temporarily save to a buffer to get the dict, then augment it.
        
        # Create the config dict that BaseTokenizer's save would create
        base_config_data = {
            "tokenizer_class": self.__class__.__name__,
            "src_vocab_size": self.vocab_size,
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "mask_token": self.mask_token,
            "special_tokens": self.special_tokens,
            "vocab": self.vocab,
            "token_counts": dict(self.token_counts) if self.token_counts else {},
            "is_trained": self.is_trained
        }
        
        # Add BPE-specific parts
        base_config_data["ordered_merges"] = self.ordered_merges
        base_config_data["normalization_rules"] = self.normalization_rules
        base_config_data["end_of_word_suffix"] = self.end_of_word_suffix
        
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        config_file_path = path / f"{name}_config.json" # Consistent naming with BaseTokenizer.load
        
        with open(config_file_path, "w", encoding="utf-8") as f:
            json.dump(base_config_data, f, ensure_ascii=False, indent=2)
        
        printer.status("SAVE", f"LanguageTokenizer (BPE) full state saved to {config_file_path}", "success")
        return [str(config_file_path)]

    @classmethod
    def load(cls, directory: Union[str, Path], name: str = "language_tokenizer") -> "LanguageTokenizer":
        """Loads a LanguageTokenizer (BPE) from a saved configuration file."""
        path = Path(directory)
        config_file_path = path / f"{name}_config.json"

        if not config_file_path.exists():
            raise FileNotFoundError(f"Tokenizer configuration file not found: {config_file_path}")

        with open(config_file_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)
        
        tokenizer_class_name = config_data.get("tokenizer_class")
        if tokenizer_class_name != cls.__name__:
            raise ValueError(
                f"Attempting to load tokenizer of type '{tokenizer_class_name}' "
                f"using '{cls.__name__}.load(). Mismatch."
            )

        # Instantiate (LanguageTokenizer's __init__ will handle BPE defaults/lt_config)
        tokenizer = cls() 
        
        # Populate BaseTokenizer attributes
        tokenizer.vocab_size = int(config_data.get("src_vocab_size", 30000)) # Ensure int
        tokenizer.pad_token = config_data.get("pad_token", "[PAD]")
        tokenizer.unk_token = config_data.get("unk_token", "[UNK]")
        tokenizer.bos_token = config_data.get("bos_token", "[BOS]")
        tokenizer.eos_token = config_data.get("eos_token", "[EOS]")
        tokenizer.mask_token = config_data.get("mask_token", "[MASK]")
        
        # Special tokens from the saved config are definitive
        tokenizer.special_tokens = config_data.get("special_tokens", [])
        
        # Vocab is primary source for token-to-ID
        tokenizer.vocab = config_data.get("vocab", {})
        if not tokenizer.vocab:
             printer.warning("Loaded tokenizer has an empty vocabulary.")
        tokenizer.inverse_vocab = {v: k for k, v in tokenizer.vocab.items()}
        
        tokenizer.token_counts = Counter(config_data.get("token_counts", {}))
        tokenizer.is_trained = config_data.get("is_trained", False)

        # Populate BPE-specific attributes
        tokenizer.ordered_merges = [tuple(pair) for pair in config_data.get("ordered_merges", [])]
        tokenizer.merges = {tuple(p): p[0] + p[1] for p in tokenizer.ordered_merges}
        tokenizer.normalization_rules = config_data.get("normalization_rules", {"lowercase": True, "form": "NFKC"})
        tokenizer.end_of_word_suffix = config_data.get("end_of_word_suffix", "</w>")
        
        # Ensure that if merges are loaded, is_trained is also true if vocab is present
        if tokenizer.ordered_merges and tokenizer.vocab and not tokenizer.is_trained:
            logger.info("Merges and vocab loaded, setting is_trained to True.")
            tokenizer.is_trained = True
        elif not tokenizer.ordered_merges and tokenizer.is_trained:
            logger.warning("Tokenizer marked as trained but no BPE merges loaded.")


        printer.status("LOAD", f"LanguageTokenizer (BPE) loaded from {config_file_path}", "success")
        return tokenizer

# Helper functions for BPE training (can be outside the class or static methods)
def get_pair_stats(word_freqs: Dict[Tuple[str, ...], int]) -> Counter:
    """Counts occurrences of adjacent pairs of symbols in the corpus words."""
    counts = Counter()
    for word_tuple, freq in word_freqs.items():
        for i in range(len(word_tuple) - 1):
            counts[(word_tuple[i], word_tuple[i+1])] += freq
    return counts

def merge_pair(word_freqs: Dict[Tuple[str, ...], int], pair_to_merge: Tuple[str, str], new_token: str) -> Dict[Tuple[str, ...], int]:
    """Merges a specific pair of symbols in all words in the corpus representation."""
    new_word_freqs = defaultdict(int)
    for word_tuple, freq in word_freqs.items():
        new_word_list = []
        i = 0
        while i < len(word_tuple):
            if i < len(word_tuple) - 1 and (word_tuple[i], word_tuple[i+1]) == pair_to_merge:
                new_word_list.append(new_token)
                i += 2
            else:
                new_word_list.append(word_tuple[i])
                i += 1
        new_word_freqs[tuple(new_word_list)] += freq
    return new_word_freqs

def normalize_text_util(text: str, lowercase: bool = True, normalization_form: Optional[str] = "NFKC") -> str:
    """Utility function for text normalization."""
    if lowercase:
        text = text.lower()
    if normalization_form:
        # Ensure text is a string before normalizing
        if not isinstance(text, str):
            text = str(text)
        text = unicodedata.normalize(normalization_form, text)
    return text


if __name__ == "__main__":
    print("\n=== Running Language Tokenizer ===\n")
    printer.status("Init", "Language Tokenizer initialized", "success")

    tokenizer = LanguageTokenizer()
    print(f"Suggestions: {tokenizer}")

    print("\n* * * * * Phase 2 Encode/Decode * * * * *\n")
    text="This would be the Greatest undertaking of OUR generation!"
    text2="baby, this is not ok! i need you here, with me. please!"

    printer.pretty("Text", tokenizer._normalize_text(text=text), "success")
    printer.pretty("Pre-Text", tokenizer._pre_tokenize_words(normalized_text=text2), "success")
    printer.pretty("Tokenizer", tokenizer.tokenize(text=text2), "success")
    print("\n* * * * * Phase 2 Encode/Decode * * * * *\n")

    print("\n=== Successfully Ran Language Tokenizer ===\n")
