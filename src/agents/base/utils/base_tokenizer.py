import json
import torch
import unicodedata
import regex as re

from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional

from src.agents.base.utils.config_loader import load_global_config, get_config_section
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Base Tokenizer")
printer = PrettyPrinter


class BaseTokenizer:
    """Core tokenizer class with common tokenization operations."""
    def __init__(self):
        self.config = load_global_config()
        self.bt_config = get_config_section('base_tokenizer')

        # Read settings from base_tokenizer section
        self.src_vocab_size = self.bt_config.get('src_vocab_size', 30000)
        self.bpe_model_path = self.bt_config.get('bpe_model_path', '')
        self.bpe_vocab_path = self.bt_config.get('bpe_vocab_path', '')
        # Convert to Path objects
        if self.bpe_model_path:
            self.bpe_model_path = Path(self.bpe_model_path)
        if self.bpe_vocab_path:
            self.bpe_vocab_path = Path(self.bpe_vocab_path)

        self.pad_token = self.bt_config.get('pad_token', '[PAD]')
        self.unk_token = self.bt_config.get('unk_token', '[UNK]')
        self.bos_token = self.bt_config.get('bos_token', '[BOS]')
        self.eos_token = self.bt_config.get('eos_token', '[EOS]')
        self.mask_token = self.bt_config.get('mask_token', '[MASK]')

        # Special tokens – ensure it's a list
        special_tokens_raw = self.bt_config.get('special_tokens', [])
        if special_tokens_raw is None:
            special_tokens_raw = []
        if isinstance(special_tokens_raw, str):
            # If it's a string, split by comma or treat as single token
            special_tokens_raw = [t.strip() for t in special_tokens_raw.split(',') if t.strip()] \
                if ',' in special_tokens_raw else [special_tokens_raw]
        self.special_tokens = special_tokens_raw

        # Add core special tokens if not already present
        core_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token, self.mask_token]
        for token in core_tokens:
            if token not in self.special_tokens:
                self.special_tokens.append(token)

        # Vocabulary
        self.vocab: Dict[str, int] = {}
        self.inverse_vocab: Dict[int, str] = {}
        # Initialize vocab with special tokens
        for i, token in enumerate(self.special_tokens):
            self.vocab[token] = i
            self.inverse_vocab[i] = token

        self.token_counts = Counter()
        self.is_trained = False

        # Load encode/decode defaults from config
        self.encode_config = get_config_section('encode')
        self.decode_config = get_config_section('decode')

        printer.status("INIT", f"Base Tokenizer initialized with vocab size: {len(self.vocab)}", "success")

    def _get_config_value(self, section, key, default, converter=None):
        """Safely get a config value, handling None and string 'None'."""
        value = section.get(key, default)
        if value is None or (isinstance(value, str) and value.lower() == 'none'):
            return default
        if converter is not None:
            try:
                return converter(value)
            except (ValueError, TypeError):
                return default
        return value

    def train(self, corpus: List[str], min_freq: int = 2, **kwargs) -> None:
        """Train the tokenizer on a list of text documents."""
        logger.info("Starting tokenizer training...")

        # Reset vocab except special tokens
        self.vocab = {token: i for i, token in enumerate(self.special_tokens)}
        self.inverse_vocab = {i: token for token, i in self.vocab.items()}
        self.token_counts = Counter()
        next_id = len(self.special_tokens)

        # Count tokens in corpus
        for line in corpus:
            tokens = self.tokenize(line)
            self.token_counts.update(tokens)

        # Add tokens that meet min_freq
        for token, freq in self.token_counts.items():
            if freq >= min_freq and token not in self.vocab:
                self.vocab[token] = next_id
                self.inverse_vocab[next_id] = token
                next_id += 1

        self.is_trained = True
        self.src_vocab_size = len(self.vocab)
        printer.status("TRAIN", f"BaseTokenizer trained with vocab size: {self.src_vocab_size}", "success")

    def tokenize(self, text: str, **kwargs) -> List[str]:
        """
        Tokenize a string into words, punctuation, emojis, and special symbols.
        Returns a flat list of string tokens.
        """
        # Normalize Unicode
        text = unicodedata.normalize("NFKC", text.lower())

        # Remove control characters and normalize whitespace
        text = re.sub(r"[\u0000-\u001F\u007F]", "", text)
        text = re.sub(r"\s+", " ", text)

        # Separate emojis, punctuation, and currency symbols
        text = re.sub(r"([\p{Emoji}\p{Punctuation}\p{Sc}])", r" \1 ", text)

        # Keep numbers and decimals together (e.g., 3.14 not split)
        text = re.sub(r"(\d)\s*\.\s*(\d)", r"\1.\2", text)

        # Handle contractions (English-specific)
        text = re.sub(r"\b(can|won|don|doesn|didn|shouldn|couldn|wouldn|hasn|haven|isn|aren|ain|wasn|weren)'t\b", r"\1't", text)
        text = re.sub(r"\b(i|you|we|they|he|she|it)'ll\b", r"\1'll", text)
        text = re.sub(r"\b(i|you|we|they|he|she|it)'ve\b", r"\1've", text)
        text = re.sub(r"\b(i|you|we|they|he|she|it)'d\b", r"\1'd", text)
        text = re.sub(r"\b(i|you|we|they|he|she|it)'m\b", r"\1'm", text)
        text = re.sub(r"\b(\w+)'s\b", r"\1's", text)

        # Split into tokens: alphanumerics, numbers, emoji, currency, punctuation
        tokens = re.findall(r"\p{L}+|\p{N}+|[\p{Emoji}\p{Sc}\p{Punctuation}]", text)
        return tokens

    def encode(self, text: str, **kwargs) -> Dict[str, Union[List[int], torch.Tensor]]:
        """Convert text to token IDs with processing options."""
        # Read encoding parameters from config, allowing overrides via kwargs
        max_length = kwargs.get('max_length', self._get_config_value(self.encode_config, 'max_length', None))
        if max_length is None:
            max_length = 128
        else:
            max_length = int(max_length)

        padding = kwargs.get('padding', self._get_config_value(self.encode_config, 'padding', 'max_length'))
        truncation = kwargs.get('truncation', self._get_config_value(self.encode_config, 'truncation', True))
        add_special_tokens = kwargs.get('add_special_tokens', self._get_config_value(self.encode_config, 'add_special_tokens', True))
        return_tensors = kwargs.get('return_tensors', self._get_config_value(self.encode_config, 'return_tensors', None))

        tokens = self.tokenize(text, **kwargs)

        # Truncation
        if truncation and max_length:
            limit = max_length - (2 if add_special_tokens else 0)
            tokens = tokens[:limit]

        # Add special tokens
        if add_special_tokens:
            tokens = [self.bos_token] + tokens + [self.eos_token]

        # Convert to IDs
        token_ids = [self.token_to_id(token) for token in tokens]

        # Padding
        if padding == "max_length" and max_length:
            pad_id = self.token_to_id(self.pad_token)
            token_ids = token_ids + [pad_id] * (max_length - len(token_ids))

        # Convert to tensor if requested
        if return_tensors == "pt":
            token_ids = torch.tensor(token_ids, dtype=torch.long)

        return {"input_ids": token_ids}

    def decode(self, token_ids: Union[List[int], torch.Tensor]) -> str:
        """Convert token IDs back to text."""
        skip_special = self._get_config_value(self.decode_config, 'skip_special_tokens', True)
        clean_spaces = self._get_config_value(self.decode_config, 'clean_up_tokenization_spaces', True)

        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        tokens = [self.id_to_token(tid) for tid in token_ids]

        if skip_special:
            specials = {self.pad_token, self.unk_token, self.bos_token, self.eos_token, self.mask_token}
            tokens = [t for t in tokens if t not in specials]

        # Reconstruct text
        text = self.detokenize(tokens)

        if clean_spaces:
            text = self.clean_text(text)

        return text

    def detokenize(self, tokens: List[str]) -> str:
        """
        Convert a list of tokens back into a text string.
        Handles punctuation, contractions, emojis, and number formatting.
        """
        if not tokens:
            return ""

        text = ""
        prev_token = ""

        for token in tokens:
            if token in {".", ",", "!", "?", ";", ":", "%", ")", "]", "}", "’", "'s"}:
                text += token
            elif token in {"(", "[", "{", "‘", "“", "$", "£", "€", "¥", "₹"}:
                text += " " + token
            elif token in {"-", "/", "–", "—"}:
                if prev_token and prev_token.isdigit():
                    text += token
                else:
                    text += " " + token
            elif token in {"'", '"'}:
                if prev_token and prev_token[-1].isalnum():
                    text += token
                else:
                    text += " " + token
            elif re.match(r"[\p{Emoji}\p{Sc}]", token):
                text += " " + token
            elif prev_token in {"$", "€", "£", "₹"} and token.isdigit():
                text += token
            else:
                text += " " + token
            prev_token = token

        return text.strip()

    def clean_text(self, text: str) -> str:
        """Clean up spacing in text output."""
        text = unicodedata.normalize("NFKC", text)

        # Remove BPE artifacts
        text = text.replace('</w> ', ' ').replace('</w>', '')
        text = re.sub(r"\s+", " ", text)

        # Remove space before punctuation
        text = re.sub(r"\s+([.,!?;:])", r"\1", text)

        # Fix contractions
        text = re.sub(r"(\S)\s+'\s*(\S)", r"\1'\2", text)

        # Fix spacing in parentheses and quotes
        text = re.sub(r"\(\s+", "(", text)
        text = re.sub(r"\s+\)", ")", text)
        text = re.sub(r"\[\s+", "[", text)
        text = re.sub(r"\s+\]", "]", text)
        text = re.sub(r'\s+"', '"', text)
        text = re.sub(r'"\s+', '"', text)
        text = re.sub(r"\s+'", "'", text)
        text = re.sub(r"'\s+", "'", text)

        # Fix spacing around dashes and slashes
        text = re.sub(r"\s*-\s*", "-", text)
        text = re.sub(r"\s*/\s*", "/", text)

        # Remove space before currency symbols + digit
        text = re.sub(r" ([\$€£¥₹]\s*\d)", r"\1", text)

        # Fix number ranges (e.g., "3 - 5" → "3-5")
        text = re.sub(r"(\d)\s*-\s*(\d)", r"\1-\2", text)

        # Remove leading/trailing punctuation artifacts
        text = re.sub(r"^[^\w(]+", "", text)
        text = re.sub(r"[^\w)]+$", "", text)

        # Fix common tokenization artifacts
        text = re.sub(r" \.", ".", text)
        text = re.sub(r" ,", ",", text)
        text = re.sub(r" !", "!", text)
        text = re.sub(r" \?", "?", text)

        # Restore possessives and abbreviations
        text = re.sub(r"\b(\w+)\s+'\s*s\b", r"\1's", text)
        text = re.sub(r"\b(\w+)\s+'\s*\b", r"\1'", text)

        return text.strip()

    def token_to_id(self, token: str) -> int:
        """Convert token to ID, return UNK ID if token not found."""
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def id_to_token(self, token_id: int) -> str:
        """Convert ID to token, return UNK token if ID not found."""
        return self.inverse_vocab.get(token_id, self.unk_token)

    def add_tokens(self, tokens: Union[str, List[str]], special: bool = False) -> int:
        """Add new tokens to vocabulary. If special, add to special_tokens list."""
        if isinstance(tokens, str):
            tokens = [tokens]

        new_count = 0
        current_max_id = max(self.inverse_vocab.keys()) if self.inverse_vocab else -1

        for token in tokens:
            if token not in self.vocab:
                current_max_id += 1
                self.vocab[token] = current_max_id
                self.inverse_vocab[current_max_id] = token
                if special and token not in self.special_tokens:
                    self.special_tokens.append(token)
                new_count += 1
        return new_count

    def get_vocab_size(self) -> int:
        """Get current vocabulary size."""
        return len(self.vocab)

    def get_vocab(self) -> Dict[str, int]:
        """Get full vocabulary dictionary."""
        return self.vocab.copy()

    def save(self, directory: Union[str, Path], name: str = "tokenizer") -> List[str]:
        """Save tokenizer configuration and vocabulary. Returns list of saved file paths."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        config_to_save = {
            "tokenizer_class": self.__class__.__name__,
            "src_vocab_size": self.src_vocab_size,
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "mask_token": self.mask_token,
            "special_tokens": self.special_tokens,
            "vocab": self.vocab,
            "token_counts": dict(self.token_counts),
            "is_trained": self.is_trained,
        }
        # Subclasses may add more (e.g., BPE merges)

        config_file = path / f"{name}_config.json"
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config_to_save, f, ensure_ascii=False, indent=2)

        printer.status("SAVE", f"BaseTokenizer config saved to {config_file}", "success")
        return [str(config_file)]

    @classmethod
    def load(cls, directory: Union[str, Path], name: str = "tokenizer") -> "BaseTokenizer":
        """Load tokenizer from directory."""
        path = Path(directory)
        config_file = path / f"{name}_config.json"
        if not config_file.exists():
            raise FileNotFoundError(f"Tokenizer config file not found: {config_file}")

        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)

        # Create instance (could be subclass)
        tokenizer_class = config.get("tokenizer_class", cls.__name__)
        # For simplicity, instantiate the class that called load
        tokenizer = cls()

        # Restore state
        tokenizer.src_vocab_size = config.get("src_vocab_size", 30000)
        tokenizer.pad_token = config.get("pad_token", "[PAD]")
        tokenizer.unk_token = config.get("unk_token", "[UNK]")
        tokenizer.bos_token = config.get("bos_token", "[BOS]")
        tokenizer.eos_token = config.get("eos_token", "[EOS]")
        tokenizer.mask_token = config.get("mask_token", "[MASK]")
        tokenizer.special_tokens = config.get("special_tokens", [])
        tokenizer.vocab = config["vocab"]
        tokenizer.inverse_vocab = {v: k for k, v in tokenizer.vocab.items()}
        tokenizer.token_counts = Counter(config.get("token_counts", {}))
        tokenizer.is_trained = config.get("is_trained", False)

        printer.status("LOAD", f"BaseTokenizer loaded from {config_file}", "success")
        return tokenizer

    def __call__(self, text: str, **kwargs) -> Dict[str, Union[List[int], torch.Tensor]]:
        """Alias for encode."""
        return self.encode(text, **kwargs)

    def __len__(self) -> int:
        return len(self.vocab)


# ----------------------------------------------------------------------
# Test block
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("\n=== Running Base Tokenizer ===\n")
    printer.status("Init", "Base Tokenizer initialized", "success")

    tokenizer = BaseTokenizer()
    print(f"Tokenizer: {tokenizer}")

    text = "Life a worth living with friends like you!"
    encoded = tokenizer.encode(text)
    printer.pretty("encode", encoded, "success")
    decoded = tokenizer.decode(encoded['input_ids'])
    printer.pretty("decode", decoded, "success")
    cleaned = tokenizer.clean_text(text)
    printer.pretty("clean_text", cleaned, "success")

    print("\n=== Successfully Ran Base Tokenizer ===\n")
