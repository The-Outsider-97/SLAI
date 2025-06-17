import json
import torch
import unicodedata
import regex as re

from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional

from src.agents.base.utils.config_loader import load_global_config, get_config_section
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Base Tokenizer")
printer = PrettyPrinter

class BaseTokenizer:
    """Core tokenizer class with common tokenization operations"""
    def __init__(self):
        self.config = load_global_config()
        self.vocab_size = self.config.get('src_vocab_size')
        self.bpe_model_path = self.config.get('bpe_model_path')
        self.bpe_vocab_path = self.config.get('bpe_vocab_path')

        self.bt_config = get_config_section('base_tokenizer')
        self.pad_token = self.bt_config.get('pad_token', '[PAD]')
        self.unk_token = self.bt_config.get('unk_token', '[UNK]')
        self.bos_token = self.bt_config.get('bos_token', '[BOS]')
        self.eos_token = self.bt_config.get('eos_token', '[EOS]')
        self.mask_token = self.bt_config.get('mask_token', '[MASK]')
        
        # Initialize with special tokens
        self.special_tokens = self.bt_config.get('special_tokens', [])
        if self.special_tokens is None or isinstance(self.special_tokens, str):
            self.special_tokens = []

        core_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token, self.mask_token]
        for token in core_tokens:
            if token not in self.special_tokens:
                self.special_tokens.append(token)

        self.special_tokens.extend([self.pad_token, self.unk_token, self.bos_token, self.eos_token, self.mask_token])
        self.special_tokens = list(set(self.special_tokens))
        
        self.vocab = {}
        self.inverse_vocab = {}
        # Initialize vocab with special tokens to ensure they have low, consistent IDs
        for i, token in enumerate(self.special_tokens):
            self.vocab[token] = i
            self.inverse_vocab[i] = token
        self.token_counts = Counter()
        self.is_trained = False

        # Load encode/decode defaults once
        self.encode_config = get_config_section('encode')
        self.decode_config = get_config_section('decode')

        printer.status("INIT", f"Base Tokenizer successfully initialized with vocab size: {self.vocab_size}", "success")

    def train(self, corpus: List[str], min_freq: int = 2, **kwargs) -> None:
        """
        Train the tokenizer on a list of text documents.
    
        Args:
            corpus: A list of strings (text samples).
            min_freq: Minimum frequency for a token to be included in the vocabulary.
        """
        logger.info("Starting tokenizer training...")
        
        # Reset vocab except special tokens
        self.vocab = {}
        self.inverse_vocab = {}
        self.token_counts = Counter()
        
        # Reserve IDs for special tokens
        for i, token in enumerate(self.special_tokens):
            self.vocab[token] = i
            self.inverse_vocab[i] = token
    
        next_id = len(self.special_tokens)
        
        # Count tokens in the entire corpus
        for line in corpus:
            tokens = self.tokenize(line)
            self.token_counts.update(tokens)
        
        # Add tokens that meet min_freq and are not already special
        for token, freq in self.token_counts.items():
            if freq >= min_freq and token not in self.vocab:
                self.vocab[token] = next_id
                self.inverse_vocab[next_id] = token
                next_id += 1
        
        self.is_trained = True
        self.vocab_size = len(self.vocab)
        printer.status("TRAIN", f"BaseTokenizer trained with vocab size: {self.vocab_size}", "success")

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
    
        # Separate out emojis, punctuation, and currency symbols
        text = re.sub(r"([\p{Emoji}\p{Punctuation}\p{Sc}])", r" \1 ", text)
    
        # Keep numbers and decimals together (e.g., 3.14 not split into 3 . 14)
        text = re.sub(r"(\d)\s*\.\s*(\d)", r"\1.\2", text)
    
        # Handle contractions (English-specific)
        text = re.sub(r"\b(can|won|don|doesn|didn|shouldn|couldn|wouldn|hasn|haven|isn|aren|ain|wasn|weren)'t\b", r"\1't", text)
        text = re.sub(r"\b(i|you|we|they|he|she|it)'ll\b", r"\1'll", text)
        text = re.sub(r"\b(i|you|we|they|he|she|it)'ve\b", r"\1've", text)
        text = re.sub(r"\b(i|you|we|they|he|she|it)'d\b", r"\1'd", text)
        text = re.sub(r"\b(i|you|we|they|he|she|it)'m\b", r"\1'm", text)
        text = re.sub(r"\b(\w+)'s\b", r"\1's", text)
    
        # Split into tokens: keep alphanumerics, emoji, currency, punctuation as separate tokens
        tokens = re.findall(r"\p{L}+|\p{N}+|[\p{Emoji}\p{Sc}\p{Punctuation}]", text)
    
        return tokens

    def encode(self, text: str, **kwargs) -> Dict[str, Union[List[int], torch.Tensor]]:
        """Convert text to token IDs with processing options"""
        raw_max_length = self.encode_config.get('max_length', 128)
        self.max_length = int(raw_max_length) if isinstance(raw_max_length, (int, float, str)) and str(raw_max_length).isdigit() else 128
        self.padding = self.encode_config.get('padding', 'max_length')
        self.truncation = self.encode_config.get('truncation', True)
        self.add_special_tokens = self.encode_config.get('add_special_tokens', True)
        self.return_tensors = self.encode_config.get('return_tensors')

        tokens = self.tokenize(text, **kwargs)

        # Apply truncation
        if self.truncation and self.max_length:
            tokens = tokens[:self.max_length - (2 if self.add_special_tokens else 0)]
        
        # Add special tokens
        if self.add_special_tokens:
            tokens = [self.bos_token] + tokens + [self.eos_token]
        
        # Convert to IDs
        token_ids = [self.token_to_id(token) for token in tokens]
        
        # Apply padding
        if self.padding == "max_length" and self.max_length:
            pad_length = self.max_length - len(token_ids)
            token_ids = token_ids + [self.token_to_id(self.pad_token)] * pad_length
        
        # Convert to tensors if requested
        if self.return_tensors == "pt":
            token_ids = torch.tensor([token_ids])
        
        return {"input_ids": token_ids}

    def decode(self, token_ids: Union[List[int], torch.Tensor]) -> str:
        """Convert token IDs back to text"""
        self.skip_special_tokens = self.decode_config.get('skip_special_tokens')
        self.clean_up_tokenization_spaces = self.decode_config.get('clean_up_tokenization_spaces')

        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        tokens = [self.id_to_token(token_id) for token_id in token_ids]
        
        # Filter special tokens
        if self.skip_special_tokens:
            specials = {self.pad_token, self.unk_token, 
                        self.bos_token, self.eos_token, self.mask_token}
            tokens = [t for t in tokens if t not in specials]
        
        # Reconstruct text
        text = self.detokenize(tokens)
        
        # Clean spaces around punctuation
        if self.clean_up_tokenization_spaces:
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
                # No space before or after if between numbers/words
                if prev_token and prev_token.isdigit():
                    text += token
                else:
                    text += " " + token
            elif token in {"'", '"'}:
                # Heuristics for quotes
                if prev_token and prev_token[-1].isalnum():
                    text += token  # closing quote
                else:
                    text += " " + token  # opening quote
            elif re.match(r"[\p{Emoji}\p{Sc}]", token):
                text += " " + token  # Emoji or currency symbol
            elif prev_token in {"$", "€", "£", "₹"} and token.isdigit():
                text += token  # Avoid space in $100
            else:
                text += " " + token
            prev_token = token
    
        # Final cleaning
        text = text.strip()
        return text

    def clean_text(self, text: str) -> str:
        """Clean up spacing in text output (can be overridden)"""

        text = unicodedata.normalize("NFKC", text)    # Unicode normalization
    
        text = text.replace('</w> ', ' ').replace('</w>', '')    # Remove BPE artifacts
        text = re.sub(r"\s+", " ", text)    # Collapse multiple spaces
        text = re.sub(r"\s+([.,!?;:])", r"\1", text)    # Remove space before punctuation

        # Fix contractions
        text = re.sub(r"(\S)\s+'\s*(\S)", r"\1'\2", text)  # don't → don ' t

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

        # Fix common tokenization artifacts (e.g., " .", " ,")
        text = re.sub(r" \.", ".", text)
        text = re.sub(r" ,", ",", text)
        text = re.sub(r" !", "!", text)
        text = re.sub(r" \?", "?", text)

        # Restore possessives and abbreviations
        text = re.sub(r"\b(\w+)\s+'\s*s\b", r"\1's", text)  # John's
        text = re.sub(r"\b(\w+)\s+'\s*\b", r"\1'", text)    # can't →
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def token_to_id(self, token: str) -> int:
        """Convert token to ID, return UNK for unknown tokens"""
        return self.vocab.get(token, self.vocab.get(self.unk_token)) # Ensure unk_token is in vocab

    def id_to_token(self, token_id: int) -> str:
        """Convert ID to token, return UNK for unknown IDs"""
        return self.inverse_vocab.get(token_id, self.unk_token) # unk_token itself

    def add_tokens(self, tokens: Union[str, List[str]], special: bool = False) -> int:
        """Add new tokens to vocabulary. If special, add to special_tokens list."""
        if isinstance(tokens, str):
            tokens = [tokens]

        new_count = 0
        current_max_id = max(self.inverse_vocab.keys()) if self.inverse_vocab else -1
        
        for token in tokens:
            if token not in self.vocab:
                current_max_id += 1
                new_id = current_max_id
                self.vocab[token] = new_id
                self.inverse_vocab[new_id] = token
                if special and token not in self.special_tokens:
                    self.special_tokens.append(token)
                new_count += 1
        return new_count

    def get_vocab_size(self) -> int:
        """Get current vocabulary size"""
        return len(self.vocab)

    def get_vocab(self) -> Dict[str, int]:
        """Get full vocabulary dictionary"""
        return self.vocab.copy()

    def save(self, directory: Union[str, Path], name: str = "tokenizer") -> None:
        """Save tokenizer configuration and vocabulary. Returns list of saved file paths."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        config_to_save = {
            "tokenizer_class": self.__class__.__name__,
            "src_vocab_size": self.vocab_size, # Target size during training
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "mask_token": self.mask_token,
            "special_tokens": self.special_tokens, # Actual list of special tokens used
            "vocab": self.vocab, # The actual vocabulary (token -> ID)
            "token_counts": dict(self.token_counts) if self.token_counts else {},
            "is_trained": self.is_trained
        }
        # Subclasses might add more to this dictionary (like BPE merges)
        
        config_file_path = path / f"{name}_config.json"
        with open(config_file_path, "w", encoding="utf-8") as f:
            json.dump(config_to_save, f, ensure_ascii=False, indent=2)
        
        printer.status("SAVE", f"BaseTokenizer config saved to {config_file_path}", "success")
        return [str(config_file_path)]

    @classmethod
    def load(cls, directory: Union[str, Path], name: str = "tokenizer") -> "BaseTokenizer":
        """Load tokenizer from directory. Note: This is a base load. Subclasses might need to override or extend."""
        path = Path(directory)
        config_file_path = path / f"{name}_config.json"
        
        if not config_file_path.exists():
            raise FileNotFoundError(f"Tokenizer configuration file not found: {config_file_path}")
            
        with open(config_file_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        
        # Instantiate the correct class (could be BaseTokenizer or a subclass)
        tokenizer_class_name = config.get("tokenizer_class", cls.__name__)
        if tokenizer_class_name != cls.__name__:
             # This logic might need adjustment if using a central registry for tokenizers
             printer.warning(f"Loading tokenizer of type '{tokenizer_class_name}' with '{cls.__name__}.load'. Ensure compatibility.")

        tokenizer = cls() 
        
        # Restore state from loaded config
        tokenizer.vocab_size = config.get("src_vocab_size", 30000)
        tokenizer.pad_token = config.get("pad_token", "[PAD]")
        tokenizer.unk_token = config.get("unk_token", "[UNK]")
        tokenizer.bos_token = config.get("bos_token", "[BOS]")
        tokenizer.eos_token = config.get("eos_token", "[EOS]")
        tokenizer.mask_token = config.get("mask_token", "[MASK]")
        tokenizer.special_tokens = config.get("special_tokens", [])
        
        tokenizer.vocab = config["vocab"]
        tokenizer.inverse_vocab = {v: k for k, v in tokenizer.vocab.items()} # Rebuild inverse_vocab
        
        tokenizer.token_counts = Counter(config.get("token_counts", {}))
        tokenizer.is_trained = config.get("is_trained", False)
        
        printer.status("LOAD", f"BaseTokenizer loaded from {config_file_path}", "success")
        return tokenizer

    def __call__(self, text: str, **kwargs) -> Dict[str, Union[List[int], torch.Tensor]]:
        """Alias for encode method for Hugging Face-style interface"""
        return self.encode(text, **kwargs)

    def __len__(self) -> int:
        """Get vocabulary size"""
        return len(self.vocab)

if __name__ == "__main__":
    print("\n=== Running Base Tokenizer ===\n")
    printer.status("Init", "Base Tokenizer initialized", "success")

    tokenizer = BaseTokenizer()
    print(f"Suggestions: {tokenizer}")

    print("\n* * * * * Phase 2 Encode/Decode * * * * *\n")
    text="Life a worth living with frineds like you!"
    token_ids=[]

    printer.pretty("encode", tokenizer.encode(text=text), "success")
    printer.pretty("decode", tokenizer.decode(token_ids=token_ids), "success")
    printer.pretty("Text", tokenizer.clean_text(text=text), "success")

    print("\n* * * * * Phase 2 Encode/Decode * * * * *\n")

    print("\n=== Successfully Ran Base Tokenizer ===\n")
