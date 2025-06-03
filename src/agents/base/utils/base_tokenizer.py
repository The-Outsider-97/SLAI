import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional

from src.agents.base.utils.config_loader import load_global_config, get_config_section
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Light Metric Store")
printer = PrettyPrinter

class BaseTokenizer:
    """Core tokenizer class with common tokenization operations"""
    def __init__(self, special_tokens: List[str] = None):
        slef.config = load_global_config()
        self.vocab_size = self.config.get('src_vocab_size', '30000')

        self.bt_config = get_config_section('base_tokenizer')
        self.pad_token = self.bt_config.get('pad_token', '[PAD]')
        self.unk_token = self.bt_config.get('unk_token', '[UNK]')
        self.bos_token = self.bt_config.get('bos_token', '[BOS]')
        self.eos_token = self.bt_config.get('eos_token', '[EOS]')
        self.mask_token = self.bt_config.get('mask_token', '[MASK]')
        
        # Initialize with special tokens
        self.special_tokens = self.bt_config.get('special_tokens', or [])
        self.special_tokens.extend([pad_token, unk_token, bos_token, eos_token, mask_token])
        self.special_tokens = list(set(self.special_tokens))
        
        self.vocab = {}
        self.inverse_vocab = {}
        self.token_counts = Counter()
        self.is_trained = False

        printer.status("INIT", "Base Tokenizer succesfully initialized with: {self.vocab_size}", "success")

    def train(self, corpus: List[str], min_freq: int = 2, **kwargs) -> None:
        """Train tokenizer on text corpus (must be implemented in subclasses)"""
        raise NotImplementedError("Subclasses must implement training method")

    def tokenize(self, text: str, **kwargs) -> List[str]:
        """Convert text to tokens (must be implemented in subclasses)"""
        raise NotImplementedError("Subclasses must implement tokenize method")

    def encode(
        self,
        text: str,
        max_length: int = None,
        padding: str = "max_length",
        truncation: bool = True,
        add_special_tokens: bool = True,
        return_tensors: str = None,
        **kwargs
    ) -> Dict[str, Union[List[int], torch.Tensor]]:
        """Convert text to token IDs with processing options"""
        tokens = self.tokenize(text, **kwargs)
        
        # Apply truncation
        if truncation and max_length:
            tokens = tokens[:max_length - (2 if add_special_tokens else 0)]
        
        # Add special tokens
        if add_special_tokens:
            tokens = [self.bos_token] + tokens + [self.eos_token]
        
        # Convert to IDs
        token_ids = [self.token_to_id(token) for token in tokens]
        
        # Apply padding
        if padding == "max_length" and max_length:
            pad_length = max_length - len(token_ids)
            token_ids = token_ids + [self.token_to_id(self.pad_token)] * pad_length
        
        # Convert to tensors if requested
        if return_tensors == "pt":
            token_ids = torch.tensor([token_ids])
        
        return {"input_ids": token_ids}

    def decode(
        self,
        token_ids: Union[List[int], torch.Tensor],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True
    ) -> str:
        """Convert token IDs back to text"""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        tokens = [self.id_to_token(token_id) for token_id in token_ids]
        
        # Filter special tokens
        if skip_special_tokens:
            specials = {self.pad_token, self.unk_token, 
                        self.bos_token, self.eos_token, self.mask_token}
            tokens = [t for t in tokens if t not in specials]
        
        # Reconstruct text
        text = self.detokenize(tokens)
        
        # Clean spaces around punctuation
        if clean_up_tokenization_spaces:
            text = self.clean_text(text)
        
        return text

    def detokenize(self, tokens: List[str]) -> str:
        """Convert tokens back to text (basic implementation)"""
        return " ".join(tokens).replace(" ##", "").replace(" .", ".").strip()

    def clean_text(self, text: str) -> str:
        """Clean up spacing in text output"""
        replacements = [
            (" ' ", "'"),
            (" ,", ","),
            (" .", "."),
            (" ?", "?"),
            (" !", "!"),
            (" ;", ";"),
            (" :", ":"),
            (" ( ", " ("),
            (" ) ", ") ")
        ]
        for old, new in replacements:
            text = text.replace(old, new)
        return text

    def token_to_id(self, token: str) -> int:
        """Convert token to ID, return UNK for unknown tokens"""
        return self.vocab.get(token, self.vocab[self.unk_token])

    def id_to_token(self, token_id: int) -> str:
        """Convert ID to token, return UNK for unknown IDs"""
        return self.inverse_vocab.get(token_id, self.unk_token)

    def add_tokens(self, tokens: Union[str, List[str]]) -> int:
        """Add new tokens to vocabulary"""
        if isinstance(tokens, str):
            tokens = [tokens]
        
        new_count = 0
        for token in tokens:
            if token not in self.vocab:
                new_id = len(self.vocab)
                self.vocab[token] = new_id
                self.inverse_vocab[new_id] = token
                new_count += 1
        return new_count

    def get_vocab_size(self) -> int:
        """Get current vocabulary size"""
        return len(self.vocab)

    def get_vocab(self) -> Dict[str, int]:
        """Get full vocabulary dictionary"""
        return self.vocab.copy()

    def save(self, directory: Union[str, Path], name: str = "tokenizer") -> None:
        """Save tokenizer configuration and vocabulary"""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        config = {
            "vocab_size": self.vocab_size,
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "mask_token": self.mask_token,
            "special_tokens": self.special_tokens,
            "vocab": self.vocab,
            "token_counts": dict(self.token_counts),
            "is_trained": self.is_trained
        }
        with open(path / f"{name}_config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, directory: Union[str, Path], name: str = "tokenizer") -> "BaseTokenizer":
        """Load tokenizer from directory"""
        path = Path(directory)
        with open(path / f"{name}_config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
        
        # Create instance
        tokenizer = cls(
            vocab_size=config["vocab_size"],
            pad_token=config["pad_token"],
            unk_token=config["unk_token"],
            bos_token=config["bos_token"],
            eos_token=config["eos_token"],
            special_tokens=config["special_tokens"]
        )
        
        # Restore state
        tokenizer.vocab = config["vocab"]
        tokenizer.inverse_vocab = {int(k): v for k, v in config["vocab"].items()}
        tokenizer.token_counts = Counter(config.get("token_counts", {}))
        tokenizer.is_trained = config["is_trained"]
        return tokenizer

    def __call__(self, text: str, **kwargs) -> Dict[str, Union[List[int], torch.Tensor]]:
        """Alias for encode method for Hugging Face-style interface"""
        return self.encode(text, **kwargs)

    def __len__(self) -> int:
        """Get vocabulary size"""
        return len(self.vocab)
