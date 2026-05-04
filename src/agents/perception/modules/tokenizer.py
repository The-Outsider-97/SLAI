import os
import json
import re
import torch
import unicodedata
from pathlib import Path
from typing import List, Dict, Mapping, Union, Optional, Tuple

from ..utils.config_loader import load_global_config, get_config_section
from ..utils.common import TensorOps, Parameter
from ...base.modules.base_tokenizer import BaseTokenizer
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Tokenizer")
printer = PrettyPrinter


class BytePairEncoder:
    """Byte‑pair encoding (BPE) tokenizer."""
    def __init__(self, merges: List[Tuple[str, str]], word_to_id: Optional[Dict[str, int]] = None,
                 unk_token: str = '[UNK]'):
        self.bpe_ranks = {tuple(merge): i for i, merge in enumerate(merges)}
        self.cache = {}
        self.word_to_id = word_to_id or {}
        self.unk_token = unk_token

    @staticmethod
    def get_pairs(word: Tuple[str, ...]) -> set:
        """Return set of symbol pairs in a word."""
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs

    def bpe(self, token: str) -> List[str]:
        """Apply BPE to a single token."""
        if token in self.cache:
            return self.cache[token]

        if not token:
            return [self.unk_token]

        word = tuple(token) + ('</w>',)
        pairs = self.get_pairs(word)

        if not pairs:
            # No pairs possible (single character)
            if token in self.word_to_id:
                return [token]
            return [self.unk_token]

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break

            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                j = i
                # Find the next occurrence of the first part
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break

                new_word.extend(word[i:j])
                i = j

                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            word = tuple(new_word)
            pairs = self.get_pairs(word)

        # Remove the sentinel </w> if it's the last element
        if word and word[-1] == '</w>':
            word = word[:-1]

        result = list(word)
        self.cache[token] = result
        return result


class Tokenizer(BaseTokenizer):
    """
    Enhanced tokenizer with BPE and multi‑modal support.
    """
    def __init__(self):
        super().__init__()
        self.token_config = get_config_section('tokenizer')
        self.max_length = self.token_config.get('max_length', 512)
        self.cls_token = self.token_config.get('cls_token', '[CLS]')
        self.sep_token = self.token_config.get('sep_token', '[SEP]')
        self.image_token = self.token_config.get('image_token', '[IMG]')
        self.audio_token = self.token_config.get('audio_token', '[AUDIO]')

        # Add the extra special tokens if not already present
        self.add_tokens([self.cls_token, self.sep_token, self.image_token, self.audio_token], special=True)

        # Ensure BPE paths are Path objects
        if self.bpe_model_path and isinstance(self.bpe_model_path, str):
            self.bpe_model_path = Path(self.bpe_model_path)
        if self.bpe_vocab_path and isinstance(self.bpe_vocab_path, str):
            self.bpe_vocab_path = Path(self.bpe_vocab_path)

        # Set up BPE processor
        self._setup_bpe_processor()

        # Pre‑compile regex for tokenization (used by base tokenizer, but we keep for custom logic)
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?[^\s\w\d]+|\s+(?!\S)|\s+""")
        self.cache = {}  # BPE cache (already in BytePairEncoder, but we keep for legacy)

        logger.info(f"Tokenizer initialized with max_length={self.max_length}")

    def _setup_bpe_processor(self):
        """Load BPE merges and vocabulary."""
        if not self.bpe_vocab_path or not self.bpe_vocab_path.exists():
            logger.warning(f"BPE vocab file not found: {self.bpe_vocab_path}")
            return
        if not self.bpe_model_path or not self.bpe_model_path.exists():
            logger.warning(f"BPE model file not found: {self.bpe_model_path}")
            return

        # Load BPE vocabulary (word → id)
        with open(self.bpe_vocab_path, "r", encoding="utf-8") as f:
            bpe_word_to_id = json.load(f)

        # Add BPE words to main vocabulary if not already present
        for token in bpe_word_to_id:
            if token not in self.vocab:
                self.add_tokens([token])

        # Load BPE merges
        with open(self.bpe_model_path, "r", encoding="utf-8") as f:
            merges_data = json.load(f)
            bpe_merges = [tuple(pair) for pair in merges_data.get("merges", [])]
            logger.info(f"Loaded {len(bpe_merges)} BPE merges")

        self.bpe_ranks = {pair: i for i, pair in enumerate(bpe_merges)}
        self.bpe_processor = BytePairEncoder(bpe_merges, self.vocab, self.unk_token)

    def tokenize(self, text: str) -> List[str]: # pyright: ignore[reportIncompatibleMethodOverride]
        """
        Tokenize text using base tokenization + BPE.
        Overrides base to apply BPE subword segmentation.
        """
        # Normalize and tokenize using base method
        tokens = super().tokenize(text)

        # Apply BPE to each token
        if hasattr(self, 'bpe_processor'):
            subword_tokens = []
            for token in tokens:
                sub_pieces = self.bpe_processor.bpe(token)
                subword_tokens.extend(sub_pieces)
            return subword_tokens
        else:
            return tokens
        
    @BaseTokenizer.word_to_id.setter # pyright: ignore[reportAttributeAccessIssue]
    def word_to_id(self, mapping: Mapping[str, int]) -> None:
        # Call the parent setter to update self.vocab and self.inverse_vocab
        super(Tokenizer, type(self)).word_to_id.fset(self, mapping) # pyright: ignore[reportAttributeAccessIssue]
        # If BPE processor exists, update its word_to_id reference
        if hasattr(self, 'bpe_processor') and self.bpe_processor is not None:
            self.bpe_processor.word_to_id = self.vocab

    def encode_multi_modal(self, text: Optional[str] = None,
                           image_features: Optional[torch.Tensor] = None,
                           audio_features: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Encode multi‑modal input by concatenating text, image, and audio tokens.
        Image and audio are represented by special tokens; their features are not embedded here.
        Returns a dict with input_ids and attention_mask.
        """
        printer.status("TOKEN", "Encoding multi‑modal input", "info")

        all_tokens = []
        if text:
            text_tokens = self.tokenize(text)
            all_tokens.extend(text_tokens)
        if image_features is not None:
            all_tokens.append(self.image_token)
        if audio_features is not None:
            all_tokens.append(self.audio_token)

        return self._prepare_single_text(all_tokens)

    def _prepare_single_text(self, tokens: List[str]) -> Dict[str, torch.Tensor]:
        """
        Convert a list of tokens to input IDs and attention mask, with truncation/padding.
        """
        # Account for [CLS] and [SEP]
        max_tokens = self.max_length - 2
        truncated = tokens[:max_tokens]

        # Add special tokens
        full_tokens = [self.cls_token] + truncated + [self.sep_token]

        # Convert to IDs
        input_ids = [self.token_to_id(tok) for tok in full_tokens]

        # Create attention mask
        attention_mask = [1] * len(input_ids)

        # Pad if needed
        pad_len = self.max_length - len(input_ids)
        if pad_len > 0:
            pad_id = self.token_to_id(self.pad_token)
            input_ids.extend([pad_id] * pad_len)
            attention_mask.extend([0] * pad_len)
        elif pad_len < 0:
            logger.warning(f"Sequence truncated to max_length={self.max_length}")
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long)
        }

    def encode(self, text: str) -> Dict[str, torch.Tensor]: # pyright: ignore[reportIncompatibleMethodOverride]
        """Encode a single text string."""
        tokens = self.tokenize(text)
        return self._prepare_single_text(tokens)

    def batch_encode(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Encode a batch of text strings."""
        all_input_ids = []
        all_attention_masks = []
        for text in texts:
            encoded = self.encode(text)
            all_input_ids.append(encoded["input_ids"])
            all_attention_masks.append(encoded["attention_mask"])
        return {
            "input_ids": torch.stack(all_input_ids),
            "attention_mask": torch.stack(all_attention_masks)
        }

    def decode(self, token_ids: Union[List[int], torch.Tensor], skip_special_tokens: bool = True) -> str: # pyright: ignore[reportIncompatibleMethodOverride]
        """
        Decode token IDs back to text, handling BPE and special tokens.
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        tokens = [self.id_to_token(tid) for tid in token_ids]

        if skip_special_tokens:
            specials = {self.pad_token, self.cls_token, self.sep_token, self.image_token, self.audio_token,
                        self.bos_token, self.eos_token, self.mask_token}
            tokens = [t for t in tokens if t not in specials]

        # Reconstruct text: join tokens with spaces, but we need to handle BPE merging
        # The base detokenize method is suitable, but we need to remove BPE sentinel
        text = self.detokenize(tokens)

        # Clean up (remove extra spaces, fix punctuation)
        text = self.clean_text(text)

        return text

    def __call__(self, text: Union[str, List[str]]) -> Dict[str, torch.Tensor]: # pyright: ignore[reportIncompatibleMethodOverride]
        if isinstance(text, str):
            return self.encode(text)
        elif isinstance(text, list):
            return self.batch_encode(text)
        else:
            raise TypeError("Input must be a string or a list of strings.")

    def chunk_sequence(self, tokens: List[str], chunk_size: int) -> List[List[str]]:
        """Split long sequences into manageable chunks."""
        return [tokens[i:i+chunk_size] for i in range(0, len(tokens), chunk_size)]

    def get_position_ids(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """Generate position IDs from attention mask."""
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)
        return position_ids


# ----------------------------------------------------------------------
# Test block
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("\n=== Running Tokenizer ===\n")
    try:
        tokenizer = Tokenizer()
        example_text = "I love you SLAI!"

        # Single text
        encoded = tokenizer.encode(example_text)
        print("Encoded:")
        print("Input IDs:", encoded['input_ids'])
        print("Attention Mask:", encoded['attention_mask'])

        # Decode
        decoded = tokenizer.decode(encoded['input_ids'])
        print("Decoded Text:", decoded)

        # Batch
        test_cases = ["Hello world!", "I'm testing contractions", "SLAI-2023"]
        batch_encoded = tokenizer.batch_encode(test_cases)
        print("\nBatch encoded shapes:", batch_encoded['input_ids'].shape)

        # Multi‑modal
        multimodal = tokenizer.encode_multi_modal(text=example_text)
        printer.pretty("Multi‑modal encoding", multimodal, "success")

    except Exception as e:
        print("Tokenizer failed:", str(e))
        import traceback
        traceback.print_exc()

    print("\n=== Successfully Ran Tokenizer ===\n")
