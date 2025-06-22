
import os
import json, yaml
import re
import torch
import unicodedata

from pathlib import Path
from typing import List, Dict, Union, Optional, Tuple

from src.agents.perception.utils.config_loader import load_global_config, get_config_section
from src.agents.perception.utils.common import TensorOps, Parameter
from src.agents.base.utils.base_tokenizer import BaseTokenizer
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Tokenizer")
printer = PrettyPrinter

class Tokenizer(BaseTokenizer):
    """
    Handles text preprocessing: tokenization, ID conversion, padding, truncation,
    and attention mask generation based on a predefined vocabulary
    """
    def __init__(self):
        super().__init__()
        """
        Initializes the Tokenizer.

        Args:
            vocab_path (Union[str, Path]): Path to the vocabulary file.
            max_length (int): The fixed sequence length for padding/truncation.
            pad_token (str): Token used for padding.
            unk_token (str): Token used for unknown words.
            cls_token (str): Classification token added at the beginning.
            sep_token (str): Separator token added at the end.
        """
        self.config = load_global_config()
        self.token_config = get_config_section('tokenizer')
        self.max_length = self.token_config.get('max_length')
        self.cls_token = self.token_config.get('cls_token')
        self.sep_token = self.token_config.get('sep_token')
        self.image_token = self.token_config.get('image_token')
        self.audio_token = self.token_config.get('audio_token')
        self.add_tokens([self.cls_token, self.sep_token], special=True)

        self.bpe_vocab_path = Path(self.bpe_vocab_path)
        self.bpe_model_path = Path(self.bpe_model_path)

        # Set up BPE processor
        self._setup_bpe_processor()
        
        # Pre-compile regex patterns
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?[^\s\w\d]+|\s+(?!\S)|\s+""")
        self.cache = {}  # BPE cache

        logger.info(f"Tokenizer initialized with max_length={self.max_length}")

    def _setup_bpe_processor(self):
        """Load BPE merges and vocabulary from base tokenizer"""
        # Base tokenizer should have already loaded these paths
        if not hasattr(self, 'bpe_model_path') or not hasattr(self, 'bpe_vocab_path'):
            raise AttributeError("BaseTokenizer did not initialize BPE paths")
        
        # Validate BPE files
        if not self.bpe_vocab_path.exists():
            raise FileNotFoundError(f"BPE vocab file not found: {self.bpe_vocab_path}")
        if not self.bpe_model_path.exists():
            raise FileNotFoundError(f"BPE model file not found: {self.bpe_model_path}")

        # Load BPE Vocabulary
        with open(self.bpe_vocab_path, "r", encoding="utf-8") as f:
            bpe_word_to_id = json.load(f)
        
        # Add BPE vocabulary to existing vocab
        for token in bpe_word_to_id:
            if token not in self.vocab:
                self.add_tokens([token])
                
        # Load BPE Merges
        with open(self.bpe_model_path, "r", encoding="utf-8") as f:
            merges_data = json.load(f)
            bpe_merges = [tuple(pair) for pair in merges_data["merges"]]
            logger.info(f"Loaded {len(bpe_merges)} BPE merges")

        self.bpe_ranks = {pair: i for i, pair in enumerate(bpe_merges)}
        self.bpe_processor = BytePairEncoder(bpe_merges, self.vocab, self.unk_token)

    def tokenize(self, text: str) -> List[str]:
        """Override base tokenization with BPE processing"""
        def normalize(text):
            text = unicodedata.normalize('NFKC', text)
            text = ''.join(c for c in text if not unicodedata.combining(c))
            return text.lower()
        
        tokens = self.pat.findall(normalize(text))
    
        if hasattr(self, 'bpe_processor'):
            subword_tokens = []
            for token in tokens:
                sub_pieces = self.bpe_processor.bpe(token)
                subword_tokens.extend(sub_pieces)
            return subword_tokens
        else:
            return tokens

    def get_token_type_ids(self, text_length: int, image_length: int,
                           audio_length: int) -> torch.Tensor:
        """Generate token type IDs for each modality"""
        return torch.cat([
            torch.zeros(text_length),          # Text type = 0
            torch.ones(image_length),          # Image type = 1
            torch.full((audio_length,), 2)     # Audio type = 2
        ])

    def create_cross_modal_mask( self, text_length: int, image_length: int,
                                audio_length: int) -> torch.Tensor:
        """Create attention mask for cross-modal attention"""
        total_length = text_length + image_length + audio_length
        mask = torch.ones(total_length, total_length)
        
        # Allow full attention within modalities
        mask[:text_length, :text_length] = 0
        mask[text_length:text_length+image_length, text_length:text_length+image_length] = 0
        mask[text_length+image_length:, text_length+image_length:] = 0
        
        # Allow cross-modal attention
        return  TensorOps.attention_mask(
        lengths=torch.tensor([text_length, image_length, audio_length]),
        causal=False
    )

    def encode_multi_modal(self, text: Optional[str] = None, image_features: Optional[torch.Tensor] = None,
                            audio_features: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        printer.status("TOKEN", "Encoding multi modal input", "info")

        sequences = []
        
        if text:
            text_tokens = self.tokenize(text)
            sequences.append(text_tokens)
        
        if image_features is not None:
            sequences.append([self.image_token])
            # (Image features would be handled separately in the model)
        
        if audio_features is not None:
            sequences.append([self.audio_token])
        
        # Flatten all tokens
        tokens = [token for seq in sequences for token in seq]
        
        # Continue with standard preparation
        return self._prepare_single_text(tokens)

    def _prepare_single_text(self, text: str) -> Tuple[List[int], List[int]]:
        """Tokenize, add special tokens, convert to IDs, handle padding/truncation"""
        printer.status("TOKEN", "Preparing single text", "info")
        tokens = self.tokenize(text)

        # Account for [CLS] and [SEP] tokens
        max_tokens = self.max_length - 2

        # Truncate before adding special tokens
        truncated_tokens = tokens[:max_tokens]

        # Add special tokens
        tokens_with_special = [self.cls_token] + truncated_tokens + [self.sep_token]

        # Convert to IDs
        input_ids = [self.token_to_id(token) for token in tokens_with_special]

        # Create attention mask
        attention_mask = [1] * len(input_ids)

        # Handle padding
        padding_length = self.max_length - len(input_ids)
        if padding_length > 0:
            pad_id = self.token_to_id(self.pad_token)
            input_ids = TensorOps.pad_sequence(input_ids, self.max_length, value=pad_id)
            attention_mask = TensorOps.pad_sequence(attention_mask, self.max_length, value=0)
            input_ids += [pad_id] * padding_length
            attention_mask += [0] * padding_length
        elif padding_length < 0:
            logger.warning(f"Sequence truncated to max_length={self.max_length}")
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]

        return input_ids, attention_mask

    def encode(self, text: str) -> Dict[str, torch.Tensor]:
        input_ids, attention_mask = self._prepare_single_text(text)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.int32),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.int32)
        }

    def batch_encode(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        printer.status("TOKEN", "Encoding batch input", "info")

        all_input_ids = []
        all_attention_masks = []

        for text in texts:
            input_ids, attention_mask = self._prepare_single_text(text)
            all_input_ids.append(input_ids)
            all_attention_masks.append(attention_mask)

        return {
            "input_ids": torch.tensor(all_input_ids, dtype=torch.int32),
            "attention_mask": torch.tensor(all_attention_masks, dtype=torch.int32)
        }

    def decode(self, token_ids: Union[List[int], torch.Tensor], skip_special_tokens: bool = True) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
    
        tokens = []
        for token_id in token_ids:
            token = self.id_to_token(token_id)
            if skip_special_tokens and token in [self.pad_token, self.cls_token, self.sep_token]:
                continue
            tokens.append(token)
    
        # Reconstruct text from BPE tokens
        decoded_text = ""
        for token in tokens:
            if token == self.unk_token:
                decoded_text += " [UNK]"
            elif token.endswith('</w>'):
                decoded_text += token[:-4] + " "
            else:
                decoded_text += token

        # Handle modality tokens
        for token in tokens:
            if token == self.image_token:
                decoded_text += " [IMAGE]"
            elif token == self.audio_token:
                decoded_text += " [AUDIO]"
                
        # Apply base class cleaning
        return self.clean_text(decoded_text.strip())

    def __call__(self, text: Union[str, List[str]]) -> Dict[str, torch.Tensor]:
        if isinstance(text, str):
            return self.encode(text)
        elif isinstance(text, list):
            return self.batch_encode(text)
        else:
            raise TypeError("Input must be a string or a list of strings.")

    def chunk_sequence(self, tokens: List[str], chunk_size: int) -> List[List[str]]:
        """Split long sequences into manageable chunks"""
        return [tokens[i:i+chunk_size] for i in range(0, len(tokens), chunk_size)]

    def get_position_ids(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """Generate positional IDs from attention mask"""
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)
        return TensorOps.sequence_mask(attention_mask.sum(dim=1))


class BytePairEncoder:
    def __init__(self, merges, word_to_id=None, unk_token='[UNK]'):
        self.bpe_ranks = {tuple(merge): i for i, merge in enumerate(merges)}
        self.cache = {}
        self.word_to_id = word_to_id or {}
        self.unk_token = unk_token

    def get_pairs(self, word):
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]

        # Handle empty token or single-character tokens
        if len(token) == 0:
            return [self.unk_token]

        word = tuple(token) + ('</w>',)
        pairs = self.get_pairs(word)

        if not pairs:
            if token in self.word_to_id:
                return [token]
            else:
                logger.warning(f"BPE could not segment token: {token}")
                return [self.unk_token]

        while True:
            # Safeguard against empty pairs during merging
            if not pairs:
                break 
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break

            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                j = i
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            word = tuple(new_word)
            pairs = self.get_pairs(word)

        if word[-1] == '</w>':
            word = word[:-1]
        self.cache[token] = list(word)
        return self.cache[token]

if __name__ == "__main__":
    print("\n=== Running Tokenizer ===\n")
    try:
        tokenizer = Tokenizer()

        # Example input
        example_text = "I love you SLAI!"

        # Tokenize
        encoded = tokenizer.encode(example_text)
        print("Encoded:")
        print("Input IDs:", encoded['input_ids'])
        print("Attention Mask:", encoded['attention_mask'])

        # Decode back
        decoded = tokenizer.decode(encoded['input_ids'])
        print("Decoded Text:", decoded)

        # Test cases
        test_cases = [
            "Hello world!", 
            "I'm testing contractions",
            "SLAI-2023"
        ]
        batch_encoded = tokenizer.batch_encode(test_cases)
        print("\nBatch Encoded IDs:", batch_encoded['input_ids'][0][:10])

    except Exception as e:
        print("Tokenizer failed:", str(e))

    print("\n* * * * * Phase 2 * * * * *\n")
    multimodal = tokenizer.encode_multi_modal(text=example_text)

    printer.pretty("TEST2", multimodal, "success")
    print("\n=== Successfully Ran Tokenizer ===\n")
