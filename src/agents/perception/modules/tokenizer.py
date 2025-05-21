
import os
import json, yaml
import re
import torch
import unicodedata

from pathlib import Path
from typing import List, Dict, Union, Optional, Tuple

from logs.logger import get_logger

logger = get_logger("Tokenizer")

BPE_MODEL_PATH = "data/embeddings/bpe_200d_50k_model.json"
BPE_VOCAB_PATH = "data/embeddings/bpe_200d_50k_vocab.json"
CONFIG_PATH = "src/agents/perception/configs/perception_config.yaml"

def load_config(config_path=CONFIG_PATH):
    with open(config_path, "r", encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def get_merged_config(user_config=None):
    base_config = load_config()
    if user_config:
        base_config.update(user_config)
    return base_config

class Tokenizer:
    """
    Handles text preprocessing: tokenization, ID conversion, padding, truncation,
    and attention mask generation based on a predefined vocabulary
    """
    def __init__(self, config):
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
        cfg = config['tokenizer']
        self.max_length = cfg['max_length']
        self.bpe_merges_path = Path(cfg['bpe_model_path'])
        self.bpe_vocab_path = Path(cfg['bpe_vocab_path'])
        self.vocab_size = cfg['vocab_size']

        # --- Special Tokens ---
        self.pad_token = cfg['pad_token']
        self.unk_token = cfg['unk_token']
        self.cls_token = cfg['cls_token']
        self.sep_token = cfg['sep_token']

        if not self.bpe_vocab_path.exists():
            raise FileNotFoundError(f"BPE vocab file not found: {self.bpe_vocab_path}")
        if not self.bpe_merges_path.exists():
             raise FileNotFoundError(f"BPE model (merges) file not found: {self.bpe_merges_path}")

        # Load BPE Vocabulary
        with open(self.bpe_vocab_path, "r", encoding="utf-8") as f:
            self.word_to_id = json.load(f)

        # Load BPE Merges
        with open(self.bpe_merges_path, "r", encoding="utf-8") as f:
            merges_data = json.load(f)
            # Assuming merges are stored under a key like "merges", adjust if needed
            bpe_merges = [tuple(pair) for pair in merges_data.get("merges", [])]

        self.bpe_ranks = {pair: i for i, pair in enumerate(bpe_merges)}
        self.bpe_processor = BytePairEncoder(bpe_merges, self.word_to_id, self.unk_token)

        # Add special tokens if they aren't already in the BPE vocab
        self.special_tokens = [self.pad_token, self.unk_token, self.cls_token, self.sep_token]
        for token in self.special_tokens:
            if token not in self.word_to_id:
                 # Add special tokens with high IDs
                 next_id = max(self.word_to_id.values()) + 1
                 self.word_to_id[token] = next_id
                 logger.info(f"Added special token '{token}' with ID {next_id}")

        self.id_to_word = {idx: token for token, idx in self.word_to_id.items()}
        self.vocab_size = len(self.word_to_id)

        # Assign specific IDs after loading
        self.pad_token_id = self.word_to_id[self.pad_token]
        self.unk_token_id = self.word_to_id[self.unk_token]
        self.cls_token_id = self.word_to_id[self.cls_token]
        self.sep_token_id = self.word_to_id[self.sep_token]

        # Pre-compile regex patterns
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?[^\s\w\d]+|\s+(?!\S)|\s+""")
        self.cache = {} # BPE cache

        logger.info(f"Tokenizer initialized with BPE vocab size: {self.vocab_size}")
        logger.info(f"Using BPE model: {self.bpe_merges_path}")
        logger.info(f"Using BPE vocab: {self.bpe_vocab_path}")

    def _load_vocab(self):
        """Loads the vocabulary from the GloVe JSON file and adds special tokens."""
        if not self.bpe_vocab_path.exists():
            logger.error(f"Vocabulary file not found at: {self.bpe_vocab_path}")
            raise FileNotFoundError(f"Vocabulary file not found: {self.bpe_vocab_path}")

        try:
            logger.info(f"Loading vocabulary from: {self.bpe_vocab_path}")
            with open(self.bpe_vocab_path, "r", encoding="utf-8") as f:
                glove_data = json.load(f)
            logger.info(f"Successfully loaded {len(glove_data)} words from GloVe file.")

            # 1. Add special tokens first to ensure consistent IDs (0, 1, 2, ...)
            current_id = 0
            all_special_tokens = self._special_tokens_list + self._additional_special_tokens
            for token in all_special_tokens:
                if token not in self.word_to_id:
                    self.word_to_id[token] = current_id
                    self.id_to_word[current_id] = token
                    current_id += 1

        #    # 2. Add GloVe vocabulary words, skipping any duplicates of special tokens
        #    for word in glove_data.keys():
        #        # Ensure GloVe words don't overwrite special tokens if they happen to exist
        #        if word not in self.word_to_id:
        #            self.word_to_id[word] = current_id
        #            self.id_to_word[current_id] = word
        #            current_id += 1

            self.vocab_size = len(self.word_to_id)

        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON from vocabulary file: {self.bpe_vocab_path}")
            raise ValueError(f"Invalid JSON file: {self.bpe_vocab_path}")
        except Exception as e:
            logger.error(f"Error loading vocabulary: {e}")
            raise

    def _tokenize(self, text: str) -> List[str]:
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

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """Converts a list of string tokens to their corresponding integer IDs."""
        ids = []
        for token in tokens:
            if token in self.word_to_id:
                ids.append(self.word_to_id[token])
            else:
                logger.warning(f"Unknown token encountered: {token}")
                ids.append(self.unk_token_id)
        return ids

    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        """Converts a list of integer IDs back to their string tokens."""
        return [self.id_to_word.get(id_val, self.unk_token) for id_val in ids]

    def _prepare_single_text(self, text: str) -> Tuple[List[int], List[int]]:
        """Internal function to tokenize, add special tokens, convert to IDs, and handle padding/truncation."""
        tokens = self._tokenize(text)

        # Account for [CLS] and [SEP] tokens that will be added
        max_tokens_for_input = self.max_length - 2

        # Truncate if necessary BEFORE adding special tokens
        truncated_tokens = tokens[:max_tokens_for_input]

        # Add special tokens
        tokens_with_special = [self.cls_token] + truncated_tokens + [self.sep_token]

        # Convert to IDs
        input_ids = self.convert_tokens_to_ids(tokens_with_special)

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1] * len(input_ids)

        # Calculate padding length
        padding_length = self.max_length - len(input_ids)

        # Apply padding
        if padding_length > 0:
            input_ids = input_ids + ([self.pad_token_id] * padding_length)
            attention_mask = attention_mask + ([0] * padding_length)
        elif padding_length < 0:
            # This case should ideally not happen if truncation logic is correct, but as a safeguard:
            logger.warning(f"Sequence length exceeded max_length ({self.max_length}) even after truncation. Truncating final IDs.")
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]

        assert len(input_ids) == self.max_length, f"Final input_ids length is {len(input_ids)}, expected {self.max_length}"
        assert len(attention_mask) == self.max_length, f"Final attention_mask length is {len(attention_mask)}, expected {self.max_length}"

        return input_ids, attention_mask

    def encode(self, text: str) -> Dict[str, torch.Tensor]:
        """
        Encodes a single string into token IDs and an attention mask.

        Args:
            text (str): The input text string.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing:
                - 'input_ids': NumPy array of token IDs (shape: [max_length]).
                - 'attention_mask': NumPy array of attention mask (shape: [max_length]).
        """
        input_ids, attention_mask = self._prepare_single_text(text)
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.int32),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.int32)
        }

    def batch_encode(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """
        Encodes a batch of strings into token IDs and attention masks.

        Args:
            texts (List[str]): A list of input text strings.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing:
                - 'input_ids': NumPy array of token IDs (shape: [batch_size, max_length]).
                - 'attention_mask': NumPy array of attention masks (shape: [batch_size, max_length]).
        """
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
        """
        Decodes a sequence of token IDs back into a string with improved spacing and punctuation handling.
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
    
        tokens = []
        for token_id in token_ids:
            token = self.id_to_word.get(token_id, self.unk_token)
            if skip_special_tokens and token in [self.pad_token, self.cls_token, self.sep_token]:
                continue
            tokens.append(token)
    
        # Reconstruct the text from BPE tokens
        current_part = []
        parts = []
        for token in tokens:
            if token.endswith('</w>'):
                # Remove the </w> and add to current_part
                stripped_token = token[:-4]
                current_part.append(stripped_token)
                # Join and add to parts, then reset current_part
                parts.append(''.join(current_part))
                current_part = []
            else:
                current_part.append(token)
        # Add any remaining parts
        if current_part:
            parts.append(''.join(current_part))
        
        # Join parts with spaces between original tokens
        decoded_text = ' '.join(parts)
        
        # Post-process to handle punctuation spacing
        # Remove spaces before punctuation
        decoded_text = re.sub(r'\s+([,.!?])', r'\1', decoded_text)
        # Replace multiple spaces with a single space
        decoded_text = re.sub(r'\s+', ' ', decoded_text)
        # Trim leading and trailing spaces
        decoded_text = decoded_text.strip()
    
        return decoded_text

    # Make the tokenizer callable like Hugging Face tokenizers
    def __call__(self, text: Union[str, List[str]]) -> Dict[str, torch.Tensor]:
        """Allows calling the tokenizer instance directly."""
        if isinstance(text, str):
            return self.encode(text)
        elif isinstance(text, list):
            return self.batch_encode(text)
        else:
            raise TypeError("Input must be a string or a list of strings.")

    def initialize_subword_embeddings(self, embedding_dim, glove_embeddings):
        """
        glove_embeddings: Dict[str, torch.Tensor] | A mapping of GloVe word → embedding vector.
        """
        self.embedding_matrix = (torch.rand(self.vocab_size, embedding_dim) * 0.2) - 0.1
    
        loaded_count = 0
        for token, idx in self.word_to_id.items():
            if token in glove_embeddings:
                self.embedding_matrix[idx] = glove_embeddings[token]
                loaded_count += 1
    
        logger.info(f"Initialized embedding matrix: "
                    f"{loaded_count} pretrained embeddings, "
                    f"{self.vocab_size - loaded_count} random-initialized (subwords or unknown).")

    def _load_bpe_merges(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            merges = [tuple(line.strip().split()) for line in f if line.strip() and not line.startswith('#')]
        self.bpe_processor = BytePairEncoder(self.word_to_id, merges)

    def load_bpe_model(self, bpe_model_path):
        with open(bpe_model_path, 'r', encoding='utf-8') as f:
            bpe_data = json.load(f)
        
        merges = [tuple(pair) for pair in bpe_data['merges']]
        self.bpe_processor = BytePairEncoder(merges)
    
        # Build subword vocab
        subword_vocab = set()
        for pair in merges:
            subword_vocab.update(pair)
        # Start from existing vocab (GloVe words)
        combined_vocab = set(self.word_to_id.keys())
        combined_vocab.update(subword_vocab)
        combined_vocab.update(self._special_tokens_list)
    
        # Assign IDs
        self.word_to_id = {token: idx for idx, token in enumerate(sorted(combined_vocab))}
        self.id_to_word = {idx: token for token, idx in self.word_to_id.items()}
        self.vocab_size = len(self.word_to_id)

        for special in self._special_tokens_list:
            if special not in self.word_to_id:
                new_id = len(self.word_to_id)
                self.word_to_id[special] = new_id
                self.id_to_word[new_id] = special
    
        # Update special token IDs
        self.pad_token_id = self.word_to_id[self.pad_token]
        self.unk_token_id = self.word_to_id[self.unk_token]
        self.cls_token_id = self.word_to_id[self.cls_token]
        self.sep_token_id = self.word_to_id[self.sep_token]

    def _special_tokens_list():
        pass

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
    config = load_config()

    try:
        tokenizer = Tokenizer(config)

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

        # Access BPE encoder directly
        test_token = "Will you...kiss me?"
        subwords = tokenizer.bpe_processor.bpe(test_token)
        print(f"BPE for '{test_token}':", subwords)

    except Exception as e:
        print("Tokenizer initialization or execution failed:", str(e))

    print("\n=== Successfully Ran Tokenizer ===\n")
