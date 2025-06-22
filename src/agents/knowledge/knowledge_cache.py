
import os
import re
import string
import json, yaml
import hashlib

from typing import Optional, Any, Dict, Union, List
from cryptography.fernet import Fernet
from collections import OrderedDict, defaultdict

from src.agents.knowledge.utils.config_loader import load_global_config, get_config_section
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Knowledge Cache")
printer = PrettyPrinter

ACTION_PATTERN = re.compile(r"action:(\w+):(.+)", re.IGNORECASE)

class KnowledgeCache:
    """LRU Cache with Semantic Hashing and Encryption"""
    def __init__(self):
        self.config = load_global_config()
        self.cache_config = get_config_section('knowledge_cache')
        self.cache = OrderedDict()
        self.cipher = Fernet(os.getenv('CACHE_ENCRYPTION_KEY', Fernet.generate_key())) \
            if self.cache_config.get('enable_encryption', True) else None

        self.max_size = self.cache_config.get('max_size')
        self.hashing_method = self.cache_config.get('hashing_method')
        self.simhash_bits = self.cache_config.get('simhash_bits')
        self.stopwords = self.cache_config.get('stopwords')
        self.tokenizer = self.cache_config.get('tokenizer')
        self.shingle_size = self.cache_config.get('shingle_size')
        self.use_tf_weights = self.cache_config.get('use_tf_weights')
        self.character_ngram = self.cache_config.get('character_ngram')
        self.enable_encryption = self.cache_config.get('enable_encryption')

        logger.info(f"Knowledge Cache initialized with: {self.cache}")

    def __contains__(self, key):
        return key in self.cache
    
    def __getitem__(self, key):
        return self.get(key)
    
    def __len__(self):
        return len(self.cache)

    def get(self, key: str) -> Optional[Any]:
        if key not in self.cache:
            return None
            
        # LRU update
        self.cache.move_to_end(key)
        value = self.cache[key]
        
        # Decrypt if enabled (using dictionary access)
        if self.enable_encryption:
            return json.loads(self.cipher.decrypt(value).decode())
        return value

    def set(self, key: str, value: Any) -> None:
        # Encrypt if enabled (using dictionary access)
        if self.cache_config.get('enable_encryption'):
            value = self.cipher.encrypt(json.dumps(value).encode())
        
        # LRU logic
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        
        # Evict oldest if over capacity
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

    def hash_query(self, query: str) -> str:
        """Semantic hashing using SimHash or MD5 with configurable methods"""
        hashing_method = self.hashing_method
        
        if hashing_method == "simhash":
            return self._simhash(query)
        else:  # Fallback to MD5
            return hashlib.md5(query.encode()).hexdigest()

    def _simhash(self, query: str) -> str:
        """
        Enhanced SimHash implementation based on Charikar's algorithm
        - Supports configurable bit-length (64/128/256)
        - Uses TF weighting by default
        - Optimized hash selection based on required bits
        """
        # Get configuration parameters
        simhash_bits = self.simhash_bits
        tokenizer_type = self.tokenizer
        use_tf_weights = self.use_tf_weights
        
        # Tokenize with appropriate method
        tokens = self._tokenize(query, tokenizer_type)
        
        # Calculate token weights (TF by default)
        token_weights = self._calculate_token_weights(tokens) if use_tf_weights \
            else {token: 1 for token in set(tokens)}
        
        # Initialize bit vector
        vector = [0] * simhash_bits
        
        # Process tokens
        for token, weight in token_weights.items():
            # Select optimal hash function based on required bits
            if simhash_bits <= 128:
                hash_func = hashlib.md5
            else:  # >128 bits
                hash_func = hashlib.sha256
                
            # Generate hash and convert to integer
            token_hash = hash_func(token.encode()).digest()
            bit_mask = int.from_bytes(token_hash, byteorder='big')
            
            # Process each bit position
            for bit_pos in range(simhash_bits):
                if bit_mask & (1 << bit_pos):
                    vector[bit_pos] += weight
                else:
                    vector[bit_pos] -= weight
        
        # Generate final fingerprint
        fingerprint = 0
        for bit_pos in range(simhash_bits):
            if vector[bit_pos] > 0:
                fingerprint |= 1 << bit_pos
        
        # Convert to fixed-length hex string
        return self._to_fixed_length_hex(fingerprint, simhash_bits)

    def _calculate_token_weights(self, tokens: List[str]) -> dict:
        """Calculate TF weights with stopword filtering"""
        stopwords = self.stopwords
        tf = defaultdict(int)
        
        for token in tokens:
            if token not in stopwords:
                tf[token] += 1
        
        return tf

    def _to_fixed_length_hex(self, value: int, num_bits: int) -> str:
        """Convert to zero-padded hex string"""
        num_hex_digits = (num_bits + 3) // 4  # Bits to hex digits
        return format(value, f'0{num_hex_digits}x')

    def _tokenize(self, text: str, tokenizer_type: str = 'word') -> List[str]:
        """Enhanced tokenizer with configurable methods"""
        text = text.lower()
        
        if tokenizer_type == "word":
            # Remove punctuation and split
            text = re.sub(r'[^\w\s]', '', text)
            return text.split()
        
        elif tokenizer_type == "char":
            # Character n-grams (configurable length)
            n = self.character_ngram
            return [text[i:i+n] for i in range(len(text) - n + 1)]
        
        elif tokenizer_type == "shingle":
            # Word n-grams (configurable length)
            n = self.shingle_size
            words = re.sub(r'[^\w\s]', '', text).split()
            return [' '.join(words[i:i+n]) for i in range(len(words) - n + 1)]
        
        else:  # Default to whole text
            return [text]

if __name__ == "__main__":
    print("\n=== Running Knowledge Cache ===\n")
    printer.status("Init", "Knowledge Cache initialized", "success")

    cache = KnowledgeCache()
    printer.status("Details", f"Cache capacity: {cache.max_size}", "info")

    query="What are the ethical concerns surrounding artificial intelligence?"

    printer.pretty("HASH", cache.hash_query(query=query), "success")

    print("\n=== Succesfully ran Knowledge Cache ===\n")
