
import os
import re
import string
import json, yaml
import hashlib

from cryptography.fernet import Fernet
from typing import Optional, Any, Dict, Union, List
from collections import OrderedDict
from types import SimpleNamespace

from logs.logger import get_logger

logger = get_logger("Knowledge Cache")

CONFIG_PATH = "src/agents/knowledge/configs/knowledge_config.yaml"
ACTION_PATTERN = re.compile(r"action:(\w+):(.+)", re.IGNORECASE)

def dict_to_namespace(d):
    """Recursively convert dicts to SimpleNamespace for dot-access."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(i) for i in d]
    return d

def get_config_section(section: Union[str, Dict], config_file_path: str):
    if isinstance(section, dict):
        return dict_to_namespace(section)
    
    with open(config_file_path, "r") as f:
        config = yaml.safe_load(f)
    if section not in config:
        raise KeyError(f"Section '{section}' not found in config file: {config_file_path}")
    return dict_to_namespace(config[section])

class KnowledgeCache:
    """LRU Cache with Semantic Hashing and Encryption"""
    def __init__(self, config_section_name: str = "knowledge_cache", config_file_path: str = CONFIG_PATH):
        self.config = get_config_section(config_section_name, config_file_path)
        self.cache = OrderedDict()
        self.cipher = Fernet(os.getenv('CACHE_ENCRYPTION_KEY', Fernet.generate_key())) \
            if self.config.enable_encryption else None
        
    def get(self, key: str) -> Optional[Any]:
        if key not in self.cache:
            return None
            
        # LRU update
        self.cache.move_to_end(key)
        value = self.cache[key]
        
        # Decrypt if enabled
        if self.config.enable_encryption:
            return json.loads(self.cipher.decrypt(value).decode())
        return value

    def set(self, key: str, value: Any) -> None:
        # Encrypt if enabled
        if self.config.enable_encryption:
            value = self.cipher.encrypt(json.dumps(value).encode())
        
        # LRU logic
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        
        # Evict oldest if over capacity
        if len(self.cache) > self.config.max_size:
            self.cache.popitem(last=False)

    def hash_query(self, query: str) -> str:
        """Semantic hashing using SimHash or MD5"""
        if self.config.hashing_method == "simhash":
            return self._simhash(query)
        return hashlib.md5(query.encode()).hexdigest()

    def _tokenize(self, text: str) -> List[str]:
        """Tokenizer for SimHash"""
        if self.config.tokenizer == "word":
            # Remove punctuation and split into words
            text = text.translate(str.maketrans('', '', string.punctuation))
            return text.lower().split()
        elif self.config.tokenizer == "char":
            return list(text.lower())
        return [text]

    def _simhash(self, query: str) -> str:
        """SimHash implementation for similar query grouping"""
        tokens = self._tokenize(query)
        hash_bits = [0] * self.config.simhash_bits

        for token in tokens:
            # Create weighted hash for each token
            token_hash = hashlib.md5(token.encode()).digest()
            bit_mask = int.from_bytes(token_hash, byteorder='big')

            for bit_pos in range(self.config.simhash_bits):
                if bit_mask & (1 << bit_pos):
                    hash_bits[bit_pos] += 1
                else:
                    hash_bits[bit_pos] -= 1

        # Convert to bit string
        simhash_value = 0
        for bit_pos in range(self.config.simhash_bits):
            if hash_bits[bit_pos] > 0:
                simhash_value |= 1 << bit_pos

        return hex(simhash_value)[2:]
if __name__ == "__main__":
    import os
    import tempfile
    import time

    # Create temporary config file (no encryption, use MD5)
    temp_config_path = os.path.join(tempfile.gettempdir(), "cache_config.yaml")
    with open(temp_config_path, "w") as f:
        f.write("""
knowledge_cache:
  max_size: 2
  enable_encryption: false
  hashing_method: "md5"
  tokenizer: "word"
  simhash_bits: 64
        """)

    # Initialize KnowledgeCache
    cache = KnowledgeCache(config_file_path=temp_config_path)
    print("📁 Cache Initialized (MD5, no encryption)\n")

    # Simulated chatbot Q&A
    questions = {
        "What is AI?": "AI stands for Artificial Intelligence.",
        "How does machine learning work?": "ML uses data to learn patterns.",
        "What is deep learning?": "Deep learning is a subset of ML using neural networks."
    }

    # Add entries to the cache (2 max entries allowed)
    for question, answer in questions.items():
        key = cache.hash_query(question)
        cache.set(key, {"question": question, "answer": answer})
        print(f"➕ Cached: {question}")

        time.sleep(0.2)  # Simulate time between requests

    print("\n📦 Cache Contents:")
    for k in cache.cache:
        print(f"- {k[:6]}")

    print("\n🔍 Retrieving previous entry (should be evicted):")
    forgotten_key = cache.hash_query("What is AI?")
    forgotten_entry = cache.get(forgotten_key)
    if forgotten_entry:
        print(f"✅ Found: {forgotten_entry}")
    else:
        print("❌ Entry for 'What is AI?' was evicted due to LRU policy")

    print("\n🔄 Retesting access to most recent question:")
    recent_key = cache.hash_query("What is deep learning?")
    recent_entry = cache.get(recent_key)
    if recent_entry:
        print(f"✅ Found: {recent_entry}")
    else:
        print("❌ Unexpected cache miss")
