
import os
import json
import hashlib

from cryptography.fernet import Fernet
from typing import Optional, Any
from collections import OrderedDict

class KnowledgeCache:
    """Lightweight cache with LRU eviction policy"""
    def __init__(self, max_size: int = 100):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.cipher = Fernet(os.getenv('CACHE_ENCRYPTION_KEY', Fernet.generate_key()))

    def get(self, key: str) -> Optional[Any]:
        encrypted = super().get(key)
        if encrypted:
            return json.loads(self.cipher.decrypt(encrypted).decode())
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def set(self, key: str, value: Any) -> None:
        encrypted = self.cipher.encrypt(json.dumps(value).encode())
        super().set(key, encrypted)
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

    def hash_query(self, query: str) -> str:
        """Semantic hashing for similar queries (simplified)"""
        return hashlib.md5(query.encode()).hexdigest()
