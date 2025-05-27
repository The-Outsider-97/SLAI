import torch
import hashlib
import pickle
import time
import yaml

from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

from logs.logger import get_logger

logger = get_logger("Language Cache")

CONFIG_PATH = "src/agents/language/configs/language_config.yaml"

def load_config(config_path=CONFIG_PATH):
    with open(config_path, "r", encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def get_merged_config(user_config=None):
    base_config = load_config()
    if user_config:
        base_config.update(user_config)
    return base_config

class LanguageCache:
    def __init__(self, config):
        self.config = config
        self.max_size = self.config.get("max_size", 100)
        self.expiry_seconds = self.config.get("expiry_seconds", None)
        self.cache_path = Path(self.config.get("cache_path")) if self.config.get("cache_path") else None

        # (embedding_tensor, timestamp)
        self.embedding_cache: OrderedDict[str, Tuple[torch.Tensor, float]] = OrderedDict()
        self.summary_cache: Dict[str, str] = {}

        if self.cache_path and self.cache_path.exists():
            self._load_from_disk()

    def _hash_key(self, text: str) -> str:
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def _is_expired(self, timestamp: float) -> bool:
        if self.expiry_seconds is None:
            return False
        return (time.time() - timestamp) > self.expiry_seconds

    def add_embedding(self, text: str, embedding: torch.Tensor):
        key = self._hash_key(text)
        now = time.time()

        if key in self.embedding_cache:
            self.embedding_cache.move_to_end(key)

        self.embedding_cache[key] = (embedding.detach().cpu(), now)

        if len(self.embedding_cache) > self.max_size:
            self.embedding_cache.popitem(last=False)

    def get_embedding(self, text: str) -> Optional[torch.Tensor]:
        key = self._hash_key(text)
        value = self.embedding_cache.get(key)

        if value is None:
            return None

        embedding, timestamp = value
        if self._is_expired(timestamp):
            del self.embedding_cache[key]
            return None

        return embedding

    def most_similar(self, embedding: torch.Tensor, top_k: int = 1) -> List[str]:
        now = time.time()
        valid_items = {
            k: v[0]
            for k, v in self.embedding_cache.items()
            if not self._is_expired(v[1])
        }

        if not valid_items:
            return []

        all_keys = list(valid_items.keys())
        all_embeddings = torch.stack([valid_items[k] for k in all_keys])
        sim_scores = torch.nn.functional.cosine_similarity(
            embedding.unsqueeze(0), all_embeddings
        )
        top_indices = sim_scores.topk(top_k).indices.tolist()
        return [all_keys[i] for i in top_indices]

    def set_summary(self, session_id: str, summary: str):
        self.summary_cache[session_id] = summary

    def get_summary(self, session_id: str) -> Optional[str]:
        return self.summary_cache.get(session_id)

    def clear(self):
        self.embedding_cache.clear()
        self.summary_cache.clear()

    def save_to_disk(self):
        if self.cache_path:
            with open(self.cache_path, "wb") as f:
                pickle.dump({
                    "embedding_cache": self.embedding_cache,
                    "summary_cache": self.summary_cache
                }, f)

    def _load_from_disk(self):
        try:
            with open(self.cache_path, "rb") as f:
                data = pickle.load(f)
                self.embedding_cache = data.get("embedding_cache", OrderedDict())
                self.summary_cache = data.get("summary_cache", {})
        except Exception as e:
            print(f"[LanguageCache] Failed to load cache from disk: {e}")
