import torch
import hashlib
import pickle
import time
import yaml
import os
import json

from abc import ABC, abstractmethod
from collections import OrderedDict, deque
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from pathlib import Path

from src.agents.language.utils.config_loader import load_global_config, get_config_section
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Language Cache")
printer = PrettyPrinter

class BaseCacheStrategy(ABC):
    """Abstract base class for cache replacement strategies"""
    @abstractmethod
    def add_item(self, key: str, value: Any):
        pass
    
    @abstractmethod
    def get_next_eviction_key(self) -> str:
        pass

class LRUCacheStrategy(BaseCacheStrategy):
    """Least Recently Used cache replacement strategy"""
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.access_order = deque(maxlen=max_size)
    
    def add_item(self, key: str, value: Any):
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)
    
    def get_next_eviction_key(self) -> str:
        return self.access_order[0] if self.access_order else ""

class LFUCacheStrategy(BaseCacheStrategy):
    """Least Frequently Used cache replacement strategy"""
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.access_count: Dict[str, int] = {}
        self.min_heap = []
    
    def add_item(self, key: str, value: Any):
        self.access_count[key] = self.access_count.get(key, 0) + 1
        # Simplified LFU implementation - real version would use heap
        
    def get_next_eviction_key(self) -> str:
        if not self.access_count:
            return ""
        return min(self.access_count, key=self.access_count.get)

class LanguageCache:
    VERSION = "1.8"
    SERIALIZATION_PROTOCOL = 4

    def __init__(self):
        self.config = load_global_config()
        self.cache_config = get_config_section('language_cache')
        self.max_size = self.cache_config.get('max_size')
        self.expiry_seconds = self.cache_config.get('expiry_seconds')
        self.cache_path = Path(self.cache_config.get("cache_path")) if self.cache_config.get("cache_path") else None
        self.strategy_name = self.config.get("strategy")
        self.enable_compression = self.config.get("enable_compression")
        self.enable_encryption = self.config.get("enable_encryption")

        self._init_caches()        # Initialize caches with flexible strategies
        self.cache_path.parent.mkdir(parents=True, exist_ok=True) # Create cache directory if needed

        # (embedding_tensor, timestamp)
        self.embedding_cache: OrderedDict[str, Tuple[torch.Tensor, float]] = OrderedDict()
        self.summary_cache: Dict[str, str] = {}

        if self.cache_path.exists():
            self._load_from_disk()

    def _init_caches(self):
        """Initialize caches with selected strategy"""
        strategy_map = {
            "LRU": LRUCacheStrategy,
            "LFU": LFUCacheStrategy
        }

        strategy_class = strategy_map.get(self.strategy_name, LRUCacheStrategy)
        self.embedding_cache: OrderedDict[str, Tuple[torch.Tensor, float]] = OrderedDict()
        self.summary_cache: Dict[str, str] = {}
        self.strategy = strategy_class(self.max_size)

    def _hash_key(self, text: str, algorithm: str = "sha256") -> str:
        """Generate cache key with configurable hashing algorithm"""
        hasher = hashlib.new(algorithm)
        hasher.update(text.encode('utf-8'))
        return f"{algorithm}:{hasher.hexdigest()}"
    
    def _is_expired(self, timestamp: float) -> bool:
        """Check if item is expired with configurable grace period"""
        if self.expiry_seconds is None or self.expiry_seconds <= 0:
            return False
        return (time.time() - timestamp) > self.expiry_seconds
    
    def add_embedding(
        self, 
        text: str, 
        embedding: torch.Tensor,
        metadata: Optional[Dict] = None,
        custom_ttl: Optional[int] = None
    ):
        """Add embedding to cache with optional metadata and custom TTL"""
        key = self._hash_key(text)
        now = time.time()
        
        # Create value with metadata
        value = (
            embedding.detach().cpu(), 
            now,
            metadata or {},
            custom_ttl or self.expiry_seconds
        )
        
        # Update cache
        self.embedding_cache[key] = value
        self.strategy.add_item(key, value)
        
        # Apply eviction policy if needed
        if len(self.embedding_cache) > self.max_size:
            evict_key = self.strategy.get_next_eviction_key()
            if evict_key in self.embedding_cache:
                self.embedding_cache.pop(evict_key, None)
    
    def get_embedding(
        self, 
        text: str,
        update_access: bool = True
    ) -> Optional[Tuple[torch.Tensor, Dict]]:
        """Retrieve embedding with optional access time update"""
        key = self._hash_key(text)
        value = self.embedding_cache.get(key)
        
        if value is None:
            return None
        
        embedding, timestamp, metadata, ttl = value
        
        # Check expiration using item-specific TTL if available
        effective_ttl = ttl if ttl is not None else self.expiry_seconds
        if effective_ttl and (time.time() - timestamp) > effective_ttl:
            self.embedding_cache.pop(key, None)
            return None
        
        # Update access time if requested
        if update_access:
            self.embedding_cache[key] = (
                embedding, 
                time.time(), 
                metadata, 
                ttl
            )
            self.strategy.add_item(key, self.embedding_cache[key])
        
        return embedding, metadata
    
    def most_similar(
        self, 
        embedding: torch.Tensor, 
        top_k: int = 1,
        min_similarity: float = 0.6,
        filter_func: Optional[Callable[[Dict], bool]] = None
    ) -> List[Tuple[str, float]]:
        """Find similar embeddings with filtering and similarity threshold"""
        now = time.time()
        valid_items = {}
        
        for key, value in self.embedding_cache.items():
            emb, ts, meta, ttl = value
            
            # Check expiration
            effective_ttl = ttl if ttl is not None else self.expiry_seconds
            if effective_ttl and (now - ts) > effective_ttl:
                continue
            
            # Apply filter if provided
            if filter_func and not filter_func(meta):
                continue
            
            valid_items[key] = emb
        
        if not valid_items:
            return []
        
        keys = list(valid_items.keys())
        embeddings = torch.stack([valid_items[k] for k in keys])
        
        # Calculate similarity
        similarity = torch.nn.functional.cosine_similarity(
            embedding.unsqueeze(0), 
            embeddings
        )
        
        # Filter and sort results
        results = []
        for i, score in enumerate(similarity.tolist()):
            if score >= min_similarity:
                results.append((keys[i], score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def set_summary(self, session_id: str, summary: str,
                    metadata: Optional[Dict] = None, custom_ttl: Optional[int] = None):
        """Store summary with metadata and custom TTL"""
        self.summary_cache[session_id] = {
            "summary": summary,
            "timestamp": time.time(),
            "metadata": metadata or {},
            "ttl": custom_ttl or self.expiry_seconds
        }
    
    def get_summary(self, session_id: str, update_access: bool = True) -> Optional[Tuple[str, Dict]]:
        """Retrieve summary with optional access time update"""
        item = self.summary_cache.get(session_id)
        if not item:
            return None
        
        now = time.time()
        effective_ttl = item["ttl"] or self.expiry_seconds
        
        # Check expiration
        if effective_ttl and (now - item["timestamp"]) > effective_ttl:
            self.summary_cache.pop(session_id, None)
            return None
        
        # Update access time if requested
        if update_access:
            item["timestamp"] = now
            self.summary_cache[session_id] = item
        
        return item["summary"], item["metadata"]
    
    def clear(self, clear_disk: bool = False):
        """Clear cache with option to remove disk storage"""
        self.embedding_cache.clear()
        self.summary_cache.clear()
        self._init_caches()
        
        if clear_disk and self.cache_path.exists():
            try:
                os.remove(self.cache_path)
                logger.info(f"Removed disk cache at {self.cache_path}")
            except Exception as e:
                logger.error(f"Failed to remove disk cache: {str(e)}")
    
    def clean_expired(self) -> Tuple[int, int]:
        """Remove expired items from both caches"""
        now = time.time()
        emb_expired = summ_expired = 0
        
        # Clean embedding cache
        keys_to_remove = []
        for key, value in self.embedding_cache.items():
            _, ts, _, ttl = value
            effective_ttl = ttl if ttl is not None else self.expiry_seconds
            if effective_ttl and (now - ts) > effective_ttl:
                keys_to_remove.append(key)
        
        emb_expired = len(keys_to_remove)
        for key in keys_to_remove:
            self.embedding_cache.pop(key, None)
        
        # Clean summary cache
        keys_to_remove = []
        for key, item in self.summary_cache.items():
            effective_ttl = item["ttl"] or self.expiry_seconds
            if effective_ttl and (now - item["timestamp"]) > effective_ttl:
                keys_to_remove.append(key)
        
        summ_expired = len(keys_to_remove)
        for key in keys_to_remove:
            self.summary_cache.pop(key, None)
        
        logger.info(f"Cleaned {emb_expired} expired embeddings and {summ_expired} expired summaries")
        return emb_expired, summ_expired
    
    def save_to_disk(self, force: bool = False):
        """Save cache to disk with periodic throttling"""
        if not self.cache_path:
            return
        
        # Throttle disk writes
        last_save = getattr(self, "_last_save", 0)
        if not force and time.time() - last_save < 60:  # 1 minute throttle
            return
        
        try:
            data = {
                "version": self.VERSION,
                "embedding_cache": self.embedding_cache,
                "summary_cache": self.summary_cache,
                "strategy": self.strategy_name,
                "timestamp": time.time()
            }
            
            with open(self.cache_path, "wb") as f:
                pickle.dump(data, f, protocol=self.SERIALIZATION_PROTOCOL)
            
            self._last_save = time.time()
            logger.debug(f"Cache saved to {self.cache_path}")
        except Exception as e:
            logger.error(f"Failed to save cache: {str(e)}")
    
    def _load_from_disk(self):
        """Load cache from disk with version compatibility"""
        if not self.cache_path.exists():
            return
        
        try:
            with open(self.cache_path, "rb") as f:
                data = pickle.load(f)
                
                # Check version compatibility
                if data.get("version") != self.VERSION:
                    logger.warning("Cache version mismatch, not loading")
                    return
                
                self.embedding_cache = data.get("embedding_cache", OrderedDict())
                self.summary_cache = data.get("summary_cache", {})
                
                # Reinitialize strategy with loaded data
                for key in self.embedding_cache:
                    self.strategy.add_item(key, self.embedding_cache[key])
                
                logger.info(f"Loaded cache with {len(self.embedding_cache)} embeddings "
                            f"and {len(self.summary_cache)} summaries")
        except Exception as e:
            logger.error(f"Failed to load cache: {str(e)}")
            # Initialize fresh cache on load failure
            self._init_caches()
    
    def export_metadata(self, path: Path) -> bool:
        """Export cache metadata for analysis"""
        try:
            metadata = {
                "stats": {
                    "embedding_count": len(self.embedding_cache),
                    "summary_count": len(self.summary_cache),
                    "created": time.ctime(),
                    "strategy": self.strategy_name
                },
                "embeddings": [
                    {
                        "key": key,
                        "hash_algo": key.split(":")[0],
                        "timestamp": ts,
                        "ttl": ttl,
                        "metadata": meta
                    }
                    for key, (_, ts, meta, ttl) in self.embedding_cache.items()
                ],
                "summaries": [
                    {
                        "session_id": sid,
                        "timestamp": item["timestamp"],
                        "ttl": item["ttl"],
                        "metadata": item["metadata"]
                    }
                    for sid, item in self.summary_cache.items()
                ]
            }
            
            with open(path, "w") as f:
                json.dump(metadata, f, indent=2)
            
            return True
        except Exception as e:
            logger.error(f"Metadata export failed: {str(e)}")
            return False

if __name__ == "__main__":
    print("\n=== Running Language Cache ===\n")
    printer.status("Init", "Language Cache initialized", "success")

    cache = LanguageCache()
    print(f"Suggestions: {cache}")

    print("\n* * * * * Phase 2 * * * * *\n")
    import uuid
    algorithm = "sha256"
    text= "They call that love from where I come from."
    timestamp=11234534
    embedding=torch.ones(512)
    metadata = {
        "topic": "emotion classification",
        "author": "user",
        "timestamp": time.time(),
        "tags": ["nlp", "torch", "session"],
        "language": "en",
        "summary_type": "final"  # could also be: "draft", "auto", "system"
    }
    custom_ttl=None
    update_access=True
    min_similarity=0.6
    filter_func=None
    top_k=1
    summary= "Explored different syntactic transformations and their impact on model outputs."
    session_id = str(uuid.uuid4())


    embed1 = cache.add_embedding(embedding=embedding, text=text, metadata=metadata, custom_ttl=custom_ttl)
    embed2 = cache.get_embedding(text=text, update_access=update_access)
    similar = cache.most_similar(embedding=embedding, top_k=top_k, min_similarity=min_similarity, filter_func=filter_func)

    summary1 = cache.set_summary(session_id=session_id, summary=summary,
                    metadata=metadata, custom_ttl=custom_ttl)
    summary2 = cache.get_summary(session_id=session_id, update_access=update_access)

    printer.pretty("KEY", cache._hash_key(text=text, algorithm=algorithm), "success")
    printer.pretty("Expired", cache._is_expired(timestamp=timestamp), "success")
    printer.pretty("embed1", embed1, "success")
    printer.pretty("embed2", embed2, "success")
    printer.pretty("sim1", similar, "success")
    printer.pretty("sim2", summary1, "success")
    printer.pretty("sim3", summary2, "success")
    print("\n=== Successfully Ran Language Cache ===\n")
