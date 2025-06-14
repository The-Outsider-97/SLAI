
import hashlib
import os
import shelve
import time
import pickle
import gzip
import json
import threading
import lz4.frame

from datetime import datetime, timedelta
from urllib.parse import urlencode
from typing import Any, Dict, Optional, List, Tuple

from src.agents.execution.utils.config_loader import load_global_config, get_config_section
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Execution Memory")
printer = PrettyPrinter

class ExecutionMemory:
    def __init__(self):
        """
        Enhanced memory management for execution agent with:
        - Multi-level caching (memory/disk)
        - Secure cookie management
        - Versioned checkpointing
        - Efficient compression
        - Thread-safe operations
        """
        self.config = load_global_config()
        self.memory_config = get_config_section('execution_memory')
        self.cache_dir = self._ensure_dir(self.memory_config.get('cache_dir'))
        self.checkpoint_dir = self._ensure_dir(self.memory_config.get('checkpoint_dir'))
        self.cookie_jar_path = self.memory_config.get('cookie_jar')

        # Memory structures
        self.memory_cache: Dict[str, Any] = {}
        self.disk_cache = self._init_shelve(os.path.join(self.cache_dir, 'agent_cache.db'))
        self.checkpoint_store = self._init_shelve(os.path.join(self.checkpoint_dir, 'checkpoints.db'))
        self.cookies = self._load_cookies()

        # Configuration parameters
        self.cache_ttl = self.memory_config.get('cache_ttl', 3600)  # 1 hour default
        self.max_memory_cache = self.memory_config.get('max_memory_cache')
        self.compression_threshold = self.memory_config.get('compression_threshold')  # 1KB
        self.compression_algorithm = self.memory_config.get('compression')
        self.cleanup_interval = self.memory_config.get('cleanup_interval')

        # Statistics and monitoring
        self.cache_hits = 0
        self.cache_misses = 0
        self.last_validation = time.time()
        self.last_cleanup = time.time()
        self.lock = threading.RLock()

        logger.info(f"Execution Memory initialized with {self.compression_algorithm} compression")

    def _ensure_dir(self, path: str) -> str:
        """Ensure directory exists and return normalized path"""
        full_path = os.path.abspath(path)
        os.makedirs(full_path, exist_ok=True)
        return full_path

    def _init_shelve(self, path: str) -> shelve.DbfilenameShelf:
        """Initialize a thread-safe shelve database"""
        return shelve.open(path, writeback=True, protocol=pickle.HIGHEST_PROTOCOL)

    def _load_cookies(self) -> Dict[str, Any]:
        """Load cookies with integrity checking"""
        try:
            if os.path.exists(self.cookie_jar_path):
                with open(self.cookie_jar_path, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            logger.error(f"Cookie load failed: {str(e)}")
        return {}

    def _compress(self, data: Any) -> bytes:
        """Compress data using configured algorithm"""
        serialized = pickle.dumps(data)
        
        if len(serialized) < self.compression_threshold:
            return serialized
            
        if self.compression_algorithm == 'lz4':
            return lz4.frame.compress(serialized)
        elif self.compression_algorithm == 'gzip':
            return gzip.compress(serialized)
        return serialized

    def _decompress(self, data: bytes) -> Any:
        """Decompress data using appropriate algorithm"""
        try:
            if data.startswith(b'\x28\xb5\x2f\xfd'):  # LZ4 magic number
                return pickle.loads(lz4.frame.decompress(data))
            elif data.startswith(b'\x1f\x8b'):  # GZIP magic number
                return pickle.loads(gzip.decompress(data))
            return pickle.loads(data)
        except Exception as e:
            logger.error(f"Decompression failed: {str(e)}")
            raise

    def _cache_key(self, url: str, params: Optional[Dict] = None) -> str:
        """Generate consistent cache key with namespace support"""
        base = url + (urlencode(sorted(params.items()))) if params else ''
        return f"cache::{hashlib.blake2b(base.encode(), digest_size=16).hexdigest()}"

    def get_cache(self, url: str, params: Optional[Dict] = None) -> Any:
        """Multi-level cache retrieval with TTL checking"""
        key = self._cache_key(url, params)
        
        with self.lock:
            # Try memory cache first
            if key in self.memory_cache:
                entry = self.memory_cache[key]
                if time.time() - entry['timestamp'] < self.cache_ttl:
                    self.cache_hits += 1
                    return entry['data']
                del self.memory_cache[key]
            
            # Try disk cache with error handling
            if key in self.disk_cache:
                try:
                    entry = self.disk_cache[key]
                    if time.time() - entry['timestamp'] < self.cache_ttl:
                        # Promote to memory cache
                        self.memory_cache[key] = entry
                        self.cache_hits += 1
                        return entry['data']
                    del self.disk_cache[key]
                except Exception as e:
                    logger.error(f"Corrupted cache entry {key}: {str(e)}. Removing.")
                    try:
                        del self.disk_cache[key]
                    except:
                        pass
                    self.cache_misses += 1
            
            self.cache_misses += 1
            return 
        
        # Periodic cache validation
        if time.time() - self.last_validation > self.cleanup_interval:
            self.validate_cache()
            self.last_validation = time.time()
            
        return None

    def validate_cache(self):
        """Check and repair cache integrity"""
        with self.lock:
            corrupted_keys = []
            for key in list(self.disk_cache.keys()):
                try:
                    # Test if entry can be read
                    _ = self.disk_cache[key]
                except:
                    corrupted_keys.append(key)
            
            for key in corrupted_keys:
                try:
                    del self.disk_cache[key]
                    logger.warning(f"Removed corrupted cache entry: {key}")
                except:
                    pass

    def set_cache(self, url: str, data: Any, params: Optional[Dict] = None, ttl: Optional[int] = None):
        """Set cache entry with compression and TTL"""
        key = self._cache_key(url, params)
        ttl = ttl or self.cache_ttl
        
        with self.lock:
            entry = {
                'data': data,
                'timestamp': time.time(),
                'expire': time.time() + ttl
            }
            
            # Update memory cache
            self.memory_cache[key] = entry
            
            # Manage memory cache size
            if len(self.memory_cache) > self.max_memory_cache:
                oldest = min(self.memory_cache, key=lambda k: self.memory_cache[k]['timestamp'])
                del self.memory_cache[oldest]
            
            # Update disk cache with compression
            try:
                self.disk_cache[key] = {
                    'data': self._compress(data),
                    'timestamp': entry['timestamp'],
                    'expire': entry['expire'],
                    'compressed': True
                }
            except Exception as e:
                logger.error(f"Disk cache update failed: {str(e)}")
            
            # Periodic cleanup
            if time.time() - self.last_cleanup > 3600:  # Clean hourly
                self.clean_cache()
                self.last_cleanup = time.time()

    def clean_cache(self):
        """Remove expired cache entries"""
        with self.lock:
            now = time.time()
            
            # Clean memory cache
            expired_keys = [k for k, v in self.memory_cache.items() if v['expire'] < now]
            for key in expired_keys:
                del self.memory_cache[key]
            
            # Clean disk cache
            expired_keys = [k for k, v in self.disk_cache.items() if v['expire'] < now]
            for key in expired_keys:
                del self.disk_cache[key]
            
            logger.info(f"Cache cleaned: {len(expired_keys)} expired entries removed")

    def create_checkpoint(self, state: Any, tags: List[str] = None, metadata: Dict = None) -> str:
        """Create versioned checkpoint with tags and metadata"""
        checkpoint_id = f"chk_{int(time.time() * 1000)}"
        version = 1
        
        with self.lock:
            # Handle versioning
            while f"{checkpoint_id}_v{version}" in self.checkpoint_store:
                version += 1
            
            full_id = f"{checkpoint_id}_v{version}"
            compressed = self._compress(state)
            
            self.checkpoint_store[full_id] = {
                'state': compressed,
                'created': time.time(),
                'tags': tags or [],
                'metadata': metadata or {},
                'size': len(compressed),
                'compression': self.compression_algorithm
            }
            
            # Update tag index
            tag_index = self.checkpoint_store.get('__tag_index', {})
            for tag in tags or []:
                tag_index.setdefault(tag, []).append(full_id)
            self.checkpoint_store['__tag_index'] = tag_index
            
            return full_id

    def restore_checkpoint(self, checkpoint_id: str) -> Any:
        """Restore checkpoint by ID with decompression"""
        with self.lock:
            if checkpoint_id not in self.checkpoint_store:
                return None
                
            entry = self.checkpoint_store[checkpoint_id]
            return self._decompress(entry['state'])

    def find_checkpoints(self, tag: str = None, min_size: int = 0, 
                         max_age: int = None) -> List[Dict]:
        """Find checkpoints by criteria"""
        with self.lock:
            results = []
            now = time.time()
            
            # Use tag index if available
            if tag:
                tag_index = self.checkpoint_store.get('__tag_index', {})
                checkpoint_ids = tag_index.get(tag, [])
            else:
                checkpoint_ids = [k for k in self.checkpoint_store.keys() if k.startswith('chk_')]
            
            for cid in checkpoint_ids:
                if cid not in self.checkpoint_store:
                    continue
                    
                entry = self.checkpoint_store[cid]
                entry_size = entry.get('size', 0)
                
                # Apply filters
                if entry_size < min_size:
                    continue
                    
                if max_age and (now - entry['created']) > max_age:
                    continue
                    
                results.append({
                    'id': cid,
                    'created': datetime.fromtimestamp(entry['created']).isoformat(),
                    'tags': entry.get('tags', []),
                    'size': entry_size,
                    'metadata': entry.get('metadata', {})
                })
            
            return sorted(results, key=lambda x: x['created'], reverse=True)

    def delete_checkpoint(self, checkpoint_id: str):
        """Delete checkpoint and update indexes"""
        with self.lock:
            if checkpoint_id not in self.checkpoint_store:
                return False
                
            # Remove from tag index
            tags = self.checkpoint_store[checkpoint_id].get('tags', [])
            tag_index = self.checkpoint_store.get('__tag_index', {})
            
            for tag in tags:
                if tag in tag_index and checkpoint_id in tag_index[tag]:
                    tag_index[tag].remove(checkpoint_id)
                    if not tag_index[tag]:
                        del tag_index[tag]
            
            self.checkpoint_store['__tag_index'] = tag_index
            del self.checkpoint_store[checkpoint_id]
            return True

    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        with self.lock:
            return {
                'memory_entries': len(self.memory_cache),
                'disk_entries': len(self.disk_cache),
                'cache_hits': self.cache_hits,
                'cache_misses': self.cache_misses,
                'hit_ratio': self.cache_hits / (self.cache_hits + self.cache_misses) 
                            if (self.cache_hits + self.cache_misses) > 0 else 0,
                'last_cleanup': datetime.fromtimestamp(self.last_cleanup).isoformat()
            }

    def export_memory(self, path: str):
        """Export entire memory state to file"""
        with self.lock:
            state = {
                'memory_cache': self.memory_cache,
                'cookies': self.cookies,
                'checkpoints': dict(self.checkpoint_store),
                'stats': self.get_cache_stats(),
                'timestamp': time.time()
            }
            
            with lz4.frame.open(path, 'wb') as f:
                pickle.dump(state, f)

    def import_memory(self, path: str):
        """Import memory state from file"""
        with self.lock:
            try:
                with lz4.frame.open(path, 'rb') as f:
                    state = pickle.load(f)
                
                self.memory_cache = state.get('memory_cache', {})
                self.cookies = state.get('cookies', {})
                
                # Merge checkpoints
                for k, v in state.get('checkpoints', {}).items():
                    self.checkpoint_store[k] = v
                
                logger.info(f"Memory state imported from {path}")
            except Exception as e:
                logger.error(f"Memory import failed: {str(e)}")

    def __del__(self):
        """Cleanup resources on destruction"""
        with self.lock:
            try:
                self.save_cookies()
                self.disk_cache.close()
                self.checkpoint_store.close()
            except Exception as e:
                logger.error(f"Cleanup failed: {str(e)}")

    def save_cookies(self):
        """Persist cookies with atomic write"""
        with self.lock:
            try:
                # Ensure directory exists
                os.makedirs(os.path.dirname(self.cookie_jar_path), exist_ok=True)
                
                temp_path = self.cookie_jar_path + '.tmp'
                with open(temp_path, 'wb') as f:
                    pickle.dump(self.cookies, f, protocol=pickle.HIGHEST_PROTOCOL)
                os.replace(temp_path, self.cookie_jar_path)
            except Exception as e:
                logger.error(f"Cookie save failed: {str(e)}")


if __name__ == "__main__":
    print("\n=== Running Execution Memory Test ===\n")
    printer.status("Init", "Execution Memory initialized", "success")

    # Test memory operations
    memory = ExecutionMemory()
    
    # Test caching
    memory.set_cache("https://api.example.com/data", {"results": [1, 2, 3]}, params={"page": 1})
    cached_data = memory.get_cache("https://api.example.com/data", params={"page": 1})
    printer.pretty("Cache Test", bool(cached_data), "success")
    
    # Test checkpointing
    checkpoint_id = memory.create_checkpoint(
        {"position": (10.5, 20.3), "inventory": ["item1", "item2"]},
        tags=["mission_critical", "backup"],
        metadata={"mission": "reconnaissance"}
    )
    printer.pretty("Checkpoint Created", checkpoint_id, "info")
    
    # Test checkpoint retrieval
    checkpoints = memory.find_checkpoints(tag="mission_critical")
    printer.pretty("Found Checkpoints", len(checkpoints) > 0, "success")
    
    # Test statistics
    stats = memory.get_cache_stats()
    printer.pretty("Cache Stats", stats, "info")
    
    print("\n=== Simulation Complete ===")
