import os
import re
import yaml
import json
import time
import hashlib
import warnings

from pathlib import Path
from typing import Optional, Any, List, Set
from collections import OrderedDict, defaultdict
from cryptography.fernet import Fernet, InvalidToken

from src.agents.knowledge.utils.knowledge_errors import CacheError
from src.agents.knowledge.utils.config_loader import load_global_config, get_config_section
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Knowledge Cache")
printer = PrettyPrinter

ACTION_PATTERN = re.compile(r"action:(\w+):(.+)", re.IGNORECASE)


class KnowledgeCache:
    """LRU cache with TTL expiration, optional encryption, and semantic hashing."""

    def __init__(self, cache_config: Optional[dict] = None, encryption_key: Optional[bytes | str] = None):
        self.config = load_global_config()
        self.cache_config = cache_config or get_config_section("knowledge_cache") or {}
        self.cache = OrderedDict()

        self.max_size = int(self.cache_config.get("max_size", 1000))
        self.hashing_method = self.cache_config.get("hashing_method", "simhash")
        self.simhash_bits = int(self.cache_config.get("simhash_bits", 64))
        self.tokenizer = self.cache_config.get("tokenizer", "word")
        self.shingle_size = int(self.cache_config.get("shingle_size", 4))
        self.use_tf_weights = bool(self.cache_config.get("use_tf_weights", True))
        self.character_ngram = int(self.cache_config.get("character_ngram", 3))
        self.enable_encryption = bool(self.cache_config.get("enable_encryption", True))
        self.stopwords = self._load_stopwords(self.cache_config.get("stopwords"))

        self._expiration: dict[str, float] = {}
        self._last_cleanup_time = time.time()
        self.cipher = self._initialize_cipher(encryption_key)

        logger.info(
            "Knowledge Cache initialized with max_size=%s hashing=%s encryption=%s",
            self.max_size,
            self.hashing_method,
            self.enable_encryption,
        )

    def __contains__(self, key):
        if key not in self.cache:
            return False
        if self._is_expired(key, time.time()):
            self._evict(key)
            return False
        return True

    def __getitem__(self, key):
        return self.get(key)

    def __len__(self):
        self.cleanup_expired()
        return len(self.cache)

    def get(self, key: str) -> Optional[Any]:
        """Retrieve a value with expiration and LRU handling."""
        current_time = time.time()

        if self._is_expired(key, current_time):
            self._evict(key)
            return None

        if key not in self.cache:
            return None

        self.cache.move_to_end(key)
        value = self.cache[key]

        if self.enable_encryption:
            try:
                return json.loads(self.cipher.decrypt(value).decode("utf-8"))
            except (InvalidToken, json.JSONDecodeError, TypeError) as exc:
                logger.error("Failed to decrypt cached value for key='%s': %s", key, exc)
                raise CacheError("get", key, f"Decryption failed: {exc}") from exc
        return value

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set a value with optional TTL in seconds."""
        current_time = time.time()

        if ttl is not None:
            self._expiration[key] = current_time + ttl
        else:
            self._expiration.pop(key, None)

        if self.enable_encryption:
            serialized = json.dumps(value).encode("utf-8")
            value = self.cipher.encrypt(serialized)

        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value

        if current_time - self._last_cleanup_time > 300:
            self.cleanup_expired()
            self._last_cleanup_time = current_time

        while len(self.cache) > self.max_size:
            evicted_key, _ = self.cache.popitem(last=False)
            self._expiration.pop(evicted_key, None)
            logger.info("Evicted LRU cache key='%s'", evicted_key)

    def delete(self, key: str) -> bool:
        """Delete a cache entry explicitly. Returns True when an entry was removed."""
        existed = key in self.cache or key in self._expiration
        self._evict(key)
        return existed

    def flush_flagged_entries(self) -> int:
        """Remove entries flagged for invalidation by common payload conventions."""
        flagged_keys = []
        for key in list(self.cache.keys()):
            value = self.get(key)
            if value is None:
                continue
            if self._is_flagged_value(value):
                flagged_keys.append(key)

        for key in flagged_keys:
            self._evict(key)

        if flagged_keys:
            logger.info("Flushed %s flagged cache entries", len(flagged_keys))
        return len(flagged_keys)

    def cleanup_expired(self) -> int:
        """Remove all expired items and return the number removed."""
        current_time = time.time()
        expired_keys = [key for key, expiry in list(self._expiration.items()) if expiry <= current_time]
        for key in expired_keys:
            self._evict(key)
        return len(expired_keys)

    def clear(self) -> None:
        self.cache.clear()
        self._expiration.clear()

    def _evict(self, key: str) -> None:
        """Remove key from all data structures."""
        self.cache.pop(key, None)
        self._expiration.pop(key, None)

    def hash_query(self, query: str) -> str:
        """Semantic hashing using SimHash or MD5 with configurable methods."""
        if self.hashing_method == "simhash":
            return self._simhash(query)
        return hashlib.md5(query.encode("utf-8")).hexdigest()

    def _simhash(self, query: str) -> str:
        """Compute a configurable SimHash fingerprint for the query."""
        tokens = self._tokenize(query, self.tokenizer)
        token_weights = self._calculate_token_weights(tokens) if self.use_tf_weights else {token: 1 for token in set(tokens)}
        vector = [0] * self.simhash_bits
        hash_func = hashlib.md5 if self.simhash_bits <= 128 else hashlib.sha256

        for token, weight in token_weights.items():
            token_hash = hash_func(token.encode("utf-8")).digest()
            bit_mask = int.from_bytes(token_hash, byteorder="big")
            for bit_pos in range(self.simhash_bits):
                if bit_mask & (1 << bit_pos):
                    vector[bit_pos] += weight
                else:
                    vector[bit_pos] -= weight

        fingerprint = 0
        for bit_pos, bit_value in enumerate(vector):
            if bit_value > 0:
                fingerprint |= 1 << bit_pos

        return self._to_fixed_length_hex(fingerprint, self.simhash_bits)

    def _calculate_token_weights(self, tokens: List[str]) -> dict[str, int]:
        """Calculate term-frequency weights with normalized stopword filtering."""
        tf = defaultdict(int)
        for token in tokens:
            normalized = token.strip().lower()
            if normalized and normalized not in self.stopwords:
                tf[normalized] += 1
        return dict(tf)

    def _to_fixed_length_hex(self, value: int, num_bits: int) -> str:
        """Convert an integer fingerprint to a zero-padded hex string."""
        num_hex_digits = (num_bits + 3) // 4
        return format(value, f"0{num_hex_digits}x")

    def _tokenize(self, text: str, tokenizer_type: str = "word") -> List[str]:
        """Tokenize text using configurable word, char, or shingle strategies."""
        text = text.lower()

        if tokenizer_type == "word":
            text = re.sub(r"[^\w\s]", "", text)
            return text.split()

        if tokenizer_type == "char":
            n = max(self.character_ngram, 1)
            if len(text) < n:
                return [text] if text else []
            return [text[i : i + n] for i in range(len(text) - n + 1)]

        if tokenizer_type == "shingle":
            n = max(self.shingle_size, 1)
            words = re.sub(r"[^\w\s]", "", text).split()
            if len(words) < n:
                return [" ".join(words)] if words else []
            return [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]

        return [text] if text else []

    def _initialize_cipher(self, encryption_key: Optional[bytes | str]) -> Optional[Fernet]:
        if not self.enable_encryption:
            return None

        key = encryption_key or os.getenv("CACHE_ENCRYPTION_KEY")
        if not key:
            # Generate a persistent key from a known secret (e.g., machine ID) OR just warn and generate a random one.
            # For production, you MUST provide a key. For development, generate a random key (breaks across restarts).
            warnings.warn(
                "No encryption key provided. Generating a random key. "
                "Cache will be unreadable after restart. Set CACHE_ENCRYPTION_KEY for persistence.",
                RuntimeWarning
            )
            key = Fernet.generate_key()
            self.enable_encryption = True  # still enabled

        if isinstance(key, str):
            key = key.encode("utf-8")
        return Fernet(key)

    def _load_stopwords(self, stopwords_config: Optional[Any]) -> Set[str]:
        if stopwords_config is None:
            return set()

        if isinstance(stopwords_config, (list, set, tuple)):
            return self._normalize_stopwords(stopwords_config)

        if isinstance(stopwords_config, str):
            stopwords_path = self._resolve_path(stopwords_config)
            if not stopwords_path.exists():
                raise FileNotFoundError(f"Stopwords file not found: {stopwords_path}")

            suffix = stopwords_path.suffix.lower()
            with open(stopwords_path, "r", encoding="utf-8") as handle:
                if suffix == ".json":
                    data = json.load(handle)
                elif suffix in {".yaml", ".yml"}:
                    data = yaml.safe_load(handle)
                else:
                    data = [line.strip() for line in handle if line.strip()]
            return self._normalize_stopwords(data)

        raise TypeError("stopwords must be a collection of tokens or a path string")

    def _normalize_stopwords(self, stopwords: Any) -> Set[str]:
        if isinstance(stopwords, dict):
            iterable = stopwords.keys()
        elif isinstance(stopwords, str):
            iterable = [stopwords]
        else:
            iterable = stopwords
        return {str(token).strip().lower() for token in iterable if str(token).strip()}

    def _resolve_path(self, path_value: str) -> Path:
        path = Path(path_value)
        if path.is_absolute():
            return path

        candidates = [Path.cwd() / path]
        config_path = self.config.get("__config_path__") if isinstance(self.config, dict) else None
        if config_path:
            config_dir = Path(config_path).resolve().parent
            candidates.append(config_dir / path)
            candidates.append(config_dir.parent / path)
            candidates.append(config_dir.parent.parent / path)

        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    def _is_expired(self, key: str, current_time: float) -> bool:
        expiry = self._expiration.get(key)
        return expiry is not None and expiry <= current_time

    def _is_flagged_value(self, value: Any) -> bool:
        if isinstance(value, dict):
            if value.get("flagged") is True:
                return True
            if str(value.get("status", "")).lower() == "flagged":
                return True
            flags = value.get("flags")
            if isinstance(flags, (list, tuple, set)) and any(str(flag).lower() == "flagged" for flag in flags):
                return True
            if isinstance(flags, dict) and any(bool(v) for k, v in flags.items() if str(k).lower() == "flagged"):
                return True
            return False

        if isinstance(value, str):
            match = ACTION_PATTERN.match(value.strip())
            return bool(match and match.group(1).lower() == "flag")

        return False


if __name__ == "__main__":
    print("\n=== Running Knowledge Cache ===\n")
    printer.status("Init", "Knowledge Cache initialized", "success")

    cache = KnowledgeCache()
    printer.status("Details", f"Cache capacity: {cache.max_size}", "info")

    query = "What are the ethical concerns surrounding artificial intelligence?"

    printer.pretty("HASH", cache.hash_query(query=query), "success")

    print("\n=== Successfully ran Knowledge Cache ===\n")