
import json
import math
import os
import threading
import time
import hashlib
import numpy as np

from collections import Counter, OrderedDict, defaultdict
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer

from .utils.config_loader import get_config_section, load_global_config
from .utils.knowledge_helpers import *
from .utils.knowledge_errors import *
from logs.logger import PrettyPrinter, get_logger # pyright: ignore[reportMissingImports]

logger = get_logger("Knowledge Memory")
printer = PrettyPrinter()


class KnowledgeMemory:
    """
    Local memory container for knowledge-centric agents.
    Focuses on agent-local, context-aware, relevance-weighted memory entries.
    Production‑ready with thread safety, query caching, and detailed metrics.
    """

    _embedding_model_cache: Dict[str, Any] = {}
    _embedding_model_lock = threading.Lock()

    def __init__(self):
        self.config = load_global_config()
        self.memory_config = get_config_section("knowledge_memory")
        self.max_entries = self.memory_config.get("max_entries", 10000)
        self.cache_size = self.memory_config.get("cache_size", 1000)
        self.relevance_mode = self.memory_config.get("relevance_mode", "hybrid")
        self.similarity_threshold = self.memory_config.get("similarity_threshold", 0.2)
        self.decay_factor = self.memory_config.get("decay_factor", 0.8)
        self.context_window = self.memory_config.get("context_window", 3)
        self.enable_ontology_expansion = self.memory_config.get("enable_ontology_expansion", True)
        self.enable_rule_engine = self.memory_config.get("enable_rule_engine", True)
        self.auto_discover_rules = self.memory_config.get("auto_discover_rules", True)
        self.min_rule_support = self.memory_config.get("min_rule_support", 0.1)
        self.use_embedding_fallback = self.memory_config.get("use_embedding_fallback", True)
        self.embedding_model = self.memory_config.get("embedding_model", "all-MiniLM-L6-v2")
        self.knowledge_dir = self.memory_config.get("knowledge_dir")
        self.autoload_on_startup = self.memory_config.get("autoload_on_startup", False)
        self.log_retrieval_hits = self.memory_config.get("log_retrieval_hits", False)
        self.log_context_updates = self.memory_config.get("log_context_updates", False)
        self.log_inference_events = self.memory_config.get("log_inference_events", False)
        self.persist_file = self.memory_config.get("persist_file")

        # Query cache TTL (seconds) – default 5 minutes
        self.query_cache_ttl = self.memory_config.get("query_cache_ttl", 300)
        self.enable_query_cache = self.memory_config.get("enable_query_cache", True)

        # Initialize mutable state before any autoload or persistence activity.
        self._store: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self._store_lock = threading.RLock()                 # Protects all store operations

        # Query result cache: key -> (timestamp, result)
        self._query_cache: OrderedDict[str, Tuple[float, Any]] = OrderedDict()
        self._cache_lock = threading.RLock()

        self.vectorizer = TfidfVectorizer()
        self.relevance_weights = self._normalize_relevance_weights(self.memory_config.get("relevance_weights"))

        # Metrics
        self._metrics: Dict[str, Any]  = {
            "queries_total": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "expired_removed_total": 0,
            "store_evictions_total": 0,
        }

        if self.autoload_on_startup and self.persist_file:
            try:
                self.load(self.persist_file)
                logger.info(f"Memory autoloaded from {self.persist_file}")
            except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
                logger.warning(f"Autoload failed from {self.persist_file}: {exc}")

        logger.info("Knowledge Memory initialized with vectorizer=%s, relevance_mode=%s, embedding_model=%s",
                    self.vectorizer, self.relevance_mode, self.embedding_model)

    # ----------------------------------------------------------------------
    # Public API – thread‑safe
    # ----------------------------------------------------------------------
    def update(self, key: str, value: Any, metadata: Optional[dict] = None,
               context: Optional[dict] = None, ttl: Optional[int] = None):
        """
        Store or update a local memory entry.

        Raises:
            MemoryUpdateError: If the key is invalid or storage fails.
        """
        if not key or not isinstance(key, str):
            raise MemoryUpdateError(key=str(key), value=value, error_details="Key must be a non‑empty string")

        if self.log_context_updates and context:
            logger.info(f"Context update for key='{key}': {context}")

        incoming_metadata = metadata or {}

        if self.log_inference_events and (
            incoming_metadata.get("type") == "inferred_fact"
            or incoming_metadata.get("inferred") is True
        ):
            logger.info(
                "Inference event stored: key='%s', "
                "confidence=%s, sector=%s",
                key,
                incoming_metadata.get("confidence"),
                incoming_metadata.get("sector"),
            )

        timestamp = time.time()
        base_metadata = {
            "timestamp": timestamp,
            "context": context,
            "expiry_time": timestamp + ttl if ttl is not None else None,
        }
        relevance = self._calculate_relevance(value, context, value_meta=base_metadata) if context else 1.0
        enriched_metadata = {**base_metadata, "relevance": relevance}
        if metadata:
            enriched_metadata.update(metadata)

        with self._store_lock:
            # Evict oldest if at capacity
            if key not in self._store and len(self._store) >= self.max_entries:
                oldest_key = min(
                    self._store.items(),
                    key=lambda kv: self._extract_timestamp(kv[1].get("metadata")),
                )[0]
                self._store.pop(oldest_key, None)
                self._metrics["store_evictions_total"] += 1

            self._store[key] = {"value": value, "metadata": enriched_metadata}

        # Invalidate query cache – any update may change recall results
        self._invalidate_query_cache()

    def save(self, path: str):
        """
        Persist memory to a JSON file.

        Raises:
            OSError: If the file cannot be written.
        """
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with self._store_lock:
            data = dict(self._store)
        try:
            with open(path, "w", encoding="utf-8") as file_handle:
                json.dump(data, file_handle, default=str, ensure_ascii=False, indent=2)
        except OSError as exc:
            logger.error(f"Failed to save memory to {path}: {exc}")
            raise

    def load(self, path: str):
        """
        Load memory from a JSON file.

        Raises:
            InvalidDocumentError: If the file format is invalid.
            OSError: If the file cannot be read.
        """
        if not os.path.exists(path):
            logger.warning(f"Memory file {path} does not exist; skipping load.")
            return

        try:
            with open(path, "r", encoding="utf-8") as file_handle:
                raw = json.load(file_handle)
        except (OSError, json.JSONDecodeError) as exc:
            logger.error(f"Failed to load memory from {path}: {exc}")
            raise InvalidDocumentError(
                document=path,
                reason=f"Invalid JSON or file error: {exc}"
            ) from exc

        loaded_store: Dict[str, Dict[str, Any]] = defaultdict(dict)

        if isinstance(raw, dict):
            for key, entry in raw.items():
                if not isinstance(entry, dict):
                    logger.warning(f"Skipping malformed entry for key='{key}' during load")
                    continue
                loaded_store[key] = {
                    "value": entry.get("value"),
                    "metadata": entry.get("metadata", {}),
                }
        elif isinstance(raw, list):
            for entry in raw:
                if not isinstance(entry, dict) or "key" not in entry:
                    logger.warning(f"Skipping malformed list entry during load: {entry}")
                    continue
                key = entry["key"]
                loaded_store[key] = {
                    "value": entry.get("value"),
                    "metadata": entry.get("metadata", {}),
                }
        else:
            raise InvalidDocumentError(
                document=path,
                reason=f"Unexpected data type: {type(raw)}"
            )

        with self._store_lock:
            self._store = loaded_store
        self._invalidate_query_cache()
        logger.info(f"Loaded {len(loaded_store)} entries from {path}")

    def add_all(self, entries: List[dict]):
        """
        Bulk add knowledge entries, usually rules, into memory.
        Each entry should have a unique 'id' or 'name'.

        Raises:
            MemoryUpdateError: If an entry lacks an identifier.
        """
        for entry in entries:
            key = entry.get("id") or entry.get("name")
            if not key:
                raise MemoryUpdateError(
                    key="unknown",
                    value=entry,
                    error_details="Entry missing 'id' or 'name'"
                )
            self.update(key=key, value=entry, metadata={"type": "system_rule"})

    def recall(self, key: Optional[str] = None, filters: Optional[dict] = None,
               sort_by: Optional[str] = None, top_k: Optional[int] = None, use_cache: bool = True,) -> List:
        """
        Retrieve entries by key, filters, and relevance.

        Args:
            key: Specific key to retrieve.
            filters: Dict of metadata filters (exact or callable).
            sort_by: Metadata key to sort by (descending).
            top_k: Limit number of results.
            use_cache: If True, use query result cache.

        Raises:
            RetrievalError: If the retrieval fails due to invalid filters.
        """
        self._metrics["queries_total"] += 1

        # Build cache key
        cache_key = None
        if use_cache and self.enable_query_cache:
            # Deterministic key based on parameters
            key_part = key or "all"
            filters_part = safe_json_dumps(filters) if filters else "none"
            sort_part = sort_by or "none"
            top_part = str(top_k or "all")
            cache_key = f"recall_{key_part}_{filters_part}_{sort_part}_{top_part}"
            cache_key = hashlib.sha256(cache_key.encode()).hexdigest()[:16]

        # Check cache
        if cache_key:
            with self._cache_lock:
                if cache_key in self._query_cache:
                    timestamp, cached_result = self._query_cache[cache_key]
                    if (time.time() - timestamp) < self.query_cache_ttl:
                        # Move to end (LRU)
                        self._query_cache.move_to_end(cache_key)
                        self._metrics["cache_hits"] += 1
                        if self.log_retrieval_hits:
                            logger.debug(f"Query cache hit for key {cache_key}")
                        return cached_result
                    else:
                        # Expired – remove
                        self._query_cache.pop(cache_key, None)
                else:
                    self._metrics["cache_misses"] += 1

        # Perform actual recall
        now = time.time()
        entries = []

        with self._store_lock:
            if key:
                item = self._store.get(key)
                if item and not self._is_expired(item, now):
                    entries.append((item["value"], item["metadata"]))
            else:
                for entry in self._store.values():
                    if not self._is_expired(entry, now):
                        entries.append((entry["value"], entry["metadata"]))

        # Filter
        if filters:
            try:
                entries = [entry for entry in entries if self._apply_filters(entry[1], filters)]
            except Exception as exc:
                raise RetrievalError(
                    query=str(filters),
                    reason=f"Filter application failed: {exc}",
                    retrieval_mode="filtered"
                ) from exc

        # Sort
        if sort_by:
            entries.sort(key=lambda entry: entry[1].get(sort_by, 0), reverse=True)

        # Slice
        result = entries[:top_k] if top_k else entries

        # Cache result (if caching enabled)
        if cache_key and self.enable_query_cache:
            with self._cache_lock:
                # Evict oldest if cache exceeds size
                if len(self._query_cache) >= self.cache_size:
                    self._query_cache.popitem(last=False)  # LRU eviction
                self._query_cache[cache_key] = (time.time(), result)
                self._query_cache.move_to_end(cache_key)

        if self.log_retrieval_hits:
            logger.info(
                f"Retrieved {len(result)} entries for key='{key}' filters={filters} top_k={top_k}"
            )

        return result

    def delete(self, key: str):
        with self._store_lock:
            if key in self._store:
                del self._store[key]
                self._invalidate_query_cache()

    def clear(self):
        with self._store_lock:
            self._store.clear()
        self._invalidate_query_cache()

    def keys(self):
        with self._store_lock:
            return list(self._store.keys())

    def get_statistics(self):
        now = time.time()
        total_entries = 0
        expired_entries = 0
        relevance_values: List[float] = []

        with self._store_lock:
            total_entries = len(self._store)
            for entry in self._store.values():
                metadata = entry.get("metadata", {})
                relevance = metadata.get("relevance")
                if isinstance(relevance, (int, float, np.floating)):
                    relevance_values.append(float(relevance))
                if self._is_expired(entry, now):
                    expired_entries += 1

        avg_relevance = float(np.mean(relevance_values)) if relevance_values else 0.0

        return {
            "total_entries": total_entries,
            "active_entries": total_entries - expired_entries,
            "avg_relevance": avg_relevance,
            "expired": expired_entries,
            "query_cache_size": len(self._query_cache),
            "metrics": self._metrics.copy(),
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Return detailed performance metrics."""
        with self._store_lock:
            total = len(self._store)
        with self._cache_lock:
            cache_size = len(self._query_cache)
        metrics = self._metrics.copy()
        metrics["store_size"] = total
        metrics["cache_size"] = cache_size
        if metrics["queries_total"] > 0:
            metrics["hit_rate"] = metrics["cache_hits"] / metrics["queries_total"]
        else:
            metrics["hit_rate"] = 0.0
        return metrics

    def cleanup_expired(self) -> int:
        """
        Remove all expired entries from the store.

        Returns:
            Number of removed entries.
        """
        now = time.time()
        removed = 0
        with self._store_lock:
            expired_keys = [
                key for key, entry in self._store.items()
                if self._is_expired(entry, now)
            ]
            for key in expired_keys:
                del self._store[key]
                removed += 1
            self._metrics["expired_removed_total"] += removed
        if removed:
            self._invalidate_query_cache()
            logger.info(f"Removed {removed} expired entries.")
        return removed

    def search_values(self, keyword: str) -> List:
        keyword_lower = keyword.lower()
        with self._store_lock:
            return [
                (key, value)
                for key, value in self._store.items()
                if keyword_lower in str(value.get("value", "")).lower()
            ]

    def shutdown(self):
        try:
            if self.persist_file:
                self.save(self.persist_file)
                logger.info("Memory saved on shutdown.")
        except OSError as exc:
            logger.warning(f"Failed to save memory on shutdown: {exc}")

    # ----------------------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------------------
    def _invalidate_query_cache(self):
        """Clear the query cache (called after mutations)."""
        with self._cache_lock:
            self._query_cache.clear()

    def _is_expired(self, entry: dict, now: float) -> bool:
        expiry = entry.get("metadata", {}).get("expiry_time")
        return expiry is not None and expiry < now

    def _apply_filters(self, metadata: dict, filters: dict) -> bool:
        for filter_key, expected in filters.items():
            actual = metadata.get(filter_key)

            if callable(expected):
                if not expected(actual):
                    return False
                continue

            if isinstance(actual, (list, tuple, set)):
                if isinstance(expected, (list, tuple, set)):
                    if not set(expected).issubset(set(actual)):
                        return False
                elif expected not in actual:
                    return False
                continue

            if isinstance(expected, (list, tuple, set)):
                if actual not in expected:
                    return False
                continue

            if actual != expected:
                return False

        return True

    def _calculate_relevance(self, value: Any, context: dict, value_meta: Optional[dict] = None) -> float:
        """Comprehensive relevance scoring with multiple dimensions."""
        val_str = str(value)
        ctx_str = self._context_to_text(context)
        value_meta = value_meta or {}

        scores = {
            "semantic": 0.0,
            "contextual": 0.0,
            "temporal": 0.0,
            "structural": 0.0,
        }

        scores["semantic"] = self._semantic_similarity(val_str, ctx_str)
        scores["contextual"] = self._contextual_term_score(val_str, ctx_str)
        scores["temporal"] = self._temporal_relevance(
            value_meta=value_meta,
            context_meta=context if isinstance(context, dict) else {},
        )

        if isinstance(value, dict) and isinstance(context, dict):
            scores["structural"] = self._structural_similarity(value, context)

        total_score = sum(
            scores[dimension] * self.relevance_weights[dimension] for dimension in scores
        )

        if not np.isfinite(total_score):
            logger.warning("Non-finite relevance score produced; returning neutral fallback")
            return 0.5

        return float(min(max(total_score, 0.0), 1.0))

    # ----------------------------------------------------------------------
    # Helper methods (unchanged from original, with minor fixes)
    # ----------------------------------------------------------------------
    def _normalize_relevance_weights(
        self, weights: Optional[Dict[str, float]]
    ) -> Dict[str, float]:
        defaults = {
            "semantic": 0.4,
            "contextual": 0.3,
            "temporal": 0.2,
            "structural": 0.1,
        }
        if not isinstance(weights, dict):
            return defaults

        merged = defaults.copy()
        for key in defaults:
            value = weights.get(key)
            if isinstance(value, (int, float)):
                merged[key] = float(value)

        total = sum(merged.values())
        if total <= 0:
            return defaults

        return {key: value / total for key, value in merged.items()}

    def _context_to_text(self, context: dict) -> str:
        if isinstance(context, dict):
            return " ".join(f"{key}={value}" for key, value in context.items())
        return str(context)

    @classmethod
    def _get_or_create_embedding_model(cls, model_name: str):
        with cls._embedding_model_lock:
            if model_name in cls._embedding_model_cache:
                return cls._embedding_model_cache[model_name]

            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                logger.info(f"sentence-transformers unavailable; semantic fallback will be used: {exc}")
                cls._embedding_model_cache[model_name] = None
                return None

            try:
                model = SentenceTransformer(model_name)
            except (OSError, RuntimeError, ValueError) as exc:
                logger.warning(
                    f"Failed to initialize embedding model '{model_name}'; semantic fallback will be used: {exc}"
                )
                model = None

            cls._embedding_model_cache[model_name] = model
            return model

    def _get_embedding_model(self):
        if self.relevance_mode not in {"embedding", "hybrid"}:
            return None
        return self._get_or_create_embedding_model(self.embedding_model)

    def _semantic_similarity(self, value_text: str, context_text: str) -> float:
        if not value_text or not context_text:
            return 0.0

        if self.relevance_mode == "tfidf":
            return self._fallback_semantic(value_text, context_text)

        model = self._get_embedding_model()
        if model is not None:
            try:
                emb_val = np.asarray(model.encode(value_text))
                emb_ctx = np.asarray(model.encode(context_text))
                return self._cosine_sim(emb_val, emb_ctx)
            except (RuntimeError, TypeError, ValueError) as exc:
                logger.warning(
                    f"Embedding similarity failed for model '{self.embedding_model}'; falling back: {exc}"
                )

        if self.use_embedding_fallback or self.relevance_mode == "tfidf":
            return self._fallback_semantic(value_text, context_text)
        return 0.0

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        if a_norm == 0.0 or b_norm == 0.0:
            return 0.0
        return float(np.dot(a, b) / (a_norm * b_norm))

    def _fallback_semantic(self, text1: str, text2: str) -> float:
        if not text1 or not text2:
            return 0.0
        try:
            tfidf = self.vectorizer.fit_transform([text1, text2])
            similarity = (tfidf * tfidf.transpose()).A[0, 1] # type: ignore
            return float(similarity)
        except ValueError as exc:
            logger.debug(f"TF-IDF semantic fallback unavailable; using sequence matcher: {exc}")
            return float(SequenceMatcher(None, text1, text2).ratio())

    def _contextual_term_score(self, value: str, context: str) -> float:
        if not value or not context:
            return 0.0
        try:
            vectorizer = TfidfVectorizer(max_features=10)
            tfidf = vectorizer.fit_transform([context, value])
            feature_names = vectorizer.get_feature_names_out()
            if len(feature_names) == 0:
                return 0.0
            context_vector = tfidf.getrow(0).toarray().ravel()
            ranked_indexes = context_vector.argsort()[::-1]
            important_terms = [
                str(feature_names[index]) for index in ranked_indexes if context_vector[index] > 0
            ]
            if not important_terms:
                return 0.0
            value_counts = Counter(value.lower().split())
            total = sum(value_counts.get(term, 0) for term in important_terms)
            return float(total / len(important_terms))
        except (TypeError, ValueError) as exc:
            logger.debug(f"Contextual term score unavailable: {exc}")
            return 0.0

    def _temporal_relevance(self, value_meta: dict, context_meta: dict) -> float:
        ctx_time = context_meta.get("timestamp", time.time())
        val_time = value_meta.get("timestamp", ctx_time)
        time_diff = abs(ctx_time - val_time)

        half_life_seconds = self.memory_config.get("temporal_half_life_seconds")
        if half_life_seconds is None:
            half_life_seconds = max(1.0, 30 * 86400 * max(float(self.decay_factor), 1e-6))

        return float(math.exp(-time_diff * math.log(2) / half_life_seconds))

    def _structural_similarity(self, dict1: dict, dict2: dict) -> float:
        def compare(a, b):
            if isinstance(a, dict) and isinstance(b, dict):
                keys = set(a.keys()) | set(b.keys())
                if not keys:
                    return 1.0
                return sum(compare(a.get(k), b.get(k)) for k in keys) / len(keys)
            if isinstance(a, list) and isinstance(b, list):
                if not a and not b:
                    return 1.0
                return sum(compare(x, y) for x, y in zip(a, b)) / max(len(a), len(b), 1)
            return 1.0 if a == b else 0.0

        return float(compare(dict1, dict2))

    def _extract_timestamp(self, metadata: Optional[dict]) -> float:
        if not isinstance(metadata, dict):
            return float("inf")
        timestamp = metadata.get("timestamp")
        return float(timestamp) if isinstance(timestamp, (int, float)) else float("inf")


if __name__ == "__main__":
    print("\n=== Knowledge Memory Test ===\n")
    memory = KnowledgeMemory()

    # 1. Update & recall
    memory.update("key1", "value1", metadata={"type": "test"}, ttl=2)
    memory.update("key2", {"nested": "data"}, metadata={"type": "test", "score": 0.9})
    print("Recall all:", memory.recall())
    print("Recall key1:", memory.recall(key="key1"))

    # 2. Filters & sorting
    filtered = memory.recall(filters={"type": "test"}, sort_by="score", top_k=1)
    print("Filtered + sorted:", filtered)

    # 3. Query cache (call twice to see hit)
    res1 = memory.recall(filters={"type": "test"})
    res2 = memory.recall(filters={"type": "test"})  # should hit cache
    print("Cache hit? (metrics):", memory.get_metrics()["cache_hits"])

    # 4. Expiry cleanup
    time.sleep(2.1)
    removed = memory.cleanup_expired()
    print(f"Removed {removed} expired entries")

    # 5. Delete & clear
    memory.delete("key1")
    memory.clear()
    print("After clear, entries:", len(memory.keys()))

    # 6. Save/load (temp file)
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        tmp_path = f.name
    memory.update("persist", "data")
    memory.save(tmp_path)
    new_memory = KnowledgeMemory()
    new_memory.autoload_on_startup = False  # avoid auto-load
    new_memory.load(tmp_path)
    print("Loaded entry:", new_memory.recall(key="persist"))
    os.unlink(tmp_path)

    # 7. Metrics
    print("Final metrics:", memory.get_metrics())
    print("\n=== Test Completed ===\n")