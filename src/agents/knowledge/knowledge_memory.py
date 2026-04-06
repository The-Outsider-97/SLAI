import json
import math
import os
import threading
import time
import numpy as np

from collections import Counter, defaultdict
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer

from src.agents.knowledge.utils.config_loader import get_config_section, load_global_config
from logs.logger import PrettyPrinter, get_logger

logger = get_logger("Knowledge Memory")
printer = PrettyPrinter


class KnowledgeMemory:
    """
    Local memory container for knowledge-centric agents.
    Focuses on agent-local, context-aware, relevance-weighted memory entries.
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

        # Initialize mutable state before any autoload or persistence activity.
        self._store: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.vectorizer = TfidfVectorizer()
        self.relevance_weights = self._normalize_relevance_weights(
            self.memory_config.get("relevance_weights")
        )

        if self.autoload_on_startup and self.persist_file:
            try:
                self.load(self.persist_file)
                logger.info(f"Memory autoloaded from {self.persist_file}")
            except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
                logger.warning(f"Autoload failed from {self.persist_file}: {exc}")

        logger.info(
            "Knowledge Memory initialized with vectorizer=%s, relevance_mode=%s, embedding_model=%s",
            self.vectorizer,
            self.relevance_mode,
            self.embedding_model,
        )

    def update(
        self,
        key: str,
        value: Any,
        metadata: Optional[dict] = None,
        context: Optional[dict] = None,
        ttl: Optional[int] = None,
    ):
        """
        Store or update a local memory entry.
        """
        if self.log_context_updates and context:
            logger.info(f"Context update for key='{key}': {context}")

        if self.log_inference_events and "inferred" in (metadata or {}):
            logger.info(
                f"Inference event stored: key='{key}', inferred={metadata.get('inferred')}"
            )

        timestamp = time.time()
        base_metadata = {
            "timestamp": timestamp,
            "context": context,
            "expiry_time": timestamp + ttl if ttl is not None else None,
        }
        relevance = self._calculate_relevance(value, context, value_meta=base_metadata) if context else 1.0
        enriched_metadata = {
            **base_metadata,
            "relevance": relevance,
        }

        if metadata:
            enriched_metadata.update(metadata)

        if key not in self._store and len(self._store) >= self.max_entries:
            oldest_key = min(
                self._store.items(),
                key=lambda kv: self._extract_timestamp(kv[1].get("metadata")),
            )[0]
            self._store.pop(oldest_key, None)

        self._store[key] = {"value": value, "metadata": enriched_metadata}

    def save(self, path: str):
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as file_handle:
            json.dump(dict(self._store), file_handle, default=str, ensure_ascii=False, indent=2)

    def load(self, path: str):
        if not os.path.exists(path):
            return

        with open(path, "r", encoding="utf-8") as file_handle:
            raw = json.load(file_handle)

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
            raise ValueError(f"Unexpected data type in {path}: {type(raw)}")

        self._store = loaded_store

    def add_all(self, entries: List[dict]):
        """
        Bulk add knowledge entries, usually rules, into memory.
        Each entry should have a unique 'id' and at least a 'name' or 'description'.
        """
        for entry in entries:
            key = entry.get("id") or entry.get("name")
            if not key:
                logger.warning(f"Skipping entry without ID or name: {entry}")
                continue
            self.update(key=key, value=entry, metadata={"type": "system_rule"})

    def recall(
        self,
        key: Optional[str] = None,
        filters: Optional[dict] = None,
        sort_by: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> List:
        """
        Retrieve entries by key, filters, and relevance.
        """
        now = time.time()
        entries = []

        if key:
            item = self._store.get(key)
            if item and not self._is_expired(item, now):
                entries.append((item["value"], item["metadata"]))
        else:
            for entry in self._store.values():
                if not self._is_expired(entry, now):
                    entries.append((entry["value"], entry["metadata"]))

        if filters:
            entries = [entry for entry in entries if self._apply_filters(entry[1], filters)]

        if sort_by:
            entries.sort(key=lambda entry: entry[1].get(sort_by, 0), reverse=True)

        if self.log_retrieval_hits:
            logger.info(
                f"Retrieved {len(entries)} entries for key='{key}' filters={filters} top_k={top_k}"
            )

        return entries[:top_k] if top_k else entries

    def delete(self, key: str):
        if key in self._store:
            del self._store[key]

    def clear(self):
        self._store.clear()

    def keys(self):
        return list(self._store.keys())

    def get_statistics(self):
        now = time.time()
        total_entries = len(self._store)
        expired_entries = 0
        relevance_values: List[float] = []

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
        }

    def search_values(self, keyword: str) -> List:
        keyword_lower = keyword.lower()
        return [
            (key, value)
            for key, value in self._store.items()
            if keyword_lower in str(value.get("value", "")).lower()
        ]

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

    def _calculate_relevance(
        self, value: Any, context: dict, value_meta: Optional[dict] = None
    ) -> float:
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

    # Helper methods ---------------------------------------------------
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
        """Extract meaningful text from context."""
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
        """Compute cosine similarity between two vectors."""
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        if a_norm == 0.0 or b_norm == 0.0:
            return 0.0
        return float(np.dot(a, b) / (a_norm * b_norm))

    def _fallback_semantic(self, text1: str, text2: str) -> float:
        """TF-IDF/SequenceMatcher fallback when embeddings are unavailable."""
        if not text1 or not text2:
            return 0.0

        try:
            tfidf = self.vectorizer.fit_transform([text1, text2])
            return float((tfidf * tfidf.T).A[0, 1])
        except ValueError as exc:
            logger.debug(f"TF-IDF semantic fallback unavailable; using sequence matcher: {exc}")
            return float(SequenceMatcher(None, text1, text2).ratio())

    def _contextual_term_score(self, value: str, context: str) -> float:
        """Weighted term importance using TF-IDF-ranked context terms."""
        if not value or not context:
            return 0.0

        try:
            vectorizer = TfidfVectorizer(max_features=10)
            tfidf = vectorizer.fit_transform([context, value])
            feature_names = vectorizer.get_feature_names_out()
            if len(feature_names) == 0:
                return 0.0

            context_vector = tfidf[0].toarray().ravel()
            ranked_indexes = context_vector.argsort()[::-1]
            important_terms = [
                feature_names[index] for index in ranked_indexes if context_vector[index] > 0
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
        """Time-based decay using context timestamp."""
        ctx_time = context_meta.get("timestamp", time.time())
        val_time = value_meta.get("timestamp", ctx_time)
        time_diff = abs(ctx_time - val_time)

        half_life_seconds = self.memory_config.get("temporal_half_life_seconds")
        if half_life_seconds is None:
            half_life_seconds = max(1.0, 30 * 86400 * max(float(self.decay_factor), 1e-6))

        return float(math.exp(-time_diff * math.log(2) / half_life_seconds))

    def _structural_similarity(self, dict1: dict, dict2: dict) -> float:
        """Recursive structural similarity for nested dicts."""

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

    def shutdown(self):
        try:
            if self.persist_file:
                self.save(self.persist_file)
                logger.info("Memory saved on shutdown.")
        except OSError as exc:
            logger.warning(f"Failed to save memory on shutdown: {exc}")


if __name__ == "__main__":
    print("\n=== Knowledge Synchronizer Test ===")
    memory = KnowledgeMemory()
    printer.status("Initial sync:", memory)
    print("\n=== Synchronization Test Completed ===\n")
