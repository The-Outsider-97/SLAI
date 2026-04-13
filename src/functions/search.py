"""Smarter search utilities with pluggable analyzers and caching."""

from __future__ import annotations

import math
import json
import pickle
import time
import threading

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence, Set, Tuple, Union
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from .utils.config_loader import load_global_config, get_config_section
from .utils.inverted_index import InvertedIndex, SearchAnalyzer, BM25Scorer
from .utils.functions_error import IndexLoadError, IndexSaveError, SearchError
from .functions_memory import TTLCache
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Search Engine")
printer = PrettyPrinter


# ----------------------------------------------------------------------
# Built‑in analyzers (simple, stem, stopword)
# ----------------------------------------------------------------------
class BasicAnalyzer:
    """Basic tokenizer: lowercases, splits on non‑alnum, discards empty tokens."""
    def analyze(self, text: str) -> List[str]:
        cleaned = "".join(c.lower() if c.isalnum() else " " for c in text)
        return [token for token in cleaned.split() if token]


class StemAnalyzer(BasicAnalyzer):
    """Lightweight suffix‑stripping stemmer (ing, ed, ly, es, s)."""
    def analyze(self, text: str) -> List[str]:
        tokens = super().analyze(text)
        suffixes = ("ing", "edly", "ed", "ly", "es", "s")
        stemmed = []
        for token in tokens:
            stem = token
            for suffix in suffixes:
                if stem.endswith(suffix) and len(stem) > len(suffix) + 2:
                    stem = stem[:-len(suffix)]
                    break
            stemmed.append(stem)
        return stemmed


class StopwordAnalyzer(BasicAnalyzer):
    """Removes common stopwords loaded from a JSON file or fallback sets."""
    def __init__(self, language: str = "en", stopwords_path: Optional[str] = None):
        super().__init__()
        self.language = language
        self._stopwords: set[str] = set()
        self._load_stopwords(stopwords_path)

    def _load_stopwords(self, path: Optional[str] = None):
        if path is None:
            config = get_config_section("language_aware")
            path = config.get("stopwords")
        if path and Path(path).exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._stopwords = set(data.get("Stopword", []))
                logger.debug(f"Loaded {len(self._stopwords)} stopwords from {path}")
            except Exception as e:
                logger.error(f"Failed to load stopwords: {e}")
                self._stopwords = self._fallback_stopwords()
        else:
            logger.warning(f"Stopwords file not found at {path}, using fallback")
            self._stopwords = self._fallback_stopwords()

    def _fallback_stopwords(self) -> set[str]:
        fallback = {
            "en": {"the", "a", "an", "and", "or", "to", "of", "for", "in", "on"},
            "es": {"el", "la", "los", "las", "y", "o", "de", "para", "en"},
            "nl": {"de", "het", "een", "en", "of", "voor", "in", "op"},
        }
        return fallback.get(self.language, set())

    def analyze(self, text: str) -> List[str]:
        tokens = super().analyze(text)
        return [t for t in tokens if t not in self._stopwords]


# ----------------------------------------------------------------------
# Search result wrapper
# ----------------------------------------------------------------------
class SearchResult:
    """Encapsulates a search result with score and explanation."""
    def __init__(self, item: Dict[str, Any], score: float, reason: str):
        self.item = item
        self.score = score
        self.reason = reason

    def __repr__(self) -> str:
        return f"SearchResult(score={self.score:.4f}, reason={self.reason})"


# ----------------------------------------------------------------------
# Main search engine
# ----------------------------------------------------------------------
class SearchEngine:
    """
    Production search engine with:
    - Pluggable analyzers
    - Persistent inverted index (save/load)
    - TTL‑based result caching
    - Thread‑safe operations
    - Configuration via YAML
    """

    def __init__(self, fields: List[str],
        analyzer: Optional[SearchAnalyzer] = None,
        cache_ttl_seconds: int = 120,
        index_persistence_path: Optional[Union[str, Path]] = None,
        default_fuzzy_threshold: float = 0.8,
    ):
        """
        Args:
            fields: List of document field names to index.
            analyzer: Tokenizer/normalizer. Defaults to BasicAnalyzer.
            cache_ttl_seconds: Time‑to‑live for search result cache.
            index_persistence_path: If provided, index will be loaded from/saved to this path.
            default_fuzzy_threshold: Minimum similarity ratio (0..1) to add fuzzy bonus.
        """
        config = get_config_section("search_engine")
        bm25_config = get_config_section("bm25")

        if not fields:
            raise ValueError("fields cannot be empty")

        self.fields = fields
        self.analyzer = analyzer or BasicAnalyzer()
        self.scorer = BM25Scorer(
            k1=bm25_config.get("k1", 1.2),
            b=bm25_config.get("b", 0.75),
        )
        self.cache_ttl = cache_ttl_seconds or config.get("cache_ttl_seconds", 120)
        self.default_fuzzy_threshold = default_fuzzy_threshold

        # Core index
        self.index = InvertedIndex(self.analyzer, self.scorer)

        # Cache
        max_cache_size = config.get("max_size", 2048)
        self._cache: TTLCache[List[SearchResult]] = TTLCache(
            max_size=max_cache_size,
            ttl_seconds=self.cache_ttl,
        )
        self._lock = threading.RLock()

        # Persistence
        self.index_path = Path(index_persistence_path) if index_persistence_path else None
        self._auto_save = False
        if self.index_path:
            if self.index_path.exists():
                self.load_index(self.index_path)
            self._auto_save = True

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------
    def index_documents(self, items: List[Dict[str, Any]]) -> None:
        """Replace the entire index with a new set of documents."""
        with self._lock:
            self.index.build(items, self.fields)
            self._auto_persist()
            self._clear_cache()
            logger.info(f"Indexed {len(items)} documents")

    def add_document(self, doc: Dict[str, Any]) -> int:
        """Add a single document. Returns its new document ID."""
        with self._lock:
            doc_id = self.index.add_document(doc)
            self._auto_persist()
            self._clear_cache()
            return doc_id

    def add_documents(self, docs: List[Dict[str, Any]]) -> List[int]:
        """Add multiple documents atomically. Returns list of new IDs."""
        with self._lock:
            ids = self.index.add_documents(docs)
            if ids:
                self._auto_persist()
                self._clear_cache()
            return ids

    def remove_document(self, doc_id: int) -> None:
        """Remove a document by its ID."""
        with self._lock:
            self.index.remove_document(doc_id)
            self._auto_persist()
            self._clear_cache()

    def update_document(self, doc_id: int, new_doc: Dict[str, Any]) -> None:
        """Replace an existing document."""
        with self._lock:
            self.index.update_document(doc_id, new_doc)
            self._auto_persist()
            self._clear_cache()

    def get_document(self, doc_id: int) -> Dict[str, Any]:
        """Retrieve a document by its ID."""
        with self._lock:
            return self.index.get_document(doc_id)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def search(self, query: str, limit: int = 10, fuzzy_threshold: Optional[float] = None) -> List[SearchResult]:
        """
        Search the index.

        Args:
            query: Search string.
            limit: Maximum number of results.
            fuzzy_threshold: Override the default fuzzy threshold (0..1).

        Returns:
            List of SearchResult objects sorted by descending score.
        """
        if not query or not query.strip():
            return []

        limit = max(1, limit)
        threshold = fuzzy_threshold if fuzzy_threshold is not None else self.default_fuzzy_threshold
        threshold = max(0.0, min(1.0, threshold))

        cache_key = f"{query.strip()}|{limit}|{threshold}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        with self._lock:
            results = self.index.search(query, limit, threshold)
            search_results = [
                SearchResult(
                    item=self.index.get_document(doc_id),
                    score=score,
                    reason=f"BM25 + fuzzy (analyzer={self.analyzer.__class__.__name__})",
                )
                for doc_id, score in results
            ]
            self._cache.set(cache_key, search_results)
            return search_results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_index(self, path: Optional[Union[str, Path]] = None) -> None:
        """Explicitly save the index to disk."""
        target = Path(path) if path else self.index_path
        if not target:
            raise ValueError("No persistence path provided")
        try:
            self.index.save(target, include_checksum=True)
        except IndexSaveError as e:
            logger.error(f"Failed to save index: {e}")
            raise

    def load_index(self, path: Optional[Union[str, Path]] = None) -> None:
        """Load index from disk, clearing any in‑memory changes."""
        source = Path(path) if path else self.index_path
        if not source:
            raise ValueError("No persistence path provided")
        try:
            self.index.load(source, verify_checksum=True)
            self._clear_cache()
        except IndexLoadError as e:
            logger.error(f"Failed to load index: {e}")
            raise

    def _auto_persist(self) -> None:
        """Save the index automatically if auto‑save is enabled and index is dirty."""
        if self._auto_save and self.index_path and self.index.is_dirty():
            try:
                self.index.save(self.index_path, include_checksum=True)
            except IndexSaveError as e:
                logger.error(f"Auto‑save failed: {e}")

    # ------------------------------------------------------------------
    # Cache & utilities
    # ------------------------------------------------------------------
    def _clear_cache(self) -> None:
        """Clear the result cache (e.g., after index modifications)."""
        self._cache.clear()   # TTLCache provides a clear() method

    def stats(self) -> Dict[str, Any]:
        """Return index statistics."""
        with self._lock:
            return self.index.stats()

    def reload_config(self) -> None:
        """Reload configuration from YAML and update runtime parameters."""
        config = get_config_section("search_engine")
        bm25_config = get_config_section("bm25")
        self.cache_ttl = config.get("cache_ttl_seconds", 120)
        self.default_fuzzy_threshold = config.get("fuzzy_threshold", 0.8)
        # Update BM25 parameters
        self.scorer = BM25Scorer(
            k1=bm25_config.get("k1", 1.2),
            b=bm25_config.get("b", 0.75),
        )
        # Note: scorer is not automatically propagated to the index – recreate index if needed
        logger.info("Search engine configuration reloaded (scorer update requires index rebuild)")