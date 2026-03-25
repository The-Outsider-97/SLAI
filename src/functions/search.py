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

from .functions_memory import TTLCache
from .utils.inverted_index import InvertedIndex, SearchAnalyzer, BM25Scorer
from .utils.config_loader import load_global_config, get_config_section
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Search Engine")
printer = PrettyPrinter


class BasicAnalyzer:
    """Basic tokenizer: lowercases, splits on non-alnum."""
    def analyze(self, text: str) -> List[str]:
        cleaned = "".join(c.lower() if c.isalnum() else " " for c in text)
        return [token for token in cleaned.split() if token]


class StemAnalyzer(BasicAnalyzer):
    """Naive stemmer for light-weight language normalization."""
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
    """Analyzer that removes stopwords loaded from a JSON file."""
    def __init__(self, language: str = "en", stopwords_path: Optional[str] = None):
        super().__init__()
        self.language = language
        self._stopwords: Set[str] = set()
        self._load_stopwords(stopwords_path)

    def _load_stopwords(self, path: Optional[str] = None):
        """Load stopwords from the given JSON file or config."""
        if path is None:
            config = get_config_section('language_aware')
            path = config.get('stopwords')
        if path and Path(path).exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Expecting a list under key "Stopword"
                    self._stopwords = set(data.get("Stopword", []))
                logger.debug(f"Loaded {len(self._stopwords)} stopwords from {path}")
            except Exception as e:
                logger.error(f"Failed to load stopwords: {e}")
        else:
            logger.warning(f"Stopwords file not found at {path}, using fallback")
            self._stopwords = self._fallback_stopwords()

    def _fallback_stopwords(self) -> Set[str]:
        """Fallback stopwords for common languages."""
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
# Search Engine facade
# ----------------------------------------------------------------------
@dataclass
class SearchResult:
    item: Dict[str, Any]
    score: float
    reason: str


class SearchEngine:
    """Production search engine with persistent indexing and caching."""

    def __init__(
        self,
        fields: List[str],
        analyzer: Optional[SearchAnalyzer] = None,
        cache_ttl_seconds: int = 120,
        index_persistence_path: Optional[Union[str, Path]] = None,
    ):
        config = get_config_section('search_engine')
        bm25_config = config.get('bm25', {})
        if not fields:
            raise ValueError("fields cannot be empty")
        self.fields = fields
        self.analyzer = analyzer or BasicAnalyzer()
        self.scorer = BM25Scorer(k1=bm25_config.get('k1', 1.2), b=bm25_config.get('b', 0.75))
        self.cache_ttl = cache_ttl_seconds or config.get('cache_ttl_seconds', 120)
        self.index = InvertedIndex(self.analyzer)
        self._cache: TTLCache[List[SearchResult]] = TTLCache(
            max_size=config.get('max_size', 2048),
            ttl_seconds=self.cache_ttl
        )
        self._lock = threading.RLock()

        if index_persistence_path:
            self.index_path = Path(index_persistence_path)
            if self.index_path.exists():
                self.load_index(self.index_path)
            self._auto_save = True
        else:
            self.index_path = None
            self._auto_save = False

    def index_documents(self, items: List[Dict[str, Any]]) -> None:
        """Build the index from a list of documents."""
        with self._lock:
            self.index.build(items, self.fields)
            if self._auto_save and self.index_path:
                self.index.save(self.index_path)
            # Clear cache because index changed
            self._cache = TTLCache(
                max_size=get_config_section('search_engine').get('max_size', 2048),
                ttl_seconds=self.cache_ttl
            )
            logger.info(f"Indexed {len(items)} documents")

    def add_document(self, doc: Dict[str, Any]) -> None:
        """Add a single document to the index."""
        with self._lock:
            self.index.add_document(doc, self.fields)
            if self._auto_save and self.index_path:
                self.index.save(self.index_path)
            # Clear cache
            self._clear_cache()

    def remove_document(self, doc_id: int) -> None:
        """Remove a document by its ID (careful: IDs shift if rebuild)."""
        with self._lock:
            self.index.remove_document(doc_id)
            if self._auto_save and self.index_path:
                self.index.save(self.index_path)
            self._clear_cache()

    def search(self, query: str, limit: int = 10, fuzzy_threshold: float = 0.8) -> List[SearchResult]:
        """Search the indexed documents."""
        # Check cache
        cache_key = f"{query}|{limit}|{fuzzy_threshold}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        # Perform search
        results = self.index.search(query, limit, fuzzy_threshold)
        search_results = [
            SearchResult(
                item=self.index.get_document(doc_id),
                score=score,
                reason=f"BM25 + fuzzy (analyzer={self.analyzer.__class__.__name__})"
            )
            for doc_id, score in results
        ]
        self._cache.set(cache_key, search_results)
        return search_results

    def save_index(self, path: Optional[Union[str, Path]] = None) -> None:
        """Save the current index to disk."""
        save_path = path or self.index_path
        if not save_path:
            raise ValueError("No persistence path provided")
        self.index.save(save_path)

    def load_index(self, path: Optional[Union[str, Path]] = None) -> None:
        """Load index from disk."""
        load_path = path or self.index_path
        if not load_path:
            raise ValueError("No persistence path provided")
        self.index.load(load_path)
        self._clear_cache()

    def _clear_cache(self) -> None:
        """Clear the search result cache (e.g., after index changes)."""
        self._cache = TTLCache(
            max_size=get_config_section('search_engine').get('max_size', 2048),
            ttl_seconds=self.cache_ttl
        )