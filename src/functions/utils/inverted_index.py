"""Inverted index with BM25 scoring, incremental updates, and persistence."""

import hashlib
import json
import math
import pickle
import threading

from difflib import SequenceMatcher
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Set, Tuple, Union

from .functions_error import (DocumentNotFoundError, IndexLoadError,
                              InconsistentFieldsError, IndexSaveError)
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Search Engine")
printer = PrettyPrinter


# ----------------------------------------------------------------------
# Analyzers (with stopword support)
# ----------------------------------------------------------------------
class SearchAnalyzer(Protocol):
    """Pluggable analyzer interface: tokenize and normalize text."""
    def analyze(self, text: str) -> List[str]:
        ...


# ----------------------------------------------------------------------
# BM25 scoring (with configurable parameters)
# ----------------------------------------------------------------------
class BM25Scorer:
    """BM25 implementation with configurable k1 and b."""
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b

    def score(self, tf: int, df: int, doc_len: int, avg_doc_len: float, n_docs: int) -> float:
        """Return BM25 score for a single term in a document."""
        if tf == 0:
            return 0.0
        # IDF with smoothing
        idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
        # Length normalisation
        norm = self.k1 * (1 - self.b + self.b * (doc_len / avg_doc_len))
        tf_norm = tf / (tf + norm)
        return idf * tf_norm


# ----------------------------------------------------------------------
# Inverted Index with persistence and incremental updates
# ----------------------------------------------------------------------
class InvertedIndex:
    """
    Thread‑safe inverted index supporting:
    - Build from list of documents
    - Incremental add / remove / update
    - BM25 + fuzzy search
    - Persistence with versioning and optional checksum
    """

    # Current serialisation version – bump on breaking changes
    _SERIAL_VERSION = 2

    def __init__(self, analyzer: SearchAnalyzer, scorer: Optional[BM25Scorer] = None):
        self.analyzer = analyzer
        self.scorer = scorer or BM25Scorer()

        self._docs: List[Dict[str, Any]] = []               # document storage
        self._doc_tokens: List[List[str]] = []              # raw token list per doc
        self._doc_token_sets: List[Set[str]] = []           # unique token set per doc (for fuzzy)
        self._doc_lengths: List[int] = []                   # token count per doc
        self._inverted: Dict[str, Set[int]] = defaultdict(set)  # term -> set of doc ids
        self._term_doc_freq: Counter[str] = Counter()       # df per term
        self._fields: List[str] = []                        # stored field names for consistency

        self._lock = threading.RLock()
        self._dirty = False                                 # whether index changed since last save

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build(self, docs: List[Dict[str, Any]], fields: List[str]) -> None:
        """Build index from scratch, discarding previous data."""
        with self._lock:
            if not docs and not fields:
                self._clear()
                return
            if not fields:
                raise ValueError("fields cannot be empty")
            self._fields = fields[:]        # copy
            self._docs = []
            self._doc_tokens = []
            self._doc_token_sets = []
            self._doc_lengths = []
            self._inverted.clear()
            self._term_doc_freq.clear()

            for doc in docs:
                self._add_document_internal(doc)
            self._dirty = True
            logger.info(f"Built index with {len(self._docs)} documents")

    def add_document(self, doc: Dict[str, Any]) -> int:
        """
        Add a single document to the index.
        Returns the new document ID.
        Raises InconsistentFieldsError if document fields differ from the index's field set.
        """
        with self._lock:
            self._ensure_fields_defined(doc)
            doc_id = self._add_document_internal(doc)
            self._dirty = True
            logger.debug(f"Added document {doc_id}")
            return doc_id

    def add_documents(self, docs: List[Dict[str, Any]]) -> List[int]:
        """Add multiple documents atomically. Returns list of new doc ids."""
        with self._lock:
            ids = []
            for doc in docs:
                self._ensure_fields_defined(doc)
                ids.append(self._add_document_internal(doc))
            if ids:
                self._dirty = True
            return ids

    def remove_document(self, doc_id: int) -> None:
        """
        Remove a document by its ID.
        Raises DocumentNotFoundError if the ID does not exist.
        """
        with self._lock:
            if doc_id < 0 or doc_id >= len(self._docs):
                raise DocumentNotFoundError(doc_id)

            # Remove from inverted index
            for token in self._doc_token_sets[doc_id]:
                self._inverted[token].discard(doc_id)
                if not self._inverted[token]:
                    del self._inverted[token]

            # Remove from document arrays
            del self._docs[doc_id]
            del self._doc_tokens[doc_id]
            del self._doc_token_sets[doc_id]
            del self._doc_lengths[doc_id]

            # Re‑index remaining documents (shift IDs)
            self._reindex_document_ids()
            self._rebuild_term_doc_freq()
            self._dirty = True
            logger.debug(f"Removed document {doc_id}")

    def update_document(self, doc_id: int, new_doc: Dict[str, Any]) -> None:
        """Replace an existing document with new content."""
        with self._lock:
            if doc_id < 0 or doc_id >= len(self._docs):
                raise DocumentNotFoundError(doc_id)
            self._ensure_fields_defined(new_doc)
            # Remove old
            for token in self._doc_token_sets[doc_id]:
                self._inverted[token].discard(doc_id)
                if not self._inverted[token]:
                    del self._inverted[token]
            # Insert new
            self._docs[doc_id] = new_doc
            text = " ".join(str(new_doc.get(field, "")) for field in self._fields)
            tokens = self.analyzer.analyze(text)
            self._doc_tokens[doc_id] = tokens
            self._doc_token_sets[doc_id] = set(tokens)
            self._doc_lengths[doc_id] = len(tokens)
            for token in self._doc_token_sets[doc_id]:
                self._inverted[token].add(doc_id)
            self._rebuild_term_doc_freq()
            self._dirty = True
            logger.debug(f"Updated document {doc_id}")

    def get_document(self, doc_id: int) -> Dict[str, Any]:
        """Return the document at the given ID."""
        with self._lock:
            if doc_id < 0 or doc_id >= len(self._docs):
                raise DocumentNotFoundError(doc_id)
            return self._docs[doc_id]

    def search(
        self,
        query: str,
        limit: int = 10,
        fuzzy_threshold: float = 0.8,
    ) -> List[Tuple[int, float]]:
        """
        Return list of (doc_id, score) sorted by BM25 + fuzzy bonus.
        If the index is empty, returns an empty list.
        """
        with self._lock:
            if not self._docs:
                return []

            query_tokens = self.analyzer.analyze(query)
            if not query_tokens:
                return []

            # Candidate docs: union of all docs containing any query token
            candidate_docs = set()
            for token in query_tokens:
                candidate_docs.update(self._inverted.get(token, set()))
            if not candidate_docs:
                # No token matches – fallback to all docs (fuzzy will still work)
                candidate_docs = set(range(len(self._docs)))

            avg_doc_len = sum(self._doc_lengths) / len(self._docs)
            n_docs = len(self._docs)

            scores = {}
            for doc_id in candidate_docs:
                lexical = self._bm25_score(
                    doc_id, query_tokens, avg_doc_len, n_docs
                )
                fuzzy = self._fuzzy_bonus(query_tokens, doc_id, fuzzy_threshold)
                total = lexical + fuzzy
                if total > 0:
                    scores[doc_id] = total

            # Sort descending by score
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            return sorted_scores[:limit]

    def stats(self) -> Dict[str, Any]:
        """Return index statistics."""
        with self._lock:
            return {
                "num_documents": len(self._docs),
                "num_unique_terms": len(self._inverted),
                "total_tokens": sum(self._doc_lengths),
                "avg_doc_length": sum(self._doc_lengths) / max(len(self._docs), 1),
                "fields": self._fields.copy(),
            }

    def save(self, path: Union[str, Path], include_checksum: bool = True) -> None:
        """
        Persist the index to disk using pickle.
        Raises IndexSaveError on failure.
        """
        with self._lock:
            data = {
                "version": self._SERIAL_VERSION,
                "fields": self._fields,
                "docs": self._docs,
                "doc_tokens": self._doc_tokens,
                "doc_token_sets": self._doc_token_sets,
                "doc_lengths": self._doc_lengths,
                "inverted": dict(self._inverted),
                "term_doc_freq": dict(self._term_doc_freq),
            }
            if include_checksum:
                # Simple checksum: SHA256 of the serialised data (without the checksum field)
                serialized = pickle.dumps(data)
                data["checksum"] = hashlib.sha256(serialized).hexdigest()

            try:
                with open(path, "wb") as f:
                    pickle.dump(data, f)
                self._dirty = False
                logger.info(f"Index saved to {path} (version {self._SERIAL_VERSION})")
            except Exception as e:
                raise IndexSaveError(str(path), str(e))

    def load(self, path: Union[str, Path], verify_checksum: bool = True) -> None:
        """
        Load index from disk.
        Raises IndexLoadError if the file is missing, corrupt, or of an incompatible version.
        """
        with self._lock:
            try:
                with open(path, "rb") as f:
                    data = pickle.load(f)
            except Exception as e:
                raise IndexLoadError(str(path), f"pickle error: {e}")

            # Version check
            if data.get("version") != self._SERIAL_VERSION:
                raise IndexLoadError(
                    str(path),
                    f"Version mismatch: file version {data.get('version')}, "
                    f"expected {self._SERIAL_VERSION}",
                )

            # Checksum verification
            if verify_checksum and "checksum" in data:
                stored_checksum = data.pop("checksum")
                serialized = pickle.dumps(data)
                if hashlib.sha256(serialized).hexdigest() != stored_checksum:
                    raise IndexLoadError(str(path), "checksum mismatch – data corrupted")
                # Put back for internal consistency
                data["checksum"] = stored_checksum

            # Restore state
            self._fields = data["fields"]
            self._docs = data["docs"]
            self._doc_tokens = data["doc_tokens"]
            self._doc_token_sets = data["doc_token_sets"]
            self._doc_lengths = data["doc_lengths"]
            self._inverted = defaultdict(set, data["inverted"])
            self._term_doc_freq = Counter(data["term_doc_freq"])
            self._dirty = False
            logger.info(f"Index loaded from {path} ({len(self._docs)} documents)")

    def is_dirty(self) -> bool:
        """Return True if the index has unsaved changes."""
        with self._lock:
            return self._dirty

    def clear(self) -> None:
        """Reset the index to empty state."""
        with self._lock:
            self._clear()
            self._dirty = True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _clear(self) -> None:
        self._docs.clear()
        self._doc_tokens.clear()
        self._doc_token_sets.clear()
        self._doc_lengths.clear()
        self._inverted.clear()
        self._term_doc_freq.clear()
        self._fields.clear()

    def _ensure_fields_defined(self, doc: Dict[str, Any]) -> None:
        """Check that the document contains all expected fields (or no fields defined yet)."""
        if not self._fields:
            # First document defines the fields
            self._fields = list(doc.keys())
        else:
            # Verify that the document's keys are exactly the stored fields (order ignored)
            if set(doc.keys()) != set(self._fields):
                raise InconsistentFieldsError(self._fields, list(doc.keys()))

    def _add_document_internal(self, doc: Dict[str, Any]) -> int:
        """Add a document without acquiring the lock. Returns new doc_id."""
        doc_id = len(self._docs)
        self._docs.append(doc)

        text = " ".join(str(doc.get(field, "")) for field in self._fields)
        tokens = self.analyzer.analyze(text)
        self._doc_tokens.append(tokens)
        token_set = set(tokens)
        self._doc_token_sets.append(token_set)
        self._doc_lengths.append(len(tokens))

        # Update inverted index
        for token in token_set:
            self._inverted[token].add(doc_id)

        # Update term doc frequencies
        for token in token_set:
            self._term_doc_freq[token] = len(self._inverted[token])

        return doc_id

    def _reindex_document_ids(self) -> None:
        """
        After deletions, document IDs shift. Rebuild the inverted index with new IDs.
        This is O(N) in number of terms, but deletions are expected to be rare.
        """
        new_inverted = defaultdict(set)
        for new_id, token_set in enumerate(self._doc_token_sets):
            for token in token_set:
                new_inverted[token].add(new_id)
        self._inverted = new_inverted

    def _rebuild_term_doc_freq(self) -> None:
        """Rebuild term document frequency from the inverted index."""
        self._term_doc_freq = Counter()
        for token, doc_set in self._inverted.items():
            self._term_doc_freq[token] = len(doc_set)

    def _bm25_score(
        self,
        doc_id: int,
        query_tokens: List[str],
        avg_doc_len: float,
        n_docs: int,
    ) -> float:
        """Compute pure BM25 score for a document and query."""
        score = 0.0
        doc_len = self._doc_lengths[doc_id]
        # Pre‑compute term frequencies for this document
        tf_map = Counter(self._doc_tokens[doc_id])
        for token in query_tokens:
            tf = tf_map.get(token, 0)
            if tf == 0:
                continue
            df = self._term_doc_freq.get(token, 0)
            if df == 0:
                continue
            score += self.scorer.score(tf, df, doc_len, avg_doc_len, n_docs)
        return score

    def _fuzzy_bonus(self, query_tokens: List[str], doc_id: int, threshold: float) -> float:
        """Compute bonus for approximate matches using difflib."""
        doc_tokens = self._doc_token_sets[doc_id]
        bonus = 0.0
        for q_token in query_tokens:
            if q_token in doc_tokens:
                continue
            best = 0.0
            for d_token in doc_tokens:
                ratio = SequenceMatcher(None, q_token, d_token).ratio()
                if ratio > best:
                    best = ratio
                    if best >= threshold:
                        break
            if best >= threshold:
                bonus += best * 0.35   # fuzzy weight
        return bonus