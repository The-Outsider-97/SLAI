"""Inverted index with BM25 scoring, incremental updates, and persistence."""

import hashlib
import hmac
import json
import math
import os
import tempfile
import threading
import copy

from difflib import SequenceMatcher
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Set, Tuple, Union

from .functions_error import *
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Search Engine")
printer = PrettyPrinter()


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
        self.k1 = float(k1)
        self.b = float(b)
        if not math.isfinite(self.k1) or self.k1 <= 0:
            raise ValueError("k1 must be finite and > 0")
        if not math.isfinite(self.b) or not 0.0 <= self.b <= 1.0:
            raise ValueError("b must be finite and within [0, 1]")

    def score(self, tf: int, df: int, doc_len: int, avg_doc_len: float, n_docs: int) -> float:
        """Return BM25 score for a single term in a document."""
        if tf <= 0 or df <= 0 or n_docs <= 0:
            return 0.0
        # IDF with smoothing
        idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
        # Length normalisation
        safe_avg_doc_len = max(float(avg_doc_len), 1.0)
        norm = self.k1 * (1 - self.b + self.b * (doc_len / safe_avg_doc_len))
        tf_norm = (tf * (self.k1 + 1.0)) / (tf + norm)
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
    _SERIAL_VERSION = 3

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
        if not fields:
            if not docs:
                with self._lock:
                    self._clear()
                    self._dirty = True
                return
            raise ValueError("fields cannot be empty")

        normalized_fields = list(fields)
        if not all(isinstance(field, str) and field for field in normalized_fields):
            raise ValueError("fields must contain non-empty strings")
        if len(set(normalized_fields)) != len(normalized_fields):
            raise ValueError("fields must not contain duplicates")

        staged = InvertedIndex(self.analyzer, self.scorer)
        staged._fields = normalized_fields
        for doc in docs:
            staged._ensure_fields_defined(doc)
            staged._add_document_internal(doc)

        with self._lock:
            self._fields = staged._fields
            self._docs = staged._docs
            self._doc_tokens = staged._doc_tokens
            self._doc_token_sets = staged._doc_token_sets
            self._doc_lengths = staged._doc_lengths
            self._inverted = staged._inverted
            self._term_doc_freq = staged._term_doc_freq
            self._dirty = True
            logger.info(f"Built index with {len(self._docs)} documents")

    def add_document(self, doc: Dict[str, Any]) -> int:
        """
        Add a single document to the index.
        Returns the new document ID.
        Raises InconsistentFieldsError if document fields differ from the index's field set.
        """
        with self._lock:
            staged_fields = self._fields.copy() or list(doc.keys())
            if set(doc.keys()) != set(staged_fields):
                raise InconsistentFieldsError(staged_fields, list(doc.keys()))
            document = copy.deepcopy(doc)
            text = " ".join(str(document.get(field, "")) for field in staged_fields)
            tokens = self.analyzer.analyze(text)
            self._fields = staged_fields
            doc_id = self._append_prepared_document(document, tokens)
            self._dirty = True
            logger.debug(f"Added document {doc_id}")
            return doc_id

    def add_documents(self, docs: List[Dict[str, Any]]) -> List[int]:
        """Add multiple documents atomically. Returns list of new doc ids."""
        with self._lock:
            if not docs:
                return []

            staged_fields = self._fields.copy() or list(docs[0].keys())
            for doc in docs:
                if set(doc.keys()) != set(staged_fields):
                    raise InconsistentFieldsError(staged_fields, list(doc.keys()))

            prepared: List[Tuple[Dict[str, Any], List[str]]] = []
            for doc in docs:
                document = copy.deepcopy(doc)
                text = " ".join(str(document.get(field, "")) for field in staged_fields)
                prepared.append((document, self.analyzer.analyze(text)))

            self._fields = staged_fields
            ids = [self._append_prepared_document(document, tokens) for document, tokens in prepared]
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
            document = copy.deepcopy(new_doc)
            text = " ".join(str(document.get(field, "")) for field in self._fields)
            tokens = self.analyzer.analyze(text)
            # Remove old
            for token in self._doc_token_sets[doc_id]:
                self._inverted[token].discard(doc_id)
                if not self._inverted[token]:
                    del self._inverted[token]
            # Insert new
            self._docs[doc_id] = document
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
            return copy.deepcopy(self._docs[doc_id])

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
            if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
                raise ValueError("limit must be a positive integer")
            if not math.isfinite(float(fuzzy_threshold)) or not 0.0 <= fuzzy_threshold <= 1.0:
                raise ValueError("fuzzy_threshold must be finite and within [0, 1]")

            query_tokens = self.analyzer.analyze(query)
            if not query_tokens:
                return []

            # Candidate docs: union of all docs containing any query token
            candidate_docs: Set[int] = set()
            for token in query_tokens:
                candidate_docs.update(self._inverted.get(token, set()))
            if fuzzy_threshold < 1.0:
                # Fuzzy matches can occur in documents with no exact query token.
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
            sorted_scores = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
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
        """Persist fields and documents as checksummed JSON using atomic replace."""
        with self._lock:
            state = {
                "version": self._SERIAL_VERSION,
                "fields": self._fields.copy(),
                "docs": copy.deepcopy(self._docs),
            }
            canonical = json.dumps(
                state,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            envelope: Dict[str, Any] = {"state": state}
            if include_checksum:
                envelope["checksum"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

            target = Path(path).expanduser()
            target.parent.mkdir(parents=True, exist_ok=True)
            temp_path: Optional[Path] = None
            try:
                fd, temp_name = tempfile.mkstemp(
                    prefix=f".{target.name}.",
                    suffix=".tmp",
                    dir=str(target.parent),
                )
                temp_path = Path(temp_name)
                with os.fdopen(fd, "w", encoding="utf-8") as stream:
                    json.dump(envelope, stream, ensure_ascii=False, sort_keys=True)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temp_path, target)
                self._dirty = False
                logger.info("Index saved to %s (version %s)", target, self._SERIAL_VERSION)
            except (OSError, TypeError, ValueError) as exc:
                raise IndexSaveError(str(target), str(exc)) from exc
            finally:
                if temp_path is not None:
                    temp_path.unlink(missing_ok=True)

    def load(self, path: Union[str, Path], verify_checksum: bool = True) -> None:
        """
        Load index from disk.
        Raises IndexLoadError if the file is missing, corrupt, or of an incompatible version.
        """
        source = Path(path).expanduser()
        try:
            with source.open("r", encoding="utf-8") as stream:
                envelope = json.load(stream)
        except (OSError, json.JSONDecodeError) as exc:
            raise IndexLoadError(str(source), str(exc)) from exc

        if not isinstance(envelope, dict) or not isinstance(envelope.get("state"), dict):
            raise IndexLoadError(str(source), "index must contain a state object")
        state = envelope["state"]
        if state.get("version") != self._SERIAL_VERSION:
            raise IndexLoadError(
                str(source),
                f"version mismatch: {state.get('version')!r}; expected {self._SERIAL_VERSION}",
            )

        canonical = json.dumps(
            state,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if verify_checksum:
            checksum = envelope.get("checksum")
            if not isinstance(checksum, str):
                raise IndexLoadError(str(source), "checksum is missing")
            if not hmac.compare_digest(
                hashlib.sha256(canonical.encode("utf-8")).hexdigest(), checksum
            ):
                raise IndexLoadError(str(source), "checksum mismatch")

        fields = state.get("fields")
        docs = state.get("docs")
        if not isinstance(fields, list) or not all(isinstance(field, str) for field in fields):
            raise IndexLoadError(str(source), "fields must be a list of strings")
        if not isinstance(docs, list) or not all(isinstance(doc, dict) for doc in docs):
            raise IndexLoadError(str(source), "docs must be a list of objects")

        staged = InvertedIndex(self.analyzer, self.scorer)
        try:
            staged.build(docs, fields)
        except (TypeError, ValueError, InconsistentFieldsError) as exc:
            raise IndexLoadError(str(source), f"invalid index state: {exc}") from exc

        with self._lock:
            self._fields = staged._fields
            self._docs = staged._docs
            self._doc_tokens = staged._doc_tokens
            self._doc_token_sets = staged._doc_token_sets
            self._doc_lengths = staged._doc_lengths
            self._inverted = staged._inverted
            self._term_doc_freq = staged._term_doc_freq
            self._dirty = False
            logger.info("Index loaded from %s (%s documents)", source, len(self._docs))

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
        document = copy.deepcopy(doc)
        text = " ".join(str(document.get(field, "")) for field in self._fields)
        tokens = self.analyzer.analyze(text)
        return self._append_prepared_document(document, tokens)

    def _append_prepared_document(self, document: Dict[str, Any], tokens: List[str]) -> int:
        doc_id = len(self._docs)
        self._docs.append(document)
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
            for d_token in sorted(doc_tokens):
                ratio = SequenceMatcher(None, q_token, d_token).ratio()
                if ratio > best:
                    best = ratio
            if best >= threshold:
                bonus += best * 0.35   # fuzzy weight
        return bonus