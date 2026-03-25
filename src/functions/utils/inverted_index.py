
import math
import pickle
import threading

from pathlib import Path
from difflib import SequenceMatcher
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Protocol, Set, Tuple, Union

from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Search Engine")
printer = PrettyPrinter


# ----------------------------------------------------------------------
# Analyzers (with stopword support)
# ----------------------------------------------------------------------
class SearchAnalyzer(Protocol):
    """Pluggable analyzer interface."""
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

    def score(self, token: str, doc_id: int, tf: int, df: int, doc_len: int, avg_doc_len: float, n_docs: int) -> float:
        if tf == 0:
            return 0.0
        # IDF
        idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
        # TF normalization
        norm = self.k1 * (1 - self.b + self.b * (doc_len / avg_doc_len))
        tf_norm = tf / (tf + norm)
        return idf * tf_norm


# ----------------------------------------------------------------------
# Inverted Index with persistence and incremental updates
# ----------------------------------------------------------------------
class InvertedIndex:
    """Inverted index with BM25 support and persistence."""
    def __init__(self, analyzer: SearchAnalyzer, scorer: Optional[BM25Scorer] = None):
        self.analyzer = analyzer
        self.scorer = scorer or BM25Scorer()
        self._docs: List[Dict[str, Any]] = []
        self._doc_tokens: List[List[str]] = []
        self._doc_lengths: List[int] = []
        self._term_doc_freq: Counter[str] = Counter()
        self._inverted: Dict[str, Set[int]] = defaultdict(set)
        self._fields: List[str] = []        # Store fields for removal
        self._lock = threading.RLock()

    def build(self, docs: List[Dict[str, Any]], fields: List[str]) -> None:
        with self._lock:
            self._fields = fields                # Save fields
            self._docs = docs
            self._doc_tokens = []
            self._doc_lengths = []
            self._inverted.clear()
            for doc_id, doc in enumerate(docs):
                text = " ".join(str(doc.get(field, "")) for field in fields)
                tokens = self.analyzer.analyze(text)
                self._doc_tokens.append(tokens)
                self._doc_lengths.append(len(tokens))
                for token in set(tokens):
                    self._inverted[token].add(doc_id)
            self._rebuild_term_doc_freq()

    def get_document(self, doc_id: int) -> Dict[str, Any]:
        with self._lock:
            return self._docs[doc_id]

    def add_document(self, doc: Dict[str, Any], fields: List[str]) -> int:
        """Add a document to the index, return its doc_id."""
        with self._lock:
            if not self._fields:                 # First add, store fields
                self._fields = fields
            # Optionally, verify that fields match previously stored ones
            if fields != self._fields:
                raise ValueError("Fields must be consistent across all documents")

            doc_id = len(self._docs)
            self._docs.append(doc)
            # Tokenize all fields
            text = " ".join(str(doc.get(field, "")) for field in fields)
            tokens = self.analyzer.analyze(text)
            self._doc_tokens.append(tokens)
            self._doc_lengths.append(len(tokens))
            # Update inverted index
            for token in set(tokens):  # set to count unique per doc for doc freq
                self._inverted[token].add(doc_id)
            # Update term doc frequencies (we'll compute from inverted)
            self._rebuild_term_doc_freq()
            return doc_id
    
    def remove_document(self, doc_id: int) -> None:
        with self._lock:
            if not self._fields:
                raise RuntimeError("Index not built yet")
            if doc_id < 0 or doc_id >= len(self._docs):
                raise IndexError(f"Document ID {doc_id} out of range")
            new_docs = [doc for i, doc in enumerate(self._docs) if i != doc_id]
            self.build(new_docs, self._fields)

    def _rebuild_term_doc_freq(self) -> None:
        """Rebuild term document frequency from inverted index."""
        self._term_doc_freq = Counter()
        for token, docs in self._inverted.items():
            self._term_doc_freq[token] = len(docs)

    def search(self, query: str, limit: int = 10, fuzzy_threshold: float = 0.8) -> List[Tuple[int, float]]:
        """Return list of (doc_id, score) sorted by BM25 + fuzzy bonus."""
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
                candidate_docs = set(range(len(self._docs)))

            # Precompute average doc length
            avg_doc_len = sum(self._doc_lengths) / max(len(self._docs), 1)
            n_docs = len(self._docs)

            scores = {}
            for doc_id in candidate_docs:
                # Lexical score (BM25)
                lexical = 0.0
                for token in query_tokens:
                    tf = self._doc_tokens[doc_id].count(token)  # term frequency in doc
                    if tf == 0:
                        continue
                    df = self._term_doc_freq[token]
                    doc_len = self._doc_lengths[doc_id]
                    lexical += self.scorer.score(token, doc_id, tf, df, doc_len, avg_doc_len, n_docs)
                # Fuzzy bonus (if enabled)
                fuzzy = self._fuzzy_bonus(query_tokens, doc_id, fuzzy_threshold)
                total = lexical + fuzzy
                if total > 0:
                    scores[doc_id] = total

            # Sort by score descending
            return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit]

    def _fuzzy_bonus(self, query_tokens: List[str], doc_id: int, threshold: float) -> float:
        """Compute bonus for approximate matches using difflib (fallback)."""
        doc_tokens = set(self._doc_tokens[doc_id])
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
                bonus += best * 0.35   # weight for fuzzy matches
        return bonus

    def save(self, path: Union[str, Path]) -> None:
        """Persist index to disk using pickle."""
        with self._lock:
            data = {
                'docs': self._docs,
                'doc_tokens': self._doc_tokens,
                'doc_lengths': self._doc_lengths,
                'inverted': dict(self._inverted),
                'term_doc_freq': dict(self._term_doc_freq),
            }
            with open(path, 'wb') as f:
                pickle.dump(data, f)
            logger.debug(f"Index saved to {path}")

    def load(self, path: Union[str, Path]) -> None:
        """Load index from disk."""
        with self._lock:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            self._docs = data['docs']
            self._doc_tokens = data['doc_tokens']
            self._doc_lengths = data['doc_lengths']
            self._inverted = defaultdict(set, data['inverted'])
            self._term_doc_freq = Counter(data['term_doc_freq'])
            logger.debug(f"Index loaded from {path}")