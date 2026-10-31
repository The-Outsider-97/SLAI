"""SLAI Knowledge RelevanceRanker v1.

Local, pointwise-scored/pairwise-trained neural reranker.  Retrieval is owned
by KnowledgeAgent; this module never generates documents or citations.

Dependency boundary: Torch and NumPy (already in SLAI). A LOCAL
sentence-transformers encoder is optional, and must be the same encoder used
for training. Never fetch pretrained weights at inference time.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import threading
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

FEATURE_VERSION = 1
FEATURE_NAMES = (
    "query_term_recall", "document_term_precision", "term_jaccard",
    "term_f1", "query_bigram_recall", "query_order_coverage",
    "character_trigram_jaccard", "exact_phrase_match", "relative_length",
    "normalized_term_frequency", "query_coverage_weighted", "semantic_cosine",
)
_TOKEN = re.compile(r"\w+", re.UNICODE)


def _terms(s: str, max_chars: int = 16000) -> list[str]:
    return _TOKEN.findall(s[:max_chars].casefold())


def _ngrams(s: Sequence[str], n: int) -> set[tuple[str, ...]]:
    return set(zip(*(s[i:] for i in range(n)))) if len(s) >= n else set()


def lexical_features(query: str, passage: str) -> list[float]:
    q, d = _terms(query), _terms(passage)
    if not q or not d:
        return [0.0] * (len(FEATURE_NAMES) - 1)
    qs, ds = set(q), set(d)
    shared = qs & ds
    recall = len(shared) / len(qs)
    precision = len(shared) / len(ds)
    jac = len(shared) / len(qs | ds)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    qb = _ngrams(q, 2)
    db = _ngrams(d, 2)
    bigram = len(qb & db) / max(len(qb), 1)
    # Count query tokens in document order (not necessarily adjacent).
    cursor, matched = 0, 0
    for word in q:
        try:
            cursor = d.index(word, cursor) + 1
            matched += 1
        except ValueError:
            break
    order = matched / len(q)
    q_text = " ".join(q)
    qc = set(q_text[i:i+3] for i in range(max(0, len(q_text)-2)))
    ds_text = " ".join(d)
    dc = set(ds_text[i:i+3] for i in range(max(0, len(ds_text)-2)))
    char = len(qc & dc) / max(len(qc | dc), 1)
    phrase = float(" ".join(q) in ds_text)
    ratio = math.exp(-abs(math.log((len(d) + 1) / (len(q) + 1))))
    counts = Counter(d)
    tf = sum(counts[t] / (counts[t] + 1.0) for t in qs) / len(qs)
    # Rare words and long terms can help disambiguate technical requests;
    # this is not corpus IDF and does not pretend to be semantic reasoning.
    weighted = sum(min(12, len(t)) for t in shared) / max(1, sum(min(12, len(t)) for t in qs))
    return [recall, precision, jac, f1, bigram, order, char, phrase, ratio, tf, weighted]


class LocalSemanticEncoder:
    """Offline sentence-embedding adapter with a bounded embedding cache."""

    def __init__(self, model_path: str, max_entries: int = 4096):
        if not Path(model_path).is_dir():
            raise FileNotFoundError(f"Local semantic encoder missing: {model_path}")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("Semantic feature requires sentence-transformers") from exc
        # The model was explicitly supplied as an existing local snapshot.
        # Inference must never trigger a download or require online access.
        self.model = SentenceTransformer(str(Path(model_path).resolve()), local_files_only=True)
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        self.cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.lock = threading.RLock()
        self.max_entries = max_entries

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Encode text in batches without losing results to LRU eviction.

        The previous version first inserted every missing embedding into the
        bounded cache and only then looked up every requested text. Once the
        cache filled, inserting misses evicted requested cache hits, causing a
        KeyError during SciFact feature extraction. Results must be retained
        independently of the cache *before* any new insertions or eviction.
        """
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        with self.lock:
            result: list[np.ndarray | None] = [None] * len(texts)
            pending: dict[str, list[int]] = {}

            for index, value in enumerate(texts):
                hit = self.cache.get(value)
                if hit is not None:
                    # Retain the vector even if it is evicted later in this call.
                    result[index] = hit
                    self.cache.move_to_end(value)
                else:
                    pending.setdefault(value, []).append(index)

            unique_missing = list(pending)
            for offset in range(0, len(unique_missing), 32):
                batch = unique_missing[offset:offset + 32]
                encoded = np.asarray(
                    self.model.encode(
                        batch,
                        batch_size=32,
                        show_progress_bar=False,
                        normalize_embeddings=True,
                    ),
                    dtype=np.float32,
                )
                if encoded.ndim != 2 or encoded.shape[0] != len(batch):
                    raise ValueError("Semantic encoder returned an unexpected embedding batch")
                if not np.isfinite(encoded).all():
                    raise ValueError("Semantic encoder returned nonfinite embeddings")

                for value, vector in zip(batch, encoded):
                    for index in pending[value]:
                        result[index] = vector
                    self.cache[value] = vector
                    self.cache.move_to_end(value)
                    while len(self.cache) > self.max_entries:
                        self.cache.popitem(last=False)

            if any(vector is None for vector in result):
                raise RuntimeError("Internal semantic feature alignment failure")
            return np.stack(result).astype(np.float32, copy=False)


class FeatureExtractor:
    def __init__(self, encoder_path: str | None = None):
        self.encoder_path = encoder_path or None
        self.encoder = LocalSemanticEncoder(self.encoder_path) if self.encoder_path else None

    def transform(self, queries: Sequence[str], documents: Sequence[str]) -> np.ndarray:
        if len(queries) != len(documents):
            raise ValueError("query/document length mismatch")
        features = np.asarray([lexical_features(q, d) for q, d in zip(queries, documents)],
                              dtype=np.float32).reshape(-1, len(FEATURE_NAMES) - 1)
        semantic = np.zeros(len(queries), dtype=np.float32)
        if self.encoder is not None and queries:
            qvec = self.encoder.encode(queries)
            dvec = self.encoder.encode(documents)
            semantic = np.sum(qvec * dvec, axis=1).astype(np.float32)
        return np.column_stack((features, semantic)).astype(np.float32)


class RankerNetwork(nn.Module):
    def __init__(self, width: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(len(FEATURE_NAMES), width), nn.ReLU(),
            nn.Linear(width, width // 2), nn.ReLU(),
            nn.Linear(width // 2, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def checkpoint_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class RelevanceRanker:
    """Thread-safe inference-only wrapper, preserving SLAI candidate metadata."""

    def __init__(self, model_dir: str | Path, *, encoder_path: str | None = None,
                 device: str = "cpu", batch_size: int = 64):
        from ..model_hub import resolve_model_bundle
        root = resolve_model_bundle(model_dir, "knowledge/relevance_ranker", ("manifest.json", "model.pt"))
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("feature_version") != FEATURE_VERSION or tuple(manifest.get("feature_names", [])) != FEATURE_NAMES:
            raise ValueError("Unsupported RelevanceRanker feature schema")
        artifact = root / "model.pt"
        if checkpoint_sha256(artifact) != manifest.get("sha256"):
            raise ValueError("RelevanceRanker model checksum mismatch")
        expected = manifest.get("encoder_path") or None
        actual = encoder_path or expected
        if bool(actual) != bool(expected) or (expected and Path(actual).resolve() != Path(expected).resolve()):
            raise ValueError("Semantic encoder must match the training configuration")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.extractor = FeatureExtractor(actual)
        self.device = torch.device(device)
        self.batch_size = batch_size
        checkpoint = torch.load(artifact, map_location="cpu", weights_only=True)
        self.network = RankerNetwork(width=int(manifest["width"]))
        self.network.load_state_dict(checkpoint["model"])
        self.network.to(self.device).eval()
        self.mean = np.asarray(checkpoint["mean"].cpu().numpy(), dtype=np.float32)
        self.std = np.asarray(checkpoint["std"].cpu().numpy(), dtype=np.float32)
        if (len(self.mean) != len(FEATURE_NAMES) or len(self.std) != len(FEATURE_NAMES)
                or not np.isfinite(self.mean).all() or not (self.std > 0).all()):
            raise ValueError("Corrupt normalizer parameters")
        self.lock = threading.RLock()
        self.version = manifest.get("model_id", "unknown")
        self.artifact_sha256 = manifest["sha256"]
        self.passed_baseline_gate = bool(manifest.get("evaluation_passed_baseline_metrics", False))

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        output: list[float] = []
        with self.lock, torch.inference_mode():
            for offset in range(0, len(passages), self.batch_size):
                block = passages[offset:offset + self.batch_size]
                x = self.extractor.transform([query] * len(block), block)
                x = np.clip((x - self.mean) / self.std, -8, 8)
                t = torch.from_numpy(x).to(self.device)
                values = self.network(t).detach().cpu().tolist()
                if not all(math.isfinite(v) for v in values):
                    raise ValueError("Nonfinite ranking score")
                output.extend(values)
        return output

    def rank(self, query: str, candidates: Sequence[tuple[float, Mapping[str, Any]]]
             ) -> list[tuple[float, Mapping[str, Any]]]:
        """Reorder but retain ORIGINAL retrieval scores and document dictionaries.

        Existing KnowledgeAgent consumers also use tuple[0] as a confidence-like
        retrieval quantity. Raw neural logits must not be passed off as calibrated
        factual confidence. Model scores are used *only for ordering*.
        """
        items = list(candidates)
        if len(items) < 2:
            return items
        scores = self.score(query, [str(doc.get("text", "")) for _, doc in items])
        ranked_indices = sorted(range(len(items)), key=lambda i: (-scores[i], i))
        return [items[i] for i in ranked_indices]
