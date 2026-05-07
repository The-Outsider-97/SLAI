from __future__ import annotations

import yaml

__version__ = "2.1.0"

"""
Knowledge Agent for SLAI (Scalable Learning Autonomous Intelligence).

Production-oriented knowledge retrieval and management agent with:
- TF-IDF retrieval from scratch
- Optional dense retrieval via SentenceTransformer
- Ontology expansion
- Governance-aware retrieval auditing
- Integrated knowledge memory, cache, sync, monitor, and action orchestration
"""

import hashlib
import json
import math
import os
import re
import threading
import time
import numpy as np

from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Callable

from .base.utils.main_config_loader import load_global_config, get_config_section
from .base_agent import BaseAgent
from .knowledge.utils.knowledge_errors import (
    BiasDetectionError,
    EmbeddingError,
    GovernanceViolation,
    InvalidDocumentError,
    MemoryUpdateError,
    OntologyError,
    RetrievalError,
)
from logs.logger import PrettyPrinter, get_logger # pyright: ignore[reportMissingImports]

logger = get_logger("Knowledge Agent")
printer = PrettyPrinter

_TOKENIZER_INSTANCE = None


def cosine_sim(v1: Sequence[float], v2: Sequence[float]) -> float:
    a = np.asarray(v1, dtype=float)
    b = np.asarray(v2, dtype=float)
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))


class KnowledgeAgent(BaseAgent):
    """Knowledge-centric retrieval and management agent."""

    _embedding_model_cache: Dict[str, Any] = {}
    _embedding_model_lock = threading.Lock()

    def __init__(
        self,
        shared_memory,
        agent_factory,
        config: Optional[dict] = None,
        persist_file: Optional[str] = None,
        knowledge_memory: Optional[Any] = None,
        knowledge_cache: Optional[Any] = None,
        perform_action: Optional[Any] = None,
        ontology_manager: Optional[Any] = None,
        governor: Optional[Any] = None,
        rule_engine: Optional[Any] = None,
        synchronizer: Optional[Any] = None,
        monitor: Optional[Any] = None,
        orchestrator: Optional[Any] = None,
    ):
        super().__init__(shared_memory=shared_memory, agent_factory=agent_factory, config=config)

        self.shared_memory = shared_memory
        self.agent_factory = agent_factory
        self.config = load_global_config()
        self.knowledge_config = get_config_section('knowledge_agent') or {}

        self.source = self.knowledge_config.get("source", "knowledge_agent")
        self.tokenize = bool(self.knowledge_config.get("tokenize", True))
        self.is_query = bool(self.knowledge_config.get("is_query", True))
        self.stopwords_path = self.knowledge_config.get("stopwords")
        self.cache_size = self._coerce_positive_int(self.knowledge_config.get("cache_size"), 1000)
        self.first_pass = bool(self.knowledge_config.get("first_pass", True))
        self.max_workers = self._coerce_positive_int(self.knowledge_config.get("max_workers"), 4)
        self._decay_factor = self._coerce_float(self.knowledge_config.get("decay_factor"), 0.8)
        self.knowledge_tag = self.knowledge_config.get("knowledge_tag", "knowledge")
        self.retrieval_mode = str(self.knowledge_config.get("retrieval_mode", "tfidf")).lower()
        self.context_window = self._coerce_positive_int(self.knowledge_config.get("context_window"), 3)
        self.bias_threshold = self._coerce_float(self.knowledge_config.get("bias_threshold"), 0.7)
        self.directory_path = str(self.knowledge_config.get("directory_path", ""))
        self.embedding_model = self.knowledge_config.get("embedding_model")
        self.use_graph_ontology = bool(self.knowledge_config.get("use_graph_ontology", True))
        self.similarity_threshold = self._coerce_float(
            self.knowledge_config.get("similarity_threshold"), 0.3
        )
        self.bias_detection_enabled = bool(
            self.knowledge_config.get("bias_detection_enabled", False)
        )
        self.use_ontology_expansion = bool(
            self.knowledge_config.get("use_ontology_expansion", True)
        )

        self.persist_file = persist_file
        self.cache_lock = threading.RLock()
        self._index_lock = threading.RLock()
        self._query_executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self._token_cache_lock = threading.RLock()

        # Retain legacy/public attributes.
        self.knowledge_agent: List[Dict[str, Any]] = []
        self.cache = None
        self.learning_agent = None
        self.reasoning_agent = None

        self.knowledge_memory = knowledge_memory or self._create_knowledge_memory()
        self.knowledge_cache = knowledge_cache or self._create_knowledge_cache()
        self.perform_action = perform_action or self._create_perform_action()
        self.ontology_manager = ontology_manager or self._create_ontology_manager()
        self.rule_engine = rule_engine or self._create_rule_engine()
        self.governor = governor
        if self.governor is None:
            self.governor = self._initialize_governance()
        self.synchronizer = synchronizer or self._create_synchronizer(self.knowledge_memory, self.rule_engine)
        self.monitor = monitor or self._create_monitor(
            self.knowledge_cache, self.rule_engine, self.governor, self.perform_action
        )
        self.orchestrator = orchestrator or self._create_orchestrator(
            self.knowledge_memory,
            self.knowledge_cache,
            self.rule_engine,
            self.governor,
            self.synchronizer,
            self.monitor,
            self.perform_action,
        )
        self.cache = self.knowledge_cache

        # Bind shared dependencies into components that can benefit from them.
        if hasattr(self.perform_action, "knowledge_memory"):
            self.perform_action.knowledge_memory = self.knowledge_memory
        if hasattr(self.monitor, "agent"):
            self.monitor.agent = self

        self.sbert_model = None
        self.doc_embeddings: Dict[str, np.ndarray] = {}
        self.doc_tf_idf_vectors: Dict[str, Dict[str, float]] = {}
        self.doc_vectors: Dict[str, np.ndarray] = {}
        self.doc_index: Dict[str, Dict[str, Any]] = {}
        self.embedding_fallback = None
        self.vocabulary: set[str] = set()
        self.document_frequency: Dict[str, int] = defaultdict(int)
        self.safety_check_callback = None
        self.total_documents = 0
        self.token_cache: Dict[str, List[str]] = {}
        self.sorted_vocab: List[str] = []
        self._vocab_dirty = False
        self.content_hashes: set[str] = set()
        self.expanded_terms_cache: Dict[str, str] = {}
        self.ontology = defaultdict(lambda: {"type": None, "relations": set()})
        self.stopwords = self._load_stopwords(self.stopwords_path)
        self._retrieval_history_key = f"context:{self.name}"

        if self.retrieval_mode in {"dense", "hybrid"}:
            self._initialize_sbert_model()

        logger.info(
            "Knowledge Agent initialized with retrieval_mode=%s max_workers=%s dense_model=%s",
            self.retrieval_mode,
            self.max_workers,
            self.embedding_model if self.sbert_model is not None else "disabled",
        )

    # ------------------------------------------------------------------
    # Configuration and initialization helpers
    # ------------------------------------------------------------------

    def _create_knowledge_memory(self):
        from src.agents.knowledge.knowledge_memory import KnowledgeMemory

        return KnowledgeMemory()

    def _create_knowledge_cache(self):
        from src.agents.knowledge.knowledge_cache import KnowledgeCache

        return KnowledgeCache()

    def _create_perform_action(self):
        from src.agents.knowledge.perform_action import PerformAction

        return PerformAction()

    def _create_ontology_manager(self):
        from src.agents.knowledge.ontology_manager import OntologyManager

        return OntologyManager()

    def _create_rule_engine(self):
        from src.agents.knowledge.utils.rule_engine import RuleEngine

        return RuleEngine()

    def _create_synchronizer(self, knowledge_memory, rule_engine):
        from src.agents.knowledge.knowledge_sync import KnowledgeSynchronizer

        return KnowledgeSynchronizer(knowledge_memory=knowledge_memory, rule_engine=rule_engine, autostart=False)

    def _create_monitor(self, knowledge_cache, rule_engine, governor, perform_action):
        from src.agents.knowledge.knowledge_monitor import KnowledgeMonitor

        return KnowledgeMonitor(
            agent=self,
            knowledge_cache=knowledge_cache,
            rule_engine=rule_engine,
            governor=governor,
            perform_action=perform_action,
            autostart=False,
        )

    def _create_orchestrator(self, knowledge_memory, knowledge_cache, rule_engine, governor, synchronizer, monitor, perform_action):
        from src.agents.knowledge.knowledge_orchestrator import KnowledgeOrchestrator

        return KnowledgeOrchestrator(
            agent=self,
            memory=knowledge_memory,
            cache=knowledge_cache,
            rule_engine=rule_engine,
            governor=governor,
            synchronizer=synchronizer,
            monitor=monitor,
            action_executor=perform_action,
            lazy_start=False,
        )

    def _coerce_positive_int(self, value: Any, default: int) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = default
        return max(parsed, 1)

    def _coerce_float(self, value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @property
    def decay_factor(self):
        return self._decay_factor

    @decay_factor.setter
    def decay_factor(self, value):
        self._decay_factor = self._coerce_float(value, 0.8)

    def _initialize_governance(self):
        if self.governor is not None:
            return self.governor
        governor_config = self.config.get("governor", {}) if isinstance(self.config, dict) else {}
        if not isinstance(governor_config, dict):
            governor_config = {}
        if governor_config.get("enabled", True):
            from src.agents.knowledge.governor import Governor

            governor = Governor(knowledge_agent=self)
            logger.info("Governance subsystem initialized")
            return governor
        logger.info("Governance subsystem disabled")
        return None

    @classmethod
    def _get_or_create_embedding_model(cls, model_name: Optional[str]):
        if not model_name:
            return None
        with cls._embedding_model_lock:
            if model_name in cls._embedding_model_cache:
                return cls._embedding_model_cache[model_name]
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
            except Exception as import_error:
                logger.warning("SentenceTransformer import unavailable: %s", import_error)
                cls._embedding_model_cache[model_name] = None
                return None
            try:
                model = SentenceTransformer(
                    model_name,
                    device="cpu",
                    cache_folder="src/agents/knowledge/models/cache",
                )
            except Exception as exc:
                logger.warning("SBERT initialization failed for '%s': %s", model_name, exc)
                model = None
            cls._embedding_model_cache[model_name] = model
            return model

    def _initialize_sbert_model(self):
        self.sbert_model = self._get_or_create_embedding_model(self.embedding_model)
        if self.sbert_model is None and self.embedding_model:
            error = EmbeddingError(
                doc_id="<startup>",
                model_name=str(self.embedding_model),
                error_details="Unable to initialize dense embedding model",
            )
            error.report()

    def _load_stopwords(self, path: Optional[str]) -> set[str]:
        if not path:
            return set()
        resolved_path = Path(path)
        if not resolved_path.is_absolute():
            config_path = self.config.get("__config_path__") if isinstance(self.config, dict) else None
            candidates = [Path.cwd() / resolved_path]
            if config_path:
                config_dir = Path(config_path).resolve().parent
                candidates.extend([config_dir / resolved_path, config_dir.parent / resolved_path])
            resolved_path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
        try:
            with open(resolved_path, "r", encoding="utf-8") as handle:
                if resolved_path.suffix.lower() in {".yaml", ".yml"}:
                    data = yaml.safe_load(handle) or {}
                else:
                    data = json.load(handle)
            if isinstance(data, dict):
                if "stopwords" in data and isinstance(data["stopwords"], dict):
                    iterable = data["stopwords"].get("items", [])
                elif "items" in data:
                    iterable = data.get("items", [])
                elif "Stopword" in data:
                    iterable = data.get("Stopword", [])
                elif "stopwords" in data and isinstance(data["stopwords"], list):
                    iterable = data.get("stopwords", [])
                else:
                    iterable = data.keys()
            elif isinstance(data, list):
                iterable = data
            else:
                iterable = []
            return {str(token).strip().lower() for token in iterable if str(token).strip()}
        except Exception as exc:
            logger.warning("Failed to load stopwords from %s: %s", resolved_path, exc)
            return set()

    # ------------------------------------------------------------------
    # Document ingestion and indexing
    # ------------------------------------------------------------------
    def load_from_directory(self):
        """Load .txt and .json files from the configured directory."""
        if not self.directory_path:
            logger.warning("Knowledge directory path is not configured")
            return 0
        if not os.path.isdir(self.directory_path):
            logger.error("Invalid knowledge directory: %s", self.directory_path)
            return 0

        initial_count = len(self.knowledge_agent)
        for filename in os.listdir(self.directory_path):
            file_path = os.path.join(self.directory_path, filename)
            try:
                if filename.endswith(".txt"):
                    with open(file_path, "r", encoding="utf-8") as handle:
                        text = handle.read().strip()
                    self.add_document(
                        text,
                        metadata={
                            "source": filename,
                            "timestamp": time.time(),
                            "path": file_path,
                        },
                    )
                elif filename.endswith(".json"):
                    with open(file_path, "r", encoding="utf-8") as handle:
                        data = json.load(handle)
                    if isinstance(data, dict):
                        entries = data.get("documents") or data.get("items") or [data]
                    elif isinstance(data, list):
                        entries = data
                    else:
                        logger.warning("Skipping unsupported JSON structure in %s", filename)
                        continue
                    for entry in entries:
                        if isinstance(entry, dict):
                            text = entry.get("text") or entry.get("content") or json.dumps(entry, ensure_ascii=False)
                            metadata = dict(entry.get("metadata", {})) if isinstance(entry.get("metadata"), dict) else {}
                            metadata.setdefault("source", filename)
                            metadata.setdefault("path", file_path)
                        elif isinstance(entry, str):
                            text = entry
                            metadata = {"source": filename, "path": file_path}
                        else:
                            logger.warning("Skipping malformed entry in %s: %r", filename, entry)
                            continue
                        self.add_document(text, metadata=metadata)
            except Exception as exc:
                logger.error("Failed to load %s: %s", filename, exc, exc_info=True)

        return len(self.knowledge_agent) - initial_count

    def add_document(self, text, doc_id=None, metadata=None):
        """Store documents, generate TF-IDF, and optionally dense embeddings."""
        try:
            if isinstance(text, tuple) and len(text) == 3:
                subject, predicate, obj = text
                self.ontology_manager.add_triple(subject, predicate, obj)
                self.add_to_ontology(subject, predicate, obj)
                return

            if not isinstance(text, str) or len(text.strip()) < 3:
                raise InvalidDocumentError(text, "Document text must be a non-empty string of length >= 3")

            normalized_text = text.strip()
            content_hash = hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()
            with self._index_lock:
                if content_hash in self.content_hashes:
                    logger.debug("Document content already indexed; skipping duplicate")
                    return
                assigned_doc_id = str(doc_id or content_hash)
                if assigned_doc_id in self.doc_index:
                    logger.debug("Document id %s already exists; skipping duplicate", assigned_doc_id)
                    return

                tokens = self._preprocess(normalized_text)
                doc_metadata = dict(metadata or {})
                doc_metadata.setdefault("timestamp", time.time())
                doc_metadata.setdefault("source", self.source)
                if self.bias_detection_enabled and self.governor is not None:
                    doc_metadata["bias"] = self._detect_bias_metadata(normalized_text)

                document = {
                    "doc_id": assigned_doc_id,
                    "text": normalized_text,
                    "tokens": tokens,
                    "metadata": doc_metadata,
                }

                self.content_hashes.add(content_hash)
                self.knowledge_agent.append(document)
                self.doc_index[assigned_doc_id] = document
                self.total_documents += 1
                self._update_vocabulary(tokens)
                self.doc_tf_idf_vectors[assigned_doc_id] = self._calculate_tfidf(tokens)
                self.doc_vectors[assigned_doc_id] = self._dict_to_numpy(
                    self.doc_tf_idf_vectors[assigned_doc_id]
                )
                self._vocab_dirty = True

            self._maybe_compute_dense_embedding(assigned_doc_id, normalized_text)
        except InvalidDocumentError as exc:
            exc.report()
            logger.warning(str(exc))
        except OntologyError:
            raise
        except Exception as exc:
            error = MemoryUpdateError(
                key=str(doc_id or "<generated>"),
                value=text,
                error_details=str(exc),
            )
            error.report()
            logger.error("Document add failed: %s", exc, exc_info=True)
            raise error from exc

    def _update_vocabulary(self, tokens: Sequence[str]) -> None:
        unique_tokens = set(tokens)
        for token in unique_tokens:
            self.document_frequency[token] += 1
        self.vocabulary.update(unique_tokens)
        self.sorted_vocab = sorted(self.vocabulary)

    def _maybe_compute_dense_embedding(self, doc_id: str, text: str) -> None:
        if self.retrieval_mode not in {"dense", "hybrid"}:
            return
        if self.sbert_model is None:
            self._initialize_sbert_model()
        if self.sbert_model is None:
            return
        try:
            embedding = np.asarray(self.sbert_model.encode(text, show_progress_bar=False))
            with self._index_lock:
                self.doc_embeddings[doc_id] = embedding
        except Exception as exc:
            error = EmbeddingError(doc_id=doc_id, model_name=str(self.embedding_model), error_details=str(exc))
            error.report()
            logger.warning("Dense embedding generation failed for %s: %s", doc_id, exc)

    def retrieve_documents_by_type(self, doc_type: str) -> List[Dict[str, Any]]:
        documents = []
        for doc in self.knowledge_agent:
            if doc.get("metadata", {}).get("type") == doc_type:
                text = doc.get("text")
                if isinstance(text, str):
                    try:
                        documents.append(json.loads(text))
                    except json.JSONDecodeError:
                        documents.append({"text": text, "metadata": doc.get("metadata", {})})
        return documents

    def attach_reasoning(self, reasoning_agent):
        self.reasoning_agent = reasoning_agent
        logger.info("Reasoning agent attached to KnowledgeAgent")

    def attach_learning(self, learning_agent):
        self.learning_agent = learning_agent
        logger.info("Learning agent attached to KnowledgeAgent")

    def add_to_ontology(self, subject: str, predicate: str, obj: str):
        self.ontology[subject]["relations"].add((predicate, obj))
        if predicate in ("is_a", "type", "class"):
            self.ontology[subject]["type"] = obj

    # ------------------------------------------------------------------
    # Retrieval and query expansion
    # ------------------------------------------------------------------
    def fetch_web_content(self, url: str) -> str:
        try:
            import requests  # type: ignore

            response = requests.get(url, timeout=5)
            response.raise_for_status()
            return response.text
        except Exception as exc:
            error = RetrievalError(query=url, reason=str(exc), retrieval_mode="http")
            error.report()
            logger.error("Web retrieval failed: %s", exc)
            return ""

    def precompute_expansions(self, terms: set, max_terms=5000):
        significant_terms = sorted({str(term) for term in terms if str(term).strip()}, key=len, reverse=True)[:max_terms]
        with ThreadPoolExecutor(max_workers=min(8, self.max_workers)) as executor:
            futures = {executor.submit(self._compute_expansion, term): term for term in significant_terms}
            for future in as_completed(futures):
                term = futures[future]
                try:
                    self.expanded_terms_cache[term] = future.result()
                except Exception as exc:
                    logger.error("Expansion failed for %s: %s", term, exc)

    def _compute_expansion(self, term: str) -> str:
        expanded = set([term])
        visited = set()
        queue = deque([term])
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            if current in self.ontology:
                node = self.ontology[current]
                if node.get("type"):
                    queue.append(node["type"])
                    expanded.add(node["type"])
                for predicate, obj in node.get("relations", set()):
                    expanded.update([predicate, obj])
                    queue.append(obj)
            if self.use_ontology_expansion:
                try:
                    for related in self.ontology_manager.expand_query([current]):
                        if related not in visited:
                            expanded.add(related)
                            queue.append(related)
                except Exception as exc:
                    error = OntologyError("expand_query", current, str(exc))
                    error.report()
                    logger.warning("Ontology expansion failed for %s: %s", current, exc)
        return " ".join(sorted(expanded))

    def get_expansion(self, term: str) -> str:
        cached = self.expanded_terms_cache.get(term)
        if cached is not None:
            return cached
        expansion = self._compute_expansion(term)
        self.expanded_terms_cache[term] = expansion
        return expansion

    def _expand_with_ontology(self, terms: Sequence[str]) -> str:
        expanded = set()
        for term in terms:
            expanded.add(term)
            expanded.update(self.get_expansion(term).split())
            try:
                expanded.update(self.ontology_manager.get_types(term))
            except Exception as exc:
                error = OntologyError("get_types", term, str(exc))
                error.report()
                logger.warning("Ontology type lookup failed for %s: %s", term, exc)
        return " ".join(sorted(expanded))

    def predict(self, state: Any = None) -> Dict[str, Any]:
        query = "" if state is None else str(state)
        results = self.retrieve(query, k=1)
        if not results:
            return {"top_result": "No results found", "confidence_score": 0.0, "document_id": None}
        score, doc = results[0]
        return {
            "top_result": doc.get("text", ""),
            "confidence_score": float(score),
            "document_id": doc.get("doc_id"),
        }

    def retrieve(self, query, k=5):
        query = "" if query is None else str(query)
        if not query.strip():
            return []

        start_time = time.time()
        cache_key = self.knowledge_cache.hash_query(
            f"knowledge-agent:{self.retrieval_mode}:{k}:{len(self.knowledge_agent)}:{query}"
        )
        with self.cache_lock:
            cached_result = self.knowledge_cache.get(cache_key)
        if cached_result is not None:
            self.performance_metrics["cache_hits"].append(1)
            self.performance_metrics["retrieval_times"].append(time.time() - start_time)
            return cached_result
        self.performance_metrics["cache_hits"].append(0)

        if query.strip() == "*":
            results = [(1.0, doc) for doc in self.knowledge_agent[:k]]
            with self.cache_lock:
                self.knowledge_cache.set(cache_key, results)
            return results

        if not self.knowledge_agent:
            return []

        original_tokens = self._preprocess(query)
        if not original_tokens:
            return []

        expanded_query = query
        if self.use_ontology_expansion and self.first_pass:
            expanded_query = self._expand_with_ontology(original_tokens)

        tfidf_results = self._retrieve_tfidf(expanded_query)
        dense_results = self._retrieve_dense(expanded_query)
        combined = self._combine_results(tfidf_results, dense_results)
        final_results = combined[:k]

        if self.governor is not None:
            try:
                self.governor.audit_retrieval(
                    query=query,
                    results=final_results,
                    context={
                        "timestamp": time.time(),
                        "user": self._safe_shared_get("current_user"),
                        "module": "knowledge_retrieval",
                        "retrieval_mode": self.retrieval_mode,
                        "expanded_query_used": expanded_query != query,
                    },
                )
            except Exception as exc:
                violation = GovernanceViolation(
                    policy_name="retrieval_audit_failure",
                    violation_details={"error": str(exc)},
                    query=query,
                )
                violation.report()
                logger.warning("Retrieval audit failed: %s", exc)

        if self.bias_detection_enabled:
            try:
                final_results = self._apply_bias_analysis(final_results)
            except Exception as exc:
                error = BiasDetectionError(text_sample=query, error_details=str(exc))
                error.report()
                logger.warning("Bias analysis failed: %s", exc)

        serializable_results = [(float(score), doc) for score, doc in final_results]
        with self.cache_lock:
            self.knowledge_cache.set(cache_key, serializable_results)

        self._safe_shared_set(
            "knowledge:last_retrieval",
            {
                "query": query,
                "retrieval_mode": self.retrieval_mode,
                "timestamp": time.time(),
                "results": [
                    {
                        "score": float(score),
                        "doc_id": doc.get("doc_id"),
                        "text": doc.get("text"),
                        "metadata": doc.get("metadata", {}),
                    }
                    for score, doc in serializable_results
                ],
            },
        )
        self._safe_shared_set(
            "retrieved_knowledge",
            [doc.get("text", "") for _, doc in serializable_results],
        )
        self._safe_shared_increment(f"knowledge:metrics:{self.name}:retrieval_count", 1)
        self.performance_metrics["retrieval_times"].append(time.time() - start_time)
        return serializable_results

    def _retrieve_tfidf(self, query_text: str) -> List[Tuple[float, Dict[str, Any]]]:
        query_tokens = self._preprocess(query_text)
        if not query_tokens:
            return []
        query_vector = self._dict_to_numpy(self._calculate_tfidf(query_tokens))
        results = []
        for doc in self.knowledge_agent:
            doc_vector = self._dict_to_numpy(self.doc_tf_idf_vectors.get(doc["doc_id"], {}))
            similarity = cosine_sim(query_vector, doc_vector)
            if similarity >= self.similarity_threshold:
                results.append((float(similarity), doc))
        return sorted(results, key=lambda item: item[0], reverse=True)

    def _retrieve_dense(self, query_text: str) -> List[Tuple[float, Dict[str, Any]]]:
        if self.retrieval_mode not in {"dense", "hybrid"}:
            return []
        if self.sbert_model is None:
            self._initialize_sbert_model()
        if self.sbert_model is None or not self.doc_embeddings:
            return []
        try:
            query_embedding = np.asarray(self.sbert_model.encode(query_text, show_progress_bar=False))
        except Exception as exc:
            error = EmbeddingError(doc_id="<query>", model_name=str(self.embedding_model), error_details=str(exc))
            error.report()
            logger.warning("Dense query encoding failed: %s", exc)
            return []

        results = []
        for doc_id, doc_embedding in self.doc_embeddings.items():
            doc = self.doc_index.get(doc_id)
            if doc is None:
                continue
            similarity = cosine_sim(query_embedding, doc_embedding)
            if similarity >= self.similarity_threshold:
                results.append((float(similarity), doc))
        return sorted(results, key=lambda item: item[0], reverse=True)

    def _combine_results(
        self,
        tfidf_results: List[Tuple[float, Dict[str, Any]]],
        dense_results: List[Tuple[float, Dict[str, Any]]],
    ) -> List[Tuple[float, Dict[str, Any]]]:
        if self.retrieval_mode == "tfidf":
            return tfidf_results
        if self.retrieval_mode == "dense":
            return dense_results

        dense_weight = 0.7
        tfidf_weight = 0.3
        merged: Dict[str, Dict[str, Any]] = {}
        for score, doc in dense_results:
            merged.setdefault(doc["doc_id"], {"score": 0.0, "doc": doc})["score"] += score * dense_weight
        for score, doc in tfidf_results:
            merged.setdefault(doc["doc_id"], {"score": 0.0, "doc": doc})["score"] += score * tfidf_weight

        if self.retrieval_mode == "hybrid" and not merged:
            return tfidf_results or dense_results

        return sorted(
            [(payload["score"], payload["doc"]) for payload in merged.values()],
            key=lambda item: item[0],
            reverse=True,
        )

    def retrieve_batch(self, original_texts: List[str], k: int = 2) -> List[List[Tuple[float, Dict[str, Any]]]]:
        if not original_texts:
            return []
        if not self.knowledge_agent:
            return [[] for _ in original_texts]

        results: List[List[Tuple[float, Dict[str, Any]]]] = [[] for _ in original_texts]
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_index = {
                executor.submit(self.retrieve, query, k): idx
                for idx, query in enumerate(original_texts)
            }
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    error = RetrievalError(
                        query=str(original_texts[idx]),
                        reason=str(exc),
                        retrieval_mode="batch",
                    )
                    error.report()
                    logger.error("Batch retrieval failed for index %s: %s", idx, exc)
                    results[idx] = []
        return results

    def _dict_to_numpy(self, vector: Dict[str, float]) -> np.ndarray:
        if not self.sorted_vocab:
            self.sorted_vocab = sorted(self.vocabulary)
        return np.asarray([vector.get(term, 0.0) for term in self.sorted_vocab], dtype=float)

    def _preprocess(self, text: str) -> List[str]:
        normalized_text = str(text or "").strip().lower()
        if not normalized_text:
            return []
        with self._token_cache_lock:
            cached = self.token_cache.get(normalized_text)
            if cached is not None:
                return list(cached)
        cleaned_text = re.sub(r"[^\w\s]", " ", normalized_text)
        tokens = [token for token in cleaned_text.split() if token and token not in self.stopwords]
        with self._token_cache_lock:
            self.token_cache[normalized_text] = list(tokens)
        return tokens

    def _calculate_tfidf(self, tokens: List[str]) -> Dict[str, float]:
        if not tokens:
            return {}
        counts = Counter(tokens)
        total_terms = float(len(tokens))
        vector: Dict[str, float] = {}
        for token, count in counts.items():
            term_tf = count / total_terms
            idf = math.log((self.total_documents + 1) / (self.document_frequency.get(token, 0) + 1)) + 1
            vector[token] = term_tf * idf
        return vector

    def _cosine_similarity(self, vec_a, vec_b):
        all_terms = sorted(set(vec_a.keys()) | set(vec_b.keys()))
        if not all_terms:
            return 0.0
        a = np.asarray([vec_a.get(term, 0.0) for term in all_terms], dtype=float)
        b = np.asarray([vec_b.get(term, 0.0) for term in all_terms], dtype=float)
        return cosine_sim(a, b)

    # ------------------------------------------------------------------
    # Knowledge memory integration
    # ------------------------------------------------------------------
    def update_memory(
        self,
        key: str,
        value: Any,
        metadata: dict = None,
        context: dict = None,
        ttl: int = None,
    ):
        try:
            self.knowledge_memory.update(key=key, value=value, metadata=metadata, context=context, ttl=ttl)
            recalled = self.knowledge_memory.recall(key=key)
            _, stored_metadata = recalled[0] if recalled else (None, {})
            return {
                "key": key,
                "value": value,
                "context": context,
                "metadata": stored_metadata,
                "relevance": stored_metadata.get("relevance", 1.0),
                "ttl": ttl,
            }
        except Exception as exc:
            error = MemoryUpdateError(key=key, value=value, error_details=str(exc))
            error.report()
            logger.error("Memory update failed: %s", exc, exc_info=True)
            raise error from exc

    def recall_memory(
        self,
        key: str = None,
        filters: dict = None,
        sort_by: str = None,
        top_k: int = None,
    ) -> list:
        return self.knowledge_memory.recall(key=key, filters=filters, sort_by=sort_by, top_k=top_k)

    def _apply_filters(self, metadata: dict, filters: dict) -> bool:
        return all(metadata.get(k) == v for k, v in (filters or {}).items())

    # ------------------------------------------------------------------
    # Higher-level behaviors
    # ------------------------------------------------------------------
    def _detect_bias_metadata(self, text: str) -> dict:
        if self.governor is None:
            return {"detected": False, "confidence": 0.0, "categories": {}}
        try:
            confidence = float(self.governor._detect_unethical_content(text))
            categories = dict(self.governor._detect_bias(text))
            return {
                "detected": confidence > self.bias_threshold,
                "confidence": confidence,
                "categories": categories,
            }
        except Exception as exc:
            error = BiasDetectionError(text_sample=text, error_details=str(exc))
            error.report()
            logger.warning("Bias metadata detection failed: %s", exc)
            return {"detected": False, "confidence": 0.0, "categories": {}}

    def _apply_bias_analysis(self, results: list) -> list:
        analyzed_results = []
        for score, doc in results:
            bias_meta = doc.get("metadata", {}).get("bias", {})
            doc_copy = dict(doc)
            doc_copy["bias_summary"] = {
                "is_biased": bias_meta.get("detected", False),
                "confidence": bias_meta.get("confidence", 0.0),
                "dominant_category": max(
                    bias_meta.get("categories", {}).items(),
                    key=lambda item: item[1],
                    default=("none", 0),
                )[0],
            }
            analyzed_results.append((score, doc_copy))
        return analyzed_results

    def contextual_search(self, query):
        context = self._safe_shared_get(self._retrieval_history_key)
        if not isinstance(context, deque):
            context = deque(maxlen=self.context_window)

        current_tokens = self._preprocess(query)
        current_terms = self._extract_significant_terms(current_tokens, top_n=10)
        context.append({"timestamp": time.time(), "terms": current_terms, "raw_query": query})
        self._safe_shared_set(self._retrieval_history_key, context)

        augmented_terms = []
        for idx, entry in enumerate(reversed(context)):
            weight = self.decay_factor ** idx
            augmented_terms.extend([term for term in entry["terms"] for _ in range(max(int(weight * 10), 1))])
        augmented_terms.extend(current_terms * 10)
        if self.bias_detection_enabled:
            augmented_terms.extend(["unbiased", "factual"])
        return self.retrieve(" ".join(augmented_terms))

    def _extract_significant_terms(self, tokens, top_n=5):
        tfidf = self._calculate_tfidf(tokens)
        sorted_terms = sorted(tfidf.items(), key=lambda item: item[1], reverse=True)
        return [term for term, _ in sorted_terms[:top_n]]

    def get_references_for_concepts(self, concepts: list, k: int = 3) -> list:
        references = []
        for concept in concepts:
            if not isinstance(concept, str) or not concept.strip():
                continue
            for _, doc in self.retrieve(concept, k=k):
                references.append(doc.get("text", ""))
        return references

    def broadcast_knowledge(self, context=None, ttl: int = 300):
        broadcast_keys = []
        for idx, doc in enumerate(self.knowledge_agent[-5:]):
            key = f"{self.knowledge_tag}:{idx}"
            text = doc.get("text", "")
            metadata = doc.get("metadata", {})
            enriched = {
                "value": text,
                "metadata": {
                    "timestamp": time.time(),
                    "tags": self.knowledge_tag if isinstance(self.knowledge_tag, list) else [self.knowledge_tag],
                    "source": self.source,
                    "context": context,
                    **metadata,
                },
            }
            self._safe_shared_set(key, enriched, ttl=ttl)
            self._safe_shared_publish(
                "broadcast_channel",
                {"key": key, "value": text, "metadata": enriched["metadata"]},
            )
            broadcast_keys.append(key)
        return broadcast_keys

    def discover_rules(self) -> list:
        if self.governor is not None:
            try:
                raw_rules = self.governor.get_approved_rules()
                return self._process_rules(raw_rules)
            except Exception as exc:
                logger.warning("Governor rule discovery failed: %s", exc)
        try:
            if hasattr(self.rule_engine, "get_rules_by_category"):
                rules = self.rule_engine.get_rules_by_category("governance_approved")
            else:
                rules = []
            return self._process_rules(rules)
        except Exception as exc:
            logger.error("Rule discovery failed: %s", exc)
            return []

    def _process_rules(self, raw_rules: list) -> list:
        min_confidence = 0.7
        processed_rules = []
        for rule in raw_rules:
            if not isinstance(rule, dict):
                continue
            if float(rule.get("confidence", 1.0)) < min_confidence:
                continue
            rule_copy = dict(rule)
            rule_id = rule_copy.get("id")
            if rule_id:
                try:
                    matches = self.recall_memory(filters={"type": "system_rule", "id": rule_id})
                    if matches:
                        _, metadata = matches[0]
                        rule_copy["metadata"] = metadata
                except Exception as exc:
                    logger.warning("Failed to annotate rule %s: %s", rule_id, exc)
            processed_rules.append(rule_copy)
        return processed_rules

    def respond_to_query(self, query):
        context = {
            "user": self._safe_shared_get("current_user"),
            "module": "query_processing",
            "timestamp": time.time(),
        }
        retrieved = self.retrieve(query)
        if not retrieved:
            return {"response": "No relevant information found", "confidence": 0.0}

        audit_report = {"violations": []}
        if self.governor is not None:
            try:
                audit_report = self.governor.audit_retrieval(query=query, results=retrieved, context=context)
                violations = audit_report.get("violations", [])
                critical_threshold = getattr(self.governor.violation_thresholds, "critical", 0.8)
                if any(v.get("score", 0.0) > critical_threshold for v in violations):
                    return {
                        "response": "Response blocked due to policy violations",
                        "violations": violations,
                    }
            except Exception as exc:
                violation = GovernanceViolation(
                    policy_name="query_response_audit_failure",
                    violation_details={"error": str(exc)},
                    query=query,
                )
                violation.report()

        bias_analysis = []
        if self.bias_detection_enabled:
            for _, doc in retrieved:
                bias_meta = doc.get("metadata", {}).get("bias", {})
                if bias_meta.get("detected"):
                    bias_analysis.append(
                        {
                            "source": doc.get("metadata", {}).get("source"),
                            "categories": bias_meta.get("categories", {}),
                        }
                    )

        current_time = time.time()
        freshness_threshold_seconds = 720 * 3600
        if self.governor is not None:
            freshness_threshold_seconds = int(self.governor.freshness_threshold) * 3600
        fresh_results = []
        for score, doc in retrieved:
            doc_time = doc.get("metadata", {}).get("timestamp", 0) or 0
            if current_time - float(doc_time) < freshness_threshold_seconds:
                fresh_results.append((score, doc))
        final_results = fresh_results if fresh_results else retrieved

        response_text = "\n\n".join(doc.get("text", "") for _, doc in final_results[:3])
        action_required = self._detect_action_trigger(query, final_results)
        if action_required:
            action_docs = [{"text": doc.get("text", "")} for _, doc in final_results]
            action_result = self.orchestrator.execute_actions(action_docs)
            self.update_memory(
                key=f"knowledge:actions:{int(time.time())}",
                value={
                    "trigger": action_required,
                    "query": query,
                    "results_used": len(final_results),
                    "action_result": action_result,
                },
                metadata={"type": "action_execution", "source": "knowledge_agent"},
                context={"query": query, "source": "respond_to_query"},
                ttl=3600,
            )
            return {
                "response": response_text,
                "confidence": float(final_results[0][0]) if final_results else 0.0,
                "action": action_required,
                "action_result": action_result,
                "bias_analysis": bias_analysis,
                "violations": audit_report.get("violations", []),
            }

        return {
            "response": response_text,
            "confidence": float(final_results[0][0]) if final_results else 0.0,
            "sources": [doc.get("metadata", {}).get("source", "unknown") for _, doc in final_results],
            "bias_analysis": bias_analysis,
            "violations": audit_report.get("violations", []),
        }

    def _detect_action_trigger(self, query, results):
        action_triggers = {
            "update",
            "modify",
            "create",
            "delete",
            "execute",
            "perform",
            "run",
            "notify",
            "alert",
            "change",
        }
        query_lower = str(query or "").lower()
        if any(trigger in query_lower for trigger in action_triggers):
            return "user_requested_action"

        for _, doc in results:
            content = doc.get("text", "").lower()
            if "required action:" in content or "next steps:" in content:
                return "content_suggested_action"

        if self.governor is not None and getattr(self.governor, "rule_engine", None) is not None:
            try:
                inferences = self.governor.rule_engine.apply(
                    {
                        "query": query,
                        "results": [doc.get("text", "") for _, doc in results],
                    }
                )
                if isinstance(inferences, dict) and inferences.get("action_required"):
                    return inferences["action_required"]
            except Exception as exc:
                logger.warning("Action inference failed: %s", exc)
        return None

    def _calculate_relevance(self, value: Any, context: Optional[dict]) -> float:
        if not context:
            return 0.5
        value_text = str(value)
        context_query = context.get("query", "") or context.get("source", "")
        if not value_text.strip() or not str(context_query).strip():
            return 0.25

        score = self._fallback_relevance_tfidf(value_text, str(context_query))
        if self.retrieval_mode in {"dense", "hybrid"}:
            if self.sbert_model is None:
                self._initialize_sbert_model()
            if self.sbert_model is not None:
                try:
                    emb_value = self.sbert_model.encode(value_text, show_progress_bar=False)
                    emb_context = self.sbert_model.encode(str(context_query), show_progress_bar=False)
                    score = cosine_sim(emb_value, emb_context)
                except Exception as exc:
                    logger.warning("SBERT relevance calculation failed: %s", exc)

        if self.bias_detection_enabled and self.governor is not None:
            try:
                bias_conf = float(self.governor._detect_unethical_content(value_text))
                if bias_conf > self.bias_threshold:
                    score *= 1.0 - min(bias_conf * 0.5, 0.3)
            except Exception as exc:
                logger.warning("Bias-aware relevance penalty failed: %s", exc)

        return round(max(0.0, min(float(score), 1.0)), 4)

    def _fallback_relevance_tfidf(self, text1: str, text2: str) -> float:
        tokens1 = self._preprocess(text1)
        tokens2 = self._preprocess(text2)
        if not tokens1 or not tokens2:
            return 0.0
        return self._cosine_similarity(self._calculate_tfidf(tokens1), self._calculate_tfidf(tokens2))

    def _validate_with_safety(self, action_params: Dict, context: Optional[Dict] = None) -> bool:
        if not self.safety_check_callback:
            return True
        try:
            result = self.safety_check_callback(action_params, context or {})
            return bool(result.get("approved", False))
        except Exception as exc:
            logger.error("Safety validation failed: %s", exc)
            return False

    def register_safety_check(self, safety_check_callback: Callable):
        if not callable(safety_check_callback):
            raise TypeError("Safety callback must be callable")
        test_result = safety_check_callback(
            {"action": "test_validation"},
            {"context": "signature_verification"},
        )
        if not isinstance(test_result, dict) or "approved" not in test_result:
            raise ValueError("Callback must return dict with 'approved' key")
        self.safety_check_callback = safety_check_callback
        logger.info("Safety validation callback registered successfully")

    def _tokenizer(self):
        global _TOKENIZER_INSTANCE
        if _TOKENIZER_INSTANCE is not None:
            return _TOKENIZER_INSTANCE
        if not self.embedding_model:
            return None
        try:
            from transformers import AutoTokenizer  # type: ignore

            _TOKENIZER_INSTANCE = AutoTokenizer.from_pretrained(self.embedding_model)
            return _TOKENIZER_INSTANCE
        except Exception as exc:
            logger.warning("Tokenizer load failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Shared memory helpers
    # ------------------------------------------------------------------
    def _safe_shared_get(self, key: str, default: Any = None) -> Any:
        try:
            if hasattr(self.shared_memory, "get"):
                value = self.shared_memory.get(key)
                return default if value is None else value
        except Exception:
            return default
        return default

    def _safe_shared_set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        try:
            if hasattr(self.shared_memory, "set"):
                if ttl is not None:
                    self.shared_memory.set(key, value, ttl=ttl)
                else:
                    self.shared_memory.set(key, value)
        except TypeError:
            if hasattr(self.shared_memory, "set"):
                self.shared_memory.set(key, value)
        except Exception as exc:
            logger.warning("Shared memory set failed for %s: %s", key, exc)

    def _safe_shared_increment(self, key: str, amount: int) -> None:
        try:
            if hasattr(self.shared_memory, "increment"):
                self.shared_memory.increment(key, amount)
                return
        except Exception:
            pass
        current = self._safe_shared_get(key, 0)
        try:
            self._safe_shared_set(key, int(current) + amount)
        except Exception:
            pass

    def _safe_shared_publish(self, channel: str, payload: Any) -> None:
        try:
            if hasattr(self.shared_memory, "publish"):
                self.shared_memory.publish(channel, payload)
        except Exception as exc:
            logger.warning("Shared memory publish failed for %s: %s", channel, exc)

    def shutdown(self) -> None:
        try:
            self.orchestrator.stop()
        finally:
            self._query_executor.shutdown(wait=False)


if __name__ == "__main__":  # pragma: no cover
    print("\n=== Running Knowledge Agent ===\n")
    printer.status("Init", "Knowledge Agent initialized", "success")
    from src.agents.collaborative.shared_memory import SharedMemory
    from src.agents.agent_factory import AgentFactory

    shared_memory = SharedMemory()
    agent_factory = AgentFactory()
    agent = KnowledgeAgent(shared_memory=shared_memory, agent_factory=agent_factory)

    loaded = agent.load_from_directory()
    printer.pretty("LOADER", f"Loaded {loaded} documents", "success" if loaded else "info")
    query = "AI ethics principles for autonomous systems"
    retriever = agent.retrieve(query=query, k=5)
    printer.status("RETRIEVE", retriever, "success" if retriever else "error")
    print("\n=== Successfully ran the Knowledge Agent ===\n")
