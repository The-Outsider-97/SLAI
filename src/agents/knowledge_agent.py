__version__ = "1.9.0"

"""
Knowledge Agent for SLAI (Scalable Learning Autonomous Intelligence)
Implements core RAG functionality with TF-IDF based retrieval from scratch

Real-World Usage:
1. Medical Decision Support: Retrieve clinical guidelines while inferring drug interactions using ontology relationships
2. Legal Research Assistants: Expand queries like "copyright law" to include related statutes (via ontology) and infer precedent applicability (via rules).
3. Supply Chain Optimization: Use rule-based inference to predict bottlenecks from logistics documents.
4. AI Governance: Audit AI behavior by cross-referencing ethical guidelines (e.g., detecting bias via alignment_thresholds in config.yaml).
"""

import os
import re
import json
import math
import time
import hashlib
import uuid
import threading
import numpy as np
import pandas as pd

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from transformers import AutoTokenizer
from heapq import nlargest
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict, deque
from sentence_transformers import SentenceTransformer

from src.agents.base.utils.main_config_loader import load_global_config, get_config_section
from src.agents.base_agent import BaseAgent
from src.agents.knowledge.utils.knowledge_errors import (RetrievalError, OntologyError, BiasDetectionError,
                                                         GovernanceViolation, CacheError, MemoryUpdateError,
                                                         InvalidDocumentError, EmbeddingError)
from src.agents.knowledge.knowledge_cache import KnowledgeCache
from src.agents.knowledge.perform_action import PerformAction
from src.agents.knowledge.ontology_manager import OntologyManager
from src.agents.knowledge.governor import Governor
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Knowledge Agent")
printer = PrettyPrinter

_tokenizer_instance = None

def cosine_sim(v1, v2):
    v1, v2 = np.array(v1), np.array(v2)
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return dot_product / (norm_v1 * norm_v2)

class KnowledgeAgent(BaseAgent):
    def __init__(self, shared_memory,
                 agent_factory,
                 config=None,
                 persist_file: str = None):
        super().__init__(
            shared_memory=shared_memory,
            agent_factory=agent_factory,
            config=config
        )
        self.knowledge_agent = []
        self.cache = None
        self.learning_agent = None
        self.reasoning_agent = None
        self.shared_memory = shared_memory
        self.agent_factory = agent_factory
        self.config = load_global_config()
        self.knowledge_config = get_config_section('knowledge_agent')
        self.source = self.knowledge_config.get('source')
        self.tokenize = self.knowledge_config.get('tokenize')
        self.is_query = self.knowledge_config.get('is_query')
        self.stopwords = self.knowledge_config.get('stopwords')
        self.cache_size = self.knowledge_config.get('cache_size')
        self.first_pass = self.knowledge_config.get('first_pass')
        self.max_workers = self.knowledge_config.get('max_workers')
        self._decay_factor = self.knowledge_config.get('decay_factor')
        self.knowledge_tag = self.knowledge_config.get('knowledge_tag')
        self.retrieval_mode = self.knowledge_config.get('retrieval_mode')
        self.context_window = self.knowledge_config.get('context_window')
        self.bias_threshold = self.knowledge_config.get('bias_threshold')
        self.directory_path = self.knowledge_config.get('directory_path')
        self.embedding_model = self.knowledge_config.get('embedding_model')
        self.use_graph_ontology = self.knowledge_config.get('use_graph_ontology')
        self.similarity_threshold = self.knowledge_config.get('similarity_threshold')
        self.bias_detection_enabled = self.knowledge_config.get('bias_detection_enabled')
        self.use_ontology_expansion = self.knowledge_config.get('use_ontology_expansion')

        if not self.embedding_model:
            logger.error("KnowledgeAgent misconfigured: 'embedding_model' is not set in config.")

        self.knowledge_cache = KnowledgeCache()
        self.perform_action = PerformAction()
        self.ontology_manager = OntologyManager()
        self.cache_lock = threading.Lock()

        self.sbert_model: Optional[SentenceTransformer] = None
        self.doc_embeddings: Dict[str, np.ndarray] = {}
        self.persist_file = persist_file
        self._initialize_sbert_model()
        self.ontology = defaultdict(lambda: {'type': None, 'relations': set()})
        self.embedding_fallback = None
        self.vocabulary = set()
        self.document_frequency = defaultdict(int)
        self.safety_check_callback = None
        self.total_documents = 0
        self.doc_vectors = {}
        self.doc_tf_idf_vectors = {}
        self.token_cache = {}
        self.sorted_vocab = []
        self.content_hashes = set()
        self.expanded_terms_cache = {}
        
        self.governor = self._initialize_governance()
        self.stopwords = self._load_stopwords(self.stopwords)

        required_configs = [
            'similarity_threshold', 
            'bias_threshold',
            'cache_size'
        ]
        for param in required_configs:
            current_value = getattr(self, param, None)
            if current_value is None:
                logger.debug(f"Missing config '{param}', using default")
                if param == 'similarity_threshold':
                    setattr(self, param, 0.3)
                elif param == 'bias_threshold':
                    setattr(self, param, 0.7)
                elif param == 'cache_size':
                    setattr(self, param, 1000)

        logger.info(f"Knowledge Agent Initialized with SBERT model: {self.embedding_model if self.sbert_model else 'Failed to load'}")

    @property
    def decay_factor(self):
        return self._decay_factor
    
    @decay_factor.setter
    def decay_factor(self, value):
        self._decay_factor = value

    def _initialize_sbert_model(self):
        """Initializes the SentenceTransformer model with robust error handling."""
        try:
            # Get model path from config with fallback
            model_path = self.embedding_model or "sentence-transformers/all-MiniLM-L6-v2"
            
            if not model_path:
                logger.error("SBERT initialization failed: No embedding_model configured")
                self.sbert_model = None
                return
    
            # Check if path is valid
            is_local_path = os.path.exists(model_path)
            
            # Log model source
            source_type = "local" if is_local_path else "Hugging Face Hub"
            logger.info(f"Loading SBERT model from {source_type}: {model_path}")
            
            # Load model with timeout
            self.sbert_model = SentenceTransformer(
                model_path, 
                device='cpu',  # Ensures compatibility
                cache_folder="src/agents/knowledge/models/cache"
            )
            logger.info(f"SBERT model loaded successfully. Embedding dimension: {self.sbert_model.get_sentence_embedding_dimension()}")
            
        except Exception as e:
            logger.error(f"SBERT initialization failed: {str(e)}")
            logger.info("Attempting fallback to default model")
            
            try:
                # Fallback to lightweight model
                self.sbert_model = SentenceTransformer(
                    "sentence-transformers/all-MiniLM-L6-v2",
                    device='cpu'
                )
                logger.warning("Using fallback SBERT model: sentence-transformers/all-MiniLM-L6-v2")
            except Exception as fallback_error:
                logger.critical(f"Fallback model failed: {str(fallback_error)}")
                self.sbert_model = None

    def _initialize_governance(self):
        """Conditionally create governor based on config"""
        if self.config.get('governor.enabled', True):
            governor = Governor(knowledge_agent=self)
            logger.info("Governance subsystem initialized")
            return governor
        logger.info("Governance subsystem disabled")
        return None
    
    def _load_stopwords(self, path: str) -> list:
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                return data.get('Stopword', [])
        except Exception as e:
            logger.error(f"Failed to load stopwords: {e}")
            return []

    def load_from_directory(self):
        """Loads all .txt and .json files in the directory as knowledge documents."""
        logger.debug(f"Loading documents from: {self.directory_path}")
        if not os.path.isdir(self.directory_path):
            logger.error(f"Invalid directory: {self.directory_path}")
            return 0
    
        initial_count = len(self.knowledge_agent)
    
        for fname in os.listdir(self.directory_path):
            fpath = os.path.join(self.directory_path, fname)
            try:
                if fname.endswith(".txt"):
                    with open(fpath, "r", encoding="utf-8") as f:
                        text = f.read().strip()
                        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
                        self.add_document(text, doc_id=content_hash, metadata={
                            "source": fname,
                            "timestamp": time.time(),
                            "checksum": content_hash
                        })
                elif fname.endswith(".json"):
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        for entry in data:
                            if isinstance(entry, dict):
                                text = entry.get("text") or str(entry)
                            elif isinstance(entry, str):
                                text = entry
                            else:
                                logger.warning(f"Unexpected entry format in {fname}: {type(entry)} — {entry}")
                                continue
                            content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
                            self.add_document(text, doc_id=content_hash, metadata={"source": fname})
            except Exception as e:
                logger.error(f"Failed to load {fname}: {str(e)}", exc_info=True)
    
        return len(self.knowledge_agent) - initial_count
    
    def predict(self, state: Any = None) -> Dict[str, Any]:
        """
        Predicts the top knowledge result for a given query (state).
        
        Args:
            state (Any, optional): The query string. Defaults to None.
            
        Returns:
            Dict[str, Any]: Prediction results with keys:
                - top_result: Text of top document
                - confidence_score: Similarity score
                - document_id: ID of the document
        """
        if state is None:
            state = ""
    
        # Retrieve top result
        results = self.retrieve(state, k=1)
        
        if not results:
            return {
                "top_result": "No results found",
                "confidence_score": 0.0,
                "document_id": None
            }
            
        score, doc = results[0]
        return {
            "top_result": doc['text'],
            "confidence_score": float(score),
            "document_id": doc['doc_id']
        }
    
    def _tokenizer(self):
        global _tokenizer_instance
        if _tokenizer_instance is not None:
            return _tokenizer_instance
    
        try:
            model_path = self.embedding_model
    
            if not model_path or not os.path.exists(model_path):
                raise FileNotFoundError(f"Tokenizer path not found: {model_path}")
    
            _tokenizer_instance = AutoTokenizer.from_pretrained(model_path)
            return _tokenizer_instance
    
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            return None
    
    def retrieve_documents_by_type(self, doc_type: str) -> List[Dict]:
        return [
            json.loads(doc["text"]) for doc in self.knowledge_agent
            if doc.get("metadata", {}).get("type") == doc_type
        ]

    def add_document(self, text, doc_id=None, metadata=None):
        """Store documents, generate TF-IDF, and dense embeddings."""
        # Generate content-based hash for deduplication
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        
        # Skip if content already exists (regardless of doc_id)
        if content_hash in self.content_hashes:
            logger.debug(f"Document content already exists. Skipping.")
            return
        self.content_hashes.add(content_hash)
    
        doc_id = doc_id or content_hash # str(uuid.uuid4())

        # Skip very short documents
        if not isinstance(text, str) or len(text.strip()) < 3:
            logger.debug(f"Skipping short document: '{text}'")
            return

        if isinstance(text, tuple) and len(text) == 3:
            self.ontology_manager.add_triple(*text)
            self.add_to_ontology(*text)
            return
        if not isinstance(text, str) or len(text.strip()) < 3:
            logger.warning(f"Document text too short or not a string: '{text}'")
            return

        # Check for existence AFTER initial processing and ID generation.
        # This prevents duplicate processing.
        if doc_id in self.doc_tf_idf_vectors:
            logger.warning(f"Document ID {doc_id} already exists. Skipping.")
            return

        # --- Dense Embedding Generation ---
        if self.sbert_model:
            try:
                embedding = self.sbert_model.encode(text, show_progress_bar=False)
                self.doc_embeddings[doc_id] = embedding
            except Exception as e:
                logger.error(f"Error generating SBERT embedding for doc_id {doc_id}: {e}")
        else:
            logger.warning(f"SBERT model not available. Cannot generate dense embedding for doc_id {doc_id}.")

        # Bias detection
        if self.bias_detection_enabled and self.governor:
            bias_metadata = self._detect_bias_metadata(text)
            if metadata is None:
                metadata = {}
            metadata['bias'] = bias_metadata

        tokens = self._preprocess(text)
        self.knowledge_agent.append({
            'doc_id': doc_id,
            'text': text,
            'tokens': tokens,
            'metadata': metadata or {}
        })

        # --- TF-IDF Vector Generation ---
        unique_tokens = set(tokens)
        for token in unique_tokens:
            self.document_frequency[token] += 1
        self.vocabulary.update(unique_tokens)
        self.total_documents += 1

        tf_idf_vector_dict = self._calculate_tfidf(tokens)
        self.doc_tf_idf_vectors[doc_id] = tf_idf_vector_dict

        # Safety validation before adding document
        if self.safety_check_callback:
            validation = self._validate_with_safety(
                {"operation": "add_document", "content": text[:500]},
                {"source": "knowledge_agent"}
            )
            if not validation:
                logger.warning("Document addition blocked by safety check")
                return  # Skip adding document
        
        logger.debug(f"Added document {doc_id}. Total docs: {self.total_documents}. Vocab size: {len(self.vocabulary)}")
        
    def attach_reasoning(self, reasoning_agent):
        """Connect ReasoningAgent to KnowledgeAgent"""
        self.reasoning_agent = reasoning_agent
        logger.info("Reasoning agent attached to KnowledgeAgent")

    def add_to_ontology(self, subject: str, predicate: str, obj: str):
        self.ontology[subject]['relations'].add((predicate, obj))
        if predicate in ('is_a', 'type', 'class'):
            self.ontology[subject]['type'] = obj

    def _detect_bias_metadata(self, text: str) -> dict:
        """Detect bias in text and return structured metadata"""
        if not self.governor:
            return {"detected": False, "confidence": 0.0}
        
        # Use Governor's bias detection capabilities
        return {
            "detected": self.governor._detect_unethical_content(text) > self.bias_threshold,
            "confidence": self.governor._detect_unethical_content(text),
            "categories": self.governor._detect_bias(text)
        }
    
    def fetch_web_content(self, url: str) -> str:
        """Fetch web content from URL"""
        try:
            import requests
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.error(f"Web retrieval failed: {str(e)}")
            return ""
        
    def precompute_expansions(self, terms: set, max_terms=5000):
        """Batch precompute with size limits and parallelization"""
        # Filter to most significant terms
        significant_terms = nlargest(max_terms, terms, key=lambda term: len(term))
        
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(self._compute_expansion, term): term for term in significant_terms}
            for future in tqdm(as_completed(futures), total=len(significant_terms), desc="Precomputing expansions"):
                term = futures[future]
                try:
                    self.expanded_terms_cache[term] = future.result()
                except Exception as e:
                    logger.error(f"Expansion failed for {term}: {e}")
    
    def _compute_expansion(self, term: str) -> str:
        """Non-recursive expansion with cycle protection"""
        expanded = set([term])
        visited = set()
        queue = deque([term])
        
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            
            # Local ontology
            if current in self.ontology:
                # Add type
                if (t := self.ontology[current].get('type')):
                    expanded.add(t)
                    queue.append(t)
                    
                # Add relations
                for pred, obj in self.ontology[current].get('relations', set()):
                    expanded.add(pred)
                    expanded.add(obj)
                    queue.append(obj)
            
            # Ontology manager
            for related in self.ontology_manager.expand_query([current]):
                expanded.add(related)
                queue.append(related)
                
        return " ".join(expanded)
            
    def get_expansion(self, term: str) -> str:
        """Get cached expansion or return term if not found"""
        return self.expanded_terms_cache.get(term, term)

    def retrieve(self, query, k=5):
        cache_key = hashlib.sha256(query.encode()).hexdigest()
        start_time = time.time()
        cache_hit = False

        # Clear cache periodically
        if len(self.expanded_terms_cache) > 100000:
            self.expanded_terms_cache.clear()
        
        # Use lock for thread-safe cache access
        with self.cache_lock:
            cached_result = self.knowledge_cache.get(cache_key)
            if cached_result is not None:
                cache_hit = True
                self.performance_metrics['cache_hits'].append(1)
                self.performance_metrics['retrieval_times'].append(time.time() - start_time)
                return cached_result
            else:
                self.performance_metrics['cache_hits'].append(0)

        # Handle wildcard query - return all documents
        if query.strip() == "*":
            return [
                (1.0, doc)  # Max similarity score
                for doc in self.knowledge_agent
            ][:k]

        # Preprocess and validate
        query_tokens = self._preprocess(query)
        if not query_tokens or not self.knowledge_agent:
            return []

        # Calculate query vector
        query_vector = self._dict_to_numpy(
            self._calculate_tfidf(
                query_tokens
        ))

        # Calculate document vectors and similarities
        similarities = []
        current_vocab = sorted(self.vocabulary)  # Get current vocabulary
        for doc in self.knowledge_agent:
            doc_vec_dict = self.doc_tf_idf_vectors.get(doc['doc_id'], {})
            doc_vec = np.array([
                doc_vec_dict.get(term, 0) for term in current_vocab
            ])
            if doc_vec.size == 0:
                continue
            similarity = cosine_sim(query_vector, doc_vec)
            if similarity >= self.similarity_threshold:
                similarities.append((similarity, doc))

        results = nlargest(k, similarities, key=lambda x: x[0])

        if self.use_ontology_expansion and self.first_pass:
            query_terms = self._preprocess(query)
            expanded_terms = [self.get_expansion(term) for term in query_terms]
            expanded_query_text = " ".join(expanded_terms)
            logger.debug(f"Original query: '{query}', Expanded query for retrieval: '{expanded_query_text}'")

        # --- Dense Retrieval ---
        dense_similarities = []
        if self.retrieval_mode in ["dense", "hybrid"] and self.sbert_model and self.doc_embeddings:
            try:
                query_embedding = self.sbert_model.encode(expanded_query_text, show_progress_bar=False)
                for doc_id, doc_embedding in self.doc_embeddings.items():
                    doc_details = next((d for d in self.knowledge_agent if d['doc_id'] == doc_id), None)
                    if doc_details:
                        similarity = cosine_sim(query_embedding, doc_embedding)
                        if similarity >= self.similarity_threshold:
                            dense_similarities.append((similarity, doc_details))
            except Exception as e:
                logger.error(f"Error during dense retrieval: {e}")
        elif self.retrieval_mode in ["dense", "hybrid"] and not self.sbert_model:
            logger.warning("Dense retrieval requested, but SBERT model is not available.")

        # --- TF-IDF Retrieval ---
        tfidf_similarities = []
        if self.retrieval_mode in ["tfidf", "hybrid", "dense"] and self.knowledge_agent:
            query_tokens = self._preprocess(expanded_query_text)
            if query_tokens:
                query_tfidf_vector_dict = self._calculate_tfidf(query_tokens)
                # Convert query TF-IDF dict to NumPy array based on current vocabulary for consistent comparison
                current_sorted_vocab = sorted(list(self.vocabulary)) # Get a stable order
                query_tfidf_numpy = np.array([query_tfidf_vector_dict.get(term, 0.0) for term in current_sorted_vocab])


                for doc_details in self.knowledge_agent:
                    doc_id = doc_details['doc_id']
                    doc_tfidf_vector_dict = self.doc_tf_idf_vectors.get(doc_id, {})
                    if doc_tfidf_vector_dict:
                        # Convert doc TF-IDF dict to NumPy array based on the same stable vocabulary
                        doc_tfidf_numpy = np.array([doc_tfidf_vector_dict.get(term, 0.0) for term in current_sorted_vocab])
                        similarity = cosine_sim(query_tfidf_numpy, doc_tfidf_numpy)
                        if similarity >= self.similarity_threshold: # Can have a separate threshold for TF-IDF
                            tfidf_similarities.append((similarity, doc_details))
            else:
                 logger.debug("No query tokens after preprocessing for TF-IDF.")


        # --- Combine Results for Hybrid Mode ---
        final_results_map = {}
        if self.retrieval_mode == "hybrid":
            # Simple weighted combination (example weights)
            dense_weight = 0.7
            tfidf_weight = 0.3
            for sim_score, doc in dense_similarities:
                final_results_map[doc['doc_id']] = final_results_map.get(doc['doc_id'], {'score': 0, 'doc': doc})
                final_results_map[doc['doc_id']]['score'] += sim_score * dense_weight
            for sim_score, doc in tfidf_similarities:
                final_results_map[doc['doc_id']] = final_results_map.get(doc['doc_id'], {'score': 0, 'doc': doc})
                final_results_map[doc['doc_id']]['score'] += sim_score * tfidf_weight
            
            combined_results = [(item['score'], item['doc']) for item in final_results_map.values()]

        elif self.retrieval_mode == "dense":
            combined_results = dense_similarities
        elif self.retrieval_mode == "tfidf":
            combined_results = tfidf_similarities
        else:
            logger.error(f"Unknown retrieval mode: {self.retrieval_mode}")
            return []
        
        # Sort and get top-k
        # Ensure unique documents if combining, then sort
        unique_docs = {}
        for score, doc in combined_results:
            if doc['doc_id'] not in unique_docs or score > unique_docs[doc['doc_id']][0]:
                 unique_docs[doc['doc_id']] = (score, doc)
        
        results_before_k = sorted(list(unique_docs.values()), key=lambda x: x[0], reverse=True)
        final_k_results = results_before_k[:k]

        # Governance and Bias Auditing
        if self.governor:
            self.governor.audit_retrieval(
                query=query, # original query
                results=final_k_results,
                context={
                    'timestamp': time.time(),
                    'user': self.shared_memory.get('current_user'),
                    'module': 'knowledge_retrieval',
                    'retrieval_mode': self.retrieval_mode,
                    'expanded_query_used': expanded_query_text != query
                }
            )
        if self.bias_detection_enabled:
            final_k_results = self._apply_bias_analysis(final_k_results)

        serializable_results = [(float(score), doc) for score, doc in final_k_results]
        
        # Thread-safe cache update
        with self.cache_lock:
            if len(self.knowledge_cache) > self.cache_size:
                self.knowledge_cache.popitem()
            self.knowledge_cache.set(cache_key, serializable_results)
            
        self.shared_memory.set("retrieved_knowledge", [doc['text'] for _, doc in final_k_results])
        self.performance_metrics['retrieval_times'].append(time.time() - start_time)
        
        safe_query = query.encode('ascii', 'replace').decode('ascii')
        logger.info(f"Retrieved {len(final_k_results)} documents for query '{safe_query}' using {self.retrieval_mode} mode.")
        return final_k_results
    
    #def retrieve_batch(self, queries: List[str], k: int = 2) -> List[Tuple[float, Dict[str, Any]]]:
    #    """
    #    Retrieves relevant documents for a batch of queries efficiently.
    #    Fixed implementation using existing attributes and methods.#

    #    Args:
    #        queries (List[str]): A list of query strings.
    #        k (int): The number of documents to retrieve for each query.

    #    Returns:
    #        A list where each element is a list of retrieved documents for the corresponding query.
    #    """
    #    if not queries or not self.knowledge_agent:
    #        return [[] for _ in queries]
            
        #try:
            # Process in chunks to avoid resource exhaustion
            #CHUNK_SIZE = 500  # Optimal for memory/CPU balance
            #batch_results = [None] * len(queries)
            
            #with tqdm(total=len(queries), desc="Batch Retrieval") as pbar:
            #    for start_idx in range(0, len(queries), CHUNK_SIZE):
            #        end_idx = min(start_idx + CHUNK_SIZE, len(queries))
            #        chunk = queries[start_idx:end_idx]
                    
            #        with ThreadPoolExecutor(max_workers=8) as executor:
            #            future_to_index = {
            #                executor.submit(self.retrieve, query, k): idx 
            #                for idx, query in enumerate(chunk, start=start_idx)
            #            }
                        
            #            for future in as_completed(future_to_index):
            #                idx = future_to_index[future]
            #                try:
            #                    batch_results[idx] = future.result()
            #                except Exception as e:
            #                    logger.error(f"Retrieval failed for query index {idx}: {e}")
            #                    batch_results[idx] = []
            #                finally:
            #                    pbar.update(1)

            # return batch_results

        #except Exception as e:
        #    logger.error(f"Batch retrieval failed: {str(e)}")
        #    return [[] for _ in queries]

    def retrieve_batch(self, original_texts: List[str], k: int = 2) -> List[Tuple[float, Dict[str, Any]]]:

        if not original_texts or not self.knowledge_agent:
            return [[] for _ in original_texts]

        CHUNK_SIZE = 1000
        batch_results = [None] * len(original_texts)
        
        for start_idx in range(0, len(original_texts), CHUNK_SIZE):
            end_idx = min(start_idx + CHUNK_SIZE, len(original_texts))
            chunk = original_texts[start_idx:end_idx]
            
            # Process chunk with separate thread pool
            with ThreadPoolExecutor(max_workers=4) as executor:
                future_to_index = {
                    executor.submit(self.retrieve, query, k=2): idx 
                    for idx, query in enumerate(chunk, start=start_idx)
                }
                
                for future in as_completed(future_to_index):
                    idx = future_to_index[future]
                    batch_results[idx] = future.result()

        return batch_results

    def _expand_with_ontology(self, terms):
        """Hierarchical expansion with property inheritance with cycle detection"""
        expanded = set()
        visited_types = set()  # Track visited types to prevent cycles
        
        for term in terms:
            # Add the term itself
            expanded.add(term)
            
            if term in self.ontology:
                # Get all ancestor types with cycle detection
                current_type = self.ontology[term]['type']
                while current_type and current_type not in visited_types:
                    visited_types.add(current_type)
                    expanded.add(current_type)
                    current_type = self.ontology.get(current_type, {}).get('type')
                
                # Add related entities
                for pred, obj in self.ontology[term]['relations']:
                    expanded.add(pred)
                    expanded.add(obj)
                    # Add type of related object if available
                    obj_type = self.ontology.get(obj, {}).get('type')
                    if obj_type:
                        expanded.add(obj_type)

            # Add ontology manager expansions
            expanded.update(self.ontology_manager.expand_query([term]))
            types = self.ontology_manager.get_types(term)
            expanded.update(types)

        return " ".join(expanded)

    def _dict_to_numpy(self, vector):
        """Convert TF-IDF dict to numpy array aligned with vocabulary"""
        return np.array([vector.get(term, 0) for term in sorted(self.vocabulary)])

    def _preprocess(self, text: str) -> List[str]:
        """Text normalization and tokenization."""
        cache_key = text.lower()
        if cache_key in self.token_cache:
            return self.token_cache[cache_key]
        
        tokens = [token for token in text.split() if token not in self.stopwords]
        self.token_cache[cache_key] = tokens
        return tokens

    def _calculate_tfidf(self, tokens: List[str]) -> Dict[str, float]:
        """Calculate TF-IDF vector (as a dict) for document or query."""
        tf = defaultdict(float)
        if not tokens: # Handle empty token list
            return {}
        total_terms = len(tokens)
        
        for token in tokens:
            tf[token] += 1.0
    
        vector = {}
        for token, freq in tf.items():
            term_tf = freq / total_terms # Normalized TF
            
            if token in self.document_frequency and self.document_frequency[token] > 0:
                idf = math.log((self.total_documents + 1) / (self.document_frequency[token] + 1)) + 1
            else:
                idf = math.log((self.total_documents + 1) / 1) + 1 # Smoothed IDF for unseen query terms
            
            vector[token] = term_tf * idf
        return vector

    def _cosine_similarity(self, vec_a, vec_b):
        """Convert sparse dicts to dense numpy arrays and compute cosine similarity"""
        all_terms = set(vec_a.keys()) | set(vec_b.keys())
        a = np.array([vec_a.get(term, 0) for term in all_terms])
        b = np.array([vec_b.get(term, 0) for term in all_terms])
        return cosine_sim(a, b)

    # ==== Long-Term memory information ====
    def update_memory(self, key: str, value: Any, metadata: dict = None, context: dict = None, ttl: int = None):
        """
        Updates the agent's memory with a key-value pair, enriching metadata with contextual information.
    
        Args:
            key (str): The key to store the value under.
            value (Any): The value to store.
            metadata (dict, optional): Additional metadata about the value. Defaults to None.
            context (dict, optional): Contextual information to enrich the metadata. Defaults to None.
            ttl (int, optional): Time-to-live for the memory entry in seconds. Defaults to None (no expiration).
        """
        timestamp = time.time()
        relevance_score = self._calculate_relevance(value, context) if context else 1.0
    
        enriched_metadata = {
            "timestamp": timestamp,
            "source": context.get("source") if context else "unknown",
            "relevance": relevance_score,
            "dependencies": context.get("dependencies") if context else [],
            "expiry_time": timestamp + ttl if ttl else None
        }
    
        if metadata:
            enriched_metadata.update(metadata)
    
        self.shared_memory.set(
            key,
            {
                "value": value,
                "metadata": enriched_metadata,
                "context": context,
                "relevance": relevance_score,
            },
            ttl=ttl
        )

        return {
            "key": key,
            "value": value,
            "context": context,
            "metadata": enriched_metadata,
            "relevance": relevance_score,
            "ttl": ttl
        }

    def _calculate_relevance(self, value, context):
        """
        Computes a relevance score between a memory value and its context.
    
        Args:
            value (str): The text or knowledge snippet being stored.
            context (dict): Context in which the value was generated (includes query, user, etc.).
    
        Returns:
            float: A relevance score between 0.0 and 1.0.
        """
        try:
            # Step 1: Preprocess both value and context (if textual)
            value_tokens = self._preprocess(value) if isinstance(value, str) else []
            context_query = context.get("query") or context.get("source", "")
            context_tokens = self._preprocess(context_query) if context_query else []
    
            # Step 2: Generate TF-IDF vectors for both
            value_vector = self._calculate_tfidf(value_tokens)
            context_vector = self._calculate_tfidf(context_tokens)
    
            # Step 3: Compute cosine similarity
            score = self._cosine_similarity(value_vector, context_vector)
    
            # Step 4: Bias-aware penalty (optional)
            if self.bias_detection_enabled and self.governor:
                bias_conf = self.governor._detect_unethical_content(value)
                if bias_conf > self.bias_threshold:
                    penalty = min(bias_conf * 0.5, 0.3)  # Clamp penalty
                    score *= (1.0 - penalty)
    
            return round(min(score, 1.0), 4)
    
        except Exception as e:
            logger.warning(f"Failed to calculate relevance: {e}")
            return 0.0

    def recall_memory(self, key: str = None, filters: dict = None, sort_by: str = None, top_k: int = None) -> list:
        """
        Recalls values from memory, with optional filtering, sorting, and top-k retrieval.
    
        Args:
            key (str, optional): The key to retrieve. If None, retrieves from all memory. Defaults to None.
            filters (dict, optional): Filters to apply to the metadata. Defaults to None.
            sort_by (str, optional): The metadata field to sort by ('relevance', 'timestamp'). Defaults to None.
            top_k (int, optional): Return only the top-k results. Defaults to None (return all).
    
        Returns:
            list: A list of memory entries matching the criteria.
        """
        results = []
    
        if key:
            entry = self.shared_memory.get(key)
            if entry:
                results.append((entry["value"], entry["metadata"]))
        else:
            results = [(entry["value"], entry["metadata"]) for entry in self.shared_memory.values()]
    
        # Apply filters
        if filters:
            results = [(value, metadata) for value, metadata in results if self._apply_filters(metadata, filters)]
    
        # Sort results
        if sort_by:
            if sort_by == "relevance":
                results.sort(key=lambda item: item[1].get("relevance", 0), reverse=True)  # Default relevance to 0
            elif sort_by == "timestamp":
                results.sort(key=lambda item: item[1].get("timestamp", 0), reverse=True)
            else:
                logger.warning(f"Unknown sort_by field: {sort_by}")
    
        # Return top-k
        if top_k:
            results = results[:top_k]
    
        return results
    
    def _apply_filters(self, metadata: dict, filters: dict) -> bool:
        """
        Helper function to apply filters to metadata.
        """
        for key, value in filters.items():
            if key not in metadata or metadata[key] != value:
                return False
        return True
    # ============================================

    def _apply_bias_analysis(self, results: list) -> list:
        """Analyze and annotate results with bias information"""
        analyzed_results = []
        for score, doc in results:
            bias_meta = doc.get('metadata', {}).get('bias', {})
            
            # Add bias summary to document
            doc['bias_summary'] = {
                "is_biased": bias_meta.get("detected", False),
                "confidence": bias_meta.get("confidence", 0.0),
                "dominant_category": max(
                    bias_meta.get("categories", {}).items(), 
                    key=lambda x: x[1], 
                    default=("none", 0)
                )[0]
            }
            analyzed_results.append((score, doc))
        return analyzed_results

    def contextual_search(self, query):
        """
        Search with consideration of recent context, using weighted term emphasis.
        
        Args:
            query: Current search query.
            context_window: Number of past interactions to consider.
            decay_factor: Weight reduction for older queries (0.5 = older terms matter half as much).
        
        Returns:
            List of (score, document) tuples, prioritized by contextual relevance.
        """
        # Retrieve or initialize context storage
        context_key = f"context:{self.name}"
        context = self.shared_memory.get(context_key)
        if context is None:
            context = deque(maxlen=self.context_window)

        # Preprocess current query and extract key terms
        current_tokens = self._preprocess(query)
        current_terms = self._extract_significant_terms(current_tokens, top_n=10)

        # Update context with current terms and their freshness
        context.append({
            "timestamp": time.time(),
            "terms": current_terms,
            "raw_query": query
        })
        self.shared_memory.set(context_key, context)

        # Generate augmented query with temporal weighting
        augmented_terms = []
        max_weight = sum(self.decay_factor ** i for i in range(len(context)))
        
        for idx, entry in enumerate(reversed(context)):
            weight = self.decay_factor ** idx  # Most recent gets highest weight
            augmented_terms.extend([term for term in entry["terms"] for _ in range(int(weight * 10))])

        # Combine with current terms (full weight)
        augmented_terms.extend(current_terms * 10)  # 10x multiplier for emphasis
        augmented_query = " ".join(augmented_terms)

        logger.debug(f"Augmented query: {augmented_query}")

        # Bias-aware context augmentation
        if self.bias_detection_enabled:
            augmented_query += " unbiased factual"

        return self.retrieve(augmented_query)

    def _extract_significant_terms(self, tokens, top_n=5):
        """Identify top TF-IDF terms from a token list."""
        tfidf = self._calculate_tfidf(tokens)
        sorted_terms = sorted(tfidf.items(), key=lambda x: x[1], reverse=True)
        return [term for term, score in sorted_terms[:top_n]]

    def get_references_for_concepts(self, concepts: list, k: int = 3) -> list:
        """
        Retrieve textual references related to a list of concept strings using semantic TF-IDF search.
        """
        references = []
        for concept in concepts:
            if not isinstance(concept, str) or not concept.strip():
                continue
            results = self.retrieve(concept, k=k)
            for score, doc in results:
                references.append(doc.get("text", ""))

        return references

    def broadcast_knowledge(self, context=None):
        """
        Stores and publishes the last 5 knowledge items into shared memory,
        with enriched metadata and broadcast to subscribers.
        """
        broadcast_keys = []
    
        for idx, doc in enumerate(self.knowledge_agent[-5:]):
            key = f"{self.knowledge_tag}:{idx}"
            text = doc["text"]
            metadata = doc.get("metadata", {})
    
            enriched = {
                "value": text,
                "metadata": {
                    "timestamp": time.time(),
                    "tags": self.knowledge_tag or [],
                    "source": self.source,
                    "context": context,
                    **metadata
                }
            }
    
            self.shared_memory.set(key, enriched, ttl=ttl)
    
            self.shared_memory.publish("broadcast_channel", {
                "key": key,
                "value": text,
                "metadata": enriched["metadata"]
            })
    
            broadcast_keys.append(key)
    
        return broadcast_keys

    def discover_rules(self) -> list:
        """Discover and retrieve governance-approved rules"""
        # Try to get rules from Governor first
        if self.governor:
            try:
                raw_rules = self.governor.get_approved_rules()
                logger.info(f"Discovered {len(raw_rules)} rules via Governor")
                return self._process_rules(raw_rules)
            except Exception as e:
                logger.error(f"Governor rule discovery failed: {e}")
        
        # Fallback to direct rule engine access
        try:
            rules = self.governor.rule_engine.get_rules_by_category("governance_approved")
            logger.info(f"Discovered {len(rules)} rules directly from RuleEngine")
            return self._process_rules(rules)
        except Exception as e:
            logger.error(f"Direct rule discovery failed: {e}")
            return []
    
    def _process_rules(self, raw_rules: list) -> list:
        """Process and filter raw rules"""
        min_confidence = 0.7
        processed_rules = []
        
        for rule in raw_rules:
            if not isinstance(rule, dict):
                continue
                
            confidence = rule.get("confidence", 1.0)
            if confidence < min_confidence:
                continue
                
            # Add metadata if available
            rule_id = rule.get("id")
            if rule_id:
                try:
                    memory_matches = self.recall_memory(
                        filters={"type": "system_rule", "id": rule_id}
                    )
                    if memory_matches:
                        _, metadata = memory_matches[0]
                        rule["metadata"] = metadata
                except Exception as e:
                    logger.warning(f"Failed to annotate rule {rule_id}: {e}")
            
            processed_rules.append(rule)
        
        return processed_rules
    
    def respond_to_query(self, query):
        """
        1. Lookup cache
        2. Retrieve memory
        3. Detect bias
        4. Apply rules
        5. Take action or return result

        Pipeline: query -> retrieve -> evaluate -> act
        """
        context = {
            'user': self.shared_memory.get('current_user'),
            'module': 'query_processing',
            'timestamp': time.time()
        }
        
        # --- RETRIEVE PHASE ---
        retrieved = self.retrieve(query)
        if not retrieved:
            return {"response": "No relevant information found", "confidence": 0.0}
        
        # --- EVALUATE PHASE ---
        # 1. Governance audit
        if self.governor:
            audit_report = self.governor.audit_retrieval(
                query=query,
                results=retrieved,
                context=context
            )
            violations = audit_report.get('violations', [])
            if any(v.get('score', 0) > self.governor.violation_thresholds.critical for v in violations):
                return {"response": "Response blocked due to policy violations", "violations": violations}
        
        # 2. Bias evaluation
        bias_analysis = []
        if self.bias_detection_enabled:
            for score, doc in retrieved:
                bias_meta = doc.get('metadata', {}).get('bias', {})
                if bias_meta.get('detected', False):
                    bias_analysis.append({
                        'source': doc.get('metadata', {}).get('source'),
                        'categories': bias_meta.get('categories', {})
                    })
        
        # 3. Freshness check
        current_time = time.time()
        fresh_results = []
        for score, doc in retrieved:
            doc_time = doc.get('metadata', {}).get('timestamp', 0)
            if current_time - doc_time < self.governor.freshness_threshold * 3600:
                fresh_results.append((score, doc))
        
        # Use fresh results if available, fallback to all results
        final_results = fresh_results if fresh_results else retrieved
        
        # --- ACT PHASE ---
        # 1. Format response
        response_text = "\n\n".join([doc['text'] for _, doc in final_results[:3]])
        
        # 2. Check if action is required
        action_required = self._detect_action_trigger(query, final_results)
        if action_required:
            action_result = self.perform_action.execute(
                action_required,
                context=context,
                references=[doc['text'] for _, doc in final_results]
            )
            return {
                "response": response_text,
                "action": action_required,
                "action_result": action_result,
                "bias_analysis": bias_analysis
            }
        
        # 3. Default knowledge response
        return {
            "response": response_text,
            "sources": [doc.get('metadata', {}).get('source', 'unknown') for _, doc in final_results],
            "bias_analysis": bias_analysis
        }
    
    def _detect_action_trigger(self, query, results):
        """
        Detect if the query requires performing an action based on:
        - Explicit action verbs in the query
        - Rule-based inference from retrieved content
        """
        ACTION_TRIGGERS = [
            "update", "modify", "create", "delete", "execute", 
            "perform", "run", "notify", "alert", "change"
        ]
        
        # Check query for action verbs
        if any(trigger in query.lower() for trigger in ACTION_TRIGGERS):
            return "user_requested_action"
        
        # Check retrieved content for action patterns
        for _, doc in results:
            content = doc.get('text', '').lower()
            if "required action:" in content or "next steps:" in content:
                return "content_suggested_action"

        # Apply rule engine to determine actions via governor
        if self.governor and self.governor.rule_engine:
            knowledge_graph = {
                "query": query,
                "results": [doc['text'] for _, doc in results]
            }
            inferences = self.governor.rule_engine.apply(knowledge_graph)
            if inferences.get("action_required"):
                return inferences["action_required"]
        
        return None

    def _calculate_relevance(self, value: Any, context: Optional[dict]) -> float:
        if not context:
            return 0.5 # Neutral relevance if no context

        value_text = str(value)
        context_query = context.get("query", "") or context.get("source", "")
        if not value_text.strip() or not context_query.strip():
            return 0.25 # Low relevance if one part is empty

        # Primary: Dense Embedding Similarity
        if self.sbert_model:
            try:
                emb_value = self.sbert_model.encode(value_text, show_progress_bar=False)
                emb_context = self.sbert_model.encode(context_query, show_progress_bar=False)
                score = cosine_sim(emb_value, emb_context)
            except Exception as e:
                logger.warning(f"SBERT relevance calculation failed: {e}. Falling back.")
                score = self._fallback_relevance_tfidf(value_text, context_query)
        else:
            # Fallback: TF-IDF Similarity
            score = self._fallback_relevance_tfidf(value_text, context_query)

        # Bias-aware penalty (if applicable)
        if self.bias_detection_enabled and self.governor:
            bias_conf = self.governor._detect_unethical_content(value_text) # Pass text
            if bias_conf > self.bias_threshold:
                penalty = min(bias_conf * 0.5, 0.3)
                score *= (1.0 - penalty)
        
        return round(max(0.0, min(score, 1.0)), 4) # Ensure score is between 0 and 1

    def _fallback_relevance_tfidf(self, text1: str, text2: str) -> float:
        tokens1 = self._preprocess(text1)
        tokens2 = self._preprocess(text2)
        if not tokens1 or not tokens2:
            return 0.0

        vec1_dict = self._calculate_tfidf(tokens1)
        vec2_dict = self._calculate_tfidf(tokens2)

        # Convert dicts to numpy arrays aligned by shared vocabulary for cosine_sim
        all_terms = sorted(list(set(vec1_dict.keys()) | set(vec2_dict.keys())))
        if not all_terms: return 0.0

        np_vec1 = np.array([vec1_dict.get(term, 0.0) for term in all_terms])
        np_vec2 = np.array([vec2_dict.get(term, 0.0) for term in all_terms])
        
        return cosine_sim(np_vec1, np_vec2)
    
    def _validate_with_safety(self, action_params: Dict, context: Optional[Dict] = None) -> bool:
        """
        Executes registered safety validation if available
        
        Returns:
            True if approved, False if rejected or no callback registered
        """
        if not self.safety_check_callback:
            return True  # Default approve if no callback
        
        try:
            result = self.safety_check_callback(action_params, context or {})
            return result.get('approved', False)
        except Exception as e:
            logger.error(f"Safety validation failed: {str(e)}")
            return False  # Reject on failure
    
    def register_safety_check(self, safety_check_callback: callable):
        """
        Registers a safety validation callback that can be used to validate 
        knowledge operations. This provides a flexible hook for security validation.
        
        Args:
            safety_check_callback: Function with signature:
                (action_params: Dict, action_context: Optional[Dict]) -> Dict
                Must return a dict with at least 'approved' (bool) key
        """
        if not callable(safety_check_callback):
            logger.error("Safety check callback must be callable")
            raise TypeError("Safety callback must be callable")
        
        # Validate callback signature
        try:
            test_result = safety_check_callback(
                {"action": "test_validation"},
                {"context": "signature_verification"}
            )
            if not isinstance(test_result, dict) or 'approved' not in test_result:
                raise ValueError("Callback must return dict with 'approved' key")
        except Exception as e:
            logger.error(f"Safety callback validation failed: {str(e)}")
            raise
        
        self.safety_check_callback = safety_check_callback
        logger.info("Safety validation callback registered successfully")
    
    def attach_learning(self, learning_agent):
        """Connect LearningAgent to KnowledgeAgent"""
        self.learning_agent = learning_agent
        logger.info("Learning agent attached to KnowledgeAgent")

if __name__ == "__main__":
    print("\n=== Running Knopwledge Agent ===\n")
    printer.status("Init", "Knopwledge Agent initialized", "success")
    from src.agents.collaborative.shared_memory import SharedMemory
    from src.agents.agent_factory import AgentFactory
    shared_memory = SharedMemory()
    agent_factory = AgentFactory()

    agent = KnowledgeAgent(shared_memory=shared_memory,agent_factory=agent_factory)
    print(agent)

    print("\n* * * * * Phase 2 * * * * *\n")

    result = agent.load_from_directory()
    printer.pretty("LOADER", f"Loaded {result} documents", "success" if result else "error")

    print("\n* * * * * Phase 3 * * * * *\n")
    query="AI ethics principles for autonomous systems"
    retriever = agent.retrieve(query=query, k=5)
    batch = agent.retrieve_batch(k=2, original_texts=query)

    printer.status("RETRIEVE", retriever, "success" if retriever else "error")
    printer.pretty("BATCH", batch, "success" if batch else "error")

    print("\n* * * * * Phase 4 - Memory * * * * *\n")
    key = "ethics_guideline_v1"
    value = {
        "statement": "AI should avoid reinforcing historical biases.",
        "category": "ethics",
        "confidence": 0.92
    }
    metadata = None
    context = {
        "query": "What ethical principles should AI follow?",
        "source": "ethics_policy.txt",
        "dependencies": ["fairness", "transparency"]
    }
    ttl = 600
    filters = None
    sort_by = None
    top_k = None

    update = agent.update_memory(key=key, value=value, metadata=metadata, context=context, ttl=ttl)
    recalling = agent.recall_memory(key=key, filters=filters, sort_by=sort_by, top_k=top_k)

    printer.pretty("MEMORY1", update, "success" if update else "error")
    printer.pretty("MEMORY2", recalling, "success" if recalling else "error")

    print("\n* * * * * Phase 5 - Search * * * * *\n")
    printer.status("SEARCH", agent.contextual_search(query=query), "success")

    print("\n* * * * * Phase 6 - Broadcast * * * * *\n")
    broadcast = agent.broadcast_knowledge()
    printer.pretty("BROADCAST", broadcast, "success" if broadcast else "error")

    response = agent.respond_to_query(query=query)

    printer.pretty("RESPONSE", response, "success" if response else "error")
    print("\n=== Successfully ran the Knopwledge Agent ===\n")
