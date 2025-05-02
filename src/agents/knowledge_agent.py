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
import numpy as np
import pandas as pd

from collections import defaultdict, deque
from heapq import nlargest
from typing import Any

from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder

from src.agents.base_agent import BaseAgent
from src.agents.knowledge.rule_engine import RuleEngine
from logs.logger import get_logger

logger = get_logger(__name__)

def cosine_sim(v1, v2):
    v1, v2 = np.array(v1), np.array(v2)
    dot = np.dot(v1, v2)
    return dot / (np.linalg.norm(v1) * np.linalg.norm(v2))


class KnowledgeAgent(BaseAgent):
    def __init__(self, shared_memory, agent_factory, config=None, language_agent=None, knowledge_agent_dir=None, persist_file: str = None, args=(), kwargs={}):
        super().__init__(
            shared_memory=shared_memory,
            agent_factory=agent_factory,
            config=config
        )
        self.knowledge_agent = []
        self.shared_memory = shared_memory
        self.agent_factory = agent_factory
        self.cache = {}
        self.cache_size = 1000
        self.vocabulary = set()
        self.document_frequency = defaultdict(int)
        self.total_documents = 0
        self.persist_file = persist_file
        self.memory = defaultdict(dict)
        self.doc_vectors = {}
        self.stopwords = set([
            'a', 'an', 'the', 'and', 'or', 'in', 'on', 'at', 'to', 'of',
            'for', 'with', 'as', 'by', 'that', 'this', 'it', 'is', 'are',
            'was', 'were', 'be', 'been', 'being', 'from', 'have', 'has',
            'had', 'do', 'does', 'did', 'will', 'would', 'should', 'can',
            'could', 'not', 'but', 'he', 'she', 'his', 'her', 'they', 'them',
            'their', 'you', 'your', 'we', 'our', 'us', 'i', 'me', 'my', 'mine'
            'if', 'then', 'when', 'which', 'what', 'how', 'while', 'after',
            'before', 'who', 'where', 'why', 'so', 'because', 'than', 'just', 'also'
        ])
        self.rule_engine = RuleEngine()
        self.ontology = defaultdict(lambda: {'type': None, 'relations': set()})

        # Semantic fallback
        self.language_agent = language_agent
        self.embedding_fallback = self.language_agent.embedder if self.language_agent else None

        if knowledge_agent_dir:
            self.load_from_directory(knowledge_agent_dir)

    def _safe_create_language_agent(self):
        try:
            self.language_agent = self.agent_factory.create('language', config={})
        except Exception as e:
            self.logger.warning(f"[KnowledgeAgent] Failed to create language agent: {e}")
            self.language_agent = None
        
    def load_from_directory(self, directory_path: str):
        """Loads all .txt and .json files in the directory as knowledge documents."""

        if not os.path.isdir(directory_path):
            self.logger.error(f"Invalid directory: {directory_path}")
            return

        for fname in os.listdir(directory_path):
            fpath = os.path.join(directory_path, fname)
            try:
                if fname.endswith(".txt"):
                    with open(fpath, "r", encoding="utf-8") as f:
                        text = f.read().strip()
                        self.add_document(text, metadata={
                            "source": fname,
                            "timestamp": time.time(),
                            "checksum": hashlib.sha1(text.encode("utf-8")).hexdigest()})

                elif fname.endswith(".json"):
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        for entry in data:
                            text = entry.get("text") or str(entry)
                            self.add_document(text, metadata={"source": fname})
            except Exception as e:
                self.logger.error(f"Failed to load {fname}: {str(e)}", exc_info=True)

    def load_knowledge_db(self, db_path: str):
        """Integrate structured knowledge from knowledge_db.json"""
        with open(db_path, 'r') as f:
            data = json.load(f)
            for triple in data["knowledge"]:
                self.add_document(triple[0], metadata={"type": "fact", "confidence": triple[1]})
            for rule in data["rules"]:
                self.rule_engine.add_rule(*rule)

    def discover_rules(self, min_support=0.1):
        """Mine association rules from knowledge triples"""
        # Convert knowledge to transactional format
        transactions = []
        for doc in self.knowledge_agent:
            if '|' in doc['text']:  # Assume "subject|predicate|object" format
                subj, pred, obj = doc['text'].split('|')
                transactions.append([f"{subj}→{pred}", f"{pred}→{obj}"])

        # Use Apriori algorithm for rule mining
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df = pd.DataFrame(te_ary, columns=te.columns_)
        freq_items = apriori(df, min_support=min_support, use_colnames=True)
        
        # Convert to logical rules
        new_rules = []
        for _, row in freq_items.iterrows():
            itemset = list(row['itemsets'])
            if len(itemset) == 2:
                rule_name = f"MLDiscoveredRule:{'→'.join(itemset)}"
                new_rules.append((rule_name, lambda kg: self._apply_ml_rule(kg, itemset), 0.65))
        
        # Update RuleEngine and save to knowledge_db.json
        self.rule_engine.rules.extend(new_rules)
        self._update_knowledge_db(new_rules)

    def _apply_ml_rule(self, knowledge_graph, pattern):
        """Auto-generated rule application logic"""
        inferred = {}
        for key, value in knowledge_graph.items():
            if pattern[0] in value and pattern[1] in value:
                inferred[f"Inferred:{pattern[0]}→{pattern[1]}"] = 0.65
        return inferred

    def retrieve(self, query, k=5, use_ontology=True, similarity_threshold=0.2):
        cache_key = hashlib.sha256(query.encode()).hexdigest()
        start_time = time.time()
        cache_hit = cache_key in self.cache
        self.performance_metrics['retrieval_times'].append(time.time() - start_time)
        self.performance_metrics['cache_hits'].append(1 if cache_hit else 0)
        
        self.shared_memory.set("retrieved_knowledge", [doc['text'] for _, doc in results])

        if cache_key in self.cache:
            return self.cache[cache_key]

        # Preprocess and validate
        query_tokens = self._preprocess(query)
        if not query_tokens or not self.knowledge_agent:
            return []

        # Calculate query vector
        query_vector = self._dict_to_numpy(
            self._calculate_tfidf(
                query_tokens, is_query=True
        ))

        # Calculate document vectors and similarities
        similarities = []
        for doc in self.knowledge_agent:
            doc_vec = self.doc_vectors[doc['doc_id']]  # Precomputed    
            similarity = cosine_sim(query_vector, doc_vec)
            if similarity < similarity_threshold and self.embedding_fallback:
                try:
                    emb_query = self.embedding_fallback.encode(query)
                    emb_doc = self.embedding_fallback.encode(doc['text'])
                    similarity = cosine_sim(emb_query, emb_doc)
                except Exception:
                    similarity = 0.0

            if similarity >= similarity_threshold:
                similarities.append((similarity, doc))

        results = nlargest(k, similarities, key=lambda x: x[0])
        if len(self.cache) > self.cache_size:
            self.cache.popitem()
        self.cache[cache_key] = results
    
        if use_ontology:
            query_terms = self._preprocess(query)
            expanded_query = self._expand_with_ontology(query_terms)
            return super().retrieve(expanded_query, k=k)
        
        # Apply inference rules to results
        knowledge_graph = {doc['doc_id']: doc['text'] for _, doc in results}
        inferred = self.rule_engine.apply(knowledge_graph)
        
        # Inferred knowledge to results
        for fact, conf in inferred.items():
            results.append((conf, {'text': fact, 'metadata': {'inferred': True}}))
        
        return nlargest(k, results, key=lambda x: x[0])

    def _expand_with_ontology(self, terms):
        """Hierarchical expansion with property inheritance"""
        expanded = set()
        for term in terms:
            if term in self.ontology:
                # Get all ancestor types
                current_type = self.ontology[term]['type']
                while current_type:
                    expanded.add(current_type)
                    current_type = self.ontology[current_type]['type']
                
                # Add related entities
                for pred, obj in self.ontology[term]['relations']:
                    expanded.update([pred, obj, self.ontology[obj].get('type')])
        
        return " ".join(expanded) + " " + " ".join(terms)

    def add_document(self, text, doc_id=None, metadata=None):
        if isinstance(text, tuple) and len(text) == 3:
            self.add_to_ontology(*text)
        if not isinstance(text, str) or len(text.strip()) < 3:
            raise ValueError("Document text must be non-empty string")
        if doc_id and doc_id in self.doc_vectors:
            raise KeyError(f"Document ID {doc_id} already exists")
        
        # Auto-generate doc_id if None
        doc_id = doc_id or hash(text) 

        """Store documents with preprocessing and vocabulary update"""
        tokens = self._preprocess(text)
        self.knowledge_agent.append({'doc_id': doc_id, 'text': text, 'tokens': tokens, 'metadata': metadata})
        tfidf_vector = self._calculate_tfidf(tokens)
        self.doc_vectors[doc_id] = self._dict_to_numpy(tfidf_vector)  # Store as numpy array
        
        # Update vocabulary and document frequency
        unique_tokens = set(tokens)
        for token in unique_tokens:
            self.document_frequency[token] += 1
        self.vocabulary.update(unique_tokens)
        self.total_documents += 1

    def add_to_ontology(self, subject: str, predicate: str, obj: str):
        self.ontology[subject]['relations'].add((predicate, obj))
        if predicate in ('is_a', 'type', 'class'):
            self.ontology[subject]['type'] = obj

    def _dict_to_numpy(self, vector):
        """Convert TF-IDF dict to numpy array aligned with vocabulary"""
        return np.array([vector.get(term, 0) for term in self.vocabulary])

    def _preprocess(self, text):
        """Text normalization and tokenization"""
        # Lowercase and remove special characters
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
        # Tokenize and remove stopwords
        tokens = [token for token in text.split() if token not in self.stopwords]
        return tokens

    def _calculate_tfidf(self, tokens, is_query=False):
        """Calculate TF-IDF vector for document or query"""
        tf = defaultdict(float)
        total_terms = len(tokens)
        
        # Term Frequency (TF)
        for token in tokens:
            tf[token] += 1/total_terms if is_query else 1  # Query normalization
    
        # Semantic weighting using word embeddings
        if self.embedding_fallback:
            semantic_weights = self._get_semantic_weights(tokens)
            for token, weight in semantic_weights.items():
                tf[token] *= (1 + weight)  # Amplify TF-IDF by semantic relevance
    
        # Inverse Document Frequency (IDF)
        vector = {}
        for token in set(tokens):
            if token in self.vocabulary:
                idf = math.log((self.total_documents + 1) / (self.document_frequency[token] + 1)) + 1
                vector[token] = tf[token] * idf
        return vector

    def _get_semantic_weights(self, tokens):
        """Use sentence transformers for contextual weighting"""
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        embeddings = model.encode(tokens)
        doc_embedding = np.mean(embeddings, axis=0)
        
        similarities = {}
        for token, emb in zip(tokens, embeddings):
            similarities[token] = cosine_sim(emb, doc_embedding)
        
        return similarities

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
        enriched_metadata = {
            "timestamp": timestamp,
            "source": context.get("source") if context else "unknown",
            "relevance": self._calculate_relevance(value, context) if context else 1.0,
            "dependencies": context.get("dependencies") if context else [],
            "expiry_time": timestamp + ttl if ttl else None
        }
        
        if metadata:
            enriched_metadata.update(metadata)

        self.memory[key] = {
            "value": value,
            "metadata": enriched_metadata
        }
    
    def _calculate_relevance(self, value: Any, context: dict) -> float:
        """
        Computes semantic relevance between the value and the context using cosine similarity.
        Falls back to embedding similarity if available.
        """
        try:
            value_text = str(value)
            context_text = json.dumps(context, ensure_ascii=False) if isinstance(context, dict) else str(context)
    
            # Preprocess
            value_tokens = self._preprocess(value_text)
            context_tokens = self._preprocess(context_text)
    
            # TF-IDF vectorization
            value_vec = self._calculate_tfidf(value_tokens, is_query=True)
            context_vec = self._calculate_tfidf(context_tokens, is_query=True)
    
            tfidf_score = self._cosine_similarity(value_vec, context_vec)
    
            # Try embedding-based similarity if available
            if self.embedding_fallback:
                try:
                    emb_val = self.embedding_fallback.encode(value_text)
                    emb_ctx = self.embedding_fallback.encode(context_text)
                    emb_score = cosine_sim(emb_val, emb_ctx)
                    return max(tfidf_score, emb_score)
                except Exception as e:
                    self.logger.warning(f"[KnowledgeAgent] Embedding relevance failed: {e}")
            
            return tfidf_score
    
        except Exception as e:
            self.logger.warning(f"[KnowledgeAgent] Failed to calculate relevance: {e}")
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
            entry = self.memory.get(key)
            if entry:
                results.append((entry["value"], entry["metadata"]))
        else:
            results = [(entry["value"], entry["metadata"]) for entry in self.memory.values()]
    
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
                self.logger.warning(f"Unknown sort_by field: {sort_by}")
    
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

    def contextual_search(self, query, context_window=3, decay_factor=0.8):
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
        context = self.shared_memory.get(context_key, deque(maxlen=context_window))
        
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
        max_weight = sum(decay_factor ** i for i in range(len(context)))
        
        for idx, entry in enumerate(reversed(context)):
            weight = decay_factor ** idx  # Most recent gets highest weight
            augmented_terms.extend([term for term in entry["terms"] for _ in range(int(weight * 10))])
        
        # Combine with current terms (full weight)
        augmented_terms.extend(current_terms * 10)  # 10x multiplier for emphasis
        augmented_query = " ".join(augmented_terms)
        
        self.logger.debug(f"Augmented query: {augmented_query}")
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

    def broadcast_knowledge(self, tag="knowledge_snippet"):
        for idx, doc in enumerate(self.knowledge_agent[-5:]):
            self.shared_memory.set(f"{tag}:{idx}", doc)
