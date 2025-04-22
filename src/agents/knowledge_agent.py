"""
Knowledge Agent for SLAI (Scalable Learning Autonomous Intelligence)
Implements core RAG functionality with TF-IDF based retrieval from scratch
"""

import os
import re
import json
import math
import time
import hashlib
import numpy as np
from collections import defaultdict, deque
from heapq import nlargest

from src.agents.base_agent import BaseAgent
from src.agents.knowledge.rule_engine import RuleEngine


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
#        self.language_agent = language_agent or self._safe_create_language_agent()
#        self.embedding_fallback = self.language_agent.embedder if self.language_agent else None

        if knowledge_agent_dir:
            self.load_from_directory(knowledge_agent_dir)

#    def _safe_create_language_agent(self):
#        try:
#            self.language_agent = self.agent_factory.create('language', config={})
#        except Exception as e:
#            self.logger.warning(f"[KnowledgeAgent] Failed to create language agent: {e}")
#            self.language_agent = None
        
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
        return results

    def _expand_with_ontology(self, terms):
        expanded = []
        for term in terms:
            if term in self.ontology:
                expanded.append(term)
                for pred, obj in self.ontology[term]['relations']:
                    expanded.extend([pred, obj])
        return " ".join(expanded)

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
        self.knowledge_agent.append({'text': text, 'tokens': tokens, 'metadata': metadata})
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
        
        # Inverse Document Frequency (IDF)
        vector = {}
        for token in set(tokens):
            if token in self.vocabulary:
                idf = math.log((self.total_documents + 1) / (self.document_frequency[token] + 1)) + 1
                vector[token] = tf[token] * idf
        return vector

    def _cosine_similarity(self, vec_a, vec_b):
        """Convert sparse dicts to dense numpy arrays and compute cosine similarity"""
        all_terms = set(vec_a.keys()) | set(vec_b.keys())
        a = np.array([vec_a.get(term, 0) for term in all_terms])
        b = np.array([vec_b.get(term, 0) for term in all_terms])
        return cosine_sim(a, b)


    def update_memory(self, key, value):
        """Store information in long-term memory"""
        self.memory[key] = value

    def recall_memory(self, key):
        """Retrieve information from long-term memory"""
        return self.memory.get(key)


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

# Example usage
#if __name__ == "__main__":
#    agent = KnowledgeAgent()
    
    # Populate knowledge agent
#    documents = [
#        "Reinforcement learning uses rewards to train agents",
#        "Neural networks are computational models inspired by biological brains",
#        "Transformer models use attention mechanisms for sequence processing",
#        "Knowledge graphs represent information as entity-relationship triples"
#    ]
#    for doc in documents:
#        agent.add_document(doc)
    
    # Perform query
#    query = "What models are inspired by biological systems?"
#    results = agent.retrieve(query)
    
#    print(f"Results for query: '{query}'")
#    for score, doc in results:
#        print(f"[Score: {score:.4f}] {doc['text']}")
