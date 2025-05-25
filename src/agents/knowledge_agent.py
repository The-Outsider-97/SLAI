__version__ = "1.8.0"

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
from src.agents.knowledge.knowledge_memory import KnowledgeMemory
from src.agents.knowledge.governor import Governor
from src.agents.knowledge.rule_engine import RuleEngine
from src.agents.knowledge.knowledge_cache import KnowledgeCache
from src.agents.knowledge.perform_action import PerformAction
from logs.logger import get_logger

logger = get_logger("Knowledge Agent")

LOCAL_CONFIG_PATH="src/agents/knowledge/configs/knowledge_config.yaml"

def cosine_sim(v1, v2):
    v1, v2 = np.array(v1), np.array(v2)
    dot = np.dot(v1, v2)
    return dot / (np.linalg.norm(v1) * np.linalg.norm(v2))


class KnowledgeAgent(BaseAgent):
    def __init__(self,
                 agent_factory,
                 config=None,
                 persist_file: str = None, args=(), kwargs={}):
        super().__init__(
            shared_memory=shared_memory,
            agent_factory=agent_factory,
            config=config
        )
        self.knowledge_agent = []
        self.embedding_fallback = None
        self.stop_words = set()
        self.shared_memory = shared_memory
        self.agent_factory = agent_factory
        self.cache = KnowledgeCache() or {}
        self.memory = KnowledgeMemory() or defaultdict(dict)
        self.cache_size = 1000
        self.vocabulary = set()
        self.document_frequency = defaultdict(int)
        self.total_documents = 0
        self.persist_file = persist_file
        self.doc_vectors = {}
        self.rule_engine = RuleEngine()
        self.perform_action = PerformAction()
        self.ontology = defaultdict(lambda: {'type': None, 'relations': set()})
        self.governor = self._initialize_governance()
        # self._initialize_semantic_fallback(language_agent)

    def _initialize_semantic_fallback(self, language_agent):
        """
        Configure embedding fallback and stopwords using the provided language agent.
        """
        self.language_agent = language_agent
    
        # Set up embedding fallback
        if self.language_agent and hasattr(self.language_agent, "embedder"):
            self.embedding_fallback = self.language_agent.embedder
        else:
            try:
                from sentence_transformers import SentenceTransformer
                self.embedding_fallback = SentenceTransformer("all-MiniLM-L6-v2")
                logger.warning("LanguageAgent has no embedder; default model loaded as fallback.")
            except Exception as e:
                self.embedding_fallback = None
                logger.error(f"Failed to load default embedding model: {e}")
    
        # Set up stopwords
        self.initialize_stopwords()

    def initialize_stopwords(self):
        """Retrieve stopwords from the NLPEngine via the initialized language_agent."""
        try:
            if self.language_agent:
                # Access NLPEngine's STOPWORDS from the language_agent
                self.stop_words = self.language_agent.nlp_engine.STOPWORDS
                logger.info("Using NLP Engine's stopwords from language_agent")
            else:
                # Fallback to class-level STOPWORDS if language_agent is unavailable
                from src.agents.language.nlp_engine import NLPEngine
                self.stop_words = NLPEngine.STOPWORDS
                logger.warning("Language agent not provided; using default NLP Engine stopwords")
        except Exception as e:
            self.logger.error(f"Failed to initialize stopwords: {e}")
            self.stop_words = set()

    def _initialize_governance(self):
        """Conditionally create governor based on config"""
        if self.config.get('governor.enabled', True):
            governor = Governor(
                knowledge_agent=self,
                config_file_path=LOCAL_CONFIG_PATH
            )
            logger.info("Governance subsystem initialized")
            return governor
        logger.info("Governance subsystem disabled")
        return None

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
            for triple in data.get("knowledge", []):
                self.add_document(triple[0], metadata={"type": "fact", "confidence": triple[1]})
            for rule in data.get("rules", []):
                try:
                    name, fn_name, weight = rule
                    # Use a placeholder rule function (you may later resolve `fn_name` from a registry)
                    dummy_rule = lambda kg: {}
                    self._safe_add_rule((name, dummy_rule, weight))
                except Exception as e:
                    logger.warning(f"Failed to validate rule {rule}: {e}")

    def _update_knowledge_db(self, new_rules):
        """Append new rules to the knowledge DB file specified in self.persist_file."""
        if not self.persist_file:
            logger.warning("No persist_file set; skipping knowledge DB update.")
            return
    
        try:
            # Load existing DB
            if os.path.exists(self.persist_file):
                with open(self.persist_file, "r", encoding="utf-8") as f:
                    db = json.load(f)
            else:
                db = {"knowledge": [], "rules": []}
    
            for rule in new_rules:
                name, func, weight = rule
                db["rules"].append([name, func.__name__, weight])  # Save function name
    
            with open(self.persist_file, "w", encoding="utf-8") as f:
                json.dump(db, f, indent=4)
    
            logger.info(f"Appended {len(new_rules)} discovered rules to {self.persist_file}")
        except Exception as e:
            logger.error(f"Failed to update knowledge DB: {e}", exc_info=True)

    def _safe_add_rule(self, rule):
        """
        Validate rule via AZR before adding.
        rule: tuple -> (rule_name: str, rule_fn: Callable, weight: float)
        """
        rule_name, rule_fn, _ = rule
        # Simplified placeholder check using rule name
        contradiction_score = self.agent_factory.validate_with_azr((rule_name, "is", "inconsistent"))

        
        if contradiction_score < 0.3:
            self.rule_engine.rules.append(rule)
            logger.info(f"Rule accepted by AZR: {rule_name}")
        else:
            logger.warning(f"Rule rejected by AZR (contradiction_score={contradiction_score:.2f}): {rule_name}")

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
                rule = (rule_name, lambda kg: self._apply_ml_rule(kg, itemset), 0.65)
                self._safe_add_rule(rule)
        
        # Update RuleEngine and save to knowledge_db.json
        self.rule_engine.rules.extend(new_rules)
        self._update_knowledge_db(new_rules)

        return

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
        cache_hit = self.cache.get(cache_key) is not None
        self.performance_metrics['retrieval_times'].append(time.time() - start_time)
        self.performance_metrics['cache_hits'].append(1 if cache_hit else 0)
        

        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            return cached_result

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
            if doc_vec.size == 0:
                continue
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

        self.shared_memory.set("retrieved_knowledge", [doc['text'] for _, doc in results])
    
        if use_ontology:
            query_terms = self._preprocess(query)
            expanded_query = self._expand_with_ontology(query_terms)
            #return super().retrieve(expanded_query, k=k)
            return self.retrieve(expanded_query, k=k, use_ontology=False)
        
        
        # Apply inference rules to results
        knowledge_graph = {doc['doc_id']: doc['text'] for _, doc in results}
        inferred = self.rule_engine.apply(knowledge_graph)
        
        # Inferred knowledge to results
        for fact, conf in inferred.items():
            results.append((conf, {'text': fact, 'metadata': {'inferred': True}}))

        # Governance audit point
        if self.governor:
            self.governor.audit_retrieval(
                query=query,
                results=results,
                context={
                    'timestamp': time.time(),
                    'user': self.shared_memory.get('current_user'),
                    'module': 'knowledge_retrieval'
                }
            )

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
        tokens = [token for token in text.split() if token not in self.stop_words]
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


if __name__ == "__main__":
    print("")
    print("\n=== Knowledge Agent ===")
    print("")

    shared_memory = {}
    agent_factory = lambda: None
    monitor = KnowledgeAgent(agent_factory=None)
    print("")
    print("\n=== Successfully ran the Knowledge Agent ===\n")

if __name__ == "__main__":
    import yaml
    import pprint
    from src.agents.language_agent import LanguageAgent

    print("\n=== Knowledge Agent ===\n")

    # Load configuration with UTF-8 decoding
    with open("src/agents/knowledge/configs/knowledge_config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    class SharedMemory:
        def __init__(self):
            self.store = {}
        def set(self, key, value):
            self.store[key] = value
        def get(self, key, default=None):
            return self.store.get(key, default)

    shared_memory = SharedMemory()
    agent_factory = lambda: None

    language_agent = LanguageAgent(agent_factory, shared_memory,config=config)

    agent = KnowledgeAgent(
        agent_factory=agent_factory,
        config=config,
        persist_file=config["knowledge_memory"]["persist_file"]
    )
    if config["governor"].get("enabled", False) and agent.governor:
        audit = agent.governor.full_audit()

    print("🧠 KnowledgeAgent Initialized\n")

    db_path = config["knowledge_memory"]["persist_file"]
    if db_path and os.path.exists(db_path):
        agent.load_knowledge_db(db_path)
        print(f"📥 Loaded structured knowledge from {db_path}")
    agent.initialize_stopwords()
    agent._initialize_semantic_fallback(language_agent)
    agent.add_document("Penguins are birds that cannot fly.", metadata={"source": "example_doc.txt"})
    agent.add_to_ontology("penguin", "is_a", "bird")
    print("➕ Sample document and ontology entry added.\n")

    query = "Can penguins fly?"
    print(f"🔍 Query: {query}")
    results = agent.retrieve(query)
    for score, doc in results:
        print(f"[{score:.2f}] {doc['text']}")

    if config["knowledge_memory"].get("auto_discover_rules", False):
        agent.discover_rules()
        print("🔎 Auto-discovered rules and updated rule engine.")

    if config["governor"].get("enabled", False) and agent.governor:
        audit = agent.governor.full_audit()
        print("\n🛡️ Governance Audit:")
        pprint.pprint(audit)

    agent.memory.save(config["knowledge_memory"]["persist_file"])
    print(f"\n💾 Knowledge saved to {config['knowledge_memory']['persist_file']}")
