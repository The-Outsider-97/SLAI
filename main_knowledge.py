import os
import json
import math
import re
from collections import defaultdict
from typing import Dict, List, Tuple
import pickle

class KnowledgeAgent:
    def __init__(self, knowledge_base_dir: str, persist_file: str = None):
        """
        Initialize the Knowledge Agent with enhanced capabilities.
        
        Args:
            knowledge_base_dir: Path to directory containing knowledge documents
            persist_file: Optional file path to save/load knowledge state
        """
        self.knowledge_base_dir = knowledge_base_dir
        self.persist_file = persist_file
        self.documents: Dict[str, str] = {}
        self.vocabulary: Dict[str, int] = {}
        self.inverted_index: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.document_norms: Dict[str, float] = {}
        self.document_count = 0
        
        if persist_file and os.path.exists(persist_file):
            self._load_state()
        else:
            self._load_documents()
            self._build_index()

    def _load_documents(self):
        """Load documents from various file formats."""
        for filename in os.listdir(self.knowledge_base_dir):
            filepath = os.path.join(self.knowledge_base_dir, filename)
            try:
                if filename.endswith('.txt'):
                    with open(filepath, 'r', encoding='utf-8') as file:
                        self.documents[filename] = file.read()
                elif filename.endswith('.json'):
                    with open(filepath, 'r', encoding='utf-8') as file:
                        data = json.load(file)
                        if isinstance(data, dict):
                            self.documents[filename] = json.dumps(data)
                        else:
                            self.documents[filename] = str(data)
            except Exception as e:
                print(f"Error loading {filename}: {str(e)}")
        
        self.document_count = len(self.documents)

    def _tokenize(self, text: str) -> List[str]:
        """Basic tokenizer with stemming and stop word removal."""
        # Convert to lowercase and remove punctuation
        text = re.sub(r'[^\w\s]', '', text.lower())
        tokens = text.split()
        
        # Simple stemming (Porter stemmer would be better)
        stemmed = []
        for token in tokens:
            if len(token) > 4:
                if token.endswith('ing'):
                    token = token[:-3]
                elif token.endswith('ly'):
                    token = token[:-2]
            stemmed.append(token)
        
        return stemmed

    def _compute_tfidf(self, term: str, document: str) -> float:
        """Compute TF-IDF score for a term in a document."""
        tf = document.count(term) / len(document.split())
        idf = math.log(self.document_count / (1 + sum(1 for d in self.documents.values() if term in d)))
        return tf * idf

    def _build_index(self):
        """Build inverted index with TF-IDF weights."""
        for doc_name, content in self.documents.items():
            tokens = self._tokenize(content)
            unique_terms = set(tokens)
            
            # Update vocabulary
            for term in unique_terms:
                if term not in self.vocabulary:
                    self.vocabulary[term] = len(self.vocabulary)
                
                # Compute TF-IDF
                self.inverted_index[term][doc_name] = self._compute_tfidf(term, content)
            
            # Precompute document norm for cosine similarity
            self.document_norms[doc_name] = math.sqrt(sum(
                weight**2 for weight in self.inverted_index[term][doc_name].values()
            ))

    def _cosine_similarity(self, query_weights: Dict[str, float], doc_name: str) -> float:
        """Compute cosine similarity between query and document."""
        dot_product = 0.0
        query_norm = math.sqrt(sum(w**2 for w in query_weights.values()))
        
        for term, q_weight in query_weights.items():
            if term in self.inverted_index and doc_name in self.inverted_index[term]:
                dot_product += q_weight * self.inverted_index[term][doc_name]
        
        doc_norm = self.document_norms.get(doc_name, 1e-10)
        return dot_product / (query_norm * doc_norm)

    def retrieve_documents(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Retrieve relevant documents using vector space model."""
        query_terms = self._tokenize(query)
        query_weights = {}
        
        # Compute query vector weights
        for term in set(query_terms):
            tf = query_terms.count(term) / len(query_terms)
            idf = math.log(self.document_count / (1 + sum(1 for d in self.documents.values() if term in d)))
            query_weights[term] = tf * idf
        
        # Score documents
        scores = []
        for doc_name in self.documents:
            score = self._cosine_similarity(query_weights, doc_name)
            scores.append((doc_name, score))
        
        # Return top K results
        return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]

    def augment_generation(self, query: str, max_length: int = 2000) -> str:
        """Generate response with context from relevant documents."""
        relevant_docs = self.retrieve_documents(query)
        response = ["Based on retrieved information:"]
        
        for doc_name, score in relevant_docs:
            content = self.documents[doc_name]
            snippet = content[:500] + "..." if len(content) > 500 else content
            response.append(f"\nFrom {doc_name} (relevance: {score:.2f}):\n{snippet}")
        
        # Ensure response doesn't exceed max length
        full_response = "\n".join(response)
        return full_response[:max_length] + "..." if len(full_response) > max_length else full_response

    def update_knowledge_base(self, new_document_path: str):
        """Add new document incrementally to the knowledge base."""
        filename = os.path.basename(new_document_path)
        with open(new_document_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        self.documents[filename] = content
        self.document_count += 1
        
        # Incremental update of inverted index
        tokens = self._tokenize(content)
        unique_terms = set(tokens)
        
        for term in unique_terms:
            if term not in self.vocabulary:
                self.vocabulary[term] = len(self.vocabulary)
            self.inverted_index[term][filename] = self._compute_tfidf(term, content)
        
        # Update document norm
        self.document_norms[filename] = math.sqrt(sum(
            weight**2 for weight in self.inverted_index[term][filename].values()
        ))

    def save_state(self):
        """Persist the current state to disk."""
        if self.persist_file:
            with open(self.persist_file, 'wb') as f:
                pickle.dump({
                    'documents': self.documents,
                    'vocabulary': self.vocabulary,
                    'inverted_index': self.inverted_index,
                    'document_norms': self.document_norms,
                    'document_count': self.document_count
                }, f)

    def _load_state(self):
        """Load persisted state from disk."""
        with open(self.persist_file, 'rb') as f:
            state = pickle.load(f)
            self.documents = state['documents']
            self.vocabulary = state['vocabulary']
            self.inverted_index = state['inverted_index']
            self.document_norms = state['document_norms']
            self.document_count = state['document_count']
