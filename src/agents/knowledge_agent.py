"""
Knowledge Agent for SLAI (Scalable Learning Autonomous Intelligence)
Implements core RAG functionality with TF-IDF based retrieval from scratch
"""

import math
import re
from collections import defaultdict
from heapq import nlargest

class KnowledgeAgent:
    def __init__(self):
        self.knowledge_base = []
        self.vocabulary = set()
        self.document_frequency = defaultdict(int)
        self.total_documents = 0
        self.memory = defaultdict(dict)
        self.stopwords = set([
            'a', 'an', 'the', 'and', 'or', 'in', 'on', 'at', 'to', 'of',
            'for', 'with', 'as', 'by', 'that', 'this', 'it', 'is', 'are',
            'was', 'were', 'be', 'been', 'being', 'from', 'have', 'has',
            'had', 'do', 'does', 'did', 'will', 'would', 'should', 'can',
            'could', 'not', 'but', 'he', 'she', 'his', 'her', 'they', 'them',
            'their', 'you', 'your', 'we', 'our', 'us', 'i', 'me', 'my', 'mine'
        ])

    def add_document(self, text, metadata=None):
        """Store documents with preprocessing and vocabulary update"""
        tokens = self._preprocess(text)
        self.knowledge_base.append({'text': text, 'tokens': tokens, 'metadata': metadata})
        
        # Update vocabulary and document frequency
        unique_tokens = set(tokens)
        for token in unique_tokens:
            self.document_frequency[token] += 1
        self.vocabulary.update(unique_tokens)
        self.total_documents += 1

    def retrieve(self, query, k=5, similarity_threshold=0.2):
        """Retrieve relevant documents using TF-IDF cosine similarity"""
        query_tokens = self._preprocess(query)
        if not query_tokens or not self.knowledge_base:
            return []

        # Calculate TF-IDF vectors
        query_vector = self._calculate_tfidf(query_tokens, is_query=True)
        document_vectors = [
            (doc, self._calculate_tfidf(doc['tokens']))
            for doc in self.knowledge_base
        ]

        # Calculate cosine similarities
        similarities = []
        for doc, doc_vector in document_vectors:
            similarity = self._cosine_similarity(query_vector, doc_vector)
            if similarity >= similarity_threshold:
                similarities.append((similarity, doc))

        # Return top k results with similarity scores
        return nlargest(k, similarities, key=lambda x: x[0])

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
        """Calculate cosine similarity between two vectors"""
        intersection = set(vec_a.keys()) & set(vec_b.keys())
        numerator = sum(vec_a[x] * vec_b[x] for x in intersection)
        
        sum_a = sum(v**2 for v in vec_a.values())
        sum_b = sum(v**2 for v in vec_b.values())
        denominator = math.sqrt(sum_a) * math.sqrt(sum_b)
        
        return numerator / denominator if denominator else 0

    def update_memory(self, key, value):
        """Store information in long-term memory"""
        self.memory[key] = value

    def recall_memory(self, key):
        """Retrieve information from long-term memory"""
        return self.memory.get(key)

    def contextual_search(self, query, context_window=3):
        """Search with consideration of recent context"""
        # Implementation stub for contextual search
        return self.retrieve(query)

# Example usage
if __name__ == "__main__":
    agent = KnowledgeAgent()
    
    # Populate knowledge base
    documents = [
        "Reinforcement learning uses rewards to train agents",
        "Neural networks are computational models inspired by biological brains",
        "Transformer models use attention mechanisms for sequence processing",
        "Knowledge graphs represent information as entity-relationship triples"
    ]
    for doc in documents:
        agent.add_document(doc)
    
    # Perform query
    query = "What models are inspired by biological systems?"
    results = agent.retrieve(query)
    
    print(f"Results for query: '{query}'")
    for score, doc in results:
        print(f"[Score: {score:.4f}] {doc['text']}")
