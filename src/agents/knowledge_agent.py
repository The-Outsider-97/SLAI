"""
Knowledge Agent for SLAI (Scalable Learning Autonomous Intelligence)
Implements core RAG functionality with TF-IDF based retrieval from scratch
"""

import os
import json
import math
import re
from collections import defaultdict
from heapq import nlargest

class KnowledgeAgent:
    def __init__(self, shared_memory, knowledge_agent_dir=None, persist_file: str = None):
        self.shared_memory = shared_memory
        self.cache = {}
        self.cache_size = 1000
        self.knowledge_agent = []
        self.vocabulary = set()
        self.document_frequency = defaultdict(int)
        self.total_documents = 0
        self.knowledge_agent = KnowledgeAgent
        self.persist_file = persist_file
        self.memory = defaultdict(dict)
        self.stopwords = set([
            'a', 'an', 'the', 'and', 'or', 'in', 'on', 'at', 'to', 'of',
            'for', 'with', 'as', 'by', 'that', 'this', 'it', 'is', 'are',
            'was', 'were', 'be', 'been', 'being', 'from', 'have', 'has',
            'had', 'do', 'does', 'did', 'will', 'would', 'should', 'can',
            'could', 'not', 'but', 'he', 'she', 'his', 'her', 'they', 'them',
            'their', 'you', 'your', 'we', 'our', 'us', 'i', 'me', 'my', 'mine'
        ])

        if knowledge_agent_dir:
            self.load_from_directory(knowledge_agent_dir)

    def execute(self, task_data):
        # Retrieve past errors from shared memory
        failures = self.shared_memory.get("agent_stats", {}).get(self.name, {}).get("errors", [])
        for err in failures:
            if self.is_similar(task_data, err["data"]):
                self.logger.info("Recognized a known problematic case, applying workaround.")
                return self.alternative_execute(task_data)
            
        errors = self.shared_memory.get(f"errors:{self.name}", [])

        # Check if current task_data has caused errors before
        for error in errors:
            if self.is_similar(task_data, error['task_data']):
                self.handle_known_issue(task_data, error)
                return

        # Proceed with normal execution
        try:
            result = self.perform_task(task_data)
            self.shared_memory.set(f"results:{self.name}", result)
        except Exception as e:
            # Log the failure in shared memory
            error_entry = {'task_data': task_data, 'error': str(e)}
            errors.append(error_entry)
            self.shared_memory.set(f"errors:{self.name}", errors)
            raise

        pass

    def alternative_execute(self, task_data):
        """
        Fallback logic when normal execution fails or matches a known failure pattern.
        Attempts to simplify, sanitize, or reroute the input for safer processing.
        """
        try:
            # Step 1: Sanitize task data (remove noise, normalize casing, trim tokens)
            if isinstance(task_data, str):
                clean_data = task_data.strip().lower().replace('\n', ' ')
            elif isinstance(task_data, dict) and "text" in task_data:
                clean_data = task_data["text"].strip().lower()
            else:
                clean_data = str(task_data).strip()

            # Step 2: Apply a safer, simplified prompt or fallback logic
            fallback_prompt = f"Can you try again with simplified input:\n{clean_data}"
            if hasattr(self, "llm") and callable(getattr(self.llm, "generate", None)):
                return self.llm.generate(fallback_prompt)

            # Step 3: If the agent wraps another processor (e.g. GrammarProcessor, LLM), reroute
            if hasattr(self, "grammar") and callable(getattr(self.grammar, "compose_sentence", None)):
                facts = {"event": "fallback", "value": clean_data}
                return self.grammar.compose_sentence(facts)

            # Step 4: Otherwise just echo the cleaned input as confirmation
            return f"[Fallback response] I rephrased your input: {clean_data}"

        except Exception as e:
            # Final fallback — very safe and generic
            return "[Fallback failure] Unable to process your request at this time."

    def is_similar(self, task_data, past_task_data):
        """
        Compares current task with past task to detect similarity.
        Uses key overlap and value resemblance heuristics.
        """
        if type(task_data) != type(past_task_data):
            return False
    
        # Handle simple text-based tasks
        if isinstance(task_data, str) and isinstance(past_task_data, str):
            return task_data.strip().lower() == past_task_data.strip().lower()
    
        # Handle dict-based structured tasks
        if isinstance(task_data, dict) and isinstance(past_task_data, dict):
            shared_keys = set(task_data.keys()) & set(past_task_data.keys())
            similarity_score = 0
            for key in shared_keys:
                if isinstance(task_data[key], str) and isinstance(past_task_data[key], str):
                    if task_data[key].strip().lower() == past_task_data[key].strip().lower():
                        similarity_score += 1
            # Consider similar if 50% or more keys match closely
            return similarity_score >= (len(shared_keys) / 2)
    
        return False
    
    def handle_known_issue(self, task_data, error):
        """
        Attempt to recover from known failure patterns.
        Could apply input transformation or fallback logic.
        """
        self.logger.warning(f"Handling known issue from error: {error.get('error')}")
    
        # Fallback strategy #1: remove problematic characters
        if isinstance(task_data, str):
            cleaned = task_data.replace("🧠", "").replace("🔥", "")
            self.logger.info(f"Retrying with cleaned input: {cleaned}")
            return self.perform_task(cleaned)
    
        # Fallback strategy #2: modify specific fields in structured input
        if isinstance(task_data, dict):
            cleaned_data = task_data.copy()
            for key, val in cleaned_data.items():
                if isinstance(val, str) and "emoji" in error.get("error", ""):
                    cleaned_data[key] = val.encode("ascii", "ignore").decode()
            self.logger.info("Retrying task with cleaned structured data.")
            return self.perform_task(cleaned_data)
    
        # Fallback strategy #3: return a graceful degradation response
        self.logger.warning("Returning fallback response for unresolvable input.")
        return {"status": "failed", "reason": "Repeated known issue", "fallback": True}
    
    def perform_task(self, task_data):
        """
        Simulated execution method — replace with actual agent logic.
        This is where core functionality would happen.
        """
        self.logger.info(f"Executing task with data: {task_data}")
    
        if isinstance(task_data, str) and "fail" in task_data.lower():
            raise ValueError("Simulated failure due to blacklisted word.")
    
        if isinstance(task_data, dict):
            # Simulate failure on missing required keys
            required_keys = ["input", "context"]
            for key in required_keys:
                if key not in task_data:
                    raise KeyError(f"Missing required key: {key}")
    
        # Simulate result
        return {"status": "success", "result": f"Processed: {task_data}"}

    def load_from_directory(self, directory_path: str):
        """Loads all .txt and .json files in the directory as knowledge documents."""

        if not os.path.isdir(directory_path):
            print(f"[KnowledgeAgent] Invalid directory: {directory_path}")
            return

        for fname in os.listdir(directory_path):
            fpath = os.path.join(directory_path, fname)
            try:
                if fname.endswith(".txt"):
                    with open(fpath, "r", encoding="utf-8") as f:
                        text = f.read().strip()
                        self.add_document(text, metadata={"source": fname})

                elif fname.endswith(".json"):
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        for entry in data:
                            text = entry.get("text") or str(entry)
                            self.add_document(text, metadata={"source": fname})
            except Exception as e:
                print(f"[KnowledgeAgent] Failed to load {fname}: {e}")

    def retrieve(self, query, k=5, similarity_threshold=0.2):
        cache_key = hash(query)
        if cache_key in self.cache:
            return self.cache[cache_key]
    
        """Retrieve relevant documents using TF-IDF cosine similarity"""
        query_tokens = self._preprocess(query)
        if not query_tokens or not self.knowledge_agent:
            return []

        # Calculate TF-IDF vectors
        query_vector = self._calculate_tfidf(query_tokens, is_query=True)
        document_vectors = [
            (doc, self._calculate_tfidf(doc['tokens']))
            for doc in self.knowledge_agent
        ]

        # Calculate cosine similarities
        similarities = []
        for doc, doc_vector in document_vectors:
            similarity = self._cosine_similarity(query_vector, doc_vector)
            if similarity >= similarity_threshold:
                similarities.append((similarity, doc))

        if len(self.cache) > self.cache_size:
            self.cache.popitem()
            self.cache[cache_key] = results
            return results
    
        # Return top k results with similarity scores
        return nlargest(k, similarities, key=lambda x: x[0])

    def add_document(self, text, metadata=None):
        if not isinstance(text, str) or len(text.strip()) < 3:
            raise ValueError("Document text must be non-empty string")

        """Store documents with preprocessing and vocabulary update"""
        tokens = self._preprocess(text)
        self.knowledge_agent.append({'text': text, 'tokens': tokens, 'metadata': metadata})
        
        # Update vocabulary and document frequency
        unique_tokens = set(tokens)
        for token in unique_tokens:
            self.document_frequency[token] += 1
        self.vocabulary.update(unique_tokens)
        self.total_documents += 1

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

# Example usage
if __name__ == "__main__":
    agent = KnowledgeAgent()
    
    # Populate knowledge agent
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
