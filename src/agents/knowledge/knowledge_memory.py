import time
import math
import yaml, json
import numpy as np

from difflib import SequenceMatcher
from collections import defaultdict, Counter
from typing import Any, Optional, List, Dict, Union

from src.agents.knowledge.utils.config_loader import load_global_config, get_config_section
from logs.logger import get_logger

logger = get_logger("Knowledge Memory")

class KnowledgeMemory:
    """
    Local memory container for knowledge-centric agents.
    Focuses on agent-local, context-aware, relevance-weighted memory entries.
    """

    def __init__(self):
        self.config = load_global_config()
        self.memory_config = get_config_section('knowledge_memory')
        self._store = defaultdict(dict)  # key -> {value, metadata}
    
    def update(self, key: str, value: Any, metadata: Optional[dict] = None,
               context: Optional[dict] = None, ttl: Optional[int] = None):
        """
        Store or update a local memory entry.
        """
        if len(self._store) >= self.memory_config.get('max_entries'):
            oldest_key = min(self._store.items(), key=lambda kv: kv[1]["metadata"]["timestamp"])[0]
            self._store.pop(oldest_key)
        timestamp = time.time()
        enriched_metadata = {
            "timestamp": timestamp,
            "context": context,
            "relevance": self._calculate_relevance(value, context) if context else 1.0,
            "expiry_time": timestamp + ttl if ttl else None,
        }

        if metadata:
            enriched_metadata.update(metadata)

        self._store[key] = {
            "value": value,
            "metadata": enriched_metadata
        }

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self._store, f, default=str)
    
    def load(self, path: str):
        with open(path, 'r') as f:
            raw = json.load(f)
            self._store = {k: v for k, v in raw.items()}

    def recall(self,
               key: Optional[str] = None,
               filters: Optional[dict] = None,
               sort_by: Optional[str] = None,
               top_k: Optional[int] = None) -> List:
        """
        Retrieve entries by key, filters, and relevance.
        """
        now = time.time()
        entries = []

        if key:
            item = self._store.get(key)
            if item and not self._is_expired(item, now):
                entries.append((item["value"], item["metadata"]))
        else:
            for entry in self._store.values():
                if not self._is_expired(entry, now):
                    entries.append((entry["value"], entry["metadata"]))

        # Apply filters
        if filters:
            entries = [e for e in entries if self._apply_filters(e[1], filters)]

        # Sort
        if sort_by:
            entries.sort(key=lambda x: x[1].get(sort_by, 0), reverse=True)

        return entries[:top_k] if top_k else entries

    def delete(self, key: str):
        if key in self._store:
            del self._store[key]

    def clear(self):
        self._store.clear()

    def keys(self):
        return list(self._store.keys())

    def get_statistics(self):
        return {
            "total_entries": len(self._store),
            "avg_relevance": np.mean([e["metadata"]["relevance"] for e in self._store.values()]),
            "expired": sum(1 for e in self._store.values() if self._is_expired(e, time.time()))
        }
    
    def search_values(self, keyword: str) -> List:
        return [(k, v) for k, v in self._store.items() if keyword.lower() in str(v["value"]).lower()]

    def _is_expired(self, entry: dict, now: float) -> bool:
        expiry = entry["metadata"].get("expiry_time")
        return expiry is not None and expiry < now

    def _apply_filters(self, metadata: dict, filters: dict) -> bool:
        return all(metadata.get(k) == v for k, v in filters.items())

    def _calculate_relevance(self, value: Any, context: dict) -> float:
        """Comprehensive relevance scoring with multiple dimensions"""
        try:
            # Normalize inputs
            val_str = str(value)
            ctx_str = self._context_to_text(context)
            
            # Initialize score components
            scores = {
                'semantic': 0.0,
                'contextual': 0.0,
                'temporal': 0.0,
                'structural': 0.0
            }
            weights = self.memory_config.get('relevance_weights')
    
            # 1. Semantic Similarity (Embedding-based)
            try:
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer('all-MiniLM-L6-v2')
                emb_val = model.encode(val_str)
                emb_ctx = model.encode(ctx_str)
                scores['semantic'] = self._cosine_sim(emb_val, emb_ctx)
            except ImportError:
                scores['semantic'] = self._fallback_semantic(val_str, ctx_str)
    
            # 2. Contextual Term Importance (TF-IDF enhanced)
            scores['contextual'] = self._contextual_term_score(val_str, ctx_str)
    
            # 3. Temporal Relevance
            scores['temporal'] = self._temporal_relevance(
                value_meta=self._store.get('_timestamp', {}),
                context_meta=context
            )
    
            # 4. Structural Similarity
            if isinstance(value, dict) and isinstance(context, dict):
                scores['structural'] = self._structural_similarity(value, context)
    
            # Weighted combination
            total_score = sum(scores[dim] * weights[dim] for dim in scores)
            return min(max(total_score, 0.0), 1.0)
    
        except Exception as e:
            logger.error(f"Relevance calculation failed: {str(e)}")
            return 0.5  # Fallback neutral score
    
    # Helper methods ---------------------------------------------------
    def _context_to_text(self, context: dict) -> str:
        """Extract meaningful text from context"""
        if isinstance(context, dict):
            return ' '.join(f"{k}={v}" for k, v in context.items())
        return str(context)
    
    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def _fallback_semantic(self, text1: str, text2: str) -> float:
        """TF-IDF/BM25 fallback when embeddings unavailable"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer()
        try:
            tfidf = vectorizer.fit_transform([text1, text2])
            return (tfidf * tfidf.T).A[0,1]
        except ValueError:
            return SequenceMatcher(None, text1, text2).ratio()
    
    def _contextual_term_score(self, value: str, context: str) -> float:
        """Weighted term importance using TF-IDF"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        try:
            # Extract key terms from context
            vectorizer = TfidfVectorizer(max_features=10)
            tfidf = vectorizer.fit_transform([context])
            important_terms = vectorizer.get_feature_names_out()
            
            # Score value based on term presence
            value_counts = Counter(value.lower().split())
            total = sum(value_counts.get(term, 0) for term in important_terms)
            return total / (len(important_terms) + 1e-5)
        except:
            return 0.0
    
    def _temporal_relevance(self, value_meta: dict, context_meta: dict) -> float:
        """Time-based decay using context timestamp"""
        ctx_time = context_meta.get('timestamp', time.time())
        val_time = value_meta.get('timestamp', ctx_time)
        time_diff = abs(ctx_time - val_time)
        
        # Exponential decay with half-life of 30 days
        half_life = 30 * 86400  # configurable
        return math.exp(-time_diff * math.log(2)/half_life)
    
    def _structural_similarity(self, dict1: dict, dict2: dict) -> float:
        """Recursive structural similarity for nested dicts"""
        def compare(a, b):
            if isinstance(a, dict) and isinstance(b, dict):
                keys = set(a.keys()) | set(b.keys())
                return sum(compare(a.get(k), b.get(k)) for k in keys)/len(keys)
            elif isinstance(a, list) and isinstance(b, list):
                return sum(compare(x, y) for x,y in zip(a,b))/max(len(a),len(b),1)
            else:
                return 1.0 if a == b else 0.0
        return compare(dict1, dict2)
        
if __name__ == "__main__":
    import readline  # Enables command history and editing
    import pprint

    km = KnowledgeMemory()
    print("📘 KnowledgeMemory Interactive CLI")
    print("Available commands: update, recall, keys, search, stats, save, load, delete, clear, exit")

    while True:
        try:
            cmd = input("\n> Command: ").strip().lower()

            if cmd == "update":
                key = input("  Key: ")
                value = input("  Value: ")
                context = input("  Context (as text or JSON): ")
                ttl = input("  TTL (in seconds, optional): ")
                try:
                    ctx_dict = json.loads(context)
                except:
                    ctx_dict = {"text": context}
                ttl_val = int(ttl) if ttl.strip() else None
                km.update(key=key, value=value, context=ctx_dict, ttl=ttl_val)
                print("✅ Entry updated.")

            elif cmd == "recall":
                key = input("  Key (optional): ").strip()
                filters = input("  Metadata filters as JSON (optional): ").strip()
                sort_by = input("  Sort by (timestamp/relevance, optional): ").strip()
                top_k = input("  Top K (optional): ").strip()

                key = key or None
                filters = json.loads(filters) if filters else None
                sort_by = sort_by or None
                top_k = int(top_k) if top_k else None

                results = km.recall(key=key, filters=filters, sort_by=sort_by, top_k=top_k)
                pprint.pprint(results)

            elif cmd == "keys":
                print("🔑 Keys in memory:", km.keys())

            elif cmd == "search":
                keyword = input("  Keyword: ")
                results = km.search_values(keyword)
                pprint.pprint(results)

            elif cmd == "stats":
                pprint.pprint(km.get_statistics())

            elif cmd == "save":
                path = input("  File path: ")
                km.save(path)
                print(f"💾 Saved to {path}")

            elif cmd == "load":
                path = input("  File path: ")
                km.load(path)
                print(f"📂 Loaded from {path}")

            elif cmd == "delete":
                key = input("  Key to delete: ")
                km.delete(key)
                print("🗑️ Entry deleted.")

            elif cmd == "clear":
                confirm = input("⚠️ Clear all memory? Type YES to confirm: ")
                if confirm == "YES":
                    km.clear()
                    print("🧹 Memory cleared.")

            elif cmd == "exit":
                print("👋 Exiting.")
                break

            else:
                print("❓ Unknown command.")

        except Exception as e:
            print(f"⚠️ Error: {e}")
