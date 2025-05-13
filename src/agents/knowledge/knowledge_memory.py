import time
import yaml, json
import numpy as np

from types import SimpleNamespace
from collections import defaultdict
from typing import Any, Optional, List, Dict, Union

from logs.logger import get_logger

logger = get_logger("Knowledge Memory")

CONFIG_PATH = "src/agents/knowledge/configs/knowledge_config.yaml"

def dict_to_namespace(d):
    """Recursively convert dicts to SimpleNamespace for dot-access."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(i) for i in d]
    return d

def get_config_section(section: Union[str, Dict], config_file_path: str):
    if isinstance(section, dict):
        return dict_to_namespace(section)
    
    with open(config_file_path, "r", encoding='utf-8') as f:
        config = yaml.safe_load(f)
    if section not in config:
        raise KeyError(f"Section '{section}' not found in config file: {config_file_path}")
    return dict_to_namespace(config[section])

class KnowledgeMemory:
    """
    Local memory container for knowledge-centric agents.
    Focuses on agent-local, context-aware, relevance-weighted memory entries.
    """

    def __init__(self,
                 config_section_name: str = "knowledge_memory",
                 config_file_path: str = CONFIG_PATH
                 ):
        self.config = get_config_section(config_section_name, config_file_path)
        self._store = defaultdict(dict)  # key -> {value, metadata}
    
    def update(self, key: str, value: Any, metadata: Optional[dict] = None,
               context: Optional[dict] = None, ttl: Optional[int] = None):
        """
        Store or update a local memory entry.
        """
        if len(self._store) >= self.config.max_entries:
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
        """
        Heuristic for relevance scoring. Override in subclass for custom logic.
        """
        # Simple heuristic: shared word count ratio
        try:
            val_tokens = set(str(value).lower().split())
            ctx_text = str(context) if isinstance(context, str) else str(context.get("text", ""))
            ctx_tokens = set(ctx_text.lower().split())
            overlap = len(val_tokens & ctx_tokens)
            return overlap / (len(val_tokens) + 1e-5)
        except Exception:
            return 0.5
        
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
