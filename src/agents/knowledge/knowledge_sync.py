"""
Knowledge Synchronization System for SLAI
- Maintains consistency between memory, cache, and external sources
- Implements conflict resolution strategies
- Handles version control and rollbacks
- Synchronizes with governance system for policy enforcement
"""

import time
import hashlib
import threading
import yaml, json

from typing import Dict, List, Optional, Any
from difflib import SequenceMatcher
from collections import defaultdict, deque
from types import SimpleNamespace


from src.agents.knowledge.knowledge_cache import KnowledgeCache
from src.agents.knowledge.knowledge_monitor import KnowledgeMonitor
from src.agents.knowledge.governor import Governor
from logs.logger import get_logger

logger = get_logger("KnowledgeSync")

CONFIG_PATH = "src/agents/knowledge/configs/knowledge_config.yaml"

def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(i) for i in d]
    return d


class KnowledgeSynchronizer:
    """Orchestrates knowledge consistency across components with version control"""
    
    def __init__(self, knowledge_agent=None, 
                 config_section_name: str = "knowledge_sync",
                 config_file_path: str = CONFIG_PATH):
        self.agent = knowledge_agent or SimpleNamespace(memory={}, rule_engine=SimpleNamespace(rules=[]))
        self.config = self._load_config(config_section_name, config_file_path)
        self.version_history = defaultdict(deque)
        self.sync_lock = threading.Lock()
        self.conflict_strategies = self._init_conflict_strategies()

        self.cache = KnowledgeCache()
        # Start background sync thread
        if self.config.auto_sync.enabled:
            self._start_sync_thread()

    @property
    def governor(self):
        if not hasattr(self, "_governor"):
            self._governor = Governor(knowledge_agent=self.agent)
        return self._governor
    
    @property
    def monitor(self):
        if not hasattr(self, "_monitor"):
            self._monitor = KnowledgeMonitor(agent=self.agent)
        return self._monitor


    def _load_config(self, section: str, path: str) -> SimpleNamespace:
        try:
            with open(path, "r", encoding='utf-8') as f:
                full_config = yaml.safe_load(f) or {}
            section_config = full_config.get(section, {})
        except Exception as e:
            logger.warning(f"Could not load config file: {e}")
            section_config = {}
    
        # Apply defaults if missing
        section_config.setdefault('auto_sync', {'enabled': False, 'interval': 300})
        section_config.setdefault('versioning', {'enabled': False, 'max_versions': 10})
        section_config.setdefault('conflict_resolution', {'strategy': 'timestamp', 'similarity_threshold': 0.7, 'auto_quarantine': False})
        section_config.setdefault('external_sources', [])
    
        return dict_to_namespace(section_config)

    def _init_conflict_strategies(self) -> Dict[str, callable]:
        """Initialize conflict resolution strategies"""
        return {
            'timestamp': self._resolve_by_timestamp,
            'confidence': self._resolve_by_confidence,
            'semantic': self._resolve_by_semantics,
            'governance': self._resolve_by_governance
        }

    def full_sync(self, components: List[str] = None) -> Dict[str, int]:
        """
        Perform complete synchronization across specified components
        Returns: Dictionary of sync statistics
        """
        stats = defaultdict(int)
        components = components or ['memory', 'cache', 'external', 'rules']
        
        with self.sync_lock:
            if 'memory' in components:
                stats.update(self._sync_memory_with_external())
            if 'cache' in components:
                stats.update(self._sync_cache_with_memory())
            if 'rules' in components:
                stats.update(self._sync_rules_with_governance())
            if 'external' in components:
                stats.update(self._sync_with_external_sources())
                
            if self.config.versioning.enabled:
                self._create_version_snapshot()
                
        return stats

    def _sync_memory_with_external(self) -> Dict[str, int]:
        """Synchronize core memory with external knowledge sources"""
        stats = {'memory_updates': 0, 'memory_conflicts': 0}
        external_data = self._fetch_external_data()
        
        for key, ext_value in external_data.items():
            mem_value = self.agent.memory.get(key)
            
            if mem_value is None:
                self._safe_memory_update(key, ext_value)
                stats['memory_updates'] += 1
            else:
                if self._detect_conflict(mem_value, ext_value):
                    stats['memory_conflicts'] += 1
                    resolved = self.resolve_conflict(key, mem_value, ext_value)
                    self._safe_memory_update(key, resolved)
                    
        return stats
    
    def _fetch_external_data(self) -> Dict[str, dict]:
        """Fetch and merge knowledge entries from configured external sources"""
        merged_data = {}
    
        for source in self.config.external_sources:
            try:
                if isinstance(source, str):
                    # Assume a local file path
                    if source.endswith((".yaml", ".yml")):
                        with open(source, "r", encoding="utf-8") as f:
                            data = yaml.safe_load(f)
                    elif source.endswith(".json"):
                        with open(source, "r", encoding="utf-8") as f:
                            data = json.load(f)
                    else:
                        logger.warning(f"Unsupported file type for source: {source}")
                        continue
                elif isinstance(source, dict):
                    # Optionally support inline dict source configs
                    data = source.get("data", {})
                else:
                    logger.warning(f"Unknown source format: {source}")
                    continue
    
                # Merge data by keys
                for key, value in data.items():
                    if isinstance(value, dict):
                        merged_data[key] = value
                    else:
                        merged_data[key] = {"text": str(value), "metadata": {"timestamp": time.time(), "confidence": 0.5}}
    
            except Exception as e:
                logger.error(f"Failed to load external source {source}: {str(e)}")
    
        return merged_data

    def _process_external_data(self, data: dict, source: Any):
        """Process data fetched from external sources"""
        # For now, just log that data was received
        logger.info(f"Processing data from source: {source}, {len(data)} items")

    def _sync_cache_with_memory(self) -> Dict[str, int]:
        """Align cache contents with current memory state"""
        stats = {'cache_invalidations': 0, 'cache_updates': 0}
        
        for key in list(self.cache.cache.keys()):
            mem_value = self.agent.memory.get(key)
            cached_value = self.cache.get(key)
            
            if not mem_value:
                del self.cache.cache[key]
                stats['cache_invalidations'] += 1
            elif cached_value != mem_value:
                self.cache.set(key, mem_value)
                stats['cache_updates'] += 1
                
        return stats

    def _sync_rules_with_governance(self) -> Dict[str, int]:
        """Update rule engine based on governance policies"""
        stats = {'rules_added': 0, 'rules_removed': 0}
        if not hasattr(self.agent, "rule_engine"):
            logger.warning("Agent has no rule engine. Skipping rule sync.")
            return stats
        current_rules = set(self.agent.rule_engine.rules)
        
        # Get governance-approved rules
        approved_rules = set(self.governor.get_approved_rules())
        
        # Add new rules
        for rule in approved_rules - current_rules:
            self.agent.rule_engine.add_rule(**rule)
            stats['rules_added'] += 1
            
        # Remove revoked rules
        for rule in current_rules - approved_rules:
            self.agent.rule_engine.remove_rule(rule.name)
            stats['rules_removed'] += 1
            
        return stats

    def resolve_conflict(self, key: str, *versions) -> Any:
        """Apply configured conflict resolution strategy"""
        strategy = self.conflict_strategies.get(
            self.config.conflict_resolution.strategy,
            self._resolve_by_timestamp
        )
        return strategy(key, *versions)

    def _resolve_by_timestamp(self, key: str, *versions) -> Any:
        """Select most recent version based on timestamp"""
        return max(versions, key=lambda v: v.metadata.get('timestamp', 0))

    def _resolve_by_confidence(self, key: str, *versions) -> Any:
        """Select version with highest confidence score"""
        return max(versions, key=lambda v: v.metadata.get('confidence', 0))

    def _resolve_by_semantics(self, key: str, *versions) -> Any:
        """Resolve conflicts using semantic similarity with governance guidelines"""
        guidelines = self.governor.get_guidelines()
        best_match = None
        highest_score = 0
        
        for version in versions:
            text = version.get('text', '')
            scores = [
                SequenceMatcher(None, text, g_text).ratio()
                for g_text in guidelines.get('principles', [])
            ]
            avg_score = sum(scores) / len(scores) if scores else 0
            
            if avg_score > highest_score:
                highest_score = avg_score
                best_match = version
                
        return best_match or versions[0]

    def _resolve_by_governance(self, key: str, *versions) -> Any:
        """Let governance system decide based on audit results"""
        audit_report = self.governor.audit_retrieval(
            query=f"Conflict resolution for {key}",
            results=[(1.0, v) for v in versions],
            context={'type': 'conflict_resolution'}
        )
        
        if audit_report.get('violations'):
            return self._handle_violating_content(versions, audit_report)
        return versions[0]

    def _handle_violating_content(self, versions, audit_report):
        """Handle content violating governance policies"""
        if self.config.conflict_resolution.auto_quarantine:
            self.monitor.quarantine_items(
                [v['text'] for v in versions],
                reason="conflict_resolution_violation"
            )
        return None  # Or default safe value

    def _sync_with_external_sources(self) -> Dict[str, int]:
        """Synchronize with configured external knowledge sources"""
        stats = {'external_fetches': 0, 'external_errors': 0}
        
        for source in self.config.external_sources:
            try:
                data = self._fetch_external_source(source)
                self._process_external_data(data, source)
                stats['external_fetches'] += 1
            except Exception as e:
                logger.error(f"External sync failed for {source}: {str(e)}")
                stats['external_errors'] += 1
                
        return stats

    def _fetch_external_source(self, source: dict) -> List[dict]:
        """Fetch data from external source with authentication"""
        # Implementation would vary by source type
        return []  # Placeholder

    def _create_version_snapshot(self):
        """Create versioned snapshot of current knowledge state"""
        snapshot = {
            'memory': dict(self.agent.memory),
            'cache': dict(self.cache.cache),
            'rules': [r.name for r in self.agent.rule_engine.rules],
            'timestamp': time.time()
        }
        
        # Hashing logic
        json_data = json.dumps(snapshot, sort_keys=True).encode("utf-8")
        version_id = hashlib.sha256(json_data).hexdigest()[:16]
    
        self.version_history[version_id].append(snapshot)
    
        if len(self.version_history) > self.config.versioning.max_versions:
            oldest = sorted(self.version_history.keys())[0]
            del self.version_history[oldest]

    def rollback_version(self, version_id: str) -> bool:
        """Restore system to previous version"""
        if version_id not in self.version_history:
            return False
            
        snapshot = self.version_history[version_id][-1]
        
        with self.sync_lock:
            # Restore memory
            self.agent.memory.clear()
            self.agent.memory.update(snapshot['memory'])
            
            # Restore cache
            self.cache.cache.clear()
            self.cache.cache.update(snapshot['cache'])
            
            # Restore rules
            self.agent.rule_engine.rules = [
                r for r in self.agent.rule_engine.rules 
                if r.name in snapshot['rules']
            ]
            
        return True

    def _safe_memory_update(self, key: str, value: dict):
        """Update memory with governance validation"""
        if self.governor:
            audit = self.governor.audit_retrieval(
                query=f"Memory update: {key}",
                results=[(1.0, value)],
                context={'type': 'memory_update'}
            )
            
            if not audit.get('violations'):
                self.agent.memory[key] = value
            else:
                self.monitor.handle_violations(audit['violations'])
        else:
            self.agent.memory[key] = value

    def _start_sync_thread(self):
        """Start background synchronization thread"""
        def sync_loop():
            while True:
                self.full_sync()
                time.sleep(self.config.auto_sync.interval)
                
        thread = threading.Thread(target=sync_loop, daemon=True)
        thread.start()

    def _detect_conflict(self, existing, new) -> bool:
        """Detect meaningful conflicts between knowledge entries"""
        if existing == new:
            return False
            
        # Content-based conflict detection
        text_sim = SequenceMatcher(
            None, 
            existing.get('text', ''), 
            new.get('text', '')
        ).ratio()
        
        return text_sim < self.config.conflict_resolution.similarity_threshold

if __name__ == "__main__":
    from unittest.mock import Mock
    
    print("\n=== Knowledge Synchronizer Test ===")
    sync = KnowledgeSynchronizer()
    print("Initial sync:", sync.full_sync())

    if sync.config.versioning.enabled:
        sync._create_version_snapshot()
        print(f"Versions stored: {len(sync.version_history)}")

        
    print("\n=== Synchronization Test Completed ===\n")
