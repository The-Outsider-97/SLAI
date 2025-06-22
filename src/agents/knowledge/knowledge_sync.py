"""
Knowledge Synchronization System for SLAI
- Maintains consistency between memory and external sources
- Implements conflict resolution strategies
"""

import time
import hashlib
import threading
import yaml, json
import requests
import psycopg2

from psycopg2 import sql
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from difflib import SequenceMatcher
from collections import defaultdict, deque

from src.agents.alignment.utils.config_loader import load_global_config, get_config_section
from src.agents.knowledge.knowledge_memory import KnowledgeMemory
from src.agents.knowledge.utils.rule_engine import RuleEngine
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Knowledge Synchronizer")
printer = PrettyPrinter

class KnowledgeSynchronizer:
    """Orchestrates knowledge consistency across components with version control"""
    
    def __init__(self):
        self.config = load_global_config()
        self.enabled = self.config.get('enabled')

        self.sync_config = get_config_section('knowledge_sync')
        self.auto_sync = self.sync_config.get('auto_sync', {
            'enabled', 'interval'
            })
        self.conflict_resolution = self.sync_config.get('conflict_resolution', {
            'strategy', 'similarity_threshold', 'auto_quarantine'
            })
        self.versioning = self.sync_config.get('versioning', {
            'enabled', 'max_versions'
            })

        self.external_config = get_config_section('external_sources')

        self.knowledge_memory = KnowledgeMemory()

        self.version_history = defaultdict(deque)
        self.sync_lock = threading.Lock()
        self.stop_event = threading.Event()

        # Initialize conflict strategies
        self.conflict_strategies = {
            'timestamp': self._resolve_by_timestamp,
            'confidence': self._resolve_by_confidence,
            'semantic': self._resolve_by_semantics,
            'rule_based': self._resolve_by_rules
        }

        # Start background sync thread
        if self.enabled:
            self._start_sync_thread()

        logger.info(f"Knowledge Synchronizer initialized")

    def _start_sync_thread(self):
        """Start background synchronization thread with graceful shutdown"""
        printer.status("SYNC", "Start background synchronization", "info")

        def sync_loop():
            logger.info("Background sync thread started")
            while not self.stop_event.is_set():
                try:
                    start_time = time.time()
                    stats = self.full_sync()
                    duration = time.time() - start_time
                    logger.info(f"Sync completed in {duration:.2f}s. "
                                f"Updates: {stats.get('memory_updates', 0)}, "
                                f"Conflicts: {stats.get('memory_conflicts', 0)}")
                except Exception as e:
                    logger.error(f"Sync failed: {str(e)}", exc_info=True)
                
                # Wait for interval or until stop event
                wait_time = self.auto_sync.get('interval', 300)
                self.stop_event.wait(wait_time)
                
            logger.info("Background sync thread stopped")
                
        self.sync_thread = threading.Thread(target=sync_loop, daemon=True)
        self.sync_thread.start()

    def resolve_conflict(self, key: str, *versions) -> Any:
        """Apply configured conflict resolution strategy"""
        printer.status("SYNC", "Apply configured conflict resolution strategy", "info")

        strategy_name = self.conflict_resolution.get('strategy', 'timestamp')
        strategy = self.conflict_strategies.get(strategy_name, self._resolve_by_timestamp)
        return strategy(key, *versions)

    def _resolve_by_timestamp(self, key: str, *versions) -> Any:
        """Select most recent version based on timestamp"""
        printer.status("SYNC", "Resolving conflicts by timestamp", "info")

        return max(versions, key=lambda v: v.get('metadata', {}).get('timestamp', 0))

    def _resolve_by_confidence(self, key: str, *versions) -> Any:
        """Select version with highest confidence score"""
        printer.status("SYNC", "Resolving conflicts with highest confidence score", "info")

        return max(versions, key=lambda v: v.get('metadata', {}).get('confidence', 0))

    def _resolve_by_semantics(self, key: str, *versions) -> Any:
        """Resolve conflicts using semantic similarity"""
        printer.status("SYNC", "Resolving conflicts using semantic similarity", "info")

        best_match = None
        highest_score = 0
        
        for version in versions:
            text = version.get('text', '')
            other_texts = [v.get('text', '') for v in versions if v != version]
            
            # Calculate average similarity to other versions
            avg_score = sum(
                SequenceMatcher(None, text, other_text).ratio()
                for other_text in other_texts
            ) / max(1, len(other_texts))
            
            if avg_score > highest_score:
                highest_score = avg_score
                best_match = version
                
        return best_match or versions[0]

    def _resolve_by_rules(self, key: str, *versions) -> Any:
        """Resolve conflicts using rule engine inference"""
        printer.status("SYNC", "Resolving conflicts using rule engine inference", "info")

        # Create knowledge base from versions
        kb = {
            f"version_{idx}": {
                "text": v.get('text', ''),
                "metadata": v.get('metadata', {})
            }
            for idx, v in enumerate(versions)
        }
        
        # Apply rule engine to infer best version
        inferred = self.rule_engine.smart_apply(kb)
        
        # Select version with highest confidence
        best_version_key = max(inferred, key=lambda k: inferred[k], default=None)
        
        if best_version_key:
            version_idx = int(best_version_key.split('_')[-1])
            return versions[version_idx]
        
        return versions[0]  # Fallback to first version

    def full_sync(self, components: List[str] = None) -> Dict[str, int]:
        """
        Perform complete synchronization across specified components
        Returns: Dictionary of sync statistics
        """
        printer.status("SYNC", "Performing complete synchronization", "info")

        stats = defaultdict(int)
        components = components or ['memory', 'external', 'rules']
        
        with self.sync_lock:
            if 'memory' in components:
                stats.update(self._sync_memory_with_external())
            if 'rules' in components:
                stats.update(self._sync_rule_engine())
            if 'external' in components:
                stats.update(self._sync_with_external_sources())
                
            if self.versioning.get('enabled', False):
                self._create_version_snapshot()
                
        return stats

    def _sync_memory_with_external(self) -> Dict[str, int]:
        """Synchronize core memory with external knowledge sources"""
        stats = {'memory_updates': 0, 'memory_conflicts': 0}
        external_data = self._fetch_external_data()
        
        for key, ext_value in external_data.items():
            mem_value = self.knowledge_memory.recall(key=key)
            
            if not mem_value:
                self.knowledge_memory.update(key, ext_value)
                stats['memory_updates'] += 1
            else:
                mem_value = mem_value[0][0]  # Unpack (value, metadata)
                if self._detect_conflict(mem_value, ext_value):
                    stats['memory_conflicts'] += 1
                    resolved = self.resolve_conflict(key, mem_value, ext_value)
                    self.knowledge_memory.update(key, resolved)
                    
        return stats

    def _sync_rule_engine(self) -> Dict[str, int]:
        """Refresh rule engine with latest rules"""
        self.rule_engine = RuleEngine()

        stats = {'rules_loaded': 0}
        try:
            # Reload all rule sectors
            self.rule_engine.load_all_sectors()
            stats['rules_loaded'] = len(self.rule_engine.rules)
            logger.info(f"Reloaded {stats['rules_loaded']} rules into RuleEngine")
        except Exception as e:
            logger.error(f"Failed to sync rule engine: {str(e)}")
            stats['errors'] = 1
        return stats

    def _sync_with_external_sources(self) -> Dict[str, int]:
        """Synchronize with configured external knowledge sources"""
        stats = {'external_fetches': 0, 'external_errors': 0}
        
        for source in self.external_config:
            try:
                stats['external_fetches'] += 1
            except Exception as e:
                logger.error(f"External sync failed for {source}: {str(e)}")
                stats['external_errors'] += 1
                
        return stats

    def _fetch_external_data(self) -> Dict[str, dict]:
        """Fetch and merge knowledge entries from configured external sources"""
        merged_data = {}
    
        for source in self.external_config:
            try:
                # Handle different source types
                if isinstance(source, str):
                    # Local file source
                    data = self._fetch_from_file(source)
                elif isinstance(source, dict):
                    # Structured source definition
                    source_type = source.get('type', 'inline')
                    if source_type == 'api':
                        data = self._fetch_from_api(source)
                    elif source_type == 'database':
                        data = self._fetch_from_database(source)
                    elif source_type == 'inline':
                        data = source.get("data", {})
                    else:
                        logger.warning(f"Unknown source type: {source_type}")
                        continue
                else:
                    logger.warning(f"Unknown source format: {source}")
                    continue
                
                # Process and validate the data
                processed_data = self._process_external_data(data, source)
                
                # Merge data by keys
                for key, value in processed_data.items():
                    if not isinstance(value, dict):
                        # Normalize to standard format
                        value = {"text": str(value), "metadata": {"timestamp": time.time(), "confidence": 0.5}}

                    if 'metadata' not in value:
                        value['metadata'] = {}
                    value['metadata']['source'] = str(source)
                    
                    merged_data[key] = value
    
            except Exception as e:
                logger.error(f"Failed to load external source {source}: {str(e)}")
    
        return merged_data

    def _fetch_from_file(self, path: str) -> dict:
        """Fetch data from local file source"""
        if path.endswith((".yaml", ".yml")):
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        elif path.endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            logger.warning(f"Unsupported file type: {path}")
            return {}

    def _fetch_from_api(self, source_config: dict) -> dict:
        """Fetch data from API endpoint"""
        endpoint = source_config.get('endpoint')
        if not endpoint:
            logger.error("API source missing endpoint")
            return {}
            
        # Handle authentication
        headers = {}
        auth_type = source_config.get('auth_type')
        if auth_type == 'bearer_token':
            token = source_config.get('token')
            if token:
                headers['Authorization'] = f'Bearer {token}'
        
        # Handle query parameters
        params = source_config.get('params', {})
        
        try:
            response = requests.get(
                endpoint,
                headers=headers,
                params=params,
                timeout=10  # 10 second timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            return {}

    def _fetch_from_database(self, source_config: dict) -> dict:
        """Fetch data from database source"""
        conn_str = source_config.get('connection_string')
        tables = source_config.get('tables', [])
        if not conn_str or not tables:
            logger.error("Database source missing connection string or tables")
            return {}
            
        try:
            data = {}
            with psycopg2.connect(conn_str) as conn:
                with conn.cursor() as cursor:
                    for table in tables:
                        query = sql.SQL("SELECT * FROM {}").format(sql.Identifier(table))
                        cursor.execute(query)
                        columns = [desc[0] for desc in cursor.description]
                        for row in cursor.fetchall():
                            # Use a composite key: table_name + primary key
                            key = f"{table}_{row[0]}"
                            data[key] = dict(zip(columns, row))
            return data
        except psycopg2.Error as e:
            logger.error(f"Database query failed: {str(e)}")
            return {}

    def _process_external_data(self, data: dict, source: Any) -> dict:
        """Process and validate data from external sources"""
        logger.info(f"Processing data from source: {source}, {len(data)} items")
        
        processed = {}
        for key, value in data.items():
            try:
                # Validate required fields
                if not key or not value:
                    logger.warning(f"Skipping invalid entry: key={key}, value={value}")
                    continue
                    
                # Normalize structure
                if not isinstance(value, dict):
                    value = {'text': str(value)}
                    
                # Add default metadata if missing
                if 'metadata' not in value:
                    value['metadata'] = {}
                    
                # Set default timestamp if missing
                if 'timestamp' not in value['metadata']:
                    value['metadata']['timestamp'] = time.time()
                    
                # Set default confidence if missing
                if 'confidence' not in value['metadata']:
                    value['metadata']['confidence'] = 0.5
                    
                processed[key] = value
            except Exception as e:
                logger.warning(f"Error processing item {key}: {str(e)}")
                
        logger.info(f"Processed {len(processed)} valid items from source: {source}")
        return processed

    def _create_version_snapshot(self) -> str:
        """Create versioned snapshot of current knowledge state and return version ID"""
        # Get current memory state
        memory_state = {
            key: self.knowledge_memory.recall(key=key)[0][0]
            for key in self.knowledge_memory.keys()
        }
        
        # Get current rule state
        rule_state = [r['name'] for r in self.rule_engine.rules]
        
        snapshot = {
            'memory': memory_state,
            'rules': rule_state,
            'timestamp': time.time()
        }
        
        # Hashing logic
        json_data = json.dumps(snapshot, sort_keys=True).encode("utf-8")
        version_id = hashlib.sha256(json_data).hexdigest()[:16]
    
        self.version_history[version_id].append(snapshot)
    
        max_versions = self.versioning.get('max_versions', 10)
        if len(self.version_history) > max_versions:
            oldest = sorted(self.version_history.keys())[0]
            del self.version_history[oldest]

        return version_id

    def rollback_version(self, version_id: str, confirm: bool = False) -> bool:
        """Safe version rollback with confirmation and backup"""
        if version_id not in self.version_history:
            logger.error(f"Rollback failed: Version {version_id} not found")
            return False
            
        if not confirm:
            logger.warning("Rollback requires explicit confirmation")
            return False
            
        try:
            # Create backup before rollback
            backup_id = self._create_version_snapshot()
            logger.info(f"Created backup version: {backup_id}")
            
            snapshot = self.version_history[version_id][-1]
            
            with self.sync_lock:
                # Restore memory
                self.knowledge_memory.clear()
                for key, value in snapshot['memory'].items():
                    self.knowledge_memory.update(key, value)
                
                # Restore rules - reload all sectors which will reset rules
                self.rule_engine.load_all_sectors()
                
            logger.info(f"Successfully rolled back to version: {version_id}")
            return True
            
        except Exception as e:
            logger.error(f"Rollback failed: {str(e)}", exc_info=True)
            return False

    def stop_sync(self):
        """Stop background synchronization"""
        self.stop_event.set()
        if self.sync_thread.is_alive():
            self.sync_thread.join(timeout=5)

    def _detect_conflict(self, existing: Dict, new: Dict) -> bool:
        """Robust conflict detection with multiple strategies"""
        # 1. Simple equality check
        if existing == new:
            return False
            
        # 2. Content-based similarity
        similarity = self._calculate_content_similarity(existing, new)
        threshold = self.conflict_resolution.get('similarity_threshold', 0.7)
        
        # 3. Confidence threshold check
        existing_conf = existing.get('metadata', {}).get('confidence', 0.5)
        new_conf = new.get('metadata', {}).get('confidence', 0.5)
        confidence_diff = abs(existing_conf - new_conf)
        
        # 4. Temporal recency check
        existing_time = existing.get('metadata', {}).get('timestamp', 0)
        new_time = new.get('metadata', {}).get('timestamp', 0)
        time_diff = abs(existing_time - new_time)
        
        # Determine conflict based on thresholds
        if similarity < threshold:
            logger.debug(f"Conflict detected by similarity: {similarity:.2f} < {threshold}")
            return True
            
        if confidence_diff > 0.3:  # Significant confidence difference
            logger.debug(f"Conflict detected by confidence diff: {confidence_diff:.2f}")
            return True
            
        if time_diff > 86400 * 30:  # 30 days difference
            logger.debug(f"Conflict detected by time diff: {time_diff/86400:.1f} days")
            return True
            
        return False

    def _calculate_content_similarity(self, item1: dict, item2: dict) -> float:
        """Calculate content similarity using multiple strategies"""
        # Strategy 1: Text similarity
        text1 = item1.get('text', '')
        text2 = item2.get('text', '')
        if text1 and text2:
            return SequenceMatcher(None, text1, text2).ratio()
        
        # Strategy 2: Key-value similarity
        keys = set(item1.keys()) | set(item2.keys())
        similarities = []
        for key in keys:
            if key == 'metadata':
                continue  # Skip metadata comparison
                
            val1 = str(item1.get(key, ''))
            val2 = str(item2.get(key, ''))
            if val1 and val2:
                similarities.append(SequenceMatcher(None, val1, val2).ratio())
                
        if similarities:
            return sum(similarities) / len(similarities)
            
        # Fallback: Structural similarity
        return 1.0 if item1 == item2 else 0.0

if __name__ == "__main__":
    print("\n=== Knowledge Synchronizer Test ===")
    sync = KnowledgeSynchronizer()
    printer.status("Initial sync:", sync,)
    printer.status("SYNC", sync._start_sync_thread(), "success")

    print("\n=== Synchronization Test Completed ===\n")
