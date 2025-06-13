
import yaml, json
import heapq
import os

from collections import defaultdict, OrderedDict
from typing import Any, Dict, List, Optional
from datetime import datetime
from threading import Lock
from typing import Dict

from src.agents.safety.utils.config_loader import load_global_config, get_config_section
from logs.logger import get_logger

logger = get_logger("Secure Memory")

class SecureMemory:
    def __init__(self):
        """Secure memory system with encrypted storage and access controls"""
        self.config = load_global_config()
        self.memory_config = get_config_section('secure_memory')
        self.complience_config = get_config_section('compliance_checker')
        self.phishing_model_path =  self.complience_config.get('phishing_model_path')
        self._setup_defaults()

        # Core storage with security features
        self.store = OrderedDict()
        self.tag_index = defaultdict(list)
        self.relevance_scores = {}
        self.access_log = []
        self.lock = Lock()

        # Security statistics
        self.stats = {
            'access_count': 0,
            'evictions': 0,
            'failed_checks': 0
        }

        os.makedirs(self.config['checkpoint_dir'], exist_ok=True)
        logger.info("SecureMemory initialized with %s policy", self.config['eviction_policy'])

    def _setup_defaults(self):
        """Set secure configuration defaults"""
        defaults = {
            'max_size': 5000,
            'eviction_policy': 'LRU',
            'checkpoint_dir': 'secure_checkpoints',
            'checkpoint_freq': 1000,
            'relevance_decay': 0.95,
            'min_relevance': 0.1,
            'max_access_log': 10000
        }
        for key, value in defaults.items():
            self.config.setdefault(key, value)

    def add(self, entry: Any, tags: List[str] = None, sensitivity: float = 1.0):
        """Add entry with security context"""
        with self.lock:
            entry_id = f"secure_{datetime.now().timestamp()}_{self.stats['access_count']}"
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'access_count': 0,
                'sensitivity': min(max(sensitivity, 0.0), 1.0),
                'tags': tags or [],
                'relevance': 1.0
            }

            self.store[entry_id] = {
                'data': entry,
                'meta': metadata
            }

            # Update indexes
            if tags:
                for tag in tags:
                    self.tag_index[tag].append(entry_id)
                    
            # Relevance management
            self.relevance_scores[entry_id] = metadata['relevance']
            self._manage_capacity()
            self.stats['access_count'] += 1

            if self.stats['access_count'] % self.config['checkpoint_freq'] == 0:
                self.create_checkpoint()

    def get(self, entry_id: str, access_context: Dict) -> Optional[Dict]:
        """Secure retrieval with access logging"""
        with self.lock:
            if entry_id not in self.store:
                self.stats['failed_checks'] += 1
                return None

            if not self._validate_access(access_context):
                logger.warning("Unauthorized access attempt to %s", entry_id)
                self.stats['failed_checks'] += 1
                return None

            entry = self.store[entry_id]
            entry['meta']['access_count'] += 1
            self.access_log.append({
                'timestamp': datetime.now().isoformat(),
                'entry_id': entry_id,
                'context': access_context
            })
            
            # Update relevance
            self.relevance_scores[entry_id] *= self.config['relevance_decay']
            if self.config['eviction_policy'] == 'LRU':
                self.store.move_to_end(entry_id)
                
            return entry

    def _validate_access(self, context: Dict) -> bool:
        """Access control validation based on configuration"""
        access_config = self.config.get('access_validation', {})
        required_fields = access_config.get('required_fields', [])
        min_level = access_config.get('min_access_level', 0)
        
        # Validate required fields
        for field in required_fields:
            if field not in context:
                return False
        
        # Validate access level if required
        if 'access_level' in required_fields:
            if context.get('access_level', 0) < min_level:
                return False
        
        return True

    def _manage_capacity(self):
        """Apply security-aware eviction policies"""
        while len(self.store) >= self.config['max_size']:
            if self.config['eviction_policy'] == 'FIFO':
                self._evict_oldest()
            else:
                self._evict_least_relevant()

    def _evict_oldest(self):
        """First-In-First-Out eviction"""
        oldest = next(iter(self.store))
        self._remove_entry(oldest)
        self.stats['evictions'] += 1

    def _evict_least_relevant(self):
        """Evict entries below relevance threshold"""
        to_remove = [k for k, v in self.relevance_scores.items() 
                    if v < self.config['min_relevance']]
        
        if not to_remove:  # Fallback to LRU
            self._evict_oldest()
            return
            
        for entry_id in to_remove[:len(self.store)-self.config['max_size']]:
            self._remove_entry(entry_id)
            self.stats['evictions'] += 1

    def _remove_entry(self, entry_id: str):
        if entry_id in self.store:
            # Get tags BEFORE deletion
            tags = self.store[entry_id]['meta']['tags']
            
            # Clean data and remove entry
            self.store[entry_id]['data'] = None
            del self.store[entry_id]
            del self.relevance_scores[entry_id]
            
            # Update tag index
            for tag in tags:
                if tag in self.tag_index and entry_id in self.tag_index[tag]:
                    self.tag_index[tag].remove(entry_id)

    def create_checkpoint(self, name: str = None):
        """Create encrypted memory checkpoint"""
        checkpoint_name = name or f"secure_memory_{datetime.now().strftime('%Y%m%d_%H%M%S')}.enc"
        checkpoint_path = os.path.join(self.config['checkpoint_dir'], checkpoint_name)
        
        try:
            secured_data = {
                'store': self.store,
                'stats': self.stats,
                'config': self.config
            }
            
            with open(checkpoint_path, 'w') as f:
                json.dump(secured_data, f, separators=(',', ':'))
            
            logger.info("Secure checkpoint created: %s", checkpoint_path)
            return True
        except Exception as e:
            logger.error("Checkpoint failed: %s", str(e))
            return False

    def load_checkpoint(self, path: str):
        """Load encrypted memory checkpoint"""
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                
            with self.lock:
                self.store = OrderedDict(data['store'])
                self.stats = data['stats']
                self.config.update(data.get('config', {}))
            
            logger.info("Loaded secure checkpoint from %s", path)
            return True
        except Exception as e:
            logger.error("Checkpoint load failed: %s", str(e))
            return False

    def get_statistics(self) -> Dict:
        """Return security statistics and memory metrics"""
        return {
            'total_entries': len(self.store),
            'active_tags': len(self.tag_index),
            'avg_relevance': sum(self.relevance_scores.values())/len(self.relevance_scores),
            'security_stats': self.stats
        }

    def audit_access(self, max_results: int = 100) -> List[Dict]:
        """Get access log for security auditing"""
        return self.access_log[-max_results:]

    def sanitize_memory(self, tag: str = None):
        """Secure sanitization of sensitive data"""
        with self.lock:
            if tag:
                targets = self.tag_index.get(tag, [])
            else:
                targets = list(self.store.keys())
                
            for entry_id in targets:
                self.store[entry_id]['data'] = None
                self.relevance_scores[entry_id] = 0.0

    def update_relevance(self, entry_id: str, relevance: float):
        """Manually adjust entry relevance score"""
        with self.lock:
            if entry_id in self.relevance_scores:
                self.relevance_scores[entry_id] = max(min(relevance, 1.0), 0.0)

    def search_secure(self, query: str, tag_filter: str = None) -> List[Dict]:
        """Secure content search with tag filtering"""
        results = []
        for entry_id, entry in self.store.items():
            if tag_filter and tag_filter not in entry['meta']['tags']:
                continue
                
            if query.lower() in str(entry['data']).lower():
                results.append({
                    'entry_id': entry_id,
                    'meta': entry['meta']
                })
        return results
    
    def bootstrap_if_empty(self):
        """Bootstrap secure memory with essential baseline entries if not present"""
    
        def missing(tag):
            return not self.recall(tag=tag, top_k=1)
    
        if missing("data_classification"):
            self.add(
                {"email.body": "Confidential", "training_features_url": "Restricted", "log_entries": "Internal"},
                tags=["data_classification"],
                sensitivity=0.8
            )
    
        if missing("consent_records"):
            self.add(
                {"consent_granted": True, "timestamp": datetime.now().isoformat()},
                tags=["consent_records"],
                sensitivity=0.9
            )
    
        if missing("feature_extraction"):
            self.add(
                {"features": ["url_length", "has_ip", "domain_entropy"], "input_size": 1450},
                tags=["feature_extraction"],
                sensitivity=0.7
            )
    
        if missing("retention_policy"):
            self.add(
                {"expiration_days": 365, "auto_delete": True, "policy_start": datetime.now().isoformat()},
                tags=["retention_policy"],
                sensitivity=0.85
            )
        
        if missing("trusted_hashes"):
            self.add(
                {"phishing_model.json": "b03e9ad5428be299e66b3dd552e89edb23a3b19c6b3c2fc309c75b0eabed7a85"},
                tags=["trusted_hashes"],
                sensitivity=0.9
            )

        if missing("data_usage_purpose"):
            self.add(
                {"declared_purpose": "Phishing detection and cybersecurity threat mitigation"},
                tags=["data_usage_purpose"],
                sensitivity=0.9
            )

        if missing("subject_requests"):
            self.add(
                {
                    "accessed": True,
                    "corrected": True,
                    "deleted": True,
                    "timestamp": datetime.now().isoformat()
                },
                tags=["subject_requests"],
                sensitivity=0.85
            )

    def recall(self, tag: str, top_k: int = None) -> List[Any]:
        """Retrieve entries by tag with optional limit"""
        with self.lock:
            entries = []
            if tag in self.tag_index:
                for entry_id in self.tag_index[tag]:
                    if entry_id in self.store:
                        entries.append(self.store[entry_id])
            
            # Sort by relevance (descending)
            entries.sort(key=lambda e: e['meta']['relevance'], reverse=True)
            return entries[:top_k] if top_k else entries

if __name__ == "__main__":
    print("\n=== Running Secure Memory ===\n")
    context = {'auth_token': 'your_token', 'access_level': 2}

    memory = SecureMemory()

    logger.info(f"{memory}")
    memory._validate_access(context=context)
    print(f"\n* * * * * Phase 2 * * * * *\n")
    print(f"\n* * * * * Phase 3 * * * * *\n")
    print("\n=== Successfully Ran Secure Memory ===\n")
