
import yaml, json
import time
import os
import heapq

from datetime import datetime
from threading import Lock
from collections import defaultdict, OrderedDict
from typing import Any, Dict, List, Optional

from logs.logger import get_logger

logger = get_logger("Evaluators Memory")

CONFIG_PATH = "src/agents/evaluators/configs/evaluator_config.yaml"

def load_config(config_path=CONFIG_PATH):
    with open(config_path, "r", encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def get_merged_config(user_config=None):
    base_config = load_config()
    if user_config:
        base_config.update(user_config)
    return base_config


class EvaluatorsMemory:
    """Memory system for evaluation processes with advanced caching, checkpointing, and tagging"""
    def __init__(self, config):
        self.config = config.get('evaluators_memory', {})
        self._setup_defaults()
        
        # Core storage with access ordering
        self.store = OrderedDict()
        self.tag_index = defaultdict(list)
        self.priority_heap = []
        self.lock = Lock()
        self.access_counter = 0
        
        # Checkpoint management
        self._init_checkpoint_dir()
        self.last_checkpoint = None

        logger.info(f"EvaluatorsMemory initialized with {self.config['max_size']} entry capacity")

    def _setup_defaults(self):
        """Set default configuration parameters"""
        self.config['max_size'] = int(self.config['max_size'])
        self.config['checkpoint_freq'] = int(self.config['checkpoint_freq'])
        self.config['tag_retention'] = int(self.config['tag_retention'])
        defaults = {
            'max_size': 5000,
            'eviction_policy': 'LRU',  # LRU or FIFO
            'checkpoint_dir': 'src/agents/evaluators/checkpoints',
            'checkpoint_freq': 500,
            'auto_save': True,
            'tag_retention': 7,  # Days to keep tagged entries
            'priority_levels': 3  # Low, Medium, High
        }
        for key, value in defaults.items():
            self.config.setdefault(key, value)

    def _init_checkpoint_dir(self):
        """Ensure checkpoint directory exists"""
        os.makedirs(self.config['checkpoint_dir'], exist_ok=True)

    def add(self, entry: Any, tags: List[str] = None, priority: str = 'medium'):
        """
        Add an evaluation entry with metadata management
        Priority: 'low', 'medium', 'high'
        """
        with self.lock:
            entry_id = f"eval_{datetime.now().timestamp()}_{self.access_counter}"
            metadata = {
                'timestamp': time.time(),
                'access_count': 0,
                'priority': priority,
                'tags': tags or [],
                'size': self._calculate_entry_size(entry)
            }

            self.store[entry_id] = {
                'data': entry,
                'metadata': metadata
            }

            # Update indexes
            if tags:
                for tag in tags:
                    self.tag_index[tag].append(entry_id)
            
            # Priority management
            heapq.heappush(self.priority_heap, (
                self._priority_to_value(priority),
                entry_id
            ))

            self._manage_capacity()
            self.access_counter += 1

            if self.config['auto_save'] and (self.access_counter % self.config['checkpoint_freq'] == 0):
                self.create_checkpoint()

    def get(self, entry_id: str = None, tag: str = None):
        """Retrieve entries by ID or tag"""
        with self.lock:
            if entry_id:
                entry = self.store.get(entry_id)
                if entry:
                    entry['metadata']['access_count'] += 1
                    if self.config['eviction_policy'] == 'LRU':
                        self.store.move_to_end(entry_id)
                return entry
            
            if tag:
                return [self.store[eid] for eid in self.tag_index.get(tag, [])]
            
            return list(self.store.values())

    def query(self, tags: List[str] = None, filters: List[str] = None, limit: int = 10) -> List[Any]:
        """Query entries by tags with optional filters"""
        with self.lock:
            results = []
            
            # Get tagged entries
            if tags:
                seen = set()
                for tag in tags:
                    for entry_id in self.tag_index.get(tag, []):
                        if entry_id not in seen:
                            seen.add(entry_id)
                            results.append(self.store[entry_id])
            
            # Apply simple value filters
            if filters:
                filtered = []
                for entry in results:
                    match = True
                    for f in filters:
                        key, val = f.split(':', 1)
                        if not self._check_filter(entry['data'], key.strip(), val.strip()):
                            match = False
                            break
                    if match:
                        filtered.append(entry)
                results = filtered
            
            return results[:limit]
    
    def _check_filter(self, data: Dict, key: str, value: str) -> bool:
        """Check if entry matches a filter condition"""
        try:
            actual_value = data.get(key.split('.')[-1])  # Simple key access
            return str(actual_value) == value
        except:
            return False

    def _manage_capacity(self):
        """Apply eviction policies when capacity is reached"""
        while len(self.store) >= self.config['max_size']:
            if self.config['eviction_policy'] == 'FIFO':
                self._evict_oldest()
            else:
                self._evict_low_priority()

    def _evict_oldest(self):
        """Remove oldest entry (FIFO)"""
        oldest = next(iter(self.store))
        self._remove_entry(oldest)

    def _evict_low_priority(self):
        """Remove lowest priority entry (LRU with priority)"""
        if self.priority_heap:
            _, entry_id = heapq.heappop(self.priority_heap)
            self._remove_entry(entry_id)

    def _remove_entry(self, entry_id: str):
        """Remove entry and update indexes"""
        if entry_id in self.store:
            # Remove from tag index
            tags = self.store[entry_id]['metadata']['tags']
            for tag in tags:
                if entry_id in self.tag_index[tag]:
                    self.tag_index[tag].remove(entry_id)
            
            # Remove from store
            del self.store[entry_id]

    def create_checkpoint(self, name: str = None):
        """Save current memory state to disk"""
        checkpoint_name = name or f"eval_memory_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        checkpoint_path = os.path.join(self.config['checkpoint_dir'], checkpoint_name)
        
        try:
            with open(checkpoint_path, 'w') as f:
                json.dump({
                    'store': self.store,
                    'tag_index': self.tag_index,
                    'config': self.config
                }, f, indent=2)
            
            self.last_checkpoint = checkpoint_path
            logger.info(f"Checkpoint created: {checkpoint_path}")
            return True
        except Exception as e:
            logger.error(f"Checkpoint failed: {str(e)}")
            return False

    def load_checkpoint(self, path: str):
        """Load memory state from checkpoint file"""
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                
            with self.lock:
                self.store = OrderedDict(data['store'])
                self.tag_index = defaultdict(list, data['tag_index'])
                self.config.update(data.get('config', {}))
            
            logger.info(f"Loaded checkpoint from {path}")
            return True
        except Exception as e:
            logger.error(f"Checkpoint load failed: {str(e)}")
            return False

    def get_statistics(self) -> Dict:
        """Return memory system statistics"""
        return {
            'total_entries': len(self.store),
            'tag_distribution': {tag: len(ids) for tag, ids in self.tag_index.items()},
            'memory_usage': sum(entry['metadata']['size'] for entry in self.store.values()),
            'checkpoint_info': {
                'last_checkpoint': self.last_checkpoint,
                'checkpoint_dir': self.config['checkpoint_dir']
            }
        }

    def clean_expired_tags(self):
        """Remove entries with expired tags based on retention policy"""
        cutoff = time.time() - (self.config['tag_retention'] * 86400)
        to_remove = []
        
        with self.lock:
            for entry_id, entry in self.store.items():
                if entry['metadata']['timestamp'] < cutoff and entry['metadata']['tags']:
                    to_remove.append(entry_id)
            
            for entry_id in to_remove:
                self._remove_entry(entry_id)

    def _priority_to_value(self, priority: str) -> int:
        """Convert priority label to numerical value"""
        return {
            'low': 0,
            'medium': 1,
            'high': 2
        }.get(priority.lower(), 1)

    def _calculate_entry_size(self, entry: Any) -> int:
        """Estimate entry size in bytes"""
        try:
            return len(json.dumps(entry).encode('utf-8'))
        except:
            return 0

    def search_entries(self,
                       search_term: str,
                       fields: Optional[List[str]] = None,
                       tags: Optional[List[str]] = None,
                       case_sensitive: bool = False) -> List[Dict]:
        """
        Perform a flexible content search across stored entries.

        Parameters:
            search_term (str): The term or phrase to search for.
            fields (List[str], optional): Specific data fields to restrict the search to.
            tags (List[str], optional): Restrict search to entries with any of the given tags.
            case_sensitive (bool): Whether the search should be case sensitive.
    
        Returns:
            List[Dict]: Entries matching the search criteria.
        """
        if not search_term:
            logger.warning("No search term provided to search_entries.")
            return []

        results = []
        search_func = (lambda content: search_term in content) if case_sensitive else \
                      (lambda content: search_term.lower() in content.lower())

        with self.lock:
            entry_ids = self.store.keys()

            # Filter by tags if specified
            if tags:
                filtered_ids = set()
                for tag in tags:
                    filtered_ids.update(self.tag_index.get(tag, []))
                entry_ids = filtered_ids & self.store.keys()

            for entry_id in entry_ids:
                entry = self.store.get(entry_id)
                if not entry:
                    continue

                try:
                    data = entry['data']
                    if fields:
                        # Only search in the specified fields
                        content_to_search = " ".join(str(data.get(field, "")) for field in fields)
                    else:
                        # Search the whole data dict
                        content_to_search = str(data)

                    if search_func(content_to_search):
                        results.append(entry)

                except Exception as e:
                    logger.error(f"Error during search in entry {entry_id}: {e}")
                    continue

        return results

# ====================== Usage Example ======================
if __name__ == "__main__":
    print("\n=== Running Adaptive Risk ===\n")
    config = load_config()
    memory = EvaluatorsMemory(config)
    
    # Store evaluation results with tags
    memory.add(
        entry={"metric": "accuracy", "value": 0.92},
        tags=["final_evaluation", "model_v3"],
        priority="high"
    )
    
    # Automatic checkpoint management
    print("Memory Statistics:", memory.get_statistics())
    
    # Manual checkpoint control
    memory.create_checkpoint("important_eval.json")
    memory.load_checkpoint("src/agents/evaluators/checkpoints/previous_eval.json")
    print(f"\n* * * * * Phase 2 * * * * *\n")
    memory._manage_capacity()

    print("Memory Capacity:", memory._manage_capacity())

    print(f"\n* * * * * Phase 3 * * * * *\n")
    search_term=None
    fields = None
    tags = None
    case_sensitive = False

    search = memory.search_entries(
        search_term=search_term,
        fields=fields,
        tags=tags,
        case_sensitive=case_sensitive)
    logger.info(f"{search}")
    print("\n=== Successfully Ran Adaptive Risk ===\n")
