import threading
import time
from datetime import datetime, timedelta
from collections import defaultdict, deque
import random

class SharedMemory:
    """
    Advanced thread-safe memory with network simulation and temporal tracking
    
    Features:
    - Versioned data storage with timestamps
    - Simulated network latency
    - Access pattern tracking
    - Data expiration
    - Priority-based queuing
    - Conflict resolution
    
    Academic References:
    1. Distributed Shared Memory: Nitzberg & Lo (1991) IEEE Computer
    2. Temporal Data Management: Jensen et al. (1998) ACM Computing Surveys
    """

    def __init__(self, network_latency=(0.01, 0.1)):
        self._storage = {}
        self._lock = threading.Lock()
        self._version_history = defaultdict(deque)
        self._access_stats = defaultdict(lambda: {'count': 0, 'last_accessed': None})
        self._network_latency = network_latency
        self._data_expiration = timedelta(hours=1)
        self._pending_operations = []

    def _simulate_network(self):
        """Simulate network latency using configurable delay"""
        min_lat, max_lat = self._network_latency
        time.sleep(random.uniform(min_lat, max_lat))

    def set(self, key, value, ttl=None):
        """Store value with timestamp and optional time-to-live"""
        with self._lock:
            timestamp = datetime.now()
            self._storage[key] = {
                'value': value,
                'timestamp': timestamp,
                'expires': timestamp + ttl if ttl else None
            }
            self._version_history[key].append((timestamp, value))
            self._prune_expired()

    def get(self, key, default=None, require_fresh=False):
        """Retrieve value with network simulation and freshness check"""
        self._simulate_network()
        
        with self._lock:
            self._prune_expired()
            entry = self._storage.get(key)
            
            if require_fresh and entry and entry['expires'] < datetime.now():
                return default

            if entry:
                self._access_stats[key]['count'] += 1
                self._access_stats[key]['last_accessed'] = datetime.now()
                return entry['value']
            return default

    def get_with_timestamp(self, key):
        """Retrieve value with its creation timestamp"""
        with self._lock:
            entry = self._storage.get(key)
            return (entry['value'], entry['timestamp']) if entry else (None, None)

    def get_version_history(self, key, max_versions=5):
        """Retrieve historical versions of a key"""
        with self._lock:
            return list(self._version_history[key])[-max_versions:]

    def update(self, updates: dict):
        """Batch update with transactional semantics"""
        with self._lock:
            timestamp = datetime.now()
            for key, value in updates.items():
                self._storage[key] = {
                    'value': value,
                    'timestamp': timestamp,
                    'expires': None
                }
                self._version_history[key].append((timestamp, value))

    def _prune_expired(self):
        """Remove expired entries and maintain history"""
        now = datetime.now()
        expired_keys = []
        
        for key, entry in self._storage.items():
            if entry['expires'] and entry['expires'] < now:
                expired_keys.append(key)
                
        for key in expired_keys:
            del self._storage[key]

    def get_access_stats(self, key):
        """Retrieve access statistics for a key"""
        with self._lock:
            return self._access_stats.get(key, {'count': 0, 'last_accessed': None})

    def atomic_swap(self, key, old_value, new_value):
        """Compare-and-swap operation for concurrency control"""
        with self._lock:
            if self._storage.get(key, {}).get('value') == old_value:
                self.set(key, new_value)
                return True
            return False

    def set_with_priority(self, key, value, priority=0):
        """Priority-based value insertion"""
        with self._lock:
            self._pending_operations.append((priority, datetime.now(), key, value))
            self._pending_operations.sort(reverse=True)  # Higher priority first
            self._process_pending()

    def _process_pending(self):
        """Process queued operations in priority order"""
        while self._pending_operations:
            _, timestamp, key, value = self._pending_operations.pop(0)
            self._storage[key] = {
                'value': value,
                'timestamp': timestamp,
                'expires': None
            }

    def sync_from_node(self, node_memory):
        """Simulate distributed synchronization"""
        with self._lock:
            self._simulate_network()
            self._storage.update(node_memory._storage)
            for key in node_memory._version_history:
                self._version_history[key].extend(node_memory._version_history[key])
            self._prune_expired()

    def get_memory_map(self):
        """Return snapshot of current memory state"""
        with self._lock:
            return {
                key: {
                    'value': entry['value'],
                    'age': (datetime.now() - entry['timestamp']).total_seconds(),
                    'access_count': self._access_stats[key]['count']
                }
                for key, entry in self._storage.items()
            }

    def clear(self):
        """Clear all memory while preserving statistics"""
        with self._lock:
            self._storage.clear()
            self._version_history.clear()
            self._pending_operations = []
