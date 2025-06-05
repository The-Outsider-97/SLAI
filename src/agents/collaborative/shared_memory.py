"""
Provides a thread-safe, in-memory shared storage mechanism with features
tailored for AI model coordination and data sharing within a single process.
"""

import time
import heapq
import pickle
import random
import fnmatch
import datetime
import threading

from pympler import asizeof
from typing import Any, OrderedDict, Optional
from collections import namedtuple, defaultdict, deque

from src.agents.adaptive.utils.config_loader import load_global_config, get_config_section
from logs.logger import get_logger

logger = get_logger("Shared Memory")

# Using deque for versions allows efficient append and potentially limiting version history size
VersionedItem = namedtuple('VersionedItem', ['timestamp', 'value'])

class SharedMemory:
    """
    A thread-safe shared memory implementation for single-process use.

    Provides versioned storage, expiration, access tracking, priority queuing,
    and basic locking for conflict resolution.

    Note: This implementation uses threading.Lock() and is suitable for use
    with multiple threads within the *same* process. It is *not* suitable
    for inter-process communication (IPC) across different processes.
    For IPC, consider using `multiprocessing.Manager` or external solutions
    like Redis or Memcached.
    """
    _instance = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance.__initialized = False
        return cls._instance
    
    def __init__(self):
        # Using deque allows efficient append and limiting version count if max_versions is set
        # Force conversion to integer or None
        if getattr(self, '_SharedMemory__initialized', False):
            return
        if self.__initialized:
            return
        self.__initialized = True
        self.config = load_global_config()
        self.memory_config = get_config_section('shared_memory')

        # Extract parameters from merged config
        self.max_memory = self.memory_config.get('max_memory_mb', 100) * 1024**2
        self._max_versions = self.memory_config.get('max_versions', 10)
        self.ttl_check_interval = self.memory_config.get('ttl_check_interval', 30) # Default time-to-live
        self.network_latency = self.memory_config.get('network_latency', 0.0)
        self._default_ttl = self.memory_config.get('default_ttl', 3600)
        # self.data = {}
        self.callbacks = defaultdict(list)
        self._lock = threading.RLock()
        self.subscribers = {}

        self._data = defaultdict(lambda: deque(maxlen=self._max_versions))
        self._expiration = {}  # key -> expiry time
        self.current_memory = 0  # Initialize memory usage counter
        self._access_log = OrderedDict()  # key -> last access time
        self._priority_queue = []
        self._last_cleanup_time = None
        self._last_cleanup_count = 0

        # Start background cleanup thread
        self._start_expiration_cleaner()

        logger.info(f"Shared Memory succesfully initialized with:")
        logger.info(f"\nCounter: {self.current_memory}\nExperation: {self._expiration}\nSubscribers: {self.subscribers}\n")

    def get_usage_stats(self) -> dict:
        """Returns detailed statistics about memory usage and system performance."""
        with self._lock:
            # Calculate memory usage percentages
            memory_usage_pct = (self.current_memory / self.max_memory) * 100 if self.max_memory > 0 else 0
            available_memory_mb = (self.max_memory - self.current_memory) / (1024 ** 2)
            
            # Count expired but not yet cleaned items
            current_time = time.time()
            pending_expiration_count = sum(
                1 for expiry in self._expiration.values() 
                if expiry <= current_time
            )
            
            # Calculate average item size
            total_items = len(self._data)
            avg_item_size = self.current_memory / total_items if total_items > 0 else 0
            
            return {
                # Memory capacity information
                'current_memory_mb': self.current_memory / (1024 ** 2),
                'max_memory_mb': self.max_memory / (1024 ** 2),
                'available_memory_mb': available_memory_mb,
                'memory_usage_percentage': round(memory_usage_pct, 2),
                
                # Item statistics
                'item_count': total_items,
                'avg_item_size_kb': round(avg_item_size / 1024, 2),
                'max_versions_per_item': self._max_versions,
                
                # Expiration information
                'expiration_count': len(self._expiration),
                'pending_expiration_cleanup': pending_expiration_count,
                'default_ttl_seconds': self._default_ttl,
                
                # System performance metrics
                'access_log_size': len(self._access_log),
                'priority_queue_size': len(self._priority_queue),
                'subscription_count': sum(len(callbacks) for callbacks in self.subscribers.values()),
                'callback_count': sum(len(callbacks) for callbacks in self.callbacks.values()),
                
                # Background cleaner information
                'ttl_check_interval': self.ttl_check_interval,
                'last_cleanup': getattr(self, '_last_cleanup_time', None),
            }

    def metrics(self) -> dict:
        """Returns operational metrics and usage patterns of the shared memory."""
        with self._lock:
            current_time = time.time()
            
            # Calculate access pattern metrics
            access_times = list(self._access_log.values())
            time_since_last_access = current_time - max(access_times) if access_times else 0
            
            # Calculate expiration metrics
            time_to_expiry = [
                expiry - current_time 
                for expiry in self._expiration.values()
                if expiry > current_time
            ]
            avg_time_to_expiry = sum(time_to_expiry) / len(time_to_expiry) if time_to_expiry else 0
            
            return {
                # Access pattern analysis
                'access_count': len(self._access_log),
                'time_since_last_access_seconds': round(time_since_last_access, 2),
                'access_pattern': {
                    'min_access_time': min(access_times) if access_times else 0,
                    'max_access_time': max(access_times) if access_times else 0,
                    'avg_access_time': sum(access_times) / len(access_times) if access_times else 0
                },
                
                # Expiration analysis
                'expiration_metrics': {
                    'earliest_expiry_seconds': min(time_to_expiry) if time_to_expiry else 0,
                    'latest_expiry_seconds': max(time_to_expiry) if time_to_expiry else 0,
                    'avg_expiry_seconds': round(avg_time_to_expiry, 2),
                    'expired_items_pending': sum(
                        1 for expiry in self._expiration.values() 
                        if expiry <= current_time
                    )
                },
                
                # Priority queue metrics
                'priority_queue_metrics': {
                    'highest_priority': -self._priority_queue[0][0] if self._priority_queue else 0,
                    'lowest_priority': -self._priority_queue[-1][0] if self._priority_queue else 0,
                    'avg_priority': (
                        sum(-priority for priority, _, _ in self._priority_queue) / 
                        len(self._priority_queue)
                    ) if self._priority_queue else 0
                },
                
                # System health metrics
                'cleanup_metrics': {
                    'last_cleanup_run': getattr(self, '_last_cleanup_time', None),
                    'items_cleaned_last_run': getattr(self, '_last_cleanup_count', 0)
                }
            }

# ==============================
#  2. Core Public API
# ==============================
    def _calculate_size(self, obj):
        """Rough estimate of object size in bytes."""
        return asizeof.asizeof(obj)

    def put(self, key, value, ttl=None, priority=None):
        """Stores a value with versioning and expiration."""
        self._simulate_network()
        current_time = time.time()
        
        with self._lock:
            # Create new version
            new_version = VersionedItem(timestamp=current_time, value=value)
            size = self._calculate_size(value)
            self.current_memory += size
            
            # Handle memory eviction if needed
            if self.current_memory > self.max_memory:
                self._evict_lru()
            
            # Store the new version
            self._data[key].append(new_version)
            self._access_log[key] = current_time
            
            # Process TTL
            effective_ttl = ttl if ttl is not None else self._default_ttl
            if effective_ttl is not None:
                if effective_ttl <= 0:
                    # Remove immediately if TTL is zero or negative
                    self._remove_key(key)
                else:
                    self._expiration[key] = current_time + effective_ttl
            elif key in self._expiration:
                # Remove existing expiration if TTL is explicitly None
                del self._expiration[key]
            
            # Add to priority queue if needed
            if priority is not None:
                heapq.heappush(self._priority_queue, (-priority, current_time, key))
                
            return current_time

    def set(self, key, value, *, ttl=None, **kwargs):
        """Set a value in shared memory with TTL and versioning"""
        self._simulate_network()
        return self.put(key, value, ttl=ttl, **kwargs)

    def get(self, key, version_timestamp=None, update_access=True, default=None):
        self._simulate_network()
        with self._lock:
            current_time = time.time()
            if key not in self._data or self._is_expired(key, current_time):
                if key in self._data:
                    self._remove_key(key)
                return default
    
            if update_access:
                if key in self._access_log:
                    self._access_log.move_to_end(key)
                else:
                    self._access_log[key] = current_time
    
            versions = self._data[key]
            if not versions:
                return default
    
            if version_timestamp is None:
                return versions[-1].value
            else:
                # Convert version_timestamp to float if possible
                try:
                    version_ts = float(version_timestamp)
                except (TypeError, ValueError):
                    version_ts = version_timestamp
                    
                # Find the version with the largest timestamp <= version_timestamp
                found_version = None
                for version in reversed(versions):
                    try:
                        if version.timestamp <= version_ts:
                            found_version = version
                            break
                    except TypeError:
                        # Handle type mismatch by skipping comparison
                        continue
                return found_version.value if found_version else default
            
        #return default

    def save_to_file(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump((self._data, self._expiration, self._access_log), f)
    
    def load_from_file(self, filename):
        with open(filename, 'rb') as f:
            self._data, self._expiration, self._access_log = pickle.load(f)

    def delete(self, key):
        """
        Deletes a key and all its associated data (versions, metadata).
        """
        with self._lock:
            if key in self._data:
                self._remove_key(key)
                return True
            return False

    def clear_all(self):
        self._data.clear()
        self._expiration.clear()
        self._access_log.clear()
        self._priority_queue.clear()

# ==============================
#  3. Versioning and Append
# ==============================
    def append(self, key: str, value: Any, ttl: Optional[int] = None, priority: Optional[int] = None):
        """
        Appends a value to a key, treating it like a new versioned entry.
        This effectively makes 'append' an alias for 'put' in terms of storage,
        but it might have different callback semantics if desired.
        """
        #with self._lock:
        #    if key not in self.data:
        #        self.data[key] = []
        #    self.data[key].append(value)
            
        #    # Trigger registered callbacks
        #    for cb in self.callbacks.get(key, []):
        #        threading.Thread(target=cb, args=(value,)).start()
        #        try:
        #            cb(value)
        #        except Exception as e:
        #            logger.error(f"Callback error for {key}: {str(e)}")



        self._simulate_network()
        current_time = time.time()

        with self._lock:
            new_version = VersionedItem(timestamp=current_time, value=value)
            item_size = self._calculate_size(value)

            # Adjust current memory: if key exists, subtract old total size of its versions
            # This is complex if append truly means "add to existing list of versions".
            # For simplicity, let's assume append just adds a new version like put.
            # If _data[key] is new, it's 0.
            
            # Store the new version in the deque
            self._data[key].append(new_version)
            self.current_memory += item_size # Add size of the new item

            # Handle memory eviction if needed
            while self.current_memory > self.max_memory and len(self._access_log) > 0: # Ensure there's something to evict
                self._evict_lru()

            self._access_log[key] = current_time
            self._access_log.move_to_end(key) # Ensure it's marked as recently accessed


            # Process TTL (same as in put)
            effective_ttl = ttl if ttl is not None else self._default_ttl
            if effective_ttl is not None:
                if effective_ttl <= 0:
                    self._remove_key(key) # Remove immediately if TTL is zero or negative
                else:
                    self._expiration[key] = current_time + effective_ttl
            elif key in self._expiration: # No TTL provided, remove any existing expiration
                del self._expiration[key]

            # Add to priority queue if needed (same as in put)
            if priority is not None:
                heapq.heappush(self._priority_queue, (-priority, current_time, key))

            # Trigger registered callbacks
            # Consider what callbacks should receive: the raw value or VersionedItem
            for cb in self.callbacks.get(key, []):
                # Run callbacks in a new thread to avoid blocking
                threading.Thread(target=self._safe_callback_call, args=(cb, value)).start()
            
            # Notify subscribers (if applicable to append logic)
            self.notify(key, value) # Assuming notify is relevant here too

            return current_time # Return timestamp similar to put

    def _safe_callback_call(self, cb, value):
        """Helper to call callbacks safely."""
        try:
            cb(value)
        except Exception as e:
            logger.error(f"Callback error: {str(e)}", exc_info=True)

    def get_all_versions(self, key, update_access=True):
        """
        Retrieves all available versions of a value associated with a key.
        """
        with self._lock:
            current_time = time.time()

            if key not in self._data or self._is_expired(key, current_time):
                if key in self._data:
                    self._remove_key(key)
                return []

            if update_access:
                self._access_log[key] = current_time

            # Return a copy of the deque as a list
            return list(self._data[key]) if key in self._data else []

    def get_access_time(self, key):
        """
        Gets the last access timestamp for a key.

        Args:
            key (hashable): The key to check.

        Returns:
            float: The timestamp of the last access (put or get),
                   or None if the key doesn't exist or has never been accessed
                   (shouldn't happen if key exists).
        """
        with self._lock:
            # Check expiration first, as expired items might still be in access log briefly
            current_time = time.time()
            if self._is_expired(key, current_time):
                if key in self._data:
                     self._remove_key(key)
                return None
            return self._access_log.get(key, None)

# ==============================
#  4. Priority Queue Support
# ==============================
    def get_next_prioritized_item(self, remove=True):
        """
        Retrieves the highest priority key from the queue.

        Checks if the retrieved key is expired before returning. Skips expired keys.

        Args:
            remove (bool): If True (default), removes the item from the queue.
                           If False, returns the item without removing it (peek).

        Returns:
            tuple(priority, key): A tuple containing the positive priority and the key,
                                  or None if the queue is empty or only contains expired/deleted items.
                                  Priority returned is positive (original value).
        """
        with self._lock:
            current_time = time.time()
            while self._priority_queue:
                if remove:
                    # Get the item with the smallest negative priority (highest actual priority)
                    neg_priority, timestamp, key = heapq.heappop(self._priority_queue)
                else:
                    neg_priority, timestamp, key = self._priority_queue[0] # Peek

                # Check if the key associated with this priority item still exists and is not expired
                if key in self._data and not self._is_expired(key, current_time):
                    if not remove:
                         # If peeking, put it back (only relevant if heapq structure needs preserving,
                         # but peeking doesn't modify) - actually not needed as [0] doesnt pop.
                         pass
                    return (-neg_priority, key) # Return positive priority
                elif not remove:
                    # If peeking and top item is invalid, we need to pop it to check next
                    heapq.heappop(self._priority_queue)
                # If removed or invalid peeked item, loop continues to find the next valid one
            return None # Queue is empty or all remaining items are invalid

# ===================================
#  5. TTL and Expiration Management
# ===================================
    def cleanup_expired(self):
        """
        Removes all items that have passed their expiration time.

        Returns:
            int: The number of items removed.
        """
        count = 0
        with self._lock:
            current_time = time.time()
            # Iterate over a copy of keys to allow deletion during iteration
            keys_to_check = list(self._expiration.keys())
            for key in keys_to_check:
                # Check expiration status again inside the loop in case state changed
                if key in self._expiration and self._expiration[key] <= current_time:
                     if key in self._data: # Ensure data exists before removing
                         self._remove_key(key)
                         count += 1
            # Update cleanup metrics
            self._last_cleanup_time = current_time
            self._last_cleanup_count = count
        return count

    def _start_expiration_cleaner(self):
        def cleaner():
            count = 0
            while True:
                time.sleep(self.ttl_check_interval)
                current_time = time.time()
                with self._lock:
                    # Create a list of keys to avoid dictionary changed during iteration
                    keys = list(self._expiration.keys())
                    for key in keys:
                        if self._is_expired(key, current_time):
                            self._remove_key(key)
                    # Update last cleanup time
                    self._last_cleanup_time = current_time
                    self._last_cleanup_count = count
        thread = threading.Thread(target=cleaner, daemon=True)
        thread.start()

    def _clean_priority_queue(self):
        """Remove entries for non-existent keys."""
        with self._lock:
            self._priority_queue = [
                item for item in self._priority_queue
                if item[2] in self._data
            ]
            heapq.heapify(self._priority_queue)

    def _is_expired(self, key, current_time):
        """Checks if a key is expired without locking."""
        return key in self._expiration and self._expiration[key] <= current_time

# ===================================
#  6. Callbacks and Subscriptions
# ===================================
    def register_callback(self, channel: str, callback: callable):
        """Register a callback for specific key updates"""
        with self._lock:
            self.callbacks[channel].append(callback)

    def publish(self, channel, message):
        for callback in self.subscribers.get(channel, []):
            callback(message)

    def subscribe(self, channel, callback):
        if channel not in self.subscribers:
            self.subscribers[channel] = []
        
        # Create wrapper that unsubscribes after first trigger
        def wrapper(value):
            """Auto-remove after first notification"""
            try:
                callback(value)
            finally:
                self.unsubscribe(channel, wrapper)

        self.subscribers[channel].append(wrapper) # Add the wrapper instead of original callback

    def unsubscribe(self, channel, callback):
        """Unsubscribe a callback."""
        if channel in self.subscribers and callback in self.subscribers[channel]:
            self.subscribers[channel].remove(callback)

    def notify(self, channel, value):
        if channel in self.subscribers:
            for callback in self.subscribers[channel]:
                threading.Thread(target=callback, args=(value,), daemon=True).start()

        for sub_channel, callbacks in self.subscribers.items():
            if fnmatch.fnmatch(channel, sub_channel):  # e.g., "music/*" matches "music/rock"
                for callback in callbacks:
                    callback(value)

# ==============================
#  7. Validation Helpers
# ==============================
    def _validate_ttl(self, value):
        """Validate TTL input (supports int/float/timedelta)."""
        if isinstance(value, datetime.timedelta):
            return value.total_seconds()
        if value is not None and value <= 0:
            raise ValueError("TTL must be positive or None.")
        return value

    def _validate_max_versions(self, value):
        """Ensure max_versions is a positive integer or None."""
        if value is None or (isinstance(value, int) and value > 0):
            return value
        logger.warning(f"Invalid max_versions: {value}. Defaulting to None.")
        return None

    def _validate_network_latency(self, value):
        """Ensure network latency is a non-negative float."""
        try:
            latency = float(value)
            return max(0.0, min(latency, 5.0))  # Clamp between 0 and 5 seconds
        except (TypeError, ValueError):
            logger.warning(f"Invalid network_latency: {value}. Using 0.5s default.")
            return 0.5
    
    def compare_and_swap(self, key, expected_value, new_value):
        with self._lock:
            current = self.get(key)
            if current == expected_value:
                self.put(key, new_value)
                return True
            return False

# ==============================
#  8. Network Simulation
# ==============================
    def _simulate_network(self):
        """Simulates network latency for distributed coordination scenarios."""
        if self.network_latency > 0:
            try:
                # Add jitter to simulate real networks
                jitter = random.uniform(-0.1, 0.1) * self.network_latency
                effective_delay = max(0, self.network_latency + jitter)
                time.sleep(effective_delay)
            except Exception as e:
                logger.warning(f"Network simulation failed: {str(e)}")

    @property
    def network_latency(self):
        """Get current network latency with validation."""
        return self._network_latency
    
    @network_latency.setter
    def network_latency(self, value):
        """Set network latency with validation and logging."""
        try:
            self._network_latency = max(0.0, min(float(value), 5.0))
            logger.info(f"Updated network latency to {self._network_latency:.2f}s")
        except (TypeError, ValueError) as e:
            logger.error(f"Invalid network latency {value}: {str(e)}")
            self._network_latency = 0.0

# ===================================
#  9. Eviction and LRU Management
# ===================================
    def _evict_lru(self):
        # Called when self.current_memory > self.max_memory
        lru_key = next(iter(self._access_log))
        self.delete(lru_key)
# ==============================
#  10. Private Key Handling
# ==============================
    def _remove_key(self, key):
        """Removes a key and associated metadata without locking."""
        if key in self._data:
            # Subtract size of all versions for this key
            size = sum(self._calculate_size(v.value) for v in self._data[key])
            self.current_memory -= size
            del self._data[key]
            
        if key in self._expiration:
            del self._expiration[key]
            
        if key in self._access_log:
            del self._access_log[key]

# ===================================
#  11. Miscellaneous Magic Methods
# ===================================
    def __len__(self):
        """Returns the exact count of non-expired keys, with live expiration checks."""
        with self._lock:
            current_time = time.time()
            expired_keys = []

            # Identify expired keys
            for key in self._data:
                if self._is_expired(key, current_time):
                    expired_keys.append(key)

            for key in expired_keys:
                self._remove_key(key)

            return len(self._data)

    def __contains__(self, key):
        """ Checks if a non-expired key exists in memory 'key in shared_memory'. """
        with self._lock:
            current_time = time.time()
            if key in self._data and not self._is_expired(key, current_time):
                return True
            # Clean up if it exists but is expired
            if key in self._data and self._is_expired(key, current_time):
                 self._remove_key(key)
            return False

    def get_all_keys(self):
        return list(self._data.keys())

# ========== Others ============
    def configure(self, default_ttl=None, max_versions=10):
        if default_ttl is not None:
            self._default_ttl = self._validate_ttl(default_ttl)
        if max_versions is not None:
            self._max_versions = self._validate_max_versions(max_versions)

    def increment(self, key, delta=1):
        with self._lock:
            current = self.get(key, update_access=False) or 0
            new_val = current + delta
            self.put(key, new_val)
            return new_val

if __name__ == "__main__":
    print("\n=== Running Shared Memory ===\n")
    sm = SharedMemory()
    print(f"\n======================================")
    print(sm.max_memory)  # Should output 200 * 1024^2
    print(sm._simulate_network())

    print("\n* * * * * Phase 2 * * * * *\n")

    weights = None
    sm = SharedMemory()
    sm.network_latency = 0.5  # Uses property setter

    # Simulates 0.5s ± 0.05s delay during operations
    sm.put("model_weights", weights) 

    print("\n* * * * * Phase 3 * * * * *\n")
    callback = callable

    caller = sm.register_callback(channel="default_channel", callback=callback)

    print(caller)
    print("\n=== Successfully Ran Shared Memory ===\n")
