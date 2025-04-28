"""
Provides a thread-safe, in-memory shared storage mechanism with features
tailored for AI model coordination and data sharing within a single process.
"""

import heapq
import logging
import sys
import time
import pickle
import fnmatch
import datetime
import threading

from multiprocessing import Manager
from typing import Tuple, Type, Any
from collections import namedtuple, defaultdict, deque
from logs.logger import get_logger

logger = get_logger(__name__)

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

    def __init__(self, max_memory_mb=100, max_versions=10, ttl_check_interval=30, network_latency=0.0):
        from src.collaborative.registry import AgentRegistry
        from src.collaborative.task_router import TaskRouter
        # Using deque allows efficient append and limiting version count if max_versions is set
        # Force conversion to integer or None
        self.max_memory = max_memory_mb * 1024**2  # Convert MB to bytes
        self.current_memory = 0  # Track total memory used
        self.data = {}
        self.callbacks = defaultdict(list)
        self._lock = threading.Lock()
        self.subscribers = {}
        self.registry = AgentRegistry(shared_memory=self)
        self.router = TaskRouter(self.registry, shared_memory=self)
        try:
            self._max_versions = int(max_versions) if max_versions is not None else None
        except (TypeError, ValueError):
            logging.warning(f"Invalid max_versions: {max_versions}. Defaulting to None.")
            self._max_versions = None
        self._default_ttl = None

        self._data = defaultdict(lambda: deque(maxlen=self._validate_max_versions(max_versions)))
        self._expiration = {}  # key -> expiry time
        self._access_log = {}  # key -> last access time
        self._priority_queue = []
        self.network_latency = network_latency
        """
        Initializes the SharedMemory instance.

        Args:
            default_ttl (int, optional): Default time-to-live in seconds for items
                                         if not specified during 'put'. Defaults to None (no expiration).
            max_versions (int, optional): Maximum number of versions to keep per key.
                                          Older versions are discarded. Defaults to None (keep all).
        """

        # Lock for thread safety
        self._lock = threading.Lock()

        # Default time-to-live
        self.ttl_check_interval = ttl_check_interval
        self._network_latency = self._validate_network_latency(network_latency)

        # Start background cleanup thread
        self._start_expiration_cleaner()

# ==============================
#  2. Core Public API
# ==============================
    def _calculate_size(self, obj):
        """Rough estimate of object size in bytes."""
        return sys.getsizeof(obj)

    def put(self, key, value, ttl=None, priority=None):
        # Normalize timedelta to seconds
        current_time = time.time()
        new_version = VersionedItem(timestamp=current_time, value=value)
        self.current_memory += self._calculate_size(value)

        if self.current_memory > self.max_memory:
            self._evict_lru()

        if isinstance(ttl, datetime.timedelta):
            ttl = ttl.total_seconds()
        if isinstance(self._default_ttl, datetime.timedelta):
            self._default_ttl = self._default_ttl.total_seconds()

        # Check TTL type
        assert isinstance(ttl, (int, float, type(None)))

        effective_ttl = ttl if ttl is not None else self._default_ttl
        if effective_ttl is not None and effective_ttl <= 0:
            raise ValueError("TTL must be a positive number of seconds")

        """
        Stores or updates a value associated with a key.
        """
        with self._lock:
            # Store the new version
            new_version = VersionedItem(timestamp=current_time, value=value)
            self._data[key].append(new_version) # deque handles max_versions automatically

            # Update access log
            self._access_log[key] = current_time

            # Set or update expiration
            effective_ttl = ttl if ttl is not None else self._default_ttl
            if isinstance(effective_ttl, datetime.timedelta):
                effective_ttl = effective_ttl.total_seconds()

            if effective_ttl is not None:
                if effective_ttl <= 0:
                    # Expire immediately or remove existing expiration
                    if key in self._expiration:
                        del self._expiration[key]
                    # If TTL is <=0 and key exists, remove it entirely
                    self._remove_key(key)
                    return current_time # Return timestamp even if immediately removed
                else:
                    self._expiration[key] = current_time + effective_ttl
            elif key in self._expiration:
                # If ttl is explicitly None, remove any existing expiration
                del self._expiration[key]

            # Add to priority queue if priority is given
            if priority is not None:
                # Use negative priority because heapq is a min-heap
                # Include timestamp as tie-breaker (FIFO for same priority)
                heapq.heappush(self._priority_queue, (-priority, current_time, key))

            return current_time

    def get(self, key, version_timestamp=None, update_access=True, default=None):
        with self._lock:
            # your normal get logic here
            current_time = time.time()
            if key not in self._data or self._is_expired(key, current_time):
                if key in self._data:
                    self._remove_key(key)
                return None
    
            if update_access:
                self._access_log[key] = current_time
    
            versions = self._data[key]
            if not versions:
                return None
    
            if version_timestamp is None:
                return versions[-1].value
            else:
                found_version = None
                for version in reversed(versions):
                    if version.timestamp <= version_timestamp:
                        found_version = version
                        break
                return found_version.value if found_version else None

    def set(self, key, value, *, ttl=None, **kwargs):
        """Set a value in shared memory with TTL and versioning"""
        self._simulate_network()

        with self._lock:
            version = self.data[key]['version'] + 1 if key in self.data else 1
            expire_at = time.time() + ttl if ttl else None
            self.data[key] = {'value': value, 'version': version, 'expire_at': expire_at}
            new_version = VersionedItem(timestamp=time.time(), value=value)
            self._data[key].append(new_version)

        if isinstance(ttl, datetime.timedelta):
            ttl = ttl.total_seconds()
        if ttl is not None:
            try:
                ttl = int(ttl)
                self._expiration[key] = time.time() + ttl
            except Exception as e:
                logging.error(f"[SharedMemory] Invalid TTL passed: {ttl} ({e})")
                return False  # <--- Stop execution if TTL is invalid

        return self.put(key, value, ttl=ttl, **kwargs)

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
                # Note: Items related to this key remain in the priority queue until popped and checked.
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
    def append(self, key: str, value: Any):
        """Store data and trigger callbacks"""
        with self._lock:
            if key not in self.data:
                self.data[key] = []
            self.data[key].append(value)
            
            # Trigger registered callbacks
            for cb in self.callbacks.get(key, []):
                try:
                    cb(value)
                except Exception as e:
                    logging.error(f"Callback error for {key}: {str(e)}")

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
        return count

    def _start_expiration_cleaner(self):
        def cleaner():
            while True:
                time.sleep(self.ttl_check_interval)
                current_time = time.time()
                expired_keys = [k for k, v in self._expiration.items() if v < current_time]
                for key in expired_keys:
                    self._remove_key(key)
        thread = threading.Thread(target=cleaner, daemon=True)
        thread.start()

    def _is_expired(self, key, current_time):
        """Checks if a key is expired without locking."""
        if key not in self._expiration:
            return False
        return self._expiration[key] <= current_time

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
        self.subscribers[channel].append(callback)

        def wrapper(value):
            """Subscribe a callback that auto-unsubscribes after first trigger."""
            callback(value)
            self.unsubscribe(channel, wrapper)
        
        self.subscribe(channel, wrapper)

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
        logging.warning(f"Invalid max_versions: {value}. Defaulting to None.")
        return None

    def _validate_network_latency(self, value):
        """Ensure network latency is a non-negative float."""
        try:
            latency = float(value)
            return max(0.0, min(latency, 5.0))  # Clamp between 0 and 5 seconds
        except (TypeError, ValueError):
            logging.warning(f"Invalid network_latency: {value}. Using 0.5s default.")
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
        time.sleep(self.network_latency)
        try:
            delay = float(self._network_latency)
            time.sleep(min(delay, 0.5))  # Cap max delay to avoid hangs
        except Exception as e:
            logging.warning(f"Invalid network latency: {self._network_latency} ({e})")

# ===================================
#  9. Eviction and LRU Management
# ===================================
    def _evict_lru(self):
        # Called when self.current_memory > self.max_memory
        lru_key = min(self._access_log, key=lambda k: self._access_log[k])
        self.delete(lru_key)

# ==============================
#  10. Private Key Handling
# ==============================
    def _remove_key(self, key):
        """Removes a key and associated metadata without locking."""
        if key in self._data:
            del self._data[key]
        if key in self._expiration:
            del self._expiration[key]
        if key in self._access_log:
            del self._access_log[key]
        # Note: Removing from priority queue efficiently is hard.
        # We'll filter expired items when retrieving from the queue.

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
            
            # Remove expired entries
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
