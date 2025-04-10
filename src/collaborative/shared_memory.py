"""
Provides a thread-safe, in-memory shared storage mechanism with features
tailored for AI model coordination or data sharing within a single process.
"""

import time
import heapq
import logging
import time as current_time
import time
import datetime
import threading

from multiprocessing import Manager
from typing import Tuple, Type
from collections import namedtuple, defaultdict, deque


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

    def __init__(self, default_ttl=None, max_versions=None, network_latency=0.5):
        from src.collaborative.registry import AgentRegistry
        from src.collaborative.task_router import TaskRouter
        # Using deque allows efficient append and limiting version count if max_versions is set
        # Force conversion to integer or None
        self.lock = threading.Lock()
        manager = Manager()
        self.data = manager.dict()
        self.subscribers = {}
        self.registry = AgentRegistry(shared_memory=self)
        self.router = TaskRouter(self.registry, shared_memory=self)
        try:
            self._max_versions = int(max_versions) if max_versions is not None else None
        except (TypeError, ValueError):
            logging.warning(f"Invalid max_versions: {max_versions}. Defaulting to None.")
            self._max_versions = None

        self._data = defaultdict(lambda: deque(maxlen=self._validate_max_versions(max_versions)))
        self._priority_queue = []

        """
        Initializes the SharedMemory instance.

        Args:
            default_ttl (int, optional): Default time-to-live in seconds for items
                                         if not specified during 'put'. Defaults to None (no expiration).
            max_versions (int, optional): Maximum number of versions to keep per key.
                                          Older versions are discarded. Defaults to None (keep all).
        """
        # Tracking last access time: key -> timestamp
        self._access_log = {}

        # Expiration time: key -> expiration_timestamp (absolute time)
        self._expiration = {}

        # Timestamp is used as a tie-breaker (FIFO for same priority).
        self._priority_queue = []

        # Lock for thread safety
        self._lock = threading.Lock()

        # Default time-to-live
        self._default_ttl = self._validate_ttl(default_ttl)
        self._network_latency = self._validate_network_latency(network_latency)

    def _validate_max_versions(self, value):
        """Ensure max_versions is a positive integer or None."""
        if value is None or (isinstance(value, int) and value > 0):
            return value
        logging.warning(f"Invalid max_versions: {value}. Defaulting to None.")
        return None

    def _validate_ttl(self, value):
        """Validate TTL input (supports int/float/timedelta)."""
        if isinstance(value, datetime.timedelta):
            return value.total_seconds()
        if value is not None and value <= 0:
            raise ValueError("TTL must be positive or None.")
        return value

    def _validate_network_latency(self, value):
        """Ensure network latency is a non-negative float."""
        try:
            latency = float(value)
            return max(0.0, min(latency, 5.0))  # Clamp between 0 and 5 seconds
        except (TypeError, ValueError):
            logging.warning(f"Invalid network_latency: {value}. Using 0.5s default.")
            return 0.5

    def subscribe(self, key, callback):
        if key not in self.subscribers:
            self.subscribers[key] = []
        self.subscribers[key].append(callback)

    def notify(self, key, value):
        if key in self.subscribers:
            for callback in self.subscribers[key]:
                callback(value)

    def _simulate_network(self):
        time.sleep(self.network_latency)
        try:
            delay = float(self._network_latency)
            time.sleep(min(delay, 0.5))  # Cap max delay to avoid hangs
        except Exception as e:
            logging.warning(f"Invalid network latency: {self._network_latency} ({e})")

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

    def _is_expired(self, key, current_time):
        """Checks if a key is expired without locking."""
        if key not in self._expiration:
            return False
        return self._expiration[key] <= current_time

    def _remove_key(self, key):
        """Removes a key and associated metadata without locking."""
        if key in self._data:
            del self._data[key]
        if key in self._access_log:
            del self._access_log[key]
        if key in self._expiration:
            del self._expiration[key]
        # Note: Removing from priority queue efficiently is hard.
        # We'll filter expired items when retrieving from the queue.

    def put(self, key, value, ttl=None, priority=None):
        # Normalize timedelta to seconds
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

        Args:
            key (hashable): The key to store the value under.
            value (any): The value to store.
            ttl (int, optional): Time-to-live in seconds for this item.
                                 Overrides the default_ttl if provided.
                                 Use None for no expiration.
                                 Use 0 or negative for immediate expiration (useful for cleanup).
            priority (int, optional): If provided, adds the key to the priority queue
                                      with this priority (lower number = higher priority).
                                      Defaults to None (not added to queue).

        Returns:
            float: The timestamp associated with this version.
        """
        with self._lock:
            current_time = time.time()

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
        """
        Retrieves a value associated with a key.

        Args:
            key (hashable): The key of the value to retrieve.
            version_timestamp (float, optional): If provided, retrieves the latest version
                                                 at or before this timestamp.
                                                 Defaults to None (retrieve the absolute latest version).
            update_access (bool): Whether to update the last access time for this key. Defaults to True.

        Returns:
            any: The requested value, or None if the key doesn't exist, is expired,
                 or the specified version doesn't exist.
        """
        self._lock = threading.Lock()

        # 1. Check existence and expiration
        if key not in self._data or self._is_expired(key, current_time):
            if key in self._data: # It exists but is expired
                self._remove_key(key) # Clean up expired item
            return None

        # 2. Update access log if requested
        if update_access:
            self._access_log[key] = current_time

        versions = self._data[key]
        if not versions: # Should not happen if key in _data, but defensive check
                return None

        # 3. Find the correct version
        if version_timestamp is None:
            # Return the latest version's value
            return versions[-1].value
        else:
            # Find the latest version <= version_timestamp
            # Iterate backwards for efficiency as timestamps are ordered
            found_version = None
            for version in reversed(versions):
                if version.timestamp <= version_timestamp:
                    found_version = version
                    break
            return found_version.value if found_version else None

    def get_all_versions(self, key, update_access=True):
        """
        Retrieves all available versions of a value associated with a key.

        Args:
            key (hashable): The key to retrieve versions for.
            update_access (bool): Whether to update the last access time. Defaults to True.

        Returns:
            list[VersionedItem]: A list of (timestamp, value) tuples,
                                 or an empty list if the key doesn't exist or is expired.
                                 Returns a copy to prevent modification.
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
            return list(self._data[key])


    def delete(self, key):
        """
        Deletes a key and all its associated data (versions, metadata).

        Args:
            key (hashable): The key to delete.

        Returns:
            bool: True if the key existed and was deleted, False otherwise.
        """
        with self._lock:
            if key in self._data:
                self._remove_key(key)
                # Note: Items related to this key remain in the priority queue until popped and checked.
                return True
            return False

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
