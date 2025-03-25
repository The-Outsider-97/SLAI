import threading

class SharedMemory:
    """
    Thread-safe shared memory for collaborative agents.
    Stores knowledge, shared states, and embeddings.
    """
    
    def __init__(self):
        self._memory = {}
        self._lock = threading.Lock()

    def set(self, key, value):
        """
        Set a value in shared memory.
        """
        with self._lock:
            self._memory[key] = value
            print(f"[SharedMemory] SET: {key} -> {value}")

    def get(self, key):
        """
        Retrieve a value from shared memory.
        """
        with self._lock:
            value = self._memory.get(key)
            print(f"[SharedMemory] GET: {key} -> {value}")
            return value

    def delete(self, key):
        """
        Delete a key from shared memory.
        """
        with self._lock:
            if key in self._memory:
                del self._memory[key]
                print(f"[SharedMemory] DELETE: {key}")

    def keys(self):
        """
        Return all keys in shared memory.
        """
        with self._lock:
            keys = list(self._memory.keys())
            print(f"[SharedMemory] KEYS: {keys}")
            return keys

    def clear(self):
        """
        Clear all shared memory.
        """
        with self._lock:
            self._memory.clear()
            print("[SharedMemory] CLEAR: All entries removed.")
