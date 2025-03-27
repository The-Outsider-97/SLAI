import threading

class SharedMemory:
    """
    Thread-safe shared memory store for agent communication.
    """

    def __init__(self):
        self._storage = {}
        self._lock = threading.Lock()

    def set(self, key, value):
        with self._lock:
            self._storage[key] = value
            print(f"[SharedMemory] SET: {key} -> {value}")

    def get(self, key, default=None):
        with self._lock:
            return self._storage.get(key, default)

    def setdefault(self, key, default=None):
        with self._lock:
            return self._storage.setdefault(key, default)

    def keys(self):
        with self._lock:
            return list(self._storage.keys())

    def update(self, updates: dict):
        with self._lock:
            self._storage.update(updates)

    def clear(self):
        with self._lock:
            self._storage.clear()
