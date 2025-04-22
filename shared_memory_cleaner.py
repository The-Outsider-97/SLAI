import time
import threading

class SharedMemoryCleaner(threading.Thread):
    def __init__(self, shared_memory, interval=60):
        super().__init__(daemon=True)
        self.shared_memory = shared_memory
        self.interval = interval
        self.running = True

    def run(self):
        while self.running:
            now = time.time()
            expired_keys = [k for k, expiry in self.shared_memory._expiration.items() if expiry < now]
            for key in expired_keys:
                self.shared_memory._remove_key(key)
            time.sleep(self.interval)

    def stop(self):
        self.running = False
