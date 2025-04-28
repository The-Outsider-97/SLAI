import time
import threading
import psutil
from prometheus_client import Counter, Gauge
from logs.logger import get_logger

logger = get_logger(__name__)

class SharedMemoryCleaner(threading.Thread):
    """
    Background thread that periodically cleans up expired keys
    from a SharedMemory instance, with adaptive timing and metrics tracking.
    """
    def __init__(self, shared_memory, interval=60, min_interval=10, max_interval=300, target_keys_per_minute=50):
        super().__init__(daemon=True)
        self.shared_memory = shared_memory
        self.interval = interval
        self.default_interval = interval
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.target_keys_per_minute = target_keys_per_minute
        self.running = True
        self.cleaned_keys_last_minute = 0
        self._last_metrics_reset = time.time()

    def run(self):
        """Start the cleaner loop."""
        logger.info("[Cleaner] Started shared memory cleaner thread (adaptive mode).")
        while self.running:
            try:
                cleaned = self.trigger_cleanup()
                self.cleaned_keys_last_minute += cleaned
                self._adjust_interval(cleaned)
                self._maybe_reset_metrics()
                if cleaned > 0:
                    logger.info(f"[Cleaner] Auto-cleaned {cleaned} expired keys (interval={self.interval}s)")
            except Exception as e:
                logger.error(f"[Cleaner] Error during cleanup: {e}")
            time.sleep(self.interval)

    def stop(self):
        """Stop the cleaner gracefully."""
        logger.info("[Cleaner] Stopping shared memory cleaner...")
        self.running = False

    def set_interval(self, new_interval):
        """Manually set a new cleaning interval."""
        logger.info(f"[Cleaner] Manually changing cleaning interval to {new_interval} seconds.")
        self.interval = new_interval

    def trigger_cleanup(self):
        """
        Immediately perform a cleanup pass.
        
        Returns:
            int: Number of keys removed.
        """
        now = time.time()
        expired_keys = [k for k, expiry in self.shared_memory._expiration.items() if expiry < now]
        for key in expired_keys:
            self.shared_memory._remove_key(key)
        logger.debug(f"[Cleaner] Manual cleanup: {len(expired_keys)} keys expired.")
        return len(expired_keys)

    def _adjust_interval(self, cleaned_keys):
        """Adaptively adjust cleaning interval based on pressure."""
        if cleaned_keys > self.target_keys_per_minute:
            # Too many expired items → clean more aggressively
            self.interval = max(self.min_interval, self.interval * 0.8)
        elif cleaned_keys < self.target_keys_per_minute * 0.5:
            # Too few expired items → relax cleaning
            self.interval = min(self.max_interval, self.interval * 1.2)

    def _maybe_reset_metrics(self):
        """Reset metrics tracking every minute."""
        now = time.time()
        if now - self._last_metrics_reset >= 60:
            logger.info(f"[Cleaner Metrics] Cleaned {self.cleaned_keys_last_minute} expired keys in last minute.")
            self.cleaned_keys_last_minute = 0
            self._last_metrics_reset = now

    def get_metrics(self):
        """Expose current cleaner metrics."""
        return {
            "cleaned_keys_last_minute": self.cleaned_keys_last_minute,
            "current_interval": self.interval,
            "running": self.running
        }
