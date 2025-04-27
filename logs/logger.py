from __future__ import annotations
import logging
import os, sys
import time
import math
import queue
import psutil
import hashlib
import zlib
import statistics

from datetime import datetime
from logging.handlers import RotatingFileHandler
from collections import deque
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.utils.system_optimizer import SystemOptimizer

# Shared logging queue
log_queue = queue.Queue()

class QueueLogHandler(logging.Handler):
    def __init__(self, q, batch_size=10, flush_interval=5):
        super().__init__()
        self.queue = q
        self.batch = []
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.last_flush = time.time()
        self.hash_chain = hashlib.sha256(b'initial_seed').hexdigest()

    def emit(self, record):
        msg = self.format(record)
        self.batch.append(msg)
        current_time = time.time()

        # Batch processing using Little's Law fundamentals
        if len(self.batch) >= self.batch_size or \
           current_time - self.last_flush >= self.flush_interval:
            self._flush_batch()

    def _flush_batch(self):
        # Cryptographic chaining for tamper evidence
        chain_hash = self.hash_chain
        for msg in self.batch:
            chain_hash = hashlib.sha256(f"{chain_hash}{msg}".encode()).hexdigest()
            self.queue.put(f"{msg} | hash_chain:{chain_hash[:8]}")
        
        self.batch.clear()
        self.last_flush = time.time()

def get_logger(name="SLAI-Core", file_level=logging.DEBUG, console_level=logging.INFO, enable_queue=True, log_dir="logs"):
    logger = logging.getLogger(name)
    logger.setLevel(min(file_level, console_level))  # Root logger level

    if not logger.handlers:
        os.makedirs(os.path.join(log_dir, "sessions"), exist_ok=True)
        session_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        logger.anomaly_detector = AnomalyDetector(window_size=200, sigma=2.5)

        # Rotating app-wide log
        rotating_handler = RotatingHandler(os.path.join(log_dir, "app.log"), maxBytes=5_000_000, backupCount=2)
        rotating_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
        rotating_handler.setLevel(file_level)
        logger.addHandler(rotating_handler)

        # Per-session log
        session_handler = logging.FileHandler(os.path.join(log_dir, "sessions", f"session_{session_id}.log"))
        session_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
        session_handler.setLevel(file_level)
        logger.addHandler(session_handler)

        # Console stream
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
        console_handler.setLevel(console_level)
        logger.addHandler(console_handler)

        # Real-time Queue handler (optional for GUI)
        if enable_queue:
            queue_handler = QueueLogHandler(log_queue)
            queue_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
            queue_handler.setLevel(console_level)
            logger.addHandler(queue_handler)

        original_handle = logger.handle

        def wrapped_handle(record):
            if logger.anomaly_detector.analyze(record):
                logger.warning("Anomalous error pattern detected!", extra={'origin': 'security'})
            return original_handle(record)
        
        logger.handle = wrapped_handle

    return logger

def get_log_queue():
    return log_queue

def cleanup_logger(name):
    """
    Clean up and close all handlers for a given logger.
    Useful before rollback or app shutdown to release file locks.
    """
    logger = logging.getLogger(name)
    handlers = logger.handlers[:]
    for handler in handlers:
        handler.flush()
        handler.close()
        logger.removeHandler(handler)

class RotatingHandler(RotatingFileHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._compress_queue = deque(maxlen=5)
        self.compress_threshold = 2  # Number of backups before compression

    def doRollover(self):
        super().doRollover()
        self._compress_queue.append(self.baseFilename + ".1")
        self._manage_compression()

    def _manage_compression(self):
        # Apply Huffman coding principles via zlib
        while len(self._compress_queue) > self.compress_threshold:
            old_log = self._compress_queue.popleft()
            with open(old_log, 'rb') as f:
                data = zlib.compress(f.read(), level=9)
            with open(old_log + '.z', 'wb') as f:
                f.write(data)
            os.remove(old_log)

class ResourceLogger:
    def __init__(self, optimizer: SystemOptimizer):
        self.optimizer = optimizer
        self.cpu_history = deque(maxlen=60)  # 60 samples for 1-min window
        self.mem_history = deque(maxlen=60)
        self._gpu_initialized = False
        
    def _initialize_gpu(self):
        try:
            pynvml.nvmlInit()
            self._gpu_initialized = True
        except:
            pass

    def collect_metrics(self) -> dict:
        metrics = {
            'cpu': self._exp_smoothed_cpu(),
            'mem': psutil.virtual_memory().percent,
            'gpu': self._get_gpu_usage(),
            'throughput': self._calc_throughput(),
            'entropy': self._log_entropy()
        }
        return metrics

    def _exp_smoothed_cpu(self, alpha=0.7):
        # Exponential smoothing for noise reduction
        current = psutil.cpu_percent()
        if not self.cpu_history:
            return current
        return alpha * current + (1-alpha) * self.cpu_history[-1]

    def _get_gpu_usage(self):
        if not self._gpu_initialized:
            self._initialize_gpu()
            return 0.0
            
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return util.gpu
        except:
            return 0.0

    def _log_entropy(self):
        # Calculate information entropy of recent logs
        log_contents = "\n".join(list(get_log_queue().queue)[-100:])
        prob = {}
        for c in log_contents:
            prob[c] = prob.get(c, 0) + 1/len(log_contents)
        return -sum(p * math.log2(p) for p in prob.values() if p > 0)

class AnomalyDetector:
    def __init__(self, window_size=100, sigma=3):
        self.error_counts = deque(maxlen=window_size)
        self.sigma = sigma
        self.mean = 0
        self.std = 0
        
    def analyze(self, record):
        if record.levelno >= logging.ERROR:
            self.error_counts.append(time.time())
            self._update_stats()
            
        return self._check_anomaly()

    def _update_stats(self):
        intervals = [t2-t1 for t1,t2 in zip(self.error_counts, self.error_counts[1:])]
        if intervals:
            self.mean = statistics.mean(intervals)
            self.std = statistics.stdev(intervals) if len(intervals) > 1 else 0

    def _check_anomaly(self):
        if len(self.error_counts) < 2 or self.std == 0:
            return False
        latest_interval = self.error_counts[-1] - self.error_counts[-2]
        z_score = (latest_interval - self.mean) / self.std
        return abs(z_score) > self.sigma
