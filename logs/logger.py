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
import atexit

from datetime import datetime
from logging.handlers import RotatingFileHandler
from collections import deque
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.utils.system_optimizer import SystemOptimizer

# Shared logging queue
log_queue = queue.Queue()

# Global flag to track initialization
_logger_initialized = False

class ColorFormatter(logging.Formatter):
    COLOR_CODES = {
        'RESET': "\033[0m",
        'BLUE': "\033[94m",
        'GREEN': "\033[92m",
        'YELLOW': "\033[93m",
        'RED': "\033[91m",
    }

    def format(self, record):
        level = record.levelname
        message = record.getMessage()

        if "initializ" in message.lower():
            color = self.COLOR_CODES['BLUE']
        elif "load" in message.lower() and record.levelno < logging.WARNING:
            color = self.COLOR_CODES['GREEN']
        elif record.levelno >= logging.CRITICAL:
            color = self.COLOR_CODES['RED']
        elif record.levelno >= logging.WARNING:
            color = self.COLOR_CODES['YELLOW']
        else:
            color = self.COLOR_CODES['RESET']

        formatted = f"{record.levelname}:{record.name}:{message}"
        return f"{color}{formatted}{self.COLOR_CODES['RESET']}"

class QueueLogHandler(logging.Handler):
    def __init__(self, q: queue.Queue, batch_size: int = 10, flush_interval: int = 5) -> None:
        super().__init__()
        self.queue = q
        self.batch = []
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.last_flush = time.time()
        self.hash_chain = hashlib.sha256(b'initial_seed').hexdigest()

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        self.batch.append(msg)
        current_time = time.time()

        # Batch processing using Little's Law fundamentals
        if len(self.batch) >= self.batch_size or \
           current_time - self.last_flush >= self.flush_interval:
            self._flush_batch()

    def _flush_batch(self) -> None:
        # Cryptographic chaining for tamper evidence
        chain_hash = self.hash_chain
        for msg in self.batch:
            chain_hash = hashlib.sha256(chain_hash.encode('utf-8') + msg.encode('utf-8')).hexdigest()
            self.queue.put((chain_hash, msg))
        self.hash_chain = chain_hash
        self.batch.clear()
        self.last_flush = time.time()

def get_logger(name: str) -> logging.Logger:
    global _logger_initialized, log_queue
    logger = logging.getLogger(name)
    
    if not _logger_initialized:
        _logger_initialized = True
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s')

        # Initialize root logger first
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)

        # Remove any existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # File handler
        file_handler = RotatingFileHandler(
            'logs/app.log', 
            maxBytes=1000000, 
            backupCount=5, 
            delay=True  # Defer file opening until first log
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        # Console handler with colors
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(ColorFormatter())
        root_logger.addHandler(console_handler)

        # Queue handler
        handler = QueueLogHandler(log_queue, batch_size=10, flush_interval=5)
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)

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

def exit_handler():
    cleanup_logger(None)  # Cleanup root logger

atexit.register(exit_handler)

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
        while len(self._compress_queue) > self.compress_threshold:
            old_log = self._compress_queue.popleft()
            try:
                # Read and compress with context managers
                with open(old_log, 'rb') as f:
                    data = zlib.compress(f.read(), level=9)
                with open(old_log + '.z', 'wb') as f:
                    f.write(data)
                os.remove(old_log)
            except PermissionError as e:
                logging.error(f"Failed to compress {old_log}: {e}")

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
        errors = list(self.error_counts)
        intervals = [t2 - t1 for t1, t2 in zip(errors, errors[1:])]
        if intervals:
            self.mean = statistics.mean(intervals)
            self.std = statistics.stdev(intervals) if len(intervals) > 1 else 0

    def _check_anomaly(self):
        if len(self.error_counts) < 2 or self.std == 0:
            return False
        latest_interval = self.error_counts[-1] - self.error_counts[-2]
        z_score = (latest_interval - self.mean) / self.std
        return abs(z_score) > self.sigma
