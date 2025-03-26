import os
import sys
import torch
import logging
import queue
import random
import json
from datetime import datetime

# Shared logging queue instance
log_queue = queue.Queue()

class QueueLogHandler(logging.Handler):
    def __init__(self, q):
        super().__init__()
        self.queue = q

    def emit(self, record):
        msg = self.format(record)
        self.queue.put(msg)

# Configure global logger
logger = logging.getLogger("SLAI")
logger.setLevel(logging.INFO)

# Attach queue handler
queue_handler = QueueLogHandler(log_queue)
queue_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(queue_handler)

# Log to file per agent/session
session_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
os.makedirs("logs/sessions", exist_ok=True)
file_handler = logging.FileHandler(f"logs/sessions/session_{session_id}.log")
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# Also log to console
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(stream_handler)

# Also write to a fixed shared log (NOT logs/logger)
general_log_path = "logs/app.log"
safe_file_handler = logging.FileHandler(general_log_path, mode='a', encoding='utf-8')
safe_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(safe_file_handler)

# Also log to console
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(stream_handler)

# Public interface
def get_logger(name, level=logging.INFO):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger

def get_log_queue():
    return log_queue
