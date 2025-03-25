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

# Public interface
def get_logger():
    return logger

def get_log_queue():
    return log_queue
