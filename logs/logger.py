import os
import sys
import torch
import logging
import intertools
import random
import queue
import json

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

# Optional: also log to file or console
file_handler = logging.FileHandler("logs/slai.log")
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(stream_handler)

# Utility access
get_logger = lambda: logger
get_log_queue = lambda: log_queue
