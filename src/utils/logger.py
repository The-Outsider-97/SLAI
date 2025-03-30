import logging
import tempfile
import subprocess
import os
import time
import ast

from datetime import datetime
from logging.handlers import RotatingFileHandler

def get_logger(name):
    """
    Return an existing logger or create a new one with default DEBUG level and single file output.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        os.makedirs("logs", exist_ok=True)
        file_handler = logging.FileHandler(f"logs/{name}.log")
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

def setup_logger(name, level=logging.INFO):
    """
    Setup a logger with rotating file and stream output.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        os.makedirs("logs", exist_ok=True)

        # Rotating file handler
        rotating_file = RotatingFileHandler("logs/app.log", maxBytes=5_000_000, backupCount=2)
        rotating_file.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
        rotating_file.setLevel(level)
        logger.addHandler(rotating_file)

        # Timestamped log file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_handler = logging.FileHandler(f"logs/{name}_{timestamp}.log")
        file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
        logger.addHandler(file_handler)

        # Stream (console) handler
        stream = logging.StreamHandler()
        stream.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream)

    return logger

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
