import logging
import os
from datetime import datetime

def setup_logger(name, log_dir='logs', level=logging.INFO):
    # Ensure log directory exists
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Generate log file name with timestamp
    time_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'{name}_{time_stamp}.log')

    # Set up logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create handlers
    file_handler = logging.FileHandler(log_file)
    stream_handler = logging.StreamHandler()

    # Format for log messages
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger
