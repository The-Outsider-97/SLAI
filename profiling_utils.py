import time
import tracemalloc
import logging
import functools
import gc
import psutil
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Profiler")

def memory_profile(func):
    """Decorator to measure memory usage of a function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        gc.collect()
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 ** 2  # MB
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        gc.collect()
        mem_after = process.memory_info().rss / 1024 ** 2  # MB
        logger.info(f"[MEMORY] {func.__name__} used {mem_after - mem_before:.2f} MB in {end - start:.2f}s")
        return result
    return wrapper

def time_profile(func):
    """Decorator to measure execution time of a function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        logger.info(f"[TIME] {func.__name__} executed in {end - start:.4f} seconds")
        return result
    return wrapper

def start_memory_tracing():
    """Start tracemalloc memory tracing."""
    tracemalloc.start()
    logger.info("[MEMORY] Tracing memory allocation started")

def display_top_memory_sources(limit=10):
    """Display top memory usage sources."""
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')
    logger.info("[MEMORY] Top memory usage sources:")
    for stat in top_stats[:limit]:
        logger.info(stat)
