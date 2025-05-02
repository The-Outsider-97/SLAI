
import multiprocessing as mp

from typing import Any, List, Callable
from concurrent.futures import ProcessPoolExecutor

# ---------------------------
# Type Definitions
# ---------------------------

Vector = List[float]

# ---------------------------
# Parallel Data Processing
# ---------------------------

class ParallelProcessor:
    """MPI-style parallel processing utilities"""
    
    @staticmethod
    def map_reduce(
        data: List[Any],
        map_fn: Callable,
        reduce_fn: Callable,
        chunk_size: int = 1000
    ) -> Any:
        """Parallel MapReduce implementation"""
        with mp.Pool() as pool:
            chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
            mapped = pool.map(map_fn, chunks)
            return reduce_fn(mapped)

    @staticmethod
    def featurewise_parallel(
        X: List[Vector],
        func: Callable[[Vector], Any]
    ) -> List[Any]:
        """Column-wise parallel processing of data matrices"""
        with ProcessPoolExecutor() as executor:
            return list(executor.map(func, zip(*X)))
