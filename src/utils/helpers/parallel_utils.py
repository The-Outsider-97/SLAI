import multiprocessing as mp
from typing import Any, List, Callable
from concurrent.futures import ProcessPoolExecutor

# ---------------------------
# Type Definitions
# ---------------------------

Vector = List[float]  # Assuming this is List[float] as in the original
Matrix = List[List[float]] # Or List[Vector]

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
        # Ensure the main process waits for the pool to finish if run in a script
        # This can be important for __main__ context issues on some OSes.
        # However, ProcessPoolExecutor handles this better.
        # For mp.Pool, ensure it's guarded by if __name__ == '__main__': if used at module level.
        with mp.Pool() as pool:
            chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
            mapped = pool.map(map_fn, chunks)
            return reduce_fn(mapped)

    @staticmethod
    def featurewise_parallel(X: List[Vector], func: Callable) -> List[Any]:
        """
        Processes features (columns of X) in parallel.
        Assumes func is picklable (e.g., top-level or importable).
        """
        # Removed dill.loads(dill.dumps(func)) as it was not standard usage
        # and pickling issues are better solved by making 'func' picklable.
        with ProcessPoolExecutor() as executor:
            # zip(*X) transposes X, so func is applied to each column-vector
            return list(executor.map(func, zip(*X)))

    @staticmethod
    def general_parallel_map(func: Callable, args_list: List[Any]) -> List[Any]:
        """
        A general-purpose parallel map that applies func to each item in args_list.
        'func' must be picklable (e.g., a top-level function).
        Each element in 'args_list' is passed as a single argument to 'func'.
        """
        with ProcessPoolExecutor() as executor:
            results = list(executor.map(func, args_list))
        return results
