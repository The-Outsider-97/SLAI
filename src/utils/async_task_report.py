import asyncio
import threading
import concurrent.futures
import time
import random
from typing import Callable, Any

executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

def run_in_thread(
    fn: Callable,
    *args,
    callback: Callable[[Any], None] = None,
    error_callback: Callable[[Exception], None] = None,
    **kwargs
) -> concurrent.futures.Future:
    """
    Enhanced thread runner with completion/error callbacks
    and execution metadata
    """
    def _wrapper():
        try:
            result = fn(*args, **kwargs)
            if callback:
                callback(result)
            return result
        except Exception as e:
            if error_callback:
                error_callback(e)
            raise

    future = executor.submit(_wrapper)
    future.add_done_callback(
        lambda f: print(f"Thread task completed: {f.result() if not f.exception() else f.exception()}")
    )
    return future

async def run_async(
    fn: Callable,
    *args,
    timeout: float = None,
    retries: int = 1,
    **kwargs
) -> Any:
    """
    Robust async runner with timeout, retries, and progress reporting
    """
    attempt = 0
    last_exception = None
    
    while attempt < retries:
        try:
            return await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, fn, *args, **kwargs
                ),
                timeout=timeout
            )
        except Exception as e:
            last_exception = e
            attempt += 1
            print(f"Attempt {attempt} failed: {str(e)}")
            if attempt < retries:
                await asyncio.sleep(1 * attempt)  # Exponential backoff
    
    raise last_exception if last_exception else RuntimeError("Unknown error")

# Example operations
def cpu_intensive_task(n: int) -> int:
    """Simulate CPU-bound work"""
    print(f"CPU task started with {n}")
    time.sleep(1)
    if random.random() < 0.3:
        raise ValueError("Random CPU task failure")
    return n * n

async def io_bound_task(url: str) -> str:
    """Simulate IO-bound work"""
    print(f"Fetching {url}")
    await asyncio.sleep(0.5)
    if random.random() < 0.3:
        raise ConnectionError("Random IO task failure")
    return f"Content from {url}"

if __name__ == "__main__":
    # Test thread-based execution
    print("=== Testing Thread Execution ===")
    thread_future = run_in_thread(
        cpu_intensive_task, 5,
        callback=lambda res: print(f"Thread callback received: {res}"),
        error_callback=lambda e: print(f"Thread error: {str(e)}")
    )
    
    # Test async execution
    async def main_async():
        print("\n=== Testing Async Execution ===")
        try:
            result = await run_async(
                cpu_intensive_task, 8,
                timeout=2,
                retries=3
            )
            print(f"Async result: {result}")
        except Exception as e:
            print(f"Final async failure: {str(e)}")
        
        print("\n=== Testing Async with Coroutine ===")
        try:
            result = await io_bound_task("https://example.com")
            print(f"IO task result: {result}")
        except Exception as e:
            print(f"IO task failed: {str(e)}")

    # Run all tests
    def mixed_operations():
        """Combine thread and async operations"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Wait for thread task
        thread_future.result()
        
        # Run async tasks
        loop.run_until_complete(main_async())
        loop.close()

    mixed_operations()
    
    # Cleanup
    executor.shutdown(wait=True)
    print("\nAll tasks completed")
