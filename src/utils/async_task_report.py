import asyncio
import threading
import concurrent.futures

executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

def run_in_thread(fn, *args, **kwargs):
    """Run function in background thread."""
    future = executor.submit(fn, *args, **kwargs)
    return future

async def run_async(fn, *args, **kwargs):
    """Run sync function asynchronously (non-blocking)."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, fn, *args, **kwargs)
