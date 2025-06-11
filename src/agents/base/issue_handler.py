import time
import json
import logging
import difflib
import traceback
from typing import Any, Dict

def handle_unicode_emoji_error(agent, task_data: Any, error_info: Dict) -> Any:
    """
    Handles errors caused by emojis or non-ASCII characters.
    Cleans the input by removing non-ASCII characters and retries the task.
    """
    logger = logging.getLogger(f"{agent.name}.UnicodeHandler")
    is_unicode_error = error_info.get("error_type") in ["UnicodeEncodeError", "UnicodeDecodeError"]
    mentions_emoji = "emoji" in error_info.get("error_message", "").lower()
    
    if not (is_unicode_error or mentions_emoji):
        return {"status": "failed", "reason": "Error not identified as unicode/emoji issue"}
    
    cleaned_input = None
    input_changed = False
    
    if isinstance(task_data, str):
        cleaned_version = task_data.encode("ascii", "ignore").decode()
        if cleaned_version != task_data:
            cleaned_input = cleaned_version
            input_changed = True
    elif isinstance(task_data, dict):
        temp_cleaned_data = task_data.copy()
        modified_in_dict = False
        for key, val in temp_cleaned_data.items():
            if isinstance(val, str):
                cleaned_val = val.encode("ascii", "ignore").decode()
                if cleaned_val != val:
                    temp_cleaned_data[key] = cleaned_val
                    modified_in_dict = True
        if modified_in_dict:
            cleaned_input = temp_cleaned_data
            input_changed = True
    
    if input_changed and cleaned_input is not None:
        logger.info(f"Retrying with ASCII-cleaned input")
        try:
            return agent.perform_task(cleaned_input)
        except Exception as e_retry:
            logger.error(f"Retry after cleaning failed: {e_retry}")
            return {"status": "failed", "reason": f"Retry after cleaning failed: {e_retry}"}
    else:
        return {"status": "failed", "reason": "No changes made after cleaning"}

def handle_network_error(agent, task_data: Any, error_info: Dict) -> Any:
    """
    Handles transient network errors with exponential backoff retries.
    """
    logger = logging.getLogger(f"{agent.name}.NetworkHandler")
    error_type = error_info.get("error_type", "")
    
    # Check if it's a network-related error
    network_errors = ["ConnectionError", "Timeout", "socket", "http", "https", "network"]
    if not any(net_err in error_type for net_err in network_errors):
        return {"status": "failed", "reason": "Not a network error"}
    
    max_retries = 3
    base_delay = 1.0  # seconds
    
    for attempt in range(1, max_retries + 1):
        delay = base_delay * (2 ** attempt)  # Exponential backoff
        logger.warning(f"Network error detected. Retrying in {delay:.1f}s (attempt {attempt}/{max_retries})")
        time.sleep(delay)
        
        try:
            return agent.perform_task(task_data)
        except Exception as e:
            logger.error(f"Retry attempt {attempt} failed: {str(e)[:200]}")
            if attempt == max_retries:
                return {"status": "failed", "reason": f"All network retries failed: {str(e)[:200]}"}

def handle_memory_error(agent, task_data: Any, error_info: Dict) -> Any:
    """
    Handles memory-related errors by reducing input size and retrying.
    """
    logger = logging.getLogger(f"{agent.name}.MemoryHandler")
    error_type = error_info.get("error_type", "")
    error_msg = error_info.get("error_message", "").lower()
    
    memory_errors = ["MemoryError", "OutOfMemory", "CUDA out of memory", "resource exhausted"]
    if not any(mem_err in error_type or mem_err in error_msg for mem_err in memory_errors):
        return {"status": "failed", "reason": "Not a memory-related error"}
    
    # Strategy 1: Try reducing input size
    if isinstance(task_data, str) and len(task_data) > 100:
        logger.info("Attempting input reduction for memory error")
        reduced_input = task_data[:len(task_data)//2]
        try:
            return agent.perform_task(reduced_input)
        except Exception as e:
            logger.error(f"Input reduction failed: {str(e)[:200]}")
    
    # Strategy 2: Try freeing up resources
    if hasattr(agent, 'free_up_memory'):
        logger.info("Attempting to free up memory resources")
        try:
            agent.free_up_memory()
            return agent.perform_task(task_data)
        except Exception as e:
            logger.error(f"Memory freeing failed: {str(e)[:200]}")
    
    return {"status": "failed", "reason": "Memory error recovery failed"}

def handle_timeout_error(agent, task_data: Any, error_info: Dict) -> Any:
    """
    Handles timeout errors by simplifying input and retrying.
    """
    logger = logging.getLogger(f"{agent.name}.TimeoutHandler")
    error_type = error_info.get("error_type", "")
    error_msg = error_info.get("error_message", "").lower()
    
    timeout_errors = ["Timeout", "timed out", "Took too long"]
    if not any(timeout_err in error_type or timeout_err in error_msg for timeout_err in timeout_errors):
        return {"status": "failed", "reason": "Not a timeout error"}
    
    # Strategy 1: Simplify complex inputs
    if isinstance(task_data, dict) and len(task_data) > 5:
        logger.info("Simplifying complex input for timeout recovery")
        simplified = {k: v for i, (k, v) in enumerate(task_data.items()) if i < 3}
        try:
            return agent.perform_task(simplified)
        except Exception as e:
            logger.error(f"Input simplification failed: {str(e)[:200]}")
    
    # Strategy 2: Reduce processing requirements
    if hasattr(agent, 'use_lightweight_mode'):
        logger.info("Switching to lightweight processing mode")
        try:
            agent.use_lightweight_mode(True)
            result = agent.perform_task(task_data)
            agent.use_lightweight_mode(False)  # Restore original mode
            return result
        except Exception as e:
            logger.error(f"Lightweight mode failed: {str(e)[:200]}")
            agent.use_lightweight_mode(False)
    
    return {"status": "failed", "reason": "Timeout recovery failed"}

def handle_runtime_error(agent, task_data: Any, error_info: Dict) -> Any:
    """
    Handles RuntimeError, including shape mismatch in tensor ops.
    """
    logger = logging.getLogger(f"{agent.name}.RuntimeErrorHandler")
    error_msg = error_info.get("error_message", "").lower()

    logger.warning("RuntimeError detected. Attempting minimal recovery...")

    # Detect shape mismatch in matrix multiplication
    if "shapes cannot be multiplied" in error_msg and "mat1" in error_msg and "mat2" in error_msg:
        logger.warning("Detected matrix shape mismatch. Attempting to adjust input dimensions.")

        try:
            # Use agent's built-in reshaping
            if hasattr(agent, "reshape_input_for_model"):
                reshaped_task = agent.reshape_input_for_model(task_data)
                if reshaped_task.get("is_reshaped"):
                    return agent.perform_task(reshaped_task)
        except Exception as e:
            logger.error(f"Reshape-based recovery failed: {e}")

    # Generic retry
    try:
        time.sleep(1)
        return agent.perform_task(task_data)
    except Exception as e:
        logger.error(f"Retry after RuntimeError failed: {e}")

    # Fallback to simplified input
    try:
        if isinstance(task_data, dict) and "data" in task_data:
            simplified = {"data": str(task_data["data"])[:100]}
            logger.info("Trying simplified input fallback.")
            return agent.perform_task(simplified)
    except Exception as e:
        logger.error(f"Simplified fallback also failed: {e}")

    return {"status": "failed", "reason": "RuntimeError recovery attempts failed"}

def handle_common_dependency_error(agent, task_data: Any, error_info: Dict) -> Any:
    """
    Handles common dependency errors (missing modules, version conflicts).
    """
    logger = logging.getLogger(f"{agent.name}.DependencyHandler")
    error_msg = error_info.get("error_message", "").lower()
    
    # Detect common dependency issues
    issues = {
        "no module named": "Missing Python module",
        "cannot import name": "Import error",
        "dll load failed": "Missing DLL/library",
        "version conflict": "Version conflict"
    }
    
    detected_issue = None
    for pattern, issue_type in issues.items():
        if pattern in error_msg:
            detected_issue = issue_type
            break
    
    if not detected_issue:
        return {"status": "failed", "reason": "No recognized dependency issue"}
    
    logger.warning(f"Dependency issue detected: {detected_issue}")
    
    # Strategy: Attempt to use alternative implementation if available
    if hasattr(agent, 'use_alternative_implementation'):
        logger.info("Attempting alternative implementation")
        try:
            agent.use_alternative_implementation(detected_issue)
            result = agent.perform_task(task_data)
            agent.use_alternative_implementation(None)  # Restore default
            return result
        except Exception as e:
            logger.error(f"Alternative implementation failed: {str(e)[:200]}")
            agent.use_alternative_implementation(None)
    
    return {"status": "failed", "reason": f"Dependency issue: {detected_issue}"}

def handle_resource_constraint(agent, task_data: Any, error_info: Dict) -> Any:
    """
    Handles resource constraint errors (CPU, GPU, memory limitations).
    """
    logger = logging.getLogger(f"{agent.name}.ResourceHandler")
    error_msg = error_info.get("error_message", "").lower()
    
    resource_errors = ["resource unavailable", "resource busy", "gpu", "cuda", "cpu", "memory"]
    if not any(res_err in error_msg for res_err in resource_errors):
        return {"status": "failed", "reason": "Not a resource constraint error"}
    
    # Strategy 1: Wait and retry
    for wait_time in [2, 5, 10]:  # Progressive waiting
        logger.info(f"Resource constraint detected. Waiting {wait_time}s before retry")
        time.sleep(wait_time)
        try:
            return agent.perform_task(task_data)
        except Exception as e:
            logger.error(f"Retry after waiting failed: {str(e)[:200]}")
    
    # Strategy 2: Fallback to CPU if GPU is busy
    if "gpu" in error_msg or "cuda" in error_msg:
        if hasattr(agent, 'switch_to_cpu'):
            logger.info("Attempting to switch to CPU execution")
            try:
                agent.switch_to_cpu()
                result = agent.perform_task(task_data)
                agent.switch_to_gpu()  # Restore GPU mode if applicable
                return result
            except Exception as e:
                logger.error(f"CPU fallback failed: {str(e)[:200]}")
    
    return {"status": "failed", "reason": "Resource constraint recovery failed"}

def handle_similar_past_error(agent, task_data: Any, error_info: Dict) -> Any:
    """
    Attempts to resolve current error by applying solutions from similar past errors.
    """
    logger = logging.getLogger(f"{agent.name}.PastErrorHandler")
    error_key = f"errors:{agent.name}"
    current_error = error_info.get("error_message", "")
    
    # Get past errors from shared memory
    past_errors = agent.shared_memory.get(error_key, [])
    if not past_errors:
        return {"status": "failed", "reason": "No past errors available"}
    
    # Find similar past errors
    similar_solutions = []
    for error in reversed(past_errors):
        if error.get("timestamp") == error_info.get("timestamp"):
            continue  # Skip current error
        
        similarity = difflib.SequenceMatcher(
            None, current_error, error.get("error_message", "")
        ).ratio()
        
        if similarity > agent.error_similarity_threshold:
            solution = error.get("solution")
            if solution:
                similar_solutions.append((similarity, solution))
    
    if not similar_solutions:
        return {"status": "failed", "reason": "No similar past errors found"}
    
    # Sort by similarity and try the most similar solution first
    similar_solutions.sort(key=lambda x: x[0], reverse=True)
    
    for similarity, solution in similar_solutions:
        logger.info(f"Applying solution from similar past error (similarity: {similarity:.2f})")
        
        # Solutions can be: 
        # 1. A function to call
        # 2. A configuration change
        # 3. An alternative method to use
        try:
            if callable(solution):
                return solution(agent, task_data)
            elif isinstance(solution, dict):
                # Apply configuration changes
                original_config = agent.config.copy()
                agent.config.update(solution)
                result = agent.perform_task(task_data)
                agent.config = original_config  # Restore original config
                return result
            elif isinstance(solution, str):
                # Could be a command to use alternative method
                if hasattr(agent, solution):
                    method = getattr(agent, solution)
                    return method(task_data)
        except Exception as e:
            logger.error(f"Solution application failed: {str(e)[:200]}")
    
    return {"status": "failed", "reason": "All similar solutions failed"}

# Registry of default issue handlers
DEFAULT_ISSUE_HANDLERS = {
    "RuntimeError": handle_runtime_error,
    "unicode": handle_unicode_emoji_error,
    "emoji": handle_unicode_emoji_error,
    "UnicodeEncodeError": handle_unicode_emoji_error,
    "UnicodeDecodeError": handle_unicode_emoji_error,
    "network": handle_network_error,
    "ConnectionError": handle_network_error,
    "Timeout": handle_timeout_error,
    "memory": handle_memory_error,
    "MemoryError": handle_memory_error,
    "dependency": handle_common_dependency_error,
    "ImportError": handle_common_dependency_error,
    "resource": handle_resource_constraint,
    "past_error": handle_similar_past_error,
    "Exception": handle_similar_past_error  # catch-all
}
