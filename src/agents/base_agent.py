__version__ = "1.9.0"

import re
import os, sys
import abc
import time
import json
import torch
import difflib
import traceback
import numpy as np
import torch.nn as nn

from typing import Any
from collections import OrderedDict, defaultdict, deque

from src.agents.base.utils.main_config_loader import load_global_config, get_config_section
from src.agents.base.lazy_agent import LazyAgent
from src.agents.base.light_metric_store import LightMetricStore
from src.agents.base.issue_handler import DEFAULT_ISSUE_HANDLERS
from src.agents.collaborative.shared_memory import SharedMemory
from logs.logger import get_logger

logger = get_logger("SLAI Base Agent")

class BaseAgent(abc.ABC):
    def __init__(self, shared_memory, agent_factory, config=None):
        self.logger = get_logger(self.__class__.__name__)
        self.name = self.__class__.__name__
        
        if shared_memory is None:
            shared_memory = SharedMemory()
        self.shared_memory = shared_memory
        self.agent_factory=agent_factory

        self.config = get_config_section('base_agent')
        self.defer_initialization = self.config.get('defer_initialization')
        self.memory_profile = self.config.get('memory_profile')
        self.network_compression = self.config.get('network_compression')
        self.max_error_log_size = self.config.get('max_error_log_size')
        self.error_similarity_threshold = self.config.get('error_similarity_threshold')
        self.max_task_retries = self.config.get('max_task_retries')
        self.task_similarity_str_threshold = self.config.get('task_similarity_str_threshold')
        self.jaccard_threshold = self.config.get('jaccard_threshold')
        self.jaccard_min_for_no_shared = self.config.get('jaccard_min_for_no_shared')
        self.final_key_threshold = self.config.get('final_key_threshold')
        self.final_value_threshold = self.config.get('final_value_threshold')
        self.task_similarity_seq_elem_threshold = self.config.get('task_similarity_seq_elem_threshold')
        self.task_timeout_seconds = None
        self.current_plan = []
        self.current_goal = []

        self.metric_store = LightMetricStore()
        
        # Initialize lazy components system
        self._component_initializers = {
            'performance_metrics': lambda: defaultdict(
                lambda: deque(maxlen=self._get_metric_buffer_size())
            )
        }
        self._lazy_components = OrderedDict()

        self.retraining_thresholds = {} # Populated by subclasses based on their specific metrics
        self.evaluation_log_dir = "evaluation_logs"
        os.makedirs(self.evaluation_log_dir, exist_ok=True)

        self._known_issue_handlers = {}
        self.register_default_known_issue_handlers()
        self._init_core_components()

    def _get_metric_buffer_size(self):
        """Determine metric buffer size based on memory profile"""
        if self.memory_profile == 'low':
            return 100
        elif self.memory_profile == 'medium':
            return 500
        else:  # 'high' or default
            return 1000

    def _init_core_components(self):
        """Initialize essential components first with compression setting"""
        # Create LazyAgent instance for expensive components
        compression_enabled = self.network_compression
        self.lazy_agent = LazyAgent(
            lambda: self._create_expensive_components(compression_enabled))
        
    def _create_expensive_components(self, compression_enabled=False):
        """Create components that should be lazily initialized"""
        return {
            'performance_metrics': defaultdict(
                lambda: deque(maxlen=self._get_metric_buffer_size()))
        }

    def register_default_known_issue_handlers(self):
        """Registers common known issue handlers. Subclasses can add more."""
        for pattern, handler in DEFAULT_ISSUE_HANDLERS.items():
            self.register_known_issue_handler(pattern, handler)

    @property
    def performance_metrics(self):
        return self.lazy_property('performance_metrics')

    def lazy_property(self, name):
        """Get or create a lazy-initialized component"""
        if name not in self._lazy_components:
            if name in self._component_initializers:
                self._lazy_components[name] = self._component_initializers[name]()
            else:
                raise AttributeError(f"No initializer for lazy component: {name}")
        return self._lazy_components[name]

    def _log_error_to_shared_memory(self, error_entry: dict):
        """Logs an error entry to shared memory, managing log size."""
        error_key = f"errors:{self.name}"
        # Ensure shared_memory.get/set are thread-safe if shared_memory is used across threads
        errors = self.shared_memory.get(error_key) or []
        errors.append(error_entry)
        if len(errors) > self.max_error_log_size:
            errors = errors[-self.max_error_log_size:]
        self.shared_memory.set(error_key, errors)

    def _check_and_log_similar_errors(self, new_error_info: dict) -> bool:
        """
        Checks for and logs similarity with past errors from shared memory.
        Returns True if a similar error was found, False otherwise.
        """
        error_key = f"errors:{self.name}"
        # Fetch errors AFTER the current one has been logged to include it in the list for subsequent calls,
        # but for current check, we compare against errors *before* the current one.
        all_errors = self.shared_memory.get(error_key) or [] 
        
        # If all_errors includes the new_error_info already (it should if _log_error_to_shared_memory was called before this)
        # then we compare against errors[:-1]. If not, then all_errors is the history.
        # Assuming new_error_info is the LATEST error, so compare against history excluding it.
        history_errors = [e for e in all_errors if e['timestamp'] < new_error_info['timestamp']]


        new_error_msg = new_error_info.get("error_message", "")
        new_error_type = new_error_info.get("error_type", "")

        for past_error_info in reversed(history_errors): 
            past_error_msg = past_error_info.get("error_message", "")
            past_error_type = past_error_info.get("error_type", "")

            if past_error_type == new_error_type: # First, match error type
                similarity = difflib.SequenceMatcher(None, new_error_msg, past_error_msg).ratio()
                if similarity >= self.error_similarity_threshold:
                    self.logger.warning(
                        f"[{self.name}] A similar {new_error_type} occurred previously "
                        f"(message similarity: {similarity:.2f}). Current error: '{new_error_msg[:100]}...'"
                    )
                    return True # Found a similar error
        return False # No significantly similar error found in history

    def execute(self, input_data):
        """
        Executes the agent's task with comprehensive error handling and recovery.
        Flow:
        1. Try `perform_task` (with retries if configured).
        2. If error, log it and check for similarity with past errors.
        3. Try `handle_known_issue` based on error information.
        4. If still unresolved, try `alternative_execute`.
        5. If all fail, log final failure.
        """
        retries_done = 0
        last_exception_in_retry_loop = None
        if self.defer_initialization and not hasattr(self, 'lazy_agent'):
            self.logger.info(f"[{self.name}] Initializing deferred components")
            self._init_core_components()

        self.metric_store.start_tracking('execute', 'performance')
        try:
            # --- STAGE 1: Perform Task (with retries) ---
            while retries_done <= self.max_task_retries:
                try:
                    self.logger.info(f"[{self.name}] Attempting task (attempt {retries_done + 1}/{self.max_task_retries + 1})...")
                    # Note: Robust timeout for self.perform_task is complex (threading, signals, or async).
                    # If self.task_timeout_seconds is set, subclasses might need to handle it internally
                    # or a more sophisticated execution wrapper would be needed here.
                    result = self.perform_task(input_data) # perform_task is now abstract

                    if hasattr(self, "evaluate_performance") and callable(self.evaluate_performance):
                        performance_data = self.extract_performance_metrics(result)
                        if performance_data: # Only evaluate if metrics are meaningfully extracted
                            self.evaluate_performance(performance_data)
                    
                    self.logger.info(f"[{self.name}] Task execution successful on attempt {retries_done + 1}.")
                    self.shared_memory.set(f"agent_stats:{self.name}", {
                        "last_run": time.time(), "success": True, 
                        "attempts": retries_done + 1,
                        "result_summary": str(result)[:200]
                    })
                    return result
                
                except Exception as e:
                    last_exception_in_retry_loop = e
                    if retries_done < self.max_task_retries:
                        retries_done += 1
                        self.logger.warning(f"[{self.name}] Task attempt {retries_done} failed. Retrying... Error: {e}")
                        time.sleep(min(retries_done * 0.5, 5)) # Simple backoff, capped at 5s
                    else: # Max retries reached
                        self.logger.error(f"[{self.name}] Task failed after {retries_done + 1} attempts. Last error: {e}")
                        break # Exit retry loop to proceed with further error handling stages

            # --- STAGE 2: Log Initial Error and Check Similarity ---
            # This stage is reached only if all retries in Stage 1 failed.
            error_info = {
                "timestamp": time.time(),
                "error_type": type(last_exception_in_retry_loop).__name__,
                "error_message": str(last_exception_in_retry_loop),
                "traceback": traceback.format_exc()
            }
            self._log_error_to_shared_memory(error_info) # Log the definitive error from perform_task attempts
            _ = self._check_and_log_similar_errors(error_info) # Log if similar error found, result not critical for flow here

            # --- STAGE 3: Handle Known Issue ---
            try:
                self.logger.info(f"[{self.name}] Attempting to handle as known issue: {error_info['error_type']}...")
                recovery_result = self.handle_known_issue(input_data, error_info) # Pass original input_data
                
                # Check if handler successfully resolved it (i.e., did not return a failure dict)
                if not (isinstance(recovery_result, dict) and recovery_result.get("status") == "failed"):
                    self.logger.info(f"[{self.name}] Task recovered by 'handle_known_issue'.")
                    self.shared_memory.set(f"agent_stats:{self.name}", {
                        "last_run": time.time(), "success": True, "recovered_by": "handle_known_issue",
                        "attempts": retries_done + 1, "result_summary": str(recovery_result)[:200]
                    })
                    return recovery_result # Success via known issue handler
                else:
                    self.logger.info(f"[{self.name}] 'handle_known_issue' did not resolve the issue: {recovery_result.get('reason')}")
            except Exception as e_known_issue_handler_crash: # If handle_known_issue itself crashes
                self.logger.error(f"[{self.name}] The 'handle_known_issue' method itself crashed: {e_known_issue_handler_crash}")
                self.logger.debug(traceback.format_exc())
                # Proceed to alternative_execute

            # --- STAGE 4: Alternative Execution ---
            try:
                self.logger.info(f"[{self.name}] Attempting alternative execution strategy...")
                alternative_result = self.alternative_execute(input_data, original_error=last_exception_in_retry_loop)
                
                # Check if alternative_execute signaled a failure (e.g., returned specific string or dict)
                is_alt_exec_failure = False
                if isinstance(alternative_result, str) and "[Fallback failure]" in alternative_result:
                    is_alt_exec_failure = True
                elif isinstance(alternative_result, dict) and alternative_result.get("status") == "failed":
                    is_alt_exec_failure = True

                if not is_alt_exec_failure:
                    self.logger.info(f"[{self.name}] Task processed by 'alternative_execute'.")
                    self.shared_memory.set(f"agent_stats:{self.name}", {
                        "last_run": time.time(), "success": True, "recovered_by": "alternative_execute",
                        "attempts": retries_done + 1, "result_summary": str(alternative_result)[:200]
                    })
                    return alternative_result # Success via alternative execution
                else:
                    self.logger.warning(f"[{self.name}] 'alternative_execute' indicated failure or could not process.")
            except Exception as e_alternative_handler_crash: # If alternative_execute itself crashes
                self.logger.error(f"[{self.name}] The 'alternative_execute' method itself crashed: {e_alternative_handler_crash}")
                self.logger.debug(traceback.format_exc())
                # Proceed to final failure reporting

            # --- STAGE 5: Final Failure ---
            final_error_message = f"All task execution and recovery attempts failed. Original error after retries: {str(last_exception_in_retry_loop)}"
            self.logger.error(f"[{self.name}] {final_error_message}")
            self.shared_memory.set(f"agent_stats:{self.name}", {
                "last_run": time.time(), "success": False, 
                "error": str(last_exception_in_retry_loop), "recovery_failed": True,
                "attempts": retries_done + 1
            })
            return {"status": "failed", "error": str(last_exception_in_retry_loop), "reason": "All recovery attempts failed."}
        finally:
            self.metric_store.stop_tracking('execute', 'performance')

    def register_known_issue_handler(self, issue_pattern_or_id: str, handler_func: callable):
        """
        Registers a handler for a known issue. 
        The pattern can be a string to check in error messages/types.
        """
        if not callable(handler_func):
            raise ValueError(f"Handler function for '{issue_pattern_or_id}' must be callable.")
        self._known_issue_handlers[issue_pattern_or_id] = handler_func
        logger.debug(f"[{self.name}] Registered known issue handler for '{issue_pattern_or_id}' using '{handler_func.__name__}'.")

    def handle_known_issue(self, task_data, error_info: dict):
        """
        Attempts to recover from known failure patterns by dispatching to registered handlers.
        Handlers should return the processed result upon success, or a dict with 
        {"status": "failed", "reason": "..."} if they cannot handle this specific instance or their attempt fails.
        If a handler itself raises an unhandled exception, it's caught here.
        """
        logger.debug(f"[{self.name}] Checking known issue handlers for error: type='{error_info.get('error_type')}', msg='{error_info.get('error_message', '')[:100]}...'")
        error_message_lower = error_info.get("error_message", "").lower()
        error_type_lower = error_info.get("error_type", "").lower()

        for issue_id_pattern, handler_func in self._known_issue_handlers.items():
            # Match if issue_id_pattern (case-insensitive) is in error message or type
            pattern_lower = issue_id_pattern.lower()
            if pattern_lower in error_message_lower or pattern_lower == error_type_lower: # Exact match for type, substring for message
                logger.info(f"[{self.name}] Potential known issue match with pattern '{issue_id_pattern}'. Attempting handler '{handler_func.__name__}'.")
                try:
                    result = handler_func(self, task_data, error_info) # Handler attempts to resolve
                    
                    # If handler explicitly returns a dict with "status": "failed", it means it matched but couldn't fix THIS instance.
                    if isinstance(result, dict) and result.get("status") == "failed":
                        logger.info(f"[{self.name}] Handler '{handler_func.__name__}' for '{issue_id_pattern}' reported it could not resolve: {result.get('reason')}")
                        # In this design, if a specific handler matches and explicitly fails, we stop and report this failure.
                        # Alternative: `continue` to try other handlers if the match was weak or handler was too specific.
                        return result 
                    else: # Handler succeeded (returned something not a failure dict)
                        logger.info(f"[{self.name}] Known issue '{issue_id_pattern}' handled successfully by '{handler_func.__name__}'.")
                        return result 
                except Exception as handler_ex: # Handler itself crashed
                    logger.error(f"[{self.name}] Handler '{handler_func.__name__}' for known issue '{issue_id_pattern}' raised an exception: {handler_ex}")
                    logger.debug(traceback.format_exc())
                    return {"status": "failed", "reason": f"Handler for '{issue_id_pattern}' crashed: {handler_ex}"}
        
        logger.info(f"[{self.name}] No specific known issue handler matched or resolved the error.")
        return {"status": "failed", "reason": "No applicable or successful known issue handler found."}    
    
    def alternative_execute(self, task_data, original_error=None):
        """
        Fallback logic without LLM dependencies. Uses:
        1. Replanning (for planning agents)
        2. Input sanitization/simplification
        3. Rule-based grammar processing
        4. Echo strategy
        """
        logger.info(f"[{self.name}] Entering alternative execution (error: {str(original_error)[:100]}...)")
    
        # Strategy 1: Replanning for planning agents
        if "PlanningAgent" in self.name and hasattr(self, 'replan') and callable(self.replan):
            if hasattr(self, 'current_goal') and self.current_goal:
                try:
                    logger.info(f"[{self.name}] Alt exec: Attempting replanning")
                    new_plan = self.replan(self.current_goal)
                    if new_plan:
                        return self.execute_plan(new_plan)
                except Exception as e:
                    logger.error(f"Replanning failed: {str(e)}")
        
        # Strategy 2: Input sanitization
        sanitized_input = self.sanitize_input(task_data)
        logger.debug(f"[{self.name}] Sanitized input: {sanitized_input[:100]}...")
        
        # Strategy 3: Rule-based processing
        rule_based_result = self.apply_rule_based_processing(sanitized_input)
        if rule_based_result:
            return rule_based_result
        
        # Strategy 4: Echo strategy
        return self.echo_strategy(sanitized_input)
    
    def sanitize_input(self, task_data):
        """Generic input sanitization without LLM dependencies"""
        if isinstance(task_data, str):
            return task_data.strip().replace('\n', ' ').replace('\r', '')[:500]
        
        if isinstance(task_data, dict):
            # Extract possible content fields
            content_keys = ['text', 'query', 'input', 'message', 'data']
            for key in content_keys:
                if key in task_data and isinstance(task_data[key], str):
                    return task_data[key].strip()[:500]
            
            # Fallback to JSON stringification
            try:
                return json.dumps({k: v for k, v in task_data.items() if isinstance(v, (str, int, float))})[:500]
            except:
                return str(task_data)[:500]
        
        return str(task_data)[:500]
    
    def apply_rule_based_processing(self, sanitized_input):
        """Rule-based fallback using predefined patterns"""
        # Common error patterns and responses
        error_patterns = {
            r"connection (timeout|refused)": "Network unavailable. Please check connectivity",
            r"invalid (format|input)": "Input format error. Please verify your request structure",
            r"out of memory": "Resource constraints detected. Try reducing request complexity",
            r"authentication failed": "Credentials invalid or expired. Please re-authenticate"
        }
        original_error=None
        
        for pattern, response in error_patterns.items():
            if re.search(pattern, sanitized_input, re.IGNORECASE):
                logger.info(f"[{self.name}] Rule-based match: {pattern}")
                return f"[Fallback] {response}"
        
        # Try grammar processor if available
        if hasattr(self, 'grammar') and callable(getattr(self.grammar, "compose_sentence", None)):
            try:
                facts = {"input": sanitized_input[:100], "error": str(original_error)[:100]}
                return f"[Grammar Response] {self.grammar.compose_sentence(facts)}"
            except Exception as e:
                logger.warning(f"Grammar processing failed: {str(e)}")
        
        return None

    def echo_strategy(self, sanitized_input):
        """Final fallback strategy"""
        if sanitized_input:
            return f"[Fallback] Processed input: {sanitized_input}"
        return "[Fallback] Unable to process request through alternative methods"

    def is_similar(self, task_data_1: Any, task_data_2: Any) -> bool:
        """
        Compares two task_data inputs for similarity.
        Uses type checking, string equality (with fuzzy matching), and dictionary heuristics.
        """
        if type(task_data_1) != type(task_data_2):
            return False

        # Handle simple text-based tasks
        if isinstance(task_data_1, str): # type(task_data_2) is also str due to above check
            s1 = task_data_1.strip().lower()
            s2 = task_data_2.strip().lower()
            if not s1 and not s2: return True # Both empty strings are similar
            if not s1 or not s2: return False # One empty, one not
            # Using SequenceMatcher for string similarity, threshold can be configured
            return difflib.SequenceMatcher(None, s1, s2).ratio() > self.task_similarity_str_threshold

        # Handle dict-based structured tasks
        if isinstance(task_data_1, dict): # task_data_2 is also dict
            keys1 = set(task_data_1.keys())
            keys2 = set(task_data_2.keys())
            
            if not keys1 and not keys2: return True # Both empty dicts
            if not keys1 or not keys2: return False # One empty

            # Jaccard similarity for keys
            key_intersection = len(keys1.intersection(keys2))
            key_union = len(keys1.union(keys2))
            key_jaccard_sim = key_intersection / key_union if key_union > 0 else 0.0

            # If key structure is too different, consider them not similar
            if key_jaccard_sim < self.jaccard_threshold:
                return False

            shared_keys = keys1.intersection(keys2)
            if not shared_keys: # No common keys, but Jaccard might be high if key sets are small subsets
                 return key_jaccard_sim > self.jaccard_min_for_no_shared

            value_similarity_scores = []
            for key in shared_keys:
                val1, val2 = task_data_1[key], task_data_2[key]
                # Recursively call is_similar for nested structures, or direct comparison
                if isinstance(val1, (dict, str, list, tuple)): # Types that is_similar handles well
                    value_similarity_scores.append(1.0 if self.is_similar(val1, val2) else 0.0)
                elif type(val1) == type(val2) and isinstance(val1, (int, float, bool)):
                     value_similarity_scores.append(1.0 if val1 == val2 else 0.0) # Simple equality for primitives
                else: # Different types or unhandled complex types for values
                    value_similarity_scores.append(0.0) 
            
            if not value_similarity_scores: # Should not happen if shared_keys is non-empty
                return key_jaccard_sim > self.jaccard_min_for_no_shared

            avg_value_similarity = sum(value_similarity_scores) / len(value_similarity_scores)
            
            # Combine key similarity and value similarity
            # Example: require high key similarity AND high average value similarity
            return (key_jaccard_sim >= self.final_key_threshold and 
                    avg_value_similarity >= self.final_value_threshold)

        # Handle list/tuple based tasks
        if isinstance(task_data_1, (list, tuple)): # task_data_2 also same type
            if len(task_data_1) != len(task_data_2): return False
            if not task_data_1: return True # Both empty sequences

            # Compare elements using is_similar for robustness
            element_similarities = []
            for item1, item2 in zip(task_data_1, task_data_2):
                element_similarities.append(1.0 if self.is_similar(item1, item2) else 0.0)
            
            if not element_similarities: return True # Should not happen if len > 0
            avg_element_similarity = sum(element_similarities) / len(element_similarities)
            return avg_element_similarity >= self.task_similarity_seq_elem_threshold

        # Fallback for other directly comparable types (bool, NoneType, numbers if not caught by dict logic)
        return task_data_1 == task_data_2

    def extract_performance_metrics(self, result: Any) -> dict:
        """
        Base implementation. Subclasses should override this to extract
        meaningful performance metrics from their specific `perform_task` result.
        """
        metrics = {}
        # Example: if result is a dict and contains common metric keys
        if isinstance(result, dict):
            for key in ['accuracy', 'precision', 'recall', 'f1_score', 'latency_ms', 'throughput']:
                if key in result and isinstance(result[key], (int, float)):
                    metrics[key] = result[key]
        # This base version is very generic. Subclasses are expected to provide specific extraction.
        if not metrics:
            self.logger.debug(f"[{self.name}] No standard performance metrics extracted from result of type {type(result)}.")
        return metrics

#    @abc.abstractmethod
    def perform_task(self, task_data: Any) -> Any:
        """Overrides BaseAgent.perform_task to handle planning-specific commands."""
        self.logger.info(f"Executing planning task with data: {task_data}")

        if not isinstance(task_data, dict):
            raise ValueError("Unsupported input format for PlanningAgent, expected a dictionary.")

        command = task_data.get("command")

        if command == "execute_plan":
            # This allows the training loop's call to work without crashing,
            # executing whatever plan is currently loaded in the agent.
            if not self.current_plan:
                self.logger.warning("Command 'execute_plan' received, but no current plan exists.")
                return {"status": "AWAITING_PLAN", "message": "No plan to execute."}
            return self.execute_plan(self.current_plan, self.current_goal)
        else:
            raise ValueError(f"Unsupported command for PlanningAgent: '{command}'")

    def execute_plan(self, plan: Any, goal: Any = None) -> Any:
        """
        Generic plan execution framework with:
        - Progress tracking
        - Error handling
        - Resource monitoring
        - Result validation
        
        Args:
            plan: Iterable of steps (dicts, objects, or functions)
            goal: Optional target state/objective
        
        Returns:
            Execution result with status metadata
        """
        # Validate plan structure
        if not hasattr(plan, '__iter__'):
            return {"status": "error", "reason": "Invalid plan format"}
        
        results = []
        step_context = {}
        resource_monitor = ResourceMonitor()
        
        try:
            # Pre-execution validation
            if not self.validate_plan(plan):
                return {"status": "error", "reason": "Plan validation failed"}
            
            for i, step in enumerate(plan):
                step_name = step.get('name', f"step_{i+1}")
                logger.info(f"[{self.name}] Executing {step_name}...")
                
                # Monitor resources before execution
                if resource_monitor.is_critical():
                    return {"status": "interrupted", "reason": "Resource limits exceeded"}
                
                # Execute step with error handling
                try:
                    result = self.execute_step(step, step_context)
                    results.append(result)
                    
                    # Update context for next steps
                    if isinstance(result, dict):
                        step_context.update(result.get('context', {}))
                    
                    # Validate intermediate results
                    if goal and self.check_goal_achieved(step_context, goal):
                        logger.info(f"[{self.name}] Early goal achievement at step {i+1}")
                        return self.compile_results(results, True)
                
                except Exception as e:
                    logger.error(f"Step {step_name} failed: {str(e)}")
                    results.append({
                        "step": step_name,
                        "status": "error",
                        "error": str(e)
                    })
                    
                    # Attempt step recovery
                    if not self.recover_step(step, step_context):
                        return self.compile_results(results, False)
            
            return self.compile_results(results, True)
        
        finally:
            # Post-execution cleanup
            self.cleanup_resources()
            logger.info(f"[{self.name}] Plan execution completed")

    def validate_plan(self, plan):
        """Basic plan validation logic"""
        required_attrs = ['name', 'handler'] if isinstance(plan[0], dict) else None
        for step in plan:
            if required_attrs:
                if not all(attr in step for attr in required_attrs):
                    return False
        return True

    def execute_step(self, step, context):
        """Flexible step execution handling multiple formats"""
        # Function-based step
        if callable(step):
            return step(context)
        
        # Dict-based step
        if isinstance(step, dict):
            handler_name = step.get('handler')
            if handler_name and hasattr(self, handler_name):
                handler = getattr(self, handler_name)
                return handler(step.get('params', {}), context)
            
            # Direct module/action specification
            if 'module' in step and 'action' in step:
                return self.call_external(
                    step['module'],
                    step['action'],
                    step.get('params', {})
                )
        
        # String-based command
        if isinstance(step, str):
            return self.process_command(step, context)
        
        raise ValueError(f"Unsupported step type: {type(step)}")

    def check_goal_achieved(self, step_context: dict, goal: Any) -> bool:
        """
        Checks if the current context indicates the goal has been achieved.
        Default implementation requires exact match of goal keys in context.
        Subclasses should override with domain-specific goal checking.
        """
        if isinstance(goal, dict):
            # Check if all goal key-value pairs exist in context
            for key, target_value in goal.items():
                if key not in step_context:
                    return False
                current_value = step_context[key]
                if isinstance(target_value, (int, float)):
                    # Numeric comparison with tolerance
                    if abs(current_value - target_value) > 1e-6:
                        return False
                elif current_value != target_value:
                    return False
            return True
        
        # For simple types, direct comparison
        return step_context.get('result') == goal

    def compile_results(self, results, success):
        """Standardized result compilation"""
        return {
            "status": "success" if success else "partial_success",
            "steps_executed": len(results),
            "success_rate": sum(1 for r in results if r.get('status') == 'success') / len(results),
            "results": results,
            "timestamp": time.time()
        }
    
    def recover_step(self, step: Any, step_context: dict) -> bool:
        """
        Attempts to recover from a failed step execution.
        Returns True if recovery was successful, False otherwise.
        """
        step_name = step.get('name', "unknown_step")
        logger.warning(f"[{self.name}] Attempting recovery for failed step: {step_name}")
        
        # Strategy 1: Check if step has a recovery handler
        if isinstance(step, dict) and 'recovery_handler' in step:
            handler_name = step['recovery_handler']
            if hasattr(self, handler_name):
                try:
                    recovery_func = getattr(self, handler_name)
                    recovery_result = recovery_func(step, step_context)
                    if recovery_result:
                        logger.info(f"[{self.name}] Step recovery successful using handler '{handler_name}'")
                        return True
                except Exception as e:
                    logger.error(f"Recovery handler '{handler_name}' failed: {str(e)}")
        
        # Strategy 2: Default retry
        try:
            logger.info(f"[{self.name}] Attempting step retry...")
            self.execute_step(step, step_context)
            logger.info(f"[{self.name}] Step recovery successful through retry")
            return True
        except Exception as e:
            logger.error(f"Step retry failed: {str(e)}")
            return False
        
    def cleanup_resources(self):
        """
        Releases any resources acquired during plan execution.
        Base implementation handles common resource types.
        """
        logger.info(f"[{self.name}] Cleaning up resources...")
        
        # Close file handles if any
        if hasattr(self, 'open_files'):
            for file in self.open_files:
                try:
                    file.close()
                except Exception as e:
                    logger.warning(f"Error closing file: {str(e)}")
            self.open_files = []
        
        # Release network connections
        if hasattr(self, 'active_connections'):
            for conn in self.active_connections:
                try:
                    conn.close()
                except Exception as e:
                    logger.warning(f"Error closing connection: {str(e)}")
            self.active_connections = []
        
        # Clear temporary data
        if hasattr(self, 'temporary_data'):
            self.temporary_data.clear()

    def call_external(self, module: str, action: str, params: dict) -> Any:
        """
        Calls an external module/action with parameters.
        Uses agent_factory to locate and execute the module.
        """
        logger.info(f"[{self.name}] Calling external: {module}.{action}()")
        
        try:
            # Get module agent from factory
            module_agent = self.agent_factory.create(module)
            
            # Get the action method
            if not hasattr(module_agent, action):
                raise AttributeError(f"Module '{module}' has no action '{action}'")
            
            action_method = getattr(module_agent, action)
            
            # Execute with parameters
            return action_method(**params)
        except Exception as e:
            logger.error(f"External call failed: {module}.{action} - {str(e)}")
            raise
    
    def process_command(self, command: str, context: dict) -> Any:
        """
        Processes a string-based command within the current context.
        Supports basic operations and context manipulation.
        """
        logger.info(f"[{self.name}] Processing command: {command}")
        
        # Simple command parsing
        parts = command.split(' ', 1)
        cmd_type = parts[0].lower()
        payload = parts[1] if len(parts) > 1 else ""
        
        try:
            if cmd_type == "log":
                # log <message>
                logger.info(f"[Command] {payload}")
                return {"status": "logged", "message": payload}
            
            elif cmd_type == "set":
                # set <key> <value>
                key, value = payload.split(' ', 1)
                context[key] = value
                return {"status": "set", "key": key, "value": value}
            
            elif cmd_type == "get":
                # get <key>
                value = context.get(payload, "NOT_FOUND")
                return {"status": "retrieved", "key": payload, "value": value}
            
            elif cmd_type == "incr":
                # incr <key> [increment=1]
                parts = payload.split(' ')
                key = parts[0]
                increment = float(parts[1]) if len(parts) > 1 else 1
                current = float(context.get(key, 0))
                context[key] = current + increment
                return {"status": "incremented", "key": key, "value": context[key]}
            
            else:
                raise ValueError(f"Unknown command: {cmd_type}")
        
        except Exception as e:
            logger.error(f"Command processing failed: {command} - {str(e)}")
            return {"status": "error", "command": command, "error": str(e)}
    
    def evaluate_performance(self, metrics: dict):
        """
        Evaluates performance based on extracted metrics, logs them, and checks thresholds for retraining.
        """
        if not metrics or not isinstance(metrics, dict): 
            self.logger.debug(f"[{self.name}] Skipping performance evaluation: No valid metrics provided.")
            return

        # Update rolling performance metrics (using the lazy property)
        # Ensure performance_metrics component is initialized and is the expected defaultdict
        perf_metrics_component = self.performance_metrics 
        if not isinstance(perf_metrics_component, defaultdict):
            self.logger.warning(f"[{self.name}] 'performance_metrics' component is not a defaultdict. Cannot update rolling metrics.")
        else:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.metric_store.metrics['timings']['performance'][key].append(value)
                else:
                    self.logger.debug(f"[{self.name}] Skipping rolling update for non-scalar metric '{key}'.")


        self.log_evaluation_result(metrics) # Log current evaluation to file
    
        # Check retraining thresholds
        for metric_key, current_value in metrics.items():
            threshold_info = self.retraining_thresholds.get(metric_key) # Could be a value or a dict like {'value': X, 'condition': 'less_than'}
            
            if threshold_info is not None:
                threshold_value = threshold_info
                condition = 'less_than' # Default condition: trigger if current_value is less than threshold_value (lower is worse)

                if isinstance(threshold_info, dict):
                    threshold_value = threshold_info.get('value')
                    condition = threshold_info.get('condition', 'less_than')

                if threshold_value is None: continue

                if isinstance(current_value, (int, float)) and isinstance(threshold_value, (int, float)):
                    trigger_retraining = False
                    if condition == 'less_than' and current_value < threshold_value:
                        trigger_retraining = True
                    elif condition == 'greater_than' and current_value > threshold_value: # e.g., error rate too high
                        trigger_retraining = True
                    
                    if trigger_retraining:
                        self.logger.warning(
                            f"[{self.name}] Performance alert: Metric '{metric_key}' ({current_value}) "
                            f"violated threshold (condition: {condition} {threshold_value})."
                        )
                        self.flag_for_retraining()
                else:
                    self.logger.debug(f"[{self.name}] Metric '{metric_key}' or its threshold is not numeric. Skipping threshold check.")

    def flag_for_retraining(self):
        """Sets a flag in shared memory indicating this agent needs retraining."""
        flag_key = f"retraining_flag:{self.name}"
        self.shared_memory.set(flag_key, True)
        self.logger.info(f"[{self.name}] Agent flagged for retraining via key '{flag_key}'.")

    def log_evaluation_result(self, metrics: dict):
        """Logs the current evaluation metrics to a JSONL file."""
        # Sanitize config for logging: avoid logging sensitive info or overly large structures
        config_summary = {
            k: v for k, v in self.config.items() 
            if k not in ['defer_initialization', 'memory_profile', 'network_compression'] and not isinstance(v, (dict, list))
        } # Log only simple config values, or define a specific subset

        log_entry = {
            "timestamp": time.time(),
            "agent_name": self.name,
            "agent_config_summary": config_summary, 
            "metrics": metrics
        }
        path = os.path.join(self.evaluation_log_dir, f"{self.name}_eval.jsonl")
        try:
            with open(path, "a", encoding="utf-8") as f: # Specify encoding
                f.write(json.dumps(log_entry, default=str) + "\n") # default=str for non-serializable
        except Exception as e:
            self.logger.error(f"[{self.name}] Failed to write evaluation log to '{path}': {e}")

    def optimized_step(self, state: Any) -> Any:
        """
        Template for a memory-aware step method, typically for reinforcement learning agents.
        Subclasses should override this with their specific model and logic.
        """
        if not hasattr(self, '_policy_net_instance_cache'): # Use a more descriptive name
            # Create and cache the network instance
            # Dimensions should ideally come from agent's config or environment spec
            self._policy_net_instance_cache = self.create_lightweight_network(input_dim=10, output_dim=2) # Example dims
        return self._policy_net_instance_cache.predict(state)

    def _init_agent_specific_components(self):
        """
        Template for subclasses to initialize their unique components (e.g., models, tools).
        This method is intended to be called by the subclass's __init__ AFTER super().__init__().
        """
        self.logger.info(f"[{self.name}] Base `_init_agent_specific_components` called. Subclass should override if it has specific components to initialize.")
        # Example:
        # if self.config.get("use_vision_model"):
        #     self.vision_model = self.agent_factory.create("VisionEncoder", self.config.get("vision_model_config"))
        #     self.vision_model.initialize() # Assuming VisionEncoder has an initialize method

    def _warm_start_if_available(self):
        """
        Template for subclasses to restore their state from shared memory if available.
        Typically called by a subclass's __init__ method.
        """
        warm_start_key = f"warm_state:{self.name}"
        cached_state_data = self.shared_memory.get(warm_start_key)
        
        if cached_state_data and isinstance(cached_state_data, dict):
            self.logger.info(f"[{self.name}] Attempting warm-start from cached state found at key '{warm_start_key}'.")
            # Define attributes that are safe and expected to be warm-started
            # WARNING: Directly using self.__dict__.update(cached_state_data) is DANGEROUS 
            # as it can overwrite methods or critical internal state.
            expected_warm_attributes = getattr(self, "warm_start_attributes", []) # Subclass can define this list
            
            attributes_loaded = 0
            for attr_name, attr_value in cached_state_data.items():
                if attr_name in expected_warm_attributes:
                    if hasattr(self, attr_name) and not callable(getattr(self, attr_name)): # Only update non-callable attributes
                        setattr(self, attr_name, attr_value)
                        attributes_loaded +=1
                    else:
                        self.logger.warning(f"[{self.name}] Warm-start: Skipping '{attr_name}', not a safe non-callable attribute or not in expected_warm_attributes.")
            
            if attributes_loaded > 0:
                 self.logger.info(f"[{self.name}] Warm-start successful for {attributes_loaded} attribute(s).")
            else:
                 self.logger.info(f"[{self.name}] Warm-start: No matching attributes found in cached state or 'warm_start_attributes' not defined.")

        else:
            self.logger.info(f"[{self.name}] No valid warm-start state found at key '{warm_start_key}'. Proceeding with fresh initialization.")

    def create_lightweight_network(self, input_dim: int = 10, output_dim: int = 2) -> nn.Module:
        """
        Returns a basic PyTorch neural network module.
        """
        class PolicyNet(nn.Module):
            def __init__(self, input_dim: int, output_dim: int):
                super().__init__()
                self.input_dim = input_dim
                self.output_dim = output_dim
                # Proper PyTorch parameter initialization
                self.weights = nn.Parameter(
                    torch.randn(input_dim, output_dim) * torch.sqrt(torch.tensor(1.0 / input_dim)))
                self.bias = nn.Parameter(torch.zeros(output_dim))
                logger.debug(f"PolicyNet initialized with input_dim={input_dim}, output_dim={output_dim}")
    
            def predict(self, state: Any) -> Any:
                try:
                    # Convert to tensor and flatten
                    state_vec = torch.tensor(state, dtype=torch.float32).flatten()
                    if state_vec.shape[0] != self.input_dim:
                        raise ValueError(f"Input state dimension {state_vec.shape[0]} "
                                         f"does not match network input dimension {self.input_dim}")
                    
                    # Use PyTorch matrix multiplication
                    logits = torch.matmul(state_vec, self.weights) + self.bias
                    # Return action index
                    return torch.argmax(logits).item()
                    
                except Exception as e:
                    logger.error(f"PolicyNet: Error processing input state: {e}")
                    # Return random action on error
                    return torch.randint(0, self.output_dim, (1,)).item()
        
        return PolicyNet(input_dim=input_dim, output_dim=output_dim)
    
    def update_projection(self, reward_scores: list, lr: float):
        """
        Placeholder/Template for updating a 'projection' attribute, possibly a torch.Tensor.
        This method's logic is highly dependent on how `self.projection` is used in the agent's
        architecture and the learning algorithm (e.g., policy gradients, value-based).
        The current implementation is a HEURISTIC and not a standard gradient update.
        It assumes `self.projection` is a `torch.Tensor` and requires gradients.
        """
        if not hasattr(self, 'projection') or not isinstance(getattr(self, 'projection', None), torch.Tensor):
            self.logger.warning(f"[{self.name}] 'projection' attribute not found or is not a torch.Tensor. Skipping 'update_projection'.")
            return

        projection_tensor = self.projection

        if not projection_tensor.requires_grad:
            self.logger.warning(f"[{self.name}] 'projection' tensor does not require_grad. It might not be part of an optimizable model.")
            # For this heuristic update, we might proceed even if requires_grad is False,
            # but for actual backpropagation, it would need to be True and part of a graph.
            # projection_tensor.requires_grad_(True) # This is generally not done here.

        try:
            rewards = torch.tensor(reward_scores, dtype=torch.float32, device=projection_tensor.device)
            
            # --- This is a HEURISTIC update rule ---
            # It assumes a direct, positive correlation between the projection tensor's values
            # and the rewards obtained. This is NOT a standard policy gradient update.
            # A proper PG update would involve gradients of action log-probabilities w.r.t. model parameters.
            
            # Calculate a pseudo-gradient based on mean reward.
            # If mean reward is positive, "reinforce" current projection values.
            # If mean reward is negative, "discourage" (not implemented here to keep it simple).
            mean_reward = rewards.mean()
            
            if mean_reward != 0: # Avoid division by zero or no change if reward is zero
                # Heuristic: scale the projection by the mean reward.
                # This is more like a scaling factor than a gradient.
                # For an actual gradient, one would use projection_tensor.grad.
                pseudo_gradient_direction = torch.sign(projection_tensor.data) * mean_reward # Scale by sign and reward
                
                # Apply update directly to data (if not using an optimizer for this tensor)
                with torch.no_grad(): # Ensure this operation is not tracked for further gradients
                    projection_tensor.data += lr * pseudo_gradient_direction
                    
                    # Optional: Apply constraints like clamping or normalization
                    # projection_tensor.data = torch.clamp(projection_tensor.data, -1.0, 1.0)
                    # projection_tensor.data /= torch.norm(projection_tensor.data) + 1e-6 # Normalize

                self.logger.debug(f"[{self.name}] Heuristically updated 'projection' tensor with lr={lr}, mean_reward={mean_reward:.4f}.")
            else:
                self.logger.debug(f"[{self.name}] 'projection' tensor not updated as mean_reward is zero.")

        except Exception as e:
            self.logger.error(f"[{self.name}] Error in 'update_projection': {e}")
            self.logger.debug(traceback.format_exc())

    def broadcast(self, key: str, value: Any):
        """Broadcast to shared memory"""
        self.shared_memory.set(key, value)

class RetrainingManager:
    """
    Manages the retraining process for an agent based on flags in shared memory
    or other triggers (though currently only flag-based).
    """
    def __init__(self, agent: BaseAgent, shared_memory: Any): # Type hint for agent and shared_memory
        if not isinstance(agent, BaseAgent): # Basic type check
            raise TypeError("RetrainingManager requires an instance of BaseAgent or its subclass.")
        if shared_memory is None: # Basic check
            raise ValueError("RetrainingManager requires a valid shared_memory instance.")

        self.agent = agent
        self.shared_memory = shared_memory
        self.logger = get_logger(f"{self.__class__.__name__}[{self.agent.name}]")

    def check_and_trigger_retraining(self):
        """
        Checks for a retraining flag in shared memory specific to the agent.
        If the flag is True, attempts to call the agent's `retrain` method.
        """
        retraining_flag_key = f"retraining_flag:{self.agent.name}"
        
        try:
            flag_value = self.shared_memory.get(retraining_flag_key)
        except Exception as e:
            self.logger.error(f"Error accessing shared memory for retraining flag '{retraining_flag_key}': {e}")
            return # Cannot proceed if shared memory access fails

        if flag_value: # Typically True if set
            self.logger.info(f"Retraining flag is set for agent '{self.agent.name}'. Initiating retraining process.")
            
            if hasattr(self.agent, 'retrain') and callable(self.agent.retrain):
                try:
                    self.agent.retrain() # Call the agent's own retrain method
                    self.logger.info(f"Retraining method called successfully for agent '{self.agent.name}'.")
                    # Reset the flag in shared memory after a successful call to retrain()
                    self.shared_memory.set(retraining_flag_key, False) 
                    self.logger.debug(f"Retraining flag '{retraining_flag_key}' reset to False.")
                except Exception as e_retrain:
                    self.logger.error(f"Error during agent '{self.agent.name}' retraining method: {e_retrain}")
                    self.logger.debug(traceback.format_exc())
                    # Policy: Do not reset the flag if retraining itself fails, so it might be retried.
                    # Or, implement a max_retrain_attempts mechanism.
            else:
                self.logger.warning(f"Agent '{self.agent.name}' is flagged for retraining but does not have a callable 'retrain' method.")
                # Reset the flag as it cannot be actioned by this manager.
                self.shared_memory.set(retraining_flag_key, False)
                self.logger.debug(f"Retraining flag '{retraining_flag_key}' reset as agent has no retrain method.")
        else:
            self.logger.debug(f"No retraining flag set for agent '{self.agent.name}'.")

class ResourceMonitor:
    """Lightweight resource monitoring"""
    def __init__(self):
        self.warning_threshold = 0.7
        self.critical_threshold = 0.9
    
    def is_critical(self):
        """Check system resource levels"""
        # Simplified actual implementation would use:
        # psutil.cpu_percent(), psutil.virtual_memory().percent
        current_cpu = 0.6  # Mock value
        current_mem = 0.8  # Mock value
        
        if current_cpu > self.critical_threshold or current_mem > self.critical_threshold:
            return True
        return False

# ====================== Usage Example ======================
if __name__ == "__main__":
    print("\n=== Running SLAI Base Agent ===\n")
    config = get_config_section('base_agent')
    shared_memory={}
    agent_factory=None

    class TempAgent(BaseAgent):
        def __init__(self, shared_memory, agent_factory, config=None):
            # Initialize logger FIRST
            self.logger = get_logger(self.__class__.__name__)
            self.name = self.__class__.__name__
            # Now call parent constructor
            super().__init__(shared_memory, agent_factory, config)

        def perform_task(self, task_data: Any) -> Any:
            return {"status": "success"}

    base1 = type('TempAgent', (BaseAgent,), {
        'perform_task': lambda self, task_data: {"status": "success"}
    })(shared_memory, agent_factory, config)

    agent = TempAgent(shared_memory, agent_factory, config)

    print(f"\n{base1}")
    print(f"\n{agent}")

    print("\n* * * * * Phase 2 * * * * *\n")
    reward_scores=None
    lr=0.01

    base2 = BaseAgent(shared_memory, agent_factory, config=None)

    update = base2.update_projection(reward_scores, lr)

    print(f"\n{base2}")
    print(f"\n{update}")

    print("\n* * * * * Phase 3 * * * * *\n")
    init_fn=callable
    agent=None
    class DummySharedMemory(dict):
        def get(self, key, default=None):
            return super().get(key, default)
        def set(self, key, value):
            self[key] = value

    shared_memory = DummySharedMemory()

    store = LightMetricStore()
    lazy = LazyAgent(init_fn)
    agent = TempAgent(shared_memory, agent_factory, config)
    manager = RetrainingManager(agent=agent, shared_memory=shared_memory)

    print(f"\n{store}\n{lazy}\n{agent}\n{manager}")
    print("\n* * * * * Phase 4 * * * * *\n")

    print("\n=== Successfully Ran SLAI Base Agent ===\n")
