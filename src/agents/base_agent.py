import logging
import os, sys
import time
import json
import difflib
import traceback
from collections import OrderedDict, defaultdict, deque

class BaseAgent:
    def __init__(self, shared_memory, agent_factory, config=None):
            self.shared_memory=shared_memory,
            self.agent_factory=agent_factory,
            self.config=config or {
                'defer_initialization': True,
                'memory_profile': 'low',
                'network_compression': True
            }
            # Initialize lazy components system
            self._component_initializers = {
                'memory_view': lambda: self.SharedMemoryView(shared_memory),
                'performance_metrics': lambda: defaultdict(lambda: deque(maxlen=500))
            }
            self._lazy_components = OrderedDict()
            
            # Initialize core components immediately
            self._init_core_components()

            self.logger = logging.getLogger(self.__class__.__name__)
            self.name = self.__class__.__name__

    def _init_core_components(self):
        """Initialize essential components first"""
        # Access memory view to initialize it
        _ = self.memory_view  # Triggers initialization through property

    @property
    def memory_view(self):
        return self.lazy_property('memory_view')

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

    def execute(self, input_data):
        try:
            self.logger.info(f"[{self.name}] Executing task...")
            result = self.perform_task(input_data)
            self.logger.info(f"[{self.name}] Task execution completed.")
            
            # Log successful stats
            self.shared_memory.set(f"agent_stats:{self.name}", {
                "last_run": time.time(),
                "success": True,
                "result_summary": str(result)[:200]
            })

            return result

        except Exception as e:
            error_entry = {
                "timestamp": time.time(),
                "error": str(e),
                "traceback": traceback.format_exc()
            }

            # Retrieve and trim old errors
            error_key = f"errors:{self.name}"
            MAX_ERRORS = 50
            errors = self.shared_memory.get(error_key) or []
            errors.append(error_entry)
            if len(errors) > MAX_ERRORS:
                errors = errors[-MAX_ERRORS:]
            self.shared_memory.set(error_key, errors)

            # Log failure stats
            self.shared_memory.set(f"agent_stats:{self.name}", {
                "last_run": time.time(),
                "success": False,
                "error": str(e)
            })

            self.logger.error(f"[{self.name}] Task failed with error: {e}")
            self.logger.debug(traceback.format_exc())

            # Match against recent errors for similarity
            SIMILARITY_THRESHOLD = 0.75
            new_error = str(e)
            new_type = type(e).__name__

            for past in reversed(errors[:-1]):
                past_error = past.get("error", "")
                past_type = past_error.split(":")[0] if ":" in past_error else ""

                # Match exception type first (e.g., ValueError vs TypeError)
                if past_type == new_type:
                    # Check textual similarity
                    similarity = difflib.SequenceMatcher(None, new_error, past_error).ratio()
                    if similarity >= SIMILARITY_THRESHOLD:
                        self.logger.warning(
                            f"[{self.name}] A similar {new_type} occurred before "
                            f"(similarity: {similarity:.2f})"
                        )
                        break

            return {"status": "failed", "error": str(e)}
    
    def alternative_execute(self, task_data):
        """
        Fallback logic when normal execution fails or matches a known failure pattern.
        Attempts to simplify, sanitize, or reroute the input for safer processing.
        """
        try:
            # Step 1: Sanitize task data (remove noise, normalize casing, trim tokens)
            if isinstance(task_data, str):
                clean_data = task_data.strip().lower().replace('\n', ' ')
            elif isinstance(task_data, dict) and "text" in task_data:
                clean_data = task_data["text"].strip().lower()
            else:
                clean_data = str(task_data).strip()

            # Step 2: Apply a safer, simplified prompt or fallback logic
            fallback_prompt = f"Can you try again with simplified input:\n{clean_data}"
            if hasattr(self, "llm") and callable(getattr(self.llm, "generate", None)):
                return self.llm.generate(fallback_prompt)

            # Step 3: If the agent wraps another processor (e.g. GrammarProcessor, LLM), reroute
            if hasattr(self, "grammar") and callable(getattr(self.grammar, "compose_sentence", None)):
                facts = {"event": "fallback", "value": clean_data}
                return self.grammar.compose_sentence(facts)

            # Step 4: Otherwise just echo the cleaned input as confirmation
            return f"[Fallback response] I rephrased your input: {clean_data}"

        except Exception as e:
            # Final fallback — very safe and generic
            return "[Fallback failure] Unable to process your request at this time."
    
    def is_similar(self, task_data, past_task_data):
        """
        Compares current task with past task to detect similarity.
        Uses key overlap and value resemblance heuristics.
        """
        if type(task_data) != type(past_task_data):
            return False
    
        # Handle simple text-based tasks
        if isinstance(task_data, str) and isinstance(past_task_data, str):
            return task_data.strip().lower() == past_task_data.strip().lower()
    
        # Handle dict-based structured tasks
        if isinstance(task_data, dict) and isinstance(past_task_data, dict):
            shared_keys = set(task_data.keys()) & set(past_task_data.keys())
            similarity_score = 0
            for key in shared_keys:
                if isinstance(task_data[key], str) and isinstance(past_task_data[key], str):
                    if task_data[key].strip().lower() == past_task_data[key].strip().lower():
                        similarity_score += 1
            # Consider similar if 50% or more keys match closely
            return similarity_score >= (len(shared_keys) / 2)
    
        return False

    def handle_known_issue(self, task_data, error):
        """
        Attempt to recover from known failure patterns.
        Could apply input transformation or fallback logic.
        """
        self.logger.warning(f"Handling known issue from error: {error.get('error')}")
    
        # Fallback strategy #1: remove problematic characters
        if isinstance(task_data, str):
            cleaned = task_data.replace("🧠", "").replace("🔥", "")
            self.logger.info(f"Retrying with cleaned input: {cleaned}")
            return self.perform_task(cleaned)
    
        # Fallback strategy #2: modify specific fields in structured input
        if isinstance(task_data, dict):
            cleaned_data = task_data.copy()
            for key, val in cleaned_data.items():
                if isinstance(val, str) and "emoji" in error.get("error", ""):
                    cleaned_data[key] = val.encode("ascii", "ignore").decode()
            self.logger.info("Retrying task with cleaned structured data.")
            return self.perform_task(cleaned_data)
    
        # Fallback strategy #3: return a graceful degradation response
        self.logger.warning("Returning fallback response for unresolvable input.")
        return {"status": "failed", "reason": "Repeated known issue", "fallback": True}
    
    def perform_task(self, task_data):
        """Handle query execution as the primary task"""
        self.logger.info(f"Executing task with data: {task_data}")
    
        if isinstance(task_data, dict) and "query" in task_data:
            return self.retrieve(task_data["query"])
        elif isinstance(task_data, str):
            return self.retrieve(task_data)
        else:
            raise ValueError("Unsupported task format")


    class SharedMemoryView:
        """Lightweight memory access with LRU caching"""
        __slots__ = ['_source', '_cache', '_cache_size']
        
        def __init__(self, shared_memory):
            self._source = shared_memory
            self._cache = OrderedDict()
            self._cache_size = 100
            
        def get(self, key):
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]

            # Fetch from source and cache    
            value = self._source.get(key)
            self._cache[key] = value

            # Enforce cache size limit
            if len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)

            return value

        def set(self, key, value):
            # Update both cache and source
            self._source.set(key, value)
            if key in self._cache:
                self._cache[key] = value

        def get_usage_stats(self):
            return {
                'cache_size': len(self._cache),
                'hit_rate': self._source.get('cache_hit_rate', 0)
            }

    def optimized_step(self, state):
        """Memory-aware step method template"""
        if not hasattr(self, '_policy_net'):
            self._policy_net = self.create_lightweight_network()
        return self._policy_net.predict(state)

    # Template methods for subclasses
    def _init_agent_specific_components(self):
        """To be implemented by subclasses"""
        pass

    def _warm_start_if_available(self):
        """Optional warm-start from shared memory"""
        pass

    def create_lightweight_network(self):
        """Override for agent-specific networks"""
        return None

class LightMetricStore:
    """Lightweight metric tracking for performance and memory"""
    def __init__(self):
        self.metrics = {
            'timings': {},
            'memory_usage': {}
        }

    def start_tracking(self, metric_name: str):
        """Start tracking a metric"""
        self.metrics['timings'][metric_name] = time.time()
        self.metrics['memory_usage'][metric_name] = self._get_current_memory()

    def stop_tracking(self, metric_name: str):
        """Stop tracking and calculate deltas"""
        if metric_name in self.metrics['timings']:
            self.metrics['timings'][metric_name] = time.time() - self.metrics['timings'][metric_name]
        if metric_name in self.metrics['memory_usage']:
            self.metrics['memory_usage'][metric_name] = self._get_current_memory() - self.metrics['memory_usage'][metric_name]

    def _get_current_memory(self) -> float:
        """Get current process memory in MB"""
        import psutil
        return psutil.Process().memory_info().rss / 1024 ** 2

    def get_metrics_report(self) -> str:
        """Generate formatted metrics report"""
        return json.dumps(self.metrics, indent=2)

class LazyAgent:
    """
    Wrapper for deferred initialization of learning sub-agents.
    Executes creation only when first accessed.
    """
    def __init__(self, init_fn):
        self._init_fn = init_fn
        self._agent = None

    def __getattr__(self, item):
        if self._agent is None:
            self._agent = self._init_fn()
        return getattr(self._agent, item)
    
#if __name__ == "__main__":
#    agent.memory_view.get('recent_errors')
#    agent.performance_metrics['response_times'].append(0.42)
