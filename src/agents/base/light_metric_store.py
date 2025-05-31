
import json
import time

from collections import defaultdict

from src.agents.base.utils.config_loader import load_global_config, get_config_section
from logs.logger import get_logger

logger = get_logger("Light Metric Store")

class LightMetricStore:
    """Lightweight metric tracking for performance (timings) and memory usage changes."""
    def __init__(self):
        self.config = load_global_config()
        self.lm_config = get_config_section('lm_store')
        
        self.metrics = {
            'timings': defaultdict(list),
            'memory_deltas': defaultdict(list)
        }
        self._start_times = {}
        self._start_memory_rss = {}
        self.enable_memory_tracking = self.lm_config.get('enable_memory_tracking', True)
        self.default_category = self.lm_config.get('default_category', 'default')

        logger.info(f"Light Metric Store succesfully initialized with:\n{self.metrics}")

    def start_tracking(self, metric_name: str, category: str = "default"):
        """Start tracking a specific operation under a category."""
        key = (category, metric_name)
        self._start_times[key] = time.perf_counter() 
        try:
            import psutil
            self._start_memory_rss[key] = psutil.Process().memory_info().rss 
        except ImportError:
            if key not in self._start_memory_rss: # Log warning only once per metric start
                 logger.debug("psutil not installed. Memory delta tracking will be zero for LightMetricStore.")
            self._start_memory_rss[key] = 0 # psutil not available, store 0

    def stop_tracking(self, metric_name: str, category: str = "default"):
        """Stop tracking and record the duration and memory delta."""
        key = (category, metric_name)
        
        # Record timing
        if key in self._start_times:
            duration_s = time.perf_counter() - self._start_times.pop(key)
            if category not in self.metrics['timings']: self.metrics['timings'][category] = defaultdict(list)
            self.metrics['timings'][category][metric_name].append(duration_s)
        else:
            logger.warning(f"Metric ('{category}', '{metric_name}') timing was stopped without being started.")

        # Record memory usage delta
        if key in self._start_memory_rss:
            start_rss = self._start_memory_rss.pop(key)
            mem_delta_mb = 0.0
            if start_rss != 0: # Implies psutil was available at start
                try:
                    import psutil
                    current_rss = psutil.Process().memory_info().rss
                    mem_delta_mb = (current_rss - start_rss) / (1024 ** 2)
                except ImportError:
                    pass # Already logged, delta remains 0
            
            if category not in self.metrics['memory_deltas']: self.metrics['memory_deltas'][category] = defaultdict(list)
            self.metrics['memory_deltas'][category][metric_name].append(mem_delta_mb)
        else:
            logger.warning(f"Metric ('{category}', '{metric_name}') memory tracking was stopped without being started.")

    def get_metrics_summary(self, category: str = "default") -> dict:
        """Generate a summary (e.g., average) of metrics for a category."""
        summary = {}
        if category in self.metrics['timings']:
            summary['timings_avg_s'] = {
                name: sum(values)/len(values) if values else 0 
                for name, values in self.metrics['timings'][category].items()
            }
        if category in self.metrics['memory_deltas']:
            summary['memory_deltas_avg_mb'] = {
                name: sum(values)/len(values) if values else 0 
                for name, values in self.metrics['memory_deltas'][category].items()
            }
        return summary

    def get_all_metrics_json(self, pretty: bool = True) -> str:
        """Generate a JSON string of all raw recorded metrics."""
        if pretty:
            return json.dumps(self.metrics, indent=2, default=lambda o: str(o)) # Handle defaultdict
        return json.dumps(self.metrics, default=lambda o: str(o))


# ====================== Usage Example ======================
if __name__ == "__main__":
    print("\n=== Running Light Metric Store ===\n")
    config = load_global_config()

    store = LightMetricStore()
    print(f"\n{store}")
    print("\n=== Successfully Ran Light Metric Store ===\n")
