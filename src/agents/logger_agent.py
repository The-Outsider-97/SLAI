import logging
from modules.monitoring import Monitoring
from src.agents.base_agent import BaseAgent

class LoggerAgent(BaseAgent):
    def __init__(self, shared_memory=None, output_path="logs/metrics_log.jsonl"):
        super().__init__(shared_memory=shared_memory)
        self.monitor = Monitoring(shared_memory=shared_memory, output_path=output_path)

    def log_metric(self, module_name, metrics: dict, tags: dict = None):
        """Automatically log a metric from any source"""
        return self.monitor.record(module_name, metrics, tags)

    def get_recent_metrics(self, metric: str):
        return self.monitor.get_rolling_stats(metric)

    def print_summary(self):
        self.monitor.print_summary()
