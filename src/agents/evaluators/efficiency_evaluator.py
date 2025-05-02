import time
import sys
import resource
from typing import List, Any, Dict, Optional
import numpy as np

class EfficiencyEvaluator:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.baseline_measurements = self.config.get('baselines', {})
        self.complexity_metrics = self.config.get('complexity_metrics', True)

    def evaluate(self, outputs: List[Any], ground_truths: List[Any]) -> Dict[str, float]:
        """Multi-dimensional efficiency assessment with resource monitoring"""
        start_time = time.perf_counter()
        start_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        
        # Calculate efficiency metrics
        metrics = {
            'temporal_efficiency': self._calculate_temporal(outputs),
            'spatial_efficiency': self._calculate_spatial(outputs),
            'computational_efficiency': self._calculate_computational(),
            'token_efficiency': self._calculate_token_efficiency(outputs)
        }
        
        if self.complexity_metrics:
            metrics.update({
                'parameter_efficiency': self.baseline_measurements.get('params', 0),
                'flops': self.baseline_measurements.get('flops', 0)
            })
            
        end_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        metrics['memory_usage_mb'] = (end_mem - start_mem) / 1024
        
        metrics['execution_time'] = time.perf_counter() - start_time
        metrics['score'] = self._calculate_composite_score(metrics)
        
        return metrics

    def _calculate_temporal(self, outputs):
        """Response latency efficiency"""
        return 1 / (len(outputs) + sys.float_info.epsilon)

    def _calculate_spatial(self, outputs):
        """Memory footprint efficiency"""
        return 1 / (sum(sys.getsizeof(o) for o in outputs) + sys.float_info.epsilon)

    def _calculate_computational(self):
        """FLOPs relative to baseline"""
        baseline = self.baseline_measurements.get('flops', 1e6)
        current = self.config.get('current_flops', baseline)
        return baseline / (current + sys.float_info.epsilon)

    def _calculate_token_efficiency(self, outputs):
        """Tokens per output for NLP systems"""
        if self.config.get('language_model', False):
            token_counts = [len(str(o).split()) for o in outputs]
            return np.mean(token_counts) if token_counts else 0
        return 0.0

    def _calculate_composite_score(self, metrics):
        weights = self.config.get(
            'efficiency_weights',
            {'temporal': 0.4, 'spatial': 0.3, 'computational': 0.3}
        )
        return sum(
            weights.get(metric, 0) * value 
            for metric, value in metrics.items()
            if metric in weights
        )
