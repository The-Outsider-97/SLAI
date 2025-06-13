
import yaml
import torch

from collections import defaultdict, deque
from typing import Dict, List, Any

from src.agents.factory.utils.config_loader import load_global_config, get_config_section
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Metrics Adapter")
printer = PrettyPrinter

class MetricsAdapter:
    """
    Bridges metrics analysis with agent configuration using control-theoretic adaptation
    Implements principles from:
    - PID Controllers (Åström & Hägglund, 1995)
    - Safe Exploration (Turchetta et al., 2020)
    - Fairness-Aware RL (D'Amour et al., 2020)
    """

    def __init__(self):
        self.config = load_global_config()
        self.meta_config = get_config_section('metrics')
        self.demographic_parity_diff = self.meta_config.get('demographic_parity_diff')
        self.calibration_error = self.meta_config.get('calibration_error')
        self.max_adaptation_rate = self.meta_config.get('max_adaptation_rate')
        self.history_size = self.meta_config.get('history_size')
        self.metric_history = deque(maxlen=self.history_size)
        self.accuracy = self.meta_config.get('accuracy')
        self.adaptation_factors = self.meta_config.get('adaptation_factors', {
            'risk_threshold', 'exploration_rate', 'learning_rate'
        })
        self.error_config = self.meta_config.get("error_config", {
            'fairness_target', 'performance_target', 'bias_target'
        })
        self.pid_params = self.meta_config.get("pid_params", {
            'Kp', 'Ki', 'Kd'
        })
        self.safety_bounds = self.meta_config.get("safety_bounds", {
            'medical', 'defaults'
        })

        #self.adaptation_factors = {
        #    'risk_threshold': 1.0,
        #    'exploration_rate': 1.0,
        #    'learning_rate': 1.0
        #}
        self._init_control_parameters()

    def _init_control_parameters(self):
        """PID tuning per Ziegler-Nichols method"""
        self.Kp = self.pid_params['Kp']
        self.Ki = self.pid_params['Ki']
        self.Kd = self.pid_params['Kd']
        self.integral = defaultdict(float)
        self.prev_error = defaultdict(float)

    def _calculate_metric_deltas(self) -> Dict[str, float]:
        """Numerical differentiation for trend analysis"""
        if len(self.metric_history) < 2:
            return {}
            
        current = self.metric_history[-1]
        previous = self.metric_history[-2]
        
        return {
            'fairness': current.get('demographic_parity', 0) - previous.get('demographic_parity', 0),
            'performance': current.get('calibration_error', 0) - previous.get('calibration_error', 0)
        }

    def _calculate_error(self, metrics: Dict[str, Any], metric_type: str) -> float:
        target = self.error_config.get(f"{metric_type}_target", 0.0)
        current = metrics.get(metric_type, {}).get("value", 0.0)
        
        try:
            if metric_type == "fairness":
                return self._calculate_fairness_error(current, target)
            elif metric_type == "performance":
                return self._calculate_performance_error(current, target)
            elif metric_type == "bias":
                return self._calculate_bias_error(metrics, target)
            else:
                raise ValueError(f"Unknown metric type: {metric_type}")
        except KeyError as e:
            logger.warning(f"Missing metric data for {metric_type}: {str(e)}")
            return 0.0

    def _calculate_fairness_error(self, current: float, target: float) -> float:
        # Normalized demographic parity difference
        dpd = current.get('demographic_parity_diff')
        return (dpd - target) / (1.0 + abs(dpd))

    def _calculate_performance_error(self, current: float, target: float) -> float:
        # Composite error including calibration and accuracy
        calibration_error = current.get('calibration_error')
        accuracy = current.get('accuracy')
        return (calibration_error - target) * (1.0 - accuracy)

    def _calculate_bias_error(self, metrics: Dict, target: float) -> float:
        # Calculate maximum disparity between any two groups
        group_metrics = metrics.get("bias", {}).get("group_metrics", {})
        if len(group_metrics) < 2:
            return 0.0
            
        values = [v.get("score", 0.0) for v in group_metrics.values()]
        max_disparity = max(values) - min(values)
        return (max_disparity - target) / (1.0 + max_disparity)

    def process_metrics(self, 
                       metrics: Dict[str, Any], 
                       agent_types: List[str]) -> Dict[str, float]:
        """
        Convert raw metrics to adaptation factors using:
        - Moving average filters
        - Differential fairness constraints
        - Calibration-aware adjustments
        """
        self.metric_history.append(metrics)
        
        # Calculate temporal derivatives
        delta = self._calculate_metric_deltas()
        
        # Apply control theory
        adjustments = {}
        for metric_type in ['fairness', 'performance', 'bias']:
            error = self._calculate_error(metrics, metric_type)
            adjustments.update(
                self._pid_control(metric_type, error, delta.get(metric_type, 0)))
        
        # Enforce ISO/IEC 24027:2021 safety bounds
        return self._apply_safety_bounds(adjustments, agent_types)

    def _apply_safety_bounds(self, adjustments, agent_types):
        for agent_type in agent_types:
            bound = self.safety_bounds.get(agent_type, self.safety_bounds.get('default', 1.0))
            for key, value in adjustments.items():
                if isinstance(value, torch.Tensor):
                    value = value.item()
                if abs(value) > bound:
                    adjustments[key] = torch.tensor(bound * (1 if value > 0 else -1))
        return adjustments

    def _pid_control(self, 
                    metric_type: str, 
                    error: float,
                    delta: float) -> Dict[str, float]:
        """Discrete PID controller implementation"""
        self.integral[metric_type] += error
        derivative = error - self.prev_error[metric_type]
        
        adjustment = (self.Kp * error +
                     self.Ki * self.integral[metric_type] +
                     self.Kd * derivative)
        
        self.prev_error[metric_type] = error
        return {
            f"{metric_type}_adjustment": adjustment
        }

    def _pid_control(self, 
                    metric_type: str, 
                    error: float,
                    delta: float) -> Dict[str, float]:
        """Discrete PID controller implementation"""
        self.integral[metric_type] += error
        derivative = error - self.prev_error[metric_type]
        
        # Calculate adjustment and convert to tensor
        adjustment = (
            self.Kp * error +
            self.Ki * self.integral[metric_type] +
            self.Kd * derivative
        )
        adjustment_tensor = torch.tensor(adjustment, dtype=torch.float32)
        
        self.prev_error[metric_type] = error
        return {
            f"{metric_type}_adjustment": adjustment_tensor
        }

    def update_factory_config(self, 
                             factory: 'AgentFactory',
                             adjustments: Dict[str, float]):
        """Dynamic reconfiguration using meta-learning gradients"""
        for agent_type in factory.registry.values():
            # Update exploration rates (handle tensor/float)
            performance_adj = adjustments.get('performance_adjustment', 0.0)
            if isinstance(performance_adj, torch.Tensor):
                performance_adj = performance_adj.item()
            new_exploration = agent_type.exploration_rate * (1 + performance_adj)
            agent_type.exploration_rate = min(new_exploration, 1.0)
            
            # Adapt risk thresholds (handle tensor/float)
            if hasattr(agent_type, 'risk_threshold'):
                fairness_adj = adjustments.get('fairness_adjustment', 0.0)
                if isinstance(fairness_adj, torch.Tensor):
                    fairness_adj = fairness_adj.item()
                agent_type.risk_threshold *= (1 - fairness_adj)

# ====================== Usage Example ======================
if __name__ == "__main__":
    print("\n=== Running Agent Meta Data ===\n")
    adapter = MetricsAdapter()

    print(f"\n{adapter}")

    print("\n* * * * * Phase 2 * * * * *\n")

    control = adapter._init_control_parameters()
    print(f"\n{control}")

    print("\n* * * * * Phase 3 * * * * *\n")

    calculate = adapter._calculate_metric_deltas()
    print(f"\n{calculate}")

    print("\n* * * * * Phase 4 * * * * *\n")

    # Test safety bounds with tensor input
    test_adjustments = {'risk_threshold_adjustment': torch.tensor(0.6)}
    safe_adjustments = adapter._apply_safety_bounds(test_adjustments, ['medical'])
    printer.pretty("Safety Bounds Test:", safe_adjustments, "success")
    
    # Test full processing pipeline with valid metrics
    test_metrics = {
        'fairness': {'value': {'demographic_parity_diff': 0.15}},
        'performance': {'value': {'calibration_error': 0.1, 'accuracy': 0.85}},
        'bias': {'group_metrics': {'groupA': {'score': 0.8}, 'groupB': {'score': 0.6}}}
    }
    processed = adapter.process_metrics(test_metrics, ['medical'])
    printer.pretty("Processed Adjustments:", processed, "success")
    
    print("\n=== Successfully Ran Agent Meta Data ===\n")
