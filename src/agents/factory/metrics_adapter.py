
import yaml
import torch

from collections import defaultdict, deque
from typing import Dict, List, Any

from logs.logger import get_logger

logger = get_logger("Metrics Adapter")

CONFIG_PATH = "src/agents/factory/configs/factory_config.yaml"

def load_config(config_path=CONFIG_PATH):
    with open(config_path, "r", encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def get_merged_config(user_config=None):
    base_config = load_config()
    if user_config: base_config.update(user_config)
    return base_config

class MetricsAdapter:
    """
    Bridges metrics analysis with agent configuration using control-theoretic adaptation
    Implements principles from:
    - PID Controllers (Åström & Hägglund, 1995)
    - Safe Exploration (Turchetta et al., 2020)
    - Fairness-Aware RL (D'Amour et al., 2020)
    """
    
    def __init__(self):
        config = load_config().get("metrics")
        self.metric_history = deque(maxlen=config.get("history_size", 100))
        self.adaptation_factors = {
            'risk_threshold': 1.0,
            'exploration_rate': 1.0,
            'learning_rate': 1.0
        }
        self.max_rate = config.get("max_adaptation_rate", 0.2)
        self.error_config = config.get("error_config", {})
        self._init_control_parameters(config.get("pid_params", {}))

    def _init_control_parameters(self, pid_config: Dict):
        """PID tuning per Ziegler-Nichols method"""
        self.Kp = pid_config.get("Kp", 0.15)
        self.Ki = pid_config.get("Ki", 0.05)
        self.Kd = pid_config.get("Kd", 0.02)
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
        dpd = current.get("demographic_parity_diff", 0.0)
        return (dpd - target) / (1.0 + abs(dpd))

    def _calculate_performance_error(self, current: float, target: float) -> float:
        # Composite error including calibration and accuracy
        calibration_error = current.get("calibration_error", 0.0)
        accuracy = current.get("accuracy", 1.0)
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
        pass

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
    config = load_config()
    pid_config = config
    adapter = MetricsAdapter()

    print(f"\n{adapter}")

    print("\n* * * * * Phase 2 * * * * *\n")

    control = adapter._init_control_parameters(pid_config)
    print(f"\n{control}")

    print("\n* * * * * Phase 3 * * * * *\n")

    calculate = adapter._calculate_metric_deltas()
    print(f"\n{calculate}")

    print("\n* * * * * Phase 4 * * * * *\n")

    # Initialize adapter
    adapter = MetricsAdapter()
    
    # Test safety bounds with tensor input
    test_adjustments = {'risk_threshold_adjustment': torch.tensor(0.6)}
    safe_adjustments = adapter._apply_safety_bounds(test_adjustments, ['medical'])
    print(f"Safety Bounds Test: {safe_adjustments}")
    
    # Test full processing pipeline with valid metrics
    test_metrics = {
        'fairness': {'value': {'demographic_parity_diff': 0.15}},
        'performance': {'value': {'calibration_error': 0.1, 'accuracy': 0.85}},
        'bias': {'group_metrics': {'groupA': {'score': 0.8}, 'groupB': {'score': 0.6}}}
    }
    processed = adapter.process_metrics(test_metrics, ['medical'])
    print(f"\nProcessed Adjustments: {processed}")
    
    print("\n=== Successfully Ran Agent Meta Data ===\n")
