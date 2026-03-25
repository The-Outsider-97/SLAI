from collections import defaultdict, deque
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch

from src.agents.factory.utils.config_loader import get_config_section, load_global_config
from logs.logger import PrettyPrinter, get_logger

if TYPE_CHECKING:
    from src.agents.agent_factory import AgentFactory

logger = get_logger("Metrics Adapter")
printer = PrettyPrinter


class MetricsAdapter:
    """Bridge metrics analysis with configuration adaptation."""

    def __init__(self):
        self.config = load_global_config()
        self.meta_config = get_config_section("metrics")

        self.history_size = self.meta_config.get("history_size", 50)
        self.metric_history = deque(maxlen=self.history_size)

        self.error_config = self.meta_config.get(
            "error_config",
            {
                "fairness_target": 0.0,
                "performance_target": 0.0,
                "bias_target": 0.0,
            },
        )
        self.pid_params = self.meta_config.get("pid_params", {"Kp": 0.1, "Ki": 0.01, "Kd": 0.05})
        self.safety_bounds = self.meta_config.get("safety_bounds", {"default": 1.0})

        self._init_control_parameters()
        self._torch_module: Optional[Any] = None
        self._torch_import_error: Optional[Exception] = None

    def _init_control_parameters(self):
        self.Kp = self.pid_params["Kp"]
        self.Ki = self.pid_params["Ki"]
        self.Kd = self.pid_params["Kd"]
        self.integral = defaultdict(float)
        self.prev_error = defaultdict(float)

    def _calculate_metric_deltas(self) -> Dict[str, float]:
        if len(self.metric_history) < 2:
            return {}

        current = self.metric_history[-1]
        previous = self.metric_history[-2]

        return {
            "fairness": current.get("demographic_parity", 0) - previous.get("demographic_parity", 0),
            "performance": current.get("calibration_error", 0) - previous.get("calibration_error", 0),
        }

    def _calculate_error(self, metrics: Dict[str, Any], metric_type: str) -> float:
        target = self.error_config.get(f"{metric_type}_target", 0.0)
        current = metrics.get(metric_type, {}).get("value", {})

        try:
            if metric_type == "fairness":
                return self._calculate_fairness_error(current, target)
            if metric_type == "performance":
                return self._calculate_performance_error(current, target)
            if metric_type == "bias":
                return self._calculate_bias_error(metrics, target)
            raise ValueError(f"Unknown metric type: {metric_type}")
        except KeyError as e:
            logger.warning(f"Missing metric data for {metric_type}: {str(e)}")
            return 0.0

    def _calculate_fairness_error(self, current: Dict[str, float], target: float) -> float:
        dpd = current.get("demographic_parity_diff", 0.0)
        return (dpd - target) / (1.0 + abs(dpd))

    def _calculate_performance_error(self, current: Dict[str, float], target: float) -> float:
        calibration_error = current.get("calibration_error", 0.0)
        accuracy = current.get("accuracy", 0.0)
        return (calibration_error - target) * (1.0 - accuracy)

    def _calculate_bias_error(self, metrics: Dict[str, Any], target: float) -> float:
        group_metrics = metrics.get("bias", {}).get("group_metrics", {})
        if len(group_metrics) < 2:
            return 0.0

        values = [v.get("score", 0.0) for v in group_metrics.values()]
        max_disparity = max(values) - min(values)
        return (max_disparity - target) / (1.0 + max_disparity)

    def _get_torch(self) -> Any:
        if self._torch_module is not None:
            return self._torch_module
        if self._torch_import_error is not None:
            raise RuntimeError("torch is unavailable for MetricsAdapter operations") from self._torch_import_error
        try:
            import torch  # type: ignore
        except Exception as exc:
            self._torch_import_error = exc
            raise RuntimeError("torch import failed during MetricsAdapter execution") from exc
        self._torch_module = torch
        return torch

    def _to_scalar(self, value: Any) -> float:
        torch = self._torch_module
        if torch is not None and isinstance(value, torch.Tensor):
            return float(value.item())
        if hasattr(value, "item") and callable(value.item):
            return float(value.item())
        return float(value)

    def process_metrics(self, metrics: Dict[str, Any], agent_types: List[str]) -> Dict[str, Any]:
        self.metric_history.append(metrics)
        delta = self._calculate_metric_deltas()

        adjustments = {}
        for metric_type in ["fairness", "performance", "bias"]:
            error = self._calculate_error(metrics, metric_type)
            adjustments.update(self._pid_control(metric_type, error, delta.get(metric_type, 0.0)))

        return self._apply_safety_bounds(adjustments, agent_types)

    def _apply_safety_bounds(self, adjustments: Dict[str, Any], agent_types: List[str]):
        torch = self._get_torch()
        for agent_type in agent_types:
            bound = self.safety_bounds.get(agent_type, self.safety_bounds.get("default", 1.0))
            for key, value in adjustments.items():
                value_f = self._to_scalar(value)
                if abs(value_f) > bound:
                    adjustments[key] = torch.tensor(bound * (1 if value_f > 0 else -1), dtype=torch.float32)
        return adjustments

    def _pid_control(self, metric_type: str, error: float, delta: float) -> Dict[str, Any]:
        torch = self._get_torch()
        self.integral[metric_type] += error
        derivative = error - self.prev_error[metric_type]

        adjustment = self.Kp * error + self.Ki * self.integral[metric_type] + self.Kd * derivative
        adjustment_tensor = torch.tensor(adjustment, dtype=torch.float32)

        self.prev_error[metric_type] = error
        return {f"{metric_type}_adjustment": adjustment_tensor}

    def update_factory_config(self, factory: "AgentFactory", adjustments: Dict[str, Any]):
        torch = self._get_torch()
        for metadata in factory.registry.agents.values():
            performance_adj = adjustments.get("performance_adjustment", torch.tensor(0.0))
            performance_adj = self._to_scalar(performance_adj)

            if hasattr(metadata, "exploration_rate"):
                metadata.exploration_rate = min(metadata.exploration_rate * (1 + performance_adj), 1.0)

            fairness_adj = adjustments.get("fairness_adjustment", torch.tensor(0.0))
            fairness_adj = self._to_scalar(fairness_adj)
            if hasattr(metadata, "risk_threshold"):
                metadata.risk_threshold *= (1 - fairness_adj)
