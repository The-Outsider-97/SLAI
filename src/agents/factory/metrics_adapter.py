"""Runtime metrics adaptation for the factory-runtime layer.

The metrics adapter translates runtime fairness, performance, and bias signals
into bounded factory-managed adjustment values. It intentionally stays inside
factory-runtime scope: it does not register agents, instantiate agents, or run
broad platform reasoning loops.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Deque, Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple

from .utils.config_loader import get_config_section, load_global_config
from .utils.factory_errors import *
from .utils.factory_helpers import *
from logs.logger import PrettyPrinter, get_logger  # pyright: ignore[reportMissingImports]

if TYPE_CHECKING:
    from ..agent_factory import AgentFactory

logger = get_logger("Metrics Adapter")
printer = PrettyPrinter()


@dataclass(slots=True)
class PIDState:
    """Mutable PID state for a single metric channel."""

    integral: float = 0.0
    previous_error: float = 0.0
    updates: int = 0

    def to_dict(self) -> Dict[str, float]:
        return {
            "integral": self.integral,
            "previous_error": self.previous_error,
            "updates": float(self.updates),
        }


@dataclass(slots=True)
class AdaptationRecord:
    """Detailed record of one metrics processing pass."""

    timestamp: str
    agent_types: Tuple[str, ...]
    errors: Dict[str, float]
    deltas: Dict[str, float]
    raw_adjustments: Dict[str, float]
    bounded_adjustments: Dict[str, float]
    clipped: Dict[str, bool]
    metric_snapshot: Dict[str, float]
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "agent_types": list(self.agent_types),
            "errors": dict(self.errors),
            "deltas": dict(self.deltas),
            "raw_adjustments": dict(self.raw_adjustments),
            "bounded_adjustments": dict(self.bounded_adjustments),
            "clipped": dict(self.clipped),
            "metric_snapshot": dict(self.metric_snapshot),
            "duration_ms": self.duration_ms,
            "metadata": safe_serialize(self.metadata, redact=True),
        }


class MetricsAdapter:
    """Bridge runtime metrics analysis with safe factory adaptation.

    ``process_metrics`` preserves the historical public contract by returning a
    dictionary of bounded adjustment values. Use ``process_metrics_result`` when
    a detailed audit record is needed.
    """

    DEFAULT_METRIC_TYPES: Tuple[str, ...] = ("fairness", "performance", "bias")

    def __init__(self) -> None:
        self.config = load_global_config()
        self.meta_config = get_config_section("metrics")
        if not isinstance(self.meta_config, MutableMapping):
            raise InvalidFactoryConfigurationError(
                "metrics config section must be a mapping",
                context={"section_type": type(self.meta_config).__name__},
                component="metrics_adapter",
                operation="load_metrics_config",
            )

        self.enabled = bool(self.meta_config.get("enabled", True))
        self.history_size = ensure_positive_int(int(self.meta_config.get("history_size", 50)), "history_size")
        self.metric_history: Deque[Mapping[str, Any]] = deque(maxlen=self.history_size)
        self.adaptation_history: Deque[AdaptationRecord] = deque(
            maxlen=ensure_positive_int(int(self.meta_config.get("adaptation_history_size", self.history_size)), "adaptation_history_size")
        )

        configured_metric_types = self.meta_config.get("metric_types", self.DEFAULT_METRIC_TYPES)
        self.metric_types = tuple(str(metric).strip() for metric in configured_metric_types if str(metric).strip())
        if not self.metric_types:
            raise InvalidFactoryConfigurationError(
                "metrics.metric_types must not be empty",
                component="metrics_adapter",
                operation="load_metrics_config",
            )

        self.error_config = dict(
            self.meta_config.get(
                "error_config",
                {"fairness_target": 0.0, "performance_target": 0.0, "bias_target": 0.0},
            )
        )
        self.pid_params = dict(self.meta_config.get("pid_params", {"Kp": 0.1, "Ki": 0.01, "Kd": 0.05}))
        self.safety_bounds = dict(validate_safety_bounds(self.meta_config.get("safety_bounds", {"default": 1.0})))
        self.adaptation_factors = dict(self.meta_config.get("adaptation_factors", {}))
        self.target_parameters = dict(self.meta_config.get("target_parameters", {}))

        self.max_adaptation_rate = ensure_non_negative_number(
            self.meta_config.get("max_adaptation_rate", 1.0),
            "max_adaptation_rate",
        )
        self.integral_limit = ensure_non_negative_number(self.meta_config.get("integral_limit", 10.0), "integral_limit")
        self.return_tensors = bool(self.meta_config.get("return_tensors", False))
        self.strict_metric_validation = bool(self.meta_config.get("strict_metric_validation", False))
        self.record_adaptation_history = bool(self.meta_config.get("record_adaptation_history", True))

        self._init_control_parameters()
        self._torch_module: Optional[Any] = None
        self._torch_import_error: Optional[Exception] = None

        logger.info("Metrics Adapter initialized")

    def _init_control_parameters(self) -> None:
        try:
            self.Kp = coerce_number(self.pid_params.get("Kp", 0.1), field_name="pid_params.Kp")
            self.Ki = coerce_number(self.pid_params.get("Ki", 0.01), field_name="pid_params.Ki")
            self.Kd = coerce_number(self.pid_params.get("Kd", 0.05), field_name="pid_params.Kd")
        except FactoryError:
            raise
        except Exception as exc:
            raise PIDControlError.from_exception(
                exc,
                message="PID parameter initialization failed",
                component="metrics_adapter",
                operation="init_control_parameters",
            ) from exc

        self.pid_state: Dict[str, PIDState] = defaultdict(PIDState)
        # Backward-compatible aliases used by older callers/tests.
        self.integral = defaultdict(float)
        self.prev_error = defaultdict(float)

    def _get_torch(self) -> Any:
        if self._torch_module is not None:
            return self._torch_module
        if self._torch_import_error is not None:
            raise TorchUnavailableError(
                "torch is unavailable for MetricsAdapter tensor output",
                cause=self._torch_import_error,
                component="metrics_adapter",
                operation="get_torch",
            )
        try:
            import torch  # type: ignore
        except Exception as exc:
            self._torch_import_error = exc
            raise TorchUnavailableError(
                "torch import failed during MetricsAdapter execution",
                cause=exc,
                component="metrics_adapter",
                operation="get_torch",
            ) from exc
        self._torch_module = torch
        return torch

    def _to_scalar(self, value: Any) -> float:
        return coerce_number(value, field_name="metric_adjustment")

    def _to_output_value(self, value: float) -> Any:
        if not self.return_tensors:
            return float(value)
        torch = self._get_torch()
        return torch.tensor(float(value), dtype=torch.float32)

    def _extract_value_mapping(self, metrics: Mapping[str, Any], metric_type: str) -> Mapping[str, Any]:
        candidate = metrics.get(metric_type, {})
        if isinstance(candidate, Mapping):
            value = candidate.get("value", candidate)
            if isinstance(value, Mapping):
                return value
        return {}

    def _metric_snapshot(self, metrics: Mapping[str, Any]) -> Dict[str, float]:
        fairness_values = self._extract_value_mapping(metrics, "fairness")
        performance_values = self._extract_value_mapping(metrics, "performance")
        bias_values = metrics.get("bias", {}) if isinstance(metrics.get("bias", {}), Mapping) else {}

        fairness_value = coerce_number(
            fairness_values.get("demographic_parity_diff", metrics.get("demographic_parity_diff", 0.0)),
            field_name="fairness.demographic_parity_diff",
        )
        calibration_error = coerce_number(
            performance_values.get("calibration_error", metrics.get("calibration_error", 0.0)),
            field_name="performance.calibration_error",
        )

        group_metrics = bias_values.get("group_metrics", {}) if isinstance(bias_values, Mapping) else {}
        bias_disparity = 0.0
        if isinstance(group_metrics, Mapping) and len(group_metrics) >= 2:
            scores = []
            for group_name, group_payload in group_metrics.items():
                if isinstance(group_payload, Mapping):
                    scores.append(coerce_number(group_payload.get("score", 0.0), field_name=f"bias.group_metrics[{group_name}].score"))
            if scores:
                bias_disparity = max(scores) - min(scores)

        return {
            "fairness": fairness_value,
            "performance": calibration_error,
            "bias": bias_disparity,
        }

    def _calculate_metric_deltas(self, current_snapshot: Optional[Mapping[str, float]] = None) -> Dict[str, float]:
        if len(self.metric_history) < 2:
            return {metric_type: 0.0 for metric_type in self.metric_types}

        current = dict(current_snapshot or self._metric_snapshot(self.metric_history[-1]))
        previous = self._metric_snapshot(self.metric_history[-2])
        return {metric_type: current.get(metric_type, 0.0) - previous.get(metric_type, 0.0) for metric_type in self.metric_types}

    def _calculate_error(self, metrics: Mapping[str, Any], metric_type: str) -> float:
        target = coerce_number(self.error_config.get(f"{metric_type}_target", 0.0), field_name=f"error_config.{metric_type}_target")
        if metric_type == "fairness":
            return self._calculate_fairness_error(self._extract_value_mapping(metrics, "fairness"), target, metrics)
        if metric_type == "performance":
            return self._calculate_performance_error(self._extract_value_mapping(metrics, "performance"), target, metrics)
        if metric_type == "bias":
            return self._calculate_bias_error(metrics, target)
        if self.strict_metric_validation:
            raise MetricsValidationError(
                f"Unknown metric type: {metric_type}",
                context={"metric_type": metric_type, "configured_metric_types": self.metric_types},
                component="metrics_adapter",
                operation="calculate_error",
            )
        return 0.0

    def _calculate_fairness_error(self, current: Mapping[str, Any], target: float, metrics: Mapping[str, Any]) -> float:
        dpd = coerce_number(current.get("demographic_parity_diff", metrics.get("demographic_parity_diff", 0.0)), field_name="fairness.demographic_parity_diff")
        return (dpd - target) / (1.0 + abs(dpd))

    def _calculate_performance_error(self, current: Mapping[str, Any], target: float, metrics: Mapping[str, Any]) -> float:
        calibration_error = coerce_number(current.get("calibration_error", metrics.get("calibration_error", 0.0)), field_name="performance.calibration_error")
        accuracy = coerce_number(current.get("accuracy", metrics.get("accuracy", 0.0)), field_name="performance.accuracy")
        accuracy = max(0.0, min(1.0, accuracy))
        return (calibration_error - target) * (1.0 - accuracy)

    def _calculate_bias_error(self, metrics: Mapping[str, Any], target: float) -> float:
        bias_payload = metrics.get("bias", {})
        group_metrics = bias_payload.get("group_metrics", {}) if isinstance(bias_payload, Mapping) else {}
        if not isinstance(group_metrics, Mapping) or len(group_metrics) < 2:
            return 0.0

        values = []
        for group_name, group_payload in group_metrics.items():
            if isinstance(group_payload, Mapping):
                values.append(coerce_number(group_payload.get("score", 0.0), field_name=f"bias.group_metrics[{group_name}].score"))
        if len(values) < 2:
            return 0.0
        max_disparity = max(values) - min(values)
        return (max_disparity - target) / (1.0 + max_disparity)

    def _pid_control(self, metric_type: str, error: float, delta: float) -> float:
        try:
            state = self.pid_state[metric_type]
            state.integral += error
            if self.integral_limit > 0:
                state.integral = max(-self.integral_limit, min(self.integral_limit, state.integral))

            derivative = error - state.previous_error
            adjustment = self.Kp * error + self.Ki * state.integral + self.Kd * derivative

            state.previous_error = error
            state.updates += 1
            self.integral[metric_type] = state.integral
            self.prev_error[metric_type] = state.previous_error
            return float(adjustment)
        except FactoryError:
            raise
        except Exception as exc:
            raise PIDControlError.from_exception(
                exc,
                message="PID control adjustment failed",
                component="metrics_adapter",
                operation="pid_control",
                context={"metric_type": metric_type, "error": error, "delta": delta},
            ) from exc

    def _effective_safety_bound(self, agent_types: Iterable[str]) -> float:
        default_bound = coerce_number(self.safety_bounds.get("default", 1.0), field_name="safety_bounds.default")
        bounds = [default_bound]
        for agent_type in agent_types:
            if agent_type in self.safety_bounds:
                bounds.append(coerce_number(self.safety_bounds[agent_type], field_name=f"safety_bounds.{agent_type}"))
        if self.max_adaptation_rate > 0:
            bounds.append(self.max_adaptation_rate)
        return max(0.0, min(bounds))

    def _apply_safety_bounds(self, adjustments: Mapping[str, Any], agent_types: Iterable[str]) -> Dict[str, Any]:
        validate_adjustment_map(adjustments)
        cleaned_agent_types = validate_agent_types(tuple(agent_types))
        effective_bound = self._effective_safety_bound(cleaned_agent_types)
        bounded: Dict[str, Any] = {}
        for key, value in adjustments.items():
            value_f = self._to_scalar(value)
            clipped = max(-effective_bound, min(effective_bound, value_f))
            bounded[str(key)] = self._to_output_value(clipped)
        return bounded

    def process_metrics(self, metrics: Mapping[str, Any], agent_types: Iterable[str]) -> Dict[str, Any]:
        """Return bounded adjustment values for the supplied runtime metrics."""
        return self.process_metrics_result(metrics, agent_types).bounded_output(self) # pyright: ignore[reportAttributeAccessIssue]

    def process_metrics_result(self, metrics: Mapping[str, Any], agent_types: Iterable[str]) -> AdaptationRecord:
        started = datetime.now(timezone.utc)
        start_ms = started.timestamp() * 1000.0

        if not self.enabled:
            agent_type_tuple = validate_agent_types(tuple(agent_types))
            record = AdaptationRecord(
                timestamp=started.isoformat(),
                agent_types=agent_type_tuple,
                errors={},
                deltas={},
                raw_adjustments={},
                bounded_adjustments={},
                clipped={},
                metric_snapshot={},
                metadata={"enabled": False},
            )
            if self.record_adaptation_history:
                self.adaptation_history.append(record)
            return record

        payload = validate_metric_payload(metrics)
        agent_type_tuple = validate_agent_types(tuple(agent_types))
        self.metric_history.append(payload)

        metric_snapshot = self._metric_snapshot(payload)
        deltas = self._calculate_metric_deltas(metric_snapshot)
        errors = {metric_type: self._calculate_error(payload, metric_type) for metric_type in self.metric_types}
        raw_adjustments = {
            f"{metric_type}_adjustment": self._pid_control(metric_type, errors[metric_type], deltas.get(metric_type, 0.0))
            for metric_type in self.metric_types
        }

        effective_bound = self._effective_safety_bound(agent_type_tuple)
        bounded_adjustments: Dict[str, float] = {}
        clipped: Dict[str, bool] = {}
        for key, value in raw_adjustments.items():
            bounded_value = max(-effective_bound, min(effective_bound, value))
            bounded_adjustments[key] = bounded_value
            clipped[key] = bounded_value != value

        duration_ms = max(0.0, datetime.now(timezone.utc).timestamp() * 1000.0 - start_ms)
        record = AdaptationRecord(
            timestamp=started.isoformat(),
            agent_types=agent_type_tuple,
            errors=errors,
            deltas=deltas,
            raw_adjustments=raw_adjustments,
            bounded_adjustments=bounded_adjustments,
            clipped=clipped,
            metric_snapshot=metric_snapshot,
            duration_ms=duration_ms,
            metadata={"effective_bound": effective_bound, "return_tensors": self.return_tensors},
        )
        if self.record_adaptation_history:
            self.adaptation_history.append(record)
        return record

    def update_factory_config(self, factory: "AgentFactory", adjustments: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Apply bounded adjustment values to factory-managed metadata fields.

        The method updates only attributes that already exist on metadata or keys
        present in ``metadata.metadata``. This keeps the adapter conservative and
        avoids inventing runtime configuration fields at call time.
        """
        validate_factory_object(factory)
        validate_adjustment_map(adjustments)

        performance_adj = self._to_scalar(adjustments.get("performance_adjustment", 0.0))
        fairness_adj = self._to_scalar(adjustments.get("fairness_adjustment", 0.0))
        bias_adj = self._to_scalar(adjustments.get("bias_adjustment", 0.0))

        update_report: Dict[str, Dict[str, Any]] = {}
        agents = getattr(factory.registry, "agents", {})
        for agent_name, metadata in agents.items():
            agent_report: Dict[str, Any] = {}
            self._update_numeric_metadata_field(metadata, "exploration_rate", performance_adj, agent_report)
            self._update_numeric_metadata_field(metadata, "learning_rate", performance_adj, agent_report)
            self._update_numeric_metadata_field(metadata, "risk_threshold", -fairness_adj, agent_report)
            self._update_numeric_metadata_field(metadata, "bias_threshold", -bias_adj, agent_report)
            if agent_report:
                update_report[str(agent_name)] = agent_report
        return update_report

    def _update_numeric_metadata_field(self, metadata: Any, field_name: str, adjustment: float, report: Dict[str, Any]) -> None:
        factor = coerce_number(self.adaptation_factors.get(field_name, 1.0), field_name=f"adaptation_factors.{field_name}")
        scaled_adjustment = adjustment * factor
        bounds = self.target_parameters.get(field_name, {})
        min_value = coerce_number(bounds.get("min", 0.0), field_name=f"target_parameters.{field_name}.min") if isinstance(bounds, Mapping) else 0.0
        max_value = coerce_number(bounds.get("max", 1.0), field_name=f"target_parameters.{field_name}.max") if isinstance(bounds, Mapping) else 1.0

        if hasattr(metadata, field_name):
            current = coerce_number(getattr(metadata, field_name), field_name=field_name)
            updated = max(min_value, min(max_value, current * (1.0 + scaled_adjustment)))
            setattr(metadata, field_name, updated)
            report[field_name] = {"previous": current, "updated": updated, "source": "attribute"}
            return

        metadata_payload = getattr(metadata, "metadata", None)
        if isinstance(metadata_payload, MutableMapping) and field_name in metadata_payload:
            current = coerce_number(metadata_payload[field_name], field_name=f"metadata.{field_name}")
            updated = max(min_value, min(max_value, current * (1.0 + scaled_adjustment)))
            metadata_payload[field_name] = updated
            report[field_name] = {"previous": current, "updated": updated, "source": "metadata"}

    def history_snapshot(self, *, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        records = list(self.adaptation_history)
        if limit is not None:
            records = records[-max(0, int(limit)) :]
        return [record.to_dict() for record in records]

    def state_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "history_size": self.history_size,
            "metric_types": list(self.metric_types),
            "pid_params": {"Kp": self.Kp, "Ki": self.Ki, "Kd": self.Kd},
            "pid_state": {metric: state.to_dict() for metric, state in self.pid_state.items()},
            "safety_bounds": dict(self.safety_bounds),
            "metric_history_size": len(self.metric_history),
            "adaptation_history_size": len(self.adaptation_history),
            "return_tensors": self.return_tensors,
        }

    def health_check(self) -> Dict[str, Any]:
        return {
            "status": "ok" if self.enabled else "disabled",
            "configured_metric_types": list(self.metric_types),
            "history_size": self.history_size,
            "metric_history_size": len(self.metric_history),
            "adaptation_history_size": len(self.adaptation_history),
            "torch_available": self._torch_module is not None or self._torch_import_error is None,
        }

    def reset(self) -> None:
        self.metric_history.clear()
        self.adaptation_history.clear()
        self._init_control_parameters()


def _record_bounded_output(self: AdaptationRecord, adapter: MetricsAdapter) -> Dict[str, Any]:
    return {key: adapter._to_output_value(value) for key, value in self.bounded_adjustments.items()}


# Attach as a method without creating another dataclass solely for output values.
AdaptationRecord.bounded_output = _record_bounded_output  # type: ignore[attr-defined]


if __name__ == "__main__":
    print("\n=== Running Metrics Adapter ===\n")
    printer.status("TEST", "Metrics Adapter initialized", "info")

    adapter = MetricsAdapter()
    sample_metrics = {
        "fairness": {"value": {"demographic_parity_diff": 0.12}},
        "performance": {"value": {"calibration_error": 0.08, "accuracy": 0.91}},
        "bias": {"group_metrics": {"group_a": {"score": 0.2}, "group_b": {"score": 0.35}}},
    }
    adjustments = adapter.process_metrics(sample_metrics, ["default"])
    assert "fairness_adjustment" in adjustments
    assert "performance_adjustment" in adjustments
    assert "bias_adjustment" in adjustments
    assert adapter.health_check()["metric_history_size"] == 1
    assert adapter.history_snapshot(limit=1)
    adapter.reset()
    assert adapter.health_check()["metric_history_size"] == 0

    print("\n=== Test ran successfully ===\n")
