"""Production-hardened recovery orchestration for learning agents.

This module coordinates escalating recovery actions for SLAI learning agents
without owning training logic. It relies on the learning subsystem's existing
configuration loader, error taxonomy, calculations, and helpers for consistent
validation, diagnostics, and recovery telemetry.
"""

from __future__ import annotations

import copy
import inspect
import time
import torch # pyright: ignore[reportMissingImports]

from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union, cast

from ..utils.config_loader import load_global_config, get_config_section
from ..utils.learning_error import *
from ..utils.learning_calculations import *
from ..utils.learning_helpers import *
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Recovery System")
printer = PrettyPrinter()


class RecoverySystem:
    """Coordinate escalating recovery actions for the learning agent stack.

    The system preserves the original recovery chain — soft reset, learning-rate
    adjustment, architecture rollback, strategy switch, and full reset — while
    making the execution path safer and more inspectable. It supports single
    agents, parent orchestrators with an ``agents`` mapping, PyTorch-style
    modules/optimizers, custom SLAI models exposing checkpoint APIs, and agents
    with lightweight scalar control attributes.
    """

    STRATEGY_ORDER: Tuple[str, ...] = (
        "soft_reset",
        "lr_adjustment",
        "architecture_rollback",
        "strategy_switch",
        "full_reset",
    )

    def __init__(self, learning_agent: Any, time_fn: Optional[Callable[[], float]] = None) -> None:
        if learning_agent is None:
            raise RecoveryError("RecoverySystem requires a learning_agent instance")

        self.learning_agent = learning_agent
        self.config = load_global_config()
        self.recover_config = get_config_section("recovery_system") or {}

        validate_probability(coerce_float(self.recover_config.get("error_decay_factor", 0.5)), "recovery_system.error_decay_factor")
        validate_positive(coerce_float(self.recover_config.get("error_decay_time", 3600.0)), "recovery_system.error_decay_time", strict=False)
        validate_positive(coerce_int(self.recover_config.get("max_recovery_attempts", 5), 5), "recovery_system.max_recovery_attempts", strict=True)
        validate_positive(coerce_int(self.recover_config.get("max_snapshots", 5), 5), "recovery_system.max_snapshots", strict=True)
        validate_in_range(coerce_float(self.recover_config.get("lr_reduction_factor", 0.5)), "recovery_system.lr_reduction_factor", 0.0, 1.0, inclusive_low=False)
        validate_positive(coerce_float(self.recover_config.get("min_learning_rate", 1e-6)), "recovery_system.min_learning_rate", strict=True)
        validate_positive(coerce_float(self.recover_config.get("max_learning_rate", 1.0)), "recovery_system.max_learning_rate", strict=True)

        self._time_fn = time_fn or time.time
        self.calculations = LearningCalculations()

        self.error_count = 0
        self.last_error_time = self._time_fn()
        self.recovery_history: List[Dict[str, Any]] = []
        self.stable_snapshots: List[Dict[str, Any]] = []
        self.error_events: Deque[Dict[str, Any]] = deque(maxlen=coerce_int(self.recover_config.get("max_error_events", 100), 100, minimum=1))
        self.error_stats = RunningStats()
        self.recovery_stats = RunningStats()

        self.recovery_strategies: Dict[str, Callable[[], Dict[str, Any]]] = {
            "soft_reset": self._recover_soft_reset,
            "lr_adjustment": self._recover_learning_rate_adjustment,
            "architecture_rollback": self._recover_architecture_rollback,
            "strategy_switch": self._recover_strategy_switch,
            "full_reset": self._recover_full_reset,
        }
        self.error_thresholds = self._normalise_thresholds(self.recover_config.get("error_thresholds", True))
        validate_non_empty_sequence(self.error_thresholds, "recovery_system.error_thresholds")

    @staticmethod
    def _normalise_thresholds(thresholds: Iterable[int]) -> List[int]:
        try:
            normalised = sorted({int(t) for t in thresholds if int(t) >= 0})
        except TypeError as exc:
            raise InvalidConfigError("error_thresholds must be an iterable of integers", config_key="recovery_system.error_thresholds", cause=exc) from exc
        return normalised or [3, 6, 9, 12]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def decay_error_count(self) -> int:
        """Decay accumulated error count after configured quiet intervals."""
        current_time = self._time_fn()
        elapsed = max(0.0, current_time - self.last_error_time)
        decay_time = coerce_float(self.recover_config.get("error_decay_time", 3600.0), 3600.0, minimum=0.0)
        if decay_time <= 0.0 or self.error_count <= 0:
            return self.error_count

        intervals = int(elapsed // decay_time)
        if intervals > 0:
            decay_factor = clamp(coerce_float(self.recover_config.get("error_decay_factor", 0.5), 0.5), 0.0, 1.0)
            decayed = float(self.error_count)
            for _ in range(intervals):
                decayed *= decay_factor
            self.error_count = max(0, int(round(decayed)))
            self.last_error_time = current_time
        return self.error_count

    def increment_error_count(self, count: int = 1, error: Optional[BaseException] = None) -> int:
        """Register one or more failures with configured severity weighting."""
        validate_non_negative(count, "count")
        self.decay_error_count()
        severity = self._classify_error_severity(error) if error is not None else int(count)
        severity = max(int(count), severity)
        self.error_count += severity
        self.last_error_time = self._time_fn()
        self.error_stats.update(float(severity))
        self.calculations.update_performance(float(-severity))
        if error is not None:
            self._record_error_event(error, severity)
            logger.warning("Error count increased to %s due to %s", self.error_count, type(error).__name__)
        else:
            logger.warning("Error count increased to %s", self.error_count)
        return self.error_count

    def register_stable_snapshot(
        self,
        label: Optional[str] = None,
        include_buffers: bool = False,
        snapshot_type: str = "stable",          # new parameter
    ) -> Optional[Dict[str, Any]]:
        """Capture a stable rollback snapshot for one or more learning agents."""
        snapshot = self._build_snapshot(include_buffers=include_buffers)
        if snapshot is None:
            logger.warning("No stable snapshot could be created.")
            return None
    
        snapshot_record = {
            "id": make_learning_id("recovery_snapshot"),
            "label": label or f"snapshot_{len(self.stable_snapshots)}",
            "type": snapshot_type,               # store the type
            "created_at": self._time_fn(),
            "hash": stable_hash(snapshot),
            "snapshot": snapshot,
        }
        self.stable_snapshots.append(snapshot_record)
        max_snapshots = coerce_int(self.recover_config.get("max_snapshots", 5), 5, minimum=1)
        if len(self.stable_snapshots) > max_snapshots:
            self.stable_snapshots = self.stable_snapshots[-max_snapshots:]
        logger.info("Stable recovery snapshot recorded: %s", snapshot_record["label"])
        return to_json_safe(snapshot_record)

    def execute_recovery(self, error: Optional[BaseException] = None, force_strategy: Optional[str] = None) -> Dict[str, Any]:
        """Execute the selected recovery strategy and return a structured result."""
        if not coerce_bool(self.recover_config.get("enabled", True)):
            return self._record_recovery_result({"status": "skipped", "strategy": "disabled", "details": {"reason": "disabled"}}, error)

        if error is not None:
            self.increment_error_count(error=error)
        else:
            self.decay_error_count()

        if coerce_bool(self.recover_config.get("snapshot_on_recovery", True)):
            self.register_stable_snapshot(label="pre_recovery", snapshot_type="pre_recovery")

        strategy_name = force_strategy or self._select_strategy_name()
        if strategy_name not in self.recovery_strategies:
            raise UnknownStrategyError(strategy_name, valid_strategies=list(self.recovery_strategies))

        start = self._time_fn()
        logger.warning("Executing recovery strategy %s at error_count=%s", strategy_name, self.error_count)
        try:
            result = self.recovery_strategies[strategy_name]()
            result.setdefault("status", "recovered")
        except Exception as exc:
            result = self._handle_strategy_failure(strategy_name, exc, error)

        result.setdefault("strategy", strategy_name)
        result.setdefault("duration_seconds", max(0.0, self._time_fn() - start))
        return self._record_recovery_result(result, error)

    def reset_error_count(self) -> None:
        """Reset failure tracking after a successful healthy period."""
        self.error_count = 0
        self.last_error_time = self._time_fn()
        self.calculations.update_performance(1.0)
        logger.info("Error count reset after successful recovery")

    def diagnostics(self) -> Dict[str, Any]:
        """Return current recovery health and telemetry."""
        recent_statuses = [entry.get("status") for entry in self.recovery_history[-10:]]
        recent_successes = sum(1 for status in recent_statuses if status == "recovered")
        success_rate = safe_divide(float(recent_successes), float(len(recent_statuses)), 0.0)
        error_values = [float(event.get("severity", 0.0)) for event in self.error_events]
        moving = self.calculations.moving_average(error_values, window_size=max(1, min(5, len(error_values)))) if error_values else []
        return {
            "enabled": coerce_bool(self.recover_config.get("enabled", True)),
            "error_count": self.error_count,
            "error_thresholds": list(self.error_thresholds),
            "selected_strategy": self._select_strategy_name(),
            "stable_snapshots": len(self.stable_snapshots),
            "history_size": len(self.recovery_history),
            "recent_success_rate": round_float(success_rate),
            "error_stats": asdict(self.error_stats.snapshot()),
            "recovery_duration_stats": asdict(self.recovery_stats.snapshot()),
            "performance_trend": round_float(self.calculations.calculate_performance_trend()),
            "recent_error_moving_average": [round_float(v) for v in moving[-5:]],
        }

    def save_state(self, path: Union[str, Path]) -> Path:
        """Save RecoverySystem bookkeeping state; does not serialize live agents."""
        path = Path(path)
        payload = {
            "error_count": self.error_count,
            "last_error_time": self.last_error_time,
            "recovery_history": to_json_safe(self.recovery_history, max_depth=8, max_items=256),
            "stable_snapshots": self.stable_snapshots,
            "error_events": list(self.error_events),
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, path)
            return path
        except Exception as exc:
            raise CheckpointError(str(path), operation="save", cause=exc) from exc

    def load_state(self, path: Union[str, Path], map_location: Optional[Any] = None) -> Dict[str, Any]:
        """Load RecoverySystem bookkeeping state."""
        path = Path(path)
        try:
            payload = torch.load(path, map_location=map_location)
            if not isinstance(payload, Mapping):
                raise CheckpointError(str(path), operation="load", message="Recovery checkpoint must be a mapping")
            self.error_count = int(payload.get("error_count", self.error_count))
            self.last_error_time = float(payload.get("last_error_time", self.last_error_time))
            self.recovery_history = list(payload.get("recovery_history", []))
            self.stable_snapshots = list(payload.get("stable_snapshots", []))
            self.error_events = deque(payload.get("error_events", []), maxlen=self.error_events.maxlen)
            return dict(payload)
        except CheckpointError:
            raise
        except Exception as exc:
            raise CheckpointError(str(path), operation="load", cause=exc) from exc

    # ------------------------------------------------------------------
    # Error classification and strategy selection
    # ------------------------------------------------------------------
    def _classify_error_severity(self, error: Optional[BaseException]) -> int:
        if error is None:
            return 1
        weights = self.recover_config.get("error_severity_weights")
        if not isinstance(weights, Mapping):
            weights = self.recover_config["error_severity_weights"]
        for cls in type(error).__mro__:
            if cls.__name__ in weights:
                return max(1, int(weights[cls.__name__]))
        return max(1, int(weights.get("Exception", 1)))

    def _record_error_event(self, error: BaseException, severity: int) -> None:
        event = {
            "id": make_learning_id("recovery_error"),
            "timestamp": self._time_fn(),
            "type": type(error).__name__,
            "severity": severity,
            "message": str(error),
        }
        if isinstance(error, LearningError):
            event["structured"] = error.to_dict()
        elif coerce_bool(self.recover_config.get("record_exception_trace", True)):
            event["traceback"] = format_exception_chain(error)
        self.error_events.append(to_json_safe(event))

    def _select_strategy_name(self) -> str:
        level = 0
        for index, threshold in enumerate(self.error_thresholds):
            if self.error_count >= threshold:
                level = index + 1
        level = min(level, len(self.STRATEGY_ORDER) - 1)
        return self.STRATEGY_ORDER[level]

    def _handle_strategy_failure(self, strategy_name: str, exc: BaseException, original_error: Optional[BaseException]) -> Dict[str, Any]:
        logger.exception("Recovery strategy %s failed: %s", strategy_name, exc)
        attempts = len(self.recovery_history) + 1
        max_attempts = coerce_int(self.recover_config.get("max_recovery_attempts", 5), 5, minimum=1)
        if attempts >= max_attempts:
            raise RecoveryExhaustedError(max_attempts=max_attempts, last_error=exc) from exc

        current_index = self.STRATEGY_ORDER.index(strategy_name) if strategy_name in self.STRATEGY_ORDER else 0
        next_index = min(current_index + 1, len(self.STRATEGY_ORDER) - 1)
        fallback_name = self.STRATEGY_ORDER[next_index]
        try:
            fallback = self.recovery_strategies[fallback_name]()
            fallback["strategy"] = fallback_name
            fallback["details"] = {**fallback.get("details", {}), "fallback_from": strategy_name, "failure": format_exception_chain(exc)}
            return fallback
        except Exception as fallback_exc:
            raise RecoveryError(
                "Recovery strategy and fallback both failed",
                context={"strategy": strategy_name, "fallback": fallback_name, "original_error": repr(original_error)},
                cause=fallback_exc,
            ) from fallback_exc

    def _record_recovery_result(self, result: Dict[str, Any], error: Optional[BaseException]) -> Dict[str, Any]:
        result.setdefault("id", make_learning_id("recovery"))
        result.setdefault("timestamp", self._time_fn())
        result.setdefault("error_count", self.error_count)
        result.setdefault("selected_level", self.STRATEGY_ORDER.index(result.get("strategy", "soft_reset")) + 1 if result.get("strategy") in self.STRATEGY_ORDER else 0)
        if error is not None:
            result.setdefault("trigger_error", {"type": type(error).__name__, "message": str(error)})
        result = to_json_safe(result, max_depth=8, max_items=256)
        self.recovery_history.append(result)
        self.recovery_stats.update(float(result.get("duration_seconds", 0.0) or 0.0))
        self.calculations.update_performance(1.0 if result.get("status") == "recovered" else -1.0)
        return result

    # ------------------------------------------------------------------
    # Agent/module iteration
    # ------------------------------------------------------------------
    def _iter_named_agents(self) -> Iterable[Tuple[str, Any]]:
        agents = getattr(self.learning_agent, "agents", None)
        if isinstance(agents, Mapping):
            for name, agent in agents.items():
                yield str(name), agent
            return
        if isinstance(agents, (list, tuple, set)):
            for index, agent in enumerate(agents):
                yield getattr(agent, "agent_id", f"agent_{index}"), agent
            return
        yield getattr(self.learning_agent, "agent_id", type(self.learning_agent).__name__), self.learning_agent

    def _iter_agents(self) -> Iterable[Any]:
        for _, agent in self._iter_named_agents():
            yield agent

    def _iter_network_modules(self, agent: Any) -> Iterable[Tuple[str, Any]]:
        for attribute in self.recover_config.get("network_attributes", {}):
            module = getattr(agent, attribute, None)
            if module is not None:
                yield str(attribute), module

    def _iter_optimizers(self, agent: Any) -> Iterable[Tuple[str, Any]]:
        for attribute in self.recover_config.get("optimizer_attributes", {}):
            optimizer = getattr(agent, attribute, None)
            if optimizer is not None:
                yield str(attribute), optimizer

    # ------------------------------------------------------------------
    # Recovery strategies
    # ------------------------------------------------------------------
    def _recover_soft_reset(self) -> Dict[str, Any]:
        reset_networks: List[str] = []
        cleared_buffers: List[str] = []
        exploration_updates: List[Dict[str, Any]] = []

        for agent_name, agent in self._iter_named_agents():
            if coerce_bool(self.recover_config.get("reset_networks_on_soft_reset", True)):
                for attribute, module in self._iter_network_modules(agent):
                    if hasattr(module, "apply") and callable(module.apply):
                        module.apply(self._reset_module_parameters)
                        reset_networks.append(f"{agent_name}.{attribute}")
                    elif hasattr(module, "reset_parameters") and callable(module.reset_parameters):
                        module.reset_parameters()
                        reset_networks.append(f"{agent_name}.{attribute}")

            if coerce_bool(self.recover_config.get("clear_buffers_on_soft_reset", True)):
                cleared_buffers.extend(f"{agent_name}.{name}" for name in self._clear_buffers(agent))

            for epsilon_attr in ("epsilon", "exploration_rate"):
                if hasattr(agent, epsilon_attr):
                    old_value = coerce_float(getattr(agent, epsilon_attr), 0.0, minimum=0.0)
                    factor = coerce_float(self.recover_config.get("reset_exploration_factor", 1.5), 1.5, minimum=0.0)
                    maximum = coerce_float(self.recover_config.get("max_exploration_rate", 1.0), 1.0, minimum=0.0)
                    new_value = clamp(old_value * factor if old_value > 0.0 else maximum, 0.0, maximum)
                    setattr(agent, epsilon_attr, new_value)
                    exploration_updates.append({"agent": agent_name, "attribute": epsilon_attr, "old": old_value, "new": new_value})

        return {"status": "recovered", "strategy": "soft_reset", "details": {
            "reset_networks": sorted(set(reset_networks)),
            "cleared_buffers": sorted(set(cleared_buffers)),
            "exploration_updates": exploration_updates,
        }}

    @staticmethod
    def _reset_module_parameters(module: Any) -> None:
        if hasattr(module, "reset_parameters") and callable(module.reset_parameters):
            module.reset_parameters()

    def _recover_learning_rate_adjustment(self) -> Dict[str, Any]:
        reduction = coerce_float(self.recover_config.get("lr_reduction_factor", 0.5), 0.5, minimum=1e-12, maximum=1.0)
        min_lr = coerce_float(self.recover_config.get("min_learning_rate", 1e-6), 1e-6, minimum=0.0)
        max_lr = coerce_float(self.recover_config.get("max_learning_rate", 1.0), 1.0, minimum=min_lr)
        adjustments: List[Dict[str, Any]] = []

        for agent_name, agent in self._iter_named_agents():
            if hasattr(agent, "learning_rate"):
                old_lr = coerce_float(getattr(agent, "learning_rate"), min_lr, minimum=0.0)
                new_lr = clamp(old_lr * reduction, min_lr, max_lr)
                setattr(agent, "learning_rate", new_lr)
                adjustments.append({"target": f"{agent_name}.learning_rate", "old_lr": old_lr, "new_lr": new_lr})

            for optimizer_name, optimizer in self._iter_optimizers(agent):
                adjustments.extend(self._adjust_optimizer_lr(f"{agent_name}.{optimizer_name}", optimizer, reduction, min_lr, max_lr))

        return {"status": "recovered", "strategy": "lr_adjustment", "details": {
            "adjustments": adjustments,
            "reduction_factor": reduction,
            "min_learning_rate": min_lr,
        }}

    def _recover_architecture_rollback(self) -> Dict[str, Any]:
        snapshot = self._select_rollback_snapshot()
        if snapshot is None:
            fallback = self._recover_strategy_switch()
            fallback["details"] = {**fallback.get("details", {}), "fallback": "strategy_switch", "reason": "missing_snapshot"}
            return fallback

        restored_agents: List[str] = []
        failed_agents: Dict[str, str] = {}
        snapshot_agents = snapshot.get("agents") if isinstance(snapshot, Mapping) else None
        if not isinstance(snapshot_agents, Mapping):
            raise RecoveryError("Rollback snapshot has invalid structure", context={"keys": list(snapshot.keys()) if isinstance(snapshot, Mapping) else type(snapshot).__name__})

        live_agents = dict(self._iter_named_agents())
        for agent_id, agent_state in snapshot_agents.items():
            agent = live_agents.get(str(agent_id))
            if agent is None:
                failed_agents[str(agent_id)] = "missing live agent"
                continue
            try:
                self._restore_agent_state(agent, agent_state)
                restored_agents.append(str(agent_id))
            except Exception as exc:
                failed_agents[str(agent_id)] = format_exception_chain(exc)

        if not restored_agents and failed_agents:
            raise RecoveryError("No agents could be restored from rollback snapshot", context={"failed_agents": failed_agents})

        return {"status": "recovered", "strategy": "architecture_rollback", "details": {
            "restored_agents": restored_agents,
            "failed_agents": failed_agents,
            "snapshot_hash": stable_hash(snapshot),
        }}

    def _recover_strategy_switch(self) -> Dict[str, Any]:
        safe_strategy = str(self.recover_config.get("safe_strategy", "rl"))
        updates: Dict[str, Any] = {}
        for attr in ("active_strategy", "current_strategy", "selected_strategy"):
            if hasattr(self.learning_agent, attr):
                previous = getattr(self.learning_agent, attr)
                setattr(self.learning_agent, attr, safe_strategy)
                updates[attr] = {"previous": previous, "new": safe_strategy}
        if not updates:
            setattr(self.learning_agent, "active_strategy", safe_strategy)
            updates["active_strategy"] = {"previous": None, "new": safe_strategy}

        guard_result = None
        safety_guard = getattr(self.learning_agent, "safety_guard", None)
        if safety_guard is not None and hasattr(safety_guard, "execute"):
            guard_result = safety_guard.execute({"task": "emergency_override", "strategy": safe_strategy})

        return {"status": "recovered", "strategy": "strategy_switch", "details": {
            "updates": updates,
            "safety_guard_result": to_json_safe(guard_result),
        }}

    def _recover_full_reset(self) -> Dict[str, Any]:
        if hasattr(self.learning_agent, "reset") and callable(self.learning_agent.reset):
            self.learning_agent.reset()
            return {"status": "recovered", "strategy": "full_reset", "details": {"mode": "reset_hook"}}
        reconstruction = self._attempt_reinitialisation()
        return {"status": "recovered", "strategy": "full_reset", "details": reconstruction}

    # ------------------------------------------------------------------
    # Strategy internals
    # ------------------------------------------------------------------
    def _clear_buffers(self, owner: Any) -> List[str]:
        cleared: List[str] = []
        for attribute in self.recover_config.get("reset_buffer_attributes", {}):
            value = getattr(owner, attribute, None)
            if value is not None and hasattr(value, "clear") and callable(value.clear):
                value.clear()
                cleared.append(str(attribute))
            elif isinstance(value, list):
                value[:] = []
                cleared.append(str(attribute))
        return cleared

    @staticmethod
    def _adjust_optimizer_lr(name: str, optimizer: Any, reduction: float, min_lr: float, max_lr: float) -> List[Dict[str, Any]]:
        adjustments: List[Dict[str, Any]] = []
        if hasattr(optimizer, "param_groups"):
            for index, group in enumerate(optimizer.param_groups):
                old_lr = coerce_float(group.get("lr", min_lr), min_lr, minimum=0.0)
                new_lr = clamp(old_lr * reduction, min_lr, max_lr)
                group["lr"] = new_lr
                adjustments.append({"target": f"{name}.param_groups[{index}]", "old_lr": old_lr, "new_lr": new_lr})
        if hasattr(optimizer, "learning_rate"):
            old_lr = coerce_float(getattr(optimizer, "learning_rate"), min_lr, minimum=0.0)
            new_lr = clamp(old_lr * reduction, min_lr, max_lr)
            setattr(optimizer, "learning_rate", new_lr)
            adjustments.append({"target": f"{name}.learning_rate", "old_lr": old_lr, "new_lr": new_lr})
        return adjustments

    def _select_rollback_snapshot(self) -> Optional[Dict[str, Any]]:
        # 1. Check agent-specific architecture history
        history = getattr(self.learning_agent, "architecture_history", None)
        if isinstance(history, Sequence) and history:
            candidate = history[-1]
            if isinstance(candidate, Mapping) and "snapshot" in candidate:
                return cast(Dict[str, Any], copy.deepcopy(candidate["snapshot"]))
            if isinstance(candidate, Mapping):
                return cast(Dict[str, Any], copy.deepcopy(candidate))
    
        # 2. Search backwards for a stable (non-pre_recovery) snapshot
        for record in reversed(self.stable_snapshots):
            if record.get("type") != "pre_recovery":
                snapshot = record.get("snapshot")
                if isinstance(snapshot, Mapping):
                    return cast(Dict[str, Any], copy.deepcopy(snapshot))
                # Fallback: treat as dict if possible
                return cast(Dict[str, Any], copy.deepcopy(snapshot))
    
        # 3. Fallback: last snapshot of any type
        if self.stable_snapshots:
            last_snapshot = self.stable_snapshots[-1].get("snapshot")
            if isinstance(last_snapshot, Mapping):
                return cast(Dict[str, Any], copy.deepcopy(last_snapshot))
            return cast(Dict[str, Any], copy.deepcopy(last_snapshot))
        return None

    def _restore_agent_state(self, agent: Any, agent_snapshot: Mapping[str, Any]) -> None:
        if not isinstance(agent_snapshot, Mapping):
            raise RecoveryError("agent_snapshot must be a mapping", context={"actual_type": type(agent_snapshot).__name__})

        if "checkpoint" in agent_snapshot and hasattr(agent, "load_checkpoint"):
            agent.load_checkpoint(copy.deepcopy(agent_snapshot["checkpoint"]))

        for attribute, value in agent_snapshot.get("modules", {}).items():
            current = getattr(agent, attribute, None)
            if current is not None and hasattr(current, "load_state_dict"):
                current.load_state_dict(copy.deepcopy(value))
            elif current is not None:
                setattr(agent, attribute, copy.deepcopy(value))

        for attribute, value in agent_snapshot.get("optimizers", {}).items():
            optimizer = getattr(agent, attribute, None)
            if optimizer is not None and hasattr(optimizer, "load_state_dict"):
                optimizer.load_state_dict(copy.deepcopy(value))

        for attribute, value in agent_snapshot.get("scalars", {}).items():
            setattr(agent, attribute, copy.deepcopy(value))

        for attribute, value in agent_snapshot.get("buffers", {}).items():
            setattr(agent, attribute, copy.deepcopy(value))

    def _attempt_reinitialisation(self) -> Dict[str, Any]:
        init_signature = inspect.signature(self.learning_agent.__class__.__init__)
        kwargs: Dict[str, Any] = {}
        known_sources = {
            "shared_memory": getattr(self.learning_agent, "shared_memory", None),
            "agent_factory": getattr(self.learning_agent, "agent_factory", None),
            "env": getattr(self.learning_agent, "env", None),
            "config": getattr(self.learning_agent, "config", None),
            "performance_metrics": getattr(self.learning_agent, "performance_metrics", None),
        }
        for name, parameter in init_signature.parameters.items():
            if name == "self":
                continue
            if name in known_sources and known_sources[name] is not None:
                kwargs[name] = known_sources[name]
            elif parameter.default is inspect._empty:
                raise RecoveryError("Cannot perform full reset safely; missing required constructor argument", context={"argument": name})
        self.learning_agent.__init__(**kwargs)
        return {"mode": "reinitialised", "kwargs": sorted(kwargs.keys())}

    def _build_snapshot(self, include_buffers: bool = False) -> Optional[Dict[str, Any]]:
        agents_snapshot: Dict[str, Any] = {}
        for agent_id, agent in self._iter_named_agents():
            agent_snapshot: Dict[str, Any] = {"modules": {}, "optimizers": {}, "scalars": {}, "buffers": {}}

            if hasattr(agent, "get_checkpoint") and callable(agent.get_checkpoint):
                try:
                    agent_snapshot["checkpoint"] = copy.deepcopy(agent.get_checkpoint())
                except Exception as exc:
                    logger.warning("Agent checkpoint snapshot failed for %s: %s", agent_id, exc)

            for attribute, module in self._iter_network_modules(agent):
                if hasattr(module, "state_dict") and callable(module.state_dict):
                    agent_snapshot["modules"][attribute] = copy.deepcopy(module.state_dict())
                else:
                    agent_snapshot["modules"][attribute] = copy.deepcopy(module)

            for attribute, optimizer in self._iter_optimizers(agent):
                if hasattr(optimizer, "state_dict") and callable(optimizer.state_dict):
                    agent_snapshot["optimizers"][attribute] = copy.deepcopy(optimizer.state_dict())
                elif hasattr(optimizer, "learning_rate"):
                    agent_snapshot["optimizers"][attribute] = {"learning_rate": copy.deepcopy(optimizer.learning_rate)}

            for scalar_attr in self.recover_config.get("scalar_state_attributes", {}):
                if hasattr(agent, scalar_attr):
                    agent_snapshot["scalars"][str(scalar_attr)] = copy.deepcopy(getattr(agent, scalar_attr))

            if include_buffers:
                for buffer_attr in self.recover_config.get("reset_buffer_attributes", {}):
                    if hasattr(agent, buffer_attr):
                        agent_snapshot["buffers"][str(buffer_attr)] = copy.deepcopy(getattr(agent, buffer_attr))

            if any(agent_snapshot.values()):
                agents_snapshot[str(agent_id)] = agent_snapshot

        return {"created_at": self._time_fn(), "agents": agents_snapshot} if agents_snapshot else None


if __name__ == "__main__":
    print("\n=== Running Recovery System ===\n")
    printer.status("TEST", "Recovery System initialized", "info")
    class Agent:
        def __init__(self):
            self.agent_id = "core"
            self.policy_net = torch.nn.Linear(4, 2)
            self.optimizer = torch.optim.SGD(self.policy_net.parameters(), lr=0.1)
            self.learning_rate = 0.1
            self.epsilon = 0.2
            self.replay_buffer = [1, 2, 3]
        def reset(self):
            self.was_reset = True

    class Parent:
        def __init__(self):
            self.agents = {"core": Agent()}
            self.active_strategy = "dqn"

    now = [100.0]
    rs = RecoverySystem(Parent(), time_fn=lambda: now[0])
    snap = rs.register_stable_snapshot("stable")
    assert snap and snap["snapshot"]["agents"]

    r1 = rs.execute_recovery(RuntimeError("spike"), force_strategy="soft_reset")
    assert r1["status"] == "recovered"
    assert rs.learning_agent.agents["core"].replay_buffer == []

    r2 = rs.execute_recovery(GradientExplosionError(norm=99.0, threshold=10.0), force_strategy="lr_adjustment")
    assert r2["details"]["adjustments"]
    assert rs.learning_agent.agents["core"].learning_rate < 0.1

    rs.learning_agent.agents["core"].learning_rate = 0.9
    r3 = rs.execute_recovery(force_strategy="architecture_rollback")
    assert r3["status"] == "recovered"
    assert rs.learning_agent.agents["core"].learning_rate == 0.1

    r4 = rs.execute_recovery(force_strategy="strategy_switch")
    assert rs.learning_agent.active_strategy == rs.recover_config.get("safe_strategy", "rl")
    assert rs.diagnostics()["history_size"] >= 4
    print("\n=== Test ran successfully ===\n")
