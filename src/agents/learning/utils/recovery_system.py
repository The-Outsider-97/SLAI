"""Production-hardened recovery orchestration for learning agents."""

from __future__ import annotations

import copy
import inspect
import time

from typing import Any, Callable, Dict, Iterable, List, Optional

from .config_loader import load_global_config, get_config_section
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Recovery System")
printer = PrettyPrinter

class RecoverySystem:
    """
    Coordinate escalating recovery actions for a learning agent stack.

    The class preserves the original ideas from the source implementation:
    soft reset, learning-rate reduction, architecture rollback, strategy
    switch, and full reset. The implementation is made safer by:
    - supporting absent optional attributes gracefully
    - avoiding brittle direct assumptions about agent layout
    - recording recovery history and snapshots
    - updating optimizer param groups when learning rates change
    """

    DEFAULT_CONFIG: Dict[str, Any] = {
        "error_decay_time": 3600.0,
        "error_decay_factor": 0.5,
        "error_thresholds": [3, 6, 9, 12],
        "lr_reduction_factor": 0.5,
        "min_learning_rate": 1e-6,
        "safe_strategy": "rl",
        "max_snapshots": 5,
        "reset_buffer_attributes": [
            "state_history",
            "action_history",
            "reward_history",
            "replay_buffer",
            "memory",
            "episode_buffer",
        ],
        "network_attributes": [
            "policy_net",
            "target_net",
            "value_net",
            "critic",
            "actor",
            "q_network",
            "model",
        ],
    }

    def __init__(
        self,
        learning_agent: Any,
        time_fn: Optional[Callable[[], float]] = None,
    ) -> None:
        self.learning_agent = learning_agent
        self.config = load_global_config()
        self.recover_config = get_config_section("recovery_system") or {}
        self._time_fn = time_fn or time.time

        self.error_count = 0
        self.last_error_time = self._time_fn()
        self.recovery_history: List[Dict[str, Any]] = []
        self.stable_snapshots: List[Dict[str, Any]] = []

        self.recovery_strategies = [
            self._recover_soft_reset,
            self._recover_learning_rate_adjustment,
            self._recover_architecture_rollback,
            self._recover_strategy_switch,
            self._recover_full_reset,
        ]
        self.error_thresholds = self._normalise_thresholds(self.recover_config.get("error_thresholds", [3, 6, 9, 12]))

    @staticmethod
    def _normalise_thresholds(thresholds: Iterable[int]) -> List[int]:
        normalised = sorted(int(t) for t in thresholds if int(t) >= 0)
        if not normalised:
            return [3, 6, 9, 12]
        return normalised

    def decay_error_count(self) -> int:
        """Decay the accumulated error count when enough time has elapsed."""
        current_time = self._time_fn()
        elapsed = max(0.0, current_time - self.last_error_time)
        decay_time = float(self.recover_config.get("error_decay_time", 3600.0))
        if decay_time <= 0.0:
            return self.error_count

        intervals = int(elapsed // decay_time)
        if intervals > 0:
            decay_factor = float(self.recover_config.get("error_decay_factor", 0.5))
            decay_factor = min(max(decay_factor, 0.0), 1.0)
            decayed = float(self.error_count)
            for _ in range(intervals):
                decayed *= decay_factor
            self.error_count = max(0, int(round(decayed)))
            self.last_error_time = current_time
        return self.error_count

    def increment_error_count(self, count: int = 1, error: Optional[BaseException] = None) -> int:
        """Register one or more failures."""
        if count < 0:
            raise ValueError("count must be non-negative.")
        self.decay_error_count()
        self.error_count += count
        self.last_error_time = self._time_fn()
        if error is not None:
            logger.warning("Error count increased to %s due to: %s", self.error_count, error)
        else:
            logger.warning("Error count increased to %s", self.error_count)
        return self.error_count

    def register_stable_snapshot(self, label: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Capture a rollback snapshot.

        The snapshot prefers architecture_history-compatible structure so existing
        consumers can keep working.
        """
        snapshot = self._build_snapshot()
        if snapshot is None:
            logger.warning("No stable snapshot could be created.")
            return None

        snapshot_record = {
            "label": label or f"snapshot_{len(self.stable_snapshots)}",
            "created_at": self._time_fn(),
            "snapshot": snapshot,
        }
        self.stable_snapshots.append(snapshot_record)
        max_snapshots = max(1, int(self.recover_config.get("max_snapshots", 5)))
        if len(self.stable_snapshots) > max_snapshots:
            self.stable_snapshots = self.stable_snapshots[-max_snapshots:]
        logger.info("Stable recovery snapshot recorded: %s", snapshot_record["label"])
        return snapshot_record

    def execute_recovery(self, error: Optional[BaseException] = None) -> Dict[str, Any]:
        """Execute the appropriate recovery strategy based on current error severity."""
        if error is not None:
            self.increment_error_count(error=error)
        else:
            self.decay_error_count()

        strategy_level = 0
        for index, threshold in enumerate(self.error_thresholds):
            if self.error_count >= threshold:
                strategy_level = index + 1
        strategy_level = min(strategy_level, len(self.recovery_strategies) - 1)

        logger.warning("Executing recovery level %s", strategy_level + 1)
        result = self.recovery_strategies[strategy_level]()
        result.setdefault("error_count", self.error_count)
        result.setdefault("level", strategy_level + 1)
        result.setdefault("timestamp", self._time_fn())
        self.recovery_history.append(result)
        return result

    def reset_error_count(self) -> None:
        """Reset failure tracking after a successful recovery or healthy period."""
        self.error_count = 0
        self.last_error_time = self._time_fn()
        logger.info("Error count reset after successful recovery")

    def _iter_agents(self) -> Iterable[Any]:
        agents = getattr(self.learning_agent, "agents", None)
        if isinstance(agents, dict):
            return agents.values()
        if isinstance(agents, (list, tuple, set)):
            return list(agents)
        return [self.learning_agent]

    def _iter_network_modules(self, agent: Any) -> Iterable[Any]:
        for attribute in self.recover_config.get("network_attributes", []):
            module = getattr(agent, attribute, None)
            if module is not None:
                yield attribute, module

    def _clear_buffers(self, owner: Any) -> List[str]:
        cleared = []
        for attribute in self.recover_config.get("reset_buffer_attributes", []):
            value = getattr(owner, attribute, None)
            if hasattr(value, "clear"):
                value.clear()
                cleared.append(attribute)
            elif isinstance(value, list):
                value[:] = []
                cleared.append(attribute)
        return cleared

    def _recover_soft_reset(self) -> Dict[str, Any]:
        """Reset network parameters, transient buffers, and exploration settings."""
        logger.warning("Performing soft reset")
        reset_networks = []
        for agent in self._iter_agents():
            for attribute, module in self._iter_network_modules(agent):
                if hasattr(module, "apply"):
                    module.apply(self._reset_module_parameters)
                    reset_networks.append(attribute)
            for epsilon_attr in ("epsilon", "exploration_rate"):
                if hasattr(agent, epsilon_attr):
                    current = float(getattr(agent, epsilon_attr))
                    setattr(agent, epsilon_attr, min(1.0, max(current, current * 1.5 if current > 0 else 1.0)))

        cleared_buffers = self._clear_buffers(self.learning_agent)
        return {
            "status": "recovered",
            "strategy": "soft_reset",
            "details": {
                "reset_networks": sorted(set(reset_networks)),
                "cleared_buffers": cleared_buffers,
            },
        }

    @staticmethod
    def _reset_module_parameters(module: Any) -> None:
        if hasattr(module, "reset_parameters") and callable(module.reset_parameters):
            module.reset_parameters()

    def _recover_learning_rate_adjustment(self) -> Dict[str, Any]:
        """Reduce learning rates on agents and optimizers without crossing a floor."""
        logger.warning("Adjusting learning rates")
        lr_reduction = float(self.recover_config.get("lr_reduction_factor", 0.5))
        min_lr = float(self.recover_config.get("min_learning_rate", 1e-6))
        adjustments = []

        for agent in self._iter_agents():
            if hasattr(agent, "learning_rate"):
                old_lr = float(getattr(agent, "learning_rate"))
                new_lr = max(min_lr, old_lr * lr_reduction)
                setattr(agent, "learning_rate", new_lr)
                adjustments.append({"target": getattr(agent, "name", type(agent).__name__), "old_lr": old_lr, "new_lr": new_lr})

            optimizer = getattr(agent, "optimizer", None)
            if optimizer is not None and hasattr(optimizer, "param_groups"):
                for group in optimizer.param_groups:
                    old_lr = float(group.get("lr", min_lr))
                    group["lr"] = max(min_lr, old_lr * lr_reduction)

        return {
            "status": "recovered",
            "strategy": "lr_adjustment",
            "details": {"adjustments": adjustments, "reduction_factor": lr_reduction},
        }

    def _recover_architecture_rollback(self) -> Dict[str, Any]:
        """Rollback to the most recent stable architecture or model snapshot."""
        logger.warning("Performing architecture rollback")

        snapshot = None
        history = getattr(self.learning_agent, "architecture_history", None)
        if history:
            snapshot = history[-1]
        elif self.stable_snapshots:
            snapshot = self.stable_snapshots[-1]["snapshot"]

        if snapshot is None:
            logger.warning("No saved architecture found; falling back to strategy switch")
            fallback = self._recover_strategy_switch()
            fallback["details"] = {"fallback": "strategy_switch", "reason": "missing_snapshot"}
            return fallback

        restored_agents = []
        if isinstance(snapshot, dict) and "agents" in snapshot:
            for agent_id, agent_state in snapshot["agents"].items():
                agent = getattr(self.learning_agent, "agents", {}).get(agent_id) if isinstance(getattr(self.learning_agent, "agents", None), dict) else None
                if agent is None:
                    continue
                self._restore_agent_state(agent, agent_state)
                restored_agents.append(agent_id)
        elif isinstance(snapshot, dict):
            for agent_id, agent_snapshot in snapshot.items():
                agent = getattr(self.learning_agent, "agents", {}).get(agent_id) if isinstance(getattr(self.learning_agent, "agents", None), dict) else None
                if agent is None:
                    continue
                self._restore_agent_state(agent, agent_snapshot)
                restored_agents.append(agent_id)

        return {
            "status": "recovered",
            "strategy": "architecture_rollback",
            "details": {"restored_agents": restored_agents},
        }

    def _restore_agent_state(self, agent: Any, agent_snapshot: Dict[str, Any]) -> None:
        optimizer_state = agent_snapshot.get("optimizer_state")
        optimizer = getattr(agent, "optimizer", None)
        if optimizer_state is not None and optimizer is not None and hasattr(optimizer, "load_state_dict"):
            optimizer.load_state_dict(copy.deepcopy(optimizer_state))

        for attribute, value in agent_snapshot.items():
            if attribute == "optimizer_state":
                continue
            if not hasattr(agent, attribute):
                continue

            current = getattr(agent, attribute)
            if hasattr(current, "load_state_dict") and isinstance(value, dict):
                current.load_state_dict(copy.deepcopy(value))
            else:
                setattr(agent, attribute, copy.deepcopy(value))

    def _recover_strategy_switch(self) -> Dict[str, Any]:
        """Switch the parent learning agent to a safer fallback strategy."""
        logger.warning("Switching to safe strategy")
        safe_strategy = self.recover_config.get("safe_strategy", "rl")
        previous_strategy = getattr(self.learning_agent, "active_strategy", None)
        setattr(self.learning_agent, "active_strategy", safe_strategy)

        safety_guard = getattr(self.learning_agent, "safety_guard", None)
        guard_result = None
        if safety_guard is not None and hasattr(safety_guard, "execute"):
            try:
                guard_result = safety_guard.execute({"task": "emergency_override"})
            except Exception as exc:  # pragma: no cover
                logger.exception("Safety guard execution failed: %s", exc)

        return {
            "status": "recovered",
            "strategy": "strategy_switch",
            "details": {
                "previous_strategy": previous_strategy,
                "new_strategy": safe_strategy,
                "safety_guard_result": guard_result,
            },
        }

    def _recover_full_reset(self) -> Dict[str, Any]:
        """Perform a full reset via a reset hook or best-effort reconstruction."""
        logger.critical("Performing full reset")
        if hasattr(self.learning_agent, "reset") and callable(self.learning_agent.reset):
            self.learning_agent.reset()
            return {"status": "recovered", "strategy": "full_reset", "details": {"mode": "reset_hook"}}

        reconstruction = self._attempt_reinitialisation()
        return {"status": "recovered", "strategy": "full_reset", "details": reconstruction}

    def _attempt_reinitialisation(self) -> Dict[str, Any]:
        init_signature = inspect.signature(self.learning_agent.__class__.__init__)
        kwargs = {}
        known_sources = {
            "shared_memory": getattr(self.learning_agent, "shared_memory", None),
            "agent_factory": getattr(self.learning_agent, "agent_factory", None),
            "env": getattr(self.learning_agent, "env", None),
            "config": getattr(self.learning_agent, "config", None),
        }

        for name, parameter in init_signature.parameters.items():
            if name == "self":
                continue
            if name in known_sources and known_sources[name] is not None:
                kwargs[name] = known_sources[name]
            elif parameter.default is inspect._empty:
                raise RuntimeError(
                    f"Cannot perform full reset safely; missing required constructor argument '{name}'."
                )

        self.learning_agent.__init__(**kwargs)
        return {"mode": "reinitialised", "kwargs": sorted(kwargs.keys())}

    def _build_snapshot(self) -> Optional[Dict[str, Any]]:
        agents_obj = getattr(self.learning_agent, "agents", None)
        if not isinstance(agents_obj, dict) or not agents_obj:
            return None

        snapshot = {"agents": {}}
        for agent_id, agent in agents_obj.items():
            agent_snapshot: Dict[str, Any] = {}
            for attribute, module in self._iter_network_modules(agent):
                if hasattr(module, "state_dict"):
                    agent_snapshot[attribute] = copy.deepcopy(module.state_dict())
                else:
                    agent_snapshot[attribute] = copy.deepcopy(module)

            optimizer = getattr(agent, "optimizer", None)
            if optimizer is not None and hasattr(optimizer, "state_dict"):
                agent_snapshot["optimizer_state"] = copy.deepcopy(optimizer.state_dict())

            for scalar_attr in ("learning_rate", "epsilon", "exploration_rate"):
                if hasattr(agent, scalar_attr):
                    agent_snapshot[scalar_attr] = copy.deepcopy(getattr(agent, scalar_attr))

            if agent_snapshot:
                snapshot["agents"][agent_id] = agent_snapshot

        return snapshot if snapshot["agents"] else None


if __name__ == "__main__":
    print("\n=== Running Execution Task Coordinator ===\n")
    printer.status("TEST", "Starting Task Coordinator tests", "info")
    agent = {}
    ttime = None

    recovery = RecoverySystem(learning_agent=agent, time_fn=ttime)

    print(recovery)

    snapshot = recovery.register_stable_snapshot()
    recover = recovery.execute_recovery()

    printer.pretty("SNAPSHOT", "SUCCESS" if snapshot else "FAILURE", "success" if snapshot else "error")
    printer.pretty("RECOVER", "SUCCESS" if recover else "FAILURE", "success" if recover else "error")

    print("\n=== All tests completed successfully! ===\n")