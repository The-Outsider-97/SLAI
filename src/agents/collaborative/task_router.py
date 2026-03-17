import time

from typing import Any, Dict, Optional

from src.agents.collaborative.reliability import ReliabilityManager
from src.agents.collaborative.router_strategy import BaseRouterStrategy, build_router_strategy
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("SLAI Task Router")
printer = PrettyPrinter


class TaskRouter:
    """Capability-based router backed by AgentRegistry with pluggable strategy and reliability controls."""

    def __init__(
        self,
        registry=None,
        shared_memory=None,
        strategy: Optional[BaseRouterStrategy] = None,
        reliability_manager: Optional[ReliabilityManager] = None,
    ):
        self.registry = registry
        self.shared_memory = shared_memory
        self.strategy = strategy or build_router_strategy("weighted")
        self.reliability_manager = reliability_manager or ReliabilityManager()

        logger.info("Task Router initialized")

    def route(self, task_type: str, task_data: Dict[str, Any]) -> Any:
        if self.registry is None:
            raise RuntimeError("TaskRouter requires a registry")

        eligible_agents = self.registry.get_agents_by_task(task_type)
        if not eligible_agents:
            raise RuntimeError(f"No agents found for task type '{task_type}'")

        ranked = self.strategy.rank_agents(eligible_agents, stats=self._get_stats(), task_data=task_data)
        last_error: Optional[Exception] = None
        for agent_name, agent_meta, _ in ranked:
            if not self.reliability_manager.is_available(agent_name):
                last_error = RuntimeError(f"Agent '{agent_name}' unavailable due to open circuit")
                continue
            try:
                self._set_active_tasks(agent_name, +1)
                result = self.reliability_manager.execute(
                    agent_name,
                    lambda: agent_meta["instance"].execute(task_data),
                )
                self._record_success(agent_name)
                return result
            except Exception as exc:
                last_error = exc
                self._record_failure(agent_name)
                logger.exception("Agent '%s' failed for task '%s'", agent_name, task_type)
            finally:
                self._set_active_tasks(agent_name, -1)

        raise RuntimeError(f"All agents failed for task type '{task_type}': {last_error}")

    def _get_stats(self) -> Dict[str, Dict[str, Any]]:
        if self.shared_memory is None:
            return {}
        return self.shared_memory.get("agent_stats", {}) or {}

    def _set_stats(self, stats: Dict[str, Dict[str, Any]]) -> None:
        if self.shared_memory is not None:
            self.shared_memory.set("agent_stats", stats)

    def _set_active_tasks(self, agent_name: str, delta: int) -> None:
        stats = self._get_stats()
        row = stats.setdefault(agent_name, {"successes": 0, "failures": 0, "active_tasks": 0, "last_seen": 0.0})
        row["active_tasks"] = max(0, int(row.get("active_tasks", 0)) + delta)
        row["last_seen"] = time.time()
        self._set_stats(stats)
        self._touch_agent_heartbeat(agent_name, row["last_seen"])

    def _record_success(self, agent_name: str) -> None:
        stats = self._get_stats()
        row = stats.setdefault(agent_name, {"successes": 0, "failures": 0, "active_tasks": 0, "last_seen": 0.0})
        row["successes"] = int(row.get("successes", 0)) + 1
        row["last_seen"] = time.time()
        self._set_stats(stats)
        self._touch_agent_heartbeat(agent_name, row["last_seen"])

    def _record_failure(self, agent_name: str) -> None:
        stats = self._get_stats()
        row = stats.setdefault(agent_name, {"successes": 0, "failures": 0, "active_tasks": 0, "last_seen": 0.0})
        row["failures"] = int(row.get("failures", 0)) + 1
        row["last_seen"] = time.time()
        self._set_stats(stats)
        self._touch_agent_heartbeat(agent_name, row["last_seen"])

    def _touch_agent_heartbeat(self, agent_name: str, timestamp: float) -> None:
        if self.shared_memory is None:
            return

        agent_key = f"agent:{agent_name}"
        heartbeat = self.shared_memory.get(agent_key, {}) or {}
        if not isinstance(heartbeat, dict):
            heartbeat = {"status": "active"}
        heartbeat["last_seen"] = float(timestamp)
        heartbeat.setdefault("status", "active")
        self.shared_memory.set(agent_key, heartbeat)

if __name__ == "__main__":
    print("\n=== Running Task Router ===\n")
    printer.status("TEST", "Starting Task Router tests", "info")

    class _Memory:
        def __init__(self):
            self._store = {}

        def get(self, key, default=None):
            return self._store.get(key, default)

        def set(self, key, value):
            self._store[key] = value

    class _Agent:
        def __init__(self, name, fail=False):
            self.name = name
            self.fail = fail

        def execute(self, task_data):
            if self.fail:
                raise RuntimeError(f"{self.name} failed")
            return {"agent": self.name, "task": task_data}

    class _Registry:
        def __init__(self):
            self._agents = {
                "A": {"instance": _Agent("A", fail=True), "capabilities": ["translate"]},
                "B": {"instance": _Agent("B"), "capabilities": ["translate"]},
            }

        def get_agents_by_task(self, task_type):
            return {k: v for k, v in self._agents.items() if task_type in v["capabilities"]}

    memory = _Memory()
    router = TaskRouter(registry=_Registry(), shared_memory=memory)

    result = router.route("translate", {"text": "hello"})
    assert result["agent"] == "B"

    stats = memory.get("agent_stats")
    assert stats["A"]["failures"] >= 1
    assert stats["B"]["successes"] >= 1
    assert stats["A"]["active_tasks"] == 0 and stats["B"]["active_tasks"] == 0

    try:
        router.route("missing", {"text": "x"})
        raise AssertionError("Expected missing-task failure")
    except RuntimeError as exc:
        assert "No agents found" in str(exc)

    print("All task_router.py tests passed.\n")
