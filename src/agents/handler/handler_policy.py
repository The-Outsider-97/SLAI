import time

from collections import defaultdict
from typing import Dict, Optional

from src.agents.handler.utils.config_loader import load_global_config, get_config_section
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Handler Policy")
printer = PrettyPrinter

class HandlerPolicy:
    """Policy guardrails for retries, circuit breaker, and evaluator hooks."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = load_global_config()

        self.policy_config = get_config_section('policy')
        self.max_retries = self.policy_config.get("max_retries", 2)
        self.circuit_breaker_threshold = self.policy_config.get("circuit_breaker_threshold", 5)
        self.cooldown_seconds = self.policy_config.get("cooldown_seconds", 30)
        self.failure_budget_window_seconds = self.policy_config.get("failure_budget_window_seconds", 300)
        self.evaluator_hooks_enabled = self.policy_config.get("evaluator_hooks_enabled", True)

        self._failure_counters = defaultdict(int)
        self._last_failures = defaultdict(list)
        self._breaker_open_until = defaultdict(float)

        logger.info(f"Handler Policy succesfully initialized")

    def can_attempt(self, agent_name: str) -> bool:
        return time.time() >= self._breaker_open_until.get(agent_name, 0.0)

    def retries_allowed(self, attempted_retries: int, max_retries: Optional[int] = None) -> bool:
        limit = self.max_retries if max_retries is None else max(0, int(max_retries))
        return attempted_retries < limit

    def record_failure(self, agent_name: str) -> None:
        now = time.time()
        self._failure_counters[agent_name] += 1
        self._last_failures[agent_name].append(now)
        self._last_failures[agent_name] = [
            ts
            for ts in self._last_failures[agent_name]
            if now - ts <= self.failure_budget_window_seconds
        ]

        if self._failure_counters[agent_name] >= self.circuit_breaker_threshold:
            self._breaker_open_until[agent_name] = now + self.cooldown_seconds

    def record_success(self, agent_name: str) -> None:
        self._failure_counters[agent_name] = 0
        self._breaker_open_until[agent_name] = 0.0

    def breaker_status(self, agent_name: str) -> Dict[str, float | bool]:
        open_until = self._breaker_open_until.get(agent_name, 0.0)
        now = time.time()
        return {
            "is_open": open_until > now,
            "open_until": open_until,
            "seconds_remaining": max(0.0, open_until - now),
        }

if __name__ == "__main__":
    policy = HandlerPolicy(config={"max_retries": 2, "cooldown_seconds": 1, "circuit_breaker_threshold": 2})
    agent_name = "demo_agent"

    print("HandlerPolicy smoke test")
    print(f"initial_can_attempt={policy.can_attempt(agent_name)}")

    policy.record_failure(agent_name)
    print(f"status_after_first_failure={policy.breaker_status(agent_name)}")

    policy.record_failure(agent_name)
    status = policy.breaker_status(agent_name)
    print(f"status_after_threshold={status}")
    print(f"retries_allowed_0={policy.retries_allowed(0)}")
    print(f"retries_allowed_2={policy.retries_allowed(2)}")

    time.sleep(1.1)
    print(f"can_attempt_after_cooldown={policy.can_attempt(agent_name)}")

    policy.record_success(agent_name)
    print(f"status_after_success={policy.breaker_status(agent_name)}")
