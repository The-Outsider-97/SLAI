from __future__ import annotations

import random
import time

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict

from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Reliability Manager")
printer = PrettyPrinter

class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class RetryPolicy:
    max_attempts: int = 1
    backoff_factor: float = 0.0
    max_backoff_seconds: float = 2.0
    jitter_seconds: float = 0.0


@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 3
    recovery_timeout_seconds: float = 5.0
    half_open_success_threshold: int = 1


class AgentCircuitBreaker:
    def __init__(self, config: CircuitBreakerConfig | None = None):
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._opened_at = 0.0

    @property
    def state(self) -> CircuitState:
        if self._state == CircuitState.OPEN and (time.time() - self._opened_at) >= self.config.recovery_timeout_seconds:
            self._state = CircuitState.HALF_OPEN
            self._success_count = 0
        return self._state

    def allow_request(self) -> bool:
        return self.state != CircuitState.OPEN

    def record_success(self) -> None:
        if self.state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.config.half_open_success_threshold:
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                self._success_count = 0
            return

        self._state = CircuitState.CLOSED
        self._failure_count = 0

    def record_failure(self) -> None:
        self._failure_count += 1
        self._success_count = 0
        if self._failure_count >= self.config.failure_threshold:
            self._state = CircuitState.OPEN
            self._opened_at = time.time()


class ReliabilityManager:
    def __init__(
        self,
        retry_policy: RetryPolicy | None = None,
        breaker_config: CircuitBreakerConfig | None = None,
    ):
        self.retry_policy = retry_policy or RetryPolicy()
        self.breaker_config = breaker_config or CircuitBreakerConfig()
        self._breakers: Dict[str, AgentCircuitBreaker] = {}

        logger.info("Reliability Manager initialized")

    def _get_breaker(self, key: str) -> AgentCircuitBreaker:
        if key not in self._breakers:
            self._breakers[key] = AgentCircuitBreaker(config=self.breaker_config)
        return self._breakers[key]

    def is_available(self, key: str) -> bool:
        return self._get_breaker(key).allow_request()

    def execute(self, key: str, operation: Callable[[], Any]) -> Any:
        breaker = self._get_breaker(key)
        if not breaker.allow_request():
            raise RuntimeError(f"Circuit open for '{key}'")

        last_error: Exception | None = None
        attempts = max(1, int(self.retry_policy.max_attempts))
        for attempt in range(1, attempts + 1):
            try:
                result = operation()
                breaker.record_success()
                return result
            except Exception as exc:
                last_error = exc
                breaker.record_failure()
                if attempt < attempts:
                    self._sleep_backoff(attempt)

        if last_error is not None:
            raise last_error
        raise RuntimeError(f"Operation failed for '{key}'")

    def status(self) -> Dict[str, Dict[str, Any]]:
        return {
            key: {
                "state": breaker.state.value,
                "failure_count": breaker._failure_count,
                "success_count": breaker._success_count,
            }
            for key, breaker in self._breakers.items()
        }

    def _sleep_backoff(self, attempt: int) -> None:
        if self.retry_policy.backoff_factor <= 0:
            return
        delay = min(
            self.retry_policy.max_backoff_seconds,
            self.retry_policy.backoff_factor * (2 ** (attempt - 1)),
        )
        jitter = random.uniform(0.0, max(0.0, self.retry_policy.jitter_seconds))
        time.sleep(delay + jitter)
