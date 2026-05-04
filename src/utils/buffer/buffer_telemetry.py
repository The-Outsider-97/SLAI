from __future__ import annotations

import time
import numpy as np

from contextlib import contextmanager
from dataclasses import dataclass
from threading import RLock
from typing import Dict, Iterator, Optional

from .utils.config_loader import get_config_section, load_global_config
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Buffer Telemetry")
printer = PrettyPrinter()


@dataclass
class MetricStats:
    count: int = 0
    total: float = 0.0
    min: float = float("inf")
    max: float = float("-inf")
    last: float = 0.0

    def update(self, value: float) -> None:
        value = float(value)
        self.count += 1
        self.total += value
        self.last = value
        self.min = min(self.min, value)
        self.max = max(self.max, value)

    def to_dict(self) -> Dict[str, float]:
        mean = self.total / self.count if self.count else 0.0
        return {
            "count": self.count,
            "total": self.total,
            "mean": mean,
            "min": 0.0 if self.min == float("inf") else self.min,
            "max": 0.0 if self.max == float("-inf") else self.max,
            "last": self.last,
        }


class BufferTelemetry:
    """Thread-safe in-memory telemetry collector for buffer operations."""

    def __init__(self, component_name: str = "buffer", user_config: Optional[dict] = None):
        self.component_name = component_name

        self.config = load_global_config()
        tel_config = dict(get_config_section("telemetry") or {})
        if user_config:
            tel_config.update(user_config.get("telemetry", {}) if isinstance(user_config, dict) else {})

        self.enabled = bool(tel_config.get("enabled", True))
        self.log_interval = int(tel_config.get("log_interval", 200))

        self.lock = RLock()
        self.counters: Dict[str, float] = {}
        self.observations: Dict[str, MetricStats] = {}

    def increment(self, name: str, amount: float = 1.0) -> None:
        if not self.enabled:
            return
        with self.lock:
            self.counters[name] = self.counters.get(name, 0.0) + float(amount)
            if self.log_interval > 0 and int(self.counters[name]) % self.log_interval == 0:
                logger.info("[%s] counter %s=%s", self.component_name, name, self.counters[name])

    def observe(self, name: str, value: float) -> None:
        if not self.enabled:
            return
        with self.lock:
            if name not in self.observations:
                self.observations[name] = MetricStats()
            self.observations[name].update(float(value))

    @contextmanager
    def time_block(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.observe(name, elapsed)

    def snapshot(self) -> Dict[str, Dict[str, float]]:
        with self.lock:
            return {
                "component": {"name": self.component_name, "enabled": float(self.enabled)},
                "counters": dict(self.counters),
                "observations": {key: stats.to_dict() for key, stats in self.observations.items()},
            }

    def reset(self) -> None:
        with self.lock:
            self.counters.clear()
            self.observations.clear()

    def export_numpy(self) -> Dict[str, np.ndarray]:
        """Useful bridge for training/analysis workflows that expect numpy tensors."""
        snap = self.snapshot()
        counters = snap["counters"]
        observations = snap["observations"]

        counter_keys = np.array(list(counters.keys()), dtype=object)
        counter_values = np.array(list(counters.values()), dtype=np.float32)

        obs_keys = np.array(list(observations.keys()), dtype=object)
        obs_means = np.array([entry["mean"] for entry in observations.values()], dtype=np.float32)

        return {
            "counter_keys": counter_keys,
            "counter_values": counter_values,
            "observation_keys": obs_keys,
            "observation_means": obs_means,
        }


__all__ = ["MetricStats", "BufferTelemetry"]
