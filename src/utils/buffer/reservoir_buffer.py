from __future__ import annotations

import random
from dataclasses import dataclass
from threading import RLock
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .buffer_telemetry import BufferTelemetry
from .buffer_validation import Transition, TransitionValidator
from .utils.config_loader import get_config_section, load_global_config


@dataclass
class ReservoirConfig:
    """Configuration for streaming reservoir replay."""

    capacity: int = 100_000
    seed: Optional[int] = None

    @classmethod
    def from_config(cls, user_config: Optional[Dict[str, Any]] = None) -> "ReservoirConfig":
        load_global_config()
        config = dict(get_config_section("reservoir") or {})
        if user_config:
            config.update(user_config.get("reservoir", {}) if isinstance(user_config, dict) else {})

        capacity = int(config.get("capacity", 100_000))
        if capacity <= 0:
            raise ValueError("reservoir capacity must be > 0")

        return cls(
            capacity=capacity,
            seed=config.get("seed"),
        )


class ReservoirReplayBuffer:
    """Streaming replay buffer using reservoir sampling.

    Every incoming item has equal probability of being retained in the
    fixed-size reservoir, making this appropriate for unbounded streams.
    """

    def __init__(
        self,
        user_config: Optional[Dict[str, Any]] = None,
        validator: Optional[TransitionValidator] = None,
        telemetry: Optional[BufferTelemetry] = None,
    ):
        self.config = ReservoirConfig.from_config(user_config=user_config)
        self.capacity = self.config.capacity

        self._rng = random.Random(self.config.seed)
        self._lock = RLock()

        self.validator = validator or TransitionValidator()
        self.telemetry = telemetry or BufferTelemetry(component_name="reservoir_replay_buffer")

        self.buffer: List[Transition] = []
        self.total_seen = 0
        self.total_replaced = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self.buffer)

    def push(self, transition: Sequence[Any]) -> None:
        with self.telemetry.time_block("push_latency_seconds"):
            with self._lock:
                normalized = self.validator.sanitize_transition(tuple(transition))
                self.total_seen += 1
                seen = self.total_seen

                if len(self.buffer) < self.capacity:
                    self.buffer.append(normalized)
                else:
                    # Reservoir sampling: replace index j with probability capacity/seen.
                    j = self._rng.randint(1, seen)
                    if j <= self.capacity:
                        self.buffer[j - 1] = normalized
                        self.total_replaced += 1

                self.telemetry.increment("push_count", 1)

    def push_components(self, agent_id: Any, state: Any, action: Any,
                        reward: Any, next_state: Any, done: Any) -> None:
        self.push((agent_id, state, action, reward, next_state, done))

    def extend(self, transitions: Sequence[Sequence[Any]]) -> None:
        for transition in transitions:
            self.push(transition)

    def sample(self, batch_size: int, replace: bool = False) -> Tuple[np.ndarray, ...]:
        with self.telemetry.time_block("sample_latency_seconds"):
            with self._lock:
                if batch_size <= 0:
                    raise ValueError("batch_size must be > 0")
                if not self.buffer:
                    raise ValueError("Cannot sample from an empty reservoir")
                if (not replace) and batch_size > len(self.buffer):
                    raise ValueError(
                        f"batch_size={batch_size} exceeds buffer size={len(self.buffer)} when replace=False"
                    )

                indices = (
                    [self._rng.randrange(len(self.buffer)) for _ in range(batch_size)]
                    if replace
                    else self._rng.sample(range(len(self.buffer)), batch_size)
                )
                batch = [self.buffer[i] for i in indices]

                self.telemetry.increment("sample_count", 1)
                self.telemetry.observe("last_batch_size", batch_size)

                agent_ids, states, actions, rewards, next_states, dones = zip(*batch)
                return (
                    np.array(agent_ids, dtype=object),
                    np.array(states, dtype=object),
                    np.array(actions, dtype=object),
                    np.array(rewards, dtype=np.float32),
                    np.array(next_states, dtype=object),
                    np.array(dones, dtype=np.bool_),
                )

    def get_all(self) -> Tuple[List[Any], List[Any], List[Any], List[float], List[Any], List[bool]]:
        with self._lock:
            if not self.buffer:
                return [], [], [], [], [], []
            agent_ids, states, actions, rewards, next_states, dones = zip(*self.buffer)
            return (
                list(agent_ids),
                list(states),
                list(actions),
                list(rewards),
                list(next_states),
                list(dones),
            )

    def clear(self) -> None:
        with self._lock:
            self.buffer.clear()
            self.total_seen = 0
            self.total_replaced = 0

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            size = len(self.buffer)
            rewards = [float(exp[3]) for exp in self.buffer] if self.buffer else []
            return {
                "size": size,
                "capacity": self.capacity,
                "total_seen": self.total_seen,
                "total_replaced": self.total_replaced,
                "replacement_rate": (self.total_replaced / self.total_seen) if self.total_seen else 0.0,
                "avg_reward": float(np.mean(rewards)) if rewards else 0.0,
                "max_reward": float(np.max(rewards)) if rewards else None,
                "min_reward": float(np.min(rewards)) if rewards else None,
            }


__all__ = ["ReservoirConfig", "ReservoirReplayBuffer"]
