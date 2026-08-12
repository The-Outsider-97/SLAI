"""Bounded uniform replay buffer.

This module provides the dependency-light replay backend used by DQN when
uniform sampling is requested.  Distributed, prioritized, reservoir, and
sequence replay remain separate strategies with their own contracts.
"""

from __future__ import annotations

import random

from collections import deque
from threading import RLock
from typing import Deque, Generic, List, Optional, TypeVar


T = TypeVar("T")

SUPPORTED_REPLAY_BACKENDS = frozenset({"distributed", "prioritized", "uniform"})
_REPLAY_BACKEND_ALIASES = {
    "legacy_per": "prioritized",
    "legacy_uniform": "uniform",
}


def normalize_replay_backend(value: object, *, default: str = "distributed") -> str:
    """Return a canonical replay backend name or reject unsupported input."""

    backend = str(value or default).strip().lower()
    backend = _REPLAY_BACKEND_ALIASES.get(backend, backend)
    if backend not in SUPPORTED_REPLAY_BACKENDS:
        raise ValueError(
            f"Unknown replay backend {value!r}; expected one of "
            f"{sorted(SUPPORTED_REPLAY_BACKENDS)}."
        )
    return backend


class ReplayBuffer(Generic[T]):
    """Thread-safe fixed-capacity buffer with uniform sampling.

    The oldest item is evicted automatically when ``capacity`` is reached.
    Sampling is performed without replacement and never mutates stored data.
    """

    def __init__(self, capacity: int, *, seed: Optional[int] = None) -> None:
        resolved_capacity = int(capacity)
        if resolved_capacity <= 0:
            raise ValueError("ReplayBuffer capacity must be a positive integer.")

        self.capacity = resolved_capacity
        self.buffer: Deque[T] = deque(maxlen=resolved_capacity)
        self._rng = random.Random(seed)
        self._lock = RLock()

    def __len__(self) -> int:
        with self._lock:
            return len(self.buffer)

    def push(self, transition: T) -> None:
        """Append one transition, evicting the oldest item when full."""

        with self._lock:
            self.buffer.append(transition)

    def sample(self, batch_size: int) -> List[T]:
        """Return a uniformly sampled batch without replacement."""

        resolved_size = int(batch_size)
        if resolved_size <= 0:
            raise ValueError("ReplayBuffer batch_size must be a positive integer.")

        with self._lock:
            available = len(self.buffer)
            if resolved_size > available:
                raise ValueError(
                    f"Insufficient replay samples ({available} available, "
                    f"requested {resolved_size})."
                )
            return self._rng.sample(list(self.buffer), resolved_size)

    def clear(self) -> None:
        """Remove all stored transitions."""

        with self._lock:
            self.buffer.clear()

    def get_all(self) -> List[T]:
        """Return a stable snapshot in insertion order."""

        with self._lock:
            return list(self.buffer)


__all__ = ["ReplayBuffer", "SUPPORTED_REPLAY_BACKENDS", "normalize_replay_backend"]
