"""Bounded, non-authoritative animation command summaries."""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class AnimationEditRecord:
    operation: str
    scene_id: str
    revision_before: int
    revision_after: int

class AnimationMemory:
    def __init__(self, capacity: int = 256):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("capacity must be positive")
        self._records: deque[AnimationEditRecord] = deque(maxlen=capacity)

    def record(self, item: AnimationEditRecord) -> None:
        self._records.append(item)

    def recent(self) -> tuple[AnimationEditRecord, ...]:
        return tuple(self._records)
