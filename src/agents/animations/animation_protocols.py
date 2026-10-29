"""Narrow scene-provider contract; application-neutral."""
from __future__ import annotations
from typing import Protocol, Any

class AnimationSceneProvider(Protocol):
    revision: int
    metadata: dict
    def evaluate(self, time: float) -> Any: ...
    def set_animation(self, animation: dict | None) -> int: ...
