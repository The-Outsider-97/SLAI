from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, Sequence

import numpy as np

from .utils.config_loader import get_config_section, load_global_config


@dataclass
class EvictionContext:
    overflow: int
    total_items: int
    metadata: Dict[str, Any]


class EvictionPolicy(Protocol):
    def select_index(self, items: Sequence[Any], context: EvictionContext) -> int:
        """Returns index of item to evict."""


class FIFOEviction:
    """Always evicts the oldest item."""

    def select_index(self, items: Sequence[Any], context: EvictionContext) -> int:
        if not items:
            raise ValueError("Cannot evict from empty sequence")
        return 0


class LIFOEviction:
    """Evicts newest item first; useful in some backpressure scenarios."""

    def select_index(self, items: Sequence[Any], context: EvictionContext) -> int:
        if not items:
            raise ValueError("Cannot evict from empty sequence")
        return len(items) - 1


class LargestEpisodeEviction:
    """Evicts largest sequence to quickly recover capacity."""

    def select_index(self, items: Sequence[Any], context: EvictionContext) -> int:
        if not items:
            raise ValueError("Cannot evict from empty sequence")
        lengths = [len(item) if hasattr(item, "__len__") else 1 for item in items]
        return int(np.argmax(lengths))


class AgeRewardHybridEviction:
    """Evicts low-value item using weighted age/reward heuristic.

    Expected item shape if it is sequence-like transition list:
    transition[-1] => done, transition[3] => reward.
    """

    def __init__(self, age_weight: float = 0.7, reward_weight: float = 0.3):
        self.age_weight = float(age_weight)
        self.reward_weight = float(reward_weight)

    def _episode_score(self, episode: Any, idx: int, total: int) -> float:
        age_score = idx / max(1, total - 1)

        reward = 0.0
        if episode and hasattr(episode, "__iter__"):
            rewards = []
            for transition in episode:
                if isinstance(transition, (list, tuple)) and len(transition) >= 4:
                    try:
                        rewards.append(abs(float(transition[3])))
                    except Exception:
                        continue
            reward = float(np.mean(rewards)) if rewards else 0.0

        # Lower score = more likely to evict.
        return self.age_weight * age_score + self.reward_weight * reward

    def select_index(self, items: Sequence[Any], context: EvictionContext) -> int:
        if not items:
            raise ValueError("Cannot evict from empty sequence")

        scores = [self._episode_score(ep, idx=i, total=len(items)) for i, ep in enumerate(items)]
        return int(np.argmin(scores))


def build_eviction_policy(user_config: Optional[Dict[str, Any]] = None) -> EvictionPolicy:
    load_global_config()
    config = dict(get_config_section("eviction") or {})
    if user_config:
        config.update(user_config.get("eviction", {}) if isinstance(user_config, dict) else {})

    policy_name = str(config.get("policy", "fifo")).lower().strip()

    if policy_name == "lifo":
        return LIFOEviction()
    if policy_name == "largest_episode":
        return LargestEpisodeEviction()
    if policy_name == "age_reward_hybrid":
        return AgeRewardHybridEviction(
            age_weight=float(config.get("age_weight", 0.7)),
            reward_weight=float(config.get("reward_weight", 0.3)),
        )
    return FIFOEviction()


__all__ = [
    "EvictionContext",
    "EvictionPolicy",
    "FIFOEviction",
    "LIFOEviction",
    "LargestEpisodeEviction",
    "AgeRewardHybridEviction",
    "build_eviction_policy",
]
