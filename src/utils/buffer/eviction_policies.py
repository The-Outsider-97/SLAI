from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple, runtime_checkable

import numpy as np  # type: ignore

from .utils.config_loader import get_config_section, load_global_config
from .utils.buffer_errors import *
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Eviction Policy")
printer = PrettyPrinter()


Numeric = Optional[float]


@dataclass(slots=True)
class EvictionContext:
    """Runtime context passed into eviction policies.

    The core fields remain compatible with the original implementation:
    - overflow: number of items that must be removed to fit capacity.
    - total_items: caller-observed item count.
    - metadata: optional per-item signals used by richer policies.

    Supported metadata keys are intentionally generic so callers from replay,
    sequence replay, and network buffer modules can reuse the same policy API:
    rewards, td_errors, surprise_scores, rarity_scores, priorities,
    access_counts, terminal_flags, rare_event_flags, protected_indices,
    pinned_indices, keep_indices, and eviction_exclusions.
    """

    overflow: int = 1
    total_items: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def normalized(self, item_count: int) -> "EvictionContext":
        if item_count <= 0:
            raise EvictionContextError(self, reason="cannot evict from an empty sequence")
        overflow = max(1, int(self.overflow or 1))
        total_items = int(self.total_items or item_count)
        if total_items < item_count:
            total_items = item_count
        if not isinstance(self.metadata, dict):
            raise EvictionContextError(self, reason="metadata must be a dictionary")
        return EvictionContext(overflow=overflow, total_items=total_items, metadata=dict(self.metadata))


@dataclass(frozen=True, slots=True)
class EvictionCandidate:
    """Normalized eviction signals for one item."""

    index: int
    length: int = 1
    recency_score: float = 0.0
    reward_score: float = 0.0
    td_error_score: float = 0.0
    rarity_score: float = 0.0
    priority_score: float = 0.0
    terminal_score: float = 0.0
    protected: bool = False


@runtime_checkable
class EvictionPolicy(Protocol):
    name: str

    def select_index(self, items: Sequence[Any], context: EvictionContext) -> int:
        """Return one valid index to evict."""
        ...

    def select_indices(self, items: Sequence[Any], context: EvictionContext) -> List[int]:
        """Return one or more valid indices to evict."""
        ...


class BaseEvictionPolicy:
    """Shared production-safe base for deterministic eviction policies."""

    name = "base"

    def select_index(self, items: Sequence[Any], context: EvictionContext) -> int:
        return self.select_indices(items, context)[0]

    def select_indices(self, items: Sequence[Any], context: EvictionContext) -> List[int]:
        ctx = self._context(items, context)
        count = min(ctx.overflow, len(items))
        ranked = self._rank_indices(items, ctx)
        if len(ranked) < count:
            raise EvictionSelectionError(self.name, "policy produced too few candidate indices", item_count=len(items))
        selected = [self._ensure_valid_index(idx, len(items)) for idx in ranked[:count]]
        if len(set(selected)) != len(selected):
            raise EvictionSelectionError(self.name, "policy produced duplicate candidate indices", item_count=len(items))
        return selected

    def _rank_indices(self, items: Sequence[Any], context: EvictionContext) -> List[int]:
        raise NotImplementedError

    def _context(self, items: Sequence[Any], context: Optional[EvictionContext]) -> EvictionContext:
        if context is None:
            context = EvictionContext(total_items=len(items))
        return context.normalized(len(items))

    def _ensure_valid_index(self, index: int, size: int) -> int:
        idx = int(index)
        if idx < 0 or idx >= size:
            raise EvictionSelectionError(self.name, f"selected invalid index {idx}", item_count=size)
        return idx

    def _candidate_table(self, items: Sequence[Any], context: EvictionContext) -> List[EvictionCandidate]:
        ctx = self._context(items, context)
        return [_extract_candidate(item=item, index=i, total=len(items), metadata=ctx.metadata) for i, item in enumerate(items)]

    def _rank_by_lowest_score(
        self, scores: Sequence[float], candidates: Sequence[EvictionCandidate], tie_breaker: str = "oldest"
    ) -> List[int]:
        if not scores:
            raise EvictionSelectionError(self.name, "no scores were produced", item_count=0)
        finite_scores = [float(s) if _is_finite_number(s) else 0.0 for s in scores]
        buckets: Dict[float, List[int]] = {}
        for idx, score in enumerate(finite_scores):
            rounded = round(float(score), 12)
            buckets.setdefault(rounded, []).append(idx)
        ranked: List[int] = []
        for score in sorted(buckets):
            tied = list(buckets[score])
            while tied:
                winner = _tie_break(tied, candidates, tie_breaker=tie_breaker)
                ranked.append(winner)
                tied.remove(winner)
        return ranked


class FIFOEviction(BaseEvictionPolicy):
    """Evict the oldest item, assuming buffer order is oldest-to-newest."""

    name = "fifo"

    def _rank_indices(self, items: Sequence[Any], context: EvictionContext) -> List[int]:
        self._context(items, context)
        return list(range(len(items)))


class LIFOEviction(BaseEvictionPolicy):
    """Compatibility policy: evict the newest item first."""

    name = "lifo"

    def _rank_indices(self, items: Sequence[Any], context: EvictionContext) -> List[int]:
        self._context(items, context)
        return list(range(len(items) - 1, -1, -1))


class LargestEpisodeEviction(BaseEvictionPolicy):
    """Compatibility policy: evict the largest sequence/episode first."""

    name = "largest_episode"

    def _rank_indices(self, items: Sequence[Any], context: EvictionContext) -> List[int]:
        self._context(items, context)
        lengths = [_item_length(item) for item in items]
        return sorted(range(len(items)), key=lambda i: (-lengths[i], i))


class LeastSurpriseEviction(BaseEvictionPolicy):
    """Evict low-surprise/common samples while keeping high-TD and rare events.

    This policy is designed for replay and sequence buffers where useful but
    uncommon learning signals should survive capacity pressure. A higher score
    means "keep"; the lowest-scoring candidate is evicted.
    """

    name = "least_surprise"

    def __init__(
        self,
        td_error_weight: float = 0.55,
        rarity_weight: float = 0.30,
        priority_weight: float = 0.10,
        reward_weight: float = 0.05,
        terminal_bonus: float = 0.05,
        protected_bonus: float = 1_000_000.0,
        tie_breaker: str = "oldest",
    ) -> None:
        self.td_error_weight = _non_negative_weight(td_error_weight, "td_error_weight", self.name)
        self.rarity_weight = _non_negative_weight(rarity_weight, "rarity_weight", self.name)
        self.priority_weight = _non_negative_weight(priority_weight, "priority_weight", self.name)
        self.reward_weight = _non_negative_weight(reward_weight, "reward_weight", self.name)
        self.terminal_bonus = _non_negative_weight(terminal_bonus, "terminal_bonus", self.name)
        self.protected_bonus = _non_negative_weight(protected_bonus, "protected_bonus", self.name)
        self.tie_breaker = _normalize_tie_breaker(tie_breaker)
        _ensure_any_weight(self.name, self.td_error_weight, self.rarity_weight, self.priority_weight, self.reward_weight)

    def _rank_indices(self, items: Sequence[Any], context: EvictionContext) -> List[int]:
        candidates = self._candidate_table(items, context)
        td = _normalize([c.td_error_score for c in candidates])
        rarity = _normalize([c.rarity_score for c in candidates])
        priority = _normalize([c.priority_score for c in candidates])
        reward = _normalize([c.reward_score for c in candidates])

        scores = []
        for i, candidate in enumerate(candidates):
            keep_score = (
                self.td_error_weight * td[i]
                + self.rarity_weight * rarity[i]
                + self.priority_weight * priority[i]
                + self.reward_weight * reward[i]
                + self.terminal_bonus * candidate.terminal_score
            )
            if candidate.protected:
                keep_score += self.protected_bonus
            scores.append(keep_score)

        ranked = self._rank_by_lowest_score(scores, candidates, tie_breaker=self.tie_breaker)
        logger.debug("LeastSurprise ranked first index=%s score=%.6f", ranked[0], float(scores[ranked[0]]))
        return ranked


class AgeRewardHybridEviction(BaseEvictionPolicy):
    """Evict the weakest item using age/recency and reward-signal strength.

    Buffer order is assumed oldest-to-newest. Higher recency/reward score means
    keep. The selected index is therefore the lowest combined keep score.
    """

    name = "age_reward_hybrid"

    def __init__(
        self,
        age_weight: float = 0.7,
        reward_weight: float = 0.3,
        priority_weight: float = 0.0,
        terminal_bonus: float = 0.0,
        protected_bonus: float = 1_000_000.0,
        tie_breaker: str = "oldest",
    ) -> None:
        self.age_weight = _non_negative_weight(age_weight, "age_weight", self.name)
        self.reward_weight = _non_negative_weight(reward_weight, "reward_weight", self.name)
        self.priority_weight = _non_negative_weight(priority_weight, "priority_weight", self.name)
        self.terminal_bonus = _non_negative_weight(terminal_bonus, "terminal_bonus", self.name)
        self.protected_bonus = _non_negative_weight(protected_bonus, "protected_bonus", self.name)
        self.tie_breaker = _normalize_tie_breaker(tie_breaker)
        _ensure_any_weight(self.name, self.age_weight, self.reward_weight, self.priority_weight)

    def _rank_indices(self, items: Sequence[Any], context: EvictionContext) -> List[int]:
        candidates = self._candidate_table(items, context)
        rewards = _normalize([c.reward_score for c in candidates])
        priorities = _normalize([c.priority_score for c in candidates])

        scores = []
        for i, candidate in enumerate(candidates):
            keep_score = (
                self.age_weight * candidate.recency_score
                + self.reward_weight * rewards[i]
                + self.priority_weight * priorities[i]
                + self.terminal_bonus * candidate.terminal_score
            )
            if candidate.protected:
                keep_score += self.protected_bonus
            scores.append(keep_score)

        ranked = self._rank_by_lowest_score(scores, candidates, tie_breaker=self.tie_breaker)
        logger.debug("AgeRewardHybrid ranked first index=%s score=%.6f", ranked[0], float(scores[ranked[0]]))
        return ranked


VALID_EVICTION_POLICIES = {
    "fifo": FIFOEviction,
    "least_surprise": LeastSurpriseEviction,
    "leastsurprise": LeastSurpriseEviction,
    "surprise": LeastSurpriseEviction,
    "age_reward_hybrid": AgeRewardHybridEviction,
    "age_reward": AgeRewardHybridEviction,
    "hybrid": AgeRewardHybridEviction,
    "lifo": LIFOEviction,
    "largest_episode": LargestEpisodeEviction,
}


def build_eviction_policy(user_config: Optional[Dict[str, Any]] = None) -> EvictionPolicy:
    """Build an eviction policy from the global buffer config plus optional overrides."""
    load_global_config()
    config = dict(get_config_section("eviction") or {})
    if user_config:
        config.update(user_config.get("eviction", {}) if isinstance(user_config, dict) else {})

    policy_name = str(config.get("policy", "fifo")).lower().strip()
    if not policy_name:
        raise EvictionPolicyError("", "policy name cannot be empty")

    policy_cls = VALID_EVICTION_POLICIES.get(policy_name)
    if policy_cls is None:
        raise UnsupportedEvictionPolicyError(policy_name, sorted(VALID_EVICTION_POLICIES))

    if policy_cls is FIFOEviction:
        return FIFOEviction()
    if policy_cls is LIFOEviction:
        return LIFOEviction()
    if policy_cls is LargestEpisodeEviction:
        return LargestEpisodeEviction()
    if policy_cls is LeastSurpriseEviction:
        return LeastSurpriseEviction(
            td_error_weight=float(config.get("td_error_weight", 0.55)),
            rarity_weight=float(config.get("rarity_weight", 0.30)),
            priority_weight=float(config.get("priority_weight", 0.10)),
            reward_weight=float(config.get("least_surprise_reward_weight", config.get("reward_weight", 0.05))),
            terminal_bonus=float(config.get("terminal_bonus", 0.05)),
            protected_bonus=float(config.get("protected_bonus", 1_000_000.0)),
            tie_breaker=str(config.get("tie_breaker", "oldest")),
        )
    if policy_cls is AgeRewardHybridEviction:
        return AgeRewardHybridEviction(
            age_weight=float(config.get("age_weight", 0.7)),
            reward_weight=float(config.get("reward_weight", 0.3)),
            priority_weight=float(config.get("priority_weight", 0.0)),
            terminal_bonus=float(config.get("terminal_bonus", 0.0)),
            protected_bonus=float(config.get("protected_bonus", 1_000_000.0)),
            tie_breaker=str(config.get("tie_breaker", "oldest")),
        )

    raise EvictionPolicyError(policy_name, "registered policy could not be instantiated")


def evict_indices(items: Sequence[Any], context: Optional[EvictionContext] = None, policy: Optional[EvictionPolicy] = None) -> List[int]:
    """Convenience wrapper for selecting all indices required by context.overflow."""
    resolved_policy = policy or build_eviction_policy()
    resolved_context = context or EvictionContext(total_items=len(items))
    return resolved_policy.select_indices(items, resolved_context)


def _extract_candidate(item: Any, index: int, total: int, metadata: Mapping[str, Any]) -> EvictionCandidate:
    reward = _metadata_float(metadata, ("rewards", "reward_values"), index)
    td_error = _metadata_float(metadata, ("td_errors", "td_error", "surprise_scores", "surprises"), index)
    rarity = _metadata_float(metadata, ("rarity_scores", "rarity", "novelty_scores", "novelty"), index)
    priority = _metadata_float(metadata, ("priorities", "priority_values", "priority"), index)
    terminal = _metadata_bool(metadata, ("terminal_flags", "done_flags", "dones"), index)
    protected = _index_protected(index, metadata)

    item_reward, item_td, item_rarity, item_priority, item_terminal = _extract_item_signals(item)
    reward = _first_number(reward, item_reward, default=0.0)
    td_error = abs(_first_number(td_error, item_td, default=0.0))
    priority = _first_number(priority, item_priority, default=0.0)
    terminal = bool(terminal if terminal is not None else item_terminal)

    if rarity is None:
        rare_flag = _metadata_bool(metadata, ("rare_event_flags", "rare_flags"), index)
        access_count = _metadata_float(metadata, ("access_counts", "sample_counts"), index)
        if rare_flag is True:
            rarity = 1.0
        elif access_count is not None and access_count >= 0:
            rarity = 1.0 / (1.0 + float(access_count))
        else:
            rarity = item_rarity if item_rarity is not None else 0.0

    return EvictionCandidate(
        index=index,
        length=_item_length(item),
        recency_score=(index / max(1, total - 1)),
        reward_score=abs(float(reward)),
        td_error_score=abs(float(td_error)),
        rarity_score=max(0.0, float(rarity)),
        priority_score=max(0.0, float(priority)),
        terminal_score=1.0 if terminal else 0.0,
        protected=protected,
    )


def _extract_item_signals(item: Any) -> Tuple[Numeric, Numeric, Numeric, Numeric, bool]:
    if isinstance(item, Mapping):
        reward = _mapping_float(item, ("reward", "reward_value", "return", "score"))
        td_error = _mapping_float(item, ("td_error", "surprise", "surprise_score"))
        rarity = _mapping_float(item, ("rarity", "rarity_score", "novelty", "novelty_score"))
        priority = _mapping_float(item, ("priority", "priority_score"))
        terminal = _mapping_bool(item, ("done", "terminal", "is_terminal"))
        return reward, td_error, rarity, priority, bool(terminal)

    if _is_transition(item):
        reward = _safe_float(item[3])
        done = bool(item[5]) if len(item) > 5 else False
        return reward, None, None, None, done

    if _is_episode(item):
        rewards: List[float] = []
        terminal = False
        td_values: List[float] = []
        priorities: List[float] = []
        rarities: List[float] = []
        for transition in item:
            r, td, rarity, priority, done = _extract_item_signals(transition)
            if r is not None:
                rewards.append(abs(float(r)))
            if td is not None:
                td_values.append(abs(float(td)))
            if priority is not None:
                priorities.append(max(0.0, float(priority)))
            if rarity is not None:
                rarities.append(max(0.0, float(rarity)))
            terminal = terminal or bool(done)
        return _mean(rewards), _mean(td_values), _mean(rarities), _mean(priorities), terminal

    reward = _attr_float(item, ("reward", "score"))
    td_error = _attr_float(item, ("td_error", "surprise", "surprise_score"))
    rarity = _attr_float(item, ("rarity", "rarity_score", "novelty_score"))
    priority = _attr_float(item, ("priority",))
    terminal = _attr_bool(item, ("done", "terminal", "expired"))
    return reward, td_error, rarity, priority, bool(terminal)


def _metadata_float(metadata: Mapping[str, Any], keys: Iterable[str], index: int) -> Numeric:
    for key in keys:
        if key in metadata:
            value = _indexed_value(metadata[key], index)
            parsed = _safe_float(value)
            if parsed is not None:
                return parsed
    return None


def _metadata_bool(metadata: Mapping[str, Any], keys: Iterable[str], index: int) -> Optional[bool]:
    for key in keys:
        if key in metadata:
            value = _indexed_value(metadata[key], index)
            if isinstance(value, (bool, np.bool_)):
                return bool(value)
    return None


def _indexed_value(value: Any, index: int) -> Any:
    if isinstance(value, Mapping):
        return value.get(index, value.get(str(index)))
    if isinstance(value, np.ndarray):
        return value[index] if 0 <= index < int(value.shape[0]) else None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value[index] if 0 <= index < len(value) else None
    return value


def _index_protected(index: int, metadata: Mapping[str, Any]) -> bool:
    protected_keys = ("protected_indices", "pinned_indices", "keep_indices", "eviction_exclusions")
    for key in protected_keys:
        raw = metadata.get(key)
        if raw is None:
            continue
        if isinstance(raw, Mapping):
            if bool(raw.get(index, raw.get(str(index), False))):
                return True
        elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
            if index in set(int(v) for v in raw if _safe_float(v) is not None):
                return True
    return False


def _mapping_float(mapping: Mapping[str, Any], keys: Iterable[str]) -> Numeric:
    for key in keys:
        if key in mapping:
            parsed = _safe_float(mapping[key])
            if parsed is not None:
                return parsed
    metadata = mapping.get("metadata")
    if isinstance(metadata, Mapping):
        return _mapping_float(metadata, keys)
    return None


def _mapping_bool(mapping: Mapping[str, Any], keys: Iterable[str]) -> Optional[bool]:
    for key in keys:
        if key in mapping and isinstance(mapping[key], (bool, np.bool_)):
            return bool(mapping[key])
    metadata = mapping.get("metadata")
    if isinstance(metadata, Mapping):
        return _mapping_bool(metadata, keys)
    return None


def _attr_float(obj: Any, names: Iterable[str]) -> Numeric:
    for name in names:
        if hasattr(obj, name):
            parsed = _safe_float(getattr(obj, name))
            if parsed is not None:
                return parsed
    metadata = getattr(obj, "metadata", None)
    if isinstance(metadata, Mapping):
        return _mapping_float(metadata, names)
    return None


def _attr_bool(obj: Any, names: Iterable[str]) -> Optional[bool]:
    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            if isinstance(value, (bool, np.bool_)):
                return bool(value)
    metadata = getattr(obj, "metadata", None)
    if isinstance(metadata, Mapping):
        return _mapping_bool(metadata, names)
    return None


def _item_length(item: Any) -> int:
    try:
        if isinstance(item, (str, bytes, bytearray, Mapping)):
            return 1
        return max(1, len(item)) if hasattr(item, "__len__") else 1
    except Exception:
        return 1


def _is_transition(item: Any) -> bool:
    return isinstance(item, (tuple, list)) and len(item) >= 6 and not _is_episode(item)


def _is_episode(item: Any) -> bool:
    if not isinstance(item, (tuple, list)) or not item:
        return False
    first = item[0]
    return isinstance(first, (tuple, list, Mapping)) and not isinstance(first, (str, bytes, bytearray))


def _safe_float(value: Any) -> Numeric:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _is_finite_number(value: Any) -> bool:
    return _safe_float(value) is not None


def _first_number(*values: Numeric, default: float = 0.0) -> float:
    for value in values:
        if value is not None and math.isfinite(float(value)):
            return float(value)
    return float(default)


def _mean(values: Sequence[float]) -> Numeric:
    return float(np.mean(np.asarray(values, dtype=float))) if values else None


def _normalize(values: Sequence[float]) -> List[float]:
    arr = np.asarray([float(v) if _is_finite_number(v) else 0.0 for v in values], dtype=float)
    if arr.size == 0:
        return []
    min_v = float(np.min(arr))
    max_v = float(np.max(arr))
    if math.isclose(max_v, min_v, rel_tol=1e-12, abs_tol=1e-12):
        return [0.0 for _ in arr]
    return [float((v - min_v) / (max_v - min_v)) for v in arr]


def _non_negative_weight(value: Any, name: str, policy: str) -> float:
    parsed = _safe_float(value)
    if parsed is None or parsed < 0.0:
        raise ConfigValueError(name, value, "finite non-negative number", section="eviction")
    return float(parsed)


def _ensure_any_weight(policy: str, *weights: float) -> None:
    if sum(float(w) for w in weights) <= 0.0:
        raise EvictionPolicyError(policy, "at least one scoring weight must be greater than zero")


def _normalize_tie_breaker(value: str) -> str:
    tie_breaker = str(value or "oldest").strip().lower()
    valid = {"oldest", "newest", "shortest", "longest"}
    if tie_breaker not in valid:
        raise ConfigValueError("tie_breaker", value, f"one of {sorted(valid)}", section="eviction")
    return tie_breaker


def _tie_break(indices: Sequence[int], candidates: Sequence[EvictionCandidate], tie_breaker: str) -> int:
    if tie_breaker == "newest":
        return max(indices)
    if tie_breaker == "shortest":
        return min(indices, key=lambda i: (candidates[i].length, i))
    if tie_breaker == "longest":
        return max(indices, key=lambda i: (candidates[i].length, -i))
    return min(indices)


__all__ = [
    "EvictionContext",
    "EvictionPolicy",
    "FIFOEviction",
    "LIFOEviction",
    "LargestEpisodeEviction",
    "LeastSurpriseEviction",
    "AgeRewardHybridEviction",
    "build_eviction_policy",
]


if __name__ == "__main__":
    print("\n=== Running  Eviction Policy ===\n")
    printer.status("TEST", " Eviction Policy initialized", "info")

    items = [
        ("a", "s0", "x", 0.1, "s1", False),
        ("b", "s1", "x", 3.0, "s2", False),
        ("c", "s2", "x", 0.2, "s3", True),
        ("d", "s3", "x", 0.4, "s4", False),
    ]
    ctx = EvictionContext(
        overflow=2,
        total_items=len(items),
        metadata={"td_errors": [0.01, 2.5, 0.2, 0.05], "rare_event_flags": [False, True, False, False]},
    )

    assert FIFOEviction().select_index(items, ctx) == 0
    assert AgeRewardHybridEviction(age_weight=0.7, reward_weight=0.3).select_index(items, ctx) == 0
    assert LeastSurpriseEviction().select_index(items, ctx) in {0, 3}
    assert len(LeastSurpriseEviction().select_indices(items, ctx)) == 2
    assert isinstance(build_eviction_policy({"eviction": {"policy": "least_surprise"}}), LeastSurpriseEviction)
    assert isinstance(build_eviction_policy({"eviction": {"policy": "age_reward_hybrid"}}), AgeRewardHybridEviction)

    printer.status("TEST", " Eviction Policy checks passed", "success")
    print("\n=== Test ran successfully ===\n")
