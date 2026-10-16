from __future__ import annotations

"""
Transient reasoning-event memory.

ReasoningMemory is intentionally NOT a persistence or checkpoint subsystem.

Durable Agent recovery belongs to:
    BaseAgent -> checkpointing/

This class owns only Reasoning-specific transient event semantics while using
SLAI's generic segment-tree implementation for priority storage.
"""

import math
import random

from collections import Counter, defaultdict, namedtuple
from dataclasses import dataclass
from datetime import datetime
from threading import RLock
from typing import Any, DefaultDict, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

from src.utils.buffer.segment_tree import SumSegmentTree # type: ignore
from .utils.config_loader import *
from .utils.reasoning_errors import *
from .utils.reasoning_helpers import *
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]


logger = get_logger("Reasoning Memory")
printer = PrettyPrinter()


# Retained for v2.3 compatibility with callers importing Transition.
Transition = namedtuple(
    "Transition",
    ["state", "action", "reward", "next_state", "done"],
)


@dataclass(frozen=True)
class MemorySample:
    """One priority-weighted ReasoningMemory sample."""

    index: int
    experience: Any
    priority: float
    probability: float
    importance_weight: float
    metadata: Dict[str, Any]


class ReasoningMemory:
    """
    Bounded transient memory for reasoning events.

    Generic mechanics:
        - SumSegmentTree -> src.utils.buffer

    Domain-specific mechanics retained here:
        - reasoning-event tags;
        - event types;
        - context extraction;
        - Reasoning MemorySample schema.

    No filesystem persistence is performed here.
    """

    def __init__(self, *, config: Optional[Mapping[str, Any]] = None) -> None:
        self.config: Dict[str, Any] = dict(config or load_global_config())
        self.memory_config: Dict[str, Any] = dict(get_config_section("reasoning_memory", self.config, default={}) or {})

        self._lock = RLock()

        self.max_size = bounded_iterations(
            self.memory_config.get("max_size", 10_000),
            minimum=1,
            maximum=10_000_000,
        )

        self.alpha = clamp_confidence(self.memory_config.get("alpha", 0.6))
        self.beta = clamp_confidence(self.memory_config.get("beta", 0.4))
        self.epsilon = self._positive_float("epsilon", self.memory_config.get("epsilon", 5e-5))
        self.default_priority = self._positive_float("default_priority", self.memory_config.get("default_priority", 1.0))
        self.min_priority = self._positive_float("min_priority", self.memory_config.get("min_priority", 1e-6))
        self.max_priority_value = self._positive_float("max_priority_value", self.memory_config.get("max_priority_value", 1_000_000.0))

        if self.max_priority_value < self.min_priority:
            raise ReasoningConfigurationError(
                "reasoning_memory.max_priority_value must be >= min_priority",
                context={
                    "min_priority": self.min_priority,
                    "max_priority_value": self.max_priority_value,
                },
            )

        self.sample_with_replacement = bool(self.memory_config.get("sample_with_replacement", True))
        self.normalize_importance_weights = bool(self.memory_config.get("normalize_importance_weights", True))
        self.strict_priorities = bool(self.memory_config.get("strict_priorities", True))
        self.context_high_priority_threshold = float(self.memory_config.get("context_high_priority_threshold", 0.85))
        self.context_recent_window = bounded_iterations(self.memory_config.get("context_recent_window", 20),
            minimum=1,
            maximum=100_000,
        )
        self.context_high_priority_ratio = clamp_confidence(self.memory_config.get("context_high_priority_ratio", 0.5))
        self.context_saturation_threshold = clamp_confidence(self.memory_config.get("context_saturation_threshold", 0.8))
        self.context_decay_factor = clamp_confidence(self.memory_config.get("context_decay_factor", 0.95))
        self.historical_context_floor = float(self.memory_config.get("historical_context_floor", 0.1))
        self.persistent_context_threshold = float(self.memory_config.get("persistent_context_threshold", 0.5))
        self.max_context_tags = bounded_iterations(self.memory_config.get("max_context_tags", 12),
            minimum=1,
            maximum=1_000,
        )

        seed = self.memory_config.get("seed")
        if (
            isinstance(seed, str)
            and seed.strip().lower()
            in {"", "none", "null"}
        ):
            seed = None

        try:
            numeric_seed = (
                None
                if seed is None
                else int(seed)
            )
        except (TypeError, ValueError) as exc:
            raise ReasoningConfigurationError(
                "reasoning_memory.seed must be an integer or null",
                cause=exc,
                context={"seed": seed},
            ) from exc

        # Per-instance RNG: never modify global Python/NumPy/Torch RNG state.
        self._rng = random.Random(numeric_seed)
        self._priority_tree = SumSegmentTree(self.max_size, dtype="float64")
        self._data: List[Optional[Any]] = [None] * self.max_size
        self._raw_priorities: List[float] = [0.0] * self.max_size
        self.metadata: Dict[int, Dict[str, Any]] = {}
        self.tag_index: DefaultDict[str, Set[int]] = defaultdict(set)
        self.type_index: DefaultDict[str, Set[int]] = defaultdict(set)

        self._write_ptr = 0
        self._size = 0
        self.access_counter = 0
        self.max_priority = self.default_priority

        self.historical_context: Dict[str, float] = {}

        logger.info("ReasoningMemory initialized | capacity=%d | persistent=False", self.max_size)

    # ------------------------------------------------------------------
    # Validation / priority helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _positive_float(name: str, value: Any) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise ReasoningConfigurationError(
                f"reasoning_memory.{name} must be numeric",
                cause=exc,
                context={
                    "key": name,
                    "value": value,
                },
            ) from exc

        if not math.isfinite(parsed) or parsed <= 0.0:
            raise ReasoningConfigurationError(
                f"reasoning_memory.{name} must be finite and > 0",
                context={
                    "key": name,
                    "value": value,
                },
            )

        return parsed

    def _normalize_priority(self, priority: Optional[Any]) -> float:
        raw = (
            self.max_priority
            if priority is None
            else priority
        )

        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise MemoryOperationError(
                "Priority must be numeric",
                cause=exc,
                context={"priority": raw},
            ) from exc

        if not math.isfinite(value):
            raise MemoryOperationError(
                "Priority must be finite",
                context={"priority": raw},
            )

        if value < 0.0:
            if self.strict_priorities:
                raise MemoryOperationError(
                    "Priority must be non-negative",
                    context={"priority": value},
                )
            value = self.min_priority

        return min(
            max(value, self.min_priority),
            self.max_priority_value,
        )

    def _tree_priority(self, raw_priority: float) -> float:
        transformed = (raw_priority + self.epsilon) ** self.alpha

        if (
            not math.isfinite(transformed)
            or transformed <= 0.0
        ):
            raise MemoryOperationError(
                "Transformed priority is invalid",
                context={
                    "raw_priority": raw_priority,
                    "tree_priority": transformed,
                },
            )

        return float(transformed)

    @staticmethod
    def _experience_type(experience: Any) -> Optional[str]:
        if isinstance(experience, Mapping):
            value = experience.get("type")
            if value is not None:
                text = str(value).strip()
                return text or None
        return None

    @staticmethod
    def _normalize_tags(tag: Optional[Union[str, Iterable[str]]], experience: Any) -> Set[str]:
        tags: Set[str] = set()

        if isinstance(tag, str):
            if tag.strip():
                tags.add(tag.strip())
        elif tag is not None:
            for item in tag:
                value = str(item).strip()
                if value:
                    tags.add(value)

        if isinstance(experience, Mapping):
            embedded = (
                experience.get("tags")
                or experience.get("tag")
            )

            if isinstance(embedded, str):
                if embedded.strip():
                    tags.add(embedded.strip())
            elif embedded is not None:
                try:
                    for item in embedded:
                        value = str(item).strip()
                        if value:
                            tags.add(value)
                except TypeError:
                    pass

        return tags

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def _occupied_indices(self) -> List[int]:
        if self._size <= 0:
            return []

        if self._size < self.max_size:
            return list(range(self._size))

        return (
            list(range(self._write_ptr, self.max_size))
            + list(range(0, self._write_ptr))
        )

    def _drop_indexes(self, index: int) -> None:
        meta = self.metadata.pop(index, None)
        if not meta:
            return

        for tag in meta.get("tags", []):
            bucket = self.tag_index.get(tag)
            if bucket is not None:
                bucket.discard(index)
                if not bucket:
                    self.tag_index.pop(tag, None)

        exp_type = meta.get("type")
        if exp_type:
            bucket = self.type_index.get(exp_type)
            if bucket is not None:
                bucket.discard(index)
                if not bucket:
                    self.type_index.pop(exp_type, None)

    def _register_metadata(
        self,
        index: int,
        experience: Any,
        raw_priority: float,
        tree_priority: float,
        tags: Set[str],
    ) -> None:
        exp_type = self._experience_type(experience)

        metadata = {
            "index": index,
            "type": exp_type,
            "tags": sorted(tags),
            "raw_priority": raw_priority,
            "tree_priority": tree_priority,
            "created_at": datetime.now().isoformat(
                timespec="seconds"
            ),
            "access_count": 0,
        }

        self.metadata[index] = metadata

        for tag in tags:
            self.tag_index[tag].add(index)

        if exp_type:
            self.type_index[exp_type].add(index)

    def _touch(self, index: int) -> None:
        meta = self.metadata.get(index)
        if meta is None:
            return

        meta["access_count"] = (int(meta.get("access_count", 0)) + 1)
        meta["last_accessed_at"] = (datetime.now().isoformat(timespec="seconds"))

    # ------------------------------------------------------------------
    # Public mutation API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.size()

    def size(self) -> int:
        with self._lock:
            return self._size

    def add(
        self,
        experience: Any,
        priority: Optional[Any] = None,
        tag: Optional[
            Union[str, Iterable[str]]
        ] = None,
    ) -> int:
        with self._lock:
            raw = self._normalize_priority(priority)
            tree_value = self._tree_priority(raw)

            index = self._write_ptr

            if self._data[index] is not None:
                self._drop_indexes(index)

            self._data[index] = experience
            self._raw_priorities[index] = raw
            self._priority_tree.update(
                index,
                tree_value,
            )

            tags = self._normalize_tags(
                tag,
                experience,
            )
            self._register_metadata(
                index,
                experience,
                raw,
                tree_value,
                tags,
            )

            self._write_ptr = (
                self._write_ptr + 1
            ) % self.max_size

            self._size = min(
                self._size + 1,
                self.max_size,
            )

            self.access_counter += 1
            self.max_priority = max(
                self.max_priority,
                raw,
            )

            return index

    def set(
        self,
        key: int,
        value: Any,
    ) -> bool:
        with self._lock:
            if (
                not isinstance(key, int)
                or key < 0
                or key >= self.max_size
                or self._data[key] is None
            ):
                return False

            old_meta = dict(
                self.metadata.get(key, {})
            )
            raw = self._raw_priorities[key]
            tree_value = self._priority_tree.get(key)

            self._drop_indexes(key)
            self._data[key] = value

            tags = self._normalize_tags(
                old_meta.get("tags", []),
                value,
            )

            self._register_metadata(
                key,
                value,
                raw,
                tree_value,
                tags,
            )

            return True

    def clear(self) -> None:
        with self._lock:
            self._priority_tree.reset()
            self._data = [
                None
            ] * self.max_size
            self._raw_priorities = [
                0.0
            ] * self.max_size

            self.metadata.clear()
            self.tag_index.clear()
            self.type_index.clear()

            self._write_ptr = 0
            self._size = 0
            self.access_counter = 0
            self.max_priority = self.default_priority
            self.historical_context.clear()

    def update_priorities(
        self,
        indices: Sequence[int],
        priorities: Sequence[Any],
    ) -> int:
        if len(indices) != len(priorities):
            raise MemoryOperationError(
                "indices and priorities must have the same length",
                context={
                    "indices": len(indices),
                    "priorities": len(priorities),
                },
            )

        updated = 0

        with self._lock:
            for index, priority in zip(
                indices,
                priorities,
            ):
                if (
                    not isinstance(index, int)
                    or index < 0
                    or index >= self.max_size
                    or self._data[index] is None
                ):
                    continue

                raw = self._normalize_priority(
                    priority
                )
                tree_value = self._tree_priority(
                    raw
                )

                self._raw_priorities[index] = raw
                self._priority_tree.update(
                    index,
                    tree_value,
                )

                meta = self.metadata.get(index)
                if meta is not None:
                    meta["raw_priority"] = raw
                    meta["tree_priority"] = tree_value
                    meta["priority_updated_at"] = (
                        datetime.now().isoformat(
                            timespec="seconds"
                        )
                    )

                self.max_priority = max(
                    self.max_priority,
                    raw,
                )
                updated += 1

        return updated

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get(
        self,
        key: Optional[Union[int, str]] = None,
        default: Any = None,
    ) -> Any:
        with self._lock:
            if key is None:
                return [
                    self._data[index]
                    for index in self._occupied_indices()
                    if self._data[index] is not None
                ]

            if isinstance(key, int):
                if (
                    0 <= key < self.max_size
                    and self._data[key] is not None
                ):
                    self._touch(key)
                    return self._data[key]
                return default

            if isinstance(key, str):
                matches = self.get_by_tag(key)
                return (
                    matches
                    if matches
                    else default
                )

            return default

    def get_by_tag(
        self,
        tag: str,
        *,
        limit: Optional[int] = None,
        newest_first: bool = False,
    ) -> List[Any]:
        normalized = str(tag).strip()
        if not normalized:
            return []

        with self._lock:
            order = {
                index: position
                for position, index
                in enumerate(
                    self._occupied_indices()
                )
            }

            indices = list(
                self.tag_index.get(
                    normalized,
                    set(),
                )
            )

            indices.sort(
                key=lambda idx: order.get(
                    idx,
                    -1,
                ),
                reverse=newest_first,
            )

            if limit is not None:
                indices = indices[
                    : max(0, int(limit))
                ]

            for index in indices:
                self._touch(index)

            return [
                self._data[index]
                for index in indices
                if self._data[index] is not None
            ]

    def get_by_type(
        self,
        experience_type: str,
    ) -> List[Any]:
        normalized = str(
            experience_type
        ).strip()

        if not normalized:
            return []

        with self._lock:
            indices = set(
                self.type_index.get(
                    normalized,
                    set(),
                )
            )

            ordered = [
                index
                for index in self._occupied_indices()
                if index in indices
            ]

            for index in ordered:
                self._touch(index)

            return [
                self._data[index]
                for index in ordered
                if self._data[index] is not None
            ]

    def query(
        self,
        *,
        tag: Optional[str] = None,
        experience_type: Optional[str] = None,
        min_priority: Optional[float] = None,
        limit: Optional[int] = None,
        newest_first: bool = True,
    ) -> List[Any]:
        with self._lock:
            indices = set(
                self._occupied_indices()
            )

            if tag:
                indices &= set(
                    self.tag_index.get(
                        str(tag).strip(),
                        set(),
                    )
                )

            if experience_type:
                indices &= set(
                    self.type_index.get(
                        str(experience_type).strip(),
                        set(),
                    )
                )

            if min_priority is not None:
                floor = float(min_priority)
                indices = {
                    index
                    for index in indices
                    if self._priority_tree.get(
                        index
                    ) >= floor
                }

            order = self._occupied_indices()

            if newest_first:
                order = list(reversed(order))

            ordered = [
                index
                for index in order
                if index in indices
            ]

            if limit is not None:
                ordered = ordered[
                    : max(0, int(limit))
                ]

            for index in ordered:
                self._touch(index)

            return [
                self._data[index]
                for index in ordered
                if self._data[index] is not None
            ]

    def get_high_priority(
        self,
        threshold: float = 0.8,
    ) -> List[Any]:
        return self.query(
            min_priority=float(threshold),
            newest_first=False,
        )

    # ------------------------------------------------------------------
    # Priority sampling
    # ------------------------------------------------------------------

    def sample(
        self,
        batch_size: int,
        *,
        with_importance: bool = True,
    ) -> List[MemorySample]:
        requested = bounded_iterations(
            batch_size,
            minimum=1,
            maximum=self.max_size,
        )

        with self._lock:
            if self._size == 0:
                return []

            if not self.sample_with_replacement:
                requested = min(
                    requested,
                    self._size,
                )

            total = float(
                self._priority_tree.total_mass
            )

            if (
                not math.isfinite(total)
                or total <= 0.0
            ):
                raise MemoryOperationError(
                    "Cannot sample from zero or invalid priority mass",
                    context={
                        "total_priority": total,
                    },
                )

            selected: List[MemorySample] = []
            seen: Set[int] = set()

            attempts = 0
            max_attempts = max(
                requested * 20,
                requested,
            )

            while (
                len(selected) < requested
                and attempts < max_attempts
            ):
                attempts += 1

                if self.sample_with_replacement:
                    segment = (
                        total / requested
                    )
                    position = len(selected)

                    low = segment * position
                    high = segment * (
                        position + 1
                    )
                    mass = low + (
                        self._rng.random()
                        * (high - low)
                    )
                else:
                    mass = (
                        self._rng.random()
                        * total
                    )

                index = (
                    self._priority_tree
                    .prefix_sum_index(mass)
                )

                experience = self._data[index]
                if experience is None:
                    continue

                if (
                    not self.sample_with_replacement
                    and index in seen
                ):
                    continue

                seen.add(index)

                priority = self._priority_tree.get(
                    index
                )
                probability = (
                    priority / total
                )

                importance = 1.0
                if with_importance:
                    importance = (
                        max(
                            probability
                            * self._size,
                            self.epsilon,
                        )
                        ** (-self.beta)
                    )

                self._touch(index)

                selected.append(
                    MemorySample(
                        index=index,
                        experience=experience,
                        priority=priority,
                        probability=probability,
                        importance_weight=importance,
                        metadata=dict(
                            self.metadata.get(
                                index,
                                {},
                            )
                        ),
                    )
                )

            if (
                with_importance
                and selected
                and self.normalize_importance_weights
            ):
                maximum = max(
                    item.importance_weight
                    for item in selected
                ) or 1.0

                selected = [
                    MemorySample(
                        index=item.index,
                        experience=item.experience,
                        priority=item.priority,
                        probability=item.probability,
                        importance_weight=(
                            item.importance_weight
                            / maximum
                        ),
                        metadata=item.metadata,
                    )
                    for item in selected
                ]

            return selected

    def sample_proportional(self, batch_size: int) -> Tuple[List[Any], List[int], List[float]]:
        samples = self.sample(batch_size, with_importance=False)

        return (
            [
                sample.experience
                for sample in samples
            ],
            [
                sample.index
                for sample in samples
            ],
            [
                sample.priority
                for sample in samples
            ],
        )

    # ------------------------------------------------------------------
    # Context / telemetry
    # ------------------------------------------------------------------

    def _decay_historical_context(self) -> None:
        for key in list(self.historical_context):
            value = (
                self.historical_context[key]
                * self.context_decay_factor
            )

            if value < self.historical_context_floor:
                self.historical_context.pop(key, None)
            else:
                self.historical_context[key] = value

    def get_current_context(self) -> List[str]:
        with self._lock:
            if self._size == 0:
                return ["empty_memory"]

            context: List[str] = []

            saturation = (
                self._size
                / self.max_size
            )

            if (
                saturation
                >= self.context_saturation_threshold
            ):
                context.append("memory_saturated")

            occupied = (self._occupied_indices())
            recent = occupied[-min(len(occupied), self.context_recent_window):]

            high_priority = 0
            tags: List[str] = []
            types: List[str] = []

            for index in recent:
                if (
                    self._priority_tree.get(index)
                    >= self.context_high_priority_threshold
                ):
                    high_priority += 1

                meta = self.metadata.get(index, {})
                tags.extend(meta.get("tags", []))
                exp_type = meta.get("type")
                if exp_type:
                    types.append(str(exp_type))

            if (
                recent
                and high_priority
                >= (len(recent) * self.context_high_priority_ratio)
            ):
                context.append("high_priority_context")

            context.extend(
                tag
                for tag, _
                in Counter(tags).most_common(3)
            )

            context.extend(
                f"type:{item_type}"
                for item_type, _
                in Counter(types).most_common(2)
            )

            self._decay_historical_context()

            for item in set(context):
                self.historical_context[item] = (self.historical_context.get(item, 0.0) + 1.0)

            persistent = sorted(
                key
                for key, value
                in self.historical_context.items()
                if value
                >= self.persistent_context_threshold
            )

            return list(
                dict.fromkeys(context + persistent)
            )[: self.max_context_tags]

    def metrics(self) -> Dict[str, Any]:
        with self._lock:
            priorities = [
                self._priority_tree.get(index)
                for index
                in self._occupied_indices()
            ]

            return {
                "size": self._size,
                "capacity": self.max_size,
                "saturation": (
                    self._size
                    / self.max_size
                ),
                "access_counter": self.access_counter,
                "tags": sorted(self.tag_index),
                "types": sorted(self.type_index),
                "total_priority": float(self._priority_tree.total_mass),
                "max_priority": (
                    max(priorities)
                    if priorities
                    else 0.0
                ),
                "min_priority": (
                    min(priorities)
                    if priorities
                    else 0.0
                ),
                "write_ptr": self._write_ptr,

                # Explicit ownership diagnostic.
                "persistent": False,
            }

    # ------------------------------------------------------------------
    # State transfer
    #
    # These methods are serialization-neutral. They do NOT write files.
    # BaseAgent/checkpointing decides whether state is ever made durable.
    # ------------------------------------------------------------------

    def export_state(self) -> Dict[str, Any]:
        with self._lock:
            records = []

            for index in self._occupied_indices():
                if self._data[index] is None:
                    continue

                records.append(
                    {
                        "experience": self._data[index],
                        "raw_priority": self._raw_priorities[index],
                        "metadata": dict(self.metadata.get(index, {})),
                    }
                )

            return json_safe_reasoning_state(
                {
                    "records": records,
                    "access_counter": self.access_counter,
                    "historical_context": dict(self.historical_context)
                }
            )

    def import_state(self, state: Mapping[str, Any]) -> None:
        records = state.get("records", [])

        if (
            not isinstance(records, Sequence)
            or isinstance(records, (str, bytes))):
            raise MemoryOperationError("ReasoningMemory state records must be a sequence")

        self.clear()

        for record in records:
            if not isinstance(record, Mapping):
                continue

            metadata = record.get("metadata", {})

            tags = (
                metadata.get("tags", [])
                if isinstance(metadata, Mapping)
                else []
                )
            self.add(record.get("experience"), priority=record.get("raw_priority", self.default_priority), tag=tags)

        try:
            self.access_counter = int(state.get("access_counter", self.access_counter))
        except (TypeError, ValueError):
            pass

        historical = state.get("historical_context", {})

        if isinstance(historical, Mapping):
            self.historical_context = {
                str(key): float(value)
                for key, value
                in historical.items()
            }


__all__ = [
    "ReasoningMemory",
    "MemorySample",
    "Transition",
]