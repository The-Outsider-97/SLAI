from __future__ import annotations

import os
import random
import tempfile
import time
import uuid
import numpy as np # type: ignore
import torch # type: ignore
import chromadb # type: ignore

from chromadb.utils import embedding_functions # type: ignore
from collections import Counter, defaultdict, namedtuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Any, DefaultDict, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

from .utils.config_loader import load_global_config, get_config_section
from .utils.reasoning_errors import *
from .utils.reasoning_helpers import *
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Reasoning Memory")
printer = PrettyPrinter()

CHROMA_AVAILABLE = True

if hasattr(torch.serialization, "add_safe_globals"):
    _np_core = getattr(np, "_core", None)
    if _np_core is not None and hasattr(_np_core, "multiarray"):
        torch.serialization.add_safe_globals([_np_core.multiarray._reconstruct])


Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])


@dataclass(frozen=True)
class MemorySample:
    """Structured sample returned by the richer sampling API.

    The legacy ``sample_proportional`` method still returns the original
    tuple ``(samples, indices, priorities)`` for existing subsystem callers.
    """

    index: int
    experience: Any
    priority: float
    probability: float
    importance_weight: float
    metadata: Dict[str, Any]


class SumTree:
    """Efficient proportional-priority store with O(log n) sampling.

    The tree stores transformed priorities in its leaves and arbitrary Python
    experiences in a parallel object array.  ``write_ptr`` implements FIFO
    replacement once capacity is reached, while the parent tree nodes keep the
    proportional sampling total up to date.
    """

    def __init__(self, capacity: int):
        self.capacity = bounded_iterations(capacity, minimum=1, maximum=10_000_000)
        self.tree = np.zeros(2 * self.capacity - 1, dtype=np.float64)
        self.data = np.empty(self.capacity, dtype=object)
        self.data[:] = None
        self.size = 0
        self.write_ptr = 0
        self.max_priority = 1.0

    @property
    def leaf_start(self) -> int:
        return self.capacity - 1

    def __len__(self) -> int:
        return int(self.size)

    def _validate_index(self, data_idx: int) -> int:
        try:
            idx = int(data_idx)
        except (TypeError, ValueError) as exc:
            raise MemoryOperationError("Memory index must be an integer", cause=exc, context={"index": data_idx}) from exc
        if idx < 0 or idx >= self.capacity:
            raise MemoryOperationError(
                "Memory index is outside SumTree capacity",
                context={"index": idx, "capacity": self.capacity},
            )
        return idx

    @staticmethod
    def _validate_priority(priority: Any) -> float:
        try:
            value = float(priority)
        except (TypeError, ValueError) as exc:
            raise MemoryOperationError("Priority must be numeric", cause=exc, context={"priority": priority}) from exc
        if not np.isfinite(value) or value < 0.0:
            raise MemoryOperationError("Priority must be finite and non-negative", context={"priority": priority})
        return value

    def _propagate(self, tree_idx: int, delta: float) -> None:
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += delta

    def total(self) -> float:
        """Return total transformed priority stored at the root."""
        return float(self.tree[0])

    def leaf_priority(self, data_idx: int) -> float:
        """Return transformed priority for a data slot."""
        idx = self._validate_index(data_idx)
        return float(self.tree[idx + self.leaf_start])

    def leaf_priorities(self) -> np.ndarray:
        """Return a copy of all leaf priorities."""
        return self.tree[self.leaf_start : self.leaf_start + self.capacity].copy()

    def occupied_indices(self) -> List[int]:
        """Return occupied data slots in chronological order."""
        if self.size <= 0:
            return []
        if self.size < self.capacity:
            return [idx for idx in range(self.write_ptr) if self.data[idx] is not None]
        return [idx for idx in range(self.write_ptr, self.capacity)] + [idx for idx in range(0, self.write_ptr)]

    def add(self, priority: Any, data: Any) -> int:
        """Insert experience and return the written data index."""
        safe_priority = self._validate_priority(priority)
        data_idx = self.write_ptr
        tree_idx = data_idx + self.leaf_start

        self.data[data_idx] = data
        delta = safe_priority - self.tree[tree_idx]
        self.tree[tree_idx] = safe_priority
        self._propagate(tree_idx, delta)

        self.write_ptr = (self.write_ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.max_priority = max(float(self.max_priority), safe_priority)
        return data_idx

    def update(self, data_idx: int, priority: Any) -> None:
        """Update transformed priority for an existing data slot."""
        idx = self._validate_index(data_idx)
        safe_priority = self._validate_priority(priority)
        tree_idx = idx + self.leaf_start
        delta = safe_priority - self.tree[tree_idx]
        self.tree[tree_idx] = safe_priority
        self._propagate(tree_idx, delta)
        self.max_priority = max(float(self.max_priority), safe_priority)

    def sample(self, value: float) -> Tuple[int, float, Any]:
        """Sample an experience by cumulative priority value."""
        total_priority = self.total()
        if self.size <= 0 or total_priority <= 0.0:
            raise MemoryOperationError("Cannot sample from an empty or zero-priority SumTree")

        cumulative = min(max(float(value), 0.0), np.nextafter(total_priority, 0.0))
        tree_idx = 0
        while True:
            left = 2 * tree_idx + 1
            right = left + 1
            if left >= len(self.tree):
                break
            if cumulative <= self.tree[left]:
                tree_idx = left
            else:
                cumulative -= self.tree[left]
                tree_idx = right

        data_idx = tree_idx - self.leaf_start
        return data_idx, float(self.tree[tree_idx]), self.data[data_idx]

    def reset(self) -> None:
        """Clear all priorities and stored experiences without changing capacity."""
        self.tree.fill(0.0)
        self.data[:] = None
        self.size = 0
        self.write_ptr = 0
        self.max_priority = 1.0


class ReasoningMemory:
    """Thread-safe prioritized reasoning memory with checkpoint support.

    This class is intentionally API-compatible with the previous subsystem
    memory object while adding production behavior:
    - bounded, validated SumTree replay storage;
    - tag/type metadata indexes;
    - proportional and importance-weighted sampling;
    - atomic checkpoint persistence and restoration;
    - context generation from recent/high-priority cognitive state;
    - structured metrics and diagnostics.
    """

    CHECKPOINT_VERSION = "2.1.0"

    def __init__(self):
        """Initialize reasoning memory using the existing config loader flow."""
        self.reasoning_memory: List[Any] = []
        self.config = load_global_config()
        self.memory_config = get_config_section("reasoning_memory")
        self._refresh_runtime_config()

        self.lock = RLock()
        self.tag_index: DefaultDict[str, Set[int]] = defaultdict(set)
        self.type_index: DefaultDict[str, Set[int]] = defaultdict(set)
        self.metadata: Dict[int, Dict[str, Any]] = {}
        self.historical_context: Dict[str, float] = {}
        self.access_counter = 0
        self.max_priority = self.default_priority
        self.last_checkpoint_path: Optional[str] = None

        enable_episodic: bool = False
        collection_name: str = "reasoning_episodes"
        self.enable_episodic = enable_episodic and CHROMA_AVAILABLE
        self.episodic_client = None
        self.episodic_collection = None
        if self.enable_episodic:
            self.episodic_client = chromadb.Client()
            self.episodic_collection = self.episodic_client.get_or_create_collection(
                name=collection_name,
                embedding_function=embedding_functions.DefaultEmbeddingFunction(),
            )

        self.tree = SumTree(self.max_size)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Reasoning Memory initialized with prioritized SumTree")

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    def _get_float_config(self, key: str, default: float) -> float:
        value = self.memory_config.get(key)
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ReasoningConfigurationError(
                f"reasoning_memory.{key} must be convertible to float",
                context={"key": key, "value": value},
            ) from exc

    def _refresh_runtime_config(self) -> None:
        """Validate and cache frequently used config values."""
        self.max_size = bounded_iterations(self.memory_config.get("max_size"), minimum=1, maximum=10_000_000)
        self.checkpoint_freq = bounded_iterations(self.memory_config.get("checkpoint_freq"), minimum=1, maximum=10_000_000)
        self.checkpoint_dir = Path(str(self.memory_config.get("checkpoint_dir"))).expanduser()
        self.auto_save = bool(self.memory_config.get("auto_save"))
        self.alpha = clamp_confidence(self.memory_config.get("alpha"), lower=0.0, upper=1.0)
        self.beta = clamp_confidence(self.memory_config.get("beta"), lower=0.0, upper=1.0)
        self.epsilon = self._get_float_config("epsilon", 0.00005)
        self.default_priority = self._get_float_config("default_priority", 1.0)
        self.min_priority = self._get_float_config("min_priority", 0.000001)
        self.max_priority_value = self._get_float_config("max_priority_value", 1000000.0)
        self.context_high_priority_threshold = self._get_float_config("context_high_priority_threshold", 0.85)
        self.historical_context_floor = self._get_float_config("historical_context_floor", 0.1)
        self.persistent_context_threshold = self._get_float_config("persistent_context_threshold", 0.5)
        self.sample_with_replacement = bool(self.memory_config.get("sample_with_replacement"))
        self.normalize_importance_weights = bool(self.memory_config.get("normalize_importance_weights"))
        self.strict_priorities = bool(self.memory_config.get("strict_priorities"))
        self.checkpoint_keep_last = bounded_iterations(self.memory_config.get("checkpoint_keep_last"), minimum=0, maximum=100000)
        self.checkpoint_prefix = str(self.memory_config.get("checkpoint_prefix") or "memory")
        self.context_recent_window = bounded_iterations(self.memory_config.get("context_recent_window"), minimum=1, maximum=100000)
        self.context_high_priority_ratio = clamp_confidence(self.memory_config.get("context_high_priority_ratio"), lower=0.0, upper=1.0)
        self.context_saturation_threshold = clamp_confidence(self.memory_config.get("context_saturation_threshold"), lower=0.0, upper=1.0)
        self.context_decay_factor = clamp_confidence(self.memory_config.get("context_decay_factor"), lower=0.0, upper=1.0)
        self.max_context_tags = bounded_iterations(self.memory_config.get("max_context_tags"), minimum=1, maximum=1000)

        seed = self.memory_config.get("seed")
        if seed is not None:
            # Handle YAML string "None" (literal) as Python None
            if isinstance(seed, str) and seed.strip().lower() == "none":
                seed = None
            if seed is not None:
                try:
                    numeric_seed = int(seed)
                except (TypeError, ValueError):
                    raise ReasoningConfigurationError(
                        "reasoning_memory.seed must be an integer or None",
                        context={"seed": seed}
                    ) from None
                random.seed(numeric_seed)
                np.random.seed(numeric_seed)

        if self.epsilon <= 0.0:
            raise ReasoningConfigurationError("reasoning_memory.epsilon must be positive", context={"epsilon": self.epsilon})
        if self.min_priority <= 0.0:
            raise ReasoningConfigurationError("reasoning_memory.min_priority must be positive", context={"min_priority": self.min_priority})
        if self.max_priority_value < self.min_priority:
            raise ReasoningConfigurationError(
                "reasoning_memory.max_priority_value must be >= min_priority",
                context={"min_priority": self.min_priority, "max_priority_value": self.max_priority_value},
            )

    # ------------------------------------------------------------------
    # Priority / metadata utilities
    # ------------------------------------------------------------------
    def store_episode(self, state: Dict[str, Any], action: str, outcome: Any, reward: float, metadata: Optional[Dict] = None):
        if not self.enable_episodic or self.episodic_collection is None:
            return
        episode_id = str(uuid.uuid4())
        document = f"State: {json.dumps(state, default=str)}\nAction: {action}\nOutcome: {outcome}\nReward: {reward}"
        self.episodic_collection.add(
            ids=[episode_id],
            documents=[document],
            metadatas=[{"timestamp": time.time(), "action": action, "reward": reward, **(metadata or {})}]
        )

    def retrieve_similar_episodes(self, current_state: Dict[str, Any], n_results: int = 5):
        if not self.enable_episodic or self.episodic_collection is None:
            return []
        query_text = json.dumps(current_state, default=str)
        results = self.episodic_collection.query(query_texts=[query_text], n_results=n_results)
        episodes = []
        for i in range(len(results['ids'][0])):
            episodes.append({
                "id": results['ids'][0][i],
                "document": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "distance": results['distances'][0][i] if 'distances' in results else None
            })
        return episodes

    def _normalize_priority(self, priority: Optional[Any]) -> float:
        """Normalize raw priority before PER alpha transformation."""
        raw = self.max_priority if priority is None else priority
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise MemoryOperationError("Priority must be numeric", cause=exc, context={"priority": raw}) from exc

        if not np.isfinite(value):
            raise MemoryOperationError("Priority must be finite", context={"priority": raw})
        if value < 0.0:
            if self.strict_priorities:
                raise MemoryOperationError("Priority must be non-negative", context={"priority": raw})
            value = self.min_priority
        return min(max(value, self.min_priority), self.max_priority_value)

    def _to_tree_priority(self, raw_priority: Optional[Any]) -> Tuple[float, float]:
        raw = self._normalize_priority(raw_priority)
        transformed = float((raw + self.epsilon) ** self.alpha)
        if not np.isfinite(transformed) or transformed <= 0.0:
            raise MemoryOperationError(
                "Transformed priority is invalid",
                context={"raw_priority": raw, "transformed_priority": transformed},
            )
        return raw, transformed

    @staticmethod
    def _normalize_tags(tag: Optional[Union[str, Iterable[str]]], experience: Any) -> Set[str]:
        tags: Set[str] = set()
        if tag is not None:
            if isinstance(tag, str):
                tags.add(tag.strip())
            else:
                tags.update(str(item).strip() for item in tag if str(item).strip())
        if isinstance(experience, Mapping):
            embedded = experience.get("tag") or experience.get("tags")
            if isinstance(embedded, str):
                tags.add(embedded.strip())
            elif isinstance(embedded, Iterable):
                tags.update(str(item).strip() for item in embedded if str(item).strip())
        return {item for item in tags if item}

    @staticmethod
    def _experience_type(experience: Any) -> Optional[str]:
        if isinstance(experience, Mapping) and experience.get("type") is not None:
            value = str(experience.get("type")).strip()
            return value or None
        if isinstance(experience, Transition):
            return "transition"
        return None

    def _drop_index_from_indexes(self, data_idx: int) -> None:
        old_meta = self.metadata.pop(data_idx, None)
        if not old_meta:
            return
        for tag in old_meta.get("tags", []):
            bucket = self.tag_index.get(tag)
            if bucket is not None:
                bucket.discard(data_idx)
                if not bucket:
                    self.tag_index.pop(tag, None)
        old_type = old_meta.get("type")
        if old_type:
            bucket = self.type_index.get(old_type)
            if bucket is not None:
                bucket.discard(data_idx)
                if not bucket:
                    self.type_index.pop(old_type, None)

    def _register_metadata(self, data_idx: int, experience: Any, raw_priority: float, tree_priority: float, tags: Set[str]) -> None:
        exp_type = self._experience_type(experience)
        now = datetime.now().isoformat(timespec="seconds")
        meta = {
            "index": data_idx,
            "type": exp_type,
            "tags": sorted(tags),
            "raw_priority": raw_priority,
            "tree_priority": tree_priority,
            "created_at": now,
            "created_at_ms": monotonic_timestamp_ms(),
            "access_count": 0,
        }
        self.metadata[data_idx] = meta
        for tag in tags:
            self.tag_index[tag].add(data_idx)
        if exp_type:
            self.type_index[exp_type].add(data_idx)

    def _touch(self, data_idx: int) -> None:
        meta = self.metadata.get(data_idx)
        if meta is not None:
            meta["access_count"] = int(meta.get("access_count", 0)) + 1
            meta["last_accessed_at"] = datetime.now().isoformat(timespec="seconds")
            meta["last_accessed_at_ms"] = monotonic_timestamp_ms()

    def _occupied_indices(self) -> List[int]:
        return self.tree.occupied_indices()

    # ------------------------------------------------------------------
    # Legacy-compatible public API
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self.size()

    def size(self) -> int:
        return len(self.tree)

    def add(self, experience: Any, priority: Optional[Any] = None, tag: Optional[Union[str, Iterable[str]]] = None) -> int:
        """Add an experience and return its memory index.

        Existing callers do not rely on the return value, but returning the
        index makes the method more useful for targeted priority updates.
        """
        with self.lock:
            raw_priority, tree_priority = self._to_tree_priority(priority)
            target_idx = self.tree.write_ptr
            if self.tree.data[target_idx] is not None:
                self._drop_index_from_indexes(target_idx)

            data_idx = self.tree.add(tree_priority, experience)
            tags = self._normalize_tags(tag, experience)
            self._register_metadata(data_idx, experience, raw_priority, tree_priority, tags)

            self.access_counter += 1
            self.max_priority = max(self.max_priority, raw_priority)

            if self.auto_save and self.access_counter % self.checkpoint_freq == 0:
                self.save_checkpoint()
            return data_idx

    def get(self, key: Optional[Union[int, str]] = None, default: Any = None) -> Any:
        """Retrieve by index/tag or return all occupied experiences."""
        with self.lock:
            if key is None:
                return [self.tree.data[idx] for idx in self._occupied_indices() if self.tree.data[idx] is not None]
            if isinstance(key, int):
                if 0 <= key < self.tree.capacity and self.tree.data[key] is not None:
                    self._touch(key)
                    return self.tree.data[key]
                return default
            if isinstance(key, str):
                matches = self.get_by_tag(key)
                return matches if matches else default
            return default

    def set(self, key: int, value: Any) -> bool:
        """Update an experience at a specific index while preserving priority."""
        with self.lock:
            if not isinstance(key, int) or key < 0 or key >= self.tree.capacity:
                return False
            if self.tree.data[key] is None:
                return False

            old_meta = dict(self.metadata.get(key, {}))
            raw_priority = float(old_meta.get("raw_priority", self.default_priority))
            tree_priority = self.tree.leaf_priority(key)
            self._drop_index_from_indexes(key)
            self.tree.data[key] = value
            tags = self._normalize_tags(old_meta.get("tags", []), value)
            self._register_metadata(key, value, raw_priority, tree_priority, tags)
            self.access_counter += 1
            return True

    def update_priorities(self, indices: Sequence[int], priorities: Sequence[Any]) -> int:
        """Update raw priorities for specific memory indices."""
        if len(indices) != len(priorities):
            raise MemoryOperationError(
                "indices and priorities must have the same length",
                context={"indices": len(indices), "priorities": len(priorities)},
            )
        updated = 0
        with self.lock:
            for idx, priority in zip(indices, priorities):
                if not isinstance(idx, int) or idx < 0 or idx >= self.tree.capacity or self.tree.data[idx] is None:
                    logger.warning(f"Skipping invalid memory priority update index: {idx}")
                    continue
                raw_priority, tree_priority = self._to_tree_priority(priority)
                self.tree.update(idx, tree_priority)
                if idx in self.metadata:
                    self.metadata[idx]["raw_priority"] = raw_priority
                    self.metadata[idx]["tree_priority"] = tree_priority
                    self.metadata[idx]["priority_updated_at"] = datetime.now().isoformat(timespec="seconds")
                self.max_priority = max(self.max_priority, raw_priority)
                updated += 1
        return updated

    def sample_proportional(self, batch_size: int) -> Tuple[List[Any], List[int], List[float]]:
        """Sample experiences proportional to priority.

        Returns the legacy tuple ``(samples, indices, transformed_priorities)``.
        """
        samples = self.sample(batch_size=batch_size, with_importance=False)
        return (
            [sample.experience for sample in samples],
            [sample.index for sample in samples],
            [sample.priority for sample in samples],
        )

    def save_checkpoint(self, name: Optional[str] = None) -> str:
        """Atomically save memory state to disk and return the checkpoint path."""
        with self.lock:
            checkpoint_path = self._resolve_checkpoint_path(name)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            payload = self._checkpoint_payload()

            fd, tmp_name = tempfile.mkstemp(prefix=f".{checkpoint_path.name}.", suffix=".tmp", dir=str(checkpoint_path.parent))
            os.close(fd)
            tmp_path = Path(tmp_name)
            try:
                torch.save(payload, tmp_path)
                os.replace(tmp_path, checkpoint_path)
            except Exception as exc:
                if tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)
                raise KnowledgePersistenceError(
                    "Failed to save reasoning memory checkpoint",
                    cause=exc,
                    context={"path": str(checkpoint_path)},
                ) from exc

            self.last_checkpoint_path = str(checkpoint_path)
            self._cleanup_old_checkpoints()
            logger.info(f"Memory checkpoint saved to {checkpoint_path}")
            return str(checkpoint_path)

    def load_checkpoint(self, path: Union[str, os.PathLike[str]]) -> bool:
        """Load memory state from a checkpoint file."""
        checkpoint_path = Path(path).expanduser()
        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint file {checkpoint_path} not found")
            return False

        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except Exception as exc:
            raise KnowledgePersistenceError(
                "Failed to load reasoning memory checkpoint",
                cause=exc,
                context={"path": str(checkpoint_path)},
            ) from exc

        with self.lock:
            self._restore_checkpoint(checkpoint, source=str(checkpoint_path))
            self.last_checkpoint_path = str(checkpoint_path)
        logger.info(f"Loaded memory checkpoint from {checkpoint_path}")
        return True

    def clear(self) -> None:
        """Reset memory while preserving loaded configuration."""
        with self.lock:
            self.tree = SumTree(self.max_size)
            self.tag_index = defaultdict(set)
            self.type_index = defaultdict(set)
            self.metadata = {}
            self.historical_context = {}
            self.access_counter = 0
            self.max_priority = self.default_priority

    def get_by_type(self, experience_type: str) -> List[Any]:
        """Get experiences by their ``type`` field."""
        normalized = str(experience_type).strip()
        if not normalized:
            return []
        with self.lock:
            indices = sorted(self.type_index.get(normalized, set()))
            if not indices:
                indices = [idx for idx in self._occupied_indices() if self._experience_type(self.tree.data[idx]) == normalized]
            for idx in indices:
                self._touch(idx)
            return [self.tree.data[idx] for idx in indices if self.tree.data[idx] is not None]

    def get_high_priority(self, threshold: float = 0.8) -> List[Any]:
        """Get experiences whose transformed priority meets a threshold."""
        safe_threshold = float(threshold)
        with self.lock:
            indices = [idx for idx in self._occupied_indices() if self.tree.leaf_priority(idx) >= safe_threshold]
            for idx in indices:
                self._touch(idx)
            return [self.tree.data[idx] for idx in indices if self.tree.data[idx] is not None]

    def metrics(self) -> Dict[str, Any]:
        """Return memory system statistics and health telemetry."""
        with self.lock:
            total_priority = self.tree.total()
            priorities = [self.tree.leaf_priority(idx) for idx in self._occupied_indices()]
            return {
                "size": len(self.tree),
                "capacity": self.tree.capacity,
                "saturation": len(self.tree) / max(1, self.tree.capacity),
                "access_counter": self.access_counter,
                "tags": sorted(self.tag_index.keys()),
                "types": sorted(self.type_index.keys()),
                "total_priority": total_priority,
                "max_priority": self.max_priority,
                "max_tree_priority": max(priorities) if priorities else 0.0,
                "min_tree_priority": min(priorities) if priorities else 0.0,
                "write_ptr": self.tree.write_ptr,
                "last_checkpoint_path": self.last_checkpoint_path,
                "checkpoint_dir": str(self.checkpoint_dir),
            }

    def get_current_context(self) -> List[str]:
        """Generate contextual tags from recent, high-priority memory state."""
        with self.lock:
            if self.tree.size == 0:
                return ["empty_memory"]

            context_tags: List[str] = []
            saturation = self.tree.size / max(1, self.tree.capacity)
            if saturation >= self.context_saturation_threshold:
                context_tags.append("memory_saturated")

            recent_indices = self._get_recent_indices(min(self.context_recent_window, self.tree.size))
            high_priority_count = 0
            recent_tags: List[str] = []
            recent_types: List[str] = []

            for data_idx in recent_indices:
                priority = self.tree.leaf_priority(data_idx)
                if priority >= self.context_high_priority_threshold:
                    high_priority_count += 1
                meta = self.metadata.get(data_idx, {})
                recent_tags.extend(meta.get("tags", []))
                if meta.get("type"):
                    recent_types.append(str(meta["type"]))

            if recent_indices and high_priority_count >= len(recent_indices) * self.context_high_priority_ratio:
                context_tags.append("high_priority_context")

            for tag, _ in Counter(recent_tags).most_common(3):
                context_tags.append(tag)
            for item_type, _ in Counter(recent_types).most_common(2):
                context_tags.append(f"type:{item_type}")

            hour = datetime.now().hour
            if 5 <= hour < 12:
                context_tags.append("morning_context")
            elif 12 <= hour < 18:
                context_tags.append("afternoon_context")
            elif 18 <= hour < 22:
                context_tags.append("evening_context")
            else:
                context_tags.append("night_context")

            self._decay_historical_context()
            for tag in set(context_tags):
                self.historical_context[tag] = self.historical_context.get(tag, 0.0) + 1.0

            persistent_tags = [
                tag for tag, weight in self.historical_context.items() if weight >= self.persistent_context_threshold
            ]
            ordered = list(dict.fromkeys(context_tags + sorted(persistent_tags)))
            return ordered[: self.max_context_tags]

    def _get_recent_indices(self, count: int) -> List[int]:
        """Get recent indices in chronological order."""
        if count <= 0 or self.tree.size <= 0:
            return []
        all_indices = self._occupied_indices()
        return all_indices[-min(count, len(all_indices)) :]

    # ------------------------------------------------------------------
    # Extended production API
    # ------------------------------------------------------------------
    def sample(self, batch_size: int, *, with_importance: bool = True) -> List[MemorySample]:
        """Sample memory entries and optionally compute PER importance weights."""
        limit = bounded_iterations(batch_size, minimum=1, maximum=max(1, self.tree.capacity))
        with self.lock:
            if len(self.tree) == 0:
                return []
            if not self.sample_with_replacement:
                limit = min(limit, len(self.tree))

            total_priority = self.tree.total()
            if total_priority <= 0.0:
                raise MemoryOperationError("Cannot sample because total priority is zero")

            selected: List[MemorySample] = []
            seen: Set[int] = set()
            attempts = 0
            max_attempts = max(limit * 10, limit)

            while len(selected) < limit and attempts < max_attempts:
                attempts += 1
                if self.sample_with_replacement:
                    segment = total_priority / limit
                    pos = len(selected)
                    value = random.uniform(segment * pos, segment * (pos + 1))
                else:
                    value = random.uniform(0.0, total_priority)

                idx, priority, data = self.tree.sample(value)
                if data is None:
                    continue
                if not self.sample_with_replacement and idx in seen:
                    continue
                seen.add(idx)
                probability = priority / total_priority if total_priority > 0 else 0.0
                importance = self._importance_weight(probability, len(self.tree)) if with_importance else 1.0
                self._touch(idx)
                selected.append(
                    MemorySample(
                        index=idx,
                        experience=data,
                        priority=priority,
                        probability=probability,
                        importance_weight=importance,
                        metadata=dict(self.metadata.get(idx, {})),
                    )
                )

            if with_importance and selected and self.normalize_importance_weights:
                max_weight = max(sample.importance_weight for sample in selected) or 1.0
                selected = [
                    MemorySample(
                        index=sample.index,
                        experience=sample.experience,
                        priority=sample.priority,
                        probability=sample.probability,
                        importance_weight=sample.importance_weight / max_weight,
                        metadata=sample.metadata,
                    )
                    for sample in selected
                ]
            return selected

    def get_by_tag(self, tag: str, *, limit: Optional[int] = None, newest_first: bool = False) -> List[Any]:
        """Return experiences carrying a tag."""
        normalized = str(tag).strip()
        if not normalized:
            return []
        with self.lock:
            indices = list(self.tag_index.get(normalized, set()))
            order_lookup = {idx: pos for pos, idx in enumerate(self._occupied_indices())}
            indices.sort(key=lambda idx: order_lookup.get(idx, -1), reverse=newest_first)
            if limit is not None:
                indices = indices[: bounded_iterations(limit, minimum=1, maximum=max(1, len(indices)))]
            for idx in indices:
                self._touch(idx)
            return [self.tree.data[idx] for idx in indices if self.tree.data[idx] is not None]

    def query(
        self,
        *,
        tag: Optional[str] = None,
        experience_type: Optional[str] = None,
        min_priority: Optional[float] = None,
        limit: Optional[int] = None,
        newest_first: bool = True,
    ) -> List[Any]:
        """Query memory by tag, type, and/or minimum transformed priority."""
        with self.lock:
            indices = set(self._occupied_indices())
            if tag:
                indices &= set(self.tag_index.get(str(tag).strip(), set()))
            if experience_type:
                indices &= set(self.type_index.get(str(experience_type).strip(), set()))
            if min_priority is not None:
                floor = float(min_priority)
                indices = {idx for idx in indices if self.tree.leaf_priority(idx) >= floor}

            order_lookup = {idx: pos for pos, idx in enumerate(self._occupied_indices())}
            ordered = sorted(indices, key=lambda idx: order_lookup.get(idx, -1), reverse=newest_first)
            if limit is not None:
                ordered = ordered[: bounded_iterations(limit, minimum=1, maximum=max(1, len(ordered)))]
            for idx in ordered:
                self._touch(idx)
            return [self.tree.data[idx] for idx in ordered if self.tree.data[idx] is not None]

    def remove(self, index: int) -> bool:
        """Remove a memory entry by index and clear its priority."""
        with self.lock:
            if not isinstance(index, int) or index < 0 or index >= self.tree.capacity or self.tree.data[index] is None:
                return False
            self._drop_index_from_indexes(index)
            self.tree.data[index] = None
            self.tree.update(index, 0.0)
            self.tree.size = max(0, sum(1 for item in self.tree.data if item is not None))
            return True

    def export_state(self) -> Dict[str, Any]:
        """Return JSON-safe diagnostics without serializing raw experiences."""
        with self.lock:
            return json_safe_reasoning_state(
                {
                    "version": self.CHECKPOINT_VERSION,
                    "metrics": self.metrics(),
                    "metadata": self.metadata,
                    "historical_context": self.historical_context,
                }
            )

    # ------------------------------------------------------------------
    # Checkpoint internals
    # ------------------------------------------------------------------
    def _resolve_checkpoint_path(self, name: Optional[str]) -> Path:
        if name:
            candidate = Path(name).expanduser()
            if candidate.is_absolute() or candidate.parent != Path("."):
                return candidate
            return self.checkpoint_dir / candidate
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return self.checkpoint_dir / f"{self.checkpoint_prefix}_{stamp}.pt"

    def _checkpoint_payload(self) -> Dict[str, Any]:
        return {
            "version": self.CHECKPOINT_VERSION,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "tree_capacity": self.tree.capacity,
            "tree_data": self.tree.data,
            "tree_structure": self.tree.tree,
            "tree_write_ptr": self.tree.write_ptr,
            "tree_size": self.tree.size,
            "tag_index": {key: sorted(value) for key, value in self.tag_index.items()},
            "type_index": {key: sorted(value) for key, value in self.type_index.items()},
            "metadata": self.metadata,
            "historical_context": self.historical_context,
            "access_counter": self.access_counter,
            "max_priority": self.max_priority,
            "config": dict(self.memory_config),
        }

    def _restore_checkpoint(self, checkpoint: Mapping[str, Any], *, source: str) -> None:
        if not isinstance(checkpoint, Mapping):
            raise KnowledgePersistenceError("Reasoning memory checkpoint must be a mapping", context={"source": source})

        tree_data = checkpoint.get("tree_data")
        tree_structure = checkpoint.get("tree_structure")
        if tree_data is None or tree_structure is None:
            raise KnowledgePersistenceError("Checkpoint is missing SumTree arrays", context={"source": source})

        capacity = int(checkpoint.get("tree_capacity") or len(tree_data))
        restored_tree = SumTree(capacity)
        restored_tree.data = np.asarray(tree_data, dtype=object)
        restored_tree.tree = np.asarray(tree_structure, dtype=np.float64)
        restored_tree.write_ptr = int(checkpoint.get("tree_write_ptr", 0)) % restored_tree.capacity
        restored_tree.size = min(int(checkpoint.get("tree_size", 0)), restored_tree.capacity)
        restored_tree.max_priority = float(np.max(restored_tree.leaf_priorities())) if restored_tree.capacity else 1.0

        if restored_tree.data.shape[0] != restored_tree.capacity:
            raise KnowledgePersistenceError(
                "Checkpoint tree_data length does not match capacity",
                context={"source": source, "capacity": restored_tree.capacity, "data_len": restored_tree.data.shape[0]},
            )
        if restored_tree.tree.shape[0] != 2 * restored_tree.capacity - 1:
            raise KnowledgePersistenceError(
                "Checkpoint tree_structure length does not match capacity",
                context={"source": source, "capacity": restored_tree.capacity, "tree_len": restored_tree.tree.shape[0]},
            )

        self.memory_config.update(dict(checkpoint.get("config", {})))
        self._refresh_runtime_config()
        if self.max_size != restored_tree.capacity:
            self.max_size = restored_tree.capacity
            self.memory_config["max_size"] = restored_tree.capacity

        self.tree = restored_tree
        self.access_counter = int(checkpoint.get("access_counter", 0))
        self.max_priority = float(checkpoint.get("max_priority", self.default_priority))
        self.historical_context = {str(k): float(v) for k, v in dict(checkpoint.get("historical_context", {})).items()}
        self.metadata = self._restore_metadata(checkpoint)
        self._rebuild_indexes(checkpoint)

    def _restore_metadata(self, checkpoint: Mapping[str, Any]) -> Dict[int, Dict[str, Any]]:
        raw_metadata = checkpoint.get("metadata") or {}
        restored: Dict[int, Dict[str, Any]] = {}
        for key, value in dict(raw_metadata).items():
            try:
                idx = int(key)
            except (TypeError, ValueError):
                if isinstance(value, Mapping):
                    index_val = value.get("index")
                    if index_val is not None:
                        try:
                            idx = int(index_val)
                        except (TypeError, ValueError):
                            idx = -1
                    else:
                        idx = -1
                else:
                    idx = -1
            if idx < 0 or idx >= self.tree.capacity or self.tree.data[idx] is None:
                continue
            meta = dict(value) if isinstance(value, Mapping) else {}
            meta.setdefault("index", idx)
            meta.setdefault("tags", [])
            meta.setdefault("type", self._experience_type(self.tree.data[idx]))
            meta.setdefault("raw_priority", self.default_priority)
            meta.setdefault("tree_priority", self.tree.leaf_priority(idx))
            restored[idx] = meta

        for idx in self._occupied_indices():
            if idx not in restored and self.tree.data[idx] is not None:
                tags = self._normalize_tags(None, self.tree.data[idx])
                restored[idx] = {
                    "index": idx,
                    "type": self._experience_type(self.tree.data[idx]),
                    "tags": sorted(tags),
                    "raw_priority": self.default_priority,
                    "tree_priority": self.tree.leaf_priority(idx),
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    "created_at_ms": monotonic_timestamp_ms(),
                    "access_count": 0,
                }
        return restored

    def _rebuild_indexes(self, checkpoint: Optional[Mapping[str, Any]] = None) -> None:
        self.tag_index = defaultdict(set)
        self.type_index = defaultdict(set)
        for idx, meta in self.metadata.items():
            if self.tree.data[idx] is None:
                continue
            for tag in meta.get("tags", []):
                if str(tag).strip():
                    self.tag_index[str(tag).strip()].add(idx)
            exp_type = meta.get("type") or self._experience_type(self.tree.data[idx])
            if exp_type:
                self.type_index[str(exp_type)].add(idx)

        if checkpoint:
            for tag, indices in dict(checkpoint.get("tag_index", {})).items():
                for idx in indices:
                    if isinstance(idx, int) and 0 <= idx < self.tree.capacity and self.tree.data[idx] is not None:
                        self.tag_index[str(tag)].add(idx)
            for exp_type, indices in dict(checkpoint.get("type_index", {})).items():
                for idx in indices:
                    if isinstance(idx, int) and 0 <= idx < self.tree.capacity and self.tree.data[idx] is not None:
                        self.type_index[str(exp_type)].add(idx)

    def _cleanup_old_checkpoints(self) -> None:
        if self.checkpoint_keep_last <= 0 or not self.checkpoint_dir.exists():
            return
        pattern = f"{self.checkpoint_prefix}_*.pt"
        checkpoints = sorted(self.checkpoint_dir.glob(pattern), key=lambda item: item.stat().st_mtime, reverse=True)
        for stale in checkpoints[self.checkpoint_keep_last :]:
            try:
                stale.unlink()
            except OSError as exc:
                logger.warning(f"Failed to remove stale memory checkpoint {stale}: {exc}")

    # ------------------------------------------------------------------
    # Context / sampling helpers
    # ------------------------------------------------------------------
    def _importance_weight(self, probability: float, population_size: int) -> float:
        if probability <= 0.0 or population_size <= 0:
            return 0.0
        return float((population_size * probability) ** (-self.beta))

    def _decay_historical_context(self) -> None:
        decayed: Dict[str, float] = {}
        for tag, weight in self.historical_context.items():
            new_weight = float(weight) * self.context_decay_factor
            if new_weight >= self.historical_context_floor:
                decayed[tag] = new_weight
        self.historical_context = decayed


if __name__ == "__main__":
    print("\n=== Running Reasoning Memory ===\n")
    printer.status("TEST", "Reasoning Memory initialized", "info")

    with tempfile.TemporaryDirectory(prefix="reasoning_memory_test_") as tmp_dir:
        memory = ReasoningMemory()
        memory.memory_config["checkpoint_dir"] = tmp_dir
        memory.memory_config["checkpoint_keep_last"] = 2
        memory._refresh_runtime_config()
        memory.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        first_idx = memory.add(
            experience={"type": "inference", "query": "A", "result": 0.82},
            priority=0.9,
            tag="inference",
        )
        second_idx = memory.add(
            experience={"type": "validation_report", "status": "ok", "tag": "validation"},
            priority=0.75,
            tag="validation",
        )
        transition_idx = memory.add(
            experience=Transition("s0", "infer", 1.0, "s1", False),
            priority=0.6,
            tag=["transition", "training"],
        )

        assert memory.size() == 3
        assert memory.get(first_idx)["type"] == "inference"
        assert memory.set(second_idx, {"type": "validation_report", "status": "passed"}) is True
        assert len(memory.get_by_type("validation_report")) == 1
        assert len(memory.get_by_tag("inference")) == 1

        samples, indices, priorities = memory.sample_proportional(batch_size=2)
        assert len(samples) == len(indices) == len(priorities) == 2
        assert all(idx in {first_idx, second_idx, transition_idx} for idx in indices)

        updated = memory.update_priorities(indices, [0.95 for _ in indices])
        assert updated == len(indices)
        assert memory.metrics()["size"] == 3
        assert memory.get_current_context()

        checkpoint_path = memory.save_checkpoint("reasoning_memory_test.pt")
        restored = ReasoningMemory()
        restored.memory_config["checkpoint_dir"] = tmp_dir
        restored._refresh_runtime_config()
        assert restored.load_checkpoint(checkpoint_path) is True
        assert restored.size() == memory.size()
        assert restored.get_by_type("inference")

        restored.remove(first_idx)
        assert restored.size() == 2
        restored.clear()
        assert restored.size() == 0

    print("\n=== Test ran successfully ===\n")
