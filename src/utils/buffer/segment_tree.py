from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np  # type: ignore

from .utils.config_loader import get_config_section, load_global_config
from .utils.buffer_errors import *
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Segment Tree")
printer = PrettyPrinter()

Number = Union[int, float, np.number]
Operation = Callable[[float, float], float]


@dataclass(frozen=True)
class SegmentTreeConfig:
    """Configuration contract for segment-tree priority storage.

    `capacity` is the logical number of replay slots. Internally, the tree is
    padded to the next power of two for stable O(log N) updates and queries.
    """

    capacity: int = 1024
    dtype: str = "float32"
    epsilon: float = 1e-12
    allow_zero_mass: bool = False

    @classmethod
    def from_config(cls, user_config: Optional[Mapping[str, Any]] = None) -> "SegmentTreeConfig":
        load_global_config()
        config: Dict[str, Any] = dict(get_config_section("segment_tree") or {})
        if user_config:
            override = user_config.get("segment_tree", user_config) if isinstance(user_config, Mapping) else {}
            if isinstance(override, Mapping):
                config.update(dict(override))

        capacity = int(config.get("capacity", cls.capacity))
        dtype = str(config.get("dtype", cls.dtype)).strip().lower()
        epsilon = float(config.get("epsilon", cls.epsilon))
        allow_zero_mass = bool(config.get("allow_zero_mass", cls.allow_zero_mass))

        if capacity <= 0:
            raise SegmentTreeCapacityError(capacity, reason="segment_tree.capacity must be > 0")
        if dtype not in {"float32", "float64"}:
            raise ConfigValueError("segment_tree.dtype", dtype, "one of: float32, float64")
        if not np.isfinite(epsilon) or epsilon < 0:
            raise ConfigValueError("segment_tree.epsilon", epsilon, "finite float >= 0")

        return cls(capacity=capacity, dtype=dtype, epsilon=epsilon, allow_zero_mass=allow_zero_mass)

    @property
    def np_dtype(self) -> Any:
        return np.float64 if self.dtype == "float64" else np.float32


def _next_power_of_two(value: int) -> int:
    if value <= 0:
        raise SegmentTreeCapacityError(value, reason="capacity must be > 0")
    return 1 << (int(value) - 1).bit_length()


def _coerce_float(value: Any, *, field_name: str, operation: str, allow_infinite: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise SegmentTreeOperationError(operation, f"{field_name} must be numeric") from exc
    valid = np.isfinite(result) or (allow_infinite and np.isinf(result))
    if not valid:
        expectation = "finite or infinite" if allow_infinite else "finite"
        raise SegmentTreeOperationError(operation, f"{field_name} must be {expectation}")
    return result


class SegmentTree:
    """Generic segment tree with logical capacity and power-of-two storage.

    The tree supports O(log N) point updates and O(log N) range reductions over
    `[start, end)`. It is safe for replay priority storage because public range
    checks use the logical capacity, while the internal padded leaves remain
    invisible to callers.
    """

    def __init__(
        self,
        capacity: int,
        operation: Operation,
        neutral_element: Number,
        *,
        dtype: Any = np.float32,
        name: str = "segment_tree",
    ) -> None:
        if capacity <= 0:
            raise SegmentTreeCapacityError(capacity, reason="capacity must be > 0")
        if not callable(operation):
            raise SegmentTreeOperationError("initialize", "operation must be callable")

        self.capacity = int(capacity)  # logical capacity exposed to callers
        self.tree_capacity = _next_power_of_two(self.capacity)
        self.operation = operation
        self.neutral_element = _coerce_float(neutral_element, field_name="neutral_element", operation="initialize", allow_infinite=True)
        self.dtype = np.dtype(dtype)
        self.name = str(name)
        self._tree: np.ndarray = np.full(2 * self.tree_capacity, self.neutral_element, dtype=self.dtype)
        self._lock = RLock()

    def __len__(self) -> int:
        return self.capacity

    def __repr__(self) -> str:
        return f"{type(self).__name__}(capacity={self.capacity}, tree_capacity={self.tree_capacity}, dtype={self.dtype})"

    @property
    def tree(self) -> List[float]:
        """Backward-compatible list view of internal tree storage."""
        with self._lock:
            return [float(v) for v in self._tree.tolist()]

    def _check_index(self, idx: int, *, operation: str) -> int:
        index = int(idx)
        if index < 0 or index >= self.capacity:
            raise SegmentTreeIndexError(index=index, capacity=self.capacity, operation=operation)
        return index

    def _normalize_range(self, start: int = 0, end: Optional[int] = None) -> Tuple[int, int]:
        resolved_start = int(start)
        resolved_end = self.capacity if end is None else int(end)
        if resolved_start < 0 or resolved_end < resolved_start or resolved_end > self.capacity:
            raise SegmentTreeRangeError(resolved_start, resolved_end, self.capacity)
        return resolved_start, resolved_end

    def update(self, idx: int, priority: Number) -> None:
        """Set one leaf value and refresh ancestors in O(log N)."""
        index = self._check_index(idx, operation="update")
        value = _coerce_float(priority, field_name="priority", operation="update")
        with self._lock:
            tree_index = index + self.tree_capacity
            self._tree[tree_index] = value
            tree_index //= 2
            while tree_index >= 1:
                self._tree[tree_index] = self.operation(float(self._tree[2 * tree_index]), float(self._tree[2 * tree_index + 1]))
                tree_index //= 2

    def set(self, idx: int, value: Number) -> None:
        """Backward-compatible alias for update()."""
        self.update(idx, value)

    def batch_update(self, indices: Sequence[int], priorities: Sequence[Number]) -> None:
        """Apply several point updates. Kept simple and deterministic for safety."""
        if len(indices) != len(priorities):
            raise SegmentTreeOperationError("batch_update", "indices and priorities must have the same length")
        for idx, priority in zip(indices, priorities):
            self.update(int(idx), priority)

    def get(self, idx: int) -> float:
        index = self._check_index(idx, operation="get")
        with self._lock:
            return float(self._tree[index + self.tree_capacity])

    def reduce(self, start: int = 0, end: Optional[int] = None) -> float:
        """Return operation applied over `[start, end)` in O(log N)."""
        left, right = self._normalize_range(start, end)
        if left == right:
            return float(self.neutral_element)

        with self._lock:
            left += self.tree_capacity
            right += self.tree_capacity
            left_result = float(self.neutral_element)
            right_result = float(self.neutral_element)

            while left < right:
                if left & 1:
                    left_result = self.operation(left_result, float(self._tree[left]))
                    left += 1
                if right & 1:
                    right -= 1
                    right_result = self.operation(float(self._tree[right]), right_result)
                left //= 2
                right //= 2

            return float(self.operation(left_result, right_result))

    def reset(self, value: Optional[Number] = None) -> None:
        """Reset all leaves and ancestors to neutral or a provided finite value."""
        fill_value = self.neutral_element if value is None else _coerce_float(value, field_name="value", operation="reset")
        with self._lock:
            self._tree.fill(fill_value)
            # Padded leaves must stay neutral when resetting to a non-neutral value.
            if self.capacity < self.tree_capacity:
                self._tree[self.tree_capacity + self.capacity : 2 * self.tree_capacity] = self.neutral_element
            for idx in range(self.tree_capacity - 1, 0, -1):
                self._tree[idx] = self.operation(float(self._tree[2 * idx]), float(self._tree[2 * idx + 1]))

    def leaf_values(self, *, include_padding: bool = False) -> np.ndarray:
        """Return a copy of logical leaves, or all padded leaves when requested."""
        with self._lock:
            end = self.tree_capacity if include_padding else self.capacity
            return self._tree[self.tree_capacity : self.tree_capacity + end].copy()

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "name": self.name,
                "capacity": self.capacity,
                "tree_capacity": self.tree_capacity,
                "dtype": str(self.dtype),
                "neutral_element": float(self.neutral_element),
                "root": float(self._tree[1]),
                "leaves": self.leaf_values().astype(float).tolist(),
            }


class SumSegmentTree(SegmentTree):
    """Segment tree specialized for priority mass and prefix-sum sampling."""

    def __init__(self, capacity: int, *, dtype: Any = np.float32, epsilon: float = 1e-12, allow_zero_mass: bool = False):
        super().__init__(capacity=capacity, operation=lambda a, b: a + b, neutral_element=0.0, dtype=dtype, name="sum_tree")
        self.epsilon = float(epsilon)
        self.allow_zero_mass = bool(allow_zero_mass)

    def update(self, idx: int, priority: Number) -> None:
        value = _coerce_float(priority, field_name="priority", operation="update")
        if value < 0:
            raise SegmentTreeOperationError("update", "sum-tree priority must be >= 0")
        super().update(idx, value)

    def sum(self, start: int = 0, end: Optional[int] = None) -> float:
        return self.reduce(start=start, end=end)

    @property
    def total_mass(self) -> float:
        return self.sum(0, self.capacity)

    def prefix_sum_index(self, mass: Number) -> int:
        """Return smallest index whose cumulative priority exceeds `mass`.

        This is the O(log N) lookup used by prioritized replay sampling. Callers
        should pass `mass` sampled from `[0, total_mass)`. Exact `total_mass` is
        tolerated and mapped to the final non-empty mass bucket.
        """
        target = _coerce_float(mass, field_name="mass", operation="prefix_sum_index")
        total = self.total_mass
        if total < 0 or not np.isfinite(total):
            raise SegmentTreeMassError(total, reason="total priority mass must be finite and non-negative")
        if total <= self.epsilon and not self.allow_zero_mass:
            raise SegmentTreeMassError(total, reason="cannot sample from zero priority mass")
        if target < -self.epsilon or target > total + self.epsilon:
            raise SegmentTreePrefixSumError(target, total, reason="mass must be within [0, total_mass]")
        if total <= self.epsilon:
            return 0

        # Numerical guardrails: prefix traversal expects a target in [0, total).
        target = max(0.0, min(target, np.nextafter(float(total), float("-inf"))))

        with self._lock:
            idx = 1
            while idx < self.tree_capacity:
                left = 2 * idx
                left_mass = float(self._tree[left])
                if target <= left_mass:
                    idx = left
                else:
                    target -= left_mass
                    idx = left + 1
            result = idx - self.tree_capacity

        if result >= self.capacity:
            # Should not happen when padded leaves are neutral; keep explicit for diagnostics.
            raise SegmentTreePrefixSumError(mass, total, reason="resolved index landed in padded tree capacity")
        return int(result)

    def find_prefixsum_idx(self, prefixsum: Number) -> int:
        """Backward-compatible alias for prefix_sum_index()."""
        return self.prefix_sum_index(prefixsum)

    def sample_index(self, rng: Optional[np.random.Generator] = None) -> int:
        generator = rng or np.random.default_rng()
        total = self.total_mass
        if total <= self.epsilon:
            raise SegmentTreeMassError(total, reason="cannot sample from zero priority mass")
        return self.prefix_sum_index(float(generator.random() * total))


class MinSegmentTree(SegmentTree):
    """Segment tree specialized for minimum priority/probability lookup.

    The minimum aggregate is used by prioritized replay to normalize importance
    sampling weights: max_weight = (p_min * N) ** (-beta).
    """

    def __init__(self, capacity: int, *, dtype: Any = np.float32):
        super().__init__(capacity=capacity, operation=min, neutral_element=float("inf"), dtype=dtype, name="min_tree")

    def update(self, idx: int, priority: Number) -> None:
        value = _coerce_float(priority, field_name="priority", operation="update")
        if value < 0:
            raise SegmentTreeOperationError("update", "min-tree priority must be >= 0")
        super().update(idx, value)

    def min(self, start: int = 0, end: Optional[int] = None) -> float:
        value = self.reduce(start=start, end=end)
        return float(value)

    @property
    def minimum(self) -> float:
        return self.min(0, self.capacity)


@dataclass
class PriorityTreeBundle:
    """Convenience bundle keeping sum/min trees synchronized for PER."""

    sum_tree: SumSegmentTree
    min_tree: MinSegmentTree

    @property
    def capacity(self) -> int:
        return self.sum_tree.capacity

    def update(self, idx: int, priority: Number) -> None:
        self.sum_tree.update(idx, priority)
        self.min_tree.update(idx, priority)

    def batch_update(self, indices: Sequence[int], priorities: Sequence[Number]) -> None:
        if len(indices) != len(priorities):
            raise SegmentTreeOperationError("batch_update", "indices and priorities must have the same length")
        for idx, priority in zip(indices, priorities):
            self.update(int(idx), priority)

    def prefix_sum_index(self, mass: Number) -> int:
        return self.sum_tree.prefix_sum_index(mass)

    def min(self) -> float:
        return self.min_tree.min()

    def total(self) -> float:
        return self.sum_tree.total_mass


class SegmentTreeFactory:
    """Config-aware factory; keeps config handling aligned with buffer modules."""

    @staticmethod
    def from_config(user_config: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        config = SegmentTreeConfig.from_config(user_config=user_config)
        sum_tree = SumSegmentTree(
            config.capacity,
            dtype=config.np_dtype,
            epsilon=config.epsilon,
            allow_zero_mass=config.allow_zero_mass,
        )
        min_tree = MinSegmentTree(config.capacity, dtype=config.np_dtype)
        bundle = PriorityTreeBundle(sum_tree=sum_tree, min_tree=min_tree)
        return {
            "sum_tree": sum_tree,
            "min_tree": min_tree,
            "priority_trees": bundle,
            "capacity": config.capacity,
            "tree_capacity": sum_tree.tree_capacity,
            "dtype": config.np_dtype,
            "config": config,
        }

    @staticmethod
    def priority_trees(user_config: Optional[Mapping[str, Any]] = None) -> PriorityTreeBundle:
        return SegmentTreeFactory.from_config(user_config=user_config)["priority_trees"]


if __name__ == "__main__":
    print("\n=== Running  Segment Tree ===\n")
    printer.status("TEST", " Segment Tree initialized", "info")

    s = SumSegmentTree(5)
    m = MinSegmentTree(5)
    vals = [0.1, 0.2, 0.3, 0.4, 1.0]
    for i, v in enumerate(vals):
        s.update(i, v)
        m.update(i, v)

    assert len(s) == 5 and s.tree_capacity == 8
    assert abs(s.sum() - 2.0) < 1e-6
    assert abs(m.min() - 0.1) < 1e-6
    assert s.prefix_sum_index(0.0) == 0
    assert s.prefix_sum_index(0.31) == 2
    assert s.prefix_sum_index(s.total_mass) == 4

    bundle = PriorityTreeBundle(SumSegmentTree(4), MinSegmentTree(4))
    bundle.batch_update([0, 1, 2, 3], [1.0, 0.5, 0.25, 0.25])
    assert bundle.prefix_sum_index(1.25) == 1
    assert abs(bundle.min() - 0.25) < 1e-6

    try:
        s.update(99, 1.0)
        raise AssertionError("expected SegmentTreeIndexError")
    except SegmentTreeIndexError:
        pass
    try:
        s.prefix_sum_index(9.0)
        raise AssertionError("expected SegmentTreePrefixSumError")
    except SegmentTreePrefixSumError:
        pass

    printer.status("TEST", " Segment Tree checks passed", "success")
    print("\n=== Test ran successfully ===\n")
