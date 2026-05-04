from __future__ import annotations

import numpy as np

from typing import Callable, List, Optional

from .utils.config_loader import get_config_section, load_global_config


class SegmentTree:
    """Generic segment tree with power-of-two internal storage."""

    def __init__(self, capacity: int, operation: Callable[[float, float], float], neutral_element: float):
        if capacity <= 0:
            raise ValueError("capacity must be > 0")

        # Keep tree shape stable and simple for index math.
        self.capacity = 1
        while self.capacity < capacity:
            self.capacity *= 2

        self.operation = operation
        self.neutral_element = neutral_element
        self.tree: List[float] = [neutral_element for _ in range(2 * self.capacity)]

    def __len__(self) -> int:
        return self.capacity

    def set(self, idx: int, value: float) -> None:
        if idx < 0 or idx >= self.capacity:
            raise IndexError(f"index {idx} out of bounds for capacity {self.capacity}")

        idx += self.capacity
        self.tree[idx] = float(value)

        idx //= 2
        while idx >= 1:
            self.tree[idx] = self.operation(self.tree[2 * idx], self.tree[2 * idx + 1])
            idx //= 2

    def get(self, idx: int) -> float:
        if idx < 0 or idx >= self.capacity:
            raise IndexError(f"index {idx} out of bounds for capacity {self.capacity}")
        return self.tree[idx + self.capacity]

    def reduce(self, start: int = 0, end: Optional[int] = None) -> float:
        """Returns operation applied over [start, end)."""
        if end is None:
            end = self.capacity
        if start < 0 or end < start or end > self.capacity:
            raise ValueError("invalid reduce range")

        start += self.capacity
        end += self.capacity

        left_result = self.neutral_element
        right_result = self.neutral_element

        while start < end:
            if start & 1:
                left_result = self.operation(left_result, self.tree[start])
                start += 1
            if end & 1:
                end -= 1
                right_result = self.operation(self.tree[end], right_result)
            start //= 2
            end //= 2

        return self.operation(left_result, right_result)


class SumSegmentTree(SegmentTree):
    def __init__(self, capacity: int):
        super().__init__(capacity=capacity, operation=lambda a, b: a + b, neutral_element=0.0)

    def sum(self, start: int = 0, end: Optional[int] = None) -> float:
        return self.reduce(start=start, end=end)

    def find_prefixsum_idx(self, prefixsum: float) -> int:
        total = self.sum(0, self.capacity)
        if prefixsum < 0:
            raise ValueError("prefixsum must be >= 0")
        if prefixsum > total + 1e-12:
            raise ValueError(f"prefixsum {prefixsum} exceeds total mass {total}")

        idx = 1
        while idx < self.capacity:
            left = 2 * idx
            if self.tree[left] >= prefixsum:
                idx = left
            else:
                prefixsum -= self.tree[left]
                idx = left + 1
        return idx - self.capacity


class MinSegmentTree(SegmentTree):
    def __init__(self, capacity: int):
        super().__init__(capacity=capacity, operation=min, neutral_element=float("inf"))

    def min(self, start: int = 0, end: Optional[int] = None) -> float:
        return self.reduce(start=start, end=end)


class SegmentTreeFactory:
    """Config-aware factory to keep config handling consistent with other buffer modules."""

    @staticmethod
    def from_config(user_config: Optional[dict] = None) -> dict:
        load_global_config()
        config = dict(get_config_section("segment_tree") or {})
        if user_config:
            config.update(user_config)

        requested_capacity = int(config.get("capacity", 1024))
        safe_capacity = max(1, requested_capacity)

        return {
            "sum_tree": SumSegmentTree(safe_capacity),
            "min_tree": MinSegmentTree(safe_capacity),
            "capacity": safe_capacity,
            "dtype": np.float32 if str(config.get("dtype", "float32")) == "float32" else np.float64,
        }


__all__ = [
    "SegmentTree",
    "SumSegmentTree",
    "MinSegmentTree",
    "SegmentTreeFactory",
]
