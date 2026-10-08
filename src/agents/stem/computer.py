"""Deterministic discrete/computational algorithms for SLAI STEM.

Repository analysis, AST manipulation, refactoring, and patch generation remain
owned by SoftwareAgent.
"""
from __future__ import annotations

__version__ = "2.3.0"

import heapq
import math

from collections import deque
from typing import Any, Callable, Dict, Hashable, Mapping, Optional, Sequence

from .utils.config_loader import get_config_section, load_global_config
from .utils.stem_errors import STEMDomainError
from .stem_memory import STEMMemory
from logs.logger import PrettyPrinter, get_logger # pyright: ignore[reportMissingImports]

logger = get_logger("SLAI Computing")
printer = PrettyPrinter()


class Computing:
    def __init__(self, config: Mapping[str, Any] | None = None, memory: Optional[STEMMemory] = None) -> None:
        self.config: Dict[str, Any] = load_global_config()
        self.computing_config = dict(get_config_section("stem_computing", config=self.config) or {})
        if config:
            self.computing_config.update(dict(config))
        self.memory = memory

    @staticmethod
    def gcd(a: int, b: int) -> int:
        if not isinstance(a, int) or not isinstance(b, int): raise STEMDomainError("gcd requires integers")
        return math.gcd(a, b)

    @staticmethod
    def lcm(a: int, b: int) -> int:
        if not isinstance(a, int) or not isinstance(b, int): raise STEMDomainError("lcm requires integers")
        return math.lcm(a, b)

    @staticmethod
    def factorial(n: int) -> int:
        if not isinstance(n, int) or n < 0: raise STEMDomainError("factorial requires non-negative integer")
        return math.factorial(n)

    @staticmethod
    def binomial(n: int, k: int) -> int:
        if not isinstance(n, int) or not isinstance(k, int) or n < 0 or k < 0 or k > n: raise STEMDomainError("invalid binomial arguments")
        return math.comb(n, k)

    @staticmethod
    def recurrence(initial: Sequence[float], step: Callable[[Sequence[float], int], float], terms: int) -> list[float]:
        if not callable(step) or terms < len(initial): raise STEMDomainError("invalid recurrence definition")
        values = [float(v) for v in initial]
        while len(values) < terms:
            result = float(step(tuple(values), len(values)))
            if not math.isfinite(result): raise STEMDomainError("recurrence produced non-finite value")
            values.append(result)
        return values

    @staticmethod
    def breadth_first_search(graph: Mapping[Hashable, Sequence[Hashable]], start: Hashable) -> list[Hashable]:
        if start not in graph: raise STEMDomainError("start node is absent from graph")
        queue = deque([start]); seen = {start}; order = []
        while queue:
            node = queue.popleft(); order.append(node)
            for neighbour in graph.get(node, ()):
                if neighbour not in seen: seen.add(neighbour); queue.append(neighbour)
        return order

    @staticmethod
    def dijkstra(graph: Mapping[Hashable, Mapping[Hashable, float]], start: Hashable) -> Mapping[Hashable, float]:
        if start not in graph: raise STEMDomainError("start node is absent from graph")
        dist: Dict[Hashable, float] = {node: math.inf for node in graph}; dist[start] = 0.0
        heap: list[tuple[float, int, Hashable]] = [(0.0, 0, start)]; serial = 1
        while heap:
            current, _, node = heapq.heappop(heap)
            if current != dist[node]: continue
            for neighbour, weight in graph.get(node, {}).items():
                w = float(weight)
                if w < 0 or not math.isfinite(w): raise STEMDomainError("Dijkstra requires finite non-negative weights")
                if neighbour not in dist: dist[neighbour] = math.inf
                candidate = current + w
                if candidate < dist[neighbour]:
                    dist[neighbour] = candidate; heapq.heappush(heap, (candidate, serial, neighbour)); serial += 1
        return dist

    @staticmethod
    def topological_sort(graph: Mapping[Hashable, Sequence[Hashable]]) -> list[Hashable]:
        indegree: Dict[Hashable, int] = {node: 0 for node in graph}
        for neighbours in graph.values():
            for node in neighbours: indegree[node] = indegree.get(node, 0) + 1
        queue = deque(sorted((node for node, degree in indegree.items() if degree == 0), key=repr)); order = []
        while queue:
            node = queue.popleft(); order.append(node)
            for neighbour in graph.get(node, ()):
                indegree[neighbour] -= 1
                if indegree[neighbour] == 0: queue.append(neighbour)
        if len(order) != len(indegree): raise STEMDomainError("Graph contains a directed cycle")
        return order


__all__ = ["Computing"]
