"""Immutable directed multigraph with the algorithms the Software subsystem needs.

Nodes are strings (module names, qualified symbol names).  Parallel edges of
different :class:`EdgeKind` between the same pair are kept, but adjacency is by
node pair so every traversal is linear in the number of distinct pairs.  All
algorithms are iterative: repository graphs routinely exceed Python's
recursion limit.

References
----------
Tarjan (1972), "Depth-first search and linear graph algorithms" (strongly
connected components); Kahn (1962) / condensation layering for ``layers``.
"""

from __future__ import annotations

__version__ = "2.4.0"

from collections import deque
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, Optional

from ..models.base_model import BaseSoftwareModel
from ..software_types import EdgeKind, Serializable
from ..utils.software_errors import SoftwareGraphError
from ..utils.software_helpers import content_digest, stable_fingerprint
from logs.logger import get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Software Base Graph")

Direction = Literal["out", "in"]
_MAX_LINES = 5


@dataclass(frozen=True, slots=True)
class GraphEdge(Serializable):
    source: str
    target: str
    kind: EdgeKind
    weight: int = 1
    tags: tuple[str, ...] = ()
    lines: tuple[int, ...] = ()
    exact: bool = True


def merge_edges(edges: Iterable[GraphEdge]) -> list[GraphEdge]:
    """Collapse parallel edges of the same ``(source, target, kind)``."""

    merged: dict[tuple[str, str, EdgeKind], GraphEdge] = {}
    for edge in edges:
        key = (edge.source, edge.target, edge.kind)
        current = merged.get(key)
        if current is None:
            merged[key] = GraphEdge(edge.source, edge.target, edge.kind, edge.weight, tuple(sorted(set(edge.tags))), edge.lines[:_MAX_LINES], edge.exact)
        else:
            merged[key] = GraphEdge(
                edge.source,
                edge.target,
                edge.kind,
                current.weight + edge.weight,
                tuple(sorted(set(current.tags) | set(edge.tags))),
                tuple(sorted(set(current.lines) | set(edge.lines)))[:_MAX_LINES],
                current.exact and edge.exact,
            )
    return list(merged.values())


class BaseGraph(BaseSoftwareModel):
    """Generic immutable digraph; subclasses add construction and domain queries."""

    model_type = "graph"
    graph_type: ClassVar[str] = "graph"

    def __init__(self, nodes: Iterable[str], edges: Iterable[GraphEdge], *, name: str = "") -> None:
        node_set = set(nodes)
        merged = merge_edges(edges)
        for edge in merged:
            node_set.add(edge.source)
            node_set.add(edge.target)
        self._nodes: tuple[str, ...] = tuple(sorted(node_set))
        self._edges: tuple[GraphEdge, ...] = tuple(sorted(merged, key=lambda e: (e.source, e.target, e.kind.value)))
        self.name = name

        by_pair: dict[tuple[str, str], list[GraphEdge]] = {}
        out: dict[str, dict[str, None]] = {node: {} for node in self._nodes}
        inn: dict[str, dict[str, None]] = {node: {} for node in self._nodes}
        for edge in self._edges:
            by_pair.setdefault((edge.source, edge.target), []).append(edge)
            out[edge.source][edge.target] = None
            inn[edge.target][edge.source] = None
        self._by_pair = {pair: tuple(items) for pair, items in by_pair.items()}
        self._out = {node: tuple(targets) for node, targets in out.items()}
        self._in = {node: tuple(sources) for node, sources in inn.items()}
        self._components: Optional[tuple[tuple[str, ...], ...]] = None
        self._digest: Optional[str] = None

    # -- identity -------------------------------------------------------------
    @property
    def digest(self) -> str:
        if self._digest is None:
            self._digest = content_digest(
                stable_fingerprint([self.graph_type, self._nodes, [(e.source, e.target, e.kind.value, e.weight) for e in self._edges]], length=32)
            )
        return self._digest

    @property
    def model_id(self) -> str:
        return f"{self.graph_type}:{self.name or self.digest[:12]}"

    # -- structure --------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, node: object) -> bool:
        return node in self._out

    @property
    def nodes(self) -> tuple[str, ...]:
        return self._nodes

    @property
    def edges(self) -> tuple[GraphEdge, ...]:
        return self._edges

    @property
    def edge_count(self) -> int:
        return len(self._edges)

    def has_node(self, node: str) -> bool:
        return node in self._out

    def has_edge(self, source: str, target: str, kind: Optional[EdgeKind] = None) -> bool:
        items = self._by_pair.get((source, target), ())
        return bool(items) if kind is None else any(item.kind is kind for item in items)

    def edges_between(self, source: str, target: str) -> tuple[GraphEdge, ...]:
        return self._by_pair.get((source, target), ())

    def _require(self, node: str) -> None:
        if node not in self._out:
            raise SoftwareGraphError("node is not part of the graph", context={"node": node, "graph": self.graph_type})

    def successors(self, node: str) -> tuple[str, ...]:
        self._require(node)
        return self._out[node]

    def predecessors(self, node: str) -> tuple[str, ...]:
        self._require(node)
        return self._in[node]

    def neighbors(self, node: str, direction: Direction = "out") -> tuple[str, ...]:
        return self.successors(node) if direction == "out" else self.predecessors(node)

    def out_degree(self, node: str) -> int:
        return len(self.successors(node))

    def in_degree(self, node: str) -> int:
        return len(self.predecessors(node))

    def out_edges(self, node: str) -> Iterator[GraphEdge]:
        for target in self.successors(node):
            yield from self._by_pair[(node, target)]

    def in_edges(self, node: str) -> Iterator[GraphEdge]:
        for source in self.predecessors(node):
            yield from self._by_pair[(source, node)]

    # -- traversal -----------------------------------------------------------------
    def reachable(
        self,
        sources: Iterable[str],
        *,
        direction: Direction = "out",
        max_depth: Optional[int] = None,
        edge_filter: Optional[Callable[[GraphEdge], bool]] = None,
    ) -> dict[str, int]:
        """Breadth-first distances from ``sources`` (sources have distance 0)."""

        adjacency = self._out if direction == "out" else self._in
        distance: dict[str, int] = {}
        queue: deque[str] = deque()
        for node in sources:
            if node in adjacency and node not in distance:
                distance[node] = 0
                queue.append(node)
        while queue:
            current = queue.popleft()
            depth = distance[current]
            if max_depth is not None and depth >= max_depth:
                continue
            for neighbour in adjacency[current]:
                if neighbour in distance:
                    continue
                if edge_filter is not None:
                    pair = (current, neighbour) if direction == "out" else (neighbour, current)
                    if not any(edge_filter(item) for item in self._by_pair[pair]):
                        continue
                distance[neighbour] = depth + 1
                queue.append(neighbour)
        return distance

    def shortest_path(self, source: str, target: str, *, direction: Direction = "out") -> Optional[tuple[str, ...]]:
        self._require(source)
        self._require(target)
        if source == target:
            return (source,)
        adjacency = self._out if direction == "out" else self._in
        parent: dict[str, str] = {}
        queue: deque[str] = deque([source])
        seen = {source}
        while queue:
            current = queue.popleft()
            for neighbour in adjacency[current]:
                if neighbour in seen:
                    continue
                seen.add(neighbour)
                parent[neighbour] = current
                if neighbour == target:
                    path = [target]
                    while path[-1] != source:
                        path.append(parent[path[-1]])
                    return tuple(reversed(path))
                queue.append(neighbour)
        return None

    # -- components, cycles, layers ----------------------------------------------------
    def components(self) -> tuple[tuple[str, ...], ...]:
        """Strongly connected components, dependencies (sinks) first.

        Iterative Tarjan: ``O(V + E)``, no recursion.
        """

        if self._components is not None:
            return self._components
        index: dict[str, int] = {}
        low: dict[str, int] = {}
        on_stack: set[str] = set()
        stack: list[str] = []
        result: list[tuple[str, ...]] = []
        counter = 0
        for root in self._nodes:
            if root in index:
                continue
            work: list[tuple[str, int]] = [(root, 0)]
            index[root] = low[root] = counter
            counter += 1
            stack.append(root)
            on_stack.add(root)
            while work:
                node, position = work[-1]
                successors = self._out[node]
                if position < len(successors):
                    work[-1] = (node, position + 1)
                    nxt = successors[position]
                    if nxt not in index:
                        index[nxt] = low[nxt] = counter
                        counter += 1
                        stack.append(nxt)
                        on_stack.add(nxt)
                        work.append((nxt, 0))
                    elif nxt in on_stack:
                        low[node] = min(low[node], index[nxt])
                else:
                    work.pop()
                    if work:
                        parent = work[-1][0]
                        low[parent] = min(low[parent], low[node])
                    if low[node] == index[node]:
                        component: list[str] = []
                        while True:
                            member = stack.pop()
                            on_stack.discard(member)
                            component.append(member)
                            if member == node:
                                break
                        result.append(tuple(sorted(component)))
        self._components = tuple(result)
        return self._components

    def cycles(self, *, include_self_loops: bool = False) -> tuple[tuple[str, ...], ...]:
        """Cyclic components (size >= 2, plus self-loops on request), largest first."""

        found = [
            component
            for component in self.components()
            if len(component) >= 2 or (include_self_loops and component[0] in self._out[component[0]])
        ]
        return tuple(sorted(found, key=lambda c: (-len(c), c)))

    def cycle_witness(self, component: Iterable[str]) -> tuple[str, ...]:
        """One concrete cycle (closed walk) inside a strongly connected component."""

        members = set(component)
        if not members:
            return ()
        start = min(members)
        # BFS back to ``start`` through members only.
        parent: dict[str, str] = {}
        queue: deque[str] = deque()
        for neighbour in self._out[start]:
            if neighbour in members:
                if neighbour == start:
                    return (start, start)
                parent[neighbour] = start
                queue.append(neighbour)
        while queue:
            current = queue.popleft()
            for neighbour in self._out[current]:
                if neighbour not in members:
                    continue
                if neighbour == start:
                    path = [current]
                    while path[-1] != start:
                        path.append(parent[path[-1]])
                    path.reverse()
                    return (*path, start)
                if neighbour not in parent:
                    parent[neighbour] = current
                    queue.append(neighbour)
        return ()

    def layers(self) -> tuple[tuple[str, ...], ...]:
        """Longest-path layering of the condensation; layer 0 depends on nothing."""

        component_of: dict[str, int] = {}
        components = self.components()
        for number, component in enumerate(components):
            for member in component:
                component_of[member] = number
        level = [0] * len(components)
        for number, component in enumerate(components):  # dependencies were emitted first
            best = -1
            for member in component:
                for target in self._out[member]:
                    other = component_of[target]
                    if other != number:
                        best = max(best, level[other])
            level[number] = best + 1
        grouped: dict[int, list[str]] = {}
        for number, component in enumerate(components):
            grouped.setdefault(level[number], []).extend(component)
        return tuple(tuple(sorted(grouped[key])) for key in sorted(grouped))

    def topological_order(self) -> Optional[tuple[str, ...]]:
        """Dependencies-first order, or ``None`` when the graph is cyclic."""

        if any(len(component) > 1 for component in self.components()):
            return None
        if any(node in self._out[node] for node in self._nodes):
            return None
        return tuple(component[0] for component in self.components())

    # -- derived graphs -----------------------------------------------------------------
    def subgraph(self, nodes: Iterable[str]) -> "BaseGraph":
        keep = {node for node in nodes if node in self._out}
        return BaseGraph(keep, (edge for edge in self._edges if edge.source in keep and edge.target in keep), name=f"{self.name}/sub")

    def filtered(self, predicate: Callable[[GraphEdge], bool], *, name: str = "") -> "BaseGraph":
        return BaseGraph(self._nodes, (edge for edge in self._edges if predicate(edge)), name=name or f"{self.name}/filtered")

    def reversed(self) -> "BaseGraph":
        return BaseGraph(
            self._nodes,
            (GraphEdge(e.target, e.source, e.kind, e.weight, e.tags, e.lines, e.exact) for e in self._edges),
            name=f"{self.name}/reversed",
        )

    # -- BaseSoftwareModel -----------------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        cyclic = self.cycles()
        n = len(self._nodes)
        pairs = len(self._by_pair)
        return {
            "graph": self.graph_type,
            "nodes": n,
            "edges": len(self._edges),
            "node_pairs": pairs,
            "density": round(pairs / (n * (n - 1)), 6) if n > 1 else 0.0,
            "cyclic_components": len(cyclic),
            "largest_cycle": len(cyclic[0]) if cyclic else 0,
            "digest": self.digest,
        }

    def to_dict(self) -> dict[str, Any]:
        return {"summary": self.summary(), "nodes": list(self._nodes), "edges": [edge.to_dict() for edge in self._edges]}


__all__ = ["BaseGraph", "Direction", "GraphEdge", "merge_edges"]
