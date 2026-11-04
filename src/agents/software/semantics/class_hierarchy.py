"""Class hierarchy analysis: bases, C3 linearization, overrides, depth, cycles."""

from __future__ import annotations

__version__ = "2.4.0"

from collections.abc import Iterable
from typing import Optional

from ..graphs.base_graph import BaseGraph, GraphEdge
from ..software_types import ClassNode, EdgeKind, HierarchyReport, SymbolKind
from ..utils.software_helpers import is_within_scope
from .symbol_index import SymbolIndex
from logs.logger import get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Software Class Hierarchy")


class ClassHierarchy:
    """Hierarchy facts derived from a :class:`SymbolIndex`."""

    def __init__(self, index: SymbolIndex) -> None:
        self.index = index
        self._graph: Optional[BaseGraph] = None

    def classes(self) -> tuple[str, ...]:
        return tuple(sorted(q for _, s in self.index.symbols() for q in [s.qualified_name] if s.kind is SymbolKind.CLASS))

    def graph(self) -> BaseGraph:
        """Inheritance graph: edge ``subclass -> base`` (internal bases only)."""

        if self._graph is None:
            names = self.classes()
            edges = [
                GraphEdge(name, base, EdgeKind.INHERITS)
                for name in names
                for base in self.index.class_bases(name).internal
            ]
            self._graph = BaseGraph(names, edges, name="inheritance")
        return self._graph

    def node(self, class_name: str) -> ClassNode:
        index = self.index
        symbol = index.require(class_name)
        bases = index.class_bases(class_name)
        order, conflict = index.linearization(class_name)
        own_methods = tuple(sorted(m.name for m in index.members(class_name) if m.kind is SymbolKind.METHOD))
        overrides = tuple(name for name in own_methods if index.overridden(f"{class_name}.{name}"))
        graph = self.graph()
        depth = max((len(order) - 1, 0))
        distances = graph.reachable([class_name])
        depth = max(distances.values(), default=0)
        return ClassNode(
            qualified_name=class_name,
            path=index.path_of(class_name) or "",
            bases=(*bases.internal, *bases.external),
            unresolved_bases=bases.unresolved,
            linearization=order,
            subclasses=index.subclasses(class_name),
            methods=own_methods,
            overrides=overrides,
            is_abstract=symbol.is_abstract,
            depth=depth,
            mro_conflict=conflict,
        )

    def analyze(self, *, scope: Optional[Iterable[str]] = None) -> HierarchyReport:
        scopes = tuple(scope or ())
        names = [name for name in self.classes() if not scopes or any(is_within_scope(name, item) for item in scopes)]
        nodes = tuple(self.node(name) for name in names)
        external: dict[str, int] = {}
        for node in nodes:
            internal = set(self.index.class_bases(node.qualified_name).internal)
            for base in node.bases:
                if base not in internal:
                    external[base] = external.get(base, 0) + 1
        roots = tuple(node.qualified_name for node in nodes if not self.index.class_bases(node.qualified_name).internal)
        cycles = tuple(self.graph().cycles(include_self_loops=True))
        return HierarchyReport(
            classes=nodes,
            roots=roots,
            cycles=cycles,
            external_bases=dict(sorted(external.items(), key=lambda item: (-item[1], item[0]))),
            max_depth=max((node.depth for node in nodes), default=0),
        )


__all__ = ["ClassHierarchy"]
