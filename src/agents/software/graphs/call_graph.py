"""Static call graph over repository symbols.

Nodes are functions, methods and classes (plus module nodes for module-level
code).  Edges:

* ``CALLS`` – a call site whose callee resolves to a repository symbol.  A call
  to a class also targets its ``__init__``; a call through a method also
  targets overriding methods in subclasses (tag ``override``, ``exact=False``)
  because the receiver's dynamic type is unknown.
* ``REFERENCES`` – a symbol used as a value (callback, decorator, registry
  entry) rather than called; included so reachability stays conservative.

Calls that cannot be bound by scope may fall back to *name matching* against
uniquely named methods (tag ``name_match``, ``exact=False``).  Calls into
external code are recorded per caller in :attr:`CallGraph.external_calls`.
"""

from __future__ import annotations

__version__ = "2.4.0"

from collections.abc import Mapping
from typing import Any, Optional

from .base_graph import BaseGraph, GraphEdge
from ..semantics.symbol_index import SymbolIndex
from ..software_types import SUPER_ROOT, EdgeKind, Resolution, SymbolKind
from logs.logger import get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Software Call Graph")

_CALLABLE_KINDS = (SymbolKind.FUNCTION, SymbolKind.METHOD, SymbolKind.CLASS)


class CallGraph(BaseGraph):
    graph_type = "call"

    def __init__(
        self,
        nodes: Any,
        edges: Any,
        *,
        name: str = "",
        external_calls: Optional[Mapping[str, tuple[str, ...]]] = None,
        unresolved_calls: Optional[Mapping[str, int]] = None,
    ) -> None:
        super().__init__(nodes, edges, name=name)
        self.external_calls: dict[str, tuple[str, ...]] = dict(external_calls or {})
        self.unresolved_calls: dict[str, int] = dict(unresolved_calls or {})

    @classmethod
    def build(cls, index: SymbolIndex, *, include_heuristic: bool = True, include_references: bool = True, name: str = "") -> "CallGraph":
        repo = index.repo
        nodes = {qname for _, symbol in index.symbols() for qname in [symbol.qualified_name] if symbol.kind in _CALLABLE_KINDS}
        edges: list[GraphEdge] = []

        def add(source: str, target: str, kind: EdgeKind, line: int, tags: tuple[str, ...] = (), exact: bool = True) -> None:
            edges.append(GraphEdge(source, target, kind, 1, tags, (line,), exact))

        # ``super().m()`` is statically bound, so it must not fan out to overriding methods.
        static_calls = {
            (model.path, call.callee_span.end_line, call.callee_span.end_col)
            for model in repo.parsed()
            for call in model.calls
            if call.callee and call.callee[0] == SUPER_ROOT
        }
        owners: dict[tuple[str, str], str] = {}

        def owner(path: str, scope: str) -> str:
            key = (path, scope)
            found = owners.get(key)
            if found is None:
                model = repo.model(path)
                found = model.owner_symbol(scope)
                owners[key] = found
            return found

        for target, refs in index.all_references(include_heuristic=include_heuristic).items():
            kind = index.kind_of(target)
            if kind not in _CALLABLE_KINDS:
                continue
            for ref in refs:
                if ref.kind not in ("call", "name", "attribute") or ref.context != "load":
                    continue
                source = owner(ref.path, ref.scope)
                line = ref.span.start_line
                tags = () if ref.exact else ("name_match",)
                if ref.kind == "call":
                    add(source, target, EdgeKind.CALLS, line, tags, ref.exact)
                    if kind is SymbolKind.CLASS:
                        init = index.class_member(target, "__init__")
                        if init.internal and init.name:
                            add(source, init.name, EdgeKind.CALLS, line, (*tags, "constructor"), ref.exact)
                    elif kind is SymbolKind.METHOD and ref.exact and (ref.path, ref.span.end_line, ref.span.end_col) not in static_calls:
                        for override in index.overriders(target):
                            add(source, override, EdgeKind.CALLS, line, ("override",), False)
                elif include_references and ref.exact:
                    add(source, target, EdgeKind.REFERENCES, line)

        external: dict[str, set[str]] = {}
        unresolved: dict[str, int] = {}
        for model in repo.parsed():
            for call in model.calls:
                if not call.callee:
                    continue
                source = owner(model.path, call.scope)
                target = index.resolve_chain(model, call.root_scope, call.callee) if call.root_scope else None
                if target is not None and target.kind == "external" and target.name:
                    external.setdefault(source, set()).add(target.name)
                elif call.root_resolution is Resolution.BUILTIN and len(call.callee) == 1:
                    external.setdefault(source, set()).add(f"builtins.{call.callee[0]}")
                elif target is None or not target.resolved:
                    if call.root_resolution is not Resolution.BUILTIN:
                        unresolved[source] = unresolved.get(source, 0) + 1

        return cls(
            nodes,
            edges,
            name=name,
            external_calls={key: tuple(sorted(value)) for key, value in external.items()},
            unresolved_calls=unresolved,
        )

    # -- queries ---------------------------------------------------------------------
    def callees(self, symbol: str, *, transitive: bool = False, exact_only: bool = False) -> tuple[str, ...]:
        graph = self.exact() if exact_only else self
        if not transitive:
            return graph.successors(symbol)
        return tuple(sorted(node for node in graph.reachable([symbol]) if node != symbol))

    def callers(self, symbol: str, *, transitive: bool = False, exact_only: bool = False) -> tuple[str, ...]:
        graph = self.exact() if exact_only else self
        if not transitive:
            return graph.predecessors(symbol)
        return tuple(sorted(node for node in graph.reachable([symbol], direction="in") if node != symbol))

    def exact(self) -> BaseGraph:
        """Subgraph restricted to edges that were bound exactly by scope."""

        return self.filtered(lambda edge: edge.exact, name=f"{self.name}/exact")

    def calls_only(self) -> BaseGraph:
        return self.filtered(lambda edge: edge.kind is EdgeKind.CALLS, name=f"{self.name}/calls")

    def recursive_groups(self) -> tuple[tuple[str, ...], ...]:
        """Mutually/self recursive symbols (call-edge cycles, including direct recursion)."""

        return self.calls_only().cycles(include_self_loops=True)

    def external_callers(self, dotted_prefix: str) -> tuple[str, ...]:
        return tuple(sorted(source for source, names in self.external_calls.items() if any(n == dotted_prefix or n.startswith(dotted_prefix + ".") for n in names)))

    def summary(self) -> dict[str, Any]:
        base = super().summary()
        kinds: dict[str, int] = {}
        for edge in self.edges:
            kinds[edge.kind.value] = kinds.get(edge.kind.value, 0) + 1
        base.update(
            {
                "edge_kinds": kinds,
                "inexact_edges": sum(1 for edge in self.edges if not edge.exact),
                "recursive_groups": len(self.recursive_groups()),
                "callers_with_external_calls": len(self.external_calls),
                "unresolved_call_sites": sum(self.unresolved_calls.values()),
            }
        )
        return base


__all__ = ["CallGraph"]
