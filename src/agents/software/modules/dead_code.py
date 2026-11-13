"""Dead-code detection by reachability from declared entry points.

A symbol is *live* when it is reachable in the call/reference graph from a
root.  Roots are (1) explicitly configured entry points, (2) test code and
``__main__`` scripts together with everything they import, (3) the public API
of the repository (``public_api_is_live``), (4) dunder methods and decorated
symbols (frameworks register those by side effect), and (5) anything whose name
appears as a string literal (``getattr``/plugin tables).  Everything else is
reported with a confidence that is lowered by each dynamic-behaviour caveat.
"""

from __future__ import annotations

__version__ = "2.4.0"

import fnmatch

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Optional

from ..graphs.call_graph import CallGraph
from ..graphs.dependency_graph import DependencyGraph
from ..semantics.interface_compatibility import public_api
from ..semantics.symbol_index import SymbolIndex
from ..software_types import DeadCodeItem, DeadCodeReport, SymbolKind, SymbolRecord
from ..utils.software_helpers import opt_bool, opt_int, opt_strings, truncate_items
from logs.logger import get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Software Dead Code")

_NEUTRAL_DECORATORS = frozenset({"staticmethod", "classmethod", "abstractmethod", "cache", "lru_cache", "cached_property", "overload", "final", "wraps", "dataclass", "total_ordering"})
_NEUTRAL_BASES = frozenset({"object", "ABC", "abc.ABC", "Protocol", "typing.Protocol", "Generic", "typing.Generic", "Exception", "BaseException"})


@dataclass(frozen=True, slots=True)
class DeadCodeOptions:
    public_api_is_live: bool = True
    entry_points: tuple[str, ...] = ()
    min_confidence: float = 0.5
    include_tests_as_roots: bool = True
    include_private_variables: bool = True
    max_items: int = 500

    @classmethod
    def from_config(cls, section: Optional[Mapping[str, Any]] = None) -> "DeadCodeOptions":
        data = section or {}
        return cls(
            public_api_is_live=opt_bool(data, "public_api_is_live", True),
            entry_points=tuple(opt_strings(data, "entry_points", ())),
            min_confidence=float(data.get("min_confidence", 0.5)),
            include_tests_as_roots=opt_bool(data, "include_tests_as_roots", True),
            include_private_variables=opt_bool(data, "include_private_variables", True),
            max_items=opt_int(data, "max_items", 500, minimum=1),
        )


class DeadCodeAnalyzer:
    def __init__(self, index: SymbolIndex, calls: CallGraph, dependencies: DependencyGraph, options: Optional[DeadCodeOptions] = None) -> None:
        self.index = index
        self.calls = calls
        self.dependencies = dependencies
        self.options = options or DeadCodeOptions()

    # -- roots ------------------------------------------------------------------------
    def _match(self, patterns: Iterable[str], names: Iterable[str]) -> set[str]:
        found: set[str] = set()
        pool = list(names)
        for pattern in patterns:
            if any(ch in pattern for ch in "*?["):
                found.update(n for n in pool if fnmatch.fnmatchcase(n, pattern))
            else:
                found.update(n for n in pool if n == pattern or n.startswith(pattern + "."))
        return found

    def _root_modules(self) -> set[str]:
        repo = self.index.repo
        roots: set[str] = set()
        for model in repo.parsed():
            name = repo.module_name_of(model.path)
            if name is None:
                continue
            if model.source.is_test and self.options.include_tests_as_roots:
                roots.add(name)
            if "__main__" in model.string_names and any(use.name == "__name__" for use in model.names):
                roots.add(name)
            if model.path.endswith("__main__.py"):
                roots.add(name)
        explicit = self._match(self.options.entry_points, repo.module_names())
        roots |= explicit
        return roots

    def _roots(self) -> tuple[set[str], set[str], list[str]]:
        """Return ``(live symbol roots, live modules, entry point labels)``."""

        index, repo = self.index, self.index.repo
        symbol_names = [q for _, s in index.symbols() for q in [s.qualified_name]]
        labels: list[str] = []
        roots: set[str] = set(self._match(self.options.entry_points, symbol_names))
        labels.extend(sorted(self._match(self.options.entry_points, [*symbol_names, *repo.module_names()])))
        modules = self._root_modules()
        if self.options.include_tests_as_roots:
            labels.append("<test modules>")
        if self.options.public_api_is_live:
            api = public_api(repo, include_tests=False)
            roots.update(api)
            modules.update(index.symbol(q).parent for q in api if index.symbol(q) and index.symbol(q).parent and repo.module_path(index.symbol(q).parent))  # type: ignore[union-attr, arg-type]
            labels.append("<public API>")
        # modules reachable by import execute their top-level code
        live_modules = set(self.dependencies.reachable([m for m in modules if m in self.dependencies]))
        live_modules |= {m for m in modules if m not in self.dependencies}
        names = index.dynamic_names()
        for _, symbol in index.symbols():
            q = symbol.qualified_name
            if symbol.kind in (SymbolKind.FUNCTION, SymbolKind.METHOD, SymbolKind.CLASS):
                if symbol.name.startswith("__") and symbol.name.endswith("__"):
                    roots.add(q)
                elif {d.rsplit(".", 1)[-1] for d in symbol.decorators} - _NEUTRAL_DECORATORS:
                    roots.add(q)
                elif symbol.name in names or q in names:
                    roots.add(q)
        return roots, live_modules, labels

    # -- analysis ------------------------------------------------------------------------
    def analyze(self, *, scope: Iterable[str] = ()) -> DeadCodeReport:
        index, graph = self.index, self.calls
        roots, live_modules, labels = self._roots()
        seeds = {r for r in roots if r in graph} | {m for m in live_modules if m in graph}
        live = set(graph.reachable(seeds))
        # a live class keeps its constructor-ish dunders (already roots); a live method keeps overriders
        stack = [q for q in live if index.kind_of(q) is SymbolKind.METHOD]
        while stack:
            for over in index.overriders(stack.pop()):
                if over not in live and over in graph:
                    live.add(over)
                    stack.append(over)
        scopes = tuple(scope)
        items: list[DeadCodeItem] = []
        total = 0
        for path, symbol in index.symbols():
            q = symbol.qualified_name
            callable_symbol = symbol.kind in (SymbolKind.FUNCTION, SymbolKind.METHOD, SymbolKind.CLASS)
            private_variable = symbol.kind is SymbolKind.VARIABLE and symbol.name.startswith("_") and self.options.include_private_variables
            if not (callable_symbol or private_variable):
                continue
            if scopes and not any(q == s or q.startswith(s + ".") for s in scopes):
                continue
            total += 1
            model = index.repo.model(path)
            if model.source.is_test:
                continue
            if symbol.kind is SymbolKind.VARIABLE:
                if index.reference_count(q) > 1 or q in roots or symbol.name in index.dynamic_names() or (model.exports and symbol.name in model.exports):
                    continue
                item = self._variable_item(path, symbol)
            else:
                if q in live:
                    continue
                item = self._callable_item(path, symbol, model)
            if item is not None and item.confidence >= self.options.min_confidence:
                items.append(item)
        items.sort(key=lambda i: (-i.confidence, i.path, i.span.start_line))
        kept, omitted = truncate_items(items, self.options.max_items)
        return DeadCodeReport(
            items=tuple(kept),
            entry_points=tuple(dict.fromkeys(labels)),
            live_symbols=len(live),
            total_symbols=total,
            truncated=omitted,
        )

    def _variable_item(self, path: str, symbol: SymbolRecord) -> DeadCodeItem:
        return DeadCodeItem(symbol.qualified_name, symbol.kind, path, symbol.full_span, "private module/class variable is never read", 0.6, ("assignment may have side effects",))

    def _callable_item(self, path: str, symbol: SymbolRecord, model: Any) -> Optional[DeadCodeItem]:
        index = self.index
        q = symbol.qualified_name
        direct = index.reference_count(q, include_heuristic=True)
        caveats: list[str] = []
        confidence = 0.9 if symbol.name.startswith("_") else 0.8
        if direct == 0:
            reason = "never referenced anywhere in the repository"
        else:
            reason = "only referenced from code that is itself unreachable from the entry points"
            confidence -= 0.1
        if symbol.kind is SymbolKind.METHOD and symbol.parent:
            bases = index.class_bases(symbol.parent)
            odd = [b for b in (*bases.external, *bases.unresolved) if b not in _NEUTRAL_BASES]
            if odd:
                caveats.append(f"class has external/unresolved base(s) {', '.join(odd[:3])} that may call this method")
                confidence -= 0.35
            if index.overridden(q):
                caveats.append("overrides a base-class method")
                confidence -= 0.25
        if symbol.is_public and not self.options.public_api_is_live:
            caveats.append("public name: external callers are not visible to the analysis")
            confidence -= 0.1
        if any(scope.dynamic for scope in model.scopes if scope.qualified_name == (symbol.parent or "")):
            caveats.append("enclosing scope uses locals()/vars()/exec")
            confidence -= 0.2
        if model.has_star_import:
            caveats.append("module uses a wildcard import")
            confidence -= 0.05
        if index.symbol(f"{model.module_name}.__getattr__") is not None:
            caveats.append("module defines __getattr__")
            confidence -= 0.2
        if symbol.decorators:
            caveats.append("decorated: " + ", ".join(symbol.decorators[:3]))
            confidence -= 0.1
        return DeadCodeItem(q, symbol.kind, path, symbol.full_span, reason, round(max(0.0, min(1.0, confidence)), 3), tuple(caveats))


__all__ = ["DeadCodeAnalyzer", "DeadCodeOptions"]
