"""Module-level import dependency graph.

An edge ``a -> b`` means module ``a`` imports module ``b``.  Edges carry the
*way* the import happens (``load_time``, ``lazy``, ``type_checking``,
``conditional``) because only load-time cycles break imports; the other kinds
are architectural coupling.
"""

from __future__ import annotations

__version__ = "2.4.0"

from collections.abc import Mapping
from typing import Any, Optional

from .base_graph import BaseGraph, GraphEdge
from ..models.repository_model import RepositoryModel
from ..software_types import EdgeKind
from ..utils.software_helpers import package_of
from logs.logger import get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Software Dependency Graph")

ROOT_PACKAGE = "<root>"


def _import_tags(record: Any) -> tuple[str, ...]:
    tags = []
    if record.type_checking:
        tags.append("type_checking")
    if record.lazy:
        tags.append("lazy")
    if record.conditional:
        tags.append("conditional")
    if not record.type_checking and not record.lazy:
        tags.append("load_time")
    return tuple(tags)


class DependencyGraph(BaseGraph):
    graph_type = "dependency"

    def __init__(
        self,
        nodes: Any,
        edges: Any,
        *,
        name: str = "",
        packages: Optional[Mapping[str, str]] = None,
        external: Optional[Mapping[str, tuple[str, ...]]] = None,
        unresolved: tuple[tuple[str, int, str], ...] = (),
        test_modules: frozenset[str] = frozenset(),
    ) -> None:
        super().__init__(nodes, edges, name=name)
        self._package_of = dict(packages or {})
        self.external: dict[str, tuple[str, ...]] = dict(external or {})
        self.unresolved = tuple(unresolved)
        self.test_modules = test_modules

    @classmethod
    def build(cls, repo: RepositoryModel, *, name: str = "") -> "DependencyGraph":
        nodes: set[str] = set()
        packages: dict[str, str] = {}
        tests: set[str] = set()
        for model in repo.models():
            if model.module_name or model.is_package:
                if repo.module_path(model.module_name) == model.path:
                    nodes.add(model.module_name)
                    packages[model.module_name] = package_of(model.module_name, model.is_package) or (model.module_name if model.is_package else ROOT_PACKAGE)
                    if model.source.is_test:
                        tests.add(model.module_name)
        edges: list[GraphEdge] = []
        external: dict[str, set[str]] = {}
        unresolved: list[tuple[str, int, str]] = []
        for path, record, resolved in repo.all_imports():
            source = repo.module_name_of(path)
            if source is None:
                continue
            if resolved.internal:
                target = resolved.target_module
                if target is None or target not in nodes:
                    if target is None:
                        unresolved.append((source, record.span.start_line, repo.absolute_module(record, repo.model(path)) or ""))
                    continue
                if target == source:
                    continue
                edges.append(GraphEdge(source, target, EdgeKind.IMPORTS, 1, _import_tags(record), (record.span.start_line,)))
            elif resolved.external_package:
                external.setdefault(source, set()).add(resolved.external_package)
        return cls(
            nodes,
            edges,
            name=name,
            packages=packages,
            external={key: tuple(sorted(value)) for key, value in external.items()},
            unresolved=tuple(sorted(unresolved)),
            test_modules=frozenset(tests),
        )

    # -- queries -------------------------------------------------------------------
    def dependencies(self, module: str, *, transitive: bool = False) -> tuple[str, ...]:
        if not transitive:
            return self.successors(module)
        return tuple(sorted(node for node in self.reachable([module]) if node != module))

    def dependents(self, module: str, *, transitive: bool = False) -> tuple[str, ...]:
        if not transitive:
            return self.predecessors(module)
        return tuple(sorted(node for node in self.reachable([module], direction="in") if node != module))

    def load_time(self) -> BaseGraph:
        """Subgraph of imports executed when a module is imported."""

        return self.filtered(lambda edge: "load_time" in edge.tags, name=f"{self.name}/load_time")

    def load_time_cycles(self) -> tuple[tuple[str, ...], ...]:
        return self.load_time().cycles()

    def package_of(self, module: str) -> str:
        return self._package_of.get(module, ROOT_PACKAGE)

    def package_graph(self, *, level: Optional[int] = None) -> BaseGraph:
        """Condense modules into packages (optionally truncated to ``level`` components)."""

        def owner(module: str) -> str:
            package = self.package_of(module)
            if level is not None and package != ROOT_PACKAGE:
                package = ".".join(package.split(".")[:level])
            return package

        edges = []
        for edge in self.edges:
            source, target = owner(edge.source), owner(edge.target)
            if source != target:
                edges.append(GraphEdge(source, target, EdgeKind.IMPORTS, edge.weight, edge.tags, edge.lines))
        return BaseGraph({owner(node) for node in self.nodes}, edges, name=f"{self.name}/packages")

    def modules_in(self, package: str) -> tuple[str, ...]:
        return tuple(node for node in self.nodes if self.package_of(node) == package or node == package)

    def external_packages(self) -> dict[str, int]:
        """Third-party package → number of internal modules that import it."""

        counts: dict[str, int] = {}
        for packages in self.external.values():
            for package in packages:
                counts[package] = counts.get(package, 0) + 1
        return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))

    def summary(self) -> dict[str, Any]:
        base = super().summary()
        base.update(
            {
                "load_time_cycles": len(self.load_time_cycles()),
                "unresolved_internal_imports": len(self.unresolved),
                "external_packages": len(self.external_packages()),
            }
        )
        return base


__all__ = ["DependencyGraph", "ROOT_PACKAGE"]
