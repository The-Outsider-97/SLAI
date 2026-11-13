"""Central handler for ``graphs/``: workspace construction and graph queries."""

from __future__ import annotations

__version__ = "2.4.0"

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Optional

from .graphs import CallGraph, DependencyGraph
from .models import RepositoryModel
from .semantics.symbol_index import IndexOptions, SymbolIndex
from .software_models import SoftwareModels
from .utils.software_errors import SoftwareGraphError, SoftwareRequestError
from logs.logger import get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Software Graphs")


@dataclass(frozen=True, slots=True)
class Workspace:
    """A repository model with its derived symbol index and graphs (built lazily, cached)."""

    repo: RepositoryModel
    index: SymbolIndex
    graphs: "SoftwareGraphs"

    @property
    def dependencies(self) -> DependencyGraph:
        return self.graphs.dependency_graph(self.repo)

    @property
    def calls(self) -> CallGraph:
        return self.graphs.call_graph(self.repo)


class SoftwareGraphs:
    def __init__(self, models: SoftwareModels) -> None:
        self.models = models
        self.index_options = IndexOptions.from_config(models.section("symbol_index"))
        graph_cfg = models.section("graphs")
        self.include_references = bool(graph_cfg.get("include_reference_edges", True))
        self.include_heuristic = bool(graph_cfg.get("include_heuristic_calls", True))

    # -- workspace ---------------------------------------------------------------------------
    def _derived(self, kind: str, repo: RepositoryModel, builder: Any) -> Any:
        key = (kind, repo.digest, self.index_options, self.include_references, self.include_heuristic)
        return self.models.memory.derived.get_or_build(key, builder)

    def symbol_index(self, repo: RepositoryModel) -> SymbolIndex:
        return self._derived("index", repo, lambda: SymbolIndex(repo, self.index_options))

    def dependency_graph(self, repo: RepositoryModel) -> DependencyGraph:
        return self._derived("dependencies", repo, lambda: DependencyGraph.build(repo, name=repo.model_id))

    def call_graph(self, repo: RepositoryModel) -> CallGraph:
        return self._derived(
            "calls",
            repo,
            lambda: CallGraph.build(self.symbol_index(repo), include_heuristic=self.include_heuristic, include_references=self.include_references, name=repo.model_id),
        )

    def workspace(self, repo: RepositoryModel) -> Workspace:
        return Workspace(repo, self.symbol_index(repo), self)

    # -- queries --------------------------------------------------------------------------------
    def call_neighbours(self, repo: RepositoryModel, symbol: str, *, direction: str = "callees", transitive: bool = False, exact_only: bool = False, limit: int = 500) -> dict[str, Any]:
        graph = self.call_graph(repo)
        if symbol not in graph:
            raise SoftwareGraphError("symbol is not a node of the call graph", context={"symbol": symbol})
        if direction not in ("callers", "callees"):
            raise SoftwareRequestError("direction must be 'callers' or 'callees'", context={"direction": direction})
        names = graph.callers(symbol, transitive=transitive, exact_only=exact_only) if direction == "callers" else graph.callees(symbol, transitive=transitive, exact_only=exact_only)
        shown = names[: max(1, limit)]
        return {
            "symbol": symbol,
            "direction": direction,
            "transitive": transitive,
            "count": len(names),
            "symbols": list(shown),
            "truncated": len(names) - len(shown),
            "external_calls": list(graph.external_calls.get(symbol, ())),
        }

    def dependency_neighbours(self, repo: RepositoryModel, module: str, *, direction: str = "dependencies", transitive: bool = False, limit: int = 500) -> dict[str, Any]:
        graph = self.dependency_graph(repo)
        if module not in graph:
            raise SoftwareGraphError("module is not a node of the dependency graph", context={"module": module})
        if direction not in ("dependencies", "dependents"):
            raise SoftwareRequestError("direction must be 'dependencies' or 'dependents'", context={"direction": direction})
        names = graph.dependencies(module, transitive=transitive) if direction == "dependencies" else graph.dependents(module, transitive=transitive)
        shown = names[: max(1, limit)]
        return {"module": module, "direction": direction, "transitive": transitive, "count": len(names), "modules": list(shown), "truncated": len(names) - len(shown)}

    def cycles(self, repo: RepositoryModel, *, graph: str = "dependency", load_time_only: bool = False, limit: int = 50) -> dict[str, Any]:
        if graph == "dependency":
            base = self.dependency_graph(repo)
            groups = base.load_time_cycles() if load_time_only else base.cycles()
            witnesses = [list(base.cycle_witness(group)) for group in groups[:limit]]
        elif graph == "call":
            base = self.call_graph(repo)
            groups = base.recursive_groups()
            witnesses = [list(base.calls_only().cycle_witness(group)) for group in groups[:limit]]
        else:
            raise SoftwareRequestError("graph must be 'dependency' or 'call'", context={"graph": graph})
        return {"graph": graph, "cycles": [list(group) for group in groups[:limit]], "witnesses": witnesses, "count": len(groups)}

    def path(self, repo: RepositoryModel, source: str, target: str, *, graph: str = "dependency") -> dict[str, Any]:
        base = self.dependency_graph(repo) if graph == "dependency" else self.call_graph(repo) if graph == "call" else None
        if base is None:
            raise SoftwareRequestError("graph must be 'dependency' or 'call'", context={"graph": graph})
        found = base.shortest_path(source, target)
        return {"graph": graph, "source": source, "target": target, "path": list(found) if found else None}

    def summary(self, repo: RepositoryModel) -> dict[str, Any]:
        return {"dependency": self.dependency_graph(repo).summary(), "call": self.call_graph(repo).summary()}


__all__ = ["SoftwareGraphs", "Workspace"]
