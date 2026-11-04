"""Repository architecture analysis: package metrics, cycles, layers, hubs.

Package metrics follow R. C. Martin (*Agile Software Development*, 2002):
afferent coupling ``Ca`` (outside modules that depend on the package),
efferent coupling ``Ce`` (outside modules the package depends on),
instability ``I = Ce / (Ca + Ce)``, abstractness ``A`` (abstract / total
classes) and distance from the main sequence ``D = |A + I - 1|``.
"""

from __future__ import annotations

__version__ = "2.4.0"

import fnmatch

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Optional

from ..graphs.dependency_graph import DependencyGraph, ROOT_PACKAGE
from ..semantics.symbol_index import SymbolIndex
from ..software_types import ArchitectureReport, LayerViolation, PackageMetrics, SymbolKind
from ..utils.software_helpers import opt_int, safe_ratio
from logs.logger import get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Software Architecture")


@dataclass(frozen=True, slots=True)
class LayerSpec:
    name: str
    patterns: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ForbiddenSpec:
    source: str
    target: str
    reason: str = ""


@dataclass(frozen=True, slots=True)
class ArchitectureOptions:
    layers: tuple[LayerSpec, ...] = ()
    forbidden: tuple[ForbiddenSpec, ...] = ()
    hub_count: int = 10
    package_level: Optional[int] = None
    include_tests: bool = False
    load_time_only_violations: bool = False

    @classmethod
    def from_config(cls, section: Optional[Mapping[str, Any]] = None) -> "ArchitectureOptions":
        data = section or {}
        layers = tuple(
            LayerSpec(str(item.get("name", f"layer{i}")), tuple(str(p) for p in item.get("packages", ()) or ()))
            for i, item in enumerate(data.get("layers", ()) or ())
            if isinstance(item, Mapping)
        )
        forbidden = tuple(
            ForbiddenSpec(str(item.get("from", "")), str(item.get("to", "")), str(item.get("reason", "")))
            for item in (data.get("forbidden", ()) or ())
            if isinstance(item, Mapping) and item.get("from") and item.get("to")
        )
        level = data.get("package_level")
        return cls(
            layers=layers,
            forbidden=forbidden,
            hub_count=opt_int(data, "hub_count", 10, minimum=1),
            package_level=int(level) if level not in (None, "", 0) else None,
            include_tests=bool(data.get("include_tests", False)),
            load_time_only_violations=bool(data.get("load_time_only_violations", False)),
        )


def _matches(name: str, patterns: Sequence[str]) -> bool:
    return any(fnmatch.fnmatchcase(name, p) or name == p or name.startswith(p.rstrip("*").rstrip(".") + ".") for p in patterns)


class ArchitectureAnalyzer:
    def __init__(self, index: SymbolIndex, dependencies: DependencyGraph, options: Optional[ArchitectureOptions] = None) -> None:
        self.index = index
        self.dependencies = dependencies
        self.options = options or ArchitectureOptions()

    def _graph(self) -> DependencyGraph:
        dg = self.dependencies
        if self.options.include_tests or not dg.test_modules:
            return dg
        keep = [n for n in dg.nodes if n not in dg.test_modules]
        sub = dg.subgraph(keep)
        return DependencyGraph(sub.nodes, sub.edges, name=dg.name, packages={n: dg.package_of(n) for n in sub.nodes}, external=dg.external, unresolved=dg.unresolved)

    def _owner(self, dg: DependencyGraph, module: str) -> str:
        package = dg.package_of(module)
        level = self.options.package_level
        if level is not None and package != ROOT_PACKAGE:
            package = ".".join(package.split(".")[:level])
        return package

    def package_metrics(self, dg: DependencyGraph) -> tuple[PackageMetrics, ...]:
        members: dict[str, list[str]] = {}
        for module in dg.nodes:
            members.setdefault(self._owner(dg, module), []).append(module)
        abstract: dict[str, list[int]] = {}
        for _, symbol in self.index.symbols():
            if symbol.kind is not SymbolKind.CLASS:
                continue
            module = symbol.qualified_name.rsplit(".", 1)[0]
            while module and module not in dg:
                module = module.rsplit(".", 1)[0] if "." in module else ""
            if not module:
                continue
            counts = abstract.setdefault(self._owner(dg, module), [0, 0])
            counts[1] += 1
            counts[0] += 1 if symbol.is_abstract else 0
        result: list[PackageMetrics] = []
        for package, modules in sorted(members.items()):
            inside = set(modules)
            afferent = {src for m in modules for src in dg.predecessors(m) if src not in inside}
            efferent = {dst for m in modules for dst in dg.successors(m) if dst not in inside}
            ca, ce = len(afferent), len(efferent)
            instability = safe_ratio(ce, ca + ce, 0.0)
            a, total = abstract.get(package, [0, 0])
            abstractness = safe_ratio(a, total, 0.0)
            result.append(PackageMetrics(package, len(modules), ca, ce, round(instability, 4), round(abstractness, 4), round(abs(abstractness + instability - 1.0), 4)))
        return tuple(result)

    def _layer_of(self, dg: DependencyGraph, module: str) -> Optional[int]:
        for rank, layer in enumerate(self.options.layers):
            if _matches(module, layer.patterns) or _matches(dg.package_of(module), layer.patterns):
                return rank
        return None

    def violations(self, dg: DependencyGraph) -> tuple[LayerViolation, ...]:
        o = self.options
        found: list[LayerViolation] = []
        for edge in dg.edges:
            if o.load_time_only_violations and "load_time" not in edge.tags:
                continue
            line = edge.lines[0] if edge.lines else None
            path = self.index.repo.module_path(edge.source)
            source_layer, target_layer = self._layer_of(dg, edge.source), self._layer_of(dg, edge.target)
            if source_layer is not None and target_layer is not None and target_layer < source_layer:
                found.append(LayerViolation(edge.source, edge.target, o.layers[source_layer].name, o.layers[target_layer].name, "lower layers must not depend on higher layers", path, line))
            for rule in o.forbidden:
                if _matches(edge.source, (rule.source,)) and _matches(edge.target, (rule.target,)):
                    found.append(LayerViolation(edge.source, edge.target, rule.source, rule.target, rule.reason or "forbidden dependency", path, line))
        return tuple(sorted(found, key=lambda v: (v.source, v.target, v.rule)))

    def analyze(self) -> ArchitectureReport:
        dg = self._graph()
        packages = self.package_metrics(dg)
        module_cycles = dg.cycles()
        package_graph = dg.package_graph(level=self.options.package_level)
        package_cycles = package_graph.cycles()
        degrees = sorted(((n, dg.in_degree(n) + dg.out_degree(n)) for n in dg.nodes), key=lambda item: (-item[1], item[0]))
        hubs = tuple(item for item in degrees[: self.options.hub_count] if item[1] > 0)
        models = self.index.repo
        orphans = tuple(
            n for n in dg.nodes
            if dg.in_degree(n) == 0 and dg.out_degree(n) == 0 and not (models.module(n) and (models.module(n).is_package or models.module(n).source.is_test))  # type: ignore[union-attr]
        )
        layers = dg.layers()
        violations = self.violations(dg)
        load_cycles = dg.load_time_cycles()
        mean_instability = sum(p.instability for p in packages) / len(packages) if packages else 0.0
        mean_distance = sum(p.distance for p in packages) / len(packages) if packages else 0.0
        summary = {
            "modules": len(dg),
            "packages": len(packages),
            "import_edges": dg.edge_count,
            "module_cycles": len(module_cycles),
            "load_time_cycles": len(load_cycles),
            "package_cycles": len(package_cycles),
            "layer_violations": len(violations),
            "dependency_depth": len(layers),
            "mean_instability": round(mean_instability, 4),
            "mean_distance": round(mean_distance, 4),
            "unresolved_internal_imports": len(dg.unresolved),
            "worst_distance": max(((p.distance, p.package) for p in packages), default=(0.0, ""))[1],
        }
        return ArchitectureReport(
            packages=packages,
            module_cycles=module_cycles,
            package_cycles=package_cycles,
            layer_violations=violations,
            hubs=hubs,
            orphans=orphans,
            layers=layers,
            external_dependencies=dg.external_packages(),
            summary=summary,
        )


__all__ = ["ArchitectureAnalyzer", "ArchitectureOptions", "ForbiddenSpec", "LayerSpec"]
