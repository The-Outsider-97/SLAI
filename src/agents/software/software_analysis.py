"""Central handler for the analysis modules: static analysis, dead code, impact, architecture."""

from __future__ import annotations

__version__ = "2.4.0"

import ast
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import replace
from typing import Any, Optional

from .models import RepositoryModel
from .modules.architecture import ArchitectureAnalyzer, ArchitectureOptions, ForbiddenSpec, LayerSpec
from .modules.change_impact import ChangeImpactAnalyzer, ImpactOptions
from .modules.dead_code import DeadCodeAnalyzer, DeadCodeOptions
from .modules.static_analysis import AnalysisOptions, StaticAnalyzer
from .software_semantics import SoftwareSemantics
from .software_types import ArchitectureReport, DeadCodeReport, ImpactReport, StaticAnalysisReport
from logs.logger import get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Software Analysis")


class SoftwareAnalysis:
    def __init__(self, semantics: SoftwareSemantics) -> None:
        self.semantics = semantics
        self.graphs = semantics.graphs
        self.models = semantics.models
        self.static_options = AnalysisOptions.from_config(self.models.section("static_analysis"))
        self.dead_options = DeadCodeOptions.from_config(self.models.section("dead_code"))
        self.impact_options = ImpactOptions.from_config(self.models.section("change_impact"))
        self.architecture_options = ArchitectureOptions.from_config(self.models.section("architecture"))

    def static(self, repo: RepositoryModel, *, paths: Optional[Iterable[str]] = None, rules: Optional[Iterable[str]] = None, options: Optional[Mapping[str, Any]] = None) -> StaticAnalysisReport:
        opts = AnalysisOptions.from_config({**self.models.section("static_analysis"), **dict(options or {})}) if options else self.static_options
        return StaticAnalyzer(self.semantics.index(repo), opts).analyze(paths, rules=rules)

    def dead_code(self, repo: RepositoryModel, *, scope: Iterable[str] = (), options: Optional[Mapping[str, Any]] = None) -> DeadCodeReport:
        opts = DeadCodeOptions.from_config({**self.models.section("dead_code"), **dict(options or {})}) if options else self.dead_options
        ws = self.graphs.workspace(repo)
        return DeadCodeAnalyzer(ws.index, ws.calls, ws.dependencies, opts).analyze(scope=scope)

    def impact(
        self,
        repo: RepositoryModel,
        *,
        symbols: Iterable[str] = (),
        paths: Iterable[str] = (),
        baseline: Optional[RepositoryModel] = None,
    ) -> ImpactReport:
        """Impact of explicit symbols/paths on ``repo``, or of ``baseline → repo`` when a baseline is given."""

        ws = self.graphs.workspace(repo)
        analyzer = ChangeImpactAnalyzer(ws.index, ws.calls, ws.dependencies, self.impact_options)
        if baseline is not None:
            return analyzer.analyze_change(baseline, repo)
        resolved = [self.semantics.resolve_symbol(repo, s) for s in symbols]
        return analyzer.analyze(symbols=resolved, paths=paths)

    def architecture(self, repo: RepositoryModel, *, layers: Optional[Iterable[Mapping[str, Any]]] = None, forbidden: Optional[Iterable[Mapping[str, Any]]] = None, options: Optional[Mapping[str, Any]] = None) -> ArchitectureReport:
        opts = self.architecture_options
        if options or layers is not None or forbidden is not None:
            merged = {**self.models.section("architecture"), **dict(options or {})}
            if layers is not None:
                merged["layers"] = list(layers)
            if forbidden is not None:
                merged["forbidden"] = list(forbidden)
            opts = ArchitectureOptions.from_config(merged)
        ws = self.graphs.workspace(repo)
        return ArchitectureAnalyzer(ws.index, ws.dependencies, opts).analyze()


    @staticmethod
    def parametric(parameters: Mapping[str, object]) -> dict[str, object]:
        """Analyze assignments and a caller-owned graph; never execute or mutate them."""
        code = parameters.get("code", "")
        graph = parameters.get("graph", {})
        if not isinstance(code, str) or len(code) > 100_000:
            raise ValueError("code must be a string under 100000 characters")
        if not isinstance(graph, Mapping) or len(graph) > 10_000:
            raise ValueError("graph must be a mapping with at most 10000 nodes")
        diagnostics: list[dict[str, object]] = []
        assignments: list[dict[str, object]] = []
        expressions: Counter[str] = Counter()
        if code:
            try:
                tree = ast.parse(code)
            except SyntaxError as exc:
                return {"assignments": [], "repeated_expressions": [], "graph": {}, "diagnostics": [{"rule_id": "SWP001", "severity": "error", "message": str(exc), "line": exc.lineno}], "test_proposals": []}
            if sum(1 for _ in ast.walk(tree)) > 20_000:
                raise ValueError("code exceeds AST budget")
            for node in ast.walk(tree):
                if not isinstance(node, (ast.Assign, ast.AnnAssign)) or node.value is None:
                    continue
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                names = [target.id for target in targets if isinstance(target, ast.Name)]
                if not names:
                    continue
                key = ast.dump(node.value, include_attributes=False)
                expressions[key] += 1
                assignments.append({"targets": names, "reads": sorted({n.id for n in ast.walk(node.value) if isinstance(n, ast.Name)}), "expression": ast.unparse(node.value), "line": node.lineno, "_key": key})
                for part in ast.walk(node.value):
                    if isinstance(part, ast.BinOp) and isinstance(part.op, ast.Div) and isinstance(part.right, ast.Constant) and part.right.value == 0:
                        diagnostics.append({"rule_id": "SWP002", "severity": "error", "message": "Literal division by zero", "line": part.lineno})
        repeated = sorted({a["expression"] for a in assignments if expressions[a["_key"]] > 1})
        helpers: list[dict[str, object]] = []
        for index, expression in enumerate(repeated, 1):
            expression_tree = ast.parse(expression, mode="eval").body
            if any(not isinstance(node, (ast.BinOp, ast.UnaryOp, ast.Name, ast.Load, ast.Constant,
                                             ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.UAdd, ast.USub))
                   for node in ast.walk(expression_tree)):
                continue  # calls, subscriptions and attributes may have side effects
            inputs = sorted({node.id for node in ast.walk(expression_tree) if isinstance(node, ast.Name)})
            function_name = f"compute_shared_{index}"
            helpers.append({"name": function_name, "inputs": inputs,
                            "proposed_code": f"def {function_name}({', '.join(inputs)}):\n    return {expression}\n",
                            "requires_review": True, "equivalence": "syntactic-only"})
        for assignment in assignments:
            assignment.pop("_key")
        adjacency: dict[str, tuple[str, ...]] = {}
        edge_count = 0
        for node, deps in graph.items():
            if not isinstance(node, str) or not node or not isinstance(deps, (list, tuple)) or len(deps) > 10_000 or any(not isinstance(d, str) or not d for d in deps):
                raise ValueError("graph entries must map node names to lists of dependency names")
            adjacency[node] = tuple(dict.fromkeys(deps))
            edge_count += len(adjacency[node])
            if edge_count > 100_000:
                raise ValueError("graph exceeds 100000 dependency edges")
        missing = sorted({d for deps in adjacency.values() for d in deps if d not in adjacency})
        state: dict[str, int] = {}
        cycles: set[tuple[str, ...]] = set()
        for root in sorted(adjacency):
            if state.get(root):
                continue
            stack: list[tuple[str, bool]] = [(root, False)]
            path: list[str] = []
            while stack:
                node, leave = stack.pop()
                if leave:
                    state[node] = 2
                    if path and path[-1] == node:
                        path.pop()
                elif state.get(node) == 1:
                    if node in path:
                        cycles.add(tuple(path[path.index(node):] + [node]))
                elif state.get(node) != 2:
                    state[node] = 1
                    path.append(node)
                    stack.append((node, True))
                    for dep in reversed(adjacency[node]):
                        if dep in adjacency:
                            stack.append((dep, False))
        changed = parameters.get("changed", [])
        if isinstance(changed, str):
            changed = [changed]
        if not isinstance(changed, (list, tuple)) or any(not isinstance(n, str) for n in changed):
            raise ValueError("changed must be a sequence of names")
        reverse: dict[str, set[str]] = {}
        for node, deps in adjacency.items():
            for dep in deps:
                reverse.setdefault(dep, set()).add(node)
        affected: set[str] = set()
        todo = list(changed)
        while todo:
            for dependent in reverse.get(todo.pop(), ()):
                if dependent not in affected:
                    affected.add(dependent)
                    todo.append(dependent)
        if cycles:
            diagnostics.append({"rule_id": "SWP004", "severity": "error", "message": "Dependency cycles", "nodes": [list(c) for c in sorted(cycles)]})
        if missing:
            diagnostics.append({"rule_id": "SWP005", "severity": "warning", "message": "Unresolved dependencies", "nodes": missing})
        return {"assignments": assignments, "repeated_expressions": repeated, "helper_proposals": helpers, "graph": {"nodes": len(adjacency), "cycles": [list(c) for c in sorted(cycles)], "unresolved": missing, "affected": sorted(affected)}, "diagnostics": diagnostics, "test_proposals": [{"target": a["targets"], "cases": ["nominal", "boundary", "invalid-input", "determinism"]} for a in assignments], "limitations": ["Syntactic repetition is not mathematical equivalence", "Runtime dependency graph remains caller-owned"]}


__all__ = ["SoftwareAnalysis"]
