"""Central handler for the analysis modules: static analysis, dead code, impact, architecture."""

from __future__ import annotations

__version__ = "2.4.0"

import ast
import math
import re
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
        diagnostics.extend(SoftwareAnalysis._domain_diagnostics(code, parameters.get("ranges", {})))
        return {"assignments": assignments, "repeated_expressions": repeated, "helper_proposals": helpers,
                "equivalence_proposals": SoftwareAnalysis._integer_equivalences(assignments, parameters.get("integer_parameters", [])),
                "graph": {"nodes": len(adjacency), "cycles": [list(c) for c in sorted(cycles)], "unresolved": missing, "affected": sorted(affected)},
                "diagnostics": diagnostics,
                "test_proposals": [{"target": a["targets"], "cases": ["nominal", "boundary", "invalid-input", "determinism"]} for a in assignments],
                "executable_test_proposal": SoftwareAnalysis._executable_tests(parameters.get("test_samples", [])),
                "limitations": ["Range diagnostics are conservative and do not prove every execution path", "Integer equivalence depends on caller-attested built-in integer types", "Runtime dependency graph remains caller-owned"]}

    @staticmethod
    def _domain_diagnostics(code: str, raw_ranges: object) -> list[dict[str, object]]:
        """Bounded interval analysis; unknown values yield no unsupported certainty claims."""
        if not isinstance(raw_ranges, Mapping) or len(raw_ranges) > 256:
            raise ValueError("ranges must be a mapping of at most 256 intervals")
        intervals: dict[str, tuple[float, float]] = {}
        for name, pair in raw_ranges.items():
            if not isinstance(name, str) or not name.isidentifier() or not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError("ranges require identifier keys and [min, max] pairs")
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in pair) or pair[0] > pair[1]:
                raise ValueError("range endpoints must be finite and ordered")
            intervals[name] = (float(pair[0]), float(pair[1]))

        def bounds(node: ast.AST) -> tuple[float, float] | None:
            if isinstance(node, ast.Constant) and type(node.value) in (int, float) and math.isfinite(node.value):
                return float(node.value), float(node.value)
            if isinstance(node, ast.Name):
                return intervals.get(node.id)
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                part = bounds(node.operand)
                return (-part[1], -part[0]) if part and isinstance(node.op, ast.USub) else part
            if not isinstance(node, ast.BinOp):
                return None
            left, right = bounds(node.left), bounds(node.right)
            if left is None or right is None:
                return None
            try:
                if isinstance(node.op, ast.Add):
                    return left[0] + right[0], left[1] + right[1]
                if isinstance(node.op, ast.Sub):
                    return left[0] - right[1], left[1] - right[0]
                if isinstance(node.op, ast.Mult):
                    products = [x * y for x in left for y in right]
                    return min(products), max(products)
                if isinstance(node.op, ast.Div) and not (right[0] <= 0 <= right[1]):
                    quotients = [x / y for x in left for y in right]
                    return min(quotients), max(quotients)
            except (ArithmeticError, OverflowError):
                return None
            return None

        if not code:
            return []
        tree = ast.parse(code)
        result: list[dict[str, object]] = []
        seen: set[tuple[str, int, int]] = set()

        def issue(rule: str, msg: str, node: ast.AST, definite: bool) -> None:
            key = (rule, getattr(node, "lineno", 0), getattr(node, "col_offset", 0))
            if key not in seen:
                seen.add(key)
                result.append({"rule_id": rule, "severity": "error" if definite else "warning",
                               "message": msg, "line": getattr(node, "lineno", None),
                               "certainty": "definite_for_supplied_ranges" if definite else "possible"})

        for node in ast.walk(tree):
            if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)):
                if isinstance(node.right, ast.Constant) and type(node.right.value) in (int, float) and node.right.value == 0:
                    continue  # SWP002 already identifies literal division by zero
                interval = bounds(node.right)
                if interval is not None and interval[0] <= 0 <= interval[1]:
                    issue("SWP010", "Denominator may include zero", node, interval == (0.0, 0.0))
            if isinstance(node, ast.Call):
                name = node.func.id if isinstance(node.func, ast.Name) else (
                    node.func.attr if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "math" else None)
                if name in {"eval", "exec", "open", "__import__"}:
                    issue("SWP011", "Side effects or dynamic execution require independent review", node, False)
                if name in {"sqrt", "log", "log10", "log2", "asin", "acos"} and node.args:
                    interval = bounds(node.args[0])
                    if interval is None:
                        continue
                    lo, hi = interval
                    invalid = lo < 0 if name == "sqrt" else (lo <= 0 if name.startswith("log") else lo < -1 or hi > 1)
                    definite = hi < 0 if name == "sqrt" else (hi <= 0 if name.startswith("log") else hi < -1 or lo > 1)
                    if invalid:
                        issue("SWP012", f"{name} domain may be invalid", node, definite)
        return result

    @staticmethod
    def _integer_equivalences(assignments: list[dict[str, object]], raw_names: object) -> list[dict[str, object]]:
        """Prove bounded integer-polynomial identities without executing source.

        This is conditional on caller-attested built-in Python int operands. It
        intentionally cannot establish equivalence for user-defined operators,
        floats, division, mutation, or function calls.
        """
        if not isinstance(raw_names, (list, tuple)) or len(raw_names) > 256 or any(not isinstance(n, str) or not n.isidentifier() for n in raw_names):
            raise ValueError("integer_parameters must be a list of identifiers")
        integers = set(raw_names)

        def normal(node: ast.AST, depth: int = 0) -> dict[tuple[str, ...], int] | None:
            if depth > 32:
                return None
            if isinstance(node, ast.Name) and node.id in integers:
                return {(node.id,): 1}
            if isinstance(node, ast.Constant) and type(node.value) is int:
                if node.value.bit_length() > 256:
                    return None
                return {(): node.value} if node.value else {}
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                inner = normal(node.operand, depth + 1)
                if inner is not None:
                    return {term: (-coef if isinstance(node.op, ast.USub) else coef)
                            for term, coef in inner.items()}
            if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult)):
                left, right = normal(node.left, depth + 1), normal(node.right, depth + 1)
                if left is None or right is None:
                    return None
                terms: dict[tuple[str, ...], int] = dict(left) if not isinstance(node.op, ast.Mult) else {}
                if isinstance(node.op, (ast.Add, ast.Sub)):
                    sign = -1 if isinstance(node.op, ast.Sub) else 1
                    for term, coef in right.items():
                        terms[term] = terms.get(term, 0) + sign * coef
                else:
                    if len(left) * len(right) > 128:
                        return None
                    for left_term, left_coef in left.items():
                        for right_term, right_coef in right.items():
                            term = tuple(sorted(left_term + right_term))
                            if len(term) > 12:
                                return None
                            terms[term] = terms.get(term, 0) + left_coef * right_coef
                result = {term: coef for term, coef in terms.items() if coef}
                if len(result) <= 128 and all(coef.bit_length() <= 256 for coef in result.values()):
                    return result
            return None

        groups: dict[tuple[tuple[tuple[str, ...], int], ...], list[str]] = {}
        for assignment in assignments:
            canonical = normal(ast.parse(str(assignment["expression"]), mode="eval").body)
            if canonical is not None:
                groups.setdefault(tuple(sorted(canonical.items())), []).append(str(assignment["expression"]))
        return [{"expressions": sorted(set(items)),
                 "basis": "exact integer-polynomial normalization (addition, subtraction, multiplication)",
                 "assumptions": "every named operand is an unmodified built-in Python int",
                 "proof_scope": "algebraic_equivalence_under_stated_assumptions",
                 "requires_review": True} for items in groups.values() if len(set(items)) > 1]

    @staticmethod
    def _executable_tests(raw_samples: object) -> dict[str, object] | None:
        """Produce executable pytest source from explicit user-provided test oracles."""
        if not isinstance(raw_samples, (list, tuple)) or len(raw_samples) > 64:
            raise ValueError("test_samples must be a list of at most 64 examples")
        if not raw_samples:
            return None
        lines = ["import importlib", "import pytest", ""]
        for index, sample in enumerate(raw_samples):
            if not isinstance(sample, Mapping):
                raise ValueError("each test sample must be a mapping")
            module, function = sample.get("module"), sample.get("function")
            if not isinstance(module, str) or not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", module) or not isinstance(function, str) or not function.isidentifier():
                raise ValueError("test module and function must be valid dotted identifiers")
            args, kwargs = sample.get("args", []), sample.get("kwargs", {})
            if not isinstance(args, list) or len(args) > 32 or not isinstance(kwargs, Mapping) or len(kwargs) > 32 or any(not isinstance(k, str) or not k.isidentifier() for k in kwargs):
                raise ValueError("test args/kwargs must be bounded and use named kwargs")
            # Valid literals only; arbitrary objects or repr() hooks must never be emitted.
            import json
            try:
                json.dumps([args, kwargs, sample.get("expected")], allow_nan=False)
            except (TypeError, ValueError) as exc:
                raise ValueError("test inputs and expected results must be finite JSON values") from exc
            has_expected = "expected" in sample
            raises = sample.get("raises")
            if has_expected == bool(raises) or (raises is not None and raises not in {"ValueError", "TypeError", "ZeroDivisionError", "ArithmeticError"}):
                raise ValueError("each sample must specify exactly one of expected or a supported raises type")
            lines += [f"def test_generated_parametric_{index}():",
                      f"    function = getattr(importlib.import_module({module!r}), {function!r})"]
            call = f"function(*{args!r}, **{dict(kwargs)!r})"
            if raises:
                lines += [f"    with pytest.raises({raises}):", f"        {call}"]
            elif type(sample["expected"]) is float:
                lines.append(f"    assert {call} == pytest.approx({sample['expected']!r})")
            else:
                lines.append(f"    assert {call} == {sample['expected']!r}")
            lines.append("")
        return {"code": "\n".join(lines), "framework": "pytest", "applied": False,
                "tested": False, "requires_review": True, "oracle_source": "caller-supplied"}


__all__ = ["SoftwareAnalysis"]
