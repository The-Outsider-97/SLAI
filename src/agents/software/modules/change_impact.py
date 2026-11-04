"""Change-impact analysis and test-impact selection.

Given changed symbols/files (or a before/after pair of repository models) the
analyzer walks the call/reference graph *backwards* to find code that may be
affected, adds inheritance effects (subclasses of a changed class, overriders
of a changed method), walks the module dependency graph backwards for module
level effects, ranks tests that exercise the affected region, and condenses
the evidence into a risk estimate.  All numbers are heuristic evidence for the
downstream verification/safety stages, not guarantees.
"""

from __future__ import annotations

__version__ = "2.4.0"

import math

from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Optional

from ..graphs.call_graph import CallGraph
from ..graphs.dependency_graph import DependencyGraph
from ..models.repository_model import RepositoryModel
from ..semantics.interface_compatibility import public_api
from ..semantics.symbol_index import SymbolIndex
from ..software_types import (
    ImpactedSymbol,
    ImpactReport,
    ParseStatus,
    Serializable,
    SymbolKind,
    TestCandidate,
    TestSelection,
)
from ..utils.software_helpers import opt_float, opt_int, truncate_items
from logs.logger import get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Software Change Impact")


@dataclass(frozen=True, slots=True)
class ChangeSet(Serializable):
    """Symbol-level difference between two repository states."""

    added: tuple[str, ...] = ()
    removed: tuple[str, ...] = ()
    modified: tuple[str, ...] = ()
    signature_changed: tuple[str, ...] = ()
    changed_paths: tuple[str, ...] = ()
    module_level: tuple[str, ...] = ()
    unparsed_paths: tuple[str, ...] = ()

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(sorted({*self.added, *self.removed, *self.modified}))


@dataclass(frozen=True, slots=True)
class ImpactOptions:
    max_depth: int = 6
    decay: float = 0.9
    inexact_penalty: float = 0.6
    max_impacted: int = 500
    max_tests: int = 200
    test_threshold: float = 0.2
    dependency_depth: int = 3
    weights: Mapping[str, float] = field(
        default_factory=lambda: {"blast_radius": 0.3, "public_api": 0.2, "centrality": 0.15, "complexity": 0.15, "test_gap": 0.15, "breadth": 0.05}
    )

    @classmethod
    def from_config(cls, section: Optional[Mapping[str, Any]] = None) -> "ImpactOptions":
        data = section or {}
        weights = dict(cls().weights)
        weights.update({k: float(v) for k, v in dict(data.get("risk_weights", {}) or {}).items() if k in weights})
        return cls(
            max_depth=opt_int(data, "max_depth", 6, minimum=1),
            decay=opt_float(data, "decay", 0.9, minimum=0.1, maximum=1.0),
            inexact_penalty=opt_float(data, "inexact_penalty", 0.6, minimum=0.05, maximum=1.0),
            max_impacted=opt_int(data, "max_impacted", 500, minimum=1),
            max_tests=opt_int(data, "max_tests", 200, minimum=1),
            test_threshold=opt_float(data, "test_threshold", 0.2, minimum=0.0, maximum=1.0),
            dependency_depth=opt_int(data, "dependency_depth", 3, minimum=1),
            weights=weights,
        )


def diff_models(before: RepositoryModel, after: RepositoryModel) -> ChangeSet:
    """Compare two repository states symbol by symbol (content digests)."""

    def table(repo: RepositoryModel) -> dict[str, Any]:
        return {s.qualified_name: s for m in repo.parsed() for s in m.symbols}

    old, new = table(before), table(after)
    added = tuple(sorted(new.keys() - old.keys()))
    removed = tuple(sorted(old.keys() - new.keys()))
    modified: list[str] = []
    signature: list[str] = []
    for name in sorted(old.keys() & new.keys()):
        a, b = old[name], new[name]
        if a.signature_digest != b.signature_digest:
            signature.append(name)
            modified.append(name)
        elif a.body_digest != b.body_digest:
            modified.append(name)
    changed_paths = after.snapshot.changed_paths(before.snapshot)
    touched_modules = {
        n for n in (after.module_name_of(p) or before.module_name_of(p) for p in changed_paths) if n
    }
    symbol_modules = {name.rsplit(".", 1)[0] for name in (*added, *removed, *modified)}
    module_level = tuple(sorted(m for m in touched_modules if not any(sm == m or sm.startswith(m + ".") for sm in symbol_modules)))
    unparsed = tuple(
        p for p in changed_paths
        if (after.get(p) is not None and after.model(p).status in (ParseStatus.SYNTAX_ERROR, ParseStatus.TOO_LARGE, ParseStatus.UNSUPPORTED))
        or (after.get(p) is None and before.get(p) is not None and before.model(p).status is ParseStatus.UNSUPPORTED)
    )
    return ChangeSet(added, removed, tuple(modified), tuple(signature), tuple(changed_paths), module_level, unparsed)


class ChangeImpactAnalyzer:
    def __init__(self, index: SymbolIndex, calls: CallGraph, dependencies: DependencyGraph, options: Optional[ImpactOptions] = None) -> None:
        self.index = index
        self.calls = calls
        self.dependencies = dependencies
        self.options = options or ImpactOptions()

    # -- inputs -------------------------------------------------------------------------
    def symbols_in_paths(self, paths: Iterable[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Symbols defined in ``paths`` and the paths that could not be mapped (unparsed)."""

        repo = self.index.repo
        symbols: list[str] = []
        unmapped: list[str] = []
        for path in paths:
            model = repo.get(path)
            if model is None or model.status is not ParseStatus.OK:
                unmapped.append(path)
                continue
            symbols.extend(s.qualified_name for s in model.symbols if s.kind in (SymbolKind.FUNCTION, SymbolKind.METHOD, SymbolKind.CLASS))
            if model.module_name:
                symbols.append(model.module_name)
        return tuple(dict.fromkeys(symbols)), tuple(unmapped)

    # -- core ------------------------------------------------------------------------------
    def _expand(self, changed: Iterable[str]) -> dict[str, tuple[int, float, tuple[str, ...]]]:
        """Reverse reachability: ``node -> (distance, confidence, path from a changed symbol)``."""

        o, graph, index = self.options, self.calls, self.index
        best: dict[str, tuple[int, float, tuple[str, ...]]] = {}
        queue: deque[str] = deque()
        for name in changed:
            if name in graph:
                best[name] = (0, 1.0, (name,))
                queue.append(name)
        while queue:
            current = queue.popleft()
            distance, confidence, path = best[current]
            if distance >= o.max_depth:
                continue
            neighbours: list[tuple[str, float]] = []
            for edge in graph.in_edges(current):
                neighbours.append((edge.source, o.decay * (1.0 if edge.exact else o.inexact_penalty)))
            symbol = index.symbol(current)
            if symbol is not None and symbol.kind is SymbolKind.CLASS:
                neighbours.extend((sub, o.decay) for sub in index.subclasses(current))
            elif symbol is not None and symbol.kind is SymbolKind.METHOD:
                neighbours.extend((over, o.decay * o.inexact_penalty) for over in index.overriders(current))
                neighbours.extend((base, o.decay * o.inexact_penalty) for base in index.overridden(current))
            for target, factor in neighbours:
                if target not in graph and index.symbol(target) is None:
                    continue
                candidate = (distance + 1, confidence * factor, (*path, target))
                known = best.get(target)
                if known is None or candidate[0] < known[0] or (candidate[0] == known[0] and candidate[1] > known[1]):
                    first_visit = known is None
                    best[target] = candidate
                    if first_visit or candidate[0] < known[0]:  # type: ignore[index]
                        queue.append(target)
        return best

    def analyze(self, *, symbols: Iterable[str] = (), paths: Iterable[str] = (), change_set: Optional[ChangeSet] = None) -> ImpactReport:
        o, index, repo = self.options, self.index, self.index.repo
        notes: list[str] = []
        changed: list[str] = [s for s in symbols]
        changed_paths = list(paths)
        unmapped: tuple[str, ...] = ()
        if change_set is not None:
            changed.extend(change_set.symbols)
            changed.extend(change_set.module_level)
            changed_paths.extend(change_set.changed_paths)
            unmapped = change_set.unparsed_paths
        if changed_paths:
            from_paths, missing = self.symbols_in_paths(changed_paths)
            if change_set is None:
                changed.extend(from_paths)
            unmapped = tuple(dict.fromkeys((*unmapped, *missing)))
        unknown = [s for s in changed if s not in self.calls and index.symbol(s) is None and not index.is_module(s)]
        if unknown:
            notes.append(f"{len(unknown)} changed symbol(s) are not present in the current model (removed or renamed): {', '.join(sorted(unknown)[:5])}")
        changed = [s for s in dict.fromkeys(changed) if s in self.calls]
        reach = self._expand(changed)

        impacted: list[ImpactedSymbol] = []
        for name, (distance, confidence, path) in reach.items():
            if distance == 0:
                continue
            kind = index.kind_of(name) or SymbolKind.MODULE
            impacted.append(ImpactedSymbol(name, kind, index.path_of(name) or "", distance, path[:-1][-4:], round(confidence, 4)))
        impacted.sort(key=lambda i: (i.distance, -i.confidence, i.qualified_name))

        changed_modules = self._modules_of(changed, changed_paths)
        impacted_modules = self._impacted_modules(changed_modules, reach)

        tests = self.select_tests(changed, changed_modules, reach, fallback=bool(unmapped))
        risk, factors = self._risk(changed, impacted, impacted_modules, tests)
        if unmapped:
            notes.append(f"{len(unmapped)} changed file(s) could not be analysed structurally ({', '.join(unmapped[:3])}); all tests are selected")
        kept, omitted = truncate_items(impacted, o.max_impacted)
        level = "low" if risk < 0.25 else "medium" if risk < 0.5 else "high" if risk < 0.75 else "critical"
        return ImpactReport(
            changed_symbols=tuple(sorted(changed)),
            changed_modules=tuple(sorted(changed_modules)),
            impacted_symbols=tuple(kept),
            impacted_modules=tuple(sorted(impacted_modules)),
            tests=tests,
            risk_score=round(risk, 4),
            risk_level=level,
            risk_factors={k: round(v, 4) for k, v in factors.items()},
            notes=tuple(notes),
            truncated=omitted,
        )

    def analyze_change(self, before: RepositoryModel, after: RepositoryModel) -> ImpactReport:
        """Impact of ``before -> after``.  ``self`` must be built over ``after`` (removed callers are gone)."""

        change_set = diff_models(before, after)
        report = self.analyze(change_set=change_set)
        return report

    def _modules_of(self, changed: Iterable[str], paths: Iterable[str]) -> set[str]:
        repo = self.index.repo
        modules = {n for p in paths if (n := repo.module_name_of(p))}
        for name in changed:
            if repo.module_path(name):
                modules.add(name)
            else:
                path = self.index.path_of(name)
                module = repo.module_name_of(path) if path else None
                if module:
                    modules.add(module)
        return modules

    def _impacted_modules(self, changed_modules: set[str], reach: Mapping[str, tuple[int, float, tuple[str, ...]]]) -> set[str]:
        dg, repo = self.dependencies, self.index.repo
        seeds = [m for m in changed_modules if m in dg]
        modules = set(dg.reachable(seeds, direction="in", max_depth=self.options.dependency_depth)) - set(changed_modules)
        for name in reach:
            path = self.index.path_of(name)
            module = repo.module_name_of(path) if path else None
            if module and module not in changed_modules:
                modules.add(module)
        return modules

    # -- tests -------------------------------------------------------------------------------
    def select_tests(
        self,
        changed: Iterable[str],
        changed_modules: set[str],
        reach: Mapping[str, tuple[int, float, tuple[str, ...]]],
        *,
        fallback: bool = False,
    ) -> TestSelection:
        o, index, repo, dg = self.options, self.index, self.index.repo, self.dependencies
        test_models = repo.test_models()
        if fallback:
            return TestSelection(
                tuple(TestCandidate(m.path, None, 1.0, ("unanalysable change: run everything",)) for m in test_models[: o.max_tests]),
                len(test_models), tuple(sorted(changed_modules)), True,
            )
        reasons: dict[str, list[tuple[float, str]]] = {}
        functions: dict[tuple[str, str], tuple[float, str]] = {}
        for name, (distance, confidence, _) in reach.items():
            path = index.path_of(name)
            model = repo.get(path) if path else None
            if model is None or not model.source.is_test:
                continue
            reasons.setdefault(model.path, []).append((confidence, f"exercises affected symbol '{name}' (distance {distance})"))
            symbol = index.symbol(name)
            if symbol is not None and symbol.kind in (SymbolKind.FUNCTION, SymbolKind.METHOD) and symbol.name.startswith("test"):
                functions[(model.path, name)] = (confidence, "calls into the changed region")
        forward_depth = {m: dg.reachable([m], max_depth=o.dependency_depth) for m in changed_modules if m in dg}
        for model in test_models:
            name = repo.module_name_of(model.path)
            if name is None or name not in dg:
                continue
            tested = dg.reachable([name], max_depth=o.dependency_depth)
            for module in changed_modules:
                distance = tested.get(module)
                if distance is not None and distance > 0:
                    reasons.setdefault(model.path, []).append((0.6 / distance, f"imports '{module}' (distance {distance})"))
            stem = model.path.rsplit("/", 1)[-1].removeprefix("test_").removesuffix("_test.py").removesuffix(".py")
            for module in changed_modules:
                if module.rsplit(".", 1)[-1] == stem:
                    reasons.setdefault(model.path, []).append((0.5, f"test file name matches module '{module}'"))
        _ = forward_depth
        candidates: list[TestCandidate] = []
        for path, items in reasons.items():
            miss = 1.0
            for score, _ in items:
                miss *= 1.0 - min(score, 0.99)
            score = 1.0 - miss  # noisy-or over independent evidence
            if score >= o.test_threshold:
                texts = [text for _, text in sorted(items, key=lambda it: -it[0])][:4]
                candidates.append(TestCandidate(path, None, round(score, 4), tuple(texts)))
        for (path, name), (score, text) in functions.items():
            if score >= o.test_threshold:
                candidates.append(TestCandidate(path, name, round(score, 4), (text,)))
        candidates.sort(key=lambda c: (-c.score, c.path, c.qualified_name or ""))
        return TestSelection(tuple(candidates[: o.max_tests]), len(test_models), tuple(sorted(changed_modules)), False)

    # -- risk ---------------------------------------------------------------------------------
    def _risk(
        self,
        changed: list[str],
        impacted: list[ImpactedSymbol],
        impacted_modules: set[str],
        tests: TestSelection,
    ) -> tuple[float, dict[str, float]]:
        o, index, graph = self.options, self.index, self.calls
        if not changed:
            return 0.0, {key: 0.0 for key in o.weights}
        blast = min(1.0, math.log1p(len(impacted) + len(impacted_modules)) / math.log1p(200))
        api = public_api(index.repo)
        api_hits = sum(1 for s in changed if s in api)
        public = min(1.0, api_hits / max(1, len(changed)) + (0.2 if api_hits else 0.0))
        degree = max((graph.in_degree(s) for s in changed if s in graph), default=0)
        centrality = min(1.0, math.log1p(degree) / math.log1p(50))
        cyclomatic = max((m.cyclomatic for s in changed if (sym := index.symbol(s)) and (m := sym.metrics)), default=1)
        complexity = min(1.0, cyclomatic / 30.0)
        covered = {c.qualified_name for c in tests.selected if c.qualified_name} | {c.path for c in tests.selected}
        test_gap = 0.0 if tests.fallback_to_all else (1.0 if not covered else max(0.0, 1.0 - min(1.0, len(tests.selected) / max(1, len(changed)))))
        packages = {s.rsplit(".", 2)[0] for s in changed}
        breadth = min(1.0, len(packages) / 5.0)
        factors = {"blast_radius": blast, "public_api": public, "centrality": centrality, "complexity": complexity, "test_gap": test_gap, "breadth": breadth}
        total_weight = sum(o.weights.values()) or 1.0
        return sum(o.weights[k] * v for k, v in factors.items()) / total_weight, factors


__all__ = ["ChangeImpactAnalyzer", "ChangeSet", "ImpactOptions", "diff_models"]
