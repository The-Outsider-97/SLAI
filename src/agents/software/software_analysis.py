"""Central handler for the analysis modules: static analysis, dead code, impact, architecture."""

from __future__ import annotations

__version__ = "2.4.0"

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


__all__ = ["SoftwareAnalysis"]
