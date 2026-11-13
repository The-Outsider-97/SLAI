"""Central handler for patch synthesis: refactorings, API migration, external diffs.

Every method returns a :class:`PatchResult` whose proposal is already recorded
in the subsystem ledger.  Nothing is applied; the proposal states the gates
(verification, safety, execution, evaluation) that must act next.
"""

from __future__ import annotations

__version__ = "2.4.0"

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Optional

from .models import RepositoryModel
from .modules.api_migration import ApiMigrator, MigrationOptions, parse_rules
from .modules.patch_builder import PatchBuilder, PatchLimits, PatchResult
from .modules.refactoring import RefactoringEngine, RefactoringOptions, regression_validator
from .software_analysis import SoftwareAnalysis
from .software_memory import LedgerEntry
from .software_types import MigrationRule, PatchProposal
from logs.logger import get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Software Refactoring")


class SoftwareRefactoring:
    def __init__(self, analysis: SoftwareAnalysis) -> None:
        self.analysis = analysis
        self.semantics = analysis.semantics
        self.graphs = analysis.graphs
        self.models = analysis.models
        self.memory = self.models.memory
        self.builder = PatchBuilder(self.models.parser, PatchLimits.from_config(self.models.section("patch")))
        self.refactoring_options = RefactoringOptions.from_config(self.models.section("refactoring"))
        self.migration_options = MigrationOptions.from_config(self.models.section("api_migration"))

    # -- internals --------------------------------------------------------------------------------
    def _engine(self, repo: RepositoryModel, parent: Optional[str]) -> RefactoringEngine:
        return RefactoringEngine(self.semantics.index(repo), self.builder, self.refactoring_options, cache=self.memory.models, parent_patch_id=parent)

    def _commit(self, result: PatchResult) -> PatchResult:
        self.memory.record(result.proposal, result.snapshot)
        logger.info("patch proposed | id=%s | kind=%s | files=%s | confidence=%.2f", result.proposal.patch_id, result.proposal.kind.value, len(result.proposal.files), result.proposal.confidence)
        return result

    # -- refactorings -------------------------------------------------------------------------------
    def rename(self, repo: RepositoryModel, symbol: str, new_name: str, *, include_heuristic: bool = False, parent_patch_id: Optional[str] = None) -> PatchResult:
        qname = self.semantics.resolve_symbol(repo, symbol)
        return self._commit(self._engine(repo, parent_patch_id).rename_symbol(qname, new_name, include_heuristic=include_heuristic))

    def remove_unused_imports(self, repo: RepositoryModel, *, paths: Optional[Iterable[str]] = None, parent_patch_id: Optional[str] = None) -> PatchResult:
        return self._commit(self._engine(repo, parent_patch_id).remove_unused_imports(paths))

    def remove_dead_code(
        self,
        repo: RepositoryModel,
        *,
        min_confidence: Optional[float] = None,
        symbols: Optional[Iterable[str]] = None,
        scope: Iterable[str] = (),
        options: Optional[Mapping[str, Any]] = None,
        parent_patch_id: Optional[str] = None,
    ) -> PatchResult:
        report = self.analysis.dead_code(repo, scope=scope, options=options)
        wanted = None if symbols is None else [self.semantics.resolve_symbol(repo, s) for s in symbols]
        return self._commit(self._engine(repo, parent_patch_id).remove_dead_code(report, min_confidence=min_confidence, symbols=wanted))

    def migrate(self, repo: RepositoryModel, rules: Sequence[Mapping[str, Any] | MigrationRule], *, parent_patch_id: Optional[str] = None) -> PatchResult:
        migrator = ApiMigrator(self.semantics.index(repo), self.builder, self.migration_options, cache=self.memory.models, parent_patch_id=parent_patch_id)
        return self._commit(migrator.migrate(parse_rules(rules)))

    def from_diff(self, repo: RepositoryModel, diff: str, *, intent: str = "apply externally supplied patch", parent_patch_id: Optional[str] = None) -> PatchResult:
        validator = regression_validator(self.semantics.index(repo), self.models.parser, cache=self.memory.models) if self.refactoring_options.regression_check else None
        return self._commit(self.builder.from_unified_diff(repo.snapshot, diff, intent=intent, validator=validator, parent_patch_id=parent_patch_id))

    # -- ledger -----------------------------------------------------------------------------------------
    def proposal(self, patch_id: str) -> PatchProposal:
        return self.memory.proposal(patch_id)

    def lineage(self, patch_id: str) -> tuple[LedgerEntry, ...]:
        return self.memory.lineage(patch_id)

    def recent(self, limit: int = 20) -> tuple[LedgerEntry, ...]:
        return self.memory.recent(limit)

    def stats(self) -> dict[str, Any]:
        return self.memory.stats()


__all__ = ["SoftwareRefactoring"]
