"""Central handler for ``models/``: snapshots, parsing and repository models."""

from __future__ import annotations

__version__ = "2.4.0"

from collections.abc import Iterable, Mapping
from typing import Any, Optional

from .models import FileInput, RepositoryModel, RepositorySnapshot, SnapshotLimits, SourceModel
from .modules.parser import ParserLimits, SoftwareParser
from .software_memory import SoftwareMemory
from .software_types import SourceFile
from .utils.config_loader import get_config_section
from .utils.software_errors import SoftwareSnapshotError
from .utils.software_helpers import DEFAULT_TEST_PATTERNS, opt_int, opt_strings
from logs.logger import get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Software Models")


class SoftwareModels:
    """Owns subsystem configuration, the private memory and the parser registry.

    Higher handlers (graphs → semantics → analysis → refactoring) are built on
    top of this object and reuse its memory; callers never touch either
    directly.  ``overrides`` (section name → mapping) exist for tests only.
    """

    def __init__(self, overrides: Optional[Mapping[str, Mapping[str, Any]]] = None) -> None:
        self._overrides = {key: dict(value) for key, value in (overrides or {}).items()}
        self.memory = SoftwareMemory(self.section("memory"))
        self.parser = SoftwareParser(self.section("parser"))
        data = self.section("repository")
        self.snapshot_limits = SnapshotLimits(
            max_files=opt_int(data, "max_files", 20_000, minimum=1),
            max_total_bytes=opt_int(data, "max_total_bytes", 256 * 1024 * 1024, minimum=1024),
        )
        self.source_roots = tuple(opt_strings(data, "source_roots", (".",)))
        self.test_patterns = tuple(opt_strings(data, "test_patterns", DEFAULT_TEST_PATTERNS))
        self.language_map = dict(data.get("language_map", {}) or {}) or None

    # -- configuration -----------------------------------------------------------------------
    def section(self, name: str) -> dict[str, Any]:
        merged = dict(get_config_section(name) or {})
        merged.update(self._overrides.get(name, {}))
        return merged

    # -- snapshots --------------------------------------------------------------------------------
    def snapshot(
        self,
        files: Optional[Mapping[str, FileInput]] = None,
        *,
        digest: Optional[str] = None,
        patch_id: Optional[str] = None,
        changes: Optional[Mapping[str, Optional[FileInput]]] = None,
        source_roots: Optional[Iterable[str]] = None,
    ) -> RepositorySnapshot:
        """Resolve a repository state from files, a remembered digest, or a prior proposal's result."""

        given = [x for x in (files is not None, digest is not None, patch_id is not None) if x]
        if len(given) != 1:
            raise SoftwareSnapshotError("exactly one of files, snapshot digest or base patch id is required", context={"given": len(given)})
        if files is not None:
            snap = RepositorySnapshot.from_files(
                files,
                source_roots=source_roots or self.source_roots,
                test_patterns=self.test_patterns,
                language_map=self.language_map,
                limits=self.snapshot_limits,
            )
        elif digest is not None:
            snap = self.memory.snapshot(digest)
        else:
            snap = self.memory.result_snapshot(str(patch_id))
        if changes:
            snap = snap.with_changes(changes)
        return self.memory.remember_snapshot(snap)

    def repository(self, snapshot: RepositorySnapshot, *, previous: Optional[RepositoryModel] = None) -> RepositoryModel:
        key = (snapshot.digest, self.parser.signature)
        found = self.memory.repositories.get(key)
        if found is not None:
            return found
        if previous is None:
            previous = self._nearest(snapshot)
        model = RepositoryModel.build(snapshot, self.parser, cache=self.memory.models, previous=previous)
        self.memory.repositories[key] = model
        return model

    def _nearest(self, snapshot: RepositorySnapshot) -> Optional[RepositoryModel]:
        """Any cached repository model with the same roots, to maximize incremental reuse."""

        best: Optional[RepositoryModel] = None
        best_changes = 10**9
        for key in list(self.memory.repositories):
            candidate = self.memory.repositories.get(key)
            if candidate is None or candidate.source_roots != snapshot.source_roots:
                continue
            changed = len(snapshot.changed_paths(candidate.snapshot))
            if changed < best_changes:
                best, best_changes = candidate, changed
        return best

    def parse(self, path: str, text: str) -> SourceModel:
        """Parse a single file outside any repository (no import resolution)."""

        source = SourceFile.from_text(path, text, test_patterns=self.test_patterns, language_map=self.language_map)
        from .utils.software_helpers import module_name_for_path, PYTHON_SUFFIXES

        name, is_package = module_name_for_path(path, self.source_roots) if path.endswith(PYTHON_SUFFIXES) else ("", False)
        return self.parser.parse(source, module_name=name, is_package=is_package)

    def languages(self) -> tuple[str, ...]:
        return self.parser.languages()


__all__ = ["SoftwareModels"]
