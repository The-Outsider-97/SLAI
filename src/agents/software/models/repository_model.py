"""Immutable repository snapshot and the module-level model built over it.

``RepositorySnapshot`` is *input data*: a content-addressed mapping of
repository-relative paths to :class:`SourceFile` objects.  The subsystem never
reads a working tree; callers (Reader/Execution side) hand over the files.

``RepositoryModel`` is *derived facts*: one :class:`SourceModel` per file, the
module/package index, and every import resolved against that index.  Building
is incremental: models are reused from a cache keyed by file digest, module
identity and parser signature, so a one-file change re-parses one file.
"""

from __future__ import annotations

__version__ = "2.4.0"

import threading

from collections.abc import Iterable, Iterator, Mapping, MutableMapping
from dataclasses import dataclass
from typing import Any, Optional, Protocol, Union, runtime_checkable

from .base_model import BaseSoftwareModel
from .source_model import SourceModel
from ..software_types import ImportRecord, ParseStatus, ResolvedImport, SourceFile
from ..utils.software_errors import SoftwareLimitError, SoftwareSnapshotError
from ..utils.software_helpers import (
    DEFAULT_TEST_PATTERNS,
    PYTHON_SUFFIXES,
    combine_digests,
    content_digest,
    is_stdlib_module,
    module_name_for_path,
    normalize_repo_path,
    normalize_source_roots,
    resolve_relative_import,
    top_level_name,
)
from logs.logger import get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Software Repository Model")

FileInput = Union[str, bytes, SourceFile]


@runtime_checkable
class SourceParser(Protocol):
    """What the repository model needs from a parser (see ``SoftwareParser``)."""

    signature: str

    def parse(self, source: SourceFile, *, module_name: str = "", is_package: bool = False, strict: bool = False) -> SourceModel:
        ...


@dataclass(frozen=True, slots=True)
class SnapshotLimits:
    max_files: int = 20_000
    max_total_bytes: int = 256 * 1024 * 1024


def _coerce_file(
    path: str,
    value: FileInput,
    *,
    test_patterns: tuple[str, ...],
    language_map: Optional[Mapping[str, str]],
) -> SourceFile:
    if isinstance(value, SourceFile):
        if value.path != path:
            raise SoftwareSnapshotError("SourceFile path does not match its snapshot key", context={"key": path, "path": value.path})
        return value
    if isinstance(value, (bytes, bytearray)):
        raw = bytes(value)
        if b"\x00" in raw:
            raise SoftwareSnapshotError("binary content is not a source file", context={"path": path})
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise SoftwareSnapshotError("source file is not valid UTF-8", context={"path": path}, cause=exc) from exc
    elif isinstance(value, str):
        text = value
    else:
        raise SoftwareSnapshotError("file content must be str, bytes or SourceFile", context={"path": path, "type": type(value).__name__})
    return SourceFile.from_text(path, text, language_map=language_map, test_patterns=test_patterns)


class RepositorySnapshot(BaseSoftwareModel):
    """Immutable, order-independent, content-addressed set of source files."""

    model_type = "snapshot"

    def __init__(
        self,
        files: Mapping[str, SourceFile],
        *,
        source_roots: Iterable[str] = ("",),
        test_patterns: tuple[str, ...] = DEFAULT_TEST_PATTERNS,
        language_map: Optional[Mapping[str, str]] = None,
        limits: Optional[SnapshotLimits] = None,
    ) -> None:
        self.limits = limits or SnapshotLimits()
        if len(files) > self.limits.max_files:
            raise SoftwareLimitError("snapshot has too many files", context={"files": len(files), "limit": self.limits.max_files})
        total = sum(item.size for item in files.values())
        if total > self.limits.max_total_bytes:
            raise SoftwareLimitError("snapshot is too large", context={"bytes": total, "limit": self.limits.max_total_bytes})
        self._files: dict[str, SourceFile] = {path: files[path] for path in sorted(files)}
        self.source_roots = normalize_source_roots(source_roots)
        self.test_patterns = tuple(test_patterns)
        self.language_map = dict(language_map) if language_map else None
        self.total_bytes = total
        self._digest = content_digest(
            combine_digests((path, item.digest) for path, item in self._files.items()) + "|" + "|".join(self.source_roots)
        )

    # -- construction --------------------------------------------------------
    @classmethod
    def from_files(
        cls,
        files: Mapping[str, FileInput],
        *,
        source_roots: Iterable[str] = ("",),
        test_patterns: tuple[str, ...] = DEFAULT_TEST_PATTERNS,
        language_map: Optional[Mapping[str, str]] = None,
        limits: Optional[SnapshotLimits] = None,
    ) -> "RepositorySnapshot":
        if not isinstance(files, Mapping):
            raise SoftwareSnapshotError("files must be a mapping of path -> content", context={"type": type(files).__name__})
        built: dict[str, SourceFile] = {}
        for raw_path, value in files.items():
            path = normalize_repo_path(raw_path)
            if path in built:
                raise SoftwareSnapshotError("two entries normalize to the same path", context={"path": path})
            built[path] = _coerce_file(path, value, test_patterns=tuple(test_patterns), language_map=language_map)
        return cls(built, source_roots=source_roots, test_patterns=tuple(test_patterns), language_map=language_map, limits=limits)

    def with_changes(self, changes: Mapping[str, Optional[FileInput]]) -> "RepositorySnapshot":
        """New snapshot with files added/replaced (content) or removed (``None``)."""

        files = dict(self._files)
        for raw_path, value in changes.items():
            path = normalize_repo_path(raw_path)
            if value is None:
                files.pop(path, None)
            else:
                files[path] = _coerce_file(path, value, test_patterns=self.test_patterns, language_map=self.language_map)
        return RepositorySnapshot(files, source_roots=self.source_roots, test_patterns=self.test_patterns, language_map=self.language_map, limits=self.limits)

    # -- access ----------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._files)

    def __contains__(self, path: object) -> bool:
        return path in self._files

    def __iter__(self) -> Iterator[str]:
        return iter(self._files)

    def get(self, path: str) -> Optional[SourceFile]:
        return self._files.get(path)

    def file(self, path: str) -> SourceFile:
        try:
            return self._files[path]
        except KeyError:
            raise SoftwareSnapshotError("file is not part of the snapshot", context={"path": path}) from None

    def paths(self) -> tuple[str, ...]:
        return tuple(self._files)

    def files(self) -> tuple[SourceFile, ...]:
        return tuple(self._files.values())

    def changed_paths(self, other: "RepositorySnapshot") -> tuple[str, ...]:
        """Paths added, removed or modified between ``other`` (older) and ``self``."""

        keys = set(self._files) | set(other._files)
        return tuple(
            sorted(
                path
                for path in keys
                if (self._files.get(path).digest if path in self._files else None)  # type: ignore[union-attr]
                != (other._files.get(path).digest if path in other._files else None)  # type: ignore[union-attr]
            )
        )

    # -- BaseSoftwareModel ---------------------------------------------------------
    @property
    def digest(self) -> str:
        return self._digest

    @property
    def model_id(self) -> str:
        return f"snapshot:{self._digest[:12]}"

    def summary(self) -> dict[str, Any]:
        languages: dict[str, int] = {}
        for item in self._files.values():
            languages[item.language] = languages.get(item.language, 0) + 1
        return {
            "files": len(self._files),
            "bytes": self.total_bytes,
            "languages": dict(sorted(languages.items())),
            "test_files": sum(1 for item in self._files.values() if item.is_test),
            "source_roots": list(self.source_roots),
            "digest": self._digest,
        }


class RepositoryModel(BaseSoftwareModel):
    """Per-file models plus module index and resolved imports for a snapshot."""

    model_type = "repository"

    def __init__(
        self,
        snapshot: RepositorySnapshot,
        models: Mapping[str, SourceModel],
        parser_signature: str,
    ) -> None:
        self.snapshot = snapshot
        self.parser_signature = parser_signature
        self._models: dict[str, SourceModel] = {path: models[path] for path in sorted(models)}
        self._lock = threading.Lock()

        modules: dict[str, str] = {}
        conflicts: dict[str, list[str]] = {}
        ranked = sorted(
            (path for path, item in self._models.items() if item.module_name or item.is_package),
            key=lambda path: (path.endswith(".pyi"), not self._models[path].is_package, path),  # regular package beats module, like the import system
        )
        for path in ranked:
            name = self._models[path].module_name
            if name in modules:
                conflicts.setdefault(name, [modules[name]]).append(path)
            else:
                modules[name] = path
        self._modules = modules
        self.module_conflicts: dict[str, tuple[str, ...]] = {name: tuple(paths) for name, paths in conflicts.items()}
        self._module_of_path = {path: name for name, path in modules.items()}

        packages: set[str] = set()
        for name, path in modules.items():
            parts = name.split(".")
            for end in range(1, len(parts)):
                packages.add(".".join(parts[:end]))
            if self._models[path].is_package:
                packages.add(name)
        self._packages = frozenset(packages)
        self._top_levels = frozenset(top_level_name(name) for name in modules if name)
        self._imports: dict[str, tuple[tuple[ImportRecord, ResolvedImport], ...]] = {
            path: tuple((record, self.resolve_import(record, item)) for record in item.imports)
            for path, item in self._models.items()
        }

    # -- construction --------------------------------------------------------
    @classmethod
    def build(
        cls,
        snapshot: RepositorySnapshot,
        parser: SourceParser,
        *,
        cache: Optional[MutableMapping[tuple[str, str, bool, str], SourceModel]] = None,
        previous: Optional["RepositoryModel"] = None,
    ) -> "RepositoryModel":
        """Parse every file, reusing identical results from ``previous``/``cache``."""

        signature = parser.signature
        reusable: dict[tuple[str, str, bool], SourceModel] = {}
        if previous is not None and previous.parser_signature == signature:
            reusable = {(item.digest, item.module_name, item.is_package): item for item in previous._models.values()}
        models: dict[str, SourceModel] = {}
        parsed = 0
        for path, source in snapshot._files.items():
            if source.language == "python" or path.endswith(PYTHON_SUFFIXES):
                name, is_package = module_name_for_path(path, snapshot.source_roots)
            else:
                name, is_package = "", False
            key = (source.digest, name, is_package)
            model = reusable.get(key) if previous is not None and previous.snapshot.get(path) is not None else None
            if model is not None and model.path != path:
                model = None
            if model is None and cache is not None:
                model = cache.get((source.digest, name, is_package, signature))  # type: ignore[call-arg]
                if model is not None and model.path != path:
                    model = None
            if model is None:
                model = parser.parse(source, module_name=name, is_package=is_package)
                parsed += 1
                if cache is not None:
                    cache[(source.digest, name, is_package, signature)] = model
            models[path] = model
        logger.debug("repository model built: %d files, %d parsed", len(models), parsed)
        result = cls(snapshot, models, signature)
        result.parsed_files = parsed
        return result

    parsed_files: int = 0

    def with_changes(
        self,
        changes: Mapping[str, Optional[FileInput]],
        parser: SourceParser,
        *,
        cache: Optional[MutableMapping[tuple[str, str, bool, str], SourceModel]] = None,
    ) -> "RepositoryModel":
        """Re-model the snapshot after ``changes``; unchanged files are reused."""

        return RepositoryModel.build(self.snapshot.with_changes(changes), parser, cache=cache, previous=self)

    # -- module index ----------------------------------------------------------
    @property
    def source_roots(self) -> tuple[str, ...]:
        return self.snapshot.source_roots

    def paths(self) -> tuple[str, ...]:
        return tuple(self._models)

    def models(self) -> tuple[SourceModel, ...]:
        return tuple(self._models.values())

    def model(self, path: str) -> SourceModel:
        try:
            return self._models[path]
        except KeyError:
            raise SoftwareSnapshotError("file is not part of the repository model", context={"path": path}) from None

    def get(self, path: str) -> Optional[SourceModel]:
        return self._models.get(path)

    def module(self, name: str) -> Optional[SourceModel]:
        path = self._modules.get(name)
        return self._models[path] if path else None

    def module_path(self, name: str) -> Optional[str]:
        return self._modules.get(name)

    def module_name_of(self, path: str) -> Optional[str]:
        return self._module_of_path.get(path)

    def module_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._modules))

    def packages(self) -> tuple[str, ...]:
        return tuple(sorted(self._packages))

    def is_package(self, name: str) -> bool:
        return name in self._packages

    def is_internal(self, dotted: str) -> bool:
        return dotted in self._modules or dotted in self._packages

    def is_internal_top_level(self, name: str) -> bool:
        return name in self._top_levels

    def parsed(self) -> tuple[SourceModel, ...]:
        return tuple(item for item in self._models.values() if item.status is ParseStatus.OK)

    def failed(self) -> tuple[SourceModel, ...]:
        return tuple(item for item in self._models.values() if item.status in (ParseStatus.SYNTAX_ERROR, ParseStatus.TOO_LARGE))

    def test_models(self) -> tuple[SourceModel, ...]:
        return tuple(item for item in self._models.values() if item.source.is_test)

    # -- import resolution -------------------------------------------------------
    def absolute_module(self, record: ImportRecord, current: SourceModel) -> Optional[str]:
        """Absolute dotted module named by the import statement (``None`` if invalid)."""

        if record.level:
            return resolve_relative_import(current.module_name, current.is_package, record.level, record.module)
        return record.module or None

    def resolve_import(self, record: ImportRecord, current: SourceModel) -> ResolvedImport:
        """Resolve one import against the repository.

        ``internal=True`` with ``module=None`` denotes an import that targets a
        repository package but names something the snapshot does not contain.
        """

        base = self.absolute_module(record, current)
        if base is None:
            return ResolvedImport(None, None, None, False, False, None)
        top = top_level_name(base)
        internal = bool(record.level) or top in self._top_levels
        if internal:
            if record.is_from:
                if not self.is_internal(base):
                    return ResolvedImport(None, None, record.name, True, False, None)
                if record.name and record.name != "*" and self.is_internal(f"{base}.{record.name}"):
                    return ResolvedImport(base, f"{base}.{record.name}", None, True, False, None)
                return ResolvedImport(base, None, record.name, True, False, None)
            if self.is_internal(base):
                return ResolvedImport(base, None, None, True, False, None)
            # ``import a.b.c`` where only ``a.b`` exists: bind to the longest known prefix.
            return ResolvedImport(None, None, None, True, False, None)
        stdlib = is_stdlib_module(base)
        return ResolvedImport(base, None, record.name if record.is_from else None, False, stdlib, None if stdlib else top)

    def imports_of(self, path: str) -> tuple[tuple[ImportRecord, ResolvedImport], ...]:
        return self._imports.get(path, ())

    def all_imports(self) -> Iterator[tuple[str, ImportRecord, ResolvedImport]]:
        for path, items in self._imports.items():
            for record, resolved in items:
                yield path, record, resolved

    def external_packages(self) -> dict[str, tuple[str, ...]]:
        """Third-party top-level packages → sorted importing paths."""

        found: dict[str, set[str]] = {}
        for path, _, resolved in self.all_imports():
            if resolved.external_package:
                found.setdefault(resolved.external_package, set()).add(path)
        return {name: tuple(sorted(paths)) for name, paths in sorted(found.items())}

    # -- BaseSoftwareModel ---------------------------------------------------------
    @property
    def digest(self) -> str:
        return content_digest(f"{self.snapshot.digest}|{self.parser_signature}")

    @property
    def model_id(self) -> str:
        return f"repository:{self.digest[:12]}"

    def summary(self) -> dict[str, Any]:
        statuses: dict[str, int] = {}
        for item in self._models.values():
            statuses[item.status.value] = statuses.get(item.status.value, 0) + 1
        return {
            **self.snapshot.summary(),
            "modules": len(self._modules),
            "packages": len(self._packages),
            "symbols": sum(len(item.symbols) for item in self._models.values()),
            "parse_status": dict(sorted(statuses.items())),
            "module_conflicts": {name: list(paths) for name, paths in self.module_conflicts.items()},
            "parser": self.parser_signature,
        }


__all__ = [
    "FileInput",
    "RepositoryModel",
    "RepositorySnapshot",
    "SnapshotLimits",
    "SourceParser",
]
