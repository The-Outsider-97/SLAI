"""Typed contracts and formal vocabulary for the SLAI Software subsystem.

The records in this module describe *facts about software artifacts* (symbols,
scopes, references, calls) and *derived reports* (findings, impact, API
compatibility, patch proposals).  They do not encode policy: whether a finding
blocks a change, or whether a patch may be applied, is decided downstream by
Verification, Safety and Execution.

Design notes
------------
* Spans use 1-based lines and 0-based **character** columns (``ast`` byte
  columns are converted at parse time), so edits and user-facing locations agree
  on non-ASCII sources.
* Every record is an immutable dataclass with a deterministic ``to_dict``.
* A :class:`PatchProposal` is evidence of an intended change.  It is never
  applied by the Software subsystem; ``applied`` is always ``False`` here.
"""

from __future__ import annotations

__version__ = "2.4.0"

import dataclasses

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from .utils.software_helpers import (
    DEFAULT_TEST_PATTERNS,
    content_digest,
    is_test_path,
    language_for_path,
    to_plain,
)
from logs.logger import get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Software Types")


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class SymbolKind(str, Enum):
    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    VARIABLE = "variable"
    PARAMETER = "parameter"


class ScopeKind(str, Enum):
    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    LAMBDA = "lambda"
    COMPREHENSION = "comprehension"


class BindingKind(str, Enum):
    PARAMETER = "parameter"
    VARIABLE = "variable"
    FUNCTION = "function"
    CLASS = "class"
    IMPORT = "import"


class Resolution(str, Enum):
    """How a name occurrence was resolved lexically (LEGB)."""

    LOCAL = "local"
    ENCLOSING = "enclosing"
    GLOBAL = "global"
    BUILTIN = "builtin"
    UNRESOLVED = "unresolved"


class EdgeKind(str, Enum):
    IMPORTS = "imports"
    CALLS = "calls"
    REFERENCES = "references"
    INHERITS = "inherits"
    DECORATES = "decorates"


class ParameterKind(str, Enum):
    POSITIONAL_ONLY = "positional_only"
    POSITIONAL_OR_KEYWORD = "positional_or_keyword"
    VAR_POSITIONAL = "var_positional"
    KEYWORD_ONLY = "keyword_only"
    VAR_KEYWORD = "var_keyword"


class ParseStatus(str, Enum):
    OK = "ok"
    SYNTAX_ERROR = "syntax_error"
    TOO_LARGE = "too_large"
    UNSUPPORTED = "unsupported"


class Severity(str, Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

    @property
    def rank(self) -> int:
        return ("info", "low", "medium", "high").index(self.value)


class FindingCategory(str, Enum):
    CORRECTNESS = "correctness"
    MAINTAINABILITY = "maintainability"
    COMPLEXITY = "complexity"
    SECURITY = "security"
    STYLE = "style"
    DEAD_CODE = "dead_code"


class CompatibilityLevel(str, Enum):
    BREAKING = "breaking"
    ADDITIVE = "additive"
    COMPATIBLE = "compatible"


class VersionBump(str, Enum):
    MAJOR = "major"
    MINOR = "minor"
    PATCH = "patch"
    NONE = "none"


class RefactoringKind(str, Enum):
    RENAME_SYMBOL = "rename_symbol"
    REMOVE_UNUSED_IMPORTS = "remove_unused_imports"
    REMOVE_DEAD_CODE = "remove_dead_code"
    MIGRATE_API = "migrate_api"
    EXTERNAL_DIFF = "external_diff"


class MigrationKind(str, Enum):
    RENAME_SYMBOL = "rename_symbol"
    RENAME_MODULE = "rename_module"
    RENAME_KEYWORD = "rename_keyword"
    REMOVE_KEYWORD = "remove_keyword"
    ADD_KEYWORD = "add_keyword"


class FileOperation(str, Enum):
    MODIFY = "modify"
    ADD = "add"
    DELETE = "delete"


#: Pipeline stages that must judge a proposal before it may take effect.  The
#: Software subsystem stops *before* all of them.
HANDOFF_GATES: tuple[str, ...] = ("verification", "safety", "execution", "evaluation")

#: Pseudo root name of an attribute chain whose receiver is ``super()`` (``super().m`` → ``("super()", "m")``).
SUPER_ROOT = "super()"


class Serializable:
    """Mixin giving frozen dataclasses a deterministic, JSON-safe ``to_dict``."""

    __slots__ = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            item.name: to_plain(getattr(self, item.name))
            for item in dataclasses.fields(self)  # type: ignore[arg-type]
        }


# ---------------------------------------------------------------------------
# Source facts
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class SourceSpan(Serializable):
    start_line: int
    start_col: int
    end_line: int
    end_col: int

    @property
    def line_count(self) -> int:
        return max(1, self.end_line - self.start_line + 1)

    def contains_line(self, line: int) -> bool:
        return self.start_line <= line <= self.end_line

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.start_line, self.start_col, self.end_line, self.end_col)

    def __str__(self) -> str:
        return f"{self.start_line}:{self.start_col}-{self.end_line}:{self.end_col}"


@dataclass(frozen=True, slots=True)
class SourceFile(Serializable):
    """One file of a repository snapshot.  ``text`` is never mutated."""

    path: str
    text: str
    language: str
    digest: str
    size: int
    is_test: bool = False

    @classmethod
    def from_text(
        cls,
        path: str,
        text: str,
        *,
        language: Optional[str] = None,
        is_test: Optional[bool] = None,
        language_map: Optional[Mapping[str, str]] = None,
        test_patterns: tuple[str, ...] = DEFAULT_TEST_PATTERNS,
    ) -> "SourceFile":
        return cls(
            path=path,
            text=text,
            language=language or language_for_path(path, language_map),
            digest=content_digest(text),
            size=len(text.encode("utf-8", errors="surrogatepass")),
            is_test=is_test_path(path, test_patterns) if is_test is None else bool(is_test),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "language": self.language,
            "digest": self.digest,
            "size": self.size,
            "is_test": self.is_test,
        }


@dataclass(frozen=True, slots=True)
class ParameterInfo(Serializable):
    name: str
    kind: ParameterKind
    has_default: bool = False
    annotation: Optional[str] = None
    default: Optional[str] = None


@dataclass(frozen=True, slots=True)
class SignatureInfo(Serializable):
    parameters: tuple[ParameterInfo, ...] = ()
    returns: Optional[str] = None
    is_async: bool = False

    def parameter(self, name: str) -> Optional[ParameterInfo]:
        return next((item for item in self.parameters if item.name == name), None)

    @property
    def positional(self) -> tuple[ParameterInfo, ...]:
        return tuple(
            item for item in self.parameters
            if item.kind in (ParameterKind.POSITIONAL_ONLY, ParameterKind.POSITIONAL_OR_KEYWORD)
        )

    @property
    def keyword_names(self) -> frozenset[str]:
        return frozenset(
            item.name for item in self.parameters
            if item.kind in (ParameterKind.POSITIONAL_OR_KEYWORD, ParameterKind.KEYWORD_ONLY)
        )

    @property
    def accepts_var_args(self) -> bool:
        return any(item.kind is ParameterKind.VAR_POSITIONAL for item in self.parameters)

    @property
    def accepts_var_kwargs(self) -> bool:
        return any(item.kind is ParameterKind.VAR_KEYWORD for item in self.parameters)

    @property
    def required(self) -> tuple[ParameterInfo, ...]:
        return tuple(
            item for item in self.parameters
            if not item.has_default
            and item.kind not in (ParameterKind.VAR_POSITIONAL, ParameterKind.VAR_KEYWORD)
        )


@dataclass(frozen=True, slots=True)
class FunctionMetrics(Serializable):
    cyclomatic: int = 1
    cognitive: int = 0
    lines: int = 0
    statements: int = 0
    parameters: int = 0
    returns: int = 0
    max_nesting: int = 0


@dataclass(frozen=True, slots=True)
class ImportRecord(Serializable):
    """One imported name.  ``import a, b`` yields two records.

    ``span`` covers the whole alias node (``name`` or ``name as alias``);
    ``name_span`` the imported name; ``alias_span`` the ``as`` target;
    ``module_span`` the module part of ``from <module> import`` statements.
    """

    module: Optional[str]
    name: Optional[str]
    alias: Optional[str]
    level: int
    is_from: bool
    scope: str
    span: SourceSpan
    name_span: SourceSpan
    statement_span: SourceSpan
    alias_span: Optional[SourceSpan] = None
    module_span: Optional[SourceSpan] = None
    type_checking: bool = False
    lazy: bool = False
    conditional: bool = False

    @property
    def is_star(self) -> bool:
        return self.is_from and self.name == "*"

    @property
    def bound_name(self) -> Optional[str]:
        """Local name introduced by this import (``None`` for ``import *``)."""

        if self.is_star:
            return None
        if self.alias:
            return self.alias
        if self.is_from:
            return self.name
        return (self.module or "").split(".", 1)[0] or None


@dataclass(frozen=True, slots=True)
class ResolvedImport(Serializable):
    """Repository-level resolution of one :class:`ImportRecord`."""

    module: Optional[str]
    submodule: Optional[str]
    attribute: Optional[str]
    internal: bool
    stdlib: bool
    external_package: Optional[str]

    @property
    def target_module(self) -> Optional[str]:
        """Most specific module this import loads (submodule when imported by name)."""

        return self.submodule or self.module


@dataclass(frozen=True, slots=True)
class ParseDiagnostic(Serializable):
    code: str
    message: str
    line: int = 0
    column: int = 0
    severity: Severity = Severity.HIGH


@dataclass(frozen=True, slots=True)
class BindingInfo(Serializable):
    name: str
    scope: str
    kind: BindingKind
    origins: tuple[str, ...]
    span: SourceSpan
    assignment_count: int = 1
    symbol: Optional[str] = None


@dataclass(frozen=True, slots=True)
class ScopeInfo(Serializable):
    qualified_name: str
    kind: ScopeKind
    parent: Optional[str]
    span: SourceSpan
    bindings: tuple[BindingInfo, ...] = ()
    global_names: tuple[str, ...] = ()
    nonlocal_names: tuple[str, ...] = ()
    dynamic: bool = False
    symbol: Optional[str] = None  # qualified symbol name for function/class scopes

    def binding(self, name: str) -> Optional[BindingInfo]:
        return next((item for item in self.bindings if item.name == name), None)


@dataclass(frozen=True, slots=True)
class NameUse(Serializable):
    name: str
    span: SourceSpan
    context: str
    scope: str
    binding_scope: Optional[str]
    resolution: Resolution


@dataclass(frozen=True, slots=True)
class AttributeUse(Serializable):
    """One ``.attr`` access.  ``chain`` is root-name plus attributes up to this one."""

    chain: tuple[str, ...]
    span: SourceSpan
    context: str
    scope: str
    rooted: bool
    root_scope: Optional[str]
    root_resolution: Resolution


@dataclass(frozen=True, slots=True)
class CallSite(Serializable):
    callee: tuple[str, ...]
    callee_span: SourceSpan
    span: SourceSpan
    scope: str
    positional: int
    keywords: tuple[str, ...]
    keyword_spans: tuple[SourceSpan, ...]
    star_args: bool = False
    star_kwargs: bool = False
    awaited: bool = False
    root_scope: Optional[str] = None
    root_resolution: Resolution = Resolution.UNRESOLVED


@dataclass(frozen=True, slots=True)
class SymbolRecord(Serializable):
    qualified_name: str
    name: str
    kind: SymbolKind
    parent: Optional[str]
    span: SourceSpan
    name_span: SourceSpan
    full_span: SourceSpan
    decorators: tuple[str, ...] = ()
    bases: tuple[str, ...] = ()
    signature: Optional[SignatureInfo] = None
    annotation: Optional[str] = None
    is_public: bool = True
    is_async: bool = False
    is_abstract: bool = False
    has_docstring: bool = False
    body_digest: str = ""
    signature_digest: str = ""
    metrics: Optional[FunctionMetrics] = None

    @property
    def is_callable(self) -> bool:
        return self.kind in (SymbolKind.FUNCTION, SymbolKind.METHOD, SymbolKind.CLASS)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Finding(Serializable):
    rule_id: str
    name: str
    category: FindingCategory
    severity: Severity
    message: str
    path: str
    span: Optional[SourceSpan] = None
    symbol: Optional[str] = None
    confidence: float = 1.0
    suggestion: Optional[str] = None
    evidence: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class StaticAnalysisReport(Serializable):
    findings: tuple[Finding, ...]
    files_analyzed: int
    summary: Mapping[str, Any]
    function_metrics: Mapping[str, FunctionMetrics] = field(default_factory=dict)
    truncated: int = 0


@dataclass(frozen=True, slots=True)
class DeadCodeItem(Serializable):
    qualified_name: str
    kind: SymbolKind
    path: str
    span: SourceSpan
    reason: str
    confidence: float
    caveats: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DeadCodeReport(Serializable):
    items: tuple[DeadCodeItem, ...]
    entry_points: tuple[str, ...]
    live_symbols: int
    total_symbols: int
    truncated: int = 0


@dataclass(frozen=True, slots=True)
class ImpactedSymbol(Serializable):
    qualified_name: str
    kind: SymbolKind
    path: str
    distance: int
    via: tuple[str, ...]
    confidence: float


@dataclass(frozen=True, slots=True)
class TestCandidate(Serializable):
    __test__ = False  # not a pytest test class

    path: str
    qualified_name: Optional[str]
    score: float
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TestSelection(Serializable):
    __test__ = False  # not a pytest test class

    selected: tuple[TestCandidate, ...]
    total_tests: int
    changed_modules: tuple[str, ...]
    fallback_to_all: bool = False


@dataclass(frozen=True, slots=True)
class ImpactReport(Serializable):
    changed_symbols: tuple[str, ...]
    changed_modules: tuple[str, ...]
    impacted_symbols: tuple[ImpactedSymbol, ...]
    impacted_modules: tuple[str, ...]
    tests: TestSelection
    risk_score: float
    risk_level: str
    risk_factors: Mapping[str, float]
    notes: tuple[str, ...] = ()
    truncated: int = 0


@dataclass(frozen=True, slots=True)
class CompatibilityChange(Serializable):
    symbol: str
    kind: str
    level: CompatibilityLevel
    detail: str
    path: Optional[str] = None
    before: Optional[str] = None
    after: Optional[str] = None


@dataclass(frozen=True, slots=True)
class CompatibilityReport(Serializable):
    changes: tuple[CompatibilityChange, ...]
    recommended_bump: VersionBump
    breaking: int
    additive: int
    compatible: int
    public_symbols_before: int
    public_symbols_after: int


@dataclass(frozen=True, slots=True)
class PackageMetrics(Serializable):
    """Martin's package metrics (afferent/efferent coupling, instability, abstractness)."""

    package: str
    modules: int
    afferent: int
    efferent: int
    instability: float
    abstractness: float
    distance: float


@dataclass(frozen=True, slots=True)
class LayerViolation(Serializable):
    source: str
    target: str
    source_layer: str
    target_layer: str
    rule: str
    path: Optional[str] = None
    line: Optional[int] = None


@dataclass(frozen=True, slots=True)
class ArchitectureReport(Serializable):
    packages: tuple[PackageMetrics, ...]
    module_cycles: tuple[tuple[str, ...], ...]
    package_cycles: tuple[tuple[str, ...], ...]
    layer_violations: tuple[LayerViolation, ...]
    hubs: tuple[tuple[str, int], ...]
    orphans: tuple[str, ...]
    layers: tuple[tuple[str, ...], ...]
    external_dependencies: Mapping[str, int]
    summary: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ClassNode(Serializable):
    qualified_name: str
    path: str
    bases: tuple[str, ...]
    unresolved_bases: tuple[str, ...]
    linearization: tuple[str, ...]  # C3 method-resolution order (not named ``mro``: that shadows type.mro)
    subclasses: tuple[str, ...]
    methods: tuple[str, ...]
    overrides: tuple[str, ...]
    is_abstract: bool
    depth: int
    mro_conflict: bool = False


@dataclass(frozen=True, slots=True)
class HierarchyReport(Serializable):
    classes: tuple[ClassNode, ...]
    roots: tuple[str, ...]
    cycles: tuple[tuple[str, ...], ...]
    external_bases: Mapping[str, int]
    max_depth: int


@dataclass(frozen=True, slots=True)
class SearchQuery(Serializable):
    text: Optional[str] = None
    mode: str = "substring"
    kinds: tuple[SymbolKind, ...] = ()
    decorator: Optional[str] = None
    base_class: Optional[str] = None
    calls: Optional[str] = None
    path_glob: Optional[str] = None
    include_private: bool = True
    limit: int = 100


@dataclass(frozen=True, slots=True)
class SearchHit(Serializable):
    qualified_name: str
    kind: SymbolKind
    path: str
    span: SourceSpan
    score: float
    snippet: Optional[str] = None


@dataclass(frozen=True, slots=True)
class SymbolReference(Serializable):
    """One occurrence that refers to a symbol.

    ``kind`` is one of ``name`` | ``attribute`` | ``call`` | ``import`` |
    ``export``.  ``exact`` is ``False`` for
    name-based (heuristic) attribute matches that could not be bound by scope.
    """

    target: str
    path: str
    span: SourceSpan
    kind: str
    scope: str
    context: str = "load"
    exact: bool = True


# ---------------------------------------------------------------------------
# Patch contracts
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class TextEdit(Serializable):
    """Replace ``[start, end)`` of the base text with ``replacement``."""

    path: str
    start: int
    end: int
    replacement: str
    description: str = ""


@dataclass(frozen=True, slots=True)
class FileChange(Serializable):
    path: str
    operation: FileOperation
    base_digest: Optional[str]
    new_digest: Optional[str]
    edits: tuple[TextEdit, ...] = ()
    new_text: Optional[str] = None
    diff: str = ""


@dataclass(frozen=True, slots=True)
class PatchProposal(Serializable):
    """Evidence of an intended change.  Never applied by the Software subsystem."""

    patch_id: str
    kind: RefactoringKind
    intent: str
    base_snapshot_digest: str
    result_snapshot_digest: str
    files: tuple[FileChange, ...]
    diff: str
    reverse_diff: str
    stats: Mapping[str, int]
    validation: Mapping[str, Any]
    risks: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    unresolved_sites: int = 0
    confidence: float = 1.0
    parent_patch_id: Optional[str] = None
    requires_gates: tuple[str, ...] = HANDOFF_GATES
    applied: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def changed_paths(self) -> tuple[str, ...]:
        return tuple(item.path for item in self.files)

    def summary(self) -> dict[str, Any]:
        """Compact form without diff bodies or edit payloads."""

        return {
            "patch_id": self.patch_id,
            "kind": self.kind.value,
            "intent": self.intent,
            "base_snapshot_digest": self.base_snapshot_digest,
            "result_snapshot_digest": self.result_snapshot_digest,
            "files": [item.path for item in self.files],
            "stats": dict(self.stats),
            "validation": to_plain(self.validation),
            "risks": list(self.risks),
            "warnings": list(self.warnings),
            "unresolved_sites": self.unresolved_sites,
            "confidence": self.confidence,
            "parent_patch_id": self.parent_patch_id,
            "requires_gates": list(self.requires_gates),
            "applied": self.applied,
        }


@dataclass(frozen=True, slots=True)
class MigrationRule(Serializable):
    """Declarative API migration rule.

    ``old``/``new`` are dotted qualified names (symbols or modules).  Keyword
    rules apply to resolved calls of ``old``; ``keyword`` is the argument name
    and ``new`` its replacement name (rename) or ``value`` the source text of a
    default to add.
    """

    kind: MigrationKind
    old: str
    new: Optional[str] = None
    keyword: Optional[str] = None
    value: Optional[str] = None
    description: str = ""


__all__ = [
    "AttributeUse",
    "ArchitectureReport",
    "BindingInfo",
    "BindingKind",
    "CallSite",
    "ClassNode",
    "CompatibilityChange",
    "CompatibilityLevel",
    "CompatibilityReport",
    "DeadCodeItem",
    "DeadCodeReport",
    "EdgeKind",
    "FileChange",
    "FileOperation",
    "Finding",
    "FindingCategory",
    "FunctionMetrics",
    "HANDOFF_GATES",
    "SUPER_ROOT",
    "HierarchyReport",
    "ImpactReport",
    "ImpactedSymbol",
    "ImportRecord",
    "LayerViolation",
    "MigrationKind",
    "MigrationRule",
    "NameUse",
    "PackageMetrics",
    "ParameterInfo",
    "ParameterKind",
    "ParseDiagnostic",
    "ParseStatus",
    "PatchProposal",
    "RefactoringKind",
    "Resolution",
    "ResolvedImport",
    "ScopeInfo",
    "ScopeKind",
    "SearchHit",
    "SearchQuery",
    "SymbolReference",
    "Serializable",
    "Severity",
    "SignatureInfo",
    "SourceFile",
    "SourceSpan",
    "StaticAnalysisReport",
    "SymbolKind",
    "SymbolRecord",
    "TestCandidate",
    "TestSelection",
    "TextEdit",
    "VersionBump",
]
