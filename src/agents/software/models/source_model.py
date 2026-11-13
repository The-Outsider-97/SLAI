"""Program model of a single source file.

``SourceModel`` is the output of a language parser: symbols, lexical scopes,
bindings, imports, name/attribute uses and call sites, all with character-exact
spans.  It is a *syntactic-lexical* model; cross-file meaning (what a name
ultimately refers to) is resolved later by :mod:`..semantics.symbol_index`.
"""

from __future__ import annotations

__version__ = "2.4.0"

import ast
import threading

from typing import Any, Optional

from .base_model import BaseSoftwareModel
from ..software_types import (
    AttributeUse,
    CallSite,
    ImportRecord,
    NameUse,
    ParseDiagnostic,
    ParseStatus,
    ScopeInfo,
    ScopeKind,
    SourceFile,
    SourceSpan,
    SymbolKind,
    SymbolRecord,
)
from ..utils.software_errors import SoftwareParseError
from ..utils.software_helpers import LineIndex
from logs.logger import get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Software Source Model")

PARSER_SCHEMA = "python-ast-v2"


class SourceModel(BaseSoftwareModel):
    """Immutable facts extracted from one source file."""

    model_type = "source"

    def __init__(
        self,
        *,
        source: SourceFile,
        module_name: str,
        is_package: bool,
        status: ParseStatus,
        diagnostics: tuple[ParseDiagnostic, ...] = (),
        symbols: tuple[SymbolRecord, ...] = (),
        scopes: tuple[ScopeInfo, ...] = (),
        imports: tuple[ImportRecord, ...] = (),
        names: tuple[NameUse, ...] = (),
        attributes: tuple[AttributeUse, ...] = (),
        calls: tuple[CallSite, ...] = (),
        exports: Optional[tuple[str, ...]] = None,
        export_spans: tuple[tuple[str, SourceSpan], ...] = (),
        string_names: frozenset[str] = frozenset(),
        line_count: int = 0,
        tree: Optional[ast.Module] = None,
    ) -> None:
        self.source = source
        self.module_name = module_name
        self.is_package = is_package
        self.status = status
        self.diagnostics = diagnostics
        self.symbols = symbols
        self.scopes = scopes
        self.imports = imports
        self.names = names
        self.attributes = attributes
        self.calls = calls
        self.exports = exports
        self.export_spans = export_spans
        self.string_names = string_names
        self.line_count = line_count
        self._tree = tree
        self._lock = threading.Lock()
        self._line_index: Optional[LineIndex] = None
        self._by_qname: Optional[dict[str, SymbolRecord]] = None
        self._children: Optional[dict[str, list[SymbolRecord]]] = None
        self._scope_by_name: Optional[dict[str, ScopeInfo]] = None

    # -- identity ----------------------------------------------------------
    @property
    def path(self) -> str:
        return self.source.path

    @property
    def digest(self) -> str:
        return self.source.digest

    @property
    def model_id(self) -> str:
        return self.source.path

    @property
    def ok(self) -> bool:
        return self.status is ParseStatus.OK

    @property
    def language(self) -> str:
        return self.source.language

    @property
    def is_test(self) -> bool:
        return self.source.is_test

    # -- lazy indexes ------------------------------------------------------
    @property
    def line_index(self) -> LineIndex:
        with self._lock:
            if self._line_index is None:
                self._line_index = LineIndex(self.source.text)
            return self._line_index

    def _ensure_indexes(self) -> None:
        with self._lock:
            if self._by_qname is not None:
                return
            by_qname: dict[str, SymbolRecord] = {}
            children: dict[str, list[SymbolRecord]] = {}
            for symbol in self.symbols:
                by_qname.setdefault(symbol.qualified_name, symbol)
                if symbol.parent:
                    children.setdefault(symbol.parent, []).append(symbol)
            self._children = children
            self._scope_by_name = {scope.qualified_name: scope for scope in self.scopes}
            self._by_qname = by_qname

    def symbol(self, qualified_name: str) -> Optional[SymbolRecord]:
        self._ensure_indexes()
        return self._by_qname.get(qualified_name)  # type: ignore[union-attr]

    def children(self, parent: str) -> tuple[SymbolRecord, ...]:
        self._ensure_indexes()
        return tuple(self._children.get(parent, ()))  # type: ignore[union-attr]

    def scope(self, qualified_name: str) -> Optional[ScopeInfo]:
        self._ensure_indexes()
        return self._scope_by_name.get(qualified_name)  # type: ignore[union-attr]

    @property
    def module_scope(self) -> Optional[ScopeInfo]:
        return next((scope for scope in self.scopes if scope.kind is ScopeKind.MODULE), None)

    def duplicate_definitions(self) -> frozenset[str]:
        counts: dict[str, int] = {}
        for symbol in self.symbols:
            counts[symbol.qualified_name] = counts.get(symbol.qualified_name, 0) + 1
        return frozenset(name for name, count in counts.items() if count > 1)

    def owner_symbol(self, scope_name: str) -> str:
        """Qualified name of the symbol that executes code in ``scope_name``.

        Lambdas and comprehensions belong to their nearest enclosing function,
        class or module; module-level code belongs to the module itself.
        """

        self._ensure_indexes()
        scopes = self._scope_by_name or {}
        current: Optional[ScopeInfo] = scopes.get(scope_name)
        while current is not None and current.kind in (ScopeKind.LAMBDA, ScopeKind.COMPREHENSION):
            current = scopes.get(current.parent) if current.parent else None
        if current is None:
            return self.module_name
        return current.symbol or current.qualified_name

    def top_level_symbols(self) -> tuple[SymbolRecord, ...]:
        return self.children(self.module_name)

    @property
    def has_star_import(self) -> bool:
        return any(item.is_star for item in self.imports)

    def public_names(self) -> frozenset[str]:
        """Names a ``from module import *`` would expose."""

        if self.exports is not None:
            return frozenset(self.exports)
        return frozenset(
            symbol.name for symbol in self.top_level_symbols() if symbol.is_public and not symbol.name.startswith("_")
        )

    # -- source access -----------------------------------------------------
    def tree(self) -> ast.Module:
        """The module AST (re-parsed on demand unless retained)."""

        if self._tree is not None:
            return self._tree
        if self.status is not ParseStatus.OK:
            raise SoftwareParseError("no syntax tree is available for this source", context={"path": self.path, "status": self.status.value})
        return ast.parse(self.source.text, filename=self.path, type_comments=False)

    def segment(self, span: SourceSpan) -> str:
        index = self.line_index
        start = index.offset(span.start_line, span.start_col)
        end = index.offset(span.end_line, span.end_col)
        return self.source.text[start:end]

    def line(self, number: int) -> str:
        return self.line_index.line_text(number)

    # -- reporting ---------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        kinds: dict[str, int] = {}
        for symbol in self.symbols:
            kinds[symbol.kind.value] = kinds.get(symbol.kind.value, 0) + 1
        return {
            "path": self.path,
            "module": self.module_name,
            "is_package": self.is_package,
            "language": self.language,
            "status": self.status.value,
            "is_test": self.is_test,
            "lines": self.line_count,
            "symbols": kinds,
            "imports": len(self.imports),
            "calls": len(self.calls),
            "exports": None if self.exports is None else len(self.exports),
            "diagnostics": [item.to_dict() for item in self.diagnostics],
        }


__all__ = ["PARSER_SCHEMA", "SourceModel"]
