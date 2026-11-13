"""Language parsers that turn source text into :class:`SourceModel` facts.

``SoftwareParser`` is a small registry of :class:`LanguageParser`
implementations.  The built-in :class:`PythonParser` uses :mod:`ast` and a
dedicated lexical-scope pass so that every name occurrence is resolved by
Python's LEGB rules (including ``global``/``nonlocal``, class-scope skipping,
comprehension scopes, walrus targets, ``match`` captures and string
annotations).  Other languages are registered through
:meth:`SoftwareParser.register`; without a parser, a file is kept in the
snapshot as an opaque, diffable text artifact.

Parsing never executes, imports or evaluates the analyzed code.

References
----------
Aho, Lam, Sethi & Ullman (2006), *Compilers*, ch. 2 and 7 (scopes, symbol
tables); Python Language Reference, "Naming and binding"; McCabe (1976) and
Campbell (2018) for the complexity measures (the cognitive measure here is an
approximation of the latter).
"""

from __future__ import annotations

__version__ = "2.4.0"

import ast
import re
import sys
import threading

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Optional, Protocol, runtime_checkable

from ..models.source_model import PARSER_SCHEMA, SourceModel
from ..software_types import (
    AttributeUse,
    BindingInfo,
    BindingKind,
    CallSite,
    FunctionMetrics,
    ImportRecord,
    NameUse,
    ParameterInfo,
    ParameterKind,
    ParseDiagnostic,
    ParseStatus,
    Resolution,
    ScopeInfo,
    ScopeKind,
    Severity,
    SignatureInfo,
    SourceFile,
    SourceSpan,
    SUPER_ROOT,
    SymbolKind,
    SymbolRecord,
)
from ..utils.config_loader import get_config_section
from ..utils.software_errors import *
from ..utils.software_helpers import (
    BUILTIN_NAMES,
    MODULE_IMPLICIT_NAMES,
    LineIndex,
    content_digest,
    opt_bool,
    opt_int,
    stable_fingerprint,
)
from logs.logger import get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Software Parser")


# ---------------------------------------------------------------------------
# Limits and registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ParserLimits:
    max_file_bytes: int = 2_000_000
    max_ast_nodes: int = 400_000
    retain_ast: bool = False
    compute_metrics: bool = True
    max_string_names: int = 20_000
    recursion_limit: int = 5_000

    @classmethod
    def from_config(cls, config: Optional[Mapping[str, Any]] = None) -> "ParserLimits":
        section = dict(get_config_section("parser") or {})
        if config:
            section.update(dict(config))
        limits = cls(
            max_file_bytes=opt_int(section, "max_file_bytes", 2_000_000, minimum=1),
            max_ast_nodes=opt_int(section, "max_ast_nodes", 400_000, minimum=1),
            retain_ast=opt_bool(section, "retain_ast", False),
            compute_metrics=opt_bool(section, "compute_metrics", True),
            max_string_names=opt_int(section, "max_string_names", 20_000, minimum=0),
            recursion_limit=opt_int(section, "recursion_limit", 5_000, minimum=0),
        )
        return limits


@runtime_checkable
class LanguageParser(Protocol):
    """Contract for pluggable language front-ends."""

    language: str

    def parse(self, source: SourceFile, *, module_name: str, is_package: bool) -> SourceModel:
        ...


def _empty_model(
    source: SourceFile,
    module_name: str,
    is_package: bool,
    status: ParseStatus,
    diagnostics: Sequence[ParseDiagnostic] = (),
) -> SourceModel:
    return SourceModel(
        source=source,
        module_name=module_name,
        is_package=is_package,
        status=status,
        diagnostics=tuple(diagnostics),
        line_count=LineIndex(source.text).line_count if source.text else 0,
    )


@contextmanager
def _raised_recursion_limit(limit: int) -> Iterator[None]:
    """Temporarily allow deeper recursion for very nested (often generated) code."""

    previous = sys.getrecursionlimit()
    if limit > previous:
        sys.setrecursionlimit(limit)
    try:
        yield
    finally:
        if limit > previous:
            sys.setrecursionlimit(previous)


# ---------------------------------------------------------------------------
# Python front-end
# ---------------------------------------------------------------------------

_DEF_RE = re.compile(r"(?:async[ \t]+)?(?:def|class)[ \t]+")
_MODULE_PART_RE = re.compile(r"from[ \t]+(\.*)[ \t]*([A-Za-z_][\w\.]*)?")
_STRING_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$")
_DYNAMIC_CALLS = frozenset({"locals", "vars", "eval", "exec", "globals"})
_VARIABLE_ORIGINS = frozenset({"assign", "tuple", "annassign"})
_SKIPPED_FIELD_TYPES = (ast.expr_context, ast.operator, ast.boolop, ast.unaryop, ast.cmpop)
_FUNCTION_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)


class _TooLarge(Exception):
    """Internal: the syntax tree exceeded ``max_ast_nodes``."""


class _Binding:
    __slots__ = ("name", "kind", "origins", "span", "count", "symbol", "annotation")

    def __init__(self, name: str, kind: BindingKind, origin: str, span: SourceSpan, symbol: Optional[str], annotation: Optional[str]) -> None:
        self.name = name
        self.kind = kind
        self.origins: list[str] = [origin]
        self.span = span
        self.count = 1
        self.symbol = symbol
        self.annotation = annotation


class _Scope:
    __slots__ = ("qname", "kind", "parent", "span", "bindings", "globals", "nonlocals", "dynamic", "symbol", "has_abstract")

    def __init__(self, qname: str, kind: ScopeKind, parent: Optional["_Scope"], span: SourceSpan, symbol: Optional[str]) -> None:
        self.qname = qname
        self.kind = kind
        self.parent = parent
        self.span = span
        self.symbol = symbol
        self.bindings: dict[str, _Binding] = {}
        self.globals: set[str] = set()
        self.nonlocals: set[str] = set()
        self.dynamic = False
        self.has_abstract = False


def _dotted(node: ast.AST) -> Optional[str]:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _is_super_call(node: ast.AST) -> bool:
    """``super()`` / ``super(Class, self)`` — the receiver of a cooperative-inheritance access."""

    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "super"
        and len(node.args) in (0, 2)
        and not node.keywords
    )


def _expr_text(node: Optional[ast.AST], limit: int = 200) -> Optional[str]:
    if node is None:
        return None
    try:
        text = ast.unparse(node)
    except Exception:  # pragma: no cover - unparse is total for parsed trees
        return "<expr>"
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _is_type_checking(test: ast.AST) -> bool:
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def _docstring_node(body: Sequence[ast.stmt]) -> Optional[ast.Constant]:
    if body and isinstance(body[0], ast.Expr):
        value = body[0].value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            return value
    return None


# -- complexity metrics --------------------------------------------------------

_NESTING_NODES = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.With, ast.AsyncWith) + (
    (ast.Match,) if hasattr(ast, "Match") else ()
) + ((ast.TryStar,) if hasattr(ast, "TryStar") else ())
_TRY_NODES = (ast.Try,) + ((ast.TryStar,) if hasattr(ast, "TryStar") else ())
_MATCH_NODE = getattr(ast, "Match", None)


def _own_nodes(node: ast.AST) -> Iterator[ast.AST]:
    """Yield descendants of a function without entering nested defs/classes/lambdas."""

    stack = list(ast.iter_child_nodes(node))
    stack.reverse()
    while stack:
        current = stack.pop()
        yield current
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        children = list(ast.iter_child_nodes(current))
        children.reverse()
        stack.extend(children)


def _cognitive(node: ast.AST, nesting: int) -> int:
    """Approximate Campbell (2018) cognitive complexity of a function body."""

    total = 0
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        if isinstance(child, ast.If):
            total += 1 + nesting + _cognitive_expr(child.test)
            total += _cognitive_block(child.body, nesting + 1)
            orelse = child.orelse
            while len(orelse) == 1 and isinstance(orelse[0], ast.If):
                total += 1 + _cognitive_expr(orelse[0].test)
                total += _cognitive_block(orelse[0].body, nesting + 1)
                orelse = orelse[0].orelse
            if orelse:
                total += 1 + _cognitive_block(orelse, nesting + 1)
        elif isinstance(child, (ast.For, ast.AsyncFor, ast.While)):
            test = child.test if isinstance(child, ast.While) else child.iter
            total += 1 + nesting + _cognitive_expr(test)
            total += _cognitive_block(child.body, nesting + 1)
            total += _cognitive_block(child.orelse, nesting + 1)
        elif isinstance(child, _TRY_NODES):
            total += _cognitive_block(child.body, nesting)
            for handler in child.handlers:
                total += 1 + nesting + _cognitive_block(handler.body, nesting + 1)
            total += _cognitive_block(child.orelse, nesting) + _cognitive_block(child.finalbody, nesting)
        elif _MATCH_NODE is not None and isinstance(child, _MATCH_NODE):
            total += 1 + nesting
            for case in child.cases:
                total += _cognitive_block(case.body, nesting + 1)
        else:
            total += _cognitive_expr(child) if isinstance(child, ast.expr) else _cognitive(child, nesting)
    return total


def _cognitive_block(body: Sequence[ast.stmt], nesting: int) -> int:
    holder = ast.Module(body=list(body), type_ignores=[])
    return _cognitive(holder, nesting)


def _cognitive_expr(node: ast.AST) -> int:
    """Boolean sequences and conditional expressions inside one expression."""

    total = 0
    stack = [node]
    while stack:
        current = stack.pop()
        if isinstance(current, (ast.Lambda,)):
            continue
        if isinstance(current, (ast.BoolOp, ast.IfExp)):
            total += 1
        stack.extend(ast.iter_child_nodes(current))
    return total


def compute_function_metrics(node: ast.FunctionDef | ast.AsyncFunctionDef) -> FunctionMetrics:
    """McCabe cyclomatic complexity plus size/shape measures of one function."""

    cyclomatic = 1
    statements = 0
    returns = 0
    for child in _own_nodes(node):
        if isinstance(child, (ast.If, ast.IfExp, ast.For, ast.AsyncFor, ast.While, ast.ExceptHandler, ast.Assert)):
            cyclomatic += 1
        elif isinstance(child, ast.BoolOp):
            cyclomatic += max(0, len(child.values) - 1)
        elif isinstance(child, ast.comprehension):
            cyclomatic += 1 + len(child.ifs)
        elif _MATCH_NODE is not None and isinstance(child, ast.match_case):
            cyclomatic += 1
        if isinstance(child, ast.stmt):
            statements += 1
            if isinstance(child, ast.Return):
                returns += 1

    def depth(stmts: Sequence[ast.stmt]) -> int:
        best = 0
        for stmt in stmts:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if isinstance(stmt, _NESTING_NODES):
                inner = 0
                for name in ("body", "orelse", "finalbody"):
                    inner = max(inner, depth(getattr(stmt, name, ()) or ()))
                for handler in getattr(stmt, "handlers", ()) or ():
                    inner = max(inner, depth(handler.body))
                for case in getattr(stmt, "cases", ()) or ():
                    inner = max(inner, depth(case.body))
                best = max(best, 1 + inner)
        return best

    end = getattr(node, "end_lineno", node.lineno) or node.lineno
    a = node.args
    parameter_count = len(a.posonlyargs) + len(a.args) + len(a.kwonlyargs) + (1 if a.vararg else 0) + (1 if a.kwarg else 0)
    return FunctionMetrics(
        cyclomatic=cyclomatic,
        cognitive=_cognitive(node, 0),
        lines=end - node.lineno + 1,
        statements=statements,
        parameters=parameter_count,
        returns=returns,
        max_nesting=depth(node.body),
    )


# -- fact builder --------------------------------------------------------------

class _PythonFactBuilder:
    """Single-pass collector followed by a resolution pass."""

    def __init__(self, source: SourceFile, module_name: str, is_package: bool, limits: ParserLimits) -> None:
        self.source = source
        self.text = source.text
        self.index = LineIndex(self.text)
        self.module_name = module_name
        self.is_package = is_package
        self.module_qname = module_name or "<root>"
        self.limits = limits

        self.scopes: list[_Scope] = []
        self._stack: list[_Scope] = []
        self._qname_counts: dict[str, int] = {}
        self.symbols: list[SymbolRecord] = []
        self.imports: list[ImportRecord] = []
        self.raw_names: list[tuple[str, SourceSpan, str, _Scope]] = []
        self.raw_attrs: list[tuple[tuple[str, ...], bool, SourceSpan, str, _Scope]] = []
        self.raw_calls: list[tuple[tuple[str, ...], SourceSpan, SourceSpan, _Scope, int, tuple[str, ...], tuple[SourceSpan, ...], bool, bool, bool]] = []
        self.exports: Optional[list[str]] = None
        self.export_spans: list[tuple[str, SourceSpan]] = []
        self.string_names: set[str] = set()

        self._origin = "assign"
        self._tc_depth = 0
        self._cond_depth = 0
        self._annotation_depth = 0
        self._literal_depth = 0
        self._awaited: set[int] = set()
        self._docstrings: set[int] = set()
        self._nodes = 0

    # -- positions ---------------------------------------------------------
    def _span(self, node: ast.AST) -> SourceSpan:
        line = getattr(node, "lineno", 1)
        end_line = getattr(node, "end_lineno", None) or line
        col = getattr(node, "col_offset", 0)
        end_col = getattr(node, "end_col_offset", None)
        if end_col is None:
            end_col = col
        return SourceSpan(line, self.index.char_column(line, col), end_line, self.index.char_column(end_line, end_col))

    def _name_span(self, node: ast.AST, length: int) -> SourceSpan:
        """Span of an identifier starting at the node start."""

        span = self._span(node)
        return SourceSpan(span.start_line, span.start_col, span.start_line, span.start_col + length)

    def _tail_name_span(self, node: ast.AST, length: int) -> SourceSpan:
        """Span of an identifier ending where the node ends."""

        span = self._span(node)
        return SourceSpan(span.end_line, max(0, span.end_col - length), span.end_line, span.end_col)

    def _def_name_span(self, node: ast.AST, name: str) -> SourceSpan:
        start = self.index.offset_from_utf8(node.lineno, node.col_offset)  # type: ignore[attr-defined]
        match = _DEF_RE.match(self.text, start)
        if match is not None:
            line, col = self.index.position(match.end())
            return SourceSpan(line, col, line, col + len(name))
        span = self._span(node)
        return SourceSpan(span.start_line, span.start_col, span.start_line, span.start_col + len(name))

    def _decorated_span(self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> SourceSpan:
        base = self._span(node)
        if not node.decorator_list:
            return base
        first = node.decorator_list[0]
        line = first.lineno
        col = self.index.char_column(line, first.col_offset)
        at = self.index.line_text(line).rfind("@", 0, col + 1)
        start_col = at if at >= 0 else col
        return SourceSpan(line, start_col, base.end_line, base.end_col)

    # -- scopes and bindings -------------------------------------------------
    @property
    def scope(self) -> _Scope:
        return self._stack[-1]

    def _push(self, kind: ScopeKind, qname: str, span: SourceSpan, symbol: Optional[str]) -> _Scope:
        count = self._qname_counts.get(qname, 0)
        self._qname_counts[qname] = count + 1
        unique = qname if not count else f"{qname}@{span.start_line}" + (f"#{count}" if count > 1 else "")
        scope = _Scope(unique, kind, self._stack[-1] if self._stack else None, span, symbol)
        self.scopes.append(scope)
        self._stack.append(scope)
        return scope

    def _pop(self) -> None:
        self._stack.pop()

    def _nearest_non_comprehension(self) -> _Scope:
        for scope in reversed(self._stack):
            if scope.kind is not ScopeKind.COMPREHENSION:
                return scope
        return self._stack[0]

    def _qualify(self, name: str) -> str:
        owner = self.scope
        return f"{owner.symbol or owner.qname}.{name}"

    def _bind(
        self,
        name: str,
        kind: BindingKind,
        origin: str,
        span: SourceSpan,
        *,
        symbol: Optional[str] = None,
        annotation: Optional[str] = None,
        scope: Optional[_Scope] = None,
    ) -> None:
        target = scope or self.scope
        if name in target.nonlocals:
            return
        if name in target.globals and target.kind is not ScopeKind.MODULE:
            target = self._stack[0]
        existing = target.bindings.get(name)
        if existing is None:
            target.bindings[name] = _Binding(name, kind, origin, span, symbol, annotation)
            return
        existing.count += 1
        if origin not in existing.origins:
            existing.origins.append(origin)
        if existing.symbol is None and symbol is not None:
            existing.symbol = symbol
        if existing.annotation is None and annotation is not None:
            existing.annotation = annotation

    # -- traversal -----------------------------------------------------------
    def visit(self, node: ast.AST) -> None:
        self._nodes += 1
        if self._nodes > self.limits.max_ast_nodes:
            raise _TooLarge()
        method = getattr(self, "visit_" + type(node).__name__, None)
        if method is None:
            self.generic_visit(node)
        else:
            method(node)

    def generic_visit(self, node: ast.AST) -> None:
        for _, value in ast.iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST) and not isinstance(item, _SKIPPED_FIELD_TYPES):
                        self.visit(item)
            elif isinstance(value, ast.AST) and not isinstance(value, _SKIPPED_FIELD_TYPES):
                self.visit(value)

    def _visit_all(self, nodes: Sequence[ast.AST]) -> None:
        for item in nodes:
            self.visit(item)

    def _visit_annotation(self, node: Optional[ast.AST]) -> None:
        if node is None:
            return
        self._annotation_depth += 1
        try:
            self.visit(node)
        finally:
            self._annotation_depth -= 1

    def _visit_target(self, node: ast.AST, origin: str) -> None:
        previous = self._origin
        self._origin = origin
        try:
            self.visit(node)
        finally:
            self._origin = previous

    def run(self, tree: ast.Module) -> None:
        last_line = self.index.line_count
        module_span = SourceSpan(1, 0, last_line, 0)
        self._push(ScopeKind.MODULE, self.module_qname, module_span, self.module_qname)
        doc = _docstring_node(tree.body)
        if doc is not None:
            self._docstrings.add(id(doc))
        self._visit_all(tree.body)
        self._pop()

    # -- statements ------------------------------------------------------------
    def visit_Import(self, node: ast.Import) -> None:
        self._record_imports(node, None, 0)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self._record_imports(node, node.module, node.level or 0)

    def _record_imports(self, node: ast.Import | ast.ImportFrom, module: Optional[str], level: int) -> None:
        is_from = isinstance(node, ast.ImportFrom)
        statement_span = self._span(node)
        module_span: Optional[SourceSpan] = None
        if is_from:
            start = self.index.offset_from_utf8(node.lineno, node.col_offset)
            match = _MODULE_PART_RE.match(self.text, start)
            if match is not None and match.group(2):
                line, col = self.index.position(match.start(2))
                module_span = SourceSpan(line, col, line, col + len(match.group(2)))
        scope = self.scope
        in_function = any(item.kind in (ScopeKind.FUNCTION, ScopeKind.LAMBDA) for item in self._stack)
        for alias in node.names:
            alias_span = self._span(alias)
            name_text = alias.name
            name_span = SourceSpan(alias_span.start_line, alias_span.start_col, alias_span.start_line, alias_span.start_col + len(name_text))
            as_span = self._tail_name_span(alias, len(alias.asname)) if alias.asname else None
            record = ImportRecord(
                module=module if is_from else name_text,
                name=name_text if is_from else None,
                alias=alias.asname,
                level=level,
                is_from=is_from,
                scope=scope.qname,
                span=alias_span,
                name_span=name_span,
                statement_span=statement_span,
                alias_span=as_span,
                module_span=module_span,
                type_checking=self._tc_depth > 0,
                lazy=in_function,
                conditional=self._cond_depth > 0,
            )
            self.imports.append(record)
            bound = record.bound_name
            if bound is not None:
                origin = "future" if (is_from and module == "__future__") else "import"
                binding_span = as_span if as_span is not None else name_span
                if not is_from and not alias.asname:
                    binding_span = SourceSpan(name_span.start_line, name_span.start_col, name_span.start_line, name_span.start_col + len(bound))
                self._bind(bound, BindingKind.IMPORT, origin, binding_span)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def _bind_type_params(self, node: ast.AST) -> None:
        for param in getattr(node, "type_params", ()) or ():
            self._bind(param.name, BindingKind.VARIABLE, "typeparam", self._span(param))

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        enclosing = self.scope
        is_async = isinstance(node, ast.AsyncFunctionDef)
        self._visit_all(node.decorator_list)
        args = node.args
        self._visit_all(args.defaults)
        self._visit_all([item for item in args.kw_defaults if item is not None])
        every_arg = [*args.posonlyargs, *args.args, *args.kwonlyargs] + [item for item in (args.vararg, args.kwarg) if item is not None]
        for arg in every_arg:
            self._visit_annotation(arg.annotation)
        self._visit_annotation(node.returns)
        self._bind_type_params(node)

        qname = self._qualify(node.name)
        is_method = enclosing.kind is ScopeKind.CLASS
        name_span = self._def_name_span(node, node.name)
        self._bind(node.name, BindingKind.FUNCTION, "def", name_span, symbol=qname)

        span = self._span(node)
        scope = self._push(ScopeKind.FUNCTION, qname, span, qname)
        doc = _docstring_node(node.body)
        if doc is not None:
            self._docstrings.add(id(doc))
        for arg in every_arg:
            self._bind(arg.arg, BindingKind.PARAMETER, "param", self._name_span(arg, len(arg.arg)))
        self._visit_all(node.body)
        self._pop()

        decorators = tuple(self._decorator_text(item) for item in node.decorator_list)
        abstract = any(item.rsplit(".", 1)[-1] == "abstractmethod" for item in decorators)
        if abstract and is_method:
            enclosing.has_abstract = True
        in_function = enclosing.kind in (ScopeKind.FUNCTION, ScopeKind.LAMBDA)
        signature_text = ast.dump(node.args, annotate_fields=False) + (ast.dump(node.returns, annotate_fields=False) if node.returns else "") + str(is_async)
        self.symbols.append(
            SymbolRecord(
                qualified_name=qname,
                name=node.name,
                kind=SymbolKind.METHOD if is_method else SymbolKind.FUNCTION,
                parent=enclosing.symbol or enclosing.qname,
                span=span,
                name_span=name_span,
                full_span=self._decorated_span(node),
                decorators=decorators,
                signature=self._signature(node, is_async),
                is_public=(not in_function) and (not node.name.startswith("_") or (node.name.startswith("__") and node.name.endswith("__"))),
                is_async=is_async,
                is_abstract=abstract,
                has_docstring=doc is not None,
                body_digest=content_digest(ast.dump(node, annotate_fields=False)),
                signature_digest=content_digest(signature_text),
                metrics=compute_function_metrics(node) if self.limits.compute_metrics else None,
            )
        )

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        enclosing = self.scope
        self._visit_all(node.decorator_list)
        self._visit_all(node.bases)
        self._visit_all([item.value for item in node.keywords])
        self._bind_type_params(node)
        qname = self._qualify(node.name)
        name_span = self._def_name_span(node, node.name)
        self._bind(node.name, BindingKind.CLASS, "class", name_span, symbol=qname)

        span = self._span(node)
        scope = self._push(ScopeKind.CLASS, qname, span, qname)
        doc = _docstring_node(node.body)
        if doc is not None:
            self._docstrings.add(id(doc))
        self._visit_all(node.body)
        self._pop()

        bases = tuple(_expr_text(item) or "" for item in node.bases)
        metaclass = next((_expr_text(item.value) for item in node.keywords if item.arg == "metaclass"), None)
        abstract_bases = {"ABC", "abc.ABC", "Protocol", "typing.Protocol", "typing_extensions.Protocol"}
        is_abstract = scope.has_abstract or any(base in abstract_bases for base in bases) or (metaclass or "").rsplit(".", 1)[-1] == "ABCMeta"
        decorators = tuple(self._decorator_text(item) for item in node.decorator_list)
        shell = [ast.dump(item, annotate_fields=False) for item in (*node.decorator_list, *node.bases, *node.keywords)]
        shell.extend(ast.dump(item, annotate_fields=False) for item in node.body if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)))
        in_function = enclosing.kind in (ScopeKind.FUNCTION, ScopeKind.LAMBDA)
        self.symbols.append(
            SymbolRecord(
                qualified_name=qname,
                name=node.name,
                kind=SymbolKind.CLASS,
                parent=enclosing.symbol or enclosing.qname,
                span=span,
                name_span=name_span,
                full_span=self._decorated_span(node),
                decorators=decorators,
                bases=bases + ((f"metaclass={metaclass}",) if metaclass else ()),
                is_public=(not in_function) and (not node.name.startswith("_")),
                is_abstract=is_abstract,
                has_docstring=doc is not None,
                body_digest=content_digest("\n".join(shell)),
                signature_digest=content_digest("|".join(bases)),
            )
        )

    def _decorator_text(self, node: ast.AST) -> str:
        target = node.func if isinstance(node, ast.Call) else node
        return _dotted(target) or _expr_text(target, 80) or ""

    def _signature(self, node: ast.FunctionDef | ast.AsyncFunctionDef, is_async: bool) -> SignatureInfo:
        a = node.args
        positional = [*a.posonlyargs, *a.args]
        offset = len(positional) - len(a.defaults)
        params: list[ParameterInfo] = []
        for position, arg in enumerate(positional):
            default = a.defaults[position - offset] if position >= offset else None
            params.append(
                ParameterInfo(
                    name=arg.arg,
                    kind=ParameterKind.POSITIONAL_ONLY if position < len(a.posonlyargs) else ParameterKind.POSITIONAL_OR_KEYWORD,
                    has_default=default is not None,
                    annotation=_expr_text(arg.annotation, 120),
                    default=_expr_text(default, 80),
                )
            )
        if a.vararg:
            params.append(ParameterInfo(a.vararg.arg, ParameterKind.VAR_POSITIONAL, False, _expr_text(a.vararg.annotation, 120)))
        for arg, default in zip(a.kwonlyargs, a.kw_defaults):
            params.append(ParameterInfo(arg.arg, ParameterKind.KEYWORD_ONLY, default is not None, _expr_text(arg.annotation, 120), _expr_text(default, 80)))
        if a.kwarg:
            params.append(ParameterInfo(a.kwarg.arg, ParameterKind.VAR_KEYWORD, False, _expr_text(a.kwarg.annotation, 120)))
        return SignatureInfo(tuple(params), _expr_text(node.returns, 120), is_async)

    # assignments ----------------------------------------------------------------
    def visit_Assign(self, node: ast.Assign) -> None:
        self.visit(node.value)
        for target in node.targets:
            self._visit_target(target, "assign")
        if self.scope.kind is ScopeKind.MODULE and any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets):
            self.exports = []
            self.export_spans = []
            self._collect_exports(node.value)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.visit(node.value)
        if isinstance(node.target, ast.Name):
            span = self._span(node.target)
            self.raw_names.append((node.target.id, span, "load", self.scope))
            self._bind(node.target.id, BindingKind.VARIABLE, "augassign", span)
            self.raw_names.append((node.target.id, span, "store", self.scope))
            if self.scope.kind is ScopeKind.MODULE and node.target.id == "__all__":
                if self.exports is None:
                    self.exports = []
                self._collect_exports(node.value)
        else:
            self.visit(node.target)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._visit_annotation(node.annotation)
        if node.value is not None:
            self.visit(node.value)
        annotation = _expr_text(node.annotation, 120)
        if isinstance(node.target, ast.Name):
            span = self._span(node.target)
            self.raw_names.append((node.target.id, span, "store", self.scope))
            self._bind(node.target.id, BindingKind.VARIABLE, "annassign", span, annotation=annotation)
            if self.scope.kind is ScopeKind.MODULE and node.target.id == "__all__" and node.value is not None:
                self.exports = []
                self.export_spans = []
                self._collect_exports(node.value)
        else:
            self.visit(node.target)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self.visit(node.value)
        span = self._span(node.target)
        self.raw_names.append((node.target.id, span, "store", self.scope))
        self._bind(node.target.id, BindingKind.VARIABLE, "walrus", span, scope=self._nearest_non_comprehension())

    def _collect_exports(self, value: ast.AST) -> None:
        if isinstance(value, (ast.List, ast.Tuple, ast.Set)):
            for element in value.elts:
                if isinstance(element, ast.Constant) and isinstance(element.value, str):
                    if self.exports is None:
                        self.exports = []
                    self.exports.append(element.value)
                    self.export_spans.append((element.value, self._span(element)))
        elif isinstance(value, ast.BinOp) and isinstance(value.op, ast.Add):
            self._collect_exports(value.left)
            self._collect_exports(value.right)

    def visit_Expr(self, node: ast.Expr) -> None:
        value = node.value
        # __all__.extend([...]) / __all__.append("x") at module level
        if (
            self.scope.kind is ScopeKind.MODULE
            and isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and isinstance(value.func.value, ast.Name)
            and value.func.value.id == "__all__"
            and value.func.attr in ("extend", "append")
            and value.args
        ):
            if self.exports is None:
                self.exports = []
            argument = value.args[0]
            if value.func.attr == "append" and isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                self.exports.append(argument.value)
                self.export_spans.append((argument.value, self._span(argument)))
            else:
                self._collect_exports(argument)
        self.visit(value)

    def visit_Global(self, node: ast.Global) -> None:
        self.scope.globals.update(node.names)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self.scope.nonlocals.update(node.names)

    # control flow -----------------------------------------------------------------
    def visit_For(self, node: ast.For) -> None:
        self.visit(node.iter)
        self._visit_target(node.target, "for")
        self._visit_all(node.body)
        self._visit_all(node.orelse)

    visit_AsyncFor = visit_For

    def visit_While(self, node: ast.While) -> None:
        self.visit(node.test)
        self._cond_depth += 1
        self._visit_all(node.body)
        self._visit_all(node.orelse)
        self._cond_depth -= 1

    def visit_If(self, node: ast.If) -> None:
        self.visit(node.test)
        type_checking = _is_type_checking(node.test)
        self._cond_depth += 1
        if type_checking:
            self._tc_depth += 1
        self._visit_all(node.body)
        if type_checking:
            self._tc_depth -= 1
        self._visit_all(node.orelse)
        self._cond_depth -= 1

    def visit_Try(self, node: ast.Try) -> None:
        self._cond_depth += 1
        self._visit_all(node.body)
        for handler in node.handlers:
            self.visit(handler)
        self._visit_all(node.orelse)
        self._cond_depth -= 1
        self._visit_all(node.finalbody)

    visit_TryStar = visit_Try

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.type is not None:
            self.visit(node.type)
        if node.name:
            span = self._span(node)
            body_start = self.index.offset(node.body[0].lineno, 0) if node.body else self.index.offset(span.end_line, span.end_col)
            start = self.index.offset(span.start_line, span.start_col)
            match = re.compile(r"\bas[ \t]+(" + re.escape(node.name) + r")\b").search(self.text, start, max(start, body_start))
            if match is not None:
                line, col = self.index.position(match.start(1))
                name_span = SourceSpan(line, col, line, col + len(node.name))
            else:
                name_span = SourceSpan(span.start_line, span.start_col, span.start_line, span.start_col)
            self.raw_names.append((node.name, name_span, "store", self.scope))
            self._bind(node.name, BindingKind.VARIABLE, "except", name_span)
        self._visit_all(node.body)

    def visit_With(self, node: ast.With) -> None:
        for item in node.items:
            self.visit(item.context_expr)
            if item.optional_vars is not None:
                self._visit_target(item.optional_vars, "with")
        self._visit_all(node.body)

    visit_AsyncWith = visit_With

    def visit_Match(self, node: Any) -> None:
        self.visit(node.subject)
        for case in node.cases:
            self._bind_pattern(case.pattern)
            if case.guard is not None:
                self.visit(case.guard)
            self._visit_all(case.body)

    def _bind_pattern(self, pattern: Any) -> None:
        kind = type(pattern).__name__
        if kind == "MatchValue":
            self.visit(pattern.value)
        elif kind in ("MatchSequence", "MatchOr"):
            for item in pattern.patterns:
                self._bind_pattern(item)
        elif kind == "MatchMapping":
            self._visit_all(pattern.keys)
            for item in pattern.patterns:
                self._bind_pattern(item)
            if pattern.rest:
                self._bind(pattern.rest, BindingKind.VARIABLE, "match", self._tail_name_span(pattern, len(pattern.rest)))
        elif kind == "MatchClass":
            self.visit(pattern.cls)
            for item in (*pattern.patterns, *pattern.kwd_patterns):
                self._bind_pattern(item)
        elif kind == "MatchStar":
            if pattern.name:
                self._bind(pattern.name, BindingKind.VARIABLE, "match", self._tail_name_span(pattern, len(pattern.name)))
        elif kind == "MatchAs":
            if pattern.pattern is not None:
                self._bind_pattern(pattern.pattern)
            if pattern.name:
                self._bind(pattern.name, BindingKind.VARIABLE, "match", self._tail_name_span(pattern, len(pattern.name)))

    def visit_TypeAlias(self, node: Any) -> None:  # Python 3.12
        self._bind_type_params(node)
        self._bind(node.name.id, BindingKind.VARIABLE, "typealias", self._span(node.name))
        self.visit(node.value)

    # expressions ------------------------------------------------------------------
    def visit_Name(self, node: ast.Name) -> None:
        span = self._span(node)
        kind = type(node.ctx).__name__
        if kind == "Store":
            self.raw_names.append((node.id, span, "store", self.scope))
            self._bind(node.id, BindingKind.VARIABLE, self._origin, span)
        elif kind == "Del":
            self.raw_names.append((node.id, span, "del", self.scope))
        else:
            self.raw_names.append((node.id, span, "load", self.scope))

    def _visit_sequence_target(self, node: ast.Tuple | ast.List | ast.Starred) -> None:
        previous = self._origin
        if previous == "assign":
            self._origin = "tuple"
        try:
            self.generic_visit(node)
        finally:
            self._origin = previous

    def visit_Tuple(self, node: ast.Tuple) -> None:
        if isinstance(node.ctx, ast.Store):
            self._visit_sequence_target(node)
        else:
            self.generic_visit(node)

    visit_List = visit_Tuple

    def visit_Starred(self, node: ast.Starred) -> None:
        if isinstance(node.ctx, ast.Store):
            self._visit_sequence_target(node)
        else:
            self.generic_visit(node)

    def _chain(self, node: ast.Attribute) -> Optional[tuple[str, ...]]:
        parts: list[str] = []
        current: ast.AST = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
            return tuple(reversed(parts))
        if _is_super_call(current):
            parts.append(SUPER_ROOT)
            return tuple(reversed(parts))
        return None

    def visit_Attribute(self, node: ast.Attribute) -> None:
        self.visit(node.value)
        chain = self._chain(node)
        context = {"Load": "load", "Store": "store", "Del": "del"}.get(type(node.ctx).__name__, "load")
        self.raw_attrs.append((chain or (node.attr,), chain is not None, self._tail_name_span(node, len(node.attr)), context, self.scope))

    def visit_Await(self, node: ast.Await) -> None:
        if isinstance(node.value, ast.Call):
            self._awaited.add(id(node.value))
        self.visit(node.value)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        callee: tuple[str, ...] = ()
        if isinstance(func, ast.Name):
            callee = (func.id,)
            if func.id in _DYNAMIC_CALLS:
                self.scope.dynamic = True
        elif isinstance(func, ast.Attribute):
            callee = self._chain(func) or ()
        keywords = tuple(item.arg for item in node.keywords if item.arg)
        keyword_spans = tuple(
            self._name_span(item, len(item.arg)) for item in node.keywords if item.arg
        )
        self.raw_calls.append(
            (
                callee,
                self._span(func),
                self._span(node),
                self.scope,
                sum(1 for item in node.args if not isinstance(item, ast.Starred)),
                keywords,
                keyword_spans,
                any(isinstance(item, ast.Starred) for item in node.args),
                any(item.arg is None for item in node.keywords),
                id(node) in self._awaited,
            )
        )
        self.visit(func)
        self._visit_all(node.args)
        self._visit_all([item.value for item in node.keywords])

    def visit_Subscript(self, node: ast.Subscript) -> None:
        name = _dotted(node.value)
        literal = name is not None and name.rsplit(".", 1)[-1] == "Literal"
        self.visit(node.value)
        if literal:
            self._literal_depth += 1
        self.visit(node.slice)
        if literal:
            self._literal_depth -= 1

    def visit_Constant(self, node: ast.Constant) -> None:
        value = node.value
        if not isinstance(value, str) or id(node) in self._docstrings:
            return
        if len(self.string_names) < self.limits.max_string_names and len(value) <= 120 and _STRING_NAME_RE.match(value):
            self.string_names.add(value)
            if "." in value:
                self.string_names.add(value.rsplit(".", 1)[1])
        if self._annotation_depth and not self._literal_depth and len(value) <= 500:
            try:
                expression = ast.parse(value.strip(), mode="eval")
            except (SyntaxError, ValueError, RecursionError):
                return
            span = self._span(node)
            for sub in ast.walk(expression):
                if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
                    self.raw_names.append((sub.id, span, "string", self.scope))

    def visit_Lambda(self, node: ast.Lambda) -> None:
        args = node.args
        self._visit_all(args.defaults)
        self._visit_all([item for item in args.kw_defaults if item is not None])
        span = self._span(node)
        self._push(ScopeKind.LAMBDA, f"{self.scope.symbol or self.scope.qname}.<lambda>@{span.start_line}:{span.start_col}", span, None)
        every_arg = [*args.posonlyargs, *args.args, *args.kwonlyargs] + [item for item in (args.vararg, args.kwarg) if item is not None]
        for arg in every_arg:
            self._bind(arg.arg, BindingKind.PARAMETER, "param", self._name_span(arg, len(arg.arg)))
        self.visit(node.body)
        self._pop()

    def _visit_comprehension(self, node: ast.AST, elements: Sequence[ast.AST]) -> None:
        generators: list[ast.comprehension] = node.generators  # type: ignore[attr-defined]
        self.visit(generators[0].iter)
        span = self._span(node)
        label = type(node).__name__.lower()
        self._push(ScopeKind.COMPREHENSION, f"{self.scope.symbol or self.scope.qname}.<{label}>@{span.start_line}:{span.start_col}", span, None)
        for position, generator in enumerate(generators):
            if position:
                self.visit(generator.iter)
            self._visit_target(generator.target, "comp")
            self._visit_all(generator.ifs)
        self._visit_all(elements)
        self._pop()

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._visit_comprehension(node, [node.elt])

    visit_SetComp = visit_ListComp
    visit_GeneratorExp = visit_ListComp

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._visit_comprehension(node, [node.key, node.value])

    # -- resolution pass ---------------------------------------------------------
    def _resolve(self, name: str, scope: _Scope) -> tuple[Optional[str], Resolution]:
        if name in scope.nonlocals:
            current = scope.parent
            while current is not None:
                if current.kind is not ScopeKind.CLASS and name in current.bindings and name not in current.nonlocals:
                    return current.qname, Resolution.ENCLOSING
                current = current.parent
            return None, Resolution.UNRESOLVED
        if name in scope.globals:
            return self._stack_root.qname, Resolution.GLOBAL
        current: Optional[_Scope] = scope
        first = True
        while current is not None:
            if first or current.kind is not ScopeKind.CLASS:
                if name in current.bindings and name not in current.globals:
                    if first:
                        return current.qname, Resolution.LOCAL
                    return current.qname, (Resolution.GLOBAL if current.kind is ScopeKind.MODULE else Resolution.ENCLOSING)
            first = False
            current = current.parent
        if name in BUILTIN_NAMES or name in MODULE_IMPLICIT_NAMES:
            return None, Resolution.BUILTIN
        return None, Resolution.UNRESOLVED

    def finish(self) -> dict[str, Any]:
        self._stack_root = self.scopes[0]
        scope_infos: list[ScopeInfo] = []
        symbols = list(self.symbols)
        defined = {item.qualified_name for item in symbols}
        for scope in self.scopes:
            bindings = tuple(
                BindingInfo(
                    name=item.name,
                    scope=scope.qname,
                    kind=item.kind,
                    origins=tuple(item.origins),
                    span=item.span,
                    assignment_count=item.count,
                    symbol=item.symbol,
                )
                for item in sorted(scope.bindings.values(), key=lambda b: b.span.as_tuple())
            )
            scope_infos.append(
                ScopeInfo(
                    qualified_name=scope.qname,
                    kind=scope.kind,
                    parent=scope.parent.qname if scope.parent else None,
                    span=scope.span,
                    bindings=bindings,
                    global_names=tuple(sorted(scope.globals)),
                    nonlocal_names=tuple(sorted(scope.nonlocals)),
                    dynamic=scope.dynamic,
                    symbol=scope.symbol,
                )
            )
            if scope.kind in (ScopeKind.MODULE, ScopeKind.CLASS):
                for item in scope.bindings.values():
                    if item.kind is not BindingKind.VARIABLE or not (set(item.origins) & _VARIABLE_ORIGINS):
                        continue
                    if item.name.startswith("__") and item.name.endswith("__"):
                        continue
                    qname = f"{scope.symbol}.{item.name}"
                    if qname in defined:
                        continue
                    defined.add(qname)
                    symbols.append(
                        SymbolRecord(
                            qualified_name=qname,
                            name=item.name,
                            kind=SymbolKind.VARIABLE,
                            parent=scope.symbol,
                            span=item.span,
                            name_span=item.span,
                            full_span=item.span,
                            annotation=item.annotation,
                            is_public=not item.name.startswith("_"),
                        )
                    )
        symbols.sort(key=lambda s: (s.full_span.start_line, s.full_span.start_col, s.full_span.end_line))

        names: list[NameUse] = []
        for name, span, context, scope in self.raw_names:
            binding_scope, resolution = self._resolve(name, scope)
            names.append(NameUse(name, span, context, scope.qname, binding_scope, resolution))

        attributes: list[AttributeUse] = []
        for chain, rooted, span, context, scope in self.raw_attrs:
            if not rooted:
                root_scope, root_resolution = None, Resolution.UNRESOLVED
            elif chain[0] == SUPER_ROOT:
                root_scope, root_resolution = scope.qname, Resolution.BUILTIN
            else:
                root_scope, root_resolution = self._resolve(chain[0], scope)
            attributes.append(AttributeUse(chain, span, context, scope.qname, rooted, root_scope, root_resolution))

        calls: list[CallSite] = []
        for callee, callee_span, span, scope, positional, keywords, keyword_spans, star_args, star_kwargs, awaited in self.raw_calls:
            if not callee:
                root_scope, root_resolution = None, Resolution.UNRESOLVED
            elif callee[0] == SUPER_ROOT:
                root_scope, root_resolution = scope.qname, Resolution.BUILTIN
            else:
                root_scope, root_resolution = self._resolve(callee[0], scope)
            calls.append(CallSite(callee, callee_span, span, scope.qname, positional, keywords, keyword_spans, star_args, star_kwargs, awaited, root_scope, root_resolution))

        return {
            "symbols": tuple(symbols),
            "scopes": tuple(scope_infos),
            "imports": tuple(self.imports),
            "names": tuple(names),
            "attributes": tuple(attributes),
            "calls": tuple(calls),
            "exports": None if self.exports is None else tuple(dict.fromkeys(self.exports)),
            "export_spans": tuple(self.export_spans),
            "string_names": frozenset(self.string_names),
        }


class PythonParser:
    """:mod:`ast`-based front-end producing lexically resolved facts."""

    language = "python"

    def __init__(self, limits: Optional[ParserLimits] = None) -> None:
        self.limits = limits or ParserLimits.from_config()

    def parse(self, source: SourceFile, *, module_name: str, is_package: bool) -> SourceModel:
        limits = self.limits
        if source.size > limits.max_file_bytes:
            return _empty_model(
                source, module_name, is_package, ParseStatus.TOO_LARGE,
                [ParseDiagnostic("SW002", f"file is {source.size} bytes (limit {limits.max_file_bytes})", 1, 0, Severity.MEDIUM)],
            )
        try:
            with _raised_recursion_limit(limits.recursion_limit):
                tree = ast.parse(source.text, filename=source.path, type_comments=False)
        except SyntaxError as exc:
            line = int(exc.lineno or 1)
            column = max(0, int(exc.offset or 1) - 1)
            return _empty_model(source, module_name, is_package, ParseStatus.SYNTAX_ERROR, [ParseDiagnostic("SW001", str(exc.msg), line, column)])
        except (ValueError, RecursionError, MemoryError) as exc:
            return _empty_model(
                source, module_name, is_package, ParseStatus.SYNTAX_ERROR,
                [ParseDiagnostic("SW001", f"{type(exc).__name__}: {exc}"[:300], 1, 0)],
            )

        builder = _PythonFactBuilder(source, module_name, is_package, limits)
        try:
            with _raised_recursion_limit(limits.recursion_limit):
                builder.run(tree)
                facts = builder.finish()
        except (_TooLarge, RecursionError):
            return _empty_model(
                source, module_name, is_package, ParseStatus.TOO_LARGE,
                [ParseDiagnostic("SW002", "syntax tree exceeds configured size/depth limits", 1, 0, Severity.MEDIUM)],
            )
        return SourceModel(
            source=source,
            module_name=module_name,
            is_package=is_package,
            status=ParseStatus.OK,
            line_count=builder.index.line_count,
            tree=tree if limits.retain_ast else None,
            **facts,
        )


class SoftwareParser:
    """Registry dispatching sources to their language parser."""

    def __init__(self, config: Optional[Mapping[str, Any]] = None, *, limits: Optional[ParserLimits] = None) -> None:
        self.limits = limits or ParserLimits.from_config(config)
        self._parsers: dict[str, LanguageParser] = {}
        self._lock = threading.RLock()
        self.register(PythonParser(self.limits))

    def register(self, parser: LanguageParser, *, replace: bool = False) -> None:
        language = str(getattr(parser, "language", "")).strip().lower()
        if not language or not callable(getattr(parser, "parse", None)):
            raise SoftwareConfigurationError("a language parser needs a non-empty 'language' and a callable 'parse'")
        with self._lock:
            if language in self._parsers and not replace:
                raise SoftwareConfigurationError("a parser for this language is already registered", context={"language": language})
            self._parsers[language] = parser

    def languages(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._parsers))

    def supports(self, language: str) -> bool:
        with self._lock:
            return language in self._parsers

    @property
    def signature(self) -> str:
        """Cache discriminator: changes whenever parse results could change."""

        with self._lock:
            languages = tuple(sorted(self._parsers))
        return f"{PARSER_SCHEMA}:{stable_fingerprint([self.limits.max_file_bytes, self.limits.max_ast_nodes, self.limits.compute_metrics, languages], length=10)}"

    def parse(self, source: SourceFile, *, module_name: str = "", is_package: bool = False, strict: bool = False) -> SourceModel:
        with self._lock:
            parser = self._parsers.get(source.language)
        if parser is None:
            if strict:
                raise UnsupportedLanguageError("no parser is registered for this language", context={"language": source.language, "path": source.path})
            return _empty_model(source, module_name, is_package, ParseStatus.UNSUPPORTED)
        model = parser.parse(source, module_name=module_name, is_package=is_package)
        if strict and model.status is not ParseStatus.OK:
            raise SoftwareParseError(
                "source could not be parsed",
                context={"path": source.path, "status": model.status.value, "diagnostics": [item.to_dict() for item in model.diagnostics]},
            )
        return model


__all__ = [
    "LanguageParser",
    "ParserLimits",
    "PythonParser",
    "SoftwareParser",
    "compute_function_metrics",
]
