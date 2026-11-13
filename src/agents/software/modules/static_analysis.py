"""Static program analysis: correctness, security, complexity, maintainability.

Rules are identified ``SWnnn``:

====== ===============================================================
SW1xx  correctness (undefined names, unused bindings, bad calls, imports)
SW2xx  suspicious constructs found on the syntax tree
SW3xx  security
SW4xx  complexity and size
SW5xx  maintainability / style
====== ===============================================================

A finding can be suppressed on its own line with ``# noqa`` or
``# noqa: SW102,SW301``.  Nothing is executed or imported.
"""

from __future__ import annotations

__version__ = "2.4.0"

import ast
import re

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Optional

from ..models.source_model import SourceModel
from ..semantics.symbol_index import SymbolIndex
from ..software_types import (
    BindingKind,
    Finding,
    FindingCategory,
    ParameterKind,
    ParseStatus,
    Resolution,
    ScopeKind,
    Severity,
    SourceSpan,
    StaticAnalysisReport,
    SymbolKind,
    SymbolRecord,
)
from ..utils.software_helpers import (
    opt_bool,
    opt_int,
    opt_strings,
    truncate_items,
)
from logs.logger import get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Software Static Analysis")

CORRECT, SECURITY, COMPLEX, MAINT, STYLE, DEAD = (
    FindingCategory.CORRECTNESS,
    FindingCategory.SECURITY,
    FindingCategory.COMPLEXITY,
    FindingCategory.MAINTAINABILITY,
    FindingCategory.STYLE,
    FindingCategory.DEAD_CODE,
)

#: rule id → (name, category, default severity)
RULES: dict[str, tuple[str, FindingCategory, Severity]] = {
    "SW101": ("undefined-name", CORRECT, Severity.HIGH),
    "SW102": ("unused-import", MAINT, Severity.LOW),
    "SW103": ("unused-variable", MAINT, Severity.LOW),
    "SW104": ("redefined-definition", CORRECT, Severity.MEDIUM),
    "SW105": ("mutable-default-argument", CORRECT, Severity.MEDIUM),
    "SW106": ("unresolved-internal-import", CORRECT, Severity.HIGH),
    "SW107": ("call-arity-mismatch", CORRECT, Severity.HIGH),
    "SW201": ("bare-except", CORRECT, Severity.MEDIUM),
    "SW202": ("swallowed-exception", MAINT, Severity.LOW),
    "SW203": ("compare-to-none", STYLE, Severity.LOW),
    "SW204": ("is-with-literal", CORRECT, Severity.MEDIUM),
    "SW205": ("assert-on-tuple", CORRECT, Severity.HIGH),
    "SW206": ("unreachable-code", DEAD, Severity.LOW),
    "SW207": ("duplicate-dict-key", CORRECT, Severity.MEDIUM),
    "SW208": ("return-in-finally", CORRECT, Severity.MEDIUM),
    "SW301": ("dynamic-code-execution", SECURITY, Severity.HIGH),
    "SW302": ("shell-injection-risk", SECURITY, Severity.HIGH),
    "SW303": ("unsafe-deserialization", SECURITY, Severity.HIGH),
    "SW304": ("hardcoded-secret", SECURITY, Severity.MEDIUM),
    "SW305": ("weak-hash", SECURITY, Severity.LOW),
    "SW306": ("insecure-temp-file", SECURITY, Severity.MEDIUM),
    "SW307": ("tls-verification-disabled", SECURITY, Severity.HIGH),
    "SW401": ("high-cyclomatic-complexity", COMPLEX, Severity.MEDIUM),
    "SW402": ("high-cognitive-complexity", COMPLEX, Severity.MEDIUM),
    "SW403": ("long-function", COMPLEX, Severity.LOW),
    "SW404": ("too-many-parameters", COMPLEX, Severity.LOW),
    "SW405": ("deep-nesting", COMPLEX, Severity.LOW),
    "SW406": ("large-class", COMPLEX, Severity.LOW),
    "SW407": ("large-module", COMPLEX, Severity.LOW),
    "SW501": ("missing-docstring", STYLE, Severity.INFO),
    "SW502": ("wildcard-import", STYLE, Severity.LOW),
    "SW503": ("shadowed-builtin", STYLE, Severity.LOW),
}

_SHADOWABLE = frozenset(
    "list dict set str int float bool tuple type id input object sum min max len map filter hash format iter next open range all any bytes dir vars zip slice".split()
)
_SECRET_NAME = re.compile(r"(passw(or)?d|passwd|secret|api[_-]?key|access[_-]?key|private[_-]?key|auth[_-]?token|\btoken\b)", re.I)
_PLACEHOLDER = re.compile(r"^(<.*>|\$\{.*\}|\{\{.*\}\}|x+|\*+|changeme|your[_-].*|todo|none|null|example.*|test.*|dummy.*)$", re.I)
_NOQA = re.compile(r"#\s*noqa(?:\s*:\s*(?P<codes>[A-Za-z0-9_,\s]+))?", re.I)
_MUTABLE_DEFAULT = re.compile(r"^(\[|\{|list\(|dict\(|set\(|bytearray\(|collections\.(defaultdict|deque|OrderedDict)\()")
_SUBPROCESS = frozenset({"subprocess.run", "subprocess.call", "subprocess.check_call", "subprocess.check_output", "subprocess.Popen", "subprocess.getoutput", "subprocess.getstatusoutput"})
_OVERLOAD_LIKE = ("overload", "setter", "getter", "deleter", "register", "singledispatch")
_HARMLESS_DECORATORS = frozenset({"staticmethod", "classmethod", "abstractmethod", "cache", "lru_cache", "cached_property"})


@dataclass(frozen=True, slots=True)
class AnalysisOptions:
    max_cyclomatic: int = 15
    max_cognitive: int = 20
    max_function_lines: int = 100
    max_parameters: int = 7
    max_nesting: int = 4
    max_class_methods: int = 30
    max_module_lines: int = 1500
    check_docstrings: bool = False
    skip_security_in_tests: bool = True
    disabled: frozenset[str] = field(default_factory=frozenset)
    min_severity: Severity = Severity.INFO
    max_findings: int = 2000

    @classmethod
    def from_config(cls, section: Optional[Mapping[str, Any]] = None) -> "AnalysisOptions":
        data = section or {}
        severity = str(data.get("min_severity", "info")).lower()
        return cls(
            max_cyclomatic=opt_int(data, "max_cyclomatic", 15, minimum=1),
            max_cognitive=opt_int(data, "max_cognitive", 20, minimum=1),
            max_function_lines=opt_int(data, "max_function_lines", 100, minimum=1),
            max_parameters=opt_int(data, "max_parameters", 7, minimum=1),
            max_nesting=opt_int(data, "max_nesting", 4, minimum=1),
            max_class_methods=opt_int(data, "max_class_methods", 30, minimum=1),
            max_module_lines=opt_int(data, "max_module_lines", 1500, minimum=1),
            check_docstrings=opt_bool(data, "check_docstrings", False),
            skip_security_in_tests=opt_bool(data, "skip_security_in_tests", True),
            disabled=frozenset(opt_strings(data, "disabled_rules", ())),
            min_severity=Severity(severity) if severity in {s.value for s in Severity} else Severity.INFO,
            max_findings=opt_int(data, "max_findings", 2000, minimum=1),
        )


def _span(node: ast.AST, model: SourceModel) -> SourceSpan:
    line = getattr(node, "lineno", 1)
    end_line = getattr(node, "end_lineno", None) or line
    index = model.line_index
    return SourceSpan(
        line,
        index.char_column(line, getattr(node, "col_offset", 0)),
        end_line,
        index.char_column(end_line, getattr(node, "end_col_offset", None) or getattr(node, "col_offset", 0)),
    )


def _dotted(node: ast.AST) -> Optional[str]:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


class _ModuleAnalysis:
    """Runs every rule against one parsed module."""

    def __init__(self, analyzer: "StaticAnalyzer", model: SourceModel) -> None:
        self.analyzer = analyzer
        self.model = model
        self.options = analyzer.options
        self.findings: list[Finding] = []
        self.aliases = self._alias_map()

    # -- plumbing -----------------------------------------------------------------
    def emit(
        self,
        rule: str,
        message: str,
        span: Optional[SourceSpan],
        *,
        symbol: Optional[str] = None,
        confidence: float = 1.0,
        suggestion: Optional[str] = None,
        severity: Optional[Severity] = None,
        evidence: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if rule in self.options.disabled:
            return
        name, category, default = RULES[rule]
        severity = severity or default
        if severity.rank < self.options.min_severity.rank:
            return
        if category is SECURITY and self.model.source.is_test and self.options.skip_security_in_tests:
            return
        if span is not None and self._suppressed(rule, span.start_line):
            return
        self.findings.append(
            Finding(rule, name, category, severity, message, self.model.path, span, symbol, round(confidence, 3), suggestion, dict(evidence or {}))
        )

    def _suppressed(self, rule: str, line: int) -> bool:
        if not 1 <= line <= self.model.line_count:
            return False
        match = _NOQA.search(self.model.line(line))
        if match is None:
            return False
        codes = match.group("codes")
        if not codes or not codes.strip():
            return True
        return rule in {c.strip().upper() for c in codes.split(",") if c.strip()}

    def _alias_map(self) -> dict[str, str]:
        """Local name → canonical dotted external path (module-scope imports)."""

        mapping: dict[str, str] = {}
        for record in self.model.imports:
            if record.level or record.is_star or not record.bound_name:
                continue
            if record.is_from:
                mapping[record.bound_name] = f"{record.module}.{record.name}"
            elif record.alias:
                mapping[record.alias] = record.module or ""
            else:
                mapping[record.bound_name] = record.bound_name
        return mapping

    def canonical(self, node: ast.AST) -> Optional[str]:
        dotted = _dotted(node)
        if dotted is None:
            return None
        head, _, rest = dotted.partition(".")
        base = self.aliases.get(head, head)
        return f"{base}.{rest}" if rest else base

    # -- driver ---------------------------------------------------------------------
    def run(self) -> list[Finding]:
        self.names_and_imports()
        self.definitions()
        self.complexity()
        self.imports_resolution()
        self.call_arity()
        self.syntax_tree_rules()
        return self.findings

    # -- facts-based rules -------------------------------------------------------------
    def names_and_imports(self) -> None:
        model = self.model
        dynamic_scopes = {scope.qualified_name for scope in model.scopes if scope.dynamic}
        star = model.has_star_import
        loads: dict[tuple[str, str], int] = {}
        for use in model.names:
            if use.context in ("load", "string", "del") and use.binding_scope is not None:
                key = (use.binding_scope, use.name)
                loads[key] = loads.get(key, 0) + 1
            if (
                use.context in ("load", "string")
                and use.resolution is Resolution.UNRESOLVED
                and not star
                and not (use.name == "__class__" and ".<" not in use.scope)
                and not self._dynamic_enclosing(use.scope, dynamic_scopes)
            ):
                self.emit(
                    "SW101",
                    f"name '{use.name}' is not defined",
                    use.span,
                    symbol=model.owner_symbol(use.scope),
                    confidence=0.6 if use.context == "string" else 0.95,
                )

        exports = set(model.exports or ())
        for record in model.imports:
            bound = record.bound_name
            if not bound or (record.is_from and record.module == "__future__"):
                continue
            if model.is_package and record.scope == (model.module_scope.qualified_name if model.module_scope else ""):
                if model.exports is None:
                    continue  # package __init__ re-export convention
            if bound in exports or loads.get((record.scope, bound)):
                continue
            if self._dynamic_enclosing(record.scope, dynamic_scopes):
                continue
            full = f"{record.module or ''}{'.' if record.module and record.name else ''}{record.name or ''}"
            self.emit(
                "SW102",
                f"'{full or bound}' imported but unused",
                record.span,
                confidence=0.6 if record.conditional else 0.95,
                suggestion="remove the import" if not record.conditional else None,
                evidence={"bound_name": bound},
            )

        for scope in model.scopes:
            if scope.kind is not ScopeKind.FUNCTION or scope.dynamic:
                continue
            for binding in scope.bindings:
                if binding.kind is not BindingKind.VARIABLE or binding.name.startswith("_") or binding.name in scope.global_names:
                    continue
                origins = set(binding.origins)
                if not origins or not origins <= {"assign", "annassign", "walrus", "except"}:
                    continue
                if loads.get((scope.qualified_name, binding.name)):
                    continue
                kind = "exception variable" if "except" in origins else "variable"
                self.emit("SW103", f"local {kind} '{binding.name}' is assigned but never used", binding.span, symbol=scope.symbol, confidence=0.9)

            for binding in scope.bindings:
                if binding.name in _SHADOWABLE and binding.kind is not BindingKind.IMPORT and not binding.name.startswith("_"):
                    owner = scope.symbol or scope.qualified_name
                    if scope.kind is ScopeKind.MODULE and binding.kind in (BindingKind.FUNCTION, BindingKind.CLASS):
                        continue
                    self.emit("SW503", f"'{binding.name}' shadows a builtin", binding.span, symbol=owner, confidence=0.7)

        for scope in model.scopes:
            if scope.kind is ScopeKind.MODULE or scope.kind is ScopeKind.CLASS:
                for binding in scope.bindings:
                    if binding.name in _SHADOWABLE and binding.kind is BindingKind.VARIABLE and not binding.name.startswith("_"):
                        self.emit("SW503", f"'{binding.name}' shadows a builtin", binding.span, symbol=scope.symbol, confidence=0.7)

        if not model.is_package:
            for record in model.imports:
                if record.is_star:
                    self.emit("SW502", f"wildcard import from '{record.module or '.' * record.level}' hides name origins", record.span, confidence=1.0)

    @staticmethod
    def _dynamic_enclosing(scope: str, dynamic: set[str]) -> bool:
        return any(scope == d or scope.startswith(d + ".") for d in dynamic)

    def definitions(self) -> None:
        options = self.options
        for symbol in self.model.symbols:
            signature = symbol.signature
            if signature is not None:
                for param in signature.parameters:
                    if param.has_default and param.default and _MUTABLE_DEFAULT.match(param.default):
                        self.emit(
                            "SW105",
                            f"parameter '{param.name}' has a mutable default ({param.default})",
                            symbol.name_span,
                            symbol=symbol.qualified_name,
                            suggestion=f"use None and create the value inside '{symbol.name}'",
                        )
            if options.check_docstrings and symbol.is_public and not symbol.has_docstring and symbol.kind in (SymbolKind.FUNCTION, SymbolKind.METHOD, SymbolKind.CLASS):
                if not symbol.name.startswith("__"):
                    self.emit("SW501", f"public {symbol.kind.value} '{symbol.name}' has no docstring", symbol.name_span, symbol=symbol.qualified_name)

    def complexity(self) -> None:
        o = self.options
        members: dict[str, int] = {}
        for symbol in self.model.symbols:
            if symbol.kind is SymbolKind.METHOD and symbol.parent:
                members[symbol.parent] = members.get(symbol.parent, 0) + 1
            m = symbol.metrics
            if m is None:
                continue
            q = symbol.qualified_name
            if m.cyclomatic > o.max_cyclomatic:
                self.emit("SW401", f"cyclomatic complexity {m.cyclomatic} exceeds {o.max_cyclomatic}", symbol.name_span, symbol=q,
                          severity=Severity.HIGH if m.cyclomatic > 2 * o.max_cyclomatic else None, evidence={"cyclomatic": m.cyclomatic})
            if m.cognitive > o.max_cognitive:
                self.emit("SW402", f"cognitive complexity {m.cognitive} exceeds {o.max_cognitive}", symbol.name_span, symbol=q,
                          severity=Severity.HIGH if m.cognitive > 2 * o.max_cognitive else None, evidence={"cognitive": m.cognitive})
            if m.lines > o.max_function_lines:
                self.emit("SW403", f"function spans {m.lines} lines (limit {o.max_function_lines})", symbol.name_span, symbol=q, evidence={"lines": m.lines})
            effective = m.parameters - (1 if symbol.kind is SymbolKind.METHOD else 0)
            if effective > o.max_parameters:
                self.emit("SW404", f"{effective} parameters (limit {o.max_parameters})", symbol.name_span, symbol=q, evidence={"parameters": effective})
            if m.max_nesting > o.max_nesting:
                self.emit("SW405", f"control-flow nesting depth {m.max_nesting} exceeds {o.max_nesting}", symbol.name_span, symbol=q, evidence={"nesting": m.max_nesting})
        for cls, count in members.items():
            if count > o.max_class_methods:
                symbol = self.model.symbol(cls)
                self.emit("SW406", f"class defines {count} methods (limit {o.max_class_methods})", symbol.name_span if symbol else None, symbol=cls)
        if self.model.line_count > o.max_module_lines:
            self.emit("SW407", f"module has {self.model.line_count} lines (limit {o.max_module_lines})", SourceSpan(1, 0, 1, 0))

    def imports_resolution(self) -> None:
        repo = self.analyzer.index.repo
        index = self.analyzer.index
        for record, resolved in repo.imports_of(self.model.path):
            if record.type_checking or not resolved.internal:
                continue
            if resolved.module is None:
                base = repo.absolute_module(record, self.model) or "."
                self.emit("SW106", f"import of '{base}' does not match any module in the repository", record.module_span or record.span, confidence=0.8 if record.conditional else 0.95)
            elif record.is_from and resolved.attribute and resolved.attribute != "*" and not resolved.submodule:
                if not index.member(resolved.module, resolved.attribute).resolved and index.symbol(f"{resolved.module}.__getattr__") is None:
                    self.emit("SW106", f"cannot import name '{resolved.attribute}' from '{resolved.module}'", record.name_span, confidence=0.7)

    def call_arity(self) -> None:
        index = self.analyzer.index
        model = self.model
        for call in model.calls:
            if call.star_args or call.star_kwargs or not call.callee or call.root_scope is None:
                continue
            target = index.resolve_chain(model, call.root_scope, call.callee)
            if target.kind != "symbol" or not target.name:
                continue
            symbol = index.symbol(target.name)
            if symbol is None:
                continue
            drop = 0
            callee = symbol
            if symbol.kind is SymbolKind.CLASS:
                if symbol.decorators or any(b.startswith("metaclass=") for b in symbol.bases):
                    continue
                bases = index.class_bases(symbol.qualified_name)
                if bases.external or bases.unresolved:
                    continue  # an external base may define the constructor
                init = index.class_member(symbol.qualified_name, "__init__")
                if not init.internal or not init.name:
                    continue
                callee = index.require(init.name)
                drop = 1
            elif symbol.kind is SymbolKind.METHOD:
                names = {d.rsplit(".", 1)[-1] for d in symbol.decorators}
                if names - _HARMLESS_DECORATORS:
                    continue
                if "staticmethod" in names:
                    drop = 0
                elif "classmethod" in names or target.instance or len(call.callee) > 1 and self._via_instance(call.callee, index, model, call.root_scope):
                    drop = 1
                else:
                    continue  # unbound call through the class: receiver passed explicitly, skip
            elif symbol.kind is SymbolKind.FUNCTION:
                if {d.rsplit(".", 1)[-1] for d in symbol.decorators} - _HARMLESS_DECORATORS:
                    continue
            else:
                continue
            problem = self._arity_problem(callee, call, drop)
            if problem:
                self.emit("SW107", f"call to '{callee.qualified_name}' {problem}", call.span, symbol=model.owner_symbol(call.scope), confidence=0.9)

    @staticmethod
    def _via_instance(callee: tuple[str, ...], index: SymbolIndex, model: SourceModel, root_scope: str) -> bool:
        root = index.resolve_name(model, root_scope, callee[0])
        return root.instance and len(callee) == 2

    @staticmethod
    def _arity_problem(symbol: SymbolRecord, call: Any, drop: int) -> Optional[str]:
        signature = symbol.signature
        if signature is None:
            return None
        params = list(signature.parameters)
        positional = [p for p in params if p.kind in (ParameterKind.POSITIONAL_ONLY, ParameterKind.POSITIONAL_OR_KEYWORD)][drop:]
        has_var_args = signature.accepts_var_args
        has_var_kw = signature.accepts_var_kwargs
        if call.positional > len(positional) and not has_var_args:
            return f"passes {call.positional} positional arguments but accepts at most {len(positional)}"
        bound = {p.name for p in positional[: call.positional]}
        names = {p.name for p in params if p.kind in (ParameterKind.POSITIONAL_OR_KEYWORD, ParameterKind.KEYWORD_ONLY)}
        for keyword in call.keywords:
            if keyword in bound:
                return f"gives multiple values for '{keyword}'"
            if keyword not in names and not has_var_kw:
                return f"uses unknown keyword argument '{keyword}'"
        supplied = bound | set(call.keywords)
        missing = [p.name for p in params[drop:] if p.kind in (ParameterKind.POSITIONAL_OR_KEYWORD, ParameterKind.KEYWORD_ONLY, ParameterKind.POSITIONAL_ONLY) and not p.has_default and p.name not in supplied]
        if missing:
            return f"is missing required argument(s): {', '.join(missing)}"
        return None

    # -- syntax-tree rules ------------------------------------------------------------------
    def syntax_tree_rules(self) -> None:
        try:
            tree = self.model.tree()
        except Exception:  # pragma: no cover - parsed models always have a tree
            return
        model = self.model
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                swallowed = all(isinstance(s, ast.Pass) or (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant)) for s in node.body)
                if node.type is None:
                    self.emit("SW201", "bare 'except:' also catches SystemExit and KeyboardInterrupt", _span(node, model), suggestion="catch 'Exception' or a specific type")
                elif swallowed and (_dotted(node.type) in ("Exception", "BaseException")):
                    self.emit("SW202", "broad exception is caught and silently ignored", _span(node, model))
                elif node.type is None or swallowed:
                    pass
                if node.type is None and swallowed:
                    self.emit("SW202", "exception is caught and silently ignored", _span(node, model))
            elif isinstance(node, ast.Compare):
                for op, right in zip(node.ops, node.comparators):
                    if isinstance(op, (ast.Eq, ast.NotEq)) and (isinstance(right, ast.Constant) and right.value is None or isinstance(node.left, ast.Constant) and node.left.value is None):
                        self.emit("SW203", "comparison to None should use 'is' / 'is not'", _span(node, model))
                    if isinstance(op, (ast.Is, ast.IsNot)):
                        for side in (node.left, right):
                            if isinstance(side, ast.Constant) and isinstance(side.value, (str, bytes, int, float)) and not isinstance(side.value, bool):
                                self.emit("SW204", "'is' comparison with a literal depends on object identity", _span(node, model), suggestion="use '=='")
                                break
            elif isinstance(node, ast.Assert):
                if isinstance(node.test, ast.Tuple) and node.test.elts:
                    self.emit("SW205", "assert on a non-empty tuple is always true", _span(node, model))
            elif isinstance(node, ast.Dict):
                seen: dict[Any, int] = {}
                for key in node.keys:
                    if isinstance(key, ast.Constant):
                        marker = (type(key.value).__name__, key.value)
                        if marker in seen:
                            self.emit("SW207", f"duplicate dictionary key {key.value!r}", _span(key, model))
                        seen[marker] = 1
            elif isinstance(node, ast.Try) and node.finalbody:
                for stmt in node.finalbody:
                    if isinstance(stmt, ast.Return):
                        self.emit("SW208", "return in 'finally' swallows exceptions", _span(stmt, model))
            elif isinstance(node, ast.Call):
                self._call_security(node)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                self._secret_assignment(node)
            body = getattr(node, "body", None)
            if isinstance(body, list):
                self._unreachable_and_redefined(body, node)
            for field_name in ("orelse", "finalbody"):
                block = getattr(node, field_name, None)
                if isinstance(block, list) and block and isinstance(block[0], ast.stmt):
                    self._unreachable_and_redefined(block, node)

    def _unreachable_and_redefined(self, block: list[ast.stmt], parent: ast.AST) -> None:
        model = self.model
        terminated = False
        defined: dict[str, ast.stmt] = {}
        for stmt in block:
            if not isinstance(stmt, ast.stmt):
                return
            if terminated and not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self.emit("SW206", "code after return/raise/break/continue is unreachable", _span(stmt, model))
                break
            if isinstance(stmt, (ast.Return, ast.Raise, ast.Break, ast.Continue)):
                terminated = True
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                decorators = [(_dotted(d.func if isinstance(d, ast.Call) else d) or "") for d in stmt.decorator_list]
                if any(any(tag in item for tag in _OVERLOAD_LIKE) for item in decorators):
                    defined.pop(stmt.name, None)
                    continue
                previous = defined.get(stmt.name)
                if previous is not None:
                    self.emit("SW104", f"'{stmt.name}' redefines the definition at line {previous.lineno}", _span(stmt, model), symbol=stmt.name)
                defined[stmt.name] = stmt
            elif isinstance(stmt, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
                for target in targets:
                    if isinstance(target, ast.Name):
                        defined.pop(target.id, None)

    def _call_security(self, node: ast.Call) -> None:
        model = self.model
        name = self.canonical(node.func)
        if name is None:
            return
        span = _span(node, model)
        keywords = {kw.arg: kw.value for kw in node.keywords if kw.arg}

        def is_true(value: Optional[ast.AST]) -> bool:
            return isinstance(value, ast.Constant) and value.value is True

        def is_false(value: Optional[ast.AST]) -> bool:
            return isinstance(value, ast.Constant) and value.value is False

        if name in ("eval", "exec") or name in ("builtins.eval", "builtins.exec"):
            literal = node.args and isinstance(node.args[0], ast.Constant)
            self.emit("SW301", f"{name.rsplit('.', 1)[-1]}() executes dynamically built code", span, confidence=0.5 if literal else 0.9,
                      suggestion="use ast.literal_eval or an explicit dispatch table")
        elif name in _SUBPROCESS and is_true(keywords.get("shell")):
            self.emit("SW302", f"{name} with shell=True is vulnerable to shell injection", span, suggestion="pass an argument list and omit shell=True")
        elif name in ("os.system", "os.popen"):
            self.emit("SW302", f"{name} runs its argument through a shell", span, suggestion="use subprocess with an argument list")
        elif name in ("pickle.load", "pickle.loads", "marshal.load", "marshal.loads", "shelve.open", "dill.load", "dill.loads", "cPickle.load", "cPickle.loads"):
            self.emit("SW303", f"{name} can execute arbitrary code when given untrusted data", span, confidence=0.8)
        elif name in ("yaml.load", "yaml.load_all"):
            loader = keywords.get("Loader")
            safe = loader is not None and "Safe" in (_dotted(loader) or "")
            if not safe:
                self.emit("SW303", f"{name} without a safe Loader can construct arbitrary objects", span, suggestion="use yaml.safe_load")
        elif name in ("hashlib.md5", "hashlib.sha1"):
            self.emit("SW305", f"{name.split('.')[-1]} is a weak hash for security purposes", span, confidence=0.6)
        elif name == "tempfile.mktemp":
            self.emit("SW306", "tempfile.mktemp is race-prone", span, suggestion="use tempfile.mkstemp or NamedTemporaryFile")
        elif name.split(".")[0] in ("requests", "httpx") and is_false(keywords.get("verify")):
            self.emit("SW307", "TLS certificate verification is disabled", span)

    def _secret_assignment(self, node: ast.Assign | ast.AnnAssign) -> None:
        value = node.value
        if not (isinstance(value, ast.Constant) and isinstance(value.value, str)) or len(value.value) < 6 or _PLACEHOLDER.match(value.value.strip()):
            return
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            label = target.id if isinstance(target, ast.Name) else target.attr if isinstance(target, ast.Attribute) else None
            if label and _SECRET_NAME.search(label) and " " not in value.value.strip():
                self.emit("SW304", f"'{label}' looks like a hard-coded credential", _span(node, self.model), confidence=0.6,
                          suggestion="load secrets from the environment or a secret store")
                return


class StaticAnalyzer:
    def __init__(self, index: SymbolIndex, options: Optional[AnalysisOptions] = None) -> None:
        self.index = index
        self.options = options or AnalysisOptions()

    def analyze_model(self, model: SourceModel) -> list[Finding]:
        if model.status is not ParseStatus.OK:
            return [
                Finding("SW000", "parse-failure", CORRECT, Severity.HIGH, d.message, model.path, SourceSpan(d.line, d.column, d.line, d.column), None, 1.0)
                for d in model.diagnostics
            ]
        seen: set[tuple[str, Optional[tuple[int, int, int, int]], str]] = set()
        unique: list[Finding] = []
        for finding in _ModuleAnalysis(self, model).run():
            key = (finding.rule_id, finding.span.as_tuple() if finding.span else None, finding.message)
            if key not in seen:
                seen.add(key)
                unique.append(finding)
        return unique

    def analyze(self, paths: Optional[Iterable[str]] = None, *, rules: Optional[Iterable[str]] = None) -> StaticAnalysisReport:
        repo = self.index.repo
        selected = [p for p in (paths if paths is not None else repo.paths()) if repo.get(p) is not None]
        wanted = frozenset(r.upper() for r in rules) if rules else None
        findings: list[Finding] = []
        metrics = {}
        for path in sorted(selected):
            model = repo.model(path)
            if model.status is ParseStatus.UNSUPPORTED:
                continue
            findings.extend(self.analyze_model(model))
            for symbol in model.symbols:
                if symbol.metrics is not None:
                    metrics[symbol.qualified_name] = symbol.metrics
        if wanted is not None:
            findings = [f for f in findings if f.rule_id in wanted]
        findings.sort(key=lambda f: (-f.severity.rank, f.path, f.span.start_line if f.span else 0, f.rule_id))
        kept, omitted = truncate_items(findings, self.options.max_findings)
        by_rule: dict[str, int] = {}
        by_severity: dict[str, int] = {}
        by_category: dict[str, int] = {}
        for f in findings:
            by_rule[f.rule_id] = by_rule.get(f.rule_id, 0) + 1
            by_severity[f.severity.value] = by_severity.get(f.severity.value, 0) + 1
            by_category[f.category.value] = by_category.get(f.category.value, 0) + 1
        return StaticAnalysisReport(
            findings=tuple(kept),
            files_analyzed=len(selected),
            summary={"total": len(findings), "by_rule": dict(sorted(by_rule.items())), "by_severity": by_severity, "by_category": by_category},
            function_metrics=metrics,
            truncated=omitted,
        )


__all__ = ["AnalysisOptions", "RULES", "StaticAnalyzer"]
