"""Behaviour-preserving refactorings that produce *proposals*, never writes.

Each refactoring computes :class:`TextEdit` objects from resolved facts (the
symbol index), hands them to the :class:`PatchBuilder`, and attaches evidence:
sites that could not be rewritten safely (``unresolved_sites``), risks
(public API, dynamic name use), and a *regression check* that re-analyzes the
touched files and their importers and reports newly introduced high-severity
findings.  Supported: rename, remove-unused-imports, remove-dead-code.
"""

from __future__ import annotations

__version__ = "2.4.0"

import ast

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional

from ..models.repository_model import RepositoryModel, RepositorySnapshot
from ..models.source_model import SourceModel
from ..semantics.interface_compatibility import public_api
from ..semantics.symbol_index import SymbolIndex
from ..software_types import (
    DeadCodeReport,
    Finding,
    ImportRecord,
    ParseStatus,
    RefactoringKind,
    ScopeKind,
    Severity,
    SourceSpan,
    SymbolKind,
    SymbolRecord,
    TextEdit,
)
from ..utils.software_errors import SoftwareNameConflictError, SoftwareRefactoringError
from ..utils.software_helpers import opt_bool, opt_float, opt_int, require_identifier
from .patch_builder import PatchBuilder, PatchResult, Validator
from .static_analysis import AnalysisOptions, StaticAnalyzer
from logs.logger import get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Software Refactoring")

_REGRESSION_RULES = ("SW101", "SW104", "SW106", "SW107")
_NEUTRAL_BASES = frozenset({"object", "ABC", "abc.ABC", "Protocol", "typing.Protocol", "Generic", "typing.Generic"})


@dataclass(frozen=True, slots=True)
class RefactoringOptions:
    min_dead_code_confidence: float = 0.8
    min_import_confidence: float = 0.9
    regression_check: bool = True
    rename_family: bool = True
    max_edits: int = 5000

    @classmethod
    def from_config(cls, section: Optional[Mapping[str, Any]] = None) -> "RefactoringOptions":
        data = section or {}
        return cls(
            min_dead_code_confidence=opt_float(data, "min_dead_code_confidence", 0.8, minimum=0.0, maximum=1.0),
            min_import_confidence=opt_float(data, "min_import_confidence", 0.9, minimum=0.0, maximum=1.0),
            regression_check=opt_bool(data, "regression_check", True),
            rename_family=opt_bool(data, "rename_family", True),
            max_edits=opt_int(data, "max_edits", 5000, minimum=1),
        )


def line_start(model: SourceModel, line: int) -> int:
    """Offset of the start of ``line`` (``len(text)`` for the line after the last)."""

    return len(model.source.text) if line > model.line_count else model.line_index.offset(line, 0)


def span_offsets(model: SourceModel, span: SourceSpan) -> tuple[int, int]:
    index = model.line_index
    return index.offset(span.start_line, span.start_col), index.offset(span.end_line, span.end_col)


def make_edit(model: SourceModel, span: SourceSpan, replacement: str, description: str) -> TextEdit:
    start, end = span_offsets(model, span)
    return TextEdit(model.path, start, end, replacement, description)


class BlockInfo:
    """Which statements are the only statement of their block (for ``pass`` insertion)."""

    def __init__(self, model: SourceModel) -> None:
        self.sole: set[tuple[int, int]] = set()
        try:
            tree = model.tree()
        except Exception:
            return
        for node in ast.walk(tree):
            for name in ("body", "orelse", "finalbody"):
                block = getattr(node, name, None)
                if isinstance(block, list) and len(block) == 1 and isinstance(block[0], ast.stmt):
                    stmt = block[0]
                    self.sole.add((stmt.lineno, stmt.col_offset))
                    if getattr(stmt, "decorator_list", None):
                        self.sole.add((stmt.decorator_list[0].lineno, stmt.decorator_list[0].col_offset - 1))

    def is_sole(self, line: int, byte_col: int) -> bool:
        return (line, byte_col) in self.sole


def import_removal_edits(model: SourceModel, blocks: BlockInfo, records: Sequence[ImportRecord], drop: Sequence[ImportRecord]) -> Optional[list[TextEdit]]:
    """Edits removing ``drop`` from one import statement (``None`` if it cannot be done safely)."""
    text = model.source.text
    statement = records[0].statement_span
    kept = [r for r in records if r not in drop]
    if not kept:
        start_line, end_line = statement.start_line, statement.end_line
        first = model.line(start_line)
        before = first[: statement.start_col]
        after_text = model.line(end_line)[statement.end_col:]
        if before.strip() or (after_text.strip() and not after_text.strip().startswith("#")):
            return None  # shares its line with other code
        begin, end = line_start(model, start_line), line_start(model, end_line + 1)
        if blocks.is_sole(start_line, len(before.encode("utf-8"))):
            return [TextEdit(model.path, begin, end, before + "pass\n", "replace sole import with pass")]
        return [TextEdit(model.path, begin, end, "", "remove unused import statement")]
    ordered = sorted(records, key=lambda r: r.span.as_tuple())
    region_start = span_offsets(model, ordered[0].span)[0]
    region_end = span_offsets(model, ordered[-1].span)[1]
    region = text[region_start:region_end]
    if "#" in region:
        return None
    gap_start = span_offsets(model, ordered[0].span)[1]
    gap_end = span_offsets(model, ordered[1].span)[0]
    separator = text[gap_start:gap_end] or ", "
    pieces = [model.segment(r.span) for r in ordered if r in kept]
    return [TextEdit(model.path, region_start, region_end, separator.join(pieces), "drop unused names from import")]


def regression_validator(index: SymbolIndex, parser: Any, *, cache: Any = None) -> Validator:
    """Validator comparing correctness findings before/after on touched files and importers."""

    def validate(old: RepositorySnapshot, new: RepositorySnapshot, changed: tuple[str, ...]) -> tuple[Mapping[str, Any], Sequence[str], Sequence[str]]:
        before_repo = index.repo
        modules = {before_repo.module_name_of(p) for p in changed} - {None}
        importers = {path for path, record, resolved in before_repo.all_imports() if resolved.target_module in modules or resolved.module in modules}
        scope = sorted({p for p in (*changed, *importers) if p in new and p in old})
        new_repo = RepositoryModel.build(new, parser, cache=cache, previous=before_repo)
        new_index = SymbolIndex(new_repo, index.options)
        options = AnalysisOptions(disabled=frozenset())
        rules = _REGRESSION_RULES
        before = StaticAnalyzer(index, options).analyze(scope, rules=rules).findings
        after = StaticAnalyzer(new_index, options).analyze(scope, rules=rules).findings

        def key(f: Finding) -> tuple[str, str, str]:
            return (f.rule_id, f.path, f.message)

        remaining = Counter(key(f) for f in before)
        introduced: list[Finding] = []
        for finding in after:
            k = key(finding)
            if remaining[k] > 0:
                remaining[k] -= 1
            else:
                introduced.append(finding)
        resolved = sum(remaining.values())
        high = [f for f in introduced if f.severity.rank >= Severity.HIGH.rank]
        risks = [f"regression check: {len(high)} new high-severity finding(s) (e.g. {high[0].rule_id} {high[0].path}:{high[0].span.start_line if high[0].span else '?'} {high[0].message})"] if high else []
        data = {
            "static_regressions": {
                "scope_files": len(scope),
                "introduced": [f.to_dict() for f in introduced[:20]],
                "introduced_count": len(introduced),
                "resolved_count": resolved,
            },
            "confidence_factor": max(0.3, 0.85 ** len(high)),
        }
        return data, risks, []

    return validate


class RefactoringEngine:
    def __init__(self, index: SymbolIndex, builder: PatchBuilder, options: Optional[RefactoringOptions] = None, *, cache: Any = None, parent_patch_id: Optional[str] = None) -> None:
        self.index = index
        self.builder = builder
        self.options = options or RefactoringOptions()
        self.cache = cache
        self.parent_patch_id = parent_patch_id

    @property
    def repo(self) -> RepositoryModel:
        return self.index.repo

    def _validator(self) -> Optional[Validator]:
        return regression_validator(self.index, self.builder.parser, cache=self.cache) if self.options.regression_check else None

    # ---------------------------------------------------------------------------- rename
    def _family(self, qname: str) -> list[str]:
        symbol = self.index.require(qname)
        if symbol.kind is not SymbolKind.METHOD or not self.options.rename_family:
            return [qname]
        family = {qname}
        stack = [qname]
        while stack:
            current = stack.pop()
            for other in (*self.index.overridden(current), *self.index.overriders(current)):
                if other not in family:
                    family.add(other)
                    stack.append(other)
        return sorted(family)

    def _check_conflicts(self, targets: Sequence[str], new_name: str) -> None:
        index = self.index
        for qname in targets:
            symbol = index.require(qname)
            parent = symbol.parent or ""
            if f"{parent}.{new_name}" in {s.qualified_name for s in index.members(parent)} or index.symbol(f"{parent}.{new_name}") is not None:
                raise SoftwareNameConflictError("a symbol with the new name already exists in the same scope", context={"symbol": qname, "conflict": f"{parent}.{new_name}"})
            if symbol.kind is SymbolKind.METHOD and symbol.parent:
                related = {*index.linearization(symbol.parent)[0], *index.subclasses(symbol.parent, transitive=True)}
                for cls in related:
                    if index.symbol(f"{cls}.{new_name}") is not None:
                        raise SoftwareNameConflictError("the new name collides with a member in the class hierarchy", context={"symbol": qname, "conflict": f"{cls}.{new_name}"})
            else:
                model = index.model_of(qname)
                scope = model.scope(parent) if model else None
                if scope is not None and scope.binding(new_name) is not None:
                    raise SoftwareNameConflictError("the new name is already bound in the defining scope", context={"symbol": qname, "scope": parent})

    def rename_symbol(self, qualified_name: str, new_name: str, *, include_heuristic: bool = False) -> PatchResult:
        index, repo = self.index, self.repo
        symbol = index.require(qualified_name)
        if symbol.kind not in (SymbolKind.CLASS, SymbolKind.FUNCTION, SymbolKind.METHOD, SymbolKind.VARIABLE):
            raise SoftwareRefactoringError("only classes, functions, methods and variables can be renamed", context={"kind": symbol.kind.value})
        new_name = require_identifier(new_name, "new_name")
        if new_name == symbol.name:
            raise SoftwareRefactoringError("new name equals the current name", context={"name": new_name})
        if symbol.name.startswith("__") and symbol.name.endswith("__"):
            raise SoftwareRefactoringError("dunder names are protocol hooks and cannot be renamed", context={"name": symbol.name})
        targets = self._family(qualified_name)
        self._check_conflicts(targets, new_name)

        old_name = symbol.name
        edits: dict[tuple[str, int, int], TextEdit] = {}
        risks: list[str] = []
        warnings: list[str] = []
        unresolved = 0
        capture: list[str] = []

        def add(model: SourceModel, span: SourceSpan, replacement: str, description: str) -> None:
            edit = make_edit(model, span, replacement, description)
            edits.setdefault((edit.path, edit.start, edit.end), edit)

        public = public_api(repo)
        for target in targets:
            record = index.require(target)
            model = index.model_of(target)
            assert model is not None
            add(model, record.name_span, new_name, f"rename declaration of {target}")
            if target in public:
                risks.append(f"'{target}' is part of the public API: callers outside the repository will break")
            bases = index.class_bases(record.parent) if record.kind is SymbolKind.METHOD and record.parent else None
            if bases is not None and any(b not in _NEUTRAL_BASES for b in (*bases.external, *bases.unresolved)):
                risks.append(f"'{record.parent}' derives from external/unresolved base(s); the method may implement an interface defined elsewhere")
            for ref in index.references(target, include_heuristic=include_heuristic):
                ref_model = repo.model(ref.path)
                text = ref_model.segment(ref.span)
                if ref.kind == "export":
                    if len(text) >= 2 and text[1:-1] == old_name:
                        add(ref_model, ref.span, text[0] + new_name + text[-1], f"update __all__ entry for {target}")
                    continue
                if text != old_name:
                    continue  # reached through an import alias: local name is unaffected
                add(ref_model, ref.span, new_name, f"rename {ref.kind} reference to {target}")
                if ref.kind in ("name", "call") and ref.exact and ref.context != "string":
                    found = index.lookup(ref_model, ref.scope, new_name)
                    if found is not None:
                        capture.append(f"{ref.path}:{ref.span.start_line}")
            if not include_heuristic:
                unresolved += len(index.references(target, include_heuristic=True)) - len(index.references(target))
        if capture:
            raise SoftwareNameConflictError("the new name would capture or be shadowed at reference sites", context={"sites": capture[:10], "count": len(capture)})
        if unresolved:
            warnings.append(f"{unresolved} attribute access(es) could only be matched by name and were left unchanged")

        dynamic_modules = sorted(m.path for m in repo.parsed() if old_name in m.string_names or qualified_name in m.string_names)
        if dynamic_modules:
            unresolved += len(dynamic_modules)
            warnings.append(f"'{old_name}' appears as a string literal in {len(dynamic_modules)} file(s) ({', '.join(dynamic_modules[:3])}); dynamic lookups are not rewritten")
        if len(edits) > self.options.max_edits:
            raise SoftwareRefactoringError("rename would produce too many edits", context={"edits": len(edits), "limit": self.options.max_edits})

        confidence = 0.97 - min(0.5, 0.03 * unresolved) - (0.15 if any("external" in r for r in risks) else 0.0)
        return self.builder.build(
            self.repo.snapshot,
            kind=RefactoringKind.RENAME_SYMBOL,
            intent=f"rename {qualified_name} to {new_name}",
            edits=edits.values(),
            risks=risks,
            warnings=warnings,
            unresolved_sites=unresolved,
            confidence=confidence,
            metadata={"symbol": qualified_name, "new_name": new_name, "family": targets, "edits": len(edits)},
            parent_patch_id=self.parent_patch_id,
            validator=self._validator(),
        )

    # ---------------------------------------------------------------- unused imports
    def remove_unused_imports(self, paths: Optional[Iterable[str]] = None) -> PatchResult:
        analyzer = StaticAnalyzer(self.index, AnalysisOptions())
        selected = list(paths) if paths is not None else list(self.repo.paths())
        report = analyzer.analyze(selected, rules=["SW102"])
        by_path: dict[str, list[Finding]] = {}
        for finding in report.findings:
            if finding.confidence >= self.options.min_import_confidence and finding.span is not None:
                by_path.setdefault(finding.path, []).append(finding)
        edits: list[TextEdit] = []
        warnings: list[str] = []
        removed = 0
        for path, findings in sorted(by_path.items()):
            model = self.repo.model(path)
            blocks = BlockInfo(model)
            unused_spans = {f.span.as_tuple() for f in findings if f.span}
            statements: dict[tuple[int, int, int, int], list[ImportRecord]] = {}
            for record in model.imports:
                statements.setdefault(record.statement_span.as_tuple(), []).append(record)
            for key, records in statements.items():
                drop = [r for r in records if r.span.as_tuple() in unused_spans]
                if not drop:
                    continue
                produced = import_removal_edits(model, blocks, records, drop)
                if produced is None:
                    warnings.append(f"{path}:{records[0].statement_span.start_line}: unused import left in place (shares a line or contains comments)")
                    continue
                edits.extend(produced)
                removed += len(drop)
        return self.builder.build(
            self.repo.snapshot,
            kind=RefactoringKind.REMOVE_UNUSED_IMPORTS,
            intent=f"remove {removed} unused import(s)",
            edits=edits,
            warnings=warnings,
            confidence=0.95,
            risks=("imports with side effects (plugin registration) would be dropped; modules using wildcard re-export are skipped",) if removed else (),
            metadata={"removed": removed, "files": len(by_path)},
            parent_patch_id=self.parent_patch_id,
            validator=self._validator(),
        )

    # ------------------------------------------------------------------- dead code
    def remove_dead_code(self, report: DeadCodeReport, *, min_confidence: Optional[float] = None, symbols: Optional[Iterable[str]] = None) -> PatchResult:
        threshold = self.options.min_dead_code_confidence if min_confidence is None else min_confidence
        only = set(symbols) if symbols is not None else None
        index, repo = self.index, self.repo
        chosen: dict[str, SymbolRecord] = {}
        warnings: list[str] = []
        for item in report.items:
            if item.confidence < threshold or (only is not None and item.qualified_name not in only):
                continue
            symbol = index.symbol(item.qualified_name)
            if symbol is None or symbol.kind not in (SymbolKind.FUNCTION, SymbolKind.METHOD, SymbolKind.CLASS):
                continue
            chosen[symbol.qualified_name] = symbol

        def within(path: str, line: int) -> bool:
            for q, s in chosen.items():
                if index.path_of(q) == path and s.full_span.start_line <= line <= s.span.end_line:
                    return True
            return False

        changed = True
        while changed:
            changed = False
            for q in list(chosen):
                model = index.model_of(q)
                exported = model is not None and model.exports is not None and chosen[q].name in model.exports and chosen[q].parent == model.module_name
                outside = [r for r in index.references(q, include_heuristic=True) if not within(r.path, r.span.start_line)]
                if exported or outside:
                    why = "listed in __all__" if exported else f"still referenced at {outside[0].path}:{outside[0].span.start_line}"
                    warnings.append(f"{q} kept: {why}")
                    del chosen[q]
                    changed = True
        # drop nested symbols whose container is removed as well
        ranges = sorted(((index.path_of(q) or "", s.full_span.start_line, s.span.end_line, q) for q, s in chosen.items()))
        keep: list[tuple[str, int, int, str]] = []
        for item in ranges:
            if keep and keep[-1][0] == item[0] and item[1] >= keep[-1][1] and item[2] <= keep[-1][2]:
                continue
            keep.append(item)
        edits: list[TextEdit] = []
        blocks_cache: dict[str, BlockInfo] = {}
        for path, start_line, end_line, q in keep:
            model = repo.model(path)
            blocks = blocks_cache.setdefault(path, BlockInfo(model))
            symbol = chosen[q]
            begin, end = line_start(model, start_line), line_start(model, end_line + 1)
            text = model.source.text
            def_line = model.line(symbol.span.start_line)
            indent = def_line[: len(def_line) - len(def_line.lstrip())]
            byte_col = len(indent.encode("utf-8"))
            sole = blocks.is_sole(symbol.span.start_line, byte_col) or blocks.is_sole(start_line, byte_col)
            if sole:
                edits.append(TextEdit(path, begin, end, indent + "pass\n", f"replace dead {symbol.kind.value} {symbol.name} with pass"))
                continue
            preceded_blank = begin == 0 or text[max(0, begin - 2):begin] in ("\n\n", "\r\n\r\n") or text[:begin].endswith("\n\n")
            while preceded_blank and end < len(text) and text[end:end + 1] in ("\n", "\r"):
                end = text.find("\n", end) + 1 or len(text)
            edits.append(TextEdit(path, begin, end, "", f"remove dead {symbol.kind.value} {symbol.name}"))
        return self.builder.build(
            repo.snapshot,
            kind=RefactoringKind.REMOVE_DEAD_CODE,
            intent=f"remove {len(keep)} unreachable definition(s)",
            edits=edits,
            warnings=[*warnings, *(["run remove-unused-imports afterwards: removed code may have been the only user of some imports"] if keep else [])],
            risks=("dead-code detection is static: dynamic dispatch, reflection and external callers are invisible to it",) if keep else (),
            confidence=min([i.confidence for i in report.items if i.qualified_name in chosen] or [1.0]) * 0.95,
            metadata={"removed": [k[3] for k in keep], "kept": len(warnings)},
            parent_patch_id=self.parent_patch_id,
            validator=self._validator(),
        )


__all__ = ["BlockInfo", "RefactoringEngine", "RefactoringOptions", "import_removal_edits", "line_start", "make_edit", "regression_validator", "span_offsets"]
