"""Declarative API migration (deprecations, renames, signature changes).

Rules (:class:`MigrationRule`) are matched against *resolved* names, not text:
``import collections as c`` / ``from collections import Mapping as M`` both
match a rule for ``collections.Mapping``.  Rules are applied one after the
other on an evolving in-memory model, so later rules see earlier results, and
the outcome is a single combined :class:`PatchProposal`.

Supported kinds
---------------
``rename_symbol``   ``old`` → ``new`` dotted names; a changed module part moves
                    the import (and rewrites ``module.attr`` accesses).
``rename_module``   ``import old`` / ``from old[.sub] import x`` → ``new``.
``rename_keyword``  call keyword ``keyword`` → ``new`` for calls of ``old``.
``remove_keyword``  drop ``keyword=...`` from calls of ``old``.
``add_keyword``     add ``keyword=value`` to calls of ``old`` lacking it.
"""

from __future__ import annotations

__version__ = "2.4.0"

import ast

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional

from ..models.repository_model import RepositoryModel
from ..models.source_model import SourceModel
from ..semantics.symbol_index import IndexOptions, SymbolIndex
from ..software_types import (
    CallSite,
    ImportRecord,
    MigrationKind,
    MigrationRule,
    RefactoringKind,
    SourceSpan,
    TextEdit,
)
from ..utils.software_errors import SoftwareRefactoringError, SoftwareRequestError
from ..utils.software_helpers import is_valid_identifier, opt_int
from .patch_builder import PatchBuilder, PatchResult, apply_edits
from .refactoring import BlockInfo, import_removal_edits, line_start, make_edit, regression_validator, span_offsets
from logs.logger import get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Software API Migration")

_KINDS_NEEDING_KEYWORD = (MigrationKind.RENAME_KEYWORD, MigrationKind.REMOVE_KEYWORD, MigrationKind.ADD_KEYWORD)


@dataclass(frozen=True, slots=True)
class MigrationOptions:
    max_rules: int = 100
    regression_check: bool = True

    @classmethod
    def from_config(cls, section: Optional[Mapping[str, Any]] = None) -> "MigrationOptions":
        data = section or {}
        return cls(max_rules=opt_int(data, "max_rules", 100, minimum=1), regression_check=bool(data.get("regression_check", True)))


def parse_rules(raw: Iterable[Mapping[str, Any] | MigrationRule]) -> list[MigrationRule]:
    """Coerce request payloads into validated :class:`MigrationRule` objects."""

    rules: list[MigrationRule] = []
    for item in raw:
        if isinstance(item, MigrationRule):
            rule = item
        else:
            if not isinstance(item, Mapping):
                raise SoftwareRequestError("a migration rule must be a mapping", context={"type": type(item).__name__})
            try:
                kind = MigrationKind(str(item.get("kind", "")).lower())
            except ValueError as exc:
                raise SoftwareRequestError("unknown migration kind", context={"kind": item.get("kind"), "kinds": [k.value for k in MigrationKind]}, cause=exc) from exc
            rule = MigrationRule(kind, str(item.get("old", "")).strip(), _opt(item.get("new")), _opt(item.get("keyword")), _opt(item.get("value")), str(item.get("description", "")))
        validate_rule(rule)
        rules.append(rule)
    return rules


def _opt(value: Any) -> Optional[str]:
    return None if value is None else str(value).strip() or None


def validate_rule(rule: MigrationRule) -> None:
    def dotted_ok(text: Optional[str]) -> bool:
        return bool(text) and all(is_valid_identifier(part) for part in text.split("."))  # type: ignore[union-attr]

    if not dotted_ok(rule.old):
        raise SoftwareRequestError("rule 'old' must be a dotted identifier path", context={"old": rule.old})
    if rule.kind in (MigrationKind.RENAME_SYMBOL, MigrationKind.RENAME_MODULE):
        if not dotted_ok(rule.new) or rule.new == rule.old:
            raise SoftwareRequestError("rule 'new' must be a different dotted identifier path", context={"old": rule.old, "new": rule.new})
        if rule.kind is MigrationKind.RENAME_SYMBOL and "." not in rule.old:
            raise SoftwareRequestError("rename_symbol needs a qualified name (module.name)", context={"old": rule.old})
    if rule.kind in _KINDS_NEEDING_KEYWORD and not is_valid_identifier(rule.keyword):
        raise SoftwareRequestError("keyword rules need a valid 'keyword'", context={"keyword": rule.keyword})
    if rule.kind is MigrationKind.RENAME_KEYWORD and not is_valid_identifier(rule.new):
        raise SoftwareRequestError("rename_keyword needs a valid 'new' keyword name", context={"new": rule.new})
    if rule.kind is MigrationKind.ADD_KEYWORD:
        try:
            ast.parse(rule.value or "", mode="eval")
        except SyntaxError as exc:
            raise SoftwareRequestError("add_keyword 'value' must be a Python expression", context={"value": rule.value}, cause=exc) from exc


class _CallNodes:
    """Lazy map from call span to its ``ast.Call`` node."""

    def __init__(self, model: SourceModel) -> None:
        self.model = model
        self._map: Optional[dict[tuple[int, int, int, int], ast.Call]] = None

    def get(self, span: SourceSpan) -> Optional[ast.Call]:
        if self._map is None:
            index = self.model.line_index
            table: dict[tuple[int, int, int, int], ast.Call] = {}
            for node in ast.walk(self.model.tree()):
                if isinstance(node, ast.Call):
                    end_line = node.end_lineno or node.lineno
                    key = (
                        node.lineno,
                        index.char_column(node.lineno, node.col_offset),
                        end_line,
                        index.char_column(end_line, node.end_col_offset or 0),
                    )
                    table[key] = node
            self._map = table
        return self._map.get(span.as_tuple())


class ApiMigrator:
    def __init__(self, index: SymbolIndex, builder: PatchBuilder, options: Optional[MigrationOptions] = None, *, cache: Any = None, parent_patch_id: Optional[str] = None) -> None:
        self.index = index
        self.builder = builder
        self.options = options or MigrationOptions()
        self.cache = cache
        self.parent_patch_id = parent_patch_id

    # -- driver ---------------------------------------------------------------------------
    def migrate(self, rules: Sequence[MigrationRule]) -> PatchResult:
        if not rules:
            raise SoftwareRequestError("no migration rules supplied")
        if len(rules) > self.options.max_rules:
            raise SoftwareRequestError("too many migration rules", context={"limit": self.options.max_rules})
        original = self.index.repo
        repo, index = original, self.index
        texts: dict[str, str] = {}
        counts: list[dict[str, Any]] = []
        warnings: list[str] = []
        unresolved = 0
        for rule in rules:
            edits, notes, skipped = self._edits_for(rule, index)
            warnings.extend(notes)
            unresolved += skipped
            counts.append({"kind": rule.kind.value, "old": rule.old, "new": rule.new, "keyword": rule.keyword, "sites": len(edits)})
            if not edits:
                continue
            per_file: dict[str, list[TextEdit]] = {}
            for edit in edits:
                per_file.setdefault(edit.path, []).append(edit)
            changes = {path: apply_edits(repo.snapshot.file(path).text, items) for path, items in per_file.items()}
            texts.update(changes)
            repo = repo.with_changes(changes, self.builder.parser, cache=self.cache)
            index = SymbolIndex(repo, index.options)
        validator = regression_validator(self.index, self.builder.parser, cache=self.cache) if self.options.regression_check else None
        if not texts:
            warnings.append("no site matched any rule")
        return self.builder.build(
            original.snapshot,
            kind=RefactoringKind.MIGRATE_API,
            intent="migrate API: " + "; ".join(r.description or f"{r.kind.value} {r.old}" for r in rules)[:240],
            replacements=texts,
            warnings=warnings,
            risks=("migration is name-based on resolved references; behavioural differences between old and new APIs are not checked",),
            unresolved_sites=unresolved,
            confidence=max(0.3, 0.9 - 0.03 * unresolved),
            metadata={"rules": counts},
            parent_patch_id=self.parent_patch_id,
            validator=validator,
        )

    def _edits_for(self, rule: MigrationRule, index: SymbolIndex) -> tuple[list[TextEdit], list[str], int]:
        handler = {
            MigrationKind.RENAME_SYMBOL: self._rename_symbol,
            MigrationKind.RENAME_MODULE: self._rename_module,
            MigrationKind.RENAME_KEYWORD: self._keywords,
            MigrationKind.REMOVE_KEYWORD: self._keywords,
            MigrationKind.ADD_KEYWORD: self._keywords,
        }[rule.kind]
        edits: dict[tuple[str, int, int], TextEdit] = {}
        notes: list[str] = []
        skipped = handler(rule, index, edits, notes)
        return list(edits.values()), notes, skipped

    # -- helpers -------------------------------------------------------------------------------
    @staticmethod
    def _put(edits: dict[tuple[str, int, int], TextEdit], edit: TextEdit) -> None:
        edits.setdefault((edit.path, edit.start, edit.end), edit)

    def _target_name(self, index: SymbolIndex, model: SourceModel, root_scope: Optional[str], chain: tuple[str, ...]) -> Optional[str]:
        target = index.resolve_chain(model, root_scope, chain) if root_scope else None
        return target.name if target is not None and target.resolved and target.kind != "binding" else None

    # -- rename_symbol -----------------------------------------------------------------------------
    def _rename_symbol(self, rule: MigrationRule, index: SymbolIndex, edits: dict, notes: list[str]) -> int:
        assert rule.new
        old_module, _, old_name = rule.old.rpartition(".")
        new_module, _, new_name = rule.new.rpartition(".")
        skipped = 0
        if index.symbol(rule.old) is not None:
            notes.append(f"'{rule.old}' is defined in this repository; only its usages are migrated, the definition is unchanged")
        for model in index.repo.parsed():
            blocks = BlockInfo(model)
            statements: dict[tuple[int, int, int, int], list[ImportRecord]] = {}
            for record in model.imports:
                statements.setdefault(record.statement_span.as_tuple(), []).append(record)
            for record in model.imports:
                if not record.is_from or record.is_star:
                    continue
                target = index.import_target(model, record)
                if target.name != rule.old:
                    continue
                if new_module == old_module:
                    if new_name != old_name:
                        self._put(edits, make_edit(model, record.name_span, new_name, f"rename imported {old_name}"))
                        if not record.alias:
                            skipped += self._rename_bound_uses(model, record, new_name, edits)
                    continue
                skipped += self._move_import(model, blocks, statements[record.statement_span.as_tuple()], record, new_module, new_name, edits, notes)
            # module.attr accesses
            for use in model.attributes:
                if not use.rooted or len(use.chain) < 2 or use.chain[-1] != old_name:
                    continue
                if self._target_name(index, model, use.root_scope, use.chain) != rule.old:
                    continue
                if new_module == old_module:
                    replacement = new_name
                elif new_module.startswith(old_module + ".") if old_module else False:
                    replacement = new_module[len(old_module) + 1:] + "." + new_name
                else:
                    notes.append(f"{model.path}:{use.span.start_line}: '{rule.old}' accessed as an attribute chain cannot be rewritten to '{rule.new}' without a new import")
                    skipped += 1
                    continue
                self._put(edits, make_edit(model, use.span, replacement, f"migrate {rule.old}"))
        return skipped

    def _rename_bound_uses(self, model: SourceModel, record: ImportRecord, new_name: str, edits: dict) -> int:
        bound = record.bound_name
        for use in model.names:
            if use.name == bound and use.binding_scope == record.scope and use.context in ("load", "store", "del"):
                self._put(edits, make_edit(model, use.span, new_name, "rename use of migrated import"))
        return 0

    def _move_import(self, model: SourceModel, blocks: BlockInfo, siblings: list[ImportRecord], record: ImportRecord, new_module: str, new_name: str, edits: dict, notes: list[str]) -> int:
        alias = f" as {record.alias}" if record.alias else ""
        if len(siblings) == 1 and record.module_span is not None and record.level == 0:
            self._put(edits, make_edit(model, record.module_span, new_module, "move import to new module"))
            if new_name != record.name:
                self._put(edits, make_edit(model, record.name_span, new_name, "rename imported name"))
                if not record.alias:
                    self._rename_bound_uses(model, record, new_name, edits)
            return 0
        removal = import_removal_edits(model, blocks, siblings, [record])
        if removal is None or record.level:
            notes.append(f"{model.path}:{record.span.start_line}: import could not be moved automatically")
            return 1
        statement = record.statement_span
        indent = model.line(statement.start_line)[: statement.start_col]
        insert_at = line_start(model, statement.end_line + 1)
        text = model.source.text
        prefix = "" if insert_at == 0 or text[insert_at - 1:insert_at] == "\n" else "\n"
        for edit in removal:
            self._put(edits, edit)
        self._put(edits, TextEdit(model.path, insert_at, insert_at, f"{prefix}{indent}from {new_module} import {new_name}{alias}\n", "add migrated import"))
        if new_name != record.name and not record.alias:
            self._rename_bound_uses(model, record, new_name, edits)
        return 0

    # -- rename_module ---------------------------------------------------------------------------------
    def _rename_module(self, rule: MigrationRule, index: SymbolIndex, edits: dict, notes: list[str]) -> int:
        assert rule.new
        old, new = rule.old, rule.new
        old_parts = old.split(".")
        skipped = 0
        if index.repo.module_path(old) is not None:
            notes.append(f"'{old}' is a module of this repository; only imports are rewritten, the module files are not moved")
        for model in index.repo.parsed():
            for record in model.imports:
                if record.level:
                    continue
                module = record.module or ""
                if record.is_from:
                    if module == old or module.startswith(old + "."):
                        if record.module_span is not None:
                            self._put(edits, make_edit(model, record.module_span, new + module[len(old):], "rename imported module"))
                    elif f"{module}.{record.name}" == old and old.rpartition(".")[0] == new.rpartition(".")[0]:
                        self._put(edits, make_edit(model, record.name_span, new.rpartition(".")[2], "rename imported submodule"))
                        if not record.alias:
                            self._rename_bound_uses(model, record, new.rpartition(".")[2], edits)
                    elif f"{module}.{record.name}" == old:
                        notes.append(f"{model.path}:{record.span.start_line}: submodule import cannot be moved to a different package automatically")
                        skipped += 1
                    continue
                if module == old or module.startswith(old + "."):
                    self._put(edits, make_edit(model, record.name_span, new + module[len(old):], "rename imported module"))
                    if not record.alias:
                        skipped += self._rewrite_chains(model, index, record, old_parts, new, edits, notes)
        return skipped

    def _rewrite_chains(self, model: SourceModel, index: SymbolIndex, record: ImportRecord, old_parts: list[str], new: str, edits: dict, notes: list[str]) -> int:
        """``import a.b`` bound ``a``: rewrite ``a.b.x`` accesses to the new dotted path."""

        skipped = 0
        dotted_old = ".".join(old_parts)
        for use in model.attributes:
            if not use.rooted or use.chain != tuple(old_parts) or use.root_scope != record.scope:
                continue
            end = span_offsets(model, use.span)[1]
            start = end - len(dotted_old)
            if start < 0 or model.source.text[start:end] != dotted_old:
                notes.append(f"{model.path}:{use.span.start_line}: dotted access split over several tokens was not rewritten")
                skipped += 1
                continue
            self._put(edits, TextEdit(model.path, start, end, new, "rewrite module access"))
        return skipped

    # -- keyword rules -------------------------------------------------------------------------------------
    def _keywords(self, rule: MigrationRule, index: SymbolIndex, edits: dict, notes: list[str]) -> int:
        keyword = rule.keyword or ""
        skipped = 0
        for model in index.repo.parsed():
            nodes = _CallNodes(model)
            for call in model.calls:
                if not call.callee or self._target_name(index, model, call.root_scope, call.callee) != rule.old:
                    continue
                if rule.kind is MigrationKind.RENAME_KEYWORD:
                    for name, span in zip(call.keywords, call.keyword_spans):
                        if name == keyword:
                            self._put(edits, make_edit(model, span, rule.new or keyword, f"rename keyword {keyword}"))
                    continue
                node = nodes.get(call.span)
                if node is None:
                    skipped += 1
                    continue
                if rule.kind is MigrationKind.REMOVE_KEYWORD:
                    skipped += self._remove_keyword(model, node, call, keyword, edits, notes)
                else:
                    skipped += self._add_keyword(model, node, call, keyword, rule.value or "None", edits, notes)
        return skipped

    def _arg_offsets(self, model: SourceModel, node: ast.AST) -> tuple[int, int]:
        index = model.line_index
        end_line = getattr(node, "end_lineno", None) or node.lineno  # type: ignore[attr-defined]
        return (
            index.offset(node.lineno, index.char_column(node.lineno, node.col_offset)),  # type: ignore[attr-defined]
            index.offset(end_line, index.char_column(end_line, getattr(node, "end_col_offset", 0) or 0)),
        )

    def _remove_keyword(self, model: SourceModel, node: ast.Call, call: CallSite, keyword: str, edits: dict, notes: list[str]) -> int:
        text = model.source.text
        siblings = sorted([*node.args, *node.keywords], key=lambda n: (n.lineno, n.col_offset))
        for position, item in enumerate(siblings):
            if not isinstance(item, ast.keyword) or item.arg != keyword:
                continue
            start, end = self._arg_offsets(model, item)
            if "#" in text[start:end]:
                notes.append(f"{model.path}:{call.span.start_line}: keyword contains a comment; left unchanged")
                return 1
            if position + 1 < len(siblings):
                nxt_start, _ = self._arg_offsets(model, siblings[position + 1])
                if "#" in text[end:nxt_start]:
                    notes.append(f"{model.path}:{call.span.start_line}: comment after keyword; left unchanged")
                    return 1
                self._put(edits, TextEdit(model.path, start, nxt_start, "", f"remove keyword {keyword}"))
            elif position > 0:
                _, prev_end = self._arg_offsets(model, siblings[position - 1])
                if "#" in text[prev_end:start]:
                    return 1
                self._put(edits, TextEdit(model.path, prev_end, end, "", f"remove keyword {keyword}"))
            else:
                tail = end
                while tail < len(text) and text[tail] in " \t\r\n":
                    tail += 1
                if tail < len(text) and text[tail] == ",":
                    end = tail + 1
                self._put(edits, TextEdit(model.path, start, end, "", f"remove keyword {keyword}"))
        return 0

    def _add_keyword(self, model: SourceModel, node: ast.Call, call: CallSite, keyword: str, value: str, edits: dict, notes: list[str]) -> int:
        if any(item.arg == keyword for item in node.keywords):
            return 0
        if call.star_kwargs:
            notes.append(f"{model.path}:{call.span.start_line}: call forwards **kwargs; keyword '{keyword}' not added")
            return 1
        siblings = sorted([*node.args, *node.keywords], key=lambda n: (n.lineno, n.col_offset))
        if siblings:
            _, last_end = self._arg_offsets(model, siblings[-1])
            self._put(edits, TextEdit(model.path, last_end, last_end, f", {keyword}={value}", f"add keyword {keyword}"))
        else:
            _, call_end = self._arg_offsets(model, node)
            self._put(edits, TextEdit(model.path, call_end - 1, call_end - 1, f"{keyword}={value}", f"add keyword {keyword}"))
        return 0


__all__ = ["ApiMigrator", "MigrationOptions", "parse_rules", "validate_rule"]
