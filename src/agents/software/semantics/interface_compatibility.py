"""Public-interface compatibility between two repository models.

Rules follow the usual semantic-versioning reading of a Python API: removing or
narrowing something callers may rely on is *breaking*; adding capability is
*additive*; changes that cannot break a caller (default values, annotations,
new bases) are *compatible* and reported for review.
"""

from __future__ import annotations

__version__ = "2.4.0"

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Optional

from ..models.repository_model import RepositoryModel
from ..software_types import (
    CompatibilityChange,
    CompatibilityLevel,
    CompatibilityReport,
    ParameterInfo,
    ParameterKind,
    SignatureInfo,
    SymbolKind,
    SymbolRecord,
    VersionBump,
)
from ..utils.software_helpers import is_within_scope
from logs.logger import get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Software Interface Compatibility")

B, A, C = CompatibilityLevel.BREAKING, CompatibilityLevel.ADDITIVE, CompatibilityLevel.COMPATIBLE
_POSITIONAL = (ParameterKind.POSITIONAL_ONLY, ParameterKind.POSITIONAL_OR_KEYWORD)
_KEYWORD = (ParameterKind.POSITIONAL_OR_KEYWORD, ParameterKind.KEYWORD_ONLY)


@dataclass(frozen=True, slots=True)
class ApiEntry:
    path: str
    symbol: SymbolRecord


def _private_module(path: str, module: str) -> bool:
    parts = [p for p in module.split(".") if p]
    return any(p.startswith("_") and not (p.startswith("__") and p.endswith("__")) for p in parts)


def public_api(repo: RepositoryModel, *, include_tests: bool = False, scope: Iterable[str] = ()) -> dict[str, ApiEntry]:
    """Public symbols of ``repo`` keyed by qualified name."""

    scopes = tuple(scope)
    api: dict[str, ApiEntry] = {}
    for model in repo.parsed():
        if not model.module_name or repo.module_path(model.module_name) != model.path:
            continue
        if (model.source.is_test and not include_tests) or _private_module(model.path, model.module_name):
            continue
        exports = set(model.exports) if model.exports is not None else None
        public_parents = {model.module_name}
        for symbol in model.symbols:
            if symbol.parent not in public_parents or not symbol.is_public:
                continue
            if exports is not None and symbol.parent == model.module_name and symbol.name not in exports:
                continue
            if scopes and not any(is_within_scope(symbol.qualified_name, item) for item in scopes):
                continue
            api[symbol.qualified_name] = ApiEntry(model.path, symbol)
            if symbol.kind is SymbolKind.CLASS:
                public_parents.add(symbol.qualified_name)
    return api


def _fmt(param: ParameterInfo) -> str:
    prefix = {ParameterKind.VAR_POSITIONAL: "*", ParameterKind.VAR_KEYWORD: "**"}.get(param.kind, "")
    return f"{prefix}{param.name}" + ("=…" if param.has_default else "")


def _signature_text(sig: Optional[SignatureInfo]) -> str:
    if sig is None:
        return ""
    return "(" + ", ".join(_fmt(p) for p in sig.parameters) + ")" + (" -> " + sig.returns if sig.returns else "")


def compare_signatures(name: str, before: SignatureInfo, after: SignatureInfo, *, drop_first: bool = False) -> list[tuple[str, CompatibilityLevel, str]]:
    """Return ``(kind, level, detail)`` for each caller-visible signature change."""

    out: list[tuple[str, CompatibilityLevel, str]] = []
    old, new = list(before.parameters), list(after.parameters)
    if drop_first:  # ``self``/``cls``
        old, new = old[1:], new[1:]
    if before.is_async != after.is_async:
        out.append(("async_changed", B, "async/sync calling convention changed"))
    new_by_name = {p.name: p for p in new}
    old_names = {p.name for p in old}
    old_pos = [p for p in old if p.kind in _POSITIONAL]
    new_pos = [p for p in new if p.kind in _POSITIONAL]
    var_kw_after = any(p.kind is ParameterKind.VAR_KEYWORD for p in new)
    renamed_to: set[str] = set()

    for index, param in enumerate(old):
        match = new_by_name.get(param.name)
        if param.kind in (ParameterKind.VAR_POSITIONAL, ParameterKind.VAR_KEYWORD):
            if match is None or match.kind is not param.kind:
                out.append(("variadic_removed", B, f"{_fmt(param)} no longer accepted"))
            continue
        if match is None:
            replacement = None
            if param in old_pos and old_pos.index(param) < len(new_pos):
                candidate = new_pos[old_pos.index(param)]
                if candidate.name not in old_names:
                    replacement = candidate.name
            renamed = replacement is not None
            if renamed:
                new_name = replacement
                renamed_to.add(new_name)
                level = C if param.kind is ParameterKind.POSITIONAL_ONLY else B
                out.append(("parameter_renamed", level, f"parameter '{param.name}' renamed to '{new_name}'"))
            else:
                level = C if (var_kw_after and param.kind is ParameterKind.KEYWORD_ONLY) else B
                out.append(("parameter_removed", level, f"parameter '{param.name}' removed"))
            continue
        if param.kind != match.kind:
            widened = (param.kind is ParameterKind.POSITIONAL_ONLY and match.kind is ParameterKind.POSITIONAL_OR_KEYWORD) or (
                param.kind is ParameterKind.KEYWORD_ONLY and match.kind is ParameterKind.POSITIONAL_OR_KEYWORD
            )
            out.append(("parameter_kind_changed", A if widened else B, f"parameter '{param.name}' changed from {param.kind.value} to {match.kind.value}"))
        if param.has_default and not match.has_default:
            out.append(("default_removed", B, f"parameter '{param.name}' lost its default"))
        elif not param.has_default and match.has_default:
            out.append(("default_added", A, f"parameter '{param.name}' gained a default"))
        elif param.default != match.default:
            out.append(("default_changed", C, f"default of '{param.name}' changed ({param.default} -> {match.default})"))
        if param.annotation != match.annotation:
            out.append(("annotation_changed", C, f"annotation of '{param.name}' changed ({param.annotation} -> {match.annotation})"))

    old_pos_names = [p.name for p in old_pos]
    for position, param in enumerate(new_pos):
        if param.name in old_names or param.name in renamed_to:
            if param.name in old_pos_names and old_pos_names.index(param.name) != position:
                out.append(("parameter_reordered", B, f"positional parameter '{param.name}' moved from {old_pos_names.index(param.name)} to {position}"))
            continue
        shifts = position < len(old_pos)
        if not param.has_default:
            out.append(("required_parameter_added", B, f"required parameter '{param.name}' added"))
        elif shifts:
            out.append(("parameter_inserted", B, f"optional parameter '{param.name}' inserted before existing positional parameters"))
        else:
            out.append(("optional_parameter_added", A, f"optional parameter '{param.name}' added"))
    for param in new:
        if param.name in old_names or param.name in renamed_to or param in new_pos:
            continue
        if param.kind is ParameterKind.KEYWORD_ONLY and not param.has_default:
            out.append(("required_parameter_added", B, f"required keyword-only parameter '{param.name}' added"))
        elif param.kind in (ParameterKind.VAR_POSITIONAL, ParameterKind.VAR_KEYWORD):
            out.append(("variadic_added", A, f"{_fmt(param)} accepted"))
        else:
            out.append(("optional_parameter_added", A, f"optional parameter '{param.name}' added"))
    if before.returns != after.returns:
        out.append(("return_annotation_changed", C, f"return annotation changed ({before.returns} -> {after.returns})"))
    return out


class InterfaceCompatibility:
    def compare(
        self,
        before: RepositoryModel,
        after: RepositoryModel,
        *,
        include_tests: bool = False,
        scope: Iterable[str] = (),
    ) -> CompatibilityReport:
        old = public_api(before, include_tests=include_tests, scope=scope)
        new = public_api(after, include_tests=include_tests, scope=scope)
        changes: list[CompatibilityChange] = []

        removed_modules = sorted(
            {e.symbol.parent for q, e in old.items() if e.symbol.parent and before.module_path(e.symbol.parent)}
            - {m for m in after.module_names()}
        )
        for module in removed_modules:
            changes.append(CompatibilityChange(module, "module_removed", B, "module with public symbols no longer exists", before.module_path(module)))

        for qname in sorted(old.keys() - new.keys()):
            entry = old[qname]
            if entry.symbol.parent in removed_modules:
                continue
            changes.append(CompatibilityChange(qname, "removed", B, f"public {entry.symbol.kind.value} removed", entry.path, _signature_text(entry.symbol.signature) or None, None))
        for qname in sorted(new.keys() - old.keys()):
            entry = new[qname]
            changes.append(CompatibilityChange(qname, "added", A, f"public {entry.symbol.kind.value} added", entry.path, None, _signature_text(entry.symbol.signature) or None))
        for qname in sorted(old.keys() & new.keys()):
            changes.extend(self._compare_symbol(qname, old[qname], new[qname]))

        counts = {B: 0, A: 0, C: 0}
        for change in changes:
            counts[change.level] += 1
        bump = VersionBump.MAJOR if counts[B] else VersionBump.MINOR if counts[A] else VersionBump.PATCH if counts[C] else VersionBump.NONE
        return CompatibilityReport(
            changes=tuple(changes),
            recommended_bump=bump,
            breaking=counts[B],
            additive=counts[A],
            compatible=counts[C],
            public_symbols_before=len(old),
            public_symbols_after=len(new),
        )

    def _compare_symbol(self, qname: str, before: ApiEntry, after: ApiEntry) -> list[CompatibilityChange]:
        out: list[CompatibilityChange] = []
        a, b = before.symbol, after.symbol

        def change(kind: str, level: CompatibilityLevel, detail: str, old: Optional[str] = None, new: Optional[str] = None) -> None:
            out.append(CompatibilityChange(qname, kind, level, detail, after.path, old, new))

        if a.kind is not b.kind:
            change("kind_changed", B, f"{a.kind.value} became {b.kind.value}", a.kind.value, b.kind.value)
            return out
        if a.kind in (SymbolKind.FUNCTION, SymbolKind.METHOD) and a.signature and b.signature:
            drop = a.kind is SymbolKind.METHOD and not any(d.rsplit(".", 1)[-1] == "staticmethod" for d in a.decorators)
            for kind, level, detail in compare_signatures(qname, a.signature, b.signature, drop_first=drop):
                change(kind, level, detail, _signature_text(a.signature), _signature_text(b.signature))
            for flag, label in (("staticmethod", "static"), ("classmethod", "class"), ("property", "property")):
                had = any(d.rsplit(".", 1)[-1] == flag for d in a.decorators)
                has = any(d.rsplit(".", 1)[-1] == flag for d in b.decorators)
                if had != has:
                    change(f"{flag}_changed", B, f"{label}-method binding changed")
            if a.is_abstract != b.is_abstract:
                change("abstractness_changed", B if b.is_abstract else C, "abstract status changed")
        elif a.kind is SymbolKind.CLASS:
            old_bases, new_bases = set(a.bases), set(b.bases)
            for base in sorted(old_bases - new_bases):
                change("base_removed", B, f"no longer inherits from {base}")
            for base in sorted(new_bases - old_bases):
                change("base_added", C, f"now inherits from {base}")
            if b.is_abstract and not a.is_abstract:
                change("became_abstract", B, "class can no longer be instantiated directly")
        elif a.kind is SymbolKind.VARIABLE and a.annotation != b.annotation:
            change("annotation_changed", C, f"annotation changed ({a.annotation} -> {b.annotation})", a.annotation, b.annotation)
        return out


__all__ = ["ApiEntry", "InterfaceCompatibility", "compare_signatures", "public_api"]
