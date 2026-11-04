"""Repository-wide symbol table, cross-module name resolution and references.

Resolution is *static and conservative*.  Lexical binding comes from the
parser (LEGB, already per occurrence); this module adds what needs the whole
repository: following ``import`` chains (including re-exports and
``from x import *`` aggregation), attribute chains through modules, ``self``/
``cls`` receivers, and class-member lookup along a C3 linearization.  Anything
it cannot bind exactly stays unresolved; optional *heuristic* references match
an unbound ``.attr`` to uniquely-named class members and are flagged
``exact=False`` so callers can decide whether to trust them.
"""

from __future__ import annotations

__version__ = "2.4.0"

import threading

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Optional

from ..models.repository_model import RepositoryModel
from ..models.source_model import SourceModel
from ..software_types import (
    BindingInfo,
    BindingKind,
    ImportRecord,
    ParseStatus,
    ScopeInfo,
    ScopeKind,
    SourceSpan,
    SUPER_ROOT,
    SymbolKind,
    SymbolRecord,
    SymbolReference,
)
from ..utils.software_errors import SoftwareSymbolError
from ..utils.software_helpers import opt_bool, opt_int
from logs.logger import get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Software Symbol Index")


@dataclass(frozen=True, slots=True)
class Target:
    """Result of resolving a name or attribute chain.

    ``kind`` is ``symbol`` | ``module`` | ``external`` | ``binding`` | ``unresolved``.  For
    ``symbol``/``module`` ``name`` is the repository qualified name; for
    ``external`` the dotted path in the third-party/stdlib module;
    ``binding`` is an existing but untracked module/class attribute.
    ``instance`` marks a ``self``/``cls`` receiver (a class target that stands
    for an instance of it or its subclasses).
    """

    kind: str
    name: Optional[str] = None
    instance: bool = False

    @property
    def resolved(self) -> bool:
        return self.kind != "unresolved"

    @property
    def internal(self) -> bool:
        return self.kind in ("symbol", "module")


UNRESOLVED = Target("unresolved")


@dataclass(frozen=True, slots=True)
class IndexOptions:
    include_heuristic: bool = True
    heuristic_max_candidates: int = 3
    max_alias_depth: int = 8

    @classmethod
    def from_config(cls, section: Optional[Mapping[str, Any]] = None) -> "IndexOptions":
        data = section or {}
        return cls(
            include_heuristic=opt_bool(data, "include_heuristic_references", True),
            heuristic_max_candidates=opt_int(data, "heuristic_max_candidates", 3, minimum=1),
            max_alias_depth=opt_int(data, "max_alias_depth", 8, minimum=1),
        )


@dataclass(frozen=True, slots=True)
class ResolvedBases:
    internal: tuple[str, ...]
    external: tuple[str, ...]
    unresolved: tuple[str, ...]


class SymbolIndex:
    """Immutable-after-construction index over one :class:`RepositoryModel`."""

    def __init__(self, repo: RepositoryModel, options: Optional[IndexOptions] = None) -> None:
        self.repo = repo
        self.options = options or IndexOptions()
        self._lock = threading.RLock()
        self._symbols: dict[str, tuple[str, SymbolRecord]] = {}
        self._by_name: dict[str, list[str]] = {}
        self._members: dict[str, list[str]] = {}
        self._module_symbols: dict[str, list[str]] = {}
        for model in repo.models():
            if model.status is not ParseStatus.OK:
                continue
            for symbol in model.symbols:
                if symbol.qualified_name in self._symbols:
                    continue  # first path wins (module-name conflicts are reported by the repository model)
                self._symbols[symbol.qualified_name] = (model.path, symbol)
                self._by_name.setdefault(symbol.name, []).append(symbol.qualified_name)
                if symbol.parent:
                    self._members.setdefault(symbol.parent, []).append(symbol.qualified_name)
        self._binding_maps: dict[tuple[str, str], dict[str, BindingInfo]] = {}
        self._import_maps: dict[tuple[str, str], dict[str, ImportRecord]] = {}
        self._bases: dict[str, ResolvedBases] = {}
        self._linearization: dict[str, tuple[tuple[str, ...], bool]] = {}
        self._subclasses: Optional[dict[str, list[str]]] = None
        self._exact: Optional[dict[str, list[SymbolReference]]] = None
        self._heuristic: dict[str, list[SymbolReference]] = {}
        self._dynamic_names: Optional[frozenset[str]] = None

    # ------------------------------------------------------------------ symbols
    def symbol(self, qualified_name: str) -> Optional[SymbolRecord]:
        item = self._symbols.get(qualified_name)
        return item[1] if item else None

    def require(self, qualified_name: str) -> SymbolRecord:
        symbol = self.symbol(qualified_name)
        if symbol is None:
            raise SoftwareSymbolError("unknown symbol", context={"symbol": qualified_name})
        return symbol

    def path_of(self, qualified_name: str) -> Optional[str]:
        item = self._symbols.get(qualified_name)
        if item:
            return item[0]
        return self.repo.module_path(qualified_name)

    def model_of(self, qualified_name: str) -> Optional[SourceModel]:
        path = self.path_of(qualified_name)
        return self.repo.get(path) if path else None

    def is_module(self, name: str) -> bool:
        return self.repo.module_path(name) is not None

    def kind_of(self, name: str) -> Optional[SymbolKind]:
        symbol = self.symbol(name)
        if symbol is not None:
            return symbol.kind
        return SymbolKind.MODULE if self.is_module(name) else None

    def symbols(self) -> Iterator[tuple[str, SymbolRecord]]:
        """``(path, symbol)`` for every indexed symbol, in qualified-name order."""

        for name in sorted(self._symbols):
            yield self._symbols[name]

    def __len__(self) -> int:
        return len(self._symbols)

    def find(self, name: str) -> tuple[str, ...]:
        """Qualified names whose *simple* name is ``name``."""

        return tuple(self._by_name.get(name, ()))

    def members(self, parent: str) -> tuple[SymbolRecord, ...]:
        return tuple(self._symbols[q][1] for q in self._members.get(parent, ()))

    def module_symbols(self, module: str) -> tuple[SymbolRecord, ...]:
        return self.members(module)

    def enclosing_class(self, qualified_name: str) -> Optional[str]:
        symbol = self.symbol(qualified_name)
        while symbol is not None and symbol.parent:
            parent = self.symbol(symbol.parent)
            if parent is None:
                return None
            if parent.kind is SymbolKind.CLASS:
                return parent.qualified_name
            symbol = parent
        return None

    # ----------------------------------------------------------------- lexical maps
    def _bindings(self, model: SourceModel, scope: str) -> dict[str, BindingInfo]:
        key = (model.path, scope)
        found = self._binding_maps.get(key)
        if found is None:
            info = model.scope(scope)
            found = {item.name: item for item in info.bindings} if info else {}
            self._binding_maps[key] = found
        return found

    def _imports(self, model: SourceModel, scope: str) -> dict[str, ImportRecord]:
        key = (model.path, scope)
        found = self._import_maps.get(key)
        if found is None:
            found = {}
            for record in model.imports:
                if record.scope == scope and record.bound_name:
                    found[record.bound_name] = record
            self._import_maps[key] = found
        return found

    def lookup(self, model: SourceModel, scope: str, name: str) -> Optional[tuple[ScopeInfo, BindingInfo]]:
        """Re-run LEGB for ``name`` as seen from ``scope`` (used for base expressions)."""

        info = model.scope(scope)
        first = True
        while info is not None:
            if first or info.kind is not ScopeKind.CLASS:
                binding = self._bindings(model, info.qualified_name).get(name)
                if binding is not None:
                    return info, binding
            first = False
            info = model.scope(info.parent) if info.parent else None
        return None

    # --------------------------------------------------------------- resolution
    def _member(self, module: str, attr: str, depth: int, seen: frozenset[str]) -> Target:
        """Resolve ``module.attr`` following re-exports and star imports."""

        qname = f"{module}.{attr}"
        if qname in self._symbols:
            return Target("symbol", qname)
        if self.is_module(qname) or self.repo.is_package(qname):
            return Target("module", qname)
        key = f"{module}:{attr}"
        if depth > self.options.max_alias_depth or key in seen:
            return UNRESOLVED
        seen = seen | {key}
        model = self.repo.module(module)
        if model is None or model.status is not ParseStatus.OK:
            return UNRESOLVED
        if model.module_scope is not None:
            binding = self._bindings(model, model.module_scope.qualified_name).get(attr)
            if binding is not None:
                return self._binding_target(model, model.module_scope.qualified_name, binding, depth + 1, seen)
        for record, resolved in self.repo.imports_of(model.path):
            if record.is_star and resolved.module and resolved.internal:
                target = self._member(resolved.module, attr, depth + 1, seen)
                if target.resolved:
                    return target
        return UNRESOLVED

    def member(self, module: str, attr: str) -> Target:
        """Resolve ``module.attr`` (symbol, submodule, re-export or star-import member)."""

        return self._member(module, attr, 0, frozenset())

    def import_target(self, model: SourceModel, record: ImportRecord) -> Target:
        """What a single import record binds (symbol, module, or external dotted name)."""

        return self._import_target(model, record, 0, frozenset())

    def _import_target(self, model: SourceModel, record: ImportRecord, depth: int, seen: frozenset[str]) -> Target:
        resolved = next((res for rec, res in self.repo.imports_of(model.path) if rec is record), None)
        if resolved is None:
            return UNRESOLVED
        if not record.is_from:
            if resolved.internal:
                if record.alias:
                    return Target("module", resolved.module) if resolved.module else UNRESOLVED
                top = (record.module or "").split(".", 1)[0]
                return Target("module", top) if self.is_module(top) or self.repo.is_package(top) else UNRESOLVED
            top = (record.module or "").split(".", 1)[0]
            return Target("external", record.module if record.alias else top)
        if resolved.internal:
            if resolved.submodule:
                return Target("module", resolved.submodule)
            if resolved.module and resolved.attribute and resolved.attribute != "*":
                return self._member(resolved.module, resolved.attribute, depth, seen)
            return UNRESOLVED
        if resolved.module and record.name:
            return Target("external", f"{resolved.module}.{record.name}")
        return UNRESOLVED

    def _binding_target(self, model: SourceModel, scope: str, binding: BindingInfo, depth: int, seen: frozenset[str]) -> Target:
        if binding.symbol and binding.symbol in self._symbols:
            return Target("symbol", binding.symbol)
        if binding.kind is BindingKind.IMPORT:
            record = self._imports(model, scope).get(binding.name)
            return self._import_target(model, record, depth, seen) if record else UNRESOLVED
        info = model.scope(scope)
        if info is not None and info.kind in (ScopeKind.MODULE, ScopeKind.CLASS) and info.symbol:
            qname = f"{info.symbol}.{binding.name}"
            if qname in self._symbols:
                return Target("symbol", qname)
            return Target("binding", qname)  # exists, but is not an indexed symbol (e.g. ``__all__``)
        return UNRESOLVED

    def resolve_name(self, model: SourceModel, binding_scope: Optional[str], name: str) -> Target:
        """Target of a name whose lexical binding lives in ``binding_scope``."""

        if binding_scope is None:
            return UNRESOLVED
        binding = self._bindings(model, binding_scope).get(name)
        if binding is None:
            return UNRESOLVED
        receiver = self._receiver_class(model, binding_scope, binding)
        if receiver is not None:
            return Target("symbol", receiver, instance=True)
        return self._binding_target(model, binding_scope, binding, 0, frozenset())

    def _receiver_class(self, model: SourceModel, scope: str, binding: BindingInfo) -> Optional[str]:
        """Class qualified name when ``binding`` is the ``self``/``cls`` parameter of a method."""

        if binding.kind is not BindingKind.PARAMETER:
            return None
        info = model.scope(scope)
        if info is None or info.kind is not ScopeKind.FUNCTION or not info.symbol:
            return None
        method = self.symbol(info.symbol)
        if method is None or method.kind is not SymbolKind.METHOD or method.signature is None:
            return None
        if any(item.rsplit(".", 1)[-1] == "staticmethod" for item in method.decorators):
            return None
        params = method.signature.parameters
        if not params or params[0].name != binding.name:
            return None
        return method.parent

    def class_member(self, class_name: str, attr: str) -> Target:
        for candidate in self.linearization(class_name)[0]:
            qname = f"{candidate}.{attr}"
            if qname in self._symbols:
                return Target("symbol", qname)
        return UNRESOLVED

    def resolve_chain(self, model: SourceModel, root_scope: Optional[str], chain: tuple[str, ...]) -> Target:
        """Resolve a rooted attribute chain; returns the target of the *whole* chain."""

        if not chain:
            return UNRESOLVED
        if chain[0] == SUPER_ROOT:
            return self._super_member(root_scope, chain[1:])
        current = self.resolve_name(model, root_scope, chain[0])
        for part in chain[1:]:
            if current.kind == "module" and current.name:
                current = self._member(current.name, part, 0, frozenset())
            elif current.kind == "symbol" and current.name:
                kind = self.kind_of(current.name)
                if kind is SymbolKind.CLASS:
                    current = self.class_member(current.name, part)
                else:
                    return UNRESOLVED
            elif current.kind == "external" and current.name:
                current = Target("external", f"{current.name}.{part}")
            else:
                return UNRESOLVED
            if not current.resolved:
                return UNRESOLVED
        return current

    def _super_member(self, scope: Optional[str], rest: tuple[str, ...]) -> Target:
        """``super().attr`` inside a method: the next definition after the enclosing class in its MRO.

        Falls back to the first external base (or ``object``) when every base is known, and stays
        unresolved when some base could not be resolved.
        """

        owner = self.enclosing_class(scope) if scope else None
        if owner is None or len(rest) != 1:
            return UNRESOLVED
        order = self.linearization(owner)[0]
        for candidate in order:
            if candidate != owner and f"{candidate}.{rest[0]}" in self._symbols:
                return Target("symbol", f"{candidate}.{rest[0]}")
        bases = [self.class_bases(name) for name in order]
        if any(item.unresolved for item in bases):
            return UNRESOLVED
        external = next((name for item in bases for name in item.external), "object")
        return Target("external", f"{external}.{rest[0]}")

    # ------------------------------------------------------------------- classes
    def class_bases(self, class_name: str) -> ResolvedBases:
        cached = self._bases.get(class_name)
        if cached is not None:
            return cached
        symbol = self.symbol(class_name)
        model = self.model_of(class_name)
        internal: list[str] = []
        external: list[str] = []
        unresolved: list[str] = []
        if symbol is not None and model is not None and symbol.kind is SymbolKind.CLASS:
            scope = symbol.parent or model.module_name
            owner_scope = self._scope_for_symbol(model, scope)
            for text in symbol.bases:
                if text.startswith("metaclass="):
                    continue
                target = self._resolve_text(model, owner_scope, text)
                if target.kind == "symbol" and target.name and self.kind_of(target.name) is SymbolKind.CLASS:
                    internal.append(target.name)
                elif target.kind == "external" and target.name:
                    external.append(target.name)
                else:
                    unresolved.append(text)
        result = ResolvedBases(tuple(internal), tuple(external), tuple(unresolved))
        self._bases[class_name] = result
        return result

    def _scope_for_symbol(self, model: SourceModel, symbol_name: str) -> str:
        info = model.scope(symbol_name)
        return info.qualified_name if info else (model.module_scope.qualified_name if model.module_scope else symbol_name)

    def _resolve_text(self, model: SourceModel, scope: str, text: str) -> Target:
        """Resolve a dotted expression string (``a.b.C``) as seen from ``scope``."""

        parts = tuple(text.split("."))
        if not all(part.isidentifier() for part in parts):
            return UNRESOLVED  # subscripts / calls such as ``Generic[T]`` are not plain names
        found = self.lookup(model, scope, parts[0])
        if found is None:
            return UNRESOLVED
        return self.resolve_chain(model, found[0].qualified_name, parts)

    def linearization(self, class_name: str) -> tuple[tuple[str, ...], bool]:
        """C3 method resolution order over *repository* classes, plus a conflict flag.

        On an inconsistent hierarchy a depth-first order is returned and the
        flag is ``True`` (Python itself would refuse to create the class).
        """

        cached = self._linearization.get(class_name)
        if cached is not None:
            return cached
        result = self._c3(class_name, frozenset())
        self._linearization[class_name] = result
        return result

    def _c3(self, class_name: str, active: frozenset[str]) -> tuple[tuple[str, ...], bool]:
        if class_name in active:
            return (class_name,), True
        known = self._linearization.get(class_name)
        if known is not None:
            return known
        bases = self.class_bases(class_name).internal
        conflict = False
        sequences: list[list[str]] = []
        for base in bases:
            order, bad = self._c3(base, active | {class_name})
            conflict = conflict or bad
            sequences.append(list(order))
        sequences.append(list(bases))
        merged: list[str] = [class_name]
        pending = [seq for seq in sequences if seq]
        while pending:
            for seq in pending:
                head = seq[0]
                if not any(head in other[1:] for other in pending):
                    break
            else:
                # inconsistent: fall back to depth-first, left-to-right
                conflict = True
                seen = set(merged)
                for seq in pending:
                    for item in seq:
                        if item not in seen:
                            merged.append(item)
                            seen.add(item)
                break
            merged.append(head)
            pending = [[item for item in seq if item != head] for seq in pending]
            pending = [seq for seq in pending if seq]
        return tuple(merged), conflict

    def subclasses(self, class_name: str, *, transitive: bool = False) -> tuple[str, ...]:
        with self._lock:
            if self._subclasses is None:
                table: dict[str, list[str]] = {}
                for qname, (_, symbol) in self._symbols.items():
                    if symbol.kind is SymbolKind.CLASS:
                        for base in self.class_bases(qname).internal:
                            table.setdefault(base, []).append(qname)
                self._subclasses = table
            table = self._subclasses
        direct = tuple(sorted(table.get(class_name, ())))
        if not transitive:
            return direct
        seen: dict[str, None] = {}
        stack = list(direct)
        while stack:
            item = stack.pop()
            if item in seen:
                continue
            seen[item] = None
            stack.extend(table.get(item, ()))
        return tuple(sorted(seen))

    def overridden(self, method_name: str) -> tuple[str, ...]:
        """Methods in base classes that ``method_name`` overrides (nearest first)."""

        symbol = self.symbol(method_name)
        if symbol is None or symbol.kind is not SymbolKind.METHOD or not symbol.parent:
            return ()
        order = self.linearization(symbol.parent)[0][1:]
        return tuple(f"{cls}.{symbol.name}" for cls in order if f"{cls}.{symbol.name}" in self._symbols)

    def overriders(self, method_name: str) -> tuple[str, ...]:
        """Methods in subclasses that override ``method_name``."""

        symbol = self.symbol(method_name)
        if symbol is None or symbol.kind is not SymbolKind.METHOD or not symbol.parent:
            return ()
        return tuple(
            f"{cls}.{symbol.name}"
            for cls in self.subclasses(symbol.parent, transitive=True)
            if f"{cls}.{symbol.name}" in self._symbols
        )

    # ---------------------------------------------------------------- references
    def _ensure_references(self) -> dict[str, list[SymbolReference]]:
        with self._lock:
            if self._exact is not None:
                return self._exact
            exact: dict[str, list[SymbolReference]] = {}
            heuristic: dict[str, list[SymbolReference]] = {}
            dynamic: set[str] = set()
            for model in self.repo.parsed():
                dynamic.update(model.string_names)
                callee_ends = {(call.callee_span.end_line, call.callee_span.end_col) for call in model.calls}

                def add(table: dict[str, list[SymbolReference]], target: str, span: SourceSpan, kind: str, scope: str, context: str, is_exact: bool) -> None:
                    if kind in ("name", "attribute") and (span.end_line, span.end_col) in callee_ends:
                        kind = "call"
                    table.setdefault(target, []).append(SymbolReference(target, model.path, span, kind, scope, context, is_exact))

                for use in model.names:
                    if use.binding_scope is None:
                        continue
                    target = self.resolve_name(model, use.binding_scope, use.name)
                    if target.internal and target.name and not target.instance:
                        add(exact, target.name, use.span, "name", use.scope, use.context, True)
                for use in model.attributes:
                    target = self.resolve_chain(model, use.root_scope, use.chain) if use.rooted else UNRESOLVED
                    if target.internal and target.name:
                        add(exact, target.name, use.span, "attribute", use.scope, use.context, True)
                    elif target.kind != "external" and self.options.include_heuristic:
                        self._heuristic_attribute(model, use.chain[-1], use.span, use.scope, use.context, heuristic, callee_ends)
                for record, resolved in self.repo.imports_of(model.path):
                    if record.is_star or not resolved.internal:
                        continue
                    target_name: Optional[str] = resolved.submodule
                    if target_name is None and resolved.module and resolved.attribute:
                        target = self._member(resolved.module, resolved.attribute, 0, frozenset())
                        target_name = target.name if target.internal else None
                    if target_name is None and not record.is_from:
                        target_name = resolved.module
                    if target_name:
                        span = record.module_span if (record.is_from is False and record.module_span) else record.name_span
                        add(exact, target_name, span, "import", record.scope, "load", True)
                if model.exports:
                    for name, span in model.export_spans:
                        target = self._member(model.module_name, name, 0, frozenset())
                        if target.internal and target.name:
                            add(exact, target.name, span, "export", model.module_name, "load", True)
            self._exact = exact
            self._heuristic = heuristic
            self._dynamic_names = frozenset(dynamic)
            return exact

    def _heuristic_attribute(
        self,
        model: SourceModel,
        attr: str,
        span: SourceSpan,
        scope: str,
        context: str,
        table: dict[str, list[SymbolReference]],
        callee_ends: set[tuple[int, int]],
    ) -> None:
        candidates = [
            q for q in self._by_name.get(attr, ())
            if (sym := self._symbols[q][1]).parent
            and (parent := self.symbol(sym.parent)) is not None
            and parent.kind is SymbolKind.CLASS
        ]
        if not candidates or len(candidates) > self.options.heuristic_max_candidates:
            return
        kind = "call" if (span.end_line, span.end_col) in callee_ends else "attribute"
        for qname in candidates:
            table.setdefault(qname, []).append(SymbolReference(qname, model.path, span, kind, scope, context, False))

    def references(self, qualified_name: str, *, include_heuristic: bool = False) -> tuple[SymbolReference, ...]:
        exact = self._ensure_references().get(qualified_name, [])
        items = list(exact)
        if include_heuristic:
            items.extend(self._heuristic.get(qualified_name, ()))
        items.sort(key=lambda r: (r.path, r.span.as_tuple(), r.kind))
        return tuple(items)

    def reference_count(self, qualified_name: str, *, include_heuristic: bool = False) -> int:
        count = len(self._ensure_references().get(qualified_name, ()))
        return count + (len(self._heuristic.get(qualified_name, ())) if include_heuristic else 0)

    def all_references(self, *, include_heuristic: bool = False) -> Mapping[str, tuple[SymbolReference, ...]]:
        merged = {key: list(value) for key, value in self._ensure_references().items()}
        if include_heuristic:
            for key, value in self._heuristic.items():
                merged.setdefault(key, []).extend(value)
        return {key: tuple(value) for key, value in merged.items()}

    def dynamic_names(self) -> frozenset[str]:
        """String literals that look like identifiers/dotted names (possible ``getattr``/plugin lookups)."""

        self._ensure_references()
        return self._dynamic_names or frozenset()

    def symbol_at(self, path: str, line: int, column: int) -> Optional[str]:
        """Qualified name of the symbol declared or referenced at ``(line, column)``."""

        model = self.repo.get(path)
        if model is None or model.status is not ParseStatus.OK:
            return None

        def inside(span: SourceSpan) -> bool:
            return (span.start_line, span.start_col) <= (line, column) < (span.end_line, span.end_col)

        for symbol in model.symbols:
            if inside(symbol.name_span) and symbol.qualified_name in self._symbols:
                return symbol.qualified_name
        for target, refs in self._ensure_references().items():
            for ref in refs:
                if ref.path == path and inside(ref.span):
                    return target
        return None

    def stats(self) -> dict[str, Any]:
        kinds: dict[str, int] = {}
        for _, symbol in self._symbols.values():
            kinds[symbol.kind.value] = kinds.get(symbol.kind.value, 0) + 1
        return {"symbols": len(self._symbols), "by_kind": dict(sorted(kinds.items())), "modules": len(self.repo.module_names())}


__all__ = ["IndexOptions", "ResolvedBases", "SymbolIndex", "Target", "UNRESOLVED"]
