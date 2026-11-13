"""Central handler for ``semantics/``: symbols, references, search, hierarchy, compatibility."""

from __future__ import annotations

__version__ = "2.4.0"

from collections.abc import Iterable, Mapping
from typing import Any, Optional

from .models import RepositoryModel
from .semantics import ClassHierarchy, CodeSearch, InterfaceCompatibility, SymbolIndex
from .software_graphs import SoftwareGraphs
from .software_types import (
    CompatibilityReport,
    HierarchyReport,
    ParseStatus,
    SearchHit,
    SearchQuery,
    SymbolKind,
)
from .utils.software_errors import SoftwareRequestError, SoftwareSymbolError
from .utils.software_helpers import opt_int, truncate_items
from logs.logger import get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Software Semantics")


class SoftwareSemantics:
    def __init__(self, graphs: SoftwareGraphs) -> None:
        self.graphs = graphs
        self.models = graphs.models
        self.max_references = opt_int(self.models.section("symbol_index"), "max_reported_references", 500, minimum=1)

    # -- symbols ---------------------------------------------------------------------------
    def index(self, repo: RepositoryModel) -> SymbolIndex:
        return self.graphs.symbol_index(repo)

    def resolve_symbol(self, repo: RepositoryModel, text: str) -> str:
        """Accept a qualified name, or a simple name that is unambiguous in the repository."""

        index = self.index(repo)
        name = str(text or "").strip()
        if not name:
            raise SoftwareRequestError("a symbol name is required")
        if index.symbol(name) is not None or index.is_module(name):
            return name
        candidates = index.find(name)
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            raise SoftwareSymbolError("symbol not found", context={"symbol": name})
        raise SoftwareSymbolError("symbol name is ambiguous; use a qualified name", context={"symbol": name, "candidates": list(candidates[:10])})

    def symbol_info(self, repo: RepositoryModel, symbol: str) -> dict[str, Any]:
        index = self.index(repo)
        qname = self.resolve_symbol(repo, symbol)
        record = index.symbol(qname)
        if record is None:  # a module
            model = repo.module(qname)
            return {"qualified_name": qname, "kind": SymbolKind.MODULE.value, "path": repo.module_path(qname), "summary": model.summary() if model else {}}
        calls = self.graphs.call_graph(repo)
        data = record.to_dict()
        data.update(
            {
                "path": index.path_of(qname),
                "references": index.reference_count(qname),
                "heuristic_references": index.reference_count(qname, include_heuristic=True) - index.reference_count(qname),
                "callers": calls.in_degree(qname) if qname in calls else 0,
                "callees": calls.out_degree(qname) if qname in calls else 0,
            }
        )
        if record.kind is SymbolKind.CLASS:
            data["linearization"] = list(index.linearization(qname)[0])
            data["subclasses"] = list(index.subclasses(qname))
        elif record.kind is SymbolKind.METHOD:
            data["overrides"] = list(index.overridden(qname))
            data["overridden_by"] = list(index.overriders(qname))
        return data

    def references(
        self,
        repo: RepositoryModel,
        *,
        symbol: Optional[str] = None,
        path: Optional[str] = None,
        line: Optional[int] = None,
        column: int = 0,
        include_heuristic: bool = False,
    ) -> dict[str, Any]:
        index = self.index(repo)
        if symbol:
            qname = self.resolve_symbol(repo, symbol)
        elif path and line:
            found = index.symbol_at(path, int(line), int(column))
            if found is None:
                raise SoftwareSymbolError("no symbol at that position", context={"path": path, "line": line, "column": column})
            qname = found
        else:
            raise SoftwareRequestError("either 'symbol' or 'path' with 'line' is required")
        refs = index.references(qname, include_heuristic=include_heuristic)
        shown, omitted = truncate_items(list(refs), self.max_references)
        record = index.symbol(qname)
        return {
            "symbol": qname,
            "definition": None if record is None else {"path": index.path_of(qname), "span": record.name_span.to_dict(), "kind": record.kind.value},
            "count": len(refs),
            "exact": sum(1 for r in refs if r.exact),
            "references": [r.to_dict() for r in shown],
            "truncated": omitted,
        }

    # -- structure ---------------------------------------------------------------------------------
    def outline(self, repo: RepositoryModel, paths: Optional[Iterable[str]] = None) -> dict[str, Any]:
        """AST-level facts for files: symbols, imports, metrics and diagnostics."""

        selected = list(paths) if paths is not None else list(repo.paths())
        files: list[dict[str, Any]] = []
        for path in selected:
            model = repo.get(path)
            if model is None:
                raise SoftwareSymbolError("file is not part of the repository", context={"path": path})
            entry: dict[str, Any] = {"path": path, "status": model.status.value, "module": model.module_name, "summary": model.summary()}
            if model.status is ParseStatus.OK:
                entry["symbols"] = [
                    {
                        "qualified_name": s.qualified_name,
                        "kind": s.kind.value,
                        "span": s.full_span.to_dict(),
                        "public": s.is_public,
                        "decorators": list(s.decorators),
                        "bases": list(s.bases),
                        "metrics": s.metrics.to_dict() if s.metrics else None,
                    }
                    for s in model.symbols
                ]
                entry["imports"] = [
                    {"module": r.module, "name": r.name, "alias": r.alias, "level": r.level, "line": r.span.start_line, "type_checking": r.type_checking, "lazy": r.lazy}
                    for r in model.imports
                ]
            else:
                entry["diagnostics"] = [d.to_dict() for d in model.diagnostics]
            files.append(entry)
        return {"files": files}

    def search(self, repo: RepositoryModel, query: SearchQuery) -> tuple[SearchHit, ...]:
        return CodeSearch(self.index(repo)).search(query)

    def search_text(self, repo: RepositoryModel, pattern: str, **options: Any) -> tuple[SearchHit, ...]:
        return CodeSearch(self.index(repo)).search_text(pattern, **options)

    def find(
        self,
        repo: RepositoryModel,
        *,
        text: Optional[str] = None,
        grep: bool = False,
        mode: str = "substring",
        kinds: Iterable[str] = (),
        decorator: Optional[str] = None,
        base_class: Optional[str] = None,
        calls: Optional[str] = None,
        path_glob: Optional[str] = None,
        include_private: bool = True,
        regex: bool = False,
        ignore_case: bool = False,
        limit: int = 100,
    ) -> dict[str, Any]:
        """Request-shaped search: symbol query (default) or line grep, as a JSON-ready mapping."""

        if grep:
            hits = self.search_text(repo, text or "", regex=regex, ignore_case=ignore_case, path_glob=path_glob, limit=limit)
        else:
            try:
                symbol_kinds = tuple(SymbolKind(str(kind).strip().lower()) for kind in kinds)
            except ValueError as exc:
                raise SoftwareRequestError("unknown symbol kind", context={"kinds": [str(k) for k in kinds], "valid": [k.value for k in SymbolKind]}, cause=exc) from exc
            hits = self.search(
                repo,
                SearchQuery(text=text, mode=mode, kinds=symbol_kinds, decorator=decorator, base_class=base_class, calls=calls, path_glob=path_glob, include_private=include_private, limit=limit),
            )
        shown, omitted = truncate_items(list(hits), self.max_references)
        return {"count": len(hits), "hits": [hit.to_dict() for hit in shown], "truncated": omitted}

    def hierarchy(self, repo: RepositoryModel, scope: Optional[Iterable[str]] = None) -> HierarchyReport:
        return ClassHierarchy(self.index(repo)).analyze(scope=scope)

    def compatibility(self, before: RepositoryModel, after: RepositoryModel, *, include_tests: bool = False, scope: Iterable[str] = ()) -> CompatibilityReport:
        return InterfaceCompatibility().compare(before, after, include_tests=include_tests, scope=scope)


__all__ = ["SoftwareSemantics"]
