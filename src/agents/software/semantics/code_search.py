"""Structured and textual code search over a repository model."""

from __future__ import annotations

__version__ = "2.4.0"

import difflib
import re

from typing import Optional

from ..models.repository_model import RepositoryModel
from ..software_types import SearchHit, SearchQuery, SourceSpan, SymbolKind, SymbolRecord
from ..utils.software_errors import SoftwareRequestError
from ..utils.software_helpers import compile_glob
from .symbol_index import SymbolIndex
from logs.logger import get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Software Code Search")

MODES = frozenset({"exact", "prefix", "substring", "regex", "fuzzy"})
_MAX_PATTERN = 200
_WORD_SPLIT = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")


def _words(name: str) -> list[str]:
    return [w.lower() for w in _WORD_SPLIT.findall(name)]


def _last(dotted: str) -> str:
    return dotted.rsplit(".", 1)[-1]


class CodeSearch:
    def __init__(self, index: SymbolIndex) -> None:
        self.index = index
        self.repo: RepositoryModel = index.repo

    # -- scoring ----------------------------------------------------------------
    @staticmethod
    def _score(mode: str, needle: str, regex: Optional["re.Pattern[str]"], symbol: SymbolRecord) -> float:
        name, qname = symbol.name.lower(), symbol.qualified_name.lower()
        if mode == "regex":
            assert regex is not None
            return 0.8 if regex.search(symbol.name) or regex.search(symbol.qualified_name) else 0.0
        if mode == "exact":
            return 1.0 if needle in (name, qname) else 0.0
        if mode == "prefix":
            return (0.95 if name.startswith(needle) else 0.0) or (0.75 if qname.startswith(needle) else 0.0)
        if mode == "fuzzy":
            ratio = difflib.SequenceMatcher(None, needle, name).ratio()
            if needle and all(any(part.startswith(w) for part in _words(symbol.name)) for w in needle.split()):
                ratio = max(ratio, 0.85)
            return ratio if ratio >= 0.6 else 0.0
        # substring
        if name == needle:
            return 1.0
        if name.startswith(needle):
            return 0.9
        if needle in name:
            return 0.7 * (0.5 + 0.5 * len(needle) / max(len(name), 1)) + 0.1
        if needle in qname:
            return 0.4
        return 0.0

    def _decorated(self, symbol: SymbolRecord, wanted: str) -> bool:
        return any(item == wanted or _last(item) == _last(wanted) for item in symbol.decorators)

    def _inherits(self, symbol: SymbolRecord, base_query: str, cache: dict[str, frozenset[str]]) -> bool:
        if symbol.kind is not SymbolKind.CLASS:
            return False
        order = self.index.linearization(symbol.qualified_name)[0][1:]
        if any(item == base_query or _last(item) == _last(base_query) for item in order):
            return True
        bases = self.index.class_bases(symbol.qualified_name)
        return any(item == base_query or _last(item) == _last(base_query) for item in (*bases.external, *bases.unresolved))

    def _callers_of(self, text: str) -> frozenset[str]:
        found: set[str] = set()
        want = text.strip()
        for model in self.repo.parsed():
            for call in model.calls:
                dotted = ".".join(call.callee)
                if dotted == want or dotted.endswith("." + want) or (call.callee and call.callee[-1] == want):
                    found.add(model.owner_symbol(call.scope))
        return frozenset(found)

    # -- API ----------------------------------------------------------------------
    def search(self, query: SearchQuery) -> tuple[SearchHit, ...]:
        mode = query.mode if query.mode in MODES else None
        if mode is None:
            raise SoftwareRequestError("unsupported search mode", context={"mode": query.mode, "modes": sorted(MODES)})
        needle = (query.text or "").strip().lower()
        regex = None
        if mode == "regex" and query.text:
            if len(query.text) > _MAX_PATTERN:
                raise SoftwareRequestError("regex pattern too long", context={"limit": _MAX_PATTERN})
            try:
                regex = re.compile(query.text)
            except re.error as exc:
                raise SoftwareRequestError("invalid regular expression", context={"pattern": query.text[:80]}, cause=exc) from exc
        glob = compile_glob(query.path_glob) if query.path_glob else None
        callers = self._callers_of(query.calls) if query.calls else None
        kinds = set(query.kinds)
        limit = max(1, min(int(query.limit), 1000))
        hits: list[SearchHit] = []
        for path, symbol in self.index.symbols():
            if kinds and symbol.kind not in kinds:
                continue
            if not query.include_private and symbol.name.startswith("_") and not symbol.name.startswith("__"):
                continue
            if glob is not None and not glob.match(path):
                continue
            if query.decorator and not self._decorated(symbol, query.decorator):
                continue
            if query.base_class and not self._inherits(symbol, query.base_class, {}):
                continue
            if callers is not None and symbol.qualified_name not in callers:
                continue
            score = 1.0 if not needle and not regex else self._score(mode, needle, regex, symbol)
            if score <= 0.0:
                continue
            model = self.repo.model(path)
            hits.append(SearchHit(symbol.qualified_name, symbol.kind, path, symbol.name_span, round(score, 4), model.line(symbol.name_span.start_line).strip()[:200]))
        hits.sort(key=lambda hit: (-hit.score, hit.qualified_name))
        return tuple(hits[:limit])

    def search_text(self, pattern: str, *, regex: bool = False, ignore_case: bool = False, path_glob: Optional[str] = None, limit: int = 100) -> tuple[SearchHit, ...]:
        """Line search (grep) attributed to the innermost enclosing symbol."""

        if not pattern:
            raise SoftwareRequestError("search text must be non-empty")
        if len(pattern) > _MAX_PATTERN:
            raise SoftwareRequestError("search pattern too long", context={"limit": _MAX_PATTERN})
        flags = re.IGNORECASE if ignore_case else 0
        try:
            compiled = re.compile(pattern if regex else re.escape(pattern), flags)
        except re.error as exc:
            raise SoftwareRequestError("invalid regular expression", context={"pattern": pattern[:80]}, cause=exc) from exc
        glob = compile_glob(path_glob) if path_glob else None
        bound = max(1, min(int(limit), 1000))
        hits: list[SearchHit] = []
        for path in self.repo.paths():
            if glob is not None and not glob.match(path):
                continue
            model = self.repo.model(path)
            for number, line in enumerate(model.source.text.splitlines(), start=1):
                match = compiled.search(line)
                if match is None:
                    continue
                owner, kind = self._owner_at(model, number)
                hits.append(SearchHit(owner, kind, path, SourceSpan(number, match.start(), number, match.end()), 1.0, line.strip()[:200]))
                if len(hits) >= bound:
                    return tuple(hits)
        return tuple(hits)

    def _owner_at(self, model, line: int) -> tuple[str, SymbolKind]:
        best: Optional[SymbolRecord] = None
        for symbol in model.symbols:
            if symbol.kind is SymbolKind.VARIABLE:
                continue
            if symbol.full_span.start_line <= line <= symbol.span.end_line:
                if best is None or (symbol.span.end_line - symbol.full_span.start_line) < (best.span.end_line - best.full_span.start_line):
                    best = symbol
        if best is None:
            return model.module_name or model.path, SymbolKind.MODULE
        return best.qualified_name, best.kind


__all__ = ["CodeSearch", "MODES"]
