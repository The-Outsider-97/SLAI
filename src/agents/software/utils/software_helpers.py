"""Centralized helper functions for Software workflows.

Generic coercion, serialization, and hashing primitives are reused from SLAI's
base helper layer.  This module adds only Software-domain operations: repository
path hygiene, module naming, import resolution arithmetic, character-accurate
source positions, unified diffs, glob matching, and option coercion.

The helpers are dependency-light and never import parsers, models, graphs, or
analyzers, so every Software layer may depend on them.  Nothing here touches the
filesystem: the Software subsystem operates exclusively on caller-supplied
source text.
"""

from __future__ import annotations

__version__ = "2.4.0"

import bisect
import builtins
import dataclasses
import difflib
import hashlib
import keyword
import math
import re
import sys

from collections.abc import Iterable, Mapping, Sequence
from enum import Enum
from functools import lru_cache
from typing import Any, Optional

from ...base.utils.base_helpers import (
    coerce_bool,
    coerce_float,
    coerce_int,
    stable_fingerprint as _base_stable_fingerprint,
    to_json_safe as _base_to_json_safe,
    utc_now_iso,
)
from .software_errors import *
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Software Helpers")
printer = PrettyPrinter()


# ---------------------------------------------------------------------------
# Static vocabularies
# ---------------------------------------------------------------------------

BUILTIN_NAMES: frozenset[str] = frozenset(dir(builtins))

#: Names implicitly available at module scope without an explicit binding.
MODULE_IMPLICIT_NAMES: frozenset[str] = frozenset(
    {
        "__name__", "__file__", "__doc__", "__package__", "__spec__", "__loader__",
        "__path__", "__builtins__", "__cached__", "__annotations__", "__dict__",
        "__qualname__", "__module__", "__class__", "__debug__",
    }
)

STDLIB_MODULES: frozenset[str] = frozenset(getattr(sys, "stdlib_module_names", ())) | frozenset(
    sys.builtin_module_names
)

LANGUAGE_BY_SUFFIX: dict[str, str] = {
    ".py": "python", ".pyi": "python", ".pyw": "python",
    ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "typescript",
    ".java": "java", ".kt": "kotlin", ".scala": "scala",
    ".c": "c", ".h": "c", ".cc": "cpp", ".cpp": "cpp", ".hpp": "cpp", ".cxx": "cpp",
    ".cs": "csharp", ".go": "go", ".rs": "rust", ".rb": "ruby", ".php": "php",
    ".swift": "swift", ".sh": "shell", ".sql": "sql",
    ".yaml": "yaml", ".yml": "yaml", ".json": "json", ".toml": "toml", ".md": "markdown",
}

DEFAULT_TEST_PATTERNS: tuple[str, ...] = (
    "tests/**", "test/**", "**/tests/**", "**/test/**",
    "**/test_*.py", "**/*_test.py", "**/conftest.py",
)

PYTHON_SUFFIXES: tuple[str, ...] = (".py", ".pyi", ".pyw")


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def to_plain(value: Any, *, max_depth: int = 32) -> Any:
    """Convert Software values into deterministic, JSON-compatible structures.

    Dataclasses exposing ``to_dict`` are serialized through it; enums become
    their values; sets are sorted; mappings get string keys.  Anything else is
    delegated to SLAI's generic ``to_json_safe``.
    """

    if max_depth < 0:
        return "[MAX_DEPTH_REACHED]"
    if isinstance(value, Enum):  # before primitives: str-based enums are ``str`` instances
        return to_plain(value.value, max_depth=max_depth - 1)
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict) and not isinstance(value, type):
        return to_plain(to_dict(), max_depth=max_depth - 1)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: to_plain(getattr(value, field.name), max_depth=max_depth - 1)
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): to_plain(item, max_depth=max_depth - 1) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return [to_plain(item, max_depth=max_depth - 1) for item in sorted(value, key=repr)]
    if isinstance(value, (list, tuple)):
        return [to_plain(item, max_depth=max_depth - 1) for item in value]
    return _base_to_json_safe(value)


def stable_fingerprint(value: Any, *, algorithm: str = "sha256", length: int = 16) -> str:
    """Fingerprint a Software value after deterministic plain-data conversion."""

    return _base_stable_fingerprint(to_plain(value), algorithm=algorithm, length=length)


def stable_id(prefix: str, *parts: Any, length: int = 16) -> str:
    """Deterministic identifier ``<prefix>-<hex>`` derived from ``parts``."""

    return f"{prefix}-{stable_fingerprint(list(parts), length=length)}"


def content_digest(text: str) -> str:
    """Digest of source text (UTF-8, lone surrogates preserved)."""

    return hashlib.sha256(text.encode("utf-8", errors="surrogatepass")).hexdigest()[:32]


def combine_digests(items: Iterable[tuple[str, str]]) -> str:
    """Order-independent digest of ``(key, digest)`` pairs."""

    hasher = hashlib.sha256()
    for key, digest in sorted(items):
        hasher.update(key.encode("utf-8", errors="surrogatepass"))
        hasher.update(b"\x00")
        hasher.update(digest.encode("ascii"))
        hasher.update(b"\x01")
    return hasher.hexdigest()[:32]


def truncate_items(items: Sequence[Any], limit: int) -> tuple[list[Any], int]:
    """Return ``(first limit items, number omitted)``."""

    bound = max(0, int(limit))
    sequence = list(items)
    return sequence[:bound], max(0, len(sequence) - bound)


# ---------------------------------------------------------------------------
# Option coercion (request/option boundary)
# ---------------------------------------------------------------------------

def require_mapping(value: object, field: str, *, error_cls: type[SoftwareError] = SoftwareValidationError) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise error_cls(f"{field} must be a mapping", context={"field": field, "type": type(value).__name__})
    return value


def require_text(value: object, field: str, *, error_cls: type[SoftwareError] = SoftwareValidationError) -> str:
    if not isinstance(value, str) or not value.strip():
        raise error_cls(f"{field} must be a non-empty string", context={"field": field})
    return value.strip()


def opt_bool(options: Mapping[str, Any], key: str, default: bool) -> bool:
    return coerce_bool(options.get(key, default), default)


def opt_int(options: Mapping[str, Any], key: str, default: int, *, minimum: Optional[int] = None, maximum: Optional[int] = None) -> int:
    value = options.get(key, default)
    if isinstance(value, bool):
        raise SoftwareValidationError(f"{key} must be an integer", context={"field": key})
    return coerce_int(value, default, minimum=minimum, maximum=maximum)


def opt_float(options: Mapping[str, Any], key: str, default: float, *, minimum: Optional[float] = None, maximum: Optional[float] = None) -> float:
    value = options.get(key, default)
    if isinstance(value, bool):
        raise SoftwareValidationError(f"{key} must be a number", context={"field": key})
    return coerce_float(value, default, minimum=minimum, maximum=maximum)


def opt_strings(options: Mapping[str, Any], key: str, default: Sequence[str] = ()) -> tuple[str, ...]:
    """Normalize a string or sequence of strings into a de-duplicated tuple."""

    value = options.get(key, None)
    if value is None:
        return tuple(default)
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, Iterable):
        raise SoftwareValidationError(f"{key} must be a string or a sequence of strings", context={"field": key})
    out: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise SoftwareValidationError(f"{key} must contain non-empty strings", context={"field": key})
        text = item.strip()
        if text not in out:
            out.append(text)
    return tuple(out)


def clamp01(value: float) -> float:
    return 0.0 if value != value else max(0.0, min(1.0, float(value)))


def safe_ratio(numerator: float, denominator: float, default: float = 0.0) -> float:
    return default if not denominator else float(numerator) / float(denominator)


# ---------------------------------------------------------------------------
# Paths, languages and module naming
# ---------------------------------------------------------------------------

def normalize_repo_path(path: object, *, max_length: int = 1024) -> str:
    """Return a canonical repository-relative POSIX path.

    Absolute paths, drive letters, parent traversal, NUL bytes, and empty paths
    are rejected: a snapshot is a logical tree, never a window onto the host
    filesystem.
    """

    if not isinstance(path, str):
        raise SoftwareSnapshotError("repository path must be a string", context={"type": type(path).__name__})
    text = path.replace("\\", "/").strip()
    if not text or "\x00" in text:
        raise SoftwareSnapshotError("repository path must be non-empty and free of NUL bytes", context={"path": path[:200]})
    if text.startswith("/") or re.match(r"^[A-Za-z]:", text):
        raise SoftwareSnapshotError("repository path must be relative", context={"path": path[:200]})
    parts: list[str] = []
    for part in text.split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            raise SoftwareSnapshotError("repository path must not traverse parents", context={"path": path[:200]})
        parts.append(part)
    if not parts:
        raise SoftwareSnapshotError("repository path must name a file", context={"path": path[:200]})
    normalized = "/".join(parts)
    if len(normalized) > max_length:
        raise SoftwareSnapshotError("repository path is too long", context={"length": len(normalized), "max": max_length})
    return normalized


def language_for_path(path: str, mapping: Optional[Mapping[str, str]] = None) -> str:
    table = LANGUAGE_BY_SUFFIX if mapping is None else mapping
    lowered = path.lower()
    dot = lowered.rfind(".")
    return table.get(lowered[dot:], "unknown") if dot >= 0 else "unknown"


@lru_cache(maxsize=512)
def compile_glob(pattern: str) -> "re.Pattern[str]":
    """Compile a POSIX glob supporting ``*``, ``?`` and ``**`` into a regex."""

    out: list[str] = []
    index = 0
    while index < len(pattern):
        char = pattern[index]
        if pattern.startswith("**/", index):
            out.append("(?:.*/)?")
            index += 3
        elif pattern.startswith("/**", index) and index + 3 == len(pattern):
            out.append("(?:/.*)?")
            index += 3
        elif pattern.startswith("**", index):
            out.append(".*")
            index += 2
        elif char == "*":
            out.append("[^/]*")
            index += 1
        elif char == "?":
            out.append("[^/]")
            index += 1
        else:
            out.append(re.escape(char))
            index += 1
    return re.compile("^" + "".join(out) + "$")


def matches_any(path: str, patterns: Iterable[str]) -> bool:
    return any(compile_glob(pattern).match(path) for pattern in patterns)


def is_test_path(path: str, patterns: Sequence[str] = DEFAULT_TEST_PATTERNS) -> bool:
    return matches_any(path, patterns)


def normalize_source_roots(roots: Iterable[str] | None) -> tuple[str, ...]:
    """Canonicalize source roots; the empty string denotes the repository root."""

    if not roots:
        return ("",)
    out: list[str] = []
    for root in roots:
        text = str(root).replace("\\", "/").strip().strip("/")
        if text in ("", "."):
            text = ""
        elif ".." in text.split("/"):
            raise SoftwareSnapshotError("source root must not traverse parents", context={"root": str(root)[:200]})
        if text not in out:
            out.append(text)
    return tuple(sorted(out, key=lambda item: (-len(item), item)))


def module_name_for_path(path: str, source_roots: Sequence[str] = ("",)) -> tuple[str, bool]:
    """Return ``(dotted_module_name, is_package)`` for a Python file path.

    The longest matching source root is stripped.  ``pkg/__init__.py`` maps to
    ``pkg`` with ``is_package=True``; a root-level ``__init__.py`` maps to ``""``.
    """

    relative = path
    for root in sorted(source_roots, key=len, reverse=True):
        if root and path.startswith(root + "/"):
            relative = path[len(root) + 1:]
            break
    for suffix in PYTHON_SUFFIXES:
        if relative.endswith(suffix):
            relative = relative[: -len(suffix)]
            break
    parts = [part for part in relative.split("/") if part]
    is_package = bool(parts) and parts[-1] == "__init__"
    if is_package:
        parts.pop()
    return ".".join(parts), is_package


def package_of(module_name: str, is_package: bool) -> str:
    """Package that owns ``module_name`` (itself when it is a package)."""

    if is_package:
        return module_name
    return module_name.rpartition(".")[0]


def resolve_relative_import(current_module: str, is_package: bool, level: int, module: Optional[str]) -> Optional[str]:
    """Resolve ``from <.*>module import ...`` to an absolute dotted name.

    Returns ``None`` when the relative import climbs above the top-level package.
    """

    if level <= 0:
        return module or None
    package = package_of(current_module, is_package)
    parts = package.split(".") if package else []
    climb = level - 1
    if climb > len(parts):
        return None
    base_parts = parts[: len(parts) - climb] if climb else parts
    if not base_parts and not module:
        return None
    base = ".".join(base_parts)
    if module:
        return f"{base}.{module}" if base else module
    return base or None


def top_level_name(dotted: str) -> str:
    return dotted.split(".", 1)[0]


def is_stdlib_module(dotted: str) -> bool:
    return top_level_name(dotted) in STDLIB_MODULES


def qualified_join(*parts: str) -> str:
    return ".".join(part for part in parts if part)


def parent_qualified(name: str) -> str:
    return name.rpartition(".")[0]


def is_within_scope(scope: str, ancestor: str) -> bool:
    """True when ``scope`` equals ``ancestor`` or is lexically nested inside it."""

    return scope == ancestor or scope.startswith(ancestor + ".")


def is_valid_identifier(name: object) -> bool:
    return isinstance(name, str) and name.isidentifier() and not keyword.iskeyword(name)


def require_identifier(name: object, field: str = "name") -> str:
    if not is_valid_identifier(name):
        raise SoftwareValidationError(f"{field} must be a valid non-keyword identifier", context={"field": field, "value": str(name)[:100]})
    return str(name)


# ---------------------------------------------------------------------------
# Text positions
# ---------------------------------------------------------------------------

_NEWLINE_RE = re.compile(r"\r\n|\r|\n")


def split_lines_keepends(text: str) -> list[str]:
    """Split on ``\\n``, ``\\r\\n`` and ``\\r`` only (the tokenizer's definition)."""

    lines: list[str] = []
    last = 0
    for match in _NEWLINE_RE.finditer(text):
        lines.append(text[last:match.end()])
        last = match.end()
    if last < len(text):
        lines.append(text[last:])
    return lines


class LineIndex:
    """Map between ``(line, column)`` pairs and absolute character offsets.

    Lines are 1-based.  Columns passed to :meth:`offset_from_utf8` follow
    :mod:`ast` (UTF-8 byte offsets); all other columns are character offsets.
    """

    __slots__ = ("text", "_starts", "_ascii")

    def __init__(self, text: str) -> None:
        self.text = text
        starts = [0]
        starts.extend(match.end() for match in _NEWLINE_RE.finditer(text))
        self._starts = starts
        self._ascii = text.isascii()

    @property
    def line_count(self) -> int:
        return len(self._starts)

    def line_bounds(self, line: int) -> tuple[int, int]:
        """Return ``(start, end)`` offsets of a line, excluding its terminator."""

        if not 1 <= line <= len(self._starts):
            raise SoftwareValidationError("line out of range", context={"line": line, "lines": len(self._starts)})
        start = self._starts[line - 1]
        end = self._starts[line] if line < len(self._starts) else len(self.text)
        while end > start and self.text[end - 1] in "\r\n":
            end -= 1
        return start, end

    def line_text(self, line: int) -> str:
        start, end = self.line_bounds(line)
        return self.text[start:end]

    def offset(self, line: int, column: int) -> int:
        """Offset for a ``(line, character column)`` pair."""

        start, end = self.line_bounds(line)
        return min(start + max(0, column), end if column >= 0 else start)

    def offset_from_utf8(self, line: int, byte_column: int) -> int:
        start, end = self.line_bounds(line)
        if self._ascii:
            return min(start + byte_column, end)
        encoded = self.text[start:end].encode("utf-8", errors="surrogatepass")
        return start + len(encoded[:byte_column].decode("utf-8", errors="ignore"))

    def char_column(self, line: int, byte_column: int) -> int:
        """Convert an ``ast`` byte column into a character column."""

        if self._ascii:
            return byte_column
        start, end = self.line_bounds(line)
        encoded = self.text[start:end].encode("utf-8", errors="surrogatepass")
        return len(encoded[:byte_column].decode("utf-8", errors="ignore"))

    def position(self, offset: int) -> tuple[int, int]:
        """Return ``(line, character column)`` for an absolute offset."""

        offset = max(0, min(offset, len(self.text)))
        line = bisect.bisect_right(self._starts, offset)
        return line, offset - self._starts[line - 1]


# ---------------------------------------------------------------------------
# Diffs
# ---------------------------------------------------------------------------

def unified_diff(old: str, new: str, old_label: str, new_label: str, *, context: int = 3) -> str:
    """Git-compatible unified diff with ``\\ No newline at end of file`` markers."""

    if old == new:
        return ""
    old_lines = split_lines_keepends(old)
    new_lines = split_lines_keepends(new)
    out: list[str] = []
    for line in difflib.unified_diff(old_lines, new_lines, fromfile=old_label, tofile=new_label, n=context, lineterm="\n"):
        if line.endswith("\n"):
            out.append(line)
        else:
            out.append(line + "\n\\ No newline at end of file\n")
    return "".join(out)


def count_diff_lines(diff: str) -> tuple[int, int]:
    """Return ``(added, removed)`` line counts of a unified diff."""

    added = removed = 0
    for line in diff.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1
    return added, removed


__all__ = [
    "BUILTIN_NAMES",
    "DEFAULT_TEST_PATTERNS",
    "LANGUAGE_BY_SUFFIX",
    "LineIndex",
    "MODULE_IMPLICIT_NAMES",
    "PYTHON_SUFFIXES",
    "STDLIB_MODULES",
    "clamp01",
    "coerce_bool",
    "coerce_float",
    "coerce_int",
    "combine_digests",
    "compile_glob",
    "content_digest",
    "count_diff_lines",
    "is_stdlib_module",
    "is_test_path",
    "is_valid_identifier",
    "is_within_scope",
    "language_for_path",
    "matches_any",
    "module_name_for_path",
    "normalize_repo_path",
    "normalize_source_roots",
    "opt_bool",
    "opt_float",
    "opt_int",
    "opt_strings",
    "package_of",
    "parent_qualified",
    "qualified_join",
    "require_identifier",
    "require_mapping",
    "require_text",
    "resolve_relative_import",
    "safe_ratio",
    "split_lines_keepends",
    "stable_fingerprint",
    "stable_id",
    "to_plain",
    "top_level_name",
    "truncate_items",
    "unified_diff",
    "utc_now_iso",
]
