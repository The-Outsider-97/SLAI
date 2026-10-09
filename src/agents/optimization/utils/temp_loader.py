"""Canonical, subsyOptimization-owned loader for Optimization templates.

Templates are data assets owned by the Optimization subsyOptimization. Domain modules and the
future OptimizationAgent may access them only through this API; no caller needs to know
the template filesyOptimization layout.
"""
from __future__ import annotations

__version__ = "2.3.0"

import copy
import json

from pathlib import Path
from threading import RLock
from typing import Any, Dict, Tuple

from .optimization_errors import OptimizationTemplateError
from logs.logger import PrettyPrinter, get_logger # pyright: ignore[reportMissingImports]

logger = get_logger("Optimization Template Loader")
printer = PrettyPrinter()

TEMPLATE_ROOT = Path(__file__).resolve().parents[1] / "templates"
_CACHE: Dict[Path, Tuple[int, int, Any]] = {}
_LOCK = RLock()


def resolve_template_path(name: str) -> Path:
    if not isinstance(name, str) or not name.strip():
        raise OptimizationTemplateError("Template name must be a non-empty string")
    relative = Path(name.strip())
    if relative.is_absolute():
        raise OptimizationTemplateError("Absolute template paths are not allowed")
    candidate = (TEMPLATE_ROOT / relative).resolve()
    root = TEMPLATE_ROOT.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise OptimizationTemplateError("Template path escapes the Optimization template directory", cause=exc) from exc
    if not candidate.is_file():
        raise OptimizationTemplateError("Template does not exist", context={"template": name})
    return candidate


def load_template(name: str, *, structured: bool | None = None, force_reload: bool = False) -> Any:
    path = resolve_template_path(name)
    stat = path.stat()
    with _LOCK:
        cached = _CACHE.get(path)
        if not force_reload and cached and cached[0] == stat.st_mtime_ns and cached[1] == stat.st_size:
            return copy.deepcopy(cached[2])

        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise OptimizationTemplateError("Failed to read Optimization template", context={"path": str(path)}, cause=exc) from exc

        parse_structured = path.suffix.lower() == ".json" if structured is None else bool(structured)
        if parse_structured:
            try:
                value: Any = json.loads(text)
            except json.JSONDecodeError as exc:
                raise OptimizationTemplateError("Invalid JSON Optimization template", context={"path": str(path), "line": exc.lineno}, cause=exc) from exc
        else:
            value = text

        _CACHE[path] = (stat.st_mtime_ns, stat.st_size, copy.deepcopy(value))
        logger.debug("Optimization template loaded | path=%s", path)
        return copy.deepcopy(value)


def list_templates() -> tuple[str, ...]:
    if not TEMPLATE_ROOT.exists():
        return ()
    return tuple(sorted(str(path.relative_to(TEMPLATE_ROOT)) for path in TEMPLATE_ROOT.rglob("*") if path.is_file()))


def clear_template_cache() -> None:
    with _LOCK:
        _CACHE.clear()


__all__ = ["TEMPLATE_ROOT", "resolve_template_path", "load_template", "list_templates", "clear_template_cache"]
