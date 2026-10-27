"""Typed settings read from an application's ``config.yaml`` through SLAI's shared ``ConfigRepository``.

The YAML file is only a source: every value is checked against a declared :class:`Field` (type and range) before the application sees
it, and a wrong value is an error that names the setting instead of a silent fallback. When the SLAI checkout (and with it
``ConfigRepository``) is not available, the declared defaults are used and the caller is told, so the application still starts.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class Field:
    default: Any
    low: float | None = None
    high: float | None = None
    kind: type = int


def _load_yaml(path: Path) -> tuple[dict[str, Any], str | None]:
    """The parsed file and, when the canonical repository could not be used, a note saying why."""
    try:
        from src.utils.configuration import bind_config              # the canonical repository
    except ImportError as exc:
        return {}, f"SLAI's ConfigRepository is not available ({exc}); the built-in defaults are used."
    return bind_config(path).load(), None


def load_section(path: Path | None, section: str, fields: Mapping[str, Field], *, label: str = "configuration",
                 overrides: Mapping[str, Any] | None = None) -> tuple[dict[str, Any], str | None]:
    """Return ``({name: value}, note)`` for one YAML section.

    ``overrides`` (command-line values) win over the file; ``None`` overrides are ignored. A missing file is not an error (defaults
    apply); an unreadable or invalid one is.
    """
    raw: dict[str, Any] = {}
    note: str | None = None
    if path is not None and path.is_file():
        data, note = _load_yaml(path)
        block = data.get(section, {})
        if block is None:
            block = {}
        if not isinstance(block, dict):
            raise ValueError(f"Invalid {label}: section '{section}' must be a mapping")
        raw = block
    elif path is not None:
        note = f"No configuration file at {path}; the built-in defaults are used."
    values: dict[str, Any] = {}
    for name, spec in fields.items():
        value = (overrides or {}).get(name)
        if value is None:
            value = raw.get(name, spec.default)
        if spec.kind is bool:
            ok = type(value) is bool
        elif spec.kind is int:
            ok = type(value) is int
        elif spec.kind is float:
            ok = isinstance(value, (int, float)) and not isinstance(value, bool)
        else:
            ok = isinstance(value, spec.kind)
        if ok and spec.kind in (int, float) and ((spec.low is not None and value < spec.low) or (spec.high is not None and value > spec.high)):
            ok = False
        if not ok:
            raise ValueError(f"Invalid {label}: {name}")
        values[name] = value
    return values, note
