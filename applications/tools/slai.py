"""The connection between an application and the SLAI checkout it lives in.

An application consumes SLAI through its public modules only and never copies them:

* ``src.utils.sldc``           the authoritative SLDC implementation (documents are read and written only through it)
* ``src.utils.configuration``  ``ConfigRepository`` / ``bind_config`` for YAML configuration
* ``applications/assets``      the SLAI-wide icons, served from there and never copied into an application

If the checkout cannot be found the application still starts; features that need it raise
:class:`~applications.tools.errors.DependencyMissing`, which the client shows as a plain message.
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from .errors import DependencyMissing

_ROOT: Path | None = None


def looks_like_slai_root(path: Path) -> bool:
    return (path / "src" / "utils" / "sldc.py").is_file()


def find_slai_root(explicit: str | os.PathLike[str] | None = None) -> Path | None:
    """The SLAI checkout: ``explicit``, then ``$SLAI_ROOT``, then the folders above this file."""
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    if os.environ.get("SLAI_ROOT"):
        candidates.append(Path(os.environ["SLAI_ROOT"]))
    candidates.extend(Path(__file__).resolve().parents)
    for candidate in candidates:
        try:
            if looks_like_slai_root(candidate):
                return candidate.resolve()
        except OSError:
            continue
    return None


def activate(root: Path | None) -> Path | None:
    """Make ``import src...`` work. The SLAI root goes last on ``sys.path`` so an application's own packages win."""
    global _ROOT
    if root is None:
        return None
    if str(root) not in sys.path:
        sys.path.append(str(root))
    _ROOT = root
    return root


def slai_root() -> Path | None:
    return _ROOT


def applications_dir() -> Path:
    """``<SLAI>/applications``: the folder that holds ``tools``, ``assets`` and every application."""
    return Path(__file__).resolve().parents[1]


def shared_assets_dir() -> Path:
    """The SLAI-wide asset folder, ``applications/assets``."""
    return applications_dir() / "assets"


def sldc() -> ModuleType:
    """The authoritative SLDC implementation. An application defines no competing format."""
    if _ROOT is None and find_slai_root() is None:
        raise DependencyMissing("The SLAI checkout was not found, so SLDC documents are unavailable. Start the application inside the "
                                "SLAI folder, set SLAI_ROOT, or pass --slai-root.")
    try:
        return importlib.import_module("src.utils.sldc")
    except ImportError as exc:
        raise DependencyMissing("The SLDC module (src/utils/sldc.py) could not be imported.", details={"import_error": str(exc)}) from exc


def sldc_info() -> dict[str, Any]:
    try:
        module = sldc()
    except DependencyMissing as exc:
        return {"available": False, "reason": exc.message}
    return {"available": True, "module": module.__name__, "formatVersion": getattr(module, "FORMAT_VERSION", None),
            "legacyVersion": getattr(module, "LEGACY_VERSION", None)}


def status(root: Path | None) -> dict[str, Any]:
    """What an application reports about its SLAI connection."""
    return {"slaiRoot": str(root) if root else None, "sldc": sldc_info(), "assets": str(shared_assets_dir())}
