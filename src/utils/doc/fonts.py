"""Reusable, read-only discovery and safe resolution of bundled document fonts.

The service does not render or embed fonts. Browser clients receive opaque IDs,
never absolute filesystem paths.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from typing import Any

FONT_ROOT = Path(__file__).resolve().parent / "fonts"
_FORMATS = {".ttf": "truetype", ".otf": "opentype", ".woff": "woff", ".woff2": "woff2"}


@dataclass(frozen=True)
class FontFace:
    id: str
    family: str
    subfamily: str
    weight: int
    italic: bool
    format: str
    path: Path
    variable_axes: tuple[tuple[str, float, float, float], ...] = ()
    coverage_count: int = 0
    license_file: str | None = None

    def public(self) -> dict[str, Any]:
        return {"id": self.id, "family": self.family, "subfamily": self.subfamily,
                "weight": self.weight, "italic": self.italic, "format": self.format,
                "axes": [{"tag": tag, "min": lo, "default": mid, "max": hi}
                         for tag, lo, mid, hi in self.variable_axes],
                "coverageCount": self.coverage_count, "license": self.license_file}


def _name(font: Any, *ids: int) -> str | None:
    for name_id in ids:
        for record in font["name"].names:
            if record.nameID == name_id:
                try:
                    value = record.toUnicode().strip()
                except (UnicodeError, ValueError):
                    continue
                if value:
                    return value
    return None


def _read_face(path: Path, root: Path) -> FontFace | None:
    try:
        from fontTools.ttLib import TTFont
        with path.open("rb") as stream, TTFont(stream, lazy=True) as font:
            family = _name(font, 16, 1)
            if not family:
                return None
            subfamily = _name(font, 17, 2) or "Regular"
            weight = int(font["OS/2"].usWeightClass) if "OS/2" in font else 400
            italic = bool(font["OS/2"].fsSelection & 1) if "OS/2" in font else "italic" in subfamily.casefold()
            axes = tuple((a.axisTag, float(a.minValue), float(a.defaultValue), float(a.maxValue))
                         for a in font["fvar"].axes) if "fvar" in font else ()
            coverage = len(font.getBestCmap() or {})
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning("Skipping invalid document font %s: %s", path.name, exc)
        return None
    rel = path.relative_to(root).as_posix()
    license_path = next((p for p in (path.parent / "OFL.txt", path.parent.parent / "OFL.txt")
                         if p.is_file() and p.resolve().is_relative_to(root)), None)
    return FontFace(sha256(rel.encode("utf-8")).hexdigest()[:24], family, subfamily,
                    weight, italic, _FORMATS[path.suffix.casefold()], path, axes, coverage,
                    license_path.relative_to(root).as_posix() if license_path else None)


@lru_cache(maxsize=4)
def discover_fonts(root: Path = FONT_ROOT) -> tuple[FontFace, ...]:
    """Index validated face metadata, ignoring malformed files and symlinks."""
    root = root.resolve()
    if not root.is_dir():
        return ()
    found: list[FontFace] = []
    for path in sorted(root.rglob("*")):
        if path.suffix.casefold() not in _FORMATS or not path.is_file() or path.is_symlink():
            continue
        if not path.resolve().is_relative_to(root) or path.stat().st_size > 40_000_000:
            continue
        face = _read_face(path, root)
        if face:
            found.append(face)
    return tuple(found)


def bundled_catalogue(root: Path = FONT_ROOT) -> dict[str, Any]:
    faces = discover_fonts(root)
    return {"families": sorted({f.family for f in faces}, key=str.casefold),
            "faces": [f.public() for f in faces]}


def resolve_bundled_font(face_id: str, root: Path = FONT_ROOT) -> FontFace | None:
    if len(face_id) != 24 or not all(ch in "0123456789abcdef" for ch in face_id):
        return None
    return next((f for f in discover_fonts(root) if f.id == face_id), None)
