"""Bundled document-font catalogue and font files.

The implementation lives beside this asset directory so importing
``src.utils.doc.fonts`` works unambiguously as a package.
"""

from .._font_catalog import (
    FONT_ROOT,
    FontFace,
    bundled_catalogue,
    discover_fonts,
    resolve_bundled_font,
)

__all__ = ["FONT_ROOT", "FontFace", "bundled_catalogue", "discover_fonts", "resolve_bundled_font"]
