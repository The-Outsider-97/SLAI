"""Format versions, per-version feature sets and schema identifiers (dependency-free).

``CURRENT_VERSION`` is what this library writes. Every version in
``LEGACY_VERSIONS`` can still be *read*: the loader validates the file against
that version's frozen schema and upgrades it in memory (see ``migration``).
A file never silently changes version on disk: only an explicit save does.
"""
from __future__ import annotations

CURRENT_VERSION = "0.2.0"
LEGACY_VERSIONS = ("0.1.0",)
SUPPORTED_VERSIONS = LEGACY_VERSIONS + (CURRENT_VERSION,)

_V1_FEATURES = frozenset({"scene-graph", "triangle-mesh", "analytic-primitives", "drawing-primitives", "families"})

#: Manifest ``requires`` entries each version understands. A reader that meets
#: a feature it does not know must refuse the file instead of guessing.
FEATURES = {
    "0.1.0": _V1_FEATURES,
    "0.2.0": _V1_FEATURES | {"annotations", "blocks", "linetypes", "nurbs", "brep", "materials", "textures",
                               "animation-v1", "animation-v2"},
}

#: Content introduced after 0.1.0 (used by the strict legacy check before upgrading).
LEGACY_SDRW_KINDS = frozenset({"point", "line", "polyline", "circle", "arc"})
LEGACY_SDRW_STYLE_KEYS = frozenset({"stroke", "stroke_width"})
LEGACY_SGEO_KINDS = frozenset({"point", "segment", "box", "sphere", "cylinder", "mesh"})
LEGACY_SGEO_MESH_KEYS = frozenset({"kind", "vertices", "faces"})
LEGACY_SGEO_OBJECT_KEYS = frozenset({"id", "name", "parent", "transform", "geometry", "metadata", "family_id", "parameters"})


def schema_id(format_id: str, version: str) -> str:
    """``slai.sdrw`` + ``0.2.0`` -> ``slai.sdrw/0.2`` (the manifest's ``schema`` value)."""
    major, minor = version.split(".")[:2]
    return "%s/%s.%s" % (format_id, major, minor)
