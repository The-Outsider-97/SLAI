"""Exact planar-face (mitered) offsets of SGEO axis-aligned boxes.

Not a general surface offset or Euclidean/Minkowski offset: corners remain
sharp and the six planes move outward/inward by the signed distance.
"""
from __future__ import annotations

from ..common.errors import UnsupportedFeatureError, ValidationError
from ..common.helpers import finite
from .geometry import validate_geometry


def offset_box_faces(geometry: dict, distance: float) -> dict:
    validate_geometry(geometry)
    if geometry["kind"] != "box":
        raise UnsupportedFeatureError("planar-face offset requires an SGEO box")
    delta = finite(distance, "face offset", limit=1e12)
    new_size = [float(s + 2 * delta) for s in geometry["size"]]
    if any(s <= 0 for s in new_size):
        raise ValidationError("inward face offset collapses the box")
    return validate_geometry({"kind": "box", "center": list(geometry["center"]), "size": new_size})


__all__ = ["offset_box_faces"]
