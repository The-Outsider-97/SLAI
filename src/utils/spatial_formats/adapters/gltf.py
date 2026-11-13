"""glTF 2.0 / GLB scene exchange (facade over the native codec in :mod:`.glb`).

The SGEO scene unit is metres, right-handed +Z up. glTF is right-handed +Y up. Coordinates and node transforms are
converted by a proper +90 degree X rotation; a non-identity root transform is never silently discarded.

Since SGEO 0.2 the codec is native (no trimesh needed) and carries PBR materials, embedded PNG/JPEG textures and
UVs. It still refuses animated, skinned, extension-using, externally-resourced and non-triangle content, because
silently skipping it would misreport what a file contains. Mesh exports are presentation geometry only.
"""
from __future__ import annotations
from ..common.errors import DependencyUnavailableError, UnsupportedFeatureError
from ..common.transforms import IDENTITY, compose, require_transform
from ..sgeo.geometry import validate_geometry
from ..sgeo.scene import Scene
from ..common.errors import ValidationError
from .contracts import Capabilities
from .glb import MAX_GLB_BYTES, Y_UP_TO_Z, Z_UP_TO_Y, read_glb, write_glb

MAX_GLTF_BYTES = MAX_GLB_BYTES


def _trimesh():
    """The optional trimesh package (no longer needed for GLB; kept for callers that still probe it)."""
    try:
        import trimesh
    except ImportError as exc:
        raise DependencyUnavailableError("trimesh is not installed (it is no longer required for GLB)") from exc
    return trimesh


def capabilities():
    return Capabilities("glTF", "native", True, ("2.0",), ("2.0",),
                        ("triangle-mesh", "node-hierarchy", "instances", "pbr-materials", "textures", "uvs"),
                        ("No animation/skin round-trip", "No extensions (e.g. KHR_texture_transform)",
                         "Only PNG/JPEG embedded images", "No exact CAD/BIM semantics: NURBS and B-reps are tessellated"))


def _matrix_to_row_major(matrix):
    import numpy as np
    m = np.asarray(matrix, dtype=float)
    if m.shape != (4, 4):
        raise ValidationError("invalid glTF node transform shape")
    return require_transform(m.reshape(16).tolist())


def _conjugate(transform, to_z=True):
    basis = Y_UP_TO_Z if to_z else Z_UP_TO_Y
    inverse = Z_UP_TO_Y if to_z else Y_UP_TO_Z
    return compose(compose(basis, transform), inverse)


def _map_vertices(vertices, *, to_z: bool):
    """Apply the exact proper basis change to position vectors (numpy array in, numpy array out)."""
    import numpy as np
    points = np.asarray(vertices, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValidationError("invalid glTF vertex array")
    if to_z:
        converted = np.column_stack((points[:, 0], -points[:, 2], points[:, 1]))
    else:
        converted = np.column_stack((points[:, 0], points[:, 2], -points[:, 1]))
    if not np.isfinite(converted).all():
        raise ValidationError("nonfinite mesh vertices")
    return converted


def import_glb_scene(data: bytes) -> Scene:
    """Parse GLB bytes into a :class:`Scene` (unit ``m``, +Z up). See :func:`.glb.read_glb` for the notes."""
    return read_glb(data)[0]


def import_glb(data: bytes) -> dict:
    """Legacy single-mesh import. Rejects scenes that would lose hierarchy."""
    scene = import_glb_scene(data)
    objects = [o for o in scene.objects if o.get("geometry") is not None]
    if len(objects) != 1 or len(scene.objects) != 1:
        raise UnsupportedFeatureError("single-mesh API cannot preserve multi-node GLB")
    if tuple(objects[0]["transform"]) != IDENTITY:
        raise UnsupportedFeatureError("single-mesh API cannot discard node transform")
    return objects[0]["geometry"]


def export_glb(geometry: dict) -> bytes:
    """One triangle mesh as a GLB (metres, +Y up)."""
    validate_geometry(geometry)
    if geometry["kind"] != "mesh":
        raise UnsupportedFeatureError("GLB mesh export requires triangles")
    scene = Scene(name="mesh", unit="m")
    scene.add_object(name="mesh", geometry=geometry)
    return write_glb(scene)


def export_glb_scene(scene: Scene) -> bytes:
    """Export triangle meshes, hierarchy, materials and textures; refuse unsupported semantics."""
    scene.validate()
    if scene.families or scene.metadata:
        raise UnsupportedFeatureError("GLB cannot preserve parametric families or scene semantic metadata")
    return write_glb(scene)
