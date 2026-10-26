"""SGEO internal scene, geometry and family constructs."""
from .geometry import GEOMETRIES, validate_geometry, local_bounds, mesh_surface_area
from .families import validate_family, resolve_parameters, generate_geometry
from .scene import Scene

from .analysis import MeshDiagnostics, analyze_mesh, transformed_bounds
from .tessellation import tessellate

from .transactions import apply_scene_operations, objects_within_bounds

from .solids import extrude, revolve
