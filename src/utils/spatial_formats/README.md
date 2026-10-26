# Native spatial infrastructure

This package implements the versioned SDRW drawing and SGEO scene models used
by SLAI v2.4. The native **file format remains 0.1.0**; it is separate from the
application version. Application facades in `src/utils/sdrw.py` and
`src/utils/sgeo.py` are not imported here.

## Responsibilities

| Module | Responsibility |
| --- | --- |
| `common` | Units, UUIDs, bounded JSON, metadata and affine transforms |
| `sdrw` | Layers, 2D primitives, measurements, constraints and drawing transactions |
| `sgeo` | Hierarchy, analytic/mesh geometry, solids, tessellation, typed families and transactions |
| `container` | Deterministic ZIP transport, hashes, binary mesh buffers and validated publication |
| `verification` | Local JSON schemas, semantic validation and conversion reports |
| `adapters` | Explicit, bounded exchange subsets and optional backend integration |

Coordinates use the document's declared length unit. SGEO is right-handed,
+Z up; transforms are flat row-major 4×4 matrices acting on column vectors.
Angles are radians. Area and volume measurements use squared/cubed document
units. The parameter evaluator tracks length and angle dimensions separately.

## Drawing edits and saving

```python
from src.utils.spatial_formats import Drawing, load_package, write_document
from src.utils.spatial_formats.sdrw import apply_drawing_operations

plan = Drawing("Plan", unit="mm")
wall = plan.add("line", {"start": [0, 0], "end": [4000, 0]})
apply_drawing_operations(plan, [
    {"op": "add_layer", "name": "Walls"},
    {"op": "update", "id": wall, "value": {"layer": "Walls"}},
], expected_revision=plan.revision)
sha256 = write_document("plan.sdrw", plan)
restored = Drawing.from_package(load_package("plan.sdrw"))
```

Drawing batches support `add`, `remove`, `update`, `transform`, `add_layer`,
`remove_layer`, `rename_layer` and `layer_visibility`. Entity updates preserve
IDs and primitive kinds. Transform values use the existing 3×3 planar similarity
matrix convention. Renaming a layer updates its entity references; removing a
referenced layer fails. Caller-owned geometry, style and metadata are detached.

Both drawing and scene batches commit once, under the model's mutation lock.
Every operation validates the candidate; a failure leaves the original state
and revision unchanged. `expected_revision` provides optimistic concurrency.
Operations must be ordered so references exist when used. Public collections
remain mutable for compatibility: direct list/dictionary edits bypass these
transaction and revision guarantees. Use the mutation APIs for concurrent work.

`write_document` captures manifest and payload under one lock, then releases it
before encoding and I/O. `write_package` remains available for detached,
explicit manifest/payload callers. Saves validate a temporary archive before
publication, never overwrite without `overwrite=True`, and return the digest of
the exact published bytes. Atomic no-clobber saves require hardlink support on
the destination filesystem; unsupported filesystems raise `ConflictError`
without exposing a partial destination. Overwrites use atomic replacement.
This is an atomic visibility guarantee, not a cross-process edit lease or a
power-loss durability guarantee for the directory entry.

## Parametric family lifecycle

```python
from src.utils.spatial_formats import Scene
from src.utils.spatial_formats.common import new_id
from src.utils.spatial_formats.sgeo import apply_scene_operations

scene = Scene("Components", unit="m")
family = {
    "id": new_id(), "name": "Block",
    "generator": {"kind": "builtin.box_v1"},
    "parameters": {
        "width": {"type": "length", "default": 2, "min": 0.1},
        "depth": {"type": "length", "default": 1, "min": 0.1},
        "height": {"type": "length", "expression": "width * 2"},
    },
}
_, instance = apply_scene_operations(scene, [
    {"op": "add_family", "value": family},
    {"op": "instantiate", "family_id": family["id"], "parameters": {"width": 3}},
])
apply_scene_operations(scene, [
    {"op": "parameters", "id": instance, "value": {"width": 4}},
])
```

`parameters` replaces the instance override map; omitted parameters use family
defaults. `update_family` takes `id` and a complete replacement `value` with the
same ID, regenerating every dependent instance atomically. Invalid overrides or
geometry roll back the entire update. `remove_family` refuses definitions still
in use. Stored instance geometry must match the allowlisted generator and
resolved overrides, including when loading a file.

Parameter declarations are validated in one evaluator, shared by families and
standalone parameter evaluation. Derived parameters cannot also define defaults
or be overridden. Bounds and read-only flags are typed, dimensions are checked,
and generators require positive **length** parameters. Arbitrary executable
code is never used. The built-in generators remain box, sphere and cylinder.

`Scene.world_transforms()` resolves each parent chain once;
`Scene.bounds_by_object()` returns a consistent snapshot of geometric-object
AABBs. `objects_within_bounds` reuses that batch calculation. These bounds are
conservative broad-phase bounds, not exact intersection tests.

## Exchange scope and validation

- DXF (`ezdxf`): planar primitives, declared units, layer names and visibility.
  Unspecified/unsupported source units require an explicit `unit` override.
  An override interprets numeric coordinates in that unit; it does not rescale.
  Paper-space content, unsupported entities, bulges, elevations and non-default
  OCS extrusion are refused. Blocks, annotations and rich styling are not
  preserved.
- GLB (`trimesh`, `numpy`): triangle meshes and hierarchy, converted between
  SGEO +Z and glTF +Y. Scene export converts positions and local translations
  to metres; the standalone mesh export assumes metre-valued coordinates.
  Float32 conversion that collapses triangles is rejected. Preflight checks
  embedded buffer ranges, accessor sizes, references, graph cycles and expanded
  instance budgets before invoking the backend. Materials, animations, sparse
  accessors, morph targets and unsupported extensions are refused.
- SVG: a deliberately narrow UTF-8 primitive subset. Native `data-unit` survives
  round-trips. Viewport transforms, root styling, foreign namespaces, nested
  content and document declarations are refused. General path import, rich
  styling and full-sweep arc export remain unsupported.
- STL remains a triangle-only presentation format. DWG, STEP and IFC retain
  their explicit optional-backend contracts; this package does not provide a
  general B-rep kernel or promise lossless industrial CAD/BIM round-trips.

Native files validate against the bundled schemas using a local reference
registry, then against model invariants. ZIP sizes, compression ratios, hashes,
JSON structures and hydrated mesh expansion are bounded. Highly compressible
writer output uses stored ZIP members when needed to satisfy the same limits
as the reader. The schemas reject unknown/missing document fields rather than
allowing constructor defaults to repair malformed persisted documents.

## Regression checks

```sh
python -m pip install pytest pytest-cov jsonschema
python -m pytest tests/spatial_formats -W error
# Exercise real optional exchange backends as well:
python -m pip install numpy ezdxf trimesh
python -m pytest tests/spatial_formats -W error --cov=src.utils.spatial_formats
```

CI covers Python 3.12/3.13 on Linux and Windows with core-only and exchange-enabled
installations. The duplicate-ZIP-member test explicitly expects its construction
warning and still verifies reader rejection; warnings are not globally hidden.
