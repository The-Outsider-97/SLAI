# Native spatial infrastructure

This package implements the versioned SDRW drawing and SGEO scene models used
by SLAI v2.4. The native **file format is 0.2.0** (files written by 0.1.0 are
read and upgraded in memory, see [Format 0.2.0](#format-020)); it is separate
from the application version. Application facades in `src/utils/sdrw.py` and
`src/utils/sgeo.py` are not imported here (they import this package, never the
reverse; a test enforces it).

**Most callers want the facades**, not the modules below:

```python
from src.utils.sdrw import SDRWDocument
from src.utils.sgeo import SGEODocument

plan = SDRWDocument.new("Plan A", unit="mm")
wall = plan.add_line((0, 0), (1500, 0), layer="walls", create_layer=True)
plan.attach_semantics(wall, {"category": "wall"}, namespace="aec")   # stored as "aec:category"
plan.validate().raise_for_errors()          # structured ValidationReport (strict by default)
plan.save("plan.sdrw")                      # atomic; refuses to overwrite unless overwrite=True

room = SGEODocument.new("Room", unit="m")
wall = room.add_box(width=3.0, depth=0.2, height=2.7, position=(0, 0, 1.35))
with SGEODocument.open("room.sgeo") as scene:        # read-only by default; mode="write" to edit
    ids = scene.query_ids(type_name="box", limit=100)
    info = scene.inspect()
```

`SDRWDocument`/`SGEODocument` wrap one validated model with a lifecycle:
`new`/`open`/`from_bytes` → edit → `validate` → `save`. Every edit is atomic and
bumps the revision once; `apply([...ops], expected_revision=n)` runs a batch as
**one** revision (all-or-nothing, at most 1000 operations) and raises
`ConflictError` if someone else changed the document first.

### Safety model

* **Read-only by default.** `open()` returns a read-only document; edits raise
  `ReadOnlyDocumentError` (`spatial.read_only`). `close()` (or leaving a `with`
  block) forbids further edits and saves; in-memory reads stay available.
* **No lost updates.** `save()` to a path never replaces an existing file unless
  `overwrite=True`. Saving back to the file a document was opened from also
  checks that the file still has the digest it was loaded with
  (`expected_sha256` / `expected_revision` pin it explicitly) and refuses to
  replace an unrelated, corrupt or different-document file without `force=True`.
* **Atomic, verified publication.** The package is built in memory, re-read
  through the full validator, fsynced, then published with a no-clobber
  hardlink (or `os.replace` for overwrite; a rename fallback on Windows).
  A failed save leaves the previous file untouched and no temporaries.
* **Structured reports.** `validate()` returns a `ValidationReport` with every
  issue (`error`/`warning`/`info`, stable `code`, JSON pointer `path`);
  `ValidationReport.raise_for_errors()` raises the typed exception. Every
  import/export returns a `ConversionReport` (accuracy class, dropped,
  approximated, assumed). `policy=` (`strict`, `lossless_where_supported`,
  `allow_approximation`, `best_effort`) decides what a lossy conversion may do.
* **Typed errors** with stable `.code` values (`spatial.validation`,
  `spatial.schema`, `spatial.integrity`, `spatial.conflict`, `spatial.read_only`,
  `spatial.unsupported`, `spatial.limit`, `spatial.security`, `spatial.dependency`, `spatial.loss`, …);
  `error_for_code(code)` maps a code back to its class.

### Metadata, units and limits

* Metadata keys are `namespace:name` or `slai.*`. `attach_semantics(id, values,
  namespace="aec")` prefixes bare keys; a dotted key such as `aec.category`
  is rejected with a hint to use `aec:category`.
* Lengths are in the document unit. `convert_unit()` rescales geometry and
  refuses scenes that contain parametric families (their parameters carry
  units of their own). Agent records default to metres.
* Budgets: 300k structural JSON nodes per member; 1.5M for a whole document;
  16 MiB per JSON member; at most 1536 ZIP members per package; 100k entities /
  10k layers / 10k blocks / 1000 linetypes per drawing; 100k objects /
  10k families / 10k materials / 1000 images per scene; 1000 operations per batch. In practice about 75–80k
  simple entities fit under the node budget. Saving costs roughly 2.5 s per 10k
  entities because the writer validates, serialises and re-reads the package.
* SVG import allocates fresh entity ids unless `preserve_ids=True`. DXF and SVG
  export a full-sweep arc as a circle; arcs with an equal start and end angle do
  not exist.
* SGEO exports to GLB (metres, +Y up) and STL (merged mesh, no units) tessellate
  analytic primitives, NURBS and B-reps; both reports list exactly what that cost.

### Command line

```sh
python -m src.utils.spatial_formats inspect  plan.sdrw [--verify digests]
python -m src.utils.spatial_formats validate room.sgeo [--mode strict|basic]
python -m src.utils.spatial_formats convert  plan.sdrw plan.svg [--policy strict] [--overwrite]
python -m src.utils.spatial_formats convert  drawing.dxf plan.sdrw
python -m src.utils.spatial_formats capabilities
```

One JSON document on stdout; exit code 0 ok, 1 failed/invalid (`code` field),
2 usage error. GLB and STL need no backend (native codecs). DXF needs
`ezdxf`; install it with `pip install ".[spatial]"` (the extra also lists
`trimesh`, which is only used by tests as an independent GLB reader).

## Format 0.2.0

`format_version` is `0.2.0` for both formats. A 0.1.0 package still opens: it is
validated against the frozen 0.1 schema *and* a content rule (a 0.1.0 file may not
contain anything newer, even if the JSON would otherwise pass), then upgraded in
memory by adding the empty registries. `source_version` tells which version a
document was read from; saving always writes 0.2.0 and never touches the original
file. The manifest `requires` list is computed from the content, so a drawing or
scene that uses none of the new features still declares only the old ones. A reader
refuses a `requires` entry or a version it does not know instead of guessing.

### SDRW: text, dimensions, hatches, blocks, linetypes

| Feature | Model |
| --- | --- |
| `text` | `position`, `text` (up to 10,000 characters, newlines and tabs allowed), `height`, `rotation`, `halign` (`left`/`center`/`right`), `valign` (`baseline`/`bottom`/`middle`/`top`), optional `width` (wrap), `line_spacing` (0.25 to 4), `font`. `text_layout()` estimates the line breaks and extents; it does not shape real fonts. |
| `dimension` | `dim_type` is `linear`, `aligned`, `angular`, `radial` or `diameter`, defined by its points (`p1`, `p2`, `dimline_point`, `angle`; `center`, `arc_point`; `point`). Optional `text` override, `text_height`, `text_position`, `precision` (0 to 8), `suffix`, `arrow_size`. The measured value and display string are **derived** (`dimension_value`, `dimension_text`), never stored, so they cannot go stale; `dimension_primitives` gives the lines, arrowheads and text that draw it. |
| `hatch` | `boundary` is 1 to 256 loops (`polygon`, `circle` or `path` of line and arc segments; loops may not cross), `pattern` is `solid`, `lines` or `cross` (patterned hatches need `spacing`), `angle`. Area is derived from the boundary. |
| `linetype` | A named dash pattern in drawing units, DXF style: positive is a dash, negative a gap, `0` a dot. `continuous` is built in. Layers and entity styles reference one by name (`linetype`, `linetype_scale`); a linetype in use cannot be removed. |
| `block` and `insert` | A block is `{name, base_point, entities}`; an `insert` places it at `position` with `rotation` and a **uniform positive** `scale` (mirrored or non-uniform inserts would turn circles and arcs into curves the model cannot hold). Blocks may nest at most 8 deep, must be acyclic and expand to at most 2,000,000 primitives. Content on layer `0` takes the layer of the insert when exploded. `explode()` replaces inserts by their content. |

The editing API is `add_text`, `add_dimension_linear/aligned/angular/radial/diameter`,
`add_hatch`, `define_linetype`, `define_block`, `add_insert` and `explode`; each edit is
one atomic revision.

### SGEO: NURBS, B-rep, materials, textures

**NURBS.** `nurbs_curve` and `nurbs_surface` carry `control_points`, `degree` (1 to 12),
clamped `knots` and optional positive `weights`; the document helpers fill uniform clamped
knots. Rational weights represent circular arcs exactly. A surface is a rectangular grid
(at most 65,536 control points). `analyze` and `surface_area` measure a surface on a
tessellation.

**B-rep.** `brep` is a native boundary representation stored as JSON:
`{vertices, curves, surfaces, edges, faces}`.

* Edge carriers: line, circle, nurbs. Face carriers: plane, cylinder, cone, sphere, torus,
  nurbs. A face has one outer loop and any number of hole loops of oriented coedges
  (`edge`, `forward`, optional `pcurve`), so faces are *trimmed*.
* Validation checks every index, that each loop is connected and each edge meets its
  vertices, that a *closed* shell uses every non-degenerate edge exactly twice with opposite
  senses, and the Euler characteristic. An open shell is a valid sheet (it has no volume).
* Faces on periodic surfaces (cylinder, cone, sphere, torus) must be cut by seam edges that
  appear in the loop twice; degenerate pole edges need a `pcurve`. The builders below do this
  for you.
* The natural normal is dP/du x dP/dv and points outward unless `reversed` is set.
  An inward-facing closed solid is reported (`brep.inverted`, negative volume), and `volume()` refuses it.
* Builders: `brep_box`, `brep_cylinder` (also cones through `radius_top`), `brep_sphere`,
  `brep_torus`, `brep_extrusion(profile, depth, holes)` (concave profiles and holes; self-intersecting
  or overlapping loops are refused) and `brep_from_nurbs_surface` (an open sheet). The document
  methods are `add_brep`, `add_brep_box`, `add_brep_cylinder`, `add_brep_sphere`, `add_brep_torus`,
  `add_brep_extrusion` and `add_brep_sheet`.
* **There is no general B-rep modelling kernel**: no curved-solid booleans, fillets,
  chamfers or general surface offsets, and STEP geometry is not read (only the
  optional STEP backend's validation exists). Native B-rep editing remains scaling,
  rigid transforms and complete replacement. The restricted rectilinear box
  operations below are separate and do **not** alter this limitation.
* **Measures are tessellation based.** `volume`, `surface_area` and `analyze` mesh the faces
  (`resolution`, default 64 segments per circle, at most 128). They are exact for planar solids and converge
  for curved ones: about 0.5 % volume error for a sphere at 128 segments. Shared edge samples make the
  mesh watertight; the per-face budget is 60,000 triangles and the whole mesh obeys the scene mesh limits.
  `world_bounds` of a curved B-rep is a conservative box (a few percent loose), exact for flat solids.

### Rectilinear modeling and native animation extension

SGEO exposes two deliberately restricted, deterministic CAD operations:

* `boolean_boxes(first_id, second_id, operation)` accepts only **world-axis-aligned boxes with
  translation-only world transforms**. `operation` is `union`, `intersection` or `difference`.
  The result is an indexed mesh with an exact planar polyhedral boundary, not an editable B-rep or
  a kernel feature history. Empty results return `None` without a revision. Edge/corner-only
  contact that produces non-manifold geometry is explicitly rejected.
* `offset_box_faces(object_id, distance)` moves all six box planes inward/outward, preserving
  sharp, mitered corners. Object/ancestor transforms must be rigid. It creates a *new* object;
  a collapsing inward offset is rejected. This is not a round-edged Euclidean solid offset.

Neither operation provides fillets, arbitrary rotations in a Boolean operand, NURBS/B-rep
Booleans, general offset surfaces or STEP conversion. Those require a verified professional
CAD-kernel backend with explicit topological reference mapping.

Versioned, declarative animation clips can be stored at document metadata key
`slai.animation.v1` without changing the SGEO 0.2.0 container. This is an **opt-in metadata
extension**: older readers can open the file but may display only the static pose. Each clip has
`duration` in seconds and bounded `tracks`, each with target `object_id`, `property`,
`interpolation`, and strictly increasing `{time, value}` keyframes. Supported properties are:

* `translation`: local translation delta `(x, y, z)`, step/linear interpolation.
* `rotation_angle`: scalar angle in radians about a fixed local `axis`, step/linear interpolation.
* `parameter.NAME`: independent numerical family parameter, step/linear interpolation;
  `integer` parameters require step interpolation.
* `visibility`: boolean, step interpolation only; visibility inherits to children.

Animation is evaluated with `document.evaluate(time_seconds)` into a **detached state**.
Static transforms remain authoritative; animation deltas compose after them. Family parameters
are regenerated only on the detached state; repeated nonsequential evaluation leaves the original
project unchanged. Time beyond the first/last keyframe is clamped. There is no looping, IK,
skin deformation, quaternion interpolation, frame-rate playback, temporal constraint solver, or
continuous collision detection yet. The extension validates target references on load/save and
rejects malformed or excessive tracks; `set_animation(None)` removes it atomically.

```python
from src.utils.sgeo import SGEODocument

doc = SGEODocument.new("Moving assembly")
moving = doc.add_box(size=[1, 1, 1])
doc.set_animation({
    "duration": 2.0,
    "tracks": [{"object_id": moving, "property": "translation", "interpolation": "linear",
                "keyframes": [{"time": 0.0, "value": [0, 0, 0]},
                              {"time": 2.0, "value": [2, 0, 0]}]}],
})
assert doc.evaluate(1.0).world_bounds(moving)[0][0] == 0.5
doc.save("moving.sgeo")
```

**Materials.** A material is a glTF-style metallic-roughness record: `base_color` (linear RGBA),
`metallic`, `roughness`, `emissive`, `alpha_mode` (`opaque`, `mask` with `alpha_cutoff`, `blend`),
`double_sided`, and up to five texture slots (`base_color`, `metallic_roughness`, `normal` with `scale`,
`occlusion` with `strength`, `emissive`), each with `image`, `wrap_u`/`wrap_v` (`repeat`, `clamp`,
`mirror`) and `filter` (`linear`, `nearest`). An object points at one material with `material`; there
is no inheritance, and only an object with geometry can have one. A material in use, or an image a material
samples, cannot be removed. The API is `add_material`, `update_material`, `set_texture`, `clear_texture`,
`set_material`, `material_of`, `remove_material`.

**Textures.** `add_image` embeds a PNG or JPEG. The bytes are checked **structurally** (signature, chunk
and marker layout, checksums, dimensions) and **never decoded**, so a hostile image cannot reach a decoder
through this package. Limits: 1000 images, 32 MiB each and 64 MiB in total, at most 16,384 pixels per side
and 8192 x 8192 pixels in all. Identical bytes are stored once. Images travel as `textures/sha256-<hex>.png|jpg`
members, byte for byte; the scene lists them in `images` with the member `path`.

**UV coordinates.** A mesh may carry `uvs`, one `[u, v]` per vertex. The origin is the **bottom-left** of the
image with `v` pointing up (glTF's top-left origin is converted on GLB export and import). Values may exceed
`[0, 1]` for tiling. `tessellate(..., uvs=True)` generates them: each box face is a unit square, a
cylinder wraps `u` once around, a sphere maps longitude/latitude, B-reps and NURBS surfaces normalise per
face. Mesh vertices, faces and UVs are stored in content-addressed `geometry/sha256-<hex>.bin` members
(f64 vertices and UVs, u32 faces); NURBS and B-reps stay inline JSON.

### Exchange of the new content

* **DXF** (`ezdxf`), SDRW to DXF and back: points, lines, polylines, circles and arcs as before;
  `text` as `TEXT`, or `MTEXT` when it has several lines or a wrap width; `dimension` as `DIMENSION` with a
  rendered block so any viewer shows it; `hatch` as `HATCH` (polyline or edge boundaries, solid or
  user-defined lines); `insert` and `block` as `INSERT` and `BLOCK`; linetypes as `LTYPE` entries; true colour,
  lineweight, linetype and linetype scale. What cannot be represented is refused, or skipped with
  `skip_unsupported=True`; either way the report lists it. Known approximations: multi-line text uses the CAD
  pitch of 5/3 x text height; dimensions are drawn by the reader's dimension style, so arrowheads can differ;
  on import `DIMLFAC`, non-decimal units and unreachable blocks (not used by model space) are ignored, and
  hatch patterns are approximated by lines.
* **SVG** is an *export* format for the new kinds. Text becomes `<text>`, hatches become paths, dimensions become
  lines and text, inserts are expanded, and linetypes become `stroke-dasharray`. Such files cannot be imported back
  (they are refused, not guessed); the original primitives still round-trip.
* **GLB**, SGEO to GLB and back, native codec (no `trimesh`): the scene graph becomes nodes, geometry is tessellated
  to triangles (analytic, NURBS and B-rep objects lose their exact definition, which the report states),
  materials, textures and UVs are written and read back, and object ids are kept in `extras.slai.id` so identity
  survives. Z-up to Y-up, metres, and float32 storage are the documented conversions. On import, extensions
  (including `KHR_*`), animations, skins, sparse accessors, external or data-URI buffers, `TEXCOORD_1`,
  non-triangle modes and sheared node matrices are refused rather than skipped; triangles collapsed by float32
  are dropped and reported.
* **STL** drops materials, textures and UVs (the report says so) and tessellates B-reps; the result is a closed
  solid after welding.

## Responsibilities

| Module | Responsibility |
| --- | --- |
| `common` | Units, UUIDs, bounded JSON, metadata and affine transforms |
| `sdrw` | Layers, 2D primitives, text, dimensions, hatches, blocks, linetypes, measurements, constraints and drawing transactions |
| `sgeo` | Hierarchy, analytic/mesh/NURBS/B-rep geometry, materials and textures, solids, tessellation, typed families and transactions |
| `container` | Deterministic ZIP transport, hashes, binary mesh and texture members, version migration and validated publication |
| `verification` | Local JSON schemas, semantic validation and conversion reports |
| `adapters` | Explicit, bounded exchange subsets: SVG, DXF (`ezdxf`), native GLB and STL; optional DWG/STEP/IFC backends |

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

- DXF (`ezdxf`): planar primitives, text, dimensions, hatches, blocks and linetypes (see
  [Exchange of the new content](#exchange-of-the-new-content)), declared units, layer names and
  visibility. Unspecified/unsupported source units require an explicit `unit` override.
  An override interprets numeric coordinates in that unit; it does not rescale.
  Paper-space content, unsupported entities, bulges, elevations and non-default
  OCS extrusion are refused.
- GLB (native, no third-party backend): triangle meshes, hierarchy, PBR materials, PNG/JPEG textures and
  UVs, converted between SGEO +Z and glTF +Y. Scene export converts positions and local translations
  to metres; the standalone mesh export assumes metre-valued coordinates.
  Float32 conversion that collapses triangles is rejected on export. Preflight checks
  embedded buffer ranges, accessor sizes, references, graph cycles and expanded
  instance budgets before any array is decoded. Animations, skins, sparse accessors, morph targets,
  extensions and external resources are refused.
- SVG: a deliberately narrow UTF-8 primitive subset on import. Native `data-unit` survives
  round-trips. Viewport transforms, root styling, foreign namespaces, nested
  content and document declarations are refused. General path import and rich
  styling remain unsupported. Arcs are written as `path` elements carrying exact
  `data-cx/cy/r/start/end` attributes and are re-imported only when the path
  data agree with those parameters; any other path is refused. Text, hatches, dimensions and
  inserts are written for viewing only.
- STL remains a triangle-only presentation format. DWG, STEP and IFC retain
  their explicit optional-backend contracts; this package provides no B-rep
  modelling kernel and does not promise lossless industrial CAD/BIM round-trips.

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
# Exercise the optional exchange backends as well (ezdxf for DXF, trimesh as an independent GLB reader):
python -m pip install numpy ezdxf trimesh
python -m pytest tests/spatial_formats -W error --cov=src.utils.spatial_formats
```

CI covers Python 3.12/3.13 on Linux and Windows with core-only and exchange-enabled
installations. The duplicate-ZIP-member test explicitly expects its construction
warning and still verifies reader rejection; warnings are not globally hidden.
