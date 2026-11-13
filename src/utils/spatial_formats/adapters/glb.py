"""Native glTF 2.0 binary (GLB) codec: triangle meshes, node hierarchy, PBR materials, textures and UVs.

Pure Python (no trimesh): every byte is validated before it is trusted, and anything the codec cannot represent
faithfully is refused rather than skipped (animations, skins, sparse accessors, morph targets, external
resources, extensions, non-triangle primitives).

Conventions
-----------
* SGEO is right-handed +Z up; glTF is right-handed +Y up. Positions map ``(x, y, z) -> (x, z, -y)`` and node
  matrices are conjugated by the same proper rotation. Lengths are converted to metres on export; an imported file
  has unit ``m``.
* Texture coordinates: SGEO has the origin at the bottom-left of the image with v up, glTF at the top-left
  with v down, so ``v`` is flipped on both export and import.
* Materials map one-to-one to glTF ``pbrMetallicRoughness`` materials; texture slots map to
  ``baseColorTexture``, ``metallicRoughnessTexture``, ``normalTexture``, ``occlusionTexture`` and
  ``emissiveTexture``; wrap and filter modes map to samplers. Images are embedded PNG/JPEG buffer views and are
  verified byte for byte on the way in.
* Object and material identifiers are stored in ``extras.slai.id`` so a SLAI to GLB to SLAI trip keeps them;
  foreign files simply get new ids.
"""
from __future__ import annotations

import json
import struct
from math import isfinite, sqrt

from ..common.errors import ResourceLimitError, UnsupportedFeatureError, ValidationError
from ..common.helpers import MAX_NODES, parse_json
from ..common.ids import is_valid_id, new_id
from ..common.transforms import IDENTITY, compose, require_transform
from ..sgeo.geometry import MAX_TRIANGLES, MAX_VERTICES, validate_geometry
from ..sgeo.images import MAX_IMAGES, texture_path
from ..sgeo.materials import MAX_MATERIALS, TEXTURE_SLOTS, describe_image, validate_material
from ..sgeo.scene import Scene

__all__ = ["write_glb", "read_glb", "MAX_GLB_BYTES", "Y_UP_TO_Z", "Z_UP_TO_Y", "ImportNotes"]

MAX_GLB_BYTES = 128 * 1024 * 1024
MAX_GLB_TABLE = 10_000
# Map glTF (x, y, z) to SGEO (x, -z, y); both matrices have determinant +1.
Y_UP_TO_Z = (1, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 1)
Z_UP_TO_Y = (1, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1)

_COMPONENT = {5120: ("b", 1), 5121: ("B", 1), 5122: ("h", 2), 5123: ("H", 2), 5125: ("I", 4), 5126: ("f", 4)}
_WIDTH = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}
_WRAP_TO_GL = {"repeat": 10497, "clamp": 33071, "mirror": 33648}
_GL_TO_WRAP = {value: key for key, value in _WRAP_TO_GL.items()}
_FILTER_TO_GL = {"linear": (9729, 9987), "nearest": (9728, 9728)}      # (magFilter, minFilter)
_ALPHA_TO_GL = {"opaque": "OPAQUE", "mask": "MASK", "blend": "BLEND"}
_GL_TO_ALPHA = {value: key for key, value in _ALPHA_TO_GL.items()}
_SLOT_TO_GL = {"base_color": ("pbr", "baseColorTexture"), "metallic_roughness": ("pbr", "metallicRoughnessTexture"),
               "normal": ("top", "normalTexture"), "occlusion": ("top", "occlusionTexture"),
               "emissive": ("top", "emissiveTexture")}
_IGNORED_ATTRIBUTES = ("NORMAL", "TANGENT", "COLOR_0")
_TRS_TOLERANCE = 1e-6


class ImportNotes(dict):
    """What an import ignored or repaired (all counts; empty dict means nothing to report)."""


def _fail(message, **context):
    raise ValidationError(message, context=context or None)


def _integer(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        _fail("invalid glTF " + label)
    return value


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        _fail("invalid glTF " + label)
    return float(value)


def _vector(value, count, label):
    if not isinstance(value, list) or len(value) != count:
        _fail("invalid glTF " + label)
    return [_number(v, label) for v in value]


# ------------------------------------------------------------------------------------------ transforms
def _to_column_major(m):
    return [m[r * 4 + c] for c in range(4) for r in range(4)]


def _from_column_major(values):
    return [values[c * 4 + r] for r in range(4) for c in range(4)]


def _conjugate(matrix, *, to_z):
    basis, inverse = (Y_UP_TO_Z, Z_UP_TO_Y) if to_z else (Z_UP_TO_Y, Y_UP_TO_Z)
    return list(compose(compose(basis, matrix), inverse))


def _trs_decomposable(m) -> bool:
    """glTF node matrices must be translation x rotation x scale: no shear, no projection."""
    if abs(m[12]) > 0 or abs(m[13]) > 0 or abs(m[14]) > 0 or abs(m[15] - 1.0) > 1e-12:
        return False
    cols = [(m[0], m[4], m[8]), (m[1], m[5], m[9]), (m[2], m[6], m[10])]
    lengths = [sqrt(sum(c * c for c in col)) for col in cols]
    if min(lengths) <= 1e-12:
        return False
    for i in range(3):
        for j in range(i + 1, 3):
            dot = sum(a * b for a, b in zip(cols[i], cols[j]))
            if abs(dot) > _TRS_TOLERANCE * lengths[i] * lengths[j]:
                return False
    return True


def _trs_to_matrix(node):
    t = node.get("translation")
    r = node.get("rotation")
    s = node.get("scale")
    t = _vector(t, 3, "node translation") if t is not None else [0.0, 0.0, 0.0]
    s = _vector(s, 3, "node scale") if s is not None else [1.0, 1.0, 1.0]
    if r is None:
        x = y = z = 0.0
        w = 1.0
    else:
        x, y, z, w = _vector(r, 4, "node rotation")
        norm = sqrt(x * x + y * y + z * z + w * w)
        if abs(norm - 1.0) > 1e-3:
            _fail("glTF node rotation is not a unit quaternion")
        x, y, z, w = x / norm, y / norm, z / norm, w / norm
    rot = ((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
           (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
           (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)))
    return [rot[0][0] * s[0], rot[0][1] * s[1], rot[0][2] * s[2], t[0],
            rot[1][0] * s[0], rot[1][1] * s[1], rot[1][2] * s[2], t[1],
            rot[2][0] * s[0], rot[2][1] * s[1], rot[2][2] * s[2], t[2], 0.0, 0.0, 0.0, 1.0]


def _node_matrix(node):
    if "matrix" in node:
        if "translation" in node or "rotation" in node or "scale" in node:
            _fail("a glTF node cannot have both matrix and TRS")
        values = _vector(node["matrix"], 16, "node matrix")
        local = _from_column_major(values)
        if not _trs_decomposable(local):
            raise UnsupportedFeatureError("glTF node matrix is not a translation-rotation-scale transform")
        return local
    return _trs_to_matrix(node)


# ------------------------------------------------------------------------------------------- writing
class _Binary:
    def __init__(self):
        self.data = bytearray()
        self.views = []
        self.accessors = []

    def view(self, raw: bytes, target=None) -> int:
        while len(self.data) % 4:
            self.data.append(0)
        view = {"buffer": 0, "byteOffset": len(self.data), "byteLength": len(raw)}
        if target is not None:
            view["target"] = target
        self.data.extend(raw)
        self.views.append(view)
        return len(self.views) - 1

    def accessor(self, raw: bytes, *, component, kind, count, target, bounds=None) -> int:
        entry = {"bufferView": self.view(raw, target), "componentType": component, "type": kind, "count": count}
        if bounds is not None:
            entry["min"], entry["max"] = bounds
        self.accessors.append(entry)
        return len(self.accessors) - 1


def _pack_floats(values, label):
    try:
        return struct.pack("<%df" % len(values), *values)
    except (struct.error, OverflowError) as exc:
        raise ValidationError(label + " is outside the float32 range") from exc


def _export_mesh(binary: _Binary, geometry: dict, scale: float) -> dict:
    vertices = geometry["vertices"]
    flat = []
    for x, y, z in vertices:                                   # Z-up -> Y-up, then to metres
        flat.extend((x * scale, z * scale, -y * scale))
    raw = _pack_floats(flat, "vertex coordinate")
    rounded = struct.unpack("<%df" % len(flat), raw)
    if any(not isfinite(c) for c in rounded):
        _fail("vertex coordinate is outside the float32 range")
    # float32 rounding can collapse a triangle: prove the stored mesh is still valid
    validate_geometry({"kind": "mesh", "faces": geometry["faces"],
                       "vertices": [[rounded[i], -rounded[i + 2], rounded[i + 1]] for i in range(0, len(rounded), 3)]})
    xs, ys, zs = rounded[0::3], rounded[1::3], rounded[2::3]
    attributes = {"POSITION": binary.accessor(raw, component=5126, kind="VEC3", count=len(vertices), target=34962,
                                              bounds=([min(xs), min(ys), min(zs)], [max(xs), max(ys), max(zs)]))}
    if "uvs" in geometry:
        tex = []
        for u, v in geometry["uvs"]:
            tex.extend((u, 1.0 - v))
        attributes["TEXCOORD_0"] = binary.accessor(_pack_floats(tex, "texture coordinate"), component=5126,
                                                   kind="VEC2", count=len(vertices), target=34962)
    indices = [i for face in geometry["faces"] for i in face]
    if len(vertices) <= 65535:
        index_raw, component = struct.pack("<%dH" % len(indices), *indices), 5123
    else:
        index_raw, component = struct.pack("<%dI" % len(indices), *indices), 5125
    primitive = {"attributes": attributes,
                 "indices": binary.accessor(index_raw, component=component, kind="SCALAR", count=len(indices),
                                            target=34963),
                 "mode": 4}
    return primitive


def write_glb(scene: Scene) -> bytes:
    """Serialise a triangle-only scene (see :func:`..sgeo.meshing.flatten_scene`) as GLB bytes.

    Deterministic: equal scenes give equal bytes. Raises :class:`UnsupportedFeatureError` for content GLB cannot
    carry (families, semantic metadata, non-mesh geometry, sheared transforms).
    """
    from ..common.units import convert_length
    scene.validate()
    if scene.families or scene.metadata:
        raise UnsupportedFeatureError("GLB cannot preserve parametric families or scene semantic metadata")
    scale = convert_length(1, scene.unit, "m")
    binary = _Binary()
    images, samplers, textures, materials = [], [], [], []
    image_index = {}
    for record in scene.images:
        raw = scene.texture_bytes(record["id"])
        entry = {"bufferView": binary.view(raw), "mimeType": record["mime"]}
        if record.get("name"):
            entry["name"] = record["name"]
        image_index[record["id"]] = len(images)
        images.append(entry)
    sampler_index, texture_index = {}, {}

    def texture(spec):
        key = (spec.get("wrap_u", "repeat"), spec.get("wrap_v", "repeat"), spec.get("filter", "linear"))
        if key not in sampler_index:
            mag, minimum = _FILTER_TO_GL[key[2]]
            sampler_index[key] = len(samplers)
            samplers.append({"magFilter": mag, "minFilter": minimum, "wrapS": _WRAP_TO_GL[key[0]],
                             "wrapT": _WRAP_TO_GL[key[1]]})
        pair = (image_index[spec["image"]], sampler_index[key])
        if pair not in texture_index:
            texture_index[pair] = len(textures)
            textures.append({"sampler": pair[1], "source": pair[0]})
        return texture_index[pair]

    material_index = {}
    for m in scene.materials:
        pbr = {"baseColorFactor": list(m.get("base_color", [0.8, 0.8, 0.8, 1.0])),
               "metallicFactor": m.get("metallic", 0.0), "roughnessFactor": m.get("roughness", 0.5)}
        entry = {"pbrMetallicRoughness": pbr, "extras": {"slai": {"id": m["id"]}}}
        if m.get("name"):
            entry["name"] = m["name"]
        emissive = list(m.get("emissive", [0.0, 0.0, 0.0]))
        if any(emissive):
            entry["emissiveFactor"] = emissive
        mode = m.get("alpha_mode", "opaque")
        if mode != "opaque":
            entry["alphaMode"] = _ALPHA_TO_GL[mode]
        if mode == "mask":
            entry["alphaCutoff"] = m.get("alpha_cutoff", 0.5)
        if m.get("double_sided"):
            entry["doubleSided"] = True
        for slot, spec in m.get("textures", {}).items():
            where, key = _SLOT_TO_GL[slot]
            reference = {"index": texture(spec)}
            if slot == "normal" and "scale" in spec:
                reference["scale"] = spec["scale"]
            if slot == "occlusion" and "strength" in spec:
                reference["strength"] = spec["strength"]
            (pbr if where == "pbr" else entry)[key] = reference
        material_index[m["id"]] = len(materials)
        materials.append(entry)

    name_counts = {}
    for o in scene.objects:
        if o.get("name"):
            name_counts[o["name"]] = name_counts.get(o["name"], 0) + 1
    reserved = {o["id"] for o in scene.objects}

    def node_name(obj):
        label = obj.get("name", "")
        if label and name_counts[label] == 1 and label not in reserved and len(label) <= 255:
            return label
        return obj["id"]

    nodes, meshes, roots = [], [], []
    node_of = {}
    pending = list(scene.objects)
    while pending:                                             # parents before children, document order kept
        later = []
        for obj in pending:
            parent = obj.get("parent")
            if parent is not None and parent not in node_of:
                later.append(obj)
                continue
            geometry = obj.get("geometry")
            if geometry is not None and geometry["kind"] != "mesh":
                raise UnsupportedFeatureError("GLB scene export requires triangle meshes")
            if obj.get("family_id") or obj.get("metadata") or obj.get("parameters"):
                raise UnsupportedFeatureError("GLB cannot preserve SGEO object semantics or families")
            node = {"name": node_name(obj), "extras": {"slai": {"id": obj["id"]}}}
            local = _conjugate(list(obj.get("transform", IDENTITY)), to_z=False)
            local[3], local[7], local[11] = local[3] * scale, local[7] * scale, local[11] * scale
            if tuple(local) != IDENTITY:
                if not _trs_decomposable(local):
                    raise UnsupportedFeatureError("GLB cannot represent a sheared or projective transform",
                                                  context={"object": obj["id"]})
                node["matrix"] = _to_column_major(local)
            if geometry is not None:
                primitive = _export_mesh(binary, geometry, scale)
                if "material" in obj:
                    primitive["material"] = material_index[obj["material"]]
                node["mesh"] = len(meshes)
                meshes.append({"primitives": [primitive]})
            node_of[obj["id"]] = len(nodes)
            nodes.append(node)
            if parent is None:
                roots.append(node_of[obj["id"]])
            else:
                nodes[node_of[parent]].setdefault("children", []).append(node_of[obj["id"]])
        if len(later) == len(pending):
            _fail("invalid scene hierarchy")
        pending = later
    if len(nodes) > MAX_GLB_TABLE or len(meshes) > MAX_GLB_TABLE:
        raise ResourceLimitError("glTF scene node/mesh count exceeds budget")

    model = {"asset": {"version": "2.0", "generator": "SLAI spatial_formats"}, "scene": 0,
             "scenes": [{"nodes": roots}]}
    if nodes:
        model["nodes"] = nodes
    if meshes:
        model["meshes"] = meshes
    if materials:
        model["materials"] = materials
    if textures:
        model["textures"] = textures
    if samplers:
        model["samplers"] = samplers
    if images:
        model["images"] = images
    if binary.data:
        model["buffers"] = [{"byteLength": len(binary.data)}]
        model["bufferViews"] = binary.views
        if binary.accessors:
            model["accessors"] = binary.accessors
    payload = json.dumps(model, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")
    payload += b" " * (-len(payload) % 4)
    chunks = len(payload).to_bytes(4, "little") + b"JSON" + payload
    if binary.data:
        padded = bytes(binary.data) + b"\0" * (-len(binary.data) % 4)
        chunks += len(padded).to_bytes(4, "little") + b"BIN\0" + padded
    data = b"glTF" + (2).to_bytes(4, "little") + (12 + len(chunks)).to_bytes(4, "little") + chunks
    if len(data) > MAX_GLB_BYTES:
        raise ResourceLimitError("GLB output exceeds the size budget")
    return data


# ------------------------------------------------------------------------------------------- reading
def _split(data: bytes):
    """``(model, binary)`` from the container (structure only)."""
    if not isinstance(data, (bytes, bytearray)) or not 20 <= len(data) <= MAX_GLB_BYTES or data[:4] != b"glTF":
        _fail("invalid or oversized GLB")
    data = bytes(data)
    if int.from_bytes(data[4:8], "little") != 2 or int.from_bytes(data[8:12], "little") != len(data):
        _fail("GLB header/version/length mismatch")
    offset, model, binary = 12, None, None
    while offset < len(data):
        if len(data) - offset < 8:
            _fail("truncated GLB chunk header")
        size = int.from_bytes(data[offset:offset + 4], "little")
        kind = data[offset + 4:offset + 8]
        offset += 8
        if size % 4 or size > len(data) - offset:
            _fail("GLB chunk exceeds file boundaries")
        payload = data[offset:offset + size]
        offset += size
        if model is None:
            if kind != b"JSON":
                _fail("GLB JSON chunk must be first")
            try:
                model = parse_json(payload.rstrip(b" "))
            except (UnicodeError, ValueError) as exc:
                raise ValidationError("invalid GLB JSON") from exc
        elif kind == b"BIN\x00":
            if binary is not None:
                _fail("duplicate GLB binary chunk")
            binary = payload
        else:
            raise UnsupportedFeatureError("unsupported GLB chunk type")
    if not isinstance(model, dict) or not isinstance(model.get("asset"), dict) or model["asset"].get("version") != "2.0":
        _fail("not a glTF 2.0 asset")
    return model, binary or b""


def _table(model, key):
    values = model.get(key, [])
    if not isinstance(values, list) or any(not isinstance(v, dict) for v in values):
        _fail("invalid glTF " + key)
    if len(values) > MAX_GLB_TABLE:
        raise ResourceLimitError("glTF table exceeds budget")
    return values


def _ref(value, values, label):
    index = _integer(value, label)
    if index >= len(values):
        _fail("glTF reference outside " + label)
    return values[index]


def _validate(model, binary):
    """Structural preflight: references, allocation sizes, unsupported features. Returns the tables."""
    if model.get("extensionsRequired") or model.get("extensionsUsed"):
        raise UnsupportedFeatureError("glTF extensions are not supported")
    if model.get("animations") or model.get("skins"):
        raise UnsupportedFeatureError("glTF animations and skins are not supported")
    tables = {key: _table(model, key) for key in ("buffers", "bufferViews", "accessors", "meshes", "nodes", "scenes",
                                                  "materials", "textures", "images", "samplers")}
    buffers, views, accessors = tables["buffers"], tables["bufferViews"], tables["accessors"]
    if len(buffers) > 1:
        raise UnsupportedFeatureError("GLB supports one embedded buffer")
    if buffers:
        if "uri" in buffers[0]:
            raise UnsupportedFeatureError("GLB external or data URI buffers are prohibited")
        size = _integer(buffers[0].get("byteLength"), "buffer size", 1)
        if not size <= len(binary) <= size + 3:
            _fail("GLB binary size disagrees with buffer")
    elif binary:
        _fail("undeclared GLB binary buffer")
    for view in views:
        buffer = _ref(view.get("buffer"), buffers, "buffer")
        offset = _integer(view.get("byteOffset", 0), "view offset")
        length = _integer(view.get("byteLength"), "view length", 1)
        if offset + length > buffer["byteLength"]:
            _fail("GLB buffer view exceeds buffer")
        if "byteStride" in view:
            stride = _integer(view["byteStride"], "stride", 4)
            if stride > 252 or stride % 4:
                _fail("invalid glTF stride")
    expanded = 0
    for accessor in accessors:
        if "sparse" in accessor:
            raise UnsupportedFeatureError("sparse glTF accessors are unsupported")
        view = _ref(accessor.get("bufferView"), views, "buffer view")
        component, kind = accessor.get("componentType"), accessor.get("type")
        if type(component) is not int or component not in _COMPONENT or not isinstance(kind, str) \
                or kind not in _WIDTH:
            raise UnsupportedFeatureError("unsupported glTF accessor element type")
        count = _integer(accessor.get("count"), "accessor count", 1)
        expanded += count * (_WIDTH[kind] + 1)
        if expanded > MAX_NODES:
            raise ResourceLimitError("glTF decoded array budget exceeded")
        offset = _integer(accessor.get("byteOffset", 0), "accessor offset")
        size = _COMPONENT[component][1]
        width = size * _WIDTH[kind]
        stride = view.get("byteStride", width)
        if stride < width or offset % size or (view.get("byteOffset", 0) + offset) % size:
            _fail("misaligned glTF accessor")
        if offset + (count - 1) * stride + width > view["byteLength"]:
            _fail("glTF accessor exceeds buffer view")
    for mesh in tables["meshes"]:
        primitives = mesh.get("primitives")
        if not isinstance(primitives, list) or not 1 <= len(primitives) <= 256:
            _fail("a glTF mesh needs 1..256 primitives")
        for primitive in primitives:
            if not isinstance(primitive, dict) or not isinstance(primitive.get("attributes"), dict):
                _fail("invalid glTF primitive")
            if primitive.get("targets") or primitive.get("extensions"):
                raise UnsupportedFeatureError("glTF morph targets and primitive extensions are unsupported")
            if primitive.get("mode", 4) != 4:
                raise UnsupportedFeatureError("only glTF TRIANGLES primitive mode is supported")
            attributes = primitive["attributes"]
            if "POSITION" not in attributes:
                _fail("a glTF primitive needs POSITION")
            for name in attributes:
                if name.startswith(("JOINTS_", "WEIGHTS_")):
                    raise UnsupportedFeatureError("glTF skinning attributes are unsupported")
                if name.startswith("TEXCOORD_") and name != "TEXCOORD_0":
                    raise UnsupportedFeatureError("only the first glTF texture coordinate set is supported")
            position = _ref(attributes["POSITION"], accessors, "position accessor")
            if position["type"] != "VEC3" or position["componentType"] != 5126:
                _fail("positions must be float32 VEC3")
            for name, index in attributes.items():
                if _ref(index, accessors, "attribute accessor")["count"] != position["count"]:
                    _fail("glTF attribute counts disagree")
            if "TEXCOORD_0" in attributes:
                uv = accessors[attributes["TEXCOORD_0"]]
                if uv["type"] != "VEC2" or uv["componentType"] not in (5121, 5123, 5126):
                    _fail("texture coordinates must be VEC2 float or normalized unsigned integers")
                if uv["componentType"] != 5126 and uv.get("normalized") is not True:
                    _fail("integer texture coordinates must be normalized")
            count = position["count"]
            if "indices" in primitive:
                indices = _ref(primitive["indices"], accessors, "index accessor")
                if indices["type"] != "SCALAR" or indices["componentType"] not in (5121, 5123, 5125):
                    _fail("invalid glTF triangle index type")
                count = indices["count"]
            if count % 3:
                _fail("incomplete glTF triangle")
            if "material" in primitive:
                _ref(primitive["material"], tables["materials"], "material")
    nodes = tables["nodes"]
    parents, budget = {}, 0
    for index, node in enumerate(nodes):
        if "mesh" in node:
            mesh = _ref(node["mesh"], tables["meshes"], "mesh")
            for primitive in mesh["primitives"]:
                position = accessors[primitive["attributes"]["POSITION"]]
                indices = accessors[primitive["indices"]] if "indices" in primitive else position
                budget += position["count"] * 4 + indices["count"] * 2
            if budget > MAX_NODES:
                raise ResourceLimitError("glTF instantiated geometry exceeds budget")
        children = node.get("children", [])
        if not isinstance(children, list):
            _fail("invalid glTF children")
        for child in children:
            _ref(child, nodes, "child node")
            if child in parents:
                _fail("glTF node has multiple parent references")
            parents[child] = index
    visited = set()
    for node in range(len(nodes)):
        chain, cursor = set(), node
        while cursor is not None and cursor not in visited:
            if cursor in chain:
                _fail("cyclic glTF hierarchy")
            chain.add(cursor)
            cursor = parents.get(cursor)
        visited.update(chain)
    for scene in tables["scenes"]:
        listed = scene.get("nodes", [])
        if not isinstance(listed, list):
            _fail("invalid glTF scene roots")
        for index in listed:
            _ref(index, nodes, "scene root")
            if index in parents:
                _fail("glTF scene root has a parent")
    if "scene" in model:
        _ref(model["scene"], tables["scenes"], "default scene")
    for image in tables["images"]:
        if "uri" in image:
            raise UnsupportedFeatureError("GLB external image references are prohibited")
        if "bufferView" not in image or not isinstance(image.get("mimeType"), str):
            _fail("an embedded glTF image needs bufferView and mimeType")
        _ref(image["bufferView"], views, "image buffer view")
    if len(tables["images"]) > MAX_IMAGES or len(tables["materials"]) > MAX_MATERIALS:
        raise ResourceLimitError("glTF image or material count exceeds budget")
    for texture in tables["textures"]:
        _ref(texture.get("source"), tables["images"], "texture source")
        if "sampler" in texture:
            _ref(texture["sampler"], tables["samplers"], "texture sampler")
    return tables


def _read_accessor(model, binary, index):
    accessor = model["accessors"][index]
    view = model["bufferViews"][accessor["bufferView"]]
    code, size = _COMPONENT[accessor["componentType"]]
    n = _WIDTH[accessor["type"]]
    count = accessor["count"]
    start = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    stride = view.get("byteStride", size * n)
    element = struct.Struct("<" + code * n)
    if stride == size * n:
        rows = list(element.iter_unpack(binary[start:start + count * stride]))
    else:
        rows = [element.unpack_from(binary, start + i * stride) for i in range(count)]
    if len(rows) != count:
        _fail("glTF accessor data is truncated")
    return rows


def _scalar_index(rows):
    return [row[0] for row in rows]


def _material(model, binary, spec, image_ids, notes):
    """A SLAI material record from one glTF material (explicit values; glTF defaults applied)."""
    pbr = spec.get("pbrMetallicRoughness", {})
    if not isinstance(pbr, dict):
        _fail("invalid glTF pbrMetallicRoughness")
    if spec.get("extensions") or pbr.get("extensions"):
        raise UnsupportedFeatureError("glTF material extensions are unsupported")
    record = {"id": None, "name": spec.get("name", "")[:255] if isinstance(spec.get("name", ""), str) else "",
              "base_color": _vector(pbr.get("baseColorFactor", [1, 1, 1, 1]), 4, "baseColorFactor"),
              "metallic": _number(pbr.get("metallicFactor", 1.0), "metallicFactor"),
              "roughness": _number(pbr.get("roughnessFactor", 1.0), "roughnessFactor"),
              "emissive": _vector(spec.get("emissiveFactor", [0, 0, 0]), 3, "emissiveFactor")}
    mode = spec.get("alphaMode", "OPAQUE")
    if not isinstance(mode, str) or mode not in _GL_TO_ALPHA:
        _fail("invalid glTF alphaMode")
    record["alpha_mode"] = _GL_TO_ALPHA[mode]
    if mode == "MASK":
        record["alpha_cutoff"] = _number(spec.get("alphaCutoff", 0.5), "alphaCutoff")
    double_sided = spec.get("doubleSided", False)
    if not isinstance(double_sided, bool):
        _fail("invalid glTF doubleSided")
    record["double_sided"] = double_sided
    textures = {}
    for slot in TEXTURE_SLOTS:
        where, key = _SLOT_TO_GL[slot]
        reference = (pbr if where == "pbr" else spec).get(key)
        if reference is None:
            continue
        if not isinstance(reference, dict) or reference.get("extensions"):
            raise UnsupportedFeatureError("glTF texture extensions (e.g. KHR_texture_transform) are unsupported")
        if reference.get("texCoord", 0) != 0:
            raise UnsupportedFeatureError("only texture coordinate set 0 is supported")
        texture = _ref(reference.get("index"), model.get("textures", []), "texture")
        sampler = model["samplers"][texture["sampler"]] if "sampler" in texture else {}
        wrap_u = _GL_TO_WRAP.get(sampler.get("wrapS", 10497))
        wrap_v = _GL_TO_WRAP.get(sampler.get("wrapT", 10497))
        if wrap_u is None or wrap_v is None:
            _fail("unsupported glTF sampler wrap mode")
        entry = {"image": image_ids[texture["source"]], "wrap_u": wrap_u, "wrap_v": wrap_v,
                 "filter": "nearest" if sampler.get("magFilter") == 9728 else "linear"}
        if slot == "normal" and "scale" in reference:
            entry["scale"] = _number(reference["scale"], "normalTexture.scale")
        if slot == "occlusion" and "strength" in reference:
            entry["strength"] = _number(reference["strength"], "occlusionTexture.strength")
        textures[slot] = entry
    record["textures"] = textures
    extras = spec.get("extras")
    if isinstance(extras, dict) and isinstance(extras.get("slai"), dict):
        candidate = extras["slai"].get("id")
        if isinstance(candidate, str) and is_valid_id(candidate):
            record["id"] = candidate
    return record


def read_glb(data: bytes):
    """Parse GLB bytes into ``(Scene, ImportNotes)``; the scene has unit ``m`` and +Z up."""
    model, binary = _split(data)
    tables = _validate(model, binary)
    notes = ImportNotes()

    def note(key, amount=1):
        notes[key] = notes.get(key, 0) + amount

    # images: verified bytes, deduplicated by content
    images, image_data, image_ids, by_digest = [], {}, {}, {}
    for index, image in enumerate(tables["images"]):
        view = model["bufferViews"][image["bufferView"]]
        start = view.get("byteOffset", 0)
        raw = bytes(binary[start:start + view["byteLength"]])
        if image["mimeType"] not in ("image/png", "image/jpeg"):
            raise UnsupportedFeatureError("only embedded PNG and JPEG images are supported")
        facts = describe_image(raw)
        if facts["mime"] != image["mimeType"]:
            _fail("a glTF image's bytes do not match its declared mimeType")
        key = (facts["sha256"], facts["mime"])
        if key not in by_digest:
            ident = new_id()
            name = image.get("name", "")
            images.append({"id": ident, "name": name[:255] if isinstance(name, str) else "", **facts})
            image_data[ident] = raw
            by_digest[key] = ident
        image_ids[index] = by_digest[key]

    materials, material_ids, used_ids = [], [], set()
    for spec in tables["materials"]:
        record = _material(model, binary, spec, image_ids, notes)
        if record["id"] is None or record["id"] in used_ids:
            record["id"] = new_id()
        used_ids.add(record["id"])
        validate_material(record, image_ids=set(image_ids.values()))
        materials.append(record)
        material_ids.append(record["id"])

    scene = Scene(name="Imported glTF", unit="m", materials=materials, images=images, image_data=image_data)

    mesh_cache = {}

    def primitive_geometry(mesh_index, primitive_index):
        key = (mesh_index, primitive_index)
        if key not in mesh_cache:
            primitive = tables["meshes"][mesh_index]["primitives"][primitive_index]
            attributes = primitive["attributes"]
            raw_positions = _read_accessor(model, binary, attributes["POSITION"])
            vertices = []
            for x, y, z in raw_positions:
                if not (isfinite(x) and isfinite(y) and isfinite(z)):
                    _fail("nonfinite mesh vertices")
                vertices.append([x, -z, y])
            uvs = None
            if "TEXCOORD_0" in attributes:
                accessor = model["accessors"][attributes["TEXCOORD_0"]]
                divisor = {5121: 255.0, 5123: 65535.0, 5126: 1.0}[accessor["componentType"]]
                uvs = []
                for u, v in _read_accessor(model, binary, attributes["TEXCOORD_0"]):
                    if not (isfinite(u) and isfinite(v)):
                        _fail("nonfinite texture coordinates")
                    uvs.append([u / divisor, 1.0 - v / divisor])
            for name in _IGNORED_ATTRIBUTES:
                if name in attributes:
                    note("ignored_" + name.lower() + "_attributes")
            if "indices" in primitive:
                flat = _scalar_index(_read_accessor(model, binary, primitive["indices"]))
            else:
                flat = list(range(len(vertices)))
            faces, dropped = [], 0
            for i in range(0, len(flat), 3):
                a, b, c = flat[i], flat[i + 1], flat[i + 2]
                if max(a, b, c) >= len(vertices):
                    _fail("glTF index refers to a missing vertex")
                if a == b or b == c or a == c:
                    dropped += 1
                    continue
                p, q, r = vertices[a], vertices[b], vertices[c]
                ux, uy, uz = q[0] - p[0], q[1] - p[1], q[2] - p[2]
                vx, vy, vz = r[0] - p[0], r[1] - p[1], r[2] - p[2]
                cx, cy, cz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
                if cx * cx + cy * cy + cz * cz <= 1e-30:
                    dropped += 1
                    continue
                faces.append([a, b, c])
            if len(faces) > MAX_TRIANGLES or len(vertices) > MAX_VERTICES:
                raise ResourceLimitError("GLB mesh exceeds the vertex/triangle budget")
            geometry = None
            if faces:
                geometry = {"kind": "mesh", "vertices": vertices, "faces": faces}
                if uvs is not None:
                    geometry["uvs"] = uvs
                validate_geometry(geometry)
            mesh_cache[key] = (geometry, dropped)
        geometry, dropped = mesh_cache[key]
        if dropped:
            note("degenerate_triangles_removed", dropped)
        if geometry is None:
            note("empty_primitives")
            return None
        return {k: ([list(x) for x in v] if isinstance(v, list) else v) for k, v in geometry.items()}

    nodes = tables["nodes"]
    parent_of = {child: index for index, node in enumerate(nodes) for child in node.get("children", [])}
    if tables["scenes"]:
        chosen = tables["scenes"][model.get("scene", 0)] if model.get("scene", 0) < len(tables["scenes"]) \
            else tables["scenes"][0]
        roots = list(chosen.get("nodes", []))
    else:
        roots = [i for i in range(len(nodes)) if i not in parent_of]
    reachable, order, queue = set(), [], list(roots)
    while queue:
        index = queue.pop(0)
        if index in reachable:
            continue
        reachable.add(index)
        order.append(index)
        queue.extend(nodes[index].get("children", []))
    if len(reachable) < len(nodes):
        note("unreachable_nodes", len(nodes) - len(reachable))

    created, used_object_ids = {}, set()

    def object_id(spec):
        extras = spec.get("extras")
        if isinstance(extras, dict) and isinstance(extras.get("slai"), dict):
            candidate = extras["slai"].get("id")
            if isinstance(candidate, str) and is_valid_id(candidate) and candidate not in used_object_ids:
                return candidate
        return None

    for index in order:
        node = nodes[index]
        if "camera" in node:
            note("cameras_dropped")
        local = _conjugate(_node_matrix(node), to_z=True)
        name = node.get("name") if isinstance(node.get("name"), str) else ""
        parent = created.get(parent_of.get(index))
        wanted = object_id(node)
        geometry_name = name or "node%d" % index
        if "mesh" in node:
            mesh = tables["meshes"][node["mesh"]]
            count = len(mesh["primitives"])
            if count == 1:
                primitive = mesh["primitives"][0]
                geometry = primitive_geometry(node["mesh"], 0)
                material = material_ids[primitive["material"]] if "material" in primitive and geometry is not None \
                    else None
                oid = scene.add_object(name=geometry_name[:255], geometry=geometry, parent=parent,
                                       transform=require_transform(local), object_id=wanted, material=material)
                created[index] = oid
                used_object_ids.add(oid)
                continue
            oid = scene.add_object(name=geometry_name[:255], parent=parent, transform=require_transform(local),
                                   object_id=wanted)
            created[index] = oid
            used_object_ids.add(oid)
            note("multi_primitive_meshes_split")
            for number, primitive in enumerate(mesh["primitives"]):
                geometry = primitive_geometry(node["mesh"], number)
                if geometry is None:
                    continue
                material = material_ids[primitive["material"]] if "material" in primitive else None
                child = scene.add_object(name=("%s#%d" % (geometry_name, number))[:255], geometry=geometry,
                                         parent=oid, material=material)
                used_object_ids.add(child)
            continue
        oid = scene.add_object(name=geometry_name[:255] if name else "", parent=parent,
                               transform=require_transform(local), object_id=wanted)
        created[index] = oid
        used_object_ids.add(oid)
    scene.validate()
    return scene, notes
