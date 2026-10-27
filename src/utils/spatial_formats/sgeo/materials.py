"""SGEO 0.2 physically based materials and the image registry they sample.

Model (mirrors glTF 2.0 metallic-roughness so interchange is lossless)::

    scene.materials = [{
        "id": uuid, "name": str,
        "base_color": [r, g, b, a],        # linear RGBA, 0..1
        "metallic": 0..1, "roughness": 0..1,
        "emissive": [r, g, b],             # linear, 0..1
        "alpha_mode": "opaque" | "mask" | "blend", "alpha_cutoff": 0..1,   # cutoff only with "mask"
        "double_sided": bool,
        "textures": {slot: {"image": uuid, "wrap_u", "wrap_v", "filter", ("scale" | "strength")}},
        "metadata": {...}                  # optional, namespaced like object metadata
    }]
    scene.images = [{"id", "name", "mime", "width", "height", "size", "sha256"}]   # bytes live beside the document

Texture slots: ``base_color`` (sRGB colour, multiplied by ``base_color``), ``metallic_roughness``
(linear data: green = roughness, blue = metallic), ``normal`` (tangent-space, ``scale``), ``occlusion``
(linear, red channel, ``strength``) and ``emissive`` (sRGB, multiplied by ``emissive``).
Texture coordinates are ``(u, v)`` with the origin at the *bottom-left* of the image and ``v`` pointing up;
glTF's top-left origin is converted on export and import.

Every field is optional in a stored record and has the documented default; the editing API always writes
explicit values. A material is attached to an object with ``object["material"]`` (no inheritance).
"""
from __future__ import annotations

from hashlib import sha256 as _sha256

from ..common.errors import ReferenceError, ResourceLimitError, ValidationError
from ..common.helpers import finite
from ..common.ids import require_id
from ..common.metadata import require_metadata
from .images import (MAX_IMAGES, MAX_IMAGE_BYTES, MAX_IMAGE_PIXELS, MAX_IMAGE_SIDE, MAX_TOTAL_IMAGE_BYTES,
                     MIME_EXTENSIONS, inspect_image)

__all__ = ["MAX_MATERIALS", "MAX_IMAGES", "TEXTURE_SLOTS", "ALPHA_MODES", "WRAP_MODES", "FILTER_MODES",
           "DEFAULT_MATERIAL", "resolve_material", "validate_material", "validate_image_record",
           "validate_registries", "verify_image_bytes", "describe_image"]

MAX_MATERIALS = 10_000
TEXTURE_SLOTS = ("base_color", "metallic_roughness", "normal", "occlusion", "emissive")
ALPHA_MODES = ("opaque", "mask", "blend")
WRAP_MODES = ("repeat", "clamp", "mirror")
FILTER_MODES = ("linear", "nearest")

DEFAULT_MATERIAL = {
    "base_color": [0.8, 0.8, 0.8, 1.0], "metallic": 0.0, "roughness": 0.5, "emissive": [0.0, 0.0, 0.0],
    "alpha_mode": "opaque", "alpha_cutoff": 0.5, "double_sided": False, "textures": {},
}
_MATERIAL_KEYS = frozenset({"id", "name", "metadata"} | set(DEFAULT_MATERIAL))
_SLOT_KEYS = {
    "base_color": {"image", "wrap_u", "wrap_v", "filter"},
    "metallic_roughness": {"image", "wrap_u", "wrap_v", "filter"},
    "emissive": {"image", "wrap_u", "wrap_v", "filter"},
    "normal": {"image", "wrap_u", "wrap_v", "filter", "scale"},
    "occlusion": {"image", "wrap_u", "wrap_v", "filter", "strength"},
}
_IMAGE_KEYS = frozenset({"id", "name", "mime", "width", "height", "size", "sha256"})
_HEX = frozenset("0123456789abcdef")


def _unit_interval(value, label):
    number = finite(value, label, limit=1e6)
    if not 0.0 <= number <= 1.0:
        raise ValidationError(label + " must lie in [0, 1]")
    return number


def _color(value, count, label):
    if not isinstance(value, (list, tuple)) or len(value) != count:
        raise ValidationError("%s must have %d components" % (label, count))
    return [_unit_interval(v, label) for v in value]


def resolve_material(material: dict) -> dict:
    """The material with every default applied (a detached copy; textures keep their slot settings)."""
    result = {k: (list(v) if isinstance(v, list) else dict(v) if isinstance(v, dict) else v)
              for k, v in DEFAULT_MATERIAL.items()}
    for key in DEFAULT_MATERIAL:
        if key in material:
            value = material[key]
            result[key] = [float(c) for c in value] if key in ("base_color", "emissive") else value
    result["id"] = material["id"]
    result["name"] = material.get("name", "")
    if "metadata" in material:
        result["metadata"] = dict(material["metadata"])
    return result


def validate_material(material, *, image_ids=None) -> dict:
    """Check one material record; ``image_ids`` (a container of ids) enables texture reference checks."""
    if not isinstance(material, dict):
        raise ValidationError("material must be an object")
    unknown = set(material) - _MATERIAL_KEYS
    if unknown:
        raise ValidationError("unknown material field(s): " + ", ".join(sorted(map(str, unknown))))
    require_id(material.get("id"))
    name = material.get("name", "")
    if not isinstance(name, str) or len(name) > 255:
        raise ValidationError("invalid material name")
    if "base_color" in material:
        _color(material["base_color"], 4, "base_color")
    if "emissive" in material:
        _color(material["emissive"], 3, "emissive")
    for key in ("metallic", "roughness", "alpha_cutoff"):
        if key in material:
            _unit_interval(material[key], key)
    mode = material.get("alpha_mode", "opaque")
    if not isinstance(mode, str) or mode not in ALPHA_MODES:
        raise ValidationError("alpha_mode must be one of: " + ", ".join(ALPHA_MODES))
    if "alpha_cutoff" in material and mode != "mask":
        raise ValidationError("alpha_cutoff only applies when alpha_mode is 'mask'")
    if "double_sided" in material and not isinstance(material["double_sided"], bool):
        raise ValidationError("double_sided must be a boolean")
    if "metadata" in material:
        require_metadata(material["metadata"])
    textures = material.get("textures", {})
    if not isinstance(textures, dict):
        raise ValidationError("textures must be an object")
    for slot, spec in textures.items():
        if not isinstance(slot, str) or slot not in _SLOT_KEYS:
            raise ValidationError("unknown texture slot %r (use one of: %s)" % (slot, ", ".join(TEXTURE_SLOTS)))
        if not isinstance(spec, dict):
            raise ValidationError("texture slot %s must be an object" % slot)
        extra = set(spec) - _SLOT_KEYS[slot]
        if extra:
            raise ValidationError("unknown field(s) in texture slot %s: %s" % (slot, ", ".join(sorted(map(str, extra)))))
        image = require_id(spec.get("image"))
        if image_ids is not None and image not in image_ids:
            raise ReferenceError("texture slot %s references an unknown image" % slot, context={"image": image})
        for key, allowed in (("wrap_u", WRAP_MODES), ("wrap_v", WRAP_MODES), ("filter", FILTER_MODES)):
            if key in spec and (not isinstance(spec[key], str) or spec[key] not in allowed):
                raise ValidationError("%s of texture slot %s must be one of: %s" % (key, slot, ", ".join(allowed)))
        if "scale" in spec and finite(spec["scale"], "normal scale", limit=1e3) < 0:
            raise ValidationError("normal scale must not be negative")
        if "strength" in spec:
            _unit_interval(spec["strength"], "occlusion strength")
    return material


def validate_image_record(record) -> dict:
    """Check one image registry record (not its bytes; see :func:`verify_image_bytes`)."""
    if not isinstance(record, dict):
        raise ValidationError("image must be an object")
    if set(record) != _IMAGE_KEYS:
        raise ValidationError("an image record needs exactly: " + ", ".join(sorted(_IMAGE_KEYS)))
    require_id(record["id"])
    if not isinstance(record["name"], str) or len(record["name"]) > 255:
        raise ValidationError("invalid image name")
    if not isinstance(record["mime"], str) or record["mime"] not in MIME_EXTENSIONS:
        raise ValidationError("image mime must be image/png or image/jpeg")
    for key in ("width", "height", "size"):
        if type(record[key]) is not int or record[key] < 1:
            raise ValidationError("image %s must be a positive integer" % key)
    if record["width"] > MAX_IMAGE_SIDE or record["height"] > MAX_IMAGE_SIDE or \
            record["width"] * record["height"] > MAX_IMAGE_PIXELS:
        raise ResourceLimitError("image dimensions exceed the limit")
    if record["size"] > MAX_IMAGE_BYTES:
        raise ResourceLimitError("image exceeds the %d byte limit" % MAX_IMAGE_BYTES)
    digest = record["sha256"]
    if not isinstance(digest, str) or len(digest) != 64 or not set(digest) <= _HEX:
        raise ValidationError("invalid image sha256")
    return record


def validate_registries(materials, images, *, image_data=None) -> None:
    """Validate both registries and every cross reference between them.

    ``image_data`` (id -> bytes) is optional: given, it must hold exactly the registered images with the
    declared sizes (hashes are checked when bytes are added or loaded, see :func:`verify_image_bytes`).
    """
    if not isinstance(materials, list) or len(materials) > MAX_MATERIALS:
        raise ValidationError("materials must be a list of at most %d" % MAX_MATERIALS)
    if not isinstance(images, list) or len(images) > MAX_IMAGES:
        raise ValidationError("images must be a list of at most %d" % MAX_IMAGES)
    image_index = {}
    total = 0
    for record in images:
        validate_image_record(record)
        if record["id"] in image_index:
            raise ValidationError("duplicate image identifier", context={"id": record["id"]})
        image_index[record["id"]] = record
        total += record["size"]
    if total > MAX_TOTAL_IMAGE_BYTES:
        raise ResourceLimitError("embedded images exceed the %d byte budget" % MAX_TOTAL_IMAGE_BYTES)
    seen = set()
    for material in materials:
        validate_material(material, image_ids=image_index)
        if material["id"] in seen:
            raise ValidationError("duplicate material identifier", context={"id": material["id"]})
        seen.add(material["id"])
    if image_data is not None:
        if not isinstance(image_data, dict) or set(image_data) != set(image_index):
            raise ValidationError("embedded image bytes do not match the image registry")
        for ident, record in image_index.items():
            raw = image_data[ident]
            if not isinstance(raw, bytes) or len(raw) != record["size"]:
                raise ValidationError("embedded image size does not match its record", context={"id": ident})


def describe_image(data) -> dict:
    """Header facts and digest of encoded image bytes (raises if they are not an allowed image)."""
    info = inspect_image(data)
    raw = bytes(data)
    return {"mime": info.mime, "width": info.width, "height": info.height, "size": len(raw),
            "sha256": _sha256(raw).hexdigest()}


def verify_image_bytes(record: dict, data) -> None:
    """Prove ``data`` is the image ``record`` describes: same digest, size, type and dimensions."""
    facts = describe_image(data)
    for key in ("size", "sha256", "mime", "width", "height"):
        if facts[key] != record[key]:
            raise ValidationError("embedded image does not match its record (%s differs)" % key,
                                  context={"id": record["id"]})
