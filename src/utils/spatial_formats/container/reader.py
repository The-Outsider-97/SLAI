"""Defensive bounded decoder for SLAI SGEO/SDRW ZIP artifacts."""
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile, ZIP_STORED, BadZipFile
from ..common.errors import ValidationError, IntegrityError, ResourceLimitError, UnsupportedFeatureError
from ..common.helpers import parse_json, digest
from ..common.ids import require_id
from ..common.units import require_unit

MIMES = {"slai.sgeo": b"application/x-slai-sgeo+zip", "slai.sdrw": b"application/x-slai-sdrw+zip"}
PAYLOADS = {"slai.sgeo": "scene.json", "slai.sdrw": "drawing.json"}
MAX_ARCHIVE_SIZE = 256 * 1024 * 1024
MAX_UNPACKED = 128 * 1024 * 1024
MAX_MEMBERS = 128
MAX_RATIO = 100

@dataclass(frozen=True)
class SpatialPackage:
    manifest: dict
    document: dict
    assets: dict[str, bytes]

def _validate_name(name: str) -> None:
    if (not name or name.startswith("/") or "\\" in name or "\x00" in name
            or any(part in ("", ".", "..") for part in name.split("/"))
            or len(name) > 512 or name.endswith("/")):
        raise ValidationError("unsafe archive member name")

def _manifest(manifest):
    if not isinstance(manifest, dict):
        raise ValidationError("manifest must be an object")
    format_id = manifest.get("format_id")
    if format_id not in MIMES:
        raise UnsupportedFeatureError("unrecognized spatial format")
    if manifest.get("format_version") != "0.1.0":
        raise UnsupportedFeatureError("unsupported spatial format version")
    if manifest.get("schema") != format_id + "/0.1":
        raise UnsupportedFeatureError("unsupported schema identifier")
    dim = 3 if format_id == "slai.sgeo" else 2
    if manifest.get("spatial_dimension") != dim:
        raise ValidationError("incorrect spatial dimension")
    kinds = ("model", "family") if dim == 3 else ("drawing",)
    if manifest.get("document_kind") not in kinds:
        raise ValidationError("invalid document kind")
    require_id(manifest.get("document_id"))
    if type(manifest.get("revision")) is not int or manifest["revision"] < 1:
        raise ValidationError("revision must be a positive integer")
    require_unit(manifest.get("units"))
    if dim == 3 and (manifest.get("axis_up") != "+Z" or manifest.get("handedness") != "right"):
        raise ValidationError("SGEO requires right-handed +Z up")
    requires = manifest.get("requires", [])
    if not isinstance(requires, list) or any(not isinstance(f, str) for f in requires):
        raise ValidationError("requires must be a string list")
    allowed = {"scene-graph", "triangle-mesh", "analytic-primitives", "drawing-primitives", "families"}
    if set(requires) - allowed:
        raise UnsupportedFeatureError("unsupported required feature")
    parts = manifest.get("parts")
    if not isinstance(parts, list) or not parts or len(parts) > MAX_MEMBERS - 2:
        raise ValidationError("invalid part index")
    seen = set()
    for part in parts:
        if not isinstance(part, dict):
            raise ValidationError("part must be an object")
        name = part.get("path")
        if not isinstance(name, str):
            raise ValidationError("part path must be a string")
        _validate_name(name)
        if name in seen or name in ("mimetype", "manifest.json"):
            raise ValidationError("duplicate or reserved part")
        seen.add(name)
        if type(part.get("size")) is not int or not 0 <= part["size"] <= MAX_UNPACKED:
            raise ResourceLimitError("invalid member size")
        hexdigest = part.get("sha256")
        if not isinstance(hexdigest, str) or len(hexdigest) != 64 or any(x not in "0123456789abcdef" for x in hexdigest):
            raise ValidationError("invalid SHA-256")
    if PAYLOADS[format_id] not in seen:
        raise ValidationError("required document payload missing")
    return format_id

def load_package(path, *, max_archive_size=MAX_ARCHIVE_SIZE, max_unpacked=MAX_UNPACKED) -> SpatialPackage:
    path = Path(path)
    if path.stat().st_size > min(max_archive_size, MAX_ARCHIVE_SIZE):
        raise ResourceLimitError("compressed spatial package too large")
    try:
        with ZipFile(path, "r", allowZip64=False) as archive:
            infos = archive.infolist()
            if len(infos) < 3 or len(infos) > MAX_MEMBERS:
                raise ResourceLimitError("invalid member count")
            names = [info.filename for info in infos]
            if len(set(names)) != len(names):
                raise ValidationError("duplicate archive members")
            for info in infos:
                _validate_name(info.filename)
                if info.flag_bits & 1:
                    raise ValidationError("encrypted ZIP entries are unsupported")
                if info.file_size > MAX_UNPACKED or (info.file_size and info.file_size > MAX_RATIO * max(1, info.compress_size)):
                    raise ResourceLimitError("ZIP member exceeds resource budget")
                if (info.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValidationError("ZIP symlink members are prohibited")
            if sum(info.file_size for info in infos) > min(max_unpacked, MAX_UNPACKED):
                raise ResourceLimitError("ZIP expanded data exceeds resource budget")
            if names[:2] != ["mimetype", "manifest.json"] or infos[0].compress_type != ZIP_STORED:
                raise ValidationError("spatial file must begin with stored mimetype and manifest")
            mimetype = archive.read("mimetype")
            if mimetype not in MIMES.values():
                raise UnsupportedFeatureError("invalid mimetype marker")
            manifest = parse_json(archive.read("manifest.json"))
            format_id = _manifest(manifest)
            if mimetype != MIMES[format_id]:
                raise IntegrityError("manifest and MIME identity disagree")
            parts = {part["path"]: part for part in manifest["parts"]}
            if set(names) != set(parts) | {"mimetype", "manifest.json"}:
                raise IntegrityError("archive contents differ from manifest index")
            data = {}
            for name, part in parts.items():
                info = archive.getinfo(name)
                if info.file_size != part["size"]:
                    raise IntegrityError("member size mismatch: " + name)
                raw = archive.read(name)
                if len(raw) != part["size"] or digest(raw) != part["sha256"]:
                    raise IntegrityError("member digest mismatch: " + name)
                data[name] = raw
            payload = PAYLOADS[format_id]
            document = parse_json(data.pop(payload))
            if not isinstance(document, dict):
                raise ValidationError("document must be an object")
            from ..verification.invariants import validate_document
            validate_document(manifest, document)
            return SpatialPackage(manifest, document, data)
    except (BadZipFile, EOFError, RuntimeError) as exc:
        raise IntegrityError("invalid or corrupted spatial ZIP package") from exc
