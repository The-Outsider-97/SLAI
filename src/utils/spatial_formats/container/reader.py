"""Defensive bounded decoder for SLAI SGEO/SDRW ZIP artifacts."""
from dataclasses import dataclass
from io import BytesIO
import os
from pathlib import Path
from zipfile import ZipFile, ZIP_STORED, ZIP_DEFLATED, BadZipFile
from ..common.errors import ValidationError, IntegrityError, ResourceLimitError, SecurityLimitError, UnsupportedFeatureError
from ..common.helpers import parse_json, digest, MAX_DOCUMENT_NODES
from ..common.ids import require_id
from ..common.units import require_unit
from .versions import CURRENT_VERSION, FEATURES, SUPPORTED_VERSIONS, schema_id

MIMES = {"slai.sgeo": b"application/x-slai-sgeo+zip", "slai.sdrw": b"application/x-slai-sdrw+zip"}
PAYLOADS = {"slai.sgeo": "scene.json", "slai.sdrw": "drawing.json"}
MAX_ARCHIVE_SIZE = 256 * 1024 * 1024
MAX_UNPACKED = 128 * 1024 * 1024
MAX_MEMBERS = 1536   # up to 1000 embedded textures plus geometry buffers
MAX_RATIO = 1024  # Absolute decompressed budgets remain authoritative.

@dataclass(frozen=True)
class SpatialPackage:
    """A fully validated package. ``manifest``/``document`` are always in the
    *current* format; ``source_version`` is the version the file was written
    with (it differs only when a legacy file was upgraded in memory)."""
    manifest: dict
    document: dict
    assets: dict[str, bytes]
    source_version: str | None = None

def _validate_name(name: str) -> None:
    if (not name or name.startswith("/") or "\\" in name or "\x00" in name
            or any(part in ("", ".", "..") for part in name.split("/"))
            or len(name) > 512 or name.endswith("/")):
        raise ValidationError("unsafe archive member name")

def _manifest(manifest):
    if not isinstance(manifest, dict):
        raise ValidationError("manifest must be an object")
    format_id = manifest.get("format_id")
    if not isinstance(format_id, str) or format_id not in MIMES:
        raise UnsupportedFeatureError("unrecognized spatial format")
    version = manifest.get("format_version")
    if not isinstance(version, str) or version not in SUPPORTED_VERSIONS:
        raise UnsupportedFeatureError("unsupported spatial format version",
                                      context={"version": version if isinstance(version, str) else None,
                                               "supported": list(SUPPORTED_VERSIONS)})
    if manifest.get("schema") != schema_id(format_id, version):
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
    unknown = set(requires) - FEATURES[version]
    if unknown:
        raise UnsupportedFeatureError("unsupported required feature: " + ", ".join(sorted(unknown)),
                                      context={"features": sorted(unknown)})
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

def _bounded_member(archive: ZipFile, info, limit: int) -> bytes:
    """Read exactly one indexed member with a hard decompressed byte limit.

    Using an explicitly bounded stream prevents a forged index or malformed
    deflate stream from allocating arbitrary decompressed buffers.
    """
    if info.file_size > limit:
        raise SecurityLimitError("archive member exceeds decompression budget")
    with archive.open(info, "r") as member:
        raw = member.read(limit + 1)
        if len(raw) > limit:
            raise SecurityLimitError("decompressed archive member exceeds budget")
        if member.read(1):
            raise SecurityLimitError("unexpected trailing decompressed bytes")
    if len(raw) != info.file_size:
        raise IntegrityError("decompressed member length differs from ZIP index")
    return raw


def _limits(max_archive_size, max_unpacked):
    if (type(max_archive_size) is not int or type(max_unpacked) is not int
            or max_archive_size < 1 or max_unpacked < 1):
        raise ValidationError("archive limits must be positive integers")
    return min(max_archive_size, MAX_ARCHIVE_SIZE), min(max_unpacked, MAX_UNPACKED)


def read_source_bytes(path, *, max_archive_size=MAX_ARCHIVE_SIZE) -> bytes:
    """Read a package file once, bounded, from a single open handle.

    The returned bytes are an immutable snapshot: hashing them and parsing them
    (``loads_package``) cannot disagree even if the path is replaced meanwhile.
    """
    limit, _ = _limits(max_archive_size, MAX_UNPACKED)
    with open(path, "rb") as source:
        size = os.fstat(source.fileno()).st_size
        if size > limit:
            raise ResourceLimitError("compressed spatial package too large")
        data = source.read(limit + 1)
    if len(data) > limit:
        raise ResourceLimitError("compressed spatial package too large")
    return data


def load_package(path, *, max_archive_size=MAX_ARCHIVE_SIZE, max_unpacked=MAX_UNPACKED) -> SpatialPackage:
    """Load a validated SGEO/SDRW document, returning *hydrated* native geometry.

    Archive geometry buffers are an on-disk transport detail. Binary descriptors
    are verified against the complete asset index and decoded before semantic
    Scene validation, which only accepts in-memory indexed meshes.
    """
    compressed_limit, expanded_limit = _limits(max_archive_size, max_unpacked)
    # Open exactly once: no separate path.stat followed by an unrelated
    # file open, avoiding an avoidable replacement race on the path.
    with open(path, "rb") as source:
        if os.fstat(source.fileno()).st_size > compressed_limit:
            raise ResourceLimitError("compressed spatial package too large")
        return _load_stream(source, expanded_limit)


def loads_package(data, *, max_archive_size=MAX_ARCHIVE_SIZE, max_unpacked=MAX_UNPACKED) -> SpatialPackage:
    """Load a package from in-memory bytes with exactly the file reader's checks."""
    compressed_limit, expanded_limit = _limits(max_archive_size, max_unpacked)
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise ValidationError("package data must be bytes")
    if len(data) > compressed_limit:
        raise ResourceLimitError("compressed spatial package too large")
    return _load_stream(BytesIO(bytes(data)), expanded_limit)


def _read_index(archive, expanded_limit):
    """Validate the ZIP index and manifest; no payload byte is decompressed.

    Returns ``(infos, manifest, format_id, parts)``. Shared by the full loader
    and the metadata-only inspector so both apply identical defensive checks.
    """
    infos = archive.infolist()
    if not 3 <= len(infos) <= MAX_MEMBERS:
        raise ResourceLimitError("invalid archive member count")
    names = [info.filename for info in infos]
    if len(set(names)) != len(names):
        raise ValidationError("duplicate archive members")
    total = 0
    for info in infos:
        _validate_name(info.filename)
        if info.compress_type not in (ZIP_STORED, ZIP_DEFLATED):
            raise UnsupportedFeatureError("unsupported ZIP compression method")
        if info.flag_bits & 1:
            raise UnsupportedFeatureError("encrypted ZIP entries are unsupported")
        if info.filename in ("mimetype", "manifest.json") and info.file_size > 1024 * 1024:
            raise ResourceLimitError("spatial container header exceeds 1 MiB")
        if info.file_size > expanded_limit or (
            info.file_size and info.file_size > MAX_RATIO * max(1, info.compress_size)
        ):
            raise SecurityLimitError("archive member exceeds resource budget")
        if (info.external_attr >> 16) & 0o170000 == 0o120000:
            raise ValidationError("ZIP symlink members are prohibited")
        total += info.file_size
        if total > expanded_limit:
            raise SecurityLimitError("total archive expansion exceeds budget")
    if names[:2] != ["mimetype", "manifest.json"] or infos[0].compress_type != ZIP_STORED:
        raise ValidationError("expected stored mimetype followed by manifest")
    mimetype = _bounded_member(archive, infos[0], 1024 * 1024)
    if mimetype not in MIMES.values():
        raise UnsupportedFeatureError("invalid mimetype marker")
    manifest = parse_json(_bounded_member(archive, infos[1], 1024 * 1024))
    format_id = _manifest(manifest)
    if mimetype != MIMES[format_id]:
        raise IntegrityError("manifest and MIME identity disagree")
    parts = {part["path"]: part for part in manifest["parts"]}
    if set(names) != set(parts) | {"mimetype", "manifest.json"}:
        raise IntegrityError("archive contents differ from manifest index")
    return infos, manifest, format_id, parts


def _load_stream(source, expanded_limit) -> SpatialPackage:
    try:
        with ZipFile(source, "r", allowZip64=False) as archive:
            infos, manifest, format_id, parts = _read_index(archive, expanded_limit)

            # Verify size/hash before JSON parsing, geometry hydration or
            # returning any data to the caller.
            payload = PAYLOADS[format_id]
            data: dict[str, bytes] = {}
            for info in infos[2:]:
                name = info.filename
                part = parts[name]
                if info.file_size != part["size"]:
                    raise IntegrityError("member size mismatch: " + name)
                limit = min(expanded_limit, part["size"])
                raw = _bounded_member(archive, info, limit)
                if digest(raw) != part["sha256"]:
                    raise IntegrityError("member SHA-256 mismatch: " + name)
                data[name] = raw

            document = parse_json(data.pop(payload), max_nodes=MAX_DOCUMENT_NODES)
            if not isinstance(document, dict):
                raise ValidationError("document must be an object")

            if format_id == "slai.sgeo":
                from .scene_buffers import hydrate_scene, validate_buffer_references
                validate_buffer_references(document, data)
                document = hydrate_scene(document, data)
            elif data:
                raise IntegrityError("SDRW archive contains unsupported auxiliary assets")

            source_version = manifest["format_version"]
            if source_version != CURRENT_VERSION:
                from .migration import upgrade_loaded
                manifest, document = upgrade_loaded(manifest, document)
            from ..verification.invariants import validate_document
            validate_document(manifest, document)
            return SpatialPackage(manifest, document, data, source_version)
    except (BadZipFile, EOFError, RuntimeError) as exc:
        raise IntegrityError("invalid or corrupted spatial ZIP package") from exc
