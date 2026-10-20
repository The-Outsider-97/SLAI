"""SLAI Document (SLDC) v0.1 reference implementation.

A deterministic, data-only ZIP container with exactly three entries::

    mimetype, manifest.json, document.json

The canonical extension is ``.sldc``. Verify format identity and integrity from the archive
contents, not the filename alone. The module depends only on the Python standard library and
imports no SLAI agents, so it can live in ``src/utils`` without circular imports.

Trust model
-----------
* The manifest checksum detects corruption and truncation. It is **not** a signature: anyone who
  can rewrite the archive can also rewrite the checksum. Authenticate files out of band.
* A SLDC is data only. Annotations, attributes and metadata are never executed or fetched.
* Every limit (archive, entry, JSON depth, block count, text size, compression ratio) is enforced
  before untrusted content is parsed, and enforced again on write so that nothing this module
  writes is rejected by this module's own reader.

"Canonical JSON" here means: UTF-8, ``sort_keys=True``, compact separators, no NaN/Infinity,
non-ASCII preserved. It is deterministic for this implementation; it is not RFC 8785 (JCS).

Errors
------
Format problems raise :class:`SLDCValidationError` (content is not valid SLDC) or
:class:`SLDCIntegrityError` (archive corrupted or does not match its manifest); both derive
from :class:`SLDCError`. Environment problems (missing file, permissions, ``FileExistsError``
when ``overwrite=False``) propagate as the usual :class:`OSError` subclasses.

Public surface
--------------
Container I/O    ``dumps_sldc`` ``loads_sldc`` ``write_sldc`` ``read_sldc``
                 ``sniff_sldc`` ``is_sldc``
Validation       ``validate_sldc`` ``extract_text``
Building         ``new_document`` ``make_block`` ``provenance_entry`` ``blocks_from_text``
                 ``document_from_text`` ``document_from_reader_result``
Using in SLAI    ``document_digest`` ``block_digest`` ``block_index`` ``chunk_document``
                 ``to_markdown``

Command line (inspection and packing)::

    python -m src.utils.sldc check  file.sldc [more.sldc ...]
    python -m src.utils.sldc text   file.sldc
    python -m src.utils.sldc pack   notes.md notes.sldc [--title T] [--overwrite]
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import stat
import struct
import sys
import tempfile
import uuid
import zipfile
import zlib

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, TypedDict


FORMAT_ID = "slai.sldc"
FORMAT_VERSION = "0.1.0"
MIMETYPE = "application/x-slai-sldc+zip"  # Provisional; not an IANA-registered media type.
EXTENSIONS = frozenset({".sldc"})
MEMBERS = ("mimetype", "manifest.json", "document.json")
BLOCK_TYPES = frozenset({"heading", "paragraph", "list_item", "quote", "code"})
MAX_ARCHIVE_BYTES = 25 * 1024 * 1024
MAX_DOCUMENT_BYTES = 16 * 1024 * 1024
MAX_MANIFEST_BYTES = 4 * 1024
MAX_UNCOMPRESSED_BYTES = 20 * 1024 * 1024
MAX_BLOCKS = 20_000
MAX_TEXT_CHARS = 8_000_000
MAX_JSON_DEPTH = 64
MAX_COMPRESSION_RATIO = 200

_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_LANGUAGE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,39}$")
_ID_PREFIX = re.compile(r"^[A-Za-z][A-Za-z0-9._:-]{0,31}$")
_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MIMETYPE_BYTES = MIMETYPE.encode("ascii")
_LOCAL_HEADER = struct.Struct("<4sHHHHHIIIHH")  # ZIP local file header, 30 bytes
_SNIFF_BYTES = 512
_COMPRESS_LEVEL = 6
_DOCUMENT_KEYS = frozenset({"id", "title", "language", "blocks", "metadata", "provenance"})
_BLOCK_KEYS = frozenset({"id", "type", "text", "annotations", "source_ref", "attributes"})
_MAX_REF_CHARS = 4096


class SLDCError(ValueError):
    """Base class for SLDC file, format and security validation errors."""


class SLDCValidationError(SLDCError):
    """Input content does not conform to SLDC v0.1."""


class SLDCIntegrityError(SLDCError):
    """An SLDC archive is corrupted or does not match its manifest."""


class _BlockRequired(TypedDict):
    id: str
    type: str
    text: str


class SLDCBlock(_BlockRequired, total=False):
    """
    Typed view of a block. Conventional attributes: heading ``level`` (1-6), code
    ``language``, list_item ``ordered`` (bool) and ``depth`` (int >= 0).
    """
    annotations: list[dict[str, Any]]
    source_ref: str
    attributes: dict[str, Any]


class _DocumentRequired(TypedDict):
    id: str
    title: str
    language: str
    blocks: list[SLDCBlock]


class SLDCDocument(_DocumentRequired, total=False):
    """Typed view of a v0.1 document."""
    metadata: dict[str, Any]
    provenance: list[dict[str, Any]]


# ----------------------------------------------------------------------------------------------
# JSON handling
# ----------------------------------------------------------------------------------------------
def _canonical_json(data: Any) -> bytes:
    try:
        return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (ValueError, TypeError, RecursionError) as exc:
        raise SLDCValidationError(f"Not valid canonical JSON: {exc}") from exc


def _unique_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SLDCValidationError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_constant(token: str) -> None:
    raise SLDCValidationError(f"Non-JSON constant is prohibited: {token}")


def _decode_json(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_object_pairs,
            parse_constant=_reject_constant,
        )
    except SLDCError:
        raise
    # ValueError covers JSONDecodeError, UnicodeError and CPython's integer-digit limit.
    except (ValueError, RecursionError) as exc:
        raise SLDCValidationError(f"Invalid {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise SLDCValidationError(f"{label} must be a JSON object")
    return value


def _check_json_tree(root: Any) -> None:
    """Iteratively require pure JSON data: str keys, finite numbers, bounded depth."""
    stack: list[tuple[Any, int]] = [(root, 1)]
    while stack:
        value, depth = stack.pop()
        if depth > MAX_JSON_DEPTH:
            raise SLDCValidationError(f"JSON nesting exceeds {MAX_JSON_DEPTH} levels")
        if value is None or isinstance(value, (bool, str, int)):
            continue
        if isinstance(value, float):
            if not math.isfinite(value):
                raise SLDCValidationError("NaN and Infinity are not valid JSON")
            continue
        if isinstance(value, dict):
            for key, item in value.items():
                if not isinstance(key, str):
                    raise SLDCValidationError(f"JSON object keys must be strings: {key!r}")
                stack.append((item, depth + 1))
        elif isinstance(value, list):
            stack.extend((item, depth + 1) for item in value)
        else:
            raise SLDCValidationError(f"Unsupported value type: {type(value).__name__}")


def _required_string(obj: Mapping[str, Any], key: str, *, max_len: int = 4096) -> str:
    value = obj.get(key)
    if not isinstance(value, str) or not value.strip() or len(value) > max_len:
        raise SLDCValidationError(f"{key!r} must be non-empty text <= {max_len} characters")
    return value


# ----------------------------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------------------------
def _validate_structure(document: dict[str, Any]) -> None:
    extra = set(document) - _DOCUMENT_KEYS
    if extra:
        raise SLDCValidationError(f"Unsupported v0.1 document keys: {sorted(extra)!r}")
    _check_json_tree(document)
    doc_id = _required_string(document, "id", max_len=128)
    if not _ID.fullmatch(doc_id):
        raise SLDCValidationError("Document id must be a stable ASCII identifier")
    _required_string(document, "title")
    language = _required_string(document, "language", max_len=40)
    if not _LANGUAGE.fullmatch(language):
        raise SLDCValidationError("language must be a tag such as 'en', 'nl-NL' or 'und'")
    blocks = document.get("blocks")
    if not isinstance(blocks, list) or len(blocks) > MAX_BLOCKS:
        raise SLDCValidationError(f"'blocks' must be a list of at most {MAX_BLOCKS} entries")
    if not isinstance(document.get("metadata", {}), dict):
        raise SLDCValidationError("'metadata' must be a JSON object")
    provenance = document.get("provenance", [])
    if not isinstance(provenance, list) or not all(isinstance(p, dict) for p in provenance):
        raise SLDCValidationError("'provenance' must be a list of objects")

    seen_ids: set[str] = set()
    total_text = 0
    for index, block in enumerate(blocks):
        if not isinstance(block, dict):
            raise SLDCValidationError(f"Block #{index} must be an object")
        unknown = set(block) - _BLOCK_KEYS
        if unknown:
            raise SLDCValidationError(f"Unsupported block #{index} keys: {sorted(unknown)!r}")
        block_id = _required_string(block, "id", max_len=128)
        if not _ID.fullmatch(block_id) or block_id in seen_ids:
            raise SLDCValidationError(f"Invalid or duplicate block id: {block_id!r}")
        seen_ids.add(block_id)
        block_type = block.get("type")
        if not isinstance(block_type, str) or block_type not in BLOCK_TYPES:
            raise SLDCValidationError(f"Invalid block type in block #{index}")
        if not isinstance(block.get("text"), str):
            raise SLDCValidationError(f"Block #{index} requires string 'text'")
        total_text += len(block["text"])
        if total_text > MAX_TEXT_CHARS:
            raise SLDCValidationError("Document text exceeds configured limit")
        annotations = block.get("annotations")
        if "annotations" in block and (
            not isinstance(annotations, list) or not all(isinstance(a, dict) for a in annotations)
        ):
            raise SLDCValidationError(f"Annotations in block #{index} must be objects")
        source_ref = block.get("source_ref")
        if "source_ref" in block and (not isinstance(source_ref, str) or len(source_ref) > _MAX_REF_CHARS):
            raise SLDCValidationError(f"Block #{index} source_ref must be text <= {_MAX_REF_CHARS} characters")
        if "attributes" in block and not isinstance(block["attributes"], dict):
            raise SLDCValidationError(f"Block #{index} attributes must be an object")


def _prepare(document: Mapping[str, Any]) -> bytes:
    """Validate ``document`` and return its canonical JSON payload (serialised once)."""
    if not isinstance(document, Mapping):
        raise SLDCValidationError("Document must be a mapping")
    plain = document if isinstance(document, dict) else dict(document)
    _validate_structure(plain)
    payload = _canonical_json(plain)
    if len(payload) > MAX_DOCUMENT_BYTES:
        raise SLDCValidationError("Serialized document exceeds 16 MiB")
    return payload


def validate_sldc(document: Mapping[str, Any]) -> None:
    """Validate a version-0.1 semantic document; never execute annotations."""
    _prepare(document)


def extract_text(document: Mapping[str, Any], *, validate: bool = True) -> str:
    """Create plain text from a SLDC without inventing missing content.

    Pass ``validate=False`` only for a document that was just read with ``read_sldc`` or
    ``loads_sldc`` (both already validate).
    """
    if validate:
        validate_sldc(document)
    return "\n\n".join(block["text"] for block in document["blocks"])


# ----------------------------------------------------------------------------------------------
# Container: bytes <-> document
# ----------------------------------------------------------------------------------------------
def _zipinfo(name: str, compression: int) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(filename=name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = compression
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o600) << 16
    return info


def _has_sldc_header(prefix: bytes) -> bool:
    """True if ``prefix`` starts with a ZIP whose first entry is a stored ``mimetype``."""
    if len(prefix) < _LOCAL_HEADER.size:
        return False
    signature, _, _, method, _, _, _, _, _, name_len, extra_len = _LOCAL_HEADER.unpack_from(prefix)
    if signature != b"PK\x03\x04" or method != zipfile.ZIP_STORED or name_len != len(MEMBERS[0]):
        return False
    name_end = _LOCAL_HEADER.size + name_len
    if prefix[_LOCAL_HEADER.size:name_end] != MEMBERS[0].encode("ascii"):
        return False
    data_start = name_end + extra_len
    return prefix[data_start:data_start + len(_MIMETYPE_BYTES)] == _MIMETYPE_BYTES


def _check_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    if archive.comment:
        raise SLDCValidationError("SLDC archives must not carry a ZIP comment")
    infos = archive.infolist()
    if [i.filename for i in infos] != list(MEMBERS):
        raise SLDCValidationError(
            "SLDC v0.1 requires only mimetype, manifest.json, document.json, in order"
        )
    if infos[0].compress_type != zipfile.ZIP_STORED:
        raise SLDCValidationError("The mimetype entry must be uncompressed")
    offsets = [i.header_offset for i in infos]
    if offsets[0] != 0 or offsets != sorted(set(offsets)):
        raise SLDCValidationError("ZIP entries must start at offset 0 and not overlap")
    entry_limits = (len(_MIMETYPE_BYTES), MAX_MANIFEST_BYTES, MAX_DOCUMENT_BYTES)
    total_size = 0
    for info, limit in zip(infos, entry_limits):
        if info.is_dir() or info.flag_bits & 0x41:  # bit 0 encrypted, bit 6 strong encryption
            raise SLDCValidationError("Directories and encrypted ZIP entries are unsupported")
        mode = (info.external_attr >> 16) & 0o170000
        if mode not in (0, stat.S_IFREG):
            raise SLDCValidationError("Symlinks and nonregular ZIP entries are prohibited")
        if info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
            raise SLDCValidationError("Unsupported ZIP compression method")
        if info.file_size > limit:
            raise SLDCValidationError(f"SLDC entry {info.filename!r} exceeds size limit")
        total_size += info.file_size
        if info.file_size and info.file_size / max(1, info.compress_size) > MAX_COMPRESSION_RATIO:
            raise SLDCValidationError("SLDC archive exceeds compression-ratio limit")
    if total_size > MAX_UNCOMPRESSED_BYTES:
        raise SLDCValidationError("SLDC uncompressed data exceeds size limit")
    return infos


def _read_member(archive: zipfile.ZipFile, info: zipfile.ZipInfo, limit: int) -> bytes:
    """Read one entry, never trusting the declared size; the ZIP CRC is verified at EOF."""
    with archive.open(info) as handle:
        raw = handle.read(limit + 1)
    if len(raw) > limit or len(raw) != info.file_size:
        raise SLDCIntegrityError(f"Entry {info.filename!r} length does not match its header")
    return raw


def _unpack(data: bytes) -> bytes:
    """Check container structure, manifest and checksum; return the raw document payload."""
    if len(data) > MAX_ARCHIVE_BYTES:
        raise SLDCValidationError("SLDC must be no larger than 25 MiB")
    if not _has_sldc_header(data[:_SNIFF_BYTES]):
        raise SLDCValidationError("Not an SLAI SLDC (missing mimetype header)")
    try:
        with zipfile.ZipFile(io.BytesIO(data), "r") as archive:
            infos = _check_members(archive)
            if _read_member(archive, infos[0], len(_MIMETYPE_BYTES)) != _MIMETYPE_BYTES:
                raise SLDCValidationError("Not an SLAI SLDC (mimetype mismatch)")
            manifest = _decode_json(_read_member(archive, infos[1], MAX_MANIFEST_BYTES), "manifest")
            if manifest.get("format") != FORMAT_ID:
                raise SLDCValidationError("Not an SLAI SLDC (manifest format mismatch)")
            if manifest.get("version") != FORMAT_VERSION:
                raise SLDCValidationError(
                    f"Unsupported SLDC version {manifest.get('version')!r}; "
                    f"this reader supports {FORMAT_VERSION}"
                )
            payload = _read_member(archive, infos[2], MAX_DOCUMENT_BYTES)
    except SLDCError:
        raise
    except (zipfile.BadZipFile, zlib.error, EOFError, OSError, RuntimeError,
            NotImplementedError, OverflowError, struct.error) as exc:
        raise SLDCIntegrityError(f"SLDC archive cannot be read: {exc}") from exc
    expected_size = manifest.get("document_size_bytes")
    expected_hash = manifest.get("document_sha256")
    if type(expected_size) is not int or expected_size != len(payload):
        raise SLDCIntegrityError("Document byte count does not match manifest")
    if not isinstance(expected_hash, str) or expected_hash != hashlib.sha256(payload).hexdigest():
        raise SLDCIntegrityError("Document checksum does not match manifest")
    return payload


def dumps_sldc(document: Mapping[str, Any]) -> bytes:
    """Serialise ``document`` to a deterministic SLDC v0.1 archive in memory.

    The result is re-checked with the reader's own container rules, so a document that would be
    rejected on read (for example, text so repetitive that it breaches the compression-ratio
    limit) fails here instead of producing an unreadable file.
    """
    payload = _prepare(document)
    manifest = {
        "format": FORMAT_ID,
        "version": FORMAT_VERSION,
        "document_sha256": hashlib.sha256(payload).hexdigest(),
        "document_size_bytes": len(payload),
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", allowZip64=False) as archive:
        archive.writestr(_zipinfo(MEMBERS[0], zipfile.ZIP_STORED), _MIMETYPE_BYTES)
        archive.writestr(_zipinfo(MEMBERS[1], zipfile.ZIP_DEFLATED), _canonical_json(manifest),
                         compresslevel=_COMPRESS_LEVEL)
        archive.writestr(_zipinfo(MEMBERS[2], zipfile.ZIP_DEFLATED), payload,
                         compresslevel=_COMPRESS_LEVEL)
    data = buffer.getvalue()
    if len(data) > MAX_ARCHIVE_BYTES:
        raise SLDCValidationError("SLDC archive exceeds 25 MiB")
    _unpack(data)
    return data


def loads_sldc(data: bytes | bytearray | memoryview) -> dict[str, Any]:
    """Read and fully validate a SLDC archive held in memory (DB blob, network, cache)."""
    payload = _unpack(bytes(data))
    document = _decode_json(payload, "document")
    _validate_structure(document)
    return document


# ----------------------------------------------------------------------------------------------
# Container: files
# ----------------------------------------------------------------------------------------------
def _fsync_directory(directory: Path) -> None:
    if os.name == "nt":  # Directories cannot be opened for fsync on Windows.
        return
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _publish_new(temp: Path, destination: Path) -> None:
    """Move ``temp`` to ``destination`` only if the latter does not exist (no overwrite)."""
    try:
        os.link(temp, destination)  # Atomic create-if-absent; the data is already complete.
        return
    except FileExistsError:
        raise
    except (OSError, NotImplementedError, AttributeError):
        pass  # Filesystem without hard links (FAT/exFAT, some network or VM mounts).
    claim = os.open(destination, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)  # FileExistsError
    os.close(claim)
    try:
        os.replace(temp, destination)
    except BaseException:
        destination.unlink(missing_ok=True)
        raise


def write_sldc(path: str | os.PathLike[str], document: Mapping[str, Any],
                  *, overwrite: bool = False) -> Path:
    """Atomically write a deterministic SLDC v0.1 container.

    The archive is built and verified in memory, written to a temporary file in the destination
    directory, flushed to disk and only then published, so readers never observe a partial file.
    With ``overwrite=False`` an existing path is never replaced (even concurrently) and
    ``FileExistsError`` is raised. The parent directory must already exist. Files are created
    with owner-only permissions (0600 on POSIX).
    """
    destination = Path(path)
    if destination.suffix.lower() not in EXTENSIONS:
        raise SLDCValidationError(f"Expected .sldc path, got {destination.name!r}")
    if not destination.parent.is_dir():
        raise SLDCValidationError("Destination directory must already exist")
    data = dumps_sldc(document)
    fd, temp_name = tempfile.mkstemp(prefix=".sldc-", suffix=".tmp", dir=destination.parent)
    temp = Path(temp_name)
    try:
        try:
            handle = os.fdopen(fd, "wb")
        except BaseException:
            os.close(fd)
            raise
        with handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temp, destination)
        else:
            _publish_new(temp, destination)
        _fsync_directory(destination.parent)
    finally:
        temp.unlink(missing_ok=True)
    return destination


def read_sldc(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read and validate an SLDC container regardless of its filename suffix."""
    source = Path(path)
    if not stat.S_ISREG(os.stat(source).st_mode):  # OSError (e.g. FileNotFoundError) propagates
        raise SLDCValidationError("SLDC must be a regular file")
    with open(source, "rb") as handle:
        data = handle.read(MAX_ARCHIVE_BYTES + 1)  # Bounded, even if the file grows meanwhile.
    return loads_sldc(data)


def sniff_sldc(path: str | os.PathLike[str]) -> bool:
    """Cheap check of the leading bytes only (no parsing, no integrity check).

    Use for routing or file-type detection; use ``is_sldc`` or ``read_sldc`` to trust it.
    """
    try:
        with open(path, "rb") as handle:
            return _has_sldc_header(handle.read(_SNIFF_BYTES))
    except OSError:
        return False


def is_sldc(path: str | os.PathLike[str], *, strict: bool = True) -> bool:
    """Return True only for a fully validated SLAI SLDC v0.1 file.

    With ``strict=False`` this is the cheap :func:`sniff_sldc` check instead.
    """
    if not strict:
        return sniff_sldc(path)
    try:
        read_sldc(path)
    except (SLDCError, OSError, ValueError):
        return False
    return True


# ----------------------------------------------------------------------------------------------
# Building documents
# ----------------------------------------------------------------------------------------------
def provenance_entry(activity: str, *, agent: str | None = None, source: str | None = None,
                     timestamp: str | None = None, **details: Any) -> dict[str, Any]:
    """Create a provenance record: what happened, who did it, from what, and when (UTC)."""
    if not isinstance(activity, str) or not activity.strip():
        raise SLDCValidationError("provenance activity must be non-empty text")
    stamp = timestamp or datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    entry: dict[str, Any] = {"activity": activity, "timestamp": stamp.replace("+00:00", "Z")}
    if agent:
        entry["agent"] = agent
    if source:
        entry["source"] = source
    entry.update({k: v for k, v in details.items() if v is not None})
    return entry


def make_block(block_type: str, text: str, *, block_id: str | None = None,
               source_ref: str | None = None, attributes: Mapping[str, Any] | None = None,
               annotations: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Create one block. Without ``block_id`` the id is left for :func:`new_document` to assign."""
    block: dict[str, Any] = {"type": block_type, "text": text}
    if block_id is not None:
        block["id"] = block_id
    if source_ref:
        block["source_ref"] = source_ref
    if attributes:
        block["attributes"] = dict(attributes)
    if annotations:
        block["annotations"] = [dict(a) for a in annotations]
    return block


def _assign_missing_ids(blocks: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = [dict(block) for block in blocks]
    used = {b["id"] for b in result if isinstance(b.get("id"), str)}
    counter = 1
    for block in result:
        if "id" in block:
            continue
        while f"b{counter:05d}" in used:
            counter += 1
        block["id"] = f"b{counter:05d}"
        used.add(block["id"])
    return result


def new_document(title: str, blocks: Iterable[Mapping[str, Any]] = (), *, language: str = "en",
                 doc_id: str | None = None, metadata: Mapping[str, Any] | None = None,
                 provenance: Iterable[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Assemble a document; missing block ids become ``b00001``, ``b00002``, ...

    The result is not validated until you write it or call :func:`validate_sldc`.
    """
    document: dict[str, Any] = {
        "id": doc_id or f"doc-{uuid.uuid4().hex}",
        "title": title,
        "language": language,
        "blocks": _assign_missing_ids(blocks),
    }
    if metadata:
        document["metadata"] = dict(metadata)
    if provenance:
        document["provenance"] = [dict(p) for p in provenance]
    return document


# --- Text and Markdown -> blocks --------------------------------------------------------------
_FENCE_OPEN = re.compile(r"^ {0,3}(`{3,}|~{3,})[ \t]*([^\s`]*)")
_HEADING = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.+?)(?:[ \t]+#+)?[ \t]*$")
_LIST_ITEM = re.compile(r"^([ \t]*)([-*+]|\d{1,9}[.)])[ \t]+(\S.*)$")
_RULE = re.compile(r"^ {0,3}([-*_])(?:[ \t]*\1){2,}[ \t]*$")
_QUOTE = re.compile(r"^ {0,3}>[ ]?(.*)$")

_Parsed = list[tuple[str, str, dict[str, Any]]]


def _parse_plain(text: str) -> _Parsed:
    """Paragraphs separated by blank lines; line breaks inside a paragraph are preserved."""
    parsed: _Parsed = []
    lines: list[str] = []
    for line in text.split("\n") + [""]:
        if line.strip():
            lines.append(line.rstrip())
        elif lines:
            parsed.append(("paragraph", "\n".join(lines).strip(), {}))
            lines = []
    return parsed


def _closes_fence(line: str, marker: str) -> bool:
    stripped = line.strip()
    return len(stripped) >= len(marker) and set(stripped) == {marker[0]}


def _parse_markdown(text: str) -> _Parsed:
    """A small, forgiving Markdown subset: ATX headings, fenced code, lists, quotes, paragraphs.

    Tables, HTML, setext headings and inline markup are not interpreted; their text is kept
    verbatim inside paragraphs. Horizontal rules are kept as one-line paragraphs.
    """
    parsed: _Parsed = []
    lines = text.split("\n")
    kind: str | None = None
    buffer: list[str] = []
    attrs: dict[str, Any] = {}

    def flush() -> None:
        nonlocal kind, buffer, attrs
        body = "\n".join(buffer).strip("\n")
        if kind is not None and body.strip():
            parsed.append((kind, body, attrs))
        kind, buffer, attrs = None, [], {}

    index = 0
    while index < len(lines):
        line = lines[index]
        index += 1
        fence = _FENCE_OPEN.match(line)
        if fence:
            flush()
            code: list[str] = []
            while index < len(lines) and not _closes_fence(lines[index], fence.group(1)):
                code.append(lines[index])
                index += 1
            if index >= len(lines):  # Unclosed fence: runs to the end, minus trailing blanks.
                while code and not code[-1].strip():
                    code.pop()
            index += 1  # Skip the closing fence.
            language = fence.group(2)
            parsed.append(("code", "\n".join(code), {"language": language} if language else {}))
            continue
        if not line.strip():
            flush()
            continue
        if _RULE.match(line):
            flush()
            parsed.append(("paragraph", line.strip(), {}))
            continue
        heading = _HEADING.match(line)
        if heading:
            flush()
            parsed.append(("heading", heading.group(2).strip(), {"level": len(heading.group(1))}))
            continue
        item = _LIST_ITEM.match(line)
        if item:
            flush()
            kind, buffer = "list_item", [item.group(3).rstrip()]
            attrs = {"ordered": item.group(2)[0].isdigit()}
            depth = len(item.group(1).expandtabs(4)) // 2
            if depth:
                attrs["depth"] = depth
            continue
        quote = _QUOTE.match(line)
        if quote:
            if kind != "quote":
                flush()
                kind = "quote"
            buffer.append(quote.group(1).rstrip())
            continue
        if kind == "list_item" and line[:1] in (" ", "\t"):  # Indented continuation line.
            buffer.append(line.strip())
        elif kind == "paragraph":
            buffer.append(line.strip())
        else:
            flush()
            kind, buffer = "paragraph", [line.strip()]
    flush()
    return parsed


def blocks_from_text(text: str, *, markdown: bool = False, source_ref: str | None = None,
                     id_prefix: str = "b", start: int = 1) -> list[dict[str, Any]]:
    """Split text into blocks with stable sequential ids (``b00001``, ``b00002``, ...).

    ``markdown=False`` yields paragraphs only; ``markdown=True`` also recognises headings (with
    ``level``), fenced code (with ``language``), list items (``ordered``/``depth``) and quotes.
    ``source_ref`` (for example a file name or ``"report.pdf#p3"``) is copied onto every block.
    """
    if not isinstance(text, str):
        raise SLDCValidationError("text must be a string")
    if not _ID_PREFIX.fullmatch(id_prefix):
        raise SLDCValidationError("id_prefix must be a short ASCII identifier")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    parsed = _parse_markdown(normalized) if markdown else _parse_plain(normalized)
    blocks: list[dict[str, Any]] = []
    for number, (kind, body, attrs) in enumerate(parsed, start):
        block: dict[str, Any] = {"id": f"{id_prefix}{number:05d}", "type": kind, "text": body}
        if attrs:
            block["attributes"] = attrs
        if source_ref:
            block["source_ref"] = source_ref
        blocks.append(block)
    return blocks


def document_from_text(text: str, *, title: str, language: str = "en", markdown: bool = False,
                       source_ref: str | None = None, doc_id: str | None = None,
                       metadata: Mapping[str, Any] | None = None,
                       provenance: Iterable[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Build a document from plain text or Markdown (see :func:`blocks_from_text`)."""
    return new_document(
        title, blocks_from_text(text, markdown=markdown, source_ref=source_ref),
        language=language, doc_id=doc_id, metadata=metadata, provenance=provenance,
    )


def _basename(value: Any) -> str:
    return re.split(r"[\\/]", str(value).rstrip("\\/"))[-1]


def document_from_reader_result(result: Mapping[str, Any], *, title: str | None = None,
                                language: str = "en", markdown: bool | None = None,
                                doc_id: str | None = None) -> dict[str, Any]:
    """Turn a Reader-agent ``parse()`` result into a SLDC, keeping its provenance.

    Duck-typed on the mapping the Reader returns (``content``, ``filename``/``source``,
    ``stem``, ``extension``, ``content_type``, ``parser``, ``metadata``); nothing is imported
    from the Reader. Only the file *name* is recorded, never the absolute path, and the raw and
    content SHA-256 values from the Reader are carried into ``provenance`` so the document can
    be traced back to the exact bytes it came from. ``markdown=None`` auto-detects ``.md``.
    """
    if not isinstance(result, Mapping) or not isinstance(result.get("content"), str):
        raise SLDCValidationError("Reader result must be a mapping with text 'content'")
    filename = result.get("filename") or (_basename(result["source"]) if result.get("source")
                                          else None)
    filename = filename if isinstance(filename, str) and filename else None
    extension = str(result.get("extension") or Path(filename or "").suffix).lower()
    content_type = result.get("content_type")
    if markdown is None:
        markdown = extension in {".md", ".markdown"} or content_type == "text/markdown"
    meta = result.get("metadata") if isinstance(result.get("metadata"), Mapping) else {}

    def digest(key: str) -> str | None:
        assert meta is not None
        value = meta.get(key)
        return value if isinstance(value, str) and _HEX_SHA256.fullmatch(value) else None

    warnings = result.get("warnings")
    parser = result.get("parser")
    entry = provenance_entry(
        "reader.parse",
        agent=parser if isinstance(parser, str) else None,
        source=filename,
        raw_sha256=digest("raw_sha256") or digest("sha256"),
        content_sha256=digest("content_sha256"),
        content_type=content_type if isinstance(content_type, str) else None,
        warning_count=len(warnings) if isinstance(warnings, (list, tuple)) else None,
    )
    stem = result.get("stem")
    return document_from_text(
        result["content"],
        title=title or (stem if isinstance(stem, str) and stem else None) or filename or "Untitled",
        language=language, markdown=bool(markdown), source_ref=filename, doc_id=doc_id,
        provenance=[entry],
    )


# ----------------------------------------------------------------------------------------------
# Using documents inside SLAI
# ----------------------------------------------------------------------------------------------
def document_digest(document: Mapping[str, Any]) -> str:
    """SHA-256 (hex) of the canonical document payload; equals the manifest checksum.

    A stable identity for provenance records, caches and de-duplication.
    """
    return hashlib.sha256(_prepare(document)).hexdigest()


def block_digest(block: Mapping[str, Any]) -> str:
    """SHA-256 (hex) of a block's ``type`` and ``text``, independent of its id and position.

    Lets an answer that cites a block be re-checked against the exact text it was based on.
    """
    if not isinstance(block.get("type"), str) or not isinstance(block.get("text"), str):
        raise SLDCValidationError("Block requires string 'type' and 'text'")
    core = {"type": block["type"], "text": block["text"]}
    return hashlib.sha256(_canonical_json(core)).hexdigest()


def block_index(document: Mapping[str, Any], *, validate: bool = True) -> dict[str, dict[str, Any]]:
    """Map block id -> block, for resolving citations (``chunk['block_ids']``) back to text."""
    if validate:
        validate_sldc(document)
    return {block["id"]: block for block in document["blocks"]}


def _split_text(text: str, limit: int) -> list[str]:
    """Split ``text`` into pieces of at most ``limit`` chars, preferring whitespace."""
    pieces: list[str] = []
    start, length = 0, len(text)
    while start < length:
        end = min(start + limit, length)
        if end < length:
            cut = max(text.rfind(" ", start, end), text.rfind("\n", start, end))
            if cut > start + limit // 2:
                end = cut
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        start = end
        while start < length and text[start].isspace():
            start += 1
    return pieces


def chunk_document(document: Mapping[str, Any], max_chars: int = 2000, *, validate: bool = True) -> list[dict[str, Any]]:
    """Group consecutive blocks into retrieval/prompt-sized chunks that remember their blocks.

    Blocks are never split unless one alone exceeds ``max_chars``; such a block is cut at
    whitespace and each piece is flagged ``"partial": True``. Each chunk is
    ``{"index", "text", "block_ids", "partial"}`` where ``text`` joins blocks with a blank line
    (as :func:`extract_text` does), so every chunk can be traced to stable block ids.
    """
    if type(max_chars) is not int or max_chars < 1:
        raise ValueError("max_chars must be a positive integer")
    if validate:
        validate_sldc(document)
    chunks: list[dict[str, Any]] = []
    ids: list[str] = []
    parts: list[str] = []
    size = 0

    def flush() -> None:
        nonlocal size
        if parts:
            chunks.append({"index": len(chunks), "text": "\n\n".join(parts), "block_ids": list(ids), "partial": False})
        ids.clear()
        parts.clear()
        size = 0

    for block in document["blocks"]:
        text = block["text"].strip()
        if not text:
            continue
        if len(text) > max_chars:
            flush()
            for piece in _split_text(text, max_chars):
                chunks.append({"index": len(chunks), "text": piece,
                               "block_ids": [block["id"]], "partial": True})
            continue
        if parts and size + 2 + len(text) > max_chars:
            flush()
        size += len(text) + (2 if parts else 0)
        parts.append(text)
        ids.append(block["id"])
    flush()
    return chunks


def to_markdown(document: Mapping[str, Any], *, validate: bool = True) -> str:
    """Render a document as Markdown for display or LLM prompts (uses conventional attributes)."""
    if validate:
        validate_sldc(document)
    result = ""
    previous = None
    for block in document["blocks"]:
        kind, text = block["type"], block["text"]
        attrs = block.get("attributes") or {}
        if kind == "heading":
            level = attrs.get("level")
            level = level if type(level) is int and 1 <= level <= 6 else 1
            piece = "#" * level + " " + " ".join(text.split("\n"))
        elif kind == "list_item":
            depth = attrs.get("depth")
            indent = "  " * (min(depth, 32) if type(depth) is int and depth > 0 else 0)
            marker = "1." if attrs.get("ordered") is True else "-"
            first, *rest = text.split("\n")
            piece = indent + marker + " " + first + "".join(f"\n{indent}  {r}" for r in rest)
        elif kind == "quote":
            piece = "\n".join(f"> {line}" if line else ">" for line in text.split("\n"))
        elif kind == "code":
            fence = "`" * max(3, max((len(m) for m in re.findall(r"`+", text)), default=0) + 1)
            language = attrs.get("language")
            tag = language if isinstance(language, str) and re.fullmatch(r"[^\s`]+", language) \
                else ""
            piece = f"{fence}{tag}\n{text}\n{fence}"
        else:
            piece = text
        if result:
            result += "\n" if kind == previous == "list_item" else "\n\n"
        result += piece
        previous = kind
    return result + "\n" if result else ""


# ----------------------------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------------------------
def _write_stdout(text: str) -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass
    sys.stdout.write(text if text.endswith("\n") else text + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sldc", description="Inspect and pack SLDC files.")
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="verify structure, integrity and content")
    check.add_argument("paths", nargs="+")
    text = commands.add_parser("text", help="print the plain text of a document")
    text.add_argument("path")
    text.add_argument("--markdown", action="store_true", help="render as Markdown instead")
    pack = commands.add_parser("pack", help="create a .sldc from a text or Markdown file")
    pack.add_argument("source")
    pack.add_argument("destination")
    pack.add_argument("--title")
    pack.add_argument("--language", default="en")
    pack.add_argument("--markdown", action="store_true", default=None,
                      help="parse Markdown (default: on for .md/.markdown files)")
    pack.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    if args.command == "check":
        failures = 0
        for path in args.paths:
            try:
                document = read_sldc(path)
            except (SLDCError, OSError) as exc:
                failures += 1
                print(f"FAIL {path}: {exc}")
            else:
                print(f"OK   {path}: id={document['id']} blocks={len(document['blocks'])} "
                      f"sha256={document_digest(document)}")
        return 1 if failures else 0
    try:
        if args.command == "text":
            document = read_sldc(args.path)
            _write_stdout(to_markdown(document, validate=False) if args.markdown
                          else extract_text(document, validate=False))
        else:
            source = Path(args.source)
            as_markdown = (source.suffix.lower() in {".md", ".markdown"}
                           if args.markdown is None else args.markdown)
            content = source.read_text(encoding="utf-8-sig")
            document = document_from_text(
                content, title=args.title or source.stem, language=args.language,
                markdown=as_markdown, source_ref=source.name,
                provenance=[provenance_entry("sldc.pack", source=source.name, raw_sha256=hashlib.sha256(source.read_bytes()).hexdigest())],
            )
            print(write_sldc(args.destination, document, overwrite=args.overwrite))
    except (SLDCError, OSError, UnicodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


__all__ = [
    "FORMAT_ID", "FORMAT_VERSION", "MIMETYPE", "EXTENSIONS", "MEMBERS", "BLOCK_TYPES",
    "MAX_ARCHIVE_BYTES", "MAX_DOCUMENT_BYTES", "MAX_BLOCKS", "MAX_TEXT_CHARS",
    "SLDCError", "SLDCValidationError", "SLDCIntegrityError",
    "SLDCBlock", "SLDCDocument",
    "validate_sldc", "extract_text",
    "dumps_sldc", "loads_sldc", "read_sldc", "write_sldc", "sniff_sldc",
    "is_sldc",
    "new_document", "make_block", "provenance_entry", "blocks_from_text", "document_from_text",
    "document_from_reader_result",
    "document_digest", "block_digest", "block_index", "chunk_document", "to_markdown",
]


if __name__ == "__main__":
    raise SystemExit(main())