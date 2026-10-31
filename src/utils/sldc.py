"""SLAI Document (SLDC): a versioned, data-only publishing format and its reference implementation.

An SLDC is a deterministic ZIP container that holds *data only*. Opening, validating, searching or
converting a document never executes anything it contains and never fetches anything it
references. The canonical extension is ``.sldc``. Verify format identity and integrity from the
archive contents, not the filename alone. The module needs only the Python standard library to
read, write, migrate and export documents; encryption and signatures use the maintained
``cryptography`` package, which is imported lazily and only when one of those features is used. The
module imports no SLAI agents, so it can live in ``src/utils`` without circular imports.

Versions
--------
``MAJOR.MINOR.PATCH`` in ``manifest.json``. A reader dispatches on the version and never guesses.

``0.1.0``  The original text container: exactly ``mimetype``, ``manifest.json``, ``document.json``
           with five block types. It is frozen and stays readable and writable through the
           ``*_sldc`` functions, byte-for-byte as before.
``1.0.0``  The publishing profile (this module's current ``FORMAT_VERSION``): inline styles,
           paragraph/character styles, page sections, tables, figures with embedded raster images,
           citations and footnotes, publication metadata, revisions, review comments, declarative
           EMBED records, detached signatures and an authenticated-encryption envelope. All of it
           is inventoried in the manifest with per-part size and SHA-256.
           Adds editable scientific equations, inline scientific/image objects and declarative
           image transforms/placement. New mandatory features are declared in ``requires``.
           Old 1.0.0 packages retain their version when read and saved unchanged.
           
Compatibility rules (1.x)
  * Unknown keys are rejected, except inside ``extensions`` (document or block level, keys of the
    form ``vendor.name``) and ``x-`` prefixed manifest keys. Both are inert data and are preserved
    by this reader.
  * ``manifest["requires"]`` names the features a reader must understand to present the document
    faithfully (``tables``, ``figures``, ``embeds``). A reader that does not know one of them must
    stop editing or fall back to a read-only preview (:func:`inspect_package` reports this without
    loading the content); it must never drop the feature silently on save.
  * A newer minor version is readable if it only adds extensions and requires nothing unknown. A
    new major version is not readable. Old files are never rewritten in place: migration produces
    a new file, keeps the original bytes and returns a machine-readable report.
  * Downgrading to 0.1.0 is an explicit, lossy export with a loss report; the result says so.

Trust model
-----------
* Manifest checksums detect corruption and truncation. They are **not** signatures: anyone who
  can rewrite an archive can rewrite its checksums.
* Provenance, authors and timestamps are assertions by whoever wrote the file.
* A **signature** (Ed25519, detached, domain-separated) binds a key to the exact bytes of a
  revision, so later changes are detected. It does *not* prove who holds the key. That needs a
  trust decision you make (``trusted_keys``). A signer's clock is not evidence of time either.
  :func:`verify_package` therefore reports cryptographic validity, signer trust and modification
  separately and never produces a single "verified" flag.
* **Encryption** wraps a complete package in AES-256-GCM. The content key is wrapped per recipient
  (password via scrypt, or X25519). Whoever can read the plaintext once can keep it: revoking a
  recipient protects only future copies. Losing every credential loses the document, unless a
  ``recovery`` recipient was declared when it was encrypted. File size and the recipient list
  stay visible. Private keys and passwords are never stored in a document.
* EMBED records are *requests*, not grants: a document may describe a registered transformation
  and the capabilities it would like, but opening it never runs one, and the host decides.
* Images are accepted only as PNG, JPEG, GIF or WebP, checked by header and size budgets and never
  decoded here. SVG is refused because it can carry script. Decode untrusted images in a
  separate, resource-limited process.
* Every limit is enforced before untrusted content is parsed and again on write, so nothing this
  module writes is rejected by this module's own reader.

"Canonical JSON" means UTF-8, ``sort_keys=True``, compact separators, no NaN/Infinity, non-ASCII
preserved. It is deterministic for this implementation; it is not RFC 8785 (JCS). Signatures and
revision ids depend on it, so another implementation must reproduce it exactly (see the
known-answer test).

Errors
------
Format problems raise :class:`SLDCValidationError` (not valid SLDC), its subclass
:class:`SLDCUnsupportedError` (valid in principle, but a version or feature this reader does not
support), :class:`SLDCIntegrityError` (corrupted or does not match its manifest),
:class:`SLDCDecryptionError` (no credential unlocks the file) or :class:`SLDCDependencyError`
(``cryptography`` is needed and missing). All derive from :class:`SLDCError`. Environment problems
(missing file, permissions, ``FileExistsError`` when ``overwrite=False``) propagate as the usual
:class:`OSError` subclasses.

Public surface
--------------
Packages (1.0)   ``SLDCPackage`` ``Asset`` ``read_package`` ``write_package`` ``loads_package``
                 ``dumps_package`` ``inspect_package`` ``validate_package`` ``new_package``
                 ``make_asset`` ``add_asset`` ``prune_assets`` ``make_table`` ``make_figure``
                 ``revision_id``
Migration        ``migrate_package`` ``migrate_file`` ``export_legacy``
Collaboration    ``new_block_id`` ``next_revision`` ``diff_documents`` ``merge_documents``
                 ``merge_packages`` ``add_comment`` ``reply_to_thread`` ``set_thread_status``
                 ``comment_anchor_status``
Signing          ``Ed25519Signer`` ``sign_package`` ``verify_package`` ``key_fingerprint``
Encryption       ``password_recipient`` ``x25519_recipient`` ``x25519_generate``
                 (``recipients=`` on ``dumps_package``/``write_package``), ``rekey``,
                 ``add_recipient``
Version 0.1      ``dumps_sldc`` ``loads_sldc`` ``write_sldc`` ``read_sldc`` (frozen text format)
Both versions    ``validate_sldc`` ``extract_text`` ``sniff_sldc`` ``is_sldc`` ``new_document``
                 ``make_block`` ``provenance_entry`` ``blocks_from_text`` ``document_from_text``
                 ``document_from_reader_result`` ``document_digest`` ``block_digest``
                 ``block_text`` ``block_index`` ``chunk_document`` ``to_markdown``

Command line::

    python -m src.utils.sldc check    file.sldc [more.sldc ...]   [--password-env VAR]
    python -m src.utils.sldc info     file.sldc
    python -m src.utils.sldc text     file.sldc [--markdown]      [--password-env VAR]
    python -m src.utils.sldc pack     notes.md notes.sldc [--title T] [--legacy] [--overwrite]
    python -m src.utils.sldc migrate  old.sldc new.sldc [--report report.json] [--overwrite]
    python -m src.utils.sldc export-legacy new.sldc old.sldc [--report report.json] [--overwrite]
    python -m src.utils.sldc verify   file.sldc [--trusted KEY_ID ...] [--revoked KEY_ID ...]

``pack`` writes 1.0.0 unless ``--legacy`` is given. Passwords are read from the environment
variable named by ``--password-env``, never from an argument. ``verify`` exits 0 only if every
signature is valid *and* made by a key you listed as trusted.
"""
from __future__ import annotations

import argparse
import base64
import bisect
import copy
import functools
import hashlib
import io
import itertools
import json
import math
import os
import re
import stat
import struct
import sys
import tempfile
import types
import unicodedata
import uuid
import zipfile
import zlib

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, TypedDict
from urllib.parse import urlsplit


FORMAT_ID = "slai.sldc"
LEGACY_VERSION = "0.1.0"
FORMAT_VERSION = "1.0.0"  # What new documents are written as.
LATEST_KNOWN_VERSION = "1.1.0"  # Most recent minor schema this reader understands.
SCHEMA = "slai.sldc/1"  # Value of ``document["schema"]`` in 1.x documents.
MIMETYPE = "application/x-slai-sldc+zip"  # Provisional; not an IANA-registered media type.
EXTENSIONS = frozenset({".sldc"})
MEMBERS = ("mimetype", "manifest.json", "document.json")  # The fixed 0.1.0 member list.
BLOCK_TYPES = frozenset({"heading", "paragraph", "list_item", "quote", "code"})  # 0.1.0
BLOCK_TYPES_V1 = BLOCK_TYPES | {"table", "figure", "embed", "equation"}
REQUIRED_FEATURES = frozenset({"tables", "figures", "embeds"})
SUPPORTED_FEATURES = REQUIRED_FEATURES | {"scientific", "inline-media", "image-layout", "font-runs", "native-toc", "paragraph-layout", "page-bands", "cover-pages"}

# --- Limits, 0.1.0 (unchanged) ---------------------------------------------------------------
MAX_ARCHIVE_BYTES = 25 * 1024 * 1024
MAX_DOCUMENT_BYTES = 16 * 1024 * 1024
MAX_MANIFEST_BYTES = 4 * 1024
MAX_UNCOMPRESSED_BYTES = 20 * 1024 * 1024
MAX_BLOCKS = 20_000
MAX_TEXT_CHARS = 8_000_000
MAX_JSON_DEPTH = 64
MAX_COMPRESSION_RATIO = 200
_RATIO_CHECK_MIN_BYTES = 1024 * 1024  # Below this the absolute size caps already bound the cost.

# --- Limits, 1.x. Images and rich content get explicit resource budgets, not larger strings. ---
MAX_PACKAGE_BYTES = 48 * 1024 * 1024
MAX_PACKAGE_UNCOMPRESSED_BYTES = 64 * 1024 * 1024
MAX_MANIFEST_BYTES_V1 = 256 * 1024
MAX_COMMENTS_BYTES = 4 * 1024 * 1024
MAX_SIGNATURE_BYTES = 64 * 1024
MAX_ASSETS = 256
MAX_ASSET_BYTES = 8 * 1024 * 1024
MAX_TOTAL_ASSET_BYTES = 32 * 1024 * 1024
MAX_SIGNATURES = 64
MAX_IMAGE_SIDE = 16_384
MAX_IMAGE_PIXELS = 64_000_000
MAX_TOTAL_IMAGE_PIXELS = 256_000_000
MAX_PARTS = MAX_ASSETS + MAX_SIGNATURES + 2
MAX_STYLES = 256
MAX_SECTIONS = 256
MAX_REFERENCES = 10_000
MAX_EMBEDS = 1_024
MAX_HISTORY = 10_000
MAX_SPANS_PER_BLOCK = 10_000
MAX_SPANS = 500_000
MAX_TABLE_ROWS = 10_000
MAX_TABLE_COLUMNS = 256
MAX_TABLE_CELLS = 200_000
MAX_COMMENT_THREADS = 10_000
MAX_THREAD_MESSAGES = 1_000
MAX_RECIPIENTS = 16
MIN_PASSWORD_CHARS = 8

_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_LANGUAGE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,39}$")
_ID_PREFIX = re.compile(r"^[A-Za-z][A-Za-z0-9._:-]{0,31}$")
_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ASSET_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9 _.-]{0,63}$")  # Style names.
_KIND = re.compile(r"^[a-z][a-z0-9._-]{0,63}$")
_CAPABILITY = re.compile(r"^[a-z][a-z0-9._:-]{0,63}$")
_EXTENSION_KEY = re.compile(r"^[a-z][a-z0-9-]{0,31}(?:\.[A-Za-z0-9_-]{1,64}){1,3}$")
_COLOR = re.compile(r"^#[0-9a-fA-F]{6}$")
_ISO_TIME = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2})(?:T(\d{2}):(\d{2}):(\d{2})(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2}))?$"
)
_MIMETYPE_BYTES = MIMETYPE.encode("ascii")
_LOCAL_HEADER = struct.Struct("<4sHHHHHIIIHH")  # ZIP local file header, 30 bytes
_SNIFF_BYTES = 512
_COMPRESS_LEVEL = 6
_DOCUMENT_KEYS = frozenset({"id", "title", "language", "blocks", "metadata", "provenance"})
_BLOCK_KEYS = frozenset({"id", "type", "text", "annotations", "source_ref", "attributes"})
_MAX_REF_CHARS = 4096


class SLDCError(ValueError):
    """Base class for SLDC file, format, security and dependency errors."""


class SLDCValidationError(SLDCError):
    """Input content does not conform to SLDC."""


class SLDCUnsupportedError(SLDCValidationError):
    """Well-formed so far, but uses a version or required feature this reader does not support."""


class SLDCIntegrityError(SLDCError):
    """An SLDC archive is corrupted or does not match its manifest."""


class SLDCDecryptionError(SLDCError):
    """No supplied credential unlocks the file (or the ciphertext was modified)."""


class SLDCDependencyError(SLDCError):
    """The optional ``cryptography`` package is needed for this operation and is not installed."""


class _BlockRequired(TypedDict):
    id: str
    type: str
    text: str


class SLDCBlock(_BlockRequired, total=False):
    """
    Typed view of a block. 0.1.0: conventional attributes ``level`` (1-6), ``language``,
    ``ordered`` (bool) and ``depth`` (int >= 0), plus free-form ``annotations``. 1.x: the same
    attributes are validated, ``annotations`` is replaced by typed ``spans``, and blocks may carry
    ``style``, ``section``, ``table``, ``figure``, ``embed`` and ``extensions``.
    """
    annotations: list[dict[str, Any]]
    source_ref: str
    attributes: dict[str, Any]
    spans: list[dict[str, Any]]
    style: str
    section: str
    table: dict[str, Any]
    figure: dict[str, Any]
    equation: dict[str, Any]
    embed: str
    extensions: dict[str, Any]


class _DocumentRequired(TypedDict):
    id: str
    title: str
    language: str
    blocks: list[SLDCBlock]


class SLDCDocument(_DocumentRequired, total=False):
    """Typed view of a document. ``schema`` marks 1.x; the keys after ``provenance`` are 1.x only."""
    metadata: dict[str, Any]
    provenance: list[dict[str, Any]]
    schema: str
    publication: dict[str, Any]
    styles: dict[str, Any]
    sections: list[dict[str, Any]]
    references: list[dict[str, Any]]
    embeds: list[dict[str, Any]]
    revision: dict[str, Any]
    history: list[dict[str, Any]]
    extensions: dict[str, Any]


@dataclass
class Asset:
    """An embedded raster image. ``media_type`` is checked against the bytes on validation."""
    id: str
    media_type: str
    data: bytes


@dataclass
class SLDCPackage:
    """Everything a file holds: the document, its assets, review comments and signatures.

    ``version`` is ``"1.0.0"`` for the publishing profile or ``"0.1.0"`` for a legacy text file
    (then only ``document`` is used). ``comments`` is the list of review threads and
    ``signatures`` the list of detached signature records.
    """
    document: dict[str, Any]
    assets: dict[str, Asset] = field(default_factory=dict)
    comments: list[dict[str, Any]] = field(default_factory=list)
    signatures: list[dict[str, Any]] = field(default_factory=list)
    version: str = FORMAT_VERSION

    def copy(self) -> SLDCPackage:
        """An independent deep copy (functions in this module return copies, never mutate)."""
        return SLDCPackage(copy.deepcopy(self.document), dict(self.assets),
                           copy.deepcopy(self.comments), copy.deepcopy(self.signatures),
                           self.version)


# ----------------------------------------------------------------------------------------------
# JSON handling (shared by every version)
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
# Small shared helpers
# ----------------------------------------------------------------------------------------------
def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64d(text: Any, label: str, *, length: int | None = None) -> bytes:
    """Strict, canonical base64url without padding (re-encoding must reproduce the input)."""
    if not isinstance(text, str) or not text.isascii():
        raise SLDCValidationError(f"{label} must be base64url text")
    try:
        raw = base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))
    except ValueError as exc:
        raise SLDCValidationError(f"{label} is not valid base64url") from exc
    if _b64e(raw) != text or (length is not None and len(raw) != length):
        raise SLDCValidationError(f"{label} is not canonical base64url of the expected length")
    return raw


def _check_time(value: Any, label: str) -> str:
    """Require an ISO 8601 date or UTC/offset timestamp. It is an assertion, never evidence."""
    match = _ISO_TIME.fullmatch(value) if isinstance(value, str) else None
    if match is None:
        raise SLDCValidationError(f"{label} must be an ISO 8601 date or timestamp")
    year, month, day, hour, minute, second = (int(g) if g else 0 for g in match.groups())
    try:
        datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)
    except ValueError as exc:
        raise SLDCValidationError(f"{label} is not a real date: {value!r}") from exc
    return value


def _guarded(function: Callable[..., Any]) -> Callable[..., Any]:
    """Turn a type-confusion accident on hostile input into a validation error.

    The validators check types explicitly; this is the safety net behind them, so that no
    malformed document can ever surface as a bare TypeError, KeyError or AttributeError.
    """
    @functools.wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return function(*args, **kwargs)
        except SLDCError:
            raise
        except (TypeError, KeyError, AttributeError, IndexError, ValueError, OverflowError,
                RecursionError) as exc:
            raise SLDCValidationError(f"Malformed content ({type(exc).__name__}: {exc})") from exc
    return wrapper


def _has(allowed: Any, value: Any) -> bool:
    """Membership that is simply False for values that cannot be members (lists, objects)."""
    try:
        return value in allowed
    except TypeError:
        return False


def _check_keys(obj: Mapping[str, Any], allowed: Iterable[str], label: str) -> None:
    unknown = set(obj) - set(allowed)
    if unknown:
        raise SLDCValidationError(f"Unsupported {label} keys: {sorted(unknown, key=str)!r}")


def _text_field(obj: Mapping[str, Any], key: str, label: str, max_len: int, *,
                required: bool = False, empty_ok: bool = False) -> str | None:
    if key not in obj:
        if required:
            raise SLDCValidationError(f"{label} requires {key!r}")
        return None
    value = obj[key]
    if not isinstance(value, str) or len(value) > max_len or not (empty_ok or value.strip()):
        raise SLDCValidationError(f"{label} {key!r} must be text of at most {max_len} characters")
    return value


def _bounded_int(value: Any, low: int, high: int, label: str) -> int:
    if type(value) is not int or not low <= value <= high:
        raise SLDCValidationError(f"{label} must be an integer from {low} to {high}")
    return value


def _bounded_number(value: Any, low: float, high: float, label: str) -> float:
    if type(value) not in (int, float) or not low <= value <= high:
        raise SLDCValidationError(f"{label} must be a number from {low} to {high}")
    return value


def _flag(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise SLDCValidationError(f"{label} must be true or false")
    return value


def _id_field(value: Any, label: str, pattern: re.Pattern[str] = _ID) -> str:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise SLDCValidationError(f"{label} must be a stable ASCII identifier, got {value!r}")
    return value


def _check_href(value: Any, label: str) -> str:
    """Links are inert data. Only absolute http(s) and mailto URLs are accepted: no javascript:,
    data:, file: or relative targets, no embedded credentials, no whitespace or control codes."""
    if not isinstance(value, str) or not 1 <= len(value) <= 2048 \
            or any(ord(ch) <= 0x20 or ord(ch) == 0x7F or ch == "\\" for ch in value):
        raise SLDCValidationError(f"{label} must be a URL without whitespace, up to 2048 characters")
    try:
        parts = urlsplit(value)
        netloc = parts.netloc
    except ValueError as exc:
        raise SLDCValidationError(f"{label} is not a valid URL") from exc
    scheme = parts.scheme.lower()
    if scheme in ("http", "https"):
        if not netloc or "@" in netloc:
            raise SLDCValidationError(f"{label} needs a host and must not embed credentials")
    elif scheme == "mailto":
        if not parts.path:
            raise SLDCValidationError(f"{label} needs an address")
    else:
        raise SLDCValidationError(f"{label} must use http, https or mailto, not {scheme or 'none'!r}")
    return value


def _check_extensions(value: Any, label: str) -> None:
    """Extensions are namespaced, inert JSON that readers preserve and otherwise ignore."""
    if not isinstance(value, dict):
        raise SLDCValidationError(f"{label} extensions must be an object")
    for key in value:
        if not _EXTENSION_KEY.fullmatch(key):
            raise SLDCValidationError(f"{label} extension key {key!r} must look like 'vendor.name'")

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



# ----------------------------------------------------------------------------------------------
# Validation, 1.x
# ----------------------------------------------------------------------------------------------
_DOCUMENT_KEYS_V1 = frozenset({
    "schema", "id", "title", "language", "blocks", "metadata", "provenance", "publication",
    "styles", "sections", "references", "embeds", "revision", "history", "extensions",
})
_BLOCK_KEYS_V1 = frozenset({
    "id", "type", "text", "spans", "source_ref", "attributes", "style", "section", "table",
    "figure", "equation", "embed", "extensions",
})
_ALIGN = frozenset({"left", "center", "right", "justify"})
_CHAR_PROPS = frozenset({"based_on", "font_family", "font_size_pt", "bold", "italic", "underline",
                         "strike", "small_caps", "color", "background", "letter_spacing_pt", "word_spacing_pt"})
_PARA_PROPS = _CHAR_PROPS | {"align", "line_height", "space_before_pt", "space_after_pt",
                             "indent_mm", "first_line_indent_mm", "keep_with_next"}
_STYLE_FLAGS = frozenset({"bold", "italic", "underline", "strike", "small_caps", "keep_with_next"})
_STYLE_RANGES = {"font_size_pt": (4, 400), "line_height": (0.5, 5), "space_before_pt": (0, 1000),
                 "space_after_pt": (0, 1000), "indent_mm": (0, 1000),
                 "first_line_indent_mm": (-100, 1000), "letter_spacing_pt": (-5, 50),
                 "word_spacing_pt": (-20, 100)}
_FONT_FAMILY = re.compile(r"^[^\x00-\x1f\x7f{};<>\\\"'`]{1,100}$")
_CODE_LANGUAGE = re.compile(r"^[^\s`]{1,40}$")
_PAGE_SIZES_MM = {"A3": (297.0, 420.0), "A4": (210.0, 297.0), "A5": (148.0, 210.0),
                  "B5": (176.0, 250.0), "Letter": (215.9, 279.4), "Legal": (215.9, 355.6),
                  "Tabloid": (279.4, 431.8)}
_NUMBER_FORMATS = frozenset({"decimal", "lower-roman", "upper-roman", "lower-alpha", "upper-alpha"})
_TEMPLATE_TOKENS = frozenset({"{page}", "{pages}", "{title}"})
_FORMAT_FLAGS = frozenset({"bold", "italic", "underline", "strike", "code", "superscript",
                           "subscript"})
_INLINE_FONT_PROPS = frozenset({"font_family", "font_size_pt", "letter_spacing_pt", "word_spacing_pt"})
_TOC_EXTENSION = "slai.sldc.toc"
_REVISION_KEYS = frozenset({"number", "parents", "created", "author", "note"})
_PUBLICATION_KEYS = frozenset({"subtitle", "authors", "publisher", "published", "identifier",
                               "rights", "keywords", "abstract", "edition", "cover_asset"})
_CITATION_FIELDS = {"key": 128, "title": 1000, "container": 500, "publisher": 300, "volume": 40,
                    "issue": 40, "pages": 40, "note": 2000}


class _Context:
    """What a block may refer to, plus running budgets (text, spans, table cells)."""

    def __init__(self, assets: Mapping[str, Any] | None) -> None:
        self.paragraph_styles: set[str] = set()
        self.character_styles: set[str] = set()
        self.sections: dict[str, int] = {}
        self.references: set[str] = set()
        self.embeds: set[str] = set()
        self.block_ids: set[str] = set()
        self.assets = assets  # None: asset existence is checked by the package, not here.
        self.referenced_assets: set[str] = set()
        self.total_text = 0
        self.total_spans = 0
        self.total_cells = 0

    def add_text(self, length: int) -> None:
        self.total_text += length
        if self.total_text > MAX_TEXT_CHARS:
            raise SLDCValidationError("Document text exceeds configured limit")


def _validate_style_props(props: Any, label: str, *, paragraph: bool) -> None:
    if not isinstance(props, dict):
        raise SLDCValidationError(f"{label} must be an object")
    _check_keys(props, _PARA_PROPS if paragraph else _CHAR_PROPS, label)
    for key, value in props.items():
        where = f"{label}.{key}"
        if key in _STYLE_FLAGS:
            _flag(value, where)
        elif key in _STYLE_RANGES:
            low, high = _STYLE_RANGES[key]
            _bounded_number(value, low, high, where)
        elif key in ("color", "background"):
            if not isinstance(value, str) or not _COLOR.fullmatch(value):
                raise SLDCValidationError(f"{where} must be a #rrggbb colour")
        elif key == "font_family":
            if not isinstance(value, str) or not _FONT_FAMILY.fullmatch(value):
                raise SLDCValidationError(f"{where} must be a plain font family name")
        elif key == "align":
            if not _has(_ALIGN, value):
                raise SLDCValidationError(f"{where} must be one of {sorted(_ALIGN)}")
        elif key == "based_on":
            _id_field(value, where, _NAME)


def _validate_styles(styles: Any, ctx: _Context) -> None:
    if not isinstance(styles, dict):
        raise SLDCValidationError("'styles' must be an object")
    _check_keys(styles, {"paragraph", "character"}, "styles")
    for kind, names in (("paragraph", ctx.paragraph_styles), ("character", ctx.character_styles)):
        group = styles.get(kind, {})
        if not isinstance(group, dict) or len(group) > MAX_STYLES:
            raise SLDCValidationError(f"styles.{kind} must be an object of at most {MAX_STYLES}")
        for name, props in group.items():
            _id_field(name, f"{kind} style name", _NAME)
            _validate_style_props(props, f"styles.{kind}.{name}", paragraph=kind == "paragraph")
        names.update(group)
        for name, props in group.items():  # based_on must exist and must not loop.
            seen, current = {name}, props.get("based_on")
            while current is not None:
                if current not in group or current in seen:
                    raise SLDCValidationError(f"styles.{kind}.{name} has a missing or circular 'based_on'")
                seen.add(current)
                current = group[current].get("based_on")


def _check_template(value: Any, label: str) -> None:
    """Header/footer text may use only {page}, {pages} and {title}; ``{{`` and ``}}`` are literal."""
    if not isinstance(value, str) or len(value) > 500:
        raise SLDCValidationError(f"{label} must be text of at most 500 characters")
    rest = value.replace("{{", "").replace("}}", "")
    for token in re.findall(r"\{[^{}]*\}", rest):
        if token not in _TEMPLATE_TOKENS:
            raise SLDCValidationError(f"{label} uses unknown field {token}")
    if re.search(r"[{}]", re.sub(r"\{[^{}]*\}", "", rest)):
        raise SLDCValidationError(f"{label} has an unmatched brace (write {{{{ or }}}} for a literal one)")


def _validate_sections(sections: Any, ctx: _Context) -> None:
    if not isinstance(sections, list) or len(sections) > MAX_SECTIONS:
        raise SLDCValidationError(f"'sections' must be a list of at most {MAX_SECTIONS} entries")
    for index, section in enumerate(sections):
        label = f"sections[{index}]"
        if not isinstance(section, dict):
            raise SLDCValidationError(f"{label} must be an object")
        _check_keys(section, {"id", "page", "columns", "column_gap_mm", "header", "footer",
                              "numbering", "role"}, label)
        if "role" in section and section["role"] not in ("cover", "body", "front_matter"):
            raise SLDCValidationError(f"{label}.role must be cover, front_matter or body")
        section_id = _id_field(section.get("id"), f"{label}.id")
        if section_id in ctx.sections:
            raise SLDCValidationError(f"Duplicate section id {section_id!r}")
        ctx.sections[section_id] = index
        if "page" in section:
            _validate_page(section["page"], f"{label}.page")
        if "columns" in section:
            _bounded_int(section["columns"], 1, 8, f"{label}.columns")
        if "column_gap_mm" in section:
            _bounded_number(section["column_gap_mm"], 0, 100, f"{label}.column_gap_mm")
        for edge in ("header", "footer"):
            if edge in section:
                template = section[edge]
                if not isinstance(template, dict):
                    raise SLDCValidationError(f"{label}.{edge} must be an object")
                _check_keys(template, {"left", "center", "right", "style"}, f"{label}.{edge}")
                for position in ("left", "center", "right"):
                    if position in template:
                        _check_template(template[position], f"{label}.{edge}.{position}")
                if "style" in template and not _has(ctx.paragraph_styles, template["style"]):
                    raise SLDCValidationError(f"{label}.{edge}.style is not a paragraph style")
        if "numbering" in section:
            numbering = section["numbering"]
            if not isinstance(numbering, dict):
                raise SLDCValidationError(f"{label}.numbering must be an object")
            _check_keys(numbering, {"start", "format", "show"}, f"{label}.numbering")
            if "show" in numbering and not isinstance(numbering["show"], bool):
                raise SLDCValidationError(f"{label}.numbering.show must be a boolean")
            if "start" in numbering:
                _bounded_int(numbering["start"], 1, 99_999, f"{label}.numbering.start")
            if "format" in numbering and not _has(_NUMBER_FORMATS, numbering["format"]):
                raise SLDCValidationError(f"{label}.numbering.format must be one of "
                                          f"{sorted(_NUMBER_FORMATS)}")


def _validate_page(page: Any, label: str) -> None:
    if not isinstance(page, dict):
        raise SLDCValidationError(f"{label} must be an object")
    _check_keys(page, {"size", "width_mm", "height_mm", "orientation", "margins_mm", "header_mm", "footer_mm"}, label)
    if "size" in page:
        if not _has(_PAGE_SIZES_MM, page["size"]) or "width_mm" in page or "height_mm" in page:
            raise SLDCValidationError(f"{label}.size must be one of {sorted(_PAGE_SIZES_MM)} "
                                      "and excludes width_mm/height_mm")
        width, height = _PAGE_SIZES_MM[page["size"]]
    elif "width_mm" in page and "height_mm" in page:
        width = _bounded_number(page["width_mm"], 20, 5000, f"{label}.width_mm")
        height = _bounded_number(page["height_mm"], 20, 5000, f"{label}.height_mm")
    else:
        raise SLDCValidationError(f"{label} needs a named 'size' or both width_mm and height_mm")
    if "orientation" in page:
        if page["orientation"] not in ("portrait", "landscape"):
            raise SLDCValidationError(f"{label}.orientation must be portrait or landscape")
        if page["orientation"] == "landscape":
            width, height = height, width
    margins = page.get("margins_mm", {})
    if not isinstance(margins, dict):
        raise SLDCValidationError(f"{label}.margins_mm must be an object")
    _check_keys(margins, {"top", "right", "bottom", "left"}, f"{label}.margins_mm")
    for side, value in margins.items():
        _bounded_number(value, 0, 500, f"{label}.margins_mm.{side}")
    for edge in ("header_mm", "footer_mm"):
        if edge in page:
            _bounded_number(page[edge], 0, height, f"{label}.{edge}")
    if page.get("header_mm", 0) + page.get("footer_mm", 0) > height - 20:
        raise SLDCValidationError(f"{label} header and footer leave less than 20 mm of body space")
    if width - margins.get("left", 0) - margins.get("right", 0) < 20 \
            or height - margins.get("top", 0) - margins.get("bottom", 0) < 20:
        raise SLDCValidationError(f"{label} margins leave less than 20 mm of content area")


def _validate_references(references: Any, ctx: _Context) -> None:
    if not isinstance(references, list) or len(references) > MAX_REFERENCES:
        raise SLDCValidationError(f"'references' must be a list of at most {MAX_REFERENCES} entries")
    for index, entry in enumerate(references):
        label = f"references[{index}]"
        if not isinstance(entry, dict):
            raise SLDCValidationError(f"{label} must be an object")
        reference_id = _id_field(entry.get("id"), f"{label}.id")
        if reference_id in ctx.references:
            raise SLDCValidationError(f"Duplicate reference id {reference_id!r}")
        ctx.references.add(reference_id)
        kind = entry.get("kind")
        if kind == "footnote":
            _check_keys(entry, {"id", "kind", "text"}, label)
            ctx.add_text(len(_text_field(entry, "text", label, 20_000, required=True) or ""))
        elif kind == "citation":
            _check_keys(entry, {"id", "kind", "authors", "year", "doi", "url", "accessed"}
                        | set(_CITATION_FIELDS), label)
            _text_field(entry, "title", label, _CITATION_FIELDS["title"], required=True)
            for key, limit in _CITATION_FIELDS.items():
                _text_field(entry, key, label, limit)
            if "authors" in entry:
                authors = entry["authors"]
                if not isinstance(authors, list) or len(authors) > 100 or not all(
                        isinstance(a, str) and 0 < len(a) <= 200 for a in authors):
                    raise SLDCValidationError(f"{label}.authors must be up to 100 names")
            if "year" in entry:
                _bounded_int(entry["year"], 1, 9999, f"{label}.year")
            if "doi" in entry and (not isinstance(entry["doi"], str)
                                   or not re.fullmatch(r"10\.\d{4,9}/\S{1,180}", entry["doi"])):
                raise SLDCValidationError(f"{label}.doi must look like 10.1234/abc")
            if "url" in entry:
                _check_href(entry["url"], f"{label}.url")
            if "accessed" in entry:
                _check_time(entry["accessed"], f"{label}.accessed")
        else:
            raise SLDCValidationError(f"{label}.kind must be 'citation' or 'footnote'")


def _validate_embeds(embeds: Any, ctx: _Context) -> None:
    """EMBED records are declarations. Nothing here is ever executed by this module."""
    if not isinstance(embeds, list) or len(embeds) > MAX_EMBEDS:
        raise SLDCValidationError(f"'embeds' must be a list of at most {MAX_EMBEDS} entries")
    for index, entry in enumerate(embeds):
        label = f"embeds[{index}]"
        if not isinstance(entry, dict):
            raise SLDCValidationError(f"{label} must be an object")
        _check_keys(entry, {"id", "kind", "schema_version", "params", "inputs", "output_sha256",
                            "capabilities", "policy_ref"}, label)
        embed_id = _id_field(entry.get("id"), f"{label}.id")
        if embed_id in ctx.embeds:
            raise SLDCValidationError(f"Duplicate embed id {embed_id!r}")
        ctx.embeds.add(embed_id)
        _id_field(entry.get("kind"), f"{label}.kind", _KIND)
        _bounded_int(entry.get("schema_version"), 1, 1_000_000, f"{label}.schema_version")
        params = entry.get("params", {})
        if not isinstance(params, dict) or len(_canonical_json(params)) > 64 * 1024:
            raise SLDCValidationError(f"{label}.params must be an object of at most 64 KiB")
        inputs = entry.get("inputs", [])
        if not isinstance(inputs, list) or len(inputs) > 64:
            raise SLDCValidationError(f"{label}.inputs must be a list of at most 64 entries")
        for item in inputs:
            if not isinstance(item, dict):
                raise SLDCValidationError(f"{label}.inputs entries must be objects")
            _check_keys(item, {"ref", "sha256"}, f"{label}.inputs")
            _id_field(item.get("ref"), f"{label}.inputs.ref")
            if not isinstance(item.get("sha256"), str) or not _HEX_SHA256.fullmatch(item["sha256"]):
                raise SLDCValidationError(f"{label}.inputs.sha256 must be 64 hex digits")
        if "output_sha256" in entry and (not isinstance(entry["output_sha256"], str)
                                         or not _HEX_SHA256.fullmatch(entry["output_sha256"])):
            raise SLDCValidationError(f"{label}.output_sha256 must be 64 hex digits")
        capabilities = entry.get("capabilities", [])
        if not isinstance(capabilities, list) or len(capabilities) > 32 or len(set(
                map(str, capabilities))) != len(capabilities):
            raise SLDCValidationError(f"{label}.capabilities must be up to 32 distinct names")
        for capability in capabilities:  # A request for a capability is never a grant.
            _id_field(capability, f"{label}.capabilities", _CAPABILITY)
        _text_field(entry, "policy_ref", label, 256)


def _validate_publication(publication: Any) -> None:
    if not isinstance(publication, dict):
        raise SLDCValidationError("'publication' must be an object")
    _check_keys(publication, _PUBLICATION_KEYS, "publication")
    for key, limit in (("subtitle", 500), ("publisher", 300), ("rights", 1000),
                       ("abstract", 5000), ("edition", 100)):
        _text_field(publication, key, "publication", limit)
    for key, limit, count in (("authors", 200, 64), ("keywords", 100, 64)):
        if key in publication:
            values = publication[key]
            if not isinstance(values, list) or len(values) > count or not all(
                    isinstance(v, str) and 0 < len(v) <= limit for v in values):
                raise SLDCValidationError(f"publication.{key} must be up to {count} short texts")
    if "published" in publication:
        _check_time(publication["published"], "publication.published")
    if "identifier" in publication:
        identifier = publication["identifier"]
        if not isinstance(identifier, dict):
            raise SLDCValidationError("publication.identifier must be an object")
        _check_keys(identifier, {"scheme", "value"}, "publication.identifier")
        _id_field(identifier.get("scheme"), "publication.identifier.scheme",
                  re.compile(r"^[a-z][a-z0-9-]{0,31}$"))
        _text_field(identifier, "value", "publication.identifier", 200, required=True)
    if "cover_asset" in publication:
        _id_field(publication["cover_asset"], "publication.cover_asset", _ASSET_ID)


def _validate_revision_record(record: Any, label: str, *, with_id: bool = False) -> None:
    if not isinstance(record, dict):
        raise SLDCValidationError(f"{label} must be an object")
    _check_keys(record, _REVISION_KEYS | ({"id"} if with_id else set()), label)
    if with_id and (not isinstance(record.get("id"), str) or not _HEX_SHA256.fullmatch(record["id"])):
        raise SLDCValidationError(f"{label}.id must be 64 hex digits")
    _bounded_int(record.get("number"), 1, 1_000_000_000, f"{label}.number")
    parents = record.get("parents", [])
    if not isinstance(parents, list) or len(parents) > 8 or len(set(map(str, parents))) != len(parents) \
            or not all(isinstance(p, str) and _HEX_SHA256.fullmatch(p) for p in parents):
        raise SLDCValidationError(f"{label}.parents must be up to 8 distinct revision ids")
    _check_time(record.get("created"), f"{label}.created")
    _text_field(record, "author", label, 200)
    _text_field(record, "note", label, 2000)


# SLDC 1.1 content contract (all values are data, never executable instructions):
# - Display equations use block type=equation, text="", equation={syntax,source,...}.
#   numbered=True requests a display number computed by the renderer from document order.
#   Refer to numbered equations via existing xref spans targeting the equation block id.
# - Inline objects use one U+FFFC character and a type=equation/image span covering it.
# - Image source bytes stay unchanged in Asset; crop is normalized source rectangle [0,1].
# - width_mm/height_mm describe a display bounding box (renderer preserves proportions).
# - rotation_deg is clockwise about the display box center; flip applies before rotation.
# - Floating x_mm/y_mm are from the top-left of the page content box or target block box;
#   the containing page is the page on which the figure's block resides. Square wrapping
#   uses the axis-aligned transformed bounding box; top_bottom displaces full lines.
# - The renderer is responsible for avoiding clipping, honoring layering, and pagination.
# Scientific sources are inert, editable data. UI/renderers must use a constrained renderer.
MAX_EQUATION_SOURCE = 65_536
MAX_SEQUENCE_NOTES = 1_024
_OBJECT_CHAR = "\uFFFC"  # Unicode OBJECT REPLACEMENT CHARACTER, one Python string index.
_MATH_SYNTAX = frozenset({"latex", "mhchem", "mathml", "sequence"})
_MATH_DOMAINS = frozenset({"mathematics", "chemistry", "biology"})
_MATHML_NS = "{http://www.w3.org/1998/Math/MathML}"
_MATHML_ELEMENTS = frozenset({
    "math", "mrow", "mi", "mn", "mo", "mtext", "mfrac", "msqrt", "mroot", "msub",
    "msup", "msubsup", "munder", "mover", "munderover", "mtable", "mtr", "mtd",
    "mfenced", "menclose", "mspace", "mmultiscripts", "mprescripts", "none",
})
_MATHML_ATTRS = frozenset({
    "display", "mathvariant", "mathsize", "mathcolor", "mathbackground", "displaystyle",
    "stretchy", "fence", "separator", "form", "lspace", "rspace", "minsize", "maxsize",
    "linethickness", "columnalign", "rowalign", "columnspacing", "rowspacing",
    "scriptlevel", "notation", "accent", "accentunder", "width", "height", "depth",
})
_SEQUENCE_ALPHABETS = {"dna": frozenset("ACGTRYSWKMBDHVN"),
                       "rna": frozenset("ACGURYSWKMBDHVN"),
                       "protein": frozenset("ACDEFGHIKLMNPQRSTVWYBXZJUO*")}


def _validate_mathml(source: str, label: str) -> None:
    # No DTD/entity declarations, processing instructions, comments or non-MathML elements.
    # ElementTree does not fetch remote resources; the small, closed grammar prevents embedded HTML.
    if "<!" in source or "<?" in source:
        raise SLDCValidationError(f"{label} contains forbidden XML markup")
    from xml.etree import ElementTree
    try:
        root = ElementTree.fromstring(source)
    except ElementTree.ParseError as exc:
        raise SLDCValidationError(f"{label} is not well-formed MathML: {exc}") from exc
    if root.tag != _MATHML_NS + "math":
        raise SLDCValidationError(f"{label} needs a MathML namespace <math> root")
    nodes = [(root, 1)]
    count = 0
    while nodes:
        node, depth = nodes.pop()
        count += 1
        if depth > 64 or count > 4096:
            raise SLDCValidationError(f"{label} MathML exceeds complexity budget")
        if not isinstance(node.tag, str) or not node.tag.startswith(_MATHML_NS) or \
                node.tag[len(_MATHML_NS):] not in _MATHML_ELEMENTS:
            raise SLDCValidationError(f"{label} contains an unsupported MathML element")
        for key, value in node.attrib.items():
            if key not in _MATHML_ATTRS or len(value) > 128 or \
                    any(ch in value.lower() for ch in ("url(", "javascript:", "data:", "<", ">")):
                raise SLDCValidationError(f"{label} contains an unsafe/unsupported MathML attribute")
        nodes.extend((child, depth + 1) for child in node)


def _validate_equation(equation: Any, label: str, *, inline: bool) -> None:
    if not isinstance(equation, dict):
        raise SLDCValidationError(f"{label} must be an object")
    _check_keys(equation, {"syntax", "source", "domain", "align", "numbered", "alphabet", "direction",
                           "start_index", "annotations"}, label)
    syntax = equation.get("syntax")
    if not _has(_MATH_SYNTAX, syntax):
        raise SLDCValidationError(f"{label}.syntax must be latex, mhchem, mathml or sequence")
    domain = equation.get("domain", "mathematics")
    if not _has(_MATH_DOMAINS, domain):
        raise SLDCValidationError(f"{label}.domain is invalid")
    if (syntax == "mhchem" and domain != "chemistry") or (syntax == "sequence" and domain != "biology"):
        raise SLDCValidationError(f"{label}.syntax is incompatible with domain")
    source = _text_field(equation, "source", label, MAX_EQUATION_SOURCE, required=True)
    if source is None:
        raise SLDCValidationError(f"{label} requires 'source'")
    if any(ord(ch) < 32 and ch not in "\t\r\n" for ch in source):
        raise SLDCValidationError(f"{label}.source contains control characters")
    if "align" in equation:
        if inline or not _has({"left", "center", "right"}, equation["align"]):
            raise SLDCValidationError(f"{label}.align is only for display equations")
    if "numbered" in equation:
        if inline:
            raise SLDCValidationError(f"{label}.numbered is only for display equations")
        _flag(equation["numbered"], f"{label}.numbered")
    if syntax == "mathml":
        _validate_mathml(source, label)
    if syntax == "sequence":
        _check_keys(equation, {"syntax", "source", "domain", "numbered", "alphabet", "direction", "start_index", "annotations"}, label)
        alphabet = equation.get("alphabet")
        if alphabet is None or not _has(_SEQUENCE_ALPHABETS, alphabet):
            raise SLDCValidationError(f"{label}.alphabet must be dna, rna or protein")
        if len(source) > 50_000 or any(base not in _SEQUENCE_ALPHABETS[alphabet] for base in source):
            raise SLDCValidationError(f"{label}.source has invalid sequence symbols")
        allowed_directions = {"dna": {"5to3", "3to5"}, "rna": {"5to3", "3to5"}, "protein": {"n_to_c", "c_to_n"}}
        if "direction" in equation and not _has(allowed_directions[alphabet], equation["direction"]):
            raise SLDCValidationError(f"{label}.direction does not match its alphabet")
        if "start_index" in equation:
            _bounded_int(equation["start_index"], 0, 1_000_000_000, label + ".start_index")
        annotations = equation.get("annotations", [])
        if not isinstance(annotations, list) or len(annotations) > MAX_SEQUENCE_NOTES:
            raise SLDCValidationError(f"{label}.annotations exceeds limit")
        last_end = 0
        for n, note in enumerate(annotations):
            here = f"{label}.annotations[{n}]"
            if not isinstance(note, dict):
                raise SLDCValidationError(f"{here} must be an object")
            _check_keys(note, {"start", "end", "label"}, here)
            start = _bounded_int(note.get("start"), 0, len(source)-1, here+".start")
            end = _bounded_int(note.get("end"), start+1, len(source), here+".end")
            if start < last_end:
                raise SLDCValidationError(f"{here} overlaps or is out of order")
            last_end = end
            _text_field(note, "label", here, 256, required=True)
    elif set(equation) & {"alphabet", "direction", "start_index", "annotations"}:
        raise SLDCValidationError(f"{label} sequence fields require syntax=sequence")


def _validate_spans(spans: Any, text: str, ctx: _Context, label: str) -> None:
    if not isinstance(spans, list) or len(spans) > MAX_SPANS_PER_BLOCK:
        raise SLDCValidationError(f"{label}.spans must be a list of at most {MAX_SPANS_PER_BLOCK}")
    ctx.total_spans += len(spans)
    if ctx.total_spans > MAX_SPANS:
        raise SLDCValidationError(f"Document has more than {MAX_SPANS} spans")
    previous: tuple[int, int, str] = (0, 0, "")
    occupied_objects: set[int] = set()
    for index, span in enumerate(spans):
        where = f"{label}.spans[{index}]"
        if not isinstance(span, dict):
            raise SLDCValidationError(f"{where} must be an object")
        kind, start, end = span.get("type"), span.get("start"), span.get("end")
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text):
            raise SLDCValidationError(f"{where} needs integer 0 <= start < end <= {len(text)}")
        if (start, end, str(kind)) < previous:
            raise SLDCValidationError(f"{label}.spans must be sorted by start, end, type")
        previous = (start, end, str(kind))
        base = {"type", "start", "end"}
        if kind == "format":
            _check_keys(span, base | _FORMAT_FLAGS | _INLINE_FONT_PROPS | {"color", "background"}, where)
            props = set(span) - base
            if not props:
                raise SLDCValidationError(f"{where} sets no formatting")
            for key in props & _FORMAT_FLAGS:
                _flag(span[key], f"{where}.{key}")
            for key in props & {"color", "background"}:
                if not isinstance(span[key], str) or not _COLOR.fullmatch(span[key]):
                    raise SLDCValidationError(f"{where}.{key} must be a #rrggbb colour")
            if "font_family" in span and (not isinstance(span["font_family"], str)
                                          or not _FONT_FAMILY.fullmatch(span["font_family"])):
                raise SLDCValidationError(f"{where}.font_family must be a plain font family name")
            if "font_size_pt" in span:
                _bounded_number(span["font_size_pt"], 4, 400, where + ".font_size_pt")
            for key in ("letter_spacing_pt", "word_spacing_pt"):
                if key in span:
                    low, high = _STYLE_RANGES[key]
                    _bounded_number(span[key], low, high, where + "." + key)
            if span.get("superscript") is True and span.get("subscript") is True:
                raise SLDCValidationError(f"{where} cannot be superscript and subscript")
        elif kind == "style":
            _check_keys(span, base | {"style"}, where)
            if not _has(ctx.character_styles, span.get("style")):
                raise SLDCValidationError(f"{where}.style is not a character style")
        elif kind == "link":
            _check_keys(span, base | {"href"}, where)
            _check_href(span.get("href"), f"{where}.href")
        elif kind == "ref":
            _check_keys(span, base | {"ref"}, where)
            if not _has(ctx.references, span.get("ref")):
                raise SLDCValidationError(f"{where}.ref is not a reference id")
        elif kind == "xref":
            _check_keys(span, base | {"target"}, where)
            if not _has(ctx.block_ids, span.get("target")):
                raise SLDCValidationError(f"{where}.target is not a block id")
        elif kind in ("equation", "image"):
            if end != start + 1 or text[start:end] != _OBJECT_CHAR or start in occupied_objects:
                raise SLDCValidationError(f"{where} needs its own U+FFFC placeholder")
            occupied_objects.add(start)
            if kind == "equation":
                _check_keys(span, base | {"equation"}, where)
                _validate_equation(span.get("equation"), where + ".equation", inline=True)
            else:
                _check_keys(span, base | {"figure"}, where)
                _validate_figure(span.get("figure"), where, ctx, inline=True)
        else:
            raise SLDCValidationError(f"{where} has unknown type {kind!r}")


def _validate_table(table: Any, label: str, ctx: _Context) -> None:
    if not isinstance(table, dict):
        raise SLDCValidationError(f"{label}.table must be an object")
    _check_keys(table, {"columns", "header_rows", "rows"}, f"{label}.table")
    columns = _bounded_int(table.get("columns"), 1, MAX_TABLE_COLUMNS, f"{label}.table.columns")
    rows = table.get("rows")
    if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_TABLE_ROWS:
        raise SLDCValidationError(f"{label}.table.rows must be 1 to {MAX_TABLE_ROWS} rows")
    _bounded_int(table.get("header_rows", 0), 0, len(rows), f"{label}.table.header_rows")
    carry = [0] * columns  # Rows each column is still covered by a rowspan from above.
    for r, row in enumerate(rows):
        if not isinstance(row, list):
            raise SLDCValidationError(f"{label}.table.rows[{r}] must be a list of cells")
        ctx.total_cells += len(row)
        if ctx.total_cells > MAX_TABLE_CELLS:
            raise SLDCValidationError(f"Document has more than {MAX_TABLE_CELLS} table cells")
        occupied = [c > 0 for c in carry]
        covered = list(occupied)
        spans_here: list[tuple[int, int, int]] = []
        column = 0
        for cell in row:
            where = f"{label}.table.rows[{r}]"
            if not isinstance(cell, dict):
                raise SLDCValidationError(f"{where} cells must be objects like {{'text': ...}}")
            _check_keys(cell, {"text", "colspan", "rowspan", "align", "header"}, where)
            if not isinstance(cell.get("text"), str):
                raise SLDCValidationError(f"{where} cells need string 'text'")
            ctx.add_text(len(cell["text"]))
            colspan = _bounded_int(cell.get("colspan", 1), 1, columns, f"{where}.colspan")
            rowspan = _bounded_int(cell.get("rowspan", 1), 1, len(rows) - r, f"{where}.rowspan")
            if "align" in cell and not _has(_ALIGN, cell["align"]):
                raise SLDCValidationError(f"{where}.align must be one of {sorted(_ALIGN)}")
            if "header" in cell:
                _flag(cell["header"], f"{where}.header")
            while column < columns and occupied[column]:
                column += 1
            if column + colspan > columns or any(covered[column:column + colspan]):
                raise SLDCValidationError(f"{where} has overlapping or overflowing cells")
            for c in range(column, column + colspan):
                covered[c] = True
            spans_here.append((column, colspan, rowspan))
            column += colspan
        if not all(covered):
            raise SLDCValidationError(f"{label}.table.rows[{r}] does not fill {columns} columns")
        carry = [max(value - 1, 0) for value in carry]
        for start, colspan, rowspan in spans_here:
            for c in range(start, start + colspan):
                carry[c] = rowspan - 1
    if any(carry):
        raise SLDCValidationError(f"{label}.table has a rowspan that runs past the last row")


def _validate_figure(figure: Any, label: str, ctx: _Context, *, inline: bool = False, block_id: str | None = None) -> None:
    if not isinstance(figure, dict):
        raise SLDCValidationError(f"{label}.figure must be an object")
    _check_keys(figure, {"asset", "alt", "decorative", "width_mm", "height_mm", "align",
                         "crop", "rotation_deg", "flip_horizontal", "flip_vertical", "placement", "credit", "rights"},
                f"{label}.figure")
    asset_id = _id_field(figure.get("asset"), f"{label}.figure.asset", _ASSET_ID)
    if ctx.assets is not None and asset_id not in ctx.assets:
        raise SLDCValidationError(f"{label}.figure refers to a missing asset {asset_id!r}")
    ctx.referenced_assets.add(asset_id)
    alt = _text_field(figure, "alt", f"{label}.figure", 2000, empty_ok=True) or ""
    decorative = _flag(figure["decorative"], f"{label}.figure.decorative") \
        if "decorative" in figure else False
    if decorative == bool(alt.strip()):
        raise SLDCValidationError(f"{label}.figure needs alt text, or decorative=true without it")
    for dimension in ("width_mm", "height_mm"):
        if dimension in figure:
            _bounded_number(figure[dimension], 0.1, 5000, f"{label}.figure.{dimension}")
    if "align" in figure and (inline or not _has(_ALIGN - {"justify"}, figure["align"])):
        raise SLDCValidationError(f"{label}.figure.align must be left, center or right (block only)")
    for key, maximum in (("credit", 500), ("rights", 1000)):
        _text_field(figure, key, f"{label}.figure", maximum)
    if "rotation_deg" in figure:
        _bounded_number(figure["rotation_deg"], -360, 360, f"{label}.figure.rotation_deg")
    for key in ("flip_horizontal", "flip_vertical"):
        if key in figure:
            _flag(figure[key], f"{label}.figure.{key}")
    if "crop" in figure:
        crop = figure["crop"]
        if not isinstance(crop, dict) or set(crop) != {"left", "top", "right", "bottom"}:
            raise SLDCValidationError(f"{label}.figure.crop needs normalized left/top/right/bottom")
        for key in crop:
            _bounded_number(crop[key], 0, 1, f"{label}.figure.crop.{key}")
        if crop["left"] >= crop["right"] or crop["top"] >= crop["bottom"]:
            raise SLDCValidationError(f"{label}.figure.crop must have a positive viewport")
    if "placement" in figure:
        placement = figure["placement"]
        if inline or not isinstance(placement, dict):
            raise SLDCValidationError(f"{label}.figure.placement only supports block figures")
        _check_keys(placement, {"mode", "anchor", "target", "x_mm", "y_mm", "wrap", "layer"},
                    f"{label}.figure.placement")
        if placement.get("mode") != "floating":
            raise SLDCValidationError(f"{label}.figure.placement.mode must be floating")
        if not _has({"page", "block"}, placement.get("anchor")):
            raise SLDCValidationError(f"{label}.figure.placement.anchor must be page or block")
        if placement["anchor"] == "block":
            target = _id_field(placement.get("target"), f"{label}.figure.placement.target")
            if target not in ctx.block_ids or target == block_id:
                raise SLDCValidationError(f"{label}.figure.placement.target is missing/self-referential")
        elif "target" in placement:
            raise SLDCValidationError(f"{label}.figure.placement.target requires block anchor")
        for key in ("x_mm", "y_mm"):
            _bounded_number(placement.get(key), -5000, 5000, f"{label}.figure.placement.{key}")
        if not _has({"none", "square", "top_bottom"}, placement.get("wrap", "none")):
            raise SLDCValidationError(f"{label}.figure.placement.wrap is unsupported")
        if not _has({"content", "behind_text", "in_front_of_text"},
                    placement.get("layer", "content")):
            raise SLDCValidationError(f"{label}.figure.placement.layer is unsupported")
        if placement.get("layer", "content") != "content" and placement.get("wrap", "none") != "none":
            raise SLDCValidationError(f"{label}.figure.placement text layering cannot wrap")
        if placement["anchor"] == "page" and placement.get("wrap", "none") != "none":
            raise SLDCValidationError(f"{label}.figure.placement page anchoring cannot wrap")
        if "align" in figure:
            raise SLDCValidationError(f"{label}.figure.align is incompatible with floating placement")
    ctx.add_text(len(alt))


def _validate_attributes(attributes: Any, block_type: str, label: str) -> None:
    if not isinstance(attributes, dict):
        raise SLDCValidationError(f"{label}.attributes must be an object")
    _check_keys(attributes, {"level", "language", "ordered", "depth", "align", "page_break_before",
                             "line_height", "space_before_pt", "space_after_pt", "indent_mm",
                             "first_line_indent_mm"},
                f"{label}.attributes")
    only = {"level": "heading", "language": "code", "ordered": "list_item", "depth": "list_item"}
    for key, value in attributes.items():
        if key in only and block_type != only[key]:
            raise SLDCValidationError(f"{label}.attributes.{key} applies to {only[key]} blocks only")
        where = f"{label}.attributes.{key}"
        if key == "level":
            _bounded_int(value, 1, 6, where)
        elif key == "depth":
            _bounded_int(value, 0, 32, where)
        elif key == "language":
            if not isinstance(value, str) or not _CODE_LANGUAGE.fullmatch(value):
                raise SLDCValidationError(f"{where} must be a short token without spaces")
        elif key == "align":
            if value not in _ALIGN:
                raise SLDCValidationError(f"{where} must be one of {sorted(_ALIGN)}")
        elif key in _STYLE_RANGES:
            low, high = _STYLE_RANGES[key]
            _bounded_number(value, low, high, where)
        else:
            _flag(value, where)


def _validate_block_v1(block: Any, index: int, ctx: _Context, seen: set[str]) -> None:
    label = f"Block #{index}"
    if not isinstance(block, dict):
        raise SLDCValidationError(f"{label} must be an object")
    _check_keys(block, _BLOCK_KEYS_V1, label.lower())
    block_id = _id_field(block.get("id"), f"{label} id")
    if block_id in seen:
        raise SLDCValidationError(f"Invalid or duplicate block id: {block_id!r}")
    seen.add(block_id)
    block_type = block.get("type")
    if not isinstance(block_type, str) or block_type not in BLOCK_TYPES_V1:
        raise SLDCValidationError(f"Invalid block type in block #{index}")
    text = block.get("text")
    if not isinstance(text, str):
        raise SLDCValidationError(f"Block #{index} requires string 'text'")
    ctx.add_text(len(text))
    for key in ("table", "figure", "embed", "equation"):
        if (key in block) != (block_type == key):
            raise SLDCValidationError(f"{label}: {key!r} belongs to {key} blocks, and they need it")
    if block_type == "equation" and text:
        raise SLDCValidationError(f"{label} equation block text must be empty; source is authoritative")
    if "spans" in block:
        if block_type in ("code", "equation", "figure", "table", "embed") and any(
                span.get("type") in ("equation", "image") for span in block["spans"] if isinstance(span, dict)):
            raise SLDCValidationError(f"{label} inline objects require a text-flow block")
        _validate_spans(block["spans"], text, ctx, label)
    if "source_ref" in block and (not isinstance(block["source_ref"], str) or len(block["source_ref"]) > _MAX_REF_CHARS):
        raise SLDCValidationError(f"{label} source_ref must be text <= {_MAX_REF_CHARS} characters")
    if "attributes" in block:
        _validate_attributes(block["attributes"], block_type, label)
    if "style" in block and not _has(ctx.paragraph_styles, block["style"]):
        raise SLDCValidationError(f"{label} style is not a paragraph style")
    if "section" in block and not _has(ctx.sections, block["section"]):
        raise SLDCValidationError(f"{label} section is not a section id")
    if block_type == "table":
        _validate_table(block["table"], label, ctx)
    elif block_type == "figure":
        _validate_figure(block["figure"], label, ctx, block_id=block_id)
    elif block_type == "equation":
        _validate_equation(block["equation"], label + ".equation", inline=False)
    elif block_type == "embed" and not _has(ctx.embeds, block["embed"]):
        raise SLDCValidationError(f"{label} embed is not an embed id")
    if "extensions" in block:
        _check_extensions(block["extensions"], label)
        toc = block["extensions"].get(_TOC_EXTENSION)
        if toc is not None:
            _validate_toc_extension(toc, block, label, ctx)


def _validate_toc_extension(toc: Any, block: dict[str, Any], label: str, ctx: _Context) -> None:
    """Validate the optional SLDC 1.1 TOC annotation without treating its cache as authoritative."""
    if not isinstance(toc, dict):
        raise SLDCValidationError(f"{label} TOC data must be an object")
    _check_keys(toc, {"id", "role", "depth", "heading", "level", "preset", "overrides", "source_text",
                      "show_page_numbers", "leader_dots", "refresh_mode"}, label + ".toc")
    toc_id = toc.get("id")
    if not isinstance(toc_id, str) or not re.fullmatch(r"[0-9a-f]{24}", toc_id):
        raise SLDCValidationError(f"{label} TOC id must be 24 lowercase hexadecimal characters")
    role = toc.get("role")
    if role == "title":
        if block["type"] != "paragraph" or "heading" in toc:
            raise SLDCValidationError(f"{label} TOC title must be a paragraph")
        _bounded_int(toc.get("depth"), 1, 6, label + ".toc.depth")
        if toc.get("preset", "standard") not in ("standard", "compact", "academic"):
            raise SLDCValidationError(f"{label} TOC preset is invalid")
        for option in ("show_page_numbers", "leader_dots"):
            if option in toc and not isinstance(toc[option], bool):
                raise SLDCValidationError(f"{label} TOC {option} must be a boolean")
        if toc.get("refresh_mode", "automatic") not in ("automatic", "manual"):
            raise SLDCValidationError(f"{label} TOC refresh mode is invalid")
        overrides = toc.get("overrides", {})
        if not isinstance(overrides, dict) or len(overrides) > 5000:
            raise SLDCValidationError(f"{label} TOC overrides must be an object")
        for heading, override in overrides.items():
            if not _has(ctx.block_ids, heading) or not isinstance(override, dict):
                raise SLDCValidationError(f"{label} TOC override has invalid heading")
            _check_keys(override, {"label", "hidden"}, label + ".toc.overrides")
            if "label" in override and (not isinstance(override["label"], str) or len(override["label"]) > 500):
                raise SLDCValidationError(f"{label} TOC label override is invalid")
            if "hidden" in override and not isinstance(override["hidden"], bool):
                raise SLDCValidationError(f"{label} TOC hidden override must be a boolean")
    elif role == "entry":
        if block["type"] not in ("list_item", "paragraph") or "depth" in toc:
            raise SLDCValidationError(f"{label} TOC entry must be a paragraph or legacy list item")
        if "level" in toc:
            _bounded_int(toc["level"], 1, 6, label + ".toc.level")
        if "source_text" in toc and (not isinstance(toc["source_text"], str) or len(toc["source_text"]) > 500):
            raise SLDCValidationError(f"{label} TOC source text is invalid")
        if not _has(ctx.block_ids, toc.get("heading")):
            raise SLDCValidationError(f"{label} TOC entry refers to an unknown heading")
    else:
        raise SLDCValidationError(f"{label} TOC role must be 'title' or 'entry'")


def _validate_v1(document: dict[str, Any], assets: Mapping[str, Any] | None = None) -> set[str]:
    """Validate a 1.x document; return the asset ids it references.

    With ``assets=None`` the existence of referenced assets is not checked (use
    :func:`validate_package` for that).
    """
    _check_keys(document, _DOCUMENT_KEYS_V1, "document")
    _check_json_tree(document)
    if document.get("schema") != SCHEMA:
        raise SLDCValidationError(f"Unsupported document schema {document.get('schema')!r}")
    _id_field(document.get("id"), "Document id")
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

    ctx = _Context(assets)
    if "publication" in document:
        _validate_publication(document["publication"])
    if "styles" in document:
        _validate_styles(document["styles"], ctx)
    if "sections" in document:
        _validate_sections(document["sections"], ctx)
    if "references" in document:
        _validate_references(document["references"], ctx)
    if "embeds" in document:
        _validate_embeds(document["embeds"], ctx)
    if "revision" in document:
        _validate_revision_record(document["revision"], "revision")
    history = document.get("history", [])
    if not isinstance(history, list) or len(history) > MAX_HISTORY:
        raise SLDCValidationError(f"'history' must be a list of at most {MAX_HISTORY} entries")
    for index, record in enumerate(history):
        _validate_revision_record(record, f"history[{index}]", with_id=True)
    if "extensions" in document:
        _check_extensions(document["extensions"], "document")

    for block in blocks:  # Pass 1: ids, so spans can point forward.
        if isinstance(block, dict) and isinstance(block.get("id"), str):
            ctx.block_ids.add(block["id"])
    seen: set[str] = set()
    current_section = 0
    for index, block in enumerate(blocks):
        _validate_block_v1(block, index, ctx, seen)
        if "section" in block:
            position = ctx.sections[block["section"]]
            if position < current_section:
                raise SLDCValidationError(f"Block #{index} returns to an earlier section; "
                                          "each section must be contiguous")
            current_section = position
    toc_titles: dict[str, int] = {}
    toc_entries: set[str] = set()
    heading_ids = {b["id"] for b in blocks if b["type"] == "heading"}
    for block in blocks:
        toc = block.get("extensions", {}).get(_TOC_EXTENSION)
        if not toc:
            continue
        if toc["role"] == "title":
            toc_titles[toc["id"]] = toc_titles.get(toc["id"], 0) + 1
        else:
            toc_entries.add(toc["id"])
            if toc["heading"] not in heading_ids:
                raise SLDCValidationError("A TOC entry must reference a semantic heading")
    if any(count != 1 for count in toc_titles.values()) or toc_entries - toc_titles.keys():
        raise SLDCValidationError("A managed TOC needs exactly one title for every entry group")
    cover = document.get("publication", {}).get("cover_asset")
    if cover is not None:
        if assets is not None and cover not in assets:
            raise SLDCValidationError(f"publication.cover_asset refers to a missing asset {cover!r}")
        ctx.referenced_assets.add(cover)
    return ctx.referenced_assets


@_guarded
def _prepare_v1(document: dict[str, Any],
                assets: Mapping[str, Any] | None = None) -> tuple[bytes, set[str]]:
    referenced = _validate_v1(document, assets)
    payload = _canonical_json(document)
    if len(payload) > MAX_DOCUMENT_BYTES:
        raise SLDCValidationError("Serialized document exceeds 16 MiB")
    return payload, referenced


def _prepare(document: Mapping[str, Any]) -> bytes:
    """Validate ``document`` (0.1.0, or 1.x when it has a ``schema`` key) and return its canonical JSON payload, serialised once."""
    if not isinstance(document, Mapping):
        raise SLDCValidationError("Document must be a mapping")
    plain = document if isinstance(document, dict) else dict(document)
    if "schema" in plain:
        return _prepare_v1(plain)[0]
    _validate_structure(plain)
    payload = _canonical_json(plain)
    if len(payload) > MAX_DOCUMENT_BYTES:
        raise SLDCValidationError("Serialized document exceeds 16 MiB")
    return payload


def _prepare_legacy(document: Mapping[str, Any]) -> bytes:
    if isinstance(document, Mapping) and "schema" in document:
        raise SLDCUnsupportedError(
            "This is a 1.x document; write it with dumps_package()/write_package(), or export_legacy() for a lossy 0.1.0 copy")
    return _prepare(document)


def _document_of(value: Any) -> Mapping[str, Any]:
    return value.document if isinstance(value, SLDCPackage) else value


def validate_sldc(document: Mapping[str, Any] | SLDCPackage) -> None:
    """Validate a document (version chosen by its ``schema`` key); never execute anything in it.

    Pass a :class:`SLDCPackage` to validate its assets, comments and signature records too.
    """
    if isinstance(document, SLDCPackage):
        validate_package(document)
    else:
        _prepare(document)


def block_text(block: Mapping[str, Any]) -> str:
    """The plain text of one block, for search, prompts and chunking.

    A table adds its caption (``text``) and then its cells, tab-separated, one row per line.
    Figure alt text and image data are not text and are not included.
    """
    text = block["text"]
    table = block.get("table")
    if block.get("type") == "table" and isinstance(table, Mapping):
        rows = "\n".join("\t".join(cell["text"] for cell in row) for row in table["rows"])
        return f"{text}\n{rows}" if text and rows else (text or rows)
    if block.get("type") == "equation":
        return block["equation"]["source"]
    objects = {span["start"]: span for span in block.get("spans", ())
               if span["type"] in ("equation", "image")}
    if objects:
        return "".join((objects[pos]["equation"]["source"] if objects[pos]["type"] == "equation"
                        else "[Image: " + objects[pos]["figure"].get("alt", "") + "]")
                       if pos in objects else char for pos, char in enumerate(text))
    return text


def extract_text(document: Mapping[str, Any] | SLDCPackage, *, validate: bool = True) -> str:
    """Create plain text from a SLDC without inventing missing content.

    Pass ``validate=False`` only for a document that was just read with ``read_package``,
    ``read_sldc`` or ``loads_*`` (all of which already validate).
    """
    if validate:
        validate_sldc(document)
    return "\n\n".join(block_text(block) for block in _document_of(document)["blocks"])


# ----------------------------------------------------------------------------------------------
# Container: ZIP primitives shared by every version
# ----------------------------------------------------------------------------------------------
_PART_NAME = re.compile(
    r"^(?:document\.json|comments\.json|assets/[A-Za-z0-9][A-Za-z0-9_-]{0,63}\.(?:png|jpg|gif|webp)"
    r"|signatures/[A-Za-z0-9][A-Za-z0-9_-]{0,63}\.json)$"
)
_ASSET_MEDIA = {"png": "image/png", "jpg": "image/jpeg", "gif": "image/gif", "webp": "image/webp"}
_MEDIA_EXTENSION = {media: ext for ext, media in _ASSET_MEDIA.items()}
_MANIFEST_KEYS = frozenset({"format", "version", "profile", "requires", "document_id", "revision_id", "parts"})
_PART_KEYS = frozenset({"name", "media_type", "role", "size", "sha256"})


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


def _check_zip(archive: zipfile.ZipFile, *, entry_ceiling: int) -> list[zipfile.ZipInfo]:
    """Version-independent structure rules; the member list and sizes are checked per version."""
    if archive.comment:
        raise SLDCValidationError("SLDC archives must not carry a ZIP comment")
    infos = archive.infolist()
    if not 2 <= len(infos) <= MAX_PARTS + 2:
        raise SLDCValidationError("SLDC archive has an unsupported number of entries")
    names = [info.filename for info in infos]
    if len({name.lower() for name in names}) != len(names):
        raise SLDCValidationError("SLDC archive has duplicate or ambiguous entry names")
    if infos[0].compress_type != zipfile.ZIP_STORED:
        raise SLDCValidationError("The mimetype entry must be uncompressed")
    offsets = [info.header_offset for info in infos]
    if offsets[0] != 0 or offsets != sorted(set(offsets)):
        raise SLDCValidationError("ZIP entries must start at offset 0 and not overlap")
    for info in infos:
        if info.is_dir() or info.flag_bits & 0x41:  # bit 0 encrypted, bit 6 strong encryption
            raise SLDCValidationError("Directories and encrypted ZIP entries are unsupported")
        mode = (info.external_attr >> 16) & 0o170000
        if mode not in (0, stat.S_IFREG):
            raise SLDCValidationError("Symlinks and nonregular ZIP entries are prohibited")
        if info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
            raise SLDCValidationError("Unsupported ZIP compression method")
        if info.file_size > entry_ceiling:
            raise SLDCValidationError(f"SLDC entry {info.filename!r} exceeds size limit")
        if (info.file_size > _RATIO_CHECK_MIN_BYTES
                and info.file_size / max(1, info.compress_size) > MAX_COMPRESSION_RATIO):
            raise SLDCValidationError("SLDC archive exceeds compression-ratio limit")
    return infos


def _read_member(archive: zipfile.ZipFile, info: zipfile.ZipInfo, limit: int) -> bytes:
    """Read one entry, never trusting the declared size; the ZIP CRC is verified at EOF."""
    with archive.open(info) as handle:
        raw = handle.read(limit + 1)
    if len(raw) > limit or len(raw) != info.file_size:
        raise SLDCIntegrityError(f"Entry {info.filename!r} length does not match its header")
    return raw


def _parse_version(value: Any) -> tuple[int, int, int]:
    match = re.fullmatch(r"(\d{1,4})\.(\d{1,5})\.(\d{1,6})", value) if isinstance(value, str) else None
    if match is None:
        raise SLDCValidationError(f"SLDC manifest version must look like 1.0.0, got {value!r}")
    return int(match[1]), int(match[2]), int(match[3])


def newest_version(*versions: str) -> str:
    """Choose the numerically newest SemVer-like SLDC version; never compare text.

    This is used for merge compatibility and must not silently downgrade newer
    1.x content to FORMAT_VERSION. The caller still validates write support.
    """
    if not versions:
        raise SLDCValidationError("newest_version needs at least one version")
    return max(versions, key=_parse_version)


def _part_role(name: str) -> str:
    if name == "document.json":
        return "content"
    if name == "comments.json":
        return "annotation"
    return "asset" if name.startswith("assets/") else "signature"


def _part_media_type(name: str) -> str:
    return _ASSET_MEDIA[name.rsplit(".", 1)[1]] if name.startswith("assets/") else "application/json"


def _part_limit(name: str) -> int:
    return {"content": MAX_DOCUMENT_BYTES, "annotation": MAX_COMMENTS_BYTES,
            "asset": MAX_ASSET_BYTES, "signature": MAX_SIGNATURE_BYTES}[_part_role(name)]


def _part_order(name: str) -> tuple[int, str]:
    return ({"content": 0, "annotation": 1, "asset": 2, "signature": 3}[_part_role(name)], name)


def _revision_digest(entries: Iterable[tuple[str, int, str]]) -> str:
    """Revision id: a digest over the content and asset parts (name, size, sha256).

    Comments and signatures are deliberately outside it, so adding a signature or a review
    comment does not change the revision that was signed.
    """
    descriptor = [{"name": n, "sha256": h, "size": s} for n, s, h in sorted(entries)]
    return _sha256(b"SLDC-REVISION-V1\0" + _canonical_json(descriptor))


@dataclass
class _Loaded:
    kind: str  # "legacy" (0.1.0), "package" (1.x) or "encrypted" (1.x envelope)
    manifest: dict[str, Any]
    parts: dict[str, bytes]


def _read_legacy(archive: zipfile.ZipFile, infos: list[zipfile.ZipInfo], manifest: dict[str, Any],
                 size: int, read_parts: bool) -> _Loaded:
    if [i.filename for i in infos] != list(MEMBERS):
        raise SLDCValidationError("SLDC v0.1 requires only mimetype, manifest.json, document.json, in order")
    if size > MAX_ARCHIVE_BYTES:
        raise SLDCValidationError("SLDC must be no larger than 25 MiB")
    limits = (len(_MIMETYPE_BYTES), MAX_MANIFEST_BYTES, MAX_DOCUMENT_BYTES)
    if any(info.file_size > limit for info, limit in zip(infos, limits)):
        raise SLDCValidationError("SLDC entry exceeds size limit")
    if sum(info.file_size for info in infos) > MAX_UNCOMPRESSED_BYTES:
        raise SLDCValidationError("SLDC uncompressed data exceeds size limit")
    if not read_parts:
        return _Loaded("legacy", manifest, {})
    payload = _read_member(archive, infos[2], MAX_DOCUMENT_BYTES)
    expected_size = manifest.get("document_size_bytes")
    expected_hash = manifest.get("document_sha256")
    if type(expected_size) is not int or expected_size != len(payload):
        raise SLDCIntegrityError("Document byte count does not match manifest")
    if not isinstance(expected_hash, str) or expected_hash != _sha256(payload):
        raise SLDCIntegrityError("Document checksum does not match manifest")
    return _Loaded("legacy", manifest, {"document.json": payload})


def _check_manifest_v1(manifest: dict[str, Any], *, check_requires: bool) -> list[dict[str, Any]]:
    """Validate the structure of a 1.x package manifest and return its part inventory."""
    extra = {key for key in manifest if key not in _MANIFEST_KEYS and not key.startswith("x-")}
    if extra:
        raise SLDCValidationError(f"Unsupported manifest keys: {sorted(extra)!r}")
    if manifest.get("profile") != "publishing":
        raise SLDCUnsupportedError(f"Unsupported SLDC profile {manifest.get('profile')!r}")
    requires = manifest.get("requires")
    if not isinstance(requires, list) or not all(isinstance(r, str) for r in requires) \
            or len(set(requires)) != len(requires):
        raise SLDCValidationError("manifest 'requires' must be a list of distinct feature names")
    unknown = sorted(set(requires) - SUPPORTED_FEATURES)
    if unknown and check_requires:
        raise SLDCUnsupportedError(f"This document requires features this reader does not understand: {unknown}. Open it read-only or with a newer reader; do not save it with this one.")
    _id_field(manifest.get("document_id"), "manifest document_id")
    if not isinstance(manifest.get("revision_id"), str) or not _HEX_SHA256.fullmatch(manifest["revision_id"]):
        raise SLDCValidationError("manifest revision_id must be 64 hex digits")
    parts = manifest.get("parts")
    if not isinstance(parts, list) or not 1 <= len(parts) <= MAX_PARTS:
        raise SLDCValidationError("manifest 'parts' must list the archive's parts")
    names: list[str] = []
    for entry in parts:
        if not isinstance(entry, dict) or set(entry) != _PART_KEYS:
            raise SLDCValidationError(f"Each manifest part needs exactly {sorted(_PART_KEYS)}")
        name = entry["name"]
        if not isinstance(name, str) or not _PART_NAME.fullmatch(name):
            raise SLDCValidationError(f"Unsupported part name {name!r}")
        if entry["role"] != _part_role(name) or entry["media_type"] != _part_media_type(name):
            raise SLDCValidationError(f"Part {name!r} has the wrong role or media type")
        if type(entry["size"]) is not int or not 0 <= entry["size"] <= _part_limit(name):
            raise SLDCValidationError(f"Part {name!r} size is invalid or exceeds its limit")
        if not isinstance(entry["sha256"], str) or not _HEX_SHA256.fullmatch(entry["sha256"]):
            raise SLDCValidationError(f"Part {name!r} sha256 must be 64 hex digits")
        names.append(name)
    if len({n.lower() for n in names}) != len(names) or names != sorted(names, key=_part_order):
        raise SLDCValidationError("Manifest parts must be unique and in canonical order")
    if names[0] != "document.json":
        raise SLDCValidationError("A 1.x package requires document.json")
    if sum(_part_role(n) == "asset" for n in names) > MAX_ASSETS \
            or sum(_part_role(n) == "signature" for n in names) > MAX_SIGNATURES:
        raise SLDCValidationError("Package has too many assets or signatures")
    if sum(e["size"] for e in parts) > MAX_PACKAGE_UNCOMPRESSED_BYTES \
            or sum(e["size"] for e in parts if e["role"] == "asset") > MAX_TOTAL_ASSET_BYTES:
        raise SLDCValidationError("Package parts exceed the size budget")
    return parts


def _read_package_parts(archive: zipfile.ZipFile, infos: list[zipfile.ZipInfo],
                        manifest: dict[str, Any], parts: list[dict[str, Any]]) -> _Loaded:
    if [i.filename for i in infos] != ["mimetype", "manifest.json"] + [p["name"] for p in parts]:
        raise SLDCValidationError("Archive entries do not match the manifest part list")
    contents: dict[str, bytes] = {}
    for info, entry in zip(infos[2:], parts):
        raw = _read_member(archive, info, _part_limit(entry["name"]))
        if len(raw) != entry["size"] or _sha256(raw) != entry["sha256"]:
            raise SLDCIntegrityError(f"Part {entry['name']!r} does not match its manifest entry")
        contents[entry["name"]] = raw
    revision = _revision_digest((e["name"], e["size"], e["sha256"]) for e in parts
                                if e["role"] in ("content", "asset"))
    if manifest["revision_id"] != revision:
        raise SLDCIntegrityError("Manifest revision_id does not match the content parts")
    return _Loaded("package", manifest, contents)


def _load_container(data: bytes, *, read_parts: bool = True,
                    check_requires: bool = True) -> _Loaded:
    """Check container structure, manifest and checksums of any supported version.

    Returns the verified raw parts without interpreting them. ``read_parts=False`` stops after the
    manifest (used to inspect a file cheaply); ``check_requires=False`` then also reports files
    that need features this reader lacks instead of refusing them.
    """
    if len(data) > MAX_PACKAGE_BYTES:
        raise SLDCValidationError("SLDC must be no larger than 48 MiB")
    if not _has_sldc_header(data[:_SNIFF_BYTES]):
        raise SLDCValidationError("Not an SLAI SLDC (missing mimetype header)")
    try:
        with zipfile.ZipFile(io.BytesIO(data), "r") as archive:
            infos = _check_zip(archive, entry_ceiling=MAX_PACKAGE_BYTES)
            if _read_member(archive, infos[0], len(_MIMETYPE_BYTES)) != _MIMETYPE_BYTES:
                raise SLDCValidationError("Not an SLAI SLDC (mimetype mismatch)")
            if infos[1].filename != "manifest.json":
                raise SLDCValidationError("The second ZIP entry must be manifest.json")
            manifest = _decode_json(_read_member(archive, infos[1], MAX_MANIFEST_BYTES_V1), "manifest")
            if manifest.get("format") != FORMAT_ID:
                raise SLDCValidationError("Not an SLAI SLDC (manifest format mismatch)")
            version = manifest.get("version")
            if version == LEGACY_VERSION:
                if infos[1].file_size > MAX_MANIFEST_BYTES:
                    raise SLDCValidationError("SLDC entry 'manifest.json' exceeds size limit")
                return _read_legacy(archive, infos, manifest, len(data), read_parts)
            major = _parse_version(version)[0]
            if major != 1:
                raise SLDCUnsupportedError(
                    f"Unsupported SLDC version {version!r}; this reader supports "
                    f"{LEGACY_VERSION} and 1.x")
            if manifest.get("container") == "encrypted":
                return _read_encrypted(archive, infos, manifest, read_parts)
            if "container" in manifest:
                raise SLDCUnsupportedError(f"Unsupported container {manifest['container']!r}")
            parts = _check_manifest_v1(manifest, check_requires=check_requires)
            if not read_parts:
                if [i.filename for i in infos] != ["mimetype", "manifest.json"] + [p["name"] for p in parts]:
                    raise SLDCValidationError("Archive entries do not match the manifest part list")
                return _Loaded("package", manifest, {})
            return _read_package_parts(archive, infos, manifest, parts)
    except SLDCError:
        raise
    except (zipfile.BadZipFile, zlib.error, EOFError, OSError, RuntimeError,
            NotImplementedError, OverflowError, struct.error) as exc:
        raise SLDCIntegrityError(f"SLDC archive cannot be read: {exc}") from exc


def _unpack(data: bytes) -> bytes:
    """Check a 0.1.0 archive and return its raw document payload (other versions are refused)."""
    if len(data) > MAX_ARCHIVE_BYTES:
        raise SLDCValidationError("SLDC must be no larger than 25 MiB")
    loaded = _load_container(data)
    if loaded.kind != "legacy":
        raise SLDCUnsupportedError(
            f"This is an SLDC {loaded.manifest.get('version')} {loaded.kind} file; read it with "
            "read_package()/loads_package(), or migrate it explicitly")
    return loaded.parts["document.json"]


def _write_zip(manifest: Mapping[str, Any], members: Iterable[tuple[str, bytes, int]]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", allowZip64=False) as archive:
        archive.writestr(_zipinfo(MEMBERS[0], zipfile.ZIP_STORED), _MIMETYPE_BYTES)
        archive.writestr(_zipinfo(MEMBERS[1], zipfile.ZIP_DEFLATED), _canonical_json(manifest),
                         compresslevel=_COMPRESS_LEVEL)
        for name, raw, method in members:
            if method == zipfile.ZIP_STORED:
                archive.writestr(_zipinfo(name, method), raw)
            else:
                archive.writestr(_zipinfo(name, method), raw, compresslevel=_COMPRESS_LEVEL)
    return buffer.getvalue()


# ----------------------------------------------------------------------------------------------
# Container, version 0.1.0 (frozen): documents only
# ----------------------------------------------------------------------------------------------
def dumps_sldc(document: Mapping[str, Any]) -> bytes:
    """Serialise a 0.1.0 ``document`` to a deterministic archive in memory.

    This is the frozen text-only profile. Use :func:`dumps_package` for the 1.x publishing
    format. The result is re-checked with the reader's own container rules, so a document that
    would be rejected on read (for example, text so repetitive that it breaches the
    compression-ratio limit) fails here instead of producing an unreadable file.
    """
    payload = _prepare_legacy(document)
    manifest = {
        "format": FORMAT_ID,
        "version": LEGACY_VERSION,
        "document_sha256": _sha256(payload),
        "document_size_bytes": len(payload),
    }
    data = _write_zip(manifest, [(MEMBERS[2], payload, zipfile.ZIP_DEFLATED)])
    if len(data) > MAX_ARCHIVE_BYTES:
        raise SLDCValidationError("SLDC archive exceeds 25 MiB")
    _unpack(data)
    return data


def loads_sldc(data: bytes | bytearray | memoryview) -> dict[str, Any]:
    """Read and fully validate a 0.1.0 archive held in memory (DB blob, network, cache).

    Files of any other version raise :class:`SLDCUnsupportedError`: this function returns only a
    document, so it must never silently drop the assets, comments or signatures of a 1.x file.
    """
    payload = _unpack(bytes(data))
    document = _decode_json(payload, "document")
    _validate_structure(document)
    return document


# ----------------------------------------------------------------------------------------------
# Images and assets
# ----------------------------------------------------------------------------------------------
def _png_info(data: bytes) -> tuple[int, int]:
    if len(data) < 33 or data[12:16] != b"IHDR" or struct.unpack(">I", data[8:12])[0] != 13:
        raise SLDCValidationError("PNG image has no valid IHDR header")
    return struct.unpack(">II", data[16:24])


_JPEG_SOF = frozenset({0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF})


def _jpeg_info(data: bytes) -> tuple[int, int]:
    position, length = 2, len(data)
    while position + 4 <= length:
        if data[position] != 0xFF:
            raise SLDCValidationError("JPEG image has a malformed segment")
        marker = data[position + 1]
        if marker == 0xFF:  # Fill byte.
            position += 1
            continue
        if marker in (0x01, 0xD8) or 0xD0 <= marker <= 0xD7:  # Standalone markers.
            position += 2
            continue
        if marker in (0xD9, 0xDA):  # End of image or start of scan before any frame header.
            break
        size = struct.unpack(">H", data[position + 2:position + 4])[0]
        if size < 2:
            break
        if marker in _JPEG_SOF:
            if size < 8 or position + 9 > length:
                break
            height, width = struct.unpack(">HH", data[position + 5:position + 9])
            return width, height
        position += 2 + size
    raise SLDCValidationError("JPEG image has no frame header")


def _gif_info(data: bytes) -> tuple[int, int]:
    if len(data) < 10:
        raise SLDCValidationError("GIF image is truncated")
    return struct.unpack("<HH", data[6:10])


def _webp_info(data: bytes) -> tuple[int, int]:
    if len(data) < 30:
        raise SLDCValidationError("WebP image is truncated")
    chunk = data[12:16]
    if chunk == b"VP8 " and data[23:26] == b"\x9d\x01\x2a":
        width, height = struct.unpack("<HH", data[26:30])
        return width & 0x3FFF, height & 0x3FFF
    if chunk == b"VP8L" and data[20] == 0x2F:
        bits = struct.unpack("<I", data[21:25])[0]
        return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
    if chunk == b"VP8X":
        return (int.from_bytes(data[24:27], "little") + 1,
                int.from_bytes(data[27:30], "little") + 1)
    raise SLDCValidationError("WebP image has no recognised header")


def _image_info(data: bytes) -> tuple[str, int, int]:
    """Identify an image from its header and return ``(media_type, width, height)``.

    Only PNG, JPEG, GIF and WebP are accepted, checked by structure and size, never decoded here.
    SVG (and everything else) is refused: SVG is XML that can carry script and external links.
    """
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        media, size = "image/png", _png_info(data)
    elif data[:3] == b"\xff\xd8\xff":
        media, size = "image/jpeg", _jpeg_info(data)
    elif data[:6] in (b"GIF87a", b"GIF89a"):
        media, size = "image/gif", _gif_info(data)
    elif data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        media, size = "image/webp", _webp_info(data)
    else:
        raise SLDCValidationError("Unsupported image: only PNG, JPEG, GIF and WebP are accepted")
    width, height = size
    if not (1 <= width <= MAX_IMAGE_SIDE and 1 <= height <= MAX_IMAGE_SIDE) \
            or width * height > MAX_IMAGE_PIXELS:
        raise SLDCValidationError(
            f"Image dimensions {width}x{height} exceed the limit "
            f"({MAX_IMAGE_SIDE} per side, {MAX_IMAGE_PIXELS:,} pixels)")
    return media, width, height


def make_asset(data: bytes | bytearray | memoryview, *, asset_id: str | None = None) -> Asset:
    """Wrap image bytes as an :class:`Asset`; the media type is detected from the bytes.

    Without ``asset_id`` the id is derived from the content (``img-`` + 16 hex digits), so the
    same image always gets the same id and identical documents serialise identically.
    """
    raw = bytes(data)
    media_type, _, _ = _image_info(raw)
    identifier = asset_id or "img-" + _sha256(raw)[:16]
    _id_field(identifier, "Asset id", _ASSET_ID)
    return Asset(identifier, media_type, raw)


def _check_assets(assets: Mapping[str, Asset], referenced: set[str]) -> None:
    if not isinstance(assets, Mapping) or len(assets) > MAX_ASSETS:
        raise SLDCValidationError(f"A package holds at most {MAX_ASSETS} assets")
    if len({key.lower() for key in assets if isinstance(key, str)}) != len(assets):
        raise SLDCValidationError("Asset ids must differ by more than letter case")
    total_bytes = total_pixels = 0
    for key, asset in assets.items():
        if not isinstance(asset, Asset) or asset.id != key:
            raise SLDCValidationError("Each asset must be an Asset stored under its own id")
        _id_field(asset.id, "Asset id", _ASSET_ID)
        if not isinstance(asset.data, (bytes, bytearray)) or len(asset.data) > MAX_ASSET_BYTES:
            raise SLDCValidationError(f"Asset {asset.id!r} must be at most {MAX_ASSET_BYTES} bytes")
        media_type, width, height = _image_info(bytes(asset.data))
        if asset.media_type != media_type:
            raise SLDCValidationError(
                f"Asset {asset.id!r} is declared {asset.media_type!r} but is {media_type!r}")
        total_bytes += len(asset.data)
        total_pixels += width * height
    if total_bytes > MAX_TOTAL_ASSET_BYTES or total_pixels > MAX_TOTAL_IMAGE_PIXELS:
        raise SLDCValidationError("Assets exceed the total size or pixel budget")
    orphans = sorted(set(assets) - referenced)
    if orphans:  # Unreferenced data would travel invisibly with the document.
        raise SLDCValidationError(f"Assets that no figure or cover uses: {orphans}; "
                                  "remove them with prune_assets()")


def add_asset(package: SLDCPackage, asset: Asset) -> SLDCPackage:
    """Return a copy of ``package`` that also holds ``asset`` (same id and bytes is a no-op)."""
    existing = package.assets.get(asset.id)
    if existing is not None and existing.data != asset.data:
        raise SLDCValidationError(f"Asset id {asset.id!r} is already used by different content")
    result = package.copy()
    result.assets[asset.id] = asset
    return result


def prune_assets(package: SLDCPackage) -> SLDCPackage:
    """Remove only assets not referenced by figures, inline images or the cover."""
    if package.version == LEGACY_VERSION:
        return package.copy()
    referenced = _validate_v1(package.document)  # References only, no requirement assets exist.
    result = package.copy()
    result.assets = {key: asset for key, asset in package.assets.items() if key in referenced}
    return result


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



# ----------------------------------------------------------------------------------------------
# Files: atomic writes and bounded reads
# ----------------------------------------------------------------------------------------------
def _destination(path: str | os.PathLike[str]) -> Path:
    destination = Path(path)
    if destination.suffix.lower() not in EXTENSIONS:
        raise SLDCValidationError(f"Expected .sldc path, got {destination.name!r}")
    if not destination.parent.is_dir():
        raise SLDCValidationError("Destination directory must already exist")
    return destination


def _write_bytes(destination: Path, data: bytes, *, overwrite: bool) -> Path:
    """Write ``data`` to a temporary file in the destination directory, flush it to disk and only
    then publish it, so readers never observe a partial file."""
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


def _read_file_bytes(path: str | os.PathLike[str], limit: int) -> bytes:
    source = Path(path)
    if not stat.S_ISREG(os.stat(source).st_mode):  # OSError (e.g. FileNotFoundError) propagates
        raise SLDCValidationError("SLDC must be a regular file")
    with open(source, "rb") as handle:
        size = os.fstat(handle.fileno()).st_size
        if size > limit:
            raise SLDCValidationError(f"SLDC must be no larger than {limit // (1024 * 1024)} MiB")
        # Allocate for the real size, not the limit; still bounded if the file grows meanwhile.
        return handle.read(size + 1)


def write_sldc(path: str | os.PathLike[str], document: Mapping[str, Any],
               *, overwrite: bool = False) -> Path:
    """Atomically write a deterministic SLDC 0.1.0 container (see :func:`write_package` for 1.x).

    The archive is built and verified in memory first. With ``overwrite=False`` an existing path
    is never replaced (even concurrently) and ``FileExistsError`` is raised. The parent directory
    must already exist. Files are created with owner-only permissions (0600 on POSIX).
    """
    destination = _destination(path)
    return _write_bytes(destination, dumps_sldc(document), overwrite=overwrite)


def read_sldc(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read and validate a 0.1.0 container regardless of its filename suffix.

    Other versions raise :class:`SLDCUnsupportedError`; use :func:`read_package` for them.
    """
    return loads_sldc(_read_file_bytes(path, MAX_ARCHIVE_BYTES))


def sniff_sldc(path: str | os.PathLike[str]) -> bool:
    """Cheap check of the leading bytes only (no parsing, no integrity check), any version.

    Use for routing or file-type detection; use ``is_sldc`` or ``read_package`` to trust it.
    """
    try:
        with open(path, "rb") as handle:
            return _has_sldc_header(handle.read(_SNIFF_BYTES))
    except OSError:
        return False


def is_sldc(path: str | os.PathLike[str], *, strict: bool = True) -> bool:
    """Return True only for a fully validated, unencrypted SLDC of any supported version.

    With ``strict=False`` this is the cheap :func:`sniff_sldc` check instead. An encrypted file
    cannot be validated without its key, so it is not "valid" here; use :func:`inspect_package`.
    """
    if not strict:
        return sniff_sldc(path)
    try:
        read_package(path)
    except (SLDCError, OSError, ValueError):
        return False
    return True


# ----------------------------------------------------------------------------------------------
# Packages (1.x): serialising and reading
# ----------------------------------------------------------------------------------------------
def _required_features(document: Mapping[str, Any]) -> list[str]:
    """Advertise every feature an older reader would otherwise silently discard."""
    by_type = {"table": "tables", "figure": "figures", "embed": "embeds", "equation": "scientific"}
    needed = {by_type[b["type"]] for b in document["blocks"] if b["type"] in by_type}
    if document.get("embeds"):
        needed.add("embeds")
    for section in document.get("sections", ()):
        if {"header_mm", "footer_mm"}.intersection(section.get("page", {})):
            needed.add("page-bands")
        if section.get("role") or "show" in section.get("numbering", {}):
            needed.add("cover-pages")
    extra_figure_keys = {"height_mm", "crop", "rotation_deg", "flip_horizontal", "flip_vertical", "placement", "credit", "rights"}
    for block in document["blocks"]:
        if {"line_height", "space_before_pt", "space_after_pt", "indent_mm", "first_line_indent_mm"}.intersection(block.get("attributes", {})):
            needed.add("paragraph-layout")
        figure = block.get("figure", {})
        if isinstance(figure, Mapping) and extra_figure_keys.intersection(figure):
            needed.add("image-layout")
        for span in block.get("spans", ()):
            if span["type"] == "format" and _INLINE_FONT_PROPS.intersection(span):
                needed.add("font-runs")
            if span["type"] == "equation":
                needed.add("scientific")
            elif span["type"] == "image":
                needed.update(("inline-media", "figures"))
                if extra_figure_keys.intersection(span["figure"]):
                    needed.add("image-layout")
        if _TOC_EXTENSION in block.get("extensions", {}):
            needed.add("native-toc")
    return sorted(needed)


@_guarded
def _build_parts(package: SLDCPackage, *, _read_only: bool = False) -> dict[str, bytes]:
    """Validate a 1.x package completely and return its serialised parts in canonical order."""
    major, minor, _ = _parse_version(package.version)
    if major != 1 or (minor > 1 and not _read_only):
        raise SLDCUnsupportedError(f"Cannot write SLDC {package.version!r} with this version")
    if not isinstance(package.document, Mapping):
        raise SLDCValidationError("Package document must be a mapping")
    plain = package.document if isinstance(package.document, dict) else dict(package.document)
    payload, referenced = _prepare_v1(plain, package.assets)
    features = set(_required_features(plain))
    if _parse_version(package.version)[1] == 0 and features - REQUIRED_FEATURES:
        raise SLDCUnsupportedError("SLDC 1.1 features require an explicit upgrade of package.version")
    _check_assets(package.assets, referenced)
    if not isinstance(package.comments, list) or not isinstance(package.signatures, list):
        raise SLDCValidationError("Package comments and signatures must be lists")
    parts: dict[str, bytes] = {"document.json": payload}
    if package.comments:
        parts["comments.json"] = _comments_part(package.comments)
    for asset_id, asset in package.assets.items():
        parts[f"assets/{asset_id}.{_MEDIA_EXTENSION[asset.media_type]}"] = bytes(asset.data)
    if len(package.signatures) > MAX_SIGNATURES:
        raise SLDCValidationError(f"A package holds at most {MAX_SIGNATURES} signatures")
    seen: set[str] = set()
    for record in package.signatures:
        _validate_signature_record(record)
        if record["id"] in seen:
            raise SLDCValidationError(f"Duplicate signature id {record['id']!r}")
        seen.add(record["id"])
        raw = _canonical_json(record)
        if len(raw) > MAX_SIGNATURE_BYTES:
            raise SLDCValidationError("Signature record is too large")
        parts[f"signatures/{record['id']}.json"] = raw
    if sum(len(raw) for raw in parts.values()) > MAX_PACKAGE_UNCOMPRESSED_BYTES:
        raise SLDCValidationError("Package parts exceed the size budget")
    return dict(sorted(parts.items(), key=lambda item: _part_order(item[0])))


def _inventory(parts: Mapping[str, bytes]) -> list[dict[str, Any]]:
    return [{"name": name, "media_type": _part_media_type(name), "role": _part_role(name),
             "size": len(raw), "sha256": _sha256(raw)} for name, raw in parts.items()]


def _plain_archive(package: SLDCPackage) -> bytes:
    if package.version == LEGACY_VERSION:
        if package.assets or package.comments or package.signatures:
            raise SLDCUnsupportedError("Version 0.1.0 holds only a document; migrate_package() before adding assets, comments or signatures")
        return dumps_sldc(package.document)
    parts = _build_parts(package)
    entries = _inventory(parts)
    manifest = {
        "format": FORMAT_ID, "version": package.version, "profile": "publishing",
        "requires": _required_features(package.document),
        "document_id": package.document["id"],
        "revision_id": _revision_digest((e["name"], e["size"], e["sha256"]) for e in entries
                                        if e["role"] in ("content", "asset")),
        "parts": entries,
    }
    members = [(name, raw, zipfile.ZIP_STORED if name.startswith("assets/") else zipfile.ZIP_DEFLATED)
               for name, raw in parts.items()]
    data = _write_zip(manifest, members)
    if len(data) > MAX_PACKAGE_BYTES:
        raise SLDCValidationError("SLDC package exceeds 48 MiB")
    _load_container(data)  # Never emit a file this module's own reader would refuse.
    return data


def dumps_package(package: SLDCPackage, *, recipients: Sequence[Recipient] | None = None,
                  recovery_policy: str = "none") -> bytes:
    """Serialise a package to a deterministic SLDC archive in memory.

    A 1.x package becomes the publishing format; a 0.1.0 package is written unchanged (and must
    then hold only a document). With ``recipients`` the archive is wrapped in an authenticated
    encryption envelope (see :func:`password_recipient`, :func:`x25519_recipient`); that output
    is intentionally not deterministic, because nonces and wrapping keys are random.
    """
    plain = _plain_archive(package)
    if recipients is None:
        return plain
    return _encrypt(plain, recipients, recovery_policy)


def revision_id(package: SLDCPackage) -> str:
    """Identity of the exact content of a package: a SHA-256 over its document and assets.

    Comments and signatures are not part of it. For a 0.1.0 package this is its :func:`document_digest`.
    """
    if package.version == LEGACY_VERSION:
        return document_digest(package.document)
    parts = _build_parts(package)
    return _revision_digest((e["name"], e["size"], e["sha256"]) for e in _inventory(parts)
                            if e["role"] in ("content", "asset"))


@_guarded
def _package_from_loaded(loaded: _Loaded) -> SLDCPackage:
    if loaded.kind == "legacy":
        document = _decode_json(loaded.parts["document.json"], "document")
        _validate_structure(document)
        return SLDCPackage(document, version=LEGACY_VERSION)
    parts, manifest = loaded.parts, loaded.manifest
    document = _decode_json(parts["document.json"], "document")
    if document.get("id") != manifest["document_id"]:
        raise SLDCIntegrityError("Manifest document_id does not match the document")
    assets: dict[str, Asset] = {}
    comments: list[dict[str, Any]] = []
    signatures: list[dict[str, Any]] = []
    for name, raw in parts.items():
        if name.startswith("assets/"):
            asset_id, extension = name[len("assets/"):].rsplit(".", 1)
            media_type, _, _ = _image_info(raw)
            if media_type != _ASSET_MEDIA[extension]:
                raise SLDCValidationError(f"Part {name!r} is not a {_ASSET_MEDIA[extension]} image")
            assets[asset_id] = Asset(asset_id, media_type, raw)
        elif name == "comments.json":
            comments = _comments_from_part(_decode_json(raw, "comments"))
        elif name.startswith("signatures/"):
            record = _decode_json(raw, "signature")
            if name != f"signatures/{record.get('id')}.json":
                raise SLDCValidationError(f"Part {name!r} does not match its signature id")
            signatures.append(record)
    package = SLDCPackage(document, assets, comments, signatures, manifest["version"])
    _build_parts(package, _read_only=True)  # Validate compatible newer minor editions.
    if not set(_required_features(document)) <= set(manifest["requires"]):
        raise SLDCValidationError("Manifest 'requires' omits a feature the document uses")
    return package


def loads_package(data: bytes | bytearray | memoryview, *, password: str | None = None,
                  private_key: Any = None) -> SLDCPackage:
    """Read and fully validate an SLDC of any supported version held in memory.

    Encrypted files need ``password`` or the recipient's X25519 ``private_key`` (raw 32 bytes or a
    ``cryptography`` key object); without a working credential :class:`SLDCDecryptionError` is
    raised. Opening never executes or fetches anything the document contains.
    """
    loaded = _load_container(bytes(data))
    if loaded.kind == "encrypted":
        loaded = _load_container(_decrypt(loaded, password=password, private_key=private_key))
        if loaded.kind == "encrypted":
            raise SLDCValidationError("Nested encryption is not supported")
    return _package_from_loaded(loaded)


def read_package(path: str | os.PathLike[str], *, password: str | None = None,
                 private_key: Any = None) -> SLDCPackage:
    """Read and validate an SLDC file of any supported version, regardless of its suffix."""
    return loads_package(_read_file_bytes(path, MAX_PACKAGE_BYTES), password=password,
                         private_key=private_key)


def write_package(path: str | os.PathLike[str], package: SLDCPackage, *, overwrite: bool = False,
                  recipients: Sequence[Recipient] | None = None,
                  recovery_policy: str = "none") -> Path:
    """Atomically write a package (optionally encrypted) with the guarantees of :func:`write_sldc`."""
    destination = _destination(path)
    data = dumps_package(package, recipients=recipients, recovery_policy=recovery_policy)
    return _write_bytes(destination, data, overwrite=overwrite)


def validate_package(package: SLDCPackage) -> None:
    """Validate everything a package holds without writing it. Raises :class:`SLDCError`."""
    if package.version == LEGACY_VERSION:
        _plain_archive(package)
    else:
        _build_parts(package)


def inspect_package(source: bytes | bytearray | memoryview | str | os.PathLike[str]) -> dict[str, Any]:
    """Describe a file from its manifest alone, without decompressing, decrypting or trusting it.

    Returns ``container`` (``legacy``, ``package`` or ``encrypted``), ``version``, ``supported``,
    ``newer_than_known`` (whether this module knows the declared 1.x minor/patch edition),
    ``requires``, ``unsupported_requires`` (features this reader lacks: open read-only, never save),
    ``document_id``, ``revision_id``, ``parts`` and, for encrypted files, ``encryption`` with the
    recipient list. Raises :class:`SLDCError` if the container itself is malformed.
    """
    data = bytes(source) if isinstance(source, (bytes, bytearray, memoryview)) \
        else _read_file_bytes(source, MAX_PACKAGE_BYTES)
    loaded = _load_container(data, read_parts=False, check_requires=False)
    manifest = loaded.manifest
    report: dict[str, Any] = {"container": loaded.kind, "format": FORMAT_ID,
                              "version": manifest["version"],
                              "newer_than_known": _parse_version(manifest["version"]) > _parse_version(LATEST_KNOWN_VERSION)
                                  if loaded.kind != "legacy" else False,
                              "supported": True, "requires": [],
                              "unsupported_requires": [], "document_id": None,
                              "revision_id": None, "parts": []}
    if loaded.kind == "legacy":
        report["parts"] = [{"name": "document.json", "role": "content",
                            "media_type": "application/json",
                            "size": manifest.get("document_size_bytes"),
                            "sha256": manifest.get("document_sha256")}]
    elif loaded.kind == "package":
        unknown = sorted(set(manifest["requires"]) - SUPPORTED_FEATURES)
        report.update(requires=list(manifest["requires"]), unsupported_requires=unknown,
                      supported=not unknown, document_id=manifest["document_id"],
                      revision_id=manifest["revision_id"],
                      parts=[dict(entry) for entry in manifest["parts"]])
    else:
        report["encryption"] = _describe_encryption(manifest)
    return report


# ----------------------------------------------------------------------------------------------
# Forward-compatible, validated, data-only text preview
# ----------------------------------------------------------------------------------------------
@_guarded
def preview_package(source: bytes | bytearray | memoryview | str | os.PathLike[str]) -> dict[str, Any]:
    """Inspect an intact SLDC archive without interpreting unknown features as editable.

    Unlike a normal read, this path deliberately permits future minor versions
    and unknown required-feature names. It still validates the ZIP inventory,
    CRC, per-member SHA-256, revision identity and canonical known data budgets
    before extracting *only* plain strings from the document.

    The returned preview is inert display data. It never retains the original
    document dictionary, assets, signatures, executable content or permissions.
    """
    data = (bytes(source) if isinstance(source, (bytes, bytearray, memoryview))
            else _read_file_bytes(source, MAX_PACKAGE_BYTES))
    loaded = _load_container(data, check_requires=False)
    if loaded.kind == "encrypted":
        raise SLDCUnsupportedError("Encrypted SLDC requires authorized decryption before preview")
    if loaded.kind == "legacy":
        document = _decode_json(loaded.parts["document.json"], "document")
        _validate_structure(document)
        content = [{"type": block["type"], "text": block["text"]}
                   for block in document["blocks"]]
        title = document["title"]
        return {"title": title, "version": LEGACY_VERSION, "complete": True,
                "text": "\n\n".join(item["text"] for item in content),
                "blocks": content, "skipped": 0, "skipped_blocks": 0,
                "unsupported_requires": [], "newer_than_known": False,
                "read_only": True}

    if loaded.kind != "package":
        raise SLDCUnsupportedError("SLDC container cannot be previewed as plaintext")
    document = _decode_json(loaded.parts["document.json"], "document")
    _check_json_tree(document)
    # Source-byte and decompression budgets are enforced by _load_container;
    # these additional limits constrain the potentially hostile preview shape.
    source_blocks = document.get("blocks")
    if not isinstance(source_blocks, list):
        raise SLDCValidationError("Preview document requires a blocks array")
    if len(source_blocks) > MAX_BLOCKS:
        raise SLDCValidationError("Preview document exceeds maximum block count")
    title = document.get("title") if isinstance(document.get("title"), str) else ""
    title = title[:2048]
    unknown_features = sorted(set(loaded.manifest["requires"]) - SUPPORTED_FEATURES)
    future_version = _parse_version(loaded.manifest["version"]) > _parse_version(LATEST_KNOWN_VERSION)
    known_document_keys = {
        "schema", "id", "title", "language", "blocks", "publication", "metadata",
        "provenance", "styles", "sections", "references", "embeds", "history",
        "revision", "extensions",
    }
    known_block_keys = {
        "id", "type", "text", "source_ref", "attributes", "annotations",
        "spans", "style", "section", "table", "figure", "equation", "embed",
        "extensions",
    }
    skipped = len(set(document) - known_document_keys)
    preview_blocks: list[dict[str, str]] = []
    budget = MAX_TEXT_CHARS
    for block in source_blocks:
        if not isinstance(block, dict):
            skipped += 1
            continue
        kind = block.get("type")
        raw_text = block.get("text")
        text = raw_text if isinstance(raw_text, str) else ""
        if not isinstance(raw_text, str):
            skipped += 1
        if not isinstance(kind, str) or kind not in BLOCK_TYPES_V1:
            skipped += 1
            kind = "paragraph"
        skipped += len(set(block) - known_block_keys)
        if kind == "equation":
            equation = block.get("equation")
            if isinstance(equation, dict) and isinstance(equation.get("source"), str):
                text = equation["source"]
            elif not text:
                skipped += 1
        if kind == "table":
            table = block.get("table")
            if isinstance(table, dict) and isinstance(table.get("rows"), list):
                rows: list[str] = []
                for row in table["rows"][:MAX_TABLE_ROWS]:
                    if not isinstance(row, list):
                        skipped += 1
                        continue
                    cells: list[str] = []
                    for cell in row[:MAX_TABLE_COLUMNS]:
                        if isinstance(cell, dict) and isinstance(cell.get("text"), str):
                            cells.append(cell["text"])
                        else:
                            skipped += 1
                    if cells:
                        rows.append("\t".join(cells))
                if rows:
                    text = (text + "\n" if text else "") + "\n".join(rows)
            else:
                skipped += 1
        if kind == "figure":
            figure = block.get("figure")
            if isinstance(figure, dict) and isinstance(figure.get("alt"), str):
                text = (text + "\n" if text else "") + figure["alt"]
        if len(text) > budget:
            text = text[:budget]
            skipped += 1
        budget -= len(text)
        if text:
            preview_blocks.append({"type": "paragraph", "text": text})
        else:
            skipped += 1
    # "complete" must agree with the normal reader: if this module can
    # fully read and validate a known edition, the preview is complete even
    # when a non-textual known block has no displayable text. Conversely, a
    # malformed known edition cannot be called complete merely because it
    # happens to contain some strings.
    complete = False
    if not unknown_features and not future_version:
        try:
            _package_from_loaded(loaded)
        except SLDCError:
            complete = False
        else:
            complete = True
    return {
        "title": title, "version": loaded.manifest["version"],
        "complete": complete, "text": "\n\n".join(item["text"] for item in preview_blocks),
        "blocks": preview_blocks, "skipped": skipped, "skipped_blocks": skipped,
        "unsupported_requires": unknown_features,
        "newer_than_known": future_version, "read_only": True,
    }


@_guarded
def preview_to_package(preview: Mapping[str, Any]) -> SLDCPackage:
    """Build a *fresh*, text-only 1.0.0 package from inert preview strings.

    This explicitly discards all source ids, styles, non-text blocks, assets,
    comments, signatures, extensions, provenance and future-version features.
    It never mutates or saves the source document.
    """
    if not isinstance(preview, Mapping):
        raise SLDCValidationError("preview must be an object")
    title = preview.get("title")
    if not isinstance(title, str) or not title.strip():
        title = "SLDC text preview"
    if len(title) > 2048:
        title = title[:2048]
    source_blocks = preview.get("blocks")
    if not isinstance(source_blocks, list) or len(source_blocks) > MAX_BLOCKS:
        raise SLDCValidationError("preview must contain a bounded blocks array")
    new_blocks: list[dict[str, Any]] = []
    total = 0
    for item in source_blocks:
        if not isinstance(item, Mapping) or not isinstance(item.get("text"), str):
            raise SLDCValidationError("preview blocks must be text records")
        raw = item["text"]
        total += len(raw)
        if total > MAX_TEXT_CHARS:
            raise SLDCValidationError("preview text exceeds permitted limit")
        if raw:
            new_blocks.append(make_block("paragraph", raw))
    if not new_blocks:
        new_blocks.append(make_block("paragraph", ""))
    package = new_package(title, new_blocks)
    validate_package(package)
    return package


# ----------------------------------------------------------------------------------------------
# Building packages (1.x)
# ----------------------------------------------------------------------------------------------
def new_block_id() -> str:
    """A fresh random block id. Use it for blocks added to existing documents.

    Sequential ids (``b00001``) are only safe inside one document; two people who each add
    ``b00042`` offline would collide when their copies are merged.
    """
    return "b" + uuid.uuid4().hex[:16]


def new_package(title: str, blocks: Iterable[Mapping[str, Any]] = (), *, language: str = "en",
                doc_id: str | None = None, publication: Mapping[str, Any] | None = None,
                metadata: Mapping[str, Any] | None = None,
                provenance: Iterable[Mapping[str, Any]] | None = None,
                styles: Mapping[str, Any] | None = None,
                sections: Iterable[Mapping[str, Any]] | None = None,
                references: Iterable[Mapping[str, Any]] | None = None,
                embeds: Iterable[Mapping[str, Any]] | None = None,
                assets: Iterable[Asset] = (),
                extensions: Mapping[str, Any] | None = None) -> SLDCPackage:
    """Assemble a 1.0.0 package; missing block ids become ``b00001``, ``b00002``, ...

    The result is not validated until you write it or call :func:`validate_package`.
    """
    document: dict[str, Any] = {
        "schema": SCHEMA,
        "id": doc_id or f"doc-{uuid.uuid4().hex}",
        "title": title,
        "language": language,
        "blocks": _assign_missing_ids(blocks),
    }
    for key, value in (("publication", publication), ("metadata", metadata),
                       ("styles", styles), ("extensions", extensions)):
        if value:
            document[key] = copy.deepcopy(dict(value))
    for key, listing in (("provenance", provenance), ("sections", sections),
                         ("references", references), ("embeds", embeds)):
        if listing:
            document[key] = [copy.deepcopy(dict(item)) for item in listing]
    return SLDCPackage(document, {asset.id: asset for asset in assets})


def make_table(rows: Sequence[Sequence[str | Mapping[str, Any]]], *, header_rows: int = 0,
               caption: str = "", columns: int | None = None,
               block_id: str | None = None) -> dict[str, Any]:
    """Create a table block. Cells are strings or ``{"text", "colspan", "rowspan", ...}`` objects.

    ``columns`` defaults to the widest row (sum of its colspans); give it explicitly when cells
    use ``rowspan``. The caption is the block's ``text``.
    """
    normalised = [[{"text": cell} if isinstance(cell, str) else dict(cell) for cell in row]
                  for row in rows]
    if columns is None:
        columns = max((sum(int(c.get("colspan", 1)) if type(c.get("colspan", 1)) is int else 1
                           for c in row) for row in normalised), default=1) or 1
    return make_block("table", caption, block_id=block_id,
                      table={"columns": columns, "header_rows": header_rows, "rows": normalised})


def make_figure(asset: Asset | str, alt: str = "", *, caption: str = "", decorative: bool = False,
                width_mm: float | None = None, align: str | None = None,
                block_id: str | None = None, height_mm: float | None = None,
                crop: Mapping[str, float] | None = None, rotation_deg: float | None = None,
                flip_horizontal: bool = False, flip_vertical: bool = False,
                placement: Mapping[str, Any] | None = None, credit: str | None = None,
                rights: str | None = None) -> dict[str, Any]:
    """Create a figure block that shows ``asset`` (an :class:`Asset` or its id).

    ``alt`` describes the image for readers who cannot see it and is required unless
    ``decorative=True``. The caption is the block's ``text``.
    """
    figure: dict[str, Any] = {"asset": asset.id if isinstance(asset, Asset) else asset, "alt": alt}
    if decorative:
        figure["decorative"] = True
    if width_mm is not None:
        figure["width_mm"] = width_mm
    if align:
        figure["align"] = align
    if height_mm is not None:
        figure["height_mm"] = height_mm
    if crop is not None:
        figure["crop"] = dict(crop)
    if rotation_deg is not None:
        figure["rotation_deg"] = rotation_deg
    if flip_horizontal:
        figure["flip_horizontal"] = True
    if flip_vertical:
        figure["flip_vertical"] = True
    if placement is not None:
        figure["placement"] = dict(placement)
    if credit is not None:
        figure["credit"] = credit
    if rights is not None:
        figure["rights"] = rights
    return make_block("figure", caption, block_id=block_id, figure=figure)


def make_equation(source: str, *, syntax: str = "latex", domain: str = "mathematics",
                  align: str | None = None, block_id: str | None = None,
                  numbered: bool = False, **sequence_fields: Any) -> dict[str, Any]:
    """Create an editable display equation. Source is authoritative (never a bitmap)."""
    equation = {"syntax": syntax, "source": source, "domain": domain, **sequence_fields}
    if align is not None:
        equation["align"] = align
    if numbered:
        equation["numbered"] = True
    _validate_equation(equation, "equation", inline=False)
    return make_block("equation", "", block_id=block_id, equation=equation)


def make_inline_equation(source: str, start: int, *, syntax: str = "latex",
                         domain: str = "mathematics", **sequence_fields: Any) -> dict[str, Any]:
    """Create an inline equation span; caller inserts U+FFFC at ``text[start]``."""
    _bounded_int(start, 0, MAX_TEXT_CHARS - 1, "inline equation start")
    equation = {"syntax": syntax, "source": source, "domain": domain, **sequence_fields}
    _validate_equation(equation, "equation", inline=True)
    return {"type": "equation", "start": start, "end": start + 1, "equation": equation}


def make_inline_image(asset: Asset | str, start: int, alt: str = "", *, decorative: bool = False,
                      width_mm: float | None = None, height_mm: float | None = None) -> dict[str, Any]:
    """Create an inline image span; caller inserts U+FFFC at ``text[start]``."""
    _bounded_int(start, 0, MAX_TEXT_CHARS - 1, "inline image start")
    figure: dict[str, Any] = {"asset": asset.id if isinstance(asset, Asset) else asset, "alt": alt}
    if decorative:
        figure["decorative"] = True
    if width_mm is not None:
        figure["width_mm"] = width_mm
    if height_mm is not None:
        figure["height_mm"] = height_mm
    return {"type": "image", "start": start, "end": start + 1, "figure": figure}


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



def _sorted_spans(spans: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = [dict(span) for span in spans]
    try:
        result.sort(key=lambda s: (s["start"], s["end"], s["type"]))
    except (KeyError, TypeError):
        pass  # Leave it as given; validation reports what is wrong.
    return result


def make_block(block_type: str, text: str, *, block_id: str | None = None,
               source_ref: str | None = None, attributes: Mapping[str, Any] | None = None,
               annotations: Sequence[Mapping[str, Any]] | None = None,
               spans: Iterable[Mapping[str, Any]] | None = None, style: str | None = None,
               section: str | None = None, table: Mapping[str, Any] | None = None,
               figure: Mapping[str, Any] | None = None, equation: Mapping[str, Any] | None = None,
               embed: str | None = None,
               extensions: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Create one block. Without ``block_id`` the id is left for the document builder to assign.

    ``annotations`` belongs to 0.1.0 documents; the remaining keywords (``spans``, ``style``,
    ``section``, ``table``, ``figure``, ``embed``, ``extensions``) to 1.x. Spans are sorted into
    the canonical order. See :func:`make_table` and :func:`make_figure` for the usual shortcuts.
    """
    block: dict[str, Any] = {"type": block_type, "text": text}
    if block_id is not None:
        block["id"] = block_id
    if source_ref:
        block["source_ref"] = source_ref
    if attributes:
        block["attributes"] = dict(attributes)
    if annotations:
        block["annotations"] = [dict(a) for a in annotations]
    if spans:
        block["spans"] = _sorted_spans(spans)
    for key, value in (("style", style), ("section", section), ("embed", embed)):
        if value:
            block[key] = value
    for key, mapping in (("table", table), ("figure", figure), ("equation", equation), ("extensions", extensions)):
        if mapping is not None and (mapping or key != "extensions"):
            block[key] = copy.deepcopy(dict(mapping))
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
    """Build a document from plain text or Markdown (see :func:`blocks_from_text`).

    Unless ``doc_id`` is given, the id is derived from the title and blocks, so identical input
    always yields the identical document (and the identical archive bytes).
    """
    blocks = blocks_from_text(text, markdown=markdown, source_ref=source_ref)
    if doc_id is None:
        fingerprint = hashlib.sha256(_canonical_json({"title": title, "blocks": blocks}))
        doc_id = "doc-" + fingerprint.hexdigest()[:32]
    return new_document(
        title, blocks, language=language, doc_id=doc_id, metadata=metadata, provenance=provenance,
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
    assert meta is not None
    parsed_at = meta.get("parsed_at")  # The Reader's own parse time keeps the entry reproducible.
    entry = provenance_entry(
        "reader.parse",
        timestamp=parsed_at if isinstance(parsed_at, str) and parsed_at else None,
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
# Using documents inside SLAI (both versions)
# ----------------------------------------------------------------------------------------------
def document_digest(document: Mapping[str, Any] | SLDCPackage) -> str:
    """SHA-256 (hex) of the canonical document payload; equals the manifest checksum in 0.1.0.

    A stable identity for provenance records, caches and de-duplication. For 1.x also see
    :func:`revision_id`, which covers the assets as well.
    """
    return _sha256(_prepare(_document_of(document)))


def block_digest(block: Mapping[str, Any], *, full: bool = False) -> str:
    """SHA-256 (hex) of a block, independent of its position.

    By default only the block's ``type`` and plain text (see :func:`block_text`) count, and not
    its id, so an answer that cites a block can be re-checked against the exact text it was based
    on. Styles, spans, attributes and source references are then *not* covered: use
    ``full=True`` (the whole block, id included) to detect any edit, and for layout or signing.
    """
    if full:
        if not isinstance(block, Mapping):
            raise SLDCValidationError("Block must be a mapping")
        return _sha256(_canonical_json(dict(block)))
    if not isinstance(block.get("type"), str) or not isinstance(block.get("text"), str):
        raise SLDCValidationError("Block requires string 'type' and 'text'")
    try:
        text = block_text(block)
    except (KeyError, TypeError) as exc:
        raise SLDCValidationError("Block table is malformed") from exc
    return _sha256(_canonical_json({"type": block["type"], "text": text}))


def block_index(document: Mapping[str, Any] | SLDCPackage, *,
                validate: bool = True) -> dict[str, dict[str, Any]]:
    """Map block id -> block, for resolving citations (``chunk['block_ids']``) back to text."""
    if validate:
        validate_sldc(document)
    return {block["id"]: block for block in _document_of(document)["blocks"]}


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


def chunk_document(document: Mapping[str, Any] | SLDCPackage, max_chars: int = 2000, *,
                   validate: bool = True) -> list[dict[str, Any]]:
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
            chunks.append({"index": len(chunks), "text": "\n\n".join(parts), "block_ids": list(ids),
                           "partial": False})
        ids.clear()
        parts.clear()
        size = 0

    for block in _document_of(document)["blocks"]:
        text = block_text(block).strip()
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


# --- Markdown export ---------------------------------------------------------------------------
_MARKDOWN_SPAN_LIMIT = 1_000  # Blocks with more spans than this render as plain text.


def _wrap_markdown(segment: str, spans: Sequence[Mapping[str, Any]]) -> str:
    if not segment.strip():
        return segment
    lead = segment[:len(segment) - len(segment.lstrip())]
    trail = segment[len(segment.rstrip()):]
    core = segment.strip()
    for span in spans:
        if span["type"] != "format":
            continue
        for flag, marker in (("code", "`"), ("strike", "~~"), ("italic", "*"), ("bold", "**")):
            if span.get(flag) is True:
                core = f"{marker}{core}{marker}"
    for span in spans:
        if span["type"] == "link":
            core = f"[{core}]({span['href'].replace('(', '%28').replace(')', '%29')})"
    return lead + core + trail


def _inline_markdown(text: str, spans: Sequence[Mapping[str, Any]] | None,
                     references: Mapping[str, Mapping[str, Any]],
                     notes: dict[str, int], image_href: Callable[[str], str]) -> str:
    """Render spans as Markdown: bold, italic, strike, code, links, footnote and citation marks.

    Underline, colour, super/subscript and named styles have no Markdown form and are dropped
    (the text is kept); cross-references render as plain text.
    """
    if not spans or len(spans) > _MARKDOWN_SPAN_LIMIT:
        return text
    cuts = sorted({0, len(text)} | {s["start"] for s in spans} | {s["end"] for s in spans})
    pieces: list[tuple[tuple[int, ...], str]] = []
    for start, end in itertools.pairwise(cuts):
        active = tuple(i for i, s in enumerate(spans) if s["start"] <= start and end <= s["end"])
        if pieces and pieces[-1][0] == active:
            pieces[-1] = (active, pieces[-1][1] + text[start:end])
        else:
            pieces.append((active, text[start:end]))
    result: list[str] = []
    offset = 0
    for active, segment in pieces:
        media = [spans[i] for i in active if spans[i]["type"] in ("equation", "image")]
        if media:
            obj = media[0]
            if obj["type"] == "equation":
                eq = obj["equation"]
                rendered = (eq["source"] if eq["syntax"] == "latex" else "\\ce{" + eq["source"] + "}" if eq["syntax"] == "mhchem" else None)
                result.append("$" + rendered + "$" if rendered is not None else
                              "`" + eq["source"].replace("`", "") + "`")
            else:
                fig = obj["figure"]
                alt = fig.get("alt", "").replace("[", "\\[").replace("]", "\\]")
                result.append(f"![{alt}]({image_href(fig['asset'])})")
        else:
            result.append(_wrap_markdown(segment, [spans[i] for i in active]))
        offset += len(segment)
        for i in active:  # The mark goes after the last piece a reference span covers.
            span = spans[i]
            if span["type"] == "ref" and span["end"] == offset:
                reference = references.get(span["ref"])
                if reference and reference["kind"] == "footnote":
                    result.append(f"[^{notes.setdefault(span['ref'], len(notes) + 1)}]")
                elif reference:
                    result.append(f"[@{span['ref']}]")
    return "".join(result)


def _table_markdown(table: Mapping[str, Any]) -> str:
    """A GitHub-style pipe table. Spanned cells keep their text in the first cell."""
    columns, rows = table["columns"], table["rows"]
    grid = [[""] * columns for _ in rows]
    busy: set[tuple[int, int]] = set()
    for r, row in enumerate(rows):
        column = 0
        for cell in row:
            while (r, column) in busy:
                column += 1
            grid[r][column] = cell["text"].replace("|", "\\|").replace("\n", " ")
            for rr in range(r, r + cell.get("rowspan", 1)):
                for cc in range(column, column + cell.get("colspan", 1)):
                    busy.add((rr, cc))
            column += cell.get("colspan", 1)
    header, body = (grid[0], grid[1:]) if table.get("header_rows", 0) >= 1 else ([""] * columns, grid)
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(" --- " for _ in header) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in body]
    return "\n".join(lines)


def to_markdown(document: Mapping[str, Any] | SLDCPackage, *, validate: bool = True,
                asset_href: Callable[[str], str] | None = None) -> str:
    """Render a document as Markdown for display or LLM prompts (uses conventional attributes).

    1.x documents also render inline formatting, links, footnotes, citations (``[@id]`` plus a
    short reference list), tables and figures. Figures become ``![alt](href)``; ``asset_href``
    maps an asset id to the href to use (default ``assets/<id>[.ext]``). Layout, styles and
    sections have no Markdown equivalent and are not rendered; this export is for reading, not
    for round-tripping (use a package to keep everything).
    """
    if validate:
        validate_sldc(document)
    package = document if isinstance(document, SLDCPackage) else None
    doc = _document_of(document)
    references = {r["id"]: r for r in doc.get("references", [])}
    notes: dict[str, int] = {}
    cited: dict[str, None] = {}

    def href(asset_id: str) -> str:
        if asset_href is not None:
            return asset_href(asset_id)
        asset = package.assets.get(asset_id) if package else None
        return f"assets/{asset_id}" + (f".{_MEDIA_EXTENSION[asset.media_type]}" if asset else "")

    result = ""
    previous = None
    for block in doc["blocks"]:
        kind, text = block["type"], block["text"]
        attrs = block.get("attributes") or {}
        spans = block.get("spans")
        if kind != "code":
            for span in spans or ():
                if span["type"] == "ref" and references.get(span["ref"], {}).get("kind") == "citation":
                    cited[span["ref"]] = None
            text = _inline_markdown(text, spans, references, notes, href)
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
        elif kind == "table":
            caption = f"*{text}*\n\n" if text else ""
            piece = caption + _table_markdown(block["table"])
        elif kind == "figure":
            figure = block["figure"]
            alt = " ".join(figure.get("alt", "").split()).replace("]", "\\]")
            piece = f"![{alt}]({href(figure['asset'])})" + (f"\n\n*{text}*" if text else "")
        elif kind == "equation":
            eq = block["equation"]
            if eq["syntax"] in ("latex", "mhchem"):
                source = eq["source"] if eq["syntax"] == "latex" else "\\ce{" + eq["source"] + "}"
                piece = "$$\n" + source + "\n$$"
            else:
                piece = "```" + ("xml" if eq["syntax"] == "mathml" else "text") + "\n" + eq["source"] + "\n```"
        else:  # paragraph, and an embed's static fallback text
            piece = text
        if not piece:
            continue
        if result:
            result += "\n" if kind == previous == "list_item" else "\n\n"
        result += piece
        previous = kind
    if notes:
        result += "\n\n" + "\n".join(
            f"[^{number}]: {references[ref_id]['text']}" for ref_id, number in notes.items())
    if cited:
        result += "\n\n" + "\n".join(_citation_line(references[ref_id]) for ref_id in cited)
    return result + "\n" if result else ""


def _citation_line(reference: Mapping[str, Any]) -> str:
    who = ", ".join(reference.get("authors", []))
    year = f" ({reference['year']})" if "year" in reference else ""
    tail = ". ".join(str(reference[k]) for k in ("container", "doi", "url") if k in reference)
    parts = [p for p in (f"{who}{year}".strip(), reference["title"], tail) if p]
    return f"[@{reference['id']}]: " + ". ".join(parts)


# ----------------------------------------------------------------------------------------------
# Migration and export between versions
# ----------------------------------------------------------------------------------------------
_LEGACY_EXTENSION = "slai.sldc.legacy"


def _attribute_fits(key: str, value: Any, block_type: str) -> bool:
    try:
        _validate_attributes({key: value}, block_type, "attribute")
    except SLDCValidationError:
        return False
    return True


def migrate_package(source: SLDCPackage | Mapping[str, Any], *,
                    timestamp: str | None = None) -> tuple[SLDCPackage, dict[str, Any]]:
    """Convert a 0.1.0 document (or package) to a 1.0.0 package without losing anything.

    Block ids are kept (they are already stable), so references and cached results stay valid.
    Anything 1.0.0 does not recognise is moved, not dropped: 0.1.0 ``annotations`` and attributes
    that fail the 1.0.0 schema land in each block's ``extensions["slai.sldc.legacy"]``. A
    ``sldc.migrate`` provenance entry (stamped ``timestamp``, now by default) records the source
    digest. Returns ``(package, report)``; the report is JSON-serialisable and states what moved.
    A package that is already 1.x is returned as an unchanged copy with ``"migrated": False``.
    """
    package = source if isinstance(source, SLDCPackage) else SLDCPackage(
        dict(source), version=LEGACY_VERSION)
    if package.version != LEGACY_VERSION:
        return package.copy(), {"kind": "sldc.migration", "migrated": False,
                                "from_version": package.version, "to_version": package.version}
    old = package.document
    source_digest = _sha256(_prepare_legacy(old))
    moved_attributes: list[dict[str, str]] = []
    moved_annotations: list[dict[str, Any]] = []
    blocks: list[dict[str, Any]] = []
    for block in old["blocks"]:
        migrated: dict[str, Any] = {"id": block["id"], "type": block["type"], "text": block["text"]}
        if "source_ref" in block:
            migrated["source_ref"] = block["source_ref"]
        kept: dict[str, Any] = {}
        legacy: dict[str, Any] = {}
        for key, value in (block.get("attributes") or {}).items():
            if _attribute_fits(key, value, block["type"]):
                kept[key] = copy.deepcopy(value)
            else:
                legacy.setdefault("attributes", {})[key] = copy.deepcopy(value)
                moved_attributes.append({"block": block["id"], "key": key})
        if block.get("annotations"):
            legacy["annotations"] = copy.deepcopy(block["annotations"])
            moved_annotations.append({"block": block["id"], "count": len(block["annotations"])})
        if kept:
            migrated["attributes"] = kept
        if legacy:
            migrated["extensions"] = {_LEGACY_EXTENSION: legacy}
        blocks.append(migrated)
    document: dict[str, Any] = {"schema": SCHEMA, "id": old["id"], "title": old["title"],
                                "language": old["language"], "blocks": blocks}
    if "metadata" in old:
        document["metadata"] = copy.deepcopy(old["metadata"])
    document["provenance"] = copy.deepcopy(old.get("provenance", [])) + [provenance_entry(
        "sldc.migrate", timestamp=timestamp, from_version=LEGACY_VERSION,
        to_version=FORMAT_VERSION, source_document_sha256=source_digest)]
    result = SLDCPackage(document)
    validate_package(result)
    report = {"kind": "sldc.migration", "migrated": True, "from_version": LEGACY_VERSION,
              "to_version": FORMAT_VERSION, "document_id": old["id"],
              "source_document_sha256": source_digest, "blocks": len(blocks),
              "block_ids_preserved": True, "lossy": False,
              "moved_attributes": moved_attributes, "moved_annotations": moved_annotations}
    return result, report


def migrate_file(source: str | os.PathLike[str], destination: str | os.PathLike[str], *,
                 overwrite: bool = False, timestamp: str | None = None) -> dict[str, Any]:
    """Migrate a 0.1.0 file to a new 1.0.0 file and return the migration report.

    The source is only ever read: it is never modified, renamed or replaced, and the destination
    must be a different file. The report also carries the SHA-256 and size of the source bytes.
    """
    src, dst = Path(source), _destination(destination)
    if dst.resolve() == src.resolve() or (dst.exists() and src.exists() and os.path.samefile(src, dst)):
        raise SLDCValidationError("Migration never overwrites its source: choose another destination")
    data = _read_file_bytes(src, MAX_PACKAGE_BYTES)
    package = loads_package(data)
    if package.version != LEGACY_VERSION:
        raise SLDCUnsupportedError(f"{src.name!r} is already SLDC {package.version}; nothing to migrate")
    migrated, report = migrate_package(package, timestamp=timestamp)
    write_package(dst, migrated, overwrite=overwrite)
    report.update(source_sha256=_sha256(data), source_bytes=len(data))
    return report


def export_legacy(package: SLDCPackage, *,
                  timestamp: str | None = None) -> tuple[SLDCPackage, dict[str, Any]]:
    """Make a lossy 0.1.0 copy of a 1.0.0 package, with a report of what was lost.

    Kept: text blocks, headings, lists, quotes, code, source references, metadata and provenance.
    Tables become tab-separated paragraphs, figures a ``[Figure] alt`` paragraph, embeds their
    static fallback text. Dropped: spans, styles, sections, references, images, comments,
    signatures (a signature belongs to the revision it signed, not to a derivative), history and
    extensions. The copy labels itself in ``metadata["sldc_export"]`` and in provenance.
    """
    if package.version == LEGACY_VERSION:
        return package.copy(), {"kind": "sldc.export_legacy", "lossy": False,
                                "from_version": LEGACY_VERSION, "to_version": LEGACY_VERSION}
    validate_package(package)
    document = package.document
    source_revision = revision_id(package)
    dropped = {"spans": 0, "block_styles": 0, "block_extensions": 0, "block_attributes": 0,
               "tables_flattened": 0, "figures_replaced": 0, "embeds_dropped": 0,
               "equations_flattened": 0, "inline_objects_flattened": 0,
               "styles": sum(len(g) for g in document.get("styles", {}).values()),
               "sections": len(document.get("sections", [])),
               "references": len(document.get("references", [])),
               "history": len(document.get("history", [])),
               "assets": len(package.assets), "comments": len(package.comments),
               "signatures": len(package.signatures)}
    blocks: list[dict[str, Any]] = []
    for block in document["blocks"]:
        kind = block["type"]
        dropped["spans"] += len(block.get("spans", ()))
        dropped["inline_objects_flattened"] += sum(s["type"] in ("equation", "image") for s in block.get("spans", ()))
        dropped["block_styles"] += int("style" in block) + int("section" in block)
        dropped["block_extensions"] += int("extensions" in block)
        legacy: dict[str, Any] = {"id": block["id"], "text": block_text(block)}
        attributes: dict[str, Any] = {}
        if kind in BLOCK_TYPES:
            legacy["type"] = kind
            for key, value in (block.get("attributes") or {}).items():
                if key in ("level", "language", "ordered", "depth"):
                    attributes[key] = value
                else:
                    dropped["block_attributes"] += 1
        elif kind == "table":
            legacy.update(type="paragraph", text=block_text(block))
            attributes["exported_from"] = "table"
            dropped["tables_flattened"] += 1
        elif kind == "figure":
            figure = block["figure"]
            figure_label = "[Figure] " + (figure["alt"] if figure.get("alt") else "decorative image")
            legacy.update(type="paragraph", text=figure_label + (f"\n{block['text']}" if block["text"] else ""))
            attributes["exported_from"] = "figure"
            dropped["figures_replaced"] += 1
        elif kind == "equation":
            legacy.update(type="paragraph", text=block_text(block))
            attributes["exported_from"] = "equation"
            dropped["equations_flattened"] += 1
        else:  # embed: keep its static fallback text, if it has any
            if not block["text"]:
                dropped["embeds_dropped"] += 1
                continue
            legacy["type"] = "paragraph"
            attributes["exported_from"] = "embed"
        if attributes:
            legacy["attributes"] = attributes
        if block.get("source_ref"):
            legacy["source_ref"] = block["source_ref"]
        blocks.append(legacy)
    summary = {key: value for key, value in dropped.items() if value}
    label: dict[str, Any] = {"lossy": True, "from_version": package.version,
                             "source_revision_id": source_revision, "dropped": summary}
    if document.get("publication"):
        label["publication"] = copy.deepcopy(document["publication"])  # Carried, not lost.
    metadata = copy.deepcopy(document.get("metadata", {}))
    metadata["sldc_export"] = label
    legacy_document = {
        "id": document["id"], "title": document["title"], "language": document["language"],
        "blocks": blocks, "metadata": metadata,
        "provenance": copy.deepcopy(document.get("provenance", [])) + [provenance_entry(
            "sldc.export_legacy", timestamp=timestamp, from_version=package.version,
            to_version=LEGACY_VERSION, source_revision_id=source_revision)],
    }
    result = SLDCPackage(legacy_document, version=LEGACY_VERSION)
    validate_package(result)
    report = {"kind": "sldc.export_legacy", "lossy": True, "from_version": package.version,
              "to_version": LEGACY_VERSION, "source_revision_id": source_revision,
              "blocks_in": len(document["blocks"]), "blocks_out": len(blocks), "dropped": summary}
    return result, report


# ----------------------------------------------------------------------------------------------
# Collaboration: revisions, comments, diff and merge
#
# The file is a portable snapshot, not a shared database. Live co-editing needs a service that
# orders operations and checks permissions (see SLAI Writer's synchronisation layer); what a file
# can do is carry stable ids, a revision lineage and review state, and let two copies be compared
# and merged at block level with conflicts reported, never resolved silently.
# ----------------------------------------------------------------------------------------------
COMMENTS_SCHEMA = "slai.sldc.comments/1"
_THREAD_KEYS = frozenset({"id", "anchor", "status", "created", "author", "messages", "resolved"})
_MESSAGE_KEYS = frozenset({"id", "author", "created", "text"})
_ANCHOR_KEYS = frozenset({"block", "start", "end", "block_sha256"})


def _require_v1(package: SLDCPackage, action: str) -> None:
    if _parse_version(package.version)[0] != 1:
        raise SLDCUnsupportedError(f"{action} needs an SLDC 1.x package")


def next_revision(package: SLDCPackage, *, author: str | None = None, note: str | None = None,
                  created: str | None = None) -> SLDCPackage:
    """Return a copy marked as the next revision of ``package``.

    The new revision records the previous revision's id as its parent, and the previous record
    moves into ``history``, so a chain of saved files proves its own lineage. Call it when you
    start editing a document that was saved, signed or shared; edit the copy afterwards. Earlier
    signatures stay with the old file and are not carried over: they sign the old revision.
    """
    _require_v1(package, "next_revision")
    result = package.copy()
    document = result.document
    stamp = _check_time(created or _utc_now(), "created")
    previous = document.get("revision")
    if previous is None:
        record: dict[str, Any] = {"number": 1, "parents": [], "created": stamp}
    else:
        previous_id = revision_id(package)
        document["history"] = list(document.get("history", [])) + [{"id": previous_id, **previous}]
        record = {"number": previous["number"] + 1, "parents": [previous_id], "created": stamp}
    if author:
        record["author"] = author
    if note:
        record["note"] = note
    document["revision"] = record
    result.signatures = []
    validate_package(result)
    return result


# --- Comparing and merging ---------------------------------------------------------------------
def _longest_increasing(values: Sequence[int]) -> set[int]:
    """Indices of one longest strictly increasing subsequence."""
    tails: list[int] = []
    tail_index: list[int] = []
    previous = [-1] * len(values)
    for index, value in enumerate(values):
        slot = bisect.bisect_left(tails, value)
        if slot == len(tails):
            tails.append(value)
            tail_index.append(index)
        else:
            tails[slot], tail_index[slot] = value, index
        previous[index] = tail_index[slot - 1] if slot else -1
    chosen: set[int] = set()
    index = tail_index[-1] if tail_index else -1
    while index >= 0:
        chosen.add(index)
        index = previous[index]
    return chosen


def diff_documents(old: Mapping[str, Any] | SLDCPackage,
                   new: Mapping[str, Any] | SLDCPackage) -> dict[str, Any]:
    """Compare two documents by stable block id.

    Returns ``added``, ``removed``, ``modified`` (any change to the block: text, spans, style,
    attributes ...) and ``moved`` (relative order changed) as lists of block ids, ``unchanged``
    (a count) and ``changed_fields`` (document-level keys, other than blocks and the revision
    record, whose values differ). Use it to show what changed since a signed revision.
    """
    first, second = _document_of(old), _document_of(new)
    _prepare(first)
    _prepare(second)
    a = {b["id"]: b for b in first["blocks"]}
    b = {block["id"]: block for block in second["blocks"]}
    added = [i for i in b if i not in a]
    removed = [i for i in a if i not in b]
    common_new = [i for i in b if i in a]
    modified = [i for i in common_new
                if block_digest(a[i], full=True) != block_digest(b[i], full=True)]
    old_position = {i: n for n, i in enumerate(i for i in a if i in b)}
    keep = _longest_increasing([old_position[i] for i in common_new])
    moved = [i for n, i in enumerate(common_new) if n not in keep]
    skipped = {"blocks", "revision", "history"}
    changed_fields = sorted(k for k in (set(first) | set(second)) - skipped
                            if first.get(k) != second.get(k))
    return {"added": added, "removed": removed, "modified": modified, "moved": moved,
            "unchanged": len(common_new) - len(set(modified) | set(moved)),
            "changed_fields": changed_fields}


_ABSENT = object()


def merge_documents(base: Mapping[str, Any] | SLDCPackage, ours: Mapping[str, Any] | SLDCPackage,
                    theirs: Mapping[str, Any] | SLDCPackage) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Three-way merge of two edited copies of ``base`` (1.x documents), block by block.

    A change made on only one side is taken. A block changed identically on both sides is taken
    once. Anything else is a **conflict**: the same block edited differently, edited on one side
    and deleted on the other, added on both sides with different content, reordered differently,
    or a document-level field changed differently. Conflicts are returned, never resolved
    silently; the merged document keeps *our* version where they occur. Blocks added on both sides
    after the same block land with ours first. Matching relies on stable ids: copies that were
    imported separately (each numbering its blocks ``b00001`` ...) are unrelated, not mergeable.

    Returns ``(document, conflicts)``; each conflict is ``{"kind", "block"|"field"}``. The
    ``revision`` and ``history`` records are taken from ours; :func:`merge_packages` sets them.
    """
    docs = []
    for name, value in (("base", base), ("ours", ours), ("theirs", theirs)):
        document = copy.deepcopy(dict(_document_of(value)))
        if document.get("schema") != SCHEMA:
            raise SLDCUnsupportedError(f"merge needs 1.x documents; {name} is not one")
        _prepare_v1(document)
        docs.append(document)
    b, o, t = docs
    conflicts: list[dict[str, Any]] = []
    merged: dict[str, Any] = {}
    for key in sorted((set(b) | set(o) | set(t)) - {"blocks", "revision", "history"}):
        base_value, ours_value, theirs_value = b.get(key, _ABSENT), o.get(key, _ABSENT), t.get(key, _ABSENT)
        if ours_value == theirs_value or theirs_value == base_value:
            chosen = ours_value
        elif ours_value == base_value:
            chosen = theirs_value
        else:
            conflicts.append({"kind": "field", "field": key})
            chosen = ours_value
        if chosen is not _ABSENT:
            merged[key] = chosen
    for key in ("revision", "history"):
        if key in o:
            merged[key] = o[key]

    by_base = {x["id"]: x for x in b["blocks"]}
    by_ours = {x["id"]: x for x in o["blocks"]}
    by_theirs = {x["id"]: x for x in t["blocks"]}
    chosen_blocks: dict[str, dict[str, Any]] = {}
    for block_id in dict.fromkeys([*by_ours, *by_theirs, *by_base]):
        base_block, ours_block, theirs_block = (by_base.get(block_id), by_ours.get(block_id),
                                                by_theirs.get(block_id))
        if base_block is None:
            if ours_block is not None and theirs_block is not None and ours_block != theirs_block:
                conflicts.append({"kind": "both_added", "block": block_id})
            pick: dict[str, Any] | None = ours_block or theirs_block
        elif ours_block == base_block:
            pick = theirs_block
        elif theirs_block == base_block or ours_block == theirs_block:
            pick = ours_block
        else:
            conflicts.append({"kind": "both_modified" if ours_block and theirs_block
                              else "delete_modify", "block": block_id})
            pick = ours_block or theirs_block
        if pick is not None:
            chosen_blocks[block_id] = pick

    order_base = [x["id"] for x in b["blocks"]]
    order_ours = [x["id"] for x in o["blocks"]]
    order_theirs = [x["id"] for x in t["blocks"]]
    survivors = [i for i in order_base if i in chosen_blocks]
    ours_common = [i for i in order_ours if i in by_base and i in chosen_blocks]
    theirs_common = [i for i in order_theirs if i in by_base and i in chosen_blocks]
    if ours_common == survivors:
        skeleton = theirs_common
    elif theirs_common in (survivors, ours_common):
        skeleton = ours_common
    else:
        conflicts.append({"kind": "order", "block": None})
        skeleton = ours_common
    after: dict[str | None, str | None] = {None: None}  # A linked list: O(1) insertion.
    tail: str | None = None
    for block_id in skeleton:
        after[tail] = block_id
        after[block_id] = None
        tail = block_id
    for order in (order_theirs, order_ours, order_base):  # Ours is walked last: ours lands first.
        previous: str | None = None
        for block_id in order:
            if block_id not in chosen_blocks:
                continue
            if block_id not in after:
                after[block_id] = after[previous]
                after[previous] = block_id
            previous = block_id
    merged["blocks"] = []
    cursor = after[None]
    while cursor is not None:
        merged["blocks"].append(chosen_blocks[cursor])
        cursor = after[cursor]
    return merged, conflicts


def _merge_threads(ours: Sequence[Mapping[str, Any]],
                   theirs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Union of comment threads by id; messages are unioned by id and ordered by time."""
    merged: dict[str, dict[str, Any]] = {t["id"]: copy.deepcopy(dict(t)) for t in ours}
    for thread in theirs:
        existing = merged.get(thread["id"])
        if existing is None:
            merged[thread["id"]] = copy.deepcopy(dict(thread))
            continue
        messages = {m["id"]: m for m in existing["messages"]}
        messages.update({m["id"]: copy.deepcopy(dict(m)) for m in thread["messages"]
                         if m["id"] not in messages})
        existing["messages"] = sorted(messages.values(), key=lambda m: (m["created"], m["id"]))
        if thread["status"] == "open" and existing["status"] == "resolved":
            existing["status"] = "open"  # Reopened on one side: stay open.
            existing.pop("resolved", None)
    return list(merged.values())


def merge_packages(base: SLDCPackage, ours: SLDCPackage, theirs: SLDCPackage, *,
                   author: str | None = None, note: str | None = None,
                   created: str | None = None) -> tuple[SLDCPackage, list[dict[str, Any]]]:
    """Merge two edited copies of ``base``: document (see :func:`merge_documents`), assets,
    comments and revision lineage.

    The result is a new revision whose parents are our and their revision ids. Assets are unioned
    (the same id with different bytes is a conflict) and then pruned to those still used. Comment
    threads are unioned. Signatures are dropped on purpose: they sign the revisions that were
    merged, not the merge. Returns ``(package, conflicts)``.
    """
    for package in (base, ours, theirs):
        _require_v1(package, "merge_packages")
    document, conflicts = merge_documents(base.document, ours.document, theirs.document)
    assets = dict(ours.assets)
    for asset_id, asset in theirs.assets.items():
        if asset_id in assets and assets[asset_id].data != asset.data:
            conflicts.append({"kind": "asset", "asset": asset_id})
        else:
            assets[asset_id] = asset
    ours_id, theirs_id = revision_id(ours), revision_id(theirs)
    history: dict[str, dict[str, Any]] = {}
    numbers = [0]
    for side, side_id in ((ours, ours_id), (theirs, theirs_id)):
        for record in side.document.get("history", []):
            history.setdefault(record["id"], record)
        head = side.document.get("revision")
        if head is not None:
            history.setdefault(side_id, {"id": side_id, **head})
            numbers.append(head["number"])
    record = {"number": max(numbers) + 1,
              "parents": list(dict.fromkeys([ours_id, theirs_id])),
              "created": _check_time(created or _utc_now(), "created")}
    if author:
        record["author"] = author
    if note:
        record["note"] = note
    document["revision"] = record
    if history:
        document["history"] = list(history.values())
    merged = prune_assets(SLDCPackage(
        document, assets, _merge_threads(ours.comments, theirs.comments),
        version=newest_version(base.version, ours.version, theirs.version)))
    validate_package(merged)
    return merged, conflicts


# --- Review comments ---------------------------------------------------------------------------
def _validate_comments(threads: Any) -> None:
    if not isinstance(threads, list) or len(threads) > MAX_COMMENT_THREADS:
        raise SLDCValidationError(f"comments must be a list of at most {MAX_COMMENT_THREADS} threads")
    seen: set[str] = set()
    for index, thread in enumerate(threads):
        label = f"comments[{index}]"
        if not isinstance(thread, dict):
            raise SLDCValidationError(f"{label} must be an object")
        _check_keys(thread, _THREAD_KEYS, label)
        thread_id = _id_field(thread.get("id"), f"{label}.id")
        if thread_id in seen:
            raise SLDCValidationError(f"Duplicate comment thread id {thread_id!r}")
        seen.add(thread_id)
        anchor = thread.get("anchor")
        if not isinstance(anchor, dict):
            raise SLDCValidationError(f"{label}.anchor must be an object")
        _check_keys(anchor, _ANCHOR_KEYS, f"{label}.anchor")
        _id_field(anchor.get("block"), f"{label}.anchor.block")
        if ("start" in anchor) != ("end" in anchor):
            raise SLDCValidationError(f"{label}.anchor needs both start and end, or neither")
        if "start" in anchor:
            start = _bounded_int(anchor["start"], 0, MAX_TEXT_CHARS, f"{label}.anchor.start")
            _bounded_int(anchor["end"], start + 1, MAX_TEXT_CHARS, f"{label}.anchor.end")
        if "block_sha256" in anchor and (not isinstance(anchor["block_sha256"], str)
                                         or not _HEX_SHA256.fullmatch(anchor["block_sha256"])):
            raise SLDCValidationError(f"{label}.anchor.block_sha256 must be 64 hex digits")
        if not _has(("open", "resolved"), thread.get("status")):
            raise SLDCValidationError(f"{label}.status must be 'open' or 'resolved'")
        _check_time(thread.get("created"), f"{label}.created")
        _text_field(thread, "author", label, 200, required=True)
        if (thread["status"] == "resolved") != ("resolved" in thread):
            raise SLDCValidationError(f"{label} must carry 'resolved' exactly when it is resolved")
        if "resolved" in thread:
            resolved = thread["resolved"]
            if not isinstance(resolved, dict):
                raise SLDCValidationError(f"{label}.resolved must be an object")
            _check_keys(resolved, {"by", "at"}, f"{label}.resolved")
            _text_field(resolved, "by", f"{label}.resolved", 200, required=True)
            _check_time(resolved.get("at"), f"{label}.resolved.at")
        messages = thread.get("messages")
        if not isinstance(messages, list) or not 1 <= len(messages) <= MAX_THREAD_MESSAGES:
            raise SLDCValidationError(f"{label}.messages must hold 1 to {MAX_THREAD_MESSAGES} messages")
        message_ids: set[str] = set()
        for message in messages:
            if not isinstance(message, dict):
                raise SLDCValidationError(f"{label} messages must be objects")
            _check_keys(message, _MESSAGE_KEYS, f"{label}.messages")
            message_id = _id_field(message.get("id"), f"{label}.messages.id")
            if message_id in message_ids:
                raise SLDCValidationError(f"Duplicate message id {message_id!r} in {thread_id!r}")
            message_ids.add(message_id)
            _text_field(message, "author", f"{label}.messages", 200, required=True)
            _check_time(message.get("created"), f"{label}.messages.created")
            _text_field(message, "text", f"{label}.messages", 20_000, required=True)


def _comments_part(threads: list[dict[str, Any]]) -> bytes:
    _validate_comments(threads)
    raw = _canonical_json({"schema": COMMENTS_SCHEMA, "threads": threads})
    if len(raw) > MAX_COMMENTS_BYTES:
        raise SLDCValidationError("Comments exceed the size limit")
    return raw


def _comments_from_part(obj: dict[str, Any]) -> list[dict[str, Any]]:
    _check_keys(obj, {"schema", "threads"}, "comments")
    if obj.get("schema") != COMMENTS_SCHEMA:
        raise SLDCValidationError(f"Unsupported comments schema {obj.get('schema')!r}")
    threads = obj.get("threads")
    if not isinstance(threads, list):
        raise SLDCValidationError("comments.threads must be an array")
    _validate_comments(threads)
    return threads


def _thread(package: SLDCPackage, thread_id: str) -> int:
    for index, thread in enumerate(package.comments):
        if thread["id"] == thread_id:
            return index
    raise SLDCValidationError(f"No comment thread {thread_id!r}")


def add_comment(package: SLDCPackage, block_id: str, text: str, *, author: str,
                created: str | None = None, start: int | None = None, end: int | None = None,
                thread_id: str | None = None,
                message_id: str | None = None) -> tuple[SLDCPackage, str]:
    """Start a review thread on a block (or on ``text[start:end]`` of it). Returns the new
    package and the thread id.

    The anchor records the block id and a digest of its text, so :func:`comment_anchor_status`
    can later say whether the commented text changed. Comments are review state: they are not
    covered by default when signing, and private notes should not be stored here at all. Authors
    are free-form assertions (use a pseudonym where identity should not be shared).
    """
    _require_v1(package, "add_comment")
    block = next((b for b in package.document["blocks"] if b["id"] == block_id), None)
    if block is None:
        raise SLDCValidationError(f"No block {block_id!r}")
    anchor: dict[str, Any] = {"block": block_id, "block_sha256": block_digest(block)}
    if start is not None or end is not None:
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(block["text"]):
            raise SLDCValidationError("start and end must select part of the block's text")
        anchor.update(start=start, end=end)
    stamp = _check_time(created or _utc_now(), "created")
    identifier = thread_id or "t" + uuid.uuid4().hex[:16]
    thread = {"id": identifier, "anchor": anchor, "status": "open", "created": stamp,
              "author": author,
              "messages": [{"id": message_id or "m" + uuid.uuid4().hex[:16], "author": author,
                            "created": stamp, "text": text}]}
    result = package.copy()
    result.comments.append(thread)
    _validate_comments(result.comments)
    return result, identifier


def reply_to_thread(package: SLDCPackage, thread_id: str, text: str, *, author: str,
                    created: str | None = None, message_id: str | None = None) -> SLDCPackage:
    """Return a copy with one more message on a thread (resolved threads accept replies)."""
    _require_v1(package, "reply_to_thread")
    result = package.copy()
    result.comments[_thread(result, thread_id)]["messages"].append({
        "id": message_id or "m" + uuid.uuid4().hex[:16], "author": author,
        "created": _check_time(created or _utc_now(), "created"), "text": text})
    _validate_comments(result.comments)
    return result


def set_thread_status(package: SLDCPackage, thread_id: str, status: str, *, by: str,
                      at: str | None = None) -> SLDCPackage:
    """Resolve or reopen a thread. Resolving keeps every message: it hides, never deletes."""
    _require_v1(package, "set_thread_status")
    if not _has(("open", "resolved"), status):
        raise SLDCValidationError("status must be 'open' or 'resolved'")
    result = package.copy()
    thread = result.comments[_thread(result, thread_id)]
    thread["status"] = status
    if status == "resolved":
        thread["resolved"] = {"by": by, "at": _check_time(at or _utc_now(), "at")}
    else:
        thread.pop("resolved", None)
    _validate_comments(result.comments)
    return result


def comment_anchor_status(package: SLDCPackage) -> dict[str, str]:
    """Say where each thread's anchor stands now: ``ok``, ``stale`` (the block's text changed
    since the comment was made) or ``orphaned`` (the block is gone)."""
    _require_v1(package, "comment_anchor_status")
    blocks = {b["id"]: b for b in package.document["blocks"]}
    states: dict[str, str] = {}
    for thread in package.comments:
        anchor = thread["anchor"]
        block = blocks.get(anchor["block"])
        if block is None:
            states[thread["id"]] = "orphaned"
        elif "block_sha256" in anchor and anchor["block_sha256"] != block_digest(block):
            states[thread["id"]] = "stale"
        else:
            states[thread["id"]] = "ok"
    return states


# ----------------------------------------------------------------------------------------------
# Cryptography (optional dependency, imported only when used)
# ----------------------------------------------------------------------------------------------
def _crypto() -> types.SimpleNamespace:
    """The pieces of the ``cryptography`` package this module uses. Nothing is hand-rolled."""
    try:
        from cryptography.exceptions import InvalidSignature, InvalidTag # type: ignore
        from cryptography.hazmat.primitives import hashes, serialization # type: ignore
        from cryptography.hazmat.primitives.asymmetric import ed25519, x25519 # type: ignore
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM # type: ignore
        from cryptography.hazmat.primitives.kdf.hkdf import HKDF # type: ignore
    except ImportError as exc:
        raise SLDCDependencyError(
            "Signatures and encryption need the 'cryptography' package "
            "(pip install cryptography); reading and writing unsigned, unencrypted "
            "documents does not") from exc
    return types.SimpleNamespace(
        InvalidSignature=InvalidSignature, InvalidTag=InvalidTag, hashes=hashes,
        serialization=serialization, ed25519=ed25519, x25519=x25519, AESGCM=AESGCM, HKDF=HKDF)


def key_fingerprint(public_key: bytes) -> str:
    """The identifier of a public key: SHA-256 (hex) of its raw bytes. Used as ``key_id``."""
    return _sha256(bytes(public_key))


# ----------------------------------------------------------------------------------------------
# Signing
#
# A signature is a detached record in ``signatures/<id>.json``. It binds a key to the exact bytes
# of one revision (the document and every asset, by SHA-256) plus the format, version and required
# features. Records can be added without touching what they cover, so several people can sign the
# same revision, and a countersignature can cover an earlier signature. Editing creates a new
# revision (see next_revision); an old signature keeps verifying against the old file and reports
# "modified" against any later content. Private keys never enter a document or this module's
# state beyond the Signer object the caller provides.
# ----------------------------------------------------------------------------------------------
SIGNATURE_SCHEMA = "slai.sldc.signature/1"
_SIGNATURE_DOMAIN = b"SLDC-SIGNATURE-V1\0"
_PURPOSES = frozenset({"approval", "authorship", "review", "witness", "countersignature"})
_RECORD_KEYS = frozenset({"schema", "id", "algorithm", "key_id", "public_key", "payload", "signature"})
_PAYLOAD_KEYS = frozenset({"domain", "format", "version", "document_id", "revision_id", "requires",
                           "covered", "purpose", "signed_at", "statement", "key_id", "algorithm",
                           "countersigns"})


class Signer(Protocol):
    """Anything that can sign bytes without handing out its private key (an in-process key, an OS
    keystore, a hardware token, a signing service)."""
    algorithm: str
    key_id: str
    public_key: bytes

    def sign(self, data: bytes) -> bytes: ...


class Ed25519Signer:
    """An Ed25519 signer backed by a ``cryptography`` private key.

    The key stays inside the object: it is not printed, not exposed and not picklable. For real
    use keep the key in an OS keystore or hardware module and implement :class:`Signer` on top.
    """
    algorithm = "ed25519"

    def __init__(self, private_key: Any) -> None:
        crypto = _crypto()
        if not isinstance(private_key, crypto.ed25519.Ed25519PrivateKey):
            raise SLDCValidationError("Ed25519Signer needs a cryptography Ed25519PrivateKey")
        self._key = private_key
        self.public_key = private_key.public_key().public_bytes(
            crypto.serialization.Encoding.Raw, crypto.serialization.PublicFormat.Raw)
        self.key_id = key_fingerprint(self.public_key)

    @classmethod
    def generate(cls) -> Ed25519Signer:
        return cls(_crypto().ed25519.Ed25519PrivateKey.generate())

    @classmethod
    def from_seed(cls, seed: bytes) -> Ed25519Signer:
        """A signer from a 32-byte secret seed (deterministic; for tests and key import)."""
        if not isinstance(seed, (bytes, bytearray)) or len(seed) != 32:
            raise SLDCValidationError("An Ed25519 seed is exactly 32 bytes")
        return cls(_crypto().ed25519.Ed25519PrivateKey.from_private_bytes(bytes(seed)))

    @classmethod
    def from_pem(cls, pem: bytes, password: bytes | None = None) -> Ed25519Signer:
        try:
            key = _crypto().serialization.load_pem_private_key(pem, password)
        except (ValueError, TypeError) as exc:
            raise SLDCValidationError(f"Cannot load the private key: {exc}") from exc
        return cls(key)

    def sign(self, data: bytes) -> bytes:
        return self._key.sign(bytes(data))

    def __repr__(self) -> str:
        return f"Ed25519Signer(key_id={self.key_id[:16]}...)"

    def __reduce__(self) -> Any:
        raise TypeError("A signer holds a private key and cannot be pickled or copied")


def _validate_signature_record(record: Any) -> None:
    """Shape only: no cryptography, so unsigned-document tooling can still open signed files."""
    if not isinstance(record, dict):
        raise SLDCValidationError("A signature record must be an object")
    _check_keys(record, _RECORD_KEYS, "signature")
    if record.keys() != _RECORD_KEYS:
        raise SLDCValidationError(f"A signature record needs exactly {sorted(_RECORD_KEYS)}")
    if record["schema"] != SIGNATURE_SCHEMA:
        raise SLDCValidationError(f"Unsupported signature schema {record['schema']!r}")
    _id_field(record["id"], "signature id", _ASSET_ID)
    _id_field(record["algorithm"], "signature algorithm", _KIND)
    for key in ("key_id",):
        if not isinstance(record[key], str) or not _HEX_SHA256.fullmatch(record[key]):
            raise SLDCValidationError(f"signature {key} must be 64 hex digits")
    known = record["algorithm"] == "ed25519"
    _b64d(record["public_key"], "signature public_key", length=32 if known else None)
    _b64d(record["signature"], "signature value", length=64 if known else None)
    payload = record["payload"]
    if not isinstance(payload, dict):
        raise SLDCValidationError("signature payload must be an object")
    _check_keys(payload, _PAYLOAD_KEYS, "signature payload")
    required = _PAYLOAD_KEYS - {"statement", "countersigns"}
    if not required <= payload.keys():
        raise SLDCValidationError(f"signature payload needs {sorted(required - payload.keys())}")
    if payload["domain"] != SIGNATURE_SCHEMA or payload["format"] != FORMAT_ID:
        raise SLDCValidationError("signature payload has the wrong domain or format")
    _parse_version(payload["version"])
    _id_field(payload["document_id"], "signature document_id")
    if not isinstance(payload["revision_id"], str) or not _HEX_SHA256.fullmatch(payload["revision_id"]):
        raise SLDCValidationError("signature revision_id must be 64 hex digits")
    if not isinstance(payload["requires"], list) or not all(isinstance(r, str) for r in payload["requires"]):
        raise SLDCValidationError("signature requires must be a list of names")
    covered = payload["covered"]
    if not isinstance(covered, list) or not 1 <= len(covered) <= MAX_PARTS:
        raise SLDCValidationError("signature covered must list the signed parts")
    for entry in covered:
        if not isinstance(entry, dict) or entry.keys() != {"name", "sha256", "size"} \
                or not isinstance(entry["name"], str) or not _PART_NAME.fullmatch(entry["name"]) \
                or not isinstance(entry["sha256"], str) or not _HEX_SHA256.fullmatch(entry["sha256"]) \
                or type(entry["size"]) is not int or entry["size"] < 0:
            raise SLDCValidationError("signature covered entries need name, sha256 and size")
    if not _has(_PURPOSES, payload["purpose"]):
        raise SLDCValidationError(f"signature purpose must be one of {sorted(_PURPOSES)}")
    _check_time(payload["signed_at"], "signature signed_at")
    _text_field(payload, "statement", "signature", 1000)
    if payload["key_id"] != record["key_id"] or payload["algorithm"] != record["algorithm"]:
        raise SLDCValidationError("signature payload does not match its record's key or algorithm")
    for entry in payload.get("countersigns", []):
        if not isinstance(entry, dict) or entry.keys() != {"id", "signature_sha256"} \
                or not isinstance(entry["signature_sha256"], str) \
                or not _HEX_SHA256.fullmatch(entry["signature_sha256"]):
            raise SLDCValidationError("countersigns entries need id and signature_sha256")
        _id_field(entry["id"], "countersigned id", _ASSET_ID)


def sign_package(package: SLDCPackage, signer: Signer, *, purpose: str = "approval",
                 signed_at: str | None = None, statement: str | None = None,
                 cover_comments: bool = False, countersign: Iterable[str] = (),
                 signature_id: str | None = None) -> SLDCPackage:
    """Return a copy of ``package`` with one more signature over its current revision.

    ``purpose`` is one of approval, authorship, review, witness or countersignature. Signing
    covers the document, every asset, the format, version and required features; add
    ``cover_comments=True`` to cover the review comments too. ``countersign`` lists signature ids
    this signature also attests (then ``purpose`` must be ``"countersignature"``). ``signed_at``
    is the signer's own claim of the time and is recorded as such: nothing here proves it.

    Freeze the revision you show the signer first (``next_revision`` after signing starts the
    next draft). The signer must be an object you control; see :class:`Ed25519Signer`.
    """
    _require_v1(package, "sign_package")
    if not _has(_PURPOSES, purpose):
        raise SLDCValidationError(f"purpose must be one of {sorted(_PURPOSES)}")
    if signer.algorithm != "ed25519":
        raise SLDCUnsupportedError(f"Unsupported signature algorithm {signer.algorithm!r}")
    if signer.key_id != key_fingerprint(signer.public_key):
        raise SLDCValidationError("Signer key_id does not match its public key")
    targets = sorted(set(countersign))
    if bool(targets) != (purpose == "countersignature"):
        raise SLDCValidationError("A countersignature names the signatures it attests, and only it does")
    parts = _build_parts(package)
    inventory = {e["name"]: e for e in _inventory(parts)}
    names = [n for n, e in inventory.items() if e["role"] in ("content", "asset")]
    if cover_comments:
        if "comments.json" not in inventory:
            raise SLDCValidationError("There are no comments to cover")
        names.append("comments.json")
    payload: dict[str, Any] = {
        "domain": SIGNATURE_SCHEMA, "format": FORMAT_ID, "version": package.version,
        "document_id": package.document["id"],
        "revision_id": _revision_digest((n, inventory[n]["size"], inventory[n]["sha256"])
                                        for n in inventory if inventory[n]["role"] in ("content", "asset")),
        "requires": _required_features(package.document),
        "covered": [{"name": n, "sha256": inventory[n]["sha256"], "size": inventory[n]["size"]}
                    for n in sorted(names)],
        "purpose": purpose, "signed_at": _check_time(signed_at or _utc_now(), "signed_at"),
        "key_id": signer.key_id, "algorithm": signer.algorithm,
    }
    if statement:
        payload["statement"] = statement
    if targets:
        known = {r["id"]: r for r in package.signatures}
        missing = [t for t in targets if t not in known]
        if missing:
            raise SLDCValidationError(f"Cannot countersign unknown signatures: {missing}")
        payload["countersigns"] = [
            {"id": t, "signature_sha256": _sha256(_b64d(known[t]["signature"], "signature"))}
            for t in targets]
    message = _SIGNATURE_DOMAIN + _canonical_json(payload)
    signature = signer.sign(message)
    if not _verify_ed25519(signer.public_key, signature, message):
        raise SLDCValidationError("The signer produced a signature that does not verify")
    record = {"schema": SIGNATURE_SCHEMA,
              "id": signature_id or "sig-" + _sha256(_canonical_json(payload))[:16],
              "algorithm": signer.algorithm, "key_id": signer.key_id,
              "public_key": _b64e(signer.public_key), "payload": payload,
              "signature": _b64e(signature)}
    if any(existing["id"] == record["id"] for existing in package.signatures):
        raise SLDCValidationError(f"Signature {record['id']!r} already exists in this package")
    result = package.copy()
    result.signatures.append(record)
    validate_package(result)
    return result


def _verify_ed25519(public_key: bytes, signature: bytes, message: bytes) -> bool:
    crypto = _crypto()
    try:
        crypto.ed25519.Ed25519PublicKey.from_public_bytes(public_key).verify(signature, message)
    except (crypto.InvalidSignature, ValueError):
        return False
    return True


def verify_package(package: SLDCPackage, *, trusted_keys: Iterable[str] | Mapping[str, Any] | None = None,
                   revoked_keys: Iterable[str] = ()) -> list[dict[str, Any]]:
    """Check every signature on ``package`` and report, per signature, what is and is not known.

    ``trusted_keys`` are the ``key_id`` values (see :func:`key_fingerprint`) *you* have decided to
    trust, for example from a directory you control; ``revoked_keys`` those you no longer trust.
    A key embedded in the file proves nothing about who holds it. Each result has:

    ``signature_valid``       the bytes verify under the embedded key (cryptography only);
    ``covers_current_content``  the signed parts are byte-for-byte what the package holds now;
    ``changed_parts``         when they are not, which signed parts differ or are missing;
    ``trust``                 ``trusted``, ``unknown_signer`` or ``revoked``;
    ``state``                 one of ``valid_trusted``, ``valid_unknown_signer``, ``valid_revoked``,
                              ``modified`` (genuine signature, but over a different revision),
                              ``invalid`` (does not verify, or its countersignatures do not hold)
                              or ``indeterminate`` (algorithm this reader does not know);
    ``signed_revision_id``    the revision that was actually signed;
    ``signed_at_claimed``     the signer's own timestamp: assertion, not evidence. No timestamp
                              authority is consulted, and a missing network is never "verified".
    """
    _require_v1(package, "verify_package")
    trusted = set(trusted_keys or ())
    revoked = set(revoked_keys)
    inventory = {e["name"]: e for e in _inventory(_build_parts(package))}
    current_revision = _revision_digest((n, e["size"], e["sha256"]) for n, e in inventory.items()
                                        if e["role"] in ("content", "asset"))
    signature_hashes = {r["id"]: _sha256(_b64d(r["signature"], "signature"))
                        for r in package.signatures}
    results: list[dict[str, Any]] = []
    for record in package.signatures:
        payload, problems = record["payload"], []
        trust = "revoked" if record["key_id"] in revoked else (
            "trusted" if record["key_id"] in trusted else "unknown_signer")
        public_key = _b64d(record["public_key"], "public_key")
        known = record["algorithm"] == "ed25519"
        valid = False
        if known:
            if key_fingerprint(public_key) != record["key_id"]:
                problems.append("key_id is not the fingerprint of the embedded public key")
            else:
                valid = _verify_ed25519(public_key, _b64d(record["signature"], "signature"),
                                        _SIGNATURE_DOMAIN + _canonical_json(payload))
                if not valid:
                    problems.append("signature does not verify")
        else:
            problems.append(f"unsupported algorithm {record['algorithm']!r}")
        changed = sorted(entry["name"] for entry in payload["covered"]
                         if inventory.get(entry["name"], {}).get("sha256") != entry["sha256"]
                         or inventory.get(entry["name"], {}).get("size") != entry["size"])
        covers = (not changed and payload["revision_id"] == current_revision
                  and payload["document_id"] == package.document["id"]
                  and payload["version"] == package.version)
        if changed:
            problems.append("signed parts changed since signing")
        elif not covers:
            problems.append("signed revision differs from the current content")
        counters_ok = True
        for entry in payload.get("countersigns", []):
            if signature_hashes.get(entry["id"]) != entry["signature_sha256"]:
                counters_ok = False
                problems.append(f"countersigned signature {entry['id']!r} is missing or altered")
        if not known:
            state = "indeterminate"
        elif not valid or not counters_ok:
            state = "invalid"
        elif not covers:
            state = "modified"
        else:
            state = f"valid_{trust}"
        results.append({"id": record["id"], "key_id": record["key_id"], "algorithm": record["algorithm"],
                        "purpose": payload["purpose"], "state": state, "trust": trust,
                        "signature_valid": valid, "covers_current_content": covers,
                        "changed_parts": changed, "signed_revision_id": payload["revision_id"],
                        "signed_at_claimed": payload["signed_at"],
                        "statement": payload.get("statement"), "problems": problems})
    return results


# ----------------------------------------------------------------------------------------------
# Encryption
#
# The whole plain archive is sealed with AES-256-GCM under a random 256-bit content key. The
# content key is wrapped once per recipient: by a key derived from a password with scrypt, or by an
# X25519 key agreement (ephemeral key, HKDF-SHA256). Everything comes from the maintained
# ``cryptography`` package and the standard library; nothing here is a custom primitive.
#
#   mimetype         stored, same sniffable header as every SLDC
#   manifest.json    {"format", "version", "container": "encrypted", "encryption": {...}}
#   payload.bin      the AES-GCM ciphertext (the plain .sldc archive plus a 16-byte tag)
#
# What is authenticated: the scheme, cipher, format, version and recovery policy are the AEAD's
# associated data. The recipient list is not, so recipients can be added without re-encrypting;
# every wrap is itself authenticated and only ever opens *this* file's content key. What stays
# visible: file size, the recipient list (kind, role, optional label, key id) and the recovery
# policy. A document's own title and metadata are inside the ciphertext.
# ----------------------------------------------------------------------------------------------
ENCRYPTION_SCHEME = "slai.sldc.enc/1"
_CIPHER = "aes-256-gcm"
_ENVELOPE_MEMBERS = ("mimetype", "manifest.json", "payload.bin")
_ENCRYPTION_KEYS = frozenset({"scheme", "cipher", "nonce", "recovery_policy", "ciphertext_size",
                              "recipients"})
_RECOVERY_POLICIES = frozenset({"none", "recovery_recipient"})
_RECIPIENT_COMMON = frozenset({"id", "kind", "role", "label", "nonce", "wrapped"})
_RECIPIENT_KEYS = {"password": _RECIPIENT_COMMON | {"kdf"},
                   "x25519": _RECIPIENT_COMMON | {"key_id", "ephemeral"}}
_KDF_KEYS = frozenset({"name", "n", "r", "p", "salt"})
_ROLES = frozenset({"primary", "recovery"})
SCRYPT_R = 8
SCRYPT_P = 1
DEFAULT_SCRYPT_N = 2 ** 17  # 128 MiB, the OWASP-recommended scrypt setting with r=8, p=1.
MIN_SCRYPT_N = 2 ** 14      # Weaker settings are refused on write and on read.
MAX_SCRYPT_N = 2 ** 18      # Caps the memory a hostile file can make a reader spend (256 MiB).
_GENERIC_DECRYPT_FAILURE = ("Cannot decrypt this SLDC: the credential is wrong, does not match any "
                            "recipient, or the file was modified")


@dataclass(frozen=True)
class Recipient:
    """Someone (or something) who can open an encrypted document.

    Build one with :func:`password_recipient` or :func:`x25519_recipient`. ``role`` is
    ``"primary"`` or ``"recovery"``; ``label`` is optional free text that is stored *unencrypted*
    in the file, so keep names and secrets out of it.
    """
    kind: str
    role: str = "primary"
    label: str | None = None
    public_key: bytes | None = None
    password: str | None = field(default=None, repr=False)
    scrypt_n: int = DEFAULT_SCRYPT_N


def _recipient_common(role: str, label: str | None) -> None:
    if not _has(_ROLES, role):
        raise SLDCValidationError(f"recipient role must be one of {sorted(_ROLES)}")
    if label is not None:
        _text_field({"label": label}, "label", "recipient", 100)


def password_recipient(password: str, *, label: str | None = None, recovery: bool = False,
                       scrypt_n: int = DEFAULT_SCRYPT_N) -> Recipient:
    """A recipient who opens the file with ``password`` (stretched with scrypt).

    The password is normalised to Unicode NFKC and must have at least ``MIN_PASSWORD_CHARS``
    characters. A document password is a password, not a login code: never reuse an account or
    one-time code as one. A long random passphrase from a password manager is best.
    """
    if not isinstance(password, str) or len(unicodedata.normalize("NFKC", password)) < MIN_PASSWORD_CHARS:
        raise SLDCValidationError(f"A document password needs at least {MIN_PASSWORD_CHARS} characters")
    _bounded_int(scrypt_n, MIN_SCRYPT_N, MAX_SCRYPT_N, "scrypt_n")
    if scrypt_n & (scrypt_n - 1):
        raise SLDCValidationError("scrypt_n must be a power of two")
    _recipient_common("recovery" if recovery else "primary", label)
    return Recipient("password", "recovery" if recovery else "primary", label,
                     password=password, scrypt_n=scrypt_n)


def _x25519_public_bytes(public_key: Any) -> bytes:
    crypto = _crypto()
    if isinstance(public_key, crypto.x25519.X25519PublicKey):
        return public_key.public_bytes(crypto.serialization.Encoding.Raw,
                                       crypto.serialization.PublicFormat.Raw)
    if isinstance(public_key, (bytes, bytearray)) and len(public_key) == 32:
        return bytes(public_key)
    raise SLDCValidationError("An X25519 public key is 32 raw bytes or a cryptography public key")


def x25519_recipient(public_key: Any, *, label: str | None = None,
                     recovery: bool = False) -> Recipient:
    """A recipient identified by an X25519 public key (32 raw bytes or a ``cryptography`` key)."""
    _recipient_common("recovery" if recovery else "primary", label)
    return Recipient("x25519", "recovery" if recovery else "primary", label,
                     public_key=_x25519_public_bytes(public_key))


def x25519_generate() -> tuple[bytes, bytes]:
    """A new X25519 key pair as raw ``(private_key, public_key)`` bytes.

    Keep the private half out of documents, logs and source control; the public half is what
    :func:`x25519_recipient` takes and what a sender may freely know.
    """
    crypto = _crypto()
    private = crypto.x25519.X25519PrivateKey.generate()
    return (private.private_bytes(crypto.serialization.Encoding.Raw,
                                  crypto.serialization.PrivateFormat.Raw,
                                  crypto.serialization.NoEncryption()),
            private.public_key().public_bytes(crypto.serialization.Encoding.Raw,
                                              crypto.serialization.PublicFormat.Raw))


def _x25519_private(private_key: Any) -> Any:
    crypto = _crypto()
    if isinstance(private_key, crypto.x25519.X25519PrivateKey):
        return private_key
    if isinstance(private_key, (bytes, bytearray)) and len(private_key) == 32:
        return crypto.x25519.X25519PrivateKey.from_private_bytes(bytes(private_key))
    raise SLDCValidationError("An X25519 private key is 32 raw bytes or a cryptography private key")


def _scrypt(password: str, salt: bytes, n: int) -> bytes:
    try:
        return hashlib.scrypt(unicodedata.normalize("NFKC", password).encode("utf-8"), salt=salt,
                              n=n, r=SCRYPT_R, p=SCRYPT_P, dklen=32,
                              maxmem=130 * n * SCRYPT_R + (1 << 20))
    except (AttributeError, ValueError) as exc:  # scrypt missing from this OpenSSL, or bad params
        raise SLDCDependencyError(f"scrypt is unavailable in this Python build: {exc}") from exc


def _x25519_kek(shared: bytes, ephemeral_public: bytes, recipient_public: bytes) -> bytes:
    crypto = _crypto()
    return crypto.HKDF(algorithm=crypto.hashes.SHA256(), length=32,
                       salt=ephemeral_public + recipient_public,
                       info=b"SLDC-X25519-KEK-V1\0").derive(shared)


def _wrap_aad(entry_id: str, kind: str) -> bytes:
    return b"SLDC-KEYWRAP-V1\0" + _canonical_json({"id": entry_id, "kind": kind})


def _envelope_aad(manifest: Mapping[str, Any]) -> bytes:
    encryption = manifest["encryption"]
    return b"SLDC-ENVELOPE-V1\0" + _canonical_json({
        "scheme": encryption["scheme"], "cipher": encryption["cipher"],
        "format": manifest["format"], "version": manifest["version"],
        "container": manifest["container"], "recovery_policy": encryption["recovery_policy"]})


def _wrap_key(recipient: Recipient, content_key: bytes, used_ids: set[str]) -> dict[str, Any]:
    crypto = _crypto()
    if recipient.kind == "password":
        if not isinstance(recipient.password, str):
            raise SLDCValidationError("A password recipient needs its password")
        _bounded_int(recipient.scrypt_n, MIN_SCRYPT_N, MAX_SCRYPT_N, "scrypt_n")
        salt = os.urandom(16)
        kek = _scrypt(recipient.password, salt, recipient.scrypt_n)
        entry: dict[str, Any] = {"kdf": {"name": "scrypt", "n": recipient.scrypt_n, "r": SCRYPT_R,
                                         "p": SCRYPT_P, "salt": _b64e(salt)}}
        prefix = "p"
    elif recipient.kind == "x25519":
        public = _x25519_public_bytes(recipient.public_key)
        ephemeral = crypto.x25519.X25519PrivateKey.generate()
        ephemeral_public = ephemeral.public_key().public_bytes(
            crypto.serialization.Encoding.Raw, crypto.serialization.PublicFormat.Raw)
        try:
            shared = ephemeral.exchange(crypto.x25519.X25519PublicKey.from_public_bytes(public))
        except ValueError as exc:  # A low-order (all-zero shared secret) public key.
            raise SLDCValidationError("That X25519 public key is not acceptable") from exc
        kek = _x25519_kek(shared, ephemeral_public, public)
        entry = {"key_id": key_fingerprint(public), "ephemeral": _b64e(ephemeral_public)}
        prefix = "x"
    else:
        raise SLDCValidationError(f"Unsupported recipient kind {recipient.kind!r}")
    _recipient_common(recipient.role, recipient.label)
    while True:
        entry_id = f"{prefix}-{os.urandom(6).hex()}"
        if entry_id not in used_ids:
            break
    nonce = os.urandom(12)
    entry.update(id=entry_id, kind=recipient.kind, role=recipient.role, nonce=_b64e(nonce),
                 wrapped=_b64e(crypto.AESGCM(kek).encrypt(nonce, content_key,
                                                         _wrap_aad(entry_id, recipient.kind))))
    entry["label"] = recipient.label
    if recipient.label is None:
        del entry["label"]
    used_ids.add(entry_id)
    return entry


def _check_recipient_set(entries: Sequence[Mapping[str, Any]], recovery_policy: str) -> None:
    if not _has(_RECOVERY_POLICIES, recovery_policy):
        raise SLDCValidationError(f"recovery_policy must be one of {sorted(_RECOVERY_POLICIES)}")
    if not 1 <= len(entries) <= MAX_RECIPIENTS:
        raise SLDCValidationError(f"An encrypted file needs 1 to {MAX_RECIPIENTS} recipients")
    has_recovery = any(entry["role"] == "recovery" for entry in entries)
    if recovery_policy == "recovery_recipient" and not has_recovery:
        raise SLDCValidationError("recovery_policy 'recovery_recipient' needs a recipient with role "
                                  "'recovery'")
    if recovery_policy == "none" and has_recovery:
        raise SLDCValidationError("A recovery recipient needs recovery_policy='recovery_recipient'; "
                                  "'none' means losing every credential loses the document")
    key_ids = [entry["key_id"] for entry in entries if entry["kind"] == "x25519"]
    if len(set(key_ids)) != len(key_ids):
        raise SLDCValidationError("The same X25519 key is listed twice")
    if len({entry["id"] for entry in entries}) != len(entries):
        raise SLDCValidationError("Recipient ids must be unique")


def _seal_envelope(manifest: dict[str, Any], payload: bytes) -> bytes:
    data = _write_zip(manifest, [("payload.bin", payload, zipfile.ZIP_STORED)])
    if len(data) > MAX_PACKAGE_BYTES:
        raise SLDCValidationError("The encrypted package exceeds 48 MiB; reduce its images")
    _load_container(data, read_parts=False)  # Never emit a file this module's own reader refuses.
    return data


def _encrypt(plain: bytes, recipients: Sequence[Recipient], recovery_policy: str) -> bytes:
    """Seal a plain archive for ``recipients``. Output is intentionally not deterministic."""
    crypto = _crypto()
    if isinstance(recipients, (str, bytes)) or not all(isinstance(r, Recipient) for r in recipients):
        raise SLDCValidationError("recipients must be a sequence of Recipient objects")
    content_key, nonce = os.urandom(32), os.urandom(12)
    used_ids: set[str] = set()
    entries = [_wrap_key(recipient, content_key, used_ids) for recipient in recipients]
    _check_recipient_set(entries, recovery_policy)
    manifest: dict[str, Any] = {
        "format": FORMAT_ID, "version": FORMAT_VERSION, "container": "encrypted",
        "encryption": {"scheme": ENCRYPTION_SCHEME, "cipher": _CIPHER, "nonce": _b64e(nonce),
                       "recovery_policy": recovery_policy, "ciphertext_size": len(plain) + 16,
                       "recipients": entries}}
    ciphertext = crypto.AESGCM(content_key).encrypt(nonce, plain, _envelope_aad(manifest))
    return _seal_envelope(manifest, ciphertext)


def _check_recipient_entry(entry: Any) -> None:
    if not isinstance(entry, dict) or not _has(_RECIPIENT_KEYS, entry.get("kind")):
        raise SLDCUnsupportedError("Unsupported recipient kind in an encrypted SLDC")
    if set(entry) - {"label"} != _RECIPIENT_KEYS[entry["kind"]] - {"label"} \
            or not set(entry) <= _RECIPIENT_KEYS[entry["kind"]]:
        raise SLDCValidationError(f"A {entry['kind']} recipient needs exactly "
                                  f"{sorted(_RECIPIENT_KEYS[entry['kind']] - {'label'})} (and an optional label)")
    _id_field(entry["id"], "recipient id", _ASSET_ID)
    if not _has(_ROLES, entry["role"]):
        raise SLDCValidationError(f"recipient role must be one of {sorted(_ROLES)}")
    _text_field(entry, "label", "recipient", 100)
    _b64d(entry["nonce"], "recipient nonce", length=12)
    _b64d(entry["wrapped"], "recipient wrapped key", length=48)
    if entry["kind"] == "password":
        kdf = entry["kdf"]
        if not isinstance(kdf, dict) or set(kdf) != _KDF_KEYS:
            raise SLDCValidationError(f"kdf needs exactly {sorted(_KDF_KEYS)}")
        if kdf["name"] != "scrypt":
            raise SLDCUnsupportedError(f"Unsupported key derivation {kdf['name']!r}")
        n = _bounded_int(kdf["n"], MIN_SCRYPT_N, MAX_SCRYPT_N, "scrypt n")
        if n & (n - 1) or kdf["r"] != SCRYPT_R or kdf["p"] != SCRYPT_P \
                or type(kdf["r"]) is not int or type(kdf["p"]) is not int:
            raise SLDCValidationError(f"scrypt needs a power-of-two n with r={SCRYPT_R}, p={SCRYPT_P}")
        _b64d(kdf["salt"], "scrypt salt", length=16)
    else:
        if not isinstance(entry["key_id"], str) or not _HEX_SHA256.fullmatch(entry["key_id"]):
            raise SLDCValidationError("recipient key_id must be 64 hex digits")
        _b64d(entry["ephemeral"], "recipient ephemeral key", length=32)


def _read_encrypted(archive: zipfile.ZipFile, infos: list[zipfile.ZipInfo],
                    manifest: dict[str, Any], read_parts: bool) -> _Loaded:
    """Check an encrypted envelope's structure without any key (and without ``cryptography``)."""
    if {key for key in manifest} != {"format", "version", "container", "encryption"}:
        raise SLDCValidationError("An encrypted SLDC manifest needs exactly format, version, container and encryption")
    if [info.filename for info in infos] != list(_ENVELOPE_MEMBERS):
        raise SLDCValidationError("An encrypted SLDC holds only mimetype, manifest.json, payload.bin")
    encryption = manifest["encryption"]
    if not isinstance(encryption, dict) or set(encryption) != _ENCRYPTION_KEYS:
        raise SLDCValidationError(f"encryption needs exactly {sorted(_ENCRYPTION_KEYS)}")
    if encryption["scheme"] != ENCRYPTION_SCHEME or encryption["cipher"] != _CIPHER:
        raise SLDCUnsupportedError(f"Unsupported encryption scheme {encryption['scheme']!r} "
                                   f"/ cipher {encryption['cipher']!r}")
    _b64d(encryption["nonce"], "encryption nonce", length=12)
    size = encryption["ciphertext_size"]
    if type(size) is not int or size != infos[2].file_size or not 16 <= size <= MAX_PACKAGE_BYTES:
        raise SLDCValidationError("Ciphertext size does not match the payload entry")
    recipients = encryption["recipients"]
    if not isinstance(recipients, list):
        raise SLDCValidationError("encryption recipients must be a list")
    for entry in recipients:
        _check_recipient_entry(entry)
    _check_recipient_set(recipients, encryption["recovery_policy"])
    if not read_parts:
        return _Loaded("encrypted", manifest, {})
    return _Loaded("encrypted", manifest, {"payload.bin": _read_member(archive, infos[2], size)})


def _describe_encryption(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """What an encrypted file discloses about itself (no keys, no wrapped material)."""
    encryption = manifest["encryption"]
    recipients = []
    for entry in encryption["recipients"]:
        described = {"id": entry["id"], "kind": entry["kind"], "role": entry["role"]}
        if "label" in entry:
            described["label"] = entry["label"]
        if entry["kind"] == "x25519":
            described["key_id"] = entry["key_id"]
        else:
            described["scrypt_n"] = entry["kdf"]["n"]
        recipients.append(described)
    return {"scheme": encryption["scheme"], "cipher": encryption["cipher"],
            "recovery_policy": encryption["recovery_policy"],
            "ciphertext_size": encryption["ciphertext_size"], "recipients": recipients}


def _unwrap_content_key(manifest: Mapping[str, Any], password: str | None,
                        private_key: Any) -> tuple[bytes, dict[str, Any]]:
    """Find the content key for the supplied credential. Every failure looks the same."""
    crypto = _crypto()
    if password is None and private_key is None:
        raise SLDCDecryptionError("This SLDC is encrypted: supply a password or a private key")
    entries = manifest["encryption"]["recipients"]
    try:
        key = _x25519_private(private_key) if private_key is not None else None
    except (SLDCValidationError, ValueError):
        key = None  # Unusable key material is just a credential that opens nothing.
    own_public = None
    if key is not None:
        own_public = key.public_key().public_bytes(crypto.serialization.Encoding.Raw,
                                                  crypto.serialization.PublicFormat.Raw)
    for entry in entries:
        try:
            nonce, wrapped = _b64d(entry["nonce"], "nonce"), _b64d(entry["wrapped"], "wrapped")
            if entry["kind"] == "password" and isinstance(password, str):
                kdf = entry["kdf"]
                kek = _scrypt(password, _b64d(kdf["salt"], "salt"), kdf["n"])
            elif entry["kind"] == "x25519" and key is not None and own_public is not None \
                    and entry["key_id"] == key_fingerprint(own_public):
                ephemeral_public = _b64d(entry["ephemeral"], "ephemeral", length=32)
                assert key is not None
                assert own_public is not None
                shared = key.exchange(crypto.x25519.X25519PublicKey.from_public_bytes(ephemeral_public))
                kek = _x25519_kek(shared, ephemeral_public, own_public)
            else:
                continue
            content_key = crypto.AESGCM(kek).decrypt(nonce, wrapped,
                                                    _wrap_aad(entry["id"], entry["kind"]))
        except (crypto.InvalidTag, ValueError):
            continue
        return content_key, entry
    raise SLDCDecryptionError(_GENERIC_DECRYPT_FAILURE)


def _open_envelope(loaded: _Loaded, password: str | None, private_key: Any) -> tuple[bytes, bytes, dict[str, Any]]:
    crypto = _crypto()
    content_key, entry = _unwrap_content_key(loaded.manifest, password, private_key)
    try:
        plain = crypto.AESGCM(content_key).decrypt(
            _b64d(loaded.manifest["encryption"]["nonce"], "nonce"), loaded.parts["payload.bin"],
            _envelope_aad(loaded.manifest))
    except crypto.InvalidTag as exc:
        raise SLDCDecryptionError(_GENERIC_DECRYPT_FAILURE) from exc
    return plain, content_key, entry


def _decrypt(loaded: _Loaded, *, password: str | None, private_key: Any) -> bytes:
    """The plain archive inside an encrypted envelope (still to be parsed and validated)."""
    return _open_envelope(loaded, password, private_key)[0]


def _load_envelope(data: bytes | bytearray | memoryview) -> _Loaded:
    loaded = _load_container(bytes(data))
    if loaded.kind != "encrypted":
        raise SLDCValidationError("This SLDC is not encrypted")
    return loaded


def rekey(data: bytes | bytearray | memoryview, recipients: Sequence[Recipient], *,
          recovery_policy: str = "none", password: str | None = None,
          private_key: Any = None) -> bytes:
    """Re-encrypt an encrypted file for a new set of ``recipients`` under a fresh content key.

    Use it to remove a recipient or change the recovery policy. This protects the *new* copy
    only: whoever already held the old file and a working credential can still open that copy,
    and may have kept the plaintext. Passing the credential of any current recipient is required.
    """
    loaded = _load_envelope(data)
    plain = _open_envelope(loaded, password, private_key)[0]
    return _encrypt(plain, recipients, recovery_policy)


def add_recipient(data: bytes | bytearray | memoryview, recipient: Recipient, *,
                  password: str | None = None, private_key: Any = None) -> bytes:
    """Add one recipient to an encrypted file without re-encrypting its content.

    A current recipient's ``password`` or ``private_key`` is needed, because the content key is
    re-wrapped for the newcomer. The ciphertext is verified first and left byte-for-byte as it
    was. A ``recovery`` recipient changes the recovery policy, which is authenticated, so that
    case uses :func:`rekey` instead.
    """
    loaded = _load_envelope(data)
    _, content_key, _ = _open_envelope(loaded, password, private_key)
    encryption = loaded.manifest["encryption"]
    if recipient.role == "recovery" and encryption["recovery_policy"] != "recovery_recipient":
        raise SLDCValidationError("Adding a recovery recipient changes the recovery policy: use rekey()")
    manifest = copy.deepcopy(loaded.manifest)
    entries = manifest["encryption"]["recipients"]
    entries.append(_wrap_key(recipient, content_key, {e["id"] for e in entries}))
    _check_recipient_set(entries, encryption["recovery_policy"])
    return _seal_envelope(manifest, loaded.parts["payload.bin"])


# ----------------------------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------------------------
def _write_stdout(text: str) -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass
    sys.stdout.write(text if text.endswith("\n") else text + "\n")


def _password_from_env(name: str | None) -> str | None:
    """Passwords come from the environment, never from argv (visible to other users, in history)."""
    if name is None:
        return None
    value = os.environ.get(name)
    if not value:
        raise SLDCValidationError(f"Environment variable {name} is not set")
    return value


def _write_report(path: str, report: Mapping[str, Any], *, overwrite: bool) -> None:
    text = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    with open(path, "w" if overwrite else "x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sldc", allow_abbrev=False,
                                     description="Inspect, pack, convert and verify SLDC files.")
    commands = parser.add_subparsers(dest="command", required=True)

    def add_command(name: str, **kwargs: Any) -> argparse.ArgumentParser:
        return commands.add_parser(name, allow_abbrev=False, **kwargs)  # --password is not --password-env

    def with_password(command: argparse.ArgumentParser) -> None:
        command.add_argument("--password-env", metavar="VAR", help="name of an environment variable holding the document password")

    check = add_command("check", help="verify structure, integrity and content")
    check.add_argument("paths", nargs="+")
    with_password(check)
    info = add_command("info", help="describe a file from its manifest alone (JSON)")
    info.add_argument("path")
    text = add_command("text", help="print the plain text of a document")
    text.add_argument("path")
    text.add_argument("--markdown", action="store_true", help="render as Markdown instead")
    with_password(text)
    pack = add_command("pack", help="create a .sldc from a text or Markdown file")
    pack.add_argument("source")
    pack.add_argument("destination")
    pack.add_argument("--title")
    pack.add_argument("--language", default="en")
    pack.add_argument("--markdown", action="store_true", default=None, help="parse Markdown (default: on for .md/.markdown files)")
    pack.add_argument("--legacy", action="store_true", help=f"write the frozen {LEGACY_VERSION} format")
    pack.add_argument("--overwrite", action="store_true")
    migrate = add_command("migrate", help=f"convert a {LEGACY_VERSION} file to {FORMAT_VERSION} (the source is never modified)")
    export = add_command("export-legacy", help=f"make a lossy {LEGACY_VERSION} copy of a "
                                 f"{FORMAT_VERSION} file")
    for command in (migrate, export):
        command.add_argument("source")
        command.add_argument("destination")
        command.add_argument("--report", metavar="FILE", help="write the JSON report here")
        command.add_argument("--overwrite", action="store_true")
    with_password(export)
    verify = add_command(
        "verify", help="check signatures; exit status 0 only if every signature is valid and trusted")
    verify.add_argument("path")
    verify.add_argument("--trusted", nargs="*", default=[], metavar="KEY_ID",
                        help="key ids (see key_fingerprint) you have decided to trust")
    verify.add_argument("--revoked", nargs="*", default=[], metavar="KEY_ID")
    with_password(verify)
    return parser


def _cmd_check(args: argparse.Namespace) -> int:
    password = _password_from_env(args.password_env)
    failures = 0
    for path in args.paths:
        try:
            package = read_package(path, password=password)
        except (SLDCError, OSError) as exc:
            failures += 1
            print(f"FAIL {path}: {exc}")
            continue
        extra = f" signatures={len(package.signatures)}" if package.signatures else ""
        print(f"OK   {path}: version={package.version} id={package.document['id']} "
              f"blocks={len(package.document['blocks'])} sha256={document_digest(package)}{extra}")
    return 1 if failures else 0


def _cmd_verify(args: argparse.Namespace) -> int:
    package = read_package(args.path, password=_password_from_env(args.password_env))
    results = verify_package(package, trusted_keys=args.trusted, revoked_keys=args.revoked)
    if not results:
        print(f"{args.path}: no signatures")
        return 1
    for result in results:
        print(f"{result['state']:<22} {result['id']} key={result['key_id'][:16]} "
              f"purpose={result['purpose']} claimed_time={result['signed_at_claimed']}")
        for problem in result["problems"]:
            print(f"    - {problem}")
    return 0 if all(r["state"] == "valid_trusted" for r in results) else 1


def _cmd_pack(args: argparse.Namespace) -> int:
    source = Path(args.source)
    as_markdown = (source.suffix.lower() in {".md", ".markdown"}
                   if args.markdown is None else args.markdown)
    raw = source.read_bytes()
    document = document_from_text(
        raw.decode("utf-8-sig"), title=args.title or source.stem, language=args.language,
        markdown=as_markdown, source_ref=source.name,
        provenance=[provenance_entry("sldc.pack", source=source.name, raw_sha256=_sha256(raw))])
    if args.legacy:
        print(write_sldc(args.destination, document, overwrite=args.overwrite))
    else:
        package = SLDCPackage({"schema": SCHEMA, **document})
        print(write_package(args.destination, package, overwrite=args.overwrite))
    return 0


def _cmd_convert(args: argparse.Namespace) -> int:
    if args.command == "migrate":
        report = migrate_file(args.source, args.destination, overwrite=args.overwrite)
    else:
        package = read_package(args.source, password=_password_from_env(args.password_env))
        legacy, report = export_legacy(package)
        write_package(args.destination, legacy, overwrite=args.overwrite)
    if args.report:
        _write_report(args.report, report, overwrite=args.overwrite)
    print(args.destination)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "check":
            return _cmd_check(args)
        if args.command == "info":
            _write_stdout(json.dumps(inspect_package(args.path), indent=2, sort_keys=True))
        elif args.command == "text":
            package = read_package(args.path, password=_password_from_env(args.password_env))
            _write_stdout(to_markdown(package, validate=False) if args.markdown
                          else extract_text(package, validate=False))
        elif args.command == "pack":
            return _cmd_pack(args)
        elif args.command == "verify":
            return _cmd_verify(args)
        else:
            return _cmd_convert(args)
    except (SLDCError, OSError, UnicodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


__all__ = [
    # Identity, versions, limits
    "FORMAT_ID", "FORMAT_VERSION", "LATEST_KNOWN_VERSION", "LEGACY_VERSION", "SCHEMA", "MIMETYPE", "EXTENSIONS", "MEMBERS",
    "BLOCK_TYPES", "BLOCK_TYPES_V1", "SUPPORTED_FEATURES", "ENCRYPTION_SCHEME", "SIGNATURE_SCHEMA",
    "COMMENTS_SCHEMA", "MAX_ARCHIVE_BYTES", "MAX_PACKAGE_BYTES", "MAX_DOCUMENT_BYTES", "MAX_BLOCKS",
    "MAX_TEXT_CHARS", "MAX_ASSETS", "MAX_ASSET_BYTES", "MAX_SIGNATURES", "MAX_RECIPIENTS",
    # Errors and types
    "SLDCError", "SLDCValidationError", "SLDCUnsupportedError", "SLDCIntegrityError",
    "SLDCDecryptionError", "SLDCDependencyError", "SLDCBlock", "SLDCDocument", "SLDCPackage",
    "Asset", "Recipient", "Signer",
    # Packages (1.x)
    "read_package", "write_package", "loads_package", "dumps_package", "inspect_package",
    "newest_version", "preview_package", "preview_to_package",
    "validate_package", "new_package", "make_asset", "add_asset", "prune_assets", "make_table",
    "make_figure", "revision_id", "make_equation", "make_inline_equation", "make_inline_image", "revision_id",
    # Migration
    "migrate_package", "migrate_file", "export_legacy",
    # Collaboration
    "new_block_id", "next_revision", "diff_documents", "merge_documents", "merge_packages",
    "add_comment", "reply_to_thread", "set_thread_status", "comment_anchor_status",
    # Signing and encryption
    "Ed25519Signer", "sign_package", "verify_package", "key_fingerprint", "password_recipient",
    "x25519_recipient", "x25519_generate", "rekey", "add_recipient",
    # Version 0.1 (frozen)
    "dumps_sldc", "loads_sldc", "read_sldc", "write_sldc",
    # Both versions
    "validate_sldc", "extract_text", "sniff_sldc", "is_sldc", "new_document", "make_block",
    "provenance_entry", "blocks_from_text", "document_from_text", "document_from_reader_result",
    "document_digest", "block_digest", "block_index", "block_text", "chunk_document", "to_markdown",
]


if __name__ == "__main__":
    raise SystemExit(main())
