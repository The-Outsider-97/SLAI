"""Production-grade Integrity Engine for the Reader subsystem.

The integrity engine decides whether a file or an artifact can be trusted, and hosts
the safety guard that every container-reading component shares.

Design goals
------------
- One expansion guard for every ZIP/gzip/bzip2/xz reader. A compressed size limit says
  nothing about what a file expands to, so member counts, declared and *actual*
  uncompressed bytes and compression ratios are bounded in one place.
- Validate structure, not just existence: package members, ZIP CRCs, PDF trailer and
  (when ``pypdf`` is present) page tree, well-formed XML/JSON, tar member paths.
- Be honest about depth. ``deep=False`` reads only headers and trailers; ``deep=True``
  reads whole files within the configured limits. Checks that could not run are
  reported as skipped, never as passed.
- Treat validation findings as findings, not failures of the engine: ``validate_file``
  returns a report; ``require_valid`` raises ``IntegrityError`` for callers that need to
  stop.
- Reuse ``DetectionEngine`` for format identification and ``reader_helpers.py`` for
  hashing, decoding, results and config validation.
"""

from __future__ import annotations

import bz2
import csv
import gzip
import importlib
import io
import json
import lzma
import tarfile
import zipfile
import zlib

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence
from xml.etree import ElementTree as ET

from .utils.config_loader import get_config_section, load_global_config
from .utils.reader_error import *
from .utils.reader_helpers import *
from .detection_engine import DetectionEngine
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Integrity Engine")
printer = PrettyPrinter()

INTEGRITY_ENGINE_VERSION = "1.0.0"

_KNOWN_CONFIG_KEYS = (
    "max_zip_members",
    "max_member_bytes",
    "max_total_bytes",
    "max_compression_ratio",
    "max_validate_bytes",
    "max_xml_parse_bytes",
    "deep_by_default",
    "require_nonempty_outputs",
    "verify_outputs",
    "reject_xml_entities",
)

# Members that must exist for a package to be usable, by format id.
_REQUIRED_MEMBERS: Dict[str, Sequence[str]] = {
    "docx": ("[Content_Types].xml", "word/document.xml"),
    "xlsx": ("[Content_Types].xml", "xl/workbook.xml"),
    "xlsm": ("[Content_Types].xml", "xl/workbook.xml"),
    "pptx": ("[Content_Types].xml", "ppt/presentation.xml"),
    "odt": ("mimetype", "content.xml"),
    "ods": ("mimetype", "content.xml"),
    "odp": ("mimetype", "content.xml"),
    "epub": ("mimetype", "META-INF/container.xml"),
}

# The one XML part per format that is parsed for well-formedness in deep mode.
_MAIN_XML_PART: Dict[str, str] = {
    "docx": "word/document.xml",
    "xlsx": "xl/workbook.xml",
    "xlsm": "xl/workbook.xml",
    "pptx": "ppt/presentation.xml",
    "odt": "content.xml",
    "ods": "content.xml",
    "odp": "content.xml",
    "epub": "META-INF/container.xml",
}

# Public aliases: SalvageEngine and CompressionEngine use the same package definitions.
PACKAGE_REQUIRED_MEMBERS = _REQUIRED_MEMBERS
PACKAGE_MAIN_XML_PART = _MAIN_XML_PART

_STREAM_OPENERS = {"gzip": gzip.open, "bzip2": bz2.open, "xz": lzma.open}
_READ_CHUNK = 1024 * 1024


def unsafe_member_name(name: str) -> bool:
    """True for archive member names that could escape an extraction directory (zip-slip)."""

    normalized = str(name).replace("\\", "/")
    parts = [part for part in normalized.split("/") if part not in ("", ".")]
    return normalized.startswith("/") or ".." in parts or (len(normalized) > 1 and normalized[1] == ":") or "\x00" in normalized


# ---------------------------------------------------------------------------
# Shared expansion guard
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExpansionLimits:
    """Upper bounds on what a compressed container may expand to."""

    max_members: int = 10_000
    max_member_bytes: int = 256 * 1024 * 1024
    max_total_bytes: int = 1024 * 1024 * 1024
    max_ratio: float = 1000.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_members": self.max_members,
            "max_member_bytes": self.max_member_bytes,
            "max_total_bytes": self.max_total_bytes,
            "max_ratio": self.max_ratio,
        }


class ExpansionGuard:
    """Bounds ZIP and stream expansion. ``0`` disables the corresponding limit.

    Declared sizes in a ZIP header are written by whoever made the file, so ``inspect_zip``
    only rejects the obvious; every real read must go through ``read_member`` or
    ``drain_stream``, which count the bytes that actually come out.
    """

    def __init__(self, limits: Optional[ExpansionLimits] = None) -> None:
        self.limits = limits or ExpansionLimits()

    @staticmethod
    def _violation(source: str, message: str, **context: Any) -> IntegrityError:
        return IntegrityError(source, message, context=context)

    def inspect_zip(self, archive: zipfile.ZipFile, *, source: str) -> Dict[str, Any]:
        """Check declared member count, sizes and ratios. Raises ``IntegrityError`` on violation."""

        limits = self.limits
        members = archive.infolist()
        if limits.max_members and len(members) > limits.max_members:
            raise self._violation(source, "Archive has too many members", members=len(members), max_members=limits.max_members)
        declared_total = 0
        worst_ratio = 0.0
        for member in members:
            declared_total += member.file_size
            if limits.max_member_bytes and member.file_size > limits.max_member_bytes:
                raise self._violation(
                    source, "Archive member expands beyond the configured limit",
                    member=member.filename, declared_bytes=member.file_size, max_member_bytes=limits.max_member_bytes,
                )
            ratio = member.file_size / max(1, member.compress_size) if member.file_size else 0.0
            worst_ratio = max(worst_ratio, ratio)
            if limits.max_ratio and ratio > limits.max_ratio and member.file_size > 1024 * 1024:
                raise self._violation(
                    source, "Archive member has a suspicious compression ratio",
                    member=member.filename, ratio=round(ratio, 1), max_ratio=limits.max_ratio,
                )
        if limits.max_total_bytes and declared_total > limits.max_total_bytes:
            raise self._violation(source, "Archive expands beyond the configured total limit",
                                  declared_bytes=declared_total, max_total_bytes=limits.max_total_bytes)
        return {"members": len(members), "declared_total_bytes": declared_total, "worst_ratio": round(worst_ratio, 2)}

    def read_member(self, archive: zipfile.ZipFile, name: str, *, source: str, limit: Optional[int] = None) -> bytes:
        """Read one member, never holding more than ``limit`` bytes (default: ``max_member_bytes``)."""

        cap = self.limits.max_member_bytes if limit is None else int(limit)
        try:
            with archive.open(name) as handle:
                data = handle.read(cap + 1) if cap else handle.read()
        except KeyError as exc:
            raise IntegrityError(source, "Archive member is missing", context={"member": name}, cause=exc) from exc
        except (zipfile.BadZipFile, EOFError, OSError, RuntimeError, NotImplementedError, zlib.error) as exc:
            raise IntegrityError(source, "Archive member could not be read", context={"member": name}, cause=exc) from exc
        if cap and len(data) > cap:
            raise self._violation(source, "Archive member exceeds the read limit while reading", member=name, limit=cap)
        return data

    def drain_stream(self, handle: Any, *, source: str, limit: Optional[int] = None) -> int:
        """Consume a (de)compressing stream completely, verifying its trailer. Returns bytes read."""

        cap = self.limits.max_total_bytes if limit is None else int(limit)
        total = 0
        try:
            while True:
                chunk = handle.read(_READ_CHUNK)
                if not chunk:
                    break
                total += len(chunk)
                if cap and total > cap:
                    raise self._violation(source, "Compressed stream expands beyond the configured limit", limit=cap)
        except IntegrityError:
            raise
        except (OSError, EOFError, ValueError, lzma.LZMAError, zlib.error) as exc:
            raise IntegrityError(source, "Compressed stream is damaged or truncated", cause=exc) from exc
        return total


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class IntegrityEngine:
    """Validates files and artifacts and hosts the shared expansion guard."""

    def __init__(self, detector: Optional[DetectionEngine] = None) -> None:
        self.config = load_global_config()
        self.ie_config = get_config_section("integrity_engine") or {}
        warn_unknown_config_keys("integrity_engine", self.ie_config, _KNOWN_CONFIG_KEYS)

        self.limits = ExpansionLimits(
            max_members=coerce_int(self.ie_config.get("max_zip_members"), 10_000, minimum=0),
            max_member_bytes=coerce_int(self.ie_config.get("max_member_bytes"), 256 * 1024 * 1024, minimum=0),
            max_total_bytes=coerce_int(self.ie_config.get("max_total_bytes"), 1024 * 1024 * 1024, minimum=0),
            max_ratio=coerce_float(self.ie_config.get("max_compression_ratio"), 1000.0, minimum=0.0),
        )
        self.guard = ExpansionGuard(self.limits)
        self.max_validate_bytes = coerce_int(self.ie_config.get("max_validate_bytes"), 64 * 1024 * 1024, minimum=1024)
        self.max_xml_parse_bytes = coerce_int(self.ie_config.get("max_xml_parse_bytes"), 32 * 1024 * 1024, minimum=1024)
        self.deep_by_default = coerce_bool(self.ie_config.get("deep_by_default"), False)
        self.require_nonempty_outputs = coerce_bool(self.ie_config.get("require_nonempty_outputs"), True)
        self.verify_outputs = coerce_bool(self.ie_config.get("verify_outputs"), True)
        self.reject_xml_entities = coerce_bool(self.ie_config.get("reject_xml_entities"), True)

        self.detector = detector or DetectionEngine()
        self.settings = {
            **self.limits.to_dict(),
            "max_validate_bytes": self.max_validate_bytes,
            "max_xml_parse_bytes": self.max_xml_parse_bytes,
            "deep_by_default": self.deep_by_default,
            "require_nonempty_outputs": self.require_nonempty_outputs,
            "verify_outputs": self.verify_outputs,
            "reject_xml_entities": self.reject_xml_entities,
        }
        logger.info("Integrity Engine initialized | deep_by_default=%s verify_outputs=%s", self.deep_by_default, self.verify_outputs)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def capabilities(self) -> Dict[str, Any]:
        return {
            "engine_version": INTEGRITY_ENGINE_VERSION,
            "checked_formats": sorted(set(_REQUIRED_MEMBERS) | {"pdf", "json", "xml", "csv", "txt", "md", "html", "gzip", "bzip2", "xz", "tar", "tar_gz", "zip"}),
            "deep_checks": ["zip_crc", "main_xml_part", "pdf_pages", "compressed_stream_trailer", "tar_members"],
            "pdf_deep_backend_available": self._pypdf() is not None,
            "settings": dict(self.settings),
        }

    def describe(self) -> Dict[str, Any]:
        """Return a safe operational description useful for health checks."""

        return {"status": "ok", **self.capabilities()}

    # ------------------------------------------------------------------
    # Check plumbing
    # ------------------------------------------------------------------

    @staticmethod
    def _check(checks: List[Dict[str, Any]], name: str, ok: bool, detail: Any = None, *, severity: str = "error",
               skipped: bool = False) -> bool:
        checks.append({
            "name": name,
            "ok": bool(ok),
            "severity": severity,
            "skipped": bool(skipped),
            "detail": detail if detail is None or isinstance(detail, (str, int, float, bool)) else json_safe(detail),
        })
        return bool(ok)

    @staticmethod
    def _status(checks: Sequence[Mapping[str, Any]]) -> str:
        if any(not item["ok"] and item["severity"] == "error" for item in checks):
            return "failed"
        if any(item["severity"] == "warning" and (not item["ok"] or item.get("skipped")) for item in checks):
            return "degraded"
        return "ok"

    @staticmethod
    def _pypdf() -> Optional[Any]:
        try:
            return importlib.import_module("pypdf")
        except ImportError:
            return None

    # ------------------------------------------------------------------
    # Per-family validators
    # ------------------------------------------------------------------

    def _check_zip(self, path: Path, format_id: str, deep: bool, checks: List[Dict[str, Any]]) -> None:
        source = str(path)
        try:
            archive = zipfile.ZipFile(path)
        except (zipfile.BadZipFile, OSError, EOFError, ValueError, zlib.error) as exc:
            self._check(checks, "zip_central_directory", False, f"{type(exc).__name__}: {exc}")
            return
        with archive:
            self._check(checks, "zip_central_directory", True)
            try:
                summary = self.guard.inspect_zip(archive, source=source)
                self._check(checks, "expansion_limits", True, summary)
            except IntegrityError as exc:
                self._check(checks, "expansion_limits", False, exc.context)
                return
            names = {member.filename for member in archive.infolist()}
            required = _REQUIRED_MEMBERS.get(format_id)
            if required:
                missing = [name for name in required if name not in names]
                self._check(checks, "required_members", not missing, {"missing": missing} if missing else None)
            if any(member.flag_bits & 0x1 for member in archive.infolist()):
                self._check(checks, "members_not_encrypted", False, "encrypted members cannot be validated", severity="warning")
            if not deep:
                self._check(checks, "zip_crc", True, "not run (deep=False)", severity="info", skipped=True)
                return
            total = 0
            try:
                for member in archive.infolist():
                    if member.is_dir() or member.flag_bits & 0x1:
                        continue
                    with archive.open(member) as handle:
                        total += self.guard.drain_stream(handle, source=source, limit=self.limits.max_member_bytes or None)
                    if self.limits.max_total_bytes and total > self.limits.max_total_bytes:
                        raise IntegrityError(source, "Archive expands beyond the configured total limit while reading",
                                             context={"max_total_bytes": self.limits.max_total_bytes})
                self._check(checks, "zip_crc", True, {"bytes_read": total})
            except IntegrityError as exc:
                self._check(checks, "zip_crc", False, str(exc))
                return
            except (zipfile.BadZipFile, RuntimeError, NotImplementedError, zlib.error) as exc:
                self._check(checks, "zip_crc", False, f"{type(exc).__name__}: {exc}")
                return
            part = _MAIN_XML_PART.get(format_id)
            if part and part in names:
                declared = archive.getinfo(part).file_size
                if declared > self.max_xml_parse_bytes:
                    self._check(checks, "main_xml_well_formed", True, f"skipped: {part} is {declared} bytes", severity="warning", skipped=True)
                else:
                    try:
                        data = self.guard.read_member(archive, part, source=source, limit=self.max_xml_parse_bytes)
                        ET.fromstring(data)
                        self._check(checks, "main_xml_well_formed", True, part)
                    except ET.ParseError as exc:
                        self._check(checks, "main_xml_well_formed", False, f"{part}: {exc}")
                    except IntegrityError as exc:
                        self._check(checks, "main_xml_well_formed", False, str(exc))

    def _check_pdf(self, path: Path, size: int, deep: bool, checks: List[Dict[str, Any]]) -> None:
        head = read_file_head(path, 1024)
        self._check(checks, "pdf_header", b"%PDF-" in head)
        tail_size = min(size, 2048)
        try:
            with path.open("rb") as handle:
                handle.seek(max(0, size - tail_size))
                tail = handle.read(tail_size)
        except OSError as exc:
            self._check(checks, "pdf_trailer", False, f"{type(exc).__name__}: {exc}")
            return
        self._check(checks, "pdf_eof_marker", b"%%EOF" in tail, "truncated files usually lack the %%EOF marker")
        self._check(checks, "pdf_startxref", b"startxref" in tail, severity="warning")
        if not deep:
            self._check(checks, "pdf_page_tree", True, "not run (deep=False)", severity="info", skipped=True)
            return
        pypdf = self._pypdf()
        if pypdf is None:
            self._check(checks, "pdf_page_tree", True, "skipped: optional 'pypdf' package is not installed", severity="warning", skipped=True)
            return
        try:
            reader = pypdf.PdfReader(str(path), strict=False)
            if getattr(reader, "is_encrypted", False):
                self._check(checks, "pdf_not_encrypted", False, "encrypted PDF cannot be read without a password", severity="warning")
                return
            pages = len(reader.pages)
            self._check(checks, "pdf_page_tree", pages > 0, {"pages": pages})
        except Exception as exc:  # pypdf raises many unrelated exception types for damaged files
            self._check(checks, "pdf_page_tree", False, f"{type(exc).__name__}: {exc}")

    def _check_stream(self, path: Path, format_id: str, deep: bool, checks: List[Dict[str, Any]]) -> None:
        opener = _STREAM_OPENERS.get("gzip" if format_id == "tar_gz" else format_id)
        if opener is None:
            return
        if not deep:
            self._check(checks, "stream_trailer", True, "not run (deep=False)", severity="info", skipped=True)
            return
        try:
            with opener(path, "rb") as handle:
                total = self.guard.drain_stream(handle, source=str(path))
            self._check(checks, "stream_trailer", True, {"decompressed_bytes": total})
        except IntegrityError as exc:
            self._check(checks, "stream_trailer", False, str(exc))
        except (OSError, EOFError, lzma.LZMAError, ValueError, zlib.error) as exc:
            self._check(checks, "stream_trailer", False, f"{type(exc).__name__}: {exc}")

    @staticmethod
    def _unsafe_member_name(name: str) -> bool:
        return unsafe_member_name(name)

    def _check_tar(self, path: Path, format_id: str, deep: bool, checks: List[Dict[str, Any]]) -> None:
        if not deep:
            self._check(checks, "tar_members", True, "not run (deep=False)", severity="info", skipped=True)
            return
        unsafe: List[str] = []
        links: List[str] = []
        count = 0
        declared = 0
        try:
            with tarfile.open(path, "r:*") as archive:
                for member in archive:
                    count += 1
                    declared += max(0, member.size)
                    if self.limits.max_members and count > self.limits.max_members:
                        self._check(checks, "tar_members", False, f"more than {self.limits.max_members} members")
                        return
                    if self._unsafe_member_name(member.name):
                        unsafe.append(member.name)
                    if member.issym() or member.islnk() or member.isdev():
                        links.append(member.name)
        except (tarfile.TarError, OSError, EOFError, ValueError, zlib.error, lzma.LZMAError) as exc:
            self._check(checks, "tar_members", False, f"{type(exc).__name__}: {exc}")
            return
        self._check(checks, "tar_members", True, {"members": count, "declared_bytes": declared})
        self._check(checks, "tar_member_paths", not unsafe, {"unsafe": unsafe[:10]} if unsafe else None)
        if links:
            self._check(checks, "tar_special_members", False, {"links_or_devices": links[:10]}, severity="warning")
        if self.limits.max_total_bytes and declared > self.limits.max_total_bytes:
            self._check(checks, "expansion_limits", False, {"declared_bytes": declared, "max_total_bytes": self.limits.max_total_bytes})

    def _check_text_like(self, path: Path, size: int, format_id: str, checks: List[Dict[str, Any]]) -> None:
        if size > self.max_validate_bytes:
            self._check(checks, "content_parse", True, f"skipped: {size} bytes exceeds max_validate_bytes", severity="warning", skipped=True)
            return
        try:
            raw = read_file_bytes(path, allowed_extensions={path.suffix.lower() or ".txt"}, max_file_size_bytes=0)
        except ReaderError as exc:
            self._check(checks, "content_readable", False, str(exc))
            return
        decoded = decode_bytes(raw, source=str(path))
        self._check(checks, "content_decodes", True, decoded.encoding)
        if decoded.warnings:
            self._check(checks, "content_decode_warnings", False, list(decoded.warnings), severity="warning")
        text = decoded.text
        if format_id == "json":
            try:
                json.loads(text)
                self._check(checks, "json_well_formed", True)
            except (ValueError, RecursionError) as exc:
                self._check(checks, "json_well_formed", False, f"{type(exc).__name__}: {exc}")
        elif format_id in {"xml", "svg", "fb2"}:
            if self.reject_xml_entities and b"<!ENTITY" in raw[:self.max_validate_bytes]:
                self._check(checks, "xml_no_entity_declarations", False, "entity declarations are rejected (expansion attacks)")
                return
            try:
                ET.fromstring(raw)
                self._check(checks, "xml_well_formed", True)
            except ET.ParseError as exc:
                self._check(checks, "xml_well_formed", False, str(exc))
        elif format_id in {"jsonl", "ndjson"}:
            bad_lines: List[int] = []
            for number, line in enumerate(text.splitlines(), start=1):
                if line.strip():
                    try:
                        json.loads(line)
                    except ValueError:
                        bad_lines.append(number)
                        if len(bad_lines) >= 20:
                            break
            self._check(checks, "jsonl_lines_valid", not bad_lines, {"invalid_lines": bad_lines} if bad_lines else None)
        elif format_id in {"csv", "tsv"}:
            try:
                delimiter = "\t" if format_id == "tsv" else ","
                widths = set()
                rows = 0
                for row in csv.reader(io.StringIO(text), delimiter=delimiter):
                    rows += 1
                    widths.add(len(row))
                    if rows >= 10_000:
                        break
                self._check(checks, "csv_parses", True, {"rows_checked": rows})
                try:
                    for _row in csv.reader(io.StringIO(text), delimiter=delimiter, strict=True):
                        pass
                except csv.Error as exc:
                    # Typically an unterminated quote: the file was cut mid-record (or has stray quotes).
                    self._check(checks, "csv_strict_parse", False, str(exc), severity="warning")
                if len(widths) > 1:
                    self._check(checks, "csv_consistent_columns", False, {"column_counts": sorted(widths)[:5]}, severity="warning")
            except csv.Error as exc:
                self._check(checks, "csv_parses", False, str(exc))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate_file(
        self,
        file_path: str | Path,
        *,
        expected_format: Optional[str] = None,
        deep: Optional[bool] = None,
        compute_hash: bool = True,
        require_nonempty: bool = False,
    ) -> Dict[str, Any]:
        """Validate one file and return a report. Findings never raise; only unusable input does."""

        timer = OperationTimer("validate")
        path = expand_path(file_path)
        if not path.exists():
            raise FileMissingError(str(path))
        if not path.is_file():
            raise FileNotRegularError(str(path))
        deep_mode = self.deep_by_default if deep is None else bool(deep)
        try:
            size = int(path.stat().st_size)
        except PermissionError as exc:
            raise FileAccessDeniedError(str(path), operation="stat", cause=exc) from exc
        except OSError as exc:
            raise ParseFailureError(str(path), f"Failed reading file metadata: {exc}", cause=exc) from exc

        checks: List[Dict[str, Any]] = []
        self._check(checks, "readable_file", True)
        self._check(checks, "non_empty", size > 0, None if size > 0 else "file is empty",
                    severity="error" if require_nonempty else "warning")

        detection = self.detector.detect(path)["payload"] if size > 0 else {}
        expected_spec = DetectionEngine.spec_for(expected_format) if expected_format else None
        format_id = (expected_spec.format_id if expected_spec else detection.get("format_id")) or "unknown"
        if format_id == "zip" and detection.get("extension_format") in _REQUIRED_MEMBERS:
            # A ZIP that claims to be a package but lacks its signature members is a damaged package.
            format_id = str(detection["extension_format"])
        if expected_format and expected_spec is None:
            self._check(checks, "expected_format_known", False, f"unknown format '{expected_format}'", severity="warning")
        if detection:
            if detection.get("damaged"):
                self._check(checks, "container_readable", False, "container signature present but structure unreadable")
            if expected_spec and detection.get("detected_by") in {"signature", "container"} \
                    and detection.get("format_id") != expected_spec.format_id \
                    and detection.get("extension_mismatch") is not False:
                self._check(checks, "format_matches_expected", False,
                            {"expected": expected_spec.format_id, "detected": detection.get("format_id")})
            elif expected_spec and detection.get("detected_by") in {"signature", "container"} \
                    and detection.get("format_id") != expected_spec.format_id:
                # Compatible variant (xlsm for xlsx): keep the stricter, expected identity.
                self._check(checks, "format_matches_expected", True, {"detected": detection.get("format_id")})
            if detection.get("encrypted"):
                self._check(checks, "not_encrypted", False, "encrypted content cannot be validated", severity="warning")

        if size > 0 and not detection.get("damaged"):
            spec = DetectionEngine.spec_for(format_id)
            if format_id in _REQUIRED_MEMBERS or format_id == "zip" or (spec and spec.container == "zip" and format_id not in {"xlsb"}):
                self._check_zip(path, format_id, deep_mode, checks)
            elif format_id == "pdf":
                self._check_pdf(path, size, deep_mode, checks)
            elif format_id in {"gzip", "bzip2", "xz"}:
                self._check_stream(path, format_id, deep_mode, checks)
            elif format_id == "tar_gz":
                self._check_stream(path, format_id, deep_mode, checks)
                self._check_tar(path, format_id, deep_mode, checks)
            elif format_id == "tar":
                self._check_tar(path, format_id, deep_mode, checks)
            elif format_id in {"json", "xml", "svg", "fb2", "csv", "tsv", "txt", "md", "html", "rst", "log", "jsonl"}:
                self._check_text_like(path, size, format_id, checks)
            else:
                self._check(checks, "structure_validated", True, f"no structural validator for '{format_id}'", severity="warning", skipped=True)

        hashes: Dict[str, Any] = {"size_bytes": size}
        if compute_hash:
            hashes["sha256"] = sha256_file(path)

        status = self._status(checks)
        payload = {
            "valid": status != "failed",
            "integrity_status": status,
            "path": str(path),
            "format_id": format_id,
            "deep": deep_mode,
            "checks": checks,
            "hashes": hashes,
            "detection": {key: detection.get(key) for key in ("format_id", "detected_by", "confidence", "extension_mismatch", "damaged", "encrypted")} if detection else {},
        }
        warnings = [f"integrity:{item['name']}" for item in checks if (not item["ok"] and item["severity"] == "warning")]
        warnings += [f"integrity_skipped:{item['name']}" for item in checks if item.get("skipped") and item["severity"] == "warning"]
        result = build_success_result(
            operation="validate",
            payload=payload,
            metadata={"engine_version": INTEGRITY_ENGINE_VERSION},
            warnings=warnings,
            timer=timer,
        )
        result["status"] = {"ok": "ok", "degraded": "partial", "failed": "error"}[status]
        return result

    def require_valid(self, file_path: str | Path, **kwargs: Any) -> Dict[str, Any]:
        """Like ``validate_file`` but raises ``IntegrityError`` when any error-level check fails."""

        report = self.validate_file(file_path, **kwargs)
        payload = report["payload"]
        if not payload["valid"]:
            failed = [item for item in payload["checks"] if not item["ok"] and item["severity"] == "error"]
            raise IntegrityError(str(file_path), "Reader integrity validation failed", checks=failed)
        return report

    def verify_output(self, file_path: str | Path, expected_format: str) -> Dict[str, Any]:
        """Verify a freshly written artifact: non-empty, well-formed for its format, hashed."""

        return self.validate_file(
            file_path,
            expected_format=expected_format,
            deep=True,
            require_nonempty=self.require_nonempty_outputs,
        )

    def verify_artifact(
        self,
        file_path: str | Path,
        *,
        expected_sha256: Optional[str] = None,
        expected_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Check a previously recorded artifact is still the same bytes (used when resuming jobs)."""

        timer = OperationTimer("verify_artifact")
        path = expand_path(file_path)
        checks: List[Dict[str, Any]] = []
        exists = path.is_file()
        self._check(checks, "artifact_exists", exists, None if exists else "file is missing")
        actual_size: Optional[int] = None
        actual_hash: Optional[str] = None
        if exists:
            actual_size = int(path.stat().st_size)
            if expected_size is not None:
                self._check(checks, "artifact_size", actual_size == int(expected_size),
                            {"expected": int(expected_size), "actual": actual_size})
            if expected_sha256:
                actual_hash = sha256_file(path)
                self._check(checks, "artifact_sha256", actual_hash == expected_sha256,
                            {"expected": expected_sha256, "actual": actual_hash})
        status = self._status(checks)
        result = build_success_result(
            operation="verify_artifact",
            payload={"valid": status != "failed", "integrity_status": status, "path": str(path), "checks": checks,
                     "hashes": {"size_bytes": actual_size, "sha256": actual_hash}},
            metadata={"engine_version": INTEGRITY_ENGINE_VERSION},
            timer=timer,
        )
        result["status"] = "ok" if status != "failed" else "error"
        return result

    def manifest(self, file_paths: Iterable[str | Path]) -> Dict[str, Any]:
        """Hash a set of files into a deterministic manifest (used by jobs and packaging)."""

        timer = OperationTimer("manifest")
        entries: List[Dict[str, Any]] = []
        for file_path in file_paths:
            path = expand_path(file_path)
            if not path.exists():
                raise FileMissingError(str(path))
            if not path.is_file():
                raise FileNotRegularError(str(path))
            entries.append({"path": str(path), "size_bytes": int(path.stat().st_size), "sha256": sha256_file(path)})
        entries.sort(key=lambda item: item["path"])
        return build_success_result(
            operation="manifest",
            payload={"files": entries, "manifest_sha256": stable_hash([[item["path"], item["sha256"]] for item in entries])},
            metadata={"engine_version": INTEGRITY_ENGINE_VERSION},
            timer=timer,
        )


if __name__ == "__main__":
    print("\n=== Running Integrity Engine ===\n")
    printer.status("TEST", "Integrity Engine initialized", "info")

    import tempfile

    with tempfile.TemporaryDirectory(prefix="reader_integrity_test_") as tmp_dir:
        tmp = Path(tmp_dir)
        engine = IntegrityEngine()

        good = tmp / "good.json"
        good.write_text('{"ok": true}', encoding="utf-8")
        report = engine.validate_file(good, deep=True)
        assert report["payload"]["valid"] and report["status"] == "ok"

        bad = tmp / "bad.json"
        bad.write_text('{"ok": ', encoding="utf-8")
        report = engine.validate_file(bad, deep=True)
        assert not report["payload"]["valid"] and report["status"] == "error"
        printer.status("TEST", "Malformed JSON reported", "success")

        try:
            engine.require_valid(bad, deep=True)
        except IntegrityError:
            printer.status("TEST", "require_valid raises IntegrityError", "success")
        else:
            raise AssertionError("Expected IntegrityError for malformed JSON")

        archive = tmp / "bomb.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as handle:
            handle.writestr("big.bin", b"\x00" * (8 * 1024 * 1024))
        guarded = IntegrityEngine()
        guarded.guard = ExpansionGuard(ExpansionLimits(max_member_bytes=1024 * 1024))
        guarded.limits = guarded.guard.limits
        report = guarded.validate_file(archive, deep=True)
        assert not report["payload"]["valid"]
        printer.status("TEST", "Expansion limit enforced", "success")

    print("\n=== Test ran successfully ===\n")
