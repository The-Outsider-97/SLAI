"""Production-grade Salvage Engine for the Reader subsystem.

The salvage engine recovers what it can from *damaged containers*, before parsing.

Boundary with ``RecoveryEngine``
--------------------------------
``SalvageEngine`` works on **bytes**: truncated or corrupt ZIP packages (docx, odt, epub, ...),
PDFs, compressed streams, tar archives, and structured text that was cut off. It writes a new,
repaired file. ``RecoveryEngine`` works on **text after parsing** (encoding damage, noise, OCR
artifacts). The two never overlap: salvage produces a file the normal ``ParserEngine`` can read,
and the parser then hands text to recovery if needed. There is deliberately no second parser here.

Design goals
------------
- Never touch the original. Output is a new file published without overwriting anything.
- Honest result levels, no invented confidence number:
  ``full`` (everything accounted for), ``partial`` (some parts lost, format kept), ``text_only``
  (only text could be recovered, layout and media are gone), ``none`` (nothing usable; raised as
  ``SalvageError`` with the report attached). Every report lists recovered and lost parts. When damage
  was detected but the lost parts cannot be enumerated, an explicit ``unverified_remainder`` entry
  keeps the level from claiming ``full``.
- Only repair what is verifiably usable. A repaired file is re-validated by ``IntegrityEngine``;
  one that does not validate is discarded rather than handed back.
- Bounded work: input size, scan window, per-member and total output are limited, and archive member
  names are sanitized before anything is rebuilt.
- Nothing is run or imported beyond the standard library and the optional ``pypdf``.
"""

from __future__ import annotations

import base64
import bz2
import csv
import io
import json
import lzma
import re
import struct
import tarfile
import tempfile
import zipfile
import zlib

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from xml.etree import ElementTree as ET

from .utils.config_loader import get_config_section, load_global_config
from .utils.reader_error import *
from .utils.reader_helpers import *
from .integrity_engine import IntegrityEngine, PACKAGE_MAIN_XML_PART, PACKAGE_REQUIRED_MEMBERS, unsafe_member_name
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Salvage Engine")
printer = PrettyPrinter()

SALVAGE_ENGINE_VERSION = "1.0.0"

_KNOWN_CONFIG_KEYS = (
    "output_dir",
    "max_input_bytes",
    "max_output_bytes",
    "max_candidates",
    "validate_output",
)

_LOCAL_HEADER = struct.Struct("<4sHHHHHIIIHH")  # signature .. extra length (30 bytes)
_LOCAL_SIG = b"PK\x03\x04"
_NEXT_SIGS = (b"PK\x03\x04", b"PK\x01\x02", b"PK\x05\x06", b"PK\x07\x08")
_ZIP_FORMATS = frozenset(PACKAGE_REQUIRED_MEMBERS) | {"zip", "xlsb"}
_STREAM_FORMATS = {"gzip": "gz", "bzip2": "bz2", "xz": "xz"}
_XML_TOKEN = re.compile(
    r"<!--.*?-->|<!\[CDATA\[.*?\]\]>|<\?.*?\?>|<!DOCTYPE[^>]*>|<(/?)([^\s/>!?][^\s/>]*)[^>]*?(/?)>",
    re.DOTALL,
)
_CHUNK = 64 * 1024


@dataclass
class SalvageReport:
    """What was recovered and what was lost. Counts and names, not a confidence score."""

    strategy: str
    level: str = "none"                  # full | partial | text_only | none
    input_bytes: int = 0
    output_bytes: int = 0
    repaired_path: Optional[str] = None
    repaired_format: Optional[str] = None
    recovered_parts: List[Dict[str, Any]] = field(default_factory=list)
    lost_parts: List[Dict[str, Any]] = field(default_factory=list)
    damage: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "level": self.level,
            "input_bytes": self.input_bytes,
            "output_bytes": self.output_bytes,
            "repaired_path": self.repaired_path,
            "repaired_format": self.repaired_format,
            "recovered_count": len(self.recovered_parts),
            "lost_count": len(self.lost_parts),
            "recovered_parts": json_safe(self.recovered_parts[:200]),
            "lost_parts": json_safe(self.lost_parts[:200]),
            "damage": list(self.damage),
            "notes": list(self.notes),
        }


@dataclass
class _Recovered:
    """A strategy's output before it is written and validated."""

    payload: bytes
    extension: str
    recovered: List[Dict[str, Any]]
    lost: List[Dict[str, Any]]
    notes: List[str] = field(default_factory=list)
    level: Optional[str] = None   # set by a strategy that can only deliver text (``text_only``)


class SalvageEngine:
    """Recovers usable content from damaged containers and writes a repaired copy."""

    def __init__(self, integrity: Optional[IntegrityEngine] = None) -> None:
        self.config = load_global_config()
        self.se_config = get_config_section("salvage_engine") or {}
        warn_unknown_config_keys("salvage_engine", self.se_config, _KNOWN_CONFIG_KEYS)

        self.output_dir = str(self.se_config.get("output_dir") or "tmp/reader/salvaged")
        self.max_input_bytes = coerce_int(self.se_config.get("max_input_bytes"), 256 * 1024 * 1024, minimum=1024)
        self.max_output_bytes = coerce_int(self.se_config.get("max_output_bytes"), 512 * 1024 * 1024, minimum=1024)
        self.max_candidates = coerce_int(self.se_config.get("max_candidates"), 4000, minimum=1)
        self.validate_output = coerce_bool(self.se_config.get("validate_output"), True)

        self.integrity = integrity or IntegrityEngine()
        self.detector = self.integrity.detector
        self.guard = self.integrity.guard
        self.settings = {
            "output_dir": self.output_dir,
            "max_input_bytes": self.max_input_bytes,
            "max_output_bytes": self.max_output_bytes,
            "max_candidates": self.max_candidates,
            "validate_output": self.validate_output,
        }
        logger.info("Salvage Engine initialized | max_input=%s validate_output=%s", self.max_input_bytes, self.validate_output)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @staticmethod
    def _pypdf() -> Optional[Any]:
        return IntegrityEngine._pypdf()

    def capabilities(self) -> Dict[str, Any]:
        return {
            "engine_version": SALVAGE_ENGINE_VERSION,
            "strategies": {
                "zip_package": sorted(_ZIP_FORMATS),
                "pdf": ["pdf"] if self._pypdf() is not None else [],
                "stream": sorted(_STREAM_FORMATS),
                "tar": ["tar", "tar_gz"],
                "text": ["json", "jsonl", "xml", "csv"],
            },
            "levels": ["full", "partial", "text_only", "none"],
            "settings": dict(self.settings),
        }

    def describe(self) -> Dict[str, Any]:
        return {"status": "ok", **self.capabilities()}

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fail(source: Any, message: str, *, cause: Optional[BaseException] = None, **context: Any) -> SalvageError:
        return SalvageError(str(source), message, context=context or None, cause=cause)

    def _read_input(self, path: Path) -> bytes:
        size = path.stat().st_size
        if size > self.max_input_bytes:
            raise FileTooLargeError(str(path), int(size), self.max_input_bytes)
        try:
            return path.read_bytes()
        except PermissionError as exc:
            raise FileAccessDeniedError(str(path), operation="read", cause=exc) from exc
        except OSError as exc:
            raise self._fail(path, f"Failed reading the damaged file: {exc}", cause=exc) from exc

    def _check_output_size(self, size: int, source: Path) -> None:
        if size > self.max_output_bytes:
            raise IntegrityError(str(source), "Salvaged output would exceed the configured limit",
                                 context={"output_bytes": size, "max_output_bytes": self.max_output_bytes})

    # ------------------------------------------------------------------
    # Structured-text repairs
    # ------------------------------------------------------------------

    def repair_json(self, text: str) -> Optional[Tuple[str, int]]:
        """Return ``(repaired_text, dropped_chars)`` for truncated JSON, or ``None``.

        The text is scanned with string/escape awareness. Cut points are commas and opening brackets
        outside strings. Whole elements are the unit of trust: when an array is open at the end, only
        that array's element boundaries are tried first, so a record cut in half is dropped rather than
        returned with fields missing. The latest cut whose prefix, with the open containers closed,
        parses as JSON wins. Nothing is invented beyond closing brackets.
        """

        stack: List[str] = []
        in_string = False
        escaped = False
        records: Deque[Tuple[int, str, int]] = deque(maxlen=self.max_candidates)  # (cut_index, closers, depth)
        for index, char in enumerate(text):
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char in "{[":
                stack.append("}" if char == "{" else "]")
                records.append((index + 1, "".join(reversed(stack)), len(stack)))
            elif char in "}]":
                if not stack or stack[-1] != char:
                    break  # structurally wrong from here on; earlier candidates still apply
                stack.pop()
            elif char == ",":
                records.append((index, "".join(reversed(stack)), len(stack)))
        stripped = text.rstrip()
        if not in_string and stripped and stripped[-1] in '}]"':
            # The text ends on a closed value, so "valid prefix, just missing closers" is trustworthy.
            # (A bare number or literal at the end may itself be cut short, so it is not trusted.)
            records.append((len(text), "".join(reversed(stack)), len(stack)))

        def attempt(candidates: Iterable[Tuple[int, str, int]]) -> Optional[Tuple[str, int]]:
            for cut, closers, _depth in candidates:
                candidate = text[:cut].rstrip()
                if candidate.endswith(","):
                    candidate = candidate[:-1].rstrip()
                candidate += closers
                try:
                    json.loads(candidate)
                except (ValueError, RecursionError):
                    continue
                return candidate, max(0, len(text) - cut)
            return None

        ordered = list(reversed(records))
        open_arrays = [position for position, closer in enumerate(stack) if closer == "]"]
        if open_arrays:
            boundary_depth = open_arrays[-1] + 1
            found = attempt(item for item in ordered if item[2] == boundary_depth)
            if found is not None:
                return found
        return attempt(ordered)

    def repair_xml(self, text: str) -> Optional[Tuple[str, int]]:
        """Keep the well-formed prefix of an XML document and close the open elements."""

        if "<!ENTITY" in text:
            return None  # never rebuild documents with entity declarations
        try:
            ET.fromstring(text.encode("utf-8"))
            return text, 0
        except ET.ParseError as exc:
            line, column = exc.position
        lines = text.split("\n")
        offset = sum(len(item) + 1 for item in lines[: max(0, line - 1)]) + column
        cut = text.rfind(">", 0, max(0, min(offset, len(text))))
        # Walk back over earlier tag ends until the prefix is acceptable (the error can be mid-tag).
        for _ in range(64):
            if cut < 0:
                return None
            prefix = text[: cut + 1]
            stack: List[str] = []
            for token in _XML_TOKEN.finditer(prefix):
                closing, name, selfclose = token.groups()
                if name is None:
                    continue
                if closing:
                    if stack and stack[-1] == name:
                        stack.pop()
                elif not selfclose:
                    stack.append(name)
            candidate = prefix + "".join(f"</{name}>" for name in reversed(stack))
            try:
                ET.fromstring(candidate.encode("utf-8"))
            except ET.ParseError:
                cut = text.rfind(">", 0, cut)
                continue
            return candidate, max(0, len(text) - (cut + 1))
        return None

    def _salvage_json(self, path: Path, raw: bytes, strategy: str) -> _Recovered:
        text = decode_bytes(raw, source=str(path)).text
        repaired = self.repair_json(text)
        if repaired is None:
            raise self._fail(path, "No parseable prefix could be found in the JSON content", strategy=strategy)
        content, dropped = repaired
        lost = [{"name": "trailing_content", "chars": dropped, "reason": "truncated_or_corrupt"}] if dropped else []
        return _Recovered(content.encode("utf-8"), ".json", [{"name": "json_prefix", "chars": len(content)}], lost,
                          ["closed open containers; nothing was invented beyond brackets"] if dropped else [])

    def _salvage_xml(self, path: Path, raw: bytes, strategy: str) -> _Recovered:
        text = decode_bytes(raw, source=str(path)).text
        repaired = self.repair_xml(text)
        if repaired is None:
            raise self._fail(path, "No well-formed XML prefix could be found (or the document declares entities)", strategy=strategy)
        content, dropped = repaired
        lost = [{"name": "trailing_content", "chars": dropped, "reason": "truncated_or_corrupt"}] if dropped else []
        return _Recovered(content.encode("utf-8"), path.suffix or ".xml", [{"name": "xml_prefix", "chars": len(content)}], lost)

    def _salvage_jsonl(self, path: Path, raw: bytes, strategy: str) -> _Recovered:
        text = decode_bytes(raw, source=str(path)).text
        kept: List[str] = []
        lost: List[Dict[str, Any]] = []
        for number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                json.loads(line)
                kept.append(line)
            except ValueError:
                lost.append({"name": f"line_{number}", "reason": "not_valid_json"})
        if not kept:
            raise self._fail(path, "No valid JSON lines were found", strategy=strategy)
        return _Recovered(("\n".join(kept) + "\n").encode("utf-8"), path.suffix or ".jsonl",
                          [{"name": "json_lines", "count": len(kept)}], lost)

    def _salvage_csv(self, path: Path, raw: bytes, strategy: str) -> _Recovered:
        text = decode_bytes(raw, source=str(path)).text
        lines = text.splitlines(keepends=True)
        reader = csv.reader(io.StringIO(text))
        rows: List[List[str]] = []
        ends: List[int] = []
        try:
            for row in reader:
                rows.append(row)
                ends.append(reader.line_num)
        except csv.Error as exc:
            raise self._fail(path, f"CSV could not be tokenised: {exc}", cause=exc, strategy=strategy) from exc
        if not rows:
            raise self._fail(path, "CSV contains no rows", strategy=strategy)
        width = max(set(len(row) for row in rows), key=[len(row) for row in rows].count)
        keep = len(rows)
        while keep > 1 and len(rows[keep - 1]) != width:
            keep -= 1  # only a torn tail is dropped; a ragged row in the middle is left to the parser
        if keep == len(rows) and keep > 1:
            try:
                for _row in csv.reader(io.StringIO(text), strict=True):
                    pass
            except csv.Error:
                keep -= 1  # unterminated quote at the end: the last record was cut mid-way
        dropped = len(rows) - keep
        content = "".join(lines[: ends[keep - 1]])
        lost = [{"name": f"row_{index + 1}", "reason": "torn_tail"} for index in range(keep, len(rows))]
        return _Recovered(content.encode("utf-8"), path.suffix or ".csv", [{"name": "csv_rows", "count": keep}], lost,
                          [f"dropped {dropped} trailing row(s) with {width}-column header mismatch"] if dropped else [])

    # ------------------------------------------------------------------
    # ZIP packages
    # ------------------------------------------------------------------

    def _inflate_partial(self, data: bytes, start: int) -> Tuple[bytes, bool, int]:
        """Inflate a raw deflate stream from ``start``. Returns ``(output, complete, end_offset)``."""

        decompressor = zlib.decompressobj(-15)
        out = bytearray()
        position = start
        try:
            while position < len(data) and not decompressor.eof:
                chunk = data[position:position + _CHUNK]
                position += len(chunk)
                out += decompressor.decompress(chunk)
                if len(out) > self.guard.limits.max_member_bytes > 0:
                    raise IntegrityError("zip", "Member expands beyond the configured limit while salvaging",
                                         context={"limit": self.guard.limits.max_member_bytes})
        except zlib.error:
            return bytes(out), False, position
        if decompressor.eof:
            return bytes(out), True, position - len(decompressor.unused_data)
        return bytes(out), False, position

    def _scan_zip(self, data: bytes, known_sizes: Mapping[str, int]) -> List[Dict[str, Any]]:
        """Walk local file headers. Works without a central directory (truncated files)."""

        members: List[Dict[str, Any]] = []
        position = 0
        total = 0
        while True:
            index = data.find(_LOCAL_SIG, position)
            if index < 0 or index + _LOCAL_HEADER.size > len(data):
                if 0 <= index < len(data):
                    members.append({"name": "<truncated header>", "complete": False, "data": b"", "reason": "truncated_header"})
                break
            (_sig, _version, flags, method, _time, _date, crc, compressed, _size, name_len, extra_len) = _LOCAL_HEADER.unpack_from(data, index)
            name_bytes = data[index + _LOCAL_HEADER.size:index + _LOCAL_HEADER.size + name_len]
            name = name_bytes.decode("utf-8" if flags & 0x800 else "cp437", errors="replace")
            body = index + _LOCAL_HEADER.size + name_len + extra_len
            entry: Dict[str, Any] = {"name": name, "complete": False, "data": b"", "reason": None, "encrypted": bool(flags & 0x1)}
            if flags & 0x1:
                entry["reason"] = "encrypted"
                members.append(entry)
                position = body + max(compressed, 0) if compressed else index + 4
                continue
            if len(name_bytes) < name_len:
                entry["reason"] = "truncated_header"
                members.append(entry)
                break
            if method == 0:
                size = compressed or known_sizes.get(name, 0)
                if not size and flags & 0x8:
                    nxt = min((data.find(sig, body) for sig in _NEXT_SIGS if data.find(sig, body) >= 0), default=len(data))
                    payload = data[body:nxt]
                else:
                    payload = data[body:body + size]
                    entry["complete"] = len(payload) == size
                    if not entry["complete"]:
                        entry["reason"] = "truncated_data"
                entry["data"] = payload
                entry["complete"] = entry["complete"] or (not size and bool(flags & 0x8))
                position = body + len(payload)
            elif method == 8:
                payload, complete, end = self._inflate_partial(data, body)
                entry["data"], entry["complete"] = payload, complete
                declared = compressed or known_sizes.get(name, 0)
                if complete:
                    position = end
                else:
                    entry["reason"] = "truncated_data" if body + declared > len(data) or not declared else "corrupt_data"
                    # A corrupt (not cut-off) stream does not end the file: jump over it to the next member.
                    position = body + declared if declared and body + declared <= len(data) else len(data)
            else:
                entry["reason"] = f"unsupported_method_{method}"
                position = body + compressed if compressed else index + 4
            # CRC proves a complete member; without it (descriptor flag) completeness rests on the stream end.
            if entry["complete"] and crc and not flags & 0x8 and zlib.crc32(entry["data"]) & 0xFFFFFFFF != crc:
                entry["complete"], entry["reason"] = False, "crc_mismatch"
            total += len(entry["data"])
            if self.guard.limits.max_total_bytes and total > self.guard.limits.max_total_bytes:
                raise IntegrityError("zip", "Salvaged members exceed the configured total limit", context={"limit": self.guard.limits.max_total_bytes})
            members.append(entry)
            if len(members) > (self.guard.limits.max_members or 10**9):
                raise IntegrityError("zip", "Too many members while salvaging", context={"limit": self.guard.limits.max_members})
            if position <= index:
                position = index + 4
        return members

    def _salvage_zip(self, path: Path, raw: bytes, format_id: str, strategy: str) -> _Recovered:
        central: Dict[str, int] = {}
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                central = {info.filename: info.compress_size for info in archive.infolist()}
        except (zipfile.BadZipFile, OSError, EOFError, ValueError):
            pass  # no usable central directory: the local headers are all there is
        scanned = self._scan_zip(raw, central)
        required = PACKAGE_REQUIRED_MEMBERS.get(format_id, ())
        main_part = PACKAGE_MAIN_XML_PART.get(format_id)

        recovered: List[Dict[str, Any]] = []
        lost: List[Dict[str, Any]] = []
        notes: List[str] = []
        kept: Dict[str, bytes] = {}
        for entry in scanned:
            name = entry["name"]
            if name.startswith("<"):
                lost.append({"name": name, "reason": entry["reason"]})
                continue
            if unsafe_member_name(name):
                lost.append({"name": name, "reason": "unsafe_path"})
                continue
            if name.endswith("/"):
                continue
            if name in kept:
                lost.append({"name": name, "reason": "duplicate_name"})
                continue
            payload = entry["data"]
            is_xml = name.lower().endswith((".xml", ".rels")) or name == main_part
            if not entry["complete"]:
                if is_xml and payload:
                    repaired = self.repair_xml(payload.decode("utf-8", errors="replace"))
                    if repaired is not None:
                        kept[name] = repaired[0].encode("utf-8")
                        recovered.append({"name": name, "bytes": len(kept[name]), "complete": False, "note": "xml_closed"})
                        lost.append({"name": name, "reason": entry["reason"] or "incomplete", "partial": True})
                        continue
                lost.append({"name": name, "reason": entry["reason"] or "incomplete"})
                continue
            kept[name] = payload
            recovered.append({"name": name, "bytes": len(payload), "complete": True})
        for name in central:
            if name not in kept and not name.endswith("/") and not any(item["name"] == name for item in lost):
                lost.append({"name": name, "reason": "missing_from_local_scan"})

        missing_required = [name for name in required if name not in kept]
        if missing_required:
            raise self._fail(path, "Required package parts could not be recovered", missing=missing_required, format=format_id,
                             recovered=len(recovered), lost=len(lost))
        if not kept:
            raise self._fail(path, "No archive members could be recovered", format=format_id)

        out = io.BytesIO()
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as target:
            names = sorted(kept, key=lambda item: (item != "mimetype", item != "[Content_Types].xml", False))
            for name in names:
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_STORED if name == "mimetype" else zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                target.writestr(info, kept[name])
        notes.append("package rebuilt from local file headers; original central directory not used" if not central
                     else "package rebuilt from verified members")
        return _Recovered(out.getvalue(), path.suffix or ".zip", recovered, lost, notes)

    # ------------------------------------------------------------------
    # PDF
    # ------------------------------------------------------------------

    def _rebuild_pdf(self, raw: bytes) -> Optional[Tuple[bytes, List[Dict[str, Any]], List[Dict[str, Any]]]]:
        """Rebuild a PDF from the pages pypdf can still read. ``None`` if the structure is unusable."""

        pypdf = self._pypdf()
        try:
            assert pypdf is not None
            reader = pypdf.PdfReader(io.BytesIO(raw), strict=False)
            if getattr(reader, "is_encrypted", False):
                return None
            page_total = len(reader.pages)
        except Exception:  # pypdf raises many unrelated exception types for damaged files
            return None
        assert pypdf is not None
        writer = pypdf.PdfWriter()
        recovered: List[Dict[str, Any]] = []
        lost: List[Dict[str, Any]] = []
        for index in range(page_total):
            try:
                page = reader.pages[index]
                page.extract_text()  # forces content-stream parsing so a broken page fails here, not later
                writer.add_page(page)
                recovered.append({"name": f"page_{index + 1}"})
            except Exception as exc:
                lost.append({"name": f"page_{index + 1}", "reason": f"{type(exc).__name__}"})
        if not recovered:
            return None
        out = io.BytesIO()
        writer.write(out)
        return out.getvalue(), recovered, lost

    @staticmethod
    def _pdf_unescape(literal: bytes) -> str:
        out = bytearray()
        index = 0
        escapes = {ord("n"): 10, ord("r"): 13, ord("t"): 9, ord("b"): 8, ord("f"): 12}
        while index < len(literal):
            byte = literal[index]
            if byte == 0x5C and index + 1 < len(literal):  # backslash
                nxt = literal[index + 1]
                if nxt in escapes:
                    out.append(escapes[nxt]); index += 2
                elif 0x30 <= nxt <= 0x37:
                    digits = literal[index + 1:index + 4]
                    octal = bytes(ch for ch in digits if 0x30 <= ch <= 0x37)
                    out.append(int(octal, 8) & 0xFF); index += 1 + len(octal)
                else:
                    out.append(nxt); index += 2
            else:
                out.append(byte); index += 1
        return out.decode("cp1252", errors="replace")

    def _scan_pdf_text(self, raw: bytes) -> List[str]:
        """Last resort for PDFs whose page tree is gone: inflate content streams and read literal text operators.

        Only literal ``(…) Tj`` / ``[…] TJ`` strings are read. Fonts that map text through hex glyph ids yield
        nothing here rather than garbage.
        """

        texts: List[str] = []
        budget = self.max_output_bytes
        for marker in re.finditer(rb"stream\r?\n", raw):
            start = marker.end()
            end = raw.find(b"endstream", start)
            body = raw[start:end if end >= 0 else len(raw)]
            header = raw[max(0, start - 400):start].rsplit(b"obj", 1)[-1]
            if b"ASCII85Decode" in header:
                try:
                    cleaned = re.sub(rb"\s+", b"", body).split(b"~>")[0]
                    body = base64.a85decode(cleaned[: len(cleaned) - len(cleaned) % 5] if end < 0 else cleaned)
                except ValueError:
                    continue
            try:
                content = zlib.decompressobj().decompress(body, min(budget, 64 * 1024 * 1024))
            except zlib.error:
                content = body if b"BT" in body[:2048] else b""
            if b"BT" not in content or not re.search(rb"T[jJ]", content):
                continue
            lines: List[str] = []
            for operation in re.finditer(rb"\[((?:\\.|[^\]\\])*)\]\s*TJ|\(((?:\\.|[^\\()])*)\)\s*(?:Tj|'|\")", content, re.DOTALL):
                if operation.group(1) is not None:
                    pieces = []
                    for item in re.finditer(rb"\(((?:\\.|[^\\()])*)\)|(-?\d+(?:\.\d+)?)", operation.group(1)):
                        if item.group(1) is not None:
                            pieces.append(self._pdf_unescape(item.group(1)))
                        elif float(item.group(2)) < -200:
                            pieces.append(" ")
                    lines.append("".join(pieces))
                else:
                    lines.append(self._pdf_unescape(operation.group(2)))
            text = "\n".join(line for line in lines if line.strip())
            if text.strip():
                texts.append(text)
        return texts

    def _salvage_pdf(self, path: Path, raw: bytes, strategy: str) -> _Recovered:
        pypdf = self._pypdf()
        if pypdf is not None:
            rebuilt = self._rebuild_pdf(raw)
            note = "pages rebuilt into a new PDF"
            if rebuilt is None and b"%%EOF" not in raw[-2048:]:
                # A truncated file has no trailer; give pypdf one so it scans for the objects itself.
                rebuilt = self._rebuild_pdf(raw + b"\nstartxref\n0\n%%EOF\n")
                note = "pages rebuilt after supplying a missing trailer"
            if rebuilt is not None:
                payload, recovered, lost = rebuilt
                if note.endswith("trailer"):
                    lost.append({"name": "unverified_remainder", "reason": "trailer_missing; total page count unknown"})
                return _Recovered(payload, ".pdf", recovered, lost, [f"{len(recovered)} page(s) {note}"])
        texts = self._scan_pdf_text(raw)
        if not texts:
            raise self._fail(path, "No PDF page or text could be recovered", pypdf_available=pypdf is not None)
        body = "\n\n".join(f"--- Recovered text block {index} ---\n{text}" for index, text in enumerate(texts, start=1))
        return _Recovered(
            body.encode("utf-8"), ".txt",
            [{"name": f"text_block_{index}", "chars": len(text)} for index, text in enumerate(texts, start=1)],
            [{"name": "layout_images_fonts", "reason": "text_only_scan"}],
            ["page structure unreadable; text read directly from content streams (literal strings only)"],
            level="text_only",
        )

    # ------------------------------------------------------------------
    # Streams and tar
    # ------------------------------------------------------------------

    def _decompress_partial(self, raw: bytes, kind: str) -> Tuple[bytes, bool]:
        """Decompress as far as the stream allows. Returns ``(bytes, reached_clean_end)``."""

        limit = self.max_output_bytes
        if kind == "gz":
            decompressor: Any = zlib.decompressobj(16 + zlib.MAX_WBITS)
        elif kind == "bz2":
            decompressor = bz2.BZ2Decompressor()
        else:
            decompressor = lzma.LZMADecompressor()
        out = bytearray()
        try:
            for start in range(0, len(raw), _CHUNK):
                out += decompressor.decompress(raw[start:start + _CHUNK])
                if len(out) > limit:
                    raise IntegrityError("stream", "Salvaged stream exceeds the configured output limit", context={"limit": limit})
                if getattr(decompressor, "eof", False):
                    return bytes(out), True
        except (zlib.error, OSError, EOFError, ValueError, lzma.LZMAError):
            return bytes(out), False
        return bytes(out), bool(getattr(decompressor, "eof", False))

    def _salvage_stream(self, path: Path, raw: bytes, format_id: str, strategy: str) -> _Recovered:
        kind = _STREAM_FORMATS[format_id]
        data, clean = self._decompress_partial(raw, kind)
        if not data:
            raise self._fail(path, "The compressed stream yielded no data", format=format_id)
        inner = path.name[: -len(path.suffix)] if path.suffix.lower() == f".{kind}" else path.name
        inner_path = Path(inner)
        recovered = [{"name": inner_path.name, "bytes": len(data)}]
        lost = [] if clean else [{"name": "stream_tail", "reason": "truncated_or_corrupt"}]
        notes = [f"decompressed {len(data)} bytes from a {'complete' if clean else 'truncated'} {kind} stream"]
        extension = inner_path.suffix or ".bin"
        text_strategy = {".json": self._salvage_json, ".xml": self._salvage_xml, ".csv": self._salvage_csv, ".jsonl": self._salvage_jsonl}.get(extension.lower())
        if text_strategy is not None and not clean:
            try:
                inner_result = text_strategy(inner_path, data, "stream+text")
                data, extension = inner_result.payload, inner_result.extension
                lost += inner_result.lost
                notes.append("inner text content repaired as well")
            except SalvageError:
                notes.append("inner content could not be repaired further; raw decompressed bytes kept")
        return _Recovered(data, extension, recovered, lost, notes)

    def _salvage_tar(self, path: Path, raw: bytes, format_id: str, strategy: str) -> _Recovered:
        if format_id == "tar_gz":
            data, clean = self._decompress_partial(raw, "gz")
        else:
            data, clean = raw, True
        recovered: List[Dict[str, Any]] = []
        lost: List[Dict[str, Any]] = []
        out = io.BytesIO()
        names: set = set()
        try:
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as source, tarfile.open(fileobj=out, mode="w", format=tarfile.PAX_FORMAT) as target:
                for member in source:
                    if not member.isfile():
                        if member.issym() or member.islnk() or member.isdev():
                            lost.append({"name": member.name, "reason": "link_or_special"})
                        continue
                    if unsafe_member_name(member.name) or member.name in names:
                        lost.append({"name": member.name, "reason": "unsafe_or_duplicate"})
                        continue
                    try:
                        handle = source.extractfile(member)
                        payload = handle.read() if handle is not None else b""
                    except (tarfile.TarError, EOFError, OSError):
                        lost.append({"name": member.name, "reason": "truncated_data"})
                        break
                    if len(payload) != member.size:
                        lost.append({"name": member.name, "reason": "truncated_data"})
                        break
                    info = tarfile.TarInfo(member.name)
                    info.size, info.mtime, info.mode, info.uid, info.gid = len(payload), 0, 0o644, 0, 0
                    target.addfile(info, io.BytesIO(payload))
                    names.add(member.name)
                    recovered.append({"name": member.name, "bytes": len(payload)})
        except (tarfile.TarError, EOFError, OSError) as exc:
            if not recovered:
                raise self._fail(path, f"Tar structure is unreadable: {exc}", cause=exc) from exc
            lost.append({"name": "archive_tail", "reason": "truncated_or_corrupt"})
        if not clean and not any(item["name"] == "archive_tail" for item in lost):
            lost.append({"name": "archive_tail", "reason": "truncated_or_corrupt"})
        if not recovered:
            raise self._fail(path, "No tar members could be recovered", format=format_id)
        return _Recovered(out.getvalue(), ".tar", recovered, lost, ["complete members rebuilt into a new, uncompressed tar"])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _strategy_for(self, format_id: str, extension: str) -> Optional[str]:
        if format_id in _ZIP_FORMATS:
            return "zip_package"
        if format_id == "pdf":
            return "pdf"
        if format_id in _STREAM_FORMATS:
            return "stream"
        if format_id in {"tar", "tar_gz"}:
            return "tar"
        ext = extension.lower()
        if format_id in {"json", "xml", "csv", "jsonl"}:
            return format_id
        return {".json": "json", ".xml": "xml", ".csv": "csv", ".jsonl": "jsonl", ".ndjson": "jsonl"}.get(ext)

    def can_salvage(self, file_path: str | Path) -> Dict[str, Any]:
        """Cheap, read-only answer to 'is there a salvage strategy for this file?'."""

        path = expand_path(file_path)
        detection = self.detector.detect(path)["payload"]
        format_id = detection.get("format_id") or "unknown"
        strategy = self._strategy_for(format_id, path.suffix)
        if format_id == "unknown" and detection.get("extension_format"):
            format_id = detection["extension_format"]
            strategy = self._strategy_for(format_id, path.suffix)
        return {"format_id": format_id, "strategy": strategy, "supported": strategy is not None and (strategy != "pdf" or self._pypdf() is not None)}

    def salvage(self, file_path: str | Path, output_dir: Optional[str | Path] = None, *, force: bool = False) -> Dict[str, Any]:
        """Recover a repaired copy of a damaged file.

        Returns a result whose ``payload.report`` states the level (``full`` or ``partial``) and lists
        recovered and lost parts. Raises ``SalvageError`` (report in its context) when nothing usable
        could be recovered. A file that validates as intact returns ``outcome='not_needed'`` and no
        copy unless ``force=True``.
        """

        timer = OperationTimer("salvage")
        path = expand_path(file_path)
        if not path.exists():
            raise FileMissingError(str(path))
        if not path.is_file():
            raise FileNotRegularError(str(path))

        detection = self.detector.detect(path)["payload"]
        format_id = detection.get("format_id") or "unknown"
        if format_id in {"unknown", "zip"} and detection.get("extension_format") in PACKAGE_REQUIRED_MEMBERS:
            format_id = str(detection["extension_format"])  # a damaged package is often only recognisable by its name
        if detection.get("encrypted"):
            raise self._fail(path, "Encrypted files cannot be salvaged", format=format_id)
        strategy = self._strategy_for(format_id, path.suffix)
        if strategy is None:
            raise self._fail(path, f"No salvage strategy exists for '{format_id}'", format=format_id)

        size = int(path.stat().st_size)
        report = SalvageReport(strategy=strategy, input_bytes=size)
        if size == 0:
            raise self._fail(path, "The file is empty; there is nothing to salvage")

        health = self.integrity.validate_file(path, deep=True, compute_hash=False)["payload"]
        report.damage = [item["name"] for item in health["checks"] if not item["ok"] and item["severity"] == "error"]
        if strategy in {"csv", "jsonl"}:
            # A torn tail on tabular text is only a warning to the validator but is exactly what salvage repairs.
            report.damage += [item["name"] for item in health["checks"] if not item["ok"] and item["severity"] == "warning"]
        if not report.damage and not force:
            report.level = "full"
            report.notes.append("file validates as intact; no repaired copy was written")
            return build_success_result(
                operation="salvage",
                payload={"report": report.to_dict(), "outcome": "not_needed", "repaired_path": None},
                metadata={"engine_version": SALVAGE_ENGINE_VERSION},
                timer=timer,
            )

        raw = self._read_input(path)
        try:
            if strategy == "zip_package":
                recovered = self._salvage_zip(path, raw, format_id, strategy)
            elif strategy == "pdf":
                recovered = self._salvage_pdf(path, raw, strategy)
            elif strategy == "stream":
                recovered = self._salvage_stream(path, raw, format_id, strategy)
            elif strategy == "tar":
                recovered = self._salvage_tar(path, raw, format_id, strategy)
            elif strategy == "json":
                recovered = self._salvage_json(path, raw, strategy)
            elif strategy == "jsonl":
                recovered = self._salvage_jsonl(path, raw, strategy)
            elif strategy == "xml":
                recovered = self._salvage_xml(path, raw, strategy)
            else:
                recovered = self._salvage_csv(path, raw, strategy)
            self._check_output_size(len(recovered.payload), path)
        except SalvageError as exc:
            exc.context.setdefault("report", report.to_dict())
            logger.warning("Salvage of %s recovered nothing: %s", path, exc)
            raise
        except (ReaderError, IntegrityError):
            raise
        except Exception as exc:
            raise self._fail(path, f"Salvage failed unexpectedly: {type(exc).__name__}: {exc}", cause=exc, strategy=strategy) from exc

        report.recovered_parts, report.lost_parts, report.notes = recovered.recovered, recovered.lost, list(recovered.notes)
        if report.damage and not recovered.lost and recovered.level is None:
            # Damage was detected but nothing specific can be named as lost: say so rather than claim "full".
            report.lost_parts.append({"name": "unverified_remainder", "reason": f"damage detected ({', '.join(report.damage[:3])}); lost content cannot be enumerated"})
        report.level = recovered.level or ("partial" if report.lost_parts else "full")

        out_dir = ensure_directory(output_dir or self.output_dir, purpose="salvage_output_dir")
        destination = prepare_output_path(out_dir, path.stem if recovered.extension != path.suffix else path.stem, recovered.extension, suffix="salvaged")
        handle, staged_name = tempfile.mkstemp(prefix=".reader_stage_", suffix=recovered.extension + ".part", dir=out_dir)
        staged = Path(staged_name)
        try:
            with open(handle, "wb") as stream:
                stream.write(recovered.payload)
            final = publish_file(staged, destination, purpose="salvaged_output")
        except (ReaderError, OSError) as exc:
            try:
                staged.unlink()
            except OSError:
                pass
            if isinstance(exc, ReaderError):
                raise
            raise self._fail(path, f"Failed writing the repaired file: {exc}", cause=exc) from exc

        report.repaired_path = str(final)
        report.output_bytes = int(final.stat().st_size)
        report.repaired_format = (self.detector.detect(final)["payload"].get("format_id") or recovered.extension.lstrip("."))
        if self.validate_output:
            check = self.integrity.validate_file(final, deep=True, compute_hash=False)["payload"]
            if not check["valid"]:
                failed = [item["name"] for item in check["checks"] if not item["ok"] and item["severity"] == "error"]
                try:
                    final.unlink()
                except OSError:
                    logger.warning("Failed removing invalid salvaged file %s", final)
                raise self._fail(path, "The repaired file failed integrity validation and was discarded", failed_checks=failed,
                                 report={**report.to_dict(), "repaired_path": None})
        logger.info("Salvaged %s -> %s | level=%s recovered=%d lost=%d", path, final, report.level, len(report.recovered_parts), len(report.lost_parts))
        warnings = [f"salvaged:{report.level}"]
        return build_success_result(
            operation="salvage",
            payload={"report": report.to_dict(), "outcome": "salvaged", "repaired_path": report.repaired_path},
            metadata={"engine_version": SALVAGE_ENGINE_VERSION},
            warnings=warnings,
            timer=timer,
        )


if __name__ == "__main__":
    print("\n=== Running Salvage Engine ===\n")
    printer.status("TEST", "Salvage Engine initialized", "info")

    with tempfile.TemporaryDirectory(prefix="reader_salvage_test_") as tmp_dir:
        tmp = Path(tmp_dir)
        engine = SalvageEngine()
        out = tmp / "out"

        truncated = tmp / "data.json"
        truncated.write_text('{"items": [{"id": 1, "name": "alpha"}, {"id": 2, "name": "be', encoding="utf-8")
        result = engine.salvage(truncated, out)
        repaired = json.loads(Path(result["payload"]["repaired_path"]).read_text(encoding="utf-8"))
        assert repaired == {"items": [{"id": 1, "name": "alpha"}]}, repaired
        assert result["payload"]["report"]["level"] == "partial"
        printer.status("TEST", "Truncated JSON keeps its complete prefix", "success")

        intact = tmp / "ok.json"
        intact.write_text('{"ok": true}', encoding="utf-8")
        assert engine.salvage(intact, out)["payload"]["outcome"] == "not_needed"
        printer.status("TEST", "Intact file is reported as not needing salvage", "success")

        hopeless = tmp / "none.json"
        hopeless.write_text("not json at all", encoding="utf-8")
        try:
            engine.salvage(hopeless, out)
        except SalvageError:
            printer.status("TEST", "Unrecoverable file raises SalvageError", "success")
        else:
            raise AssertionError("Expected SalvageError")

    print("\n=== Test ran successfully ===\n")
