from __future__ import annotations

"""Production-grade Compression Engine for the Reader subsystem.

The compression engine packages, optimizes and expands files. Every operation produces a
*new* artifact and a report; the original is never modified.

Design goals
------------
- Three explicit operations: ``package`` (several files into one archive), ``optimize``
  (make one file smaller while keeping its format) and ``expand`` (safe extraction).
  ``compress`` picks between them for the agent.
- Lossless by default and honest about it. Every report states ``lossless`` and *how* the
  result was verified. Lossy image shrinking only runs when ``lossy=True`` is passed.
- A result that is not smaller is not delivered: ``optimize`` returns ``no_gain`` and keeps
  the original instead of publishing a bigger file.
- Safe extraction. Member names are sanitized (zip-slip, Windows reserved names), links and
  devices are skipped, nothing is overwritten, and the shared ``ExpansionGuard`` counts the
  bytes that actually come out, so a header that lies about sizes cannot fill the disk.
- Atomic publication. Output is staged next to its destination, verified, then published with
  ``publish_file`` so two workers can never overwrite each other's file.
- Deterministic archives (fixed timestamps and ownership, no gzip mtime) so equal input gives
  equal bytes, which makes jobs idempotent and results diff-able.
- No external programs are run.
"""

import bz2
import gzip
import hashlib
import importlib
import io
import lzma
import os
import re
import stat
import tarfile
import tempfile
import uuid
import zipfile
import zlib

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .detection_engine import DetectionEngine
from .integrity_engine import ExpansionGuard, IntegrityEngine, unsafe_member_name
from .utils.config_loader import get_config_section, load_global_config
from .utils.reader_error import *
from .utils.reader_helpers import *
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Compression Engine")
printer = PrettyPrinter()

COMPRESSION_ENGINE_VERSION = "1.0.0"

_KNOWN_CONFIG_KEYS = (
    "output_dir",
    "default_package_format",
    "compression_level",
    "min_gain_ratio",
    "max_input_bytes",
    "max_package_files",
    "max_expand_files",
    "verify",
    "verify_text_max_pages",
    "image_max_dimension",
    "jpeg_quality",
)

PACKAGE_FORMATS: Dict[str, str] = {"zip": ".zip", "tar": ".tar", "tar.gz": ".tar.gz", "tar.xz": ".tar.xz", "tar.bz2": ".tar.bz2"}
_PACKAGE_ALIASES = {"tgz": "tar.gz", "gztar": "tar.gz", "txz": "tar.xz", "xztar": "tar.xz", "tbz2": "tar.bz2", "bztar": "tar.bz2"}
STREAM_METHODS: Dict[str, str] = {"gz": ".gz", "bz2": ".bz2", "xz": ".xz"}
_STREAM_ALIASES = {"gzip": "gz", "bzip2": "bz2", "lzma": "xz"}

# Formats whose zip container is rewritten at maximum deflate.
_ZIP_OPTIMIZABLE = frozenset({"zip", "docx", "xlsx", "xlsm", "pptx", "odt", "ods", "odp", "epub"})
_ARCHIVE_EXTENSIONS = (".zip", ".tar", ".tar.gz", ".tgz", ".tar.xz", ".txz", ".tar.bz2", ".tbz2", ".gz", ".bz2", ".xz", ".7z", ".rar")
_DETERMINISTIC_ZIP_TIME = (1980, 1, 1, 0, 0, 0)
_WINDOWS_RESERVED = frozenset({"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))})
_COMPONENT_BAD_CHARS = re.compile(r'[<>:"|?*\x00-\x1f]')
_CHUNK = 1024 * 1024


@dataclass
class CompressionReport:
    """What an operation did and how far the result can be trusted."""

    operation: str
    method: str
    outcome: str = "ok"               # ok | no_gain
    input_bytes: int = 0
    output_bytes: int = 0
    lossless: bool = True
    verified: bool = False
    verified_by: str = "not_verified"
    output_path: Optional[str] = None
    input_files: int = 1
    notes: List[str] = field(default_factory=list)

    @property
    def ratio(self) -> Optional[float]:
        return round(self.output_bytes / self.input_bytes, 4) if self.input_bytes else None

    def to_dict(self) -> Dict[str, Any]:
        saved = self.input_bytes - self.output_bytes
        return {
            "operation": self.operation,
            "method": self.method,
            "outcome": self.outcome,
            "input_bytes": self.input_bytes,
            "output_bytes": self.output_bytes,
            "ratio": self.ratio,
            "saved_bytes": saved,
            "saved_percent": round(100.0 * saved / self.input_bytes, 2) if self.input_bytes else None,
            "lossless": self.lossless,
            "verified": self.verified,
            "verified_by": self.verified_by,
            "output_path": self.output_path,
            "input_files": self.input_files,
            "notes": list(self.notes),
        }


class CompressionEngine:
    """Packages, optimizes and safely expands files."""

    def __init__(self, integrity: Optional[IntegrityEngine] = None) -> None:
        self.config = load_global_config()
        self.ce_config = get_config_section("compression_engine") or {}
        warn_unknown_config_keys("compression_engine", self.ce_config, _KNOWN_CONFIG_KEYS)

        self.output_dir = str(self.ce_config.get("output_dir") or "tmp/reader/compressed")
        self.default_package_format = self._normalize_package_format(self.ce_config.get("default_package_format") or "zip")
        self.compression_level = coerce_int(self.ce_config.get("compression_level"), 9, minimum=1, maximum=9)
        self.min_gain_ratio = coerce_float(self.ce_config.get("min_gain_ratio"), 0.01, minimum=0.0, maximum=0.99)
        self.max_input_bytes = coerce_int(self.ce_config.get("max_input_bytes"), 512 * 1024 * 1024, minimum=0)
        self.max_package_files = coerce_int(self.ce_config.get("max_package_files"), 10_000, minimum=1)
        self.max_expand_files = coerce_int(self.ce_config.get("max_expand_files"), 10_000, minimum=1)
        self.verify = coerce_bool(self.ce_config.get("verify"), True)
        self.verify_text_max_pages = coerce_int(self.ce_config.get("verify_text_max_pages"), 200, minimum=0)
        self.image_max_dimension = coerce_int(self.ce_config.get("image_max_dimension"), 2048, minimum=16)
        self.jpeg_quality = coerce_int(self.ce_config.get("jpeg_quality"), 85, minimum=1, maximum=95)

        self.integrity = integrity or IntegrityEngine()
        self.detector: DetectionEngine = self.integrity.detector
        self.guard: ExpansionGuard = self.integrity.guard
        self.settings = {
            "output_dir": self.output_dir,
            "default_package_format": self.default_package_format,
            "compression_level": self.compression_level,
            "min_gain_ratio": self.min_gain_ratio,
            "max_input_bytes": self.max_input_bytes,
            "max_package_files": self.max_package_files,
            "max_expand_files": self.max_expand_files,
            "verify": self.verify,
            "verify_text_max_pages": self.verify_text_max_pages,
            "image_max_dimension": self.image_max_dimension,
            "jpeg_quality": self.jpeg_quality,
        }
        logger.info("Compression Engine initialized | package=%s level=%s verify=%s", self.default_package_format, self.compression_level, self.verify)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @staticmethod
    def _module(name: str) -> Optional[Any]:
        try:
            return importlib.import_module(name)
        except ImportError:
            return None

    def capabilities(self) -> Dict[str, Any]:
        return {
            "engine_version": COMPRESSION_ENGINE_VERSION,
            "operations": ["package", "optimize", "expand", "compress_file"],
            "package_formats": sorted(PACKAGE_FORMATS),
            "stream_methods": sorted(STREAM_METHODS),
            "optimize_formats": sorted(_ZIP_OPTIMIZABLE | {"pdf", "png", "jpeg (lossy=True only)"}),
            "pdf_optimize_available": self._module("pypdf") is not None,
            "image_optimize_available": self._module("PIL") is not None,
            "settings": dict(self.settings),
        }

    def describe(self) -> Dict[str, Any]:
        return {"status": "ok", **self.capabilities()}

    # ------------------------------------------------------------------
    # Shared plumbing
    # ------------------------------------------------------------------

    @staticmethod
    def _fail(source: Any, method: str, message: str, cause: Optional[BaseException] = None, **context: Any) -> CompressionError:
        return CompressionError(str(source), method, message, cause, context=context or None)

    @staticmethod
    def _normalize_package_format(value: Any) -> str:
        key = str(value or "zip").strip().lower().lstrip(".")
        key = _PACKAGE_ALIASES.get(key, key)
        if key not in PACKAGE_FORMATS:
            raise CompressionError(str(value), "package", f"Unsupported package format '{value}'", context={"supported": sorted(PACKAGE_FORMATS)})
        return key

    @staticmethod
    def _normalize_stream_method(value: Any) -> str:
        key = str(value or "").strip().lower().lstrip(".")
        key = _STREAM_ALIASES.get(key, key)
        if key not in STREAM_METHODS:
            raise CompressionError(str(value), "compress_file", f"Unsupported stream method '{value}'", context={"supported": sorted(STREAM_METHODS)})
        return key

    def _input_file(self, file_path: str | Path, method: str) -> Path:
        path = expand_path(file_path)
        if not path.exists():
            raise FileMissingError(str(path))
        if not path.is_file():
            raise FileNotRegularError(str(path))
        if self.max_input_bytes and path.stat().st_size > self.max_input_bytes:
            raise FileTooLargeError(str(path), int(path.stat().st_size), self.max_input_bytes)
        return path

    def _stage_path(self, out_dir: Path, suffix: str) -> Path:
        """A unique, same-directory temp path so the final publish is a cheap rename."""

        handle, name = tempfile.mkstemp(prefix=".reader_stage_", suffix=suffix + ".part", dir=out_dir)
        os.close(handle)
        return Path(name)

    @staticmethod
    def _discard(path: Optional[Path]) -> None:
        if path is None:
            return
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.warning("Failed removing staged Reader file %s: %s", path, exc)

    def _publish(self, staged: Path, destination: Path, method: str) -> Path:
        try:
            return publish_file(staged, destination, purpose="compressed_output")
        except ReaderError:
            self._discard(staged)
            raise

    def _target(self, output_dir: Optional[str | Path], stem: str, extension: str) -> Tuple[Path, Path]:
        out_dir = ensure_directory(output_dir or self.output_dir, purpose="compression_output_dir")
        return out_dir, out_dir / f"{safe_filename(stem, fallback='archive')}{extension}"

    @staticmethod
    def _hash_handle(handle: Any, *, limit: int) -> Tuple[str, int]:
        digest = hashlib.sha256()
        total = 0
        while True:
            chunk = handle.read(_CHUNK)
            if not chunk:
                break
            total += len(chunk)
            if limit and total > limit:
                raise IntegrityError("stream", "Stream expands beyond the configured limit", context={"limit": limit})
            digest.update(chunk)
        return digest.hexdigest(), total

    def _set_level(self, info: zipfile.ZipInfo) -> None:
        """ZipFile.open(info, 'w') ignores the archive-level setting; set the entry level explicitly."""

        if hasattr(info, "compress_level"):  # public since Python 3.13
            info.compress_level = self.compression_level
        else:
            info._compresslevel = self.compression_level  # type: ignore[attr-defined]

    @staticmethod
    def _safe_component(part: str) -> str:
        cleaned = _COMPONENT_BAD_CHARS.sub("_", part).rstrip(" .")
        if not cleaned:
            return "_"
        if cleaned.split(".")[0].upper() in _WINDOWS_RESERVED:
            cleaned = f"_{cleaned}"
        return cleaned[:200]

    def _safe_relpath(self, name: str) -> Optional[str]:
        """Sanitized relative path for an archive member, or ``None`` when it must not be extracted."""

        if unsafe_member_name(name):
            return None
        parts = [self._safe_component(part) for part in str(name).replace("\\", "/").split("/") if part not in ("", ".")]
        return "/".join(parts) or None

    def _report_result(self, operation: str, report: CompressionReport, timer: OperationTimer,
                       extra: Optional[Mapping[str, Any]] = None, warnings: Optional[Iterable[str]] = None) -> Dict[str, Any]:
        payload = {"report": report.to_dict(), "output_path": report.output_path, "outcome": report.outcome, **dict(extra or {})}
        all_warnings = list(warnings or [])
        if report.outcome == "no_gain":
            all_warnings.append("compression:no_gain")
        if not report.lossless:
            all_warnings.append("compression:lossy")
        if self.verify and not report.verified and report.outcome == "ok":
            all_warnings.append("compression:not_verified")
        return build_success_result(operation=operation, payload=payload, metadata={"engine_version": COMPRESSION_ENGINE_VERSION},
                                    warnings=all_warnings, timer=timer)

    # ------------------------------------------------------------------
    # Packaging
    # ------------------------------------------------------------------

    def _collect_sources(self, files: Iterable[str | Path], *, exclude: Optional[Path] = None) -> List[Tuple[Path, str]]:
        """Resolve inputs (files or directories) to ``(path, arcname)`` pairs with unique, safe names.

        ``exclude`` is the output directory: packaging a folder that contains it must not swallow
        the archive being written (or earlier archives).
        """

        raw = [expand_path(item) for item in files]
        if not raw:
            raise self._fail("", "package", "No files were given to package")
        pairs: List[Tuple[Path, Path]] = []  # (file, base for its relative name)
        for path in raw:
            if not path.exists():
                raise FileMissingError(str(path))
            if path.is_dir():
                for child in sorted(path.rglob("*")):
                    if child.is_symlink():
                        logger.warning("Skipping symbolic link while packaging: %s", child)
                        continue
                    if exclude is not None and is_path_within(child, exclude):
                        continue
                    if child.is_file():
                        pairs.append((child, path.parent))
            elif path.is_file():
                pairs.append((path, path.parent))
            else:
                raise FileNotRegularError(str(path))
            if len(pairs) > self.max_package_files:
                raise self._fail(path, "package", "Too many files to package", limit=self.max_package_files)
        if not pairs:
            raise self._fail("", "package", "Nothing to package: the given directories contain no files")

        seen: Dict[str, int] = {}
        out: List[Tuple[Path, str]] = []
        for path, base in pairs:
            relative = self._safe_relpath(path.relative_to(base).as_posix()) or self._safe_component(path.name)
            key = relative.lower()
            if key in seen:
                seen[key] += 1
                stem, dot, ext = relative.rpartition(".")
                relative = f"{stem}_{seen[key]}.{ext}" if dot else f"{relative}_{seen[key]}"
            else:
                seen[key] = 1
            out.append((path, relative))
        total = sum(path.stat().st_size for path, _ in out)
        if self.max_input_bytes and total > self.max_input_bytes:
            raise FileTooLargeError(str(out[0][0]), total, self.max_input_bytes)
        return out

    def _write_zip(self, staged: Path, sources: Sequence[Tuple[Path, str]]) -> None:
        with zipfile.ZipFile(staged, "w", zipfile.ZIP_DEFLATED, compresslevel=self.compression_level) as archive:
            for path, arcname in sources:
                info = zipfile.ZipInfo(arcname, date_time=_DETERMINISTIC_ZIP_TIME)
                info.compress_type = zipfile.ZIP_DEFLATED
                self._set_level(info)
                info.external_attr = (stat.S_IFREG | 0o644) << 16
                with path.open("rb") as source, archive.open(info, "w", force_zip64=path.stat().st_size > 0xFFFFFFF) as target:
                    while True:
                        chunk = source.read(_CHUNK)
                        if not chunk:
                            break
                        target.write(chunk)

    def _write_tar(self, staged: Path, sources: Sequence[Tuple[Path, str]], package_format: str) -> None:
        def add_all(archive: tarfile.TarFile) -> None:
            for path, arcname in sources:
                info = tarfile.TarInfo(arcname)
                info.size = path.stat().st_size
                info.mtime = 0
                info.mode = 0o644
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                with path.open("rb") as source:
                    archive.addfile(info, source)

        with staged.open("wb") as raw:
            if package_format == "tar.gz":
                with gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=self.compression_level, mtime=0, filename="") as stream:
                    with tarfile.open(fileobj=stream, mode="w", format=tarfile.PAX_FORMAT) as archive:
                        add_all(archive)
            elif package_format == "tar.xz":
                with lzma.LZMAFile(raw, "wb", preset=self.compression_level) as stream:
                    with tarfile.open(fileobj=stream, mode="w", format=tarfile.PAX_FORMAT) as archive:
                        add_all(archive)
            elif package_format == "tar.bz2":
                with bz2.BZ2File(raw, "wb", compresslevel=self.compression_level) as stream:
                    with tarfile.open(fileobj=stream, mode="w", format=tarfile.PAX_FORMAT) as archive:
                        add_all(archive)
            else:
                with tarfile.open(fileobj=raw, mode="w", format=tarfile.PAX_FORMAT) as archive:
                    add_all(archive)

    def _verify_package(self, staged: Path, package_format: str, sources: Sequence[Tuple[Path, str]]) -> None:
        """Round trip: read every member back from the staged archive and compare SHA-256 with its source."""

        expected = {arcname: sha256_file(path) for path, arcname in sources}
        actual: Dict[str, str] = {}
        limit = self.guard.limits.max_member_bytes
        if package_format == "zip":
            with zipfile.ZipFile(staged) as archive:
                for info in archive.infolist():
                    with archive.open(info) as handle:
                        actual[info.filename] = self._hash_handle(handle, limit=limit)[0]
        else:
            with tarfile.open(staged, "r:*") as archive:
                for member in archive:
                    handle = archive.extractfile(member)
                    if handle is not None:
                        actual[member.name] = self._hash_handle(handle, limit=limit)[0]
        if actual != expected:
            raise self._fail(staged, "package", "Round-trip verification failed: archive contents differ from the sources",
                             missing=sorted(set(expected) - set(actual))[:10], mismatched=sorted(k for k in expected if k in actual and actual[k] != expected[k])[:10])

    def package(
        self,
        files: Iterable[str | Path],
        output_dir: Optional[str | Path] = None,
        *,
        package_format: Optional[str] = None,
        name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Bundle files (or whole directories) into one new archive and verify it by round trip."""

        timer = OperationTimer("package")
        fmt = self._normalize_package_format(package_format or self.default_package_format)
        out_dir = ensure_directory(output_dir or self.output_dir, purpose="compression_output_dir")
        sources = self._collect_sources(files, exclude=out_dir)
        stem = name or (Path(sources[0][1]).stem if len(sources) == 1 else "reader_package")
        out_dir, destination = self._target(out_dir, stem, PACKAGE_FORMATS[fmt])
        staged = self._stage_path(out_dir, PACKAGE_FORMATS[fmt])
        try:
            if fmt == "zip":
                self._write_zip(staged, sources)
            else:
                self._write_tar(staged, sources, fmt)
            verified = False
            if self.verify:
                self._verify_package(staged, fmt, sources)
                verified = True
            report = CompressionReport(
                operation="package", method=fmt, input_bytes=sum(path.stat().st_size for path, _ in sources),
                output_bytes=staged.stat().st_size, lossless=True, verified=verified,
                verified_by="round_trip_sha256" if verified else "not_verified", input_files=len(sources))
            final = self._publish(staged, destination, fmt)
        except (ReaderError, IntegrityError):
            self._discard(staged)
            raise
        except (OSError, zipfile.BadZipFile, tarfile.TarError, lzma.LZMAError, EOFError, zlib.error) as exc:
            self._discard(staged)
            raise self._fail(destination, fmt, f"Failed building the archive: {exc}", exc) from exc
        report.output_path = str(final)
        logger.info("Packaged %d file(s) -> %s (%d -> %d bytes)", len(sources), final, report.input_bytes, report.output_bytes)
        return self._report_result("package", report, timer, extra={"members": [arc for _, arc in sources]})

    # ------------------------------------------------------------------
    # Single-stream compression
    # ------------------------------------------------------------------

    def compress_file(self, file_path: str | Path, output_dir: Optional[str | Path] = None, *, method: str = "gz") -> Dict[str, Any]:
        """Compress one file as a gzip/bzip2/xz stream (``name.ext`` -> ``name.ext.gz``), verified by hash."""

        timer = OperationTimer("compress_file")
        key = self._normalize_stream_method(method)
        path = self._input_file(file_path, key)
        suffix = STREAM_METHODS[key]
        out_dir = ensure_directory(output_dir or self.output_dir, purpose="compression_output_dir")
        destination = out_dir / f"{safe_filename(path.name, fallback='file')}{suffix}"
        staged = self._stage_path(out_dir, suffix)
        try:
            with path.open("rb") as source, staged.open("wb") as raw:
                if key == "gz":
                    with gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=self.compression_level, mtime=0, filename="") as target:
                        self._copy(source, target)
                elif key == "bz2":
                    with bz2.BZ2File(raw, "wb", compresslevel=self.compression_level) as target:
                        self._copy(source, target)
                else:
                    with lzma.LZMAFile(raw, "wb", preset=self.compression_level) as target:
                        self._copy(source, target)
            verified = False
            if self.verify:
                opener = {"gz": gzip.open, "bz2": bz2.open, "xz": lzma.open}[key]
                with opener(staged, "rb") as handle:
                    digest, _ = self._hash_handle(handle, limit=self.guard.limits.max_total_bytes)
                if digest != sha256_file(path):
                    raise self._fail(path, key, "Round-trip verification failed: decompressed bytes differ from the source")
                verified = True
            report = CompressionReport(operation="compress_file", method=key, input_bytes=path.stat().st_size, output_bytes=staged.stat().st_size,
                                       lossless=True, verified=verified, verified_by="round_trip_sha256" if verified else "not_verified")
            final = self._publish(staged, destination, key)
        except (ReaderError, IntegrityError):
            self._discard(staged)
            raise
        except (OSError, EOFError, lzma.LZMAError, zlib.error) as exc:
            self._discard(staged)
            raise self._fail(path, key, f"Failed compressing the file: {exc}", exc) from exc
        report.output_path = str(final)
        return self._report_result("compress_file", report, timer)

    @staticmethod
    def _copy(source: Any, target: Any) -> None:
        while True:
            chunk = source.read(_CHUNK)
            if not chunk:
                break
            target.write(chunk)

    # ------------------------------------------------------------------
    # Optimization (same format, smaller file)
    # ------------------------------------------------------------------

    def _optimize_zip(self, path: Path, staged: Path, report: CompressionReport) -> None:
        source_hashes: Dict[str, str] = {}
        limit = self.guard.limits.max_member_bytes
        with zipfile.ZipFile(path) as source:
            self.guard.inspect_zip(source, source=str(path))
            if any(info.flag_bits & 0x1 for info in source.infolist()):
                raise self._fail(path, "zip_recompress", "Encrypted ZIP members cannot be optimized")
            with zipfile.ZipFile(staged, "w", zipfile.ZIP_DEFLATED, compresslevel=self.compression_level) as target:
                for info in source.infolist():
                    new_info = zipfile.ZipInfo(info.filename, date_time=info.date_time)
                    new_info.external_attr = info.external_attr
                    new_info.comment = info.comment
                    new_info.create_system = info.create_system
                    # ODF and EPUB readers require the mimetype entry stored, uncompressed and first.
                    new_info.compress_type = zipfile.ZIP_STORED if info.filename == "mimetype" else (
                        zipfile.ZIP_STORED if info.is_dir() else zipfile.ZIP_DEFLATED)
                    if new_info.compress_type == zipfile.ZIP_DEFLATED:
                        self._set_level(new_info)
                    digest = hashlib.sha256()
                    total = 0
                    with source.open(info) as reader, target.open(new_info, "w", force_zip64=info.file_size > 0xFFFFFFF) as writer:
                        while True:
                            chunk = reader.read(_CHUNK)
                            if not chunk:
                                break
                            total += len(chunk)
                            if limit and total > limit:
                                raise IntegrityError(str(path), "Archive member expands beyond the configured limit", context={"member": info.filename})
                            digest.update(chunk)
                            writer.write(chunk)
                    source_hashes[info.filename] = digest.hexdigest()
        if self.verify:
            with zipfile.ZipFile(staged) as result:
                result_hashes = {}
                for info in result.infolist():
                    with result.open(info) as handle:
                        result_hashes[info.filename] = self._hash_handle(handle, limit=limit)[0]
            if result_hashes != source_hashes:
                raise self._fail(path, "zip_recompress", "Round-trip verification failed: member contents changed")
            report.verified, report.verified_by = True, "member_sha256"

    def _optimize_pdf(self, path: Path, staged: Path, report: CompressionReport) -> None:
        pypdf = self._module("pypdf")
        if pypdf is None:
            raise self._fail(path, "pdf_optimize", "PDF optimization needs the optional 'pypdf' package")
        try:
            reader = pypdf.PdfReader(str(path), strict=False)
            if getattr(reader, "is_encrypted", False):
                raise self._fail(path, "pdf_optimize", "Encrypted PDFs cannot be optimized")
            pages_before = len(reader.pages)
            writer = pypdf.PdfWriter(clone_from=reader)
            for page in writer.pages:
                page.compress_content_streams()
            writer.compress_identical_objects(remove_identicals=True, remove_orphans=True)
            with staged.open("wb") as handle:
                writer.write(handle)
        except CompressionError:
            raise
        except Exception as exc:  # pypdf raises many unrelated exception types for damaged files
            raise self._fail(path, "pdf_optimize", f"PDF could not be rewritten: {type(exc).__name__}: {exc}", exc) from exc
        if not self.verify:
            return
        try:
            after = pypdf.PdfReader(str(staged), strict=False)
            if len(after.pages) != pages_before:
                raise self._fail(path, "pdf_optimize", "Verification failed: page count changed", before=pages_before, after=len(after.pages))
            if pages_before <= self.verify_text_max_pages:
                for index in range(pages_before):
                    if (reader.pages[index].extract_text() or "") != (after.pages[index].extract_text() or ""):
                        raise self._fail(path, "pdf_optimize", "Verification failed: page text changed", page=index + 1)
                report.verified, report.verified_by = True, "page_count+page_text"
            else:
                report.verified, report.verified_by = True, "page_count"
                report.notes.append(f"text comparison skipped: more than {self.verify_text_max_pages} pages")
        except CompressionError:
            raise
        except Exception as exc:
            raise self._fail(path, "pdf_optimize", f"Verification could not read the optimized PDF: {exc}", exc) from exc
        report.notes.append("lossless for page content; interactive features are carried over by pypdf but not independently compared")

    def _optimize_image(self, path: Path, staged: Path, format_id: str, lossy: bool, report: CompressionReport) -> None:
        pil = self._module("PIL.Image")
        if pil is None:
            raise self._fail(path, "image_optimize", "Image optimization needs the optional 'Pillow' package")
        chops = self._module("PIL.ImageChops")
        try:
            with pil.open(path) as image:
                image.load()
                if getattr(image, "n_frames", 1) > 1:
                    raise self._fail(path, "image_optimize", "Animated images are not optimized (frames would be lost)")
                if format_id == "png":
                    keep = {key: image.info[key] for key in ("transparency", "icc_profile", "dpi", "gamma") if key in image.info}
                    image.save(staged, format="PNG", optimize=True, compress_level=9, **keep)
                    report.method = "png_optimize"
                    if self.verify:
                        with pil.open(staged) as result:
                            result.load()
                            if result.size != image.size or result.mode != image.mode or (chops is not None and chops.difference(image, result).getbbox() is not None):
                                raise self._fail(path, "png_optimize", "Verification failed: pixels changed")
                        report.verified, report.verified_by = True, "pixel_equality"
                elif format_id == "jpeg" and lossy:
                    work = image.copy()
                    work.thumbnail((self.image_max_dimension, self.image_max_dimension))
                    work.save(staged, format="JPEG", quality=self.jpeg_quality, optimize=True, icc_profile=image.info.get("icc_profile"))
                    report.method, report.lossless = "jpeg_recompress", False
                    report.notes.append(f"lossy: max dimension {self.image_max_dimension}px, quality {self.jpeg_quality}")
                    if self.verify:
                        with pil.open(staged) as result:
                            result.load()
                        report.verified, report.verified_by = True, "decodes"
                else:
                    raise self._fail(path, "image_optimize", "Lossless optimization is only available for PNG; pass lossy=True to recompress JPEG",
                                     format=format_id)
        except CompressionError:
            raise
        except Exception as exc:
            raise self._fail(path, "image_optimize", f"Image could not be processed: {type(exc).__name__}: {exc}", exc) from exc

    def optimize(self, file_path: str | Path, output_dir: Optional[str | Path] = None, *, lossy: bool = False) -> Dict[str, Any]:
        """Make one file smaller while keeping its format. Returns ``no_gain`` (and no file) if it cannot."""

        timer = OperationTimer("optimize")
        path = self._input_file(file_path, "optimize")
        detection = self.detector.detect(path)["payload"]
        format_id = detection.get("format_id") or "unknown"
        if detection.get("damaged") or detection.get("encrypted"):
            raise self._fail(path, "optimize", "Damaged or encrypted files cannot be optimized", format=format_id,
                             damaged=bool(detection.get("damaged")), encrypted=bool(detection.get("encrypted")))
        extension = path.suffix or f".{format_id}"
        out_dir = ensure_directory(output_dir or self.output_dir, purpose="compression_output_dir")
        destination = prepare_output_path(out_dir, path.stem, extension, suffix="optimized")
        staged = self._stage_path(out_dir, extension)
        report = CompressionReport(operation="optimize", method=format_id, input_bytes=path.stat().st_size)
        try:
            if format_id in _ZIP_OPTIMIZABLE:
                report.method = "zip_recompress"
                self._optimize_zip(path, staged, report)
            elif format_id == "pdf":
                report.method = "pdf_optimize"
                self._optimize_pdf(path, staged, report)
            elif format_id in {"png", "jpeg"}:
                self._optimize_image(path, staged, format_id, lossy, report)
            else:
                raise self._fail(path, "optimize", f"Optimization is not supported for '{format_id}'; use package or compress_file instead",
                                 format=format_id, supported=sorted(_ZIP_OPTIMIZABLE | {"pdf", "png", "jpeg"}))
            report.output_bytes = staged.stat().st_size
            gain_needed = report.input_bytes * (1.0 - self.min_gain_ratio)
            if report.output_bytes >= gain_needed:
                self._discard(staged)
                report.outcome, report.output_bytes, report.verified = "no_gain", report.input_bytes, False
                report.notes.append("no meaningful saving; the original was kept")
                logger.info("Optimize %s: no gain (%d -> %d bytes); original kept", path, report.input_bytes, staged.stat().st_size if staged.exists() else report.input_bytes)
                return self._report_result("optimize", report, timer)
            final = self._publish(staged, destination, report.method)
        except (ReaderError, IntegrityError):
            self._discard(staged)
            raise
        except (OSError, zipfile.BadZipFile, EOFError, zlib.error) as exc:
            self._discard(staged)
            raise self._fail(path, report.method, f"Failed optimizing the file: {exc}", exc) from exc
        report.output_path = str(final)
        if self.integrity.verify_outputs and self.verify:
            check = self.integrity.verify_output(final, format_id)
            if not check["payload"]["valid"]:
                raise self._fail(path, report.method, "Optimized file failed integrity validation and was left in place for inspection",
                                 output=str(final), failed=[c["name"] for c in check["payload"]["checks"] if not c["ok"]])
        return self._report_result("optimize", report, timer)

    # ------------------------------------------------------------------
    # Dispatch used by the agent
    # ------------------------------------------------------------------

    def compress(
        self,
        files: Iterable[str | Path],
        output_dir: Optional[str | Path] = None,
        *,
        method: str = "auto",
        package_format: Optional[str] = None,
        lossy: bool = False,
        name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """One entry point: ``auto`` optimizes a single optimizable file, otherwise packages.

        ``method`` may also be ``package``, ``optimize`` or a stream method (``gz``, ``bz2``, ``xz``).
        """

        items = list(files)
        key = str(method or "auto").strip().lower()
        if key in {"package", "archive", "zip"} or key in PACKAGE_FORMATS or key in _PACKAGE_ALIASES:
            fmt = package_format or (key if key in PACKAGE_FORMATS or key in _PACKAGE_ALIASES else None)
            return self.package(items, output_dir, package_format=fmt, name=name)
        if key == "optimize":
            if len(items) != 1:
                raise self._fail("", "optimize", "Optimize works on exactly one file", count=len(items))
            return self.optimize(items[0], output_dir, lossy=lossy)
        if key in STREAM_METHODS or key in _STREAM_ALIASES:
            if len(items) != 1:
                raise self._fail("", key, "Stream compression works on exactly one file; use package for several", count=len(items))
            return self.compress_file(items[0], output_dir, method=key)
        if key != "auto":
            raise self._fail("", key, f"Unknown compression method '{method}'",
                             supported=["auto", "package", "optimize", *sorted(STREAM_METHODS), *sorted(PACKAGE_FORMATS)])
        if len(items) == 1 and expand_path(items[0]).is_file():
            detected = self.detector.detect(expand_path(items[0]))["payload"].get("format_id")
            if detected in (_ZIP_OPTIMIZABLE | {"pdf", "png"}):
                return self.optimize(items[0], output_dir, lossy=lossy)
        return self.package(items, output_dir, package_format=package_format, name=name)

    # ------------------------------------------------------------------
    # Expansion
    # ------------------------------------------------------------------

    def _unique_dir(self, base: Path, stem: str) -> Path:
        base.mkdir(parents=True, exist_ok=True)
        safe = safe_filename(stem, fallback="extracted")
        for attempt in range(100):
            candidate = base / (safe if attempt == 0 else f"{safe}_{utc_compact_timestamp()}_{uuid.uuid4().hex[:8]}")
            try:
                candidate.mkdir()
                return candidate
            except FileExistsError:
                continue
        raise self._fail(base, "expand", "Failed finding a free extraction directory")

    def _extract_stream(self, source: Any, destination: Path, state: Dict[str, Any], archive: Path) -> Dict[str, Any]:
        digest = hashlib.sha256()
        total = 0
        member_cap = self.guard.limits.max_member_bytes
        total_cap = self.guard.limits.max_total_bytes
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as target:
            while True:
                chunk = source.read(_CHUNK)
                if not chunk:
                    break
                total += len(chunk)
                state["bytes"] += len(chunk)
                if member_cap and total > member_cap:
                    raise IntegrityError(str(archive), "Archive member expands beyond the configured limit", context={"member": destination.name, "limit": member_cap})
                if total_cap and state["bytes"] > total_cap:
                    raise IntegrityError(str(archive), "Archive expands beyond the configured total limit", context={"limit": total_cap})
                digest.update(chunk)
                target.write(chunk)
        return {"size_bytes": total, "sha256": digest.hexdigest()}

    def expand(
        self,
        archive_path: str | Path,
        output_dir: Optional[str | Path] = None,
    ) -> Dict[str, Any]:
        """Safely extract a zip, tar (optionally gz/xz/bz2) or single-stream archive into a new directory.

        Nested archives are listed, never expanded. Unsafe members, links and devices are skipped and
        reported. On any limit violation the partial output is removed.
        """

        timer = OperationTimer("expand")
        path = self._input_file(archive_path, "expand")
        detection = self.detector.detect(path)["payload"]
        format_id = detection.get("format_id") or "unknown"
        if detection.get("damaged"):
            raise self._fail(path, "expand", "Archive is damaged; try the salvage operation first", format=format_id)
        if detection.get("encrypted"):
            raise self._fail(path, "expand", "Encrypted archives cannot be expanded", format=format_id)
        container = detection.get("container")
        is_zip = container == "zip" or format_id == "zip"
        if not (is_zip or format_id in {"tar", "tar_gz", "gzip", "bzip2", "xz"}):
            raise self._fail(path, "expand", f"'{format_id}' is not an archive Reader can expand", format=format_id)

        stem = path.name
        for ext in sorted(_ARCHIVE_EXTENSIONS, key=len, reverse=True):
            if stem.lower().endswith(ext):
                stem = stem[: -len(ext)] or path.stem
                break
        base = expand_path(output_dir or self.output_dir)
        target_root = self._unique_dir(base, stem)
        state: Dict[str, Any] = {"bytes": 0}
        files: List[Dict[str, Any]] = []
        skipped: List[Dict[str, str]] = []
        warnings: List[str] = []
        try:
            if is_zip:
                self._expand_zip(path, target_root, state, files, skipped)
            elif format_id in {"tar", "tar_gz"} or (format_id in {"xz", "bzip2", "gzip"} and tarfile.is_tarfile(path)):
                self._expand_tar(path, target_root, state, files, skipped)
            else:
                opener = {"gzip": gzip.open, "bzip2": bz2.open, "xz": lzma.open}[format_id]
                name = self._safe_component(stem if stem != path.name else f"{path.name}.out")
                with opener(path, "rb") as handle:
                    record = self._extract_stream(handle, target_root / name, state, path)
                files.append({"path": name, **record})
        except (ReaderError, IntegrityError):
            self._cleanup_dir(target_root)
            raise
        except (OSError, zipfile.BadZipFile, tarfile.TarError, EOFError, lzma.LZMAError, zlib.error) as exc:
            self._cleanup_dir(target_root)
            raise self._fail(path, "expand", f"Failed expanding the archive: {exc}", exc) from exc

        if skipped:
            warnings.append(f"expand:skipped_{len(skipped)}_members")
        nested = [item["path"] for item in files if item["path"].lower().endswith(_ARCHIVE_EXTENSIONS)]
        if nested:
            warnings.append("expand:nested_archives_not_expanded")
        report = CompressionReport(operation="expand", method=format_id, input_bytes=path.stat().st_size, output_bytes=state["bytes"], lossless=True,
                                   verified=True, verified_by="stream_checked_during_extract", output_path=str(target_root), input_files=len(files))
        logger.info("Expanded %s -> %s (%d files, %d bytes)", path, target_root, len(files), state["bytes"])
        return self._report_result("expand", report, timer, warnings=warnings,
                                   extra={"output_dir": str(target_root), "files": files, "skipped": skipped, "nested_archives": nested})

    @staticmethod
    def _cleanup_dir(root: Path) -> None:
        for child in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
            try:
                child.rmdir() if child.is_dir() else child.unlink()
            except OSError:
                pass
        try:
            root.rmdir()
        except OSError:
            logger.warning("Partial extraction left in %s", root)

    def _place(self, root: Path, name: str, skipped: List[Dict[str, str]]) -> Optional[Tuple[str, Path]]:
        relative = self._safe_relpath(name)
        if relative is None:
            skipped.append({"member": name, "reason": "unsafe_path"})
            return None
        destination = root / relative
        if not is_path_within(destination, root):
            skipped.append({"member": name, "reason": "outside_target"})
            return None
        return relative, destination

    def _expand_zip(self, path: Path, root: Path, state: Dict[str, Any], files: List[Dict[str, Any]], skipped: List[Dict[str, str]]) -> None:
        with zipfile.ZipFile(path) as archive:
            self.guard.inspect_zip(archive, source=str(path))
            for info in archive.infolist():
                if len(files) >= self.max_expand_files:
                    raise self._fail(path, "expand", "Archive has too many members to expand", limit=self.max_expand_files)
                if info.flag_bits & 0x1:
                    skipped.append({"member": info.filename, "reason": "encrypted"})
                    continue
                if stat.S_ISLNK(info.external_attr >> 16):
                    skipped.append({"member": info.filename, "reason": "symlink"})
                    continue
                placed = self._place(root, info.filename, skipped)
                if placed is None:
                    continue
                relative, destination = placed
                if info.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                if destination.exists():
                    skipped.append({"member": info.filename, "reason": "duplicate_name"})
                    continue
                with archive.open(info) as handle:
                    files.append({"path": relative, **self._extract_stream(handle, destination, state, path)})

    def _expand_tar(self, path: Path, root: Path, state: Dict[str, Any], files: List[Dict[str, Any]], skipped: List[Dict[str, str]]) -> None:
        count = 0
        with tarfile.open(path, "r:*") as archive:
            for member in archive:
                count += 1
                if count > self.max_expand_files:
                    raise self._fail(path, "expand", "Archive has too many members to expand", limit=self.max_expand_files)
                if member.issym() or member.islnk() or member.isdev() or member.isfifo():
                    skipped.append({"member": member.name, "reason": "link_or_special"})
                    continue
                placed = self._place(root, member.name, skipped)
                if placed is None:
                    continue
                relative, destination = placed
                if member.isdir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                if not member.isfile():
                    skipped.append({"member": member.name, "reason": "unsupported_type"})
                    continue
                if destination.exists():
                    skipped.append({"member": member.name, "reason": "duplicate_name"})
                    continue
                handle = archive.extractfile(member)
                if handle is None:
                    skipped.append({"member": member.name, "reason": "unreadable"})
                    continue
                with handle:
                    files.append({"path": relative, **self._extract_stream(handle, destination, state, path)})


if __name__ == "__main__":
    print("\n=== Running Compression Engine ===\n")
    printer.status("TEST", "Compression Engine initialized", "info")

    with tempfile.TemporaryDirectory(prefix="reader_compression_test_") as tmp_dir:
        tmp = Path(tmp_dir)
        engine = CompressionEngine()
        out = tmp / "out"

        first = tmp / "a.txt"
        second = tmp / "b.txt"
        first.write_text("alpha " * 5000, encoding="utf-8")
        second.write_text("beta " * 5000, encoding="utf-8")

        for fmt in ("zip", "tar.gz", "tar.xz"):
            result = engine.package([first, second], out, package_format=fmt, name=f"bundle_{fmt.replace('.', '_')}")
            report = result["payload"]["report"]
            assert report["verified"] and report["output_bytes"] < report["input_bytes"], report
        printer.status("TEST", "Package zip/tar.gz/tar.xz verified by round trip", "success")

        one = engine.package([first, second], out, name="det")
        two = engine.package([first, second], out, name="det")
        assert one["payload"]["output_path"] != two["payload"]["output_path"]
        assert sha256_file(one["payload"]["output_path"]) == sha256_file(two["payload"]["output_path"])
        printer.status("TEST", "Archives are deterministic and never overwrite", "success")

        gz = engine.compress_file(first, out, method="gz")
        assert gz["payload"]["report"]["verified"]
        expanded = engine.expand(gz["payload"]["output_path"], tmp / "expanded")
        assert expanded["payload"]["files"][0]["sha256"] == sha256_file(first)
        printer.status("TEST", "gzip stream compress and expand round trip", "success")

        zipped = engine.expand(one["payload"]["output_path"], tmp / "expanded")
        assert {item["path"] for item in zipped["payload"]["files"]} == {"a.txt", "b.txt"}
        printer.status("TEST", "Expand reproduces the original members", "success")

        evil = tmp / "evil.zip"
        with zipfile.ZipFile(evil, "w") as handle:
            handle.writestr("../escape.txt", "x")
            handle.writestr("good.txt", "ok")
        result = engine.expand(evil, tmp / "expanded")
        assert [item["path"] for item in result["payload"]["files"]] == ["good.txt"]
        assert result["payload"]["skipped"][0]["reason"] == "unsafe_path"
        assert not (tmp / "escape.txt").exists() and not (tmp / "expanded" / "escape.txt").exists()
        printer.status("TEST", "Zip-slip member skipped and reported", "success")

    print("\n=== Test ran successfully ===\n")
