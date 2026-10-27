"""Shared lifecycle for the public SDRW/SGEO document objects.

The facades ``src/utils/sdrw.py`` and ``src/utils/sgeo.py`` expose small,
stable classes (``SDRWDocument``/``SGEODocument``). Everything those classes
share — opening and saving packages, optimistic concurrency, read-only mode,
dirty tracking, validation reports, inspection — lives here, once, so the two
formats cannot drift apart. Format-specific editing and queries stay in the
facades; the models (:class:`Drawing`, :class:`Scene`) stay the single source of
truth for invariants.

Concurrency model
-----------------
A document is safe to use from several threads: every mutation runs under the
model's re-entrant lock and either completes or leaves the model unchanged.
Across *processes* (or editors) the contract is compare-and-swap on disk:
``save()`` back to the file a document was opened from refuses to overwrite it
if the file changed since (digest check), and callers may additionally pin an
``expected_revision`` / ``expected_sha256``. The check-then-replace window on
the filesystem is short but not zero; callers needing strict serialisation of
writers must provide their own lock.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from .common.atomic import write_bytes_atomic
from .common.errors import (ConflictError, ReadOnlyDocumentError, SpatialFormatError,
                            UnsupportedFeatureError, ValidationError)
from .common.helpers import MAX_DOCUMENT_NODES, canonical_json, digest
from .common.metadata import require_metadata
from .container.inspect import PackageInfo, inspect_bytes
from .container.reader import (MAX_ARCHIVE_SIZE, MAX_UNPACKED, SpatialPackage, loads_package,
                               read_source_bytes)
from .container.versions import CURRENT_VERSION
from .container.writer import build_package
from .verification.report import ValidationReport

__all__ = ["SaveResult", "SpatialDocument", "MODES"]

MODES = ("read", "write")


@dataclass(frozen=True)
class SaveResult:
    """What a successful :meth:`SpatialDocument.save` published."""

    path: str
    sha256: str
    size: int
    document_id: str
    revision: int
    replaced: bool

    def as_dict(self) -> dict:
        return {"path": self.path, "sha256": self.sha256, "size": self.size,
                "document_id": self.document_id, "revision": self.revision, "replaced": self.replaced}


class SpatialDocument:
    """Base class: one validated in-memory model plus its on-disk identity."""

    #: Overridden by subclasses.
    FORMAT_ID: str = ""
    FILE_SUFFIX: str = ""
    KIND_LABEL: str = ""

    # -- subclass hooks -------------------------------------------------------
    @classmethod
    def _model_from_package(cls, package: SpatialPackage):  # pragma: no cover - abstract
        raise NotImplementedError

    def _check(self, mode: str, max_issues: int) -> ValidationReport:  # pragma: no cover - abstract
        raise NotImplementedError

    def _apply(self, operations, expected_revision):  # pragma: no cover - abstract
        raise NotImplementedError

    def _counts(self) -> dict:  # pragma: no cover - abstract
        raise NotImplementedError

    def _item_metadata(self, item_id: str) -> dict:  # pragma: no cover - abstract
        raise NotImplementedError

    def attach_semantics(self, item_id: str, values: dict, *, replace: bool = False,
                         namespace: str | None = None) -> dict:  # pragma: no cover - abstract
        raise NotImplementedError

    # -- construction ---------------------------------------------------------
    def __init__(self, model, *, path=None, source_sha256=None, read_only=False, source_version=None):
        self._model = model
        self._source_version: str | None = source_version
        self._path: Path | None = Path(path) if path is not None else None
        self._source_sha256: str | None = source_sha256
        self._saved_revision: int | None = model.revision if source_sha256 is not None else None
        self._read_only = bool(read_only)
        self._closed = False

    @classmethod
    def open(cls, path, *, mode: str = "read", max_archive_size: int = MAX_ARCHIVE_SIZE,
             max_unpacked: int = MAX_UNPACKED):
        """Load and fully validate a package file.

        ``mode="read"`` (default) yields an immutable document: every mutator
        raises :class:`ReadOnlyDocumentError`. ``mode="write"`` yields an
        editable one that remembers the file's digest for conflict-safe saves.
        The file is read once into memory, so the digest, the parse and the
        reported size always describe the same bytes.
        """
        if mode not in MODES:
            raise ValidationError("mode must be 'read' or 'write'")
        data = read_source_bytes(path, max_archive_size=max_archive_size)
        package = loads_package(data, max_archive_size=max_archive_size, max_unpacked=max_unpacked)
        cls._require_format(package)
        return cls(cls._model_from_package(package), path=Path(path).absolute(),
                   source_sha256=sha256(data).hexdigest(), read_only=(mode == "read"),
                   source_version=package.source_version)

    @classmethod
    def from_bytes(cls, data, *, mode: str = "write", max_unpacked: int = MAX_UNPACKED):
        """Load a package held in memory (not bound to any file)."""
        if mode not in MODES:
            raise ValidationError("mode must be 'read' or 'write'")
        package = loads_package(data, max_unpacked=max_unpacked)
        cls._require_format(package)
        return cls(cls._model_from_package(package), read_only=(mode == "read"),
                   source_version=package.source_version)

    @classmethod
    def from_package(cls, package: SpatialPackage, *, mode: str = "write"):
        cls._require_format(package)
        return cls(cls._model_from_package(package), read_only=(mode == "read"),
                   source_version=package.source_version)

    @classmethod
    def _require_format(cls, package: SpatialPackage) -> None:
        found = package.manifest.get("format_id")
        if found != cls.FORMAT_ID:
            raise UnsupportedFeatureError(
                f"expected a {cls.FORMAT_ID} package but found {found}",
                context={"expected": cls.FORMAT_ID, "found": found})

    # -- state ------------------------------------------------------------------
    @property
    def model(self):
        """The underlying model. Read-only documents return a detached copy.

        Mutating the live model directly bypasses read-only enforcement and
        dirty tracking; prefer the document's own methods.
        """
        return self._model._detached_copy() if self._read_only else self._model

    @property
    def path(self) -> Path | None:
        return self._path

    @property
    def source_version(self) -> str | None:
        """Format version of the file this document was loaded from (``None`` for a new document).

        A legacy file (for example ``0.1.0``) is upgraded in memory on load; saving writes the
        current version, after which this reports it.
        """
        return self._source_version

    @property
    def source_sha256(self) -> str | None:
        """SHA-256 of the file bytes this document was last opened from / saved as."""
        return self._source_sha256

    @property
    def read_only(self) -> bool:
        return self._read_only

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def document_id(self) -> str:
        return self._model.document_id

    @property
    def revision(self) -> int:
        return self._model.revision

    @property
    def unit(self) -> str:
        return self._model.unit

    @property
    def name(self) -> str:
        return self._model.name

    @property
    def metadata(self) -> dict:
        return deepcopy(self._model.metadata)

    @property
    def saved_revision(self) -> int | None:
        """Revision last persisted or loaded; ``None`` for a never-saved document."""
        return self._saved_revision

    @property
    def dirty(self) -> bool:
        """True when the model changed since it was loaded or last saved."""
        return self._saved_revision is None or self._model.revision != self._saved_revision

    # -- guards -----------------------------------------------------------------
    def _writable(self) -> None:
        if self._closed:
            raise ReadOnlyDocumentError("document is closed")
        if self._read_only:
            raise ReadOnlyDocumentError("document was opened read-only; reopen with mode='write'")

    def _live(self) -> None:
        if self._closed:
            raise ReadOnlyDocumentError("document is closed")

    @contextmanager
    def _edit(self):
        """Serialise one logical edit and refuse it on read-only/closed documents."""
        self._writable()
        with self._model._mutation_lock:
            yield self._model

    def close(self) -> None:
        """Forbid further edits/saves. Idempotent; never touches the file."""
        self._closed = True

    def __enter__(self):
        self._live()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    # -- document-level edits -----------------------------------------------------
    def rename(self, name: str) -> str:
        """Set the document name (one revision)."""
        if not isinstance(name, str) or not name or len(name) > 255:
            raise ValidationError("name must be a non-empty string of at most 255 characters")
        with self._edit() as model:
            model.name = name
            model.revision += 1
        return name

    def set_metadata(self, values: dict, *, replace: bool = False) -> dict:
        """Merge (or replace) document-level metadata; keys are ``namespace:name`` or ``slai.*``."""
        safe = _checked_metadata(values)
        with self._edit() as model:
            merged = safe if replace else {**model.metadata, **safe}
            require_metadata(merged)
            model.metadata = merged
            model.revision += 1
            return deepcopy(model.metadata)

    def add_claim(self, item_id: str, predicate: str, *, evidence, confidence: float = 1.0,
                  kind: str = "assertion", agent: str = "") -> dict:
        """Attach an attributable claim to an entity/object (stored under metadata ``slai.claims``).

        Claims are inert data with provenance (who asserted it, with what
        evidence and confidence); nothing here verifies authorship.
        """
        from datetime import datetime, timezone
        from .common.ids import require_id
        from .common.metadata import require_claim
        claim = require_claim({"subject": require_id(item_id), "predicate": predicate, "evidence": evidence,
                               "confidence": confidence, "kind": kind, "agent": agent,
                               "timestamp": datetime.now(timezone.utc).isoformat()})
        existing = list(self._item_metadata(item_id).get("slai.claims", []))
        self.attach_semantics(item_id, {"slai.claims": existing + [claim]})
        return claim

    def apply(self, operations: list, *, expected_revision: int | None = None) -> list:
        """Apply a batch atomically as ONE revision (all-or-nothing).

        Pass ``expected_revision`` to refuse a stale patch: if another actor
        changed the document, :class:`ConflictError` is raised and nothing
        changes. Operation vocabulary is documented on the concrete class.
        """
        self._writable()
        return self._apply(operations, expected_revision)

    # -- views ----------------------------------------------------------------------
    def snapshot(self):
        """A detached deep copy of the model (safe to mutate independently)."""
        with self._model._mutation_lock:
            return self._model._detached_copy()

    def manifest(self) -> dict:
        return self._model.manifest()

    def as_dict(self) -> dict:
        """The validated payload exactly as it would be written (detached copy)."""
        return self._model.as_dict()

    def content_digest(self) -> str:
        """SHA-256 of the canonical manifest+payload; equal for equal logical content."""
        with self._model._mutation_lock:
            body = canonical_json({"manifest": self._model.manifest(), "document": self._model.as_dict()},
                                  max_nodes=MAX_DOCUMENT_NODES)
        return digest(body)

    def validate(self, mode: str = "strict", *, max_issues: int = 200,
                 raise_on_error: bool = False) -> ValidationReport:
        """Return a structured :class:`ValidationReport` (never a bare boolean).

        ``mode="basic"`` checks the in-memory model; ``mode="strict"`` also
        proves schema conformance, save budgets and a deterministic save/load
        round trip. ``raise_on_error=True`` raises the first error as its typed
        exception (with every error in ``exc.context["issues"]``).
        """
        self._live()
        report = self._check(mode, max_issues)
        if raise_on_error:
            report.raise_for_errors()
        return report

    def inspect(self) -> dict:
        """JSON-safe summary of this document (identity, counts, persistence state)."""
        info = {
            "format_id": self.FORMAT_ID, "format_version": CURRENT_VERSION, "kind": self.KIND_LABEL,
            "name": self.name, "document_id": self.document_id, "revision": self.revision,
            "unit": self.unit, "read_only": self._read_only, "closed": self._closed,
            "dirty": self.dirty, "saved_revision": self._saved_revision,
            "path": str(self._path) if self._path else None, "source_sha256": self._source_sha256,
            "source_version": self._source_version,
            "metadata_keys": sorted(self._model.metadata),
        }
        info.update(self._counts())
        return info

    # -- persistence ------------------------------------------------------------------
    def dumps(self) -> bytes:
        """Serialise to a verified package in memory (the exact bytes ``save`` would write)."""
        self._live()
        return self._build()[0]

    def _build(self):
        with self._model._mutation_lock:
            manifest = self._model.manifest()
            payload = self._model.as_dict()
            revision = self._model.revision
            textures = dict(getattr(self._model, "image_data", None) or {}) or None
        data, final_manifest = build_package(manifest, payload, textures=textures)
        return data, final_manifest, revision

    def package_info(self) -> PackageInfo:
        """Metadata-only description of what :meth:`dumps` would produce."""
        return inspect_bytes(self.dumps(), verify="digests")

    def save(self, path=None, *, overwrite: bool | None = None, expected_revision: int | None = None,
             expected_sha256: str | None = None, force: bool = False,
             allow_any_suffix: bool = False) -> SaveResult:
        """Persist atomically and verify before publishing.

        * ``path`` defaults to the file the document was opened from.
        * ``overwrite`` defaults to ``True`` only when saving back to that same
          file, and ``False`` for any other destination.
        * When replacing the file a document was opened from, the file's current
          digest must still equal ``source_sha256``; otherwise another writer got
          there first and :class:`ConflictError` is raised (``force=True`` skips
          this check *and* the identity checks below).
        * ``expected_revision`` / ``expected_sha256`` pin the revision / digest
          of the file being replaced (compare-and-swap).
        * Replacing a file that is not a valid package of this format, or that
          holds a *different* document id, requires ``force=True``.

        Nothing is written unless the complete package re-reads and validates; a
        failure leaves any existing file untouched.
        """
        self._live()
        target = Path(path) if path is not None else self._path
        if target is None:
            raise ValidationError("no path: this document was not opened from a file")
        if not allow_any_suffix and target.suffix.lower() != self.FILE_SUFFIX:
            raise ValidationError(f"{self.KIND_LABEL} files use the {self.FILE_SUFFIX} extension "
                                  f"(pass allow_any_suffix=True to override)", path=str(target))
        same_file = self._path is not None and _same_path(target, self._path)
        if overwrite is None:
            overwrite = same_file
        existed = target.exists()
        if existed:
            if not overwrite:
                raise ConflictError("output file already exists", path=str(target))
            if target.is_dir():
                raise ValidationError("output path is a directory", path=str(target))
            self._check_replace(target, same_file, expected_revision, expected_sha256, force)
        elif expected_revision is not None or expected_sha256 is not None:
            raise ConflictError("expected an existing file to replace but none exists", path=str(target))
        data, manifest, revision = self._build()
        file_sha = write_bytes_atomic(target, data, overwrite=bool(overwrite))
        self._path = target.absolute()
        self._source_sha256 = file_sha
        self._source_version = CURRENT_VERSION
        self._saved_revision = revision
        return SaveResult(str(self._path), file_sha, len(data), manifest["document_id"], revision, existed)

    def _check_replace(self, target: Path, same_file: bool, expected_revision, expected_sha256, force):
        if force:
            return
        current = _file_digest(target)
        if expected_sha256 is not None and current != expected_sha256:
            raise ConflictError("existing file digest differs from expected_sha256", path=str(target),
                                context={"expected": expected_sha256, "actual": current})
        if same_file and expected_sha256 is None and self._source_sha256 is not None \
                and current != self._source_sha256:
            raise ConflictError("file changed on disk since it was opened; reload or save with force=True",
                                path=str(target),
                                context={"opened_sha256": self._source_sha256, "actual": current})
        try:
            existing = inspect_bytes(Path(target).read_bytes(), verify="index", summary=False)
        except SpatialFormatError as exc:
            raise ConflictError("existing file is not a valid SLAI package; refusing to replace it "
                                "(pass force=True to override)", path=str(target)) from exc
        if existing.format_id != self.FORMAT_ID:
            raise ConflictError(f"existing file is a {existing.format_id} package; refusing to replace it",
                                path=str(target), context={"found": existing.format_id})
        if existing.document_id != self.document_id:
            raise ConflictError("existing file holds a different document; refusing to replace it "
                                "(pass force=True to override)", path=str(target),
                                context={"existing_document_id": existing.document_id,
                                         "document_id": self.document_id})
        if expected_revision is not None and existing.revision != expected_revision:
            raise ConflictError("existing file revision differs from expected_revision", path=str(target),
                                context={"expected": expected_revision, "actual": existing.revision})

    def __repr__(self) -> str:
        flags = ",".join(f for f, on in (("read-only", self._read_only), ("closed", self._closed),
                                         ("dirty", self.dirty)) if on)
        where = f" path={str(self._path)!r}" if self._path else ""
        return (f"<{type(self).__name__} {self.name!r} id={self.document_id[:8]} r{self.revision} "
                f"{self.unit}{where}{' [' + flags + ']' if flags else ''}>")


def _checked_metadata(values) -> dict:
    """Validate metadata and add a pointer to the accepted key syntax on failure."""
    if not isinstance(values, dict):
        raise ValidationError("metadata must be a mapping")
    try:
        return require_metadata(values)
    except ValidationError as exc:
        key = (exc.context or {}).get("key")
        if isinstance(key, str) and "." in key and ":" not in key and not key.startswith("slai."):
            namespace, _, name = key.partition(".")
            raise ValidationError(f"{exc.args[0]}: use '{namespace}:{name}' instead of '{key}'",
                                  context=exc.context) from exc
        raise


def _same_path(left: Path, right: Path) -> bool:
    try:
        return os.path.samefile(left, right)
    except OSError:
        return left.absolute() == right.absolute()


def _file_digest(path: Path) -> str:
    from .common.atomic import file_sha256
    return file_sha256(path)
