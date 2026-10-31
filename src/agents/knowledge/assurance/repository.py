"""Single SQLite canonical-store adapter for structured knowledge revisions.

No cloud service, ontology duplication, source-quality scoring, or in-memory
conversational persistence is introduced. The store uses immutable revisions,
SQLite transactions, optimistic concurrency, scope-first retrieval, and an
append-only relationship/impact history. An authenticated caller supplies a
TRUSTED AccessContext. SQL storage encryption is not provided by this module.
"""
from __future__ import annotations

import datetime as dt
import json
import sqlite3
import threading
import uuid

from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Iterator

from .acl import can_read, require_write
from .contracts import *
from ..utils.config_loader import get_config_section, load_global_config
from ..utils.knowledge_errors import *
from ..utils.knowledge_helpers import *
from logs.logger import PrettyPrinter, configure_logging, get_logger # pyright: ignore[reportMissingImports]


logger = get_logger("Assurance Repository")
printer = PrettyPrinter()


SCHEMA_VERSION = 1


class KnowledgeRepository:
    """Transactions and row-level scope filtering for authoritative knowledge."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with self._connect() as db:
            self._schema(db)

        logger.info(f"Knowledge Repository initialized")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(str(self.path), timeout=20, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA busy_timeout=20000")
        db.execute("PRAGMA journal_mode=WAL")
        try:
            yield db
        finally:
            db.close()

    def _schema(self, db: sqlite3.Connection) -> None:
        version = int(db.execute("PRAGMA user_version").fetchone()[0])
        if version not in (0, SCHEMA_VERSION):
            raise RuntimeError(f"Unsupported canonical knowledge schema {version}")
        with self._lock:
            db.execute("BEGIN IMMEDIATE")
            try:
                db.executescript("""
                CREATE TABLE IF NOT EXISTS knowledge_heads (
                    record_id TEXT PRIMARY KEY,
                    current_revision INTEGER NOT NULL CHECK (current_revision >= 1)
                );
                CREATE TABLE IF NOT EXISTS knowledge_revisions (
                    record_id TEXT NOT NULL,
                    revision INTEGER NOT NULL CHECK (revision >= 1),
                    kind TEXT NOT NULL,
                    scope TEXT NOT NULL CHECK (scope IN ('global','organization','project')),
                    project_id TEXT,
                    organization_id TEXT,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    source_ids_json TEXT NOT NULL,
                    evidence_json TEXT NOT NULL,
                    valid_from TEXT,
                    valid_until TEXT,
                    recorded_at TEXT NOT NULL,
                    actor_id TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    digest TEXT NOT NULL,
                    PRIMARY KEY (record_id, revision),
                    FOREIGN KEY (record_id) REFERENCES knowledge_heads(record_id)
                );
                CREATE INDEX IF NOT EXISTS idx_knowledge_scope ON knowledge_revisions(scope, project_id, organization_id, kind);
                CREATE INDEX IF NOT EXISTS idx_knowledge_time ON knowledge_revisions(record_id, recorded_at);
                CREATE TRIGGER IF NOT EXISTS no_update_revisions BEFORE UPDATE ON knowledge_revisions
                  BEGIN SELECT RAISE(ABORT, 'knowledge revisions are immutable'); END;
                CREATE TRIGGER IF NOT EXISTS no_delete_revisions BEFORE DELETE ON knowledge_revisions
                  BEGIN SELECT RAISE(ABORT, 'knowledge revisions are immutable'); END;
                CREATE TABLE IF NOT EXISTS relation_events (
                    event_id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    relation_type TEXT NOT NULL,
                    active INTEGER NOT NULL DEFAULT 1,
                    recorded_at TEXT NOT NULL,
                    actor_id TEXT NOT NULL,
                    FOREIGN KEY(source_id) REFERENCES knowledge_heads(record_id),
                    FOREIGN KEY(target_id) REFERENCES knowledge_heads(record_id)
                );
                CREATE INDEX IF NOT EXISTS idx_relation_target ON relation_events(target_id, relation_type);
                CREATE INDEX IF NOT EXISTS idx_relation_source ON relation_events(source_id, relation_type);
                CREATE TABLE IF NOT EXISTS impact_events (
                    impact_id TEXT PRIMARY KEY,
                    changed_record_id TEXT NOT NULL,
                    changed_revision INTEGER NOT NULL,
                    dependent_record_id TEXT NOT NULL,
                    recorded_at TEXT NOT NULL,
                    FOREIGN KEY(changed_record_id) REFERENCES knowledge_heads(record_id),
                    FOREIGN KEY(dependent_record_id) REFERENCES knowledge_heads(record_id)
                );
                CREATE INDEX IF NOT EXISTS idx_impacts_dependent ON impact_events(dependent_record_id, recorded_at);
                """)
                db.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
                db.commit()
            except Exception:
                db.rollback()
                raise

    @staticmethod
    def _decode(row: sqlite3.Row) -> RecordRevision:
        draft = RecordDraft(
            kind=row["kind"], scope=Scope(row["scope"]),
            project_id=row["project_id"], organization_id=row["organization_id"],
            payload=json.loads(row["payload_json"]), status=Lifecycle(row["status"]),
            source_ids=tuple(json.loads(row["source_ids_json"])),
            evidence=tuple(json.loads(row["evidence_json"])),
            valid_from=row["valid_from"], valid_until=row["valid_until"],
        )
        return RecordRevision(record_id=row["record_id"], revision=row["revision"],
                              draft=draft, recorded_at=row["recorded_at"],
                              actor_id=row["actor_id"], reason=row["reason"], digest=row["digest"])

    @staticmethod
    def _values(draft: RecordDraft, record_id: str, rev: int,
                recorded_at: str, actor: str, reason: str) -> tuple[Any, ...]:
        payload = json_object(draft.payload, "payload")
        evidence = [json_object(e, "evidence entry", 16384) for e in draft.evidence]
        content = {
            "record_id": record_id, "revision": rev, "kind": draft.kind,
            "scope": draft.scope.value, "project_id": draft.project_id,
            "organization_id": draft.organization_id, "status": draft.status.value,
            "payload": payload, "source_ids": list(draft.source_ids), "evidence": evidence,
            "valid_from": normalize_time(draft.valid_from, "valid_from"),
            "valid_until": normalize_time(draft.valid_until, "valid_until"),
            "recorded_at": recorded_at, "actor_id": actor, "reason": reason,
        }
        return (record_id, rev, draft.kind, draft.scope.value, draft.project_id,
                draft.organization_id, draft.status.value, canonical_json(payload),
                canonical_json(list(draft.source_ids)), canonical_json(evidence),
                content["valid_from"], content["valid_until"], recorded_at,
                actor, reason, digest_json(content))

    @staticmethod
    def _insert_revision(db: sqlite3.Connection, values: tuple[Any, ...]) -> None:
        db.execute("""INSERT INTO knowledge_revisions
            (record_id, revision, kind, scope, project_id, organization_id,
             status, payload_json, source_ids_json, evidence_json,
             valid_from, valid_until, recorded_at, actor_id, reason, digest)
             VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", values)

    def create(self, draft: RecordDraft, access: AccessContext,
               *, record_id: str | None = None, reason: str = "created") -> RecordRevision:
        require_write(draft, access)
        record_id = validate_id(record_id or f"kn:{uuid.uuid4().hex}", "record_id")
        if not isinstance(reason, str) or not reason.strip():
            raise ValidationFailure("reason is required")
        with self._lock, self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                db.execute("INSERT INTO knowledge_heads VALUES (?,1)", (record_id,))
                self._insert_revision(db, self._values(draft, record_id, 1, utc_now(), access.principal_id, reason))
                db.commit()
            except sqlite3.IntegrityError as exc:
                db.rollback()
                raise RevisionConflict("record_id already exists", expected_revision=0, actual_revision=1) from exc
            except Exception:
                db.rollback()
                raise
        record = self.get(record_id, access)
        assert record is not None
        return record

    @staticmethod
    def _latest_row(db: sqlite3.Connection, record_id: str,
                    as_known: str | None = None) -> sqlite3.Row | None:
        if as_known:
            return db.execute("""SELECT * FROM knowledge_revisions WHERE record_id=?
                AND recorded_at<=? ORDER BY revision DESC LIMIT 1""", (record_id, as_known)).fetchone()
        return db.execute("""SELECT * FROM knowledge_revisions WHERE record_id=?
            ORDER BY revision DESC LIMIT 1""", (record_id,)).fetchone()

    def revise(self, record_id: str, expected_revision: int, draft: RecordDraft,
               access: AccessContext, *, reason: str) -> RecordRevision:
        validate_id(record_id, "record_id")
        require_write(draft, access)
        if not reason or not reason.strip():
            raise ValidationFailure("revision reason is required")
        with self._lock, self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                old_row = self._latest_row(db, record_id)
                if old_row is None:
                    raise NotFound("Knowledge record not found", entity_id=record_id)
                old = self._decode(old_row)
                # Editing an approved record back into a draft may be performed by
                # its authorized owner; entering reviewed/active still requires review.
                require_write(replace(old.draft, status=Lifecycle.DRAFT), access)
                if old.revision != expected_revision:
                    raise RevisionConflict(f"Expected revision {expected_revision}, found {old.revision}",
                                           expected_revision=expected_revision,
                                           actual_revision=old.revision)
                if (old.draft.kind != draft.kind or old.draft.scope != draft.scope
                        or old.draft.project_id != draft.project_id
                        or old.draft.organization_id != draft.organization_id):
                    raise ValidationFailure("Record identity, kind and owner scope are immutable")
                current_time = utc_now()
                if current_time <= old.recorded_at:
                    previous = dt.datetime.fromisoformat(old.recorded_at.replace("Z", "+00:00"))
                    current_time = (previous + dt.timedelta(microseconds=1)).isoformat(timespec="microseconds").replace("+00:00", "Z")
                new_revision = old.revision + 1
                self._insert_revision(db, self._values(draft, record_id, new_revision, current_time, access.principal_id, reason))
                db.execute("UPDATE knowledge_heads SET current_revision=? WHERE record_id=?", (new_revision, record_id))
                for dependent in self._dependent_ids(db, record_id):
                    db.execute("INSERT INTO impact_events VALUES (?,?,?,?,?)",
                               (uuid.uuid4().hex, record_id, new_revision, dependent, current_time))
                db.commit()
            except Exception:
                db.rollback()
                raise
        record = self.get(record_id, access)
        assert record is not None
        return record

    def get(self, record_id: str, access: AccessContext | None = None,
            *, as_known: str | None = None, as_valid: str | None = None,
            include_inactive: bool = True) -> RecordRevision | None:
        validate_id(record_id, "record_id")
        as_known = normalize_time(as_known, "as_known")
        as_valid = normalize_time(as_valid, "as_valid")
        with self._connect() as db:
            row = self._latest_row(db, record_id, as_known)
        if row is None:
            return None
        revision = self._decode(row)
        if not can_read(revision.draft, access):
            return None  # No authorization oracle.
        if not include_inactive and revision.draft.status in {Lifecycle.INVALIDATED, Lifecycle.SUPERSEDED, Lifecycle.ARCHIVED}:
            return None
        if as_valid and ((revision.draft.valid_from and revision.draft.valid_from > as_valid)
                         or (revision.draft.valid_until and revision.draft.valid_until <= as_valid)):
            return None
        return revision

    def query(self, access: AccessContext | None = None, *, kind: str | None = None,
              as_known: str | None = None, as_valid: str | None = None,
              text: str | None = None, project_id: str | None = None,
              include_inactive: bool = False,
              limit: int = 100, offset: int = 0) -> list[RecordRevision]:
        """Scope filtering occurs in SQL BEFORE contents are considered/ranked."""
        if not isinstance(limit, int) or not 1 <= limit <= 1000 or offset < 0:
            raise ValidationFailure("limit must be 1..1000 and offset nonnegative")
        if kind and not isinstance(kind, str):
            raise ValidationFailure("kind must be a string")
        as_known = normalize_time(as_known, "as_known")
        as_valid = normalize_time(as_valid, "as_valid")
        filters: list[str] = ["r.scope='global'"]
        args: list[Any] = []
        if access is not None:
            for scope, column, ids in (("project", "project_id", access.project_ids),
                                        ("organization", "organization_id", access.organization_ids)):
                if ids:
                    placeholders = ",".join("?" for _ in ids)
                    filters.append(f"(r.scope='{scope}' AND r.{column} IN ({placeholders}))")
                    args.extend(sorted(ids))
        clause = " OR ".join(filters)
        sql = f"""SELECT r.* FROM knowledge_revisions r
          WHERE r.revision=(SELECT MAX(x.revision) FROM knowledge_revisions x
                               WHERE x.record_id=r.record_id {"AND x.recorded_at<=?" if as_known else ""})
            AND ({clause})"""
        bindings: list[Any] = ([as_known] if as_known else []) + args
        if kind:
            sql += " AND r.kind=?"
            bindings.append(kind)
        if project_id is not None:
            validate_id(project_id, "project_id")
            sql += " AND r.scope='project' AND r.project_id=?"
            bindings.append(project_id)
        if not include_inactive:
            sql += " AND r.status NOT IN ('invalidated','superseded','archived')"
        if as_valid:
            sql += " AND (r.valid_from IS NULL OR r.valid_from<=?) AND (r.valid_until IS NULL OR r.valid_until>?)"
            bindings.extend((as_valid, as_valid))
        if text:
            # Not an FTS search. Project filtering precedes matching; no embedding data leakage.
            escaped = text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            sql += " AND r.payload_json LIKE ? ESCAPE '\\'"
            bindings.append("%" + escaped + "%")
        sql += " ORDER BY r.recorded_at DESC, r.record_id LIMIT ? OFFSET ?"
        bindings.extend((limit, offset))
        with self._connect() as db:
            rows = db.execute(sql, bindings).fetchall()
        return [rev for row in rows if can_read((rev := self._decode(row)).draft, access)]

    def _dependent_ids(self, db: sqlite3.Connection, record_id: str) -> list[str]:
        """Find downstream dependency closure using only active relation events."""
        edges = db.execute("""SELECT source_id,target_id FROM relation_events e
          WHERE e.relation_type='depends_on' AND e.active=1
            AND e.rowid=(SELECT MAX(rowid) FROM relation_events x WHERE x.source_id=e.source_id
                AND x.target_id=e.target_id AND x.relation_type=e.relation_type)""").fetchall()
        by_target: dict[str, set[str]] = {}
        for row in edges:
            by_target.setdefault(row["target_id"], set()).add(row["source_id"])
        found: set[str] = set()
        waiting = [record_id]
        while waiting:
            parent = waiting.pop()
            for child in by_target.get(parent, ()):
                if child != record_id and child not in found:
                    found.add(child)
                    waiting.append(child)
        return sorted(found)

    def link(self, dependent_id: str, dependency_id: str,
             access: AccessContext, *, relation_type: str = "depends_on") -> str:
        """Typed links are immutable events. Project -> global is permitted, reverse forbidden."""
        validate_id(dependent_id, "dependent_id")
        validate_id(dependency_id, "dependency_id")
        if dependent_id == dependency_id:
            raise ValidationFailure("Self dependency prohibited")
        if relation_type not in {"depends_on", "supported_by", "refers_to", "supersedes", "implements"}:
            raise ValidationFailure("Unknown relationship type")
        with self._lock, self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                dep_row = self._latest_row(db, dependent_id)
                base_row = self._latest_row(db, dependency_id)
                if dep_row is None or base_row is None:
                    missing_id = dependent_id if dep_row is None else dependency_id
                    raise NotFound("One or more relationship endpoints do not exist", entity_id=missing_id)
                dependent, dependency = self._decode(dep_row), self._decode(base_row)
                require_write(dependent.draft, access)
                if not can_read(dependency.draft, access):
                    raise AuthorizationDenied("Dependency is not accessible", action="read")
                # A globally available record may not acquire a private dependency.
                if dependent.draft.scope == Scope.GLOBAL and dependency.draft.scope != Scope.GLOBAL:
                    raise ValidationFailure("Global knowledge cannot depend on private scope")
                if dependent.draft.scope == Scope.ORGANIZATION and dependency.draft.scope == Scope.PROJECT:
                    raise ValidationFailure("Organization knowledge cannot depend on project scope")
                if dependent.draft.scope == Scope.PROJECT and dependency.draft.scope == Scope.PROJECT and dependent.draft.project_id != dependency.draft.project_id:
                    raise ValidationFailure("Cross-project dependency prohibited")
                if relation_type == "depends_on" and dependency_id in self._dependent_ids(db, dependent_id):
                    # Existing downstream closure: adding this edge would close a cycle.
                    raise ValidationFailure("Dependency cycle prohibited")
                event_id = uuid.uuid4().hex
                db.execute("INSERT INTO relation_events VALUES (?,?,?,?,?,?,?)",
                           (event_id, dependent_id, dependency_id, relation_type, 1, utc_now(), access.principal_id))
                db.commit()
                return event_id
            except Exception:
                db.rollback()
                raise

    def impacts(self, changed_record_id: str, access: AccessContext) -> list[dict[str, Any]]:
        """Returns pending downstream records; never exposes hidden project IDs."""
        validate_id(changed_record_id, "changed_record_id")
        changed = self.get(changed_record_id, access)
        if changed is None:
            return []
        with self._connect() as db:
            candidate_ids = self._dependent_ids(db, changed_record_id)
            events = db.execute("SELECT * FROM impact_events WHERE changed_record_id=? ORDER BY recorded_at",
                                (changed_record_id,)).fetchall()
        result = []
        for candidate_id in candidate_ids:
            rec = self.get(candidate_id, access)
            if rec is None:
                continue
            pending = [e for e in events if e["dependent_record_id"] == candidate_id and rec.recorded_at < e["recorded_at"]]
            if pending:
                result.append({"record_id": candidate_id, "revision": rec.revision,
                               "latest_trigger_revision": max(e["changed_revision"] for e in pending),
                               "status": "review_required"})
        return result

    def close(self) -> None:
        """SQLite connections are per-operation, so there is no persistent handle to close."""


__all__ = [
    "KnowledgeRepository"
]