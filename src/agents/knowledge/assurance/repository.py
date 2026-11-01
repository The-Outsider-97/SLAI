"""Single SQLite canonical-store adapter for structured knowledge revisions.

No cloud service, ontology duplication, source-quality scoring, or in-memory
conversational persistence is introduced. The store uses immutable revisions,
SQLite transactions, optimistic concurrency, scope-first retrieval, and an
append-only relationship/impact history. An authenticated caller supplies a
TRUSTED AccessContext. SQL storage encryption is not provided by this module.
"""
from __future__ import annotations
import os
from contextlib import nullcontext
from dataclasses import dataclass

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
from src.agents.knowledge.utils.config_loader import get_config_section, load_global_config
from src.agents.knowledge.utils.knowledge_errors import *
from src.agents.knowledge.utils.knowledge_helpers import *
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
                ensure_extended_schema(db)
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
                downstream = self._dependent_ids(db, record_id)
                for dependent in downstream:
                    db.execute("INSERT INTO impact_events VALUES (?,?,?,?,?)",
                               (uuid.uuid4().hex, record_id, new_revision, dependent, current_time))
                record_revision_effects(db, record_id, new_revision, downstream, current_time)
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
              limit: int = 100, offset: int = 0,
              scope_context_project: str | None = None,
              _connection: sqlite3.Connection | None = None) -> list[RecordRevision]:
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
        if scope_context_project is not None:
            if scope_context_project:
                validate_id(scope_context_project, "scope_context_project")
                if access is None or scope_context_project not in access.project_ids:
                    raise AuthorizationDenied(access.principal_id if access else "anonymous", "query_project")
                sql += " AND (r.scope!='project' OR r.project_id=?)"
                bindings.append(scope_context_project)
            else:
                sql += " AND r.scope!='project'"
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
        with (nullcontext(_connection) if _connection is not None else self._connect()) as db:
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
                recorded_at = utc_now()
                db.execute("INSERT INTO relation_events VALUES (?,?,?,?,?,?,?)",
                           (event_id, dependent_id, dependency_id, relation_type, 1, recorded_at, access.principal_id))
                record_relation_effects(db, event_id, dependent_id, recorded_at)
                db.commit()
                return event_id
            except Exception:
                db.rollback()
                raise

    def unlink(self, dependent_id: str, dependency_id: str, access: AccessContext,
               *, relation_type: str = "depends_on") -> str:
        """Append a deactivation event rather than deleting historical relation data."""
        from .contracts import AuthorizationDenied
        validate_id(dependent_id, "dependent_id")
        validate_id(dependency_id, "dependency_id")
        if relation_type not in {"depends_on", "supported_by", "refers_to", "supersedes", "implements"}:
            raise ValidationFailure("Unknown relationship type")
        with self._lock, self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                dep_row = self._latest_row(db, dependent_id)
                base_row = self._latest_row(db, dependency_id)
                if dep_row is None or base_row is None:
                    raise NotFound("knowledge_relation", dependent_id)
                require_write(self._decode(dep_row).draft, access)
                if not can_read(self._decode(base_row).draft, access):
                    raise AuthorizationDenied(access.principal_id, "read", resource=dependency_id)
                state = db.execute("""SELECT active FROM relation_events
                    WHERE source_id=? AND target_id=? AND relation_type=?
                    ORDER BY rowid DESC LIMIT 1""",
                    (dependent_id, dependency_id, relation_type)).fetchone()
                if state is None or not state["active"]:
                    raise NotFound("knowledge_relation", dependent_id)
                event_id = uuid.uuid4().hex
                recorded_at = utc_now()
                db.execute("INSERT INTO relation_events VALUES(?,?,?,?,?,?,?)",
                           (event_id, dependent_id, dependency_id, relation_type,
                            0, recorded_at, access.principal_id))
                record_relation_effects(db, event_id, dependent_id, recorded_at)
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

def ensure_extended_schema(db: sqlite3.Connection) -> None:
    db.executescript("""
    CREATE TABLE IF NOT EXISTS assessment_snapshots (
        assessment_id TEXT PRIMARY KEY,
        project_id TEXT NOT NULL,
        object_record_id TEXT NOT NULL,
        object_revision INTEGER NOT NULL CHECK(object_revision>0),
        rule_record_id TEXT NOT NULL,
        rule_revision INTEGER NOT NULL CHECK(rule_revision>0),
        outcome TEXT NOT NULL,
        evaluator TEXT NOT NULL,
        input_digest TEXT NOT NULL,
        context_digest TEXT NOT NULL,
        valid_at TEXT NOT NULL,
        evaluated_at TEXT NOT NULL,
        recorded_at TEXT NOT NULL,
        actor_id TEXT NOT NULL,
        evidence_refs_json TEXT NOT NULL,
        assessment_digest TEXT NOT NULL,
        FOREIGN KEY(object_record_id) REFERENCES knowledge_heads(record_id),
        FOREIGN KEY(rule_record_id) REFERENCES knowledge_heads(record_id)
    );
    CREATE INDEX IF NOT EXISTS idx_assessments_project ON assessment_snapshots(project_id, recorded_at);
    CREATE TABLE IF NOT EXISTS assessment_dependencies (
        assessment_id TEXT NOT NULL,
        record_id TEXT NOT NULL,
        pinned_revision INTEGER NOT NULL CHECK(pinned_revision>0),
        PRIMARY KEY(assessment_id,record_id),
        FOREIGN KEY(assessment_id) REFERENCES assessment_snapshots(assessment_id),
        FOREIGN KEY(record_id) REFERENCES knowledge_heads(record_id)
    );
    CREATE INDEX IF NOT EXISTS idx_assessment_dependencies_record ON assessment_dependencies(record_id);
    CREATE TABLE IF NOT EXISTS assessment_stale_events (
        assessment_id TEXT NOT NULL,
        changed_record_id TEXT NOT NULL,
        changed_revision INTEGER NOT NULL,
        reason TEXT NOT NULL,
        recorded_at TEXT NOT NULL,
        PRIMARY KEY(assessment_id,changed_record_id,changed_revision,reason),
        FOREIGN KEY(assessment_id) REFERENCES assessment_snapshots(assessment_id)
    );
    CREATE TABLE IF NOT EXISTS knowledge_outbox (
        event_id TEXT PRIMARY KEY,
        event_type TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        attempts INTEGER NOT NULL DEFAULT 0,
        next_retry_at TEXT NOT NULL,
        leased_until TEXT,
        lease_token TEXT,
        delivered_at TEXT,
        dead_letter_at TEXT,
        last_error_code TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_knowledge_outbox_pending
        ON knowledge_outbox(delivered_at,dead_letter_at,next_retry_at,leased_until);
    CREATE TABLE IF NOT EXISTS promotion_audit (
        public_record_id TEXT PRIMARY KEY,
        origin_record_id TEXT NOT NULL,
        origin_revision INTEGER NOT NULL,
        reviewer_id TEXT NOT NULL,
        privacy_evidence_id TEXT NOT NULL,
        licence_evidence_id TEXT NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY(public_record_id) REFERENCES knowledge_heads(record_id),
        FOREIGN KEY(origin_record_id) REFERENCES knowledge_heads(record_id)
    );
    CREATE TRIGGER IF NOT EXISTS immutable_assessment_update BEFORE UPDATE ON assessment_snapshots
        BEGIN SELECT RAISE(ABORT,'assessment snapshot is immutable'); END;
    CREATE TRIGGER IF NOT EXISTS immutable_assessment_delete BEFORE DELETE ON assessment_snapshots
        BEGIN SELECT RAISE(ABORT,'assessment snapshot is immutable'); END;
    CREATE TRIGGER IF NOT EXISTS immutable_assessment_dep_update BEFORE UPDATE ON assessment_dependencies
        BEGIN SELECT RAISE(ABORT,'assessment dependency is immutable'); END;
    CREATE TRIGGER IF NOT EXISTS immutable_assessment_dep_delete BEFORE DELETE ON assessment_dependencies
        BEGIN SELECT RAISE(ABORT,'assessment dependency is immutable'); END;
    CREATE TRIGGER IF NOT EXISTS immutable_stale_update BEFORE UPDATE ON assessment_stale_events
        BEGIN SELECT RAISE(ABORT,'stale event is immutable'); END;
    CREATE TRIGGER IF NOT EXISTS immutable_stale_delete BEFORE DELETE ON assessment_stale_events
        BEGIN SELECT RAISE(ABORT,'stale event is immutable'); END;
    CREATE TRIGGER IF NOT EXISTS immutable_promotion_update BEFORE UPDATE ON promotion_audit
        BEGIN SELECT RAISE(ABORT,'promotion event is immutable'); END;
    CREATE TRIGGER IF NOT EXISTS immutable_promotion_delete BEFORE DELETE ON promotion_audit
        BEGIN SELECT RAISE(ABORT,'promotion event is immutable'); END;
    """)


def _enqueue(db: sqlite3.Connection, event_type: str, key: str, payload: dict, now: str) -> None:
    event_id = digest_json({"namespace": "slai.assurance.v1", "event_type": event_type, "key": key})
    db.execute("""INSERT OR IGNORE INTO knowledge_outbox
        (event_id,event_type,payload_json,created_at,next_retry_at)
        VALUES(?,?,?,?,?)""", (event_id, event_type,
        json.dumps(payload, sort_keys=True, separators=(",", ":")), now, now))


def record_revision_effects(db: sqlite3.Connection, record_id: str, revision: int,
                            downstream: Iterable[str], now: str) -> None:
    """Invoked INSIDE KnowledgeRepository.revise's BEGIN IMMEDIATE transaction."""
    affected = sorted(set(downstream) | {record_id})
    for dependency in affected:
        db.execute("""INSERT OR IGNORE INTO assessment_stale_events
            (assessment_id,changed_record_id,changed_revision,reason,recorded_at)
            SELECT d.assessment_id,?,?,?,?
              FROM assessment_dependencies d
              JOIN assessment_snapshots s ON s.assessment_id=d.assessment_id
             WHERE d.record_id=? AND s.recorded_at<=?""",
             (record_id, revision, "knowledge_revision", now, dependency, now))
    _enqueue(db, "knowledge.revised", f"{record_id}:{revision}",
             {"record_id": record_id, "revision": revision,
              "affected_dependency_count": len(affected)}, now)


def record_relation_effects(db: sqlite3.Connection, event_id: str,
                            source_record_id: str, now: str) -> None:
    # Relation changes may alter an assessment's derivation even without a
    # document revision. `-1` is a typed graph event marker, not a revision.
    db.execute("""INSERT OR IGNORE INTO assessment_stale_events
         (assessment_id,changed_record_id,changed_revision,reason,recorded_at)
         SELECT d.assessment_id,?,-1,?,?
           FROM assessment_dependencies d
           JOIN assessment_snapshots s ON s.assessment_id=d.assessment_id
          WHERE d.record_id=? AND s.recorded_at<=?""",
        (source_record_id, f"relation_changed:{event_id}", now, source_record_id, now))
    _enqueue(db, "knowledge.relation_changed", event_id,
             {"source_record_id": source_record_id, "relation_event_id": event_id}, now)

@dataclass(frozen=True, slots=True)
class OutboxDelivery:
    event_id: str
    event_type: str
    payload: dict[str, Any]
    attempts: int
    lease_token: str


def _plus_seconds(now: str, seconds: int) -> str:
    value = dt.datetime.fromisoformat(now.replace("Z", "+00:00"))
    return (value + dt.timedelta(seconds=seconds)).isoformat(timespec="microseconds").replace("+00:00", "Z")


class KnowledgeOutbox:
    """Internal-use only; do not expose lease/ack endpoints to client apps."""

    def __init__(self, repository: KnowledgeRepository, *, max_attempts: int = 7):
        if not 1 <= max_attempts <= 100:
            raise ValidationFailure("max_attempts must be 1..100")
        self.repository = repository
        self.max_attempts = max_attempts

    def lease(self, *, batch_size: int = 25, lease_seconds: int = 60,
              now: str | None = None) -> list[OutboxDelivery]:
        if not isinstance(batch_size, int) or not 1 <= batch_size <= 100:
            raise ValidationFailure("batch size must be 1..100")
        if not isinstance(lease_seconds, int) or not 1 <= lease_seconds <= 3600:
            raise ValidationFailure("lease duration must be 1..3600")
        now = normalize_time(now or utc_now(), "now")
        leased_until = _plus_seconds(now, lease_seconds)
        with self.repository._lock, self.repository._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                rows = db.execute("""SELECT * FROM knowledge_outbox
                    WHERE delivered_at IS NULL AND dead_letter_at IS NULL
                      AND next_retry_at<=? AND (leased_until IS NULL OR leased_until<=?)
                      AND attempts<?
                    ORDER BY created_at,event_id LIMIT ?""",
                    (now, now, self.max_attempts, batch_size)).fetchall()
                result = []
                for row in rows:
                    token = uuid.uuid4().hex
                    db.execute("""UPDATE knowledge_outbox SET leased_until=?,lease_token=?,attempts=attempts+1
                        WHERE event_id=?""", (leased_until, token, row["event_id"]))
                    result.append(OutboxDelivery(row["event_id"], row["event_type"],
                                                 json.loads(row["payload_json"]),
                                                 row["attempts"] + 1, token))
                db.commit()
            except Exception:
                db.rollback()
                raise
        return result

    def ack(self, event_id: str, lease_token: str, *, now: str | None = None) -> bool:
        validate_id(event_id, "event_id")
        validate_id(lease_token, "lease_token")
        now = normalize_time(now or utc_now(), "now")
        with self.repository._lock, self.repository._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            result = db.execute("""UPDATE knowledge_outbox
                SET delivered_at=?,leased_until=NULL,lease_token=NULL
                WHERE event_id=? AND lease_token=? AND delivered_at IS NULL
                  AND dead_letter_at IS NULL AND leased_until>?""",
                (now, event_id, lease_token, now))
            db.commit()
        return result.rowcount == 1

    def fail(self, event_id: str, lease_token: str, *,
             error_code: str = "delivery_failed", now: str | None = None) -> bool:
        validate_id(event_id, "event_id")
        validate_id(lease_token, "lease_token")
        validate_id(error_code, "error_code")
        now = normalize_time(now or utc_now(), "now")
        with self.repository._lock, self.repository._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("""SELECT attempts FROM knowledge_outbox
                WHERE event_id=? AND lease_token=? AND delivered_at IS NULL
                  AND dead_letter_at IS NULL AND leased_until>?""",
                (event_id, lease_token, now)).fetchone()
            if row is None:
                db.rollback()
                return False
            attempts = row["attempts"]
            next_attempt = _plus_seconds(now, min(3600, 2 ** min(attempts, 11)))
            db.execute("""UPDATE knowledge_outbox SET lease_token=NULL,leased_until=NULL,
                next_retry_at=?,dead_letter_at=?,last_error_code=? WHERE event_id=?""",
                (next_attempt, now if attempts >= self.max_attempts else None,
                 error_code, event_id))
            db.commit()
            return True

    def status(self) -> dict[str, int]:
        with self.repository._connect() as db:
            row = db.execute("""SELECT
                SUM(CASE WHEN delivered_at IS NOT NULL THEN 1 ELSE 0 END) delivered,
                SUM(CASE WHEN delivered_at IS NULL AND dead_letter_at IS NULL THEN 1 ELSE 0 END) pending,
                SUM(CASE WHEN dead_letter_at IS NOT NULL THEN 1 ELSE 0 END) dead_letter
                FROM knowledge_outbox""").fetchone()
        return {name: row[name] or 0 for name in ("delivered", "pending", "dead_letter")}

class KnowledgeBackupService:
    def __init__(self, repository: KnowledgeRepository):
        self.repository = repository

    @staticmethod
    def validate(path: str | Path) -> bool:
        path = Path(path)
        if not path.is_file():
            raise ValidationFailure("Backup file not found")
        uri = f"file:{path.resolve().as_posix()}?mode=ro"
        with sqlite3.connect(uri, uri=True, timeout=10) as db:
            integrity = db.execute("PRAGMA integrity_check").fetchone()[0]
            head_count = db.execute("SELECT COUNT(*) FROM knowledge_heads").fetchone()[0]
            db.execute("SELECT COUNT(*) FROM assessment_snapshots").fetchone()
        if integrity != "ok" or head_count < 0:
            raise ValidationFailure("Backup integrity validation failed")
        return True

    def create(self, destination: str | Path, *, overwrite: bool = False) -> Path:
        destination = Path(destination).expanduser().resolve()
        if destination == self.repository.path:
            raise ValidationFailure("Backup cannot overwrite active database")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and not overwrite:
            raise ValidationFailure("Backup destination exists; refusing overwrite")
        temp = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
        try:
            with sqlite3.connect(str(self.repository.path)) as source:
                with sqlite3.connect(str(temp)) as target:
                    source.backup(target)
            if os.name == "posix":
                os.chmod(temp, 0o600)
            self.validate(temp)
            os.replace(temp, destination)
            return destination
        finally:
            if temp.exists():
                temp.unlink()
