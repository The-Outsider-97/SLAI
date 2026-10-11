"""SQLite WAL persistence for SLAI QualityMemory v2.3.

Records are stored once per immutable quality record ID. Historical JSON and
JSONL source files are never modified by this module. No automatic migration.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import zlib

from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional
from logs.logger import PrettyPrinter, get_logger # pyright: ignore[reportMissingImports]

logger = get_logger("Quality SQLite WAL")
printer = PrettyPrinter()

KINDS = {
    "quality_snapshot_recorded": ("snapshot_id", "source_id", None, "created_at"),
    "schema_version_registered": ("version_id", "source_id", None, "recorded_at"),
    "drift_baseline_recorded": ("baseline_id", "source_id", "metric", "recorded_at"),
    "drift_observation_recorded": ("observation_id", "source_id", "metric", "recorded_at"),
    "threshold_decision_recorded": ("decision_id", "rule_id", None, "recorded_at"),
    "remediation_outcome_recorded": ("outcome_id", "rule_id", None, "recorded_at"),
    "conflict_reconciled": ("conflict_id", "scope_key", None, "recorded_at"),
    "source_reliability_recorded": ("event_id", "source_id", None, "recorded_at"),
}

LEGACY_SECTIONS = (
    ("snapshots_by_source", "quality_snapshot_recorded", True),
    ("snapshots_by_batch", "quality_snapshot_recorded", False),
    ("schema_versions", "schema_version_registered", True),
    ("drift_baselines", "drift_baseline_recorded", True),
    ("drift_observations", "drift_observation_recorded", True),
    ("threshold_decisions", "threshold_decision_recorded", True),
    ("remediation_history", "remediation_outcome_recorded", True),
    ("conflict_history", "conflict_reconciled", True),
    ("source_reliability_history", "source_reliability_recorded", True),
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS records (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    record_id TEXT NOT NULL UNIQUE,
    kind TEXT NOT NULL,
    scope TEXT NOT NULL,
    subscope TEXT NOT NULL DEFAULT '',
    batch_id TEXT,
    happened_at REAL NOT NULL,
    compressed_payload BLOB NOT NULL,
    payload_sha256 TEXT NOT NULL,
    archived INTEGER NOT NULL DEFAULT 0 CHECK(archived IN (0, 1))
);
CREATE INDEX IF NOT EXISTS idx_quality_scope
    ON records(kind, scope, subscope, happened_at DESC, seq DESC);
CREATE INDEX IF NOT EXISTS idx_quality_batch
    ON records(kind, batch_id, happened_at DESC, seq DESC);
CREATE INDEX IF NOT EXISTS idx_quality_archive
    ON records(kind, archived, happened_at DESC, seq DESC);
"""


class SqliteQualityStore:
    def __init__(self, path: Path, *, create: bool = False, wal: bool = True,
                 timeout: float = 30.0) -> None:
        self.path = Path(path)
        if not create and not self.path.is_file():
            raise FileNotFoundError(
                f"SQLite Quality Memory missing: {self.path}. "
                "Run quality_memory_migrate.py --apply before enabling backend: sqlite."
            )
        if create:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(self.path), timeout=timeout,
                                          isolation_level=None, check_same_thread=False)
        self.connection.execute("PRAGMA busy_timeout = 30000")
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA synchronous = FULL")
        if wal:
            mode = self.connection.execute("PRAGMA journal_mode = WAL").fetchone()[0]
            if mode.lower() != "wal":
                raise RuntimeError(f"WAL unavailable: {mode}")
        if create:
            self.connection.executescript(SCHEMA)
        else:
            row = self.connection.execute(
                "SELECT value FROM metadata WHERE key='migration_complete'"
            ).fetchone()
            if row is None or row[0] != "1":
                self.connection.close()
                raise RuntimeError("Quality Memory SQLite database has not passed migration")

        logger.info(f"Sqlite Quality Store initialized")

    def close(self) -> None:
        self.connection.close()

    def set_meta(self, key: str, value: str) -> None:
        printer.status("SET META", f"Setting meta for the Sqlite Quality Store with: \n{key} \n{value}" "info")
        self.connection.execute(
            "INSERT INTO metadata(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value)
        )

    def get_meta(self, key: str) -> Optional[str]:
        printer.status("GET META", f"Getting meta from the Sqlite Quality Store with: \n{key} " "info")
        result = self.connection.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
        return None if result is None else str(result[0])

    @staticmethod
    def _pack(record: Mapping[str, Any]) -> tuple[bytes, str]:
        raw = json.dumps(record, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
        return zlib.compress(raw, level=6), hashlib.sha256(raw).hexdigest()

    @staticmethod
    def _unpack(blob: bytes) -> Dict[str, Any]:
        value = json.loads(zlib.decompress(blob))
        if not isinstance(value, dict):
            raise ValueError("Quality record must be a JSON object")
        return value

    def insert(self, kind: str, record: Mapping[str, Any],
               *, allow_duplicate: bool = False) -> bool:
        if kind not in KINDS:
            raise ValueError(f"Unsupported event type: {kind}")
        id_key, scope_key, sub_key, time_key = KINDS[kind]
        record_id = str(record[id_key])
        scope = str(record[scope_key])
        subscope = str(record[sub_key]) if sub_key else ""
        happened = float(record[time_key])
        batch = record.get("batch_id")
        blob, fingerprint = self._pack(record)
        existing = self.connection.execute(
            "SELECT kind,scope,subscope,batch_id,happened_at,payload_sha256,compressed_payload "
            "FROM records WHERE record_id=?", (record_id,)
        ).fetchone()
        if existing:
            # Legacy JSON can mutate is_active on older schema entries; _apply_event
            # computes active status from registration ordering instead.
            if existing[0] != kind or existing[1] != scope or existing[2] != subscope:
                raise ValueError(f"Conflicting duplicate record ID: {record_id}")
            if existing[5] != fingerprint:
                prior = self._unpack(existing[6])
                now = dict(record)
                if kind == "schema_version_registered":
                    prior.pop("is_active", None)
                    now.pop("is_active", None)
                if prior != now:
                    raise ValueError(f"Conflicting duplicate record payload: {record_id}")
            if not allow_duplicate:
                raise ValueError(f"Duplicate live record: {record_id}")
            return False
        self.connection.execute(
            "INSERT INTO records(record_id,kind,scope,subscope,batch_id,happened_at,"
            "compressed_payload,payload_sha256) VALUES(?,?,?,?,?,?,?,?)",
            (record_id, kind, scope, subscope, None if batch is None else str(batch),
             happened, blob, fingerprint)
        )
        return True

    def append(self, kind: str, record: Mapping[str, Any],
               *, hot_batch_limit: int) -> None:
        # Entire mutation and archive status changes are one durable transaction.
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            self.insert(kind, record)
            if kind == "quality_snapshot_recorded":
                self.archive_old_batches(hot_batch_limit)
            self.connection.execute("COMMIT")
        except BaseException:
            self.connection.execute("ROLLBACK")
            raise

    def archive_old_batches(self, hot_batch_limit: int) -> None:
        if hot_batch_limit < 1:
            raise ValueError("max_snapshots_by_batch must be >= 1")
        # Keep the newest N snapshot records hot; old snapshots remain queryable.
        cutoff = self.connection.execute(
            "SELECT happened_at,seq FROM records WHERE kind=? "
            "ORDER BY happened_at DESC,seq DESC LIMIT 1 OFFSET ?",
            ("quality_snapshot_recorded", hot_batch_limit - 1)
        ).fetchone()
        if cutoff is None:
            return
        self.connection.execute(
            "UPDATE records SET archived=1 WHERE kind=? AND archived=0 "
            "AND (happened_at < ? OR (happened_at=? AND seq < ?))",
            ("quality_snapshot_recorded", cutoff[0], cutoff[0], cutoff[1])
        )

    def history(self, kind: str, scope: str, subscope: str = "",
                limit: Optional[int] = None) -> list[Dict[str, Any]]:
        if kind not in KINDS:
            raise ValueError(kind)
        if limit is None:
            rows = self.connection.execute(
                "SELECT compressed_payload FROM records "
                "WHERE kind=? AND scope=? AND subscope=? "
                "ORDER BY happened_at,seq", (kind, scope, subscope)
            )
        else:
            rows = self.connection.execute(
                "SELECT compressed_payload FROM ("
                " SELECT compressed_payload,happened_at,seq FROM records "
                " WHERE kind=? AND scope=? AND subscope=? "
                " ORDER BY happened_at DESC,seq DESC LIMIT ?) "
                "ORDER BY happened_at,seq", (kind, scope, subscope, limit)
            )
        return [self._unpack(row[0]) for row in rows]

    def latest(self, kind: str, scope: str, subscope: str = "") -> Optional[Dict[str, Any]]:
        row = self.connection.execute(
            "SELECT compressed_payload FROM records WHERE kind=? AND scope=? AND subscope=? "
            "ORDER BY happened_at DESC,seq DESC LIMIT 1", (kind, scope, subscope)
        ).fetchone()
        return None if row is None else self._unpack(row[0])

    def batch(self, batch_id: str) -> Optional[Dict[str, Any]]:
        row = self.connection.execute(
            "SELECT compressed_payload FROM records WHERE kind='quality_snapshot_recorded' "
            "AND batch_id=? ORDER BY happened_at DESC,seq DESC LIMIT 1", (batch_id,)
        ).fetchone()
        return None if row is None else self._unpack(row[0])

    def scan_hot(self, limits: Mapping[str, int]) -> Iterator[tuple[str, Dict[str, Any]]]:
        # Select bounded per-scope history (no full-state reconstruction).
        # SQLite window functions require SQLite 3.25+ (Python 3.9+ distributions).
        rows = self.connection.execute(
            "SELECT kind,compressed_payload FROM ("
            " SELECT kind,compressed_payload,happened_at,seq,"
            " ROW_NUMBER() OVER (PARTITION BY kind,scope,subscope "
            " ORDER BY happened_at DESC,seq DESC) AS rn"
            " FROM records) WHERE rn <= CASE kind "
            " WHEN 'quality_snapshot_recorded' THEN ? "
            " WHEN 'drift_baseline_recorded' THEN ? "
            " WHEN 'drift_observation_recorded' THEN ? "
            " WHEN 'schema_version_registered' THEN ? "
            " WHEN 'threshold_decision_recorded' THEN ? "
            " WHEN 'remediation_outcome_recorded' THEN ? "
            " WHEN 'conflict_reconciled' THEN ? "
            " WHEN 'source_reliability_recorded' THEN ? END "
            " ORDER BY happened_at,seq",
            (
                limits["snapshots_per_source"], limits["drift_baselines_per_metric"],
                limits["drift_observations_per_metric"], limits["schema_versions_per_source"],
                limits["threshold_decisions_per_rule"], limits["remediation_outcomes_per_rule"],
                limits["conflicts_per_scope"], limits["source_reliability_events"],
            )
        )
        for kind, blob in rows:
            yield str(kind), self._unpack(blob)

    def statistics(self) -> Dict[str, Any]:
        by_kind = dict(self.connection.execute(
            "SELECT kind,COUNT(*) FROM records GROUP BY kind"
        ).fetchall())
        archived = self.connection.execute(
            "SELECT COUNT(*) FROM records WHERE archived=1"
        ).fetchone()[0]
        latest = self.connection.execute("SELECT MAX(happened_at) FROM records").fetchone()[0]
        return {"records_by_kind": by_kind, "archived_snapshots": archived,
                "latest_recorded_at": latest,
                "total_records": sum(by_kind.values())}

    def batch_count(self) -> int:
        return int(self.connection.execute(
            "SELECT COUNT(DISTINCT batch_id) FROM records "
            "WHERE kind='quality_snapshot_recorded'"
        ).fetchone()[0])

    def scope_count(self, kind: str) -> int:
        return int(self.connection.execute(
            "SELECT COUNT(DISTINCT scope) FROM records WHERE kind=?", (kind,)
        ).fetchone()[0])

    def checkpoint(self, mode: str = "PASSIVE") -> tuple[int, int, int]:
        if mode not in {"PASSIVE", "FULL", "RESTART", "TRUNCATE"}:
            raise ValueError(mode)
        return tuple(self.connection.execute(f"PRAGMA wal_checkpoint({mode})").fetchone())

    def integrity_check(self) -> bool:
        return self.connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def iter_legacy_state(path: Path) -> Iterator[tuple[str, Dict[str, Any]]]:
    with Path(path).open("r", encoding="utf-8") as f:
        state = json.load(f)
    if not isinstance(state, dict):
        raise ValueError("Legacy state root is not an object")
    for section, kind, _nested in LEGACY_SECTIONS:
        container = state[section]
        if not isinstance(container, dict):
            raise ValueError(f"Invalid legacy state section: {section}")
        for value in container.values():
            if section == "snapshots_by_batch":
                values = [value]
            elif section in {"drift_baselines", "drift_observations"}:
                values = (record for bucket in value.values() for record in bucket)
            else:
                values = value
            for record in values:
                if not isinstance(record, dict):
                    raise ValueError(f"Bad legacy record in {section}")
                yield kind, record


def iter_legacy_journal(path: Path) -> Iterator[tuple[str, Dict[str, Any]]]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
                kind = event["event_type"]
                record = event["payload"]
                if not isinstance(record, dict):
                    raise ValueError("Event payload must be a dictionary")
                if kind not in KINDS:
                    raise ValueError(f"Unsupported event type: {kind}")
            except (KeyError, ValueError, TypeError) as exc:
                raise ValueError(f"Journal invalid at line {line_number}: {exc}") from exc
            yield kind, record


def file_fingerprint(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return Path(path).stat().st_size, digest.hexdigest()


def migrate(state_path: Path, journal_path: Path, destination: Path,
            *, hot_batch_limit: int = 2000) -> Dict[str, Any]:
    """Manual, fail-closed migration. Caller MUST stop all SLAI writers first.

    No changes are made to input files, and destination must not exist.
    An incomplete .building database may be removed manually and retried.
    """
    state_path, journal_path, destination = map(Path, (state_path, journal_path, destination))
    if not state_path.is_file() or not journal_path.is_file():
        raise FileNotFoundError("Both canonical state and journal are required for migration")
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")
    building = destination.with_name(destination.name + ".building")
    if building.exists():
        raise FileExistsError(f"Incomplete prior migration needs inspection: {building}")
    source_before = (file_fingerprint(state_path), file_fingerprint(journal_path))
    store = SqliteQualityStore(building, create=True, wal=False)
    counts: Counter[str] = Counter()
    try:
        store.connection.execute("BEGIN IMMEDIATE")
        # Journal contains event history, possibly including entries absent from the
        # trimmed snapshot. The JSON state can include records missing from journal.
        for kind, record in iter_legacy_journal(journal_path):
            counts["journal_rows"] += 1
            if store.insert(kind, record, allow_duplicate=True):
                counts["unique_from_journal"] += 1
        for kind, record in iter_legacy_state(state_path):
            counts["snapshot_rows"] += 1
            if store.insert(kind, record, allow_duplicate=True):
                counts["unique_from_snapshot"] += 1
        store.archive_old_batches(hot_batch_limit)
        with state_path.open("r", encoding="utf-8") as state_handle:
            legacy_metadata = json.load(state_handle)["metadata"]
        if not isinstance(legacy_metadata, dict):
            raise ValueError("Legacy metadata is not a JSON object")
        store.set_meta("legacy_metadata", json.dumps(legacy_metadata, sort_keys=True))
        store.set_meta("migration_complete", "1")
        store.set_meta("schema_version", "1")
        store.set_meta("source_state_sha256", source_before[0][1])
        store.set_meta("source_journal_sha256", source_before[1][1])
        store.connection.execute("COMMIT")
        if not store.integrity_check():
            raise RuntimeError("SQLite integrity check failed")
        stats = store.statistics()
        if stats["total_records"] != (
            counts["unique_from_journal"] + counts["unique_from_snapshot"]
        ):
            raise RuntimeError("Migration record count mismatch")
        store.close()
        if source_before != (file_fingerprint(state_path), file_fingerprint(journal_path)):
            raise RuntimeError("Legacy source changed while migrating. Keep the .building file for inspection")
        os.replace(building, destination)
        return {"counts": dict(counts), "database": str(destination), "statistics": stats}
    except BaseException:
        store.close()
        raise
