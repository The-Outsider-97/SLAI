"""Revision-bound, application-neutral workflow planning.

This is the durable *plan state* counterpart to PlanningMemory's statistical
checkpoints. It never executes agent code or commits host-owned documents.
Callers must supply authoritative revision changes; the host must still check
the expected base revision when committing a proposed mutation.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence


class RevisionWorkflowError(ValueError):
    """Invalid plan, transition, or stale execution attempt."""


_SUCCESS = frozenset({"succeeded", "verified"})


def _identifier(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RevisionWorkflowError(f"{field_name} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True)
class RevisionTask:
    """A declarative task, with exact read revisions where known.

    An empty ``reads`` map means the task conservatively depends on the whole
    project revision. ``writes`` declares intended changes, not permission to
    make them. Dependencies refer to task IDs in this same workflow.
    """

    task_id: str
    depends_on: tuple[str, ...] = ()
    reads: Mapping[str, str] = field(default_factory=dict)
    writes: tuple[str, ...] = ()
    capability: Optional[str] = None
    verification_required: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _identifier(self.task_id, "task_id"))
        deps = tuple(_identifier(d, "dependency") for d in self.depends_on)
        if len(deps) != len(set(deps)) or self.task_id in deps:
            raise RevisionWorkflowError("task dependencies must be unique and cannot include self")
        object.__setattr__(self, "depends_on", deps)
        reads = {_identifier(k, "read artifact"): _identifier(v, "read revision") for k, v in self.reads.items()}
        object.__setattr__(self, "reads", reads)
        writes = tuple(_identifier(w, "write artifact") for w in self.writes)
        if len(writes) != len(set(writes)):
            raise RevisionWorkflowError("write artifacts must be unique")
        object.__setattr__(self, "writes", writes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id, "depends_on": list(self.depends_on),
            "reads": dict(self.reads), "writes": list(self.writes),
            "capability": self.capability, "verification_required": self.verification_required,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RevisionTask":
        return cls(**dict(payload))


def _validate_graph(tasks: Mapping[str, RevisionTask]) -> list[str]:
    for task in tasks.values():
        missing = set(task.depends_on) - tasks.keys()
        if missing:
            raise RevisionWorkflowError(f"{task.task_id}: missing dependencies: {sorted(missing)}")
    visiting: set[str] = set()
    visited: set[str] = set()
    ordered: list[str] = []

    def visit(task_id: str) -> None:
        if task_id in visiting:
            raise RevisionWorkflowError(f"cyclic task dependency involving {task_id}")
        if task_id in visited:
            return
        visiting.add(task_id)
        for parent in sorted(tasks[task_id].depends_on):
            visit(parent)
        visiting.remove(task_id)
        visited.add(task_id)
        ordered.append(task_id)

    for identifier in sorted(tasks):
        visit(identifier)
    return ordered


def _conflicts(a: RevisionTask, b: RevisionTask) -> bool:
    a_reads, b_reads = set(a.reads), set(b.reads)
    a_writes, b_writes = set(a.writes), set(b.writes)
    # Unknown read sets require conservative serialization against writes.
    if (not a_reads and b_writes) or (not b_reads and a_writes):
        return True
    return bool(a_writes & (b_reads | b_writes) or b_writes & a_reads)


class RevisionWorkflowStore:
    """SQLite-backed plan revisions, transitions and replayable history.

    Uses SQLite write transactions and a monotonic state version, rather than
    rewriting a mutable JSON checkpoint for each task transition. A new process
    can reopen the same database and resume without executing prior effects.
    """

    def __init__(self, path: str | Path = "data/planning/revision_workflows.sqlite3") -> None:
        selected = Path(path).expanduser() if str(path) != ":memory:" else None
        if selected is not None and not selected.is_absolute():
            selected = Path(__file__).resolve().parents[3] / selected
        self.path = str(selected.resolve()) if selected is not None else ":memory:"
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path, timeout=15, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA busy_timeout=15000")
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._lock = threading.RLock()
        with self._connection:
            self._connection.execute("CREATE TABLE IF NOT EXISTS revision_workflows ("
                                     "plan_id TEXT PRIMARY KEY, state_version INTEGER NOT NULL, data TEXT NOT NULL)")
            self._connection.execute("CREATE TABLE IF NOT EXISTS revision_workflow_events ("
                                     "sequence INTEGER PRIMARY KEY AUTOINCREMENT, plan_id TEXT NOT NULL, "
                                     "event_id TEXT NOT NULL UNIQUE, event_type TEXT NOT NULL, payload TEXT NOT NULL, "
                                     "FOREIGN KEY(plan_id) REFERENCES revision_workflows(plan_id))")
            self._connection.execute("CREATE INDEX IF NOT EXISTS ix_revision_workflow_events_plan "
                                     "ON revision_workflow_events(plan_id, sequence)")

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def _load(self, plan_id: str) -> dict[str, Any]:
        row = self._connection.execute("SELECT data FROM revision_workflows WHERE plan_id=?", (plan_id,)).fetchone()
        if row is None:
            raise RevisionWorkflowError(f"unknown plan_id: {plan_id}")
        return json.loads(row["data"])

    def get_plan(self, plan_id: str) -> dict[str, Any]:
        with self._lock:
            return self._load(_identifier(plan_id, "plan_id"))

    def events(self, plan_id: str) -> list[dict[str, Any]]:
        with self._lock:
            self._load(plan_id)
            rows = self._connection.execute(
                "SELECT sequence, event_id, event_type, payload FROM revision_workflow_events "
                "WHERE plan_id=? ORDER BY sequence", (plan_id,),
            ).fetchall()
            return [{"sequence": row["sequence"], "event_id": row["event_id"],
                     "event_type": row["event_type"], "payload": json.loads(row["payload"])} for row in rows]

    def _event(self, plan_id: str, event_type: str, payload: Mapping[str, Any]) -> None:
        enriched = {"timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), **payload}
        self._connection.execute(
            "INSERT INTO revision_workflow_events(plan_id,event_id,event_type,payload) VALUES (?,?,?,?)",
            (plan_id, uuid.uuid4().hex, event_type, json.dumps(enriched, sort_keys=True)),
        )

    def publish_provenance_events(self, plan_id: str, provenance_agent: Any) -> int:
        """Replay durable plan events through the ProvenanceAgent's public API.

        This is an idempotent recovery bridge, not a distributed transaction.
        A failed publication can be retried after the host restores connectivity.
        The receiving provenance store rejects conflicting reuse of event IDs.
        """
        state = self.get_plan(plan_id)
        published = 0
        for event in self.events(plan_id):
            detail = dict(event["payload"])
            outputs = detail.get("output_refs") or {}
            records = list(outputs.items()) if outputs else [(None, None)]
            for artifact_id, revision_id in records:
                payload = {
                    "event_id": f"{event['event_id']}:{artifact_id}" if artifact_id else event["event_id"],
                    "event_type": event["event_type"],
                    "project_id": state["project_id"], "plan_id": plan_id,
                    "task_id": detail.get("task_id"), "run_id": detail.get("run_id"),
                    "artifact_id": artifact_id, "revision_id": revision_id,
                    "timestamp": detail.get("timestamp"),
                    "metadata": {key: value for key, value in detail.items() if key != "timestamp"},
                }
                if payload["timestamp"] is None:
                    payload.pop("timestamp")
                provenance_agent.record_event({"event_type": "revision_event", "event_id": payload["event_id"],
                                               "payload": payload})
                published += 1
        return published

    def create_plan(self, project_id: str, tasks: Sequence[RevisionTask],
                    source_revisions: Mapping[str, str], *, plan_id: Optional[str] = None) -> dict[str, Any]:
        project = _identifier(project_id, "project_id")
        identifier = _identifier(plan_id or uuid.uuid4().hex, "plan_id")
        by_id = {task.task_id: task for task in tasks}
        if not by_id or len(by_id) != len(tasks):
            raise RevisionWorkflowError("plan requires distinct task identifiers")
        order = _validate_graph(by_id)
        revisions = {_identifier(k, "artifact"): _identifier(v, "revision") for k, v in source_revisions.items()}
        state = {
            "plan_id": identifier, "project_id": project, "plan_revision": 1,
            "state_version": 1, "source_revisions": revisions,
            "tasks": {key: item.to_dict() for key, item in by_id.items()},
            "order": order, "status": {key: "pending" for key in by_id},
            "attempts": {key: [] for key in by_id},
        }
        with self._lock, self._connection:
            self._connection.execute("INSERT INTO revision_workflows(plan_id,state_version,data) VALUES (?,?,?)",
                                     (identifier, 1, json.dumps(state, sort_keys=True)))
            self._event(identifier, "plan.created", {"plan_revision": 1, "tasks": order, "source_revisions": revisions})
        return state

    def _mutate(self, plan_id: str, event_type: str,
                modifier: Callable[[dict[str, Any]], Mapping[str, Any]]) -> dict[str, Any]:
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                state = self._load(plan_id)
                old_version = state["state_version"]
                detail = dict(modifier(state))
                if detail.pop("_no_change", False):
                    self._connection.rollback()
                    return state
                state["state_version"] = old_version + 1
                cursor = self._connection.execute("UPDATE revision_workflows SET state_version=?,data=? "
                                                  "WHERE plan_id=? AND state_version=?",
                                                  (state["state_version"], json.dumps(state, sort_keys=True), plan_id, old_version))
                if cursor.rowcount != 1:
                    raise RevisionWorkflowError("concurrent plan update")
                self._event(plan_id, event_type, {"plan_revision": state["plan_revision"], **detail})
                self._connection.commit()
                return state
            except Exception:
                self._connection.rollback()
                raise

    @staticmethod
    def _ready(state: Mapping[str, Any], task_id: str) -> bool:
        if state["status"][task_id] != "pending":
            return False
        task = state["tasks"][task_id]
        if any(state["status"][dep] not in _SUCCESS for dep in task["depends_on"]):
            return False
        return all(state["source_revisions"].get(key) == revision for key, revision in task["reads"].items())

    def ready_groups(self, plan_id: str, *, max_parallel: int = 4) -> list[list[str]]:
        if max_parallel < 1:
            raise RevisionWorkflowError("max_parallel must be positive")
        state = self.get_plan(plan_id)
        batches: list[list[str]] = []
        candidates = [task_id for task_id in state["order"] if self._ready(state, task_id)]
        for task_id in candidates:
            task = RevisionTask.from_dict(state["tasks"][task_id])
            for batch in batches:
                if len(batch) < max_parallel and all(
                    not _conflicts(task, RevisionTask.from_dict(state["tasks"][other])) for other in batch
                ):
                    batch.append(task_id)
                    break
            else:
                batches.append([task_id])
        return batches

    def claim_task(self, plan_id: str, task_id: str) -> dict[str, Any]:
        task_id = _identifier(task_id, "task_id")
        run_id = uuid.uuid4().hex

        def change(state: dict[str, Any]) -> dict[str, Any]:
            if task_id not in state["tasks"] or not self._ready(state, task_id):
                raise RevisionWorkflowError(f"task {task_id} is not ready against current revisions")
            state["status"][task_id] = "running"
            task = state["tasks"][task_id]
            inputs = dict(task["reads"] or state["source_revisions"])
            state["attempts"][task_id].append({"run_id": run_id, "input_revisions": inputs,
                                                "status": "running", "plan_revision": state["plan_revision"]})
            return {"task_id": task_id, "run_id": run_id, "input_revisions": inputs}

        state = self._mutate(plan_id, "task.started", change)
        return state["attempts"][task_id][-1].copy()

    def complete_task(self, plan_id: str, task_id: str, run_id: str, *,
                      output_refs: Optional[Mapping[str, str]] = None, error: Optional[str] = None) -> dict[str, Any]:
        output_refs = dict(output_refs or {})
        for key, value in output_refs.items():
            _identifier(key, "output artifact")
            _identifier(value, "output revision")

        def change(state: dict[str, Any]) -> dict[str, Any]:
            if state["status"].get(task_id) != "running":
                raise RevisionWorkflowError("task is not running (possibly invalidated)")
            attempt = state["attempts"][task_id][-1]
            if attempt["run_id"] != run_id:
                raise RevisionWorkflowError("execution attempt mismatch")
            for key, revision in attempt["input_revisions"].items():
                if state["source_revisions"].get(key) != revision:
                    raise RevisionWorkflowError("input revision changed during execution")
            status = "failed" if error else (
                "awaiting_verification" if state["tasks"][task_id]["verification_required"] else "succeeded"
            )
            attempt.update({"status": status, "output_refs": output_refs, "error": error})
            state["status"][task_id] = status
            return {"task_id": task_id, "run_id": run_id, "status": status, "output_refs": output_refs}

        return self._mutate(plan_id, "task.failed" if error else "task.completed", change)

    def verify_task(self, plan_id: str, task_id: str, *, accepted: bool, evidence_id: str) -> dict[str, Any]:
        evidence_id = _identifier(evidence_id, "evidence_id")

        def change(state: dict[str, Any]) -> dict[str, Any]:
            if state["status"].get(task_id) != "awaiting_verification":
                raise RevisionWorkflowError("task has no pending verification")
            status = "verified" if accepted else "failed"
            state["status"][task_id] = status
            state["attempts"][task_id][-1].update({"status": status, "evidence_id": evidence_id})
            return {"task_id": task_id, "status": status, "evidence_id": evidence_id}

        return self._mutate(plan_id, "verification.completed", change)

    def apply_change_set(self, plan_id: str, changed_revisions: Mapping[str, str],
                         *, reason: str = "source revision changed") -> dict[str, Any]:
        changed = {_identifier(k, "artifact"): _identifier(v, "revision") for k, v in changed_revisions.items()}

        def change(state: dict[str, Any]) -> dict[str, Any]:
            actual = {key: revision for key, revision in changed.items()
                      if state["source_revisions"].get(key) != revision}
            if not actual:
                return {"_no_change": True}
            state["source_revisions"].update(actual)
            affected = set()
            for task_id in state["order"]:
                task = state["tasks"][task_id]
                if not task["reads"] or set(task["reads"]) & set(actual) or set(task["depends_on"]) & affected:
                    affected.add(task_id)
            for task_id in affected:
                status = state["status"][task_id]
                if status == "outcome_unknown":
                    # An external side effect may already have occurred. Keep
                    # the uncertainty gate until its owner reconciles it.
                    continue
                if status == "running":
                    state["attempts"][task_id][-1]["status"] = "stale"
                if status != "stale":
                    state["status"][task_id] = "stale"
            state["plan_revision"] += 1
            return {"changed": actual, "invalidated": sorted(affected), "reason": reason}

        return self._mutate(plan_id, "plan.invalidated", change)

    def replan_stale(self, plan_id: str, *, reason: str = "recompute invalidated tasks") -> dict[str, Any]:
        """Explicitly rebase stale tasks; completed attempts remain in history."""

        def change(state: dict[str, Any]) -> dict[str, Any]:
            stale = [task_id for task_id in state["order"] if state["status"][task_id] == "stale"]
            for task_id in stale:
                task = state["tasks"][task_id]
                if any(key not in state["source_revisions"] for key in task["reads"]):
                    raise RevisionWorkflowError(f"cannot rebase {task_id}: required source revision is unknown")
                task["reads"] = {key: state["source_revisions"][key] for key in task["reads"]}
                state["status"][task_id] = "pending"
            if stale:
                state["plan_revision"] += 1
                return {"reset_tasks": stale, "reason": reason}
            return {"_no_change": True}

        return self._mutate(plan_id, "plan.revised", change)

    def mark_interrupted_tasks(self, plan_id: str) -> dict[str, Any]:
        """On recovery, mark orphaned runs uncertain; never retry them blindly."""

        def change(state: dict[str, Any]) -> dict[str, Any]:
            interrupted = [name for name in state["order"] if state["status"][name] == "running"]
            if not interrupted:
                return {"_no_change": True}
            for name in interrupted:
                state["status"][name] = "outcome_unknown"
                state["attempts"][name][-1]["status"] = "outcome_unknown"
            return {"tasks": interrupted, "reason": "recovery requires external outcome reconciliation"}

        return self._mutate(plan_id, "task.outcome_unknown", change)

    def resolve_interrupted_task(self, plan_id: str, task_id: str, run_id: str,
                                 *, succeeded: bool, evidence_id: str) -> dict[str, Any]:
        """Resolve an uncertain run only with evidence from its execution owner."""
        _identifier(evidence_id, "evidence_id")

        def change(state: dict[str, Any]) -> dict[str, Any]:
            if state["status"].get(task_id) != "outcome_unknown":
                raise RevisionWorkflowError("task does not have an uncertain execution outcome")
            attempt = state["attempts"][task_id][-1]
            if attempt["run_id"] != run_id:
                raise RevisionWorkflowError("execution attempt mismatch")
            if any(state["source_revisions"].get(key) != value
                   for key, value in attempt["input_revisions"].items()):
                raise RevisionWorkflowError("cannot certify obsolete inputs as current")
            status = ("awaiting_verification" if state["tasks"][task_id]["verification_required"]
                      else "succeeded") if succeeded else "failed"
            state["status"][task_id] = status
            attempt.update({"status": status, "reconciliation_evidence_id": evidence_id})
            return {"task_id": task_id, "run_id": run_id, "status": status,
                    "reconciliation_evidence_id": evidence_id}

        return self._mutate(plan_id, "task.reconciled", change)


__all__ = ["RevisionTask", "RevisionWorkflowError", "RevisionWorkflowStore"]
