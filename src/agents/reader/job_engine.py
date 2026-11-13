"""Production-grade Job Engine for the Reader subsystem.

The job engine makes multi-file Reader work durable: a batch can be killed at any point and
resumed later without redoing the items that already finished.

Design goals
------------
- Reuse ``ReaderMemory`` for persistence (atomic writes, integrity hash, JSON envelopes).
  Job records are written with ``exact=True`` so they are stored verbatim and can be acted on.
- Append-only journal. Every state change is one small immutable checkpoint carrying a
  monotonically increasing ``seq``; a job is rebuilt by folding its events in order. Writing
  an event is O(1) however many items the job has, and a torn or corrupt event only loses
  that one event (the item is simply redone), never the whole job.
- A completed item is only reused on resume if the artifacts it recorded still exist with
  the recorded size and SHA-256 (verified by ``IntegrityEngine``).
- Explicit lifecycle with validated transitions (``JobStateError`` on a bad one), cooperative
  cancellation and stale-run detection so a crashed process does not block resume forever.
- The job store is separate from the diagnostic checkpoint store, so checkpoint pruning
  can never delete a job.
- Single writer per job is assumed; the engine serialises its own writes with a lock but
  does not coordinate across processes.
"""

from __future__ import annotations

import copy
import os
import threading
import uuid

from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from .utils.config_loader import get_config_section, load_global_config
from .utils.reader_error import *
from .utils.reader_helpers import *
from .integrity_engine import IntegrityEngine
from .reader_memory import ReaderMemory
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Job Engine")
printer = PrettyPrinter()

JOB_ENGINE_VERSION = "1.0.0"
JOB_SCHEMA_VERSION = 1

_CREATED_STEP = "job_created"
_EVENT_STEP = "job_event"

JOB_STATUSES = ("created", "running", "completed", "partial", "failed", "cancelled", "interrupted")
ITEM_STATUSES = ("pending", "running", "done", "failed", "skipped")
TERMINAL_STATUSES = frozenset({"completed", "partial", "failed", "cancelled"})

# Allowed lifecycle transitions. Retrying a partial/failed job goes through ``running`` again.
_TRANSITIONS: Dict[str, frozenset] = {
    "created": frozenset({"running", "cancelled"}),
    "running": frozenset({"completed", "partial", "failed", "cancelled", "interrupted"}),
    "interrupted": frozenset({"running", "cancelled"}),
    "partial": frozenset({"running"}),
    "failed": frozenset({"running"}),
    "completed": frozenset({"interrupted"}),  # only through reopen_job(): recorded artifacts went missing
    "cancelled": frozenset(),
}

_KNOWN_CONFIG_KEYS = (
    "job_dir",
    "max_jobs",
    "stale_after_seconds",
    "max_item_result_bytes",
    "verify_artifacts_on_resume",
    "retain_terminal_seconds",
)


class JobEngine:
    """Durable job journal with resume, artifact reuse and cooperative cancellation."""

    def __init__(self, memory: Optional[ReaderMemory] = None, integrity: Optional[IntegrityEngine] = None) -> None:
        self.config = load_global_config()
        self.je_config = get_config_section("job_engine") or {}
        warn_unknown_config_keys("job_engine", self.je_config, _KNOWN_CONFIG_KEYS)

        # Pinned to an absolute path now: a relative store would silently move if the working directory changes.
        self.job_dir = str(resolve_path(str(self.je_config.get("job_dir") or "tmp/reader/jobs")))
        self.max_jobs = coerce_int(self.je_config.get("max_jobs"), 1000, minimum=1)
        self.stale_after_seconds = coerce_int(self.je_config.get("stale_after_seconds"), 3600, minimum=1)
        self.max_item_result_bytes = coerce_int(self.je_config.get("max_item_result_bytes"), 65_536, minimum=256)
        self.verify_artifacts_on_resume = coerce_bool(self.je_config.get("verify_artifacts_on_resume"), True)
        self.retain_terminal_seconds = coerce_int(self.je_config.get("retain_terminal_seconds"), 0, minimum=0)

        # A dedicated store: job records must never compete with diagnostic checkpoints for retention,
        # and per-event index rewrites / audit files would make long jobs quadratic.
        self.memory = memory or ReaderMemory(
            {
                "checkpoint_dir": self.job_dir,
                "max_checkpoint_entries": 10_000_000,
                "write_indexes": False,
                "enable_audit_log": False,
                "prune_on_init": False,
                "checkpoint_prefix": "job",
            }
        )
        self.integrity = integrity or IntegrityEngine()
        self._lock = threading.RLock()
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._cancel_requested: set = set()
        self.settings = {
            "job_dir": self.job_dir,
            "max_jobs": self.max_jobs,
            "stale_after_seconds": self.stale_after_seconds,
            "max_item_result_bytes": self.max_item_result_bytes,
            "verify_artifacts_on_resume": self.verify_artifacts_on_resume,
            "retain_terminal_seconds": self.retain_terminal_seconds,
        }
        logger.info("Job Engine initialized | job_dir=%s max_jobs=%s", self.job_dir, self.max_jobs)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def capabilities(self) -> Dict[str, Any]:
        return {
            "engine_version": JOB_ENGINE_VERSION,
            "schema_version": JOB_SCHEMA_VERSION,
            "job_statuses": list(JOB_STATUSES),
            "item_statuses": list(ITEM_STATUSES),
            "settings": dict(self.settings),
        }

    def describe(self) -> Dict[str, Any]:
        return {"status": "ok", **self.capabilities()}

    # ------------------------------------------------------------------
    # Journal plumbing
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_job_id(job_id: Any) -> str:
        text = str(job_id or "").strip()
        normalized = safe_filename(text, fallback="")
        if not normalized or normalized != text or len(text) > 120:
            raise JobError("Reader job id is invalid", {"job_id": text[:120]})
        return normalized

    def _write(self, job_id: str, step: str, payload: Mapping[str, Any]) -> str:
        try:
            return self.memory.write_checkpoint(step, payload, run_id=job_id, exact=True)
        except ReaderError as exc:
            raise JobError("Failed writing Reader job journal", {"job_id": job_id, "step": step}, exc) from exc
        except OSError as exc:
            raise JobError("Failed writing Reader job journal", {"job_id": job_id, "step": step}, exc) from exc

    def _state(self, job_id: str, *, fresh: bool = False) -> Dict[str, Any]:
        """The folded job. Cached and updated in place by this engine's own writes, so a long job
        costs one journal read, not one per item. ``fresh=True`` re-reads (needed to see other processes)."""

        with self._lock:
            if fresh or job_id not in self._jobs:
                self._jobs[job_id] = self._fold(job_id)
            return self._jobs[job_id]

    def _append(self, job_id: str, event: str, body: Mapping[str, Any]) -> int:
        with self._lock:
            job = self._state(job_id)
            seq = int(job["seq"]) + 1
            record = {"schema": JOB_SCHEMA_VERSION, "job_id": job_id, "seq": seq, "event": event, "at": utc_timestamp(), **dict(body)}
            self._write(job_id, _EVENT_STEP, record)  # a failed write leaves the cached state untouched
            self._apply(job, record)
            return seq

    def _read_entries(self, job_id: str) -> List[Dict[str, Any]]:
        try:
            entries = self.memory.list_checkpoints(run_id=job_id, newest_first=False)
        except ReaderError as exc:
            raise JobError("Failed listing Reader job journal", {"job_id": job_id}, exc) from exc
        records: List[Dict[str, Any]] = []
        for entry in entries:
            try:
                envelope = self.memory.read_checkpoint(entry["path"], verify_integrity=True)
                body = envelope["payload"]["payload"]
                records.append({"step": envelope["payload"].get("step"), "body": body})
            except Exception as exc:
                # A lost event costs at most a redo of one item; the rest of the job stays usable.
                logger.warning("Skipping unreadable job journal entry %s: %s", entry.get("path"), exc)
        return records

    def _fold(self, job_id: str) -> Dict[str, Any]:
        """Rebuild a job by folding its journal. Raises ``JobNotFoundError`` if it never existed."""

        created: Optional[Dict[str, Any]] = None
        events: List[Dict[str, Any]] = []
        for record in self._read_entries(job_id):
            if record["step"] == _CREATED_STEP and created is None:
                created = record["body"]
            elif record["step"] == _EVENT_STEP:
                events.append(record["body"])
        if created is None:
            raise JobNotFoundError(job_id)
        events.sort(key=lambda item: (int(item.get("seq", 0)), float(item.get("at", 0.0))))

        job: Dict[str, Any] = {
            "job_id": job_id,
            "kind": created.get("kind"),
            "status": "created",
            "created_at": created.get("created_at"),
            "updated_at": created.get("created_at"),
            "params": created.get("params", {}),
            "idempotency_key": created.get("idempotency_key"),
            "items": {},
            "item_order": [],
            "owner": None,
            "reason": None,
            "attempts": 0,
            "seq": 0,
        }
        for item in created.get("items", []):
            item_id = str(item["item_id"])
            job["items"][item_id] = {"item_id": item_id, "key": item.get("key"), "data": item.get("data", {}),
                                     "status": "pending", "attempt": 0, "result": None, "artifacts": [], "error": None}
            job["item_order"].append(item_id)
        job["counts"] = self._counts(job)
        for event in events:
            self._apply(job, event)
        return job

    def _apply(self, job: Dict[str, Any], event: Mapping[str, Any]) -> None:
        job["seq"] = max(int(job["seq"]), int(event.get("seq", 0)))
        job["updated_at"] = event.get("at", job["updated_at"])
        kind = event.get("event")
        if kind == "state":
            job["status"] = event["status"]
            job["reason"] = event.get("reason")
            job["owner"] = event.get("owner", job["owner"])
            if event["status"] == "running":
                job["attempts"] += 1
        elif kind == "item":
            item = job["items"].get(str(event.get("item_id")))
            if item is not None:
                item["status"] = event["status"]
                item["attempt"] = int(event.get("attempt", item["attempt"]))
                if event["status"] == "done":
                    item["result"] = event.get("result")
                    item["artifacts"] = list(event.get("artifacts", []))
                    item["error"] = None
                elif event["status"] in {"failed", "skipped"}:
                    item["error"] = event.get("error")
                elif event["status"] == "pending":
                    item["result"], item["artifacts"], item["error"] = None, [], None
        job["counts"] = self._counts(job)

    @staticmethod
    def _counts(job: Mapping[str, Any]) -> Dict[str, int]:
        counts = {name: 0 for name in ITEM_STATUSES}
        for item in job["items"].values():
            counts[item["status"]] = counts.get(item["status"], 0) + 1
        counts["total"] = len(job["items"])
        return counts

    def _transition(self, job: Mapping[str, Any], target: str, *, reason: Optional[str] = None,
                    owner: Optional[str] = None) -> None:
        current = job["status"]
        if target not in _TRANSITIONS.get(current, frozenset()):
            raise JobStateError(job["job_id"], f"Cannot move job from '{current}' to '{target}'",
                                {"from": current, "to": target})
        body: Dict[str, Any] = {"status": target, "reason": reason}
        if owner is not None:
            body["owner"] = owner
        self._append(job["job_id"], "state", body)
        logger.info("Job %s: %s -> %s%s", job["job_id"], current, target, f" ({reason})" if reason else "")

    def _bounded_result(self, result: Any) -> Any:
        safe = json_safe(result)
        if len(stable_json_dumps(safe)) <= self.max_item_result_bytes:
            return safe
        return {"truncated": True, "result_sha256": stable_hash(safe),
                "note": f"result exceeded max_item_result_bytes={self.max_item_result_bytes}"}

    def _artifact_record(self, artifact: Any) -> Dict[str, Any]:
        path = expand_path(artifact["path"] if isinstance(artifact, Mapping) else artifact)
        if not path.is_file():
            raise JobError("Job artifact is not a file", {"path": str(path)})
        return {"path": str(path), "size_bytes": int(path.stat().st_size), "sha256": sha256_file(path)}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def create_job(
        self,
        kind: str,
        items: Iterable[Mapping[str, Any] | str],
        *,
        params: Optional[Mapping[str, Any]] = None,
        job_id: Optional[str] = None,
        idempotency_key: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Create a job, or return the existing one when ``idempotency_key`` was already used.

        ``items`` are strings (used as both id and key) or mappings with ``key`` (stable identity,
        for example a path) and optional ``item_id`` and ``data``.
        """

        timer = OperationTimer("create_job")
        kind_name = safe_filename(str(kind or ""), fallback="")
        if not kind_name:
            raise JobError("Reader job kind is required", {"kind": kind})
        if idempotency_key is not None and job_id is None:
            job_id = f"{kind_name}_{stable_hash(idempotency_key)[:20]}"
        job_id = self._normalize_job_id(job_id) if job_id else f"{kind_name}_{utc_compact_timestamp()}_{uuid.uuid4().hex[:8]}"

        with self._lock:
            try:
                existing = copy.deepcopy(self._state(job_id, fresh=True))
            except JobNotFoundError:
                self._jobs.pop(job_id, None)
                existing = None
            if existing is not None:
                if idempotency_key is None:
                    raise JobError("Reader job already exists", {"job_id": job_id})
                logger.info("Job %s already exists; returning it (idempotent create)", job_id)
                return build_success_result(
                    operation="create_job", payload={"job": existing, "created": False},
                    metadata={"engine_version": JOB_ENGINE_VERSION}, timer=timer)

            normalized_items: List[Dict[str, Any]] = []
            seen: set = set()
            for index, raw in enumerate(items):
                spec = dict(raw) if isinstance(raw, Mapping) else {"key": str(raw)}
                key = str(spec.get("key", spec.get("item_id", index)))
                item_id = safe_filename(str(spec.get("item_id") or f"{index:05d}_{stable_hash(key)[:10]}"), fallback=f"item{index:05d}")
                if item_id in seen:
                    item_id = f"{item_id}_{index}"
                seen.add(item_id)
                normalized_items.append({"item_id": item_id, "key": key, "data": json_safe(spec.get("data", {}))})
            if not normalized_items:
                raise JobError("Reader job needs at least one item", {"job_id": job_id})

            self._write(job_id, _CREATED_STEP, {
                "schema": JOB_SCHEMA_VERSION, "job_id": job_id, "kind": kind_name, "created_at": utc_timestamp(),
                "params": json_safe(params or {}), "idempotency_key": stable_hash(idempotency_key) if idempotency_key is not None else None,
                "items": normalized_items,
            })
            job = copy.deepcopy(self._state(job_id, fresh=True))
        logger.info("Job %s created | kind=%s items=%d", job_id, kind_name, len(normalized_items))
        return build_success_result(operation="create_job", payload={"job": job, "created": True},
                                    metadata={"engine_version": JOB_ENGINE_VERSION}, timer=timer)

    def get_job(self, job_id: str, *, fresh: bool = False) -> Dict[str, Any]:
        """Return a copy of the job. ``fresh=True`` re-reads the journal (sees other processes' writes)."""

        job_id = self._normalize_job_id(job_id)
        with self._lock:
            job = copy.deepcopy(self._state(job_id, fresh=fresh))
        job["stale"] = self._is_stale(job)
        return job

    def _is_stale(self, job: Mapping[str, Any]) -> bool:
        return job["status"] == "running" and (utc_timestamp() - float(job.get("updated_at") or 0.0)) > self.stale_after_seconds

    def start_job(self, job_id: str, *, owner: Optional[str] = None, force: bool = False) -> Dict[str, Any]:
        """Move a job to ``running`` (first start, or resume after interruption / failure).

        A job found ``running`` is only taken over when it is stale or ``force`` is set, so two
        processes cannot silently drive the same job.
        """

        job_id = self._normalize_job_id(job_id)
        owner_name = owner or f"pid:{os.getpid()}"
        with self._lock:
            job = self._state(job_id, fresh=True)
            if job["status"] == "running":
                if not (force or self._is_stale(job)):
                    raise JobStateError(job_id, "Job is already running; pass force=True to take it over",
                                        {"owner": job.get("owner"), "stale_after_seconds": self.stale_after_seconds})
                self._transition(job, "interrupted", reason="taken over" if force else "stale run detected")
            self._transition(job, "running", owner=owner_name)
            # Items left 'running' by a dead process are redone.
            for item in job["items"].values():
                if item["status"] == "running":
                    self._append(job_id, "item", {"item_id": item["item_id"], "status": "pending", "attempt": item["attempt"]})
            return copy.deepcopy(self._state(job_id))

    def begin_item(self, job_id: str, item_id: str) -> Dict[str, Any]:
        job_id = self._normalize_job_id(job_id)
        with self._lock:
            job = self._state(job_id)
            item = self._require_item(job, item_id)
            if job["status"] != "running":
                raise JobStateError(job_id, "Items can only be started while the job is running", {"status": job["status"]})
            attempt = item["attempt"] + 1
            self._append(job_id, "item", {"item_id": item["item_id"], "status": "running", "attempt": attempt})
            return {**item, "status": "running", "attempt": attempt}

    def complete_item(self, job_id: str, item_id: str, result: Any = None, *,
                      artifacts: Optional[Sequence[Any]] = None) -> Dict[str, Any]:
        """Record an item as done together with the size and SHA-256 of every artifact it produced."""

        job_id = self._normalize_job_id(job_id)
        records = [self._artifact_record(item) for item in (artifacts or [])]
        with self._lock:
            job = self._state(job_id)
            item = self._require_item(job, item_id)
            self._append(job_id, "item", {"item_id": item["item_id"], "status": "done", "attempt": max(1, item["attempt"]),
                                          "result": self._bounded_result(result), "artifacts": records})
            return {**item, "status": "done", "artifacts": records}

    def fail_item(self, job_id: str, item_id: str, error: BaseException | Mapping[str, Any] | str) -> Dict[str, Any]:
        job_id = self._normalize_job_id(job_id)
        if isinstance(error, BaseException):
            payload = reader_error_payload(normalize_reader_error(error, stage=ReaderErrorStage.JOB))
        elif isinstance(error, Mapping):
            payload = json_safe(error)
        else:
            payload = {"message": str(error)}
        with self._lock:
            job = self._state(job_id)
            item = self._require_item(job, item_id)
            self._append(job_id, "item", {"item_id": item["item_id"], "status": "failed", "attempt": max(1, item["attempt"]), "error": payload})
            return {**item, "status": "failed", "error": payload}

    def skip_item(self, job_id: str, item_id: str, reason: str) -> Dict[str, Any]:
        job_id = self._normalize_job_id(job_id)
        with self._lock:
            job = self._state(job_id)
            item = self._require_item(job, item_id)
            self._append(job_id, "item", {"item_id": item["item_id"], "status": "skipped", "attempt": item["attempt"], "error": {"message": reason}})
            return {**item, "status": "skipped"}

    @staticmethod
    def _require_item(job: Mapping[str, Any], item_id: str) -> Dict[str, Any]:
        item = job["items"].get(str(item_id))
        if item is None:
            raise JobError("Reader job item does not exist", {"job_id": job["job_id"], "item_id": str(item_id)})
        return item

    def finish_job(self, job_id: str) -> Dict[str, Any]:
        """Derive the terminal status from the items: completed / partial / failed."""

        job_id = self._normalize_job_id(job_id)
        with self._lock:
            job = self._state(job_id)
            counts = job["counts"]
            unfinished = counts["pending"] + counts["running"]
            if unfinished:
                raise JobStateError(job_id, "Job still has unfinished items", {"pending": counts["pending"], "running": counts["running"]})
            if counts["failed"] == 0:
                target = "completed"
            elif counts["done"] + counts["skipped"] == 0:
                target = "failed"
            else:
                target = "partial"
            self._transition(job, target, reason=f"{counts['done']} done, {counts['failed']} failed, {counts['skipped']} skipped")
            return copy.deepcopy(self._state(job_id))

    def cancel_job(self, job_id: str, reason: str = "cancelled by caller") -> Dict[str, Any]:
        job_id = self._normalize_job_id(job_id)
        with self._lock:
            job = self._state(job_id)
            self._transition(job, "cancelled", reason=reason)
            self._cancel_requested.add(job_id)
            return copy.deepcopy(self._state(job_id))

    def reopen_job(self, job_id: str, reason: str) -> Dict[str, Any]:
        """Make a completed job resumable again, e.g. because its recorded artifacts were deleted or changed.

        ``start_job`` still refuses a completed job; reopening is a deliberate, journaled step.
        """

        job_id = self._normalize_job_id(job_id)
        with self._lock:
            job = self._state(job_id)
            self._transition(job, "interrupted", reason=reason)
            return copy.deepcopy(self._state(job_id))

    def interrupt_job(self, job_id: str, reason: str) -> Dict[str, Any]:
        """Mark a running job interrupted (resumable) instead of waiting for the stale timeout."""

        job_id = self._normalize_job_id(job_id)
        with self._lock:
            job = self._state(job_id)
            if job["status"] == "running":
                self._transition(job, "interrupted", reason=reason)
            return copy.deepcopy(self._state(job_id))

    def is_cancelled(self, job_id: str, *, fresh: bool = False) -> bool:
        """Cooperative cancellation check. ``fresh=True`` re-reads the journal to see another process's cancel."""

        job_id = self._normalize_job_id(job_id)
        if job_id in self._cancel_requested:
            return True
        try:
            return self.get_job(job_id, fresh=fresh)["status"] == "cancelled"
        except JobNotFoundError:
            return False

    # ------------------------------------------------------------------
    # Resume support
    # ------------------------------------------------------------------

    def pending_items(self, job_id: str, *, retry_failed: bool = False) -> List[Dict[str, Any]]:
        job = self.get_job(job_id)
        wanted = {"pending", "running"} | ({"failed"} if retry_failed else set())
        return [job["items"][item_id] for item_id in job["item_order"] if job["items"][item_id]["status"] in wanted]

    def reusable_item(self, job_id: str, item_id: str) -> Optional[Dict[str, Any]]:
        """Return a finished item only if every artifact it recorded is still byte-identical."""

        job = self.get_job(job_id)
        item = job["items"].get(str(item_id))
        if item is None or item["status"] != "done":
            return None
        if self.verify_artifacts_on_resume and not self._artifacts_intact(job_id, item):
            return None
        return item

    def reusable_items(self, job_id: str) -> Dict[str, Dict[str, Any]]:
        """All finished items whose recorded artifacts are still byte-identical, keyed by item id.

        One pass over the job; use this instead of calling ``reusable_item`` per item on large jobs.
        """

        job = self.get_job(job_id)
        reusable: Dict[str, Dict[str, Any]] = {}
        for item_id in job["item_order"]:
            item = job["items"][item_id]
            if item["status"] != "done":
                continue
            if self.verify_artifacts_on_resume and not self._artifacts_intact(job_id, item):
                continue
            reusable[item_id] = item
        return reusable

    def _artifacts_intact(self, job_id: str, item: Mapping[str, Any]) -> bool:
        for artifact in item["artifacts"]:
            report = self.integrity.verify_artifact(
                artifact["path"], expected_sha256=artifact.get("sha256"), expected_size=artifact.get("size_bytes"))
            if not report["payload"]["valid"]:
                logger.warning("Job %s item %s: artifact %s changed or is missing; it will be redone", job_id, item["item_id"], artifact["path"])
                return False
        return True

    def reset_item(self, job_id: str, item_id: str) -> None:
        job_id = self._normalize_job_id(job_id)
        with self._lock:
            job = self._state(job_id)
            item = self._require_item(job, item_id)
            self._append(job_id, "item", {"item_id": item["item_id"], "status": "pending", "attempt": item["attempt"]})

    def run(
        self,
        job_id: str,
        handler: Callable[[Dict[str, Any]], Optional[Mapping[str, Any]]],
        *,
        retry_failed: bool = False,
        owner: Optional[str] = None,
        force: bool = False,
    ) -> Dict[str, Any]:
        """Drive a job: start (or resume) it and run ``handler`` for each item that still needs work.

        ``handler(item)`` returns ``None`` or a mapping with optional ``result`` and ``artifacts``.
        Raising marks only that item failed. Finished items whose artifacts still verify are reused
        and their handler is not called. Cancellation is honoured between items.
        """

        timer = OperationTimer("run_job")
        job_id = self._normalize_job_id(job_id)
        job = self.get_job(job_id)
        if retry_failed:
            for item in job["items"].values():
                if item["status"] == "failed":
                    self.reset_item(job_id, item["item_id"])
        self.start_job(job_id, owner=owner, force=force)

        reused = executed = failed = 0
        warnings: List[str] = []
        snapshot = self.get_job(job_id)
        for position, item_id in enumerate(snapshot["item_order"]):
            # Cheap in-memory check every item; re-read the journal now and then to see an external cancel.
            if self.is_cancelled(job_id, fresh=(position % 25 == 0 and position > 0)):
                warnings.append("job_cancelled_before_all_items_ran")
                break
            item = self.get_job(job_id)["items"][item_id]
            if item["status"] in {"failed", "skipped"}:
                continue
            if item["status"] == "done":
                if self.reusable_item(job_id, item_id) is not None:
                    reused += 1
                    continue
                self.reset_item(job_id, item_id)
            try:
                started = self.begin_item(job_id, item_id)
                outcome = handler(started) or {}
                self.complete_item(job_id, item_id, outcome.get("result"), artifacts=outcome.get("artifacts"))
                executed += 1
            except JobStateError:
                raise
            except Exception as exc:
                failed += 1
                logger.warning("Job %s item %s failed: %s", job_id, item_id, exc)
                self.fail_item(job_id, item_id, exc)

        final = self.get_job(job_id)
        if final["status"] == "running":
            final = self.finish_job(job_id)
        payload = {"job": final, "executed": executed, "reused": reused, "failed": failed}
        return build_success_result(operation="run_job", payload=payload, metadata={"engine_version": JOB_ENGINE_VERSION},
                                    warnings=warnings, timer=timer)

    # ------------------------------------------------------------------
    # Listing and retention
    # ------------------------------------------------------------------

    def list_jobs(self, *, status: Optional[str] = None, kind: Optional[str] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Summaries of known jobs, newest first. Opens only the creation records plus each job's events."""

        job_ids: Dict[str, float] = {}
        try:
            for entry in self.memory.list_checkpoints(step=_CREATED_STEP, newest_first=True):
                if entry.get("run_id"):
                    job_ids.setdefault(entry["run_id"], float(entry.get("created_at") or 0.0))
        except ReaderError as exc:
            raise JobError("Failed listing Reader jobs", cause=exc) from exc
        summaries: List[Dict[str, Any]] = []
        for job_id in job_ids:
            try:
                job = self.get_job(job_id)
            except JobError as exc:
                logger.warning("Skipping unreadable job %s: %s", job_id, exc)
                continue
            if status and job["status"] != status:
                continue
            if kind and job["kind"] != kind:
                continue
            summaries.append({key: job[key] for key in ("job_id", "kind", "status", "created_at", "updated_at", "counts", "stale", "attempts")})
        if limit is not None:
            summaries = summaries[: max(0, int(limit))]
        return summaries

    def delete_job(self, job_id: str) -> int:
        """Delete a job's journal. Returns the number of journal files removed."""

        job_id = self._normalize_job_id(job_id)
        removed = 0
        with self._lock:
            for entry in self.memory.list_checkpoints(run_id=job_id):
                if self.memory.delete_checkpoint(entry["path"]):
                    removed += 1
            self._jobs.pop(job_id, None)
            self._cancel_requested.discard(job_id)
        return removed

    def prune_jobs(self, *, older_than_seconds: Optional[int] = None, keep: Optional[int] = None) -> Dict[str, Any]:
        """Delete terminal jobs older than ``older_than_seconds`` (default: ``retain_terminal_seconds``)
        and trim the oldest terminal jobs beyond ``keep`` (default: ``max_jobs``). Never touches live jobs."""

        age = self.retain_terminal_seconds if older_than_seconds is None else int(older_than_seconds)
        limit = self.max_jobs if keep is None else int(keep)
        now = utc_timestamp()
        jobs = self.list_jobs()
        terminal = [job for job in jobs if job["status"] in TERMINAL_STATUSES]
        doomed = {job["job_id"] for job in terminal if age > 0 and (now - float(job["updated_at"] or now)) > age}
        overflow = len(jobs) - limit
        if overflow > 0:
            for job in reversed(terminal):
                if overflow <= 0:
                    break
                if job["job_id"] not in doomed:
                    doomed.add(job["job_id"])
                overflow -= 1
        removed = sum(self.delete_job(job_id) > 0 for job_id in doomed)
        return {"jobs_removed": removed, "jobs_remaining": len(jobs) - removed}

    def healthcheck(self) -> Dict[str, Any]:
        store = self.memory.healthcheck()
        return {"status": "ok" if store.get("status", "ok") == "ok" else "degraded", "job_dir": self.job_dir, "store": store}


if __name__ == "__main__":
    print("\n=== Running Job Engine ===\n")
    printer.status("TEST", "Job Engine initialized", "info")

    import tempfile

    with tempfile.TemporaryDirectory(prefix="reader_job_test_") as tmp_dir:
        tmp = Path(tmp_dir)
        memory = ReaderMemory({
            "checkpoint_dir": str(tmp / "jobs"), "cache_dir": str(tmp / "cache"), "audit_dir": str(tmp / "audit"),
            "max_checkpoint_entries": 100_000, "write_indexes": False, "enable_audit_log": False, "checkpoint_prefix": "job",
        })
        engine = JobEngine(memory=memory)

        outputs = []
        for index in range(3):
            path = tmp / f"out{index}.txt"
            path.write_text(f"result {index}", encoding="utf-8")
            outputs.append(path)

        created = engine.create_job("convert", [{"key": str(path)} for path in outputs], params={"target": "txt"}, idempotency_key=["t", 1])
        job_id = created["payload"]["job"]["job_id"]
        assert created["payload"]["created"] is True
        assert engine.create_job("convert", [{"key": "x"}], idempotency_key=["t", 1])["payload"]["created"] is False
        printer.status("TEST", "Idempotent create returns the existing job", "success")

        calls: List[str] = []

        def handler(item: Dict[str, Any]) -> Dict[str, Any]:
            calls.append(item["key"])
            if item["key"].endswith("out1.txt") and len(calls) == 2:
                raise RuntimeError("simulated crash in item")
            return {"result": {"ok": True}, "artifacts": [item["key"]]}

        first = engine.run(job_id, handler)
        assert first["payload"]["job"]["status"] == "partial", first["payload"]["job"]["status"]
        assert first["payload"]["executed"] == 2 and first["payload"]["failed"] == 1
        printer.status("TEST", "One failed item yields a partial job", "success")

        calls.clear()
        second = engine.run(job_id, handler, retry_failed=True)
        assert second["payload"]["job"]["status"] == "completed"
        assert second["payload"]["reused"] == 2 and calls == [str(outputs[1])], calls
        printer.status("TEST", "Resume reuses finished items and redoes only the failed one", "success")

        outputs[0].write_text("tampered", encoding="utf-8")
        assert engine.reusable_item(job_id, engine.get_job(job_id)["item_order"][0]) is None
        printer.status("TEST", "Changed artifact is not reused", "success")

        try:
            engine.start_job(job_id)
        except JobStateError:
            printer.status("TEST", "Completed job cannot be restarted", "success")
        else:
            raise AssertionError("Expected JobStateError")

        try:
            engine.get_job("missing_job")
        except JobNotFoundError:
            printer.status("TEST", "Unknown job raises JobNotFoundError", "success")
        else:
            raise AssertionError("Expected JobNotFoundError")

        assert engine.list_jobs(status="completed")[0]["job_id"] == job_id
        assert engine.delete_job(job_id) > 0

    print("\n=== Test ran successfully ===\n")
