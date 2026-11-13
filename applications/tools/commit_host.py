"""The host side of SLAI's revision-aware project workflows, shared by every application.

``src/agents/planning/project_adapter.py`` defines what an application must offer an automated workflow: current revisions, a commit that
is checked against an expected revision, and a lookup of earlier commits by idempotency key. It deliberately supplies no implementation,
because only the application knows its documents. :class:`CommitHost` is the part that is the same everywhere and easy to get wrong:

* **one critical section** around "is the artifact still at the revision the proposal was made for" and "apply the change", so no edit
  can slip in between (the application's own lock, passed in);
* **the change and its record together**: the change and the ledger entry that says it happened run inside the application's own
  transaction (``atomic``), so a crash cannot leave a change applied but unrecorded, or recorded but not applied;
* **idempotency**: a key that was already decided returns the decision it got, byte for byte, and is never applied twice; a key reused for a
  different request is refused;
* **receipts that say only what happened**: ``confirmed`` carries the revision the application reports after the change, ``rejected``
  carries the application's own reason, and a failure nobody can classify is *not* turned into either: it propagates, and the outbox
  (``CommitReconciliationStore``) then records the outcome as unknown and asks :meth:`lookup_commit` instead of retrying.

What stays with the application is everything that makes a commit real: which revisions exist (``revisions``), whether the proposal
exists and has the person's consent, and the validated mutation itself (``apply``). The kit invents none of it. The ledger of decisions
belongs to the application's own persistence (SPADRA keeps it inside the project so it is saved with it); :class:`FileCommitLedger` is for
applications that keep their data elsewhere.

The planning types are imported when the first receipt is made, so an application that never uses workflows does not need them.
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib
import json
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, ContextManager, Mapping, Protocol


from .errors import AppError, DependencyMissing
from .storage import atomic_write_json

ADAPTER_MODULE = "src.agents.planning.project_adapter"


def planning_types() -> Any:
    """The module with ``HostCommitRequest`` / ``HostCommitReceipt``: SLAI's, never a copy."""
    try:
        return importlib.import_module(ADAPTER_MODULE)
    except ImportError as exc:
        raise DependencyMissing("SLAI's commit adapters (src/agents/planning/project_adapter.py) are not in this checkout.",
                                hint="Use a SLAI checkout that includes the planning commit adapters.", details={"import_error": str(exc)}) from exc


@dataclass(frozen=True)
class Applied:
    """What the application reports after it applied a change: the artifact's new revision and an identifier a person can find again."""

    revision: str
    evidence_id: str


class CommitRefused(Exception):
    """The application declines this commit (stale revision, missing consent, a change its validators refuse). ``reason`` is shown as is."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class CommitOutcomeUnknown(AppError):
    """A commit was sent to the host and its result could not be established. Nothing is retried; the commit is looked up by its key."""

    status, code = 502, "commit_outcome_unknown"


class CommitLedger(Protocol):
    """Where decided commits are kept. ``put`` must be durable together with the change it records, or the application must say that it is not."""

    def get(self, key: str) -> dict[str, Any] | None: ...

    def put(self, key: str, entry: dict[str, Any]) -> None: ...


class FileCommitLedger:
    """A JSON file of decisions, for applications that keep no document of their own to put them in."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._entries: dict[str, dict[str, Any]] = json.loads(self.path.read_text(encoding="utf-8")) if self.path.is_file() else {}

    def get(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            entry = self._entries.get(key)
            return json.loads(json.dumps(entry)) if entry is not None else None

    def put(self, key: str, entry: dict[str, Any]) -> None:
        with self._lock:
            updated = {**self._entries, key: entry}
            atomic_write_json(self.path, updated)                    # on disk first: memory never claims what the disk does not hold
            self._entries = updated


def fingerprint(request: Any) -> str:
    return hashlib.sha256(json.dumps(asdict(request), sort_keys=True).encode("utf-8")).hexdigest()


class CommitHost:
    """Implements ``AuthoritativeProjectAdapter`` for an application. See the module docstring for what is and is not its job."""

    def __init__(self, *, revisions: Callable[[str], Mapping[str, str]], apply: Callable[[Any], Applied], ledger: CommitLedger,
                 lock: ContextManager[Any] | None = None, atomic: Callable[[], ContextManager[Any]] | None = None) -> None:
        self._revisions, self._apply, self._ledger = revisions, apply, ledger
        self._lock: ContextManager[Any] = lock or threading.RLock()
        self._atomic: Callable[[], ContextManager[Any]] = atomic or contextlib.nullcontext

    # ------------------------------------------------------------------------------------------------------ the adapter protocol
    def current_revisions(self, project_id: str) -> Mapping[str, str]:
        with self._lock:
            return dict(self._revisions(project_id))

    def commit(self, request: Any) -> Any:
        types = planning_types()
        with self._lock:
            prior = self._ledger.get(request.idempotency_key)
            if prior is not None:
                if prior["fingerprint"] != fingerprint(request):
                    return self._receipt(types, request, "rejected", reason="The idempotency key was already used for a different commit request.")
                return types.HostCommitReceipt(**prior["receipt"])
            current = dict(self._revisions(request.project_id)).get(request.artifact_id)
            if current is None:                                      # not something this application holds, so there is nothing to record the decision in
                return self._receipt(types, request, "rejected", reason=f"There is no artifact {request.artifact_id!r} in project {request.project_id!r}.")
            if current != request.expected_revision:
                return self._decide(types, request, "rejected",
                                    reason=f"Stale: the proposal was made for revision {request.expected_revision}, but the artifact is at revision {current}.")
            try:
                with self._atomic():                                 # a refusal leaves the block by exception, which rolls the change back
                    applied = self._apply(request)
                    return self._decide(types, request, "confirmed", revision=applied.revision, evidence=applied.evidence_id)
            except CommitRefused as refusal:
                reason = refusal.reason
            except AppError as error:                                # the application's own validation said no; nothing was applied
                reason = error.message
            with self._atomic():
                return self._decide(types, request, "rejected", reason=reason)

    def lookup_commit(self, project_id: str, idempotency_key: str) -> Any | None:
        types = planning_types()
        with self._lock:
            entry = self._ledger.get(idempotency_key)
            if entry is None or entry["receipt"]["project_id"] != project_id:
                return None
            return types.HostCommitReceipt(**entry["receipt"])

    # ------------------------------------------------------------------------------------------------------------------ internals
    @staticmethod
    def _receipt(types: Any, request: Any, status: str, *, revision: str | None = None, evidence: str | None = None, reason: str | None = None) -> Any:
        return types.HostCommitReceipt(project_id=request.project_id, artifact_id=request.artifact_id, idempotency_key=request.idempotency_key,
                                       status=status, committed_revision=revision, evidence_id=evidence, reason=reason)

    def _decide(self, types: Any, request: Any, status: str, **fields: Any) -> Any:
        receipt = self._receipt(types, request, status, **fields)
        self._ledger.put(request.idempotency_key, {"fingerprint": fingerprint(request), "receipt": asdict(receipt)})
        return receipt


class Outbox:
    """SLAI's durable ``CommitReconciliationStore`` over an application's host, plus the one sequence every approval follows: persist the
    intent, send it once, and settle an uncertain outcome by lookup. The store is created on first use, in ``path``."""

    SCOPES = frozenset({"workflow:commit", "workflow:commit:reconcile"})
    AUTHORIZATION_SECONDS = 60

    def __init__(self, path: Path | str, host: CommitHost) -> None:
        self.path, self.host = Path(path), host
        self._store: Any = None
        self._lock = threading.Lock()

    def store(self) -> Any:
        with self._lock:
            if self._store is None:
                planning_types()                                      # one clear message if the adapters are not in this checkout
                module = importlib.import_module("src.agents.planning.commit_reconciliation")
                self._store = module.CommitReconciliationStore(self.path, self.host)
            return self._store

    def close(self) -> None:
        with self._lock:
            if self._store is not None:
                self._store.close()
                self._store = None

    def authorization(self, project_id: str, actor: str) -> Any:
        """What a person's approval entitles a commit to. Call it only with an approval the server itself received."""
        runtime = importlib.import_module("src.agents.planning.workflow_runtime")
        return runtime.AuthorizationContext(actor_id=actor, project_id=project_id, scopes=self.SCOPES, expires_at=time.time() + self.AUTHORIZATION_SECONDS)

    def commit(self, request: Any, *, actor: str) -> dict[str, Any]:
        """Prepare, dispatch once and reconcile if uncertain. Returns the settled record (``confirmed`` or ``rejected``); raises
        :class:`CommitOutcomeUnknown` when the host cannot say, and does not send the request again."""
        auth, store = self.authorization(request.project_id, actor), self.store()
        store.prepare(request, auth=auth)
        record = store.dispatch(request.idempotency_key, auth=auth)
        if record["status"] == "outcome_unknown":
            record = store.reconcile(request.idempotency_key, auth=auth)
        if record["status"] == "outcome_unknown":
            raise CommitOutcomeUnknown("The application could not establish whether the change was applied.",
                                       hint="Check the item's current state; the change will not be sent again.", details={"idempotency_key": request.idempotency_key})
        return record


__all__ = ["Applied", "CommitHost", "CommitOutcomeUnknown", "Outbox", "CommitLedger", "CommitRefused", "FileCommitLedger", "fingerprint", "planning_types"]
