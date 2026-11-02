"""Application-neutral contract for authoritative project revisions and commits.

Host applications own revision tokens, mutation validation and persistence. These
adapters only transport explicit requests and receipts; they never fabricate a
successful commit or infer authority from an agent-generated payload.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Protocol, runtime_checkable


@dataclass(frozen=True)
class HostCommitRequest:
    project_id: str
    artifact_id: str
    expected_revision: str
    proposal_ref: str
    idempotency_key: str

    def __post_init__(self) -> None:
        for name in ("project_id", "artifact_id", "expected_revision", "proposal_ref", "idempotency_key"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty string")


@dataclass(frozen=True)
class HostCommitReceipt:
    """Host-confirmed outcome; an unknown outcome has no claimed revision."""

    project_id: str
    artifact_id: str
    idempotency_key: str
    status: str  # confirmed, rejected, unknown
    committed_revision: Optional[str] = None
    evidence_id: Optional[str] = None
    reason: Optional[str] = None

    def __post_init__(self) -> None:
        if self.status not in {"confirmed", "rejected", "unknown"}:
            raise ValueError("invalid authoritative commit status")
        if self.status == "confirmed" and (not self.committed_revision or not self.evidence_id):
            raise ValueError("confirmed receipts require a revision and host evidence ID")
        if self.status != "confirmed" and self.committed_revision is not None:
            raise ValueError("unconfirmed receipts cannot assert a committed revision")


@runtime_checkable
class AuthoritativeProjectAdapter(Protocol):
    """The host MUST provide atomic revision checking and idempotent commits.

    lookup_commit must consult authoritative persisted host state (not an
    in-memory guess). Unknown outcomes are not automatically retried.
    """

    def current_revisions(self, project_id: str) -> Mapping[str, str]: ...

    def commit(self, request: HostCommitRequest) -> HostCommitReceipt: ...

    def lookup_commit(self, project_id: str, idempotency_key: str) -> Optional[HostCommitReceipt]: ...


class CallbackProjectAdapter:
    """Concrete adapter for applications with existing revision/commit APIs.

    Application code injects real callbacks. No generic in-memory mutation is
    used as a substitute for the application's authoritative transaction.
    """

    def __init__(self, *, revisions: Callable[[str], Mapping[str, str]],
                 commit: Callable[[HostCommitRequest], HostCommitReceipt],
                 lookup: Callable[[str, str], Optional[HostCommitReceipt]]) -> None:
        if not all(callable(method) for method in (revisions, commit, lookup)):
            raise TypeError("all three authoritative callbacks are required")
        self._revisions = revisions
        self._commit = commit
        self._lookup = lookup

    def current_revisions(self, project_id: str) -> Mapping[str, str]:
        return dict(self._revisions(project_id))

    def commit(self, request: HostCommitRequest) -> HostCommitReceipt:
        return self._commit(request)

    def lookup_commit(self, project_id: str, idempotency_key: str) -> Optional[HostCommitReceipt]:
        return self._lookup(project_id, idempotency_key)


__all__ = ["HostCommitRequest", "HostCommitReceipt", "AuthoritativeProjectAdapter",
           "CallbackProjectAdapter"]
