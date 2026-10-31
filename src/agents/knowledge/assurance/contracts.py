"""Typed, JSON-compatible contracts for evidence-bearing SLAI knowledge.

These objects are proposed v1 contracts; they are NOT existing SLAI v2.4 APIs.
All times use UTC ISO-8601 strings, e.g. ``2026-10-31T12:00:00Z``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping
import datetime as dt
import hashlib
import json
import re


class KnowledgeAssuranceError(Exception):
    """Base error for the canonical knowledge service."""


class AuthorizationDenied(KnowledgeAssuranceError):
    pass


class ValidationFailure(KnowledgeAssuranceError):
    pass


class RevisionConflict(KnowledgeAssuranceError):
    pass


class NotFound(KnowledgeAssuranceError):
    pass


class UnsupportedEvaluation(KnowledgeAssuranceError):
    pass


class MissingInput(KnowledgeAssuranceError):
    def __init__(self, fields: set[str]):
        self.fields = frozenset(fields)
        super().__init__(", ".join(sorted(fields)))


class Scope(str, Enum):
    GLOBAL = "global"
    ORGANIZATION = "organization"
    PROJECT = "project"


class Lifecycle(str, Enum):
    DRAFT = "draft"
    REVIEWED = "reviewed"
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    INVALIDATED = "invalidated"
    ARCHIVED = "archived"


class RuleOutcome(str, Enum):
    SATISFIED = "satisfied"
    VIOLATED = "violated"
    NOT_APPLICABLE = "not_applicable"
    UNDETERMINED = "undetermined"
    EVALUATION_ERROR = "evaluation_error"
    UNSUPPORTED = "unsupported_evaluation_method"


IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}$")
KIND = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


def validate_id(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
        raise ValidationFailure(f"Invalid {field_name}: required nonempty safe identifier (max 128 characters)")
    return value


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def normalize_time(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValidationFailure(f"{field_name} must be an ISO-8601 UTC timestamp")
    try:
        x = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValidationFailure(f"Invalid {field_name}") from exc
    if x.tzinfo is None or x.utcoffset() != dt.timedelta(0):
        raise ValidationFailure(f"{field_name} must specify UTC")
    return x.astimezone(dt.timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def json_object(value: Any, field_name: str, max_bytes: int = 524288) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(k, str) for k in value):
        raise ValidationFailure(f"{field_name} must be an object with string keys")
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                             allow_nan=False, default=lambda _: (_ for _ in ()).throw(TypeError()))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationFailure(f"{field_name} must contain only finite JSON-compatible values") from exc
    if len(encoded.encode("utf-8")) > max_bytes:
        raise ValidationFailure(f"{field_name} exceeds {max_bytes} bytes")
    return json.loads(encoded)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class AccessContext:
    """TRUSTED context created by an authenticated application boundary.

    A caller must never hydrate this object directly from an untrusted JSON request.
    Membership/roles are authoritative only if verified by the caller's auth service.
    """
    principal_id: str
    project_ids: frozenset[str] = field(default_factory=frozenset)
    organization_ids: frozenset[str] = field(default_factory=frozenset)
    can_write: bool = False
    can_publish_global: bool = False
    can_review: bool = False
    can_attest: bool = False

    def __post_init__(self) -> None:
        validate_id(self.principal_id, "principal_id")
        for x in self.project_ids:
            validate_id(x, "project_id")
        for x in self.organization_ids:
            validate_id(x, "organization_id")


@dataclass(frozen=True, slots=True)
class RecordDraft:
    kind: str
    scope: Scope
    payload: dict[str, Any]
    project_id: str | None = None
    organization_id: str | None = None
    status: Lifecycle = Lifecycle.DRAFT
    source_ids: tuple[str, ...] = ()
    evidence: tuple[dict[str, Any], ...] = ()
    valid_from: str | None = None
    valid_until: str | None = None

    def __post_init__(self) -> None:
        if not KIND.fullmatch(self.kind):
            raise ValidationFailure("kind must be a lowercase snake_case identifier")
        if not isinstance(self.scope, Scope):
            raise ValidationFailure("scope must be a Scope enumeration")
        if self.scope == Scope.PROJECT:
            validate_id(self.project_id, "project_id")  # type: ignore[arg-type]
            if self.organization_id is not None:
                validate_id(self.organization_id, "organization_id")
        elif self.scope == Scope.ORGANIZATION:
            validate_id(self.organization_id, "organization_id")  # type: ignore[arg-type]
            if self.project_id is not None:
                raise ValidationFailure("organization record cannot identify a project")
        elif self.project_id is not None or self.organization_id is not None:
            raise ValidationFailure("global record cannot carry a project/organization owner")
        if not isinstance(self.status, Lifecycle):
            raise ValidationFailure("status must be a Lifecycle enumeration")
        json_object(self.payload, "payload")
        for ref in self.source_ids:
            validate_id(ref, "source_id")
        for att in self.evidence:
            json_object(att, "evidence entry", 16384)
        start = normalize_time(self.valid_from, "valid_from")
        end = normalize_time(self.valid_until, "valid_until")
        if start and end and start >= end:
            raise ValidationFailure("valid_from must precede valid_until")


@dataclass(frozen=True, slots=True)
class RecordRevision:
    record_id: str
    revision: int
    draft: RecordDraft
    recorded_at: str
    actor_id: str
    reason: str
    digest: str


@dataclass(frozen=True, slots=True)
class RuleEvaluation:
    rule_id: str
    rule_revision: int
    outcome: RuleOutcome
    evaluator: str
    message: str
    missing_inputs: tuple[str, ...] = ()
    input_digest: str = ""
    evaluated_at: str = ""
    evidence: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class Applicability:
    outcome: str  # applicable / not_applicable / undetermined
    reasons: tuple[str, ...]
    assumptions: tuple[str, ...] = ()
