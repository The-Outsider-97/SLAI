"""Trusted provenance for data-classification assertions (GOV-P0-01).

Why this exists
---------------
``ProcessingContext.classification_verified`` was a boolean any caller could set. A claim that data is non-personal is the
single most valuable lie in a GDPR system, so the claim is now *evidence*: a signed, expiring, scope-bound record issued by
this authority after a classifier actually looked at the data.

Four origins are distinguished and only two of them can ever relax a decision:

==============  =====================================================  ===============
origin          meaning                                                can relax?
==============  =====================================================  ===============
DECLARED        the caller says so (no evidence at all)                 never
INFERRED        a classifier looked, but heuristically / low confidence escalate only
VERIFIED        deterministic classifier, confidence >= threshold       yes (bounded)
AUTHORITATIVE   attested by an independent, role-holding human/agent    yes (bounded)
==============  =====================================================  ===============

Evidence is bound to tenant, purpose, principal, resource and a *keyed digest of the data itself*. Change the data, purpose,
tenant or caller and the evidence stops applying (``scope_mismatch``); re-classifying a resource supersedes earlier evidence
(``superseded``); single-use evidence is consumed atomically (``consumed``). The Privacy Agent stays the only detector:
``PrivacyClassifierAdapter`` wraps its ``DataID`` and adds no detection logic here.

Stdlib only (like every mandatory control in this package).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Protocol, Sequence, Tuple, Union

from .processing_context import DataClass, IdentityContext, ProcessingContext, class_strictness, most_restrictive_class, new_id
from .utils.store import open_db

__all__ = [
    "ClassificationAuthority",
    "ClassificationError",
    "ClassificationEvidence",
    "ClassifierResult",
    "ClassifierUnavailable",
    "EvidenceOrigin",
    "PrivacyClassifierAdapter",
    "Resolution",
]


class ClassificationError(PermissionError):
    """``code``: missing|malformed|forged|expired|unavailable|superseded|revoked|consumed|scope_mismatch|unknown|low_confidence|not_authorised"""

    def __init__(self, code: str, message: str = ""):
        self.code = code
        super().__init__(message or code)


class ClassifierUnavailable(RuntimeError):
    """The classifier could not run. The caller gets no evidence, which fails closed (UNCLASSIFIED => quarantine)."""


class EvidenceOrigin(str, Enum):
    DECLARED = "declared"
    INFERRED = "inferred"
    VERIFIED = "verified"
    AUTHORITATIVE = "authoritative"


_RELAXING = frozenset({EvidenceOrigin.VERIFIED, EvidenceOrigin.AUTHORITATIVE})
_NON_PERSONAL = frozenset({DataClass.NON_PERSONAL, DataClass.ANONYMISED})


@dataclass(frozen=True)
class ClassifierResult:
    """What a classifier reports. ``structural`` is the claim "provably non-personal by structure" (numbers/booleans only)."""

    data_class: DataClass
    categories: Tuple[DataClass, ...] = ()
    confidence: float = 0.0
    classifier_id: str = "unknown"
    classifier_version: str = "0"
    deterministic: bool = True
    structural: bool = False
    findings: Mapping[str, int] = field(default_factory=dict)


class Classifier(Protocol):
    def __call__(self, payload: Any, *, purpose: str) -> ClassifierResult: ...


@dataclass(frozen=True)
class ClassificationEvidence:
    evidence_id: str
    origin: EvidenceOrigin
    data_class: DataClass
    categories: Tuple[DataClass, ...]
    confidence: float
    classifier_id: str
    classifier_version: str
    tenant_id: Optional[str]
    principal_id: Optional[str]
    purpose: str
    resource: Optional[str]
    data_digest: Optional[str]
    request_id: Optional[str]
    single_use: bool
    issued_at: float
    expires_at: float
    attested_by: Optional[str] = None

    def to_payload(self) -> Dict[str, Any]:
        return {
            "eid": self.evidence_id, "org": self.origin.value, "cls": self.data_class.value,
            "cats": [c.value for c in self.categories], "conf": self.confidence, "cid": self.classifier_id,
            "cver": self.classifier_version, "ten": self.tenant_id, "pri": self.principal_id, "pur": self.purpose,
            "res": self.resource, "dig": self.data_digest, "req": self.request_id, "su": self.single_use,
            "iat": self.issued_at, "exp": self.expires_at, "att": self.attested_by, "v": 1,
        }

    @classmethod
    def from_payload(cls, d: Mapping[str, Any]) -> "ClassificationEvidence":
        return cls(
            evidence_id=str(d["eid"]), origin=EvidenceOrigin(d["org"]), data_class=DataClass(d["cls"]),
            categories=tuple(DataClass(c) for c in d.get("cats", ())), confidence=float(d["conf"]),
            classifier_id=str(d["cid"]), classifier_version=str(d["cver"]), tenant_id=d.get("ten"), principal_id=d.get("pri"),
            purpose=str(d["pur"]), resource=d.get("res"), data_digest=d.get("dig"), request_id=d.get("req"),
            single_use=bool(d.get("su", False)), issued_at=float(d["iat"]), expires_at=float(d["exp"]), attested_by=d.get("att"),
        )


@dataclass(frozen=True)
class Resolution:
    """The class governance may act on, and why. ``status`` is recorded in the decision (explainability)."""

    primary: DataClass
    classes: Tuple[DataClass, ...]
    status: str
    evidence_id: Optional[str] = None
    origin: Optional[EvidenceOrigin] = None


def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64d(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


_SCHEMA = """
CREATE TABLE IF NOT EXISTS evidence (
    evidence_id TEXT PRIMARY KEY, status TEXT NOT NULL, tenant_id TEXT, resource TEXT, data_digest TEXT,
    single_use INTEGER NOT NULL, issued_at REAL NOT NULL, expires_at REAL NOT NULL, status_reason TEXT, updated_at REAL NOT NULL);
CREATE INDEX IF NOT EXISTS evidence_resource ON evidence(tenant_id, resource, status);
"""


class ClassificationAuthority:
    """Issues and verifies classification evidence. Without a signing key it is unavailable and nothing verifies."""

    def __init__(
        self,
        key: Optional[bytes] = None,
        *,
        path: Union[str, Path] = ":memory:",
        classifier: Optional[Classifier] = None,
        min_confidence: float = 0.85,
        ttl: float = 900.0,
        attestation_roles: Iterable[str] = ("dpo", "governance_admin", "classification_authority"),
        clock: Callable[[], float] = time.time,
    ):
        self._key = key if key and len(key) >= 32 else None
        self._digest_key = hmac.new(self._key, b"slai-gov-v1:classification-digest", hashlib.sha256).digest() if self._key else None
        self._classifier = classifier
        self.min_confidence = float(min_confidence)
        self.ttl = float(ttl)
        self.attestation_roles = frozenset(attestation_roles)
        self._clock = clock
        self._lock = RLock()
        self._db = open_db(path)
        self._db.executescript(_SCHEMA)
        self.stats = {"issued": 0, "verified": 0, "rejected": 0, "classifier_unavailable": 0}
        self.on_change: Optional[Callable[[str, str, str], None]] = None  # set by the core

    # ------------------------------------------------------------------ properties
    @property
    def available(self) -> bool:
        return self._key is not None

    @property
    def classifier_attached(self) -> bool:
        return self._classifier is not None

    def attach_classifier(self, classifier: Optional[Classifier]) -> None:
        self._classifier = classifier

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # ------------------------------------------------------------------ data binding
    def digest(self, payload: Any) -> str:
        """Keyed digest of the canonical payload: binds evidence to this data without making it guessable."""
        if self._digest_key is None:
            raise ClassificationError("unavailable", "classification key not configured")
        blob = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"), ensure_ascii=False)
        return hmac.new(self._digest_key, blob.encode("utf-8"), hashlib.sha256).hexdigest()

    # ------------------------------------------------------------------ issuing
    def classify(
        self,
        payload: Any,
        *,
        purpose: str,
        tenant_id: Optional[str],
        identity: Optional[IdentityContext] = None,
        resource: Optional[str] = None,
        request_id: Optional[str] = None,
        single_use: bool = False,
        ttl: Optional[float] = None,
    ) -> str:
        """Run the Privacy Agent's classifier on ``payload`` and issue evidence bound to this exact scope.

        Raises ``ClassifierUnavailable`` when no classifier is attached or it fails: the caller then holds no evidence and
        governance quarantines (it never guesses).
        """
        if not self.available:
            raise ClassificationError("unavailable", "classification key not configured")
        if self._classifier is None:
            with self._lock:
                self.stats["classifier_unavailable"] += 1
            raise ClassifierUnavailable("no classifier attached")
        try:
            result = self._classifier(payload, purpose=purpose)
        except ClassifierUnavailable:
            with self._lock:
                self.stats["classifier_unavailable"] += 1
            raise
        except Exception as exc:  # a crashing classifier is an unavailable classifier, never a verdict
            with self._lock:
                self.stats["classifier_unavailable"] += 1
            raise ClassifierUnavailable(f"classifier failed: {type(exc).__name__}") from exc

        data_class = result.data_class
        origin = EvidenceOrigin.VERIFIED if (result.deterministic and result.confidence >= self.min_confidence) else EvidenceOrigin.INFERRED
        if data_class in _NON_PERSONAL and not result.structural:
            # Absence of detections in free text is not proof of absence of personal data.
            data_class, origin = DataClass.UNCLASSIFIED, EvidenceOrigin.INFERRED
        cats = tuple(dict.fromkeys(result.categories or (data_class,)))
        return self._issue(
            origin=origin, data_class=data_class, categories=cats, confidence=result.confidence,
            classifier_id=result.classifier_id, classifier_version=result.classifier_version, tenant_id=tenant_id,
            principal_id=identity.principal_id if identity else None, purpose=purpose, resource=resource,
            data_digest=self.digest(payload), request_id=request_id, single_use=single_use, ttl=ttl, attested_by=None,
        )

    def attest(
        self,
        attester: IdentityContext,
        *,
        data_class: DataClass,
        data_digest: str,
        purpose: str,
        tenant_id: Optional[str],
        rationale: str,
        resource: Optional[str] = None,
        categories: Sequence[DataClass] = (),
        ttl: Optional[float] = None,
    ) -> str:
        """Authoritative evidence from a verified role-holder (e.g. the DPO). The attester must have been verified by the
        IdentityAuthority; a hand-built ``IdentityContext`` is rejected by the caller (``GovernanceCore.attest_classification``)."""
        if not self.available:
            raise ClassificationError("unavailable", "classification key not configured")
        if not attester.is_valid(self._clock()) or not (self.attestation_roles & set(attester.roles)):
            raise ClassificationError("not_authorised", "attester lacks a classification-attestation role")
        if len(rationale.strip()) < 15:
            raise ClassificationError("malformed", "attestation needs a rationale")
        if not data_digest:
            raise ClassificationError("malformed", "attestation must be bound to a data digest")
        if tenant_id and attester.tenant_id not in (None, tenant_id):
            raise ClassificationError("scope_mismatch", "attester belongs to another tenant")
        return self._issue(
            origin=EvidenceOrigin.AUTHORITATIVE, data_class=data_class, categories=tuple(categories) or (data_class,), confidence=1.0,
            classifier_id=f"attestation:{attester.principal_id}", classifier_version="1", tenant_id=tenant_id, principal_id=None,
            purpose=purpose, resource=resource, data_digest=data_digest, request_id=None, single_use=False, ttl=ttl,
            attested_by=attester.principal_id,
        )

    def _issue(self, *, origin, data_class, categories, confidence, classifier_id, classifier_version, tenant_id, principal_id, purpose,
               resource, data_digest, request_id, single_use, ttl, attested_by) -> str:
        now = self._clock()
        if data_class in _NON_PERSONAL and not data_digest:
            raise ClassificationError("malformed", "a non-personal conclusion must be bound to the classified data")
        ev = ClassificationEvidence(
            evidence_id=new_id("cls"), origin=origin, data_class=data_class, categories=categories, confidence=float(confidence),
            classifier_id=classifier_id, classifier_version=classifier_version, tenant_id=tenant_id, principal_id=principal_id,
            purpose=purpose, resource=resource, data_digest=data_digest, request_id=request_id, single_use=bool(single_use),
            issued_at=now, expires_at=now + min(float(ttl if ttl is not None else self.ttl), self.ttl), attested_by=attested_by,
        )
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                superseded = ()
                if resource:
                    superseded = self.evidence_ids_for_resource(tenant_id, resource)
                    # Reclassification: new evidence for the same resource supersedes every earlier, different one.
                    self._db.execute(
                        "UPDATE evidence SET status='superseded', status_reason='reclassified', updated_at=? "
                        "WHERE status='active' AND tenant_id IS ? AND resource=? AND (data_digest IS NOT ? OR ?<>'')",
                        (now, tenant_id, resource, data_digest, ""),
                    )
                self._db.execute(
                    "INSERT INTO evidence(evidence_id,status,tenant_id,resource,data_digest,single_use,issued_at,expires_at,updated_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?)",
                    (ev.evidence_id, "active", tenant_id, resource, data_digest, int(single_use), ev.issued_at, ev.expires_at, now),
                )
                self._db.execute("COMMIT")
            except Exception:
                self._db.execute("ROLLBACK")
                raise
            self.stats["issued"] += 1
        if self.on_change:
            for eid in superseded:
                if eid != ev.evidence_id:
                    self.on_change("classification_superseded", f"evidence:{eid}", "reclassified")
        body = _b64e(json.dumps(ev.to_payload(), sort_keys=True, separators=(",", ":")).encode("utf-8"))
        return f"{body}.{self._sign(body)}"

    # ------------------------------------------------------------------ verification
    def verify(self, token: Optional[str], ctx: Optional[ProcessingContext] = None, *, now: Optional[float] = None) -> ClassificationEvidence:
        try:
            ev = self._verify(token, ctx, self._clock() if now is None else now)
        except ClassificationError:
            with self._lock:
                self.stats["rejected"] += 1
            raise
        with self._lock:
            self.stats["verified"] += 1
        return ev

    def _verify(self, token: Optional[str], ctx: Optional[ProcessingContext], now: float) -> ClassificationEvidence:
        if not token:
            raise ClassificationError("missing")
        if not self.available:
            raise ClassificationError("unavailable")
        try:
            body, sig = token.split(".", 1)
        except ValueError as exc:
            raise ClassificationError("malformed") from exc
        if not hmac.compare_digest(sig, self._sign(body)):
            raise ClassificationError("forged")
        try:
            ev = ClassificationEvidence.from_payload(json.loads(_b64d(body)))
        except (ValueError, KeyError, TypeError) as exc:
            raise ClassificationError("malformed") from exc
        if now >= ev.expires_at:
            raise ClassificationError("expired")
        with self._lock:
            row = self._db.execute("SELECT status FROM evidence WHERE evidence_id=?", (ev.evidence_id,)).fetchone()
        if row is None:
            raise ClassificationError("unknown", "evidence was not issued by this authority instance")
        if row[0] != "active":
            raise ClassificationError(row[0])  # superseded | revoked | consumed
        if ctx is not None:
            self._check_scope(ev, ctx)
        return ev

    @staticmethod
    def _check_scope(ev: ClassificationEvidence, ctx: ProcessingContext) -> None:
        if ev.tenant_id != ctx.tenant_id:
            raise ClassificationError("scope_mismatch", "tenant")
        if ev.purpose != ctx.purpose:
            raise ClassificationError("scope_mismatch", "purpose")
        if ev.principal_id is not None and (ctx.identity is None or ctx.identity.principal_id != ev.principal_id):
            raise ClassificationError("scope_mismatch", "principal")
        if ev.resource is not None and ev.resource != ctx.resource:
            raise ClassificationError("scope_mismatch", "resource")
        if ev.request_id is not None and ev.request_id != ctx.request_id:
            raise ClassificationError("scope_mismatch", "request")
        if ev.data_digest is not None and ev.data_digest != ctx.data_digest:
            raise ClassificationError("scope_mismatch", "data")
        if ev.data_digest is None and ev.data_class in _NON_PERSONAL:
            raise ClassificationError("scope_mismatch", "non-personal evidence without data binding")

    # ------------------------------------------------------------------ resolution (what the engine uses)
    def resolve(self, ctx: ProcessingContext, *, now: Optional[float] = None) -> Resolution:
        """Derive the class governance acts on from the declaration *and* the evidence. Never raises."""
        declared = ctx.data_class
        base = DataClass.UNCLASSIFIED if declared in _NON_PERSONAL else declared
        token = ctx.classification_evidence
        if not token:
            return Resolution(base, (base,), "declared_only" if declared not in _NON_PERSONAL else "unverified_non_personal_claim")
        try:
            ev = self.verify(token, ctx, now=now)
        except ClassificationError as exc:
            # A rejected token is NOT ignored silently: it is reported, and the declaration is treated as unverified.
            return Resolution(base, (base,), f"evidence_rejected:{exc.code}")
        classes = tuple(dict.fromkeys(ev.categories or (ev.data_class,)))
        if ev.origin in _RELAXING and ev.confidence >= self.min_confidence:
            # Bounded relaxation: never below what the caller itself declared as more sensitive.
            primary = most_restrictive_class(ev.data_class, *(c for c in classes), *( [declared] if declared not in _NON_PERSONAL else []))
            merged = tuple(dict.fromkeys(classes + ((declared,) if declared not in _NON_PERSONAL else ())))
            return Resolution(primary, merged, f"evidence_ok:{ev.origin.value}", ev.evidence_id, ev.origin)
        # DECLARED / INFERRED / low-confidence: may only escalate.
        stricter = [c for c in classes if class_strictness(c) > class_strictness(base)]
        if stricter:
            primary = most_restrictive_class(base, *stricter)
            return Resolution(primary, tuple(dict.fromkeys((base,) + tuple(stricter))), f"evidence_escalated:{ev.origin.value}", ev.evidence_id, ev.origin)
        return Resolution(base, (base,), f"evidence_not_relaxing:{ev.origin.value}", ev.evidence_id, ev.origin)

    def evidence_id_in_use(self, ctx: ProcessingContext) -> Optional[str]:
        """Id of the evidence that actually shaped ``ctx``'s resolution (None when none did)."""
        status = ctx.classification_status
        if not status.startswith("evidence_ok") and not status.startswith("evidence_escalated"):
            return None
        from .processing_context import evidence_id_of

        return evidence_id_of(ctx.classification_evidence)

    # ------------------------------------------------------------------ lifecycle
    def consume(self, evidence_id: str, *, now: Optional[float] = None) -> bool:
        """Atomically consume single-use evidence (a no-op for reusable evidence). False: not active any more (consumed / superseded)."""
        t = self._clock() if now is None else now
        with self._lock:
            row = self._db.execute("SELECT single_use, status FROM evidence WHERE evidence_id=?", (evidence_id,)).fetchone()
            if row is None or row[1] != "active":
                return False
            if not row[0]:
                return True  # reusable evidence is never consumed
            cur = self._db.execute(
                "UPDATE evidence SET status='consumed', updated_at=? WHERE evidence_id=? AND status='active' AND single_use=1", (t, evidence_id)
            )
            return cur.rowcount == 1

    def revoke(self, evidence_id: str, reason: str) -> bool:
        if not reason.strip():
            raise ValueError("reason required")
        with self._lock:
            cur = self._db.execute(
                "UPDATE evidence SET status='revoked', status_reason=?, updated_at=? WHERE evidence_id=? AND status='active'",
                (reason, self._clock(), evidence_id),
            )
        if cur.rowcount == 1 and self.on_change:
            self.on_change("classification_revoked", f"evidence:{evidence_id}", reason)
        return cur.rowcount == 1

    def invalidate_resource(self, tenant_id: Optional[str], resource: str, reason: str) -> int:
        """Material change to the data or its processing conditions: every active evidence for the resource stops applying."""
        with self._lock:
            ids = self.evidence_ids_for_resource(tenant_id, resource)
            count = self._db.execute(
                "UPDATE evidence SET status='superseded', status_reason=?, updated_at=? WHERE status='active' AND tenant_id IS ? AND resource=?",
                (reason, self._clock(), tenant_id, resource),
            ).rowcount
        if self.on_change:
            for eid in ids:
                self.on_change("classification_superseded", f"evidence:{eid}", reason)
        return count

    def evidence_ids_for_resource(self, tenant_id: Optional[str], resource: str) -> Tuple[str, ...]:
        with self._lock:
            return tuple(r[0] for r in self._db.execute(
                "SELECT evidence_id FROM evidence WHERE status='active' AND tenant_id IS ? AND resource=?", (tenant_id, resource)))

    def _sign(self, body: str) -> str:
        assert self._key is not None
        return _b64e(hmac.new(hmac.new(self._key, b"slai-gov-v1:classification", hashlib.sha256).digest(), body.encode("ascii"), hashlib.sha256).digest())


class PrivacyClassifierAdapter:
    """Adapts the Privacy Agent's ``DataID`` (the only sensitive-data detector in SLAI) to the ``Classifier`` protocol.

    No detection happens here. The adapter maps DataID findings to GDPR classes using a small, configurable table and states
    only what DataID can support:

    * any detected entity => at least PERSONAL (confidence = lowest detector confidence among findings, so a doubtful
      detection can never become VERIFIED evidence);
    * health attributes / ``health_attribute`` entities => SPECIAL_CATEGORY (Art. 9);
    * no findings in free text => UNCLASSIFIED (heuristics cannot prove absence);
    * NON_PERSONAL only when the payload is *structurally* non-personal (numbers/booleans), via the Privacy Agent's own
      conservative structural check (``structural_check``).
    """

    SPECIAL_ENTITY_CATEGORIES = frozenset({"health_attribute", "biometric", "genetic"})
    SPECIAL_ATTRIBUTES = frozenset({"health", "biometric", "genetic", "religion", "ethnicity", "political", "sexual_orientation", "union"})
    CRIMINAL_ATTRIBUTES = frozenset({"criminal", "criminal_offence", "convictions"})

    def __init__(self, data_id: Any, *, structural_check: Optional[Callable[[Any], bool]] = None, classifier_id: str = "privacy.data_id",
                 version: str = "1", special_attributes: Iterable[str] = (), criminal_attributes: Iterable[str] = ()):
        if not callable(getattr(data_id, "identify_entities", None)):
            raise TypeError("data_id must expose identify_entities()")
        self._data_id = data_id
        self._structural = structural_check
        self._id, self._version = classifier_id, version
        self._special = self.SPECIAL_ATTRIBUTES | frozenset(special_attributes)
        self._criminal = self.CRIMINAL_ATTRIBUTES | frozenset(criminal_attributes)

    def __call__(self, payload: Any, *, purpose: str) -> ClassifierResult:
        if self._structural is not None and self._structural(payload):
            return ClassifierResult(DataClass.NON_PERSONAL, (DataClass.NON_PERSONAL,), 1.0, self._id, self._version, True, True, {})
        try:
            res = self._data_id.identify_entities(payload, purpose=purpose)  # no request_id: the classifier must have no side effects
        except Exception as exc:
            raise ClassifierUnavailable(f"data_id failed: {type(exc).__name__}") from exc
        entities = list(res.get("detected_entities") or ())
        attrs = list(res.get("sensitive_attributes") or ())
        found: Dict[str, int] = dict(res.get("entity_counts") or {})
        if not entities and not attrs:
            return ClassifierResult(DataClass.UNCLASSIFIED, (DataClass.UNCLASSIFIED,), 0.0, self._id, self._version, True, False, found)
        classes = {DataClass.PERSONAL}
        if any(str(e.get("category")) in self.SPECIAL_ENTITY_CATEGORIES for e in entities) or any(str(a.get("attribute")) in self._special for a in attrs):
            classes.add(DataClass.SPECIAL_CATEGORY)
        if any(str(a.get("attribute")) in self._criminal for a in attrs):
            classes.add(DataClass.CRIMINAL_OFFENCE)
        confidences = [float(e.get("confidence", 0.0)) for e in entities] + [float(a.get("confidence", 0.0)) for a in attrs]
        primary = most_restrictive_class(*classes)
        return ClassifierResult(primary, tuple(sorted(classes, key=class_strictness, reverse=True)), min(confidences) if confidences else 0.0,
                                self._id, self._version, True, False, found)
