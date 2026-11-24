"""``GovernanceCore``: the stdlib-only composition of every mandatory governance component.

Agents, applications and tests that need a regulatory decision use this class
(directly or via ``GovernanceAgent``). It deliberately has no dependency on
``BaseAgent``/torch so enforcement keeps working when optional analytics (or
other agents) are unavailable - that is the "degraded but still enforcing" rule.
"""

from __future__ import annotations

import base64
import json
import os
import time

from pathlib import Path
from threading import RLock
from typing import Any, Callable, Dict, List, Mapping, Optional, Union

from .compliance_engine import ComplianceEngine, DeploymentProfile
from .human_oversight import OversightManager
from .incident_registry import IncidentKind, IncidentRegistry
from .lawful_basis import ConsentProvider, RestrictionProvider
from .policy_engine import PolicyEngine, TransferPolicy
from .policy_registry import CapabilityRegistry, PolicyRegistry
from .processing_context import (
    Decision,
    GovernanceDenied,
    ProcessingContext,
    ProcessingDecision,
    fingerprint,
)
from .processing_registry import ProcessingRegistry
from .regulatory_context import AuthorizationGrant, GrantAuthority, GrantError, derive_key
from .incident_bridge import IncidentIntake
from .learning_governance import LearningDeletionAdapter, LearningRegistry
from .privacy_dashboard import PrivacyDashboard
from .privacy_ledger import PrivacyLedger
from .subject_rights import SubjectRights
from .transparency import TransparencyMarker
from .regulatory_evidence import EvidenceStore
from .regulatory_reporting import RegulatoryReporter
from .risk_classification import ProfileRegistry
from .utils.identity import IdentityAuthority
from .utils.store import open_db

__all__ = ["GovernanceCore", "MASTER_KEY_ENV", "MASTER_KEY_FILE_ENV"]

MASTER_KEY_ENV = "SLAI_GOVERNANCE_MASTER_KEY"
MASTER_KEY_FILE_ENV = "SLAI_GOVERNANCE_MASTER_KEY_FILE"

_instances: Dict[str, "GovernanceCore"] = {}
_instances_lock = RLock()


def load_master_key(env: Optional[Mapping[str, str]] = None) -> Optional[bytes]:
    """Key comes from the environment or a key file; it is never generated or stored here."""
    env = os.environ if env is None else env
    raw = env.get(MASTER_KEY_ENV)
    if not raw and env.get(MASTER_KEY_FILE_ENV):
        try:
            raw = Path(env[MASTER_KEY_FILE_ENV]).read_text(encoding="utf-8").strip()
        except OSError:
            raw = None
    if not raw:
        return None
    try:
        key: Optional[bytes] = base64.b64decode(raw, validate=True)
    except ValueError:
        key = None
    if key is None or len(key) < 32:
        # Not base64, or base64 that decodes to < 32 bytes (a 40-character alphanumeric passphrase is "valid base64" yet
        # decodes to 30 bytes): use the text itself rather than silently reporting "no key".
        key = raw.encode("utf-8")
    return key if len(key) >= 32 else None


class GovernanceCore:
    def __init__(
        self,
        state_dir: Union[str, Path],
        *,
        master_key: Optional[bytes] = None,
        config: Optional[Mapping[str, Any]] = None,
        consent: Optional[ConsentProvider] = None,
        restrictions: Optional[RestrictionProvider] = None,
        release_gate: Optional[Callable[[str], bool]] = None,
        clock: Callable[[], float] = time.time,
    ):
        cfg = dict(config or {})
        self.config = cfg
        self.state_dir = Path(state_dir)
        self._clock = clock
        key = master_key if master_key is not None else load_master_key()
        self.key_configured = bool(key and len(key) >= 32)

        self.identity = IdentityAuthority(derive_key(key, "identity") if self.key_configured else None)
        self.policies = PolicyRegistry()
        self.capabilities = CapabilityRegistry()
        self.profiles = ProfileRegistry()
        self.activities = ProcessingRegistry(self.state_dir / "processing.db")
        self.oversight = OversightManager(
            self.state_dir / "oversight.db",
            qualified_roles=cfg.get("qualified_roles", ("oversight_reviewer", "dpo", "governance_admin")),
            emergency_roles=cfg.get("emergency_roles", ("safety_officer", "governance_admin")),
            min_review_seconds=float(cfg.get("min_review_seconds", 5.0)),
            max_approval_ttl=float(cfg.get("max_approval_ttl_seconds", 3600.0)),
            authority=self.identity,
        )
        self.grants = GrantAuthority(
            self.policies,
            master_key=key if self.key_configured else None,
            path=self.state_dir / "grants.db",
            max_ttl=float(cfg.get("grant_ttl_seconds", 900.0)),
            max_delegation_depth=int(cfg.get("max_delegation_depth", 2)),
        )
        self.evidence = EvidenceStore(self.identity, path=self.state_dir / "evidence.db", master_key=key if self.key_configured else None)
        self.incidents = IncidentRegistry(
            self.state_dir / "incidents.db", authorised_roles=cfg.get("authorised_reporter_roles", ("dpo", "authorised_reporter"))
        )
        self.incident_intake = IncidentIntake(self.incidents)
        self.compliance = ComplianceEngine(
            self.evidence, independent_producers=cfg.get("independent_evidence_producers", ("verification_agent", "ci_pipeline"))
        )
        self.reporter = RegulatoryReporter()
        self.engine = PolicyEngine(
            policies=self.policies, activities=self.activities, capabilities=self.capabilities, profiles=self.profiles,
            oversight=self.oversight, identity=self.identity, consent=consent, restrictions=restrictions,
            transfers=TransferPolicy(), release_gate=release_gate, clock=clock,
        )
        self.ledger = PrivacyLedger(
            self.state_dir / "privacy.db",
            tombstone_salt=derive_key(key, "tombstone") if self.key_configured else b"unconfigured-" + b"x" * 19,
        )
        self.learning = LearningRegistry(
            self.state_dir / "learning.db",
            digest_salt=derive_key(key, "learning-digest") if self.key_configured else b"unconfigured-" + b"y" * 19,
        )
        self.rights = SubjectRights(self.state_dir / "rights.db", processing_info=self._processing_information)
        self.dashboard = PrivacyDashboard(self.identity, self.ledger, self.rights, self.activities)
        self.transparency = TransparencyMarker(derive_key(key, "transparency") if self.key_configured else None)
        # Erasure must reach the learning pipeline: it opens model impacts that block completion until remediated.
        self.ledger.register_adapter(LearningDeletionAdapter(self.learning))
        # The durable ledger is the default authority for consent/restriction state (audit 4.2);
        # an explicitly supplied provider still wins (tests, migration).
        if consent is None:
            consent = self.ledger.provider()
        if restrictions is None:
            restrictions = self.ledger.provider()
        self.engine.consent, self.engine.restrictions = consent, restrictions
        self._consent = consent
        self._restrictions = restrictions
        self._decisions = open_db(self.state_dir / "decisions.db")
        self._decisions.execute(
            "CREATE TABLE IF NOT EXISTS decisions (seq INTEGER PRIMARY KEY AUTOINCREMENT, request_id TEXT NOT NULL, "
            "fingerprint TEXT NOT NULL, decision TEXT NOT NULL, rules TEXT NOT NULL, policy_version TEXT NOT NULL, at REAL NOT NULL)"
        )
        self._lock = RLock()

    # ------------------------------------------------------------------ singleton access
    @classmethod
    def shared(cls, state_dir: Union[str, Path], **kwargs: Any) -> "GovernanceCore":
        """One core per state directory per process: no duplicate initialisation or divergent state."""
        key = str(Path(state_dir).resolve())
        with _instances_lock:
            inst = _instances.get(key)
            if inst is None:
                inst = _instances[key] = cls(state_dir, **kwargs)
            return inst

    @classmethod
    def reset_shared(cls) -> None:
        with _instances_lock:
            _instances.clear()

    # ------------------------------------------------------------------ providers (wired by the privacy ledger)
    def attach_providers(self, *, consent: Optional[ConsentProvider] = None, restrictions: Optional[RestrictionProvider] = None) -> None:
        if consent is not None:
            self.engine.consent = self._consent = consent
        if restrictions is not None:
            self.engine.restrictions = self._restrictions = restrictions

    # ------------------------------------------------------------------ decisions
    def authorize(self, ctx: ProcessingContext) -> ProcessingDecision:
        """Evaluate, record (metadata only) and consume any single-use approval on a positive outcome."""
        decision = self.engine.evaluate(ctx)
        if decision.allowed:
            for control in decision.conditions:
                if control.startswith("consume_approval:"):
                    try:
                        self.oversight.consume(control.split(":", 1)[1], ctx, now=self._clock())
                    except Exception:
                        decision = self._deny(ctx, decision, "GOV-HO-14", "approval_consumption_failed")
        self._record(decision)
        return decision

    def enforce(self, ctx: ProcessingContext, *, accept_modifications: bool = False) -> ProcessingDecision:
        decision = self.authorize(ctx)
        if decision.allowed or (accept_modifications and decision.requires_modification):
            return decision
        raise GovernanceDenied(decision)

    def issue_grant(self, ctx: ProcessingContext, decision: ProcessingDecision, **kwargs: Any) -> str:
        if ctx.identity is None:
            raise GrantError("identity_required")
        activity = self.activities.get(ctx.activity_id)
        return self.grants.issue(
            ctx, decision, principal_id=ctx.identity.principal_id,
            lawful_basis=activity.basis.value if activity else None,
            special_category_condition=(activity.art9_condition.value if activity and activity.art9_condition else None),
            now=self._clock(), **kwargs,
        )

    def verify_grant(self, token: Optional[str], ctx: Optional[ProcessingContext] = None) -> AuthorizationGrant:
        return self.grants.verify(token, ctx=ctx, now=self._clock())

    # ------------------------------------------------------------------ health
    def health(self) -> Dict[str, Any]:
        """Truthful readiness: 'ready' only if every mandatory dependency is operational.

        Enforcement itself never degrades to permissive; a missing dependency
        instead makes the affected requests QUARANTINE/BLOCK.
        """
        chain_ok, bad = self.evidence.verify_chain() if self.evidence.available else (False, None)
        components = {
            "identity_authority": self.identity.available,
            "grant_authority": self.grants.available,
            "evidence_signing": self.evidence.available,
            "evidence_chain_intact": chain_ok,
            "consent_provider": self._consent is not None,
            "restriction_provider": self._restrictions is not None,
            "mandatory_policies": len(self.policies.list()),
        }
        problems = [k for k, v in components.items() if v is False]
        return {
            "enforcing": True,  # structurally non-optional
            "ready": not problems,
            "status": "ready" if not problems else "degraded_enforcing",
            "problems": problems,
            "policy_version": self.policies.active_version(),
            "components": components,
            "first_bad_evidence_seq": bad,
        }

    def _processing_information(self, tenant_id: str) -> Dict[str, Any]:
        """Art. 15(1)(a)-(h) information drawn from the Art. 30 register (no internal evidence fields)."""
        acts = self.activities.export_ropa()
        return {
            "purposes": sorted({p for a in acts for p in a["purposes"]}),
            "categories_of_data": sorted({str(c) for a in acts for c in a["data_categories"]}),
            "recipients": sorted({r for a in acts for r in a["recipients"]}),
            "retention_days": {a["activity_id"]: a["retention_days"] for a in acts},
            "rights": ["access", "rectification", "erasure", "restriction", "portability", "objection", "withdraw_consent", "complain_to_supervisory_authority"],
        }

    def close(self) -> None:
        for part in (self.activities, self.oversight, self.incidents, self.evidence, self.ledger, self.learning, self.rights, self._decisions):
            try:
                part.close() if hasattr(part, "close") else None
            except Exception:
                pass

    # ------------------------------------------------------------------ internals
    def _deny(self, ctx, decision: ProcessingDecision, rule_id: str, reason: str) -> ProcessingDecision:
        from .processing_context import RuleResult
        results = decision.results + (RuleResult(rule_id, Decision.BLOCK, reason),)
        return ProcessingDecision(decision.request_id, Decision.BLOCK, results, decision.policy_version, decision.context_fingerprint, self._clock())

    def _record(self, d: ProcessingDecision) -> None:
        with self._lock:
            self._decisions.execute(
                "INSERT INTO decisions(request_id, fingerprint, decision, rules, policy_version, at) VALUES (?,?,?,?,?,?)",
                (d.request_id, d.context_fingerprint, d.decision.value, json.dumps([r.rule_id for r in d.results]), d.policy_version, d.decided_at),
            )

    def decision_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        rows = self._decisions.execute(
            "SELECT request_id, fingerprint, decision, rules, policy_version, at FROM decisions ORDER BY seq DESC LIMIT ?", (limit,)
        ).fetchall()
        return [{"request_id": r[0], "fingerprint": r[1], "decision": r[2], "rules": json.loads(r[3]), "policy_version": r[4], "at": r[5]} for r in rows]
