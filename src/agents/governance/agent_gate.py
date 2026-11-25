"""Central agent gate: every ``BaseAgent.execute`` passes through the Governance Agent (section 9, audit 4.6).

One gate, installed process-wide by ``GovernanceAgent``; no agent owns a GDPR/AI-Act engine. ``BaseAgent`` calls
``gate_before`` / ``gate_after`` (module functions below), which are no-ops while no gate is installed - and
``status()`` / ``GovernanceAgent.health()`` say so, because an uninstalled gate is not integration.

Modes (``governance_agent.agent_gate.mode``):
  * ``off``      - nothing is evaluated (reported as such; never the default).
  * ``observe``  - (default) every execution is evaluated and durably recorded; a non-permitting decision is counted and
                   logged as ``would_block`` but execution proceeds. This is NOT enforcement and is reported as such.
  * ``enforce``  - a non-permitting decision (or a gate failure) raises ``GovernanceDenied`` BEFORE the agent runs.
Always enforced, whatever the mode: (a) agents listed in ``enforce_agents``; (b) any call whose input carries an explicit
``governance_context`` (the caller asked for a governed execution and must get one).

Without an explicit context the gate evaluates a conservative default: UNCLASSIFIED data for the agent's configured purpose.
The policy engine treats UNCLASSIFIED as potentially personal, so ``enforce`` correctly refuses uncontextualised executions;
this is why the default is ``observe`` until callers supply contexts.

Learning (AI Act Art. 10 / GDPR Art. 5(1)(b)): executions of ``learning_agents`` in a learning mode are evaluated as
``Operation.LEARN``; in enforce mode every ``dataset_ids`` entry must be registered for the tenant in the LearningRegistry.

Art. 50 marking: for agents in ``marking_agents`` a *Mapping* result is returned as a copy with ``ai_content_marking``
(signed provenance manifest, disclosures). Results that are not mappings (e.g. plain strings) cannot carry a marking without
changing their type. In observe mode they are returned unchanged and counted in
``marking_unattached``; when enforcement applies, they cannot be released unmarked.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from contextlib import contextmanager

from threading import RLock
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Tuple

from .effect_boundary import current_scope, governed_scope
from .processing_context import (Confidentiality, DataClass, Decision, GovernanceDenied, Operation, ProcessingContext,
                                 ProcessingDecision)
from .risk_classification import AISystemProfile
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Agent Gate")

CONTEXT_KEY = "governance_context"
_FIELDS = {f.name for f in dataclasses.fields(ProcessingContext)}
_ENUMS = {"operation": Operation, "data_class": DataClass, "confidentiality": Confidentiality}
_TUPLES = ("subject_ids", "recipients", "ai_practices")


class AgentGate:
    MODES = ("off", "observe", "enforce")

    def __init__(self, core: Any, *, mode: str = "observe", enforce_agents: Iterable[str] = (), exempt_agents: Iterable[str] = ("governance",),
                 default_purposes: Optional[Mapping[str, str]] = None, default_purpose: str = "agent_execution",
                 learning_agents: Iterable[str] = ("learning",), learn_modes: Iterable[str] = ("train", "auto", "fine_tune", "finetune"),
                 marking_agents: Optional[Mapping[str, Mapping[str, Any]]] = None, default_tenant_id: Optional[str] = None,
                 aliases: Optional[Mapping[str, str]] = None, internal_service_identity: bool = True,
                 classify_default_inputs: bool = True, max_classified_bytes: int = 65536) -> None:
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")
        self.core = core
        self._mode = mode
        self.mode = mode  # the setter keeps the side-effect boundary on the same mode
        #: Internal agent-to-agent executions of provably non-personal data run under the agent's own short-lived service
        #: identity. Personal data, tenant-bound capabilities and anything unclassified still need an externally supplied identity.
        self.internal_service_identity = bool(internal_service_identity)
        self.classify_default_inputs = bool(classify_default_inputs)
        self.max_classified_bytes = int(max_classified_bytes)
        self.enforce_agents = frozenset(enforce_agents)
        self.exempt = frozenset(exempt_agents)
        self.default_purposes = dict(default_purposes or {})
        self.default_purpose = default_purpose
        self.learning_agents = frozenset(learning_agents)
        self.learn_modes = frozenset(learn_modes)
        self.marking_agents = {k: dict(v) for k, v in (marking_agents or {}).items()}
        self.default_tenant_id = default_tenant_id
        self.aliases = dict(aliases or {})  # BaseAgent.name (class name) -> factory key used in configuration and capabilities
        self._lock = RLock()
        self._stats: Dict[str, int] = {"evaluated": 0, "allowed": 0, "would_block": 0, "blocked": 0, "gate_errors": 0,
                                       "explicit_context": 0, "marked": 0, "marking_unattached": 0, "marking_errors": 0,
                                       "deduplicated": 0, "release_blocked": 0, "release_would_block": 0, "default_classified": 0, "default_unclassified": 0, "nested": 0}
        self._per_agent: Dict[str, Dict[str, int]] = {}

    # ------------------------------------------------------------------ helpers
    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, value: str) -> None:
        if value not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")
        self._mode = value
        if hasattr(self.core, "enforcement_mode"):
            self.core.enforcement_mode = value  # one switch for the gate and the effect boundary

    def _bump(self, key: str, agent: Optional[str] = None) -> None:
        with self._lock:
            self._stats[key] += 1
            if agent is not None:
                row = self._per_agent.setdefault(agent, {})
                row[key] = row.get(key, 0) + 1

    def status(self) -> Dict[str, Any]:
        with self._lock:
            stats = dict(self._stats)
            per_agent = {a: dict(v) for a, v in self._per_agent.items()}
        return {"installed": True, "mode": self.mode, "enforcing": self.mode == "enforce",
                "enforce_agents": sorted(self.enforce_agents), "marking_agents": sorted(self.marking_agents), "stats": stats,
                "runtime_per_agent": per_agent,
                "note": "observe mode records decisions but does not block" if self.mode == "observe" else ""}

    def _explicit(self, input_data: Any) -> Optional[ProcessingContext]:
        if not isinstance(input_data, Mapping) or CONTEXT_KEY not in input_data:
            return None
        raw = input_data[CONTEXT_KEY]
        if isinstance(raw, ProcessingContext):
            return raw
        if not isinstance(raw, Mapping):
            raise ValueError(f"{CONTEXT_KEY} must be a ProcessingContext or a mapping")
        kw: Dict[str, Any] = {k: v for k, v in raw.items() if k in _FIELDS and k not in ("identity", "requested_at", "request_id")}
        for key, enum in _ENUMS.items():
            if key in kw and not isinstance(kw[key], enum):
                kw[key] = enum(kw[key])
        for key in _TUPLES:
            if key in kw:
                kw[key] = tuple(kw[key])
        token = raw.get("identity_token")
        if token:
            kw["identity"] = self.core.identity.verify(token, expected_tenant=kw.get("tenant_id"))
        return ProcessingContext(**kw)

    def _default_context(self, agent: str, input_data: Any) -> ProcessingContext:
        """Context for an execution that supplied none. Proportionate: data the Privacy Agent's classifier can prove
        non-personal (signed evidence, bound to this exact input) is authorised as such under the agent's own service
        identity; everything else stays UNCLASSIFIED and is judged conservatively."""
        learning = agent in self.learning_agents and isinstance(input_data, Mapping) and str(input_data.get("mode", "train")).lower() in self.learn_modes
        tenant = input_data.get("tenant_id") if isinstance(input_data, Mapping) and isinstance(input_data.get("tenant_id"), str) else self.default_tenant_id
        operation = Operation.LEARN if learning else Operation.EXECUTE
        purpose = "model_training" if learning else self.default_purposes.get(agent, self.default_purpose)
        attrs = {"gate": "agent_gate", "default_context": True}
        base = dict(operation=operation, purpose=purpose, data_class=DataClass.UNCLASSIFIED, tenant_id=tenant, agent_id=agent, attributes=attrs)
        if learning or not (self.classify_default_inputs and self.internal_service_identity):
            self._bump("default_unclassified", agent)
            return ProcessingContext(**base)
        try:
            payload = input_data
            if len(json.dumps(payload, default=str)) > self.max_classified_bytes:
                raise ValueError("input too large to classify at the gate")
            ident = self.core.identity.verify(
                self.core.identity.issue(f"agent:{agent}", tenant, actor_type="agent", ttl_seconds=60.0), expected_tenant=tenant)
            token = self.core.classification.classify(payload, purpose=purpose, tenant_id=tenant, identity=ident, single_use=False, ttl=60.0)
            ev = self.core.classification.verify(token)
            if ev.data_class in (DataClass.NON_PERSONAL, DataClass.ANONYMISED) and ev.origin.value in ("verified", "authoritative"):
                self._bump("default_classified", agent)
                return ProcessingContext(**{**base, "data_class": ev.data_class, "identity": ident,
                                            "classification_evidence": token, "data_digest": self.core.classification.digest(payload)})
        except Exception:  # no key, no classifier, unclassifiable input: stay UNCLASSIFIED (fail closed), never guess
            pass
        self._bump("default_unclassified", agent)
        return ProcessingContext(**base)

    def _lineage_failure(self, ctx: ProcessingContext, input_data: Any) -> Optional[str]:
        if ctx.operation is not Operation.LEARN:
            return None
        ids = input_data.get("dataset_ids") if isinstance(input_data, Mapping) else None
        if not isinstance(ids, (list, tuple)) or not ids:
            return "learning requires a non-empty list of registered dataset_ids (lineage)"
        if any(not isinstance(d, str) or not d.strip() for d in ids):
            return "learning dataset_ids must contain non-empty strings"
        if not ctx.tenant_id:
            return "learning requires a tenant"
        missing = [d for d in dict.fromkeys(ids) if not self.core.learning.dataset_registered(d, ctx.tenant_id)]
        return f"unregistered dataset(s) for tenant: {len(missing)}" if missing else None

    # ------------------------------------------------------------------ before execution
    def before(self, agent: str, input_data: Any) -> Optional[ProcessingDecision]:
        return self._before(agent, input_data)[0]

    @contextmanager
    def execution(self, agent: str, input_data: Any) -> Iterator[Optional[ProcessingDecision]]:
        """Authorise, then run the body inside the ambient governed scope so every side effect inherits this decision. The
        lease is completed when the body ends (also on error) - but only by the execution that created it."""
        decision, ctx, fresh = self._before(agent, input_data)
        with governed_scope(self.aliases.get(agent, agent), decision, ctx):
            try:
                yield decision
            finally:
                if fresh and decision is not None and decision.authorization_id:
                    try:
                        self.core.reassessment.complete(decision.authorization_id)
                    except Exception:  # completing a lease must never mask the execution outcome
                        pass

    def _before(self, agent: str, input_data: Any) -> Tuple[Optional[ProcessingDecision], Optional[ProcessingContext], bool]:
        """(decision, context, fresh): ``fresh`` is False when a nested execution reused an enclosing scope's decision."""
        agent = self.aliases.get(agent, agent)
        if agent in self.exempt:
            return None, None, False
        # Presence, not successful parsing, determines the enforcement boundary.
        explicit = isinstance(input_data, Mapping) and CONTEXT_KEY in input_data
        enforce = self.mode == "enforce" or agent in self.enforce_agents or explicit
        ctx: Optional[ProcessingContext] = None
        try:
            if explicit:
                self._bump("explicit_context")
            ctx = self._explicit(input_data)
            failure = None
            if ctx is not None:
                if ctx.agent_id is None:
                    ctx = dataclasses.replace(ctx, agent_id=agent)
                elif self.aliases.get(ctx.agent_id, ctx.agent_id) != agent:
                    failure = ("GOV-CAP-13", "context_agent_does_not_match_execution_target")
                if ctx.capability:
                    capability = self.core.capabilities.get(ctx.capability)
                    if capability is not None and capability.owner_agent != agent:
                        failure = ("GOV-CAP-13", "capability_owner_does_not_match_execution_target")
                if agent in self.learning_agents and isinstance(input_data, Mapping) and str(input_data.get("mode", "train")).lower() in self.learn_modes and ctx.operation is not Operation.LEARN:
                    failure = ("GOV-GATE-LRN-01", "training_execution_requires_learn_operation")
            elif not enforce and self.mode == "off":
                return None, None, False
            else:
                ctx = self._default_context(agent, input_data)
            if ctx.operation is Operation.EXECUTE and not ctx.capability:
                ctx = dataclasses.replace(ctx, capability=f"{agent}.execute")
            lineage_failure = self._lineage_failure(ctx, input_data)
            if lineage_failure and failure is None:
                failure = ("GOV-GATE-LRN-01", lineage_failure)
            # Workflow-level de-duplication: an execution nested inside an authorised scope that was already evaluated for this
            # exact request reuses that decision while its lease is still valid. It is never reused across scopes.
            parent = current_scope()
            reused = self._reuse(parent, ctx) if (parent is not None and failure is None and not ctx.approval_id) else None
            if reused is not None:
                self._bump("deduplicated", agent)
                return reused, ctx, False
            if parent is not None:
                self._bump("nested", agent)
            # The final outcome is recorded once, before any single-use approval can
            # be consumed for an operation that fails a boundary precondition.
            decision = self.core.authorize(ctx, precondition_failure=failure)
            self._bump("evaluated", agent)
            if parent is not None and decision.allowed and not ctx.approval_id:
                parent.children[ctx.fingerprint()] = decision
        except Exception as exc:  # fail closed when enforcing; never break an agent while observing
            self._bump("gate_errors")
            logger.error("Governance gate error for agent '%s': %s", agent, type(exc).__name__)
            if enforce:
                self._bump("blocked")
                decision = self._error_decision(ctx, exc)
                try:
                    self.core._record(decision)
                except Exception as record_error:
                    logger.error("Governance gate denial could not be recorded: %s", type(record_error).__name__)
                raise GovernanceDenied(decision) from exc
            return None, ctx, False
        if decision.decision.permits_execution:
            self._bump("allowed")
            return decision, ctx, True
        if enforce:
            self._bump("blocked")
            raise GovernanceDenied(decision)
        self._bump("would_block")
        logger.warning("Governance (observe): agent '%s' would be %s by %s", agent, decision.decision.value,
                       ",".join(r.rule_id for r in decision.blocking_rules) or "policy")
        return decision, ctx, True

    def _reuse(self, scope: Any, ctx: ProcessingContext) -> Optional[ProcessingDecision]:
        key = ctx.fingerprint()
        while scope is not None:
            cached = scope.children.get(key)
            if cached is not None:
                try:
                    self.core.reassessment.check(cached.authorization_id)
                    return cached
                except Exception:
                    scope.children.pop(key, None)
                    return None
            scope = scope.parent
        return None

    def _error_decision(self, ctx: Optional[ProcessingContext], exc: Exception) -> ProcessingDecision:
        from .processing_context import RuleResult, new_id

        rid = ctx.request_id if ctx is not None else new_id("gov_req")
        return ProcessingDecision(rid, Decision.BLOCK, (RuleResult("GOV-GATE-ERR-01", Decision.BLOCK, f"gate failure: {type(exc).__name__}"),),
                                  self.core.policies.active_version(), ctx.fingerprint() if ctx is not None else "", self.core._clock())

    # ------------------------------------------------------------------ after execution
    def after(self, agent: str, input_data: Any, result: Any, decision: Optional[ProcessingDecision] = None) -> Any:
        """Mark (Art. 50) and then enforce the RELEASE-phase conditions of this execution's authorisation: an output whose
        mandatory release conditions are unresolved is not released in enforce mode (observe counts it)."""
        agent = self.aliases.get(agent, agent)
        released = self._mark(agent, input_data, result, decision)
        auth_id = decision.authorization_id if decision is not None else None
        if auth_id:
            try:
                unmet = self.core.conditions.release_unmet(auth_id)
            except Exception as exc:
                self._bump("gate_errors")
                logger.error("Release-condition check failed for agent '%s': %s", agent, type(exc).__name__)
                unmet = ["release_check_error"]
            if unmet:
                enforce = self.mode == "enforce" or agent in self.enforce_agents or (isinstance(input_data, Mapping) and CONTEXT_KEY in input_data)
                self._bump("release_blocked" if enforce else "release_would_block", agent)
                if enforce:
                    from .conditions import ConditionsUnmet, Phase

                    raise ConditionsUnmet(auth_id, Phase.RELEASE, unmet, "GOV-CND-04")
        return released

    def _mark(self, agent: str, input_data: Any, result: Any, decision: Optional[ProcessingDecision] = None) -> Any:
        spec = self.marking_agents.get(agent)
        if not spec:
            return result
        enforce = self.mode == "enforce" or agent in self.enforce_agents or (
            isinstance(input_data, Mapping) and CONTEXT_KEY in input_data
        )
        if not isinstance(result, Mapping):
            self._bump("marking_unattached")
            if enforce:
                self._bump("marking_errors")
                raise GovernanceDenied(self._error_decision(None, TypeError("marking requires a mapping result")))
            return result
        try:
            profile = AISystemProfile(
                system_id=str(spec.get("system_id", agent)), name=str(spec.get("name", agent)),
                interacts_with_natural_persons=bool(spec.get("interacts_with_natural_persons", True)),
                generates_synthetic_content=bool(spec.get("generates_synthetic_content", True)),
                deepfake_content=bool(spec.get("deepfake_content", False)), public_interest_text=bool(spec.get("public_interest_text", False)))
            payload = json.dumps(dict(result), sort_keys=True, default=str, ensure_ascii=False)
            released = self.core.transparency.release(profile, payload, model=str(spec.get("model", agent)), kind="text",
                                                      provider=str(spec.get("provider", "")), language=str(spec.get("language", "en")))
            marked = dict(result)
            marked["ai_content_marking"] = {"manifest": released.manifest, "disclosures": list(released.disclosures),
                                            "visible_label": released.visible_label, "measures": list(released.measures_applied),
                                            "limitations": list(released.limitations)}
            self._bump("marked")
            if decision is not None and decision.authorization_id:
                digest = hashlib.sha256(json.dumps(released.manifest, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]
                self.core.conditions.satisfy_system(decision.authorization_id, "art50_transparency_obligations", f"marking:{digest}")
            return marked
        except Exception as exc:
            self._bump("marking_errors")
            logger.error("Art. 50 marking failed for agent '%s': %s", agent, type(exc).__name__)
            if enforce:
                raise GovernanceDenied(self._error_decision(None, exc)) from exc  # enforcing: an output that must be marked is not released unmarked
            return result  # observing: counted in marking_errors and visible in status()


# ---------------------------------------------------------------------- process-wide installation
_lock = RLock()
_gate: Optional[AgentGate] = None


def install_gate(gate: Optional[AgentGate]) -> None:
    global _gate
    with _lock:
        _gate = gate


def installed_gate() -> Optional[AgentGate]:
    with _lock:
        return _gate


def gate_before(agent: str, input_data: Any) -> Optional[ProcessingDecision]:
    gate = installed_gate()
    return gate.before(agent, input_data) if gate is not None else None


@contextmanager
def governed_execution(agent: str, input_data: Any) -> Iterator[Optional[ProcessingDecision]]:
    """The one wrapper ``BaseAgent.execute`` uses: gate + ambient scope. A plain pass-through while no gate is installed."""
    gate = installed_gate()
    if gate is None:
        yield None
        return
    with gate.execution(agent, input_data) as decision:
        yield decision


def gate_after(agent: str, input_data: Any, result: Any, decision: Optional[ProcessingDecision] = None) -> Any:
    gate = installed_gate()
    return gate.after(agent, input_data, result, decision) if gate is not None else result
