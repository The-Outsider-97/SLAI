"""Governance-aware inter-agent coordination (GOV-P1-02).

The Governance Agent does not replace the Privacy, Safety, Alignment or Provenance agents; it decides
*which* of them a request needs, calls them through the same governed delegation path everything
else uses, and folds what they report back into the decision - restrictively only.

Guarantees
----------
* **Selective**: a specialist is invoked only when its contract trigger holds for the *resolved*
  request (never for caller assertions) and a result could still change the outcome. A request that is
  already BLOCKED/QUARANTINED invokes nothing; a classification that is already backed by verified
  evidence does not invoke the Privacy agent again.
* **Evidence contract**: a specialist report is accepted only if it names the right specialist, an
  evidence kind the contract allows, a known verdict, and the fingerprint of *this* request. Anything
  else is invalid evidence and treated as a failure.
* **Monotone**: verdict ``clear`` never relaxes anything; ``concern`` -> REVIEW_REQUIRED (overridable by a
  verified human approval, like any other review); ``violation`` -> BLOCK; ``unknown`` or a failed /
  invalid / timed-out *required* specialist -> QUARANTINE. A failed *optional* specialist is recorded and
  changes nothing (it cannot relax either).
* **No recursion, no duplication**: a specialist already on the invocation stack (or equal to the
  requesting agent) is never re-entered; stack depth is bounded; one call per specialist per request
  fingerprint and policy epoch is reused from a small TTL cache.
* **Data minimisation**: the coordinator never stores or logs the payload. It is handed to the invoke
  port only, which is the agent call.

Evidence provenance: reports are in-process return values of an invocation the coordinator itself made
and bound to the request fingerprint; they are not accepted from callers. They are not independently
signed - an agent that is compromised in-process can lie, which is why a report can only restrict.
"""

from __future__ import annotations

import concurrent.futures
import contextvars
import threading
import time

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple

from .processing_context import (
    DataClass,
    Decision,
    Operation,
    PERSONAL_DATA_CLASSES,
    ProcessingContext,
    RuleResult,
)

__all__ = [
    "ADAPTERS",
    "CoordinationResult",
    "Coordinator",
    "DEFAULT_CONTRACTS",
    "SpecialistContract",
    "SpecialistReport",
    "Verdict",
    "agent_invoker",
]


class Verdict(str, Enum):
    CLEAR = "clear"
    CONCERN = "concern"
    VIOLATION = "violation"
    UNKNOWN = "unknown"


_SIDE_EFFECT_OPS = frozenset({Operation.EXECUTE, Operation.SHARE, Operation.EXPORT, Operation.TRANSFER})
Facts = Mapping[str, Any]
Invoke = Callable[[str, str, Mapping[str, Any]], Mapping[str, Any]]


@dataclass(frozen=True)
class SpecialistContract:
    name: str  # factory agent type
    action: str  # public delegable action
    evidence_kinds: FrozenSet[str]
    trigger: Callable[[Facts], bool]
    why: str
    required: bool = True  # a failed required specialist QUARANTINEs
    timeout_seconds: float = 30.0


def _trig_privacy(f: Facts) -> bool:
    # Classification backed by verified evidence is not re-derived; only unverified personal-looking data or learning on it.
    return (f["data_class"] == DataClass.UNCLASSIFIED.value and not f["classification_verified"]) or (
        f["operation"] == Operation.LEARN.value and f["personal"]
    )


def _trig_safety(f: Facts) -> bool:
    return bool(f["ai_practices"]) or (f["high_risk"] and f["operation"] in {o.value for o in _SIDE_EFFECT_OPS})


def _trig_provenance(f: Facts) -> bool:
    return f["operation"] == Operation.LEARN.value


def _trig_alignment(f: Facts) -> bool:
    return bool(f["automated_decision"]) and bool(f["significant_effect"])


DEFAULT_CONTRACTS: Tuple[SpecialistContract, ...] = (
    SpecialistContract("privacy", "evaluate_privacy", frozenset({"privacy_assessment"}), _trig_privacy,
                       "personal-data exposure must be assessed by the Privacy agent before use or learning"),
    SpecialistContract("safety", "validate_action", frozenset({"safety_assessment"}), _trig_safety,
                       "Art. 5 practice tags or high-risk side effects need an independent safety assessment"),
    SpecialistContract("provenance", "get_provenance", frozenset({"provenance_attestation"}), _trig_provenance,
                       "learning requires data lineage", required=False),
    SpecialistContract("alignment", "verify_alignment", frozenset({"alignment_assessment"}), _trig_alignment,
                       "significant automated decisions need an alignment review", required=False),
)


@dataclass(frozen=True)
class SpecialistReport:
    specialist: str
    action: str
    evidence_kind: str
    verdict: Verdict
    request_fingerprint: str
    summary: str = ""


@dataclass(frozen=True)
class CoordinationResult:
    request_fingerprint: str
    invoked: Tuple[str, ...] = ()
    reused: Tuple[str, ...] = ()
    skipped: Tuple[Tuple[str, str], ...] = ()  # (specialist, reason)
    reports: Tuple[SpecialistReport, ...] = ()
    failures: Tuple[Tuple[str, str], ...] = ()
    results: Tuple[RuleResult, ...] = ()  # rule results to merge into the engine evaluation (can only restrict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_fingerprint": self.request_fingerprint,
            "invoked": list(self.invoked),
            "reused": list(self.reused),
            "skipped": [list(s) for s in self.skipped],
            "reports": [{"specialist": r.specialist, "kind": r.evidence_kind, "verdict": r.verdict.value, "summary": r.summary} for r in self.reports],
            "failures": [list(f) for f in self.failures],
        }


_stack: "contextvars.ContextVar[Tuple[str, ...]]" = contextvars.ContextVar("slai_coordination_stack", default=())


class Coordinator:
    MAX_DEPTH = 2

    def __init__(
        self,
        contracts: Sequence[SpecialistContract] = DEFAULT_CONTRACTS,
        *,
        cache_ttl_seconds: float = 300.0,
        clock: Callable[[], float] = time.time,
        epoch: Callable[[], int] = lambda: 0,
        policy_version: Callable[[], str] = lambda: "",
        high_risk: Callable[[ProcessingContext], bool] = lambda ctx: False,
    ):
        self._contracts = {c.name: c for c in contracts}
        self._ttl = cache_ttl_seconds
        self._clock = clock
        self._epoch = epoch
        self._policy_version = policy_version
        self._high_risk = high_risk
        self._lock = threading.RLock()
        self._cache: Dict[Tuple[str, str, str, int, str], Tuple[float, SpecialistReport]] = {}
        self.stats: Dict[str, int] = {"requests": 0, "invoked": 0, "reused": 0, "skipped": 0, "failed": 0, "recursion_prevented": 0}

    # ------------------------------------------------------------------ planning
    def facts(self, ctx: ProcessingContext) -> Dict[str, Any]:
        status = str(getattr(ctx, "classification_status", "") or "")
        return {
            "operation": ctx.operation.value,
            "data_class": ctx.effective_data_class.value,
            "personal": ctx.effective_data_class in PERSONAL_DATA_CLASSES,
            "classification_verified": status.startswith("evidence_ok"),
            "ai_practices": tuple(ctx.ai_practices),
            "high_risk": bool(self._high_risk(ctx)),
            "automated_decision": ctx.automated_decision,
            "significant_effect": ctx.significant_effect,
        }

    def plan(self, ctx: ProcessingContext, decision: Decision) -> Tuple[List[SpecialistContract], List[Tuple[str, str]]]:
        """Which specialists could still change the outcome of ``ctx`` given the preliminary ``decision``."""
        skipped: List[Tuple[str, str]] = []
        if decision in (Decision.BLOCK, Decision.QUARANTINE):
            # Reports are restrictive-only: nothing a specialist says can improve a refused request.
            return [], [(n, f"decision_already_{decision.value.lower()}") for n in self._contracts]
        f = self.facts(ctx)
        selected: List[SpecialistContract] = []
        for c in self._contracts.values():
            try:
                hit = bool(c.trigger(f))
            except Exception:
                hit = True  # a broken trigger must not silently drop a safeguard
            if hit:
                selected.append(c)
            else:
                skipped.append((c.name, "not_triggered"))
        return selected, skipped

    # ------------------------------------------------------------------ execution
    def coordinate(self, ctx: ProcessingContext, decision: Decision, invoke: Invoke, *, payload: Any = None,
                   requester: Optional[str] = None) -> CoordinationResult:
        fp = ctx.fingerprint()
        self.stats["requests"] += 1
        selected, skipped = self.plan(ctx, decision)
        invoked: List[str] = []
        reused: List[str] = []
        reports: List[SpecialistReport] = []
        failures: List[Tuple[str, str]] = []
        results: List[RuleResult] = []
        stack = _stack.get()
        for c in selected:
            if c.name in stack or c.name == (requester or ctx.agent_id) or len(stack) >= self.MAX_DEPTH:
                self.stats["recursion_prevented"] += 1
                skipped.append((c.name, "recursion_prevented"))
                if c.required:  # not re-entering is fine, but a required safeguard that never runs cannot be assumed
                    failures.append((c.name, "recursion_prevented"))
                continue
            cached = self._cached(c, fp)
            if cached is not None:
                reused.append(c.name)
                self.stats["reused"] += 1
                reports.append(cached)
                continue
            report, error = self._call(c, fp, invoke, payload, stack, {"purpose": ctx.purpose, "operation": ctx.operation.value})
            if report is None:
                failures.append((c.name, error))
                self.stats["failed"] += 1
                continue
            invoked.append(c.name)
            self.stats["invoked"] += 1
            reports.append(report)
            self._store(c, fp, report)
        self.stats["skipped"] += len(skipped)

        for rep in reports:
            results.extend(self._map(rep))
        required = {c.name for c in selected if c.required}
        for name, why in failures:
            if name in required:
                results.append(RuleResult("GOV-COORD-03", Decision.QUARANTINE, f"required_specialist_unavailable:{name}:{why}", ("AI Act Art. 14", "GDPR Art. 25")))
        return CoordinationResult(fp, tuple(invoked), tuple(reused), tuple(skipped), tuple(reports), tuple(failures), tuple(results))

    # ------------------------------------------------------------------ internals
    def _key(self, c: SpecialistContract, fp: str) -> Tuple[str, str, str, int, str]:
        return (c.name, c.action, fp, self._epoch(), self._policy_version())

    def _cached(self, c: SpecialistContract, fp: str) -> Optional[SpecialistReport]:
        with self._lock:
            hit = self._cache.get(self._key(c, fp))
            if hit and self._clock() - hit[0] <= self._ttl:
                return hit[1]
            return None

    def _store(self, c: SpecialistContract, fp: str, report: SpecialistReport) -> None:
        with self._lock:
            if len(self._cache) > 512:
                oldest = sorted(self._cache.items(), key=lambda kv: kv[1][0])[:128]
                for k, _ in oldest:
                    self._cache.pop(k, None)
            self._cache[self._key(c, fp)] = (self._clock(), report)

    def _call(self, c: SpecialistContract, fp: str, invoke: Invoke, payload: Any, stack: Tuple[str, ...], meta: Mapping[str, Any]):
        request = {"request_fingerprint": fp, "specialist": c.name, "action": c.action, "payload": payload, **meta}

        def run() -> Mapping[str, Any]:
            token = _stack.set(stack + (c.name,))
            try:
                return invoke(c.name, c.action, request)
            finally:
                _stack.reset(token)

        ctxvars = contextvars.copy_context()
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"coord-{c.name}")
        try:
            future = pool.submit(ctxvars.run, run)
            raw = future.result(timeout=c.timeout_seconds)
        except concurrent.futures.TimeoutError:
            return None, "timeout"
        except Exception as exc:  # includes GovernanceDenied raised by the specialist's own gate
            return None, f"error:{type(exc).__name__}"
        finally:
            pool.shutdown(wait=False)
        return self._validate(c, fp, raw)

    def _validate(self, c: SpecialistContract, fp: str, raw: Any):
        if not isinstance(raw, Mapping):
            return None, "invalid_evidence:not_a_mapping"
        if raw.get("specialist") != c.name:
            return None, "invalid_evidence:wrong_specialist"
        if raw.get("evidence_kind") not in c.evidence_kinds:
            return None, "invalid_evidence:kind_not_in_contract"
        if raw.get("request_fingerprint") != fp:
            return None, "invalid_evidence:not_bound_to_this_request"
        try:
            verdict = Verdict(raw.get("verdict"))
        except ValueError:
            return None, "invalid_evidence:unknown_verdict"
        return SpecialistReport(c.name, c.action, str(raw["evidence_kind"]), verdict, fp, str(raw.get("summary", ""))[:200]), ""

    @staticmethod
    def _map(rep: SpecialistReport) -> List[RuleResult]:
        reason = f"{rep.specialist}:{rep.evidence_kind}:{rep.verdict.value}"
        if rep.verdict is Verdict.VIOLATION:
            return [RuleResult("GOV-COORD-01", Decision.BLOCK, reason, ("AI Act Art. 14",))]
        if rep.verdict is Verdict.CONCERN:
            return [RuleResult("GOV-COORD-02", Decision.REVIEW_REQUIRED, reason, ("AI Act Art. 14",), overridable=True)]
        if rep.verdict is Verdict.UNKNOWN:
            return [RuleResult("GOV-COORD-04", Decision.QUARANTINE, reason, ("GDPR Art. 25",))]
        return []  # CLEAR never relaxes anything


# ----------------------------------------------------------------------------------------------------------------------
# Port implementation over the real agents. Adapters translate each agent's *own* result shape into the evidence contract.
# Anything they do not recognise becomes ``unknown`` (restrictive) - never ``clear``.
# ----------------------------------------------------------------------------------------------------------------------
def _as_mapping(payload: Any) -> Mapping[str, Any]:
    return payload if isinstance(payload, Mapping) else {"content": payload}


def _adapt_privacy(raw: Any) -> Tuple[str, str]:
    decision = str(raw.get("decision", "")).lower() if isinstance(raw, Mapping) else ""
    return {"allow": ("clear", ""), "modify": ("concern", "minimisation_required"), "escalate": ("concern", "escalated"), "block": ("violation", "blocked")}.get(
        decision, ("unknown", "unrecognised_privacy_decision"))


def _adapt_safety(raw: Any) -> Tuple[str, str]:
    decision = str(raw.get("decision", "")).lower() if isinstance(raw, Mapping) else ""
    if decision == "allow" and raw.get("approved") is True:
        return "clear", ""
    return {"review": ("concern", "human_review"), "block": ("violation", "blocked")}.get(decision, ("unknown", "unrecognised_safety_decision"))


def _adapt_alignment(raw: Any) -> Tuple[str, str]:
    d = raw.get("decision") if isinstance(raw, Mapping) else None
    if isinstance(d, Mapping):
        if d.get("approved") is True and not d.get("requires_review"):
            return "clear", ""
        if d.get("requires_review") or d.get("decision") == "review_required":
            return "concern", "review_required"
    return "unknown", "unrecognised_alignment_decision"


def _adapt_provenance(raw: Any) -> Tuple[str, str]:
    # Lineage present and error-free is the only thing this action can attest; empty lineage means "no provenance known".
    if isinstance(raw, Mapping) and raw and not raw.get("error") and (raw.get("lineage") or raw.get("sources") or raw.get("artifact_id")):
        return "clear", ""
    return "unknown", "no_lineage_for_artifact"


ADAPTERS: Mapping[str, Callable[[Any], Tuple[str, str]]] = {
    "privacy": _adapt_privacy, "safety": _adapt_safety, "alignment": _adapt_alignment, "provenance": _adapt_provenance,
}
_KIND = {"privacy": "privacy_assessment", "safety": "safety_assessment", "alignment": "alignment_assessment", "provenance": "provenance_attestation"}


def _kwargs(name: str, request: Mapping[str, Any]) -> Dict[str, Any]:
    payload = request.get("payload")
    purpose, operation = request.get("purpose"), request.get("operation")
    if name == "privacy":
        return {"payload": _as_mapping(payload), "request_id": str(request["request_fingerprint"])[:24], "purpose": purpose, "action": operation}
    if name == "safety":
        return {"action_params": {"action_name": f"{operation}:{purpose}", "parameters": _as_mapping(payload)}}
    if name == "alignment":
        return {"task_data": dict(_as_mapping(payload))}
    if name == "provenance":
        artifact = payload.get("artifact_id") if isinstance(payload, Mapping) else None
        if not artifact:
            raise ValueError("provenance needs payload['artifact_id']")
        return {"artifact_id": str(artifact)}
    raise KeyError(name)


def agent_invoker(agent_factory: Any, shared_memory: Any, *, caller: str = "governance_agent") -> Invoke:
    """Build the coordinator's ``invoke`` over the AgentFactory, through the same governed delegation guard the
    orchestrator uses (``check_delegable``): only public, class-defined, authorised actions run."""
    from .delegation_guard import check_delegable

    def invoke(name: str, action: str, request: Mapping[str, Any]) -> Mapping[str, Any]:
        adapter = ADAPTERS.get(name)
        if adapter is None:
            raise KeyError(f"no adapter for specialist '{name}'")
        agent = agent_factory.create(agent_factory.normalize_agent_type(name), shared_memory)
        method = check_delegable(agent, target_name=name, action=action, caller=caller)
        verdict, summary = adapter(method(**_kwargs(name, request)))
        return {"specialist": name, "evidence_kind": _KIND[name], "verdict": verdict,
                "request_fingerprint": request["request_fingerprint"], "summary": summary}

    return invoke
