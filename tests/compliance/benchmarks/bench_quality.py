"""Governance decision-quality benchmark (GOV-P2-02): false positives / false negatives, latency percentiles, adaptive evaluation.

Run:    python -m tests.compliance.benchmarks.bench_quality            (full)
        python -m tests.compliance.benchmarks.bench_quality --smoke    (small, fast, used by CI)

How to read the numbers
-----------------------
* **Ground truth comes from the template, never from the engine.** (One template label was corrected after a first run: reusable
  classification evidence reused for the *same* payload was initially labelled "must restrict"; the specified behaviour is that only
  ``single_use`` evidence is consumed. The scenario was split into a single-use replay (must restrict) and a reusable-evidence reuse
  (must proceed). This is recorded in the final report; no other label was changed.) Every case is generated from a scenario whose expected
  outcome follows from the regulation it exercises (e.g. "GDPR Art. 6: no registered activity => must not proceed"). The labels
  are written down before the system is run and are not adjusted to make the system look better; any disagreement is reported.
  The corpus is synthetic and small: it measures *whether the controls behave as specified on these scenarios*, NOT real-world
  precision/recall, and it says nothing about legal compliance.
* Positive class = "must NOT proceed as submitted". FN (a must-restrict request was allowed) is a compliance leak; FP (a
  legitimate request was restricted) is over-blocking. ``restricted`` = any outcome other than ALLOW / ALLOW_WITH_CONTROLS.
* No before/after of the original code is claimed: the governance package has no committed pre-assignment baseline (git HEAD does
  not contain it), so historical numbers cannot be reproduced. Comparisons here are between configurations of the *current*
  code (governance enforcing vs. no governance; adaptive off vs. on; priority vs. FIFO triage) on the same data.
* Adaptive evaluation uses a disjoint test split that is never recorded, validated on or fitted from.
* The triage comparison is a *queueing simulation* with stated assumptions, not field evidence.
* Latency depends on the machine; the environment block is part of the output and the figures must not be quoted without it.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
import statistics
import sys
import tempfile
import time

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from src.agents.governance.core import GovernanceCore
from src.agents.governance.decision_intel import rank
from src.agents.governance.processing_context import (
    Art9Condition, DataClass, Decision, LawfulBasis, Operation, ProcessingContext,
)
from src.agents.governance.risk_classification import AISystemProfile
from tests.compliance._fixtures import (
    KEY, FakeConsent, FakeRestrictions, make_activity, structural_test_classifier,
)

SEED = 20261125


# ----------------------------------------------------------------------------------------------------------------------
# statistics helpers
# ----------------------------------------------------------------------------------------------------------------------
def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return (round(max(0.0, (c - m) / d), 4), round(min(1.0, (c + m) / d), 4))


def pct(samples: Sequence[float], q: float) -> float:
    s = sorted(samples)
    return s[min(len(s) - 1, int(q * len(s)))] if s else 0.0


def latency(samples_us: Sequence[float]) -> Dict[str, float]:
    return {"n": len(samples_us), "p50_us": round(pct(samples_us, .5), 1), "p95_us": round(pct(samples_us, .95), 1),
            "p99_us": round(pct(samples_us, .99), 1), "mean_us": round(statistics.mean(samples_us), 1) if samples_us else 0.0}


def confusion(rows: Sequence[Tuple[bool, bool]]) -> Dict[str, Any]:
    """rows = (must_restrict, was_restricted)."""
    tp = sum(1 for t, p in rows if t and p)
    fn = sum(1 for t, p in rows if t and not p)
    fp = sum(1 for t, p in rows if not t and p)
    tn = sum(1 for t, p in rows if not t and not p)
    pos, neg = tp + fn, fp + tn
    return {
        "n": len(rows), "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "recall": round(tp / pos, 4) if pos else None, "recall_ci95": wilson(tp, pos),
        "false_negative_rate": round(fn / pos, 4) if pos else None, "fnr_ci95": wilson(fn, pos),
        "specificity": round(tn / neg, 4) if neg else None,
        "false_positive_rate": round(fp / neg, 4) if neg else None, "fpr_ci95": wilson(fp, neg),
        "precision": round(tp / (tp + fp), 4) if (tp + fp) else None,
    }


# ----------------------------------------------------------------------------------------------------------------------
# environment + corpus
# ----------------------------------------------------------------------------------------------------------------------
class Env:
    def __init__(self, tmp: str):
        GovernanceCore.reset_shared()
        self.consent, self.restrictions = FakeConsent(), FakeRestrictions()
        self.core = GovernanceCore(tmp, master_key=KEY, consent=self.consent, restrictions=self.restrictions)
        self.core.classification.attach_classifier(structural_test_classifier)
        a = self.core.activities
        std_ops = (Operation.PROCESS, Operation.STORE, Operation.RETRIEVE, Operation.TRANSFER, Operation.SHARE)
        a.register(make_activity("act-1", operations=std_ops, transfers=(
            {"country": "JP", "mechanism": "adequacy", "ref": "ec-2019-419"},), recipients=("processor-a",)), actor="dpo")
        a.register(make_activity("act-us", operations=std_ops, transfers=({"country": "US", "mechanism": "adequacy", "ref": "dpf", "dpf_certified": True},)), actor="dpo")
        a.register(make_activity("act-us-bad", operations=std_ops, transfers=({"country": "US", "mechanism": "adequacy", "ref": "dpf"},)), actor="dpo")
        a.register(make_activity("act-consent", basis=LawfulBasis.CONSENT, purposes=("personalisation",)), actor="dpo")
        a.register(make_activity("act-learn", operations=(Operation.PROCESS, Operation.LEARN), allows_learning=True), actor="dpo")
        admin = self.core.identity.verify(self.core.identity.issue("profile-admin", None, actor_type="human", roles=("ai_governance_officer",), ttl_seconds=3600))
        self.core.profiles.register(AISystemProfile("sys-min", "assistant", intended_purpose="drafting"), admin, "benchmark")
        self.core.profiles.register(AISystemProfile("sys-scoring", "scoring", intended_purpose="test"), admin, "benchmark")
        self._i = 0

    def ident(self, tenant: Optional[str] = "tenant-a", *, now: Optional[float] = None, ttl: float = 3600.0):
        self._i += 1
        return self.core.identity.verify(self.core.identity.issue(f"svc-{self._i}", tenant, actor_type="service", ttl_seconds=ttl, now=now), now=now)

    def base(self, rng: random.Random, **over: Any) -> ProcessingContext:
        d: Dict[str, Any] = dict(operation=Operation.PROCESS, purpose="document_editing", data_class=DataClass.PERSONAL, activity_id="act-1",
                                 subject_ids=(f"subj-{rng.randrange(10**6)}",), tenant_id="tenant-a", identity=self.ident(),
                                 resource=f"res-{rng.randrange(10**9)}")
        d.update(over)
        return ProcessingContext(**d)

    def nonpersonal(self, rng: random.Random, payload: Any = None, *, single_use: bool = False, **over: Any) -> ProcessingContext:
        payload = payload if payload is not None else {"value": rng.randrange(10**6), "n": rng.random()}
        ident = over.pop("identity", None) or self.ident()
        token = self.core.classification.classify(payload, purpose="analysis", tenant_id=over.get("tenant_id"), identity=ident, resource=over.get("resource"),
                                                  single_use=single_use)
        return ProcessingContext(Operation.PROCESS, "analysis", data_class=DataClass.NON_PERSONAL, classification_evidence=token,
                                 data_digest=self.core.classification.digest(payload), identity=ident, **over)


@dataclass(frozen=True)
class Case:
    name: str
    category: str
    must_restrict: bool
    reference: str  # the rule that fixes the expected outcome
    build: Callable[[Env, random.Random], ProcessingContext]


def _corpus() -> List[Case]:
    C = Case
    restrict = True
    def consent_ctx(env, rng, **over):
        return env.base(rng, activity_id="act-consent", purpose="personalisation", **over)

    def withdrawn(env, rng):
        c = consent_ctx(env, rng)
        env.consent.withdraw(c.subject_ids[0], c.purpose)
        return c

    def granted(env, rng, **extra):
        c = consent_ctx(env, rng, **({"subject_is_child": True} if extra.get("child") else {}))
        env.consent.grant(c.subject_ids[0], c.purpose, **({"parental_authorisation": True} if extra.get("parental") else {}))
        return c

    def child_no_parent(env, rng):
        c = consent_ctx(env, rng, subject_is_child=True)
        env.consent.grant(c.subject_ids[0], c.purpose)
        return c

    def restricted(env, rng):
        c = env.base(rng)
        env.restrictions.state[c.subject_ids[0]] = {"restricted": True}
        return c

    def objected(env, rng):
        c = env.base(rng)
        env.restrictions.state[c.subject_ids[0]] = {"objected_purposes": [c.purpose]}
        return c

    def erasure(env, rng):
        c = env.base(rng)
        env.restrictions.state[c.subject_ids[0]] = {"erasure_pending": True}
        return c

    def replay(env, rng):
        c = env.nonpersonal(rng, single_use=True)
        env.core.authorize(c)  # the legitimate first use consumes the single-use evidence
        return c

    def reuse_same_payload(env, rng):
        c = env.nonpersonal(rng)  # reusable evidence is bound to the payload digest and its TTL, so the same payload may reuse it
        env.core.authorize(c)
        return c

    def other_payload(env, rng):
        ident = env.ident()
        token = env.core.classification.classify({"v": 1}, purpose="analysis", tenant_id="tenant-a", identity=ident)
        return ProcessingContext(Operation.PROCESS, "analysis", data_class=DataClass.NON_PERSONAL, classification_evidence=token,
                                 data_digest=env.core.classification.digest({"name": "Jan Jansen", "bsn": "x"}), identity=ident)

    def expired(env, rng):
        t = time.time() - 7200
        return env.base(rng, identity=env.ident(now=t, ttl=60.0))

    return [
        # ---- must restrict -------------------------------------------------------------------------------------------
        C("no_registered_activity", "lawful_basis", restrict, "GDPR Art. 6/30", lambda e, r: e.base(r, activity_id="nope")),
        C("purpose_not_in_activity", "purpose_limitation", restrict, "GDPR Art. 5(1)(b)", lambda e, r: e.base(r, purpose="advertising")),
        C("operation_not_in_activity", "purpose_limitation", restrict, "GDPR Art. 5(1)(b)", lambda e, r: e.base(r, operation=Operation.EXPORT)),
        C("unregistered_recipient", "recipients", restrict, "GDPR Art. 28/30", lambda e, r: e.base(r, recipients=("rogue-processor",))),
        C("special_category_without_art9", "special_category", restrict, "GDPR Art. 9", lambda e, r: e.base(r, data_class=DataClass.SPECIAL_CATEGORY)),
        C("tenant_mismatch", "isolation", restrict, "GDPR Art. 32", lambda e, r: e.base(r, tenant_id="tenant-b")),
        C("expired_identity", "identity", restrict, "GDPR Art. 32", expired),
        C("missing_identity", "identity", restrict, "GDPR Art. 32", lambda e, r: e.base(r, identity=None)),
        C("unclassified_payload", "classification", restrict, "GDPR Art. 4(1)/5(2)", lambda e, r: e.base(r, data_class=DataClass.UNCLASSIFIED)),
        C("attack_caller_claims_nonpersonal", "classification_attack", restrict, "GOV-P0-01",
          lambda e, r: e.base(r, data_class=DataClass.NON_PERSONAL, classification_verified=True)),
        C("attack_evidence_for_other_payload", "classification_attack", restrict, "GOV-P0-01", other_payload),
        C("attack_single_use_evidence_replay", "classification_attack", restrict, "GOV-P0-01 (single-use evidence)", replay),
        C("prohibited_practice", "ai_act", restrict, "AI Act Art. 5",
          lambda e, r: e.nonpersonal(r, ai_system_id="sys-scoring", ai_practices=("social_scoring",))),
        C("ai_system_unregistered", "ai_act", restrict, "AI Act Art. 6/49", lambda e, r: e.nonpersonal(r, ai_system_id="ghost-system")),
        C("consent_withdrawn", "consent", restrict, "GDPR Art. 7(3)", withdrawn),
        C("consent_missing", "consent", restrict, "GDPR Art. 7", lambda e, r: consent_ctx(e, r)),
        C("child_without_parental_authorisation", "consent", restrict, "GDPR Art. 8; UAVG art. 5", child_no_parent),
        C("restriction_active", "data_subject_rights", restrict, "GDPR Art. 18", restricted),
        C("objection_to_purpose", "data_subject_rights", restrict, "GDPR Art. 21", objected),
        C("erasure_pending", "data_subject_rights", restrict, "GDPR Art. 17", erasure),
        C("transfer_without_mechanism", "transfers", restrict, "GDPR Art. 44-46", lambda e, r: e.base(r, operation=Operation.TRANSFER, destination_country="IN")),
        C("us_transfer_recipient_not_certified", "transfers", restrict, "GDPR Art. 45 (DPF)",
          lambda e, r: e.base(r, activity_id="act-us-bad", operation=Operation.TRANSFER, destination_country="US")),
        C("disclosure_destination_unknown", "transfers", restrict, "GDPR Art. 44", lambda e, r: e.base(r, operation=Operation.SHARE)),
        C("automated_decision_no_art22_basis", "automated_decisions", restrict, "GDPR Art. 22",
          lambda e, r: e.base(r, automated_decision=True, significant_effect=True)),
        C("learning_not_authorised", "learning", restrict, "GDPR Art. 5(1)(b)/6(4)", lambda e, r: e.base(r, operation=Operation.LEARN)),
        # ---- must proceed --------------------------------------------------------------------------------------------
        C("routine_processing", "lawful_routine", False, "GDPR Art. 6(1)(b)", lambda e, r: e.base(r)),
        C("store_within_activity", "lawful_routine", False, "GDPR Art. 6(1)(b)", lambda e, r: e.base(r, operation=Operation.STORE)),
        C("retrieve_within_activity", "lawful_routine", False, "GDPR Art. 6(1)(b)", lambda e, r: e.base(r, operation=Operation.RETRIEVE)),
        C("registered_recipient", "lawful_routine", False, "GDPR Art. 28", lambda e, r: e.base(r, recipients=("processor-a",))),
        C("nonpersonal_with_signed_evidence", "nonpersonal", False, "GDPR Art. 2(1) scope", lambda e, r: e.nonpersonal(r)),
        C("reusable_evidence_same_payload", "nonpersonal", False, "GOV-P0-01 (evidence is bound to the payload digest, not to one call)", reuse_same_payload),
        C("nonpersonal_registered_ai_system", "nonpersonal", False, "AI Act Art. 6", lambda e, r: e.nonpersonal(r, ai_system_id="sys-min")),
        C("transfer_within_eea", "transfers_ok", False, "GDPR Art. 44 (no restricted transfer)", lambda e, r: e.base(r, operation=Operation.TRANSFER, destination_country="DE")),
        C("transfer_adequate_country", "transfers_ok", False, "GDPR Art. 45", lambda e, r: e.base(r, operation=Operation.TRANSFER, destination_country="JP")),
        C("us_transfer_certified_recipient", "transfers_ok", False, "GDPR Art. 45 (DPF)",
          lambda e, r: e.base(r, activity_id="act-us", operation=Operation.TRANSFER, destination_country="US")),
        C("consent_granted", "consent_ok", False, "GDPR Art. 7", lambda e, r: granted(e, r)),
        C("child_with_parental_authorisation", "consent_ok", False, "GDPR Art. 8", lambda e, r: granted(e, r, child=True, parental=True)),
        C("learning_authorised", "learning_ok", False, "GDPR Art. 6(4)", lambda e, r: e.base(r, activity_id="act-learn", operation=Operation.LEARN)),
    ]


def run_corpus(env: Env, variants: int, rng: random.Random) -> Dict[str, Any]:
    corpus = _corpus()
    rows: List[Tuple[bool, bool]] = []
    per_case: Dict[str, List[Tuple[bool, bool]]] = {}
    per_cat: Dict[str, List[Tuple[bool, bool]]] = {}
    lat: List[float] = []
    lat_by: Dict[str, List[float]] = {"must_restrict": [], "must_proceed": []}
    mismatches: List[Dict[str, Any]] = []
    for case in corpus:
        for v in range(variants):
            ctx = case.build(env, rng)
            t = time.perf_counter()
            d = env.core.authorize(ctx)
            us = (time.perf_counter() - t) * 1e6
            restricted = not d.decision.permits_execution
            rows.append((case.must_restrict, restricted))
            per_case.setdefault(case.name, []).append((case.must_restrict, restricted))
            per_cat.setdefault(case.category, []).append((case.must_restrict, restricted))
            lat.append(us)
            lat_by["must_restrict" if case.must_restrict else "must_proceed"].append(us)
            if restricted != case.must_restrict and len(mismatches) < 50:
                mismatches.append({"case": case.name, "reference": case.reference, "expected": "restrict" if case.must_restrict else "proceed",
                                   "got": d.decision.value, "rules": [r.rule_id for r in d.results if not r.decision.permits_execution]})
    return {
        "variants_per_scenario": variants, "scenarios": len(corpus),
        "overall": confusion(rows),
        "by_category": {k: confusion(v) for k, v in sorted(per_cat.items())},
        "scenarios_with_any_mismatch": sorted({m["case"] for m in mismatches}),
        "mismatches": mismatches,
        "latency_authorize": {"all": latency(lat), "must_restrict": latency(lat_by["must_restrict"]), "must_proceed": latency(lat_by["must_proceed"])},
        "no_governance_baseline": confusion([(t, False) for t, _ in rows]),  # what an ungoverned system does: everything proceeds
    }


# ----------------------------------------------------------------------------------------------------------------------
# adaptive evaluation (disjoint train / test; poisoning with and without the defences)
# ----------------------------------------------------------------------------------------------------------------------
def adaptive_eval(tmp: str, n_train: int, n_test: int, seed: int, fp_ceiling: float = 0.2, poisoning: bool = True) -> Dict[str, Any]:
    """Identical data for every call with the same seed. ``fp_ceiling`` is the operator's tolerance for over-escalation (default 0.2)."""
    rng = random.Random(seed)
    GovernanceCore.reset_shared()
    core = GovernanceCore(tmp, master_key=KEY, config={"adaptive": {"mode": "enforce", "min_samples": 15, "min_validation": 20,
                                                                       "max_escalation_rate": 0.7, "max_false_positive_rate": fp_ceiling,
                                                                       "max_per_source_per_hour": 10**6, "min_live_outcomes": 10**9}})
    core.classification.attach_classifier(structural_test_classifier)
    ad = core.adaptive
    ident = lambda role, name: core.identity.verify(core.identity.issue(name, "tenant-a", actor_type="human", roles=(role,), ttl_seconds=10**6))
    admin = core.identity.verify(core.identity.issue("gov-admin", None, actor_type="human", roles=("governance_admin",), ttl_seconds=10**6))
    reviewers = [ident("oversight_reviewer", f"rev-{i}") for i in range(4)]
    attacker = ident("oversight_reviewer", "rev-attacker")
    ops = [Operation.PROCESS, Operation.STORE, Operation.RETRIEVE, Operation.SHARE, Operation.EXPORT, Operation.TRANSFER]
    classes = [DataClass.PERSONAL, DataClass.SPECIAL_CATEGORY]
    buckets = [(o, c, d) for o in ops for c in classes for d in (None, "IN")]
    hidden = {b: rng.choice([0.03, 0.05, 0.1, 0.85, 0.9, 0.95]) for b in buckets}  # hidden truth: how often a human really needed to look

    def draw(i: int):
        b = rng.choice(buckets)
        ctx = ProcessingContext(b[0], "p", data_class=b[1], destination_country=b[2], resource=f"x{i}-{rng.random()}", tenant_id="tenant-a")
        return core.engine.resolve(ctx), rng.random() < hidden[b]

    def record(ctx, needed, who, flip=0.05):
        label = "should_escalate" if (needed != (rng.random() < flip)) else "routine"
        try:
            core.record_outcome(ctx, label, who, evidence_ref="benchmark")
        except Exception:
            pass

    train = [draw(i) for i in range(n_train)]
    test = [draw(n_train + i) for i in range(n_test)]
    for ctx, needed in train:
        record(ctx, needed, rng.choice(reviewers))
    result: Dict[str, Any] = {"train": n_train, "test": n_test, "label_noise": 0.05, "note": "synthetic: hidden per-bucket review need; reviewer noise 5%"}

    def evaluate(label: str) -> Dict[str, Any]:
        rows = []
        for ctx, needed in test:
            rows.append((needed, bool(ad.rule(ctx))))
        return {label: confusion(rows)}

    result["before_mandatory_floor_only"] = evaluate("no_adaptive")["no_adaptive"]
    p = ad.propose(admin)
    result["candidate_validation"] = p["validation"]
    if p["validation"]["passed"]:
        ad.promote(p["version"], admin, "benchmark candidate passed validation", expected_digest=p["digest"])
    result["after_adaptive_enforce"] = evaluate("adaptive")["adaptive"]
    result["adaptive_state"] = ad.status()["state"]

    # poisoning: attacker floods inverse labels on the buckets that really need review (wants them to stop escalating)
    def poisoned(defences: bool, tag: str) -> Dict[str, Any]:
        sub = f"{tmp}/{tag}"
        GovernanceCore.reset_shared()
        c2 = GovernanceCore(sub, master_key=KEY, config={"adaptive": {"mode": "enforce", "min_samples": 15, "min_validation": 20, "max_escalation_rate": 0.7,
                                                                        "max_false_positive_rate": fp_ceiling, "max_per_source_per_hour": 10**6,
                                                                        "min_live_outcomes": 10**9}})
        c2.classification.attach_classifier(structural_test_classifier)
        if not defences:
            c2.adaptive._cfg.update(max_per_source=10**9, min_sources=1)
        a2 = c2.adaptive
        revs = [c2.identity.verify(c2.identity.issue(f"r{i}", "tenant-a", actor_type="human", roles=("oversight_reviewer",), ttl_seconds=10**6)) for i in range(4)]
        atk = c2.identity.verify(c2.identity.issue("attacker", "tenant-a", actor_type="human", roles=("oversight_reviewer",), ttl_seconds=10**6))
        adm = c2.identity.verify(c2.identity.issue("adm", None, actor_type="human", roles=("governance_admin",), ttl_seconds=10**6))
        hot = {b for b, p_ in hidden.items() if p_ > 0.5}
        for ctx, needed in train:
            r = c2.engine.resolve(ProcessingContext(ctx.operation, "p", data_class=ctx.effective_data_class, destination_country=ctx.destination_country,
                                                    resource=ctx.resource, tenant_id="tenant-a"))
            label = "should_escalate" if needed else "routine"
            c2.record_outcome(r, label, rng.choice(revs), evidence_ref="b")
        flood = 0
        per_bucket = max(60, int(3 * n_train / len(buckets)))  # the attacker out-writes the legitimate reviewers 3:1 per bucket
        for b in hot:
            for k in range(per_bucket):
                ctx = ProcessingContext(b[0], "p", data_class=b[1], destination_country=b[2], resource=f"atk-{tag}-{b}-{k}", tenant_id="tenant-a")
                # mostly-opposite labels, mixed enough to stay under the label-flip detector when it is on
                c2.record_outcome(c2.engine.resolve(ctx), "should_escalate" if k % 10 < 3 else "routine", atk, evidence_ref="flood")
                flood += 1
        pp = a2.propose(adm)
        if pp["validation"]["passed"]:
            a2.promote(pp["version"], adm, "poisoning experiment promote", expected_digest=pp["digest"])
        rows = [(needed, bool(a2.rule(ctx))) for ctx, needed in test]
        out = {"flood_outcomes": flood, "candidate_passed_validation": pp["validation"]["passed"], "failed_gates": pp["validation"]["failed_gates"],
               "metrics": confusion(rows)}
        c2.close()
        return out

    result["fp_ceiling"] = fp_ceiling
    if poisoning:
        result["poisoning_with_defences"] = poisoned(True, "def")
        result["poisoning_defences_disabled"] = poisoned(False, "nodef")
    core.close()
    return result


# ----------------------------------------------------------------------------------------------------------------------
# triage simulation
# ----------------------------------------------------------------------------------------------------------------------
def triage_sim(n: int, rng: random.Random) -> Dict[str, Any]:
    """Single reviewer, 110% load, 900 s validity. FIFO vs. ``decision_intel.rank``. Simulation only."""
    ttl, arrival, service = 900.0, 10.0, 9.0  # one item per 10 s; one review per 9 s of reviewer time... overloaded via bursts below
    items = []
    t = 0.0
    for i in range(n):
        t += rng.expovariate(1 / arrival) * (0.55 if (i // 40) % 3 == 1 else 1.0)  # periodic bursts
        urgent = rng.random() < 0.12
        triage = ({"source": rng.choice(["GOV-COORD-02", "GOV-HO-14"]), "sensitive": rng.random() < 0.6, "personal": True, "significant": rng.random() < 0.5} if urgent
                  else {"source": rng.choice(["GOV-ADP-01", "GOV-CND-02"]), "sensitive": False, "personal": rng.random() < 0.5, "significant": False})
        items.append({"approval_id": f"i{i}", "level": "approval", "created_at": t, "expires_at": t + ttl, "context_fingerprint": f"i{i}", "triage": triage, "urgent": urgent})

    def simulate(policy: str) -> Dict[str, Any]:
        clock, idx, queue, done = 0.0, 0, [], {}
        while idx < len(items) or queue:
            while idx < len(items) and items[idx]["created_at"] <= clock:
                queue.append(items[idx]); idx += 1
            queue = [q for q in queue if q["expires_at"] > clock or done.setdefault(q["approval_id"], ("expired", None)) is None]
            if not queue:
                clock = items[idx]["created_at"] if idx < len(items) else clock
                continue
            if policy == "fifo":
                pick = min(queue, key=lambda q: (q["created_at"], q["approval_id"]))
            else:
                best = rank(queue, now=clock)[0].approval_id
                pick = next(q for q in queue if q["approval_id"] == best)
            queue.remove(pick)
            done[pick["approval_id"]] = ("reviewed", clock - pick["created_at"])
            clock += rng_service(pick)
        res: Dict[str, Any] = {}
        for label, flag in (("urgent", True), ("routine", False)):
            grp = [it for it in items if it["urgent"] is flag]
            waits = [done[it["approval_id"]][1] for it in grp if done[it["approval_id"]][0] == "reviewed"]
            expired = sum(1 for it in grp if done[it["approval_id"]][0] == "expired")
            res[label] = {"n": len(grp), "expired_unreviewed": expired, "expired_rate": round(expired / len(grp), 4) if grp else 0.0,
                          "wait_s": {"p50": round(pct(waits, .5), 1), "p95": round(pct(waits, .95), 1), "p99": round(pct(waits, .99), 1)}}
        return res

    srv = random.Random(SEED + 1)
    svc_times = {it["approval_id"]: srv.uniform(0.7, 1.3) * service for it in items}

    def rng_service(pick):
        return svc_times[pick["approval_id"]]

    return {"assumptions": {"validity_s": ttl, "mean_arrival_s": arrival, "mean_service_s": service, "urgent_share": 0.12,
                            "bursts": "arrival rate x1.8 during every third block of 40", "caveat": "simulation; urgency labels are generator-defined"},
            "fifo": simulate("fifo"), "priority": simulate("priority")}


# ----------------------------------------------------------------------------------------------------------------------
def main(smoke: bool = False) -> Dict[str, Any]:
    rng = random.Random(SEED)
    out: Dict[str, Any] = {"seed": SEED, "mode": "smoke" if smoke else "full"}
    with tempfile.TemporaryDirectory() as tmp:
        env = Env(tmp + "/corpus")
        out["corpus"] = run_corpus(env, 3 if smoke else 40, rng)
        env.core.close()
        n_tr, n_te = (400, 200) if smoke else (3000, 1500)
        # Every seed is reported, passing or not: the validation gate legitimately rejects some random worlds.
        seeds = [SEED + k for k in range(2 if smoke else 5)]
        out["adaptive_runs_default_gates"] = [dict(seed=sd, **adaptive_eval(f"{tmp}/a{k}", n_tr, n_te, sd, fp_ceiling=0.2, poisoning=(k < 1)))
                                              for k, sd in enumerate(seeds)]
    out["triage"] = triage_sim(300 if smoke else 4000, random.Random(SEED + 2))
    out["environment"] = {"python": sys.version.split()[0], "platform": platform.platform(), "machine": platform.machine(),
                          "note": "latency figures are for this machine only; sqlite WAL"}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--json", help="write full JSON here")
    args = ap.parse_args()
    res = main(args.smoke)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2)
    print(json.dumps(res, indent=2))
    c = res["corpus"]["overall"]
    if c["fn"] or c["fp"]:
        print(f"\nDISAGREEMENTS: fn={c['fn']} fp={c['fp']} (see corpus.mismatches)", file=sys.stderr)
        sys.exit(1)
