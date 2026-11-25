"""Decision intelligence and escalation optimisation (GOV-P2-01).

Scope, stated plainly
---------------------
This module never decides anything. It reads decisions and open human-review requests and helps humans spend their
limited review time where it matters:

* ``rank`` orders the open review queue by an explicit, additive, explainable priority (no learned weights): the reason
  a review is needed, the sensitivity of the data, significant automated decisions, the review level, and a rising urgency
  term so that nothing starves and requests are seen before they expire. Duplicates (same request fingerprint) are merged
  in the view, never in the store.
* ``analytics`` summarises recorded decisions and review outcomes: escalation rate, which rules drive it, review latency
  (P50/P95/P99), approval/rejection ratio, expiry (wasted escalation) rate, and "fast approval" rate (a rubber-stamping signal).

What was evaluated and deliberately NOT shipped: a decision cache. Consent and restriction state come from external
providers that do not announce changes, so a cached allow could outlive a withdrawal. Latency is measured in the
benchmarks; the engine evaluation cost does not justify that risk.

Priority is advisory ordering only: it cannot approve, skip or reorder anything that is bound to an authorisation check.
"""

from __future__ import annotations

import json
import statistics

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

__all__ = ["RankedReview", "SOURCE_WEIGHTS", "analytics", "rank", "triage_hints"]

# Why a human is needed (higher = more urgent). Rule ids are those in policy_resolution.BASES.
SOURCE_WEIGHTS: Mapping[str, int] = {
    "GOV-COORD-02": 40,  # a specialist agent raised a concern
    "GOV-HO-14": 30,  # mandatory human oversight (AI Act Art. 14 / GDPR Art. 22)
    "GOV-TRF-09": 30,  # Art. 49 transfer derogation
    "GOV-CND-02": 20,  # unmet mandatory pre-condition
    "GOV-ADP-01": 10,  # adaptive, advisory only
}
_LEVEL_WEIGHT = {"approval": 10, "review": 5, "notify": 0}
_SENSITIVE = {"special_category", "biometric", "criminal_offence"}
URGENCY_MAX = 30
EXPIRY_BOOST = 20


@dataclass(frozen=True)
class RankedReview:
    approval_id: str
    score: float
    rank: int
    components: Mapping[str, float]
    duplicates: Tuple[str, ...] = ()
    age_seconds: float = 0.0
    expires_in_seconds: float = 0.0


def triage_hints(decision: Any, ctx: Any = None) -> Dict[str, Any]:
    """Descriptive routing metadata to pass as ``request_approval(..., triage=...)``. Contains no payload and no identifiers."""
    res = getattr(decision, "resolution", None) or {}
    governing = list(res.get("governing") or [])
    review_rules = [r.rule_id for r in getattr(decision, "results", ()) if getattr(r.decision, "value", "") == "REVIEW_REQUIRED"]
    source = next((g for g in governing if g in SOURCE_WEIGHTS), None) or next((g for g in review_rules if g in SOURCE_WEIGHTS), "")
    data_class = getattr(decision, "effective_data_class", "") or ""
    return {
        "source": source,
        "data_class": data_class,
        "sensitive": data_class in _SENSITIVE,
        "personal": data_class in {"personal", "pseudonymised", "unclassified"} or data_class in _SENSITIVE,
        "significant": bool(getattr(ctx, "significant_effect", False) and getattr(ctx, "automated_decision", False)),
    }


def _components(item: Mapping[str, Any], now: float) -> Dict[str, float]:
    hints = item.get("triage") or {}
    created = now if item.get("created_at") is None else float(item["created_at"])
    expires = now if item.get("expires_at") is None else float(item["expires_at"])
    span = max(1.0, expires - created)
    frac = min(1.0, max(0.0, (now - created) / span))
    remaining = max(0.0, expires - now)
    comp = {
        "source": float(SOURCE_WEIGHTS.get(str(hints.get("source", "")), 15)),  # unknown source: middling, not ignored
        "sensitivity": 25.0 if hints.get("sensitive") else (10.0 if hints.get("personal") else 0.0),
        "significant_automated_decision": 20.0 if hints.get("significant") else 0.0,
        "level": float(_LEVEL_WEIGHT.get(str(item.get("level")), 0)),
        "urgency": round(URGENCY_MAX * frac * frac, 3),
        "about_to_expire": float(EXPIRY_BOOST) if remaining / span <= 0.10 else 0.0,
    }
    return comp


def rank(items: Iterable[Mapping[str, Any]], *, now: float) -> List[RankedReview]:
    """Rank open reviews (as returned by ``OversightManager.pending``). Deterministic: ties break on age, then id."""
    by_fp: Dict[str, List[Mapping[str, Any]]] = {}
    for it in items:
        by_fp.setdefault(str(it.get("context_fingerprint") or it["approval_id"]), []).append(it)
    scored: List[Tuple[float, float, str, Mapping[str, float], Tuple[str, ...], Mapping[str, Any]]] = []
    for group in by_fp.values():
        group = sorted(group, key=lambda g: (float(g.get("created_at") or 0), g["approval_id"]))
        lead, dups = group[0], tuple(g["approval_id"] for g in group[1:])
        comp = _components(lead, now)
        scored.append((sum(comp.values()), float(lead.get("created_at") or 0), lead["approval_id"], comp, dups, lead))
    scored.sort(key=lambda t: (-t[0], t[1], t[2]))
    return [RankedReview(aid, round(score, 3), i + 1, comp, dups, round(now - float(lead.get("created_at") if lead.get("created_at") is not None else now), 3),
                         round(max(0.0, float(lead.get("expires_at") if lead.get("expires_at") is not None else now) - now), 3))
            for i, (score, _, aid, comp, dups, lead) in enumerate(scored)]


def _pct(values: Sequence[float], q: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(q * len(s)))] if s else 0.0


def analytics(core: Any) -> Dict[str, Any]:
    """Aggregate view over recorded decisions and review outcomes. Read-only."""
    dec = core._decisions.execute("SELECT decision, rules FROM decisions").fetchall()
    total = len(dec)
    by_decision: Dict[str, int] = {}
    by_rule: Dict[str, int] = {}
    for decision, rules in dec:
        by_decision[decision] = by_decision.get(decision, 0) + 1
        if decision != "ALLOW":
            for r in set(json.loads(rules)):
                by_rule[r] = by_rule.get(r, 0) + 1
    ov = core.oversight._db
    status = dict(ov.execute("SELECT status, COUNT(*) FROM approvals GROUP BY status").fetchall())
    requested = dict(ov.execute("SELECT approval_id, at FROM oversight_log WHERE event='requested'").fetchall())
    latencies: List[float] = []
    fast = approved = 0
    for approval_id, event, detail, at in ov.execute("SELECT approval_id, event, detail, at FROM oversight_log WHERE event IN ('approved','rejected')").fetchall():
        if approval_id in requested:
            latencies.append(at - requested[approval_id])
        if event == "approved":
            approved += 1
            fast += bool(json.loads(detail).get("fast"))
    decided = status.get("approved", 0) + status.get("rejected", 0) + status.get("consumed", 0) + status.get("revoked", 0)
    escalated = by_decision.get("REVIEW_REQUIRED", 0)
    return {
        "decisions": total,
        "by_decision": by_decision,
        "escalation_rate": round(escalated / total, 4) if total else 0.0,
        "top_restricting_rules": sorted(by_rule.items(), key=lambda kv: (-kv[1], kv[0]))[:10],
        "reviews": {
            "by_status": status,
            "expired_unreviewed_rate": round(status.get("expired", 0) / max(1, sum(status.values())), 4),
            "rejection_rate": round(status.get("rejected", 0) / decided, 4) if decided else 0.0,
            "fast_approval_rate": round(fast / approved, 4) if approved else 0.0,
            "latency_seconds": {"n": len(latencies), "p50": round(_pct(latencies, .5), 3), "p95": round(_pct(latencies, .95), 3),
                                "p99": round(_pct(latencies, .99), 3), "mean": round(statistics.mean(latencies), 3) if latencies else 0.0},
        },
    }
