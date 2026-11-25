"""Controlled adaptive governance (GOV-P1-04).

What adapts, and what cannot
----------------------------
Governance may learn *when to ask a human sooner*, nothing else. The only product of adaptation is the
advisory rule ``GOV-ADP-01`` which can emit ``REVIEW_REQUIRED`` (overridable by a verified human approval, like
any other review). Because aggregation is "most restrictive wins" this makes it structurally impossible for the
adaptive layer to:

* relax or override a mandatory control (it never emits ALLOW-class results and never touches BLOCK/QUARANTINE);
* change any mandatory rule, policy, profile, threshold or legal date (it owns no handle on them);
* affect a request in ``off`` mode, while ``suspended`` or ``tampered``, or when it errors (an adaptive failure
  falls back to the mandatory floor - the floor is the safe state, so this is the one place where "no result" is right).

Learning signal and poisoning defences
--------------------------------------
Only *verified outcomes* count: a post-hoc label ("this request did / did not warrant human review") recorded by a
verified identity holding an outcome role (human reviewers, DPO, the independent verification agent). Agents cannot
label their own requests: the recorder must be a different principal from ``requester``. Defences, all enforced in code:

1. one outcome per request fingerprint (replays are ignored, not double counted);
2. per-source rate limit and a per-source cap on how much one source can contribute to a bucket;
3. a bucket needs a minimum number of samples *and* distinct sources;
4. a source whose labels contradict every other source in a bucket is excluded from training (label-flip defence);
5. candidates are validated on a deterministic held-out split that is never used for fitting: they must not raise the
   false-negative rate, must stay under a false-positive and escalation-rate ceiling, and need a minimum validation size;
6. promotion is a separate, authorised, reasoned human act bound to the candidate's content digest; a live version whose
   false-negative rate on new outcomes drifts beyond tolerance is automatically suspended.

Known limit: a source that stays under every defence can still *degrade or block* adaptation (its capped rows also sit in the
validation fold, so a flood can push a legitimate candidate below a gate). That is a denial of the optimisation, not of safety:
the fallback is the mandatory floor, and no defence-evading input can relax a mandatory control or promote a model that the
gates reject. A robust, per-source-median aggregation would narrow this further and is not implemented.

Modes: ``off`` | ``shadow`` (default: measure what would have been escalated, escalate nothing) | ``enforce``.
Every version, promotion, rollback and suspension is appended to a hash-chained, HMAC-signed log (tamper => the
adaptive layer disables itself).

Stdlib only. Not a statistical claim: with small samples the validation gates simply refuse to promote.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import threading
import time

from enum import Enum
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .processing_context import Decision, ProcessingContext, RuleResult
from .utils.store import open_db

__all__ = ["AdaptiveError", "AdaptiveGovernance", "AdaptiveMode", "Outcome"]

_GENESIS = "0" * 64
_SCHEMA = """
CREATE TABLE IF NOT EXISTS outcomes(seq INTEGER PRIMARY KEY AUTOINCREMENT, fingerprint TEXT NOT NULL UNIQUE, bucket TEXT NOT NULL,
  label TEXT NOT NULL, source TEXT NOT NULL, evidence_ref TEXT NOT NULL, at REAL NOT NULL, fold INTEGER NOT NULL);
CREATE INDEX IF NOT EXISTS ix_outcomes_bucket ON outcomes(bucket);
CREATE TABLE IF NOT EXISTS versions(version INTEGER PRIMARY KEY, params TEXT NOT NULL, digest TEXT NOT NULL, validation TEXT NOT NULL,
  passed INTEGER NOT NULL, created_by TEXT NOT NULL, created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS log(seq INTEGER PRIMARY KEY AUTOINCREMENT, event TEXT NOT NULL, version INTEGER, digest TEXT, actor TEXT NOT NULL,
  reason TEXT NOT NULL, at REAL NOT NULL, prev_hash TEXT NOT NULL, entry_hash TEXT NOT NULL, signature TEXT NOT NULL);
"""
LABELS = ("should_escalate", "routine")


class AdaptiveError(Exception):
    def __init__(self, code: str, message: str = ""):
        super().__init__(f"{code}: {message}" if message else code)
        self.code = code


class AdaptiveMode(str, Enum):
    OFF = "off"
    SHADOW = "shadow"
    ENFORCE = "enforce"


class Outcome(str, Enum):
    ESCALATE = "should_escalate"
    ROUTINE = "routine"


def _wilson_lower(pos: int, n: int, z: float = 1.64) -> float:
    if n <= 0:
        return 0.0
    p = pos / n
    d = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return max(0.0, (centre - margin) / d)


class AdaptiveGovernance:
    def __init__(
        self,
        *,
        path: Any = ":memory:",
        key: Optional[bytes] = None,
        authority: Any = None,
        mode: str = "shadow",
        outcome_roles: Sequence[str] = ("oversight_reviewer", "dpo", "governance_admin", "safety_officer", "verification_agent"),
        promote_roles: Sequence[str] = ("ai_governance_officer", "dpo", "governance_admin"),
        bucket_of: Optional[Callable[[ProcessingContext], str]] = None,
        clock: Callable[[], float] = time.time,
        min_samples: int = 20,
        min_sources: int = 2,
        max_per_source: int = 10,
        max_per_source_per_hour: int = 200,
        escalate_threshold: float = 0.5,
        min_validation: int = 20,
        max_false_positive_rate: float = 0.2,
        max_escalation_rate: float = 0.3,
        drift_tolerance: float = 0.15,
        min_live_outcomes: int = 20,
    ):
        self._lock = threading.RLock()
        self._clock = clock
        self._key = key if key and len(key) >= 32 else None
        self._authority = authority
        self._outcome_roles = frozenset(outcome_roles)
        self._promote_roles = frozenset(promote_roles)
        self._bucket_of = bucket_of or default_bucket
        self._cfg = dict(min_samples=min_samples, min_sources=min_sources, max_per_source=max_per_source, rate=max_per_source_per_hour,
                         threshold=escalate_threshold, min_validation=min_validation, max_fp=max_false_positive_rate,
                         max_esc=max_escalation_rate, drift=drift_tolerance, min_live=min_live_outcomes)
        try:
            self.mode = AdaptiveMode(mode)
        except ValueError:
            self.mode = AdaptiveMode.OFF  # an unrecognised mode is never permissive: no adaptive effect at all
        self.stats: Dict[str, int] = {"evaluated": 0, "escalated": 0, "shadow_would_escalate": 0, "errors": 0, "outcomes": 0, "rejected_outcomes": 0}
        self._db = open_db(path)
        self._db.executescript(_SCHEMA)
        self._bad_pairs: List[str] = []
        self._live_model: Optional[Dict[str, Any]] = None
        self._live_version: Optional[int] = None
        self._state = "none"  # none | active | suspended | tampered
        self._reload()

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # ------------------------------------------------------------------ the one effect: an advisory escalation
    def rule(self, ctx: ProcessingContext, env: Any = None) -> List[RuleResult]:
        """Engine rule. Emits only REVIEW_REQUIRED; any internal failure yields no result (the mandatory floor stands)."""
        try:
            self.stats["evaluated"] += 1
            if self.mode is AdaptiveMode.OFF or self._state != "active" or self._live_model is None:
                return []
            bucket = self._bucket_of(ctx)
            if not self._escalates(self._live_model, bucket):
                return []
            if self.mode is AdaptiveMode.SHADOW:
                self.stats["shadow_would_escalate"] += 1
                return []
            self.stats["escalated"] += 1
            return [RuleResult("GOV-ADP-01", Decision.REVIEW_REQUIRED, f"adaptive_escalation:v{self._live_version}:{bucket}",
                               ("AI Act Art. 14",), overridable=True)]
        except Exception:
            self.stats["errors"] += 1
            return []

    # ------------------------------------------------------------------ verified outcomes
    def record_outcome(self, ctx: ProcessingContext, label: str, recorder: Any, *, evidence_ref: str, requester: Optional[str] = None) -> bool:
        """Record a verified post-hoc label for ``ctx``. Returns False for a duplicate fingerprint (not counted twice)."""
        if label not in LABELS:
            raise AdaptiveError("invalid_label", f"label must be one of {LABELS}")
        if not (evidence_ref or "").strip():
            raise AdaptiveError("evidence_required", "an outcome needs an evidence reference")
        who = self._verify(recorder, self._outcome_roles, "outcome")
        if requester is not None and requester == who:
            self.stats["rejected_outcomes"] += 1
            raise AdaptiveError("self_labelling", "a principal cannot label its own request")
        now = self._clock()
        fp = ctx.fingerprint()
        with self._lock:
            recent = self._db.execute("SELECT COUNT(*) FROM outcomes WHERE source=? AND at>?", (who, now - 3600)).fetchone()[0]
            if recent >= self._cfg["rate"]:
                self.stats["rejected_outcomes"] += 1
                raise AdaptiveError("rate_limited", "too many outcomes from this source in the last hour")
            fold = int(hashlib.sha256(fp.encode()).hexdigest(), 16) % 5  # fold 0 is the held-out validation set, never used to fit
            try:
                self._db.execute("INSERT INTO outcomes(fingerprint,bucket,label,source,evidence_ref,at,fold) VALUES (?,?,?,?,?,?,?)",
                                 (fp, self._bucket_of(ctx), label, who, evidence_ref.strip()[:200], now, fold))
            except Exception as exc:
                if "UNIQUE" in str(exc):
                    return False
                raise
            self.stats["outcomes"] += 1
        self._guard_drift()
        return True

    # ------------------------------------------------------------------ candidates, promotion, rollback
    def propose(self, actor: Any) -> Dict[str, Any]:
        """Fit a candidate from training folds and validate it on the held-out fold. Never goes live by itself."""
        who = self._verify(actor, self._promote_roles, "proposal")
        with self._lock:
            params, excluded = self._fit()
            validation = self._validate(params, set(tuple(e.split("|||", 1)) for e in self._bad_pairs))
            version = (self._db.execute("SELECT COALESCE(MAX(version),0) FROM versions").fetchone()[0]) + 1
            blob = json.dumps(params, sort_keys=True)
            digest = hashlib.sha256(blob.encode()).hexdigest()
            validation["excluded_sources"] = excluded
            self._db.execute("INSERT INTO versions(version,params,digest,validation,passed,created_by,created_at) VALUES (?,?,?,?,?,?,?)",
                             (version, blob, digest, json.dumps(validation, sort_keys=True), int(validation["passed"]), who, self._clock()))
            self._append("proposed", version, digest, who, "candidate fitted from verified outcomes")
        return {"version": version, "digest": digest, "validation": validation}

    def promote(self, version: int, actor: Any, reason: str, *, expected_digest: str) -> None:
        who = self._verify(actor, self._promote_roles, "promotion")
        self._need_reason(reason)
        with self._lock:
            self._require_intact()
            row = self._db.execute("SELECT digest, passed FROM versions WHERE version=?", (version,)).fetchone()
            if row is None:
                raise AdaptiveError("unknown_version")
            if row[0] != expected_digest:
                raise AdaptiveError("digest_mismatch", "the candidate changed since it was reviewed")
            if not row[1]:
                raise AdaptiveError("validation_failed", "a candidate that failed validation cannot be promoted")
            self._append("promoted", version, row[0], who, reason)
            self._reload()

    def rollback(self, actor: Any, reason: str, *, to_version: Optional[int] = None) -> Optional[int]:
        """Return to ``to_version`` (an earlier promoted, validated version) or, without one, to the mandatory floor."""
        who = self._verify(actor, self._promote_roles, "rollback")
        self._need_reason(reason)
        with self._lock:
            self._require_intact()
            if to_version is not None:
                ok = self._db.execute("SELECT 1 FROM log WHERE event='promoted' AND version=?", (to_version,)).fetchone()
                if not ok:
                    raise AdaptiveError("never_promoted", f"version {to_version} was never promoted")
                self._append("promoted", to_version, self._digest_of(to_version), who, f"rollback: {reason}")
            else:
                self._append("disabled", None, None, who, f"rollback to mandatory floor: {reason}")
            self._reload()
            return self._live_version

    def suspend(self, actor: Any, reason: str) -> None:
        who = self._verify(actor, self._promote_roles, "suspension")
        self._need_reason(reason)
        with self._lock:
            self._require_intact()
            self._append("suspended", self._live_version, None, who, reason)
            self._reload()

    # ------------------------------------------------------------------ introspection
    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {"mode": self.mode.value, "state": self._state, "live_version": self._live_version, "log_intact": self.verify(), "stats": dict(self.stats)}

    def history(self) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._db.execute("SELECT event,version,digest,actor,reason,at FROM log ORDER BY seq").fetchall()
        return [dict(zip(("event", "version", "digest", "actor", "reason", "at"), r)) for r in rows]

    def verify(self) -> bool:
        with self._lock:
            return self._verify_chain()

    # ------------------------------------------------------------------ internals: fit / validate
    def _rows(self, folds: str) -> List[Tuple[str, str, str, str]]:
        return self._db.execute(f"SELECT fingerprint,bucket,label,source FROM outcomes WHERE fold {folds} ORDER BY seq").fetchall()

    def _fit(self) -> Tuple[Dict[str, Any], List[str]]:
        rows = self._rows("!= 0")
        by_bucket: Dict[str, List[Tuple[str, str]]] = {}
        for _, bucket, label, source in rows:
            by_bucket.setdefault(bucket, []).append((label, source))
        excluded: List[str] = []
        self._bad_pairs = []
        buckets: Dict[str, Dict[str, Any]] = {}
        for bucket, items in by_bucket.items():
            sources = sorted({s for _, s in items})
            # label-flip defence: a source contradicting every other source on a bucket is not trusted for that bucket
            bad = set()
            if len(sources) >= 3:
                for s in sources:
                    mine = [l for l, src in items if src == s]
                    others = [l for l, src in items if src != s]
                    if len(mine) >= 5 and len(others) >= 5:
                        m, o = mine.count("should_escalate") / len(mine), others.count("should_escalate") / len(others)
                        if abs(m - o) >= 0.8:
                            bad.add(s)
            excluded.extend(f"{bucket}:{s}" for s in sorted(bad))
            self._bad_pairs.extend(f"{bucket}|||{s}" for s in sorted(bad))
            used: Dict[str, int] = {}
            n = pos = 0
            for label, src in items:
                if src in bad or used.get(src, 0) >= self._cfg["max_per_source"]:
                    continue  # per-source cap: no single source dominates a bucket
                used[src] = used.get(src, 0) + 1
                n += 1
                pos += label == "should_escalate"
            kept_sources = len(used)
            buckets[bucket] = {"n": n, "pos": pos, "sources": kept_sources, "lcb": round(_wilson_lower(pos, n), 6)}
        return {"threshold": self._cfg["threshold"], "min_samples": self._cfg["min_samples"], "min_sources": self._cfg["min_sources"], "buckets": buckets}, excluded

    def _escalates(self, model: Mapping[str, Any], bucket: str) -> bool:
        b = model["buckets"].get(bucket)
        return bool(b) and b["n"] >= model["min_samples"] and b["sources"] >= model["min_sources"] and b["lcb"] >= model["threshold"]

    def _validate(self, params: Mapping[str, Any], bad: Optional[set] = None) -> Dict[str, Any]:
        # The held-out fold gets the same poisoning defences as the fitting folds: excluded sources are dropped and no single
        # source can contribute more than ``max_per_source`` rows per bucket (otherwise a flood could also decide validation).
        held, seen = [], {}
        for row in self._rows("== 0"):
            if bad and (row[1], row[3]) in bad:
                continue
            key = (row[1], row[3])
            if seen.get(key, 0) >= self._cfg["max_per_source"]:
                continue
            seen[key] = seen.get(key, 0) + 1
            held.append(row)
        live = self._live_model

        def rates(model: Optional[Mapping[str, Any]]) -> Tuple[float, float, float]:
            pos = [r for r in held if r[2] == "should_escalate"]
            neg = [r for r in held if r[2] == "routine"]
            esc = [r for r in held if model is not None and self._escalates(model, r[1])]
            fn = sum(1 for r in pos if model is None or not self._escalates(model, r[1])) / len(pos) if pos else 0.0
            fp = sum(1 for r in neg if model is not None and self._escalates(model, r[1])) / len(neg) if neg else 0.0
            return fn, fp, (len(esc) / len(held) if held else 0.0)

        fn_b, fp_b, esc_b = rates(live)
        fn_c, fp_c, esc_c = rates(params)
        reasons: List[str] = []
        if len(held) < self._cfg["min_validation"]:
            reasons.append("validation_set_too_small")
        if fn_c > fn_b:
            reasons.append("false_negative_rate_worse")
        if fp_c > max(fp_b, self._cfg["max_fp"]):
            reasons.append("false_positive_rate_above_ceiling")
        if esc_c > self._cfg["max_esc"]:
            reasons.append("escalation_rate_above_ceiling")
        if not (fn_c < fn_b or fp_c < fp_b):
            reasons.append("no_measurable_improvement")
        return {"validation_size": len(held), "baseline": {"fn": fn_b, "fp": fp_b, "escalation": esc_b},
                "candidate": {"fn": fn_c, "fp": fp_c, "escalation": esc_c}, "passed": not reasons, "failed_gates": reasons}

    def _guard_drift(self) -> None:
        """Auto-suspend a live version whose false-negative rate on outcomes recorded since promotion exceeds validation + tolerance."""
        with self._lock:
            if self._state != "active" or self._live_model is None:
                return
            promoted_at = self._db.execute("SELECT at FROM log WHERE event='promoted' ORDER BY seq DESC LIMIT 1").fetchone()
            rows = self._db.execute("SELECT bucket,label FROM outcomes WHERE at>=?", (promoted_at[0],)).fetchall() if promoted_at else []
            if len(rows) < self._cfg["min_live"]:
                return
            pos = [r for r in rows if r[1] == "should_escalate"]
            if not pos:
                return
            fn = sum(1 for r in pos if not self._escalates(self._live_model, r[0])) / len(pos)
            val = json.loads(self._db.execute("SELECT validation FROM versions WHERE version=?", (self._live_version,)).fetchone()[0])
            if fn > val["candidate"]["fn"] + self._cfg["drift"]:
                self._append("suspended", self._live_version, None, "governance:auto", f"drift: live false-negative rate {fn:.2f} exceeds validated {val['candidate']['fn']:.2f}")
                self._reload()

    # ------------------------------------------------------------------ internals: chain, identity, state
    def _need_reason(self, reason: str) -> None:
        if len((reason or "").strip()) < 10:
            raise AdaptiveError("reason_required", "a documented reason (at least 10 characters) is required")

    def _verify(self, actor: Any, roles: frozenset, what: str) -> str:
        if self._authority is None or actor is None:
            raise AdaptiveError("not_authorised", f"{what} requires a verified identity")
        from .utils.identity import IdentityVerificationError

        try:
            ident = self._authority.verify(getattr(actor, "token", None), now=self._clock())
        except IdentityVerificationError as exc:
            raise AdaptiveError("not_authorised", f"identity invalid: {exc.code}") from exc
        if not (roles & set(ident.roles)):
            raise AdaptiveError("not_authorised", f"no role permitted to make this {what}")
        return ident.principal_id

    def _require_intact(self) -> None:
        if not self._verify_chain():
            self._state = "tampered"
            raise AdaptiveError("tampered", "the adaptive log failed verification")

    def _digest_of(self, version: int) -> str:
        return self._db.execute("SELECT digest FROM versions WHERE version=?", (version,)).fetchone()[0]

    def _entry_hash(self, prev: str, fields: Tuple[Any, ...]) -> str:
        return hashlib.sha256((prev + json.dumps(fields, sort_keys=True, default=str)).encode()).hexdigest()

    def _sign(self, entry_hash: str) -> str:
        return hmac.new(self._key, entry_hash.encode(), hashlib.sha256).hexdigest() if self._key else ""

    def _append(self, event: str, version: Optional[int], digest: Optional[str], actor: str, reason: str) -> None:
        row = self._db.execute("SELECT entry_hash FROM log ORDER BY seq DESC LIMIT 1").fetchone()
        prev = row[0] if row else _GENESIS
        at = self._clock()
        entry = self._entry_hash(prev, (event, version, digest, actor, reason, repr(at)))
        self._db.execute("INSERT INTO log(event,version,digest,actor,reason,at,prev_hash,entry_hash,signature) VALUES (?,?,?,?,?,?,?,?,?)",
                         (event, version, digest, actor, reason, at, prev, entry, self._sign(entry)))

    def _verify_chain(self) -> bool:
        prev = _GENESIS
        for event, version, digest, actor, reason, at, prev_hash, entry_hash, signature in self._db.execute(
                "SELECT event,version,digest,actor,reason,at,prev_hash,entry_hash,signature FROM log ORDER BY seq").fetchall():
            if prev_hash != prev or self._entry_hash(prev, (event, version, digest, actor, reason, repr(at))) != entry_hash:
                return False
            if self._key is not None and not hmac.compare_digest(signature, self._sign(entry_hash)):
                return False
            prev = entry_hash
        return True

    def _reload(self) -> None:
        """Derive the live version from the verified log. A broken chain means no adaptive effect, permanently until investigated."""
        if not self._verify_chain():
            self._state, self._live_model, self._live_version = "tampered", None, None
            return
        live: Optional[int] = None
        state = "none"
        for event, version in self._db.execute("SELECT event,version FROM log ORDER BY seq").fetchall():
            if event == "promoted":
                live, state = version, "active"
            elif event == "suspended":
                state = "suspended"
            elif event == "disabled":
                live, state = None, "none"
        self._live_version, self._state = live, state
        self._live_model = None
        if live is not None:
            row = self._db.execute("SELECT params, digest FROM versions WHERE version=?", (live,)).fetchone()
            if row is None or hashlib.sha256(row[0].encode()).hexdigest() != row[1]:
                self._state = "tampered"  # the stored model no longer matches the digest that was reviewed
                return
            self._live_model = json.loads(row[0])


def default_bucket(ctx: ProcessingContext) -> str:
    """Coarse, content-free request features (no identifiers, no payload)."""
    return "|".join((
        ctx.operation.value, ctx.effective_data_class.value, f"dest={'ext' if ctx.destination_country else 'none'}",
        f"auto={int(bool(ctx.automated_decision))}", f"ai={int(bool(ctx.ai_system_id))}",
    ))
