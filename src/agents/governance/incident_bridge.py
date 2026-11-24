"""Bridge from Handler / Observability events into the unified :class:`IncidentRegistry` (assignment section 12).

The Handler and Observability agents publish failure/incident records (Observability writes
``<prefix>:latest_incident`` into SharedMemory). This bridge normalises such records, decides - with documented
deterministic markers - whether they are governance-relevant, and opens an incident (state DETECTED) so the GDPR
Art. 33 clock (aware_at) and AI Act reporting deadlines start. It NEVER notifies an authority; submission stays a
recorded human act in the registry. An event may carry ``governance_kind`` to be authoritative; otherwise markers apply.

Duplicates (same source id, or same normalised fingerprint) are collapsed and persisted across restarts.
"""
from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Callable, Dict, Iterable, Mapping, Optional

from .incident_registry import IncidentKind, IncidentRegistry
from .utils.diagnostics import redact_text

_BREACH_MARKERS = ("data_breach", "personal_data", "pii_leak", "pii", "exfiltrat", "unauthorized_access", "unauthorised_access",
                   "tenant_isolation", "cross_tenant", "credential_leak", "secret_leak", "integrity_check_failed", "tamper")
_AI_MARKERS = ("serious_incident", "harm_to_health", "fundamental_rights", "critical_infrastructure_disruption", "safety_incident")
_PROHIBITED_MARKERS = ("prohibited_practice", "art5", "social_scoring", "manipulative")
_GOV_MARKERS = ("governance_failure", "deletion_failed", "consent_violation", "policy_engine_error", "fail_closed_degraded", "erasure_incomplete")

_SCHEMA = "CREATE TABLE IF NOT EXISTS incident_intake(fingerprint TEXT PRIMARY KEY, incident_id TEXT NOT NULL, first_seen REAL NOT NULL, last_seen REAL NOT NULL, count INTEGER NOT NULL)"


def _flatten(event: Mapping[str, Any]) -> str:
    parts = []
    for key in ("category", "family", "error_type", "type", "kind", "tags", "labels", "summary", "description", "message", "incident_level", "severity", "level"):
        v = event.get(key)
        if isinstance(v, (list, tuple, set)):
            parts.extend(str(x) for x in v)
        elif v is not None:
            parts.append(str(v))
    return " ".join(parts).lower().replace("-", "_").replace(" ", "_")


def _event_key(raw: Any) -> Any:
    """SharedMemory reports the key as ``repr(key)`` (``"'a:b'"``); recover the real key without evaluating anything."""
    if isinstance(raw, str) and len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "'\"":
        return raw[1:-1]
    return raw


class IncidentIntake:
    def __init__(self, registry: IncidentRegistry, *, clock: Callable[[], float] = time.time, dedupe_window_seconds: float = 1800.0) -> None:
        self._reg, self._clock, self._window = registry, clock, dedupe_window_seconds
        self._reg._db.execute(_SCHEMA)
        self.ignored = 0

    def classify(self, event: Mapping[str, Any]) -> Optional[IncidentKind]:
        explicit = event.get("governance_kind")
        if explicit:
            try:
                return IncidentKind(str(explicit))
            except ValueError:
                return IncidentKind.GOVERNANCE_FAILURE  # unknown explicit kind: still surface it, never drop
        text = _flatten(event)
        for markers, kind in ((_PROHIBITED_MARKERS, IncidentKind.PROHIBITED_PRACTICE_ATTEMPT), (_BREACH_MARKERS, IncidentKind.PERSONAL_DATA_BREACH),
                              (_AI_MARKERS, IncidentKind.AI_SERIOUS_INCIDENT), (_GOV_MARKERS, IncidentKind.GOVERNANCE_FAILURE)):
            if any(m in text for m in markers):
                return kind
        return None

    def ingest(self, event: Mapping[str, Any], *, source: str = "observability") -> Optional[str]:
        """Return the (new or existing) incident id, or None when the event is not governance-relevant."""
        if not isinstance(event, Mapping) or not event:
            self.ignored += 1
            return None
        kind = self.classify(event)
        if kind is None:
            self.ignored += 1
            return None
        src_id = str(event.get("incident_id") or event.get("id") or "")
        description = redact_text(str(event.get("summary") or event.get("description") or event.get("message") or f"{kind.value} reported by {source}"))[:400]
        fp = hashlib.sha256(f"{kind.value}\x1f{src_id or description}".encode()).hexdigest()
        now = self._clock()
        row = self._reg._db.execute("SELECT incident_id,last_seen,count FROM incident_intake WHERE fingerprint=?", (fp,)).fetchone()
        if row and now - row[1] <= self._window:
            self._reg._db.execute("UPDATE incident_intake SET last_seen=?, count=count+1 WHERE fingerprint=?", (now, fp))
            return row[0]
        aware = event.get("detected_at") or event.get("timestamp")
        aware_at = float(aware) if isinstance(aware, (int, float)) and 0 < aware <= now else now
        count = event.get("affected_subject_count")
        iid = self._reg.detect(
            kind, reporter=source, description=description,
            affected_systems=tuple(str(s) for s in (event.get("affected_systems") or ())[:20]),
            affected_subject_count=int(count) if isinstance(count, int) and count >= 0 else None,
            evidence_refs=(f"{source}:{src_id}",) if src_id else (), aware_at=aware_at, now=now,
        )
        self._reg._db.execute("INSERT OR REPLACE INTO incident_intake VALUES(?,?,?,?,?)", (fp, iid, now, now, 1))
        return iid

    def attach(self, shared_memory: Any, patterns: Iterable[str] = ("*:latest_incident",), *, source: str = "observability") -> None:
        """Subscribe persistently to SharedMemory channels; callback failures are contained so publishers are never broken.

        The real ``SharedMemory`` notifies subscribers with a *change event* (``event_type``, ``key``, fingerprint), not the
        stored value, so the incident is read back with ``get(key)``. A plain mapping without ``event_type`` is treated as the
        incident itself (direct publishers, other pub/sub implementations). Real SharedMemory delivers callbacks on worker threads
        by default, so ingestion is asynchronous.
        """
        for pattern in patterns:
            def callback(payload, _src=source, _sm=shared_memory):
                try:
                    event = payload if isinstance(payload, Mapping) else {}
                    if "event_type" in event and "key" in event:
                        if event.get("event_type") not in ("put", "compare_and_swap"):
                            return
                        value = _sm.get(_event_key(event["key"]))
                        event = value if isinstance(value, Mapping) else {}
                    self.ingest(event, source=_src)
                except Exception:  # an intake failure must not break the publisher; the registry health check surfaces gaps
                    self.ignored += 1
            shared_memory.subscribe(pattern, callback, once=False)
