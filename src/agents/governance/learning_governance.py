"""Governed learning and model registry (EU AI Act Art. 10, 53; GDPR Art. 5, 17, 25; assignment section 10).

Principles enforced here:
* Learning is a distinct purpose. A training run is only registered when the caller presents an ALLOWED
  governance decision whose *purpose* is the learning purpose (never the serving purpose) for the same tenant.
* Dataset membership and training lineage are kept as salted digests - no personal identifiers.
* Erasure of a source record is NOT unlearning. When a subject is erased, every model trained on a dataset that
  contained them gets an *open impact*. An impact is only closed with recorded remediation evidence
  (retrained without the subject, independently verified unlearning, or a documented DPO exemption). The registry
  never marks anything as unlearned on its own, and an erasure request stays "incomplete" while impacts are open.
* A model cannot be approved with open impacts, a prohibited classification, or missing GPAI documentation.
Stdlib only.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import time
from threading import RLock
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from .processing_context import Decision
from .risk_classification import RiskClassification, RiskTier
from .utils.store import open_db

LEARNING_PURPOSES = frozenset({"model_training", "model_fine_tuning", "model_evaluation"})
RESOLUTIONS = ("retrained_without_subject", "unlearning_verified", "exempt_documented", "not_personal_data_in_model")
SYSTEMIC_RISK_FLOPS = 1e25

_SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets(dataset_id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, purpose TEXT NOT NULL,
  activity_id TEXT NOT NULL, source_decision TEXT NOT NULL, created_by TEXT NOT NULL, created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS dataset_members(dataset_id TEXT NOT NULL, subject_digest TEXT NOT NULL,
  PRIMARY KEY(dataset_id, subject_digest));
CREATE INDEX IF NOT EXISTS ix_members_digest ON dataset_members(subject_digest);
CREATE TABLE IF NOT EXISTS models(model_id TEXT NOT NULL, version TEXT NOT NULL, tenant_id TEXT, system_id TEXT NOT NULL,
  status TEXT NOT NULL, gpai INTEGER NOT NULL DEFAULT 0, training_flops REAL, docs TEXT NOT NULL DEFAULT '{}',
  approved_by TEXT, updated_at REAL NOT NULL, PRIMARY KEY(model_id, version));
CREATE TABLE IF NOT EXISTS training_runs(run_id TEXT PRIMARY KEY, model_id TEXT NOT NULL, version TEXT NOT NULL,
  tenant_id TEXT NOT NULL, purpose TEXT NOT NULL, decision_ref TEXT NOT NULL, started_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS run_datasets(run_id TEXT NOT NULL, dataset_id TEXT NOT NULL, PRIMARY KEY(run_id, dataset_id));
CREATE TABLE IF NOT EXISTS impacts(impact_id INTEGER PRIMARY KEY AUTOINCREMENT, tenant_id TEXT NOT NULL,
  subject_digest TEXT NOT NULL, model_id TEXT NOT NULL, version TEXT NOT NULL, run_id TEXT NOT NULL,
  status TEXT NOT NULL, resolution TEXT, evidence_ref TEXT, reviewed_by TEXT, opened_at REAL NOT NULL, closed_at REAL,
  UNIQUE(subject_digest, model_id, version, run_id));
"""


class LearningGovernanceError(Exception):
    """A learning-governance rule was violated."""


class LearningRegistry:
    def __init__(self, path, *, digest_salt: bytes, clock: Callable[[], float] = time.time) -> None:
        if not digest_salt or len(digest_salt) < 16:
            raise LearningGovernanceError("digest_salt (>=16 bytes) required")
        self._db = open_db(path)
        self._db.executescript(_SCHEMA)
        self._salt = digest_salt
        self._clock = clock
        self._lock = RLock()

    def _digest(self, tenant_id: str, subject_id: str) -> str:
        return hmac.new(self._salt, f"{tenant_id}\x1f{subject_id}".encode(), hashlib.sha256).hexdigest()

    @staticmethod
    def _need(**v: Any) -> None:
        for k, x in v.items():
            if x is None or (isinstance(x, str) and not x.strip()):
                raise LearningGovernanceError(f"{k} is required")

    # -------------------------------------------------------------- datasets and lineage
    def register_dataset(self, dataset_id: str, tenant_id: str, *, purpose: str, activity_id: str,
                         decision: Any, created_by: str, subject_ids: Sequence[str] = ()) -> None:
        self._need(dataset_id=dataset_id, tenant_id=tenant_id, activity_id=activity_id, created_by=created_by)
        self._check_decision(decision, tenant_id, purpose)
        with self._lock:
            self._db.execute("INSERT INTO datasets VALUES(?,?,?,?,?,?,?)",
                             (dataset_id, tenant_id, purpose, activity_id, decision.request_id, created_by, self._clock()))
            for sid in subject_ids:
                self._db.execute("INSERT OR IGNORE INTO dataset_members VALUES(?,?)", (dataset_id, self._digest(tenant_id, sid)))

    def dataset_registered(self, dataset_id: str, tenant_id: str) -> bool:
        row = self._db.execute("SELECT tenant_id FROM datasets WHERE dataset_id=?", (dataset_id,)).fetchone()
        return bool(row and row[0] == tenant_id)

    def add_subject(self, dataset_id: str, tenant_id: str, subject_id: str) -> None:
        row = self._db.execute("SELECT tenant_id FROM datasets WHERE dataset_id=?", (dataset_id,)).fetchone()
        if not row or row[0] != tenant_id:
            raise LearningGovernanceError("unknown dataset for tenant")
        self._db.execute("INSERT OR IGNORE INTO dataset_members VALUES(?,?)", (dataset_id, self._digest(tenant_id, subject_id)))

    @staticmethod
    def _check_decision(decision: Any, tenant_id: str, purpose: str) -> None:
        """A learning step needs an ALLOWED decision for the learning purpose; serving approvals never count."""
        if purpose not in LEARNING_PURPOSES:
            raise LearningGovernanceError(f"'{purpose}' is not a learning purpose {sorted(LEARNING_PURPOSES)}")
        if decision is None or getattr(decision, "decision", None) not in (Decision.ALLOW, Decision.ALLOW_WITH_CONTROLS):
            raise LearningGovernanceError("a governance decision allowing this learning processing is required")
        fp = getattr(decision, "context_fingerprint", "") or ""
        if not fp:
            raise LearningGovernanceError("decision carries no context fingerprint")

    # -------------------------------------------------------------- models and runs
    def register_model(self, model_id: str, version: str, *, system_id: str, tenant_id: Optional[str] = None,
                       gpai: bool = False, training_flops: Optional[float] = None, docs: Optional[Mapping[str, str]] = None) -> None:
        self._need(model_id=model_id, version=version, system_id=system_id)
        self._db.execute("INSERT INTO models VALUES(?,?,?,?,?,?,?,?,?,?)", (model_id, version, tenant_id, system_id, "candidate",
                         1 if gpai else 0, training_flops, json.dumps(dict(docs or {}), sort_keys=True), None, self._clock()))

    def register_training_run(self, run_id: str, model_id: str, version: str, tenant_id: str, *, dataset_ids: Sequence[str],
                              purpose: str, decision: Any) -> None:
        self._need(run_id=run_id, model_id=model_id, tenant_id=tenant_id)
        self._check_decision(decision, tenant_id, purpose)
        if not dataset_ids:
            raise LearningGovernanceError("a training run needs at least one registered dataset")
        if not self._db.execute("SELECT 1 FROM models WHERE model_id=? AND version=?", (model_id, version)).fetchone():
            raise LearningGovernanceError("unknown model version")
        for ds in dataset_ids:
            row = self._db.execute("SELECT tenant_id,purpose FROM datasets WHERE dataset_id=?", (ds,)).fetchone()
            if not row:
                raise LearningGovernanceError(f"unregistered dataset {ds!r}: lineage must be complete")
            if row[0] != tenant_id:
                raise LearningGovernanceError(f"dataset {ds!r} belongs to another tenant")
        with self._lock:
            self._db.execute("INSERT INTO training_runs VALUES(?,?,?,?,?,?,?)", (run_id, model_id, version, tenant_id, purpose, decision.request_id, self._clock()))
            for ds in dataset_ids:
                self._db.execute("INSERT INTO run_datasets VALUES(?,?)", (run_id, ds))

    def models_trained_on(self, tenant_id: str, subject_id: str) -> List[Dict[str, str]]:
        d = self._digest(tenant_id, subject_id)
        rows = self._db.execute(
            "SELECT DISTINCT r.model_id, r.version, r.run_id FROM dataset_members m JOIN run_datasets rd ON rd.dataset_id=m.dataset_id "
            "JOIN training_runs r ON r.run_id=rd.run_id WHERE m.subject_digest=?", (d,)).fetchall()
        return [{"model_id": a, "version": b, "run_id": c} for a, b, c in rows]

    # -------------------------------------------------------------- erasure impact (never "unlearning by deletion")
    def open_erasure_impacts(self, tenant_id: str, subject_id: str) -> int:
        d = self._digest(tenant_id, subject_id)
        opened = 0
        with self._lock:
            for m in self.models_trained_on(tenant_id, subject_id):
                cur = self._db.execute("INSERT OR IGNORE INTO impacts(tenant_id,subject_digest,model_id,version,run_id,status,opened_at) VALUES(?,?,?,?,?,?,?)",
                                       (tenant_id, d, m["model_id"], m["version"], m["run_id"], "open", self._clock()))
                opened += cur.rowcount
            self._db.execute("DELETE FROM dataset_members WHERE subject_digest=? AND dataset_id IN (SELECT dataset_id FROM datasets WHERE tenant_id=?)", (d, tenant_id))
        return opened

    def resolve_impact(self, impact_id: int, *, resolution: str, evidence_ref: str, reviewed_by: str) -> None:
        if resolution not in RESOLUTIONS:
            raise LearningGovernanceError(f"resolution must be one of {RESOLUTIONS}")
        self._need(evidence_ref=evidence_ref, reviewed_by=reviewed_by)
        cur = self._db.execute("UPDATE impacts SET status='closed', resolution=?, evidence_ref=?, reviewed_by=?, closed_at=? WHERE impact_id=? AND status='open'",
                               (resolution, evidence_ref, reviewed_by, self._clock(), impact_id))
        if cur.rowcount != 1:
            raise LearningGovernanceError("unknown or already closed impact")

    def open_impacts(self, *, model_id: Optional[str] = None, version: Optional[str] = None) -> List[Dict[str, Any]]:
        q, args = "SELECT impact_id,tenant_id,model_id,version,run_id,opened_at FROM impacts WHERE status='open'", []
        if model_id:
            q += " AND model_id=?"; args.append(model_id)
        if version:
            q += " AND version=?"; args.append(version)
        return [dict(zip(("impact_id", "tenant_id", "model_id", "version", "run_id", "opened_at"), r)) for r in self._db.execute(q, args)]

    def subject_in_learning_data(self, tenant_id: str, subject_id: str) -> bool:
        return self._db.execute("SELECT 1 FROM dataset_members WHERE subject_digest=? LIMIT 1", (self._digest(tenant_id, subject_id),)).fetchone() is not None

    def open_impacts_for_subject(self, tenant_id: str, subject_id: str) -> int:
        return self._db.execute("SELECT COUNT(*) FROM impacts WHERE subject_digest=? AND tenant_id=? AND status='open'",
                                (self._digest(tenant_id, subject_id), tenant_id)).fetchone()[0]

    # -------------------------------------------------------------- GPAI assessment (Art. 53/55)
    GPAI_DOCS = ("technical_documentation_ref", "downstream_provider_information_ref", "copyright_policy_ref", "training_content_summary_ref")
    SYSTEMIC_DOCS = ("model_evaluation_ref", "systemic_risk_mitigation_ref", "serious_incident_process_ref", "cybersecurity_protection_ref")

    def gpai_assessment(self, model_id: str, version: str) -> Dict[str, Any]:
        row = self._db.execute("SELECT gpai,training_flops,docs FROM models WHERE model_id=? AND version=?", (model_id, version)).fetchone()
        if not row:
            raise LearningGovernanceError("unknown model version")
        gpai, flops, docs = bool(row[0]), row[1], json.loads(row[2])
        if not gpai:
            return {"gpai": False, "missing": [], "systemic_risk": False, "note": "not declared as a general-purpose AI model"}
        systemic = flops is None or flops >= SYSTEMIC_RISK_FLOPS  # unknown compute => assume systemic until shown otherwise
        needed = list(self.GPAI_DOCS) + (list(self.SYSTEMIC_DOCS) if systemic else [])
        missing = [d for d in needed if not str(docs.get(d, "")).strip()]
        return {"gpai": True, "systemic_risk": systemic, "systemic_assumed_because_compute_unknown": flops is None,
                "missing": missing, "complete": not missing}

    # -------------------------------------------------------------- release
    def approve_model(self, model_id: str, version: str, *, classification: RiskClassification, approved_by: str) -> None:
        self._need(approved_by=approved_by)
        blockers: List[str] = []
        if classification.tier is RiskTier.PROHIBITED or classification.prohibited_tags:
            blockers.append("prohibited_classification")
        if classification.open_questions:
            blockers.append("classification_open_questions:" + ",".join(classification.open_questions))
        if self.open_impacts(model_id=model_id, version=version):
            blockers.append("open_erasure_impacts")
        gp = self.gpai_assessment(model_id, version)
        if gp["gpai"] and gp["missing"]:
            blockers.append("gpai_documentation_missing:" + ",".join(gp["missing"]))
        if not self._db.execute("SELECT 1 FROM training_runs WHERE model_id=? AND version=?", (model_id, version)).fetchone():
            blockers.append("no_registered_training_lineage")
        if blockers:
            raise LearningGovernanceError("model cannot be approved: " + "; ".join(blockers))
        self._db.execute("UPDATE models SET status='approved', approved_by=?, updated_at=? WHERE model_id=? AND version=?",
                         (approved_by, self._clock(), model_id, version))

    def model_status(self, model_id: str, version: str) -> str:
        """'approved' degrades to 'remediation_required' while erasure impacts are open."""
        row = self._db.execute("SELECT status FROM models WHERE model_id=? AND version=?", (model_id, version)).fetchone()
        if not row:
            return "unknown"
        if row[0] == "approved" and self.open_impacts(model_id=model_id, version=version):
            return "remediation_required"
        return row[0]

    def close(self) -> None:
        self._db.close()


class LearningDeletionAdapter:
    """PrivacyLedger adapter: removes dataset membership and opens model impacts; verifies only when impacts are closed."""

    name = "learning_pipeline"

    def __init__(self, registry: LearningRegistry) -> None:
        self._r = registry

    def delete(self, tenant_id: str, subject_id: str) -> Mapping[str, Any]:
        return {"impacts_opened": self._r.open_erasure_impacts(tenant_id, subject_id)}

    def verify_absent(self, tenant_id: str, subject_id: str) -> bool:
        return not self._r.subject_in_learning_data(tenant_id, subject_id) and self._r.open_impacts_for_subject(tenant_id, subject_id) == 0
