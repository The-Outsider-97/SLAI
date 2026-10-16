"""SLAI v2.3: learned delegation scoring, privacy-minimal event capture.

No SLAI imports are needed here; the router retains ownership of eligibility,
execution, retry and policy enforcement. No model is fabricated on load failure.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import threading
import time
import uuid

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping

from logs.logger import PrettyPrinter, get_logger  # pyright: ignore[reportMissingImports]

LOG = get_logger("Delegation Ranker")
printer = PrettyPrinter()

SCHEMA = 1
BUCKETS = 16
FEATURE_NAMES = (
    "bias", "capability_coverage", "capability_jaccard", "required_count_log",
    "candidate_count_log", "active_tasks_log", "history_success_rate",
    "history_attempts_log", "historical_latency_log", "priority_normalized",
    "task_size_log", "complexity", "load_fraction", "preferred", "healthy",
    "deadline_log", "task_type_hash", "capability_interactions_hash",
)  # last two expand into BUCKETS each
INPUT_DIM = 16 + 2 * BUCKETS


def _number(value: Any, default: float = 0.0, lo: float = 0.0, hi: float = 1e9) -> float:
    try:
        z = float(value)
        return max(lo, min(hi, z)) if math.isfinite(z) else default
    except (TypeError, ValueError, OverflowError):
        return default


def _tokens(value: Any) -> set[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple, set, frozenset)):
        return set()
    return {str(v).strip().lower()[:96] for v in value if isinstance(v, str) and v.strip()}


def _bucket(name: str) -> int:
    return int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:4], "big") % BUCKETS


def feature_vector(task: Mapping[str, Any], candidate: Mapping[str, Any], stats: Mapping[str, Any]) -> list[float]:
    """Only uses pre-decision state. Avoids raw payloads, user text and agent IDs."""
    req = _tokens(task.get("required_capabilities", ()))
    have = _tokens(candidate.get("capabilities", ()))
    tt = str(task.get("task_type", task.get("type", "unknown")))[:96].lower()
    if not req and tt != "unknown":
        req = {tt}
    common = req & have
    active = _number(stats.get("active_tasks", candidate.get("active_tasks", 0)))
    attempts = _number(stats.get("successes", 0)) + _number(stats.get("failures", 0))
    successes = _number(stats.get("successes", 0))
    rate = (1.0 + successes) / (2.0 + attempts)  # smoothed prior, not an outcome label
    limit = _number(candidate.get("max_tasks", candidate.get("max_concurrent_tasks", 0)))
    preferred = _tokens(task.get("preferred_agents", []))
    preferred |= _tokens(task.get("preferred_agent", []))
    cand_name = str(candidate.get("name", candidate.get("agent_name", ""))).lower()
    v = [
        1.0,
        len(common) / max(1, len(req)),
        len(common) / max(1, len(req | have)),
        math.log1p(len(req)),
        math.log1p(len(have)),
        math.log1p(active),
        rate,
        math.log1p(attempts),
        math.log1p(_number(stats.get("avg_latency_ms", stats.get("latency_ms", 0)))) / 10.0,
        _number(task.get("priority", 0), hi=100) / 100.0,
        math.log1p(_number(task.get("estimated_size", task.get("task_size", 0)))) / 15.0,
        _number(task.get("complexity", 0), hi=10) / 10.0,
        active / max(1.0, limit) if limit else 0.0,
        float(cand_name in preferred),
        float(str(stats.get("circuit_state", "closed")).lower() != "open" and stats.get("available", True) is not False),
        math.log1p(_number(task.get("timeout_seconds", 0))) / 10.0,
    ]
    type_hash = [0.0] * BUCKETS
    type_hash[_bucket("type:" + tt)] = 1.0
    cap_hash = [0.0] * BUCKETS
    for cap in sorted(req | have):
        cap_hash[_bucket("required:" + cap if cap in req else "offered:" + cap)] += 1.0
    cap_hash = [x / max(1, len(req | have)) for x in cap_hash]
    result = v + type_hash + cap_hash
    assert len(result) == INPUT_DIM
    return result


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _sqlite_connect(path: str | Path):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(target), timeout=20)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA busy_timeout=20000")
    db.execute("""CREATE TABLE IF NOT EXISTS decisions(
      decision_id TEXT PRIMARY KEY, task_id TEXT NOT NULL, created_at REAL NOT NULL,
      task_type TEXT NOT NULL, mode TEXT NOT NULL, policy TEXT NOT NULL,
      chosen TEXT NOT NULL, candidate_features_json TEXT NOT NULL,
      model_version TEXT, schema_version INTEGER NOT NULL)""")
    db.execute("""CREATE TABLE IF NOT EXISTS outcomes(
      decision_id TEXT NOT NULL, attempt_no INTEGER NOT NULL, agent_id TEXT NOT NULL,
      status TEXT NOT NULL, duration_ms REAL, observed_at REAL NOT NULL,
      source TEXT NOT NULL, PRIMARY KEY(decision_id, attempt_no),
      FOREIGN KEY(decision_id) REFERENCES decisions(decision_id))""")
    db.execute("CREATE INDEX IF NOT EXISTS idx_outcomes_decision ON outcomes(decision_id)")
    try:
        with db:
            yield db
    finally:
        db.close()


class DelegationEventStore:
    """SQLite event store. Stores safe numeric candidate features, not task bodies."""
    def __init__(self, path: str | Path):
        self.path = str(path)
        with _sqlite_connect(self.path) as con:
            con.execute("PRAGMA user_version=1")

    def decision(self, *, task_id: str, task_type: str, mode: str,
                 policy: str, chosen: str, candidates: Mapping[str, list[float]],
                 model_version: str | None) -> str:
        decision_id = uuid.uuid4().hex
        clean = {str(k)[:128]: [float(x) for x in v] for k, v in candidates.items()}
        with _sqlite_connect(self.path) as con:
            con.execute("INSERT INTO decisions VALUES(?,?,?,?,?,?,?,?,?,?)", (
                decision_id, str(task_id)[:128], time.time(), str(task_type)[:96],
                mode, policy, chosen, json.dumps(clean), model_version, SCHEMA))
        return decision_id

    def outcomes(self, decision_id: str, route: Mapping[str, Any]) -> None:
        # Skips, rejections, missing outcomes are deliberately NOT failure labels.
        with _sqlite_connect(self.path) as con:
            for i, attempt in enumerate(route.get("attempts", ())):
                if not isinstance(attempt, Mapping):
                    continue
                status = str(attempt.get("status", "unknown"))
                if status not in {"success", "failed", "timeout", "cancelled", "skipped"}:
                    status = "unknown"
                con.execute("INSERT OR IGNORE INTO outcomes VALUES(?,?,?,?,?,?,?)", (
                    decision_id, i, str(attempt.get("agent_name", ""))[:128], status,
                    _number(attempt.get("duration_ms"), lo=0) if status in {"success", "failed"} else None,
                    time.time(), "slai_task_router_attempt"))

    def samples(self) -> tuple[list[dict[str, Any]], dict[str, int]]:
        rows: list[dict[str, Any]] = []
        counts: dict[str, int] = {}
        with _sqlite_connect(self.path) as con:
            result = con.execute("""SELECT d.decision_id,d.task_id,d.created_at,d.candidate_features_json,
              o.agent_id,o.status,o.duration_ms,o.source
              FROM decisions d JOIN outcomes o ON d.decision_id=o.decision_id
              WHERE d.schema_version=? ORDER BY d.created_at,d.decision_id,o.attempt_no""", (SCHEMA,))
            for decision_id, task_id, ts, cand_json, name, status, latency, source in result:
                counts[status] = counts.get(status, 0) + 1
                if status not in {"success", "failed"}:
                    continue
                features = json.loads(cand_json).get(name)
                if not isinstance(features, list) or len(features) != INPUT_DIM:
                    continue
                rows.append({"decision_id": decision_id, "task_id": task_id,
                             "timestamp": ts, "agent_id": name, "x": features,
                             "success": int(status == "success"), "latency_ms": latency,
                             "provenance": source})
        return rows, counts

    def audit(self) -> dict[str, Any]:
        with _sqlite_connect(self.path) as con:
            d = con.execute("SELECT count(*) FROM decisions").fetchone()[0]
            o = con.execute("SELECT status,count(*) FROM outcomes GROUP BY status").fetchall()
        rows, _ = self.samples()
        return {"decisions": d, "outcomes": dict(o), "trusted_labeled_attempts": len(rows),
                "positive": sum(r["success"] for r in rows), "negative": sum(not r["success"] for r in rows),
                "note": "Only observed attempts are labeled; no counterfactual outcomes."}


class DelegationRanker:
    """Safe inference-only shared-weight MLP. The trainer owns all optimization."""
    def __init__(self, directory: str | Path, device: str = "cpu"):
        import torch # type: ignore
        from torch import nn # type: ignore
        self.path = Path(directory)
        meta = json.loads((self.path / "manifest.json").read_text(encoding="utf-8"))
        checkpoint = self.path / "model.pt"
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if digest != meta.get("sha256") or meta.get("schema") != SCHEMA or meta.get("input_dim") != INPUT_DIM:
            raise ValueError("ADR checkpoint hash or feature schema mismatch")
        self.torch = torch
        self.device = device
        self.net = nn.Sequential(nn.Linear(INPUT_DIM, meta["width"]), nn.ReLU(), nn.Linear(meta["width"], 2))
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.net.load_state_dict(state, strict=True)
        self.net.to(device).eval()
        features = json.loads((self.path / "features.json").read_text(encoding="utf-8"))
        self.mean, self.scale = features["mean"], features["scale"]
        if len(self.mean) != INPUT_DIM or len(self.scale) != INPUT_DIM:
            raise ValueError("ADR preprocessing schema mismatch")
        self.version = meta["version"]
        self.has_latency = bool(meta.get("has_latency", False))

    def predict(self, rows: list[list[float]]) -> list[dict[str, float | None]]:
        if not rows:
            return []
        torch = self.torch
        matrix = [[(float(v) - m) / max(1e-6, s) for v, m, s in zip(row, self.mean, self.scale)] for row in rows]
        if any(len(row) != INPUT_DIM for row in rows):
            raise ValueError("ADR feature dimensionality mismatch")
        with torch.inference_mode():
            out = self.net(torch.tensor(matrix, dtype=torch.float32, device=self.device)).cpu()
            probs = torch.sigmoid(out[:, 0]).tolist()
            latencies = torch.expm1(out[:, 1].clamp(min=0, max=18)).tolist()
        return [{"success_probability": float(p),
                 "latency_ms": float(t) if self.has_latency else None,
                 "uncertainty_proxy": float(4 * p * (1 - p))} for p, t in zip(probs, latencies)]


class DelegationRankerStrategy:
    """Composable strategy shim. Existing routing and agent-instance metadata unchanged."""
    name = "delegation_ranker"

    def __init__(self, baseline: Any, config: Mapping[str, Any]):
        self.baseline = baseline
        self.mode = str(config.get("mode", "disabled")).lower()
        if self.mode not in {"disabled", "shadow", "advisory", "active"}:
            raise ValueError("Invalid ADR rollout mode")
        self.model = None
        self.min_confidence = _number(config.get("min_confidence", 0.0), hi=1.0)
        self.success_value = float(config.get("success_value", 1.0))
        self.latency_weight = _number(config.get("latency_penalty_per_ms", 0.0))
        self.advisory_mix = _number(config.get("advisory_mix", 0.25), hi=1.0)
        self.policy = getattr(baseline, "name", "weighted")
        self._pending = threading.local()  # route and hook execute in same thread
        self.store = DelegationEventStore(config.get("event_db", "data/training/collaborative/delegation_ranker/raw/events.sqlite3"))
        if self.mode != "disabled":
            try:
                self.model = DelegationRanker(config["model_dir"])
            except (FileNotFoundError, ValueError, KeyError, RuntimeError, ImportError) as exc:
                LOG.warning("ADR unavailable, deterministic routing retained: %s", exc)
        self.last_report = None

    def rank_agents(self, agents: Mapping[str, Any], stats: Mapping[str, Any], task_data: Mapping[str, Any]):
        baseline = list(self.baseline.rank_agents(agents, stats, task_data))
        if not baseline:
            return baseline
        rows = {}
        for name, meta, _ in baseline:
            safe_meta = {k: meta.get(k) for k in ("capabilities", "max_tasks", "max_concurrent_tasks", "active_tasks")}
            safe_meta["name"] = name
            rows[name] = feature_vector(task_data, safe_meta, stats.get(name, {}))
        ranked = baseline
        predictions = None
        if self.model is not None:
            try:
                predictions = self.model.predict([rows[n] for n, _, _ in baseline])
                utilities = {}
                for (name, _, _), pred in zip(baseline, predictions):
                    success_probability = pred["success_probability"]
                    p = float(success_probability) if success_probability is not None else 0.0
                    lat = pred["latency_ms"]
                    utility = self.success_value * p - (self.latency_weight * lat if lat is not None else 0)
                    utilities[name] = utility
                confident = all(
                    p["uncertainty_proxy"] is not None
                    and float(p["uncertainty_proxy"]) <= 1 - self.min_confidence
                    for p in predictions
                )
                if self.mode == "active" and confident:
                    ranked = sorted(((n,m,utilities[n]) for n,m,_ in baseline), key=lambda t: (-t[2], t[0]))
                elif self.mode == "advisory" and confident:
                    scores = [s for _, _, s in baseline]
                    low, high = min(scores), max(scores)
                    combined = [(n,m,(1-self.advisory_mix)*((score-low)/max(high-low,1e-9)) + self.advisory_mix*utilities[n]) for n,m,score in baseline]
                    ranked = sorted(combined, key=lambda t: (-t[2], t[0]))
            except (RuntimeError, ValueError, OverflowError) as exc:
                LOG.warning("ADR inference failure; baseline retained: %s", exc)
                ranked = baseline
        # Ranking tuple contract preserved; scores report learned utility in active mode.
        task_id = str(task_data.get("task_id", uuid.uuid4().hex))
        if self.mode != "disabled":
            try:
                did = self.store.decision(task_id=task_id,
                    task_type=str(task_data.get("task_type", "unknown")), mode=self.mode,
                    policy=self.policy, chosen=ranked[0][0], candidates=rows,
                    model_version=self.model.version if self.model else None)
                self._pending.id = (task_id, did)
            except (sqlite3.Error, OSError, ValueError) as exc:
                LOG.warning("ADR event collection failed (no raw task data logged): %s", exc)
        return ranked

    def on_route(self, route: Mapping[str, Any]) -> None:
        pending = getattr(self._pending, "id", None)
        self._pending.id = None
        if not pending or str(route.get("task_id")) != pending[0]:
            return
        try:
            self.store.outcomes(pending[1], route)
        except (OSError, sqlite3.Error, ValueError) as exc:
            LOG.warning("ADR outcome collection failed: %s", exc)
