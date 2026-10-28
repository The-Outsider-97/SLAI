"""JSON-level front door for :class:`ExternalLearningService` (what ``LearningAgent.perform_task(mode="external")`` calls).

An environment adapter - in-process or behind an HTTP service - sends plain dictionaries::

    {"mode": "external", "op": "configure", "name": "squad", "feature_dim": 26, "action_dim": 8, "checkpoint_dir": "...", "hyper": {...}}
    {"mode": "external", "op": "start_episode", "name": "squad", "episode_id": "g-123", "learner_mode": "train" | "evaluate"}
    {"op": "decide", "name": "squad", "episode_id": "g-123", "decision_id": "g-123/40/0", "features": [...],
     "action_mask": [...], "prior_logits": [...]}
    {"op": "feedback" | "executed" | "end_episode" | "abort_episode" | "set_mode" | "flush" | "checkpoint" | "restore" | "status" ...}

Every call returns a dictionary with ``status``: ``"ok"``, ``"rejected"`` (the request violated the protocol or a contract; ``code`` is
stable and countable) or ``"error"`` (SLAI could not do what was asked, e.g. a corrupt checkpoint).  Nothing is silently defaulted:
a malformed observation is rejected, never turned into a zero vector.

This module is environment-independent; it contains no knowledge of any particular environment.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Mapping, Optional

from .contracts import CheckpointIntegrityError, ContractError, Observation, TransitionRejected
from .external_learning import ExternalLearningConfig, ExternalLearningService
from .softmax_policy import PolicyHyperparameters

OPS = ("configure", "start_episode", "decide", "feedback", "executed", "end_episode", "abort_episode", "set_mode", "flush",
       "checkpoint", "restore", "status", "reward_report", "metadata")

_CONFIG_KEYS = ("seed", "mode", "learning_enabled", "min_episodes_per_update", "max_episodes_per_update", "evaluate_greedy",
                "checkpoint_dir", "autosave_every_updates", "restore_on_start", "max_live_episodes", "max_decisions_per_episode",
                "max_pending_episodes")


def _config_digest(payload: Mapping[str, Any]) -> str:
    keep = {k: payload.get(k) for k in ("name", "feature_dim", "action_dim", "hyper", "checkpoint_dir", *_CONFIG_KEYS) if k in payload}
    return hashlib.sha256(json.dumps(keep, sort_keys=True, default=str).encode()).hexdigest()


def _service(services: Dict[str, ExternalLearningService], name: Any) -> ExternalLearningService:
    if not isinstance(name, str) or not name:
        raise ContractError("invalid_name", "'name' (the learner/head name) is required")
    svc = services.get(name)
    if svc is None:
        raise TransitionRejected("unknown_learner", f"learner {name!r} has not been configured")
    return svc


def dispatch(services: Dict[str, ExternalLearningService], payload: Mapping[str, Any], *,
             defaults: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Execute one external-learning operation.  ``services`` is the caller-owned registry (name -> service)."""
    op = payload.get("op")
    if op not in OPS:
        return {"status": "rejected", "code": "unknown_op", "message": f"op must be one of {list(OPS)}, got {op!r}"}
    try:
        return _run(services, op, payload, defaults or {})
    except TransitionRejected as exc:
        return {"status": "rejected", "code": exc.code, "message": exc.message, "op": op}
    except ContractError as exc:
        return {"status": "rejected", "code": exc.code, "message": exc.message, "op": op}
    except CheckpointIntegrityError as exc:
        return {"status": "error", "code": "checkpoint_integrity", "message": str(exc), "op": op}
    except (ValueError, TypeError, KeyError) as exc:
        return {"status": "rejected", "code": "invalid_request", "message": f"{type(exc).__name__}: {exc}", "op": op}
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed: the caller sees status=error and the type
        return {"status": "error", "code": type(exc).__name__, "message": str(exc), "op": op}


def _run(services: Dict[str, ExternalLearningService], op: str, p: Mapping[str, Any], defaults: Mapping[str, Any]) -> Dict[str, Any]:
    if op == "configure":
        name = p.get("name")
        if not isinstance(name, str) or not name:
            raise ContractError("invalid_name", "'name' is required")
        merged: Dict[str, Any] = {**defaults, **{k: v for k, v in p.items() if k not in ("op", "mode")}}   # "mode" is perform_task's own key
        if "learner_mode" in merged:
            merged["mode"] = merged.pop("learner_mode")
        digest = _config_digest(merged)
        existing = services.get(name)
        if existing is not None:
            if getattr(existing, "_config_digest", None) == digest:
                return {"status": "ok", "created": False, "policy": existing.metadata().to_dict()}
            if not p.get("replace"):
                raise ContractError("learner_exists", f"learner {name!r} exists with a different configuration (pass replace=true)")
        hyper = PolicyHyperparameters(**dict(merged.get("hyper") or {}))
        cfg = ExternalLearningConfig(name=name, feature_dim=int(merged["feature_dim"]), action_dim=int(merged["action_dim"]), hyper=hyper,
                                     **{k: merged[k] for k in _CONFIG_KEYS if k in merged and merged[k] is not None})
        svc = ExternalLearningService(cfg)
        svc._config_digest = digest  # type: ignore[attr-defined]
        services[name] = svc
        return {"status": "ok", "created": True, "policy": svc.metadata().to_dict(), "restore": svc.status()["restore"],
                "persistent": bool(cfg.checkpoint_dir)}

    if op == "status":
        if p.get("name"):
            return {"status": "ok", "learners": {p["name"]: _service(services, p["name"]).status()}}
        return {"status": "ok", "learners": {n: s.status() for n, s in services.items()}}

    svc = _service(services, p.get("name"))
    if op == "metadata":
        return {"status": "ok", "policy": svc.metadata().to_dict()}
    if op == "reward_report":
        return {"status": "ok", "reward": svc.reward_report()}
    if op == "start_episode":
        return {"status": "ok", **svc.start_episode(p["episode_id"], mode=p.get("learner_mode"), metadata=p.get("metadata"))}
    if op == "decide":
        obs = Observation.make(p.get("features"), p.get("action_mask"), p.get("prior_logits"),
                               feature_dim=svc.policy.feature_dim, action_dim=svc.policy.action_dim)
        res = svc.decide(p["episode_id"], p["decision_id"], obs, sequence=p.get("sequence"))
        a = res.action
        return {"status": "ok", "action": a.index, "probability": a.probability, "probabilities": list(a.probabilities),
                "greedy": a.greedy, "source": a.source, "policy": res.policy, "episode_mode": res.episode_mode}
    if op == "feedback":
        svc.feedback(p["episode_id"], p["decision_id"], p.get("reward"))
        return {"status": "ok"}
    if op == "executed":
        svc.executed(p["episode_id"], p["decision_id"], int(p["action"]))
        return {"status": "ok"}
    if op == "end_episode":
        res = svc.end_episode(p["episode_id"], terminal_reward=p.get("terminal_reward", 0.0), terminated=bool(p.get("terminated", True)),
                              truncated=bool(p.get("truncated", False)))
        return {"status": "ok", "update": res.to_dict(), "policy": svc.metadata().brief()}
    if op == "abort_episode":
        return {"status": "ok", "aborted": svc.abort_episode(p["episode_id"], str(p.get("reason", "aborted")))}
    if op == "set_mode":
        if "learner_mode" in p:
            svc.set_mode(p["learner_mode"])
        if "learning_enabled" in p:
            svc.set_learning_enabled(bool(p["learning_enabled"]))
        return {"status": "ok", "policy": svc.metadata().to_dict()}
    if op == "flush":
        res = svc.flush()
        return {"status": "ok", "update": None if res is None else res.to_dict()}
    if op == "checkpoint":
        return {"status": "ok", "checkpoint": svc.save_checkpoint(str(p.get("reason", "manual")), force=bool(p.get("force", False))).to_dict()}
    if op == "restore":
        ck = svc.restore(p.get("version"))
        return {"status": "ok", "restored": ck is not None, "checkpoint": None if ck is None else ck.to_dict(), "restore": svc.status()["restore"]}
    raise ContractError("unknown_op", op)  # pragma: no cover
