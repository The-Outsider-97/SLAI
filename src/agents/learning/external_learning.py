"""Learning from an *external* environment: the generic entry point SLAI was missing.

An environment (through a thin adapter) drives this service with a small, domain-free protocol::

    configure -> start_episode -> [ decide -> (feedback) -> (executed) ]* -> end_episode | abort_episode
                                        \\__ save_checkpoint / restore at any time between episodes

Vocabulary used in code, telemetry and docs (they are different things and are reported separately):

* **inference**             ``decide`` answers from the current parameters; it never changes them.
* **within-episode adaptation**  *not performed here* - parameters are frozen while an episode runs.  Any adaptation inside
                            an episode (e.g. an opponent model) belongs to the environment adapter and must be reported by it.
* **persistent learning**   ``end_episode`` in *train* mode feeds the finished episode to ``ResidualSoftmaxPolicy.update``;
                            each applied update changes the parameter hash and increments ``training_step``.
* **fallback**              an answer produced by the *environment* when this service did not answer.  The adapter must label
                            it so; this service only ever reports ``source="learned"``.
* **evaluation**            ``mode="evaluate"`` (or ``learning_enabled=False``): greedy answers, episodes are recorded for
                            statistics only and are never queued for training.  It cannot silently train.

Checkpoints go through SLAI's ``CheckpointManager`` (atomic, sha256-verified, lineage-aware); a restore re-hashes the loaded
arrays and refuses a checkpoint whose contents do not match its recorded hash.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional

import numpy as np  # type: ignore

from .contracts import (
    SCHEMA_VERSION,
    Action,
    CheckpointIntegrityError,
    CheckpointMetadata,
    ContractError,
    EpisodeRecord,
    LearningUpdateResult,
    Observation,
    PolicyMetadata,
    TransitionRejected,
    describe_sequence,
    require_id,
)
from .softmax_policy import ALGORITHM, PolicyHyperparameters, ResidualSoftmaxPolicy
from .transition_store import TransitionStore

logger = logging.getLogger(__name__)

MODES = ("train", "evaluate")


@dataclass
class ExternalLearningConfig:
    name: str
    feature_dim: int
    action_dim: int
    hyper: PolicyHyperparameters = field(default_factory=PolicyHyperparameters)
    seed: int = 0
    mode: str = "train"
    learning_enabled: bool = True
    min_episodes_per_update: int = 1
    max_episodes_per_update: int = 16
    evaluate_greedy: bool = True
    checkpoint_dir: Optional[str] = None
    autosave_every_updates: int = 0          # 0 = only explicit save_checkpoint()
    restore_on_start: bool = True            # with a checkpoint_dir: load the newest valid checkpoint (fresh start if none)
    max_live_episodes: int = 64
    max_decisions_per_episode: int = 20_000
    max_pending_episodes: int = 256

    def validate(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        if self.min_episodes_per_update < 1 or self.max_episodes_per_update < self.min_episodes_per_update:
            raise ValueError("need 1 <= min_episodes_per_update <= max_episodes_per_update")
        if self.autosave_every_updates < 0:
            raise ValueError("autosave_every_updates must be >= 0")
        if self.autosave_every_updates and not self.checkpoint_dir:
            raise ValueError("autosave_every_updates requires checkpoint_dir (learning is configured to persist but cannot)")


@dataclass(frozen=True)
class DecisionResult:
    action: Action
    policy: Dict[str, Any]      # PolicyMetadata.brief() at decision time
    episode_mode: str


class _RewardStats:
    """Streaming distribution of per-decision rewards plus per-episode totals (bounded)."""

    def __init__(self) -> None:
        self.n = 0; self.s = 0.0; self.ss = 0.0; self.lo = math.inf; self.hi = -math.inf
        self.zero = 0; self.pos = 0; self.neg = 0
        self.episode_totals: Deque[float] = deque(maxlen=2000)
        self.episodes = 0; self.episodes_with_terminal_reward = 0; self.episodes_truncated = 0

    def add_episode(self, rec: EpisodeRecord, terminal_reward: float) -> None:
        for t in rec.transitions:
            r = t.reward
            self.n += 1; self.s += r; self.ss += r * r; self.lo = min(self.lo, r); self.hi = max(self.hi, r)
            if r == 0.0: self.zero += 1
            elif r > 0: self.pos += 1
            else: self.neg += 1
        self.episodes += 1
        self.episodes_with_terminal_reward += 1 if terminal_reward != 0.0 else 0
        self.episodes_truncated += 1 if rec.truncated else 0
        self.episode_totals.append(rec.total_reward)

    def report(self) -> Dict[str, Any]:
        mean = self.s / self.n if self.n else None
        std = math.sqrt(max(self.ss / self.n - mean * mean, 0.0)) if self.n else None
        return {"decisions": self.n, "min": None if not self.n else self.lo, "max": None if not self.n else self.hi,
                "mean": mean, "std": std, "zero_fraction": None if not self.n else self.zero / self.n,
                "positive": self.pos, "negative": self.neg, "zero": self.zero,
                "positive_to_negative_ratio": None if not self.neg else self.pos / self.neg,
                "episodes": self.episodes, "terminal_reward_coverage": None if not self.episodes else self.episodes_with_terminal_reward / self.episodes,
                "episodes_truncated": self.episodes_truncated, "episode_totals": describe_sequence(self.episode_totals)}


class ExternalLearningService:
    def __init__(self, config: ExternalLearningConfig, *, checkpoint_manager: Any = None) -> None:
        config.validate()
        self.config = config
        self._lock = threading.RLock()
        self.policy = ResidualSoftmaxPolicy(config.feature_dim, config.action_dim, config.hyper, seed=config.seed, name=config.name)
        self.store = TransitionStore(max_live_episodes=config.max_live_episodes, max_decisions_per_episode=config.max_decisions_per_episode,
                                     max_pending_episodes=config.max_pending_episodes)
        self.mode = config.mode
        self.learning_enabled = config.learning_enabled
        self._episode_mode: Dict[str, str] = {}
        self._counters: Counter = Counter()
        self._skips: Counter = Counter()
        self._updates_applied = 0
        self._last_update: Optional[LearningUpdateResult] = None
        self._last_error: Optional[Dict[str, Any]] = None
        self._rewards = {"train": _RewardStats(), "evaluate": _RewardStats()}
        self._manager = checkpoint_manager
        self._checkpoint_id: Optional[str] = None
        self._parent_checkpoint_id: Optional[str] = None
        self._restored_from: Optional[str] = None
        self._last_saved_state_hash: Optional[str] = None
        self._last_checkpoint: Optional[CheckpointMetadata] = None
        self._restore_report: Dict[str, Any] = {"status": "not_attempted"}
        self._generation = 0
        if config.checkpoint_dir and config.restore_on_start:
            self.restore()          # raises CheckpointIntegrityError when checkpoints exist but none is valid

    # ------------------------------------------------------------------ identity
    def metadata(self) -> PolicyMetadata:
        p = self.policy
        return PolicyMetadata(
            policy_id=p.policy_id, algorithm=ALGORITHM, schema_version=SCHEMA_VERSION, parameter_hash=p.parameter_hash,
            state_hash=p.state_hash, hyperparameter_hash=p.hyper.digest(), training_step=p.training_step,
            consumed_transitions=p.consumed_transitions, consumed_episodes=p.consumed_episodes,
            learning_enabled=self.learning_enabled and self.mode == "train", mode=self.mode, checkpoint_id=self._checkpoint_id,
            parent_checkpoint_id=self._parent_checkpoint_id, restored_from=self._restored_from, action_dim=p.action_dim,
            feature_dim=p.feature_dim, name=self.config.name)

    # ------------------------------------------------------------------ mode control
    def set_mode(self, mode: str) -> None:
        if mode not in MODES:
            raise ContractError("invalid_mode", f"mode must be one of {MODES}, got {mode!r}")
        with self._lock:
            self.mode = mode

    def set_learning_enabled(self, enabled: bool) -> None:
        with self._lock:
            self.learning_enabled = bool(enabled)

    # ------------------------------------------------------------------ episode protocol
    def start_episode(self, episode_id: str, *, mode: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        with self._lock:
            m = mode or self.mode
            if m not in MODES:
                raise ContractError("invalid_mode", f"mode must be one of {MODES}, got {m!r}")
            effective = "train" if (m == "train" and self.learning_enabled) else "evaluate"
            self._guard(lambda: self.store.start_episode(episode_id, policy_version=self.policy.state_hash[:16], mode=effective))
            self._episode_mode[episode_id] = effective
            if len(self._episode_mode) > 4 * self.config.max_live_episodes:
                for k in [k for k in self._episode_mode if not self.store.is_live(k)]:
                    del self._episode_mode[k]
            self._counters["episodes_started"] += 1
            return {"episode_id": episode_id, "mode": effective, "policy": self.metadata().brief()}

    def decide(self, episode_id: str, decision_id: str, observation: Observation, *, sequence: Optional[int] = None) -> DecisionResult:
        with self._lock:
            if not self.store.is_live(episode_id):
                self._guard(lambda: self.store.add_decision(episode_id, decision_id, observation, 0))   # raises the precise reason
            mode = self._episode_mode.get(episode_id, "evaluate")
            greedy = mode == "evaluate" and self.config.evaluate_greedy
            try:
                action = self.policy.act(observation, greedy=greedy)
            except ContractError as exc:
                self._counters[f"decide_rejected:{exc.code}"] += 1
                raise
            self._guard(lambda: self.store.add_decision(
                episode_id, decision_id, observation, action.index, action_probability=action.probability,
                policy_version=self.policy.state_hash[:16], sequence=sequence))
            self._counters["decisions"] += 1
            return DecisionResult(action=action, policy=self.metadata().brief(), episode_mode=mode)

    def feedback(self, episode_id: str, decision_id: str, reward: Any) -> None:
        with self._lock:
            self._guard(lambda: self.store.add_reward(episode_id, decision_id, reward))
            self._counters["feedback_events"] += 1

    def executed(self, episode_id: str, decision_id: str, action_index: int) -> None:
        """Tell the service which action the environment actually executed (it may override the policy's choice)."""
        with self._lock:
            self._guard(lambda: self.store.mark_executed(episode_id, decision_id, action_index))

    def end_episode(self, episode_id: str, *, terminal_reward: Any = 0.0, terminated: bool = True, truncated: bool = False) -> LearningUpdateResult:
        with self._lock:
            mode = self._episode_mode.get(episode_id, "evaluate")
            retain = mode == "train" and self.learning_enabled
            rec = self._guard(lambda: self.store.end_episode(episode_id, terminal_reward=terminal_reward, terminated=terminated,
                                                              truncated=truncated, retain=retain))
            self._episode_mode.pop(episode_id, None)
            self._rewards[mode].add_episode(rec, float(terminal_reward))
            self._counters[f"episodes_ended:{mode}"] += 1
            if not retain:
                reason = "evaluation_mode" if mode == "evaluate" else "learning_disabled"
                self._skips[reason] += 1
                return self._remember_update(self._skipped(reason, rec))
            return self._maybe_update()

    def abort_episode(self, episode_id: str, reason: str = "aborted") -> bool:
        with self._lock:
            self._episode_mode.pop(episode_id, None)
            ok = self._guard(lambda: self.store.abort_episode(episode_id, reason))
            if ok:
                self._counters["episodes_aborted"] += 1
            return ok

    # ------------------------------------------------------------------ learning
    def _skipped(self, reason: str, rec: Optional[EpisodeRecord] = None) -> LearningUpdateResult:
        p = self.policy
        return LearningUpdateResult(
            updated=False, skip_reason=reason, step_before=p.training_step, step_after=p.training_step, episodes=1 if rec else 0,
            transitions=len(rec.transitions) if rec else 0, learning_rate=p.hyper.learning_rate, param_norm_before=p.param_norm(),
            param_norm_after=p.param_norm(), parameter_hash_before=p.parameter_hash, parameter_hash_after=p.parameter_hash,
            state_hash_before=p.state_hash, state_hash_after=p.state_hash, baseline=p.baseline_mean)

    def _remember_update(self, res: LearningUpdateResult) -> LearningUpdateResult:
        self._last_update = res
        return res

    def _maybe_update(self) -> LearningUpdateResult:
        if self.store.pending_count() < self.config.min_episodes_per_update:
            self._skips["insufficient_batch"] += 1
            return self._remember_update(self._skipped("insufficient_batch"))
        batch = self.store.take(self.config.max_episodes_per_update)
        try:
            res = self.policy.update(batch)
        except Exception as exc:  # noqa: BLE001 - surfaced in the result, status() and logs; the batch is returned to the queue
            self.store.give_back(batch)
            self._counters["update_errors"] += 1
            self._last_error = {"where": "update", "type": type(exc).__name__, "message": str(exc), "at": time.time()}
            logger.exception("external learning update failed (%s)", self.config.name)
            return self._remember_update(self._skipped(f"error:{type(exc).__name__}"))
        # inference reads the same parameter arrays this object just updated; make that explicit and checkable
        if res.updated and res.parameter_hash_after != self.policy.parameter_hash:
            raise AssertionError("update result does not describe the parameters used for inference")  # pragma: no cover
        if res.updated:
            self._updates_applied += 1
        else:
            self._skips[res.skip_reason or "unknown"] += 1
        self._counters["transitions_consumed"] += res.transitions
        if res.updated and self.config.autosave_every_updates and self._updates_applied % self.config.autosave_every_updates == 0:
            try:
                self.save_checkpoint(reason="autosave")
            except Exception as exc:  # noqa: BLE001
                self._counters["checkpoint_failures"] += 1
                self._last_error = {"where": "autosave", "type": type(exc).__name__, "message": str(exc), "at": time.time()}
                logger.exception("external learning autosave failed (%s)", self.config.name)
        return self._remember_update(res)

    def flush(self) -> Optional[LearningUpdateResult]:
        """Update on whatever is pending even if fewer than ``min_episodes_per_update`` episodes have arrived."""
        with self._lock:
            if self.mode != "train" or not self.learning_enabled or self.store.pending_count() == 0:
                return None
            saved = self.config.min_episodes_per_update
            self.config.min_episodes_per_update = 1
            try:
                return self._maybe_update()
            finally:
                self.config.min_episodes_per_update = saved

    def _guard(self, fn: Any) -> Any:
        try:
            return fn()
        except TransitionRejected as exc:
            self._counters[f"rejected:{exc.code}"] += 1
            raise

    # ------------------------------------------------------------------ persistence
    def _checkpoint_manager(self) -> Any:
        if self._manager is None:
            if not self.config.checkpoint_dir:
                raise CheckpointIntegrityError("no checkpoint_dir configured: this learner cannot persist")
            from checkpointing.checkpoint_manager import CheckpointManager  # type: ignore  # lazy: core learning needs no checkpoint stack

            self._manager = CheckpointManager(self.config.checkpoint_dir)
        return self._manager

    def save_checkpoint(self, reason: str = "manual", *, force: bool = False) -> CheckpointMetadata:
        """Persist the learned state.  Writes nothing (``saved=False``) when the state is unchanged since the last checkpoint."""
        with self._lock:
            p = self.policy
            if not force and self._last_checkpoint is not None and self._last_saved_state_hash == p.state_hash:
                return CheckpointMetadata(**{**self._last_checkpoint.to_dict(), "saved": False})
            from checkpointing.checkpoint_types import CheckpointLineage, CheckpointProvenance  # type: ignore

            cm = self._checkpoint_manager()
            version = f"s{p.training_step:08d}-{p.state_hash[:12]}"
            md = self.metadata()
            meta = {"schema_version": SCHEMA_VERSION, "algorithm": ALGORITHM, "name": self.config.name, "state_hash": p.state_hash,
                    "parameter_hash": p.parameter_hash, "hyperparameter_hash": md.hyperparameter_hash, "training_step": p.training_step,
                    "consumed_transitions": p.consumed_transitions, "consumed_episodes": p.consumed_episodes,
                    "action_dim": p.action_dim, "feature_dim": p.feature_dim, "parent_checkpoint_id": self._checkpoint_id or "",
                    "reason": reason}
            prov = CheckpointProvenance(
                agent_id=f"external-learning:{self.config.name}"[:120], checkpoint_reason=reason,
                lineage=CheckpointLineage(parent_checkpoint_id=self._checkpoint_id, generation=self._generation + 1))
            res = cm.save_components({"model": p.export_arrays()}, version=version, codec_ids={"model": "numpy"}, step=p.training_step,
                                     metrics={"baseline": float(p.baseline_mean), "updates_applied": float(self._updates_applied)},
                                     metadata=meta, provenance=prov, overwrite=False)
            man = res.record.manifest
            self._parent_checkpoint_id, self._checkpoint_id = self._checkpoint_id, man.checkpoint_id
            self._generation += 1
            self._last_saved_state_hash = p.state_hash
            self._last_checkpoint = CheckpointMetadata(
                checkpoint_id=man.checkpoint_id, version=man.version, path=str(res.path) if hasattr(res, "path") and res.path else None,
                state_hash=p.state_hash, parameter_hash=p.parameter_hash, hyperparameter_hash=md.hyperparameter_hash,
                parent_checkpoint_id=self._parent_checkpoint_id, training_step=p.training_step,
                consumed_transitions=p.consumed_transitions, consumed_episodes=p.consumed_episodes, schema_version=SCHEMA_VERSION,
                algorithm=ALGORITHM, created_at=man.created_at, saved=True)
            self._counters["checkpoints_saved"] += 1
            return self._last_checkpoint

    def restore(self, version: Optional[str] = None) -> Optional[CheckpointMetadata]:
        """Load the newest *valid* checkpoint (or the named ``version``, which must be valid).

        Returns ``None`` for a fresh start (no checkpoint exists).  Raises ``CheckpointIntegrityError`` when checkpoints exist
        but none is valid, when a named version is corrupt/incompatible, or when the loaded arrays do not hash to the value the
        checkpoint recorded.  Corrupt candidates that were skipped are listed in ``status()['restore']``.
        """
        with self._lock:
            cm = self._checkpoint_manager()
            skipped: List[Dict[str, Any]] = []
            if version is None:
                decision = cm.select_checkpoint()
                for ev in decision.evaluations:
                    if not ev.eligible:
                        skipped.append({"version": ev.record.manifest.version, "checkpoint_id": ev.record.manifest.checkpoint_id,
                                        "reasons": [getattr(r, "code", str(r)) for r in getattr(ev, "reasons", ())]})
                if decision.selected is None:
                    if decision.evaluations:
                        self._restore_report = {"status": "failed_all_invalid", "skipped": skipped}
                        raise CheckpointIntegrityError(f"{len(decision.evaluations)} checkpoint(s) exist but none is valid: {skipped}")
                    self._restore_report = {"status": "fresh_start", "skipped": []}
                    return None
                version = decision.selected.manifest.version
            try:
                res = cm.load_components(version, components=["model"], expected_codecs={"model": "numpy"})
            except Exception as exc:
                self._restore_report = {"status": "failed", "version": version, "error": f"{type(exc).__name__}: {exc}", "skipped": skipped}
                raise
            man = res.record.manifest
            md = dict(man.metadata)
            p = self.policy
            if md.get("schema_version") != SCHEMA_VERSION or md.get("algorithm") != ALGORITHM:
                raise CheckpointIntegrityError(f"incompatible checkpoint schema/algorithm: {md.get('schema_version')}/{md.get('algorithm')}")
            if int(md.get("action_dim", -1)) != p.action_dim or int(md.get("feature_dim", -1)) != p.feature_dim:
                raise CheckpointIntegrityError("checkpoint dimensions do not match this learner")
            p.load_state(res.components["model"], {k: md.get(k, 0) for k in ("training_step", "consumed_transitions", "consumed_episodes")},
                         expect_parameter_hash=md.get("parameter_hash"), expect_state_hash=md.get("state_hash"))
            lin = man.provenance.lineage
            self._checkpoint_id = man.checkpoint_id
            self._parent_checkpoint_id = lin.parent_checkpoint_id
            self._restored_from = man.checkpoint_id
            self._generation = int(lin.generation)
            self._last_saved_state_hash = p.state_hash
            self._last_checkpoint = CheckpointMetadata(
                checkpoint_id=man.checkpoint_id, version=man.version, path=None, state_hash=p.state_hash, parameter_hash=p.parameter_hash,
                hyperparameter_hash=str(md.get("hyperparameter_hash", "")), parent_checkpoint_id=lin.parent_checkpoint_id,
                training_step=p.training_step, consumed_transitions=p.consumed_transitions, consumed_episodes=p.consumed_episodes,
                schema_version=SCHEMA_VERSION, algorithm=ALGORITHM, created_at=man.created_at, saved=False)
            self._restore_report = {"status": "restored", "version": man.version, "checkpoint_id": man.checkpoint_id, "skipped": skipped,
                                    "hyperparameters_changed": md.get("hyperparameter_hash") != p.hyper.digest()}
            self._counters["checkpoints_restored"] += 1
            return self._last_checkpoint

    # ------------------------------------------------------------------ telemetry
    def reward_report(self) -> Dict[str, Any]:
        with self._lock:
            return {"train": self._rewards["train"].report(), "evaluate": self._rewards["evaluate"].report()}

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "policy": self.metadata().to_dict(), "mode": self.mode, "learning_enabled": self.learning_enabled,
                "store": self.store.stats(), "counters": dict(self._counters),
                "updates": {"applied": self._updates_applied, "skipped": dict(self._skips)},
                "last_update": None if self._last_update is None else self._last_update.to_dict(),
                "reward": self.reward_report(), "restore": dict(self._restore_report),
                "checkpoint": None if self._last_checkpoint is None else self._last_checkpoint.to_dict(),
                "last_error": self._last_error,
                "terminology": {"inference": "decide", "within_episode_adaptation": "not performed by this service",
                                "persistent_learning": "training_step / parameter_hash change", "fallback": "environment-side; label it",
                                "evaluation": "mode=evaluate never queues episodes for training"},
            }
