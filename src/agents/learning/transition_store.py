"""Episode-aware, bounded transition storage for external environments.

Guarantees (each covered by tests/learning/test_transition_store.py):

* episodes are isolated - a decision can only be added to a live episode and never crosses into another one;
* an episode id is accepted once - a repeated ``start_episode`` for a live or already-finished id is rejected, so a replayed
  or retried episode cannot be learned from twice;
* a decision id (and sequence number) is accepted once per episode, so a retried request cannot duplicate a transition;
* nothing can be added after the episode ended; the terminal flags and terminal reward are applied exactly once;
* decisions may arrive out of order - they are ordered by ``sequence`` when the episode ends and gaps are counted;
* every bound is explicit (live episodes, decisions per episode, pending episodes, remembered ids) and every drop is counted
  and visible in ``stats()``; nothing is cleared silently;
* rejections raise ``TransitionRejected`` with a stable ``code`` - they are never swallowed.

The store knows nothing about the environment: observations are finite vectors, actions are integers, rewards are floats.
"""

from __future__ import annotations

import threading
import time
from collections import Counter, OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

from .contracts import (
    ContractError,
    EpisodeRecord,
    Observation,
    Transition,
    TransitionRejected,
    finite_reward,
    require_id,
)


@dataclass
class _Pending:
    decision_id: str
    sequence: int
    observation: Observation
    action: int
    action_probability: float
    policy_version: str
    timestamp: float
    reward: float = 0.0
    reward_events: int = 0
    executed_action: Optional[int] = None


@dataclass
class _Live:
    episode_id: str
    started_at: float
    policy_version: str
    mode: str
    decisions: Dict[str, _Pending] = field(default_factory=dict)
    sequences: set = field(default_factory=set)
    next_sequence: int = 0


class TransitionStore:
    def __init__(self, *, max_live_episodes: int = 64, max_decisions_per_episode: int = 20_000,
                 max_pending_episodes: int = 256, remembered_episode_ids: int = 4096) -> None:
        for name, v in (("max_live_episodes", max_live_episodes), ("max_decisions_per_episode", max_decisions_per_episode),
                        ("max_pending_episodes", max_pending_episodes), ("remembered_episode_ids", remembered_episode_ids)):
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise ValueError(f"{name} must be a positive integer, got {v!r}")
        self.max_live_episodes = max_live_episodes
        self.max_decisions_per_episode = max_decisions_per_episode
        self.max_pending_episodes = max_pending_episodes
        self.remembered_episode_ids = remembered_episode_ids
        self._lock = threading.RLock()
        self._live: "OrderedDict[str, _Live]" = OrderedDict()
        self._finished: "OrderedDict[str, str]" = OrderedDict()   # episode_id -> how it finished (bounded memory of ids)
        self._pending: Deque[EpisodeRecord] = deque()
        self.counters: Counter = Counter()
        self.rejections: Counter = Counter()

    # ------------------------------------------------------------------ helpers
    def _reject(self, code: str, message: str) -> None:
        self.rejections[code] += 1
        raise TransitionRejected(code, message)

    def _reward(self, value: Any, name: str) -> float:
        try:
            return finite_reward(value, name=name)
        except ContractError as exc:
            self.rejections[exc.code] += 1
            raise TransitionRejected(exc.code, exc.message) from exc

    def _remember(self, episode_id: str, how: str) -> None:
        self._finished[episode_id] = how
        self._finished.move_to_end(episode_id)
        while len(self._finished) > self.remembered_episode_ids:
            self._finished.popitem(last=False)

    def _live_episode(self, episode_id: str) -> _Live:
        ep = self._live.get(episode_id)
        if ep is not None:
            return ep
        if episode_id in self._finished:
            self._reject("episode_closed", f"episode {episode_id!r} already {self._finished[episode_id]}")
        self._reject("unknown_episode", f"episode {episode_id!r} was never started")
        raise AssertionError  # pragma: no cover

    # ------------------------------------------------------------------ lifecycle
    def start_episode(self, episode_id: str, *, policy_version: str = "", mode: str = "train",
                      timestamp: Optional[float] = None) -> None:
        require_id(episode_id, name="episode_id")
        with self._lock:
            if episode_id in self._live:
                self._reject("duplicate_episode", f"episode {episode_id!r} is already live")
            if episode_id in self._finished:
                self._reject("duplicate_episode", f"episode {episode_id!r} already {self._finished[episode_id]}")
            while len(self._live) >= self.max_live_episodes:
                old_id, _ = self._live.popitem(last=False)   # an abandoned game (quit, crash, timeout) must not accumulate
                self._remember(old_id, "evicted")
                self.counters["episodes_evicted_abandoned"] += 1
            self._live[episode_id] = _Live(episode_id, timestamp if timestamp is not None else time.time(), policy_version, mode)
            self.counters["episodes_started"] += 1

    def add_decision(self, episode_id: str, decision_id: str, observation: Observation, action: int, *,
                     action_probability: float = 1.0, policy_version: str = "", sequence: Optional[int] = None,
                     timestamp: Optional[float] = None) -> int:
        """Record a decision; returns its sequence number."""
        require_id(decision_id, name="decision_id")
        if isinstance(action, bool) or not isinstance(action, int) or action < 0:
            self._reject("invalid_action", f"action must be a non-negative int, got {action!r}")
        with self._lock:
            ep = self._live_episode(episode_id)
            if decision_id in ep.decisions:
                self._reject("duplicate_decision", f"decision {decision_id!r} already recorded in episode {episode_id!r}")
            if len(ep.decisions) >= self.max_decisions_per_episode:
                self._reject("too_many_decisions", f"episode {episode_id!r} reached {self.max_decisions_per_episode} decisions")
            seq = ep.next_sequence if sequence is None else sequence
            if isinstance(seq, bool) or not isinstance(seq, int) or seq < 0:
                self._reject("invalid_sequence", f"sequence must be a non-negative int, got {seq!r}")
            if seq in ep.sequences:
                self._reject("duplicate_decision", f"sequence {seq} already recorded in episode {episode_id!r}")
            if sequence is not None and seq < ep.next_sequence:
                self.counters["out_of_order_decisions"] += 1
            ep.sequences.add(seq)
            ep.next_sequence = max(ep.next_sequence, seq + 1)
            ep.decisions[decision_id] = _Pending(decision_id, seq, observation, action, float(action_probability), policy_version,
                                                 timestamp if timestamp is not None else time.time())
            self.counters["decisions_recorded"] += 1
            return seq

    def add_reward(self, episode_id: str, decision_id: str, reward: Any) -> None:
        """Credit ``reward`` to a decision.  Several rewards for one decision accumulate (shaping + delayed feedback)."""
        r = self._reward(reward, "reward")
        with self._lock:
            ep = self._live_episode(episode_id)
            d = ep.decisions.get(decision_id)
            if d is None:
                self._reject("unknown_decision", f"decision {decision_id!r} not in episode {episode_id!r}")
            d.reward += r
            d.reward_events += 1
            self.counters["rewards_recorded"] += 1

    def mark_executed(self, episode_id: str, decision_id: str, executed_action: int) -> None:
        """Record that the environment executed ``executed_action`` for this decision (equal or different from the policy's)."""
        with self._lock:
            ep = self._live_episode(episode_id)
            d = ep.decisions.get(decision_id)
            if d is None:
                self._reject("unknown_decision", f"decision {decision_id!r} not in episode {episode_id!r}")
            d.executed_action = int(executed_action)

    def end_episode(self, episode_id: str, *, terminal_reward: Any = 0.0, terminated: bool = True, truncated: bool = False,
                    retain: bool = True, timestamp: Optional[float] = None) -> EpisodeRecord:
        """Close an episode exactly once.  ``retain=False`` (evaluation) finalizes it but never queues it for training."""
        tr = self._reward(terminal_reward, "terminal_reward")
        with self._lock:
            ep = self._live_episode(episode_id)
            ordered = sorted(ep.decisions.values(), key=lambda p: p.sequence)
            if ordered:
                gaps = ordered[-1].sequence + 1 - len(ordered)
                if gaps > 0:
                    self.counters["sequence_gaps"] += gaps
                ordered[-1].reward += tr   # the terminal outcome is credited to the last decision (delayed credit)
                if tr != 0.0:
                    ordered[-1].reward_events += 1
            else:
                self.counters["episodes_without_decisions"] += 1
            transitions: List[Transition] = []
            for i, p in enumerate(ordered):
                last = i == len(ordered) - 1
                transitions.append(Transition(
                    episode_id=episode_id, decision_id=p.decision_id, sequence=p.sequence, observation=p.observation, action=p.action,
                    reward=p.reward, next_observation=None if last else ordered[i + 1].observation,
                    terminated=bool(terminated) and last, truncated=bool(truncated) and last,
                    policy_version=p.policy_version, timestamp=p.timestamp, action_probability=p.action_probability,
                    executed_action=p.executed_action))
            self.counters["decisions_without_reward_event"] += sum(1 for p in ordered if p.reward_events == 0)
            record = EpisodeRecord(
                episode_id=episode_id, transitions=tuple(transitions), terminated=bool(terminated), truncated=bool(truncated),
                total_reward=float(sum(t.reward for t in transitions)), started_at=ep.started_at,
                ended_at=timestamp if timestamp is not None else time.time(), policy_version_start=ep.policy_version, mode=ep.mode)
            del self._live[episode_id]
            self._remember(episode_id, "ended")
            self.counters["episodes_ended"] += 1
            if retain and transitions:
                self._pending.append(record)
                while len(self._pending) > self.max_pending_episodes:
                    self._pending.popleft()
                    self.counters["pending_episodes_dropped_overflow"] += 1
            elif not retain:
                self.counters["episodes_not_retained"] += 1
            return record

    def abort_episode(self, episode_id: str, reason: str = "aborted") -> bool:
        """Discard a live episode (timeout, quit, crash).  Its decisions are never learned from; the id cannot be reused."""
        with self._lock:
            if episode_id in self._live:
                del self._live[episode_id]
                self._remember(episode_id, f"aborted ({reason})")
                self.counters["episodes_aborted"] += 1
                return True
            if episode_id in self._finished:
                return False
            self._reject("unknown_episode", f"episode {episode_id!r} was never started")
            return False  # pragma: no cover

    # ------------------------------------------------------------------ consumption
    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)

    def pending_transitions(self) -> int:
        with self._lock:
            return sum(len(r.transitions) for r in self._pending)

    def take(self, max_episodes: Optional[int] = None) -> List[EpisodeRecord]:
        """Remove and return pending episodes (oldest first).  Each episode is handed out exactly once."""
        with self._lock:
            n = len(self._pending) if max_episodes is None else min(max_episodes, len(self._pending))
            out = [self._pending.popleft() for _ in range(n)]
            self.counters["episodes_taken"] += len(out)
            self.counters["transitions_taken"] += sum(len(r.transitions) for r in out)
            return out

    def give_back(self, records: List[EpisodeRecord]) -> None:
        """Return un-consumed episodes to the front of the queue (an update that could not run must not lose data)."""
        with self._lock:
            for r in reversed(records):
                self._pending.appendleft(r)
            self.counters["episodes_taken"] -= len(records)
            self.counters["transitions_taken"] -= sum(len(r.transitions) for r in records)

    # ------------------------------------------------------------------ introspection
    def is_live(self, episode_id: str) -> bool:
        with self._lock:
            return episode_id in self._live

    def live_count(self) -> int:
        with self._lock:
            return len(self._live)

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {"live_episodes": len(self._live), "pending_episodes": len(self._pending),
                    "pending_transitions": sum(len(r.transitions) for r in self._pending),
                    "remembered_episode_ids": len(self._finished), "counters": dict(self.counters),
                    "rejections": dict(self.rejections),
                    "bounds": {"max_live_episodes": self.max_live_episodes, "max_decisions_per_episode": self.max_decisions_per_episode,
                               "max_pending_episodes": self.max_pending_episodes, "remembered_episode_ids": self.remembered_episode_ids}}
