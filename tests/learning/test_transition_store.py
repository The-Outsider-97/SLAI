"""TransitionStore: isolation, exactly-once episodes/decisions, terminal handling, bounds, rejections.  numpy only."""
from __future__ import annotations

import os, sys, threading, unittest

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np

from src.agents.learning.contracts import ContractError, Observation, TransitionRejected
from src.agents.learning.transition_store import TransitionStore


def obs(v: float = 0.0, dim: int = 3) -> Observation:
    return Observation.make([v] * dim, feature_dim=dim)


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.s = TransitionStore(max_live_episodes=4, max_decisions_per_episode=50, max_pending_episodes=3, remembered_episode_ids=8)

    def code(self, fn, *a, **k):
        with self.assertRaises(TransitionRejected) as cm:
            fn(*a, **k)
        return cm.exception.code

    def test_transition_has_every_required_field(self):
        self.s.start_episode("e1", policy_version="v7")
        self.s.add_decision("e1", "d0", obs(1), 2, action_probability=0.4, policy_version="v7")
        self.s.add_decision("e1", "d1", obs(2), 1, action_probability=0.6, policy_version="v7")
        self.s.add_reward("e1", "d0", 0.5)
        rec = self.s.end_episode("e1", terminal_reward=1.0)
        t0, t1 = rec.transitions
        for t in (t0, t1):
            self.assertEqual(t.episode_id, "e1"); self.assertEqual(t.policy_version, "v7")
            self.assertGreater(t.timestamp, 0); self.assertIsInstance(t.sequence, int)
        self.assertEqual((t0.decision_id, t1.decision_id), ("d0", "d1"))
        self.assertEqual((t0.reward, t1.reward), (0.5, 1.0))
        np.testing.assert_array_equal(t0.next_observation.features, t1.observation.features)   # next_observation chained
        self.assertIsNone(t1.next_observation)
        self.assertEqual((t0.terminated, t0.truncated, t1.terminated, t1.truncated), (False, False, True, False))
        self.assertEqual(rec.total_reward, 1.5)

    def test_truncated_episode_flags_last_transition_only(self):
        self.s.start_episode("e"); self.s.add_decision("e", "a", obs(), 0); self.s.add_decision("e", "b", obs(), 0)
        rec = self.s.end_episode("e", terminated=False, truncated=True)
        self.assertEqual([t.truncated for t in rec.transitions], [False, True])
        self.assertEqual([t.terminated for t in rec.transitions], [False, False])

    def test_episodes_are_isolated(self):
        for e in ("a", "b"):
            self.s.start_episode(e)
        self.s.add_decision("a", "d", obs(1), 0); self.s.add_decision("b", "d", obs(2), 1)   # same decision id, different episodes
        ra = self.s.end_episode("a"); rb = self.s.end_episode("b")
        self.assertEqual([t.action for t in ra.transitions], [0]); self.assertEqual([t.action for t in rb.transitions], [1])
        self.assertEqual(ra.transitions[0].observation.features[0], 1.0); self.assertEqual(rb.transitions[0].observation.features[0], 2.0)

    def test_duplicate_episode_rejected_live_and_finished(self):
        self.s.start_episode("e")
        self.assertEqual(self.code(self.s.start_episode, "e"), "duplicate_episode")
        self.s.end_episode("e")
        self.assertEqual(self.code(self.s.start_episode, "e"), "duplicate_episode")        # a replayed episode id cannot be re-learned
        self.assertEqual(self.s.rejections["duplicate_episode"], 2)

    def test_duplicate_decision_and_sequence_rejected(self):
        self.s.start_episode("e"); self.s.add_decision("e", "d0", obs(), 0, sequence=0)
        self.assertEqual(self.code(self.s.add_decision, "e", "d0", obs(), 0), "duplicate_decision")
        self.assertEqual(self.code(self.s.add_decision, "e", "other", obs(), 0, sequence=0), "duplicate_decision")
        self.assertEqual(len(self.s.end_episode("e").transitions), 1)

    def test_unknown_and_closed_episode(self):
        self.assertEqual(self.code(self.s.add_decision, "nope", "d", obs(), 0), "unknown_episode")
        self.s.start_episode("e"); self.s.end_episode("e")
        self.assertEqual(self.code(self.s.add_decision, "e", "d", obs(), 0), "episode_closed")
        self.assertEqual(self.code(self.s.add_reward, "e", "d", 1.0), "episode_closed")
        self.assertEqual(self.code(self.s.end_episode, "e"), "episode_closed")              # terminal applied exactly once

    def test_out_of_order_decisions_sorted_and_gaps_counted(self):
        self.s.start_episode("e")
        self.s.add_decision("e", "c", obs(3), 0, sequence=2); self.s.add_decision("e", "a", obs(1), 0, sequence=0)
        rec = self.s.end_episode("e", terminal_reward=1.0)
        self.assertEqual([t.decision_id for t in rec.transitions], ["a", "c"])
        self.assertEqual(rec.transitions[-1].reward, 1.0)
        self.assertEqual(self.s.counters["out_of_order_decisions"], 1); self.assertEqual(self.s.counters["sequence_gaps"], 1)

    def test_abort_never_learns_and_blocks_reuse(self):
        self.s.start_episode("e"); self.s.add_decision("e", "d", obs(), 0)
        self.assertTrue(self.s.abort_episode("e", "timeout"))
        self.assertEqual(self.s.pending_count(), 0)
        self.assertEqual(self.code(self.s.start_episode, "e"), "duplicate_episode")
        self.assertEqual(self.code(self.s.abort_episode, "never-started"), "unknown_episode")

    def test_invalid_values_rejected_not_coerced(self):
        self.s.start_episode("e"); self.s.add_decision("e", "d", obs(), 0)
        for bad in (float("nan"), float("inf"), "x", None, True):
            with self.assertRaises(TransitionRejected):
                self.s.add_reward("e", "d", bad)
        with self.assertRaises(TransitionRejected): self.s.add_decision("e", "d2", obs(), -1)
        with self.assertRaises(TransitionRejected): self.s.add_decision("e", "d3", obs(), 1.5)
        with self.assertRaises(TransitionRejected): self.s.end_episode("e", terminal_reward=float("nan"))
        self.assertTrue(self.s.is_live("e"))                                                  # a rejected end leaves the episode intact
        self.assertEqual(len(self.s.end_episode("e").transitions), 1)

    def test_unknown_decision_for_reward(self):
        self.s.start_episode("e")
        self.assertEqual(self.code(self.s.add_reward, "e", "missing", 1.0), "unknown_decision")

    def test_bounds_are_enforced_and_counted(self):
        for i in range(6):
            self.s.start_episode(f"live{i}")
        self.assertEqual(self.s.live_count(), 4); self.assertEqual(self.s.counters["episodes_evicted_abandoned"], 2)
        self.assertEqual(self.code(self.s.add_decision, "live0", "d", obs(), 0), "episode_closed")   # evicted -> cannot be revived
        for i in range(5):
            e = f"p{i}"; self.s.start_episode(e); self.s.add_decision(e, "d", obs(), 0); self.s.end_episode(e)
        self.assertEqual(self.s.pending_count(), 3); self.assertEqual(self.s.counters["pending_episodes_dropped_overflow"], 2)
        e = "big"; self.s.abort_episode("live2"); self.s.start_episode(e)
        for i in range(50):
            self.s.add_decision(e, f"d{i}", obs(), 0)
        self.assertEqual(self.code(self.s.add_decision, e, "d50", obs(), 0), "too_many_decisions")

    def test_take_hands_each_episode_out_once_and_give_back_restores(self):
        for e in ("a", "b"):
            self.s.start_episode(e); self.s.add_decision(e, "d", obs(), 0); self.s.end_episode(e)
        got = self.s.take(1)
        self.assertEqual([r.episode_id for r in got], ["a"]); self.assertEqual(self.s.pending_count(), 1)
        self.s.give_back(got)
        self.assertEqual([r.episode_id for r in self.s.take()], ["a", "b"]); self.assertEqual(self.s.take(), [])

    def test_retain_false_is_not_queued(self):
        self.s.start_episode("e", mode="evaluate"); self.s.add_decision("e", "d", obs(), 0)
        self.s.end_episode("e", retain=False)
        self.assertEqual(self.s.pending_count(), 0); self.assertEqual(self.s.counters["episodes_not_retained"], 1)

    def test_concurrent_episodes_do_not_mix(self):
        s = TransitionStore(max_live_episodes=64)
        errors = []
        def worker(k):
            try:
                e = f"w{k}"; s.start_episode(e)
                for i in range(100):
                    s.add_decision(e, f"d{i}", obs(float(k)), k % 4, sequence=i)
                s.end_episode(e, terminal_reward=float(k))
            except Exception as exc:   # noqa: BLE001
                errors.append(exc)
        th = [threading.Thread(target=worker, args=(k,)) for k in range(16)]
        [t.start() for t in th]; [t.join() for t in th]
        self.assertEqual(errors, [])
        for r in s.take():
            k = int(r.episode_id[1:])
            self.assertEqual(len(r.transitions), 100)
            self.assertTrue(all(t.action == k % 4 and t.observation.features[0] == k for t in r.transitions))
            self.assertEqual(r.transitions[-1].reward, float(k))

    def test_observation_contract_rejects_bad_vectors(self):
        for bad in ([], [float("nan")], [[1.0]], ["a"]):
            with self.assertRaises(ContractError): Observation.make(bad)
        with self.assertRaises(ContractError): Observation.make([1, 2], feature_dim=3)
        with self.assertRaises(ContractError): Observation.make([1, 2], action_mask=[False, False], action_dim=2)
        with self.assertRaises(ContractError): Observation.make([1, 2], prior_logits=[0, float("inf")], action_dim=2)


if __name__ == "__main__":
    unittest.main()
