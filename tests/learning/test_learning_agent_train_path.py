"""LearningAgent: the train path must really update the strategy learner (regression for fb195fe0), say so truthfully when it does
not, keep evaluation read-only, and expose the external-environment entry point.  Runs the real methods against fake collaborators."""
from __future__ import annotations

import logging, os, sys, threading, unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np

from _support import import_learning_agent

LA, WHY = import_learning_agent()


class _Tensorish:
    def __init__(self, v): self.v = np.asarray(v, dtype=np.float32).reshape(-1)
    def numel(self): return self.v.size
    def detach(self): return self
    def cpu(self): return self
    def numpy(self): return self.v


class _Env:
    def __init__(self, steps=5): self.steps, self.t = steps, 0
    def reset(self, seed=None): self.t = 0; return np.zeros(4, np.float32), {}
    def step(self, a):
        self.t += 1
        return np.full(4, self.t, np.float32), 1.0, self.t >= self.steps, {}


class _Learner:
    def __init__(self, warmup=0): self.stored, self.trained, self.explore, self.warmup = 0, 0, [], warmup
    def select_action(self, s, explore=True): self.explore.append(explore); return 0
    def store_transition(self, *t): self.stored += 1; self.last = t
    def train(self):
        self.trained += 1
        return None if self.trained <= self.warmup else 0.25


def make_agent():
    a = LA.LearningAgent.__new__(LA.LearningAgent)
    a.env = _Env(); a._lock = threading.RLock()
    a.state_processor = mock.Mock(); a.state_processor.process = lambda st: _Tensorish(st)
    a.multi_task_learner = mock.MagicMock(); a.strategy_selector = mock.MagicMock(); a.strategy_selector.train_from_embeddings.return_value = None
    a.learning_factory = mock.MagicMock(); a.learning_calculations = mock.MagicMock()
    a.observation_count = 0; a.training_iterations = 0; a.learner_update_count = 0; a.learner_update_skips = 0
    a.performance_history = []; a.performance_metrics = {"strategy_loss": {}, "strategy_accuracy": {}}
    a._global_reward_stats = mock.MagicMock(); a._reward_stats = {}; a.error_history = []
    a._external_services = {}; a.learning_config = {}
    return a


@unittest.skipIf(LA is None, f"LearningAgent cannot be imported here: {WHY}")
class TrainPath(unittest.TestCase):
    def test_train_episode_updates_the_learner_every_step_and_reports_it(self):
        a, lr = make_agent(), _Learner()
        rep = a._run_episode("dqn", lr, max_steps=50, seed=1, train=True)
        self.assertEqual((rep["steps"], lr.stored, lr.trained), (5, 5, 5))
        self.assertEqual((rep["learner_updates"], rep["loss_samples"], rep["avg_loss"], rep["trained"]), (5, 5, 0.25, True))
        self.assertEqual((a.learner_update_count, a.learner_update_skips), (5, 0))
        s, act, r, ns, done = lr.last
        self.assertEqual((act, r, done), (0, 1.0, True)); np.testing.assert_array_equal(s, np.full(4, 4)); np.testing.assert_array_equal(ns, np.full(4, 5))
        self.assertTrue(all(lr.explore))                                              # training explores

    def test_warmup_is_reported_as_no_update_not_as_zero_loss(self):
        a, lr = make_agent(), _Learner(warmup=99)
        with self.assertLogs(level="WARNING") as logs:
            rep = a._run_episode("dqn", lr, max_steps=50, seed=1, train=True)
        self.assertIsNone(rep["avg_loss"]); self.assertFalse(rep["trained"]); self.assertEqual((rep["learner_updates"], rep["learner_update_skips"]), (0, 5))
        self.assertEqual(a.learner_update_count, 0); self.assertEqual(a.learner_update_skips, 5)
        self.assertTrue(any("without any learner update" in m for m in logs.output))

    def test_learner_without_train_is_visible_as_skipped(self):
        class NoTrain:
            def select_action(self, s, explore=True): return 0
        a = make_agent()
        with self.assertLogs(level="WARNING"):
            rep = a._run_episode("rsi", NoTrain(), max_steps=10, seed=0, train=True)
        self.assertEqual((rep["trained"], rep["learner_update_skips"]), (False, 5))

    def test_evaluation_never_touches_the_learner(self):
        a, lr = make_agent(), _Learner()
        rep = a._run_episode("dqn", lr, max_steps=50, seed=None, train=False)
        self.assertEqual((lr.stored, lr.trained, a.learner_update_count), (0, 0, 0))
        self.assertFalse(any(lr.explore)); self.assertFalse(rep["trained"]); self.assertIsNone(rep["avg_loss"])
        a.multi_task_learner.rebalance.assert_not_called(); a.strategy_selector.train_from_embeddings.assert_not_called()

    def test_train_cycles_summarises_real_updates_and_warns_when_there_were_none(self):
        for warmup, expect_trained in ((0, True), (999, False)):
            a, lr = make_agent(), _Learner(warmup=warmup)
            a.max_episode_steps = 50; a.recovery_trigger_threshold = 99; a.retraining_interval = None
            a._select_strategy = lambda s, m: ("dqn", lr, {}); a._record_shared_memory = lambda *args: None
            a._performance_trend = lambda: 0.0; a.recovery_system = mock.MagicMock()
            from datetime import timedelta
            with mock.patch.object(LA, "utc_now", create=True):
                res = a._train_cycles({"episodes": 2, "seed": 1})
            self.assertEqual(res["trained"], expect_trained)
            self.assertEqual(res["learner_updates"], 10 if expect_trained else 0)
            self.assertEqual("warning" in res, not expect_trained)
            self.assertEqual(a.training_iterations, 2)                                 # episodes, kept as before - not mistaken for updates

    def test_checkpoint_state_carries_update_counters_and_old_checkpoints_still_load(self):
        a = make_agent(); a.CHECKPOINT_SCHEMA = LA.LearningAgent.CHECKPOINT_SCHEMA
        a.task_ids = ["dqn"]; a.learner_update_count, a.learner_update_skips = 7, 3
        from datetime import datetime
        a.last_training_time = datetime(2026, 1, 1); a.multi_task_learner.get_weights.return_value = {"dqn": 1.0}
        a.learning_factory.snapshot_learning_state.return_value = {}; a.strategy_selector.snapshot.return_value = {}
        a.performance_history = []
        st = a._export_checkpoint_state()
        self.assertEqual((st["learner_update_count"], st["learner_update_skips"]), (7, 3))
        a.performance_history = __import__("collections").deque(); a._global_reward_stats = LA.RunningStats()
        a.learner_update_count = a.learner_update_skips = 0
        a._import_checkpoint_state(dict(st)); self.assertEqual((a.learner_update_count, a.learner_update_skips), (7, 3))
        old = {k: v for k, v in st.items() if k not in ("learner_update_count", "learner_update_skips")}
        a._import_checkpoint_state(old); self.assertEqual((a.learner_update_count, a.learner_update_skips), (0, 0))

    def test_external_mode_round_trip_through_perform_task(self):
        a = make_agent()
        def call(**kw): return a.perform_task({"mode": "external", **kw})
        r = call(op="configure", name="head", feature_dim=3, action_dim=3, hyper={"learning_rate": 0.5}); self.assertEqual(r["status"], "ok")
        self.assertTrue(r["created"]); self.assertFalse(r["persistent"])
        self.assertFalse(call(op="configure", name="head", feature_dim=3, action_dim=3, hyper={"learning_rate": 0.5})["created"])   # idempotent
        self.assertEqual(call(op="configure", name="head", feature_dim=4, action_dim=3)["code"], "learner_exists")
        self.assertEqual(call(op="start_episode", name="head", episode_id="e1")["status"], "ok")
        d = call(op="decide", name="head", episode_id="e1", decision_id="e1/0", features=[1, 0, 0], action_mask=[True, True, False])
        self.assertEqual(d["status"], "ok"); self.assertIn(d["action"], (0, 1)); self.assertEqual(d["source"], "learned")
        self.assertEqual(call(op="feedback", name="head", episode_id="e1", decision_id="e1/0", reward=1.0)["status"], "ok")
        bad = call(op="decide", name="head", episode_id="e1", decision_id="e1/1", features=[1, float("nan"), 0])
        self.assertEqual((bad["status"], bad["code"]), ("rejected", "non_finite"))
        end = call(op="end_episode", name="head", episode_id="e1", terminal_reward=1.0)
        self.assertTrue(end["update"]["updated"]); self.assertEqual(end["policy"]["training_step"], 1)
        self.assertEqual(call(op="start_episode", name="head", episode_id="e1")["code"], "duplicate_episode")
        self.assertEqual(call(op="decide", name="nope", episode_id="x", decision_id="y", features=[1])["code"], "unknown_learner")
        self.assertEqual(a.perform_task({"mode": "external", "op": "bogus"})["code"], "unknown_op")
        st = call(op="status", name="head")["learners"]["head"]
        self.assertEqual(st["policy"]["training_step"], 1); self.assertEqual(st["counters"]["rejected:duplicate_episode"], 1)

    def test_unsupported_mode_message_lists_the_real_modes(self):
        a = make_agent()
        with self.assertRaises(Exception) as cm: a.perform_task({"mode": "bogus"})
        self.assertIn("external", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
