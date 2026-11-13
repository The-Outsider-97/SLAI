"""ExternalLearningService on generic toy environments (contextual bandit, corridor): learning, evaluation safety, lifecycle."""
from __future__ import annotations

import os, sys, threading, unittest

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np

from _support import ContextualBandit, Corridor, eval_bandit, make_service, run_bandit_episode
from src.agents.learning.contracts import ContractError, Observation, TransitionRejected
from src.agents.learning.external_learning import ExternalLearningConfig, ExternalLearningService
from src.agents.learning.softmax_policy import PolicyHyperparameters

SEEDS = list(range(1000, 1020))


def train(svc, env, n, seed=0, delayed=False, prefix="t"):
    rng = np.random.default_rng(seed)
    last = None
    for i in range(n):
        _, _, last = run_bandit_episode(svc, env, f"{prefix}{i}", rng, delayed=delayed)
    return last


class LearningImproves(unittest.TestCase):
    def test_policy_clearly_improves_and_every_stage_is_observable(self):
        env = ContextualBandit()
        svc = make_service(env, seed=1)
        h0, id0 = svc.policy.parameter_hash, svc.policy.policy_id
        before = eval_bandit(svc, env, SEEDS, "pre")
        self.assertEqual(svc.policy.parameter_hash, h0)                                       # evaluation did not train
        train(svc, env, 150)
        after = eval_bandit(svc, env, SEEDS, "post")
        self.assertLess(before, 0.5); self.assertGreater(after, 0.9, f"learning did not improve: {before} -> {after}")
        md = svc.metadata()
        self.assertNotEqual(md.parameter_hash, h0); self.assertNotEqual(md.policy_id, id0)
        self.assertEqual(md.training_step, svc.status()["updates"]["applied"]); self.assertGreater(md.training_step, 50)
        self.assertEqual(md.consumed_episodes, 150); self.assertEqual(md.consumed_transitions, 150 * env.steps)
        lu = svc.status()["last_update"]
        self.assertIsNotNone(lu)

    def test_delayed_terminal_reward_only_still_learns(self):
        env = ContextualBandit(contexts=3, actions=3, steps=4)
        svc = make_service(env, seed=2, lr=0.3)
        before = eval_bandit(svc, env, SEEDS, "pre")
        train(svc, env, 400, seed=3, delayed=True)
        after = eval_bandit(svc, env, SEEDS, "post")
        self.assertGreater(after, before + 0.15, f"{before} -> {after}")

    def test_corridor_delayed_reward(self):
        env = Corridor(n=6)
        svc = make_service(env, seed=4, lr=0.5, hyper=PolicyHyperparameters(learning_rate=0.5, gamma=0.95))
        rng = np.random.default_rng(0)

        def run(ep, mode=None):
            svc.start_episode(ep, mode=mode); pos = int(rng.integers(1, env.n - 1)); steps = 0
            while 0 < pos < env.n - 1 and steps < env.max_steps:
                d = svc.decide(ep, f"{ep}/{steps}", env.obs(pos), sequence=steps)
                pos += 1 if d.action.index == 1 else -1; steps += 1
            reached = pos == env.n - 1
            svc.end_episode(ep, terminal_reward=1.0 if reached else 0.0, terminated=reached or pos == 0, truncated=not (reached or pos == 0))
            return reached, steps

        base = np.mean([run(f"pre{i}", "evaluate")[0] for i in range(40)])
        for i in range(300): run(f"tr{i}")
        post = np.mean([run(f"post{i}", "evaluate")[0] for i in range(40)])
        self.assertGreater(post, max(base + 0.3, 0.9), f"{base} -> {post}")


class EvaluationNeverTrains(unittest.TestCase):
    def test_evaluate_mode_records_but_never_updates(self):
        env = ContextualBandit(); svc = make_service(env)
        h0 = svc.policy.state_hash
        for i in range(20):
            _, _, res = run_bandit_episode(svc, env, f"ev{i}", np.random.default_rng(i), mode="evaluate")
            self.assertFalse(res.updated); self.assertEqual(res.skip_reason, "evaluation_mode")
        st = svc.status()
        self.assertEqual(svc.policy.state_hash, h0); self.assertEqual(st["policy"]["training_step"], 0)
        self.assertEqual(st["store"]["pending_episodes"], 0); self.assertEqual(st["updates"]["applied"], 0)
        self.assertEqual(st["reward"]["evaluate"]["episodes"], 20); self.assertEqual(st["reward"]["train"]["episodes"], 0)

    def test_service_level_mode_and_learning_enabled_switches(self):
        env = ContextualBandit(); svc = make_service(env, mode="evaluate")
        h0 = svc.policy.state_hash
        train(svc, env, 10)                                                     # mode=evaluate by default for every episode
        self.assertEqual(svc.policy.state_hash, h0); self.assertFalse(svc.metadata().learning_enabled)
        svc.set_mode("train"); svc.set_learning_enabled(False)
        train(svc, env, 10, prefix="d")
        self.assertEqual(svc.policy.state_hash, h0); self.assertFalse(svc.metadata().learning_enabled)
        self.assertEqual(svc.status()["updates"]["skipped"].get("evaluation_mode"), 20)
        svc.set_learning_enabled(True); train(svc, env, 5, prefix="e")
        self.assertNotEqual(svc.policy.state_hash, h0)
        with self.assertRaises(ContractError): svc.set_mode("fast")

    def test_episode_keeps_the_mode_it_started_with(self):
        env = ContextualBandit(); svc = make_service(env)
        svc.start_episode("a"); svc.decide("a", "a0", env.obs(0))
        svc.set_mode("evaluate")
        svc.decide("a", "a1", env.obs(1)); svc.feedback("a", "a0", 1.0)
        res = svc.end_episode("a", terminal_reward=1.0)
        self.assertTrue(res.updated)                                            # started in train mode -> trains, whatever happened since

    def test_learning_configured_but_unable_to_persist_fails_visibly(self):
        with self.assertRaises(ValueError):
            ExternalLearningConfig(name="x", feature_dim=2, action_dim=2, autosave_every_updates=1).validate()


class LifecycleAndRobustness(unittest.TestCase):
    def test_duplicate_episode_is_not_learned_twice(self):
        env = ContextualBandit(); svc = make_service(env)
        train(svc, env, 1, prefix="x")
        h = svc.policy.state_hash
        with self.assertRaises(TransitionRejected) as cm: svc.start_episode("x0")
        self.assertEqual(cm.exception.code, "duplicate_episode")
        with self.assertRaises(TransitionRejected): svc.decide("x0", "again", env.obs(0))
        with self.assertRaises(TransitionRejected): svc.end_episode("x0", terminal_reward=1.0)
        self.assertEqual(svc.policy.state_hash, h); self.assertEqual(svc.status()["counters"]["rejected:duplicate_episode"], 1)

    def test_invalid_observations_are_rejected_visibly_and_change_nothing(self):
        env = ContextualBandit(); svc = make_service(env); svc.start_episode("e")
        h = svc.policy.state_hash
        for bad in (Observation.make([1.0] * (env.feature_dim + 1)), Observation.make([0.0] * env.feature_dim, prior_logits=[0.0] * (env.A + 1))):
            with self.assertRaises(ContractError): svc.decide("e", "d", bad)
        with self.assertRaises(ContractError): Observation.make([float("nan")] * env.feature_dim)
        with self.assertRaises(ContractError): Observation.make([0.0] * env.feature_dim, action_mask=[False] * env.A)
        self.assertEqual(svc.store.stats()["counters"].get("decisions_recorded", 0), 0)
        self.assertEqual(svc.policy.state_hash, h)
        self.assertGreaterEqual(sum(v for k, v in svc.status()["counters"].items() if k.startswith("decide_rejected")), 2)

    def test_unknown_and_closed_episodes_are_named(self):
        env = ContextualBandit(); svc = make_service(env)
        with self.assertRaises(TransitionRejected) as cm: svc.decide("ghost", "d", env.obs(0))
        self.assertEqual(cm.exception.code, "unknown_episode")
        with self.assertRaises(TransitionRejected) as cm: svc.feedback("ghost", "d", 1.0)
        self.assertEqual(cm.exception.code, "unknown_episode")

    def test_aborted_episode_is_never_learned(self):
        env = ContextualBandit(); svc = make_service(env); svc.start_episode("a")
        for t in range(5): svc.decide("a", f"a{t}", env.obs(t % env.K)); svc.feedback("a", f"a{t}", 1.0)
        h = svc.policy.state_hash
        self.assertTrue(svc.abort_episode("a", "timeout"))
        self.assertEqual(svc.policy.state_hash, h); self.assertEqual(svc.store.pending_count(), 0)
        with self.assertRaises(TransitionRejected): svc.end_episode("a", terminal_reward=1.0)

    def test_missing_feedback_is_counted_and_does_not_crash(self):
        env = ContextualBandit(); svc = make_service(env); svc.start_episode("m")
        for t in range(4): svc.decide("m", f"m{t}", env.obs(t % env.K))
        res = svc.end_episode("m")                                              # no reward ever delivered
        self.assertFalse(res.updated); self.assertEqual(res.skip_reason, "zero_gradient")
        self.assertEqual(svc.status()["store"]["counters"]["decisions_without_reward_event"], 4)

    def test_delayed_feedback_arrives_after_later_decisions(self):
        env = ContextualBandit(); svc = make_service(env); svc.start_episode("d")
        for t in range(3): svc.decide("d", f"d{t}", env.obs(0), sequence=t)
        svc.feedback("d", "d0", 1.0)                                            # reward for the first decision arrives last
        res = svc.end_episode("d")
        self.assertTrue(res.updated)

    def test_out_of_order_decisions_are_ordered_at_the_end(self):
        env = ContextualBandit(); svc = make_service(env); svc.start_episode("o")
        for t in (2, 0, 1): svc.decide("o", f"o{t}", env.obs(t), sequence=t)
        rec = svc.store.end_episode("o", retain=False)                          # inspect the finalized record
        self.assertEqual([t.decision_id for t in rec.transitions], ["o0", "o1", "o2"])

    def test_concurrent_episodes_are_isolated_and_all_learn(self):
        env = ContextualBandit(); svc = make_service(env, min_episodes_per_update=1); errors = []
        def worker(k):
            try:
                rng = np.random.default_rng(k)
                for i in range(10): run_bandit_episode(svc, env, f"w{k}-{i}", rng)
            except Exception as exc:   # noqa: BLE001
                errors.append(exc)
        th = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
        [t.start() for t in th]; [t.join() for t in th]
        self.assertEqual(errors, [])
        self.assertEqual(svc.policy.consumed_episodes, 80); self.assertEqual(svc.policy.consumed_transitions, 80 * env.steps)
        self.assertEqual(svc.store.live_count(), 0)

    def test_batching_defers_the_update_until_enough_episodes(self):
        env = ContextualBandit(); svc = make_service(env, min_episodes_per_update=3)
        rng = np.random.default_rng(0)
        r1 = run_bandit_episode(svc, env, "b1", rng)[2]; r2 = run_bandit_episode(svc, env, "b2", rng)[2]
        self.assertEqual((r1.skip_reason, r2.skip_reason, svc.policy.training_step), ("insufficient_batch", "insufficient_batch", 0))
        r3 = run_bandit_episode(svc, env, "b3", rng)[2]
        self.assertTrue(r3.updated); self.assertEqual(r3.episodes, 3)

    def test_flush_consumes_a_partial_batch(self):
        env = ContextualBandit(); svc = make_service(env, min_episodes_per_update=5)
        run_bandit_episode(svc, env, "f1", np.random.default_rng(0))
        self.assertEqual(svc.policy.training_step, 0)
        res = svc.flush(); self.assertTrue(res.updated); self.assertEqual(svc.config.min_episodes_per_update, 5)

    def test_update_error_is_surfaced_and_data_is_kept(self):
        env = ContextualBandit(); svc = make_service(env)
        orig = svc.policy.update
        svc.policy.update = lambda batch: (_ for _ in ()).throw(RuntimeError("boom"))
        with self.assertLogs("src.agents.learning.external_learning", level="ERROR"):
            _, _, res = run_bandit_episode(svc, env, "x", np.random.default_rng(0))
        self.assertFalse(res.updated); self.assertTrue(res.skip_reason.startswith("error:RuntimeError"))
        st = svc.status(); self.assertEqual(st["counters"]["update_errors"], 1); self.assertEqual(st["last_error"]["message"], "boom")
        self.assertEqual(st["store"]["pending_episodes"], 1)                    # the batch went back to the queue
        svc.policy.update = orig
        self.assertTrue(svc.flush().updated)

    def test_overridden_actions_are_reported_and_not_credited(self):
        env = ContextualBandit(); svc = make_service(env, seed=7)
        svc.start_episode("g")
        for t in range(5):
            d = svc.decide("g", f"g{t}", env.obs(0))
            svc.executed("g", f"g{t}", (d.action.index + 1) % env.A)            # the environment overrode every decision (guardrail/fallback)
            svc.feedback("g", f"g{t}", 1.0)
        res = svc.end_episode("g")
        self.assertFalse(res.updated); self.assertEqual(res.skip_reason, "no_on_policy_transitions")
        svc.start_episode("g2")
        d = svc.decide("g2", "x", env.obs(0)); svc.executed("g2", "x", d.action.index); svc.feedback("g2", "x", 1.0)
        self.assertTrue(svc.end_episode("g2").updated)

    def test_decisions_carry_policy_identity_and_never_claim_to_be_fallback(self):
        env = ContextualBandit(); svc = make_service(env); svc.start_episode("p")
        d = svc.decide("p", "p0", env.obs(0))
        self.assertEqual(d.action.source, "learned"); self.assertEqual(d.policy["policy_id"], svc.policy.policy_id)
        self.assertEqual(d.policy["training_step"], 0); self.assertEqual(d.policy["mode"], "train")
        self.assertAlmostEqual(sum(d.action.probabilities), 1.0); self.assertEqual(d.action.probability, d.action.probabilities[d.action.index])

    def test_inference_output_changes_after_an_update(self):
        env = ContextualBandit(); svc = make_service(env)
        suite = [env.obs(c) for c in range(env.K)]
        before = [svc.policy.probabilities(o).copy() for o in suite]
        train(svc, env, 30)
        after = [svc.policy.probabilities(o) for o in suite]
        self.assertTrue(any(not np.allclose(a, b) for a, b in zip(before, after)))
        self.assertGreater(np.mean([after[c][env.best(c)] for c in range(env.K)]), np.mean([before[c][env.best(c)] for c in range(env.K)]) + 0.1)


class RewardReport(unittest.TestCase):
    def test_distribution_report_matches_hand_computation(self):
        env = ContextualBandit(steps=3); svc = make_service(env)
        plan = {"r0": ([1.0, 0.0, 0.0], 1.0), "r1": ([0.0, -0.5, 0.0], -1.0), "r2": ([0.0, 0.0, 0.0], 0.0)}
        for ep, (rs, term) in plan.items():
            svc.start_episode(ep)
            for t, r in enumerate(rs):
                svc.decide(ep, f"{ep}/{t}", env.obs(0), sequence=t); svc.feedback(ep, f"{ep}/{t}", r)
            svc.end_episode(ep, terminal_reward=term)
        rep = svc.reward_report()["train"]
        decisions = [1.0, 0.0, 1.0, 0.0, -0.5, -1.0, 0.0, 0.0, 0.0]
        arr = np.array(decisions)
        self.assertEqual(rep["decisions"], 9)
        self.assertEqual((rep["min"], rep["max"]), (-1.0, 1.0)); self.assertAlmostEqual(rep["mean"], arr.mean()); self.assertAlmostEqual(rep["std"], arr.std())
        self.assertAlmostEqual(rep["zero_fraction"], 5 / 9); self.assertEqual((rep["positive"], rep["negative"]), (2, 2))
        self.assertAlmostEqual(rep["terminal_reward_coverage"], 2 / 3)
        self.assertEqual(rep["episodes"], 3); self.assertEqual(rep["episode_totals"]["count"], 3)
        self.assertAlmostEqual(rep["episode_totals"]["mean"], np.mean([2.0, -1.5, 0.0]))


if __name__ == "__main__":
    unittest.main()
