"""ResidualSoftmaxPolicy: masking, prior, identity/hash, a measurable and correct parameter update, safety bounds.  numpy only."""
from __future__ import annotations

import copy, os, sys, unittest

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np

from src.agents.learning.contracts import CheckpointIntegrityError, ContractError, EpisodeRecord, Observation, Transition
from src.agents.learning.softmax_policy import PolicyHyperparameters, ResidualSoftmaxPolicy

D, A = 3, 4


def ob(x=(1.0, 0.0, 0.5), mask=None, prior=None) -> Observation:
    return Observation.make(list(x), mask, prior, feature_dim=D, action_dim=A)


def episode(items, ep="e", terminal=None) -> EpisodeRecord:
    """items: [(obs, action, reward)] -> EpisodeRecord"""
    ts = tuple(Transition(episode_id=ep, decision_id=f"{ep}/{i}", sequence=i, observation=o, action=a, reward=r, next_observation=None,
                          terminated=i == len(items) - 1, truncated=False, policy_version="v", timestamp=1.0)
               for i, (o, a, r) in enumerate(items))
    return EpisodeRecord(ep, ts, True, False, sum(t.reward for t in ts), 0.0, 1.0, "v")


class PolicyTests(unittest.TestCase):
    def test_zero_residual_is_the_prior(self):
        p = ResidualSoftmaxPolicy(D, A)
        prior = np.array([2.0, 0.0, -1.0, 0.5])
        z = np.exp(prior - prior.max()); z /= z.sum()
        np.testing.assert_allclose(p.probabilities(ob(prior=prior)), z)
        np.testing.assert_allclose(p.probabilities(ob()), np.full(A, 0.25))

    def test_illegal_actions_are_never_chosen(self):
        p = ResidualSoftmaxPolicy(D, A, seed=3)
        o = ob(mask=[False, True, False, True], prior=[9.0, 0.0, 9.0, 0.0])
        picks = {p.act(o, greedy=False).index for _ in range(500)} | {p.act(o, greedy=True).index}
        self.assertEqual(picks, {1, 3})
        self.assertEqual(p.probabilities(o)[0], 0.0)

    def test_greedy_is_deterministic_and_sampling_is_seeded(self):
        a = ResidualSoftmaxPolicy(D, A, seed=5); b = ResidualSoftmaxPolicy(D, A, seed=5)
        o = ob()
        self.assertEqual([a.act(o, greedy=False).index for _ in range(30)], [b.act(o, greedy=False).index for _ in range(30)])
        self.assertEqual({a.act(o, greedy=True).index for _ in range(10)}, {0})        # tie -> lowest index

    def test_wrong_shapes_rejected(self):
        p = ResidualSoftmaxPolicy(D, A)
        with self.assertRaises(ContractError): p.probabilities(Observation.make([1, 2], feature_dim=2))
        with self.assertRaises(ContractError): p.probabilities(Observation.make([1, 2, 3], prior_logits=[0, 0]))

    # ------------------------------------------------------------------ identity
    def test_identity_is_derived_from_parameters(self):
        a = ResidualSoftmaxPolicy(D, A, name="q"); b = ResidualSoftmaxPolicy(D, A, name="q", seed=99)
        self.assertEqual(a.parameter_hash, b.parameter_hash)                 # same parameters -> same hash, whatever the rng seed
        self.assertEqual(a.policy_id, b.policy_id)
        b.W[0, 0] = 1e-9; b._hash_cache = None
        self.assertNotEqual(a.parameter_hash, b.parameter_hash); self.assertNotEqual(a.policy_id, b.policy_id)

    def test_negative_zero_and_layout_do_not_change_hash(self):
        a = ResidualSoftmaxPolicy(D, A); b = ResidualSoftmaxPolicy(D, A)
        b.W = np.asfortranarray(-np.zeros((A, D))); b._hash_cache = None
        self.assertEqual(a.parameter_hash, b.parameter_hash)

    def test_different_shapes_never_share_a_hash(self):
        self.assertNotEqual(ResidualSoftmaxPolicy(3, 4).parameter_hash, ResidualSoftmaxPolicy(4, 3).parameter_hash)

    def test_hyperparameter_digest_is_stable_and_sensitive(self):
        h = PolicyHyperparameters(); self.assertEqual(h.digest(), PolicyHyperparameters().digest())
        self.assertNotEqual(h.digest(), PolicyHyperparameters(learning_rate=0.1).digest())
        for bad in ({"learning_rate": 0}, {"gamma": 1.5}, {"temperature": -1}, {"baseline_momentum": 1.0}, {"learning_rate": float("nan")}):
            with self.assertRaises(ValueError): PolicyHyperparameters(**bad)

    # ------------------------------------------------------------------ learning
    def test_update_raises_probability_of_a_rewarded_action_and_changes_everything_it_should(self):
        p = ResidualSoftmaxPolicy(D, A, PolicyHyperparameters(learning_rate=0.3))
        o = ob(); before = p.probabilities(o)[2]
        h0, s0, id0 = p.parameter_hash, p.state_hash, p.policy_id
        res = p.update([episode([(o, 2, 1.0)] * 5)])
        self.assertTrue(res.updated); self.assertIsNone(res.skip_reason)
        self.assertGreater(p.probabilities(o)[2], before)                                  # inference sees the update
        self.assertNotEqual(p.parameter_hash, h0); self.assertNotEqual(p.state_hash, s0); self.assertNotEqual(p.policy_id, id0)
        self.assertEqual((res.step_before, res.step_after, p.training_step), (0, 1, 1))
        self.assertEqual((res.parameter_hash_before, res.parameter_hash_after), (h0, p.parameter_hash))
        self.assertTrue(res.inference_uses_update)
        self.assertEqual((res.episodes, res.transitions, res.credited_transitions), (1, 5, 5))
        for f in ("loss", "grad_norm", "grad_norm_clipped", "param_norm_before", "param_norm_after", "param_delta_norm", "learning_rate",
                  "mean_return", "baseline", "entropy"):
            self.assertIsNotNone(getattr(res, f), f)
        self.assertEqual(res.param_norm_before, 0.0); self.assertGreater(res.param_norm_after, 0.0); self.assertGreater(res.param_delta_norm, 0.0)
        self.assertEqual((p.consumed_episodes, p.consumed_transitions), (1, 5))

    def test_baseline_is_subtracted_so_expected_returns_produce_no_push(self):
        p = ResidualSoftmaxPolicy(D, A, PolicyHyperparameters(learning_rate=0.3, baseline_momentum=0.5)); o = ob()
        first = p.update([episode([(o, 2, 1.0)] * 4, ep="a")])
        self.assertGreater(first.grad_norm, 0)
        for i in range(30): last = p.update([episode([(o, 2, 1.0)] * 4, ep=f"b{i}")])  # same return every time -> baseline converges to it
        self.assertLess(last.grad_norm, first.grad_norm * 1e-3)
        self.assertAlmostEqual(last.baseline, 2.5, places=6)   # mean of the returns-to-go [4, 3, 2, 1]

    def test_negative_return_lowers_the_probability_of_the_chosen_action(self):
        p = ResidualSoftmaxPolicy(D, A, PolicyHyperparameters(learning_rate=0.3)); o = ob(); before = p.probabilities(o)[1]
        p.update([episode([(o, 1, -1.0)] * 4)])
        self.assertLess(p.probabilities(o)[1], before)

    def test_zero_advantage_changes_nothing_and_keeps_identity(self):
        p = ResidualSoftmaxPolicy(D, A); id0, h0, s0 = p.policy_id, p.parameter_hash, p.state_hash
        res = p.update([episode([(ob(), 1, 0.0)] * 3)])
        self.assertFalse(res.updated); self.assertEqual(res.skip_reason, "zero_gradient")
        self.assertEqual((p.policy_id, p.parameter_hash, p.training_step), (id0, h0, 0))   # a new identity only on a real change
        self.assertNotEqual(p.state_hash, s0)      # ...but the learner's own statistics (baseline sample count) did advance
        self.assertEqual(p.consumed_transitions, 3)

    def test_overridden_decisions_are_not_credited(self):
        p = ResidualSoftmaxPolicy(D, A); o = ob()
        ep = episode([(o, 1, 1.0)] * 3)
        ep = EpisodeRecord(ep.episode_id, tuple(Transition(**{**t.__dict__, "executed_action": 2}) for t in ep.transitions), True, False, 3.0, 0, 1, "v")
        h0 = p.parameter_hash
        res = p.update([ep])
        self.assertFalse(res.updated); self.assertEqual(res.skip_reason, "no_on_policy_transitions"); self.assertEqual(p.parameter_hash, h0)
        self.assertEqual(res.credited_transitions, 0)

    def test_invalid_transitions_are_counted_and_skipped(self):
        p = ResidualSoftmaxPolicy(D, A); good = ob()
        masked = ob(mask=[True, False, True, True])
        res = p.update([episode([(masked, 1, 1.0), (good, 0, 1.0)])])                     # action 1 was illegal in `masked`
        self.assertTrue(res.updated); self.assertEqual(res.invalid_values, 1); self.assertEqual(res.credited_transitions, 1)
        res = p.update([episode([(masked, 1, 1.0)], ep="x")])
        self.assertEqual(res.skip_reason, "no_valid_transitions")
        self.assertEqual(p.update([]).skip_reason, "empty_batch")

    def test_gradient_and_parameter_bounds(self):
        p = ResidualSoftmaxPolicy(D, A, PolicyHyperparameters(learning_rate=10.0, max_grad_norm=0.1, max_param_norm=0.2, advantage_clip=100))
        res = p.update([episode([(ob((5, 5, 5)), 0, 50.0)] * 20)])
        self.assertLessEqual(res.grad_norm_clipped, 0.1 + 1e-12); self.assertGreater(res.grad_norm, 0.1)
        self.assertLessEqual(p.param_norm(), 0.2 + 1e-9)

    def test_learned_residual_never_overrides_masks_or_prior_scale(self):
        p = ResidualSoftmaxPolicy(D, A, PolicyHyperparameters(learning_rate=1.0, max_param_norm=0.5))
        o = ob(mask=[True, True, True, False], prior=[0, 0, 5.0, 0])
        for _ in range(50): p.update([episode([(o, 0, 1.0)] * 5)])
        self.assertEqual(p.probabilities(o)[3], 0.0)
        self.assertEqual(p.act(o, greedy=True).index, 2)                                  # a bounded residual cannot beat a strong prior

    def test_non_finite_batch_is_refused_and_parameters_are_untouched(self):
        p = ResidualSoftmaxPolicy(D, A, PolicyHyperparameters(learning_rate=0.1)); o = ob()
        p.update([episode([(o, 1, 1.0)] * 2)]); snap = copy.deepcopy(p.export_arrays()); h = p.parameter_hash
        bad = episode([(o, 1, float("nan"))])
        res = p.update([bad])
        self.assertFalse(res.updated); self.assertEqual(p.parameter_hash, h)
        np.testing.assert_array_equal(p.W, snap["W"])

    def test_entropy_bonus_and_weight_decay_run(self):
        p = ResidualSoftmaxPolicy(D, A, PolicyHyperparameters(learning_rate=0.1, entropy_coef=0.05, weight_decay=0.01))
        self.assertTrue(p.update([episode([(ob(), 0, 1.0)] * 4)]).updated)

    # ------------------------------------------------------------------ persistence primitives
    def test_load_state_round_trip_and_validation(self):
        a = ResidualSoftmaxPolicy(D, A, PolicyHyperparameters(learning_rate=0.3)); a.update([episode([(ob(), 2, 1.0)] * 3)])
        b = ResidualSoftmaxPolicy(D, A)
        b.load_state(a.export_arrays(), a.counters(), expect_parameter_hash=a.parameter_hash, expect_state_hash=a.state_hash)
        self.assertEqual((b.parameter_hash, b.state_hash, b.training_step, b.consumed_transitions), (a.parameter_hash, a.state_hash, 1, 3))
        np.testing.assert_array_equal(b.probabilities(ob()), a.probabilities(ob()))

    def test_load_state_failure_leaves_live_state_untouched(self):
        a = ResidualSoftmaxPolicy(D, A, PolicyHyperparameters(learning_rate=0.3)); a.update([episode([(ob(), 2, 1.0)] * 3)])
        h, s = a.parameter_hash, a.state_hash
        arrs = a.export_arrays()
        for bad in ({**arrs, "W": np.zeros((A + 1, D))}, {**arrs, "W": arrs["W"] * np.nan}, {"W": arrs["W"]}):
            with self.assertRaises(CheckpointIntegrityError): a.load_state(bad, a.counters())
        with self.assertRaises(CheckpointIntegrityError): a.load_state(arrs, a.counters(), expect_parameter_hash="0" * 64)
        self.assertEqual((a.parameter_hash, a.state_hash), (h, s))


if __name__ == "__main__":
    unittest.main()
