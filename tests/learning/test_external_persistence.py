"""Checkpoint identity, atomic persistence and restart behaviour of ExternalLearningService (uses SLAI's CheckpointManager)."""
from __future__ import annotations

import glob, os, sys, tempfile, unittest

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np

from _support import ContextualBandit, checkpoint_stack, eval_bandit, make_service, run_bandit_episode
from src.agents.learning.contracts import CheckpointIntegrityError
from src.agents.learning.external_learning import ExternalLearningConfig, ExternalLearningService

OK, WHY = checkpoint_stack()
SEEDS = list(range(2000, 2012))


def train(svc, env, n, seed=0, prefix="t"):
    rng = np.random.default_rng(seed)
    for i in range(n):
        run_bandit_episode(svc, env, f"{prefix}{i}", rng)


def corrupt_latest_payload(root: str, version: str) -> None:
    path = glob.glob(os.path.join(root, version, "components", "*", "payload.npz"))[0]
    with open(path, "r+b") as f:
        f.write(b"XXXXXXXX")


@unittest.skipUnless(OK, f"checkpoint stack not importable: {WHY}")
class Persistence(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.dir = os.path.join(self.tmp.name, "ckpt"); self.env = ContextualBandit()

    def svc(self, **kw) -> ExternalLearningService:
        return make_service(self.env, checkpoint_dir=self.dir, **kw)

    def test_load_evaluate_train_change_save_restart_same_state(self):
        """The deterministic consistency test: load -> evaluate suite -> train on controlled transitions -> state changed ->
        re-evaluate (different) -> save -> reload in a new service (restart) -> identical state, hashes and outputs."""
        a = self.svc(seed=1)
        suite = [self.env.obs(c) for c in range(self.env.K)]
        out0 = np.array([a.policy.probabilities(o) for o in suite]); h0, s0 = a.policy.parameter_hash, a.policy.state_hash
        self.assertEqual(a.restore(), None)                                           # nothing to restore yet: fresh start, said so
        self.assertEqual(a.status()["restore"]["status"], "fresh_start")
        train(a, self.env, 40)
        out1 = np.array([a.policy.probabilities(o) for o in suite])
        self.assertNotEqual(a.policy.parameter_hash, h0); self.assertNotEqual(a.policy.state_hash, s0); self.assertFalse(np.allclose(out0, out1))
        ck = a.save_checkpoint("test")
        self.assertTrue(ck.saved); self.assertEqual(ck.state_hash, a.policy.state_hash); self.assertEqual(ck.training_step, a.policy.training_step)
        b = self.svc(seed=99)                                                         # restart: a fresh process would do exactly this
        self.assertEqual(b.status()["restore"]["status"], "restored")
        self.assertEqual((b.policy.parameter_hash, b.policy.state_hash), (a.policy.parameter_hash, a.policy.state_hash))
        self.assertEqual(b.policy.policy_id, a.policy.policy_id)
        self.assertEqual((b.policy.training_step, b.policy.consumed_transitions, b.policy.consumed_episodes),
                         (a.policy.training_step, a.policy.consumed_transitions, a.policy.consumed_episodes))
        np.testing.assert_array_equal(np.array([b.policy.probabilities(o) for o in suite]), out1)
        self.assertEqual(b.metadata().checkpoint_id, ck.checkpoint_id); self.assertEqual(b.metadata().restored_from, ck.checkpoint_id)
        self.assertEqual(eval_bandit(a, self.env, SEEDS, "a"), eval_bandit(b, self.env, SEEDS, "b"))

    def test_restored_learner_keeps_learning_and_records_lineage(self):
        a = self.svc(seed=1); train(a, self.env, 20); c1 = a.save_checkpoint()
        b = self.svc(); step = b.policy.training_step; train(b, self.env, 20, seed=5, prefix="u")
        self.assertGreater(b.policy.training_step, step)                              # continued from the restored step, not from zero
        c2 = b.save_checkpoint()
        self.assertEqual(c2.parent_checkpoint_id, c1.checkpoint_id); self.assertNotEqual(c2.checkpoint_id, c1.checkpoint_id)
        self.assertNotEqual(c2.parameter_hash, c1.parameter_hash)
        self.assertEqual(len(glob.glob(os.path.join(self.dir, "s*"))), 2)

    def test_unchanged_state_writes_no_new_checkpoint(self):
        a = self.svc(seed=1); train(a, self.env, 5); c1 = a.save_checkpoint()
        c2 = a.save_checkpoint()
        self.assertFalse(c2.saved); self.assertEqual(c2.checkpoint_id, c1.checkpoint_id)
        self.assertEqual(len(glob.glob(os.path.join(self.dir, "s*"))), 1)

    def test_newest_corrupt_checkpoint_is_skipped_visibly(self):
        a = self.svc(seed=1); train(a, self.env, 10); good = a.save_checkpoint(); train(a, self.env, 10, prefix="v"); bad = a.save_checkpoint()
        corrupt_latest_payload(self.dir, bad.version)
        b = self.svc()
        rep = b.status()["restore"]
        self.assertEqual(rep["status"], "restored"); self.assertEqual(rep["checkpoint_id"], good.checkpoint_id)
        self.assertEqual([s["version"] for s in rep["skipped"]], [bad.version])
        self.assertEqual(b.policy.parameter_hash, good.parameter_hash)                # the older, valid state - not the corrupt one, not fresh

    def test_explicit_corrupt_version_fails_loudly(self):
        a = self.svc(seed=1); train(a, self.env, 10); ck = a.save_checkpoint()
        corrupt_latest_payload(self.dir, ck.version)
        with self.assertRaises(Exception): self.svc(restore_on_start=False).restore(ck.version)

    def test_all_checkpoints_corrupt_refuses_to_start_instead_of_resetting(self):
        a = self.svc(seed=1); train(a, self.env, 10); ck = a.save_checkpoint()
        corrupt_latest_payload(self.dir, ck.version)
        with self.assertRaises(CheckpointIntegrityError): self.svc()

    def test_content_that_does_not_match_recorded_hash_is_refused(self):
        a = self.svc(seed=1); train(a, self.env, 10); a.save_checkpoint()
        cm = a._checkpoint_manager()
        arrs = a.policy.export_arrays(); arrs["W"] = arrs["W"] + 1.0                   # a valid file whose arrays are not what its metadata says
        from checkpointing.checkpoint_types import CheckpointLineage, CheckpointProvenance
        md = {"schema_version": "slai.external-learning.v1", "algorithm": "residual-softmax-reinforce", "name": "toy",
              "state_hash": a.policy.state_hash, "parameter_hash": a.policy.parameter_hash, "hyperparameter_hash": "x", "training_step": 99,
              "consumed_transitions": 0, "consumed_episodes": 0, "action_dim": a.policy.action_dim, "feature_dim": a.policy.feature_dim}
        cm.save_components({"model": arrs}, version="s99999999-forged", codec_ids={"model": "numpy"}, step=99, metadata=md)
        with self.assertRaises(CheckpointIntegrityError): self.svc()                  # newest "valid file" has the wrong content -> loud

    def test_incompatible_dimensions_are_refused(self):
        a = self.svc(seed=1); train(a, self.env, 5); a.save_checkpoint()
        other = ContextualBandit(contexts=6, actions=4)
        with self.assertRaises(CheckpointIntegrityError): make_service(other, checkpoint_dir=self.dir)

    def test_restore_on_start_false_leaves_a_fresh_learner(self):
        a = self.svc(seed=1); train(a, self.env, 5); a.save_checkpoint()
        b = self.svc(restore_on_start=False)
        self.assertEqual(b.policy.training_step, 0); self.assertIsNone(b.metadata().checkpoint_id)

    def test_autosave_writes_checkpoints_as_the_policy_changes(self):
        a = self.svc(seed=1, autosave_every_updates=5); train(a, self.env, 12)
        self.assertGreaterEqual(len(glob.glob(os.path.join(self.dir, "s*"))), 2)
        self.assertEqual(a.status()["counters"]["checkpoints_saved"], len(glob.glob(os.path.join(self.dir, "s*"))))

    def test_evaluation_after_restart_does_not_alter_the_checkpoint_identity(self):
        a = self.svc(seed=1); train(a, self.env, 15); ck = a.save_checkpoint()
        b = self.svc(mode="evaluate"); eval_bandit(b, self.env, SEEDS, "e")
        self.assertEqual(b.policy.state_hash, ck.state_hash); self.assertFalse(b.save_checkpoint().saved)

    def test_saving_without_a_directory_fails_visibly(self):
        svc = make_service(self.env)
        with self.assertRaises(CheckpointIntegrityError): svc.save_checkpoint()


if __name__ == "__main__":
    unittest.main()
