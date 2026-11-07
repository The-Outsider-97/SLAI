"""Tests for the advisory QualityPredictor (runtime module + QualityAgent integration).

The runtime contract is tested with a tiny deterministic checkpoint built in memory, so the tests need
neither numpy nor a trained model. The integration tests run the real QualityAgent (isolated in-memory
QualityMemory) exactly as ``train_qp.py`` does.
"""
from __future__ import annotations

import copy
import json
import math
import random
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.quality.modules.quality_predictor import (  # noqa: E402
    DEFECTS, FEATURE_SCHEMA_ID, FEATURES, FINAL_RECORD_TYPE, TARGETS, QualityPredictionModel, QualityPredictor,
    extract_features, labels_from_decision,
)

NF, NT = len(FEATURES), len(TARGETS)


def mini_model(*, promotion: str = "experimental", bias=None) -> dict:
    """Constant-output checkpoint: W1 = W2 = 0, so probabilities are sigmoid(bias) for every input."""
    bias = bias if bias is not None else [0.0] * NT
    member = {"W1": [[0.0, 0.0] for _ in range(NF)], "b1": [0.0, 0.0], "W2": [[0.0] * NT, [0.0] * NT], "b2": list(bias)}
    return {
        "schema": 1, "version": "test-1", "promotion": promotion, "features": list(FEATURES), "targets": list(TARGETS),
        "scaler": {"mean": [0.0] * NF, "scale": [1.0] * NF},
        "domain": {"min": [-1e9] * NF, "max": [1e9] * NF},
        "members": [member, copy.deepcopy(member)], "calibration": [[1.0, 0.0] for _ in range(NT)],
        "selection": {"low_risk_below": 0.1, "high_risk_at": 0.6, "focus_threshold": 0.4, "max_uncertainty": 1.0, "max_ood_fraction": 1.0},
        "provenance": {"dataset_fingerprint": "test", "feature_schema": FEATURE_SCHEMA_ID, "synthetic_only": True,
                       "target_sources": {"synthetic_agent_run": 1}, "untrained_targets": []},
    }


def records(n: int = 30, seed: int = 1) -> list:
    rng = random.Random(seed)
    return [{"id": f"r{i:04d}", "text": "sample text %d" % rng.randint(0, 999), "value": round(rng.random(), 4),
             "label": rng.choice(["a", "b"])} for i in range(n)]


class FeatureTests(unittest.TestCase):
    def test_vector_is_complete_finite_and_deterministic(self):
        first = extract_features(records(), source_id="s", use_case="training", now=1_700_000_000.0)
        second = extract_features(records(), source_id="s", use_case="training", now=1_700_000_000.0)
        self.assertEqual(set(first), set(FEATURES))
        self.assertTrue(all(isinstance(v, float) and math.isfinite(v) for v in first.values()))
        self.assertEqual(first, second)

    def test_defects_move_the_matching_features(self):
        clean = extract_features(records(), source_id="s", use_case="general")
        dirty_records = records() + records()[:10]
        for r in dirty_records[:8]:
            r["value"] = None
        dirty = extract_features(dirty_records, source_id="s", use_case="general")
        self.assertNotEqual(clean, dirty)

    def test_empty_and_malformed_input_does_not_raise(self):
        self.assertEqual(set(extract_features([], source_id="s", use_case="general")), set(FEATURES))
        self.assertEqual(set(extract_features([{}, None, 5, "x"], source_id=None, use_case="nonsense")), set(FEATURES))

    def test_sampling_is_bounded(self):
        big = records(5000)
        started = time.perf_counter()
        extract_features(big, source_id="s", use_case="general", max_sample_records=128)
        self.assertLess(time.perf_counter() - started, 2.0)


class ModelTests(unittest.TestCase):
    def test_constant_model_scores_and_assesses(self):
        model = QualityPredictionModel.from_model(mini_model(bias=[0.0] * NT))
        feats = extract_features(records(), source_id="s", use_case="general")
        scored = model.score(feats)
        self.assertEqual(set(scored["probabilities"]), set(TARGETS))
        self.assertAlmostEqual(scored["probabilities"]["block"], 0.5, places=6)
        self.assertAlmostEqual(scored["uncertainty"], 0.0, places=9)
        assessed = model.assess(feats)
        self.assertIn(assessed["risk_tier"], ("low", "elevated", "high"))
        self.assertTrue(assessed["advice"]["mandatory_checks_unchanged"])

    def test_tiers_follow_thresholds(self):
        low_bias = [-6.0] * NT
        high_bias = [6.0] * NT
        feats = extract_features(records(), source_id="s", use_case="general")
        self.assertEqual(QualityPredictionModel.from_model(mini_model(bias=low_bias)).assess(feats)["risk_tier"], "low")
        self.assertEqual(QualityPredictionModel.from_model(mini_model(bias=high_bias)).assess(feats)["risk_tier"], "high")

    def test_optional_checks_are_only_deprioritised_when_approved_reliable_and_low(self):
        feats = extract_features(records(), source_id="s", use_case="general")
        low = [-6.0] * NT
        experimental = QualityPredictionModel.from_model(mini_model(bias=low)).assess(feats)
        approved = QualityPredictionModel.from_model(mini_model(promotion="approved", bias=low)).assess(feats)
        self.assertFalse(experimental["advice"]["deprioritize_optional_checks"])
        self.assertTrue(approved["advice"]["deprioritize_optional_checks"])

    def test_invalid_checkpoints_are_rejected(self):
        for mutate in (lambda m: m.update(features=list(reversed(FEATURES))),
                       lambda m: m.update(targets=TARGETS[:-1]),
                       lambda m: m["scaler"].update(scale=[0.0] * NF),
                       lambda m: m["members"][0].update(b2=[0.0]),
                       lambda m: m.update(promotion="maybe"),
                       lambda m: m["calibration"].pop()):
            bad = mini_model()
            mutate(bad)
            with self.assertRaises(Exception):
                QualityPredictionModel.from_model(bad)

    def test_file_roundtrip_and_experimental_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text(json.dumps(mini_model()), encoding="utf-8")
            self.assertEqual(QualityPredictionModel(path, allow_experimental=True).promotion, "experimental")
            with self.assertRaises(Exception):
                QualityPredictionModel(path, allow_experimental=False)


class LabelTests(unittest.TestCase):
    def test_labels_follow_verdict_and_checks_that_ran(self):
        decision = {"verdict": "block", "subsystem_findings": [
            {"domain": "structural", "check_name": "required_completeness", "verdict": "block"},
            {"domain": "statistical", "check_name": "duplicates", "verdict": "pass"}]}
        labels = labels_from_decision(decision)
        self.assertEqual(labels["block"], 1.0)
        self.assertEqual(labels["attention"], 1.0)
        self.assertEqual(labels["required"], 1.0)
        self.assertEqual(labels["duplicates"], 0.0)
        self.assertIsNone(labels["drift"])  # no drift check ran: unknown, not "clean"

    def test_pass_has_no_attention(self):
        labels = labels_from_decision({"verdict": "pass", "subsystem_findings": []})
        self.assertEqual((labels["block"], labels["attention"]), (0.0, 0.0))


class PredictorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "manifest.json"
        self.path.write_text(json.dumps(mini_model(bias=[0.5] * NT)), encoding="utf-8")

    def test_disabled_by_default_and_silent(self):
        predictor = QualityPredictor(config={})
        self.assertFalse(predictor.status()["enabled"])
        self.assertEqual(predictor.predict_batch(records(), source_id="s", use_case="general"), {})

    def test_collect_mode_records_features_without_a_model(self):
        predictor = QualityPredictor(config={"enabled": True, "mode": "collect", "model_path": str(Path(self.tmp.name) / "none.json")})
        prediction = predictor.predict_batch(records(), source_id="s", use_case="training")
        self.assertEqual(prediction["mode"], "collect")
        self.assertEqual(set(prediction["features"]), set(FEATURES))
        self.assertNotIn("probabilities", prediction)

    def test_shadow_mode_predicts_and_observes(self):
        predictor = QualityPredictor(config={"enabled": True, "mode": "shadow", "model_path": str(self.path)})
        prediction = predictor.predict_batch(records(), source_id="s", use_case="general")
        self.assertIn("probabilities", prediction)
        self.assertEqual(prediction["authority"], "advisory_only")
        observed = predictor.observe(prediction, {"verdict": "pass", "subsystem_findings": []})
        self.assertIn("observed", observed)

    def test_advisory_mode_requires_an_approved_checkpoint(self):
        predictor = QualityPredictor(config={"enabled": True, "mode": "advisory", "model_path": str(self.path)})
        self.assertEqual(predictor.predict_batch(records(), source_id="s", use_case="general").get("probabilities"), None)

    def test_invalid_config_raises_and_failures_are_open(self):
        with self.assertRaises(Exception):
            QualityPredictor(config={"enabled": True, "mode": "nonsense"})
        predictor = QualityPredictor(config={"enabled": True, "mode": "shadow", "model_path": str(self.path)})
        self.assertEqual(predictor.predict_batch(object(), source_id="s", use_case="general"), {})


class AgentIntegrationTests(unittest.TestCase):
    """Real QualityAgent: the predictor attaches advice and NEVER changes a verdict."""

    @classmethod
    def setUpClass(cls):
        try:
            import train_qp  # noqa: F401  (AgentRig lives in the trainer so that training and serving share one path)
        except Exception as exc:  # pragma: no cover - environment dependent
            raise unittest.SkipTest("train_qp/AgentRig unavailable: %s" % exc)

    def batch(self, n: int = 60, defective: bool = False):
        import train_qp as T
        rng = random.Random(3)
        arch = T.ARCHETYPES["documents"]
        recs = [arch["make"](rng, i, "int", T.text_pool(T.DEFAULT_CORPUS), time.time() - 3600) for i in range(n)]
        if defective:
            recs = T.inject("duplicates", rng, recs, arch, 0.4)
        return dict(records=recs, dataset_id="int", source_id="int", batch_id="int-b0", schema=arch["schema"], label_field="label",
                    use_case="general", provenance=T.make_provenance(rng, "int", {"source_type": "api", "optional_completeness": 1.0, "age_days": 1.0}, time.time(), False))

    def test_collect_mode_attaches_features_and_outcome(self):
        import train_qp as T
        rig = T.AgentRig()
        decision = rig.evaluate(self.batch())
        prediction = decision["quality_prediction"]
        self.assertEqual(prediction["feature_schema"], FEATURE_SCHEMA_ID)
        self.assertEqual(set(prediction["features"]), set(FEATURES))
        self.assertIn("observed", prediction)
        row = T.row_from_decision(decision, source_id="int", batch_id="int-b0", order=0, use_case="general", target_source="agent_run")
        self.assertEqual(len(row["x"]), NF)

    def test_prediction_never_changes_the_verdict(self):
        import train_qp as T
        rig = T.AgentRig()
        with_predictor = rig.evaluate(self.batch(defective=True))
        rig.agent.quality_predictor = None
        # A second isolated rig with the predictor removed must reach the same verdict.
        rig2 = T.AgentRig()
        rig2.agent.quality_predictor = None
        without = rig2.evaluate(self.batch(defective=True))
        self.assertEqual(with_predictor["verdict"], without["verdict"])
        self.assertNotIn("quality_prediction", without)

    def test_final_decisions_are_persisted_for_retraining(self):
        import train_qp as T
        rig = T.AgentRig()
        rig.evaluate(self.batch())
        state = rig.memory.export_state()
        finals = [s for snaps in (state.get("snapshots_by_source") or {}).values() for s in snaps
                  if (s.get("context") or {}).get("record_type") == FINAL_RECORD_TYPE]
        self.assertTrue(finals)


if __name__ == "__main__":
    unittest.main()
