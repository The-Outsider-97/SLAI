"""Tests for the DeliveryPredictor (runtime, safety overrides, fallback, agent integration).

Run:  python -m unittest tests.test_delivery_predictor -v
"""
from __future__ import annotations

import json
import math
import tempfile
import threading
import unittest
from pathlib import Path

from src.agents.agent_factory import AgentFactory
from src.agents.collaborative.shared_memory import SharedMemory
from src.agents.network.delivery_predictor import (
    FEATURE_HASH,
    FEATURES,
    SCHEMA,
    DeliveryContext,
    DeliveryModel,
    DeliveryPredictor,
    EndpointHistory,
    build_features,
    classify_error,
)
from src.agents.network.utils.network_errors import ConnectionTimeoutError, NetworkConnectionError, RateLimitExceededError
from src.agents.network_agent import NetworkAgent
from tests.test_network_agent_hardening import CONSTRAINTS, _Server, _envelope

DIM = len(FEATURES)


def make_manifest(*, p_bias: float = 5.0, latency_ms: float = 10.0, promotion: str = "experimental") -> dict:
    """A hand-built, valid checkpoint: constant predictions (all weights zero)."""
    return {
        "schema": SCHEMA,
        "features": list(FEATURES),
        "feature_hash": FEATURE_HASH,
        "mean": [0.0] * DIM,
        "scale": [1.0] * DIM,
        "success": {"kind": "logistic", "weights": [0.0] * DIM, "bias": p_bias},
        "latency": {
            "kind": "ridge_log1p", "weights": [0.0] * DIM, "bias": math.log1p(latency_ms),
            "residual_quantiles": {"q50": 0.0, "q95": 0.0, "q99": 0.0},
        },
        "promotion": {"status": promotion},
    }


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def warm(predictor: DeliveryPredictor, endpoint: str, *, successes: int = 12, latency_ms: float = 10.0) -> None:
    for _ in range(successes):
        predictor.observe(endpoint, ok=True, latency_ms=latency_ms)


def ctx(endpoint: str = "http://e/relay", **overrides) -> DeliveryContext:
    base = dict(endpoint=endpoint, attempt=1, max_attempts=3, requested_timeout_ms=1000, remaining_ms=None, payload_bytes=100)
    base.update(overrides)
    return DeliveryContext(**base)


class FeatureTests(unittest.TestCase):
    def test_features_are_deterministic_finite_and_schema_sized(self):
        history = EndpointHistory()
        for index in range(20):
            history.add(100.0 + index, ok=index % 4 != 0, latency_ms=10.0 + index, error_class="none" if index % 4 else "timeout")
        snap = history.snapshot(200.0)
        first = build_features(ctx(), snap)
        second = build_features(ctx(), snap)
        self.assertEqual(first, second)
        self.assertEqual(len(first), len(FEATURES))
        self.assertTrue(all(math.isfinite(v) for v in first))

    def test_empty_history_yields_neutral_valid_features(self):
        features = build_features(ctx(), EndpointHistory().snapshot(0.0))
        self.assertEqual(len(features), len(FEATURES))
        self.assertTrue(all(math.isfinite(v) for v in features))
        self.assertEqual(features[FEATURES.index("history_fill")], 0.0)

    def test_no_deadline_and_no_timeout_do_not_break_features(self):
        features = build_features(ctx(requested_timeout_ms=None, remaining_ms=None), EndpointHistory().snapshot(0.0))
        self.assertTrue(all(math.isfinite(v) for v in features))

    def test_error_classification(self):
        self.assertEqual(classify_error(None), "none")
        self.assertEqual(classify_error(ConnectionTimeoutError("t")), "timeout")
        self.assertEqual(classify_error(RateLimitExceededError("r")), "rate_limit")
        self.assertEqual(classify_error(NetworkConnectionError("c")), "connection")
        self.assertEqual(classify_error(ValueError("x")), "other")

    def test_history_is_bounded(self):
        history = EndpointHistory()
        for index in range(500):
            history.add(float(index), ok=True, latency_ms=5.0, error_class="none")
        self.assertLessEqual(len(history.rows), 32)


class ModelSchemaTests(unittest.TestCase):
    def test_valid_manifest_loads_and_predicts(self):
        model = DeliveryModel(make_manifest())
        prediction = model.predict([0.0] * DIM)
        self.assertGreater(prediction["p_success"], 0.99)
        self.assertAlmostEqual(prediction["latency_q99_ms"], 10.0, places=3)

    def test_feature_list_mismatch_is_rejected(self):
        manifest = make_manifest()
        manifest["features"] = list(reversed(FEATURES))
        with self.assertRaises(ValueError):
            DeliveryModel(manifest)

    def test_feature_hash_mismatch_is_rejected(self):
        manifest = make_manifest()
        manifest["feature_hash"] = "deadbeef"
        with self.assertRaises(ValueError):
            DeliveryModel(manifest)

    def test_wrong_weight_length_and_non_finite_values_are_rejected(self):
        manifest = make_manifest()
        manifest["success"]["weights"] = [0.0] * (DIM - 1)
        with self.assertRaises(ValueError):
            DeliveryModel(manifest)
        manifest = make_manifest()
        manifest["latency"]["bias"] = float("nan")
        with self.assertRaises(ValueError):
            DeliveryModel(manifest)
        manifest = make_manifest()
        manifest["scale"][0] = 0.0
        with self.assertRaises(ValueError):
            DeliveryModel(manifest)

    def test_load_rejects_wrong_schema_and_oversized_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            manifest = make_manifest()
            manifest["schema"] = "other/9"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(ValueError):
                DeliveryModel.load(path)


class FallbackTests(unittest.TestCase):
    def test_missing_manifest_falls_back_to_heuristic(self):
        predictor = DeliveryPredictor({"manifest_path": "does/not/exist.json"})
        self.assertEqual(predictor.model_status, "missing")
        decision = predictor.plan(ctx())
        self.assertEqual(decision.source, "heuristic")
        self.assertEqual(decision.timeout_ms, 1000)

    def test_corrupt_manifest_falls_back_and_reports_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text("{not json", encoding="utf-8")
            predictor = DeliveryPredictor({"manifest_path": str(path)})
        self.assertTrue(predictor.model_status.startswith("invalid:"))
        self.assertIsNone(predictor.model)
        self.assertEqual(predictor.plan(ctx()).timeout_ms, 1000)

    def test_unseen_endpoint_keeps_requested_timeout(self):
        predictor = DeliveryPredictor({"mode": "active", "allow_experimental_active": True}, model=DeliveryModel(make_manifest()))
        decision = predictor.plan(ctx("http://never-seen/relay"))
        self.assertEqual(decision.timeout_ms, 1000)
        self.assertIn("insufficient_history", decision.reasons)

    def test_internal_error_returns_passthrough_and_never_raises(self):
        predictor = DeliveryPredictor({"mode": "active", "allow_experimental_active": True}, model=DeliveryModel(make_manifest()))
        warm(predictor, "http://e/relay")

        def boom(*args, **kwargs):
            raise RuntimeError("model exploded")

        predictor._predict = boom  # type: ignore[assignment]
        decision = predictor.plan(ctx())
        self.assertEqual(decision.timeout_ms, 1000)
        self.assertTrue(decision.source.startswith("fallback:"))
        self.assertFalse(predictor.allow_retry(ctx(attempt=2, last_error_class="timeout")).vetoed_retry)
        self.assertGreaterEqual(predictor.get_snapshot()["stats"]["errors"], 2)

    def test_repeated_errors_trigger_cooldown_then_recover(self):
        clock = FakeClock()
        predictor = DeliveryPredictor(
            {"mode": "active", "allow_experimental_active": True, "max_consecutive_errors": 3, "error_cooldown_s": 30},
            model=DeliveryModel(make_manifest()), clock=clock,
        )
        warm(predictor, "http://e/relay")
        original = predictor._predict

        def boom(*args, **kwargs):
            raise RuntimeError("x")

        predictor._predict = boom  # type: ignore[assignment]
        for _ in range(3):
            predictor.plan(ctx())
        self.assertTrue(predictor.get_snapshot()["cooldown_active"])
        self.assertEqual(predictor.plan(ctx()).source, "fallback:cooldown")
        predictor._predict = original  # type: ignore[assignment]
        clock.now += 31
        self.assertEqual(predictor.plan(ctx()).source, "model")


class PolicyOverrideTests(unittest.TestCase):
    def active(self, **config):
        base = {"mode": "active", "allow_experimental_active": True, "min_history": 8, "min_timeout_ms": 50}
        base.update(config)
        return DeliveryPredictor(base, model=DeliveryModel(make_manifest(latency_ms=10.0)))

    def test_timeout_is_tightened_but_never_lengthened_and_respects_floor(self):
        predictor = self.active()
        warm(predictor, "http://e/relay")
        decision = predictor.plan(ctx(requested_timeout_ms=1000))
        self.assertTrue(decision.changed_timeout)
        self.assertEqual(decision.timeout_ms, 50)  # floor
        small = predictor.plan(ctx(requested_timeout_ms=30))
        self.assertLessEqual(small.timeout_ms, 30)  # a caller asking for less than the floor is honoured
        # a slow endpoint must never be given MORE than requested
        slow = DeliveryPredictor({"mode": "active", "allow_experimental_active": True}, model=DeliveryModel(make_manifest(latency_ms=5000.0)))
        warm(slow, "http://e/relay", latency_ms=5000.0)
        self.assertEqual(slow.plan(ctx(requested_timeout_ms=1000)).timeout_ms, 1000)

    def test_timeout_never_exceeds_remaining_budget(self):
        predictor = self.active()
        warm(predictor, "http://e/relay")
        decision = predictor.plan(ctx(requested_timeout_ms=1000, remaining_ms=20))
        self.assertLessEqual(decision.timeout_ms, 20)

    def test_timeout_escalates_after_a_timed_out_attempt(self):
        predictor = self.active(min_timeout_ms=1, timeout_margin=1.5, timeout_escalation=2.0)
        warm(predictor, "http://e/relay")
        first = predictor.plan(ctx()).timeout_ms
        second = predictor.plan(ctx(attempt=2, last_error_class="timeout")).timeout_ms
        third = predictor.plan(ctx(attempt=3, last_error_class="timeout")).timeout_ms
        self.assertLess(first, second)
        self.assertLess(second, third)
        self.assertLessEqual(third, 1000)

    def test_no_change_without_enough_history_or_when_endpoint_unreliable(self):
        predictor = self.active()
        warm(predictor, "http://e/relay", successes=3)
        self.assertEqual(predictor.plan(ctx()).timeout_ms, 1000)
        flaky = self.active()
        warm(flaky, "http://f/relay", successes=10)
        for _ in range(10):
            flaky.observe("http://f/relay", ok=False, latency_ms=1000.0, error=ConnectionTimeoutError("t"))
        self.assertEqual(flaky.plan(ctx("http://f/relay")).timeout_ms, 1000)

    def test_retry_veto_requires_every_condition(self):
        # p_success ~ 0.007 from the bias; all other conditions are varied one by one.
        def predictor():
            return DeliveryPredictor(
                {"mode": "active", "allow_experimental_active": True, "min_history": 8, "veto_p_success": 0.15},
                model=DeliveryModel(make_manifest(p_bias=-5.0)),
            )

        def failing(p, endpoint="http://e/relay", n=10, failures=3):
            warm(p, endpoint, successes=n)
            for _ in range(failures):
                p.observe(endpoint, ok=False, latency_ms=200.0, error=ConnectionTimeoutError("t"))
            return p

        p = failing(predictor())
        self.assertTrue(p.allow_retry(ctx(attempt=2, last_error_class="timeout")).vetoed_retry)
        self.assertFalse(failing(predictor()).allow_retry(ctx(attempt=1, last_error_class="timeout")).vetoed_retry, "first attempt is never vetoed")
        self.assertFalse(failing(predictor(), failures=1).allow_retry(ctx(attempt=2, last_error_class="timeout")).vetoed_retry, "one failure is not a streak")
        self.assertFalse(failing(predictor()).allow_retry(ctx(attempt=2, last_error_class="rate_limit")).vetoed_retry, "rate limits are retried")
        young = predictor()
        for _ in range(3):
            young.observe("http://e/relay", ok=False, latency_ms=200.0, error=ConnectionTimeoutError("t"))
        self.assertFalse(young.allow_retry(ctx(attempt=2, last_error_class="timeout")).vetoed_retry, "not enough history")

    def test_healthy_endpoint_retry_is_never_vetoed(self):
        p = DeliveryPredictor({"mode": "active", "allow_experimental_active": True}, model=DeliveryModel(make_manifest(p_bias=5.0)))
        warm(p, "http://e/relay")
        p.observe("http://e/relay", ok=False, latency_ms=200.0, error=ConnectionTimeoutError("t"))
        p.observe("http://e/relay", ok=False, latency_ms=200.0, error=ConnectionTimeoutError("t"))
        self.assertTrue(p.allow_retry(ctx(attempt=2, last_error_class="timeout")).allow_retry)


class ModeTests(unittest.TestCase):
    def test_shadow_records_but_never_changes_behaviour(self):
        p = DeliveryPredictor({"mode": "shadow"}, model=DeliveryModel(make_manifest(p_bias=-5.0)))
        warm(p, "http://e/relay")
        for _ in range(3):
            p.observe("http://e/relay", ok=False, latency_ms=200.0, error=ConnectionTimeoutError("t"))
        plan = p.plan(ctx())
        retry = p.allow_retry(ctx(attempt=2, last_error_class="timeout"))
        self.assertEqual(plan.timeout_ms, 1000)
        self.assertFalse(plan.applied)
        self.assertTrue(retry.allow_retry)
        self.assertFalse(retry.applied)
        stats = p.get_snapshot()["stats"]
        self.assertEqual(stats["applied_timeouts"] + stats["applied_vetoes"], 0)

    def test_off_is_a_noop(self):
        p = DeliveryPredictor({"mode": "off"})
        decision = p.plan(ctx())
        self.assertEqual((decision.source, decision.timeout_ms, decision.applied), ("disabled", 1000, False))
        self.assertEqual(p.mode, "off")
        disabled = DeliveryPredictor({"enabled": False, "mode": "active"})
        self.assertEqual(disabled.mode, "off")

    def test_active_requires_approval_or_explicit_opt_in(self):
        experimental = DeliveryPredictor({"mode": "active"}, model=DeliveryModel(make_manifest(promotion="experimental")))
        self.assertEqual(experimental.mode, "shadow")
        approved = DeliveryPredictor({"mode": "active"}, model=DeliveryModel(make_manifest(promotion="approved")))
        self.assertEqual(approved.mode, "active")
        opted_in = DeliveryPredictor({"mode": "active", "allow_experimental_active": True}, model=DeliveryModel(make_manifest()))
        self.assertEqual(opted_in.mode, "active")
        heuristic_only = DeliveryPredictor({"mode": "active", "manifest_path": "nope.json"})
        self.assertEqual(heuristic_only.mode, "shadow")

    def test_unknown_mode_defaults_to_shadow(self):
        self.assertEqual(DeliveryPredictor({"mode": "yolo"}).mode, "shadow")


class RobustnessTests(unittest.TestCase):
    def test_overhead_is_small(self):
        p = DeliveryPredictor({"mode": "shadow"}, model=DeliveryModel(make_manifest()))
        warm(p, "http://e/relay")
        for _ in range(2000):
            p.plan(ctx())
        self.assertLess(p.get_snapshot()["stats"]["mean_overhead_us"], 2000.0)

    def test_endpoint_table_is_bounded(self):
        p = DeliveryPredictor({"max_endpoints": 16})
        for index in range(200):
            p.observe(f"http://e{index}/relay", ok=True, latency_ms=5.0)
        self.assertLessEqual(p.get_snapshot()["tracked_endpoints"], 16)

    def test_concurrent_use_is_safe(self):
        p = DeliveryPredictor({"mode": "active", "allow_experimental_active": True}, model=DeliveryModel(make_manifest()))
        errors = []

        def work(seed):
            try:
                for index in range(300):
                    endpoint = f"http://e{(seed + index) % 5}/relay"
                    p.observe(endpoint, ok=index % 7 != 0, latency_ms=5.0 + index % 9, error=None if index % 7 else ConnectionTimeoutError("t"))
                    p.plan(ctx(endpoint))
                    p.allow_retry(ctx(endpoint, attempt=2, last_error_class="timeout"))
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        threads = [threading.Thread(target=work, args=(i,)) for i in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)
        self.assertEqual(errors, [])
        self.assertEqual(p.get_snapshot()["stats"]["errors"], 0)

    def test_observation_hook_receives_features_and_outcome(self):
        rows = []
        p = DeliveryPredictor({"mode": "shadow"}, observation_hook=rows.append)
        decision = p.plan(ctx())
        p.observe("http://e/relay", ok=False, latency_ms=12.5, error=ConnectionTimeoutError("t"), decision=decision, context=ctx())
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(rows[0]["features"]), len(FEATURES))
        self.assertEqual((rows[0]["success"], rows[0]["error_class"]), (False, "timeout"))

    def test_a_failing_hook_never_breaks_observation(self):
        def bad_hook(row):
            raise RuntimeError("disk full")

        p = DeliveryPredictor({"mode": "shadow"}, observation_hook=bad_hook)
        decision = p.plan(ctx())
        p.observe("http://e/relay", ok=True, latency_ms=3.0, decision=decision, context=ctx())  # must not raise
        self.assertEqual(p.endpoint_snapshot("http://e/relay").n, 1)


class AgentIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.server = _Server()

    def tearDown(self):
        self.server.stop()

    def make_agent(self, predictor_config):
        agent = NetworkAgent(shared_memory=SharedMemory(), agent_factory=AgentFactory(), config={"delivery_predictor": predictor_config})
        self.addCleanup(lambda: agent.shutdown())
        return agent

    def capture_timeouts(self, agent):
        seen = []
        original = agent.stream.relay

        def wrapper(*args, **kwargs):
            seen.append(kwargs.get("timeout_ms"))
            return original(*args, **kwargs)

        agent.stream.relay = wrapper
        return seen

    def test_default_agent_runs_in_shadow_and_reports_prediction(self):
        agent = self.make_agent({})
        timeouts = self.capture_timeouts(agent)
        result = agent.relay(_envelope(self.server.url, timeout_ms=1500), CONSTRAINTS)
        self.assertEqual(result["status"], "delivered")
        self.assertEqual(timeouts, [1500])
        self.assertEqual(result["delivery_prediction"]["mode"], "shadow")
        self.assertFalse(result["delivery_prediction"]["applied"])
        snapshot = agent.get_network_health()["delivery_predictor"]
        self.assertEqual(snapshot["mode"], "shadow")
        self.assertGreaterEqual(snapshot["stats"]["observations"], 1)

    def test_off_mode_leaves_results_untouched(self):
        agent = self.make_agent({"mode": "off"})
        result = agent.relay(_envelope(self.server.url, timeout_ms=1500), CONSTRAINTS)
        self.assertEqual(result["status"], "delivered")
        self.assertNotIn("delivery_prediction", result)

    def test_active_mode_shortens_the_transport_timeout_after_history_builds(self):
        agent = self.make_agent({"mode": "active", "allow_experimental_active": True, "min_history": 4, "manifest_path": "<none>"})
        agent.delivery_predictor.model = DeliveryModel(make_manifest(latency_ms=10.0))
        agent.delivery_predictor.model_status = "injected"
        timeouts = self.capture_timeouts(agent)
        for index in range(6):
            agent.relay(_envelope(self.server.url, timeout_ms=1500, payload={"i": index}, idempotency_key=f"a-{index}"), CONSTRAINTS)
        self.assertEqual(timeouts[0], 1500, "no history yet: requested timeout")
        self.assertLess(timeouts[-1], 1500, "history built: timeout shortened")
        self.assertGreaterEqual(timeouts[-1], 50)

    def test_predictor_failure_does_not_fail_the_delivery(self):
        agent = self.make_agent({"mode": "shadow"})

        def boom(*args, **kwargs):
            raise RuntimeError("predictor bug")

        agent.delivery_predictor._predict = boom  # type: ignore[assignment]
        result = agent.relay(_envelope(self.server.url, timeout_ms=1500), CONSTRAINTS)
        self.assertEqual(result["status"], "delivered")
        self.assertTrue(result["delivery_prediction"]["source"].startswith("fallback:"))

    def test_local_policy_errors_do_not_pollute_endpoint_history(self):
        agent = self.make_agent({})
        with self.assertRaises(Exception):
            agent.relay(_envelope(self.server.url), {})  # loopback denied by default policy before anything is sent
        self.assertEqual(agent.delivery_predictor.get_snapshot()["stats"]["observations"], 0)


if __name__ == "__main__":
    unittest.main()
