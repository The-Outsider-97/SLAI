"""Regression tests for the SLAI v2.4 Network Agent hardening.

Every test drives the real NetworkAgent / network subsystem against a loopback HTTP server;
nothing is mocked except where a failure has to be injected at a named seam.
Run:  python -m unittest tests.test_network_agent_hardening -v
"""
from __future__ import annotations

import json
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from src.agents.agent_factory import AgentFactory
from src.agents.collaborative.shared_memory import SharedMemory
from src.agents.network.utils.network_errors import (
    DestinationDeniedError,
    NoRouteAvailableError,
    SendFailureError,
    SessionClosedError,
)
from src.agents.network.utils.network_helpers import json_safe, sanitize_for_logging
from src.agents.network_agent import NetworkAgent

CONSTRAINTS = {"tls_required": False, "allow_loopback_hosts": True}


class _Server:
    """Loopback HTTP server. ``script(hit_index) -> (status, delay_s)``."""

    def __init__(self, script=None):
        self.hits = []
        self.script = script or (lambda i: (200, 0.0))
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):  # silence
                pass

            def _do(self):
                length = int(self.headers.get("Content-Length", "0") or 0)
                if length:
                    self.rfile.read(length)
                index = len(outer.hits)
                outer.hits.append((self.command, self.path))
                status, delay = outer.script(index)
                if delay:
                    time.sleep(delay)
                body = json.dumps({"ok": status < 400, "hit": index}).encode()
                try:
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                except OSError:
                    pass

            do_GET = do_POST = do_PUT = _do

        self._srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._srv.daemon_threads = True
        threading.Thread(target=self._srv.serve_forever, daemon=True).start()

    @property
    def url(self) -> str:
        host, port = self._srv.server_address
        return f"http://{host}:{port}/relay"

    def stop(self) -> None:
        self._srv.shutdown()
        self._srv.server_close()


def _envelope(url: str, **overrides) -> dict:
    envelope = {
        "payload": {"hello": "test"},
        "channel": "http",
        "protocol": "http",
        "endpoint": url,
        "route": "primary",
        "operation": "send",
        "timeout_ms": 1500,
        "metadata": {"source": "unit-test"},
    }
    envelope.update(overrides)
    return envelope


class _AgentTestCase(unittest.TestCase):
    def setUp(self):
        self.server = _Server()
        self.shared_memory = SharedMemory()
        self.agent = NetworkAgent(shared_memory=self.shared_memory, agent_factory=AgentFactory())

    def tearDown(self):
        self.server.stop()
        try:
            self.agent.shutdown()
        except Exception:
            pass


class JsonSafeTests(unittest.TestCase):
    def _nested(self, depth: int):
        value = {"leaf": "x" * 40}
        for _ in range(depth):
            value = {"child": value, "pad": list(range(5))}
        return value

    def test_json_safe_is_idempotent_on_deep_structures(self):
        deep = self._nested(40)
        once = json_safe(deep)
        twice = json_safe(once)
        thrice = json_safe(twice)
        self.assertEqual(once, twice)
        self.assertEqual(twice, thrice)

    def test_json_safe_size_does_not_grow_when_reapplied(self):
        deep = self._nested(40)
        value = json_safe(deep)
        size = len(json.dumps(value))
        for _ in range(8):
            value = json_safe(value)
        self.assertEqual(len(json.dumps(value)), size)

    def test_depth_limited_repr_is_bounded(self):
        value = json_safe({"a": [{"b": "y" * 100_000}]}, max_depth=1)
        self.assertLess(len(json.dumps(value)), 2_000)

    def test_sanitize_does_not_leak_secrets_at_depth_limit(self):
        secret = "s3cr3t-token-value"
        deep = {"l1": {"l2": {"l3": {"l4": {"l5": {"l6": {"l7": {"l8": {"l9": {"l10": {"l11": {"password": secret}}}}}}}}}}}}
        rendered = json.dumps(sanitize_for_logging(deep))
        self.assertNotIn(secret, rendered)


class RelayDeliveryTests(_AgentTestCase):
    def test_bookkeeping_failure_after_delivery_does_not_resend(self):
        calls = {"n": 0}
        original = self.agent.metrics.record_success

        def flaky(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("metrics backend unavailable (injected)")
            return original(*args, **kwargs)

        self.agent.metrics.record_success = flaky
        result = self.agent.relay(_envelope(self.server.url), CONSTRAINTS)
        self.assertEqual(result["status"], "delivered")
        self.assertEqual(len(self.server.hits), 1, "message must be delivered exactly once")
        self.assertTrue(result.get("bookkeeping_warnings"))

    def test_keyboard_interrupt_releases_idempotency_claim(self):
        original = self.agent.stream.relay

        def boom(*args, **kwargs):
            raise KeyboardInterrupt()

        self.agent.stream.relay = boom
        with self.assertRaises(KeyboardInterrupt):
            self.agent.relay(_envelope(self.server.url, idempotency_key="k-cancel"), CONSTRAINTS)
        stats = (self.shared_memory.get(self.agent.agent_stats_key) or {}).get("NetworkAgent", {})
        self.assertEqual(stats.get("active_tasks"), 0)

        self.agent.stream.relay = original
        result = self.agent.relay(_envelope(self.server.url, idempotency_key="k-cancel"), CONSTRAINTS)
        self.assertEqual(result["status"], "delivered")

    def test_relay_after_shutdown_is_refused(self):
        self.agent.relay(_envelope(self.server.url), CONSTRAINTS)
        self.agent.shutdown()
        with self.assertRaises(SessionClosedError):
            self.agent.relay(_envelope(self.server.url, payload={"n": 2}), CONSTRAINTS)
        self.assertEqual(len(self.server.hits), 1)

    def test_history_entries_are_compact(self):
        # SharedMemory may be process-wide, so count only the rows this test adds.
        before = len(self.shared_memory.get(self.agent.history_shared_key) or [])
        for index in range(3):
            self.agent.relay(_envelope(self.server.url, payload={"i": index}, idempotency_key=f"h-{index}"), CONSTRAINTS)
        history = self.shared_memory.get(self.agent.history_shared_key) or []
        added = history[before:] if len(history) > before else history[-3:]
        self.assertEqual(len(added), 3)
        for row in added:
            self.assertLess(len(json.dumps(row)), 2_000)
            self.assertEqual(row.get("status"), "delivered")

    def test_repeated_relays_keep_memory_bounded(self):
        sizes = []
        for index in range(12):
            self.agent.relay(_envelope(self.server.url, payload={"i": index}, idempotency_key=f"g-{index}"), CONSTRAINTS)
            sizes.append(len(json.dumps(json_safe(self.shared_memory.get(self.agent.history_shared_key) or []))))
        # Linear in history length (compact rows), never geometric.
        self.assertLess(sizes[-1], sizes[0] * 20)

    def test_envelope_max_attempts_cannot_exceed_ceiling(self):
        server = _Server(lambda i: (503, 0.0))
        try:
            with self.assertRaises(Exception):
                self.agent.relay(_envelope(server.url, metadata={"max_attempts": 500}), CONSTRAINTS)
            self.assertLessEqual(len(server.hits), self.agent.max_attempts_ceiling)
        finally:
            server.stop()

    def test_overall_deadline_bounds_wall_clock(self):
        server = _Server(lambda i: (200, 1.0))
        try:
            started = time.monotonic()
            with self.assertRaises(Exception):
                self.agent.relay(_envelope(server.url, timeout_ms=5000), {**CONSTRAINTS, "deadline_ms": 300})
            self.assertLess(time.monotonic() - started, 1.0)
        finally:
            server.stop()

    def test_concurrent_relays_keep_stats_consistent(self):
        total, errors = 16, []

        def work(index):
            try:
                self.agent.relay(_envelope(self.server.url, payload={"i": index}, idempotency_key=f"c-{index}"), CONSTRAINTS)
            except Exception as exc:  # pragma: no cover - failure path asserted below
                errors.append(exc)

        threads = [threading.Thread(target=work, args=(i,)) for i in range(total)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)
        self.assertEqual(errors, [])
        row = (self.shared_memory.get(self.agent.agent_stats_key) or {}).get("NetworkAgent", {})
        self.assertEqual(row.get("active_tasks"), 0)
        self.assertEqual(len(self.server.hits), total)


class PolicyParityTests(_AgentTestCase):
    def test_relay_and_receive_apply_the_same_destination_policy(self):
        with self.assertRaises(DestinationDeniedError):
            self.agent.relay(_envelope(self.server.url), {})
        with self.assertRaises(DestinationDeniedError):
            self.agent.receive("http", 1500, endpoint=self.server.url, protocol="http")
        self.assertEqual(self.server.hits, [], "denied destinations must not be contacted")


class RetryBackoffTests(_AgentTestCase):
    def test_backoff_uses_wait_ms(self):
        agent = self.agent
        agent.retry_sleep_enabled = True
        agent.max_retry_sleep_ms = 1500
        payload = {"retry": {"decision": {"wait_ms": 373, "retry_after_ms": None}}}
        self.assertEqual(agent._planned_sleep_ms(payload), 373)

    def test_backoff_is_capped_and_disabled_when_configured(self):
        agent = self.agent
        agent.retry_sleep_enabled = True
        agent.max_retry_sleep_ms = 500
        self.assertEqual(agent._planned_sleep_ms({"retry": {"decision": {"wait_ms": 9999}}}), 500)
        agent.retry_sleep_enabled = False
        self.assertEqual(agent._planned_sleep_ms({"retry": {"decision": {"wait_ms": 9999}}}), 0)

    def test_backoff_falls_back_to_retry_after_ms(self):
        agent = self.agent
        agent.retry_sleep_enabled = True
        agent.max_retry_sleep_ms = 1500
        self.assertEqual(agent._planned_sleep_ms({"retry": {"decision": {"retry_after_ms": 250}}}), 250)

    def test_never_retryable_error_is_not_retried(self):
        error = NoRouteAvailableError("no route", context={"endpoint": "http://x/relay", "channel": "http", "protocol": "http"})
        result = self.agent.reliability.retry_policy.evaluate(
            error, attempt=1, max_attempts=5, profile="default", endpoint="http://x/relay",
            channel="http", protocol="http", route="primary", idempotent=True,
        )
        self.assertFalse(result["decision"]["should_retry"])

    def test_transient_error_is_still_retried(self):
        error = SendFailureError("upstream 503", context={"endpoint": "http://x/relay", "channel": "http", "protocol": "http"})
        result = self.agent.reliability.retry_policy.evaluate(
            error, attempt=1, max_attempts=5, profile="default", endpoint="http://x/relay",
            channel="http", protocol="http", route="primary", idempotent=True, status_code=503,
        )
        self.assertTrue(result["decision"]["should_retry"])


class EndpointHealthTests(_AgentTestCase):
    def test_single_failure_does_not_mark_endpoint_unhealthy(self):
        metrics = self.agent.metrics
        status = metrics._derive_endpoint_status({"total_events": 1, "success_rate": 0.0, "error_rate": 1.0, "consecutive_failures": 1})
        self.assertNotEqual(status, "unhealthy")

    def test_consecutive_failures_still_trip_unhealthy_with_low_volume(self):
        metrics = self.agent.metrics
        status = metrics._derive_endpoint_status(
            {"total_events": metrics.unhealthy_consecutive_failures_threshold, "success_rate": 0.0, "error_rate": 1.0,
             "consecutive_failures": metrics.unhealthy_consecutive_failures_threshold}
        )
        self.assertEqual(status, "unhealthy")

    def test_sustained_high_error_rate_is_unhealthy(self):
        metrics = self.agent.metrics
        status = metrics._derive_endpoint_status({"total_events": 100, "success_rate": 0.5, "error_rate": 0.5, "consecutive_failures": 1})
        self.assertEqual(status, "unhealthy")

class LoneEndpointAvailabilityTests(_AgentTestCase):
    """The single-endpoint lockout chain: timeout poisons the pooled connection, one sample disqualifies the
    endpoint, a disqualified endpoint gets no traffic, and an open circuit never reaches half-open."""

    def _safe_relay(self, url, **kw):
        self.agent.lifecycle.deliveries.default_max_attempts = 1  # one attempt per call: each call observes the endpoint once
        try:
            return self.agent.relay(_envelope(url, **kw), CONSTRAINTS), None
        except Exception as exc:  # noqa: BLE001
            return None, exc

    def test_endpoint_is_reachable_again_right_after_a_timeout(self):
        server = _Server(lambda i: (200, 0.6) if i == 0 else (200, 0.0))
        try:
            _, error = self._safe_relay(server.url, timeout_ms=150, idempotency_key="ts-1")
            self.assertIsNotNone(error, "first call must time out")
            result, error = self._safe_relay(server.url, timeout_ms=1500, idempotency_key="ts-2")
            self.assertIsNone(error, f"lone endpoint must not stay locked out: {error!r}")
            self.assertEqual(result["status"], "delivered")
            self.assertEqual(len(server.hits), 2, "second call must reach the server on a fresh connection")
        finally:
            server.stop()

    def test_one_failure_does_not_stop_traffic_to_the_only_endpoint(self):
        server = _Server(lambda i: (503, 0.0) if i == 0 else (200, 0.0))
        try:
            self._safe_relay(server.url, idempotency_key="lo-1")
            result, error = self._safe_relay(server.url, idempotency_key="lo-2")
            self.assertIsNone(error, repr(error))
            self.assertGreaterEqual(len(server.hits), 2)
        finally:
            server.stop()

    def test_open_circuit_on_lone_endpoint_recovers_after_open_timeout(self):
        from datetime import datetime, timedelta, timezone

        breaker = self.agent.reliability.circuit_breaker
        healthy = {"now": False}
        server = _Server(lambda i: (200, 0.0) if healthy["now"] else (503, 0.0))
        try:
            for index in range(8):
                self._safe_relay(server.url, idempotency_key=f"oc-{index}")
            self.assertTrue(breaker._circuits, "the failures must have created a circuit")
            self.assertIn("open", {record.state for record in breaker._circuits.values()})
            # Let the open window elapse (backdate it rather than sleeping 15-45 s).
            expired = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
            for record in breaker._circuits.values():
                if record.state == "open":
                    record.open_until = expired
            healthy["now"] = True  # the endpoint has recovered while the circuit was open
            result, error = self._safe_relay(server.url, idempotency_key="oc-final")
            self.assertIsNone(error, repr(error))
            self.assertEqual(result["status"], "delivered")
        finally:
            server.stop()


class RetryBookkeepingTests(_AgentTestCase):
    def test_retried_then_delivered_message_is_recorded_as_completed(self):
        server = _Server(lambda i: (200, 0.6) if i == 0 else (200, 0.0))
        try:
            result = self.agent.relay(_envelope(server.url, timeout_ms=150, idempotency_key="rb-1"), CONSTRAINTS)
            self.assertEqual(result["status"], "delivered")
            self.assertEqual(len(server.hits), 2)
            self.assertFalse(result.get("bookkeeping_warnings"), result.get("bookkeeping_warnings"))
            record = self.agent.lifecycle.idempotency.get_record("rb-1", include_history=False)
            self.assertEqual(record["status"], "completed")
        finally:
            server.stop()


class BoundedGrowthTests(_AgentTestCase):
    def test_stream_session_records_are_capped(self):
        cap = self.agent.stream.max_session_records
        for index in range(cap + 20):
            self.agent.relay(_envelope(self.server.url, payload={"i": index}, idempotency_key=f"sc-{index}"), CONSTRAINTS)
        self.assertLessEqual(len(self.agent.stream._sessions), cap)

    def test_session_snapshots_are_capped(self):
        memory = self.agent.network_memory
        cap = memory.max_session_snapshots
        for index in range(cap + 15):
            memory.update_session_snapshot(f"snap-{index}", {"n": index})
        stored = memory.get("network.session.snapshot", default={}, include_metadata=False)
        self.assertLessEqual(len(stored), cap)
        self.assertIn(f"snap-{cap + 14}", stored, "newest snapshot must be kept")
        self.assertNotIn("snap-0", stored, "oldest snapshot must be evicted")


if __name__ == "__main__":
    unittest.main()
