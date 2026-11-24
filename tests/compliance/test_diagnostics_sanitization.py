"""Audit finding 4.9: no PII/secret may reach log handlers."""
from __future__ import annotations

import io
import logging
import unittest

from src.agents.governance.utils.diagnostics import REDACTED, SanitizingFilter, install_log_sanitizer, redact_text, sanitize


class DiagnosticsTests(unittest.TestCase):
    def test_text_patterns(self):
        for raw in (
            "mail jan.devries@example.nl now", "Authorization: Bearer abcdefghijklmnop123", "call +31 6 12345678 please",
            "iban NL91ABNA0417164300", "card 4111 1111 1111 1111", "from 192.168.10.20", "password=hunter2",
            "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTYifQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV",
        ):
            out = redact_text(raw)
            self.assertIn(REDACTED, out, raw)

    def test_sensitive_keys_redacted_but_ids_kept(self):
        out = sanitize({"email": "a@b.nl", "request_id": "r1", "nested": {"api_key": "x", "count": 3}, "full_name": "Jan"})
        self.assertEqual(out["email"], REDACTED)
        self.assertEqual(out["full_name"], REDACTED)
        self.assertEqual(out["request_id"], "r1")
        self.assertEqual(out["nested"], {"api_key": REDACTED, "count": 3})

    def test_bounds_and_never_raises(self):
        deep = cur = {}
        for _ in range(20):
            cur["k"] = {}
            cur = cur["k"]
        sanitize(deep)

        class Bad:
            def __repr__(self):
                raise RuntimeError("no")

        self.assertEqual(sanitize(Bad()), REDACTED)

    def test_logging_filter_end_to_end(self):
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        logger = logging.getLogger("slai.test.diag")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        logger.addHandler(handler)
        self.assertTrue(install_log_sanitizer(logger))
        self.assertFalse(install_log_sanitizer(logger))  # idempotent
        logger.info("user %s sent %s", "jan@example.nl", {"password": "p"})
        try:
            raise ValueError("failed for jan@example.nl")
        except ValueError:
            logger.exception("boom")
        text = stream.getvalue()
        self.assertNotIn("jan@example.nl", text)
        self.assertNotIn('"p"', text)
        self.assertIn(REDACTED, text)
        logger.removeHandler(handler)

    def test_filter_is_a_filter(self):
        self.assertIsInstance(SanitizingFilter(), logging.Filter)


if __name__ == "__main__":
    unittest.main()
