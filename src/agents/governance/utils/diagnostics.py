"""Central diagnostics sanitisation (audit 4.9, GDPR Art. 5(1)(c), 5(1)(f), 25, 32).

SLAI had at least five independent ``sanitize_for_logging``/``redact_text`` helpers with
different coverage. This module is the single, stdlib-only reference implementation and
also a ``logging.Filter`` that sanitises every record (message, args and exception text) so
personal data cannot reach any handler regardless of which helper a caller used.

Deterministic, pattern based, conservative: it is a safety net, NOT a substitute for not
logging personal data. Detection is best effort and is documented as such.
"""
from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from typing import Any

REDACTED = "[REDACTED]"
_MAX_TEXT = 4096
_MAX_DEPTH = 8

_SENSITIVE_KEY = re.compile(
    r"(pass(word|wd)?|secret|token|api[_-]?key|authorization|cookie|credential|private[_-]?key|"
    r"ssn|bsn|iban|card|cvv|pin|email|phone|mobile|address|birth|dob|name|surname|passport|"
    r"national[_-]?id|health|diagnos|biometric|face|fingerprint)",
    re.IGNORECASE,
)
_NON_SENSITIVE_KEY = re.compile(r"(^|_)(request_id|record_id|entry_id|policy|policy_id|status|decision|rule_id|count|size)(_|$)", re.I)

_PATTERNS = (
    ("email", re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")),
    ("bearer", re.compile(r"(?i)\b(bearer|basic)\s+[A-Za-z0-9._~+/=\-]{8,}")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\b")),
    ("iban", re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}(?:[A-Z0-9]?){0,16}\b")),
    ("card", re.compile(r"\b(?:\d[ \-]?){13,19}\b")),
    ("ipv4", re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")),
    ("phone", re.compile(r"(?<!\w)\+?\d[\d \-()]{8,}\d(?!\w)")),
    ("api_key", re.compile(r"\b(?:sk|pk|ghp|gho|xox[abp]|AKIA)[A-Za-z0-9_\-]{12,}\b")),
    ("kv_secret", re.compile(r"(?i)\b(password|passwd|secret|token|api[_-]?key)\s*[=:]\s*\S+")),
)


def redact_text(text: Any) -> str:
    value = text if isinstance(text, str) else str(text)
    if len(value) > _MAX_TEXT:
        value = value[:_MAX_TEXT] + "…[truncated]"
    for _label, pattern in _PATTERNS:
        value = pattern.sub(REDACTED, value)
    return value


def sanitize(value: Any, *, _depth: int = 0, _key: str = "") -> Any:
    """Return a copy of ``value`` that is safe to log. Never raises."""
    try:
        if _depth > _MAX_DEPTH:
            return "[DEPTH_LIMIT]"
        if _key and _SENSITIVE_KEY.search(_key) and not _NON_SENSITIVE_KEY.search(_key):
            return REDACTED
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, (bytes, bytearray)):
            return f"[{len(value)} bytes]"
        if isinstance(value, str):
            return redact_text(value)
        if isinstance(value, Mapping):
            return {str(k)[:128]: sanitize(v, _depth=_depth + 1, _key=str(k)) for k, v in list(value.items())[:256]}
        if isinstance(value, (list, tuple, set, frozenset)):
            return [sanitize(v, _depth=_depth + 1, _key=_key) for v in list(value)[:256]]
        return redact_text(repr(value))
    except Exception:  # sanitiser failure must not leak: replace entirely
        return REDACTED


class SanitizingFilter(logging.Filter):
    """Sanitise message, arguments and exception text of every record passing through."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        try:
            if record.args:
                if isinstance(record.args, Mapping):
                    record.args = {k: sanitize(v, _key=str(k)) for k, v in record.args.items()}
                else:
                    record.args = tuple(sanitize(a) for a in record.args)
            record.msg = redact_text(record.msg) if isinstance(record.msg, str) else sanitize(record.msg)
            if record.exc_info:
                record.exc_text = redact_text(logging.Formatter().formatException(record.exc_info))
                record.exc_info = None
        except Exception:
            record.msg, record.args, record.exc_info = REDACTED, (), None
        return True


_INSTALLED = "_slai_sanitizing_filter"


def install_log_sanitizer(logger: logging.Logger | None = None) -> bool:
    """Attach the filter to every handler of ``logger`` (root by default). Idempotent."""
    target = logger or logging.getLogger()
    installed = False
    for handler in list(target.handlers):
        if not getattr(handler, _INSTALLED, False):
            handler.addFilter(SanitizingFilter())
            setattr(handler, _INSTALLED, True)
            installed = True
    return installed
