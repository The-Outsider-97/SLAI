"""Immutable, reusable failure taxonomy.

The BaseAgent runtime consumes a ``FailureTaxonomy`` to decide whether a
task result is a failure. Subclasses extend the default by deriving a new
instance — taxonomies are frozen and never mutated in place.

This module demonstrates three cross-cutting patterns used throughout the
base subsystem:

* **Error handling** — every public entry point raises ``BaseValidationError``
  or ``BaseConfigurationError`` on malformed input, and config loading
  failures are translated into structured exceptions with rich context.
* **Helper consumption** — text normalization, mapping merging, coercion,
  and JSON-safe repr all come from ``base_helpers`` rather than being
  reimplemented locally.
* **Logging** — lifecycle events (config load, taxonomy extension, skipped
  inputs) are emitted through the shared ``SLAI Failure Taxonomy`` logger.
"""

from __future__ import annotations

import time

from dataclasses import dataclass, replace
from typing import Any, Dict, FrozenSet, Iterable, Mapping, Optional, Tuple

from .utils.config_loader import get_config_section
from .utils.base_errors import *
from .utils.base_helpers import *
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("SLAI Failure Taxonomy")
printer = PrettyPrinter()

_CONFIG_SECTION = "failure_taxonomy"


# ---------------------------------------------------------------------------
# Local normalization
#
# Replaces the previously-imported private ``_normalize`` from base_helpers.
# That symbol was never bound because ``from .utils.base_helpers import *``
# honors ``__all__`` and ``_normalize`` starts with an underscore. Keeping a
# tiny local normalizer is more honest about the dependency boundary and
# documents the exact normalization contract for taxonomy tokens.
# ---------------------------------------------------------------------------

def _coerce_token_sequence(values: Any, *, field: str) -> Tuple[str, ...]:
    """Validate and normalize an iterable of taxonomy tokens.

    Empty/whitespace-only tokens are silently dropped with a debug log.
    Non-iterable inputs raise ``BaseValidationError``.
    """
    if values is None:
        return ()
    try:
        sequence = ensure_sequence(values, field, allow_str=True)
    except BaseValidationError:
        raise
    except Exception as exc:
        raise BaseValidationError(
            f"{field} must be an iterable of strings.",
            component="failure_taxonomy",
            details={"field": field, "received": safe_repr(values)},
            cause=exc,
        ) from exc

    cleaned: list[str] = []
    for raw in sequence:
        token = _normalize_token(raw)
        if not token:
            logger.debug("Dropping empty token in %s: %r", field, raw)
            continue
        cleaned.append(token)
    return tuple(dict.fromkeys(cleaned))  # de-dup, preserve order


# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FailureTaxonomy:
    """Declarative failure vocabulary consumed by ``BaseAgent._is_failure_result``.

    Use the ``with_*`` builders to derive an extended taxonomy. Prefer
    declaring it as a class attribute on the agent::

        class TradingAgent(BaseAgent):
            FAILURE_TAXONOMY = DEFAULT_FAILURE_TAXONOMY.with_statuses(
                "market_closed", "no_liquidity", "slippage_exceeded",
            )
    """

    failure_statuses: FrozenSet[str] = frozenset()
    success_flag_keys: Tuple[str, ...] = ()
    failure_flag_keys: Tuple[str, ...] = ()
    error_carrier_keys: Tuple[str, ...] = ()
    failure_string_prefixes: Tuple[str, ...] = ()
    http_status_keys: Tuple[str, ...] = ()
    treat_sequence_failures_as_failure: bool = False

    # ----------------------------------------------------------
    # Immutable builders
    # ----------------------------------------------------------

    def with_statuses(self, *statuses: str) -> "FailureTaxonomy":
        tokens = _coerce_token_sequence(statuses, field="statuses")
        if not tokens:
            logger.warning("with_statuses called with no usable statuses; returning self.")
            return self
        extended = self.failure_statuses | frozenset(tokens)
        added = len(extended) - len(self.failure_statuses)
        logger.info(
            "Extended failure_statuses by %d (now %d total). Added: %s",
            added, len(extended), sorted(set(tokens) - self.failure_statuses),
        )
        return replace(self, failure_statuses=extended)

    def with_success_flags(self, *keys: str) -> "FailureTaxonomy":
        cleaned = _clean_flag_keys(keys, field="success_flag_keys")
        if not cleaned:
            logger.warning("with_success_flags called with no usable keys; returning self.")
            return self
        merged = tuple(dict.fromkeys((*self.success_flag_keys, *cleaned)))
        logger.info("Extended success_flag_keys by %d (now %d).", len(merged) - len(self.success_flag_keys), len(merged))
        return replace(self, success_flag_keys=merged)

    def with_failure_flags(self, *keys: str) -> "FailureTaxonomy":
        cleaned = _clean_flag_keys(keys, field="failure_flag_keys")
        if not cleaned:
            logger.warning("with_failure_flags called with no usable keys; returning self.")
            return self
        merged = tuple(dict.fromkeys((*self.failure_flag_keys, *cleaned)))
        logger.info("Extended failure_flag_keys by %d (now %d).", len(merged) - len(self.failure_flag_keys), len(merged))
        return replace(self, failure_flag_keys=merged)

    def with_error_carriers(self, *keys: str) -> "FailureTaxonomy":
        cleaned = _clean_flag_keys(keys, field="error_carrier_keys")
        if not cleaned:
            logger.warning("with_error_carriers called with no usable keys; returning self.")
            return self
        merged = tuple(dict.fromkeys((*self.error_carrier_keys, *cleaned)))
        logger.info("Extended error_carrier_keys by %d (now %d).", len(merged) - len(self.error_carrier_keys), len(merged))
        return replace(self, error_carrier_keys=merged)

    def with_string_prefixes(self, *prefixes: str) -> "FailureTaxonomy":
        cleaned = _clean_string_prefixes(prefixes)
        if not cleaned:
            logger.warning("with_string_prefixes called with no usable prefixes; returning self.")
            return self
        merged = tuple(dict.fromkeys((*self.failure_string_prefixes, *cleaned)))
        logger.info("Extended failure_string_prefixes by %d (now %d).", len(merged) - len(self.failure_string_prefixes), len(merged))
        return replace(self, failure_string_prefixes=merged)

    def with_http_status_keys(self, *keys: str) -> "FailureTaxonomy":
        cleaned = _clean_flag_keys(keys, field="http_status_keys")
        if not cleaned:
            logger.warning("with_http_status_keys called with no usable keys; returning self.")
            return self
        merged = tuple(dict.fromkeys((*self.http_status_keys, *cleaned)))
        logger.info("Extended http_status_keys by %d (now %d).", len(merged) - len(self.http_status_keys), len(merged))
        return replace(self, http_status_keys=merged)

    def with_sequence_failures(self, enabled: bool = True) -> "FailureTaxonomy":
        flag = coerce_bool(enabled, default=False)
        if flag == self.treat_sequence_failures_as_failure:
            logger.debug("with_sequence_failures: no change (already %s).", flag)
            return self
        logger.info("treat_sequence_failures_as_failure -> %s", flag)
        return replace(self, treat_sequence_failures_as_failure=flag)

    # ----------------------------------------------------------
    # Construction from config
    # ----------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        base: Optional["FailureTaxonomy"] = None,
        *,
        section: Optional[Mapping[str, Any]] = None,
    ) -> "FailureTaxonomy":
        """Derive a taxonomy by layering an optional config section over ``base``.

        Config shape (all keys optional)::

            failure_taxonomy:
              treat_sequence_failures_as_failure: false
              extra_failure_statuses:        [market_closed, no_liquidity]
              extra_success_flag_keys:       [healthy]
              extra_failure_flag_keys:       [underwater]
              extra_error_carrier_keys:      [broker_reject_reason]
              extra_string_prefixes:         ["[broker reject]"]
              extra_http_status_keys:        [result_code]
        """
        taxonomy = base if base is not None else DEFAULT_FAILURE_TAXONOMY

        if section is None:
            section = load_failure_taxonomy_config()
        else:
            try:
                section = ensure_mapping(section, "failure_taxonomy_section")
            except BaseValidationError:
                logger.warning(
                    "from_config: section is not a mapping (%s); ignoring.",
                    type(section).__name__,
                )
                return taxonomy

        if not section:
            logger.debug("from_config: empty section; returning base taxonomy.")
            return taxonomy

        if "treat_sequence_failures_as_failure" in section:
            taxonomy = taxonomy.with_sequence_failures(coerce_bool(section["treat_sequence_failures_as_failure"], default=False))

        taxonomy = taxonomy.with_statuses(*_section_list(section, "extra_failure_statuses"))
        taxonomy = taxonomy.with_success_flags(*_section_list(section, "extra_success_flag_keys"))
        taxonomy = taxonomy.with_failure_flags(*_section_list(section, "extra_failure_flag_keys"))
        taxonomy = taxonomy.with_error_carriers(*_section_list(section, "extra_error_carrier_keys"))
        taxonomy = taxonomy.with_string_prefixes(*_section_list(section, "extra_string_prefixes"))
        taxonomy = taxonomy.with_http_status_keys(*_section_list(section, "extra_http_status_keys"))

        logger.info("FailureTaxonomy.from_config -> %s", taxonomy.summary())
        return taxonomy

    # ----------------------------------------------------------
    # Queries
    # ----------------------------------------------------------

    @staticmethod
    def normalize_status(raw: Any) -> str:
        return _normalize_token(raw)

    def is_failure_status(self, raw: Any) -> bool:
        return _normalize_token(raw) in self.failure_statuses

    def summary(self) -> Dict[str, Any]:
        """Compact dict view, useful for logs and diagnostics."""
        return {
            "failure_statuses": len(self.failure_statuses),
            "success_flag_keys": len(self.success_flag_keys),
            "failure_flag_keys": len(self.failure_flag_keys),
            "error_carrier_keys": len(self.error_carrier_keys),
            "failure_string_prefixes": len(self.failure_string_prefixes),
            "http_status_keys": len(self.http_status_keys),
            "treat_sequence_failures_as_failure": self.treat_sequence_failures_as_failure,
        }


# ---------------------------------------------------------------------------
# Token-cleaning helpers used by the builders
# ---------------------------------------------------------------------------

def _clean_flag_keys(keys: Iterable[str], *, field: str) -> Tuple[str, ...]:
    """Normalize and validate a batch of mapping-key tokens."""
    cleaned: list[str] = []
    for raw in keys:
        try:
            token = ensure_non_empty_string(raw, field, strip=True)
        except BaseValidationError:
            logger.debug("Skipping non-string or empty %s entry: %r", field, raw)
            continue
        cleaned.append(token.lower())
    return tuple(dict.fromkeys(cleaned))


def _clean_string_prefixes(prefixes: Iterable[str]) -> Tuple[str, ...]:
    """Normalize string prefixes; prefixes are lowercased but otherwise preserved."""
    cleaned: list[str] = []
    for raw in prefixes:
        try:
            token = ensure_text(raw)
        except Exception:
            logger.debug("Skipping unrenderable prefix: %r", raw)
            continue
        if not token.strip():
            continue
        cleaned.append(token.strip().lower())
    return tuple(dict.fromkeys(cleaned))


def _section_list(section: Mapping[str, Any], key: str) -> Tuple[str, ...]:
    """Extract a list-valued config key, tolerating scalars."""
    value = section.get(key)
    if value is None:
        return ()
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(value)
    # Tolerate a single scalar as a one-element list.
    return (value,)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_failure_taxonomy_config(*, config: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Load the ``failure_taxonomy`` section with full error handling.

    Never raises for a *missing* section — returns ``{}`` instead.  Raises
    ``BaseConfigurationError`` only when the section exists but cannot be
    interpreted as a mapping, or when the underlying loader itself fails in
    an unexpected way.
    """
    started = time.perf_counter()
    try:
        if config is None:
            raw_section = get_config_section(_CONFIG_SECTION) or {}
        else:
            merged = deep_merge_dicts({"__root__": config}, inplace=False)
            root = merged.get("__root__", {})
            raw_section = root.get(_CONFIG_SECTION) or {}
    except BaseConfigurationError:
        raise
    except Exception as exc:
        logger.exception("Failure taxonomy config load failed.")
        raise BaseConfigurationError(
            "Unable to load failure_taxonomy configuration.",
            component="failure_taxonomy",
            details={"section": _CONFIG_SECTION},
            cause=exc,
        ) from exc

    if raw_section in (None, {}):
        logger.debug("No failure_taxonomy section present; using defaults.")
        return {}

    if not isinstance(raw_section, Mapping):
        raise BaseConfigurationError(
            "failure_taxonomy config section must be a mapping.",
            component="failure_taxonomy",
            details={
                "section": _CONFIG_SECTION,
                "received_type": type(raw_section).__name__,
                "received_preview": safe_repr(raw_section),
            },
        )

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    logger.info("Loaded failure_taxonomy config in %.2f ms (keys=%s).", elapsed_ms, sorted(raw_section.keys()))
    return dict(raw_section)


def default_taxonomy_with_config(*, config: Optional[Mapping[str, Any]] = None) -> FailureTaxonomy:
    """Convenience wrapper: default taxonomy + any configured extensions."""
    try:
        section = load_failure_taxonomy_config(config=config)
    except BaseConfigurationError:
        logger.warning("Falling back to default taxonomy after config error.")
        return DEFAULT_FAILURE_TAXONOMY
    return FailureTaxonomy.from_config(DEFAULT_FAILURE_TAXONOMY, section=section)


# ---------------------------------------------------------------------------
# The default taxonomy
# ---------------------------------------------------------------------------

DEFAULT_FAILURE_TAXONOMY = FailureTaxonomy(
    failure_statuses=frozenset({
        # generic failure
        "fail", "failed", "failure", "failures", "error", "errored",
        "errors", "exception", "fatal", "crash", "crashed", "panic",
        # partial / incomplete
        "partial", "partial_success", "partially_successful",
        "incomplete", "incompleted", "unfinished", "aborted", "abort",
        "terminated", "termination",
        # cancellation / rejection / blocking
        "rejected", "denied", "refused", "blocked", "cancelled",
        "canceled", "cancel", "stopped", "halted", "suspended", "paused",
        # authorization / auth
        "forbidden", "unauthorized", "unauthenticated",
        "unverified", "auth_failed", "authentication_failed",
        # not found / missing / empty
        "not_found", "notfound", "missing", "absent", "gone",
        "unavailable", "not_available", "no_content", "no_data",
        "no_result", "no_results", "empty", "null", "nil", "none",
        "noop", "no_op", "skipped", "skip",
        # time
        "timeout", "timed_out", "timedout", "expired",
        "deadline_exceeded", "deadline_exceeded_error",
        # network / infra
        "unreachable", "offline", "disconnected", "connection_error",
        "connection_refused", "connection_reset", "network_error",
        "service_unavailable", "internal_error", "server_error",
        "bad_gateway", "gateway_timeout", "upstream_error",
        "circuit_open", "circuit_breaker_open",
        # validation
        "invalid", "malformed", "corrupted", "corrupt",
        "unprocessable", "unprocessable_entity", "bad_request",
        "validation_error", "validation_failed", "schema_error",
        "parse_error", "serialization_error", "deserialization_error",
        # conflict / resource
        "conflict", "locked", "busy", "rate_limited", "throttled",
        "quota_exceeded", "resource_exhausted", "overloaded",
        "too_many_requests", "out_of_memory", "oom", "disk_full",
        # health
        "degraded", "unhealthy", "down", "broken", "faulted",
        # explicit negative markers
        "unsuccessful", "not_ok", "not_okay", "not_successful",
        "not_valid", "not_allowed", "no",
    }),
    success_flag_keys=(
        "success", "succeeded", "ok", "okay", "valid", "passed",
        "completed", "complete", "done", "finished",
        "accepted", "allowed", "verified",
    ),
    failure_flag_keys=(
        "failed", "failure", "error", "errored", "errored_out",
        "invalid", "aborted", "cancelled", "canceled",
        "rejected", "denied", "blocked", "forbidden",
        "unauthorized", "timeout", "timed_out",
        "corrupt", "corrupted", "malformed", "degraded",
    ),
    error_carrier_keys=(
        "error", "errors", "exception", "exceptions",
        "error_message", "errmsg", "err_msg",
        "error_type", "error_code", "error_id",
        "traceback", "stack_trace", "stacktrace",
        "stack", "cause", "reason",
    ),
    failure_string_prefixes=(
        "[fallback failure]",
        "[error]", "[failure]", "[failed]", "[fail]",
        "[fatal]", "[panic]", "[crash]", "[aborted]",
        "[timeout]", "[unavailable]", "[rejected]",
        "error:", "failure:", "failed:", "fail:",
        "exception:", "fatal:", "panic:",
        "traceback (most recent call last)",
    ),
    http_status_keys=(
        "status_code", "statuscode", "http_status", "http_status_code",
        "response_code", "code",
    ),
)


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------

__all__ = [
    "FailureTaxonomy",
    "DEFAULT_FAILURE_TAXONOMY",
    "load_failure_taxonomy_config",
    "default_taxonomy_with_config",
]