"""Immutable, reusable success taxonomy.

The BaseAgent runtime consumes a ``SuccessTaxonomy`` to decide whether a
task result positively identifies as successful. It is the mirror of
``failure_taxonomy`` and follows the same three cross-cutting patterns:

* **Error handling** — every public entry point raises
  ``BaseValidationError`` or ``BaseConfigurationError`` on malformed input,
  and config loading failures are translated into structured exceptions.
* **Helper consumption** — text normalization, mapping merging, coercion,
  and JSON-safe repr all come from ``base_helpers``.
* **Logging** — lifecycle events (config load, taxonomy extension, skipped
  inputs) are emitted through the shared ``SLAI Success Taxonomy`` logger.

Success is **not** the strict negation of failure. A result can be neither
success nor failure (an unmarked string, an empty mapping, ``None``). The
taxonomy therefore describes positive signals only; suppression by failure
signals is the responsibility of ``BaseAgent._is_success_result``.
"""

from __future__ import annotations

import time

from dataclasses import dataclass, replace
from typing import Any, Dict, FrozenSet, Iterable, Mapping, Optional, Tuple

from .utils.config_loader import get_config_section
from .utils.base_errors import *
from .utils.base_helpers import *
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("SLAI Success Taxonomy")
printer = PrettyPrinter()

_CONFIG_SECTION = "success_taxonomy"


# ---------------------------------------------------------------------------
# Local normalization (identical contract to failure_taxonomy)
# ---------------------------------------------------------------------------

def _coerce_token_sequence(values: Any, *, field: str) -> Tuple[str, ...]:
    """Validate and normalize an iterable of taxonomy tokens."""
    if values is None:
        return ()
    try:
        sequence = ensure_sequence(values, field, allow_str=True)
    except BaseValidationError:
        raise
    except Exception as exc:
        raise BaseValidationError(
            f"{field} must be an iterable of strings.",
            component="success_taxonomy",
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
    return tuple(dict.fromkeys(cleaned))


# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SuccessTaxonomy:
    """Declarative success vocabulary consumed by ``BaseAgent._is_success_result``.

    Use the ``with_*`` builders to derive an extended taxonomy::

        class SettlementAgent(BaseAgent):
            SUCCESS_TAXONOMY = DEFAULT_SUCCESS_TAXONOMY.with_statuses("settled", "cleared", "shipped")
    """

    success_statuses: FrozenSet[str] = frozenset()
    success_flag_keys: Tuple[str, ...] = ()       # True -> success
    failure_flag_keys: Tuple[str, ...] = ()       # True -> NOT success
    error_carrier_keys: Tuple[str, ...] = ()      # truthy -> NOT success
    success_string_prefixes: Tuple[str, ...] = ()
    http_status_keys: Tuple[str, ...] = ()        # 2xx/3xx -> success
    treat_sequence_successes_as_success: bool = False

    # ----------------------------------------------------------
    # Immutable builders
    # ----------------------------------------------------------

    def with_statuses(self, *statuses: str) -> "SuccessTaxonomy":
        tokens = _coerce_token_sequence(statuses, field="statuses")
        if not tokens:
            logger.warning("with_statuses called with no usable statuses; returning self.")
            return self
        extended = self.success_statuses | frozenset(tokens)
        added = len(extended) - len(self.success_statuses)
        logger.info(
            "Extended success_statuses by %d (now %d total). Added: %s",
            added, len(extended), sorted(set(tokens) - self.success_statuses),
        )
        return replace(self, success_statuses=extended)

    def with_success_flags(self, *keys: str) -> "SuccessTaxonomy":
        cleaned = _clean_flag_keys(keys, field="success_flag_keys")
        if not cleaned:
            logger.warning("with_success_flags called with no usable keys; returning self.")
            return self
        merged = tuple(dict.fromkeys((*self.success_flag_keys, *cleaned)))
        logger.info("Extended success_flag_keys by %d (now %d).", len(merged) - len(self.success_flag_keys), len(merged))
        return replace(self, success_flag_keys=merged)

    def with_failure_flags(self, *keys: str) -> "SuccessTaxonomy":
        cleaned = _clean_flag_keys(keys, field="failure_flag_keys")
        if not cleaned:
            logger.warning("with_failure_flags called with no usable keys; returning self.")
            return self
        merged = tuple(dict.fromkeys((*self.failure_flag_keys, *cleaned)))
        logger.info("Extended failure_flag_keys by %d (now %d).", len(merged) - len(self.failure_flag_keys), len(merged))
        return replace(self, failure_flag_keys=merged)

    def with_error_carriers(self, *keys: str) -> "SuccessTaxonomy":
        cleaned = _clean_flag_keys(keys, field="error_carrier_keys")
        if not cleaned:
            logger.warning("with_error_carriers called with no usable keys; returning self.")
            return self
        merged = tuple(dict.fromkeys((*self.error_carrier_keys, *cleaned)))
        logger.info("Extended error_carrier_keys by %d (now %d).", len(merged) - len(self.error_carrier_keys), len(merged))
        return replace(self, error_carrier_keys=merged)

    def with_string_prefixes(self, *prefixes: str) -> "SuccessTaxonomy":
        cleaned = _clean_string_prefixes(prefixes)
        if not cleaned:
            logger.warning("with_string_prefixes called with no usable prefixes; returning self.")
            return self
        merged = tuple(dict.fromkeys((*self.success_string_prefixes, *cleaned)))
        logger.info("Extended success_string_prefixes by %d (now %d).", len(merged) - len(self.success_string_prefixes), len(merged))
        return replace(self, success_string_prefixes=merged)

    def with_http_status_keys(self, *keys: str) -> "SuccessTaxonomy":
        cleaned = _clean_flag_keys(keys, field="http_status_keys")
        if not cleaned:
            logger.warning("with_http_status_keys called with no usable keys; returning self.")
            return self
        merged = tuple(dict.fromkeys((*self.http_status_keys, *cleaned)))
        logger.info("Extended http_status_keys by %d (now %d).", len(merged) - len(self.http_status_keys), len(merged))
        return replace(self, http_status_keys=merged)

    def with_sequence_successes(self, enabled: bool = True) -> "SuccessTaxonomy":
        flag = coerce_bool(enabled, default=False)
        if flag == self.treat_sequence_successes_as_success:
            logger.debug("with_sequence_successes: no change (already %s).", flag)
            return self
        logger.info("treat_sequence_successes_as_success -> %s", flag)
        return replace(self, treat_sequence_successes_as_success=flag)

    # ----------------------------------------------------------
    # Construction from config
    # ----------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        base: Optional["SuccessTaxonomy"] = None,
        *,
        section: Optional[Mapping[str, Any]] = None,
    ) -> "SuccessTaxonomy":
        """Derive a taxonomy by layering an optional config section over ``base``.

        Config shape (all keys optional)::

            success_taxonomy:
              treat_sequence_successes_as_success: false
              extra_success_statuses:      [settled, cleared]
              extra_success_flag_keys:     [healthy]
              extra_failure_flag_keys:     [underwater]
              extra_error_carrier_keys:    [settlement_error]
              extra_string_prefixes:       ["[settled]"]
              extra_http_status_keys:      [result_code]
        """
        taxonomy = base if base is not None else DEFAULT_SUCCESS_TAXONOMY

        if section is None:
            section = load_success_taxonomy_config()
        else:
            try:
                section = ensure_mapping(section, "success_taxonomy_section")
            except BaseValidationError:
                logger.warning(
                    "from_config: section is not a mapping (%s); ignoring.",
                    type(section).__name__,
                )
                return taxonomy

        if not section:
            logger.debug("from_config: empty section; returning base taxonomy.")
            return taxonomy

        if "treat_sequence_successes_as_success" in section:
            taxonomy = taxonomy.with_sequence_successes(coerce_bool(section["treat_sequence_successes_as_success"], default=False))

        taxonomy = taxonomy.with_statuses(*_section_list(section, "extra_success_statuses"))
        taxonomy = taxonomy.with_success_flags(*_section_list(section, "extra_success_flag_keys"))
        taxonomy = taxonomy.with_failure_flags(*_section_list(section, "extra_failure_flag_keys"))
        taxonomy = taxonomy.with_error_carriers(*_section_list(section, "extra_error_carrier_keys"))
        taxonomy = taxonomy.with_string_prefixes(*_section_list(section, "extra_string_prefixes"))
        taxonomy = taxonomy.with_http_status_keys(*_section_list(section, "extra_http_status_keys"))

        logger.info("SuccessTaxonomy.from_config -> %s", taxonomy.summary())
        return taxonomy

    # ----------------------------------------------------------
    # Queries
    # ----------------------------------------------------------

    @staticmethod
    def normalize_status(raw: Any) -> str:
        return _normalize_token(raw)

    def is_success_status(self, raw: Any) -> bool:
        return _normalize_token(raw) in self.success_statuses

    def summary(self) -> Dict[str, Any]:
        """Compact dict view, useful for logs and diagnostics."""
        return {
            "success_statuses": len(self.success_statuses),
            "success_flag_keys": len(self.success_flag_keys),
            "failure_flag_keys": len(self.failure_flag_keys),
            "error_carrier_keys": len(self.error_carrier_keys),
            "success_string_prefixes": len(self.success_string_prefixes),
            "http_status_keys": len(self.http_status_keys),
            "treat_sequence_successes_as_success": self.treat_sequence_successes_as_success,
        }


# ---------------------------------------------------------------------------
# Token-cleaning helpers
# ---------------------------------------------------------------------------

def _clean_flag_keys(keys: Iterable[str], *, field: str) -> Tuple[str, ...]:
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
    value = section.get(key)
    if value is None:
        return ()
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(value)
    return (value,)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_success_taxonomy_config(*, config: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Load the ``success_taxonomy`` section with full error handling.

    Never raises for a *missing* section — returns ``{}`` instead. Raises
    ``BaseConfigurationError`` only when the section exists but cannot be
    interpreted as a mapping, or when the underlying loader itself fails.
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
        logger.exception("Success taxonomy config load failed.")
        raise BaseConfigurationError(
            "Unable to load success_taxonomy configuration.",
            component="success_taxonomy",
            details={"section": _CONFIG_SECTION},
            cause=exc,
        ) from exc

    if raw_section in (None, {}):
        logger.debug("No success_taxonomy section present; using defaults.")
        return {}

    if not isinstance(raw_section, Mapping):
        raise BaseConfigurationError(
            "success_taxonomy config section must be a mapping.",
            component="success_taxonomy",
            details={
                "section": _CONFIG_SECTION,
                "received_type": type(raw_section).__name__,
                "received_preview": safe_repr(raw_section),
            },
        )

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    logger.info("Loaded success_taxonomy config in %.2f ms (keys=%s).", elapsed_ms, sorted(raw_section.keys()))
    return dict(raw_section)


def default_taxonomy_with_config(*, config: Optional[Mapping[str, Any]] = None) -> SuccessTaxonomy:
    """Convenience wrapper: default taxonomy + any configured extensions."""
    try:
        section = load_success_taxonomy_config(config=config)
    except BaseConfigurationError:
        logger.warning("Falling back to default taxonomy after config error.")
        return DEFAULT_SUCCESS_TAXONOMY
    return SuccessTaxonomy.from_config(DEFAULT_SUCCESS_TAXONOMY, section=section)


# ---------------------------------------------------------------------------
# The default taxonomy
# ---------------------------------------------------------------------------

DEFAULT_SUCCESS_TAXONOMY = SuccessTaxonomy(
    success_statuses=frozenset({
        # explicit success
        "success", "succeeded", "successful",
        "ok", "okay", "fine",
        # validity / passing
        "valid", "validated", "pass", "passed",
        # completion
        "complete", "completed", "done", "finished",
        # lifecycle / availability
        "ready", "healthy", "available",
        # authorization / approval
        "accepted", "allowed", "approved", "confirmed", "verified",
        # resolution
        "resolved", "settled", "cleared",
        # common CRUD-ish affirmative markers
        "created", "updated", "retrieved", "matched",
    }),
    success_flag_keys=(
        "success", "succeeded", "ok", "okay", "valid", "passed",
        "completed", "complete", "done", "finished",
        "accepted", "allowed", "verified", "approved", "confirmed",
        "healthy",
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
        "stack", "cause",
    ),
    success_string_prefixes=(
        "[success]", "[ok]", "[done]", "[complete]", "[passed]",
        "[ready]", "[healthy]", "[accepted]", "[verified]",
        "success:", "ok:", "done:", "completed:", "passed:",
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
    "SuccessTaxonomy",
    "DEFAULT_SUCCESS_TAXONOMY",
    "load_success_taxonomy_config",
    "default_taxonomy_with_config",
]