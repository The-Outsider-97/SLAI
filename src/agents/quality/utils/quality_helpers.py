"""
Central helper primitives for the SLAI Data Quality Agent subsystem.

Location
--------
    src/agents/quality/utils/quality_helper.py

Purpose
-------
This module centralizes cross-cutting helper logic shared by:

- ``src/agents/quality_agent.py``
- ``src/agents/quality/structural_quality.py``
- ``src/agents/quality/statistical_quality.py``
- ``src/agents/quality/semantic_quality.py``
- ``src/agents/quality/workflow_control.py``
- ``src/agents/quality/quality_memory.py``

The module deliberately does *not* own structural, statistical, semantic,
workflow, memory, or orchestration policy. Those responsibilities remain with
their respective components. It provides stable primitives for:

- validation and coercion,
- score / threshold / weight handling,
- verdict and severity normalization,
- deterministic serialization and hashing,
- record and schema normalization,
- quality-finding normalization and aggregation,
- error reconstruction,
- shared-memory compatibility adapters,
- runtime-bridge resolution.

Circular-import rule
--------------------
``quality_helper`` may import ``quality_error`` because ``quality_error`` is a
lower-level utility and does not import this module. This module must never
import the quality agent, quality memory, workflow control, or any quality
checker.

Design principles
-----------------
1. Pure functions where practical.
2. Deterministic output for hashing/persistence.
3. Fail explicitly on malformed quality contracts.
4. Preserve caller-owned domain logic.
5. Avoid hidden global mutable state.
6. Keep compatibility with Python 3.10+ used by SLAI.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import re
import time
import uuid

from copy import deepcopy
from dataclasses import asdict, dataclass, is_dataclass
from datetime import date, datetime, time as datetime_time, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Type, TypeVar

from .quality_error import *


QUALITY_VERDICTS: Tuple[str, ...] = ("pass", "warn", "block")
QUALITY_SEVERITIES: Tuple[str, ...] = ("low", "medium", "high", "critical")

VERDICT_RANK: Dict[str, int] = {
    "pass": 0,
    "warn": 1,
    "block": 2,
}

SEVERITY_RANK: Dict[str, int] = {
    "low": 10,
    "medium": 20,
    "high": 30,
    "critical": 40,
}

DEFAULT_VERDICT_SCORES: Dict[str, float] = {
    "pass": 1.0,
    "warn": 0.75,
    "block": 0.0,
}

_TRUE_VALUES = frozenset(
    {
        "1",
        "true",
        "yes",
        "y",
        "on",
        "enabled",
        "enable",
    }
)
_FALSE_VALUES = frozenset(
    {
        "0",
        "false",
        "no",
        "n",
        "off",
        "disabled",
        "disable",
    }
)
_IDENTIFIER_PATTERN = re.compile(r"[^a-zA-Z0-9_.-]+")
_DEFAULT_MAX_JSON_DEPTH = 12
_DEFAULT_ENCODING = "utf-8"

_E = TypeVar("_E", bound=Enum)


# ---------------------------------------------------------------------------
# Internal error construction
# ---------------------------------------------------------------------------

def _make_quality_error(
    message: str,
    *,
    error_type: QualityErrorType = QualityErrorType.CONFIGURATION_INVALID,
    severity: QualitySeverity = QualitySeverity.MEDIUM,
    stage: QualityStage = QualityStage.VALIDATION,
    domain: QualityDomain = QualityDomain.SYSTEM,
    disposition: QualityDisposition = QualityDisposition.WARN,
    retryable: bool = False,
    remediation: Optional[str] = None,
    context: Optional[Mapping[str, Any]] = None,
    cause: Optional[BaseException] = None,
) -> DataQualityError:
    """Create the subsystem's canonical structured validation error."""
    return DataQualityError(
        message=message,
        error_type=error_type,
        severity=severity,
        retryable=retryable,
        stage=stage,
        domain=domain,
        disposition=disposition,
        remediation=remediation,
        context=dict(context or {}),
        cause=cause,
    )


def _coerce_enum(value: Any, enum_cls: Type[_E], default: _E) -> _E:
    """Best-effort enum reconstruction used only for serialized error payloads."""
    if isinstance(value, enum_cls):
        return value
    if isinstance(value, Enum):
        value = value.value
    if isinstance(value, str):
        candidate = value.strip()
        if candidate:
            try:
                return enum_cls(candidate)
            except ValueError:
                pass
    return default


# ---------------------------------------------------------------------------
# Stable threshold contract
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class QualityThresholds:
    """Validated pass/warn threshold pair for score-based quality decisions.

    The score convention used by the quality subsystem is monotonic:
    larger values are better.

    ``score >= pass_threshold`` -> ``pass``
    ``warn_threshold <= score < pass_threshold`` -> ``warn``
    ``score < warn_threshold`` -> ``block``
    """

    pass_threshold: float = 0.90
    warn_threshold: float = 0.75

    def __post_init__(self) -> None:
        pass_value = bounded_score(self.pass_threshold, field_name="pass_threshold")
        warn_value = bounded_score(self.warn_threshold, field_name="warn_threshold")
        if warn_value > pass_value:
            raise _make_quality_error(
                "warn_threshold must be less than or equal to pass_threshold",
                error_type=QualityErrorType.POLICY_THRESHOLD_INVALID,
                severity=QualitySeverity.HIGH,
                stage=QualityStage.SCORING,
                disposition=QualityDisposition.ESCALATE,
                remediation=(
                    "Set score thresholds so warn_threshold <= pass_threshold."
                ),
                context={
                    "warn_threshold": warn_value,
                    "pass_threshold": pass_value,
                },
            )
        object.__setattr__(self, "pass_threshold", pass_value)
        object.__setattr__(self, "warn_threshold", warn_value)

    def verdict(self, score: Any) -> str:
        return verdict_from_score(
            score,
            pass_threshold=self.pass_threshold,
            warn_threshold=self.warn_threshold,
        )


# ---------------------------------------------------------------------------
# Time and identifiers
# ---------------------------------------------------------------------------

def utc_now() -> datetime:
    """Return a timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


def utc_timestamp() -> float:
    """Return a POSIX UTC timestamp."""
    return time.time()


def utc_iso_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return utc_now().isoformat()


def generate_quality_id(prefix: str) -> str:
    """Generate a collision-resistant, log-friendly quality identifier.

    Format:
        ``<normalized_prefix>_<epoch_ms>_<uuid8>``
    """
    normalized_prefix = _IDENTIFIER_PATTERN.sub(
        "_",
        nonempty_text(prefix, "prefix"),
    ).strip("_.-")
    normalized_prefix = normalized_prefix or "quality"
    return (
        f"{normalized_prefix}_{int(time.time() * 1000)}_"
        f"{uuid.uuid4().hex[:8]}"
    )


# ---------------------------------------------------------------------------
# Serialization and deterministic hashing
# ---------------------------------------------------------------------------

def safe_deepcopy(value: Any) -> Any:
    """Deep-copy when possible, otherwise return a JSON-safe representation."""
    try:
        return deepcopy(value)
    except Exception:
        return json_safe(value)


def json_safe(value: Any, *, max_depth: int = _DEFAULT_MAX_JSON_DEPTH, _depth: int = 0) -> Any:
    """Convert arbitrary values into deterministic JSON-compatible structures.

    The function is intentionally conservative. Unsupported runtime objects are
    reduced to ``repr(value)`` rather than retaining live references.
    """
    if max_depth < 1:
        raise ValueError("max_depth must be >= 1")

    if _depth >= max_depth:
        return repr(value)

    if value is None or isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        if math.isfinite(value):
            return value
        # JSON interoperability is safer when non-finite values are textual.
        return str(value)

    if isinstance(value, Enum):
        return json_safe(value.value, max_depth=max_depth, _depth=_depth + 1)

    if isinstance(value, (datetime, date, datetime_time)):
        return value.isoformat()

    if isinstance(value, timedelta):
        return value.total_seconds()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        try:
            return raw.decode(_DEFAULT_ENCODING)
        except UnicodeDecodeError:
            return {
                "encoding": "base64",
                "length": len(raw),
                "data": base64.b64encode(raw).decode("ascii"),
            }

    if isinstance(value, DataQualityError):
        return value.to_dict(include_traceback=False)

    if isinstance(value, BaseException):
        return {
            "type": type(value).__name__,
            "message": str(value),
        }

    if is_dataclass(value) and not isinstance(value, type):
        try:
            return json_safe(
                asdict(value),
                max_depth=max_depth,
                _depth=_depth + 1,
            )
        except Exception:
            return repr(value)

    if isinstance(value, Mapping):
        return {
            str(key): json_safe(
                item,
                max_depth=max_depth,
                _depth=_depth + 1,
            )
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set, frozenset)):
        items = [
            json_safe(item, max_depth=max_depth, _depth=_depth + 1)
            for item in value
        ]
        if isinstance(value, (set, frozenset)):
            # Deterministic order for hashing/persistence.
            return sorted(items, key=lambda item: repr(item))
        return items

    # NumPy / Torch scalar or array compatibility without importing either.
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return json_safe(
                tolist(),
                max_depth=max_depth,
                _depth=_depth + 1,
            )
        except Exception:
            pass

    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            return json_safe(
                item_method(),
                max_depth=max_depth,
                _depth=_depth + 1,
            )
        except Exception:
            pass

    return repr(value)


def stable_json_dumps(value: Any, *, ensure_ascii: bool = False, sort_keys: bool = True) -> str:
    """Serialize a value deterministically for hashing and persistence."""
    try:
        return json.dumps(
            json_safe(value),
            ensure_ascii=ensure_ascii,
            sort_keys=sort_keys,
            separators=(",", ":"),
        )
    except Exception as exc:
        raise _make_quality_error(
            "Failed to serialize quality payload to deterministic JSON",
            error_type=QualityErrorType.INTERNAL_QUALITY_AGENT_FAILURE,
            severity=QualitySeverity.MEDIUM,
            stage=QualityStage.PERSISTENCE,
            disposition=QualityDisposition.WARN,
            remediation="Inspect the payload for unsupported runtime values.",
            context={"value_type": type(value).__name__},
            cause=exc,
        ) from exc


def stable_hash(value: Any, *, algorithm: str = "sha256") -> str:
    """Return a deterministic digest for an arbitrary quality payload."""
    algorithm_name = nonempty_text(algorithm, "algorithm").lower()
    try:
        digest = hashlib.new(algorithm_name)
    except (TypeError, ValueError) as exc:
        raise _make_quality_error(
            f"Unsupported hash algorithm '{algorithm_name}'",
            error_type=QualityErrorType.CONFIGURATION_INVALID,
            severity=QualitySeverity.MEDIUM,
            stage=QualityStage.VALIDATION,
            disposition=QualityDisposition.WARN,
            remediation="Use a hash algorithm exposed by hashlib.",
            context={"algorithm": algorithm_name},
            cause=exc,
        ) from exc

    digest.update(stable_json_dumps(value).encode(_DEFAULT_ENCODING))
    return digest.hexdigest()


def normalized_mapping(
    value: Optional[Mapping[str, Any]],
    *,
    field_name: str = "mapping",
    allow_none: bool = True,
) -> Dict[str, Any]:
    """Normalize a mapping into string-keyed, JSON-safe form."""
    if value is None:
        if allow_none:
            return {}
        raise _make_quality_error(
            f"{field_name} must be a mapping and cannot be null",
            context={"field_name": field_name},
        )

    if not isinstance(value, Mapping):
        raise _make_quality_error(
            f"{field_name} must be a mapping",
            context={
                "field_name": field_name,
                "actual_type": type(value).__name__,
            },
        )

    return {
        str(key): json_safe(item)
        for key, item in value.items()
    }


# ---------------------------------------------------------------------------
# Scalar and collection validation
# ---------------------------------------------------------------------------

def clamp(value: Any, minimum: float, maximum: float) -> float:
    """Clamp a numeric value to a closed interval."""
    try:
        numeric = float(value)
        lower = float(minimum)
        upper = float(maximum)
    except (TypeError, ValueError) as exc:
        raise _make_quality_error(
            "Clamp arguments must be numeric",
            error_type=QualityErrorType.POLICY_THRESHOLD_INVALID,
            stage=QualityStage.SCORING,
            context={
                "value": value,
                "minimum": minimum,
                "maximum": maximum,
            },
            cause=exc,
        ) from exc

    if not all(math.isfinite(item) for item in (numeric, lower, upper)):
        raise _make_quality_error(
            "Clamp arguments must be finite numeric values",
            error_type=QualityErrorType.POLICY_THRESHOLD_INVALID,
            stage=QualityStage.SCORING,
            context={
                "value": value,
                "minimum": minimum,
                "maximum": maximum,
            },
        )

    if lower > upper:
        raise _make_quality_error(
            "minimum must be less than or equal to maximum",
            error_type=QualityErrorType.POLICY_THRESHOLD_INVALID,
            stage=QualityStage.SCORING,
            context={"minimum": lower, "maximum": upper},
        )

    return max(lower, min(upper, numeric))


def bounded_score(value: Any, *, field_name: str = "score") -> float:
    """Validate a finite score in the canonical [0.0, 1.0] interval."""
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise _make_quality_error(
            f"{field_name} must be numeric",
            error_type=QualityErrorType.POLICY_THRESHOLD_INVALID,
            stage=QualityStage.SCORING,
            context={"field_name": field_name, "value": value},
            remediation="Provide a numeric score within [0.0, 1.0].",
            cause=exc,
        ) from exc

    if not math.isfinite(score) or score < 0.0 or score > 1.0:
        raise _make_quality_error(
            f"{field_name} must be a finite value within [0.0, 1.0]",
            error_type=QualityErrorType.POLICY_THRESHOLD_INVALID,
            stage=QualityStage.SCORING,
            context={"field_name": field_name, "value": value},
            remediation="Adjust the score so it remains within [0.0, 1.0].",
        )
    return score


def positive_int(value: Any, field_name: str = "value") -> int:
    """Coerce a strictly positive integer."""
    if isinstance(value, bool):
        raise _make_quality_error(
            f"{field_name} must be a positive integer",
            context={"field_name": field_name, "value": value},
        )
    try:
        resolved = int(value)
    except (TypeError, ValueError) as exc:
        raise _make_quality_error(
            f"{field_name} must be a positive integer",
            context={"field_name": field_name, "value": value},
            cause=exc,
        ) from exc
    if resolved <= 0:
        raise _make_quality_error(
            f"{field_name} must be greater than zero",
            context={"field_name": field_name, "value": value},
        )
    return resolved


def nonnegative_int(value: Any, field_name: str = "value") -> int:
    """Coerce a non-negative integer."""
    if isinstance(value, bool):
        raise _make_quality_error(
            f"{field_name} must be a non-negative integer",
            context={"field_name": field_name, "value": value},
        )
    try:
        resolved = int(value)
    except (TypeError, ValueError) as exc:
        raise _make_quality_error(
            f"{field_name} must be a non-negative integer",
            context={"field_name": field_name, "value": value},
            cause=exc,
        ) from exc
    if resolved < 0:
        raise _make_quality_error(
            f"{field_name} must be greater than or equal to zero",
            context={"field_name": field_name, "value": value},
        )
    return resolved


def optional_nonnegative_int(value: Any, field_name: str = "value") -> Optional[int]:
    """Return ``None`` or a validated non-negative integer."""
    if value is None:
        return None
    return nonnegative_int(value, field_name)


def coerce_bool(
    value: Any,
    *,
    field_name: str = "value",
    default: Optional[bool] = None,
) -> bool:
    """Strictly coerce common boolean representations.

    Unlike ``bool("false")``, this function correctly maps ``"false"`` to
    ``False`` and rejects ambiguous values.
    """
    if value is None:
        if default is not None:
            return bool(default)
        raise _make_quality_error(
            f"{field_name} must be boolean",
            context={"field_name": field_name, "value": value},
        )

    if isinstance(value, bool):
        return value

    if isinstance(value, int) and value in (0, 1):
        return bool(value)

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUE_VALUES:
            return True
        if normalized in _FALSE_VALUES:
            return False

    raise _make_quality_error(
        f"{field_name} must be a recognized boolean value",
        context={"field_name": field_name, "value": value},
        remediation=(
            "Use true/false, 1/0, yes/no, on/off, or an actual boolean."
        ),
    )


def nonempty_text(value: Any, field_name: str = "value") -> str:
    """Return a stripped non-empty string."""
    text = "" if value is None else str(value).strip()
    if not text:
        raise _make_quality_error(
            f"{field_name} must not be empty",
            context={"field_name": field_name},
        )
    return text


def string_list(
    values: Any,
    *,
    drop_empty: bool = True,
    deduplicate: bool = False,
) -> List[str]:
    """Normalize scalar/iterable input to a list of strings.

    Strings are treated as one item rather than iterated character-by-character.
    """
    if values is None:
        return []

    if isinstance(values, (str, bytes, bytearray)):
        raw_values: Iterable[Any] = [values]
    elif isinstance(values, Mapping):
        # Mapping keys are normally what configuration callers mean here.
        raw_values = values.keys()
    else:
        try:
            raw_values = iter(values)
        except TypeError:
            raw_values = [values]

    result: List[str] = []
    seen = set()

    for item in raw_values:
        if isinstance(item, bytes):
            text = item.decode(_DEFAULT_ENCODING, errors="replace").strip()
        elif isinstance(item, bytearray):
            text = bytes(item).decode(
                _DEFAULT_ENCODING,
                errors="replace",
            ).strip()
        else:
            text = str(item).strip()

        if drop_empty and not text:
            continue
        if deduplicate:
            if text in seen:
                continue
            seen.add(text)
        result.append(text)

    return result


def merge_unique_strings(*groups: Any) -> List[str]:
    """Merge string groups while preserving first-seen order."""
    merged: List[str] = []
    seen = set()
    for group in groups:
        for item in string_list(group):
            if item in seen:
                continue
            seen.add(item)
            merged.append(item)
    return merged


def float_mapping(value: Any, *, field_name: str = "mapping", finite_only: bool = True) -> Dict[str, float]:
    """Normalize a mapping into ``str -> float``."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise _make_quality_error(
            f"{field_name} must be a mapping",
            context={
                "field_name": field_name,
                "actual_type": type(value).__name__,
            },
        )

    result: Dict[str, float] = {}
    for key, item in value.items():
        try:
            numeric = float(item)
        except (TypeError, ValueError) as exc:
            raise _make_quality_error(
                f"{field_name}.{key} must be numeric",
                context={
                    "field_name": field_name,
                    "key": str(key),
                    "value": item,
                },
                cause=exc,
            ) from exc
        if finite_only and not math.isfinite(numeric):
            raise _make_quality_error(
                f"{field_name}.{key} must be finite",
                context={
                    "field_name": field_name,
                    "key": str(key),
                    "value": item,
                },
            )
        result[str(key)] = numeric
    return result


# ---------------------------------------------------------------------------
# Verdict, severity, weighting, and score aggregation
# ---------------------------------------------------------------------------

def normalize_verdict(verdict: Any, *, default: Optional[str] = None) -> str:
    """Normalize the canonical quality verdict: pass / warn / block."""
    if isinstance(verdict, Enum):
        verdict = verdict.value
    value = "" if verdict is None else str(verdict).strip().lower()
    if not value and default is not None:
        value = str(default).strip().lower()
    if value not in QUALITY_VERDICTS:
        raise _make_quality_error(
            f"Unsupported quality verdict '{verdict}'",
            error_type=QualityErrorType.CONFIGURATION_INVALID,
            stage=QualityStage.ROUTING,
            context={
                "verdict": verdict,
                "supported": list(QUALITY_VERDICTS),
            },
            remediation="Use one of: pass, warn, block.",
        )
    return value


def normalize_severity(severity: Any, *, default: str = "medium", strict: bool = False) -> str:
    """Normalize quality severity.

    ``strict=False`` preserves the quality stack's existing behavior of
    defaulting unsupported severities to ``medium``.
    """
    if isinstance(severity, Enum):
        severity = severity.value
    value = "" if severity is None else str(severity).strip().lower()
    if value in QUALITY_SEVERITIES:
        return value

    default_value = str(default).strip().lower()
    if default_value not in QUALITY_SEVERITIES:
        default_value = "medium"

    if strict:
        raise _make_quality_error(
            f"Unsupported quality severity '{severity}'",
            context={
                "severity": severity,
                "supported": list(QUALITY_SEVERITIES),
            },
        )
    return default_value


def score_from_verdict(
    verdict: Any,
    *,
    scores: Optional[Mapping[str, Any]] = None,
) -> float:
    """Map a canonical verdict to a bounded representative score."""
    normalized = normalize_verdict(verdict)
    mapping = dict(DEFAULT_VERDICT_SCORES)
    if scores is not None:
        for key, value in scores.items():
            mapping[normalize_verdict(key)] = bounded_score(
                value,
                field_name=f"verdict_scores.{key}",
            )
    return bounded_score(
        mapping[normalized],
        field_name=f"verdict_scores.{normalized}",
    )


def verdict_from_score(
    score: Any,
    *,
    pass_threshold: Any = 0.90,
    warn_threshold: Any = 0.75,
) -> str:
    """Convert a score to pass / warn / block using validated thresholds."""
    numeric = bounded_score(score, field_name="score")
    passing = bounded_score(
        pass_threshold,
        field_name="pass_threshold",
    )
    warning = bounded_score(
        warn_threshold,
        field_name="warn_threshold",
    )
    if warning > passing:
        raise _make_quality_error(
            "warn_threshold must be <= pass_threshold",
            error_type=QualityErrorType.POLICY_THRESHOLD_INVALID,
            severity=QualitySeverity.HIGH,
            stage=QualityStage.SCORING,
            disposition=QualityDisposition.ESCALATE,
            context={
                "warn_threshold": warning,
                "pass_threshold": passing,
            },
        )
    if numeric >= passing:
        return "pass"
    if numeric >= warning:
        return "warn"
    return "block"


def worst_verdict(*verdicts: Any, default: str = "pass") -> str:
    """Return the most restrictive verdict from the supplied values."""
    normalized = [
        normalize_verdict(item)
        for item in verdicts
        if item is not None and str(item).strip()
    ]
    if not normalized:
        return normalize_verdict(default)
    return max(normalized, key=lambda item: VERDICT_RANK[item])


def worst_severity(
    *severities: Any,
    default: str = "low",
) -> str:
    """Return the highest severity from the supplied values."""
    normalized = [
        normalize_severity(item)
        for item in severities
        if item is not None and str(item).strip()
    ]
    if not normalized:
        return normalize_severity(default)
    return max(normalized, key=lambda item: SEVERITY_RANK[item])


def normalize_weights(
    weights: Optional[Mapping[str, Any]],
    *,
    expected_keys: Optional[Sequence[str]] = None,
    defaults: Optional[Mapping[str, Any]] = None,
    field_name: str = "weights",
    reject_unknown: bool = False,
) -> Dict[str, float]:
    """Validate and normalize a non-negative weight mapping to sum to 1.0.

    ``expected_keys`` provides deterministic ordering and can be used with
    ``defaults`` to fill omitted keys. Missing expected keys without defaults
    are rejected rather than silently receiving invented weights.
    """
    raw = dict(weights or {})
    default_map = dict(defaults or {})

    if expected_keys is None:
        keys = [str(key) for key in raw.keys()]
        if not keys and default_map:
            keys = [str(key) for key in default_map.keys()]
    else:
        keys = [nonempty_text(key, f"{field_name}.key") for key in expected_keys]

    if not keys:
        raise _make_quality_error(
            f"{field_name} must contain at least one weight",
            error_type=QualityErrorType.POLICY_THRESHOLD_INVALID,
            severity=QualitySeverity.HIGH,
            stage=QualityStage.SCORING,
            disposition=QualityDisposition.ESCALATE,
            context={"field_name": field_name},
        )

    if reject_unknown:
        unknown = sorted(set(str(key) for key in raw) - set(keys))
        if unknown:
            raise _make_quality_error(
                f"{field_name} contains unsupported weight keys",
                error_type=QualityErrorType.CONFIGURATION_INVALID,
                context={
                    "field_name": field_name,
                    "unknown_keys": unknown,
                    "expected_keys": keys,
                },
            )

    normalized: Dict[str, float] = {}
    total = 0.0

    for key in keys:
        if key in raw:
            source_value = raw[key]
        elif key in default_map:
            source_value = default_map[key]
        else:
            raise _make_quality_error(
                f"{field_name} is missing required weight '{key}'",
                error_type=QualityErrorType.CONFIGURATION_INVALID,
                severity=QualitySeverity.HIGH,
                stage=QualityStage.SCORING,
                disposition=QualityDisposition.ESCALATE,
                context={
                    "field_name": field_name,
                    "missing_key": key,
                },
            )

        try:
            numeric = float(source_value)
        except (TypeError, ValueError) as exc:
            raise _make_quality_error(
                f"{field_name}.{key} must be numeric",
                error_type=QualityErrorType.POLICY_THRESHOLD_INVALID,
                stage=QualityStage.SCORING,
                context={
                    "field_name": field_name,
                    "key": key,
                    "value": source_value,
                },
                cause=exc,
            ) from exc

        if not math.isfinite(numeric) or numeric < 0.0:
            raise _make_quality_error(
                f"{field_name}.{key} must be finite and non-negative",
                error_type=QualityErrorType.POLICY_THRESHOLD_INVALID,
                stage=QualityStage.SCORING,
                context={
                    "field_name": field_name,
                    "key": key,
                    "value": source_value,
                },
            )

        normalized[key] = numeric
        total += numeric

    if total <= 0.0:
        raise _make_quality_error(
            f"{field_name} must contain at least one positive weight",
            error_type=QualityErrorType.POLICY_THRESHOLD_INVALID,
            severity=QualitySeverity.HIGH,
            stage=QualityStage.SCORING,
            disposition=QualityDisposition.ESCALATE,
            context={"field_name": field_name},
        )

    return {
        key: numeric / total
        for key, numeric in normalized.items()
    }


def weighted_score(
    scores: Mapping[str, Any],
    weights: Mapping[str, Any],
    *,
    default: float = 0.0,
    ignore_missing: bool = True,
) -> float:
    """Compute a bounded weighted mean from named quality scores.

    If ``ignore_missing`` is true, weights are renormalized over available
    score keys. This is useful when a subsystem is disabled or intentionally
    skipped.
    """
    score_map = {
        str(key): bounded_score(
            value,
            field_name=f"scores.{key}",
        )
        for key, value in dict(scores or {}).items()
    }
    weight_map = float_mapping(weights, field_name="weights")

    active: List[Tuple[float, float]] = []
    for key, raw_weight in weight_map.items():
        if raw_weight < 0.0:
            raise _make_quality_error(
                f"weights.{key} must be non-negative",
                error_type=QualityErrorType.POLICY_THRESHOLD_INVALID,
                stage=QualityStage.SCORING,
                context={"key": key, "value": raw_weight},
            )
        if key not in score_map:
            if ignore_missing:
                continue
            raise _make_quality_error(
                f"Missing score for weighted key '{key}'",
                error_type=QualityErrorType.SCORING_PIPELINE_FAILED,
                stage=QualityStage.SCORING,
                context={"missing_key": key},
            )
        active.append((score_map[key], raw_weight))

    if not active:
        return bounded_score(default, field_name="default")

    denominator = sum(weight for _, weight in active)
    if denominator <= 0.0:
        raise _make_quality_error(
            "Active quality weights must sum to a positive value",
            error_type=QualityErrorType.POLICY_THRESHOLD_INVALID,
            stage=QualityStage.SCORING,
        )

    return clamp(
        sum(score * weight for score, weight in active) / denominator,
        0.0,
        1.0,
    )


# ---------------------------------------------------------------------------
# Record and schema normalization
# ---------------------------------------------------------------------------

def normalize_records(
    records: Any,
    *,
    field_name: str = "records",
    require_nonempty: bool = False,
) -> List[Dict[str, Any]]:
    """Normalize a sequence of records to string-keyed dictionaries."""
    if records is None:
        if require_nonempty:
            raise _make_quality_error(
                f"{field_name} must contain at least one record",
                error_type=QualityErrorType.SCHEMA_VALIDATION_FAILED,
                severity=QualitySeverity.HIGH,
                stage=QualityStage.VALIDATION,
                disposition=QualityDisposition.BLOCK,
                context={"field_name": field_name},
            )
        return []

    if isinstance(records, Mapping):
        raise _make_quality_error(
            f"{field_name} must be a sequence of record mappings",
            error_type=QualityErrorType.SCHEMA_VALIDATION_FAILED,
            severity=QualitySeverity.HIGH,
            stage=QualityStage.VALIDATION,
            disposition=QualityDisposition.BLOCK,
            context={
                "field_name": field_name,
                "actual_type": type(records).__name__,
            },
        )

    if isinstance(records, (str, bytes, bytearray)):
        raise _make_quality_error(
            f"{field_name} must be a sequence of mappings, not text/bytes",
            error_type=QualityErrorType.SCHEMA_VALIDATION_FAILED,
            severity=QualitySeverity.HIGH,
            stage=QualityStage.VALIDATION,
            disposition=QualityDisposition.BLOCK,
            context={
                "field_name": field_name,
                "actual_type": type(records).__name__,
            },
        )

    try:
        raw_records = list(records)
    except TypeError as exc:
        raise _make_quality_error(
            f"{field_name} must be iterable",
            error_type=QualityErrorType.SCHEMA_VALIDATION_FAILED,
            severity=QualitySeverity.HIGH,
            stage=QualityStage.VALIDATION,
            disposition=QualityDisposition.BLOCK,
            context={
                "field_name": field_name,
                "actual_type": type(records).__name__,
            },
            cause=exc,
        ) from exc

    normalized: List[Dict[str, Any]] = []
    for index, record in enumerate(raw_records):
        if not isinstance(record, Mapping):
            raise _make_quality_error(
                "Each quality record must be a mapping",
                error_type=QualityErrorType.SCHEMA_VALIDATION_FAILED,
                severity=QualitySeverity.HIGH,
                stage=QualityStage.VALIDATION,
                disposition=QualityDisposition.BLOCK,
                context={
                    "field_name": field_name,
                    "record_index": index,
                    "record_type": type(record).__name__,
                },
                remediation=(
                    "Normalize upstream records into dictionaries before "
                    "invoking the quality gate."
                ),
            )
        normalized.append(
            {
                str(key): json_safe(value)
                for key, value in record.items()
            }
        )

    if require_nonempty and not normalized:
        raise _make_quality_error(
            f"{field_name} must contain at least one record",
            error_type=QualityErrorType.SCHEMA_VALIDATION_FAILED,
            severity=QualitySeverity.HIGH,
            stage=QualityStage.VALIDATION,
            disposition=QualityDisposition.BLOCK,
            context={"field_name": field_name},
        )

    return normalized


def resolve_record_id(
    record: Mapping[str, Any],
    index: int,
    *,
    candidates: Sequence[str] = (
        "record_id",
        "id",
        "row_id",
        "sample_id",
        "uuid",
    ),
    fallback_prefix: str = "record",
) -> str:
    """Resolve a stable record identifier from common candidate fields."""
    if not isinstance(record, Mapping):
        raise _make_quality_error(
            "record must be a mapping",
            error_type=QualityErrorType.SCHEMA_VALIDATION_FAILED,
            context={"record_type": type(record).__name__},
        )

    for key in candidates:
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()

    return f"{nonempty_text(fallback_prefix, 'fallback_prefix')}_{int(index)}"


def record_fingerprint(
    record: Mapping[str, Any],
    *,
    exclude_fields: Optional[Iterable[str]] = None,
) -> str:
    """Return a deterministic SHA-256 fingerprint for a record."""
    excluded = set(string_list(exclude_fields))
    payload = {
        str(key): value
        for key, value in record.items()
        if str(key) not in excluded
    }
    return stable_hash(payload, algorithm="sha256")


def schema_required_fields(schema: Optional[Mapping[str, Any]]) -> List[str]:
    """Extract required field names from supported schema shapes."""
    if not isinstance(schema, Mapping):
        return []

    required = schema.get("required_fields")
    if required is None:
        required = schema.get("required")

    if isinstance(required, Mapping):
        explicit = [
            str(key)
            for key, enabled in required.items()
            if bool(enabled)
        ]
    else:
        explicit = string_list(required, deduplicate=True)

    fields = schema.get("fields")
    if isinstance(fields, Mapping):
        for field_name, field_spec in fields.items():
            if isinstance(field_spec, Mapping) and bool(
                field_spec.get("required", False)
            ):
                explicit.append(str(field_name))

    return merge_unique_strings(explicit)


def schema_version(
    schema: Optional[Mapping[str, Any]],
    fallback: Optional[Mapping[str, Any]] = None,
) -> Optional[str]:
    """Resolve schema-version metadata from explicit schema then fallback."""
    for source in (schema, fallback):
        if not isinstance(source, Mapping):
            continue
        for key in ("schema_version", "version"):
            value = source.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
    return None


def schema_fingerprint(
    schema: Mapping[str, Any],
) -> str:
    """Return a deterministic SHA-256 fingerprint for a schema mapping."""
    if not isinstance(schema, Mapping):
        raise _make_quality_error(
            "schema must be a mapping",
            error_type=QualityErrorType.SCHEMA_VALIDATION_FAILED,
            context={"schema_type": type(schema).__name__},
        )
    return stable_hash(schema, algorithm="sha256")


# ---------------------------------------------------------------------------
# Finding normalization and quality aggregation
# ---------------------------------------------------------------------------

def normalize_finding(
    finding: Mapping[str, Any],
    *,
    default_domain: Optional[str] = None,
    default_checker: Optional[str] = None,
    default_verdict: str = "warn",
    default_severity: str = "medium",
) -> Dict[str, Any]:
    """Normalize a quality finding into the shared cross-subsystem contract."""
    if not isinstance(finding, Mapping):
        raise _make_quality_error(
            "Quality finding must be a mapping",
            error_type=QualityErrorType.SCORING_PIPELINE_FAILED,
            stage=QualityStage.SCORING,
            context={"finding_type": type(finding).__name__},
        )

    payload = {
        str(key): json_safe(value)
        for key, value in finding.items()
    }

    checker = (
        payload.get("checker")
        or payload.get("check_name")
        or payload.get("check")
        or default_checker
        or default_domain
        or "quality"
    )
    domain = (
        payload.get("domain")
        or default_domain
        or "system"
    )

    payload["checker"] = str(checker).strip() or "quality"
    payload["domain"] = str(domain).strip().lower() or "system"
    payload["verdict"] = normalize_verdict(
        payload.get("verdict"),
        default=default_verdict,
    )
    payload["severity"] = normalize_severity(
        payload.get("severity"),
        default=default_severity,
    )
    payload["confidence"] = bounded_score(
        payload.get("confidence", 1.0),
        field_name="finding.confidence",
    )

    if "score" in payload and payload["score"] is not None:
        payload["score"] = bounded_score(
            payload["score"],
            field_name="finding.score",
        )

    payload["flags"] = string_list(
        payload.get("flags"),
        deduplicate=True,
    )
    payload["remediation_actions"] = string_list(
        payload.get("remediation_actions"),
        deduplicate=True,
    )
    payload["affected_records"] = merge_unique_strings(
        payload.get("affected_records"),
        payload.get("affected_record_ids"),
    )

    error_type = payload.get("error_type")
    if isinstance(error_type, Enum):
        payload["error_type"] = str(error_type.value)
    elif error_type is None:
        payload["error_type"] = ""
    else:
        payload["error_type"] = str(error_type)

    if payload.get("message") is not None:
        payload["message"] = str(payload["message"]).strip()

    return payload


def normalize_findings(
    findings: Any,
    *,
    default_domain: Optional[str] = None,
    default_checker: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Normalize an iterable of findings."""
    if findings is None:
        return []
    if isinstance(findings, Mapping):
        raw_findings = [findings]
    else:
        try:
            raw_findings = list(findings)
        except TypeError as exc:
            raise _make_quality_error(
                "findings must be iterable",
                error_type=QualityErrorType.SCORING_PIPELINE_FAILED,
                stage=QualityStage.SCORING,
                context={"findings_type": type(findings).__name__},
                cause=exc,
            ) from exc

    return [
        normalize_finding(
            item,
            default_domain=default_domain,
            default_checker=default_checker,
        )
        for item in raw_findings
    ]


def collect_flags(*finding_groups: Any) -> List[str]:
    """Collect unique flags from findings or result mappings."""
    groups: List[Any] = []
    for group in finding_groups:
        if group is None:
            continue
        if isinstance(group, Mapping):
            if "flags" in group:
                groups.append(group.get("flags"))
            if "findings" in group:
                groups.extend(
                    item.get("flags")
                    for item in group.get("findings", [])
                    if isinstance(item, Mapping)
                )
        else:
            try:
                for item in group:
                    if isinstance(item, Mapping):
                        groups.append(item.get("flags"))
                    else:
                        groups.append(item)
            except TypeError:
                groups.append(group)
    return merge_unique_strings(*groups)


def collect_remediation_actions(*finding_groups: Any) -> List[str]:
    """Collect unique remediation actions from findings or result mappings."""
    groups: List[Any] = []
    for group in finding_groups:
        if group is None:
            continue
        if isinstance(group, Mapping):
            if "remediation_actions" in group:
                groups.append(group.get("remediation_actions"))
            if "findings" in group:
                groups.extend(
                    item.get("remediation_actions")
                    for item in group.get("findings", [])
                    if isinstance(item, Mapping)
                )
        else:
            try:
                for item in group:
                    if isinstance(item, Mapping):
                        groups.append(item.get("remediation_actions"))
                    else:
                        groups.append(item)
            except TypeError:
                groups.append(group)
    return merge_unique_strings(*groups)


def record_level_flags(subsystem_results: Mapping[str, Mapping[str, Any]]) -> Dict[str, List[str]]:
    """Build ``record_id -> scoped flags`` from subsystem findings."""
    result: Dict[str, List[str]] = {}

    for subsystem, subsystem_result in dict(
        subsystem_results or {}
    ).items():
        if not isinstance(subsystem_result, Mapping):
            continue

        for raw_finding in subsystem_result.get("findings", []) or []:
            if not isinstance(raw_finding, Mapping):
                continue

            flags = string_list(raw_finding.get("flags"))
            record_ids = merge_unique_strings(
                raw_finding.get("affected_records"),
                raw_finding.get("affected_record_ids"),
            )

            for record_id in record_ids:
                bucket = result.setdefault(record_id, [])
                for flag in flags:
                    scoped = f"{subsystem}:{flag}"
                    if scoped not in bucket:
                        bucket.append(scoped)

    return result


def aggregate_subsystem_score(
    subsystem_results: Mapping[str, Mapping[str, Any]],
    weights: Mapping[str, Any],
    *,
    ignore_disabled: bool = True,
    default_score: float = 0.0,
) -> float:
    """Compute the weighted aggregate score across quality subsystems."""
    scores: Dict[str, float] = {}

    for name, result in dict(subsystem_results or {}).items():
        if not isinstance(result, Mapping):
            continue
        if ignore_disabled and result.get("enabled") is False:
            continue
        raw_score = result.get("batch_score")
        if raw_score is None:
            raw_score = score_from_verdict(
                result.get("verdict", "warn")
            )
        scores[str(name)] = bounded_score(
            raw_score,
            field_name=f"{name}.batch_score",
        )

    return weighted_score(
        scores,
        weights,
        default=default_score,
        ignore_missing=True,
    )


def aggregate_subsystem_verdict(
    subsystem_results: Mapping[str, Mapping[str, Any]],
    *,
    batch_score: Any,
    pass_threshold: Any = 0.90,
    warn_threshold: Any = 0.75,
    ignore_disabled: bool = True,
) -> str:
    """Aggregate explicit subsystem verdicts with the score-derived verdict.

    The most restrictive explicit subsystem verdict wins over a less restrictive
    aggregate score. This preserves fail-safe semantics without introducing
    subsystem-specific policy into the helper layer.
    """
    verdicts: List[str] = [
        verdict_from_score(
            batch_score,
            pass_threshold=pass_threshold,
            warn_threshold=warn_threshold,
        )
    ]

    for result in dict(subsystem_results or {}).values():
        if not isinstance(result, Mapping):
            continue
        if ignore_disabled and result.get("enabled") is False:
            continue
        verdicts.append(
            normalize_verdict(
                result.get("verdict"),
                default="warn",
            )
        )

    return worst_verdict(*verdicts)


def decision_confidence(
    subsystem_results: Mapping[str, Mapping[str, Any]],
    *,
    pass_threshold: Any = 0.90,
    warn_threshold: Any = 0.75,
    pass_fallback: float = 0.95,
    warn_fallback: float = 0.80,
    block_fallback: float = 0.65,
) -> float:
    """Estimate decision confidence using finding confidence where available.

    For subsystems without findings, the fallback confidence follows the current
    quality-agent convention and is selected from the subsystem batch score.
    """
    thresholds = QualityThresholds(
        pass_threshold=pass_threshold,
        warn_threshold=warn_threshold,
    )
    fallback_map = {
        "pass": bounded_score(
            pass_fallback,
            field_name="pass_fallback",
        ),
        "warn": bounded_score(
            warn_fallback,
            field_name="warn_fallback",
        ),
        "block": bounded_score(
            block_fallback,
            field_name="block_fallback",
        ),
    }

    confidences: List[float] = []

    for result in dict(subsystem_results or {}).values():
        if not isinstance(result, Mapping):
            continue
        if result.get("enabled") is False:
            continue

        findings = result.get("findings") or []
        found_confidence = False

        for finding in findings:
            if not isinstance(finding, Mapping):
                continue
            confidences.append(
                bounded_score(
                    finding.get("confidence", 1.0),
                    field_name="finding.confidence",
                )
            )
            found_confidence = True

        if not found_confidence:
            score = bounded_score(
                result.get(
                    "batch_score",
                    score_from_verdict(
                        result.get("verdict", "warn")
                    ),
                ),
                field_name="subsystem.batch_score",
            )
            confidences.append(
                fallback_map[thresholds.verdict(score)]
            )

    if not confidences:
        return 1.0

    return clamp(
        sum(confidences) / len(confidences),
        0.0,
        1.0,
    )


# ---------------------------------------------------------------------------
# Error payload reconstruction
# ---------------------------------------------------------------------------

def quality_error_from_payload(
    payload: Mapping[str, Any],
    *,
    default_error_type: QualityErrorType = (
        QualityErrorType.INTERNAL_QUALITY_AGENT_FAILURE
    ),
    default_severity: QualitySeverity = QualitySeverity.HIGH,
    default_stage: QualityStage = QualityStage.UNKNOWN,
    default_domain: QualityDomain = QualityDomain.SYSTEM,
    default_disposition: QualityDisposition = QualityDisposition.BLOCK,
) -> DataQualityError:
    """Reconstruct ``DataQualityError`` safely from a serialized payload.

    Invalid enum strings fall back to caller-provided defaults instead of
    raising a secondary ``ValueError`` while error handling is already active.
    """
    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a mapping")

    return DataQualityError(
        message=str(
            payload.get("message")
            or "Quality subsystem failure"
        ),
        error_type=_coerce_enum(
            payload.get("error_type"),
            QualityErrorType,
            default_error_type,
        ),
        severity=_coerce_enum(
            payload.get("severity"),
            QualitySeverity,
            default_severity,
        ),
        retryable=bool(payload.get("retryable", False)),
        context=normalized_mapping(
            payload.get("context")
            if isinstance(payload.get("context"), Mapping)
            else {},
            field_name="error.context",
        ),
        remediation=(
            str(payload["remediation"])
            if payload.get("remediation") is not None
            else None
        ),
        stage=_coerce_enum(
            payload.get("stage"),
            QualityStage,
            default_stage,
        ),
        domain=_coerce_enum(
            payload.get("domain"),
            QualityDomain,
            default_domain,
        ),
        disposition=_coerce_enum(
            payload.get("disposition"),
            QualityDisposition,
            default_disposition,
        ),
        dataset_id=(
            str(payload["dataset_id"])
            if payload.get("dataset_id") is not None
            else None
        ),
        source_id=(
            str(payload["source_id"])
            if payload.get("source_id") is not None
            else None
        ),
        batch_id=(
            str(payload["batch_id"])
            if payload.get("batch_id") is not None
            else None
        ),
        record_id=(
            str(payload["record_id"])
            if payload.get("record_id") is not None
            else None
        ),
        rule_id=(
            str(payload["rule_id"])
            if payload.get("rule_id") is not None
            else None
        ),
        tags=(
            {
                str(key): str(value)
                for key, value in (
                    raw_tags.items()
                    if isinstance(raw_tags, Mapping)
                    else {}
                )
            }
            if isinstance(raw_tags := payload.get("tags"), Mapping)
            else {}
        ),
        correlation_id=str(
            payload.get("correlation_id")
            or uuid.uuid4().hex
        ),
    )


# ---------------------------------------------------------------------------
# Shared-memory compatibility adapters
# ---------------------------------------------------------------------------

def shared_memory_get(shared_memory: Any, key: str, default: Any = None) -> Any:
    """Read a value from dict-like or SLAI shared-memory implementations."""
    if shared_memory is None:
        return default

    normalized_key = nonempty_text(key, "shared_memory.key")

    if isinstance(shared_memory, Mapping):
        return shared_memory.get(normalized_key, default)

    getter = getattr(shared_memory, "get", None)
    if not callable(getter):
        raise _make_quality_error(
            "shared_memory does not expose a compatible get method",
            error_type=QualityErrorType.QUALITY_MEMORY_UNAVAILABLE,
            severity=QualitySeverity.MEDIUM,
            stage=QualityStage.RETRIEVAL,
            domain=QualityDomain.MEMORY,
            disposition=QualityDisposition.FALLBACK,
            retryable=True,
            context={
                "shared_memory_type": type(shared_memory).__name__,
                "key": normalized_key,
            },
        )

    try:
        return getter(normalized_key, default)
    except TypeError:
        value = getter(normalized_key)
        return default if value is None else value


def shared_memory_set(shared_memory: Any, key: str, value: Any, *, ttl: Optional[int] = None) -> None:
    """Write a value through dict, ``set`` or ``put`` shared-memory APIs."""
    if shared_memory is None:
        raise _make_quality_error(
            "shared_memory is unavailable",
            error_type=QualityErrorType.QUALITY_MEMORY_UNAVAILABLE,
            severity=QualitySeverity.MEDIUM,
            stage=QualityStage.PERSISTENCE,
            domain=QualityDomain.MEMORY,
            disposition=QualityDisposition.FALLBACK,
            retryable=True,
        )

    normalized_key = nonempty_text(key, "shared_memory.key")
    normalized_ttl = optional_nonnegative_int(
        ttl,
        "shared_memory.ttl",
    )

    if isinstance(shared_memory, MutableMapping):
        shared_memory[normalized_key] = value
        return

    setter = getattr(shared_memory, "set", None)
    if not callable(setter):
        setter = getattr(shared_memory, "put", None)

    if not callable(setter):
        raise _make_quality_error(
            "shared_memory does not expose a compatible set/put method",
            error_type=QualityErrorType.QUALITY_MEMORY_UNAVAILABLE,
            severity=QualitySeverity.MEDIUM,
            stage=QualityStage.PERSISTENCE,
            domain=QualityDomain.MEMORY,
            disposition=QualityDisposition.FALLBACK,
            retryable=True,
            context={
                "shared_memory_type": type(shared_memory).__name__,
                "key": normalized_key,
            },
        )

    try:
        if normalized_ttl is None:
            setter(normalized_key, value)
        else:
            setter(normalized_key, value, ttl=normalized_ttl)
    except TypeError:
        # Preserve compatibility with implementations that do not accept TTL.
        setter(normalized_key, value)


def shared_memory_append(
    shared_memory: Any,
    key: str,
    values: Any,
    *,
    ttl: Optional[int] = None,
    max_items: Optional[int] = None,
) -> List[Any]:
    """Append items to a shared-memory list.

    A native ``append``/``extend`` API is preferred. If unavailable, a
    read-modify-write fallback is used. The fallback is not atomic and should
    only be used with shared-memory backends whose access pattern is already
    serialized by the caller.
    """
    normalized_key = nonempty_text(key, "shared_memory.key")
    normalized_values = (
        list(values)
        if isinstance(values, (list, tuple))
        else [values]
    )
    normalized_max = (
        positive_int(max_items, "shared_memory.max_items")
        if max_items is not None
        else None
    )

    native_extend = getattr(shared_memory, "extend", None)
    if callable(native_extend):
        try:
            if ttl is None:
                native_extend(normalized_key, normalized_values)
            else:
                native_extend(
                    normalized_key,
                    normalized_values,
                    ttl=optional_nonnegative_int(ttl, "shared_memory.ttl"),
                )
            latest = shared_memory_get(shared_memory, normalized_key, default=[])
            result = (
                list(latest)
                if isinstance(latest, (list, tuple))
                else normalized_values
            )
            if normalized_max is not None:
                result = result[-normalized_max:]
                shared_memory_set(shared_memory, normalized_key, result, ttl=ttl)
            return result
        except TypeError:
            pass

    current = shared_memory_get(shared_memory, normalized_key, default=[])
    if current is None:
        current_list: List[Any] = []
    elif isinstance(current, list):
        current_list = list(current)
    elif isinstance(current, tuple):
        current_list = list(current)
    else:
        current_list = [current]

    current_list.extend(normalized_values)

    if normalized_max is not None and len(current_list) > normalized_max:
        current_list = current_list[-normalized_max:]

    shared_memory_set(
        shared_memory,
        normalized_key,
        current_list,
        ttl=ttl,
    )
    return current_list


def shared_memory_publish(shared_memory: Any, channel: str, payload: Any) -> bool:
    """Publish an optional quality event.

    Returns ``False`` when the backend intentionally has no publish interface.
    Publication is optional in the current quality architecture.
    """
    if shared_memory is None:
        return False

    publisher = getattr(shared_memory, "publish", None)
    if not callable(publisher):
        return False

    publisher(nonempty_text(channel, "shared_memory.channel"), payload)
    return True


# ---------------------------------------------------------------------------
# Runtime bridge resolution
# ---------------------------------------------------------------------------

def resolve_factory_candidate(
    factory: Any,
    candidate: str,
    *,
    method_names: Sequence[str] = (
        "get",
        "create",
        "create_agent",
        "build",
        "resolve",
    ),
) -> Any:
    """Resolve one candidate from a callable or object-style agent factory.

    ``TypeError`` is treated as a signature mismatch for a candidate method.
    Other exceptions are intentionally propagated so the caller can normalize
    them through ``quality_error.py`` with the correct runtime context.
    """
    if factory is None:
        return None

    normalized_candidate = nonempty_text(
        candidate,
        "factory.candidate",
    )

    if callable(factory):
        try:
            return factory(normalized_candidate)
        except TypeError:
            # A callable object may still expose explicit factory methods.
            pass

    for method_name in method_names:
        method = getattr(factory, str(method_name), None)
        if not callable(method):
            continue
        try:
            resolved = method(normalized_candidate)
        except TypeError:
            continue
        if resolved is not None:
            return resolved

    return None


def resolve_runtime_bridge(
    factory: Any,
    candidates: Any,
    *,
    method_names: Sequence[str] = (
        "get",
        "create",
        "create_agent",
        "build",
        "resolve",
    ),
) -> Any:
    """Resolve the first available runtime bridge from ordered candidates."""
    for candidate in string_list(
        candidates,
        drop_empty=True,
        deduplicate=True,
    ):
        resolved = resolve_factory_candidate(
            factory,
            candidate,
            method_names=method_names,
        )
        if resolved is not None:
            return resolved
    return None



__all__ = [
    # Constants / policy primitives
    "QUALITY_VERDICTS",
    "QUALITY_SEVERITIES",
    "VERDICT_RANK",
    "SEVERITY_RANK",
    "DEFAULT_VERDICT_SCORES",
    "QualityThresholds",
    # Time / identifiers
    "utc_now",
    "utc_timestamp",
    "utc_iso_timestamp",
    "generate_quality_id",
    # Generic serialization / hashing
    "safe_deepcopy",
    "json_safe",
    "stable_json_dumps",
    "stable_hash",
    "normalized_mapping",
    # Scalar / collection validation
    "clamp",
    "bounded_score",
    "positive_int",
    "nonnegative_int",
    "optional_nonnegative_int",
    "coerce_bool",
    "nonempty_text",
    "string_list",
    "merge_unique_strings",
    "float_mapping",
    # Verdict / severity / scoring
    "normalize_verdict",
    "normalize_severity",
    "score_from_verdict",
    "verdict_from_score",
    "worst_verdict",
    "worst_severity",
    "normalize_weights",
    "weighted_score",
    # Record / schema helpers
    "normalize_records",
    "resolve_record_id",
    "record_fingerprint",
    "schema_required_fields",
    "schema_version",
    "schema_fingerprint",
    # Finding helpers
    "normalize_finding",
    "normalize_findings",
    "collect_flags",
    "collect_remediation_actions",
    "record_level_flags",
    "aggregate_subsystem_score",
    "aggregate_subsystem_verdict",
    "decision_confidence",
    # Error conversion
    "quality_error_from_payload",
    # Shared-memory adapters
    "shared_memory_get",
    "shared_memory_set",
    "shared_memory_append",
    "shared_memory_publish",
    # Runtime bridge helpers
    "resolve_factory_candidate",
    "resolve_runtime_bridge",
]
