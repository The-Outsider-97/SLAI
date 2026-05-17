"""
Production-grade shared helpers for the learning subsystem.

This module centralizes broad, reusable primitives across the learning stack
(e.g., RL agents, strategy selectors, adaptation flows, memory/orchestration,
and recovery policies) so all layers share consistent semantics.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
import uuid
import torch  # type: ignore

from statistics import median
from collections import deque
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Deque, Dict, Iterator, List, Optional, Tuple, TypeVar

from .config_loader import get_config_section, load_global_config
from .learning_error import *
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Learning Helpers")
printer = PrettyPrinter()

T = TypeVar("T")
JsonDict = Dict[str, Any]

EPSILON = 1e-12
DEFAULT_FLOAT_PRECISION = 8


@dataclass(frozen=True)
class LearningStepRecord:
    """Canonical representation of one learning transition/log step."""

    step_id: str
    state: Any
    action: Any
    reward: float
    next_state: Any
    done: bool = False
    meta: JsonDict = field(default_factory=dict)
    timestamp_utc: str = field(default_factory=lambda: utc_now().isoformat())

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass(frozen=True)
class RunningStatsSnapshot:
    """Snapshot for online running statistics."""

    count: int
    mean: float
    variance: float
    std: float
    minimum: float
    maximum: float


@dataclass(frozen=True)
class EpisodeMetrics:
    """Canonical episode-level aggregate metrics."""

    episode_index: int
    total_reward: float
    length: int
    average_reward: float
    moving_average_reward: float
    completion_rate: float


@dataclass(frozen=True)
class RetryPolicy:
    """Common retry/backoff policy for learning workflows."""

    max_attempts: int = 3
    initial_delay: float = 0.1
    multiplier: float = 2.0
    max_delay: float = 5.0
    jitter_ratio: float = 0.0

    def delay_for_attempt(self, attempt: int) -> float:
        if attempt < 1:
            raise InvalidConfigError("attempt must be >= 1")
        base = self.initial_delay * (self.multiplier ** (attempt - 1))
        bounded = min(self.max_delay, max(0.0, base))
        if self.jitter_ratio <= 0:
            return bounded
        spread = bounded * self.jitter_ratio
        return max(0.0, random.uniform(bounded - spread, bounded + spread))


class RunningStats:
    """Numerically stable online stats using Welford's algorithm."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._count = 0
        self._mean = 0.0
        self._m2 = 0.0
        self._min = float("inf")
        self._max = float("-inf")

    def update(self, value: float) -> None:
        x = float(value)
        self._count += 1
        delta = x - self._mean
        self._mean += delta / self._count
        delta2 = x - self._mean
        self._m2 += delta * delta2
        self._min = min(self._min, x)
        self._max = max(self._max, x)

    def extend(self, values: Iterable[float]) -> None:
        for value in values:
            self.update(value)

    @property
    def count(self) -> int:
        return self._count

    def snapshot(self) -> RunningStatsSnapshot:
        variance = (self._m2 / self._count) if self._count > 0 else 0.0
        return RunningStatsSnapshot(
            count=self._count,
            mean=self._mean if self._count > 0 else 0.0,
            variance=max(0.0, variance),
            std=math.sqrt(max(0.0, variance)),
            minimum=self._min if self._count > 0 else 0.0,
            maximum=self._max if self._count > 0 else 0.0,
        )


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def monotonic_seconds() -> float:
    return time.monotonic()


def make_learning_id(prefix: str = "learn") -> str:
    safe = slugify(prefix, fallback="learn")
    return f"{safe}_{uuid.uuid4().hex[:12]}"


def stable_hash(value: Any, *, digest_size: int = 16) -> str:
    payload = json.dumps(to_json_safe(value), sort_keys=True, ensure_ascii=False)
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=digest_size).hexdigest()


def coerce_float(value: Any, default: float = 0.0, *, minimum: Optional[float] = None, maximum: Optional[float] = None) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        result = default
    if minimum is not None:
        result = max(minimum, result)
    if maximum is not None:
        result = min(maximum, result)
    return result


def coerce_int(value: Any, default: int = 0, *, minimum: Optional[int] = None, maximum: Optional[int] = None) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        result = default
    if minimum is not None:
        result = max(minimum, result)
    if maximum is not None:
        result = min(maximum, result)
    return result


def coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    return default


def clamp(value: float, minimum: float, maximum: float) -> float:
    if minimum > maximum:
        raise InvalidConfigError(f"Invalid clamp range: minimum ({minimum}) > maximum ({maximum})")
    return max(minimum, min(maximum, float(value)))


def clamp01(value: float) -> float:
    return clamp(value, 0.0, 1.0)


def round_float(value: float, precision: int = DEFAULT_FLOAT_PRECISION) -> float:
    return round(float(value), max(0, int(precision)))


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    if abs(float(denominator)) <= EPSILON:
        return default
    return float(numerator) / float(denominator)


def normalize_probabilities(values: Sequence[float], *, epsilon: float = EPSILON) -> List[float]:
    arr = [max(0.0, float(v)) for v in values]
    total = sum(arr)
    if total <= epsilon:
        if not arr:
            return []
        uniform = 1.0 / len(arr)
        return [uniform for _ in arr]
    return [v / total for v in arr]


def argmax(values: Sequence[float]) -> int:
    if not values:
        raise InvalidConfigError("argmax requires a non-empty sequence")
    best_idx = 0
    best_val = float(values[0])
    for idx, value in enumerate(values[1:], start=1):
        score = float(value)
        if score > best_val:
            best_idx, best_val = idx, score
    return best_idx


def argmin(values: Sequence[float]) -> int:
    if not values:
        raise InvalidConfigError("argmin requires a non-empty sequence")
    best_idx = 0
    best_val = float(values[0])
    for idx, value in enumerate(values[1:], start=1):
        score = float(value)
        if score < best_val:
            best_idx, best_val = idx, score
    return best_idx


def discounted_returns(rewards: Sequence[float], gamma: float = 0.99, *, normalize: bool = False) -> List[float]:
    gamma = clamp(gamma, 0.0, 1.0)
    out: List[float] = [0.0] * len(rewards)
    running = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        running = float(rewards[i]) + gamma * running
        out[i] = running
    if normalize and out:
        return z_score_normalize(out)
    return out


def generalized_advantage_estimate(
    rewards: Sequence[float],
    values: Sequence[float],
    dones: Sequence[bool],
    *,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> List[float]:
    if not (len(rewards) == len(values) == len(dones)):
        raise InvalidConfigError("rewards, values, and dones must have the same length")
    gamma = clamp01(gamma)
    lam = clamp01(lam)
    advantages = [0.0] * len(rewards)
    gae = 0.0
    next_value = 0.0
    for idx in range(len(rewards) - 1, -1, -1):
        mask = 0.0 if dones[idx] else 1.0
        delta = float(rewards[idx]) + gamma * next_value * mask - float(values[idx])
        gae = delta + gamma * lam * mask * gae
        advantages[idx] = gae
        next_value = float(values[idx])
    return advantages


def z_score_normalize(values: Sequence[float], eps: float = EPSILON) -> List[float]:
    if not values:
        return []
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    std = math.sqrt(max(variance, 0.0))
    if std <= eps:
        return [0.0 for _ in values]
    return [(float(v) - mean) / std for v in values]


def min_max_scale(values: Sequence[float], *, low: float = 0.0, high: float = 1.0) -> List[float]:
    if not values:
        return []
    if low >= high:
        raise InvalidConfigError("min_max_scale expects low < high")
    vmin, vmax = min(values), max(values)
    if math.isclose(vmin, vmax):
        mid = (low + high) / 2.0
        return [mid for _ in values]
    span = vmax - vmin
    scale = (high - low) / span
    return [low + (float(v) - vmin) * scale for v in values]


def winsorize(values: Sequence[float], lower_q: float = 0.05, upper_q: float = 0.95) -> List[float]:
    if not values:
        return []
    if lower_q < 0 or upper_q > 1 or lower_q >= upper_q:
        raise InvalidConfigError("winsorize requires 0 <= lower_q < upper_q <= 1")
    sorted_values = sorted(float(v) for v in values)
    li = int((len(sorted_values) - 1) * lower_q)
    ui = int((len(sorted_values) - 1) * upper_q)
    lo = sorted_values[li]
    hi = sorted_values[ui]
    return [clamp(float(v), lo, hi) for v in values]


def linear_schedule(step: int, start: float, end: float, duration: int) -> float:
    if duration <= 0:
        return float(end)
    ratio = clamp(step / duration, 0.0, 1.0)
    return float(start + (end - start) * ratio)


def exponential_decay(step: int, initial: float, decay_rate: float, min_value: float = 0.0) -> float:
    if decay_rate <= 0:
        raise InvalidConfigError("decay_rate must be > 0")
    value = float(initial) * (float(decay_rate) ** max(0, int(step)))
    return max(float(min_value), value)


def cosine_anneal(step: int, initial: float, min_value: float, cycle_steps: int) -> float:
    if cycle_steps <= 0:
        raise InvalidConfigError("cycle_steps must be > 0")
    phase = (max(0, step) % cycle_steps) / cycle_steps
    return min_value + 0.5 * (initial - min_value) * (1 + math.cos(math.pi * phase))


def moving_average(values: Sequence[float], window_size: int = 10) -> List[float]:
    if window_size <= 0:
        raise InvalidConfigError("window_size must be > 0")
    history: Deque[float] = deque(maxlen=window_size)
    out: List[float] = []
    for value in values:
        history.append(float(value))
        out.append(sum(history) / len(history))
    return out


def ema(values: Sequence[float], alpha: float = 0.1) -> List[float]:
    if not values:
        return []
    alpha = clamp(alpha, EPSILON, 1.0)
    out = [float(values[0])]
    for value in values[1:]:
        out.append(alpha * float(value) + (1 - alpha) * out[-1])
    return out


def rolling_window(sequence: Sequence[T], window_size: int) -> Iterator[Tuple[T, ...]]:
    if window_size <= 0:
        raise InvalidConfigError("window_size must be > 0")
    if len(sequence) < window_size:
        return iter(())
    return (tuple(sequence[i : i + window_size]) for i in range(0, len(sequence) - window_size + 1))


def chunked(sequence: Sequence[T], chunk_size: int) -> List[List[T]]:
    if chunk_size <= 0:
        raise InvalidConfigError("chunk_size must be > 0")
    return [list(sequence[i : i + chunk_size]) for i in range(0, len(sequence), chunk_size)]


def ensure_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def sample_minibatch(items: Sequence[T], batch_size: int, *, replace: bool = False, rng: Optional[random.Random] = None) -> List[T]:
    if batch_size <= 0:
        raise InvalidConfigError("batch_size must be > 0")
    if not items:
        return []
    rand = rng or random
    if replace:
        return [items[rand.randrange(0, len(items))] for _ in range(batch_size)]
    if batch_size >= len(items):
        return list(items)
    return rand.sample(list(items), batch_size)


def train_validation_split(items: Sequence[T], validation_ratio: float = 0.2, *, rng: Optional[random.Random] = None) -> Tuple[List[T], List[T]]:
    if not items:
        return [], []
    ratio = clamp(validation_ratio, 0.0, 1.0)
    indices = list(range(len(items)))
    (rng or random).shuffle(indices)
    split = int(round(len(items) * (1.0 - ratio)))
    train_idx = indices[:split]
    val_idx = indices[split:]
    return [items[i] for i in train_idx], [items[i] for i in val_idx]



def deep_merge_dicts(*mappings: Optional[Mapping[str, Any]]) -> JsonDict:
    result: JsonDict = {}
    for mapping in mappings:
        if not mapping:
            continue
        for key, value in mapping.items():
            if key in result and isinstance(result[key], MutableMapping) and isinstance(value, Mapping):
                result[key] = deep_merge_dicts(result[key], value)
            else:
                result[key] = value
    return result


def flatten_dict(data: Mapping[str, Any], *, parent_key: str = "", sep: str = ".") -> JsonDict:
    output: JsonDict = {}
    for key, value in data.items():
        composed = f"{parent_key}{sep}{key}" if parent_key else str(key)
        if isinstance(value, Mapping):
            output.update(flatten_dict(value, parent_key=composed, sep=sep))
        else:
            output[composed] = value
    return output


def prune_none(data: Any) -> Any:
    if isinstance(data, Mapping):
        return {k: prune_none(v) for k, v in data.items() if v is not None}
    if isinstance(data, list):
        return [prune_none(v) for v in data if v is not None]
    return data


def slugify(text: str, fallback: str = "value") -> str:
    clean = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(text)).strip("_")
    while "__" in clean:
        clean = clean.replace("__", "_")
    return clean or fallback


def to_json_safe(value: Any, *, max_depth: int = 6, max_items: int = 64) -> Any:
    def _convert(obj: Any, depth: int) -> Any:
        if depth > max_depth:
            return "<max-depth>"
        if obj is None or isinstance(obj, (bool, int, float, str)):
            return obj
        if isinstance(obj, Mapping):
            out: JsonDict = {}
            for i, (k, v) in enumerate(obj.items()):
                if i >= max_items:
                    out["<truncated>"] = len(obj) - max_items
                    break
                out[str(k)] = _convert(v, depth + 1)
            return out
        if isinstance(obj, (list, tuple, set, frozenset, deque)):
            seq = list(obj)
            converted = [_convert(v, depth + 1) for v in seq[:max_items]]
            if len(seq) > max_items:
                converted.append(f"<truncated:{len(seq)-max_items}>")
            return converted
        if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
            return _convert(obj.to_dict(), depth + 1)
        return repr(obj)

    return _convert(value, 0)


def summarize_rewards(rewards: Sequence[float]) -> JsonDict:
    if not rewards:
        return {"count": 0, "mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "sum": 0.0}
    values = [float(v) for v in rewards]
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "median": median(values),
        "min": min(values),
        "max": max(values),
        "sum": sum(values),
    }


def make_step_record(state: Any, action: Any, reward: float, next_state: Any, done: bool, *, meta: Optional[JsonDict] = None, step_id: Optional[str] = None) -> LearningStepRecord:
    return LearningStepRecord(
        step_id=step_id or make_learning_id("step"),
        state=state,
        action=action,
        reward=float(reward),
        next_state=next_state,
        done=bool(done),
        meta=dict(meta or {}),
    )


def make_episode_metrics(episode_index: int, rewards: Sequence[float], *, moving_average_reward: float = 0.0, completion_rate: float = 0.0) -> EpisodeMetrics:
    total = sum(float(r) for r in rewards)
    length = len(rewards)
    average = safe_divide(total, length, default=0.0)
    return EpisodeMetrics(
        episode_index=max(0, int(episode_index)),
        total_reward=total,
        length=length,
        average_reward=average,
        moving_average_reward=float(moving_average_reward),
        completion_rate=clamp01(float(completion_rate)),
    )


def benchmark(operation: Callable[[], T]) -> Tuple[T, float]:
    start = monotonic_seconds()
    result = operation()
    elapsed = monotonic_seconds() - start
    return result, elapsed



# ===========================================================================
# Validation helpers (free functions — no class instantiation required)
# ===========================================================================
def validate_positive(
    value: Any,
    name: str,
    strict: bool = True,
    error_class: Type[LearningError] = InvalidConfigError,
) -> None:
    """Assert that *value* is a positive real number.
 
    Parameters
    ----------
    value :
        The value to check.
    name :
        Human-readable name used in the error message (e.g. ``"learning_rate"``).
    strict :
        If True (default), require value > 0.  If False, require value >= 0.
    error_class :
        Exception class to raise.  Defaults to ``InvalidConfigError``.
 
    Raises
    ------
    InvalidConfigError (or *error_class*)
        If the assertion fails or if the value is not numeric.
    """
    try:
        fval = float(value)
    except (TypeError, ValueError) as exc:
        raise error_class(
            f"'{name}' must be numeric, got {type(value).__name__!r}",
            context={"name": name, "received": repr(value)},
            cause=exc,
        ) from exc
 
    if not math.isfinite(fval):
        raise error_class(
            f"'{name}' must be finite, got {fval}",
            context={"name": name, "received": fval},
        )
 
    if strict and fval <= 0:
        raise error_class(
            f"'{name}' must be > 0, got {fval}",
            context={"name": name, "received": fval},
        )
    if not strict and fval < 0:
        raise error_class(
            f"'{name}' must be >= 0, got {fval}",
            context={"name": name, "received": fval},
        )
 
 
def validate_non_negative(value: Any, name: str, error_class: Type[LearningError] = InvalidConfigError) -> None:
    """Assert that *value* >= 0.  Convenience wrapper over ``validate_positive``."""
    validate_positive(value, name, strict=False, error_class=error_class)
 
 
def validate_in_range(value: Any, name: str, low: float, high: float, inclusive_low: bool = True,
                      inclusive_high: bool = True, error_class: Type[LearningError] = InvalidConfigError) -> None:
    """Assert that *value* lies within [low, high] (or open variants thereof).
 
    Parameters
    ----------
    value :
        The value to check.
    name :
        Human-readable parameter name.
    low, high :
        The lower and upper bounds.
    inclusive_low, inclusive_high :
        Whether each bound is inclusive (default: both inclusive).
    error_class :
        Exception class to raise.
 
    Raises
    ------
    InvalidConfigError (or *error_class*)
    """
    if low > high:
        raise ValueError(f"validate_in_range: low ({low}) must be <= high ({high})")
 
    try:
        fval = float(value)
    except (TypeError, ValueError) as exc:
        raise error_class(
            f"'{name}' must be numeric, got {type(value).__name__!r}",
            context={"name": name, "received": repr(value)},
            cause=exc,
        ) from exc
 
    lo_ok = (fval >= low) if inclusive_low else (fval > low)
    hi_ok = (fval <= high) if inclusive_high else (fval < high)
 
    if not (lo_ok and hi_ok):
        lo_bracket = "[" if inclusive_low else "("
        hi_bracket = "]" if inclusive_high else ")"
        raise error_class(
            f"'{name}' must be in {lo_bracket}{low}, {high}{hi_bracket}, got {fval}",
            context={"name": name, "received": fval, "low": low, "high": high},
        )
 
 
def validate_probability(value: Any, name: str, error_class: Type[LearningError] = InvalidConfigError) -> None:
    """Assert that *value* is a valid probability: a finite float in [0, 1]."""
    validate_in_range(value, name, low=0.0, high=1.0, error_class=error_class)
 
 
def validate_finite(value: Any, name: str, error_class: Type[Union[NaNException, InfException]] = NaNException) -> None:
    """Assert that *value* is finite (not NaN, not Inf).

    Works with plain Python floats and — when torch/numpy are available —
    with scalar tensors and numpy scalars.

    Parameters
    ----------
    value :
        The value to check.
    name :
        Human-readable name used in the error message.
    error_class :
        Exception class to raise (default: ``NaNException``).
        Must be either ``NaNException`` or ``InfException`` because they
        accept the ``location`` parameter used internally.

    Raises
    ------
    NaNException (or *error_class*)
        If the value is NaN.
    InfException
        If the value is Inf.
    """
    # Try to convert to Python float for a uniform check
    try:
        # Handle torch Tensors
        try:
            import torch as _t  # type: ignore[import-not-found]
            if isinstance(value, _t.Tensor):
                if value.numel() != 1:
                    raise error_class(
                        f"'{name}' must be a scalar tensor for finiteness check",
                        location=name,
                    )
                fval = float(value.detach().item())
            else:
                fval = float(value)
        except ImportError:
            fval = float(value)
    except (TypeError, ValueError) as exc:
        raise error_class(
            f"'{name}' is not numeric: {type(value).__name__}",
            location=name,
            cause=exc,
        ) from exc

    if math.isnan(fval):
        raise NaNException(
            f"NaN detected in '{name}'",
            location=name,
        )
    if math.isinf(fval):
        raise InfException(
            f"Inf detected in '{name}'",
            location=name,
        )
 
 
def validate_type(value: Any, name: str, expected: Union[type, Tuple[type, ...]],
                  error_class: Type[LearningError] = InvalidConfigError) -> None:
    """Assert that *value* is an instance of *expected*.
 
    Parameters
    ----------
    value :
        The value to type-check.
    name :
        Human-readable name.
    expected :
        A type or tuple of types.
    error_class :
        Exception class to raise.
 
    Raises
    ------
    InvalidConfigError (or *error_class*)
    """
    if not isinstance(value, expected):
        if isinstance(expected, tuple):
            expected_str = " | ".join(t.__name__ for t in expected)
        else:
            expected_str = expected.__name__
        raise error_class(
            f"'{name}' must be of type {expected_str}, got {type(value).__name__!r}",
            context={
                "name": name,
                "expected_type": expected_str,
                "actual_type": type(value).__name__,
                "received": repr(value),
            },
        )
 

def validate_required_keys(payload: Mapping[str, Any], required_keys: Iterable[str], *, name: str = "payload") -> None:
    """Assert that all *required_keys* are present in *mapping*."""
    missing = [key for key in required_keys if key not in payload]
    if missing:
        raise InvalidConfigError(f"{name} missing required keys: {missing}")

def _validate_required_keys(
    mapping: Mapping[str, Any],
    required_keys: Iterable[str],
    section: str = "config",
    error_class: Type[LearningError] = MissingConfigKeyError,
) -> None:
    missing = [k for k in required_keys if k not in mapping]
    if missing:
        raise error_class(section, missing)  # type: ignore[call-arg]
 
 
def validate_non_empty_sequence(seq: Any, name: str, error_class: Type[LearningError] = InvalidConfigError) -> None:
    """Assert that *seq* is a non-empty sequence (list, tuple, or similar).
 
    Parameters
    ----------
    seq :
        The sequence to check.
    name :
        Human-readable parameter name.
    error_class :
        Exception class to raise.
 
    Raises
    ------
    InvalidConfigError (or *error_class*)
    """
    if not hasattr(seq, "__len__"):
        raise error_class(
            f"'{name}' must be a sequence, got {type(seq).__name__!r}",
            context={"name": name, "actual_type": type(seq).__name__},
        )
    if len(seq) == 0:  # type: ignore[arg-type]
        raise error_class(
            f"'{name}' must be a non-empty sequence",
            context={"name": name},
        )

