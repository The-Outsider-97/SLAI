from __future__ import annotations

import math
import time
import numpy as np # type: ignore

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .utils.config_loader import get_config_section, load_global_config
from .utils.buffer_errors import *
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Buffer Validation")
printer = PrettyPrinter()

Transition = Tuple[Any, Any, Any, float, Any, bool]
_TRANSITION_FIELDS: Tuple[str, str, str, str, str, str] = (
    "agent_id", "state", "action", "reward", "next_state", "done"
)
_DEFAULT_MAPPING_KEYS: Dict[str, str] = {field_name: field_name for field_name in _TRANSITION_FIELDS}


@dataclass(frozen=True)
class TransitionValidationIssue:
    """Structured validation issue for one failed transition."""

    index: Optional[int]
    error_type: str
    message: str
    field_name: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_exception(cls, exc: BaseException, index: Optional[int] = None) -> "TransitionValidationIssue":
        field_name = getattr(exc, "field_name", None)
        details: Dict[str, Any] = {}
        if hasattr(exc, "to_dict"):
            try:
                payload = exc.to_dict(include_cause=False) # type: ignore[attr-defined]
                details = dict(payload.get("context", {}).get("details", {}) or {})
                field_name = field_name or payload.get("context", {}).get("field_name")
            except Exception:
                details = {}
        return cls(
            index=index,
            error_type=type(exc).__name__,
            message=str(exc),
            field_name=field_name,
            details=details,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "error_type": self.error_type,
            "message": self.message,
            "field_name": self.field_name,
            "details": dict(self.details),
        }


@dataclass
class ValidationReport:
    """Structured report for transition or batch validation."""

    valid: int = 0
    invalid: int = 0
    errors: List[str] = field(default_factory=list)
    issues: List[TransitionValidationIssue] = field(default_factory=list)
    valid_indices: List[int] = field(default_factory=list)
    invalid_indices: List[int] = field(default_factory=list)
    normalized: List[Transition] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    @property
    def total(self) -> int:
        return self.valid + self.invalid

    @property
    def is_clean(self) -> bool:
        return self.invalid == 0

    @property
    def rejection_rate(self) -> float:
        return self.invalid / self.total if self.total else 0.0

    def add_success(self, index: int, transition: Optional[Transition] = None, *, keep_normalized: bool = True) -> None:
        self.valid += 1
        self.valid_indices.append(index)
        if keep_normalized and transition is not None:
            self.normalized.append(transition)

    def add_error(self, index: int, exc: BaseException, *, max_errors: Optional[int] = None) -> None:
        self.invalid += 1
        self.invalid_indices.append(index)
        if max_errors is not None and len(self.errors) >= max_errors:
            return
        issue = TransitionValidationIssue.from_exception(exc, index=index)
        self.issues.append(issue)
        self.errors.append(f"index={index}: {issue.error_type}: {issue.message}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "invalid": self.invalid,
            "total": self.total,
            "is_clean": self.is_clean,
            "rejection_rate": self.rejection_rate,
            "valid_indices": list(self.valid_indices),
            "invalid_indices": list(self.invalid_indices),
            "errors": list(self.errors),
            "issues": [issue.to_dict() for issue in self.issues],
            "elapsed_seconds": self.elapsed_seconds,
        }


@dataclass
class TransitionSchema:
    """Schema contract for replay transition validation and canonical coercion.

    Canonical transition shape:
        (agent_id, state, action, reward, next_state, done)
    """

    required_length: int = 6
    enforce_numeric_reward: bool = True
    enforce_boolean_done: bool = True
    allow_none_state: bool = False
    allow_none_next_state: bool = False
    max_abs_reward: Optional[float] = None
    allow_none_agent_id: bool = True
    allow_none_action: bool = True
    allow_nan_reward: bool = False
    allow_inf_reward: bool = False
    allow_mapping_transition: bool = True
    coerce_reward_to_float: bool = True
    coerce_numpy_scalars: bool = True
    coerce_done_from_int: bool = False
    coerce_done_from_string: bool = False
    require_state_next_state_shape_match: bool = False
    max_report_errors: int = 100
    keep_normalized_batch: bool = True
    field_names: Tuple[str, str, str, str, str, str] = _TRANSITION_FIELDS
    mapping_keys: Dict[str, str] = field(default_factory=lambda: dict(_DEFAULT_MAPPING_KEYS))

    @classmethod
    def from_config(cls, user_config: Optional[Mapping[str, Any]] = None) -> "TransitionSchema":
        load_global_config()
        config = dict(get_config_section("validation") or {})
        if user_config:
            overlay = (
                user_config.get("validation", {})
                if isinstance(user_config.get("validation"), Mapping)
                else user_config
            )
            config.update(dict(overlay))

        mapping_keys = dict(_DEFAULT_MAPPING_KEYS)
        if isinstance(config.get("mapping_keys"), Mapping):
            mapping_keys.update({str(k): str(v) for k, v in dict(config["mapping_keys"]).items()})

        field_names_raw = config.get("field_names", _TRANSITION_FIELDS)
        field_names = tuple(str(v) for v in field_names_raw)
        if len(field_names) != 6:
            raise ConfigValueError("validation.field_names", field_names_raw, "exactly 6 field names", section="validation")

        schema = cls(
            required_length=int(config.get("required_length", 6)),
            enforce_numeric_reward=bool(config.get("enforce_numeric_reward", True)),
            enforce_boolean_done=bool(config.get("enforce_boolean_done", True)),
            allow_none_state=bool(config.get("allow_none_state", False)),
            allow_none_next_state=bool(config.get("allow_none_next_state", False)),
            max_abs_reward=(float(config["max_abs_reward"]) if config.get("max_abs_reward") is not None else None),
            allow_none_agent_id=bool(config.get("allow_none_agent_id", True)),
            allow_none_action=bool(config.get("allow_none_action", True)),
            allow_nan_reward=bool(config.get("allow_nan_reward", False)),
            allow_inf_reward=bool(config.get("allow_inf_reward", False)),
            allow_mapping_transition=bool(config.get("allow_mapping_transition", True)),
            coerce_reward_to_float=bool(config.get("coerce_reward_to_float", True)),
            coerce_numpy_scalars=bool(config.get("coerce_numpy_scalars", True)),
            coerce_done_from_int=bool(config.get("coerce_done_from_int", False)),
            coerce_done_from_string=bool(config.get("coerce_done_from_string", False)),
            require_state_next_state_shape_match=bool(config.get("require_state_next_state_shape_match", False)),
            max_report_errors=max(0, int(config.get("max_report_errors", 100))),
            keep_normalized_batch=bool(config.get("keep_normalized_batch", True)),
            field_names=field_names, # type: ignore[arg-type]
            mapping_keys=mapping_keys,
        )
        schema.validate_schema()
        return schema

    def validate_schema(self) -> None:
        if self.required_length != 6:
            raise TransitionSchemaError("required_length must remain 6 for canonical replay transitions", schema=self)
        if self.max_abs_reward is not None and self.max_abs_reward < 0:
            raise ConfigValueError("validation.max_abs_reward", self.max_abs_reward, ">= 0 or null", section="validation")
        if self.max_report_errors < 0:
            raise ConfigValueError("validation.max_report_errors", self.max_report_errors, ">= 0", section="validation")
        missing_keys = [name for name in self.field_names if name not in self.mapping_keys]
        if missing_keys:
            raise TransitionSchemaError(f"mapping_keys missing canonical fields: {missing_keys}", schema=self)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "required_length": self.required_length,
            "enforce_numeric_reward": self.enforce_numeric_reward,
            "enforce_boolean_done": self.enforce_boolean_done,
            "allow_none_state": self.allow_none_state,
            "allow_none_next_state": self.allow_none_next_state,
            "max_abs_reward": self.max_abs_reward,
            "allow_none_agent_id": self.allow_none_agent_id,
            "allow_none_action": self.allow_none_action,
            "allow_nan_reward": self.allow_nan_reward,
            "allow_inf_reward": self.allow_inf_reward,
            "allow_mapping_transition": self.allow_mapping_transition,
            "coerce_reward_to_float": self.coerce_reward_to_float,
            "coerce_numpy_scalars": self.coerce_numpy_scalars,
            "coerce_done_from_int": self.coerce_done_from_int,
            "coerce_done_from_string": self.coerce_done_from_string,
            "require_state_next_state_shape_match": self.require_state_next_state_shape_match,
            "max_report_errors": self.max_report_errors,
            "keep_normalized_batch": self.keep_normalized_batch,
            "field_names": list(self.field_names),
            "mapping_keys": dict(self.mapping_keys),
        }


class TransitionValidator:
    """Validates and sanitizes replay transitions for buffer ingestion.

    The validator is deliberately buffer-safe: it does not import replay buffers,
    persistence, telemetry, or business logic. It only enforces the canonical
    transition contract shared by replay, reservoir, n-step, and distributed buffers.
    """

    def __init__(self, schema: Optional[TransitionSchema] = None,
                 user_config: Optional[Mapping[str, Any]] = None):
        self.schema = schema or TransitionSchema.from_config(user_config=user_config)

    def _payload_from_mapping(self, transition: Mapping[str, Any]) -> Tuple[Any, ...]:
        if not self.schema.allow_mapping_transition:
            raise TransitionTypeError(transition, expected="tuple/list canonical transition")
        values: List[Any] = []
        missing: List[str] = []
        for canonical_name in self.schema.field_names:
            source_key = self.schema.mapping_keys.get(canonical_name, canonical_name)
            if source_key not in transition:
                missing.append(source_key)
            else:
                values.append(transition[source_key])
        if missing:
            raise TransitionValidationError(
                f"Transition mapping missing required keys: {missing}",
                field_name="transition",
                expected=f"keys={list(self.schema.mapping_keys.values())}",
            )
        return tuple(values)

    def _ensure_tuple(self, transition: Any) -> Tuple[Any, ...]:
        if isinstance(transition, tuple):
            payload = transition
        elif isinstance(transition, list):
            payload = tuple(transition)
        elif isinstance(transition, Mapping):
            payload = self._payload_from_mapping(transition)
        else:
            raise TransitionTypeError(transition, expected="tuple, list, or transition mapping")

        if len(payload) != self.schema.required_length:
            raise TransitionLengthError(expected=self.schema.required_length, actual=len(payload))
        return payload

    def _coerce_scalar(self, value: Any) -> Any:
        if self.schema.coerce_numpy_scalars and isinstance(value, np.generic):
            return value.item()
        return value

    def _coerce_reward(self, reward: Any) -> float:
        reward = self._coerce_scalar(reward)
        if not self.schema.enforce_numeric_reward:
            return reward # type: ignore[return-value]
        if not isinstance(reward, (int, float, np.number)) or isinstance(reward, (bool, np.bool_)):
            raise TransitionRewardError(reward)
        try:
            reward_value = float(reward) if self.schema.coerce_reward_to_float else reward
        except Exception as exc:
            raise TransitionCoercionError("reward", reward, "float", str(exc)) from exc
        if isinstance(reward_value, float):
            if math.isnan(reward_value) and not self.schema.allow_nan_reward:
                raise TransitionRewardError(reward, reason="Reward cannot be NaN.")
            if math.isinf(reward_value) and not self.schema.allow_inf_reward:
                raise TransitionRewardError(reward, reason="Reward cannot be infinite.")
            if self.schema.max_abs_reward is not None and abs(reward_value) > self.schema.max_abs_reward:
                raise TransitionRewardError(
                    reward,
                    max_abs=self.schema.max_abs_reward,
                    reason=f"Reward abs({reward_value}) exceeds max_abs_reward={self.schema.max_abs_reward}.",
                )
        return reward_value # type: ignore[return-value]

    def _coerce_done(self, done: Any) -> bool:
        done = self._coerce_scalar(done)
        if not self.schema.enforce_boolean_done:
            return done # type: ignore[return-value]
        if isinstance(done, (bool, np.bool_)):
            return bool(done)
        if self.schema.coerce_done_from_int and isinstance(done, (int, np.integer)) and int(done) in (0, 1):
            return bool(done)
        if self.schema.coerce_done_from_string and isinstance(done, str):
            normalized = done.strip().lower()
            if normalized in {"true", "1", "yes", "y"}:
                return True
            if normalized in {"false", "0", "no", "n"}:
                return False
        raise TransitionDoneError(done)

    def _validate_required_payload(self, value: Any, field_name: str, allow_none: bool) -> None:
        if value is None and not allow_none:
            raise TransitionNoneStateError(field_name)

    @staticmethod
    def _shape_signature(value: Any) -> Optional[Tuple[int, ...]]:
        shape = getattr(value, "shape", None)
        if shape is None:
            return None
        try:
            return tuple(int(dim) for dim in shape)
        except Exception:
            return None

    def _validate_state_pair(self, state: Any, next_state: Any) -> None:
        if not self.schema.require_state_next_state_shape_match:
            return
        state_shape = self._shape_signature(state)
        next_shape = self._shape_signature(next_state)
        if state_shape is not None and next_shape is not None and state_shape != next_shape:
            raise TransitionValidationError(
                f"state and next_state shapes differ: {state_shape} != {next_shape}",
                field_name="next_state",
                value=next_shape,
                expected=f"shape={state_shape}",
            )

    def validate_transition(self, transition: Any, coerce: bool = True,
                            transition_index: Optional[int] = None) -> Transition:
        payload = self._ensure_tuple(transition)
        agent_id, state, action, reward, next_state, done = payload

        self._validate_required_payload(agent_id, "agent_id", self.schema.allow_none_agent_id)
        self._validate_required_payload(state, "state", self.schema.allow_none_state)
        self._validate_required_payload(action, "action", self.schema.allow_none_action)
        self._validate_required_payload(next_state, "next_state", self.schema.allow_none_next_state)
        self._validate_state_pair(state, next_state)

        if coerce:
            reward = self._coerce_reward(reward)
            done = self._coerce_done(done)
        else:
            if self.schema.enforce_numeric_reward and (
                not isinstance(reward, (int, float, np.number)) or isinstance(reward, (bool, np.bool_))
            ):
                raise TransitionRewardError(reward)
            if self.schema.enforce_boolean_done and not isinstance(done, (bool, np.bool_)):
                raise TransitionDoneError(done)

        try:
            return agent_id, state, action, reward, next_state, done # type: ignore[return-value]
        except Exception as exc:
            raise TransitionValidationError(
                "Unable to construct canonical transition tuple.",
                transition_index=transition_index,
                cause=exc,
            ) from exc

    def validate_components(self, agent_id: Any, state: Any, action: Any,
                            reward: Any, next_state: Any, done: Any,
                            coerce: bool = True) -> Transition:
        return self.validate_transition((agent_id, state, action, reward, next_state, done), coerce=coerce)

    def validate_batch(self, transitions: Iterable[Any], coerce: bool = True, *,
                       raise_on_invalid: bool = False, fail_fast: bool = False,
                       keep_normalized: Optional[bool] = None) -> ValidationReport:
        report = ValidationReport()
        keep = self.schema.keep_normalized_batch if keep_normalized is None else bool(keep_normalized)
        start = time.perf_counter()
        for idx, transition in enumerate(transitions):
            try:
                normalized = self.validate_transition(transition=transition, coerce=coerce, transition_index=idx)
                report.add_success(idx, normalized, keep_normalized=keep)
            except TransitionValidationError as exc:
                report.add_error(idx, exc, max_errors=self.schema.max_report_errors)
                if fail_fast:
                    break
        report.elapsed_seconds = time.perf_counter() - start

        if report.invalid:
            logger.warning("Validation report: valid=%s invalid=%s rejection_rate=%.4f",
                           report.valid, report.invalid, report.rejection_rate)
            if raise_on_invalid:
                raise TransitionBatchValidationError(report.invalid, report.total, report.errors)
        return report

    def sanitize_transition(self, transition: Any) -> Transition:
        """Alias for validate_transition(coerce=True)."""
        return self.validate_transition(transition=transition, coerce=True)

    def sanitize_batch(self, transitions: Iterable[Any], *, fail_fast: bool = True) -> List[Transition]:
        report = self.validate_batch(transitions, coerce=True, raise_on_invalid=True, fail_fast=fail_fast)
        return list(report.normalized)

    def require_valid_batch(self, transitions: Iterable[Any], *, coerce: bool = True) -> List[Transition]:
        report = self.validate_batch(transitions, coerce=coerce, raise_on_invalid=True, fail_fast=False)
        return list(report.normalized)

    def is_valid_transition(self, transition: Any, *, coerce: bool = True) -> bool:
        try:
            self.validate_transition(transition, coerce=coerce)
            return True
        except TransitionValidationError:
            return False


__all__ = [
    "TransitionValidator",
    "ValidationReport",
    "TransitionSchema",
    "Transition",
]


if __name__ == "__main__":
    print("\n=== Running  Buffer Validation ===\n")
    printer.status("TEST", " Buffer Validation initialized", "info")

    schema = TransitionSchema(max_abs_reward=10.0, coerce_done_from_int=True)
    validator = TransitionValidator(schema=schema)

    t = validator.sanitize_transition(("agent", [1, 2], 0, np.float32(1.5), [2, 3], 0))
    assert t[3] == 1.5 and t[5] is False

    mapped = validator.sanitize_transition({
        "agent_id": "agent", "state": [0], "action": 1,
        "reward": 2, "next_state": [1], "done": True,
    })
    assert mapped[3] == 2.0 and mapped[5] is True

    batch = [t, mapped, ("bad", None, 0, "nan", [1], False)]
    report = validator.validate_batch(batch)
    assert report.valid == 2 and report.invalid == 1
    assert report.rejection_rate == 1 / 3

    try:
        validator.sanitize_transition(("agent", [1], 0, 99.0, [2], False))
        raise AssertionError("max_abs_reward check failed")
    except TransitionValidationError:
        pass

    printer.status("TEST", " Buffer Validation checks passed", "success")
    print("\n=== Test ran successfully ===\n")
