from __future__ import annotations

import numpy as np

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from .utils.config_loader import get_config_section, load_global_config
from .utils.buffer_errors import TransitionValidationError
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Buffer Validation")
printer = PrettyPrinter()

Transition = Tuple[Any, Any, Any, float, Any, bool]


class TransitionValidationError(ValueError):
    """Raised when a transition payload is invalid."""


@dataclass
class ValidationReport:
    """Structured report for bulk transition validation."""

    valid: int = 0
    invalid: int = 0
    errors: List[str] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return self.invalid == 0


@dataclass
class TransitionSchema:
    """Schema contract for transition validation/coercion."""

    required_length: int = 6
    enforce_numeric_reward: bool = True
    enforce_boolean_done: bool = True
    allow_none_state: bool = False
    allow_none_next_state: bool = False
    max_abs_reward: Optional[float] = None

    @classmethod
    def from_config(cls, user_config: Optional[Mapping[str, Any]] = None) -> "TransitionSchema":
        load_global_config()  # ensures global path metadata is available consistently
        config = dict(get_config_section("validation") or {})
        if user_config:
            config.update(dict(user_config))
        return cls(
            required_length=int(config.get("required_length", 6)),
            enforce_numeric_reward=bool(config.get("enforce_numeric_reward", True)),
            enforce_boolean_done=bool(config.get("enforce_boolean_done", True)),
            allow_none_state=bool(config.get("allow_none_state", False)),
            allow_none_next_state=bool(config.get("allow_none_next_state", False)),
            max_abs_reward=(
                float(config["max_abs_reward"]) if config.get("max_abs_reward") is not None else None
            ),
        )


class TransitionValidator:
    """Validates and optionally coerces replay transitions.

    Expected canonical tuple format:
    (agent_id, state, action, reward, next_state, done)
    """

    def __init__(self, schema: Optional[TransitionSchema] = None,
                 user_config: Optional[Mapping[str, Any]] = None):
        self.schema = schema or TransitionSchema.from_config(user_config=user_config)

    def _ensure_tuple(self, transition: Any) -> Tuple[Any, ...]:
        if isinstance(transition, tuple):
            payload = transition
        elif isinstance(transition, list):
            payload = tuple(transition)
        else:
            raise TransitionValidationError(
                f"Transition must be tuple/list, got {type(transition).__name__}."
            )

        if len(payload) != self.schema.required_length:
            raise TransitionValidationError(
                f"Transition length must be {self.schema.required_length}, got {len(payload)}."
            )
        return payload

    def _coerce_reward(self, reward: Any) -> float:
        if not self.schema.enforce_numeric_reward:
            return reward
        if not isinstance(reward, (int, float, np.number)):
            raise TransitionValidationError(
                f"Reward must be numeric, got {type(reward).__name__}."
            )
        reward_value = float(reward)
        if self.schema.max_abs_reward is not None and abs(reward_value) > self.schema.max_abs_reward:
            raise TransitionValidationError(
                f"Reward abs({reward_value}) exceeds max_abs_reward={self.schema.max_abs_reward}."
            )
        return reward_value

    def _coerce_done(self, done: Any) -> bool:
        if not self.schema.enforce_boolean_done:
            return done
        if isinstance(done, (bool, np.bool_)):
            return bool(done)
        raise TransitionValidationError(f"Done must be boolean, got {type(done).__name__}.")

    def _validate_state(self, state: Any, field_name: str, allow_none: bool) -> None:
        if state is None and not allow_none:
            raise TransitionValidationError(f"{field_name} cannot be None.")

    def validate_transition(self, transition: Any, coerce: bool = True) -> Transition:
        payload = self._ensure_tuple(transition)
        agent_id, state, action, reward, next_state, done = payload

        self._validate_state(state, "state", self.schema.allow_none_state)
        self._validate_state(next_state, "next_state", self.schema.allow_none_next_state)

        if coerce:
            reward = self._coerce_reward(reward)
            done = self._coerce_done(done)
        else:
            if self.schema.enforce_numeric_reward and not isinstance(reward, (int, float, np.number)):
                raise TransitionValidationError("Reward is not numeric.")
            if self.schema.enforce_boolean_done and not isinstance(done, (bool, np.bool_)):
                raise TransitionValidationError("Done is not boolean.")

        return agent_id, state, action, reward, next_state, done

    def validate_batch(self, transitions: Iterable[Any], coerce: bool = True) -> ValidationReport:
        report = ValidationReport()
        for idx, transition in enumerate(transitions):
            try:
                self.validate_transition(transition=transition, coerce=coerce)
                report.valid += 1
            except TransitionValidationError as exc:
                report.invalid += 1
                report.errors.append(f"index={idx}: {exc}")

        if report.invalid:
            logger.warning(
                "Validation report: valid=%s invalid=%s", report.valid, report.invalid
            )
        return report

    def sanitize_transition(self, transition: Any) -> Transition:
        """Alias for validate_transition(coerce=True) for readability."""
        return self.validate_transition(transition=transition, coerce=True)


__all__ = [
    "Transition",
    "TransitionSchema",
    "TransitionValidator",
    "ValidationReport",
    "TransitionValidationError",
]
