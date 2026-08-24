"""Resource limits and tuning-mutation policy for SLAI's QNN capability.

This module performs deterministic admission checks before state-vector
allocations, batches, and gradient probes. It does not schedule trials, rank
candidates, promote checkpoints, or duplicate SLAI's tuning subsystem.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .utils.config_loader import get_config_section, load_global_config
from .utils.quantum_errors import *
from .utils.quantum_helpers import *
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("QNN Quantum Policy")
printer = PrettyPrinter()

# These names describe implemented mutation capabilities. The YAML policy may
# narrow this set, but it cannot make an unsupported field tunable.
TUNABLE_PARAMETERS = frozenset(
    {
        "num_quantum_layers",
        "learning_rate",
        "entanglement",
        "loss",
        "gradient_method",
        "finite_difference_step",
        "gradient_clip_norm",
    }
)


def _configured_tunables(value: Any) -> frozenset[str]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise QNNConfigurationError(
            "quantum_policy.allowed_tuning_parameters must be a sequence"
        )
    names: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise QNNConfigurationError(
                "quantum_policy.allowed_tuning_parameters"
                f"[{index}] must be a non-empty string"
            )
        names.append(item.strip())
    if len(names) != len(set(names)):
        raise QNNConfigurationError(
            "quantum_policy.allowed_tuning_parameters contains duplicates"
        )
    unsupported = set(names) - TUNABLE_PARAMETERS
    if unsupported:
        raise QNNConfigurationError(
            "quantum_policy declares unsupported tuning parameter(s): "
            f"{', '.join(sorted(unsupported))}"
        )
    return frozenset(names)


@dataclass(frozen=True, slots=True)
class QuantumExecutionPolicy:
    """Apply configured limits before QNN allocations or gradient probes."""

    statevector_bytes: int
    parameter_bytes: int
    parameter_count: int
    max_working_set_bytes: int
    max_sequence_length: int
    max_tasks_per_request: int
    max_gradient_evaluations: int
    max_training_steps: int
    config: dict[str, Any] = field(init=False, repr=False, compare=False)
    policy_config: dict[str, Any] = field(init=False, repr=False, compare=False)
    allowed_tuning_parameters: frozenset[str] = field(init=False)

    def __post_init__(self) -> None:
        config = load_global_config()
        raw_section = config.get("quantum_policy")
        if not isinstance(raw_section, Mapping):
            raise QNNConfigurationError(
                "quantum_policy configuration must be a mapping"
            )
        policy_config = (
            get_config_section(
                "quantum_policy",
                config=config,
            )
            or {}
        )
        allowed = policy_config.get("allowed_tuning_parameters")
        if allowed is None:
            raise QNNConfigurationError(
                "quantum_policy.allowed_tuning_parameters is required"
            )

        object.__setattr__(self, "config", config)
        object.__setattr__(self, "policy_config", policy_config)
        object.__setattr__(self, "allowed_tuning_parameters", _configured_tunables(allowed))

        for name in (
            "statevector_bytes",
            "parameter_bytes",
            "parameter_count",
            "max_working_set_bytes",
            "max_sequence_length",
            "max_tasks_per_request",
            "max_gradient_evaluations",
            "max_training_steps",
        ):
            object.__setattr__(self, name, positive_int(getattr(self, name), name))

        minimum = 4 * self.statevector_bytes + 4 * self.parameter_bytes
        if minimum > self.max_working_set_bytes:
            raise QNNConfigurationError(
                "max_working_set_bytes is smaller than one conservative task "
                f"working set: required={minimum}, "
                f"limit={self.max_working_set_bytes}"
            )
        logger.debug(
            "Initialized QNN execution policy state_bytes=%d parameter_count=%d",
            self.statevector_bytes,
            self.parameter_count,
        )

    @classmethod
    def from_config(cls, config: Any) -> QuantumExecutionPolicy:
        """Construct policy from a validated :class:`QNNConfig`-like object."""

        required = (
            "statevector_bytes",
            "parameter_bytes",
            "parameter_count",
            "max_working_set_bytes",
            "max_sequence_length",
            "max_tasks_per_request",
            "max_gradient_evaluations",
            "max_training_steps",
        )
        missing = [name for name in required if not hasattr(config, name)]
        if missing:
            raise QNNConfigurationError(
                f"QNN policy source is missing field(s): {', '.join(sorted(missing))}"
            )
        return cls(**{name: getattr(config, name) for name in required})

    def validate_tuning_parameters(self, parameters: Mapping[str, Any]) -> None:
        """Reject tuning mutations outside the configured QNN allowlist."""

        if not isinstance(parameters, Mapping):
            raise TypeError("tuning parameters must be a mapping")
        invalid_keys = [key for key in parameters if not isinstance(key, str)]
        if invalid_keys:
            raise QNNConfigurationError("QNN tuning parameter names must be strings")
        unknown = set(parameters) - self.allowed_tuning_parameters
        if unknown:
            raise QNNConfigurationError(
                "unsupported or disallowed QNN tuning parameter(s): "
                f"{', '.join(sorted(unknown))}"
            )

    def validate_sequence_length(self, length: int, *, name: str) -> None:
        """Enforce the state count accepted by one sequence."""

        count = positive_int(length, f"{name}_length")
        if count > self.max_sequence_length:
            raise QNNResourceLimitError(
                f"{name} contains {count} states; configured maximum is "
                f"{self.max_sequence_length}"
            )

    def validate_tasks(self, tasks: Sequence[Any]) -> None:
        """Enforce the number of QNN tasks admitted in one request."""

        if isinstance(tasks, (str, bytes, bytearray)) or not isinstance(
            tasks,
            Sequence,
        ):
            raise TypeError("tasks must be a sequence")
        count = positive_int(len(tasks), "task_count")
        if count > self.max_tasks_per_request:
            raise QNNResourceLimitError(
                f"request contains {count} tasks; configured maximum is "
                f"{self.max_tasks_per_request}"
            )

    def validate_working_set(self, state_slots: int, *, operation: str) -> int:
        """Return the conservative byte estimate or reject the operation."""

        slots = positive_int(state_slots, "state_slots")
        if not isinstance(operation, str) or not operation.strip():
            raise QNNConfigurationError("operation must be a non-empty string")
        required = slots * self.statevector_bytes + 4 * self.parameter_bytes
        if required > self.max_working_set_bytes:
            raise QNNResourceLimitError(
                f"{operation} exceeds max_working_set_bytes: "
                f"required={required}, limit={self.max_working_set_bytes}"
            )
        return required

    def validate_training_work(self, *, sequence_length: int, steps: int) -> int:
        """Bound two-probe-per-parameter numerical gradient work."""

        length = positive_int(sequence_length, "sequence_length")
        step_count = positive_int(steps, "steps")
        self.validate_sequence_length(length, name="training sequence")
        if step_count > self.max_training_steps:
            raise QNNResourceLimitError(
                f"steps exceeds configured maximum {self.max_training_steps}"
            )
        evaluations = 2 * self.parameter_count * length * step_count
        if evaluations > self.max_gradient_evaluations:
            raise QNNResourceLimitError(
                "training exceeds max_gradient_evaluations: "
                f"required={evaluations}, limit={self.max_gradient_evaluations}"
            )
        return evaluations


__all__ = ["TUNABLE_PARAMETERS", "QuantumExecutionPolicy"]
